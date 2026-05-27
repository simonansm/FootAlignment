"""Training script for foot arch classification."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MORPHOLOGICAL_FEATURES, EARLY_STOPPING_PATIENCE, MAX_SAVED_MODELS
)
from dataloaders.dataloader_arch import ImageMeasurementDataset
from models.inception_v3_attention import InceptionWithAttentionFusion
from utils.ml_class_experiment_utils import (
    evaluate_predictions_detailed,
    print_dataset_summary,
    print_detailed_metrics,
    print_split_class_distribution,
)


def get_project_root():
    """Get the root directory (parent of this script)."""
    return Path(__file__).parent


def setup_paths():
    """Set up and create necessary directories."""
    root = get_project_root()
    data_dir = root / "data"
    checkpoint_dir = root / "checkpoints"
    last5_dir = checkpoint_dir / "last5"
    
    checkpoint_dir.mkdir(exist_ok=True)
    last5_dir.mkdir(exist_ok=True)
    
    return data_dir, checkpoint_dir, last5_dir


def load_data(csv_file, image_dir, label_col, transform):
    """Load and split data."""
    full_df = pd.read_csv(csv_file)
    labels = full_df[label_col].values
    idx_all = np.arange(len(full_df))
    
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        idx_all, labels, test_size=0.35, random_state=20, stratify=labels
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=20, stratify=temp_labels
    )
    
    print_dataset_summary(full_df, MORPHOLOGICAL_FEATURES, label_col, str(csv_file))
    print_split_class_distribution("Train", full_df.loc[train_idx, label_col].values)
    print_split_class_distribution("Validation", full_df.loc[val_idx, label_col].values)
    print_split_class_distribution("Test", full_df.loc[test_idx, label_col].values)
    
    dataset = ImageMeasurementDataset(str(csv_file), str(image_dir), transform=transform)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, full_df, train_idx


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    detailed = evaluate_predictions_detailed(y_true, y_pred, y_prob)
    return {
        'acc': detailed['accuracy'],
        'f1': detailed['macro_f1'],
        'precision': detailed['macro_precision'],
        'recall': detailed['macro_recall'],
        'auc': detailed['macro_auc'],
        'detailed': detailed,
    }


def save_checkpoint(model, epoch, f1_score, checkpoint_dir, max_saved=5):
    """Save model checkpoint."""
    save_name = f'model_epoch{epoch}_f1{f1_score:.4f}.pt'
    save_path = checkpoint_dir / save_name
    torch.save(model.state_dict(), str(save_path))
    
    files = sorted(
        [f for f in checkpoint_dir.iterdir() if f.suffix == '.pt'],
        key=lambda x: x.stat().st_mtime
    )
    
    if len(files) > max_saved:
        for f in files[:-max_saved]:
            f.unlink()
    
    return save_path


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_losses = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    
    for images, features, labels in tqdm(train_loader, desc="Training"):
        images, features, labels = images.to(device), features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        preds = outputs.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_y_true.extend(labels.cpu().numpy())
        all_y_pred.extend(preds)
        all_y_prob.extend(probs)
    
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)), np.mean(train_losses)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_losses = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    
    with torch.no_grad():
        for images, features, labels in tqdm(val_loader, desc="Validation"):
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)
    
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)), np.mean(val_losses)


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_y_true, all_y_pred, all_y_prob = [], [], []
    
    with torch.no_grad():
        for images, features, labels in tqdm(test_loader, desc="Evaluation"):
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)
    
    return compute_metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob))


def main():
    """Main training loop."""
    root = get_project_root()
    data_dir, checkpoint_dir, last5_dir = setup_paths()
    
    # Paths
    csv_file = root / "data" / "foot_3class.csv"
    image_dir = root / "archive" / "Plantar_dataset"
    best_model_path = checkpoint_dir / "best_model_inception_arch.pt"
    
    # Verify data exists
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    train_loader, val_loader, test_loader, full_df, train_idx = load_data(
        csv_file, image_dir, "GT_label_arch", transform
    )
    
    # Class weights
    train_labels = full_df.loc[train_idx, "GT_label_arch"].values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")
    
    # Model
    model = InceptionWithAttentionFusion(num_features=19, num_classes=NUM_CLASSES)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    best_f1 = 0
    epochs_since_improvement = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_loss = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print_detailed_metrics(train_metrics['detailed'], split_name="train", epoch=epoch)
        print_detailed_metrics(val_metrics['detailed'], split_name="validation", epoch=epoch)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, str(best_model_path))
            print(f"✓ Saved best model: {best_model_path}")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        save_checkpoint(model, epoch, val_metrics['f1'], checkpoint_dir, MAX_SAVED_MODELS)
    
    print("\n" + "="*60)
    print("Training finished. Evaluating on test set...")
    
    state = torch.load(str(best_model_path), map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    
    test_metrics = evaluate(model, test_loader, device)
    print_detailed_metrics(test_metrics['detailed'], split_name="test")


if __name__ == "__main__":
    main()
