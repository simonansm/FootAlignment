"""Evaluation script for foot arch classification."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from dataloaders.dataloader_arch import ImageMeasurementDataset
from models.inception_v3_attention import InceptionWithAttentionFusion
from utils.ml_class_experiment_utils import (
    MORPHOLOGICAL_FEATURES,
    bootstrap_metrics_from_predictions,
    print_bootstrap_summary,
    print_dataset_summary,
    print_split_class_distribution,
)


def get_project_root():
    """Get the root directory (parent of this script)."""
    return Path(__file__).parent.parent


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    model = InceptionWithAttentionFusion(num_features=19, num_classes=3)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Handle DataParallel models
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    return model.to(device)


def collect_predictions(model, loader, device):
    """Collect predictions on a dataset."""
    model.eval()
    all_y_true, all_y_pred, all_y_prob = [], [], []
    
    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(images, features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)
    
    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)


def main():
    """Main evaluation function."""
    root = get_project_root()
    
    # Paths
    csv_file = root / "data" / "foot_3class.csv"
    image_dir = root / "archive" / "Plantar_dataset"
    checkpoint_path = root / "checkpoints" / "best_model_inception_arch.pt"
    
    # Verify files exist
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    full_df = pd.read_csv(csv_file)
    labels = full_df["GT_label_arch"].values
    idx_all = np.arange(len(full_df))
    
    train_idx, temp_idx, _, temp_labels = train_test_split(
        idx_all, labels, test_size=0.35, random_state=20, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=20, stratify=temp_labels
    )
    
    print_dataset_summary(full_df, MORPHOLOGICAL_FEATURES, "GT_label_arch", str(csv_file))
    print_split_class_distribution("Train", full_df.loc[train_idx, "GT_label_arch"].values)
    print_split_class_distribution("Validation", full_df.loc[val_idx, "GT_label_arch"].values)
    print_split_class_distribution("Test", full_df.loc[test_idx, "GT_label_arch"].values)
    
    # Load model
    model = load_model(checkpoint_path, device)
    print(f"\n✓ Loaded model from {checkpoint_path}")
    
    # Test evaluation
    dataset = ImageMeasurementDataset(str(csv_file), str(image_dir), transform=eval_transform)
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nCollecting predictions on test set ({len(test_idx)} samples)...")
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)
    
    print("\nRunning bootstrap evaluation (100 resamples, 95% CI)...")
    bootstrap_result = bootstrap_metrics_from_predictions(
        y_true, y_pred, y_prob,
        n_resamples=100,
        random_state=0,
        ci_percent=95
    )
    
    print_bootstrap_summary("arch", bootstrap_result)


if __name__ == "__main__":
    main()
