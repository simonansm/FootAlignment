import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ml_class_experiment_utils import (
    MORPHOLOGICAL_FEATURES,
    bootstrap_metrics_from_predictions,
    print_bootstrap_summary,
    print_dataset_summary,
    print_split_class_distribution,
    save_bootstrap_evaluation_csvs,
)


def load_model_flexibly(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("Direct load failed. Trying after adjusting keys...")
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    return model


def collect_predictions_fusion(model, loader, device):
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
    return (
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob),
    )


def collect_predictions_image_only(model, loader, device):
    model.eval()
    all_y_true, all_y_pred, all_y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)
    return (
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob),
    )


def _run_bootstrap_evaluation(
    region_name,
    csv_file,
    image_dir,
    label_col,
    checkpoint_path,
    dataset_cls,
    model_factory,
    collect_fn,
    output_dir,
    batch_size,
    n_bootstrap,
    bootstrap_seed,
    ci_percent,
    device,
    input_size=224,
    image_only=False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = os.path.join("evaluation_outputs", region_name)

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_df = pd.read_csv(csv_file)
    labels = full_df[label_col].values
    idx_all = np.arange(len(full_df))

    train_idx, temp_idx, _, temp_labels = train_test_split(
        idx_all, labels, test_size=0.35, random_state=20, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=20, stratify=temp_labels
    )

    print_dataset_summary(full_df, MORPHOLOGICAL_FEATURES, label_col, csv_file)
    print_split_class_distribution("Train", full_df.loc[train_idx, label_col].values)
    print_split_class_distribution("Validation", full_df.loc[val_idx, label_col].values)
    print_split_class_distribution("Test", full_df.loc[test_idx, label_col].values)

    dataset = dataset_cls(csv_file, image_dir, transform=eval_transform)
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nTest set size (full held-out): {len(test_idx)}")
    if torch.cuda.is_available() and "cuda" in str(device):
        gpu_id = device.split(":")[-1] if ":" in str(device) else "0"
        try:
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(int(gpu_id))}")
        except Exception:
            pass

    if image_only:
        model = model_factory(3).to(device)
    else:
        model = model_factory(19, 3).to(device)
    model = load_model_flexibly(model, checkpoint_path, device)

    print("\nRunning inference on the full held-out test set...")
    y_true, y_pred, y_prob = collect_fn(model, test_loader, device)

    print(
        f"\nBootstrap resampling on test predictions "
        f"(B={n_bootstrap}, with replacement, seed={bootstrap_seed})..."
    )
    bootstrap_result = bootstrap_metrics_from_predictions(
        y_true,
        y_pred,
        y_prob,
        n_resamples=n_bootstrap,
        random_state=bootstrap_seed,
        ci_percent=ci_percent,
    )

    print_bootstrap_summary(region_name, bootstrap_result)
    save_bootstrap_evaluation_csvs(
        output_dir=output_dir,
        prefix=region_name,
        full_df=full_df,
        input_features=MORPHOLOGICAL_FEATURES,
        label_col=label_col,
        csv_path=csv_file,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        bootstrap_result=bootstrap_result,
    )

    return bootstrap_result


def run_dl_evaluation(
    region_name,
    csv_file,
    image_dir,
    label_col,
    checkpoint_path,
    dataset_cls,
    model_factory,
    output_dir=None,
    n_bootstrap=100,
    bootstrap_seed=0,
    ci_percent=95,
    batch_size=8,
    device=None,
):
    return _run_bootstrap_evaluation(
        region_name=region_name,
        csv_file=csv_file,
        image_dir=image_dir,
        label_col=label_col,
        checkpoint_path=checkpoint_path,
        dataset_cls=dataset_cls,
        model_factory=model_factory,
        collect_fn=collect_predictions_fusion,
        output_dir=output_dir,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        ci_percent=ci_percent,
        device=device,
        input_size=224,
        image_only=False,
    )


def run_dl_evaluation_io(
    region_name,
    csv_file,
    image_dir,
    label_col,
    checkpoint_path,
    dataset_cls,
    model_factory,
    output_dir=None,
    n_bootstrap=100,
    bootstrap_seed=0,
    ci_percent=95,
    batch_size=8,
    device=None,
    input_size=299,
):
    """Image-only model evaluation with test-set bootstrap."""
    return _run_bootstrap_evaluation(
        region_name=region_name,
        csv_file=csv_file,
        image_dir=image_dir,
        label_col=label_col,
        checkpoint_path=checkpoint_path,
        dataset_cls=dataset_cls,
        model_factory=model_factory,
        collect_fn=collect_predictions_image_only,
        output_dir=output_dir,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        ci_percent=ci_percent,
        device=device,
        input_size=input_size,
        image_only=True,
    )
