import os

import numpy as np
import pandas as pd

MORPHOLOGICAL_FEATURES = [
    "gender", "side", "len_foot", "len_arch", "len_mm_med", "len_mm_lat",
    "len_latdors", "len_arch_lat", "len_arch_med", "wid_fore", "wid_heel",
    "wid_instep", "wid_meta", "size_eu", "arch_med", "arch_reg", "arch_lat",
    "arch_index", "pron_angle",
]

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def print_dataset_summary(df, input_features, output_feature, csv_path):
    """Print raw dataset information and class distribution."""
    print("\n===== Raw Dataset Summary =====")
    print(f"CSV path: {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"Target column: {output_feature}")
    print(f"Number of features: {len(input_features)}")
    print(f"Feature columns: {input_features}")

    missing = df[input_features + [output_feature]].isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nMissing values:")
        print(missing.to_string())
    else:
        print("\nMissing values: none")

    print("\nClass distribution (full dataset):")
    class_counts = df[output_feature].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = 100.0 * count / len(df)
        print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

    print("\nFeature summary (first 5 rows):")
    print(df[input_features].head().to_string())

    print("\nFeature descriptive statistics:")
    print(df[input_features].describe().T.to_string())


def print_split_class_distribution(split_name, labels):
    labels = np.asarray(labels)
    print(f"\n{split_name} split: {len(labels)} samples")
    classes, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(classes, counts):
        pct = 100.0 * count / len(labels)
        print(f"  Class {cls}: {count} samples ({pct:.1f}%)")


def _get_predict_proba(model, X_test):
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X_test)
    except Exception:
        return None


def compute_per_class_specificity(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificity = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificity[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return specificity


def compute_per_class_accuracy_ovr(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    acc = {}
    for cls in classes:
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        acc[cls] = accuracy_score(y_true_bin, y_pred_bin)
    return acc


def evaluate_predictions_detailed(y_true, y_pred, y_prob, classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        classes = np.sort(classes)

    support = {cls: int((y_true == cls).sum()) for cls in classes}
    precision = dict(zip(classes, precision_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )))
    recall = dict(zip(classes, recall_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )))
    f1 = dict(zip(classes, f1_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )))
    specificity = compute_per_class_specificity(y_true, y_pred, classes)
    per_class_acc = compute_per_class_accuracy_ovr(y_true, y_pred, classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    auc = {}
    ap = {}
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
            auc[cls] = float("nan")
            ap[cls] = float("nan")
            continue
        if y_prob.ndim == 2 and y_prob.shape[1] > i:
            auc[cls] = roc_auc_score(y_true_bin, y_prob[:, i])
            ap[cls] = average_precision_score(y_true_bin, y_prob[:, i])
        else:
            auc[cls] = float("nan")
            ap[cls] = float("nan")

    macro_auc = np.nanmean(list(auc.values()))
    macro_ap = np.nanmean(list(ap.values()))

    return {
        "classes": classes,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auc": auc,
        "average_precision": ap,
        "per_class_accuracy_ovr": per_class_acc,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0
        ),
    }


def evaluate_model_detailed(model, X_test, y_test, classes=None):
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)
    proba = _get_predict_proba(model, X_test)
    if proba is None:
        proba = np.zeros((len(y_test), len(np.unique(y_test))))
    return evaluate_predictions_detailed(y_test, y_pred, proba, classes=classes)


def print_detailed_metrics(metrics, split_name="test", epoch=None, model_name=None, seed=None):
    if epoch is not None:
        print(f"\n----- Epoch {epoch} | {split_name} set -----")
    elif model_name is not None and seed is not None:
        print(f"\n----- {model_name} | seed={seed} | {split_name} set -----")
    else:
        print(f"\n----- {split_name} set -----")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision:  {metrics['macro_precision']:.4f}")
    print(f"Macro recall:     {metrics['macro_recall']:.4f}")
    print(f"Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"Macro AUC (OvR):  {metrics['macro_auc']:.4f}")
    print(f"Macro AP (OvR):   {metrics['macro_ap']:.4f}")

    print("\nPer-class metrics:")
    header = (
        f"{'Class':<8}{'Support':<10}{'Precision':<12}{'Recall':<12}"
        f"{'F1':<12}{'Specificity':<14}{'AUC':<12}{'AP':<12}{'Acc(OvR)':<12}"
    )
    print(header)
    print("-" * len(header))
    for cls in metrics["classes"]:
        print(
            f"{cls:<8}"
            f"{metrics['support'][cls]:<10}"
            f"{metrics['precision'][cls]:<12.4f}"
            f"{metrics['recall'][cls]:<12.4f}"
            f"{metrics['f1'][cls]:<12.4f}"
            f"{metrics['specificity'][cls]:<14.4f}"
            f"{metrics['auc'][cls]:<12.4f}"
            f"{metrics['average_precision'][cls]:<12.4f}"
            f"{metrics['per_class_accuracy_ovr'][cls]:<12.4f}"
        )

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"Classes: {list(metrics['classes'])}")
    print(metrics["confusion_matrix"])

    print("\nClassification report:")
    print(metrics["classification_report"])


def _aggregate_class_dicts(run_metrics, key):
    classes = run_metrics[0]["classes"]
    values = {cls: [] for cls in classes}
    for m in run_metrics:
        for cls in classes:
            val = m[key].get(cls, np.nan)
            if not np.isnan(val):
                values[cls].append(val)
    return {cls: (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)
            for cls, vals in values.items()}


def print_aggregated_summary(model_name, run_metrics_list):
    arr = np.array([
        [m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"],
         m["macro_auc"], m["macro_ap"]]
        for m in run_metrics_list
    ])
    macro_names = [
        "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
        "AUC (macro, OvR)", "AP (macro, OvR)",
    ]

    print(f"\n===== {model_name} | Aggregated over {len(run_metrics_list)} seeds =====")
    for i, name in enumerate(macro_names):
        print(f"{name}: {arr[:, i].mean():.4f} ± {arr[:, i].std():.4f}")

    for metric_key, label in [
        ("support", "Support"),
        ("precision", "Precision"),
        ("recall", "Recall / Sensitivity"),
        ("f1", "F1-score"),
        ("specificity", "Specificity"),
        ("auc", "AUC (OvR)"),
        ("average_precision", "Average Precision (PR-AUC)"),
        ("per_class_accuracy_ovr", "Accuracy (OvR)"),
    ]:
        agg = _aggregate_class_dicts(run_metrics_list, metric_key)
        print(f"\nPer-class {label} (mean ± std):")
        for cls in run_metrics_list[0]["classes"]:
            mean, std = agg[cls]
            if metric_key == "support":
                print(f"  Class {cls}: {mean:.1f} ± {std:.1f}")
            else:
                print(f"  Class {cls}: {mean:.4f} ± {std:.4f}")

    avg_cm = np.mean([m["confusion_matrix"] for m in run_metrics_list], axis=0)
    print("\nMean confusion matrix across seeds (rows=true, cols=predicted):")
    print(f"Classes: {list(run_metrics_list[0]['classes'])}")
    print(np.round(avg_cm, 2))


MACRO_METRIC_KEYS = [
    ("accuracy", "accuracy"),
    ("macro_precision", "macro_precision"),
    ("macro_recall", "macro_recall"),
    ("macro_f1", "macro_f1"),
    ("macro_auc", "macro_auc"),
    ("macro_ap", "macro_ap"),
]

PER_CLASS_METRIC_KEYS = [
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "f1"),
    ("specificity", "specificity"),
    ("auc", "auc"),
    ("average_precision", "average_precision"),
    ("per_class_accuracy_ovr", "accuracy_ovr"),
]


def bootstrap_metrics_from_predictions(
    y_true, y_pred, y_prob, n_resamples=100, random_state=0, ci_percent=95, classes=None
):
    """Non-parametric bootstrap on held-out test predictions (resample with replacement)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    if classes is None:
        classes = np.sort(np.unique(y_true))

    point_estimate = evaluate_predictions_detailed(
        y_true, y_pred, y_prob, classes=classes
    )

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    bootstrap_metrics = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_metrics.append(
            evaluate_predictions_detailed(
                y_true[idx], y_pred[idx], y_prob[idx], classes=classes
            )
        )

    summary = _summarize_bootstrap_metrics(bootstrap_metrics, ci_percent=ci_percent)
    return {
        "point_estimate": point_estimate,
        "bootstrap_metrics": bootstrap_metrics,
        "summary": summary,
        "n_resamples": n_resamples,
        "ci_percent": ci_percent,
    }


def _summarize_bootstrap_metrics(bootstrap_metrics, ci_percent=95):
    alpha = (100 - ci_percent) / 2.0
    hi = 100 - alpha

    macro_summary = {}
    for key, label in MACRO_METRIC_KEYS:
        vals = [m[key] for m in bootstrap_metrics if not np.isnan(m[key])]
        if vals:
            macro_summary[label] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lower": float(np.percentile(vals, alpha)),
                "ci_upper": float(np.percentile(vals, hi)),
            }
        else:
            macro_summary[label] = {
                "mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan
            }

    classes = list(bootstrap_metrics[0]["classes"])
    per_class_summary = {}
    for cls in classes:
        per_class_summary[cls] = {}
        for key, label in PER_CLASS_METRIC_KEYS:
            vals = [
                m[key][cls] for m in bootstrap_metrics
                if cls in m[key] and not np.isnan(m[key][cls])
            ]
            if vals:
                per_class_summary[cls][label] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "ci_lower": float(np.percentile(vals, alpha)),
                    "ci_upper": float(np.percentile(vals, hi)),
                }
            else:
                per_class_summary[cls][label] = {
                    "mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan
                }

    return {"macro": macro_summary, "per_class": per_class_summary, "classes": classes}


def print_bootstrap_summary(region_name, bootstrap_result):
    point = bootstrap_result["point_estimate"]
    summary = bootstrap_result["summary"]
    n = bootstrap_result["n_resamples"]
    ci = bootstrap_result["ci_percent"]

    print(f"\n===== {region_name} | Point estimate (full test set) =====")
    print_detailed_metrics(point, split_name="test (full)")

    print(
        f"\n===== {region_name} | Bootstrap on test predictions "
        f"(B={n}, {ci}% percentile CI) ====="
    )
    print(f"{'Metric':<22}{'Mean':<10}{'Std':<10}{'CI lower':<12}{'CI upper':<12}")
    print("-" * 66)
    for label, stats in summary["macro"].items():
        print(
            f"{label:<22}{stats['mean']:<10.4f}{stats['std']:<10.4f}"
            f"{stats['ci_lower']:<12.4f}{stats['ci_upper']:<12.4f}"
        )

    print(f"\nPer-class metrics (mean ± std, {ci}% CI):")
    for cls in summary["classes"]:
        print(f"\n  Class {cls}:")
        print(f"  {'Metric':<20}{'Mean±Std':<18}{'95% CI':<22}")
        for label, stats in summary["per_class"][cls].items():
            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            print(f"  {label:<20}{mean_std:<18}{ci_str:<22}")


def bootstrap_macro_summary_df(summary):
    rows = []
    for metric, stats in summary["macro"].items():
        rows.append({
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
        })
    return pd.DataFrame(rows)


def bootstrap_per_class_summary_df(summary):
    rows = []
    for cls in summary["classes"]:
        for metric, stats in summary["per_class"][cls].items():
            rows.append({
                "class": cls,
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
            })
    return pd.DataFrame(rows)


def save_bootstrap_evaluation_csvs(
    output_dir,
    prefix,
    full_df,
    input_features,
    label_col,
    csv_path,
    train_idx,
    val_idx,
    test_idx,
    bootstrap_result,
):
    os.makedirs(output_dir, exist_ok=True)
    point = bootstrap_result["point_estimate"]
    summary = bootstrap_result["summary"]

    paths = {
        f"{prefix}_raw_data_summary.csv": build_raw_data_summary_df(
            full_df, input_features, label_col, csv_path, train_idx, val_idx, test_idx
        ),
        f"{prefix}_feature_statistics.csv": build_feature_stats_df(full_df, input_features),
        f"{prefix}_point_estimate_per_class.csv": detailed_metrics_to_per_class_df(point, "full_test"),
        f"{prefix}_point_estimate_macro.csv": detailed_metrics_to_macro_df(point, "full_test"),
        f"{prefix}_bootstrap_macro_summary.csv": bootstrap_macro_summary_df(summary),
        f"{prefix}_bootstrap_per_class_summary.csv": bootstrap_per_class_summary_df(summary),
    }

    classes = list(point["classes"])
    cm_df = pd.DataFrame(
        point["confusion_matrix"],
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    paths[f"{prefix}_point_estimate_confusion_matrix.csv"] = cm_df.reset_index().rename(
        columns={"index": "true_class"}
    )

    per_run_macro = [
        detailed_metrics_to_macro_df(m, b)
        for b, m in enumerate(bootstrap_result["bootstrap_metrics"])
    ]
    per_run_per_class = [
        detailed_metrics_to_per_class_df(m, b)
        for b, m in enumerate(bootstrap_result["bootstrap_metrics"])
    ]
    paths[f"{prefix}_bootstrap_macro_by_resample.csv"] = pd.concat(
        per_run_macro, ignore_index=True
    )
    paths[f"{prefix}_bootstrap_per_class_by_resample.csv"] = pd.concat(
        per_run_per_class, ignore_index=True
    )

    saved = []
    for filename, df in paths.items():
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        saved.append(path)

    print(f"\nSaved {len(saved)} CSV files to {output_dir}:")
    for p in saved:
        print(f"  - {p}")
    return saved


def build_raw_data_summary_df(df, input_features, label_col, csv_path, train_idx, val_idx, test_idx):
    rows = [
        {"section": "metadata", "split": "all", "class": "", "key": "csv_path", "value": csv_path},
        {"section": "metadata", "split": "all", "class": "", "key": "total_samples", "value": len(df)},
        {"section": "metadata", "split": "all", "class": "", "key": "label_column", "value": label_col},
        {"section": "metadata", "split": "all", "class": "", "key": "num_features", "value": len(input_features)},
        {"section": "metadata", "split": "all", "class": "", "key": "features", "value": ", ".join(input_features)},
    ]
    split_map = {
        "full": df.index,
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }
    for split_name, idx in split_map.items():
        counts = df.loc[idx, label_col].value_counts().sort_index()
        n = len(idx)
        for cls, count in counts.items():
            rows.append({
                "section": "class_distribution",
                "split": split_name,
                "class": cls,
                "key": "count",
                "value": int(count),
                "percent": round(100.0 * count / n, 2) if n else 0.0,
            })
    return pd.DataFrame(rows)


def build_feature_stats_df(df, input_features):
    stats = df[input_features].describe().T.reset_index().rename(columns={"index": "feature"})
    stats.insert(0, "section", "feature_statistics")
    return stats


def detailed_metrics_to_per_class_df(detailed, run_seed):
    rows = []
    for cls in detailed["classes"]:
        rows.append({
            "run_seed": run_seed,
            "class": cls,
            "support": detailed["support"][cls],
            "precision": detailed["precision"][cls],
            "recall": detailed["recall"][cls],
            "f1": detailed["f1"][cls],
            "specificity": detailed["specificity"][cls],
            "auc": detailed["auc"][cls],
            "average_precision": detailed["average_precision"][cls],
            "accuracy_ovr": detailed["per_class_accuracy_ovr"][cls],
        })
    return pd.DataFrame(rows)


def detailed_metrics_to_macro_df(detailed, run_seed):
    return pd.DataFrame([{
        "run_seed": run_seed,
        "accuracy": detailed["accuracy"],
        "macro_precision": detailed["macro_precision"],
        "macro_recall": detailed["macro_recall"],
        "macro_f1": detailed["macro_f1"],
        "macro_auc": detailed["macro_auc"],
        "macro_ap": detailed["macro_ap"],
    }])


def confusion_matrix_to_df(detailed, run_seed):
    classes = list(detailed["classes"])
    cm = detailed["confusion_matrix"]
    df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    df.insert(0, "run_seed", run_seed)
    return df.reset_index().rename(columns={"index": "true_class"})


def aggregate_per_class_metrics_df(run_metrics_list):
    rows = []
    metric_keys = [
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("specificity", "specificity"),
        ("auc", "auc"),
        ("average_precision", "average_precision"),
        ("per_class_accuracy_ovr", "accuracy_ovr"),
    ]
    for cls in run_metrics_list[0]["classes"]:
        row = {"class": cls}
        support_vals = [m["support"][cls] for m in run_metrics_list]
        row["support_mean"] = float(np.mean(support_vals))
        row["support_std"] = float(np.std(support_vals))
        for key, col in metric_keys:
            vals = [m[key][cls] for m in run_metrics_list if not np.isnan(m[key][cls])]
            row[f"{col}_mean"] = float(np.mean(vals)) if vals else np.nan
            row[f"{col}_std"] = float(np.std(vals)) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_macro_metrics_df(run_metrics_list):
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc", "macro_ap"]
    row = {}
    for key in keys:
        vals = [m[key] for m in run_metrics_list if not np.isnan(m[key])]
        row[f"{key}_mean"] = float(np.mean(vals)) if vals else np.nan
        row[f"{key}_std"] = float(np.std(vals)) if vals else np.nan
    return pd.DataFrame([row])


def save_evaluation_csvs(
    output_dir,
    prefix,
    full_df,
    input_features,
    label_col,
    csv_path,
    train_idx,
    val_idx,
    test_idx,
    run_metrics_list,
):
    os.makedirs(output_dir, exist_ok=True)

    raw_df = build_raw_data_summary_df(
        full_df, input_features, label_col, csv_path, train_idx, val_idx, test_idx
    )
    feat_df = build_feature_stats_df(full_df, input_features)

    per_class_frames = [
        detailed_metrics_to_per_class_df(m, i) for i, m in enumerate(run_metrics_list)
    ]
    macro_frames = [
        detailed_metrics_to_macro_df(m, i) for i, m in enumerate(run_metrics_list)
    ]
    cm_frames = [
        confusion_matrix_to_df(m, i) for i, m in enumerate(run_metrics_list)
    ]

    paths = {
        f"{prefix}_raw_data_summary.csv": raw_df,
        f"{prefix}_feature_statistics.csv": feat_df,
        f"{prefix}_per_class_metrics_by_run.csv": pd.concat(per_class_frames, ignore_index=True),
        f"{prefix}_macro_metrics_by_run.csv": pd.concat(macro_frames, ignore_index=True),
        f"{prefix}_per_class_metrics_aggregated.csv": aggregate_per_class_metrics_df(run_metrics_list),
        f"{prefix}_macro_metrics_aggregated.csv": aggregate_macro_metrics_df(run_metrics_list),
        f"{prefix}_confusion_matrix_by_run.csv": pd.concat(cm_frames, ignore_index=True),
    }

    avg_cm = np.mean([m["confusion_matrix"] for m in run_metrics_list], axis=0)
    classes = list(run_metrics_list[0]["classes"])
    mean_cm_df = pd.DataFrame(
        avg_cm,
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    paths[f"{prefix}_confusion_matrix_mean.csv"] = mean_cm_df.reset_index().rename(
        columns={"index": "true_class"}
    )

    saved = []
    for filename, df in paths.items():
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        saved.append(path)

    print(f"\nSaved {len(saved)} CSV files to {output_dir}:")
    for p in saved:
        print(f"  - {p}")
    return saved


def apply_smote(X, y, random_state=0):
    """Apply SMOTE with k_neighbors adapted to the smallest class size."""
    from imblearn.over_sampling import SMOTE

    y_arr = np.asarray(y)
    _, counts = np.unique(y_arr, return_counts=True)
    min_class_count = int(counts.min())
    if min_class_count < 2:
        print("Warning: SMOTE skipped (minority class has < 2 samples).")
        return X, y

    k_neighbors = min(5, max(1, min_class_count - 1))
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    return smote.fit_resample(X, y)


def run_ml_class_experiment(
    csv_path,
    input_features,
    output_feature,
    seeds,
    grid_search_random_state=0,
    test_size=0.25,
    smote_random_state=0,
):
    from colorama import Fore
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    print(Fore.CYAN + "Loading dataset...")
    df = pd.read_csv(csv_path)
    X = df[input_features]
    y = df[output_feature]

    print_dataset_summary(df, input_features, output_feature, csv_path)

    model_names = ["Random Forest", "XGBoost", "Logistic Regression", "MLPClassifier"]
    detailed_results = {name: [] for name in model_names}

    print(Fore.YELLOW + "\nPerforming GridSearchCV for all models (only once)...")
    X_train_base, _, y_train_base, _ = train_test_split(
        X, y, test_size=0.30, random_state=grid_search_random_state, stratify=y
    )
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_train_base_res, y_train_base_res = apply_smote(
        X_train_base_scaled, y_train_base, random_state=smote_random_state
    )

    best_params = {}

    rf = RandomForestClassifier(class_weight="balanced", random_state=0)
    gs_rf = GridSearchCV(
        rf,
        {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [1, 2, 3],
            "min_samples_leaf": [1, 2],
        },
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )
    gs_rf.fit(X_train_base_res, y_train_base_res)
    best_params["Random Forest"] = gs_rf.best_params_
    print("Best params for Random Forest:", gs_rf.best_params_)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(y.unique()),
        eval_metric="mlogloss",
        random_state=0,
    )
    gs_xgb = GridSearchCV(
        xgb,
        {
            "n_estimators": [200, 300, 400],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.5, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.8, 1.0],
        },
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )
    gs_xgb.fit(X_train_base_res, y_train_base_res)
    best_params["XGBoost"] = gs_xgb.best_params_
    print("Best params for XGBoost:", gs_xgb.best_params_)

    lr = LogisticRegression(
        solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=0
    )
    gs_lr = GridSearchCV(
        lr,
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]},
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )
    gs_lr.fit(X_train_base_res, y_train_base_res)
    best_params["Logistic Regression"] = gs_lr.best_params_
    print("Best params for Logistic Regression:", gs_lr.best_params_)

    mlp = MLPClassifier(max_iter=2000, random_state=0)
    gs_mlp = GridSearchCV(
        mlp,
        {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "learning_rate": ["constant", "adaptive"],
        },
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )
    gs_mlp.fit(X_train_base_res, y_train_base_res)
    best_params["MLPClassifier"] = gs_mlp.best_params_
    print("Best params for MLPClassifier:", gs_mlp.best_params_)

    num_classes = len(y.unique())

    for seed in seeds:
        print(Fore.CYAN + f"\n================ SEED {seed} ================")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        print(f"\nTrain size: {len(y_train)} | Test size: {len(y_test)}")
        print("Train class distribution:")
        for cls, count in y_train.value_counts().sort_index().items():
            print(f"  Class {cls}: {count}")
        print("Test class distribution:")
        for cls, count in y_test.value_counts().sort_index().items():
            print(f"  Class {cls}: {count}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_res, y_train_res = apply_smote(X_train_scaled, y_train, random_state=seed)

        models = {
            "Random Forest": RandomForestClassifier(
                **best_params["Random Forest"],
                class_weight="balanced",
                random_state=seed,
            ),
            "XGBoost": XGBClassifier(
                **best_params["XGBoost"],
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                random_state=seed,
            ),
            "Logistic Regression": LogisticRegression(
                **best_params["Logistic Regression"],
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            ),
            "MLPClassifier": MLPClassifier(
                **best_params["MLPClassifier"], max_iter=2000, random_state=seed
            ),
        }

        for model_name, model in models.items():
            model.fit(X_train_res, y_train_res)
            metrics = evaluate_model_detailed(model, X_test_scaled, y_test)
            detailed_results[model_name].append(metrics)
            print_detailed_metrics(
                metrics, split_name="test", model_name=model_name, seed=seed
            )

    print(Fore.MAGENTA + "\n\n========== FINAL AGGREGATED SUMMARY ==========")
    for model_name in model_names:
        print_aggregated_summary(model_name, detailed_results[model_name])
