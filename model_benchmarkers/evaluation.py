"""
Research-grade evaluation metrics for prompt guardrails benchmarking.

Metrics computed:
    - Accuracy, Precision, Recall, F1 (binary + macro/weighted)
    - Specificity (True Negative Rate)
    - AUC-ROC (with bootstrap CI)
    - Matthews Correlation Coefficient (MCC)
    - Cohen's Kappa
    - FPR / FNR / FDR / FOR
    - Per-category breakdown
    - Latency stats (mean, p50, p95, p99)
    - Bootstrap confidence intervals (95%)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from scipy import stats


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return round(lower, 4), round(upper, 4)


def evaluate_model(y_true, y_pred, y_scores=None, categories=None,
                   latencies=None, label="model", n_bootstrap=10000):
    """
    Compute full research-grade evaluation metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_scores: Optional prediction confidence scores for AUC-ROC
        categories: Optional attack category labels for per-class breakdown
        latencies: Optional list of per-prompt inference times (seconds)
        label: Model name
        n_bootstrap: Number of bootstrap samples for CI

    Returns:
        dict with all metrics and confidence intervals
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1_binary, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
    forge = fn / (fn + tn) if (fn + tn) > 0 else 0

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # AUC-ROC
    auc_roc = None
    auc_ci = None
    if y_scores is not None:
        y_scores = np.array(y_scores)
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            boot_aucs = []
            rng = np.random.RandomState(42)
            for _ in range(n_bootstrap):
                idx = rng.choice(len(y_true), size=len(y_true), replace=True)
                if len(np.unique(y_true[idx])) > 1:
                    boot_aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
            if boot_aucs:
                auc_ci = (
                    round(np.percentile(boot_aucs, 2.5), 4),
                    round(np.percentile(boot_aucs, 97.5), 4),
                )
        except ValueError:
            pass

    # Bootstrap CIs for key metrics
    binary_correct = (y_true == y_pred).astype(int)
    acc_ci = bootstrap_ci(binary_correct, n_bootstrap)

    tp_mask = ((y_true == 1) & (y_pred == 1)).astype(int)
    if tp_mask.sum() > 0:
        prec_ci = bootstrap_ci(tp_mask, n_bootstrap)
    else:
        prec_ci = (0.0, 0.0)

    recall_mask = ((y_true == 1) & (y_pred == 1)).astype(int)
    denom = y_true
    if denom.sum() > 0:
        rec_vals = recall_mask / denom
        rec_ci = bootstrap_ci(rec_vals[denom == 1], n_bootstrap) if denom.sum() > 0 else (0.0, 0.0)
    else:
        rec_ci = (0.0, 0.0)

    # Latency stats
    latency_stats = None
    if latencies is not None:
        latencies = np.array(latencies)
        latency_stats = {
            "mean_ms": round(np.mean(latencies) * 1000, 2),
            "median_ms": round(np.median(latencies) * 1000, 2),
            "p95_ms": round(np.percentile(latencies, 95) * 1000, 2),
            "p99_ms": round(np.percentile(latencies, 99) * 1000, 2),
            "std_ms": round(np.std(latencies) * 1000, 2),
            "total_seconds": round(np.sum(latencies), 2),
            "prompts_per_second": round(len(latencies) / np.sum(latencies), 1) if np.sum(latencies) > 0 else 0,
        }

    results = {
        "label": label,
        "total_samples": len(y_true),
        "class_distribution": {
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum()),
        },
        "confusion_matrix": {
            "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
        },
        "accuracy": round(acc, 4),
        "accuracy_ci95": acc_ci,
        "precision": round(precision, 4),
        "precision_ci95": prec_ci,
        "recall": round(recall, 4),
        "recall_ci95": rec_ci,
        "f1_binary": round(f1_binary, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "false_discovery_rate": round(fdr, 4),
        "false_omission_rate": round(forge, 4),
        "matthews_corrcoef": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "auc_roc": round(auc_roc, 4) if auc_roc else None,
        "auc_roc_ci95": auc_ci,
    }

    if latency_stats:
        results["latency"] = latency_stats

    if categories is not None:
        cat_report = {}
        for cat in sorted(categories.unique()):
            mask = categories.values == cat
            cat_true = y_true[mask]
            cat_pred = y_pred[mask]
            if len(cat_true) > 0:
                cat_acc = accuracy_score(cat_true, cat_pred)
                cat_p, cat_r, cat_f1, cat_sup = precision_recall_fscore_support(
                    cat_true, cat_pred, average="binary", pos_label=1, zero_division=0
                )
                cat_tn, cat_fp, cat_fn, cat_tp = confusion_matrix(
                    cat_true, cat_pred, labels=[0, 1]
                ).ravel()
                cat_report[cat] = {
                    "samples": int(mask.sum()),
                    "positive": int(cat_true.sum()),
                    "negative": int(len(cat_true) - cat_true.sum()),
                    "accuracy": round(cat_acc, 4),
                    "precision": round(cat_p, 4),
                    "recall": round(cat_r, 4),
                    "f1": round(cat_f1, 4),
                    "specificity": round(cat_tn / (cat_tn + cat_fp), 4) if (cat_tn + cat_fp) > 0 else 0,
                    "fpr": round(cat_fp / (cat_fp + cat_tn), 4) if (cat_fp + cat_tn) > 0 else 0,
                    "fnr": round(cat_fn / (cat_fn + cat_tp), 4) if (cat_fn + cat_tp) > 0 else 0,
                }
        results["per_category"] = cat_report

    return results


def compare_models(all_results):
    """Build comparison DataFrame from multiple evaluate_model() outputs."""
    rows = []
    for r in all_results:
        row = {
            "Model": r["label"],
            "Accuracy": r["accuracy"],
            "Acc CI95": r.get("accuracy_ci95", ""),
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1 Binary": r["f1_binary"],
            "F1 Macro": r["f1_macro"],
            "Specificity": r["specificity"],
            "MCC": r["matthews_corrcoef"],
            "Kappa": r["cohen_kappa"],
            "AUC-ROC": r.get("auc_roc", ""),
            "FPR": r["false_positive_rate"],
            "FNR": r["false_negative_rate"],
            "TP": r["confusion_matrix"]["tp"],
            "TN": r["confusion_matrix"]["tn"],
            "FP": r["confusion_matrix"]["fp"],
            "FN": r["confusion_matrix"]["fn"],
        }
        if "latency" in r:
            row["Latency (ms)"] = r["latency"]["mean_ms"]
            row["P95 (ms)"] = r["latency"]["p95_ms"]
            row["Prompts/s"] = r["latency"]["prompts_per_second"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 Binary", ascending=False).reset_index(drop=True)
    return df


def print_comparison(all_results):
    """Pretty print model comparison table."""
    df = compare_models(all_results)

    print("\n" + "=" * 120)
    print("MODEL BENCHMARK COMPARISON")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)

    if len(df) > 1:
        best_f1 = df.loc[df["F1 Binary"].idxmax(), "Model"]
        best_acc = df.loc[df["Accuracy"].idxmax(), "Model"]
        best_recall = df.loc[df["Recall"].idxmax(), "Model"]
        best_specificity = df.loc[df["Specificity"].idxmax(), "Model"]
        print(f"\nBest F1:        {best_f1}")
        print(f"Best Accuracy:  {best_acc}")
        print(f"Best Recall:    {best_recall}")
        print(f"Best Specificity: {best_specificity}")

    return df
