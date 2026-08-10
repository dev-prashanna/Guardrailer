"""
Train and benchmark the hybrid lightweight scorer.

Memory-optimized: processes data in chunks, no full-dense TF-IDF matrix.
Uses GPU if cuML is available, else sklearn with n_jobs=-1.
"""

import gc
import sys
import json
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from scorer import HybridScorer


DATASET_PATH = Path("/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet")
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"


def train(n_samples=30000, seed=42):
    print("=" * 70)
    print("HYBRID LIGHTWEIGHT SCORER — TRAINING (Memory-Optimized)")
    print("=" * 70)

    tracemalloc.start()

    print(f"\nLoading dataset...")
    df = pd.read_parquet(DATASET_PATH, columns=["prompt_text", "is_malicious", "attack_category"])
    print(f"Full dataset: {len(df):,} rows")

    if n_samples < len(df):
        print(f"Subsampling to {n_samples:,} (stratified)...")
        parts = []
        for label, group in df.groupby("is_malicious"):
            n_take = int(n_samples * len(group) / len(df))
            parts.append(group.sample(n=min(n_take, len(group)), random_state=seed))
        df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"  Got {len(df):,} samples")

    texts = df["prompt_text"].astype(str).tolist()
    labels = df["is_malicious"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    del texts, labels
    gc.collect()

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory before training: current={current/1e6:.1f}MB, peak={peak/1e6:.1f}MB")

    scorer = HybridScorer()

    print("\nTraining all layers...")
    start = time.time()
    scorer.fit(X_train, y_train, model_dir=MODEL_DIR)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.1f}s")

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory after training: current={current/1e6:.1f}MB, peak={peak/1e6:.1f}MB")

    print("\nBenchmarking on test set...")
    start = time.time()
    y_pred = []
    y_scores = []
    for i in range(0, len(X_test), 500):
        batch = X_test[i:i + 500]
        for text in batch:
            result = scorer.score(text)
            y_pred.append(int(result["is_malicious"]))
            y_scores.append(result["score"])
    inference_time = time.time() - start

    n_test = len(X_test)
    latency_ms = inference_time / n_test * 1000
    throughput = n_test / inference_time

    print(f"\nInference: {inference_time:.2f}s for {n_test:,} samples")
    print(f"  Latency: {latency_ms:.3f}ms/prompt | Throughput: {throughput:.0f}/s")

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred_arr).ravel()
    acc = accuracy_score(y_test_arr, y_pred_arr)
    prec = precision_score(y_test_arr, y_pred_arr, zero_division=0)
    rec = recall_score(y_test_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred_arr, zero_division=0)
    mcc = matthews_corrcoef(y_test_arr, y_pred_arr)
    kappa = cohen_kappa_score(y_test_arr, y_pred_arr)

    try:
        auc = roc_auc_score(y_test_arr, y_scores_arr)
    except ValueError:
        auc = None

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(classification_report(y_test_arr, y_pred_arr, target_names=["Safe", "Malicious"]))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print(f"AUC-ROC:   {auc:.4f}" if auc else "AUC-ROC: N/A")
    print(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    model_sizes = scorer.get_model_size(MODEL_DIR)
    print(f"\nModel sizes:")
    for name, size in model_sizes.items():
        print(f"  {name}: {size:.2f} MB")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "experiment": "hybrid_lightweight_scorer",
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATASET_PATH),
        "n_samples_total": len(df) if n_samples >= len(df) else n_samples,
        "n_train": len(X_train),
        "n_test": n_test,
        "train_time_seconds": round(train_time, 2),
        "inference": {
            "total_seconds": round(inference_time, 2),
            "latency_ms": round(latency_ms, 3),
            "throughput_per_sec": round(throughput, 1),
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "kappa": round(kappa, 4),
            "auc_roc": round(auc, 4) if auc else None,
        },
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "model_sizes_mb": model_sizes,
        "memory": {
            "peak_mb": round(peak / 1e6, 1),
        },
    }

    results_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\nPer-category breakdown:")
    if "attack_category" in df.columns:
        cats = df["attack_category"].values
        X_test_cats = [cats[i] for i in range(len(X_test))]

        seen_cats = set()
        for i, (yp, yt) in enumerate(zip(y_pred, y_test)):
            if i < len(X_test_cats):
                seen_cats.add(X_test_cats[i])

        for cat in sorted(seen_cats):
            mask = np.array([1 if c == cat else 0 for c in X_test_cats])
            cat_yt = y_test_arr[mask == 1]
            cat_yp = y_pred_arr[mask == 1]
            if len(cat_yt) > 0:
                cat_acc = accuracy_score(cat_yt, cat_yp)
                cat_f1 = f1_score(cat_yt, cat_yp, zero_division=0)
                print(f"  {cat:30s}: acc={cat_acc:.4f} f1={cat_f1:.4f} n={mask.sum()}")

    return scorer, results


if __name__ == "__main__":
    train()
