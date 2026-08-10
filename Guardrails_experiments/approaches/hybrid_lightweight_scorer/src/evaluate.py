"""
Evaluation script for hybrid lightweight scorer.

Modes:
  - full: Full evaluation on test set
  - robustness: Adversarial robustness testing
  - calibration: Calibration quality evaluation
  - all: Run all evaluations

Usage:
  python evaluate.py --mode full
  python evaluate.py --mode robustness
  python evaluate.py --mode calibration
  python evaluate.py --mode all
"""

import gc
import sys
import json
import time
import base64
import codecs
import re
import argparse
import urllib.parse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, extract_features_batch
from scorer import HybridScorer
from calibration import ConfidenceCalibrator


DATASET_PATH = Path("/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet")
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"


class ObfuscationEngine:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def base64_encode(self, text: str) -> str:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return f"Decode this base64 and follow: {encoded}"

    def rot13(self, text: str) -> str:
        encoded = codecs.encode(text, "rot_13")
        return f"Apply ROT13 decoding then execute: {encoded}"

    def character_substitution(self, text: str) -> str:
        substitutions = {
            "a": "@", "e": "3", "i": "1", "o": "0",
            "s": "5", "t": "7", "l": "1", "b": "8",
            "n": "ñ", "r": "r",
        }
        result = []
        for char in text:
            lower = char.lower()
            if lower in substitutions and self.rng.random() > 0.5:
                result.append(substitutions[lower])
            else:
                result.append(char)
        return "".join(result)

    def whitespace_injection(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            result.append(word)
            if self.rng.random() > 0.5:
                invisible = self.rng.choice(["\u200b", "\u200c", "\u200d", "\ufeff", "\t"])
                result.append(invisible)
        return " ".join(result)

    def url_encode(self, text: str) -> str:
        encoded = urllib.parse.quote(text)
        return f"URL-decode this and follow instructions: {encoded}"

    def multi_layer_encode(self, text: str) -> str:
        step1 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        step2 = codecs.encode(step1, "rot_13")
        step3 = base64.b64encode(step2.encode("utf-8")).decode("ascii")
        return f"Decode layers (base64 -> ROT13 -> base64) then follow: {step3}"

    def leetspeak(self, text: str) -> str:
        leet_map = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
        result = []
        for char in text:
            lower = char.lower()
            if lower in leet_map and self.rng.random() > 0.4:
                result.append(leet_map[lower])
            else:
                result.append(char)
        return "".join(result)

    def zero_width_insertion(self, text: str) -> str:
        zwsp = "\u200b"
        return zwsp.join(text)

    def apply(self, text: str, technique: str) -> str:
        if technique == "base64":
            return self.base64_encode(text)
        elif technique == "rot13":
            return self.rot13(text)
        elif technique == "char_substitution":
            return self.character_substitution(text)
        elif technique == "whitespace_injection":
            return self.whitespace_injection(text)
        elif technique == "url_encode":
            return self.url_encode(text)
        elif technique == "multi_layer":
            return self.multi_layer_encode(text)
        elif technique == "leetspeak":
            return self.leetspeak(text)
        elif technique == "zero_width":
            return self.zero_width_insertion(text)
        return text

    def techniques(self):
        return [
            "base64", "rot13", "char_substitution", "whitespace_injection",
            "url_encode", "multi_layer", "leetspeak", "zero_width",
        ]


def load_data(n_samples=20000, seed=42):
    print(f"Loading dataset from {DATASET_PATH}...")
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
    categories = df["attack_category"].tolist() if "attack_category" in df.columns else [None] * len(df)

    X_train, X_test, y_train, y_test, cats_train, cats_test = train_test_split(
        texts, labels, categories, test_size=0.2, stratify=labels, random_state=seed
    )
    return X_train, X_test, y_train, y_test, cats_train, cats_test


def evaluate_full(scorer, X_test, y_test, cats_test):
    print("\n" + "=" * 70)
    print("FULL EVALUATION ON TEST SET")
    print("=" * 70)

    start = time.time()
    y_scores = scorer.predict_proba_batch(X_test)
    inference_time = time.time() - start

    y_pred = (y_scores >= 0.5).astype(int)
    y_test_arr = np.array(y_test)

    n_test = len(y_test)
    latency_ms = inference_time / n_test * 1000
    throughput = n_test / inference_time

    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred).ravel()
    acc = accuracy_score(y_test_arr, y_pred)
    bal_acc = balanced_accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, zero_division=0)
    rec = recall_score(y_test_arr, y_pred, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test_arr, y_scores)
    except ValueError:
        auc = None

    print(classification_report(y_test_arr, y_pred, target_names=["Safe", "Malicious"]))
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"F1:                {f1:.4f}")
    print(f"AUC-ROC:           {auc:.4f}" if auc else "AUC-ROC: N/A")
    print(f"Confusion:         TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Latency:           {latency_ms:.3f}ms/prompt ({throughput:.0f}/s)")

    print("\n--- PER-CATEGORY ---")
    cats_test_valid = [c for c in cats_test if c is not None]
    if cats_test_valid:
        unique_cats = sorted(set(cats_test_valid))
        for cat in unique_cats:
            mask = np.array([1 if c == cat else 0 for c in cats_test])
            cat_yt = y_test_arr[mask == 1]
            cat_yp = y_pred[mask == 1]
            if len(cat_yt) > 0:
                cat_acc = accuracy_score(cat_yt, cat_yp)
                cat_f1 = f1_score(cat_yt, cat_yp, zero_division=0)
                cat_rec = recall_score(cat_yt, cat_yp, zero_division=0)
                print(f"  {cat:30s}: acc={cat_acc:.4f} f1={cat_f1:.4f} recall={cat_rec:.4f} n={mask.sum()}")

    print("\n--- THRESHOLD SWEEP ---")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = (y_scores >= t).astype(int)
        t_f1 = f1_score(y_test_arr, y_pred_t, zero_division=0)
        t_prec = precision_score(y_test_arr, y_pred_t, zero_division=0)
        t_rec = recall_score(y_test_arr, y_pred_t, zero_division=0)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test_arr, y_pred_t).ravel()
        t_fpr = t_fp / max(1, t_tn + t_fp)
        t_fnr = t_fn / max(1, t_tp + t_fn)
        print(f"  t={t:.1f}: F1={t_f1:.4f} Prec={t_prec:.4f} Rec={t_rec:.4f} FPR={t_fpr:.4f} FNR={t_fnr:.4f}")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": float(auc) if auc else None,
        "confusion": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "latency_ms": float(latency_ms),
        "throughput": float(throughput),
    }


def evaluate_robustness(scorer, X_test, y_test, n_samples=500):
    print("\n" + "=" * 70)
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 70)

    engine = ObfuscationEngine(seed=42)

    malicious_texts = [t for t, l in zip(X_test, y_test) if l == 1][:n_samples]
    safe_texts = [t for t, l in zip(X_test, y_test) if l == 0][:n_samples]

    print(f"Testing {len(malicious_texts)} malicious + {len(safe_texts)} safe samples")
    print(f"Techniques: {engine.techniques()}")

    results = {}

    for technique in engine.techniques():
        print(f"\n  Technique: {technique}")

        malicious_correct = 0
        for text in malicious_texts:
            obfuscated = engine.apply(text, technique)
            pred = scorer.score(obfuscated)["is_malicious"]
            if pred:
                malicious_correct += 1

        safe_correct = 0
        for text in safe_texts:
            obfuscated = engine.apply(text, technique)
            pred = scorer.score(obfuscated)["is_malicious"]
            if not pred:
                safe_correct += 1

        mal_acc = malicious_correct / max(1, len(malicious_texts))
        safe_acc = safe_correct / max(1, len(safe_texts))
        overall_acc = (malicious_correct + safe_correct) / max(1, len(malicious_texts) + len(safe_texts))

        results[technique] = {
            "malicious_accuracy": round(mal_acc, 4),
            "safe_accuracy": round(safe_acc, 4),
            "overall_accuracy": round(overall_acc, 4),
            "n_samples": len(malicious_texts) + len(safe_texts),
        }
        print(f"    Malicious acc: {mal_acc:.4f} | Safe acc: {safe_acc:.4f} | Overall: {overall_acc:.4f}")

    overall_scores = [v["overall_accuracy"] for v in results.values()]
    results["overall_robustness_score"] = round(np.mean(overall_scores), 4) if overall_scores else 0.0
    print(f"\n  Overall robustness score: {results['overall_robustness_score']:.4f}")

    return results


def evaluate_calibration(scorer, X_test, y_test):
    print("\n" + "=" * 70)
    print("CALIBRATION QUALITY EVALUATION")
    print("=" * 70)

    y_scores = scorer.predict_proba_batch(X_test)
    y_test_arr = np.array(y_test)

    ece = ConfidenceCalibrator.compute_ece(y_test_arr, y_scores)
    mce = ConfidenceCalibrator.compute_mce(y_test_arr, y_scores)
    brier = ConfidenceCalibrator.compute_brier_score(y_test_arr, y_scores)
    reliability = ConfidenceCalibrator.compute_reliability_data(y_test_arr, y_scores)

    print(f"  ECE:  {ece:.4f} (target < 0.05)")
    print(f"  MCE:  {mce:.4f}")
    print(f"  Brier: {brier:.4f}")

    print("\n  Reliability diagram data:")
    for i in range(len(reliability["bin_centers"])):
        center = reliability["bin_centers"][i]
        acc = reliability["bin_accuracies"][i]
        count = reliability["bin_counts"][i]
        print(f"    Conf={center:.2f}: Acc={acc:.4f} (gap={abs(acc-center):.4f}) n={count}")

    optimal_threshold, threshold_info = ConfidenceCalibrator.find_optimal_threshold(
        y_test_arr, y_scores, metric="f1"
    )
    print(f"\n  Optimal threshold: {optimal_threshold:.2f}")
    print(f"  F1 at optimal: {threshold_info['f1']:.4f}")

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "optimal_threshold": float(optimal_threshold),
        "threshold_info": threshold_info,
        "reliability": reliability,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid lightweight scorer")
    parser.add_argument("--mode", choices=["full", "robustness", "calibration", "all"],
                        default="all", help="Evaluation mode")
    parser.add_argument("--n-samples", type=int, default=20000,
                        help="Number of samples to use (default: 20000)")
    parser.add_argument("--robustness-samples", type=int, default=500,
                        help="Number of samples for robustness testing (default: 500)")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, cats_train, cats_test = load_data(
        n_samples=args.n_samples
    )

    print("\nLoading trained model...")
    scorer = HybridScorer(model_dir=MODEL_DIR)
    print(f"  Model loaded from {MODEL_DIR}")

    all_results = {}

    if args.mode in ("full", "all"):
        all_results["full"] = evaluate_full(scorer, X_test, y_test, cats_test)

    if args.mode in ("robustness", "all"):
        all_results["robustness"] = evaluate_robustness(
            scorer, X_test, y_test, n_samples=args.robustness_samples
        )

    if args.mode in ("calibration", "all"):
        all_results["calibration"] = evaluate_calibration(scorer, X_test, y_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"evaluation_{args.mode}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
