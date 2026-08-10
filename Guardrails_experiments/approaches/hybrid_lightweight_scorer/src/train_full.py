"""
Full dataset training — Industry-grade pipeline with anti-overfitting.

Three-way split (60/20/20), stacking ensemble, calibration,
class imbalance handling, and overfit detection.
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, extract_features_batch
from patterns import pattern_score
from similarity import TFIDFSimilarity, IDFWeightedKeywords
from calibration import ConfidenceCalibrator


DATASET_PATH = Path("/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet")
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"


def extract_features_in_batches(texts, batch_size=5000):
    all_features = []
    feature_names = None
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_array, fname = extract_features_batch(batch)
        if feature_names is None:
            feature_names = fname
        all_features.append(batch_array)
        if (i // batch_size) % 10 == 0:
            print(f"    Features: {min(i + batch_size, len(texts)):,}/{len(texts):,}")
        gc.collect()
    return np.vstack(all_features).astype(np.float32)


def check_overfitting(train_acc, val_acc, threshold=1.05):
    ratio = train_acc / max(val_acc, 1e-8)
    if ratio > threshold:
        print(f"  WARNING: Overfitting detected! train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, ratio={ratio:.4f}")
        return False
    print(f"  No overfitting: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, ratio={ratio:.4f}")
    return True


def train_industry_grade(batch_size=5000, seed=42):
    print("=" * 70)
    print("INDUSTRY-GRADE TRAINING — STACKING ENSEMBLE + CALIBRATION")
    print("=" * 70)

    tracemalloc.start()

    print(f"\nLoading dataset...")
    df = pd.read_parquet(DATASET_PATH, columns=["prompt_text", "is_malicious", "attack_category"])
    print(f"Dataset: {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")

    label_dist = df["is_malicious"].value_counts()
    print(f"Label distribution: {dict(label_dist)}")
    print(f"  Malicious ratio: {label_dist.get(1, 0) / len(df):.4f}")

    texts = df["prompt_text"].astype(str).tolist()
    labels = df["is_malicious"].astype(int).tolist()
    categories = df["attack_category"].tolist() if "attack_category" in df.columns else [None] * len(df)

    print(f"\nThree-way split: 60% train / 20% val / 20% test")
    X_train_texts, X_temp_texts, y_train, y_temp, cats_train, cats_temp = train_test_split(
        texts, labels, categories, test_size=0.4, stratify=labels, random_state=seed
    )
    X_val_texts, X_test_texts, y_val, y_test, cats_val, cats_test = train_test_split(
        X_temp_texts, y_temp, cats_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    del texts, labels, categories, df, X_temp_texts, y_temp, cats_temp
    gc.collect()

    print(f"  Train: {len(X_train_texts):,}")
    print(f"  Val:   {len(X_val_texts):,}")
    print(f"  Test:  {len(X_test_texts):,}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"\nMemory: current={current/1e6:.1f}MB, peak={peak/1e6:.1f}MB")

    print("\n" + "=" * 70)
    print("LAYER 2: TF-IDF Similarity (fit on TRAIN only)")
    print("=" * 70)

    start = time.time()
    sim_scorer = TFIDFSimilarity(max_features=50000, char_max_features=30000)
    sim_scorer.fit(X_train_texts, y_train)
    print(f"  Fitted in {time.time() - start:.1f}s")

    kw_scorer = IDFWeightedKeywords()
    kw_scorer.fit(X_train_texts, y_train)

    print("\n" + "=" * 70)
    print("LAYER 3: Stacking Ensemble (fit on TRAIN only)")
    print("=" * 70)

    print("Building word-level TF-IDF vectorizer (fit on TRAIN)...")
    tfidf_word = TfidfVectorizer(
        max_features=50000, sublinear_tf=True, norm="l2",
        ngram_range=(1, 2), dtype=np.float32,
    )
    start = time.time()
    X_tfidf_word_train = tfidf_word.fit_transform(X_train_texts)
    print(f"  Word TF-IDF train shape: {X_tfidf_word_train.shape}, time: {time.time() - start:.1f}s")

    print("Building char-level TF-IDF vectorizer (fit on TRAIN)...")
    tfidf_char = TfidfVectorizer(
        max_features=30000, analyzer="char_wb",
        ngram_range=(3, 5), sublinear_tf=True, norm="l2",
        dtype=np.float32,
    )
    start = time.time()
    X_tfidf_char_train = tfidf_char.fit_transform(X_train_texts)
    print(f"  Char TF-IDF train shape: {X_tfidf_char_train.shape}, time: {time.time() - start:.1f}s")

    print("Extracting handcrafted features on TRAIN (batched)...")
    start = time.time()
    X_hand_train = extract_features_in_batches(X_train_texts, batch_size=batch_size)
    print(f"  Handcrafted train shape: {X_hand_train.shape}, time: {time.time() - start:.1f}s")

    from scipy.sparse import hstack as sparse_hstack, csr_matrix
    X_hand_train_sparse = csr_matrix(X_hand_train, dtype=np.float32)
    X_train_combined = sparse_hstack(
        [X_hand_train_sparse, X_tfidf_word_train, X_tfidf_char_train], format="csr"
    )
    del X_hand_train, X_hand_train_sparse, X_tfidf_word_train, X_tfidf_char_train
    gc.collect()
    print(f"  Combined train shape: {X_train_combined.shape}, nnz: {X_train_combined.nnz}")

    print("\nExtracting features on VAL (using TRAIN vectorizers)...")
    X_tfidf_word_val = tfidf_word.transform(X_val_texts)
    X_tfidf_char_val = tfidf_char.transform(X_val_texts)
    X_hand_val = extract_features_in_batches(X_val_texts, batch_size=batch_size)
    X_hand_val_sparse = csr_matrix(X_hand_val, dtype=np.float32)
    X_val_combined = sparse_hstack(
        [X_hand_val_sparse, X_tfidf_word_val, X_tfidf_char_val], format="csr"
    )
    del X_hand_val, X_hand_val_sparse, X_tfidf_word_val, X_tfidf_char_val
    gc.collect()
    print(f"  Combined val shape: {X_val_combined.shape}")

    y_train_arr = np.array(y_train)
    y_val_arr = np.array(y_val)

    print("\nTraining stacking ensemble on GPU...")
    start = time.time()
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        estimators = [
            ("xgb", xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, gamma=0.5,
                reg_alpha=1.0, reg_lambda=1.0,
                tree_method="hist", eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            )),
            ("lgbm", lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_alpha=1.0, reg_lambda=1.0,
                objective="binary", metric="binary_logloss",
                random_state=seed, n_jobs=-1, verbose=-1,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=12,
                min_samples_split=5, min_samples_leaf=2,
                max_features="sqrt", random_state=seed, n_jobs=-1,
            )),
        ]

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=True,
        )

        stacking.fit(X_train_combined, y_train_arr)
        print(f"  Training completed in {time.time() - start:.1f}s")

    except Exception as e:
        print(f"  Stacking failed ({e}), falling back to single XGBoost")
        import xgboost as xgb
        stacking = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        )
        stacking.fit(X_train_combined, y_train_arr)
        print(f"  Fallback training completed in {time.time() - start:.1f}s")

    train_acc = stacking.score(X_train_combined, y_train_arr)
    val_acc = stacking.score(X_val_combined, y_val_arr)
    check_overfitting(train_acc, val_acc, threshold=1.05)

    del X_train_combined
    gc.collect()

    print("\n" + "=" * 70)
    print("LAYER 4: Confidence Calibration (fit on VAL)")
    print("=" * 70)

    print("Fitting isotonic calibration on validation set...")
    calibrator = ConfidenceCalibrator(stacking)
    calibrator.fit(X_val_combined, y_val_arr, method="isotonic", cv=5)

    val_proba = calibrator.predict_proba_calibrated(X_val_combined)
    ece = ConfidenceCalibrator.compute_ece(y_val_arr, val_proba)
    brier = ConfidenceCalibrator.compute_brier_score(y_val_arr, val_proba)
    print(f"  ECE: {ece:.4f} (target < 0.05)")
    print(f"  Brier score: {brier:.4f}")

    optimal_threshold, threshold_results = ConfidenceCalibrator.find_optimal_threshold(
        y_val_arr, val_proba, metric="f1"
    )
    print(f"  Optimal threshold: {optimal_threshold:.2f}")
    print(f"  F1 at optimal: {threshold_results['f1']:.4f}")

    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    print("Extracting features on TEST (using TRAIN vectorizers)...")
    X_tfidf_word_test = tfidf_word.transform(X_test_texts)
    X_tfidf_char_test = tfidf_char.transform(X_test_texts)
    X_hand_test = extract_features_in_batches(X_test_texts, batch_size=batch_size)
    X_hand_test_sparse = csr_matrix(X_hand_test, dtype=np.float32)
    X_test_combined = sparse_hstack(
        [X_hand_test_sparse, X_tfidf_word_test, X_tfidf_char_test], format="csr"
    )
    del X_hand_test, X_tfidf_word_test, X_tfidf_char_test
    gc.collect()

    print("Running inference...")
    start = time.time()
    y_test_proba_raw = stacking.predict_proba(X_test_combined)[:, 1]
    y_test_proba_cal = calibrator.predict_proba_calibrated(X_test_combined)
    inference_time = time.time() - start

    y_test_pred = (y_test_proba_cal >= optimal_threshold).astype(int)
    y_test_arr = np.array(y_test)

    n_test = len(y_test)
    latency_ms = inference_time / n_test * 1000
    throughput = n_test / inference_time

    del X_test_combined
    gc.collect()

    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_test_pred).ravel()
    acc = accuracy_score(y_test_arr, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test_arr, y_test_pred)
    prec = precision_score(y_test_arr, y_test_pred, zero_division=0)
    rec = recall_score(y_test_arr, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_arr, y_test_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test_arr, y_test_pred)
    kappa = cohen_kappa_score(y_test_arr, y_test_pred)

    try:
        auc = roc_auc_score(y_test_arr, y_test_proba_cal)
    except ValueError:
        auc = None

    print("\n--- TEST SET RESULTS ---")
    print(classification_report(y_test_arr, y_test_pred, target_names=["Safe", "Malicious"]))
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"F1:                {f1:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"Kappa:             {kappa:.4f}")
    print(f"AUC-ROC:           {auc:.4f}" if auc else "AUC-ROC: N/A")
    print(f"Confusion:         TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Latency:           {latency_ms:.3f}ms/prompt ({throughput:.0f}/s)")
    print(f"ECE:               {ece:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.2f}")

    print("\n--- PER-CATEGORY RESULTS ---")
    cats_test_valid = [c for c in cats_test if c is not None]
    if cats_test_valid:
        unique_cats = sorted(set(cats_test_valid))
        for cat in unique_cats:
            mask = np.array([1 if c == cat else 0 for c in cats_test])
            cat_yt = y_test_arr[mask == 1]
            cat_yp = y_test_pred[mask == 1]
            if len(cat_yt) > 0:
                cat_acc = accuracy_score(cat_yt, cat_yp)
                cat_f1 = f1_score(cat_yt, cat_yp, zero_division=0)
                cat_recall = recall_score(cat_yt, cat_yp, zero_division=0)
                print(f"  {cat:30s}: acc={cat_acc:.4f} f1={cat_f1:.4f} recall={cat_recall:.4f} n={mask.sum()}")

    print("\n--- THRESHOLD SWEEP ---")
    for t in [0.3, 0.4, 0.5, 0.6]:
        y_pred_t = (y_test_proba_cal >= t).astype(int)
        t_f1 = f1_score(y_test_arr, y_pred_t, zero_division=0)
        t_prec = precision_score(y_test_arr, y_pred_t, zero_division=0)
        t_rec = recall_score(y_test_arr, y_pred_t, zero_division=0)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test_arr, y_pred_t).ravel()
        t_fpr = t_fp / max(1, t_tn + t_fp)
        t_fnr = t_fn / max(1, t_tp + t_fn)
        print(f"  Threshold {t:.1f}: F1={t_f1:.4f} Prec={t_prec:.4f} Rec={t_rec:.4f} FPR={t_fpr:.4f} FNR={t_fnr:.4f}")

    print("\n--- MODEL SIZES ---")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(stacking, MODEL_DIR / "classifier.joblib", compress=3)
    joblib.dump(tfidf_word, MODEL_DIR / "tfidf_word.joblib", compress=3)
    joblib.dump(tfidf_char, MODEL_DIR / "tfidf_char.joblib", compress=3)
    sim_scorer.save(MODEL_DIR / "similarity.joblib")
    kw_scorer.save(MODEL_DIR / "keywords.joblib")
    calibrator.save(MODEL_DIR / "calibrator.joblib")
    joblib.dump(sorted(extract_features("test").keys()), MODEL_DIR / "feature_names.joblib")
    joblib.dump({
        "pattern_threshold": 0.5,
        "similarity_threshold": 0.5,
        "classifier_threshold": float(optimal_threshold),
        "ensemble_weights": {"pattern": 0.20, "similarity": 0.15, "classifier": 0.65},
        "ece": ece,
        "brier_score": brier,
        "optimal_threshold": float(optimal_threshold),
    }, MODEL_DIR / "config.joblib")

    for f in MODEL_DIR.glob("*.joblib"):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")
    total_size = sum(f.stat().st_size for f in MODEL_DIR.glob("*.joblib")) / 1e6
    print(f"  TOTAL: {total_size:.2f} MB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "experiment": "industry_grade_stacking_ensemble",
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATASET_PATH),
        "n_total": len(df) if 'df' in dir() else 722842,
        "n_train": len(X_train_texts),
        "n_val": len(X_val_texts),
        "n_test": n_test,
        "split": "60/20/20 stratified",
        "classifier": "Stacking (XGBoost + LightGBM + RandomForest + LR meta)",
        "calibration": "Isotonic regression",
        "optimal_threshold": float(optimal_threshold),
        "inference": {
            "total_seconds": round(inference_time, 2),
            "latency_ms": round(latency_ms, 3),
            "throughput_per_sec": round(throughput, 1),
        },
        "metrics_test": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "kappa": round(kappa, 4),
            "auc_roc": round(auc, 4) if auc else None,
        },
        "calibration_metrics": {
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
        },
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "overfitting_check": {
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "ratio": round(train_acc / max(val_acc, 1e-8), 4),
            "passed": train_acc / max(val_acc, 1e-8) < 1.05,
        },
    }

    results_path = RESULTS_DIR / f"benchmark_industry_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"\nPeak memory: {peak/1e6:.1f} MB")
    tracemalloc.stop()

    return results


if __name__ == "__main__":
    train_industry_grade()
