"""
Enhanced Hybrid Scorer — Full Dataset Training.

Trains the 4-layer hybrid model on the complete Guardrailer dataset.
Uses GPU-accelerated XGBoost and lightweight embeddings.

Target: >90% accuracy with <200MB model size.
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
from enhanced_features import extract_enhanced_features, extract_enhanced_features_batch
from patterns import pattern_score
from similarity import TFIDFSimilarity, IDFWeightedKeywords
from embedding_layer import LightweightEmbeddingLayer


DATASET_PATH = Path("/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet")
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = Path(__file__).parent.parent / "training.log"

_log_file = None

def log(msg):
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def train_enhanced(batch_size=2000, seed=42, use_embeddings=True):
    global _log_file
    _log_file = open(LOG_PATH, "w")
    
    try:
        _train_inner(batch_size, seed, use_embeddings)
    finally:
        _log_file.close()
        _log_file = None


def _train_inner(batch_size=2000, seed=42, use_embeddings=True):
    log("=" * 70)
    log("ENHANCED HYBRID SCORER — FULL DATASET TRAINING")
    log("=" * 70)
    log(f"Embeddings: {'ENABLED' if use_embeddings else 'DISABLED'}")
    log("=" * 70)
    
    tracemalloc.start()
    
    log(f"\nLoading full dataset...")
    df = pd.read_parquet(DATASET_PATH)
    log(f"Dataset: {len(df):,} rows")
    
    texts = df["prompt_text"].astype(str).tolist()
    labels = df["is_malicious"].astype(int).tolist()
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    del texts, labels, df
    gc.collect()
    
    log(f"Train: {len(X_train_texts):,} | Test: {len(X_test_texts):,}")
    
    current, peak = tracemalloc.get_traced_memory()
    log(f"Memory: current={current/1e6:.1f}MB, peak={peak/1e6:.1f}MB")
    
    # === Layer 2: TF-IDF Similarity ===
    log("\n" + "=" * 70)
    log("LAYER 2: TF-IDF Similarity")
    log("=" * 70)
    
    log("Fitting similarity scorer on training data...")
    start = time.time()
    sim_scorer = TFIDFSimilarity(max_features=50000)
    sim_scorer.fit(X_train_texts, y_train)
    log(f"  Fitted in {time.time() - start:.1f}s")
    
    log("Fitting keyword scorer...")
    kw_scorer = IDFWeightedKeywords()
    kw_scorer.fit(X_train_texts, y_train)
    
    # === Layer 3: Embedding Layer ===
    embedding_layer = None
    if use_embeddings:
        log("\n" + "=" * 70)
        log("LAYER 3: Embedding Layer (BAAI/bge-small-en-v1.5)")
        log("=" * 70)
        
        log("Initializing embedding layer...")
        embedding_layer = LightweightEmbeddingLayer()
        
        log("Computing training embeddings...")
        start = time.time()
        embedding_layer.fit(X_train_texts, y_train)
        log(f"  Embeddings computed in {time.time() - start:.1f}s")
    
    # === Layer 4: Enhanced Features + XGBoost ===
    log("\n" + "=" * 70)
    log("LAYER 4: Enhanced Features + XGBoost")
    log("=" * 70)
    
    log("Extracting enhanced features (batched)...")
    start = time.time()
    X_hand_train, feature_names = extract_enhanced_features_batch(X_train_texts, batch_size=batch_size)
    log(f"  Handcrafted train shape: {X_hand_train.shape}, time: {time.time() - start:.1f}s")
    
    # Add embedding features (uses cache from fit() — no re-encoding)
    if use_embeddings and embedding_layer is not None:
        log("Computing embedding features (from cache)...")
        start = time.time()
        X_emb_train, _ = embedding_layer.compute_similarity_features_batch(X_train_texts, batch_size=256)
        X_hand_train = np.hstack([X_hand_train, X_emb_train]).astype(np.float32)
        log(f"  With embeddings: {X_hand_train.shape}, time: {time.time() - start:.1f}s")
    
    # TF-IDF features
    log("Building TF-IDF feature matrix...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, norm="l2", dtype=np.float32)
    log("Fitting TF-IDF on training data...")
    start = time.time()
    X_tfidf_train = tfidf.fit_transform(X_train_texts)
    log(f"  TF-IDF train shape: {X_tfidf_train.shape}, time: {time.time() - start:.1f}s")
    
    # Combine all features
    from scipy.sparse import hstack as sparse_hstack, csr_matrix
    X_hand_sparse = csr_matrix(X_hand_train, dtype=np.float32)
    X_train_combined = sparse_hstack([X_hand_sparse, X_tfidf_train], format="csr")
    del X_hand_train, X_hand_sparse, X_tfidf_train
    gc.collect()
    
    # Free embedding model GPU memory before XGBoost
    if use_embeddings and embedding_layer is not None:
        embedding_layer._unload_model()
    gc.collect()
    
    log(f"  Combined train shape: {X_train_combined.shape}, nnz: {X_train_combined.nnz}")
    
    y_train_arr = np.array(y_train)
    
    # Train XGBoost
    log("\nTraining XGBoost on GPU...")
    start = time.time()
    try:
        import xgboost as xgb
        log("  Using XGBoost with GPU (device='cuda')")
        classifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            device="cuda",
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        classifier.fit(X_train_combined, y_train_arr, verbose=False)
        log(f"  GPU training completed in {time.time() - start:.1f}s")
    except Exception as e:
        log(f"  GPU training failed ({e}), falling back to CPU XGBoost")
        import xgboost as xgb
        classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        classifier.fit(X_train_combined, y_train_arr, verbose=False)
        log(f"  CPU training completed in {time.time() - start:.1f}s")
    
    del X_train_combined
    gc.collect()
    
    # === Evaluate ===
    log("\n" + "=" * 70)
    log("EVALUATION ON TEST SET")
    log("=" * 70)
    
    log("Extracting test features (batched)...")
    X_hand_test, _ = extract_enhanced_features_batch(X_test_texts, batch_size=batch_size)
    
    if use_embeddings and embedding_layer is not None:
        log("Computing test embedding features...")
        X_emb_test, _ = embedding_layer.compute_similarity_features_batch(X_test_texts, batch_size=256)
        X_hand_test = np.hstack([X_hand_test, X_emb_test]).astype(np.float32)
    
    X_tfidf_test = tfidf.transform(X_test_texts)
    X_hand_test_sparse = csr_matrix(X_hand_test, dtype=np.float32)
    X_test_combined = sparse_hstack([X_hand_test_sparse, X_tfidf_test], format="csr")
    del X_hand_test, X_tfidf_test
    gc.collect()
    
    log("Running inference...")
    start = time.time()
    y_pred_proba = classifier.predict_proba(X_test_combined)[:, 1]
    inference_time = time.time() - start
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_test_arr = np.array(y_test)
    
    n_test = len(y_test)
    latency_ms = inference_time / n_test * 1000
    throughput = n_test / inference_time
    
    del X_test_combined
    gc.collect()
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred).ravel()
    acc = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, zero_division=0)
    rec = recall_score(y_test_arr, y_pred, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test_arr, y_pred)
    kappa = cohen_kappa_score(y_test_arr, y_pred)
    try:
        auc = roc_auc_score(y_test_arr, y_pred_proba)
    except ValueError:
        auc = None
    
    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(classification_report(y_test_arr, y_pred, target_names=["Safe", "Malicious"]))
    log(f"Accuracy:  {acc:.4f}")
    log(f"Precision: {prec:.4f}")
    log(f"Recall:    {rec:.4f}")
    log(f"F1:        {f1:.4f}")
    log(f"MCC:       {mcc:.4f}")
    log(f"Kappa:     {kappa:.4f}")
    log(f"AUC-ROC:   {auc:.4f}" if auc else "AUC-ROC: N/A")
    log(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    log(f"Latency:   {latency_ms:.3f}ms/prompt ({throughput:.0f}/s)")
    
    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(classifier, MODEL_DIR / "classifier.joblib", compress=3)
    joblib.dump(tfidf, MODEL_DIR / "tfidf.joblib", compress=3)
    sim_scorer.save(MODEL_DIR / "similarity.joblib")
    kw_scorer.save(MODEL_DIR / "keywords.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
    joblib.dump({
        "pattern_threshold": 0.5,
        "similarity_threshold": 0.5,
        "embedding_threshold": 0.5,
        "classifier_threshold": 0.5,
        "ensemble_weights": {
            "pattern": 0.20,
            "similarity": 0.15,
            "embedding": 0.25,
            "classifier": 0.40,
        },
        "use_embeddings": use_embeddings,
    }, MODEL_DIR / "config.joblib")
    log(f"\nModels saved to {MODEL_DIR}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_sizes = {}
    for f in MODEL_DIR.glob("*.joblib"):
        model_sizes[f.stem] = round(f.stat().st_size / 1e6, 2)
    if use_embeddings:
        model_sizes["embedding_model"] = 130.0
    model_sizes["total"] = sum(model_sizes.values())
    
    results = {
        "experiment": "enhanced_hybrid_scorer",
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATASET_PATH),
        "n_train": len(X_train_texts),
        "n_test": n_test,
        "use_embeddings": use_embeddings,
        "classifier": "XGBoost (device=cuda)",
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
    }
    
    results_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {results_path}")
    
    current, peak = tracemalloc.get_traced_memory()
    log(f"\nPeak memory: {peak/1e6:.1f} MB")
    tracemalloc.stop()
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Enhanced Hybrid Scorer")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable embedding layer (CPU-only mode)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for feature extraction")
    args = parser.parse_args()
    
    train_enhanced(
        batch_size=args.batch_size,
        use_embeddings=not args.no_embeddings,
    )
