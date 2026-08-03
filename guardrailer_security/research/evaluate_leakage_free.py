"""
evaluate_leakage_free.py
Comprehensive, leakage-free evaluation of the Guardrailer multi-signal security system.

Addresses ALL data leakage issues identified in the audit:
  1. Centroids computed from full corpus -> FIXED: per-fold training-only centroids
  2. IDF computed from full corpus -> FIXED: per-fold training-only document frequencies
  3. avg_text_length from full corpus -> FIXED: per-fold training-only average

Architecture:
  For each of 5 CV folds:
    1. Split data into train/val (80/20 stratified)
    2. Compute ALL corpus-dependent statistics from TRAINING fold ONLY
    3. Compute features for validation fold using training-fold statistics
    4. Train model on training fold
    5. Evaluate on validation fold

Signals (7 total, all leakage-free):
  1. centroid:       cosine similarity to nearest TRAINING-FOLD category centroid
  2. sparse_idf:     IDF score computed from TRAINING-FOLD document frequencies
  3. perplexity:     computed from text (no corpus dependency)
  4. entropy:        computed from text (no corpus dependency)
  5. token_freq:     computed from text + static word lists
  6. ngram:          computed from text + static attack ngram set
  7. length_norm:    log(1+len) / log(1+mean_train_length)
"""

import os
import sys
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/prashanna/Documents/Guardrailer")
sys.path.insert(0, "/home/prashanna/Documents/Guardrailer/guardrailer_security")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("/home/prashanna/Documents/Guardrailer/evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = RESULTS_DIR / "leakage_free_report.json"

PARQUET_PATH = "/home/prashanna/Documents/Guardrailer/guardrailer_security/unified_security_dataset.parquet"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64
N_SAMPLES = 10000
N_FOLDS = 5

FEATURE_NAMES = ["centroid", "sparse_idf", "perplexity", "entropy", "token_freq", "ngram", "length_norm"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# 1. Static constants (no data dependency)
# ===========================================================================

from constants import SPARSE_KEYWORDS

COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
    "had", "does", "did", "am", "may", "might", "shall", "should",
}

ATTACK_WORDS = {
    "ignore", "override", "bypass", "jailbreak", "reveal", "extract",
    "system", "prompt", "instructions", "rules", "restrictions", "safety",
    "mode", "debug", "admin", "root", "developer", "unrestricted", "evil",
    "dan", "persona", "roleplay", "pretend", "hypothetical", "imagine",
    "base64", "rot13", "hex", "encoded", "obfuscated", "decode",
    "output", "display", "print", "show", "expose",
}

ATTACK_NGRAMS = {
    "ignore previous", "override instructions", "bypass safety",
    "jailbreak prompt", "system prompt", "output instructions",
    "reveal rules", "extract prompt", "new instructions",
    "you are now", "do anything now", "developer mode",
    "debug mode", "admin mode", "root mode", "no restrictions",
    "no rules", "no limits", "unrestricted", "evil mode",
    "pretend you", "act as if", "roleplay as", "hypothetical scenario",
    "base64 encoded", "rot13 encoded", "hex encoded", "decode this",
    "ignore all", "forget instructions", "disregard rules",
    "override safety", "bypass restrictions", "break guidelines",
    "hidden instructions", "secret instructions", "hidden rules",
    "what are your rules", "how do you work", "what instructions",
    "show me your", "output your", "reveal your", "display your",
    "admin rights", "giving admin", "grant admin", "elevate privileges",
    "escalate privileges", "root access", "superuser", "sudo access",
    "override system", "replace instructions", "new system prompt",
    "your new instructions", "disregard all", "forget everything",
    "ignore safety", "bypass safety", "disable safety",
    "you are now a", "act as if", "pretend to be",
    "simulate being", "roleplay as", "impersonate",
    "hypothetically", "in a hypothetical", "if you were",
    "imagine you are", "assume you are",
}


# ===========================================================================
# 2. Leakage-free signal computation functions
# ===========================================================================

def compute_centroid_score_single(embedding, centroids):
    """Compute max cosine similarity to training-fold category centroids."""
    q_norm = np.linalg.norm(embedding)
    if q_norm < 1e-8:
        return 0.0
    best = 0.0
    for centroid in centroids.values():
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-8:
            continue
        sim = float(np.dot(embedding, centroid) / (q_norm * c_norm))
        if sim > best:
            best = sim
    return best


def compute_centroid_scores_batch(embeddings, centroids):
    """Vectorized centroid score computation for a batch of embeddings."""
    if not centroids:
        return np.zeros(len(embeddings), dtype=np.float64)
    centroid_matrix = np.array(list(centroids.values()), dtype=np.float64)
    q_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-8)
    c_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    c_norms = np.maximum(c_norms, 1e-8)
    centroid_matrix_normed = centroid_matrix / c_norms
    sims = (embeddings / q_norms) @ centroid_matrix_normed.T
    return np.max(sims, axis=1)


def compute_train_centroids(train_embeddings, train_categories):
    """Compute per-category centroids from training data only."""
    unique_cats = np.unique(train_categories)
    centroids = {}
    for cat in unique_cats:
        mask = train_categories == cat
        if mask.sum() > 0:
            mean_emb = train_embeddings[mask].mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-8:
                centroids[cat] = (mean_emb / norm).astype(np.float64)
            else:
                centroids[cat] = mean_emb.astype(np.float64)
    return centroids


def compute_train_idf(train_texts):
    """Compute IDF values from training texts only."""
    doc_freq = Counter()
    N_train = len(train_texts)
    for text in train_texts:
        lower = text.lower()
        for kw in SPARSE_KEYWORDS:
            if kw in lower:
                doc_freq[kw] += 1
    keyword_idf = {
        kw: math.log((N_train + 1) / (df + 1)) + 1
        for kw, df in doc_freq.items()
    }
    return keyword_idf


def compute_sparse_idf_score(text, keyword_idf):
    """Compute IDF-weighted sparse score using per-fold IDF values."""
    lower = text.lower()
    total_idf = 0.0
    for kw in SPARSE_KEYWORDS:
        if kw in lower:
            total_idf += keyword_idf.get(kw, 1.0)
    max_possible = sum(keyword_idf.values()) if keyword_idf else len(SPARSE_KEYWORDS)
    if max_possible > 0:
        return total_idf / max_possible
    return 0.0


def compute_sparse_idf_scores_batch(texts, keyword_idf):
    """Vectorized-ish batch IDF score computation."""
    scores = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        scores[i] = compute_sparse_idf_score(text, keyword_idf)
    return scores


def compute_perplexity_score(text):
    """Perplexity-based anomaly score (no corpus dependency)."""
    tokens = text.lower().split()
    if len(tokens) < 2:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    avg_token_len = np.mean([len(t) for t in tokens])
    char_counts = Counter(text.lower())
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    char_entropy = -sum(
        (c / total_chars) * math.log2(c / total_chars)
        for c in char_counts.values()
    )
    entropy_score = char_entropy / 8.0
    diversity_score = unique_ratio
    perplexity_anomaly = (
        0.4 * entropy_score
        + 0.3 * diversity_score
        + 0.3 * min(1.0, avg_token_len / 10.0)
    )
    return max(0.0, min(1.0, perplexity_anomaly))


def compute_perplexity_scores_batch(texts):
    scores = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        scores[i] = compute_perplexity_score(text)
    return scores


def compute_entropy_score(text):
    """Shannon entropy score (no corpus dependency)."""
    if not text:
        return 0.0
    char_counts = Counter(text.lower())
    total = len(text)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in char_counts.values()
    )
    words = text.lower().split()
    if not words:
        return 0.0
    word_counts = Counter(words)
    total_words = len(words)
    word_entropy = -sum(
        (c / total_words) * math.log2(c / total_words)
        for c in word_counts.values()
    )
    char_norm = entropy / 6.5
    word_norm = word_entropy / math.log2(max(len(word_counts), 2))
    return max(0.0, min(1.0, 0.6 * char_norm + 0.4 * word_norm))


def compute_entropy_scores_batch(texts):
    scores = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        scores[i] = compute_entropy_score(text)
    return scores


def compute_token_frequency_score(text):
    """Token frequency anomaly score (uses static word lists)."""
    words = text.lower().split()
    if not words:
        return 0.0
    common_count = sum(1 for w in words if w in COMMON_WORDS)
    attack_count = sum(1 for w in words if w in ATTACK_WORDS)
    common_ratio = common_count / len(words)
    attack_ratio = attack_count / len(words)
    rarity_score = 1.0 - common_ratio
    attack_score = min(1.0, attack_ratio * 3.0)
    return max(0.0, min(1.0, 0.5 * rarity_score + 0.5 * attack_score))


def compute_token_freq_scores_batch(texts):
    scores = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        scores[i] = compute_token_frequency_score(text)
    return scores


def compute_ngram_overlap_score(text):
    """N-gram overlap with static attack ngram set (no corpus dependency)."""
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) < 2:
        return 0.0
    bigrams = [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    bigram_matches = sum(1 for bg in bigrams if bg in ATTACK_NGRAMS)
    trigram_matches = sum(1 for tg in trigrams if tg in ATTACK_NGRAMS)
    total_ngrams = len(bigrams) + len(trigrams)
    if total_ngrams == 0:
        return 0.0
    match_score = (bigram_matches * 1.0 + trigram_matches * 1.5) / total_ngrams
    substring_matches = sum(1 for phrase in ATTACK_NGRAMS if phrase in text_lower)
    substring_score = min(1.0, substring_matches * 0.2)
    return max(0.0, min(1.0, 0.5 * match_score + 0.5 * substring_score))


def compute_ngram_scores_batch(texts):
    scores = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        scores[i] = compute_ngram_overlap_score(text)
    return scores


def compute_length_norm_scores_batch(text_lengths, mean_train_length):
    """Length normalization: log(1+len) / log(1+mean_train_length)."""
    denom = math.log1p(mean_train_length) if mean_train_length > 0 else 1.0
    return np.log1p(text_lengths.astype(np.float64)) / denom


# ===========================================================================
# 3. Full leakage-free feature matrix builder
# ===========================================================================

def build_features_for_fold(texts, embeddings, train_texts, train_embeddings, train_categories):
    """Build the complete 7-signal feature matrix for a set of samples,
    using ONLY training-fold statistics for corpus-dependent signals.

    Returns: np.ndarray of shape (len(texts), 7)
    """
    # Corpus-dependent: centroids from training fold
    centroids = compute_train_centroids(train_embeddings, train_categories)
    centroid_scores = compute_centroid_scores_batch(embeddings, centroids)

    # Corpus-dependent: IDF from training fold
    keyword_idf = compute_train_idf(train_texts)
    idf_scores = compute_sparse_idf_scores_batch(texts, keyword_idf)

    # Corpus-dependent: avg text length from training fold
    train_lengths = np.array([len(t) for t in train_texts], dtype=np.float64)
    mean_train_length = float(np.mean(train_lengths))

    # Text-only signals (no corpus dependency)
    perplexity_scores = compute_perplexity_scores_batch(texts)
    entropy_scores = compute_entropy_scores_batch(texts)
    token_freq_scores = compute_token_freq_scores_batch(texts)
    ngram_scores = compute_ngram_scores_batch(texts)

    # Length norm from training mean
    text_lengths = np.array([len(t) for t in texts], dtype=np.float64)
    length_norm_scores = compute_length_norm_scores_batch(text_lengths, mean_train_length)

    features = np.column_stack([
        centroid_scores,
        idf_scores,
        perplexity_scores,
        entropy_scores,
        token_freq_scores,
        ngram_scores,
        length_norm_scores,
    ])
    return features


# ===========================================================================
# 4. Metrics helpers
# ===========================================================================

def compute_metrics(y_true, y_pred, y_proba=None):
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, matthews_corrcoef,
        roc_auc_score, precision_score, recall_score, confusion_matrix,
    )
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0
    return {
        "balanced_accuracy": float(ba),
        "f1": float(f1),
        "mcc": float(mcc),
        "auc_roc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.RandomState(SEED)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(scores, alpha * 100)),
        float(np.percentile(scores, (1 - alpha) * 100)),
    )


# ===========================================================================
# 5. Main evaluation pipeline
# ===========================================================================

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    log("=" * 70)
    log("  GUARDRAILER LEAKAGE-FREE EVALUATION")
    log("  All corpus-dependent statistics computed per-fold from training data only")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # 5a. Load and sample dataset
    # -----------------------------------------------------------------------
    log("Loading dataset...")
    df = pd.read_parquet(PARQUET_PATH)
    log(f"  Full dataset: {len(df)} rows, columns: {list(df.columns)}")

    df["is_malicious_int"] = df["is_malicious"].astype(int)

    target_per_class = N_SAMPLES // 2
    stratified_parts = []

    for label in [0, 1]:
        subset = df[df["is_malicious_int"] == label]
        categories = subset["attack_category"].unique()
        per_category = max(1, target_per_class // len(categories))
        sampled_parts = []
        for cat in categories:
            cat_subset = subset[subset["attack_category"] == cat]
            n_take = min(per_category, len(cat_subset))
            sampled_parts.append(cat_subset.sample(n=n_take, random_state=SEED))
        part = pd.concat(sampled_parts, ignore_index=True)
        if len(part) > target_per_class:
            part = part.sample(n=target_per_class, random_state=SEED)
        stratified_parts.append(part)

    sample_df = pd.concat(stratified_parts, ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_mal = int(sample_df["is_malicious_int"].sum())
    n_ben = len(sample_df) - n_mal
    log(f"  Sampled {len(sample_df)} rows ({n_mal} malicious, {n_ben} benign)")
    log(f"  Category distribution:\n{sample_df['attack_category'].value_counts().to_string()}")

    texts = sample_df["prompt_text"].tolist()
    labels = sample_df["is_malicious_int"].values
    categories = sample_df["attack_category"].values

    # -----------------------------------------------------------------------
    # 5b. Compute embeddings (one-time, no leakage since embeddings are
    #     per-sample text embeddings, not corpus statistics)
    # -----------------------------------------------------------------------
    log("Loading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    log(f"  Model loaded: {EMBEDDING_MODEL}")

    log("Computing embeddings for all samples...")
    embeddings = st_model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )
    log(f"  Embeddings shape: {embeddings.shape}")

    # -----------------------------------------------------------------------
    # 5c. 5-fold stratified CV with per-fold leakage-free feature computation
    # -----------------------------------------------------------------------
    log("Starting 5-fold stratified CV with per-fold feature computation...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    classifier_templates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED,
        ),
    }

    all_fold_results = {name: [] for name in classifier_templates}
    all_fold_y_true = {name: [] for name in classifier_templates}
    all_fold_y_pred = {name: [] for name in classifier_templates}
    all_fold_y_proba = {name: [] for name in classifier_templates}

    all_fold_features_val = []
    all_fold_labels_val = []
    all_fold_categories_val = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        fold_t0 = time.time()
        log(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        log(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        train_texts_fold = [texts[i] for i in train_idx]
        train_embeddings_fold = embeddings[train_idx]
        train_labels_fold = labels[train_idx]
        train_categories_fold = categories[train_idx]

        val_texts_fold = [texts[i] for i in val_idx]
        val_embeddings_fold = embeddings[val_idx]
        val_labels_fold = labels[val_idx]
        val_categories_fold = categories[val_idx]

        log(f"  Building features with TRAINING-ONLY statistics...")
        train_features = build_features_for_fold(
            train_texts_fold, train_embeddings_fold,
            train_texts_fold, train_embeddings_fold, train_categories_fold,
        )
        val_features = build_features_for_fold(
            val_texts_fold, val_embeddings_fold,
            train_texts_fold, train_embeddings_fold, train_categories_fold,
        )

        all_fold_features_val.append(val_features)
        all_fold_labels_val.append(val_labels_fold)
        all_fold_categories_val.append(val_categories_fold)

        log(f"  Feature shapes: train={train_features.shape}, val={val_features.shape}")

        for clf_name, clf_template in classifier_templates.items():
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(train_features)
            X_val_s = scaler.transform(val_features)

            from copy import deepcopy
            clf = deepcopy(clf_template)
            clf.fit(X_train_s, train_labels_fold)

            y_pred = clf.predict(X_val_s)
            y_proba = clf.predict_proba(X_val_s)[:, 1] if hasattr(clf, "predict_proba") else None

            fold_m = compute_metrics(val_labels_fold, y_pred, y_proba)
            fold_m["fold"] = fold
            fold_m["train_size"] = len(train_idx)
            fold_m["val_size"] = len(val_idx)

            all_fold_results[clf_name].append(fold_m)
            all_fold_y_true[clf_name].append(val_labels_fold)
            all_fold_y_pred[clf_name].append(y_pred)
            if y_proba is not None:
                all_fold_y_proba[clf_name].append(y_proba)

            log(f"  {clf_name}: BA={fold_m['balanced_accuracy']:.4f} "
                f"F1={fold_m['f1']:.4f} FNR={fold_m['fnr']:.4f}")

        fold_elapsed = time.time() - fold_t0
        log(f"  Fold {fold + 1} completed in {fold_elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 5d. Aggregate classifier results
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  CLASSIFIER RESULTS (aggregated across 5 folds)")
    log("=" * 70)

    classifier_summary = {}
    for clf_name in classifier_templates:
        fold_metrics = all_fold_results[clf_name]
        metric_keys = [k for k in fold_metrics[0] if k not in ("fold", "train_size", "val_size")]

        agg = {}
        for key in metric_keys:
            vals = [fm[key] for fm in fold_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

        y_true_all = np.concatenate(all_fold_y_true[clf_name])
        y_pred_all = np.concatenate(all_fold_y_pred[clf_name])
        y_proba_all = (
            np.concatenate(all_fold_y_proba[clf_name])
            if all_fold_y_proba[clf_name] else None
        )

        ci_low, ci_high = bootstrap_ci(y_true_all, y_pred_all, n_bootstrap=1000)
        agg["balanced_accuracy_ci_95"] = [ci_low, ci_high]

        if y_proba_all is not None:
            from sklearn.metrics import roc_auc_score
            agg["auc_roc_pooled"] = float(roc_auc_score(y_true_all, y_proba_all))

        classifier_summary[clf_name] = agg
        log(f"  {clf_name}: BA={agg['balanced_accuracy_mean']:.4f} "
            f"CI=[{ci_low:.4f}, {ci_high:.4f}] "
            f"F1={agg['f1_mean']:.4f} AUC={agg.get('auc_roc_pooled', 0):.4f}")

    best_clf_name = max(classifier_summary, key=lambda k: classifier_summary[k]["balanced_accuracy_mean"])
    best_ba = classifier_summary[best_clf_name]["balanced_accuracy_mean"]
    log(f"\n  Best classifier: {best_clf_name} (BA={best_ba:.4f})")

    # -----------------------------------------------------------------------
    # 5e. Threshold optimization for best classifier
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log(f"  THRESHOLD OPTIMIZATION for {best_clf_name}")
    log("=" * 70)

    y_true_best = np.concatenate(all_fold_y_true[best_clf_name])
    y_proba_best = np.concatenate(all_fold_y_proba[best_clf_name])

    threshold_results = []
    for threshold_10x in range(30, 71):
        threshold = threshold_10x / 100.0
        y_pred_t = (y_proba_best >= threshold).astype(int)
        m = compute_metrics(y_true_best, y_pred_t, y_proba_best)
        threshold_results.append({
            "threshold": threshold,
            "balanced_accuracy": m["balanced_accuracy"],
            "f1": m["f1"],
            "fnr": m["fnr"],
            "fpr": m["fpr"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    feasible = [r for r in threshold_results if r["fnr"] <= 0.05]
    if feasible:
        optimal_threshold_entry = max(feasible, key=lambda r: r["balanced_accuracy"])
    else:
        optimal_threshold_entry = min(threshold_results, key=lambda r: r["fnr"])

    optimal_threshold = optimal_threshold_entry["threshold"]
    log(f"  Optimal threshold: {optimal_threshold:.2f}")
    log(f"  BA={optimal_threshold_entry['balanced_accuracy']:.4f} "
        f"FNR={optimal_threshold_entry['fnr']:.4f} "
        f"FPR={optimal_threshold_entry['fpr']:.4f}")

    y_pred_optimal = (y_proba_best >= optimal_threshold).astype(int)
    optimal_metrics = compute_metrics(y_true_best, y_pred_optimal, y_proba_best)
    ci_low_opt, ci_high_opt = bootstrap_ci(y_true_best, y_pred_optimal, n_bootstrap=1000)
    optimal_metrics["balanced_accuracy_ci_95"] = [ci_low_opt, ci_high_opt]
    optimal_metrics["threshold"] = optimal_threshold

    log(f"  Optimal model: BA={optimal_metrics['balanced_accuracy']:.4f} "
        f"CI=[{ci_low_opt:.4f}, {ci_high_opt:.4f}] "
        f"FNR={optimal_metrics['fnr']:.4f}")

    # Pareto frontier
    pareto = []
    seen_ba = set()
    for r in sorted(threshold_results, key=lambda x: x["fnr"]):
        ba_r = round(r["balanced_accuracy"], 4)
        if ba_r not in seen_ba:
            pareto.append(r)
            seen_ba.add(ba_r)

    # -----------------------------------------------------------------------
    # 5f. Per-category accuracy (best classifier, default threshold)
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  PER-CATEGORY ACCURACY (best classifier)")
    log("=" * 70)

    y_pred_best = np.concatenate(all_fold_y_pred[best_clf_name])
    cats_best = np.concatenate(all_fold_categories_val)

    per_category = {}
    for cat in np.unique(cats_best):
        mask = cats_best == cat
        if mask.sum() == 0:
            continue
        yt = y_true_best[mask]
        yp = y_pred_best[mask]
        correct = (yt == yp).sum()
        acc = correct / mask.sum()
        cat_mal = int((yt == 1).sum())
        cat_ben = int((yt == 0).sum())
        per_category[cat] = {
            "accuracy": float(acc),
            "n_samples": int(mask.sum()),
            "n_malicious": cat_mal,
            "n_benign": cat_ben,
        }
        log(f"  {cat}: acc={acc:.4f} n={mask.sum()} (mal={cat_mal}, ben={cat_ben})")

    # -----------------------------------------------------------------------
    # 5g. Ablation studies
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  ABLATION STUDIES (drop each signal, retrain best classifier)")
    log("=" * 70)

    from copy import deepcopy as _deepcopy

    ablation_results = {}
    for exclude_idx, exclude_name in enumerate(FEATURE_NAMES):
        mask = np.ones(len(FEATURE_NAMES), dtype=bool)
        mask[exclude_idx] = False

        fold_bas = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
            train_texts_fold = [texts[i] for i in train_idx]
            train_embeddings_fold = embeddings[train_idx]
            train_categories_fold = categories[train_idx]
            val_texts_fold = [texts[i] for i in val_idx]
            val_embeddings_fold = embeddings[val_idx]

            train_features_full = build_features_for_fold(
                train_texts_fold, train_embeddings_fold,
                train_texts_fold, train_embeddings_fold, train_categories_fold,
            )
            val_features_full = build_features_for_fold(
                val_texts_fold, val_embeddings_fold,
                train_texts_fold, train_embeddings_fold, train_categories_fold,
            )

            X_train_abl = train_features_full[:, mask]
            X_val_abl = val_features_full[:, mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_abl)
            X_val_s = scaler.transform(X_val_abl)

            clf_template = classifier_templates[best_clf_name]
            clf = _deepcopy(clf_template)
            clf.fit(X_train_s, labels[train_idx])
            y_pred_abl = clf.predict(X_val_s)
            fold_bas.append(float(balanced_accuracy_score(labels[val_idx], y_pred_abl)))

        mean_ba = float(np.mean(fold_bas))
        delta = best_ba - mean_ba
        ablation_results[exclude_name] = {
            "mean_balanced_accuracy": mean_ba,
            "std_balanced_accuracy": float(np.std(fold_bas)),
            "delta": delta,
            "delta_pct": float(delta * 100),
        }
        log(f"  Drop {exclude_name}: BA={mean_ba:.4f} (delta={delta:+.4f}, {delta * 100:+.2f}%)")

    # -----------------------------------------------------------------------
    # 5h. Feature importance from best model (trained on full data)
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  FEATURE IMPORTANCE")
    log("=" * 70)

    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(
        np.concatenate(all_fold_features_val)
    )
    y_full = np.concatenate(all_fold_labels_val)

    clf_final = _deepcopy(classifier_templates[best_clf_name])
    clf_final.fit(X_full_scaled, y_full)

    if hasattr(clf_final, "feature_importances_"):
        importances = clf_final.feature_importances_
    elif hasattr(clf_final, "coef_"):
        importances = np.abs(clf_final.coef_[0])
    else:
        importances = np.zeros(len(FEATURE_NAMES))

    importances_sum = importances.sum()
    importances_norm = importances / importances_sum if importances_sum > 0 else importances
    feature_importance = {
        name: float(imp) for name, imp in zip(FEATURE_NAMES, importances_norm)
    }
    for name, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        log(f"  {name}: {imp:.4f}")

    # -----------------------------------------------------------------------
    # 5i. Confusion matrix for best model (pooled)
    # -----------------------------------------------------------------------
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_best, y_pred_best)
    cm_dict = {
        "true_benign_pred_benign": int(cm[0, 0]),
        "true_benign_pred_malicious": int(cm[0, 1]),
        "true_malicious_pred_benign": int(cm[1, 0]),
        "true_malicious_pred_malicious": int(cm[1, 1]),
    }
    log(f"\n  Confusion matrix (best model, default threshold):")
    log(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    log(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # -----------------------------------------------------------------------
    # 5j. Signal correlations
    # -----------------------------------------------------------------------
    all_features_pooled = np.concatenate(all_fold_features_val)
    corr_matrix = np.corrcoef(all_features_pooled.T)
    signal_correlations = {}
    for i, name_i in enumerate(FEATURE_NAMES):
        signal_correlations[name_i] = {}
        for j, name_j in enumerate(FEATURE_NAMES):
            signal_correlations[name_i][name_j] = float(corr_matrix[i, j])

    # -----------------------------------------------------------------------
    # 5k. Signal distributions (benign vs malicious)
    # -----------------------------------------------------------------------
    y_full_all = np.concatenate(all_fold_labels_val)
    signal_distributions = {}
    for i, name in enumerate(FEATURE_NAMES):
        benign_vals = all_features_pooled[y_full_all == 0, i]
        malicious_vals = all_features_pooled[y_full_all == 1, i]
        signal_distributions[name] = {
            "benign": {
                "mean": float(np.mean(benign_vals)),
                "std": float(np.std(benign_vals)),
                "median": float(np.median(benign_vals)),
                "min": float(np.min(benign_vals)),
                "max": float(np.max(benign_vals)),
            },
            "malicious": {
                "mean": float(np.mean(malicious_vals)),
                "std": float(np.std(malicious_vals)),
                "median": float(np.median(malicious_vals)),
                "min": float(np.min(malicious_vals)),
                "max": float(np.max(malicious_vals)),
            },
        }

    # -----------------------------------------------------------------------
    # 5l. Feature statistics
    # -----------------------------------------------------------------------
    feature_stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = all_features_pooled[:, i]
        feature_stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    # -----------------------------------------------------------------------
    # 5m. Build and save report
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  BUILDING REPORT")
    log("=" * 70)

    leakage_free_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "evaluate_leakage_free.py",
            "description": "Leakage-free evaluation of Guardrailer multi-signal security system",
            "n_samples": len(sample_df),
            "n_malicious": n_mal,
            "n_benign": n_ben,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "embedding_model": EMBEDDING_MODEL,
            "dataset_path": PARQUET_PATH,
            "dataset_full_size": len(df),
            "n_folds": N_FOLDS,
            "random_seed": SEED,
            "leakage_mitigation": {
                "centroids": "computed per-fold from training data only",
                "idf": "computed per-fold from training document frequencies only",
                "avg_text_length": "computed per-fold from training texts only",
            },
        },
        "classifier_results": classifier_summary,
        "best_classifier": best_clf_name,
        "best_balanced_accuracy_default_threshold": best_ba,
        "target_met": best_ba >= 0.90,
        "threshold_optimization": {
            "optimal_threshold": optimal_threshold,
            "optimal_metrics": optimal_metrics,
            "pareto_frontier": pareto,
            "all_thresholds": threshold_results,
        },
        "per_category_accuracy": per_category,
        "ablation": ablation_results,
        "ablation_summary": {
            "most_important_signal": max(ablation_results, key=lambda k: ablation_results[k]["delta"]),
            "least_important_signal": min(ablation_results, key=lambda k: ablation_results[k]["delta"]),
        },
        "feature_importance": feature_importance,
        "confusion_matrix": cm_dict,
        "signal_correlations": signal_correlations,
        "signal_distributions": signal_distributions,
        "feature_statistics": feature_stats,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(leakage_free_report, f, indent=2, default=str)

    log(f"  Report saved to {REPORT_PATH}")

    # -----------------------------------------------------------------------
    # 5n. Final summary
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  FINAL SUMMARY")
    log("=" * 70)
    log(f"  Dataset: {len(sample_df)} samples (50/50 stratified from {len(df)} full)")
    log(f"  Signals: {len(FEATURE_NAMES)} leakage-free features")
    log(f"  CV: {N_FOLDS}-fold stratified, per-fold centroid/IDF/length computation")
    log(f"")
    log(f"  Classifiers:")
    for clf_name in classifier_summary:
        s = classifier_summary[clf_name]
        log(f"    {clf_name}: BA={s['balanced_accuracy_mean']:.4f} "
            f"CI=[{s['balanced_accuracy_ci_95'][0]:.4f}, {s['balanced_accuracy_ci_95'][1]:.4f}] "
            f"F1={s['f1_mean']:.4f} AUC={s.get('auc_roc_pooled', 0):.4f}")
    log(f"")
    log(f"  Best classifier: {best_clf_name}")
    log(f"  Default threshold (0.5) BA: {best_ba:.4f}")
    log(f"  Optimized threshold ({optimal_threshold:.2f}) BA: {optimal_metrics['balanced_accuracy']:.4f} "
        f"FNR: {optimal_metrics['fnr']:.4f}")
    log(f"  Target (BA >= 0.90): {'MET' if best_ba >= 0.90 else 'NOT MET'}")
    log(f"")
    log(f"  Most important signal (ablation): {max(ablation_results, key=lambda k: ablation_results[k]['delta'])}")
    log(f"  Least important signal (ablation): {min(ablation_results, key=lambda k: ablation_results[k]['delta'])}")
    log("=" * 70)


if __name__ == "__main__":
    main()
