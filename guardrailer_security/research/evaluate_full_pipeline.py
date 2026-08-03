"""
evaluate_full_pipeline.py
Full Guardrailer scoring pipeline evaluation with all 10 real signals.

Evaluates balanced accuracy >= 0.90 using stratified sampling, multiple classifiers,
5-fold stratified CV, bootstrap CIs, ablation studies, and visualization.
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/prashanna/Documents/Guardrailer")
sys.path.insert(0, "/home/prashanna/Documents/Guardrailer/guardrailer_security")

import numpy as np
import pandas as pd
import math
from collections import Counter
from pathlib import Path

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("/home/prashanna/Documents/Guardrailer/evaluation_results")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = "/home/prashanna/Documents/Guardrailer/guardrailer_security/unified_security_dataset.parquet"
CORPUS_META_PATH = "/home/prashanna/Documents/Guardrailer/guardrailer_security/corpus_meta.json"
LOGISTIC_MODEL_PATH = "/home/prashanna/Documents/Guardrailer/guardrailer_security/models/logistic_weights.json"
REPORT_PATH = RESULTS_DIR / "full_pipeline_report.json"

N_SAMPLES = 4000
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "guardrailer_security_enhanced"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64

FEATURE_NAMES = [
    "dense", "sparse_idf", "centroid", "cross_encoder",
    "perplexity", "entropy", "token_frequency", "ngram_overlap",
    "uniqueness", "length_norm",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# 1. Load corpus metadata
# ===========================================================================

log("Loading corpus metadata...")
with open(CORPUS_META_PATH) as f:
    corpus_meta = json.load(f)

keyword_idf = corpus_meta.get("keyword_idf", {})
N_docs = corpus_meta.get("total_documents", 693456)
avg_text_length = corpus_meta.get("avg_text_length", 130.0)
avg_doc_length = corpus_meta.get("avg_doc_length", 20.5)
category_centroids_raw = corpus_meta.get("category_centroids", {})

category_centroids = {}
for cat, vec in category_centroids_raw.items():
    category_centroids[cat] = np.array(vec, dtype=np.float32)

log(f"  Loaded: {len(keyword_idf)} IDF entries, {len(category_centroids)} centroids, N={N_docs}")


# ===========================================================================
# 2. Load sparse keywords from constants
# ===========================================================================

from constants import SPARSE_KEYWORDS

log(f"  Loaded {len(SPARSE_KEYWORDS)} sparse keywords")


# ===========================================================================
# 3. Score computation functions
# ===========================================================================

def compute_text_idf_score(text, meta=None):
    lower = text.lower()
    total_idf = 0.0
    for kw in SPARSE_KEYWORDS:
        if kw in lower:
            idf = keyword_idf.get(kw, math.log(N_docs / 2.0))
            total_idf += idf
    max_possible = sum(keyword_idf.values()) if keyword_idf else len(SPARSE_KEYWORDS)
    if max_possible > 0:
        return total_idf / max_possible
    return 0.0


def compute_centroid_score(embedding):
    q_norm = np.linalg.norm(embedding)
    if q_norm < 1e-8:
        return 0.0
    best = 0.0
    for cat, centroid in category_centroids.items():
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-8:
            continue
        sim = float(np.dot(embedding, centroid) / (q_norm * c_norm))
        if sim > best:
            best = sim
    return best


def compute_length_normalization(text_length):
    return math.log1p(text_length) / math.log1p(avg_text_length)


def compute_perplexity_score(text):
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


def compute_entropy_score(text):
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


def compute_token_frequency_score(text):
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
}


def compute_ngram_overlap_score(text):
    text_lower = text.lower()
    words = text_lower.split()
    if len(words) < 2:
        return 0.0
    bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    bigram_matches = sum(1 for bg in bigrams if bg in ATTACK_NGRAMS)
    trigram_matches = sum(1 for tg in trigrams if tg in ATTACK_NGRAMS)
    total_ngrams = len(bigrams) + len(trigrams)
    if total_ngrams == 0:
        return 0.0
    match_score = (bigram_matches * 1.0 + trigram_matches * 1.5) / total_ngrams
    substring_matches = sum(1 for phrase in ATTACK_NGRAMS if phrase in text_lower)
    substring_score = min(1.0, substring_matches * 0.2)
    return max(0.0, min(1.0, 0.5 * match_score + 0.5 * substring_score))


# ===========================================================================
# 4. Load dataset and stratified sample
# ===========================================================================

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

log(f"  Sampled {len(sample_df)} rows ({sample_df['is_malicious_int'].sum()} malicious, "
    f"{(sample_df['is_malicious_int'] == 0).sum()} benign)")
log(f"  Category distribution:\n{sample_df['attack_category'].value_counts().to_string()}")


# ===========================================================================
# 5. Load embedding model and Qdrant
# ===========================================================================

log("Loading sentence-transformers model...")
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer(EMBEDDING_MODEL)
log(f"  Model loaded: {EMBEDDING_MODEL}")

log("Connecting to Qdrant...")
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

qclient = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
col_info = qclient.get_collection(COLLECTION_NAME)
log(f"  Qdrant connected: {col_info.points_count} points")


# ===========================================================================
# 6. Compute all 10 signals for each sample
# ===========================================================================

log("Computing embeddings...")
texts = sample_df["prompt_text"].tolist()
labels = sample_df["is_malicious_int"].values
n_samples = len(texts)

embeddings = st_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
log(f"  Embeddings shape: {embeddings.shape}")

log("Computing Qdrant queries + signal features...")
features = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float32)

QDRANT_BATCH = 32
for batch_start in range(0, n_samples, QDRANT_BATCH):
    batch_end = min(batch_start + QDRANT_BATCH, n_samples)
    batch_embeddings = embeddings[batch_start:batch_end]
    batch_texts = texts[batch_start:batch_end]

    for i in range(batch_end - batch_start):
        idx = batch_start + i
        text = batch_texts[i]
        emb = batch_embeddings[i]

        try:
            results = qclient.query_points(
                COLLECTION_NAME,
                query=emb.tolist(),
                using="dense",
                limit=1,
                with_payload=True,
            )
            if results.points:
                top = results.points[0]
                dense_score = max(0.0, min(1.0, top.score))
                payload = top.payload if top.payload else {}
                cross_encoder = payload.get("cross_encoder_score", 0.5)
                uniqueness = payload.get("uniqueness", 0.5)
            else:
                dense_score = 0.0
                cross_encoder = 0.5
                uniqueness = 0.5
        except Exception as e:
            dense_score = 0.0
            cross_encoder = 0.5
            uniqueness = 0.5

        sparse_idf = compute_text_idf_score(text)
        centroid = compute_centroid_score(emb)
        perplexity = compute_perplexity_score(text)
        entropy = compute_entropy_score(text)
        token_freq = compute_token_frequency_score(text)
        ngram = compute_ngram_overlap_score(text)
        length_norm = compute_length_normalization(len(text))

        features[idx, 0] = dense_score
        features[idx, 1] = sparse_idf
        features[idx, 2] = centroid
        features[idx, 3] = cross_encoder
        features[idx, 4] = perplexity
        features[idx, 5] = entropy
        features[idx, 6] = token_freq
        features[idx, 7] = ngram
        features[idx, 8] = uniqueness
        features[idx, 9] = length_norm

    if (batch_start // QDRANT_BATCH) % 50 == 0:
        log(f"  Processed {batch_end}/{n_samples} samples...")

log(f"  Feature matrix shape: {features.shape}")

feature_stats = {}
for i, name in enumerate(FEATURE_NAMES):
    col = features[:, i]
    feature_stats[name] = {
        "mean": float(np.mean(col)),
        "std": float(np.std(col)),
        "min": float(np.min(col)),
        "max": float(np.max(col)),
    }
log(f"  Feature stats computed")


# ===========================================================================
# 7. Train classifiers and evaluate
# ===========================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    log("  XGBoost available")
except ImportError:
    log("  XGBoost not available, skipping")


def compute_metrics(y_true, y_pred, y_proba=None):
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
        "balanced_accuracy": ba,
        "f1": f1,
        "mcc": mcc,
        "auc_roc": auc,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(SEED)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1 - alpha) * 100))


def get_classifiers():
    clfs = {}
    clfs["LogisticRegression"] = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED, C=1.0
    )
    clfs["RandomForest"] = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        random_state=SEED, n_jobs=-1
    )
    clfs["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=SEED
    )
    if XGB_AVAILABLE:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(labels == 0).sum() / max((labels == 1).sum(), 1),
            eval_metric="logloss", random_state=SEED, n_jobs=-1,
            verbosity=0,
        )
    return clfs


log("Training classifiers with 5-fold stratified CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

classifiers = get_classifiers()
all_results = {}

for clf_name, clf_template in classifiers.items():
    log(f"  Evaluating {clf_name}...")
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = type(clf_template)(**clf_template.get_params())
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, "predict_proba") else None

        fold_m = compute_metrics(y_test, y_pred, y_proba)
        fold_metrics.append(fold_m)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            all_y_proba.extend(y_proba)

    avg_metrics = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics]
        avg_metrics[key] = float(np.mean(vals))
        avg_metrics[f"{key}_std"] = float(np.std(vals))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba) if len(all_y_proba) > 0 else None

    ci_low, ci_high = bootstrap_ci(all_y_true, all_y_pred, n_bootstrap=1000)
    avg_metrics["balanced_accuracy_ci_low"] = ci_low
    avg_metrics["balanced_accuracy_ci_high"] = ci_high

    all_results[clf_name] = avg_metrics
    log(f"    {clf_name}: BA={avg_metrics['balanced_accuracy']:.4f} "
        f"CI=[{ci_low:.4f}, {ci_high:.4f}] F1={avg_metrics['f1']:.4f} "
        f"AUC={avg_metrics['auc_roc']:.4f} MCC={avg_metrics['mcc']:.4f}")


# ===========================================================================
# 8. Existing logistic model baseline
# ===========================================================================

log("Loading existing trained logistic model as baseline...")
with open(LOGISTIC_MODEL_PATH) as f:
    lr_weights = json.load(f)

coef = np.array(lr_weights["coef"])
intercept = np.array(lr_weights["intercept"])
lr_scaler_mean = np.array(lr_weights["scaler_mean"])
lr_scaler_scale = np.array(lr_weights["scaler_scale"])

X_lr = (features - lr_scaler_mean) / lr_scaler_scale
logits = X_lr @ coef.T + intercept
lr_proba = 1.0 / (1.0 + np.exp(-logits))
if lr_proba.ndim > 1:
    lr_proba = lr_proba[:, 0]
lr_pred = (lr_proba >= 0.5).astype(int)
baseline_metrics = compute_metrics(labels, lr_pred, lr_proba)
ci_low, ci_high = bootstrap_ci(labels, lr_pred, n_bootstrap=1000)
baseline_metrics["balanced_accuracy_ci_low"] = ci_low
baseline_metrics["balanced_accuracy_ci_high"] = ci_high
all_results["PretrainedLogisticBaseline"] = baseline_metrics
log(f"    PretrainedLR: BA={baseline_metrics['balanced_accuracy']:.4f} "
    f"CI=[{ci_low:.4f}, {ci_high:.4f}]")


# ===========================================================================
# 9. Ablation studies
# ===========================================================================

log("Running ablation studies...")
ablation_results = {}
best_clf_name = max(all_results, key=lambda k: all_results[k]["balanced_accuracy"])
log(f"  Best classifier for ablation: {best_clf_name}")

for exclude_idx, exclude_name in enumerate(FEATURE_NAMES):
    mask = np.ones(len(FEATURE_NAMES), dtype=bool)
    mask[exclude_idx] = False
    X_abl = features[:, mask]

    fold_ba = []
    for train_idx, test_idx in skf.split(X_abl, labels):
        X_train, X_test = X_abl[train_idx], X_abl[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if best_clf_name == "XGBoost" and XGB_AVAILABLE:
            clf = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0,
            )
        elif best_clf_name == "RandomForest":
            clf = RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
            )
        elif best_clf_name == "GradientBoosting":
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=SEED
            )
        else:
            clf = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED
            )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        fold_ba.append(balanced_accuracy_score(y_test, y_pred))

    mean_ba = float(np.mean(fold_ba))
    delta = all_results[best_clf_name]["balanced_accuracy"] - mean_ba
    ablation_results[exclude_name] = {
        "mean_balanced_accuracy": mean_ba,
        "delta": delta,
        "delta_pct": float(delta * 100),
    }
    log(f"    Drop {exclude_name}: BA={mean_ba:.4f} (delta={delta:+.4f})")


# ===========================================================================
# 10. Feature importance from best model
# ===========================================================================

log("Extracting feature importance from best model...")
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(features)

if best_clf_name == "XGBoost" and XGB_AVAILABLE:
    best_clf_final = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(labels == 0).sum() / max((labels == 1).sum(), 1),
        eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0,
    )
elif best_clf_name == "RandomForest":
    best_clf_final = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
elif best_clf_name == "GradientBoosting":
    best_clf_final = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, random_state=SEED
    )
else:
    best_clf_final = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED
    )

best_clf_final.fit(X_full_scaled, labels)

if hasattr(best_clf_final, "feature_importances_"):
    importances = best_clf_final.feature_importances_
elif hasattr(best_clf_final, "coef_"):
    importances = np.abs(best_clf_final.coef_[0])
else:
    importances = np.zeros(len(FEATURE_NAMES))

importances_norm = importances / importances.sum() if importances.sum() > 0 else importances
feature_importance = {
    name: float(imp) for name, imp in zip(FEATURE_NAMES, importances_norm)
}
log(f"  Feature importances: { {k: round(v, 4) for k, v in feature_importance.items()} }")


# ===========================================================================
# 11. Visualizations
# ===========================================================================

log("Generating visualization plots...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Plot 1: Classifier comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
clf_names = list(all_results.keys())
ba_scores = [all_results[c]["balanced_accuracy"] for c in clf_names]
ba_lows = [all_results[c]["balanced_accuracy_ci_low"] for c in clf_names]
ba_highs = [all_results[c]["balanced_accuracy_ci_high"] for c in clf_names]
errors_low = [b - l for b, l in zip(ba_scores, ba_lows)]
errors_high = [h - b for b, h in zip(ba_scores, ba_highs)]

colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"][: len(clf_names)]
bars = ax.barh(clf_names, ba_scores, xerr=[errors_low, errors_high], color=colors,
               edgecolor="black", linewidth=0.5, capsize=5)
ax.axvline(x=0.90, color="red", linestyle="--", linewidth=1.5, label="Target: 0.90")
ax.set_xlabel("Balanced Accuracy", fontsize=12)
ax.set_title("Classifier Comparison (5-fold Stratified CV)", fontsize=14)
ax.set_xlim(0.5, 1.0)
for bar, score in zip(bars, ba_scores):
    ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.4f}",
            va="center", fontsize=10, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "classifier_comparison.png", dpi=150)
plt.close()

# --- Plot 2: Ablation study ---
fig, ax = plt.subplots(figsize=(12, 6))
abl_names = list(ablation_results.keys())
abl_deltas = [ablation_results[a]["delta_pct"] for a in abl_names]
abl_bas = [ablation_results[a]["mean_balanced_accuracy"] for a in abl_names]

sorted_idx = np.argsort(abl_deltas)[::-1]
abl_names_s = [abl_names[i] for i in sorted_idx]
abl_deltas_s = [abl_deltas[i] for i in sorted_idx]
abl_bas_s = [abl_bas[i] for i in sorted_idx]

bar_colors = ["#F44336" if d > 0.5 else "#4CAF50" for d in abl_deltas_s]
ax.barh(abl_names_s, abl_deltas_s, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Performance Delta (%)", fontsize=12)
ax.set_title(f"Ablation Study (drop each signal, classifier={best_clf_name})", fontsize=14)
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ablation_study.png", dpi=150)
plt.close()

# --- Plot 3: Feature importance ---
fig, ax = plt.subplots(figsize=(10, 6))
imp_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
imp_names = [x[0] for x in imp_sorted]
imp_vals = [x[1] for x in imp_sorted]
ax.barh(imp_names, imp_vals, color="#3F51B5", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Relative Importance", fontsize=12)
ax.set_title(f"Feature Importance ({best_clf_name})", fontsize=14)
for i, v in enumerate(imp_vals):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
plt.close()

# --- Plot 4: Signal distributions (benign vs malicious) ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, (name, ax) in enumerate(zip(FEATURE_NAMES, axes.flatten())):
    benign_vals = features[labels == 0, i]
    malicious_vals = features[labels == 1, i]
    ax.hist(benign_vals, bins=30, alpha=0.6, color="#4CAF50", label="Benign", density=True)
    ax.hist(malicious_vals, bins=30, alpha=0.6, color="#F44336", label="Malicious", density=True)
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8)
fig.suptitle("Signal Distributions: Benign vs Malicious", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "signal_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 5: Confusion matrix for best model ---
from sklearn.model_selection import cross_val_predict

best_clf_cm = type(best_clf_final)(**best_clf_final.get_params())
y_cross_pred = cross_val_predict(best_clf_cm, X_full_scaled, labels, cv=skf, method="predict")
cm = confusion_matrix(labels, y_cross_pred)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=[0, 1], yticks=[0, 1],
    xticklabels=["Benign", "Malicious"],
    yticklabels=["Benign", "Malicious"],
    ylabel="True Label", xlabel="Predicted Label",
    title=f"Confusion Matrix ({best_clf_name})",
)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# --- Plot 6: ROC curves for all classifiers ---
fig, ax = plt.subplots(figsize=(8, 8))
from sklearn.metrics import roc_curve

for idx, clf_name in enumerate(classifiers):
    fprs_tprs_aucs = []
    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        scaler_cv = StandardScaler()
        X_train_s = scaler_cv.fit_transform(X_train)
        X_test_s = scaler_cv.transform(X_test)
        clf_cv = type(classifiers[clf_name])(**classifiers[clf_name].get_params())
        clf_cv.fit(X_train_s, y_train)
        y_proba_cv = clf_cv.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba_cv)
        fprs_tprs_aucs.append((fpr, tpr, roc_auc_score(y_test, y_proba_cv)))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for fpr, tpr, auc_val in fprs_tprs_aucs:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_val)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=colors[idx % len(colors)],
            label=f"{clf_name} (AUC = {mean_auc:.3f} +/- {std_auc:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves (5-fold Stratified CV)", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
plt.close()

# --- Plot 7: Signal correlation heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = np.corrcoef(features.T)
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(range(len(FEATURE_NAMES)))
ax.set_yticks(range(len(FEATURE_NAMES)))
ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(FEATURE_NAMES, fontsize=10)
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
ax.set_title("Signal Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "signal_correlation.png", dpi=150)
plt.close()

log(f"  Saved 7 plots to {FIGURES_DIR}")


# ===========================================================================
# 12. Build comprehensive report
# ===========================================================================

log("Building report...")
best_clf_name_report = max(all_results, key=lambda k: all_results[k]["balanced_accuracy"])
best_ba = all_results[best_clf_name_report]["balanced_accuracy"]

report = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": n_samples,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "embedding_model": EMBEDDING_MODEL,
        "qdrant_collection": COLLECTION_NAME,
        "qdrant_points": col_info.points_count,
        "dataset_size": len(df),
        "random_seed": SEED,
    },
    "classifiers": all_results,
    "best_classifier": best_clf_name_report,
    "best_balanced_accuracy": best_ba,
    "target_met": best_ba >= 0.90,
    "ablation": ablation_results,
    "feature_importance": feature_importance,
    "feature_statistics": feature_stats,
    "signal_correlations": {
        FEATURE_NAMES[i]: {FEATURE_NAMES[j]: float(corr_matrix[i, j])
                           for j in range(len(FEATURE_NAMES))}
        for i in range(len(FEATURE_NAMES))
    },
    "ablation_summary": {
        "most_important_signal": max(ablation_results, key=lambda k: ablation_results[k]["delta"]),
        "least_important_signal": min(ablation_results, key=lambda k: ablation_results[k]["delta"]),
        "total_delta_if_all_dropped": sum(a["delta"] for a in ablation_results.values()),
    },
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

log(f"  Report saved to {REPORT_PATH}")

log("=" * 60)
log(f"  FINAL RESULT: {best_clf_name_report} -> Balanced Accuracy = {best_ba:.4f}")
log(f"  Target: >= 0.90 -> {'MET' if best_ba >= 0.90 else 'NOT MET'}")
log(f"  95% CI: [{all_results[best_clf_name_report]['balanced_accuracy_ci_low']:.4f}, "
    f"{all_results[best_clf_name_report]['balanced_accuracy_ci_high']:.4f}]")
log("=" * 60)
