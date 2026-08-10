"""
evaluate_prefilter.py
Re-evaluate the Phase 5 leakage-free pipeline with Tier-1 pre-filter integration.

Adds a fast keyword/pattern pre-filter before the 7-signal Random Forest to
catch short, simple attacks (Tier 1) that evade the RF due to weak signals.
Re-computes per-tier detection rates and generates updated figures.
"""

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/prashanna/Documents/Guardrailer")
sys.path.insert(0, "/home/prashanna/Documents/Guardrailer/guardrailer_security")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("/home/prashanna/Documents/Guardrailer/evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = "/home/prashanna/Documents/Guardrailer/guardrailer_security/unified_security_dataset.parquet"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64
N_SAMPLES = 10000
N_FOLDS = 5

FEATURE_NAMES = ["centroid", "sparse_idf", "perplexity", "entropy", "token_freq", "ngram", "length_norm"]

TIER_MAP = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def compute_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def build_features_for_fold(texts, embeddings, ref_texts, ref_embeddings, ref_categories):
    from guardrailer_security.research.evaluate_leakage_free import build_features_for_fold as _build
    return _build(texts, embeddings, ref_texts, ref_embeddings, ref_categories)


def main():
    from guardrailer_security.research.tier1_prefilter import tier1_prefilter, hybrid_predict

    log("Loading dataset...")
    df = pd.read_parquet(PARQUET_PATH)
    log(f"Full corpus: {len(df)} samples")

    df["is_malicious_int"] = df["is_malicious"].astype(int)

    np.random.seed(SEED)
    malicious_idx = np.where(df["is_malicious_int"] == 1)[0]
    benign_idx = np.where(df["is_malicious_int"] == 0)[0]
    n_per_class = N_SAMPLES // 2

    rng = np.random.RandomState(SEED)
    sel_mal = rng.choice(malicious_idx, size=min(n_per_class, len(malicious_idx)), replace=False)
    sel_ben = rng.choice(benign_idx, size=min(n_per_class, len(benign_idx)), replace=False)
    sel = np.sort(np.concatenate([sel_mal, sel_ben]))

    sample_df = df.iloc[sel].reset_index(drop=True)
    texts = sample_df["prompt_text"].tolist()
    labels = sample_df["is_malicious_int"].values
    categories = sample_df["attack_category"].values
    risk_levels = sample_df["risk_level"].values if "risk_level" in sample_df.columns else np.array(["none"] * len(sample_df))

    tiers = np.array([TIER_MAP.get(rl, 0) for rl in risk_levels])

    log(f"Selected {len(texts)} samples ({(labels == 1).sum()} malicious, {(labels == 0).sum()} benign)")

    log("Computing embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        log(f"Embedding computation failed ({e}), using cached embeddings")
        emb_path = RESULTS_DIR / "cached_embeddings.npy"
        if emb_path.exists():
            embeddings = np.load(str(emb_path))
        else:
            raise RuntimeError("No embeddings available")

    log(f"Embeddings shape: {embeddings.shape}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rf_template = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )

    all_fold_preds_rf = []
    all_fold_texts = []
    all_fold_labels = []
    all_fold_tiers = []
    all_fold_categories = []

    log("Running 5-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        log(f"  Fold {fold + 1}/{N_FOLDS}")

        train_texts_fold = [texts[i] for i in train_idx]
        train_embeddings_fold = embeddings[train_idx]
        train_categories_fold = categories[train_idx]
        val_texts_fold = [texts[i] for i in val_idx]
        val_embeddings_fold = embeddings[val_idx]

        train_features = build_features_for_fold(
            train_texts_fold, train_embeddings_fold,
            train_texts_fold, train_embeddings_fold, train_categories_fold,
        )
        val_features = build_features_for_fold(
            val_texts_fold, val_embeddings_fold,
            train_texts_fold, train_embeddings_fold, train_categories_fold,
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train_features)
        X_val_s = scaler.transform(val_features)

        from copy import deepcopy
        clf = deepcopy(rf_template)
        clf.fit(X_train_s, labels[train_idx])
        y_pred_rf = clf.predict(X_val_s)

        all_fold_preds_rf.extend(y_pred_rf.tolist())
        all_fold_texts.extend(val_texts_fold)
        all_fold_labels.extend(labels[val_idx].tolist())
        all_fold_tiers.extend(tiers[val_idx].tolist())
        all_fold_categories.extend(categories[val_idx].tolist())

    rf_preds = np.array(all_fold_preds_rf)
    all_labels = np.array(all_fold_labels)
    all_tiers = np.array(all_fold_tiers)
    all_categories = np.array(all_fold_categories)

    log("Running hybrid predict with Tier-1 pre-filter...")
    all_texts_for_pf = all_fold_texts
    rf_proba_placeholder = np.where(rf_preds == 1, 0.85, 0.15).tolist()

    hybrid_preds, hybrid_probas, per_sample_info = hybrid_predict(
        all_texts_for_pf, rf_preds.tolist(), rf_proba_placeholder, threshold=0.70,
    )
    hybrid_preds = np.array(hybrid_preds)

    n_overridden = sum(1 for info in per_sample_info if info.get("overridden", False))
    log(f"  Pre-filter overrode {n_overridden} RF predictions (benign -> malicious)")

    log("\nComputing metrics...")
    rf_ba = compute_balanced_accuracy(all_labels, rf_preds)
    hybrid_ba = compute_balanced_accuracy(all_labels, hybrid_preds)

    log(f"  RF-only Balanced Accuracy:       {rf_ba:.4f}")
    log(f"  Hybrid (RF + Pre-filter) BA:      {hybrid_ba:.4f}")
    log(f"  Improvement:                      {hybrid_ba - rf_ba:+.4f}")

    per_tier_rf = {}
    per_tier_hybrid = {}
    for tier in sorted(set(all_tiers)):
        mask = all_tiers == tier
        if mask.sum() > 0:
            per_tier_rf[str(tier)] = round(float((rf_preds[mask] == 1).mean()), 4)
            per_tier_hybrid[str(tier)] = round(float((hybrid_preds[mask] == 1).mean()), 4)

    log("\nPer-Tier Detection Rates:")
    log(f"  {'Tier':<8} {'RF-only':<12} {'Hybrid':<12} {'Delta':<10} {'Samples':<10}")
    for tier_str in sorted(per_tier_rf.keys()):
        n_tier = int((all_tiers == int(tier_str)).sum())
        delta = per_tier_hybrid[tier_str] - per_tier_rf[tier_str]
        log(f"  {tier_str:<8} {per_tier_rf[tier_str]:<12.4f} {per_tier_hybrid[tier_str]:<12.4f} {delta:+.4f}     {n_tier:<10}")

    per_cat_rf = {}
    per_cat_hybrid = {}
    for cat in sorted(set(all_categories)):
        mask = all_categories == cat
        if mask.sum() > 0:
            per_cat_rf[cat] = round(float((rf_preds[mask] == all_labels[mask]).mean()), 4)
            per_cat_hybrid[cat] = round(float((hybrid_preds[mask] == all_labels[mask]).mean()), 4)

    log("\nPer-Category Accuracy:")
    for cat in sorted(per_cat_rf.keys()):
        delta = per_cat_hybrid[cat] - per_cat_rf[cat]
        log(f"  {cat:<30} RF={per_cat_rf[cat]:.4f}  Hybrid={per_cat_hybrid[cat]:.4f}  {delta:+.4f}")

    cm = confusion_matrix(all_labels, hybrid_preds)
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    tn = int(cm[0, 0])
    fn = int(cm[1, 0])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    mcc = matthews_corrcoef(all_labels, hybrid_preds)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    tier1_mask = all_tiers == 1
    tier1_ba_rf = compute_balanced_accuracy(all_labels[tier1_mask], rf_preds[tier1_mask]) if tier1_mask.sum() > 0 else 0
    tier1_ba_hybrid = compute_balanced_accuracy(all_labels[tier1_mask], hybrid_preds[tier1_mask]) if tier1_mask.sum() > 0 else 0

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "description": "Phase 5 leakage-free evaluation with Tier-1 pre-filter",
            "n_samples": len(texts),
            "n_folds": N_FOLDS,
            "signals": 7,
            "feature_names": FEATURE_NAMES,
            "classifier": "RandomForest",
            "prefilter": {
                "type": "exact-phrase + regex pattern matching",
                "threshold": 0.70,
                "description": "Fast pre-filter for short, simple attacks (Tier 1) that evade RF",
            },
        },
        "results_rf_only": {
            "balanced_accuracy": round(rf_ba, 4),
            "per_tier_detection_rate": per_tier_rf,
            "per_category_accuracy": per_cat_rf,
        },
        "results_hybrid": {
            "balanced_accuracy": round(hybrid_ba, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "per_tier_detection_rate": per_tier_hybrid,
            "per_category_accuracy": per_cat_hybrid,
        },
        "tier1_pre_filter_analysis": {
            "n_overridden": n_overridden,
            "tier1_rf_ba": round(tier1_ba_rf, 4),
            "tier1_hybrid_ba": round(tier1_ba_hybrid, 4),
            "tier1_improvement": round(tier1_ba_hybrid - tier1_ba_rf, 4),
        },
        "success_criteria": {
            "tier1_detection_ge_85": per_tier_hybrid.get("1", 0) >= 0.85,
            "overall_ba_ge_90": hybrid_ba >= 0.90,
            "fpr_le_10": fpr <= 0.10,
        },
    }

    report_path = RESULTS_DIR / "prefilter_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nReport saved to {report_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        tier_labels_map = {0: "Tier 0\n(none)", 1: "Tier 1\n(low)", 2: "Tier 2\n(medium)",
                           3: "Tier 3\n(high)", 4: "Tier 4\n(critical)"}
        tier_keys = sorted(per_tier_hybrid.keys())
        tier_names = [tier_labels_map.get(int(k), f"Tier {k}") for k in tier_keys]
        rf_vals = [per_tier_rf[k] for k in tier_keys]
        hybrid_vals = [per_tier_hybrid[k] for k in tier_keys]
        n_tiers = [int((all_tiers == int(k)).sum()) for k in tier_keys]

        x_pos = np.arange(len(tier_keys))
        width = 0.35
        ax1.bar(x_pos - width / 2, rf_vals, width, label="RF-only", color="#90CAF9", edgecolor="white")
        ax1.bar(x_pos + width / 2, hybrid_vals, width, label="Hybrid (RF + Pre-filter)", color="#1565C0", edgecolor="white")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tier_names, fontsize=9)
        ax1.set_ylabel("Detection Rate")
        ax1.set_title("Per-Tier Detection Rate: RF-only vs Hybrid")
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.85, color="green", linestyle="--", alpha=0.5, label="85% target")
        for i, (rv, hv) in enumerate(zip(rf_vals, hybrid_vals)):
            ax1.text(i - width / 2, rv + 0.02, f"{rv:.3f}", ha="center", fontsize=8)
            ax1.text(i + width / 2, hv + 0.02, f"{hv:.3f}", ha="center", fontsize=8)
        ax1.legend(fontsize=8)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        cats = sorted(per_cat_hybrid.keys())
        cat_vals_rf = [per_cat_rf[c] for c in cats]
        cat_vals_hybrid = [per_cat_hybrid[c] for c in cats]
        colors_cat = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))
        bars = ax2.barh(cats, cat_vals_hybrid, color=colors_cat, height=0.5, edgecolor="white")
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Accuracy")
        ax2.set_title("Hybrid Per-Category Accuracy")
        for bar, val in zip(bars, cat_vals_hybrid):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax2.invert_yaxis()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure_05_per_tier_hybrid.png", bbox_inches="tight")
        plt.close(fig)
        log(f"Saved figure_05_per_tier_hybrid.png")
    except Exception as e:
        log(f"Figure generation failed: {e}")

    log("\n" + "=" * 60)
    log("  HYBRID EVALUATION SUMMARY")
    log("=" * 60)
    log(f"  RF-only BA:       {rf_ba:.4f}")
    log(f"  Hybrid BA:        {hybrid_ba:.4f}")
    log(f"  Tier 1 (RF):      {per_tier_rf.get('1', 0):.4f}")
    log(f"  Tier 1 (Hybrid):  {per_tier_hybrid.get('1', 0):.4f}")
    log(f"  Pre-filter overrides: {n_overridden}")
    log(f"  FPR:              {fpr:.4f}")
    log(f"  FNR:              {fnr:.4f}")
    log(f"  Tier 1 >= 85%:    {'MET' if per_tier_hybrid.get('1', 0) >= 0.85 else 'NOT MET'}")
    log(f"  Overall BA >= 90%:{'MET' if hybrid_ba >= 0.90 else 'NOT MET'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
