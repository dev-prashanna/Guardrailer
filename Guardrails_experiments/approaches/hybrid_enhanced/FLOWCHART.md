# Enhanced Hybrid Scorer — Training Flowchart

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: 722K labeled prompts                       │
│                    Split: 80% train (578K) / 20% test (144K)        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: PATTERN MATCHING (~0.01ms per prompt)                     │
│  ───────────────────────────────────────────────────────────────────│
│  • 43 regex attack patterns (injection, jailbreak, encoding)        │
│  • 54 sparse keywords (ignore, bypass, DAN, etc.)                   │
│  • Text anomaly detection (structural, positional)                  │
│  OUTPUT: 19 binary pattern signals                                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: TF-IDF SIMILARITY (~50s to fit)                           │
│  ───────────────────────────────────────────────────────────────────│
│  • TF-IDF vectorizer (50K features, sublinear TF, L2 norm)         │
│  • Computes malicious centroid from training data                   │
│  • IDF-weighted keyword scorer                                      │
│  OUTPUT: Cosine similarity to malicious cluster + keyword scores    │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: EMBEDDING SIMILARITY (all-MiniLM-L6-v2)                   │
│  ───────────────────────────────────────────────────────────────────│
│  • Model: all-MiniLM-L6-v2 (22M params, 384 dims, ~80MB)           │
│  • Encodes 578K texts → 578K × 384 embedding matrix                 │
│  • Computes centroids: malicious cluster centroid, benign centroid   │
│  • Features: centroid_distance_malicious, centroid_distance_benign, │
│    centroid_diff, embedding_norm                                     │
│  • Model unloaded after fit() → frees GPU for XGBoost               │
│  OUTPUT: 7 embedding features per prompt                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: COMBINED FEATURE MATRIX                                   │
│  ───────────────────────────────────────────────────────────────────│
│                                                                      │
│  ┌──────────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ 56 Handcrafted   │  │ 7 Embedding  │  │ 5,000 TF-IDF          │  │
│  │ Features         │  │ Features     │  │ Features               │  │
│  │                  │  │              │  │                        │  │
│  │ • char_count     │  │ • centroid_  │  │ • Sublinear TF         │  │
│  │ • word_count     │  │   distance_  │  │ • L2 normalized        │  │
│  │ • char_entropy   │  │   malicious  │  │ • 5000 max features    │  │
│  │ • leetspeak_     │  │ • centroid_  │  │                        │  │
│  │   ratio          │  │   distance_  │  │                        │  │
│  │ • unicode_       │  │   benign     │  │                        │  │
│  │   anomaly_ratio  │  │ • centroid_  │  │                        │  │
│  │ • attack_        │  │   diff       │  │                        │  │
│  │   keyword_count  │  │ • embedding_ │  │                        │  │
│  │ • structural_    │  │   norm       │  │                        │  │
│  │   hits           │  │ • embedding_ │  │                        │  │
│  │ • text_          │  │   dim        │  │                        │  │
│  │   perplexity     │  │              │  │                        │  │
│  │ • ... (56 total) │  │              │  │                        │  │
│  └──────────────────┘  └──────────────┘  └───────────────────────┘  │
│                                                                      │
│  Combined: sparse_hstack → 578K × 5,063 feature matrix (CSR)        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5: XGBOOST CLASSIFIER (GPU-accelerated)                      │
│  ───────────────────────────────────────────────────────────────────│
│  • n_estimators=300, max_depth=6, learning_rate=0.1                 │
│  • subsample=0.8, colsample_bytree=0.8                             │
│  • device=cuda, tree_method=hist                                    │
│  • eval_metric=logloss                                              │
│  • Trains on 578K × 5,063 sparse matrix                            │
│  OUTPUT: P(malicious) for each prompt                               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENSEMBLE SCORING                                                   │
│  ───────────────────────────────────────────────────────────────────│
│  score = 0.20 × pattern + 0.15 × similarity +                       │
│          0.25 × embedding + 0.40 × classifier                       │
│                                                                      │
│  Threat levels:                                                     │
│    critical: score ≥ 0.8                                            │
│    high:     score ≥ 0.6                                            │
│    medium:   score ≥ 0.4                                            │
│    low:      score ≥ 0.2                                            │
│    safe:     score < 0.2                                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EVALUATION ON TEST SET (144K prompts)                              │
│  ───────────────────────────────────────────────────────────────────│
│  Metrics: Accuracy, Precision, Recall, F1, MCC, AUC-ROC             │
│  Confusion matrix: TP, TN, FP, FN                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Timeline (Estimated)

| Step | Duration | Bottleneck |
|------|----------|------------|
| Load dataset | ~2s | I/O |
| TF-IDF similarity fit | ~50s | CPU |
| Keyword scorer fit | ~2s | CPU |
| **Embedding encode (578K)** | **~5-10 min** | **GPU (all-MiniLM-L6-v2)** |
| Handcrafted features | ~600s | CPU (batched) |
| TF-IDF vectorizer fit | ~50s | CPU |
| XGBoost GPU training | ~20s | GPU |
| Test set evaluation | ~30s | CPU |
| **Total** | **~15-20 min** | |

## Feature Count

| Source | Features |
|--------|----------|
| Handcrafted | 56 |
| Embedding | 7 |
| TF-IDF | 5,000 |
| **Total** | **5,063** |

## Model Artifacts

| File | Size | Description |
|------|------|-------------|
| classifier.joblib | ~0.5 MB | XGBoost model |
| tfidf.joblib | ~0.1 MB | TF-IDF vectorizer |
| similarity.joblib | ~2 MB | TF-IDF similarity scorer |
| keywords.joblib | ~4.5 MB | IDF keyword scorer |
| feature_names.joblib | ~1 KB | Feature list |
| **Total** | **~7 MB** | (+ ~80MB embedding model at inference) |
