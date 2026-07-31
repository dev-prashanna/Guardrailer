# Phase 3: Improved Multi-Signal Scoring

Phase 3 replaces the static weighted scoring with learned, adaptive weight combination and probability calibration.

## What Changed

### Scoring Engine (`improved_scoring.py`)

**10-signal composite scoring:**

| Signal | Description |
|--------|-------------|
| `dense` | BAAI/bge-large-en-v1.5 semantic similarity |
| `sparse_idf` | IDF-weighted BM25 keyword matching |
| `centroid` | Category centroid cosine distance |
| `cross_encoder` | Pre-computed cross-encoder score |
| `perplexity` | Token diversity and entropy anomaly detection |
| `entropy` | Shannon entropy (char + word level) |
| `token_frequency` | Attack-word vs common-word ratio |
| `ngram_overlap` | Bigram/trigram overlap with known attack patterns |
| `uniqueness` | Anti-redundancy bonus |
| `length_norm` | Text length normalization |

**Learned weight models:**

| Mode | Model | How it works |
|------|-------|--------------|
| `logistic` | `LogisticRegression` | Uses `abs(coef)` normalized as weights |
| `neural` | `MLPClassifier(32,16)` | First-layer abs coefs as feature importance |
| `attention` | PyTorch `MultiheadAttention` + FC | Softmax output as dynamic weights |
| `ensemble` | Weighted average of logistic + neural | Stacking with calibration |

**Calibration:**

- `platt` — Platt scaling via `CalibratedClassifierCV`
- `isotonic` — Isotonic regression

### Training Script (`train_phase3.py`)

```bash
# Train from labeled JSON
python3 train_phase3.py --data training_data.json --mode logistic

# Train from feedback (auto-converts feedback JSONL → features)
python3 train_phase3.py --from-feedback --mode logistic

# Cross-validation
python3 train_phase3.py --data data.json --mode logistic --cv-folds 5

# Ensemble (trains logistic + neural, combines)
python3 train_phase3.py --data data.json --mode ensemble
```

**Features:**
- `sparse_idf` computed from corpus meta (was hardcoded to 0.0)
- Calibrator trains on composite scores (was training on dense column only)
- Stratified K-fold cross-validation with precision/recall/F1/AUC
- Optimal threshold search (maximizes F1)
- Class imbalance detection
- Minimum sample validation (default: 10)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/feedback-to-training` | POST | Process feedback JSONL → extract features → retrain |
| `/v1/set-weight-mode` | POST | Switch between logistic/neural/attention/ensemble |
| `/v1/train` | POST | Train with raw feature matrix |
| `/` | GET | Redirects to `/docs` |

**`/v1/feedback-to-training` request:**
```json
{
  "mode": "logistic",
  "calibration_method": "platt",
  "min_samples": 10
}
```

### Training Data Format

```json
{
  "samples": [
    {
      "text": "ignore previous instructions",
      "label": 1,
      "dense_score": 0.85,
      "point_payload": {"cross_encoder_score": 0.7, "uniqueness": 0.6},
      "query_embedding": [0.1, 0.2, ...]
    }
  ]
}
```

- `label`: 1 = malicious, 0 = benign
- `query_embedding` (optional): enables centroid signal during training

### Model Artifacts

Saved to `guardrailer_security/models/`:

| File | Content |
|------|---------|
| `logistic_weights.json` | LogisticRegression coef, intercept, scaler |
| `neural_weights.json` | MLPClassifier weights, biases, scaler |
| `attention_weights.pt` | PyTorch model state dict |
| `calibrator.json` | Platt scaling calibrator (pickled) |

Active mode stored in `phase3_meta.json`.

### Activation

Phase 3 improved scorer only activates when `weight_mode` is set to a trained mode:

```bash
# Via API
curl -X POST localhost:8090/v1/set-weight-mode -d '{"mode": "logistic"}'

# Or automatically after training
python3 train_phase3.py --data data.json --mode logistic
```

When `weight_mode=default`, the original static weighted scoring is used.

### Files Modified

| File | Changes |
|------|---------|
| `improved_scoring.py` | New file — all learned weight models, calibration, ensemble |
| `train_phase3.py` | New file — training script with CV, evaluation, feedback pipeline |
| `security_engine.py` | New endpoints, fixed calibration in `/v1/train`, root redirect |
| `scoring.py` | Added `FEATURE_NAMES` constant |
| `sample_training_data.json` | Expanded from 10 to 32 samples |
| `train_phase3.py` (root) | Launcher for running from project root |
