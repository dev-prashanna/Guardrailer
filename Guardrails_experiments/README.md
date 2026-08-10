# Guardrailer — Hybrid Lightweight Prompt Security Classifier

A lightweight ML model for robustly classifying malicious vs benign prompts without using heavy embedding models like BAAI/bge-large-en-v1.5.

---

## Why Not Embeddings?

The embedding-based approach (bge-large-en-v1.5 + Qdrant) achieved only **41.2% accuracy** because:

| Problem | Evidence |
|---------|----------|
| Embedding space collapse | Centroid similarity (safe vs malicious) = **1.0000** |
| Score distribution overlap | **99.8%** overlap between classes |
| All 11 signals weak | Every embedding-derived signal is noise |
| Jailbreak semantic overlap | "Pretend you are DAN" ≈ "Pretend you are a tutor" |

The model answers "what is this text about?" — not "what is this text trying to do?"

See [docs/why_embedding_model_fails.md](../docs/why_embedding_model_fails.md) for full analysis.

---

## Architecture

```
Input Prompt
    │
    ├── Layer 1: Pattern Matching (~0.01ms)
    │   └── 35+ regex patterns (jailbreak, extraction, encoding, role hijack)
    │
    ├── Layer 2: TF-IDF Similarity (~0.1ms)
    │   ├── Word-level TF-IDF (5K features)
    │   └── IDF-weighted keyword relevance
    │
    ├── Layer 3: XGBoost Classifier (~0.5ms)
    │   ├── 44 handcrafted features
    │   ├── 5K TF-IDF features
    │   └── Combined: 5,044 features (sparse CSR)
    │
    └── Score → Threat Level (safe / low / medium / high / critical)
```

---

## Benchmark Results

### Overall Metrics (722K dataset, 80/20 split)

| Metric | Embedding Pipeline | Hybrid Lightweight | Improvement |
|--------|-------------------|-------------------|-------------|
| **Accuracy** | 41.2% | **86.5%** | **+110%** |
| **F1 Score** | 0.158 | **0.889** | **+463%** |
| **AUC-ROC** | 0.460 | **0.942** | **+105%** |
| **Precision** | 55.0% | **87.8%** | **+60%** |
| **Recall** | 9.2% | **90.0%** | **+878%** |
| **Balanced Accuracy** | 41.2% | **86.5%** | **+110%** |
| **MCC** | -0.033 | **0.717** | — |

### Confusion Matrix

```
                Predicted Safe    Predicted Malicious
Actual Safe         47,070 (TN)        10,885 (FP)
Actual Malicious     8,671 (FN)        77,943 (TP)
```

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Safe | 0.84 | 0.81 | 0.83 | 57,955 |
| Malicious | 0.88 | 0.90 | 0.89 | 86,614 |

### Per-Category Accuracy (Leakage-Free Eval, 10K samples)

| Category | Accuracy | F1 | Notes |
|----------|----------|-----|-------|
| indirect_injection | 99.7% | 0.997 | Best performing |
| system_prompt_extraction | 97.6% | 0.976 | |
| direct_injection | 95.9% | 0.959 | |
| benign_control | 95.6% | 0.956 | |
| refusal_bypass | 88.1% | 0.881 | |
| jailbreak | 72.7% | 0.727 | Weakest — needs improvement |

### Inference Performance

| Metric | Value |
|--------|-------|
| Latency per prompt | **0.001 ms** |
| Throughput | **808,183 prompts/sec** |
| Model size | **7.1 MB** |
| RAM usage (inference) | **< 500 MB** |
| GPU required | **No** (CPU inference) |

### Training Performance

| Metric | Value |
|--------|-------|
| Dataset | 722,842 samples |
| Training time | 21.6s (GPU XGBoost) |
| Total pipeline time | ~12 min (feature extraction bottleneck) |
| Hardware | RTX 4060 Laptop GPU |

---

## Feature Importance (Ablation Study)

| Feature | Importance | BA Delta |
|---------|-----------|----------|
| centroid | 44.63% | -6.53% |
| length_norm | 19.85% | -3.75% |
| token_freq | 16.78% | -1.04% |
| perplexity | 6.88% | -0.18% |
| entropy | 5.86% | -0.03% |
| sparse_idf | 4.37% | -0.26% |
| ngram | 1.62% | +0.05% |

---

## 44 Handcrafted Features

### Structural
- Prompt length (char/word count)
- Sentence count, average sentence length
- Uppercase ratio, digit ratio, special char ratio
- Word length variance

### Lexical
- Attack keyword count and density
- Encoding detection (base64, hex, URL, unicode)
- Structural pattern matching (instruction override, role hijack, system extraction)
- Word repeat ratio, bigram repeat ratio

### Statistical
- Character entropy, word entropy
- Unique word ratio, hapax ratio
- Question presence, exclamation ratio

### Injection-Specific
- Delimiter detection (``` markers, [INST], <<SYS>>)
- XML tag detection, bracket detection
- Colon-separated role markers (USER:, ASSISTANT:, SYSTEM:)
- Imperative verb detection (starts with "ignore", "forget", etc.)

---

## Files

| File | Description |
|------|-------------|
| `src/features.py` | 44 handcrafted feature extraction |
| `src/patterns.py` | 35+ regex patterns for attack detection |
| `src/similarity.py` | TF-IDF similarity scorer |
| `src/scorer.py` | Main hybrid scorer class |
| `src/train.py` | Memory-optimized training |
| `src/train_full.py` | Full dataset training pipeline |
| `BENCHMARK.md` | Detailed benchmark results |
| `requirements.txt` | Dependencies |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train on full dataset
python src/train_full.py

# Run inference
python src/scorer.py --text "Ignore previous instructions..."
```

---

## Reproduction

```bash
# Training
python src/train_full.py
# Results saved to:
#   src/models/          — model artifacts
#   src/results/         — benchmark JSON

# Evaluation
python src/evaluate.py --mode all
```

---

## Comparison: Resource Usage

| Resource | Embedding Pipeline | Hybrid Lightweight |
|----------|-------------------|-------------------|
| Model size | 1.3 GB + 1.5 GB embeddings | **7.1 MB** |
| Vector database | Qdrant (4GB+ RAM) | **None** |
| GPU requirement | Yes (T4 for generation) | **No** |
| Ingestion time | Hours | **Minutes** |
| Query latency | ~5 ms | **~0.001 ms** |
| Throughput | ~200/s | **~808K/s** |
| Storage | GB-scale | **MB-scale** |

---

*Generated: 2026-08-10 | Dataset: guardrailer_dataset_v1.parquet (722,842 samples) | Model: XGBoost GPU (300 trees, max_depth=6)*
