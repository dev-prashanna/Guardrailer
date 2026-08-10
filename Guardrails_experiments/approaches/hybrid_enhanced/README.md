# Enhanced Hybrid Lightweight Scorer

**Production-Ready Prompt Injection Detection**

A 4-layer hybrid classifier combining traditional ML with modern embeddings for robust, lightweight prompt injection detection.

## Architecture

```
Input Prompt
    │
    ├── Layer 1: Pattern Matching (~0.01ms)
    │   ├── Regex patterns (injection, jailbreak, encoding)
    │   ├── Keyword frequency scoring
    │   └── Text anomaly detection
    │
    ├── Layer 2: TF-IDF Similarity (~0.1ms)
    │   ├── Cosine similarity to known attacks (50K features)
    │   └── IDF-weighted keyword relevance
    │
    ├── Layer 3: Embedding Similarity (~1ms)
    │   ├── BAAI/bge-small-en-v1.5 (33M params, 384 dims)
    │   ├── Centroid distance to malicious/benign clusters
    │   └── Semantic similarity features
    │
    ├── Layer 4: Enhanced Feature Classifier (~0.5ms)
    │   ├── 55+ handcrafted features (adversarial robust)
    │   ├── 384 embedding features
    │   ├── 5,000 TF-IDF features
    │   └── XGBoost GPU (300 trees)
    │
    └── Ensemble Score → Threat Level
```

## Key Improvements Over Base Model

| Feature | Base Hybrid | Enhanced Hybrid |
|---------|-------------|-----------------|
| Accuracy | 86.5% | **>90%** (target) |
| Embedding Layer | None | BAAI/bge-small-en-v1.5 |
| Handcrafted Features | 44 | **55+** |
| Adversarial Robustness | Limited | **Leetspeak, Unicode, Obfuscation** |
| Model Size | 7.1 MB | **~200 MB** (with embeddings) |
| Inference Latency | 0.001ms | **~1ms** (still sub-2ms) |

## Features

### Adversarial Robustness Features
- Leetspeak detection (1337 → elite)
- Unicode anomaly detection (homoglyphs, zero-width chars)
- Character entropy analysis
- Text perplexity estimation
- Encoding pattern detection

### Enhanced Structural Features
- Instruction nesting depth
- Role switching detection
- Delimiter injection patterns
- Persona switch detection
- Quotation and emphasis analysis

### Embedding Features (New)
- Centroid distance to malicious cluster
- Centroid distance to benign cluster
- Semantic similarity scores
- Embedding norm and dimension

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train with embeddings (recommended, GPU)
python src/train_enhanced.py

# Train without embeddings (CPU-only mode)
python src/train_enhanced.py --no-embeddings

# Run inference
python src/scorer_demo.py --text "Ignore previous instructions..."
```

## Model Size

| Component | Size |
|-----------|------|
| XGBoost classifier | ~1 MB |
| TF-IDF vectorizer | ~0.1 MB |
| Similarity scorer | ~2 MB |
| Keyword scorer | ~4.5 MB |
| Embedding model (bge-small) | ~130 MB |
| **Total** | **~140 MB** |

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Accuracy | >90% | Production-grade detection |
| Recall (Malicious) | >95% | Minimize missed attacks |
| Precision | >85% | Limit false positives |
| Latency | <2ms | Real-time inference |
| Throughput | >500K/s | High-volume deployment |

## Files

```
hybrid_enhanced/
├── src/
│   ├── __init__.py
│   ├── enhanced_scorer.py      # Main scorer class
│   ├── embedding_layer.py      # Lightweight embedding layer
│   ├── enhanced_features.py    # 55+ features with adversarial robustness
│   ├── patterns.py             # Pattern matching (from base model)
│   ├── similarity.py           # TF-IDF similarity (from base model)
│   └── train_enhanced.py       # Training script
├── models/                     # Saved model artifacts
├── results/                    # Benchmark results
├── requirements.txt
└── README.md
```

## Training

The model trains on the full Guardrailer dataset (722K samples) with:

1. **Stratified 80/20 split** — balanced train/test distribution
2. **GPU-accelerated XGBoost** — fast training on RTX 4060
3. **Batch feature extraction** — memory-safe for large datasets
4. **Embedding precomputation** — one-time cost, cached for inference

## Reproduction

```bash
# Full training with embeddings
python src/train_enhanced.py

# Results saved to:
#   src/models/       — model artifacts
#   src/results/      — benchmark JSON
```

## Notes

- Embedding layer adds ~130MB but significantly improves accuracy
- Can be disabled with `--no-embeddings` for CPU-only deployment
- Model supports both single-text and batch inference
- All features are interpretable for security audit trails
