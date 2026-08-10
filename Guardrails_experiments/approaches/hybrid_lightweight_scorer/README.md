# Hybrid Lightweight Scorer

**Goal:** Replace embedding+Qdrant pipeline with a lightweight multi-signal scoring system.

## Architecture

```
Input Prompt
    │
    ├── Layer 1: Pattern Matching (instant, ~0.01ms)
    │   ├── Regex patterns (injection, jailbreak, encoding)
    │   ├── Keyword frequency scoring
    │   └── Text anomaly detection
    │
    ├── Layer 2: TF-IDF BM25 (fast, ~0.1ms)
    │   ├── Cosine similarity to known attacks
    │   └── Keyword relevance scoring
    │
    ├── Layer 3: Feature-based Classifier (fast, ~0.5ms)
    │   ├── 50 handcrafted features
    │   ├── Logistic Regression / LightGBM
    │   └── Calibrated probabilities
    │
    └── Ensemble Score → Threat Level
```

## Resources Required

| Resource | Current (Embeddings+Qdrant) | Hybrid Lightweight |
|----------|---------------------------|-------------------|
| Storage | 1.5GB embeddings + Qdrant | ~10MB models |
| GPU | Required for generation | Not required |
| RAM | 4GB+ for Qdrant | <500MB |
| Query time | ~5ms | ~0.5ms |
| Ingestion | Hours (GPU needed) | Minutes (CPU only) |

## Signals Used

1. **Pattern Detection:** Regex for known attack vectors
2. **Keyword Scoring:** IDF-weighted keyword matching
3. **Text Statistics:** Length, entropy, encoding density
4. **Structural Analysis:** Instruction nesting, role switching
5. **N-gram Overlap:** Character/word n-gram similarity to attacks

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train the classifier
python src/train.py --data ../../dataset/guardrailer_dataset_v1.parquet

# Run inference
python src/scorer.py --text "Ignore previous instructions..."

# Run evaluation
python src/evaluate.py --data ../../dataset/guardrailer_dataset_v1.parquet
```

## Status

- [x] Directory structure created
- [x] Feature extraction pipeline
- [x] Pattern matching layer
- [x] TF-IDF similarity layer
- [x] Classifier training
- [x] Evaluation against full dataset
- [x] Benchmark comparison

See [BENCHMARK.md](BENCHMARK.md) for full benchmark results.
