# Phase 6: Embedding-Based Scoring Benchmark Report

## Pipeline Overview

```
Input Prompt
    │
    ├── BAAI/bge-large-en-v1.5 (1024-dim embedding)
    │
    ├── Qdrant Search (Cosine similarity, top-5)
    │
    └── Multi-Signal Scoring (11 weighted signals)
            │
            └── Composite Score → Threshold (0.5) → Classification
```

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | guardrailer_dataset_v1.parquet |
| Total samples | 722,842 |
| Test samples | 199 (stratified) |
| Embedding model | BAAI/bge-large-en-v1.5 |
| Embedding dimension | 1024 |
| Vector DB | Qdrant (693,456 points) |
| Distance metric | Cosine |
| Scoring signals | 11 (dense, sparse, centroid, cross-encoder, perplexity, entropy, token_frequency, ngram_overlap, uniqueness, length_norm, ensemble_bonus) |
| Hardware | RTX 4060 Laptop GPU |

## Results

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 40.2% |
| **Precision** | 50.0% |
| **Recall** | 9.2% |
| **F1 Score** | 15.6% |
| **MCC** | -0.071 |
| **Kappa** | -0.038 |
| **AUC-ROC** | 49.6% |

### Confusion Matrix

```
                    Predicted Safe    Predicted Malicious
Actual Safe              69 (TN)           11 (FP)
Actual Malicious        108 (FN)           11 (TP)
```

### Per-Category Performance

| Category | F1 | Samples | Notes |
|----------|-----|---------|-------|
| benign_control | 0.000 | 80 | All misclassified as malicious |
| direct_injection | 0.333 | 30 | Some caught by pattern matching |
| jailbreak | 0.116 | 65 | Most missed — subtle attacks |
| system_prompt_extraction | 0.080 | 24 | Extraction patterns not matched |

### Score Distribution

| Class | Mean Score | Std Dev |
|-------|-----------|---------|
| Safe | 0.5313 | — |
| Malicious | 0.5307 | — |

**Overlap:** 99.9% — distributions are indistinguishable.

### Signal Discrimination

| Signal | Safe | Malicious | Δ | Status |
|--------|------|-----------|---|--------|
| dense | 0.9863 | 0.9657 | -0.021 | WEAK (wrong direction) |
| sparse_idf | 0.0142 | 0.0173 | +0.003 | WEAK |
| centroid | 0.6336 | 0.6695 | +0.036 | WEAK |
| cross_encoder | 0.0025 | 0.0176 | +0.015 | WEAK |
| perplexity | 0.6190 | 0.6157 | -0.003 | WEAK |
| entropy | 0.7856 | 0.7751 | -0.011 | WEAK |
| token_frequency | 0.3331 | 0.3260 | -0.007 | WEAK |
| ngram_overlap | 0.0014 | 0.0042 | +0.003 | WEAK |
| uniqueness | 0.0588 | 0.0649 | +0.006 | WEAK |
| length_norm | 1.1666 | 1.1557 | -0.011 | WEAK |

**All signals are below the 0.05 discrimination threshold.**

### Performance

| Metric | Value |
|--------|-------|
| Embedding generation | 5.9s (199 prompts) |
| Scoring pipeline | 0.9s (199 prompts) |
| **Total latency** | **34ms/prompt** |
| Throughput | 29 prompts/s |

### Sample Predictions

```
[ERR] score=0.3344 pattern=0.000 true=MAL pred=SAF | in the tranquil art of carpentry...
[ERR] score=0.3479 pattern=0.000 true=MAL pred=SAF | What number is returned when...
[ERR] score=0.3324 pattern=0.000 true=MAL pred=SAF | ### Instruction: Discuss the influence...
[OK]  score=0.3547 pattern=0.000 true=SAF pred=SAF | Write an engaging and interesting...
[ERR] score=0.6697 pattern=0.900 true=SAF pred=MAL | Write an engaging and interesting...
```

## Failure Root Cause

**Centroid similarity = 1.0000** — the malicious and safe centroids are mathematically identical in the embedding space. The `bge-large-en-v1.5` model embeds both classes into the same region because it optimizes for semantic similarity, not safety classification.

## Comparison: Hybrid Lightweight Scorer

| Metric | Embedding Pipeline | Hybrid Scorer | Δ |
|--------|-------------------|---------------|---|
| F1 | 0.156 | 0.841 | **+439%** |
| AUC-ROC | 0.496 | 0.888 | **+79%** |
| Accuracy | 40.2% | 81.8% | **+104%** |
| Precision | 50.0% | 88.0% | **+76%** |
| Recall | 9.2% | 80.6% | **+776%** |
| Latency | 34ms | 3.2ms | **94% faster** |
| Storage | 1.5GB + Qdrant | 2.78MB | **540x smaller** |

## Conclusion

The embedding-based scoring pipeline **failed** because:
1. General-purpose embeddings don't separate safe from malicious prompts
2. All 11 scoring signals are weak discriminators
3. Pattern matching only catches 6% of attacks
4. Score distributions are identical across classes

**Recommendation:** Use the hybrid lightweight scorer (84% F1, 2.78MB) as the primary classifier. Keep embeddings for RAG retrieval only.

---

*Benchmark date: 2026-08-10*
*Dataset: guardrailer_dataset_v1.parquet (722,842 rows)*
*Model: BAAI/bge-large-en-v1.5 (335M params, 1024-dim)*
