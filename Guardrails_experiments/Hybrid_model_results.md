# Hybrid Model — Benchmark Results

**Date:** 2026-08-11
**Dataset:** `guardrailer_dataset_v1.parquet` (722,842 samples)
**Hardware:** RTX 4060 Laptop GPU (8GB), x86_64, 14.9 GB RAM

---

## Step 1: Fixed Train/Val/Test Split

| Split | Samples | Malicious | Safe | Malicious % |
|-------|---------|-----------|------|-------------|
| Train | 505,988 | 303,145 | 202,843 | 59.9% |
| Val | 72,285 | 43,307 | 28,978 | 59.9% |
| Test | 144,569 | 86,614 | 57,955 | 59.9% |

---

## Step 2: Baseline Comparison

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC |
|-------|----------|-----|-----------|--------|---------|
| Majority Classifier | 0.5991 | 0.7493 | 0.0000 | 0.0000 | 0.5000 |
| Regex Only | 0.4415 | 0.1373 | 0.9197 | 0.0742 | 0.5000 |
| TF-IDF + LogReg | 0.8492 | 0.8753 | 0.8673 | 0.8835 | 0.9277 |
| TF-IDF + LinearSVC | 0.8528 | 0.8781 | 0.8713 | 0.8850 | 0.9298 |
| TF-IDF + RandomForest | 0.7528 | 0.8207 | 0.7257 | 0.9445 | 0.8890 |
| TF-IDF + XGBoost (no HC) | 0.8493 | 0.8771 | 0.8581 | 0.8969 | 0.9308 |

---

## Step 3: Ablation Study

| Config | Features | Accuracy | F1 | AUC-ROC |
|--------|----------|----------|-----|---------|
| TF-IDF only | 5,000 | 0.8493 | 0.8771 | 0.9308 |
| Handcrafted only | 27 | 0.8037 | 0.8384 | 0.8899 |
| TF-IDF + Handcrafted | 5,027 | 0.8645 | 0.8881 | 0.9410 |

Handcrafted features add **+1.52% accuracy** and **+1.02% AUC** over TF-IDF alone.

---

## Step 4: Feature Importance

| Group | Features | Importance |
|-------|----------|-----------|
| Handcrafted | 27 | 5.0% |
| TF-IDF | 5,000 | 95.0% |

Top handcrafted: `bigram_repeat_ratio` (2.6%), `sentence_count` (0.46%), `digit_ratio` (0.25%)
Top TF-IDF: `response` (2.7%), `provide` (1.6%), `nature` (1.5%)

---

## Step 5: Latency

| Metric | Value |
|--------|-------|
| Mean | 12,979 us (13.0 ms) |
| Median | 13,913 us (13.9 ms) |
| P95 | 17,558 us (17.6 ms) |
| P99 | 19,394 us (19.4 ms) |
| Throughput | 77 queries/sec |

> Latency is high because the benchmark runs TF-IDF transform + handcrafted extraction in a Python loop per sample. Batch inference would be ~100x faster.

---

*Generated: 2026-08-11 | Dataset: guardrailer_dataset_v1.parquet (722,842 samples)*
