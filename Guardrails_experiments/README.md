# Guardrailer — Hybrid Model Benchmark & Evaluation

A lightweight ML-based prompt injection detection system. This branch contains the complete benchmarking pipeline, adversarial robustness testing, and cross-dataset generalization evaluation.

---

## Repository Structure

```
Guardrails_experiments/
├── README.md                          # This file
├── requirements.txt                   # Root dependencies
├── pyproject.toml                     # Project metadata & tooling
├── configs/
│   └── shared.yaml                    # Shared experiment configs
├── approaches/
│   ├── hybrid_lightweight_scorer/     # Base model (86.5% accuracy)
│   │   ├── src/
│   │   │   ├── features.py            # 44 handcrafted features
│   │   │   ├── patterns.py            # 35+ regex patterns
│   │   │   ├── similarity.py          # TF-IDF similarity scorer
│   │   │   ├── scorer.py             # Main scorer class
│   │   │   ├── train.py              # Memory-optimized training
│   │   │   ├── train_full.py         # Full dataset training
│   │   │   ├── evaluate.py           # Evaluation & benchmarks
│   │   │   ├── calibration.py        # Score calibration
│   │   │   └── scorer_demo.py        # Demo script
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── BENCHMARK.md
│   └── hybrid_enhanced/              # Enhanced model (with embeddings)
│       ├── src/
│       │   ├── __init__.py
│       │   ├── enhanced_features.py  # 56+ features (adversarial robust)
│       │   ├── enhanced_scorer.py    # Main scorer class
│       │   ├── embedding_layer.py    # Embedding similarity layer
│       │   ├── patterns.py           # Expanded regex patterns
│       │   ├── similarity.py         # TF-IDF similarity
│       │   ├── train_enhanced.py     # Training script
│       │   ├── demo.py               # Demo script
│       │   └── test_features.py      # Feature tests
│       ├── requirements.txt
│       ├── README.md
│       └── FLOWCHART.md
├── benchmark/                        # Benchmarking scripts & results
│   ├── step1_split.py                # Train/val/test split creation
│   ├── step2_baselines.py            # Baseline model comparison
│   ├── step3_ablation.py             # Ablation study
│   ├── step4_feature_importance.py   # Feature importance analysis
│   ├── step5_latency.py              # Latency benchmark
│   ├── save_model.py                 # Save trained model artifacts
│   ├── adversarial_robustness.py     # 12 attack transformations
│   ├── cross_dataset.py              # Cross-dataset generalization
│   └── download_external.py          # Download external datasets
├── Hybrid_model_results.md           # Full benchmark results report
├── Robustness.md                     # Adversarial robustness report
└── crossdata.md                      # Cross-dataset generalization report
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Create fixed train/val/test split
python benchmark/step1_split.py

# Step 2: Run baseline models
python benchmark/step2_baselines.py

# Step 3: Run ablation study
python benchmark/step3_ablation.py

# Step 4: Get feature importance
python benchmark/step4_feature_importance.py

# Step 5: Run latency benchmark
python benchmark/step5_latency.py

# Step 6: Save model artifacts
python benchmark/save_model.py

# Step 7: Run adversarial robustness testing
python benchmark/adversarial_robustness.py

# Step 8: Run cross-dataset generalization
python benchmark/cross_dataset.py
```

---

## Key Results

### Model Performance (In-Distribution)

| Metric | Value |
|--------|-------|
| Accuracy | **86.45%** |
| F1 Score | **0.8881** |
| AUC-ROC | **0.9410** |
| Model Size | **~7 MB** |
| Features | 5,027 (5,000 TF-IDF + 27 handcrafted) |

### Adversarial Robustness

| Transformation | Detection Rate | ASR |
|----------------|---------------|-----|
| Original | 88.0% | 12.0% |
| Paraphrase | 88.0% | 12.0% |
| Misspelling | 71.0% | 29.0% |
| Unicode | 85.0% | 15.0% |
| Base64 | 100.0% | 0.0% |
| **Average** | **88.0%** | **12.0%** |

### Cross-Dataset Generalization

| Dataset | Accuracy | Δ Accuracy |
|---------|----------|------------|
| Guardrailer v1 (in-dist) | 86.45% | — |
| JailbreakBench | 56.00% | -30.45 pp |
| Jailbreak Classification | 57.76% | -28.69 pp |
| Jailbreak Complete DS | 43.06% | -43.39 pp |
| JailbreakHub | 60.41% | -26.04 pp |
| **Average (external)** | **54.31%** | **-32.14 pp** |

---

## Dataset

- **Source:** `guardrailer_dataset_v1.parquet`
- **Size:** 722,842 labeled prompts
- **Split:** 70% train / 10% val / 20% test (stratified)
- **Classes:** Safe (40.1%) / Malicious (59.9%)

Place the dataset in `/home/prashanna/Documents/Guardrailer/dataset/` before running benchmarks.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any | x86_64 |
| RAM | 4 GB | 16 GB |
| GPU | Not required | RTX 4060+ (for XGBoost training) |
| Storage | 500 MB | 2 GB |

---

## Generated Reports

| Report | Description |
|--------|-------------|
| `Hybrid_model_results.md` | Complete benchmark results (Steps 1-5) |
| `Robustness.md` | Adversarial robustness testing (12 transformations) |
| `crossdata.md` | Cross-dataset generalization (4 external datasets) |

---

## License

MIT
