# Cross-Dataset Generalization Report

**Date:** 2026-08-11
**Model:** Hybrid Lightweight Scorer (TF-IDF + 27 Handcrafted Features + XGBoost)
**Training Dataset:** Guardrailer v1 (722,842 samples)

---

## Summary

The classifier achieved **86.45% accuracy** on the in-distribution test set but dropped to an average of **54.31% accuracy** on external datasets — a **32.14 percentage point decline**. This indicates **weak cross-dataset generalization**.

---

## Results

| Dataset | Samples | Accuracy | F1 | AUC-ROC | Δ Accuracy |
|---------|---------|----------|-----|---------|------------|
| Guardrailer v1 (in-dist) | 144,569 | **86.45%** | **0.888** | **0.941** | — |
| JailbreakBench | 200 | 56.00% | 0.688 | 0.673 | -30.45 pp |
| Jailbreak Classification | 1,044 | 57.76% | 0.650 | 0.655 | -28.69 pp |
| Jailbreak Complete DS | 11,383 | 43.06% | 0.582 | 0.450 | -43.39 pp |
| JailbreakHub | 15,140 | 60.41% | 0.246 | 0.689 | -26.04 pp |
| **Average (external)** | **27,767** | **54.31%** | **0.541** | **0.617** | **-32.14 pp** |

---

## Root Cause Analysis

1. **TF-IDF vocabulary mismatch** — vectorizer fit on Guardrailer prompts; external prompts have OOV tokens
2. **Feature distribution shift** — handcrafted features tuned to Guardrailer's attack patterns
3. **Class imbalance sensitivity** — model biased toward predicting malicious (60% malicious in training)
4. **Label semantics differ** — external datasets define "malicious" differently

---

## Recommendations

1. Add external data to training set (10-20% mix)
2. Retrain TF-IDF on combined corpus
3. Add character-level TF-IDF for obfuscation robustness
4. Add embedding features for semantic understanding
5. Multi-dataset training for generalization

---

*Generated: 2026-08-11 | External Datasets: 4 (27,767 samples)*
