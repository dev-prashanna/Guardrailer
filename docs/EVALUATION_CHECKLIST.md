# Guardrailer Evaluation Methodology: Implementation Checklist

## Quick Reference Guide

**Document:** `RESEARCH_EXECUTION_PLAN.md` (full version)
**Date:** 2026-08-03
**Status:** Ready for Execution

---

## Critical Path (P0 - Week 1-2)

### Dataset Engineering
- [ ] Acquire HarmBench dataset (510 samples)
- [ ] Acquire AdvBench dataset (520 samples)
- [ ] Acquire WildJailbench dataset (10,000 samples)
- [ ] Generate 2,500 benign samples across 6 categories
- [ ] Compute hardness scores for all adversarial samples
- [ ] Assign tiers (1-4) based on hardness scores
- [ ] Validate dataset composition (4,000+ samples)
- [ ] Version and hash dataset (SHA-256)

### Statistical Framework
- [ ] Implement Stratified K-Fold CV (K=5)
- [ ] Implement Bootstrap CI computation (10,000 iterations)
- [ ] Run power analysis for required sample sizes
- [ ] Set random seeds (GLOBAL_SEED=42)
- [ ] Verify reproducibility across 3 runs

### Validation Pipeline
- [ ] Create dataset manifest (JSON)
- [ ] Implement dataset validation checks
- [ ] Set up environment specification (environment.yml)
- [ ] Create artifact management structure

---

## High Priority (P1 - Week 3-4)

### Evaluation Pipeline
- [ ] Run stratified 5-fold CV evaluation
- [ ] Compute bootstrap CIs for all metrics
- [ ] Perform 7 ablation studies (A1-A7)
- [ ] Conduct statistical significance tests (McNemar's)
- [ ] Profile latency (p50, p95, p99)

### Robustness Testing
- [ ] Test 6 obfuscation techniques (50 samples each)
- [ ] Test 5 language families (30 samples each)
- [ ] Compute robustness scores per technique
- [ ] Document failure modes and edge cases

### Baseline Comparison
- [ ] Benchmark against 3+ baseline systems
- [ ] Compute effect sizes (Cohen's h)
- [ ] Generate comparison tables
- [ ] Create visualization plots

---

## Medium Priority (P2 - Week 5-6)

### Documentation
- [ ] Prepare manuscript draft
- [ ] Create supplementary materials
- [ ] Document ablation study results
- [ ] Write robustness analysis section

### Production Readiness
- [ ] Code review and cleanup
- [ ] Final reproducibility verification
- [ ] Dataset release preparation
- [ ] Peer review preparation

---

## Key Metrics to Report

### Primary Metrics (with 95% CI)
| Metric | Target | Status |
|--------|--------|--------|
| Balanced Accuracy | ≥ 90% | [ ] |
| ECE | ≤ 5% | [ ] |
| AUC-ROC | ≥ 95% | [ ] |
| F1-Score | ≥ 88% | [ ] |
| MCC | ≥ 85% | [ ] |

### Secondary Metrics
| Metric | Target | Status |
|--------|--------|--------|
| False Positive Rate | ≤ 5% | [ ] |
| False Negative Rate | ≤ 10% | [ ] |
| Latency (p95) | ≤ 100ms | [ ] |
| Robustness Score | ≥ 80% | [ ] |

---

## Reproducibility Requirements

### Required Artifacts
- [ ] Dataset: `data/guardrailer_benchmark_v2.parquet`
- [ ] Manifest: `data/manifest.json`
- [ ] Models: `models_v2/`
- [ ] Results: `results/evaluation_v2.json`
- [ ] Report: `reports/evaluation_report_v2.md`

### Verification Steps
1. [ ] Git commit hash documented
2. [ ] Dataset SHA-256 hash verified
3. [ ] All random seeds fixed (GLOBAL_SEED=42)
4. [ ] Environment exported (environment.yml)
5. [ ] Hardware specs documented
6. [ ] Execution time logged

---

## Ablation Study Checklist

| ID | Study | Component Removed | Status |
|----|-------|-------------------|--------|
| A1 | No Dense Signal | Dense embedding | [ ] |
| A2 | No Sparse Keywords | Sparse IDF | [ ] |
| A3 | No Cross-Encoder | Cross-encoder relevance | [ ] |
| A4 | No LLM Evaluation | LLM layer | [ ] |
| A5 | Default Weights | Learned weights | [ ] |
| A6 | No Calibration | Probability calibration | [ ] |
| A7 | Single Signal | Multi-signal fusion | [ ] |

---

## Robustness Test Checklist

### Obfuscation Techniques
- [ ] Base64 encoding (50 samples)
- [ ] ROT13 cipher (50 samples)
- [ ] Character substitution (50 samples)
- [ ] Whitespace injection (50 samples)
- [ ] URL encoding (50 samples)
- [ ] Multi-layer encoding (30 samples)

### Multilingual Tests
- [ ] Germanic (German, Dutch, Swedish)
- [ ] Romance (French, Spanish, Italian)
- [ ] Slavic (Russian, Polish, Czech)
- [ ] CJK (Chinese, Japanese, Korean)
- [ ] Indic (Hindi, Bengali, Tamil)

---

## Statistical Significance Tests

| Comparison | Test | α Level | Status |
|------------|------|---------|--------|
| Model vs Baseline 1 | McNemar's | 0.05 | [ ] |
| Model vs Baseline 2 | McNemar's | 0.05 | [ ] |
| Model vs Baseline 3 | McNemar's | 0.05 | [ ] |
| Multiple comparisons | Bonferroni | 0.05/k | [ ] |

---

## Success Criteria Summary

| Criterion | Target | Verified |
|-----------|--------|----------|
| Dataset Size | ≥ 4,000 samples | [ ] |
| Balanced Accuracy | ≥ 90% ± CI | [ ] |
| ECE | ≤ 5% ± CI | [ ] |
| AUC-ROC | ≥ 95% ± CI | [ ] |
| Reproducibility | 100% match | [ ] |
| Statistical Significance | p < 0.05 | [ ] |
| Robustness Score | ≥ 80% | [ ] |
| Manuscript | Submission-ready | [ ] |

---

## Resource Requirements

| Resource | Quantity | Purpose |
|----------|----------|---------|
| GPU | 1x T4+ | Inference, embeddings |
| CPU | 8+ cores | Parallel evaluation |
| RAM | 32GB+ | Dataset processing |
| Storage | 100GB+ | Artifacts |
| Time | 6 weeks | Full cycle |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Dataset delays | Start early, use cached versions |
| GPU unavailable | Use cloud (Kaggle, Colab) |
| Reproducibility failure | Continuous verification |
| Insufficient power | Run power analysis first |
| Baseline unavailable | Use public benchmarks |

---

**Next Steps:**
1. Review full plan: `RESEARCH_EXECUTION_PLAN.md`
2. Begin P0 tasks (dataset acquisition)
3. Set up project structure for new evaluation pipeline
4. Schedule weekly progress reviews
