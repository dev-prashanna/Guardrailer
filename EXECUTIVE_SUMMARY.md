# Guardrailer Evaluation Methodology: Executive Summary

**Date:** 2026-08-03
**Version:** 1.0
**Status:** Complete

---

## Problem Statement

The current Guardrailer evaluation methodology has critical gaps that undermine research credibility:

| Gap | Current State | Required State |
|-----|---------------|----------------|
| **Benchmark Size** | 200 samples | ≥4,000 samples |
| **Data Contamination** | Manual YAML curation | Standardized adversarial datasets |
| **Class Imbalance** | 50% benign, ~3.3% per attack | Stratified balanced sampling |
| **Statistical Rigor** | Single evaluation run | K-Fold CV + Bootstrap CIs + Power Analysis |
| **Reproducibility** | No versioning or seed control | Full deterministic pipeline |

---

## Solution Overview

### 1. Methodological Reconstruction

**Key Components:**
- **Stratified K-Fold CV**: 5-fold cross-validation with nested stratification (label → category → technique)
- **Bootstrap Confidence Intervals**: 10,000 iterations for 95% CIs on all metrics
- **Power Analysis**: Minimum detectable effect size of 5% with 80% power

**Statistical Rigor Requirements:**
- Point estimates with 4 decimal places
- 95% bootstrap CIs (BCa method)
- McNemar's test for significance (p < 0.05)
- Cohen's h for effect size reporting

### 2. Dataset Engineering Protocol

**Adversarial Datasets:**
| Dataset | Samples | Purpose |
|---------|---------|---------|
| HarmBench | 510 | Comprehensive harm evaluation |
| AdvBench | 520 | Adversarial attack benchmark |
| WildJailbreak | 10,000 | Real-world jailbreak attempts |
| JailbreakBench | 100 | Jailbreak detection |

**Benign Class Redesign:**
- 6 stratified real-world usage categories
- 2,500 total benign samples
- Template-based generation with diversity constraints

**Adversarial Hardness Gradient:**
- Tier 1 (Easy): 25% - Known patterns
- Tier 2 (Medium): 25% - Variations with modifications
- Tier 3 (Hard): 30% - Novel attacks, obfuscated
- Tier 4 (Adversarial): 20% - Evasion-focused, gradient-based

**Target Dataset:** 4,000 samples total

### 3. Experimental Design & Validation

**Four-Phase Lifecycle:**
1. **Dataset Construction** (2 weeks): Acquisition, generation, validation
2. **Model Training** (1 week): Feature extraction, weight learning, calibration
3. **Evaluation** (2 weeks): Cross-validation, ablation studies, robustness testing
4. **External Comparison** (1 week): Baseline comparison, manuscript preparation

**Ablation Studies:**
- A1: No Dense Signal
- A2: No Sparse Keywords
- A3: No Cross-Encoder
- A4: No LLM Evaluation
- A5: Default Weights
- A6: No Calibration
- A7: Single Signal

**Research-Grade Metrics:**
| Metric | Target | Reporting |
|--------|--------|-----------|
| Balanced Accuracy | ≥ 90% | Mean ± 95% CI |
| ECE | ≤ 5% | Mean ± 95% CI |
| AUC-ROC | ≥ 95% | Mean ± 95% CI |
| F1-Score | ≥ 88% | Mean ± 95% CI |
| MCC | ≥ 85% | Mean ± 95% CI |

### 4. Robustness & Reproducibility Framework

**Adversarial Robustness Testing:**
- 6 obfuscation techniques (Base64, ROT13, character substitution, etc.)
- 5 language families (Germanic, Romance, Slavic, CJK, Indic)
- Minimum 30 samples per test condition

**Reproducibility Checklist:**
- Git commit hash documentation
- Dataset SHA-256 hash verification
- Global seed control (GLOBAL_SEED=42)
- Environment specification (environment.yml)
- Hardware specification logging
- Execution time tracking

### 5. Implementation Roadmap

**Timeline:** 6 weeks total

```
Week 1-2:  [==================] P0: Dataset & Methodology
Week 3-4:  [==================] P1: Evaluation & Robustness
Week 5-6:  [==================] P2: Manuscript & Release
```

**Priority Breakdown:**

| Priority | Tasks | Effort | Impact |
|----------|-------|--------|--------|
| **P0** | Dataset acquisition, statistical framework, reproducibility | 12 days | Critical |
| **P1** | Ablation studies, robustness testing, baseline comparison | 16 days | High |
| **P2** | Manuscript preparation, code cleanup, release | 13 days | Medium |

**Resource Requirements:**
- GPU: 1x T4 or better
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+
- Duration: 6 weeks

---

## Key Deliverables

| Deliverable | Format | Location |
|-------------|--------|----------|
| Dataset | Parquet | `data/guardrailer_benchmark_v2.parquet` |
| Manifest | JSON | `data/manifest.json` |
| Models | JSON/PT | `models_v2/` |
| Results | JSON | `results/evaluation_v2.json` |
| Report | Markdown | `reports/evaluation_report_v2.md` |
| Checklist | Markdown | `EVALUATION_CHECKLIST.md` |
| Full Plan | Markdown | `RESEARCH_EXECUTION_PLAN.md` |

---

## Success Criteria

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

## Risk Mitigation

| Risk | Probability | Mitigation Strategy |
|------|-------------|---------------------|
| Dataset acquisition delays | Medium | Start early, use cached versions |
| GPU unavailability | Low | Use cloud instances (Kaggle, Colab) |
| Reproducibility failures | Low | Continuous verification, seed control |
| Insufficient statistical power | Medium | Run power analysis early, increase sample size |
| Baseline comparison unavailable | Medium | Use public benchmarks, request access |

---

## Next Steps

1. **Review** the full research execution plan (`RESEARCH_EXECUTION_PLAN.md`)
2. **Begin** P0 tasks: dataset acquisition and statistical framework implementation
3. **Set up** project structure for new evaluation pipeline
4. **Schedule** weekly progress reviews
5. **Document** all decisions and deviations from the plan

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-08-03 | Research Team | Initial release |

**Review Schedule:** Weekly during execution, monthly during planning phase.
**Approval Required:** Lead Researcher and Security Engineer sign-off before execution begins.
