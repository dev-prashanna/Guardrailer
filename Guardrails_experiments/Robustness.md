# Adversarial Robustness Report

**Date:** 2026-08-11
**Model:** Hybrid Lightweight Scorer (TF-IDF + 27 Handcrafted Features + XGBoost)
**Test Set:** 100 random malicious prompts from Guardrailer test split

---

## Summary

The classifier was tested against 12 attack transformations. **Average detection rate: 88.0%**, with **misspelling** being the primary vulnerability (29% ASR).

---

## Results

| Transformation | Detection Rate | ASR | Score Delta |
|----------------|---------------|-----|-------------|
| original | 88.0% | 12.0% | +0.0000 |
| paraphrase | 88.0% | 12.0% | -0.0037 |
| **misspelling** | **71.0%** | **29.0%** | **-0.1588** |
| unicode | 85.0% | 15.0% | -0.0656 |
| whitespace | 91.0% | 9.0% | +0.0063 |
| char_insert | 89.0% | 11.0% | -0.0030 |
| indirect_framing | 92.0% | 8.0% | +0.0503 |
| roleplay | 86.0% | 14.0% | +0.0034 |
| multilingual | 88.0% | 12.0% | +0.0002 |
| **base64** | **100.0%** | **0.0%** | -0.0563 |
| euphemistic | 88.0% | 12.0% | +0.0000 |
| long_context | 90.0% | 10.0% | +0.0647 |

---

## Overall Metrics

| Metric | Value |
|--------|-------|
| Average detection rate | 88.0% |
| Average ASR | 12.0% |
| Worst-case ASR | 29.0% (misspelling) |
| Lowest detection rate | 71.0% (misspelling) |

---

## Key Findings

- **Misspelling is the primary vulnerability** — 29% ASR, score drops by 0.16
- **Base64 encoding is fully detected** — 100% detection rate
- **Most transformations have minimal impact** — paraphrase, whitespace, char_insert maintain 88-91%
- **Indirect framing slightly improves detection** — 92% rate (vs 88% baseline)

---

*Generated: 2026-08-11 | Test samples: 100 | Transformations: 12*
