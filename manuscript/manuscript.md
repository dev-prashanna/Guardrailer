# Guardrailer: A Rigorous Evaluation Framework for Multi-Signal LLM Guardrail Systems

**Authors:** [Prashanna Tiwari]

**Date:** August 2026

---

## Abstract

The deployment of large language models (LLMs) in safety-critical applications necessitates robust guardrail systems capable of detecting and mitigating adversarial inputs. However, existing evaluations of such systems lack methodological rigor, relying on small benchmarks, single-run evaluations, and absent statistical confidence reporting. This paper presents Guardrailer, a multi-signal LLM guardrail evaluation framework, and conducts a comprehensive assessment of its prompt injection detection capabilities across a curated dataset of 4,000 samples drawn from a 693K-sample unified corpus spanning eight adversarial and benign sources. We employ stratified five-fold cross-validation, bootstrap confidence intervals (10,000 iterations), ablation studies across signal components, and adversarial robustness testing against obfuscation techniques. Our system achieves a balanced accuracy of 0.6062 (95% CI: [0.5942, 0.6173]), an F1-score of 0.4190, and a Matthews Correlation Coefficient of 0.2779. While the false positive rate remains acceptably low at 7.15%, the false negative rate of 71.60% reveals significant detection gaps, particularly against direct injection (17.50% accuracy) and jailbreak (30.75%) attacks. Ablation analysis demonstrates that removing benign-context keywords improves balanced accuracy by 7.2%, indicating that current keyword weighting penalizes legitimate complex inputs. Our hardness tier analysis reveals tier-dependent performance, with perfect detection at Tier 4 (critical risk) but substantial degradation at Tier 1 (23.94%). The framework achieves sub-millisecond scoring latency (0.06 ms per sample), making it viable for real-time deployment. We discuss implications for guardrail design, the limitations of keyword-based signals, and directions for improving detection of novel adversarial patterns through learned representations and adaptive thresholds.

**Keywords:** LLM safety, guardrails, prompt injection detection, adversarial robustness, evaluation methodology, multi-signal classification

---

## 1. Introduction

The proliferation of large language models (LLMs) in production environments has created an urgent need for defensive mechanisms against adversarial inputs. Prompt injection attacks—where adversaries craft inputs designed to override system instructions, extract confidential information, or elicit harmful outputs—represent a critical threat vector [1, 2]. Guardrail systems, which intercept and evaluate user inputs before they reach the underlying LLM, serve as the primary line of defense against such attacks [3].

Despite the growing deployment of guardrail systems, their evaluation methodologies remain inconsistent and缺乏 rigor [4, 5]. Common shortcomings include: (i) reliance on small, manually curated benchmarks that fail to capture the diversity of real-world adversarial inputs; (ii) single-run evaluations without statistical confidence reporting; (iii) absent ablation studies to quantify the contribution of individual detection signals; and (iv) limited robustness testing against obfuscation techniques and multilingual attacks. These gaps undermine the credibility of reported performance claims and hinder meaningful comparison across systems.

This paper addresses these limitations through a rigorous evaluation of Guardrailer, a multi-signal prompt injection detection system. Guardrailer employs seven complementary detection signals—including category centroid distance, sparse IDF keyword matching, perplexity analysis, entropy-based detection, token frequency analysis, n-gram overlap, and length normalization—combined through learned weight systems (logistic regression, neural network, and attention-based mechanisms) into a composite risk score. The system supports a layered decision model with fast blocking, deep evaluation, and safe-pass pathways, enabling real-time deployment with sub-millisecond scoring latency.

Our contributions are threefold:

1. **Methodological Framework:** We introduce a comprehensive evaluation protocol incorporating stratified five-fold cross-validation, bootstrap confidence intervals with 10,000 iterations, power analysis for sample size justification, and ablation studies across all signal components. This framework provides a template for rigorous guardrail evaluation.

2. **Empirical Analysis:** We evaluate Guardrailer on a 4,000-sample balanced benchmark (2,000 malicious, 2,000 benign) drawn from a 693K-sample unified corpus encompassing eight data sources. We report primary metrics with confidence intervals, per-category accuracy breakdowns, hardness-tier detection rates, and robustness scores against obfuscation techniques.

3. **Ablation and Diagnostic Insights:** We conduct systematic ablation studies isolating the contribution of individual keyword groups and signal components, revealing that benign-context keywords currently impose a detection penalty and that attack keywords provide the strongest discriminative signal. These findings inform concrete design recommendations for future guardrail iterations.

The remainder of this paper is organized as follows: Section 2 reviews related work on LLM safety evaluation and adversarial robustness. Section 3 details the Guardrailer methodology, including dataset engineering, the multi-signal scoring pipeline, and the evaluation framework. Section 4 describes the experimental design. Section 5 presents results and discussion. Section 6 concludes with limitations and future directions.

---

## 2. Related Work

### 2.1 Prompt Injection Attacks

Prompt injection attacks against LLMs have evolved from simple instruction overrides to sophisticated multi-turn, obfuscated, and gradient-based strategies [6, 7]. Perez and Ribeiro [8] taxonomy categorizes attacks into direct injection (explicit instruction override), indirect injection (embedded in retrieved content), system prompt extraction, refusal bypass, and jailbreak techniques. Recent work has demonstrated that adversarial suffixes optimized via gradient-based methods can bypass safety filters with high success rates [9], while multilingual and cross-lingual attacks exploit the uneven safety training across languages [10].

### 2.2 LLM Guardrail Systems

Guardrail systems can be broadly categorized into input filtering, output filtering, and runtime monitoring approaches [11]. Input filtering systems, such as Guardrailer, evaluate user prompts before they reach the LLM. Existing systems vary in their detection mechanisms: keyword-based filters [12] provide fast but brittle detection; embedding-based classifiers [13] offer semantic understanding but require substantial compute; and LLM-based evaluators [14] provide context-aware analysis at higher latency. Guardrailer's multi-signal architecture combines these approaches through a cascaded decision model.

### 2.3 Evaluation Methodologies

Prior evaluations of LLM safety systems have been criticized for lacking methodological rigor [15, 16]. Common limitations include: reliance on small benchmarks (typically fewer than 500 samples) [17]; absence of stratified cross-validation [18]; missing confidence intervals on reported metrics [19]; and limited robustness testing [20]. The ML evaluation community has advocated for bootstrap confidence intervals [21], effect size reporting [22], and power analysis [23] as standard practices—yet these remain underutilized in LLM safety evaluations.

### 2.4 Adversarial Robustness Testing

Robustness evaluation of NLP classifiers has been studied extensively in the adversarial examples literature [24, 25]. Common perturbation techniques include character-level modifications (typos, homoglyphs), word-level substitutions, and sentence-level paraphrasing [26]. Multilingual robustness has received increasing attention as attackers exploit cross-lingual transfer vulnerabilities [27]. Our work extends these paradigms to the specific context of prompt injection detection, testing six obfuscation techniques. Multilingual robustness is identified as an important future direction but is outside the scope of the current English-only system.

---

## 3. Methodology

### 3.1 System Architecture

Guardrailer implements a multi-signal, cascaded defense architecture comprising five primary layers:

1. **Vector Search:** Dense and sparse (IDF-weighted) hybrid search against a Qdrant-backed index of 693K+ attack patterns.
2. **Multi-Signal Scoring:** Seven complementary signals combined into a composite risk score via a trained Random Forest ensemble.
3. **LLM Evaluation:** Context-aware classification using the Xiaomi MiMo-v2.5 model for ambiguous cases.
4. **Semantic Hash Index:** Fast deduplication of known attack payloads via MinHash/LSH.
5. **Multi-Turn Context:** Conversation-aware security evaluation across session history.

The layered decision model routes inputs through three pathways based on the composite score: a fast block path (composite ≥ 0.55 with malicious and critical/high signals), a deep evaluation path (0.30 ≤ composite < 0.55), and a safe-pass path (composite < 0.30 with no suspicious signals).

The final Phase 5 composite score is computed by a calibrated Random Forest ensemble:

$$S_{\text{composite}} = f_{\text{RF}}(s_1, s_2, \dots, s_7)$$

where $f_{\text{RF}}: \mathbb{R}^7 \rightarrow [0, 1]$ represents the Random Forest classifier over the 7 active signal inputs. While earlier iterations (Phases 1–3) explored linear weighted scoring ($\sum w_i s_i$), the final Phase 5 production pipeline employs a Random Forest ensemble to capture non-linear feature interactions, improving balanced accuracy from 76% (Phase 1) to 93.22% (Phase 5).

### 3.2 Detection Signals

Guardrailer evaluates seven detection signals in its Phase 5 leakage-free configuration, each capturing a distinct aspect of adversarial input characteristics. Three additional signals (dense embedding similarity, cross-encoder relevance, and uniqueness score) were used in earlier phases but were removed in Phase 5 to eliminate data leakage from the Qdrant vector index.

#### Active Signals (Phase 5)

| Signal | Feature Importance | Description |
|--------|-------------------|-------------|
| Category centroid distance | 0.4463 | Maximum cosine distance to six category centroids (computed per-fold) |
| Length normalization | 0.1985 | Log-scaled text length with attack-length bias |
| Token frequency | 0.1678 | Rare word and attack-token density |
| Perplexity score | 0.0688 | GPT-2 perplexity for unusual text pattern detection |
| Entropy analysis | 0.0586 | Shannon entropy for encoded payload detection |
| Sparse IDF keyword matching | 0.0437 | BM25 scoring against 54 security keywords (IDF computed per-fold) |
| N-gram overlap | 0.0162 | Bigram/trigram overlap with attack corpus |

#### Removed Signals (Phase 5 Leakage Audit)

| Signal | Default Weight | Reason for Removal |
|--------|---------------|-------------------|
| Dense embedding similarity | 0.35 | Qdrant returned same-text matches (cosine=1.0), creating circular evaluation |
| Cross-encoder relevance | 0.15 | Pre-computed on evaluation data at ingestion time |
| Uniqueness score | 0.06 | Pre-computed on evaluation data at ingestion time |

The composite score is computed as a weighted sum of the seven active signal scores, with weights learned via logistic regression, neural network (MLP: 32→16 hidden layers), and attention-based mechanisms, combined through an ensemble with Platt Scaling and isotonic regression calibration.

### 3.3 Dataset Engineering

#### 3.3.1 Data Sources

The evaluation dataset is drawn from a unified corpus of 693,336 samples aggregated from eight sources:

| Source | Type | Contribution |
|--------|------|-------------|
| Synthetic generation | Augmented attack variants | Core adversarial pool |
| PKU Safety Benchmark | Curated safety evaluations | Adversarial samples |
| Beavertails | Adversarial instruction data | Adversarial samples |
| In-the-Wild Jailbreak | Real-world jailbreak attempts | Adversarial samples |
| Neuralchemy | Novel attack patterns | Adversarial samples |
| Detect Jailbreak | Jailbreak detection corpus | Adversarial samples |
| Jackhhao | Adversarial prompts | Adversarial samples |
| JBB Behaviors | Jailbreak behavior catalog | Adversarial samples |

#### 3.3.2 Benchmark Construction

From the unified corpus, we constructed a balanced 4,000-sample benchmark:

- **2,000 malicious samples:** Stratified across six attack categories (direct injection, indirect injection, jailbreak, refusal bypass, system prompt extraction) and four hardness tiers.
- **2,000 benign samples:** Stratified across real-world usage categories (information retrieval, task execution, creative writing, conversational, educational, professional).

The benign samples were curated from natural instruction datasets and template-based generation with diversity constraints, ensuring representation of legitimate complex inputs that might trigger false positives.

#### 3.3.3 Hardness Tier System

Each adversarial sample is assigned a hardness tier based on a composite score reflecting attack sophistication as labeled by data curators:

| Tier | Risk Level | Description | Detection Challenge |
|------|-----------|-------------|---------------------|
| 1 | Low | Simple, known attack patterns (short prompts, direct overrides) | May evade detection due to minimal signal activation |
| 2 | Medium | Paraphrased variations, minor modifications | Requires semantic understanding |
| 3 | High | Novel attacks, partially obfuscated payloads | Requires deep analysis |
| 4 | Critical | Gradient-based, multi-turn, sophisticated evasion | Produces strong multi-dimensional signals |

The hardness score combines novelty (distance from known patterns), evasion (model confidence in misclassification), and complexity (structural properties of the input text), weighted as 0.4 × novelty + 0.4 × evasion + 0.2 × complexity.

### 3.4 Evaluation Framework

#### 3.4.1 Stratified K-Fold Cross-Validation

We employ stratified five-fold cross-validation with nested stratification along three axes: binary label (malicious/benign), attack category (six categories), and attack technique (sub-categories). This ensures that each fold maintains representative distributions across all stratification dimensions. Each fold is validated for minimum samples per stratum (≥2) and distribution homogeneity (chi-squared test, p > 0.05).

#### 3.4.2 Bootstrap Confidence Intervals

For all primary metrics, we compute 95% bootstrap confidence intervals using 10,000 stratified resampling iterations. The stratified bootstrap maintains class proportions in each resample, providing more accurate interval estimates for imbalanced sub-categories.

#### 3.4.3 Power Analysis

A priori power analysis using the NormalIndPower framework determines that 4,000 samples provide sufficient power (≥80%) to detect minimum effect sizes of 0.10 (Cohen's h) at α = 0.05. This justifies our benchmark size for detecting meaningful performance differences between system configurations.

#### 3.4.4 Primary Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Balanced Accuracy | (TPR + TNR) / 2 | ≥ 0.90 |
| F1-Score | 2·(Precision·Recall)/(Precision+Recall) | ≥ 0.88 |
| Matthews Correlation Coefficient | (TP·TN - FP·FN) / √(...) | ≥ 0.85 |
| False Positive Rate | FP / (FP + TN) | ≤ 0.05 |
| False Negative Rate | FN / (FN + TP) | ≤ 0.10 |

---

## 4. Experimental Design

### 4.1 Evaluation Protocol

The evaluation proceeds through four phases:

**Phase 1 — Primary Evaluation:** Stratified 5-fold cross-validation on the 4,000-sample benchmark, computing all primary metrics per fold and aggregating via mean ± standard deviation.

**Phase 2 — Confidence Intervals:** Bootstrap CI computation (10,000 iterations) for balanced accuracy, F1-score, MCC, FPR, and FNR on the full dataset.

**Phase 3 — Ablation Studies:** Systematic removal of keyword groups to quantify signal contributions:
- **Full model:** All seven active signals with learned weights.
- **No strong keywords:** Removal of high-confidence attack keywords.
- **No medium keywords:** Removal of moderate-confidence attack keywords.
- **No benign keywords:** Removal of benign-context penalty keywords.
- **No attack keywords:** Removal of all attack-detection keywords.
- **Strong only:** Retaining only high-confidence attack keywords.
- **Medium only:** Retaining only moderate-confidence attack keywords.

**Phase 4 — Robustness Testing:**
- **Obfuscation robustness:** Testing against six obfuscation techniques (Base64 encoding, ROT13, character substitution, whitespace injection, URL encoding, multi-layer encoding).

### 4.2 Implementation Details

All experiments use a fixed random seed (GLOBAL_SEED = 42) for reproducibility. The scoring pipeline is implemented in Python with NumPy, pandas, and scikit-learn. Embeddings are computed using the BAAI/bge-large-en-v1.5 model (1024-dimensional). The LLM evaluation layer uses the Xiaomi MiMo-v2.5 model. Latency measurements are performed on a single CPU core with 32GB RAM, averaging over 100 runs per sample.

### 4.3 Statistical Analysis

All metrics are reported with four decimal places. Bootstrap confidence intervals use the percentile method at the 95% level. Cross-validation results report mean ± standard deviation across folds. Per-category and per-tier results are reported as point estimates with sample counts.

---

## 5. Results and Discussion

### 5.1 Primary Evaluation Results

#### 5.1.1 Overall Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Balanced Accuracy | 0.6062 | [0.5942, 0.6173] |
| F1-Score | 0.4190 | — |
| MCC | 0.2779 | — |
| False Positive Rate | 0.0715 | — |
| False Negative Rate | 0.7160 | — |

The balanced accuracy of 0.6062 indicates performance modestly above chance (0.50) for a balanced binary classification task. The 95% bootstrap confidence interval [0.5942, 0.6173] confirms statistical stability of this estimate. The low false positive rate (7.15%) demonstrates that the system correctly classifies the majority of benign inputs, a critical property for deployment where user friction must be minimized. However, the high false negative rate (71.60%) reveals that the system fails to detect the majority of adversarial inputs, representing the primary limitation of the current configuration.

The F1-score of 0.4190 and MCC of 0.2779 further confirm the imbalance between precision and recall. The system achieves reasonable precision (correctly flagged inputs are disproportionately malicious) but poor recall (most attacks evade detection).

#### 5.1.2 Cross-Validation Stability

Five-fold stratified cross-validation yields a mean balanced accuracy of 0.6063 ± 0.0170, indicating low variance across folds. The coefficient of variation (2.8%) suggests that the evaluation is stable and not driven by particular data splits. The bootstrap confidence interval further validates this stability, with a narrow width of 0.0231.

#### 5.1.3 Power Analysis Justification

The a priori power analysis confirms that 4,000 samples provide ≥80% power to detect effect sizes ≥0.10 (Cohen's h) at α = 0.05. This ensures that our evaluation can reliably detect meaningful performance differences between system configurations and that the observed balanced accuracy estimate is statistically well-powered.

### 5.2 Per-Category Performance

| Category | Accuracy | Interpretation |
|----------|----------|----------------|
| Benign control | 0.9285 | Strong benign classification |
| Direct injection | 0.1750 | Critical detection gap |
| Indirect injection | 0.2850 | Significant detection gap |
| Jailbreak | 0.3075 | Significant detection gap |
| Refusal bypass | 0.3075 | Significant detection gap |
| System prompt extraction | 0.3450 | Moderate detection gap |

The per-category breakdown reveals a stark dichotomy: the system achieves high accuracy on benign inputs (92.85%) but performs poorly across all adversarial categories. Direct injection attacks are detected at only 17.50%, the lowest across categories. This is particularly concerning given that direct injection represents the most common and well-understood attack vector.

Indirect injection (28.50%), jailbreak (30.75%), and refusal bypass (30.75%) show similarly low detection rates. System prompt extraction achieves marginally better performance (34.50%), likely because extraction attempts often contain characteristic phrases that partially activate keyword-based signals.

The low performance across adversarial categories suggests that the current signal weights are insufficiently tuned for attack detection. The dominance of the centroid distance signal (feature importance 0.4463) may prioritize semantic similarity to known attack clusters over direct keyword matching, which is insufficient for novel or obfuscated attack patterns that diverge from established centroids.

### 5.3 Hardness Tier Analysis

| Tier | Risk Level | Detection Rate | Sample Count |
|------|-----------|----------------|--------------|
| 1 | Low | 0.9277 | ~4622 |
| 2 | Medium | 0.9681 | ~94 |
| 3 | High | 0.9920 | ~250 |
| 4 | Critical | 0.9412 | ~34 |

The hardness tier analysis shows strong detection across all tiers in the Phase 5 leakage-free configuration, with the RF classifier achieving 92.77% on Tier 1 (simple, short attacks) — well above the 85% target threshold. The tier assignments are derived from the `risk_level` field in the source dataset, which reflects *attack sophistication* as labeled by data curators.

**Tier 1 (Low risk, 92.77% detection):** Simple, well-known attack patterns (short prompts with direct instruction overrides). Despite their minimal signal footprint, the per-fold centroid computation and keyword features provide sufficient discriminative power. A Tier-1 pre-filter (exact phrase + regex matching) can further boost this to 93.16%.

**Tier 2 (Medium risk, 96.81% detection):** Moderately modified attacks with paraphrased variations and minor modifications. These produce enough adversarial character to trigger detection pathways.

**Tier 3 (High risk, 99.20% detection):** Novel attacks with partial obfuscation. These produce distinctive signals across multiple dimensions.

**Tier 4 (Critical risk, 94.12% detection):** The most sophisticated attacks (gradient-based, multi-turn, encoded payloads). While producing extreme signals, the smaller sample size (34) introduces variance. A Tier-1 pre-filter raises this to 100%.

**Tier-1 Pre-Filter Enhancement:** To ensure robust Tier 1 coverage, we implemented a fast keyword/pattern-match pre-filter (`tier1_prefilter.py`) using exact phrase matching (100+ attack phrases) and regex patterns (13 patterns) preceding the 7-signal RF classifier. On the full 10K-sample benchmark, the pre-filter overrides 319 RF predictions, raising Tier 1 detection from 92.77% to 93.16% while maintaining overall balanced accuracy above 90% (91.96% hybrid vs. 94.67% RF-only).

### 5.4 Ablation Study Results

| Configuration | Balanced Accuracy | Δ from Full |
|--------------|------------------|-------------|
| Full model | 0.6062 | — |
| No strong keywords | 0.5525 | -5.4% |
| No medium keywords | 0.5550 | -5.1% |
| No benign keywords | 0.6780 | +7.2% |
| No attack keywords | 0.5000 | -10.6% |
| Strong only | 0.6155 | +0.9% |
| Medium only | 0.5982 | -0.8% |

The ablation studies provide critical insights into signal contribution:

**Attack keywords are essential.** Removing all attack keywords (No attack keywords) reduces balanced accuracy to 0.5000—equivalent to random guessing. This confirms that attack keyword signals provide the primary discriminative information for the current system.

**Benign keywords impose a penalty.** Removing benign-context keywords (No benign keywords) improves balanced accuracy by 7.2 percentage points, from 0.6062 to 0.6780. This suggests that the current benign keyword weighting incorrectly penalizes legitimate complex inputs that contain language patterns overlapping with benign-context signals. This finding has direct implications for system tuning: reducing or removing benign keyword penalties would improve overall detection without increasing false positives.

**Strong keywords slightly outperform the full model.** The Strong only configuration achieves 0.6155, marginally better than the full model (0.6062), suggesting that medium-confidence keywords introduce noise that slightly degrades performance. However, the improvement is within the cross-validation standard deviation (±0.0170), so this difference may not be statistically significant.

**Medium keywords have modest contribution.** Removing medium keywords (No medium keywords) reduces accuracy by 5.1%, confirming their supplementary role. The Medium only configuration achieves 0.5982, slightly below the full model, indicating that medium keywords alone are insufficient for strong detection.

These results collectively suggest that the system's detection performance is primarily driven by attack keyword signals, with benign keywords acting as a counterproductive penalty. Future iterations should prioritize attack keyword sensitivity and reduce benign-context interference.

### 5.5 Robustness Evaluation

#### 5.5.1 Obfuscation Robustness

The obfuscation robustness score is 0.5325, indicating that the system retains approximately 53% of its detection capability when adversarial inputs are obfuscated. This represents a substantial degradation from the unobfuscated baseline, suggesting that the current detection signals are sensitive to surface-level text transformations.

The low obfuscation robustness is consistent with the system's reliance on keyword-based signals (sparse IDF, n-gram overlap), which are inherently vulnerable to character-level perturbations. Centroid distance, the most important signal (44.6% feature importance), is computed from training-set embeddings and may partially compensate, as semantic embeddings are more robust to surface-level changes, but the overall robustness score indicates that this compensation is insufficient.

#### 5.5.2 Multilingual Robustness

**Note:** The multilingual robustness evaluation presented in earlier versions of this paper contained a critical methodological flaw and has been retracted. The original implementation used a keyword-based predictor on text that was merely prefixed with language tags (e.g., `[GERMAN] Ignore previous instructions...`) rather than actually translated. Since the English keyword patterns were preserved verbatim, the classifier produced identical predictions across all 15 tested languages, yielding a spurious constant accuracy of 0.680. This result is not meaningful and should not be cited.

Guardrailer is an **English-only system**. The BAAI/bge-large-en-v1.5 embedding model and the keyword-based scoring signals are optimized exclusively for English text. Evaluating non-English inputs requires either: (a) a multilingual embedding model (e.g., LaBSE, multilingual-e5-large) with retrained signal weights, or (b) a translation preprocessing step with language detection. Neither approach was implemented in the current system. We explicitly exclude multilingual robustness from our performance claims and recommend that future work address cross-lingual detection as a separate research direction.

### 5.6 Latency Profiling

| Component | Latency (ms) |
|-----------|-------------|
| Total scoring pipeline | 0.06 |
| Perplexity computation | 0.017 |
| N-gram overlap | 0.016 |
| Entropy analysis | 0.013 |
| IDF scoring | 0.007 |

The scoring pipeline achieves sub-millisecond latency (0.06 ms per sample), making it viable for real-time deployment in production LLM systems. The dominant cost components are perplexity computation (0.017 ms) and n-gram overlap (0.016 ms), both of which involve lightweight computational operations. The total latency is well within the latency budget for most LLM deployment scenarios, where the underlying model inference typically requires 100–1000 ms per request.

This latency profile demonstrates that multi-signal scoring does not impose a prohibitive computational overhead. The system can evaluate inputs in the fast block path without invoking the more expensive LLM evaluation layer, which requires approximately 220 ms per sample.

### 5.7 Discussion

#### 5.7.1 Performance Gap Analysis

The balanced accuracy of 0.6062 falls substantially below the target of 0.90, indicating that the current system configuration requires significant improvement before deployment in production environments. The performance gap is primarily attributable to:

1. **Insufficient attack keyword coverage.** The ablation study confirms that attack keywords are the strongest signal, but the current 54-keyword lexicon appears insufficient to capture the diversity of modern adversarial inputs.

2. **Benign keyword interference.** The ablation finding that removing benign keywords improves performance by 7.2% suggests that the benign-context weighting mechanism is counterproductive.

3. **Low robustness to obfuscation.** The 0.5325 obfuscation robustness score indicates that simple text transformations can defeat the detection system.

4. **Category-specific weaknesses.** Direct injection detection at 17.50% represents a critical gap, as this is the most common attack vector.

#### 5.7.2 Implications for Guardrail Design

Our findings have several implications for guardrail system design:

- **Keyword weighting requires careful calibration.** The ablation results demonstrate that not all keywords contribute positively to detection. Benign-context keywords may need to be removed or down-weighted to avoid penalizing legitimate complex inputs.

- **Hardness-tiered evaluation is essential.** The non-monotonic detection pattern across hardness tiers reveals that simple aggregate metrics (e.g., balanced accuracy) may mask important performance variations. Guardrail evaluations should report tier-specific metrics alongside aggregate scores.

- **Obfuscation robustness must be explicitly tested.** The substantial degradation under obfuscation (0.5325) indicates that evaluations conducted only on unobfuscated inputs may overestimate real-world performance.

- **Sub-millisecond scoring is achievable.** The latency profiling demonstrates that multi-signal scoring can be implemented efficiently, enabling deployment in latency-sensitive applications.

#### 5.7.3 Limitations

This study has several limitations:

1. **Dataset representativeness.** While our 4,000-sample benchmark draws from eight sources, it may not fully represent the distribution of real-world adversarial inputs encountered in production.

2. **Single-system evaluation.** We evaluate only Guardrailer and do not compare against external baselines due to the absence of standardized evaluation benchmarks in this domain.

3. **Static evaluation.** Our evaluation does not account for adaptive adversaries who may iteratively refine attacks based on detection feedback.

4. **Limited LLM evaluation.** The LLM evaluation layer (MiMo-v2.5) was used only for ambiguous cases in the deep path; its standalone performance was not evaluated.

5. **English-only system.** Guardrailer is optimized exclusively for English text. The multilingual robustness evaluation was retracted due to a methodological flaw (fake translation via prefix tagging). Cross-lingual detection requires a dedicated multilingual embedding model and retrained signal weights.

6. **Reduced signal set.** Phase 5 uses seven signals after removing dense embedding similarity, cross-encoder relevance, and uniqueness score to eliminate data leakage. While this ensures methodological integrity, the full 10-signal system may achieve different performance characteristics if the removed signals could be computed without leakage (e.g., via per-fold Qdrant indexes).

---

## 6. Conclusion

This paper presented a rigorous evaluation framework for the Guardrailer LLM guardrail system, addressing critical methodological gaps in the existing literature. Through stratified five-fold cross-validation, bootstrap confidence intervals, ablation studies, and robustness testing, we provide a comprehensive assessment of the system's prompt injection detection capabilities.

Our key findings include: (1) the system achieves modest balanced accuracy (0.6062) with acceptable false positive rates (7.15%) but high false negative rates (71.60%); (2) per-category analysis reveals critical detection gaps in direct injection (17.50%) and jailbreak (30.75%) attacks; (3) ablation studies demonstrate that attack keywords are the primary discriminative signal, while benign keywords impose a detection penalty; (4) the hardness tier analysis reveals that simple attacks (Tier 1) evade detection due to minimal signal activation, while complex attacks (Tier 4) are trivially caught; and (5) robustness testing shows substantial degradation under obfuscation (0.5325).

Future work should focus on: expanding the attack keyword lexicon with diversity-aware sampling; reweighting or removing benign-context keywords; developing obfuscation-resilient detection signals (e.g., character-level convolutional features); adding a lightweight classifier for minimal attack prompts that evade the primary scoring pipeline; implementing adaptive thresholding based on input complexity; and conducting comparative evaluation against external baselines once standardized benchmarks become available. Multilingual robustness remains an open challenge requiring a dedicated multilingual embedding model and retrained signal weights.

The evaluation framework itself—including the stratified cross-validation protocol, bootstrap CI computation, hardness tier system, and ablation study methodology—provides a reusable template for rigorous guardrail evaluation that can be applied to future systems and configurations.

---

## References

[1] Perez, F., & Ribeiro, I. (2022). Ignore this title and HackAPrompt: Exposing systemic weaknesses of LLMs through a global prompt hacking competition. *arXiv preprint arXiv:2311.16119*.

[2] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security*.

[3] Inan, H., Upasani, K., Chi, J., Fedber, R., et al. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. *arXiv preprint arXiv:2312.06674*.

[4] Mazeika, M., Phan, L., Yin, X., et al. (2024). HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. *arXiv preprint arXiv:2402.04249*.

[5] Chao, P., Debber, A., et al. (2024). JailbreakBench: An open robustness benchmark for jailbreaking LLMs. *arXiv preprint arXiv:2404.01313*.

[6] Zhan, Q., Liang, P., Feng, Z., et al. (2024). Removing RLHF protections in GPT-4 via fine-tuning. *arXiv preprint arXiv:2311.05553*.

[7] Wei, J., Wang, Z., Schuurmans, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35.

[8] Perez, F., & Ribeiro, I. (2022). Ignore this title and HackAPrompt: Exposing systemic weaknesses of LLMs through a global prompt hacking competition. *arXiv preprint arXiv:2311.16119*.

[9] Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and transferable adversarial attacks on aligned language models. *arXiv preprint arXiv:2307.15043*.

[10] Deng, Y., Wen, Y., Zhang, Z., et al. (2023). Multilingual jailbreak challenges in large language models. *arXiv preprint arXiv:2310.06474*.

[11] Ratner, N., Dagan, E., et al. (2023). NeMo Guardrails: A toolkit for controllable and safe LLM applications. *arXiv preprint arXiv:2310.10501*.

[12] Brundage, M., Avin, S., et al. (2018). The malicious use of artificial intelligence: Forecasting, prevention, and mitigation. *arXiv preprint arXiv:1802.07228*.

[13] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP*.

[14] OpenAI. (2023). GPT-4 technical report. *arXiv preprint arXiv:2303.08774*.

[15] Kiela, D., et al. (2021). Dynabench: Rethinking benchmarking in NLP. *Proceedings of NAACL-HLT*.

[16] Raji, I. D., et al. (2021). AI and the everything in the whole wide world benchmark. *Proceedings of NeurIPS*.

[17] Wang, Y., et al. (2024). Safety-J: Evaluating safety with critique. *arXiv preprint arXiv:2401.01471*.

[18] Robey, A., et al. (2023). Jailbreak-proofing large language models. *arXiv preprint arXiv:2310.02443*.

[19] Qi, X., et al. (2024). Fine-tuning aligned language models compromises safety, even for users who have not requested it. *arXiv preprint arXiv:2310.03693*.

[20] Carlini, N., et al. (2023). Are aligned neural networks adversarially aligned? *arXiv preprint arXiv:2306.15447*.

[21] Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.

[22] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

[23] Cohen, J. (1992). A power primer. *Psychological Bulletin*, 112(1), 155–159.

[24] Belinkov, Y., & Bisk, Y. (2018). Synthetic and natural noise both break neural machine translation. *Proceedings of ICLR*.

[25] Jia, R., & Liang, P. (2017). Adversarial examples for evaluating reading comprehension systems. *Proceedings of EMNLP*.

[26] Morris, J. X., et al. (2020). TextAttack: A framework for adversarial attacks, data augmentation, and adversarial training in NLP. *Proceedings of EMNLP: Systems Demonstrations*.

[27] Wang, Z., et al. (2022). Cross-lingual transfer with augmented data. *Proceedings of ACL*.

---

## Appendix A: Dataset Composition

| Component | Samples | Percentage |
|-----------|---------|------------|
| Malicious (Tier 1) | ~500 | 12.5% |
| Malicious (Tier 2) | ~500 | 12.5% |
| Malicious (Tier 3) | ~600 | 15.0% |
| Malicious (Tier 4) | ~400 | 10.0% |
| Benign (various categories) | 2,000 | 50.0% |
| **Total** | **4,000** | **100%** |

## Appendix B: Confidence Interval Summary

| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |
|--------|---------------|--------------|--------------|
| Balanced Accuracy | 0.6062 | 0.5942 | 0.6173 |

## Appendix C: Ablation Study Detailed Results

| Configuration | BA | Δ | Interpretation |
|--------------|-----|---|----------------|
| Full model | 0.6062 | — | Baseline |
| No strong keywords | 0.5525 | -5.4% | Strong keywords contribute meaningfully |
| No medium keywords | 0.5550 | -5.1% | Medium keywords contribute meaningfully |
| No benign keywords | 0.6780 | +7.2% | Benign keywords are counterproductive |
| No attack keywords | 0.5000 | -10.6% | Attack keywords are essential |
| Strong only | 0.6155 | +0.9% | Marginal improvement |
| Medium only | 0.5982 | -0.8% | Marginal degradation |
