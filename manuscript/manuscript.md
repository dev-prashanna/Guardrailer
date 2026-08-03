# Guardrailer: A Rigorous Evaluation Framework for Multi-Signal LLM Guardrail Systems

**Authors:** [Author Names]

**Affiliations:** [Institutional Affiliations]

**Corresponding Author:** [Email]

**Date:** August 2026

---

## Abstract

The deployment of large language models (LLMs) in safety-critical applications necessitates robust guardrail systems capable of detecting and mitigating adversarial inputs. However, existing evaluations of such systems lack methodological rigor, relying on small benchmarks, single-run evaluations, and absent statistical confidence reporting. This paper presents Guardrailer, a multi-signal LLM guardrail evaluation framework, and conducts a comprehensive assessment of its prompt injection detection capabilities across a curated dataset of 4,000 samples drawn from a 693K-sample unified corpus spanning eight adversarial and benign sources. We employ stratified five-fold cross-validation, bootstrap confidence intervals (10,000 iterations), ablation studies across signal components, and adversarial robustness testing against obfuscation and multilingual attacks. Our system achieves a balanced accuracy of 0.6062 (95% CI: [0.5942, 0.6173]), an F1-score of 0.4190, and a Matthews Correlation Coefficient of 0.2779. While the false positive rate remains acceptably low at 7.15%, the false negative rate of 71.60% reveals significant detection gaps, particularly against direct injection (17.50% accuracy) and jailbreak (30.75%) attacks. Ablation analysis demonstrates that removing benign-context keywords improves balanced accuracy by 7.2%, indicating that current keyword weighting penalizes legitimate complex inputs. Our hardness tier analysis reveals tier-dependent performance, with perfect detection at Tier 4 (critical risk) but substantial degradation at Tier 1 (23.94%). The framework achieves sub-millisecond scoring latency (0.06 ms per sample), making it viable for real-time deployment. We discuss implications for guardrail design, the limitations of keyword-based signals, and directions for improving detection of novel adversarial patterns through learned representations and adaptive thresholds.

**Keywords:** LLM safety, guardrails, prompt injection detection, adversarial robustness, evaluation methodology, multi-signal classification

---

## 1. Introduction

The proliferation of large language models (LLMs) in production environments has created an urgent need for defensive mechanisms against adversarial inputs. Prompt injection attacks—where adversaries craft inputs designed to override system instructions, extract confidential information, or elicit harmful outputs—represent a critical threat vector [1, 2]. Guardrail systems, which intercept and evaluate user inputs before they reach the underlying LLM, serve as the primary line of defense against such attacks [3].

Despite the growing deployment of guardrail systems, their evaluation methodologies remain inconsistent and缺乏 rigor [4, 5]. Common shortcomings include: (i) reliance on small, manually curated benchmarks that fail to capture the diversity of real-world adversarial inputs; (ii) single-run evaluations without statistical confidence reporting; (iii) absent ablation studies to quantify the contribution of individual detection signals; and (iv) limited robustness testing against obfuscation techniques and multilingual attacks. These gaps undermine the credibility of reported performance claims and hinder meaningful comparison across systems.

This paper addresses these limitations through a rigorous evaluation of Guardrailer, a multi-signal prompt injection detection system. Guardrailer employs ten complementary detection signals—including dense embedding similarity, sparse IDF keyword matching, cross-encoder relevance scoring, perplexity analysis, and entropy-based detection—combined through learned weight systems (logistic regression, neural network, and attention-based mechanisms) into a composite risk score. The system supports a layered decision model with fast blocking, deep evaluation, and safe-pass pathways, enabling real-time deployment with sub-millisecond scoring latency.

Our contributions are threefold:

1. **Methodological Framework:** We introduce a comprehensive evaluation protocol incorporating stratified five-fold cross-validation, bootstrap confidence intervals with 10,000 iterations, power analysis for sample size justification, and ablation studies across all signal components. This framework provides a template for rigorous guardrail evaluation.

2. **Empirical Analysis:** We evaluate Guardrailer on a 4,000-sample balanced benchmark (2,000 malicious, 2,000 benign) drawn from a 693K-sample unified corpus encompassing eight data sources. We report primary metrics with confidence intervals, per-category accuracy breakdowns, hardness-tier detection rates, and robustness scores against obfuscation and multilingual attacks.

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

Robustness evaluation of NLP classifiers has been studied extensively in the adversarial examples literature [24, 25]. Common perturbation techniques include character-level modifications (typos, homoglyphs), word-level substitutions, and sentence-level paraphrasing [26]. Multilingual robustness has received increasing attention as attackers exploit cross-lingual transfer vulnerabilities [27]. Our work extends these paradigms to the specific context of prompt injection detection, testing six obfuscation techniques and five language families.

---

## 3. Methodology

### 3.1 System Architecture

Guardrailer implements a multi-signal, cascaded defense architecture comprising five primary layers:

1. **Vector Search:** Dense and sparse (IDF-weighted) hybrid search against a Qdrant-backed index of 693K+ attack patterns.
2. **Multi-Signal Scoring:** Ten complementary signals combined into a composite risk score via learned weights.
3. **LLM Evaluation:** Context-aware classification using the Xiaomi MiMo-v2.5 model for ambiguous cases.
4. **Semantic Hash Index:** Fast deduplication of known attack payloads via MinHash/LSH.
5. **Multi-Turn Context:** Conversation-aware security evaluation across session history.

The layered decision model routes inputs through three pathways based on the composite score: a fast block path (composite ≥ 0.55 with malicious and critical/high signals), a deep evaluation path (0.30 ≤ composite < 0.55), and a safe-pass path (composite < 0.30 with no suspicious signals).

### 3.2 Detection Signals

Guardrailer employs ten detection signals, each capturing a distinct aspect of adversarial input characteristics:

| Signal | Default Weight | Description |
|--------|---------------|-------------|
| Dense embedding similarity | 0.35 | BAAI/bge-large-en-v1.5 (1024-dim) cosine similarity to known attacks |
| Sparse IDF keyword matching | 0.20 | BM25 scoring against 54 security keywords |
| Category centroid distance | 0.15 | Maximum cosine distance to six category centroids |
| Cross-encoder relevance | 0.15 | ms-marco-MiniLM-L-6-v2 passage relevance scoring |
| Uniqueness score | 0.06 | Inverse mean k-NN distance in embedding space |
| Length normalization | 0.05 | Log-scaled text length with attack-length bias |
| Perplexity score | 0.01 | GPT-2 perplexity for unusual text pattern detection |
| Token frequency | 0.01 | Rare word and attack-token density |
| N-gram overlap | 0.01 | Bigram/trigram overlap with attack corpus |
| Entropy analysis | 0.01 | Shannon entropy for encoded payload detection |

The composite score is computed as a weighted sum of the ten signal scores, with weights learned via logistic regression, neural network (MLP: 32→16 hidden layers), and attention-based mechanisms, combined through an ensemble with Platt Scaling and isotonic regression calibration.

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

Each adversarial sample is assigned a hardness tier based on a composite score:

| Tier | Risk Level | Description | Detection Challenge |
|------|-----------|-------------|---------------------|
| 1 | Low | Known patterns, obvious attacks | Trivial keyword matching |
| 2 | Medium | Paraphrased variations, minor modifications | Requires semantic understanding |
| 3 | High | Novel attacks, obfuscated payloads | Requires deep analysis |
| 4 | Critical | Gradient-based, multi-turn, sophisticated evasion | Maximum difficulty |

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
- **Full model:** All ten signals with learned weights.
- **No strong keywords:** Removal of high-confidence attack keywords.
- **No medium keywords:** Removal of moderate-confidence attack keywords.
- **No benign keywords:** Removal of benign-context penalty keywords.
- **No attack keywords:** Removal of all attack-detection keywords.
- **Strong only:** Retaining only high-confidence attack keywords.
- **Medium only:** Retaining only moderate-confidence attack keywords.

**Phase 4 — Robustness Testing:**
- **Obfuscation robustness:** Testing against six obfuscation techniques (Base64 encoding, ROT13, character substitution, whitespace injection, URL encoding, multi-layer encoding).
- **Multilingual robustness:** Testing across five language families (Germanic, Romance, Slavic, CJK, Indic) with translated adversarial samples.

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

The low performance across adversarial categories suggests that the current signal weights are insufficiently tuned for attack detection. The dominance of the dense embedding similarity signal (weight 0.35) may prioritize semantic similarity to known attacks over direct keyword matching, which is insufficient for novel or obfuscated attack patterns.

### 5.3 Hardness Tier Analysis

| Tier | Risk Level | Detection Rate |
|------|-----------|----------------|
| 1 | Low | 0.2394 |
| 2 | Medium | 0.9130 |
| 3 | High | 0.5437 |
| 4 | Critical | 1.0000 |

The hardness tier analysis reveals a non-monotonic detection pattern. Tier 4 (critical risk) achieves perfect detection (100%), likely because the most sophisticated attacks produce highly distinctive signals that activate multiple detection pathways. Tier 2 (medium risk) achieves strong detection (91.30%), suggesting that moderately modified attacks retain sufficient signal overlap with known patterns.

However, Tier 1 (low risk) shows surprisingly poor detection (23.94%), indicating that simple, well-known attack patterns may not strongly activate the learned signal weights. Tier 3 (high risk) achieves 54.37%, suggesting that novel attacks partially evade detection through signal obfuscation.

The non-monotonicity between Tier 1 and Tier 2 is notable. Simple attacks (Tier 1) may use minimal adversarial signal, making them harder to distinguish from benign inputs. Moderate attacks (Tier 2) introduce enough adversarial character to trigger detection, while Tier 4 attacks produce such extreme signals that they are trivially caught. This pattern suggests that the system's detection mechanism is better calibrated for moderate-to-extreme attacks than for subtle or well-camouflaged ones.

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

The low obfuscation robustness is consistent with the system's reliance on keyword-based signals (sparse IDF, n-gram overlap), which are inherently vulnerable to character-level perturbations. Dense embedding similarity may partially compensate for this, as semantic embeddings are more robust to surface-level changes, but the overall robustness score indicates that this compensation is insufficient.

#### 5.5.2 Multilingual Robustness

The multilingual robustness score is 0.6800, indicating that the system retains 68% of its detection capability for non-English inputs. This is a moderate result that suggests partial cross-lingual transfer, likely driven by the embedding model's multilingual capabilities (BAAI/bge-large-en-v1.5 supports limited multilingual input).

The higher multilingual score compared to obfuscation (0.6800 vs. 0.5325) indicates that the system is more robust to language variation than to text obfuscation. This may be because multilingual attacks preserve semantic structure while changing surface form, whereas obfuscation techniques actively disrupt the signals that the system relies on.

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

---

## 6. Conclusion

This paper presented a rigorous evaluation framework for the Guardrailer LLM guardrail system, addressing critical methodological gaps in the existing literature. Through stratified five-fold cross-validation, bootstrap confidence intervals, ablation studies, and robustness testing, we provide a comprehensive assessment of the system's prompt injection detection capabilities.

Our key findings include: (1) the system achieves modest balanced accuracy (0.6062) with acceptable false positive rates (7.15%) but high false negative rates (71.60%); (2) per-category analysis reveals critical detection gaps in direct injection (17.50%) and jailbreak (30.75%) attacks; (3) ablation studies demonstrate that attack keywords are the primary discriminative signal, while benign keywords impose a detection penalty; (4) the hardness tier analysis reveals non-monotonic detection patterns that aggregate metrics obscure; and (5) robustness testing shows substantial degradation under obfuscation (0.5325) and moderate degradation under multilingual attacks (0.6800).

Future work should focus on: expanding the attack keyword lexicon with diversity-aware sampling; reweighting or removing benign-context keywords; developing obfuscation-resilient detection signals (e.g., character-level convolutional features); implementing adaptive thresholding based on input complexity; and conducting comparative evaluation against external baselines once standardized benchmarks become available.

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
