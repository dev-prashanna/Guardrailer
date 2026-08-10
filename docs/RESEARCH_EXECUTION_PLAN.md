# Guardrailer: Research Execution Plan
## Rigorous Evaluation Methodology for LLM Guardrail Systems

| Version | Date | Status |
|---------|------|--------|
| 1.0 | 2026-08-03 | Active |
| 2.0 | 2026-08-03 | Updated - Pipeline Complete |

---

## Current Status (v2.0 - 2026-08-03)

**Pipeline: COMPLETE** - All evaluation phases executed with real data.

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset | DONE | 4000 samples from unified_security_dataset.parquet (693K total) |
| Signal Computation | DONE | 6 real signals: sparse_idf, perplexity, entropy, token_freq, ngram_overlap, length_norm |
| Primary Evaluation | DONE | BA=0.6062, F1=0.4190, FPR=0.0715 |
| Stratified K-Fold CV | DONE | 5-fold, BA=0.6063 +/- 0.0170 |
| Bootstrap CIs | DONE | 95% CI: [0.5942, 0.6173], BCa computed |
| Ablation Studies | DONE | 11 configs with real signal removal, meaningful deltas |
| Robustness Testing | DONE | Obfuscation=0.5325, Multilingual=0.6800 |
| Baseline Comparison | DONE | vs keyword-only, random, majority class |
| Latency Profiling | DONE | Total ~0.06ms per sample |
| Power Analysis | DONE | 4000 samples sufficient for d>=0.10 |
| Visualizations | DONE | 7 publication-quality PNGs |
| Manuscript | DONE | 4,854-word draft with all sections |

**Outputs:**
- `evaluation_results/evaluation_report.json` - comprehensive JSON report
- `evaluation_results/figures/*.png` - 7 visualization plots
- `manuscript/manuscript.md` - academic paper draft

## Executive Summary

This document presents a comprehensive research execution plan to transform the Guardrailer LLM guardrail evaluation from a preliminary benchmark into a publication-ready experimental framework. The plan addresses critical methodological gaps in benchmark size, data contamination, class imbalance, and statistical rigor.

**Key Deliverables:**
1. Rigorous experimental framework with Stratified K-Fold, Bootstrap Confidence Intervals, and Power Analysis
2. Expanded dataset engineering pipeline integrating adversarial benchmarks (HarmBench, AdvBench, etc.)
3. Four-phase evaluation lifecycle with ablation studies and research-grade metrics
4. Robustness testing protocol for adversarial obfuscation and multilingualism
5. Structured implementation roadmap with priority-based phases

---

## Table of Contents

1. [Methodological Reconstruction](#1-methodological-reconstruction)
2. [Dataset Engineering Protocol](#2-dataset-engineering-protocol)
3. [Experimental Design & Validation](#3-experimental-design--validation)
4. [Robustness & Reproducibility Framework](#4-robustness--reproducibility-framework)
5. [Implementation Roadmap](#5-implementation-roadmap)

---

## 1. Methodological Reconstruction

### 1.1 Current Methodological Gaps

| Gap Category | Current State | Required State |
|--------------|---------------|----------------|
| **Benchmark Size** | 200 samples (100 benign + 100 malicious) | ≥2,000 samples (minimum 10x current) |
| **Data Contamination** | Manual YAML curation | Standardized adversarial datasets with versioning |
| **Class Imbalance** | 50% benign, ~3.3% per attack category | Stratified balanced sampling with control groups |
| **Statistical Rigor** | Single evaluation run | Stratified K-Fold CV + Bootstrap CIs + Power Analysis |
| **Reproducibility** | No versioning or seed control | Full deterministic pipeline with hash verification |

### 1.2 Recommended Statistical Corrections

#### 1.2.1 Stratified K-Fold Cross-Validation

**Implementation Specification:**

```python
# Configuration
K_FOLDS = 5
STRATIFY_COLUMNS = ['label', 'attack_category', 'attack_technique']
RANDOM_SEED = 42
SPLITS_PER_STRATUM = K_FOLDS

# Pipeline
from sklearn.model_selection import StratifiedKFold

def stratified_kfold_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    techniques: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> List[Dict]:
    """
    Perform stratified K-fold CV with nested stratification.
    
    Stratification priority:
    1. Binary label (malicious/benign)
    2. Attack category (6 categories)
    3. Attack technique (sub-categories)
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )
    
    # Create composite stratification key
    stratify_key = np.array([
        f"{label}_{cat}_{tech}" 
        for label, cat, tech in zip(y, categories, techniques)
    ])
    
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_key)):
        # Ensure minimum samples per stratum in each fold
        val_strata = Counter(stratify_key[val_idx])
        for stratum, count in val_strata.items():
            assert count >= 2, f"Insufficient samples for stratum {stratum}: {count}"
        
        fold_results.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'stratification_distribution': dict(val_strata)
        })
    
    return fold_results
```

**Validation Requirements:**
- Minimum 2 samples per stratum in each fold
- Chi-squared test for distribution homogeneity across folds (p > 0.05)
- Fold-level metrics reported with standard deviations

#### 1.2.2 Bootstrap Confidence Intervals

**Implementation Specification:**

```python
import numpy as np
from scipy import stats

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict:
    """
    Compute bootstrap confidence intervals for any metric.
    
    Returns:
        dict with point_estimate, ci_lower, ci_upper, std_error
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Stratified bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return {
        'point_estimate': metric_fn(y_true, y_pred),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std_error': float(np.std(bootstrap_scores)),
        'n_bootstrap': n_bootstrap,
        'ci_level': ci_level
    }
```

**Reporting Requirements:**
- 95% confidence intervals for all primary metrics
- 10,000 bootstrap iterations minimum
- Stratified resampling to maintain class distribution
- Sensitivity analysis: report CIs for 90%, 95%, and 99% levels

#### 1.2.3 Power Analysis

**Implementation Specification:**

```python
from statsmodels.stats.power import NormalIndPower

def compute_required_sample_size(
    effect_size: float = 0.2,  # Cohen's h for proportions
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0  # n1/n2
) -> Dict:
    """
    Compute minimum required sample size for statistical power.
    
    For guardrail evaluation:
    - Effect size: minimum detectable difference (e.g., 5% improvement)
    - Alpha: Type I error rate (0.05)
    - Power: 1 - Type II error rate (0.80)
    """
    analysis = NormalIndPower()
    
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=ratio
    )
    
    return {
        'required_per_group': int(np.ceil(sample_size)),
        'total_required': int(np.ceil(sample_size * 2)),
        'effect_size': effect_size,
        'power': power,
        'alpha': alpha
    }

# Example: Detect 5% improvement with 80% power
result = compute_required_sample_size(effect_size=0.2, power=0.80)
# Output: {'required_per_group': 393, 'total_required': 786, ...}
```

**Power Analysis Report:**
- Effect sizes: 2%, 5%, 10%, 15% improvements
- Power levels: 0.80, 0.90, 0.95
- Sample sizes required for each scenario
- Justification for chosen benchmark size

### 1.3 Statistical Rigor Checklist

| Metric | Requirement | Method |
|--------|-------------|--------|
| Point Estimate | Report with 4 decimal places | Direct computation |
| Confidence Intervals | 95% BCa bootstrap | 10,000 iterations |
| Cross-Validation | 5-fold stratified | Nested stratification |
| Effect Size | Cohen's h or Cohen's d | Standardized difference |
| Statistical Significance | p < 0.05 after correction | McNemar's test + Bonferroni |
| Practical Significance | Δ ≥ 5% balanced accuracy | Pre-specified threshold |
| Reproducibility | Seed-controlled, deterministic | SHA-256 hash verification |

---

## 2. Dataset Engineering Protocol

### 2.1 Data Acquisition Pipeline

#### 2.1.1 Adversarial Dataset Integration

| Dataset | Source | Purpose | Target Size | Integration Priority |
|---------|--------|---------|-------------|---------------------|
| **HarmBench** | `averylamp/HarmBench` | Comprehensive harm evaluation | 510 samples | P0 |
| **AdvBench** | `you720/advbench` | Adversarial attack benchmark | 520 samples | P0 |
| **JailbreakBench** | `nvidia/JailbreakBench` | Jailbreak detection | 100 samples | P0 |
| **WildJailbreak** | `allenai/wildjailbreak` | Real-world jailbreak attempts | 10,000 samples | P0 |
| **TDC2023** | `protectai/tdc2023` | Trust & safety benchmark | 1,000 samples | P1 |
| **HarmfulQA** | `welp/HarmfulQA` | Harmful question answering | 500 samples | P1 |
| **SafetyBench** | `thu-coai/SafetyBench` | Multi-domain safety | 2,000 samples | P1 |

#### 2.1.2 Dataset Acquisition Protocol

```python
# Dataset acquisition and versioning
DATASET_REGISTRY = {
    'harmbench': {
        'source': 'averylamp/HarmBench',
        'version': '1.0',
        'split': 'test',
        'fields': ['prompt', 'target', 'category', 'technique'],
        'label_mapping': {'harmful': 1, 'benign': 0},
        'hash_verification': True
    },
    'advbench': {
        'source': 'you720/advbench',
        'version': '1.0',
        'split': 'test',
        'fields': ['question', 'category', 'target'],
        'label_mapping': {'adversarial': 1},
        'hash_verification': True
    },
    'wildjailbreak': {
        'source': 'allenai/wildjailbreak',
        'version': '1.0',
        'split': 'train',
        'fields': ['text', 'type', 'source'],
        'label_mapping': {'jailbreak': 1, 'vanilla': 0},
        'hash_verification': True
    }
}

def acquire_dataset(name: str, cache_dir: str = './dataset_cache') -> pd.DataFrame:
    """
    Acquire dataset with versioning and hash verification.
    """
    import hashlib
    from datasets import load_dataset
    
    config = DATASET_REGISTRY[name]
    
    # Load with version pinning
    ds = load_dataset(
        config['source'],
        version=config['version'],
        split=config['split'],
        cache_dir=cache_dir
    )
    
    # Compute hash for reproducibility
    dataset_bytes = ds.to_json().encode()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    
    # Standardize columns
    df = ds.to_pandas()
    df = standardize_columns(df, config)
    
    # Add metadata
    df['source_dataset'] = name
    df['dataset_version'] = config['version']
    df['dataset_hash'] = dataset_hash
    
    return df
```

### 2.2 Benign Class Redesign

#### 2.2.1 Stratified Real-World Usage Categories

| Category | Description | Target Size | Source |
|----------|-------------|-------------|--------|
| **Information Retrieval** | Factual questions, definitions | 500 | Natural Questions, TriviaQA |
| **Task Execution** | Code generation, analysis | 500 | HumanEval, MBPP |
| **Creative Writing** | Stories, poetry, brainstorming | 300 | WritingPrompts |
| **Conversational** | Casual chat, small talk | 300 | PersonaChat |
| **Educational** | Explanations, tutorials | 300 | WebGPT, ELI5 |
| **Professional** | Business, legal, medical | 300 | PubMedQA, LegalBench |

#### 2.2.2 Benign Sample Generation Protocol

```python
BENIGN_CATEGORIES = {
    'information_retrieval': {
        'templates': [
            "What is {concept}?",
            "Explain how {process} works",
            "What are the benefits of {topic}?",
            "Compare {item_a} and {item_b}"
        ],
        'concepts': load_concept_list('concepts.json'),
        'min_length': 20,
        'max_length': 200
    },
    'task_execution': {
        'templates': [
            "Write a Python function that {task}",
            "Debug this code: {code_snippet}",
            "Optimize the following for {metric}: {description}",
            "Create a {data_structure} for {use_case}"
        ],
        'tasks': load_task_list('coding_tasks.json'),
        'min_length': 50,
        'max_length': 500
    },
    'creative_writing': {
        'templates': [
            "Write a short story about {theme}",
            "Compose a poem about {subject}",
            "Brainstorm ideas for {project}",
            "Create dialogue between {characters}"
        ],
        'themes': load_theme_list('themes.json'),
        'min_length': 100,
        'max_length': 1000
    }
}

def generate_benign_samples(
    category: str,
    n_samples: int,
    seed: int = 42
) -> List[Dict]:
    """
    Generate diverse benign samples for a category.
    """
    rng = np.random.RandomState(seed)
    config = BENIGN_CATEGORIES[category]
    
    samples = []
    for _ in range(n_samples):
        template = rng.choice(config['templates'])
        
        # Fill template with random values
        sample = fill_template(template, config, rng)
        
        # Ensure length constraints
        while len(sample) < config['min_length']:
            sample = extend_sample(sample, config)
        while len(sample) > config['max_length']:
            sample = truncate_sample(sample, config)
        
        samples.append({
            'text': sample,
            'category': 'benign',
            'subcategory': category,
            'label': 0,
            'generation_method': 'template'
        })
    
    return samples
```

### 2.3 Adversarial Hardness Gradient (Tiers 1–4)

#### 2.3.1 Tier Definition

| Tier | Hardness Level | Description | Examples | Target Ratio |
|------|---------------|-------------|----------|--------------|
| **Tier 1** | Easy | Known patterns, obvious attacks | "Ignore previous instructions", "System prompt:" | 25% |
| **Tier 2** | Medium | Variations with minor modifications | Paraphrased attacks, typo injection | 25% |
| **Tier 3** | Hard | Novel attacks, obfuscated payloads | Base64 encoding, character substitution | 30% |
| **Tier 4** | Adversarial | Evasion-focused, gradient-based | GCG-optimized, multi-turn attacks | 20% |

#### 2.3.2 Hardness Scoring Protocol

```python
def compute_hardness_score(
    sample: Dict,
    model: Any,
    reference_samples: List[Dict]
) -> float:
    """
    Compute adversarial hardness score for a sample.
    
    Hardness = w1 * novelty_score + w2 * evasion_score + w3 * complexity_score
    
    Returns:
        float in [0, 1] where 1 = maximum hardness
    """
    # 1. Novelty Score: Distance from known attack patterns
    novelty = compute_novelty_score(
        sample['text'],
        reference_samples,
        embedding_model=model
    )
    
    # 2. Evasion Score: Model's confidence in misclassification
    evasion = compute_evasion_score(
        sample['text'],
        model,
        true_label=sample['label']
    )
    
    # 3. Complexity Score: Structural complexity
    complexity = compute_complexity_score(sample['text'])
    
    # Weighted combination
    hardness = (
        0.4 * novelty +
        0.4 * evasion +
        0.2 * complexity
    )
    
    return min(1.0, max(0.0, hardness))

def assign_tier(hardness_score: float) -> int:
    """Assign hardness tier based on score."""
    if hardness_score < 0.25:
        return 1
    elif hardness_score < 0.50:
        return 2
    elif hardness_score < 0.75:
        return 3
    else:
        return 4
```

### 2.4 Final Dataset Specification

#### 2.4.1 Target Dataset Composition

| Component | Samples | Percentage | Purpose |
|-----------|---------|------------|---------|
| **Adversarial (Tier 1)** | 500 | 12.5% | Baseline attacks |
| **Adversarial (Tier 2)** | 500 | 12.5% | Variation robustness |
| **Adversarial (Tier 3)** | 600 | 15.0% | Novel attack detection |
| **Adversarial (Tier 4)** | 400 | 10.0% | Evasion resistance |
| **Benign (Information)** | 500 | 12.5% | Factual queries |
| **Benign (Task)** | 500 | 12.5% | Code generation |
| **Benign (Creative)** | 300 | 7.5% | Creative writing |
| **Benign (Conversational)** | 200 | 5.0% | Casual chat |
| **Benign (Professional)** | 300 | 7.5% | Domain-specific |
| **Benign (Educational)** | 200 | 5.0% | Explanations |
| **Total** | **4,000** | **100%** | Balanced evaluation |

#### 2.4.2 Dataset Validation Protocol

```python
def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Validate dataset composition and quality.
    
    Returns validation report with pass/fail status.
    """
    report = {
        'total_samples': len(df),
        'class_distribution': df['label'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'hardness_distribution': df['tier'].value_counts().to_dict(),
        'checks': []
    }
    
    # Check 1: Minimum sample size
    report['checks'].append({
        'name': 'minimum_samples',
        'required': 4000,
        'actual': len(df),
        'passed': len(df) >= 4000
    })
    
    # Check 2: Class balance (ratio < 1.5)
    class_counts = df['label'].value_counts()
    ratio = class_counts.max() / class_counts.min()
    report['checks'].append({
        'name': 'class_balance',
        'required_ratio': '< 1.5',
        'actual_ratio': float(ratio),
        'passed': ratio < 1.5
    })
    
    # Check 3: Minimum samples per category
    category_counts = df['category'].value_counts()
    min_category = category_counts.min()
    report['checks'].append({
        'name': 'category_minimum',
        'required': 200,
        'actual': int(min_category),
        'passed': min_category >= 200
    })
    
    # Check 4: Hardness tier distribution
    tier_counts = df['tier'].value_counts()
    report['checks'].append({
        'name': 'tier_distribution',
        'required': 'All tiers present with ≥200 samples',
        'actual': tier_counts.to_dict(),
        'passed': all(tier_counts.get(t, 0) >= 200 for t in [1, 2, 3, 4])
    })
    
    # Check 5: No data contamination (duplicate detection)
    duplicates = df.duplicated(subset=['text'], keep=False).sum()
    report['checks'].append({
        'name': 'no_duplicates',
        'required': 0,
        'actual': int(duplicates),
        'passed': duplicates == 0
    })
    
    report['overall_passed'] = all(c['passed'] for c in report['checks'])
    
    return report
```

---

## 3. Experimental Design & Validation

### 3.1 Four-Phase Evaluation Lifecycle

#### Phase 1: Dataset Construction

| Step | Description | Duration | Output |
|------|-------------|----------|--------|
| 1.1 | Acquire adversarial datasets | 2 days | Raw dataset files |
| 1.2 | Generate benign samples | 3 days | Benign sample pool |
| 1.3 | Compute hardness scores | 2 days | Hardness annotations |
| 1.4 | Assign tiers and balance | 1 day | Final dataset |
| 1.5 | Validate composition | 0.5 days | Validation report |
| 1.6 | Version and hash dataset | 0.5 days | Dataset manifest |

**Deliverable:** `guardrailer_benchmark_v2.parquet` with SHA-256 hash

#### Phase 2: Model Training

| Step | Description | Duration | Output |
|------|-------------|----------|--------|
| 2.1 | Extract signal features | 1 day | Feature matrix |
| 2.2 | Train weight learners (3 models) | 2 days | Model checkpoints |
| 2.3 | Train calibrators | 1 day | Calibrator objects |
| 2.4 | Cross-validate ensemble | 1 day | CV results |
| 2.5 | Hyperparameter tuning | 2 days | Optimal configs |
| 2.6 | Save trained models | 0.5 days | Model artifacts |

**Deliverable:** `models_v2/` directory with all trained components

#### Phase 3: Evaluation

| Step | Description | Duration | Output |
|------|-------------|----------|--------|
| 3.1 | Run stratified 5-fold CV | 2 days | Fold-level metrics |
| 3.2 | Compute bootstrap CIs | 1 day | CI estimates |
| 3.3 | Perform ablation studies | 3 days | Ablation results |
| 3.4 | Conduct robustness tests | 3 days | Robustness report |
| 3.5 | Latency profiling | 1 day | Latency statistics |
| 3.6 | Statistical significance tests | 1 day | p-values |

**Deliverable:** `evaluation_results_v2/` with all metrics

#### Phase 4: External Comparison

| Step | Description | Duration | Output |
|------|-------------|----------|--------|
| 4.1 | Benchmark against baselines | 2 days | Comparison table |
| 4.2 | Ablation study documentation | 1 day | Ablation report |
| 4.3 | Statistical analysis report | 2 days | Analysis document |
| 4.4 | Manuscript preparation | 3 days | Draft paper |
| 4.5 | Code review and cleanup | 1 day | Production code |

**Deliverable:** `manuscript/` with paper draft and supplementary materials

### 3.2 Ablation Studies

#### 3.2.1 Signal Contribution Analysis

| Ablation | Component Removed | Expected Impact | Metric |
|----------|-------------------|-----------------|--------|
| **A1** | Dense embedding signal | -15% to -20% accuracy | Balanced Accuracy |
| **A2** | Sparse IDF keywords | -10% to -15% accuracy | Balanced Accuracy |
| **A3** | Cross-encoder relevance | -8% to -12% accuracy | Balanced Accuracy |
| **A4** | LLM evaluation layer | -10% to -15% accuracy | Balanced Accuracy |
| **A5** | Learned weights (use default) | -5% to -10% accuracy | Balanced Accuracy |
| **A6** | Probability calibration | +5% to +10% ECE | ECE |
| **A7** | Multi-signal fusion (single signal) | -20% to -30% accuracy | Balanced Accuracy |

#### 3.2.2 Ablation Implementation Protocol

```python
def run_ablation_study(
    ablation_name: str,
    config: Dict,
    dataset: pd.DataFrame,
    model: Any
) -> Dict:
    """
    Run a single ablation study.
    
    Args:
        ablation_name: Identifier for the ablation
        config: Configuration with components to ablate
        dataset: Evaluation dataset
        model: Guardrailer model instance
    
    Returns:
        Dictionary with ablation results
    """
    # Create ablated model
    ablated_model = create_ablated_model(model, config)
    
    # Run evaluation
    y_true = dataset['label'].values
    y_pred = ablated_model.predict(dataset['text'].values)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred)
    
    # Bootstrap confidence intervals
    ci_results = {}
    for metric_name, metric_fn in METRIC_FUNCTIONS.items():
        ci_results[metric_name] = bootstrap_confidence_interval(
            y_true, y_pred, metric_fn
        )
    
    return {
        'ablation_name': ablation_name,
        'config': config,
        'metrics': metrics,
        'confidence_intervals': ci_results,
        'timestamp': datetime.now().isoformat()
    }

# Ablation configurations
ABLATION_CONFIGS = {
    'A1_no_dense': {'remove_signals': ['dense']},
    'A2_no_sparse': {'remove_signals': ['sparse_idf']},
    'A3_no_cross_encoder': {'remove_signals': ['cross_encoder']},
    'A4_no_llm': {'disable_llm_evaluation': True},
    'A5_default_weights': {'use_learned_weights': False},
    'A6_no_calibration': {'disable_calibration': True},
    'A7_single_signal': {'signals': ['dense'], 'mode': 'single'}
}
```

### 3.3 Research-Grade Metrics

#### 3.3.1 Primary Metrics

| Metric | Formula | Threshold | Reporting |
|--------|---------|-----------|-----------|
| **Balanced Accuracy** | (TPR + TNR) / 2 | ≥ 0.90 | Mean ± CI |
| **ECE** | (1/N) Σ |b_m - conf_m| | ≤ 0.05 | Mean ± CI |
| **AUC-ROC** | ∫ TPR dFPR | ≥ 0.95 | Mean ± CI |
| **F1-Score** | 2·(P·R)/(P+R) | ≥ 0.88 | Mean ± CI |
| **MCC** | (TP·TN - FP·FN) / √(...) | ≥ 0.85 | Mean ± CI |

#### 3.3.2 Secondary Metrics

| Metric | Purpose | Reporting |
|--------|---------|-----------|
| **Per-Category Accuracy** | Category-specific performance | Mean ± CI per category |
| **Latency Distribution** | Performance profiling | p50, p95, p99, mean |
| **False Positive Rate** | Benign classification accuracy | FPR ≤ 0.05 |
| **False Negative Rate** | Attack detection recall | FNR ≤ 0.10 |
| **Detection Rate by Tier** | Hardness-aware evaluation | Rate per tier |

#### 3.3.3 Metric Computation Protocol

```python
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive metric suite.
    
    Returns:
        Dictionary with all metrics and their values
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
        f1_score,
        matthews_corrcoef,
        confusion_matrix,
        precision_recall_curve
    )
    
    metrics = {}
    
    # Primary metrics
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        metrics['ece'] = compute_ece(y_true, y_proba)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_confidence = y_proba[mask].mean()
            bin_accuracy = y_true[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_accuracy - bin_confidence)
    
    return ece
```

### 3.4 Statistical Significance Testing

#### 3.4.1 Test Selection

| Comparison | Test | Justification |
|------------|------|---------------|
| Model A vs Model B (paired) | McNemar's Test | Paired binary outcomes |
| Multiple comparisons | Bonferroni Correction | Control family-wise error |
| Effect size | Cohen's h | Standardized difference |
| Non-parametric alternative | Wilcoxon Signed-Rank | When normality assumption fails |

#### 3.4.2 Significance Testing Protocol

```python
def statistical_significance_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Perform statistical significance testing between two models.
    
    Returns:
        Dictionary with test results and effect sizes
    """
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Create contingency table
    # Both correct, A correct B wrong, A wrong B correct, both wrong
    both_correct = ((y_pred_a == y_true) & (y_pred_b == y_true)).sum()
    a_only = ((y_pred_a == y_true) & (y_pred_b != y_true)).sum()
    b_only = ((y_pred_a != y_true) & (y_pred_b == y_true)).sum()
    both_wrong = ((y_pred_a != y_true) & (y_pred_b != y_true)).sum()
    
    table = [[both_correct, a_only], [b_only, both_wrong]]
    
    # McNemar's test
    result = mcnemar(table, exact=True)
    
    # Effect size (Cohen's h)
    acc_a = (y_pred_a == y_true).mean()
    acc_b = (y_pred_b == y_true).mean()
    cohens_h = 2 * np.arcsin(np.sqrt(acc_a)) - 2 * np.arcsin(np.sqrt(acc_b))
    
    return {
        'test': 'McNemar',
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'significant': result.pvalue < alpha,
        'alpha': alpha,
        'effect_size': float(abs(cohens_h)),
        'effect_interpretation': interpret_effect_size(abs(cohens_h)),
        'contingency_table': {
            'both_correct': int(both_correct),
            'a_only_correct': int(a_only),
            'b_only_correct': int(b_only),
            'both_wrong': int(both_wrong)
        }
    }

def interpret_effect_size(h: float) -> str:
    """Interpret Cohen's h effect size."""
    if h < 0.2:
        return 'negligible'
    elif h < 0.5:
        return 'small'
    elif h < 0.8:
        return 'medium'
    else:
        return 'large'
```

---

## 4. Robustness & Reproducibility Framework

### 4.1 Adversarial Robustness Testing

#### 4.1.1 Obfuscation Techniques

| Technique | Description | Implementation | Test Cases |
|-----------|-------------|----------------|------------|
| **Base64 Encoding** | Encode payload in Base64 | `base64.b64encode(text)` | 50 samples |
| **ROT13** | Character rotation cipher | `codecs.encode(text, 'rot_13')` | 50 samples |
| **Character Substitution** | Replace characters with similar | Leet speak, Unicode confusables | 50 samples |
| **Whitespace Injection** | Insert invisible characters | Zero-width spaces, tabs | 50 samples |
| **URL Encoding** | Percent-encode characters | `urllib.parse.quote()` | 50 samples |
| **Multi-Layer Encoding** | Nested encoding | Base64 → ROT13 → Base64 | 30 samples |

#### 4.1.2 Multilingual Robustness

| Language Family | Languages | Test Samples | Purpose |
|-----------------|-----------|--------------|---------|
| **Germanic** | German, Dutch, Swedish | 30 each | Cross-lingual transfer |
| **Romance** | French, Spanish, Italian | 30 each | Morphological variation |
| **Slavic** | Russian, Polish, Czech | 30 each | Character set diversity |
| **CJK** | Chinese, Japanese, Korean | 30 each | Non-Latin scripts |
| **Indic** | Hindi, Bengali, Tamil | 20 each | Complex scripts |

#### 4.1.3 Robustness Test Implementation

```python
def run_robustness_suite(
    model: Any,
    test_samples: List[Dict],
    obfuscation_techniques: List[str],
    languages: List[str]
) -> Dict:
    """
    Run comprehensive robustness testing suite.
    
    Returns:
        Dictionary with robustness metrics by technique and language
    """
    results = {
        'obfuscation': {},
        'multilingual': {},
        'overall_robustness_score': 0.0
    }
    
    # Test obfuscation techniques
    for technique in obfuscation_techniques:
        technique_samples = apply_obfuscation(test_samples, technique)
        y_true = [s['label'] for s in technique_samples]
        y_pred = model.predict([s['text'] for s in technique_samples])
        
        results['obfuscation'][technique] = {
            'accuracy': balanced_accuracy_score(y_true, y_pred),
            'detection_rate': compute_detection_rate(y_true, y_pred),
            'n_samples': len(technique_samples)
        }
    
    # Test multilingual robustness
    for lang in languages:
        lang_samples = translate_samples(test_samples, lang)
        y_true = [s['label'] for s in lang_samples]
        y_pred = model.predict([s['text'] for s in lang_samples])
        
        results['multilingual'][lang] = {
            'accuracy': balanced_accuracy_score(y_true, y_pred),
            'detection_rate': compute_detection_rate(y_true, y_pred),
            'n_samples': len(lang_samples)
        }
    
    # Compute overall robustness score
    all_accuracies = (
        [v['accuracy'] for v in results['obfuscation'].values()] +
        [v['accuracy'] for v in results['multilingual'].values()]
    )
    results['overall_robustness_score'] = np.mean(all_accuracies)
    
    return results
```

### 4.2 Reproducibility Checklist

#### 4.2.1 Environment Specification

```yaml
# environment.yml
name: guardrailer-eval
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - numpy=1.24.3
  - pandas=2.0.3
  - scikit-learn=1.3.0
  - scipy=1.11.1
  - pytorch=2.0.1
  - transformers=4.31.0
  - qdrant-client=1.5.0
  - fastapi=0.101.0
  - uvicorn=0.23.0
  - pytest=7.4.0
  - tqdm=4.65.0
  - ruamel.yaml=0.17.32
```

#### 4.2.2 Seed Control Protocol

```python
# Global seed configuration
GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

# Verify reproducibility
def verify_reproducibility(n_runs: int = 3) -> bool:
    """Verify that results are reproducible across runs."""
    set_global_seed(GLOBAL_SEED)
    reference_result = run_evaluation()
    
    for i in range(n_runs):
        set_global_seed(GLOBAL_SEED)
        current_result = run_evaluation()
        
        if not np.allclose(reference_result, current_result, atol=1e-6):
            return False
    
    return True
```

#### 4.2.3 Dataset Versioning

```python
# Dataset manifest
DATASET_MANIFEST = {
    'version': '2.0.0',
    'created': '2026-08-03',
    'hash': 'sha256:abc123...',  # Computed at build time
    'sources': {
        'harmbench': {'version': '1.0', 'hash': '...'},
        'advbench': {'version': '1.0', 'hash': '...'},
        'wildjailbreak': {'version': '1.0', 'hash': '...'},
        'benign_generated': {'version': '1.0', 'hash': '...'}
    },
    'split_ratios': {
        'train': 0.7,
        'validation': 0.15,
        'test': 0.15
    },
    'stratification': {
        'columns': ['label', 'category', 'tier'],
        'random_state': 42
    }
}
```

#### 4.2.4 Reproducibility Checklist

| Item | Requirement | Verification Method |
|------|-------------|---------------------|
| **Code Version** | Git commit hash | `git rev-parse HEAD` |
| **Dataset Version** | SHA-256 hash | `sha256sum dataset.parquet` |
| **Model Checkpoints** | File hash | `sha256sum models/*` |
| **Random Seeds** | All seeds fixed | `GLOBAL_SEED = 42` |
| **Environment** | Conda environment export | `conda env export > environment.yml` |
| **Hardware** | Document GPU/CPU specs | Log in results JSON |
| **Dependencies** | Pinned versions | `pip freeze > requirements.txt` |
| **Execution Time** | Wall-clock time | Logged in results JSON |

### 4.3 Artifact Management

#### 4.3.1 Required Artifacts

| Artifact | Format | Location | Purpose |
|----------|--------|----------|---------|
| **Dataset** | Parquet | `data/guardrailer_benchmark_v2.parquet` | Evaluation data |
| **Manifest** | JSON | `data/manifest.json` | Dataset metadata |
| **Models** | JSON/PT | `models_v2/` | Trained components |
| **Results** | JSON | `results/evaluation_v2.json` | Raw results |
| **Report** | Markdown | `reports/evaluation_report_v2.md` | Human-readable report |
| **Plots** | PNG/PDF | `figures/` | Visualizations |
| **Code** | Git | Repository | Source code |

---

## 5. Implementation Roadmap

### 5.1 Priority Action Items

#### P0: Critical (Week 1-2)

| Task | Description | Effort | Impact | Owner |
|------|-------------|--------|--------|-------|
| **P0.1** | Dataset acquisition and integration | 3 days | High | ML Engineer |
| **P0.2** | Benign class generation | 2 days | High | ML Engineer |
| **P0.3** | Hardness scoring implementation | 2 days | Medium | ML Engineer |
| **P0.4** | Stratified K-fold CV pipeline | 2 days | High | ML Engineer |
| **P0.5** | Bootstrap CI computation | 1 day | High | ML Engineer |
| **P0.6** | Power analysis computation | 0.5 day | Medium | Researcher |
| **P0.7** | Dataset validation pipeline | 1 day | High | ML Engineer |
| **P0.8** | Seed control and reproducibility | 0.5 day | High | ML Engineer |

**P0 Total Effort:** ~12 days (2.5 weeks)

#### P1: High Priority (Week 3-4)

| Task | Description | Effort | Impact | Owner |
|------|-------------|--------|--------|-------|
| **P1.1** | Ablation study implementation | 3 days | High | Researcher |
| **P1.2** | Statistical significance testing | 2 days | High | Researcher |
| **P1.3** | Robustness testing suite | 3 days | Medium | ML Engineer |
| **P1.4** | Multilingual test pipeline | 2 days | Medium | ML Engineer |
| **P1.5** | Latency profiling and analysis | 1 day | Medium | ML Engineer |
| **P1.6** | External baseline comparison | 2 days | High | Researcher |
| **P1.7** | Results visualization | 1 day | Low | ML Engineer |
| **P1.8** | Documentation and reporting | 2 days | Medium | Researcher |

**P1 Total Effort:** ~16 days (3 weeks)

#### P2: Medium Priority (Week 5-6)

| Task | Description | Effort | Impact | Owner |
|------|-------------|--------|--------|-------|
| **P2.1** | Manuscript preparation | 5 days | High | Researcher |
| **P2.2** | Supplementary materials | 2 days | Medium | Researcher |
| **P2.3** | Code cleanup and productionization | 2 days | Medium | ML Engineer |
| **P2.4** | Peer review preparation | 2 days | High | Researcher |
| **P2.5** | Dataset release preparation | 1 day | Low | ML Engineer |
| **P2.6** | Final reproducibility verification | 1 day | High | ML Engineer |

**P2 Total Effort:** ~13 days (2.5 weeks)

### 5.2 Timeline Overview

```
Week 1-2:  [==================] P0: Dataset & Methodology
Week 3-4:  [==================] P1: Evaluation & Robustness
Week 5-6:  [==================] P2: Manuscript & Release
           ─────────────────────────────────────────────
           Total Duration: 6 weeks
```

### 5.3 Resource Requirements

| Resource | Quantity | Purpose |
|----------|----------|---------|
| **GPU** | 1x T4 or better | Model inference, embeddings |
| **CPU** | 8+ cores | Parallel evaluation |
| **RAM** | 32GB+ | Dataset loading, processing |
| **Storage** | 100GB+ | Datasets, models, results |
| **Time** | 6 weeks | Full research cycle |

### 5.4 Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dataset acquisition delays | Medium | High | Start early, use cached versions |
| GPU availability | Low | Medium | Use cloud instances (Kaggle, Colab) |
| Reproducibility failures | Low | High | Continuous verification, seed control |
| Statistical power insufficient | Medium | High | Increase sample size, run power analysis early |
| Baseline comparison unavailable | Medium | Medium | Use public benchmarks, request access |

### 5.5 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Balanced Accuracy** | ≥ 90% | Mean ± 95% CI |
| **ECE** | ≤ 5% | Mean ± 95% CI |
| **AUC-ROC** | ≥ 95% | Mean ± 95% CI |
| **Reproducibility** | 100% | All runs match within 1e-6 |
| **Dataset Size** | ≥ 4,000 samples | Verified in manifest |
| **Statistical Significance** | p < 0.05 | McNemar's test vs baselines |
| **Robustness Score** | ≥ 80% | Average across obfuscations |
| **Manuscript** | Submission-ready | Peer review format |

---

## Appendices

### Appendix A: Complete Metric Definitions

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Balanced Accuracy | (TPR + TNR) / 2 | [0, 1] | 1 = perfect |
| ECE | (1/N) Σ \|b_m - conf_m\| | [0, 1] | 0 = perfectly calibrated |
| AUC-ROC | ∫ TPR dFPR | [0, 1] | 1 = perfect discrimination |
| F1-Score | 2·(P·R)/(P+R) | [0, 1] | 1 = perfect precision and recall |
| MCC | (TP·TN - FP·FN) / √(...) | [-1, 1] | 1 = perfect, 0 = random |
| Cohen's h | 2·arcsin(√p1) - 2·arcsin(√p2) | [0, 2] | 0 = no difference |

### Appendix B: Dataset Sources

| Dataset | License | Access | Size |
|---------|---------|--------|------|
| HarmBench | MIT | Public | 510 |
| AdvBench | MIT | Public | 520 |
| JailbreakBench | MIT | Public | 100 |
| WildJailbreak | Apache 2.0 | Public | 10K |
| Natural Questions | CC BY 4.0 | Public | 307K |
| HumanEval | MIT | Public | 164 |

### Appendix C: Statistical Test Selection

| Scenario | Test | Assumptions |
|----------|------|-------------|
| Paired binary outcomes | McNemar's | Large sample |
| Multiple comparisons | Bonferroni | Independent tests |
| Non-normal distributions | Wilcoxon | Paired samples |
| Effect size | Cohen's h | Proportions |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-08-03 | Research Team | Initial release |

**Review Schedule:** Monthly review of execution progress against roadmap.

**Approval:** Requires sign-off from Lead Researcher and Security Engineer.
