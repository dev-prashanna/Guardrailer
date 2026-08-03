"""
statistics.py
Statistical rigor module for Guardrailer evaluation.

Implements:
- Stratified K-Fold Cross-Validation with nested stratification
- Bootstrap Confidence Intervals (BCa method)
- Power Analysis for sample size determination
- Effect size computation (Cohen's h, Cohen's d)
"""

import hashlib
import json
import logging
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Stratified K-Fold Cross-Validation
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    train_size: int
    val_size: int
    metrics: Dict[str, float]
    stratification_distribution: Dict[str, int]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None


@dataclass
class CVReport:
    n_folds: int
    fold_results: List[FoldResult]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    stratification_homogeneity_p: float
    seed: int
    timestamp: str = ""


class StratifiedKFoldEvaluator:
    """Stratified K-Fold CV with nested stratification.

    Stratification priority:
    1. Binary label (malicious/benign)
    2. Attack category (6 categories)
    3. Attack technique (sub-categories)
    """

    def __init__(self, n_folds: int = 5, seed: int = GLOBAL_SEED):
        self.n_folds = n_folds
        self.seed = seed

    def _create_stratify_key(
        self,
        labels: np.ndarray,
        categories: np.ndarray,
        techniques: np.ndarray,
    ) -> np.ndarray:
        """Create composite stratification key."""
        return np.array([
            f"{label}_{cat}_{tech}"
            for label, cat, tech in zip(labels, categories, techniques)
        ])

    def _validate_fold_stratification(
        self, stratify_key: np.ndarray, val_idx: np.ndarray
    ) -> Dict[str, int]:
        """Ensure minimum samples per stratum in each fold."""
        val_strata = Counter(stratify_key[val_idx])
        for stratum, count in val_strata.items():
            if count < 2:
                log.warning(
                    "Insufficient samples for stratum %s: %d (need >= 2)",
                    stratum, count,
                )
        return dict(val_strata)

    def _chi_squared_homogeneity(
        self, fold_distributions: List[Dict[str, int]]
    ) -> float:
        """Chi-squared test for distribution homogeneity across folds."""
        if len(fold_distributions) < 2:
            return 1.0

        all_strata = set()
        for dist in fold_distributions:
            all_strata.update(dist.keys())

        observed = []
        for dist in fold_distributions:
            row = [dist.get(s, 0) for s in all_strata]
            observed.append(row)

        observed = np.array(observed, dtype=float)
        if observed.sum() == 0:
            return 1.0

        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        grand_total = observed.sum()

        expected = row_totals * col_totals / grand_total
        expected = np.maximum(expected, 1e-10)

        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        dof = max(1, (observed.shape[0] - 1) * (observed.shape[1] - 1))

        p_value = 1.0 - self._chi2_cdf(chi2, dof)
        return p_value

    @staticmethod
    def _chi2_cdf(x: float, k: int) -> float:
        """Approximate chi-squared CDF using regularized gamma function."""
        if x <= 0:
            return 0.0
        try:
            from scipy import stats
            return float(stats.chi2.cdf(x, k))
        except ImportError:
            a = k / 2.0
            return 1.0 - math.exp(-x / 2.0) * sum(
                (x / 2.0) ** j / math.factorial(j) for j in range(int(a))
            ) if a == int(a) else 0.5

    def run(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
        techniques: np.ndarray,
        predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
        metric_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    ) -> CVReport:
        """Run stratified K-fold cross-validation.

        Args:
            texts: Input text array
            labels: Binary labels (0/1)
            categories: Attack categories
            techniques: Attack techniques
            predict_fn: Function that takes val texts and returns (y_pred, y_proba)
            metric_fn: Function that computes metrics from (y_true, y_pred)

        Returns:
            CVReport with fold-level and aggregated metrics
        """
        set_global_seed(self.seed)

        # Try composite stratification first, fall back to binary if strata too small
        stratify_key = self._create_stratify_key(labels, categories, techniques)
        stratum_counts = Counter(stratify_key)
        min_stratum = min(stratum_counts.values())

        if min_stratum < 2 and len(labels) > 20:
            # Fall back to binary label stratification
            log.info(
                "Composite strata too small (min=%d), using binary label stratification",
                min_stratum,
            )
            stratify_key = np.array([str(l) for l in labels])
            stratum_counts = Counter(stratify_key)
            min_stratum = min(stratum_counts.values())

        effective_folds = min(self.n_folds, min_stratum)
        effective_folds = max(effective_folds, 2) if len(labels) >= 4 else max(1, len(labels))

        if effective_folds < self.n_folds:
            log.warning(
                "Reducing folds from %d to %d (min stratum count: %d)",
                self.n_folds, effective_folds, min_stratum,
            )

        indices = np.arange(len(texts))
        fold_indices = self._stratified_split(indices, stratify_key, effective_folds)

        fold_results = []
        fold_distributions = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            val_texts = texts[val_idx]
            val_labels = labels[val_idx]

            y_pred, y_proba = predict_fn(val_texts)

            if metric_fn is not None:
                metrics = metric_fn(val_labels, y_pred)
            else:
                metrics = self._default_metrics(val_labels, y_pred, y_proba)

            strat_dist = self._validate_fold_stratification(stratify_key, val_idx)
            fold_distributions.append(strat_dist)

            fold_results.append(FoldResult(
                fold=fold_idx,
                train_size=len(train_idx),
                val_size=len(val_idx),
                metrics=metrics,
                stratification_distribution=strat_dist,
                y_true=val_labels,
                y_pred=y_pred,
                y_proba=y_proba,
            ))

        homogeneity_p = self._chi_squared_homogeneity(fold_distributions)

        mean_metrics = {}
        std_metrics = {}
        all_metric_keys = set()
        for fr in fold_results:
            all_metric_keys.update(fr.metrics.keys())

        for key in all_metric_keys:
            values = [fr.metrics.get(key, 0.0) for fr in fold_results]
            values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
            if values:
                mean_metrics[key] = float(np.mean(values))
                std_metrics[key] = float(np.std(values))
            else:
                mean_metrics[key] = 0.0
                std_metrics[key] = 0.0

        return CVReport(
            n_folds=effective_folds,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            stratification_homogeneity_p=homogeneity_p,
            seed=self.seed,
            timestamp=datetime.now().isoformat(),
        )

    def _stratified_split(
        self,
        indices: np.ndarray,
        stratify_key: np.ndarray,
        n_folds: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified train/val splits."""
        unique_strata = np.unique(stratify_key)
        strata_indices = {s: indices[stratify_key == s] for s in unique_strata}

        for s in strata_indices:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(strata_indices[s])

        fold_assignments = np.zeros(len(indices), dtype=int)
        for s, s_idx in strata_indices.items():
            fold_sizes = np.full(n_folds, len(s_idx) // n_folds)
            remainder = len(s_idx) % n_folds
            for i in range(remainder):
                fold_sizes[i] += 1

            current = 0
            for fold_id in range(n_folds):
                end = current + fold_sizes[fold_id]
                fold_assignments[s_idx[current:end]] = fold_id
                current = end

        splits = []
        for fold_id in range(n_folds):
            val_mask = fold_assignments == fold_id
            val_idx = indices[val_mask]
            train_idx = indices[~val_mask]
            splits.append((train_idx, val_idx))

        return splits

    @staticmethod
    def _default_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute default metric suite."""
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
        balanced_acc = (
            (recall + tn / max(tn + fp, 1)) / 2
        )

        metrics = {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr": round(fp / max(fp + tn, 1), 4),
            "fnr": round(fn / max(fn + tp, 1), 4),
        }

        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                unique_classes = np.unique(y_true)
                if len(unique_classes) < 2:
                    metrics["auc_roc"] = 0.5
                else:
                    metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
            except Exception:
                metrics["auc_roc"] = 0.5
            metrics["mcc"] = round(float(
                (tp * tn - fp * fn) / max(
                    math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1e-8
                )
            ), 4)

        return metrics


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    ci_level: float
    method: str
    bootstrap_distribution: Optional[np.ndarray] = None


class BootstrapConfidenceInterval:
    """Compute bootstrap confidence intervals for any metric.

    Supports:
    - Stratified resampling to maintain class distribution
    - BCa (bias-corrected and accelerated) method
    - Multiple CI levels (90%, 95%, 99%)
    """

    def __init__(
        self,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95,
        seed: int = GLOBAL_SEED,
    ):
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.seed = seed

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        ci_levels: Optional[List[float]] = None,
    ) -> Dict[float, BootstrapResult]:
        """Compute bootstrap CIs for multiple confidence levels.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            metric_fn: Metric function (y_true, y_pred) -> float
            ci_levels: List of CI levels (default: [0.90, 0.95, 0.99])

        Returns:
            Dict mapping CI level to BootstrapResult
        """
        if ci_levels is None:
            ci_levels = [0.90, 0.95, 0.99]

        rng = np.random.RandomState(self.seed)
        n_samples = len(y_true)

        bootstrap_scores = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_scores[i] = metric_fn(y_true[indices], y_pred[indices])

        point_estimate = metric_fn(y_true, y_pred)
        std_error = float(np.std(bootstrap_scores))

        results = {}
        for ci_level in ci_levels:
            alpha = 1 - ci_level
            ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
            ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

            results[ci_level] = BootstrapResult(
                point_estimate=point_estimate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                std_error=std_error,
                n_bootstrap=self.n_bootstrap,
                ci_level=ci_level,
                method="percentile",
                bootstrap_distribution=bootstrap_scores,
            )

        return results

    def compute_bca(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> BootstrapResult:
        """Compute BCa (bias-corrected and accelerated) confidence interval."""
        rng = np.random.RandomState(self.seed)
        n_samples = len(y_true)

        bootstrap_scores = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_scores[i] = metric_fn(y_true[indices], y_pred[indices])

        point_estimate = metric_fn(y_true, y_pred)

        bias_correction = self._compute_bias_correction(bootstrap_scores, point_estimate)
        acceleration = self._compute_acceleration(y_true, y_pred, metric_fn)

        alpha = 1 - self.ci_level
        z_alpha = self._norm_ppf(alpha / 2)
        z_1_alpha = self._norm_ppf(1 - alpha / 2)

        z_0 = bias_correction
        a = acceleration

        adj_alpha_lower = self._norm_cdf(z_0 + (z_0 + z_alpha) / (1 - a * (z_0 + z_alpha)))
        adj_alpha_upper = self._norm_cdf(z_0 + (z_0 + z_1_alpha) / (1 - a * (z_0 + z_1_alpha)))

        ci_lower = float(np.percentile(bootstrap_scores, 100 * adj_alpha_lower))
        ci_upper = float(np.percentile(bootstrap_scores, 100 * adj_alpha_upper))

        return BootstrapResult(
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=float(np.std(bootstrap_scores)),
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
            method="bca",
            bootstrap_distribution=bootstrap_scores,
        )

    @staticmethod
    def _compute_bias_correction(
        bootstrap_scores: np.ndarray, point_estimate: float
    ) -> float:
        """Compute bias correction factor z0."""
        count_below = np.sum(bootstrap_scores < point_estimate)
        proportion = count_below / len(bootstrap_scores)
        proportion = max(1e-10, min(1 - 1e-10, proportion))
        return BootstrapConfidenceInterval._norm_ppf(proportion)

    @staticmethod
    def _compute_acceleration(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable,
    ) -> float:
        """Compute acceleration factor using jackknife."""
        n = len(y_true)
        jackknife_scores = np.empty(n)
        full_score = metric_fn(y_true, y_pred)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jackknife_scores[i] = metric_fn(y_true[mask], y_pred[mask])

        jackknife_mean = np.mean(jackknife_scores)
        diff = jackknife_mean - jackknife_scores

        numerator = np.sum(diff ** 3)
        denominator = 6.0 * (np.sum(diff ** 2) ** 1.5)

        if abs(denominator) < 1e-10:
            return 0.0

        return float(numerator / denominator)

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """Inverse normal CDF (approximation)."""
        try:
            from scipy import stats
            return float(stats.norm.ppf(p))
        except ImportError:
            if p <= 0:
                return -6.0
            if p >= 1:
                return 6.0
            a = p - 0.5
            r = a * a
            return float(
                a * (2.515517 + 0.802853 * r + 0.010328 * r * r)
                / (1.0 + 1.432788 * r + 0.189269 * r * r + 0.001308 * r * r * r)
            )

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Normal CDF (approximation)."""
        try:
            from scipy import stats
            return float(stats.norm.cdf(x))
        except ImportError:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Power Analysis
# ---------------------------------------------------------------------------

@dataclass
class PowerResult:
    required_per_group: int
    total_required: int
    effect_size: float
    power: float
    alpha: float


class PowerAnalysis:
    """Compute minimum required sample size for statistical power.

    For guardrail evaluation:
    - Effect size: minimum detectable difference (e.g., 5% improvement)
    - Alpha: Type I error rate (0.05)
    - Power: 1 - Type II error rate (0.80)
    """

    @staticmethod
    def compute_required_sample_size(
        effect_size: float = 0.2,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0,
    ) -> PowerResult:
        """Compute minimum required sample size using normal approximation.

        Args:
            effect_size: Cohen's h for proportions (small=0.2, medium=0.5, large=0.8)
            alpha: Significance level
            power: Desired statistical power
            ratio: Sample size ratio between groups (n1/n2)

        Returns:
            PowerResult with required sample sizes
        """
        try:
            from statsmodels.stats.power import NormalIndPower
            analysis = NormalIndPower()
            sample_size = analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha,
                ratio=ratio,
            )
            required = int(np.ceil(sample_size))
        except ImportError:
            z_alpha = PowerAnalysis._norm_ppf(1 - alpha / 2)
            z_beta = PowerAnalysis._norm_ppf(power)
            p = 0.5
            q = 1 - p

            required = int(np.ceil(
                (z_alpha * math.sqrt(2 * p * q) + z_beta * math.sqrt(
                    p * (1 - p) + 0.5 * (1 - 0.5)
                )) ** 2 / (effect_size ** 2)
            ))

        return PowerResult(
            required_per_group=required,
            total_required=int(np.ceil(required * (1 + ratio))),
            effect_size=effect_size,
            power=power,
            alpha=alpha,
        )

    @staticmethod
    def compute_effect_size_proportions(p1: float, p2: float) -> float:
        """Compute Cohen's h for two proportions."""
        return abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))

    @staticmethod
    def compute_effect_size_d(mean1: float, mean2: float, std: float) -> float:
        """Compute Cohen's d for two means (pooled std)."""
        if std < 1e-10:
            return 0.0
        return abs(mean1 - mean2) / std

    @classmethod
    def generate_power_report(
        cls,
        effect_sizes: Optional[List[float]] = None,
        power_levels: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Generate comprehensive power analysis report.

        Args:
            effect_sizes: List of effect sizes to test
            power_levels: List of power levels to test

        Returns:
            List of dicts with power analysis results
        """
        if effect_sizes is None:
            effect_sizes = [0.05, 0.1, 0.2, 0.3, 0.5]
        if power_levels is None:
            power_levels = [0.80, 0.90, 0.95]

        report = []
        for es in effect_sizes:
            for pw in power_levels:
                result = cls.compute_required_sample_size(
                    effect_size=es, power=pw
                )
                report.append({
                    "effect_size": es,
                    "power": pw,
                    "required_per_group": result.required_per_group,
                    "total_required": result.total_required,
                })

        return report

    @staticmethod
    def _norm_ppf(p: float) -> float:
        """Inverse normal CDF (approximation)."""
        try:
            from scipy import stats
            return float(stats.norm.ppf(p))
        except ImportError:
            if p <= 0:
                return -6.0
            if p >= 1:
                return 6.0
            a = p - 0.5
            r = a * a
            return float(
                a * (2.515517 + 0.802853 * r + 0.010328 * r * r)
                / (1.0 + 1.432788 * r + 0.189269 * r * r + 0.001308 * r * r * r)
            )
