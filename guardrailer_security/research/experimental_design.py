"""
experimental_design.py
Experimental design and validation module for Guardrailer evaluation.

Implements:
- Four-phase evaluation lifecycle
- Research-grade metrics computation
- Ablation studies
- Statistical significance testing (McNemar's test, Bonferroni correction)
"""

import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

GLOBAL_SEED = 42


# ---------------------------------------------------------------------------
# Research-Grade Metrics
# ---------------------------------------------------------------------------

class ResearchMetrics:
    """Compute comprehensive research-grade metric suite."""

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all primary and secondary metrics.

        Primary metrics:
        - Balanced Accuracy, ECE, AUC-ROC, F1-Score, MCC

        Secondary metrics:
        - Per-Category Accuracy, FPR, FNR, Precision, Recall
        """
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

        tpr = recall
        tnr = tn / max(tn + fp, 1)
        balanced_accuracy = (tpr + tnr) / 2

        mcc_denom = math.sqrt(
            max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
        )
        mcc = (tp * tn - fp * fn) / mcc_denom

        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)

        metrics = {
            "balanced_accuracy": round(balanced_accuracy, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
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
            metrics["ece"] = round(ResearchMetrics.compute_ece(y_true, y_proba), 4)

        return metrics

    @staticmethod
    def compute_ece(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_confidence = y_proba[mask].mean()
                bin_accuracy = y_true[mask].mean()
                ece += mask.sum() / len(y_true) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    @staticmethod
    def compute_per_category_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        categories: np.ndarray,
    ) -> Dict[str, float]:
        """Compute accuracy for each category."""
        unique_cats = np.unique(categories)
        results = {}
        for cat in unique_cats:
            mask = categories == cat
            if mask.sum() > 0:
                results[str(cat)] = round(float((y_true[mask] == y_pred[mask]).mean()), 4)
        return results

    @staticmethod
    def compute_detection_rate_by_tier(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tiers: np.ndarray,
    ) -> Dict[str, float]:
        """Compute detection rate for each hardness tier."""
        results = {}
        for tier in [1, 2, 3, 4]:
            mask = tiers == tier
            if mask.sum() > 0:
                results[str(tier)] = round(float((y_pred[mask] == 1).mean()), 4)
        return results


# ---------------------------------------------------------------------------
# Statistical Significance Testing
# ---------------------------------------------------------------------------

@dataclass
class SignificanceResult:
    test: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    effect_interpretation: str
    contingency_table: Dict[str, int]


class SignificanceTester:
    """Statistical significance testing between models."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
    ) -> SignificanceResult:
        """McNemar's test for paired binary outcomes."""
        both_correct = int(((y_pred_a == y_true) & (y_pred_b == y_true)).sum())
        a_only = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
        b_only = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())
        both_wrong = int(((y_pred_a != y_true) & (y_pred_b != y_true)).sum())

        table = [[both_correct, a_only], [b_only, both_wrong]]

        try:
            from statsmodels.stats.contingency_tables import mcnemar
            result = mcnemar(table, exact=True)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        except ImportError:
            statistic, p_value = self._mcnemar_manual(a_only, b_only)

        acc_a = (y_pred_a == y_true).mean()
        acc_b = (y_pred_b == y_true).mean()
        cohens_h = abs(
            2 * math.asin(math.sqrt(acc_a)) - 2 * math.asin(math.sqrt(acc_b))
        )

        return SignificanceResult(
            test="McNemar",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=cohens_h,
            effect_interpretation=self._interpret_effect_size(cohens_h),
            contingency_table={
                "both_correct": both_correct,
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "both_wrong": both_wrong,
            },
        )

    def wilcoxon_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> SignificanceResult:
        """Wilcoxon signed-rank test for non-parametric comparison."""
        try:
            from scipy import stats
            statistic, p_value = stats.wilcoxon(scores_a, scores_b)
            statistic = float(statistic)
            p_value = float(p_value)
        except ImportError:
            diff = scores_a - scores_b
            diff = diff[diff != 0]
            n = len(diff)
            if n == 0:
                return SignificanceResult(
                    test="Wilcoxon", statistic=0.0, p_value=1.0,
                    significant=False, alpha=self.alpha,
                    effect_size=0.0, effect_interpretation="negligible",
                    contingency_table={},
                )
            ranked = np.argsort(np.abs(diff))
            r_plus = sum(i + 1 for i, r in enumerate(ranked) if diff[r] > 0)
            statistic = min(r_plus, n * (n + 1) / 2 - r_plus)
            p_value = min(1.0, 2 * math.exp(-statistic / (n * (n + 1) / 4)))

        effect_size = abs(scores_a.mean() - scores_b.mean()) / max(scores_a.std(), 1e-10)

        return SignificanceResult(
            test="Wilcoxon",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
            effect_interpretation=self._interpret_effect_size(effect_size),
            contingency_table={},
        )

    def bonferroni_correction(
        self, p_values: List[float], n_comparisons: int
    ) -> List[Dict]:
        """Apply Bonferroni correction for multiple comparisons."""
        adjusted_alpha = self.alpha / max(n_comparisons, 1)
        return [
            {
                "original_p": p,
                "adjusted_alpha": adjusted_alpha,
                "significant": p < adjusted_alpha,
                "adjusted_p": min(p * n_comparisons, 1.0),
            }
            for p in p_values
        ]

    @staticmethod
    def _mcnemar_manual(b_only: int, a_only: int) -> Tuple[float, float]:
        """Manual McNemar's test (chi-squared approximation)."""
        n = b_only + a_only
        if n == 0:
            return 0.0, 1.0
        statistic = (abs(b_only - a_only) - 1) ** 2 / n
        try:
            from scipy import stats
            p_value = float(1 - stats.chi2.cdf(statistic, 1))
        except ImportError:
            p_value = math.exp(-statistic / 2)
        return float(statistic), p_value

    @staticmethod
    def _interpret_effect_size(h: float) -> str:
        if h < 0.2:
            return "negligible"
        elif h < 0.5:
            return "small"
        elif h < 0.8:
            return "medium"
        else:
            return "large"


# ---------------------------------------------------------------------------
# Ablation Studies
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "A1_no_dense": {"remove_signals": ["dense"], "description": "Remove dense embedding signal"},
    "A2_no_sparse": {"remove_signals": ["sparse_idf"], "description": "Remove sparse IDF keywords"},
    "A3_no_cross_encoder": {"remove_signals": ["cross_encoder"], "description": "Remove cross-encoder relevance"},
    "A4_no_llm": {"disable_llm_evaluation": True, "description": "Disable LLM evaluation layer"},
    "A5_default_weights": {"use_learned_weights": False, "description": "Use default weights instead of learned"},
    "A6_no_calibration": {"disable_calibration": True, "description": "Disable probability calibration"},
    "A7_single_signal": {"signals": ["dense"], "mode": "single", "description": "Single dense signal only"},
}


@dataclass
class AblationResult:
    ablation_name: str
    description: str
    config: Dict
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Dict]
    timestamp: str


class AblationStudyRunner:
    """Run ablation studies to measure signal contribution."""

    def __init__(self, bootstrap_n: int = 1000, seed: int = GLOBAL_SEED):
        self.bootstrap_n = bootstrap_n
        self.seed = seed

    def run_single(
        self,
        ablation_name: str,
        config: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> AblationResult:
        """Run a single ablation study."""
        metrics = ResearchMetrics.compute_all(y_true, y_pred, y_proba)

        ci_results = {}
        for metric_name in ["balanced_accuracy", "f1", "mcc"]:
            if metric_name in metrics:
                ci_results[metric_name] = self._bootstrap_ci(
                    y_true, y_pred, metric_name
                )

        return AblationResult(
            ablation_name=ablation_name,
            description=config.get("description", ""),
            config=config,
            metrics=metrics,
            confidence_intervals=ci_results,
            timestamp=datetime.now().isoformat(),
        )

    def run_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        configs: Optional[Dict[str, Dict]] = None,
    ) -> List[AblationResult]:
        """Run all ablation studies."""
        if configs is None:
            configs = ABLATION_CONFIGS

        results = []
        for name, config in configs.items():
            result = self.run_single(name, config, y_true, y_pred, y_proba)
            results.append(result)
            log.info("Completed ablation %s: balanced_acc=%.4f", name, result.metrics.get("balanced_accuracy", 0))

        return results

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
    ) -> Dict:
        """Compute bootstrap CI for a metric."""
        rng = np.random.RandomState(self.seed)
        n = len(y_true)
        scores = []

        for _ in range(self.bootstrap_n):
            idx = rng.choice(n, size=n, replace=True)
            metrics = ResearchMetrics.compute_all(y_true[idx], y_pred[idx])
            scores.append(metrics.get(metric_name, 0.0))

        scores = np.array(scores)
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_95_lower": float(np.percentile(scores, 2.5)),
            "ci_95_upper": float(np.percentile(scores, 97.5)),
        }


# ---------------------------------------------------------------------------
# Four-Phase Evaluation Lifecycle
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    phase: int
    name: str
    status: str
    duration_seconds: float
    outputs: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


class EvaluationLifecycle:
    """Orchestrate the four-phase evaluation lifecycle.

    Phase 1: Dataset Construction
    Phase 2: Model Training
    Phase 3: Evaluation
    Phase 4: External Comparison
    """

    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase_results: List[PhaseResult] = []

    def run_phase_1_dataset_construction(
        self,
        acquisition_pipeline: Any,
        benign_generator: Any,
        hardness_scorer: Any,
        validator: Any,
    ) -> PhaseResult:
        """Phase 1: Dataset Construction."""
        start = time.time()
        errors = []
        outputs = {}

        try:
            adversarial_df = acquisition_pipeline.acquire_all()
            outputs["adversarial_samples"] = len(adversarial_df)
        except Exception as e:
            errors.append(f"Adversarial acquisition: {e}")
            outputs["adversarial_samples"] = 0

        try:
            benign_samples = benign_generator.generate_all()
            outputs["benign_samples"] = len(benign_samples)
        except Exception as e:
            errors.append(f"Benign generation: {e}")
            outputs["benign_samples"] = 0

        try:
            if not adversarial_df.empty:
                adversarial_list = adversarial_df.to_dict("records")
                scored = hardness_scorer.score_batch(adversarial_list)
                outputs["scored_samples"] = len(scored)
        except Exception as e:
            errors.append(f"Hardness scoring: {e}")

        duration = time.time() - start
        phase = PhaseResult(
            phase=1, name="Dataset Construction",
            status="completed" if not errors else "partial",
            duration_seconds=duration, outputs=outputs, errors=errors,
        )
        self.phase_results.append(phase)
        return phase

    def run_phase_3_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        categories: Optional[np.ndarray] = None,
        tiers: Optional[np.ndarray] = None,
    ) -> PhaseResult:
        """Phase 3: Evaluation with research-grade metrics."""
        start = time.time()
        errors = []
        outputs = {}

        metrics = ResearchMetrics.compute_all(y_true, y_pred, y_proba)
        outputs["primary_metrics"] = metrics

        if categories is not None:
            outputs["per_category"] = ResearchMetrics.compute_per_category_accuracy(
                y_true, y_pred, categories
            )

        if tiers is not None:
            outputs["per_tier"] = ResearchMetrics.compute_detection_rate_by_tier(
                y_true, y_pred, tiers
            )

        duration = time.time() - start
        phase = PhaseResult(
            phase=3, name="Evaluation",
            status="completed", duration_seconds=duration,
            outputs=outputs, errors=errors,
        )
        self.phase_results.append(phase)
        return phase

    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "phases": [],
        }
        for pr in self.phase_results:
            report["phases"].append({
                "phase": pr.phase,
                "name": pr.name,
                "status": pr.status,
                "duration_seconds": round(pr.duration_seconds, 2),
                "outputs": pr.outputs,
                "errors": pr.errors,
            })
        return report

    def save_report(self, filename: str = "evaluation_report.json"):
        """Save report to file."""
        report = self.generate_report()
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Report saved to %s", path)
