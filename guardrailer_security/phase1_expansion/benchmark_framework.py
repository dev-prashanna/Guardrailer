"""
benchmark_framework.py
Comprehensive benchmarking framework for Guardrailer evaluation.

Evaluates the guardrail system across multiple dimensions:
  - Per-category accuracy (10 attack categories)
  - Per-technique accuracy (20+ attack techniques)
  - Per-source accuracy (research datasets vs synthetic)
  - Per-language accuracy (English vs multilingual)
  - Per-difficulty accuracy (easy/medium/hard/adversarial)
  - False positive/negative analysis
  - Latency percentiles
  - Layer distribution
  - Generalization metrics (cross-source, cross-language)
  - Robustness metrics (obfuscation resistance, encoding invariance)

Produces structured JSON reports and human-readable summaries.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional

from .schema import BenchmarkEntry, SampleRecord, compute_stats, DatasetStats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    entry_id: str
    text: str
    expected: bool
    predicted: bool
    correct: bool
    latency_ms: float
    composite_score: Optional[float] = None
    layer: Optional[str] = None
    category: str = ""
    technique: str = ""
    risk_level: str = ""
    source: str = ""
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    difficulty_score: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class CategoryMetrics:
    category: str
    total: int = 0
    correct: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class DimensionMetrics:
    dimension: str
    groups: dict[str, CategoryMetrics] = field(default_factory=dict)

    @property
    def macro_accuracy(self) -> float:
        values = [cm.accuracy for cm in self.groups.values() if cm.total > 0]
        return sum(values) / len(values) if values else 0.0

    @property
    def macro_f1(self) -> float:
        values = [cm.f1 for cm in self.groups.values() if cm.total > 0]
        return sum(values) / len(values) if values else 0.0

    @property
    def weighted_accuracy(self) -> float:
        total = sum(cm.total for cm in self.groups.values())
        if total == 0:
            return 0.0
        return sum(cm.accuracy * cm.total for cm in self.groups.values()) / total


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    report_name: str
    timestamp: str
    engine_url: str

    # Overall metrics
    total_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    balanced_accuracy: float = 0.0

    # Per-dimension breakdowns
    per_category: DimensionMetrics = field(default_factory=lambda: DimensionMetrics("category"))
    per_technique: DimensionMetrics = field(default_factory=lambda: DimensionMetrics("technique"))
    per_source: DimensionMetrics = field(default_factory=lambda: DimensionMetrics("source"))
    per_language: DimensionMetrics = field(default_factory=lambda: DimensionMetrics("language"))
    per_difficulty: DimensionMetrics = field(default_factory=lambda: DimensionMetrics("difficulty"))

    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Layer distribution
    layer_distribution: dict = field(default_factory=dict)

    # Generalization metrics
    generalization_gap: float = 0.0
    worst_category_accuracy: float = 0.0
    worst_category_name: str = ""
    most_confused_category: str = ""

    # Robustness
    obfuscation_accuracy: float = 0.0
    multilingual_accuracy: float = 0.0
    unicode_accuracy: float = 0.0

    # Per-sample results
    results: list[dict] = field(default_factory=list)

    # Dataset stats
    dataset_stats: Optional[dict] = None

    def _serialize_dimension(self, dm) -> dict:
        """Manually serialize DimensionMetrics to include computed properties."""
        groups_out = {}
        for name, cm in dm.groups.items():
            groups_out[name] = {
                "total": cm.total,
                "correct": cm.correct,
                "true_positives": cm.true_positives,
                "false_positives": cm.false_positives,
                "true_negatives": cm.true_negatives,
                "false_negatives": cm.false_negatives,
                "accuracy": round(cm.accuracy, 4),
                "precision": round(cm.precision, 4),
                "recall": round(cm.recall, 4),
                "f1": round(cm.f1, 4),
            }
        return {
            "dimension": dm.dimension,
            "macro_accuracy": round(dm.macro_accuracy, 4),
            "macro_f1": round(dm.macro_f1, 4),
            "weighted_accuracy": round(dm.weighted_accuracy, 4),
            "groups": groups_out,
        }

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k in ("per_category", "per_technique", "per_source", "per_language", "per_difficulty"):
                d[k] = self._serialize_dimension(getattr(self, k))
            elif k == "results":
                d[k] = v
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class GuardrailerBenchmark:
    """
    Benchmark runner for the Guardrailer Security Engine.
    
    Supports:
    - Direct API evaluation (calls /v1/evaluate-prompt)
    - Mock evaluation (for testing the framework itself)
    - Multi-dimensional analysis
    - Report generation
    """

    def __init__(
        self,
        engine_url: str = "http://localhost:8090",
        timeout: float = 30.0,
    ):
        self.engine_url = engine_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # API evaluation
    # ------------------------------------------------------------------

    def call_engine(self, text: str) -> dict:
        """Call the Guardrailer Security Engine API."""
        payload = json.dumps({"query": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.engine_url}/v1/evaluate-prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def evaluate_entry(self, entry: BenchmarkEntry) -> EvalResult:
        """Evaluate a single benchmark entry against the engine."""
        t0 = time.time()
        try:
            response = self.call_engine(entry.text)
            latency = (time.time() - t0) * 1000

            predicted = bool(response.get("is_blocked", False))
            entry.predicted_label = predicted
            entry.is_correct = predicted == entry.expected_label
            entry.prediction_latency_ms = response.get("latency_ms", latency)
            entry.composite_score = response.get("composite_score")
            entry.layer = response.get("layer")

            return EvalResult(
                entry_id=entry.id,
                text=entry.text[:200],
                expected=entry.expected_label,
                predicted=predicted,
                correct=entry.is_correct,
                latency_ms=entry.prediction_latency_ms,
                composite_score=entry.composite_score,
                layer=entry.layer,
                category=entry.category,
                technique=entry.technique,
                risk_level=entry.risk_level,
                source=entry.source,
                language=entry.language,
                tags=list(entry.tags),
                difficulty_score=entry.difficulty_score,
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return EvalResult(
                entry_id=entry.id,
                text=entry.text[:200],
                expected=entry.expected_label,
                predicted=False,
                correct=False,
                latency_ms=latency,
                category=entry.category,
                technique=entry.technique,
                risk_level=entry.risk_level,
                source=entry.source,
                language=entry.language,
                tags=list(entry.tags),
                difficulty_score=entry.difficulty_score,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        entries: list[BenchmarkEntry],
        report_name: str = "phase1_benchmark",
    ) -> BenchmarkReport:
        """
        Run the full benchmark on a set of entries.
        
        Args:
            entries: List of BenchmarkEntry objects to evaluate.
            report_name: Name for the report.
        
        Returns:
            BenchmarkReport with all metrics.
        """
        log.info("=" * 70)
        log.info("Guardrailer Phase 1 Benchmark")
        log.info("=" * 70)
        log.info("Evaluating %d samples ...", len(entries))

        # Evaluate all entries
        results = []
        for i, entry in enumerate(entries):
            result = self.evaluate_entry(entry)
            results.append(result)
            if (i + 1) % 50 == 0:
                correct_so_far = sum(1 for r in results if r.correct)
                log.info("  [%d/%d] Running accuracy: %.2f%%",
                         i + 1, len(entries),
                         100 * correct_so_far / len(results))

        # Build report
        report = self._build_report(report_name, entries, results)

        log.info("Benchmark complete. Accuracy: %.2f%%, F1: %.2f%%",
                 report.accuracy * 100, report.f1_score * 100)

        return report

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_report(
        self,
        report_name: str,
        entries: list[BenchmarkEntry],
        results: list[EvalResult],
    ) -> BenchmarkReport:
        """Build a comprehensive benchmark report from evaluation results."""

        report = BenchmarkReport(
            report_name=report_name,
            timestamp=datetime.utcnow().isoformat(),
            engine_url=self.engine_url,
            total_samples=len(results),
        )

        # Overall metrics
        tp = sum(1 for r in results if r.expected and r.predicted)
        fp = sum(1 for r in results if not r.expected and r.predicted)
        tn = sum(1 for r in results if not r.expected and not r.predicted)
        fn = sum(1 for r in results if r.expected and not r.predicted)

        report.accuracy = (tp + tn) / len(results) if results else 0.0
        report.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        report.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        report.f1_score = (
            2 * report.precision * report.recall / (report.precision + report.recall)
            if (report.precision + report.recall) > 0 else 0.0
        )

        # Balanced accuracy
        mal_results = [r for r in results if r.expected]
        ben_results = [r for r in results if not r.expected]
        mal_acc = sum(1 for r in mal_results if r.correct) / len(mal_results) if mal_results else 0.0
        ben_acc = sum(1 for r in ben_results if r.correct) / len(ben_results) if ben_results else 0.0
        report.balanced_accuracy = (mal_acc + ben_acc) / 2

        # Per-dimension breakdowns
        report.per_category = self._compute_dimension(results, "category")
        report.per_technique = self._compute_dimension(results, "technique")
        report.per_source = self._compute_dimension(results, "source")
        report.per_language = self._compute_dimension(results, "language")
        report.per_difficulty = self._compute_difficulty_dimension(results)

        # Latency
        latencies = [r.latency_ms for r in results]
        report.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies)
        report.p50_latency_ms = self._percentile(sorted_lat, 0.50)
        report.p95_latency_ms = self._percentile(sorted_lat, 0.95)
        report.p99_latency_ms = self._percentile(sorted_lat, 0.99)

        # Layer distribution
        layer_counts = defaultdict(int)
        for r in results:
            layer_counts[r.layer or "unknown"] += 1
        report.layer_distribution = dict(layer_counts)

        # Generalization metrics
        report.worst_category_accuracy = min(
            (cm.accuracy for cm in report.per_category.groups.values() if cm.total > 0),
            default=0.0,
        )
        report.worst_category_name = min(
            (name for name, cm in report.per_category.groups.items() if cm.total > 0),
            key=lambda n: report.per_category.groups[n].accuracy,
            default="none",
        )

        # Generalization gap: difference between best and worst source accuracy
        source_accs = [cm.accuracy for cm in report.per_source.groups.values() if cm.total > 0]
        if source_accs:
            report.generalization_gap = max(source_accs) - min(source_accs)

        # Robustness metrics
        obf_results = [r for r in results if "unicode_trick" in r.tags or "obfuscated" in r.source.lower()]
        report.obfuscation_accuracy = (
            sum(1 for r in obf_results if r.correct) / len(obf_results)
            if obf_results else 0.0
        )

        ml_results = [r for r in results if r.language != "en"]
        report.multilingual_accuracy = (
            sum(1 for r in ml_results if r.correct) / len(ml_results)
            if ml_results else 0.0
        )

        unicode_results = [r for r in results if "unicode_trick" in r.tags]
        report.unicode_accuracy = (
            sum(1 for r in unicode_results if r.correct) / len(unicode_results)
            if unicode_results else 0.0
        )

        # Per-sample results
        report.results = [
            {
                "id": r.entry_id,
                "text": r.text,
                "expected": r.expected,
                "predicted": r.predicted,
                "correct": r.correct,
                "latency_ms": round(r.latency_ms, 2),
                "composite_score": r.composite_score,
                "layer": r.layer,
                "category": r.category,
                "technique": r.technique,
                "risk_level": r.risk_level,
                "source": r.source,
                "language": r.language,
                "tags": r.tags,
                "difficulty_score": r.difficulty_score,
                "error": r.error,
            }
            for r in results
        ]

        # Dataset stats from entries
        records = [e.to_dict() for e in entries]
        # Minimal dataset stats
        report.dataset_stats = {
            "total": len(entries),
            "malicious": sum(1 for e in entries if e.expected_label),
            "benign": sum(1 for e in entries if not e.expected_label),
        }

        return report

    def _compute_dimension(
        self,
        results: list[EvalResult],
        dimension: str,
    ) -> DimensionMetrics:
        """Compute per-group metrics for a given dimension."""
        groups: dict[str, CategoryMetrics] = defaultdict(lambda: CategoryMetrics(category=""))

        for r in results:
            key = getattr(r, dimension, "unknown")
            cm = groups[key]
            cm.category = key
            cm.total += 1
            if r.correct:
                cm.correct += 1
            if r.expected and r.predicted:
                cm.true_positives += 1
            elif not r.expected and r.predicted:
                cm.false_positives += 1
            elif not r.expected and not r.predicted:
                cm.true_negatives += 1
            elif r.expected and not r.predicted:
                cm.false_negatives += 1

        return DimensionMetrics(dimension=dimension, groups=dict(groups))

    def _compute_difficulty_dimension(
        self,
        results: list[EvalResult],
    ) -> DimensionMetrics:
        """Compute per-difficulty-bucket metrics."""
        buckets = {
            "easy (0-0.3)": [],
            "medium (0.3-0.6)": [],
            "hard (0.6-0.8)": [],
            "adversarial (0.8-1.0)": [],
        }

        for r in results:
            d = r.difficulty_score
            if d < 0.3:
                buckets["easy (0-0.3)"].append(r)
            elif d < 0.6:
                buckets["medium (0.3-0.6)"].append(r)
            elif d < 0.8:
                buckets["hard (0.6-0.8)"].append(r)
            else:
                buckets["adversarial (0.8-1.0)"].append(r)

        groups = {}
        for name, bucket_results in buckets.items():
            if not bucket_results:
                continue
            cm = CategoryMetrics(category=name)
            cm.total = len(bucket_results)
            cm.correct = sum(1 for r in bucket_results if r.correct)
            cm.true_positives = sum(1 for r in bucket_results if r.expected and r.predicted)
            cm.false_positives = sum(1 for r in bucket_results if not r.expected and r.predicted)
            cm.true_negatives = sum(1 for r in bucket_results if not r.expected and not r.predicted)
            cm.false_negatives = sum(1 for r in bucket_results if r.expected and not r.predicted)
            groups[name] = cm

        return DimensionMetrics(dimension="difficulty", groups=groups)

    def _percentile(self, sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(len(sorted_vals) * p)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_benchmark_report(report: BenchmarkReport):
    """Print a human-readable benchmark report."""
    print()
    print("=" * 72)
    print("  GUARDRAILER PHASE 1 BENCHMARK REPORT")
    print("=" * 72)
    print(f"  Report: {report.report_name}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"  Engine: {report.engine_url}")
    print()

    print(f"  Total Samples: {report.total_samples}")
    print()

    print("  --- Overall Metrics ---")
    print(f"  Accuracy:          {report.accuracy:.4f} ({report.accuracy * 100:.2f}%)")
    print(f"  Balanced Accuracy: {report.balanced_accuracy:.4f} ({report.balanced_accuracy * 100:.2f}%)")
    print(f"  Precision:         {report.precision:.4f}")
    print(f"  Recall:            {report.recall:.4f}")
    print(f"  F1 Score:          {report.f1_score:.4f}")
    print()

    print("  --- Per-Category Breakdown ---")
    print(f"  {'Category':<35s} {'Total':>6s} {'Acc':>8s} {'P':>8s} {'R':>8s} {'F1':>8s}")
    print("  " + "-" * 73)
    for name, cm in sorted(report.per_category.groups.items(), key=lambda x: x[1].accuracy):
        print(f"  {name:<35s} {cm.total:>6d} {cm.accuracy:>8.4f} {cm.precision:>8.4f} {cm.recall:>8.4f} {cm.f1:>8.4f}")
    print()

    print("  --- Per-Technique Breakdown ---")
    print(f"  {'Technique':<35s} {'Total':>6s} {'Acc':>8s}")
    print("  " + "-" * 50)
    for name, cm in sorted(report.per_technique.groups.items(), key=lambda x: x[1].accuracy):
        print(f"  {name:<35s} {cm.total:>6d} {cm.accuracy:>8.4f}")
    print()

    print("  --- Per-Source Breakdown ---")
    print(f"  {'Source':<35s} {'Total':>6s} {'Acc':>8s}")
    print("  " + "-" * 50)
    for name, cm in sorted(report.per_source.groups.items(), key=lambda x: x[1].accuracy):
        print(f"  {name:<35s} {cm.total:>6d} {cm.accuracy:>8.4f}")
    print()

    print("  --- Per-Language Breakdown ---")
    print(f"  {'Language':<35s} {'Total':>6s} {'Acc':>8s}")
    print("  " + "-" * 50)
    for name, cm in sorted(report.per_language.groups.items(), key=lambda x: x[1].accuracy):
        print(f"  {name:<35s} {cm.total:>6d} {cm.accuracy:>8.4f}")
    print()

    print("  --- Per-Difficulty Breakdown ---")
    print(f"  {'Difficulty':<35s} {'Total':>6s} {'Acc':>8s}")
    print("  " + "-" * 50)
    for name, cm in report.per_difficulty.groups.items():
        print(f"  {name:<35s} {cm.total:>6d} {cm.accuracy:>8.4f}")
    print()

    print("  --- Robustness Metrics ---")
    print(f"  Obfuscation Accuracy:    {report.obfuscation_accuracy:.4f}")
    print(f"  Multilingual Accuracy:   {report.multilingual_accuracy:.4f}")
    print(f"  Unicode Accuracy:        {report.unicode_accuracy:.4f}")
    print()

    print("  --- Generalization ---")
    print(f"  Generalization Gap:      {report.generalization_gap:.4f}")
    print(f"  Worst Category:          {report.worst_category_name} ({report.worst_category_accuracy:.4f})")
    print()

    print("  --- Latency ---")
    print(f"  Average:  {report.avg_latency_ms:.1f} ms")
    print(f"  P50:      {report.p50_latency_ms:.1f} ms")
    print(f"  P95:      {report.p95_latency_ms:.1f} ms")
    print(f"  P99:      {report.p99_latency_ms:.1f} ms")
    print()

    print("  --- Layer Distribution ---")
    for layer, count in sorted(report.layer_distribution.items()):
        print(f"    {layer:<20s} {count}")
    print()

    # False positive/negative analysis
    fps = [r for r in report.results if not r["expected"] and r["predicted"]]
    fns = [r for r in report.results if r["expected"] and not r["predicted"]]

    if fps:
        print(f"  --- Top False Positives ({len(fps)} total) ---")
        for fp in fps[:5]:
            print(f"    [{fp['category']}] {fp['text'][:80]}...")
        print()

    if fns:
        print(f"  --- Top False Negatives ({len(fns)} total) ---")
        for fn in fns[:5]:
            print(f"    [{fn['category']}] {fn['text'][:80]}...")
        print()

    print("=" * 72)


def save_benchmark_report(report: BenchmarkReport, path: str):
    """Save the benchmark report as JSON."""
    output = report.to_dict()
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Report saved to %s", path)
