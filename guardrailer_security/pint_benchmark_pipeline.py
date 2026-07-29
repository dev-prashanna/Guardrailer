"""
pint_benchmark_pipeline.py
PINT Benchmark integration for Guardrailer Security Engine.

Evaluates the Guardrailer prompt injection detection system against the
Lakera PINT Benchmark dataset format. Produces balanced accuracy scores,
per-category breakdowns, and latency statistics.

Usage:
    python pint_benchmark_pipeline.py \\
        --url http://localhost:8090 \\
        --dataset ./pint_dataset.yaml \\
        --output pint_report.json

    # Use the built-in example dataset:
    python pint_benchmark_pipeline.py --url http://localhost:8090

    # Request access to the full PINT dataset from support@lakera.ai
    # then point to it:
    python pint_benchmark_pipeline.py --dataset /path/to/pint-benchmark-dataset.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Dataset format (YAML)
# ---------------------------------------------------------------------------
# Each entry:
#   - text: "input string"
#     category: "prompt_injection" | "jailbreak" | "hard_negatives" | "chat" | "documents"
#     label: true | false   (true = contains prompt injection)

DEFAULT_DATASET_PATH = str(Path(__file__).resolve().parent / "pint_dataset.yaml")


def load_dataset_yaml(path: str) -> list[dict]:
    """Load a PINT-format YAML dataset file."""
    try:
        from ruamel.yaml import YAML
    except ImportError:
        try:
            from yaml import safe_load
            with open(path) as f:
                data = safe_load(f)
            return data if isinstance(data, list) else []
        except ImportError:
            log.error("Neither ruamel.yaml nor PyYAML is installed. Install with: pip install 'ruamel.yaml<0.18.0'")
            sys.exit(1)

    yaml = YAML()
    with open(path) as f:
        data = yaml.load(f)
    return data if isinstance(data, list) else []


def dataset_to_records(data: list[dict]) -> list[dict]:
    """Normalise dataset entries to flat dicts."""
    records = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", "")
        category = entry.get("category", "unknown")
        label = entry.get("label", False)
        if text:
            records.append({"text": str(text), "category": str(category), "label": bool(label)})
    return records


# ---------------------------------------------------------------------------
# Guardrailer evaluation function
# ---------------------------------------------------------------------------

def make_guardrailer_evaluator(
    url: str = "http://localhost:8090",
    timeout: float = 30.0,
) -> Callable[[str], bool]:
    """
    Return an evaluation function that calls the Guardrailer Security Engine.

    The returned function accepts a prompt string and returns True if the
    engine flags it as malicious (prompt injection detected).
    """

    def evaluate(prompt: str) -> bool:
        payload = json.dumps({"query": prompt}).encode("utf-8")
        req = urllib.request.Request(
            f"{url}/v1/evaluate-prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return bool(result.get("is_blocked", False))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Guardrailer API error HTTP {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Guardrailer API error: {e}") from e

    return evaluate


# ---------------------------------------------------------------------------
# Benchmark evaluation logic
# ---------------------------------------------------------------------------

@dataclass
class CategoryResult:
    category: str
    label: bool
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class BenchmarkReport:
    model_name: str
    total_inputs: int
    score_balanced: float
    score_imbalanced: float
    category_results: list[CategoryResult]
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    per_input: list[dict]
    test_date: str


def evaluate_dataset(
    records: list[dict],
    eval_fn: Callable[[str], bool],
    desc: str = "Evaluating",
) -> tuple[list[dict], list[float]]:
    """
    Run eval_fn on every record and return (results, latencies).
    Each result dict adds 'prediction' and 'correct' keys.
    """
    import tqdm

    results = []
    latencies = []

    for i, rec in enumerate(tqdm.tqdm(records, total=len(records), desc=desc)):
        text = rec["text"]
        t0 = time.time()
        try:
            prediction = eval_fn(text)
            latency = (time.time() - t0) * 1000
        except Exception as e:
            latency = (time.time() - t0) * 1000
            prediction = False
            log.error("  [%d/%d] Error: %s", i + 1, len(records), e)

        correct = prediction == rec["label"]
        results.append({
            **rec,
            "prediction": prediction,
            "correct": correct,
            "latency_ms": round(latency, 2),
        })
        latencies.append(latency)

    return results, latencies


def compute_balanced_score(results: list[dict]) -> float:
    """
    Balanced accuracy: average of per-label accuracies.
    For each unique label value (True/False), compute accuracy, then average.
    """
    labels = set(r["label"] for r in results)
    per_label = {}
    for lbl in labels:
        subset = [r for r in results if r["label"] == lbl]
        if subset:
            correct = sum(1 for r in subset if r["correct"])
            per_label[lbl] = correct / len(subset)
        else:
            per_label[lbl] = 0.0

    return sum(per_label.values()) / len(per_label) if per_label else 0.0


def compute_imbalanced_score(results: list[dict]) -> float:
    """Overall accuracy without balancing."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def build_report(
    model_name: str,
    records: list[dict],
    results: list[dict],
    latencies: list[float],
    weight: Literal["balanced", "imbalanced"] = "balanced",
) -> BenchmarkReport:
    """Aggregate results into a BenchmarkReport."""

    score_balanced = compute_balanced_score(results)
    score_imbalanced = compute_imbalanced_score(results)

    # Per-category breakdown
    cat_map: dict[str, dict[str, dict]] = {}
    for r in results:
        cat = r["category"]
        lbl = r["label"]
        if cat not in cat_map:
            cat_map[cat] = {}
        if lbl not in cat_map[cat]:
            cat_map[cat][lbl] = {"total": 0, "correct": 0}
        cat_map[cat][lbl]["total"] += 1
        if r["correct"]:
            cat_map[cat][lbl]["correct"] += 1

    category_results = []
    for cat in sorted(cat_map.keys()):
        for lbl in sorted(cat_map[cat].keys(), reverse=True):
            d = cat_map[cat][lbl]
            category_results.append(CategoryResult(
                category=cat,
                label=lbl,
                total=d["total"],
                correct=d["correct"],
            ))

    sorted_lat = sorted(latencies)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    from datetime import date
    test_date = date.today().isoformat()

    return BenchmarkReport(
        model_name=model_name,
        total_inputs=len(results),
        score_balanced=score_balanced,
        score_imbalanced=score_imbalanced,
        category_results=category_results,
        latency_ms_avg=round(avg_lat, 2),
        latency_ms_p50=round(percentile(sorted_lat, 0.50), 2),
        latency_ms_p95=round(percentile(sorted_lat, 0.95), 2),
        latency_ms_p99=round(percentile(sorted_lat, 0.99), 2),
        per_input=results,
        test_date=test_date,
    )


def print_report(report: BenchmarkReport, weight: Literal["balanced", "imbalanced"] = "balanced"):
    score = report.score_balanced if weight == "balanced" else report.score_imbalanced

    print()
    print("=" * 70)
    print("  PINT BENCHMARK - Guardrailer Security Engine")
    print("=" * 70)
    print()
    print(f"  Model: {report.model_name}")
    print(f"  Score ({weight}): {score * 100:.4f}%")
    print(f"  Total inputs: {report.total_inputs}")
    print(f"  Test date: {report.test_date}")
    print()
    print("  --- Category Breakdown ---")
    print(f"  {'Category':<20s} {'Label':<8s} {'Correct':<10s} {'Total':<8s} {'Accuracy':<10s}")
    print("  " + "-" * 56)
    for cr in report.category_results:
        print(f"  {cr.category:<20s} {str(cr.label):<8s} {cr.correct:<10d} {cr.total:<8d} {cr.accuracy * 100:.2f}%")
    print()
    print("  --- Latency ---")
    print(f"  Average:  {report.latency_ms_avg:.1f} ms")
    print(f"  P50:      {report.latency_ms_p50:.1f} ms")
    print(f"  P95:      {report.latency_ms_p95:.1f} ms")
    print(f"  P99:      {report.latency_ms_p99:.1f} ms")
    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pint_benchmark(
    dataset_path: str,
    url: str = "http://localhost:8090",
    output_path: str = "pint_report.json",
    weight: Literal["balanced", "imbalanced"] = "balanced",
    quiet: bool = False,
) -> BenchmarkReport:
    """
    Run the PINT Benchmark on the Guardrailer Security Engine.

    Args:
        dataset_path: Path to a PINT-format YAML dataset file.
        url: Base URL of the Guardrailer Security Engine.
        output_path: Path to write the JSON report.
        weight: Scoring mode ('balanced' or 'imbalanced').
        quiet: Suppress printing of results.

    Returns:
        BenchmarkReport with scores and per-input results.
    """
    log.info("Loading dataset from %s ...", dataset_path)
    raw_data = load_dataset_yaml(dataset_path)
    records = dataset_to_records(raw_data)
    log.info("Loaded %d records.", len(records))

    if not records:
        log.error("Dataset is empty. Check the YAML file format.")
        sys.exit(1)

    evaluator = make_guardrailer_evaluator(url=url)

    log.info("Running PINT benchmark against Guardrailer at %s ...", url)
    results, latencies = evaluate_dataset(records, evaluator, desc="PINT Benchmark")

    report = build_report(
        model_name="Guardrailer Security Engine",
        records=records,
        results=results,
        latencies=latencies,
        weight=weight,
    )

    if not quiet:
        print_report(report, weight=weight)

    # Save JSON report
    output = {
        "summary": {
            "model_name": report.model_name,
            "score_balanced": round(report.score_balanced * 100, 4),
            "score_imbalanced": round(report.score_imbalanced * 100, 4),
            "total_inputs": report.total_inputs,
            "test_date": report.test_date,
            "latency_avg_ms": report.latency_ms_avg,
            "latency_p50_ms": report.latency_ms_p50,
            "latency_p95_ms": report.latency_ms_p95,
            "latency_p99_ms": report.latency_ms_p99,
            "category_breakdown": [
                {
                    "category": cr.category,
                    "label": cr.label,
                    "correct": cr.correct,
                    "total": cr.total,
                    "accuracy": round(cr.accuracy * 100, 4),
                }
                for cr in report.category_results
            ],
        },
        "results": report.per_input,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Report saved to %s", output_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="PINT Benchmark for Guardrailer Security Engine"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GUARDRAILER_URL", "http://localhost:8090"),
        help="Guardrailer Security Engine base URL (default: http://localhost:8090)",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET_PATH", DEFAULT_DATASET_PATH),
        help="Path to PINT-format YAML dataset (default: built-in example dataset)",
    )
    parser.add_argument(
        "--output",
        default="pint_report.json",
        help="Output JSON report path (default: pint_report.json)",
    )
    parser.add_argument(
        "--weight",
        choices=["balanced", "imbalanced"],
        default="balanced",
        help="Scoring weight mode (default: balanced)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress printed output")
    args = parser.parse_args()

    pint_benchmark(
        dataset_path=args.dataset,
        url=args.url,
        output_path=args.output,
        weight=args.weight,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
