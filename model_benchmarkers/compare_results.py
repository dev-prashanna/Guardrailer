"""
Research-grade comparison analysis across all benchmarked models.
Generates detailed reports with statistical significance and per-category breakdowns.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from .evaluation import compare_models, print_comparison


RESULTS_DIR = Path(__file__).parent / "results"


def load_latest_results(results_dir=RESULTS_DIR):
    json_files = sorted(results_dir.glob("benchmark_*.json"))
    if not json_files:
        raise FileNotFoundError("No benchmark results found.")
    latest = json_files[-1]
    print(f"Loading: {latest.name}")
    return json.load(open(latest))


def compare_saved_results(results_dir=RESULTS_DIR):
    all_results = load_latest_results(results_dir)
    comparison = print_comparison(all_results)
    return comparison


def generate_report(results_dir=RESULTS_DIR):
    json_files = sorted(results_dir.glob("benchmark_*.json"))
    if not json_files:
        print("No results found.")
        return

    all_results = []
    for f in json_files:
        results = json.load(open(f))
        all_results.extend(results)

    if not all_results:
        print("No results to compare.")
        return

    comparison = compare_models(all_results)

    print("\n" + "=" * 120)
    print("RESEARCH BENCHMARK REPORT")
    print("=" * 120)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models evaluated: {len(all_results)}")
    print(f"Sample size: {all_results[0].get('total_samples', 'N/A')}")
    print("=" * 120)
    print(comparison.to_string(index=False))
    print("=" * 120)

    # Statistical comparison table
    if len(all_results) > 1:
        print("\n\nSTATISTICAL SUMMARY:")
        print("-" * 120)
        print(f"{'Metric':<25} ", end="")
        for r in all_results:
            print(f"{r['label']:<20} ", end="")
        print()
        print("-" * 120)

        metrics = [
            ("Accuracy", "accuracy", "accuracy_ci95"),
            ("Precision", "precision", "precision_ci95"),
            ("Recall", "recall", "recall_ci95"),
            ("F1 Binary", "f1_binary", None),
            ("F1 Macro", "f1_macro", None),
            ("Specificity", "specificity", None),
            ("MCC", "matthews_corrcoef", None),
            ("Kappa", "cohen_kappa", None),
            ("AUC-ROC", "auc_roc", "auc_roc_ci95"),
            ("FPR (lower=better)", "false_positive_rate", None),
            ("FNR (lower=better)", "false_negative_rate", None),
        ]

        for metric_name, key, ci_key in metrics:
            print(f"{metric_name:<25} ", end="")
            for r in all_results:
                val = r.get(key, "N/A")
                ci = r.get(ci_key) if ci_key else None
                if ci:
                    print(f"{val:.4f} [{ci[0]:.3f}-{ci[1]:.3f}]  ", end="")
                elif val is not None and val != "N/A":
                    print(f"{val:<20} ", end="")
                else:
                    print(f"{'N/A':<20} ", end="")
            print()

        print("-" * 120)

        # Best model ranking
        print("\n\nMODEL RANKING:")
        print("-" * 120)
        df = comparison
        for metric in ["F1 Binary", "Accuracy", "MCC", "AUC-ROC"]:
            if metric in df.columns:
                valid = df[df[metric] != ""]
                if not valid.empty:
                    best_idx = valid[metric].idxmax()
                    best_model = valid.loc[best_idx, "Model"]
                    best_val = valid.loc[best_idx, metric]
                    print(f"  Best {metric:<20}: {best_model} ({best_val})")
        print("-" * 120)

    # Per-category breakdown
    per_category_reports = {}
    for r in all_results:
        if "per_category" in r:
            per_category_reports[r["label"]] = r["per_category"]

    if per_category_reports:
        print("\n\nPER-CATEGORY BREAKDOWN:")
        print("=" * 120)

        all_cats = set()
        for cats in per_category_reports.values():
            all_cats.update(cats.keys())

        for cat in sorted(all_cats):
            print(f"\n  Category: {cat}")
            print(f"  {'Model':<25} {'Samples':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
                  f"{'F1':>8} {'Spec':>8} {'FPR':>8} {'FNR':>8}")
            print(f"  {'-'*95}")
            for model_name, cats in per_category_reports.items():
                if cat in cats:
                    m = cats[cat]
                    print(f"  {model_name:<25} {m['samples']:>8} {m['accuracy']:>8.4f} "
                          f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                          f"{m['specificity']:>8.4f} {m['fpr']:>8.4f} {m['fnr']:>8.4f}")

    # Latency comparison
    latency_data = []
    for r in all_results:
        if "latency" in r:
            lat = r["latency"]
            latency_data.append({
                "Model": r["label"],
                "Mean (ms)": lat["mean_ms"],
                "Median (ms)": lat["median_ms"],
                "P95 (ms)": lat["p95_ms"],
                "P99 (ms)": lat["p99_ms"],
                "Std (ms)": lat["std_ms"],
                "Prompts/s": lat["prompts_per_second"],
                "Total (s)": lat["total_seconds"],
            })

    if latency_data:
        print("\n\nLATENCY COMPARISON:")
        print("-" * 120)
        lat_df = pd.DataFrame(latency_data).sort_values("P95 (ms)")
        print(lat_df.to_string(index=False))
        print("-" * 120)

    # Save full report
    report_path = RESULTS_DIR / f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write("RESEARCH BENCHMARK REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(comparison.to_string())
        f.write("\n\n")
        for r in all_results:
            f.write(f"\n{r['label']}:\n")
            f.write(json.dumps(r, indent=2))
            f.write("\n")
    print(f"\nFull report saved: {report_path}")

    return comparison


if __name__ == "__main__":
    generate_report()
