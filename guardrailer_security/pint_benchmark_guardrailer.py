#!/usr/bin/env python3
"""
Guardrailer PINT Benchmark Adapter

Tests Guardrailer against the Lakera PINT Benchmark dataset.
Requires Guardrailer engine running locally.

Usage:
    # 1. Start Guardrailer engine
    python3 guardrailer_security/security_engine.py

    # 2. Run benchmark
    python3 guardrailer_security/pint_benchmark_guardrailer.py

    # Or with custom settings:
    python3 guardrailer_security/pint_benchmark_guardrailer.py \
        --url http://localhost:8090 \
        --dataset /path/to/pint-dataset.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import pandas as pd
import tqdm

# Add pint-benchmark to path
PINT_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "pint-benchmark"
if PINT_BENCHMARK_DIR.exists():
    sys.path.insert(0, str(PINT_BENCHMARK_DIR / "benchmark"))

try:
    from ruamel.yaml import YAML
except ImportError:
    print("Installing ruamel.yaml ...")
    os.system(f"{sys.executable} -m pip install -q 'ruamel.yaml<0.18.0'")
    from ruamel.yaml import YAML


def evaluate_guardrailer(prompt: str, url: str = "http://localhost:8090", timeout: float = 5.0) -> bool:
    """
    Send a prompt to Guardrailer and return True if flagged as malicious.
    """
    try:
        resp = requests.post(
            f"{url}/v1/evaluate-prompt",
            json={"query": prompt},
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"  [WARN] Guardrailer returned {resp.status_code}: {resp.text[:100]}")
            return False
        data = resp.json()
        return data.get("is_blocked", False)
    except requests.exceptions.ConnectionError:
        print("  [ERROR] Cannot connect to Guardrailer. Is it running?")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def evaluate_dataset(df: pd.DataFrame, eval_fn, desc: str = "Evaluating") -> pd.DataFrame:
    """Run eval_fn on each row, return accuracy metrics per category/label."""
    df = df.copy()
    df["prediction"] = False

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=desc):
        df.at[i, "prediction"] = eval_fn(prompt=str(row["text"]))

    df["correct"] = df["prediction"] == df["label"]

    return (
        df.groupby(["category", "label"])
        .agg({"correct": ["mean", "sum", "count"]})
        .droplevel(0, axis=1)
        .rename(columns={"mean": "accuracy", "sum": "correct", "count": "total"})
    )


def pint_benchmark(
    df: pd.DataFrame,
    model_name: str,
    eval_fn,
    weight: str = "balanced",
) -> tuple:
    """Run PINT benchmark and print results."""
    benchmark = evaluate_dataset(df, eval_fn, desc=f"Testing {model_name}")

    if weight == "imbalanced":
        score = benchmark["correct"].sum() / benchmark["total"].sum()
    else:
        score = float(
            benchmark.groupby("label")
            .agg({"total": "sum", "correct": "sum"})
            .assign(accuracy=lambda x: x["correct"] / x["total"])["accuracy"]
            .mean()
        )

    print("\n" + "=" * 60)
    print("PINT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Score:    {round(score * 100, 4)}% ({weight})")
    print(f"Date:     {pd.to_datetime('today').strftime('%Y-%m-%d')}")
    print("=" * 60)
    print(benchmark)
    print("=" * 60)

    return (model_name, score, benchmark)


def load_dataset(path: str) -> pd.DataFrame:
    """Load YAML dataset into DataFrame."""
    yaml = YAML()
    data = yaml.load(Path(path))
    return pd.DataFrame.from_records(data)


def main():
    parser = argparse.ArgumentParser(description="Run PINT benchmark on Guardrailer")
    parser.add_argument("--url", default="http://localhost:8090", help="Guardrailer API URL")
    parser.add_argument("--dataset", default=None, help="Path to PINT dataset YAML")
    parser.add_argument("--timeout", type=float, default=5.0, help="Request timeout in seconds")
    parser.add_argument("--weight", default="balanced", choices=["balanced", "imbalanced"])
    args = parser.parse_args()

    # Find dataset
    dataset_path = args.dataset
    if not dataset_path:
        # Try common locations
        candidates = [
            PINT_BENCHMARK_DIR / "benchmark" / "data" / "pint-benchmark-dataset.yaml",
            PINT_BENCHMARK_DIR / "benchmark" / "data" / "example-dataset.yaml",
        ]
        for c in candidates:
            if c.exists():
                dataset_path = str(c)
                break

    if not dataset_path or not os.path.exists(dataset_path):
        print("Error: No dataset found.")
        print("Download the PINT dataset or use --dataset flag.")
        print(f"Example dataset: {PINT_BENCHMARK_DIR / 'benchmark' / 'data' / 'example-dataset.yaml'}")
        sys.exit(1)

    print(f"Dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Loaded {len(df)} samples")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Test connection
    print(f"\nConnecting to Guardrailer at {args.url} ...")
    try:
        resp = requests.get(f"{args.url}/health", timeout=3)
        if resp.status_code == 200:
            print("  Connected!")
        else:
            print(f"  Warning: health check returned {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure Guardrailer is running: python3 security_engine.py")
        sys.exit(1)

    # Create eval function with URL
    def eval_fn(prompt):
        return evaluate_guardrailer(prompt, url=args.url, timeout=args.timeout)

    # Run benchmark
    start = time.time()
    model_name, score, results = pint_benchmark(
        df=df,
        model_name="Guardrailer",
        eval_fn=eval_fn,
        weight=args.weight,
    )
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.1f}s ({len(df)/elapsed:.0f} prompts/sec)")

    # Save results
    results_path = Path(__file__).resolve().parent / "pint_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "score": score,
            "score_pct": round(score * 100, 4),
            "dataset": dataset_path,
            "weight": args.weight,
            "timestamp": pd.to_datetime("today").isoformat(),
            "total_samples": len(df),
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
