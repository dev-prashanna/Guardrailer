"""
Main benchmark runner for research-grade evaluation.
Loads sample, runs each model with latency tracking, evaluates with full metrics, saves results.
"""

import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

from .models import load_model
from .evaluation import evaluate_model, print_comparison


CONFIG_PATH = Path(__file__).parent / "configs" / "models.json"
RESULTS_DIR = Path(__file__).parent / "results"
SAMPLE_DIR = Path(__file__).parent / "samples"


class BenchmarkRunner:
    def __init__(self, config_path=CONFIG_PATH):
        self.config = json.load(open(config_path))
        self.benchmark_cfg = self.config["benchmark"]
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_sample(self, sample_path=None):
        if sample_path is None:
            samples = sorted(SAMPLE_DIR.glob("benchmark_sample_*.parquet"))
            if not samples:
                raise FileNotFoundError("No sample found. Run sample_dataset.py first.")
            sample_path = samples[-1]

        self.sample = pd.read_parquet(sample_path)
        self.texts = self.sample["prompt_text"].astype(str).tolist()
        self.y_true = self.sample["is_malicious"].astype(int).tolist()
        self.categories = self.sample["attack_category"]
        print(f"Loaded sample: {len(self.texts):,} prompts from {Path(sample_path).name}")
        return self

    def run_model(self, model_name):
        model_cfg = self.config["models"][model_name]
        model = load_model(model_cfg)

        n = len(self.texts)
        print(f"\n{'='*70}")
        print(f"  Model: {model_cfg['name']}")
        print(f"  Type:  {model_cfg['type']}")
        print(f"  Samples: {n:,}")
        print(f"{'='*70}")

        start_time = time.time()

        def progress(done, total):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"\r  [{done}/{n}] {done/n*100:.0f}% | {rate:.1f}/s | ETA {eta:.0f}s", end="", flush=True)

        y_pred = model.predict(self.texts, progress_callback=progress)
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s ({n/elapsed:.1f} prompts/s)")

        latencies = model.get_latencies()

        results = evaluate_model(
            self.y_true, y_pred,
            categories=self.categories,
            latencies=latencies,
            label=model_cfg["name"],
        )
        results["elapsed_seconds"] = round(elapsed, 2)
        results["model_config"] = model_name

        self._print_model_summary(results)
        return results

    def run_guardrailer(self):
        """Run Guardrailer scoring pipeline for comparison."""
        print(f"\n{'='*70}")
        print(f"  Model: Guardrailer")
        print(f"  Type:  custom_scoring_pipeline")
        print(f"  Samples: {len(self.texts):,}")
        print(f"{'='*70}")

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "guardrailer_security"))
            from scoring import SecurityScorer

            scorer = SecurityScorer()
            y_pred = []
            latencies = []
            for i, text in enumerate(self.texts):
                start = time.time()
                result = scorer.score(text)
                latencies.append(time.time() - start)
                y_pred.append(result["is_malicious"])
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start if i == 0 else elapsed
                    print(f"\r  [{i+1}/{len(self.texts)}]", end="", flush=True)
            print()

            results = evaluate_model(
                self.y_true, y_pred,
                categories=self.categories,
                latencies=latencies,
                label="Guardrailer",
            )
            self._print_model_summary(results)
            return results

        except Exception as e:
            print(f"  Guardrailer error: {e}")
            return None

    def _print_model_summary(self, results):
        cm = results["confusion_matrix"]
        lat = results.get("latency", {})
        print(f"\n  Results:")
        print(f"    Accuracy:  {results['accuracy']:.4f}  (95% CI: {results['accuracy_ci95']})")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall:    {results['recall']:.4f}")
        print(f"    F1 Binary: {results['f1_binary']:.4f}")
        print(f"    F1 Macro:  {results['f1_macro']:.4f}")
        print(f"    MCC:       {results['matthews_corrcoef']:.4f}")
        print(f"    Kappa:     {results['cohen_kappa']:.4f}")
        print(f"    Specificity: {results['specificity']:.4f}")
        print(f"    AUC-ROC:   {results.get('auc_roc', 'N/A')}")
        print(f"    Confusion: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
        if lat:
            print(f"    Latency:   mean={lat['mean_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms  "
                  f"p99={lat['p99_ms']:.1f}ms  {lat['prompts_per_second']:.1f}/s")

    def run_all(self, include_guardrailer=True):
        all_results = []

        for model_name in self.config["models"]:
            try:
                result = self.run_model(model_name)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR running {model_name}: {e}")

        if include_guardrailer:
            gr_result = self.run_guardrailer()
            if gr_result:
                all_results.append(gr_result)

        if all_results:
            comparison = print_comparison(all_results)
            self._save_results(all_results, comparison)

        return all_results

    def _save_results(self, all_results, comparison):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_path = self.results_dir / f"benchmark_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        comparison_path = self.results_dir / f"comparison_{timestamp}.csv"
        comparison.to_csv(comparison_path, index=False)

        print(f"\nResults saved:")
        print(f"  {results_path}")
        print(f"  {comparison_path}")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.load_sample()
    runner.run_all()
