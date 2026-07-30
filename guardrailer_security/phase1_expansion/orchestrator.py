"""
orchestrator.py
Phase 1 Dataset Expansion orchestrator.

Ties together research dataset collection, synthetic generation,
edge case injection, and benchmarking into a single pipeline.

Usage:
    # Full pipeline (collect + generate + benchmark)
    python -m phase1_expansion.orchestrator --mode full

    # Collect only
    python -m phase1_expansion.orchestrator --mode collect

    # Benchmark only (loads saved dataset)
    python -m phase1_expansion.orchestrator --mode benchmark --dataset phase1_dataset.parquet

    # Generate benchmark YAML
    python -m phase1_expansion.orchestrator --mode yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import (
    SampleRecord,
    BenchmarkEntry,
    compute_stats,
)
from .research_collectors import collect_all_research_datasets
from .synthetic_generator import SyntheticAttackGenerator
from .edge_cases import EdgeCaseGenerator
from .benchmark_framework import (
    GuardrailerBenchmark,
    print_benchmark_report,
    save_benchmark_report,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent
PHASE1_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Phase1Orchestrator:
    """
    Orchestrates the Phase 1 Dataset Expansion pipeline.
    
    Steps:
    1. Collect from research datasets (PINT, Anthropic, OpenAI, WildJailbreak, PromptInject)
    2. Generate synthetic attacks (novel variants, paraphrases, multi-turn, obfuscated)
    3. Generate edge cases (ambiguous, multi-language, Unicode, context-dependent)
    4. Deduplicate and balance
    5. Export to parquet and YAML formats
    6. Run benchmark evaluation
    """

    def __init__(
        self,
        seed: int = 42,
        engine_url: str = "http://localhost:8090",
        output_dir: Optional[str] = None,
    ):
        self.seed = seed
        self.engine_url = engine_url
        self.output_dir = Path(output_dir) if output_dir else PHASE1_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.synthetic_gen = SyntheticAttackGenerator(seed=seed)
        self.edge_case_gen = EdgeCaseGenerator(seed=seed)
        self.benchmark = GuardrailerBenchmark(engine_url=engine_url)

    # ------------------------------------------------------------------
    # Step 1: Collect research datasets
    # ------------------------------------------------------------------

    def collect_research_datasets(
        self,
        include_pint: bool = True,
        include_wild_jailbreak: bool = True,
        include_anthropic: bool = True,
        include_openai: bool = True,
        include_prompt_inject: bool = True,
        pint_yaml_path: Optional[str] = None,
    ) -> list[SampleRecord]:
        """Collect samples from all specified research datasets."""
        return collect_all_research_datasets(
            pint_yaml_path=pint_yaml_path,
            include_wild_jailbreak=include_wild_jailbreak,
            include_anthropic=include_anthropic,
            include_openai=include_openai,
            include_prompt_inject=include_prompt_inject,
            include_pint=include_pint,
        )

    # ------------------------------------------------------------------
    # Step 2: Generate synthetic attacks
    # ------------------------------------------------------------------

    def generate_synthetic(
        self,
        seed_texts: Optional[list[str]] = None,
        novel_per_category: int = 100,
        paraphrase_count: int = 50,
        multi_turn_count: int = 30,
        obfuscation_count: int = 40,
    ) -> list[SampleRecord]:
        """Generate synthetic attacks."""
        return self.synthetic_gen.generate_all(
            novel_per_category=novel_per_category,
            paraphrase_count=paraphrase_count,
            multi_turn_count=multi_turn_count,
            obfuscation_count=obfuscation_count,
            seed_texts=seed_texts,
        )

    # ------------------------------------------------------------------
    # Step 3: Generate edge cases
    # ------------------------------------------------------------------

    def generate_edge_cases(
        self,
        ambiguous_count: int = 40,
        multi_lang_count: int = 60,
        unicode_count: int = 50,
        context_count: int = 30,
    ) -> list[SampleRecord]:
        """Generate edge case samples."""
        return self.edge_case_gen.generate_all(
            ambiguous_count=ambiguous_count,
            multi_lang_count=multi_lang_count,
            unicode_count=unicode_count,
            context_count=context_count,
        )

    # ------------------------------------------------------------------
    # Step 4: Deduplicate and balance
    # ------------------------------------------------------------------

    @staticmethod
    def deduplicate(records: list[SampleRecord]) -> list[SampleRecord]:
        """Deduplicate records by normalized text hash."""
        seen = set()
        deduped = []
        for r in records:
            normalized = hashlib.md5(r.text.lower().strip().encode()).hexdigest()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(r)
        return deduped

    @staticmethod
    def balance(
        records: list[SampleRecord],
        target_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> list[SampleRecord]:
        """
        Balance the dataset across categories.
        
        If target_per_category is None, uses the median category count.
        Oversamples minority classes, undersamples majority classes.
        """
        rng = random.Random(seed)
        by_category: dict[str, list[SampleRecord]] = {}
        for r in records:
            by_category.setdefault(r.attack_category, []).append(r)

        if target_per_category is None:
            counts = [len(v) for v in by_category.values()]
            target_per_category = int(sorted(counts)[len(counts) // 2]) if counts else 100

        balanced = []
        for cat, cat_records in by_category.items():
            if len(cat_records) >= target_per_category:
                balanced.extend(rng.sample(cat_records, target_per_category))
            else:
                balanced.extend(cat_records)
                while len(balanced) < len(cat_records) + (target_per_category - len(cat_records)):
                    balanced.append(rng.choice(cat_records))

        return balanced

    # ------------------------------------------------------------------
    # Step 5: Export
    # ------------------------------------------------------------------

    def export_parquet(
        self,
        records: list[SampleRecord],
        filename: str = "phase1_expanded_dataset.parquet",
    ) -> str:
        """Export records to Parquet format."""
        rows = []
        for r in records:
            d = r.to_dict()
            # Flatten for parquet
            d.pop("encoding_metadata", None)
            d.pop("multi_turn_messages", None)
            d["tags"] = json.dumps(d.get("tags", []))
            d["metadata"] = json.dumps(d.get("metadata", {}))
            rows.append(d)

        df = pd.DataFrame(rows)
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        log.info("Exported %d records to %s", len(records), path)
        return str(path)

    def export_yaml(
        self,
        records: list[SampleRecord],
        filename: str = "phase1_benchmark.yaml",
    ) -> str:
        """Export records to PINT-format YAML for benchmarking."""
        try:
            from ruamel.yaml import YAML
            yaml = YAML()
            yaml.default_flow_style = False
        except ImportError:
            import yaml as pyyaml

        entries = [r.to_pint_yaml_entry() for r in records]
        path = self.output_dir / filename

        try:
            from ruamel.yaml import YAML
            yaml = YAML()
            with open(path, "w") as f:
                yaml.dump(entries, f)
        except ImportError:
            import yaml as pyyaml
            with open(path, "w") as f:
                pyyaml.dump(entries, f, default_flow_style=False)

        log.info("Exported %d entries to %s", len(entries), path)
        return str(path)

    def export_benchmark_entries(
        self,
        records: list[SampleRecord],
        filename: str = "phase1_benchmark_entries.json",
    ) -> str:
        """Export records as benchmark entries (JSON)."""
        entries = [r.to_benchmark_entry() for r in records]
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2, default=str)
        log.info("Exported %d benchmark entries to %s", len(entries), path)
        return str(path)

    # ------------------------------------------------------------------
    # Step 6: Benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        records: list[SampleRecord],
        report_name: str = "phase1_benchmark",
    ) -> dict:
        """Run benchmark evaluation on the expanded dataset."""
        entries = [r.to_benchmark_entry() for r in records]
        report = self.benchmark.run(entries, report_name=report_name)
        print_benchmark_report(report)

        # Save report
        report_path = self.output_dir / f"{report_name}_report.json"
        save_benchmark_report(report, str(report_path))

        return report.to_dict()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        novel_per_category: int = 100,
        paraphrase_count: int = 50,
        multi_turn_count: int = 30,
        obfuscation_count: int = 40,
        edge_ambiguous: int = 40,
        edge_multilang: int = 60,
        edge_unicode: int = 50,
        edge_context: int = 30,
        balance: bool = True,
        skip_benchmark: bool = False,
    ) -> dict:
        """
        Run the complete Phase 1 pipeline.
        
        Returns:
            Dict with dataset statistics and benchmark results.
        """
        log.info("=" * 70)
        log.info("PHASE 1: DATASET EXPANSION PIPELINE")
        log.info("=" * 70)
        start_time = time.time()

        all_records = []

        # 1. Research datasets
        log.info("\n--- Step 1: Collecting research datasets ---")
        research = self.collect_research_datasets()
        all_records.extend(research)
        log.info("Research datasets: %d samples", len(research))

        # 2. Synthetic attacks
        log.info("\n--- Step 2: Generating synthetic attacks ---")
        # Use malicious research texts as seeds for paraphrasing/obfuscation
        seed_texts = [r.text for r in research if r.is_malicious][:50]
        synthetic = self.generate_synthetic(
            seed_texts=seed_texts,
            novel_per_category=novel_per_category,
            paraphrase_count=paraphrase_count,
            multi_turn_count=multi_turn_count,
            obfuscation_count=obfuscation_count,
        )
        all_records.extend(synthetic)
        log.info("Synthetic attacks: %d samples", len(synthetic))

        # 3. Edge cases
        log.info("\n--- Step 3: Generating edge cases ---")
        edge_cases = self.generate_edge_cases(
            ambiguous_count=edge_ambiguous,
            multi_lang_count=edge_multilang,
            unicode_count=edge_unicode,
            context_count=edge_context,
        )
        all_records.extend(edge_cases)
        log.info("Edge cases: %d samples", len(edge_cases))

        # 4. Deduplicate
        log.info("\n--- Step 4: Deduplication ---")
        before_dedup = len(all_records)
        all_records = self.deduplicate(all_records)
        log.info("Deduplication: %d -> %d samples (%.1f%% unique)",
                 before_dedup, len(all_records),
                 100 * len(all_records) / before_dedup if before_dedup else 0)

        # 5. Balance
        if balance:
            log.info("\n--- Step 5: Balancing ---")
            all_records = self.balance(all_records, seed=self.seed)
            log.info("After balancing: %d samples", len(all_records))

        # 6. Compute stats
        stats = compute_stats(all_records)
        log.info("\n--- Dataset Statistics ---")
        log.info(stats.summary())

        # 7. Export
        log.info("\n--- Step 6: Exporting ---")
        parquet_path = self.export_parquet(all_records)
        yaml_path = self.export_yaml(all_records)
        entries_path = self.export_benchmark_entries(all_records)

        # 8. Benchmark (optional)
        benchmark_results = None
        if not skip_benchmark:
            log.info("\n--- Step 7: Benchmarking ---")
            try:
                benchmark_results = self.run_benchmark(all_records)
            except Exception as e:
                log.warning("Benchmark failed (engine may not be running): %s", e)
                log.info("Skipping benchmark. Run later with:")
                log.info("  python -m phase1_expansion.orchestrator --mode benchmark --dataset %s", parquet_path)

        elapsed = time.time() - start_time

        result = {
            "pipeline": "phase1_dataset_expansion",
            "elapsed_seconds": round(elapsed, 1),
            "dataset_stats": stats.to_dict(),
            "exports": {
                "parquet": parquet_path,
                "yaml": yaml_path,
                "benchmark_entries": entries_path,
            },
            "benchmark_results": benchmark_results,
        }

        # Save pipeline summary
        summary_path = self.output_dir / "phase1_pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Pipeline summary saved to %s", summary_path)

        log.info("\nPhase 1 pipeline complete in %.1f seconds", elapsed)
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Guardrailer Phase 1: Dataset Expansion Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "collect", "generate", "benchmark", "yaml"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GUARDRAILER_URL", "http://localhost:8090"),
        help="Guardrailer engine URL",
    )
    parser.add_argument(
        "--dataset",
        help="Path to existing dataset (for benchmark/yaml mode)",
    )
    parser.add_argument(
        "--output",
        default=str(PHASE1_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--novel-per-category", type=int, default=100,
        help="Novel variants per category",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip benchmark evaluation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    orchestrator = Phase1Orchestrator(
        seed=args.seed,
        engine_url=args.url,
        output_dir=args.output,
    )

    if args.mode == "full":
        orchestrator.run_full_pipeline(
            novel_per_category=args.novel_per_category,
            skip_benchmark=args.skip_benchmark,
        )

    elif args.mode == "collect":
        records = orchestrator.collect_research_datasets()
        stats = compute_stats(records)
        log.info("Collected %d records", len(records))
        log.info(stats.summary())
        orchestrator.export_parquet(records, "phase1_research_only.parquet")

    elif args.mode == "generate":
        records = orchestrator.generate_synthetic(
            novel_per_category=args.novel_per_category,
        )
        edge_records = orchestrator.generate_edge_cases()
        all_records = records + edge_records
        all_records = Phase1Orchestrator.deduplicate(all_records)
        stats = compute_stats(all_records)
        log.info("Generated %d records", len(all_records))
        log.info(stats.summary())
        orchestrator.export_parquet(all_records, "phase1_synthetic_only.parquet")

    elif args.mode == "benchmark":
        if not args.dataset:
            log.error("--dataset required for benchmark mode")
            sys.exit(1)
        df = pd.read_parquet(args.dataset)
        records = []
        for _, row in df.iterrows():
            records.append(SampleRecord(
                text=row.get("text", ""),
                is_malicious=bool(row.get("is_malicious", False)),
                attack_category=row.get("attack_category", "unknown"),
                attack_technique=row.get("attack_technique", "none"),
                risk_level=row.get("risk_level", "none"),
                source_dataset=row.get("source_dataset", "unknown"),
                language=row.get("language", "en"),
            ))
        orchestrator.run_benchmark(records)

    elif args.mode == "yaml":
        if not args.dataset:
            log.error("--dataset required for yaml mode")
            sys.exit(1)
        df = pd.read_parquet(args.dataset)
        records = []
        for _, row in df.iterrows():
            records.append(SampleRecord(
                text=row.get("text", ""),
                is_malicious=bool(row.get("is_malicious", False)),
                attack_category=row.get("attack_category", "unknown"),
            ))
        path = orchestrator.export_yaml(records)
        log.info("YAML exported to %s", path)


if __name__ == "__main__":
    main()
