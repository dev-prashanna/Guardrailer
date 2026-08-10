#!/usr/bin/env python3
"""
deploy_final_project.py
Automated deployment script for the Guardrailer Final Project.

Creates Final_Guardrailer_Project/ in ~/Documents/ and copies only
essential files from the current codebase.

Usage:
    python3 deploy_final_project.py                    # Use default file list
    python3 deploy_final_project.py --config my.json   # Use custom config
    python3 deploy_final_project.py --dry-run           # Preview without copying
    python3 deploy_final_project.py --force             # Overwrite existing dir
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = Path.home() / "Documents"
TARGET_DIR = DOCUMENTS_DIR / "Final_Guardrailer_Project"

# Default files to copy — grouped by purpose
DEFAULT_FILE_LIST = {
    "evaluation_results": {
        "description": "Final evaluation results and reports",
        "files": [
            "evaluation_results/phase5_prototype_results.json",
            "evaluation_results/PHASE5_RESULTS.md",
            "evaluation_results/leakage_free_report.json",
            "evaluation_results/full_pipeline_report.json",
            "evaluation_results/evaluation_report.json",
            "evaluation_results/evaluation_report_v2.json",
        ],
    },
    "evaluation_figures": {
        "description": "Publication-quality visualization plots",
        "files": [
            "evaluation_results/figures/01_primary_metrics.png",
            "evaluation_results/figures/02_per_category_accuracy.png",
            "evaluation_results/figures/03_ablation_study.png",
            "evaluation_results/figures/04_robustness.png",
            "evaluation_results/figures/05_per_tier_detection.png",
            "evaluation_results/figures/06_latency_profiling.png",
            "evaluation_results/figures/07_baseline_comparison.png",
            "evaluation_results/figures/ablation_study.png",
            "evaluation_results/figures/classifier_comparison.png",
            "evaluation_results/figures/confusion_matrix.png",
            "evaluation_results/figures/feature_importance.png",
            "evaluation_results/figures/roc_curves.png",
            "evaluation_results/figures/signal_correlation.png",
            "evaluation_results/figures/signal_distributions.png",
        ],
    },
    "research_scripts": {
        "description": "Research evaluation scripts",
        "files": [
            "guardrailer_security/research/evaluate_leakage_free.py",
            "guardrailer_security/research/evaluate_full_pipeline.py",
            "guardrailer_security/research/run_evaluation.py",
            "guardrailer_security/research/statistics.py",
            "guardrailer_security/research/dataset_engineering.py",
            "guardrailer_security/research/experimental_design.py",
            "guardrailer_security/research/robustness.py",
            "guardrailer_security/research/__init__.py",
        ],
    },
    "core_scoring": {
        "description": "Core scoring and model files",
        "files": [
            "guardrailer_security/security_engine.py",
            "guardrailer_security/scoring.py",
            "guardrailer_security/improved_scoring.py",
            "guardrailer_security/constants.py",
            "guardrailer_security/train_phase3.py",
        ],
    },
    "trained_models": {
        "description": "Pre-trained model weights and calibrators",
        "files": [
            "guardrailer_security/models/logistic_weights.json",
            "guardrailer_security/models/neural_weights.json",
            "guardrailer_security/models/calibrator.json",
        ],
    },
    "corpus_metadata": {
        "description": "Corpus statistics (IDF, centroids, weights)",
        "files": [
            "guardrailer_security/corpus_meta.json",
        ],
    },
    "manuscript": {
        "description": "Academic paper draft",
        "files": [
            "manuscript/manuscript.md",
        ],
    },
    "documentation": {
        "description": "Project documentation",
        "files": [
            "docs/RESEARCH_EXECUTION_PLAN.md",
            "docs/EVALUATION_CHECKLIST.md",
            "docs/EXECUTIVE_SUMMARY.md",
            "docs/architecture.md",
            "README.md",
            "LICENSE",
        ],
    },
    "reproducibility": {
        "description": "Reproducibility manifests",
        "files": [
            "reproducibility/manifest.json",
        ],
    },
    "configuration": {
        "description": "Environment and dependency files",
        "files": [
            "guardrailer_security/requirements.txt",
        ],
    },
}


def load_custom_config(config_path: str) -> dict:
    """Load a custom file list from a JSON configuration file.

    Expected format:
    {
        "category_name": {
            "description": "What this group contains",
            "files": ["relative/path/to/file1", "relative/path/to/file2"]
        }
    }
    """
    with open(config_path) as f:
        return json.load(f)


def collect_files(file_list: dict, source_root: Path) -> list:
    """Resolve all file paths and return list of (src, dst) tuples.

    Skips files that don't exist and logs warnings.
    """
    pairs = []
    missing = []

    for category, info in file_list.items():
        files = info.get("files", [])
        for rel_path in files:
            src = (source_root / rel_path).resolve()
            # Destination preserves the relative structure under the target
            # Strip leading ../ to flatten into the target directory
            dst_rel = rel_path.lstrip("../")
            dst = TARGET_DIR / dst_rel

            if src.exists():
                pairs.append((src, dst, category))
            else:
                missing.append((rel_path, src))

    return pairs, missing


def print_summary(pairs: list, missing: list, file_list: dict):
    """Print a summary of what will be copied."""
    print("\n" + "=" * 60)
    print("  DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"\n  Target: {TARGET_DIR}")
    print(f"  Source: {SOURCE_ROOT}")

    # Group by category
    by_category = {}
    for src, dst, cat in pairs:
        by_category.setdefault(cat, []).append((src, dst))

    total_files = 0
    for cat, items in by_category.items():
        desc = file_list.get(cat, {}).get("description", "")
        print(f"\n  [{cat}] {desc}")
        for src, dst in items:
            rel = src.relative_to(SOURCE_ROOT)
            print(f"    {rel}")
            total_files += 1

    print(f"\n  Total files to copy: {total_files}")

    if missing:
        print(f"\n  Missing files (will be skipped):")
        for rel_path, src in missing:
            print(f"    {rel_path}")

    print("=" * 60)


def deploy(file_list: dict, dry_run: bool = False, force: bool = False):
    """Execute the deployment."""
    global TARGET_DIR

    # Check target directory
    if TARGET_DIR.exists():
        if force:
            print(f"  Removing existing directory: {TARGET_DIR}")
            if not dry_run:
                shutil.rmtree(TARGET_DIR)
        else:
            print(f"  ERROR: Directory already exists: {TARGET_DIR}")
            print(f"  Use --force to overwrite, or choose a different name.")
            sys.exit(1)

    # Collect files
    pairs, missing = collect_files(file_list, SOURCE_ROOT)

    if not pairs:
        print("  ERROR: No files found to copy.")
        sys.exit(1)

    # Print summary
    print_summary(pairs, missing, file_list)

    if dry_run:
        print("\n  DRY RUN — no files copied.\n")
        return

    # Create target directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy files
    copied = 0
    errors = []
    for src, dst, cat in pairs:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            errors.append((src, str(e)))

    # Write deployment manifest
    manifest = {
        "deployed_at": datetime.now().isoformat(),
        "source_root": str(SOURCE_ROOT),
        "target_dir": str(TARGET_DIR),
        "files_copied": copied,
        "files_missing": len(missing),
        "categories": list(file_list.keys()),
        "file_list_used": "default" if file_list is DEFAULT_FILE_LIST else "custom",
    }
    manifest_path = TARGET_DIR / "deployment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Deployment complete!")
    print(f"  Copied: {copied} files")
    if errors:
        print(f"  Errors: {len(errors)}")
        for src, err in errors:
            print(f"    {src}: {err}")
    print(f"  Target: {TARGET_DIR}")
    print(f"  Manifest: {manifest_path}\n")


def main():
    global TARGET_DIR

    parser = argparse.ArgumentParser(
        description="Deploy Guardrailer final project files"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom JSON config file (default: use built-in list)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be copied without actually copying",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target directory",
    )
    parser.add_argument(
        "--target",
        default=None,
        help=f"Override target directory (default: {TARGET_DIR})",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all file categories and exit",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List all files that would be copied and exit",
    )
    args = parser.parse_args()

    if args.target:
        TARGET_DIR = Path(args.target)

    # Load file list
    if args.config:
        file_list = load_custom_config(args.config)
        print(f"  Loaded custom config: {args.config}")
    else:
        file_list = DEFAULT_FILE_LIST

    # List modes
    if args.list_categories:
        print("\n  File categories:")
        for cat, info in file_list.items():
            n = len(info.get("files", []))
            print(f"    {cat:25s} ({n:2d} files) — {info.get('description', '')}")
        return

    if args.list_files:
        pairs, missing = collect_files(file_list, SOURCE_ROOT)
        print(f"\n  Files that will be copied ({len(pairs)} total):")
        for src, dst, cat in pairs:
            rel = src.relative_to(SOURCE_ROOT)
            print(f"    {rel}")
        if missing:
            print(f"\n  Missing ({len(missing)}):")
            for rel, src in missing:
                print(f"    {rel}")
        return

    deploy(file_list, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
