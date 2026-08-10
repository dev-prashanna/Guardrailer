#!/usr/bin/env python3
"""
Clean the final dataset by applying quality fixes identified during inspection.

Steps:
1. Deduplicate on prompt_text (keep first occurrence)
2. Drop dead columns (uniqueness, cross_encoder_score, neighbor_ids, cluster_id)
3. Reconcile label vs category for HF:ahsanayub/malicious-prompts benign rows
4. Normalize length_norm using min-max scaling
5. Filter extreme lengths (<5 chars or >20K chars)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


INPUT_PATH = "data/final/final_dataset.parquet"
OUTPUT_PATH = "data/final/final_dataset_cleaned.parquet"
SUMMARY_PATH = "data/final/cleaning_summary.json"


def clean_dataset() -> pd.DataFrame:
    print("=" * 80)
    print("DATASET CLEANING PIPELINE")
    print("=" * 80)

    df = pd.read_parquet(INPUT_PATH)
    original_count = len(df)
    print(f"\nLoaded {original_count:,} rows, {len(df.columns)} columns")

    # --- Step 1: Deduplicate on prompt_text ---
    print(f"\n--- Step 1: Deduplicate on prompt_text ---")
    before = len(df)
    df = df.drop_duplicates(subset=["prompt_text"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    print(f"  Removed {removed:,} duplicate rows ({removed/original_count*100:.2f}%)")
    print(f"  Remaining: {len(df):,} rows")

    # --- Step 2: Drop dead columns ---
    print(f"\n--- Step 2: Drop dead columns ---")
    dead_cols = ["uniqueness", "cross_encoder_score", "neighbor_ids", "cluster_id"]
    existing_dead = [c for c in dead_cols if c in df.columns]
    df = df.drop(columns=existing_dead)
    print(f"  Dropped: {existing_dead}")
    print(f"  Remaining columns: {list(df.columns)}")

    # --- Step 3: Reconcile label vs category ---
    print(f"\n--- Step 3: Reconcile label vs category for HF:ahsanayub benign rows ---")
    mask = (df["source_dataset"] == "HF:ahsanayub/malicious-prompts") & (df["is_malicious"] == False)
    affected = mask.sum()
    df.loc[mask, "attack_category"] = "benign_control"
    df.loc[mask, "attack_technique"] = "none"
    df.loc[mask, "risk_level"] = "none"
    print(f"  Fixed {affected:,} rows: remapped attack_category to benign_control")

    # --- Step 4: Normalize length_norm ---
    print(f"\n--- Step 4: Normalize length_norm ---")
    text_lengths = df["prompt_text"].str.len().astype(float)
    df["length_norm"] = (text_lengths - text_lengths.min()) / (text_lengths.max() - text_lengths.min())
    print(f"  Recomputed length_norm using min-max normalization")
    print(f"  New range: [{df['length_norm'].min():.6f}, {df['length_norm'].max():.6f}]")
    print(f"  Mean: {df['length_norm'].mean():.6f}, Std: {df['length_norm'].std():.6f}")

    # --- Step 5: Filter extreme lengths ---
    print(f"\n--- Step 5: Filter extreme lengths ---")
    df["_text_len"] = df["prompt_text"].str.len()
    short_mask = df["_text_len"] < 5
    long_mask = df["_text_len"] > 20000
    remove_mask = short_mask | long_mask
    short_count = short_mask.sum()
    long_count = long_mask.sum()
    df = df[~remove_mask].drop(columns=["_text_len"]).reset_index(drop=True)
    print(f"  Removed {short_count:,} texts < 5 chars")
    print(f"  Removed {long_count:,} texts > 20K chars")
    print(f"  Remaining: {len(df):,} rows")

    # --- Step 6: Regenerate IDs ---
    print(f"\n--- Step 6: Regenerate unique IDs ---")
    df["id"] = [f"cleaned_{i:07d}" for i in range(len(df))]
    assert df["id"].is_unique, "ID generation failed"
    print(f"  Generated {len(df):,} unique IDs")

    return df, {
        "original_count": original_count,
        "final_count": len(df),
        "rows_removed": original_count - len(df),
        "duplicates_removed": removed,
        "dead_columns_dropped": existing_dead,
        "label_category_fixes": int(affected),
        "short_texts_removed": int(short_count),
        "long_texts_removed": int(long_count),
        "ids_regenerated": True,
    }


def validate_cleaned(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print("POST-CLEANING VALIDATION")
    print("=" * 80)

    checks = []

    # No missing values
    missing = df.isnull().sum().sum()
    checks.append(("No missing values", missing == 0, f"{missing} found"))

    # No duplicates
    dup_texts = df["prompt_text"].duplicated().sum()
    checks.append(("No duplicate texts", dup_texts == 0, f"{dup_texts} found"))

    # Unique IDs
    checks.append(("All IDs unique", df["id"].is_unique, ""))

    # No dead columns
    dead_cols = ["uniqueness", "cross_encoder_score", "neighbor_ids", "cluster_id"]
    remaining_dead = [c for c in dead_cols if c in df.columns]
    checks.append(("Dead columns removed", len(remaining_dead) == 0, f"Remaining: {remaining_dead}"))

    # Label-category consistency
    benign_nonbenign = ((df["is_malicious"] == False) & (df["attack_category"] != "benign_control")).sum()
    checks.append(("Label-category consistency", benign_nonbenign == 0, f"{benign_nonbenign} mismatches"))

    # Length bounds
    text_lens = df["prompt_text"].str.len()
    checks.append(("No texts < 5 chars", (text_lens < 5).sum() == 0, ""))
    checks.append(("No texts > 20K chars", (text_lens > 20000).sum() == 0, ""))

    # length_norm in [0, 1]
    checks.append(("length_norm in [0,1]", df["length_norm"].min() >= 0 and df["length_norm"].max() <= 1, ""))

    # Required columns
    required = ["id", "prompt_text", "is_malicious", "attack_category", "attack_technique", "risk_level", "source_dataset"]
    missing_cols = [c for c in required if c not in df.columns]
    checks.append(("Required columns present", len(missing_cols) == 0, f"Missing: {missing_cols}"))

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail and not passed:
            msg += f" — {detail}"
        print(msg)
        if not passed:
            all_pass = False

    return all_pass


def print_summary(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print("CLEANED DATASET SUMMARY")
    print("=" * 80)

    print(f"\nRows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")

    print(f"\nClass distribution:")
    vc = df["is_malicious"].value_counts()
    for val, cnt in vc.items():
        print(f"  {val}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

    print(f"\nAttack category:")
    print(df["attack_category"].value_counts().to_string(header=False))

    print(f"\nSource dataset:")
    print(df["source_dataset"].value_counts().to_string(header=False))

    print(f"\nRisk level:")
    print(df["risk_level"].value_counts().to_string(header=False))

    print(f"\nText length stats:")
    tl = df["prompt_text"].str.len()
    print(f"  Min: {tl.min()}, Max: {tl.max()}, Mean: {tl.mean():.1f}, Median: {tl.median():.0f}")

    print(f"\nlength_norm stats:")
    print(f"  Min: {df['length_norm'].min():.6f}, Max: {df['length_norm'].max():.6f}, Mean: {df['length_norm'].mean():.6f}")

    print(f"\nCross-tab: is_malicious x attack_category")
    ct = pd.crosstab(df["is_malicious"], df["attack_category"], margins=True)
    print(ct.to_string())


def main():
    start = datetime.now()

    df, cleaning_stats = clean_dataset()
    all_pass = validate_cleaned(df)
    print_summary(df)

    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": INPUT_PATH,
        "output_file": OUTPUT_PATH,
        "cleaning_steps": cleaning_stats,
        "validation_passed": all_pass,
        "final_stats": {
            "total_samples": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "malicious_ratio": float(df["is_malicious"].mean()),
            "attack_category": df["attack_category"].value_counts().to_dict(),
            "source_dataset": df["source_dataset"].value_counts().to_dict(),
            "risk_level": df["risk_level"].value_counts().to_dict(),
            "text_length_mean": float(df["prompt_text"].str.len().mean()),
            "text_length_median": float(df["prompt_text"].str.len().median()),
        },
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved cleaning summary to {SUMMARY_PATH}")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
