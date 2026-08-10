#!/usr/bin/env python3
"""
Balance top 5 sources into a 1M dataset.
Targets ~167K per category where possible.
"""

import pandas as pd
import numpy as np
import uuid
import json
import gc
from pathlib import Path
from datetime import datetime

BASE = Path("/home/prashanna/Documents/Guardrailer")
OUTPUT_DIR = BASE / "guardrailer_security" / "guardrailer_output_new"
FINAL_DIR = BASE / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def make_uuid(text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(text)[:200]))


def main():
    print("=" * 80)
    print("BALANCING TOP 5 SOURCES → 1M DATASET")
    print("=" * 80)

    # Load
    p = OUTPUT_DIR / "enhanced_payloads.parquet"
    df = pd.read_parquet(p)
    print(f"Loaded: {len(df):,} rows")

    # Filter top 5 sources
    top5 = [
        "llm_jailbreak_injection",
        "HF:Lakera/mosscap_prompt_injection",
        "HF:gabrielchua/system-prompt-leakage",
        "HF:quickium/prompt-security-v0",
        "HF:aurora-m/adversarial-prompts",
    ]
    df = df[df["source_dataset"].isin(top5)].copy()
    print(f"Top 5 sources: {len(df):,} rows")

    # Remove duplicates
    df = df.drop_duplicates(subset=["prompt_text"], keep="first")
    print(f"After dedup: {len(df):,}")

    # Current distribution
    print("\nCurrent distribution:")
    for cat in sorted(df["attack_category"].unique()):
        n = (df["attack_category"] == cat).sum()
        print(f"  {cat:30s}: {n:>8,}")

    # Target: 1M total, balanced across categories
    # Priority: fill each category to ~167K, use remaining for dominant categories
    target_per_cat = 166667
    target_total = 1_000_000

    parts = []
    category_counts = {}

    for cat in sorted(df["attack_category"].unique()):
        cat_df = df[df["attack_category"] == cat]
        current = len(cat_df)

        if current >= target_per_cat:
            # Sample down
            sampled = cat_df.sample(n=target_per_cat, random_state=42)
        else:
            # Keep all — these are the rare categories
            sampled = cat_df

        parts.append(sampled)
        category_counts[cat] = len(sampled)
        print(f"  {cat:30s}: {current:>8,} → {len(sampled):>8,}")

    combined = pd.concat(parts, ignore_index=True)
    remaining = target_total - len(combined)

    # Fill remaining with underrepresented categories
    if remaining > 0:
        print(f"\n  Filling {remaining:,} remaining slots...")
        # Priority: jailbreak (most available), then benign, then direct_injection
        fill_priority = ["jailbreak", "benign_control", "direct_injection", "system_prompt_extraction"]

        for cat in fill_priority:
            if remaining <= 0:
                break
            cat_df = df[df["attack_category"] == cat]
            already = category_counts.get(cat, 0)
            available = len(cat_df) - already
            if available > 0:
                take = min(remaining, available)
                # Sample from remaining (not already selected)
                already_selected = set(combined[combined["attack_category"] == cat]["prompt_text"].values)
                not_yet = cat_df[~cat_df["prompt_text"].isin(already_selected)]
                fill_sample = not_yet.sample(n=min(take, len(not_yet)), random_state=42)
                combined = pd.concat([combined, fill_sample], ignore_index=True)
                remaining -= len(fill_sample)
                category_counts[cat] = category_counts.get(cat, 0) + len(fill_sample)
                print(f"    +{len(fill_sample):,} {cat}")

    # Final shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Regenerate IDs
    combined["id"] = combined["prompt_text"].apply(make_uuid)

    # Ensure all required columns
    combined["cluster_id"] = 0
    combined["neighbor_ids"] = [[] for _ in range(len(combined))]
    combined["uniqueness"] = 0.5
    combined["cross_encoder_score"] = 0.0
    combined["length_norm"] = combined["prompt_text"].astype(str).apply(lambda x: len(x.split()) / 100.0)

    print(f"\n{'='*80}")
    print(f"FINAL DATASET")
    print(f"{'='*80}")
    print(f"Total: {len(combined):,}")
    print(f"\nBy category:")
    for cat in sorted(combined["attack_category"].unique()):
        n = (combined["attack_category"] == cat).sum()
        print(f"  {cat:30s}: {n:>8,} ({n/len(combined)*100:.1f}%)")

    print(f"\nBy source:")
    for src in combined["source_dataset"].value_counts().index:
        n = (combined["source_dataset"] == src).sum()
        print(f"  {src:50s}: {n:>8,}")

    print(f"\nMalicious: {combined['is_malicious'].sum():,} ({combined['is_malicious'].mean()*100:.1f}%)")
    print(f"Benign:    {(~combined['is_malicious']).sum():,} ({(~combined['is_malicious']).mean()*100:.1f}%)")

    # Save
    combined.to_parquet(OUTPUT_DIR / "enhanced_payloads.parquet", index=False)
    combined.to_parquet(FINAL_DIR / "balanced_1M_top5.parquet", index=False)

    print(f"\n✓ Saved to {OUTPUT_DIR / 'enhanced_payloads.parquet'}")
    print(f"✓ Saved to {FINAL_DIR / 'balanced_1M_top5.parquet'}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(combined),
        "sources": combined["source_dataset"].value_counts().to_dict(),
        "categories": combined["attack_category"].value_counts().to_dict(),
        "malicious_ratio": float(combined["is_malicious"].mean()),
    }
    with open(FINAL_DIR / "balanced_1M_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
