#!/usr/bin/env python3
"""Step 1: Combine staging data → Guardrailer schema → enhanced_payloads.parquet"""
import pandas as pd
import uuid
import json
from pathlib import Path

STAGE = Path("/home/prashanna/Documents/Guardrailer/guardrailer_security/staging")
OUT = Path("/home/prashanna/Documents/Guardrailer/guardrailer_security/guardrailer_output_new")
OUT.mkdir(parents=True, exist_ok=True)

keep_files = [
    "BeaverTails.parquet", "PKU-SafeRLHF.parquet", "toxic-chat.parquet",
    "alpaca.parquet", "hh_rlhf.parquet", "wildchat.parquet",
    "deepset_injections.parquet", "benign_diverse.parquet",
    "direct_injection_patterns.parquet", "jailbreak_patterns.parquet",
    "refusal_bypass_patterns.parquet", "system_prompt_extraction_patterns.parquet",
    "indirect_injection_patterns.parquet",
]

all_dfs = []
for f in keep_files:
    p = STAGE / f
    if p.exists():
        df = pd.read_parquet(p)
        all_dfs.append(df)
        print(f"  {f}: {len(df):,} ({df['category'].iloc[0]})")

# Add existing real-world — this file has columns: prompt, category, source_repository
existing = STAGE / "existing_parquet.parquet"
if existing.exists():
    edf = pd.read_parquet(existing)
    tmp = pd.DataFrame({
        "prompt": edf["prompt"].astype(str),
        "category": edf["category"].astype(str),
        "source": edf["source_repository"].astype(str)
    })
    all_dfs.append(tmp)
    print(f"  existing_parquet: {len(tmp):,}")

combined = pd.concat(all_dfs, ignore_index=True)
combined = combined.drop_duplicates(subset=["prompt"], keep="first")
combined = combined[combined["prompt"].str.strip().str.len() >= 3]

print(f"\nTotal unique: {len(combined):,}")
for cat in sorted(combined["category"].unique()):
    print(f"  {cat}: {(combined['category'] == cat).sum():,}")

# Convert to Guardrailer schema
cat_info = {
    "benign_control": ("benign_control", "none", "none"),
    "jailbreak": ("jailbreak", "pattern_matching", "high"),
    "direct_injection": ("direct_injection", "pattern_matching", "high"),
    "indirect_injection": ("indirect_injection", "context_injection", "critical"),
    "refusal_bypass": ("refusal_bypass", "social_engineering", "medium"),
    "system_prompt_extraction": ("system_prompt_extraction", "extraction", "high"),
}

records = []
for _, row in combined.iterrows():
    cat = row["category"]
    attack_cat, technique, risk = cat_info.get(cat, (cat, "unknown", "low"))
    records.append({
        "id": str(uuid.uuid4()),
        "prompt_text": str(row["prompt"])[:2000],
        "is_malicious": cat != "benign_control",
        "attack_category": attack_cat,
        "attack_technique": technique,
        "risk_level": risk,
        "source_dataset": row.get("source", "unknown"),
        "cluster_id": 0,
        "neighbor_ids": [],
        "uniqueness": 0.0,
        "cross_encoder_score": 0.0,
        "length_norm": len(str(row["prompt"]).split()) / 100.0,
    })

df = pd.DataFrame(records)
df.to_parquet(OUT / "enhanced_payloads.parquet", index=False)
print(f"\n✓ Saved {len(df):,} → guardrailer_output_new/enhanced_payloads.parquet")
