#!/usr/bin/env python3
"""
Download all available real-world datasets, map to categories, save to structured folders.
Checkpoint-resilient: each dataset saved independently.
"""

import pandas as pd
import numpy as np
import uuid
import json
import gc
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from datasets import load_dataset

BASE = Path("/home/prashanna/Documents/Guardrailer")
RAW_DIR = BASE / "data" / "raw"
PROCESSED_DIR = BASE / "data" / "processed"
FINAL_DIR = BASE / "data" / "final"
CACHE_DIR = BASE / ".hf_cache"
CHECKPOINT = PROCESSED_DIR / "download_checkpoint.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cp():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_cp(cp):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)

def done(cp, name):
    if name not in cp["completed"]:
        cp["completed"].append(name)
    save_cp(cp)

def failed(cp, name, err):
    cp["failed"] = [f for f in cp["failed"] if f["name"] != name]
    cp["failed"].append({"name": name, "error": str(err)[:200]})
    save_cp(cp)

def make_uuid(text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(text)[:200]))


# ===========================================================================
# DOWNLOAD HELPERS
# ===========================================================================
def safe_load(name, split="train", config=None):
    cache_key = f"{name.replace('/', '_')}_{config or 'default'}_{split}"
    cache_path = CACHE_DIR / f"{cache_key}.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            print(f"    [cached] {len(df):,} rows")
            return df
        except:
            cache_path.unlink(missing_ok=True)
    try:
        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        ds = load_dataset(name, **kwargs)
        df = ds.to_pandas()
        # Flatten nested columns
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().iloc[:3]
                if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x))
        try:
            df.to_parquet(cache_path, index=False)
        except:
            pass
        print(f"    ✓ {len(df):,} rows")
        return df
    except Exception as e:
        print(f"    ✗ {str(e)[:100]}")
        return None

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            try:
                if df[c].dropna().str.len().mean() > 15:
                    return c
            except:
                pass
    return df.columns[0]


# ===========================================================================
# DATASET DEFINITIONS
# ===========================================================================
DATASETS = [
    # === LARGE DATASETS ===
    {
        "name": "Lakera/mosscap_prompt_injection",
        "split": "train",
        "category": "direct_injection",
        "col": ["prompt", "text", "level"],
        "source": "HF:Lakera/mosscap_prompt_injection",
        "priority": "large",
    },
    {
        "name": "gabrielchua/system-prompt-leakage",
        "split": "train",
        "category": "system_prompt_extraction",
        "col": ["system_prompt", "content", "prompt", "text"],
        "source": "HF:gabrielchua/system-prompt-leakage",
        "priority": "large",
    },
    {
        "name": "quickium/prompt-security-v0",
        "split": "train",
        "category": "auto",  # has labels
        "col": ["text", "prompt", "id"],
        "source": "HF:quickium/prompt-security-v0",
        "priority": "large",
    },
    {
        "name": "cyberec/llm-prompt-injection-attacks",
        "split": "train",
        "category": "direct_injection",
        "col": ["text", "prompt", "input"],
        "source": "HF:cyberec/llm-prompt-injection-attacks",
        "priority": "large",
    },
    {
        "name": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "split": "train",
        "category": "benign_control",
        "col": ["prompt", "text", "content"],
        "source": "HF:nvidia/Aegis-v2",
        "priority": "large",
    },
    {
        "name": "aurora-m/adversarial-prompts",
        "split": "train",
        "category": "jailbreak",
        "col": ["text", "prompt"],
        "source": "HF:aurora-m/adversarial-prompts",
        "priority": "large",
    },

    # === MEDIUM DATASETS ===
    {
        "name": "xTRam1/safe-guard-prompt-injection",
        "split": "train",
        "category": "direct_injection",
        "col": ["text", "prompt"],
        "source": "HF:xTRam1/safe-guard",
        "priority": "medium",
    },
    {
        "name": "tom-gibbs/multi-turn_jailbreak_attack_datasets",
        "split": "train",
        "category": "jailbreak",
        "col": ["Prompt", "text", "prompt"],
        "source": "HF:tom-gibbs/multi-turn",
        "priority": "medium",
    },
    {
        "name": "nihabilal/clean-jailbreak-prompts",
        "split": "train",
        "category": "jailbreak",
        "col": ["prompt", "text"],
        "source": "HF:nihabilal/clean-jailbreak",
        "priority": "medium",
    },
    {
        "name": "nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
        "split": "train",
        "category": "benign_control",
        "col": ["text", "prompt", "content"],
        "source": "HF:nvidia/Aegis-v1",
        "priority": "medium",
    },
    {
        "name": "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
        "split": "train",
        "category": "indirect_injection",
        "col": ["id", "text", "prompt"],
        "source": "HF:nvidia/Nemotron-Indirect",
        "priority": "medium",
    },
    {
        "name": "Simsonsun/JailbreakPrompts",
        "split": "Dataset_2",
        "category": "jailbreak",
        "col": ["Prompt", "text", "prompt"],
        "source": "HF:Simsonsun/JailbreakPrompts",
        "priority": "medium",
    },

    # === EXISTING LOCAL DATA ===
    {
        "name": "local:unified_security_dataset",
        "category": "all_local",
        "priority": "local",
    },
]


# ===========================================================================
# DOWNLOAD & PROCESS
# ===========================================================================
def download_all_datasets(cp):
    print("\n" + "=" * 80)
    print("DOWNLOADING ALL DATASETS")
    print("=" * 80)

    for ds in DATASETS:
        ds_name = ds["name"].split("/")[-1].replace("local:", "local_")
        if ds_name in cp["completed"]:
            print(f"\n[skip] {ds['name']}")
            continue

        print(f"\n[{ds['priority']}] {ds['name']}")

        if ds["name"].startswith("local:"):
            process_local_data(cp, ds)
            continue

        config = ds.get("config")
        df = safe_load(ds["name"], split=ds.get("split", "train"), config=config)

        if df is None or len(df) == 0:
            failed(cp, ds_name, "No data loaded")
            continue

        # Find prompt column
        col = find_col(df, ds["col"])
        print(f"    Column: {col}")

        # Handle auto-categorization (has labels)
        if ds["category"] == "auto":
            df = auto_categorize(df, col, ds["source"])
        else:
            prompts = df[col].astype(str).tolist()

            # Handle nested conversation data
            clean_prompts = []
            for p in prompts:
                if not isinstance(p, str):
                    p = str(p) if p is not None else ""
                if p.startswith("[") or p.startswith("{"):
                    try:
                        parsed = json.loads(p)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            first = parsed[0]
                            if isinstance(first, dict):
                                p = first.get("content", first.get("message", str(first)))
                            else:
                                p = str(first)
                        elif isinstance(parsed, dict):
                            p = parsed.get("content", parsed.get("message", str(parsed)))
                    except:
                        pass
                clean_prompts.append(str(p)[:2000])

            df = pd.DataFrame({
                "prompt": clean_prompts,
                "category": ds["category"],
                "source": ds["source"]
            })

        # Filter short prompts
        df = df[df["prompt"].str.strip().str.len() >= 3]

        # Save to raw category folder
        cat = df["category"].iloc[0] if "category" in df.columns else ds["category"]
        if cat in ["direct_injection", "indirect_injection", "system_prompt_extraction",
                    "refusal_bypass", "jailbreak", "benign_control"]:
            out_path = RAW_DIR / cat / f"{ds_name}.parquet"
        else:
            out_path = RAW_DIR / f"{ds_name}.parquet"

        df.to_parquet(out_path, index=False)
        print(f"    → {len(df):,} rows → {out_path.relative_to(BASE)}")

        done(cp, ds_name)
        del df
        gc.collect()


def auto_categorize(df, col, source):
    """Auto-categorize based on labels."""
    prompts = df[col].astype(str).tolist()

    # Check for label columns
    label_col = None
    for c in ["label", "labels", "prompt_label", "type", "domain"]:
        if c in df.columns:
            label_col = c
            break

    categories = []
    for _, row in df.iterrows():
        label = str(row.get(label_col, "")).lower() if label_col else ""

        if any(x in label for x in ["injection", "attack", "malicious", "unsafe", "harmful"]):
            categories.append("direct_injection")
        elif any(x in label for x in ["jailbreak", "dan", "bypass"]):
            categories.append("jailbreak")
        elif any(x in label for x in ["extraction", "leak", "reveal"]):
            categories.append("system_prompt_extraction")
        elif any(x in label for x in ["refusal", "comply", "obey"]):
            categories.append("refusal_bypass")
        elif any(x in label for x in ["indirect", "context", "embedded"]):
            categories.append("indirect_injection")
        else:
            categories.append("benign_control")

    df = df.copy()
    df["prompt"] = prompts
    df["category"] = categories
    df["source"] = source
    return df


def process_local_data(cp, ds):
    """Process existing local unified dataset."""
    local_path = BASE / "guardrailer_security" / "unified_security_dataset.parquet"
    if not local_path.exists():
        print("    ✗ File not found")
        return

    df = pd.read_parquet(local_path)
    rw_sources = [
        'pku_safety', 'beavertails', 'in_the_wild_jailbreak',
        'neuralchemy_prompt_injection', 'detect_jailbreak',
        'jackhhao_jailbreak', 'jbb_behaviors'
    ]
    rw = df[df["source_dataset"].isin(rw_sources)]

    cat_map = {
        "direct_injection": "direct_injection",
        "indirect_injection": "indirect_injection",
        "system_prompt_extraction": "system_prompt_extraction",
        "refusal_bypass": "refusal_bypass",
        "jailbreak": "jailbreak",
        "benign_control": "benign_control"
    }

    for attack_cat, target_cat in cat_map.items():
        subset = rw[rw["attack_category"] == attack_cat]
        if len(subset) > 0:
            out = pd.DataFrame({
                "prompt": subset["prompt_text"].astype(str),
                "category": target_cat,
                "source": "local:" + subset["source_dataset"]
            })
            out_path = RAW_DIR / target_cat / "local_unified.parquet"
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                out = pd.concat([existing, out], ignore_index=True)
                out = out.drop_duplicates(subset=["prompt"], keep="first")
            out.to_parquet(out_path, index=False)
            print(f"    → {len(out):,} {target_cat}")

    done(cp, "local_unified")


# ===========================================================================
# BALANCE DATASET
# ===========================================================================
def balance_dataset(cp):
    if "balanced" in cp["completed"]:
        print("\n[skip] Balance step")
        return

    print("\n" + "=" * 80)
    print("BALANCING DATASET")
    print("=" * 80)

    # Load all raw data
    all_dfs = []
    for cat_dir in RAW_DIR.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f)
                    if "category" not in df.columns:
                        df["category"] = cat_dir.name
                    all_dfs.append(df)
                except:
                    pass

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["prompt"], keep="first")
    combined = combined[combined["prompt"].str.strip().str.len() >= 3]

    print(f"\nTotal unique samples: {len(combined):,}")
    print("\nBefore balancing:")
    for cat in sorted(combined["category"].unique()):
        n = (combined["category"] == cat).sum()
        print(f"  {cat:30s}: {n:>8,}")

    # Target: 1M total, balanced across categories
    # Target per category: ~167K each (1M / 6)
    target_per_cat = 166667
    target_total = target_per_cat * 6

    balanced_parts = []
    for cat in combined["category"].unique():
        cat_df = combined[combined["category"] == cat]
        current = len(cat_df)

        if current >= target_per_cat:
            # Sample down
            sampled = cat_df.sample(n=target_per_cat, random_state=42)
            print(f"  {cat:30s}: {current:>8,} → {target_per_cat:>8,} (sampled)")
        else:
            # Keep all, may need augmentation later
            sampled = cat_df
            print(f"  {cat:30s}: {current:>8,} → {current:>8,} (all kept, deficit: {target_per_cat - current:,})")

        balanced_parts.append(sampled)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.drop_duplicates(subset=["prompt"], keep="first")

    print(f"\nBalanced total: {len(balanced):,}")
    print("\nAfter balancing:")
    for cat in sorted(balanced["category"].unique()):
        n = (balanced["category"] == cat).sum()
        print(f"  {cat:30s}: {n:>8,}")

    # Add metadata
    balanced["uuid"] = balanced["prompt"].apply(make_uuid)
    balanced["is_synthetic"] = False
    balanced["language"] = "en"

    # Save
    balanced.to_parquet(FINAL_DIR / "balanced_1M.parquet", index=False)
    print(f"\n✓ Saved to {FINAL_DIR / 'balanced_1M.parquet'}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(balanced),
        "category_counts": balanced["category"].value_counts().to_dict(),
        "source_counts": balanced["source"].value_counts().to_dict() if "source" in balanced.columns else {},
    }
    with open(FINAL_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    done(cp, "balanced")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 80)
    print("GUARDRAILER DATASET COLLECTION")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 80)

    cp = load_cp()
    print(f"Checkpoint: {len(cp['completed'])} completed, {len(cp['failed'])} failed")

    download_all_datasets(cp)
    balance_dataset(cp)

    # Final summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    for cat_dir in sorted(RAW_DIR.iterdir()):
        if cat_dir.is_dir():
            total = 0
            for f in cat_dir.glob("*.parquet"):
                df = pd.read_parquet(f)
                total += len(df)
            print(f"  {cat_dir.name:30s}: {total:>8,}")

    final = FINAL_DIR / "balanced_1M.parquet"
    if final.exists():
        df = pd.read_parquet(final)
        print(f"\n  FINAL balanced dataset: {len(df):,}")

    print(f"\n  Folder structure:")
    print(f"    data/raw/{{category}}/     — raw downloads per category")
    print(f"    data/processed/            — intermediate processing")
    print(f"    data/final/                — balanced output")


if __name__ == "__main__":
    main()
