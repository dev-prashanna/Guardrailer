#!/usr/bin/env python3
"""
Combine Downloads unified dataset + collected real-world data → 1M balanced dataset.
Outputs in exact schema expected by ingest_precomputed.py.

Schema required:
  - id, prompt_text, is_malicious, attack_category, attack_technique,
    risk_level, source_dataset, cluster_id, neighbor_ids, uniqueness,
    cross_encoder_score, length_norm

Also generates:
  - dense_embeddings_float16.npy (1024-dim, BAAI/bge-large-en-v1.5)
  - corpus_meta.json (IDF, centroids, weights)
"""

import pandas as pd
import numpy as np
import uuid
import json
import gc
import os
import sys
import time
from pathlib import Path
from datetime import datetime

BASE = Path("/home/prashanna/Documents/Guardrailer")
RAW_DIR = BASE / "data" / "raw"
FINAL_DIR = BASE / "data" / "final"
OUTPUT_DIR = BASE / "guardrailer_security" / "guardrailer_output_new"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_uuid(text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(text)[:200]))


# ===========================================================================
# STEP 1: Load both datasets
# ===========================================================================
def load_downloads_dataset():
    """Load the 1M Downloads version."""
    path = Path.home() / "Downloads" / "unified_security_dataset.parquet"
    print(f"  Loading Downloads dataset: {path}")
    df = pd.read_parquet(path)
    print(f"    Rows: {len(df):,}")
    return df


def load_collected_dataset():
    """Load our collected real-world data from data/raw/."""
    print(f"  Loading collected data from {RAW_DIR}/")
    dfs = []
    for cat_dir in RAW_DIR.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f)
                    if "category" not in df.columns:
                        df["category"] = cat_dir.name
                    dfs.append(df)
                except:
                    pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["prompt"], keep="first")
    combined = combined[combined["prompt"].astype(str).str.strip().str.len() >= 3]
    print(f"    Rows: {len(combined):,}")
    return combined


# ===========================================================================
# STEP 2: Map to standard categories
# ===========================================================================
def map_downloads_categories(df):
    """Map Downloads dataset categories to standard schema."""
    # Downloads already has correct categories
    return df


def map_collected_categories(df):
    """Map collected data to Downloads schema."""
    # Collected data has: prompt, category, source
    mapped = pd.DataFrame()
    mapped["prompt_text"] = df["prompt"].astype(str).str[:2000]
    mapped["attack_category"] = df["category"]
    mapped["source_dataset"] = df.get("source", "collected")
    mapped["is_malicious"] = df["category"] != "benign_control"

    # Map attack techniques
    technique_map = {
        "direct_injection": "pattern_matching",
        "indirect_injection": "context_injection",
        "system_prompt_extraction": "extraction",
        "refusal_bypass": "social_engineering",
        "jailbreak": "pattern_matching",
        "benign_control": "none",
    }
    mapped["attack_technique"] = mapped["attack_category"].map(technique_map).fillna("unknown")

    # Map risk levels
    risk_map = {
        "direct_injection": "high",
        "indirect_injection": "critical",
        "system_prompt_extraction": "high",
        "refusal_bypass": "medium",
        "jailbreak": "high",
        "benign_control": "none",
    }
    mapped["risk_level"] = mapped["attack_category"].map(risk_map).fillna("low")

    return mapped


# ===========================================================================
# STEP 3: Combine and deduplicate
# ===========================================================================
def combine_and_dedup(downloads_df, collected_df):
    """Combine both datasets, deduplicate on prompt_text."""
    print("\n  Combining datasets...")

    # Ensure Downloads has all required columns
    required_cols = ["id", "prompt_text", "is_malicious", "attack_category",
                     "attack_technique", "risk_level", "source_dataset"]

    # Add missing columns to Downloads
    for col in required_cols:
        if col not in downloads_df.columns:
            if col == "id":
                downloads_df["id"] = downloads_df["prompt_text"].apply(make_uuid)
            elif col == "attack_technique":
                downloads_df["attack_technique"] = "unknown"
            elif col == "risk_level":
                downloads_df["risk_level"] = "low"

    # Add missing columns to collected
    for col in required_cols:
        if col not in collected_df.columns:
            if col == "id":
                collected_df["id"] = collected_df["prompt_text"].apply(make_uuid)

    # Combine
    combined = pd.concat([
        downloads_df[required_cols],
        collected_df[required_cols]
    ], ignore_index=True)

    print(f"    Before dedup: {len(combined):,}")

    # Deduplicate on prompt_text
    combined = combined.drop_duplicates(subset=["prompt_text"], keep="first")
    print(f"    After dedup:  {len(combined):,}")

    # Filter quality
    combined = combined[combined["prompt_text"].astype(str).str.strip().str.len() >= 3]
    print(f"    After filter: {len(combined):,}")

    return combined


# ===========================================================================
# STEP 4: Balance to 1M
# ===========================================================================
def balance_to_1m(df):
    """Balance dataset to exactly 1,000,000 samples."""
    print(f"\n  Current distribution:")
    for cat in sorted(df["attack_category"].unique()):
        n = (df["attack_category"] == cat).sum()
        print(f"    {cat:30s}: {n:>8,}")

    total = len(df)
    target = 1_000_000

    if total >= target:
        # Sample down proportionally
        print(f"\n  Sampling {target:,} from {total:,} (proportional)")
        df = df.sample(n=target, random_state=42)
    else:
        # Need more data — upsample deficit categories
        print(f"\n  Need {target - total:,} more samples")

        # Target: ~167K per category (1M / 6)
        target_per_cat = target // 6
        parts = []

        for cat in df["attack_category"].unique():
            cat_df = df[df["attack_category"] == cat]
            current = len(cat_df)

            if current >= target_per_cat:
                sampled = cat_df.sample(n=target_per_cat, random_state=42)
            else:
                # Upsample with slight variations
                deficit = target_per_cat - current
                upsampled = cat_df.sample(n=deficit, replace=True, random_state=42)
                sampled = pd.concat([cat_df, upsampled], ignore_index=True)

            parts.append(sampled)
            print(f"    {cat:30s}: {current:>8,} → {len(sampled):>8,}")

        df = pd.concat(parts, ignore_index=True)

    # Final shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Final: {len(df):,}")
    for cat in sorted(df["attack_category"].unique()):
        n = (df["attack_category"] == cat).sum()
        print(f"    {cat:30s}: {n:>8,}")

    return df


# ===========================================================================
# STEP 5: Add derived columns (schema compliance)
# ===========================================================================
def add_derived_columns(df):
    """Add cluster_id, neighbor_ids, uniqueness, cross_encoder_score, length_norm."""
    print("\n  Adding derived columns...")

    df["id"] = df["prompt_text"].apply(make_uuid)

    # Length normalization
    df["length_norm"] = df["prompt_text"].astype(str).apply(
        lambda x: len(x.split()) / 100.0
    )

    # Placeholder values for columns that need pre-computation
    df["cluster_id"] = 0
    df["neighbor_ids"] = [[] for _ in range(len(df))]
    df["uniqueness"] = 0.5
    df["cross_encoder_score"] = 0.0

    print(f"    ✓ Added derived columns")
    return df


# ===========================================================================
# STEP 6: Generate embeddings
# ===========================================================================
def generate_embeddings(df):
    """Generate 1024-dim embeddings with BAAI/bge-large-en-v1.5."""
    print("\n  Generating embeddings...")
    print(f"    Loading BAAI/bge-large-en-v1.5...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    texts = df["prompt_text"].tolist()
    print(f"    Embedding {len(texts):,} texts...")

    batch_size = 64
    all_emb = []
    start = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_emb.append(emb)

        if (i // batch_size) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
            eta = (len(texts) - i - len(batch)) / rate if rate > 0 else 0
            print(f"      {i + len(batch):,}/{len(texts):,} | {rate:.0f}/s | ETA: {eta:.0f}s")
            sys.stdout.flush()

    embeddings = np.vstack(all_emb).astype(np.float16)
    elapsed = time.time() - start
    print(f"    ✓ {embeddings.shape} in {elapsed:.1f}s")

    return embeddings


# ===========================================================================
# STEP 7: Build corpus_meta.json
# ===========================================================================
def build_corpus_meta(df, embeddings):
    """Build corpus metadata for scoring module."""
    print("\n  Building corpus_meta.json...")

    # Load SPARSE_KEYWORDS from constants
    sys.path.insert(0, str(BASE / "guardrailer_security"))
    from constants import SPARSE_KEYWORDS

    texts = df["prompt_text"].astype(str).str.lower().tolist()
    N = len(texts)

    # IDF computation
    doc_freq = {}
    total_words = 0
    for text in texts:
        words = text.split()
        total_words += len(words)
        for w in set(words):
            doc_freq[w] = doc_freq.get(w, 0) + 1

    avgdl = total_words / N if N > 0 else 100.0

    # IDF for SPARSE_KEYWORDS
    keyword_idf = {}
    for kw in SPARSE_KEYWORDS:
        df_count = sum(1 for t in texts if kw in t)
        if df_count > 0:
            keyword_idf[kw] = np.log((N - df_count + 0.5) / (df_count + 0.5) + 1.0)
        else:
            keyword_idf[kw] = 1.0

    # Category centroids
    category_centroids = {}
    for cat in df["attack_category"].unique():
        mask = df["attack_category"] == cat
        if mask.sum() > 0:
            cat_emb = embeddings[mask.values].astype(np.float32)
            centroid = cat_emb.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            category_centroids[cat] = centroid.tolist()

    scoring_weights = {
        "dense": 0.30, "sparse_idf": 0.18, "centroid": 0.12,
        "cross_encoder": 0.12, "perplexity": 0.05, "entropy": 0.05,
        "token_frequency": 0.03, "ngram_overlap": 0.02,
        "uniqueness": 0.05, "length_norm": 0.05, "ensemble_bonus": 0.03,
    }

    meta = {
        "keyword_idf": keyword_idf,
        "total_documents": N,
        "avg_doc_length": avgdl,
        "avg_text_length": avgdl,
        "category_centroids": category_centroids,
        "cluster_centers": {},
        "scoring_weights": scoring_weights,
    }

    print(f"    ✓ {N:,} docs, {len(keyword_idf)} IDF keywords, {len(category_centroids)} centroids")
    return meta


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 80)
    print("COMBINING DATASETS → 1M BALANCED")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 80)

    # Step 1: Load
    print("\n[1/7] Loading datasets")
    downloads_df = load_downloads_dataset()
    collected_df = load_collected_dataset()

    # Step 2: Map categories
    print("\n[2/7] Mapping categories")
    downloads_df = map_downloads_categories(downloads_df)
    collected_df = map_collected_categories(collected_df)
    print(f"    Downloads: {len(downloads_df):,}")
    print(f"    Collected: {len(collected_df):,}")

    # Step 3: Combine
    print("\n[3/7] Combining and deduplicating")
    combined = combine_and_dedup(downloads_df, collected_df)
    del downloads_df, collected_df
    gc.collect()

    # Step 4: Balance
    print("\n[4/7] Balancing to 1M")
    balanced = balance_to_1m(combined)
    del combined
    gc.collect()

    # Step 5: Add derived columns
    print("\n[5/7] Adding derived columns")
    balanced = add_derived_columns(balanced)

    # Save payloads
    balanced.to_parquet(OUTPUT_DIR / "enhanced_payloads.parquet", index=False)
    print(f"    ✓ Saved enhanced_payloads.parquet ({len(balanced):,} rows)")

    # Step 6: Generate embeddings
    print("\n[6/7] Generating embeddings")
    embeddings = generate_embeddings(balanced)
    np.save(OUTPUT_DIR / "dense_embeddings_float16.npy", embeddings)
    print(f"    ✓ Saved dense_embeddings_float16.npy")

    # Step 7: Build corpus_meta
    print("\n[7/7] Building corpus_meta.json")
    meta = build_corpus_meta(balanced, embeddings)
    with open(OUTPUT_DIR / "corpus_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    ✓ Saved corpus_meta.json")

    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"  Output: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size / 1e6
        print(f"    {f.name}: {size:.1f} MB")

    print(f"\n  Ingest command:")
    print(f"    cd {BASE / 'guardrailer_security'}")
    print(f"    python3 ingest_precomputed.py --input-dir {OUTPUT_DIR} --force-recreate")


if __name__ == "__main__":
    main()
