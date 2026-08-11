#!/usr/bin/env python3
"""Step 1: Create fixed train/val/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATASET_PATH = Path("/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet")
OUT_DIR = Path(__file__).parent

df = pd.read_parquet(DATASET_PATH, columns=["prompt_text", "is_malicious", "attack_category"])
df["is_malicious"] = df["is_malicious"].astype(int)

print(f"Full dataset: {len(df):,} rows")

train_val, test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["is_malicious"]
)
train, val = train_test_split(
    train_val, test_size=0.125, random_state=42, stratify=train_val["is_malicious"]
)

train.to_parquet(OUT_DIR / "split_train.parquet", index=False)
val.to_parquet(OUT_DIR / "split_val.parquet", index=False)
test.to_parquet(OUT_DIR / "split_test.parquet", index=False)

print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
