"""
Create a stratified 3000-sample benchmark set from the full dataset.
Ensures proportional representation of all attack categories.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


DATASET_PATH = Path("/home/prashanna/Documents/Guardrailer/dataset/guardrailer_dataset_v1.parquet")
SAMPLE_DIR = Path("/home/prashanna/Documents/Guardrailer/model_benchmarkers/samples")


def create_benchmark_sample(
    n_samples=3000,
    seed=42,
    output_dir=SAMPLE_DIR,
    dataset_path=DATASET_PATH,
):
    """
    Create a stratified sample with proportional category distribution.

    Distribution (from full dataset of 722,842):
        benign_control:          289,776 (40.1%)
        jailbreak:               267,091 (36.9%)
        direct_injection:         92,206 (12.8%)
        system_prompt_extraction: 73,769 (10.2%)

    Sample of 3000:
        benign_control:          1203
        jailbreak:               1107
        direct_injection:         384
        system_prompt_extraction: 306
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    print(f"Full dataset: {len(df):,} rows")

    rng = np.random.RandomState(seed)

    categories = df["attack_category"].value_counts()
    total = len(df)

    sample_counts = {}
    remaining = n_samples
    for cat in categories.index[:-1]:
        proportion = categories[cat] / total
        count = round(n_samples * proportion)
        count = min(count, categories[cat], remaining)
        sample_counts[cat] = count
        remaining -= count
    sample_counts[categories.index[-1]] = min(remaining, categories[categories.index[-1]])

    print(f"\nSample distribution (target: {n_samples}):")
    sampled_dfs = []
    for cat, count in sample_counts.items():
        cat_df = df[df["attack_category"] == cat]
        indices = rng.choice(cat_df.index, size=count, replace=False)
        sampled_dfs.append(df.loc[indices])
        print(f"  {cat:30s}: {count:>5} / {categories[cat]:>8,}")

    sample = pd.concat(sampled_dfs).sample(frac=1, random_state=rng).reset_index(drop=True)

    assert len(sample) == n_samples, f"Expected {n_samples} samples, got {len(sample)}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = output_dir / f"benchmark_sample_{n_samples}_{timestamp}.parquet"
    sample.to_parquet(sample_path, index=False)

    meta = {
        "n_samples": n_samples,
        "seed": seed,
        "dataset_source": str(dataset_path),
        "category_distribution": {cat: int(count) for cat, count in sample_counts.items()},
        "is_malicious_distribution": sample["is_malicious"].value_counts().to_dict(),
        "created_at": datetime.now().isoformat(),
        "file": str(sample_path),
    }
    meta_path = output_dir / f"benchmark_sample_{n_samples}_{timestamp}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {sample_path}")
    print(f"Meta:  {meta_path}")

    return sample, meta


if __name__ == "__main__":
    sample, meta = create_benchmark_sample()
