#!/usr/bin/env python3
"""Reassemble dense embeddings from chunks downloaded from Kaggle."""

import json
import os
import sys
import numpy as np


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads")
    output_path = os.path.join(input_dir, "dense_embeddings_float16.npy")

    meta_path = os.path.join(input_dir, "embeddings_meta.json")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    total = meta["total_samples"]
    dim = meta["dim"]
    n_chunks = meta["n_chunks"]
    dtype = np.float16

    print(f"Reassembling {total:,} samples x {dim} dim from {n_chunks} chunks ...")

    embeddings = np.empty((total, dim), dtype=dtype)
    offset = 0

    for i in range(n_chunks):
        chunk_path = os.path.join(input_dir, f"dense_embeddings_part_{i}.npy")
        if not os.path.exists(chunk_path):
            print(f"Error: missing {chunk_path}")
            sys.exit(1)
        chunk = np.load(chunk_path)
        embeddings[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
        print(f"  Part {i}: loaded {len(chunk):,} samples (total: {offset:,})")

    np.save(output_path, embeddings)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved {output_path} ({size_mb:.1f} MB)")
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")


if __name__ == "__main__":
    main()
