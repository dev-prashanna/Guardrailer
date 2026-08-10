#!/usr/bin/env python3
"""Step 2: Generate dense embeddings. Run as standalone process."""
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

OUT = Path("/home/prashanna/Documents/Guardrailer/guardrailer_security/guardrailer_output_new")

print("Loading payloads...")
df = pd.read_parquet(OUT / "enhanced_payloads.parquet")
texts = df["prompt_text"].tolist()
print(f"  {len(texts):,} texts")

print("Loading BAAI/bge-large-en-v1.5...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

print("Embedding...")
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
        print(f"  {i + len(batch):,}/{len(texts):,} | {rate:.0f}/s | ETA: {eta:.0f}s")
        sys.stdout.flush()

embeddings = np.vstack(all_emb).astype(np.float16)
np.save(OUT / "dense_embeddings_float16.npy", embeddings)

elapsed = time.time() - start
print(f"\n✓ {embeddings.shape} saved in {elapsed:.1f}s")
