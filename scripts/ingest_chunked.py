#!/usr/bin/env python3
"""
ingest_chunked.py — Recompile 70MB embedding chunks and ingest into local Qdrant.

Usage:
    python3 ingest_chunked.py --input-dir /path/to/guardrailer_embeddings
    python3 ingest_chunked.py --verify  # verify collection integrity
    python3 ingest_chunked.py --force-recreate  # delete and rebuild

The input directory should contain:
    output/enhanced_payloads.parquet
    output/corpus_meta.json
    chunks/embeddings_chunk_0000.npy
    chunks/embeddings_chunk_0001.npy
    ...
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("GUARDRAILER_COLLECTION", "guardrailer_security_enhanced")
BATCH_SIZE = 64

# Load sparse keywords from constants
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from constants import SPARSE_KEYWORDS
except ImportError:
    SPARSE_KEYWORDS = [
        "ignore previous", "override", "bypass", "jailbreak", "system prompt",
        "your instructions", "forget", "disregard", "dan", "do anything now",
        "act as", "roleplay", "pretend you", "hypothetical", "in theory",
        "markdown injection", "code comment", "readme", "yaml", "json payload",
        "<script>", "]]>", "```", "<!--", "-->", "eval(", "exec(",
        "base64", "rot13", "hex encoded", "obfuscated",
        "ignore all", "new instructions", "you are now", "persona",
        "reveal", "output", "display", "print", "show", "expose",
        "previous instructions", "earlier instructions", "initial instructions",
        "developer mode", "debug mode", "admin mode", "root mode",
        "you must", "you will", "you shall", "comply", "obey",
        "no restrictions", "no rules", "no limits", "unrestricted",
        "evil", "uncensored", "unfiltered", "without guidelines",
    ]


# ---------------------------------------------------------------------------
# Sparse vector builder
# ---------------------------------------------------------------------------
def build_idf_sparse_vector(text, idf_values, avgdl, N):
    from qdrant_client.models import SparseVector

    lower = text.lower()
    indices = []
    values = []

    for i, keyword in enumerate(SPARSE_KEYWORDS):
        if keyword in lower:
            idf = idf_values.get(keyword, 1.0)
            word_count = len(lower.split())
            tf = lower.count(keyword) / max(word_count, 1)
            bm25_tf = (tf * 2.0) / (tf + 1.5 * (1.0 - 0.75 + 0.75 * word_count / max(avgdl, 1)))
            score = idf * (bm25_tf + 1.0)
            indices.append(i)
            values.append(score)

    return SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------------------
# Chunk recompilation
# ---------------------------------------------------------------------------
def recompile_chunks(chunks_dir):
    """Load all 70MB chunks and concatenate into full embedding matrix."""
    chunk_files = sorted(Path(chunks_dir).glob("embeddings_chunk_*.npy"))

    if not chunk_files:
        log.error("No chunk files found in %s", chunks_dir)
        sys.exit(1)

    log.info("Found %d chunks in %s", len(chunk_files), chunks_dir)

    chunks = []
    total_rows = 0
    for f in chunk_files:
        chunk = np.load(f)
        chunks.append(chunk)
        total_rows += chunk.shape[0]
        log.info("  %s: %s (%.1f MB)", f.name, chunk.shape, f.stat().st_size / 1e6)

    embeddings = np.vstack(chunks)
    log.info("Recompiled embeddings: %s (%.1f MB)", embeddings.shape, embeddings.nbytes / 1e6)

    return embeddings


# ---------------------------------------------------------------------------
# Qdrant operations
# ---------------------------------------------------------------------------
def ensure_collection(client, force_recreate=False):
    """Create collection only if it doesn't exist."""
    from qdrant_client.models import (
        Distance, VectorParams, ScalarQuantization,
        ScalarQuantizationConfig, ScalarType,
        SparseVectorParams, SparseIndexParams,
    )

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME in names:
        if force_recreate:
            log.warning("FORCE RECREATE: Deleting collection '%s'", COLLECTION_NAME)
            client.delete_collection(COLLECTION_NAME)
        else:
            info = client.get_collection(COLLECTION_NAME)
            log.info("Collection '%s' exists with %d points. REUSING.", COLLECTION_NAME, info.points_count)
            return

    log.info("Creating collection '%s' ...", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE, on_disk=False)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
        ),
        on_disk_payload=False,
    )
    log.info("  Collection created.")


def ingest_batch(client, points, max_retries=3, base_delay=2.0):
    """Upsert a batch with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            return len(points)
        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)
            log.warning("  Batch attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, max_retries, e, delay)
            time.sleep(delay)
    raise last_error


def save_checkpoint(path, data):
    """Atomic checkpoint write."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            log.warning("Corrupt checkpoint, starting fresh")
            return None
    return None


def verify_collection(client, expected_count):
    """Verify collection has expected number of points."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        actual = info.points_count
        if actual == expected_count:
            log.info("VERIFY OK: Collection has %d points (expected %d)", actual, expected_count)
            return True
        else:
            log.error("VERIFY FAIL: Collection has %d points (expected %d)", actual, expected_count)
            return False
    except Exception as e:
        log.error("VERIFY ERROR: %s", e)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Recompile chunks and ingest into Qdrant")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with output/ and chunks/ subdirectories")
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force-recreate", action="store_true",
                        help="Delete existing collection and start fresh")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify collection integrity")
    args = parser.parse_args()

    if args.collection:
        global COLLECTION_NAME
        COLLECTION_NAME = args.collection

    input_dir = Path(args.input_dir)
    payloads_path = input_dir / "output" / "enhanced_payloads.parquet"
    meta_path = input_dir / "output" / "corpus_meta.json"
    chunks_dir = input_dir / "chunks"

    log.info("=" * 70)
    log.info("Guardrailer Chunked Ingestion")
    log.info("=" * 70)

    # Verify files exist
    for p in [payloads_path, meta_path]:
        if not p.exists():
            log.error("Missing file: %s", p)
            sys.exit(1)

    if not chunks_dir.exists():
        log.error("Missing directory: %s", chunks_dir)
        sys.exit(1)

    # Load payloads
    log.info("Loading payloads from %s ...", payloads_path)
    df = pd.read_parquet(payloads_path)
    log.info("  Rows: %d, Columns: %s", len(df), list(df.columns))

    # Load corpus meta
    log.info("Loading corpus_meta from %s ...", meta_path)
    with open(meta_path) as f:
        corpus_meta = json.load(f)
    log.info("  IDF keywords: %d", len(corpus_meta.get("keyword_idf", {})))
    log.info("  Categories: %s", list(corpus_meta.get("category_centroids", {}).keys()))

    # Recompile embeddings
    log.info("Recompiling embedding chunks ...")
    embeddings = recompile_chunks(chunks_dir)

    if len(df) != embeddings.shape[0]:
        log.error("Mismatch: %d payloads vs %d embeddings", len(df), embeddings.shape[0])
        sys.exit(1)

    # Connect to Qdrant
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        log.error("qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    log.info("Connecting to Qdrant at %s ...", QDRANT_URL)
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)

    try:
        client.get_collections()
        log.info("  Connected.")
    except Exception as e:
        log.error("  Failed to connect: %s", e)
        sys.exit(1)

    # Verify mode
    if args.verify:
        ok = verify_collection(client, len(df))
        sys.exit(0 if ok else 1)

    # Load checkpoint
    checkpoint_path = str(input_dir / "ingest_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint and checkpoint.get("completed"):
        log.info("Previous ingestion completed. Starting fresh.")
        checkpoint = None

    start_idx = checkpoint.get("last_idx", 0) if checkpoint else 0
    total_points = checkpoint.get("total_points", 0) if checkpoint else 0

    # Ensure collection
    ensure_collection(client, force_recreate=args.force_recreate)

    idf_values = corpus_meta.get("keyword_idf", {})
    avgdl = corpus_meta.get("avg_doc_length", 100.0)
    N = corpus_meta.get("total_documents", len(df))

    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    if start_idx > 0:
        log.info("RESUMING from index %d: %d remaining in %d batches",
                 start_idx, len(df) - start_idx, total_batches)
    else:
        log.info("Starting fresh: %d samples in %d batches", len(df), total_batches)

    stop_flag = {"stop": False}

    def signal_handler(sig, frame):
        log.info("\nReceived signal %s. Saving checkpoint ...", sig)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for batch_start in range(start_idx, len(df), args.batch_size):
        if stop_flag["stop"]:
            break

        batch_end = min(batch_start + args.batch_size, len(df))
        batch_num = batch_start // args.batch_size + 1

        from qdrant_client.models import PointStruct

        points = []
        for i in range(batch_start, batch_end):
            row = df.iloc[i]
            sparse = build_idf_sparse_vector(row["prompt_text"], idf_values, avgdl, N)

            payload = {
                "prompt_text": str(row["prompt_text"])[:500],
                "is_malicious": bool(row["is_malicious"]),
                "attack_category": str(row["attack_category"]),
                "attack_technique": str(row["attack_technique"]),
                "risk_level": str(row["risk_level"]),
                "source_dataset": str(row["source_dataset"]),
                "cluster_id": int(row.get("cluster_id", 0)),
                "neighbor_ids": [int(x) for x in row.get("neighbor_ids", [])],
                "uniqueness": float(row.get("uniqueness", 0.5)),
                "cross_encoder_score": float(row.get("cross_encoder_score", 0.0)),
                "length_norm": float(row.get("length_norm", 0.0)),
            }

            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "dense": embeddings[i].tolist(),
                        "sparse": sparse,
                    },
                    payload=payload,
                )
            )

        try:
            count = ingest_batch(client, points)
            total_points += count
        except Exception as e:
            log.error("  Batch %d failed: %s", batch_num, e)

        # Checkpoint every batch
        save_checkpoint(checkpoint_path, {
            "last_idx": batch_end,
            "total_points": total_points,
            "timestamp": datetime.now().isoformat(),
        })

        if batch_num % 10 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            rate = total_points / elapsed if elapsed > 0 else 0
            remaining = len(df) - batch_end
            eta = remaining / rate if rate > 0 else 0
            log.info("  Batch %d/%d | Points: %d/%d | Rate: %.0f pts/s | ETA: %.0fs",
                     batch_num, total_batches, total_points, len(df), rate, eta)

    # Final checkpoint
    save_checkpoint(checkpoint_path, {
        "last_idx": len(df),
        "total_points": total_points,
        "timestamp": datetime.now().isoformat(),
        "completed": True,
    })

    elapsed_total = time.time() - start_time
    log.info("=" * 70)
    log.info("Ingestion complete.")
    log.info("  Total points: %d", total_points)
    log.info("  Total time:   %.1fs", elapsed_total)
    log.info("  Rate:         %.0f pts/s", total_points / elapsed_total if elapsed_total > 0 else 0)

    if verify_collection(client, len(df)):
        log.info("  Data integrity: VERIFIED")
    else:
        log.warning("  Data integrity: MISMATCH")

    # Copy corpus_meta to guardrailer_security/
    target_meta = Path(__file__).resolve().parent / "corpus_meta.json"
    import shutil
    shutil.copy2(meta_path, target_meta)
    log.info("  Copied corpus_meta.json to %s", target_meta)

    log.info("=" * 70)
    log.info("Ready. Start the engine with: python3 security_engine.py")


if __name__ == "__main__":
    main()
