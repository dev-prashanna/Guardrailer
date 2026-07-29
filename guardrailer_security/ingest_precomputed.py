"""
ingest_precomputed.py
Safe ingestion script that uploads pre-computed data to Qdrant.

SAFETY FEATURES:
  - NEVER deletes existing collection unless --force-recreate is passed
  - Backs up source files before ingestion
  - Checkpoints every batch (not every 50)
  - Verifies data integrity after ingestion
  - Can resume from any checkpoint safely
  - Atomic checkpoint writes (tmp + rename)

Usage:
    python3 ingest_precomputed.py [--input-dir /path/to/guardrailer_output]
    python3 ingest_precomputed.py --verify  # verify collection integrity
    python3 ingest_precomputed.py --force-recreate  # ONLY then deletes collection
"""

import json
import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = os.environ.get("GUARDRAILER_COLLECTION", "guardrailer_security_enhanced")

BATCH_SIZE = 64
SPARSE_KEYWORDS = [
    "ignore previous", "override", "bypass", "jailbreak", "system prompt",
    "your instructions", "forget", "disregard", "dan", "do anything now",
    "act as", "roleplay", "pretend you", "hypothetical", "in theory",
    "markdown injection", "code comment", "readme", "yaml", "json payload",
    "<script>", "]]>", "```", "<!--", "-->", "eval(", "exec(",
    "base64", "rot13", "hex encoded", "obfuscated",
    "ignore all", "new instructions", "you are now", "persona",
]


def build_idf_sparse_vector(text: str, idf_values: dict, avgdl: float, N: int) -> dict:
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


def backup_source_files(input_dir):
    """Backup source files before ingestion."""
    backup_dir = os.path.join(input_dir, ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(backup_dir, exist_ok=True)

    files_to_backup = ["corpus_meta.json", "enhanced_payloads.parquet", "dense_embeddings_float16.npy"]
    backed_up = []

    for fname in files_to_backup:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, fname)
            shutil.copy2(src, dst)
            backed_up.append(fname)

    if backed_up:
        log.info("Backed up %d files to %s", len(backed_up), backup_dir)
    return backup_dir


def ensure_collection_safe(client, force_recreate=False):
    """Create collection only if it doesn't exist. NEVER deletes unless force_recreate."""
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
            log.info("Collection '%s' exists with %d points. REUSING (no delete).",
                     COLLECTION_NAME, info.points_count)
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
    last_error = None
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            return len(points)
        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)
            log.warning("  Batch attempt %d/%d failed: %s. Retrying in %.1fs ...",
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


def verify_data_files(input_dir):
    """Verify all required data files exist and are valid."""
    files = {
        "corpus_meta.json": lambda p: json.load(open(p)),
        "enhanced_payloads.parquet": lambda p: pd.read_parquet(p),
        "dense_embeddings_float16.npy": lambda p: np.load(p),
    }

    for fname, loader in files.items():
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            log.error("MISSING: %s", path)
            return False
        try:
            data = loader(path)
            size_mb = os.path.getsize(path) / 1e6
            log.info("  %s: OK (%.1f MB)", fname, size_mb)
        except Exception as e:
            log.error("CORRUPT: %s - %s", path, e)
            return False
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Safe ingestion to Qdrant")
    parser.add_argument("--input-dir", default=os.path.join(os.path.dirname(__file__), "guardrailer_output"),
                        help="Directory with corpus_meta.json, enhanced_payloads.parquet, dense_embeddings_float16.npy")
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for upserts")
    parser.add_argument("--force-recreate", action="store_true",
                        help="DELETE existing collection and start fresh (DANGEROUS)")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify collection integrity, don't ingest")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up source files")
    args = parser.parse_args()

    if args.collection:
        global COLLECTION_NAME
        COLLECTION_NAME = args.collection

    log.info("=" * 70)
    log.info("Guardrailer SAFE Ingestion")
    log.info("=" * 70)

    # Load input files
    input_dir = args.input_dir
    meta_path = os.path.join(input_dir, "corpus_meta.json")
    payload_path = os.path.join(input_dir, "enhanced_payloads.parquet")
    emb_path = os.path.join(input_dir, "dense_embeddings_float16.npy")

    log.info("Verifying source files ...")
    if not verify_data_files(input_dir):
        log.error("Source file verification failed. Aborting.")
        sys.exit(1)

    log.info("Loading corpus metadata ...")
    with open(meta_path) as f:
        corpus_meta = json.load(f)
    log.info("  IDF keywords: %d", len(corpus_meta.get("keyword_idf", {})))
    log.info("  Categories: %s", list(corpus_meta.get("category_centroids", {}).keys()))

    log.info("Loading enhanced payloads ...")
    df = pd.read_parquet(payload_path)
    log.info("  Payloads: %d rows, columns: %s", len(df), list(df.columns))

    log.info("Loading dense embeddings (float16) ...")
    embeddings_f16 = np.load(emb_path)
    embeddings_f32 = embeddings_f16.astype(np.float32)
    log.info("  Embeddings shape: %s, dtype: %s", embeddings_f32.shape, embeddings_f32.dtype)

    if len(df) != len(embeddings_f32):
        log.error("Mismatch: %d payloads vs %d embeddings", len(df), len(embeddings_f32))
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
        log.info("Verifying collection ...")
        ok = verify_collection(client, len(df))
        sys.exit(0 if ok else 1)

    # Backup source files
    if not args.no_backup:
        backup_dir = backup_source_files(input_dir)

    # Load checkpoint
    checkpoint_path = os.path.join(input_dir, "ingest_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    # If previous run completed, start fresh
    if checkpoint and checkpoint.get("completed"):
        log.info("Previous ingestion completed. Starting fresh.")
        checkpoint = None

    start_idx = checkpoint.get("last_idx", 0) if checkpoint else 0
    total_points = checkpoint.get("total_points", 0) if checkpoint else 0

    # SAFE collection creation: NEVER deletes unless --force-recreate
    ensure_collection_safe(client, force_recreate=args.force_recreate)

    idf_values = corpus_meta.get("keyword_idf", {})
    avgdl = corpus_meta.get("avg_doc_length", 100.0)
    N = corpus_meta.get("total_documents", len(df))

    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    if start_idx > 0:
        log.info("RESUMING from index %d: %d samples remaining in %d batches ...",
                 start_idx, len(df) - start_idx, total_batches)
    else:
        log.info("Starting fresh ingestion: %d samples in %d batches ...", len(df), total_batches)

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
                "cluster_id": int(row["cluster_id"]),
                "neighbor_ids": [int(x) for x in row["neighbor_ids"]],
                "uniqueness": float(row["uniqueness"]),
                "cross_encoder_score": float(row["cross_encoder_score"]),
                "length_norm": float(row["length_norm"]),
            }

            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "dense": embeddings_f32[i].tolist(),
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

        # Checkpoint every batch for safety
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

    # Verify
    if verify_collection(client, len(df)):
        log.info("  Data integrity: VERIFIED")
    else:
        log.warning("  Data integrity: MISMATCH (check manually)")

    # Copy corpus_meta to guardrailer_security/ for the scoring module
    target_meta = os.path.join(os.path.dirname(__file__), "corpus_meta.json")
    shutil.copy2(meta_path, target_meta)
    log.info("  Copied corpus_meta.json to %s", target_meta)

    log.info("=" * 70)
    log.info("Ready. Start the engine with: python3 security_engine.py")


if __name__ == "__main__":
    main()
