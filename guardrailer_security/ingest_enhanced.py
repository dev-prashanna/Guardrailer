"""
ingest_enhanced.py
Enhanced ingestion pipeline that pre-computes all multi-signal features
for the Guardrailer RAG scoring system.

Pre-computed features stored per point:
  - dense embeddings (BAAI/bge-large-en-v1.5, 1024-dim)
  - sparse IDF-weighted vectors (BM25-style keyword matching)
  - cluster_id (k-means cluster assignment)
  - neighbor_ids (top-5 nearest neighbors in embedding space)
  - uniqueness_score (inverse mean distance to k nearest neighbors)
  - cross_encoder_score (cross-encoder relevance to attack category)
  - length_norm (text length normalization factor)

Corpus-level metadata saved to corpus_meta.json:
  - keyword_idf: IDF value per keyword
  - avg_doc_length: average document length (words)
  - avg_text_length: average text length (characters)
  - total_documents: total document count
  - category_centroids: mean embedding per attack category
  - cluster_centers: k-means cluster centers
  - scoring_weights: optimal weights for composite scoring
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

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

PARQUET_PATH = os.path.join(os.path.dirname(__file__), "unified_security_dataset.parquet")
CORPUS_META_PATH = os.path.join(os.path.dirname(__file__), "corpus_meta.json")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = "guardrailer_security_enhanced"

DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DENSE_DIMENSION = 1024

BATCH_SIZE = 64
N_CLUSTERS = 128
N_NEIGHBORS = 10
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SPARSE_KEYWORDS = [
    "ignore previous", "override", "bypass", "jailbreak", "system prompt",
    "your instructions", "forget", "disregard", "dan", "do anything now",
    "act as", "roleplay", "pretend you", "hypothetical", "in theory",
    "markdown injection", "code comment", "readme", "yaml", "json payload",
    "<script>", "]]>", "```", "<!--", "-->", "eval(", "exec(",
    "base64", "rot13", "hex encoded", "obfuscated",
    "ignore all", "new instructions", "you are now", "persona",
]

CATEGORY_DESCRIPTIONS = {
    "jailbreak": "prompt designed to bypass AI safety restrictions and make the model behave without guidelines",
    "direct_injection": "prompt that directly attempts to override system instructions or prepend new ones",
    "indirect_injection": "prompt embedded in external content that tries to manipulate AI behavior",
    "system_prompt_extraction": "prompt attempting to reveal or leak the system prompt or hidden instructions",
    "refusal_bypass": "prompt trying to make the model refuse less or comply with harmful requests",
    "benign_control": "legitimate non-malicious prompt used as a control sample",
}


# ---------------------------------------------------------------------------
# Phase 1: Corpus statistics
# ---------------------------------------------------------------------------

def compute_keyword_idf(df: pd.DataFrame) -> dict:
    """Compute IDF for each keyword across the corpus."""
    N = len(df)
    df_counts = {kw: 0 for kw in SPARSE_KEYWORDS}

    for text in df["prompt_text"]:
        lower = text.lower()
        for kw in SPARSE_KEYWORDS:
            if kw in lower:
                df_counts[kw] += 1

    idf = {}
    for kw in SPARSE_KEYWORDS:
        idf[kw] = math.log((N + 1) / (df_counts[kw] + 1)) + 1.0

    return idf


def compute_doc_length_stats(df: pd.DataFrame) -> dict:
    """Compute document length statistics."""
    word_counts = df["prompt_text"].str.split().str.len()
    char_counts = df["prompt_text"].str.len()

    return {
        "avg_doc_length": float(word_counts.mean()),
        "median_doc_length": float(word_counts.median()),
        "avg_text_length": float(char_counts.mean()),
        "total_documents": len(df),
    }


# ---------------------------------------------------------------------------
# Phase 2: Dense embeddings
# ---------------------------------------------------------------------------

def load_dense_model():
    from sentence_transformers import SentenceTransformer
    log.info("Loading dense model: %s", DENSE_MODEL_NAME)
    model = SentenceTransformer(DENSE_MODEL_NAME)
    log.info("  Model loaded. Dimension: %d", model.get_embedding_dimension())
    return model


def compute_all_embeddings(model, texts: list[str]) -> np.ndarray:
    """Compute dense embeddings for all texts."""
    log.info("Computing dense embeddings for %d texts ...", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    log.info("  Embeddings computed. Shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# Phase 3: Category centroids
# ---------------------------------------------------------------------------

def compute_category_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """Compute mean embedding per attack category."""
    centroids = {}
    for category in df["attack_category"].unique():
        mask = df["attack_category"] == category
        cat_embeddings = embeddings[mask]
        centroid = cat_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[category] = centroid.tolist()
        log.info("  Centroid for '%s': %d samples", category, int(mask.sum()))
    return centroids


# ---------------------------------------------------------------------------
# Phase 4: k-Means clustering
# ---------------------------------------------------------------------------

def compute_cluster_assignments(embeddings: np.ndarray) -> tuple:
    """Assign each document to a cluster using MiniBatchKMeans."""
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(N_CLUSTERS, len(embeddings) // 10)
    log.info("Computing %d cluster assignments ...", n_clusters)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    log.info("  Cluster distribution: min=%d, max=%d, mean=%.1f",
             cluster_ids.min(), cluster_ids.max(),
             cluster_ids.mean())

    return kmeans, cluster_ids


# ---------------------------------------------------------------------------
# Phase 5: Neighbor graph
# ---------------------------------------------------------------------------

def compute_neighbor_graph(embeddings: np.ndarray, n_neighbors: int = 10) -> tuple:
    """Compute k-nearest neighbors for each document."""
    from sklearn.neighbors import NearestNeighbors

    log.info("Computing neighbor graph (k=%d) ...", n_neighbors)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    log.info("  Neighbor graph computed. Mean distance to nearest: %.4f",
             distances[:, 1].mean())

    return distances, indices


# ---------------------------------------------------------------------------
# Phase 6: Uniqueness scores
# ---------------------------------------------------------------------------

def compute_uniqueness_scores(distances: np.ndarray) -> np.ndarray:
    """Compute uniqueness as inverse mean distance to k nearest neighbors."""
    mean_distances = distances[:, 1:].mean(axis=1)
    uniqueness = 1.0 / (mean_distances + 1e-8)
    uniqueness = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min() + 1e-8)
    return uniqueness


# ---------------------------------------------------------------------------
# Phase 7: Cross-encoder pre-scoring
# ---------------------------------------------------------------------------

def compute_cross_encoder_scores(df: pd.DataFrame) -> np.ndarray:
    """Pre-score each document against its attack category description."""
    try:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder model: %s", CROSS_ENCODER_MODEL)
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    except ImportError:
        log.warning("sentence-transformers CrossEncoder not available. Using category-based heuristic.")
        return _heuristic_cross_encoder_scores(df)

    pairs = []
    for _, row in df.iterrows():
        cat = row["attack_category"]
        desc = CATEGORY_DESCRIPTIONS.get(cat, cat)
        pairs.append((row["prompt_text"], desc))

    log.info("Computing cross-encoder scores for %d pairs ...", len(pairs))
    scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=True)

    scores = np.array(scores, dtype=np.float32)
    scores = 1.0 / (1.0 + np.exp(-scores))
    log.info("  Cross-encoder scores: min=%.4f, max=%.4f, mean=%.4f",
             scores.min(), scores.max(), scores.mean())

    return scores


def _heuristic_cross_encoder_scores(df: pd.DataFrame) -> np.ndarray:
    """Fallback heuristic scoring when cross-encoder is unavailable."""
    scores = np.zeros(len(df), dtype=np.float32)
    for i, row in df.iterrows():
        text = row["prompt_text"].lower()
        cat = row["attack_category"]

        score = 0.3
        if cat != "benign_control":
            if any(kw in text for kw in ["jailbreak", "bypass", "override", "ignore previous"]):
                score += 0.3
            if any(kw in text for kw in ["system prompt", "your instructions", "forget"]):
                score += 0.2
            if row.get("risk_level") in ("critical", "high"):
                score += 0.2

        scores[i] = min(1.0, score)

    return scores


# ---------------------------------------------------------------------------
# Phase 8: Build enhanced Qdrant points
# ---------------------------------------------------------------------------

def build_sparse_vector(text: str, idf_values: dict, avgdl: float, N: int) -> dict:
    """Build IDF-weighted sparse vector."""
    from qdrant_client.models import SparseVector

    lower = text.lower()
    indices = []
    values = []

    for i, keyword in enumerate(SPARSE_KEYWORDS):
        if keyword in lower:
            idf = idf_values.get(keyword, math.log(N / 2.0))
            word_count = len(lower.split())
            tf = lower.count(keyword) / max(word_count, 1)
            bm25_tf = (tf * 2.0) / (tf + 1.5 * (1.0 - 0.75 + 0.75 * word_count / max(avgdl, 1)))
            score = idf * (bm25_tf + 1.0)
            indices.append(i)
            values.append(score)

    return SparseVector(indices=indices, values=values)


def build_enhanced_points(
    df: pd.DataFrame,
    dense_embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    neighbor_indices: np.ndarray,
    uniqueness_scores: np.ndarray,
    cross_encoder_scores: np.ndarray,
    idf_values: dict,
    avgdl: float,
    start_id: int,
) -> list:
    """Build Qdrant PointStruct objects with all pre-computed features."""
    from qdrant_client.models import PointStruct

    points = []
    for i in range(len(df)):
        row = df.iloc[i]
        text_len = len(row["prompt_text"])
        avg_text_len = df["prompt_text"].str.len().mean()
        length_norm = math.log1p(text_len) / math.log1p(max(avg_text_len, 1))

        neighbor_ids = neighbor_indices[i][1:6].tolist()

        payload = {
            "prompt_text": row["prompt_text"][:500],
            "is_malicious": bool(row["is_malicious"]),
            "attack_category": str(row["attack_category"]),
            "attack_technique": str(row["attack_technique"]),
            "risk_level": str(row["risk_level"]),
            "source_dataset": str(row["source_dataset"]),
            "cluster_id": int(cluster_ids[i]),
            "neighbor_ids": neighbor_ids,
            "uniqueness": float(uniqueness_scores[i]),
            "cross_encoder_score": float(cross_encoder_scores[i]),
            "length_norm": float(length_norm),
        }

        sparse = build_sparse_vector(row["prompt_text"], idf_values, avgdl, len(df))

        point_id = start_id + i
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse,
                },
                payload=payload,
            )
        )

    return points


# ---------------------------------------------------------------------------
# Qdrant collection setup
# ---------------------------------------------------------------------------

def create_enhanced_collection(client):
    from qdrant_client.models import (
        Distance,
        VectorParams,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        SparseVectorParams,
        SparseIndexParams,
    )

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        log.info("Collection '%s' already exists. Deleting and recreating.", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    log.info("Creating enhanced collection '%s' ...", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIMENSION,
                distance=Distance.COSINE,
                on_disk=False,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
        on_disk_payload=False,
    )

    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config={"indexing_threshold": 20000},
        )
    except Exception:
        pass

    log.info("  Enhanced collection created with SQ8 + HNSW + sparse index.")


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def ingest_batch(client, points: list, max_retries: int = 3, base_delay: float = 2.0) -> int:
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
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import signal

    log.info("=" * 70)
    log.info("Guardrailer Enhanced Vector DB Ingestion")
    log.info("Pre-computing: IDF, centroids, clusters, neighbors, uniqueness, cross-encoder")
    log.info("=" * 70)

    if not os.path.exists(PARQUET_PATH):
        log.error("Parquet file not found at %s. Run download_and_merge.py first.", PARQUET_PATH)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    log.info("Loaded %d samples from Parquet", len(df))
    log.info("  Malicious: %d", int(df["is_malicious"].sum()))
    log.info("  Benign:    %d", int((~df["is_malicious"]).sum()))

    # --- Phase 1: Corpus statistics ---
    log.info("=" * 70)
    log.info("Phase 1: Computing corpus statistics ...")
    idf_values = compute_keyword_idf(df)
    length_stats = compute_doc_length_stats(df)
    log.info("  IDF computed for %d keywords", len(idf_values))
    log.info("  Avg doc length: %.1f words, %.1f chars",
             length_stats["avg_doc_length"], length_stats["avg_text_length"])

    # --- Phase 2: Dense embeddings ---
    log.info("=" * 70)
    log.info("Phase 2: Computing dense embeddings ...")
    dense_model = load_dense_model()
    texts = df["prompt_text"].tolist()
    dense_embeddings = compute_all_embeddings(dense_model, texts)

    # --- Phase 3: Category centroids ---
    log.info("=" * 70)
    log.info("Phase 3: Computing category centroids ...")
    category_centroids = compute_category_centroids(df, dense_embeddings)

    # --- Phase 4: k-Means clustering ---
    log.info("=" * 70)
    log.info("Phase 4: Computing cluster assignments ...")
    kmeans_model, cluster_ids = compute_cluster_assignments(dense_embeddings)

    # --- Phase 5: Neighbor graph ---
    log.info("=" * 70)
    log.info("Phase 5: Computing neighbor graph ...")
    distances, neighbor_indices = compute_neighbor_graph(dense_embeddings)

    # --- Phase 6: Uniqueness scores ---
    log.info("=" * 70)
    log.info("Phase 6: Computing uniqueness scores ...")
    uniqueness_scores = compute_uniqueness_scores(distances)
    log.info("  Uniqueness: min=%.4f, max=%.4f, mean=%.4f",
             uniqueness_scores.min(), uniqueness_scores.max(), uniqueness_scores.mean())

    # --- Phase 7: Cross-encoder pre-scoring ---
    log.info("=" * 70)
    log.info("Phase 7: Computing cross-encoder scores ...")
    cross_encoder_scores = compute_cross_encoder_scores(df)

    # --- Save corpus metadata ---
    log.info("=" * 70)
    log.info("Saving corpus metadata ...")
    corpus_meta = {
        "keyword_idf": idf_values,
        "total_documents": length_stats["total_documents"],
        "avg_doc_length": length_stats["avg_doc_length"],
        "avg_text_length": length_stats["avg_text_length"],
        "category_centroids": category_centroids,
        "cluster_centers": kmeans_model.cluster_centers_.tolist(),
        "scoring_weights": {
            "dense": 0.40,
            "sparse_idf": 0.20,
            "centroid": 0.15,
            "cross_encoder": 0.15,
            "uniqueness": 0.05,
            "length_norm": 0.05,
        },
    }
    tmp = CORPUS_META_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(corpus_meta, f, indent=2)
    os.replace(tmp, CORPUS_META_PATH)
    log.info("  Corpus metadata saved to %s", CORPUS_META_PATH)

    # --- Phase 8: Ingest into Qdrant ---
    log.info("=" * 70)
    log.info("Phase 8: Ingesting into Qdrant ...")

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        log.error("qdrant-client not installed.")
        sys.exit(1)

    log.info("Connecting to Qdrant at %s ...", QDRANT_URL)
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)

    try:
        client.get_collections()
        log.info("  Connected to Qdrant.")
    except Exception as e:
        log.error("  Failed to connect: %s", e)
        sys.exit(1)

    create_enhanced_collection(client)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "enhanced_ingestion_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    start_idx = checkpoint.get("last_idx", 0) if checkpoint else 0
    total_points = checkpoint.get("total_points", 0) if checkpoint else 0
    failed_batches = checkpoint.get("failed_batches", []) if checkpoint else []

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    log.info("Starting ingestion from index %d: %d samples in %d batches ...",
             start_idx, len(df), total_batches)

    stop_flag = {"stop": False}

    def signal_handler(sig, frame):
        log.info("\nReceived signal %s. Saving checkpoint ...", sig)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for batch_start in range(start_idx, len(df), BATCH_SIZE):
        if stop_flag["stop"]:
            break

        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_num = batch_start // BATCH_SIZE + 1

        batch_df = df.iloc[batch_start:batch_end].copy()

        points = build_enhanced_points(
            df=batch_df,
            dense_embeddings=dense_embeddings[batch_start:batch_end],
            cluster_ids=cluster_ids[batch_start:batch_end],
            neighbor_indices=neighbor_indices[batch_start:batch_end],
            uniqueness_scores=uniqueness_scores[batch_start:batch_end],
            cross_encoder_scores=cross_encoder_scores[batch_start:batch_end],
            idf_values=idf_values,
            avgdl=length_stats["avg_doc_length"],
            start_id=batch_start,
        )

        try:
            count = ingest_batch(client, points)
            total_points += count
        except Exception as e:
            log.error("  Batch %d failed: %s", batch_num, e)
            failed_batches.append(batch_num)

        if batch_num % 10 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            rate = total_points / elapsed if elapsed > 0 else 0
            log.info("  Batch %d/%d | Points: %d | Rate: %.0f pts/s | Elapsed: %.1fs",
                     batch_num, total_batches, total_points, rate, elapsed)

        if batch_num % 50 == 0:
            save_checkpoint(checkpoint_path, {
                "last_idx": batch_end,
                "total_points": total_points,
                "failed_batches": failed_batches,
            })

    save_checkpoint(checkpoint_path, {
        "last_idx": len(df),
        "total_points": total_points,
        "failed_batches": failed_batches,
    })

    elapsed_total = time.time() - start_time
    log.info("=" * 70)
    log.info("Enhanced ingestion complete.")
    log.info("  Total points:    %d", total_points)
    log.info("  Expected points: %d", len(df))
    log.info("  Failed batches:  %d / %d", len(failed_batches), total_batches)
    log.info("  Total time:      %.1fs", elapsed_total)
    log.info("  Average rate:    %.0f pts/s", total_points / elapsed_total if elapsed_total > 0 else 0)

    try:
        info = client.get_collection(COLLECTION_NAME)
        log.info("  Collection info:")
        log.info("    Points:       %d", info.points_count)
        log.info("    Indexed:      %d", info.indexed_vectors_count)
        log.info("    Status:       %s", info.status)
    except Exception as e:
        log.warning("  Could not retrieve collection info: %s", e)

    if total_points >= len(df) * 0.95:
        try:
            os.remove(checkpoint_path)
            log.info("  Checkpoint removed.")
        except Exception:
            pass

    log.info("=" * 70)
    log.info("Enhanced ingestion ready. Collection: %s", COLLECTION_NAME)
    log.info("Corpus metadata: %s", CORPUS_META_PATH)


if __name__ == "__main__":
    main()
