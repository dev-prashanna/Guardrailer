"""
scoring.py
Multi-signal scoring module for Guardrailer RAG.

Combines dense semantic similarity, IDF-weighted BM25, category centroid distance,
cross-encoder pre-scores, text length normalization, and uniqueness into a single
composite score for prompt security evaluation.

Phase 2: Extended to support ensemble embedding signals.
Phase 3: Added improved scoring with learned weights, new signals, calibration, ensemble.
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

# Phase 3: Import improved scoring components
try:
    from improved_scoring import (
        compute_perplexity_score,
        compute_entropy_score,
        compute_token_frequency_score,
        compute_ngram_overlap_score,
        ImprovedScorer,
    )
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

CATEGORY_DESCRIPTIONS = {
    "jailbreak": "prompt designed to bypass AI safety restrictions and make the model behave without guidelines",
    "direct_injection": "prompt that directly attempts to override system instructions or prepend new ones",
    "indirect_injection": "prompt embedded in external content that tries to manipulate AI behavior",
    "system_prompt_extraction": "prompt attempting to reveal or leak the system prompt or hidden instructions",
    "refusal_bypass": "prompt trying to make the model refuse less or comply with harmful requests",
    "benign_control": "legitimate non-malicious prompt used as a control sample",
}

# Phase 2: Updated scoring weights with ensemble support
DEFAULT_WEIGHTS = {
    "dense": 0.30,
    "sparse_idf": 0.18,
    "centroid": 0.12,
    "cross_encoder": 0.12,
    "perplexity": 0.05,
    "entropy": 0.05,
    "token_frequency": 0.03,
    "ngram_overlap": 0.02,
    "uniqueness": 0.05,
    "length_norm": 0.05,
    "ensemble_bonus": 0.03,
}

# ---------------------------------------------------------------------------
# Corpus metadata (loaded once at startup)
# ---------------------------------------------------------------------------

_corpus_meta = None
_META_PATH = Path(__file__).resolve().parent / "corpus_meta.json"


def load_corpus_meta() -> dict:
    global _corpus_meta
    if _corpus_meta is not None:
        return _corpus_meta
    if _META_PATH.exists():
        with open(_META_PATH) as f:
            _corpus_meta = json.load(f)
    else:
        _corpus_meta = {}
    return _corpus_meta


def save_corpus_meta(meta: dict):
    global _corpus_meta
    _corpus_meta = meta
    tmp = str(_META_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, str(_META_PATH))


# ---------------------------------------------------------------------------
# IDF-weighted sparse vector
# ---------------------------------------------------------------------------

def build_idf_sparse_vector(text: str, meta: Optional[dict] = None) -> dict:
    """Build an IDF-weighted sparse vector from keyword matching.

    Uses pre-computed IDF values from corpus metadata instead of binary 0/1.
    Falls back to uniform weight if IDF values are not available.
    """
    from qdrant_client.models import SparseVector

    if meta is None:
        meta = load_corpus_meta()

    idf_values = meta.get("keyword_idf", {})
    avgdl = meta.get("avg_doc_length", 100.0)
    N = meta.get("total_documents", 1)

    lower = text.lower()
    indices = []
    values = []

    for i, keyword in enumerate(SPARSE_KEYWORDS):
        if keyword in lower:
            idf = idf_values.get(keyword, math.log(N / 2.0))
            tf = lower.count(keyword) / max(len(lower.split()), 1)
            bm25_tf = (tf * 2.0) / (tf + 1.5 * (1.0 - 0.75 + 0.75 * len(lower.split()) / max(avgdl, 1)))
            score = idf * (bm25_tf + 1.0)
            indices.append(i)
            values.append(score)

    return SparseVector(indices=indices, values=values)


def compute_text_idf_score(text: str, meta: Optional[dict] = None) -> float:
    """Compute the sum of IDF scores for keywords present in the text."""
    if meta is None:
        meta = load_corpus_meta()

    idf_values = meta.get("keyword_idf", {})
    N = meta.get("total_documents", 1)
    lower = text.lower()

    total_idf = 0.0
    for keyword in SPARSE_KEYWORDS:
        if keyword in lower:
            idf = idf_values.get(keyword, math.log(N / 2.0))
            total_idf += idf

    max_possible = sum(idf_values.values()) if idf_values else len(SPARSE_KEYWORDS)
    if max_possible > 0:
        return total_idf / max_possible
    return 0.0


# ---------------------------------------------------------------------------
# Category centroid distance
# ---------------------------------------------------------------------------

_centroids = None


def load_centroids(meta: Optional[dict] = None) -> dict:
    """Load category centroids from corpus metadata."""
    global _centroids
    if _centroids is not None:
        return _centroids
    if meta is None:
        meta = load_corpus_meta()

    raw = meta.get("category_centroids", {})
    _centroids = {}
    for cat, vec in raw.items():
        _centroids[cat] = np.array(vec, dtype=np.float32)
    return _centroids


def compute_centroid_scores(query_embedding: np.ndarray, meta: Optional[dict] = None) -> dict:
    """Compute cosine similarity to each category centroid.

    Returns dict of {category: score} where score is in [0, 1].
    """
    centroids = load_centroids(meta)
    if not centroids:
        return {}

    q_norm = np.linalg.norm(query_embedding)
    if q_norm < 1e-8:
        return {}

    scores = {}
    for cat, centroid in centroids.items():
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-8:
            scores[cat] = 0.0
        else:
            scores[cat] = float(np.dot(query_embedding, centroid) / (q_norm * c_norm))
    return scores


def best_centroid_score(query_embedding: np.ndarray, meta: Optional[dict] = None) -> float:
    """Return the maximum centroid similarity across all categories."""
    scores = compute_centroid_scores(query_embedding, meta)
    if not scores:
        return 0.0
    return max(scores.values())


def best_centroid_category(query_embedding: np.ndarray, meta: Optional[dict] = None) -> str:
    """Return the category with the highest centroid similarity."""
    scores = compute_centroid_scores(query_embedding, meta)
    if not scores:
        return "unknown"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Text length normalization
# ---------------------------------------------------------------------------

def compute_length_normalization(text_length: int, meta: Optional[dict] = None) -> float:
    """Compute a length normalization factor.

    Longer texts tend to have lower cosine similarity due to the curse of
    dimensionality. This factor centers around 1.0 for average-length texts.
    """
    if meta is None:
        meta = load_corpus_meta()

    avg_length = meta.get("avg_text_length", 200.0)
    if avg_length <= 0:
        avg_length = 200.0

    return math.log1p(text_length) / math.log1p(avg_length)


# ---------------------------------------------------------------------------
# Uniqueness score
# ---------------------------------------------------------------------------

def get_uniqueness(point_payload: dict) -> float:
    """Extract pre-computed uniqueness score from point payload."""
    return point_payload.get("uniqueness", 0.5)


# ---------------------------------------------------------------------------
# Cross-encoder pre-score
# ---------------------------------------------------------------------------

def get_cross_encoder_score(point_payload: dict) -> float:
    """Extract pre-computed cross-encoder score from point payload."""
    return point_payload.get("cross_encoder_score", 0.5)


# ---------------------------------------------------------------------------
# Combined multi-signal scoring
# ---------------------------------------------------------------------------

def compute_composite_score(
    dense_score: float,
    text: str,
    query_embedding: np.ndarray,
    point_payload: dict,
    weights: Optional[dict] = None,
    meta: Optional[dict] = None,
    ensemble_agreement: Optional[float] = None,
) -> dict:
    """Compute the combined multi-signal composite score.

    Phase 2: Added ensemble_agreement parameter for multi-model bonus.
    Phase 3: Added perplexity, entropy, token frequency, n-gram overlap signals.

    Returns a dict with the composite score and individual signal values.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if meta is None:
        meta = load_corpus_meta()

    # 1. Dense semantic similarity (already from Qdrant)
    s_dense = max(0.0, min(1.0, dense_score))

    # 2. IDF-weighted sparse/BM25 score
    s_sparse = compute_text_idf_score(text, meta)

    # 3. Category centroid distance
    s_centroid = best_centroid_score(query_embedding, meta)

    # 4. Cross-encoder pre-score
    s_cross = get_cross_encoder_score(point_payload)

    # 5. Uniqueness bonus
    s_uniqueness = get_uniqueness(point_payload)

    # 6. Text length normalization
    text_len = len(text)
    s_length = compute_length_normalization(text_len, meta)

    # 7. Ensemble agreement bonus (Phase 2)
    s_ensemble = ensemble_agreement if ensemble_agreement is not None else 0.5

    # Phase 3: New signals
    s_perplexity = 0.0
    s_entropy = 0.0
    s_token_freq = 0.0
    s_ngram_overlap = 0.0

    if PHASE3_AVAILABLE:
        s_perplexity = compute_perplexity_score(text)
        s_entropy = compute_entropy_score(text)
        s_token_freq = compute_token_frequency_score(text)
        s_ngram_overlap = compute_ngram_overlap_score(text)

    # Weighted combination
    composite = (
        weights.get("dense", 0.30) * s_dense
        + weights.get("sparse_idf", 0.18) * s_sparse
        + weights.get("centroid", 0.12) * s_centroid
        + weights.get("cross_encoder", 0.12) * s_cross
        + weights.get("perplexity", 0.05) * s_perplexity
        + weights.get("entropy", 0.05) * s_entropy
        + weights.get("token_frequency", 0.03) * s_token_freq
        + weights.get("ngram_overlap", 0.02) * s_ngram_overlap
        + weights.get("uniqueness", 0.05) * s_uniqueness
        + weights.get("length_norm", 0.05) * s_length
        + weights.get("ensemble_bonus", 0.03) * s_ensemble
    )

    return {
        "composite_score": composite,
        "dense_score": s_dense,
        "sparse_idf_score": s_sparse,
        "centroid_score": s_centroid,
        "cross_encoder_score": s_cross,
        "uniqueness_score": s_uniqueness,
        "length_norm_score": s_length,
        "ensemble_agreement": s_ensemble,
        "perplexity_score": s_perplexity,
        "entropy_score": s_entropy,
        "token_frequency_score": s_token_freq,
        "ngram_overlap_score": s_ngram_overlap,
    }


# ---------------------------------------------------------------------------
# Cluster-aware search helpers
# ---------------------------------------------------------------------------

def get_cluster_id(point_payload: dict) -> int:
    """Extract pre-computed cluster assignment from point payload."""
    return point_payload.get("cluster_id", -1)


def get_neighbor_ids(point_payload: dict) -> list[int]:
    """Extract pre-computed neighbor IDs from point payload."""
    return point_payload.get("neighbor_ids", [])


# ---------------------------------------------------------------------------
# Score normalization utilities
# ---------------------------------------------------------------------------

def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-8:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Sigmoid function for score calibration."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))
