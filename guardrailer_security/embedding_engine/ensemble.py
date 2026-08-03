"""
ensemble.py
Ensemble embedding strategies for combining multiple model outputs.

Provides:
- Weighted average ensemble
- Late fusion with learned projection
- Max-similarity ensemble
- Adaptive weighting based on query characteristics
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import ModelConfig, get_model_config, get_ensemble_configs

log = logging.getLogger(__name__)


class EnsembleEmbedding:
    """
    Combines embeddings from multiple models using various strategies.
    
    Strategies:
    - weighted_average: Simple weighted mean of normalized embeddings
    - late_fusion: Concatenate + project to target dimension
    - max_sim: Per-dimension max absolute value selection
    - adaptive: Weight models based on query characteristics
    """

    STRATEGIES = ("weighted_average", "late_fusion", "max_sim", "adaptive")

    def __init__(
        self,
        strategy: str = "weighted_average",
        target_dimension: int = 1024,
        adaptive_weights: Optional[dict[str, float]] = None,
    ):
        self.strategy = strategy
        self.target_dimension = target_dimension
        self.adaptive_weights = adaptive_weights or {}
        self._projection_matrix: Optional[np.ndarray] = None

    def combine(
        self,
        embeddings: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """
        Combine multiple model embeddings.
        
        Args:
            embeddings: List of (embedding_array, model_weight, model_name) tuples.
        
        Returns:
            Combined embedding array.
        """
        if not embeddings:
            raise ValueError("No embeddings to combine")
        if len(embeddings) == 1:
            return embeddings[0][0]

        if self.strategy == "weighted_average":
            return self._weighted_average(embeddings)
        elif self.strategy == "late_fusion":
            return self._late_fusion(embeddings)
        elif self.strategy == "max_sim":
            return self._max_similarity(embeddings)
        elif self.strategy == "adaptive":
            return self._adaptive_combine(embeddings)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_average(
        self,
        embeddings: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """Weighted average with optional adaptive reweighting."""
        total_weight = 0.0
        combined = None

        for emb, base_weight, name in embeddings:
            adaptive_mult = self.adaptive_weights.get(name, 1.0)
            weight = base_weight * adaptive_mult
            total_weight += weight

            weighted = emb * weight
            if combined is None:
                combined = weighted
            else:
                combined = combined + weighted

        if total_weight < 1e-8:
            total_weight = 1.0
        combined = combined / total_weight

        return self._normalize(combined)

    def _late_fusion(
        self,
        embeddings: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """
        Late fusion: concatenate all model outputs and project.
        
        Preserves per-model information better than averaging.
        Uses a fixed projection matrix (diagonal blocks for stability).
        """
        all_embs = [emb.reshape(-1) for emb, _, _ in embeddings]
        concatenated = np.concatenate(all_embs, axis=-1)
        total_dim = concatenated.shape[-1]

        if total_dim == self.target_dimension:
            result = concatenated
        elif total_dim > self.target_dimension:
            # Chunked average pooling to reduce dimension
            chunk_size = total_dim // self.target_dimension
            n_chunks = total_dim // chunk_size
            truncated = concatenated[:n_chunks * chunk_size]
            result = truncated.reshape(n_chunks, chunk_size).mean(axis=1)
        else:
            # Zero-pad to target dimension
            pad_size = self.target_dimension - total_dim
            result = np.pad(concatenated, (0, pad_size))

        result = self._normalize(result.reshape(1, -1))
        return result[0]

    def _max_similarity(
        self,
        embeddings: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """
        Per-dimension max absolute value selection.
        
        Each dimension takes the value from whichever model is most
        confident (highest absolute value) for that dimension.
        """
        all_embs = [emb.reshape(-1) for emb, _, _ in embeddings]
        stacked = np.stack(all_embs, axis=0)  # (n_models, dim)
        abs_stacked = np.abs(stacked)
        max_indices = np.argmax(abs_stacked, axis=0)  # (dim,)

        # Use advanced indexing to gather values
        result = stacked[max_indices, np.arange(stacked.shape[1])]

        result = self._normalize(result.reshape(1, -1))
        return result[0]

    def _adaptive_combine(
        self,
        embeddings: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """
        Adaptive weighting based on inter-model agreement.
        
        Models that agree with the majority get higher weight.
        Models that disagree get down-weighted.
        """
        if len(embeddings) <= 2:
            return self._weighted_average(embeddings)

        # Compute pairwise similarities
        all_embs = [emb.reshape(-1) for emb, _, _ in embeddings]
        names = [name for _, _, name in embeddings]

        # Compute centroid of all embeddings
        stacked = np.stack(all_embs, axis=0)  # (n_models, dim)
        centroid = stacked.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-8:
            return self._weighted_average(embeddings)

        # Weight by similarity to centroid (agreement with majority)
        adaptive_weights = {}
        for i, (emb, base_weight, name) in enumerate(embeddings):
            emb_1d = emb.reshape(-1)
            emb_norm = np.linalg.norm(emb_1d)
            if emb_norm < 1e-8:
                similarity = 0.0
            else:
                similarity = float(np.dot(emb_1d, centroid) / (emb_norm * centroid_norm))
            # Boost models that agree with majority
            adaptive_weights[name] = max(0.1, similarity) * base_weight

        self.adaptive_weights = adaptive_weights
        return self._weighted_average(embeddings)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    def compute_alignment_score(
        self,
        embeddings: list[np.ndarray],
    ) -> float:
        """
        Compute alignment score between multiple model embeddings.
        
        Higher alignment means models agree more. Used for confidence estimation.
        """
        if len(embeddings) <= 1:
            return 1.0

        stacked = np.stack(embeddings, axis=0)
        centroid = stacked.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-8:
            return 0.0

        similarities = []
        for emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 1e-8:
                sim = float(np.dot(emb, centroid) / (emb_norm * centroid_norm))
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0
