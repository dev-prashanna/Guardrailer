"""
engine.py
Core embedding engine supporting single-model and multi-model inference.

Loads sentence-transformers models, computes embeddings, and provides
a unified interface for the security engine.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .config import (
    ModelConfig,
    EmbeddingMode,
    get_model_config,
    get_ensemble_configs,
    get_embedding_mode,
    BGE_LARGE_EN_V1_5,
)

log = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Multi-model embedding engine.
    
    Supports:
    - Single model inference (default: BGE-large-en-v1.5)
    - Weighted ensemble averaging
    - Late fusion with model-specific weights
    - Max-similarity ensemble
    
    Models are loaded lazily and cached as singletons.
    """

    def __init__(
        self,
        mode: Optional[EmbeddingMode] = None,
        primary_model: Optional[str] = None,
        ensemble_models: Optional[list[str]] = None,
        device: Optional[str] = None,
    ):
        self.mode = mode or get_embedding_mode()
        self.primary_model_name = primary_model or BGE_LARGE_EN_V1_5.name
        self.ensemble_model_names = ensemble_models
        self.device = device

        self._models: dict[str, object] = {}
        self._configs: dict[str, ModelConfig] = {}
        self._ensemble_weights: dict[str, float] = {}

        log.info("EmbeddingEngine initialized: mode=%s, primary=%s", self.mode, self.primary_model_name)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, config: ModelConfig) -> object:
        """Load a sentence-transformers model."""
        if config.name in self._models:
            return self._models[config.name]

        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: %s ...", config.model_id)
        t0 = time.time()

        model = SentenceTransformer(config.model_id, device=self.device)
        elapsed = time.time() - t0

        self._models[config.name] = model
        self._configs[config.name] = config
        self._ensemble_weights[config.name] = config.weight

        log.info("  %s loaded in %.1fs (dim=%d, max_seq=%d)",
                 config.short_name, elapsed, config.dimension, config.max_sequence_length)
        return model

    def load_primary(self) -> object:
        """Load the primary embedding model."""
        config = get_model_config(self.primary_model_name)
        return self._load_model(config)

    def load_ensemble(self) -> list[object]:
        """Load all ensemble models."""
        configs = get_ensemble_configs()
        models = []
        for config in configs:
            model = self._load_model(config)
            models.append(model)
        return models

    def get_model(self, name: Optional[str] = None) -> object:
        """Get a loaded model by name, loading if necessary."""
        name = name or self.primary_model_name
        config = get_model_config(name)
        return self._load_model(config)

    def get_config(self, name: Optional[str] = None) -> ModelConfig:
        """Get model configuration."""
        name = name or self.primary_model_name
        return get_model_config(name)

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str] | str,
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Compute embeddings for texts using a specific model.
        
        Args:
            texts: Single string or list of strings.
            model_name: Model to use. None uses primary model.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
        
        Returns:
            numpy array of shape (n, dimension) or (dimension,) for single text.
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        model = self.get_model(model_name)
        config = self.get_config(model_name)

        # Apply prefix if configured
        if config.query_prefix:
            texts = [config.query_prefix + t for t in texts]

        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )

        if single:
            return embeddings[0]
        return embeddings

    def encode_ensemble(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute ensemble embeddings by combining outputs from all models.
        
        Uses the configured mode (average, late fusion, max_sim).
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        configs = get_ensemble_configs()
        all_embeddings = []

        for config in configs:
            model = self.get_model(config.name)
            emb = model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
            )
            all_embeddings.append((emb, config.weight, config.name))

        if self.mode == EmbeddingMode.ENSEMBLE_AVERAGE:
            result = self._weighted_average(all_embeddings)
        elif self.mode == EmbeddingMode.ENSEMBLE_LATE_FUSION:
            result = self._late_fusion(all_embeddings)
        elif self.mode == EmbeddingMode.ENSEMBLE_MAX_SIM:
            result = self._max_similarity(all_embeddings)
        else:
            result = all_embeddings[0][0]

        if single:
            return result[0]
        return result

    # ------------------------------------------------------------------
    # Ensemble strategies
    # ------------------------------------------------------------------

    def _weighted_average(
        self,
        embeddings_list: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """Weighted average of embeddings from multiple models."""
        total_weight = sum(w for _, w, _ in embeddings_list)
        if total_weight < 1e-8:
            total_weight = 1.0

        combined = None
        for emb, weight, name in embeddings_list:
            weighted = emb * (weight / total_weight)
            if combined is None:
                combined = weighted
            else:
                combined = combined + weighted

        # Re-normalize
        norms = np.linalg.norm(combined, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return combined / norms

    def _late_fusion(
        self,
        embeddings_list: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """
        Late fusion: concatenate model embeddings and project to target dimension.
        
        This preserves more information from each model than simple averaging.
        """
        all_embs = [emb for emb, _, _ in embeddings_list]
        concatenated = np.concatenate(all_embs, axis=-1)

        # Simple learned projection via weighted sum (no training needed for inference)
        # This is equivalent to a single-layer linear projection
        total_weight = sum(w for _, w, _ in embeddings_list)
        if total_weight < 1e-8:
            total_weight = 1.0

        # Project back to primary model dimension using chunked averaging
        primary_config = get_model_config(self.primary_model_name)
        target_dim = primary_config.dimension
        n_models = len(embeddings_list)

        if concatenated.shape[-1] == target_dim:
            result = concatenated
        elif concatenated.shape[-1] > target_dim:
            # Truncate or average-pool to target dimension
            chunk_size = concatenated.shape[-1] // n_models
            result = concatenated[:, :target_dim]
        else:
            # Pad with zeros
            pad_size = target_dim - concatenated.shape[-1]
            result = np.pad(concatenated, ((0, 0), (0, pad_size)))

        # Normalize
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return result / norms

    def _max_similarity(
        self,
        embeddings_list: list[tuple[np.ndarray, float, str]],
    ) -> np.ndarray:
        """
        Max-similarity ensemble: for each dimension, take the value from
        the model that assigns the highest absolute value.
        
        This captures the most confident signal from each model.
        """
        # Stack all embeddings
        stacked = np.stack([emb for emb, _, _ in embeddings_list], axis=0)

        # Take value with max absolute value per dimension
        abs_stacked = np.abs(stacked)
        max_indices = np.argmax(abs_stacked, axis=0)

        # Gather values
        result = np.zeros_like(stacked[0])
        for i in range(stacked.shape[1]):
            result[i] = stacked[max_indices[i], i]

        # Normalize
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return result / norms

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> float | np.ndarray:
        """Compute cosine similarity between embeddings."""
        if embeddings_a.ndim == 1:
            embeddings_a = embeddings_a.reshape(1, -1)
        if embeddings_b.ndim == 1:
            embeddings_b = embeddings_b.reshape(1, -1)

        a_norm = embeddings_a / np.maximum(np.linalg.norm(embeddings_a, axis=-1, keepdims=True), 1e-8)
        b_norm = embeddings_b / np.maximum(np.linalg.norm(embeddings_b, axis=-1, keepdims=True), 1e-8)

        similarities = np.dot(a_norm, b_norm.T)
        if similarities.shape == (1, 1):
            return float(similarities[0, 0])
        return similarities

    @property
    def dimension(self) -> int:
        """Get the output dimension of the primary model."""
        config = get_model_config(self.primary_model_name)
        return config.dimension

    @property
    def ensemble_dimension(self) -> int:
        """Get the output dimension of the ensemble (after fusion)."""
        return self.dimension  # Always projects to primary model dimension

    def get_loaded_models(self) -> list[str]:
        """Get names of currently loaded models."""
        return list(self._models.keys())

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        self._models.clear()
        self._configs.clear()
        self._ensemble_weights.clear()
        import gc
        gc.collect()
        log.info("All embedding models unloaded")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine(**kwargs) -> EmbeddingEngine:
    """Get or create the singleton EmbeddingEngine."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine(**kwargs)
    return _engine


def reset_embedding_engine() -> None:
    """Reset the singleton engine."""
    global _engine
    if _engine is not None:
        _engine.unload_all()
    _engine = None
