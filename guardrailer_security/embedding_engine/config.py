"""
config.py
Model configurations and registry for the embedding engine.

Defines available embedding models, their dimensions, and ensemble strategies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EmbeddingMode(str, Enum):
    """Embedding operation modes."""
    SINGLE = "single"               # Use one model
    ENSEMBLE_AVERAGE = "average"    # Weighted average of all models
    ENSEMBLE_LATE_FUSION = "late"   # Late fusion with learned weights
    ENSEMBLE_MAX_SIM = "max_sim"    # Max similarity across models


@dataclass
class ModelConfig:
    """Configuration for a single embedding model."""
    name: str
    model_id: str
    dimension: int
    max_sequence_length: int
    weight: float = 1.0             # Weight in ensemble averaging
    is_multilingual: bool = False
    supports_sparse: bool = False   # Whether model natively produces sparse vectors
    normalize: bool = True          # Whether to L2-normalize embeddings
    query_prefix: str = ""          # Prefix added to queries (some models require this)
    passage_prefix: str = ""        # Prefix added to passages
    description: str = ""

    @property
    def short_name(self) -> str:
        return self.model_id.split("/")[-1]


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

BGE_LARGE_EN_V1_5 = ModelConfig(
    name="bge-large-en-v1.5",
    model_id="BAAI/bge-large-en-v1.5",
    dimension=1024,
    max_sequence_length=512,
    weight=0.40,
    is_multilingual=False,
    normalize=True,
    query_prefix="Represent this sentence for searching relevant passages: ",
    description="BAAI BGE large English model. Strong general-purpose retrieval.",
)

BGE_M3 = ModelConfig(
    name="bge-m3",
    model_id="BAAI/bge-m3",
    dimension=1024,
    max_sequence_length=8192,
    weight=0.35,
    is_multilingual=True,
    supports_sparse=True,
    normalize=True,
    description="BAAI BGE-M3. Multilingual, multi-granularity, multi-functional. "
                "Supports dense + sparse + colbert retrieval.",
)

MXBAI_EMBED_LARGE_V1 = ModelConfig(
    name="mxbai-embed-large-v1",
    model_id="mixedbread-ai/mxbai-embed-large-v1",
    dimension=1024,
    max_sequence_length=512,
    weight=0.25,
    is_multilingual=False,
    normalize=True,
    query_prefix="Represent this sentence for searching relevant passages: ",
    description="mixedbread-ai MxBAGE large model. Competitive with OpenAI ada-002.",
)

# Legacy model (kept for backward compatibility)
BGE_BASE_EN_V1_5 = ModelConfig(
    name="bge-base-en-v1.5",
    model_id="BAAI/bge-base-en-v1.5",
    dimension=768,
    max_sequence_length=512,
    weight=1.0,
    is_multilingual=False,
    normalize=True,
    query_prefix="Represent this sentence for searching relevant passages: ",
    description="BAAI BGE base English model. Smaller, faster.",
)

# Registry
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "bge-large-en-v1.5": BGE_LARGE_EN_V1_5,
    "bge-m3": BGE_M3,
    "mxbai-embed-large-v1": MXBAI_EMBED_LARGE_V1,
    "bge-base-en-v1.5": BGE_BASE_EN_V1_5,
}

# Default ensemble configuration
DEFAULT_ENSEMBLE_MODELS = ["bge-large-en-v1.5", "bge-m3", "mxbai-embed-large-v1"]

# Environment variable overrides
EMBEDDING_MODE = os.environ.get("GUARDRAILER_EMBEDDING_MODE", "single")
EMBEDDING_PRIMARY_MODEL = os.environ.get("GUARDRAILER_EMBEDDING_MODEL", "bge-large-en-v1.5")
EMBEDDING_ENSEMBLE_MODELS = os.environ.get(
    "GUARDRAILER_EMBEDDING_ENSEMBLE",
    ",".join(DEFAULT_ENSEMBLE_MODELS),
)


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")


def get_ensemble_configs() -> list[ModelConfig]:
    """Get configurations for all ensemble models."""
    model_names = [n.strip() for n in EMBEDDING_ENSEMBLE_MODELS.split(",")]
    return [get_model_config(n) for n in model_names if n in MODEL_REGISTRY]


def get_embedding_mode() -> EmbeddingMode:
    """Get the configured embedding mode."""
    try:
        return EmbeddingMode(EMBEDDING_MODE)
    except ValueError:
        return EmbeddingMode.SINGLE
