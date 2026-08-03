"""
embedding_engine
Advanced embedding models module for Guardrailer Phase 2.

Provides multi-model embedding support, ensemble strategies,
contrastive fine-tuning, and hard negative mining.
"""

from .engine import EmbeddingEngine
from .config import ModelConfig, EmbeddingMode
from .ensemble import EnsembleEmbedding
from .fine_tune import ContrastiveTrainer
from .hard_negatives import HardNegativeMiner

__all__ = [
    "EmbeddingEngine",
    "ModelConfig",
    "EmbeddingMode",
    "EnsembleEmbedding",
    "ContrastiveTrainer",
    "HardNegativeMiner",
]
