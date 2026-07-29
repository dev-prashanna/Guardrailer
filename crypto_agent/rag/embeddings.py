import os
import numpy as np
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

_MODEL_NAME = "all-MiniLM-L6-v2"
_instance: Optional["SentenceTransformer"] = None


def get_model() -> Optional["SentenceTransformer"]:
    global _instance
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    if _instance is None:
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        _instance = SentenceTransformer(_MODEL_NAME, cache_folder=cache_dir)
    return _instance


def encode_texts(texts: list[str]) -> Optional[np.ndarray]:
    model = get_model()
    if model is None:
        return None
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def encode_query(text: str) -> Optional[np.ndarray]:
    model = get_model()
    if model is None:
        return None
    return model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
