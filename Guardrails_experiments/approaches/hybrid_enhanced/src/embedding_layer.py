"""
Lightweight embedding layer using BAAI/bge-small-en-v1.5.

Provides semantic similarity features without the overhead of large models.
Model size: ~33M params, 384 dimensions, ~130MB on disk.
"""

import gc
import os
import sys
import numpy as np
from typing import Optional, List, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class LightweightEmbeddingLayer:
    """
    Embedding layer using BAAI/bge-small-en-v1.5 for semantic features.
    
    Features:
    - 384-dimensional embeddings (vs 1024 for bge-large)
    - ~33M parameters (vs 330M for bge-large)
    - Supports Matryoshka representation (can use 64/128/256/384 dims)
    - Caches training embeddings to avoid recomputation
    """
    
    MODEL_ID = "all-MiniLM-L6-v2"
    DIMENSION = 384
    
    def __init__(self, device: Optional[str] = None, use_matryoshka: bool = False, matryoshka_dim: int = 256):
        self.device = device or ("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.use_matryoshka = use_matryoshka
        self.matryoshka_dim = matryoshka_dim if use_matryoshka else self.DIMENSION
        self._model: Optional["SentenceTransformer"] = None
        self._fitted = False
        self._malicious_centroid: Optional[np.ndarray] = None
        self._benign_centroid: Optional[np.ndarray] = None
        self._cached_embeddings: Optional[np.ndarray] = None
        self._cached_texts_hash: Optional[int] = None
        
    def _log(self, msg: str):
        import sys
        print(msg, flush=True, file=sys.stderr)
        
    def _clear_gpu_cache(self):
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def _load_model(self) -> "SentenceTransformer":
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
            
            cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            self._model = SentenceTransformer(
                self.MODEL_ID,
                cache_folder=cache_dir,
                device=self.device
            )
        return self._model
    
    def _apply_matryoshka(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.use_matryoshka or self.matryoshka_dim >= self.DIMENSION:
            return embeddings
        
        import torch.nn.functional as F
        
        if isinstance(embeddings, np.ndarray):
            embeddings_t = torch.from_numpy(embeddings)
        else:
            embeddings_t = embeddings
        
        embeddings_t = F.layer_norm(embeddings_t, normalized_shape=(embeddings_t.shape[1],))
        embeddings_t = embeddings_t[:, :self.matryoshka_dim]
        embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
        
        return embeddings_t.numpy()
    
    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        model = self._load_model()
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        return self._apply_matryoshka(embeddings)
    
    def _unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._clear_gpu_cache()
    
    def fit(self, texts: List[str], labels: List[int]) -> "LightweightEmbeddingLayer":
        labels = np.array(labels)
        
        self._log(f"  Encoding {len(texts):,} texts with {self.MODEL_ID}...")
        start = __import__('time').time()
        embeddings = self.encode(texts, batch_size=256)
        elapsed = __import__('time').time() - start
        self._log(f"  Encoding done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/s)")
        
        malicious_mask = labels == 1
        benign_mask = labels == 0
        
        if malicious_mask.sum() > 0:
            self._malicious_centroid = embeddings[malicious_mask].mean(axis=0)
            self._malicious_centroid /= np.linalg.norm(self._malicious_centroid) + 1e-8
        
        if benign_mask.sum() > 0:
            self._benign_centroid = embeddings[benign_mask].mean(axis=0)
            self._benign_centroid /= np.linalg.norm(self._benign_centroid) + 1e-8
        
        self._cached_embeddings = embeddings
        self._cached_texts_hash = hash(texts[0]) if texts else None
        self._fitted = True
        
        self._unload_model()
        
        return self
    
    def compute_similarity_features(self, text: str) -> dict:
        embedding = self.encode([text], batch_size=1)[0]
        return self._features_from_embedding(embedding)
    
    def _features_from_embedding(self, embedding: np.ndarray) -> dict:
        features = {
            "embedding_dim": float(self.matryoshka_dim),
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
        
        if self._malicious_centroid is not None:
            malicious_sim = float(np.dot(embedding, self._malicious_centroid))
            features["centroid_similarity_malicious"] = malicious_sim
            features["centroid_distance_malicious"] = 1.0 - malicious_sim
        else:
            features["centroid_similarity_malicious"] = 0.0
            features["centroid_distance_malicious"] = 1.0
            
        if self._benign_centroid is not None:
            benign_sim = float(np.dot(embedding, self._benign_centroid))
            features["centroid_similarity_benign"] = benign_sim
            features["centroid_distance_benign"] = 1.0 - benign_sim
        else:
            features["centroid_similarity_benign"] = 0.0
            features["centroid_distance_benign"] = 1.0
            
        features["centroid_diff"] = features["centroid_distance_benign"] - features["centroid_distance_malicious"]
        return features
    
    def compute_similarity_features_batch(self, texts: List[str], batch_size: int = 256) -> Tuple[np.ndarray, List[str]]:
        use_cached = (
            self._cached_embeddings is not None 
            and len(self._cached_embeddings) == len(texts)
        )
        
        if use_cached:
            self._log(f"  Using cached embeddings ({len(texts):,} texts)")
            embeddings = self._cached_embeddings
        else:
            self._log(f"  Encoding {len(texts):,} texts for features...")
            all_embeddings = []
            effective_batch = min(batch_size, 256)
            for i in range(0, len(texts), effective_batch):
                batch = texts[i:i + effective_batch]
                emb = self.encode(batch, batch_size=min(len(batch), 256))
                all_embeddings.append(emb)
                if i % (effective_batch * 10) == 0 and i > 0:
                    self._log(f"    Embeddings: {i:,}/{len(texts):,}")
            embeddings = np.vstack(all_embeddings)
        
        malicious_sims = embeddings @ self._malicious_centroid if self._malicious_centroid is not None else np.zeros(len(texts))
        benign_sims = embeddings @ self._benign_centroid if self._benign_centroid is not None else np.zeros(len(texts))
        
        norms = np.linalg.norm(embeddings, axis=1)
        
        feature_matrix = np.column_stack([
            np.full(len(texts), float(self.matryoshka_dim)),
            norms,
            malicious_sims,
            1.0 - malicious_sims,
            benign_sims,
            1.0 - benign_sims,
            (1.0 - benign_sims) - (1.0 - malicious_sims),
        ]).astype(np.float32)
        
        feature_names = self.get_feature_names()
        return feature_matrix, feature_names
    
    def get_feature_names(self) -> List[str]:
        return [
            "centroid_diff",
            "centroid_distance_benign",
            "centroid_distance_malicious",
            "centroid_similarity_benign",
            "centroid_similarity_malicious",
            "embedding_dim",
            "embedding_norm",
        ]
    
    @property
    def dimension(self) -> int:
        return self.matryoshka_dim
    
    @property
    def model_size_mb(self) -> float:
        return 130.0
