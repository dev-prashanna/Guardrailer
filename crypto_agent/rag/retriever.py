import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .embeddings import encode_query, encode_texts
from .knowledge_base import KNOWLEDGE_BASE


@dataclass
class RetrievedExample:
    text: str
    category: str
    technique: str
    severity: str
    description: str
    similarity: float


_cache_path = Path(__file__).resolve().parent / ".cache"
_embeddings_file = _cache_path / "kb_embeddings.npy"


class Retriever:
    def __init__(self):
        self.examples = KNOWLEDGE_BASE
        self.texts = [e["text"] for e in self.examples]
        self.embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if _embeddings_file.exists():
            try:
                self.embeddings = np.load(_embeddings_file)
                if self.embeddings.shape[0] == len(self.examples):
                    return
            except Exception:
                pass

        self.embeddings = encode_texts(self.texts)
        if self.embeddings is not None:
            _cache_path.mkdir(parents=True, exist_ok=True)
            np.save(_embeddings_file, self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedExample]:
        self._ensure_loaded()

        if self.embeddings is None:
            return self._fallback_keyword(query, top_k)

        query_embedding = encode_query(query)
        if query_embedding is None:
            return self._fallback_keyword(query, top_k)

        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.15:
                continue
            ex = self.examples[idx]
            results.append(RetrievedExample(
                text=ex["text"],
                category=ex["category"],
                technique=ex["technique"],
                severity=ex["severity"],
                description=ex["description"],
                similarity=round(sim, 4),
            ))

        return results

    def _fallback_keyword(self, query: str, top_k: int) -> list[RetrievedExample]:
        query_lower = query.lower()
        scored = []
        for ex in self.examples:
            text_lower = ex["text"].lower()
            words = set(query_lower.split())
            text_words = set(text_lower.split())
            overlap = len(words & text_words)
            total = max(len(words), 1)
            score = overlap / total
            if score > 0.1:
                scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, ex in scored[:top_k]:
            results.append(RetrievedExample(
                text=ex["text"],
                category=ex["category"],
                technique=ex["technique"],
                severity=ex["severity"],
                description=ex["description"],
                similarity=round(score, 4),
            ))
        return results

    def get_stats(self) -> dict:
        self._ensure_loaded()
        categories = {}
        for ex in self.examples:
            cat = ex["category"]
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_examples": len(self.examples),
            "categories": categories,
            "embeddings_loaded": self.embeddings is not None,
        }


_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
