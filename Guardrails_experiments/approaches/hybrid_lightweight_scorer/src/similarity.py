"""
TF-IDF similarity layer — memory-optimized with character-level TF-IDF.

Uses sklearn TfidfVectorizer with sparse matrices.
Stores only malicious corpus centroid, not all doc vectors.
Supports both word-level and character-level TF-IDF for obfuscation robustness.
"""

import gc
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter


class TFIDFSimilarity:
    """
    Sparse TF-IDF similarity scorer with dual word+char support.

    Stores only the centroid of malicious documents (not all vectors).
    Uses sklearn TfidfVectorizer for memory-efficient sparse representation.
    """

    def __init__(self, max_features=50000, char_max_features=30000):
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.vectorizer = None
        self.char_vectorizer = None
        self.malicious_centroid = None
        self.char_malicious_centroid = None
        self.idf_array = None
        self.fitted = False

    def fit(self, texts: List[str], labels: Optional[List[int]] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        print(f"  Fitting word-level TF-IDF on {len(texts):,} docs (max_features={self.max_features})...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            sublinear_tf=True,
            norm="l2",
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        X = self.vectorizer.fit_transform(texts)
        print(f"  Word TF-IDF shape: {X.shape}, nnz: {X.nnz:,}")

        self.idf_array = np.array(self.vectorizer.idf_, dtype=np.float32)

        if labels is not None:
            labels = np.array(labels)
            malicious_mask = labels == 1
            if malicious_mask.sum() > 0:
                X_malicious = X[malicious_mask]
                self.malicious_centroid = np.asarray(X_malicious.mean(axis=0)).flatten().astype(np.float32)
                centroid_norm = np.linalg.norm(self.malicious_centroid)
                if centroid_norm > 0:
                    self.malicious_centroid /= centroid_norm
                print(f"  Word malicious centroid computed from {malicious_mask.sum():,} docs")

                X_benign = X[~malicious_mask]
                self.benign_centroid = np.asarray(X_benign.mean(axis=0)).flatten().astype(np.float32)
                benign_norm = np.linalg.norm(self.benign_centroid)
                if benign_norm > 0:
                    self.benign_centroid /= benign_norm

        del X
        gc.collect()

        print(f"  Fitting char-level TF-IDF (max_features={self.char_max_features})...")
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.char_max_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        X_char = self.char_vectorizer.fit_transform(texts)
        print(f"  Char TF-IDF shape: {X_char.shape}, nnz: {X_char.nnz:,}")

        if labels is not None and malicious_mask.sum() > 0:
            X_char_malicious = X_char[malicious_mask]
            self.char_malicious_centroid = np.asarray(X_char_malicious.mean(axis=0)).flatten().astype(np.float32)
            char_centroid_norm = np.linalg.norm(self.char_malicious_centroid)
            if char_centroid_norm > 0:
                self.char_malicious_centroid /= char_centroid_norm
            print(f"  Char malicious centroid computed")

        del X_char
        gc.collect()

        self.fitted = True
        return self

    def _word_similarity(self, text: str) -> float:
        if not self.fitted or self.malicious_centroid is None or self.vectorizer is None:
            return 0.0

        X = self.vectorizer.transform([text])
        if X.nnz == 0:
            return 0.0

        query_vec = np.asarray(X.todense()).flatten().astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return 0.0

        sim = float(np.dot(query_vec, self.malicious_centroid) / query_norm)
        normalized = 1 / (1 + np.exp(-5 * (sim - 0.3)))
        return normalized

    def _char_similarity(self, text: str) -> float:
        if not self.fitted or self.char_malicious_centroid is None or self.char_vectorizer is None:
            return 0.0

        X = self.char_vectorizer.transform([text])
        if X.nnz == 0:
            return 0.0

        query_vec = np.asarray(X.todense()).flatten().astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return 0.0

        sim = float(np.dot(query_vec, self.char_malicious_centroid) / query_norm)
        normalized = 1 / (1 + np.exp(-5 * (sim - 0.3)))
        return normalized

    def score(self, text: str) -> float:
        word_sim = self._word_similarity(text)
        char_sim = self._char_similarity(text)
        return max(word_sim, char_sim)

    def classify(self, text: str, threshold: float = 0.5) -> Tuple[bool, float]:
        score = self.score(text)
        return score >= threshold, score

    def save(self, path):
        import joblib
        joblib.dump({
            "vectorizer": self.vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "malicious_centroid": self.malicious_centroid,
            "char_malicious_centroid": self.char_malicious_centroid,
            "benign_centroid": getattr(self, "benign_centroid", None),
            "idf_array": self.idf_array,
            "max_features": self.max_features,
            "char_max_features": self.char_max_features,
        }, path)

    def load(self, path):
        import joblib
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.char_vectorizer = data.get("char_vectorizer")
        self.malicious_centroid = data["malicious_centroid"]
        self.char_malicious_centroid = data.get("char_malicious_centroid")
        self.benign_centroid = data.get("benign_centroid")
        self.idf_array = data["idf_array"]
        self.max_features = data["max_features"]
        self.char_max_features = data.get("char_max_features", 30000)
        self.fitted = True


class IDFWeightedKeywords:
    """
    Simple IDF-weighted keyword scorer.
    Fast (~0.01ms), no training needed.
    """

    def __init__(self):
        self.keywords = {}
        self.fitted = False

    def fit(self, texts: List[str], labels: List[int]):
        n_docs = len(texts)
        malicious_doc_freq = Counter()
        safe_doc_freq = Counter()

        for text, label in zip(texts, labels):
            words = set(text.lower().split())
            if label == 1:
                for w in words:
                    malicious_doc_freq[w] += 1
            else:
                for w in words:
                    safe_doc_freq[w] += 1

        for word, df in malicious_doc_freq.items():
            if df >= 3:
                idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                safe_count = safe_doc_freq.get(word, 0)
                malicious_ratio = df / (df + safe_count + 1)
                self.keywords[word] = idf * malicious_ratio

        self.fitted = True
        return self

    def score(self, text: str) -> float:
        if not self.fitted:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        score = sum(self.keywords.get(w, 0) for w in words)
        normalized = score / len(words)
        return float(1 / (1 + np.exp(-10 * (normalized - 0.1))))

    def classify(self, text: str, threshold: float = 0.5) -> Tuple[bool, float]:
        score = self.score(text)
        return score >= threshold, score

    def save(self, path):
        import joblib
        joblib.dump(self.keywords, path)

    def load(self, path):
        import joblib
        self.keywords = joblib.load(path)
        self.fitted = True
