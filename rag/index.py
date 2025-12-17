"""
Lightweight vector index with FAISS preferred, scikit-learn fallback, and a pure-Python
backup for offline environments.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

try:  # Optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional
    faiss = None
    _HAS_FAISS = False

try:  # pragma: no cover - optional
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    TfidfVectorizer = None  # type: ignore
    NearestNeighbors = None  # type: ignore
    _HAS_SKLEARN = False


class VectorIndex:
    def __init__(self, use_faiss: Optional[bool] = None) -> None:
        if use_faiss is None:
            use_faiss = _HAS_FAISS
        self.use_faiss = use_faiss and _HAS_FAISS and np is not None
        self.vectorizer = TfidfVectorizer() if _HAS_SKLEARN else None
        self._faiss_index = None
        self._nn = None
        self._corpus: List[str] = []
        self._bow: List[Dict[str, int]] = []

    def fit(self, docs: Sequence[str]) -> None:
        self._corpus = list(docs)
        if self.use_faiss and self.vectorizer and np is not None:
            matrix = self.vectorizer.fit_transform(self._corpus)
            embeddings = matrix.astype(np.float32)
            dense = embeddings.toarray()
            dim = dense.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(dense)
            self._faiss_index.add(dense)
            return
        if _HAS_SKLEARN and self.vectorizer:
            matrix = self.vectorizer.fit_transform(self._corpus)
            self._nn = NearestNeighbors(metric="cosine")
            self._nn.fit(matrix)
            return
        # Pure-Python fallback
        self._bow = [Counter(doc.lower().split()) for doc in self._corpus]

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self.use_faiss and self._faiss_index is not None and self.vectorizer and np is not None:
            query_vec = self.vectorizer.transform([query]).astype(np.float32)
            dense = query_vec.toarray()
            faiss.normalize_L2(dense)
            scores, indices = self._faiss_index.search(dense, k)
            return [(self._corpus[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
        if self._nn is not None and self.vectorizer:
            query_vec = self.vectorizer.transform([query])
            distances, indices = self._nn.kneighbors(query_vec, n_neighbors=k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                score = 1.0 - float(dist)
                results.append((self._corpus[idx], score))
            return results
        if not self._bow:
            raise RuntimeError("Index not built")
        query_vec = Counter(query.lower().split())
        scores: List[Tuple[str, float]] = []
        for doc, bow in zip(self._corpus, self._bow):
            score = self._cosine(query_vec, bow)
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        dot = sum(a[token] * b.get(token, 0) for token in a)
        norm_a = sum(v * v for v in a.values()) ** 0.5
        norm_b = sum(v * v for v in b.values()) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
