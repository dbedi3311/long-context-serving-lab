"""
RAG retrieval helper that wraps the vector index with a simple API.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from rag.index import VectorIndex


class Retriever:
    def __init__(self) -> None:
        self.index = VectorIndex()

    def ingest(self, docs: Sequence[str]) -> None:
        self.index.fit(docs)

    def query(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        return self.index.search(text, k=k)
