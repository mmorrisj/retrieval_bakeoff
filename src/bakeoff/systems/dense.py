"""Dense retrieval over an exhaustive (flat) vector index.

The index is a plain matrix and search is a single matmul. That is a choice
worth defending: an approximate index (HNSW, IVF) would be faster on a large
corpus, but it introduces a recall/latency knob that differs per library and
would confound the very comparison this benchmark exists to make. Flat search
is exact, so quality differences between rows are attributable to the embedding
model rather than to somebody's ef_search setting.

The consequence is that dense latency here scales linearly with corpus size and
is a *ceiling*, not what a tuned production index would give you. The report
says so.
"""

from __future__ import annotations

import numpy as np

from bakeoff.systems.base import Document, Hit, Retriever
from bakeoff.systems.embedders import Embedder


class DenseRetriever(Retriever):
    """Embed the corpus once, then rank by cosine similarity."""

    name = "dense"

    def __init__(self, embedder: Embedder, name: str | None = None) -> None:
        super().__init__(name)
        self.embedder = embedder
        self._doc_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def index(self, corpus: list[Document]) -> None:
        self._doc_ids = [doc.doc_id for doc in corpus]

        tokens_before = self.embedder.tokens
        self._matrix = self.embedder.encode([doc.text for doc in corpus], is_query=False)
        self.usage.index_tokens += self.embedder.tokens - tokens_before

        if self.embedder.price_key:
            self.usage.price_keys = sorted({*self.usage.price_keys, self.embedder.price_key})
        self.usage.tokens_estimated |= self.embedder.tokens_estimated

    def search(self, query: str, k: int) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")

        tokens_before = self.embedder.tokens
        query_vector = self.embedder.encode([query], is_query=True)[0]
        self.usage.query_tokens += self.embedder.tokens - tokens_before
        self.usage.queries += 1
        self.usage.tokens_estimated |= self.embedder.tokens_estimated

        # Both sides are unit-norm, so the dot product is cosine similarity.
        scores = self._matrix @ query_vector

        # argpartition finds the top k without fully sorting the corpus, which
        # matters once the corpus is large enough for the sort to dominate.
        top_k = min(k, len(self._doc_ids))
        if top_k == 0:
            return []
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]

        return [Hit(self._doc_ids[position], float(scores[position])) for position in ordered]

    def describe(self) -> dict[str, str]:
        return {"type": "DenseRetriever", **self.embedder.describe()}
