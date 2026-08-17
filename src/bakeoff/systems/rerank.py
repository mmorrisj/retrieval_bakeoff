"""Cross-encoder reranking on top of a first-stage retriever.

A reranker scores each (query, document) pair jointly instead of comparing two
independently-computed vectors, which is far more accurate and far more
expensive -- the cost is O(candidates) model calls per query rather than one.
That trade is the single most consequential decision in a RAG stack, and it is
the main thing this benchmark is built to quantify: how many nDCG points does
reranking buy, and what does it add to p95 latency and to the bill?

The first stage's `candidate_k` is the knob. Too small and the reranker cannot
recover documents the retriever missed; too large and latency and cost climb
with nothing to show. The config sweeps it.
"""

from __future__ import annotations

import os

from bakeoff.cost import Usage
from bakeoff.systems.base import Document, Hit, NotInstalled, Retriever


class CrossEncoderReranker(Retriever):
    """Local cross-encoder rerank. No invoice, but real compute per candidate."""

    name = "rerank"

    def __init__(
        self,
        base: Retriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        name: str | None = None,
        *,
        candidate_k: int = 100,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        super().__init__(name)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise NotInstalled(
                "sentence-transformers is not installed; run `pip install -e '.[dense]'`"
            ) from exc

        self.base = base
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)
        self._texts: dict[str, str] = {}

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)
        # The reranker needs the document text at query time, which the base
        # retriever is not obliged to keep.
        self._texts = {doc.doc_id: doc.text for doc in corpus}

    def search(self, query: str, k: int) -> list[Hit]:
        candidates = self.base.search(query, self.candidate_k)
        if not candidates:
            self._usage.queries += 1
            return []

        pairs = [(query, self._texts.get(hit.doc_id, "")) for hit in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        self._usage.queries += 1
        self._usage.reranked_docs += len(pairs)

        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].doc_id),
        )[:k]
        return [Hit(hit.doc_id, float(score)) for hit, score in reranked]

    @property
    def usage(self) -> Usage:  # type: ignore[override]
        return self._usage.merge(self.base.usage)

    @usage.setter
    def usage(self, value: Usage) -> None:
        self._usage = value

    def describe(self) -> dict[str, str]:
        return {
            "type": "CrossEncoderReranker",
            "model": self.model_name,
            "candidate_k": str(self.candidate_k),
            "base": self.base.name,
        }

    def close(self) -> None:
        self.base.close()


class CohereReranker(Retriever):
    """Hosted rerank, billed per search unit rather than per token."""

    name = "cohere-rerank"

    def __init__(
        self,
        base: Retriever,
        model_name: str = "rerank-english-v3.0",
        name: str | None = None,
        *,
        candidate_k: int = 100,
    ) -> None:
        super().__init__(name)
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise NotInstalled("cohere is not installed; run `pip install -e '.[api]'`") from exc
        if not os.environ.get("COHERE_API_KEY"):
            raise NotInstalled("COHERE_API_KEY is not set")

        self.base = base
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self._texts: dict[str, str] = {}
        self._usage.price_keys = [f"cohere/{model_name}"]

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)
        self._texts = {doc.doc_id: doc.text for doc in corpus}

    def search(self, query: str, k: int) -> list[Hit]:
        candidates = self.base.search(query, self.candidate_k)
        if not candidates:
            self._usage.queries += 1
            return []

        documents = [self._texts.get(hit.doc_id, "") for hit in candidates]
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model_name,
            top_n=k,
        )

        self._usage.queries += 1
        self._usage.reranked_docs += len(documents)

        # Results carry an index back into the documents list we sent, not a
        # document id, so the mapping has to go through `candidates`.
        return [
            Hit(candidates[result.index].doc_id, float(result.relevance_score))
            for result in response.results
        ]

    @property
    def usage(self) -> Usage:  # type: ignore[override]
        return self._usage.merge(self.base.usage)

    @usage.setter
    def usage(self, value: Usage) -> None:
        self._usage = value

    def describe(self) -> dict[str, str]:
        return {
            "type": "CohereReranker",
            "model": self.model_name,
            "candidate_k": str(self.candidate_k),
            "base": self.base.name,
        }

    def close(self) -> None:
        self.base.close()
