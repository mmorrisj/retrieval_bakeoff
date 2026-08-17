"""Hybrid retrieval by Reciprocal Rank Fusion.

RRF combines rankings by position rather than by score:

    score(d) = sum over systems s of  weight[s] / (rrf_k + rank_s(d))

The reason to prefer it over a weighted sum of the raw scores is that BM25 and
cosine similarity live on incomparable scales -- BM25 is unbounded and corpus
dependent, cosine is bounded in [-1, 1] -- so any score blend needs a
normalisation step that has to be re-tuned per corpus. RRF needs no tuning and
is what most production hybrid stacks actually ship.

`rrf_k` defaults to 60, the value from the original Cormack et al. paper. It
damps the influence of the very top ranks; smaller values trust rank 1 more.
"""

from __future__ import annotations

from collections import defaultdict

from bakeoff.cost import Usage
from bakeoff.systems.base import Document, Hit, Retriever

DEFAULT_RRF_K = 60


class HybridRetriever(Retriever):
    """Fuses the rankings of two or more component retrievers."""

    name = "hybrid"

    def __init__(
        self,
        components: list[Retriever],
        name: str | None = None,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = 100,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__(name)
        if len(components) < 2:
            raise ValueError("a hybrid needs at least two component retrievers")
        if weights is not None and len(weights) != len(components):
            raise ValueError("weights must have one entry per component")

        self.components = components
        self.rrf_k = rrf_k
        # Each component is asked for more candidates than the final cutoff so
        # fusion has something to work with; a document ranked 40th by BM25 and
        # 40th by dense should be able to reach the final top 10.
        self.candidate_k = candidate_k
        self.weights = weights or [1.0] * len(components)

    def index(self, corpus: list[Document]) -> None:
        for component in self.components:
            component.index(corpus)

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)

        for component, weight in zip(self.components, self.weights, strict=True):
            for rank, hit in enumerate(component.search(query, self.candidate_k), start=1):
                fused[hit.doc_id] += weight / (self.rrf_k + rank)

        # Increment the backing field, not the property: reading `self.usage`
        # returns a freshly merged copy, so a mutation through it would be lost.
        self._usage.queries += 1

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [Hit(doc_id, score) for doc_id, score in ranked]

    @property
    def usage(self) -> Usage:  # type: ignore[override]
        """Cost is the sum of the components', plus this layer's own (nothing).

        Computed on read rather than accumulated, because the components are
        the ones actually spending and they update their own usage as they go.
        """
        total = self._usage
        for component in self.components:
            total = total.merge(component.usage)
        return total

    @usage.setter
    def usage(self, value: Usage) -> None:
        self._usage = value

    def describe(self) -> dict[str, str]:
        return {
            "type": "HybridRetriever",
            "rrf_k": str(self.rrf_k),
            "candidate_k": str(self.candidate_k),
            "components": ", ".join(component.name for component in self.components),
            "weights": ", ".join(str(weight) for weight in self.weights),
        }

    def close(self) -> None:
        for component in self.components:
            component.close()
