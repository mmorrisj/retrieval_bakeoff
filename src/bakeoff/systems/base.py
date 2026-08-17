"""The interface every retrieval system implements.

A system is anything that can (a) build an index over a corpus and (b) return a
ranked list of document ids for a query. That is a deliberately small surface:
BM25, a dense embedder, a hybrid fusion of both, and a cross-encoder rerank
stage all fit it, which is what makes them comparable in one table.

Systems report their own billable `Usage`. The alternative -- having the runner
infer cost from the outside -- breaks as soon as a system batches, caches or
composes, so each system is made responsible for declaring what it consumed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bakeoff.cost import Usage


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str

    @property
    def tokens_estimate(self) -> int:
        from bakeoff.cost import estimate_tokens

        return estimate_tokens([self.text])


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever(ABC):
    """Base class for every system in the bake-off."""

    #: Short label used as the row name in reports. Must be unique per run.
    name: str = "unnamed"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name
        self.usage = Usage()

    @abstractmethod
    def index(self, corpus: list[Document]) -> None:
        """Build whatever structure the system needs to search the corpus.

        Implementations should record indexing cost on `self.usage`.
        """

    @abstractmethod
    def search(self, query: str, k: int) -> list[Hit]:
        """Return the top `k` documents for `query`, best first.

        Called once per query at batch size 1 so the runner can measure latency
        honestly. Implementations should record per-query cost on `self.usage`.
        """

    def describe(self) -> dict[str, str]:
        """Configuration worth recording in the results file for reproducibility."""
        return {"type": type(self).__name__}

    def close(self) -> None:  # noqa: B027 - an optional hook, deliberately not abstract
        """Release any resources. Called by the runner even when a system fails.

        Most systems hold nothing that needs releasing, so this is a no-op hook
        rather than an abstract method; forcing every subclass to implement an
        empty `close` would be noise.
        """


class NotInstalled(RuntimeError):
    """Raised when a system's optional dependency or API key is missing.

    The runner catches this and records the system as skipped rather than
    failing the whole run, so `make demo` works with nothing installed and
    `make bench` degrades gracefully when only some API keys are present.
    """
