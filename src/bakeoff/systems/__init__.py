"""System construction from config.

`build_system` turns a config block into a `Retriever`. Composite systems
(hybrid, rerank) nest a `base`/`components` block, so a config can describe
"BM25, reranked by a cross-encoder over its top 100" without any code changes.
"""

from __future__ import annotations

from typing import Any

from bakeoff.systems.base import Document, Hit, NotInstalled, Retriever
from bakeoff.systems.bm25 import BM25Retriever
from bakeoff.systems.dense import DenseRetriever
from bakeoff.systems.embedders import build_embedder
from bakeoff.systems.hybrid import HybridRetriever
from bakeoff.systems.rerank import CohereReranker, CrossEncoderReranker

__all__ = [
    "BM25Retriever",
    "CohereReranker",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Document",
    "Hit",
    "HybridRetriever",
    "NotInstalled",
    "Retriever",
    "build_system",
]


def build_system(spec: dict[str, Any]) -> Retriever:
    """Construct a retriever from a config block.

    The block's `kind` selects the class; every other key is passed through as a
    constructor argument, except `base` and `components`, which are built
    recursively first.
    """
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if not kind:
        raise ValueError(f"system config is missing a 'kind': {spec}")
    name = spec.pop("name", None)

    if kind == "bm25":
        return BM25Retriever(name, **spec)

    if kind == "dense":
        embedder_spec = dict(spec.pop("embedder", {}))
        embedder_kind = embedder_spec.pop("kind", None)
        if not embedder_kind:
            raise ValueError(f"dense system {name!r} is missing an 'embedder.kind'")
        return DenseRetriever(build_embedder(embedder_kind, **embedder_spec), name, **spec)

    if kind == "hybrid":
        components = [build_system(component) for component in spec.pop("components", [])]
        return HybridRetriever(components, name, **spec)

    if kind in {"rerank", "cohere-rerank"}:
        base_spec = spec.pop("base", None)
        if not base_spec:
            raise ValueError(f"{kind} system {name!r} is missing a 'base' block")
        base = build_system(base_spec)
        cls = CrossEncoderReranker if kind == "rerank" else CohereReranker
        return cls(base, name=name, **spec)

    known = "bm25, dense, hybrid, rerank, cohere-rerank"
    raise ValueError(f"unknown system kind {kind!r}; known kinds are: {known}")
