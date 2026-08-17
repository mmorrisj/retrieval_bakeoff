"""Embedding cache behaviour.

The two properties that matter most here are negative ones: the cache must not
touch the query path (that would make the latency column measure disk speed),
and a cache hit must not zero out the reported cost (a $0 index column because
the cache was warm is worse than no cost column at all).
"""

import numpy as np
import pytest

from bakeoff.config import RunConfig
from bakeoff.runner import run
from bakeoff.systems import build_system
from bakeoff.systems.embedders import CachedEmbedder, Embedder, build_embedder


class _CountingEmbedder(Embedder):
    """Records how many times it was actually asked to do work."""

    price_key = "openai/text-embedding-3-small"

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.document_calls = 0
        self.query_calls = 0
        self.encoded_texts = 0

    def encode(self, texts, *, is_query):
        if is_query:
            self.query_calls += 1
        else:
            self.document_calls += 1
        self.encoded_texts += len(texts)

        # Deterministic per-text vectors, so a cache hit can be compared
        # against a live call for exact equality.
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, len(text) % self.dim] = 1.0

        self.tokens += 10 * len(texts)
        return vectors

    def describe(self):
        return {"embedder": "_CountingEmbedder", "dim": str(self.dim)}


DOCS = ["alpha", "beta gamma", "delta epsilon zeta"]


def test_second_encode_of_the_same_documents_hits_the_cache(tmp_path):
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    first = cached.encode(DOCS, is_query=False)
    second = cached.encode(DOCS, is_query=False)

    assert inner.document_calls == 1, "the second call should not reach the model"
    assert cached.hits == 1 and cached.misses == 1
    assert np.array_equal(first, second)


def test_a_cache_hit_still_reports_the_tokens_the_index_cost(tmp_path):
    # A warm cache must not make the index look free. Two separate embedders
    # sharing a cache dir should report identical token counts.
    cold = CachedEmbedder(_CountingEmbedder(), tmp_path)
    cold.encode(DOCS, is_query=False)

    warm = CachedEmbedder(_CountingEmbedder(), tmp_path)
    warm.encode(DOCS, is_query=False)

    assert warm.hits == 1
    assert warm.tokens == cold.tokens == 30


def test_queries_are_never_cached(tmp_path):
    # Caching the query path would turn the latency measurement into a disk
    # benchmark, so every query must reach the model.
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    for _ in range(3):
        cached.encode(["the same query"], is_query=True)

    assert inner.query_calls == 3
    assert cached.hits == 0
    assert not list(tmp_path.glob("*.npz"))


def test_different_corpora_do_not_collide(tmp_path):
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)

    cached.encode(DOCS, is_query=False)
    cached.encode(["completely different text"], is_query=False)

    assert inner.document_calls == 2
    assert cached.hits == 0


def test_different_models_do_not_collide(tmp_path):
    # Same texts, different model descriptor -- must not share an entry.
    first = CachedEmbedder(_CountingEmbedder(dim=8), tmp_path)
    second = CachedEmbedder(_CountingEmbedder(dim=16), tmp_path)

    first.encode(DOCS, is_query=False)
    second.encode(DOCS, is_query=False)

    assert second.hits == 0
    assert second.misses == 1


def test_document_order_is_part_of_the_key(tmp_path):
    # Vectors are returned positionally, so a reordered corpus must miss.
    cached = CachedEmbedder(_CountingEmbedder(), tmp_path)
    cached.encode(DOCS, is_query=False)
    cached.encode(list(reversed(DOCS)), is_query=False)

    assert cached.hits == 0


def test_a_corrupt_entry_is_treated_as_a_miss(tmp_path):
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    cached.encode(DOCS, is_query=False)

    for entry in tmp_path.glob("*.npz"):
        entry.write_bytes(b"not a real npz file")

    # An interrupted run leaves truncated files behind; they must not crash the
    # next run.
    vectors = cached.encode(DOCS, is_query=False)
    assert inner.document_calls == 2
    assert vectors.shape == (len(DOCS), inner.dim)


def test_an_unwritable_cache_dir_does_not_fail_the_run(tmp_path):
    blocker = tmp_path / "blocked"
    # A file where the cache directory should be: mkdir will fail.
    blocker.write_text("not a directory")

    cached = CachedEmbedder(_CountingEmbedder(), blocker)
    vectors = cached.encode(DOCS, is_query=False)

    assert vectors.shape[0] == len(DOCS)


def test_build_embedder_wraps_only_when_a_cache_dir_is_given(tmp_path):
    assert isinstance(build_embedder("hashing", cache_dir=tmp_path), CachedEmbedder)
    assert not isinstance(build_embedder("hashing"), CachedEmbedder)
    assert not isinstance(build_embedder("hashing", cache_dir=None), CachedEmbedder)


def test_cache_dir_reaches_embedders_nested_inside_composites(tmp_path):
    # The regression this guards: `e5-base` appears standalone, as a hybrid
    # component, and as a rerank base in the real config. If the cache dir
    # stops at the top level, the corpus gets embedded once per mention.
    system = build_system(
        {
            "kind": "hybrid",
            "name": "h",
            "components": [
                {"kind": "bm25", "name": "s"},
                {"kind": "dense", "name": "d", "embedder": {"kind": "hashing", "dim": 32}},
            ],
        },
        cache_dir=tmp_path,
    )
    dense = system.components[1]
    assert isinstance(dense.embedder, CachedEmbedder)


def test_two_systems_sharing_a_model_embed_the_corpus_once(tmp_path):
    # The whole point of the cache, end to end through the runner.
    config = RunConfig(
        name="cache-test",
        datasets=[{"source": "smoke"}],
        systems=[
            {"kind": "dense", "name": "a", "embedder": {"kind": "hashing", "dim": 64}},
            {"kind": "dense", "name": "b", "embedder": {"kind": "hashing", "dim": 64}},
        ],
        metrics=["ndcg@10"],
        top_k=10,
        warmup_queries=0,
        cache_dir=tmp_path,
    )
    payload = run(config, verbose=False)

    assert all(row["status"] == "ok" for row in payload["results"])
    # One corpus, one model, two systems -> exactly one cache entry.
    assert len(list(tmp_path.glob("*.npz"))) == 1

    # And both systems must still score identically -- the cache must not
    # change results, only how long they take to get.
    scores = {row["system"]: row["metrics"]["ndcg@10"] for row in payload["results"]}
    assert scores["a"] == pytest.approx(scores["b"])


def test_config_accepts_a_null_cache_dir(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: x\n"
        "datasets:\n  - source: smoke\n"
        "systems:\n  - kind: bm25\n    name: bm25\n"
        "metrics:\n  - ndcg@10\n"
        "top_k: 10\n"
        "cache_dir: null\n",
        encoding="utf-8",
    )
    assert RunConfig.from_file(path).cache_dir is None
