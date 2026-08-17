"""Retriever behaviour: ranking sanity, determinism, and composition."""

import numpy as np
import pytest

from bakeoff.systems import BM25Retriever, DenseRetriever, HybridRetriever, build_system
from bakeoff.systems.base import Document, Hit, Retriever
from bakeoff.systems.embedders import HashingEmbedder, l2_normalize

CORPUS = [
    Document("d1", "the cat sat on the mat in the kitchen"),
    Document("d2", "dogs bark loudly at the postman every morning"),
    Document("d3", "a cat is a small domesticated carnivorous mammal"),
    Document("d4", "sailing upwind requires tacking through the no-go zone"),
]


def test_bm25_ranks_the_on_topic_document_first():
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    hits = retriever.search("domesticated cat mammal", k=4)

    assert hits[0].doc_id == "d3"
    assert {hit.doc_id for hit in hits} >= {"d1", "d3"}


def test_bm25_returns_nothing_for_out_of_vocabulary_queries():
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    assert retriever.search("zzzz qqqq", k=10) == []


def test_bm25_respects_k():
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    assert len(retriever.search("the", k=2)) <= 2


def test_bm25_scores_are_descending():
    retriever = BM25Retriever()
    retriever.index(CORPUS)
    scores = [hit.score for hit in retriever.search("cat kitchen mat", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_bm25_is_deterministic_across_instances():
    first = BM25Retriever()
    first.index(CORPUS)
    second = BM25Retriever()
    second.index(list(CORPUS))

    query = "the cat"
    assert [hit.doc_id for hit in first.search(query, 4)] == [
        hit.doc_id for hit in second.search(query, 4)
    ]


def test_bm25_ties_break_on_document_id():
    # Two identical documents must come back in a stable, id-sorted order.
    corpus = [Document("b", "identical text here"), Document("a", "identical text here")]
    retriever = BM25Retriever()
    retriever.index(corpus)
    assert [hit.doc_id for hit in retriever.search("identical text", 2)] == ["a", "b"]


def test_l2_normalize_leaves_zero_vectors_finite():
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normalised = l2_normalize(matrix)

    assert np.all(np.isfinite(normalised))
    assert normalised[0].tolist() == [0.0, 0.0]
    assert np.linalg.norm(normalised[1]) == pytest.approx(1.0)


def test_hashing_embedder_is_stable_across_calls():
    embedder = HashingEmbedder(dim=64)
    first = embedder.encode(["repeatable text"], is_query=False)
    second = embedder.encode(["repeatable text"], is_query=False)
    assert np.allclose(first, second)


def test_dense_retriever_ranks_by_cosine_similarity():
    retriever = DenseRetriever(HashingEmbedder(dim=256))
    retriever.index(CORPUS)
    hits = retriever.search("cat mammal domesticated", k=4)

    assert hits[0].doc_id == "d3"
    assert len(hits) == 4


def test_dense_search_before_index_is_an_error():
    retriever = DenseRetriever(HashingEmbedder())
    with pytest.raises(RuntimeError, match="index"):
        retriever.search("anything", 5)


def test_dense_handles_k_larger_than_the_corpus():
    retriever = DenseRetriever(HashingEmbedder(dim=64))
    retriever.index(CORPUS)
    assert len(retriever.search("cat", k=100)) == len(CORPUS)


def test_dense_records_index_and_query_tokens_separately():
    retriever = DenseRetriever(HashingEmbedder(dim=64))
    retriever.index(CORPUS)
    indexed = retriever.usage.index_tokens
    assert indexed > 0
    assert retriever.usage.query_tokens == 0

    retriever.search("cat", k=2)
    assert retriever.usage.index_tokens == indexed  # unchanged by searching
    assert retriever.usage.query_tokens > 0
    assert retriever.usage.queries == 1


class _StubRetriever(Retriever):
    """Returns a fixed ranking, so fusion can be checked exactly."""

    def __init__(self, name, ranking):
        super().__init__(name)
        self.ranking = ranking

    def index(self, corpus):
        pass

    def search(self, query, k):
        self.usage.queries += 1
        return [Hit(doc_id, 1.0) for doc_id in self.ranking[:k]]


def test_rrf_promotes_the_document_both_systems_rank_well():
    # The property RRF exists for: consistently-good beats spectacular-but-
    # disputed. "b" is 2nd on both sides; "a" and "d" are each 1st on one side
    # and last on the other. With rrf_k=60:
    #   a: 1/61 + 1/64 = 0.032018
    #   b: 1/62 + 1/62 = 0.032258  -> b wins
    #   c: 1/63 + 1/63 = 0.031746
    #   d: 1/64 + 1/61 = 0.032018  -> ties with a, broken on doc id
    left = _StubRetriever("left", ["a", "b", "c", "d"])
    right = _StubRetriever("right", ["d", "b", "c", "a"])
    hybrid = HybridRetriever([left, right], candidate_k=4)
    hybrid.index(CORPUS)

    assert [hit.doc_id for hit in hybrid.search("q", 3)] == ["b", "a", "d"]


def test_hybrid_weights_shift_the_fusion():
    left = _StubRetriever("left", ["a", "b"])
    right = _StubRetriever("right", ["b", "a"])
    # Weighting the left system heavily should hand it the top slot.
    hybrid = HybridRetriever([left, right], candidate_k=2, weights=[10.0, 1.0])
    hybrid.index(CORPUS)
    assert hybrid.search("q", 1)[0].doc_id == "a"


def test_hybrid_needs_at_least_two_components():
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever([_StubRetriever("only", ["a"])])


def test_hybrid_rejects_mismatched_weights():
    components = [_StubRetriever("a", ["x"]), _StubRetriever("b", ["y"])]
    with pytest.raises(ValueError, match="one entry per component"):
        HybridRetriever(components, weights=[1.0])


def test_hybrid_usage_counts_each_query_once_not_once_per_component():
    left = _StubRetriever("left", ["a", "b"])
    right = _StubRetriever("right", ["b", "a"])
    hybrid = HybridRetriever([left, right], candidate_k=2)
    hybrid.index(CORPUS)
    hybrid.search("q", 2)

    assert hybrid.usage.queries == 1


def test_build_system_constructs_a_nested_hybrid():
    system = build_system(
        {
            "kind": "hybrid",
            "name": "h",
            "candidate_k": 10,
            "components": [
                {"kind": "bm25", "name": "s"},
                {"kind": "dense", "name": "d", "embedder": {"kind": "hashing", "dim": 32}},
            ],
        }
    )
    assert isinstance(system, HybridRetriever)
    assert system.name == "h"
    assert [component.name for component in system.components] == ["s", "d"]


def test_build_system_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="unknown system kind"):
        build_system({"kind": "telepathy"})


def test_build_system_requires_a_kind():
    with pytest.raises(ValueError, match="missing a 'kind'"):
        build_system({"name": "nameless"})


def test_build_system_requires_an_embedder_for_dense():
    with pytest.raises(ValueError, match="embedder.kind"):
        build_system({"kind": "dense", "name": "d"})
