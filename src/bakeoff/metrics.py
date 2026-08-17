"""Ranking metrics.

These are implemented here rather than pulled from a library so the definitions
are visible and testable. Every function is checked in `tests/test_metrics.py`
against values computed by hand, because a benchmark whose metrics are wrong is
worse than no benchmark at all.

Conventions, which are the ones BEIR and the TREC tooling use:

* Relevance judgements ("qrels") are graded non-negative integers. A document
  absent from the qrels is treated as relevance 0, not as unjudged-and-skipped.
* `Recall@k` binarises those grades at `rel_threshold` (default 1).
* Metrics are computed per query and then averaged unweighted across queries,
  so a query with many relevant documents does not dominate the mean.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# A ranked list of document ids, best first.
Ranking = Sequence[str]
# query id -> {doc id -> graded relevance}
Qrels = Mapping[str, Mapping[str, int]]


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of an already-ordered list of gains.

    Uses the standard log2(rank + 1) discount with ranks starting at 1, so the
    first position is undiscounted.
    """
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranking: Ranking, relevance: Mapping[str, int], k: int) -> float:
    """Normalised DCG at k for a single query.

    Returns 0.0 when the query has no relevant documents. That is a deliberate
    choice: such queries are unanswerable, and scoring them 1.0 (as some
    implementations do) inflates every system equally and hides the problem.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    actual = dcg([float(relevance.get(doc_id, 0)) for doc_id in ranking[:k]])

    ideal_gains = sorted((float(g) for g in relevance.values() if g > 0), reverse=True)
    ideal = dcg(ideal_gains[:k])

    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(
    ranking: Ranking,
    relevance: Mapping[str, int],
    k: int,
    rel_threshold: int = 1,
) -> float:
    """Fraction of the query's relevant documents that appear in the top k.

    This is the metric that matters most for a RAG pipeline: anything the
    retriever misses at this stage cannot be recovered by a reranker or by the
    generator downstream.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = {doc_id for doc_id, grade in relevance.items() if grade >= rel_threshold}
    if not relevant:
        return 0.0

    found = sum(1 for doc_id in ranking[:k] if doc_id in relevant)
    return found / len(relevant)


def mrr_at_k(
    ranking: Ranking,
    relevance: Mapping[str, int],
    k: int,
    rel_threshold: int = 1,
) -> float:
    """Reciprocal rank of the first relevant document, or 0.0 if none in top k."""
    if k <= 0:
        raise ValueError("k must be positive")

    for rank, doc_id in enumerate(ranking[:k], start=1):
        if relevance.get(doc_id, 0) >= rel_threshold:
            return 1.0 / rank
    return 0.0


# Metric name -> callable, used to drive evaluation from config strings such as
# "ndcg@10". Keeping the registry here means adding a metric is a one-line change.
_METRICS = {
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "mrr": mrr_at_k,
}


def parse_metric(spec: str) -> tuple[str, int]:
    """Split a metric spec such as ``"ndcg@10"`` into ``("ndcg", 10)``."""
    name, _, cutoff = spec.partition("@")
    name = name.strip().lower()
    if name not in _METRICS:
        known = ", ".join(sorted(_METRICS))
        raise ValueError(f"unknown metric {name!r}; known metrics are: {known}")
    if not cutoff:
        raise ValueError(f"metric {spec!r} is missing a cutoff, e.g. {name}@10")
    k = int(cutoff)
    if k <= 0:
        raise ValueError(f"metric {spec!r} has a non-positive cutoff")
    return name, k


def evaluate(
    runs: Mapping[str, Ranking],
    qrels: Qrels,
    metrics: Sequence[str],
) -> dict[str, float]:
    """Average each metric over the queries present in `qrels`.

    Queries in `qrels` with no entry in `runs` score zero rather than being
    dropped, so a system that fails on some queries is penalised for it instead
    of quietly being measured on an easier subset.
    """
    parsed = [(spec, *parse_metric(spec)) for spec in metrics]
    if not qrels:
        return {spec: 0.0 for spec, _, _ in parsed}

    totals: dict[str, float] = {spec: 0.0 for spec, _, _ in parsed}
    for query_id, relevance in qrels.items():
        ranking = runs.get(query_id, ())
        for spec, name, k in parsed:
            totals[spec] += _METRICS[name](ranking, relevance, k)

    return {spec: total / len(qrels) for spec, total in totals.items()}
