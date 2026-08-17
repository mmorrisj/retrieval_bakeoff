"""Metrics checked against values computed by hand.

Every expected number below is derived in the comment above it. A benchmark
whose metrics are subtly wrong produces confident, wrong rankings, so these are
the tests that matter most in the repo.
"""

import math

import pytest

from bakeoff.metrics import (
    dcg,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    parse_metric,
    recall_at_k,
)


def test_dcg_applies_log2_rank_plus_one_discount():
    # Positions 1, 2, 3 are discounted by log2(2)=1, log2(3), log2(4)=2.
    expected = 3 / 1.0 + 2 / math.log2(3) + 1 / 2.0
    assert dcg([3, 2, 1]) == pytest.approx(expected)


def test_dcg_of_empty_is_zero():
    assert dcg([]) == 0.0


def test_ndcg_is_one_for_the_ideal_ranking():
    relevance = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], relevance, 3) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking():
    # actual = 1/1 + 2/log2(3) + 3/2 = 1 + 1.26186 + 1.5 = 3.76186
    # ideal  = 3/1 + 2/log2(3) + 1/2 = 3 + 1.26186 + 0.5 = 4.76186
    relevance = {"a": 3, "b": 2, "c": 1}
    actual = 1 + 2 / math.log2(3) + 1.5
    ideal = 3 + 2 / math.log2(3) + 0.5
    assert ndcg_at_k(["c", "b", "a"], relevance, 3) == pytest.approx(actual / ideal)


def test_ndcg_ideal_is_truncated_at_k():
    # With k=1 the ideal DCG is just the single best gain, so retrieving the
    # best document scores 1.0 even though two relevant documents were missed.
    relevance = {"a": 3, "b": 3, "c": 3}
    assert ndcg_at_k(["a"], relevance, 1) == pytest.approx(1.0)


def test_ndcg_ignores_documents_beyond_k():
    relevance = {"a": 1}
    assert ndcg_at_k(["x", "y", "a"], relevance, 2) == 0.0


def test_ndcg_is_zero_when_nothing_is_relevant():
    # Deliberate: an unanswerable query scores 0, not 1.
    assert ndcg_at_k(["a", "b"], {}, 10) == 0.0
    assert ndcg_at_k(["a", "b"], {"a": 0}, 10) == 0.0


def test_unjudged_documents_count_as_zero_not_as_skipped():
    # "x" is absent from the qrels; it occupies rank 1 and pushes "a" down,
    # which must reduce the score rather than being ignored.
    relevance = {"a": 1}
    assert ndcg_at_k(["x", "a"], relevance, 2) == pytest.approx(1 / math.log2(3))


def test_recall_binarises_at_the_threshold():
    relevance = {"a": 2, "b": 1, "c": 0}
    # Threshold 1: both "a" and "b" count, one of two found.
    assert recall_at_k(["a", "z"], relevance, 2) == pytest.approx(0.5)
    # Threshold 2: only "a" counts, and it was found.
    assert recall_at_k(["a", "z"], relevance, 2, rel_threshold=2) == pytest.approx(1.0)


def test_recall_denominator_is_all_relevant_not_k():
    relevance = {"a": 1, "b": 1, "c": 1, "d": 1}
    assert recall_at_k(["a", "b"], relevance, 2) == pytest.approx(0.5)


def test_recall_is_zero_with_no_relevant_documents():
    assert recall_at_k(["a"], {"a": 0}, 5) == 0.0


def test_mrr_uses_the_first_relevant_rank():
    relevance = {"c": 1}
    assert mrr_at_k(["a", "b", "c"], relevance, 10) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_the_hit_falls_outside_k():
    relevance = {"c": 1}
    assert mrr_at_k(["a", "b", "c"], relevance, 2) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_non_positive_cutoffs(k):
    for metric in (ndcg_at_k, recall_at_k, mrr_at_k):
        with pytest.raises(ValueError):
            metric(["a"], {"a": 1}, k)


def test_parse_metric_splits_name_and_cutoff():
    assert parse_metric("ndcg@10") == ("ndcg", 10)
    assert parse_metric("Recall@100") == ("recall", 100)


@pytest.mark.parametrize("spec", ["ndcg", "bogus@10", "ndcg@0", "ndcg@-5"])
def test_parse_metric_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_metric(spec)


def test_evaluate_averages_over_queries_unweighted():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    runs = {"q1": ["a"], "q2": ["z"]}
    # q1 scores 1.0, q2 scores 0.0, mean 0.5.
    assert evaluate(runs, qrels, ["ndcg@10"])["ndcg@10"] == pytest.approx(0.5)


def test_evaluate_scores_missing_runs_as_zero_rather_than_dropping_them():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    # A system that answered only q1 must not be measured on q1 alone.
    assert evaluate({"q1": ["a"]}, qrels, ["ndcg@10"])["ndcg@10"] == pytest.approx(0.5)


def test_evaluate_handles_empty_qrels():
    assert evaluate({}, {}, ["ndcg@10"]) == {"ndcg@10": 0.0}
