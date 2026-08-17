"""Cost model tests, including the rounding behaviour that trips people up."""

import datetime as dt

import pytest

from bakeoff.cost import (
    PRICES,
    CostBreakdown,
    Price,
    Usage,
    estimate_tokens,
    price_usage,
    stale_entries,
)


def test_price_requires_exactly_one_rate():
    with pytest.raises(ValueError):
        Price(
            provider="x",
            model="y",
            as_of=dt.date(2026, 1, 1),
            source="",
            usd_per_million_tokens=1.0,
            usd_per_1k_searches=1.0,
        )
    with pytest.raises(ValueError):
        Price(provider="x", model="y", as_of=dt.date(2026, 1, 1), source="")


def test_every_published_price_has_a_source_and_a_date():
    for key, price in PRICES.items():
        assert price.source.startswith("http"), f"{key} has no source link"
        assert price.as_of <= dt.date.today(), f"{key} is dated in the future"


def test_stale_entries_flags_old_prices():
    old = dt.date.today() - dt.timedelta(days=400)
    assert Price(
        provider="x", model="y", as_of=old, source="http://e", usd_per_million_tokens=1.0
    ).is_stale(dt.date.today())
    # The real registry is checked as of its own dates, so nothing is stale
    # relative to the newest entry in it.
    newest = max(price.as_of for price in PRICES.values())
    assert stale_entries(newest) == []


def test_index_and_query_costs_are_reported_separately():
    usage = Usage(
        index_tokens=1_000_000,
        query_tokens=1_000,
        queries=100,
        price_keys=["openai/text-embedding-3-small"],
    )
    cost = price_usage(usage)

    # 1M tokens at $0.02/M.
    assert cost.index_usd == pytest.approx(0.02)
    # 1k query tokens at $0.02/M = $0.00002 over 100 queries, scaled to 1000.
    assert cost.usd_per_1k_queries == pytest.approx(0.0002)


def test_rerank_search_units_round_up_per_query():
    # 100 queries reranking 25 documents each. Cohere bills per 100-document
    # search unit, and a 25-document call still costs a whole unit -- so this
    # is 100 units, not 25. Getting this wrong understates rerank cost 4x.
    usage = Usage(
        reranked_docs=2_500,
        queries=100,
        price_keys=["cohere/rerank-english-v3.0"],
    )
    cost = price_usage(usage)
    # 100 units / 1000 * $2.00 = $0.20 over 100 queries -> $2.00 per 1k.
    assert cost.usd_per_1k_queries == pytest.approx(2.00)


def test_rerank_over_a_hundred_docs_costs_two_units():
    usage = Usage(
        reranked_docs=150 * 10,
        queries=10,
        price_keys=["cohere/rerank-english-v3.0"],
    )
    cost = price_usage(usage)
    # 150 docs per query -> 2 units per query -> 20 units total.
    assert cost.usd_per_1k_queries == pytest.approx(20 / 1_000 * 2.00 / 10 * 1_000)


def test_local_compute_is_priced_when_an_hourly_rate_is_given():
    usage = Usage(queries=1_000, index_seconds=36.0, query_seconds=3.6)
    free = price_usage(usage)
    assert free.index_usd == 0.0
    assert free.includes_compute is False

    paid = price_usage(usage, compute_usd_per_hour=1.0)
    assert paid.index_usd == pytest.approx(0.01)  # 36s at $1/hr
    assert paid.usd_per_1k_queries == pytest.approx(0.001)  # 3.6s at $1/hr over 1k
    assert paid.includes_compute is True


def test_unknown_price_key_is_a_loud_failure():
    usage = Usage(queries=1, price_keys=["nobody/nothing"])
    with pytest.raises(KeyError, match="nobody/nothing"):
        price_usage(usage)


def test_zero_queries_does_not_divide_by_zero():
    assert price_usage(Usage()).usd_per_1k_queries == 0.0


def test_usage_merge_sums_spend_but_takes_max_queries():
    # Two components each answering the same 10 queries is 10 queries, not 20.
    a = Usage(index_tokens=100, queries=10, price_keys=["openai/text-embedding-3-small"])
    b = Usage(index_tokens=50, queries=10, price_keys=["voyage/voyage-3"])
    merged = a.merge(b)

    assert merged.index_tokens == 150
    assert merged.queries == 10
    assert merged.price_keys == ["openai/text-embedding-3-small", "voyage/voyage-3"]


def test_merge_propagates_the_estimated_flag():
    exact = Usage(tokens_estimated=False)
    estimated = Usage(tokens_estimated=True)
    assert exact.merge(estimated).tokens_estimated is True


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens([]) == 0
    assert estimate_tokens(["a" * 400]) == 100


def test_short_texts_cost_at_least_one_token_each():
    # Regression guard. Summing characters across the batch and dividing once
    # priced anything under four characters at zero, which made query-side cost
    # disappear on short queries -- the workload where it matters most.
    assert estimate_tokens(["cat"]) == 1
    assert estimate_tokens(["a", "b", "c"]) == 3
    # Empty strings are not billable.
    assert estimate_tokens(["", ""]) == 0


def test_breakdown_serialises_roundly():
    payload = CostBreakdown(1.23456789, 0.000123456, True).as_dict()
    assert payload["index_usd"] == 1.234568
    assert payload["tokens_estimated"] is True
