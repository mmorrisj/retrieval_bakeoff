"""Cost modelling.

The point of this module is to make the second question a buyer asks -- "what
will it cost to run?" -- answerable in the same table as the first one. Two
things it tries hard to get right:

1. **Indexing and querying are billed separately.** Embedding a corpus is a
   one-off (or re-run-on-change) capital cost; embedding a query and reranking
   its candidates is a marginal cost paid on every request. Blending them into
   a single number flatters large-corpus/low-traffic systems and punishes the
   reverse, so they are reported as separate columns.

2. **Local models are not free.** A self-hosted embedder has no invoice, but it
   has a machine under it. Given an hourly rate, the harness converts measured
   wall-clock into dollars so "free" open models can be compared honestly with
   metered APIs.

Prices go stale. Every entry carries the date it was checked and a link, and
`stale_entries()` flags anything older than the freshness window so a report
cannot silently quote year-old pricing.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

# Approximate characters per token for English prose. Used only when a provider
# does not return exact usage; every estimate derived from it is marked as such
# in the report so nobody mistakes it for billing truth.
CHARS_PER_TOKEN = 4.0

# How old a price may be before the report warns about it.
PRICE_FRESHNESS_DAYS = 90


@dataclass(frozen=True)
class Price:
    """A provider's published rate for one billable unit.

    Exactly one of the two rate fields is set. Embedding models bill per token;
    rerankers generally bill per search unit (one query against up to N
    documents), which is a different shape and cannot be converted to tokens.
    """

    provider: str
    model: str
    as_of: dt.date
    source: str
    usd_per_million_tokens: float | None = None
    usd_per_1k_searches: float | None = None
    # Documents included in a single billable search unit, for rerankers that
    # define one (Cohere bills every 100 documents as one unit).
    docs_per_search_unit: int = 100

    def __post_init__(self) -> None:
        rates = [self.usd_per_million_tokens, self.usd_per_1k_searches]
        if sum(rate is not None for rate in rates) != 1:
            raise ValueError(
                f"{self.provider}/{self.model}: set exactly one of "
                "usd_per_million_tokens or usd_per_1k_searches"
            )

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def is_stale(self, today: dt.date, window_days: int = PRICE_FRESHNESS_DAYS) -> bool:
        return (today - self.as_of).days > window_days


# ---------------------------------------------------------------------------
# Published rates. Verified on the date in each entry; re-check before quoting.
# Adding a provider means adding a line here and nothing else.
# ---------------------------------------------------------------------------
PRICES: dict[str, Price] = {
    price.key: price
    for price in [
        Price(
            provider="openai",
            model="text-embedding-3-small",
            usd_per_million_tokens=0.02,
            as_of=dt.date(2026, 8, 17),
            source="https://openai.com/api/pricing/",
        ),
        Price(
            provider="openai",
            model="text-embedding-3-large",
            usd_per_million_tokens=0.13,
            as_of=dt.date(2026, 8, 17),
            source="https://openai.com/api/pricing/",
        ),
        Price(
            provider="cohere",
            model="embed-english-v3.0",
            usd_per_million_tokens=0.10,
            as_of=dt.date(2026, 8, 17),
            source="https://cohere.com/pricing",
        ),
        Price(
            provider="cohere",
            model="rerank-english-v3.0",
            usd_per_1k_searches=2.00,
            docs_per_search_unit=100,
            as_of=dt.date(2026, 8, 17),
            source="https://cohere.com/pricing",
        ),
        Price(
            provider="voyage",
            model="voyage-3",
            usd_per_million_tokens=0.06,
            as_of=dt.date(2026, 8, 17),
            source="https://docs.voyageai.com/docs/pricing",
        ),
    ]
}


def stale_entries(today: dt.date | None = None) -> list[Price]:
    """Prices older than the freshness window, for the report to warn about."""
    today = today or dt.date.today()
    return [price for price in PRICES.values() if price.is_stale(today)]


def estimate_tokens(texts: list[str]) -> int:
    """Rough token count from character length.

    Only used when a provider gives no usage figure. Deliberately crude and
    clearly labelled -- a better estimate would need the model's own tokeniser,
    which would mean pulling a heavyweight dependency into the core harness.

    Counted per text and rounded up, so a short query costs one token rather
    than zero. Summing the characters first and dividing once looks equivalent
    but silently prices every sub-four-character input at nothing, which makes
    query-side cost vanish on exactly the workloads where it matters most.
    """
    return sum(math.ceil(len(text) / CHARS_PER_TOKEN) for text in texts if text)


@dataclass
class Usage:
    """Billable work done by one system during one benchmark run.

    Populated by the retriever itself: API-backed systems record exact usage
    from the provider response, local systems record wall-clock instead.
    """

    # Tokens sent while building the index (the whole corpus, once).
    index_tokens: int = 0
    # Tokens sent while running the evaluation queries.
    query_tokens: int = 0
    # Documents scored by a reranker across all evaluation queries.
    reranked_docs: int = 0
    # Number of evaluation queries these figures cover.
    queries: int = 0
    # Wall-clock seconds of local compute, split the same way.
    index_seconds: float = 0.0
    query_seconds: float = 0.0
    # True when token counts came from a heuristic rather than a provider.
    tokens_estimated: bool = False
    # Which price entries this system's cost depends on.
    price_keys: list[str] = field(default_factory=list)

    def merge(self, other: Usage) -> Usage:
        """Combine usage from composed systems, e.g. a retriever plus a reranker."""
        return Usage(
            index_tokens=self.index_tokens + other.index_tokens,
            query_tokens=self.query_tokens + other.query_tokens,
            reranked_docs=self.reranked_docs + other.reranked_docs,
            queries=max(self.queries, other.queries),
            index_seconds=self.index_seconds + other.index_seconds,
            query_seconds=self.query_seconds + other.query_seconds,
            tokens_estimated=self.tokens_estimated or other.tokens_estimated,
            price_keys=sorted({*self.price_keys, *other.price_keys}),
        )


@dataclass(frozen=True)
class CostBreakdown:
    """What one system costs, split into the two figures worth comparing."""

    index_usd: float
    usd_per_1k_queries: float
    tokens_estimated: bool
    # Set when the figures include amortised local compute rather than API spend.
    includes_compute: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "index_usd": round(self.index_usd, 6),
            "usd_per_1k_queries": round(self.usd_per_1k_queries, 6),
            "tokens_estimated": self.tokens_estimated,
            "includes_compute": self.includes_compute,
        }


def price_usage(usage: Usage, *, compute_usd_per_hour: float = 0.0) -> CostBreakdown:
    """Convert recorded usage into an indexing cost and a per-1k-query cost.

    `compute_usd_per_hour` is the hourly rate of the machine running local
    models; leave it at zero to price API spend alone. A sensible value is
    whatever the equivalent cloud instance costs -- the config ships with one.
    """
    index_usd = 0.0
    query_usd = 0.0

    for key in usage.price_keys:
        price = PRICES.get(key)
        if price is None:
            raise KeyError(f"no published price for {key!r}; add it to bakeoff.cost.PRICES")

        if price.usd_per_million_tokens is not None:
            index_usd += usage.index_tokens / 1_000_000 * price.usd_per_million_tokens
            query_usd += usage.query_tokens / 1_000_000 * price.usd_per_million_tokens

        if price.usd_per_1k_searches is not None:
            # Rerankers bill per search unit, where a unit covers up to
            # docs_per_search_unit documents and any remainder rounds up.
            units_per_query = _ceil_div(
                _ceil_div(usage.reranked_docs, max(usage.queries, 1)),
                price.docs_per_search_unit,
            )
            total_units = units_per_query * max(usage.queries, 1)
            query_usd += total_units / 1_000 * price.usd_per_1k_searches

    includes_compute = compute_usd_per_hour > 0 and (
        usage.index_seconds > 0 or usage.query_seconds > 0
    )
    if includes_compute:
        usd_per_second = compute_usd_per_hour / 3_600
        index_usd += usage.index_seconds * usd_per_second
        query_usd += usage.query_seconds * usd_per_second

    # Normalise the marginal cost to a per-1000-queries figure, which is the
    # unit people actually budget in.
    per_1k = query_usd / usage.queries * 1_000 if usage.queries else 0.0

    return CostBreakdown(
        index_usd=index_usd,
        usd_per_1k_queries=per_1k,
        tokens_estimated=usage.tokens_estimated,
        includes_compute=includes_compute,
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)
