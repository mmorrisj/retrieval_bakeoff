"""Latency measurement.

Two decisions here matter more than the code:

* **Queries are timed one at a time.** It is tempting to embed all 300
  evaluation queries in one batch and divide -- it is faster and it makes every
  system look good. It also measures throughput, not latency, and no user ever
  issues a batch of 300. Latency is measured at batch size 1 because that is
  the shape of a real request.

* **Warm-up runs are discarded.** The first call to a local model pays for lazy
  weight loading, kernel autotuning and allocator warm-up; the first call to an
  API pays for TLS and connection setup. Including those inflates the tail by
  an order of magnitude and tells you nothing about steady state.

Percentiles use linear interpolation between the two nearest ranks, matching
`numpy.percentile`'s default, so figures here line up with the ones people get
when they check the raw timings in `timings.json`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

DEFAULT_WARMUP_QUERIES = 3


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0-100) of `values`, by linear interpolation."""
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class LatencyStats:
    """Summary of per-query latencies, in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class LatencyRecorder:
    """Collects per-query wall-clock times and summarises them.

    Times recorded before `warmup` samples have been seen are kept in
    `warmup_ms` for inspection but excluded from the summary.
    """

    warmup: int = DEFAULT_WARMUP_QUERIES
    samples_ms: list[float] = field(default_factory=list)
    warmup_ms: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Time the enclosed block as one query."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1_000
            if len(self.warmup_ms) < self.warmup:
                self.warmup_ms.append(elapsed_ms)
            else:
                self.samples_ms.append(elapsed_ms)

    @property
    def total_seconds(self) -> float:
        """Wall-clock across every query including warm-up, for costing."""
        return (sum(self.samples_ms) + sum(self.warmup_ms)) / 1_000

    def stats(self) -> LatencyStats:
        # With fewer queries than the warm-up budget there is nothing left to
        # summarise; fall back to the warm-up samples and let the low `count`
        # in the report signal that the figure is weak.
        measured = self.samples_ms or self.warmup_ms
        if not measured:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return LatencyStats(
            count=len(measured),
            mean_ms=sum(measured) / len(measured),
            p50_ms=percentile(measured, 50),
            p95_ms=percentile(measured, 95),
            p99_ms=percentile(measured, 99),
            max_ms=max(measured),
        )


@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """Time a one-off phase such as index construction.

    Yields a single-element list that holds the elapsed seconds once the block
    exits, so callers can read it without a class.
    """
    holder: list[float] = []
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder.append(time.perf_counter() - start)
