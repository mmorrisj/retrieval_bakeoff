"""Benchmark execution.

The runner's job is to make every row in the table comparable: same corpus,
same queries in the same order, same cutoff, same timing discipline. Anything
that varies between systems has to be the system itself.

A system that cannot run -- missing extra, missing API key -- is recorded as
skipped with the reason and the run continues. A benchmark that refuses to
produce any numbers because one of six optional API keys is absent is a
benchmark nobody runs twice.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from bakeoff.config import RunConfig
from bakeoff.corpus import Dataset, load_dataset
from bakeoff.cost import CostBreakdown, price_usage, stale_entries
from bakeoff.metrics import evaluate
from bakeoff.systems import NotInstalled, build_system
from bakeoff.timing import LatencyRecorder, LatencyStats, stopwatch


@dataclass
class SystemResult:
    """One row of the final table."""

    system: str
    dataset: str
    status: str  # "ok" | "skipped" | "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float | int] = field(default_factory=dict)
    cost: dict[str, float | bool] = field(default_factory=dict)
    index_seconds: float = 0.0
    config: dict[str, str] = field(default_factory=dict)
    detail: str = ""


def _environment() -> dict[str, str]:
    """Recorded with every run, because latency figures are meaningless without it."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }


def run(config: RunConfig, *, verbose: bool = True) -> dict[str, Any]:
    """Execute every (dataset, system) pair and return the full result payload."""
    results: list[SystemResult] = []
    dataset_stats: dict[str, dict[str, int]] = {}

    for dataset_spec in config.datasets:
        dataset = load_dataset(dataset_spec, config.data_dir)
        dataset_stats[dataset.name] = dataset.stats
        if verbose:
            print(f"\n[{dataset.name}] {dataset.stats}", file=sys.stderr)

        # One fixed query order for every system on this dataset. Sorting by id
        # rather than relying on dict order keeps runs comparable across
        # machines and Python versions.
        query_ids = sorted(dataset.queries)
        if config.max_queries is not None:
            query_ids = query_ids[: config.max_queries]

        for system_spec in config.systems:
            result = _run_one(config, dataset, query_ids, system_spec, verbose=verbose)
            results.append(result)

    stale = stale_entries()
    return {
        "run": config.name,
        "config": config.as_dict(),
        "environment": _environment(),
        "datasets": dataset_stats,
        "results": [asdict(result) for result in results],
        "warnings": [
            f"price for {price.key} was last checked {price.as_of.isoformat()}" for price in stale
        ],
    }


def _run_one(
    config: RunConfig,
    dataset: Dataset,
    query_ids: list[str],
    system_spec: dict[str, Any],
    *,
    verbose: bool,
) -> SystemResult:
    label = system_spec.get("name", system_spec.get("kind", "?"))

    try:
        system = build_system(system_spec)
    except NotInstalled as exc:
        if verbose:
            print(f"  {label}: skipped ({exc})", file=sys.stderr)
        return SystemResult(label, dataset.name, "skipped", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - a bad config should not abort the run
        return SystemResult(label, dataset.name, "failed", detail=f"{type(exc).__name__}: {exc}")

    try:
        with stopwatch() as index_elapsed:
            system.index(dataset.documents)

        recorder = LatencyRecorder(warmup=config.warmup_queries)
        runs: dict[str, list[str]] = {}
        for query_id in query_ids:
            query = dataset.queries[query_id]
            # Timed one query at a time -- see bakeoff.timing for why.
            with recorder.measure():
                hits = system.search(query, config.top_k)
            runs[query_id] = [hit.doc_id for hit in hits]

        qrels = {qid: dataset.qrels[qid] for qid in query_ids}
        scores = evaluate(runs, qrels, config.metrics)
        latency = recorder.stats()

        usage = system.usage
        usage.index_seconds += index_elapsed[0]
        usage.query_seconds += recorder.total_seconds
        cost = price_usage(usage, compute_usd_per_hour=config.compute_usd_per_hour)

        if verbose:
            print(f"  {label}: {_summarise(scores, latency, cost)}", file=sys.stderr)

        return SystemResult(
            system=label,
            dataset=dataset.name,
            status="ok",
            metrics={name: round(value, 5) for name, value in scores.items()},
            latency=latency.as_dict(),
            cost=cost.as_dict(),
            index_seconds=round(index_elapsed[0], 3),
            config=system.describe(),
        )
    except NotInstalled as exc:
        return SystemResult(label, dataset.name, "skipped", detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        if verbose:
            traceback.print_exc()
        return SystemResult(label, dataset.name, "failed", detail=f"{type(exc).__name__}: {exc}")
    finally:
        system.close()


def _summarise(scores: dict[str, float], latency: LatencyStats, cost: CostBreakdown) -> str:
    parts = [f"{name}={value:.4f}" for name, value in scores.items()]
    parts.append(f"p95={latency.p95_ms:.1f}ms")
    parts.append(f"${cost.usd_per_1k_queries:.4f}/1k")
    return "  ".join(parts)


def write_results(payload: dict[str, Any], out_dir: str | pathlib.Path) -> pathlib.Path:
    """Persist the run payload so a report can be regenerated without re-running."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
