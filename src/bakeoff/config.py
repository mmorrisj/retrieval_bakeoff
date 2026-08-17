"""Run configuration.

A run is fully described by one YAML file: which datasets, which systems, which
metrics, and the environment assumptions used for costing. Keeping it in a file
rather than in CLI flags is what makes a result reproducible -- the config is
copied into the results directory alongside the numbers, so a table can always
be traced back to the exact setup that produced it.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_METRICS = ("ndcg@10", "recall@50", "mrr@10")


@dataclass
class RunConfig:
    name: str
    datasets: list[dict[str, Any]]
    systems: list[dict[str, Any]]
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    # Documents retrieved per query. Must be at least the largest metric cutoff
    # or the metrics silently measure a truncated ranking.
    top_k: int = 100
    # Queries timed and discarded before latency recording starts.
    warmup_queries: int = 3
    # Cap on evaluation queries per dataset, for quick runs. None means all.
    max_queries: int | None = None
    # Hourly rate of the machine, used to price local compute. Defaults to zero
    # so API spend is reported alone unless the operator opts in.
    compute_usd_per_hour: float = 0.0
    data_dir: pathlib.Path = pathlib.Path("data")
    seed: int = 0

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> RunConfig:
        path = pathlib.Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        unknown = set(payload) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"{path}: unknown config keys: {', '.join(sorted(unknown))}")

        if "data_dir" in payload:
            payload["data_dir"] = pathlib.Path(payload["data_dir"])

        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.datasets:
            raise ValueError("config lists no datasets")
        if not self.systems:
            raise ValueError("config lists no systems")

        names = [system.get("name", system.get("kind")) for system in self.systems]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"system names must be unique; repeated: {', '.join(sorted(duplicates))}"
            )

        from bakeoff.metrics import parse_metric

        cutoffs = [parse_metric(metric)[1] for metric in self.metrics]
        if cutoffs and max(cutoffs) > self.top_k:
            raise ValueError(
                f"top_k={self.top_k} is below the largest metric cutoff ({max(cutoffs)}); "
                "the metrics would be computed over a truncated ranking"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "datasets": self.datasets,
            "systems": self.systems,
            "metrics": self.metrics,
            "top_k": self.top_k,
            "warmup_queries": self.warmup_queries,
            "max_queries": self.max_queries,
            "compute_usd_per_hour": self.compute_usd_per_hour,
            "data_dir": str(self.data_dir),
            "seed": self.seed,
        }
