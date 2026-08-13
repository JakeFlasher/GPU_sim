"""Load the reusable NCU metric-name catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class MetricCatalogError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Metric:
    key: str
    ncu_metric: str
    unit: str


@lru_cache(maxsize=1)
def load_metrics() -> tuple[Metric, ...]:
    path = Path(__file__).with_name("metric_registry.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("metrics")
    if not isinstance(rows, list) or not rows:
        raise MetricCatalogError("metric catalog must contain a nonempty metrics array")
    metrics: list[Metric] = []
    keys: set[str] = set()
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise MetricCatalogError("each metric row must be an object")
        try:
            metric = Metric(
                key=str(row["key"]),
                ncu_metric=str(row["ncu_metric"]),
                unit=str(row["unit"]),
            )
        except KeyError as error:
            raise MetricCatalogError(f"metric row is missing {error.args[0]}") from error
        if not metric.key or not metric.ncu_metric or metric.unit not in {
            "cycles",
            "ns",
            "pct",
            "count",
        }:
            raise MetricCatalogError(f"invalid metric row: {row!r}")
        if metric.key in keys or metric.ncu_metric in names:
            raise MetricCatalogError(f"duplicate metric row: {row!r}")
        keys.add(metric.key)
        names.add(metric.ncu_metric)
        metrics.append(metric)
    return tuple(metrics)


def metric_names(keys: list[str] | None = None) -> list[str]:
    metrics = load_metrics()
    if keys is None:
        return [metric.ncu_metric for metric in metrics]
    by_key = {metric.key: metric.ncu_metric for metric in metrics}
    try:
        return [by_key[key] for key in keys]
    except KeyError as error:
        raise MetricCatalogError(f"unknown metric key: {error.args[0]}") from error
