"""Small, dependency-light scoring helpers for direct simulator experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence


class ScoringError(ValueError):
    """The supplied observations cannot be scored honestly."""


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    mean_absolute_error: float
    mean_absolute_percentage_error: float | None
    root_mean_squared_error: float


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    observed_numeric_count: int
    predicted_numeric_count: int
    shared_numeric_count: int
    observed_fraction: float
    predicted_fraction: float


@dataclass(frozen=True, slots=True)
class KeyedMetricSummary:
    coverage: CoverageSummary
    shared_keys: tuple[str, ...]
    errors: MetricSummary


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{name} contains a non-numeric value")
    try:
        number = float(value)
    except OverflowError as error:
        raise ScoringError(f"{name} contains a value outside the finite float range") from error
    if not math.isfinite(number):
        raise ScoringError(f"{name} contains a non-finite value")
    return number


def _finite(values: Iterable[object], *, name: str) -> tuple[float, ...]:
    material = tuple(_finite_number(value, name=name) for value in values)
    if not material:
        raise ScoringError(f"{name} is empty")
    return material


def summarize_metric(
    observed: Iterable[object], predicted: Iterable[object]
) -> MetricSummary:
    """Return MAE, RMSE, and MAPE when all observations are nonzero."""

    truth = _finite(observed, name="observed values")
    estimate = _finite(predicted, name="predicted values")
    if len(truth) != len(estimate):
        raise ScoringError("observed and predicted lengths differ")

    absolute_errors = tuple(
        abs(actual - guess) for actual, guess in zip(truth, estimate)
    )
    squared_errors = tuple(
        (actual - guess) ** 2 for actual, guess in zip(truth, estimate)
    )
    percentage_errors = (
        None
        if any(actual == 0.0 for actual in truth)
        else tuple(
            error / abs(actual)
            for error, actual in zip(absolute_errors, truth)
        )
    )
    return MetricSummary(
        count=len(truth),
        mean_absolute_error=fmean(absolute_errors),
        mean_absolute_percentage_error=(
            None if percentage_errors is None else fmean(percentage_errors)
        ),
        root_mean_squared_error=math.sqrt(fmean(squared_errors)),
    )


def _numeric_metrics(
    values: Mapping[str, object], *, name: str
) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ScoringError(f"{name} contains a non-string key")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric[key] = _finite_number(value, name=f"{name} metric {key!r}")
    if not numeric:
        raise ScoringError(f"{name} contains no numeric metrics")
    return numeric


def summarize_keyed_metrics(
    observed: Mapping[str, object], predicted: Mapping[str, object]
) -> KeyedMetricSummary:
    """Join shared numeric keys and report coverage plus direct errors."""

    truth = _numeric_metrics(observed, name="observed object")
    estimate = _numeric_metrics(predicted, name="predicted object")
    shared_keys = tuple(sorted(truth.keys() & estimate.keys()))
    if not shared_keys:
        raise ScoringError("observed and predicted objects share no numeric keys")
    shared_count = len(shared_keys)
    return KeyedMetricSummary(
        coverage=CoverageSummary(
            observed_numeric_count=len(truth),
            predicted_numeric_count=len(estimate),
            shared_numeric_count=shared_count,
            observed_fraction=shared_count / len(truth),
            predicted_fraction=shared_count / len(estimate),
        ),
        shared_keys=shared_keys,
        errors=summarize_metric(
            (truth[key] for key in shared_keys),
            (estimate[key] for key in shared_keys),
        ),
    )


def _load_payload(
    path: Path, key: str | None
) -> list[object] | dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            raise ScoringError(f"{path} does not contain key {key!r}")
        value = value[key]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value
    raise ScoringError(f"{path} must contain a JSON array or object")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two JSON arrays or keyed numeric objects."
    )
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument(
        "--key", help="optional object key containing the arrays or metric objects"
    )
    args = parser.parse_args(argv)
    try:
        observed = _load_payload(args.observed, args.key)
        predicted = _load_payload(args.predicted, args.key)
        if isinstance(observed, list) and isinstance(predicted, list):
            summary = summarize_metric(observed, predicted)
        elif isinstance(observed, dict) and isinstance(predicted, dict):
            summary = summarize_keyed_metrics(observed, predicted)
        else:
            raise ScoringError(
                "observed and predicted JSON roots must have the same type"
            )
    except (OSError, ValueError, ScoringError) as error:
        print(f"eval.direct_score: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
