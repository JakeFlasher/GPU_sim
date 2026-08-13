"""Small, dependency-light scoring helpers for direct simulator experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence


class ScoringError(ValueError):
    """The supplied observations cannot be scored honestly."""


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    mean_absolute_error: float
    mean_absolute_percentage_error: float | None
    root_mean_squared_error: float


def _finite(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    material = tuple(float(value) for value in values)
    if not material:
        raise ScoringError(f"{name} is empty")
    if any(not math.isfinite(value) for value in material):
        raise ScoringError(f"{name} contains a non-finite value")
    return material


def summarize_metric(
    observed: Iterable[float], predicted: Iterable[float]
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


def _load_values(path: Path, key: str | None) -> list[float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            raise ScoringError(f"{path} does not contain key {key!r}")
        value = value[key]
    if not isinstance(value, list):
        raise ScoringError(f"{path} must contain a JSON array")
    return [float(item) for item in value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two JSON arrays.")
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--key", help="optional object key containing the arrays")
    args = parser.parse_args(argv)
    try:
        summary = summarize_metric(
            _load_values(args.observed, args.key),
            _load_values(args.predicted, args.key),
        )
    except (OSError, ValueError, ScoringError) as error:
        print(f"eval.direct_score: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
