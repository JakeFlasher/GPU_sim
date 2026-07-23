"""Accuracy metrics for the cache-replay comparison.

Reuses the canonical metric helpers from
``workloads/simbench_bw/monotonic_calibration_head.py`` so the replay comparison
is computed on exactly the same contract (epsilon-floored relative error,
log-space geometric mean) as the existing v5 / Nugteren baseline comparison.
A self-contained fallback reproduces the same formulas if the canonical module
cannot be imported.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SIMBENCH = _REPO_ROOT / "workloads" / "simbench_bw"

EPSILON = 0.01

try:
    if str(_SIMBENCH) not in sys.path:
        sys.path.insert(0, str(_SIMBENCH))
    from monotonic_calibration_head import (  # type: ignore[import-not-found]
        geomean_relative_error as _canonical_geomean,
        median_absolute_error as _canonical_median_abs,
    )
    _HAVE_CANONICAL = True
except Exception:  # pragma: no cover - defensive fallback
    _HAVE_CANONICAL = False


def median_absolute_error(predictions: Sequence[float], labels: Sequence[float]) -> float:
    if _HAVE_CANONICAL:
        return float(_canonical_median_abs(list(predictions), list(labels)))
    return float(statistics.median(
        abs(p - y) for p, y in zip(predictions, labels)
    ))


def geomean_relative_error(
    predictions: Sequence[float], labels: Sequence[float], *, epsilon: float = EPSILON,
) -> float:
    if _HAVE_CANONICAL:
        return float(_canonical_geomean(list(predictions), list(labels), epsilon=epsilon))
    rels = [abs(p - y) / max(abs(y), epsilon) for p, y in zip(predictions, labels)]
    return math.exp(sum(math.log(r + 1e-12) for r in rels) / len(rels))


def p90_absolute_error(predictions: Sequence[float], labels: Sequence[float]) -> float:
    errs = sorted(abs(p - y) for p, y in zip(predictions, labels))
    if not errs:
        return 0.0
    if len(errs) == 1:
        return errs[0]
    # nearest-rank 90th percentile
    rank = max(0, min(len(errs) - 1, math.ceil(0.9 * len(errs)) - 1))
    return errs[rank]
