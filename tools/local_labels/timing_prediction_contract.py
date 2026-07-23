"""Loader, threshold resolver, and cycle-error metric implementation for the
execution-time prediction contract.

The committed contract artifact lives at
``data/local_labels/timing_prediction_contract_sm_86.json`` and is emitted by
``tools/local_labels/emit_timing_prediction_contract.py``. This module is the
single consumer-side surface for that artifact: it loads the document,
validates it against the canonical JSON Schema
``workloads/simbench_bw/schemas/timing_prediction_contract.schema.json``
(never a hand-rolled field check that the schema already pins), resolves every
pinned threshold through typed accessors, and implements the pinned cycle-error
metrics so the downstream accuracy runner reuses one definition rather than
re-deriving it.

Fail-closed discipline: a missing artifact, malformed JSON, a document that
does not satisfy the canonical schema (for example one that promotes the
nanosecond leg to a headline role or lets the nanosecond leg close the success
terminal state on its own), or a non-positive cycle input all raise
``TimingContractError`` rather than returning a default.

Cycle-error definition (the only admitted formula for the cycle target):

    r_i = ln(p_i / t_i)                       per-cell signed log-ratio
    G   = exp(mean_i(|r_i|)) - 1              log-space geomean relative error
    M   = median_i(|r_i|)                     median absolute log-relative error

The 95 percent resampled upper bound on ``G`` reuses the same resample count
and confidence the cache runner pins (2000 / 0.95) and the same percentile
index selection. No epsilon floor is applied because cycle predictions and
labels are strictly positive; the bounded-fraction cache formula
``abs(pred - true) / max(true, epsilon)`` is rejected for the cycle target.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "timing_prediction_contract.schema.json"
)
DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "data" / "local_labels" / "timing_prediction_contract_sm_86.json"
)

# The only admitted cycle-error method identifier. A caller that asks for the
# bounded-fraction cache method is refused (see ``score_cycle_metrics``).
PINNED_CYCLE_ERROR_METHOD = "log_ratio"
FORBIDDEN_CYCLE_ERROR_METHOD = "bounded_fraction"

# Deterministic default seed for the resampled upper bound. It keeps fixture
# scoring reproducible; the production accuracy runner sources the frozen
# pre-registration seed instead of relying on this default.
DEFAULT_RESAMPLE_SEED = 20260531

PairSequence = Sequence[Sequence[float]]


class TimingContractError(ValueError):
    """Raised when the contract artifact is missing, malformed, fails schema
    validation, or when a cycle-error input violates the strictly-positive
    requirement / the pinned-method requirement."""


def load_schema() -> dict:
    """Return the canonical contract JSON Schema as a dict."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_contract(doc: dict) -> None:
    """Validate a candidate contract dict against the canonical schema.

    Delegates entirely to ``jsonschema`` against the canonical schema so the
    validation surface and the producer share one source of truth. Raises
    ``TimingContractError`` (wrapping the underlying validation error) when the
    document is non-conformant.
    """
    try:
        jsonschema.validate(instance=doc, schema=load_schema())
    except jsonschema.ValidationError as exc:
        raise TimingContractError(
            f"contract failed schema validation: {exc.message}"
        ) from exc


def load_contract(path: str | Path | None = None) -> dict:
    """Load and schema-validate the committed contract artifact."""
    target = Path(path) if path is not None else DEFAULT_CONTRACT_PATH
    if not target.is_file():
        raise TimingContractError(f"contract artifact not found: {target}")
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TimingContractError(
            f"contract artifact is not valid JSON: {exc}"
        ) from exc
    validate_contract(doc)
    return doc


# --- threshold + rule accessors --------------------------------------------


def publication_headline_metric_id(doc: dict) -> str:
    """Return the sole publication headline target metric id."""
    return doc["headline"]["publication_target_metric_id"]


def cycles_is_sole_headline(doc: dict) -> bool:
    """Return whether the cycle count is declared the sole publication
    headline (the schema pins this true, so a conformant contract always
    reports True)."""
    return bool(doc["headline"]["is_sole_publication_headline"])


def publication_gate(doc: dict) -> dict:
    """Return the publication-gate threshold block."""
    return doc["publication_gate"]


def smoke_tier(doc: dict) -> dict:
    """Return the smoke-tier block (reports-only; cannot close success)."""
    return doc["smoke_tier"]


def ns_leg(doc: dict) -> dict:
    """Return the nanosecond-leg block."""
    return doc["deploy_ns_leg"]


def ns_requires_crosswalk(doc: dict) -> bool:
    """Return whether a deploy-nanosecond claim requires the crosswalk
    artifact (the schema pins this true)."""
    return bool(doc["deploy_ns_leg"]["requires_crosswalk_artifact"])


def ns_can_close_success_alone(doc: dict) -> bool:
    """Return whether the nanosecond leg may close the success terminal state
    on its own (the schema pins this false)."""
    return bool(doc["deploy_ns_leg"]["closes_research_success_alone"])


def dual_source_noise_floor_ns(doc: dict) -> int:
    """Return the provenanced dual-source noise floor in nanoseconds."""
    return int(doc["deploy_ns_leg"]["dual_source_noise_floor_ns"])


def min_cohort_duration_ns(doc: dict, *, tighter: bool = False) -> int:
    """Return the minimum cohort kernel duration (ns) for a floor-relative
    nanosecond claim at the default 20 percent bound, or the tighter 10
    percent bound when ``tighter`` is set."""
    leg = doc["deploy_ns_leg"]
    if tighter:
        return int(leg["min_cohort_duration_ns_for_tighter_bound"])
    return int(leg["min_cohort_duration_ns_for_default_bound"])


def terminal_states(doc: dict) -> dict:
    """Return the terminal-state vocabulary block."""
    return doc["terminal_states"]


def evidence_statuses(doc: dict) -> list:
    """Return the admitted single-architecture evidence statuses."""
    return list(doc["evidence_statuses"])


# --- pinned cycle-error metrics --------------------------------------------


def signed_log_ratio(predicted_cycles: float, true_cycles: float) -> float:
    """Return the per-cell signed log-ratio ``ln(p_i / t_i)``.

    Both inputs must be strictly positive; no epsilon floor is applied, so a
    non-positive input raises rather than being clamped.
    """
    if predicted_cycles <= 0.0 or true_cycles <= 0.0:
        raise TimingContractError(
            "cycle prediction and label must be strictly positive "
            f"(got predicted={predicted_cycles!r}, true={true_cycles!r}); "
            "no epsilon floor is applied for the cycle target"
        )
    return math.log(predicted_cycles / true_cycles)


def _abs_log_ratios(pairs: PairSequence) -> list[float]:
    return [abs(signed_log_ratio(p, t)) for p, t in pairs]


def _geomean_from_abs_ratios(abs_ratios: Sequence[float]) -> float:
    return math.exp(sum(abs_ratios) / len(abs_ratios)) - 1.0


def geomean_relative_error(pairs: PairSequence) -> float:
    """Return ``G = exp(mean_i(|r_i|)) - 1`` over the scored cells."""
    abs_ratios = _abs_log_ratios(pairs)
    if not abs_ratios:
        raise TimingContractError("no scored cells for the geomean")
    return _geomean_from_abs_ratios(abs_ratios)


def median_abs_log_relative_error(pairs: PairSequence) -> float:
    """Return ``M = median_i(|r_i|)`` over the scored cells.

    Even-length cohorts use the mean of the two central order statistics.
    """
    abs_ratios = sorted(_abs_log_ratios(pairs))
    n = len(abs_ratios)
    if n == 0:
        raise TimingContractError("no scored cells for the median")
    mid = n // 2
    if n % 2 == 1:
        return abs_ratios[mid]
    return 0.5 * (abs_ratios[mid - 1] + abs_ratios[mid])


def bootstrap_geomean_upper_bound(
    pairs: PairSequence,
    *,
    resamples: int,
    confidence: float,
    seed: int = DEFAULT_RESAMPLE_SEED,
) -> float:
    """Return the resampled upper bound on ``G``.

    The cells are resampled with replacement ``resamples`` times; each resample
    recomputes ``G``; the upper bound is the percentile at ``confidence``. The
    resample count, confidence, and percentile-index selection mirror the
    pinned cache-runner bootstrap.
    """
    abs_ratios = _abs_log_ratios(pairs)
    n = len(abs_ratios)
    if n == 0:
        raise TimingContractError("no scored cells for the bootstrap")
    if n == 1:
        return _geomean_from_abs_ratios(abs_ratios)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(resamples):
        sample = [abs_ratios[rng.randrange(n)] for _ in range(n)]
        stats.append(_geomean_from_abs_ratios(sample))
    stats.sort()
    alpha = 1.0 - confidence
    upper_idx = min(
        resamples - 1,
        int(math.ceil((1.0 - alpha / 2.0) * resamples)) - 1,
    )
    return stats[upper_idx]


def score_cycle_metrics(
    pairs: PairSequence,
    *,
    method: str = PINNED_CYCLE_ERROR_METHOD,
    resamples: int,
    confidence: float,
    seed: int = DEFAULT_RESAMPLE_SEED,
) -> dict:
    """Score a cohort of ``(predicted_cycles, true_cycles)`` pairs.

    Only the pinned ``log_ratio`` method is admitted. A caller that requests
    the bounded-fraction cache method (``abs(pred - true) / max(true, eps)``)
    is refused, because that bounded-fraction formula does not transfer to the
    multiplicative cycle target.
    """
    if method != PINNED_CYCLE_ERROR_METHOD:
        raise TimingContractError(
            f"cycle error must use the pinned {PINNED_CYCLE_ERROR_METHOD!r} "
            "formula (r_i = ln(p_i / t_i)); the bounded-fraction cache formula "
            "abs(pred - true) / max(true, epsilon) is forbidden for the cycle "
            f"target (requested method={method!r})"
        )
    return {
        "n_cells": len(list(pairs)),
        "geomean_relative_error": geomean_relative_error(pairs),
        "median_abs_log_relative_error": median_abs_log_relative_error(pairs),
        "bootstrap_geomean_upper_bound": bootstrap_geomean_upper_bound(
            pairs, resamples=resamples, confidence=confidence, seed=seed
        ),
    }


def evaluate_publication_gate(doc: dict, scores: dict) -> dict:
    """Compare a ``score_cycle_metrics`` result against the contract's
    publication-gate thresholds.

    Returns a per-criterion verdict plus an overall ``passes`` boolean. A
    cohort below the minimum scored-cell floor never passes regardless of the
    metric values.
    """
    gate = publication_gate(doc)
    n_ok = scores["n_cells"] >= gate["min_scored_cells"]
    geomean_ok = scores["geomean_relative_error"] <= gate["geomean_relative_error_max"]
    median_ok = (
        scores["median_abs_log_relative_error"]
        <= gate["median_abs_log_relative_error_max"]
    )
    bootstrap_ok = (
        scores["bootstrap_geomean_upper_bound"]
        <= gate["bootstrap_geomean_upper_bound_max"]
    )
    return {
        "min_scored_cells_met": n_ok,
        "geomean_relative_error_met": geomean_ok,
        "median_abs_log_relative_error_met": median_ok,
        "bootstrap_geomean_upper_bound_met": bootstrap_ok,
        "passes": bool(n_ok and geomean_ok and median_ok and bootstrap_ok),
    }
