"""Typed-unavailable bundle factory + reasons enum +
coverage-floor evaluator (deployed scope).

When the predictor cannot honestly score a cell — because
the cell is predeclared typed-unavailable, the trace
budget truncated, the cache geometry is unresolved, the
admitted-cell count fell below the smoke floor, or any
other contract-pinned failure mode — it emits a structured
typed-unavailable bundle instead of a scalar hit-rate
estimate. The bundle carries a non-empty list of reasons
drawn from a fixed enum; downstream report tables, bundle
schemas, and comparison pairing layers consume the
reasons array directly.

Public surface:

  * `TYPED_UNAVAILABLE_RESULT_KIND` — the fixed string
    `predictor_typed_unavailable` returned as the
    bundle's `result_kind`.
  * `PREDICTOR_ABSTENTION_REASONS` — frozenset of every
    abstention reason a bundle may carry. The enum is
    closed: any reason not in this set raises typed
    `predictor_reason_invalid` at validation time.
  * `COVERAGE_FLOOR_SMOKE`, `COVERAGE_FLOOR_PUBLICATION`,
    `COVERAGE_FLOOR_ASPIRATIONAL` — admitted-cell-count
    thresholds (5 / 10 / 20).
  * `CoverageTier` — string-constant tier vocabulary
    (`aspirational`, `publication`, `smoke`,
    `below_smoke`).
  * `CoverageTierEvaluation` — frozen dataclass with
    `tier`, `admitted_cell_count`,
    `scalar_headline_allowed`, `coverage_reason`.
  * `TypedUnavailableBundle` — frozen dataclass:
    `cell_name`, `result_kind`, `reasons`, optional
    `provenance` mapping.
  * `TypedUnavailableBundleError(typed_reason,
    message)` with typed reasons:
    `predictor_reason_array_missing`,
    `predictor_reason_invalid`,
    `predictor_bundle_result_kind_invalid`,
    `predictor_bundle_cell_name_missing`.
  * `make_typed_unavailable_bundle(cell_name, *,
    extra_reasons=(), provenance=None)` — factory.
  * `validate_typed_unavailable_bundle(bundle)` —
    validator.
  * `evaluate_coverage_tier(admitted_cell_count)` —
    coverage-floor gate.

Importing this module does NOT invoke any external
process. It is pure data + pure functions and imports
only stdlib +
`baselines_harness.metrics.predictor_design_coverage`.
The deployed predictor / bundle / comparison layers can
import it freely; the NCU quarantine boundary is
preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from baselines_harness.metrics.predictor_design_coverage import (
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    PREDICTOR_TYPED_UNAVAILABLE_REASONS,
    PredictorDesignCoverageError,
    lookup_predictor_design_coverage,
)


TYPED_UNAVAILABLE_RESULT_KIND: str = "predictor_typed_unavailable"


# Coverage thresholds: how many admitted cells the
# scoreboard needs before each report-tier headline is
# allowed. Below the smoke floor, NO scalar headline is
# emitted — the scoreboard returns a typed-unavailable
# placeholder instead.
COVERAGE_FLOOR_SMOKE: int = 5
COVERAGE_FLOOR_PUBLICATION: int = 10
COVERAGE_FLOOR_ASPIRATIONAL: int = 20


class CoverageTier:
    BELOW_SMOKE: str = "below_smoke"
    SMOKE: str = "smoke"
    PUBLICATION: str = "publication"
    ASPIRATIONAL: str = "aspirational"


# Closed enum of every reason a typed-unavailable bundle
# may carry. Includes:
#   * 4 partition typed-unavailable subclasses (from
#     the deployed predictor-design-coverage module).
#   * 1 trace-budget reason matching the existing
#     producer/predictor vocabulary in
#     `cache_predictor.py:142`
#     (`TYPED_UNAVAILABLE_REASON_TRACE_BUDGET_EXCEEDED`).
#   * 1 cache-geometry reason matching the existing
#     `cache_geometry.py:89` vocabulary.
#   * 3 coverage-floor reasons (one per tier suppression
#     boundary).
# Any future abstention surface that wants to compose a
# bundle MUST extend this set or the validator rejects
# the off-enum reason.
PREDICTOR_ABSTENTION_REASONS: frozenset[str] = frozenset({
    # Partition typed-unavailable subclasses
    *PREDICTOR_TYPED_UNAVAILABLE_REASONS,
    # Trace-state reason (producer-side vocabulary)
    "trace_budget_exceeded",
    # Cache-geometry reason
    "cache_geometry_unresolved",
    # Coverage-floor suppression reasons
    "coverage_below_smoke_floor",
    "coverage_below_publication_floor",
    "coverage_below_aspirational_floor",
    # Per-launch predictor-body abstention reasons:
    #   `policy_uncalibrated` fires when any access record's
    #     `cache_access_policy.route_access` returns
    #     `policy_uncalibrated=True` (unknown op_kind / unknown
    #     cache modifier / unrecognized address space). The
    #     whole launch routes to typed-unavailable since the
    #     simulator cannot be trusted on an unrecognized policy.
    #   `no_admissible_accesses` fires when the L1 OR L2
    #     simulator has zero admissible records (every access
    #     was routed away from that layer). The predictor
    #     cannot honestly score a layer with zero accesses.
    "policy_uncalibrated",
    "no_admissible_accesses",
    # Warmed-sequence helper reasons (call-level abstentions
    # the per-frame predictor body cannot raise):
    #   `warmup_count_out_of_range` fires when the caller passes
    #     a `warmup_count` outside `[0, len(frames))`. The
    #     warmed-sequence helper cannot honour NCU's
    #     `--launch-skip N --launch-count 1` semantics if the
    #     requested warmup-then-measure split does not fit the
    #     captured frames.
    #   `frames_sequence_empty` fires when the caller passes a
    #     zero-length `frames` sequence. The helper has no
    #     frame to measure.
    "warmup_count_out_of_range",
    "frames_sequence_empty",
    # `measured_cache_initial_state_invalid` fires when the
    #   warmed-sequence helper is invoked with a
    #   `measured_cache_initial_state` value outside the
    #   recognized `{warm, cold}` surface (including the
    #   `unknown` sentinel from the deployed warmup_protocol
    #   module — the prediction caller MUST commit to one or
    #   the other at scoring time; an unknown state cannot be
    #   honestly scored against either NCU `--replay-mode
    #   kernel` (cold) or `--replay-mode application` (warm)).
    "measured_cache_initial_state_invalid",
    # Trace-denominator-consistency reason. Fires
    # when the Tier-2 instrumentation captures
    # materially fewer L2 read sectors per launch
    # than the NCU label denominator the cell was
    # calibrated against (e.g., Tier-2 sees 8
    # sectors per launch while NCU's label is over
    # ~600 read sectors). The downstream
    # publication discipline routes such cells to
    # typed-unavailable rather than calibrating
    # over the denominator gap — calibration over
    # a too-narrow trace surface can only absorb a
    # constant bias, not the real per-cell variance
    # in cache behaviour.
    "trace_denominator_mismatch",
    # Tiered-gate publication-discipline reason.
    # Fires when the calibrated tiered gate's
    # `overall_passed` flag is False — at least
    # one cohort × cache-level bucket missed its
    # ±5 pp (friendly) or ±15 pp (hostile)
    # threshold. Publishing scalar
    # `cache_prediction` rows under a failing
    # tiered gate would contradict the tiered-
    # accuracy-gate fail-closed contract.
    "tiered_gate_overall_failed",
})


TYPED_UNAVAILABLE_BUNDLE_TYPED_REASONS: frozenset[str] = frozenset({
    "predictor_reason_array_missing",
    "predictor_reason_invalid",
    "predictor_bundle_result_kind_invalid",
    "predictor_bundle_cell_name_missing",
})


class TypedUnavailableBundleError(Exception):
    """Raised by the bundle factory and validator on
    contract violations. Carries the typed-reason string
    so downstream callers (the predictor body, the bundle
    schema, the comparison runner) can compose
    typed-unavailable bundles or report-table errors
    without parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class TypedUnavailableBundle:
    cell_name: str
    result_kind: str
    reasons: tuple[str, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageTierEvaluation:
    tier: str
    admitted_cell_count: int
    scalar_headline_allowed: bool
    coverage_reason: str


def make_typed_unavailable_bundle(
    cell_name: str,
    *,
    extra_reasons: Iterable[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> TypedUnavailableBundle:
    """Compose a typed-unavailable bundle for `cell_name`.

    The bundle's reasons array is built in this order:
      1. If `cell_name` is in the predictor design-
         coverage partition AND its bucket is
         `predictor_typed_unavailable`, the partition's
         `typed_unavailable_reason` is the first entry.
         A friendly or hostile cell contributes nothing
         from the partition lookup (the caller must
         supply at least one reason via `extra_reasons`).
      2. Each value in `extra_reasons` is appended in
         order. Duplicates are removed (first-seen wins)
         so callers can pass overlapping reason sources
         without producing redundant entries.

    Cells outside the partition propagate
    `PredictorDesignCoverageError` (typed reason
    `cell_missing_cohort_partition`) unchanged. Downstream
    callers can catch and reraise, or surface the typed
    reason directly.

    The result is validated through
    `validate_typed_unavailable_bundle` before return.

    Typed errors raised:
      `predictor_bundle_cell_name_missing`: empty cell name.
      `predictor_reason_array_missing`: no reasons after
        composition.
      `predictor_reason_invalid`: a reason is not in
        `PREDICTOR_ABSTENTION_REASONS`.
    """
    if not cell_name or not cell_name.strip():
        raise TypedUnavailableBundleError(
            "predictor_bundle_cell_name_missing",
            "cell_name argument is empty",
        )
    collected: list[str] = []
    seen: set[str] = set()
    try:
        entry = lookup_predictor_design_coverage(cell_name)
    except PredictorDesignCoverageError:
        # Missing-cell errors propagate to the caller as
        # the typed `cell_missing_cohort_partition` reason
        # the partition layer already raises. The caller
        # decides whether to surface that or recover.
        raise
    if entry.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        partition_reason = entry.typed_unavailable_reason
        if partition_reason and partition_reason not in seen:
            collected.append(partition_reason)
            seen.add(partition_reason)
    for reason in extra_reasons:
        if reason in seen:
            continue
        collected.append(reason)
        seen.add(reason)
    bundle = TypedUnavailableBundle(
        cell_name=cell_name,
        result_kind=TYPED_UNAVAILABLE_RESULT_KIND,
        reasons=tuple(collected),
        provenance=dict(provenance) if provenance else {},
    )
    validate_typed_unavailable_bundle(bundle)
    return bundle


def validate_typed_unavailable_bundle(
    bundle: TypedUnavailableBundle,
) -> None:
    """Raise typed errors on contract violations:
      * `predictor_bundle_result_kind_invalid` — the
        `result_kind` is not
        `TYPED_UNAVAILABLE_RESULT_KIND`.
      * `predictor_bundle_cell_name_missing` — the
        `cell_name` is empty.
      * `predictor_reason_array_missing` — the `reasons`
        tuple is empty.
      * `predictor_reason_invalid` — any reason is not in
        `PREDICTOR_ABSTENTION_REASONS`.

    Returns silently when the bundle is clean.
    """
    if not isinstance(bundle, TypedUnavailableBundle):
        raise TypedUnavailableBundleError(
            "predictor_bundle_result_kind_invalid",
            f"argument is not a TypedUnavailableBundle; "
            f"got {type(bundle).__name__}",
        )
    if bundle.result_kind != TYPED_UNAVAILABLE_RESULT_KIND:
        raise TypedUnavailableBundleError(
            "predictor_bundle_result_kind_invalid",
            f"bundle.result_kind must be "
            f"{TYPED_UNAVAILABLE_RESULT_KIND!r}; got "
            f"{bundle.result_kind!r}",
        )
    if not bundle.cell_name or not bundle.cell_name.strip():
        raise TypedUnavailableBundleError(
            "predictor_bundle_cell_name_missing",
            "bundle.cell_name is empty",
        )
    if not bundle.reasons:
        raise TypedUnavailableBundleError(
            "predictor_reason_array_missing",
            f"bundle for cell {bundle.cell_name!r} "
            f"claims abstention but carries no reasons",
        )
    for reason in bundle.reasons:
        if reason not in PREDICTOR_ABSTENTION_REASONS:
            raise TypedUnavailableBundleError(
                "predictor_reason_invalid",
                f"bundle for cell {bundle.cell_name!r} "
                f"carries reason {reason!r}; expected one "
                f"of "
                f"{sorted(PREDICTOR_ABSTENTION_REASONS)}",
            )


def evaluate_coverage_tier(
    admitted_cell_count: int,
) -> CoverageTierEvaluation:
    """Map an admitted-cell count to a coverage tier and
    decide whether the scoreboard may emit a scalar
    headline.

    Tier thresholds:
      * `admitted_count < 5` → `below_smoke`,
        `scalar_headline_allowed=False`, coverage_reason
        `coverage_below_smoke_floor`.
      * `5 <= admitted_count < 10` → `smoke`,
        `scalar_headline_allowed=True`, coverage_reason
        empty.
      * `10 <= admitted_count < 20` → `publication`,
        `scalar_headline_allowed=True`, coverage_reason
        empty.
      * `admitted_count >= 20` → `aspirational`,
        `scalar_headline_allowed=True`, coverage_reason
        empty.

    Negative admitted counts collapse to `below_smoke`.
    Callers that need stricter gates (e.g. a
    publication-or-better report) read the `tier` field
    and add their own
    `coverage_below_publication_floor` /
    `coverage_below_aspirational_floor` reason to a
    bundle.
    """
    count = int(admitted_cell_count)
    if count < COVERAGE_FLOOR_SMOKE:
        return CoverageTierEvaluation(
            tier=CoverageTier.BELOW_SMOKE,
            admitted_cell_count=count,
            scalar_headline_allowed=False,
            coverage_reason="coverage_below_smoke_floor",
        )
    if count < COVERAGE_FLOOR_PUBLICATION:
        return CoverageTierEvaluation(
            tier=CoverageTier.SMOKE,
            admitted_cell_count=count,
            scalar_headline_allowed=True,
            coverage_reason="",
        )
    if count < COVERAGE_FLOOR_ASPIRATIONAL:
        return CoverageTierEvaluation(
            tier=CoverageTier.PUBLICATION,
            admitted_cell_count=count,
            scalar_headline_allowed=True,
            coverage_reason="",
        )
    return CoverageTierEvaluation(
        tier=CoverageTier.ASPIRATIONAL,
        admitted_cell_count=count,
        scalar_headline_allowed=True,
        coverage_reason="",
    )


__all__ = [
    "COVERAGE_FLOOR_ASPIRATIONAL",
    "COVERAGE_FLOOR_PUBLICATION",
    "COVERAGE_FLOOR_SMOKE",
    "CoverageTier",
    "CoverageTierEvaluation",
    "PREDICTOR_ABSTENTION_REASONS",
    "TYPED_UNAVAILABLE_BUNDLE_TYPED_REASONS",
    "TYPED_UNAVAILABLE_RESULT_KIND",
    "TypedUnavailableBundle",
    "TypedUnavailableBundleError",
    "evaluate_coverage_tier",
    "make_typed_unavailable_bundle",
    "validate_typed_unavailable_bundle",
]
