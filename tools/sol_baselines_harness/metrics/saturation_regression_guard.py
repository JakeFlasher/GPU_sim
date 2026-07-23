"""Saturation regression guard (deployed scope).

Two surfaces:

  * **Per-cell saturation diagnostic**: `evaluate_saturation_diagnostic(
    predicted_hit_rate)` returns `simulator_saturated = True` for any
    cell whose predicted hit rate is strictly above
    `SATURATION_THRESHOLD` (0.999). Downstream callers (bundle
    factory, report tables, the registered `cache_prediction`
    surface) carry this flag through to the per-cell manifest entry
    so a reviewer / report consumer can spot the saturation regime
    at a glance.

  * **5/5 activation-saturation block**:
    `evaluate_activation_saturation_block(predictions, labels)`
    encodes the failure mode the sibling project hit at 32.89 %
    geomean rel-err: every activation cell (LayerNorm / RMSNorm /
    SiLU / Softmax / well-behaved GEMM) emits a predicted L2 hit
    rate at 1.0000 while the NCU label disagrees. The block returns
    a frozen `ActivationBlockResult` that fails closed when ALL 5
    cells predict above `BLOCK_PREDICTION_SATURATION_FLOOR` (0.99)
    AND at least one of those cells has an NCU label that is NOT
    also strictly above `BLOCK_LABEL_SATURATION_FLOOR` (0.99). The
    plan's same-cell exception is strict: a predicted-saturated
    cell agrees with its label ONLY when the label is also
    strictly `> 0.99` on the same cell — labels at exactly `0.99`,
    or anywhere below, are disagreements. When fewer than 5 cells
    saturate, the block passes irrespective of labels (no quorum);
    when 5/5 cells saturate AND every label is strictly above
    0.99, the block passes (genuine saturation regime).

The 5 activation cells are pinned to
`ACTIVATION_BLOCK_CELL_NAMES`, a frozenset drawn from the
deployed calibration-corpus `predictor_friendly` cohort B. The
guard enforces canonical-set membership: callers must pass
exactly the five pinned cells (in any order). Any missing or
extra cell, or a duplicate pinned cell name that hides a
missing one, raises `saturation_guard_input_invalid` BEFORE
the block computation runs.

Imports only stdlib. No calibration-subtree imports; the NCU
quarantine boundary is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


# Per-cell saturation flag flips strictly above this threshold.
# Plan-defined positive-test value: "predicted hit rate above
# 0.999".
SATURATION_THRESHOLD: float = 0.999

# Floor used by the 5/5 activation block when counting how many
# cells saturated. Plan-defined positive-test value: "predicted
# L2 hit rate above 0.99". Note the difference from the
# per-cell flag's
# 0.999 floor — the block is a coarser screen looking for
# many-cell saturation; the per-cell flag is a finer diagnostic.
BLOCK_PREDICTION_SATURATION_FLOOR: float = 0.99

# Strictly above this label floor, the cell's NCU label is
# treated as genuinely saturated. The 5/5 block accepts 5/5
# prediction saturation iff EVERY cell's label is ALSO strictly
# above this floor; a label at exactly 0.99 (or anywhere below)
# is a disagreement. The strict-`>` boundary mirrors the
# per-cell `simulator_saturated` flag's strict-`>`
# `SATURATION_THRESHOLD` comparison.
BLOCK_LABEL_SATURATION_FLOOR: float = 0.99

# Plan-required activation block size. The regression-guard
# contract is "5 activation cells"; the block fails closed when
# the caller passes a count other than 5.
ACTIVATION_BLOCK_SIZE: int = 5

# Plan-pinned activation block cells. These are the 5 cells from
# the deployed predictor design-coverage partition that cover the
# named activation primitives independently: one LayerNorm BF16
# microbench (`calib_layernorm_fwd_bf16`, cohort A), one RMSNorm
# BF16 cell (`problem_001_fused_add_rmsnorm_h2048`, cohort B),
# one well-behaved GEMM cell
# (`problem_004_gemm_n128_k2048`, cohort B), one Softmax BF16
# cell (`problem_046_attention_softmax_with_softcapping_and_dropout`,
# cohort B), and one SiLU BF16 cell
# (`problem_074_fused_gated_mlp_silu`, cohort B). The set spans
# both cohort A and cohort B because the LayerNorm primitive
# lives only as a microbench in the partition; the
# saturation-regression check is a structural verdict on
# per-cell predicted hit rate vs measured NCU label and is not
# an averaged headline aggregate, so it is exempt from the
# cohort-A/cohort-B separate-headlines policy. The frozenset is
# exported so report tables and the bundle factory can label
# rows under the same canonical names.
ACTIVATION_BLOCK_CELL_NAMES: frozenset[str] = frozenset({
    "calib_layernorm_fwd_bf16",
    "problem_001_fused_add_rmsnorm_h2048",
    "problem_004_gemm_n128_k2048",
    "problem_046_attention_softmax_with_softcapping_and_dropout",
    "problem_074_fused_gated_mlp_silu",
})


SATURATION_REGRESSION_GUARD_TYPED_REASONS: frozenset[str] = frozenset({
    "saturation_guard_input_invalid",
    "activation_block_size_mismatch",
})


class SaturationRegressionGuardError(Exception):
    """Raised on saturation-guard contract violations.

    Carries the typed-reason string so downstream callers (the
    bundle factory, the report tables) can compose typed-
    unavailable bundles or report-table errors without parsing
    message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class SaturationDiagnostic:
    """Per-cell saturation diagnostic.

    `predicted_hit_rate` is the calibrated predicted L1 OR L2
    hit-rate fraction in `[0, 1]`. `simulator_saturated` flips
    strictly above `SATURATION_THRESHOLD` (0.999), the plan-
    defined positive-test value for the saturation regression
    guard.
    """

    cell_name: str
    cache_level: str
    predicted_hit_rate: float
    simulator_saturated: bool


@dataclass(frozen=True)
class ActivationBlockResult:
    """5/5 activation-saturation block result.

    `passed = False` iff ALL `ACTIVATION_BLOCK_SIZE` cells
    predict above `BLOCK_PREDICTION_SATURATION_FLOOR` AND at
    least one of those cells' labels is NOT strictly above
    `BLOCK_LABEL_SATURATION_FLOOR` (the same-cell saturation
    exception is strict: label > 0.99 is required for
    agreement; label == 0.99 is a disagreement). Otherwise
    `passed = True`.

    `disagreement_cells` is the ordered tuple of cell names
    whose prediction crossed the saturation floor AND whose
    label is at or below `BLOCK_LABEL_SATURATION_FLOOR`. The
    order mirrors the caller-provided input order so a report
    consumer can map disagreements back to column position
    deterministically. The tuple is empty on a passing block.
    """

    passed: bool
    predicted_saturated_count: int
    label_saturated_count: int
    disagreement_cells: tuple[str, ...]
    prediction_saturation_floor: float
    label_saturation_floor: float


def _validate_unit_interval(
    name: str, value: float, *, where: str,
) -> float:
    if not isinstance(value, (int, float)):
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"{name} must be a number; got "
            f"{type(value).__name__} ({where})",
        )
    fvalue = float(value)
    if math.isnan(fvalue) or math.isinf(fvalue):
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"{name} is non-finite ({fvalue!r}) ({where})",
        )
    if fvalue < 0.0 or fvalue > 1.0:
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"{name} is out of [0, 1] ({fvalue!r}) ({where}); "
            f"hit-rate fractions live in [0, 1]",
        )
    return fvalue


def evaluate_saturation_diagnostic(
    *,
    cell_name: str,
    cache_level: str,
    predicted_hit_rate: float,
) -> SaturationDiagnostic:
    """Build a per-cell `SaturationDiagnostic`.

    `simulator_saturated` is `True` iff `predicted_hit_rate >
    SATURATION_THRESHOLD` (strictly). Both `predicted_hit_rate
    == 0.999` and small noise just above (e.g. `0.9991`) are
    on opposite sides of the boundary: the former returns
    `False`, the latter returns `True`.
    """
    if not isinstance(cell_name, str) or not cell_name.strip():
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            "cell_name must be a non-empty string",
        )
    if cache_level not in ("l1", "l2"):
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"cache_level must be 'l1' or 'l2'; got "
            f"{cache_level!r}",
        )
    p = _validate_unit_interval(
        "predicted_hit_rate", predicted_hit_rate,
        where=f"cell={cell_name!r}, cache={cache_level!r}",
    )
    return SaturationDiagnostic(
        cell_name=cell_name,
        cache_level=cache_level,
        predicted_hit_rate=p,
        simulator_saturated=p > SATURATION_THRESHOLD,
    )


def evaluate_activation_saturation_block(
    *,
    cell_names: Sequence[str],
    predicted_hit_rates: Sequence[float],
    ncu_labels: Sequence[float],
) -> ActivationBlockResult:
    """Evaluate the 5/5 activation-saturation block.

    Inputs:
      cell_names: ordered tuple/list of cell names; must have
        exactly `ACTIVATION_BLOCK_SIZE` entries AND match
        `ACTIVATION_BLOCK_CELL_NAMES` as a set (any order).
      predicted_hit_rates: predicted L2 hit-rate fractions in
        `[0, 1]`, same length and order as `cell_names`.
      ncu_labels: NCU read-only sector hit-rate labels in
        `[0, 1]`, same length and order as `cell_names`.

    Returns an `ActivationBlockResult`:
      passed: False iff ALL 5 cells predict above
        `BLOCK_PREDICTION_SATURATION_FLOOR` AND at least one
        of those 5 cells' labels is NOT strictly above
        `BLOCK_LABEL_SATURATION_FLOOR` (i.e., label <= 0.99).
        Otherwise True. The same-cell exception is strict:
        a label of exactly 0.99 is a disagreement.

    Typed errors:
      `saturation_guard_input_invalid`: NaN / inf / out-of-
        range entries; empty cell name; mismatched sequence
        lengths (when the lengths differ from each other);
        duplicate pinned cell names; cell-set not equal to
        the canonical `ACTIVATION_BLOCK_CELL_NAMES`.
      `activation_block_size_mismatch`: when the caller passes
        a count other than `ACTIVATION_BLOCK_SIZE` cells.
    """
    n_cells = len(cell_names)
    n_pred = len(predicted_hit_rates)
    n_label = len(ncu_labels)
    if n_cells != n_pred or n_cells != n_label:
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"cell_names, predicted_hit_rates, and ncu_labels "
            f"must have the same length; got {n_cells}, "
            f"{n_pred}, {n_label}",
        )
    if n_cells != ACTIVATION_BLOCK_SIZE:
        raise SaturationRegressionGuardError(
            "activation_block_size_mismatch",
            f"activation block requires exactly "
            f"{ACTIVATION_BLOCK_SIZE} cells; got {n_cells}",
        )

    # Validate every entry up front so a single non-finite or
    # out-of-range value fails loudly rather than poisoning a
    # subsequent comparison.
    cleaned_names: list[str] = []
    cleaned_preds: list[float] = []
    cleaned_labels: list[float] = []
    for name, pred, label in zip(cell_names, predicted_hit_rates, ncu_labels):
        if not isinstance(name, str) or not name.strip():
            raise SaturationRegressionGuardError(
                "saturation_guard_input_invalid",
                f"cell_name must be a non-empty string; got "
                f"{name!r}",
            )
        cleaned_names.append(name)
        cleaned_preds.append(_validate_unit_interval(
            "predicted_hit_rate", pred, where=f"cell={name!r}",
        ))
        cleaned_labels.append(_validate_unit_interval(
            "ncu_label", label, where=f"cell={name!r}",
        ))

    # Reject duplicates BEFORE the frozenset comparison so a
    # duplicate pinned name that masks a missing canonical cell
    # produces a duplicate-specific diagnostic instead of being
    # silently absorbed into the set-equality check.
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in cleaned_names:
        if name in seen:
            duplicates.append(name)
        else:
            seen.add(name)
    if duplicates:
        # Use sorted order so the error message is
        # deterministic across runs with different caller
        # orderings.
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"cell_names contains duplicate entries: "
            f"{sorted(set(duplicates))!r}; the activation "
            f"block requires the five canonical names exactly",
        )

    # Enforce canonical-set membership. A missing or extra
    # cell points to a downstream wiring mistake; surface
    # missing AND extra cells in the error message so the
    # caller can fix both without re-running.
    actual_set = frozenset(cleaned_names)
    if actual_set != ACTIVATION_BLOCK_CELL_NAMES:
        missing = sorted(ACTIVATION_BLOCK_CELL_NAMES - actual_set)
        extra = sorted(actual_set - ACTIVATION_BLOCK_CELL_NAMES)
        raise SaturationRegressionGuardError(
            "saturation_guard_input_invalid",
            f"cell_names must equal the canonical activation "
            f"block (any order); missing={missing!r}, "
            f"extra={extra!r}",
        )

    predicted_saturated_count = sum(
        1 for p in cleaned_preds
        if p > BLOCK_PREDICTION_SATURATION_FLOOR
    )
    label_saturated_count = sum(
        1 for ell in cleaned_labels
        if ell > BLOCK_LABEL_SATURATION_FLOOR
    )

    # Disagreement: prediction strictly above the saturation
    # floor AND label NOT strictly above the same floor
    # (i.e., label <= 0.99). The plan's same-cell exception is
    # "label is ALSO above 0.99": a label at exactly 0.99 is a
    # disagreement under the strict-`>` boundary. The order of
    # the returned tuple mirrors the caller's input order so a
    # report consumer can map disagreements back to their
    # column position deterministically.
    disagreement_cells: list[str] = []
    for name, pred, label in zip(cleaned_names, cleaned_preds, cleaned_labels):
        if (
            pred > BLOCK_PREDICTION_SATURATION_FLOOR
            and not (label > BLOCK_LABEL_SATURATION_FLOOR)
        ):
            disagreement_cells.append(name)

    # passed = NOT (5/5 predicted saturated AND any label
    # disagrees). When fewer than 5 cells predicted saturated,
    # the block has no quorum to fail; pass. When 5/5
    # predicted saturated AND every label is strictly above
    # the saturation floor (no disagreement), pass — that is
    # the genuine-saturation regime the guard is supposed to
    # allow.
    passed = not (
        predicted_saturated_count == ACTIVATION_BLOCK_SIZE
        and len(disagreement_cells) > 0
    )

    return ActivationBlockResult(
        passed=passed,
        predicted_saturated_count=predicted_saturated_count,
        label_saturated_count=label_saturated_count,
        disagreement_cells=tuple(disagreement_cells),
        prediction_saturation_floor=BLOCK_PREDICTION_SATURATION_FLOOR,
        label_saturation_floor=BLOCK_LABEL_SATURATION_FLOOR,
    )


__all__ = [
    "ACTIVATION_BLOCK_CELL_NAMES",
    "ACTIVATION_BLOCK_SIZE",
    "ActivationBlockResult",
    "BLOCK_LABEL_SATURATION_FLOOR",
    "BLOCK_PREDICTION_SATURATION_FLOOR",
    "SATURATION_REGRESSION_GUARD_TYPED_REASONS",
    "SATURATION_THRESHOLD",
    "SaturationDiagnostic",
    "SaturationRegressionGuardError",
    "evaluate_activation_saturation_block",
    "evaluate_saturation_diagnostic",
]
