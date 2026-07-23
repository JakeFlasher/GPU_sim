"""Deterministic cohort-stratified split generator.

Replaces an earlier-3 alphabetical 80/20 split with a generator
that:

* Splits ONLY measured cells (empty input → empty split, with a
  fail-closed flag).
* Resolves each cell's bucket via the canonical predictor
  partition (so the split reflects the load-bearing source of
  truth, not the ledger's row-side bucket).
* Stratifies the split so the validation set contains at least
  one friendly AND one hostile cell when both cohorts have
  measured cells.
* Fails closed when stratification is impossible (e.g., only
  one cohort has measured cells AND `require_both_cohorts` is
  set).
* Emits a TOML compatible with the existing
  `calibration_validation_split_v1.toml` shape: `[manifest_meta]`
  + repeated `[[splits]]` rows.

This is a pure-Python generator with no NCU / harvester
dependency; it can run on any host.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


STRATIFIED_SPLIT_TYPED_REASONS: frozenset = frozenset({
    "stratified_split_no_measured_cells",
    "stratified_split_hostile_cohort_absent",
    "stratified_split_friendly_cohort_absent",
    "stratified_split_validation_too_small",
})


class StratifiedSplitError(Exception):
    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in STRATIFIED_SPLIT_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of {sorted(STRATIFIED_SPLIT_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class StratifiedSplit:
    """Result of `build_stratified_split`."""

    calibration_only: tuple
    validation_only: tuple
    cohort_friendly_calibration: tuple
    cohort_friendly_validation: tuple
    cohort_hostile_calibration: tuple
    cohort_hostile_validation: tuple
    fraction_calibration: float
    measured_count: int


def _bucket_for(cell_name: str, bucket_lookup: Mapping[str, str]) -> str:
    return bucket_lookup.get(cell_name, "predictor_hostile")


def build_stratified_split(
    measured_cells: Iterable[str],
    *,
    bucket_lookup: Mapping[str, str],
    fraction_calibration: float = 0.80,
    require_both_cohorts_in_validation: bool = True,
) -> StratifiedSplit:
    """Produce a deterministic cohort-stratified split.

    Algorithm:
      * Sort measured cells alphabetically.
      * For each cohort, take the first `fraction_calibration`
        × cohort-size cells for calibration; the rest for
        validation.
      * Merge cohorts into the final split lists.

    Fail-closed:
      * Empty measured_cells →
        `stratified_split_no_measured_cells`.
      * `require_both_cohorts_in_validation` AND the validation
        set lacks one cohort → typed error naming the absent
        cohort.

    The result is deterministic: the same inputs always produce
    the same split.
    """
    cells = sorted(set(measured_cells))
    if not cells:
        raise StratifiedSplitError(
            "stratified_split_no_measured_cells",
            "no measured cells supplied; cannot stratify an empty input",
        )
    friendly = [
        c for c in cells
        if _bucket_for(c, bucket_lookup) == "predictor_friendly"
    ]
    hostile = [
        c for c in cells
        if _bucket_for(c, bucket_lookup) == "predictor_hostile"
    ]

    def split_cohort(cohort: list) -> tuple:
        n_cal = max(1, int(round(len(cohort) * fraction_calibration)))
        if n_cal >= len(cohort) and len(cohort) > 1:
            n_cal = len(cohort) - 1
        return tuple(cohort[:n_cal]), tuple(cohort[n_cal:])

    f_cal, f_val = split_cohort(friendly) if friendly else ((), ())
    h_cal, h_val = split_cohort(hostile) if hostile else ((), ())

    if require_both_cohorts_in_validation:
        if friendly and not f_val:
            raise StratifiedSplitError(
                "stratified_split_validation_too_small",
                f"friendly cohort has {len(friendly)} cells but stratified "
                f"split produced an empty friendly_validation; increase "
                f"cohort size or lower fraction_calibration",
            )
        if hostile and not h_val:
            raise StratifiedSplitError(
                "stratified_split_validation_too_small",
                f"hostile cohort has {len(hostile)} cells but stratified "
                f"split produced an empty hostile_validation; increase "
                f"cohort size or lower fraction_calibration",
            )
        # If a cohort entirely absent, the validation set CAN'T
        # carry both cohorts; surface the typed-fail-closed.
        if not friendly:
            raise StratifiedSplitError(
                "stratified_split_friendly_cohort_absent",
                "no predictor_friendly cells in measured input; "
                "validation cannot represent both cohorts. Set "
                "require_both_cohorts_in_validation=False to "
                "override.",
            )
        if not hostile:
            raise StratifiedSplitError(
                "stratified_split_hostile_cohort_absent",
                "no predictor_hostile cells in measured input; "
                "validation cannot represent both cohorts. Set "
                "require_both_cohorts_in_validation=False to "
                "override.",
            )

    calibration_only = tuple(sorted(f_cal + h_cal))
    validation_only = tuple(sorted(f_val + h_val))
    return StratifiedSplit(
        calibration_only=calibration_only,
        validation_only=validation_only,
        cohort_friendly_calibration=f_cal,
        cohort_friendly_validation=f_val,
        cohort_hostile_calibration=h_cal,
        cohort_hostile_validation=h_val,
        fraction_calibration=fraction_calibration,
        measured_count=len(cells),
    )


def emit_split_toml(split: StratifiedSplit, host_arch: str) -> str:
    """Serialize a StratifiedSplit to the TOML shape compatible
    with the existing calibration_validation_split files.
    """
    now = datetime.now(timezone.utc).isoformat()
    lines: list = [
        '# sm_120 calibration / validation split.',
        '# Cohort-balanced: friendly + hostile represented in BOTH calibration',
        '# and validation when both cohorts have measured cells. Produced by',
        '# stratified_split.build_stratified_split.',
        '',
        '[manifest_meta]',
        'schema_version = 1',
        f'created_at_utc = "{now}"',
        f'updated_at_utc = "{now}"',
        f'host_arch = "{host_arch}"',
        f'calibration_count = {len(split.calibration_only)}',
        f'validation_count = {len(split.validation_only)}',
        f'fraction_calibration = {split.fraction_calibration!r}',
        f'measured_count = {split.measured_count}',
        'generator = "stratified_split.build_stratified_split"',
        'stratification_policy = "alphabetical per-cohort split; friendly + hostile both represented in calibration AND validation"',
        f'cohort_friendly_calibration_count = {len(split.cohort_friendly_calibration)}',
        f'cohort_friendly_validation_count = {len(split.cohort_friendly_validation)}',
        f'cohort_hostile_calibration_count = {len(split.cohort_hostile_calibration)}',
        f'cohort_hostile_validation_count = {len(split.cohort_hostile_validation)}',
        '',
    ]
    for cell in split.calibration_only:
        lines += ['[[splits]]', f'slug = "{cell}"', 'split = "calibration_only"', '']
    for cell in split.validation_only:
        lines += ['[[splits]]', f'slug = "{cell}"', 'split = "validation_only"', '']
    return '\n'.join(lines) + '\n'


__all__ = [
    "STRATIFIED_SPLIT_TYPED_REASONS",
    "StratifiedSplit",
    "StratifiedSplitError",
    "build_stratified_split",
    "emit_split_toml",
]
