"""Publication-grade cohort gate library for the sm_86 submission track.

The headline runner consumes a cohort manifest validated by
``validate_cohort_manifest.py``. This module turns that validated
manifest into a set of typed verdicts that bind to the
submission-track plan's three publication gates:

  - **Family-bucket coverage**: every workload family is represented
    in the admitted cohort at the named minima
    (DL-primitives >= 4 slugs, GEMM-family >= 4 slugs,
    attention-conv >= 2 slugs, memory-stress >= 3 slugs,
    irregular-hostile >= 4 slugs, vendor-baselines >= 1 slug).
    Also enforces family-balance-across-splits: every admitted family
    bucket must appear in at least 2 of the 3 splits
    (calibration / validation / holdout). This is the
    reviewer-defensible cohort breadth bar from the plan's positive-test set.
  - **Depth-first coverage**: at least 6 unique slugs carry at least
    3 distinct shape_point values each. This is the depth-first
    invariant from the plan that prevents a cohort built entirely
    from one-shape-per-slug filler cells.
  - **Contract floor** (the project's broader sanity gate): the
    admitted cohort meets ``coverage_gate.publication_grade_floor_per_arch``
    (10 cells under the base contract). Failing this gate stops the
    headline pipeline with the typed diagnostic
    ``cohort_below_contract_floor``.
  - **Submission gate** (the overlay's narrower target): the
    admitted cohort meets the overlay's
    ``cohort_paper_target_unique_slugs`` (30) AND
    ``cohort_paper_target_cells`` (31). Failing this gate emits the
    typed diagnostic ``cohort_below_paper_target`` and prevents the
    holdout-evaluation report from being marked publication-grade,
    but does NOT stop the pipeline (it tags the report
    `submission_gate_not_met` instead).
  - **Replacement policy**: a post-freeze cell swap requires a fresh
    ``sweep_id``, a documented ``exclusion_typed_reason`` from the
    closed enum, AND an append to ``replacement_history``. The
    helper enforces all three under one typed exception.

The consolidated publication gate combines family-bucket + depth-first +
admitted-cell-count + unique-slug-count + balancing (slug <= 1/3 of
any split's cells) + family-balance-across-splits under OR-semantics:
a single violation fails the consolidated gate.

Typed diagnostic literals exported from this module stay in lockstep
with the headline runner + the tests + the
``PUBLICATION_GATE_FORWARD_REFERENCE`` constant in
``cache_predictor.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from validate_cohort_manifest import (  # noqa: E402
    CohortManifest,
    CohortManifestCell,
)


# ---------------------------------------------------------------------
# Multi-surface lockstep tokens
# ---------------------------------------------------------------------

CONTRACT_FLOOR_DIAGNOSTIC = "cohort_below_contract_floor"
"""Typed diagnostic emitted by the contract-floor gate. Mirrored in
the runner's stderr output and the provenance-discipline test fixture."""

SUBMISSION_TARGET_DIAGNOSTIC = "cohort_below_paper_target"
"""Typed diagnostic emitted by the submission gate."""


# ---------------------------------------------------------------------
# Family-bucket coverage thresholds (per the plan's positive-test enum)
# ---------------------------------------------------------------------

FAMILY_BUCKET_MINIMA: Mapping[str, int] = {
    "dl_primitives": 4,
    "gemm_family": 4,
    "attention_conv": 2,
    "memory_stress": 3,
    "irregular_hostile": 4,
    "vendor_baselines": 1,
}
"""Per-bucket admitted-slug minima. The total is 18 covered slugs;
the remaining slugs needed to reach 30 may distribute freely across
the six buckets."""


# ---------------------------------------------------------------------
# Depth-first thresholds
# ---------------------------------------------------------------------

DEPTH_FIRST_REQUIRED_SLUGS = 6
"""Minimum number of unique slugs that must each carry at least
``DEPTH_FIRST_MIN_SHAPE_POINTS`` distinct shape_points."""

DEPTH_FIRST_MIN_SHAPE_POINTS = 3
"""Per-slug shape_point count required for that slug to count toward
the depth-first set."""


# ---------------------------------------------------------------------
# Balancing constraint
# ---------------------------------------------------------------------

SLUG_MAX_FRACTION_PER_SPLIT = 1.0 / 3.0
"""No single slug may contribute more than 1/3 of any split's cell
count. Prevents a multi-shape slug from dominating a split when
shape_points are unequal across slugs."""


# ---------------------------------------------------------------------
# Verdict dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyBucketGateVerdict:
    passed: bool
    per_bucket_slug_counts: Mapping[str, int]
    deficient_buckets: tuple[str, ...]
    family_balance_violations: tuple[str, ...]


@dataclass(frozen=True)
class DepthFirstGateVerdict:
    passed: bool
    slugs_meeting_depth: tuple[str, ...]
    slug_shape_point_counts: Mapping[str, int]


@dataclass(frozen=True)
class ContractFloorGateVerdict:
    passed: bool
    admitted_count: int
    required_floor: int
    typed_diagnostic: str | None


@dataclass(frozen=True)
class SubmissionGateVerdict:
    passed: bool
    unique_slugs: int
    total_cells: int
    required_unique_slugs: int
    required_cells: int
    typed_diagnostic: str | None


@dataclass(frozen=True)
class BalancingGateVerdict:
    passed: bool
    violations: tuple[tuple[str, str, int, int], ...]
    """Each violation is (split, dominant_slug, slug_cells_in_split,
    split_total_cells)."""


@dataclass(frozen=True)
class ConsolidatedPublicationGateVerdict:
    """OR-semantics consolidated verdict: passes only if every
    sub-gate passes."""

    passed: bool
    family_bucket: FamilyBucketGateVerdict
    depth_first: DepthFirstGateVerdict
    contract_floor: ContractFloorGateVerdict
    submission_gate: SubmissionGateVerdict
    balancing: BalancingGateVerdict


# ---------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------


def _admitted_cells(manifest: CohortManifest) -> tuple[CohortManifestCell, ...]:
    return tuple(
        cell for cell in manifest.cells
        if cell.admission_status == "admitted"
    )


def check_family_bucket_coverage(
    manifest: CohortManifest,
) -> FamilyBucketGateVerdict:
    """Enforce per-bucket admitted-slug minima + family-balance-across-splits.

    Per-bucket counts use UNIQUE slugs (a slug with multiple
    shape_points counts once). Family-balance asserts every admitted
    bucket appears in at least 2 of the 3 splits.
    """
    admitted = _admitted_cells(manifest)
    bucket_slugs: dict[str, set[str]] = {
        bucket: set() for bucket in FAMILY_BUCKET_MINIMA
    }
    bucket_splits: dict[str, set[str]] = {
        bucket: set() for bucket in FAMILY_BUCKET_MINIMA
    }
    for cell in admitted:
        bucket = cell.family_bucket
        if bucket in bucket_slugs:
            bucket_slugs[bucket].add(cell.slug)
            bucket_splits[bucket].add(cell.split_assignment)
    per_bucket_counts = {
        bucket: len(slugs) for bucket, slugs in bucket_slugs.items()
    }
    deficient = tuple(
        bucket
        for bucket, count in per_bucket_counts.items()
        if count < FAMILY_BUCKET_MINIMA[bucket]
    )
    balance_violations = tuple(
        bucket
        for bucket, splits in bucket_splits.items()
        if len(splits) >= 1 and len(splits) < 2
    )
    passed = not deficient and not balance_violations
    return FamilyBucketGateVerdict(
        passed=passed,
        per_bucket_slug_counts=per_bucket_counts,
        deficient_buckets=deficient,
        family_balance_violations=balance_violations,
    )


def check_depth_first_coverage(
    manifest: CohortManifest,
) -> DepthFirstGateVerdict:
    """Enforce >= DEPTH_FIRST_REQUIRED_SLUGS unique slugs that each
    carry >= DEPTH_FIRST_MIN_SHAPE_POINTS distinct shape_point values
    in the admitted set."""
    admitted = _admitted_cells(manifest)
    slug_shapes: dict[str, set[str]] = {}
    for cell in admitted:
        slug_shapes.setdefault(cell.slug, set()).add(cell.shape_point)
    slug_counts = {slug: len(shapes) for slug, shapes in slug_shapes.items()}
    slugs_meeting = tuple(
        sorted(
            slug for slug, count in slug_counts.items()
            if count >= DEPTH_FIRST_MIN_SHAPE_POINTS
        )
    )
    return DepthFirstGateVerdict(
        passed=len(slugs_meeting) >= DEPTH_FIRST_REQUIRED_SLUGS,
        slugs_meeting_depth=slugs_meeting,
        slug_shape_point_counts=slug_counts,
    )


def check_contract_floor(
    manifest: CohortManifest,
    publication_grade_floor: int,
) -> ContractFloorGateVerdict:
    """Enforce the base contract's per-arch publication-grade floor.

    Failing this gate emits the typed diagnostic
    ``cohort_below_contract_floor``; the runner is expected to stop
    the pipeline rather than continue with a sub-floor cohort.
    """
    admitted = manifest.admitted_cell_count
    passed = admitted >= publication_grade_floor
    return ContractFloorGateVerdict(
        passed=passed,
        admitted_count=admitted,
        required_floor=publication_grade_floor,
        typed_diagnostic=(
            None if passed else CONTRACT_FLOOR_DIAGNOSTIC
        ),
    )


def check_submission_gate(
    manifest: CohortManifest,
    required_unique_slugs: int,
    required_cells: int,
) -> SubmissionGateVerdict:
    """Enforce the overlay's submission target (unique_slugs +
    cells). Failing this gate emits the typed diagnostic
    ``cohort_below_paper_target`` and prevents the holdout report
    from being marked publication-grade."""
    admitted = _admitted_cells(manifest)
    unique_slugs = len({cell.slug for cell in admitted})
    total_cells = len(admitted)
    passed = (
        unique_slugs >= required_unique_slugs
        and total_cells >= required_cells
    )
    return SubmissionGateVerdict(
        passed=passed,
        unique_slugs=unique_slugs,
        total_cells=total_cells,
        required_unique_slugs=required_unique_slugs,
        required_cells=required_cells,
        typed_diagnostic=(
            None if passed else SUBMISSION_TARGET_DIAGNOSTIC
        ),
    )


def check_balancing(manifest: CohortManifest) -> BalancingGateVerdict:
    """Assert no slug contributes more than 1/3 of any split's cell
    count. Operates only on the admitted set."""
    admitted = _admitted_cells(manifest)
    split_slug_counts: dict[str, dict[str, int]] = {}
    split_totals: dict[str, int] = {}
    for cell in admitted:
        split_slug_counts.setdefault(cell.split_assignment, {})
        split_slug_counts[cell.split_assignment].setdefault(cell.slug, 0)
        split_slug_counts[cell.split_assignment][cell.slug] += 1
        split_totals.setdefault(cell.split_assignment, 0)
        split_totals[cell.split_assignment] += 1
    violations: list[tuple[str, str, int, int]] = []
    for split, slug_counts in split_slug_counts.items():
        total = split_totals[split]
        if total == 0:
            continue
        for slug, count in slug_counts.items():
            if count / total > SLUG_MAX_FRACTION_PER_SPLIT:
                violations.append((split, slug, count, total))
    return BalancingGateVerdict(
        passed=not violations,
        violations=tuple(violations),
    )


def consolidated_publication_gate(
    manifest: CohortManifest,
    *,
    publication_grade_floor: int,
    required_unique_slugs: int,
    required_cells: int,
) -> ConsolidatedPublicationGateVerdict:
    """Run all gates and return the consolidated verdict.

    OR-semantics: the consolidated gate passes only if EVERY sub-gate
    passes. A single sub-gate failure is sufficient to fail the
    consolidated gate. The fixture-driven test in
    ``test_publication_gate.py`` exercises each OR branch independently.
    """
    family_bucket = check_family_bucket_coverage(manifest)
    depth_first = check_depth_first_coverage(manifest)
    contract_floor = check_contract_floor(
        manifest, publication_grade_floor=publication_grade_floor,
    )
    submission_gate = check_submission_gate(
        manifest,
        required_unique_slugs=required_unique_slugs,
        required_cells=required_cells,
    )
    balancing = check_balancing(manifest)
    passed = (
        family_bucket.passed
        and depth_first.passed
        and contract_floor.passed
        and submission_gate.passed
        and balancing.passed
    )
    return ConsolidatedPublicationGateVerdict(
        passed=passed,
        family_bucket=family_bucket,
        depth_first=depth_first,
        contract_floor=contract_floor,
        submission_gate=submission_gate,
        balancing=balancing,
    )


# ---------------------------------------------------------------------
# Replacement-policy lock
# ---------------------------------------------------------------------


class CohortReplacementPolicyError(Exception):
    """Raised when a post-freeze cohort replacement violates the
    replacement-policy contract:
      - fresh `sweep_id` (must differ from any sweep_id seen in the
        prior manifest's replacement_history);
      - documented `exclusion_typed_reason` from the closed enum;
      - append to `replacement_history` (new entry's
        old_identity_tuple must match the prior manifest's cell being
        replaced; new_identity_tuple must match the new manifest's
        cell at the same (slug, shape_point)).
    """

    def __init__(self, *, violation: str, detail: str = ""):
        self.violation = violation
        self.detail = detail
        super().__init__(
            f"cohort replacement policy violation: {violation}"
            + (f" ({detail})" if detail else "")
        )


_VALID_EXCLUSION_REASONS = frozenset({
    "trace_budget_exceeded",
    "ncu_repeatability_fail",
    "build_fail",
    "typed_unavailable_emit",
    "trace_missing",
    "label_missing",
    "none_admitted",
})


_IDENTITY_TUPLE_FIELDS = (
    "slug",
    "shape_point",
    "arch",
    "device",
    "binary_sha256",
    "kernel_signature_sha256",
    "source_commit",
)
"""Full 7-field identity tuple. ALL fields are part of the
replacement-policy lock; a mismatch on ANY field rejects the swap."""


def _full_identity_key(identity: Mapping[str, str]) -> tuple[str, ...]:
    """Project a mapping onto the canonical 7-field identity tuple
    key. Missing fields land as empty strings so the comparison
    fails on the typed-error arm rather than silently passing."""
    return tuple(str(identity.get(field, "")) for field in _IDENTITY_TUPLE_FIELDS)


def _normalize_replacement_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize a replacement-history entry into a comparable
    representation. The mapping's keys + their stringified values
    determine equality."""
    return {str(k): entry[k] for k in entry}


def _placeholder_kernel_signature(slug: str, shape_point: str) -> str:
    """Reproduce the deterministic placeholder
    ``kernel_signature_sha256`` produced by the cohort-curation step
    (`tools/local_labels/expand_cohort_manifest_from_catalog.py`).
    Curated ``attempted_excluded`` entries use this stable
    placeholder until a live sweep substitutes the real signature.
    The helper rejects any newly admitted cell still carrying
    this placeholder so admission cannot land without the live
    trace + NCU evidence."""
    import hashlib
    seed = f"pending_kernel_signature:{slug}:{shape_point}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def check_cohort_replacement(
    prior_manifest: CohortManifest,
    new_manifest: CohortManifest,
    *,
    new_sweep_id: str,
    new_typed_reason: str,
) -> None:
    """Enforce the replacement-policy contract on a post-freeze cell
    swap. ``prior_manifest`` and ``new_manifest`` must reflect the
    cohort BEFORE and AFTER the swap. Raises
    :class:`CohortReplacementPolicyError` on any of:

      - ``new_typed_reason`` not in the closed exclusion-typed-reason
        enum;
      - ``new_sweep_id`` already present in the prior manifest's
        ``replacement_history``;
      - the new manifest's ``replacement_history`` does NOT preserve
        every prior entry in declaration order + append EXACTLY ONE
        new entry whose ``sweep_id == new_sweep_id`` AND
        ``reason == new_typed_reason``;
      - the appended entry's ``old_identity_tuple`` does not match a
        cell in the prior manifest on the FULL 7-field tuple
        (slug, shape_point, arch, device, binary_sha256,
        kernel_signature_sha256, source_commit);
      - the appended entry's ``new_identity_tuple`` does not match a
        cell in the new manifest on the same FULL 7-field tuple.

    One-for-one swap semantics (the appended row MUST describe a
    real swap, not an arbitrary mutation):

      - The appended ``old_identity_tuple`` is in prior_identities
        AND absent from new_identities.
      - The appended ``new_identity_tuple`` is absent from
        prior_identities AND in new_identities.
      - Every other identity (prior minus old) is preserved in new
        (new minus new_identity).
      - ``len(new_manifest.cells) == len(prior_manifest.cells)``,
        i.e. a one-for-one swap not an insert / delete / expansion.

    Admission-status accounting (the swap MUST preserve the
    admission ledger or expose its delta explicitly):

      - ``new_manifest.attempted_cell_count ==
        prior_manifest.attempted_cell_count ==
        len(new_manifest.cells)``: the top-level counter stays in
        lockstep with the cells list across the swap.
      - ``prior_manifest.admitted_cell_count`` matches the observed
        admitted-cell count in ``prior_manifest.cells``. A forged
        prior counter is rejected with
        ``admitted_cell_count_inconsistent_with_cells_in_prior_manifest``.
      - ``new_manifest.admitted_cell_count`` matches the observed
        admitted-cell count in ``new_manifest.cells``. A forged
        new counter is rejected with
        ``admitted_cell_count_inconsistent_with_cells_in_new_manifest``.
      - The appended row's ``old_admission_status`` field MUST
        match the prior cell's ``admission_status``; mismatch
        trips ``replacement_history_old_admission_status_mismatch``.
      - The appended row's ``new_admission_status`` field MUST
        match the new cell's ``admission_status``; mismatch
        trips ``replacement_history_new_admission_status_mismatch``.
        The ``cohort_manifest.schema.json``'s
        ``replacement_history_item`` requires both fields, so a
        row without them is rejected at validate time; the helper
        cross-references the recorded delta against the actual
        cells to catch drift between the two surfaces. Together
        these subsume the legacy
        ``replacement_admission_status_delta_unrecorded`` arm
        (status-changing swaps are now recordable, not forbidden).
      - A newly admitted cell must NOT carry the deterministic
        placeholder ``kernel_signature_sha256`` produced by
        ``tools/local_labels/expand_cohort_manifest_from_catalog.py``;
        admission requires the real signature from the live trace
        + NCU sweep. Trips
        ``replacement_admitted_cell_carries_placeholder_kernel_signature``.
      - For status-preserving swaps (both admitted-to-admitted
        AND excluded-to-excluded):
        ``new_manifest.admitted_cell_count ==
        prior_manifest.admitted_cell_count``. A forged top-level
        admitted counter on a status-preserving swap is rejected
        with ``admitted_cell_count_delta_outside_one_for_one_swap``.
        Status-changing swaps adjust the counter and the
        actual-cell lockstep above already enforces consistency.
      - Every identity preserved across the swap (i.e. in
        ``prior_identities - {old_key}``) keeps the same
        ``admission_status``. A silent per-identity flip on an
        unaffected cell is rejected with
        ``unaffected_identity_admission_status_changed``.
    """
    if new_typed_reason not in _VALID_EXCLUSION_REASONS:
        raise CohortReplacementPolicyError(
            violation="exclusion_typed_reason_outside_closed_enum",
            detail=(
                f"reason={new_typed_reason!r} not in "
                f"{sorted(_VALID_EXCLUSION_REASONS)!r}"
            ),
        )
    prior_history = list(prior_manifest.replacement_history)
    prior_sweep_ids = {
        str(entry.get("sweep_id", "")) for entry in prior_history
    }
    if new_sweep_id in prior_sweep_ids:
        raise CohortReplacementPolicyError(
            violation="sweep_id_not_rotated",
            detail=(
                f"new_sweep_id={new_sweep_id!r} already in prior "
                f"manifest replacement_history"
            ),
        )
    # The new manifest's replacement_history MUST preserve every
    # prior entry in declaration order AND append exactly one new
    # entry. Truncation, reordering, or in-place mutation of prior
    # entries is rejected.
    new_history = list(new_manifest.replacement_history)
    if len(new_history) != len(prior_history) + 1:
        raise CohortReplacementPolicyError(
            violation="replacement_history_size_mismatch",
            detail=(
                f"expected prior history length {len(prior_history)} "
                f"+ 1 appended entry; got new history length "
                f"{len(new_history)}"
            ),
        )
    for i, prior_entry in enumerate(prior_history):
        if (
            _normalize_replacement_entry(new_history[i])
            != _normalize_replacement_entry(prior_entry)
        ):
            raise CohortReplacementPolicyError(
                violation="replacement_history_prior_entry_mutated",
                detail=(
                    f"index {i}: prior entry was not preserved "
                    f"in new manifest"
                ),
            )
    appended_entry = new_history[-1]
    if (
        str(appended_entry.get("sweep_id", "")) != new_sweep_id
        or str(appended_entry.get("reason", "")) != new_typed_reason
    ):
        raise CohortReplacementPolicyError(
            violation="replacement_history_not_appended",
            detail=(
                f"appended entry sweep_id="
                f"{appended_entry.get('sweep_id')!r} / reason="
                f"{appended_entry.get('reason')!r} does not match "
                f"new_sweep_id={new_sweep_id!r} + "
                f"new_typed_reason={new_typed_reason!r}"
            ),
        )
    old_identity = appended_entry.get("old_identity_tuple", {})
    prior_identities = {
        _full_identity_key(cell.identity_tuple)
        for cell in prior_manifest.cells
    }
    old_key = _full_identity_key(old_identity)
    if old_key not in prior_identities:
        raise CohortReplacementPolicyError(
            violation="old_identity_tuple_not_in_prior_manifest",
            detail=f"old_key={old_key!r}",
        )
    new_identity = appended_entry.get("new_identity_tuple", {})
    new_identities = {
        _full_identity_key(cell.identity_tuple)
        for cell in new_manifest.cells
    }
    new_key = _full_identity_key(new_identity)
    if new_key not in new_identities:
        raise CohortReplacementPolicyError(
            violation="new_identity_tuple_not_in_new_manifest",
            detail=f"new_key={new_key!r}",
        )
    # One-for-one swap semantics. The appended row claims a swap of
    # old_key -> new_key, so the identity-set diff between prior and
    # new must be exactly that.
    if old_key in new_identities:
        raise CohortReplacementPolicyError(
            violation="old_identity_tuple_still_present_in_new_manifest",
            detail=(
                f"old_key={old_key!r} is still present in the new "
                f"manifest; a real swap removes it"
            ),
        )
    if new_key in prior_identities:
        raise CohortReplacementPolicyError(
            violation="new_identity_tuple_already_present_in_prior_manifest",
            detail=(
                f"new_key={new_key!r} was already present in the "
                f"prior manifest; a real swap introduces a NEW "
                f"identity"
            ),
        )
    if len(new_manifest.cells) != len(prior_manifest.cells):
        raise CohortReplacementPolicyError(
            violation="cell_count_delta_outside_one_for_one_swap",
            detail=(
                f"new manifest has {len(new_manifest.cells)} cells; "
                f"prior had {len(prior_manifest.cells)}. A one-for-one "
                f"swap preserves cell count."
            ),
        )
    expected_new = (prior_identities - {old_key}) | {new_key}
    if new_identities != expected_new:
        unexpected_drops = (prior_identities - {old_key}) - new_identities
        unexpected_adds = new_identities - ((prior_identities - {old_key}) | {new_key})
        raise CohortReplacementPolicyError(
            violation="unaffected_identities_not_preserved",
            detail=(
                f"new manifest identity set diverges from "
                f"(prior - old_key) + new_key beyond the declared "
                f"swap: dropped={sorted(unexpected_drops)!r}, "
                f"added={sorted(unexpected_adds)!r}"
            ),
        )
    # Admission-status accounting. The identity-set + cell-count
    # checks above ensure exactly one identity was swapped, but they
    # do not see whether the swap preserved admission status. A
    # post-freeze swap that demotes admitted capacity by replacing
    # an admitted cell with an attempted_excluded one must not pass
    # silently; the replacement_history schema does not yet carry an
    # explicit admission-status delta field, so any status-changing
    # swap is rejected at this layer.
    if (
        new_manifest.attempted_cell_count
        != prior_manifest.attempted_cell_count
        or new_manifest.attempted_cell_count != len(new_manifest.cells)
    ):
        raise CohortReplacementPolicyError(
            violation="attempted_cell_count_drift_outside_one_for_one_swap",
            detail=(
                f"prior.attempted_cell_count="
                f"{prior_manifest.attempted_cell_count}; "
                f"new.attempted_cell_count="
                f"{new_manifest.attempted_cell_count}; "
                f"len(new.cells)={len(new_manifest.cells)}. A "
                f"one-for-one swap preserves attempted_cell_count and "
                f"keeps it in lockstep with the actual cells list."
            ),
        )
    prior_cells_by_identity = {
        _full_identity_key(cell.identity_tuple): cell
        for cell in prior_manifest.cells
    }
    new_cells_by_identity = {
        _full_identity_key(cell.identity_tuple): cell
        for cell in new_manifest.cells
    }
    # Actual-cell lockstep: the manifest's top-level
    # admitted_cell_count counter must equal the observed count of
    # admitted cells in the cells list. Without this check, a
    # caller can forge admitted_cell_count to mask an unaffected-
    # identity admission-status flip while the per-pair check and
    # the admitted-count delta check both pass.
    prior_admitted_actual = sum(
        1 for cell in prior_manifest.cells
        if cell.admission_status == "admitted"
    )
    if prior_manifest.admitted_cell_count != prior_admitted_actual:
        raise CohortReplacementPolicyError(
            violation=(
                "admitted_cell_count_inconsistent_with_cells_in_prior_manifest"
            ),
            detail=(
                f"prior_manifest.admitted_cell_count="
                f"{prior_manifest.admitted_cell_count}, observed "
                f"admitted in cells={prior_admitted_actual}. The "
                f"prior manifest's top-level admitted counter must "
                f"match the actual admitted cells in the cells list."
            ),
        )
    new_admitted_actual = sum(
        1 for cell in new_manifest.cells
        if cell.admission_status == "admitted"
    )
    if new_manifest.admitted_cell_count != new_admitted_actual:
        raise CohortReplacementPolicyError(
            violation=(
                "admitted_cell_count_inconsistent_with_cells_in_new_manifest"
            ),
            detail=(
                f"new_manifest.admitted_cell_count="
                f"{new_manifest.admitted_cell_count}, observed "
                f"admitted in cells={new_admitted_actual}. The "
                f"new manifest's top-level admitted counter must "
                f"match the actual admitted cells in the cells list."
            ),
        )
    old_cell = prior_cells_by_identity[old_key]
    new_cell = new_cells_by_identity[new_key]
    # Admission-status delta on the appended row. The schema
    # requires `old_admission_status` + `new_admission_status` on
    # every replacement_history item; the helper verifies they
    # match the actual cells' admission_status so the recorded
    # delta is auditable. The legacy
    # `replacement_admission_status_delta_unrecorded` arm is
    # subsumed: the schema enforces the fields exist, and the
    # mismatch literals below catch any drift between the recorded
    # delta and the actual cell statuses.
    recorded_old_status = str(
        appended_entry.get("old_admission_status", "")
    )
    recorded_new_status = str(
        appended_entry.get("new_admission_status", "")
    )
    if recorded_old_status != old_cell.admission_status:
        raise CohortReplacementPolicyError(
            violation="replacement_history_old_admission_status_mismatch",
            detail=(
                f"appended entry old_admission_status="
                f"{recorded_old_status!r}; old cell admission_status="
                f"{old_cell.admission_status!r}"
            ),
        )
    if recorded_new_status != new_cell.admission_status:
        raise CohortReplacementPolicyError(
            violation="replacement_history_new_admission_status_mismatch",
            detail=(
                f"appended entry new_admission_status="
                f"{recorded_new_status!r}; new cell admission_status="
                f"{new_cell.admission_status!r}"
            ),
        )
    # A new admitted cell carries a real kernel signature, not the
    # deterministic placeholder produced by the curation step. The
    # placeholder pattern is
    # sha256("pending_kernel_signature:"+slug+":"+shape_point);
    # any admitted cell still holding that placeholder failed to
    # complete the live trace + NCU label sweep.
    if new_cell.admission_status == "admitted":
        placeholder = _placeholder_kernel_signature(
            new_cell.slug, new_cell.shape_point,
        )
        actual_signature = str(
            new_cell.identity_tuple.get("kernel_signature_sha256", "")
        )
        if actual_signature == placeholder:
            raise CohortReplacementPolicyError(
                violation=(
                    "replacement_admitted_cell_carries_placeholder_kernel_signature"
                ),
                detail=(
                    f"new admitted cell {new_cell.slug}/"
                    f"{new_cell.shape_point} carries the deterministic "
                    f"placeholder kernel_signature_sha256; admission "
                    f"requires the real signature from the live trace + "
                    f"NCU sweep."
                ),
            )
    # Admitted-count invariance applies to EVERY accepted
    # status-preserving swap. The legacy guard fired for
    # admitted-to-admitted; status-changing swaps now go through
    # the schema-recorded path and ALSO obey actual-cell counter
    # lockstep (already enforced above). Only status-preserving
    # swaps need a direct counter-delta check.
    if old_cell.admission_status == new_cell.admission_status:
        if (
            new_manifest.admitted_cell_count
            != prior_manifest.admitted_cell_count
        ):
            raise CohortReplacementPolicyError(
                violation=(
                    "admitted_cell_count_delta_outside_one_for_one_swap"
                ),
                detail=(
                    f"prior.admitted_cell_count="
                    f"{prior_manifest.admitted_cell_count}; "
                    f"new.admitted_cell_count="
                    f"{new_manifest.admitted_cell_count}. A "
                    f"status-preserving one-for-one swap preserves the "
                    f"top-level admitted_cell_count counter."
                ),
            )
    # Unaffected-identity admission-status preservation: every
    # identity present in both manifests (the prior set minus the
    # swap-out, equivalently the new set minus the swap-in) must
    # have the same admission_status across the swap. Without this
    # check, a balanced pair of opposite-direction flips on
    # unaffected identities would keep every counter consistent
    # while silently changing per-identity status.
    for identity_key in prior_identities - {old_key}:
        prior_unaffected_cell = prior_cells_by_identity[identity_key]
        new_unaffected_cell = new_cells_by_identity[identity_key]
        if (
            prior_unaffected_cell.admission_status
            != new_unaffected_cell.admission_status
        ):
            raise CohortReplacementPolicyError(
                violation="unaffected_identity_admission_status_changed",
                detail=(
                    f"identity_key={identity_key!r} preserved across "
                    f"the swap but admission_status changed from "
                    f"{prior_unaffected_cell.admission_status!r} to "
                    f"{new_unaffected_cell.admission_status!r}"
                ),
            )


__all__ = [
    "CONTRACT_FLOOR_DIAGNOSTIC",
    "SUBMISSION_TARGET_DIAGNOSTIC",
    "FAMILY_BUCKET_MINIMA",
    "DEPTH_FIRST_REQUIRED_SLUGS",
    "DEPTH_FIRST_MIN_SHAPE_POINTS",
    "SLUG_MAX_FRACTION_PER_SPLIT",
    "FamilyBucketGateVerdict",
    "DepthFirstGateVerdict",
    "ContractFloorGateVerdict",
    "SubmissionGateVerdict",
    "BalancingGateVerdict",
    "ConsolidatedPublicationGateVerdict",
    "CohortReplacementPolicyError",
    "check_family_bucket_coverage",
    "check_depth_first_coverage",
    "check_contract_floor",
    "check_submission_gate",
    "check_balancing",
    "consolidated_publication_gate",
    "check_cohort_replacement",
]
