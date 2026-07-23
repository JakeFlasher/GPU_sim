"""Predictor design-coverage partition (deployed scope).

Maps every calibration cell to one of three predictor
buckets — friendly, hostile, or typed_unavailable — and,
for typed_unavailable cells, the product-language reason
that explains why the predictor cannot honestly score it.

The partition is contract-pinned BEFORE any calibration
sweep runs. Downstream predictor and bundle layers consume
this mapping directly; the calibration ground-truth
manifest lives in a separate quarantined subtree and is
authoritative, but deployed code MUST NOT import from
that subtree (the runtime NCU surface is forbidden in
deployed scope).

The mapping in this module is a deployed-scope duplicate
of the calibration manifest's per-cell view. A sync test
under the project test tree mechanically enforces that
the two stay equal: drift between the manifest and this
partition fails CI immediately.

Public surface:

  * `PREDICTOR_BUCKET_FRIENDLY`,
    `PREDICTOR_BUCKET_HOSTILE`,
    `PREDICTOR_BUCKET_TYPED_UNAVAILABLE` — string
    constants matching the predictor bucket enum.
  * `PREDICTOR_BUCKETS` — frozenset of the three values.
  * `PREDICTOR_TYPED_UNAVAILABLE_REASONS` — frozenset of
    product-language reasons for typed_unavailable
    cells.
  * `PredictorDesignCoverageEntry` — frozen dataclass:
    `cell_name`, `predictor_bucket`,
    `typed_unavailable_reason` (empty when not typed
    unavailable).
  * `PREDICTOR_DESIGN_COVERAGE_BY_CELL` — immutable
    mapping keyed by `cell_name`. The canonical
    partition.
  * `PredictorDesignCoverageError(typed_reason,
    message)` — exception with typed reasons:
    `cell_missing_cohort_partition`,
    `partition_bucket_invalid`,
    `partition_typed_unavailable_missing_reason`,
    `partition_post_hoc_bucket_reassignment`.
  * `lookup_predictor_design_coverage(cell_name)` —
    O(1) lookup; raises `cell_missing_cohort_partition`
    on miss.
  * `validate_predictor_design_coverage_entry(entry)` —
    raises typed errors for invalid bucket / missing
    reason / etc.

Importing this module does NOT invoke any external
process. It is pure data + pure functions.
"""

from __future__ import annotations

import tomllib
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PREDICTOR_BUCKET_FRIENDLY: str = "predictor_friendly"
PREDICTOR_BUCKET_HOSTILE: str = "predictor_hostile"
PREDICTOR_BUCKET_TYPED_UNAVAILABLE: str = (
    "predictor_typed_unavailable"
)


PREDICTOR_BUCKETS: frozenset[str] = frozenset({
    PREDICTOR_BUCKET_FRIENDLY,
    PREDICTOR_BUCKET_HOSTILE,
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
})


# Product-language reason vocabulary for typed_unavailable
# cells. Drawn from the corpus contract; populated only
# when bucket == predictor_typed_unavailable.
PREDICTOR_TYPED_UNAVAILABLE_REASONS: frozenset[str] = frozenset({
    "hostile_subclass_uncalibrated",
    "shared_memory_no_read_only_sector_traffic",
    "local_memory_private_cache_policy",
    "warm_cold_protocol_unpinned",
})


PREDICTOR_DESIGN_COVERAGE_TYPED_REASONS: frozenset[str] = frozenset({
    "cell_missing_cohort_partition",
    "partition_bucket_invalid",
    "partition_typed_unavailable_missing_reason",
    "partition_post_hoc_bucket_reassignment",
    "partition_duplicate_cell_source",
})


class PredictorDesignCoverageError(Exception):
    """Raised by the predictor-coverage lookup or
    validator on partition violations. Carries the
    typed-reason string so downstream callers (the
    predictor body, the bundle layer, the comparison
    runner) can compose typed-unavailable bundles or
    report-table errors without parsing message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class PredictorDesignCoverageEntry:
    cell_name: str
    predictor_bucket: str
    typed_unavailable_reason: str = ""


def _entry(
    cell_name: str,
    predictor_bucket: str,
    typed_unavailable_reason: str = "",
) -> tuple[str, PredictorDesignCoverageEntry]:
    return (
        cell_name,
        PredictorDesignCoverageEntry(
            cell_name=cell_name,
            predictor_bucket=predictor_bucket,
            typed_unavailable_reason=typed_unavailable_reason,
        ),
    )


# Cohort A — sm_86 NVBit microbench cells. 18 entries.
# Three cells are typed_unavailable: local-spill
# (local-memory private cache policy) and the two shared-
# memory cells (shared memory bypasses L1 and L2, so the
# read-only sector predictor has nothing to score).
_COHORT_A_ENTRIES: tuple[
    tuple[str, PredictorDesignCoverageEntry], ...
] = (
    _entry("calib_atomic_global", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_atomic_rmw_global", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_branch_16active", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_broadcast_global", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_cg_cache_bypass", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_coalesced_32x4B", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_coalesced_32x8B", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_cpasync_global_to_shared", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_divergent_lanes", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_hostile_gemm_stream_smoke", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_hostile_gqa_paged_decode_smoke", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_layernorm_fwd_bf16", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_warm_cold_synthetic", PREDICTOR_BUCKET_FRIENDLY),
    _entry(
        "calib_local_spill",
        PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
        "local_memory_private_cache_policy",
    ),
    _entry(
        "calib_shared_32way_conflict",
        PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
        "shared_memory_no_read_only_sector_traffic",
    ),
    _entry(
        "calib_shared_broadcast",
        PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
        "shared_memory_no_read_only_sector_traffic",
    ),
    _entry("calib_smoke_broadcast_loop", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_smoke_capacity_overflow_loop", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_smoke_coalesced_loop", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_stride2_32x4B", PREDICTOR_BUCKET_HOSTILE),
    _entry("calib_vector_32x16B", PREDICTOR_BUCKET_FRIENDLY),
    _entry("calib_vector_cross_32B", PREDICTOR_BUCKET_HOSTILE),
)


# Cohort B — SOL-ExecBench representative cells. 15
# entries. Every cell now carries a predictor-side
# bucket (friendly or hostile); the two KV-cache RoPE
# kernels (problem_018, problem_071) consume the
# `measured_cache_initial_state` carrier emitted by
# the calibration corpus manifest so the warm/cold
# residency input is a generic launch-frame signal
# rather than a slug-specific shortcut.
_COHORT_B_ENTRIES: tuple[
    tuple[str, PredictorDesignCoverageEntry], ...
] = (
    _entry("problem_001_fused_add_rmsnorm_h2048", PREDICTOR_BUCKET_FRIENDLY),
    _entry("problem_003_fused_add_rmsnorm_h7168", PREDICTOR_BUCKET_FRIENDLY),
    _entry("problem_004_gemm_n128_k2048", PREDICTOR_BUCKET_FRIENDLY),
    _entry("problem_007_gemm_n4096_k4096", PREDICTOR_BUCKET_FRIENDLY),
    _entry(
        "problem_046_attention_softmax_with_softcapping_and_dropout",
        PREDICTOR_BUCKET_FRIENDLY,
    ),
    _entry("problem_074_fused_gated_mlp_silu", PREDICTOR_BUCKET_FRIENDLY),
    _entry("problem_011_gemm_n28672_k4096", PREDICTOR_BUCKET_HOSTILE),
    _entry(
        "problem_012_gqa_paged_decode_h32_kv4_d128_ps1",
        PREDICTOR_BUCKET_HOSTILE,
    ),
    _entry(
        "problem_014_gqa_paged_prefill_causal_h32_kv4_d128_ps1",
        PREDICTOR_BUCKET_HOSTILE,
    ),
    _entry(
        "problem_003_fp8_mlp_gate_up_projection",
        PREDICTOR_BUCKET_HOSTILE,
    ),
    _entry("problem_020_nvfp4_linear_layer", PREDICTOR_BUCKET_HOSTILE),
    _entry(
        "problem_029_nvfp4_fused_mlp_silu_gating",
        PREDICTOR_BUCKET_HOSTILE,
    ),
    _entry(
        "problem_020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
        PREDICTOR_BUCKET_HOSTILE,
    ),
    _entry(
        "problem_018_fused_rope_with_qk_norm_and_kv_cache_update",
        PREDICTOR_BUCKET_HOSTILE,
    ),
    _entry(
        "problem_071_kv_cache_update_with_rope",
        PREDICTOR_BUCKET_HOSTILE,
    ),
)


def _build_partition() -> Mapping[str, PredictorDesignCoverageEntry]:
    partition: dict[str, PredictorDesignCoverageEntry] = {}
    for cell_name, entry in (*_COHORT_A_ENTRIES, *_COHORT_B_ENTRIES):
        if cell_name in partition:
            raise PredictorDesignCoverageError(
                "partition_duplicate_cell_source",
                f"cell {cell_name!r} appears twice in the "
                f"deployed partition source; this is a "
                f"distinct contract violation from "
                f"manifest-vs-partition drift (see "
                f"compare_partition_to_manifest_view)",
            )
        partition[cell_name] = entry
    return types.MappingProxyType(partition)


_EXPLICIT_PREDICTOR_DESIGN_COVERAGE_BY_CELL: Mapping[
    str, PredictorDesignCoverageEntry,
] = _build_partition()


# Relative path of the canonical host-arch availability manifest
# under the repo root. Kept as a constant here (rather than
# imported from `accuracy_diagnostic`) to avoid the circular
# dependency. The accuracy diagnostic and this module agree on
# the path because both are anchored to the same canonical
# filesystem layout.
_HOST_ARCH_MANIFEST_RELATIVE_PATH: str = (
    "scripts/baselines_harness/calibration/"
    "cache_prediction/host_arch_availability_v1.toml"
)


def _read_in_tree_host_arch_manifest_entries() -> list[Mapping[str, Any]]:
    """Best-effort read of the in-tree host-arch manifest.

    Returns the raw `[[problems]]` list from the TOML on
    success, or an empty list when:
      * The manifest file is absent (a fresh checkout
        before the manifest landed; unit-test harnesses
        that move the working directory).
      * The TOML fails to parse.
      * The `[[problems]]` array is malformed.

    This module's contract is to keep importing successfully
    in every environment; manifest-driven partition growth
    is a best-effort extension on top of the curated 36-cell
    floor. The strict validation lives in
    `accuracy_diagnostic.load_host_arch_availability_manifest`
    and runs on the production diagnostic path; if the
    manifest is malformed there the diagnostic fails closed.
    """
    here = Path(__file__).resolve()
    # `.../scripts/baselines_harness/metrics/predictor_design_coverage.py`
    # The repo root sits four `.parent` hops above this file.
    repo_root = here.parents[3]
    manifest_path = repo_root / _HOST_ARCH_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        return []
    try:
        raw = manifest_path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    problems = data.get("problems", [])
    if not isinstance(problems, list):
        return []
    return [row for row in problems if isinstance(row, dict)]


def _build_extended_partition() -> Mapping[
    str, PredictorDesignCoverageEntry,
]:
    """Merge the explicit curated partition with manifest-
    derived synthesized entries.

    For every `[[problems]]` row in the in-tree host-arch
    manifest whose `status == sm86_compatible_cohort`:
      * If the slug already lives in the explicit partition,
        the curated entry wins (preserves hand-reviewed
        typed-unavailable reasons + cohort buckets).
      * Otherwise, synthesize a `PredictorDesignCoverageEntry`
        carrying the manifest's `cohort_bucket` suggestion.
        Buckets outside `PREDICTOR_BUCKETS` (or set to
        `predictor_typed_unavailable`) are skipped silently;
        those skips surface as production
        `cell_missing_cohort_partition` errors on
        `lookup_predictor_design_coverage()`, which is the
        correct fail-closed behavior.

    The returned mapping is wrapped in `MappingProxyType` so
    downstream modules cannot mutate it.
    """
    merged: dict[str, PredictorDesignCoverageEntry] = dict(
        _EXPLICIT_PREDICTOR_DESIGN_COVERAGE_BY_CELL,
    )
    for row in _read_in_tree_host_arch_manifest_entries():
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if slug in merged:
            continue
        if row.get("status") != _HOST_ARCH_MANIFEST_SM86_COMPATIBLE_STATUS:
            continue
        bucket = row.get("cohort_bucket")
        if not isinstance(bucket, str) or bucket not in PREDICTOR_BUCKETS:
            continue
        if bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            # The manifest never emits a typed_unavailable
            # suggestion; the explicit partition is
            # authoritative for that cohort. Skip to keep the
            # extension safe.
            continue
        merged[slug] = PredictorDesignCoverageEntry(
            cell_name=slug,
            predictor_bucket=bucket,
            typed_unavailable_reason="",
        )
    return types.MappingProxyType(merged)


# `PREDICTOR_DESIGN_COVERAGE_BY_CELL` is assigned at the bottom
# of this module (after `_HOST_ARCH_MANIFEST_SM86_COMPATIBLE_STATUS`
# is defined; the merge helper consults that constant). Forward-
# declare the type so consumers reading downstream code see the
# intended shape.
PREDICTOR_DESIGN_COVERAGE_BY_CELL: Mapping[
    str, PredictorDesignCoverageEntry,
] = _EXPLICIT_PREDICTOR_DESIGN_COVERAGE_BY_CELL


def lookup_predictor_design_coverage(
    cell_name: str,
) -> PredictorDesignCoverageEntry:
    """Return the partition entry for `cell_name`.

    Raises `PredictorDesignCoverageError` with typed
    reason `cell_missing_cohort_partition` when the cell
    is not predeclared in the partition. The predictor
    body and the bundle layer rely on this exact typed
    reason to fail closed on unmapped cells.
    """
    entry = PREDICTOR_DESIGN_COVERAGE_BY_CELL.get(cell_name)
    if entry is None:
        raise PredictorDesignCoverageError(
            "cell_missing_cohort_partition",
            f"cell {cell_name!r} is not predeclared in the "
            f"predictor design-coverage partition; add it "
            f"to the deployed partition + the corpus "
            f"contract before running the calibration "
            f"sweep on it",
        )
    return entry


def partition_size() -> int:
    """Return the deployed partition's cell count.

    The accuracy diagnostic's `[overall_accuracy_summary]`
    table validator and the `_build_overall_summary`
    builder consult this so cohort-partition growth (new
    Cohort-B entries landing as the SOL-ExecBench corpus
    is hand-ported) does not require a lockstep edit of
    the assertion's literal value. The 235-problem total
    is a separate count and lives in
    `[coverage_summary].total_problems`.
    """
    return len(PREDICTOR_DESIGN_COVERAGE_BY_CELL)


# Vocabulary string the host-arch manifest uses to mark
# sm_86-compatible slugs. Keeping this constant here (rather
# than importing from `accuracy_diagnostic`) avoids a circular
# dependency — the diagnostic module imports this file, not
# the other way around.
_HOST_ARCH_MANIFEST_SM86_COMPATIBLE_STATUS: str = "sm86_compatible_cohort"


def _read_manifest_status_and_bucket(
    entry: Any,
) -> tuple[str | None, str | None]:
    """Duck-typed accessor for a host-arch manifest entry.

    The manifest loader returns a mapping whose values are
    `HostArchAvailabilityEntry` dataclasses with `.status` and
    `.cohort_bucket` attributes. Tests sometimes pass plain
    dicts derived from parsed TOML — the accessor checks
    attribute access first and falls back to mapping access.
    Returns `(status, bucket)` with `None` for missing
    components.
    """
    if isinstance(entry, Mapping):
        return (
            entry.get("status"),
            entry.get("cohort_bucket"),
        )
    return (
        getattr(entry, "status", None),
        getattr(entry, "cohort_bucket", None),
    )


def propose_cohort_classification(
    slug: str,
    manifest_view: Mapping[str, Any] | None = None,
) -> PredictorDesignCoverageEntry | None:
    """Propose a cohort bucket for `slug`.

    Lookup order:

      1. The explicit deployed partition
         (`PREDICTOR_DESIGN_COVERAGE_BY_CELL`). If `slug` is
         predeclared here, return that entry unchanged. The
         explicit partition is authoritative for any slug it
         covers (it carries hand-reviewed typed-unavailable
         reasons that the manifest does not encode).
      2. The host-arch manifest's `cohort_bucket` suggestion
         for `sm86_compatible_cohort` slugs. When the slug is
         present in `manifest_view` with status equal to
         `sm86_compatible_cohort` and a non-empty
         `cohort_bucket` in `PREDICTOR_BUCKETS` (excluding
         `predictor_typed_unavailable` — the manifest never
         emits that suggestion), the helper synthesizes a
         `PredictorDesignCoverageEntry` carrying the
         manifest's bucket. This satisfies the contract that
         every sm_86-compatible canonical slug must resolve to
         a non-`None` bucket.
      3. `None` — the slug is unknown to both sources OR the
         manifest marks it `host_arch_unsupported` (those
         slugs are carved out of the cohort partition entirely;
         the diagnostic's host-arch detector handles them on
         the per-problem row builder side).

    The helper is pure: no I/O. `manifest_view` may be `None`
    when callers (e.g. legacy code paths) do not have a
    manifest in hand; the helper then degrades to the
    explicit-partition-only lookup.
    """
    explicit = PREDICTOR_DESIGN_COVERAGE_BY_CELL.get(slug)
    if explicit is not None:
        return explicit
    if manifest_view is None:
        return None
    entry = manifest_view.get(slug)
    if entry is None:
        return None
    status, bucket = _read_manifest_status_and_bucket(entry)
    if status != _HOST_ARCH_MANIFEST_SM86_COMPATIBLE_STATUS:
        return None
    if not bucket or bucket not in PREDICTOR_BUCKETS:
        return None
    if bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        # The manifest's cohort_bucket suggestion is
        # informational only; the canonical typed-unavailable
        # cohort comes from the explicit partition above.
        # Reject this case so the caller cannot bypass the
        # curated typed_unavailable vocabulary by hand-editing
        # the manifest.
        return None
    return PredictorDesignCoverageEntry(
        cell_name=slug,
        predictor_bucket=bucket,
        typed_unavailable_reason="",
    )


def validate_predictor_design_coverage_entry(
    entry: PredictorDesignCoverageEntry,
) -> None:
    """Raise typed errors on invalid partition entries.

    Checks:
      * `predictor_bucket` is one of `PREDICTOR_BUCKETS`.
      * If bucket is `predictor_typed_unavailable`, the
        `typed_unavailable_reason` is non-empty AND in
        `PREDICTOR_TYPED_UNAVAILABLE_REASONS`.
      * If bucket is friendly or hostile,
        `typed_unavailable_reason` is empty (no spurious
        reason on non-typed-unavailable cells).
    """
    if not isinstance(entry, PredictorDesignCoverageEntry):
        raise PredictorDesignCoverageError(
            "partition_bucket_invalid",
            f"entry must be PredictorDesignCoverageEntry; "
            f"got {type(entry).__name__}",
        )
    if entry.predictor_bucket not in PREDICTOR_BUCKETS:
        raise PredictorDesignCoverageError(
            "partition_bucket_invalid",
            f"entry for cell {entry.cell_name!r} has "
            f"predictor_bucket {entry.predictor_bucket!r}; "
            f"expected one of {sorted(PREDICTOR_BUCKETS)}",
        )
    if entry.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        if not entry.typed_unavailable_reason.strip():
            raise PredictorDesignCoverageError(
                "partition_typed_unavailable_missing_reason",
                f"entry for cell {entry.cell_name!r} is "
                f"predictor_typed_unavailable but carries "
                f"no typed_unavailable_reason",
            )
        if entry.typed_unavailable_reason not in (
            PREDICTOR_TYPED_UNAVAILABLE_REASONS
        ):
            raise PredictorDesignCoverageError(
                "partition_typed_unavailable_missing_reason",
                f"entry for cell {entry.cell_name!r} has "
                f"typed_unavailable_reason "
                f"{entry.typed_unavailable_reason!r}; "
                f"expected one of "
                f"{sorted(PREDICTOR_TYPED_UNAVAILABLE_REASONS)}",
            )
    else:
        if entry.typed_unavailable_reason:
            raise PredictorDesignCoverageError(
                "partition_bucket_invalid",
                f"entry for cell {entry.cell_name!r} is "
                f"{entry.predictor_bucket} but carries a "
                f"typed_unavailable_reason "
                f"({entry.typed_unavailable_reason!r}); "
                f"only predictor_typed_unavailable entries "
                f"may carry a reason",
            )


def compare_partition_to_manifest_view(
    manifest_view: Mapping[str, tuple[str, str]],
    partition_view: Mapping[str, tuple[str, str]],
) -> None:
    """Raise `partition_post_hoc_bucket_reassignment` when
    any shared cell's `(predictor_bucket,
    typed_unavailable_subclass)` tuple differs between
    the two views, OR when the cell-set symmetric
    difference is non-empty (a cell appears in only one
    view).

    Both views are plain dict mappings. The helper takes
    no calibration objects so deployed code can call it
    without importing from the quarantined subtree.
    Callers convert their domain objects (manifest cells,
    partition entries) to `(bucket, subclass)` tuples
    before calling.

    Empty `typed_unavailable_subclass` is significant: a
    friendly cell has tuple `("predictor_friendly", "")`
    while a typed-unavailable cell has tuple
    `("predictor_typed_unavailable",
    "local_memory_private_cache_policy")`. Drift in
    either coordinate fails the helper.
    """
    manifest_cells = set(manifest_view)
    partition_cells = set(partition_view)
    only_in_manifest = sorted(manifest_cells - partition_cells)
    only_in_partition = sorted(partition_cells - manifest_cells)
    if only_in_manifest or only_in_partition:
        raise PredictorDesignCoverageError(
            "partition_post_hoc_bucket_reassignment",
            f"cell-set drift between manifest and "
            f"partition:\n"
            f"  manifest-only: {only_in_manifest}\n"
            f"  partition-only: {only_in_partition}",
        )
    drifted: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    for cell_name in sorted(manifest_cells & partition_cells):
        manifest_tuple = manifest_view[cell_name]
        partition_tuple = partition_view[cell_name]
        if manifest_tuple != partition_tuple:
            drifted.append(
                (cell_name, manifest_tuple, partition_tuple),
            )
    if drifted:
        diff_lines = "\n".join(
            f"  {cell}: manifest={mt} partition={pt}"
            for cell, mt, pt in drifted
        )
        raise PredictorDesignCoverageError(
            "partition_post_hoc_bucket_reassignment",
            f"{len(drifted)} cell(s) have drifted between "
            f"manifest and partition "
            f"(predictor_bucket, typed_unavailable_subclass) "
            f"tuples:\n{diff_lines}",
        )


# Extend the partition with manifest-derived synthesized
# entries (best-effort; the curated explicit partition is the
# floor when no manifest is available). The merge runs at
# module import so `lookup_predictor_design_coverage()`
# satisfies the cohort-completeness contract: every
# `sm86_compatible_cohort` slug in the manifest resolves to a
# non-`None` partition entry on the production lookup path.
PREDICTOR_DESIGN_COVERAGE_BY_CELL = _build_extended_partition()


# Validate the extended partition at import time. Any
# drift between this module's data and the canonical
# constraints would fail import, which is the strictest
# possible CI signal.
for _entry_value in PREDICTOR_DESIGN_COVERAGE_BY_CELL.values():
    validate_predictor_design_coverage_entry(_entry_value)
del _entry_value


__all__ = [
    "PREDICTOR_BUCKET_FRIENDLY",
    "PREDICTOR_BUCKET_HOSTILE",
    "PREDICTOR_BUCKET_TYPED_UNAVAILABLE",
    "PREDICTOR_BUCKETS",
    "PREDICTOR_DESIGN_COVERAGE_BY_CELL",
    "PREDICTOR_DESIGN_COVERAGE_TYPED_REASONS",
    "PREDICTOR_TYPED_UNAVAILABLE_REASONS",
    "PredictorDesignCoverageEntry",
    "PredictorDesignCoverageError",
    "compare_partition_to_manifest_view",
    "lookup_predictor_design_coverage",
    "partition_size",
    "propose_cohort_classification",
    "validate_predictor_design_coverage_entry",
]
