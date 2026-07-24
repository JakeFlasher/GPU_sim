"""End-to-end smoke harness: trace -> simulator
-> prediction -> ledger pairing -> first measured
residual artifact.

Drives the deployed cache predictor against a
subset of cells from the existing sm_86 NCU
label ledger. For each cell:

  1. Look up the ledger entry (NCU labels +
     binary_sha256 + warmup metadata).
  2. Locate the binary on disk by matching
     ``binary_sha256``.
  3. Invoke the NVBit ``metric_extractor.so``
     against the binary with
     ``METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1``
     so the Tier-2 sector-event sidecar emits.
  4. Parse the sidecar into a list of
     ``LaunchFrame`` records via the deployed
     ``tier2_sidecar_parser``.
  5. Resolve the sm_86 cache geometry via the
     deployed ``cache_geometry`` resolver
     (static device card, no CUDA reading).
  6. Call the deployed
     ``predict_warmed_launch_sequence`` with
     ``warmup_count = ledger.launch_skip_count``
     so frames[0..launch_skip_count) are
     replayed silently into one persistent
     simulator pair before the
     ``launch_skip_count``-th frame is
     measured. This mirrors NCU's
     ``--launch-skip N --launch-count 1``
     semantics — the same protocol the
     calibration ledger labels were captured
     with.
  7. Pair the prediction to the ledger's NCU
     labels and compute per-cell absolute
     percentage-point errors.

Emits a TOML artifact:
``end_to_end_smoke_report.toml`` carrying
per-cell records + cohort aggregates +
identity/provenance.

The harness lives in the calibration subtree
(calibration -> deployed import direction is
allowed by the NCU quarantine rule). The
deployed predictor + tier2 parser +
cache_geometry modules are import-legal in
this direction; the harness adds no NCU
runtime dependency because the labels are
read from the existing ledger TOML, not from
a live NCU invocation.

Known framework gap surfaced by this harness
(documented but out-of-scope for this round):

* The NVBit Tier-2 sidecar instruments a
  narrower set of operations than NCU's
  hardware counters cover; per-launch sector
  counts in the sidecar can be smaller than
  the NCU-measured sector totals. This is the
  instrumentation surface; it does not block
  the measurement but bounds the achievable
  prediction fidelity at the current
  instrumentation coverage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from sol_baselines_harness.calibration.cache_prediction.calibration_sweep import (  # noqa: E402
    LabelLedger,
    LedgerEntry,
    load_ledger_from_toml,
)
from sol_baselines_harness.metrics.cache_geometry import (  # noqa: E402
    resolve_cache_geometry,
)
from sol_baselines_harness.metrics.cache_predictor import (  # noqa: E402
    compose_predictor_bundle,
    predict_from_launch_frame,
    predict_warmed_launch_sequence,
)
from sol_baselines_harness.metrics.framework_contract import (  # noqa: E402
    CellJoinKey,
    FrameworkContractError,
    InputSidecarDigest,
)
from sol_baselines_harness.metrics.isotonic_calibrator import (  # noqa: E402
    CalibrationInputRecord,
    IsotonicCalibrator,
    IsotonicCalibratorError,
    LOOCVResult,
    run_loocv,
)
from sol_baselines_harness.metrics.predictor_bundle_schema import (  # noqa: E402
    ManifestPairingIndex,
    PredictorBundle,
    PredictorBundleSchemaError,
)
from sol_baselines_harness.metrics.predictor_design_coverage import (  # noqa: E402
    PREDICTOR_BUCKET_FRIENDLY,
    PREDICTOR_BUCKET_HOSTILE,
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    lookup_predictor_design_coverage,
)
from sol_baselines_harness.metrics.cache_access_policy import (  # noqa: E402
    audit_l2_read_denominator,
)
from sol_baselines_harness.metrics.tier2_sidecar_parser import (  # noqa: E402
    LaunchFrame,
    Tier2ParserError,
    parse_launch_frames,
)
from sol_baselines_harness.metrics.tiered_gate_evaluator import (  # noqa: E402
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    CACHE_LEVEL_L1,
    CACHE_LEVEL_L2,
    SECONDARY_REL_ERR_EPSILON,
    CohortGateResult,
    TieredGateEvaluatorError,
    TieredGateReport,
    evaluate_tiered_gate,
)
from sol_baselines_harness.metrics.typed_unavailable_bundle import (  # noqa: E402
    COVERAGE_FLOOR_SMOKE,
)
from sol_baselines_harness.surfaces.cache_prediction import (  # noqa: E402
    CACHE_PREDICTION_SURFACE_ARTIFACT_RELPATH,
    CachePredictionSurfaceEntry,
    CachePredictionSurfaceError,
    emit_cache_prediction_surface_entry,
)


END_TO_END_SMOKE_TYPED_REASONS: frozenset[str] = frozenset({
    "smoke_input_invalid",
    "smoke_binary_not_found",
    "smoke_trace_capture_failed",
    "smoke_sidecar_parse_failed",
    "smoke_sidecar_digest_mismatch",
    "smoke_launch_index_out_of_range",
    "smoke_predictor_typed_unavailable",
    "smoke_ledger_load_failed",
    "smoke_report_commit_sha_mismatch",
})


# Audit-row typed-reason vocabulary. The audit diagnostic
# is the empirical contract for what classes the producer
# covers vs misses; a measured cell that cannot be audited
# must emit a typed-unavailable audit row rather than be
# silently omitted from the audit array.
L2_DENOMINATOR_AUDIT_REASON_OK: str = "ok"
L2_DENOMINATOR_AUDIT_REASON_TRACE_PATH_MISSING: str = (
    "tier2_trace_path_missing"
)
L2_DENOMINATOR_AUDIT_REASON_TRACE_UNPARSEABLE: str = (
    "tier2_trace_unparseable"
)
L2_DENOMINATOR_AUDIT_REASON_FRAME_INDEX_OUT_OF_RANGE: str = (
    "tier2_launch_frame_index_out_of_range"
)
L2_DENOMINATOR_AUDIT_REASON_NO_CLASS_ROWS: str = (
    "l2_denominator_audit_no_class_rows"
)
L2_DENOMINATOR_AUDIT_TYPED_REASONS: frozenset[str] = frozenset({
    L2_DENOMINATOR_AUDIT_REASON_OK,
    L2_DENOMINATOR_AUDIT_REASON_TRACE_PATH_MISSING,
    L2_DENOMINATOR_AUDIT_REASON_TRACE_UNPARSEABLE,
    L2_DENOMINATOR_AUDIT_REASON_FRAME_INDEX_OUT_OF_RANGE,
    L2_DENOMINATOR_AUDIT_REASON_NO_CLASS_ROWS,
})
_L2_DENOMINATOR_AUDIT_STATUS_OK: str = "ok"
_L2_DENOMINATOR_AUDIT_STATUS_UNAVAILABLE: str = (
    "l2_denominator_audit_unavailable"
)


_DEFAULT_NVBIT_SO = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
)
_DEFAULT_CALIBRATION_BIN_DIRS: tuple = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "tests" / "calibration",
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "tests" / "calibration_cohort_b",
    # Direct-NCU-harvest binaries for cohort-B-INELIGIBLE SOL problem
    # slugs (custom-CUDA, no cuBLAS / cuDNN dispatch — the deployed
    # baseline itself is compiled as a standalone calibration
    # executable rather than via a surrogate probe). See the
    # warm/cold cache-protocol design dossier for the path convention.
    _REPO_ROOT / "baselines" / "L1_direct_ncu_binaries",
)
# Backwards-compat alias for callers that pass a single
# explicit directory via positional argument. The
# multi-root scanner accepts either an iterable of
# directories or a single Path; the singleton remains
# resolvable through the same callers.
_DEFAULT_CALIBRATION_BIN_DIR = _DEFAULT_CALIBRATION_BIN_DIRS[0]

# Default per-cell subprocess timeout for the NVBit
# Tier-2 trace capture. The legacy 60s was insufficient
# for the larger ``problem_NNN`` cells in the cohort-B
# calibration corpus (the NVBit-instrumented binary
# adds substantial overhead to a stock kernel run);
# 300s accommodates the worst-case admitted hostile
# cells while still bounding total smoke-run wallclock.
# Operators can override via the env var
# ``METRIC_EXTRACTOR_TIER2_CAPTURE_TIMEOUT_SECONDS``.
_DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS: int = 300

# Bounded default slot-pool cap for Tier-2 address-dump
# subprocesses spawned by ``_capture_tier2_trace``. When
# the address-dump gate is on, every slot in the
# metric_extractor's launch-slot pool also costs
# ``sizeof(TierTwoBuffer)`` (~478 MiB per slot at the
# canonical TIER2_RECORDS_PER_LAUNCH=524288 cap), so the
# default 4096-slot allocation would request well over
# a terabyte of managed memory and trip the
# metric_extractor's per-process budget clamp loudly on
# every smoke run. The calibration smoke pipeline only
# captures a handful of launches per cell (~6 launches
# for the warm/cold direct binaries; the cohort-B GEMM
# binaries stay well under 32 launches), so a small
# default cap keeps managed-memory usage proportional
# to the actual smoke surface. The operator can override
# via the ``METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS`` env var
# when running smoke against a larger surface.
_DEFAULT_TIER2_CAPTURE_MAX_LAUNCH_SLOTS: int = 8

# Bounded default per-process slot-pool budget for Tier-2
# address-dump subprocesses spawned by ``_capture_tier2_trace``.
# Honors the operator's
# ``METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES`` override when set;
# otherwise injects this default (4 GiB). The bumped Tier-2 record
# cap (524,288 records / ~478 MiB per slot) means the smoke
# pipeline needs more than the compile-time default 512 MiB
# slot-pool budget to fit the warmup-plus-measured launch sequence
# at one slot per launch. With the 4 GiB override the budget clamp
# admits roughly 8 slots, enough for the 6-launch warm/cold direct
# binaries while staying comfortably bounded against host managed
# memory pressure. The metric_extractor's ``nvbit_tool_init``
# clamps the requested slot count against this budget so an
# off-budget configuration still allocates only what fits.
_DEFAULT_TIER2_CAPTURE_PER_PROCESS_BUDGET_BYTES: int = (
    4 * 1024 * 1024 * 1024
)
_DEFAULT_LEDGER_PATH = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "label_ledger_v1_full_partition.toml"
)


# Per-cell status enum:
_STATUS_MEASURED: str = "measured"
_STATUS_TYPED_UNAVAILABLE: str = "typed_unavailable"
_STATUS_FAILED: str = "failed"


class EndToEndSmokeError(Exception):
    """Raised by the smoke harness on per-cell
    contract violations. Carries the typed
    reason so a downstream consumer can compose
    typed-unavailable bundles without parsing
    message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class SmokeCellSpec:
    """One cell to drive through the smoke
    harness. Resolved from a ``LedgerEntry`` +
    a binary-path lookup.

    The cohort + join-key fields
    (``arch``, ``ncu_metric_version``,
    ``shape_point``, ``source_commit``,
    ``kernel_signature_sha256``) carry the
    full 7-tuple cell join key (with
    ``cell_name`` providing the slug) so the
    calibration wiring downstream can
    reconstruct the deployed
    ``framework_contract.CellJoinKey`` per
    cell without re-loading the ledger. Each
    field comes from the matching
    ``LedgerEntry`` field at spec-construction
    time.
    """

    cell_name: str
    binary_path: Path
    binary_sha256: str
    launch_skip_count: int
    predictor_bucket: str
    ncu_l1_hit_rate: float
    ncu_l2_hit_rate: float
    ncu_l1_sectors_total: int
    ncu_l1_sectors_hit: int
    ncu_l2_sectors_total: int
    ncu_l2_sectors_hit: int
    arch: str = ""
    ncu_metric_version: str = ""
    shape_point: str = ""
    source_commit: str = ""
    kernel_signature_sha256: str = ""
    # Cache initial state at the measured launch boundary,
    # threaded directly from the ledger entry. Under NCU's
    # default ``--replay-mode kernel`` (the project-pinned
    # calibration invocation), this is ``"cold"`` because
    # the profiler flushes L1 between metric replays; the
    # predictor's cold mode instantiates fresh L1 + L2
    # simulators for the measured frame and skips the
    # warmup-frame replay into those sims. A ledger entry
    # stamped under ``--replay-mode application`` or under
    # the paired-launch ``warm_cold_pair`` protocol would
    # carry the value the ledger declares; the spec passes
    # it verbatim to the predictor.
    cache_initial_state: str = ""


@dataclass(frozen=True)
class CapturedSmokeRecord:
    """One per-cell measurement output."""

    cell_name: str
    predictor_bucket: str
    status: str  # one of MEASURED / TYPED_UNAVAILABLE / FAILED
    ncu_l1_hit_rate: float
    ncu_l2_hit_rate: float
    predicted_l1_hit_rate: Optional[float]
    predicted_l2_hit_rate: Optional[float]
    l1_abs_pp_error: Optional[float]
    l2_abs_pp_error: Optional[float]
    l1_sectors_hit: Optional[int]
    l1_sectors_total: Optional[int]
    l2_sectors_hit: Optional[int]
    l2_sectors_total: Optional[int]
    launch_skip_count: int
    # Warmed-helper protocol provenance sourced directly from
    # the predictor result's provenance dict. These fields
    # mirror the canonical ledger fields the calibration
    # corpus pinned: ``warmup_protocol`` matches the deployed
    # ``warmup_protocol`` module constants
    # (``ncu_launch_skip_n_count_1`` for launch_skip>=1,
    # ``unknown`` for the cold-equivalent launch_skip=0 path),
    # ``cache_initial_state`` records the cache state at the
    # measured frame's launch (``warm`` for launch_skip>=1,
    # ``cold`` for launch_skip=0). ``measured_frame_uid``
    # identifies the specific LaunchFrame whose hit/miss
    # counts populated the residual; ``total_frame_count``
    # records how many frames the producer captured for this
    # cell.
    warmup_protocol: Optional[str]
    cache_initial_state: Optional[str]
    measured_frame_uid: Optional[int]
    total_frame_count: Optional[int]
    # Calibration outputs populated by the
    # ``IsotonicCalibrator`` LOOCV pass against the
    # cohort of admitted measured cells. ``None`` on
    # non-measured rows (typed-unavailable / failed)
    # AND on measured rows when the LOOCV admitted
    # cohort fell below the smoke floor (the entire
    # report's ``calibration_typed_reason`` then names
    # ``coverage_below_smoke_floor``).
    predicted_l1_calibrated: Optional[float]
    predicted_l2_calibrated: Optional[float]
    l1_calibrated_abs_pp_error: Optional[float]
    l2_calibrated_abs_pp_error: Optional[float]
    l1_rel_err: Optional[float]
    l2_rel_err: Optional[float]
    # Durable per-cell trace path. The
    # ``_capture_tier2_trace`` helper copies the
    # producer-written ``tier2_address_trace.t2a``
    # to a stable per-cell location under
    # ``out_dir/traces/<cell_name>/`` so a
    # downstream consumer of the bundle's
    # ``InputSidecarDigest(sector_stream)`` can
    # SHA-256 the named file and verify it
    # matches the bundle's recorded digest. The
    # producer-written scratch path is
    # overwritten by each subsequent cell, so
    # the durable per-cell copy is what closes
    # the AC-FI-1 causal-lineage claim.
    # ``None`` on non-measured rows (typed-
    # unavailable / failed before capture).
    tier2_trace_path: Optional[str]
    typed_reason: Optional[str]
    typed_message: Optional[str]
    sidecar_sha256: Optional[str]
    # Join keys against the calibration label ledger. The
    # diagnostic's `predictor_callback_from_smoke_records`
    # compares these against `entries[*].binary_sha256` and
    # `entries[*].kernel_signature_sha256` to fire
    # `diagnostic_failed / join_key_drift` when the smoke run
    # rebuilt the kernel after the labels were harvested. `None`
    # when the producer has not yet wired these (the join-key
    # detector then degrades to a no-op for that cell, never to
    # a false-positive drift signal).
    binary_sha256: Optional[str] = None
    kernel_signature_sha256: Optional[str] = None


@dataclass(frozen=True)
class CohortAggregate:
    cohort: str
    cell_count: int
    l1_mean_abs_pp_error: Optional[float]
    l2_mean_abs_pp_error: Optional[float]
    l1_max_abs_pp_error: Optional[float]
    l2_max_abs_pp_error: Optional[float]


@dataclass(frozen=True)
class GateCohortResultRow:
    """Flat, TOML-serializable view of one
    ``CohortGateResult``. Holds the (cohort,
    cache_level) pair plus all gate-evaluator
    outputs in a single dataclass so the
    serializer's
    ``dataclasses.fields(record)`` loop emits
    every field uniformly.
    """

    cohort: str
    cache_level: str
    sample_count: int
    mean_abs_pp_error: float
    max_abs_pp_error: float
    threshold_pp: float
    passed: bool
    mean_abs_pp_error_ci_lower: float
    mean_abs_pp_error_ci_upper: float
    secondary_rel_err_geomean: float
    secondary_rel_err_geomean_ci_lower: float
    secondary_rel_err_geomean_ci_upper: float
    secondary_rel_err_geomean_ci_half_width: float


@dataclass(frozen=True)
class TieredGateReportRow:
    """Flat, TOML-serializable view of one
    ``TieredGateReport``. The
    ``calibration_typed_reason`` field is non-
    empty (and ``overall_passed`` False) when
    the LOOCV admitted-cell count fell below
    the smoke floor; the gate cohort rows
    then carry zero sample counts.
    """

    overall_passed: bool
    bootstrap_resamples: int
    bootstrap_confidence: float
    random_seed: int
    admitted_cell_count: int
    calibration_typed_reason: str
    calibration_typed_message: str


@dataclass(frozen=True)
class CachePredictionSurfaceEntryRow:
    """Flat, TOML-serializable view of one
    ``CachePredictionSurfaceEntry``.

    The deployed surface dataclass nests a
    ``CachePredictionProvenance`` block and a
    ``ManifestPairingIndex``; this row flattens
    both into scalar fields so the existing
    ``dataclasses.fields(record)`` iteration in
    the TOML serializer emits every field
    uniformly.

    On the abstention path (per-cohort smoke
    floor unmet, no bundles composed), the row
    records ``status="unavailable"`` and the
    ``waiver_reason`` names the typed
    abstention from the calibration step; the
    pairing-index + artifact fields are empty
    strings (None on bool fields is rejected by
    the deployed validator, but None on
    Optional[str] is fine here since this is the
    TOML row, not the deployed entry).
    """

    status: str
    prediction_mode: str
    calibration_source: str
    confidence_band: str
    b200_validated: bool
    artifact_path: Optional[str]
    command: Optional[str]
    waiver_reason: Optional[str]
    pairing_slug: str
    pairing_shape_point: str
    pairing_binary_sha256: str
    pairing_source_commit: str
    pairing_kernel_signature_sha256: str
    bundles_written: int


@dataclass(frozen=True)
class L2DenominatorAuditClassRow:
    """One row of the per-class breakdown for a cell's
    L2-read-denominator audit. Keys identify the routed
    classification triple; ``sectors`` is the total number
    of 32-byte sectors the producer assigned to that class
    with ``counts_for_l2_read_score=True``.
    """

    op_kind: int
    address_space: int
    l1_access_policy: int
    sectors: int


@dataclass(frozen=True)
class L2DenominatorAuditRow:
    """Per-cell L2-read-denominator audit comparing the
    producer's captured total against the calibration
    ledger's recorded denominator
    (``ncu_l2_sectors_total``). ``denominator_gap`` is
    ledger_total - producer_total (positive when the
    producer undercounts); ``denominator_gap_ratio`` is
    the gap divided by the ledger total and is the
    canonical "fraction of the ledger denominator the
    producer fails to cover" metric. The per-class
    breakdown lets a downstream consumer attribute the gap
    to specific routed classes (and to spot classes that
    are entirely missing from the producer's
    instrumentation surface).

    ``audit_status`` is one of ``"ok"`` or
    ``"l2_denominator_audit_unavailable"``. The
    typed-unavailable status carries the canonical
    ``typed_reason`` so a downstream consumer knows why
    the audit could not be performed (missing trace path,
    unparseable trace, frame-index out of range, or zero
    per-class rows). Typed-unavailable rows carry an
    empty ``per_class_breakdown`` and zero producer total
    but still record the ledger total so the gap-ratio
    field documents the cell's nominal denominator size.
    """

    cell_name: str
    tier2_l2_sectors_total: int
    ncu_l2_sectors_total: int
    denominator_gap: int
    denominator_gap_ratio: float
    per_class_breakdown: tuple[L2DenominatorAuditClassRow, ...]
    audit_status: str = "ok"
    typed_reason: str = ""


@dataclass(frozen=True)
class EndToEndSmokeReport:
    schema_version: str
    generated_at_utc: str
    host_arch: str
    ledger_path: str
    nvbit_so_sha256: str
    commit_sha: str
    per_cell_records: tuple[CapturedSmokeRecord, ...]
    cohort_aggregates: tuple[CohortAggregate, ...]
    tiered_gate_report: Optional[TieredGateReportRow]
    gate_cohort_results: tuple[GateCohortResultRow, ...]
    # One ``CachePredictionSurfaceEntryRow`` per
    # admitted measured cell. The deployed
    # ``CachePredictionSurfaceEntry`` shape is
    # per-baseline (its bundles must all share a
    # ``ManifestPairingIndex``), so the smoke
    # harness emits N entries for N calibration
    # cells. On the abstention path
    # (per-cohort floor unmet, no calibratable
    # cells), the tuple is a singleton
    # ``status=unavailable`` row with a
    # canonical ``waiver_reason`` instead.
    cache_prediction_surface_entries: tuple[
        CachePredictionSurfaceEntryRow, ...
    ]
    # One audit row per measured cell, comparing the
    # producer's captured L2-read-denominator total
    # against the NCU ledger's
    # `lts__t_sectors_op_read.sum`. The audit's per-class
    # breakdown is the empirical contract for which
    # classes the producer's instrumentation surface
    # covers and which it misses; downstream
    # producer-side broadening targets the classes the
    # audit reveals are missing. Defaults to an empty
    # tuple so legacy fixtures that construct the report
    # without the audit continue to build (the field is
    # purely additive and the smoke harness always
    # populates it from the measured launch frames).
    l2_denominator_audit: tuple[L2DenominatorAuditRow, ...] = ()
    # Producer-side trace budget configured for this
    # smoke run (the
    # ``METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES``
    # env var if set, else ``"default"`` to indicate
    # the producer's compiled-in default — currently
    # 64 MiB). Recorded in the artifact's ``[meta]``
    # block so a downstream consumer can audit
    # whether a `trace_budget_exceeded` abstention
    # path fired under an elevated budget. Defaults
    # to ``"default"`` so legacy test fixtures that
    # build the report without setting the budget
    # continue to construct.
    trace_budget_bytes: str = "default"
    # Per-cell subprocess timeout (the
    # ``METRIC_EXTRACTOR_TIER2_CAPTURE_TIMEOUT_SECONDS``
    # env var if set, else the helper's default).
    # Recorded so a downstream consumer can audit
    # whether a `smoke_trace_capture_failed` was a
    # subprocess timeout under a known budget. Defaults
    # to the helper's default so legacy fixtures
    # continue to construct.
    trace_capture_timeout_seconds: int = (
        _DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS
    )


def _validate_iterable(name: str, value: Any) -> None:
    if not hasattr(value, "__iter__"):
        raise EndToEndSmokeError(
            "smoke_input_invalid",
            f"{name} must be iterable; got {type(value).__name__}",
        )


def _file_sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_binary_by_sha256(
    binary_sha256: str,
    search_dir,
) -> Optional[Path]:
    """Scan a directory or iterable of directories for an
    executable whose SHA-256 matches. Returns the path or
    ``None`` if not found.

    ``search_dir`` may be a single :class:`Path` (legacy
    single-directory mode preserved for tests that pass
    one explicit directory) or any iterable of paths
    (multi-root scan; the calibration corpus lives across
    two root directories — `tests/calibration` for the
    primary calibration cohort and
    `tests/calibration_cohort_b` for the cohort-B kernels
    derived from the SOL-ExecBench problem corpus).
    Iteration order is preserved so the resolution is
    deterministic.
    """
    if isinstance(search_dir, Path):
        roots: tuple = (search_dir,)
    else:
        roots = tuple(search_dir)
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_file():
                continue
            if not os.access(entry, os.X_OK):
                continue
            try:
                actual = _file_sha256_hex(entry)
            except OSError:
                continue
            if actual == binary_sha256:
                return entry
    return None


def _ledger_entry_to_smoke_spec(
    entry: LedgerEntry, *, search_dir,
) -> SmokeCellSpec:
    """Build a :class:`SmokeCellSpec` from a
    :class:`LedgerEntry`.

    ``search_dir`` accepts either a single :class:`Path`
    (legacy single-directory mode preserved for tests
    that drive a fixture binary at a known location) or
    an iterable of directories — the multi-root scanner
    walks each in order and returns the first match.
    """
    binary_path = _resolve_binary_by_sha256(
        entry.binary_sha256, search_dir,
    )
    if binary_path is None:
        raise EndToEndSmokeError(
            "smoke_binary_not_found",
            f"cell {entry.cell_name!r}: no binary in "
            f"{search_dir} matched sha256 "
            f"{entry.binary_sha256!r}",
        )
    return SmokeCellSpec(
        cell_name=entry.cell_name,
        binary_path=binary_path,
        binary_sha256=entry.binary_sha256,
        launch_skip_count=entry.launch_skip_count,
        predictor_bucket=entry.predictor_bucket,
        ncu_l1_hit_rate=entry.label.l1_global_load_hit_rate,
        ncu_l2_hit_rate=entry.label.l2_read_hit_rate,
        ncu_l1_sectors_total=entry.label.l1_global_load_sectors_total,
        ncu_l1_sectors_hit=entry.label.l1_global_load_sectors_hit,
        ncu_l2_sectors_total=entry.label.l2_read_sectors_total,
        ncu_l2_sectors_hit=entry.label.l2_read_sectors_hit,
        arch=entry.arch,
        ncu_metric_version=entry.ncu_metric_version,
        shape_point=entry.shape_point,
        source_commit=entry.source_commit,
        kernel_signature_sha256=entry.kernel_signature_sha256,
        cache_initial_state=entry.cache_initial_state,
    )


def _capture_tier2_trace(
    spec: SmokeCellSpec,
    *,
    out_dir: Path,
    nvbit_so: Path,
    timeout_seconds: Optional[int] = None,
) -> Path:
    """Drive the NVBit injector against the cell's
    binary with Tier-2 address-dump enabled.

    The producer writes the sidecar to its own
    fixed filename
    (``tier2_address_trace.t2a``) next to the
    launches JSONL. Every cell's invocation
    overwrites that file. To preserve a durable
    per-cell copy whose SHA-256 matches the
    bundle's ``InputSidecarDigest(sector_stream)``
    entry, this helper copies the post-capture
    bytes to
    ``out_dir/traces/<cell_name>/tier2_address_trace.t2a``
    and returns that per-cell path. A downstream
    consumer of the bundle's named sidecar path
    can SHA-256 the bytes at the named path and
    verify they match the bundle's digest.

    ``timeout_seconds`` controls the per-cell
    subprocess timeout. When ``None`` (default),
    the helper honors the
    ``METRIC_EXTRACTOR_TIER2_CAPTURE_TIMEOUT_SECONDS``
    env var if set, falling back to
    ``_DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS``
    (300s; large `problem_NNN` cells in the
    calibration corpus need more than the
    legacy 60s for the NVBit-instrumented
    binary to finish). The producer-side
    trace budget envelope is controlled by
    ``METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES``
    (default 64 MiB inside the producer); the
    parent env is passed through unmodified
    so the operator can raise the budget for
    a smoke run that includes large cells.
    """
    if timeout_seconds is None:
        env_timeout = os.environ.get(
            "METRIC_EXTRACTOR_TIER2_CAPTURE_TIMEOUT_SECONDS",
        )
        try:
            timeout_seconds = (
                int(env_timeout) if env_timeout
                else _DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS
            )
        except ValueError:
            timeout_seconds = (
                _DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    launches_path = out_dir / f"{spec.cell_name}_launches.jsonl"
    scratch_sidecar_path = (
        launches_path.parent / "tier2_address_trace.t2a"
    )
    if scratch_sidecar_path.exists():
        scratch_sidecar_path.unlink()
    env = {
        **os.environ,
        "CUDA_INJECTION64_PATH": str(nvbit_so),
        "METRIC_EXTRACTOR_USER_MODULE_REGEX": ".*",
        "METRIC_EXTRACTOR_LAUNCHES_PATH": str(launches_path),
        "METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP": "1",
        "NOBANNER": "1",
    }
    # Bound the metric_extractor slot pool when the operator has not
    # supplied their own cap. With address-dump enabled every slot
    # pays both Tier-1 and Tier-2 footprint; the unbounded default
    # (4096) would trigger the producer's per-process budget clamp
    # warning on every smoke invocation. The smoke pipeline only
    # needs enough slots to capture the warmup-plus-measured launch
    # sequence (typically <= 8 launches per direct binary).
    if "METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS" not in env:
        env["METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"] = str(
            _DEFAULT_TIER2_CAPTURE_MAX_LAUNCH_SLOTS
        )
    # Raise the per-process slot-pool budget to fit the bumped
    # Tier-2 per-launch record cap (~478 MiB per slot) multiplied
    # by the warmup-plus-measured launch sequence. The
    # metric_extractor's compile-time default (512 MiB) only fits a
    # single slot at the bumped cap, which would drop the warmup
    # launches and prevent the warm-aligned predictor from reaching
    # the targeted frame.
    if "METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES" not in env:
        env["METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES"] = str(
            _DEFAULT_TIER2_CAPTURE_PER_PROCESS_BUDGET_BYTES
        )
    try:
        proc = subprocess.run(
            [str(spec.binary_path)],
            env=env,
            cwd=str(spec.binary_path.parent),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise EndToEndSmokeError(
            "smoke_trace_capture_failed",
            f"cell {spec.cell_name!r}: subprocess timed "
            f"out after {timeout_seconds}s",
        ) from exc
    if proc.returncode != 0:
        raise EndToEndSmokeError(
            "smoke_trace_capture_failed",
            f"cell {spec.cell_name!r}: subprocess exit "
            f"{proc.returncode}; stderr={proc.stderr[:500]!r}",
        )
    if not scratch_sidecar_path.exists():
        raise EndToEndSmokeError(
            "smoke_trace_capture_failed",
            f"cell {spec.cell_name!r}: sidecar not emitted "
            f"at {scratch_sidecar_path}",
        )
    durable_dir = out_dir / "traces" / spec.cell_name
    durable_dir.mkdir(parents=True, exist_ok=True)
    durable_sidecar_path = durable_dir / "tier2_address_trace.t2a"
    durable_sidecar_path.write_bytes(scratch_sidecar_path.read_bytes())
    return durable_sidecar_path


def _select_warm_aligned_frame(
    frames: Sequence[LaunchFrame], *, launch_skip_count: int,
    cell_name: str,
) -> LaunchFrame:
    """Frame index ``launch_skip_count`` is the
    NCU-measured launch under
    ``--launch-skip N --launch-count 1``. A
    ``_loop`` binary that emits exactly
    ``launch_skip_count + 1`` launches puts the
    warm-aligned frame at index
    ``launch_skip_count``.
    """
    if not frames:
        raise EndToEndSmokeError(
            "smoke_launch_index_out_of_range",
            f"cell {cell_name!r}: sidecar contained zero "
            f"launch frames",
        )
    if launch_skip_count < 0 or launch_skip_count >= len(frames):
        raise EndToEndSmokeError(
            "smoke_launch_index_out_of_range",
            f"cell {cell_name!r}: launch_skip_count "
            f"{launch_skip_count} out of [0, "
            f"{len(frames) - 1}] frames",
        )
    return frames[launch_skip_count]


def _classified_cohort(predictor_bucket: str) -> str:
    if predictor_bucket == PREDICTOR_BUCKET_FRIENDLY:
        return "predictor_friendly"
    if predictor_bucket == PREDICTOR_BUCKET_HOSTILE:
        return "predictor_hostile"
    if predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        return "predictor_typed_unavailable"
    return "unknown"


def _measure_one_cell(
    spec: SmokeCellSpec,
    *,
    out_dir: Path,
    nvbit_so: Path,
    geometry_spec: Mapping[str, Any],
) -> CapturedSmokeRecord:
    if spec.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
        return CapturedSmokeRecord(
            cell_name=spec.cell_name,
            predictor_bucket=spec.predictor_bucket,
            status=_STATUS_TYPED_UNAVAILABLE,
            ncu_l1_hit_rate=spec.ncu_l1_hit_rate,
            ncu_l2_hit_rate=spec.ncu_l2_hit_rate,
            predicted_l1_hit_rate=None,
            predicted_l2_hit_rate=None,
            l1_abs_pp_error=None,
            l2_abs_pp_error=None,
            l1_sectors_hit=None,
            l1_sectors_total=None,
            l2_sectors_hit=None,
            l2_sectors_total=None,
            launch_skip_count=spec.launch_skip_count,
            warmup_protocol=None,
            cache_initial_state=None,
            measured_frame_uid=None,
            total_frame_count=None,
            predicted_l1_calibrated=None,
            predicted_l2_calibrated=None,
            l1_calibrated_abs_pp_error=None,
            l2_calibrated_abs_pp_error=None,
            l1_rel_err=None,
            l2_rel_err=None,
            tier2_trace_path=None,
            typed_reason="predictor_typed_unavailable_cohort",
            typed_message=(
                "cell's canonical partition bucket is "
                "predictor_typed_unavailable; excluded "
                "from cohort residuals"
            ),
            sidecar_sha256=None,
            binary_sha256=spec.binary_sha256 or None,
            kernel_signature_sha256=spec.kernel_signature_sha256 or None,
        )

    try:
        sidecar_path = _capture_tier2_trace(
            spec, out_dir=out_dir, nvbit_so=nvbit_so,
        )
    except EndToEndSmokeError as exc:
        return _failed_record(spec, exc)

    try:
        frames = parse_launch_frames(sidecar_path)
    except Exception as exc:  # parser raises plain ValueError
        return _failed_record(
            spec,
            EndToEndSmokeError(
                "smoke_sidecar_parse_failed",
                f"cell {spec.cell_name!r}: parser raised "
                f"{type(exc).__name__}: {exc}",
            ),
        )

    # Validate frame count up-front so the harness can emit a
    # ``smoke_launch_index_out_of_range`` typed record instead
    # of letting the predictor's
    # ``warmup_count_out_of_range`` slip through as a generic
    # smoke_predictor_typed_unavailable. The warmed helper
    # below is the source of truth for cold/warm semantics;
    # this gate is purely the harness's pre-flight contract.
    if spec.launch_skip_count < 0 or spec.launch_skip_count >= len(frames):
        return _failed_record(
            spec,
            EndToEndSmokeError(
                "smoke_launch_index_out_of_range",
                f"cell {spec.cell_name!r}: launch_skip_count "
                f"{spec.launch_skip_count} out of [0, "
                f"{len(frames)}); cannot run warmed sequence "
                f"with {spec.launch_skip_count} warmup frames "
                f"on {len(frames)} captured frames",
            ),
        )

    # Warmed-sequence helper: replays
    # frames[0..launch_skip_count) for validation, then
    # measures frames[launch_skip_count]. The
    # ``measured_cache_initial_state`` argument is threaded
    # from the ledger entry's ``cache_initial_state`` so the
    # calibration protocol matches the prediction protocol:
    # under NCU ``--replay-mode kernel`` the value is
    # ``cold`` and the predictor instantiates fresh L1/L2
    # simulators for the measured frame (no warm replay);
    # under ``--replay-mode application`` or a paired-launch
    # warm protocol the value is ``warm`` and the predictor
    # uses the warmed simulators.
    measured_cache_initial_state = (
        spec.cache_initial_state or "warm"
    )
    result = predict_warmed_launch_sequence(
        frames,
        warmup_count=spec.launch_skip_count,
        cell_name=spec.cell_name,
        geometry_spec=geometry_spec,
        measured_cache_initial_state=measured_cache_initial_state,
    )
    if result.get("result_kind") != "scalar":
        reasons = result.get("reasons") or ()
        prov = result.get("provenance") or {}
        return CapturedSmokeRecord(
            cell_name=spec.cell_name,
            predictor_bucket=spec.predictor_bucket,
            status=_STATUS_TYPED_UNAVAILABLE,
            ncu_l1_hit_rate=spec.ncu_l1_hit_rate,
            ncu_l2_hit_rate=spec.ncu_l2_hit_rate,
            predicted_l1_hit_rate=None,
            predicted_l2_hit_rate=None,
            l1_abs_pp_error=None,
            l2_abs_pp_error=None,
            l1_sectors_hit=None,
            l1_sectors_total=None,
            l2_sectors_hit=None,
            l2_sectors_total=None,
            launch_skip_count=spec.launch_skip_count,
            warmup_protocol=prov.get("warmup_protocol"),
            cache_initial_state=prov.get("cache_initial_state"),
            measured_frame_uid=prov.get("measured_frame_uid"),
            total_frame_count=prov.get("total_frame_count"),
            predicted_l1_calibrated=None,
            predicted_l2_calibrated=None,
            l1_calibrated_abs_pp_error=None,
            l2_calibrated_abs_pp_error=None,
            l1_rel_err=None,
            l2_rel_err=None,
            tier2_trace_path=str(sidecar_path),
            typed_reason="smoke_predictor_typed_unavailable",
            typed_message=(
                f"predictor returned typed-unavailable: "
                f"reasons={list(reasons)!r}"
            ),
            sidecar_sha256=_file_sha256_hex(sidecar_path),
            binary_sha256=spec.binary_sha256 or None,
            kernel_signature_sha256=spec.kernel_signature_sha256 or None,
        )

    raw_l1 = float(result["raw_l1_hit_rate"])
    raw_l2 = float(result["raw_l2_hit_rate"])
    l1_pp = abs(raw_l1 - spec.ncu_l1_hit_rate) * 100.0
    l2_pp = abs(raw_l2 - spec.ncu_l2_hit_rate) * 100.0
    prov = result.get("provenance") or {}
    return CapturedSmokeRecord(
        cell_name=spec.cell_name,
        predictor_bucket=spec.predictor_bucket,
        status=_STATUS_MEASURED,
        ncu_l1_hit_rate=spec.ncu_l1_hit_rate,
        ncu_l2_hit_rate=spec.ncu_l2_hit_rate,
        predicted_l1_hit_rate=raw_l1,
        predicted_l2_hit_rate=raw_l2,
        l1_abs_pp_error=l1_pp,
        l2_abs_pp_error=l2_pp,
        l1_sectors_hit=int(result.get("l1_hit_count", 0)),
        l1_sectors_total=int(result.get("l1_total_count", 0)),
        l2_sectors_hit=int(result.get("l2_hit_count", 0)),
        l2_sectors_total=int(result.get("l2_total_count", 0)),
        launch_skip_count=spec.launch_skip_count,
        warmup_protocol=prov.get("warmup_protocol"),
        cache_initial_state=prov.get("cache_initial_state"),
        measured_frame_uid=prov.get("measured_frame_uid"),
        total_frame_count=prov.get("total_frame_count"),
        predicted_l1_calibrated=None,
        predicted_l2_calibrated=None,
        l1_calibrated_abs_pp_error=None,
        l2_calibrated_abs_pp_error=None,
        l1_rel_err=None,
        l2_rel_err=None,
        tier2_trace_path=str(sidecar_path),
        typed_reason=None,
        typed_message=None,
        sidecar_sha256=_file_sha256_hex(sidecar_path),
        binary_sha256=spec.binary_sha256 or None,
        kernel_signature_sha256=spec.kernel_signature_sha256 or None,
    )


def _failed_record(
    spec: SmokeCellSpec, exc: EndToEndSmokeError,
) -> CapturedSmokeRecord:
    return CapturedSmokeRecord(
        cell_name=spec.cell_name,
        predictor_bucket=spec.predictor_bucket,
        status=_STATUS_FAILED,
        ncu_l1_hit_rate=spec.ncu_l1_hit_rate,
        ncu_l2_hit_rate=spec.ncu_l2_hit_rate,
        predicted_l1_hit_rate=None,
        predicted_l2_hit_rate=None,
        l1_abs_pp_error=None,
        l2_abs_pp_error=None,
        l1_sectors_hit=None,
        l1_sectors_total=None,
        l2_sectors_hit=None,
        l2_sectors_total=None,
        launch_skip_count=spec.launch_skip_count,
        warmup_protocol=None,
        cache_initial_state=None,
        measured_frame_uid=None,
        total_frame_count=None,
        predicted_l1_calibrated=None,
        predicted_l2_calibrated=None,
        l1_calibrated_abs_pp_error=None,
        l2_calibrated_abs_pp_error=None,
        l1_rel_err=None,
        l2_rel_err=None,
        tier2_trace_path=None,
        typed_reason=exc.typed_reason,
        typed_message=exc.message,
        sidecar_sha256=None,
        binary_sha256=spec.binary_sha256 or None,
        kernel_signature_sha256=spec.kernel_signature_sha256 or None,
    )


def _aggregate_cohorts(
    records: Sequence[CapturedSmokeRecord],
) -> tuple[CohortAggregate, ...]:
    out: list[CohortAggregate] = []
    for cohort_bucket in (
        PREDICTOR_BUCKET_FRIENDLY,
        PREDICTOR_BUCKET_HOSTILE,
    ):
        cohort_records = [
            r for r in records
            if r.predictor_bucket == cohort_bucket
            and r.status == _STATUS_MEASURED
        ]
        if not cohort_records:
            out.append(
                CohortAggregate(
                    cohort=cohort_bucket,
                    cell_count=0,
                    l1_mean_abs_pp_error=None,
                    l2_mean_abs_pp_error=None,
                    l1_max_abs_pp_error=None,
                    l2_max_abs_pp_error=None,
                )
            )
            continue
        l1_errors = [r.l1_abs_pp_error for r in cohort_records]
        l2_errors = [r.l2_abs_pp_error for r in cohort_records]
        out.append(
            CohortAggregate(
                cohort=cohort_bucket,
                cell_count=len(cohort_records),
                l1_mean_abs_pp_error=sum(l1_errors) / len(l1_errors),
                l2_mean_abs_pp_error=sum(l2_errors) / len(l2_errors),
                l1_max_abs_pp_error=max(l1_errors),
                l2_max_abs_pp_error=max(l2_errors),
            )
        )
    return tuple(out)


def _git_head_sha(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def _calibration_input_from_record(
    record: CapturedSmokeRecord, spec: SmokeCellSpec,
) -> CalibrationInputRecord:
    """Build a deployed ``CalibrationInputRecord``
    from one measured smoke row + its source spec.

    Constructs the 7-tuple ``CellJoinKey`` from the
    spec's ledger-derived identity fields, then
    bundles the raw simulator predictions + NCU
    labels under the canonical predictor bucket so
    ``run_loocv`` can fit + score in one pass.
    Caller MUST guarantee ``record.status ==
    "measured"`` and the spec's identity fields
    are populated; ``_select_calibratable_records``
    enforces this.
    """
    join_key = CellJoinKey(
        slug=spec.cell_name,
        shape_point=spec.shape_point,
        binary_sha256=spec.binary_sha256,
        source_commit=spec.source_commit,
        kernel_signature_sha256=spec.kernel_signature_sha256,
        arch=spec.arch,
        ncu_metric_version=spec.ncu_metric_version,
    )
    return CalibrationInputRecord(
        cell_name=record.cell_name,
        arch=spec.arch,
        launch_uid=int(record.measured_frame_uid or 0),
        predictor_bucket=record.predictor_bucket,
        raw_l1_prediction=float(record.predicted_l1_hit_rate or 0.0),
        raw_l2_prediction=float(record.predicted_l2_hit_rate or 0.0),
        ncu_l1_label=record.ncu_l1_hit_rate,
        ncu_l2_label=record.ncu_l2_hit_rate,
        cell_join_key=join_key,
    )


def _secondary_rel_err(pred: float, label: float) -> float:
    """Secondary metric: absolute error over a label
    denominator floored at ``SECONDARY_REL_ERR_EPSILON``.
    Matches the ``tiered_gate_evaluator`` implementation
    verbatim so per-cell rel_err numbers in the artifact
    agree with the bucket-level geomean the gate evaluator
    emits.
    """
    return abs(pred - label) / max(label, SECONDARY_REL_ERR_EPSILON)


def _cohort_gate_result_to_row(
    result: CohortGateResult,
) -> GateCohortResultRow:
    return GateCohortResultRow(
        cohort=result.cohort,
        cache_level=result.cache_level,
        sample_count=result.sample_count,
        mean_abs_pp_error=result.mean_abs_pp_error,
        max_abs_pp_error=result.max_abs_pp_error,
        threshold_pp=result.threshold_pp,
        passed=result.passed,
        mean_abs_pp_error_ci_lower=result.mean_abs_pp_error_ci_lower,
        mean_abs_pp_error_ci_upper=result.mean_abs_pp_error_ci_upper,
        secondary_rel_err_geomean=result.secondary_rel_err_geomean,
        secondary_rel_err_geomean_ci_lower=(
            result.secondary_rel_err_geomean_ci_lower
        ),
        secondary_rel_err_geomean_ci_upper=(
            result.secondary_rel_err_geomean_ci_upper
        ),
        secondary_rel_err_geomean_ci_half_width=(
            result.secondary_rel_err_geomean_ci_half_width
        ),
    )


def _gate_report_to_rows(
    report: Optional[TieredGateReport],
    *,
    admitted_cell_count: int,
    calibration_typed_reason: str = "",
    calibration_typed_message: str = "",
) -> tuple[TieredGateReportRow, tuple[GateCohortResultRow, ...]]:
    """Flatten a ``TieredGateReport`` (or a
    typed-unavailable abstention) into the
    artifact-serializable
    ``(TieredGateReportRow, gate_cohort_results)``
    pair. When ``report`` is ``None`` the
    typed-unavailable surface is captured: the
    cohort tuple is empty and the row records the
    abstention reason.
    """
    if report is None:
        return (
            TieredGateReportRow(
                overall_passed=False,
                bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
                random_seed=0,
                admitted_cell_count=admitted_cell_count,
                calibration_typed_reason=calibration_typed_reason,
                calibration_typed_message=calibration_typed_message,
            ),
            (),
        )
    rows = (
        _cohort_gate_result_to_row(report.friendly_l1),
        _cohort_gate_result_to_row(report.friendly_l2),
        _cohort_gate_result_to_row(report.hostile_l1),
        _cohort_gate_result_to_row(report.hostile_l2),
    )
    return (
        TieredGateReportRow(
            overall_passed=report.overall_passed,
            bootstrap_resamples=report.bootstrap_resamples,
            bootstrap_confidence=report.bootstrap_confidence,
            random_seed=report.random_seed,
            admitted_cell_count=admitted_cell_count,
            calibration_typed_reason="",
            calibration_typed_message="",
        ),
        rows,
    )


def _count_loocv_residuals_per_cohort(
    loocv_result: LOOCVResult,
) -> tuple[int, int]:
    """Return ``(friendly_count, hostile_count)`` —
    the number of admitted LOOCV residuals whose
    canonical partition bucket is
    ``predictor_friendly`` and ``predictor_hostile``
    respectively. ``run_loocv`` already excluded
    typed-unavailable cells via ``_select_admitted``,
    so the residuals here are guaranteed to be in one
    of the two scorable buckets — but the canonical
    bucket is looked up fresh via
    ``lookup_predictor_design_coverage`` rather than
    trusting any stored bucket label, matching the
    deployed tiered-gate evaluator's discipline.
    """
    friendly_count = 0
    hostile_count = 0
    for residual in loocv_result.per_cell_residuals:
        entry = lookup_predictor_design_coverage(residual.cell_name)
        if entry.predictor_bucket == PREDICTOR_BUCKET_FRIENDLY:
            friendly_count += 1
        elif entry.predictor_bucket == PREDICTOR_BUCKET_HOSTILE:
            hostile_count += 1
    return friendly_count, hostile_count


def _enforce_per_cohort_smoke_floor(
    gate_report: TieredGateReport,
    *,
    friendly_count: int,
    hostile_count: int,
) -> tuple[Optional[TieredGateReport], str, str]:
    """Apply the per-cohort
    ``COVERAGE_FLOOR_SMOKE`` floor to
    ``gate_report``.

    The deployed evaluator's per-bucket
    ``CohortGateResult`` buckets whatever
    residuals arrive without a coverage-floor
    check — that gate lives in the calibrator
    (``run_loocv`` enforces the floor on the
    TOTAL admitted set) and again here for the
    per-cohort scalar headline.

    The abstention-discipline contract applies
    PER COHORT — friendly and hostile each must
    reach the ``COVERAGE_FLOOR_SMOKE`` admitted
    count before publishing a scalar gate
    verdict. Below the floor on either side,
    the harness emits
    ``coverage_below_smoke_floor`` instead of a
    passing scalar.

    Returns ``(filtered_report, typed_reason,
    typed_message)``:

      * ``filtered_report = None`` when EITHER
        cohort is below the floor. The caller
        emits a typed-unavailable
        ``[tiered_gate_report]`` row and
        suppresses every
        ``[[gate_cohort_result]]`` row.
      * ``filtered_report = gate_report``
        unchanged when both cohorts meet the
        floor. ``typed_reason`` /
        ``typed_message`` are empty strings.

    The message names each undercovered cohort
    + its admitted count so a downstream
    consumer can diagnose without re-running
    the harness.
    """
    undercovered: list[str] = []
    if friendly_count < COVERAGE_FLOOR_SMOKE:
        undercovered.append(
            f"{PREDICTOR_BUCKET_FRIENDLY} "
            f"admitted={friendly_count}"
        )
    if hostile_count < COVERAGE_FLOOR_SMOKE:
        undercovered.append(
            f"{PREDICTOR_BUCKET_HOSTILE} "
            f"admitted={hostile_count}"
        )
    if not undercovered:
        return gate_report, "", ""
    message = (
        f"cohort smoke floor ({COVERAGE_FLOOR_SMOKE}) "
        f"unmet on: "
        + "; ".join(undercovered)
        + (
            "; scalar gate verdict suppressed "
            "for the undercovered cohort(s) "
            "and overall_passed forced to false "
            "per the abstention-discipline "
            "contract"
        )
    )
    return None, "coverage_below_smoke_floor", message


_BUNDLE_ALGORITHM_VERSION: str = "isotonic_loocv_lower_bound_v1"
_BUNDLE_SUBDIR: str = "cache_prediction_bundles"
_BUNDLE_PREDICTION_MODE: str = "in_distribution"
_BUNDLE_CALIBRATION_SOURCE: str = "sm_86"
_BUNDLE_B200_VALIDATED: bool = False

# Tier-2 vs NCU read-sector denominator tolerance:
# the Tier-2 measured ``l2_sectors_total`` for a
# calibration cell may be smaller than the NCU
# ledger's measured denominator because the
# producer's narrow instrumentation surface does
# not cover the same scope as NCU's hardware
# counters. A ratio under this floor signals a
# materially-different denominator class — the
# calibrated head can absorb a constant bias but
# cannot honestly extrapolate from 8 sectors to
# 600. Cells whose
# ``tier2_l2_total / ncu_l2_total < _TIER2_NCU_DENOMINATOR_RATIO_FLOOR``
# route to ``trace_denominator_mismatch``.
_TIER2_NCU_DENOMINATOR_RATIO_FLOOR: float = 0.05

# Symmetric upper-bound tolerance on producer
# over-count vs the NCU denominator. The denominator
# filter is bidirectional: cells whose Tier-2
# ``l2_sectors_total`` exceeds
# ``ncu_l2_sectors_total * (1 + _TIER2_NCU_OVER_COUNT_TOLERANCE)``
# route to the same ``trace_denominator_mismatch``
# typed-abstention path the under-count case uses.
# A symmetric upper bound is required because the
# producer's L2 read-score class definition (which
# SASS opcodes / address-space classes contribute
# to the denominator) must match the calibration
# label's denominator class definition. The label
# denominator ``lts__t_sectors_srcunit_tex_op_read.sum``
# is a NARROW class on Ampere sm_86 (LSU-pipe
# source-unit traffic only — no constant-cache, no
# texture-cache, no instruction-cache); a producer
# that over-counts indicates the producer is
# instrumenting an instruction class the label
# denominator does not cover. The 10 % over-count
# tolerance absorbs minor classification noise
# (e.g., a single extra LDC sector slipping into
# the score) while still flagging material
# producer-class drift.
_TIER2_NCU_OVER_COUNT_TOLERANCE: float = 0.10


def _ci_half_width_from_loocv(
    loocv_result: LOOCVResult,
) -> float:
    """Derive a single ``ci_half_width`` fraction
    from the LOOCV residual distribution.

    Uses the 95th-percentile of |L2 residual_pp|
    (the calibration target is the L2 read-only
    sector hit-rate) divided by 100 to convert
    pp into a [0, 1] fraction matching the
    schema's
    ``predicted_hit_rate``-space contract.

    Floors at 1e-6 so a residual-perfect cohort
    still emits a non-zero half-width — the
    bundle composer's CI bounds are derived
    arithmetically from this value, and a zero
    half-width would collapse the symmetric
    interval to a point.
    """
    if not loocv_result.per_cell_residuals:
        return 0.05
    l2_residuals_pp = [
        abs(r.l2_residual_pp)
        for r in loocv_result.per_cell_residuals
    ]
    l2_residuals_pp.sort()
    idx = max(
        0,
        min(
            len(l2_residuals_pp) - 1,
            int(round(0.95 * (len(l2_residuals_pp) - 1))),
        ),
    )
    p95_pp = l2_residuals_pp[idx]
    fraction = p95_pp / 100.0
    return max(fraction, 1e-6)


def _confidence_band_summary(
    loocv_result: LOOCVResult,
    ci_half_width_fraction: float,
) -> str:
    """Compose a human-readable
    ``confidence_band`` string for the
    ``CachePredictionProvenance``.

    Reports the half-width in pp + the LOOCV
    admitted cell count so a downstream
    consumer can audit whether the confidence
    interval came from a publication-floor or
    smoke-floor cohort.
    """
    half_width_pp = ci_half_width_fraction * 100.0
    admitted = loocv_result.admitted_cell_count
    return (
        f"loocv_p95_l2_residual_pp_{half_width_pp:.2f}_"
        f"admitted_{admitted}"
    )


def _build_cell_predictor_bundle(
    record: CapturedSmokeRecord,
    spec: SmokeCellSpec,
    *,
    calibrator: IsotonicCalibrator,
    ci_half_width: float,
    out_dir: Path,
) -> PredictorBundle:
    """Build a single ``PredictorBundle`` for one
    measured smoke record.

    The launch-result dict shape mirrors what
    ``predict_warmed_launch_sequence`` returns
    for a scalar path, reconstructed from the
    smoke record's stored raw fields so the
    bundle composer's
    ``_compose_scalar_arm`` body sees the same
    inputs it would have seen on the original
    predictor call.

    The bundle's emitted
    ``InputSidecarDigest.path`` is derived from
    the verified ``record.tier2_trace_path``
    relative to ``out_dir`` so a downstream
    consumer can resolve the named path
    against the same harness output root and
    re-hash the bytes; the SHA-256 of those
    bytes must equal the bundle's recorded
    ``sector_stream`` digest. The composer
    fails closed if the trace file is missing,
    is outside ``out_dir``, or its bytes do
    not match ``record.sidecar_sha256``.
    """
    launch_result = {
        "result_kind": "scalar",
        "raw_l1_hit_rate": float(record.predicted_l1_hit_rate or 0.0),
        "raw_l2_hit_rate": float(record.predicted_l2_hit_rate or 0.0),
        "l1_hit_count": int(record.l1_sectors_hit or 0),
        "l1_total_count": int(record.l1_sectors_total or 0),
        "l2_hit_count": int(record.l2_sectors_hit or 0),
        "l2_total_count": int(record.l2_sectors_total or 0),
        "provenance": {"cell_name": record.cell_name},
    }
    pairing_index = ManifestPairingIndex(
        slug=record.cell_name,
        shape_point=spec.shape_point,
        binary_sha256=spec.binary_sha256,
        source_commit=spec.source_commit,
        kernel_signature_sha256=spec.kernel_signature_sha256,
    )
    # Byte-verify the durable per-cell trace
    # path. SHA-256 the file at the record's
    # stored absolute path and assert equality
    # with the recorded ``sidecar_sha256``. A
    # mismatch (or a missing file, or an unset
    # ``tier2_trace_path``) raises
    # ``EndToEndSmokeError(smoke_sidecar_digest_mismatch, ...)``
    # so the fail-closed wrapper collapses the
    # bundle emission to a typed-unavailable
    # surface row. The bundle records a
    # relative path DERIVED from the verified
    # absolute path against ``out_dir`` — not a
    # hard-coded ``traces/<cell>/...`` string —
    # so the named path is bound to the actual
    # bytes that just passed the digest check.
    # The framework-contract validator rejects
    # absolute paths + ``..`` traversal so the
    # ``relative_to(out_dir)`` shape is what
    # lives in ``InputSidecarDigest.path``.
    if (
        record.tier2_trace_path is None
        or not record.sidecar_sha256
    ):
        raise EndToEndSmokeError(
            "smoke_sidecar_digest_mismatch",
            f"cell {record.cell_name!r}: "
            f"tier2_trace_path or sidecar_sha256 missing "
            f"on the measured record; cannot byte-verify "
            f"the bundle's sector_stream digest provenance",
        )
    trace_path = Path(record.tier2_trace_path)
    if not trace_path.is_file():
        raise EndToEndSmokeError(
            "smoke_sidecar_digest_mismatch",
            f"cell {record.cell_name!r}: the durable "
            f"per-cell trace path "
            f"{record.tier2_trace_path!r} does not "
            f"exist on disk; cannot byte-verify the "
            f"bundle's sector_stream digest provenance",
        )
    resolved_trace = trace_path.resolve()
    resolved_root = out_dir.resolve()
    try:
        derived_relative = resolved_trace.relative_to(resolved_root)
    except ValueError as exc:
        raise EndToEndSmokeError(
            "smoke_sidecar_digest_mismatch",
            f"cell {record.cell_name!r}: trace file "
            f"{record.tier2_trace_path!r} resolves to "
            f"{str(resolved_trace)!r}, which is outside "
            f"the harness output root "
            f"{str(resolved_root)!r}; the bundle "
            f"composer refuses to emit an "
            f"InputSidecarDigest naming a path the "
            f"downstream consumer cannot resolve against "
            f"the artifact bundle: {exc}",
        )
    on_disk_digest = _file_sha256_hex(trace_path)
    if on_disk_digest != record.sidecar_sha256:
        raise EndToEndSmokeError(
            "smoke_sidecar_digest_mismatch",
            f"cell {record.cell_name!r}: file at "
            f"{record.tier2_trace_path!r} hashed to "
            f"{on_disk_digest!r} but the smoke "
            f"record's recorded sidecar_sha256 is "
            f"{record.sidecar_sha256!r}; the named "
            f"sector_stream bytes do not match the "
            f"bundle's digest claim",
        )
    sidecar_digest = InputSidecarDigest(
        name="sector_stream",
        path=str(derived_relative),
        sha256=record.sidecar_sha256,
    )
    return compose_predictor_bundle(
        launch_result,
        manifest_pairing_index=pairing_index,
        algorithm_version=_BUNDLE_ALGORITHM_VERSION,
        input_sidecar_digests=(sidecar_digest,),
        predictor_design_coverage=record.predictor_bucket,
        calibrator=calibrator,
        ci_half_width=ci_half_width,
        ci_confidence=0.95,
    )


def _surface_entry_to_row(
    entry: CachePredictionSurfaceEntry,
    *,
    bundles_written: int,
) -> CachePredictionSurfaceEntryRow:
    """Flatten a ``CachePredictionSurfaceEntry``
    + its bundle count to a TOML-serializable
    row.
    """
    return CachePredictionSurfaceEntryRow(
        status=entry.status,
        prediction_mode=entry.provenance.prediction_mode,
        calibration_source=entry.provenance.calibration_source,
        confidence_band=entry.provenance.confidence_band,
        b200_validated=entry.provenance.b200_validated,
        artifact_path=entry.artifact_path,
        command=entry.command,
        waiver_reason=entry.waiver_reason,
        pairing_slug=entry.manifest_pairing_index.slug,
        pairing_shape_point=entry.manifest_pairing_index.shape_point,
        pairing_binary_sha256=(
            entry.manifest_pairing_index.binary_sha256
        ),
        pairing_source_commit=(
            entry.manifest_pairing_index.source_commit
        ),
        pairing_kernel_signature_sha256=(
            entry.manifest_pairing_index.kernel_signature_sha256
        ),
        bundles_written=bundles_written,
    )


def _abstention_surface_entry_row(
    *,
    waiver_reason: str,
) -> CachePredictionSurfaceEntryRow:
    """Build a ``status=unavailable`` surface
    entry row for the abstention path (per-
    cohort floor unmet OR no calibratable cells).

    Pairing-index fields are empty strings: the
    deployed validator only enforces the
    canonical 5-field shape on
    ``status=ok`` entries; on the abstention
    surface the pairing index is not
    semantically bound to any single bundle.
    """
    return CachePredictionSurfaceEntryRow(
        status="unavailable",
        prediction_mode=_BUNDLE_PREDICTION_MODE,
        calibration_source=_BUNDLE_CALIBRATION_SOURCE,
        confidence_band="",
        b200_validated=_BUNDLE_B200_VALIDATED,
        artifact_path=None,
        command=None,
        waiver_reason=waiver_reason,
        pairing_slug="",
        pairing_shape_point="",
        pairing_binary_sha256="",
        pairing_source_commit="",
        pairing_kernel_signature_sha256="",
        bundles_written=0,
    )


def _check_trace_denominator_consistency(
    record: CapturedSmokeRecord,
    spec: SmokeCellSpec,
) -> bool:
    """Return ``True`` iff the Tier-2-measured
    ``l2_sectors_total`` for the cell is in the
    same denominator class as the NCU ledger's
    ``ncu_l2_sectors_total`` for the cell.

    The check is BIDIRECTIONAL:

      * under-count: the ratio
        ``tier2_l2_total / ncu_l2_total`` must be
        at least
        ``_TIER2_NCU_DENOMINATOR_RATIO_FLOOR``
        (5 % by default). Below that the trace
        surface is too narrow to honestly score
        against the NCU label denominator — the
        calibrated head can absorb a constant
        bias but cannot extrapolate from a
        5 %-coverage trace to the real
        distribution of cache behaviour the
        label was measured over.
      * over-count: the Tier-2 total must not
        exceed
        ``ncu_l2_total * (1 + _TIER2_NCU_OVER_COUNT_TOLERANCE)``.
        Producer over-count signals an
        instruction class the label denominator
        does not cover (e.g., constant-cache
        traffic the producer instruments but the
        ``srcunit_tex_op_read`` label does not
        count). A 10 % tolerance absorbs minor
        classification noise while still
        flagging material class drift.

    Returns ``False`` when:

      * the record is non-measured (no Tier-2
        total to compare),
      * the NCU denominator is zero or missing
        (no honest comparison possible),
      * the ratio is below the under-count
        floor,
      * the producer total exceeds the upper
        over-count bound.

    A ``True`` return means the cell may
    proceed to bundle composition + scalar
    publication; a ``False`` return must be
    routed to ``trace_denominator_mismatch``
    typed abstention by the caller.
    """
    if record.status != _STATUS_MEASURED:
        return False
    tier2_total = record.l2_sectors_total
    ncu_total = spec.ncu_l2_sectors_total
    if tier2_total is None or ncu_total is None:
        return False
    if ncu_total <= 0:
        return False
    if tier2_total <= 0:
        return False
    upper_bound = float(ncu_total) * (
        1.0 + _TIER2_NCU_OVER_COUNT_TOLERANCE
    )
    if float(tier2_total) > upper_bound:
        return False
    ratio = float(tier2_total) / float(ncu_total)
    return ratio >= _TIER2_NCU_DENOMINATOR_RATIO_FLOOR


def _emit_cache_prediction_surface_for_admitted_cohort(
    records: Sequence[CapturedSmokeRecord],
    spec_by_cell: Mapping[str, SmokeCellSpec],
    calibration_inputs: Sequence[CalibrationInputRecord],
    loocv_result: LOOCVResult,
    *,
    bundle_dir: Path,
    out_dir: Path,
) -> tuple[CachePredictionSurfaceEntryRow, ...]:
    """Build per-cell ``PredictorBundle`` objects
    from the admitted measured smoke records,
    write each through
    ``write_cache_prediction_sidecar`` into a
    per-cell bundle subdirectory, emit one
    registered surface entry per cell via
    ``emit_cache_prediction_surface_entry``,
    and return the flattened rows for inclusion
    in the smoke report.

    The deployed surface contract requires all
    bundles in one entry to share a
    ``ManifestPairingIndex`` (each entry is
    per-baseline). Each calibration cell is a
    distinct baseline (different binary +
    kernel + shape_point), so the harness
    emits N entries — one per admitted cell —
    instead of attempting to pack the cohort
    into a single entry.

    A single ``IsotonicCalibrator`` is fit on
    the entire admitted set (vs the LOOCV's
    leave-one-out folds), so every per-cell
    bundle's calibrated prediction comes from
    one consistent fitted head. The per-cell
    bundle directory is
    ``bundle_dir / <cell_name>``; the surface
    entry's ``artifact_path`` is the canonical
    ``cache_prediction/predictions.jsonl``
    relative to that per-cell directory.

    Surface-layer errors on any single cell
    propagate as
    ``CachePredictionSurfaceError`` from the
    helper; the caller catches them and routes
    the WHOLE cohort to the abstention row
    with the typed reason. A future iteration
    could degrade per-cell instead of
    cohort-wide; for the smoke harness's
    bundle-leg-first contract, cohort-wide
    abstention on any per-cell failure is the
    honest signal.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    fitted_calibrator = IsotonicCalibrator()
    fitted_calibrator.fit(calibration_inputs)
    ci_half_width = _ci_half_width_from_loocv(loocv_result)
    confidence_band = _confidence_band_summary(
        loocv_result, ci_half_width,
    )
    admitted_cell_names = {
        r.cell_name for r in loocv_result.per_cell_residuals
    }
    # Denominator-consistency filtering happens
    # UPSTREAM in ``run_end_to_end_smoke`` BEFORE
    # ``calibration_inputs`` is built — so any
    # cell that reaches this helper inside
    # ``admitted_cell_names`` has already passed
    # the denominator gate. No re-check here.
    rows: list[CachePredictionSurfaceEntryRow] = []
    for record in records:
        if record.status != _STATUS_MEASURED:
            continue
        if record.cell_name not in admitted_cell_names:
            continue
        spec = spec_by_cell.get(record.cell_name)
        if spec is None:
            continue
        per_cell_bundle_dir = bundle_dir / record.cell_name
        bundle = _build_cell_predictor_bundle(
            record, spec,
            calibrator=fitted_calibrator,
            ci_half_width=ci_half_width,
            out_dir=out_dir,
        )
        entry = emit_cache_prediction_surface_entry(
            [bundle],
            per_cell_bundle_dir,
            prediction_mode=_BUNDLE_PREDICTION_MODE,
            calibration_source=_BUNDLE_CALIBRATION_SOURCE,
            confidence_band=confidence_band,
            b200_validated=_BUNDLE_B200_VALIDATED,
        )
        rows.append(_surface_entry_to_row(entry, bundles_written=1))
    return tuple(rows)


def _denominator_mismatched_surface_rows(
    records: Sequence[CapturedSmokeRecord],
    spec_by_cell: Mapping[str, SmokeCellSpec],
    denominator_mismatched_cell_names: set,
) -> tuple[CachePredictionSurfaceEntryRow, ...]:
    """Emit one ``status="unavailable"`` row per
    denominator-mismatched measured cell.

    These rows precede the publication-path rows
    in the smoke artifact's
    ``[[cache_prediction_surface_entry]]``
    array — they document per-cell why specific
    cells were filtered out of the
    ``calibration_inputs`` set BEFORE LOOCV /
    gate evaluation. A downstream consumer
    seeing
    ``waiver_reason="trace_denominator_mismatch: ..."``
    knows that cell never entered the scalar
    publication chain, regardless of whether
    the cohort-level abstention row that
    follows reports
    ``coverage_below_smoke_floor`` or
    ``tiered_gate_overall_failed`` or
    ``status="ok"`` for the remaining cells.
    """
    rows: list[CachePredictionSurfaceEntryRow] = []
    for record in records:
        if record.cell_name not in denominator_mismatched_cell_names:
            continue
        spec = spec_by_cell.get(record.cell_name)
        rows.append(
            _abstention_surface_entry_row(
                waiver_reason=(
                    f"trace_denominator_mismatch: "
                    f"cell {record.cell_name!r} Tier-2 "
                    f"l2_sectors_total="
                    f"{record.l2_sectors_total} vs NCU "
                    f"ncu_l2_sectors_total="
                    f"{spec.ncu_l2_sectors_total if spec else None} "
                    f"outside bidirectional bounds "
                    f"(under-count ratio floor "
                    f"{_TIER2_NCU_DENOMINATOR_RATIO_FLOOR}, "
                    f"over-count tolerance "
                    f"{_TIER2_NCU_OVER_COUNT_TOLERANCE})"
                ),
            )
        )
    return tuple(rows)


def _audit_unavailable_row(
    cell_name: str,
    ncu_total: int,
    typed_reason: str,
) -> L2DenominatorAuditRow:
    """Build a typed-unavailable audit row for a measured
    cell that cannot be audited. The row carries the
    canonical typed_reason + the ledger's recorded
    denominator so a downstream consumer can still see the
    cell's nominal denominator size; the per-class
    breakdown is empty and ``tier2_l2_sectors_total=0``.
    """
    gap = ncu_total
    gap_ratio = 1.0 if ncu_total > 0 else 0.0
    return L2DenominatorAuditRow(
        cell_name=cell_name,
        tier2_l2_sectors_total=0,
        ncu_l2_sectors_total=ncu_total,
        denominator_gap=gap,
        denominator_gap_ratio=gap_ratio,
        per_class_breakdown=(),
        audit_status=_L2_DENOMINATOR_AUDIT_STATUS_UNAVAILABLE,
        typed_reason=typed_reason,
    )


def _build_l2_denominator_audit_rows(
    records: Sequence[CapturedSmokeRecord],
    spec_by_cell: Mapping[str, SmokeCellSpec],
) -> tuple[L2DenominatorAuditRow, ...]:
    """Build a per-cell L2-read-denominator audit row for
    every measured ``CapturedSmokeRecord``. The audit
    re-parses the durable per-cell trace, indexes the
    measured frame at ``spec.launch_skip_count``, and
    groups the routed compact records by ``(op_kind,
    address_space, l1_access_policy)`` via
    ``audit_l2_read_denominator``.

    The audit is fail-closed: every measured cell produces
    EXACTLY ONE audit row. On the ok path the row carries
    ``audit_status="ok"`` + the producer total + per-class
    breakdown. On any failure path (missing trace path,
    unparseable trace, frame index out of range, zero
    per-class rows after routing) the row carries
    ``audit_status="l2_denominator_audit_unavailable"`` +
    the canonical typed_reason + an empty per-class
    breakdown + ``tier2_l2_sectors_total=0`` (so the
    gap-ratio still records the cell's nominal denominator
    size for the downstream consumer).

    Non-measured records (typed-unavailable / failed) are
    skipped because there is nothing to audit when no
    trace bytes were captured; only ``status="measured"``
    cells produce audit rows.
    """
    rows: list[L2DenominatorAuditRow] = []
    for record in records:
        if record.status != _STATUS_MEASURED:
            continue
        spec = spec_by_cell.get(record.cell_name)
        ncu_total = (
            int(spec.ncu_l2_sectors_total) if spec is not None else 0
        )
        if record.tier2_trace_path is None:
            rows.append(_audit_unavailable_row(
                record.cell_name, ncu_total,
                L2_DENOMINATOR_AUDIT_REASON_TRACE_PATH_MISSING,
            ))
            continue
        trace_path = Path(record.tier2_trace_path)
        if not trace_path.is_file():
            rows.append(_audit_unavailable_row(
                record.cell_name, ncu_total,
                L2_DENOMINATOR_AUDIT_REASON_TRACE_PATH_MISSING,
            ))
            continue
        if spec is None:
            # No spec means no ledger denominator to compare
            # against; the cell cannot be audited.
            rows.append(_audit_unavailable_row(
                record.cell_name, 0,
                L2_DENOMINATOR_AUDIT_REASON_NO_CLASS_ROWS,
            ))
            continue
        try:
            frames = parse_launch_frames(trace_path)
        except (ValueError, Tier2ParserError, OSError):
            rows.append(_audit_unavailable_row(
                record.cell_name, ncu_total,
                L2_DENOMINATOR_AUDIT_REASON_TRACE_UNPARSEABLE,
            ))
            continue
        if not frames or spec.launch_skip_count >= len(frames):
            rows.append(_audit_unavailable_row(
                record.cell_name, ncu_total,
                L2_DENOMINATOR_AUDIT_REASON_FRAME_INDEX_OUT_OF_RANGE,
            ))
            continue
        measured_frame = frames[spec.launch_skip_count]
        per_class_totals = audit_l2_read_denominator(
            measured_frame.records,
        )
        if not per_class_totals:
            rows.append(_audit_unavailable_row(
                record.cell_name, ncu_total,
                L2_DENOMINATOR_AUDIT_REASON_NO_CLASS_ROWS,
            ))
            continue
        # Per-class breakdown: routing-only counts grouped by
        # (op_kind, address_space, l1_access_policy). This is
        # informational -- it shows which routing classes
        # contributed to the L2-read-eligible pool BEFORE the
        # coupled-hierarchy simulator filtered the stream to
        # L1-miss + direct-L2 sectors. Store classes always
        # show ``sectors=0`` here because
        # ``cache_access_policy.route_access`` excludes them
        # from the L2 read score upstream.
        per_class_rows = tuple(
            L2DenominatorAuditClassRow(
                op_kind=op_kind,
                address_space=address_space,
                l1_access_policy=l1_access_policy,
                sectors=sectors,
            )
            for (op_kind, address_space, l1_access_policy), sectors
            in sorted(per_class_totals.items())
        )
        # The cell-level audit total compares NCU's L2
        # read-only sector denominator to the SIMULATED L2
        # read-lookup total (sectors that actually probed L2
        # via coupled L1 -> L2 hierarchy: L1 misses + direct-L2
        # routes). Earlier rounds compared NCU against the raw
        # ``audit_l2_read_denominator`` candidate count, which
        # included L1 hits and therefore over-counted by the
        # L1 hit population. The simulator-side total mirrors
        # NCU's ``lts__t_sectors_srcunit_tex_op_read.sum``
        # semantics: only sectors that reach LSU-pipe ->
        # tex source unit contribute to the L2 read
        # denominator. The smoke record's
        # ``l2_sectors_total`` field carries that exact
        # post-coupled-scoring count.
        producer_total = (
            record.l2_sectors_total
            if record.l2_sectors_total is not None
            else 0
        )
        gap = ncu_total - producer_total
        gap_ratio = (
            float(gap) / float(ncu_total)
            if ncu_total > 0
            else 0.0
        )
        rows.append(
            L2DenominatorAuditRow(
                cell_name=record.cell_name,
                tier2_l2_sectors_total=producer_total,
                ncu_l2_sectors_total=ncu_total,
                denominator_gap=gap,
                denominator_gap_ratio=gap_ratio,
                per_class_breakdown=per_class_rows,
                audit_status=_L2_DENOMINATOR_AUDIT_STATUS_OK,
                typed_reason=L2_DENOMINATOR_AUDIT_REASON_OK,
            )
        )
    return tuple(rows)


def _apply_loocv_to_records(
    records: Sequence[CapturedSmokeRecord],
    loocv_result: LOOCVResult,
) -> tuple[CapturedSmokeRecord, ...]:
    """Rewrite each ``CapturedSmokeRecord`` whose
    cell appears in the LOOCV per-cell residuals
    with the calibrated L1/L2 predictions, the abs
    pp errors against the held-out NCU labels, and
    the secondary rel-err. Non-residual rows
    (typed-unavailable / failed / non-admitted)
    pass through untouched.
    """
    residual_by_cell = {
        r.cell_name: r for r in loocv_result.per_cell_residuals
    }
    out: list[CapturedSmokeRecord] = []
    for record in records:
        residual = residual_by_cell.get(record.cell_name)
        if residual is None:
            out.append(record)
            continue
        rel_l1 = _secondary_rel_err(
            residual.predicted_l1_calibrated, residual.ncu_l1_label,
        )
        rel_l2 = _secondary_rel_err(
            residual.predicted_l2_calibrated, residual.ncu_l2_label,
        )
        out.append(
            dataclasses.replace(
                record,
                predicted_l1_calibrated=residual.predicted_l1_calibrated,
                predicted_l2_calibrated=residual.predicted_l2_calibrated,
                l1_calibrated_abs_pp_error=residual.l1_residual_pp,
                l2_calibrated_abs_pp_error=residual.l2_residual_pp,
                l1_rel_err=rel_l1,
                l2_rel_err=rel_l2,
            )
        )
    return tuple(out)


def run_end_to_end_smoke(
    cells: Iterable[SmokeCellSpec],
    *,
    out_dir: Path,
    nvbit_so: Path = _DEFAULT_NVBIT_SO,
    ledger_path: Path = _DEFAULT_LEDGER_PATH,
    host_arch: str = "sm_86",
) -> EndToEndSmokeReport:
    """Drive the smoke harness across the given
    cells. Returns the aggregated report.

    The caller passes already-resolved
    ``SmokeCellSpec`` records (see
    ``build_smoke_specs_from_ledger`` below for
    the canonical resolver). The harness runs the
    measurement loop, then fits the deployed
    ``IsotonicCalibrator`` against the admitted
    measured cohort via LOOCV and evaluates the
    tiered gate. The returned report carries both
    raw simulator residuals (per-cell) and
    calibrated residuals + per-cohort gate verdicts
    with the deployed bootstrap-CI parameters
    (``BOOTSTRAP_RESAMPLES=2000``,
    ``BOOTSTRAP_CONFIDENCE=0.95``). When the
    admitted-cell count falls below the smoke
    floor, the tiered-gate surface emits the
    typed reason ``coverage_below_smoke_floor``
    in lieu of cohort gate results.
    """
    _validate_iterable("cells", cells)
    cells_seq = tuple(cells)
    if not cells_seq:
        raise EndToEndSmokeError(
            "smoke_input_invalid",
            "cells must be non-empty",
        )
    geometry_spec, _evidence = resolve_cache_geometry(
        arch=host_arch, cuda_l2_cache_size_bytes=None,
    )
    records: list[CapturedSmokeRecord] = []
    spec_by_cell: dict[str, SmokeCellSpec] = {}
    for spec in cells_seq:
        spec_by_cell[spec.cell_name] = spec
        records.append(
            _measure_one_cell(
                spec,
                out_dir=out_dir,
                nvbit_so=nvbit_so,
                geometry_spec=geometry_spec,
            )
        )
    # Denominator-consistency partition BEFORE
    # LOOCV / gate evaluation. Cells whose Tier-2
    # ``l2_sectors_total`` falls below
    # ``_TIER2_NCU_DENOMINATOR_RATIO_FLOOR`` *
    # NCU ``ncu_l2_sectors_total`` cannot be
    # honestly scored — the calibrator absorbs a
    # constant bias only, not the variance over a
    # ledger denominator the trace surface does
    # not cover. Mismatched cells are filtered
    # out of ``calibration_inputs`` so they never
    # enter ``run_loocv``, ``evaluate_tiered_gate``,
    # or the ``[[gate_cohort_result]]`` rows.
    # Each filtered cell emits a per-cell
    # ``status="unavailable"`` surface row with
    # ``waiver_reason="trace_denominator_mismatch"``
    # downstream.
    denominator_mismatched_cell_names: set[str] = set()
    calibration_inputs: list[CalibrationInputRecord] = []
    for record in records:
        if record.status != _STATUS_MEASURED:
            continue
        spec = spec_by_cell.get(record.cell_name)
        if spec is None or not spec.arch or not spec.source_commit:
            continue
        if not _check_trace_denominator_consistency(record, spec):
            denominator_mismatched_cell_names.add(record.cell_name)
            continue
        calibration_inputs.append(
            _calibration_input_from_record(record, spec),
        )
    admitted_cell_count = len(calibration_inputs)
    denominator_filtered_count = len(denominator_mismatched_cell_names)
    denominator_message_fragment = (
        f"trace_denominator_mismatch filtered="
        f"{denominator_filtered_count}; "
        f"admitted_after_denominator_filter="
        f"{admitted_cell_count}"
        if denominator_filtered_count > 0
        else ""
    )
    loocv_result: Optional[LOOCVResult] = None
    gate_report: Optional[TieredGateReport] = None
    calibration_typed_reason = ""
    calibration_typed_message = ""
    try:
        if calibration_inputs:
            loocv_result = run_loocv(calibration_inputs)
            gate_report = evaluate_tiered_gate(loocv_result)
        else:
            calibration_typed_reason = "coverage_below_smoke_floor"
            base_message = (
                "no calibratable measured cells; the "
                "smoke harness emitted zero rows with "
                "status=measured AND fully-populated "
                "spec identity"
            )
            calibration_typed_message = (
                f"{base_message}; {denominator_message_fragment}"
                if denominator_message_fragment
                else base_message
            )
    except IsotonicCalibratorError as exc:
        calibration_typed_reason = exc.typed_reason
        calibration_typed_message = (
            f"{exc.message}; {denominator_message_fragment}"
            if denominator_message_fragment
            else exc.message
        )
    except TieredGateEvaluatorError as exc:
        calibration_typed_reason = exc.typed_reason
        calibration_typed_message = (
            f"{exc.message}; {denominator_message_fragment}"
            if denominator_message_fragment
            else exc.message
        )
    if loocv_result is not None:
        records = list(_apply_loocv_to_records(records, loocv_result))
    # Per-cohort smoke-floor enforcement: the deployed
    # ``run_loocv`` already enforced the floor on the
    # TOTAL admitted set, but the scalar gate verdict
    # is published per cohort and must abstain when
    # either cohort lacks ``COVERAGE_FLOOR_SMOKE``
    # admitted residuals — otherwise the report
    # publishes accuracy-by-undercoverage on the
    # smaller side.
    per_cohort_floor_unmet = False
    if gate_report is not None and loocv_result is not None:
        friendly_count, hostile_count = (
            _count_loocv_residuals_per_cohort(loocv_result)
        )
        filtered_report, floor_typed_reason, floor_typed_message = (
            _enforce_per_cohort_smoke_floor(
                gate_report,
                friendly_count=friendly_count,
                hostile_count=hostile_count,
            )
        )
        if filtered_report is None:
            # Cohort floor violated: surface the typed
            # abstention reason + message verbatim. When
            # denominator filtering removed cells upstream
            # of LOOCV, append the deterministic fragment
            # naming the denominator cause so a downstream
            # consumer can attribute the floor failure to
            # the upstream filter without scanning surface
            # entry rows.
            gate_report = None
            calibration_typed_reason = floor_typed_reason
            calibration_typed_message = (
                f"{floor_typed_message}; "
                f"{denominator_message_fragment}"
                if denominator_message_fragment
                else floor_typed_message
            )
            per_cohort_floor_unmet = True
    aggregates = _aggregate_cohorts(records)
    gate_row, gate_cohort_results = _gate_report_to_rows(
        gate_report,
        admitted_cell_count=admitted_cell_count,
        calibration_typed_reason=calibration_typed_reason,
        calibration_typed_message=calibration_typed_message,
    )
    # Compose + persist per-cell ``PredictorBundle``
    # objects through the registered
    # ``cache_prediction`` surface helpers, but only
    # when the calibrated gate has cleared every
    # discipline check (LOOCV admitted + per-cohort
    # floor). On any abstention path the surface
    # entries collapse to a single
    # ``status=unavailable`` row with the canonical
    # typed reason so a downstream consumer sees an
    # honest no-publication signal without scanning
    # for missing artifacts.
    # Per-cell denominator-mismatch unavailable rows
    # are independent of the cohort-level
    # publication-eligibility branch: they
    # document upstream filtering that already
    # excluded cells from the LOOCV / gate path,
    # so they prepend the publication-path rows
    # regardless of which branch follows.
    denominator_rows = _denominator_mismatched_surface_rows(
        records, spec_by_cell, denominator_mismatched_cell_names,
    )
    # Gate-aware publication discipline: an
    # ok cache_prediction entry requires (a) LOOCV
    # admitted, (b) per-cohort smoke floor met,
    # (c) tiered gate exists, AND (d) the gate's
    # ``overall_passed`` flag is True. A failing
    # tiered gate on a cohort that meets the
    # smoke floor must still abstain — publishing
    # under a failing gate contradicts the
    # fail-closed tiered-accuracy-gate contract.
    publication_eligible = (
        loocv_result is not None
        and not per_cohort_floor_unmet
        and gate_report is not None
        and gate_report.overall_passed
    )
    publication_path_rows: tuple[
        CachePredictionSurfaceEntryRow, ...
    ] = ()
    if publication_eligible:
        bundle_dir = out_dir / _BUNDLE_SUBDIR
        # Fail-closed wrapper: ANY exception from the
        # calibrator fit, bundle composition,
        # sidecar write, surface emission, or
        # framework-contract validator collapses
        # the cohort to a single
        # ``status="unavailable"`` row with the
        # typed reason. The deployed helpers raise
        # four typed-error classes plus possibly a
        # bare ``Exception`` on unanticipated
        # paths; the wrapper catches all of them
        # to honour the fail-closed publication
        # contract.
        try:
            publication_path_rows = (
                _emit_cache_prediction_surface_for_admitted_cohort(
                    records,
                    spec_by_cell,
                    calibration_inputs,
                    loocv_result,
                    bundle_dir=bundle_dir,
                    out_dir=out_dir,
                )
            )
        except (
            IsotonicCalibratorError,
            PredictorBundleSchemaError,
            FrameworkContractError,
            CachePredictionSurfaceError,
            EndToEndSmokeError,
        ) as exc:
            typed_reason = getattr(exc, "typed_reason", "")
            message = getattr(exc, "message", str(exc))
            waiver = (
                f"{typed_reason}: {message}"
                if typed_reason
                else str(exc)
            )
            publication_path_rows = (
                _abstention_surface_entry_row(
                    waiver_reason=waiver,
                ),
            )
    elif (
        gate_report is not None
        and not gate_report.overall_passed
        and not per_cohort_floor_unmet
    ):
        # Cohorts cleared the smoke floor but the
        # tiered gate's per-cohort × cache-level
        # checks fired; publication discipline
        # forbids emitting status=ok against a
        # failing gate.
        publication_path_rows = (
            _abstention_surface_entry_row(
                waiver_reason="tiered_gate_overall_failed",
            ),
        )
    else:
        publication_path_rows = (
            _abstention_surface_entry_row(
                waiver_reason=(
                    calibration_typed_reason
                    or "calibration_step_did_not_complete"
                ),
            ),
        )
    cache_prediction_surface_entries = (
        denominator_rows + publication_path_rows
    )
    l2_denominator_audit = _build_l2_denominator_audit_rows(
        records, spec_by_cell,
    )
    trace_budget_env = os.environ.get(
        "METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES",
    )
    trace_capture_timeout_env = os.environ.get(
        "METRIC_EXTRACTOR_TIER2_CAPTURE_TIMEOUT_SECONDS",
    )
    try:
        trace_capture_timeout_value = (
            int(trace_capture_timeout_env)
            if trace_capture_timeout_env
            else _DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS
        )
    except ValueError:
        trace_capture_timeout_value = (
            _DEFAULT_TIER2_CAPTURE_TIMEOUT_SECONDS
        )
    return EndToEndSmokeReport(
        schema_version="1.4",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        host_arch=host_arch,
        ledger_path=str(ledger_path),
        nvbit_so_sha256=_file_sha256_hex(nvbit_so),
        commit_sha=_git_head_sha(_REPO_ROOT),
        trace_budget_bytes=(
            trace_budget_env if trace_budget_env else "default"
        ),
        trace_capture_timeout_seconds=trace_capture_timeout_value,
        per_cell_records=tuple(records),
        cohort_aggregates=aggregates,
        tiered_gate_report=gate_row,
        gate_cohort_results=gate_cohort_results,
        cache_prediction_surface_entries=(
            cache_prediction_surface_entries
        ),
        l2_denominator_audit=l2_denominator_audit,
    )


def build_smoke_specs_from_ledger(
    ledger: LabelLedger,
    *,
    cell_names: Optional[Sequence[str]] = None,
    binary_search_dir = _DEFAULT_CALIBRATION_BIN_DIRS,
) -> tuple[SmokeCellSpec, ...]:
    """Convert a loaded ``LabelLedger`` to a tuple
    of ``SmokeCellSpec``.

    ``binary_search_dir`` accepts either a single
    :class:`Path` (legacy single-directory mode) or
    an iterable of directories. The default scans
    both `tests/calibration` and
    `tests/calibration_cohort_b` so ledger entries
    whose binary lives in either calibration root
    resolve uniformly. Iteration order is
    deterministic; the first directory whose entry
    matches the recorded SHA-256 wins.

    If ``cell_names`` is provided, only those cells
    are returned (in the provided order). Otherwise
    every ledger entry is included. Entries whose
    binary cannot be located by sha256 are skipped
    silently (the per-cell failure surfaces during
    execution if a caller explicitly requested the
    cell).
    """
    spec_map: dict[str, SmokeCellSpec] = {}
    for entry in ledger.entries:
        try:
            spec_map[entry.cell_name] = _ledger_entry_to_smoke_spec(
                entry, search_dir=binary_search_dir,
            )
        except EndToEndSmokeError:
            continue
    if cell_names is None:
        return tuple(spec_map[name] for name in sorted(spec_map.keys()))
    out: list[SmokeCellSpec] = []
    for name in cell_names:
        if name in spec_map:
            out.append(spec_map[name])
    return tuple(out)


def _toml_escape(value: str) -> str:
    return (
        '"'
        + value.replace("\\", "\\\\").replace('"', '\\"')
        + '"'
    )


def _toml_value(value: Any) -> str:
    if value is None:
        return '"null"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_escape(value)
    raise EndToEndSmokeError(
        "smoke_input_invalid",
        f"unserializable TOML value: type={type(value).__name__}",
    )


def serialize_smoke_report_to_toml(
    report: EndToEndSmokeReport,
) -> str:
    """Render the smoke report as a TOML string
    with a ``[meta]`` block, one ``[[per_cell]]``
    table per record, and one
    ``[[cohort_aggregate]]`` table per cohort.
    """
    lines: list[str] = []
    lines.append("[meta]")
    lines.append(f"schema_version = {_toml_escape(report.schema_version)}")
    lines.append(f"generated_at_utc = {_toml_escape(report.generated_at_utc)}")
    lines.append(f"host_arch = {_toml_escape(report.host_arch)}")
    lines.append(f"ledger_path = {_toml_escape(report.ledger_path)}")
    lines.append(f"nvbit_so_sha256 = {_toml_escape(report.nvbit_so_sha256)}")
    lines.append(f"commit_sha = {_toml_escape(report.commit_sha)}")
    lines.append(
        f"trace_budget_bytes = {_toml_escape(report.trace_budget_bytes)}"
    )
    lines.append(
        f"trace_capture_timeout_seconds = "
        f"{report.trace_capture_timeout_seconds}"
    )
    lines.append("")
    for record in report.per_cell_records:
        lines.append("[[per_cell]]")
        for field_obj in dataclasses.fields(record):
            name = field_obj.name
            lines.append(f"{name} = {_toml_value(getattr(record, name))}")
        lines.append("")
    for cohort in report.cohort_aggregates:
        lines.append("[[cohort_aggregate]]")
        for field_obj in dataclasses.fields(cohort):
            name = field_obj.name
            lines.append(f"{name} = {_toml_value(getattr(cohort, name))}")
        lines.append("")
    if report.tiered_gate_report is not None:
        lines.append("[tiered_gate_report]")
        for field_obj in dataclasses.fields(report.tiered_gate_report):
            name = field_obj.name
            lines.append(
                f"{name} = "
                f"{_toml_value(getattr(report.tiered_gate_report, name))}"
            )
        lines.append("")
    for gate_cohort in report.gate_cohort_results:
        lines.append("[[gate_cohort_result]]")
        for field_obj in dataclasses.fields(gate_cohort):
            name = field_obj.name
            lines.append(
                f"{name} = {_toml_value(getattr(gate_cohort, name))}"
            )
        lines.append("")
    for surface_entry in report.cache_prediction_surface_entries:
        lines.append("[[cache_prediction_surface_entry]]")
        for field_obj in dataclasses.fields(surface_entry):
            name = field_obj.name
            lines.append(
                f"{name} = "
                f"{_toml_value(getattr(surface_entry, name))}"
            )
        lines.append("")
    # L2-read-denominator audit emits two top-level
    # arrays so the per-class breakdown can be queried
    # independently: ``[[l2_denominator_audit]]`` carries
    # cell-level totals + gap; ``[[l2_denominator_audit_class]]``
    # carries one row per (cell, op_kind, address_space,
    # l1_access_policy) class with ``cell_name`` as the
    # foreign key.
    for audit_row in report.l2_denominator_audit:
        lines.append("[[l2_denominator_audit]]")
        lines.append(
            f"cell_name = {_toml_escape(audit_row.cell_name)}"
        )
        lines.append(
            f"tier2_l2_sectors_total = "
            f"{_toml_value(audit_row.tier2_l2_sectors_total)}"
        )
        lines.append(
            f"ncu_l2_sectors_total = "
            f"{_toml_value(audit_row.ncu_l2_sectors_total)}"
        )
        lines.append(
            f"denominator_gap = "
            f"{_toml_value(audit_row.denominator_gap)}"
        )
        lines.append(
            f"denominator_gap_ratio = "
            f"{_toml_value(audit_row.denominator_gap_ratio)}"
        )
        lines.append(
            f"audit_status = "
            f"{_toml_escape(audit_row.audit_status)}"
        )
        lines.append(
            f"typed_reason = "
            f"{_toml_escape(audit_row.typed_reason)}"
        )
        lines.append("")
    for audit_row in report.l2_denominator_audit:
        for class_row in audit_row.per_class_breakdown:
            lines.append("[[l2_denominator_audit_class]]")
            lines.append(
                f"cell_name = {_toml_escape(audit_row.cell_name)}"
            )
            for field_obj in dataclasses.fields(class_row):
                name = field_obj.name
                lines.append(
                    f"{name} = "
                    f"{_toml_value(getattr(class_row, name))}"
                )
            lines.append("")

    # Accuracy-diagnostic sections (DEC-6 direct-call wiring).
    # The diagnostic builder walks all 235 canonical SOL-ExecBench
    # problems, decides each per-cell row's status via the
    # filesystem + ledger + partition detector chain, and emits
    # six diagnostic sections that are additively appended to
    # this smoke report. The diagnostic is calibration-scoped: it
    # reads the static label ledger as offline ground truth and
    # does NOT invoke NCU at runtime.
    diagnostic_sections = _serialize_accuracy_diagnostic_sections(
        smoke_records=report.per_cell_records,
        commit_sha=report.commit_sha,
        nvbit_so_sha256=report.nvbit_so_sha256,
        trace_budget_bytes=report.trace_budget_bytes,
        trace_capture_timeout_seconds=report.trace_capture_timeout_seconds,
        ledger_path=report.ledger_path,
    )
    lines.append(diagnostic_sections)
    return "\n".join(lines).rstrip() + "\n"


def _serialize_accuracy_diagnostic_sections(
    *,
    smoke_records: Sequence[Any],
    commit_sha: str,
    nvbit_so_sha256: str,
    trace_budget_bytes: Any,
    trace_capture_timeout_seconds: int,
    ledger_path: Optional[str] = None,
) -> str:
    """Build the accuracy diagnostic sections that accompany the
    smoke report.

    ``ledger_path`` (when supplied) names the ledger the smoke
    run used; the diagnostic loads the same file so its
    ``[meta].ledger_path`` / ``ledger_sha256`` / per-cell join
    keys reflect the smoke run, not the repo default. When
    ``None`` or absent, the diagnostic builder falls back to its
    repo-default ledger path.

    Returns either the serialized diagnostic sections (when the
    builder succeeds) or a one-line comment recording why the
    diagnostic was skipped (e.g. the canonical problem partition
    is unavailable in a test fixture environment). Errors from
    the diagnostic do NOT fail the smoke harness: the smoke
    report is the primary artifact, and the diagnostic is a
    layered analytic surface.
    """

    from sol_baselines_harness.calibration.cache_prediction.accuracy_diagnostic import (
        build_accuracy_diagnostic_report,
        predictor_callback_from_smoke_records,
        serialize_diagnostic_sections_for_smoke_report,
        AccuracyGateError,
        PerProblemRecordSchemaError,
    )

    try:
        # Convert the trace_budget_bytes (`"default"` sentinel or
        # a stringified int) to an int for the diagnostic meta
        # table. The diagnostic's `[meta].trace_budget_bytes`
        # field is typed as int; the smoke report's same-named
        # field can be the sentinel string.
        try:
            trace_budget_int = int(trace_budget_bytes)
        except (TypeError, ValueError):
            trace_budget_int = 0

        # Resolve the ledger path: honor the smoke run's ledger
        # when supplied so non-default `--ledger` invocations
        # do not silently fall back to the repo default. Load
        # the file so the callback can run the join-key drift
        # detector against the same ledger.
        import tomllib as _tomllib
        if ledger_path:
            ledger_path_obj = Path(ledger_path)
        else:
            ledger_path_obj = (
                Path(__file__).resolve().parents[4]
                / "scripts" / "baselines_harness" / "calibration"
                / "cache_prediction" / "label_ledger_v1_full_partition.toml"
            )
        ledger_data = None
        if ledger_path_obj.is_file():
            ledger_data = _tomllib.loads(
                ledger_path_obj.read_bytes().decode("utf-8"),
            )

        callback = predictor_callback_from_smoke_records(
            smoke_records, ledger_data=ledger_data,
        )
        # Translate the smoke harness's no-git sentinel
        # (`"unknown"`) to the diagnostic meta validator's
        # accepted sentinel (`"git_unavailable"`) so a no-git
        # smoke run still produces appended diagnostic sections
        # instead of silently degrading to the skipped-comment
        # fallback. The diagnostic builder also re-derives a real
        # HEAD SHA via `git rev-parse HEAD` when
        # `implementation_commit_sha=None`, but we have no way
        # here to distinguish "smoke ran with git" from "smoke
        # could not resolve HEAD" except by the sentinel value.
        diagnostic_commit_sha = (
            "git_unavailable" if commit_sha == "unknown" else commit_sha
        )
        report = build_accuracy_diagnostic_report(
            implementation_commit_sha=diagnostic_commit_sha,
            predictor_callback=callback,
            nvbit_so_sha256=nvbit_so_sha256,
            trace_budget_bytes=trace_budget_int,
            timeout_seconds=trace_capture_timeout_seconds,
            ledger_path=ledger_path_obj,
        )
        return serialize_diagnostic_sections_for_smoke_report(report)
    except (
        AccuracyGateError,
        PerProblemRecordSchemaError,
        FileNotFoundError,
        tomllib.TOMLDecodeError,
        UnicodeDecodeError,
        OSError,
    ) as exc:
        # Any failure inside the diagnostic-append path degrades
        # to the skipped-comment line. The smoke report itself is
        # the primary artifact; per the helper's contract,
        # diagnostic-section errors (including a malformed or
        # unreadable ledger TOML) must never propagate up to
        # `serialize_smoke_report_to_toml` and lose the primary
        # smoke report.
        return (
            "\n# --- Accuracy diagnostic sections skipped: "
            f"{type(exc).__name__}: {exc} ---\n"
        )


def assert_smoke_report_commit_sha_matches(
    report: EndToEndSmokeReport, *, expected_commit_sha: str,
) -> None:
    """Raise ``EndToEndSmokeError("smoke_report_commit_sha_mismatch", ...)``
    if the smoke report's ``commit_sha`` does not match
    ``expected_commit_sha``.

    Used by tests / external validators to fail fast when
    a committed artifact's metadata identifies the wrong
    implementation revision (e.g., the report was generated
    against an older code commit and never refreshed).
    """
    if not isinstance(expected_commit_sha, str) or not expected_commit_sha:
        raise EndToEndSmokeError(
            "smoke_input_invalid",
            f"expected_commit_sha must be a non-empty string; "
            f"got {expected_commit_sha!r}",
        )
    if report.commit_sha != expected_commit_sha:
        raise EndToEndSmokeError(
            "smoke_report_commit_sha_mismatch",
            f"smoke report's commit_sha is "
            f"{report.commit_sha!r} but the expected "
            f"revision is {expected_commit_sha!r}; "
            f"regenerate the artifact against the current "
            f"implementation commit",
        )


def dump_smoke_report_toml(
    report: EndToEndSmokeReport, path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_smoke_report_to_toml(report))


def load_smoke_report_toml(path: Path) -> Mapping[str, Any]:
    """Roundtrip loader for the smoke report
    TOML. Returns the parsed mapping (not the
    typed dataclass) since downstream consumers
    typically inspect the report by key.
    """
    with open(path, "rb") as fp:
        return tomllib.load(fp)


__all__ = [
    "CachePredictionSurfaceEntryRow",
    "CapturedSmokeRecord",
    "CohortAggregate",
    "END_TO_END_SMOKE_TYPED_REASONS",
    "EndToEndSmokeError",
    "EndToEndSmokeReport",
    "GateCohortResultRow",
    "L2DenominatorAuditClassRow",
    "L2DenominatorAuditRow",
    "SmokeCellSpec",
    "TieredGateReportRow",
    "assert_smoke_report_commit_sha_matches",
    "build_smoke_specs_from_ledger",
    "dump_smoke_report_toml",
    "load_smoke_report_toml",
    "run_end_to_end_smoke",
    "serialize_smoke_report_to_toml",
]


def _main_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end smoke harness on sm_86 "
            "and emit a measured-residual TOML artifact."
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=_DEFAULT_LEDGER_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help=(
            "subset of cell names to run; defaults to all "
            "ledger entries whose binary can be resolved by "
            "sha256"
        ),
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp"))
        / "solbench_end_to_end_smoke",
    )
    args = parser.parse_args()
    ledger = load_ledger_from_toml(args.ledger)
    specs = build_smoke_specs_from_ledger(ledger, cell_names=args.cells)
    if not specs:
        print(
            "no resolvable cells; nothing to run", file=sys.stderr,
        )
        return 2
    report = run_end_to_end_smoke(
        specs,
        out_dir=args.scratch_dir,
        ledger_path=args.ledger,
    )
    dump_smoke_report_toml(report, args.output)
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "per_cell_count": len(report.per_cell_records),
                "cohort_aggregates": [
                    {
                        "cohort": c.cohort,
                        "cell_count": c.cell_count,
                        "l1_mean_abs_pp_error": c.l1_mean_abs_pp_error,
                        "l2_mean_abs_pp_error": c.l2_mean_abs_pp_error,
                    }
                    for c in report.cohort_aggregates
                ],
            },
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli())
