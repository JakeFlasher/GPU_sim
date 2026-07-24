"""Calibration sweep driver + versioned label ledger
(calibration subtree, NCU import legal).

Iterates over a list of `LedgerCellSpec` records. For each
spec that names a real binary on disk, the driver derives a
`WarmupProtocolRecord` from the spec's launch_skip_count
+ warmup_protocol_name (so the ledger entry carries the
normalized warmup_protocol AND cache_initial_state), then
calls `harvest_ncu_label_with_provenance` under a runner-wrapper that
captures the raw NCU CSV bytes for provenance SHA-256, and
emits a `LedgerEntry`. For each spec marked unavailable
(binary path `None`, non-empty `unavailable_reason` drawn
from the closed enum), the driver emits a
`LedgerUnavailable` instead. Every cell is resolved against
the deployed predictor design-coverage partition; unmapped
cells raise the typed `sweep_predictor_bucket_unmapped`
reason so the ledger cannot silently admit a cell outside
the partition.

The emitted `LabelLedger` carries schema_version,
captured_at_utc, host_arch, host_descriptor_hash,
ncu_version, driver_version, plus the per-entry and
per-unavailable payloads. The companion serializer writes a
TOML document with `[meta]` + `[[entries]]` + `[[unavailable]]`
tables; the loader round-trips the document back through
the validator.

A separate `validate_label_ledger_complete_for_manifest`
helper enforces that the ledger's union of entry and
unavailable cell names equals an expected set (the deployed
partition's 33 cells, when called from the artifact-emission
path). A subset ledger cannot pass this gate.

Imports stdlib + tomllib + the calibration subtree's
existing `ncu_harvester` / `label_parser` + `corpus_manifest`
+ the deployed `predictor_design_coverage` and
`warmup_protocol` modules. The calibration -> deployed
import direction is allowed; the deployed -> calibration
direction is the one the NCU quarantine forbids.
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import hashlib
import tomllib
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from sol_baselines_harness.calibration.cache_prediction.corpus_manifest import (
    CalibrationCorpusManifest,
    load_calibration_corpus_manifest,
)
from sol_baselines_harness.calibration.cache_prediction.label_parser import (
    NcuLabelParserError,
    NcuLabelRecord,
)
from sol_baselines_harness.calibration.cache_prediction.ncu_harvester import (
    NcuHarvesterError,
    _run_checked_ncu,
    harvest_ncu_label_with_provenance,
    validate_ncu_label_provenance,
)
from sol_baselines_harness.metrics.framework_contract import (
    FrameworkContractError,
    cell_join_key_from_ledger_entry,
)
from sol_baselines_harness.metrics.predictor_design_coverage import (
    PREDICTOR_BUCKET_TYPED_UNAVAILABLE,
    PREDICTOR_BUCKETS,
    PREDICTOR_DESIGN_COVERAGE_BY_CELL,
    PredictorDesignCoverageError,
    lookup_predictor_design_coverage,
)
from sol_baselines_harness.metrics.warmup_protocol import (
    CACHE_INITIAL_STATE_COLD,
    CACHE_INITIAL_STATE_WARM,
    WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
    WarmupProtocolError,
    WarmupProtocolRecord,
    derive_warmup_protocol,
)


LABEL_LEDGER_SCHEMA_VERSION: str = "1.0"


# Arch-keyed preflight dispatch (capability probe + arch table).
# When True for a given `host_arch`, the sweep refuses to construct
# any NCU argv until the NCU capability probe is parsed + validated by
# `ncu_capability_probe.preflight_for_arch`. New arches added
# here inherit the gate without flow-control edits in the sweep
# body.
_PREFLIGHT_REQUIRED_BY_ARCH: dict[str, bool] = {
    "sm_120": True,
}


CALIBRATION_SWEEP_TYPED_REASONS: frozenset[str] = frozenset({
    "sweep_cell_spec_invalid",
    "sweep_binary_missing",
    "sweep_ledger_schema_violation",
    "sweep_predictor_bucket_unmapped",
    "sweep_capability_probe_required",
})


# Closed enum of every reason a LedgerUnavailable row may
# carry. Drawn from three sources:
#   * Partition typed-unavailable subclasses (prefixed
#     `partition_typed_unavailable_` to disambiguate from the
#     partition's own enum entries; the predeclared partition
#     reason is appended verbatim after the prefix).
#   * Hardware-architecture constraints (FP8 / NVFP4 / MoE-FP8
#     need sm_100+ features that sm_86 cannot exercise; the
#     access-pattern labels harvested on sm_86 would mis-
#     represent the actual B200 behavior).
#   * Implementation-pending constraints (attention-template
#     suite for GQA paged decode/prefill / softcap+dropout
#     not yet built on sm_86; honest acknowledgment, not a
#     deferral escape hatch).
#   * Harvester failure propagation (`ncu_invocation_failed`).
LEDGER_UNAVAILABLE_TYPED_REASONS: frozenset[str] = frozenset({
    # Partition-level constraints
    "partition_typed_unavailable_warm_cold_protocol_unpinned",
    "partition_typed_unavailable_local_memory_private_cache_policy",
    "partition_typed_unavailable_shared_memory_no_read_only_sector_traffic",
    "partition_typed_unavailable_hostile_subclass_uncalibrated",
    # Hardware-architecture constraints
    "kernel_requires_sm_100_features",
    # Implementation-pending constraints
    "kernel_attention_template_pending",
    # Harvester-failure
    "ncu_invocation_failed",
    # Kernel produced no read-only L1/L2 traffic — the
    # hit-rate label is undefined (zero-denominator) for
    # the read-only sector predictor. Examples: pure-
    # atomic kernels (only RED instructions, no LDG),
    # divergent-branch kernels that route all reads
    # through shared memory.
    "ncu_label_no_read_only_traffic",
    # Batch-driver constraint: the landed `*.cu` exists in
    # `baselines/` but the operator has not yet built the
    # corresponding NCU probe binary at
    # `<probe_binary_root>/<slug>`. The discovery layer
    # surfaces this as an unavailable row so the batch
    # report is honest about coverage; the operator's next
    # action is to build the missing probe binary.
    "probe_binary_missing_for_slug",
})


class CalibrationSweepError(Exception):
    """Raised by the sweep driver and the ledger validator
    on contract violations. Carries the typed-reason string
    so downstream callers can compose typed-unavailable
    bundles or report-table errors without parsing message
    strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class LedgerCellSpec:
    """One cell scheduled for the sweep.

    A spec with `binary_path != None` and `unavailable_reason
    == ""` is the normal case: the driver derives a warmup
    protocol from `launch_skip_count` + `warmup_protocol_name`,
    runs NCU on the binary, and emits a `LedgerEntry`. A spec
    with `binary_path is None` AND a non-empty
    `unavailable_reason` (drawn from
    `LEDGER_UNAVAILABLE_TYPED_REASONS`) is the unavailable
    case: the driver skips NCU and emits a `LedgerUnavailable`
    with the typed reason preserved. Any other combination
    raises `sweep_cell_spec_invalid`.

    `measured_cache_initial_state` is the warm/cold cache-
    protocol carrier surfaced from the corpus manifest. Empty
    default means "use the protocol-derived
    `cache_initial_state` from `derive_warmup_protocol`"; an
    explicit `"warm"` or `"cold"` overrides the derived value
    in `_derive_protocol` and is recorded as the ledger
    entry's `cache_initial_state` field.
    """

    cell_name: str
    binary_path: Path | None
    launch_skip_count: int
    warmup_protocol_name: str
    unavailable_reason: str = ""
    shape_point: str = ""
    kernel_source_path: Path | None = None
    measured_cache_initial_state: str = ""


@dataclass(frozen=True)
class LedgerEntry:
    """A real, NCU-derived ledger entry for one cell.

    Carries the canonical predictor bucket from the deployed
    partition (so a future re-sweep on the same cell cannot
    silently drift its bucket relative to the deployed
    module) PLUS the full causal cache-key set: arch,
    ncu_metric_version, warmup protocol, cache_initial_state
    for warm/cold provenance, launch-skip count, binary
    SHA-256, raw NCU CSV SHA-256, shape_point,
    source_commit, and kernel_signature_sha256. The last
    three close the framework-contract 7-tuple required by
    AC-FI-1 so downstream consumers can extract a
    CellJoinKey via
    `framework_contract.cell_join_key_from_ledger_entry`.

    The nine NCU-provenance fields (`ncu_cli_path`,
    `ncu_cli_sha256`, `ncu_version_stdout`, `driver_version`,
    `gpu_device_name`, `compute_capability`,
    `clock_policy_observed`, `command_argv`,
    `raw_report_sha256`) close the ledger's full-provenance
    contract: a re-harvest on a different NCU build or a
    different driver / GPU on the same calibration host
    surfaces immediately as a hash drift. The `raw_csv_sha256`
    field is preserved as a transitional alias of
    `raw_report_sha256`; both carry the same value on writes
    produced after this round. Legacy ledger entries created
    before the provenance schema landed populate the
    provenance fields with the literal sentinel
    `"unknown_legacy_entry"` so the ledger loader keeps
    accepting them while a new write fails closed on missing
    real provenance via `validate_ncu_label_provenance()`.
    """

    cell_name: str
    predictor_bucket: str
    arch: str
    ncu_metric_version: str
    warmup_protocol: str
    cache_initial_state: str
    launch_skip_count: int
    binary_sha256: str
    raw_csv_sha256: str
    shape_point: str
    source_commit: str
    kernel_signature_sha256: str
    label: NcuLabelRecord
    ncu_cli_path: str = ""
    ncu_cli_sha256: str = ""
    ncu_version_stdout: str = ""
    driver_version: str = ""
    gpu_device_name: str = ""
    compute_capability: str = ""
    clock_policy_observed: str = ""
    command_argv: tuple[str, ...] = ()
    raw_report_sha256: str = ""


_NOT_APPLICABLE: str = "not_applicable"


@dataclass(frozen=True)
class LedgerUnavailable:
    """A structured unavailable entry for a cell that the
    sweep could not produce a real NCU label for. The
    `unavailable_reason` is a typed string drawn from the
    closed `LEDGER_UNAVAILABLE_TYPED_REASONS` enum so a
    downstream consumer can branch on a CONCRETE constraint
    (hardware unavailable, attention-template missing,
    partition-typed-unavailable, harvester failure) rather
    than a generic deferral.

    Protocol-provenance fields (`launch_skip_count`,
    `warmup_protocol`, `cache_initial_state`) carry the
    same cache-key components `LedgerEntry` carries. For
    never-invoked rows (partition-typed-unavailable,
    hardware-required), the protocol fields are populated
    with the sentinel `"not_applicable"` and
    `launch_skip_count = 0`. For attempted-but-failed rows
    (parser denominator-zero, harvester invocation failure
    after invoking NCU), the fields preserve the derived
    protocol that WOULD have been recorded had the harvest
    succeeded.
    """

    cell_name: str
    predictor_bucket: str
    unavailable_reason: str
    launch_skip_count: int
    warmup_protocol: str
    cache_initial_state: str
    shape_point: str
    source_commit: str
    kernel_signature_sha256: str


@dataclass(frozen=True)
class LabelLedger:
    """Versioned ledger emitted by `run_calibration_sweep`.

    `entries` carries the cells that produced real NCU
    labels; `unavailable` carries cells the sweep could not
    label, each with a typed reason. Together they form the
    complete record of the sweep's intended cell set.
    """

    schema_version: str
    captured_at_utc: str
    host_arch: str
    host_descriptor_hash: str
    ncu_version: str
    driver_version: str
    entries: tuple[LedgerEntry, ...]
    unavailable: tuple[LedgerUnavailable, ...]


def _default_clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_cell_spec(spec: LedgerCellSpec) -> None:
    if not isinstance(spec, LedgerCellSpec):
        raise CalibrationSweepError(
            "sweep_cell_spec_invalid",
            f"spec must be LedgerCellSpec; got "
            f"{type(spec).__name__}",
        )
    if not spec.cell_name or not spec.cell_name.strip():
        raise CalibrationSweepError(
            "sweep_cell_spec_invalid",
            "cell_name is empty",
        )
    if not isinstance(spec.launch_skip_count, int) or spec.launch_skip_count < 0:
        raise CalibrationSweepError(
            "sweep_cell_spec_invalid",
            f"cell {spec.cell_name!r}: launch_skip_count "
            f"must be a non-negative int; got "
            f"{spec.launch_skip_count!r}",
        )
    if not spec.warmup_protocol_name or not spec.warmup_protocol_name.strip():
        raise CalibrationSweepError(
            "sweep_cell_spec_invalid",
            f"cell {spec.cell_name!r}: warmup_protocol_name "
            f"is empty",
        )
    if spec.binary_path is None:
        if not spec.unavailable_reason or not spec.unavailable_reason.strip():
            raise CalibrationSweepError(
                "sweep_cell_spec_invalid",
                f"cell {spec.cell_name!r}: binary_path is "
                f"None so unavailable_reason must be non-empty",
            )
        if spec.unavailable_reason not in LEDGER_UNAVAILABLE_TYPED_REASONS:
            raise CalibrationSweepError(
                "sweep_cell_spec_invalid",
                f"cell {spec.cell_name!r}: unavailable_reason "
                f"{spec.unavailable_reason!r} is not in the "
                f"closed enum "
                f"{sorted(LEDGER_UNAVAILABLE_TYPED_REASONS)}",
            )
    else:
        if spec.unavailable_reason:
            raise CalibrationSweepError(
                "sweep_cell_spec_invalid",
                f"cell {spec.cell_name!r}: binary_path is "
                f"set so unavailable_reason must be empty "
                f"(got {spec.unavailable_reason!r})",
            )
    if spec.measured_cache_initial_state not in _MEASURED_CACHE_INITIAL_STATE_ALLOWED:
        raise CalibrationSweepError(
            "sweep_cell_spec_invalid",
            f"cell {spec.cell_name!r}: measured_cache_initial_state "
            f"{spec.measured_cache_initial_state!r} is not in the "
            f"closed set "
            f"{sorted(_MEASURED_CACHE_INITIAL_STATE_ALLOWED)} "
            f"(empty = use protocol-derived default)",
        )


# Closed vocabulary for the per-spec warm/cold carrier. Empty string
# means "use the protocol-derived default from `derive_warmup_protocol`"
# (the legacy implicit behavior for cells without an explicit
# manifest-pinned cache initial state). Explicit `"warm"` and
# `"cold"` override the derived value in `_derive_protocol`. Matches
# the manifest-side closed vocabulary in
# `CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES`.
_MEASURED_CACHE_INITIAL_STATE_ALLOWED: frozenset[str] = frozenset({
    "",
    CACHE_INITIAL_STATE_WARM,
    CACHE_INITIAL_STATE_COLD,
})


def _resolve_partition_bucket(cell_name: str) -> str:
    try:
        entry = lookup_predictor_design_coverage(cell_name)
    except PredictorDesignCoverageError as exc:
        raise CalibrationSweepError(
            "sweep_predictor_bucket_unmapped",
            f"cell {cell_name!r} is not predeclared in the "
            f"deployed predictor design-coverage partition "
            f"(reason: {exc.typed_reason})",
        ) from exc
    return entry.predictor_bucket


def _derive_protocol(spec: LedgerCellSpec) -> WarmupProtocolRecord:
    try:
        protocol = derive_warmup_protocol(
            spec.launch_skip_count,
            protocol=spec.warmup_protocol_name,
        )
    except WarmupProtocolError as exc:
        raise CalibrationSweepError(
            "sweep_cell_spec_invalid",
            f"cell {spec.cell_name!r}: warmup protocol "
            f"derivation failed ({exc.typed_reason}: "
            f"{exc.message})",
        ) from exc
    # Warm/cold cache-protocol carrier override. When the spec
    # supplies an explicit `"warm"` / `"cold"` value (validated by
    # `_validate_cell_spec` and the upstream manifest loader), it
    # replaces the protocol-derived `cache_initial_state` on the
    # returned record. An empty carrier preserves the legacy
    # protocol-derived value so cells authored before the carrier
    # landed keep their existing behavior.
    if spec.measured_cache_initial_state:
        return WarmupProtocolRecord(
            launch_skip_count=protocol.launch_skip_count,
            warmup_protocol=protocol.warmup_protocol,
            cache_initial_state=spec.measured_cache_initial_state,
        )
    return protocol


def _default_source_commit_resolver() -> str:
    """Resolve the current git HEAD commit.

    Returns a 40-character lowercase hex SHA on success.
    Falls back to `"not_applicable"` when git is not
    available or the working tree is not a git checkout —
    a sweep on a tarball release is still legitimate, and
    the sentinel preserves AC-FI-1's "every join-key field
    must be a non-empty string" contract.
    """
    import subprocess
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return _NOT_APPLICABLE
    if completed.returncode != 0:
        return _NOT_APPLICABLE
    sha = completed.stdout.strip().lower()
    if len(sha) != 40 or any(
        c not in "0123456789abcdef" for c in sha
    ):
        return _NOT_APPLICABLE
    # The framework-contract 7-tuple requires 64-character
    # SHA-256 for binary_sha256 + kernel_signature_sha256;
    # source_commit is free-form per the sibling project
    # (`comparison_contract.toml [join_key].fields` does
    # NOT pin a SHA-256 shape for source_commit), so the
    # 40-char git SHA passes through.
    return sha


def _default_kernel_source_hash(
    source_path: Path | None,
    *,
    sha256_of_file: Callable[[Path], str],
) -> str:
    """Hash a kernel `.cu` source file.

    `source_path is None` returns the sentinel
    `"not_applicable"` for never-invoked unavailable
    cells; otherwise routes through the injected
    `sha256_of_file` helper so tests can stub the file
    hash deterministically without needing the actual
    file on disk.
    """
    if source_path is None:
        return _NOT_APPLICABLE
    if not source_path.is_file():
        return _NOT_APPLICABLE
    return sha256_of_file(source_path)


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.calibration_sweep.run_calibration_sweep')
def run_calibration_sweep(
    cell_specs: Iterable[LedgerCellSpec],
    *,
    ncu_metric_version: str,
    host_arch: str,
    host_descriptor_hash: str,
    ncu_version: str,
    driver_version: str,
    runner: Callable[[Sequence[str]], Any] | None = None,
    clock: Callable[[], str] | None = None,
    sha256_of_file: Callable[[Path], str] | None = None,
    source_commit_resolver: Callable[[], str] | None = None,
    ncu_cli_path: str | None = None,
    ncu_cli_sha256_reader: Callable[[Path | str], str] | None = None,
    ncu_version_runner: Callable[[Sequence[str]], Any] | None = None,
    nvidia_smi_runner: Callable[[Sequence[str]], Any] | None = None,
    probe_path: Path | str | None = None,
) -> LabelLedger:
    """Run the sweep and emit a `LabelLedger`.

    For each spec with a binary, this routes the harvest
    through `harvest_ncu_label_with_provenance()` so every
    emitted `LedgerEntry` carries the full nine-field NCU
    provenance block populated by the same host probes that
    captured the label. The runner wrapper hashes the
    captured NCU stdout (the raw CSV) for the
    `raw_csv_sha256` provenance.

    Every spec's `cell_name` is resolved against the
    canonical deployed partition; unmapped cells raise
    `sweep_predictor_bucket_unmapped`. For each non-
    unavailable spec, the warmup protocol is derived via
    `derive_warmup_protocol`; the resulting normalized
    `warmup_protocol` AND `cache_initial_state` are written
    to the ledger entry. The validator afterwards enforces
    no-duplicate cell names across entries + unavailable
    AND the full NCU-provenance schema on every entry.

    Probe-injection seams (`ncu_cli_path`,
    `ncu_cli_sha256_reader`, `ncu_version_runner`,
    `nvidia_smi_runner`) mirror the seams on
    `harvest_ncu_label_with_provenance()` so unit tests can
    exercise the full provenance write path without
    spawning real `ncu` or `nvidia-smi` subprocesses.

    Typed errors raised: `sweep_cell_spec_invalid`,
    `sweep_binary_missing`,
    `sweep_predictor_bucket_unmapped`,
    `sweep_ledger_schema_violation` (via the validator).
    """
    # Capability preflight: arch-keyed dispatch — when the host_arch's
    # `_PREFLIGHT_REQUIRED_BY_ARCH` entry is True, require +
    # validate the NCU capability probe BEFORE constructing any
    # NCU argv. The dispatch table makes the discipline data-
    # driven (per the arch dispatch table); arches added to the table
    # inherit the gate without a flow-control edit.
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.calibration_sweep.run_calibration_sweep')
    if _PREFLIGHT_REQUIRED_BY_ARCH.get(host_arch, False):
        if probe_path is None:
            raise CalibrationSweepError(
                "sweep_capability_probe_required",
                f"host_arch={host_arch!r} requires probe_path; "
                "the capability preflight gate refuses to start NCU "
                "runs without a validated capability probe "
                "artifact",
            )
        from .ncu_capability_probe import preflight_for_arch
        preflight_for_arch(host_arch, probe_path)

    clock_fn = clock if clock is not None else _default_clock
    file_hash = (
        sha256_of_file
        if sha256_of_file is not None
        else _default_sha256_of_file
    )
    resolve_source_commit = (
        source_commit_resolver
        if source_commit_resolver is not None
        else _default_source_commit_resolver
    )
    # Resolve once at sweep start so every entry + unavailable
    # row in this ledger shares the same source_commit
    # (the sweep is one logical event; a re-sweep on a
    # different commit produces a different ledger).
    source_commit = resolve_source_commit()

    entries: list[LedgerEntry] = []
    unavailable_rows: list[LedgerUnavailable] = []
    for spec in cell_specs:
        _validate_cell_spec(spec)
        predictor_bucket = _resolve_partition_bucket(spec.cell_name)
        shape_point = spec.shape_point or spec.cell_name
        kernel_signature_sha256 = _default_kernel_source_hash(
            spec.kernel_source_path, sha256_of_file=file_hash,
        )
        if spec.binary_path is None:
            # Never-invoked unavailable row: protocol fields
            # explicitly record `not_applicable` so the
            # downstream reader sees zero protocol provenance
            # (no warmup actually happened).
            unavailable_rows.append(
                LedgerUnavailable(
                    cell_name=spec.cell_name,
                    predictor_bucket=predictor_bucket,
                    unavailable_reason=spec.unavailable_reason,
                    launch_skip_count=0,
                    warmup_protocol=_NOT_APPLICABLE,
                    cache_initial_state=_NOT_APPLICABLE,
                    shape_point=shape_point,
                    source_commit=source_commit,
                    kernel_signature_sha256=kernel_signature_sha256,
                ),
            )
            continue
        if not spec.binary_path.is_file():
            raise CalibrationSweepError(
                "sweep_binary_missing",
                f"cell {spec.cell_name!r}: binary_path "
                f"{str(spec.binary_path)!r} does not exist "
                f"on disk",
            )
        # Derive warmup protocol BEFORE invoking NCU so a
        # bad launch_skip_count (=0) or unknown protocol
        # name fails fast.
        protocol = _derive_protocol(spec)
        captured_csv = bytearray()

        def _capturing_runner(argv, _inner=runner):
            inner = _inner if _inner is not None else _run_checked_ncu
            completed = inner(argv)
            stdout = completed.stdout
            captured_csv.extend(stdout.encode("utf-8"))
            return completed

        binary_sha256 = file_hash(spec.binary_path)
        try:
            label, provenance = harvest_ncu_label_with_provenance(
                spec.binary_path,
                launch_skip_count=spec.launch_skip_count,
                ncu_metric_version=ncu_metric_version,
                binary_sha256=binary_sha256,
                warmup_protocol=protocol.warmup_protocol,
                _runner=_capturing_runner,
                ncu_cli_path=ncu_cli_path,
                ncu_cli_sha256_reader=ncu_cli_sha256_reader,
                ncu_version_runner=ncu_version_runner,
                nvidia_smi_runner=nvidia_smi_runner,
            )
        except NcuLabelParserError as exc:
            # Parser failed: most commonly because the
            # kernel produced no read-only traffic at this
            # cache level (`ncu_label_denominator_zero`).
            # Emit a structured unavailable row instead of
            # propagating the parser exception. The actual
            # derived protocol IS recorded — the harvest was
            # attempted, the warmup did happen, and the
            # provenance reader needs the cache-key triple.
            if exc.typed_reason == "ncu_label_denominator_zero":
                unavailable_rows.append(LedgerUnavailable(
                    cell_name=spec.cell_name,
                    predictor_bucket=predictor_bucket,
                    unavailable_reason=(
                        "ncu_label_no_read_only_traffic"
                    ),
                    launch_skip_count=spec.launch_skip_count,
                    warmup_protocol=protocol.warmup_protocol,
                    cache_initial_state=protocol.cache_initial_state,
                    shape_point=shape_point,
                    source_commit=source_commit,
                    kernel_signature_sha256=kernel_signature_sha256,
                ))
                continue
            # Other parser failures are typed by the parser
            # itself; surface them via the harvester-failed
            # typed reason so the ledger reader sees one
            # canonical failure family. Protocol provenance
            # preserved (harvest was attempted).
            unavailable_rows.append(LedgerUnavailable(
                cell_name=spec.cell_name,
                predictor_bucket=predictor_bucket,
                unavailable_reason="ncu_invocation_failed",
                launch_skip_count=spec.launch_skip_count,
                warmup_protocol=protocol.warmup_protocol,
                cache_initial_state=protocol.cache_initial_state,
                shape_point=shape_point,
                source_commit=source_commit,
                kernel_signature_sha256=kernel_signature_sha256,
            ))
            continue
        except NcuHarvesterError:
            unavailable_rows.append(LedgerUnavailable(
                cell_name=spec.cell_name,
                predictor_bucket=predictor_bucket,
                unavailable_reason="ncu_invocation_failed",
                launch_skip_count=spec.launch_skip_count,
                warmup_protocol=protocol.warmup_protocol,
                cache_initial_state=protocol.cache_initial_state,
                shape_point=shape_point,
                source_commit=source_commit,
                kernel_signature_sha256=kernel_signature_sha256,
            ))
            continue
        raw_csv_sha256 = _sha256_of_bytes(bytes(captured_csv))
        entries.append(LedgerEntry(
            cell_name=spec.cell_name,
            predictor_bucket=predictor_bucket,
            arch=host_arch,
            ncu_metric_version=ncu_metric_version,
            warmup_protocol=protocol.warmup_protocol,
            cache_initial_state=protocol.cache_initial_state,
            launch_skip_count=spec.launch_skip_count,
            binary_sha256=binary_sha256,
            raw_csv_sha256=raw_csv_sha256,
            shape_point=shape_point,
            source_commit=source_commit,
            kernel_signature_sha256=kernel_signature_sha256,
            label=label,
            ncu_cli_path=provenance.ncu_cli_path,
            ncu_cli_sha256=provenance.ncu_cli_sha256,
            ncu_version_stdout=provenance.ncu_version_stdout,
            driver_version=provenance.driver_version,
            gpu_device_name=provenance.gpu_device_name,
            compute_capability=provenance.compute_capability,
            clock_policy_observed=provenance.clock_policy_observed,
            command_argv=tuple(provenance.command_argv),
            raw_report_sha256=provenance.raw_report_sha256,
        ))
    ledger = LabelLedger(
        schema_version=LABEL_LEDGER_SCHEMA_VERSION,
        captured_at_utc=clock_fn(),
        host_arch=host_arch,
        host_descriptor_hash=host_descriptor_hash,
        ncu_version=ncu_version,
        driver_version=driver_version,
        entries=tuple(entries),
        unavailable=tuple(unavailable_rows),
    )
    validate_label_ledger(ledger)
    return ledger


# Explicit sentinel value used to mark legacy ledger entries
# that pre-date the NCU-provenance schema. Pre-schema rows are
# migrated to this shape: every string provenance field is set
# to the sentinel literal AND `command_argv` is the singleton
# tuple `(_LEGACY_PROVENANCE_SENTINEL,)`. Any other shape —
# including the all-blank pre-migration shape — fails closed.
# A future ledger rewrite via the batch harvest driver replaces
# the sentinel with the real probed values.
_LEGACY_PROVENANCE_SENTINEL: str = "unknown_legacy_entry"


# The nine NCU provenance string fields that must be populated
# on every `LedgerEntry`. `command_argv` is the tenth provenance
# coordinate and is handled separately because it is a tuple of
# strings, not a single string.
_NCU_PROVENANCE_LEDGER_STRING_FIELDS: tuple[str, ...] = (
    "ncu_cli_path",
    "ncu_cli_sha256",
    "ncu_version_stdout",
    "driver_version",
    "gpu_device_name",
    "compute_capability",
    "clock_policy_observed",
    "raw_report_sha256",
)


def _entry_provenance_is_explicit_sentinel(entry: LedgerEntry) -> bool:
    """Return True iff the entry's provenance fields exactly
    match the explicit legacy-sentinel shape: every string
    field equals `_LEGACY_PROVENANCE_SENTINEL` AND
    `command_argv == (_LEGACY_PROVENANCE_SENTINEL,)`. Any other
    shape — including the all-blank pre-migration shape — fails
    this check and proceeds to strict provenance validation.
    """
    for field in _NCU_PROVENANCE_LEDGER_STRING_FIELDS:
        if getattr(entry, field) != _LEGACY_PROVENANCE_SENTINEL:
            return False
    return tuple(entry.command_argv) == (_LEGACY_PROVENANCE_SENTINEL,)


def _validate_entry_ncu_provenance(entry: LedgerEntry) -> None:
    """Enforce the NCU-provenance schema on a single
    `LedgerEntry`. The validator accepts exactly two shapes:

      1. The explicit legacy-sentinel shape: every string
         provenance field equals `_LEGACY_PROVENANCE_SENTINEL`
         AND `command_argv == (_LEGACY_PROVENANCE_SENTINEL,)`.
         The in-tree pre-schema ledger is migrated to this
         shape so it continues to load. Sentinel rows may
         legitimately carry historical `raw_csv_sha256` values
         that diverge from the sentinel `raw_report_sha256`
         because the migration deliberately did not rewrite
         the per-row body content; the alias-equality check
         below is bypassed for sentinel rows.
      2. The fully-populated normal write shape, validated by
         `validate_ncu_label_provenance()`. Every string field
         is non-empty, `command_argv` is a non-empty tuple of
         strings, AND `raw_csv_sha256 == raw_report_sha256`
         (the transitional alias-equality contract; both
         fields are SHA-256 hex digests of the SAME raw NCU
         report bytes, and a normal write produces them from
         one capture so divergence indicates corruption).

    Any other shape (all-blank fields, partially-populated
    fields, partial-sentinel fields, malformed `command_argv`,
    mismatched raw-report hashes) is rejected as
    `sweep_ledger_schema_violation` with the underlying typed
    reason chained from the harvester validator.
    """
    if _entry_provenance_is_explicit_sentinel(entry):
        return

    provenance_view = {
        "ncu_cli_path": entry.ncu_cli_path,
        "ncu_cli_sha256": entry.ncu_cli_sha256,
        "ncu_version_stdout": entry.ncu_version_stdout,
        "driver_version": entry.driver_version,
        "gpu_device_name": entry.gpu_device_name,
        "compute_capability": entry.compute_capability,
        "clock_policy_observed": entry.clock_policy_observed,
        "command_argv": entry.command_argv,
        "raw_report_sha256": entry.raw_report_sha256,
    }

    try:
        validate_ncu_label_provenance(provenance_view)
    except NcuHarvesterError as exc:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"entry {entry.cell_name!r}: NCU provenance "
            f"validation failed ({exc.typed_reason}: "
            f"{exc.message})",
        ) from exc

    if entry.raw_csv_sha256 != entry.raw_report_sha256:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"entry {entry.cell_name!r}: raw_csv_sha256 "
            f"({entry.raw_csv_sha256!r}) must equal "
            f"raw_report_sha256 ({entry.raw_report_sha256!r}) "
            f"on a non-sentinel row; both fields are the "
            f"SHA-256 hex digest of the same raw NCU report "
            f"bytes and a normal write produces them from one "
            f"capture, so divergence indicates corruption or "
            f"a partially-migrated entry",
        )


def validate_label_ledger(ledger: LabelLedger) -> None:
    """Raise `sweep_ledger_schema_violation` on:
      * schema_version not equal to
        `LABEL_LEDGER_SCHEMA_VERSION`.
      * empty entries AND empty unavailable (a ledger that
        records nothing is meaningless).
      * a cell name appearing in both entries and unavailable
        (the per-cell row must be either captured or
        unavailable, not both).
      * any entry whose `predictor_bucket` is not in the
        canonical `PREDICTOR_BUCKETS` enum.
      * any entry whose `cache_initial_state` is empty.
      * any unavailable row whose `predictor_bucket` is not
        in the canonical enum.
      * any unavailable row whose `unavailable_reason` is not
        in `LEDGER_UNAVAILABLE_TYPED_REASONS`.
      * any entry whose `cell_name` is empty.
    """
    if not isinstance(ledger, LabelLedger):
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"ledger must be LabelLedger; got "
            f"{type(ledger).__name__}",
        )
    if ledger.schema_version != LABEL_LEDGER_SCHEMA_VERSION:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"ledger.schema_version is "
            f"{ledger.schema_version!r}; expected "
            f"{LABEL_LEDGER_SCHEMA_VERSION!r}",
        )
    if not ledger.entries and not ledger.unavailable:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            "ledger has no entries and no unavailable rows; "
            "a ledger that records nothing is meaningless",
        )
    seen_cell_names: dict[str, str] = {}
    for entry in ledger.entries:
        if not entry.cell_name or not entry.cell_name.strip():
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                "ledger entry has empty cell_name",
            )
        if entry.predictor_bucket not in PREDICTOR_BUCKETS:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"entry {entry.cell_name!r}: "
                f"predictor_bucket "
                f"{entry.predictor_bucket!r} is not in "
                f"{sorted(PREDICTOR_BUCKETS)}",
            )
        if not entry.cache_initial_state or not entry.cache_initial_state.strip():
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"entry {entry.cell_name!r}: "
                f"cache_initial_state is empty",
            )
        if entry.cell_name in seen_cell_names:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"cell {entry.cell_name!r} appears twice "
                f"in the ledger "
                f"(first as {seen_cell_names[entry.cell_name]}, "
                f"now as entry)",
            )
        # Framework-contract 7-tuple enforcement: each real
        # LedgerEntry must yield a valid CellJoinKey via
        # the deployed factory helper. The helper runs
        # full validation (non-empty free-form fields +
        # SHA-256 shape on binary_sha256 +
        # kernel_signature_sha256), so a malformed entry
        # with empty shape_point / source_commit /
        # kernel_signature_sha256 fails closed here at
        # the schema-validator boundary rather than
        # propagating to the future predictor bundle
        # factory.
        try:
            cell_join_key_from_ledger_entry(entry)
        except FrameworkContractError as exc:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"entry {entry.cell_name!r}: framework-"
                f"contract join-key validation failed "
                f"({exc.typed_reason}: {exc.message})",
            ) from exc
        # NCU-provenance enforcement: a `LedgerEntry` must carry
        # exactly one of two shapes — (a) the explicit legacy-
        # sentinel shape (every string provenance field equals
        # `_LEGACY_PROVENANCE_SENTINEL` AND `command_argv ==
        # (_LEGACY_PROVENANCE_SENTINEL,)`), used by the migrated
        # in-tree `label_ledger_v1_full_partition.toml`; OR
        # (b) the fully-populated normal write shape with every
        # provenance field non-empty + well-typed AND
        # `raw_csv_sha256 == raw_report_sha256` (the
        # transitional alias-equality contract; both fields are
        # SHA-256 hex digests of the same raw NCU report bytes).
        # Anything else — all-blank, partial-sentinel,
        # partially-populated, or mismatched raw-report hashes —
        # is rejected. The write path constructs full provenance
        # from `harvest_ncu_label_with_provenance()` so divergent
        # hashes indicate corruption.
        _validate_entry_ncu_provenance(entry)
        seen_cell_names[entry.cell_name] = "entry"
    for u in ledger.unavailable:
        if not u.cell_name or not u.cell_name.strip():
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                "ledger unavailable row has empty cell_name",
            )
        if u.predictor_bucket not in PREDICTOR_BUCKETS:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"unavailable {u.cell_name!r}: "
                f"predictor_bucket "
                f"{u.predictor_bucket!r} is not "
                f"in {sorted(PREDICTOR_BUCKETS)}",
            )
        if u.unavailable_reason not in LEDGER_UNAVAILABLE_TYPED_REASONS:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"unavailable {u.cell_name!r}: "
                f"unavailable_reason "
                f"{u.unavailable_reason!r} is not in the "
                f"closed enum "
                f"{sorted(LEDGER_UNAVAILABLE_TYPED_REASONS)}",
            )
        if not u.warmup_protocol or not u.warmup_protocol.strip():
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"unavailable {u.cell_name!r}: "
                f"warmup_protocol is empty",
            )
        if not u.cache_initial_state or not u.cache_initial_state.strip():
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"unavailable {u.cell_name!r}: "
                f"cache_initial_state is empty",
            )
        if u.cell_name in seen_cell_names:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"cell {u.cell_name!r} appears "
                f"twice in the ledger "
                f"(first as {seen_cell_names[u.cell_name]}, "
                f"now as unavailable)",
            )
        seen_cell_names[u.cell_name] = "unavailable"


def validate_label_ledger_complete_for_manifest(
    ledger: LabelLedger, expected_cells: Iterable[str],
) -> None:
    """Raise `sweep_ledger_schema_violation` when the union
    of entry cell names and unavailable cell names does not
    equal `expected_cells`.

    Used at the artifact-emission path to catch subset
    ledgers that pass the basic validator. The expected
    set is normally
    `PREDICTOR_DESIGN_COVERAGE_BY_CELL.keys()` (33 cells)
    but callers may pass a smaller set for cohort-restricted
    sweeps.
    """
    validate_label_ledger(ledger)
    expected = set(expected_cells)
    captured: set[str] = set()
    captured.update(entry.cell_name for entry in ledger.entries)
    captured.update(u.cell_name for u in ledger.unavailable)
    only_in_expected = sorted(expected - captured)
    only_in_captured = sorted(captured - expected)
    if only_in_expected or only_in_captured:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"ledger cell-set drift from expected:\n"
            f"  expected-only: {only_in_expected}\n"
            f"  captured-only: {only_in_captured}",
        )


# ---------------------------------------------------------------------------
# Manifest-driven sweep-spec builder
# ---------------------------------------------------------------------------


def _cohort_a_loop_basename(canonical_basename: str) -> str:
    """Map the manifest's cohort A canonical basename to the
    `_loop` variant used by the NCU sweep. The single-launch
    single-launch microbench binaries are NCU-unprofilable (no measured
    launch after warmup) so the sweep targets their looped
    counterparts.

    Smoke-loop cells already carry the `_loop` suffix in the
    manifest; they pass through unchanged. Partition-typed-
    unavailable cells (calib_local_spill, calib_shared_*)
    likewise pass through; the sweep emits them as
    LedgerUnavailable rows and never invokes their binary.
    """
    if canonical_basename.endswith("_loop"):
        return canonical_basename
    return canonical_basename + "_loop"


# Mapping of canonical cohort A cells to whether they should
# be looped + which `partition_typed_unavailable_*` reason
# they carry when partition-typed-unavailable.
_PARTITION_TYPED_UNAVAILABLE_PREFIX = "partition_typed_unavailable_"


# Cohort B cells the sweep cannot label on sm_86 in Phase 1.
# Keyed by manifest cell_name; value is the typed reason
# from `LEDGER_UNAVAILABLE_TYPED_REASONS`.
_COHORT_B_SM_100_REQUIRED: dict[str, str] = {
    "problem_003_fp8_mlp_gate_up_projection":
        "kernel_requires_sm_100_features",
    "problem_020_nvfp4_linear_layer":
        "kernel_requires_sm_100_features",
    "problem_029_nvfp4_fused_mlp_silu_gating":
        "kernel_requires_sm_100_features",
    "problem_020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048":
        "kernel_requires_sm_100_features",
}


def build_sweep_specs_from_manifest(
    manifest: CalibrationCorpusManifest,
    *,
    repo_root: Path,
    cohort_b_binary_root: Path | None = None,
) -> tuple[LedgerCellSpec, ...]:
    """Build the `LedgerCellSpec` tuple for a full-partition
    sweep from a loaded calibration corpus manifest.

    Resolves each cohort A cell to its `_loop` variant under
    `repo_root / cohort_a.binary_root` and each cohort B cell
    to the operator-supplied `cohort_b_binary_root` (default:
    `repo_root / "nvbit_tracer/metric_extractor/tests/calibration_cohort_b"`).

    Partition-typed-unavailable cells (calib_local_spill,
    calib_shared_*, problem_018_*, problem_071_*) are
    emitted with `binary_path=None` and an
    `unavailable_reason` matching their partition typed
    reason. Cohort B cells that require sm_100+ features OR
    the attention template suite are emitted with
    `binary_path=None` and the appropriate constraint
    reason.

    Asserts the resulting cell set equals the manifest's
    own cohort A + cohort B cell set AND that every emitted
    cell is mapped in `PREDICTOR_DESIGN_COVERAGE_BY_CELL`
    (the partition is a superset of the calibration corpus
    once the host-arch manifest extension lands). Raises
    `sweep_ledger_schema_violation` on either drift.
    """
    cohort_a_root = repo_root / manifest.cohort_a_binary_root
    cohort_b_root = (
        cohort_b_binary_root
        if cohort_b_binary_root is not None
        else repo_root
        / "nvbit_tracer/metric_extractor/tests/calibration_cohort_b"
    )
    specs: list[LedgerCellSpec] = []
    for cell in manifest.cohort_a_cells:
        canonical = lookup_predictor_design_coverage(cell.cell_name)
        if canonical.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            reason = (
                _PARTITION_TYPED_UNAVAILABLE_PREFIX
                + canonical.typed_unavailable_reason
            )
            specs.append(LedgerCellSpec(
                cell_name=cell.cell_name,
                binary_path=None,
                launch_skip_count=0,
                warmup_protocol_name="ncu_partition_typed_unavailable",
                unavailable_reason=reason,
                shape_point=cell.shape_point,
                kernel_source_path=None,
            ))
            continue
        loop_basename = _cohort_a_loop_basename(cell.binary_basename)
        binary_path = cohort_a_root / loop_basename
        specs.append(LedgerCellSpec(
            cell_name=cell.cell_name,
            binary_path=binary_path,
            launch_skip_count=5,
            warmup_protocol_name=WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
            shape_point=cell.shape_point,
            kernel_source_path=binary_path.with_suffix(".cu"),
            measured_cache_initial_state=cell.measured_cache_initial_state,
        ))
    for cell in manifest.cohort_b_cells:
        canonical = lookup_predictor_design_coverage(cell.cell_name)
        if canonical.predictor_bucket == PREDICTOR_BUCKET_TYPED_UNAVAILABLE:
            reason = (
                _PARTITION_TYPED_UNAVAILABLE_PREFIX
                + canonical.typed_unavailable_reason
            )
            specs.append(LedgerCellSpec(
                cell_name=cell.cell_name,
                binary_path=None,
                launch_skip_count=0,
                warmup_protocol_name="ncu_partition_typed_unavailable",
                unavailable_reason=reason,
                shape_point=cell.shape_point,
                kernel_source_path=None,
            ))
            continue
        if cell.cell_name in _COHORT_B_SM_100_REQUIRED:
            specs.append(LedgerCellSpec(
                cell_name=cell.cell_name,
                binary_path=None,
                launch_skip_count=0,
                warmup_protocol_name="ncu_unavailable",
                unavailable_reason=_COHORT_B_SM_100_REQUIRED[cell.cell_name],
                shape_point=cell.shape_point,
                kernel_source_path=None,
            ))
            continue
        cohort_b_binary = cohort_b_root / cell.cell_name
        specs.append(LedgerCellSpec(
            cell_name=cell.cell_name,
            binary_path=cohort_b_binary,
            launch_skip_count=5,
            warmup_protocol_name=WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
            shape_point=cell.shape_point,
            kernel_source_path=cohort_b_binary.with_suffix(".cu"),
            measured_cache_initial_state=cell.measured_cache_initial_state,
        ))
    spec_cells = {s.cell_name for s in specs}
    manifest_cells = {
        cell.cell_name
        for cell in (
            *manifest.cohort_a_cells,
            *manifest.cohort_b_cells,
        )
    }
    only_in_specs = sorted(spec_cells - manifest_cells)
    only_in_manifest = sorted(manifest_cells - spec_cells)
    if only_in_specs or only_in_manifest:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"manifest spec cell-set drifts from the "
            f"manifest input cells:\n"
            f"  specs-only: {only_in_specs}\n"
            f"  manifest-only: {only_in_manifest}",
        )
    partition_cells = set(PREDICTOR_DESIGN_COVERAGE_BY_CELL)
    spec_cells_outside_partition = sorted(spec_cells - partition_cells)
    if spec_cells_outside_partition:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"manifest spec contains cells unmapped in the "
            f"deployed partition:\n"
            f"  specs-only: {spec_cells_outside_partition}",
        )
    return tuple(specs)


# ---------------------------------------------------------------------------
# TOML serialization
# ---------------------------------------------------------------------------


def _toml_escape(value: str) -> str:
    out: list[str] = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_kv_str(key: str, value: str) -> str:
    return f'{key} = {_toml_escape(value)}\n'


def _toml_kv_int(key: str, value: int) -> str:
    return f'{key} = {int(value)}\n'


def _toml_kv_float(key: str, value: float) -> str:
    return f'{key} = {float(value)!r}\n'


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.calibration_sweep.serialize_ledger_to_toml')
def serialize_ledger_to_toml(
    ledger: LabelLedger, output_path: Path,
) -> None:
    """Write `ledger` to `output_path` as TOML.

    Validates before serializing so a malformed in-memory
    ledger cannot land on disk silently. The on-disk
    layout is:

      [meta]
      schema_version = "1.0"
      captured_at_utc = "..."
      host_arch = "..."
      host_descriptor_hash = "..."
      ncu_version = "..."
      driver_version = "..."

      [[entries]]
      cell_name = "..."
      predictor_bucket = "..."
      arch = "..."
      ncu_metric_version = "..."
      warmup_protocol = "..."
      cache_initial_state = "warm"
      launch_skip_count = 5
      binary_sha256 = "..."
      raw_csv_sha256 = "..."

      [entries.label]
      l1_global_load_hit_rate = 0.987...
      ... (all NcuLabelRecord fields)

      [[unavailable]]
      cell_name = "..."
      predictor_bucket = "..."
      unavailable_reason = "..."
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.calibration_sweep.serialize_ledger_to_toml')
    validate_label_ledger(ledger)
    lines: list[str] = []
    lines.append("[meta]\n")
    lines.append(_toml_kv_str("schema_version", ledger.schema_version))
    lines.append(_toml_kv_str("captured_at_utc", ledger.captured_at_utc))
    lines.append(_toml_kv_str("host_arch", ledger.host_arch))
    lines.append(_toml_kv_str("host_descriptor_hash", ledger.host_descriptor_hash))
    lines.append(_toml_kv_str("ncu_version", ledger.ncu_version))
    lines.append(_toml_kv_str("driver_version", ledger.driver_version))
    lines.append("\n")
    for entry in ledger.entries:
        lines.append("[[entries]]\n")
        lines.append(_toml_kv_str("cell_name", entry.cell_name))
        lines.append(_toml_kv_str("predictor_bucket", entry.predictor_bucket))
        lines.append(_toml_kv_str("arch", entry.arch))
        lines.append(_toml_kv_str("ncu_metric_version", entry.ncu_metric_version))
        lines.append(_toml_kv_str("warmup_protocol", entry.warmup_protocol))
        lines.append(_toml_kv_str(
            "cache_initial_state", entry.cache_initial_state,
        ))
        lines.append(_toml_kv_int("launch_skip_count", entry.launch_skip_count))
        lines.append(_toml_kv_str("binary_sha256", entry.binary_sha256))
        lines.append(_toml_kv_str("raw_csv_sha256", entry.raw_csv_sha256))
        lines.append(_toml_kv_str("shape_point", entry.shape_point))
        lines.append(_toml_kv_str("source_commit", entry.source_commit))
        lines.append(_toml_kv_str(
            "kernel_signature_sha256", entry.kernel_signature_sha256,
        ))
        # NCU-provenance fields are always emitted for every
        # entry — sentinel rows and fully-populated rows alike.
        # The serializer never elides them; downstream readers
        # rely on the field being present to fail closed on
        # silent omission.
        lines.append(_toml_kv_str("ncu_cli_path", entry.ncu_cli_path))
        lines.append(_toml_kv_str("ncu_cli_sha256", entry.ncu_cli_sha256))
        lines.append(_toml_kv_str(
            "ncu_version_stdout", entry.ncu_version_stdout,
        ))
        lines.append(_toml_kv_str("driver_version", entry.driver_version))
        lines.append(_toml_kv_str(
            "gpu_device_name", entry.gpu_device_name,
        ))
        lines.append(_toml_kv_str(
            "compute_capability", entry.compute_capability,
        ))
        lines.append(_toml_kv_str(
            "clock_policy_observed", entry.clock_policy_observed,
        ))
        argv_literal = "[" + ", ".join(
            _toml_escape(a) for a in entry.command_argv
        ) + "]"
        lines.append(
            f'command_argv = {argv_literal}\n'
        )
        lines.append(_toml_kv_str(
            "raw_report_sha256", entry.raw_report_sha256,
        ))
        lines.append("[entries.label]\n")
        lines.append(_toml_kv_float(
            "l1_global_load_hit_rate", entry.label.l1_global_load_hit_rate,
        ))
        lines.append(_toml_kv_int(
            "l1_global_load_sectors_total",
            entry.label.l1_global_load_sectors_total,
        ))
        lines.append(_toml_kv_int(
            "l1_global_load_sectors_hit",
            entry.label.l1_global_load_sectors_hit,
        ))
        lines.append(_toml_kv_float(
            "l2_read_hit_rate", entry.label.l2_read_hit_rate,
        ))
        lines.append(_toml_kv_int(
            "l2_read_sectors_total", entry.label.l2_read_sectors_total,
        ))
        lines.append(_toml_kv_int(
            "l2_read_sectors_hit", entry.label.l2_read_sectors_hit,
        ))
        lines.append(_toml_kv_str(
            "ncu_metric_version", entry.label.ncu_metric_version,
        ))
        lines.append(_toml_kv_str("binary_sha256", entry.label.binary_sha256))
        lines.append(_toml_kv_int(
            "launch_skip_count", entry.label.launch_skip_count,
        ))
        lines.append(_toml_kv_str("warmup_protocol", entry.label.warmup_protocol))
        lines.append("\n")
    for u in ledger.unavailable:
        lines.append("[[unavailable]]\n")
        lines.append(_toml_kv_str("cell_name", u.cell_name))
        lines.append(_toml_kv_str("predictor_bucket", u.predictor_bucket))
        lines.append(_toml_kv_str("unavailable_reason", u.unavailable_reason))
        lines.append(_toml_kv_int("launch_skip_count", u.launch_skip_count))
        lines.append(_toml_kv_str("warmup_protocol", u.warmup_protocol))
        lines.append(_toml_kv_str(
            "cache_initial_state", u.cache_initial_state,
        ))
        lines.append(_toml_kv_str("shape_point", u.shape_point))
        lines.append(_toml_kv_str("source_commit", u.source_commit))
        lines.append(_toml_kv_str(
            "kernel_signature_sha256", u.kernel_signature_sha256,
        ))
        lines.append("\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def _kind_name(kind: type | tuple[type, ...]) -> str:
    if isinstance(kind, tuple):
        return " | ".join(k.__name__ for k in kind)
    return kind.__name__


def _require_field(
    table: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    context: str,
) -> Any:
    if key not in table:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"{context} is missing required field {key!r}",
        )
    value = table[key]
    if isinstance(value, bool) and kind is not bool:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"{context}: field {key!r} is a boolean; "
            f"expected {_kind_name(kind)}",
        )
    if not isinstance(value, kind):
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"{context}: field {key!r} must be "
            f"{_kind_name(kind)}; got {type(value).__name__}",
        )
    return value


def load_ledger_from_toml(path: Path) -> LabelLedger:
    """Read-side round trip: load a `LabelLedger` from a
    TOML document on disk. Validates the schema version
    first, rejects unknown top-level tables, and produces
    the same frozen dataclass tree the in-memory driver
    builds.

    Raises `sweep_ledger_schema_violation` on any field
    type mismatch, missing required field, or schema
    version drift.
    """
    raw = path.read_bytes()
    try:
        doc = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"ledger TOML at {str(path)!r} is malformed: "
            f"{exc}",
        ) from exc
    meta = _require_field(doc, "meta", dict, "ledger root")
    schema_version = _require_field(
        meta, "schema_version", str, "[meta] table",
    )
    if schema_version != LABEL_LEDGER_SCHEMA_VERSION:
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            f"ledger schema_version on disk is "
            f"{schema_version!r}; this build supports "
            f"{LABEL_LEDGER_SCHEMA_VERSION!r}",
        )
    captured_at_utc = _require_field(
        meta, "captured_at_utc", str, "[meta] table",
    )
    host_arch = _require_field(meta, "host_arch", str, "[meta] table")
    host_descriptor_hash = _require_field(
        meta, "host_descriptor_hash", str, "[meta] table",
    )
    ncu_version = _require_field(meta, "ncu_version", str, "[meta] table")
    driver_version = _require_field(
        meta, "driver_version", str, "[meta] table",
    )
    entries_raw = doc.get("entries", [])
    unavailable_raw = doc.get("unavailable", [])
    if not isinstance(entries_raw, list):
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            "[[entries]] must be an array of tables",
        )
    if not isinstance(unavailable_raw, list):
        raise CalibrationSweepError(
            "sweep_ledger_schema_violation",
            "[[unavailable]] must be an array of tables",
        )
    entries: list[LedgerEntry] = []
    for index, table in enumerate(entries_raw):
        if not isinstance(table, dict):
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"[[entries]] index {index} is not a table",
            )
        context = f"[[entries]] index {index}"
        label_table = _require_field(table, "label", dict, context)
        label = NcuLabelRecord(
            l1_global_load_hit_rate=float(_require_field(
                label_table, "l1_global_load_hit_rate",
                (int, float), f"{context} [entries.label]",
            )),
            l1_global_load_sectors_total=int(_require_field(
                label_table, "l1_global_load_sectors_total",
                int, f"{context} [entries.label]",
            )),
            l1_global_load_sectors_hit=int(_require_field(
                label_table, "l1_global_load_sectors_hit",
                int, f"{context} [entries.label]",
            )),
            l2_read_hit_rate=float(_require_field(
                label_table, "l2_read_hit_rate",
                (int, float), f"{context} [entries.label]",
            )),
            l2_read_sectors_total=int(_require_field(
                label_table, "l2_read_sectors_total",
                int, f"{context} [entries.label]",
            )),
            l2_read_sectors_hit=int(_require_field(
                label_table, "l2_read_sectors_hit",
                int, f"{context} [entries.label]",
            )),
            ncu_metric_version=_require_field(
                label_table, "ncu_metric_version",
                str, f"{context} [entries.label]",
            ),
            binary_sha256=_require_field(
                label_table, "binary_sha256",
                str, f"{context} [entries.label]",
            ),
            launch_skip_count=int(_require_field(
                label_table, "launch_skip_count",
                int, f"{context} [entries.label]",
            )),
            warmup_protocol=_require_field(
                label_table, "warmup_protocol",
                str, f"{context} [entries.label]",
            ),
        )
        command_argv_raw = _require_field(
            table, "command_argv", list, context,
        )
        if not command_argv_raw:
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"{context}: field 'command_argv' must be a "
                f"non-empty TOML array of strings",
            )
        for arg_index, arg in enumerate(command_argv_raw):
            if not isinstance(arg, str):
                raise CalibrationSweepError(
                    "sweep_ledger_schema_violation",
                    f"{context}: command_argv[{arg_index}] "
                    f"must be a string; got "
                    f"{type(arg).__name__}",
                )
        entries.append(LedgerEntry(
            cell_name=_require_field(table, "cell_name", str, context),
            predictor_bucket=_require_field(
                table, "predictor_bucket", str, context,
            ),
            arch=_require_field(table, "arch", str, context),
            ncu_metric_version=_require_field(
                table, "ncu_metric_version", str, context,
            ),
            warmup_protocol=_require_field(
                table, "warmup_protocol", str, context,
            ),
            cache_initial_state=_require_field(
                table, "cache_initial_state", str, context,
            ),
            launch_skip_count=int(_require_field(
                table, "launch_skip_count", int, context,
            )),
            binary_sha256=_require_field(
                table, "binary_sha256", str, context,
            ),
            raw_csv_sha256=_require_field(
                table, "raw_csv_sha256", str, context,
            ),
            shape_point=_require_field(
                table, "shape_point", str, context,
            ),
            source_commit=_require_field(
                table, "source_commit", str, context,
            ),
            kernel_signature_sha256=_require_field(
                table, "kernel_signature_sha256", str, context,
            ),
            label=label,
            ncu_cli_path=_require_field(
                table, "ncu_cli_path", str, context,
            ),
            ncu_cli_sha256=_require_field(
                table, "ncu_cli_sha256", str, context,
            ),
            ncu_version_stdout=_require_field(
                table, "ncu_version_stdout", str, context,
            ),
            driver_version=_require_field(
                table, "driver_version", str, context,
            ),
            gpu_device_name=_require_field(
                table, "gpu_device_name", str, context,
            ),
            compute_capability=_require_field(
                table, "compute_capability", str, context,
            ),
            clock_policy_observed=_require_field(
                table, "clock_policy_observed", str, context,
            ),
            command_argv=tuple(command_argv_raw),
            raw_report_sha256=_require_field(
                table, "raw_report_sha256", str, context,
            ),
        ))
    unavailable_rows: list[LedgerUnavailable] = []
    for index, table in enumerate(unavailable_raw):
        if not isinstance(table, dict):
            raise CalibrationSweepError(
                "sweep_ledger_schema_violation",
                f"[[unavailable]] index {index} is not a table",
            )
        context = f"[[unavailable]] index {index}"
        unavailable_rows.append(LedgerUnavailable(
            cell_name=_require_field(table, "cell_name", str, context),
            predictor_bucket=_require_field(
                table, "predictor_bucket", str, context,
            ),
            unavailable_reason=_require_field(
                table, "unavailable_reason", str, context,
            ),
            launch_skip_count=int(_require_field(
                table, "launch_skip_count", int, context,
            )),
            warmup_protocol=_require_field(
                table, "warmup_protocol", str, context,
            ),
            cache_initial_state=_require_field(
                table, "cache_initial_state", str, context,
            ),
            shape_point=_require_field(
                table, "shape_point", str, context,
            ),
            source_commit=_require_field(
                table, "source_commit", str, context,
            ),
            kernel_signature_sha256=_require_field(
                table, "kernel_signature_sha256", str, context,
            ),
        ))
    ledger = LabelLedger(
        schema_version=schema_version,
        captured_at_utc=captured_at_utc,
        host_arch=host_arch,
        host_descriptor_hash=host_descriptor_hash,
        ncu_version=ncu_version,
        driver_version=driver_version,
        entries=tuple(entries),
        unavailable=tuple(unavailable_rows),
    )
    validate_label_ledger(ledger)
    return ledger


__all__ = [
    "CALIBRATION_SWEEP_TYPED_REASONS",
    "CalibrationSweepError",
    "LABEL_LEDGER_SCHEMA_VERSION",
    "LEDGER_UNAVAILABLE_TYPED_REASONS",
    "LabelLedger",
    "LedgerCellSpec",
    "LedgerEntry",
    "LedgerUnavailable",
    "build_sweep_specs_from_manifest",
    "load_ledger_from_toml",
    "run_calibration_sweep",
    "serialize_ledger_to_toml",
    "validate_label_ledger",
    "validate_label_ledger_complete_for_manifest",
]
