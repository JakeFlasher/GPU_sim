#!/usr/bin/env python3
"""Roster-wide NVBit trace capture toward gate M1 (plan §M1, sim input side).

Captures one NVBit dynamic trace per repeatability-matrix roster cell at its
timing shape, then validates every capture with the offline trace-contract
validator (``validate_trace_dir``). Standard portable_lcd slugs go through
``trace_activation_cell`` (identity + admission sidecar); the ac4 multi-case
cells go through a direct-injection path with per-case output dirs (the
orchestrator derives its output dir from the slug, which would collide).

NVBit-only: never co-run with NCU capture (repo rule). Traces are
timing-free (instruction/address streams), so the clock protocol does not
gate trace validity — but the manifest records the host clock state anyway.

Usage:
  .venv/bin/python tools/local_labels/roster_trace_capture.py \
      --roster tools/local_labels/repeatability_matrix_roster_full.json \
      --output-root results/nvbit_capture_staging/roster_trace_capture
"""
from __future__ import annotations

import argparse
import contextvars
import errno
import json
import math
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "nvbit_tracer"))

from tools.local_labels.activation_trace_orchestrator import (  # noqa: E402
    trace_activation_cell,
    _MAX_ACTIVATION_APP_ARGS,
    _MAX_ACTIVATION_APP_ARG_BYTES,
    _SOURCE_COMMIT_AUTHORITY_PATHS as _ACTIVATION_SOURCE_COMMIT_AUTHORITY_PATHS,
    _activation_output_plan,
    _assert_staging_output,
    _lock_activation_cell,
    _new_capture_attempt_id,
    _raw_generation_record,
    _reject_output_dir_extras,
    _require_app_arguments,
    _require_hex_identity,
    _require_identity_string,
    _require_nonnegative_integer,
    _retire_matching_locator,
    _trace_generation_commit_limits,
    _validate_trace_admission_identities,
    _verify_committed_trace_generation,
)
from tools.local_labels.immutable_artifact import (  # noqa: E402
    VERIFY_RESOURCE_BOUND,
    ImmutableArtifactError,
    ImmutableArtifactResourceError,
    MetadataSnapshot,
    PinnedDirectoryReader,
    RetainedRegular,
    close_retained_regular_files_authenticated,
    create_exclusive_directory_stage,
    open_retained_executable,
    open_retained_regular,
    preserve_retained_regular_file,
    publish_regular_file,
    publish_retained_regular_file,
    read_regular_snapshot,
    require_file_descriptor_headroom,
    retire_regular_file,
    safe_relative_path,
    strict_json_loads,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    TRUSTED_GIT_EXECUTABLE,
    run_retained_git,
)
from tools.local_labels.roster_path_resolver import resolve_cell_root  # noqa: E402
from tools.nvbit_tracer import capture_process as nvbit_process  # noqa: E402
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
    EXTERNAL_TRACER_SOURCE_KIND,
    ExternalTracerAuthority,
    require_external_tracer_authority,
    verify_retained_tracer_authority,
)
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    DEFAULT_PB_LIB,
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
    sanitized_capture_environment,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    EXPECTED_MEMORY_PAYLOAD_CLASSES,
    EXPECTED_TRACE_CONTRACT_ID,
    TraceAdmission,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)


_DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "nvbit_capture_staging" / "roster_trace_capture"
)
_ACTIVE_RETAINED_GIT: contextvars.ContextVar[object | None] = (
    contextvars.ContextVar("roster_retained_git", default=None)
)

# One roster transaction retains all cell source/tree capabilities until the
# aggregate publication closes. The canonical roster has 17 cells; 32 leaves
# deliberate growth room while bounding descriptors, launch records, and JSON.
MAX_RETAINED_ROSTER_CELLS = 32
_MAX_ROSTER_LAUNCHES_PER_CELL = 1024
_MAX_ROSTER_APP_ARGS_PER_CELL = _MAX_ACTIVATION_APP_ARGS
_MAX_ROSTER_APP_ARG_BYTES = _MAX_ACTIVATION_APP_ARG_BYTES
_MAX_ROSTER_INPUT_BYTES = 1024 * 1024
# The compositional preflight below reserves 5 KiB for every admitted launch,
# 64 KiB of fixed identity/path/diagnostic material per cell, and a 1 MiB
# envelope. At 32 cells x 1024 launches this is < 192 MiB, so every accepted
# roster can serialize its worst-case bounded manifest without late capture loss.
_MAX_ROSTER_MANIFEST_BYTES = 192 * 1024 * 1024
_MAX_ROSTER_MANIFEST_ENVELOPE_BYTES = 1024 * 1024
_MAX_ROSTER_CELL_MANIFEST_BYTES = 64 * 1024
_MAX_ROSTER_LAUNCH_MANIFEST_BYTES = 5 * 1024
_MAX_ROSTER_KERNEL_NAME_BYTES = 512
_MAX_ROSTER_JSON_INTEGER = (1 << 53) - 1
_ROSTER_DESCRIPTOR_BASE_HEADROOM = 128
_ROSTER_DESCRIPTOR_HEADROOM_PER_CELL = 128
_CLOSE_DIAGNOSTIC_BYTES = 400
_DIRECT_CAPTURE_TIMEOUT_SECONDS = 15 * 60
_DIRECT_CAPTURE_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_DIRECT_CAPTURE_TIMEOUT_CODE = "roster_nvbit_capture_timeout"
_DIRECT_CAPTURE_LAUNCH_CODE = "roster_nvbit_capture_launch_failed"
_DIRECT_FAILURE_STREAM_MAX_BYTES = _DIRECT_CAPTURE_MAX_OUTPUT_BYTES
_DIRECT_CAPTURE_RESOURCE_ERRNOS = frozenset({
    errno.EAGAIN,
    errno.EMFILE,
    errno.ENFILE,
    errno.ENOMEM,
    errno.ENOSPC,
})


class RosterDirectCaptureError(RuntimeError):
    """A typed, byte-preserving direct-roster capture failure."""

    TYPED_REASON = "capture_exception"

    def __init__(
        self,
        detail: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        process_id: int | None = None,
        observed_output_bytes_lower_bound: int | None = None,
    ) -> None:
        self.detail = detail
        self.stdout = stdout
        self.stderr = stderr
        self.process_id = process_id
        self.observed_output_bytes_lower_bound = observed_output_bytes_lower_bound
        super().__init__(f"{self.TYPED_REASON}: {detail}")


class RosterDirectCaptureTimeout(RosterDirectCaptureError):
    TYPED_REASON = "capture_timeout"


class RosterDirectCaptureOutputLimitExceeded(RosterDirectCaptureError):
    TYPED_REASON = "capture_output_limit"


class RosterDirectCaptureLaunchFailed(RosterDirectCaptureError):
    TYPED_REASON = "capture_launch_failed"


class RosterDirectCaptureCleanupFailed(RosterDirectCaptureError):
    TYPED_REASON = "capture_cleanup_failed"


class RosterDirectCaptureNonzero(RosterDirectCaptureError):
    TYPED_REASON = "capture_nonzero"

    def __init__(
        self,
        *,
        returncode: int,
        stdout: bytes,
        stderr: bytes,
        failure_evidence: tuple[dict, ...] = (),
    ) -> None:
        self.returncode = returncode
        self.failure_evidence = failure_evidence
        diagnostic = _bounded_diagnostic_text(stderr, max_bytes=300)
        super().__init__(
            f"child returned {returncode}; stderr (tail): {diagnostic!r}",
            stdout=stdout,
            stderr=stderr,
        )


class RosterDirectCaptureResourceUnavailable(ImmutableArtifactResourceError):
    TYPED_REASON = "trace_snapshot_resource_unavailable"

    def __init__(
        self,
        detail: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        process_id: int | None = None,
        observed_output_bytes_lower_bound: int | None = None,
    ) -> None:
        self.detail = detail
        self.stdout = stdout
        self.stderr = stderr
        self.process_id = process_id
        self.observed_output_bytes_lower_bound = observed_output_bytes_lower_bound
        super().__init__(
            f"{VERIFY_RESOURCE_BOUND}: roster direct capture: {detail}"
        )


def _direct_capture_resource_failure(error: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        if (
            isinstance(current, OSError)
            and current.errno in _DIRECT_CAPTURE_RESOURCE_ERRNOS
        ):
            return True
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


def _map_direct_capture_error(
    error: nvbit_process.NvbitCaptureProcessError,
) -> RosterDirectCaptureError | RosterDirectCaptureResourceUnavailable:
    fields = {
        "stdout": error.stdout,
        "stderr": error.stderr,
        "process_id": error.process_id,
        "observed_output_bytes_lower_bound": (
            error.observed_output_bytes_lower_bound
        ),
    }
    if error.process_cleanup_incomplete:
        return RosterDirectCaptureCleanupFailed(error.detail, **fields)
    if error.code == nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT:
        return RosterDirectCaptureOutputLimitExceeded(error.detail, **fields)
    if error.code == _DIRECT_CAPTURE_TIMEOUT_CODE:
        return RosterDirectCaptureTimeout(error.detail, **fields)
    if _direct_capture_resource_failure(error):
        return RosterDirectCaptureResourceUnavailable(error.detail, **fields)
    return RosterDirectCaptureLaunchFailed(error.detail, **fields)


def _bounded_diagnostic_text(value: object, *, max_bytes: int) -> str:
    if isinstance(value, bytes):
        encoded = value[-max_bytes:]
        return encoded.decode("utf-8", errors="replace")
    text = str(value)
    encoded = text.encode("utf-8", errors="replace")[:max_bytes]
    return encoded.decode("utf-8", errors="replace")


def _retain_direct_failure_streams(
    *,
    sources: ExitStack,
    parent: Path,
    pinned_parent: PinnedDirectoryReader,
    cell_dir: Path,
    stdout: bytes,
    stderr: bytes,
) -> tuple[dict, ...]:
    evidence: list[dict] = []
    for role, payload in (("stdout", stdout), ("stderr", stderr)):
        if type(payload) is not bytes or len(payload) > _DIRECT_FAILURE_STREAM_MAX_BYTES:
            raise ValueError(f"direct capture {role} bytes exceed retained bound")
        retained = sources.enter_context(publish_retained_regular_file(
            parent,
            f"capture_nonzero.{role}.bin",
            payload,
            pinned_parent=pinned_parent,
        ))
        retained.verify_unchanged()
        evidence.append({
            "role": role,
            "path": retained.path.relative_to(cell_dir).as_posix(),
            "size_bytes": retained.size_bytes,
            "sha256": retained.sha256,
        })
    return tuple(evidence)


_DIRECT_SOURCE_COMMIT_AUTHORITY_PATHS = (
    "tools/local_labels/roster_trace_capture.py",
    *_ACTIVATION_SOURCE_COMMIT_AUTHORITY_PATHS,
)


def _git_head(retained_git: object | None = None) -> str:
    if retained_git is None:
        retained_git = _ACTIVE_RETAINED_GIT.get()
    if retained_git is None:
        with open_retained_executable(TRUSTED_GIT_EXECUTABLE) as owned_git:
            return _git_head(owned_git)
    result = run_retained_git(
        retained_git,
        repo_root=REPO_ROOT,
        arguments=("rev-parse", "--verify", "HEAD"),
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot resolve roster source_commit from Git HEAD")
    return _require_hex_identity(
        result.stdout.strip(), label="roster source_commit", length=40,
    )


def _direct_source_path(binary: Path) -> Path:
    source_stem = binary.name[:-4] if binary.name.endswith("_run") else binary.name
    source = binary.with_name(f"{source_stem}.cu")
    if not source.is_file():
        raise FileNotFoundError(f"direct roster source is missing: {source}")
    return source


def _resolve_direct_source_commit(
    source: Path,
    retained_authority: dict[str, object],
    retained_git: object | None = None,
    *,
    _legacy_head_call: bool = False,
) -> str:
    if retained_git is None:
        active_git = _ACTIVE_RETAINED_GIT.get()
        if active_git is not None:
            return _resolve_direct_source_commit(
                source,
                retained_authority,
                active_git,
                _legacy_head_call=True,
            )
        with open_retained_executable(TRUSTED_GIT_EXECUTABLE) as owned_git:
            return _resolve_direct_source_commit(
                source,
                retained_authority,
                owned_git,
                _legacy_head_call=True,
            )
    try:
        source_relative = source.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"direct roster source is outside repository: {source}") from exc
    authority_paths = (*_DIRECT_SOURCE_COMMIT_AUTHORITY_PATHS, source_relative)
    expected = set(authority_paths)
    if (
        len(authority_paths) != len(expected)
        or set(retained_authority) != expected
        or len(retained_authority) != len(expected)
    ):
        missing = sorted(expected - set(retained_authority))
        extra = sorted(set(retained_authority) - expected)
        raise RuntimeError(
            "direct roster source-commit authority set is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    source_commit = (
        _git_head() if _legacy_head_call else _git_head(retained_git)
    )
    status = run_retained_git(
        retained_git,
        repo_root=REPO_ROOT,
        arguments=(
            "status", "--porcelain=v1", "--untracked-files=all", "--",
            *authority_paths,
        ),
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("cannot verify direct roster source-commit cleanliness")
    if status.stdout.strip():
        raise RuntimeError(
            "direct roster source-commit authority is dirty or untracked: "
            f"{status.stdout.strip()!r}"
        )
    for relative_path in authority_paths:
        committed = run_retained_git(
            retained_git,
            repo_root=REPO_ROOT,
            arguments=("show", f"{source_commit}:{relative_path}"),
        )
        if committed.returncode != 0:
            raise RuntimeError(
                f"direct roster authority is absent from source_commit "
                f"{source_commit}: {relative_path}"
            )
        retained = retained_authority[relative_path]
        if committed.stdout != retained.read_bytes():
            raise RuntimeError(
                "retained direct roster authority bytes disagree with "
                f"source_commit for {relative_path}"
            )
        retained.verify_unchanged()
    final_head = (
        _git_head() if _legacy_head_call else _git_head(retained_git)
    )
    if final_head != source_commit:
        raise RuntimeError(
            "roster source_commit changed during direct authority verification"
        )
    return source_commit


def _verify_roster_source_commit_head(
    source_commit: str, retained_git: object | None = None,
) -> None:
    observed_head = (
        _git_head() if retained_git is None else _git_head(retained_git)
    )
    if observed_head != source_commit:
        raise RuntimeError("roster source_commit changed during capture publication")


def _verify_retained_direct_authority(
    retained_authority: dict[str, object],
) -> None:
    for retained in retained_authority.values():
        retained.verify_unchanged()


def _raise_roster_resource_error(detail: str) -> None:
    raise ImmutableArtifactResourceError(
        f"{VERIFY_RESOURCE_BOUND}: roster trace capture: {detail}"
    )


def _validate_roster_entries(
    roster: object,
    *,
    require_binary: bool,
) -> list[dict]:
    if type(roster) is not list:
        raise ValueError("roster must be an exact JSON array")
    if not roster:
        raise ValueError("roster must contain at least one cell")
    if len(roster) > MAX_RETAINED_ROSTER_CELLS:
        _raise_roster_resource_error(
            f"cell count {len(roster)} exceeds retained-capability maximum "
            f"{MAX_RETAINED_ROSTER_CELLS}"
        )
    total_argument_bytes = 0
    for index, cell in enumerate(roster):
        if type(cell) is not dict:
            raise ValueError(f"roster row {index} must be an exact object")
        _require_identity_string(
            cell.get("slug"), label=f"roster row {index} slug", component=True,
        )
        _require_identity_string(
            cell.get("shape_point"),
            label=f"roster row {index} shape_point",
            component=True,
        )
        app_args = cell.get("app_args", [])
        if type(app_args) is not list:
            raise ValueError(
                f"roster row {index} app_args must be an exact array"
            )
        if len(app_args) > _MAX_ROSTER_APP_ARGS_PER_CELL:
            _raise_roster_resource_error(
                f"row {index} app_args count {len(app_args)} exceeds "
                f"{_MAX_ROSTER_APP_ARGS_PER_CELL}"
            )
        checked_args = _require_app_arguments(
            tuple(app_args), label=f"roster row {index} app_args",
        )
        total_argument_bytes += sum(
            len(argument.encode("utf-8")) for argument in checked_args
        )
        if total_argument_bytes > _MAX_ROSTER_APP_ARG_BYTES:
            _raise_roster_resource_error(
                "aggregate app-argument bytes exceed "
                f"{_MAX_ROSTER_APP_ARG_BYTES}"
            )
        if require_binary or "binary" in cell:
            binary = _require_identity_string(
                cell.get("binary"),
                label=f"roster row {index} binary",
                max_bytes=4096,
            )
            safe_relative_path(binary)
    return roster


def _validate_plan_vector(
    plans: object,
    *,
    expected_cells: int,
) -> list[dict[str, Path | bool]]:
    if type(plans) is not list or len(plans) != expected_cells:
        raise ValueError("roster plans must exactly match roster cell count")
    for index, plan in enumerate(plans):
        if type(plan) is not dict:
            raise ValueError(f"roster plan {index} must be an exact object")
        if type(plan.get("portable")) is not bool:
            raise ValueError(f"roster plan {index} portable must be boolean")
        for key in ("cell_dir", "traces_folder", "workload_output_root"):
            if not isinstance(plan.get(key), Path):
                raise ValueError(
                    f"roster plan {index} {key} must be a Path capability"
                )
    return plans


def _require_roster_descriptor_budget(cell_count: int) -> None:
    _require_nonnegative_integer(cell_count, label="roster cell_count")
    require_file_descriptor_headroom(
        _ROSTER_DESCRIPTOR_BASE_HEADROOM
        + cell_count * _ROSTER_DESCRIPTOR_HEADROOM_PER_CELL,
        label="roster-wide retained capabilities",
    )


def _require_roster_manifest_budget(cell_count: int) -> int:
    """Reserve enough aggregate space for every accepted launch/name bound."""
    _require_nonnegative_integer(cell_count, label="roster cell_count")
    worst_case_bytes = (
        _MAX_ROSTER_MANIFEST_ENVELOPE_BYTES
        + cell_count
        * (
            _MAX_ROSTER_CELL_MANIFEST_BYTES
            + _MAX_ROSTER_LAUNCHES_PER_CELL
            * _MAX_ROSTER_LAUNCH_MANIFEST_BYTES
        )
    )
    if worst_case_bytes > _MAX_ROSTER_MANIFEST_BYTES:
        _raise_roster_resource_error(
            "worst-case aggregate manifest budget "
            f"{worst_case_bytes} exceeds {_MAX_ROSTER_MANIFEST_BYTES} bytes"
        )
    return worst_case_bytes


def _roster_capture_plan(
    *,
    roster: list[dict],
    sku: str,
    arch_dir: str,
    output_root: Path | str,
    manifest_out: Path | str | None,
) -> tuple[list[dict[str, Path | bool]], Path]:
    """Clear every delegated/direct output before the first capture."""

    roster = _validate_roster_entries(roster, require_binary=False)
    _require_roster_manifest_budget(len(roster))
    sku = _require_identity_string(sku, label="roster plan sku", component=True)
    arch_dir = _require_identity_string(
        arch_dir, label="roster plan arch_dir", component=True,
    )
    root = _assert_staging_output(output_root, operation="select roster output root")
    manifest = _assert_staging_output(
        manifest_out if manifest_out is not None else root / "capture_manifest.json",
        operation="write roster capture manifest",
    )
    plans: list[dict[str, Path | bool]] = []
    seen_identities: dict[tuple[str, str], int] = {}
    for index, cell in enumerate(roster):
        slug = cell["slug"]
        shape_point = cell["shape_point"]
        identity = (slug, shape_point)
        prior_index = seen_identities.get(identity)
        if prior_index is not None:
            raise ValueError(
                "duplicate roster trace output identity "
                f"slug={slug!r}, shape_point={shape_point!r} at rows "
                f"{prior_index} and {index}"
            )
        seen_identities[identity] = index
        extras = tuple(
            arg for arg in cell.get("app_args", [])
            if arg != f"--{shape_point}"
        )
        _reject_output_dir_extras(extras)
        portable = (
            REPO_ROOT / "workloads" / "portable_lcd" / slug
        ).is_dir()
        if portable:
            activation = _activation_output_plan(
                slug=slug,
                shape_point=shape_point,
                sku=sku,
                arch_dir=arch_dir,
                fallback_source="portable_lcd",
                output_root=root,
                extra_app_args=extras,
            )
            plans.append({"portable": True, **activation})
            continue
        direct_root = root / "direct" / slug
        direct = {
            "portable": False,
            "cell_dir": direct_root,
            "traces_folder": direct_root / f"trace_{shape_point}",
            "workload_output_root": direct_root / "workload_output",
            "portable_manifest": (
                direct_root / "workload_output" / slug / "manifest.json"
            ),
        }
        plans.append({
            name: (
                value if isinstance(value, bool) else _assert_staging_output(
                    value, operation=f"prepare direct {name}",
                )
            )
            for name, value in direct.items()
        })
    return plans, manifest


def _host_clock_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=persistence_mode,clocks.sm,clocks.mem",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return {"nvidia_smi_persistence_sm_mem": out[:4096]}
    except Exception as exc:  # pragma: no cover - informational only
        return {"nvidia_smi_error": str(exc)[:4096]}


def _all_admitted_launch_records(admission: TraceAdmission) -> list[dict]:
    """Serialize the complete, deterministically indexed admitted launch vector."""

    _validate_trace_admission_identities(
        admission, producer="direct roster trace",
    )
    if admission.launch_count > _MAX_ROSTER_LAUNCHES_PER_CELL:
        _raise_roster_resource_error(
            f"direct launch_count {admission.launch_count} exceeds per-cell "
            f"maximum {_MAX_ROSTER_LAUNCHES_PER_CELL}"
        )
    if admission.launch_count != len(admission.launches):
        raise RuntimeError(
            "direct roster trace: admitted launch_count "
            f"{admission.launch_count} disagrees with "
            f"{len(admission.launches)} retained launch record(s)"
        )
    indexes = [launch.launch_index for launch in admission.launches]
    expected_indexes = list(range(admission.launch_count))
    if indexes != expected_indexes:
        raise RuntimeError(
            "direct roster trace: admitted launch indexes must be contiguous "
            f"and ordered; got {indexes!r}, expected {expected_indexes!r}"
        )
    records = [asdict(launch) for launch in admission.launches]
    for launch_index, record in enumerate(records):
        kernel_name = record.get("kernel_name")
        if (
            type(kernel_name) is str
            and len(kernel_name.encode("utf-8")) > _MAX_ROSTER_KERNEL_NAME_BYTES
        ):
            _raise_roster_resource_error(
                f"direct launch {launch_index} kernel_name exceeds "
                f"{_MAX_ROSTER_KERNEL_NAME_BYTES} bytes"
            )
        for field, value in record.items():
            if (
                field != "kernel_name"
                and type(value) is int
                and value > _MAX_ROSTER_JSON_INTEGER
            ):
                _raise_roster_resource_error(
                    f"direct launch {launch_index} {field} exceeds bounded "
                    "JSON integer authority"
                )
        _validate_launch_record(
            record,
            expected_index=launch_index,
            label=f"direct launch {launch_index}",
        )
    return records


def _activation_capture_record(
    *,
    slug: str,
    cell_dir: Path,
    sidecar: dict,
) -> dict:
    """Project an admitted activation sidecar without dropping its identities."""

    slug = _require_identity_string(
        slug, label="activation capture record slug", component=True,
    )
    if type(sidecar) is not dict:
        raise ValueError("activation sidecar must be an exact object")
    _require_identity_string(
        sidecar.get("trace_capture_attempt_id"),
        label="activation trace_capture_attempt_id",
        component=True,
    )
    raw_generation_model = sidecar.get("raw_trace_generation")
    if type(raw_generation_model) is not dict:
        raise ValueError("activation raw_trace_generation must be an exact object")
    raw_generation = dict(raw_generation_model)
    _require_identity_string(
        raw_generation.get("path"),
        label="activation raw generation path",
        max_bytes=4096,
    )
    traces_folder = cell_dir / raw_generation["path"]
    identity_keys = (
        "source_commit",
        "binary_sha256",
        "nvbit_tracer_build_hash",
        "tracer_source_kind",
        "trace_contract_id",
        "memory_payload_classes",
        "trace_admission_sha256",
        "trace_filesystem_generation_sha256",
        "dynamic_trace_size_bytes",
        "dynamic_trace_sha256",
        "allocations_sidecar_size_bytes",
        "allocations_sidecar_sha256",
        "stats_size_bytes",
        "stats_sha256",
        "threadblock_payload_size_bytes",
        "threadblock_payload_manifest_sha256",
    )
    return {
        "slug": slug,
        "ok": True,
        "contract_ok": True,
        "trace_capture_attempt_id": sidecar["trace_capture_attempt_id"],
        "raw_trace_generation": raw_generation,
        "traces_folder": str(traces_folder),
        "trace_bytes": sidecar["trace_bytes"],
        "trace_pb_bytes": sidecar["trace_bytes"],
        "records_expected": sidecar["expected_record_count"],
        "records_admitted": sidecar["admitted_record_count"],
        "drop_rate": sidecar["actual_drop_rate"],
        **{key: sidecar[key] for key in identity_keys},
    }


class _RetainedDirectSources:
    """Retain one direct attempt's launch and raw-tree authorities."""

    def __init__(self) -> None:
        self.capture_sources = ExitStack()
        self.admitted_sources = ExitStack()
        self.raw_generation = None
        self.failure_record: dict | None = None

    def bind_failure_record(self, record: dict) -> None:
        if type(record) is not dict or record.get("ok") is not False:
            raise TypeError("direct failure diagnostic target must be a failed record")
        self.failure_record = record

    def __enter__(self) -> "_RetainedDirectSources":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        closure_errors: list[BaseException] = []
        for sources in (self.capture_sources, self.admitted_sources):
            try:
                sources.close()
            except BaseException as closure_error:
                closure_errors.append(closure_error)
        if self.raw_generation is not None:
            try:
                _verify_committed_trace_generation(self.raw_generation)
            except BaseException as closure_error:
                closure_errors.append(closure_error)
        if not closure_errors:
            return False
        primary = closure_errors[0]
        for additional in closure_errors[1:]:
            try:
                primary.add_note(f"additional retained-source failure: {additional}")
            except (AttributeError, TypeError):
                pass
        if exc is not None:
            if self.failure_record is not None:
                diagnostics = self.failure_record.setdefault(
                    "close_diagnostics", []
                )
                if type(diagnostics) is not list:
                    diagnostics = []
                    self.failure_record["close_diagnostics"] = diagnostics
                for closure_error in closure_errors:
                    diagnostics.append({
                        "typed_reason": "retained_source_close_failed",
                        "detail": _bounded_diagnostic_text(
                            closure_error, max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
                        ),
                    })
            try:
                exc.add_note(f"retained direct source closure failed: {primary}")
            except (AttributeError, TypeError):
                pass
            return False
        raise primary


def _capture_direct(
    slug: str,
    binary: Path,
    app_args: list[str],
    traces_folder: Path,
    workload_output_root: Path,
    *,
    retention_stack: ExitStack | None = None,
    attempt_id: str | None = None,
    tracer_authority: ExternalTracerAuthority | None = None,
) -> dict:
    """Capture one direct cell as immutable, descriptor-bound attempt evidence."""

    slug = _require_identity_string(
        slug, label="direct capture slug", component=True,
    )
    if not isinstance(binary, Path):
        raise ValueError("direct capture binary must be a Path")
    if type(app_args) is not list:
        raise ValueError("direct capture app_args must be an exact list")
    app_args = list(
        _require_app_arguments(tuple(app_args), label="direct capture app_args")
    )
    if retention_stack is not None and not isinstance(retention_stack, ExitStack):
        raise ValueError("direct capture retention_stack must be an ExitStack")
    if tracer_authority is None:
        tracer_authority = require_external_tracer_authority(repo_root=REPO_ROOT)
    elif not isinstance(tracer_authority, ExternalTracerAuthority):
        raise ValueError("direct capture tracer_authority is invalid")
    _reject_output_dir_extras(tuple(app_args))
    generation_root = _assert_staging_output(
        traces_folder, operation="prepare direct trace generation root",
    )
    workload_output_root = _assert_staging_output(
        workload_output_root, operation="prepare direct workload output",
    )
    if attempt_id is None:
        attempt_id = _new_capture_attempt_id()
    attempt_id = _require_identity_string(
        attempt_id, label="direct trace_capture_attempt_id", component=True,
    )
    owns_sources = retention_stack is None
    retained_guard = _RetainedDirectSources()
    if retention_stack is None:
        retained_guard.__enter__()
    else:
        retention_stack.enter_context(retained_guard)
    sources = retained_guard.capture_sources

    try:
        source = _direct_source_path(binary)
        source_relative = source.relative_to(REPO_ROOT).as_posix()
        retained_binary = sources.enter_context(
            open_retained_executable(binary)
        )
        retained_git = sources.enter_context(
            open_retained_executable(TRUSTED_GIT_EXECUTABLE)
        )
        git_token = _ACTIVE_RETAINED_GIT.set(retained_git)
        sources.callback(_ACTIVE_RETAINED_GIT.reset, git_token)
        retained_tracer = sources.enter_context(
            open_retained_regular(
                tracer_authority.path,
                max_bytes=EXTERNAL_TRACER_MAX_BYTES,
            )
        )
        verify_retained_tracer_authority(retained_tracer, tracer_authority)
        retained_authority: dict[str, object] = {}
        for relative_path in (
            *_DIRECT_SOURCE_COMMIT_AUTHORITY_PATHS,
            source_relative,
        ):
            retained_authority[relative_path] = sources.enter_context(
                open_retained_regular(REPO_ROOT / relative_path)
            )
        source_commit = _resolve_direct_source_commit(
            source, retained_authority,
        )
        stage = create_exclusive_directory_stage(generation_root, attempt_id)
        with stage:
            workload_output_root.mkdir(parents=True, exist_ok=True)
            workload_attempt = workload_output_root / attempt_id
            workload_attempt.mkdir(exist_ok=False)
            retained_workload_output = sources.enter_context(
                PinnedDirectoryReader(workload_attempt)
            )

            with retained_tracer_runtime_env(
                retained_tracer.path,
                base_env=sanitized_capture_environment(),
                pb_lib=DEFAULT_PB_LIB,
                runtime_manifest=default_runtime_manifest_path(retained_tracer.path),
                retained_descriptor=retained_tracer.descriptor,
            ) as retained_runtime:
                env = dict(retained_runtime.environment)
                env["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
                env["USER_DEFINED_FOLDERS"] = "1"
                env["TRACES_FOLDER"] = f"/proc/self/fd/{stage.descriptor}"
                command = [
                    retained_binary.proc_path,
                    *app_args,
                    "--output_dir",
                    f"/proc/self/fd/{retained_workload_output.descriptor}",
                ]
                inherited_descriptors = (
                    retained_binary.descriptor,
                    retained_tracer.descriptor,
                    stage.descriptor,
                    retained_workload_output.descriptor,
                    *retained_runtime.pass_fds,
                )
                retained_binary.verify_unchanged()
                retained_tracer.verify_unchanged()
                _verify_retained_direct_authority(retained_authority)
                _verify_roster_source_commit_head(source_commit)
                retained_workload_output.verify_path()
                t0 = time.perf_counter()
                try:
                    proc = nvbit_process.execute_capture_process(
                        tuple(command),
                        environment=env,
                        working_directory=str(binary.parent),
                        inherited_descriptors=inherited_descriptors,
                        timeout_seconds=_DIRECT_CAPTURE_TIMEOUT_SECONDS,
                        max_output_bytes=_DIRECT_CAPTURE_MAX_OUTPUT_BYTES,
                        timeout_code=_DIRECT_CAPTURE_TIMEOUT_CODE,
                        launch_code=_DIRECT_CAPTURE_LAUNCH_CODE,
                    )
                except nvbit_process.NvbitCaptureProcessError as process_error:
                    mapped = _map_direct_capture_error(process_error)
                    raise mapped from process_error
                except (MemoryError, OSError) as process_error:
                    if _direct_capture_resource_failure(process_error):
                        raise RosterDirectCaptureResourceUnavailable(
                            str(process_error)
                        ) from process_error
                    raise RosterDirectCaptureLaunchFailed(
                        str(process_error)
                    ) from process_error
                elapsed = time.perf_counter() - t0
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            _verify_retained_direct_authority(retained_authority)
            _verify_roster_source_commit_head(source_commit)
            retained_workload_output.verify_path()
            binary_sha256 = _require_hex_identity(
                retained_binary.sha256,
                label="direct binary_sha256",
                length=64,
            )
            tracer_build_sha256 = _require_hex_identity(
                retained_tracer.sha256,
                label="direct nvbit_tracer_build_hash",
                length=64,
            )
            if proc.returncode != 0:
                failure_evidence = _retain_direct_failure_streams(
                    sources=sources,
                    parent=workload_attempt,
                    pinned_parent=retained_workload_output,
                    cell_dir=generation_root.parent,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
                raise RosterDirectCaptureNonzero(
                    returncode=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    failure_evidence=failure_evidence,
                )
            raw_generation = stage.commit(
                require_nonempty=True,
                **_trace_generation_commit_limits(),
            )
            retained_guard.raw_generation = raw_generation
            _verify_committed_trace_generation(raw_generation)
            retained_binary.verify_unchanged()
            retained_tracer.verify_unchanged()
            _verify_retained_direct_authority(retained_authority)
            _verify_roster_source_commit_head(source_commit)
            retained_workload_output.verify_path()
            record = {}

        if proc.returncode == 0:
            raw_generation_record = _raw_generation_record(
                raw_generation,
                cell_dir=generation_root.parent,
                attempt_id=attempt_id,
            )
            _verify_committed_trace_generation(raw_generation)
            try:
                admission = require_trace_admission(raw_generation.path)
            except TraceAdmissionError as exc:
                record = {
                    "slug": slug,
                    "ok": False,
                    "typed_reason": exc.typed_reason,
                    "detail": exc.detail,
                    "trace_capture_attempt_id": attempt_id,
                    "raw_trace_generation": raw_generation_record,
                    "returncode": proc.returncode,
                    "binary_sha256": binary_sha256,
                    "nvbit_tracer_build_hash": tracer_build_sha256,
                }
            else:
                _verify_committed_trace_generation(raw_generation)
                trace_sources = retained_guard.admitted_sources.enter_context(
                    open_admitted_trace_sources(
                        raw_generation.path,
                        expected_admission=admission,
                    )
                )
                trace_sources.verify_unchanged()
                _verify_committed_trace_generation(raw_generation)
                retained_binary.verify_unchanged()
                retained_tracer.verify_unchanged()
                _verify_retained_direct_authority(retained_authority)
                _verify_roster_source_commit_head(source_commit)
                retained_workload_output.verify_path()
                expected = admission.expected_record_count
                admitted = admission.admitted_record_count
                launch_records = _all_admitted_launch_records(admission)
                record = {
                    "slug": slug,
                    "ok": True,
                    "contract_ok": True,
                    "trace_capture_attempt_id": attempt_id,
                    "raw_trace_generation": raw_generation_record,
                    "traces_folder": str(raw_generation.path),
                    "trace_bytes": admission.dynamic_trace_size_bytes,
                    "trace_pb_bytes": admission.dynamic_trace_size_bytes,
                    "capture_seconds": round(elapsed, 2),
                    "records_expected": expected,
                    "records_admitted": admitted,
                    "drop_count": max(expected - admitted, 0),
                    "source_commit": source_commit,
                    "binary_sha256": binary_sha256,
                    "nvbit_tracer_build_hash": tracer_build_sha256,
                    "tracer_source_kind": tracer_authority.source_kind,
                    "trace_contract_id": admission.trace_contract_id,
                    "memory_payload_classes": list(
                        admission.memory_payload_classes
                    ),
                    "trace_admission_sha256": admission.trace_admission_sha256,
                    "trace_filesystem_generation_sha256": (
                        admission.trace_filesystem_generation_sha256
                    ),
                    "dynamic_trace_size_bytes": (
                        admission.dynamic_trace_size_bytes
                    ),
                    "dynamic_trace_sha256": admission.dynamic_trace_sha256,
                    "allocations_sidecar_size_bytes": (
                        admission.allocations_sidecar_size_bytes
                    ),
                    "allocations_sidecar_sha256": (
                        admission.allocations_sidecar_sha256
                    ),
                    "stats_size_bytes": admission.stats_size_bytes,
                    "stats_sha256": admission.stats_sha256,
                    "threadblock_payload_size_bytes": (
                        admission.threadblock_payload_size_bytes
                    ),
                    "threadblock_payload_manifest_sha256": (
                        admission.threadblock_payload_manifest_sha256
                    ),
                    "launch_count": admission.launch_count,
                    "launches": launch_records,
                }
    except RosterDirectCaptureNonzero as capture_error:
        record = {
            "slug": slug,
            "ok": False,
            "typed_reason": capture_error.TYPED_REASON,
            "trace_capture_attempt_id": attempt_id,
            "raw_trace_stage": str(
                stage.staging_path.relative_to(generation_root.parent)
            ),
            "returncode": capture_error.returncode,
            "stderr_tail": _bounded_diagnostic_text(
                capture_error.stderr, max_bytes=300,
            ),
            "process_failure_evidence": list(capture_error.failure_evidence),
            "binary_sha256": binary_sha256,
            "nvbit_tracer_build_hash": tracer_build_sha256,
        }
        retained_guard.bind_failure_record(record)
        if owns_sources:
            retained_guard.__exit__(
                type(capture_error), capture_error, capture_error.__traceback__,
            )
        return record
    except RosterDirectCaptureError as capture_error:
        diagnostic = capture_error.detail
        if capture_error.stderr:
            diagnostic += "; stderr=" + _bounded_diagnostic_text(
                capture_error.stderr,
                max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
            )
        record = {
            "slug": slug,
            "ok": False,
            "typed_reason": capture_error.TYPED_REASON,
            "trace_capture_attempt_id": attempt_id,
            "detail": _bounded_diagnostic_text(
                diagnostic, max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
            ),
        }
        retained_guard.bind_failure_record(record)
        if owns_sources:
            retained_guard.__exit__(
                type(capture_error), capture_error, capture_error.__traceback__,
            )
        return record
    except ImmutableArtifactResourceError as capture_error:
        record = {
            "slug": slug,
            "ok": False,
            "typed_reason": "trace_snapshot_resource_unavailable",
            "trace_capture_attempt_id": attempt_id,
            "detail": _bounded_diagnostic_text(
                capture_error, max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
            ),
        }
        retained_guard.bind_failure_record(record)
        if owns_sources:
            retained_guard.__exit__(
                type(capture_error), capture_error, capture_error.__traceback__,
            )
        return record
    except Exception as capture_error:
        record = {
            "slug": slug,
            "ok": False,
            "typed_reason": "capture_exception",
            "trace_capture_attempt_id": attempt_id,
            "detail": _bounded_diagnostic_text(
                capture_error, max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
            ),
        }
        # Bind the synthesized record before any retained-source close so a
        # close-time capability failure remains visible in attempt evidence.
        retained_guard.bind_failure_record(record)
        if owns_sources:
            retained_guard.__exit__(
                type(capture_error), capture_error, capture_error.__traceback__,
            )
        return record
    except BaseException:
        if owns_sources:
            retained_guard.__exit__(*sys.exc_info())
        raise

    if record.get("ok") is False:
        retained_guard.bind_failure_record(record)
    if owns_sources:
        if record.get("ok"):
            retained_guard.__exit__(None, None, None)
        else:
            failure = RuntimeError("direct trace capture already failed")
            retained_guard.__exit__(type(failure), failure, None)
    return record


_RAW_GENERATION_FIELDS = frozenset({
    "attempt_id",
    "path",
    "content_manifest_sha256",
    "filesystem_generation_sha256",
    "file_count",
    "directory_count",
    "total_file_bytes",
})
_LAUNCH_FIELDS = frozenset({
    "launch_index",
    "device_id",
    "stream_id",
    "kernel_id",
    "kernel_name",
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem_static_bytes",
    "shared_mem_dynamic_bytes",
    "regs_per_thread",
    "inst_count",
})
_SUCCESS_IDENTITY_FIELDS = frozenset({
    "source_commit",
    "binary_sha256",
    "nvbit_tracer_build_hash",
    "tracer_source_kind",
    "trace_contract_id",
    "memory_payload_classes",
    "trace_admission_sha256",
    "trace_filesystem_generation_sha256",
    "dynamic_trace_size_bytes",
    "dynamic_trace_sha256",
    "allocations_sidecar_size_bytes",
    "allocations_sidecar_sha256",
    "stats_size_bytes",
    "stats_sha256",
    "threadblock_payload_size_bytes",
    "threadblock_payload_manifest_sha256",
})
_ACTIVATION_SUCCESS_FIELDS = frozenset({
    "slug",
    "shape_point",
    "ok",
    "contract_ok",
    "trace_capture_attempt_id",
    "raw_trace_generation",
    "traces_folder",
    "trace_bytes",
    "trace_pb_bytes",
    "records_expected",
    "records_admitted",
    "drop_rate",
}) | _SUCCESS_IDENTITY_FIELDS
_DIRECT_SUCCESS_FIELDS = frozenset({
    "slug",
    "shape_point",
    "ok",
    "contract_ok",
    "trace_capture_attempt_id",
    "raw_trace_generation",
    "traces_folder",
    "trace_bytes",
    "trace_pb_bytes",
    "capture_seconds",
    "records_expected",
    "records_admitted",
    "drop_count",
    "launch_count",
    "launches",
    "via",
}) | _SUCCESS_IDENTITY_FIELDS
_GENERIC_FAILURE_FIELDS = frozenset({
    "slug", "shape_point", "ok", "typed_reason", "detail",
})
_DIRECT_EXCEPTION_FAILURE_FIELDS = _GENERIC_FAILURE_FIELDS | frozenset({
    "trace_capture_attempt_id", "via",
})
_DIRECT_PROCESS_FAILURE_FIELDS = frozenset({
    "slug",
    "shape_point",
    "ok",
    "typed_reason",
    "trace_capture_attempt_id",
    "raw_trace_stage",
    "returncode",
    "stderr_tail",
    "process_failure_evidence",
    "binary_sha256",
    "nvbit_tracer_build_hash",
    "via",
})
_DIRECT_ADMISSION_FAILURE_FIELDS = frozenset({
    "slug",
    "shape_point",
    "ok",
    "typed_reason",
    "detail",
    "trace_capture_attempt_id",
    "raw_trace_generation",
    "returncode",
    "binary_sha256",
    "nvbit_tracer_build_hash",
    "via",
})
_FAILURE_FIELD_SETS = (
    _GENERIC_FAILURE_FIELDS,
    _DIRECT_EXCEPTION_FAILURE_FIELDS,
    _DIRECT_PROCESS_FAILURE_FIELDS,
    _DIRECT_ADMISSION_FAILURE_FIELDS,
)
_FAILURE_STREAM_EVIDENCE_FIELDS = frozenset({
    "role", "path", "size_bytes", "sha256",
})
_HASH_FIELDS = frozenset({
    "dynamic_trace_sha256",
    "stats_sha256",
    "threadblock_payload_manifest_sha256",
    "allocations_sidecar_sha256",
    "trace_admission_sha256",
    "trace_filesystem_generation_sha256",
    "binary_sha256",
    "nvbit_tracer_build_hash",
})
_RECORD_INTEGER_FIELDS = frozenset({
    "trace_bytes",
    "trace_pb_bytes",
    "records_expected",
    "records_admitted",
    "drop_count",
    "dynamic_trace_size_bytes",
    "stats_size_bytes",
    "threadblock_payload_size_bytes",
    "allocations_sidecar_size_bytes",
    "launch_count",
})


def _require_exact_field_set(
    value: dict,
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"{label} field set is not exact; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _require_exact_nonnegative_integer(value: object, *, label: str) -> int:
    if (
        type(value) is not int
        or value < 0
        or value > _MAX_ROSTER_JSON_INTEGER
    ):
        raise ValueError(
            f"{label} must be a bounded nonnegative exact integer"
        )
    return value


def _require_exact_signed_integer(value: object, *, label: str) -> int:
    if (
        type(value) is not int
        or value < -_MAX_ROSTER_JSON_INTEGER
        or value > _MAX_ROSTER_JSON_INTEGER
    ):
        raise ValueError(f"{label} must be a bounded exact signed integer")
    return value


def _require_finite_nonnegative_float(
    value: object,
    *,
    label: str,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative exact float")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def _require_bounded_text(value: object, *, label: str, max_bytes: int) -> str:
    if type(value) is not str or "\x00" in value:
        raise ValueError(f"{label} must be an exact bounded string")
    try:
        size_bytes = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc
    if size_bytes > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    return value


def _validate_failure_stream_evidence(
    value: object,
    *,
    attempt_id: str,
    label: str,
) -> list[dict]:
    if type(value) is not list or len(value) != 2:
        raise ValueError(f"{label} must contain exact stdout/stderr identities")
    observed_roles: list[str] = []
    for index, artifact in enumerate(value):
        artifact_label = f"{label} {index}"
        if type(artifact) is not dict:
            raise ValueError(f"{artifact_label} must be an exact object")
        _require_exact_field_set(
            artifact, _FAILURE_STREAM_EVIDENCE_FIELDS, label=artifact_label,
        )
        role = artifact["role"]
        if role not in {"stdout", "stderr"}:
            raise ValueError(f"{artifact_label} role is invalid")
        observed_roles.append(role)
        relative = safe_relative_path(_require_identity_string(
            artifact["path"], label=f"{artifact_label} path", max_bytes=4096,
        ))
        if (
            len(relative.parts) != 3
            or relative.parts[1] != attempt_id
            or relative.parts[2] != f"capture_nonzero.{role}.bin"
        ):
            raise ValueError(f"{artifact_label} path identity is invalid")
        size_bytes = _require_exact_nonnegative_integer(
            artifact["size_bytes"], label=f"{artifact_label} size_bytes",
        )
        if size_bytes > _DIRECT_FAILURE_STREAM_MAX_BYTES:
            raise ValueError(f"{artifact_label} exceeds retained byte bound")
        _require_hex_identity(
            artifact["sha256"], label=f"{artifact_label} sha256", length=64,
        )
    if observed_roles != ["stdout", "stderr"]:
        raise ValueError(f"{label} roles are not canonical")
    return value


def _verify_direct_failure_stream_artifacts(
    record: dict,
    *,
    cell_dir: Path,
) -> None:
    evidence = record.get("process_failure_evidence")
    if evidence is None:
        return
    for artifact in evidence:
        relative = safe_relative_path(artifact["path"])
        try:
            with open_retained_regular(
                cell_dir / relative,
                max_bytes=_DIRECT_FAILURE_STREAM_MAX_BYTES,
            ) as retained:
                if (
                    retained.size_bytes != artifact["size_bytes"]
                    or retained.sha256 != artifact["sha256"]
                ):
                    raise ImmutableArtifactError(
                        "direct failure stream artifact identity changed"
                    )
        except ImmutableArtifactError:
            raise
        except OSError as exc:
            raise ImmutableArtifactError(
                "direct failure stream artifact is unavailable"
            ) from exc


def _validate_raw_generation(value: object, *, label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact object")
    _require_exact_field_set(value, _RAW_GENERATION_FIELDS, label=label)
    _require_identity_string(
        value["attempt_id"], label=f"{label} attempt_id", component=True,
    )
    raw_path = _require_identity_string(
        value["path"], label=f"{label} path", max_bytes=4096,
    )
    safe_relative_path(raw_path)
    for field in ("file_count", "directory_count", "total_file_bytes"):
        _require_exact_nonnegative_integer(
            value[field], label=f"{label} {field}",
        )
    for field in (
        "content_manifest_sha256", "filesystem_generation_sha256",
    ):
        _require_hex_identity(
            value[field], label=f"{label} {field}", length=64,
        )
    return value


def _validate_raw_generation_location(
    raw_generation: dict,
    *,
    trace_capture_attempt_id: str,
    traces_folder: str,
    label: str,
) -> None:
    """Bind the producer-relative generation path to its attempt and locator."""
    raw_path = safe_relative_path(raw_generation["path"])
    if len(raw_path.parts) != 2:
        raise ValueError(
            f"{label} path must be <generation-root>/<attempt-id>"
        )
    if (
        raw_path.parts[-1] != raw_generation["attempt_id"]
        or raw_path.parts[-1] != trace_capture_attempt_id
    ):
        raise ValueError(f"{label} path attempt identity disagrees")

    trace_location = Path(traces_folder)
    if (
        "\\" in traces_folder
        or "\x00" in traces_folder
        or trace_location.as_posix() != traces_folder
        or any(part in {".", ".."} for part in trace_location.parts)
        or len(trace_location.parts) < len(raw_path.parts)
        or tuple(trace_location.parts[-len(raw_path.parts):])
        != raw_path.parts
    ):
        raise ValueError(
            f"{label} path disagrees with traces_folder suffix"
        )


def _validate_launch_record(
    launch: object,
    *,
    expected_index: int,
    label: str,
) -> dict:
    if type(launch) is not dict:
        raise ValueError(f"{label} must be an exact object")
    _require_exact_field_set(launch, _LAUNCH_FIELDS, label=label)
    for field in _LAUNCH_FIELDS - {"kernel_name"}:
        _require_exact_nonnegative_integer(
            launch[field], label=f"{label} {field}",
        )
    if launch["launch_index"] != expected_index:
        raise ValueError(f"{label} launch indexes are not contiguous")
    _require_identity_string(
        launch["kernel_name"],
        label=f"{label} kernel_name",
        max_bytes=_MAX_ROSTER_KERNEL_NAME_BYTES,
    )
    return launch


def _validate_close_diagnostics(value: object, *, label: str) -> list[dict]:
    if type(value) is not list or len(value) > 3:
        raise ValueError(f"{label} is not a bounded exact array")
    for diagnostic_index, diagnostic in enumerate(value):
        diagnostic_label = f"{label} {diagnostic_index}"
        if type(diagnostic) is not dict:
            raise ValueError(f"{diagnostic_label} must be an exact object")
        _require_exact_field_set(
            diagnostic,
            frozenset({"typed_reason", "detail"}),
            label=diagnostic_label,
        )
        _require_identity_string(
            diagnostic["typed_reason"],
            label=f"{diagnostic_label} typed_reason",
            component=True,
        )
        _require_bounded_text(
            diagnostic["detail"],
            label=f"{diagnostic_label} detail",
            max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
        )
    return value


def _validate_capture_record(
    record: object,
    *,
    expected_slug: str,
    expected_shape_point: str,
    index: int,
) -> dict:
    if type(record) is not dict:
        raise ValueError(f"aggregate cell {index} must be an exact object")
    if any(type(key) is not str for key in record):
        raise ValueError(f"aggregate cell {index} keys must be exact strings")
    label = f"aggregate cell {index}"
    base_record = dict(record)
    diagnostics = base_record.pop("close_diagnostics", None)

    if type(base_record.get("ok")) is not bool:
        raise ValueError(f"{label} ok must be boolean")
    if base_record["ok"]:
        expected_fields = (
            _DIRECT_SUCCESS_FIELDS
            if "via" in base_record
            else _ACTIVATION_SUCCESS_FIELDS
        )
        _require_exact_field_set(base_record, expected_fields, label=label)
        if base_record["contract_ok"] is not True:
            raise ValueError(f"{label} contract_ok must be exactly true")
        if base_record["trace_contract_id"] != EXPECTED_TRACE_CONTRACT_ID:
            raise ValueError(f"{label} trace contract is not split-v2")
        if base_record["memory_payload_classes"] != list(
            EXPECTED_MEMORY_PAYLOAD_CLASSES
        ):
            raise ValueError(f"{label} capability vector is not canonical")
        if base_record["tracer_source_kind"] != EXTERNAL_TRACER_SOURCE_KIND:
            raise ValueError(f"{label} tracer source kind is not external")
    else:
        observed_fields = frozenset(base_record)
        if observed_fields not in _FAILURE_FIELD_SETS:
            allowed = [sorted(fields) for fields in _FAILURE_FIELD_SETS]
            raise ValueError(
                f"{label} failure field set is not exact; observed="
                f"{sorted(observed_fields)}, allowed={allowed}"
            )
        if (
            observed_fields == _DIRECT_PROCESS_FAILURE_FIELDS
            and base_record["typed_reason"] != "capture_nonzero"
        ):
            raise ValueError(
                f"{label} direct process failure must be capture_nonzero"
            )

    slug = _require_identity_string(
        base_record["slug"], label=f"{label} slug", component=True,
    )
    shape_point = _require_identity_string(
        base_record["shape_point"],
        label=f"{label} shape_point",
        component=True,
    )
    if slug != expected_slug or shape_point != expected_shape_point:
        raise ValueError(f"{label} identity disagrees with roster row")

    if "trace_capture_attempt_id" in base_record:
        _require_identity_string(
            base_record["trace_capture_attempt_id"],
            label=f"{label} trace_capture_attempt_id",
            component=True,
        )
    if "typed_reason" in base_record:
        _require_identity_string(
            base_record["typed_reason"],
            label=f"{label} typed_reason",
            component=True,
        )
    if "via" in base_record and base_record["via"] != "direct_injection":
        raise ValueError(f"{label} via identity is invalid")
    for field in ("traces_folder", "raw_trace_stage"):
        if field in base_record:
            _require_identity_string(
                base_record[field], label=f"{label} {field}", max_bytes=4096,
            )
    for field in ("detail", "stderr_tail"):
        if field in base_record:
            _require_bounded_text(
                base_record[field], label=f"{label} {field}", max_bytes=4096,
            )

    if "process_failure_evidence" in base_record:
        _validate_failure_stream_evidence(
            base_record["process_failure_evidence"],
            attempt_id=base_record["trace_capture_attempt_id"],
            label=f"{label} process_failure_evidence",
        )
    if "source_commit" in base_record:
        _require_hex_identity(
            base_record["source_commit"],
            label=f"{label} source_commit",
            length=40,
        )
    for field in _HASH_FIELDS & base_record.keys():
        value = base_record[field]
        if value is None and field == "allocations_sidecar_sha256":
            continue
        _require_hex_identity(value, label=f"{label} {field}", length=64)
    for field in _RECORD_INTEGER_FIELDS & base_record.keys():
        value = base_record[field]
        if value is None and field == "allocations_sidecar_size_bytes":
            continue
        _require_exact_nonnegative_integer(value, label=f"{label} {field}")
    if "returncode" in base_record:
        _require_exact_signed_integer(
            base_record["returncode"], label=f"{label} returncode",
        )

    if "allocations_sidecar_size_bytes" in base_record:
        allocation_size = base_record["allocations_sidecar_size_bytes"]
        allocation_hash = base_record["allocations_sidecar_sha256"]
        if (allocation_size is None) != (allocation_hash is None):
            raise ValueError(f"{label} allocations sidecar identity is incomplete")

    if "raw_trace_generation" in base_record:
        raw_generation = _validate_raw_generation(
            base_record["raw_trace_generation"],
            label=f"{label} raw_trace_generation",
        )
    else:
        raw_generation = None

    if "capture_seconds" in base_record:
        _require_finite_nonnegative_float(
            base_record["capture_seconds"], label=f"{label} capture_seconds",
        )
    if "drop_rate" in base_record:
        drop_rate = _require_finite_nonnegative_float(
            base_record["drop_rate"],
            label=f"{label} drop_rate",
            maximum=1.0,
        )
    else:
        drop_rate = None

    if "records_expected" in base_record:
        expected = base_record["records_expected"]
        admitted = base_record["records_admitted"]
        if admitted > expected:
            raise ValueError(f"{label} records_admitted exceeds records_expected")
        dropped = expected - admitted
        if "drop_count" in base_record and base_record["drop_count"] != dropped:
            raise ValueError(f"{label} drop_count relationship is invalid")
        if drop_rate is not None:
            expected_rate = (dropped / expected) if expected else 0.0
            if not math.isclose(
                drop_rate, expected_rate, rel_tol=0.0, abs_tol=1e-15,
            ):
                raise ValueError(f"{label} drop_rate relationship is invalid")

    if base_record["ok"]:
        assert raw_generation is not None
        if (
            raw_generation["attempt_id"]
            != base_record["trace_capture_attempt_id"]
        ):
            raise ValueError(
                f"{label} raw generation attempt identity disagrees"
            )
        _validate_raw_generation_location(
            raw_generation,
            trace_capture_attempt_id=base_record["trace_capture_attempt_id"],
            traces_folder=base_record["traces_folder"],
            label=f"{label} raw generation",
        )
        if raw_generation["file_count"] < 1:
            raise ValueError(f"{label} raw generation file_count must be positive")
        if raw_generation["directory_count"] < 1:
            raise ValueError(
                f"{label} raw generation directory_count must include the root"
            )
        required_raw_bytes = (
            base_record["dynamic_trace_size_bytes"]
            + base_record["stats_size_bytes"]
            + base_record["threadblock_payload_size_bytes"]
            + (base_record["allocations_sidecar_size_bytes"] or 0)
        )
        if raw_generation["total_file_bytes"] < required_raw_bytes:
            raise ValueError(
                f"{label} raw generation total_file_bytes is incomplete"
            )
        trace_bytes = base_record["trace_bytes"]
        if not (
            trace_bytes
            == base_record["trace_pb_bytes"]
            == base_record["dynamic_trace_size_bytes"]
        ):
            raise ValueError(f"{label} trace byte identities disagree")

    launches = base_record.get("launches")
    if launches is not None:
        if type(launches) is not list:
            raise ValueError(f"{label} launches must be an exact array")
        if len(launches) > _MAX_ROSTER_LAUNCHES_PER_CELL:
            _raise_roster_resource_error(
                f"{label} launch vector exceeds "
                f"{_MAX_ROSTER_LAUNCHES_PER_CELL}"
            )
        if base_record["launch_count"] != len(launches):
            raise ValueError(f"{label} launch_count relationship is invalid")
        for launch_index, launch in enumerate(launches):
            validated_launch = _validate_launch_record(
                launch,
                expected_index=launch_index,
                label=f"{label} launch {launch_index}",
            )
            launch_bytes = len(json.dumps(
                validated_launch,
                indent=1,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"))
            if launch_bytes > _MAX_ROSTER_LAUNCH_MANIFEST_BYTES:
                _raise_roster_resource_error(
                    f"{label} launch {launch_index} serialized bytes "
                    f"{launch_bytes} exceed "
                    f"{_MAX_ROSTER_LAUNCH_MANIFEST_BYTES}"
                )

    if diagnostics is not None:
        _validate_close_diagnostics(
            diagnostics, label=f"{label} close_diagnostics",
        )

    fixed_record = dict(record)
    if "launches" in fixed_record:
        fixed_record["launches"] = []
    fixed_bytes = len(json.dumps(
        fixed_record,
        indent=1,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8"))
    if fixed_bytes > _MAX_ROSTER_CELL_MANIFEST_BYTES:
        _raise_roster_resource_error(
            f"{label} fixed serialized bytes {fixed_bytes} exceed "
            f"{_MAX_ROSTER_CELL_MANIFEST_BYTES}"
        )
    return record


def _validate_aggregate_manifest(manifest: object, roster: list[dict]) -> dict:
    if type(manifest) is not dict:
        raise ValueError("aggregate manifest must be an exact object")
    _require_exact_field_set(
        manifest,
        frozenset({
            "generated_by",
            "capture_attempt_id",
            "source_commit",
            "host_clock_state",
            "tracer_source_kind",
            "tracer_source_path",
            "tracer_sha256",
            "trace_contract_id",
            "memory_payload_classes",
            "cells",
        }),
        label="aggregate manifest",
    )
    if manifest["generated_by"] != "roster_trace_capture":
        raise ValueError("aggregate generated_by identity is invalid")
    source_commit_model = manifest["source_commit"]
    source_commit = (
        None
        if source_commit_model is None
        else _require_hex_identity(
            source_commit_model,
            label="aggregate source_commit",
            length=40,
        )
    )
    if manifest["tracer_source_kind"] != EXTERNAL_TRACER_SOURCE_KIND:
        raise ValueError("aggregate tracer source is not external")
    tracer_source_path = _require_identity_string(
        manifest["tracer_source_path"],
        label="aggregate tracer_source_path",
        max_bytes=4096,
    )
    if not Path(tracer_source_path).is_absolute():
        raise ValueError("aggregate tracer_source_path must be absolute")
    _require_hex_identity(
        manifest["tracer_sha256"],
        label="aggregate tracer_sha256",
        length=64,
    )
    if manifest["trace_contract_id"] != EXPECTED_TRACE_CONTRACT_ID:
        raise ValueError("aggregate trace contract is not split-v2")
    if manifest["memory_payload_classes"] != list(
        EXPECTED_MEMORY_PAYLOAD_CLASSES
    ):
        raise ValueError("aggregate capability vector is not canonical")
    _require_identity_string(
        manifest["capture_attempt_id"],
        label="aggregate capture_attempt_id",
        component=True,
    )
    host_clock_state = manifest["host_clock_state"]
    if type(host_clock_state) is not dict or len(host_clock_state) > 2:
        raise ValueError("aggregate host_clock_state must be a bounded exact object")
    for key, value in host_clock_state.items():
        _require_identity_string(
            key, label="aggregate host clock key", component=True,
        )
        _require_bounded_text(
            value, label="aggregate host clock value", max_bytes=4096,
        )
    if type(roster) is not list or not roster:
        raise ValueError("aggregate roster authority must be a nonempty exact array")
    cells = manifest["cells"]
    if type(cells) is not list or len(cells) != len(roster):
        raise ValueError("aggregate cells must exactly match roster length")
    for index, (record, row) in enumerate(zip(cells, roster, strict=True)):
        if type(row) is not dict:
            raise ValueError(f"aggregate roster row {index} must be an exact object")
        validated_record = _validate_capture_record(
            record,
            expected_slug=row["slug"],
            expected_shape_point=row["shape_point"],
            index=index,
        )
        if validated_record.get("ok"):
            if source_commit is None:
                raise ValueError(
                    "successful aggregate cell requires aggregate source_commit"
                )
            if validated_record["source_commit"] != source_commit:
                raise ValueError(
                    f"aggregate cell {index} source_commit disagrees with aggregate"
                )

    envelope = dict(manifest)
    envelope["cells"] = []
    envelope_bytes = len(json.dumps(
        envelope,
        indent=1,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8"))
    if envelope_bytes > _MAX_ROSTER_MANIFEST_ENVELOPE_BYTES:
        _raise_roster_resource_error(
            f"aggregate envelope bytes {envelope_bytes} exceed "
            f"{_MAX_ROSTER_MANIFEST_ENVELOPE_BYTES}"
        )
    return manifest


def _bounded_manifest_payload(manifest: dict, roster: list[dict]) -> bytes:
    _validate_aggregate_manifest(manifest, roster)
    payload = (json.dumps(
        manifest,
        indent=1,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    if len(payload) > _MAX_ROSTER_MANIFEST_BYTES:
        _raise_roster_resource_error(
            f"aggregate manifest bytes {len(payload)} exceed "
            f"{_MAX_ROSTER_MANIFEST_BYTES}"
        )
    return payload


def _run_roster_capture(
    *,
    roster: list[dict],
    plans: list[dict[str, Path | bool]],
    manifest_path: Path,
    sku: str,
    arch_dir: str,
    output_root: Path,
    tracer_authority: ExternalTracerAuthority,
) -> int:
    roster = _validate_roster_entries(roster, require_binary=True)
    plans = _validate_plan_vector(plans, expected_cells=len(roster))
    sku = _require_identity_string(sku, label="roster sku", component=True)
    arch_dir = _require_identity_string(
        arch_dir, label="roster arch_dir", component=True,
    )
    if not isinstance(tracer_authority, ExternalTracerAuthority):
        raise ValueError("roster tracer_authority is invalid")
    _require_roster_descriptor_budget(len(roster))
    _require_roster_manifest_budget(len(roster))
    out = _assert_staging_output(
        manifest_path, operation="write roster capture manifest"
    )
    attempt_id = _require_identity_string(
        _new_capture_attempt_id(),
        label="roster capture_attempt_id",
        component=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    attempt_root = _assert_staging_output(
        out.parent / "capture_manifest_attempts",
        operation="write roster capture manifest attempt",
    )
    active_payload: bytes | None = None
    active_locator: RetainedRegular | None = None
    activation_locators: list[
        tuple[Path, bytes, MetadataSnapshot]
    ] = []

    try:
        with ExitStack() as aggregate_lock_owner:
            aggregate_lock = aggregate_lock_owner.enter_context(
                PinnedDirectoryReader(out.parent)
            )
            retained_git = aggregate_lock_owner.enter_context(
                open_retained_executable(TRUSTED_GIT_EXECUTABLE)
            )
            aggregate_git_token = _ACTIVE_RETAINED_GIT.set(retained_git)
            aggregate_lock_owner.callback(
                _ACTIVE_RETAINED_GIT.reset, aggregate_git_token,
            )
            _lock_activation_cell(aggregate_lock)
            # The prior aggregate cannot remain active while this rerun retires
            # and replaces its per-cell locators. Preserve it under a hidden name
            # before the first capture-side mutation.
            retire_regular_file(out)
            attempt_root.mkdir(exist_ok=True)

            retained_sources = ExitStack()
            retained_sources.__enter__()
            retained_sources_closed = False
            try:
                manifest = {
                    "generated_by": "roster_trace_capture",
                    "capture_attempt_id": attempt_id,
                    "source_commit": None,
                    "host_clock_state": _host_clock_state(),
                    "tracer_source_kind": tracer_authority.source_kind,
                    "tracer_source_path": str(tracer_authority.path),
                    "tracer_sha256": tracer_authority.sha256,
                    "trace_contract_id": EXPECTED_TRACE_CONTRACT_ID,
                    "memory_payload_classes": list(
                        EXPECTED_MEMORY_PAYLOAD_CLASSES
                    ),
                    "cells": [],
                }
                failures = 0
                for index, (cell, plan) in enumerate(
                    zip(roster, plans, strict=True)
                ):
                    slug = cell["slug"]
                    binary = REPO_ROOT / cell["binary"]
                    portable = plan["portable"]
                    print(f"[trace] {slug} ...", flush=True)
                    cell_locators: list[
                        tuple[Path, bytes, MetadataSnapshot]
                    ] = []
                    try:
                        if portable:
                            sidecar = trace_activation_cell(
                                slug=slug,
                                shape_point=cell["shape_point"],
                                arch_dir=arch_dir,
                                sku=sku,
                                extra_app_args=tuple(
                                    item for item in cell.get("app_args", [])
                                    if item != f"--{cell['shape_point']}"
                                ),
                                output_root=output_root,
                                tracer_source_path=tracer_authority.path,
                                tracer_sha256=tracer_authority.sha256,
                                _precleared_plan={
                                    key: value for key, value in plan.items()
                                    if key != "portable"
                                },
                                _retention_stack=retained_sources,
                                _published_locators=cell_locators,
                            )
                            record = _activation_capture_record(
                                slug=slug,
                                cell_dir=plan["cell_dir"],
                                sidecar=sidecar,
                            )
                        else:
                            record = _capture_direct(
                                slug,
                                binary,
                                list(cell.get("app_args", [])),
                                plan["traces_folder"],
                                plan["workload_output_root"],
                                retention_stack=retained_sources,
                                tracer_authority=tracer_authority,
                            )
                            if type(record) is not dict:
                                raise ValueError(
                                    "direct capture returned a non-object record"
                                )
                            record["via"] = "direct_injection"
                        if type(record) is not dict:
                            raise ValueError(
                                "capture returned a non-object aggregate entry"
                            )
                        record.setdefault("shape_point", cell["shape_point"])
                        if (
                            record.get("ok")
                            and record.get("contract_ok") is not True
                        ):
                            record["ok"] = False
                            record["typed_reason"] = "trace_admission_missing"
                        _validate_capture_record(
                            record,
                            expected_slug=slug,
                            expected_shape_point=cell["shape_point"],
                            index=index,
                        )
                        _verify_direct_failure_stream_artifacts(
                            record, cell_dir=plan["cell_dir"],
                        )
                        if record.get("ok"):
                            record_source_commit = record["source_commit"]
                            aggregate_source_commit = manifest["source_commit"]
                            if aggregate_source_commit is None:
                                manifest["source_commit"] = record_source_commit
                            elif aggregate_source_commit != record_source_commit:
                                raise ValueError(
                                    "roster cells resolved mixed source_commit "
                                    "authorities"
                                )
                    except ImmutableArtifactResourceError as exc:
                        record = {
                            "slug": slug,
                            "shape_point": cell["shape_point"],
                            "ok": False,
                            "typed_reason": "trace_snapshot_resource_unavailable",
                            "detail": _bounded_diagnostic_text(
                                exc, max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
                            ),
                        }
                    except Exception as exc:
                        record = {
                            "slug": slug,
                            "shape_point": cell["shape_point"],
                            "ok": False,
                            "typed_reason": "capture_exception",
                            "detail": _bounded_diagnostic_text(
                                exc, max_bytes=_CLOSE_DIAGNOSTIC_BYTES,
                            ),
                        }
                    finally:
                        activation_locators.extend(cell_locators)
                    if not (
                        record.get("ok")
                        and record.get("contract_ok") is True
                    ):
                        failures += 1
                    manifest["cells"].append(record)

                if failures:
                    failure = RuntimeError(
                        "roster capture contains failed or unadmitted cells"
                    )
                    retained_sources.__exit__(type(failure), failure, None)
                    retained_sources_closed = True
                    # Direct failed-capture guards append bounded close diagnostics
                    # to their original record during the close above. Serialize
                    # only after those diagnostics are visible.
                    payload = _bounded_manifest_payload(manifest, roster)
                    attempt_path = publish_regular_file(
                        attempt_root,
                        f"{attempt_id}.json",
                        payload,
                    )
                else:
                    source_commit = manifest["source_commit"]
                    if type(source_commit) is not str:
                        raise RuntimeError(
                            "successful roster aggregate has no source_commit"
                        )
                    _verify_roster_source_commit_head(source_commit)
                    payload = _bounded_manifest_payload(manifest, roster)
                    attempt_path = publish_regular_file(
                        attempt_root,
                        f"{attempt_id}.json",
                        payload,
                    )
                    _verify_roster_source_commit_head(source_commit)
                    active_locator = publish_retained_regular_file(
                        out.parent, out.name, payload,
                    )
                    active_payload = payload
                    active_locator.verify_unchanged()
                    _verify_roster_source_commit_head(source_commit)
                    retained_sources.close()
                    retained_sources_closed = True
                    _verify_roster_source_commit_head(source_commit)
                    # All retained cell closures are now complete. Drain and
                    # close the exact aggregate capability while its publication
                    # lock is still held; a monitor-close mutation is followed by
                    # a descriptor/path/payload reauthentication.
                    active_locator.verify_unchanged()
                    # Transfer the aggregate lock reader into the authenticated
                    # handoff before its final close. If setup fails before the
                    # callback runs, the detached owner closes against the active
                    # exception; once invoked, ExitStack has consumed the reader
                    # callback and no caller-owned reader close remains.
                    transferred_aggregate_lock = aggregate_lock_owner.pop_all()
                    aggregate_lock_close_invoked = False

                    def close_transferred_aggregate_lock() -> None:
                        nonlocal aggregate_lock_close_invoked
                        aggregate_lock_close_invoked = True
                        transferred_aggregate_lock.close()

                    try:
                        close_retained_regular_files_authenticated(
                            (active_locator,),
                            lock_reader=aggregate_lock,
                            closure_callbacks=(
                                lambda: _verify_roster_source_commit_head(
                                    source_commit
                                ),
                                close_transferred_aggregate_lock,
                            ),
                        )
                    finally:
                        if not aggregate_lock_close_invoked:
                            transferred_aggregate_lock.__exit__(*sys.exc_info())
                    active_locator = None
            finally:
                if not retained_sources_closed:
                    retained_sources.__exit__(*sys.exc_info())

        n_ok = sum(
            1 for record in manifest["cells"]
            if record.get("ok") and record.get("contract_ok")
        )
        print(
            f"trace manifest attempt: {attempt_path} — "
            f"{n_ok}/{len(manifest['cells'])} cells "
            f"captured+contract-valid, {failures} failures"
        )
        if failures:
            return 1
        print(f"active trace manifest: {out}")
        return 0
    except BaseException as exc:
        for locator, expected_payload, expected_snapshot in activation_locators:
            try:
                _retire_matching_locator(
                    locator, expected_payload, expected_snapshot,
                )
            except BaseException as retirement_error:
                try:
                    exc.add_note(
                        f"failed roster cell locator retirement for {locator}: "
                        f"{retirement_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        if active_locator is not None and active_payload is not None:
            try:
                preserve_retained_regular_file(
                    active_locator, expected_payload=active_payload,
                )
            except BaseException as preservation_error:
                try:
                    exc.add_note(
                        f"failed roster manifest preservation for {out}: "
                        f"{preservation_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            try:
                active_locator.close()
            except BaseException as close_error:
                try:
                    exc.add_note(
                        f"failed roster manifest capability close for {out}: "
                        f"{close_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        if not isinstance(exc, Exception):
            raise
        print(
            f"roster_trace_capture: publication failed: {exc}",
            file=sys.stderr,
        )
        return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", required=True)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--sku", default="rtx_3090_ti")
    ap.add_argument("--arch-dir", default="sm_86")
    ap.add_argument("--nvbit-tracer", type=Path, default=None)
    ap.add_argument("--nvbit-tracer-sha256", default=None)
    ap.add_argument(
        "--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT,
        help="Active output root below results/nvbit_capture_staging.",
    )
    args = ap.parse_args(argv)

    try:
        tracer_authority = require_external_tracer_authority(
            args.nvbit_tracer,
            args.nvbit_tracer_sha256,
            repo_root=REPO_ROOT,
        )
        roster_snapshot = read_regular_snapshot(
            Path(args.roster), max_bytes=_MAX_ROSTER_INPUT_BYTES,
        )
        roster = strict_json_loads(
            roster_snapshot.payload, label=f"roster {args.roster}",
        )
        plans, out = _roster_capture_plan(
            roster=roster,
            sku=args.sku,
            arch_dir=args.arch_dir,
            output_root=args.output_root,
            manifest_out=args.manifest_out,
        )
    except (ImmutableArtifactError, KeyError, OSError, ValueError) as exc:
        print(f"roster_trace_capture: output refusal: {exc}", file=sys.stderr)
        return 2

    # Tracer dependency preflight is intentionally per-cell and descriptor-bound
    # inside the delegated/direct capture paths; never run ldd against the
    # mutable tracer pathname here.
    try:
        return _run_roster_capture(
            roster=roster,
            plans=plans,
            manifest_path=out,
            sku=args.sku,
            arch_dir=args.arch_dir,
            output_root=args.output_root,
            tracer_authority=tracer_authority,
        )
    except (ImmutableArtifactError, KeyError, OSError, ValueError) as exc:
        print(f"roster_trace_capture: resource refusal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
