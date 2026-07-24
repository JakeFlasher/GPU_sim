#!/usr/bin/env python3
"""Capture preflight/provenance gate for the sm_86 RTX 3090 Ti live corpus.

Refuses to start a capture session unless the host is in the pinned reference
state: a clean, commit-identified working tree; the pinned toolchain (NCU /
driver / nvcc / NVBit); the v2.1 reference clocks locked (SM 1350, mem 10501,
persistence on); a WORKING NVBit injection runtime under ``.toolchain/pb/lib``,
proven by a real smoke that injects the tracer into a deterministic workload and
yields a nonzero contract-valid trace (the failure mode being a SILENT ``dlopen``
miss that runs the app untraced, which ldd alone cannot catch); and free disk in
excess of the declared trace-retention budget. Emits a provenance manifest
recording exactly what it saw, and gates on it.

The gate is fail-closed: a missing pins file, a missing budget file, a dirty
tree, an unlocked clock, or an unresolved injection runtime all block capture
BEFORE any kernel runs. NCU metric values are never inputs here — only
environment/toolchain identities (Rule 1 governs metric values).

Usage:
  .venv/bin/python tools/local_labels/campaign_preflight.py \
      --output /path/to/preflight_manifest.json
Exit: 0 capture admitted; 1 one or more blockers; 2 semantic emission failed;
3 typed storage stop (the smoke tree is preserved and stderr carries a stable
``CAMPAIGN_PREFLIGHT_STORAGE_FAILURE:`` JSON marker).
"""
from __future__ import annotations

import argparse
import ast
import csv
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # This script is invoked by absolute/relative path and deliberately runs the
    # smoke workload from the binary directory with all outputs under /tmp.
    # Python otherwise exposes only tools/local_labels as the script import
    # root, so the post-smoke payload consumer cannot import
    # tools.trace_sampling.  Bind imports to this checkout explicitly instead
    # of depending on the caller's cwd or PYTHONPATH.
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_PINS = REPO_ROOT / "tools" / "local_labels" / "campaign_protocol_pins.json"
DEFAULT_BUDGET = (
    REPO_ROOT / "data" / "local_labels" / "campaign" / "trace_retention_budget.json"
)
DEFAULT_MATRIX = (
    REPO_ROOT / "data" / "local_labels" / "campaign"
    / "sm86_reference_capture_matrix.json"
)

from tools.local_labels import sol_capture_environment as sol_environment  # noqa: E402
from tools.local_labels import build_sol_representative_execution_matrix as representative_execution  # noqa: E402
from tools.local_labels import capture_toolchain_contract as toolchain_contract  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import ncu_action_executor as ncu_executor  # noqa: E402
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_external_output,
    assert_write_allowed,
)

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

NCU_EXECUTION_MODE = "direct_inner_elf_v1"
NCU_HOST_MACHINE = "x86_64"
NCU_LAUNCHER_TARGET_RELPATH = "target/linux-desktop-glibc_2_11_3-x64/ncu"

PREFLIGHT_STORAGE_MARKER = "CAMPAIGN_PREFLIGHT_STORAGE_FAILURE:"
PREFLIGHT_STORAGE_CAPACITY = "capacity"
PREFLIGHT_STORAGE_INTEGRITY = "integrity"
_STORAGE_CAPACITY_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
_STORAGE_INTEGRITY_ERRNOS = frozenset({
    errno.EIO,
    errno.EROFS,
    *(
        value for value in (
            getattr(errno, "ENXIO", None),
            getattr(errno, "ESTALE", None),
        ) if value is not None
    ),
})


class PreflightStorageFailure(RuntimeError):
    """Stable child-process storage stop; the smoke tree remains evidence."""

    def __init__(
        self, kind: str, *, error_number: int | None, operation: str,
        detail: str,
    ):
        if kind not in {
            PREFLIGHT_STORAGE_CAPACITY, PREFLIGHT_STORAGE_INTEGRITY,
        }:
            raise ValueError(f"unknown preflight storage kind: {kind!r}")
        self.kind = kind
        self.error_number = error_number
        self.operation = operation
        self.detail = detail
        super().__init__(f"{kind}: {operation}: {detail}")

    def document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "errno": self.error_number,
            "operation": self.operation,
            "detail_sha256": hashlib.sha256(
                self.detail.encode("utf-8", errors="surrogateescape")
            ).hexdigest(),
            "smoke_tree_preserved": True,
        }

    def marker(self) -> str:
        return PREFLIGHT_STORAGE_MARKER + json.dumps(
            self.document(), sort_keys=True, separators=(",", ":"),
        )


def preflight_storage_failure_from_exception(
    exc: BaseException, *, operation: str,
) -> PreflightStorageFailure | None:
    """Recognize only the closed storage errnos through nested causes."""
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, PreflightStorageFailure):
            return current
        if isinstance(current, OSError):
            if current.errno in _STORAGE_CAPACITY_ERRNOS:
                kind = PREFLIGHT_STORAGE_CAPACITY
            elif current.errno in _STORAGE_INTEGRITY_ERRNOS:
                kind = PREFLIGHT_STORAGE_INTEGRITY
            else:
                return None
            return PreflightStorageFailure(
                kind, error_number=current.errno, operation=operation,
                detail=str(current),
            )
        current = current.__cause__ or current.__context__
    return None


def _probe_failed_smoke_output_storage(traces: Path) -> None:
    """Prove a child failure belongs to the explicit smoke-output domain.

    Free-form child diagnostics can describe executable, library, or workload
    input failures, so their text is never storage authority.  This local
    create/write/fsync/unlink probe is the only post-failure ownership check.
    """
    probe = traces / f".preflight-storage-probe-{os.getpid()}"
    descriptor: int | None = None
    failure: tuple[OSError, PreflightStorageFailure] | None = None

    def record(exc: OSError) -> None:
        nonlocal failure
        mapped = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_failure_output_probe",
        )
        if mapped is None:
            raise exc
        if failure is None:
            failure = (exc, mapped)

    try:
        descriptor = os.open(
            probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
        )
    except OSError as exc:
        record(exc)
    if failure is None:
        try:
            payload = b"campaign-output-storage-probe\n"
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(errno.EIO, "short smoke output storage probe write")
        except OSError as exc:
            record(exc)
    if failure is None:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            record(exc)
    if failure is None:
        try:
            os.close(descriptor)
            descriptor = None
        except OSError as exc:
            record(exc)
    if failure is None:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            record(exc)
    if failure is not None:
        # First closed output fault wins.  Do not close/unlink/mutate after it.
        origin, mapped = failure
        raise mapped from origin


def _smoke_output_lstat(path: Path, *, operation: str) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation=operation,
        )
        if storage_failure is not None:
            raise storage_failure from exc
        raise


def _smoke_output_mkdir(
    path: Path, *, parents: bool, exist_ok: bool, operation: str,
) -> None:
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation=operation,
        )
        if storage_failure is not None:
            raise storage_failure from exc
        raise


def _sha256_smoke_output(path: Path, *, operation: str) -> str:
    try:
        return _sha256_file(path)
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation=operation,
        )
        if storage_failure is not None:
            raise storage_failure from exc
        raise


def _cleanup_smoke_output(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_cleanup",
        )
        if storage_failure is not None:
            raise storage_failure from exc
        raise


def parse_preflight_storage_marker(
    stderr: str,
) -> dict[str, object] | None:
    """Parse exactly one well-formed stable marker from child stderr."""
    matches = [
        line.removeprefix(PREFLIGHT_STORAGE_MARKER)
        for line in stderr.splitlines()
        if line.startswith(PREFLIGHT_STORAGE_MARKER)
    ]
    if len(matches) != 1:
        return None
    try:
        document = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(document, dict)
        or set(document) != {
            "schema_version", "kind", "errno", "operation",
            "detail_sha256", "smoke_tree_preserved",
        }
        or document.get("schema_version") != 1
        or document.get("kind") not in {
            PREFLIGHT_STORAGE_CAPACITY, PREFLIGHT_STORAGE_INTEGRITY,
        }
        or (
            document.get("errno") is not None
            and type(document.get("errno")) is not int
        )
        or not isinstance(document.get("operation"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(document.get("detail_sha256", ""))
        ) is None
        or document.get("smoke_tree_preserved") is not True
    ):
        return None
    return document

PREFLIGHT_FRESHNESS_SECONDS = 900.0
PREFLIGHT_FUTURE_SKEW_SECONDS = 2.0

# Probes whose non-ok status blocks capture. The covariate sample is recorded
# but never blocks (an Ampere boost-equilibrium thermal excursion is telemetry,
# not a fault — same discipline as the pchase covariate verdict).
_BLOCKING_CHECKS = (
    "worktree",
    "ncu",
    "driver",
    "nvcc",
    "nvbit",
    "capture_processes",
    "clock_state",
    "injection_runtime",
    "trace_budget",
    "sol_environment",
)

_EXPECTED_TRACE_CONTRACT_ID = "sm86_addr_carry_v1"
_EXPECTED_MEMORY_PAYLOAD_CLASSES = frozenset({1, 2, 3, 6, 8})
_CONFLICTING_PROCESS_SCRIPTS = frozenset({
    "campaign_preflight.py",
    "eval_driver.py",
    "run_dataset.py",
    "sm86_capture_chunk.py",
    "sol_reconstitution.py",
    "sol_shrink_apply.py",
    "sol_workload_capture_campaign.py",
    "sol_workload_capture_chunk.py",
})
_PYTHON_EXECUTABLE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|pypy\d*)$")
_PYTEST_EXECUTABLES = frozenset({"pytest", "py.test"})
_STAGING_ROOT = REPO_ROOT / "results" / "nvbit_capture_staging"
_DEFAULT_OUTPUT = _STAGING_ROOT / "campaign_preflight" / "preflight_manifest.json"


def _assert_staging_output(path: Path | str, *, operation: str) -> Path:
    candidate = assert_write_allowed(
        path, repo_root=REPO_ROOT, operation=operation,
    )
    staging = _STAGING_ROOT.resolve(strict=False)
    try:
        candidate.relative_to(staging)
    except ValueError as exc:
        raise ProtectedPathError(
            f"refusing to {operation} outside {staging}: {candidate}"
        ) from exc
    return candidate


def _injection_temp_plan(
    smoke_temp_root: Path | str | None = None,
) -> dict[str, Path]:
    """Choose and clear external smoke outputs without creating them."""

    candidate = (
        Path(smoke_temp_root)
        if smoke_temp_root is not None
        else Path(tempfile.gettempdir())
        / f"campaign_preflight_smoke_{os.getpid()}_{time.time_ns()}"
    )
    root = assert_external_output(
        candidate, repo_root=REPO_ROOT, operation="prepare injection smoke root",
    )
    return {
        "root": root,
        "traces": assert_external_output(
            root / "smoke_out",
            repo_root=REPO_ROOT,
            operation="prepare injection smoke trace",
        ),
        "workload_output": assert_external_output(
            root / "workload_output",
            repo_root=REPO_ROOT,
            operation="prepare injection workload output",
        ),
    }


def _campaign_output_plan(
    *, output: Path | str = _DEFAULT_OUTPUT,
    smoke_temp_root: Path | str | None = None,
) -> tuple[Path, dict[str, Path]]:
    output_path = _assert_staging_output(
        output, operation="write campaign preflight manifest",
    )
    return output_path, _injection_temp_plan(smoke_temp_root)


# --------------------------------------------------------------------------- #
# Pure helpers (no subprocess / no GPU) — the unit-testable core.
# --------------------------------------------------------------------------- #
def version_matches(observed: str | None, expected: str, *, mode: str) -> bool:
    """``mode='exact'`` requires equality; ``mode='prefix'`` requires observed
    to start with expected (e.g. NCU '2026.1.0.0' satisfies pin '2026.1.0')."""
    if observed is None:
        return False
    observed = observed.strip()
    if mode == "exact":
        return observed == expected
    if mode == "prefix":
        return observed.startswith(expected)
    raise ValueError(f"unknown match mode {mode!r}")


def remaining_dirty_lines(
    porcelain_lines: list[str], ignore_relpaths: list[str]
) -> list[str]:
    """Filter ``git status --porcelain -uall`` lines, dropping any whose path is
    under an ignore prefix (typically the about-to-be-written manifest path).
    A porcelain line is ``XY <path>`` (or ``XY <old> -> <new>`` for renames)."""
    kept: list[str] = []
    norm_ignores = [p.strip("/") for p in ignore_relpaths if p]
    for line in porcelain_lines:
        if not line.strip():
            continue
        path = line[3:] if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if any(path == ig or path.startswith(ig + "/") for ig in norm_ignores):
            continue
        kept.append(line)
    return kept


def disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def evaluate_budget(budget: dict, free_gib: float) -> tuple[str, str]:
    ceiling = budget.get("retention_budget_gib")
    metadata = budget.get("metadata_margin_gib")
    staging = budget.get("raw_staging_margin_gib")
    fields = {
        "retention_budget_gib": ceiling,
        "metadata_margin_gib": metadata,
        "raw_staging_margin_gib": staging,
    }
    invalid = [
        name for name, value in fields.items()
        if isinstance(value, bool) or not isinstance(value, (int, float))
        or value <= 0
    ]
    if invalid:
        return (
            STATUS_FAILED,
            "trace_retention_budget missing/non-positive fields: "
            + ", ".join(invalid),
        )
    if metadata >= ceiling:
        return (
            STATUS_FAILED,
            f"metadata margin {metadata} GiB must be smaller than retention "
            f"ceiling {ceiling} GiB",
        )
    required = ceiling + staging
    if free_gib > required:
        return (
            STATUS_OK,
            f"free {free_gib:.1f} GiB > retention {ceiling} GiB + raw "
            f"staging {staging} GiB = {required} GiB",
        )
    return (
        STATUS_MISMATCH,
        f"free {free_gib:.1f} GiB <= retention+staging requirement "
        f"{required} GiB (insufficient headroom)",
    )


def injection_smoke_verdict(
    *,
    ldd_missing: list[str],
    returncode: int,
    pb_present: bool,
    stats_present: bool,
    expected: int | None,
    admitted: int | None,
    validator_ok: bool | None,
    validator_reason: str | None,
    threadblocks_pb_count: int | None,
    root_trace_ok: bool | None,
    root_trace_reason: str | None,
    root_trace_launch_count: int | None,
    banner_seen: bool,
    full_coverage_ok: bool | None,
    full_coverage_reason: str | None,
) -> tuple[str, str]:
    """Adjudicate a captured injection smoke down to ``(status, detail)``.

    Fail-closed and ordered from cheapest signal to strongest: an unresolved
    ``.so`` dependency (the SILENT ``CUDA_INJECTION64_PATH`` dlopen miss that runs
    the app untraced), then a nonzero workload exit, a missing trace payload,
    unparseable/zero admitted records, a trace the tracer's own contract validator
    rejects, and finally a validator that passed only *vacuously* — ``ldd``
    resolving is necessary but NOT sufficient, and neither is a green
    ``validate_trace_dir``: that walker returns ``ok=True`` on a present-but-EMPTY
    ``threadblocks/`` (its per-record loop simply never runs), so the admitted
    count from ``stats.csv`` and the validator verdict are decoupled from the
    per-threadblock records that ARE the simulator input. We therefore also
    require ``threadblocks_pb_count >= 1`` — at least one contract-validated
    threadblock record actually walked — before admitting. Pure (no I/O), so the
    block/pass boundary — 'ldd ok but no trace', 'trace present but
    contract-invalid', and 'validator vacuously green on an empty tree' — is unit
    tested without a GPU. ``banner_seen`` is diagnostic only (a genuinely
    contract-valid nonzero trace proves injection worked even if the startup
    banner was buffered away), never a gate.
    """
    if ldd_missing:
        return STATUS_FAILED, (
            "tracer .so dependencies unresolved — CUDA_INJECTION64_PATH would "
            f"dlopen-fail SILENTLY and run untraced: {ldd_missing}")
    if returncode != 0:
        return STATUS_FAILED, f"smoke workload exited {returncode} (no trusted trace)"
    if not pb_present:
        return STATUS_FAILED, (
            "injection produced no dynamic_trace.pb (silent untraced run)")
    if not root_trace_ok:
        return STATUS_FAILED, (
            "dynamic_trace.pb fails the root launch contract: "
            f"{root_trace_reason or 'unknown'}")
    if not root_trace_launch_count or root_trace_launch_count < 1:
        return STATUS_FAILED, (
            "dynamic_trace.pb parsed only as an empty/default Trace — zero "
            "launches were bound to grid/block geometry")
    if not stats_present:
        return STATUS_FAILED, "no stats.csv — cannot derive admission counters"
    if admitted is None or expected is None:
        return STATUS_FAILED, "stats.csv unparseable — no admission counters"
    if admitted <= 0:
        return STATUS_FAILED, (
            f"zero admitted records (expected={expected}, admitted={admitted}) — "
            "trace carries no instruction payload")
    if not validator_ok:
        return STATUS_FAILED, (
            f"trace fails the contract validator: {validator_reason or 'unknown'}")
    if not threadblocks_pb_count or threadblocks_pb_count < 1:
        return STATUS_FAILED, (
            "validate_trace_dir passed only vacuously — present-but-empty "
            "threadblocks/ walked zero contract-validated records (no instruction "
            "payload despite nonzero stats.csv)")
    if not full_coverage_ok:
        return STATUS_FAILED, (
            "full-trace coverage contract failed: "
            f"{full_coverage_reason or 'unknown'}")
    detail = (f"contract-valid smoke: admitted={admitted}/{expected} insts, "
              f"{threadblocks_pb_count} threadblock record(s) validated")
    if not banner_seen:
        detail += " (note: NVBit banner not captured on stderr)"
    return STATUS_OK, detail


def full_trace_coverage_verdict(
    *, root_launch_count: int, grid_threadblocks: int,
    decoded_threadblocks: int, threadblock_pb_count: int,
    decoded_records: int, expected_records: int | None,
    admitted_records: int | None, stats_row_count: int,
    sampling_sidecar_present: bool, zero_record_threadblocks: int,
    expected_launch_count: int = 1,
) -> tuple[str, str]:
    """Require an exact, unsampled trace with the declared launch count.

    The stats CSV and protobuf tree are independent producer surfaces.  A full
    trace is admissible only when the decoded consumer payload itself covers
    every CTA exactly once and its decoded instruction count equals the zero-
    drop producer counters.  Sampling is a separate sidecar-authorized path.
    """
    integer_fields = {
        "root_launch_count": root_launch_count,
        "grid_threadblocks": grid_threadblocks,
        "decoded_threadblocks": decoded_threadblocks,
        "threadblock_pb_count": threadblock_pb_count,
        "decoded_records": decoded_records,
        "stats_row_count": stats_row_count,
        "zero_record_threadblocks": zero_record_threadblocks,
    }
    invalid = {
        name: value for name, value in integer_fields.items()
        if type(value) is not int
    }
    if invalid:
        return STATUS_FAILED, f"coverage fields are not exact integers: {invalid!r}"
    if type(sampling_sidecar_present) is not bool:
        return STATUS_FAILED, (
            "sampling_sidecar_present is not an exact boolean: "
            f"{sampling_sidecar_present!r}")
    if sampling_sidecar_present:
        return STATUS_FAILED, "unsampled full trace unexpectedly carries a sampling sidecar"
    if type(expected_launch_count) is not int or expected_launch_count <= 0:
        return STATUS_FAILED, (
            f"expected_launch_count={expected_launch_count!r}, expected positive integer"
        )
    if root_launch_count != expected_launch_count:
        return STATUS_FAILED, (
            f"root launch_count={root_launch_count}, expected exactly "
            f"{expected_launch_count}"
        )
    if stats_row_count != expected_launch_count:
        return STATUS_FAILED, (
            f"stats_row_count={stats_row_count}, expected exactly "
            f"{expected_launch_count}"
        )
    if grid_threadblocks <= 0:
        return STATUS_FAILED, f"grid contains {grid_threadblocks} threadblocks"
    if decoded_threadblocks != grid_threadblocks:
        return STATUS_FAILED, (
            f"decoded_threadblocks={decoded_threadblocks} != grid={grid_threadblocks}")
    if threadblock_pb_count != decoded_threadblocks:
        return STATUS_FAILED, (
            f"threadblock_pb_count={threadblock_pb_count} != "
            f"decoded_threadblocks={decoded_threadblocks}")
    if zero_record_threadblocks != 0:
        return STATUS_FAILED, (
            f"zero_record_threadblocks={zero_record_threadblocks}, expected 0")
    if decoded_records <= 0:
        return STATUS_FAILED, f"decoded_records={decoded_records}, expected positive"
    if (type(expected_records) is not int or type(admitted_records) is not int
            or expected_records <= 0):
        return STATUS_FAILED, (
            f"invalid producer counters expected={expected_records!r} "
            f"admitted={admitted_records!r}")
    if admitted_records != expected_records:
        return STATUS_FAILED, (
            f"producer drop detected: admitted={admitted_records} != "
            f"expected={expected_records}")
    return STATUS_OK, (
        f"exact {expected_launch_count}-launch full grid: "
        f"{decoded_threadblocks}/{grid_threadblocks} CTAs, "
        f"{decoded_records} decoded warp-instruction records; producer counters "
        f"zero-drop at {admitted_records}/{expected_records} thread-instructions")


def _decode_all_launch_coverage(traces_dir: Path, root_trace: dict) -> tuple[int, int, int, int]:
    """Decode every declared launch and require exact per-launch CTA coverage."""
    pb_generated = REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
    if str(pb_generated) not in sys.path:
        sys.path.insert(0, str(pb_generated))
    import threadblock_pb2  # type: ignore[import-not-found]

    launches = root_trace.get("launches")
    if not isinstance(launches, list) or not launches:
        raise ValueError("root trace has no launch sequence")
    expected_dirs: set[Path] = set()
    grid_blocks = decoded_blocks = decoded_records = zero_record_blocks = 0
    for launch in launches:
        grid = launch.get("grid_dim")
        if not isinstance(grid, list) or len(grid) != 3:
            raise ValueError(f"launch grid is malformed: {launch!r}")
        gx, gy, gz = grid
        launch_grid_blocks = gx * gy * gz
        grid_blocks += launch_grid_blocks
        launch_dir = (
            traces_dir / "threadblocks"
            / f"device_{launch['device_id']}" / f"stream_{launch['stream_id']}"
            / f"kernel_{launch['kernel_id']}"
        )
        expected_dirs.add(launch_dir.resolve())
        if launch_dir.is_symlink() or not launch_dir.is_dir():
            raise ValueError(f"launch payload directory missing: {launch_dir}")
        seen: set[tuple[int, int, int]] = set()
        payloads = sorted(launch_dir.glob("*.pb"))
        if len(payloads) != launch_grid_blocks:
            raise ValueError(
                f"{launch_dir}: payload count={len(payloads)} != grid={launch_grid_blocks}"
            )
        for path in payloads:
            block = threadblock_pb2.threadblock()
            block.ParseFromString(path.read_bytes())
            if not block.HasField("block_id"):
                raise ValueError(f"{path}: threadblock missing block_id")
            coords = (
                int(block.block_id.x), int(block.block_id.y), int(block.block_id.z)
            )
            if not (
                0 <= coords[0] < gx and 0 <= coords[1] < gy
                and 0 <= coords[2] < gz
            ):
                raise ValueError(f"{path}: CTA {coords!r} outside grid {grid!r}")
            if coords in seen:
                raise ValueError(f"{path}: duplicate CTA {coords!r}")
            seen.add(coords)
            records = sum(
                len(block.warps[warp_id].instructions) for warp_id in block.warps
            )
            decoded_records += records
            zero_record_blocks += records == 0
        decoded_blocks += len(seen)
    actual_dirs = {
        path.resolve()
        for path in (traces_dir / "threadblocks").glob(
            "device_*/stream_*/kernel_*"
        )
        if path.is_dir()
    }
    if actual_dirs != expected_dirs:
        raise ValueError(
            "threadblock launch directory set mismatch: "
            f"missing={sorted(map(str, expected_dirs - actual_dirs))!r} "
            f"extra={sorted(map(str, actual_dirs - expected_dirs))!r}"
        )
    return grid_blocks, decoded_blocks, decoded_records, zero_record_blocks


def _analyze_full_trace_coverage(
    traces_dir: Path, root_trace: dict, expected: int | None,
    admitted: int | None, stats_row_count: int,
    *, expected_launch_count: int = 1,
) -> dict:
    """Decode the actual simulator payload and adjudicate exact coverage."""
    if expected_launch_count == 1:
        # Repository-code import is immutable input, not smoke-output I/O.
        from tools.trace_sampling.counting_pass import analyze_trace_dir
    try:
        if expected_launch_count == 1:
            stats = analyze_trace_dir(traces_dir, "preflight_smoke")
            decoded_blocks = stats.n_blocks
            decoded_records = stats.total_records
            zero_record_blocks = sum(block.records <= 0 for block in stats.blocks)
            grid_blocks = stats.grid[0] * stats.grid[1] * stats.grid[2]
        else:
            (
                grid_blocks, decoded_blocks, decoded_records, zero_record_blocks,
            ) = _decode_all_launch_coverage(traces_dir, root_trace)
        decode_error = None
    except PreflightStorageFailure:
        raise
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_payload_read",
        )
        if storage_failure is not None:
            raise storage_failure from exc
        decoded_blocks = decoded_records = grid_blocks = zero_record_blocks = 0
        decode_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - ordinary malformed payload
        decoded_blocks = decoded_records = grid_blocks = zero_record_blocks = 0
        decode_error = str(exc)
    pb_root = traces_dir / "threadblocks"
    pb_root_stat = _smoke_output_lstat(
        pb_root, operation="injection_smoke_payload_tree_stat",
    )
    try:
        pb_count = (
            len(list(pb_root.rglob("*.pb")))
            if pb_root_stat is not None and stat.S_ISDIR(pb_root_stat.st_mode)
            else 0
        )
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_payload_tree_walk",
        )
        if storage_failure is not None:
            raise storage_failure from exc
        raise
    sampling_present = any(
        _smoke_output_lstat(path, operation="injection_smoke_sidecar_stat")
        is not None
        for path in (
            traces_dir / "trace_sampling.json",
            traces_dir / "extra_info" / "trace_sampling_capture.json",
        )
    )
    if decode_error:
        status, detail = STATUS_FAILED, f"payload decode failed: {decode_error}"
    else:
        status, detail = full_trace_coverage_verdict(
            root_launch_count=root_trace.get("launch_count", 0),
            grid_threadblocks=grid_blocks,
            decoded_threadblocks=decoded_blocks,
            threadblock_pb_count=pb_count,
            decoded_records=decoded_records,
            expected_records=expected, admitted_records=admitted,
            stats_row_count=stats_row_count,
            sampling_sidecar_present=sampling_present,
            zero_record_threadblocks=zero_record_blocks,
            expected_launch_count=expected_launch_count,
        )
    return {
        "status": status, "detail": detail,
        "grid_threadblock_count": grid_blocks,
        "decoded_threadblock_count": decoded_blocks,
        "threadblocks_pb_count": pb_count,
        "decoded_record_count": decoded_records,
        "zero_record_threadblock_count": zero_record_blocks,
        "stats_row_count": stats_row_count,
        "sampling_sidecar_present": sampling_present,
    }


def root_trace_verdict(
    *,
    pb_bytes: int,
    parse_error: str | None,
    trace_contract_id: str | None,
    memory_payload_classes: list[int] | None,
    launch_geometries: list[dict] | None,
) -> tuple[str, str]:
    """Adjudicate the root ``dynamic_trace.pb`` without trusting file presence.

    A zero-byte protobuf and a protobuf default message both parse too easily;
    admission therefore requires a nonempty payload, the exact address-carrying
    contract/capabilities, at least one launch, and positive grid/block geometry
    on every launch.
    """
    if pb_bytes <= 0:
        return STATUS_FAILED, "dynamic_trace.pb is empty"
    if parse_error:
        return STATUS_FAILED, f"Trace protobuf parse failed: {parse_error}"
    if trace_contract_id != _EXPECTED_TRACE_CONTRACT_ID:
        return STATUS_FAILED, (
            f"trace_contract_id={trace_contract_id!r}, expected "
            f"{_EXPECTED_TRACE_CONTRACT_ID!r}")
    classes = set(memory_payload_classes or [])
    missing = sorted(_EXPECTED_MEMORY_PAYLOAD_CLASSES - classes)
    if missing:
        return STATUS_FAILED, f"memory_payload_classes missing {missing}"
    launches = launch_geometries or []
    if not launches:
        return STATUS_FAILED, "Trace contains zero kernel launches"
    for i, launch in enumerate(launches):
        grid = launch.get("grid_dim") or []
        block = launch.get("block_dim") or []
        if (len(grid) != 3 or len(block) != 3
                or any(type(v) is not int or v <= 0 for v in [*grid, *block])):
            return STATUS_FAILED, (
                f"launch[{i}] has non-positive/invalid geometry: "
                f"grid={grid!r} block={block!r}")
    return STATUS_OK, f"root Trace binds {len(launches)} launch(es)"


def _validate_root_trace(pb_path: Path, repo_root: Path) -> dict:
    """Parse and validate the top-level Trace protobuf emitted by the smoke."""
    try:
        raw = pb_path.read_bytes()
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_root_read",
        )
        if storage_failure is not None:
            raise storage_failure from exc
        raw = b""
        read_error = str(exc)
    else:
        read_error = None

    contract_id: str | None = None
    classes: list[int] | None = None
    launches: list[dict] | None = None
    parse_error = read_error
    if parse_error is None and raw:
        pb_generated = repo_root / "workloads" / "simbench_bw" / "_pb_generated"
        try:
            if str(pb_generated) not in sys.path:
                sys.path.insert(0, str(pb_generated))
            import trace_pb2  # type: ignore[import-not-found]

            trace = trace_pb2.Trace()
            trace.ParseFromString(raw)
            contract_id = trace.trace_contract_id
            classes = [int(v) for v in trace.memory_payload_classes]
            launches = []
            for device in trace.gpu_device.values():
                for stream in device.streams.values():
                    for kernel in stream.kernels:
                        launches.append({
                            "device_id": int(device.id),
                            "stream_id": int(stream.id),
                            "kernel_id": int(kernel.id),
                            "kernel_name": kernel.name,
                            "grid_dim": [
                                int(kernel.grid_dim.x), int(kernel.grid_dim.y),
                                int(kernel.grid_dim.z),
                            ],
                            "block_dim": [
                                int(kernel.block_dim.x), int(kernel.block_dim.y),
                                int(kernel.block_dim.z),
                            ],
                        })
        except Exception as exc:  # noqa: BLE001 — typed below
            parse_error = str(exc)

    status, detail = root_trace_verdict(
        pb_bytes=len(raw), parse_error=parse_error,
        trace_contract_id=contract_id, memory_payload_classes=classes,
        launch_geometries=launches,
    )
    return {
        "status": status,
        "detail": detail,
        "pb_bytes": len(raw),
        "trace_contract_id": contract_id,
        "memory_payload_classes": classes,
        "launch_count": len(launches or []),
        "launches": launches or [],
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], *, timeout: int = 30, cwd: Path | None = None,
         env: dict | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd) if cwd else None, env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(cmd)}"


def _parse_stats_admission_details(stats_path: Path) -> dict:
    """Sum ``(total_insts, total_reported_insts)`` across every kernel-launch row
    of a tracer ``stats.csv``. Mirrors
    ``activation_trace_orchestrator._parse_stats_csv`` so the smoke's admitted
    counters match exactly what the capture harness would compute."""
    if not stats_path.is_file():
        return {"expected": None, "admitted": None, "row_count": 0,
                "error": "stats file missing"}
    try:
        with stats_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_stats_read",
        )
        if storage_failure is not None:
            raise storage_failure from exc
        return {"expected": None, "admitted": None, "row_count": 0,
                "error": "stats file unreadable"}
    if not rows:
        return {"expected": None, "admitted": None, "row_count": 0,
                "error": "stats file empty"}
    header = [h.strip() for h in rows[0]]
    try:
        ti = header.index("total_insts")
        tri = header.index("total_reported_insts")
    except ValueError:
        return {"expected": None, "admitted": None, "row_count": 0,
                "error": "required stats columns missing"}
    expected = admitted = 0
    row_count = 0
    for line_no, row in enumerate(rows[1:], start=2):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if len(cells) <= max(ti, tri):
            return {"expected": None, "admitted": None, "row_count": row_count,
                    "error": f"stats row {line_no} is truncated"}
        try:
            row_expected = int(cells[ti])
            row_admitted = int(cells[tri])
        except ValueError:
            return {"expected": None, "admitted": None, "row_count": row_count,
                    "error": f"stats row {line_no} has non-integer counters"}
        if row_expected < 0 or row_admitted < 0:
            return {"expected": None, "admitted": None, "row_count": row_count,
                    "error": f"stats row {line_no} has negative counters"}
        expected += row_expected
        admitted += row_admitted
        row_count += 1
    if row_count == 0:
        return {"expected": None, "admitted": None, "row_count": 0,
                "error": "stats file has zero data rows"}
    return {"expected": expected, "admitted": admitted,
            "row_count": row_count, "error": None}


def _parse_stats_admission(stats_path: Path) -> tuple[int | None, int | None]:
    details = _parse_stats_admission_details(stats_path)
    return details["expected"], details["admitted"]


def _validate_smoke_trace(traces_dir: Path, repo_root: Path
                          ) -> tuple[bool | None, str | None]:
    """Run the tracer's own fail-closed contract validator (``validate_trace_dir``)
    on a captured smoke trace. Imported lazily so the pure verdict helper stays
    importable on a host without the tracer package on ``sys.path``. Returns
    ``(ok, typed_reason)``; ``(None, reason)`` if the validator can't be run."""
    nvbit = str(repo_root / "tools" / "nvbit_tracer")
    if nvbit not in sys.path:
        sys.path.insert(0, nvbit)
    try:
        from import_trace_contract import validate_trace_dir  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        return None, f"validator import failed: {exc}"
    try:
        result = validate_trace_dir(traces_dir)
    except PreflightStorageFailure:
        raise
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="injection_smoke_contract_read",
        )
        if storage_failure is not None:
            raise storage_failure from exc
        return None, f"validator raised: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"validator raised: {exc}"
    ok = bool(getattr(result, "ok", False))
    reason = None if ok else (getattr(result, "typed_reason", None) or "unknown")
    return ok, reason


def _smoke_workload_argv(shape_argv: list[str], temp_root: Path) -> list[str]:
    """Keep the portable smoke's own measurement manifest inside temp scope."""
    if "--output_dir" in shape_argv:
        raise ValueError("pinned smoke argv must not pre-bind --output_dir")
    return [*shape_argv, "--output_dir", str(temp_root / "workload_output")]


# --------------------------------------------------------------------------- #
# Probes.
# --------------------------------------------------------------------------- #
def probe_worktree(repo_root: Path, ignore_relpaths: list[str]) -> dict:
    rc, out, err = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if rc != 0 or not out.strip():
        return {"status": STATUS_FAILED, "detail": f"git rev-parse HEAD failed: {err.strip()}"}
    head = out.strip()
    rc, out, err = _run(["git", "status", "--porcelain", "-uall"], cwd=repo_root)
    if rc != 0:
        return {"status": STATUS_FAILED, "git_head": head,
                "detail": f"git status failed: {err.strip()}"}
    dirty = remaining_dirty_lines(out.splitlines(), ignore_relpaths)
    clean = not dirty
    return {
        "status": STATUS_OK if clean else STATUS_MISMATCH,
        "git_head": head,
        "working_tree_clean": clean,
        "dirty_paths": dirty[:20],
    }


def _run_retained_ncu_version(descriptor: int) -> tuple[int, str, str]:
    command = [f"/proc/self/fd/{descriptor}", "--version"]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        pass_fds=(descriptor,),
        env=ncu_executor.ncu_action_environment(),
        timeout=30,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _ncu_probe_result(
    *, status: str, expected: str | None, detail: str | None = None,
    execution_mode: str | None = None, host_machine: str | None = None,
    launcher_resolved_path: str | None = None,
    launcher_sha256: str | None = None,
    launcher_target_relpath: str | None = None,
    resolved_path: str | None = None, executable_sha256: str | None = None,
    version_line: str | None = None,
) -> dict:
    result = {
        "status": status,
        "expected": expected,
        "observed": version_line,
        "execution_mode": execution_mode,
        "host_machine": host_machine,
        "launcher_resolved_path": launcher_resolved_path,
        "launcher_sha256": launcher_sha256,
        "launcher_target_relpath": launcher_target_relpath,
        "resolved_path": resolved_path,
        "executable_sha256": executable_sha256,
        "version_line": version_line,
    }
    if detail:
        result["detail"] = detail
    return result


def _validate_retained_ncu_file(
    retained: immutable.RetainedExecutable, *, role: str,
) -> None:
    try:
        writable = os.access(
            f"/proc/self/fd/{retained.descriptor}",
            os.W_OK,
            effective_ids=True,
        )
    except (OSError, TypeError) as exc:
        raise ValueError(f"cannot establish {role} write protection: {exc}") from exc
    if writable:
        raise ValueError(f"{role} is writable by the preflight process")


def _validate_ncu_inner_elf(retained: immutable.RetainedExecutable) -> None:
    _validate_retained_ncu_file(retained, role="NCU inner executable")
    try:
        header = os.pread(retained.descriptor, 20, 0)
    except OSError as exc:
        raise ValueError(f"cannot read NCU inner ELF header: {exc}") from exc
    if len(header) != 20 or header[:4] != b"\x7fELF":
        raise ValueError("NCU inner executable is not an ELF file")
    if header[4] != 2 or header[5] != 1:
        raise ValueError("NCU inner executable is not little-endian ELF64")
    if int.from_bytes(header[16:18], "little") != 2:
        raise ValueError("NCU inner executable is not ET_EXEC")
    if int.from_bytes(header[18:20], "little") != 62:
        raise ValueError("NCU inner executable is not EM_X86_64")
    retained.verify_unchanged()


def probe_ncu(
    pins: dict, *, resolve_command=None, open_executable=None,
    run_version=None, host_machine: str | None = None,
) -> dict:
    raw_expected = pins.get("toolchain", {}).get("ncu_version_prefix")
    expected_for_failure = raw_expected if isinstance(raw_expected, str) else None
    try:
        expected = toolchain_contract.normalized_ncu_version_prefix(raw_expected)
    except toolchain_contract.CaptureToolchainContractError as exc:
        return _ncu_probe_result(
            status=STATUS_FAILED,
            expected=expected_for_failure,
            detail=str(exc),
        )

    machine = os.uname().machine if host_machine is None else host_machine
    if machine != NCU_HOST_MACHINE:
        return _ncu_probe_result(
            status=STATUS_FAILED,
            expected=expected,
            detail=f"unsupported NCU host machine: {machine!r}",
        )
    resolver = resolve_command or shutil.which
    opener = open_executable or immutable.open_retained_executable
    runner = run_version or _run_retained_ncu_version
    try:
        launcher_path_text = resolver("ncu")
    except OSError as exc:
        return _ncu_probe_result(
            status=STATUS_UNAVAILABLE,
            expected=expected,
            detail=f"cannot resolve ncu through PATH: {exc}",
        )
    if not isinstance(launcher_path_text, str) or not launcher_path_text:
        return _ncu_probe_result(
            status=STATUS_UNAVAILABLE,
            expected=expected,
            detail="ncu is unresolved on PATH",
        )
    if (
        not Path(launcher_path_text).is_absolute()
        or os.path.normpath(launcher_path_text) != launcher_path_text
    ):
        return _ncu_probe_result(
            status=STATUS_FAILED,
            expected=expected,
            detail=(
                "resolved ncu launcher path is not canonical absolute: "
                f"{launcher_path_text!r}"
            ),
        )

    launcher_path = Path(launcher_path_text)
    inner_path = launcher_path.parent / NCU_LAUNCHER_TARGET_RELPATH
    try:
        with opener(launcher_path) as launcher:
            _validate_retained_ncu_file(launcher, role="NCU launcher")
            with opener(inner_path) as inner:
                _validate_ncu_inner_elf(inner)
                rc, out, err = runner(inner.descriptor)
                if (
                    type(rc) is not int
                    or not isinstance(out, str)
                    or not isinstance(err, str)
                ):
                    raise ValueError("NCU version runner returned an invalid result")
                if rc != 0:
                    return _ncu_probe_result(
                        status=STATUS_UNAVAILABLE,
                        expected=expected,
                        detail=(
                            "retained NCU inner --version failed: "
                            f"{err.strip()}"
                        ),
                    )
                version_lines = [
                    line.strip()
                    for line in out.splitlines()
                    if re.match(r"^\s*Version\s+", line)
                ]
                if len(version_lines) != 1:
                    return _ncu_probe_result(
                        status=STATUS_MISMATCH,
                        expected=expected,
                        detail=(
                            "NCU inner version output has no unique Version line: "
                            f"{version_lines!r}"
                        ),
                    )
                observed_line = version_lines[0]
                if not toolchain_contract.ncu_version_matches(
                    observed_line, expected,
                ):
                    return _ncu_probe_result(
                        status=STATUS_MISMATCH,
                        expected=expected,
                        detail=f"NCU version line mismatch: {observed_line!r}",
                    )
                launcher_resolved_path = str(launcher.path)
                launcher_sha256 = launcher.sha256
                resolved_path = str(inner.path)
                executable_sha256 = inner.sha256
        return _ncu_probe_result(
            status=STATUS_OK,
            expected=expected,
            execution_mode=NCU_EXECUTION_MODE,
            host_machine=machine,
            launcher_resolved_path=launcher_resolved_path,
            launcher_sha256=launcher_sha256,
            launcher_target_relpath=NCU_LAUNCHER_TARGET_RELPATH,
            resolved_path=resolved_path,
            executable_sha256=executable_sha256,
            version_line=observed_line,
        )
    except (
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
        immutable.ImmutableArtifactError,
    ) as exc:
        return _ncu_probe_result(
            status=STATUS_FAILED,
            expected=expected,
            detail=f"NCU executable identity is invalid: {exc}",
        )


def probe_driver(pins: dict) -> dict:
    expected = pins["toolchain"]["driver_version"]
    rc, out, err = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if rc != 0:
        return {"status": STATUS_UNAVAILABLE, "expected": expected,
                "detail": f"nvidia-smi failed: {err.strip()}"}
    observed = out.strip().splitlines()[0].strip() if out.strip() else None
    ok = version_matches(observed, expected, mode="exact")
    return {"status": STATUS_OK if ok else STATUS_MISMATCH,
            "observed": observed, "expected": expected}


def probe_nvcc(pins: dict, *, executable: str = "nvcc") -> dict:
    try:
        expected = toolchain_contract.structured_nvcc_release(
            pins["toolchain"]["nvcc_release"]
        )
    except toolchain_contract.CaptureToolchainContractError as exc:
        return {
            "status": STATUS_FAILED,
            "expected": pins.get("toolchain", {}).get("nvcc_release"),
            "executable": executable,
            "detail": str(exc),
        }
    rc, out, err = _run([executable, "--version"])
    if rc != 0:
        return {"status": STATUS_UNAVAILABLE, "expected": expected,
                "executable": executable,
                "detail": f"{executable} --version failed: {err.strip()}"}
    try:
        observed = toolchain_contract.parse_nvcc_release_banner(out)
        toolchain_contract.require_nvcc_release(observed, expected)
        ok = True
    except toolchain_contract.CaptureToolchainContractError:
        observed = None
        ok = False
    return {"status": STATUS_OK if ok else STATUS_MISMATCH,
            "observed": observed, "expected": expected,
            "executable": executable}


def probe_nvbit(pins: dict, repo_root: Path) -> dict:
    expected = pins["toolchain"]["nvbit_version"]
    header = repo_root / pins["injection_runtime"]["nvbit_version_header_relpath"]
    if not header.is_file():
        return {"status": STATUS_UNAVAILABLE, "expected": expected,
                "detail": f"nvbit header absent: {header}"}
    m = re.search(r'#define\s+NVBIT_VERSION\s+"([^"]+)"', header.read_text())
    observed = m.group(1) if m else None
    ok = version_matches(observed, expected, mode="exact")
    return {"status": STATUS_OK if ok else STATUS_MISMATCH,
            "observed": observed, "expected": expected}


def _row_argv(row: dict) -> tuple[str, ...]:
    """Return one exact argv, retaining a compatibility path for old tests."""
    argv = row.get("argv")
    if isinstance(argv, (list, tuple)) and all(isinstance(item, str) for item in argv):
        return tuple(argv)
    command = row.get("command")
    if not isinstance(command, str) or not command:
        return ()
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return ()


def _is_sol_execbench_module(name: str) -> bool:
    return name == "sol_execbench" or name.startswith("sol_execbench.")


def _python_source_imports_sol_execbench(source: str) -> bool:
    """Recognize imports in a real Python ``-c`` program, not shell prose."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_sol_execbench_module(alias.name) for alias in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and isinstance(node.module, str)
            and _is_sol_execbench_module(node.module)
        ):
            return True
    return False


def _is_sol_execbench_path(value: str) -> bool:
    """Match a path component exactly; never match arbitrary argv prose."""
    if not value or any(character.isspace() for character in value):
        return False
    return "sol_execbench" in Path(value).parts


def _pytest_targets_sol_execbench(argv: tuple[str, ...], start: int) -> bool:
    return any(_is_sol_execbench_path(value) for value in argv[start:])


def _python_process_marker(argv: tuple[str, ...]) -> str | None:
    """Classify the program Python is actually executing."""
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == "--":
            index += 1
            break
        if value == "-m" and index + 1 < len(argv):
            module = argv[index + 1]
            if _is_sol_execbench_module(module):
                return "sol_execbench"
            if module in {"pytest", "py.test"} and _pytest_targets_sol_execbench(
                argv, index + 2,
            ):
                return "sol_execbench"
            return None
        if value == "-c" and index + 1 < len(argv):
            if _python_source_imports_sol_execbench(argv[index + 1]):
                return "sol_execbench"
            return None
        if value in {"-W", "-X"} and index + 1 < len(argv):
            index += 2
            continue
        if value.startswith("-"):
            index += 1
            continue
        break
    if index >= len(argv):
        return None
    target = argv[index]
    basename = Path(target).name.lower()
    if basename == "sol-execbench":
        # A console-script shebang appears in /proc as Python followed by the
        # script path; /proc/<pid>/exe therefore names Python, not this target.
        return "sol-execbench"
    if basename in _CONFLICTING_PROCESS_SCRIPTS:
        return basename
    if basename in _PYTEST_EXECUTABLES:
        return (
            "sol_execbench"
            if _pytest_targets_sol_execbench(argv, index + 1)
            else None
        )
    if _is_sol_execbench_path(target):
        return "sol_execbench"
    return None


def _capture_process_marker(row: dict) -> str | None:
    """Identify an executing capture tool from argv/executable identities."""
    argv = _row_argv(row)
    if not argv:
        return None
    executable = row.get("executable")
    argv0_name = Path(argv[0]).name.lower()
    if isinstance(executable, str) and executable:
        # /proc/<pid>/exe resolves executable symlinks and cannot be spoofed by
        # a caller-controlled argv[0].  Fall back to argv[0] only when that
        # kernel identity is unavailable.
        identity_name = Path(
            executable.removesuffix(" (deleted)")
        ).name.lower()
    else:
        identity_name = argv0_name
    if identity_name == "ncu":
        return "ncu"
    if identity_name == "sol-execbench":
        return "sol-execbench"
    if identity_name in _CONFLICTING_PROCESS_SCRIPTS:
        return identity_name
    if identity_name in _PYTEST_EXECUTABLES:
        return "sol_execbench" if _pytest_targets_sol_execbench(argv, 1) else None
    if _PYTHON_EXECUTABLE.fullmatch(identity_name):
        return _python_process_marker(argv)
    # Shells, monitors, editors, and arbitrary commands are deliberately not
    # searched for marker substrings.  Their argv can contain command text or
    # environment paths without executing that text themselves; the actual
    # child executable is independently present in /proc and is classified.
    return None


def _verified_ancestor_pids(processes: list[dict], current_pid: int) -> set[int]:
    """Return self + the PID-reuse-safe, directly verified parent chain."""
    rows = {
        row["pid"]: row
        for row in processes
        if type(row.get("pid")) is int
    }
    if current_pid not in rows:
        raise ValueError("current process is absent from the /proc snapshot")
    verified: set[int] = set()
    child_pid = current_pid
    while child_pid not in verified:
        child = rows.get(child_pid)
        if child is None:
            break
        verified.add(child_pid)
        parent_pid = child.get("ppid")
        child_started = child.get("start_time_ticks")
        if type(parent_pid) is not int or parent_pid <= 0:
            break
        parent = rows.get(parent_pid)
        if parent is None:
            break
        parent_started = parent.get("start_time_ticks")
        if (
            type(child_started) is not int
            or type(parent_started) is not int
            or parent_started > child_started
        ):
            # A later-starting process occupying the recorded parent PID is
            # not the parent we observed; never grant it the ancestor ignore.
            break
        child_pid = parent_pid
    return verified


def _read_process_stat(path: Path) -> tuple[int, int]:
    """Read PPID and start ticks from one Linux /proc/<pid>/stat identity."""
    raw = path.read_text()
    close = raw.rfind(")")
    if close < 0:
        raise ValueError(f"malformed process stat: {path}")
    fields = raw[close + 1:].split()
    if len(fields) <= 19:
        raise ValueError(f"short process stat: {path}")
    return int(fields[1]), int(fields[19])


def _snapshot_processes(proc_root: Path = Path("/proc")) -> list[dict]:
    """Read race-checked Linux process identities with exact argv boundaries."""
    processes: list[dict] = []
    for child in sorted(proc_root.iterdir(), key=lambda path: os.fsencode(path.name)):
        if not child.name.isdigit():
            continue
        try:
            before = _read_process_stat(child / "stat")
            raw_argv = (child / "cmdline").read_bytes()
            try:
                executable = os.readlink(child / "exe")
            except OSError:
                executable = None
            raw_argv_after = (child / "cmdline").read_bytes()
            try:
                executable_after = os.readlink(child / "exe")
            except OSError:
                executable_after = None
            after = _read_process_stat(child / "stat")
        except (OSError, UnicodeError, ValueError):
            # Processes may exit while /proc is enumerated.  The double stat
            # read prevents a recycled PID from inheriting another argv.
            continue
        if (
            before != after
            or raw_argv != raw_argv_after
            or executable != executable_after
        ):
            continue
        argv = tuple(
            item.decode("utf-8", "surrogateescape")
            for item in raw_argv.rstrip(b"\0").split(b"\0")
        )
        processes.append({
            "pid": int(child.name),
            "ppid": before[0],
            "start_time_ticks": before[1],
            "argv": argv,
            "executable": executable,
        })
    return processes


def process_hygiene_verdict(
    processes: list[dict], *, ignored_pids: set[int] | frozenset[int] = frozenset(),
) -> tuple[str, str, list[dict]]:
    """Classify exact process identities, ignoring only verified ancestors."""
    conflicts: list[dict] = []
    for row in processes:
        if row.get("pid") in ignored_pids:
            continue
        marker = _capture_process_marker(row)
        if marker is not None:
            argv = _row_argv(row)
            conflicts.append({
                "pid": row.get("pid"),
                "marker": marker,
                "command": shlex.join(argv)[:500],
            })
    if conflicts:
        return STATUS_FAILED, f"{len(conflicts)} conflicting capture process(es)", conflicts
    return STATUS_OK, "no SOL evaluator or NCU process is active", []


def probe_capture_processes() -> dict:
    """Persist a read-only process snapshot proving capture exclusivity."""
    try:
        processes = _snapshot_processes()
        current_pid = os.getpid()
        ignored = _verified_ancestor_pids(processes, current_pid)
    except (OSError, ValueError) as exc:
        return {
            "status": STATUS_FAILED,
            "detail": f"process enumeration failed: {exc}",
            "conflicts": [],
        }
    status, detail, conflicts = process_hygiene_verdict(
        processes, ignored_pids=ignored,
    )
    return {
        "status": status,
        "detail": detail,
        "enumerated_processes": len(processes) - len(ignored),
        "ignored_self_pid": current_pid,
        "ignored_verified_ancestor_pids": sorted(ignored - {current_pid}),
        "conflicts": conflicts,
    }


def clock_state_verdict(samples: list[dict], ref: dict) -> tuple[str, str]:
    """Require repeated exact clock readback plus P0 and persistence.

    Driver 610 no longer exposes a direct getter for ``-lgc/-lmc`` lock ranges,
    so this is the strongest read-only proxy available: three independently
    queried exact samples before and after the injection smoke.
    """
    if len(samples) < 3:
        return STATUS_FAILED, f"only {len(samples)} clock sample(s), need at least 3"
    for i, observed in enumerate(samples):
        if (observed.get("persistence_mode") != ref["persistence_mode"]
                or observed.get("sm_clock_mhz") != ref["sm_clock_mhz"]
                or observed.get("graphics_clock_mhz") != ref["sm_clock_mhz"]
                or observed.get("mem_clock_mhz") != ref["mem_clock_mhz"]
                or observed.get("pstate") != "P0"):
            return STATUS_MISMATCH, (
                f"clock sample[{i}] does not match the reference lock proxy: "
                f"{observed!r}")
    return STATUS_OK, (
        f"{len(samples)} consecutive exact P0 samples at "
        f"{ref['sm_clock_mhz']}/{ref['mem_clock_mhz']} MHz")


def probe_clock_state(pins: dict) -> dict:
    ref = pins["reference_clock_state"]
    samples: list[dict] = []
    for i in range(3):
        rc, out, err = _run([
            "nvidia-smi",
            "--query-gpu=persistence_mode,clocks.gr,clocks.sm,clocks.mem,pstate",
            "--format=csv,noheader,nounits",
        ])
        if rc != 0:
            return {"status": STATUS_UNAVAILABLE, "expected": ref,
                    "detail": f"nvidia-smi clock query failed: {err.strip()}",
                    "samples": samples}
        line = out.strip().splitlines()[0] if out.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        observed = {
            "persistence_mode": None,
            "graphics_clock_mhz": None,
            "sm_clock_mhz": None,
            "mem_clock_mhz": None,
            "pstate": None,
        }
        try:
            observed.update(
                persistence_mode=parts[0], graphics_clock_mhz=int(parts[1]),
                sm_clock_mhz=int(parts[2]), mem_clock_mhz=int(parts[3]),
                pstate=parts[4],
            )
        except (IndexError, ValueError):
            return {"status": STATUS_FAILED, "expected": ref,
                    "observed": observed, "samples": samples,
                    "detail": f"could not parse clock query: {line!r}"}
        samples.append(observed)
        if i < 2:
            time.sleep(0.05)
    status, detail = clock_state_verdict(samples, ref)
    rc, event_out, _ = _run([
        "nvidia-smi",
        "--query-gpu=clocks_event_reasons.applications_clocks_setting",
        "--format=csv,noheader",
    ])
    return {
        "status": status,
        "detail": detail,
        "observed": samples[-1],
        "samples": samples,
        "sample_count": len(samples),
        "expected": {k: ref[k] for k in
                     ("persistence_mode", "sm_clock_mhz", "mem_clock_mhz")},
        "lock_state_evidence": "repeated_exact_readback_proxy",
        "applications_clock_setting": (
            event_out.strip() if rc == 0 else "query_unavailable"),
    }


def wait_for_clock_state(
    pins: dict, *, max_attempts: int = 31, poll_interval_s: float = 1.0,
) -> dict:
    """Wait for, then return, a complete three-sample exact-P0 verdict.

    A completed CUDA workload can leave the device reporting the compute P2
    performance state briefly even though the pinned clock frequencies never
    drifted. The safety contract still requires P0; repeat the *entire*
    three-sample gate rather than accepting P2 or weakening the verdict.
    Exhaustion remains a typed mismatch/failure so a caller can preserve the
    refusal and wait again without bypassing the gate.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not isinstance(poll_interval_s, (int, float)) or poll_interval_s < 0:
        raise ValueError("poll_interval_s must be nonnegative")
    attempts: list[dict] = []
    last: dict | None = None
    for ordinal in range(1, max_attempts + 1):
        last = probe_clock_state(pins)
        attempts.append({
            "ordinal": ordinal,
            "status": last.get("status"),
            "detail": last.get("detail"),
            "observed": last.get("observed"),
        })
        if last.get("status") == STATUS_OK:
            settled = dict(last)
            settled["settle_attempt_count"] = ordinal
            settled["settle_attempts"] = attempts
            return settled
        if ordinal < max_attempts:
            time.sleep(float(poll_interval_s))
    assert last is not None
    exhausted = dict(last)
    exhausted["detail"] = (
        f"clock state did not settle after {max_attempts} complete gate "
        f"attempt(s): {last.get('detail')}"
    )
    exhausted["settle_attempt_count"] = max_attempts
    exhausted["settle_attempts"] = attempts
    return exhausted


def probe_injection_runtime(
    pins: dict,
    repo_root: Path,
    *,
    smoke_temp_root: Path | str | None = None,
) -> dict:
    """Real contract-valid NVBit injection smoke — the true capture-admission gate.

    ldd-resolution of the tracer ``.so`` is only a subcheck here: without
    ``.toolchain/pb/lib`` on ``LD_LIBRARY_PATH`` the ``CUDA_INJECTION64_PATH`` load
    fails with zero stderr and the app runs UNTRACED (per the injection playbook).
    But a resolvable ``.so`` still does not prove injection yields a usable trace,
    so this actually injects the pinned tracer into the smallest deterministic
    portable_lcd binary (into a temp dir OUTSIDE the repo), then requires a
    ``dynamic_trace.pb`` + ``stats.csv``, nonzero admitted instruction records, and
    a pass from the tracer's own ``validate_trace_dir`` contract validator. The
    smoke reads only admission counters + contract validity — never an NCU metric
    (Rule 1). Fail ⇒ capture blocked before any roster cell runs.
    """
    inj = pins["injection_runtime"]
    pb_lib = repo_root / inj["pb_lib_relpath"]
    tracer_so = repo_root / inj["tracer_so_relpath"]
    smoke_binary = repo_root / inj["smoke_binary_relpath"]
    shape_argv = list(inj.get("smoke_shape_argv", []))
    result: dict = {
        "tracer_so": inj["tracer_so_relpath"],
        "pb_lib": inj["pb_lib_relpath"],
        "smoke_binary": inj["smoke_binary_relpath"],
        "smoke_shape_argv": shape_argv,
    }
    if not pb_lib.is_dir():
        result.update(status=STATUS_FAILED, detail=f"pb runtime dir absent: {pb_lib}")
        return result
    if not tracer_so.is_file():
        result.update(status=STATUS_FAILED,
                      detail=(
                          "tracer .so absent; restore the byte-exact preserved "
                          f"artifact before capture: {tracer_so}"
                      ))
        return result
    if not (smoke_binary.is_file() and os.access(smoke_binary, os.X_OK)):
        result.update(status=STATUS_FAILED,
                      detail=f"smoke binary absent/not executable: {smoke_binary}")
        return result
    result["tracer_so_sha256"] = _sha256_file(tracer_so)
    result["smoke_binary_sha256"] = _sha256_file(smoke_binary)

    # Clearance happens before ldd, temp creation, or workload launch.
    temp_plan = _injection_temp_plan(smoke_temp_root)

    # Build the injection env EXACTLY as the capture harness does.
    env = os.environ.copy()
    prev = env.get("LD_LIBRARY_PATH", "")
    if str(pb_lib) not in prev.split(":"):
        env["LD_LIBRARY_PATH"] = f"{pb_lib}:{prev}" if prev else str(pb_lib)

    rc, out, err = _run(["ldd", str(tracer_so)], env=env)
    ldd_missing = (["ldd failed: " + err.strip()] if rc != 0
                   else [ln.strip() for ln in out.splitlines() if "not found" in ln])
    result["ldd_missing_deps"] = ldd_missing
    if ldd_missing:
        status, detail = injection_smoke_verdict(
            ldd_missing=ldd_missing, returncode=0, pb_present=False,
            stats_present=False, expected=None, admitted=None,
            validator_ok=None, validator_reason=None,
            threadblocks_pb_count=None, root_trace_ok=None,
            root_trace_reason=None, root_trace_launch_count=None,
            banner_seen=False, full_coverage_ok=None,
            full_coverage_reason=None)
        result.update(status=status, detail=detail)
        return result

    tmp = temp_plan["root"]
    preserve_smoke_tree = False
    smoke_tree_created = False
    try:
        _smoke_output_mkdir(
            tmp, parents=True, exist_ok=False,
            operation="injection_smoke_root_create",
        )
        smoke_tree_created = True
        traces = temp_plan["traces"]
        _smoke_output_mkdir(
            traces, parents=False, exist_ok=False,
            operation="injection_smoke_traces_create",
        )
        workload_output = temp_plan["workload_output"]
        _smoke_output_mkdir(
            workload_output, parents=True, exist_ok=True,
            operation="injection_smoke_workload_output_create",
        )
        effective_shape_argv = [
            *shape_argv, "--output_dir", str(workload_output),
        ]
        result["smoke_effective_argv"] = effective_shape_argv
        env["CUDA_INJECTION64_PATH"] = str(tracer_so)
        env["USER_DEFINED_FOLDERS"] = "1"
        env["TRACES_FOLDER"] = str(traces)
        rc, _out, serr = _run([str(smoke_binary), *effective_shape_argv],
                              timeout=300, cwd=smoke_binary.parent, env=env)
        if rc != 0:
            _probe_failed_smoke_output_storage(traces)
        banner_seen = any("nvbit" in ln.lower() for ln in serr.splitlines())
        pb = traces / "dynamic_trace.pb"
        stats = traces / "stats.csv"
        pb_stat = _smoke_output_lstat(
            pb, operation="injection_smoke_root_stat",
        )
        stats_stat = _smoke_output_lstat(
            stats, operation="injection_smoke_stats_stat",
        )
        pb_present = pb_stat is not None and stat.S_ISREG(pb_stat.st_mode)
        stats_present = (
            stats_stat is not None and stat.S_ISREG(stats_stat.st_mode)
        )
        root_trace = (_validate_root_trace(pb, repo_root) if pb_present else {
            "status": STATUS_FAILED,
            "detail": "dynamic_trace.pb missing",
            "pb_bytes": 0,
            "trace_contract_id": None,
            "memory_payload_classes": None,
            "launch_count": 0,
            "launches": [],
        })
        stats_details = (
            _parse_stats_admission_details(stats) if stats_present else
            {"expected": None, "admitted": None, "row_count": 0,
             "error": "stats file missing"}
        )
        expected, admitted = stats_details["expected"], stats_details["admitted"]
        validator_ok, validator_reason = _validate_smoke_trace(traces, repo_root)
        tb_root = traces / "threadblocks"
        tb_stat = _smoke_output_lstat(
            tb_root, operation="injection_smoke_threadblocks_stat",
        )
        try:
            tb_count = (
                len(list(tb_root.rglob("*.pb")))
                if tb_stat is not None and stat.S_ISDIR(tb_stat.st_mode)
                else 0
            )
        except OSError as exc:
            storage_failure = preflight_storage_failure_from_exception(
                exc, operation="injection_smoke_threadblocks_walk",
            )
            if storage_failure is not None:
                raise storage_failure from exc
            raise
        full_coverage = _analyze_full_trace_coverage(
            traces, root_trace, expected, admitted, stats_details["row_count"],
        )
        status, detail = injection_smoke_verdict(
            ldd_missing=ldd_missing, returncode=rc, pb_present=pb_present,
            stats_present=stats_present, expected=expected, admitted=admitted,
            validator_ok=validator_ok, validator_reason=validator_reason,
            threadblocks_pb_count=tb_count,
            root_trace_ok=root_trace["status"] == STATUS_OK,
            root_trace_reason=root_trace["detail"],
            root_trace_launch_count=root_trace["launch_count"],
            banner_seen=banner_seen,
            full_coverage_ok=full_coverage["status"] == STATUS_OK,
            full_coverage_reason=full_coverage["detail"])
        result.update(
            status=status, detail=detail,
            smoke_returncode=rc, nvbit_banner_seen=banner_seen,
            expected_insts=expected, admitted_insts=admitted,
            validator_ok=validator_ok, validator_reason=validator_reason,
            threadblocks_pb_count=tb_count,
            root_trace_ok=root_trace["status"] == STATUS_OK,
            root_trace_detail=root_trace["detail"],
            root_trace_pb_bytes=root_trace["pb_bytes"],
            root_trace_contract_id=root_trace["trace_contract_id"],
            root_trace_memory_payload_classes=root_trace["memory_payload_classes"],
            root_trace_launch_count=root_trace["launch_count"],
            root_trace_launches=root_trace["launches"],
            full_coverage_ok=full_coverage["status"] == STATUS_OK,
            full_coverage_detail=full_coverage["detail"],
            grid_threadblock_count=full_coverage["grid_threadblock_count"],
            decoded_threadblock_count=full_coverage["decoded_threadblock_count"],
            decoded_record_count=full_coverage["decoded_record_count"],
            zero_record_threadblock_count=(
                full_coverage["zero_record_threadblock_count"]),
            sampling_sidecar_present=full_coverage["sampling_sidecar_present"],
            stats_row_count=full_coverage["stats_row_count"],
            stats_parse_error=stats_details["error"],
            dynamic_trace_pb_sha256=(
                _sha256_smoke_output(
                    pb, operation="injection_smoke_root_hash",
                ) if pb_present else None
            ),
            stats_csv_sha256=(
                _sha256_smoke_output(
                    stats, operation="injection_smoke_stats_hash",
                ) if stats_present else None
            ),
            smoke_stderr_tail=serr[-300:] if status != STATUS_OK else None,
        )
        # An unsuccessful/ambiguous probe is evidence.  Keep it even when the
        # output filesystem is healthy and classification remains semantic.
        preserve_smoke_tree = status != STATUS_OK
        return result
    except PreflightStorageFailure:
        # The incomplete tree is diagnostic evidence.  More importantly,
        # deletion is another mutation against the filesystem that just
        # reported a capacity/integrity fault, so do not attempt it.
        preserve_smoke_tree = True
        raise
    except BaseException:
        # Ambiguous probe failures are preserved, but never promoted to a
        # campaign-volume receipt merely because a nested input read used a
        # storage-like errno.
        preserve_smoke_tree = True
        raise
    finally:
        if smoke_tree_created and not preserve_smoke_tree:
            assert_external_output(
                tmp, repo_root=REPO_ROOT,
                operation="remove injection smoke root",
            )
            try:
                _cleanup_smoke_output(tmp)
            except PreflightStorageFailure:
                # Cleanup failure is itself evidence about the smoke output
                # filesystem.  Never retry or hide it.
                preserve_smoke_tree = True
                raise


def probe_trace_budget(budget_path: Path, output_root: Path) -> dict:
    if not budget_path.is_file():
        return {"status": STATUS_FAILED,
                "detail": f"trace-retention budget file absent: {budget_path}"}
    try:
        budget = json.loads(budget_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": STATUS_FAILED, "detail": f"budget unreadable: {exc}"}
    free = disk_free_gib(output_root)
    status, detail = evaluate_budget(budget, free)
    return {
        "status": status,
        "detail": detail,
        "retention_budget_gib": budget.get("retention_budget_gib"),
        "metadata_margin_gib": budget.get("metadata_margin_gib"),
        "raw_staging_margin_gib": budget.get("raw_staging_margin_gib"),
        "required_free_gib": (
            budget.get("retention_budget_gib", 0)
            + budget.get("raw_staging_margin_gib", 0)
            if all(
                isinstance(budget.get(name), (int, float))
                and not isinstance(budget.get(name), bool)
                for name in ("retention_budget_gib", "raw_staging_margin_gib")
            ) else None
        ),
        "free_gib": round(free, 2),
        "output_root": str(output_root),
        "budget_sha256": _sha256_file(budget_path),
    }


def probe_sol_native_environment(matrix_path: Path) -> dict:
    """Validate the matrix pin and restore the certified SOL compile identity."""
    if matrix_path.is_symlink() or not matrix_path.is_file():
        return {
            "status": STATUS_FAILED,
            "typed_error": "SOL_ENV_MATRIX_MISSING",
            "detail": f"capture matrix absent: {matrix_path}",
            "matrix_path": str(matrix_path),
            "matrix_sha256": None,
        }
    try:
        matrix = json.loads(matrix_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return {
            "status": STATUS_FAILED,
            "typed_error": "SOL_ENV_MATRIX_INVALID",
            "detail": str(exc),
            "matrix_path": str(matrix_path),
            "matrix_sha256": _sha256_file(matrix_path),
        }
    if not isinstance(matrix, dict):
        return {
            "status": STATUS_FAILED,
            "typed_error": "SOL_ENV_MATRIX_INVALID",
            "detail": "capture matrix root is not an object",
            "matrix_path": str(matrix_path),
            "matrix_sha256": _sha256_file(matrix_path),
        }
    if (
        matrix.get("schema_version") == 2
        and matrix.get("artifact_id")
        == sol_environment.SUPPLEMENTAL_MATRIX_ARTIFACT_ID
    ):
        probe = sol_environment.probe_supplemental_sol_environment(
            matrix, matrix_path=matrix_path, ambient=os.environ,
        )
    elif (
        matrix.get("schema_version") == 1
        and matrix.get("artifact_id") == representative_execution.ARTIFACT_ID
        and matrix.get("execution_profile")
        == representative_execution.EXECUTION_PROFILE
    ):
        try:
            representative_execution.validate_execution_matrix(
                matrix, repo_root=REPO_ROOT, require_clean_head=True,
                enforce_current_counts=True,
                matrix_path=matrix_path,
            )
        except representative_execution.RepresentativeExecutionError as exc:
            probe = {
                "status": STATUS_FAILED,
                "typed_error": "SOL_ENV_REPRESENTATIVE_MATRIX_INVALID",
                "detail": str(exc),
            }
        else:
            # Environment certification consumes only the common, re-sealed
            # variant/input-pin surface.  The temporary identity projection is
            # never serialized and cannot bypass the representative validator.
            environment_view = {
                **matrix,
                "schema_version": 2,
                "artifact_id": sol_environment.SUPPLEMENTAL_MATRIX_ARTIFACT_ID,
            }
            probe = sol_environment.probe_supplemental_sol_environment(
                environment_view, matrix_path=matrix_path, ambient=os.environ,
            )
            probe["execution_profile"] = (
                representative_execution.EXECUTION_PROFILE
            )
    elif (
        matrix.get("schema_version") == 1
        and matrix.get("matrix_id") == "sm86_reference_capture_matrix_v1"
    ):
        probe = sol_environment.probe_sol_environment(
            matrix, matrix_path=matrix_path, ambient=os.environ,
        )
    else:
        probe = {
            "status": STATUS_FAILED,
            "typed_error": "SOL_ENV_MATRIX_INVALID",
            "detail": "capture matrix identity is not canonical or supplemental",
        }
    return {
        **probe,
        "matrix_path": str(matrix_path),
        "matrix_sha256": _sha256_file(matrix_path),
    }


def probe_covariates() -> dict:
    """One read-only NVML telemetry sample (never gates; never mutates state)."""
    rc, out, _ = _run([
        "nvidia-smi",
        "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
        "clocks_event_reasons.active",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return {"status": STATUS_UNAVAILABLE}
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    try:
        return {
            "status": STATUS_OK,
            "sm_clock_mhz": float(parts[0]),
            "mem_clock_mhz": float(parts[1]),
            "temperature_c": float(parts[2]),
            "power_w": float(parts[3]),
            "throttle_reasons": int(parts[4], 16),
        }
    except (ValueError, IndexError):
        return {"status": STATUS_UNAVAILABLE}


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def build_manifest(
    *, pins_path: Path, budget_path: Path, output_root: Path,
    ignore_relpaths: list[str], repo_root: Path = REPO_ROOT,
    matrix_path: Path = DEFAULT_MATRIX,
    smoke_temp_root: Path | str | None = None,
) -> dict:
    pins = json.loads(pins_path.read_text())
    sol_environment_probe = probe_sol_native_environment(matrix_path)
    sol_contract = sol_environment_probe.get("environment_contract") or {}
    cuda_home = sol_contract.get("CUDA_HOME")
    nvcc_executable = (
        str(Path(cuda_home) / "bin" / "nvcc")
        if isinstance(cuda_home, str) and cuda_home else "nvcc"
    )
    checks = {
        "worktree": probe_worktree(repo_root, ignore_relpaths),
        "ncu": probe_ncu(pins),
        "driver": probe_driver(pins),
        "nvcc": probe_nvcc(pins, executable=nvcc_executable),
        "nvbit": probe_nvbit(pins, repo_root),
        "capture_processes": probe_capture_processes(),
        "clock_state": probe_clock_state(pins),
        "trace_budget": probe_trace_budget(budget_path, output_root),
        "sol_environment": sol_environment_probe,
    }
    cheap_blockers = [name for name in _BLOCKING_CHECKS
                      if name != "injection_runtime"
                      and checks[name].get("status") != STATUS_OK]
    if cheap_blockers:
        checks["injection_runtime"] = {
            "status": STATUS_FAILED,
            "not_run": True,
            "detail": (
                "real injection smoke not run because capture preconditions "
                f"failed: {cheap_blockers}"),
        }
    else:
        checks["injection_runtime"] = probe_injection_runtime(
            pins, repo_root, smoke_temp_root=smoke_temp_root,
        )

        # Close the race across the smoke itself: clocks and process exclusivity
        # must still be valid after the kernel exits, not merely before it starts.
        post_clock = wait_for_clock_state(pins)
        checks["clock_state"]["post_smoke"] = post_clock
        if post_clock.get("status") != STATUS_OK:
            checks["clock_state"]["status"] = post_clock.get("status", STATUS_FAILED)
            checks["clock_state"]["detail"] = (
                "post-smoke clock state failed: " + str(post_clock.get("detail")))
        post_processes = probe_capture_processes()
        checks["capture_processes"]["post_smoke"] = post_processes
        if post_processes.get("status") != STATUS_OK:
            checks["capture_processes"]["status"] = STATUS_FAILED
            checks["capture_processes"]["detail"] = (
                "post-smoke process hygiene failed: "
                + str(post_processes.get("detail")))
    blockers = [name for name in _BLOCKING_CHECKS
                if checks[name].get("status") != STATUS_OK]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": 3,
        "arch": pins.get("arch"),
        "sku": pins.get("sku"),
        "captured_at_utc": generated_at,
        "generated_at_utc": generated_at,
        "git_head": checks["worktree"].get("git_head"),
        "working_tree_clean": checks["worktree"].get("working_tree_clean", False),
        "pins_sha256": _sha256_file(pins_path),
        "capture_matrix_sha256": checks["sol_environment"].get(
            "matrix_sha256"
        ),
        "protocol_version": pins.get("measurement_protocol", {}).get("version"),
        "checks": checks,
        "covariates": probe_covariates(),
        "blockers": blockers,
        "capture_admitted": not blockers,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                    help="path to write the preflight provenance manifest")
    ap.add_argument("--pins", default=str(DEFAULT_PINS))
    ap.add_argument("--trace-budget", default=str(DEFAULT_BUDGET))
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--output-root", default=None,
                    help="filesystem whose free space is checked (default: repo root)")
    ap.add_argument(
        "--smoke-temp-root", default=None,
        help="Optional explicit smoke-temp root outside every Git worktree.",
    )
    args = ap.parse_args(argv)

    pins_path = Path(args.pins)
    if not pins_path.is_absolute():
        pins_path = REPO_ROOT / pins_path
    budget_path = Path(args.trace_budget)
    if not budget_path.is_absolute():
        budget_path = REPO_ROOT / budget_path
    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = REPO_ROOT / matrix_path
    try:
        output_path, smoke_plan = _campaign_output_plan(
            output=args.output,
            smoke_temp_root=args.smoke_temp_root,
        )
    except ProtectedPathError as exc:
        print(f"campaign_preflight: output refusal: {exc}", file=sys.stderr)
        return 2
    output_root = Path(args.output_root) if args.output_root else REPO_ROOT

    if not pins_path.is_file():
        print(f"campaign_preflight: pins file absent: {pins_path}", file=sys.stderr)
        return 2

    # Ignore the manifest we are about to write from the clean-tree check.
    ignore = []
    try:
        ignore.append(str(output_path.relative_to(REPO_ROOT)))
    except ValueError:
        pass

    try:
        manifest = build_manifest(
            pins_path=pins_path, budget_path=budget_path,
            output_root=output_root, ignore_relpaths=ignore,
            matrix_path=matrix_path,
            smoke_temp_root=smoke_plan["root"],
        )
    except PreflightStorageFailure as exc:
        print(exc.marker(), file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 — emission failure is exit 2
        print(f"campaign_preflight: manifest build failed: {exc}", file=sys.stderr)
        return 2

    try:
        output_path = _assert_staging_output(
            output_path, operation="write campaign preflight manifest",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        storage_failure = preflight_storage_failure_from_exception(
            exc, operation="preflight_manifest_emission",
        )
        if storage_failure is not None:
            print(storage_failure.marker(), file=sys.stderr)
            return 3
        print(f"campaign_preflight: could not write manifest: {exc}", file=sys.stderr)
        return 2

    if manifest["capture_admitted"]:
        print(f"campaign_preflight: PASS capture admitted — "
              f"git_head={manifest['git_head'][:12]} "
              f"protocol={manifest['protocol_version']} manifest={output_path}")
        return 0
    print(f"campaign_preflight: BLOCKED — blockers={manifest['blockers']}",
          file=sys.stderr)
    for name in manifest["blockers"]:
        print(f"  {name}: {manifest['checks'][name]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
