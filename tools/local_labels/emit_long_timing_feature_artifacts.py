#!/usr/bin/env python3
"""Attach deploy-legal feature artifacts to the long-duration timing cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import campaign_preflight  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import protected_paths  # noqa: E402
from tools.nvbit_tracer import capture_process as nvbit_process  # noqa: E402
from tools.nvbit_tracer import trace_admission as trace_admission_contract  # noqa: E402
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
)
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)
from tools.nvbit_tracer.protobuf_wire_preflight import (  # noqa: E402
    ProtobufDecodedResourceLimitExceeded,
    ProtobufWireMalformed,
    preflight_threadblock_protobuf,
    preflight_trace_protobuf,
)
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    DEFAULT_PB_LIB,
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
    sanitized_capture_environment,
)


DEFAULT_COHORT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_OUTPUT = DEFAULT_COHORT
DEFAULT_FEATURE_DIR = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort" / "feature_artifacts"
)
DEFAULT_TRACE_ROOT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort" / "nvbit_launch_traces"
)
DEFAULT_CACHE_HIT_PREDICTIONS = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_cache_hit_predictions_sm_86.json"
)
DEFAULT_STATIC_WORK = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "static_work"
    / "gemm_hmma_bf16_sm80_sm_86.json"
)
DEFAULT_STATIC_FEATURES = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "static_sass_features"
    / "gemm_hmma_bf16_sm80_sm_86.json"
)
DEFAULT_L1_HEAD = REPO_ROOT / "data" / "local_labels" / "sm_86_l1_head.json"
DEFAULT_L2_HEAD = REPO_ROOT / "data" / "local_labels" / "sm_86_v5_head.json"
DEFAULT_FAMILY_BUCKETS = (
    REPO_ROOT / "data" / "local_labels" / "sm120_cohort_family_buckets.toml"
)

TRACE_INSTR_BEGIN = 168
TRACE_INSTR_END = 174
OBSERVED_TRACE_MODE = "bounded_observed_global_store_trace"
OBSERVED_SECTOR_PRODUCER = "nvbit_threadblock_proto_summarize_launch_v1"
CACHE_PREDICTION_PRODUCER = "timing_per_cell_cache_head_prediction_v2"
CACHE_PREDICTION_METHOD = "per_cell_l1_l2_monotonic_calibration_head_v5"
CACHE_HEAD_SCORER_NAME = "PerBucketCalibratorEnsemble.predict"
L2_CAPACITY_BYTES = 6 * 1024 * 1024
L2_SLICE_COUNT = 84
SECTOR_SIZE_BYTES = 32

_TIMING_RETAINED_TRACER_NAME = "nvbit_retained_tracer.so"
_TIMING_STDOUT_NAME = "nvbit_launch.stdout.bin"
_TIMING_STDERR_NAME = "nvbit_launch.stderr.bin"
_TIMING_START_NAME = "nvbit_launch_start.json"
_TIMING_ATTEMPT_NAME = "nvbit_launch_attempt.json"
_TIMING_TRACE_LIMITS = trace_admission_contract.DEFAULT_TRACE_ADMISSION_LIMITS
_TIMING_TRACE_CONTRACT_ID = "sm86_addr_carry_shmem_split_v2"
_TIMING_CAPTURE_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_TIMING_CAPTURE_TIMEOUT_CODE = "TIMING_NVBIT_CAPTURE_TIMEOUT"
_TIMING_CAPTURE_LAUNCH_CODE = "TIMING_NVBIT_CAPTURE_LAUNCH_FAILED"
_SECTOR_SUMMARY_MAX_BYTES = 8 * 1024 * 1024
_OBSERVED_SECTOR_SUMMARY_MAX_BYTES = 8 * 1024 * 1024
_SECTOR_SUMMARY_SCHEMA = (
    REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "timing_nvbit_sector_summary.schema.json"
)
_SHAPE_POINT_MAX_BYTES = 200

_ACCESS_KIND_UNSPECIFIED = 0
_ACCESS_KIND_LOAD = 1
_ACCESS_KIND_STORE = 2
_ACCESS_KIND_ATOMIC = 3
_ACCESS_KIND_ASYNC_COPY = 4
_MEMORY_OP_CLASS_GLOBAL = 1


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _declared_absolute_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    return Path(os.path.abspath(os.fspath(candidate)))


def _require_no_symlink_components(path: Path, *, operation: str) -> None:
    absolute = _declared_absolute_path(path)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            mode = os.lstat(cursor).st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise protected_paths.ProtectedPathError(
                f"cannot inspect output path while preparing to {operation}: "
                f"{cursor}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise protected_paths.ProtectedPathError(
                f"refusing to {operation} through symlinked output path: {cursor}"
            )


def _assert_output_path(path: Path, *, operation: str) -> Path:
    candidate = protected_paths.assert_write_allowed(
        path,
        repo_root=REPO_ROOT,
        operation=operation,
    )
    _require_no_symlink_components(path, operation=operation)
    return candidate


def _assert_output_directory(path: Path, *, operation: str) -> Path:
    return _assert_output_path(path, operation=operation)


def _assert_output_target(path: Path, *, operation: str) -> Path:
    return _assert_output_path(path, operation=operation)


def _path_entry_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _require_absent_output_target(path: Path, *, operation: str) -> None:
    if _path_entry_exists(path):
        raise immutable.ImmutableArtifactError(
            f"refusing to {operation}; output already exists: {path}"
        )


def _require_shape_point(cell: dict[str, Any]) -> str:
    value = cell.get("shape_point")
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("shape_point must be a nonempty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("shape_point must be valid UTF-8") from exc
    if (
        len(encoded) > _SHAPE_POINT_MAX_BYTES
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError("shape_point must be one safe bounded path component")
    return value


def _validated_cohort_cells(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    cells = cohort.get("cells", [])
    if type(cells) is not list:
        raise RuntimeError("timing cohort cells must be a list")
    seen: set[str] = set()
    for cell in cells:
        if type(cell) is not dict:
            raise RuntimeError("timing cohort cell must be an object")
        shape_point = _require_shape_point(cell)
        if shape_point in seen:
            raise RuntimeError(f"duplicate timing cohort shape_point: {shape_point!r}")
        seen.add(shape_point)
    return cells


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _binary_path_for(slug_or_cell: str | dict[str, Any]) -> Path:
    if isinstance(slug_or_cell, dict):
        explicit = slug_or_cell.get("binary_path")
        if explicit:
            path = REPO_ROOT / str(explicit)
            if path.is_file():
                return path
        slug = str(slug_or_cell["slug"])
    else:
        slug = slug_or_cell
    candidates = (
        REPO_ROOT / "workloads" / "portable_lcd" / slug / f"{slug}_run",
        REPO_ROOT / "workloads" / "simbench_bw" / "l1" / slug / f"{slug}_run",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"binary for slug {slug!r} not found")


def _trace_window_for(cell: dict[str, Any]) -> dict[str, int]:
    window = cell.get("trace_window") or {}
    return {
        "instr_begin": int(window.get("instr_begin", TRACE_INSTR_BEGIN)),
        "instr_end": int(window.get("instr_end", TRACE_INSTR_END)),
    }


def _ceil_div(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor))


def _occupancy_from_launch(cell: dict[str, Any], regs_per_thread: int) -> dict[str, Any]:
    geom = cell["launch_geometry"]
    block_threads = int(geom["block_x"]) * int(geom["block_y"]) * int(geom["block_z"])
    warps_per_block = max(1, _ceil_div(block_threads, 32))
    regs_per_block = max(1, regs_per_thread * block_threads)
    shared_per_block = max(
        1,
        int(geom["shared_mem_static_bytes"]) + int(geom["shared_mem_dynamic_bytes"]),
    )
    max_blocks = min(
        32,
        65536 // regs_per_block,
        102400 // shared_per_block,
        64 // warps_per_block,
    )
    active_blocks = max(1, int(max_blocks))
    active_warps = active_blocks * warps_per_block
    return {
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "warps_per_block": warps_per_block,
        "source": "launch_geometry_registers_shared_memory_limits",
    }


def _trace_dir_for(trace_root: Path, cell: dict[str, Any]) -> Path:
    return trace_root / _require_shape_point(cell)


def _require_external_tracer(path: Path, *, trace_root: Path) -> Path:
    tracer = campaign_preflight._absolute_external_tracer(
        path, repo_root=REPO_ROOT,
    )
    try:
        tracer.relative_to(trace_root.resolve())
    except ValueError:
        return tracer
    raise ValueError("NVBit tracer must be outside the timing capture root")


def _timing_capture_base_environment() -> dict[str, str]:
    base_env = sanitized_capture_environment()
    base_env.update(
        {
            "LANG": "C",
            "LANGUAGE": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return base_env


def _stream_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="surrogatepass")


def _environment_sha256(environment: dict[str, str]) -> str:
    return immutable.sha256_bytes(immutable.canonical_json_bytes(environment))


def _timing_invocation(
    cell: dict[str, Any],
    window: dict[str, int],
    *,
    environment_sha256: str | None,
    environment_scope: str,
    timeout_s: int,
) -> dict[str, Any]:
    return {
        "command": ["target_binary_descriptor", *cell["run_args"]],
        "working_directory": str(REPO_ROOT),
        "timeout_seconds": timeout_s,
        "trace_window": {
            "instr_begin": window["instr_begin"],
            "instr_end": window["instr_end"],
            "dynamic_kernel_limit_end": 1,
            "terminate_upon_limit": 0,
        },
        "environment_scope": environment_scope,
        "environment_sha256": environment_sha256,
    }


def _run_timing_capture(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    pass_fds: tuple[int, ...],
) -> nvbit_process.NvbitCaptureProcessResult:
    """Run one descriptor-bound timing launch with contained descendants."""

    return nvbit_process.execute_capture_process(
        tuple(command),
        environment=env,
        working_directory=cwd,
        inherited_descriptors=pass_fds,
        timeout_seconds=timeout,
        max_output_bytes=_TIMING_CAPTURE_MAX_OUTPUT_BYTES,
        timeout_code=_TIMING_CAPTURE_TIMEOUT_CODE,
        launch_code=_TIMING_CAPTURE_LAUNCH_CODE,
    )


def _write_timing_capture_start(
    trace_root: immutable.PinnedDirectoryReader,
    *,
    cell: dict[str, Any],
    binary: immutable.RetainedExecutable,
    tracer: immutable.RetainedRegular,
    tracer_source_path: Path,
    tracer_payload: bytes,
    base_environment: dict[str, str],
    window: dict[str, int],
    timeout_s: int,
) -> str:
    immutable.write_regular_bytes_at(
        trace_root.descriptor,
        _TIMING_RETAINED_TRACER_NAME,
        tracer_payload,
        mode=0o444,
    )
    start = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_launch_start",
        "status": "started",
        "shape_point": cell["shape_point"],
        "target_binary": {
            "path": str(Path(binary.path).resolve()),
            "expected_sha256": cell.get("binary_sha256"),
            "sha256": binary.sha256,
            "size_bytes": os.fstat(binary.descriptor).st_size,
        },
        "tracer": {
            "source_kind": "external_non_git_retained_descriptor_v1",
            "source_path": str(tracer_source_path),
            "sha256": tracer.sha256,
            "size_bytes": tracer.size_bytes,
            "snapshot_path": _TIMING_RETAINED_TRACER_NAME,
            "execution_mode": "retained_proc_fd_v1",
        },
        "descriptor_bindings": {
            "command_0": "target_binary",
            "CUDA_INJECTION64_PATH": "tracer",
            "TRACES_FOLDER": "trace_directory",
        },
        "invocation": _timing_invocation(
            cell,
            window,
            environment_sha256=_environment_sha256(base_environment),
            environment_scope="filtered_base_environment_v1",
            timeout_s=timeout_s,
        ),
    }
    sealed = immutable.seal_model(start, field="start_seal_sha256")
    immutable.write_regular_bytes_at(
        trace_root.descriptor,
        _TIMING_START_NAME,
        immutable.canonical_json_bytes(sealed),
        mode=0o444,
    )
    trace_root.verify_path()
    return str(sealed["start_seal_sha256"])


def _write_timing_capture_attempt(
    trace_root: immutable.PinnedDirectoryReader,
    *,
    cell: dict[str, Any],
    binary: immutable.RetainedExecutable,
    tracer: immutable.RetainedRegular,
    start_seal_sha256: str,
    status: str,
    stdout: bytes,
    stderr: bytes,
    returncode: int | None,
    error: BaseException | None,
    invocation: dict[str, Any],
    trace_source_generation: dict[str, Any] | None,
) -> str:
    immutable.write_regular_bytes_at(
        trace_root.descriptor,
        _TIMING_STDOUT_NAME,
        stdout,
        mode=0o444,
    )
    immutable.write_regular_bytes_at(
        trace_root.descriptor,
        _TIMING_STDERR_NAME,
        stderr,
        mode=0o444,
    )
    attempt = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_launch_attempt",
        "status": status,
        "shape_point": cell["shape_point"],
        "start_seal_sha256": start_seal_sha256,
        "target_binary": {
            "path": str(Path(binary.path).resolve()),
            "expected_sha256": cell.get("binary_sha256"),
            "sha256": binary.sha256,
            "size_bytes": os.fstat(binary.descriptor).st_size,
        },
        "tracer": {
            "sha256": tracer.sha256,
            "size_bytes": tracer.size_bytes,
            "snapshot_path": _TIMING_RETAINED_TRACER_NAME,
            "execution_mode": "retained_proc_fd_v1",
        },
        "trace_source_generation": trace_source_generation,
        "process": {
            "returncode": returncode,
            "error_code": (
                getattr(error, "code", type(error).__name__)
                if error is not None else None
            ),
            "error_detail": str(error) if error is not None else None,
            "process_id": getattr(error, "process_id", None),
            "started_at_utc": getattr(error, "started_at_utc", None),
        },
        "invocation": invocation,
        "streams": {
            "stdout": {
                "path": _TIMING_STDOUT_NAME,
                "sha256": immutable.sha256_bytes(stdout),
                "size_bytes": len(stdout),
            },
            "stderr": {
                "path": _TIMING_STDERR_NAME,
                "sha256": immutable.sha256_bytes(stderr),
                "size_bytes": len(stderr),
            },
        },
    }
    sealed = immutable.seal_model(attempt, field="attempt_seal_sha256")
    immutable.write_regular_bytes_at(
        trace_root.descriptor,
        _TIMING_ATTEMPT_NAME,
        immutable.canonical_json_bytes(sealed),
        mode=0o444,
    )
    trace_root.verify_path()
    return str(sealed["attempt_seal_sha256"])


def _capture_nvbit_observed_sector_trace(
    *,
    cell: dict[str, Any],
    trace_root: Path,
    tracer_so_path: Path,
    timeout_s: int = 900,
) -> dict[str, str]:
    _require_shape_point(cell)
    trace_root = _assert_output_directory(
        trace_root,
        operation="create timing NVBit capture output",
    )
    tracer_so_path = _require_external_tracer(
        tracer_so_path, trace_root=trace_root,
    )
    immutable.ensure_directory(trace_root)
    trace_dir = _trace_dir_for(trace_root, cell)
    with immutable.PinnedDirectoryReader(trace_root) as retained_trace_root:
        try:
            os.mkdir(
                cell["shape_point"],
                mode=0o755,
                dir_fd=retained_trace_root.descriptor,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"refusing to replace existing NVBit timing trace: {trace_dir}"
            ) from exc
        os.fsync(retained_trace_root.descriptor)
        retained_trace_root.verify_path()
    binary = _binary_path_for(cell)
    window = _trace_window_for(cell)
    with (
        immutable.open_retained_regular(
            tracer_so_path, max_bytes=EXTERNAL_TRACER_MAX_BYTES
        ) as retained_tracer,
        immutable.open_retained_executable(binary) as retained_binary,
        immutable.PinnedDirectoryReader(trace_dir) as retained_trace_dir,
    ):
        tracer_payload = retained_tracer.read_bytes()
        base_environment = _timing_capture_base_environment()
        start_seal_sha256 = _write_timing_capture_start(
            retained_trace_dir,
            cell=cell,
            binary=retained_binary,
            tracer=retained_tracer,
            tracer_source_path=tracer_so_path,
            tracer_payload=tracer_payload,
            base_environment=base_environment,
            window=window,
            timeout_s=timeout_s,
        )
        proc: subprocess.CompletedProcess[bytes] | None = None
        env: dict[str, str] | None = None
        trace_source_generation: dict[str, Any] | None = None
        failure: BaseException | None = None
        try:
            expected_binary_sha256 = cell.get("binary_sha256")
            if retained_binary.sha256 != expected_binary_sha256:
                raise RuntimeError(
                    "timing binary differs from cohort authority: "
                    f"expected={expected_binary_sha256!r}; "
                    f"observed={retained_binary.sha256}"
                )
            trace_proc_path = f"/proc/self/fd/{retained_trace_dir.descriptor}"
            with retained_tracer_runtime_env(
                tracer_so_path,
                base_env=base_environment,
                pb_lib=DEFAULT_PB_LIB,
                runtime_manifest=default_runtime_manifest_path(tracer_so_path),
                retained_descriptor=retained_tracer.descriptor,
            ) as retained_runtime:
                env = sanitized_capture_environment(
                    retained_runtime.environment
                )
                env.update(
                    {
                        "CUDA_INJECTION64_PATH": retained_tracer.proc_path,
                        "USER_DEFINED_FOLDERS": "1",
                        "TRACES_FOLDER": trace_proc_path,
                        "SKIP_ENHANCED_TRACER": "1",
                        "INSTR_BEGIN": str(window["instr_begin"]),
                        "INSTR_END": str(window["instr_end"]),
                        "DYNAMIC_KERNEL_LIMIT_END": "1",
                        "TERMINATE_UPON_LIMIT": "0",
                        "NOBANNER": "1",
                    }
                )
                proc = _run_timing_capture(
                    [retained_binary.proc_path, *cell["run_args"]],
                    cwd=str(REPO_ROOT),
                    env=env,
                    timeout=timeout_s,
                    pass_fds=(
                        retained_binary.descriptor,
                        retained_tracer.descriptor,
                        retained_trace_dir.descriptor,
                        *retained_runtime.pass_fds,
                    ),
                )
                retained_binary.verify_unchanged()
                retained_tracer.verify_unchanged()
                retained_trace_dir.verify_path()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"NVBit observed sector capture failed for "
                        f"{cell['shape_point']} rc={proc.returncode}"
                    )
                trace_pb = trace_dir / "dynamic_trace.pb"
                if not trace_pb.is_file():
                    raise RuntimeError(f"NVBit capture did not emit {trace_pb}")
                threadblocks = trace_dir / "threadblocks"
                if not threadblocks.is_dir():
                    raise RuntimeError(
                        f"NVBit capture emitted no threadblock directory at {trace_dir}"
                    )
                source_snapshot = _snapshot_timing_trace_source(trace_dir)
                _require_authenticated_launch_geometry(
                    cell=cell,
                    admitted_launch=source_snapshot["admission"].launches[0],
                    stats_row=_read_stats_row_bytes(source_snapshot["stats_payload"]),
                )
                trace_source_generation = source_snapshot["source_generation"]
                retained_trace_dir.verify_path()
        except BaseException as exc:
            failure = exc

        stdout = _stream_bytes(
            proc.stdout if proc is not None else getattr(failure, "stdout", b"")
        )
        stderr = _stream_bytes(
            proc.stderr if proc is not None else getattr(failure, "stderr", b"")
        )
        returncode = proc.returncode if proc is not None else None
        status = "failed" if failure is not None else "succeeded"
        attempt_seal_sha256 = _write_timing_capture_attempt(
            retained_trace_dir,
            cell=cell,
            binary=retained_binary,
            tracer=retained_tracer,
            start_seal_sha256=start_seal_sha256,
            status=status,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            error=failure,
            invocation=_timing_invocation(
                cell,
                window,
                environment_sha256=(
                    _environment_sha256(env) if env is not None else None
                ),
                environment_scope=(
                    "effective_process_environment_v1"
                    if env is not None
                    else "not_constructed"
                ),
                timeout_s=timeout_s,
            ),
            trace_source_generation=trace_source_generation,
        )
        if failure is not None:
            raise failure
        assert trace_source_generation is not None
        return {
            "binary_sha256": retained_binary.sha256,
            "tracer_so_sha256": retained_tracer.sha256,
            "nvbit_launch_attempt_seal_sha256": attempt_seal_sha256,
            "trace_source_generation_sha256": trace_source_generation[
                "source_generation_sha256"
            ],
        }


_LAUNCH_GEOMETRY_FIELDS = (
    "grid_x",
    "grid_y",
    "grid_z",
    "block_x",
    "block_y",
    "block_z",
    "shared_mem_static_bytes",
    "shared_mem_dynamic_bytes",
)
_STATS_GEOMETRY_FIELDS = {
    "grid_dimX": "grid_x",
    "grid_dimY": "grid_y",
    "grid_dimZ": "grid_z",
    "block_dimX": "block_x",
    "block_dimY": "block_y",
    "block_dimZ": "block_z",
}


def _require_authenticated_launch_geometry(
    *,
    cell: dict[str, Any],
    admitted_launch: Any,
    stats_row: dict[str, str],
) -> dict[str, Any]:
    admitted_geometry = admitted_launch.kernel_signature_inputs()
    catalog_geometry = cell["launch_geometry"]
    catalog_kernel_name = str(catalog_geometry["kernel_name"])
    admitted_kernel_name = str(admitted_geometry["kernel_name"])
    admitted_kernel_base = (
        admitted_kernel_name[:-4]
        if admitted_kernel_name.endswith("___0")
        else admitted_kernel_name
    )
    if admitted_kernel_base != catalog_kernel_name:
        raise RuntimeError(
            "authenticated trace kernel name disagrees with timing catalog: "
            f"{admitted_kernel_name!r} != {catalog_kernel_name!r}"
        )

    for field in _LAUNCH_GEOMETRY_FIELDS:
        admitted_value = int(admitted_geometry[field])
        catalog_value = int(catalog_geometry[field])
        if admitted_value != catalog_value:
            raise RuntimeError(
                "authenticated trace launch geometry disagrees with timing catalog: "
                f"{field}={admitted_value} != {catalog_value}"
            )

    admitted_regs = int(admitted_geometry["regs_per_thread"])
    catalog_regs = int(cell["regs_per_thread"])
    if admitted_regs != catalog_regs:
        raise RuntimeError(
            "authenticated trace register count disagrees with timing catalog: "
            f"{admitted_regs} != {catalog_regs}"
        )

    identity_fields = {
        "device_id": str(int(admitted_launch.device_id)),
        "stream_id": str(int(admitted_launch.stream_id)),
        "kernel id": f"kernel-{int(admitted_launch.kernel_id)}.trace",
        "kernel mangled name": catalog_kernel_name,
    }
    for stats_field, expected_value in identity_fields.items():
        stats_value = stats_row.get(stats_field)
        if stats_value != expected_value:
            raise RuntimeError(
                "stats.csv launch identity disagrees with authenticated trace: "
                f"{stats_field}={stats_value!r} != {expected_value!r}"
            )
    for stats_field, geometry_field in _STATS_GEOMETRY_FIELDS.items():
        stats_value = stats_row.get(stats_field)
        if stats_value is None or stats_value == "":
            raise RuntimeError(
                "stats.csv launch geometry is incomplete: "
                f"missing {stats_field}"
            )
        try:
            parsed_stats_value = int(stats_value)
        except ValueError as exc:
            raise RuntimeError(
                "stats.csv launch geometry is not integral: "
                f"{stats_field}={stats_value!r}"
            ) from exc
        admitted_value = int(admitted_geometry[geometry_field])
        if parsed_stats_value != admitted_value:
            raise RuntimeError(
                "stats.csv launch geometry disagrees with authenticated trace: "
                f"{stats_field}={parsed_stats_value} != {admitted_value}"
            )
    return admitted_geometry


def _emit_nvbit_launch_summary(
    *,
    cell: dict[str, Any],
    trace_dir: Path,
    admission: Any,
    stats_payload: bytes,
    binary_sha256: str,
    tracer_build_sha256: str,
    source_generation: dict[str, Any],
) -> dict[str, Any]:
    from workloads.simbench_bw.nvbit_trace_summary_adapter import (
        emit_admitted_nvbit_trace_summary,
    )

    if (
        admission.launch_count != 1
        or len(admission.launches) != 1
        or admission.launches[0].launch_index != 0
    ):
        raise RuntimeError(
            "timing NVBit trace must carry exactly one authenticated launch"
        )
    if admission.stats_row_count != 1:
        raise RuntimeError("timing NVBit stats.csv must carry exactly one data row")
    admitted_launch = admission.launches[0]
    stats_path = trace_dir / "stats.csv"
    stats_row = _read_stats_row_bytes(stats_payload)
    admitted_geometry = _require_authenticated_launch_geometry(
        cell=cell,
        admitted_launch=admitted_launch,
        stats_row=stats_row,
    )

    summary_name = "nvbit_launch_summary.json"
    summary_path = trace_dir / summary_name
    with immutable.PinnedDirectoryReader(trace_dir) as output_parent:
        raw = emit_admitted_nvbit_trace_summary(
            admission=admission,
            slug=cell["slug"],
            shape_point=cell["shape_point"],
            repeat_index=0,
            launch_index=0,
            binary_sha256=binary_sha256,
            tracer_build_sha256=tracer_build_sha256,
            output_path=Path(summary_name),
            output_descriptor=output_parent.descriptor,
        )
        summary_payload = output_parent.read(summary_name)
    launch_geometry = {
        "kernel_name": admitted_geometry["kernel_name"],
        **{
            field: int(admitted_geometry[field])
            for field in _LAUNCH_GEOMETRY_FIELDS
        },
    }
    geom = cell["launch_geometry"]
    return {
        "path": _repo_rel(summary_path),
        "sha256": immutable.sha256_bytes(summary_payload),
        "raw_kernel_name": raw["kernel_name"],
        "raw_kernel_signature_sha256": raw["kernel_signature_sha256"],
        "harmonized_kernel_name": geom["kernel_name"],
        "harmonized_kernel_signature_sha256": cell["kernel_signature_sha256"],
        "inst_count": int(raw["inst_count"]),
        "launch_index": 0,
        "repeat_index": 0,
        "stats_csv_path": _repo_rel(stats_path),
        "stats_csv_sha256": immutable.sha256_bytes(stats_payload),
        "trace_contract_id": source_generation["trace_contract_id"],
        "trace_source_generation": source_generation,
        "launch_geometry": launch_geometry,
        "instruction_trace_mode": OBSERVED_TRACE_MODE,
    }


def _read_stats_row_bytes(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"stats.csv is not UTF-8: {exc}") from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise RuntimeError("stats.csv has no header")
        normalized_header = [str(name).strip() for name in reader.fieldnames]
        if len(set(normalized_header)) != len(normalized_header):
            raise RuntimeError("stats.csv has duplicate normalized header fields")
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            if None in raw_row:
                raise RuntimeError("stats.csv row has more fields than its header")
            row = {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in raw_row.items()
            }
            if not any(row.values()):
                continue
            rows.append(row)
            if len(rows) > 1:
                raise RuntimeError(
                    "timing NVBit stats.csv must carry exactly one data row"
                )
    except csv.Error as exc:
        raise RuntimeError(f"stats.csv is malformed: {exc}") from exc
    if len(rows) != 1:
        raise RuntimeError("timing NVBit stats.csv must carry exactly one data row")
    return rows[0]


def _nearest_rank_percentile(sorted_values: list[int], percentile: float) -> int:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile requires non-empty input")
    rank = max(1, math.ceil(percentile / 100.0 * n))
    return sorted_values[rank - 1]


def _preflight_threadblock_payload(raw: bytes, *, label: str) -> None:
    try:
        preflight_threadblock_protobuf(raw)
    except ProtobufDecodedResourceLimitExceeded as exc:
        raise RuntimeError(
            f"NVBit threadblock protobuf exceeds decoded resource limits at "
            f"{label}: {exc}"
        ) from exc
    except ProtobufWireMalformed as exc:
        raise RuntimeError(
            f"NVBit threadblock protobuf is malformed at {label}: {exc}"
        ) from exc


def _sha256_of_threadblock_snapshots(
    payloads: Sequence[tuple[str, bytes]],
) -> str:
    h = hashlib.sha256()
    for relative_path, payload in payloads:
        h.update(relative_path.encode("utf-8"))
        h.update(b"\t")
        h.update(immutable.sha256_bytes(payload).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _timing_tree_generation_limits(
    limits: trace_admission_contract.TraceAdmissionLimits,
) -> dict[str, int]:
    policy = trace_admission_contract.raw_trace_generation_limits(limits)
    return {
        name: int(policy[name])
        for name in (
            "max_files",
            "max_directories",
            "max_total_file_bytes",
            "max_total_relative_path_bytes",
        )
    }


def _timing_trace_source_inventory_unchecked(
    tree: immutable.PinnedTreeSnapshot,
    *,
    limits: trace_admission_contract.TraceAdmissionLimits = _TIMING_TRACE_LIMITS,
) -> tuple[
    tuple[tuple[str, immutable.MetadataSnapshot], ...],
    tuple[tuple[str, immutable.MetadataSnapshot], ...],
]:
    policy = trace_admission_contract.raw_trace_generation_limits(limits)
    max_file_bytes = int(policy["max_file_bytes"])
    path_limits = policy["max_file_bytes_by_path"]
    assert isinstance(path_limits, dict)
    for relative_path, metadata in tree.files:
        file_limit = min(
            max_file_bytes,
            int(path_limits.get(relative_path, max_file_bytes)),
        )
        if metadata[4] > file_limit:
            raise immutable.ImmutableArtifactLimitError(
                "timing trace source exceeds the bounded per-file limit: "
                f"{relative_path}: {metadata[4]} > {file_limit}"
            )
    if len(tree.files) > limits.tree_files:
        raise immutable.ImmutableArtifactLimitError(
            "timing trace source exceeds the bounded tree file-count limit"
        )
    if len(tree.directories) > limits.tree_directories:
        raise immutable.ImmutableArtifactLimitError(
            "timing trace source exceeds the bounded tree directory-count limit"
        )
    total_file_bytes = sum(metadata[4] for _name, metadata in tree.files)
    if total_file_bytes > int(policy["max_total_file_bytes"]):
        raise immutable.ImmutableArtifactLimitError(
            "timing trace source exceeds the bounded total-byte limit"
        )

    file_metadata = dict(tree.files)
    required = {"dynamic_trace.pb", "stats.csv"}
    missing = sorted(required - file_metadata.keys())
    if missing:
        raise RuntimeError(f"timing NVBit trace source is incomplete: {missing}")
    threadblocks = sorted(
        name
        for name in file_metadata
        if name.startswith("threadblocks/") and name.endswith(".pb")
    )
    if not threadblocks:
        raise RuntimeError("timing NVBit trace has no threadblock protobufs")
    if len(threadblocks) > limits.threadblock_files:
        raise immutable.ImmutableArtifactLimitError(
            "timing NVBit trace exceeds the bounded threadblock file-count limit"
        )

    dynamic_size = file_metadata["dynamic_trace.pb"][4]
    if dynamic_size > limits.trace_root_bytes:
        raise immutable.ImmutableArtifactLimitError(
            "timing NVBit dynamic_trace.pb exceeds the per-file byte limit"
        )
    stats_size = file_metadata["stats.csv"][4]
    if stats_size > limits.stats_bytes:
        raise immutable.ImmutableArtifactLimitError(
            "timing NVBit stats.csv exceeds the per-file byte limit"
        )
    threadblock_total = 0
    for name in threadblocks:
        size = file_metadata[name][4]
        if size > limits.threadblock_bytes:
            raise immutable.ImmutableArtifactLimitError(
                f"timing NVBit {name} exceeds the per-file byte limit"
            )
        if threadblock_total > limits.total_threadblock_bytes - size:
            raise immutable.ImmutableArtifactLimitError(
                "timing NVBit threadblock payloads exceed the total-byte limit"
            )
        threadblock_total += size

    source_names = ["dynamic_trace.pb", "stats.csv"]
    if "extra_info/allocations.json" in file_metadata:
        source_names.append("extra_info/allocations.json")
    source_names.extend(threadblocks)
    source_files = tuple(
        (name, file_metadata[name])
        for name in source_names
    )
    source_directories = tuple(
        (name, metadata)
        for name, metadata in tree.directories
        if (
            name == "threadblocks"
            or name.startswith("threadblocks/")
            or (
                "extra_info/allocations.json" in file_metadata
                and name == "extra_info"
            )
        )
    )
    return source_files, source_directories


def _timing_trace_source_inventory(
    tree: immutable.PinnedTreeSnapshot,
    *,
    limits: trace_admission_contract.TraceAdmissionLimits = _TIMING_TRACE_LIMITS,
) -> tuple[
    tuple[tuple[str, immutable.MetadataSnapshot], ...],
    tuple[tuple[str, immutable.MetadataSnapshot], ...],
]:
    try:
        return _timing_trace_source_inventory_unchecked(tree, limits=limits)
    except MemoryError as exc:
        raise immutable.ImmutableArtifactResourceError(
            "VERIFY_RESOURCE_BOUND: timing trace inventory exhausted memory"
        ) from exc


def _require_timing_trace_contract_id(raw: bytes, *, label: str) -> str:
    try:
        preflight_trace_protobuf(raw)
    except ProtobufDecodedResourceLimitExceeded as exc:
        raise RuntimeError(
            f"NVBit root protobuf exceeds decoded resource limits at {label}: {exc}"
        ) from exc
    except ProtobufWireMalformed as exc:
        raise RuntimeError(f"NVBit root protobuf is malformed at {label}: {exc}") from exc
    bindings = load_dynamic_trace_bindings()
    root = bindings.trace_pb2.Trace()
    try:
        root.ParseFromString(raw)
    except MemoryError as exc:
        raise RuntimeError(f"NVBit root protobuf exhausted memory at {label}") from exc
    except Exception as exc:
        raise RuntimeError(f"NVBit root protobuf is unparseable at {label}: {exc}") from exc
    trace_contract_id = str(root.trace_contract_id)
    if trace_contract_id != _TIMING_TRACE_CONTRACT_ID:
        raise RuntimeError(
            "timing NVBit trace contract mismatch: "
            f"got {trace_contract_id!r}, required {_TIMING_TRACE_CONTRACT_ID!r}"
        )
    return trace_contract_id


def _trace_source_generation(
    *,
    admission: Any,
    trace_contract_id: str,
    source_files: Sequence[tuple[str, immutable.MetadataSnapshot]],
    limits: trace_admission_contract.TraceAdmissionLimits,
) -> dict[str, Any]:
    model = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_retained_source_generation",
        "trace_contract_id": trace_contract_id,
        "trace_admission_sha256": admission.trace_admission_sha256,
        "trace_filesystem_generation_sha256": (
            admission.trace_filesystem_generation_sha256
        ),
        "bounded_inventory": {
            "source_file_count": len(source_files),
            "source_total_bytes": sum(metadata[4] for _name, metadata in source_files),
            "max_tree_files": limits.tree_files,
            "max_tree_directories": limits.tree_directories,
            "max_trace_root_bytes": limits.trace_root_bytes,
            "max_stats_bytes": limits.stats_bytes,
            "max_threadblock_files": limits.threadblock_files,
            "max_threadblock_bytes": limits.threadblock_bytes,
            "max_total_threadblock_bytes": limits.total_threadblock_bytes,
        },
    }
    return immutable.seal_model(model, field="source_generation_sha256")


def _snapshot_timing_trace_source(
    trace_dir: Path,
    *,
    limits: trace_admission_contract.TraceAdmissionLimits = _TIMING_TRACE_LIMITS,
) -> dict[str, Any]:
    with immutable.PinnedDirectoryReader(trace_dir) as reader:
        with reader.retain_tree_generation(
            **_timing_tree_generation_limits(limits)
        ) as tree:
            source_files, source_directories = _timing_trace_source_inventory(
                tree,
                limits=limits,
            )
            snapshots = {}
            for name, metadata in source_files:
                max_bytes = (
                    limits.trace_root_bytes
                    if name == "dynamic_trace.pb"
                    else limits.stats_bytes
                    if name == "stats.csv"
                    else limits.allocations_sidecar_bytes
                    if name == "extra_info/allocations.json"
                    else limits.threadblock_bytes
                )
                snapshots[name] = reader.read_snapshot(
                    name,
                    max_bytes=max_bytes,
                    expected_metadata=metadata,
                )
            threadblocks = tuple(
                (name, snapshots[name].payload)
                for name, _metadata in source_files
                if name.startswith("threadblocks/")
            )
            for relative_path, payload in threadblocks:
                _preflight_threadblock_payload(payload, label=relative_path)
            dynamic_trace_payload = snapshots["dynamic_trace.pb"].payload
            trace_contract_id = _require_timing_trace_contract_id(
                dynamic_trace_payload,
                label="dynamic_trace.pb",
            )
            stats_payload = snapshots["stats.csv"].payload
            _read_stats_row_bytes(stats_payload)
            admission = trace_admission_contract.require_trace_admission(
                trace_dir,
                limits=limits,
            )
            if (
                admission.launch_count != 1
                or len(admission.launches) != 1
                or admission.launches[0].launch_index != 0
            ):
                raise RuntimeError(
                    "timing NVBit trace must carry exactly one authenticated launch"
                )
            if admission.stats_row_count != 1:
                raise RuntimeError(
                    "timing NVBit stats.csv must carry exactly one data row"
                )
            if admission.dynamic_trace_sha256 != immutable.sha256_bytes(
                dynamic_trace_payload
            ):
                raise RuntimeError(
                    "trace admission root hash differs from retained timing snapshot"
                )
            if admission.stats_sha256 != immutable.sha256_bytes(stats_payload):
                raise RuntimeError(
                    "trace admission stats hash differs from retained timing snapshot"
                )
            source_generation = _trace_source_generation(
                admission=admission,
                trace_contract_id=trace_contract_id,
                source_files=source_files,
                limits=limits,
            )
    return {
        "admission": admission,
        "dynamic_trace_payload": dynamic_trace_payload,
        "stats_payload": stats_payload,
        "threadblocks": threadblocks,
        "source_files": source_files,
        "source_directories": source_directories,
        "trace_contract_id": trace_contract_id,
        "source_generation": source_generation,
        "limits": limits,
    }


def _verify_timing_trace_source(
    trace_dir: Path, source_snapshot: dict[str, Any],
) -> None:
    expected_files = source_snapshot["source_files"]
    expected_directories = source_snapshot["source_directories"]
    limits = source_snapshot.get("limits", _TIMING_TRACE_LIMITS)
    with immutable.PinnedDirectoryReader(trace_dir) as reader:
        with reader.retain_tree_generation(
            **_timing_tree_generation_limits(limits)
        ) as tree:
            observed_files, observed_directories = _timing_trace_source_inventory(
                tree,
                limits=limits,
            )
            if (
                observed_files != expected_files
                or observed_directories != expected_directories
            ):
                raise RuntimeError("timing NVBit trace source generation changed")
            for name, metadata in expected_files:
                reader.verify_regular(name, metadata)


def _publish_json_summary(path: Path, summary: dict[str, Any]) -> None:
    path = _assert_output_target(path, operation="publish timing JSON artifact")
    immutable.ensure_directory(path.parent)
    immutable.publish_regular_file(
        path.parent,
        path.name,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _observed_summary_path(trace_dir: Path) -> Path:
    return trace_dir / "observed_sector_summary.json"


def _build_observed_sector_summary(
    *,
    cell: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    from tools.nvbit_tracer import import_trace_contract
    from tools.hrd_profiler.reuse_distance_tracker import ReuseDistanceTracker

    proto_payloads = source["threadblocks"]
    threadblock_pb2_mod = import_trace_contract._import_pb_modules()[0]

    sectors: list[int] = []
    per_op_class_counts: dict[int, int] = {}
    per_access_kind_counts: dict[int, int] = {}
    memory_instruction_records = 0
    for _relative_path, raw in proto_payloads:
        tb = threadblock_pb2_mod.threadblock()
        tb.ParseFromString(raw)
        for warp_index in sorted(tb.warps.keys()):
            for inst in tb.warps[warp_index].instructions:
                if not inst.HasField("memory_op_class"):
                    continue
                if not inst.addresses:
                    continue
                and_mask = inst.active_mask & inst.predicate_mask
                if and_mask == 0:
                    continue
                op_class = int(inst.memory_op_class)
                access_kind = (
                    int(inst.memory_op_access_kind)
                    if inst.HasField("memory_op_access_kind")
                    else _ACCESS_KIND_UNSPECIFIED
                )
                before = len(sectors)
                for addr_record in inst.addresses:
                    lane_vector, _reason = import_trace_contract._reconstruct_lane_vector(
                        addr_record,
                        and_mask,
                    )
                    if lane_vector is None:
                        continue
                    data_width = (
                        addr_record.data_width
                        if addr_record.data_width
                        else inst.data_width
                    )
                    if data_width <= 0:
                        continue
                    for lane_index, lane_addr in enumerate(lane_vector):
                        if not ((and_mask >> lane_index) & 1):
                            continue
                        if lane_addr == import_trace_contract.LANE_INACTIVE_SENTINEL:
                            continue
                        access_start = int(lane_addr)
                        access_end = access_start + int(data_width)
                        sectors.extend(
                            range(access_start // 32, (access_end - 1) // 32 + 1)
                        )
                added = len(sectors) - before
                if added:
                    memory_instruction_records += 1
                    per_op_class_counts[op_class] = (
                        per_op_class_counts.get(op_class, 0) + added
                    )
                    per_access_kind_counts[access_kind] = (
                        per_access_kind_counts.get(access_kind, 0) + added
                    )

    if not sectors:
        raise RuntimeError(f"observed sector trace has zero accesses for {cell['shape_point']}")

    tracker = ReuseDistanceTracker(max_accesses=len(sectors))
    sectors_per_l2_slice = max(1, L2_CAPACITY_BYTES // L2_SLICE_COUNT // SECTOR_SIZE_BYTES)
    cold_count = 0
    finite_miss_count = 0
    finite_rds: list[int] = []
    for timestamp, sector_id in enumerate(sectors, start=1):
        rd, state = tracker.access(sector_id, timestamp)
        if state == "cold":
            cold_count += 1
            continue
        if state == "finite":
            if rd is not None:
                finite_rds.append(int(rd))
            if rd is None or rd >= sectors_per_l2_slice:
                finite_miss_count += 1

    unique_sector_count = len(set(sectors))
    sector_frequency: dict[int, int] = {}
    for sector_id in sectors:
        sector_frequency[sector_id] = sector_frequency.get(sector_id, 0) + 1
    working_set_sector_count = sum(1 for count in sector_frequency.values() if count >= 2)
    finite_rds.sort()
    observed_access_count = len(sectors)
    observed_hit_count = observed_access_count - cold_count - finite_miss_count
    stats_row = _read_stats_row_bytes(source["stats_payload"])

    summary = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_observed_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": OBSERVED_SECTOR_PRODUCER,
        "trace_window": {
            "instruction_trace_mode": OBSERVED_TRACE_MODE,
            **_trace_window_for(cell),
            "window_role": "per_cell_observed_global_memory_window",
        },
        "observation_basis": {
            "trace_contract_ok": True,
            "trace_contract_id": source["trace_contract_id"],
            "trace_source_generation": source["source_generation"],
            "threadblock_proto_file_count": len(proto_payloads),
            "threadblock_proto_set_sha256": _sha256_of_threadblock_snapshots(
                proto_payloads
            ),
            "memory_instruction_record_count": memory_instruction_records,
            "observed_access_count": observed_access_count,
            "observed_hit_count": observed_hit_count,
            "observed_miss_count": cold_count + finite_miss_count,
            "stats_total_insts": int(stats_row.get("total_insts", "0") or 0),
            "stats_total_reported_insts": int(
                stats_row.get("total_reported_insts", "0") or 0
            ),
        },
        "summarize_launch": {
            "cold_count": cold_count,
            "finite_miss_count": finite_miss_count,
            "unique_sector_count": unique_sector_count,
            "working_set_sector_count": working_set_sector_count,
            "reuse_distance_finite_p50": (
                _nearest_rank_percentile(finite_rds, 50.0) if finite_rds else None
            ),
            "reuse_distance_finite_p90": (
                _nearest_rank_percentile(finite_rds, 90.0) if finite_rds else None
            ),
            "reuse_distance_finite_p99": (
                _nearest_rank_percentile(finite_rds, 99.0) if finite_rds else None
            ),
            "per_op_class_counts": {str(k): v for k, v in per_op_class_counts.items()},
            "per_access_kind_counts": {
                str(k): v for k, v in per_access_kind_counts.items()
            },
            "load_count": per_access_kind_counts.get(_ACCESS_KIND_LOAD, 0),
            "store_count": per_access_kind_counts.get(_ACCESS_KIND_STORE, 0),
            "atomic_count": per_access_kind_counts.get(_ACCESS_KIND_ATOMIC, 0),
            "async_copy_count": per_access_kind_counts.get(_ACCESS_KIND_ASYNC_COPY, 0),
            "unspecified_kind_count": per_access_kind_counts.get(
                _ACCESS_KIND_UNSPECIFIED, 0
            ),
        },
    }
    return summary


def _summarize_observed_sector_trace(
    *,
    cell: dict[str, Any],
    trace_root: Path,
    source_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_dir = _trace_dir_for(trace_root, cell)
    source = source_snapshot or _snapshot_timing_trace_source(trace_dir)
    summary = _build_observed_sector_summary(cell=cell, source=source)
    _publish_json_summary(_observed_summary_path(trace_dir), summary)
    return summary


def _derive_sector_counts_from_observation(observed_summary: dict[str, Any]) -> dict[str, int]:
    summary = observed_summary["summarize_launch"]
    basis = observed_summary["observation_basis"]
    per_op = {int(k): int(v) for k, v in summary["per_op_class_counts"].items()}
    per_kind = {int(k): int(v) for k, v in summary["per_access_kind_counts"].items()}
    total_sectors = int(basis["observed_access_count"])
    global_accesses = per_op.get(_MEMORY_OP_CLASS_GLOBAL)
    if global_accesses is None or global_accesses != total_sectors:
        raise RuntimeError(
            "observed sector summary is not an all-global sector stream: "
            f"global={global_accesses!r}; total={total_sectors}"
        )
    global_writes = min(global_accesses, per_kind.get(_ACCESS_KIND_STORE, 0))
    global_reads = min(
        global_accesses - global_writes,
        per_kind.get(_ACCESS_KIND_LOAD, 0),
    )
    total_bytes = total_sectors * SECTOR_SIZE_BYTES
    return {
        "global_read_32b_sectors": int(global_reads),
        "global_write_32b_sectors": int(global_writes),
        "total_32b_sectors": total_sectors,
        "total_128b_lines": _ceil_div(total_bytes, 128),
        "estimated_bytes": total_bytes,
    }


def _require_timing_capture_attempt(
    trace_dir: Path,
    *,
    shape_point: str,
    binary_sha256: str,
    tracer_sha256: str,
    expected_attempt_seal_sha256: str,
) -> dict[str, str]:
    with immutable.PinnedDirectoryReader(trace_dir) as reader:
        with reader.retain_tree_generation(
            **_timing_tree_generation_limits(_TIMING_TRACE_LIMITS)
        ):
            attempt_payload = reader.read(
                _TIMING_ATTEMPT_NAME,
                max_bytes=1024 * 1024,
            )
            start_payload = reader.read(
                _TIMING_START_NAME,
                max_bytes=1024 * 1024,
            )
            tracer_payload = reader.read(
                _TIMING_RETAINED_TRACER_NAME,
                max_bytes=_TIMING_TRACE_LIMITS.trace_root_bytes,
            )
            stdout_payload = reader.read(
                _TIMING_STDOUT_NAME,
                max_bytes=_TIMING_TRACE_LIMITS.stats_bytes,
            )
            stderr_payload = reader.read(
                _TIMING_STDERR_NAME,
                max_bytes=_TIMING_TRACE_LIMITS.stats_bytes,
            )

    attempt = immutable.strict_json_loads(
        attempt_payload,
        label="timing NVBit launch attempt",
    )
    if not isinstance(attempt, dict):
        raise RuntimeError("timing NVBit launch attempt is not an object")
    immutable.verify_model_seal(attempt, field="attempt_seal_sha256")
    if (
        attempt.get("artifact_kind") != "timing_nvbit_launch_attempt"
        or attempt.get("status") != "succeeded"
        or attempt.get("shape_point") != shape_point
        or attempt.get("attempt_seal_sha256") != expected_attempt_seal_sha256
        or attempt.get("target_binary", {}).get("sha256") != binary_sha256
        or attempt.get("tracer", {}).get("sha256") != tracer_sha256
    ):
        raise RuntimeError("timing NVBit launch attempt identity mismatch")

    source_generation = attempt.get("trace_source_generation")
    if not isinstance(source_generation, dict):
        raise RuntimeError(
            "successful timing NVBit launch attempt lacks a retained source generation"
        )
    immutable.verify_model_seal(
        source_generation,
        field="source_generation_sha256",
    )
    if (
        source_generation.get("artifact_kind")
        != "timing_nvbit_retained_source_generation"
        or source_generation.get("trace_contract_id") != _TIMING_TRACE_CONTRACT_ID
    ):
        raise RuntimeError("timing NVBit retained source generation identity mismatch")

    start = immutable.strict_json_loads(
        start_payload,
        label="timing NVBit launch start",
    )
    if not isinstance(start, dict):
        raise RuntimeError("timing NVBit launch start is not an object")
    immutable.verify_model_seal(start, field="start_seal_sha256")
    if start.get("start_seal_sha256") != attempt.get("start_seal_sha256"):
        raise RuntimeError("timing NVBit launch start seal mismatch")
    if start.get("descriptor_bindings") != {
        "command_0": "target_binary",
        "CUDA_INJECTION64_PATH": "tracer",
        "TRACES_FOLDER": "trace_directory",
    }:
        raise RuntimeError("timing NVBit launch descriptor binding mismatch")
    if immutable.sha256_bytes(tracer_payload) != tracer_sha256:
        raise RuntimeError("timing NVBit retained tracer snapshot mismatch")

    for stream_name, expected_name, stream_payload in (
        ("stdout", _TIMING_STDOUT_NAME, stdout_payload),
        ("stderr", _TIMING_STDERR_NAME, stderr_payload),
    ):
        stream = attempt.get("streams", {}).get(stream_name, {})
        if stream.get("path") != expected_name:
            raise RuntimeError(f"timing NVBit {stream_name} path mismatch")
        if (
            stream.get("sha256") != immutable.sha256_bytes(stream_payload)
            or stream.get("size_bytes") != len(stream_payload)
        ):
            raise RuntimeError(f"timing NVBit {stream_name} identity mismatch")

    attempt_path = trace_dir / _TIMING_ATTEMPT_NAME
    start_path = trace_dir / _TIMING_START_NAME
    return {
        "path": _repo_rel(attempt_path),
        "sha256": immutable.sha256_bytes(attempt_payload),
        "attempt_seal_sha256": expected_attempt_seal_sha256,
        "start_path": _repo_rel(start_path),
        "start_sha256": immutable.sha256_bytes(start_payload),
        "start_seal_sha256": str(start["start_seal_sha256"]),
        "trace_source_generation": source_generation,
    }


def build_sector_summary(
    cell: dict[str, Any],
    *,
    regs_per_thread: int,
    static_work_path: Path,
    static_features_path: Path,
    trace_root: Path,
    binary_sha256: str,
    tracer_build_sha256: str,
    nvbit_launch_attempt_seal_sha256: str,
) -> dict[str, Any]:
    _require_shape_point(cell)
    trace_root = _assert_output_directory(
        trace_root,
        operation="publish timing NVBit derived summaries",
    )
    trace_dir = _trace_dir_for(trace_root, cell)
    window = _trace_window_for(cell)
    trace_pb = trace_dir / "dynamic_trace.pb"
    capture_attempt = _require_timing_capture_attempt(
        trace_dir,
        shape_point=cell["shape_point"],
        binary_sha256=binary_sha256,
        tracer_sha256=tracer_build_sha256,
        expected_attempt_seal_sha256=nvbit_launch_attempt_seal_sha256,
    )
    source_snapshot = _snapshot_timing_trace_source(trace_dir)
    if (
        capture_attempt["trace_source_generation"]
        != source_snapshot["source_generation"]
    ):
        raise RuntimeError(
            "timing NVBit launch attempt and retained trace source generation differ"
        )
    with trace_admission_contract.open_admitted_trace_sources(
        trace_dir,
        expected_admission=source_snapshot["admission"],
        limits=source_snapshot["limits"],
    ) as retained_sources:
        retained_sources.verify_unchanged()
        launch_summary = _emit_nvbit_launch_summary(
            cell=cell,
            trace_dir=trace_dir,
            admission=source_snapshot["admission"],
            stats_payload=source_snapshot["stats_payload"],
            binary_sha256=binary_sha256,
            tracer_build_sha256=tracer_build_sha256,
            source_generation=source_snapshot["source_generation"],
        )
        observed_summary = _summarize_observed_sector_trace(
            cell=cell,
            trace_root=trace_root,
            source_snapshot=source_snapshot,
        )
        retained_sources.verify_unchanged()
    _verify_timing_trace_source(trace_dir, source_snapshot)
    observed_payload = (
        json.dumps(observed_summary, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    observed_path = _observed_summary_path(trace_dir)
    return {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": binary_sha256,
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": "nvbit_observed_summarize_launch_sector_summary_v1",
        "source_trace": {
            "nvbit_trace_pb_path": _repo_rel(trace_pb),
            "nvbit_trace_pb_sha256": immutable.sha256_bytes(
                source_snapshot["dynamic_trace_payload"]
            ),
            "trace_dir": _repo_rel(trace_dir),
            "capture_mode": "per_cell_same_binary_same_shape_nvbit_bounded_observed_threadblock_trace",
            "instruction_trace_mode": OBSERVED_TRACE_MODE,
            "instr_begin": window["instr_begin"],
            "instr_end": window["instr_end"],
            "trace_contract_id": source_snapshot["trace_contract_id"],
            "trace_source_generation": source_snapshot["source_generation"],
            "threadblock_proto_file_count": observed_summary["observation_basis"][
                "threadblock_proto_file_count"
            ],
            "threadblock_proto_set_sha256": observed_summary["observation_basis"][
                "threadblock_proto_set_sha256"
            ],
            "threadblock_proto_retention": "retained_raw_threadblock_protos_after_digest",
            "nvbit_launch_attempt": capture_attempt,
        },
        "nvbit_launch_summary": launch_summary,
        "observed_sector_summary": {
            "path": _repo_rel(observed_path),
            "sha256": immutable.sha256_bytes(observed_payload),
            "artifact_kind": "timing_nvbit_observed_sector_summary",
        },
        "summarize_launch": observed_summary["summarize_launch"],
        "observation_basis": observed_summary["observation_basis"],
        "static_inputs": {
            "static_sass_features_path": _repo_rel(static_features_path),
            "static_sass_features_sha256": sha256_file(static_features_path),
            "static_work_path": _repo_rel(static_work_path),
            "static_work_sha256": sha256_file(static_work_path),
        },
        "sector_counts": _derive_sector_counts_from_observation(observed_summary),
        "sector_count_derivation": {
            "source": "nvbit_observed_threadblock_proto_sector_stream",
            "observed_access_count": observed_summary["observation_basis"][
                "observed_access_count"
            ],
            "observed_sector_summary_path": _repo_rel(observed_path),
            "instruction_trace_mode": OBSERVED_TRACE_MODE,
        },
        "theoretical_occupancy": _occupancy_from_launch(cell, regs_per_thread),
        "deploy_legal": True,
    }


def _feature_ref(path: Path, artifact_kind: str) -> dict[str, str]:
    return {
        "path": _repo_rel(path),
        "sha256": sha256_file(path),
        "artifact_kind": artifact_kind,
    }


def _launch_config_for_cell(cell: dict[str, Any]) -> dict[str, Any]:
    geom = cell["launch_geometry"]
    return {
        "kernel_name": geom["kernel_name"],
        "grid_x": int(geom["grid_x"]),
        "grid_y": int(geom["grid_y"]),
        "grid_z": int(geom["grid_z"]),
        "block_x": int(geom["block_x"]),
        "block_y": int(geom["block_y"]),
        "block_z": int(geom["block_z"]),
        "shared_mem_static_bytes": int(geom["shared_mem_static_bytes"]),
        "shared_mem_dynamic_bytes": int(geom["shared_mem_dynamic_bytes"]),
        "regs_per_thread": int(cell["regs_per_thread"]),
    }


def _static_paths_for_cell(feature_dir: Path, cell: dict[str, Any]) -> tuple[Path, Path]:
    stem = _require_shape_point(cell)
    return (
        feature_dir / f"{stem}_static_sass_features.json",
        feature_dir / f"{stem}_static_work.json",
    )


def _emit_static_artifacts_for_cell(
    *,
    feature_dir: Path,
    cell: dict[str, Any],
    source_commit: str,
) -> tuple[Path, Path]:
    from workloads.simbench_bw.static_sass_features import extract_records

    feature_dir = _assert_output_directory(
        feature_dir,
        operation="publish timing static feature artifacts",
    )
    immutable.ensure_directory(feature_dir)
    features_path, work_path = _static_paths_for_cell(feature_dir, cell)
    _require_absent_output_target(
        features_path,
        operation="publish timing static SASS features",
    )
    _require_absent_output_target(
        work_path,
        operation="publish timing static work",
    )
    feature_record, work_record = extract_records(
        binary=_binary_path_for(cell),
        slug=cell["slug"],
        launch_config=_launch_config_for_cell(cell),
        source_commit=source_commit,
        kernel=cell["launch_geometry"]["kernel_name"],
        arch=cell["arch"],
    )
    _publish_json_summary(features_path, feature_record)
    _publish_json_summary(work_path, work_record)
    return features_path, work_path


def _artifact_ref(path: Path) -> dict[str, str]:
    return {"path": _repo_rel(path), "sha256": sha256_file(path)}


def _repo_rel_or_path(path: Path) -> str:
    try:
        return _repo_rel(path)
    except ValueError:
        return str(path)


def _sector_identity_mismatch(
    field: str,
    *,
    expected: Any,
    observed: Any,
) -> None:
    raise RuntimeError(
        "foreign NVBit sector summary identity mismatch: "
        f"{field}: expected={expected!r}; observed={observed!r}"
    )


def _validate_sector_summary_schema(summary: dict[str, Any]) -> None:
    with immutable.PinnedDirectoryReader(_SECTOR_SUMMARY_SCHEMA.parent) as reader:
        schema_payload = reader.read(
            _SECTOR_SUMMARY_SCHEMA.name,
            max_bytes=1024 * 1024,
        )
    schema = immutable.strict_json_loads(
        schema_payload,
        label="authoritative timing NVBit sector summary schema",
    )
    if not isinstance(schema, dict):
        raise RuntimeError(
            "authoritative timing NVBit sector summary schema is not an object"
        )
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(summary),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except jsonschema.SchemaError as exc:
        raise RuntimeError(
            "authoritative timing NVBit sector summary schema is invalid"
        ) from exc
    if errors:
        error = errors[0]
        location = "$"
        for part in error.absolute_path:
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        raise RuntimeError(
            "timing NVBit sector summary violates authoritative schema at "
            f"{location}: {error.message}"
        )


def _require_exact_json_binding(
    field: str,
    *,
    expected: Any,
    observed: Any,
) -> None:
    try:
        expected_bytes = immutable.canonical_json_bytes(expected)
        observed_bytes = immutable.canonical_json_bytes(observed)
        matches = expected_bytes == observed_bytes
    except (TypeError, ValueError):
        matches = False
    if not matches:
        _sector_identity_mismatch(field, expected=expected, observed=observed)


def _require_sector_summary_identity(
    summary: dict[str, Any],
    *,
    cell: dict[str, Any],
    trace_root: Path,
) -> None:
    expected_top_level = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": _require_shape_point(cell),
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": "nvbit_observed_summarize_launch_sector_summary_v1",
        "deploy_legal": True,
    }
    for field, expected in expected_top_level.items():
        observed = summary.get(field)
        if observed != expected or type(observed) is not type(expected):
            _sector_identity_mismatch(
                field,
                expected=expected,
                observed=observed,
            )

    launch_summary = summary.get("nvbit_launch_summary")
    if not isinstance(launch_summary, dict):
        _sector_identity_mismatch(
            "nvbit_launch_summary",
            expected="object",
            observed=launch_summary,
        )
    expected_kernel_name = cell["launch_geometry"]["kernel_name"]
    launch_identity = {
        "harmonized_kernel_name": expected_kernel_name,
        "harmonized_kernel_signature_sha256": cell["kernel_signature_sha256"],
        "launch_index": 0,
        "repeat_index": 0,
        "instruction_trace_mode": OBSERVED_TRACE_MODE,
    }
    for field, expected in launch_identity.items():
        observed = launch_summary.get(field)
        if observed != expected or type(observed) is not type(expected):
            _sector_identity_mismatch(
                f"nvbit_launch_summary.{field}",
                expected=expected,
                observed=observed,
            )
    launch_geometry = launch_summary.get("launch_geometry")
    if not isinstance(launch_geometry, dict):
        _sector_identity_mismatch(
            "nvbit_launch_summary.launch_geometry",
            expected="object",
            observed=launch_geometry,
        )
    observed_kernel_name = launch_geometry.get("kernel_name")
    normalized_kernel_name = (
        observed_kernel_name[:-4]
        if isinstance(observed_kernel_name, str)
        and observed_kernel_name.endswith("___0")
        else observed_kernel_name
    )
    if normalized_kernel_name != expected_kernel_name:
        _sector_identity_mismatch(
            "nvbit_launch_summary.launch_geometry.kernel_name",
            expected=expected_kernel_name,
            observed=observed_kernel_name,
        )
    for field in _LAUNCH_GEOMETRY_FIELDS:
        expected = int(cell["launch_geometry"][field])
        observed = launch_geometry.get(field)
        if type(observed) is not int or observed != expected:
            _sector_identity_mismatch(
                f"nvbit_launch_summary.launch_geometry.{field}",
                expected=expected,
                observed=observed,
            )

    trace_dir = _trace_dir_for(trace_root, cell)
    source_trace = summary.get("source_trace")
    if not isinstance(source_trace, dict):
        _sector_identity_mismatch(
            "source_trace",
            expected="object",
            observed=source_trace,
        )
    expected_trace_identity = {
        "trace_dir": _repo_rel(trace_dir),
        "nvbit_trace_pb_path": _repo_rel(trace_dir / "dynamic_trace.pb"),
        "capture_mode": (
            "per_cell_same_binary_same_shape_nvbit_bounded_observed_threadblock_trace"
        ),
        "instruction_trace_mode": OBSERVED_TRACE_MODE,
        "instr_begin": _trace_window_for(cell)["instr_begin"],
        "instr_end": _trace_window_for(cell)["instr_end"],
        "trace_contract_id": _TIMING_TRACE_CONTRACT_ID,
    }
    for field, expected in expected_trace_identity.items():
        observed = source_trace.get(field)
        if observed != expected or type(observed) is not type(expected):
            _sector_identity_mismatch(
                f"source_trace.{field}",
                expected=expected,
                observed=observed,
            )

    observed_summary = summary.get("observed_sector_summary")
    if not isinstance(observed_summary, dict):
        _sector_identity_mismatch(
            "observed_sector_summary",
            expected="object",
            observed=observed_summary,
        )
    expected_observed_identity = {
        "path": _repo_rel(_observed_summary_path(trace_dir)),
        "artifact_kind": "timing_nvbit_observed_sector_summary",
    }
    for field, expected in expected_observed_identity.items():
        observed = observed_summary.get(field)
        if observed != expected or type(observed) is not type(expected):
            _sector_identity_mismatch(
                f"observed_sector_summary.{field}",
                expected=expected,
                observed=observed,
            )

    attempt_ref = source_trace.get("nvbit_launch_attempt")
    if not isinstance(attempt_ref, dict):
        _sector_identity_mismatch(
            "source_trace.nvbit_launch_attempt",
            expected="object",
            observed=attempt_ref,
        )
    attempt_path = trace_dir / _TIMING_ATTEMPT_NAME
    with immutable.PinnedDirectoryReader(trace_dir) as trace_reader:
        attempt_payload = trace_reader.read(
            _TIMING_ATTEMPT_NAME,
            max_bytes=1024 * 1024,
        )
    attempt = immutable.strict_json_loads(
        attempt_payload,
        label="timing NVBit launch attempt",
    )
    if not isinstance(attempt, dict):
        raise RuntimeError("timing NVBit launch attempt is not an object")
    immutable.verify_model_seal(attempt, field="attempt_seal_sha256")
    tracer_sha256 = attempt.get("tracer", {}).get("sha256")
    if type(tracer_sha256) is not str:
        raise RuntimeError("timing NVBit launch attempt tracer identity is missing")
    expected_attempt_seal = attempt_ref.get("attempt_seal_sha256")
    if type(expected_attempt_seal) is not str:
        _sector_identity_mismatch(
            "source_trace.nvbit_launch_attempt.attempt_seal_sha256",
            expected="lowercase 64-hex capture seal",
            observed=expected_attempt_seal,
        )
    verified_attempt = _require_timing_capture_attempt(
        trace_dir,
        shape_point=cell["shape_point"],
        binary_sha256=cell["binary_sha256"],
        tracer_sha256=tracer_sha256,
        expected_attempt_seal_sha256=expected_attempt_seal,
    )
    if attempt_ref != verified_attempt:
        _sector_identity_mismatch(
            "source_trace.nvbit_launch_attempt",
            expected=verified_attempt,
            observed=attempt_ref,
        )
    source_generation = source_trace.get("trace_source_generation")
    if source_generation != verified_attempt["trace_source_generation"]:
        _sector_identity_mismatch(
            "source_trace.trace_source_generation",
            expected=verified_attempt["trace_source_generation"],
            observed=source_generation,
        )
    observation_basis = summary["observation_basis"]
    for container_name, container in (
        ("nvbit_launch_summary", launch_summary),
        ("observation_basis", observation_basis),
    ):
        if container.get("trace_contract_id") != source_trace["trace_contract_id"]:
            _sector_identity_mismatch(
                f"{container_name}.trace_contract_id",
                expected=source_trace["trace_contract_id"],
                observed=container.get("trace_contract_id"),
            )
        _require_exact_json_binding(
            f"{container_name}.trace_source_generation",
            expected=source_generation,
            observed=container.get("trace_source_generation"),
        )
    for field in (
        "threadblock_proto_file_count",
        "threadblock_proto_set_sha256",
    ):
        if source_trace[field] != observation_basis[field]:
            _sector_identity_mismatch(
                f"source_trace.{field}",
                expected=observation_basis[field],
                observed=source_trace[field],
            )
    if attempt_ref.get("path") != _repo_rel(attempt_path):
        _sector_identity_mismatch(
            "source_trace.nvbit_launch_attempt.path",
            expected=_repo_rel(attempt_path),
            observed=attempt_ref.get("path"),
        )


def _require_observed_summary_semantics(observed: dict[str, Any]) -> None:
    basis = observed["observation_basis"]
    launch = observed["summarize_launch"]
    access_count = int(basis["observed_access_count"])
    hit_count = int(basis["observed_hit_count"])
    miss_count = int(basis["observed_miss_count"])
    if hit_count + miss_count != access_count:
        _sector_identity_mismatch(
            "observation_basis.observed_hit_count+observed_miss_count",
            expected=access_count,
            observed=hit_count + miss_count,
        )
    summarized_misses = int(launch["cold_count"]) + int(
        launch["finite_miss_count"]
    )
    if summarized_misses != miss_count:
        _sector_identity_mismatch(
            "summarize_launch.cold_count+finite_miss_count",
            expected=miss_count,
            observed=summarized_misses,
        )

    per_op = {
        int(key): int(value)
        for key, value in launch["per_op_class_counts"].items()
    }
    per_kind = {
        int(key): int(value)
        for key, value in launch["per_access_kind_counts"].items()
    }
    if sum(per_op.values()) != access_count:
        _sector_identity_mismatch(
            "summarize_launch.per_op_class_counts",
            expected=f"counts summing to {access_count}",
            observed=launch["per_op_class_counts"],
        )
    if sum(per_kind.values()) != access_count:
        _sector_identity_mismatch(
            "summarize_launch.per_access_kind_counts",
            expected=f"counts summing to {access_count}",
            observed=launch["per_access_kind_counts"],
        )
    scalar_kind_counts = {
        "load_count": _ACCESS_KIND_LOAD,
        "store_count": _ACCESS_KIND_STORE,
        "atomic_count": _ACCESS_KIND_ATOMIC,
        "async_copy_count": _ACCESS_KIND_ASYNC_COPY,
        "unspecified_kind_count": _ACCESS_KIND_UNSPECIFIED,
    }
    for field, access_kind in scalar_kind_counts.items():
        expected = per_kind.get(access_kind, 0)
        if launch[field] != expected:
            _sector_identity_mismatch(
                f"summarize_launch.{field}",
                expected=expected,
                observed=launch[field],
            )
    if int(launch["unique_sector_count"]) > access_count:
        _sector_identity_mismatch(
            "summarize_launch.unique_sector_count",
            expected=f"at most {access_count}",
            observed=launch["unique_sector_count"],
        )
    if int(launch["working_set_sector_count"]) > int(
        launch["unique_sector_count"]
    ):
        _sector_identity_mismatch(
            "summarize_launch.working_set_sector_count",
            expected=f"at most {launch['unique_sector_count']}",
            observed=launch["working_set_sector_count"],
        )


def _require_observed_summary_binding(
    summary: dict[str, Any],
    *,
    cell: dict[str, Any],
    trace_root: Path,
) -> dict[str, Any]:
    trace_dir = _trace_dir_for(trace_root, cell)
    observed_ref = summary["observed_sector_summary"]
    with immutable.PinnedDirectoryReader(trace_dir) as reader:
        observed_payload = reader.read(
            _observed_summary_path(trace_dir).name,
            max_bytes=_OBSERVED_SECTOR_SUMMARY_MAX_BYTES,
        )
    observed_sha256 = immutable.sha256_bytes(observed_payload)
    if observed_ref["sha256"] != observed_sha256:
        _sector_identity_mismatch(
            "observed_sector_summary.sha256",
            expected=observed_sha256,
            observed=observed_ref["sha256"],
        )
    observed = immutable.strict_json_loads(
        observed_payload,
        label="referenced timing NVBit observed sector summary",
    )
    if not isinstance(observed, dict):
        raise RuntimeError(
            "referenced timing NVBit observed sector summary is not an object"
        )
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "arch",
        "slug",
        "shape_point",
        "binary_sha256",
        "kernel_signature_sha256",
        "source_commit",
        "producer",
        "trace_window",
        "observation_basis",
        "summarize_launch",
    }
    if set(observed) != expected_keys:
        _sector_identity_mismatch(
            "observed_sector_summary.keys",
            expected=sorted(expected_keys),
            observed=sorted(observed),
        )
    expected_identity = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_observed_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": _require_shape_point(cell),
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": OBSERVED_SECTOR_PRODUCER,
    }
    for field, expected in expected_identity.items():
        value = observed.get(field)
        if value != expected or type(value) is not type(expected):
            _sector_identity_mismatch(
                f"observed_sector_summary.{field}",
                expected=expected,
                observed=value,
            )
    expected_window = {
        "instruction_trace_mode": OBSERVED_TRACE_MODE,
        **_trace_window_for(cell),
        "window_role": "per_cell_observed_global_memory_window",
    }
    _require_exact_json_binding(
        "observed_sector_summary.trace_window",
        expected=expected_window,
        observed=observed.get("trace_window"),
    )

    source_snapshot = _snapshot_timing_trace_source(trace_dir)
    source_trace = summary["source_trace"]
    _require_exact_json_binding(
        "source_trace.trace_source_generation",
        expected=source_snapshot["source_generation"],
        observed=source_trace["trace_source_generation"],
    )
    raw_source_bindings = {
        "nvbit_trace_pb_sha256": immutable.sha256_bytes(
            source_snapshot["dynamic_trace_payload"]
        ),
        "threadblock_proto_file_count": len(source_snapshot["threadblocks"]),
        "threadblock_proto_set_sha256": _sha256_of_threadblock_snapshots(
            source_snapshot["threadblocks"]
        ),
    }
    for field, expected in raw_source_bindings.items():
        if source_trace[field] != expected:
            _sector_identity_mismatch(
                f"source_trace.{field}",
                expected=expected,
                observed=source_trace[field],
            )
    admitted_launch = source_snapshot["admission"].launches[0]
    _require_authenticated_launch_geometry(
        cell=cell,
        admitted_launch=admitted_launch,
        stats_row=_read_stats_row_bytes(source_snapshot["stats_payload"]),
    )
    launch_summary = summary["nvbit_launch_summary"]
    expected_launch_source = {
        "inst_count": int(admitted_launch.inst_count),
        "stats_csv_sha256": immutable.sha256_bytes(source_snapshot["stats_payload"]),
    }
    for field, expected in expected_launch_source.items():
        if launch_summary[field] != expected:
            _sector_identity_mismatch(
                f"nvbit_launch_summary.{field}",
                expected=expected,
                observed=launch_summary[field],
            )
    recomputed_observed = _build_observed_sector_summary(
        cell=cell,
        source=source_snapshot,
    )
    _require_exact_json_binding(
        "observed_sector_summary.contents",
        expected=recomputed_observed,
        observed=observed,
    )
    for field in ("observation_basis", "summarize_launch"):
        _require_exact_json_binding(
            field,
            expected=observed.get(field),
            observed=summary[field],
        )
    _require_observed_summary_semantics(observed)

    derived_counts = _derive_sector_counts_from_observation(observed)
    _require_exact_json_binding(
        "sector_counts",
        expected=derived_counts,
        observed=summary["sector_counts"],
    )
    expected_derivation = {
        "source": "nvbit_observed_threadblock_proto_sector_stream",
        "observed_access_count": observed["observation_basis"][
            "observed_access_count"
        ],
        "observed_sector_summary_path": observed_ref["path"],
        "instruction_trace_mode": OBSERVED_TRACE_MODE,
    }
    _require_exact_json_binding(
        "sector_count_derivation",
        expected=expected_derivation,
        observed=summary["sector_count_derivation"],
    )
    _verify_timing_trace_source(trace_dir, source_snapshot)
    return observed


def _load_current_sector_summary(
    path: Path,
    *,
    cell: dict[str, Any],
    trace_root: Path,
) -> tuple[dict[str, Any], str]:
    with immutable.PinnedDirectoryReader(path.parent) as reader:
        payload = reader.read(path.name, max_bytes=_SECTOR_SUMMARY_MAX_BYTES)
    summary = immutable.strict_json_loads(
        payload,
        label="timing NVBit sector summary",
    )
    if not isinstance(summary, dict):
        raise RuntimeError("timing NVBit sector summary is not an object")
    _validate_sector_summary_schema(summary)
    _require_sector_summary_identity(summary, cell=cell, trace_root=trace_root)
    _require_observed_summary_binding(summary, cell=cell, trace_root=trace_root)
    return summary, immutable.sha256_bytes(payload)


def _head_artifact_ref(
    path: Path,
    *,
    typed_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    try:
        return {
            "path": _repo_rel_or_path(path),
            "sha256": sha256_file(path),
            "artifact_status": "available",
            "typed_unavailable_reason": None,
        }
    except Exception as exc:
        return {
            "path": _repo_rel_or_path(path),
            "sha256": None,
            "artifact_status": "unavailable",
            "typed_unavailable_reason": (
                typed_unavailable_reason or f"head_artifact_unreadable:{exc}"
            ),
        }


def _available_head_artifact_ref(path: Path) -> dict[str, Any]:
    ref = _head_artifact_ref(path)
    if ref["artifact_status"] != "available":
        raise RuntimeError(ref["typed_unavailable_reason"])
    return ref


def _head_training_cells(head: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in head.get("cohort_scope", {}).get("admitted", []):
        out.append(
            {
                "identity": (
                    cell["arch"],
                    cell["slug"],
                    cell["shape_point"],
                ),
                "arch": cell["arch"],
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "family_bucket": cell["family_bucket"],
                "features": dict(cell["features"]),
                "label": float(cell["label"]),
            }
        )
    return out


def _build_head_ensemble(head: dict[str, Any]) -> Any:
    from tools.local_labels.fit_monotonic_calibration_head_v5 import (
        _model_class_factory_for,
    )
    from workloads.simbench_bw import monotonic_calibration_head as mch

    if head.get("fit_status") != "ok":
        raise RuntimeError(f"head fit_status={head.get('fit_status')!r}")
    selected_name = head.get("selected_class", {}).get("name")
    if not selected_name:
        raise RuntimeError("head selected_class.name is missing")
    cells = _head_training_cells(head)
    if not cells:
        raise RuntimeError("head cohort_scope.admitted is empty")
    ensemble = mch.PerBucketCalibratorEnsemble(
        model_class_factory=_model_class_factory_for(selected_name),
    )
    ensemble.fit(cells)
    return ensemble


def _head_route(
    head: dict[str, Any],
    *,
    arch: str,
    family_bucket: str,
) -> dict[str, Any]:
    provenance = head.get("per_bucket_provenance", {})
    by_bucket = provenance.get("by_bucket", {})
    bucket_key = f"{arch}/{family_bucket}"
    pooled_present = bool(provenance.get("pooled_fallback_present"))
    if bucket_key in by_bucket:
        entry = by_bucket[bucket_key]
        return {
            "bucket_key": bucket_key,
            "bucket_in_head_scope": True,
            "kind": entry["kind"],
            "n_train": int(entry["n_train"]),
            "pooled_fallback_present": pooled_present,
        }
    if pooled_present:
        return {
            "bucket_key": bucket_key,
            "bucket_in_head_scope": False,
            "kind": "pooled_fallback_unseen_bucket_zero_offset",
            "n_train": None,
            "pooled_fallback_present": True,
        }
    raise RuntimeError(f"no head route for {bucket_key!r}")


def _ordered_canonical_features(features: dict[str, float]) -> dict[str, float]:
    from workloads.simbench_bw import canonical_features as cf

    missing = [name for name in cf.CANONICAL_FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"canonical feature vector missing {missing!r}")
    return {
        name: float(cf._validate_feature_value(name, features[name]))
        for name in cf.CANONICAL_FEATURE_NAMES
    }


def _raw_observed_hit_fraction(sector: dict[str, Any]) -> float:
    basis = sector["observation_basis"]
    access_count = int(basis["observed_access_count"])
    hit_count = int(basis["observed_hit_count"])
    if access_count <= 0:
        raise RuntimeError("observed sector summary has no accesses")
    if hit_count < 0 or hit_count > access_count:
        raise RuntimeError(
            "observed sector summary has invalid hit/access counts: "
            f"{hit_count}/{access_count}"
        )
    return hit_count / access_count


def _l1_head_features(raw_prediction: float) -> dict[str, float]:
    from workloads.simbench_bw import canonical_features as cf

    features = {name: 0.0 for name in cf.CANONICAL_FEATURE_NAMES}
    features["raw_prediction"] = raw_prediction
    return _ordered_canonical_features(features)


def _l2_head_features(
    sector: dict[str, Any],
    *,
    raw_prediction: float,
) -> dict[str, float]:
    from workloads.simbench_bw import canonical_features as cf

    summary = sector["summarize_launch"]
    counts = sector["sector_counts"]
    launch_geom = sector["nvbit_launch_summary"]["launch_geometry"]
    finite_l2_slice_misses = int(summary["finite_miss_count"])
    if finite_l2_slice_misses != 0:
        raise RuntimeError(
            "observed sector summary lacks a long-RD histogram for nonzero "
            "finite L2-slice misses"
        )
    global_count = (
        int(counts["global_read_32b_sectors"])
        + int(counts["global_write_32b_sectors"])
    )
    grid_volume = (
        int(launch_geom["grid_x"])
        * int(launch_geom["grid_y"])
        * int(launch_geom["grid_z"])
    )
    features = {
        "raw_prediction": raw_prediction,
        "footprint_ratio": cf.compute_footprint_ratio(
            unique_sector_count=int(summary["unique_sector_count"]),
            cache_model_capacity_bytes=L2_CAPACITY_BYTES,
            sector_size_bytes=SECTOR_SIZE_BYTES,
        ),
        "long_rd_tail_mass": 0.0,
        "intra_lane_reuse": 0.0,
        "global_share": cf.compute_global_share(
            global_count=global_count,
            total_count=int(counts["total_32b_sectors"]),
        ),
        "gridDim_density": cf.compute_gridDim_density(
            grid_volume=grid_volume,
            device=sector["arch"],
        ),
    }
    return _ordered_canonical_features(features)


def _typed_unavailable_layer_prediction(
    *,
    layer: str,
    reason: str,
    head_path: Path,
    head: dict[str, Any] | None,
    family_bucket: str | None = None,
    features: dict[str, float] | None = None,
    route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cache_layer": layer,
        "result_kind": "typed_unavailable",
        "predicted_sector_hit_rate": None,
        "typed_unavailable_reason": reason,
        "head_artifact": _head_artifact_ref(
            head_path,
            typed_unavailable_reason=reason,
        ),
        "head_kind": (head or {}).get("head_kind"),
        "head_scope": (head or {}).get("head_scope"),
        "head_source_commit": (head or {}).get("source_commit"),
        "fit_status": (head or {}).get("fit_status"),
        "selected_class": (head or {}).get("selected_class"),
        "family_bucket": family_bucket,
        "route": route,
        "input_features": features,
        "input_feature_sha256": (
            _canonical_json_sha256(features) if features is not None else None
        ),
        "feature_schema_sha256": (
            _canonical_json_sha256((head or {}).get("feature_schema"))
            if (head or {}).get("feature_schema") is not None
            else None
        ),
        "scorer_name": CACHE_HEAD_SCORER_NAME,
    }


def _score_cache_head(
    *,
    layer: str,
    head_path: Path,
    head: dict[str, Any] | None,
    ensemble: Any | None,
    head_error: str | None,
    features: dict[str, float] | None,
    arch: str,
    family_bucket: str,
) -> dict[str, Any]:
    if head is None:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=head_error or "head_artifact_load_failed",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
        )
    if features is None:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=head_error or "canonical_feature_assembly_failed",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
        )
    route: dict[str, Any] | None
    try:
        route = _head_route(head, arch=arch, family_bucket=family_bucket)
    except Exception as exc:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=f"head_route_unavailable:{exc}",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
        )
    if ensemble is None:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=head_error or "head_ensemble_unavailable",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
            route=route,
        )
    try:
        predicted = float(ensemble.predict(arch, family_bucket, features))
    except Exception as exc:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=f"head_scoring_failed:{exc}",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
            route=route,
        )
    try:
        head_artifact = _available_head_artifact_ref(head_path)
    except Exception as exc:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=f"head_artifact_digest_unavailable:{exc}",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
            route=route,
        )
    return {
        "cache_layer": layer,
        "result_kind": "scalar",
        "predicted_sector_hit_rate": max(0.0, min(1.0, predicted)),
        "typed_unavailable_reason": None,
        "head_artifact": head_artifact,
        "head_kind": head.get("head_kind"),
        "head_scope": head.get("head_scope"),
        "head_source_commit": head.get("source_commit"),
        "fit_status": head.get("fit_status"),
        "selected_class": head.get("selected_class"),
        "family_bucket": family_bucket,
        "route": route,
        "input_features": features,
        "input_feature_sha256": _canonical_json_sha256(features),
        "feature_schema_sha256": _canonical_json_sha256(head.get("feature_schema")),
        "scorer_name": CACHE_HEAD_SCORER_NAME,
    }


def _build_cache_hit_predictions(
    *,
    cohort: dict[str, Any],
    output_path: Path,
    trace_root: Path,
    l1_head_path: Path,
    l2_head_path: Path,
    family_buckets_path: Path,
    source_commit: str | None,
) -> dict[str, Any]:
    from tools.local_labels.fit_monotonic_calibration_head_v5 import (
        _load_family_buckets,
        _resolve_family_bucket,
    )

    output_path = _assert_output_target(
        output_path,
        operation="publish timing cache-hit predictions",
    )
    _require_absent_output_target(
        output_path,
        operation="publish timing cache-hit predictions",
    )
    family_to_bucket = _load_family_buckets(family_buckets_path)
    l1_head: dict[str, Any] | None = None
    l2_head: dict[str, Any] | None = None
    l1_ensemble: Any | None = None
    l2_ensemble: Any | None = None
    l1_error: str | None = None
    l2_error: str | None = None
    try:
        l1_head = _load(l1_head_path)
        l1_ensemble = _build_head_ensemble(l1_head)
    except Exception as exc:
        l1_error = str(exc)
    try:
        l2_head = _load(l2_head_path)
        l2_ensemble = _build_head_ensemble(l2_head)
    except Exception as exc:
        l2_error = str(exc)

    cells = []
    for cell in cohort.get("cells", []):
        refs = cell["feature_artifacts"]
        sector_ref = refs["nvbit_sector_summary"]
        if sector_ref.get("artifact_kind") != "timing_nvbit_sector_summary":
            _sector_identity_mismatch(
                "feature_artifacts.nvbit_sector_summary.artifact_kind",
                expected="timing_nvbit_sector_summary",
                observed=sector_ref.get("artifact_kind"),
            )
        sector_path = REPO_ROOT / sector_ref["path"]
        sector, sector_sha256 = _load_current_sector_summary(
            sector_path,
            cell=cell,
            trace_root=trace_root,
        )
        if sector_ref.get("sha256") != sector_sha256:
            _sector_identity_mismatch(
                "feature_artifacts.nvbit_sector_summary.sha256",
                expected=sector_sha256,
                observed=sector_ref.get("sha256"),
            )
        family_bucket = _resolve_family_bucket(cell["slug"], family_to_bucket)
        try:
            raw_prediction = _raw_observed_hit_fraction(sector)
            l1_features: dict[str, float] | None = _l1_head_features(raw_prediction)
            l2_features: dict[str, float] | None = _l2_head_features(
                sector,
                raw_prediction=raw_prediction,
            )
            feature_error: str | None = None
        except Exception as exc:
            l1_features = None
            l2_features = None
            feature_error = str(exc)
        l1_prediction = _score_cache_head(
            layer="L1TEX",
            head_path=l1_head_path,
            head=l1_head,
            ensemble=l1_ensemble,
            head_error=l1_error or feature_error,
            features=l1_features,
            arch=cell["arch"],
            family_bucket=family_bucket,
        )
        l2_prediction = _score_cache_head(
            layer="L2",
            head_path=l2_head_path,
            head=l2_head,
            ensemble=l2_ensemble,
            head_error=l2_error or feature_error,
            features=l2_features,
            arch=cell["arch"],
            family_bucket=family_bucket,
        )
        if (
            l1_prediction["result_kind"] == "scalar"
            and l2_prediction["result_kind"] == "scalar"
        ):
            result_kind = "scalar"
            l1_value = l1_prediction["predicted_sector_hit_rate"]
            l2_value = l2_prediction["predicted_sector_hit_rate"]
            typed_reason = None
        else:
            result_kind = "typed_unavailable"
            l1_value = None
            l2_value = None
            reasons = []
            for name, prediction in (
                ("l1", l1_prediction),
                ("l2", l2_prediction),
            ):
                if prediction["result_kind"] != "scalar":
                    reasons.append(
                        f"{name}_head_unscorable:"
                        f"{prediction['typed_unavailable_reason']}"
                    )
            typed_reason = ";".join(reasons)
        cells.append(
            {
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "arch": cell["arch"],
                "binary_sha256": cell["binary_sha256"],
                "kernel_signature_sha256": cell["kernel_signature_sha256"],
                "source_commit": cell["source_commit"],
                "result_kind": result_kind,
                "l1_sector_hit_rate": l1_value,
                "l2_sector_hit_rate": l2_value,
                "typed_unavailable_reason": typed_reason,
                "input_feature_refs": {
                    "nvbit_sector_summary": refs["nvbit_sector_summary"],
                    "observed_sector_summary": sector["observed_sector_summary"],
                    "static_sass_features": refs["static_sass_features"],
                    "static_work": refs["static_work"],
                },
                "head_provenance": {
                    "observed_sector_summary": sector["observed_sector_summary"],
                    "l1_prediction": l1_prediction,
                    "l2_prediction": l2_prediction,
                    "prediction_method": CACHE_PREDICTION_METHOD,
                    "producer": CACHE_PREDICTION_PRODUCER,
                },
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": "timing_cache_hit_prediction_bundle",
        "arch": cohort["arch"],
        "source_commit": source_commit or _source_commit(),
        "producer": CACHE_PREDICTION_PRODUCER,
        "prediction_scope": "long_duration_timing_cohort",
        "cohort_artifact": _repo_rel(DEFAULT_COHORT),
        "head_artifacts": {
            "l1_head_artifact": _head_artifact_ref(
                l1_head_path,
                typed_unavailable_reason=l1_error,
            ),
            "l2_head_artifact": _head_artifact_ref(
                l2_head_path,
                typed_unavailable_reason=l2_error,
            ),
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    _publish_json_summary(output_path, payload)
    return payload


def attach_feature_artifacts_to_cohort(
    cohort: dict[str, Any],
    *,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    cache_hit_predictions_path: Path = DEFAULT_CACHE_HIT_PREDICTIONS,
    static_work_path: Path = DEFAULT_STATIC_WORK,
    static_features_path: Path = DEFAULT_STATIC_FEATURES,
    tracer_so_path: Path | None = None,
    l1_head_path: Path = DEFAULT_L1_HEAD,
    l2_head_path: Path = DEFAULT_L2_HEAD,
    family_buckets_path: Path = DEFAULT_FAMILY_BUCKETS,
    source_commit: str | None = None,
    capture_nvbit: bool = False,
    capture_missing_nvbit: bool = False,
) -> dict[str, Any]:
    out = json.loads(json.dumps(cohort))
    cells = _validated_cohort_cells(out)
    feature_dir = _assert_output_directory(
        feature_dir,
        operation="publish timing feature artifacts",
    )
    trace_root = _assert_output_directory(
        trace_root,
        operation="publish timing NVBit trace artifacts",
    )
    cache_hit_predictions_path = _assert_output_target(
        cache_hit_predictions_path,
        operation="publish timing cache-hit predictions",
    )
    missing_sector_shapes: set[str] = set()
    for cell in cells:
        sector_path = feature_dir / (
            f"{cell['shape_point']}_nvbit_sector_summary.json"
        )
        if _path_entry_exists(sector_path):
            _load_current_sector_summary(
                sector_path,
                cell=cell,
                trace_root=trace_root,
            )
        else:
            missing_sector_shapes.add(cell["shape_point"])
    capture_shapes = (
        {cell["shape_point"] for cell in cells}
        if capture_nvbit
        else missing_sector_shapes if capture_missing_nvbit
        else set()
    )
    unbacked_missing = sorted(missing_sector_shapes - capture_shapes)
    if unbacked_missing:
        raise RuntimeError(
            "missing NVBit sector summaries require a fresh descriptor-bound "
            f"capture: {unbacked_missing!r}"
        )
    replacement_shapes = sorted(capture_shapes - missing_sector_shapes)
    if replacement_shapes:
        raise RuntimeError(
            "refusing to replace existing NVBit sector summaries: "
            f"{replacement_shapes!r}"
        )
    if capture_shapes:
        if tracer_so_path is None:
            raise RuntimeError(
                "NVBit timing capture requires --tracer-so with an explicit "
                "external non-Git split-capable tracer"
            )
        tracer_so_path = _require_external_tracer(
            tracer_so_path, trace_root=trace_root,
        )
    _require_absent_output_target(
        cache_hit_predictions_path,
        operation="publish timing cache-hit predictions",
    )
    immutable.ensure_directory(feature_dir)
    immutable.ensure_directory(trace_root)
    commit = source_commit or _source_commit()
    for cell in cells:
        cell_static_features_path, cell_static_work_path = _static_paths_for_cell(
            feature_dir, cell
        )
        if (
            capture_nvbit
            or not cell_static_features_path.is_file()
            or not cell_static_work_path.is_file()
        ):
            cell_static_features_path, cell_static_work_path = (
                _emit_static_artifacts_for_cell(
                    feature_dir=feature_dir,
                    cell=cell,
                    source_commit=commit,
                )
            )
        sector_path = feature_dir / f"{cell['shape_point']}_nvbit_sector_summary.json"
        capture_identity = None
        if cell["shape_point"] in capture_shapes:
            assert tracer_so_path is not None
            capture_identity = _capture_nvbit_observed_sector_trace(
                cell=cell,
                trace_root=trace_root,
                tracer_so_path=tracer_so_path,
            )
        if capture_identity is not None:
            sector = build_sector_summary(
                cell,
                regs_per_thread=int(cell["regs_per_thread"]),
                static_work_path=cell_static_work_path,
                static_features_path=cell_static_features_path,
                trace_root=trace_root,
                binary_sha256=capture_identity["binary_sha256"],
                tracer_build_sha256=capture_identity["tracer_so_sha256"],
                nvbit_launch_attempt_seal_sha256=capture_identity[
                    "nvbit_launch_attempt_seal_sha256"
                ],
            )
            _publish_json_summary(sector_path, sector)
        cell["feature_artifacts"] = {
            "nvbit_sector_summary": _feature_ref(
                sector_path, "timing_nvbit_sector_summary"
            ),
            "static_sass_features": _feature_ref(
                cell_static_features_path, "static_sass_features"
            ),
            "static_work": _feature_ref(cell_static_work_path, "static_work"),
        }
    cache_predictions = _build_cache_hit_predictions(
        cohort=out,
        output_path=cache_hit_predictions_path,
        trace_root=trace_root,
        l1_head_path=l1_head_path,
        l2_head_path=l2_head_path,
        family_buckets_path=family_buckets_path,
        source_commit=commit,
    )
    cache_ref = _feature_ref(
        cache_hit_predictions_path,
        "timing_cache_hit_prediction_bundle",
    )
    for cell in out.get("cells", []):
        cell["feature_artifacts"]["cache_hit_predictions"] = cache_ref
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--cache-hit-predictions",
        type=Path,
        default=DEFAULT_CACHE_HIT_PREDICTIONS,
    )
    parser.add_argument("--static-work", type=Path, default=DEFAULT_STATIC_WORK)
    parser.add_argument("--static-features", type=Path, default=DEFAULT_STATIC_FEATURES)
    parser.add_argument("--tracer-so", type=Path)
    parser.add_argument("--l1-head", type=Path, default=DEFAULT_L1_HEAD)
    parser.add_argument("--l2-head", type=Path, default=DEFAULT_L2_HEAD)
    parser.add_argument("--family-buckets", type=Path, default=DEFAULT_FAMILY_BUCKETS)
    parser.add_argument("--capture-nvbit", action="store_true")
    parser.add_argument("--capture-missing-nvbit", action="store_true")
    args = parser.parse_args(argv)
    if (args.capture_nvbit or args.capture_missing_nvbit) and args.tracer_so is None:
        parser.error(
            "--capture-nvbit/--capture-missing-nvbit require --tracer-so "
            "with an external non-Git split-capable tracer"
        )
    output_path = _assert_output_target(
        args.output,
        operation="publish long timing cohort",
    )
    _require_absent_output_target(
        output_path,
        operation="publish long timing cohort",
    )
    cohort = _load(args.cohort)
    out = attach_feature_artifacts_to_cohort(
        cohort,
        feature_dir=args.feature_dir,
        trace_root=args.trace_root,
        cache_hit_predictions_path=args.cache_hit_predictions,
        static_work_path=args.static_work,
        static_features_path=args.static_features,
        tracer_so_path=args.tracer_so,
        l1_head_path=args.l1_head,
        l2_head_path=args.l2_head,
        family_buckets_path=args.family_buckets,
        capture_nvbit=args.capture_nvbit,
        capture_missing_nvbit=args.capture_missing_nvbit,
    )
    _publish_json_summary(output_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
