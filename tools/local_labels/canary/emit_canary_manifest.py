#!/usr/bin/env python3
"""Capture the address-carrying-contract canary and emit its evidence manifest.

Reproducible end-to-end flow on the current dev host (RTX 3090 Ti):

  1. Build `sm86_addr_carry_canary.cu` for sm_86 (nvcc 13.2 toolkit include).
  2. Capture it under the freshly-built new-contract tracer (address-carrying
     contract) into a trace dir — LD_LIBRARY_PATH must carry the pinned protobuf
     lib or NVBit injection fails SILENTLY (no trace); we assert the trace exists.
  3. Admit and retain the captured protobuf generation with `trace_admission`.
  4. Read `trace_contract_id` + `memory_payload_classes`; count admitted ATOMIC
     (class 6) and ASYNC_COPY (class 8) records + the per-address `address_space`
     histogram + how many LDGSTS/ASYNC_COPY records resolved an `allocation_id`;
     assert 0 `unsupported_memory_path` exclusions for the flipped classes.
  5. Emit a manifest JSON with the pinned provenance (git head, clean-tree, tool
     pins, clock-lock state) + the canary evidence.

The canary is proven by: `sm86_addr_carry_shmem_split_v2` + the exact ordered
`[GLOBAL, SHARED, LOCAL, ATOMIC, ASYNC_COPY]` capability vector, nonzero admitted
atomic/async records, 0 unexpected `unsupported_memory_path`, and the LDGSTS
global source resolving an `allocation_id` (the operand-ordering adjudication
confirmed end-to-end on silicon).

Targets-never-features boundary: no NCU metric is read here; this is a
trace-input generator + validator.

Usage:
  Provide --nvbit-tracer and --nvbit-tracer-sha256 (or the matching explicit
  NVBIT_EXTERNAL_TRACER_SO/NVBIT_EXTERNAL_TRACER_SHA256 environment pair).
  .venv/bin/python tools/local_labels/canary/emit_canary_manifest.py \
      --nvbit-tracer /absolute/external/tracer_tool.so \
      --nvbit-tracer-sha256 <64-lowercase-hex> \
      --out results/nvbit_capture_staging/canary/canary_manifest.json \
      [--work <scratch_dir>]
Exit: 0 all canary checks pass; 1 a check failed; 2 setup/build/capture error.
"""
from __future__ import annotations

import argparse
import contextvars
import errno
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    create_exclusive_directory_stage,
    open_retained_executable,
    open_retained_regular,
    publish_regular_file,
    replace_regular_bytes,
    retire_regular_file,
)
from tools.nvbit_tracer import capture_process as nvbit_process  # noqa: E402
from tools.nvbit_tracer.external_tracer import (  # noqa: E402
    EXTERNAL_TRACER_MAX_BYTES,
    ExternalTracerAuthority,
    ExternalTracerAuthorityError,
    require_external_tracer_authority,
    verify_retained_tracer_authority,
)
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)
from tools.nvbit_tracer.runtime_env import (  # noqa: E402
    default_runtime_manifest_path,
    retained_tracer_runtime_env,
    sanitized_capture_environment,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    DEFAULT_TRACE_ADMISSION_LIMITS,
    EXPECTED_MEMORY_PAYLOAD_CLASSES,
    EXPECTED_TRACE_CONTRACT_ID,
    AdmittedTraceSources,
    TraceAdmission,
    TraceAdmissionError,
    open_admitted_trace_sources,
    raw_trace_generation_limits,
    require_trace_admission,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    TRUSTED_GIT_EXECUTABLE,
    ProtectedPathError,
    assert_write_allowed,
    run_retained_git,
)

CANARY_DIR = REPO / "tools" / "local_labels" / "canary"
CANARY_CU = CANARY_DIR / "sm86_addr_carry_canary.cu"
PB_LIB = REPO / ".toolchain" / "pb" / "lib"
NVCC = "/usr/local/cuda-13.2/bin/nvcc"
HOST_COMPILER = Path("/usr/bin/g++-14")
CUDA_INC = "/usr/local/cuda-13.2/include"

ATOMIC, ASYNC_COPY, GLOBAL = 6, 8, 1
CLASS_NAME = {ATOMIC: "ATOMIC", ASYNC_COPY: "ASYNC_COPY", GLOBAL: "GLOBAL"}
STAGING_ROOT = REPO / "results" / "nvbit_capture_staging"
DEFAULT_CANARY_ROOT = STAGING_ROOT / "canary"
_ACTIVE_RETAINED_GIT: contextvars.ContextVar[object | None] = (
    contextvars.ContextVar("canary_retained_git", default=None)
)
_CANARY_CAPTURE_TIMEOUT_SECONDS = 15 * 60
_CANARY_CAPTURE_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_CANARY_CAPTURE_TIMEOUT_CODE = "canary_nvbit_capture_timeout"
_CANARY_CAPTURE_LAUNCH_CODE = "canary_nvbit_capture_failed"
_CANARY_CAPTURE_DIAGNOSTIC_BYTES = 4096
_CANARY_CAPTURE_RESOURCE_ERRNOS = frozenset({
    errno.EAGAIN,
    errno.EMFILE,
    errno.ENFILE,
    errno.ENOMEM,
    errno.ENOSPC,
})


class CanaryCaptureError(RuntimeError):
    """A typed, byte-preserving canary capture failure."""

    TOKEN = "canary_nvbit_capture_failed"

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
        super().__init__(f"{self.TOKEN}: {detail}")


class CanaryCaptureTimeout(CanaryCaptureError):
    TOKEN = "canary_nvbit_capture_timeout"


class CanaryCaptureOutputLimitExceeded(CanaryCaptureError):
    TOKEN = "canary_nvbit_capture_output_limit"


class CanaryCaptureLaunchFailed(CanaryCaptureError):
    TOKEN = "canary_nvbit_capture_launch_failed"


class CanaryCaptureResourceUnavailable(CanaryCaptureError):
    TOKEN = "canary_nvbit_capture_resource_unavailable"


class CanaryCaptureCleanupFailed(CanaryCaptureError):
    TOKEN = "canary_nvbit_capture_cleanup_failed"


class CanaryCaptureNonzero(CanaryCaptureError):
    TOKEN = "canary_nvbit_capture_nonzero"

    def __init__(self, *, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self.returncode = returncode
        diagnostic = _canary_capture_diagnostic(stderr)
        super().__init__(
            f"target exited {returncode}; stderr: {diagnostic!r}",
            stdout=stdout,
            stderr=stderr,
        )


def _canary_capture_diagnostic(payload: bytes) -> str:
    return payload[-_CANARY_CAPTURE_DIAGNOSTIC_BYTES:].decode(
        "utf-8", errors="replace",
    )


def _canary_resource_failure(error: BaseException) -> bool:
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
            and current.errno in _CANARY_CAPTURE_RESOURCE_ERRNOS
        ):
            return True
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


def _map_canary_capture_error(
    error: nvbit_process.NvbitCaptureProcessError,
) -> CanaryCaptureError:
    fields = {
        "stdout": error.stdout,
        "stderr": error.stderr,
        "process_id": error.process_id,
        "observed_output_bytes_lower_bound": (
            error.observed_output_bytes_lower_bound
        ),
    }
    if error.process_cleanup_incomplete:
        return CanaryCaptureCleanupFailed(error.detail, **fields)
    if error.code == nvbit_process.NVBIT_CAPTURE_OUTPUT_LIMIT:
        return CanaryCaptureOutputLimitExceeded(error.detail, **fields)
    if error.code == _CANARY_CAPTURE_TIMEOUT_CODE:
        return CanaryCaptureTimeout(error.detail, **fields)
    if _canary_resource_failure(error):
        return CanaryCaptureResourceUnavailable(error.detail, **fields)
    return CanaryCaptureLaunchFailed(error.detail, **fields)


@dataclass(frozen=True)
class CapturedTrace:
    path: Path
    attempt_id: str
    admission: TraceAdmission
    content_manifest_sha256: str
    filesystem_generation_sha256: str


# The two named scorability witnesses (the plan's positive-test cells): a tensor-core HMMA
# GEMM whose cp.async operand loads (LDGSTS -> ASYNC_COPY) drive the op_ld surface,
# and a global-atomic histogram whose atomics (-> ATOMIC) drive the L2 surface.
# `surface_class` is the flipped op class that was STRIPPED pre-flip (making the
# surface `na`); the witness passes when that class is admitted into the trace AND
# into the sim's L2 consuming path (cache_replay), never left in the exclusion
# sidecar.
WITNESSES = [
    {"name": "gemm_hmma_bf16", "cu": "ac4_gemm_hmma_bf16.cu",
     "surface": "op_ld", "surface_class": ASYNC_COPY},
    {"name": "ac4_global_atomic_histogram", "cu": "ac4_global_atomic_histogram.cu",
     "surface": "l2", "surface_class": ATOMIC},
]


_SOURCE_COMMIT_AUTHORITY_PATHS = (
    "tools/local_labels/canary/emit_canary_manifest.py",
    "tools/local_labels/canary/sm86_addr_carry_canary.cu",
    "tools/local_labels/canary/ac4_gemm_hmma_bf16.cu",
    "tools/local_labels/canary/ac4_global_atomic_histogram.cu",
    "tools/local_labels/delegated_cgroup.py",
    "tools/local_labels/immutable_artifact.py",
    "tools/local_labels/protected_paths.py",
    "tools/nvbit_tracer/capture_process.py",
    "tools/nvbit_tracer/external_tracer.py",
    "tools/nvbit_tracer/import_trace_contract.py",
    "tools/nvbit_tracer/proto_binding_loader.py",
    "tools/nvbit_tracer/protobuf_wire_preflight.py",
    "tools/nvbit_tracer/runtime_env.py",
    "tools/nvbit_tracer/trace_admission.py",
    "tools/cache_replay/sector_cache_replay.py",
    "workloads/simbench_bw/_pb_generated/address_pb2.py",
    "workloads/simbench_bw/_pb_generated/cuda_stream_pb2.py",
    "workloads/simbench_bw/_pb_generated/dim3d_pb2.py",
    "workloads/simbench_bw/_pb_generated/gpu_device_pb2.py",
    "workloads/simbench_bw/_pb_generated/instruction_pb2.py",
    "workloads/simbench_bw/_pb_generated/kernel_pb2.py",
    "workloads/simbench_bw/_pb_generated/threadblock_pb2.py",
    "workloads/simbench_bw/_pb_generated/trace_pb2.py",
    "workloads/simbench_bw/_pb_generated/warp_pb2.py",
)


def _canary_git_head(retained_git: object | None = None) -> str:
    if retained_git is None:
        retained_git = _ACTIVE_RETAINED_GIT.get()
    if retained_git is None:
        with open_retained_executable(TRUSTED_GIT_EXECUTABLE) as owned_git:
            return _canary_git_head(owned_git)
    result = run_retained_git(
        retained_git,
        repo_root=REPO,
        arguments=("rev-parse", "--verify", "HEAD"),
        text=True,
    )
    head = result.stdout.strip()
    if (
        result.returncode != 0
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise RuntimeError("cannot resolve exact canary source_commit")
    return head


def _resolve_canary_source_commit(
    retained_authority: dict[str, object],
    retained_git: object | None = None,
    *,
    _legacy_head_call: bool = False,
) -> str:
    if retained_git is None:
        active_git = _ACTIVE_RETAINED_GIT.get()
        if active_git is not None:
            return _resolve_canary_source_commit(
                retained_authority,
                active_git,
                _legacy_head_call=True,
            )
        with open_retained_executable(TRUSTED_GIT_EXECUTABLE) as owned_git:
            return _resolve_canary_source_commit(
                retained_authority,
                owned_git,
                _legacy_head_call=True,
            )
    expected = set(_SOURCE_COMMIT_AUTHORITY_PATHS)
    if (
        len(_SOURCE_COMMIT_AUTHORITY_PATHS) != len(expected)
        or set(retained_authority) != expected
        or len(retained_authority) != len(expected)
    ):
        missing = sorted(expected - set(retained_authority))
        extra = sorted(set(retained_authority) - expected)
        raise RuntimeError(
            "canary source-commit authority set is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    source_commit = (
        _canary_git_head()
        if _legacy_head_call
        else _canary_git_head(retained_git)
    )
    status = run_retained_git(
        retained_git,
        repo_root=REPO,
        arguments=(
            "status", "--porcelain=v1", "--untracked-files=all", "--",
            *_SOURCE_COMMIT_AUTHORITY_PATHS,
        ),
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("cannot verify canary source-commit cleanliness")
    if status.stdout.strip():
        raise RuntimeError(
            "canary source-commit authority is dirty or untracked: "
            f"{status.stdout.strip()!r}"
        )
    for relative_path in _SOURCE_COMMIT_AUTHORITY_PATHS:
        committed = run_retained_git(
            retained_git,
            repo_root=REPO,
            arguments=("show", f"{source_commit}:{relative_path}"),
        )
        if committed.returncode != 0:
            raise RuntimeError(
                f"canary authority is absent from source_commit {source_commit}: "
                f"{relative_path}"
            )
        retained = retained_authority[relative_path]
        if committed.stdout != retained.read_bytes():
            raise RuntimeError(
                "retained canary authority bytes disagree with source_commit for "
                f"{relative_path}"
            )
        retained.verify_unchanged()
    final_head = (
        _canary_git_head()
        if _legacy_head_call
        else _canary_git_head(retained_git)
    )
    if final_head != source_commit:
        raise RuntimeError(
            "canary source_commit changed during authority verification"
        )
    return source_commit


def _verify_canary_source_authority(
    source_commit: str,
    retained_authority: dict[str, object],
    retained_git: object | None = None,
) -> None:
    for retained in retained_authority.values():
        retained.verify_unchanged()
    observed_head = (
        _canary_git_head()
        if retained_git is None
        else _canary_git_head(retained_git)
    )
    if observed_head != source_commit:
        raise RuntimeError("canary source_commit changed during active publication")


def _assert_staging_output(path: Path | str, *, operation: str) -> Path:
    candidate = assert_write_allowed(path, repo_root=REPO, operation=operation)
    staging = STAGING_ROOT.resolve(strict=False)
    try:
        candidate.relative_to(staging)
    except ValueError as exc:
        raise ProtectedPathError(
            f"refusing to {operation} outside {staging}: {candidate}"
        ) from exc
    return candidate


def _canary_output_plan(
    *, out: Path | str | None = None, work: Path | str | None = None,
) -> dict[str, Path]:
    """Clear the complete canary output graph before any probe or build."""

    work_root = _assert_staging_output(
        work if work is not None else DEFAULT_CANARY_ROOT / "work",
        operation="select canary work root",
    )
    manifest = _assert_staging_output(
        out if out is not None else DEFAULT_CANARY_ROOT / "canary_manifest.json",
        operation="write canary manifest",
    )
    names = [CANARY_CU.stem, *(Path(w["cu"]).stem for w in WITNESSES)]
    candidates: dict[str, Path] = {
        "work": work_root,
        "out": manifest,
        "canary_binary": work_root / names[0],
        "canary_trace": work_root / "canary_trace",
    }
    for witness, stem in zip(WITNESSES, names[1:], strict=True):
        candidates[f"{witness['name']}_binary"] = work_root / stem
        candidates[f"{witness['name']}_trace"] = (
            work_root / f"{witness['name']}_trace"
        )
    return {
        name: _assert_staging_output(path, operation=f"prepare {name}")
        for name, path in candidates.items()
    }


def _run(cmd, *, nvbit_capture: bool = False, **kw):
    if not nvbit_capture:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    try:
        return nvbit_process.execute_capture_process(
            tuple(cmd),
            environment=dict(kw["env"]),
            working_directory=str(kw["cwd"]),
            inherited_descriptors=tuple(kw.get("pass_fds", ())),
            timeout_seconds=_CANARY_CAPTURE_TIMEOUT_SECONDS,
            max_output_bytes=_CANARY_CAPTURE_MAX_OUTPUT_BYTES,
            timeout_code=_CANARY_CAPTURE_TIMEOUT_CODE,
            launch_code=_CANARY_CAPTURE_LAUNCH_CODE,
        )
    except nvbit_process.NvbitCaptureProcessError as process_error:
        mapped = _map_canary_capture_error(process_error)
        raise mapped from process_error
    except (MemoryError, OSError) as process_error:
        if _canary_resource_failure(process_error):
            raise CanaryCaptureResourceUnavailable(
                str(process_error)
            ) from process_error
        raise CanaryCaptureLaunchFailed(str(process_error)) from process_error


def _git(retained_git: object, *args: str) -> str:
    result = run_retained_git(
        retained_git,
        repo_root=REPO,
        arguments=tuple(args),
        text=True,
    )
    if result.returncode != 0:
        command = " ".join(args)
        detail = result.stderr.strip()
        raise RuntimeError(
            f"canary Git {command} exited {result.returncode}: {detail}"
        )
    return result.stdout.strip()


def _clock_state() -> dict:
    p = _run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,persistence_mode",
              "--format=csv,noheader"])
    return {"nvidia_smi": p.stdout.strip() if p.returncode == 0 else "unavailable"}


def build_cu(
    cu: Path,
    work: Path,
    *,
    binary: Path | None = None,
    retained_descriptor: int | None = None,
    retained_nvcc: object | None = None,
    retained_host_compiler: object | None = None,
) -> Path:
    binary = _assert_staging_output(
        binary if binary is not None else work / cu.stem,
        operation="build canary binary",
    )
    with ExitStack() as compiler_sources:
        if retained_nvcc is None:
            retained_nvcc = compiler_sources.enter_context(
                open_retained_executable(Path(NVCC))
            )
        if retained_host_compiler is None:
            retained_host_compiler = compiler_sources.enter_context(
                open_retained_executable(
                    HOST_COMPILER, allow_hardlinks=True,
                )
            )
        if retained_descriptor is None:
            retained_source = compiler_sources.enter_context(
                open_retained_regular(cu)
            )
            retained_descriptor = retained_source.descriptor
        if type(retained_descriptor) is not int or retained_descriptor < 0:
            raise RuntimeError("retained CUDA source descriptor is invalid")
        for capability, label in (
            (retained_nvcc, "NVCC"),
            (retained_host_compiler, "host compiler"),
        ):
            if (
                type(getattr(capability, "descriptor", None)) is not int
                or getattr(capability, "descriptor") < 0
                or type(getattr(capability, "proc_path", None)) is not str
                or not callable(getattr(capability, "verify_unchanged", None))
            ):
                raise RuntimeError(f"retained {label} capability is invalid")
            capability.verify_unchanged()
        p = _run(
            [
                retained_nvcc.proc_path,
                f"-ccbin={retained_host_compiler.proc_path}",
                "-O3",
                "-std=c++17",
                "-gencode",
                "arch=compute_86,code=sm_86",
                "-lineinfo",
                f"-I{CUDA_INC}",
                "-x",
                "cu",
                f"/proc/self/fd/{retained_descriptor}",
                "-o",
                str(binary),
            ],
            pass_fds=(
                retained_nvcc.descriptor,
                retained_host_compiler.descriptor,
                retained_descriptor,
            ),
            env={
                "HOME": "/nonexistent",
                "LANG": "C",
                "LANGUAGE": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
        )
        retained_nvcc.verify_unchanged()
        retained_host_compiler.verify_unchanged()
    if p.returncode != 0 or not binary.is_file():
        raise SystemExit(f"[canary] nvcc build failed for {cu.name}:\n{p.stderr}")
    return binary


def _new_capture_attempt_id() -> str:
    return f"{time.time_ns():020d}-{os.urandom(8).hex()}"


def capture(
    binary: Path,
    out: Path,
    *,
    tracer_authority: ExternalTracerAuthority,
) -> CapturedTrace:
    """Capture one no-reuse immutable generation and return its admission seal."""

    if not isinstance(tracer_authority, ExternalTracerAuthority):
        raise TypeError("tracer_authority must be an ExternalTracerAuthority")
    generation_root = _assert_staging_output(
        out, operation="capture canary trace generation"
    )
    generation_root.mkdir(parents=True, exist_ok=True)
    attempt_id = _new_capture_attempt_id()
    stage = create_exclusive_directory_stage(generation_root, attempt_id)
    with stage:
        with (
            open_retained_executable(binary) as retained_binary,
            open_retained_regular(
                tracer_authority.path,
                max_bytes=EXTERNAL_TRACER_MAX_BYTES,
            ) as retained_tracer,
        ):
            verify_retained_tracer_authority(retained_tracer, tracer_authority)
            with retained_tracer_runtime_env(
                retained_tracer.path,
                base_env=sanitized_capture_environment(),
                pb_lib=PB_LIB,
                runtime_manifest=default_runtime_manifest_path(
                    retained_tracer.path
                ),
                retained_descriptor=retained_tracer.descriptor,
            ) as retained_runtime:
                env = dict(retained_runtime.environment)
                env["CUDA_INJECTION64_PATH"] = retained_tracer.proc_path
                env["USER_DEFINED_FOLDERS"] = "1"
                env["TRACES_FOLDER"] = stage.proc_path
                p = _run(
                    [retained_binary.proc_path],
                    nvbit_capture=True,
                    env=env,
                    cwd=str(retained_binary.path.parent),
                    pass_fds=(
                        retained_binary.descriptor,
                        retained_tracer.descriptor,
                        stage.descriptor,
                        *retained_runtime.pass_fds,
                    ),
                )
                retained_binary.verify_unchanged()
                retained_tracer.verify_unchanged()
        stage.verify_path()
        if p.returncode != 0:
            raise CanaryCaptureNonzero(
                returncode=p.returncode,
                stdout=p.stdout,
                stderr=p.stderr,
            )
        generation = stage.commit(
            require_nonempty=True,
            **raw_trace_generation_limits(DEFAULT_TRACE_ADMISSION_LIMITS),
        )

    try:
        admission = require_trace_admission(generation.path)
    except TraceAdmissionError as exc:
        raise SystemExit(f"[canary] captured trace is not admissible: {exc}") from exc
    return CapturedTrace(
        path=generation.path,
        attempt_id=attempt_id,
        admission=admission,
        content_manifest_sha256=generation.content_manifest_sha256,
        filesystem_generation_sha256=generation.filesystem_generation_sha256,
    )


def _capability_vector_matches(memory_payload_classes: object) -> bool:
    return bool(
        isinstance(memory_payload_classes, (list, tuple))
        and all(type(value) is int for value in memory_payload_classes)
        and tuple(memory_payload_classes)
        == EXPECTED_MEMORY_PAYLOAD_CLASSES
    )


def _decode_admitted_access_stream(sources: AdmittedTraceSources):
    sys.path.insert(0, str(REPO / "tools" / "cache_replay"))
    from sector_cache_replay import (  # type: ignore
        decode_admitted_access_stream,
    )

    return decode_admitted_access_stream(sources)


def analyze(sources: AdmittedTraceSources, unsupported_memory_path) -> dict:
    bindings = load_dynamic_trace_bindings()
    trace_pb2 = bindings.trace_pb2
    threadblock_pb2 = bindings.threadblock_pb2

    t = trace_pb2.Trace()
    t.ParseFromString(sources.read_dynamic_trace_snapshot().payload)

    cls = Counter()
    space = Counter()
    async_with_id = 0
    for relative_path in sources.threadblock_paths:
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(
            sources.read_threadblock_snapshot(relative_path).payload
        )
        for wi in tb.warps:
            for inst in tb.warps[wi].instructions:
                if inst.HasField("memory_op_class") and inst.addresses:
                    cls[inst.memory_op_class] += 1
                    for a in inst.addresses:
                        if a.HasField("address_space"):
                            space[a.address_space] += 1
                        if (inst.memory_op_class == ASYNC_COPY
                                and a.HasField("allocation_id") and a.allocation_id):
                            async_with_id += 1
    excl = json.loads(unsupported_memory_path.read_bytes())
    flipped_excl = [
        entry
        for entry in excl
        if entry.get("excluded_op_class") in ("ATOMIC", "ASYNC_COPY")
    ]
    allocations = sources.allocations
    sources.verify_unchanged()
    unsupported_memory_path.verify_unchanged()

    return {
        "validate_trace_dir": {
            "ok": True,
            "typed_reason": "trace_admission_passed",
        },
        "trace_contract_id": t.trace_contract_id,
        "memory_payload_classes": list(t.memory_payload_classes),
        "admitted_records_by_class": {int(k): int(n) for k, n in cls.items()},
        "admitted_atomic": cls.get(ATOMIC, 0),
        "admitted_async_copy": cls.get(ASYNC_COPY, 0),
        "admitted_global": cls.get(GLOBAL, 0),
        "address_space_histogram": {int(k): int(n) for k, n in space.items()},
        "ldgsts_source_records_with_allocation_id": async_with_id,
        "unsupported_memory_path_total": len(excl),
        "unsupported_memory_path_flipped_classes": len(flipped_excl),
        "allocations_sidecar_entries": len(allocations or {}),
    }


def analyze_witness(
    sources: AdmittedTraceSources,
    unsupported_memory_path,
    surface: str,
    surface_class: int,
) -> dict:
    """Prove a surface (op_ld / L2) went from `na` to scorable.

    The surface-relevant class must be admitted into the trace, have no record in
    the unsupported-path sidecar, and contribute at least one global-source sector
    to cache replay. ASYNC_COPY may also carry a legitimate shared destination;
    those non-L2 operands remain diagnostic exclusions rather than witness failure.
    """
    ev = analyze(sources, unsupported_memory_path)
    stream = _decode_admitted_access_stream(sources)
    admitted_surface = ev["admitted_records_by_class"].get(surface_class, 0)
    surface_included_in_l2 = int(
        stream.included_op_class_counts.get(surface_class, 0)
    )
    surface_excluded_from_l2 = int(
        stream.excluded_op_class_counts.get(surface_class, 0)
    )
    scorable = bool(
        ev["validate_trace_dir"]["ok"]
        and ev["trace_contract_id"] == EXPECTED_TRACE_CONTRACT_ID
        and _capability_vector_matches(ev["memory_payload_classes"])
        and admitted_surface > 0
        and ev["unsupported_memory_path_flipped_classes"] == 0
        and surface_included_in_l2 > 0
    )
    sources.verify_unchanged()
    unsupported_memory_path.verify_unchanged()
    return {
        "surface": surface,
        "surface_class": CLASS_NAME.get(surface_class, surface_class),
        "trace_contract_id": ev["trace_contract_id"],
        "memory_payload_classes": ev["memory_payload_classes"],
        "validate_ok": ev["validate_trace_dir"]["ok"],
        "admitted_surface_class_records": admitted_surface,
        "unsupported_memory_path_flipped_classes": ev["unsupported_memory_path_flipped_classes"],
        "cache_replay_l2_stream_sectors": int(stream.n_global),
        "cache_replay_surface_class_included_in_l2": surface_included_in_l2,
        "cache_replay_surface_class_excluded_from_l2": surface_excluded_from_l2,
        "before": "na (records stripped to unsupported_memory_path pre-flip; surface starved)",
        "after": "scorable" if scorable else "na",
        "scorable": scorable,
    }


def witness_verdict(name: str, w: dict) -> list[str]:
    """Fail-closed witness conditions: the surface must be scorable."""
    fails = []
    if not w["validate_ok"]:
        fails.append(f"witness {name}: validate_trace_dir failed")
    if w["trace_contract_id"] != EXPECTED_TRACE_CONTRACT_ID:
        fails.append(f"witness {name}: wrong trace_contract_id {w['trace_contract_id']!r}")
    if not _capability_vector_matches(w["memory_payload_classes"]):
        fails.append(
            f"witness {name}: memory_payload_classes "
            f"{w['memory_payload_classes']!r} != exact "
            f"{list(EXPECTED_MEMORY_PAYLOAD_CLASSES)!r}"
        )
    if w["admitted_surface_class_records"] <= 0:
        fails.append(f"witness {name}: {w['surface']} surface has 0 admitted "
                     f"{w['surface_class']} records (still na)")
    if w["unsupported_memory_path_flipped_classes"] != 0:
        fails.append(f"witness {name}: flipped-class records still in unsupported_memory_path")
    if w["cache_replay_surface_class_included_in_l2"] <= 0:
        fails.append(
            f"witness {name}: {w['surface_class']} has no admitted L2 sectors"
        )
    if not w["scorable"]:
        fails.append(f"witness {name}: {w['surface']} surface not scorable (remains na)")
    return fails


def canary_verdict(ev: dict) -> tuple[bool, list[str]]:
    fails = []
    if not ev["validate_trace_dir"]["ok"]:
        fails.append(f"validate_trace_dir failed: {ev['validate_trace_dir']['typed_reason']}")
    if ev["trace_contract_id"] != EXPECTED_TRACE_CONTRACT_ID:
        fails.append(
            f"trace_contract_id={ev['trace_contract_id']!r} != "
            f"{EXPECTED_TRACE_CONTRACT_ID!r}"
        )
    if not _capability_vector_matches(ev["memory_payload_classes"]):
        fails.append(
            f"memory_payload_classes={ev['memory_payload_classes']!r} != exact "
            f"{list(EXPECTED_MEMORY_PAYLOAD_CLASSES)!r}"
        )
    if ev["admitted_atomic"] <= 0:
        fails.append("no admitted ATOMIC records")
    if ev["admitted_async_copy"] <= 0:
        fails.append("no admitted ASYNC_COPY records")
    if ev["unsupported_memory_path_flipped_classes"] != 0:
        fails.append("ATOMIC/ASYNC records still address-stripped to unsupported_memory_path")
    if ev["ldgsts_source_records_with_allocation_id"] <= 0:
        fails.append("no LDGSTS global-source record resolved an allocation_id")
    return (not fails), fails


def _open_retained_capture(
    stack: ExitStack, captured: CapturedTrace
) -> tuple[AdmittedTraceSources, object]:
    sources = stack.enter_context(
        open_admitted_trace_sources(
            captured.path,
            expected_admission=captured.admission,
        )
    )
    unsupported = stack.enter_context(
        open_retained_regular(
            captured.path / "extra_info" / "unsupported_memory_path.json",
            max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.trace_root_bytes,
        )
    )
    return sources, unsupported


def _capture_identity(captured: CapturedTrace) -> dict[str, object]:
    return {
        "attempt_id": captured.attempt_id,
        "path": str(captured.path.relative_to(REPO)),
        "content_manifest_sha256": captured.content_manifest_sha256,
        "filesystem_generation_sha256": captured.filesystem_generation_sha256,
        "trace_admission_sha256": captured.admission.trace_admission_sha256,
        "trace_contract_id": captured.admission.trace_contract_id,
        "memory_payload_classes": list(
            captured.admission.memory_payload_classes
        ),
        "trace_filesystem_generation_sha256": (
            captured.admission.trace_filesystem_generation_sha256
        ),
    }


def _verify_retained_captures(
    retained: list[tuple[AdmittedTraceSources, object]],
) -> None:
    for sources, unsupported in retained:
        sources.verify_unchanged()
        unsupported.verify_unchanged()


def _manifest_generation_root(out: Path) -> Path:
    return _assert_staging_output(
        out.with_name(f"{out.name}.generations"),
        operation="publish canary manifest generation",
    )


def _canonical_manifest_generation_relative_path(
    *, out: Path, generation_root: Path, attempt_id: str
) -> str:
    return str(
        (generation_root / f"{attempt_id}.json").relative_to(out.parent)
    )


def _is_current_manifest_locator(
    payload: bytes,
    *,
    out: Path,
    generation_root: Path,
) -> bool:
    try:
        model = json.loads(payload)
        if (
            not isinstance(model, dict)
            or set(model) != {
                "schema_version",
                "artifact_kind",
                "manifest_generation",
            }
            or type(model.get("schema_version")) is not int
            or model["schema_version"] != 1
            or model.get("artifact_kind")
            != "sm86_addr_carry_canary_manifest_current"
        ):
            return False
        generation = model.get("manifest_generation")
        if not isinstance(generation, dict) or set(generation) != {
            "attempt_id",
            "path",
            "sha256",
            "size_bytes",
        }:
            return False
        attempt_id = generation.get("attempt_id")
        relative = generation.get("path")
        digest = generation.get("sha256")
        size_bytes = generation.get("size_bytes")
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or Path(attempt_id).name != attempt_id
            or not isinstance(relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size_bytes) is not int
            or size_bytes <= 0
            or size_bytes
            > DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes
        ):
            return False
        canonical_relative = _canonical_manifest_generation_relative_path(
            out=out,
            generation_root=generation_root,
            attempt_id=attempt_id,
        )
        if relative != canonical_relative:
            return False
        generation_path = out.parent / canonical_relative
        with open_retained_regular(
            generation_path,
            max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes,
        ) as retained:
            if retained.size_bytes != size_bytes or retained.sha256 != digest:
                return False
            retained.verify_unchanged()
        return True
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
        ImmutableArtifactError,
    ):
        return False


def _verify_manifest_generation(path: Path, payload: bytes) -> None:
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    expected_size = len(payload)
    with open_retained_regular(
        path,
        max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes,
    ) as existing:
        if (
            existing.size_bytes != expected_size
            or existing.sha256 != expected_sha256
        ):
            raise RuntimeError(
                "canary manifest generation collision has different size or SHA-256"
            )
        if existing.read_bytes(
            max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes
        ) != payload:
            raise RuntimeError(
                "canary manifest generation collision has different bytes"
            )
        existing.verify_unchanged()


def _verify_legacy_manifest_generation(path: Path, payload: bytes) -> None:
    with open_retained_regular(
        path,
        max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes,
    ) as existing:
        if existing.read_bytes(
            max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes
        ) != payload:
            raise RuntimeError(
                "legacy canary manifest digest name has different bytes"
            )
        existing.verify_unchanged()


def _preserve_legacy_manifest(out: Path, generation_root: Path) -> None:
    if not out.exists():
        return
    with open_retained_regular(
        out,
        max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes,
    ) as retained:
        payload = retained.read_bytes(
            max_bytes=DEFAULT_TRACE_ADMISSION_LIMITS.allocations_sidecar_bytes
        )
        if _is_current_manifest_locator(
            payload, out=out, generation_root=generation_root
        ):
            return
        legacy_name = f"legacy-{hashlib.sha256(payload).hexdigest()}.json"
        legacy_path = generation_root / legacy_name
        try:
            publish_regular_file(generation_root, legacy_name, payload)
        except ImmutableArtifactError as publication_error:
            try:
                _verify_legacy_manifest_generation(legacy_path, payload)
            except (OSError, ImmutableArtifactError, RuntimeError) as verify_error:
                raise publication_error from verify_error
        else:
            _verify_legacy_manifest_generation(legacy_path, payload)
        retained.verify_unchanged()


@contextmanager
def _retire_active_manifest_on_source_failure(state: dict[str, Path]):
    try:
        yield
    except BaseException as error:
        active = state.get("active")
        if active is not None:
            try:
                retire_regular_file(active)
            except BaseException as retirement_error:
                try:
                    error.add_note(
                        "failed to retire canary active locator after source "
                        f"authority failure: {retirement_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        raise


def _publish_manifest(
    out: Path,
    payload: bytes,
    *,
    attempt_id: str,
    publication_check: Callable[[], None] | None = None,
) -> Path:
    """Publish one immutable manifest generation, then atomically advance current."""

    if publication_check is not None:
        publication_check()
    out = _assert_staging_output(out, operation="publish canary manifest locator")
    out.parent.mkdir(parents=True, exist_ok=True)
    generation_root = _manifest_generation_root(out)
    generation_root.mkdir(parents=True, exist_ok=True)
    _preserve_legacy_manifest(out, generation_root)

    generation_name = f"{attempt_id}.json"
    generation_path = generation_root / generation_name
    try:
        published_generation = publish_regular_file(
            generation_root,
            generation_name,
            payload,
        )
    except ImmutableArtifactError as publication_error:
        try:
            _verify_manifest_generation(generation_path, payload)
        except (OSError, ImmutableArtifactError, RuntimeError) as verify_error:
            raise publication_error from verify_error
    else:
        if not isinstance(published_generation, Path):
            raise RuntimeError("manifest generation publication returned no path")
        generation_path = published_generation
    canonical_relative = _canonical_manifest_generation_relative_path(
        out=out,
        generation_root=generation_root,
        attempt_id=attempt_id,
    )
    locator = {
        "schema_version": 1,
        "artifact_kind": "sm86_addr_carry_canary_manifest_current",
        "manifest_generation": {
            "attempt_id": attempt_id,
            "path": canonical_relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
    }
    if publication_check is not None:
        publication_check()
    replace_regular_bytes(
        out,
        (json.dumps(locator, indent=2, sort_keys=True) + "\n").encode(),
    )
    if publication_check is not None:
        try:
            publication_check()
        except BaseException:
            retire_regular_file(out)
            raise
    return generation_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--nvbit-tracer", type=Path, default=None)
    ap.add_argument("--nvbit-tracer-sha256", default=None)
    args = ap.parse_args(argv)

    try:
        tracer_authority = require_external_tracer_authority(
            args.nvbit_tracer,
            args.nvbit_tracer_sha256,
            repo_root=REPO,
        )
        plan = _canary_output_plan(out=args.out, work=args.work)
    except (ProtectedPathError, ExternalTracerAuthorityError) as exc:
        print(f"[canary] setup refusal: {exc}", file=sys.stderr)
        return 2

    # Re-clear the directory immediately before its first mutation.  The
    # batch plan above proves the complete graph up front; this second check
    # closes the interval between planning and mkdir (including a symlink
    # substitution at the selected work path).
    work = _assert_staging_output(
        plan["work"], operation="create canary work root",
    )
    work.mkdir(parents=True, exist_ok=True)

    retained: list[tuple[AdmittedTraceSources, object]] = []
    publication_state: dict[str, Path] = {}
    with ExitStack() as evidence_stack:
        evidence_stack.enter_context(
            _retire_active_manifest_on_source_failure(publication_state)
        )
        retained_authority: dict[str, object] = {}
        try:
            retained_git = evidence_stack.enter_context(
                open_retained_executable(TRUSTED_GIT_EXECUTABLE)
            )
            git_token = _ACTIVE_RETAINED_GIT.set(retained_git)
            evidence_stack.callback(_ACTIVE_RETAINED_GIT.reset, git_token)
            retained_nvcc = evidence_stack.enter_context(
                open_retained_executable(Path(NVCC))
            )
            retained_host_compiler = evidence_stack.enter_context(
                open_retained_executable(
                    HOST_COMPILER, allow_hardlinks=True,
                )
            )
            for relative_path in _SOURCE_COMMIT_AUTHORITY_PATHS:
                retained_authority[relative_path] = evidence_stack.enter_context(
                    open_retained_regular(REPO / relative_path)
                )
            source_commit = _resolve_canary_source_commit(
                retained_authority,
            )
        except (ImmutableArtifactError, OSError, RuntimeError) as exc:
            print(f"[canary] source authority refusal: {exc}", file=sys.stderr)
            return 2
        _verify_canary_source_authority(
            source_commit, retained_authority,
        )
        # The canary: exercises every flipped class; proves admission + contract id.
        binary = build_cu(
            CANARY_CU,
            work,
            binary=plan["canary_binary"],
            retained_descriptor=retained_authority[
                "tools/local_labels/canary/sm86_addr_carry_canary.cu"
            ].descriptor,
            retained_nvcc=retained_nvcc,
            retained_host_compiler=retained_host_compiler,
        )
        _verify_canary_source_authority(
            source_commit, retained_authority,
        )
        try:
            captured = capture(
                binary,
                plan["canary_trace"],
                tracer_authority=tracer_authority,
            )
            _verify_canary_source_authority(
                source_commit, retained_authority,
            )
        except CanaryCaptureError as exc:
            print(f"[canary] capture failed: {exc}", file=sys.stderr)
            return 2
        sources, unsupported = _open_retained_capture(evidence_stack, captured)
        retained.append((sources, unsupported))
        ev = analyze(sources, unsupported)
        ev["trace_generation"] = _capture_identity(captured)
        ok, fails = canary_verdict(ev)

        # The two named scorability witnesses: prove op_ld / L2 na -> scorable.
        witnesses = {}
        for w in WITNESSES:
            witness_relative = f"tools/local_labels/canary/{w['cu']}"
            wbin = build_cu(
                CANARY_DIR / w["cu"],
                work,
                binary=plan[f"{w['name']}_binary"],
                retained_descriptor=retained_authority[
                    witness_relative
                ].descriptor,
                retained_nvcc=retained_nvcc,
                retained_host_compiler=retained_host_compiler,
            )
            _verify_canary_source_authority(
                source_commit, retained_authority,
            )
            try:
                witness_capture = capture(
                    wbin,
                    plan[f"{w['name']}_trace"],
                    tracer_authority=tracer_authority,
                )
                _verify_canary_source_authority(
                    source_commit, retained_authority,
                )
            except CanaryCaptureError as exc:
                print(f"[canary] capture failed: {exc}", file=sys.stderr)
                return 2
            witness_sources, witness_unsupported = _open_retained_capture(
                evidence_stack, witness_capture
            )
            retained.append((witness_sources, witness_unsupported))
            wev = analyze_witness(
                witness_sources,
                witness_unsupported,
                w["surface"],
                w["surface_class"],
            )
            wev["trace_generation"] = _capture_identity(witness_capture)
            witnesses[w["name"]] = wev
            wfails = witness_verdict(w["name"], wev)
            fails.extend(wfails)
        ok = not fails

        manifest = {
            "schema_version": 1,
            "artifact_id": "sm86_addr_carry_canary_split_v2",
            "arch": "sm_86",
            "canary_pass": ok,
            "canary_failures": fails,
            "canary_source": "tools/local_labels/canary/sm86_addr_carry_canary.cu",
            "provenance": {
                "source_commit": source_commit,
                "git_head": source_commit,
                "git_clean": _git(
                    retained_git,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ) == "",
                "tracer_source_kind": tracer_authority.source_kind,
                "tracer_source_path": str(tracer_authority.path),
                "tracer_sha256": tracer_authority.sha256,
                "trace_contract_id": EXPECTED_TRACE_CONTRACT_ID,
                "memory_payload_classes": list(
                    EXPECTED_MEMORY_PAYLOAD_CLASSES
                ),
                "clock_state": _clock_state(),
                "cuda_toolkit": "13.2",
            },
            "evidence": ev,
            "scorability_witnesses": witnesses,
        }
        out = _assert_staging_output(
            plan["out"], operation="write canary manifest"
        )
        manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode()
        _verify_retained_captures(retained)
        _verify_canary_source_authority(
            source_commit, retained_authority,
        )

        manifest_generation = _publish_manifest(
            out,
            manifest_payload,
            attempt_id=captured.attempt_id,
            publication_check=lambda: _verify_canary_source_authority(
                source_commit, retained_authority,
            ),
        )
        publication_state["active"] = out
    verdict = "PASS" if ok else "FAIL"
    for name, wev in witnesses.items():
        print(
            f"  witness {name}: {wev['surface']} surface "
            f"{wev['before'].split()[0]}->{wev['after']} "
            f"(admitted {wev['surface_class']}="
            f"{wev['admitted_surface_class_records']}, "
            f"l2_sectors={wev['cache_replay_l2_stream_sectors']})"
        )
    print(f"canary manifest [{verdict}] -> {manifest_generation} "
          f"(current {out}) | contract={ev['trace_contract_id']} "
          f"ATOMIC={ev['admitted_atomic']} ASYNC_COPY={ev['admitted_async_copy']} "
          f"ldgsts_id={ev['ldgsts_source_records_with_allocation_id']} "
          f"flipped_exclusions={ev['unsupported_memory_path_flipped_classes']}")
    for f in fails:
        print(f"  FAIL: {f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
