"""Fail-closed runtime environment for loading an NVBit tracer tool.

The custom tracer links the checkout-local protobuf/Abseil runtime. CUDA's
``CUDA_INJECTION64_PATH`` loader can fail silently when one of those DSOs is
missing, leaving the target application to run without instrumentation. Keep
that loader contract in one place for every Python capture entry point.
"""

from __future__ import annotations

import argparse
import contextvars
import errno
import hashlib
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from tools.local_labels import immutable_artifact as immutable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PB_LIB = REPO_ROOT / ".toolchain" / "pb" / "lib"
_PINNED_DSO_PREFIXES = ("libprotobuf.so", "libabsl_", "libutf8_")
RUNTIME_MANIFEST_SCHEMA_VERSION = "nvbit_runtime_dso_manifest.v1"
RUNTIME_MANIFEST_AUTHORITY_KIND = "operator_supplied_exact_dso_identity_v1"
RUNTIME_MANIFEST_SUFFIX = ".nvbit-runtime-manifest.json"
RUNTIME_MANIFEST_MAX_BYTES = 1024 * 1024
_TRACER_IDENTITY_MAX_BYTES = 256 * 1024 * 1024
LDD_TIMEOUT_SECONDS = 10.0
LDD_CLEANUP_TIMEOUT_SECONDS = 2.0
LDD_MAX_STDOUT_BYTES = 1024 * 1024
LDD_MAX_STDERR_BYTES = 1024 * 1024
_LDD_EXECUTABLE = Path("/usr/bin/ldd")
_ACTIVE_RETAINED_LDD: contextvars.ContextVar[
    immutable.RetainedRegular | None
] = contextvars.ContextVar("active_retained_ldd", default=None)
_LDD_READ_CHUNK_BYTES = 64 * 1024
_PROCESS_RESOURCE_ERRNOS = frozenset({
    errno.EAGAIN, errno.EMFILE, errno.ENFILE, errno.ENOMEM,
})
_PINNED_RUNTIME_MAX_DSOS = 512
_PINNED_RUNTIME_MAX_TREE_FILES = _PINNED_RUNTIME_MAX_DSOS * 2
_PINNED_RUNTIME_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_PINNED_RUNTIME_TREE_MAX_TOTAL_BYTES = (
    _PINNED_RUNTIME_MAX_TOTAL_BYTES * (_PINNED_RUNTIME_MAX_DSOS + 1)
)
_RUNTIME_COPY_CHUNK_BYTES = 1024 * 1024
NVBIT_CAPTURE_ENVIRONMENT_CONTROLS = frozenset(
    {
        "ACTIVE_FROM_START",
        "CUDA_INJECTION32_PATH",
        "CUDA_INJECTION64_PATH",
        "DYNAMIC_KERNEL_LIMIT_END",
        "DYNAMIC_KERNEL_LIMIT_START",
        "ENHANCED_TRACER_MAX_PARSER_INPUT_KB",
        "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S",
        "EXCLUDE_PRED_OFF",
        "GATHER_REGISTERS",
        "INSTR_BEGIN",
        "INSTR_END",
        "INTERMEDIATE_EXTRA_FILES_PERSISTANCE",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "MAX_ACCEPTED_ACTIVE_LAUNCHES",
        "NOBANNER",
        "NVBIT_PB_PREFIX",
        "NVBIT_TOOL",
        "SIMBENCH_NVBIT_ASYNC_ORDER",
        "SIMBENCH_NVBIT_PHASE_INJECT",
        "SIMBENCH_NVBIT_PHASE_REPEAT_INDEX",
        "SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS",
        "SIMBENCH_PHASE_TEST_STALL_ITERS",
        "SIMBENCH_PHASE_TEST_STALL_LAUNCH",
        "SIMBENCH_PHASE_TEST_STALL_TEARDOWN_ITERS",
        "SKIP_ENHANCED_TRACER",
        "TERMINATE_UPON_LIMIT",
        "THRESHOLD_UNIQUE_KERNEL_CHECKING",
        "TOOL_TRACE_CORE",
        "TOOL_VERBOSE",
        "TRACES_FOLDER",
        "TRACE_SAMPLING_MODE",
        "TRACE_SAMPLING_SELECTED_BLOCKS",
        "TRACE_SAMPLING_SIDECAR",
        "USER_DEFINED_FOLDERS",
    }
)


class TracerRuntimeError(RuntimeError):
    """The tracer cannot be loaded reliably in the requested environment."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        legacy_message = detail is None
        if legacy_message:
            detail = reason
            reason = "tracer_runtime_invalid"
        self.reason = reason
        self.detail = detail
        super().__init__(detail if legacy_message else f"{reason}: {detail}")


@dataclass(frozen=True)
class RuntimeDsoExpectation:
    """One independently authorized ldd alias and exact target identity."""

    soname: str
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RuntimeManifestAuthority:
    """Retained operator authority for one tracer and its runtime DSO closure."""

    path: Path
    sha256: str
    runtime_library_path: Path
    tracer_sha256: str
    tracer_size_bytes: int
    dsos: Mapping[str, RuntimeDsoExpectation]
    retained: immutable.RetainedRegular = field(repr=False)

    def verify_unchanged(self) -> None:
        try:
            self.retained.verify_unchanged()
        except immutable.ImmutableArtifactError as exc:
            raise TracerRuntimeError(
                "runtime_authority_changed",
                f"retained runtime manifest changed: {exc}",
            ) from exc


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_retained_descriptor(descriptor: int) -> tuple[str, int]:
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            f"retained tracer descriptor is unavailable: {exc}",
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            "retained tracer descriptor must identify a regular file",
        )
    if before.st_size <= 0 or before.st_size > _TRACER_IDENTITY_MAX_BYTES:
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            "retained tracer size is outside the runtime identity bound: "
            f"size={before.st_size}, max={_TRACER_IDENTITY_MAX_BYTES}",
        )
    digest = hashlib.sha256()
    offset = 0
    try:
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(_RUNTIME_COPY_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not chunk:
                raise OSError(errno.EIO, "short retained tracer read")
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            f"could not hash retained tracer descriptor: {exc}",
        ) from exc
    if _descriptor_identity(after) != _descriptor_identity(before):
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            "retained tracer descriptor changed during identity hashing",
        )
    return digest.hexdigest(), before.st_size


def default_runtime_manifest_path(tracer_so: Path) -> Path:
    """Return the nonambient sidecar location for an operator DSO manifest.

    Published external tracer bundles have an exact four-file inventory, so the
    sidecar for ``tracer_tool.so`` lives beside (not inside) the bundle. Other
    standalone tracers use a same-directory sidecar.
    """

    tracer = Path(tracer_so).resolve()
    if tracer.name == "tracer_tool.so":
        return tracer.parent.parent / f"{tracer.parent.name}{RUNTIME_MANIFEST_SUFFIX}"
    return tracer.with_name(f"{tracer.name}{RUNTIME_MANIFEST_SUFFIX}")


def _manifest_path(
    tracer_so: Path,
    runtime_manifest: Path | None,
) -> Path:
    candidate = (
        default_runtime_manifest_path(tracer_so)
        if runtime_manifest is None
        else Path(runtime_manifest)
    )
    if not candidate.is_absolute():
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            f"runtime manifest path must be absolute: {candidate}",
        )
    if any(
        candidate.parts[index : index + 2] == (".claude", "worktrees")
        for index in range(len(candidate.parts) - 1)
    ):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest must not come from a Claude worktree",
        )
    return candidate


def _manifest_prerequisite_detail(path: Path) -> str:
    return (
        f"exact runtime DSO authority is missing: {path}; retain the default "
        "operator sidecar or have the capture caller pass runtime_manifest explicitly; "
        "the contract requires schema_version "
        f"{RUNTIME_MANIFEST_SCHEMA_VERSION!r}, authority_kind "
        f"{RUNTIME_MANIFEST_AUTHORITY_KIND!r}, exact tracer sha256/size_bytes, "
        "an absolute runtime_library_path, and a sorted exact dsos list of "
        "soname/path/sha256/size_bytes entries obtained from independent build "
        "provenance; runtime discovery never mints these expected identities"
    )


def _load_runtime_manifest(
    resources: ExitStack,
    tracer_so: Path,
    *,
    runtime_manifest: Path | None,
    retained_descriptor: int | None,
    retained_manifest_descriptor: int | None,
) -> RuntimeManifestAuthority:
    path = _manifest_path(tracer_so, runtime_manifest)
    try:
        path.lstat()
    except FileNotFoundError as exc:
        raise TracerRuntimeError(
            "runtime_authority_prerequisite_missing",
            _manifest_prerequisite_detail(path),
        ) from exc
    except OSError as exc:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            f"runtime manifest identity is unavailable: {exc}",
        ) from exc
    try:
        retained = resources.enter_context(
            immutable.open_retained_regular(
                path, max_bytes=RUNTIME_MANIFEST_MAX_BYTES,
            )
        )
        if retained_manifest_descriptor is not None:
            if (
                type(retained_manifest_descriptor) is not int
                or retained_manifest_descriptor < 0
            ):
                raise TracerRuntimeError(
                    "runtime_authority_invalid",
                    "retained runtime manifest descriptor must be a nonnegative integer",
                )
            try:
                supplied_metadata = os.fstat(retained_manifest_descriptor)
                opened_metadata = os.fstat(retained.descriptor)
            except OSError as exc:
                raise TracerRuntimeError(
                    "runtime_authority_changed",
                    f"retained runtime manifest descriptor is unavailable: {exc}",
                ) from exc
            if (
                not stat.S_ISREG(supplied_metadata.st_mode)
                or _descriptor_identity(supplied_metadata)
                != _descriptor_identity(opened_metadata)
                or supplied_metadata.st_size > RUNTIME_MANIFEST_MAX_BYTES
            ):
                raise TracerRuntimeError(
                    "runtime_authority_changed",
                    "runtime manifest path differs from the caller-retained descriptor",
                )
            try:
                supplied_sha256, supplied_size = _hash_retained_descriptor(
                    retained_manifest_descriptor,
                )
            except TracerRuntimeError as exc:
                raise TracerRuntimeError(
                    "runtime_authority_changed",
                    f"retained runtime manifest descriptor changed: {exc.detail}",
                ) from exc
            if (
                supplied_sha256 != retained.sha256
                or supplied_size != retained.size_bytes
            ):
                raise TracerRuntimeError(
                    "runtime_authority_changed",
                    "runtime manifest bytes differ from the caller-retained descriptor",
                )
        payload = retained.read_bytes(max_bytes=RUNTIME_MANIFEST_MAX_BYTES)
        document = immutable.strict_json_loads(
            payload, label="NVBit runtime DSO manifest",
        )
        if type(document) is not dict:
            raise immutable.ImmutableArtifactError(
                "NVBit runtime DSO manifest must be an exact JSON object"
            )
        if immutable.pretty_json_bytes(document) != payload:
            raise immutable.ImmutableArtifactError(
                "NVBit runtime DSO manifest must use sorted pretty JSON plus LF"
            )
        immutable.verify_model_seal(document)
    except TracerRuntimeError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            f"runtime manifest cannot be retained and validated: {exc}",
        ) from exc

    expected_top = {
        "authority_kind",
        "dsos",
        "model_seal_sha256",
        "runtime_library_path",
        "schema_version",
        "tracer",
    }
    if set(document) != expected_top:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest fields are not exact: "
            f"expected={sorted(expected_top)}, observed={sorted(document)}",
        )
    if document["schema_version"] != RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            f"unsupported runtime manifest schema: {document['schema_version']!r}",
        )
    if document["authority_kind"] != RUNTIME_MANIFEST_AUTHORITY_KIND:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            f"unsupported runtime authority kind: {document['authority_kind']!r}",
        )

    tracer = document["tracer"]
    if type(tracer) is not dict or set(tracer) != {"sha256", "size_bytes"}:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest tracer identity must contain only sha256 and size_bytes",
        )
    if not _is_sha256(tracer["sha256"]):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest tracer sha256 must be lowercase 64-hex",
        )
    if (
        type(tracer["size_bytes"]) is not int
        or not 1 <= tracer["size_bytes"] <= _TRACER_IDENTITY_MAX_BYTES
    ):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest tracer size_bytes is outside the admitted bound",
        )

    runtime_library_path = document["runtime_library_path"]
    if (
        type(runtime_library_path) is not str
        or not runtime_library_path
        or os.pathsep in runtime_library_path
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in runtime_library_path)
    ):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime_library_path must be one exact control-free path",
        )
    runtime_path = Path(runtime_library_path)
    if not runtime_path.is_absolute():
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime_library_path must be absolute",
        )
    if any(
        runtime_path.parts[index : index + 2] == (".claude", "worktrees")
        for index in range(len(runtime_path.parts) - 1)
    ):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime_library_path must not name a Claude worktree",
        )

    rows = document["dsos"]
    if type(rows) is not list or not 1 <= len(rows) <= _PINNED_RUNTIME_MAX_DSOS:
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest dsos must be a nonempty bounded exact list",
        )
    expectations: dict[str, RuntimeDsoExpectation] = {}
    observed_order: list[str] = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "path", "sha256", "size_bytes", "soname",
        }:
            raise TracerRuntimeError(
                "runtime_authority_invalid",
                f"runtime manifest dsos[{index}] fields are not exact",
            )
        soname = row["soname"]
        if (
            type(soname) is not str
            or not soname.startswith(_PINNED_DSO_PREFIXES)
            or "/" in soname
            or soname in {"", ".", ".."}
        ):
            raise TracerRuntimeError(
                "runtime_authority_invalid",
                f"runtime manifest dsos[{index}].soname is invalid",
            )
        relative_text = row["path"]
        try:
            relative = immutable.safe_relative_path(relative_text)
        except (TypeError, immutable.ImmutableArtifactError) as exc:
            raise TracerRuntimeError(
                "runtime_authority_invalid",
                f"runtime manifest dsos[{index}].path is invalid: {exc}",
            ) from exc
        if not _is_sha256(row["sha256"]):
            raise TracerRuntimeError(
                "runtime_authority_invalid",
                f"runtime manifest dsos[{index}].sha256 must be lowercase 64-hex",
            )
        size_bytes = row["size_bytes"]
        if (
            type(size_bytes) is not int
            or not 1 <= size_bytes <= _PINNED_RUNTIME_MAX_TOTAL_BYTES
        ):
            raise TracerRuntimeError(
                "runtime_authority_invalid",
                f"runtime manifest dsos[{index}].size_bytes is invalid",
            )
        if soname in expectations:
            raise TracerRuntimeError(
                "runtime_authority_invalid",
                f"runtime manifest repeats soname {soname!r}",
            )
        observed_order.append(soname)
        expectations[soname] = RuntimeDsoExpectation(
            soname=soname,
            relative_path=relative.as_posix(),
            sha256=row["sha256"],
            size_bytes=size_bytes,
        )
    if observed_order != sorted(observed_order, key=os.fsencode):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest dsos must be sorted by soname byte order",
        )
    if not any(name.startswith("libprotobuf.so") for name in expectations):
        raise TracerRuntimeError(
            "runtime_authority_invalid",
            "runtime manifest exact DSO closure omits libprotobuf.so",
        )

    if retained_descriptor is None:
        try:
            retained_tracer = resources.enter_context(
                immutable.open_retained_regular(
                    Path(tracer_so).resolve(), max_bytes=_TRACER_IDENTITY_MAX_BYTES,
                )
            )
            tracer_sha256 = retained_tracer.sha256
            tracer_size_bytes = retained_tracer.size_bytes
        except (OSError, immutable.ImmutableArtifactError) as exc:
            raise TracerRuntimeError(
                "tracer_path_invalid",
                f"tracer identity cannot be retained: {exc}",
            ) from exc
    else:
        tracer_sha256, tracer_size_bytes = _hash_retained_descriptor(
            retained_descriptor,
        )
    if (
        tracer_sha256 != tracer["sha256"]
        or tracer_size_bytes != tracer["size_bytes"]
    ):
        raise TracerRuntimeError(
            "runtime_authority_tracer_mismatch",
            "runtime manifest is not bound to the retained tracer: "
            f"expected={tracer['sha256']}/{tracer['size_bytes']}, "
            f"observed={tracer_sha256}/{tracer_size_bytes}",
        )

    authority = RuntimeManifestAuthority(
        path=retained.path,
        sha256=retained.sha256,
        runtime_library_path=runtime_path,
        tracer_sha256=tracer["sha256"],
        tracer_size_bytes=tracer["size_bytes"],
        dsos=MappingProxyType(expectations),
        retained=retained,
    )
    authority.verify_unchanged()
    return authority


def _configured_pb_lib(
    pb_lib: Path | None,
    environment: Mapping[str, str],
) -> Path:
    """Resolve only an explicit path; ambient NVBIT_PB_PREFIX is never authority."""

    del environment
    selected = DEFAULT_PB_LIB if pb_lib is None else Path(pb_lib)
    if not selected.is_absolute():
        raise TracerRuntimeError(
            "runtime_path_invalid",
            f"pinned protobuf runtime path must be absolute: {selected}",
        )
    return selected


def _validated_pb_lib(
    pb_lib: Path | None,
    environment: Mapping[str, str],
) -> Path:
    selected_pb_lib = _configured_pb_lib(pb_lib, environment)
    cursor = selected_pb_lib
    symlinked = None
    while cursor != cursor.parent:
        if cursor.is_symlink():
            symlinked = cursor
            break
        cursor = cursor.parent
    if symlinked is not None:
        raise TracerRuntimeError(
            "runtime_path_invalid",
            f"pinned protobuf runtime directory must not be a symlink: {symlinked}",
        )
    resolved = selected_pb_lib.resolve()
    if not resolved.is_dir():
        raise TracerRuntimeError(
            "runtime_path_invalid",
            f"pinned protobuf runtime directory is missing: {resolved}",
        )
    return resolved


def _retained_tracer_path(
    tracer_so: Path, retained_descriptor: int | None,
) -> tuple[str, tuple[int, ...]]:
    if retained_descriptor is None:
        tracer_path = str(Path(tracer_so).resolve())
        if not Path(tracer_path).is_file():
            raise TracerRuntimeError(
                "tracer_path_invalid", f"tracer .so is missing: {tracer_path}"
            )
        return tracer_path, ()
    if type(retained_descriptor) is not int or retained_descriptor < 0:
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            "retained tracer descriptor must be a nonnegative integer",
        )
    try:
        descriptor_stat = os.fstat(retained_descriptor)
    except OSError as exc:
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            f"retained tracer descriptor is unavailable: {exc}",
        ) from exc
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise TracerRuntimeError(
            "tracer_descriptor_invalid",
            "retained tracer descriptor must identify a regular file",
        )
    return f"/proc/self/fd/{retained_descriptor}", (retained_descriptor,)


def _note_cleanup_failure(error: BaseException, detail: str) -> None:
    try:
        error.add_note(detail)
    except (AttributeError, TypeError):
        pass


def _terminate_ldd_after_failure(
    process: subprocess.Popen[bytes], error: BaseException,
) -> None:
    cleanup_errors: list[BaseException] = []
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        process.wait(timeout=LDD_CLEANUP_TIMEOUT_SECONDS)
    except BaseException as exc:
        cleanup_errors.append(exc)
    for cleanup_error in cleanup_errors:
        _note_cleanup_failure(
            error,
            "ldd process cleanup failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}",
        )


def _run_ldd(
    tracer_path: str,
    *,
    env: Mapping[str, str],
    pass_fds: tuple[int, ...],
    retained_ldd: immutable.RetainedRegular | None = None,
) -> subprocess.CompletedProcess[str]:
    if retained_ldd is None:
        retained_ldd = _ACTIVE_RETAINED_LDD.get()
    if retained_ldd is None:
        try:
            with immutable.open_retained_executable(
                _LDD_EXECUTABLE
            ) as owned_ldd:
                return _run_ldd(
                    tracer_path,
                    env=env,
                    pass_fds=pass_fds,
                    retained_ldd=owned_ldd,
                )
        except immutable.ImmutableArtifactError as exc:
            raise TracerRuntimeError(
                "ldd_start_failed", f"could not retain trusted ldd: {exc}",
            ) from exc
    retained_ldd.verify_unchanged()
    command = [retained_ldd.proc_path, tracer_path]
    ldd_environment = dict(env)
    ldd_environment.update({
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    })
    for hostile_name in ("BASH_ENV", "ENV", "SHELLOPTS"):
        ldd_environment.pop(hostile_name, None)
    ldd_pass_fds = tuple(dict.fromkeys(
        (*pass_fds, retained_ldd.descriptor)
    ))
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=ldd_environment,
            pass_fds=ldd_pass_fds,
            start_new_session=True,
        )
    except (OSError, MemoryError) as exc:
        reason = (
            "runtime_resource_unavailable"
            if isinstance(exc, MemoryError) or (
                isinstance(exc, OSError)
                and exc.errno in _PROCESS_RESOURCE_ERRNOS
            )
            else "ldd_start_failed"
        )
        raise TracerRuntimeError(
            reason, f"could not execute ldd: {exc}"
        ) from exc

    stdout = process.stdout
    stderr = process.stderr
    selector: selectors.BaseSelector | None = None
    result: subprocess.CompletedProcess[str] | None = None
    primary: BaseException | None = None
    try:
        if stdout is None or stderr is None:
            raise TracerRuntimeError(
                "ldd_start_failed", "ldd output pipes were not created"
            )
        selector = selectors.DefaultSelector()
        streams = {
            "stdout": stdout,
            "stderr": stderr,
        }
        buffers = {
            "stdout": bytearray(),
            "stderr": bytearray(),
        }
        limits = {
            "stdout": LDD_MAX_STDOUT_BYTES,
            "stderr": LDD_MAX_STDERR_BYTES,
        }
        for name, stream in streams.items():
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = time.monotonic() + LDD_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TracerRuntimeError(
                    "ldd_timeout",
                    f"ldd exceeded {LDD_TIMEOUT_SECONDS:g} seconds for retained tracer",
                )
            try:
                events = selector.select(remaining)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                reason = (
                    "runtime_resource_unavailable"
                    if exc.errno in _PROCESS_RESOURCE_ERRNOS
                    else "ldd_output_read_failed"
                )
                raise TracerRuntimeError(
                    reason,
                    f"could not read bounded ldd output: {exc}",
                ) from exc
            if not events:
                raise TracerRuntimeError(
                    "ldd_timeout",
                    f"ldd exceeded {LDD_TIMEOUT_SECONDS:g} seconds for retained tracer",
                )
            for key, _events in events:
                name = key.data
                buffer = buffers[name]
                limit = limits[name]
                read_size = min(
                    _LDD_READ_CHUNK_BYTES,
                    limit - len(buffer) + 1,
                )
                try:
                    chunk = os.read(key.fd, read_size)
                except OSError as exc:
                    reason = (
                        "runtime_resource_unavailable"
                        if exc.errno in _PROCESS_RESOURCE_ERRNOS
                        else "ldd_output_read_failed"
                    )
                    raise TracerRuntimeError(
                        reason,
                        f"could not read bounded ldd {name}: {exc}",
                    ) from exc
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining_capacity = limit - len(buffer)
                buffer.extend(chunk[:remaining_capacity])
                if len(chunk) > remaining_capacity:
                    raise TracerRuntimeError(
                        "ldd_output_limit_exceeded",
                        f"ldd {name} exceeded its {limit}-byte limit",
                    )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TracerRuntimeError(
                "ldd_timeout",
                f"ldd exceeded {LDD_TIMEOUT_SECONDS:g} seconds for retained tracer",
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise TracerRuntimeError(
                "ldd_timeout",
                f"ldd exceeded {LDD_TIMEOUT_SECONDS:g} seconds for retained tracer",
            ) from exc
        result = subprocess.CompletedProcess(
            command,
            returncode,
            bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
            bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        )
    except MemoryError as exc:
        primary = TracerRuntimeError(
            "runtime_resource_unavailable",
            f"could not retain bounded ldd output: {exc}",
        )
        _terminate_ldd_after_failure(process, primary)
    except OSError as exc:
        reason = (
            "runtime_resource_unavailable"
            if exc.errno in _PROCESS_RESOURCE_ERRNOS
            else "ldd_output_read_failed"
        )
        primary = TracerRuntimeError(
            reason,
            f"could not collect bounded ldd output: {exc}",
        )
        _terminate_ldd_after_failure(process, primary)
    except BaseException as exc:
        primary = exc
        _terminate_ldd_after_failure(process, exc)
    finally:
        cleanup_errors: list[BaseException] = []
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        for stream in (stdout, stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary is not None:
                for cleanup_error in cleanup_errors:
                    _note_cleanup_failure(
                        primary,
                        "ldd stream cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    )
            else:
                primary = TracerRuntimeError(
                    "ldd_cleanup_failed",
                    "could not close ldd output streams: "
                    + "; ".join(
                        f"{type(exc).__name__}: {exc}"
                        for exc in cleanup_errors
                    ),
                )
    retained_ldd.verify_unchanged()
    if primary is not None:
        raise primary
    assert result is not None
    return result


def _run_with_retained_ldd(
    retained_ldd: immutable.RetainedRegular,
    tracer_path: str,
    *,
    env: Mapping[str, str],
    pass_fds: tuple[int, ...],
) -> subprocess.CompletedProcess[str]:
    token = _ACTIVE_RETAINED_LDD.set(retained_ldd)
    try:
        return _run_ldd(tracer_path, env=env, pass_fds=pass_fds)
    finally:
        _ACTIVE_RETAINED_LDD.reset(token)


def _ldd_output(proc: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part)


def _require_successful_ldd(proc: subprocess.CompletedProcess[str]) -> str:
    output = _ldd_output(proc)
    missing = [line.strip() for line in output.splitlines() if "not found" in line]
    if proc.returncode != 0 or missing:
        detail = missing or [proc.stderr.strip() or f"ldd exited {proc.returncode}"]
        raise TracerRuntimeError(
            "ldd_dependencies_unresolved",
            "tracer dependencies are unresolved; CUDA_INJECTION64_PATH would "
            f"silently run untraced: {detail}",
        )
    return output


def _pinned_runtime_mismatches(ldd_output: str, pb_lib: Path) -> list[str]:
    """Return protobuf-family DSOs that resolved outside ``pb_lib``."""

    mismatches: list[str] = []
    protobuf_seen = False
    for raw_line in ldd_output.splitlines():
        fields = raw_line.strip().split()
        if not fields:
            continue
        soname = fields[0]
        if not soname.startswith(_PINNED_DSO_PREFIXES):
            continue
        protobuf_seen = protobuf_seen or soname.startswith("libprotobuf.so")
        if "=>" not in fields:
            mismatches.append(f"{soname} has no resolved path")
            continue
        try:
            resolved = fields[fields.index("=>") + 1]
        except IndexError:
            mismatches.append(f"{soname} has no resolved path")
            continue
        if resolved == "not":
            continue
        try:
            Path(resolved).resolve().relative_to(pb_lib)
        except (OSError, ValueError):
            mismatches.append(f"{soname} => {resolved}")
    if not protobuf_seen:
        mismatches.append("libprotobuf.so mapping is absent")
    return mismatches


def _pinned_runtime_mappings(
    ldd_output: str, pb_lib: Path,
) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    protobuf_seen = False
    for raw_line in ldd_output.splitlines():
        fields = raw_line.strip().split()
        if not fields or not fields[0].startswith(_PINNED_DSO_PREFIXES):
            continue
        soname = fields[0]
        protobuf_seen = protobuf_seen or soname.startswith("libprotobuf.so")
        if (
            "/" in soname
            or soname in mappings
            or "=>" not in fields
            or fields.index("=>") + 1 >= len(fields)
        ):
            raise TracerRuntimeError(
                "runtime_dso_mapping_invalid", f"ambiguous ldd mapping for {soname!r}"
            )
        resolved_text = fields[fields.index("=>") + 1]
        if resolved_text == "not":
            raise TracerRuntimeError(
                "ldd_dependencies_unresolved", f"{soname} was not found"
            )
        try:
            resolved = Path(resolved_text).resolve(strict=True)
            resolved.relative_to(pb_lib)
        except (OSError, ValueError) as exc:
            raise TracerRuntimeError(
                "runtime_dso_mapping_invalid",
                f"{soname} resolved outside the pinned runtime: {resolved_text}",
            ) from exc
        mappings[soname] = resolved
    if not protobuf_seen:
        raise TracerRuntimeError(
            "runtime_dso_mapping_invalid", "libprotobuf.so mapping is absent"
        )
    if len(mappings) > _PINNED_RUNTIME_MAX_DSOS:
        raise TracerRuntimeError(
            "runtime_dso_limit_exceeded",
            f"protobuf-family DSO count {len(mappings)} exceeds "
            f"{_PINNED_RUNTIME_MAX_DSOS}",
        )
    return mappings


def _verify_runtime_manifest_mappings(
    mappings: Mapping[str, Path],
    targets_by_soname: Mapping[str, immutable.RetainedRegular],
    pb_lib: Path,
    authority: RuntimeManifestAuthority,
) -> None:
    """Bind discovered aliases and retained bytes to independent expectations."""

    expected_names = set(authority.dsos)
    observed_names = set(mappings)
    if observed_names != expected_names:
        raise TracerRuntimeError(
            "runtime_dso_authority_mismatch",
            "runtime DSO closure differs from the exact operator manifest: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unknown={sorted(observed_names - expected_names)}",
        )
    if set(targets_by_soname) != observed_names:
        raise TracerRuntimeError(
            "runtime_dso_authority_mismatch",
            "retained runtime targets do not cover the discovered DSO closure",
        )
    for soname in sorted(observed_names, key=os.fsencode):
        expectation = authority.dsos[soname]
        try:
            relative = mappings[soname].relative_to(pb_lib).as_posix()
        except ValueError as exc:
            raise TracerRuntimeError(
                "runtime_dso_authority_mismatch",
                f"{soname} target escaped the authorized runtime directory",
            ) from exc
        retained = targets_by_soname[soname]
        if (
            relative != expectation.relative_path
            or retained.sha256 != expectation.sha256
            or retained.size_bytes != expectation.size_bytes
        ):
            raise TracerRuntimeError(
                "runtime_dso_authority_mismatch",
                f"{soname} differs from independent runtime authority: "
                f"expected={expectation.relative_path}/"
                f"{expectation.sha256}/{expectation.size_bytes}, "
                f"observed={relative}/{retained.sha256}/{retained.size_bytes}",
            )
    authority.verify_unchanged()


def _alias_identity(directory_descriptor: int, soname: str) -> tuple[int, int, int]:
    try:
        observed = os.stat(
            soname, dir_fd=directory_descriptor, follow_symlinks=True,
        )
    except OSError as exc:
        raise TracerRuntimeError(
            "runtime_dso_mapping_invalid", f"could not stat {soname}: {exc}"
        ) from exc
    if not stat.S_ISREG(observed.st_mode):
        raise TracerRuntimeError(
            "runtime_dso_mapping_invalid", f"{soname} is not a regular DSO"
        )
    return observed.st_dev, observed.st_ino, observed.st_size


def _cleanup_private_runtime_tree(path: Path) -> None:
    try:
        if path.exists():
            os.chmod(path, 0o700, follow_symlinks=False)
            shutil.rmtree(path)
    except OSError as exc:
        raise TracerRuntimeError(
            "runtime_cleanup_failed",
            f"could not remove private retained runtime tree {path}: {exc}",
        ) from exc


def _close_resources_after_failure(
    resources: ExitStack,
    error: BaseException,
) -> None:
    try:
        resources.close()
    except BaseException as cleanup_error:
        try:
            error.add_note(f"retained runtime cleanup failed: {cleanup_error}")
        except (AttributeError, TypeError):
            pass


def _open_private_runtime(
    path: str | os.PathLike[str],
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
    label: str,
) -> int:
    try:
        return os.open(path, flags, mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in {
            errno.EMFILE,
            errno.ENFILE,
            errno.ENOMEM,
            errno.ENOSPC,
            errno.EDQUOT,
        }:
            raise TracerRuntimeError(
                "runtime_resource_unavailable",
                f"could not open {label}: {exc}",
            ) from exc
        raise TracerRuntimeError(
            "runtime_dso_generation_changed",
            f"private retained runtime identity changed while opening {label}: {exc}",
        ) from exc


def _copy_retained_regular(
    retained: immutable.RetainedRegular,
    destination_directory: int,
    destination_name: str,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    descriptor = _open_private_runtime(
        destination_name,
        flags,
        0o600,
        dir_fd=destination_directory,
        label=f"private runtime DSO {destination_name}",
    )
    try:
        retained.verify_unchanged()
        os.lseek(retained.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(retained.descriptor, _RUNTIME_COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied += len(chunk)
            if copied > retained.size_bytes:
                raise TracerRuntimeError(
                    "runtime_dso_generation_changed",
                    f"retained DSO grew while copying: {retained.path}",
                )
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(descriptor, chunk[offset:])
                if written <= 0:
                    raise OSError(errno.EIO, "short private runtime DSO write")
                offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        if copied != retained.size_bytes or digest.hexdigest() != retained.sha256:
            raise TracerRuntimeError(
                "runtime_dso_generation_changed",
                f"retained DSO bytes changed while copying: {retained.path}",
            )
        retained.verify_unchanged()
    finally:
        os.lseek(retained.descriptor, 0, os.SEEK_SET)
        os.close(descriptor)


def _private_runtime_tree(
    resources: ExitStack,
    targets_by_soname: Mapping[str, immutable.RetainedRegular],
) -> tuple[
    Path,
    immutable.PinnedDirectoryReader,
    immutable.PinnedTreeSnapshot,
    Mapping[str, tuple[int, int, int]],
]:
    """Build a non-writable descriptor tree from authenticated retained bytes."""

    try:
        private_path = Path(tempfile.mkdtemp(prefix="nvbit-runtime-"))
    except (OSError, MemoryError) as exc:
        raise TracerRuntimeError(
            "runtime_resource_unavailable",
            f"could not allocate private retained runtime tree: {exc}",
        ) from exc
    resources.callback(_cleanup_private_runtime_tree, private_path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        directory_descriptor = _open_private_runtime(
            private_path,
            flags,
            label="private retained runtime directory",
        )
    except BaseException as error:
        try:
            private_path.rmdir()
        except OSError as cleanup_error:
            _note_cleanup_failure(
                error,
                "private retained runtime cleanup after open failure failed: "
                f"{cleanup_error}",
            )
        raise
    try:
        private_total_bytes = 0
        for soname, retained in sorted(targets_by_soname.items()):
            private_total_bytes += retained.size_bytes
            if private_total_bytes > _PINNED_RUNTIME_MAX_TOTAL_BYTES:
                raise TracerRuntimeError(
                    "runtime_dso_limit_exceeded",
                    "private protobuf-family DSO bytes "
                    f"{private_total_bytes} exceed "
                    f"{_PINNED_RUNTIME_MAX_TOTAL_BYTES}",
                )
            _copy_retained_regular(
                retained, directory_descriptor, soname,
            )
        os.fchmod(directory_descriptor, 0o555)
        os.fsync(directory_descriptor)
    except OSError as exc:
        if exc.errno in {
            errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC, errno.EDQUOT,
        }:
            raise TracerRuntimeError(
                "runtime_resource_unavailable",
                f"could not construct private retained runtime tree: {exc}",
            ) from exc
        raise
    finally:
        os.close(directory_descriptor)

    runtime_reader = resources.enter_context(
        immutable.PinnedDirectoryReader(private_path)
    )
    runtime_tree = resources.enter_context(
        runtime_reader.retain_tree_generation(
            max_files=_PINNED_RUNTIME_MAX_DSOS,
            max_directories=1,
            max_total_file_bytes=_PINNED_RUNTIME_MAX_TOTAL_BYTES,
            max_total_relative_path_bytes=1 << 20,
        )
    )
    files = dict(runtime_tree.files)
    expected_names = set(targets_by_soname)
    if set(files) != expected_names or runtime_tree.directories[0][0] != ".":
        raise TracerRuntimeError(
            "runtime_dso_generation_changed",
            "private retained runtime tree inventory differs",
        )
    alias_identities = MappingProxyType({
        soname: _alias_identity(runtime_reader.descriptor, soname)
        for soname in sorted(targets_by_soname)
    })
    for soname, retained in targets_by_soname.items():
        alias_identity = alias_identities[soname]
        if alias_identity[2] != retained.size_bytes:
            raise TracerRuntimeError(
                "runtime_dso_generation_changed",
                f"private retained runtime DSO size differs: {soname}",
            )
        snapshot = runtime_reader.read_snapshot(
            soname,
            max_bytes=retained.size_bytes,
            expected_metadata=files[soname],
        )
        if hashlib.sha256(snapshot.payload).hexdigest() != retained.sha256:
            raise TracerRuntimeError(
                "runtime_dso_generation_changed",
                f"private retained runtime DSO hash differs: {soname}",
            )
    return private_path, runtime_reader, runtime_tree, alias_identities


@dataclass
class RetainedTracerRuntime:
    """Descriptor-backed protobuf runtime retained through one child execution."""

    environment: dict[str, str]
    dso_sha256: Mapping[str, str]
    _runtime_reader: immutable.PinnedDirectoryReader
    _runtime_tree: immutable.PinnedTreeSnapshot
    _monitor: immutable.PinnedPathSetMonitor
    _runtime_authority: RuntimeManifestAuthority
    _targets_by_soname: Mapping[str, immutable.RetainedRegular]
    _alias_identities: Mapping[str, tuple[int, int, int]]
    _resources: ExitStack = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def runtime_manifest_path(self) -> Path:
        return self._runtime_authority.path

    @property
    def runtime_manifest_sha256(self) -> str:
        return self._runtime_authority.sha256

    @property
    def pass_fds(self) -> tuple[int, ...]:
        if self._closed:
            raise TracerRuntimeError(
                "runtime_capability_closed", "retained runtime is closed"
            )
        return (self._runtime_reader.descriptor,)

    def verify_unchanged(self) -> None:
        if self._closed:
            raise TracerRuntimeError(
                "runtime_capability_closed", "retained runtime is closed"
            )
        try:
            self._runtime_authority.verify_unchanged()
            self._monitor.verify_unchanged()
            self._runtime_reader.verify_path()
            observed_tree = self._runtime_reader.snapshot_tree(
                max_files=_PINNED_RUNTIME_MAX_DSOS,
                max_directories=1,
                max_total_file_bytes=_PINNED_RUNTIME_MAX_TOTAL_BYTES,
                max_total_relative_path_bytes=1 << 20,
            )
            if observed_tree != self._runtime_tree:
                raise TracerRuntimeError(
                    "runtime_dso_generation_changed",
                    "private retained runtime tree changed",
                )
            for soname, target in self._targets_by_soname.items():
                target.verify_unchanged()
                if _alias_identity(self._runtime_reader.descriptor, soname) != (
                    self._alias_identities[soname]
                ):
                    raise TracerRuntimeError(
                        "runtime_dso_generation_changed",
                        f"private runtime alias changed: {soname}",
                    )
            self._monitor.verify_unchanged()
            self._runtime_reader.verify_path()
            if self._runtime_reader.snapshot_tree(
                max_files=_PINNED_RUNTIME_MAX_DSOS,
                max_directories=1,
                max_total_file_bytes=_PINNED_RUNTIME_MAX_TOTAL_BYTES,
                max_total_relative_path_bytes=1 << 20,
            ) != self._runtime_tree:
                raise TracerRuntimeError(
                    "runtime_dso_generation_changed",
                    "private retained runtime tree changed during verification",
                )
            self._runtime_authority.verify_unchanged()
        except TracerRuntimeError:
            raise
        except immutable.ImmutableArtifactLimitError as exc:
            raise TracerRuntimeError(
                "runtime_dso_limit_exceeded", str(exc)
            ) from exc
        except (immutable.ImmutableArtifactResourceError, MemoryError) as exc:
            raise TracerRuntimeError(
                "runtime_resource_unavailable", str(exc)
            ) from exc
        except (immutable.ImmutableArtifactError, OSError) as exc:
            raise TracerRuntimeError(
                "runtime_dso_generation_changed", str(exc)
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._resources.close()
        except immutable.ImmutableArtifactLimitError as exc:
            raise TracerRuntimeError(
                "runtime_dso_limit_exceeded", str(exc)
            ) from exc
        except (immutable.ImmutableArtifactResourceError, MemoryError) as exc:
            raise TracerRuntimeError(
                "runtime_resource_unavailable", str(exc)
            ) from exc
        except (immutable.ImmutableArtifactError, OSError) as exc:
            raise TracerRuntimeError(
                "runtime_dso_generation_changed", str(exc)
            ) from exc

    def __enter__(self) -> "RetainedTracerRuntime":
        try:
            self.verify_unchanged()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                _note_cleanup_failure(
                    error,
                    "retained tracer runtime cleanup after entry failure failed: "
                    f"{cleanup_error}",
                )
            raise
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        if exc is not None:
            try:
                self.verify_unchanged()
            except BaseException as verification_error:
                try:
                    exc.add_note(
                        "retained tracer runtime revalidation failed: "
                        f"{verification_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            try:
                self.close()
            except BaseException as cleanup_error:
                try:
                    exc.add_note(
                        f"retained tracer runtime cleanup failed: {cleanup_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            return False
        try:
            self.verify_unchanged()
        finally:
            self.close()
        return False


def retained_tracer_runtime_env(
    tracer_so: Path,
    *,
    base_env: Mapping[str, str] | None = None,
    pb_lib: Path | None = None,
    runtime_manifest: Path | None = None,
    retained_descriptor: int | None = None,
    retained_manifest_descriptor: int | None = None,
) -> RetainedTracerRuntime:
    """Authenticate and retain an independently authorized runtime for child exec.

    A retained operator manifest binds the exact tracer, runtime directory, ldd
    alias closure, DSO target paths, sizes, and SHA-256 values before discovery.
    The first bounded ``ldd`` may only resolve through that manifest-bound source
    directory. Exact authenticated DSO bytes are copied into a private,
    non-writable descriptor-backed tree, and a second bounded ``ldd`` resolves
    only through that retained tree. Callers must pass ``pass_fds`` to the child
    and keep this context open until the child has exited.
    """

    supplied_snapshot = dict(os.environ if base_env is None else base_env)
    tracer_path, tracer_pass_fds = _retained_tracer_path(
        tracer_so, retained_descriptor,
    )
    environment_snapshot = sanitized_capture_environment(supplied_snapshot)
    resources = ExitStack()
    try:
        runtime_authority = _load_runtime_manifest(
            resources,
            tracer_so,
            runtime_manifest=runtime_manifest,
            retained_descriptor=retained_descriptor,
            retained_manifest_descriptor=retained_manifest_descriptor,
        )
        authorized_pb_lib = _validated_pb_lib(
            runtime_authority.runtime_library_path, {},
        )
        if pb_lib is None:
            selected_pb_lib = authorized_pb_lib
        else:
            selected_pb_lib = _validated_pb_lib(pb_lib, environment_snapshot)
            if selected_pb_lib != authorized_pb_lib:
                raise TracerRuntimeError(
                    "runtime_authority_prefix_mismatch",
                    "explicit protobuf runtime differs from independent authority: "
                    f"expected={authorized_pb_lib}, observed={selected_pb_lib}",
                )
        retained_ldd = resources.enter_context(
            immutable.open_retained_executable(_LDD_EXECUTABLE)
        )
        source_reader = resources.enter_context(
            immutable.PinnedDirectoryReader(selected_pb_lib)
        )
        source_proc_path = f"/proc/self/fd/{source_reader.descriptor}"
        source_env = dict(environment_snapshot)
        source_env["LD_LIBRARY_PATH"] = source_proc_path
        discovery_pass_fds = tuple(dict.fromkeys(
            (*tracer_pass_fds, source_reader.descriptor)
        ))
        discovery = _require_successful_ldd(_run_with_retained_ldd(
            retained_ldd,
            tracer_path,
            env=source_env,
            pass_fds=discovery_pass_fds,
        ))
        mappings = _pinned_runtime_mappings(discovery, selected_pb_lib)

        retained_by_path: dict[Path, immutable.RetainedRegular] = {}
        targets_by_soname: dict[str, immutable.RetainedRegular] = {}
        total_bytes = 0
        for soname, target_path in sorted(mappings.items()):
            retained = retained_by_path.get(target_path)
            if retained is None:
                retained = resources.enter_context(
                    immutable.open_retained_regular(target_path)
                )
                retained_by_path[target_path] = retained
                total_bytes += retained.size_bytes
                if total_bytes > _PINNED_RUNTIME_MAX_TOTAL_BYTES:
                    raise TracerRuntimeError(
                        "runtime_dso_limit_exceeded",
                        f"protobuf-family DSO bytes {total_bytes} exceed "
                        f"{_PINNED_RUNTIME_MAX_TOTAL_BYTES}",
                    )
            targets_by_soname[soname] = retained

        _verify_runtime_manifest_mappings(
            mappings,
            targets_by_soname,
            selected_pb_lib,
            runtime_authority,
        )
        target_relpaths = tuple(sorted(
            target.relative_to(selected_pb_lib.parent).as_posix()
            for target in retained_by_path
        ))
        monitor = resources.enter_context(immutable.PinnedPathSetMonitor(
            selected_pb_lib.parent,
            regular_paths=target_relpaths,
            recursive_directories=(),
            filtered_directories={
                selected_pb_lib.name: tuple(sorted(mappings)),
            },
            max_files=_PINNED_RUNTIME_MAX_TREE_FILES,
            max_directories=4,
            max_total_file_bytes=_PINNED_RUNTIME_TREE_MAX_TOTAL_BYTES,
            max_total_relative_path_bytes=1 << 20,
        ))
        for soname, retained in targets_by_soname.items():
            observed = _alias_identity(source_reader.descriptor, soname)
            expected = (
                retained.snapshot[0], retained.snapshot[1], retained.size_bytes,
            )
            if observed != expected:
                raise TracerRuntimeError(
                    "runtime_dso_mapping_invalid",
                    f"{soname} no longer names its retained DSO target",
                )
        monitor.verify_unchanged()

        (
            private_path,
            runtime_reader,
            runtime_tree,
            alias_identities,
        ) = _private_runtime_tree(resources, targets_by_soname)
        runtime_proc_path = f"/proc/self/fd/{runtime_reader.descriptor}"
        env = dict(environment_snapshot)
        env["LD_LIBRARY_PATH"] = runtime_proc_path
        authenticated_pass_fds = tuple(dict.fromkeys(
            (*tracer_pass_fds, runtime_reader.descriptor)
        ))
        authenticated_output = _require_successful_ldd(_run_with_retained_ldd(
            retained_ldd,
            tracer_path,
            env=env,
            pass_fds=authenticated_pass_fds,
        ))
        authenticated = _pinned_runtime_mappings(
            authenticated_output, private_path,
        )
        if set(authenticated) != set(mappings):
            raise TracerRuntimeError(
                "runtime_dso_generation_changed",
                "protobuf-family ldd mappings changed in the private retained tree",
            )
        for soname, path in authenticated.items():
            observed = path.stat()
            if (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
            ) != alias_identities[soname]:
                raise TracerRuntimeError(
                    "runtime_dso_generation_changed",
                    f"ldd did not resolve retained private DSO alias: {soname}",
                )
        monitor.verify_unchanged()

        runtime = RetainedTracerRuntime(
            environment=env,
            dso_sha256=MappingProxyType({
                soname: retained.sha256
                for soname, retained in sorted(targets_by_soname.items())
            }),
            _runtime_reader=runtime_reader,
            _runtime_tree=runtime_tree,
            _monitor=monitor,
            _runtime_authority=runtime_authority,
            _targets_by_soname=MappingProxyType(targets_by_soname),
            _alias_identities=alias_identities,
            _resources=resources.pop_all(),
        )
        try:
            runtime.verify_unchanged()
        except BaseException as error:
            try:
                runtime.close()
            except BaseException as cleanup_error:
                _note_cleanup_failure(
                    error,
                    "retained tracer runtime cleanup after construction failure "
                    f"failed: {cleanup_error}",
                )
            raise
        return runtime
    except TracerRuntimeError as exc:
        _close_resources_after_failure(resources, exc)
        raise
    except immutable.ImmutableArtifactLimitError as exc:
        translated = TracerRuntimeError(
            "runtime_dso_limit_exceeded", str(exc)
        )
        _close_resources_after_failure(resources, translated)
        raise translated from exc
    except (immutable.ImmutableArtifactResourceError, MemoryError) as exc:
        translated = TracerRuntimeError(
            "runtime_resource_unavailable", str(exc)
        )
        _close_resources_after_failure(resources, translated)
        raise translated from exc
    except (immutable.ImmutableArtifactError, OSError) as exc:
        translated = TracerRuntimeError(
            "runtime_dso_generation_changed", str(exc)
        )
        _close_resources_after_failure(resources, translated)
        raise translated from exc


def sanitized_capture_environment(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Remove ambient loader and tracer controls before setting a capture contract."""

    source = os.environ if base_env is None else base_env
    return {
        name: value
        for name, value in source.items()
        if name not in NVBIT_CAPTURE_ENVIRONMENT_CONTROLS
    }


def tracer_runtime_env(
    tracer_so: Path,
    *,
    base_env: Mapping[str, str] | None = None,
    pb_lib: Path | None = None,
    retained_descriptor: int | None = None,
) -> dict[str, str]:
    """Return only a bounded compatibility preflight environment.

    This helper does not confer runtime DSO authority and must never be used for
    a capture child. Security-sensitive execution must use
    :func:`retained_tracer_runtime_env`, whose independently supplied manifest,
    retained directory descriptor, and DSO capabilities survive through child
    exit.
    """

    supplied_snapshot = dict(os.environ if base_env is None else base_env)
    tracer_path, tracer_pass_fds = _retained_tracer_path(
        tracer_so, retained_descriptor,
    )
    env = sanitized_capture_environment(supplied_snapshot)
    selected_pb_lib = _validated_pb_lib(pb_lib, env)
    env["LD_LIBRARY_PATH"] = str(selected_pb_lib)
    try:
        with immutable.open_retained_executable(_LDD_EXECUTABLE) as retained_ldd:
            output = _require_successful_ldd(_run_with_retained_ldd(
                retained_ldd,
                tracer_path,
                env=env,
                pass_fds=tracer_pass_fds,
            ))
    except immutable.ImmutableArtifactError as exc:
        raise TracerRuntimeError(
            "ldd_start_failed", f"could not retain trusted ldd: {exc}",
        ) from exc
    wrong_prefix = _pinned_runtime_mismatches(output, selected_pb_lib)
    if wrong_prefix:
        raise TracerRuntimeError(
            "runtime_dso_mapping_invalid",
            "tracer protobuf dependencies did not resolve from the pinned "
            f"runtime {selected_pb_lib}: {wrong_prefix}",
        )
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracer_so", type=Path)
    parser.add_argument("--pb-lib", type=Path)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        help=(
            "absolute operator-supplied exact DSO manifest; defaults to the "
            "nonambient sidecar path derived from tracer_so"
        ),
    )
    args = parser.parse_args(argv)
    try:
        with retained_tracer_runtime_env(
            args.tracer_so,
            pb_lib=args.pb_lib,
            runtime_manifest=args.runtime_manifest,
        ) as runtime:
            manifest_path = runtime.runtime_manifest_path
            manifest_sha256 = runtime.runtime_manifest_sha256
            dso_count = len(runtime.dso_sha256)
    except TracerRuntimeError as exc:
        print(f"tracer_runtime: FAIL {exc}", file=sys.stderr)
        return 1
    print(
        "tracer_runtime: ok "
        f"manifest={manifest_path} manifest_sha256={manifest_sha256} "
        f"dso_count={dso_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
