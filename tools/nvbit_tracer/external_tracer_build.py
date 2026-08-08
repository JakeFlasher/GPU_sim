#!/usr/bin/env python3
"""Build and verify one immutable external sm_86 NVBit tracer bundle.

The producer reads source bytes only from a captured clean SHA-1 commit through
``git cat-file``.  It never discovers or consumes a checkout-local build
artifact, and it publishes exactly one no-replace four-file directory.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import os
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import jsonschema

from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer.external_tracer import (
    EXTERNAL_TRACER_MAX_BYTES,
    EXTERNAL_TRACER_SOURCE_KIND,
    ExternalTracerAuthorityError,
    require_external_non_git_artifact,
    require_external_non_git_root,
    require_external_tracer_authority,
    verify_retained_tracer_authority,
)


SOURCE_SCHEMA_VERSION = "nvbit_external_tracer_source_manifest.v1"
BUILD_SCHEMA_VERSION = "nvbit_external_tracer_build_receipt.v3"
ARTIFACT_SCHEMA_VERSION = "nvbit_external_tracer_artifact_identity.v1"
SOURCE_SCHEMA_NAME = "nvbit_external_tracer_source_manifest.schema.json"
BUILD_SCHEMA_NAME = "nvbit_external_tracer_build_receipt.schema.json"
ARTIFACT_SCHEMA_NAME = "nvbit_external_tracer_artifact_identity.schema.json"
SOURCE_MANIFEST_NAME = "source_manifest.json"
BUILD_RECEIPT_NAME = "build_receipt.json"
ARTIFACT_IDENTITY_NAME = "artifact_identity.json"
TRACER_NAME = "tracer_tool.so"
CANDIDATE_AUTHORITY_KIND = "candidate_observed_sha256.v1"
REVIEWED_AUTHORITY_KIND = "independent_reviewed_candidate_sha256.v1"
BUNDLE_FILES = (
    ARTIFACT_IDENTITY_NAME,
    BUILD_RECEIPT_NAME,
    SOURCE_MANIFEST_NAME,
    TRACER_NAME,
)
CONTRACT = {
    "architecture": "sm_86",
    "nvbit_release": "NVBit1.8",
    "binary_contract": "binary86",
    "binary_version": 86,
    "trace_contract": "sm86_addr_carry_shmem_split_v2",
    "supported_trace_schema_versions": [1, 2, 3, 6, 8],
}
SOURCE_PATHSPECS = (
    "tools/nvbit_tracer/Makefile",
    "tools/nvbit_tracer/isa_def",
    "tools/nvbit_tracer/metric_extractor/tier2_sha256.h",
    "tools/nvbit_tracer/nvbit_release/core",
    "tools/nvbit_tracer/tracer_tool",
    "tools/nvbit_tracer/traces_enhanced",
)
REQUIRED_SOURCE_PATHS = frozenset(
    {
        "tools/nvbit_tracer/Makefile",
        "tools/nvbit_tracer/tracer_tool/Makefile",
        "tools/nvbit_tracer/traces_enhanced/Makefile",
    }
)
GENERATED_PROTO_PREFIXES = (
    "tools/nvbit_tracer/traces_enhanced/pb_trace/include/",
    "tools/nvbit_tracer/traces_enhanced/pb_trace/src/",
)
PROHIBITED_TRACKED_OUTPUTS = frozenset(
    {
        "tools/nvbit_tracer/tracer_tool/tracer_tool.so",
        "tools/nvbit_tracer/tracer_tool/trace_printer",
    }
)
TOOL_ROLES = (
    "git",
    "bash",
    "make",
    "nvcc",
    "ptxas",
    "cuobjdump",
    "host_cxx",
    "protoc",
    "pkg_config",
)
TOOL_ALIASES = {
    "git": "git",
    "bash": "bash",
    "make": "make",
    "nvcc": "nvcc",
    "ptxas": "ptxas",
    "cuobjdump": "cuobjdump",
    "host_cxx": "g++-14",
    "protoc": "protoc",
    "pkg_config": "pkg-config",
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_MAKE_PATH = re.compile(r"/[A-Za-z0-9._+/-]+\Z")
_CUOBJDUMP_ELF_LINE = re.compile(
    rb"^ELF file[ \t]+[0-9]+:[ \t]+.*\b(sm_[0-9]+[a-z]?)\.cubin[ \t]*$"
)
_LOCAL_INCLUDE = re.compile(rb"^[ \t]*#[ \t]*include[ \t]*\"([^\"\r\n]+)\"", re.MULTILINE)
EXTERNAL_QUOTED_HEADERS = frozenset({"cuda.h"})
MAX_GIT_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SOURCE_FILES = 4096
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_TOOL_BYTES = 512 * 1024 * 1024
MAX_PROTOBUF_FILES = 16384
MAX_PROTOBUF_DIRECTORIES = 4096
MAX_PROTOBUF_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 900
MAX_TIMEOUT_SECONDS = 3600
DEFAULT_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
PROCESS_DRAIN_SECONDS = 5.0
SCHEMA_DIR_RELATIVE = Path("workloads/simbench_bw/schemas")


class ExternalTracerBuildAuthorityError(RuntimeError):
    """Repository, source, tool, schema, or publication authority failed."""


class ExternalTracerBuildProcessError(RuntimeError):
    """The authorized external operation failed or exceeded one deadline."""


@dataclass(frozen=True)
class OperationDeadline:
    timeout_seconds: int
    expires_at: float

    @classmethod
    def start(cls, timeout_seconds: int) -> "OperationDeadline":
        return cls(timeout_seconds, time.monotonic() + timeout_seconds)

    def remaining(self, *, label: str) -> float:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0:
            raise ExternalTracerBuildProcessError(
                "end-to-end timeout exceeded: "
                f"{self.timeout_seconds}s during {label}"
            )
        return remaining

    def check(self, *, label: str) -> None:
        self.remaining(label=label)


@contextlib.contextmanager
def _deadline_alarm(
    deadline: OperationDeadline, *, label: str
) -> Iterable[None]:
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        _authority("end-to-end operation deadline requires POSIX setitimer")
    remaining = deadline.remaining(label=label)
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        _authority("end-to-end operation deadline found an active SIGALRM timer")

    def expire(_signum: int, _frame: Any) -> NoReturn:
        raise ExternalTracerBuildProcessError(
            "end-to-end timeout exceeded: "
            f"{deadline.timeout_seconds}s during {label}"
        )

    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    except (OSError, ValueError) as exc:
        _authority(f"cannot arm end-to-end operation deadline: {exc}")
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class BuildRequest:
    repo_root: Path
    build_root: Path
    bundle_dir: Path
    git: Path
    bash: Path
    make: Path
    nvcc: Path
    ptxas: Path
    cuobjdump: Path
    host_cxx: Path
    protoc: Path
    pkg_config: Path
    protobuf_prefix: Path
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_OUTPUT_BYTES


@dataclass(frozen=True)
class ProcessReceipt:
    argv: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    returncode: int
    stdout_sha256: str
    stdout_size_bytes: int
    stderr_sha256: str
    stderr_size_bytes: int


@dataclass(frozen=True)
class OutputValidation:
    elf: Mapping[str, Any]
    cuda: Mapping[str, Any]


@dataclass(frozen=True)
class SourceSnapshot:
    commit: str
    tree: str
    entries: tuple[dict[str, Any], ...]
    payloads: Mapping[str, bytes]
    inventory_sha256: str
    total_size_bytes: int


@dataclass(frozen=True)
class ToolSet:
    retained: Mapping[str, immutable.RetainedExecutable]
    identities: tuple[dict[str, Any], ...]
    inventory_sha256: str


@dataclass(frozen=True)
class ExtractedInputGeneration:
    source: immutable.PinnedPathSetMonitor
    generated_protobuf: immutable.PinnedPathSetMonitor


@dataclass(frozen=True)
class ProtobufSnapshot:
    prefix: Path
    entries: tuple[dict[str, Any], ...]
    inventory_sha256: str
    file_count: int
    total_size_bytes: int


def _authority(message: str) -> NoReturn:
    raise ExternalTracerBuildAuthorityError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha1_blob(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _authority(f"{label} must be one full lowercase SHA-256")
    return value


def _require_make_safe_path(path: Path, *, label: str) -> None:
    value = os.fspath(path)
    if _SAFE_MAKE_PATH.fullmatch(value) is None:
        _authority(
            f"{label} contains syntax unsafe for exact GNU Make expansion"
        )


def _absolute_lexical(path: Path | str, *, label: str) -> Path:
    candidate = Path(path)
    raw = os.fspath(candidate)
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw):
        _authority(f"{label} contains a control character")
    if not candidate.is_absolute():
        _authority(f"{label} must be an absolute lexical path")
    if any(part in {".", ".."} for part in candidate.parts):
        _authority(f"{label} must not contain dot path components")
    if any(
        candidate.parts[index : index + 2] == (".claude", "worktrees")
        for index in range(len(candidate.parts) - 1)
    ):
        _authority(f"{label} must not name a Claude worktree")
    return candidate


def _require_lexically_external(
    path: Path,
    *,
    repo_root: Path,
    label: str,
) -> None:
    """Reject repository-contained caller paths before touching the candidate."""

    try:
        path.relative_to(repo_root)
    except ValueError:
        pass
    else:
        _authority(
            f"{label} must be outside the repository before any identity access"
        )
    if ".git" in path.parts:
        _authority(f"{label} must not be inside a Git metadata directory")


def _preflight_external_inputs(request: BuildRequest, *, repo_root: Path) -> None:
    tool_paths = {
        "git": request.git,
        "bash": request.bash,
        "make": request.make,
        "nvcc": request.nvcc,
        "ptxas": request.ptxas,
        "cuobjdump": request.cuobjdump,
        "host_cxx": request.host_cxx,
        "protoc": request.protoc,
        "pkg_config": request.pkg_config,
    }
    for role, raw_path in tool_paths.items():
        path = _absolute_lexical(raw_path, label=f"{role} tool")
        _require_make_safe_path(path, label=f"{role} tool path")
        _require_lexically_external(
            path, repo_root=repo_root, label=f"{role} tool"
        )
    protobuf_prefix = _absolute_lexical(
        request.protobuf_prefix, label="protobuf prefix"
    )
    _require_make_safe_path(protobuf_prefix, label="protobuf prefix")
    _require_lexically_external(
        protobuf_prefix,
        repo_root=repo_root,
        label="protobuf prefix",
    )


def _validate_timeout_seconds(timeout_seconds: int) -> None:
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        _authority(f"timeout_seconds must be in [1,{MAX_TIMEOUT_SECONDS}]")


def _validate_request_bounds(request: BuildRequest) -> None:
    _validate_timeout_seconds(request.timeout_seconds)
    if (
        type(request.max_output_bytes) is not int
        or not 1 <= request.max_output_bytes <= MAX_OUTPUT_BYTES
    ):
        _authority(
            f"max_output_bytes must be in [1,{MAX_OUTPUT_BYTES}]"
        )


_PROCESS_CONTAINMENT_LOCK = threading.Lock()
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37


def _subreaper_state() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    observed = ctypes.c_int()
    result = libc.prctl(
        _PR_GET_CHILD_SUBREAPER,
        ctypes.byref(observed),
        0,
        0,
        0,
    )
    if result != 0:
        error = ctypes.get_errno()
        _authority(
            "Linux descendant containment requires PR_GET_CHILD_SUBREAPER: "
            f"errno={error}"
        )
    return observed.value


def _set_subreaper(enabled: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(_PR_SET_CHILD_SUBREAPER, enabled, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        _authority(
            "Linux descendant containment requires PR_SET_CHILD_SUBREAPER: "
            f"errno={error}"
        )


def _process_children(pid: int) -> set[int]:
    task_root = Path(f"/proc/{pid}/task")
    try:
        tasks = tuple(task_root.iterdir())
    except FileNotFoundError:
        return set()
    except OSError as exc:
        _authority(f"cannot inspect descendant containment for pid {pid}: {exc}")
    children: set[int] = set()
    for task in tasks:
        try:
            payload = (task / "children").read_text(encoding="ascii").strip()
        except FileNotFoundError:
            continue
        except OSError as exc:
            _authority(
                f"cannot inspect descendant containment task {task.name}: {exc}"
            )
        if not payload:
            continue
        try:
            children.update(int(value) for value in payload.split())
        except ValueError as exc:
            _authority(f"malformed procfs descendant inventory: {exc}")
    return children


def _process_state(pid: int) -> str | None:
    try:
        payload = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except OSError as exc:
        _authority(f"cannot inspect contained process state for pid {pid}: {exc}")
    for line in payload.splitlines():
        if line.startswith("State:"):
            fields = line.split()
            if len(fields) >= 2:
                return fields[1]
    _authority(f"contained process status lacks State for pid {pid}")


class _DescendantContainment:
    """Single-threaded Linux subreaper containment for one process tree."""

    def __init__(self, *, deadline: float, timeout_seconds: int) -> None:
        self.deadline = deadline
        self.timeout_seconds = timeout_seconds
        self.previous_subreaper: int | None = None
        self.previous_signal_handlers: dict[int, Any] = {}
        self.process: subprocess.Popen[bytes] | None = None
        self.locked = False

    def __enter__(self) -> "_DescendantContainment":
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not _PROCESS_CONTAINMENT_LOCK.acquire(timeout=remaining):
            raise ExternalTracerBuildProcessError(
                "end-to-end timeout exceeded before process containment launch: "
                f"{self.timeout_seconds}s"
            )
        self.locked = True
        try:
            task_root = Path("/proc/self/task")
            if not task_root.is_dir():
                _authority("process launch requires Linux procfs descendant containment")
            try:
                task_count = sum(1 for _member in task_root.iterdir())
            except OSError as exc:
                _authority(f"cannot inspect process containment threads: {exc}")
            if task_count != 1:
                _authority(
                    "process launch requires one thread for exclusive descendant "
                    f"containment, observed={task_count}"
                )
            existing = _process_children(os.getpid())
            if existing:
                _authority(
                    "process launch cannot prove an exclusive descendant set: "
                    f"preexisting_children={sorted(existing)}"
                )
            self.previous_subreaper = _subreaper_state()
            _set_subreaper(1)
            if _subreaper_state() != 1:
                _authority("kernel did not establish descendant subreaper containment")
            for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT):
                self.previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._interrupt)
            if time.monotonic() >= self.deadline:
                raise ExternalTracerBuildProcessError(
                    "end-to-end timeout exceeded before contained process launch: "
                    f"{self.timeout_seconds}s"
                )
            return self
        except BaseException:
            self._restore()
            raise

    def _interrupt(self, signum: int, _frame: Any) -> NoReturn:
        raise ExternalTracerBuildProcessError(
            f"contained process operation interrupted by signal {signum}"
        )

    def attach(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process

    def reap_exited_descendants(self) -> None:
        while True:
            try:
                pid, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid <= 0:
                return

    def descendants(self) -> set[int]:
        roots = _process_children(os.getpid())
        descendants: set[int] = set()
        frontier = list(roots)
        while frontier:
            pid = frontier.pop()
            if pid in descendants:
                continue
            descendants.add(pid)
            frontier.extend(_process_children(pid) - descendants)
        return descendants

    def _await_stopped_or_gone(self, pid: int, *, expires_at: float) -> None:
        while True:
            state = _process_state(pid)
            if state is None or state in {"T", "t", "Z", "X"}:
                return
            if time.monotonic() >= expires_at:
                raise ExternalTracerBuildProcessError(
                    f"cannot freeze contained process before cleanup deadline: pid={pid}"
                )
            time.sleep(0.005)

    def terminate(self) -> None:
        cleanup_expires = time.monotonic() + PROCESS_DRAIN_SECONDS
        frozen: set[int] = set()
        while True:
            observed = self.descendants()
            pending = observed - frozen
            if not pending:
                break
            for pid in sorted(pending):
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    continue
                except OSError as exc:
                    raise ExternalTracerBuildProcessError(
                        f"cannot freeze contained process {pid}: {exc}"
                    ) from exc
                self._await_stopped_or_gone(pid, expires_at=cleanup_expires)
                frozen.add(pid)
            if time.monotonic() >= cleanup_expires:
                raise ExternalTracerBuildProcessError(
                    "cannot stabilize descendant containment before cleanup deadline"
                )

        for pid in sorted(self.descendants(), reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise ExternalTracerBuildProcessError(
                    f"cannot terminate contained process {pid}: {exc}"
                ) from exc

        process = self.process
        if process is not None and process.returncode is None:
            remaining = max(0.001, cleanup_expires - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise ExternalTracerBuildProcessError(
                    "contained process leader did not terminate after SIGKILL"
                ) from exc

        while True:
            for pid in _process_children(os.getpid()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            reaped = False
            while True:
                try:
                    pid, _status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    pid = 0
                if pid <= 0:
                    break
                reaped = True
            if not _process_children(os.getpid()):
                return
            if time.monotonic() >= cleanup_expires:
                raise ExternalTracerBuildProcessError(
                    "contained descendants did not terminate before cleanup deadline"
                )
            if not reaped:
                time.sleep(0.005)

    def _restore(self) -> None:
        try:
            if self.previous_subreaper is not None:
                if _process_children(os.getpid()):
                    self.terminate()
                _set_subreaper(self.previous_subreaper)
                if _subreaper_state() != self.previous_subreaper:
                    _authority("kernel did not restore descendant subreaper state")
        finally:
            for signum, handler in self.previous_signal_handlers.items():
                signal.signal(signum, handler)
            self.previous_signal_handlers.clear()
            self.previous_subreaper = None
            if self.locked:
                self.locked = False
                _PROCESS_CONTAINMENT_LOCK.release()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self._restore()


def _run_bounded_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    max_output_bytes: int,
    process_failure: bool,
    executable: str | None = None,
    pass_fds: Sequence[int] = (),
    absolute_deadline: float | None = None,
) -> tuple[int, bytes, bytes]:
    if not argv or any(type(value) is not str or not value for value in argv):
        _authority("process argv must be a nonempty lexical string vector")
    if any(
        type(key) is not str or type(value) is not str
        for key, value in environment.items()
    ):
        _authority("process environment must contain exact strings")
    if executable is not None and (type(executable) is not str or not executable):
        _authority("process executable capability path must be a nonempty string")
    if any(type(descriptor) is not int or descriptor < 0 for descriptor in pass_fds):
        _authority("process retained descriptors must be nonnegative integers")
    if len(set(pass_fds)) != len(pass_fds):
        _authority("process retained descriptors must be unique")
    if absolute_deadline is not None and type(absolute_deadline) is not float:
        _authority("process absolute deadline must be one monotonic float")
    launch_started = time.monotonic()
    local_deadline = launch_started + timeout_seconds
    deadline = (
        local_deadline
        if absolute_deadline is None
        else min(local_deadline, absolute_deadline)
    )
    error_type = (
        ExternalTracerBuildProcessError
        if process_failure
        else ExternalTracerBuildAuthorityError
    )
    result: tuple[int, bytes, bytes] | None = None
    with _DescendantContainment(
        deadline=deadline,
        timeout_seconds=timeout_seconds,
    ) as containment:
        try:
            process = subprocess.Popen(
                list(argv),
                executable=executable,
                cwd=str(cwd),
                env=dict(environment),
                pass_fds=tuple(pass_fds),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise error_type(f"cannot start process {argv[0]!r}: {exc}") from exc
        containment.attach(process)

        assert process.stdout is not None and process.stderr is not None
        streams = {
            process.stdout.fileno(): bytearray(),
            process.stderr.fileno(): bytearray(),
        }
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            selector.register(process.stderr, selectors.EVENT_READ)
            failure: str | None = None
            timeout_failure = False
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = f"process timeout exceeded: {timeout_seconds}s"
                    timeout_failure = True
                    break
                events = selector.select(min(remaining, 0.2))
                if not events and process.poll() is not None:
                    events = selector.select(0)
                for key, _mask in events:
                    descriptor = key.fileobj.fileno()
                    try:
                        chunk = os.read(descriptor, 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target = streams[descriptor]
                    if len(target) + len(chunk) > max_output_bytes:
                        failure = (
                            "process output limit exceeded: "
                            f"stream_max={max_output_bytes}"
                        )
                        break
                    target.extend(chunk)
                if failure is not None:
                    break

            if failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = f"process timeout exceeded: {timeout_seconds}s"
                    timeout_failure = True
                else:
                    try:
                        returncode = process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        failure = f"process timeout exceeded: {timeout_seconds}s"
                        timeout_failure = True

            if failure is not None:
                containment.terminate()
                if timeout_failure:
                    raise ExternalTracerBuildProcessError(failure)
                raise error_type(failure)

            containment.reap_exited_descendants()
            lingering = containment.descendants()
            if lingering:
                containment.terminate()
                raise error_type(
                    "process leader exited with running contained descendants: "
                    f"pids={sorted(lingering)}"
                )
            result = (
                returncode,
                bytes(streams[stdout_fd]),
                bytes(streams[stderr_fd]),
            )
        finally:
            selector.close()
            process.stdout.close()
            process.stderr.close()
    assert result is not None
    return result


def _git_environment(git: immutable.RetainedExecutable) -> dict[str, str]:
    if os.pathsep in str(git.path.parent):
        _authority("Git tool parent cannot be represented in closed PATH")
    support = "/usr/bin:/bin"
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": f"{git.path.parent}:{support}",
        "TZ": "UTC",
    }


def _run_git(
    git: immutable.RetainedExecutable,
    repo_root: Path,
    arguments: Sequence[str],
    *,
    deadline: OperationDeadline,
    max_output_bytes: int = MAX_GIT_OUTPUT_BYTES,
) -> bytes:
    deadline.check(label="Git authority setup")
    argv = (str(git.path), "-C", str(repo_root), *arguments)
    returncode, stdout, stderr = _run_bounded_process(
        argv,
        cwd=repo_root,
        environment=_git_environment(git),
        timeout_seconds=60,
        max_output_bytes=max_output_bytes,
        process_failure=False,
        executable=git.proc_path,
        pass_fds=(git.descriptor,),
        absolute_deadline=deadline.expires_at,
    )
    deadline.check(label="Git authority command")
    if returncode != 0:
        detail = stderr[:4096].decode("utf-8", errors="replace")
        _authority(
            f"Git authority command failed ({returncode}): {arguments!r}: {detail}"
        )
    return stdout


def _one_line(payload: bytes, *, label: str) -> str:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        _authority(f"{label} is not ASCII: {exc}")
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        _authority(f"{label} must contain exactly one nonempty line")
    return lines[0]


def _repository_identity(
    git: immutable.RetainedExecutable,
    repo_root: Path,
    *,
    deadline: OperationDeadline,
) -> tuple[str, str]:
    object_format = _one_line(
        _run_git(
            git,
            repo_root,
            ("rev-parse", "--show-object-format"),
            deadline=deadline,
        ),
        label="Git object format",
    )
    if object_format != "sha1":
        _authority(f"repository object format must be sha1, got {object_format!r}")
    commit = _one_line(
        _run_git(
            git,
            repo_root,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            deadline=deadline,
        ),
        label="HEAD commit",
    )
    tree = _one_line(
        _run_git(
            git,
            repo_root,
            ("rev-parse", "--verify", "HEAD^{tree}"),
            deadline=deadline,
        ),
        label="HEAD tree",
    )
    if _SHA1.fullmatch(commit) is None or _SHA1.fullmatch(tree) is None:
        _authority("HEAD commit/tree must be full lowercase SHA-1 object IDs")
    status_payload = _run_git(
        git,
        repo_root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        deadline=deadline,
    )
    if status_payload:
        _authority(
            "repository checkout is dirty; commit every tracked and nonignored "
            "change before external tracer build"
        )
    return commit, tree


def _reverify_repository(
    git: immutable.RetainedExecutable,
    repo_root: Path,
    *,
    commit: str,
    tree: str,
    deadline: OperationDeadline,
) -> None:
    observed_commit, observed_tree = _repository_identity(
        git, repo_root, deadline=deadline
    )
    if observed_commit != commit or observed_tree != tree:
        _authority(
            "repository HEAD drifted during external tracer build: "
            f"expected={commit}/{tree}, observed={observed_commit}/{observed_tree}"
        )


def _parse_ls_tree(payload: bytes) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, object_type, oid = metadata.decode("ascii").split(" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            _authority(f"malformed git ls-tree record: {exc}")
        try:
            immutable.safe_relative_path(path)
        except immutable.ImmutableArtifactError as exc:
            _authority(f"unsafe tracked source path {path!r}: {exc}")
        if path in seen:
            _authority(f"duplicate tracked source path: {path}")
        seen.add(path)
        if object_type != "blob" or mode not in {"100644", "100755"}:
            _authority(
                "source closure admits only regular tracked blobs: "
                f"path={path}, mode={mode}, type={object_type}"
            )
        if _SHA1.fullmatch(oid) is None:
            _authority(f"tracked source has malformed SHA-1 blob OID: {path}")
        rows.append((path, mode, oid))
    if not rows:
        _authority("Git source closure is empty")
    if len(rows) > MAX_SOURCE_FILES:
        _authority(
            f"Git source file limit exceeded: {len(rows)} > {MAX_SOURCE_FILES}"
        )
    return tuple(sorted(rows, key=lambda row: row[0].encode("utf-8")))


def _read_source_snapshot(
    git: immutable.RetainedExecutable,
    repo_root: Path,
    *,
    commit: str,
    tree: str,
    deadline: OperationDeadline,
) -> SourceSnapshot:
    listing = _run_git(
        git,
        repo_root,
        ("ls-tree", "-rz", "--full-tree", commit, "--", *SOURCE_PATHSPECS),
        deadline=deadline,
    )
    rows = _parse_ls_tree(listing)
    observed_paths = {path for path, _mode, _oid in rows}
    missing = sorted(REQUIRED_SOURCE_PATHS - observed_paths)
    if missing:
        _authority(f"source closure omits required build files: {missing!r}")
    prohibited = sorted(PROHIBITED_TRACKED_OUTPUTS & observed_paths)
    if prohibited:
        _authority(f"build outputs must never be tracked source inputs: {prohibited!r}")

    entries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    total = 0
    for path, mode, oid in rows:
        deadline.check(label=f"Git source blob {path}")
        payload = _run_git(
            git,
            repo_root,
            ("cat-file", "blob", oid),
            deadline=deadline,
            max_output_bytes=min(MAX_GIT_OUTPUT_BYTES, MAX_SOURCE_BYTES + 1),
        )
        if _sha1_blob(payload) != oid:
            _authority(f"Git object bytes do not match declared blob OID: {path}")
        total += len(payload)
        if total > MAX_SOURCE_BYTES:
            _authority(
                f"Git source byte limit exceeded: {total} > {MAX_SOURCE_BYTES}"
            )
        entry = {
            "path": path,
            "mode": mode,
            "blob_oid_sha1": oid,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
        entries.append(entry)
        payloads[path] = payload

    _validate_local_include_closure(payloads)
    deadline.check(label="source dependency-closure validation")
    inventory_sha256 = _sha256(immutable.canonical_json_bytes(entries))
    return SourceSnapshot(
        commit=commit,
        tree=tree,
        entries=tuple(entries),
        payloads=payloads,
        inventory_sha256=inventory_sha256,
        total_size_bytes=total,
    )


def _normalize_join(parent: PurePosixPath, reference: str) -> PurePosixPath | None:
    parts = list(parent.parts)
    for component in reference.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(component)
    return PurePosixPath(*parts)


def _validate_local_include_closure(payloads: Mapping[str, bytes]) -> None:
    inventory = set(payloads)
    suffixes = {path: tuple(PurePosixPath(path).parts) for path in inventory}
    for path, payload in payloads.items():
        if PurePosixPath(path).suffix not in {".c", ".cc", ".cpp", ".cu", ".h", ".hpp"}:
            continue
        for match in _LOCAL_INCLUDE.finditer(payload):
            try:
                reference = match.group(1).decode("utf-8")
            except UnicodeDecodeError as exc:
                _authority(f"non-UTF-8 local include in {path}: {exc}")
            if "\\" in reference or "\x00" in reference or reference.startswith("/"):
                _authority(f"local include is not a safe relative path: {path}: {reference!r}")
            resolved = _normalize_join(PurePosixPath(path).parent, reference)
            if resolved is None or resolved.parts[:2] != ("tools", "nvbit_tracer"):
                _authority(f"local include escapes extracted source closure: {path}: {reference}")
            resolved_text = resolved.as_posix()
            if resolved_text in inventory:
                continue
            reference_parts = tuple(PurePosixPath(reference).parts)
            if any(parts[-len(reference_parts) :] == reference_parts for parts in suffixes.values()):
                continue
            if reference.startswith(("google/protobuf/", "absl/")):
                continue
            if reference.endswith("/..."):
                # RapidJSON's public header documents this placeholder inside
                # block comments; it is not a preprocessor dependency.
                continue
            if reference in EXTERNAL_QUOTED_HEADERS:
                continue
            _authority(
                f"local include is absent from tracked source closure: {path}: {reference}"
            )


def _create_fresh_direct_tmp_root(path: Path) -> Path:
    candidate = _absolute_lexical(path, label="build root")
    _require_make_safe_path(candidate, label="build root")
    if candidate.parent != Path("/tmp") or candidate.name in {"", ".", ".."}:
        _authority("build root must be a fresh direct child of /tmp")
    try:
        os.mkdir(candidate, 0o700)
    except FileExistsError as exc:
        _authority(f"build root already exists: {candidate}")
    except OSError as exc:
        _authority(f"cannot create fresh build root {candidate}: {exc}")
    return candidate


def _write_extracted_sources(root: Path, snapshot: SourceSnapshot) -> None:
    for entry in snapshot.entries:
        relative = immutable.safe_relative_path(entry["path"])
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(destination, flags, 0o600)
        try:
            payload = snapshot.payloads[entry["path"]]
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    _authority(f"short write while extracting {entry['path']}")
                view = view[written:]
            os.fchmod(descriptor, 0o755 if entry["mode"] == "100755" else 0o644)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _snapshot_regular(path: Path, *, max_bytes: int) -> tuple[str, int, str]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        _authority(f"extracted source is unavailable: {path}: {exc}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _authority(f"extracted source is not a regular file: {path}")
    expected_mode = "100755" if metadata.st_mode & 0o111 else "100644"
    try:
        with immutable.open_retained_regular(path, max_bytes=max_bytes) as retained:
            return retained.sha256, retained.size_bytes, expected_mode
    except immutable.ImmutableArtifactError as exc:
        _authority(f"cannot authenticate extracted source {path}: {exc}")


def _verify_extracted_sources(root: Path, snapshot: SourceSnapshot) -> None:
    entries = {entry["path"]: entry for entry in snapshot.entries}
    for path, entry in entries.items():
        observed_sha, observed_size, observed_mode = _snapshot_regular(
            root / PurePosixPath(path), max_bytes=MAX_SOURCE_BYTES
        )
        if (
            observed_sha != entry["sha256"]
            or observed_size != entry["size_bytes"]
            or observed_mode != entry["mode"]
        ):
            prefix = (
                "generated protobuf drifted"
                if path.startswith(GENERATED_PROTO_PREFIXES)
                else "extracted tracked source drifted"
            )
            _authority(f"{prefix}: {path}")

    for prefix in GENERATED_PROTO_PREFIXES:
        expected = {path for path in entries if path.startswith(prefix)}
        directory = root / PurePosixPath(prefix)
        if not directory.is_dir():
            _authority(f"generated protobuf directory is unavailable: {prefix}")
        observed: set[str] = set()
        for candidate in directory.iterdir():
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                _authority(f"generated protobuf inventory contains non-file: {candidate}")
            observed.add(candidate.relative_to(root).as_posix())
        if observed != expected:
            _authority(
                "generated protobuf inventory drifted: "
                f"prefix={prefix}, missing={sorted(expected-observed)!r}, "
                f"unknown={sorted(observed-expected)!r}"
            )


def _verify_exact_discovered_source_sets(
    root: Path,
    snapshot: SourceSnapshot,
) -> None:
    expected_paths = {entry["path"] for entry in snapshot.entries}
    tracer_directory = root / "tools/nvbit_tracer/tracer_tool"
    traces_source = root / "tools/nvbit_tracer/traces_enhanced/src"
    expected_cuda = {
        path
        for path in expected_paths
        if path.startswith("tools/nvbit_tracer/tracer_tool/")
        and "/" not in path[len("tools/nvbit_tracer/tracer_tool/") :]
        and path.endswith(".cu")
    }
    expected_cc = {
        path
        for path in expected_paths
        if path.startswith("tools/nvbit_tracer/traces_enhanced/src/")
        and path.endswith(".cc")
    }
    observed_cuda: set[str] = set()
    try:
        with os.scandir(tracer_directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".cu"):
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    _authority(f"wildcard CUDA input is not regular: {entry.path}")
                observed_cuda.add(
                    Path(entry.path).relative_to(root).as_posix()
                )
    except OSError as exc:
        _authority(f"cannot inventory wildcard CUDA inputs: {exc}")

    observed_cc: set[str] = set()
    pending = [traces_source]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                rows = list(entries)
        except OSError as exc:
            _authority(f"cannot inventory find-based C++ inputs: {exc}")
        for entry in rows:
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                _authority(f"find-based C++ input tree contains symlink: {entry.path}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode) and entry.name.endswith(".cc"):
                observed_cc.add(Path(entry.path).relative_to(root).as_posix())
    if observed_cuda != expected_cuda:
        _authority(
            "wildcard CUDA source inventory differs from manifest: "
            f"missing={sorted(expected_cuda-observed_cuda)!r}, "
            f"unknown={sorted(observed_cuda-expected_cuda)!r}"
        )
    if observed_cc != expected_cc:
        _authority(
            "find-based C++ source inventory differs from manifest: "
            f"missing={sorted(expected_cc-observed_cc)!r}, "
            f"unknown={sorted(observed_cc-expected_cc)!r}"
        )


def _retain_extracted_input_generation(
    stack: contextlib.ExitStack,
    *,
    root: Path,
    snapshot: SourceSnapshot,
) -> ExtractedInputGeneration:
    generated_prefixes = tuple(
        PurePosixPath(prefix).as_posix() for prefix in GENERATED_PROTO_PREFIXES
    )
    ordinary_paths = tuple(
        entry["path"]
        for entry in snapshot.entries
        if not entry["path"].startswith(GENERATED_PROTO_PREFIXES)
    )
    limits = {
        "max_files": MAX_SOURCE_FILES,
        "max_directories": MAX_SOURCE_FILES,
        "max_total_file_bytes": MAX_SOURCE_BYTES,
        "max_total_relative_path_bytes": MAX_SOURCE_BYTES,
    }
    try:
        source = stack.enter_context(
            immutable.PinnedPathSetMonitor(
                root,
                regular_paths=ordinary_paths,
                recursive_directories=(),
                **limits,
            )
        )
        generated = stack.enter_context(
            immutable.PinnedPathSetMonitor(
                root,
                regular_paths=(),
                recursive_directories=generated_prefixes,
                **limits,
            )
        )
        return ExtractedInputGeneration(source, generated)
    except immutable.ImmutableArtifactError as exc:
        _authority(f"cannot retain extracted source generation: {exc}")


def _open_external_executable(
    stack: contextlib.ExitStack,
    *,
    role: str,
    raw_path: Path,
    repo_root: Path,
) -> immutable.RetainedExecutable:
    path = _absolute_lexical(raw_path, label=f"{role} tool")
    _require_make_safe_path(path, label=f"{role} tool path")
    _require_lexically_external(path, repo_root=repo_root, label=f"{role} tool")
    try:
        path = require_external_non_git_artifact(
            path,
            root=path.parent,
            repo_root=repo_root,
            label=f"{role} tool",
            max_bytes=MAX_TOOL_BYTES,
        )
        return stack.enter_context(immutable.open_retained_executable(path))
    except (
        OSError,
        ExternalTracerAuthorityError,
        immutable.ImmutableArtifactError,
    ) as exc:
        _authority(f"cannot retain exact external {role} tool {path}: {exc}")


def _open_tool_set(
    stack: contextlib.ExitStack,
    paths: Mapping[str, Path],
    *,
    repo_root: Path,
) -> ToolSet:
    if tuple(paths) != TOOL_ROLES:
        _authority(f"tool role order must be exact: {TOOL_ROLES!r}")

    # Complete the no-access lexical pass for every caller path before any one
    # tool is statted, opened, hashed, or executed. This keeps a repository-local
    # ignored tracer from being probed merely because it was supplied as a tool.
    lexical_paths: dict[str, Path] = {}
    for role, raw_path in paths.items():
        path = _absolute_lexical(raw_path, label=f"{role} tool")
        _require_make_safe_path(path, label=f"{role} tool path")
        _require_lexically_external(
            path, repo_root=repo_root, label=f"{role} tool"
        )
        lexical_paths[role] = path

    retained: dict[str, immutable.RetainedExecutable] = {}
    identities: list[dict[str, Any]] = []
    for role, path in lexical_paths.items():
        capability = _open_external_executable(
            stack,
            role=role,
            raw_path=path,
            repo_root=repo_root,
        )
        size_bytes = capability.snapshot[4]
        retained[role] = capability
        identities.append(
            {
                "role": role,
                "path": str(path),
                "sha256": capability.sha256,
                "size_bytes": size_bytes,
            }
        )
    inventory_sha256 = _sha256(immutable.canonical_json_bytes(identities))
    return ToolSet(retained, tuple(identities), inventory_sha256)


def _verify_tools(tools: ToolSet) -> None:
    for role in TOOL_ROLES:
        try:
            tools.retained[role].verify_unchanged()
        except immutable.ImmutableArtifactError as exc:
            _authority(f"retained {role} tool drifted: {exc}")


def _install_tool_aliases(build_root: Path, tools: ToolSet) -> Path:
    private_bin = build_root / ".task630-tool-bin"
    private_bin.mkdir(mode=0o700)
    for role, alias in TOOL_ALIASES.items():
        os.symlink(tools.retained[role].proc_path, private_bin / alias)
    return private_bin


def _retain_tool_alias_generation(
    stack: contextlib.ExitStack,
    *,
    build_root: Path,
) -> immutable.PinnedPathSetMonitor:
    try:
        return stack.enter_context(
            immutable.PinnedPathSetMonitor(
                build_root,
                regular_paths=(),
                recursive_directories=(),
                filtered_directories={
                    ".task630-tool-bin": tuple(TOOL_ALIASES.values())
                },
            )
        )
    except immutable.ImmutableArtifactError as exc:
        _authority(f"cannot retain exact build-tool alias generation: {exc}")


def _verify_generation_monitor(
    monitor: immutable.PinnedPathSetMonitor, *, label: str
) -> None:
    try:
        monitor.verify_unchanged()
    except immutable.ImmutableArtifactError as exc:
        _authority(f"{label} drifted during external tracer build: {exc}")


def _verify_extracted_generation(
    generation: ExtractedInputGeneration,
) -> None:
    _verify_generation_monitor(
        generation.source, label="extracted tracked source"
    )
    _verify_generation_monitor(
        generation.generated_protobuf, label="generated protobuf"
    )


def _tool_descriptors(tools: ToolSet) -> tuple[int, ...]:
    return tuple(tools.retained[role].descriptor for role in TOOL_ROLES)


def _validate_protobuf_prefix(prefix: Path, *, repo_root: Path) -> Path:
    candidate = _absolute_lexical(prefix, label="protobuf prefix")
    _require_make_safe_path(candidate, label="protobuf prefix")
    _require_lexically_external(
        candidate, repo_root=repo_root, label="protobuf prefix"
    )
    try:
        resolved = require_external_non_git_root(
            candidate,
            repo_root=repo_root,
            label="protobuf prefix",
        )
    except ExternalTracerAuthorityError as exc:
        _authority(f"protobuf prefix lacks external non-Git authority: {exc}")
    for required in ("include", "lib"):
        child = resolved / required
        try:
            child_metadata = child.lstat()
        except OSError as exc:
            _authority(f"protobuf prefix lacks {required}/: {exc}")
        if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISDIR(child_metadata.st_mode):
            _authority(f"protobuf prefix {required}/ must be a real directory")
    return resolved


def _protobuf_node_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _snapshot_protobuf(prefix: Path) -> ProtobufSnapshot:
    entries: list[dict[str, Any]] = []
    total = 0
    directory_count = 1
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        _authority("protobuf inventory requires O_NOFOLLOW and O_DIRECTORY")
    directory_flags = (
        os.O_RDONLY
        | nofollow
        | directory_flag
        | getattr(os, "O_CLOEXEC", 0)
    )

    def visit(directory_descriptor: int, relative_parts: tuple[str, ...]) -> None:
        nonlocal directory_count, total
        before = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(before.st_mode):
            _authority("protobuf inventory descriptor is not a directory")

        rows: list[tuple[str, os.stat_result]] = []
        with os.scandir(directory_descriptor) as scanner:
            for entry in scanner:
                name = entry.name
                if type(name) is not str or name in {"", ".", ".."} or "/" in name:
                    _authority(f"protobuf prefix contains an unsafe name: {name!r}")
                if name == ".git":
                    _authority(
                        "protobuf prefix contains nested Git metadata: "
                        f"{prefix.joinpath(*relative_parts, name)}"
                    )
                # This is the sole classification stat. The explicit false flag
                # obtains lstat semantics and cannot traverse a symlink target.
                rows.append((name, entry.stat(follow_symlinks=False)))
        rows.sort(key=lambda row: os.fsencode(row[0]))

        for name, metadata in rows:
            child_parts = (*relative_parts, name)
            relative = PurePosixPath(*child_parts).as_posix()
            immutable.safe_relative_path(relative)
            child = prefix.joinpath(*child_parts)
            if stat.S_ISLNK(metadata.st_mode):
                _authority(f"protobuf prefix contains a symlink member: {child}")
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                if directory_count > MAX_PROTOBUF_DIRECTORIES:
                    _authority(
                        "protobuf prefix directory limit exceeded: "
                        f"{directory_count} > {MAX_PROTOBUF_DIRECTORIES}"
                    )
                try:
                    child_descriptor = os.open(
                        name,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                except OSError as exc:
                    _authority(f"cannot pin protobuf directory {child}: {exc}")
                try:
                    opened = os.fstat(child_descriptor)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or _protobuf_node_identity(opened)
                        != _protobuf_node_identity(metadata)
                    ):
                        _authority(
                            f"protobuf directory changed before traversal: {child}"
                        )
                    visit(child_descriptor, child_parts)
                    linked = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISLNK(linked.st_mode)
                        or _protobuf_node_identity(linked)
                        != _protobuf_node_identity(opened)
                        or _protobuf_node_identity(os.fstat(child_descriptor))
                        != _protobuf_node_identity(opened)
                    ):
                        _authority(
                            f"protobuf directory changed during traversal: {child}"
                        )
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                _authority(f"protobuf prefix contains a special file: {child}")
            if metadata.st_nlink != 1:
                _authority(
                    f"protobuf prefix member is a hard-linked alias: {child}"
                )
            try:
                with immutable.open_retained_regular(
                    child, max_bytes=MAX_TOOL_BYTES
                ) as retained:
                    digest = retained.sha256
                    size_bytes = retained.size_bytes
            except immutable.ImmutableArtifactError as exc:
                _authority(f"cannot authenticate protobuf file {child}: {exc}")
            linked = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(linked.st_mode)
                or _protobuf_node_identity(linked)
                != _protobuf_node_identity(metadata)
            ):
                _authority(f"protobuf file changed during inventory: {child}")
            total += size_bytes
            if total > MAX_PROTOBUF_BYTES:
                _authority(
                    "protobuf prefix byte limit exceeded: "
                    f"{total} > {MAX_PROTOBUF_BYTES}"
                )
            entries.append(
                {
                    "path": relative,
                    "kind": "regular",
                    "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                    "sha256": digest,
                    "size_bytes": size_bytes,
                }
            )
            if len(entries) > MAX_PROTOBUF_FILES:
                _authority(
                    "protobuf prefix file limit exceeded: "
                    f"{len(entries)} > {MAX_PROTOBUF_FILES}"
                )

        after = os.fstat(directory_descriptor)
        if _protobuf_node_identity(after) != _protobuf_node_identity(before):
            _authority(
                "protobuf directory changed while its members were inventoried: "
                f"{prefix.joinpath(*relative_parts)}"
            )

    try:
        with immutable.PinnedDirectoryReader(prefix) as pinned_root:
            visit(pinned_root.descriptor, ())
            pinned_root.verify_path()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _authority(f"protobuf prefix inventory is unavailable: {exc}")
    if not entries:
        _authority("protobuf prefix inventory is empty")
    entries.sort(key=lambda row: row["path"].encode("utf-8"))
    return ProtobufSnapshot(
        prefix=prefix,
        entries=tuple(entries),
        inventory_sha256=_sha256(immutable.canonical_json_bytes(entries)),
        file_count=len(entries),
        total_size_bytes=total,
    )


def _retain_protobuf_generation(
    stack: contextlib.ExitStack,
    prefix: Path,
) -> None:
    try:
        reader = stack.enter_context(immutable.PinnedDirectoryReader(prefix))
        stack.enter_context(
            reader.retain_tree_generation(
                max_files=MAX_PROTOBUF_FILES,
                max_directories=MAX_PROTOBUF_DIRECTORIES,
                max_total_file_bytes=MAX_PROTOBUF_BYTES,
                max_total_relative_path_bytes=MAX_PROTOBUF_BYTES,
            )
        )
    except immutable.ImmutableArtifactError as exc:
        detail = str(exc)
        if "single-link regular file" in detail:
            _authority(
                "cannot retain pinned protobuf generation: hard-linked alias or "
                f"nonregular member: {detail}"
            )
        if "symlink in immutable tree" in detail:
            _authority(
                "cannot retain pinned protobuf generation: symlink member: "
                f"{detail}"
            )
        _authority(f"cannot retain pinned protobuf generation: {exc}")


def _retain_protobuf_files(
    stack: contextlib.ExitStack,
    snapshot: ProtobufSnapshot,
) -> immutable.PinnedPathSetMonitor:
    try:
        return stack.enter_context(
            immutable.PinnedPathSetMonitor(
                snapshot.prefix,
                regular_paths=(entry["path"] for entry in snapshot.entries),
                recursive_directories=(),
                max_files=MAX_PROTOBUF_FILES,
                max_directories=MAX_PROTOBUF_DIRECTORIES,
                max_total_file_bytes=MAX_PROTOBUF_BYTES,
                max_total_relative_path_bytes=MAX_PROTOBUF_BYTES,
            )
        )
    except immutable.ImmutableArtifactError as exc:
        _authority(f"cannot retain pinned protobuf files: {exc}")


def _reverify_protobuf(expected: ProtobufSnapshot) -> None:
    observed = _snapshot_protobuf(expected.prefix)
    if observed != expected:
        _authority("pinned protobuf prefix drifted during external tracer build")


def _probe_pin(
    tool: immutable.RetainedExecutable,
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    label: str,
    deadline: OperationDeadline,
) -> bytes:
    deadline.check(label=f"{label} pin probe")
    argv = (str(tool.path), *arguments)
    returncode, stdout, stderr = _run_bounded_process(
        argv,
        cwd=cwd,
        environment=environment,
        timeout_seconds=30,
        max_output_bytes=1024 * 1024,
        process_failure=False,
        executable=tool.proc_path,
        pass_fds=(tool.descriptor,),
        absolute_deadline=deadline.expires_at,
    )
    deadline.check(label=f"{label} pin probe")
    if returncode != 0:
        _authority(f"{label} pin probe exited {returncode}")
    return stdout + stderr


def _verify_pinned_toolchain(
    tools: ToolSet,
    *,
    protobuf_prefix: Path,
    environment: Mapping[str, str],
    deadline: OperationDeadline,
) -> None:
    nvcc = _probe_pin(
        tools.retained["nvcc"],
        ("--version",),
        cwd=protobuf_prefix,
        environment=environment,
        label="nvcc",
        deadline=deadline,
    )
    if re.search(rb"release[ ,]+13\.2(?:[ ,]|$)", nvcc) is None:
        _authority("nvcc pin must report CUDA release 13.2")
    ptxas = _probe_pin(
        tools.retained["ptxas"],
        ("--version",),
        cwd=protobuf_prefix,
        environment=environment,
        label="ptxas",
        deadline=deadline,
    )
    if re.search(rb"release[ ,]+13\.2(?:[ ,]|$)", ptxas) is None:
        _authority("ptxas pin must report CUDA release 13.2")
    cuobjdump = _probe_pin(
        tools.retained["cuobjdump"],
        ("--version",),
        cwd=protobuf_prefix,
        environment=environment,
        label="cuobjdump",
        deadline=deadline,
    )
    if re.search(rb"release[ ,]+13\.2(?:[ ,]|$)", cuobjdump) is None:
        _authority("cuobjdump pin must report CUDA release 13.2")
    host_version = _one_line(
        _probe_pin(
            tools.retained["host_cxx"],
            ("-dumpversion",),
            cwd=protobuf_prefix,
            environment=environment,
            label="host C++",
            deadline=deadline,
        ),
        label="host C++ major version",
    )
    if host_version.split(".", 1)[0] != "14":
        _authority(f"host C++ pin must be major 14, got {host_version!r}")
    protoc_version = _one_line(
        _probe_pin(
            tools.retained["protoc"],
            ("--version",),
            cwd=protobuf_prefix,
            environment=environment,
            label="protoc",
            deadline=deadline,
        ),
        label="protoc version",
    )
    if protoc_version != "libprotoc 34.1":
        _authority(f"protoc pin must be 34.1, got {protoc_version!r}")
    protobuf_version = _one_line(
        _probe_pin(
            tools.retained["pkg_config"],
            ("--define-prefix", "--modversion", "protobuf"),
            cwd=protobuf_prefix,
            environment=environment,
            label="pkg-config protobuf version",
            deadline=deadline,
        ),
        label="libprotobuf version",
    )
    if protobuf_version != "34.1.0":
        _authority(
            f"libprotobuf pin must be 34.1.0, got {protobuf_version!r}"
        )
    observed_prefix = _one_line(
        _probe_pin(
            tools.retained["pkg_config"],
            ("--define-prefix", "--variable=prefix", "protobuf"),
            cwd=protobuf_prefix,
            environment=environment,
            label="pkg-config protobuf prefix",
            deadline=deadline,
        ),
        label="libprotobuf prefix",
    )
    if observed_prefix != str(protobuf_prefix):
        _authority(
            "pkg-config protobuf prefix mismatch: "
            f"expected={protobuf_prefix}, observed={observed_prefix}"
        )


def _tool_alias(private_bin: Path, role: str) -> Path:
    return private_bin / TOOL_ALIASES[role]


def _build_environment(
    request: BuildRequest, *, private_bin: Path, protobuf_prefix: Path
) -> dict[str, str]:
    cuda_root = request.nvcc.parent.parent
    _require_make_safe_path(cuda_root, label="CUDA toolkit root")
    return {
        "ARCH": CONTRACT["architecture"],
        "CPLUS_INCLUDE_PATH": str(protobuf_prefix / "include"),
        "CUDA_INSTALL_PATH": str(cuda_root),
        "CXX": str(_tool_alias(private_bin, "host_cxx")),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": str(protobuf_prefix / "lib"),
        "NVBIT_TEMPCOPY_BUILD_OK": "1",
        "NVCC_CCBIN": str(_tool_alias(private_bin, "host_cxx")),
        "PATH": f"{private_bin}:/usr/bin:/bin",
        "PB_PREFIX": str(protobuf_prefix),
        "PKG_CONFIG": str(_tool_alias(private_bin, "pkg_config")),
        "PKG_CONFIG_LIBDIR": ":".join(
            str(path)
            for path in (
                protobuf_prefix / "lib/pkgconfig",
                protobuf_prefix / "share/pkgconfig",
            )
        ),
        "PKG_CONFIG_PATH": "",
        "PROTOC": str(_tool_alias(private_bin, "protoc")),
        "SHELL": str(_tool_alias(private_bin, "bash")),
        "TZ": "UTC",
    }


def _exact_build_input_assignments(
    source_paths: Iterable[str],
) -> tuple[str, ...]:
    paths = tuple(source_paths)
    tracer_prefix = "tools/nvbit_tracer/tracer_tool/"
    traces_prefix = "tools/nvbit_tracer/traces_enhanced/"
    tracer_cuda = sorted(
        path[len(tracer_prefix) :]
        for path in paths
        if path.startswith(tracer_prefix)
        and "/" not in path[len(tracer_prefix) :]
        and path.endswith(".cu")
    )
    trace_cc = sorted(
        path[len(traces_prefix + "src/") :]
        for path in paths
        if path.startswith(traces_prefix + "src/") and path.endswith(".cc")
    )
    proto_files = sorted(
        path[len(traces_prefix) :]
        for path in paths
        if path.startswith(traces_prefix + "dynamic_trace/")
        and path.endswith(".proto")
    )
    protobuf_cc = sorted(
        path[len(traces_prefix + "pb_trace/src/") :]
        for path in paths
        if path.startswith(traces_prefix + "pb_trace/src/")
        and path.endswith(".pb.cc")
    )
    if not tracer_cuda or not trace_cc or not proto_files or not protobuf_cc:
        _authority("source manifest lacks one exact Make input class")
    values = {
        "OBJECTS": " ".join(f"{PurePosixPath(path).stem}.o" for path in tracer_cuda),
        "OBJ": " ".join(
            f"obj/{PurePosixPath(path).with_suffix('.o').as_posix()}"
            for path in trace_cc
        ),
        "PROTO_FILES": " ".join(proto_files),
        "TRACES_ENHANCED_OBJ": " ".join(
            "../traces_enhanced/obj/"
            f"{PurePosixPath(path).with_suffix('.o').as_posix()}"
            for path in trace_cc
        ),
        "TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ": " ".join(
            "../traces_enhanced/pb_trace/obj/"
            f"{PurePosixPath(path).with_suffix('').with_suffix('.pb.o').as_posix()}"
            for path in protobuf_cc
        ),
    }
    for name, value in values.items():
        if not value or re.fullmatch(r"[A-Za-z0-9._/ -]+", value) is None:
            _authority(f"exact Make input assignment is unsafe: {name}")
    return tuple(f"{name}={values[name]}" for name in (
        "OBJECTS",
        "OBJ",
        "PROTO_FILES",
        "TRACES_ENHANCED_OBJ",
        "TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ",
    ))


def _build_argv(
    *,
    external_top: Path,
    private_bin: Path,
    protobuf_prefix: Path,
    source_paths: Iterable[str],
) -> tuple[str, ...]:
    return (
        str(_tool_alias(private_bin, "make")),
        "-C",
        str(external_top),
        "all",
        f"ARCH={CONTRACT['architecture']}",
        "NVBIT_TEMPCOPY_BUILD_OK=1",
        f"SHELL={_tool_alias(private_bin, 'bash')}",
        f"MAKE={_tool_alias(private_bin, 'make')}",
        "NVCC="
        f"{_tool_alias(private_bin, 'nvcc')} "
        f"-ccbin={_tool_alias(private_bin, 'host_cxx')} -D_FORCE_INLINES",
        f"PTXAS={_tool_alias(private_bin, 'ptxas')}",
        f"CXX={_tool_alias(private_bin, 'host_cxx')}",
        f"NVCC_CCBIN={_tool_alias(private_bin, 'host_cxx')}",
        f"PROTOC={_tool_alias(private_bin, 'protoc')}",
        f"PKG_CONFIG={_tool_alias(private_bin, 'pkg_config')}",
        "PKG_CONFIG_FLAGS=--define-prefix",
        f"PB_PREFIX={protobuf_prefix}",
        *_exact_build_input_assignments(source_paths),
    )


def _run_external_build(
    request: BuildRequest,
    *,
    external_top: Path,
    private_bin: Path,
    tools: ToolSet,
    source: SourceSnapshot,
    environment: Mapping[str, str],
    protobuf_prefix: Path,
    deadline: OperationDeadline,
) -> ProcessReceipt:
    argv = _build_argv(
        external_top=external_top,
        private_bin=private_bin,
        protobuf_prefix=protobuf_prefix,
        source_paths=(entry["path"] for entry in source.entries),
    )
    make = tools.retained["make"]
    returncode, stdout, stderr = _run_bounded_process(
        argv,
        cwd=external_top,
        environment=environment,
        timeout_seconds=request.timeout_seconds,
        max_output_bytes=request.max_output_bytes,
        process_failure=True,
        executable=make.proc_path,
        pass_fds=_tool_descriptors(tools),
        absolute_deadline=deadline.expires_at,
    )
    deadline.check(label="external top-level build")
    if returncode != 0:
        raise ExternalTracerBuildProcessError(
            f"external top-level build exited {returncode}; "
            f"stdout_sha256={_sha256(stdout)}, stderr_sha256={_sha256(stderr)}"
        )
    return ProcessReceipt(
        argv=argv,
        cwd=str(external_top),
        environment=tuple(sorted(environment.items())),
        returncode=returncode,
        stdout_sha256=_sha256(stdout),
        stdout_size_bytes=len(stdout),
        stderr_sha256=_sha256(stderr),
        stderr_size_bytes=len(stderr),
    )


def _validate_elf_shared_object(payload: bytes) -> dict[str, Any]:
    if len(payload) < 64:
        _authority("tracer output is too short for an ELF64 header")
    try:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", payload, 0)
    except struct.error as exc:
        _authority(f"tracer output has a malformed ELF64 header: {exc}")
    ident = header[0]
    if ident[:4] != b"\x7fELF":
        _authority("tracer output is not ELF")
    if ident[4] != 2 or ident[5] != 1 or ident[6] != 1:
        _authority("tracer output must be ELF64 little-endian version 1")
    if ident[7] not in {0, 3}:
        _authority(f"tracer output has unsupported ELF OS ABI {ident[7]}")
    elf_type, machine, version = header[1:4]
    phoff, shoff = header[5], header[6]
    ehsize, phentsize, phnum, shentsize, shnum, shstrndx = header[8:14]
    if elf_type != 3:
        _authority(f"tracer output ELF type must be ET_DYN, got {elf_type}")
    if machine != 62:
        _authority(f"tracer output ELF machine must be EM_X86_64, got {machine}")
    if version != 1 or ehsize != 64:
        _authority("tracer output has an invalid ELF header version or size")
    if phnum < 1 or phentsize != 56:
        _authority("tracer output lacks an exact ELF64 program-header table")
    if shnum < 2 or shnum == 0xFFFF or shentsize != 64:
        _authority("tracer output lacks an exact ELF64 section-header table")
    if shstrndx == 0 or shstrndx >= shnum or shstrndx == 0xFFFF:
        _authority("tracer output has an invalid ELF section-name table index")
    if phoff < ehsize or phoff + phnum * phentsize > len(payload):
        _authority("tracer output program-header table is out of bounds")
    if shoff < ehsize or shoff + shnum * shentsize > len(payload):
        _authority("tracer output section-header table is out of bounds")

    program_types = {
        struct.unpack_from("<I", payload, phoff + index * phentsize)[0]
        for index in range(phnum)
    }
    if 1 not in program_types or 2 not in program_types:
        _authority("tracer output must contain PT_LOAD and PT_DYNAMIC segments")

    sections: list[tuple[int, int, int, int]] = []
    for index in range(shnum):
        row = struct.unpack_from("<IIQQQQIIQQ", payload, shoff + index * shentsize)
        name_offset, section_type, _flags, _address, offset, size = row[:6]
        if section_type != 8 and (offset > len(payload) or size > len(payload) - offset):
            _authority(f"tracer output ELF section {index} is out of bounds")
        sections.append((name_offset, section_type, offset, size))
    name_row = sections[shstrndx]
    if name_row[1] != 3 or name_row[3] == 0:
        _authority("tracer output ELF section-name table is invalid")
    names = payload[name_row[2] : name_row[2] + name_row[3]]
    if not names.endswith(b"\0"):
        _authority("tracer output ELF section-name table is unterminated")

    named_sections: dict[str, tuple[int, int]] = {}
    for index, (name_offset, section_type, _offset, size) in enumerate(sections):
        if name_offset >= len(names):
            _authority(f"tracer output ELF section {index} name is out of bounds")
        end = names.find(b"\0", name_offset)
        if end < 0:
            _authority(f"tracer output ELF section {index} name is unterminated")
        try:
            name = names[name_offset:end].decode("ascii")
        except UnicodeDecodeError as exc:
            _authority(f"tracer output ELF section {index} name is not ASCII: {exc}")
        if name and name in named_sections:
            _authority(f"tracer output ELF contains duplicate section {name!r}")
        if name:
            named_sections[name] = (section_type, size)
    fatbin = named_sections.get(".nv_fatbin")
    if fatbin is None or fatbin[0] != 1 or fatbin[1] == 0:
        _authority("tracer output lacks a nonempty CUDA .nv_fatbin section")
    if ".text" not in named_sections or named_sections[".text"][1] == 0:
        _authority("tracer output lacks a nonempty host .text section")
    return {
        "class": "ELF64",
        "data": "little_endian",
        "type": "ET_DYN",
        "machine": "EM_X86_64",
        "cuda_fatbin_section": ".nv_fatbin",
    }


def _inspect_cuda_architectures(
    output: immutable.RetainedRegular,
    inspector: immutable.RetainedExecutable,
    *,
    cwd: Path,
    deadline: OperationDeadline,
) -> dict[str, Any]:
    deadline.check(label="CUDA artifact inspection")
    argv = (str(inspector.path), "--list-elf", str(output.path))
    returncode, stdout, stderr = _run_bounded_process(
        argv,
        cwd=cwd,
        environment={
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        },
        timeout_seconds=30,
        max_output_bytes=4 * 1024 * 1024,
        process_failure=False,
        executable=inspector.proc_path,
        pass_fds=(inspector.descriptor,),
        absolute_deadline=deadline.expires_at,
    )
    deadline.check(label="CUDA artifact inspection")
    if returncode != 0:
        _authority(f"cuobjdump --list-elf rejected tracer output: exit={returncode}")
    if stderr:
        _authority("cuobjdump --list-elf emitted diagnostic stderr")
    architectures: list[str] = []
    cubin_count = 0
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"ELF file"):
            continue
        match = _CUOBJDUMP_ELF_LINE.fullmatch(line)
        if match is None:
            _authority("cuobjdump emitted an unparseable CUDA ELF inventory row")
        cubin_count += 1
        architecture = match.group(1).decode("ascii")
        if architecture not in architectures:
            architectures.append(architecture)
    if cubin_count < 1:
        _authority("cuobjdump found no embedded CUDA cubin")
    if architectures != [CONTRACT["architecture"]]:
        _authority(
            "tracer output CUDA architectures violate exact contract: "
            f"expected={[CONTRACT['architecture']]!r}, observed={architectures!r}"
        )
    return {
        "inspector": "cuobjdump_13.2_list_elf.v1",
        "architectures": architectures,
        "cubin_count": cubin_count,
        "stdout": {
            "sha256": _sha256(stdout),
            "size_bytes": len(stdout),
        },
        "stderr": {
            "sha256": _sha256(stderr),
            "size_bytes": len(stderr),
        },
    }


def _validate_output_contract(
    output: immutable.RetainedRegular,
    inspector: immutable.RetainedExecutable,
    *,
    expected_sha256: str | None,
    cwd: Path,
    deadline: OperationDeadline,
) -> OutputValidation:
    deadline.check(label="artifact byte validation")
    if expected_sha256 is not None:
        expected = _require_sha256(
            expected_sha256, label="independent expected tracer digest"
        )
        if output.sha256 != expected:
            _authority(
                "external build output differs from independent caller authority: "
                f"expected={expected}, observed={output.sha256}"
            )
    payload = output.read_bytes(max_bytes=EXTERNAL_TRACER_MAX_BYTES)
    deadline.check(label="artifact byte hashing")
    elf = _validate_elf_shared_object(payload)
    cuda = _inspect_cuda_architectures(
        output, inspector, cwd=cwd, deadline=deadline
    )
    output.verify_unchanged()
    inspector.verify_unchanged()
    return OutputValidation(elf=elf, cuda=cuda)


def _source_document(snapshot: SourceSnapshot) -> dict[str, Any]:
    return immutable.seal_model(
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_kind": "git_sha1_dependency_closed_blob_inventory_v1",
            "repository": {
                "object_format": "sha1",
                "head_commit_sha1": snapshot.commit,
                "head_tree_sha1": snapshot.tree,
            },
            "dependency_pathspecs": list(SOURCE_PATHSPECS),
            "files": list(snapshot.entries),
            "inventory": {
                "file_count": len(snapshot.entries),
                "total_size_bytes": snapshot.total_size_bytes,
                "sha256": snapshot.inventory_sha256,
            },
            "contract": dict(CONTRACT),
        }
    )


def _file_binding(name: str, payload: bytes, document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "model_seal_sha256": document["model_seal_sha256"],
    }


def _build_document(
    snapshot: SourceSnapshot,
    source_payload: bytes,
    source_document: Mapping[str, Any],
    tools: ToolSet,
    process: ProcessReceipt,
    protobuf: ProtobufSnapshot,
    timeout_seconds: int,
    output_authority_kind: str,
    authorized_tracer_sha256: str,
    output_validation: OutputValidation,
) -> dict[str, Any]:
    return immutable.seal_model(
        {
            "schema_version": BUILD_SCHEMA_VERSION,
            "source_manifest": _file_binding(
                SOURCE_MANIFEST_NAME, source_payload, source_document
            ),
            "source_identity": {
                "head_commit_sha1": snapshot.commit,
                "head_tree_sha1": snapshot.tree,
                "inventory_sha256": snapshot.inventory_sha256,
            },
            "tools": list(tools.identities),
            "tool_inventory_sha256": tools.inventory_sha256,
            "protobuf": {
                "prefix": str(protobuf.prefix),
                "protoc_version": "34.1",
                "libprotobuf_version": "34.1.0",
                "inventory_sha256": protobuf.inventory_sha256,
                "file_count": protobuf.file_count,
                "total_size_bytes": protobuf.total_size_bytes,
            },
            "invocation": {
                "tool_execution": "retained_descriptor_aliases.v1",
                "argv": list(process.argv),
                "cwd": process.cwd,
                "environment": [
                    {"name": name, "value": value}
                    for name, value in process.environment
                ],
            },
            "output_authority": {
                "kind": output_authority_kind,
                "expected_sha256": authorized_tracer_sha256,
            },
            "artifact_validation": {
                "elf": dict(output_validation.elf),
                "cuda": dict(output_validation.cuda),
            },
            "process": {
                "returncode": process.returncode,
                "timeout_seconds": timeout_seconds,
                "stdout": {
                    "sha256": process.stdout_sha256,
                    "size_bytes": process.stdout_size_bytes,
                },
                "stderr": {
                    "sha256": process.stderr_sha256,
                    "size_bytes": process.stderr_size_bytes,
                },
            },
            "contract": dict(CONTRACT),
        }
    )


def _artifact_document(
    snapshot: SourceSnapshot,
    source_payload: bytes,
    source_document: Mapping[str, Any],
    build_payload: bytes,
    build_document: Mapping[str, Any],
    tracer_sha256: str,
    tracer_size: int,
) -> dict[str, Any]:
    return immutable.seal_model(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_manifest": _file_binding(
                SOURCE_MANIFEST_NAME, source_payload, source_document
            ),
            "build_receipt": _file_binding(
                BUILD_RECEIPT_NAME, build_payload, build_document
            ),
            "source_identity": {
                "head_commit_sha1": snapshot.commit,
                "head_tree_sha1": snapshot.tree,
                "inventory_sha256": snapshot.inventory_sha256,
            },
            "artifact": {
                "name": TRACER_NAME,
                "sha256": tracer_sha256,
                "size_bytes": tracer_size,
                "source_kind": EXTERNAL_TRACER_SOURCE_KIND,
            },
            "bundle_files": list(BUNDLE_FILES),
            "contract": dict(CONTRACT),
        }
    )


def _schema_path(repo_root: Path, name: str) -> Path:
    return repo_root / SCHEMA_DIR_RELATIVE / name


def _load_schema(repo_root: Path, name: str) -> dict[str, Any]:
    path = _schema_path(repo_root, name)
    try:
        with immutable.open_retained_regular(path, max_bytes=1024 * 1024) as retained:
            value = immutable.strict_json_loads(
                retained.read_bytes(max_bytes=1024 * 1024), label=name
            )
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _authority(f"cannot load retained schema {name}: {exc}")
    if type(value) is not dict:
        _authority(f"schema must be a JSON object: {name}")
    try:
        jsonschema.Draft202012Validator.check_schema(value)
    except jsonschema.SchemaError as exc:
        _authority(f"invalid Draft-2020-12 schema {name}: {exc}")
    return value


def _validate_document(repo_root: Path, name: str, document: Mapping[str, Any]) -> None:
    schema = _load_schema(repo_root, name)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        detail = "; ".join(
            f"{error.json_path}: {error.message}" for error in errors[:8]
        )
        _authority(f"document does not satisfy {name}: {detail}")


def _strict_document(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = immutable.strict_json_loads(payload, label=label)
    except immutable.ImmutableArtifactError as exc:
        _authority(str(exc))
    if type(value) is not dict:
        _authority(f"{label} must be an exact JSON object")
    if immutable.pretty_json_bytes(value) != payload:
        _authority(f"{label} must use deterministic sorted pretty JSON plus LF")
    try:
        immutable.verify_model_seal(value)
    except immutable.ImmutableArtifactError as exc:
        _authority(f"{label} seal failed: {exc}")
    return value


def _require_equal(label: str, *values: Any) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        _authority(f"cross-document binding mismatch: {label}")


def _receipt_build_paths(invocation: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    external_top = _absolute_lexical(invocation["cwd"], label="receipt build cwd")
    if external_top.name != "nvbit_tracer" or external_top.parent.name != "tools":
        _authority("build receipt cwd is not the extracted tracer top level")
    build_root = external_top.parent.parent
    if build_root.parent != Path("/tmp"):
        _authority("build receipt cwd is not under one direct /tmp build root")
    _require_make_safe_path(build_root, label="receipt build root")
    return external_top, build_root, build_root / ".task630-tool-bin"


def _expected_receipt_environment(
    build: Mapping[str, Any],
    tool_by_role: Mapping[str, Mapping[str, Any]],
    *,
    private_bin: Path,
) -> dict[str, str]:
    protobuf_prefix = Path(build["protobuf"]["prefix"])
    cuda_root = Path(tool_by_role["nvcc"]["path"]).parent.parent
    return {
        "ARCH": "sm_86",
        "CPLUS_INCLUDE_PATH": f"{protobuf_prefix}/include",
        "CUDA_INSTALL_PATH": str(cuda_root),
        "CXX": str(_tool_alias(private_bin, "host_cxx")),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": f"{protobuf_prefix}/lib",
        "NVBIT_TEMPCOPY_BUILD_OK": "1",
        "NVCC_CCBIN": str(_tool_alias(private_bin, "host_cxx")),
        "PATH": f"{private_bin}:/usr/bin:/bin",
        "PB_PREFIX": str(protobuf_prefix),
        "PKG_CONFIG": str(_tool_alias(private_bin, "pkg_config")),
        "PKG_CONFIG_LIBDIR": (
            f"{protobuf_prefix}/lib/pkgconfig:"
            f"{protobuf_prefix}/share/pkgconfig"
        ),
        "PKG_CONFIG_PATH": "",
        "PROTOC": str(_tool_alias(private_bin, "protoc")),
        "SHELL": str(_tool_alias(private_bin, "bash")),
        "TZ": "UTC",
    }


def _verify_document_invariants(
    source: Mapping[str, Any],
    build: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    files = source["files"]
    paths = [row["path"] for row in files]
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")):
        _authority("source manifest paths are not in canonical byte order")
    if len(paths) != len(set(paths)):
        _authority("source manifest contains duplicate paths")
    expected_inventory = {
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "sha256": _sha256(immutable.canonical_json_bytes(files)),
    }
    _require_equal("source inventory", source.get("inventory"), expected_inventory)
    if any(path in PROHIBITED_TRACKED_OUTPUTS for path in paths):
        _authority("source manifest contains a prohibited tracked build output")
    if not REQUIRED_SOURCE_PATHS.issubset(paths):
        _authority("source manifest omits a required top-level build dependency")

    tools = build["tools"]
    roles = [row["role"] for row in tools]
    if roles != list(TOOL_ROLES):
        _authority("build receipt tool roles are not exact or ordered")
    _require_equal(
        "tool inventory",
        build.get("tool_inventory_sha256"),
        _sha256(immutable.canonical_json_bytes(tools)),
    )
    tool_by_role = {row["role"]: row for row in tools}
    invocation = build["invocation"]
    if invocation.get("tool_execution") != "retained_descriptor_aliases.v1":
        _authority("build receipt does not bind retained-descriptor execution")
    external_top, _build_root, private_bin = _receipt_build_paths(invocation)
    expected_argv = list(
        _build_argv(
            external_top=external_top,
            private_bin=private_bin,
            protobuf_prefix=Path(build["protobuf"]["prefix"]),
            source_paths=paths,
        )
    )
    if invocation["argv"] != expected_argv:
        _authority("build argv differs from the one exact pinned invocation")

    environment_rows = invocation["environment"]
    names = [row["name"] for row in environment_rows]
    if names != sorted(names) or len(names) != len(set(names)):
        _authority("build environment receipt is not sorted and unique")
    environment = {row["name"]: row["value"] for row in environment_rows}
    expected_environment = _expected_receipt_environment(
        build, tool_by_role, private_bin=private_bin
    )
    if environment != expected_environment:
        _authority("build environment differs from the exact closed contract")
    expected_output = build["output_authority"]
    if expected_output.get("kind") not in {
        CANDIDATE_AUTHORITY_KIND,
        REVIEWED_AUTHORITY_KIND,
    }:
        _authority("build receipt has an unknown output authority kind")
    _require_equal(
        "independent output digest",
        expected_output.get("expected_sha256"),
        artifact["artifact"]["sha256"],
    )
    validation = build["artifact_validation"]
    if validation.get("elf") != {
        "class": "ELF64",
        "data": "little_endian",
        "type": "ET_DYN",
        "machine": "EM_X86_64",
        "cuda_fatbin_section": ".nv_fatbin",
    }:
        _authority("build receipt ELF validation facts violate exact contract")
    cuda = validation.get("cuda", {})
    if (
        cuda.get("inspector") != "cuobjdump_13.2_list_elf.v1"
        or cuda.get("architectures") != ["sm_86"]
        or type(cuda.get("cubin_count")) is not int
        or cuda["cubin_count"] < 1
    ):
        _authority("build receipt CUDA validation facts violate exact contract")
    if artifact["artifact"]["source_kind"] != EXTERNAL_TRACER_SOURCE_KIND:
        _authority("artifact source kind differs from final capture authority")


def _verify_cross_bindings(
    source_payload: bytes,
    source: Mapping[str, Any],
    build_payload: bytes,
    build: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    _require_equal("contract", source.get("contract"), build.get("contract"), artifact.get("contract"), CONTRACT)
    source_binding = _file_binding(SOURCE_MANIFEST_NAME, source_payload, source)
    _require_equal(
        "source manifest file identity",
        build.get("source_manifest"),
        artifact.get("source_manifest"),
        source_binding,
    )
    _require_equal(
        "source identity",
        build.get("source_identity"),
        artifact.get("source_identity"),
        {
            "head_commit_sha1": source["repository"]["head_commit_sha1"],
            "head_tree_sha1": source["repository"]["head_tree_sha1"],
            "inventory_sha256": source["inventory"]["sha256"],
        },
    )
    _require_equal(
        "build receipt file identity",
        artifact.get("build_receipt"),
        _file_binding(BUILD_RECEIPT_NAME, build_payload, build),
    )
    _require_equal("bundle files", artifact.get("bundle_files"), list(BUNDLE_FILES))
    if build.get("process", {}).get("returncode") != 0:
        _authority("build receipt does not record a successful process")
    if any("/proc/self/fd/" in value for value in build["invocation"]["argv"]):
        _authority("build argv receipt persists a process-local FD path")
    for row in build["invocation"]["environment"]:
        if "/proc/self/fd/" in row["value"]:
            _authority("build environment receipt persists a process-local FD path")
    _verify_document_invariants(source, build, artifact)


def verify_admitted_bundle_documents(
    source_payload: bytes,
    build_payload: bytes,
    artifact_payload: bytes,
    *,
    repo_root: Path,
    expected_tracer_sha256: str,
    expected_tracer_size_bytes: int,
) -> dict[str, Any]:
    """Replay the admitted document chain without executing build tools."""

    expected_digest = _require_sha256(
        expected_tracer_sha256,
        label="independent expected tracer digest",
    )
    if (
        type(expected_tracer_size_bytes) is not int
        or expected_tracer_size_bytes <= 0
        or expected_tracer_size_bytes > EXTERNAL_TRACER_MAX_BYTES
    ):
        _authority("expected admitted tracer size is invalid")
    repo = _absolute_lexical(repo_root, label="repository root")
    try:
        repo = repo.resolve(strict=True)
    except OSError as exc:
        _authority(f"repository root is unavailable: {exc}")
    source = _strict_document(source_payload, label=SOURCE_MANIFEST_NAME)
    build = _strict_document(build_payload, label=BUILD_RECEIPT_NAME)
    artifact = _strict_document(artifact_payload, label=ARTIFACT_IDENTITY_NAME)
    _validate_document(repo, SOURCE_SCHEMA_NAME, source)
    _validate_document(repo, BUILD_SCHEMA_NAME, build)
    _validate_document(repo, ARTIFACT_SCHEMA_NAME, artifact)
    _verify_cross_bindings(
        source_payload,
        source,
        build_payload,
        build,
        artifact,
    )
    tracer = artifact["artifact"]
    _require_equal(
        "admitted tracer document authority",
        REVIEWED_AUTHORITY_KIND,
        build["output_authority"]["kind"],
    )
    _require_equal(
        "admitted tracer document digest",
        expected_digest,
        build["output_authority"]["expected_sha256"],
        tracer["sha256"],
    )
    _require_equal(
        "admitted tracer document size",
        expected_tracer_size_bytes,
        tracer["size_bytes"],
    )
    return dict(artifact)


def _read_bundle_file(bundle: Path, name: str, *, max_bytes: int) -> bytes:
    try:
        with immutable.open_retained_regular(
            bundle / name, max_bytes=max_bytes
        ) as retained:
            return retained.read_bytes(max_bytes=max_bytes)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _authority(f"cannot retain bundle file {name}: {exc}")


def _verify_bundle_with_inspector(
    bundle_dir: Path,
    *,
    repo_root: Path,
    expected_tracer_sha256: str | None,
    required_authority_kind: str,
    inspector: immutable.RetainedExecutable,
    deadline: OperationDeadline,
) -> dict[str, Any]:
    deadline.check(label="bundle verification setup")
    expected_digest = (
        None
        if expected_tracer_sha256 is None
        else _require_sha256(
            expected_tracer_sha256, label="independent expected tracer digest"
        )
    )
    bundle = require_external_non_git_root(
        _absolute_lexical(bundle_dir, label="bundle directory"),
        repo_root=repo_root,
        label="external tracer bundle",
    )
    source_payload = _read_bundle_file(
        bundle, SOURCE_MANIFEST_NAME, max_bytes=MAX_SOURCE_BYTES
    )
    build_payload = _read_bundle_file(
        bundle, BUILD_RECEIPT_NAME, max_bytes=16 * 1024 * 1024
    )
    artifact_payload = _read_bundle_file(
        bundle, ARTIFACT_IDENTITY_NAME, max_bytes=4 * 1024 * 1024
    )
    deadline.check(label="bundle document hashing")
    source = _strict_document(source_payload, label=SOURCE_MANIFEST_NAME)
    build = _strict_document(build_payload, label=BUILD_RECEIPT_NAME)
    artifact = _strict_document(artifact_payload, label=ARTIFACT_IDENTITY_NAME)
    _validate_document(repo_root, SOURCE_SCHEMA_NAME, source)
    _validate_document(repo_root, BUILD_SCHEMA_NAME, build)
    _validate_document(repo_root, ARTIFACT_SCHEMA_NAME, artifact)
    _verify_cross_bindings(source_payload, source, build_payload, build, artifact)
    deadline.check(label="bundle schema and cross-binding validation")

    tool_by_role = {row["role"]: row for row in build["tools"]}
    recorded_inspector = tool_by_role["cuobjdump"]
    if recorded_inspector != {
        "role": "cuobjdump",
        "path": str(inspector.path),
        "sha256": inspector.sha256,
        "size_bytes": inspector.snapshot[4],
    }:
        _authority("verification cuobjdump differs from exact build inspector")
    tracer_identity = artifact["artifact"]
    if build["output_authority"]["kind"] != required_authority_kind:
        _authority(
            "bundle output authority phase mismatch: "
            f"expected={required_authority_kind}, "
            f"observed={build['output_authority']['kind']}"
        )
    if expected_digest is not None:
        _require_equal(
            "independent caller tracer digest",
            expected_digest,
            build["output_authority"]["expected_sha256"],
            tracer_identity["sha256"],
        )

    tracer = require_external_non_git_artifact(
        bundle / TRACER_NAME,
        root=bundle,
        repo_root=repo_root,
        label="published NVBit tracer",
        max_bytes=EXTERNAL_TRACER_MAX_BYTES,
    )
    authority = require_external_tracer_authority(
        tracer,
        tracer_identity["sha256"],
        repo_root=repo_root,
        environment={},
    )
    try:
        with immutable.open_retained_regular(
            tracer, max_bytes=EXTERNAL_TRACER_MAX_BYTES
        ) as retained:
            if retained.size_bytes != tracer_identity["size_bytes"]:
                _authority("published tracer size differs from artifact identity")
            observed_validation = _validate_output_contract(
                retained,
                inspector,
                expected_sha256=expected_digest,
                cwd=bundle,
                deadline=deadline,
            )
            _require_equal(
                "target-free artifact validation",
                build["artifact_validation"],
                {
                    "elf": dict(observed_validation.elf),
                    "cuda": dict(observed_validation.cuda),
                },
            )
            verify_retained_tracer_authority(retained, authority)
    except (ExternalTracerAuthorityError, immutable.ImmutableArtifactError) as exc:
        _authority(f"published tracer final authority failed: {exc}")

    material = (
        immutable.ImmutableFile.from_bytes(ARTIFACT_IDENTITY_NAME, artifact_payload),
        immutable.ImmutableFile.from_bytes(BUILD_RECEIPT_NAME, build_payload),
        immutable.ImmutableFile.from_bytes(SOURCE_MANIFEST_NAME, source_payload),
        immutable.ImmutableFile(
            TRACER_NAME,
            tracer_identity["sha256"],
            tracer_identity["size_bytes"],
            source_path=tracer,
        ),
    )
    try:
        immutable.verify_published_files(bundle, material)
    except immutable.ImmutableArtifactError as exc:
        _authority(f"published bundle inventory failed: {exc}")
    deadline.check(label="bundle inventory verification")
    return dict(artifact)


def verify_bundle(
    bundle_dir: Path,
    *,
    repo_root: Path,
    expected_tracer_sha256: str,
    cuobjdump: Path | immutable.RetainedExecutable,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    _validate_timeout_seconds(timeout_seconds)
    deadline = OperationDeadline.start(timeout_seconds)
    with _deadline_alarm(deadline, label="external bundle verification"):
        return _verify_bundle_operation(
            bundle_dir,
            repo_root=repo_root,
            expected_tracer_sha256=expected_tracer_sha256,
            cuobjdump=cuobjdump,
            deadline=deadline,
        )


def _verify_bundle_operation(
    bundle_dir: Path,
    *,
    repo_root: Path,
    expected_tracer_sha256: str,
    cuobjdump: Path | immutable.RetainedExecutable,
    deadline: OperationDeadline,
) -> dict[str, Any]:
    repo = _absolute_lexical(repo_root, label="repository root")
    try:
        repo = repo.resolve(strict=True)
    except OSError as exc:
        _authority(f"repository root is unavailable: {exc}")
    if isinstance(cuobjdump, immutable.RetainedExecutable):
        return _verify_bundle_with_inspector(
            bundle_dir,
            repo_root=repo,
            expected_tracer_sha256=expected_tracer_sha256,
            required_authority_kind=REVIEWED_AUTHORITY_KIND,
            inspector=cuobjdump,
            deadline=deadline,
        )
    with contextlib.ExitStack() as stack:
        inspector = _open_external_executable(
            stack,
            role="cuobjdump",
            raw_path=cuobjdump,
            repo_root=repo,
        )
        probe_environment = {
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        }
        observed_version = _probe_pin(
            inspector,
            ("--version",),
            cwd=repo,
            environment=probe_environment,
            label="cuobjdump",
            deadline=deadline,
        )
        if re.search(rb"release[ ,]+13\.2(?:[ ,]|$)", observed_version) is None:
            _authority("cuobjdump pin must report CUDA release 13.2")
        return _verify_bundle_with_inspector(
            bundle_dir,
            repo_root=repo,
            expected_tracer_sha256=expected_tracer_sha256,
            required_authority_kind=REVIEWED_AUTHORITY_KIND,
            inspector=inspector,
            deadline=deadline,
        )


def admit_candidate(
    candidate_dir: Path,
    bundle_dir: Path,
    *,
    repo_root: Path,
    expected_tracer_sha256: str,
    cuobjdump: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    _validate_timeout_seconds(timeout_seconds)
    deadline = OperationDeadline.start(timeout_seconds)
    with _deadline_alarm(deadline, label="external candidate admission"):
        return _admit_candidate_operation(
            candidate_dir,
            bundle_dir,
            repo_root=repo_root,
            expected_tracer_sha256=expected_tracer_sha256,
            cuobjdump=cuobjdump,
            deadline=deadline,
        )


def _admit_candidate_operation(
    candidate_dir: Path,
    bundle_dir: Path,
    *,
    repo_root: Path,
    expected_tracer_sha256: str,
    cuobjdump: Path,
    deadline: OperationDeadline,
) -> Path:
    expected_digest = _require_sha256(
        expected_tracer_sha256,
        label="independent reviewed candidate tracer digest",
    )
    repo = _absolute_lexical(repo_root, label="repository root")
    try:
        repo = repo.resolve(strict=True)
    except OSError as exc:
        _authority(f"repository root is unavailable: {exc}")
    candidate = require_external_non_git_root(
        _absolute_lexical(candidate_dir, label="candidate bundle directory"),
        repo_root=repo,
        label="external tracer candidate bundle",
    )
    destination = _absolute_lexical(bundle_dir, label="bundle directory")
    output_root = require_external_non_git_root(
        destination.parent,
        repo_root=repo,
        label="external admitted-bundle publication root",
    )
    if destination.name in {"", ".", ".."} or destination.name.startswith("."):
        _authority("admitted bundle directory requires one public final name")
    try:
        name = immutable.safe_relative_path(destination.name)
    except immutable.ImmutableArtifactError as exc:
        _authority(f"admitted bundle directory name is unsafe: {exc}")
    if len(name.parts) != 1 or len(os.fsencode(destination.name)) > 246:
        _authority("admitted bundle directory name is not one safe component")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        _authority(f"admitted bundle destination status is unavailable: {exc}")
    else:
        _authority(f"admitted bundle destination already exists: {destination}")
    deadline.check(label="candidate admission preflight")

    with contextlib.ExitStack() as stack:
        inspector = _open_external_executable(
            stack,
            role="cuobjdump",
            raw_path=cuobjdump,
            repo_root=repo,
        )
        probe_environment = {
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        }
        observed_version = _probe_pin(
            inspector,
            ("--version",),
            cwd=repo,
            environment=probe_environment,
            label="cuobjdump",
            deadline=deadline,
        )
        if re.search(rb"release[ ,]+13\.2(?:[ ,]|$)", observed_version) is None:
            _authority("cuobjdump pin must report CUDA release 13.2")
        _verify_bundle_with_inspector(
            candidate,
            repo_root=repo,
            expected_tracer_sha256=expected_digest,
            required_authority_kind=CANDIDATE_AUTHORITY_KIND,
            inspector=inspector,
            deadline=deadline,
        )

        source_payload = _read_bundle_file(
            candidate, SOURCE_MANIFEST_NAME, max_bytes=MAX_SOURCE_BYTES
        )
        build_payload = _read_bundle_file(
            candidate, BUILD_RECEIPT_NAME, max_bytes=16 * 1024 * 1024
        )
        artifact_payload = _read_bundle_file(
            candidate, ARTIFACT_IDENTITY_NAME, max_bytes=4 * 1024 * 1024
        )
        source_document = _strict_document(
            source_payload, label=SOURCE_MANIFEST_NAME
        )
        build_document = _strict_document(
            build_payload, label=BUILD_RECEIPT_NAME
        )
        artifact_document = _strict_document(
            artifact_payload, label=ARTIFACT_IDENTITY_NAME
        )
        build_document["output_authority"] = {
            "kind": REVIEWED_AUTHORITY_KIND,
            "expected_sha256": expected_digest,
        }
        build_document["model_seal_sha256"] = immutable.model_seal(
            build_document
        )
        admitted_build_payload = immutable.pretty_json_bytes(build_document)
        artifact_document["build_receipt"] = _file_binding(
            BUILD_RECEIPT_NAME,
            admitted_build_payload,
            build_document,
        )
        artifact_document["model_seal_sha256"] = immutable.model_seal(
            artifact_document
        )
        admitted_artifact_payload = immutable.pretty_json_bytes(artifact_document)
        _validate_document(repo, BUILD_SCHEMA_NAME, build_document)
        _validate_document(repo, ARTIFACT_SCHEMA_NAME, artifact_document)
        _verify_cross_bindings(
            source_payload,
            source_document,
            admitted_build_payload,
            build_document,
            artifact_document,
        )
        deadline.check(label="candidate admission document sealing")

        tracer_path = require_external_non_git_artifact(
            candidate / TRACER_NAME,
            root=candidate,
            repo_root=repo,
            label="candidate NVBit tracer",
            max_bytes=EXTERNAL_TRACER_MAX_BYTES,
        )
        try:
            tracer = stack.enter_context(
                immutable.open_retained_regular(
                    tracer_path, max_bytes=EXTERNAL_TRACER_MAX_BYTES
                )
            )
        except immutable.ImmutableArtifactError as exc:
            _authority(f"cannot retain candidate tracer: {exc}")
        if tracer.sha256 != expected_digest:
            _authority("candidate tracer changed after independent review")
        material = (
            immutable.ImmutableFile.from_bytes(
                ARTIFACT_IDENTITY_NAME, admitted_artifact_payload
            ),
            immutable.ImmutableFile.from_bytes(
                BUILD_RECEIPT_NAME, admitted_build_payload
            ),
            immutable.ImmutableFile.from_bytes(
                SOURCE_MANIFEST_NAME, source_payload
            ),
            immutable.ImmutableFile(
                TRACER_NAME,
                tracer.sha256,
                tracer.size_bytes,
                source_path=tracer.path,
            ),
        )

        def reauthorize(*_unused: Any) -> None:
            deadline.check(label="candidate admission reauthorization")
            tracer.verify_unchanged()
            inspector.verify_unchanged()
            _verify_bundle_with_inspector(
                candidate,
                repo_root=repo,
                expected_tracer_sha256=expected_digest,
                required_authority_kind=CANDIDATE_AUTHORITY_KIND,
                inspector=inspector,
                deadline=deadline,
            )

        def finalize_after_commit(*_unused: Any) -> None:
            reauthorize()
            _verify_bundle_with_inspector(
                output_root / destination.name,
                repo_root=repo,
                expected_tracer_sha256=expected_digest,
                required_authority_kind=REVIEWED_AUTHORITY_KIND,
                inspector=inspector,
                deadline=deadline,
            )
            stack.close()
            deadline.check(label="admitted bundle postcommit capability closure")

        try:
            published = immutable.publish_directory(
                output_root,
                destination.name,
                material,
                before_commit=reauthorize,
                after_commit=finalize_after_commit,
                accept_existing=False,
            )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            _authority(f"admitted bundle no-replace publication failed: {exc}")
    return published


def build_bundle(request: BuildRequest) -> Path:
    _validate_request_bounds(request)
    deadline = OperationDeadline.start(request.timeout_seconds)
    with _deadline_alarm(deadline, label="external candidate build"):
        return _build_bundle_operation(request, deadline=deadline)


def _build_bundle_operation(
    request: BuildRequest, *, deadline: OperationDeadline
) -> Path:
    repo_root = _absolute_lexical(request.repo_root, label="repository root")
    try:
        repo_root = repo_root.resolve(strict=True)
    except OSError as exc:
        _authority(f"repository root is unavailable: {exc}")
    _preflight_external_inputs(request, repo_root=repo_root)
    bundle_dir = _absolute_lexical(request.bundle_dir, label="bundle directory")
    output_root = require_external_non_git_root(
        bundle_dir.parent,
        repo_root=repo_root,
        label="external bundle publication root",
    )
    if bundle_dir.name in {"", ".", ".."}:
        _authority("bundle directory requires one final name")
    try:
        bundle_name = immutable.safe_relative_path(bundle_dir.name)
    except immutable.ImmutableArtifactError as exc:
        _authority(f"bundle directory name is unsafe: {exc}")
    if (
        len(bundle_name.parts) != 1
        or bundle_dir.name.startswith(".")
        or len(os.fsencode(bundle_dir.name)) > 246
        or bundle_dir.name.endswith(
            (".staging", ".m5-partial", ".m6-partial", ".ncu-details-partial")
        )
        or ".partial-" in bundle_dir.name
    ):
        _authority("bundle directory name is not an admitted publication name")
    try:
        bundle_dir.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        _authority(f"bundle destination status is unavailable: {exc}")
    else:
        _authority(f"bundle destination already exists: {bundle_dir}")
    deadline.check(label="external candidate preflight")

    with contextlib.ExitStack() as stack:
        tool_paths = {
            "git": request.git,
            "bash": request.bash,
            "make": request.make,
            "nvcc": request.nvcc,
            "ptxas": request.ptxas,
            "cuobjdump": request.cuobjdump,
            "host_cxx": request.host_cxx,
            "protoc": request.protoc,
            "pkg_config": request.pkg_config,
        }
        tools = _open_tool_set(stack, tool_paths, repo_root=repo_root)
        protobuf_prefix = _validate_protobuf_prefix(
            request.protobuf_prefix, repo_root=repo_root
        )
        _retain_protobuf_generation(stack, protobuf_prefix)
        protobuf = _snapshot_protobuf(protobuf_prefix)
        deadline.check(label="protobuf inventory hashing")
        protobuf_files = _retain_protobuf_files(stack, protobuf)
        git_tool = tools.retained["git"]
        commit, tree = _repository_identity(
            git_tool, repo_root, deadline=deadline
        )
        source = _read_source_snapshot(
            git_tool,
            repo_root,
            commit=commit,
            tree=tree,
            deadline=deadline,
        )
        _reverify_repository(
            git_tool,
            repo_root,
            commit=commit,
            tree=tree,
            deadline=deadline,
        )
        _verify_tools(tools)
        deadline.check(label="source and tool authority capture")

        build_root = _create_fresh_direct_tmp_root(request.build_root)
        _write_extracted_sources(build_root, source)
        _verify_extracted_sources(build_root, source)
        _verify_exact_discovered_source_sets(build_root, source)
        deadline.check(label="source extraction and hashing")
        source_generation = _retain_extracted_input_generation(
            stack, root=build_root, snapshot=source
        )
        external_top = build_root / "tools/nvbit_tracer"
        require_external_non_git_root(
            external_top, repo_root=repo_root, label="external tracer source root"
        )
        private_bin = _install_tool_aliases(build_root, tools)
        alias_generation = _retain_tool_alias_generation(
            stack, build_root=build_root
        )
        environment = _build_environment(
            request, private_bin=private_bin, protobuf_prefix=protobuf_prefix
        )
        _verify_pinned_toolchain(
            tools,
            protobuf_prefix=protobuf_prefix,
            environment=environment,
            deadline=deadline,
        )
        _reverify_repository(
            git_tool,
            repo_root,
            commit=commit,
            tree=tree,
            deadline=deadline,
        )
        _verify_tools(tools)
        _verify_extracted_generation(source_generation)
        _verify_generation_monitor(
            alias_generation, label="retained tool-alias generation"
        )
        _verify_extracted_sources(build_root, source)
        _verify_exact_discovered_source_sets(build_root, source)
        _verify_generation_monitor(
            protobuf_files, label="pinned protobuf prefix"
        )
        _reverify_protobuf(protobuf)
        deadline.check(label="pre-build authority revalidation")

        process = _run_external_build(
            request,
            external_top=external_top,
            private_bin=private_bin,
            tools=tools,
            source=source,
            environment=environment,
            protobuf_prefix=protobuf_prefix,
            deadline=deadline,
        )
        _reverify_repository(
            git_tool,
            repo_root,
            commit=commit,
            tree=tree,
            deadline=deadline,
        )
        _verify_tools(tools)
        _verify_extracted_generation(source_generation)
        _verify_generation_monitor(
            alias_generation, label="retained tool-alias generation"
        )
        _verify_extracted_sources(build_root, source)
        _verify_exact_discovered_source_sets(build_root, source)
        _verify_generation_monitor(
            protobuf_files, label="pinned protobuf prefix"
        )
        _reverify_protobuf(protobuf)
        _verify_pinned_toolchain(
            tools,
            protobuf_prefix=protobuf_prefix,
            environment=environment,
            deadline=deadline,
        )
        deadline.check(label="post-build authority revalidation")

        output_path = require_external_non_git_artifact(
            external_top / "tracer_tool/tracer_tool.so",
            root=build_root,
            repo_root=repo_root,
            label="external build tracer output",
            max_bytes=EXTERNAL_TRACER_MAX_BYTES,
        )
        try:
            output = stack.enter_context(
                immutable.open_retained_regular(
                    output_path, max_bytes=EXTERNAL_TRACER_MAX_BYTES
                )
            )
        except immutable.ImmutableArtifactError as exc:
            _authority(f"cannot retain external build output: {exc}")
        output_validation = _validate_output_contract(
            output,
            tools.retained["cuobjdump"],
            expected_sha256=None,
            cwd=external_top,
            deadline=deadline,
        )

        source_document = _source_document(source)
        source_payload = immutable.pretty_json_bytes(source_document)
        build_document = _build_document(
            source,
            source_payload,
            source_document,
            tools,
            process,
            protobuf,
            request.timeout_seconds,
            CANDIDATE_AUTHORITY_KIND,
            output.sha256,
            output_validation,
        )
        build_payload = immutable.pretty_json_bytes(build_document)
        artifact_document = _artifact_document(
            source,
            source_payload,
            source_document,
            build_payload,
            build_document,
            output.sha256,
            output.size_bytes,
        )
        artifact_payload = immutable.pretty_json_bytes(artifact_document)
        _validate_document(repo_root, SOURCE_SCHEMA_NAME, source_document)
        _validate_document(repo_root, BUILD_SCHEMA_NAME, build_document)
        _validate_document(repo_root, ARTIFACT_SCHEMA_NAME, artifact_document)
        _verify_cross_bindings(
            source_payload,
            source_document,
            build_payload,
            build_document,
            artifact_document,
        )
        deadline.check(label="candidate document sealing and schema validation")

        material = (
            immutable.ImmutableFile.from_bytes(
                ARTIFACT_IDENTITY_NAME, artifact_payload
            ),
            immutable.ImmutableFile.from_bytes(BUILD_RECEIPT_NAME, build_payload),
            immutable.ImmutableFile.from_bytes(
                SOURCE_MANIFEST_NAME, source_payload
            ),
            immutable.ImmutableFile(
                TRACER_NAME,
                output.sha256,
                output.size_bytes,
                source_path=output.path,
            ),
        )

        def reauthorize(*_unused: Any) -> None:
            deadline.check(label="candidate publication reauthorization")
            _reverify_repository(
                git_tool,
                repo_root,
                commit=commit,
                tree=tree,
                deadline=deadline,
            )
            _verify_tools(tools)
            _verify_extracted_generation(source_generation)
            _verify_generation_monitor(
                alias_generation, label="retained tool-alias generation"
            )
            _verify_extracted_sources(build_root, source)
            _verify_exact_discovered_source_sets(build_root, source)
            _verify_generation_monitor(
                protobuf_files, label="pinned protobuf prefix"
            )
            _reverify_protobuf(protobuf)
            output.verify_unchanged()
            deadline.check(label="candidate publication reauthorization")

        def finalize_after_commit(*_unused: Any) -> None:
            reauthorize()
            # This full sealed-document, inventory, and final capture-authority
            # verification runs while publish_directory still owns rollback
            # guards for the committed generation. Any failure retires that exact
            # bundle before it can remain in the active output namespace.
            _verify_bundle_with_inspector(
                output_root / bundle_dir.name,
                repo_root=repo_root,
                expected_tracer_sha256=None,
                required_authority_kind=CANDIDATE_AUTHORITY_KIND,
                inspector=tools.retained["cuobjdump"],
                deadline=deadline,
            )
            # Capability revalidation/close is itself postcommit authority.
            # Empty the enclosing ExitStack here, while the directory publisher
            # can still retire the exact committed generation; the later with-
            # statement exit is then a no-op rather than an unguarded failure.
            stack.close()
            deadline.check(label="candidate postcommit capability closure")

        try:
            published = immutable.publish_directory(
                output_root,
                bundle_dir.name,
                material,
                before_commit=reauthorize,
                after_commit=finalize_after_commit,
                accept_existing=False,
            )
        except (OSError, immutable.ImmutableArtifactError) as exc:
            _authority(f"external bundle no-replace publication failed: {exc}")

    return published


def _add_tool_arguments(parser: argparse.ArgumentParser) -> None:
    for option in (
        "git",
        "bash",
        "make",
        "nvcc",
        "ptxas",
        "cuobjdump",
        "protoc",
    ):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--host-cxx", required=True, type=Path)
    parser.add_argument("--pkg-config", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-candidate", help="build and publish one review candidate"
    )
    build.add_argument("--repo-root", required=True, type=Path)
    build.add_argument("--build-root", required=True, type=Path)
    build.add_argument("--bundle-dir", required=True, type=Path)
    build.add_argument("--protobuf-prefix", required=True, type=Path)
    build.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    build.add_argument(
        "--max-output-bytes", type=int, default=DEFAULT_OUTPUT_BYTES
    )
    _add_tool_arguments(build)
    admit = subparsers.add_parser(
        "admit", help="admit one independently reviewed candidate"
    )
    admit.add_argument("--repo-root", required=True, type=Path)
    admit.add_argument("--candidate-dir", required=True, type=Path)
    admit.add_argument("--bundle-dir", required=True, type=Path)
    admit.add_argument("--expected-tracer-sha256", required=True)
    admit.add_argument("--cuobjdump", required=True, type=Path)
    admit.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    verify = subparsers.add_parser("verify", help="verify one admitted bundle")
    verify.add_argument("--repo-root", required=True, type=Path)
    verify.add_argument("--bundle-dir", required=True, type=Path)
    verify.add_argument("--expected-tracer-sha256", required=True)
    verify.add_argument("--cuobjdump", required=True, type=Path)
    verify.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-candidate":
            request = BuildRequest(
                repo_root=args.repo_root,
                build_root=args.build_root,
                bundle_dir=args.bundle_dir,
                git=args.git,
                bash=args.bash,
                make=args.make,
                nvcc=args.nvcc,
                ptxas=args.ptxas,
                cuobjdump=args.cuobjdump,
                host_cxx=args.host_cxx,
                protoc=args.protoc,
                pkg_config=args.pkg_config,
                protobuf_prefix=args.protobuf_prefix,
                timeout_seconds=args.timeout_seconds,
                max_output_bytes=args.max_output_bytes,
            )
            published = build_bundle(request)
            print(published)
        elif args.command == "admit":
            published = admit_candidate(
                args.candidate_dir,
                args.bundle_dir,
                repo_root=args.repo_root,
                expected_tracer_sha256=args.expected_tracer_sha256,
                cuobjdump=args.cuobjdump,
                timeout_seconds=args.timeout_seconds,
            )
            print(published)
        else:
            verify_bundle(
                args.bundle_dir,
                repo_root=args.repo_root,
                expected_tracer_sha256=args.expected_tracer_sha256,
                cuobjdump=args.cuobjdump,
                timeout_seconds=args.timeout_seconds,
            )
            print(args.bundle_dir)
        return 0
    except ExternalTracerBuildProcessError as exc:
        print(f"PROCESS_ERROR: {exc}", file=sys.stderr)
        return 1
    except (
        ExternalTracerBuildAuthorityError,
        ExternalTracerAuthorityError,
        immutable.ImmutableArtifactError,
        OSError,
    ) as exc:
        print(f"AUTHORITY_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
