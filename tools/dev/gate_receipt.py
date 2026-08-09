#!/usr/bin/env python3
"""Run one repository gate as an exact child and seal its observed result.

The canonical JSON seal and read-only generation provide consistency, not a
trust anchor against an owner rewrite. Authority requires supplying the
content-manifest SHA-256 from an independently trusted surface; hidden Git
administration semantics additionally rely on the clean, checked-in producer
or a current-source replay.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import decimal
import errno
import hashlib
import json
import math
import os
import platform
import re
import resource
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import protected_paths  # noqa: E402
from tools.local_labels.immutable_artifact import (  # noqa: E402
    DirectoryGenerationFile,
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    PinnedDirectoryReader,
    PinnedPathSetMonitor,
    create_exclusive_directory_stage,
    inspect_immutable_directory_generation,
    open_retained_executable,
    open_retained_regular,
    pretty_json_bytes,
    require_file_descriptor_headroom,
    safe_relative_path,
    seal_model,
    sha256_file,
    strict_json_loads,
    verify_model_seal,
    write_regular_bytes_at,
)

SCHEMA_VERSION = "gate_receipt.v3"
RECEIPT_NAME = "gate_receipt.json"
STDOUT_NAME = "stdout.bin"
STDERR_NAME = "stderr.bin"
_MAX_STREAM_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_STATUS_BYTES = 4 * 1024 * 1024
_MAX_GIT_BRANCH_BYTES = 1024
_MAX_ARG_BYTES = 256 * 1024
_MAX_ENV_BYTES = 256 * 1024
_MAX_TRACKED_ENTRIES = 32_768
_MAX_TRACKED_DIRECTORIES = 32_768
_MAX_TRACKED_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_TRACKED_RELATIVE_PATH_BYTES = 2 * 1024 * 1024
_MAX_TRACKED_DIRECTORY_PATH_BYTES = 2 * 1024 * 1024
_MAX_TRACKED_PATH_BYTES = 4096
_MAX_TRACKED_FILE_BYTES = 64 * 1024 * 1024
_MAX_TRACKED_TOTAL_FILE_BYTES = 1024 * 1024 * 1024
_MAX_TRACKED_WORKTREE_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_GIT_INDEX_BYTES = 16 * 1024 * 1024
_MAX_GIT_ADMIN_FILE_BYTES = 16 * 1024 * 1024
_MAX_GIT_CONFIG_RECEIPT_BYTES = 64 * 1024
_MAX_GIT_PACKED_REFS_RECEIPT_BYTES = 128 * 1024
_MAX_GIT_CONFIG_KEYS = 32_768
_MAX_GIT_PROCESS_STDOUT_BYTES = 4 * 1024 * 1024
_MAX_GIT_PROCESS_STDERR_BYTES = 64 * 1024
_MAX_GIT_PROCESS_SECONDS = 30.0
_MAX_GIT_REF_BYTES = 4096
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_TIMEOUT_NS = _MAX_TIMEOUT_SECONDS * 1_000_000_000
_POLL_SECONDS = 0.02
_PIPE_READ_QUANTUM = 64 * 1024
_PIPE_DRAIN_TIMEOUT_NS = 5 * 1_000_000_000
_GENERATION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SECRET_FRAGMENTS = (
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
_DEFAULT_ENVIRONMENT = (
    "CARGO_HOME",
    "CC",
    "CPATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CXX",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "LOGNAME",
    "NVCC_CCBIN",
    "PATH",
    "PKG_CONFIG_PATH",
    "RUSTFLAGS",
    "RUSTUP_HOME",
    "TZ",
    "USER",
)
_ALLOWED_ENVIRONMENT = frozenset(_DEFAULT_ENVIRONMENT)
_SUPERVISION_POLICY = {
    "child_identity": "waitid_wnowait_then_wait4_exact_leader_v1",
    "reaper_authority": "single_native_thread_default_sigchld_v1",
    "containment": "original_process_group_only_descendant_setsid_or_setpgid_escape_not_excluded",
    "stream_capture": "parent_owned_unix_socketpairs_cumulative_limit_bounded_eof_v1",
    "deadline": "terminal_observation_wins_at_exact_deadline_otherwise_strict_v1",
    "termination_signals": "deferred_default_hup_quit_term_until_exact_reap_v1",
    "realtime": "timestamps_observational_monotonic_only_ordering_v1",
    "environment": "closed_audited_allowlist_v1",
    "temporary_paths": "fresh_declared_cache_roots_only_not_complete_cache_isolation",
}
_BLOCKABLE_SIGNALS = frozenset(
    value
    for value in signal.valid_signals()
    if value not in {signal.SIGKILL, signal.SIGSTOP}
)
_DEFERRED_TERMINATION_SIGNALS = (
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGTERM,
)


class GateReceiptError(RuntimeError):
    """Typed refusal for gate execution or receipt verification."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.context = dict(context or {})


@dataclasses.dataclass(frozen=True, slots=True)
class _ChildObservation:
    pid: int
    waited_pid: int
    started_at_unix_ns: int
    finished_at_unix_ns: int
    monotonic_start_ns: int
    monotonic_stop_ns: int
    requested_timeout_ns: int
    monotonic_deadline_ns: int
    monotonic_decision_ns: int
    decision: str
    wait_status: int
    return_code: int
    timed_out: bool
    usage: resource.struct_rusage
    stdout: bytes
    stderr: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class GateRunResult:
    generation_path: Path
    document: dict[str, Any]
    shell_return_code: int
    content_manifest_sha256: str
    filesystem_generation_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class _TrackedInputManifest:
    paths: tuple[str, ...]
    mode_by_path: tuple[tuple[str, str], ...]
    document: dict[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class _RetainedTemporaryRoot:
    path: Path
    binding: dict[str, str]
    reader: PinnedDirectoryReader


@dataclasses.dataclass(frozen=True, slots=True)
class _MonitoredGateExecution:
    observation: _ChildObservation
    environment: dict[str, str]
    executable: dict[str, Any]
    temporary_root: dict[str, Any]
    source: dict[str, Any]


def _fail(code: str, detail: str, **context: Any) -> None:
    raise GateReceiptError(code, detail, context=context)


def _timeout_nanoseconds(value: str | int | float) -> int:
    if isinstance(value, bool):
        _fail("GATE_TIMEOUT_INVALID", "gate timeout must be an exact positive decimal")
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            _fail("GATE_TIMEOUT_INVALID", "gate timeout must be finite")
        text = repr(value)
    elif isinstance(value, str):
        text = value
    else:
        _fail("GATE_TIMEOUT_INVALID", "gate timeout has an unsupported type")
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", text):
        _fail("GATE_TIMEOUT_INVALID", "gate timeout must be an exact positive decimal")
    try:
        scaled = decimal.Decimal(text) * decimal.Decimal(1_000_000_000)
    except decimal.InvalidOperation:
        _fail("GATE_TIMEOUT_INVALID", "gate timeout decimal is invalid")
    integral = scaled.to_integral_value()
    if scaled != integral:
        _fail("GATE_TIMEOUT_INVALID", "gate timeout must resolve to whole nanoseconds")
    timeout_ns = int(integral)
    if not 1 <= timeout_ns <= _MAX_TIMEOUT_NS:
        _fail("GATE_TIMEOUT_INVALID", "gate timeout is outside the admitted range")
    return timeout_ns


def _canonical_path(path: str | Path, *, role: str, directory: bool) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        _fail("GATE_PATH_NOT_ABSOLUTE", f"{role} must be an absolute path", path=str(path))
    lexical = Path(os.path.abspath(os.fspath(requested)))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _fail("GATE_PATH_UNAVAILABLE", f"{role} is unavailable", path=str(path), errno=exc.errno)
    if lexical != resolved:
        _fail("GATE_PATH_NOT_CANONICAL", f"{role} must not traverse a symlink", path=str(path))
    try:
        metadata = resolved.stat()
    except OSError as exc:
        _fail("GATE_PATH_UNAVAILABLE", f"{role} is unreadable", path=str(path), errno=exc.errno)
    if directory != stat.S_ISDIR(metadata.st_mode):
        kind = "directory" if directory else "regular file"
        _fail("GATE_PATH_TYPE_INVALID", f"{role} must be a {kind}", path=str(path))
    return resolved


def _external_directory(path: str | Path, *, role: str) -> Path:
    resolved = _canonical_path(path, role=role, directory=True)
    try:
        inside_worktree = protected_paths.is_inside_git_worktree(resolved)
    except protected_paths.ProtectedPathError as exc:
        _fail(
            "GATE_EXTERNAL_PATH_STATUS_UNAVAILABLE",
            f"{role} Git-worktree containment is indeterminate",
            path=str(resolved),
            error=str(exc),
        )
    if inside_worktree:
        _fail("GATE_EXTERNAL_PATH_IN_WORKTREE", f"{role} is inside a Git worktree", path=str(resolved))
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _require_disjoint_path(
    path: Path,
    *,
    role: str,
    others: Sequence[tuple[str, Path]],
    code: str,
) -> None:
    for other_role, other_path in others:
        if _paths_overlap(path, other_path):
            _fail(
                code,
                f"{role} overlaps {other_role}",
                path=str(path),
                other_path=str(other_path),
            )


def _directory_binding(path: Path, *, descriptor: int | None = None) -> dict[str, str]:
    value = os.fstat(descriptor) if descriptor is not None else path.stat()
    if not stat.S_ISDIR(value.st_mode):
        _fail("GATE_PATH_TYPE_INVALID", "retained path is not a directory", path=str(path))
    return {
        "path": str(path),
        "device": str(value.st_dev),
        "inode": str(value.st_ino),
        "mode": str(value.st_mode),
        "uid": str(value.st_uid),
        "gid": str(value.st_gid),
    }


def _requested_executable_binding(path: Path) -> dict[str, Any]:
    try:
        value = path.lstat()
    except OSError as exc:
        _fail("GATE_EXECUTABLE_UNAVAILABLE", "requested executable is unavailable", path=str(path), errno=exc.errno)
    result: dict[str, Any] = {
        "path": str(path),
        "device": str(value.st_dev),
        "inode": str(value.st_ino),
        "mode": str(value.st_mode),
        "link_count": str(value.st_nlink),
        "size_bytes": str(value.st_size),
        "modified_ns": str(value.st_mtime_ns),
        "changed_ns": str(value.st_ctime_ns),
    }
    if stat.S_ISLNK(value.st_mode):
        try:
            result["symlink_target"] = os.readlink(path)
        except OSError as exc:
            _fail("GATE_EXECUTABLE_UNAVAILABLE", "requested executable symlink is unreadable", path=str(path), errno=exc.errno)
    else:
        result["symlink_target"] = None
    return result


def _resolved_executable(path: str) -> tuple[Path, dict[str, Any]]:
    requested = Path(path)
    if not requested.is_absolute():
        _fail("GATE_EXECUTABLE_NOT_ABSOLUTE", "gate executable must be an absolute path", path=path)
    lexical = Path(os.path.abspath(os.fspath(requested)))
    binding = _requested_executable_binding(lexical)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        _fail("GATE_EXECUTABLE_UNAVAILABLE", "gate executable cannot be resolved", path=path, errno=exc.errno)
    if lexical != resolved or binding["symlink_target"] is not None:
        _fail(
            "GATE_EXECUTABLE_NOT_CANONICAL",
            "gate executable path must be canonical and non-symlinked",
            path=path,
        )
    if not resolved.is_file():
        _fail("GATE_EXECUTABLE_INVALID", "gate executable is not a regular file", path=path)
    return resolved, binding


def _git_environment() -> dict[str, str]:
    result = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.untrackedCache",
        "GIT_CONFIG_VALUE_1": "false",
    }
    return result


def _git_process_leader_is_still_child(process: subprocess.Popen) -> bool:
    while True:
        try:
            observation = os.waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOWAIT | os.WNOHANG,
            )
            break
        except InterruptedError:
            continue
        except (ChildProcessError, OSError) as exc:
            if isinstance(exc, ChildProcessError) or getattr(
                exc, "errno", None
            ) == errno.ECHILD:
                process._gate_numeric_signal_disarmed = True
                process._gate_leader_reaped = True
                return False
            _fail(
                "GATE_GIT_CLEANUP_FAILED",
                "Git process leader ownership could not be inspected",
                error=type(exc).__name__,
            )
    if observation is not None and observation.si_pid != process.pid:
        _fail(
            "GATE_GIT_CLEANUP_FAILED",
            "Git process leader ownership probe observed a different child",
        )
    return True


def _terminate_git_process_group(process: subprocess.Popen) -> None:
    if getattr(process, "returncode", None) is not None:
        process._gate_numeric_signal_disarmed = True
        process._gate_leader_reaped = True
        return
    if not getattr(process, "_gate_numeric_signal_disarmed", False):
        if not _git_process_leader_is_still_child(process):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            _fail(
                "GATE_GIT_CLEANUP_FAILED",
                "Git process group could not be terminated",
                errno=exc.errno,
            )
        process._gate_numeric_signal_disarmed = True
    if not getattr(process, "_gate_leader_reaped", False):
        _reap_git_process_leader(process)


def _git_terminal_observed(process: subprocess.Popen) -> bool:
    try:
        observation = os.waitid(
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOWAIT | os.WNOHANG,
        )
    except InterruptedError:
        return False
    except (ChildProcessError, OSError) as exc:
        _fail(
            "GATE_GIT_CLEANUP_FAILED",
            "Git process leader could not be observed before reaping",
            error=type(exc).__name__,
        )
    return observation is not None and observation.si_pid == process.pid


def _kill_residual_git_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        _fail(
            "GATE_GIT_CLEANUP_FAILED",
            "residual Git process group could not be terminated",
            errno=exc.errno,
        )
    process._gate_numeric_signal_disarmed = True


def _reap_git_process_leader(process: subprocess.Popen) -> None:
    if not getattr(process, "_gate_numeric_signal_disarmed", False):
        _fail(
            "GATE_GIT_CLEANUP_FAILED",
            "Git process leader cannot be reaped before numeric signaling is disarmed",
        )
    prior_reap_started = getattr(process, "_gate_exact_reap_started", False)
    process._gate_exact_reap_started = True
    while True:
        try:
            waited_pid, wait_status = os.waitpid(process.pid, 0)
            process._gate_leader_reaped = True
            break
        except InterruptedError:
            continue
        except (ChildProcessError, OSError) as exc:
            if prior_reap_started and (
                isinstance(exc, ChildProcessError)
                or getattr(exc, "errno", None) == errno.ECHILD
            ):
                process._gate_leader_reaped = True
                return
            _fail(
                "GATE_GIT_CLEANUP_FAILED",
                "Git process leader could not be reaped exactly",
                error=type(exc).__name__,
            )
    if waited_pid != process.pid:
        _fail(
            "GATE_GIT_CLEANUP_FAILED",
            "Git wait returned a different process",
        )
    process.returncode = os.waitstatus_to_exitcode(wait_status)


def _bounded_git_communicate(
    process: subprocess.Popen,
    *,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    try:
        if stdout_pipe is None or stderr_pipe is None:
            _fail("GATE_GIT_UNAVAILABLE", "Git output pipes are unavailable")
        streams = {
            stdout_pipe.fileno(): (
                "stdout",
                _MAX_GIT_PROCESS_STDOUT_BYTES,
                bytearray(),
            ),
            stderr_pipe.fileno(): (
                "stderr",
                _MAX_GIT_PROCESS_STDERR_BYTES,
                bytearray(),
            ),
        }
        poller = select.poll()
        event_mask = select.POLLIN | select.POLLHUP | select.POLLERR
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            poller.register(descriptor, event_mask)
        deadline_ns = time.monotonic_ns() + int(
            timeout_seconds * 1_000_000_000
        )
        active = set(streams)
        terminal_observed = False
        residual_group_killed = False
        poll_quantum_ms = max(1, math.ceil(_POLL_SECONDS * 1000))
        while active or not terminal_observed:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                if terminal_observed:
                    _fail(
                        "GATE_GIT_CLEANUP_FAILED",
                        "Git output pipes remained open after residual group cleanup",
                    )
                _fail("GATE_GIT_UNAVAILABLE", "Git identity probe timed out")
            if not terminal_observed and _git_terminal_observed(process):
                terminal_observed = True
                _kill_residual_git_process_group(process)
                residual_group_killed = True
            if not active:
                time.sleep(
                    min(
                        _POLL_SECONDS,
                        remaining_ns / 1_000_000_000,
                    )
                )
                continue
            timeout_ms = min(
                poll_quantum_ms,
                max(1, math.ceil(remaining_ns / 1_000_000)),
            )
            try:
                events = poller.poll(timeout_ms)
            except InterruptedError:
                continue
            if not events:
                continue
            for descriptor, event in events:
                if descriptor not in active:
                    continue
                if event & select.POLLNVAL:
                    _fail(
                        "GATE_GIT_UNAVAILABLE",
                        "Git output descriptor became invalid",
                    )
                try:
                    payload = os.read(descriptor, _PIPE_READ_QUANTUM)
                except BlockingIOError:
                    continue
                except InterruptedError:
                    continue
                if not payload:
                    poller.unregister(descriptor)
                    active.remove(descriptor)
                    continue
                role, limit, buffer = streams[descriptor]
                if len(buffer) + len(payload) > limit:
                    _fail(
                        "GATE_GIT_OUTPUT_OVERSIZED",
                        f"Git {role} exceeds its process output bound",
                    )
                buffer.extend(payload)
        if not residual_group_killed:
            _fail(
                "GATE_GIT_CLEANUP_FAILED",
                "Git leader was not retained through process-group cleanup",
            )
        stdout = bytes(streams[stdout_pipe.fileno()][2])
        stderr = bytes(streams[stderr_pipe.fileno()][2])
        _reap_git_process_leader(process)
        return stdout, stderr
    except BaseException as active_error:
        try:
            _terminate_git_process_group(process)
        except BaseException as cleanup_error:
            cleanup_error.add_note(
                f"Git transport failure before cleanup: {active_error}"
            )
            raise
        raise
    finally:
        if stdout_pipe is not None:
            stdout_pipe.close()
        if stderr_pipe is not None:
            stderr_pipe.close()


def _run_git(
    cwd: Path,
    arguments: Sequence[str],
    *,
    allowed_returncodes: set[int] = {0},
    cwd_descriptor: int | None = None,
) -> tuple[int, bytes]:
    git = shutil.which("git", path="/usr/bin:/bin")
    if git is None:
        _fail("GATE_GIT_UNAVAILABLE", "Git is unavailable at a fixed system path")
    git_path = Path(git).resolve(strict=True)
    runtime_cwd = cwd if cwd_descriptor is None else Path(f"/proc/self/fd/{cwd_descriptor}")
    try:
        with open_retained_executable(git_path) as retained:
            pass_fds = (
                (retained.descriptor,)
                if cwd_descriptor is None
                else (retained.descriptor, cwd_descriptor)
            )
            process = subprocess.Popen.__new__(subprocess.Popen)
            try:
                subprocess.Popen.__init__(
                    process,
                    [git, *arguments],
                    executable=retained.proc_path,
                    cwd=runtime_cwd,
                    env=_git_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=pass_fds,
                    start_new_session=True,
                )
            except BaseException:
                try:
                    pid = getattr(process, "pid", None)
                    if type(pid) is int and pid > 0:
                        _terminate_git_process_group(process)
                finally:
                    for name in ("stdout", "stderr"):
                        stream = getattr(process, name, None)
                        if stream is not None:
                            stream.close()
                raise
            stdout, stderr = _bounded_git_communicate(
                process,
                timeout_seconds=_MAX_GIT_PROCESS_SECONDS,
            )
            return_code = process.returncode
    except (ImmutableArtifactError, OSError, subprocess.SubprocessError) as exc:
        _fail("GATE_GIT_UNAVAILABLE", "Git identity probe failed", error=type(exc).__name__)
    if return_code not in allowed_returncodes:
        _fail(
            "GATE_GIT_PROBE_FAILED",
            "Git identity probe returned an unexpected status",
            arguments=list(arguments),
            return_code=return_code,
            stderr=stderr[:4096].decode("utf-8", errors="replace"),
        )
    return return_code, stdout


def _decode_line(payload: bytes, *, role: str, allow_empty: bool = False) -> str:
    if len(payload) > _MAX_STATUS_BYTES:
        _fail("GATE_GIT_OUTPUT_OVERSIZED", f"{role} exceeds its byte limit")
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError:
        _fail("GATE_GIT_OUTPUT_INVALID", f"{role} is not UTF-8")
    value = value.rstrip("\n")
    if not allow_empty and not value:
        _fail("GATE_GIT_OUTPUT_INVALID", f"{role} is empty")
    return value


def _valid_git_branch_text(value: str | None) -> bool:
    return value is None or (
        type(value) is str
        and bool(value)
        and len(value.encode("utf-8")) <= _MAX_GIT_BRANCH_BYTES
        and not any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        )
    )


def _git_snapshot(
    cwd: Path, *, cwd_descriptor: int | None = None,
) -> dict[str, Any]:
    def run(
        arguments: Sequence[str], *, allowed_returncodes: set[int] = {0},
    ) -> tuple[int, bytes]:
        return _run_git(
            cwd,
            arguments,
            allowed_returncodes=allowed_returncodes,
            cwd_descriptor=cwd_descriptor,
        )

    _, root_bytes = run(("rev-parse", "--show-toplevel"))
    root = Path(_decode_line(root_bytes, role="Git repository root"))
    root = _canonical_path(root, role="Git repository root", directory=True)
    _, commit_bytes = run(("rev-parse", "--verify", "HEAD^{commit}"))
    _, tree_bytes = run(("rev-parse", "--verify", "HEAD^{tree}"))
    branch_code, branch_bytes = run(
        ("symbolic-ref", "--quiet", "HEAD"),
        allowed_returncodes={0, 1},
    )
    _, status_bytes = run(
        ("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"),
    )
    if len(status_bytes) > _MAX_STATUS_BYTES:
        _fail("GATE_GIT_OUTPUT_OVERSIZED", "Git status exceeds its byte limit")
    try:
        status_text = status_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _fail("GATE_GIT_OUTPUT_INVALID", "Git status is not UTF-8")
    commit = _decode_line(commit_bytes, role="Git commit")
    tree = _decode_line(tree_bytes, role="Git tree")
    if (
        len(commit) not in {40, 64}
        or len(tree) != len(commit)
        or not re.fullmatch(r"[0-9a-f]+", commit)
        or not re.fullmatch(r"[0-9a-f]+", tree)
    ):
        _fail("GATE_GIT_OUTPUT_INVALID", "Git commit or tree identity is malformed")
    if branch_code == 1:
        branch = None
    else:
        branch_ref = _decode_line(branch_bytes, role="Git branch ref")
        if not branch_ref.startswith("refs/heads/"):
            _fail(
                "GATE_GIT_OUTPUT_INVALID",
                "Git symbolic HEAD is outside the branch namespace",
            )
        branch = branch_ref.removeprefix("refs/heads/")
    if not _valid_git_branch_text(branch):
        _fail(
            "GATE_GIT_OUTPUT_INVALID",
            "Git branch exceeds its admitted text contract",
        )
    git_path = Path(shutil.which("git", path="/usr/bin:/bin") or "").resolve(strict=True)
    return {
        "repository_root": str(root),
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "status_porcelain_v1": status_text,
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "clean": status_text == "",
        "git_executable_path": str(git_path),
        "git_executable_sha256": sha256_file(git_path)[0],
    }


def _tracked_input_manifest(
    cwd: Path, *, cwd_descriptor: int,
) -> _TrackedInputManifest:
    sparse_code, sparse_bytes = _run_git(
        cwd,
        ("config", "--bool", "core.sparseCheckout"),
        allowed_returncodes={0, 1},
        cwd_descriptor=cwd_descriptor,
    )
    if sparse_code == 0 and _decode_line(
        sparse_bytes, role="Git sparse-checkout setting"
    ).lower() == "true":
        _fail("GATE_SOURCE_LAYOUT_UNSUPPORTED", "sparse Git checkouts are not admitted")
    _, shared_index_bytes = _run_git(
        cwd,
        ("rev-parse", "--shared-index-path"),
        cwd_descriptor=cwd_descriptor,
    )
    if _decode_line(
        shared_index_bytes,
        role="Git shared-index path",
        allow_empty=True,
    ):
        _fail("GATE_SOURCE_LAYOUT_UNSUPPORTED", "split Git indexes are not admitted")
    _, payload = _run_git(
        cwd,
        ("ls-files", "--cached", "--stage", "--full-name", "-z"),
        cwd_descriptor=cwd_descriptor,
    )
    if len(payload) > _MAX_TRACKED_MANIFEST_BYTES:
        _fail("GATE_TRACKED_INPUT_BOUND_EXCEEDED", "tracked input manifest is oversized")
    entries = payload.split(b"\0")
    if not entries or entries[-1] != b"":
        _fail("GATE_TRACKED_INPUT_INVALID", "tracked input manifest is not NUL terminated")
    entries.pop()
    if not entries or len(entries) > _MAX_TRACKED_ENTRIES:
        _fail("GATE_TRACKED_INPUT_BOUND_EXCEEDED", "tracked input count is outside its bound")
    paths: list[str] = []
    path_modes: list[tuple[str, str]] = []
    mode_counts = {"100644": 0, "100755": 0}
    relative_path_bytes = 0
    maximum_relative_path_bytes = 0
    for entry in entries:
        metadata, separator, encoded_path = entry.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            _fail("GATE_TRACKED_INPUT_INVALID", "tracked input index row is malformed")
        mode_bytes, object_id, stage_bytes = fields
        try:
            mode = mode_bytes.decode("ascii")
            stage = stage_bytes.decode("ascii")
            path_text = encoded_path.decode("utf-8")
        except UnicodeDecodeError:
            _fail("GATE_TRACKED_INPUT_INVALID", "tracked input index row is not canonical text")
        if (
            mode not in mode_counts
            or stage != "0"
            or not re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", object_id)
        ):
            _fail(
                "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                "tracked input is not a stage-0 regular Git entry",
            )
        try:
            relative = safe_relative_path(path_text)
        except ImmutableArtifactError as exc:
            _fail("GATE_TRACKED_INPUT_INVALID", "tracked input path is unsafe", error=str(exc))
        canonical_path = relative.as_posix()
        if canonical_path != path_text:
            _fail("GATE_TRACKED_INPUT_INVALID", "tracked input path is not canonical")
        path_size = len(encoded_path)
        if path_size > _MAX_TRACKED_PATH_BYTES:
            _fail(
                "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
                "tracked input path exceeds its per-path byte bound",
            )
        relative_path_bytes += path_size
        maximum_relative_path_bytes = max(
            maximum_relative_path_bytes,
            path_size,
        )
        if relative_path_bytes > _MAX_TRACKED_RELATIVE_PATH_BYTES:
            _fail("GATE_TRACKED_INPUT_BOUND_EXCEEDED", "tracked input paths exceed their byte bound")
        paths.append(canonical_path)
        path_modes.append((canonical_path, mode))
        mode_counts[mode] += 1
    if len(set(paths)) != len(paths):
        _fail("GATE_TRACKED_INPUT_INVALID", "tracked input paths are not unique")
    _, flags_payload = _run_git(
        cwd,
        ("ls-files", "--cached", "-v", "--full-name", "-z"),
        cwd_descriptor=cwd_descriptor,
    )
    if len(flags_payload) > _MAX_TRACKED_MANIFEST_BYTES:
        _fail(
            "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
            "tracked input flag inventory is oversized",
        )
    if not flags_payload.endswith(b"\0"):
        _fail(
            "GATE_TRACKED_INPUT_INVALID",
            "tracked input flag inventory is not NUL terminated",
        )
    if flags_payload.count(b"\0") != len(paths):
        _fail(
            "GATE_TRACKED_INPUT_INVALID",
            "tracked input flag inventory disagrees with the index",
        )
    flag_rows = flags_payload[:-1].split(b"\0")
    flagged_paths: list[str] = []
    for row in flag_rows:
        if not row.startswith(b"H "):
            _fail(
                "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                "tracked input has assume-unchanged or skip-worktree state",
            )
        try:
            flagged_path = row[2:].decode("utf-8")
        except UnicodeDecodeError:
            _fail(
                "GATE_TRACKED_INPUT_INVALID",
                "tracked input flag path is not UTF-8",
            )
        try:
            canonical_flagged = safe_relative_path(flagged_path).as_posix()
        except ImmutableArtifactError as exc:
            _fail(
                "GATE_TRACKED_INPUT_INVALID",
                "tracked input flag path is unsafe",
                error=str(exc),
            )
        if canonical_flagged != flagged_path:
            _fail(
                "GATE_TRACKED_INPUT_INVALID",
                "tracked input flag path is not canonical",
            )
        flagged_paths.append(flagged_path)
    sorted_paths = tuple(sorted(paths))
    if tuple(sorted(flagged_paths)) != sorted_paths:
        _fail(
            "GATE_TRACKED_INPUT_INVALID",
            "tracked input flag inventory disagrees with the index",
        )
    return _TrackedInputManifest(
        paths=sorted_paths,
        mode_by_path=tuple(sorted(path_modes)),
        document={
            "format": "git_ls_files_stage0_regular_v1",
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "flag_format": "git_ls_files_v_normal_only_v1",
            "flag_manifest_sha256": hashlib.sha256(flags_payload).hexdigest(),
            "flag_manifest_bytes": len(flags_payload),
            "entry_count": len(paths),
            "relative_path_bytes": relative_path_bytes,
            "maximum_relative_path_bytes": maximum_relative_path_bytes,
            "mode_counts": mode_counts,
            "monitor": "pinned_path_set_inotify_v1",
            "execution_scope": (
                "real_checkout_tracked_files_monitored_untracked_and_ignored_siblings_mutable_v1"
            ),
            "limits": {
                "max_entries": _MAX_TRACKED_ENTRIES,
                "max_manifest_bytes": _MAX_TRACKED_MANIFEST_BYTES,
                "max_relative_path_bytes": _MAX_TRACKED_RELATIVE_PATH_BYTES,
                "max_path_bytes": _MAX_TRACKED_PATH_BYTES,
                "max_index_bytes": _MAX_GIT_INDEX_BYTES,
            },
        },
    )


def _tracked_worktree_manifest(
    reader: PinnedDirectoryReader,
    tracked: _TrackedInputManifest,
) -> dict[str, Any]:
    if len(tracked.paths) != len(tracked.mode_by_path):
        _fail(
            "GATE_TRACKED_INPUT_INVALID",
            "tracked path and mode inventories disagree",
        )
    expected_paths = tuple(path for path, _mode in tracked.mode_by_path)
    if expected_paths != tracked.paths:
        _fail(
            "GATE_TRACKED_INPUT_INVALID",
            "tracked path and mode inventories are not canonically ordered",
        )

    directory_path_set = {"."}
    for relative_path in tracked.paths:
        parts = safe_relative_path(relative_path).parts
        for depth in range(1, len(parts)):
            directory_path_set.add("/".join(parts[:depth]))
    directory_paths = tuple(sorted(directory_path_set))
    directory_relative_path_bytes = sum(
        len(path.encode("utf-8")) for path in directory_paths
    )
    maximum_directory_path_bytes = max(
        len(path.encode("utf-8")) for path in directory_paths
    )
    if (
        len(directory_paths) > _MAX_TRACKED_DIRECTORIES
        or directory_relative_path_bytes
        > _MAX_TRACKED_DIRECTORY_PATH_BYTES
        or maximum_directory_path_bytes > _MAX_TRACKED_PATH_BYTES
    ):
        _fail(
            "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
            "checked-out tracked directories exceed their path bounds",
        )

    manifest = hashlib.sha256()
    manifest_magic = (
        b"gate_checked_out_worktree_files_and_directories_sha256_v1\0"
    )
    manifest.update(manifest_magic)
    manifest.update(len(tracked.paths).to_bytes(8, "big"))
    manifest.update(len(directory_paths).to_bytes(8, "big"))
    manifest_bytes = len(manifest_magic) + 16
    for relative_path in directory_paths:
        metadata = reader.directory_metadata_snapshot(relative_path)
        actual_mode = metadata[2]
        if not stat.S_ISDIR(actual_mode):
            _fail(
                "GATE_TRACKED_INPUT_INVALID",
                "checked-out tracked ancestor is not a stable directory",
                path=relative_path,
            )
        encoded_path = relative_path.encode("utf-8")
        manifest_bytes += 8 + len(encoded_path) + 8
        if manifest_bytes > _MAX_TRACKED_WORKTREE_MANIFEST_BYTES:
            _fail(
                "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
                "checked-out worktree manifest exceeds its byte bound",
            )
        manifest.update(len(encoded_path).to_bytes(8, "big"))
        manifest.update(encoded_path)
        manifest.update(actual_mode.to_bytes(8, "big"))

    total_file_bytes = 0
    maximum_file_bytes = 0
    executable_entries = 0
    for relative_path, git_mode in tracked.mode_by_path:
        try:
            snapshot = reader.read_snapshot(
                relative_path,
                max_bytes=_MAX_TRACKED_FILE_BYTES,
            )
        except ImmutableArtifactLimitError:
            _fail(
                "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
                "checked-out tracked file exceeds its byte bound",
                path=relative_path,
            )
        metadata = snapshot.metadata
        actual_mode = metadata[2]
        if not stat.S_ISREG(actual_mode) or metadata[4] != len(snapshot.payload):
            _fail(
                "GATE_TRACKED_INPUT_INVALID",
                "checked-out tracked input is not a stable regular file",
                path=relative_path,
            )
        expected_executable = git_mode == "100755"
        actual_executable = bool(actual_mode & stat.S_IXUSR)
        if actual_executable != expected_executable:
            _fail(
                "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                "checked-out executable mode disagrees with the Git index",
                path=relative_path,
            )
        executable_entries += int(actual_executable)

        payload_size = len(snapshot.payload)
        total_file_bytes += payload_size
        maximum_file_bytes = max(maximum_file_bytes, payload_size)
        if total_file_bytes > _MAX_TRACKED_TOTAL_FILE_BYTES:
            _fail(
                "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
                "checked-out tracked files exceed their aggregate byte bound",
            )
        encoded_path = relative_path.encode("utf-8")
        encoded_entry_bytes = 8 + len(encoded_path) + 6 + 8 + 8 + 32
        manifest_bytes += encoded_entry_bytes
        if manifest_bytes > _MAX_TRACKED_WORKTREE_MANIFEST_BYTES:
            _fail(
                "GATE_TRACKED_INPUT_BOUND_EXCEEDED",
                "checked-out worktree manifest exceeds its byte bound",
            )
        manifest.update(len(encoded_path).to_bytes(8, "big"))
        manifest.update(encoded_path)
        manifest.update(git_mode.encode("ascii"))
        manifest.update(actual_mode.to_bytes(8, "big"))
        manifest.update(payload_size.to_bytes(8, "big"))
        manifest.update(hashlib.sha256(snapshot.payload).digest())

    return {
        "format": (
            "descriptor_relative_checked_out_regular_and_directory_sha256_v1"
        ),
        "manifest_sha256": manifest.hexdigest(),
        "manifest_bytes": manifest_bytes,
        "entry_count": len(tracked.paths),
        "relative_path_bytes": tracked.document["relative_path_bytes"],
        "directory_entry_count": len(directory_paths),
        "directory_relative_path_bytes": directory_relative_path_bytes,
        "maximum_directory_path_bytes": maximum_directory_path_bytes,
        "total_file_bytes": total_file_bytes,
        "maximum_file_bytes": maximum_file_bytes,
        "executable_entry_count": executable_entries,
        "git_mode_contract": (
            "actual_owner_exec_bit_matches_stage0_git_mode_v1"
        ),
        "directory_mode_contract": (
            "tracked_ancestor_and_repository_root_modes_bound_v1"
        ),
        "limits": {
            "max_file_bytes": _MAX_TRACKED_FILE_BYTES,
            "max_total_file_bytes": _MAX_TRACKED_TOTAL_FILE_BYTES,
            "max_manifest_bytes": _MAX_TRACKED_WORKTREE_MANIFEST_BYTES,
            "max_directories": _MAX_TRACKED_DIRECTORIES,
            "max_directory_relative_path_bytes": (
                _MAX_TRACKED_DIRECTORY_PATH_BYTES
            ),
            "max_path_bytes": _MAX_TRACKED_PATH_BYTES,
        },
    }


def _git_index_path(cwd: Path, *, cwd_descriptor: int) -> Path:
    _, payload = _run_git(
        cwd,
        ("rev-parse", "--path-format=absolute", "--git-path", "index"),
        cwd_descriptor=cwd_descriptor,
    )
    return _canonical_path(
        Path(_decode_line(payload, role="Git index path")),
        role="Git index",
        directory=False,
    )


def _regular_snapshot_binding(
    *, path: Path, snapshot: Any,
) -> dict[str, str]:
    metadata = snapshot.metadata
    return {
        "path": str(path),
        "sha256": hashlib.sha256(snapshot.payload).hexdigest(),
        "device": str(metadata[0]),
        "inode": str(metadata[1]),
        "mode": str(metadata[2]),
        "link_count": str(metadata[3]),
        "size_bytes": str(metadata[4]),
        "modified_ns": str(metadata[5]),
        "changed_ns": str(metadata[6]),
    }


def _relative_admin_path(root: Path, relative_path: str) -> Path:
    relative = safe_relative_path(relative_path)
    return root.joinpath(*relative.parts)


def _required_admin_regular(
    reader: PinnedDirectoryReader,
    root: Path,
    relative_path: str,
) -> tuple[dict[str, str], bytes]:
    path = _relative_admin_path(root, relative_path)
    snapshot = reader.read_snapshot(
        relative_path,
        max_bytes=_MAX_GIT_ADMIN_FILE_BYTES,
    )
    return _regular_snapshot_binding(path=path, snapshot=snapshot), snapshot.payload


def _optional_admin_regular(
    reader: PinnedDirectoryReader,
    root: Path,
    relative_path: str,
) -> tuple[dict[str, Any], bytes | None]:
    path = _relative_admin_path(root, relative_path)
    if not os.path.lexists(path):
        return {"state": "absent", "path": str(path)}, None
    binding, payload = _required_admin_regular(
        reader,
        root,
        relative_path,
    )
    return {"state": "present", "binding": binding}, payload


def _single_git_admin_line(payload: bytes, *, role: str) -> str:
    if (
        not payload
        or len(payload) > _MAX_GIT_ADMIN_FILE_BYTES
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
        or b"\x00" in payload
    ):
        _fail(
            "GATE_GIT_ADMIN_INVALID",
            f"{role} is not one bounded NUL-free line",
        )
    try:
        value = payload[:-1].decode("utf-8")
    except UnicodeDecodeError:
        _fail("GATE_GIT_ADMIN_INVALID", f"{role} is not UTF-8")
    if not value or any(ord(character) < 32 for character in value):
        _fail("GATE_GIT_ADMIN_INVALID", f"{role} contains invalid text")
    return value


def _git_pointer_target(
    payload: bytes,
    *,
    role: str,
    base: Path,
    prefix: str = "",
    directory: bool,
) -> Path:
    value = _single_git_admin_line(payload, role=role)
    if prefix:
        if not value.startswith(prefix):
            _fail("GATE_GIT_ADMIN_INVALID", f"{role} has an invalid prefix")
        value = value[len(prefix):]
    if not value:
        _fail("GATE_GIT_ADMIN_INVALID", f"{role} has an empty target")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return _canonical_path(candidate, role=role, directory=directory)


def _valid_git_refname(value: str) -> bool:
    if (
        type(value) is not str
        or not value.startswith("refs/")
        or len(value.encode("utf-8")) > _MAX_GIT_REF_BYTES
        or value.endswith(("/", "."))
        or "//" in value
        or ".." in value
        or "@{" in value
        or any(
            ord(character) < 32
            or ord(character) == 127
            or character in " ~^:?*[\\"
            for character in value
        )
    ):
        return False
    components = value.split("/")
    return all(
        component
        and not component.startswith(".")
        and not component.endswith(".lock")
        for component in components
    )


def _parse_head(payload: bytes) -> tuple[str, str | None, str | None]:
    value = _single_git_admin_line(payload, role="Git HEAD")
    if value.startswith("ref: "):
        ref = value[5:]
        try:
            canonical_ref = safe_relative_path(ref).as_posix()
        except ImmutableArtifactError as exc:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "Git HEAD symbolic ref is unsafe",
                error=str(exc),
            )
        if (
            canonical_ref != ref
            or not ref.startswith("refs/heads/")
            or not _valid_git_refname(ref)
        ):
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "Git HEAD symbolic ref is outside the admitted branch namespace",
            )
        return "symbolic", ref, None
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
        _fail("GATE_GIT_ADMIN_INVALID", "detached Git HEAD OID is malformed")
    return "detached", None, value


def _parse_loose_ref(payload: bytes, *, role: str) -> str:
    value = _single_git_admin_line(payload, role=role)
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
        _fail("GATE_GIT_ADMIN_INVALID", f"{role} OID is malformed")
    return value


def _packed_ref_oid(payload: bytes, ref_name: str) -> str:
    if len(payload) > _MAX_GIT_ADMIN_FILE_BYTES:
        _fail("GATE_GIT_ADMIN_INVALID", "packed-refs exceeds its byte bound")
    matches: list[str] = []
    for raw_line in payload.splitlines():
        if not raw_line or raw_line.startswith((b"#", b"^")):
            continue
        oid_bytes, separator, ref_bytes = raw_line.partition(b" ")
        if not separator:
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs row is malformed")
        try:
            observed_ref = ref_bytes.decode("utf-8")
            oid = oid_bytes.decode("ascii")
        except UnicodeDecodeError:
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs row is not text")
        if observed_ref == ref_name:
            if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid):
                _fail("GATE_GIT_ADMIN_INVALID", "packed active-ref OID is malformed")
            matches.append(oid)
    if len(matches) != 1:
        _fail(
            "GATE_GIT_ADMIN_INVALID",
            "packed-refs does not contain exactly one active ref",
            ref=ref_name,
            matches=len(matches),
        )
    return matches[0]


def _validate_packed_refs_rows(
    payload: bytes,
    *,
    expected_oid_width: int,
) -> None:
    if expected_oid_width not in {40, 64}:
        _fail(
            "GATE_GIT_ADMIN_INVALID",
            "packed-refs object-ID width is unsupported",
        )
    if (
        not payload
        or len(payload) > _MAX_GIT_PACKED_REFS_RECEIPT_BYTES
        or not payload.endswith(b"\n")
        or b"\0" in payload
    ):
        _fail(
            "GATE_GIT_ADMIN_INVALID",
            "packed-refs violates its retained byte and line contract",
        )
    observed_refs: set[str] = set()
    records_started = False
    header_seen = False
    peeled_advertised = False
    sorted_declared = False
    previous_ref_bytes: bytes | None = None
    peeled_row_allowed = False
    for raw_line in payload[:-1].split(b"\n"):
        if not raw_line:
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs contains an empty row")
        if raw_line.startswith(b"#"):
            header_prefix = b"# pack-refs with: "
            if (
                records_started
                or header_seen
                or not raw_line.startswith(header_prefix)
                or not raw_line.endswith(b" ")
            ):
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    "packed-refs header row is malformed",
                )
            capabilities = raw_line[len(header_prefix):-1].split(b" ")
            admitted_capabilities = (b"peeled", b"fully-peeled", b"sorted")
            if (
                not capabilities
                or b"" in capabilities
                or len(set(capabilities)) != len(capabilities)
                or any(
                    capability not in admitted_capabilities
                    for capability in capabilities
                )
                or tuple(
                    sorted(
                        capabilities,
                        key=admitted_capabilities.index,
                    )
                )
                != tuple(capabilities)
                or (
                    b"fully-peeled" in capabilities
                    and b"peeled" not in capabilities
                )
            ):
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    "packed-refs capabilities are unsupported",
                )
            header_seen = True
            peeled_advertised = b"peeled" in capabilities
            sorted_declared = b"sorted" in capabilities
            peeled_row_allowed = False
            continue
        if raw_line.startswith(b"^"):
            peeled_oid = raw_line[1:]
            if (
                not peeled_row_allowed
                or (header_seen and not peeled_advertised)
                or len(peeled_oid) != expected_oid_width
                or not re.fullmatch(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})", peeled_oid)
            ):
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    "packed-refs peeled row is malformed or orphaned",
                )
            peeled_row_allowed = False
            continue
        records_started = True
        oid_bytes, separator, ref_bytes = raw_line.partition(b" ")
        if not separator:
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs row is malformed")
        try:
            oid = oid_bytes.decode("ascii")
            ref = ref_bytes.decode("utf-8")
        except UnicodeDecodeError:
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs row is not text")
        try:
            canonical_ref = safe_relative_path(ref).as_posix()
        except ImmutableArtifactError:
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs row is unsafe")
        if (
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid)
            or len(oid_bytes) != expected_oid_width
            or canonical_ref != ref
            or not _valid_git_refname(ref)
            or ref in observed_refs
            or (
                sorted_declared
                and previous_ref_bytes is not None
                and ref_bytes <= previous_ref_bytes
            )
        ):
            _fail("GATE_GIT_ADMIN_INVALID", "packed-refs row is unsafe")
        observed_refs.add(ref)
        previous_ref_bytes = ref_bytes
        peeled_row_allowed = True


def _strict_git_config_scope(
    payload: bytes,
    *,
    role: str,
) -> None:
    if len(payload) > _MAX_GIT_CONFIG_RECEIPT_BYTES or b"\0" in payload:
        _fail(
            "GATE_GIT_ADMIN_INVALID",
            f"{role} Git config exceeds its retained text contract",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        _fail("GATE_GIT_ADMIN_INVALID", f"{role} Git config is not UTF-8")
    section: str | None = None
    names: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            match = re.fullmatch(
                r'\[([A-Za-z][A-Za-z0-9-]*)(?: "([^"\\\x00-\x1f]*)")?\]'
                r'(?:\s*[#;].*)?',
                line,
            )
            if match is None:
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    f"{role} Git config section syntax is unsupported",
                    line=line_number,
                )
            section = match.group(1).lower()
            continue
        if section is None:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                f"{role} Git config key appears before a section",
                line=line_number,
            )
        trimmed = raw_line.rstrip()
        if (len(trimmed) - len(trimmed.rstrip("\\"))) % 2 == 1:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                f"{role} Git config continuations are unsupported",
                line=line_number,
            )
        match = re.fullmatch(
            r"\s*([A-Za-z][A-Za-z0-9-]*)\s*(?:=.*)?",
            raw_line,
        )
        if match is None:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                f"{role} Git config key syntax is unsupported",
                line=line_number,
            )
        variable = match.group(1).lower()
        name = f"{section}.{variable}"
        lowered = name.lower()
        if (
            lowered.startswith("include.")
            or lowered.startswith("includeif.")
            or lowered in {"core.excludesfile", "core.attributesfile"}
        ):
            _fail(
                "GATE_GIT_CONFIG_EXTERNAL_INPUT",
                f"{role} Git config references an unretained external input",
                line=line_number,
            )
        names.append(lowered)
        if len(names) > _MAX_GIT_CONFIG_KEYS:
            _fail("GATE_GIT_ADMIN_INVALID", f"{role} Git config has too many keys")


def _git_config_policy(
    *,
    common_payload: bytes,
    worktree_payload: bytes | None,
) -> dict[str, str]:
    _strict_git_config_scope(
        common_payload,
        role="common",
    )
    if worktree_payload is not None:
        _strict_git_config_scope(
            worktree_payload,
            role="worktree",
        )
    return {
        "probe_environment": (
            "system_global_and_system_attributes_disabled_v1"
        ),
        "producer_check": (
            "include_and_core_excludes_attributes_external_inputs_rejected_v1"
        ),
        "privacy": "opaque_bindings_no_parsed_names_or_values_v1",
    }


def _git_path_probe(
    cwd: Path,
    *,
    cwd_descriptor: int,
    relative_path: str,
) -> Path:
    _, payload = _run_git(
        cwd,
        (
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            relative_path,
        ),
        cwd_descriptor=cwd_descriptor,
    )
    value = _decode_line(payload, role=f"Git path {relative_path}")
    path = Path(value)
    if not path.is_absolute() or str(Path(os.path.abspath(value))) != value:
        _fail(
            "GATE_GIT_ADMIN_INVALID",
            f"Git path {relative_path} is not canonically spelled",
        )
    return path


@dataclasses.dataclass(slots=True)
class _GitAdministrationRetention:
    work: Path
    work_descriptor: int
    repository_root: Path
    repository_reader: PinnedDirectoryReader
    marker_guard: PinnedPathSetMonitor
    marker_file_monitor: PinnedPathSetMonitor | None
    layout: str
    marker_path: Path
    git_dir: Path
    git_reader: PinnedDirectoryReader
    git_monitor: PinnedPathSetMonitor
    common_dir: Path
    common_reader: PinnedDirectoryReader
    common_monitor: PinnedPathSetMonitor
    active_ref: str | None

    def verify(self) -> None:
        self.marker_guard.verify_unchanged()
        if self.marker_file_monitor is not None:
            self.marker_file_monitor.verify_unchanged()
        self.repository_reader.verify_path()
        self.git_monitor.verify_unchanged()
        self.git_reader.verify_path()
        self.common_monitor.verify_unchanged()
        self.common_reader.verify_path()
        self.marker_guard.verify_unchanged()

    def _crosscheck_discovery(self) -> None:
        _, payload = _run_git(
            self.work,
            (
                "rev-parse",
                "--path-format=absolute",
                "--git-dir",
                "--git-common-dir",
            ),
            cwd_descriptor=self.work_descriptor,
        )
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            _fail("GATE_GIT_ADMIN_INVALID", "Git directory discovery is not UTF-8")
        if lines != [str(self.git_dir), str(self.common_dir)]:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "Git directory discovery disagrees with retained pointers",
            )
        _, top_payload = _run_git(
            self.work,
            ("rev-parse", "--show-toplevel"),
            cwd_descriptor=self.work_descriptor,
        )
        if _decode_line(top_payload, role="Git repository root") != str(
            self.repository_root
        ):
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "Git repository root disagrees with its retained directory",
            )
        _, format_payload = _run_git(
            self.work,
            ("rev-parse", "--show-ref-format"),
            cwd_descriptor=self.work_descriptor,
        )
        if _decode_line(format_payload, role="Git ref format") != "files":
            _fail("GATE_SOURCE_LAYOUT_UNSUPPORTED", "Git reftable is not admitted")

        expected_paths = {
            "HEAD": self.git_dir / "HEAD",
            "index": self.git_dir / "index",
            "config": self.common_dir / "config",
            "packed-refs": self.common_dir / "packed-refs",
            "info/exclude": self.common_dir / "info" / "exclude",
            "info/attributes": self.common_dir / "info" / "attributes",
        }
        if self.active_ref is not None:
            expected_paths[self.active_ref] = self.common_dir.joinpath(
                *safe_relative_path(self.active_ref).parts
            )
        for relative_path, expected in expected_paths.items():
            observed = _git_path_probe(
                self.work,
                cwd_descriptor=self.work_descriptor,
                relative_path=relative_path,
            )
            if observed != expected:
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    "Git path discovery disagrees with retained administration",
                    relative_path=relative_path,
                )

    def document(self, *, commit: str) -> dict[str, Any]:
        self.verify()
        if self.layout == "ordinary_gitdir":
            marker = {
                "kind": "directory",
                "binding": _directory_binding(
                    self.marker_path,
                    descriptor=self.git_reader.descriptor,
                ),
                "target": None,
                "line": None,
            }
        else:
            marker_binding, marker_payload = _required_admin_regular(
                self.repository_reader,
                self.repository_root,
                ".git",
            )
            marker_target = _git_pointer_target(
                marker_payload,
                role="linked-worktree .git marker",
                base=self.repository_root,
                prefix="gitdir: ",
                directory=True,
            )
            if marker_target != self.git_dir:
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    "linked-worktree marker target changed",
                )
            marker = {
                "kind": "gitfile",
                "binding": marker_binding,
                "target": str(marker_target),
                "line": _single_git_admin_line(
                    marker_payload,
                    role="linked-worktree .git marker",
                ),
            }

        commondir, commondir_payload = _optional_admin_regular(
            self.git_reader,
            self.git_dir,
            "commondir",
        )
        gitdir_pointer, gitdir_payload = _optional_admin_regular(
            self.git_reader,
            self.git_dir,
            "gitdir",
        )
        if self.layout == "ordinary_gitdir":
            if commondir["state"] != "absent" or gitdir_pointer["state"] != "absent":
                _fail(
                    "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                    "ordinary Git directory contains linked-worktree pointers",
                )
            commondir_target = None
            gitdir_target = None
            commondir_line = None
            gitdir_line = None
        else:
            if commondir_payload is None or gitdir_payload is None:
                _fail(
                    "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                    "linked worktree is missing an administrative pointer",
                )
            observed_common = _git_pointer_target(
                commondir_payload,
                role="linked-worktree commondir pointer",
                base=self.git_dir,
                directory=True,
            )
            observed_gitfile = _git_pointer_target(
                gitdir_payload,
                role="linked-worktree gitdir pointer",
                base=self.git_dir,
                directory=False,
            )
            if observed_common != self.common_dir or observed_gitfile != self.marker_path:
                _fail(
                    "GATE_GIT_ADMIN_INVALID",
                    "linked-worktree pointer target changed",
                )
            commondir_target = str(observed_common)
            gitdir_target = str(observed_gitfile)
            commondir_line = _single_git_admin_line(
                commondir_payload,
                role="linked-worktree commondir pointer",
            )
            gitdir_line = _single_git_admin_line(
                gitdir_payload,
                role="linked-worktree gitdir pointer",
            )

        common_config, common_config_payload = _required_admin_regular(
            self.common_reader,
            self.common_dir,
            "config",
        )
        worktree_config, worktree_config_payload = _optional_admin_regular(
            self.git_reader,
            self.git_dir,
            "config.worktree",
        )
        head_binding, head_payload = _required_admin_regular(
            self.git_reader,
            self.git_dir,
            "HEAD",
        )
        head_kind, head_ref, detached_oid = _parse_head(head_payload)
        if head_ref != self.active_ref:
            _fail("GATE_GIT_ADMIN_INVALID", "Git HEAD ref changed during retention")

        loose_ref = {"state": "not_applicable", "path": None}
        packed_refs, packed_payload = _optional_admin_regular(
            self.common_reader,
            self.common_dir,
            "packed-refs",
        )
        if head_kind == "detached":
            active_storage = "detached"
            active_oid = detached_oid
        else:
            if head_ref is None:
                _fail("GATE_GIT_ADMIN_INVALID", "symbolic Git HEAD has no ref")
            loose_ref, loose_payload = _optional_admin_regular(
                self.common_reader,
                self.common_dir,
                head_ref,
            )
            if loose_payload is not None:
                active_storage = "loose"
                active_oid = _parse_loose_ref(
                    loose_payload,
                    role="Git active loose ref",
                )
            else:
                if packed_payload is None:
                    _fail(
                        "GATE_GIT_ADMIN_INVALID",
                        "active ref is absent from loose and packed storage",
                    )
                active_storage = "packed_refs"
                active_oid = _packed_ref_oid(packed_payload, head_ref)
        if active_oid != commit:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "retained Git active ref disagrees with the recorded commit",
            )
        if packed_payload is not None:
            _validate_packed_refs_rows(
                packed_payload,
                expected_oid_width=len(commit),
            )

        symbolic_code, symbolic_payload = _run_git(
            self.work,
            ("symbolic-ref", "--quiet", "HEAD"),
            allowed_returncodes={0, 1},
            cwd_descriptor=self.work_descriptor,
        )
        symbolic_ref = (
            None
            if symbolic_code == 1
            else _decode_line(symbolic_payload, role="Git symbolic HEAD")
        )
        if symbolic_ref != head_ref:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "Git symbolic-ref probe disagrees with retained HEAD bytes",
            )

        info_exclude, _ = _optional_admin_regular(
            self.common_reader,
            self.common_dir,
            "info/exclude",
        )
        info_attributes, _ = _optional_admin_regular(
            self.common_reader,
            self.common_dir,
            "info/attributes",
        )
        config_policy = _git_config_policy(
            common_payload=common_config_payload,
            worktree_payload=worktree_config_payload,
        )
        self._crosscheck_discovery()
        self.verify()
        return {
            "format": "descriptor_retained_git_admin_v3",
            "retention": (
                "descriptor_rooted_inotify_transient_reject_through_final_observation_v2"
            ),
            "hidden_admin_semantics": {
                "authority": "trusted_clean_source_producer_attestation_v1",
                "live_recheck": (
                    "require_current_source_recomputes_retained_hidden_bytes_v1"
                ),
                "privacy": (
                    "config_keys_values_subsections_and_inactive_ref_names_"
                    "not_published_v1"
                ),
            },
            "layout": self.layout,
            "ref_format": "files",
            "git_dir": _directory_binding(
                self.git_dir,
                descriptor=self.git_reader.descriptor,
            ),
            "git_common_dir": _directory_binding(
                self.common_dir,
                descriptor=self.common_reader.descriptor,
            ),
            "worktree_git_entry": marker,
            "commondir_pointer": {
                "file": commondir,
                "target": commondir_target,
                "line": commondir_line,
            },
            "gitdir_pointer": {
                "file": gitdir_pointer,
                "target": gitdir_target,
                "line": gitdir_line,
            },
            "config": {
                "common": common_config,
                "worktree": worktree_config,
                "policy": config_policy,
            },
            "excludes_and_attributes": {
                "info_exclude": info_exclude,
                "info_attributes": info_attributes,
            },
            "head": {
                "kind": head_kind,
                "ref": head_ref,
                "oid": active_oid,
                "binding": head_binding,
            },
            "active_ref": {
                "storage": active_storage,
                "name": head_ref,
                "oid": active_oid,
                "loose": loose_ref,
                "packed_refs": packed_refs,
            },
            "index_path": str(self.git_dir / "index"),
        }


@contextlib.contextmanager
def _retain_git_administration(
    *,
    work: Path,
    work_reader: PinnedDirectoryReader,
    repository_root: Path,
    repository_reader: PinnedDirectoryReader,
):
    with contextlib.ExitStack() as stack:
        marker_guard = stack.enter_context(
            PinnedPathSetMonitor.from_reader(
                repository_reader,
                regular_paths=(),
                recursive_directories=(),
                filtered_directories={".": (".git",)},
            )
        )
        marker_guard.verify_unchanged()
        try:
            marker_metadata = os.stat(
                ".git",
                dir_fd=repository_reader.descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            _fail(
                "GATE_GIT_ADMIN_INVALID",
                "worktree .git entry is unavailable",
                errno=exc.errno,
            )
        marker_path = repository_root / ".git"
        marker_file_monitor = None
        if stat.S_ISDIR(marker_metadata.st_mode):
            layout = "ordinary_gitdir"
            git_dir = _canonical_path(
                marker_path,
                role="ordinary Git directory",
                directory=True,
            )
        elif stat.S_ISREG(marker_metadata.st_mode) and marker_metadata.st_nlink == 1:
            layout = "linked_worktree"
            marker_file_monitor = stack.enter_context(
                PinnedPathSetMonitor.from_reader(
                    repository_reader,
                    regular_paths=(".git",),
                    recursive_directories=(),
                )
            )
            _marker_binding, marker_payload = _required_admin_regular(
                repository_reader,
                repository_root,
                ".git",
            )
            git_dir = _git_pointer_target(
                marker_payload,
                role="linked-worktree .git marker",
                base=repository_root,
                prefix="gitdir: ",
                directory=True,
            )
        else:
            _fail(
                "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                "worktree .git entry is not an admitted directory or gitfile",
            )

        git_reader = stack.enter_context(PinnedDirectoryReader(git_dir))
        git_monitor = stack.enter_context(
            PinnedPathSetMonitor.from_reader(
                git_reader,
                regular_paths=("HEAD",),
                optional_regular_paths=(
                    "commondir",
                    "gitdir",
                    "config.worktree",
                ),
                recursive_directories=(),
            )
        )
        commondir, commondir_payload = _optional_admin_regular(
            git_reader,
            git_dir,
            "commondir",
        )
        gitdir_pointer, gitdir_payload = _optional_admin_regular(
            git_reader,
            git_dir,
            "gitdir",
        )
        if layout == "ordinary_gitdir":
            if commondir["state"] != "absent" or gitdir_pointer["state"] != "absent":
                _fail(
                    "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                    "ordinary Git directory contains linked-worktree pointers",
                )
            common_dir = git_dir
        else:
            if commondir_payload is None or gitdir_payload is None:
                _fail(
                    "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                    "linked worktree is missing commondir or gitdir",
                )
            common_dir = _git_pointer_target(
                commondir_payload,
                role="linked-worktree commondir pointer",
                base=git_dir,
                directory=True,
            )
            gitfile_target = _git_pointer_target(
                gitdir_payload,
                role="linked-worktree gitdir pointer",
                base=git_dir,
                directory=False,
            )
            if (
                gitfile_target != marker_path
                or git_dir.parent.name != "worktrees"
                or git_dir.parent.parent != common_dir
            ):
                _fail(
                    "GATE_SOURCE_LAYOUT_UNSUPPORTED",
                    "linked-worktree administrative topology is unsupported",
                )

        common_reader = stack.enter_context(PinnedDirectoryReader(common_dir))
        _head_binding, head_payload = _required_admin_regular(
            git_reader,
            git_dir,
            "HEAD",
        )
        _head_kind, active_ref, _detached_oid = _parse_head(head_payload)
        common_optional_paths = [
            "packed-refs",
            "info/exclude",
            "info/attributes",
        ]
        if active_ref is not None:
            common_optional_paths.append(active_ref)
        common_monitor = stack.enter_context(
            PinnedPathSetMonitor.from_reader(
                common_reader,
                regular_paths=("config",),
                optional_regular_paths=tuple(common_optional_paths),
                recursive_directories=(),
            )
        )
        retention = _GitAdministrationRetention(
            work=work,
            work_descriptor=work_reader.descriptor,
            repository_root=repository_root,
            repository_reader=repository_reader,
            marker_guard=marker_guard,
            marker_file_monitor=marker_file_monitor,
            layout=layout,
            marker_path=marker_path,
            git_dir=git_dir,
            git_reader=git_reader,
            git_monitor=git_monitor,
            common_dir=common_dir,
            common_reader=common_reader,
            common_monitor=common_monitor,
            active_ref=active_ref,
        )
        retention.verify()
        yield retention
        retention.verify()


def _validate_env_name(name: str) -> None:
    if not _ENV_NAME_RE.fullmatch(name):
        _fail("GATE_ENVIRONMENT_INVALID", "environment variable name is invalid", name=name)
    upper = name.upper()
    if any(fragment in upper for fragment in _SECRET_FRAGMENTS) or upper.endswith("_KEY") or "_KEY_" in upper:
        _fail("GATE_ENVIRONMENT_SECRET_FORBIDDEN", "secret-bearing environment names cannot enter a gate receipt", name=name)


def _environment(
    temp_root: Path,
    *,
    inherited_names: Sequence[str],
    assignments: Sequence[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (*_DEFAULT_ENVIRONMENT, *inherited_names):
        _validate_env_name(name)
        if name not in _ALLOWED_ENVIRONMENT:
            _fail(
                "GATE_ENVIRONMENT_NOT_ALLOWED",
                "environment variable is outside the audited gate allowlist",
                name=name,
            )
        if name in os.environ:
            result[name] = os.environ[name]
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator:
            _fail("GATE_ENVIRONMENT_INVALID", "--env requires NAME=VALUE", assignment=assignment)
        _validate_env_name(name)
        if name not in _ALLOWED_ENVIRONMENT:
            _fail(
                "GATE_ENVIRONMENT_NOT_ALLOWED",
                "environment variable is outside the audited gate allowlist",
                name=name,
            )
        if "\x00" in value:
            _fail("GATE_ENVIRONMENT_INVALID", "environment value contains NUL", name=name)
        result[name] = value
    fixed = {
        "TMP": str(temp_root / "tmp"),
        "TEMP": str(temp_root / "tmp"),
        "TMPDIR": str(temp_root / "tmp"),
        "CARGO_TARGET_DIR": str(temp_root / "cargo-target"),
        "PYTHONPYCACHEPREFIX": str(temp_root / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, value in fixed.items():
        if name in result and result[name] != value:
            _fail("GATE_ENVIRONMENT_RESERVED", "gate-owned environment variable cannot be overridden", name=name)
        result[name] = value
    encoded_size = sum(len(key.encode()) + len(value.encode()) for key, value in result.items())
    if encoded_size > _MAX_ENV_BYTES:
        _fail("GATE_ENVIRONMENT_OVERSIZED", "gate environment exceeds its byte limit")
    return dict(sorted(result.items()))


def _temporary_lexical_path(path: str | Path) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        _fail(
            "GATE_TEMP_PATH_NOT_ABSOLUTE",
            "temporary root must be absolute",
            path=str(path),
        )
    lexical = Path(os.path.abspath(os.fspath(requested)))
    if lexical.parent == lexical:
        _fail(
            "GATE_TEMP_PATH_NOT_CANONICAL",
            "temporary root cannot be a filesystem root",
        )
    try:
        leaf = safe_relative_path(lexical.name)
    except ImmutableArtifactError as exc:
        _fail(
            "GATE_TEMP_PATH_NOT_CANONICAL",
            "temporary root leaf is unsafe",
            error=str(exc),
        )
    if len(leaf.parts) != 1:
        _fail(
            "GATE_TEMP_PATH_NOT_CANONICAL",
            "temporary root leaf is not one component",
        )
    return lexical


@contextlib.contextmanager
def _create_temp_root(path: str | Path):
    lexical = _temporary_lexical_path(path)
    leaf = safe_relative_path(lexical.name)
    parent_path = lexical.parent
    with PinnedDirectoryReader(parent_path) as parent_reader:
        parent = _external_directory(parent_path, role="gate temporary parent")
        if parent_path != parent:
            _fail("GATE_TEMP_PATH_NOT_CANONICAL", "temporary root parent is not canonical", path=str(path))
        parent_reader.verify_path()
        try:
            os.mkdir(leaf.name, mode=0o700, dir_fd=parent_reader.descriptor)
        except FileExistsError:
            _fail("GATE_TEMP_PATH_EXISTS", "temporary root already exists", path=str(lexical))
        except OSError as exc:
            _fail("GATE_TEMP_CREATE_FAILED", "temporary root could not be created", path=str(lexical), errno=exc.errno)
        created = os.stat(
            leaf.name,
            dir_fd=parent_reader.descriptor,
            follow_symlinks=False,
        )
        with PinnedDirectoryReader(lexical) as reader:
            retained = os.fstat(reader.descriptor)
            if (
                not stat.S_ISDIR(created.st_mode)
                or created.st_dev != retained.st_dev
                or created.st_ino != retained.st_ino
            ):
                _fail("GATE_TEMP_CREATE_FAILED", "temporary root identity changed during creation")
            for name in ("tmp", "cargo-target", "pycache"):
                os.mkdir(name, mode=0o700, dir_fd=reader.descriptor)
            reader.verify_path()
            parent_reader.verify_path()
            temporary = _RetainedTemporaryRoot(
                path=lexical,
                binding=_directory_binding(lexical, descriptor=reader.descriptor),
                reader=reader,
            )
            try:
                yield temporary
            finally:
                reader.verify_path()
                parent_reader.verify_path()


def _seconds_to_ns(value: float, *, role: str) -> int:
    if not math.isfinite(value) or value < 0:
        _fail("GATE_RESOURCE_USAGE_INVALID", f"{role} is not finite and nonnegative")
    return int(round(value * 1_000_000_000))


def _realtime_ns() -> int:
    return time.time_ns()


def _supervision_decision(*, terminal_observed: bool, now_ns: int, deadline_ns: int) -> str:
    if terminal_observed and now_ns <= deadline_ns:
        return "terminal"
    if now_ns >= deadline_ns:
        return "timeout"
    return "continue"


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@dataclasses.dataclass(slots=True)
class _TerminationSignalState:
    signum: int | None = None


class _TerminationSignalRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"termination signal {signum} requested")
        self.signum = signum


def _install_termination_signal_deferral(
    state: _TerminationSignalState,
) -> dict[int, Any]:
    try:
        current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    except (OSError, ValueError) as exc:
        _fail(
            "GATE_SIGNAL_MASK_UNAVAILABLE",
            "termination signal mask could not be inspected",
            error=type(exc).__name__,
        )
    bindings: dict[int, Any] = {}

    def defer(signum: int, _frame: Any) -> None:
        if state.signum is None:
            state.signum = signum

    try:
        for signum in _DEFERRED_TERMINATION_SIGNALS:
            disposition = signal.getsignal(signum)
            if disposition == signal.SIG_IGN:
                continue
            if disposition != signal.SIG_DFL:
                _fail(
                    "GATE_SIGNAL_DISPOSITION_UNSUPPORTED",
                    "termination signal has a custom disposition",
                    signal=signum,
                )
            if signum in current_mask:
                _fail(
                    "GATE_SIGNAL_DISPOSITION_UNSUPPORTED",
                    "termination signal is blocked by the supervisor",
                    signal=signum,
                )
            bindings[signum] = disposition
            signal.signal(signum, defer)
    except BaseException:
        for signum, disposition in reversed(tuple(bindings.items())):
            try:
                signal.signal(signum, disposition)
            except BaseException:
                pass
        raise
    return bindings


def _restore_termination_signal_handlers(
    bindings: Mapping[int, Any],
) -> list[str]:
    failures: list[str] = []
    for signum, disposition in reversed(tuple(bindings.items())):
        try:
            signal.signal(signum, disposition)
        except BaseException as exc:
            failures.append(
                f"signal-{signum}:{type(exc).__name__}:{exc}"
            )
    return failures


def _raise_deferred_termination(state: _TerminationSignalState) -> None:
    if state.signum is not None:
        raise _TerminationSignalRequested(state.signum)


@dataclasses.dataclass(slots=True)
class _ChildAuthority:
    process: Any | None = None
    pid: int | None = None
    launch_committed: bool = False
    numeric_signal_disarmed: bool = False
    waitid_exact: bool = False
    wait4_started: bool = False
    reaped: bool = False
    termination_error: BaseException | None = None
    waitid_error: BaseException | None = None
    wait4_error: BaseException | None = None
    fallback_error: BaseException | None = None


def _require_exact_child_platform() -> None:
    if platform.system() != "Linux" or not all(
        hasattr(os, name)
        for name in ("P_PID", "WNOWAIT", "wait4", "waitid", "waitpid")
    ):
        _fail(
            "GATE_PLATFORM_UNSUPPORTED",
            "exact-child gate supervision requires Linux waitid/wait4",
        )


def _require_exclusive_reaper_authority() -> None:
    try:
        disposition = signal.getsignal(signal.SIGCHLD)
    except (OSError, ValueError) as exc:
        _fail(
            "GATE_REAPER_AUTHORITY_UNAVAILABLE",
            "SIGCHLD disposition could not be inspected",
            error=type(exc).__name__,
        )
    if disposition != signal.SIG_DFL:
        _fail(
            "GATE_REAPER_AUTHORITY_UNAVAILABLE",
            "exact-child supervision requires default SIGCHLD disposition",
        )
    try:
        task_count = len(os.listdir("/proc/self/task"))
    except OSError as exc:
        _fail(
            "GATE_REAPER_AUTHORITY_UNAVAILABLE",
            "native thread inventory could not be inspected",
            errno=exc.errno,
        )
    if task_count != 1:
        _fail(
            "GATE_REAPER_AUTHORITY_UNAVAILABLE",
            "exact-child supervision requires one native supervisor thread",
            native_thread_count=task_count,
        )


def _block_async_signals() -> set[signal.Signals]:
    try:
        previous = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _BLOCKABLE_SIGNALS,
        )
    except (OSError, ValueError) as exc:
        _fail(
            "GATE_SIGNAL_MASK_UNAVAILABLE",
            "asynchronous signals could not be deferred",
            error=type(exc).__name__,
        )
    return previous


def _restore_async_signals(previous: set[signal.Signals]) -> None:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _launch_preowned_process(
    authority: _ChildAuthority,
    *,
    argv: Sequence[str],
    retained: Any,
    cwd_descriptor: int,
    environment: Mapping[str, str],
    stdout_descriptor: int,
    stderr_descriptor: int,
) -> None:
    previous_mask = _block_async_signals()
    try:
        process = subprocess.Popen.__new__(subprocess.Popen)
        authority.process = process

        def restore_child_signal_mask() -> None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        initialized = False
        try:
            subprocess.Popen.__init__(
                process,
                argv,
                executable=retained.proc_path,
                cwd=Path(f"/proc/self/fd/{cwd_descriptor}"),
                env=dict(environment),
                stdout=stdout_descriptor,
                stderr=stderr_descriptor,
                pass_fds=(retained.descriptor, cwd_descriptor),
                start_new_session=True,
                preexec_fn=restore_child_signal_mask,
            )
            initialized = True
        finally:
            pid = getattr(process, "pid", None)
            return_code = getattr(process, "returncode", None)
            if type(pid) is int and pid > 0:
                authority.pid = pid
            if return_code is not None:
                authority.numeric_signal_disarmed = True
                authority.wait4_started = True
                authority.reaped = True
            if initialized:
                authority.launch_committed = True
    finally:
        _restore_async_signals(previous_mask)


def _terminate_before_reap(authority: _ChildAuthority) -> None:
    pid = authority.pid
    if (
        pid is None
        or authority.reaped
        or authority.numeric_signal_disarmed
        or authority.wait4_started
    ):
        return
    try:
        try:
            _kill_group(pid)
        except BaseException as exc:
            authority.termination_error = exc
            try:
                os.kill(-pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException:
                pass
        if not authority.launch_committed or authority.termination_error is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as exc:
                if authority.termination_error is None:
                    authority.termination_error = exc
    finally:
        authority.numeric_signal_disarmed = True


def _accept_waitid_observation(
    authority: _ChildAuthority,
    observation: Any | None,
) -> bool:
    if observation is None or observation.si_pid != authority.pid:
        authority.waitid_error = RuntimeError(
            "waitid observed a missing or different child"
        )
        return False
    authority.waitid_exact = True
    return True


def _blocking_waitid(authority: _ChildAuthority) -> Any | None:
    pid = authority.pid
    if pid is None:
        authority.waitid_error = RuntimeError("child PID authority is absent")
        return None
    while True:
        try:
            observation = os.waitid(
                os.P_PID,
                pid,
                os.WEXITED | os.WNOWAIT,
            )
            break
        except InterruptedError:
            continue
        except BaseException as exc:
            authority.waitid_error = exc
            if isinstance(exc, (ChildProcessError, OSError)) and getattr(
                exc, "errno", errno.ECHILD
            ) == errno.ECHILD:
                authority.numeric_signal_disarmed = True
            return None
    _accept_waitid_observation(authority, observation)
    return observation


def _fallback_exact_reap(authority: _ChildAuthority) -> None:
    pid = authority.pid
    if pid is None or authority.reaped:
        return
    authority.numeric_signal_disarmed = True
    authority.wait4_started = True
    while True:
        try:
            waited_pid, _wait_status = os.waitpid(pid, 0)
            authority.reaped = True
            if waited_pid != pid:
                authority.fallback_error = RuntimeError(
                    "fallback waitpid reaped a different child"
                )
            return
        except InterruptedError:
            continue
        except (ChildProcessError, OSError) as exc:
            if isinstance(exc, ChildProcessError) or getattr(
                exc, "errno", None
            ) == errno.ECHILD:
                authority.reaped = True
                return
            authority.fallback_error = exc
            return
        except BaseException as exc:
            authority.fallback_error = exc
            return


def _wait4_exact(
    authority: _ChildAuthority,
) -> tuple[int, int, resource.struct_rusage] | None:
    pid = authority.pid
    if pid is None:
        authority.wait4_error = RuntimeError("child PID authority is absent")
        return None
    authority.numeric_signal_disarmed = True
    authority.wait4_started = True
    while True:
        try:
            waited_pid, wait_status, usage = os.wait4(pid, 0)
            authority.reaped = True
            break
        except InterruptedError:
            continue
        except BaseException as exc:
            authority.wait4_error = exc
            _fallback_exact_reap(authority)
            return None
    if waited_pid != pid:
        authority.wait4_error = RuntimeError("wait4 reaped a different child")
        return None
    return waited_pid, wait_status, usage


def _authority_failures(authority: _ChildAuthority) -> list[str]:
    failures: list[str] = []
    for role, value in (
        ("termination", authority.termination_error),
        ("waitid", authority.waitid_error),
        ("wait4", authority.wait4_error),
        ("fallback", authority.fallback_error),
    ):
        if value is not None:
            failures.append(f"{role}:{type(value).__name__}:{value}")
    if authority.pid is not None and not authority.reaped:
        failures.append("exact child was not proven reaped")
    return failures


def _finish_child_authority(
    authority: _ChildAuthority,
    *,
    terminal_observation: Any | None,
) -> tuple[int, int, resource.struct_rusage]:
    previous_mask = _block_async_signals()
    result: tuple[int, int, resource.struct_rusage] | None = None
    try:
        _terminate_before_reap(authority)
        if terminal_observation is None:
            _blocking_waitid(authority)
        else:
            _accept_waitid_observation(authority, terminal_observation)
        if authority.waitid_exact:
            result = _wait4_exact(authority)
    finally:
        _restore_async_signals(previous_mask)
    failures = _authority_failures(authority)
    if result is None or failures:
        _fail(
            "GATE_CLEANUP_AUTHORITY_LOST",
            "exact-child termination or reap authority was not proven",
            failures=failures,
        )
    return result


def _cleanup_child_authority(authority: _ChildAuthority) -> list[str]:
    if authority.pid is None or authority.reaped:
        return _authority_failures(authority)
    previous_mask: set[signal.Signals] | None = None
    mask_error: BaseException | None = None
    try:
        previous_mask = _block_async_signals()
    except BaseException as exc:
        mask_error = exc
    restore_error: BaseException | None = None
    try:
        if not authority.wait4_started:
            _terminate_before_reap(authority)
            if not authority.waitid_exact:
                _blocking_waitid(authority)
            _wait4_exact(authority)
        elif not authority.reaped:
            _fallback_exact_reap(authority)
    finally:
        if previous_mask is not None:
            try:
                _restore_async_signals(previous_mask)
            except BaseException as exc:
                restore_error = exc
    failures = _authority_failures(authority)
    if mask_error is not None:
        failures.append(
            f"signal-mask:{type(mask_error).__name__}:{mask_error}"
        )
    if restore_error is not None:
        failures.append(
            f"signal-restore:{type(restore_error).__name__}:{restore_error}"
        )
    return failures


@dataclasses.dataclass(slots=True)
class _PipeDrain:
    role: str
    descriptor: int
    maximum_bytes: int
    payload: bytearray = dataclasses.field(default_factory=bytearray)
    eof: bool = False

    def consume_once(self) -> None:
        if self.eof:
            return
        request_bytes = min(
            _PIPE_READ_QUANTUM,
            self.maximum_bytes + 1 - len(self.payload),
        )
        try:
            chunk = os.read(self.descriptor, max(1, request_bytes))
        except InterruptedError:
            return
        except BlockingIOError:
            return
        except OSError as exc:
            _fail(
                "GATE_STREAM_READ_FAILED",
                f"gate {self.role} pipe could not be read",
                stream=self.role,
                errno=exc.errno,
            )
        if not chunk:
            self.eof = True
            return
        if len(self.payload) + len(chunk) > self.maximum_bytes:
            _fail(
                "GATE_STREAM_OVERSIZED",
                f"gate {self.role} exceeded its cumulative byte limit",
                stream=self.role,
                observed_minimum_bytes=self.maximum_bytes + 1,
                limit_bytes=self.maximum_bytes,
            )
        self.payload.extend(chunk)


def _spawn_exact_child(
    *,
    retained: Any,
    process_argv0: str,
    arguments: Sequence[str],
    cwd_descriptor: int,
    environment: Mapping[str, str],
    timeout_ns: int,
) -> _ChildObservation:
    _require_exact_child_platform()
    _require_exclusive_reaper_authority()
    argv = [process_argv0, *arguments]
    started_at_unix_ns = _realtime_ns()
    monotonic_start_ns = time.monotonic_ns()
    monotonic_deadline_ns = monotonic_start_ns + timeout_ns
    authority = _ChildAuthority()
    termination_state = _TerminationSignalState()
    termination_bindings = _install_termination_signal_deferral(
        termination_state
    )
    stream_sockets: list[socket.socket] = []
    try:
        try:
            stdout_parent, stdout_child = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_STREAM
            )
            stderr_parent, stderr_child = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_STREAM
            )
            stream_sockets.extend(
                (stdout_parent, stdout_child, stderr_parent, stderr_child)
            )
            stdout_parent.shutdown(socket.SHUT_WR)
            stderr_parent.shutdown(socket.SHUT_WR)
            stdout_child.shutdown(socket.SHUT_RD)
            stderr_child.shutdown(socket.SHUT_RD)
            for parent_endpoint in (stdout_parent, stderr_parent):
                parent_endpoint.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_OOBINLINE,
                    1,
                )
                if (
                    parent_endpoint.getsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_OOBINLINE,
                    )
                    != 1
                ):
                    _fail(
                        "GATE_STREAM_CAPTURE_UNAVAILABLE",
                        "gate stream urgent-data capture could not be enabled",
                    )
        except OSError as exc:
            _fail(
                "GATE_STREAM_CAPTURE_UNAVAILABLE",
                "gate stream socketpairs could not be created",
                errno=exc.errno,
            )
        _raise_deferred_termination(termination_state)
        try:
            _launch_preowned_process(
                authority,
                argv=argv,
                retained=retained,
                cwd_descriptor=cwd_descriptor,
                environment=environment,
                stdout_descriptor=stdout_child.fileno(),
                stderr_descriptor=stderr_child.fileno(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _fail(
                "GATE_SPAWN_FAILED",
                "gate child could not be spawned",
                errno=getattr(exc, "errno", None),
                error=type(exc).__name__,
            )
        if authority.pid is None or not authority.launch_committed:
            _fail(
                "GATE_SPAWN_FAILED",
                "gate child launch did not commit exact PID authority",
            )
        _raise_deferred_termination(termination_state)
        stdout_child.close()
        stderr_child.close()
        stream_sockets.remove(stdout_child)
        stream_sockets.remove(stderr_child)
        stdout_drain = _PipeDrain("stdout", stdout_parent.fileno(), _MAX_STREAM_BYTES)
        stderr_drain = _PipeDrain("stderr", stderr_parent.fileno(), _MAX_STREAM_BYTES)
        drains = (stdout_drain, stderr_drain)
        poller = select.poll()
        stream_mask = (
            select.POLLIN
            | select.POLLPRI
            | select.POLLHUP
            | select.POLLERR
        )
        try:
            for drain in drains:
                os.set_blocking(drain.descriptor, False)
                poller.register(drain.descriptor, stream_mask)
        except (OSError, ValueError) as exc:
            _fail(
                "GATE_STREAM_CAPTURE_UNAVAILABLE",
                "gate pipe capture could not be initialized",
                errno=getattr(exc, "errno", None),
            )

        terminal_observation: Any | None = None
        decision = "continue"
        monotonic_decision_ns = monotonic_start_ns
        while decision == "continue":
            _raise_deferred_termination(termination_state)
            now_ns = time.monotonic_ns()
            remaining_ns = monotonic_deadline_ns - now_ns
            if remaining_ns > 0:
                timeout_ms = min(
                    max(1, int(math.ceil(_POLL_SECONDS * 1000))),
                    max(1, (remaining_ns + 999_999) // 1_000_000),
                )
                try:
                    events = poller.poll(timeout_ms)
                except InterruptedError:
                    continue
                ready = {descriptor for descriptor, _mask in events}
                for drain in drains:
                    if drain.descriptor in ready:
                        drain.consume_once()
                        if drain.eof:
                            try:
                                poller.unregister(drain.descriptor)
                            except KeyError:
                                pass
            _raise_deferred_termination(termination_state)
            while True:
                try:
                    terminal_observation = os.waitid(
                        os.P_PID,
                        authority.pid,
                        os.WEXITED | os.WNOHANG | os.WNOWAIT,
                    )
                    break
                except InterruptedError:
                    continue
                except OSError as exc:
                    if exc.errno == errno.ECHILD:
                        authority.numeric_signal_disarmed = True
                    _fail(
                        "GATE_WAIT_FAILED",
                        "waitid supervision failed",
                        errno=exc.errno,
                    )
            monotonic_decision_ns = time.monotonic_ns()
            decision = _supervision_decision(
                terminal_observed=terminal_observation is not None,
                now_ns=monotonic_decision_ns,
                deadline_ns=monotonic_deadline_ns,
            )

        _raise_deferred_termination(termination_state)
        waited_pid, wait_status, usage = _finish_child_authority(
            authority,
            terminal_observation=terminal_observation,
        )
        _raise_deferred_termination(termination_state)
        monotonic_stop_ns = time.monotonic_ns()
        finished_at_unix_ns = _realtime_ns()
        return_code = os.waitstatus_to_exitcode(wait_status)
        if authority.process is not None:
            authority.process.returncode = return_code

        drain_deadline_ns = time.monotonic_ns() + _PIPE_DRAIN_TIMEOUT_NS
        while not all(drain.eof for drain in drains):
            _raise_deferred_termination(termination_state)
            for drain in drains:
                if not drain.eof:
                    drain.consume_once()
                    if drain.eof:
                        try:
                            poller.unregister(drain.descriptor)
                        except KeyError:
                            pass
            if all(drain.eof for drain in drains):
                break
            remaining_ns = drain_deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                _fail(
                    "GATE_STREAM_DRAIN_TIMEOUT",
                    "gate stream writers remained after exact-child cleanup",
                    streams=[drain.role for drain in drains if not drain.eof],
                    timeout_ns=_PIPE_DRAIN_TIMEOUT_NS,
                )
            try:
                poller.poll(max(1, min(20, (remaining_ns + 999_999) // 1_000_000)))
            except InterruptedError:
                continue

        _raise_deferred_termination(termination_state)
        return _ChildObservation(
            pid=authority.pid,
            waited_pid=waited_pid,
            started_at_unix_ns=started_at_unix_ns,
            finished_at_unix_ns=finished_at_unix_ns,
            monotonic_start_ns=monotonic_start_ns,
            monotonic_stop_ns=monotonic_stop_ns,
            requested_timeout_ns=timeout_ns,
            monotonic_deadline_ns=monotonic_deadline_ns,
            monotonic_decision_ns=monotonic_decision_ns,
            decision=decision,
            wait_status=wait_status,
            return_code=return_code,
            timed_out=decision == "timeout",
            usage=usage,
            stdout=bytes(stdout_drain.payload),
            stderr=bytes(stderr_drain.payload),
        )
    except BaseException as exc:
        cleanup_failures = _cleanup_child_authority(authority)
        if cleanup_failures:
            if (
                isinstance(exc, GateReceiptError)
                and exc.code == "GATE_CLEANUP_AUTHORITY_LOST"
            ):
                raise
            raise GateReceiptError(
                "GATE_CLEANUP_AUTHORITY_LOST",
                "gate failure cleanup did not prove exact-child authority",
                context={
                    "primary_error": type(exc).__name__,
                    "failures": cleanup_failures,
                },
            ) from exc
        raise
    finally:
        for stream_socket in stream_sockets:
            try:
                stream_socket.close()
            except OSError:
                pass
        restore_failures = _restore_termination_signal_handlers(
            termination_bindings
        )
        if restore_failures:
            raise GateReceiptError(
                "GATE_SIGNAL_RESTORE_FAILED",
                "termination signal handlers could not be restored",
                context={"failures": restore_failures},
            )
        if termination_state.signum is not None and (
            authority.pid is None or authority.reaped
        ):
            os.kill(os.getpid(), termination_state.signum)
            raise RuntimeError("termination signal propagation returned")


def _stream_identity(name: str, payload: bytes) -> dict[str, Any]:
    return {"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _shell_return_code_from_wait(wait: Mapping[str, Any]) -> int:
    if wait["timed_out"]:
        return 124
    return_code = wait["return_code"]
    if return_code < 0:
        return min(255, 128 + -return_code)
    return min(255, return_code)


def _outcome(observation: _ChildObservation) -> tuple[dict[str, Any], int]:
    wait = _wait_document(observation)
    shell_return_code = _shell_return_code_from_wait(wait)
    if observation.timed_out:
        return {"status": "failed", "reason": "TIMEOUT"}, shell_return_code
    if observation.return_code == 0:
        return {"status": "passed", "reason": None}, shell_return_code
    if observation.return_code < 0:
        return {"status": "failed", "reason": "SIGNAL"}, shell_return_code
    return {"status": "failed", "reason": "NONZERO_EXIT"}, shell_return_code


def _wait_document(observation: _ChildObservation) -> dict[str, Any]:
    exited = os.WIFEXITED(observation.wait_status)
    signaled = os.WIFSIGNALED(observation.wait_status)
    return {
        "pid": observation.pid,
        "waited_pid": observation.waited_pid,
        "process_group_id": observation.pid,
        "session_id": observation.pid,
        "raw_wait_status": observation.wait_status,
        "return_code": observation.return_code,
        "exit_code": os.WEXITSTATUS(observation.wait_status) if exited else None,
        "signal": os.WTERMSIG(observation.wait_status) if signaled else None,
        "timed_out": observation.timed_out,
    }


def _retained_regular_identity(
    retained: Any,
    *,
    path: Path,
) -> dict[str, str]:
    snapshot = retained.snapshot
    return {
        "path": str(path),
        "sha256": retained.sha256,
        "device": str(snapshot[0]),
        "inode": str(snapshot[1]),
        "mode": str(snapshot[2]),
        "link_count": str(snapshot[3]),
        "size_bytes": str(snapshot[4]),
        "modified_ns": str(snapshot[5]),
        "changed_ns": str(snapshot[6]),
    }


def _require_clean_source_preflight(work: Path) -> dict[str, Any]:
    try:
        with PinnedDirectoryReader(work) as reader:
            snapshot = _git_snapshot(
                work, cwd_descriptor=reader.descriptor
            )
            if not snapshot["clean"]:
                _fail(
                    "GATE_SOURCE_DIRTY",
                    "gate source checkout is not clean",
                    status=snapshot["status_porcelain_v1"],
                )
            repository_root = Path(snapshot["repository_root"])
            with PinnedDirectoryReader(repository_root) as repository_reader:
                with _retain_git_administration(
                    work=work,
                    work_reader=reader,
                    repository_root=repository_root,
                    repository_reader=repository_reader,
                ) as administration:
                    admin_document = administration.document(
                        commit=snapshot["commit"],
                    )
            return {
                "git": snapshot,
                "git_administration": admin_document,
            }
    except ImmutableArtifactError as exc:
        _fail(
            "GATE_SOURCE_DRIFT",
            "gate source path changed during clean-source preflight",
            error=str(exc),
        )


def _execute_monitored_gate(
    *,
    work: Path,
    executable_path: Path,
    publication_root: Path,
    requested_binding_before: Mapping[str, Any],
    logical_argv0: str,
    arguments: Sequence[str],
    temp_root: str | Path,
    inherited_environment: Sequence[str],
    environment_assignments: Sequence[str],
    timeout_ns: int,
) -> _MonitoredGateExecution:
    try:
        with PinnedDirectoryReader(work) as work_reader:
            initial_git = _git_snapshot(
                work, cwd_descriptor=work_reader.descriptor
            )
            if not initial_git["clean"]:
                _fail(
                    "GATE_SOURCE_DIRTY",
                    "gate source checkout became dirty before source monitoring",
                    status=initial_git["status_porcelain_v1"],
                )
            repository_root = Path(initial_git["repository_root"])
            with PinnedDirectoryReader(repository_root) as repository_reader:
                work_reader.verify_path()
                require_file_descriptor_headroom(
                    128, label="gate tracked-source monitor"
                )
                initial_tracked = _tracked_input_manifest(
                    work, cwd_descriptor=work_reader.descriptor
                )
                index_path = _git_index_path(
                    work, cwd_descriptor=work_reader.descriptor
                )
                temporary_candidate = _temporary_lexical_path(temp_root)
                _require_disjoint_path(
                    temporary_candidate,
                    role="gate temporary root",
                    others=(
                        ("Git repository root", repository_root),
                        ("Git index", index_path),
                        ("publication root", publication_root),
                    ),
                    code="GATE_PATH_OVERLAP",
                )
                _require_disjoint_path(
                    publication_root,
                    role="publication root",
                    others=(
                        ("Git repository root", repository_root),
                        ("Git index", index_path),
                    ),
                    code="GATE_PATH_OVERLAP",
                )
                with open_retained_regular(
                    index_path, max_bytes=_MAX_GIT_INDEX_BYTES
                ) as retained_index, PinnedPathSetMonitor.from_reader(
                    repository_reader,
                    regular_paths=initial_tracked.paths,
                    recursive_directories=(),
                ) as source_monitor, _retain_git_administration(
                    work=work,
                    work_reader=work_reader,
                    repository_root=repository_root,
                    repository_reader=repository_reader,
                ) as git_administration:
                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    tracked_before = _tracked_input_manifest(
                        work, cwd_descriptor=work_reader.descriptor
                    )
                    worktree_before = _tracked_worktree_manifest(
                        repository_reader,
                        tracked_before,
                    )
                    git_before = _git_snapshot(
                        work, cwd_descriptor=work_reader.descriptor
                    )
                    git_administration_before = (
                        git_administration.document(
                            commit=git_before["commit"],
                        )
                    )
                    if Path(
                        git_administration_before["index_path"]
                    ) != index_path:
                        _fail(
                            "GATE_GIT_ADMIN_INVALID",
                            "retained Git index path disagrees with its probe",
                        )
                    retained_git_dir = Path(
                        git_administration_before["git_dir"]["path"]
                    )
                    retained_common_dir = Path(
                        git_administration_before["git_common_dir"]["path"]
                    )
                    _require_disjoint_path(
                        temporary_candidate,
                        role="gate temporary root",
                        others=(
                            ("Git directory", retained_git_dir),
                            ("Git common directory", retained_common_dir),
                        ),
                        code="GATE_PATH_OVERLAP",
                    )
                    _require_disjoint_path(
                        publication_root,
                        role="publication root",
                        others=(
                            ("Git directory", retained_git_dir),
                            ("Git common directory", retained_common_dir),
                        ),
                        code="GATE_PATH_OVERLAP",
                    )
                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    if tracked_before != initial_tracked or git_before != initial_git:
                        _fail(
                            "GATE_SOURCE_DRIFT",
                            "Git source identity changed while source monitoring was armed",
                        )
                    if not git_before["clean"]:
                        _fail(
                            "GATE_SOURCE_DIRTY",
                            "gate source checkout became dirty before spawn",
                            status=git_before["status_porcelain_v1"],
                        )
                    cwd_binding = _directory_binding(
                        work, descriptor=work_reader.descriptor
                    )
                    git_index_identity = _retained_regular_identity(
                        retained_index, path=index_path
                    )

                    try:
                        with _create_temp_root(temp_root) as temporary:
                            environment = _environment(
                                temporary.path,
                                inherited_names=inherited_environment,
                                assignments=environment_assignments,
                            )
                            try:
                                with open_retained_executable(
                                    executable_path
                                ) as retained:
                                    requested_snapshot = tuple(
                                        int(requested_binding_before[field])
                                        for field in (
                                            "device",
                                            "inode",
                                            "mode",
                                            "link_count",
                                            "size_bytes",
                                            "modified_ns",
                                            "changed_ns",
                                        )
                                    )
                                    if retained.snapshot != requested_snapshot:
                                        _fail(
                                            "GATE_EXECUTABLE_DRIFT",
                                            "retained executable differs from its admitted path binding",
                                        )
                                    executable_identity = {
                                        "requested_path": str(
                                            requested_binding_before["path"]
                                        ),
                                        "resolved_path": str(executable_path),
                                        "process_argv0": logical_argv0,
                                        "sha256": retained.sha256,
                                        "size_bytes": str(retained.snapshot[4]),
                                        "mode": str(retained.snapshot[2]),
                                        "requested_binding": dict(
                                            requested_binding_before
                                        ),
                                    }
                                    observation = _spawn_exact_child(
                                        retained=retained,
                                        process_argv0=logical_argv0,
                                        arguments=arguments,
                                        cwd_descriptor=work_reader.descriptor,
                                        environment=environment,
                                        timeout_ns=timeout_ns,
                                    )
                                    retained.verify_unchanged()
                            except ImmutableArtifactError as exc:
                                _fail(
                                    "GATE_EXECUTABLE_DRIFT",
                                    "retained executable identity failed",
                                    error=str(exc),
                                )

                            requested_binding_after = (
                                _requested_executable_binding(
                                    Path(requested_binding_before["path"])
                                )
                            )
                            if requested_binding_after != requested_binding_before:
                                _fail(
                                    "GATE_EXECUTABLE_DRIFT",
                                    "requested executable binding changed during gate",
                                )
                            temporary_document = {
                                "path": str(temporary.path),
                                "binding": temporary.binding,
                                "children": ["cargo-target", "pycache", "tmp"],
                                "policy": (
                                    "mkdirat_retained_parent_monitored_path_v1"
                                ),
                            }
                            temporary.reader.verify_path()
                    except ImmutableArtifactError as exc:
                        _fail(
                            "GATE_TEMP_DRIFT",
                            "retained temporary root changed during gate",
                            error=str(exc),
                        )

                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    work_reader.verify_path()
                    repository_reader.verify_path()
                    tracked_after = _tracked_input_manifest(
                        work, cwd_descriptor=work_reader.descriptor
                    )
                    worktree_after = _tracked_worktree_manifest(
                        repository_reader,
                        tracked_after,
                    )
                    git_after = _git_snapshot(
                        work, cwd_descriptor=work_reader.descriptor
                    )
                    git_administration_after = git_administration.document(
                        commit=git_after["commit"],
                    )
                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    if (
                        tracked_after != tracked_before
                        or worktree_after != worktree_before
                        or git_after != git_before
                        or git_administration_after
                        != git_administration_before
                    ):
                        _fail(
                            "GATE_SOURCE_DRIFT",
                            "Git source identity changed during execution",
                        )

                    return _MonitoredGateExecution(
                        observation=observation,
                        environment=environment,
                        executable=executable_identity,
                        temporary_root=temporary_document,
                        source={
                            "cwd_binding": cwd_binding,
                            "git": git_before,
                            "git_index": git_index_identity,
                            "tracked_inputs": tracked_before.document,
                            "checked_out_worktree": worktree_before,
                            "git_administration": git_administration_before,
                        },
                    )
    except ImmutableArtifactError as exc:
        _fail(
            "GATE_SOURCE_DRIFT",
            "tracked source changed during gate execution",
            error=str(exc),
        )


def run_gate(
    *,
    output_root: str | Path,
    generation: str,
    cwd: str | Path,
    temp_root: str | Path,
    timeout_seconds: str | int | float,
    argv: Sequence[str],
    process_argv0: str | None = None,
    inherited_environment: Sequence[str] = (),
    environment_assignments: Sequence[str] = (),
) -> GateRunResult:
    if not _GENERATION_RE.fullmatch(generation):
        _fail(
            "GATE_GENERATION_NAME_INVALID",
            "gate generation name is invalid",
            generation=generation,
        )
    if not argv:
        _fail("GATE_COMMAND_EMPTY", "gate command is empty")
    if any(type(value) is not str for value in argv):
        _fail("GATE_ARGUMENTS_INVALID", "gate arguments must be strings")
    logical_argv0 = argv[0] if process_argv0 is None else process_argv0
    if (
        type(logical_argv0) is not str
        or not logical_argv0
        or "\x00" in logical_argv0
    ):
        _fail(
            "GATE_ARGV0_INVALID",
            "logical process argv[0] must be a nonempty NUL-free string",
        )
    logical_argv0_path = Path(logical_argv0)
    if (
        not logical_argv0_path.is_absolute()
        or str(Path(os.path.abspath(os.fspath(logical_argv0_path))))
        != logical_argv0
    ):
        _fail(
            "GATE_ARGV0_INVALID",
            "logical process argv[0] must use an absolute canonical spelling",
        )
    command_argv = [logical_argv0, *argv[1:]]
    timeout_ns = _timeout_nanoseconds(timeout_seconds)
    _require_exact_child_platform()
    _require_exclusive_reaper_authority()
    if (
        sum(len(value.encode("utf-8")) for value in command_argv)
        > _MAX_ARG_BYTES
        or any("\x00" in value for value in command_argv)
    ):
        _fail(
            "GATE_ARGUMENTS_INVALID",
            "gate arguments are oversized or contain NUL",
        )

    output_candidate = _canonical_path(
        output_root, role="gate output root", directory=True
    )
    if output_candidate.parent == output_candidate:
        _fail(
            "GATE_OUTPUT_ROOT_INVALID",
            "gate output root cannot be a filesystem root",
        )
    work = _canonical_path(cwd, role="gate working directory", directory=True)
    executable_path, requested_binding_before = _resolved_executable(argv[0])
    temporary_candidate = _temporary_lexical_path(temp_root)
    source_preflight = _require_clean_source_preflight(work)
    preflight_git = source_preflight["git"]
    preflight_admin = source_preflight["git_administration"]
    repository_root = Path(preflight_git["repository_root"])
    git_dir = Path(preflight_admin["git_dir"]["path"])
    git_common_dir = Path(preflight_admin["git_common_dir"]["path"])
    git_index_path = Path(preflight_admin["index_path"])
    _require_disjoint_path(
        output_candidate,
        role="gate output root",
        others=(
            ("gate working directory", work),
            ("Git repository root", repository_root),
            ("Git directory", git_dir),
            ("Git common directory", git_common_dir),
            ("Git index", git_index_path),
            ("gate executable", executable_path),
        ),
        code="GATE_PATH_OVERLAP",
    )
    _require_disjoint_path(
        temporary_candidate,
        role="gate temporary root",
        others=(
            ("gate output root", output_candidate),
            ("gate working directory", work),
            ("Git repository root", repository_root),
            ("Git directory", git_dir),
            ("Git common directory", git_common_dir),
            ("Git index", git_index_path),
            ("gate executable", executable_path),
        ),
        code="GATE_PATH_OVERLAP",
    )

    with PinnedDirectoryReader(output_candidate) as output_reader:
        output = _external_directory(output_candidate, role="gate output root")
        output_reader.verify_path()
        publication_binding = _directory_binding(
            output, descriptor=output_reader.descriptor
        )
        with create_exclusive_directory_stage(
            output,
            generation,
            pinned_root=output_reader,
        ) as stage:
            execution = _execute_monitored_gate(
                work=work,
                executable_path=executable_path,
                publication_root=output,
                requested_binding_before=requested_binding_before,
                logical_argv0=logical_argv0,
                arguments=argv[1:],
                temp_root=temporary_candidate,
                inherited_environment=inherited_environment,
                environment_assignments=environment_assignments,
                timeout_ns=timeout_ns,
            )
            observation = execution.observation
            outcome, shell_return_code = _outcome(observation)
            stdout_identity = _stream_identity(
                STDOUT_NAME, observation.stdout
            )
            stderr_identity = _stream_identity(
                STDERR_NAME, observation.stderr
            )
            document: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "integrity": {
                    "seal": "canonical_json_unkeyed_sha256_consistency_v1",
                    "authority": (
                        "requires_external_content_manifest_sha256_anchor_and_"
                        "trusted_clean_source_producer_v1"
                    ),
                },
                "command": {"argv": command_argv, "cwd": str(work)},
                "executable": execution.executable,
                "environment": execution.environment,
                "supervision": {
                    **_SUPERVISION_POLICY,
                    "requested_timeout_ns": observation.requested_timeout_ns,
                    "monotonic_deadline_ns": str(
                        observation.monotonic_deadline_ns
                    ),
                    "monotonic_decision_ns": str(
                        observation.monotonic_decision_ns
                    ),
                    "decision": observation.decision,
                },
                "temporary_root": execution.temporary_root,
                "publication": {
                    "output_root": str(output),
                    "generation": generation,
                    "root_binding": publication_binding,
                    "binding": (
                        "pinned_directory_reader_descriptor_relative_v1"
                    ),
                },
                "source": execution.source,
                "time": {
                    "started_at_unix_ns": str(
                        observation.started_at_unix_ns
                    ),
                    "finished_at_unix_ns": str(
                        observation.finished_at_unix_ns
                    ),
                    "monotonic_start_ns": str(
                        observation.monotonic_start_ns
                    ),
                    "monotonic_stop_ns": str(
                        observation.monotonic_stop_ns
                    ),
                    "duration_ns": (
                        observation.monotonic_stop_ns
                        - observation.monotonic_start_ns
                    ),
                },
                "wait": _wait_document(observation),
                "resource_usage": {
                    "user_cpu_ns": _seconds_to_ns(
                        float(observation.usage.ru_utime),
                        role="user CPU time",
                    ),
                    "system_cpu_ns": _seconds_to_ns(
                        float(observation.usage.ru_stime),
                        role="system CPU time",
                    ),
                    "peak_rss_kib": int(observation.usage.ru_maxrss),
                    "minor_faults": int(observation.usage.ru_minflt),
                    "major_faults": int(observation.usage.ru_majflt),
                    "voluntary_context_switches": int(
                        observation.usage.ru_nvcsw
                    ),
                    "involuntary_context_switches": int(
                        observation.usage.ru_nivcsw
                    ),
                },
                "streams": {
                    "stdout": stdout_identity,
                    "stderr": stderr_identity,
                },
                "outcome": outcome,
            }
            document = seal_model(document, field="receipt_sha256")
            receipt_payload = pretty_json_bytes(document)
            _validate_receipt_bundle(
                receipt_payload=receipt_payload,
                stdout_payload=observation.stdout,
                stderr_payload=observation.stderr,
                generation_path=stage.final_path,
            )
            write_regular_bytes_at(
                stage.descriptor, STDOUT_NAME, observation.stdout
            )
            write_regular_bytes_at(
                stage.descriptor, STDERR_NAME, observation.stderr
            )
            write_regular_bytes_at(
                stage.descriptor, RECEIPT_NAME, receipt_payload
            )
            expected_files = tuple(
                sorted(
                    (
                        DirectoryGenerationFile(
                            RECEIPT_NAME,
                            hashlib.sha256(receipt_payload).hexdigest(),
                            len(receipt_payload),
                        ),
                        DirectoryGenerationFile(
                            STDERR_NAME,
                            stderr_identity["sha256"],
                            stderr_identity["size_bytes"],
                        ),
                        DirectoryGenerationFile(
                            STDOUT_NAME,
                            stdout_identity["sha256"],
                            stdout_identity["size_bytes"],
                        ),
                    ),
                    key=lambda identity: identity.relative_path,
                )
            )
            committed_generation = stage.commit(
                max_files=3,
                max_directories=1,
                max_total_file_bytes=(
                    2 * _MAX_STREAM_BYTES + _MAX_RECEIPT_BYTES
                ),
                max_total_relative_path_bytes=256,
                max_file_bytes_by_path={
                    STDOUT_NAME: _MAX_STREAM_BYTES,
                    STDERR_NAME: _MAX_STREAM_BYTES,
                    RECEIPT_NAME: _MAX_RECEIPT_BYTES,
                },
                expected_files=expected_files,
            )
            output_reader.verify_path()
            return GateRunResult(
                generation_path=stage.final_path,
                document=document,
                shell_return_code=shell_return_code,
                content_manifest_sha256=(
                    committed_generation.content_manifest_sha256
                ),
                filesystem_generation_sha256=(
                    committed_generation.filesystem_generation_sha256
                ),
            )


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, role: str) -> None:
    if set(value) != expected:
        _fail("GATE_RECEIPT_INVALID", f"{role} keys are not exact", observed=sorted(value), expected=sorted(expected))


def _object(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("GATE_RECEIPT_INVALID", f"{role} must be an object")
    return value


def _u64(value: Any, *, role: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 64) - 1:
        _fail("GATE_RECEIPT_INVALID", f"{role} must be an unsigned 64-bit integer")
    return value


def _decimal_u64(value: Any, *, role: str) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdigit()
        or str(int(value)) != value
        or int(value) > (1 << 64) - 1
    ):
        _fail("GATE_RECEIPT_INVALID", f"{role} must be a canonical decimal unsigned 64-bit integer")
    return int(value)


def _canonical_absolute_string(value: Any, *, role: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or value.startswith("//")
    ):
        _fail("GATE_RECEIPT_INVALID", f"{role} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute() or str(Path(os.path.abspath(os.fspath(path)))) != value:
        _fail("GATE_RECEIPT_INVALID", f"{role} must be a canonical absolute spelling")
    return value


def _validate_directory_binding(
    value: Any,
    *,
    role: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    binding = _object(value, role=role)
    _require_exact_keys(
        binding,
        {"path", "device", "inode", "mode", "uid", "gid"},
        role=role,
    )
    path = _canonical_absolute_string(binding["path"], role=f"{role}.path")
    values = {
        field: _decimal_u64(binding[field], role=f"{role}.{field}")
        for field in ("device", "inode", "mode", "uid", "gid")
    }
    if values["inode"] == 0 or not stat.S_ISDIR(values["mode"]):
        _fail("GATE_RECEIPT_INVALID", f"{role} is not a directory binding")
    if expected_path is not None and path != expected_path:
        _fail("GATE_RECEIPT_INVALID", f"{role} path disagrees")
    return binding


def _validate_regular_binding(
    value: Any,
    *,
    role: str,
    max_bytes: int,
    expected_path: str | None = None,
) -> dict[str, Any]:
    binding = _object(value, role=role)
    _require_exact_keys(
        binding,
        {
            "path",
            "sha256",
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "modified_ns",
            "changed_ns",
        },
        role=role,
    )
    path = _canonical_absolute_string(binding["path"], role=f"{role}.path")
    if (
        type(binding["sha256"]) is not str
        or not _SHA256_RE.fullmatch(binding["sha256"])
    ):
        _fail("GATE_RECEIPT_INVALID", f"{role} SHA-256 is malformed")
    values = {
        field: _decimal_u64(binding[field], role=f"{role}.{field}")
        for field in (
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "modified_ns",
            "changed_ns",
        )
    }
    if (
        values["inode"] == 0
        or not stat.S_ISREG(values["mode"])
        or values["link_count"] != 1
        or values["size_bytes"] > max_bytes
    ):
        _fail(
            "GATE_RECEIPT_INVALID",
            f"{role} is not a producer-admissible regular-file binding",
        )
    if expected_path is not None and path != expected_path:
        _fail("GATE_RECEIPT_INVALID", f"{role} path disagrees")
    return binding


def _require_regular_binding_payload(
    binding: Mapping[str, Any],
    payload: bytes,
    *,
    role: str,
) -> None:
    if (
        int(binding["size_bytes"]) != len(payload)
        or binding["sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        _fail(
            "GATE_RECEIPT_INVALID",
            f"{role} binding disagrees with its claimed semantic payload",
        )


def _validate_git_pointer_line(
    line: Any,
    binding: Mapping[str, Any],
    *,
    role: str,
    base: str,
    expected_target: str,
    prefix: str = "",
) -> None:
    if (
        type(line) is not str
        or not line
        or any(ord(character) < 32 or ord(character) == 127 for character in line)
    ):
        _fail("GATE_RECEIPT_INVALID", f"{role} line is malformed")
    payload = f"{line}\n".encode("utf-8")
    _require_regular_binding_payload(binding, payload, role=role)
    value = line
    if prefix:
        if not value.startswith(prefix):
            _fail("GATE_RECEIPT_INVALID", f"{role} prefix is malformed")
        value = value[len(prefix):]
    if not value:
        _fail("GATE_RECEIPT_INVALID", f"{role} target is empty")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(base) / candidate
    observed_target = str(Path(os.path.abspath(os.fspath(candidate))))
    if observed_target != expected_target:
        _fail("GATE_RECEIPT_INVALID", f"{role} target disagrees with its line")


def _validate_optional_admin_regular(
    value: Any,
    *,
    role: str,
    expected_path: str,
    allow_not_applicable: bool = False,
    max_bytes: int = _MAX_GIT_ADMIN_FILE_BYTES,
) -> dict[str, Any]:
    item = _object(value, role=role)
    state = item.get("state")
    if state == "absent":
        _require_exact_keys(item, {"state", "path"}, role=role)
        path = _canonical_absolute_string(item["path"], role=f"{role}.path")
        if path != expected_path:
            _fail("GATE_RECEIPT_INVALID", f"{role} path disagrees")
    elif state == "present":
        _require_exact_keys(item, {"state", "binding"}, role=role)
        _validate_regular_binding(
            item["binding"],
            role=f"{role}.binding",
            max_bytes=max_bytes,
            expected_path=expected_path,
        )
    elif state == "not_applicable" and allow_not_applicable:
        _require_exact_keys(item, {"state", "path"}, role=role)
        if item["path"] is not None:
            _fail("GATE_RECEIPT_INVALID", f"{role} non-applicable path is not null")
    else:
        _fail("GATE_RECEIPT_INVALID", f"{role} state is unsupported")
    return item


def _validate_git_administration(
    value: Any,
    *,
    repository_root: str,
    commit: str,
    index_path: str,
) -> dict[str, Any]:
    admin = _object(value, role="Git administration")
    _require_exact_keys(
        admin,
        {
            "format",
            "retention",
            "hidden_admin_semantics",
            "layout",
            "ref_format",
            "git_dir",
            "git_common_dir",
            "worktree_git_entry",
            "commondir_pointer",
            "gitdir_pointer",
            "config",
            "excludes_and_attributes",
            "head",
            "active_ref",
            "index_path",
        },
        role="Git administration",
    )
    if (
        admin["format"] != "descriptor_retained_git_admin_v3"
        or admin["retention"]
        != "descriptor_rooted_inotify_transient_reject_through_final_observation_v2"
        or admin["layout"] not in {"ordinary_gitdir", "linked_worktree"}
        or admin["ref_format"] != "files"
    ):
        _fail("GATE_RECEIPT_INVALID", "Git administration policy is unsupported")
    hidden_semantics = _object(
        admin["hidden_admin_semantics"],
        role="hidden Git administration semantics",
    )
    _require_exact_keys(
        hidden_semantics,
        {"authority", "live_recheck", "privacy"},
        role="hidden Git administration semantics",
    )
    if hidden_semantics != {
        "authority": "trusted_clean_source_producer_attestation_v1",
        "live_recheck": (
            "require_current_source_recomputes_retained_hidden_bytes_v1"
        ),
        "privacy": (
            "config_keys_values_subsections_and_inactive_ref_names_"
            "not_published_v1"
        ),
    }:
        _fail(
            "GATE_RECEIPT_INVALID",
            "hidden Git administration semantics are unsupported",
        )
    layout = admin["layout"]
    marker_path = str(Path(repository_root) / ".git")
    git_dir_binding = _validate_directory_binding(
        admin["git_dir"], role="Git directory binding"
    )
    common_dir_binding = _validate_directory_binding(
        admin["git_common_dir"], role="Git common-directory binding"
    )
    git_dir = git_dir_binding["path"]
    common_dir = common_dir_binding["path"]

    marker = _object(admin["worktree_git_entry"], role="worktree .git entry")
    _require_exact_keys(
        marker,
        {"kind", "binding", "target", "line"},
        role="worktree .git entry",
    )
    if layout == "ordinary_gitdir":
        if (
            marker["kind"] != "directory"
            or marker["target"] is not None
            or marker["line"] is not None
        ):
            _fail("GATE_RECEIPT_INVALID", "ordinary .git entry is inconsistent")
        marker_binding = _validate_directory_binding(
            marker["binding"],
            role="ordinary .git entry binding",
            expected_path=marker_path,
        )
        if (
            git_dir != marker_path
            or common_dir != git_dir
            or marker_binding != git_dir_binding
            or common_dir_binding != git_dir_binding
        ):
            _fail(
                "GATE_RECEIPT_INVALID",
                "ordinary Git directory identities disagree",
            )
    else:
        if marker["kind"] != "gitfile" or marker["target"] != git_dir:
            _fail("GATE_RECEIPT_INVALID", "linked .git entry is inconsistent")
        marker_binding = _validate_regular_binding(
            marker["binding"],
            role="linked .git marker binding",
            max_bytes=_MAX_GIT_ADMIN_FILE_BYTES,
            expected_path=marker_path,
        )
        _validate_git_pointer_line(
            marker["line"],
            marker_binding,
            role="linked .git marker",
            base=repository_root,
            expected_target=git_dir,
            prefix="gitdir: ",
        )
        git_dir_path = Path(git_dir)
        if (
            git_dir_path.parent.name != "worktrees"
            or str(git_dir_path.parent.parent) != common_dir
            or git_dir == common_dir
        ):
            _fail(
                "GATE_RECEIPT_INVALID",
                "linked Git directory topology is inconsistent",
            )

    commondir = _object(
        admin["commondir_pointer"], role="commondir pointer"
    )
    gitdir_pointer = _object(
        admin["gitdir_pointer"], role="gitdir pointer"
    )
    for role, pointer in (
        ("commondir pointer", commondir),
        ("gitdir pointer", gitdir_pointer),
    ):
        _require_exact_keys(
            pointer,
            {"file", "target", "line"},
            role=role,
        )
    commondir_file = _validate_optional_admin_regular(
        commondir["file"],
        role="commondir pointer file",
        expected_path=str(Path(git_dir) / "commondir"),
    )
    gitdir_file = _validate_optional_admin_regular(
        gitdir_pointer["file"],
        role="gitdir pointer file",
        expected_path=str(Path(git_dir) / "gitdir"),
    )
    if layout == "ordinary_gitdir":
        if (
            commondir_file["state"] != "absent"
            or gitdir_file["state"] != "absent"
            or commondir["target"] is not None
            or gitdir_pointer["target"] is not None
            or commondir["line"] is not None
            or gitdir_pointer["line"] is not None
        ):
            _fail(
                "GATE_RECEIPT_INVALID",
                "ordinary Git pointers are inconsistent",
            )
    elif (
        commondir_file["state"] != "present"
        or gitdir_file["state"] != "present"
        or commondir["target"] != common_dir
        or gitdir_pointer["target"] != marker_path
    ):
        _fail("GATE_RECEIPT_INVALID", "linked Git pointers are inconsistent")
    else:
        _validate_git_pointer_line(
            commondir["line"],
            commondir_file["binding"],
            role="commondir pointer",
            base=git_dir,
            expected_target=common_dir,
        )
        _validate_git_pointer_line(
            gitdir_pointer["line"],
            gitdir_file["binding"],
            role="gitdir pointer",
            base=git_dir,
            expected_target=marker_path,
        )

    config = _object(admin["config"], role="Git configuration")
    _require_exact_keys(
        config,
        {"common", "worktree", "policy"},
        role="Git configuration",
    )
    _validate_regular_binding(
        config["common"],
        role="common Git config",
        max_bytes=_MAX_GIT_CONFIG_RECEIPT_BYTES,
        expected_path=str(Path(common_dir) / "config"),
    )
    _validate_optional_admin_regular(
        config["worktree"],
        role="worktree Git config",
        expected_path=str(Path(git_dir) / "config.worktree"),
        max_bytes=_MAX_GIT_CONFIG_RECEIPT_BYTES,
    )
    config_policy = _object(config["policy"], role="Git config policy")
    _require_exact_keys(
        config_policy,
        {"probe_environment", "producer_check", "privacy"},
        role="Git config policy",
    )
    if config_policy != {
        "probe_environment": (
            "system_global_and_system_attributes_disabled_v1"
        ),
        "producer_check": (
            "include_and_core_excludes_attributes_external_inputs_rejected_v1"
        ),
        "privacy": "opaque_bindings_no_parsed_names_or_values_v1",
    }:
        _fail("GATE_RECEIPT_INVALID", "Git config policy is unsupported")

    exclusions = _object(
        admin["excludes_and_attributes"],
        role="Git excludes and attributes",
    )
    _require_exact_keys(
        exclusions,
        {"info_exclude", "info_attributes"},
        role="Git excludes and attributes",
    )
    _validate_optional_admin_regular(
        exclusions["info_exclude"],
        role="Git info exclude",
        expected_path=str(Path(common_dir) / "info" / "exclude"),
    )
    _validate_optional_admin_regular(
        exclusions["info_attributes"],
        role="Git info attributes",
        expected_path=str(Path(common_dir) / "info" / "attributes"),
    )

    head = _object(admin["head"], role="Git HEAD")
    _require_exact_keys(
        head, {"kind", "ref", "oid", "binding"}, role="Git HEAD"
    )
    head_binding = _validate_regular_binding(
        head["binding"],
        role="Git HEAD binding",
        max_bytes=_MAX_GIT_ADMIN_FILE_BYTES,
        expected_path=str(Path(git_dir) / "HEAD"),
    )
    active = _object(admin["active_ref"], role="Git active ref")
    _require_exact_keys(
        active,
        {
            "storage",
            "name",
            "oid",
            "loose",
            "packed_refs",
        },
        role="Git active ref",
    )
    packed = _validate_optional_admin_regular(
        active["packed_refs"],
        role="Git packed refs",
        expected_path=str(Path(common_dir) / "packed-refs"),
    )
    if packed["state"] == "present" and int(
        packed["binding"]["size_bytes"]
    ) > _MAX_GIT_PACKED_REFS_RECEIPT_BYTES:
        _fail(
            "GATE_RECEIPT_INVALID",
            "Git packed refs exceeds its retained byte bound",
        )
    if head["oid"] != commit or active["oid"] != commit:
        _fail("GATE_RECEIPT_INVALID", "Git HEAD/ref OID disagrees with commit")
    if head["kind"] == "detached":
        if (
            head["ref"] is not None
            or active["storage"] != "detached"
            or active["name"] is not None
        ):
            _fail("GATE_RECEIPT_INVALID", "detached Git HEAD is inconsistent")
        _require_regular_binding_payload(
            head_binding,
            f"{commit}\n".encode("ascii"),
            role="detached Git HEAD",
        )
        loose = _validate_optional_admin_regular(
            active["loose"],
            role="detached loose ref",
            expected_path=str(Path(common_dir) / "refs" / "heads" / "unused"),
            allow_not_applicable=True,
        )
        if loose["state"] != "not_applicable":
            _fail("GATE_RECEIPT_INVALID", "detached Git loose ref is applicable")
    elif head["kind"] == "symbolic":
        ref = head["ref"]
        if (
            type(ref) is not str
            or active["name"] != ref
            or not ref.startswith("refs/heads/")
            or not _valid_git_refname(ref)
        ):
            _fail("GATE_RECEIPT_INVALID", "symbolic Git HEAD is inconsistent")
        try:
            if safe_relative_path(ref).as_posix() != ref:
                raise ImmutableArtifactError("noncanonical ref")
        except ImmutableArtifactError:
            _fail("GATE_RECEIPT_INVALID", "symbolic Git ref path is unsafe")
        _require_regular_binding_payload(
            head_binding,
            f"ref: {ref}\n".encode("utf-8"),
            role="symbolic Git HEAD",
        )
        loose = _validate_optional_admin_regular(
            active["loose"],
            role="Git active loose ref",
            expected_path=str(Path(common_dir).joinpath(*Path(ref).parts)),
        )
        if (
            active["storage"] == "loose"
            and loose["state"] != "present"
        ) or (
            active["storage"] == "packed_refs"
            and (
                loose["state"] != "absent"
                or packed["state"] != "present"
            )
        ):
            _fail("GATE_RECEIPT_INVALID", "Git active-ref storage is inconsistent")
        if active["storage"] == "loose":
            _require_regular_binding_payload(
                loose["binding"],
                f"{commit}\n".encode("ascii"),
                role="Git active loose ref",
            )
        elif active["storage"] == "packed_refs":
            minimum_active_row_bytes = (
                len(commit.encode("ascii"))
                + 1
                + len(ref.encode("utf-8"))
                + 1
            )
            if int(packed["binding"]["size_bytes"]) < minimum_active_row_bytes:
                _fail(
                    "GATE_RECEIPT_INVALID",
                    "Git packed refs is too short to contain its active row",
                )
        if active["storage"] not in {"loose", "packed_refs"}:
            _fail("GATE_RECEIPT_INVALID", "Git active-ref storage is unsupported")
    else:
        _fail("GATE_RECEIPT_INVALID", "Git HEAD kind is unsupported")

    observed_index = _canonical_absolute_string(
        admin["index_path"], role="Git administration index path"
    )
    if observed_index != index_path or observed_index != str(Path(git_dir) / "index"):
        _fail("GATE_RECEIPT_INVALID", "Git administration index path disagrees")
    return admin


def _validate_stream(value: Any, *, role: str, expected_path: str) -> dict[str, Any]:
    stream = _object(value, role=role)
    _require_exact_keys(stream, {"path", "size_bytes", "sha256"}, role=role)
    if (
        stream["path"] != expected_path
        or type(stream["size_bytes"]) is not int
        or not 0 <= stream["size_bytes"] <= _MAX_STREAM_BYTES
        or not isinstance(stream["sha256"], str)
        or not _SHA256_RE.fullmatch(stream["sha256"])
    ):
        _fail("GATE_RECEIPT_INVALID", f"{role} identity is malformed")
    return stream


def _validate_document(document: Any) -> dict[str, Any]:
    root = _object(document, role="gate receipt")
    if root.get("schema_version") != SCHEMA_VERSION:
        _fail(
            "GATE_RECEIPT_VERSION_UNSUPPORTED",
            "gate receipt schema version is unsupported",
        )
    _require_exact_keys(
        root,
        {
            "schema_version",
            "integrity",
            "command",
            "executable",
            "environment",
            "supervision",
            "temporary_root",
            "publication",
            "source",
            "time",
            "wait",
            "resource_usage",
            "streams",
            "outcome",
            "receipt_sha256",
        },
        role="gate receipt",
    )
    try:
        verify_model_seal(root, field="receipt_sha256")
    except ImmutableArtifactError as exc:
        _fail("GATE_RECEIPT_SEAL_MISMATCH", "gate receipt seal is invalid", error=str(exc))

    integrity = _object(root["integrity"], role="receipt integrity")
    _require_exact_keys(
        integrity,
        {"seal", "authority"},
        role="receipt integrity",
    )
    if integrity != {
        "seal": "canonical_json_unkeyed_sha256_consistency_v1",
        "authority": (
            "requires_external_content_manifest_sha256_anchor_and_"
            "trusted_clean_source_producer_v1"
        ),
    }:
        _fail("GATE_RECEIPT_INVALID", "gate receipt integrity scope is unsupported")

    command = _object(root["command"], role="command")
    _require_exact_keys(command, {"argv", "cwd"}, role="command")
    if not isinstance(command["argv"], list) or not command["argv"] or any(type(item) is not str for item in command["argv"]) or type(command["cwd"]) is not str:
        _fail("GATE_RECEIPT_INVALID", "gate command is malformed")
    if (
        sum(len(item.encode("utf-8")) for item in command["argv"]) > _MAX_ARG_BYTES
        or any("\x00" in item for item in command["argv"])
    ):
        _fail("GATE_RECEIPT_INVALID", "gate command exceeds its byte or character limits")
    _canonical_absolute_string(command["argv"][0], role="command executable")
    _canonical_absolute_string(command["cwd"], role="command working directory")

    executable = _object(root["executable"], role="executable")
    _require_exact_keys(
        executable,
        {
            "requested_path", "resolved_path", "process_argv0", "sha256",
            "size_bytes", "mode", "requested_binding",
        },
        role="executable",
    )
    if executable["process_argv0"] != command["argv"][0]:
        _fail("GATE_RECEIPT_INVALID", "recorded process argv[0] disagrees with command")
    _canonical_absolute_string(executable["requested_path"], role="requested executable")
    _canonical_absolute_string(executable["resolved_path"], role="resolved executable")
    _canonical_absolute_string(executable["process_argv0"], role="process argv[0]")
    if executable["requested_path"] != executable["resolved_path"]:
        _fail("GATE_RECEIPT_INVALID", "requested executable is not its canonical retained target")
    if type(executable["sha256"]) is not str or not _SHA256_RE.fullmatch(executable["sha256"]):
        _fail("GATE_RECEIPT_INVALID", "executable SHA-256 is malformed")
    executable_size = _decimal_u64(
        executable["size_bytes"], role="executable.size_bytes"
    )
    executable_mode = _decimal_u64(executable["mode"], role="executable.mode")
    binding = _object(executable["requested_binding"], role="requested executable binding")
    _require_exact_keys(
        binding,
        {
            "path", "device", "inode", "mode", "link_count", "size_bytes",
            "modified_ns", "changed_ns", "symlink_target",
        },
        role="requested executable binding",
    )
    if binding["path"] != executable["requested_path"]:
        _fail("GATE_RECEIPT_INVALID", "requested executable binding path disagrees")
    _canonical_absolute_string(binding["path"], role="requested executable binding path")
    binding_values = {
        field: _decimal_u64(
            binding[field], role=f"requested executable binding.{field}"
        )
        for field in (
            "device", "inode", "mode", "link_count", "size_bytes",
            "modified_ns", "changed_ns",
        )
    }
    if (
        binding["symlink_target"] is not None
        or binding_values["inode"] == 0
        or binding_values["size_bytes"] == 0
        or binding["path"] != executable["resolved_path"]
        or not stat.S_ISREG(binding_values["mode"])
        or binding_values["mode"] & 0o111 == 0
        or binding_values["link_count"] != 1
        or executable_size != binding_values["size_bytes"]
        or executable_mode != binding_values["mode"]
    ):
        _fail("GATE_RECEIPT_INVALID", "retained executable binding is not producer-admissible")

    environment = _object(root["environment"], role="environment")
    if any(
        type(key) is not str
        or type(value) is not str
        or "\x00" in value
        for key, value in environment.items()
    ):
        _fail("GATE_RECEIPT_INVALID", "gate environment must contain NUL-free string pairs")
    for name in environment:
        _validate_env_name(name)
    if sum(
        len(key.encode("utf-8")) + len(value.encode("utf-8"))
        for key, value in environment.items()
    ) > _MAX_ENV_BYTES:
        _fail("GATE_RECEIPT_INVALID", "gate environment exceeds its byte limit")
    if list(environment) != sorted(environment):
        _fail("GATE_RECEIPT_INVALID", "gate environment is not canonically ordered")
    allowed_recorded_environment = _ALLOWED_ENVIRONMENT | {
        "TMP", "TEMP", "TMPDIR", "CARGO_TARGET_DIR",
        "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE",
    }
    if not set(environment).issubset(allowed_recorded_environment):
        _fail("GATE_RECEIPT_INVALID", "gate environment exceeds its audited allowlist")
    supervision = _object(root["supervision"], role="supervision")
    supervision_dynamic_keys = {
        "requested_timeout_ns", "monotonic_deadline_ns",
        "monotonic_decision_ns", "decision",
    }
    _require_exact_keys(
        supervision,
        set(_SUPERVISION_POLICY) | supervision_dynamic_keys,
        role="supervision",
    )
    if any(supervision.get(key) != value for key, value in _SUPERVISION_POLICY.items()):
        _fail("GATE_RECEIPT_INVALID", "gate supervision policy is unsupported")
    requested_timeout_ns = _u64(
        supervision["requested_timeout_ns"],
        role="supervision.requested_timeout_ns",
    )
    if not 1 <= requested_timeout_ns <= _MAX_TIMEOUT_NS:
        _fail("GATE_RECEIPT_INVALID", "recorded gate timeout is outside the admitted range")
    monotonic_deadline_ns = _decimal_u64(
        supervision["monotonic_deadline_ns"],
        role="supervision.monotonic_deadline_ns",
    )
    monotonic_decision_ns = _decimal_u64(
        supervision["monotonic_decision_ns"],
        role="supervision.monotonic_decision_ns",
    )
    if supervision["decision"] not in {"terminal", "timeout"}:
        _fail("GATE_RECEIPT_INVALID", "gate supervision decision is unsupported")
    temporary = _object(root["temporary_root"], role="temporary root")
    _require_exact_keys(
        temporary,
        {"path", "binding", "children", "policy"},
        role="temporary root",
    )
    temporary_root = _canonical_absolute_string(
        temporary["path"], role="temporary root path"
    )
    if Path(temporary_root).parent == Path(temporary_root):
        _fail(
            "GATE_RECEIPT_INVALID",
            "temporary root cannot be a filesystem root",
        )
    temporary_binding = _validate_directory_binding(
        temporary["binding"],
        role="temporary root binding",
        expected_path=temporary_root,
    )
    if stat.S_IMODE(int(temporary_binding["mode"])) != 0o700:
        _fail(
            "GATE_RECEIPT_INVALID",
            "temporary root binding does not record producer-owned mode",
        )
    if temporary["children"] != ["cargo-target", "pycache", "tmp"]:
        _fail(
            "GATE_RECEIPT_INVALID",
            "temporary root child inventory is unsupported",
        )
    if temporary["policy"] != "mkdirat_retained_parent_monitored_path_v1":
        _fail("GATE_RECEIPT_INVALID", "temporary root policy is unsupported")
    expected_fixed_environment = {
        "TMP": f"{temporary_root}/tmp",
        "TEMP": f"{temporary_root}/tmp",
        "TMPDIR": f"{temporary_root}/tmp",
        "CARGO_TARGET_DIR": f"{temporary_root}/cargo-target",
        "PYTHONPYCACHEPREFIX": f"{temporary_root}/pycache",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, expected in expected_fixed_environment.items():
        if environment.get(name) != expected:
            _fail(
                "GATE_RECEIPT_INVALID",
                "gate-owned environment path is inconsistent",
                name=name,
            )

    publication = _object(root["publication"], role="publication")
    _require_exact_keys(
        publication,
        {"output_root", "generation", "root_binding", "binding"},
        role="publication",
    )
    output_root = _canonical_absolute_string(
        publication["output_root"], role="publication output root"
    )
    if Path(output_root).parent == Path(output_root):
        _fail(
            "GATE_RECEIPT_INVALID",
            "publication output root cannot be a filesystem root",
        )
    if (
        type(publication["generation"]) is not str
        or not _GENERATION_RE.fullmatch(publication["generation"])
    ):
        _fail("GATE_RECEIPT_INVALID", "publication generation is malformed")
    _validate_directory_binding(
        publication["root_binding"],
        role="publication root binding",
        expected_path=output_root,
    )
    if (
        publication["binding"]
        != "pinned_directory_reader_descriptor_relative_v1"
    ):
        _fail("GATE_RECEIPT_INVALID", "publication binding is unsupported")

    source = _object(root["source"], role="source")
    _require_exact_keys(
        source,
        {
            "cwd_binding",
            "git",
            "git_index",
            "tracked_inputs",
            "checked_out_worktree",
            "git_administration",
        },
        role="source",
    )
    _validate_directory_binding(
        source["cwd_binding"],
        role="working directory binding",
        expected_path=command["cwd"],
    )
    git_index = _validate_regular_binding(
        source["git_index"],
        role="Git index binding",
        max_bytes=_MAX_GIT_INDEX_BYTES,
    )
    if int(git_index["size_bytes"]) == 0:
        _fail("GATE_RECEIPT_INVALID", "Git index binding is empty")

    tracked = _object(source["tracked_inputs"], role="tracked inputs")
    _require_exact_keys(
        tracked,
        {
            "format",
            "manifest_sha256",
            "flag_format",
            "flag_manifest_sha256",
            "flag_manifest_bytes",
            "entry_count",
            "relative_path_bytes",
            "maximum_relative_path_bytes",
            "mode_counts",
            "monitor",
            "execution_scope",
            "limits",
        },
        role="tracked inputs",
    )
    if tracked["format"] != "git_ls_files_stage0_regular_v1":
        _fail("GATE_RECEIPT_INVALID", "tracked input format is unsupported")
    if (
        type(tracked["manifest_sha256"]) is not str
        or not _SHA256_RE.fullmatch(tracked["manifest_sha256"])
    ):
        _fail("GATE_RECEIPT_INVALID", "tracked input manifest hash is malformed")
    if (
        tracked["flag_format"] != "git_ls_files_v_normal_only_v1"
        or type(tracked["flag_manifest_sha256"]) is not str
        or not _SHA256_RE.fullmatch(tracked["flag_manifest_sha256"])
    ):
        _fail("GATE_RECEIPT_INVALID", "tracked input flag identity is malformed")
    flag_manifest_bytes = _u64(
        tracked["flag_manifest_bytes"],
        role="tracked_inputs.flag_manifest_bytes",
    )
    entry_count = _u64(
        tracked["entry_count"], role="tracked_inputs.entry_count"
    )
    relative_path_bytes = _u64(
        tracked["relative_path_bytes"],
        role="tracked_inputs.relative_path_bytes",
    )
    maximum_relative_path_bytes = _u64(
        tracked["maximum_relative_path_bytes"],
        role="tracked_inputs.maximum_relative_path_bytes",
    )
    if (
        not 1 <= entry_count <= _MAX_TRACKED_ENTRIES
        or not entry_count
        <= relative_path_bytes
        <= _MAX_TRACKED_RELATIVE_PATH_BYTES
        or not 1
        <= maximum_relative_path_bytes
        <= _MAX_TRACKED_PATH_BYTES
        or maximum_relative_path_bytes > relative_path_bytes
        or relative_path_bytes > entry_count * maximum_relative_path_bytes
        or flag_manifest_bytes != relative_path_bytes + 3 * entry_count
        or flag_manifest_bytes > _MAX_TRACKED_MANIFEST_BYTES
    ):
        _fail("GATE_RECEIPT_INVALID", "tracked input bounds are inconsistent")
    mode_counts = _object(tracked["mode_counts"], role="tracked input modes")
    _require_exact_keys(
        mode_counts, {"100644", "100755"}, role="tracked input modes"
    )
    regular_count = _u64(mode_counts["100644"], role="tracked mode 100644")
    executable_count = _u64(
        mode_counts["100755"], role="tracked mode 100755"
    )
    if regular_count + executable_count != entry_count:
        _fail("GATE_RECEIPT_INVALID", "tracked input mode counts disagree")
    if tracked["monitor"] != "pinned_path_set_inotify_v1":
        _fail("GATE_RECEIPT_INVALID", "tracked input monitor is unsupported")
    if (
        tracked["execution_scope"]
        != "real_checkout_tracked_files_monitored_untracked_and_ignored_siblings_mutable_v1"
    ):
        _fail("GATE_RECEIPT_INVALID", "tracked input scope is unsupported")
    limits = _object(tracked["limits"], role="tracked input limits")
    expected_limits = {
        "max_entries": _MAX_TRACKED_ENTRIES,
        "max_manifest_bytes": _MAX_TRACKED_MANIFEST_BYTES,
        "max_relative_path_bytes": _MAX_TRACKED_RELATIVE_PATH_BYTES,
        "max_path_bytes": _MAX_TRACKED_PATH_BYTES,
        "max_index_bytes": _MAX_GIT_INDEX_BYTES,
    }
    _require_exact_keys(limits, set(expected_limits), role="tracked input limits")
    if limits != expected_limits:
        _fail("GATE_RECEIPT_INVALID", "tracked input limits are unsupported")

    worktree = _object(
        source["checked_out_worktree"],
        role="checked-out worktree",
    )
    _require_exact_keys(
        worktree,
        {
            "format",
            "manifest_sha256",
            "manifest_bytes",
            "entry_count",
            "relative_path_bytes",
            "directory_entry_count",
            "directory_relative_path_bytes",
            "maximum_directory_path_bytes",
            "total_file_bytes",
            "maximum_file_bytes",
            "executable_entry_count",
            "git_mode_contract",
            "directory_mode_contract",
            "limits",
        },
        role="checked-out worktree",
    )
    if (
        worktree["format"]
        != "descriptor_relative_checked_out_regular_and_directory_sha256_v1"
        or type(worktree["manifest_sha256"]) is not str
        or not _SHA256_RE.fullmatch(worktree["manifest_sha256"])
        or worktree["git_mode_contract"]
        != "actual_owner_exec_bit_matches_stage0_git_mode_v1"
        or worktree["directory_mode_contract"]
        != "tracked_ancestor_and_repository_root_modes_bound_v1"
    ):
        _fail(
            "GATE_RECEIPT_INVALID",
            "checked-out worktree identity is malformed",
        )
    worktree_entry_count = _u64(
        worktree["entry_count"], role="checked_out_worktree.entry_count"
    )
    worktree_relative_path_bytes = _u64(
        worktree["relative_path_bytes"],
        role="checked_out_worktree.relative_path_bytes",
    )
    directory_entry_count = _u64(
        worktree["directory_entry_count"],
        role="checked_out_worktree.directory_entry_count",
    )
    directory_relative_path_bytes = _u64(
        worktree["directory_relative_path_bytes"],
        role="checked_out_worktree.directory_relative_path_bytes",
    )
    maximum_directory_path_bytes = _u64(
        worktree["maximum_directory_path_bytes"],
        role="checked_out_worktree.maximum_directory_path_bytes",
    )
    worktree_manifest_bytes = _u64(
        worktree["manifest_bytes"],
        role="checked_out_worktree.manifest_bytes",
    )
    total_file_bytes = _u64(
        worktree["total_file_bytes"],
        role="checked_out_worktree.total_file_bytes",
    )
    maximum_file_bytes = _u64(
        worktree["maximum_file_bytes"],
        role="checked_out_worktree.maximum_file_bytes",
    )
    worktree_executable_count = _u64(
        worktree["executable_entry_count"],
        role="checked_out_worktree.executable_entry_count",
    )
    worktree_manifest_header_bytes = len(
        b"gate_checked_out_worktree_files_and_directories_sha256_v1\0"
    ) + 16
    if (
        worktree_entry_count != entry_count
        or worktree_relative_path_bytes != relative_path_bytes
        or worktree_executable_count != executable_count
        or not 1 <= directory_entry_count <= _MAX_TRACKED_DIRECTORIES
        or not directory_entry_count
        <= directory_relative_path_bytes
        <= _MAX_TRACKED_DIRECTORY_PATH_BYTES
        or not 1
        <= maximum_directory_path_bytes
        <= _MAX_TRACKED_PATH_BYTES
        or maximum_directory_path_bytes > directory_relative_path_bytes
        or directory_relative_path_bytes
        > directory_entry_count * maximum_directory_path_bytes
        or worktree_manifest_bytes
        != worktree_manifest_header_bytes
        + directory_relative_path_bytes
        + directory_entry_count * (8 + 8)
        + relative_path_bytes
        + entry_count * (8 + 6 + 8 + 8 + 32)
        or worktree_manifest_bytes > _MAX_TRACKED_WORKTREE_MANIFEST_BYTES
        or total_file_bytes > _MAX_TRACKED_TOTAL_FILE_BYTES
        or maximum_file_bytes > _MAX_TRACKED_FILE_BYTES
        or maximum_file_bytes > total_file_bytes
        or total_file_bytes > entry_count * maximum_file_bytes
    ):
        _fail(
            "GATE_RECEIPT_INVALID",
            "checked-out worktree bounds are inconsistent",
        )
    worktree_limits = _object(
        worktree["limits"], role="checked-out worktree limits"
    )
    expected_worktree_limits = {
        "max_file_bytes": _MAX_TRACKED_FILE_BYTES,
        "max_total_file_bytes": _MAX_TRACKED_TOTAL_FILE_BYTES,
        "max_manifest_bytes": _MAX_TRACKED_WORKTREE_MANIFEST_BYTES,
        "max_directories": _MAX_TRACKED_DIRECTORIES,
        "max_directory_relative_path_bytes": (
            _MAX_TRACKED_DIRECTORY_PATH_BYTES
        ),
        "max_path_bytes": _MAX_TRACKED_PATH_BYTES,
    }
    _require_exact_keys(
        worktree_limits,
        set(expected_worktree_limits),
        role="checked-out worktree limits",
    )
    if worktree_limits != expected_worktree_limits:
        _fail(
            "GATE_RECEIPT_INVALID",
            "checked-out worktree limits are unsupported",
        )

    git = _object(source["git"], role="Git source identity")
    expected_git_keys = {
        "repository_root", "commit", "tree", "branch", "status_porcelain_v1",
        "status_sha256", "clean", "git_executable_path", "git_executable_sha256",
    }
    _require_exact_keys(git, expected_git_keys, role="Git source identity")
    repository_root = _canonical_absolute_string(
        git["repository_root"], role="Git repository root"
    )
    try:
        Path(command["cwd"]).relative_to(repository_root)
    except ValueError:
        _fail("GATE_RECEIPT_INVALID", "gate working directory is outside its Git root")
    branch = git["branch"]
    if not _valid_git_branch_text(branch):
        _fail("GATE_RECEIPT_INVALID", "Git branch is malformed")
    _canonical_absolute_string(git["git_executable_path"], role="Git executable path")
    if type(git["git_executable_sha256"]) is not str or not _SHA256_RE.fullmatch(git["git_executable_sha256"]):
        _fail("GATE_RECEIPT_INVALID", "Git executable SHA-256 is malformed")
    status = git["status_porcelain_v1"]
    if (
        type(status) is not str
        or len(status.encode("utf-8")) > _MAX_STATUS_BYTES
        or type(git["status_sha256"]) is not str
        or not _SHA256_RE.fullmatch(git["status_sha256"])
        or hashlib.sha256(status.encode()).hexdigest() != git["status_sha256"]
        or git["clean"] is not (status == "")
    ):
        _fail("GATE_RECEIPT_INVALID", "Git status identity is inconsistent")
    if not git["clean"]:
        _fail("GATE_RECEIPT_INVALID", "gate receipt records a dirty Git source")
    for field in ("commit", "tree"):
        if (
            type(git[field]) is not str
            or len(git[field]) not in {40, 64}
            or not re.fullmatch(r"[0-9a-f]+", git[field])
        ):
            _fail("GATE_RECEIPT_INVALID", f"Git {field} is malformed")
    if len(git["commit"]) != len(git["tree"]):
        _fail(
            "GATE_RECEIPT_INVALID",
            "Git commit and tree object formats disagree",
        )
    validated_git_admin = _validate_git_administration(
        source["git_administration"],
        repository_root=repository_root,
        commit=git["commit"],
        index_path=git_index["path"],
    )
    retained_head = validated_git_admin["head"]
    expected_branch = (
        None
        if retained_head["kind"] == "detached"
        else retained_head["ref"].removeprefix("refs/heads/")
    )
    if branch != expected_branch:
        _fail(
            "GATE_RECEIPT_INVALID",
            "Git branch disagrees with retained HEAD",
        )
    minimum_manifest_bytes = relative_path_bytes + entry_count * (
        11 + len(git["commit"])
    )
    if minimum_manifest_bytes > _MAX_TRACKED_MANIFEST_BYTES:
        _fail(
            "GATE_RECEIPT_INVALID",
            "tracked input inventory cannot fit its manifest byte bound",
        )
    temporary_path = Path(temporary_root)
    publication_path = Path(output_root)
    repository_path = Path(repository_root)
    git_directory_path = Path(validated_git_admin["git_dir"]["path"])
    git_common_directory_path = Path(
        validated_git_admin["git_common_dir"]["path"]
    )
    executable_path = Path(executable["resolved_path"])
    git_index_path = Path(git_index["path"])
    _require_disjoint_path(
        temporary_path,
        role="temporary root",
        others=(
            ("publication output root", publication_path),
            ("Git repository root", repository_path),
            ("Git directory", git_directory_path),
            ("Git common directory", git_common_directory_path),
            ("retained executable", executable_path),
            ("Git index", git_index_path),
        ),
        code="GATE_RECEIPT_INVALID",
    )
    _require_disjoint_path(
        publication_path,
        role="publication output root",
        others=(
            ("Git repository root", repository_path),
            ("Git directory", git_directory_path),
            ("Git common directory", git_common_directory_path),
            ("retained executable", executable_path),
            ("Git index", git_index_path),
        ),
        code="GATE_RECEIPT_INVALID",
    )

    timing = _object(root["time"], role="time")
    _require_exact_keys(timing, {"started_at_unix_ns", "finished_at_unix_ns", "monotonic_start_ns", "monotonic_stop_ns", "duration_ns"}, role="time")
    started_at_unix_ns = _decimal_u64(
        timing["started_at_unix_ns"], role="time.started_at_unix_ns"
    )
    finished_at_unix_ns = _decimal_u64(
        timing["finished_at_unix_ns"], role="time.finished_at_unix_ns"
    )
    monotonic_start_ns = _decimal_u64(
        timing["monotonic_start_ns"], role="time.monotonic_start_ns"
    )
    monotonic_stop_ns = _decimal_u64(
        timing["monotonic_stop_ns"], role="time.monotonic_stop_ns"
    )
    duration_ns = _u64(timing["duration_ns"], role="time.duration_ns")
    if (
        started_at_unix_ns == 0
        or finished_at_unix_ns == 0
        or monotonic_start_ns == 0
        or monotonic_stop_ns == 0
    ):
        _fail("GATE_RECEIPT_INVALID", "gate timestamps must be nonzero")
    if (
        monotonic_stop_ns < monotonic_start_ns
        or duration_ns != monotonic_stop_ns - monotonic_start_ns
        or monotonic_deadline_ns != monotonic_start_ns + requested_timeout_ns
        or not monotonic_start_ns <= monotonic_decision_ns <= monotonic_stop_ns
        or (
            supervision["decision"] == "terminal"
            and monotonic_decision_ns > monotonic_deadline_ns
        )
        or (
            supervision["decision"] == "timeout"
            and monotonic_decision_ns < monotonic_deadline_ns
        )
    ):
        _fail("GATE_RECEIPT_INVALID", "gate timing is inconsistent")

    wait = _object(root["wait"], role="wait")
    _require_exact_keys(
        wait,
        {
            "pid", "waited_pid", "process_group_id", "session_id",
            "raw_wait_status", "return_code", "exit_code", "signal",
            "timed_out",
        },
        role="wait",
    )
    for field in ("pid", "waited_pid", "process_group_id", "session_id"):
        value = _u64(wait[field], role=f"wait.{field}")
        if value == 0:
            _fail("GATE_RECEIPT_INVALID", f"wait.{field} must be nonzero")
    if not (
        wait["pid"] == wait["waited_pid"]
        == wait["process_group_id"]
        == wait["session_id"]
    ):
        _fail("GATE_RECEIPT_INVALID", "gate child, wait, process-group, and session identities disagree")
    raw_wait_status = _u64(
        wait["raw_wait_status"], role="wait.raw_wait_status"
    )
    if raw_wait_status > 0xFFFF:
        _fail("GATE_RECEIPT_INVALID", "raw wait status exceeds Linux bounds")
    if (
        type(wait["return_code"]) is not int
        or wait["return_code"] < -255
        or wait["return_code"] > 255
        or type(wait["timed_out"]) is not bool
        or (wait["exit_code"] is not None and (type(wait["exit_code"]) is not int or not 0 <= wait["exit_code"] <= 255))
        or (wait["signal"] is not None and (type(wait["signal"]) is not int or not 1 <= wait["signal"] <= 255))
    ):
        _fail("GATE_RECEIPT_INVALID", "gate wait result is malformed")
    if (supervision["decision"] == "timeout") is not wait["timed_out"]:
        _fail("GATE_RECEIPT_INVALID", "supervision decision disagrees with timeout status")
    if not (os.WIFEXITED(wait["raw_wait_status"]) or os.WIFSIGNALED(wait["raw_wait_status"])):
        _fail("GATE_RECEIPT_INVALID", "raw wait status is not terminal")
    try:
        projected_return_code = os.waitstatus_to_exitcode(wait["raw_wait_status"])
    except ValueError:
        _fail("GATE_RECEIPT_INVALID", "raw wait status cannot be projected")
    if projected_return_code != wait["return_code"]:
        _fail("GATE_RECEIPT_INVALID", "raw wait status disagrees with return code")
    expected_exit = os.WEXITSTATUS(wait["raw_wait_status"]) if os.WIFEXITED(wait["raw_wait_status"]) else None
    expected_signal = os.WTERMSIG(wait["raw_wait_status"]) if os.WIFSIGNALED(wait["raw_wait_status"]) else None
    if wait["exit_code"] != expected_exit or wait["signal"] != expected_signal:
        _fail("GATE_RECEIPT_INVALID", "raw wait status projection is inconsistent")

    usage = _object(root["resource_usage"], role="resource usage")
    _require_exact_keys(
        usage,
        {
            "user_cpu_ns", "system_cpu_ns", "peak_rss_kib", "minor_faults",
            "major_faults", "voluntary_context_switches",
            "involuntary_context_switches",
        },
        role="resource usage",
    )
    for field in usage:
        _u64(usage[field], role=f"resource_usage.{field}")

    streams = _object(root["streams"], role="streams")
    _require_exact_keys(streams, {"stdout", "stderr"}, role="streams")
    _validate_stream(streams["stdout"], role="stdout", expected_path=STDOUT_NAME)
    _validate_stream(streams["stderr"], role="stderr", expected_path=STDERR_NAME)

    outcome = _object(root["outcome"], role="outcome")
    _require_exact_keys(outcome, {"status", "reason"}, role="outcome")
    if wait["timed_out"]:
        expected_outcome = ("failed", "TIMEOUT")
    elif wait["return_code"] == 0:
        expected_outcome = ("passed", None)
    elif wait["return_code"] < 0:
        expected_outcome = ("failed", "SIGNAL")
    else:
        expected_outcome = ("failed", "NONZERO_EXIT")
    if (outcome["status"], outcome["reason"]) != expected_outcome:
        _fail("GATE_RECEIPT_INVALID", "gate outcome disagrees with wait status")
    return root


def _validate_receipt_bundle(
    *,
    receipt_payload: bytes,
    stdout_payload: bytes,
    stderr_payload: bytes,
    generation_path: Path,
) -> dict[str, Any]:
    try:
        document = strict_json_loads(receipt_payload, label="gate receipt")
    except (ImmutableArtifactError, ValueError) as exc:
        _fail("GATE_RECEIPT_INVALID", "gate receipt JSON is invalid", error=str(exc))
    root = _validate_document(document)
    if pretty_json_bytes(root) != receipt_payload:
        _fail("GATE_RECEIPT_INVALID", "gate receipt bytes are not canonical")
    expected_generation_path = (
        Path(root["publication"]["output_root"])
        / root["publication"]["generation"]
    )
    if expected_generation_path != generation_path:
        _fail(
            "GATE_GENERATION_PATH_MISMATCH",
            "gate generation path disagrees with its receipt",
        )
    publication_root_path = Path(root["publication"]["output_root"])
    try:
        if protected_paths.is_inside_git_worktree(publication_root_path):
            _fail(
                "GATE_PUBLICATION_ROOT_MISMATCH",
                "gate publication root is inside a Git worktree",
            )
        with PinnedDirectoryReader(publication_root_path) as publication_reader:
            current_publication_binding = _directory_binding(
                publication_root_path,
                descriptor=publication_reader.descriptor,
            )
    except protected_paths.ProtectedPathError as exc:
        _fail(
            "GATE_PUBLICATION_ROOT_MISMATCH",
            "gate publication root containment is indeterminate",
            error=str(exc),
        )
    except ImmutableArtifactError as exc:
        _fail(
            "GATE_PUBLICATION_ROOT_MISMATCH",
            "gate publication root could not be retained",
            error=str(exc),
        )
    if current_publication_binding != root["publication"]["root_binding"]:
        _fail(
            "GATE_PUBLICATION_ROOT_MISMATCH",
            "gate publication root disagrees with its receipt",
        )
    for role, payload in (("stdout", stdout_payload), ("stderr", stderr_payload)):
        identity = root["streams"][role]
        if (
            len(payload) != identity["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != identity["sha256"]
        ):
            _fail("GATE_STREAM_IDENTITY_MISMATCH", f"gate {role} bytes disagree with receipt")
    return root


def _current_source_document(work: Path) -> dict[str, Any]:
    try:
        with PinnedDirectoryReader(work) as work_reader:
            initial_git = _git_snapshot(
                work, cwd_descriptor=work_reader.descriptor
            )
            repository_root = Path(initial_git["repository_root"])
            with PinnedDirectoryReader(repository_root) as repository_reader:
                work_reader.verify_path()
                require_file_descriptor_headroom(
                    128, label="current gate source verification"
                )
                initial_tracked = _tracked_input_manifest(
                    work, cwd_descriptor=work_reader.descriptor
                )
                index_path = _git_index_path(
                    work, cwd_descriptor=work_reader.descriptor
                )
                with open_retained_regular(
                    index_path, max_bytes=_MAX_GIT_INDEX_BYTES
                ) as retained_index, PinnedPathSetMonitor.from_reader(
                    repository_reader,
                    regular_paths=initial_tracked.paths,
                    recursive_directories=(),
                ) as source_monitor, _retain_git_administration(
                    work=work,
                    work_reader=work_reader,
                    repository_root=repository_root,
                    repository_reader=repository_reader,
                ) as git_administration:
                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    tracked = _tracked_input_manifest(
                        work, cwd_descriptor=work_reader.descriptor
                    )
                    worktree = _tracked_worktree_manifest(
                        repository_reader,
                        tracked,
                    )
                    git = _git_snapshot(
                        work, cwd_descriptor=work_reader.descriptor
                    )
                    git_administration_document = (
                        git_administration.document(commit=git["commit"])
                    )
                    if Path(
                        git_administration_document["index_path"]
                    ) != index_path:
                        _fail(
                            "GATE_CURRENT_SOURCE_MISMATCH",
                            "current Git index path disagrees with administration",
                        )
                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    if tracked != initial_tracked or git != initial_git:
                        _fail(
                            "GATE_CURRENT_SOURCE_MISMATCH",
                            "current source changed during replay verification",
                        )
                    result = {
                        "cwd_binding": _directory_binding(
                            work, descriptor=work_reader.descriptor
                        ),
                        "git": git,
                        "git_index": _retained_regular_identity(
                            retained_index, path=index_path
                        ),
                        "tracked_inputs": tracked.document,
                        "checked_out_worktree": worktree,
                        "git_administration": git_administration_document,
                    }
                    source_monitor.verify_unchanged()
                    retained_index.verify_unchanged()
                    return result
    except ImmutableArtifactError as exc:
        _fail(
            "GATE_CURRENT_SOURCE_MISMATCH",
            "current source identity could not be retained",
            error=str(exc),
        )


def verify_generation(
    path: str | Path,
    *,
    require_current_source: bool = False,
    expected_content_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_content_manifest_sha256 is not None and (
        type(expected_content_manifest_sha256) is not str
        or not _SHA256_RE.fullmatch(expected_content_manifest_sha256)
    ):
        _fail(
            "GATE_CONTENT_MANIFEST_ANCHOR_INVALID",
            "expected content-manifest SHA-256 is malformed",
        )
    generation_path = _canonical_path(path, role="gate receipt generation", directory=True)
    generation_limits = {
        "max_files": 3,
        "max_directories": 1,
        "max_total_file_bytes": 2 * _MAX_STREAM_BYTES + _MAX_RECEIPT_BYTES,
        "max_total_relative_path_bytes": 256,
    }
    file_limits = {
        STDOUT_NAME: _MAX_STREAM_BYTES,
        STDERR_NAME: _MAX_STREAM_BYTES,
        RECEIPT_NAME: _MAX_RECEIPT_BYTES,
    }
    with PinnedDirectoryReader(generation_path) as reader:
        with reader.retain_tree_generation(
            require_readonly_files=True,
            require_readonly_directories=True,
            **generation_limits,
        ):
            inspected = inspect_immutable_directory_generation(
                generation_path,
                max_file_bytes_by_path=file_limits,
                **generation_limits,
            )
            observed_paths = tuple(item.relative_path for item in inspected.files)
            expected_paths = tuple(sorted((RECEIPT_NAME, STDERR_NAME, STDOUT_NAME)))
            if observed_paths != expected_paths:
                _fail(
                    "GATE_GENERATION_INVENTORY_MISMATCH",
                    "gate generation file inventory is not exact",
                    observed=list(observed_paths),
                )
            receipt_payload = reader.read(RECEIPT_NAME, max_bytes=_MAX_RECEIPT_BYTES)
            stdout_payload = reader.read(STDOUT_NAME, max_bytes=_MAX_STREAM_BYTES)
            stderr_payload = reader.read(STDERR_NAME, max_bytes=_MAX_STREAM_BYTES)
            root = _validate_receipt_bundle(
                receipt_payload=receipt_payload,
                stdout_payload=stdout_payload,
                stderr_payload=stderr_payload,
                generation_path=generation_path,
            )
            retained_payloads = {
                RECEIPT_NAME: receipt_payload,
                STDOUT_NAME: stdout_payload,
                STDERR_NAME: stderr_payload,
            }
            expected_files = tuple(
                DirectoryGenerationFile(
                    relative_path=name,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
                for name, payload in sorted(retained_payloads.items())
            )
            if inspected.files != expected_files:
                _fail(
                    "GATE_GENERATION_IDENTITY_MISMATCH",
                    "gate generation identities disagree with retained bytes",
                )
            reinspected = inspect_immutable_directory_generation(
                generation_path,
                max_file_bytes_by_path=file_limits,
                **generation_limits,
            )
            if reinspected != inspected:
                _fail(
                    "GATE_GENERATION_DRIFT",
                    "gate generation changed during replay verification",
                )
    anchor_status = "unanchored"
    if expected_content_manifest_sha256 is not None:
        if inspected.content_manifest_sha256 != expected_content_manifest_sha256:
            _fail(
                "GATE_CONTENT_MANIFEST_ANCHOR_MISMATCH",
                "gate generation disagrees with the expected content-manifest SHA-256",
            )
        anchor_status = "matched"
    current_source_status = "not_requested"
    if require_current_source:
        try:
            cwd = _canonical_path(
                root["command"]["cwd"],
                role="recorded gate working directory",
                directory=True,
            )
            current_source = _current_source_document(cwd)
        except GateReceiptError as exc:
            if exc.code == "GATE_CURRENT_SOURCE_MISMATCH":
                raise
            _fail(
                "GATE_CURRENT_SOURCE_MISMATCH",
                "current source identity is unavailable",
                cause=exc.code,
            )
        if current_source != root["source"]:
            _fail(
                "GATE_CURRENT_SOURCE_MISMATCH",
                "current source no longer matches the gate receipt",
            )
        current_source_status = "matched"
    return {
        "status": (
            "verified_anchored"
            if anchor_status == "matched"
            else "consistency_checked_unanchored"
        ),
        "generation_path": str(generation_path),
        "content_manifest_sha256": inspected.content_manifest_sha256,
        "content_manifest_anchor": anchor_status,
        "current_source": current_source_status,
        "filesystem_generation_sha256": inspected.filesystem_generation_sha256,
        "outcome": root["outcome"],
        "receipt_sha256": root["receipt_sha256"],
        "shell_return_code": _shell_return_code_from_wait(root["wait"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run and seal one exact-child gate")
    run.add_argument("--output-root", required=True)
    run.add_argument("--generation", required=True)
    run.add_argument("--cwd", required=True)
    run.add_argument("--temp-root", required=True)
    run.add_argument("--timeout-seconds", required=True)
    run.add_argument("--argv0")
    run.add_argument("--inherit-env", action="append", default=[])
    run.add_argument("--env", action="append", default=[])
    run.add_argument("argv", nargs=argparse.REMAINDER)

    verify = subparsers.add_parser("verify", help="verify one sealed gate generation")
    verify.add_argument("generation_path")
    verify.add_argument("--require-current-source", action="store_true")
    verify.add_argument("--expected-content-manifest-sha256")
    return parser


def _error_payload(error: GateReceiptError) -> str:
    return json.dumps(
        {
            "status": "error",
            "typed_error": error.code,
            "detail": error.detail,
            "context": error.context,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            command = list(args.argv)
            if command and command[0] == "--":
                command = command[1:]
            result = run_gate(
                output_root=args.output_root,
                generation=args.generation,
                cwd=args.cwd,
                temp_root=args.temp_root,
                timeout_seconds=args.timeout_seconds,
                argv=command,
                process_argv0=args.argv0,
                inherited_environment=args.inherit_env,
                environment_assignments=args.env,
            )
            print(
                json.dumps(
                    {
                        "generation_path": str(result.generation_path),
                        "outcome": result.document["outcome"],
                        "receipt_sha256": result.document["receipt_sha256"],
                        "producer_content_manifest_sha256": (
                            result.content_manifest_sha256
                        ),
                        "producer_filesystem_generation_sha256": (
                            result.filesystem_generation_sha256
                        ),
                        "producer_manifest_source": (
                            "race_free_exclusive_stage_commit_return_v1"
                        ),
                    },
                    sort_keys=True,
                )
            )
            return result.shell_return_code
        result = verify_generation(
            args.generation_path,
            require_current_source=args.require_current_source,
            expected_content_manifest_sha256=(
                args.expected_content_manifest_sha256
            ),
        )
        print(json.dumps(result, sort_keys=True))
        return result["shell_return_code"]
    except GateReceiptError as exc:
        print(_error_payload(exc), file=sys.stderr)
        return 125
    except (ImmutableArtifactError, OSError) as exc:
        wrapped = GateReceiptError(
            "GATE_INTERNAL_ARTIFACT_ERROR",
            "gate artifact operation failed closed",
            context={"error": str(exc), "errno": getattr(exc, "errno", None)},
        )
        print(_error_payload(wrapped), file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
