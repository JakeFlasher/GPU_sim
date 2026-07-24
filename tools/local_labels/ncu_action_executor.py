#!/usr/bin/env python3
"""Descriptor-bound execution of independently authorized NCU actions.

This is not a process sandbox. Authorized actions must not daemonize or leave the
new process group; bounded cleanup terminates and reaps that process group.
"""

from __future__ import annotations

import math
import os
import re
import selectors
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from tools.local_labels import immutable_artifact as immutable


NCU_IDENTITY_INPUT_INVALID = "NCU_IDENTITY_INPUT_INVALID"
NCU_ACTION_REQUEST_INVALID = "NCU_ACTION_REQUEST_INVALID"
NCU_EXECUTABLE_IDENTITY_INVALID = "NCU_EXECUTABLE_IDENTITY_INVALID"
NCU_VERSION_PROBE_FAILED = "NCU_VERSION_PROBE_FAILED"
NCU_VERSION_LINE_MISMATCH = "NCU_VERSION_LINE_MISMATCH"
NCU_ACTION_LAUNCH_FAILED = "NCU_ACTION_LAUNCH_FAILED"
NCU_ACTION_TIMEOUT = "NCU_ACTION_TIMEOUT"
NCU_ACTION_OUTPUT_LIMIT = "NCU_ACTION_OUTPUT_LIMIT"

_MAX_PATH_BYTES = 4096
_MAX_VERSION_BYTES = 4096
_MAX_ARGUMENTS = 4096
_MAX_SINGLE_ARGUMENT_BYTES = 131071
_MAX_ARGUMENT_BYTES = 1024 * 1024
_MAX_VERSION_OUTPUT_BYTES = 1024 * 1024
_MAX_ACTION_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
_VERSION_TIMEOUT_SECONDS = 30
_TERMINATION_GRACE_SECONDS = 1
_IO_CHUNK_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^Version\s+\S")


@dataclass(frozen=True)
class ExpectedNcuIdentity:
    resolved_path: str
    executable_sha256: str
    version_line: str


@dataclass(frozen=True)
class NcuActionResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class NcuActionExecutorError(RuntimeError):
    """A typed NCU authority or bounded-execution failure."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise NcuActionExecutorError(code, detail)


def _strict_utf8_size(value: str, *, code: str, label: str) -> int:
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise NcuActionExecutorError(
            code, f"{label} is not strict UTF-8: {exc}"
        ) from exc


def _validate_expected(expected: ExpectedNcuIdentity) -> Path:
    if type(expected) is not ExpectedNcuIdentity:
        _fail(NCU_IDENTITY_INPUT_INVALID, "expected identity has the wrong type")
    path_text = expected.resolved_path
    if (
        type(path_text) is not str
        or not path_text
        or "\x00" in path_text
        or not Path(path_text).is_absolute()
        or path_text.startswith("//")
        or os.path.normpath(path_text) != path_text
        or _strict_utf8_size(
            path_text,
            code=NCU_IDENTITY_INPUT_INVALID,
            label="resolved_path",
        ) > _MAX_PATH_BYTES
    ):
        _fail(
            NCU_IDENTITY_INPUT_INVALID,
            "resolved_path is not a bounded canonical absolute path",
        )
    if (
        type(expected.executable_sha256) is not str
        or _SHA256_RE.fullmatch(expected.executable_sha256) is None
    ):
        _fail(
            NCU_IDENTITY_INPUT_INVALID,
            "executable_sha256 is not lowercase 64-hex",
        )
    version = expected.version_line
    if (
        type(version) is not str
        or not version
        or version != version.strip()
        or "\x00" in version
        or version.splitlines() != [version]
        or _VERSION_RE.match(version) is None
        or _strict_utf8_size(
            version,
            code=NCU_IDENTITY_INPUT_INVALID,
            label="version_line",
        ) > _MAX_VERSION_BYTES
    ):
        _fail(
            NCU_IDENTITY_INPUT_INVALID,
            "version_line is not one bounded canonical Version line",
        )
    return Path(path_text)


def _validate_action(
    arguments: Sequence[str], timeout_seconds: float,
) -> tuple[str, ...]:
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        _fail(NCU_ACTION_REQUEST_INVALID, "arguments is not a sequence of strings")
    try:
        argument_count = len(arguments)
    except Exception as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            f"argument count is unavailable: {exc}",
        ) from exc
    if not 1 <= argument_count <= _MAX_ARGUMENTS:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            f"argument count must be in [1, {_MAX_ARGUMENTS}]",
        )
    action: list[str] = []
    encoded_size = 0
    for index in range(argument_count):
        try:
            argument = arguments[index]
        except Exception as exc:
            raise NcuActionExecutorError(
                NCU_ACTION_REQUEST_INVALID,
                f"argument {index} is unavailable: {exc}",
            ) from exc
        if type(argument) is not str or "\x00" in argument:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"argument {index} is not a NUL-free string",
            )
        if len(argument) > _MAX_SINGLE_ARGUMENT_BYTES:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"argument {index} exceeds the per-string exec bound",
            )
        encoded_argument_size = _strict_utf8_size(
            argument,
            code=NCU_ACTION_REQUEST_INVALID,
            label=f"argument {index}",
        )
        if encoded_argument_size > _MAX_SINGLE_ARGUMENT_BYTES:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"argument {index} exceeds the per-string exec bound",
            )
        encoded_size += encoded_argument_size + 1
        if encoded_size > _MAX_ARGUMENT_BYTES:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"encoded arguments exceed {_MAX_ARGUMENT_BYTES} bytes",
            )
        action.append(argument)
    if (
        type(timeout_seconds) not in {int, float}
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_TIMEOUT_SECONDS
        or not math.isfinite(timeout_seconds)
    ):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            f"timeout_seconds must be finite and in (0, {_MAX_TIMEOUT_SECONDS}]",
        )
    return tuple(action)


def _c_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "LANGUAGE": "C",
        "PATH": os.defpath,
    }


def _signal_group(process: subprocess.Popen[bytes], signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.send_signal(signal_number)
        except ProcessLookupError:
            return


def _stop_and_reap(process: subprocess.Popen[bytes]) -> None:
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise NcuActionExecutorError(
                NCU_ACTION_LAUNCH_FAILED,
                "cannot reap retained NCU process after SIGKILL",
            ) from exc
    finally:
        # The direct process can exit before a stubborn descendant in its group.
        _signal_group(process, signal.SIGKILL)


def _close_monitor_resources(
    selector: selectors.BaseSelector | None,
    process: subprocess.Popen[bytes],
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    if selector is not None:
        try:
            selector.close()
        except BaseException as exc:
            errors.append(exc)
    for pipe in (process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except BaseException as exc:
                errors.append(exc)
    return tuple(errors)


def _collect_bounded(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
) -> NcuActionResult:
    selector: selectors.BaseSelector | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    deadline = time.monotonic() + timeout_seconds
    primary_error: BaseException | None = None
    try:
        if process.stdout is None or process.stderr is None:
            raise NcuActionExecutorError(
                NCU_ACTION_LAUNCH_FAILED, "retained NCU pipes are unavailable",
            )
        try:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        except (OSError, ValueError, KeyError) as exc:
            raise NcuActionExecutorError(
                NCU_ACTION_LAUNCH_FAILED,
                f"cannot initialize retained NCU output monitor: {exc}",
            ) from exc
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NcuActionExecutorError(
                    timeout_code,
                    f"retained NCU action exceeded {timeout_seconds} seconds",
                )
            try:
                events = selector.select(timeout=min(0.1, remaining))
            except OSError as exc:
                raise NcuActionExecutorError(
                    NCU_ACTION_LAUNCH_FAILED,
                    f"cannot monitor retained NCU output: {exc}",
                ) from exc
            for key, _mask in events:
                try:
                    chunk = os.read(key.fd, _IO_CHUNK_BYTES)
                except OSError as exc:
                    raise NcuActionExecutorError(
                        NCU_ACTION_LAUNCH_FAILED,
                        f"cannot read retained NCU output: {exc}",
                    ) from exc
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > max_output_bytes:
                    raise NcuActionExecutorError(
                        NCU_ACTION_OUTPUT_LIMIT,
                        f"retained NCU output exceeds {max_output_bytes} bytes",
                    )
                buffers[key.data].extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NcuActionExecutorError(
                timeout_code,
                f"retained NCU action exceeded {timeout_seconds} seconds",
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise NcuActionExecutorError(
                timeout_code,
                f"retained NCU action exceeded {timeout_seconds} seconds",
            ) from exc
        return NcuActionResult(
            returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]),
        )
    except BaseException as active_error:
        primary_error = active_error
        try:
            _stop_and_reap(process)
        except BaseException as cleanup_error:
            try:
                active_error.add_note(f"process cleanup failed: {cleanup_error}")
            except (AttributeError, TypeError):
                pass
        raise
    finally:
        close_errors = _close_monitor_resources(selector, process)
        if close_errors:
            detail = "; ".join(str(error) for error in close_errors)
            if primary_error is not None:
                try:
                    primary_error.add_note(
                        f"process resource closure failed: {detail}"
                    )
                except (AttributeError, TypeError):
                    pass
            else:
                raise NcuActionExecutorError(
                    NCU_ACTION_LAUNCH_FAILED,
                    f"cannot close retained NCU process resources: {detail}",
                ) from close_errors[0]


def _run_bounded(
    descriptor: int,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NcuActionResult:
    command = [f"/proc/self/fd/{descriptor}", *arguments]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_c_environment(),
            pass_fds=(descriptor,),
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise NcuActionExecutorError(
            launch_code, f"cannot launch retained NCU executable: {exc}"
        ) from exc
    try:
        return _collect_bounded(
            process,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            timeout_code=timeout_code,
        )
    except NcuActionExecutorError as exc:
        if exc.code in {NCU_ACTION_TIMEOUT, NCU_ACTION_OUTPUT_LIMIT}:
            raise
        if exc.code == timeout_code:
            raise
        raise NcuActionExecutorError(
            launch_code, exc.detail,
        ) from exc


def _identity_error(detail: str, cause: BaseException) -> NcuActionExecutorError:
    return NcuActionExecutorError(
        NCU_EXECUTABLE_IDENTITY_INVALID, f"{detail}: {cause}"
    )


def _reverify(
    retained: immutable.RetainedExecutable,
    *,
    pending: NcuActionExecutorError | None = None,
) -> None:
    try:
        retained.verify_unchanged()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        error = _identity_error("retained executable changed", exc)
        if pending is not None:
            raise error from pending
        raise error from exc
    if pending is not None:
        raise pending


def _run_and_reverify(
    retained: immutable.RetainedExecutable,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NcuActionResult:
    try:
        result = _run_bounded(
            retained.descriptor,
            arguments,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            timeout_code=timeout_code,
            launch_code=launch_code,
        )
    except NcuActionExecutorError as exc:
        _reverify(retained, pending=exc)
        raise AssertionError("unreachable")
    _reverify(retained)
    return result


def _execute_opened(
    retained: immutable.RetainedExecutable,
    expected: ExpectedNcuIdentity,
    action: tuple[str, ...],
    timeout_seconds: float,
) -> NcuActionResult:
    if str(retained.path) != expected.resolved_path:
        _fail(
            NCU_EXECUTABLE_IDENTITY_INVALID,
            "retained executable path differs from independent authority",
        )
    if retained.sha256 != expected.executable_sha256:
        _fail(
            NCU_EXECUTABLE_IDENTITY_INVALID,
            "retained executable SHA-256 differs from independent authority",
        )
    try:
        writable = os.access(
            f"/proc/self/fd/{retained.descriptor}",
            os.W_OK,
            effective_ids=True,
        )
    except (OSError, TypeError) as exc:
        raise _identity_error(
            "cannot establish retained executable write protection", exc,
        ) from exc
    if writable:
        _fail(
            NCU_EXECUTABLE_IDENTITY_INVALID,
            "retained executable is writable by the capture process",
        )
    probe = _run_and_reverify(
        retained,
        ("--version",),
        timeout_seconds=_VERSION_TIMEOUT_SECONDS,
        max_output_bytes=_MAX_VERSION_OUTPUT_BYTES,
        timeout_code=NCU_VERSION_PROBE_FAILED,
        launch_code=NCU_VERSION_PROBE_FAILED,
    )
    if probe.returncode != 0:
        _reverify(
            retained,
            pending=NcuActionExecutorError(
                NCU_VERSION_PROBE_FAILED,
                f"retained NCU --version exited {probe.returncode}",
            ),
        )
    decoded = probe.stdout.decode("utf-8", errors="surrogateescape")
    version_lines = [
        line.strip()
        for line in decoded.splitlines()
        if re.match(r"^\s*Version\s+", line)
    ]
    if len(version_lines) != 1 or version_lines[0] != expected.version_line:
        _reverify(
            retained,
            pending=NcuActionExecutorError(
                NCU_VERSION_LINE_MISMATCH,
                "retained NCU version line differs from independent authority",
            ),
        )
    _reverify(retained)
    return _run_and_reverify(
        retained,
        action,
        timeout_seconds=timeout_seconds,
        max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
        timeout_code=NCU_ACTION_TIMEOUT,
        launch_code=NCU_ACTION_LAUNCH_FAILED,
    )


def execute_ncu_action(
    expected: ExpectedNcuIdentity,
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
) -> NcuActionResult:
    """Run one bounded NCU action through an independently authorized inode."""
    path = _validate_expected(expected)
    action = _validate_action(arguments, timeout_seconds)
    try:
        retained = immutable.open_retained_executable(path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise _identity_error("cannot retain expected executable", exc) from exc

    try:
        result = _execute_opened(retained, expected, action, timeout_seconds)
    except BaseException as active_error:
        try:
            retained.close()
        except BaseException as close_error:
            try:
                active_error.add_note(
                    f"retained executable close failed: {close_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    try:
        retained.close()
    except BaseException as exc:
        raise _identity_error("cannot close retained executable", exc) from exc
    return result
