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
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from tools.local_labels import immutable_artifact as immutable


NCU_IDENTITY_INPUT_INVALID = "NCU_IDENTITY_INPUT_INVALID"
NCU_ACTION_REQUEST_INVALID = "NCU_ACTION_REQUEST_INVALID"
NCU_EXECUTABLE_IDENTITY_INVALID = "NCU_EXECUTABLE_IDENTITY_INVALID"
NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID = (
    "NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID"
)
NCU_ACTION_INPUT_IDENTITY_INVALID = "NCU_ACTION_INPUT_IDENTITY_INVALID"
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
_MAX_DESCRIPTOR = (1 << 31) - 1
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
class ExpectedExecutableIdentity:
    resolved_path: str
    executable_sha256: str
    executable_mode: int


@dataclass(frozen=True)
class NcuActionResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class _AuthorizedNcuProcPath:
    __slots__ = ()


AUTHORIZED_NCU_PROC_PATH = _AuthorizedNcuProcPath()

_REQUIRED_COLLECTION_ENVIRONMENT = {
    "LANG": "C",
    "LANGUAGE": "C",
    "LC_ALL": "C",
    "PYTHONNOUSERSITE": "1",
    "LABEL_DRIVER_GPU_INDEX": "0",
    "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
}
_OPTIONAL_COLLECTION_ENVIRONMENT_NAMES = frozenset({
    "CC",
    "CUDAHOSTCXX",
    "CUDA_HOME",
    "CUTLASS_DIR",
    "CXX",
    "FLASHINFER_TRACE_DIR",
    "HOME",
    "NVCC_CCBIN",
    "PYTORCH_ALLOC_CONF",
    "SOL_EXECBENCH_CAPTURE_REQUEST",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
})
_MAX_ENVIRONMENT_VALUE_BYTES = 64 * 1024
_COLLECTION_PATH = "/usr/local/cuda-13.2/bin:/usr/bin:/bin"


@dataclass(frozen=True, init=False)
class NcuCollectionEnvironment:
    """One closed, name-allowlisted environment for NCU collection."""

    _items: tuple[tuple[str, str], ...]

    def __init__(self, values: Mapping[str, str]) -> None:
        if type(values) is not dict:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection environment is not an exact dictionary",
            )
        if any(type(name) is not str for name in values):
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection environment contains a non-string name",
            )
        allowed_names = (
            frozenset(_REQUIRED_COLLECTION_ENVIRONMENT)
            | _OPTIONAL_COLLECTION_ENVIRONMENT_NAMES
            | {"PATH", "PWD"}
        )
        names = set(values)
        unexpected = sorted(names - allowed_names)
        if unexpected:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection environment contains non-allowlisted names: "
                + ", ".join(unexpected),
            )
        missing = sorted(
            (set(_REQUIRED_COLLECTION_ENVIRONMENT) | {"PATH", "PWD"}) - names
        )
        if missing:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection environment is missing required names: "
                + ", ".join(missing),
            )
        for name, required in _REQUIRED_COLLECTION_ENVIRONMENT.items():
            if values[name] != required:
                _fail(
                    NCU_ACTION_REQUEST_INVALID,
                    f"collection environment {name} must equal {required!r}",
                )
        normalized: list[tuple[str, str]] = []
        for name in sorted(names):
            value = values[name]
            if (
                type(value) is not str
                or not value
                or "\x00" in value
                or value.splitlines() != [value]
                or _strict_utf8_size(
                    value,
                    code=NCU_ACTION_REQUEST_INVALID,
                    label=f"collection environment {name}",
                ) > _MAX_ENVIRONMENT_VALUE_BYTES
            ):
                _fail(
                    NCU_ACTION_REQUEST_INVALID,
                    f"collection environment {name} is not one bounded value",
                )
            normalized.append((name, value))
        if values["PATH"] != _COLLECTION_PATH:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"collection PATH must equal {_COLLECTION_PATH!r}",
            )
        path_parts = values["PATH"].split(os.pathsep)
        if any(
            not part
            or not Path(part).is_absolute()
            or part.startswith("//")
            or os.path.normpath(part) != part
            for part in path_parts
        ):
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection PATH is not a canonical absolute search list",
            )
        pwd = values["PWD"]
        if (
            not Path(pwd).is_absolute()
            or pwd.startswith("//")
            or os.path.normpath(pwd) != pwd
        ):
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection PWD is not a canonical absolute path",
            )
        object.__setattr__(self, "_items", tuple(normalized))

    def as_dict(self) -> dict[str, str]:
        return dict(self._items)


@dataclass(frozen=True)
class _BorrowedActionDescriptors:
    report: immutable.RetainedRegular | None
    inherited: tuple[int, ...]
    auxiliary: tuple[int, ...]
    auxiliary_identities: tuple[tuple[int, int, int], ...]


class NcuActionExecutorError(RuntimeError):
    """A typed NCU authority or bounded-execution failure."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        self.stdout = b""
        self.stderr = b""
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


def validate_expected_ncu_identity(
    expected: ExpectedNcuIdentity,
) -> ExpectedNcuIdentity:
    """Validate and return one exact typed NCU executable authority."""

    _validate_expected(expected)
    return expected


def build_expected_ncu_identity(
    *, resolved_path: str, executable_sha256: str, version_line: str,
) -> ExpectedNcuIdentity:
    """Build one validated authority without probing or resolving ambient state."""

    return validate_expected_ncu_identity(ExpectedNcuIdentity(
        resolved_path=resolved_path,
        executable_sha256=executable_sha256,
        version_line=version_line,
    ))


def _validate_expected_executable(
    expected: ExpectedExecutableIdentity,
) -> Path:
    if type(expected) is not ExpectedExecutableIdentity:
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "expected supervised executable identity has the wrong type",
        )
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
            code=NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            label="supervised executable resolved_path",
        ) > _MAX_PATH_BYTES
    ):
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "supervised executable resolved_path is not a bounded "
            "canonical absolute path",
        )
    if (
        type(expected.executable_sha256) is not str
        or _SHA256_RE.fullmatch(expected.executable_sha256) is None
    ):
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "supervised executable SHA-256 is not lowercase 64-hex",
        )
    mode = expected.executable_mode
    if (
        type(mode) is not int
        or not 0 <= mode <= 0o7777
        or mode & 0o111 == 0
    ):
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "supervised executable mode is not one executable permission mode",
        )
    return Path(path_text)


def validate_expected_executable_identity(
    expected: ExpectedExecutableIdentity,
) -> ExpectedExecutableIdentity:
    """Validate and return one exact supervised-executable authority."""

    _validate_expected_executable(expected)
    return expected


def build_expected_executable_identity(
    *, resolved_path: str, executable_sha256: str, executable_mode: int,
) -> ExpectedExecutableIdentity:
    """Build one validated supervised-executable authority without probing."""

    return validate_expected_executable_identity(ExpectedExecutableIdentity(
        resolved_path=resolved_path,
        executable_sha256=executable_sha256,
        executable_mode=executable_mode,
    ))


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


def _validate_supervised_command(
    command: Sequence[object], timeout_seconds: float,
) -> tuple[object, ...]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "supervised command is not a sequence",
        )
    try:
        item_count = len(command)
    except Exception as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            f"supervised command count is unavailable: {exc}",
        ) from exc
    if not 1 <= item_count <= _MAX_ARGUMENTS:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            f"supervised command count must be in [1, {_MAX_ARGUMENTS}]",
        )
    rendered: list[str] = []
    template: list[object] = []
    placeholder_count = 0
    for index in range(item_count):
        try:
            item = command[index]
        except Exception as exc:
            raise NcuActionExecutorError(
                NCU_ACTION_REQUEST_INVALID,
                f"supervised command item {index} is unavailable: {exc}",
            ) from exc
        if item is AUTHORIZED_NCU_PROC_PATH:
            if index == 0:
                _fail(
                    NCU_ACTION_REQUEST_INVALID,
                    "authorized NCU path cannot be the supervised executable",
                )
            placeholder_count += 1
            rendered.append("/proc/self/fd/3")
            template.append(item)
        else:
            rendered.append(item)
            template.append(item)
    validated = _validate_action(rendered, timeout_seconds)
    if placeholder_count != 1:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "supervised command must contain one authorized NCU path placeholder",
        )
    executable = validated[0]
    if (
        not Path(executable).is_absolute()
        or executable.startswith("//")
        or os.path.normpath(executable) != executable
    ):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "supervised executable is not a canonical absolute path",
        )
    return tuple(
        original if original is AUTHORIZED_NCU_PROC_PATH else value
        for original, value in zip(template, validated, strict=True)
    )


def _validate_supervised_context(
    environment: NcuCollectionEnvironment,
    working_directory: Path,
) -> str:
    if type(environment) is not NcuCollectionEnvironment:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "collection environment is not an authorized environment value",
        )
    if not isinstance(working_directory, Path):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "supervised working directory is not a path",
        )
    working_directory_text = str(working_directory)
    if (
        not working_directory_text
        or "\x00" in working_directory_text
        or working_directory_text.splitlines() != [working_directory_text]
        or not working_directory.is_absolute()
        or working_directory_text.startswith("//")
        or os.path.normpath(working_directory_text) != working_directory_text
        or _strict_utf8_size(
            working_directory_text,
            code=NCU_ACTION_REQUEST_INVALID,
            label="supervised working directory",
        ) > _MAX_PATH_BYTES
    ):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "supervised working directory is not a bounded canonical absolute path",
        )
    try:
        metadata = os.stat(working_directory, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            f"supervised working directory is unavailable: {exc}",
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "supervised working directory is not a no-follow directory",
        )
    if environment.as_dict()["PWD"] != working_directory_text:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "collection PWD differs from supervised working directory",
        )
    return working_directory_text


def _render_supervised_command(
    template: tuple[object, ...], retained: immutable.RetainedExecutable,
) -> tuple[str, ...]:
    retained_path = proc_fd_path(retained.descriptor)
    return tuple(
        retained_path if item is AUTHORIZED_NCU_PROC_PATH else item
        for item in template
    )


def _render_supervised_executable_command(
    template: tuple[object, ...],
    retained_ncu: immutable.RetainedExecutable,
    retained_executable: immutable.RetainedExecutable,
) -> tuple[str, ...]:
    rendered = _render_supervised_command(template, retained_ncu)
    return (proc_fd_path(retained_executable.descriptor), *rendered[1:])


def _descriptor_identity(descriptor: int) -> tuple[int, int, int]:
    try:
        metadata = os.fstat(descriptor)
    except (OSError, OverflowError) as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            "borrowed descriptor is not an open platform descriptor",
        ) from exc
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def proc_fd_path(descriptor: int) -> str:
    """Return the canonical child-visible path for one open borrowed descriptor."""

    if (
        type(descriptor) is not int
        or not 3 <= descriptor <= _MAX_DESCRIPTOR
    ):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "borrowed descriptor is not an integer greater than stderr",
        )
    _descriptor_identity(descriptor)
    return f"/proc/self/fd/{descriptor}"


def _validate_borrowed_descriptors(
    action: tuple[str, ...],
    retained_report: immutable.RetainedRegular | None,
    auxiliary_descriptors: Sequence[int],
) -> _BorrowedActionDescriptors:
    if retained_report is not None and type(retained_report) is not immutable.RetainedRegular:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "retained_report is not an exact retained regular-file capability",
        )
    if (
        isinstance(auxiliary_descriptors, (str, bytes))
        or not isinstance(auxiliary_descriptors, Sequence)
    ):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "auxiliary_descriptors is not a sequence of integers",
        )
    try:
        auxiliary_count = len(auxiliary_descriptors)
    except Exception as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            f"auxiliary descriptor count is unavailable: {exc}",
        ) from exc
    if auxiliary_count > _MAX_ARGUMENTS:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            f"auxiliary descriptor count exceeds {_MAX_ARGUMENTS}",
        )
    auxiliary: list[int] = []
    identities: list[tuple[int, int, int]] = []
    for index in range(auxiliary_count):
        try:
            descriptor = auxiliary_descriptors[index]
        except Exception as exc:
            raise NcuActionExecutorError(
                NCU_ACTION_REQUEST_INVALID,
                f"auxiliary descriptor {index} is unavailable: {exc}",
            ) from exc
        if (
            type(descriptor) is not int
            or not 3 <= descriptor <= _MAX_DESCRIPTOR
        ):
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"auxiliary descriptor {index} is not an integer greater than stderr",
            )
        auxiliary.append(descriptor)
        identities.append(_descriptor_identity(descriptor))
    if len(set(auxiliary)) != len(auxiliary):
        _fail(NCU_ACTION_REQUEST_INVALID, "auxiliary descriptors are not unique")

    report_descriptor: int | None = None
    if retained_report is not None:
        try:
            report_descriptor = retained_report.descriptor
        except immutable.ImmutableArtifactError as exc:
            raise NcuActionExecutorError(
                NCU_ACTION_REQUEST_INVALID,
                f"retained report descriptor is unavailable: {exc}",
            ) from exc
        if not 3 <= report_descriptor <= _MAX_DESCRIPTOR:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "retained report descriptor is not greater than stderr",
            )
        if report_descriptor in auxiliary:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "retained report descriptor is duplicated as an auxiliary",
            )

    inherited = (
        (() if report_descriptor is None else (report_descriptor,))
        + tuple(auxiliary)
    )
    registered_paths = {f"/proc/self/fd/{descriptor}" for descriptor in inherited}
    for argument in action:
        if "/proc/self/fd/" in argument and argument not in registered_paths:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"action contains an unregistered or noncanonical descriptor path: {argument!r}",
            )
    for path in registered_paths:
        if action.count(path) != 1:
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                f"borrowed descriptor path must occur exactly once: {path}",
            )
    return _BorrowedActionDescriptors(
        retained_report,
        inherited,
        tuple(auxiliary),
        tuple(identities),
    )


_ACTION_ENVIRONMENT_ITEMS = (
    ("LANG", "C"),
    ("LANGUAGE", "C"),
    ("LC_ALL", "C"),
    ("PATH", "/bin:/usr/bin"),
)


def ncu_action_environment() -> dict[str, str]:
    """Return the exact launcher environment supplied to retained NCU actions."""

    return dict(_ACTION_ENVIRONMENT_ITEMS)


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
                remaining_capacity = max_output_bytes - total
                buffers[key.data].extend(chunk[:remaining_capacity])
                total += len(chunk)
                if total > max_output_bytes:
                    raise NcuActionExecutorError(
                        NCU_ACTION_OUTPUT_LIMIT,
                        f"retained NCU output exceeds {max_output_bytes} bytes",
                    )
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
        if isinstance(active_error, NcuActionExecutorError):
            active_error.stdout = bytes(buffers["stdout"])
            active_error.stderr = bytes(buffers["stderr"])
        try:
            _stop_and_reap(process)
        except BaseException as cleanup_error:
            try:
                active_error.add_note(f"process cleanup failed: {cleanup_error}")
            except (AttributeError, TypeError):
                pass
        raise
    finally:
        # A successful direct child may still leave a descendant holding inherited
        # capabilities after closing its pipes. The dedicated process group never
        # survives this synchronous boundary.
        _signal_group(process, signal.SIGKILL)
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


def _run_command_bounded(
    command: tuple[str, ...],
    *,
    environment: dict[str, str],
    working_directory: str,
    inherited_descriptors: tuple[int, ...],
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NcuActionResult:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=working_directory,
            pass_fds=inherited_descriptors,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise NcuActionExecutorError(
            launch_code, f"cannot launch supervised process: {exc}"
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
        wrapped = NcuActionExecutorError(launch_code, exc.detail)
        wrapped.stdout = exc.stdout
        wrapped.stderr = exc.stderr
        raise wrapped from exc


def execute_bounded_command(
    command: tuple[str, ...],
    *,
    environment: dict[str, str],
    working_directory: str,
    inherited_descriptors: tuple[int, ...],
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NcuActionResult:
    """Run a retained-capability command with bounded process cleanup."""
    return _run_command_bounded(
        command,
        environment=environment,
        working_directory=working_directory,
        inherited_descriptors=inherited_descriptors,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        timeout_code=timeout_code,
        launch_code=launch_code,
    )


def _run_bounded(
    descriptor: int,
    arguments: tuple[str, ...],
    *,
    inherited_descriptors: tuple[int, ...] = (),
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NcuActionResult:
    command = (f"/proc/self/fd/{descriptor}", *arguments)
    return _run_command_bounded(
        command,
        environment=ncu_action_environment(),
        working_directory="/",
        inherited_descriptors=(descriptor, *inherited_descriptors),
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        timeout_code=timeout_code,
        launch_code=launch_code,
    )


def _identity_error(detail: str, cause: BaseException) -> NcuActionExecutorError:
    return NcuActionExecutorError(
        NCU_EXECUTABLE_IDENTITY_INVALID, f"{detail}: {cause}"
    )


def _input_identity_error(
    detail: str, cause: BaseException,
) -> NcuActionExecutorError:
    return NcuActionExecutorError(
        NCU_ACTION_INPUT_IDENTITY_INVALID, f"{detail}: {cause}"
    )


def _supervised_executable_identity_error(
    detail: str, cause: BaseException,
) -> NcuActionExecutorError:
    return NcuActionExecutorError(
        NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
        f"{detail}: {cause}",
    )


def _authorize_distinct_opened_executables(
    retained_ncu: immutable.RetainedExecutable,
    retained_executable: immutable.RetainedExecutable,
) -> None:
    try:
        ncu_metadata = os.fstat(retained_ncu.descriptor)
    except OSError as exc:
        raise _identity_error(
            "cannot identify retained NCU executable", exc,
        ) from exc
    try:
        executable_metadata = os.fstat(retained_executable.descriptor)
    except OSError as exc:
        raise _supervised_executable_identity_error(
            "cannot identify retained supervised executable", exc,
        ) from exc
    if (
        ncu_metadata.st_dev,
        ncu_metadata.st_ino,
    ) == (
        executable_metadata.st_dev,
        executable_metadata.st_ino,
    ):
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "supervised executable aliases the retained NCU executable",
        )


def _authorize_supervised_executable_opened(
    retained: immutable.RetainedExecutable,
    expected: ExpectedExecutableIdentity,
) -> None:
    if str(retained.path) != expected.resolved_path:
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "retained supervised executable path differs from independent authority",
        )
    if retained.sha256 != expected.executable_sha256:
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "retained supervised executable SHA-256 differs from independent authority",
        )
    try:
        mode = stat.S_IMODE(os.fstat(retained.descriptor).st_mode)
    except OSError as exc:
        raise _supervised_executable_identity_error(
            "cannot read retained supervised executable mode", exc,
        ) from exc
    if mode != expected.executable_mode:
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "retained supervised executable mode differs from independent authority",
        )
    try:
        retained.verify_unchanged()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise _supervised_executable_identity_error(
            "retained supervised executable changed", exc,
        ) from exc


def _reverify_with_supervised_executable(
    retained_ncu: immutable.RetainedExecutable,
    borrowed: _BorrowedActionDescriptors,
    retained_executable: immutable.RetainedExecutable,
    *,
    pending: BaseException | None = None,
) -> None:
    failures: list[tuple[NcuActionExecutorError, BaseException]] = []
    try:
        _reverify(retained_ncu, borrowed)
    except NcuActionExecutorError as exc:
        failures.append((exc, exc.__cause__ or exc))
    try:
        retained_executable.verify_unchanged()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        failures.append((
            _supervised_executable_identity_error(
                "retained supervised executable changed", exc,
            ),
            exc,
        ))
    if failures:
        primary, cause = failures[0]
        for secondary, _secondary_cause in failures[1:]:
            try:
                primary.add_note(f"additional identity failure: {secondary}")
            except (AttributeError, TypeError):
                pass
        if pending is not None:
            if isinstance(pending, NcuActionExecutorError):
                primary.stdout = pending.stdout
                primary.stderr = pending.stderr
            raise primary from pending
        raise primary from cause
    if pending is not None:
        raise pending


def _reverify_authorities(
    retained: immutable.RetainedExecutable,
    borrowed: _BorrowedActionDescriptors,
    retained_supervised_executable: immutable.RetainedExecutable | None,
    *,
    pending: BaseException | None = None,
) -> None:
    if retained_supervised_executable is None:
        _reverify(retained, borrowed, pending=pending)
    else:
        _reverify_with_supervised_executable(
            retained,
            borrowed,
            retained_supervised_executable,
            pending=pending,
        )


def _verify_auxiliary_descriptors(
    borrowed: _BorrowedActionDescriptors,
) -> None:
    for descriptor, expected in zip(
        borrowed.auxiliary, borrowed.auxiliary_identities, strict=True,
    ):
        try:
            observed = os.fstat(descriptor)
        except OSError as exc:
            raise _input_identity_error(
                f"auxiliary descriptor {descriptor} is no longer open", exc,
            ) from exc
        identity = (
            observed.st_dev,
            observed.st_ino,
            stat.S_IFMT(observed.st_mode),
        )
        if identity != expected:
            cause = RuntimeError(
                f"expected={expected!r}, observed={identity!r}"
            )
            raise _input_identity_error(
                f"auxiliary descriptor {descriptor} changed identity", cause,
            ) from cause


def _reverify(
    retained: immutable.RetainedExecutable,
    borrowed: _BorrowedActionDescriptors,
    *,
    pending: BaseException | None = None,
) -> None:
    failures: list[tuple[NcuActionExecutorError, BaseException]] = []
    try:
        retained.verify_unchanged()
    except (OSError, immutable.ImmutableArtifactError) as exc:
        failures.append((_identity_error("retained executable changed", exc), exc))
    if borrowed.report is not None:
        try:
            borrowed.report.verify_unchanged()
        except (OSError, immutable.ImmutableArtifactError) as exc:
            failures.append((
                _input_identity_error("retained report changed", exc), exc,
            ))
    try:
        _verify_auxiliary_descriptors(borrowed)
    except NcuActionExecutorError as exc:
        failures.append((exc, exc.__cause__ or exc))
    if failures:
        primary, cause = failures[0]
        for secondary, _secondary_cause in failures[1:]:
            try:
                primary.add_note(f"additional identity failure: {secondary}")
            except (AttributeError, TypeError):
                pass
        if pending is not None:
            if isinstance(pending, NcuActionExecutorError):
                primary.stdout = pending.stdout
                primary.stderr = pending.stderr
            raise primary from pending
        raise primary from cause
    if pending is not None:
        raise pending


def _run_and_reverify(
    retained: immutable.RetainedExecutable,
    borrowed: _BorrowedActionDescriptors,
    arguments: tuple[str, ...],
    *,
    retained_supervised_executable: immutable.RetainedExecutable | None = None,
    inherited_descriptors: tuple[int, ...] = (),
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    launch_code: str,
) -> NcuActionResult:
    try:
        result = _run_bounded(
            retained.descriptor,
            arguments,
            inherited_descriptors=inherited_descriptors,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            timeout_code=timeout_code,
            launch_code=launch_code,
        )
    except BaseException as exc:
        _reverify_authorities(
            retained,
            borrowed,
            retained_supervised_executable,
            pending=exc,
        )
        raise AssertionError("unreachable")
    _reverify_authorities(
        retained, borrowed, retained_supervised_executable,
    )
    return result


def _authorize_opened(
    retained: immutable.RetainedExecutable,
    expected: ExpectedNcuIdentity,
    borrowed: _BorrowedActionDescriptors,
    *,
    retained_supervised_executable: immutable.RetainedExecutable | None = None,
) -> None:
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
    _reverify_authorities(
        retained, borrowed, retained_supervised_executable,
    )
    probe = _run_and_reverify(
        retained,
        borrowed,
        ("--version",),
        retained_supervised_executable=retained_supervised_executable,
        timeout_seconds=_VERSION_TIMEOUT_SECONDS,
        max_output_bytes=_MAX_VERSION_OUTPUT_BYTES,
        timeout_code=NCU_VERSION_PROBE_FAILED,
        launch_code=NCU_VERSION_PROBE_FAILED,
    )
    if probe.returncode != 0:
        _reverify_authorities(
            retained,
            borrowed,
            retained_supervised_executable,
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
        _reverify_authorities(
            retained,
            borrowed,
            retained_supervised_executable,
            pending=NcuActionExecutorError(
                NCU_VERSION_LINE_MISMATCH,
                "retained NCU version line differs from independent authority",
            ),
        )
    _reverify_authorities(
        retained, borrowed, retained_supervised_executable,
    )


def _execute_opened(
    retained: immutable.RetainedExecutable,
    expected: ExpectedNcuIdentity,
    action: tuple[str, ...],
    timeout_seconds: float,
    borrowed: _BorrowedActionDescriptors,
) -> NcuActionResult:
    _authorize_opened(retained, expected, borrowed)
    return _run_and_reverify(
        retained,
        borrowed,
        action,
        inherited_descriptors=borrowed.inherited,
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
    retained_report: immutable.RetainedRegular | None = None,
    auxiliary_descriptors: Sequence[int] = (),
) -> NcuActionResult:
    """Run one bounded NCU action through independently retained capabilities."""
    path = _validate_expected(expected)
    action = _validate_action(arguments, timeout_seconds)
    borrowed = _validate_borrowed_descriptors(
        action, retained_report, auxiliary_descriptors,
    )
    try:
        retained = immutable.open_retained_executable(path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise _identity_error("cannot retain expected executable", exc) from exc

    try:
        result = _execute_opened(
            retained, expected, action, timeout_seconds, borrowed,
        )
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


def execute_ncu_supervised_child(
    expected: ExpectedNcuIdentity,
    command: Sequence[object],
    *,
    timeout_seconds: float,
    environment: NcuCollectionEnvironment,
    working_directory: Path = Path("/"),
) -> NcuActionResult:
    """Run one bounded child with exactly one inherited authorized NCU path."""

    path = _validate_expected(expected)
    template = _validate_supervised_command(command, timeout_seconds)
    working_directory_text = _validate_supervised_context(
        environment, working_directory,
    )
    borrowed = _BorrowedActionDescriptors(None, (), (), ())
    try:
        retained = immutable.open_retained_executable(path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise _identity_error("cannot retain expected executable", exc) from exc
    try:
        _authorize_opened(retained, expected, borrowed)
        rendered = _render_supervised_command(template, retained)
        try:
            result = _run_command_bounded(
                rendered,
                environment=environment.as_dict(),
                working_directory=working_directory_text,
                inherited_descriptors=(retained.descriptor,),
                timeout_seconds=timeout_seconds,
                max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
                timeout_code=NCU_ACTION_TIMEOUT,
                launch_code=NCU_ACTION_LAUNCH_FAILED,
            )
        except BaseException as action_error:
            _reverify(retained, borrowed, pending=action_error)
            raise AssertionError("unreachable")
        _reverify(retained, borrowed)
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


def execute_ncu_supervised_executable(
    expected: ExpectedNcuIdentity,
    expected_executable: ExpectedExecutableIdentity,
    command: Sequence[object],
    *,
    timeout_seconds: float,
    environment: NcuCollectionEnvironment,
    working_directory: Path = Path("/"),
) -> NcuActionResult:
    """Run a descriptor-bound child with one descriptor-bound NCU authority."""

    ncu_path = _validate_expected(expected)
    executable_path = _validate_expected_executable(expected_executable)
    if ncu_path == executable_path:
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "supervised executable must be distinct from NCU authority",
        )
    template = _validate_supervised_command(command, timeout_seconds)
    if template[0] != expected_executable.resolved_path:
        _fail(
            NCU_SUPERVISED_EXECUTABLE_IDENTITY_INVALID,
            "supervised command path differs from independent authority",
        )
    working_directory_text = _validate_supervised_context(
        environment, working_directory,
    )

    borrowed = _BorrowedActionDescriptors(None, (), (), ())
    try:
        retained_ncu = immutable.open_retained_executable(ncu_path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise _identity_error("cannot retain expected executable", exc) from exc
    try:
        retained_executable = immutable.open_retained_executable(executable_path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        error = _supervised_executable_identity_error(
            "cannot retain expected supervised executable", exc,
        )
        try:
            retained_ncu.close()
        except BaseException as close_error:
            try:
                error.add_note(f"retained NCU close failed: {close_error}")
            except (AttributeError, TypeError):
                pass
        raise error from exc

    try:
        _authorize_distinct_opened_executables(
            retained_ncu, retained_executable,
        )
        _authorize_supervised_executable_opened(
            retained_executable, expected_executable,
        )
        _authorize_opened(
            retained_ncu,
            expected,
            borrowed,
            retained_supervised_executable=retained_executable,
        )
        rendered = _render_supervised_executable_command(
            template, retained_ncu, retained_executable,
        )
        try:
            result = _run_command_bounded(
                rendered,
                environment=environment.as_dict(),
                working_directory=working_directory_text,
                inherited_descriptors=(
                    retained_ncu.descriptor,
                    retained_executable.descriptor,
                ),
                timeout_seconds=timeout_seconds,
                max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
                timeout_code=NCU_ACTION_TIMEOUT,
                launch_code=NCU_ACTION_LAUNCH_FAILED,
            )
        except BaseException as action_error:
            _reverify_with_supervised_executable(
                retained_ncu,
                borrowed,
                retained_executable,
                pending=action_error,
            )
            raise AssertionError("unreachable")
        _reverify_with_supervised_executable(
            retained_ncu, borrowed, retained_executable,
        )
    except BaseException as active_error:
        for label, retained in (
            ("supervised executable", retained_executable),
            ("NCU executable", retained_ncu),
        ):
            try:
                retained.close()
            except BaseException as close_error:
                try:
                    active_error.add_note(
                        f"retained {label} close failed: {close_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        raise

    try:
        retained_executable.close()
    except BaseException as exc:
        try:
            retained_ncu.close()
        except BaseException as close_error:
            try:
                exc.add_note(f"retained NCU close failed: {close_error}")
            except (AttributeError, TypeError):
                pass
        raise _supervised_executable_identity_error(
            "cannot close retained supervised executable", exc,
        ) from exc
    try:
        retained_ncu.close()
    except BaseException as exc:
        raise _identity_error("cannot close retained executable", exc) from exc
    return result
