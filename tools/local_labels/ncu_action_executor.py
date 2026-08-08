#!/usr/bin/env python3
"""Descriptor-bound execution of independently authorized NCU actions.

Production collection may require a delegated cgroup-v2 leaf so descendants remain
bounded even after changing their process group or session.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from tools.local_labels import delegated_cgroup
from tools.local_labels import immutable_artifact as immutable
from tools.nvbit_tracer import capture_process as nvbit_capture_process


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
    process_id: int | None = field(default=None, compare=False)
    started_at_utc: str | None = field(default=None, compare=False)
    containment: delegated_cgroup.CgroupContainmentEvidence | None = field(
        default=None, compare=False,
    )


class _AuthorizedNcuProcPath:
    __slots__ = ()


AUTHORIZED_NCU_PROC_PATH = _AuthorizedNcuProcPath()


_AUTHORIZED_EXECUTABLE_PROC_PATH_SEAL = object()


class _AuthorizedExecutableProcPath:
    """One authenticated runtime descriptor and canonical durable path."""

    __slots__ = ("canonical_path", "descriptor", "descriptor_identity", "_seal")

    def __init__(
        self,
        descriptor: int,
        canonical_path: str,
        descriptor_identity: tuple[int, int, int],
        seal: object,
    ) -> None:
        if seal is not _AUTHORIZED_EXECUTABLE_PROC_PATH_SEAL:
            raise TypeError(
                "authorized executable paths require a retained capability"
            )
        self.descriptor = descriptor
        self.canonical_path = canonical_path
        self.descriptor_identity = descriptor_identity
        self._seal = seal


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
    "LD_SOURCE_COMMIT_OVERRIDE",
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
        source_commit = values.get("LD_SOURCE_COMMIT_OVERRIDE")
        if source_commit is not None and (
            type(source_commit) is not str
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        ):
            _fail(
                NCU_ACTION_REQUEST_INVALID,
                "collection environment LD_SOURCE_COMMIT_OVERRIDE must be "
                "lowercase 40-hex",
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

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        process_cleanup_incomplete: bool = False,
    ) -> None:
        if type(process_cleanup_incomplete) is not bool:
            raise TypeError("process_cleanup_incomplete must be an exact boolean")
        self.code = code
        self.detail = detail
        self.process_cleanup_incomplete = process_cleanup_incomplete
        self.stdout = b""
        self.stderr = b""
        self.process_id: int | None = None
        self.started_at_utc: str | None = None
        self.process_role: str | None = None
        self.process_command: tuple[str, ...] | None = None
        self.process_environment: dict[str, str] | None = None
        self.process_tool_sha256: str | None = None
        self.containment: delegated_cgroup.CgroupContainmentEvidence | None = None
        super().__init__(f"{code}: {detail}")


def _bind_process_failure(
    error: BaseException,
    *,
    role: str,
    command: tuple[str, ...],
    environment: dict[str, str],
    tool_sha256: str,
) -> None:
    if isinstance(error, NcuActionExecutorError):
        error.process_role = role
        error.process_command = command
        error.process_environment = dict(environment)
        error.process_tool_sha256 = tool_sha256


def _copy_process_failure(
    target: NcuActionExecutorError,
    source: NcuActionExecutorError,
) -> None:
    target.stdout = source.stdout
    target.stderr = source.stderr
    target.process_id = source.process_id
    target.started_at_utc = source.started_at_utc
    target.process_role = source.process_role
    target.process_command = source.process_command
    target.process_environment = (
        None
        if source.process_environment is None
        else dict(source.process_environment)
    )
    target.process_tool_sha256 = source.process_tool_sha256
    target.containment = source.containment
    target.process_cleanup_incomplete = (
        target.process_cleanup_incomplete or source.process_cleanup_incomplete
    )


def _copy_process_result(
    target: NcuActionExecutorError,
    source: NcuActionResult,
) -> None:
    target.stdout = source.stdout
    target.stderr = source.stderr
    target.process_id = source.process_id
    target.started_at_utc = source.started_at_utc
    target.containment = source.containment


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
        elif type(item) is _AuthorizedExecutableProcPath:
            if index == 0:
                _fail(
                    NCU_ACTION_REQUEST_INVALID,
                    "authorized auxiliary path cannot be the supervised executable",
                )
            canonical_path = item.canonical_path
            if (
                type(canonical_path) is not str
                or not canonical_path
                or "\x00" in canonical_path
                or canonical_path.splitlines() != [canonical_path]
                or _strict_utf8_size(
                    canonical_path,
                    code=NCU_ACTION_REQUEST_INVALID,
                    label="authorized auxiliary canonical path",
                ) > _MAX_PATH_BYTES
                or not Path(canonical_path).is_absolute()
                or canonical_path.startswith("//")
                or os.path.normpath(canonical_path) != canonical_path
                or "/proc/self/fd/" in canonical_path
            ):
                _fail(
                    NCU_ACTION_REQUEST_INVALID,
                    "authorized auxiliary path lacks a canonical durable path",
                )
            _validate_authorized_executable_proc_path(item)
            rendered.append(proc_fd_path(item.descriptor))
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
        original
        if (
            original is AUTHORIZED_NCU_PROC_PATH
            or type(original) is _AuthorizedExecutableProcPath
        )
        else value
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
        retained_path
        if item is AUTHORIZED_NCU_PROC_PATH
        else (
            proc_fd_path(item.descriptor)
            if type(item) is _AuthorizedExecutableProcPath
            else item
        )
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


def _validate_authorized_executable_proc_path(
    value: _AuthorizedExecutableProcPath,
) -> None:
    if value._seal is not _AUTHORIZED_EXECUTABLE_PROC_PATH_SEAL:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "authorized auxiliary path lacks retained-capability authority",
        )
    descriptor_identity = _descriptor_identity(value.descriptor)
    if descriptor_identity != value.descriptor_identity:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "authorized auxiliary descriptor identity changed",
        )
    try:
        path_metadata = os.stat(value.canonical_path, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            f"authorized auxiliary canonical path is unavailable: {exc}",
        ) from exc
    path_identity = (
        path_metadata.st_dev,
        path_metadata.st_ino,
        stat.S_IFMT(path_metadata.st_mode),
    )
    if path_identity != descriptor_identity or not stat.S_ISREG(path_metadata.st_mode):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "authorized auxiliary canonical path differs from its descriptor",
        )


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


def authorized_executable_proc_path(
    retained: immutable.RetainedExecutable,
) -> _AuthorizedExecutableProcPath:
    """Bind a caller-retained executable to runtime and durable argv forms."""
    if type(retained) is not immutable.RetainedExecutable:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "authorized auxiliary executable is not an exact retained capability",
        )
    try:
        retained.verify_unchanged()
        descriptor = retained.descriptor
    except immutable.ImmutableArtifactError as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_REQUEST_INVALID,
            f"authorized auxiliary executable is unavailable: {exc}",
        ) from exc
    canonical_path = str(retained.path)
    if (
        not Path(canonical_path).is_absolute()
        or canonical_path.startswith("//")
        or os.path.normpath(canonical_path) != canonical_path
        or "/proc/self/fd/" in canonical_path
    ):
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "authorized auxiliary executable path is not canonical",
        )
    descriptor_identity = _descriptor_identity(descriptor)
    return _AuthorizedExecutableProcPath(
        descriptor,
        canonical_path,
        descriptor_identity,
        _AUTHORIZED_EXECUTABLE_PROC_PATH_SEAL,
    )


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


def _process_exited_unreaped(process: subprocess.Popen[bytes]) -> bool:
    try:
        status = os.waitid(
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_LAUNCH_FAILED,
            "retained NCU process authority was released before cleanup",
        ) from exc
    except OSError as exc:
        raise NcuActionExecutorError(
            NCU_ACTION_LAUNCH_FAILED,
            f"cannot observe retained NCU process without reaping it: {exc}",
        ) from exc
    return status is not None


def _wait_for_exit_unreaped(
    process: subprocess.Popen[bytes], timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _process_exited_unreaped(process):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def _stop_and_reap(
    process: subprocess.Popen[bytes],
    containment: delegated_cgroup.DelegatedCgroupV2 | None = None,
) -> None:
    termination_error: BaseException | None = None
    try:
        _signal_group(process, signal.SIGTERM)
        if not _wait_for_exit_unreaped(process, _TERMINATION_GRACE_SECONDS):
            try:
                if containment is None:
                    _signal_group(process, signal.SIGKILL)
                else:
                    containment.kill_all()
            except BaseException as exc:
                termination_error = exc
            else:
                if not _wait_for_exit_unreaped(
                    process, _TERMINATION_GRACE_SECONDS
                ):
                    termination_error = NcuActionExecutorError(
                        NCU_ACTION_LAUNCH_FAILED,
                        "retained NCU process did not exit after forced termination",
                        process_cleanup_incomplete=True,
                    )
        else:
            # Keep the exited leader as PID/PGID authority while removing
            # descendants, but never let descendant-cleanup failure skip reaping.
            try:
                if containment is None:
                    _signal_group(process, signal.SIGKILL)
                else:
                    containment.kill_all()
            except BaseException as exc:
                termination_error = exc
    except BaseException as exc:
        termination_error = exc

    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except BaseException as reap_error:
        failure = NcuActionExecutorError(
            NCU_ACTION_LAUNCH_FAILED,
            f"cannot reap retained NCU process: {reap_error}",
            process_cleanup_incomplete=True,
        )
        if termination_error is not None:
            try:
                failure.add_note(
                    f"retained NCU descendant cleanup also failed: "
                    f"{termination_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise failure from reap_error

    if termination_error is not None:
        raise NcuActionExecutorError(
            NCU_ACTION_LAUNCH_FAILED,
            f"retained NCU descendant cleanup failed: {termination_error}",
            process_cleanup_incomplete=True,
        ) from termination_error


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


def _new_bounded_output_buffers() -> dict[str, bytearray]:
    return {"stdout": bytearray(), "stderr": bytearray()}


def _copy_bounded_stream_prefix(buffer: bytearray, length: int) -> bytes:
    if length == len(buffer):
        return bytes(buffer)
    return bytes(memoryview(buffer)[:length])


def _materialize_bounded_stream(
    buffer: bytearray,
) -> tuple[bytes, MemoryError | None]:
    """Copy one stream, retaining a safe exact prefix after allocation failure."""

    first_error: MemoryError | None = None
    candidate_length = len(buffer)
    retry_full = True
    while candidate_length > 0:
        try:
            return _copy_bounded_stream_prefix(buffer, candidate_length), first_error
        except MemoryError as exc:
            if first_error is None:
                first_error = exc
            if retry_full:
                retry_full = False
            else:
                candidate_length //= 2
    return b"", first_error


def _collect_bounded(
    process: subprocess.Popen[bytes],
    *,
    started_at_utc: str,
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_code: str,
    containment: delegated_cgroup.DelegatedCgroupV2 | None = None,
    deadline_monotonic: float | None = None,
) -> NcuActionResult:
    # Popen has already succeeded.  Keep every allocation and output copy under
    # this cleanup guard so MemoryError cannot strand the retained process.
    selector: selectors.BaseSelector | None = None
    buffers: dict[str, bytearray] | None = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    close_error: BaseException | None = None
    returncode: int | None = None
    process_reaped = False
    try:
        buffers = _new_bounded_output_buffers()
        total = 0
        deadline = (
            time.monotonic() + timeout_seconds
            if deadline_monotonic is None
            else deadline_monotonic
        )
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
        child_exit_observed = False
        while selector.get_map():
            if not child_exit_observed and _process_exited_unreaped(process):
                # Preserve the exited leader as PID/PGID authority until every
                # same-group descendant has been synchronously terminated.
                if containment is None:
                    _signal_group(process, signal.SIGKILL)
                else:
                    containment.kill_all()
                child_exit_observed = True
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
                if remaining_capacity > 0:
                    retained_chunk = (
                        chunk
                        if len(chunk) <= remaining_capacity
                        else chunk[:remaining_capacity]
                    )
                    buffers[key.data].extend(retained_chunk)
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
        if not child_exit_observed:
            if not _wait_for_exit_unreaped(process, remaining):
                raise NcuActionExecutorError(
                    timeout_code,
                    f"retained NCU action exceeded {timeout_seconds} seconds",
                )
            if containment is None:
                _signal_group(process, signal.SIGKILL)
            else:
                containment.kill_all()
        returncode = process.wait()
        process_reaped = True
    except BaseException as active_error:
        primary_error = active_error
    finally:
        if not process_reaped:
            try:
                _stop_and_reap(process, containment)
                process_reaped = True
            except BaseException as exc:
                cleanup_error = exc
                # A transient allocation/resource failure inside teardown must
                # not be the only stop/reap attempt.
                try:
                    _stop_and_reap(process, containment)
                    process_reaped = True
                except BaseException as retry_error:
                    try:
                        cleanup_error.add_note(
                            f"retained NCU process cleanup retry failed: {retry_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                else:
                    cleanup_error = None
        try:
            close_errors = _close_monitor_resources(selector, process)
        except BaseException as exc:
            close_error = exc
        else:
            if close_errors:
                try:
                    detail = "; ".join(str(error) for error in close_errors)
                    close_error = NcuActionExecutorError(
                        NCU_ACTION_LAUNCH_FAILED,
                        "cannot close retained NCU process resources: " + detail,
                    )
                    close_error.__cause__ = close_errors[0]
                except BaseException as exc:
                    close_error = exc

    stdout = b""
    stderr = b""
    materialization_error: MemoryError | None = None
    if buffers is not None:
        stdout, stdout_error = _materialize_bounded_stream(buffers["stdout"])
        stderr, stderr_error = _materialize_bounded_stream(buffers["stderr"])
        materialization_error = stdout_error or stderr_error

    if cleanup_error is not None:
        cleanup_failure = NcuActionExecutorError(
            NCU_ACTION_LAUNCH_FAILED,
            f"retained NCU process cleanup failed: {cleanup_error}",
            process_cleanup_incomplete=True,
        )
        cleanup_failure.stdout = stdout
        cleanup_failure.stderr = stderr
        cleanup_failure.process_id = process.pid
        cleanup_failure.started_at_utc = started_at_utc
        if primary_error is not None:
            try:
                cleanup_failure.add_note(
                    f"active retained NCU process error: {primary_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise cleanup_failure from cleanup_error

    if materialization_error is not None:
        output_failure = NcuActionExecutorError(
            NCU_ACTION_LAUNCH_FAILED,
            "retained NCU bounded output materialization exhausted memory",
        )
        output_failure.stdout = stdout
        output_failure.stderr = stderr
        output_failure.process_id = process.pid
        output_failure.started_at_utc = started_at_utc
        if primary_error is not None:
            try:
                output_failure.add_note(
                    f"active retained NCU process error: {primary_error}"
                )
            except (AttributeError, TypeError):
                pass
        if close_error is not None:
            try:
                output_failure.add_note(
                    f"process resource closure failed: {close_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise output_failure from materialization_error

    if close_error is not None:
        if primary_error is not None:
            try:
                primary_error.add_note(
                    f"process resource closure failed: {close_error}"
                )
            except (AttributeError, TypeError):
                pass
        else:
            closure_failure = (
                close_error
                if isinstance(close_error, NcuActionExecutorError)
                else NcuActionExecutorError(
                    NCU_ACTION_LAUNCH_FAILED,
                    f"cannot close retained NCU process resources: {close_error}",
                )
            )
            closure_failure.stdout = stdout
            closure_failure.stderr = stderr
            closure_failure.process_id = process.pid
            closure_failure.started_at_utc = started_at_utc
            raise closure_failure from close_error

    if primary_error is not None:
        if isinstance(primary_error, NcuActionExecutorError):
            primary_error.stdout = stdout
            primary_error.stderr = stderr
            primary_error.process_id = process.pid
            primary_error.started_at_utc = started_at_utc
            raise primary_error
        if isinstance(primary_error, MemoryError):
            resource_failure = NcuActionExecutorError(
                NCU_ACTION_LAUNCH_FAILED,
                "retained NCU bounded output collection exhausted memory",
            )
            resource_failure.stdout = stdout
            resource_failure.stderr = stderr
            resource_failure.process_id = process.pid
            resource_failure.started_at_utc = started_at_utc
            raise resource_failure from primary_error
        raise primary_error

    assert returncode is not None
    return NcuActionResult(
        returncode,
        stdout,
        stderr,
        process_id=process.pid,
        started_at_utc=started_at_utc,
    )


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
    require_cgroup_containment: bool = False,
) -> NcuActionResult:
    started_at_utc = (
        dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    deadline_monotonic = time.monotonic() + timeout_seconds
    containment: delegated_cgroup.DelegatedCgroupV2 | None = None
    process: subprocess.Popen[bytes] | None = None
    popen_arguments = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": environment,
        "cwd": working_directory,
        "pass_fds": inherited_descriptors,
        "start_new_session": True,
    }
    try:
        with nvbit_capture_process.bounded_launch(deadline_monotonic):
            if require_cgroup_containment:
                try:
                    containment = delegated_cgroup.DelegatedCgroupV2.create()
                except delegated_cgroup.CgroupContainmentError as exc:
                    raise NcuActionExecutorError(
                        launch_code,
                        f"cannot establish delegated cgroup containment: {exc}",
                    ) from exc
            process = (
                subprocess.Popen(command, **popen_arguments)
                if containment is None
                else containment.popen(command, **popen_arguments)
            )
    except BaseException as active_error:
        cleanup_errors: list[BaseException] = []
        if process is not None:
            try:
                _stop_and_reap(process, containment)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                cleanup_errors.extend(
                    _close_monitor_resources(None, process)
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if containment is not None:
            try:
                if process is None:
                    containment.close_after_error()
                else:
                    containment.finish()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            cleanup_failure = NcuActionExecutorError(
                launch_code,
                "supervised process launch cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors),
                process_cleanup_incomplete=True,
            )
            cleanup_failure.process_id = (
                None if process is None else process.pid
            )
            cleanup_failure.started_at_utc = started_at_utc
            try:
                cleanup_failure.add_note(
                    f"active supervised process launch error: {active_error}"
                )
            except (AttributeError, TypeError):
                pass
            raise cleanup_failure from cleanup_errors[0]
        if isinstance(
            active_error, nvbit_capture_process.CaptureLaunchTimeout
        ):
            raise NcuActionExecutorError(
                timeout_code,
                f"supervised process launch exceeded {timeout_seconds} seconds",
            ) from active_error
        if not isinstance(active_error, Exception):
            raise
        if isinstance(active_error, NcuActionExecutorError):
            raise
        raise NcuActionExecutorError(
            launch_code,
            f"cannot launch supervised process: {active_error}",
        ) from active_error

    assert process is not None
    try:
        result = _collect_bounded(
            process,
            started_at_utc=started_at_utc,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            timeout_code=timeout_code,
            containment=containment,
            deadline_monotonic=deadline_monotonic,
        )
    except BaseException as active_error:
        if isinstance(active_error, NcuActionExecutorError):
            process_error = active_error
        else:
            process_error = NcuActionExecutorError(
                launch_code,
                f"supervised process execution failed: {active_error}",
            )
            process_error.process_id = process.pid
            process_error.started_at_utc = started_at_utc
        if containment is not None:
            try:
                process_error.containment = containment.finish()
            except BaseException as cleanup_error:
                wrapped = NcuActionExecutorError(
                    launch_code,
                    f"delegated cgroup cleanup failed: {cleanup_error}",
                    process_cleanup_incomplete=True,
                )
                _copy_process_failure(wrapped, process_error)
                try:
                    wrapped.add_note(f"active process error: {active_error}")
                except (AttributeError, TypeError):
                    pass
                raise wrapped from cleanup_error
        if not isinstance(active_error, Exception):
            raise
        if not isinstance(active_error, NcuActionExecutorError):
            raise process_error from active_error
        if active_error.code in {NCU_ACTION_TIMEOUT, NCU_ACTION_OUTPUT_LIMIT}:
            raise
        if active_error.code == timeout_code:
            raise
        wrapped = NcuActionExecutorError(launch_code, active_error.detail)
        _copy_process_failure(wrapped, active_error)
        raise wrapped from active_error
    if containment is not None:
        try:
            containment_evidence = containment.finish()
        except BaseException as cleanup_error:
            error = NcuActionExecutorError(
                launch_code,
                f"delegated cgroup cleanup failed: {cleanup_error}",
                process_cleanup_incomplete=True,
            )
            _copy_process_result(error, result)
            raise error from cleanup_error
        result = NcuActionResult(
            result.returncode,
            result.stdout,
            result.stderr,
            process_id=result.process_id,
            started_at_utc=result.started_at_utc,
            containment=containment_evidence,
        )
    return result


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
    require_cgroup_containment: bool = False,
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
        require_cgroup_containment=require_cgroup_containment,
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
    require_cgroup_containment: bool = False,
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
        require_cgroup_containment=require_cgroup_containment,
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
                _copy_process_failure(primary, pending)
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
                _copy_process_failure(primary, pending)
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
    require_cgroup_containment: bool = False,
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
            require_cgroup_containment=require_cgroup_containment,
        )
    except BaseException as exc:
        _reverify_authorities(
            retained,
            borrowed,
            retained_supervised_executable,
            pending=exc,
        )
        raise AssertionError("unreachable")
    try:
        _reverify_authorities(
            retained, borrowed, retained_supervised_executable,
        )
    except NcuActionExecutorError as exc:
        _copy_process_result(exc, result)
        raise
    return result


def _authorize_opened(
    retained: immutable.RetainedExecutable,
    expected: ExpectedNcuIdentity,
    borrowed: _BorrowedActionDescriptors,
    *,
    retained_supervised_executable: immutable.RetainedExecutable | None = None,
    require_cgroup_containment: bool = False,
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
    try:
        probe = _run_and_reverify(
            retained,
            borrowed,
            ("--version",),
            retained_supervised_executable=retained_supervised_executable,
            timeout_seconds=_VERSION_TIMEOUT_SECONDS,
            max_output_bytes=_MAX_VERSION_OUTPUT_BYTES,
            timeout_code=NCU_VERSION_PROBE_FAILED,
            launch_code=NCU_VERSION_PROBE_FAILED,
            require_cgroup_containment=require_cgroup_containment,
        )
    except BaseException as exc:
        _bind_process_failure(
            exc,
            role="ncu_version_probe",
            command=(expected.resolved_path, "--version"),
            environment=ncu_action_environment(),
            tool_sha256=expected.executable_sha256,
        )
        raise
    if probe.returncode != 0:
        error = NcuActionExecutorError(
            NCU_VERSION_PROBE_FAILED,
            f"retained NCU --version exited {probe.returncode}",
        )
        error.stdout = probe.stdout
        error.stderr = probe.stderr
        error.process_id = probe.process_id
        error.started_at_utc = probe.started_at_utc
        _bind_process_failure(
            error,
            role="ncu_version_probe",
            command=(expected.resolved_path, "--version"),
            environment=ncu_action_environment(),
            tool_sha256=expected.executable_sha256,
        )
        _reverify_authorities(
            retained,
            borrowed,
            retained_supervised_executable,
            pending=error,
        )
    decoded = probe.stdout.decode("utf-8", errors="surrogateescape")
    version_lines = [
        line.strip()
        for line in decoded.splitlines()
        if re.match(r"^\s*Version\s+", line)
    ]
    if len(version_lines) != 1 or version_lines[0] != expected.version_line:
        error = NcuActionExecutorError(
            NCU_VERSION_LINE_MISMATCH,
            "retained NCU version line differs from independent authority",
        )
        error.stdout = probe.stdout
        error.stderr = probe.stderr
        error.process_id = probe.process_id
        error.started_at_utc = probe.started_at_utc
        _bind_process_failure(
            error,
            role="ncu_version_probe",
            command=(expected.resolved_path, "--version"),
            environment=ncu_action_environment(),
            tool_sha256=expected.executable_sha256,
        )
        _reverify_authorities(
            retained,
            borrowed,
            retained_supervised_executable,
            pending=error,
        )
    try:
        _reverify_authorities(
            retained, borrowed, retained_supervised_executable,
        )
    except NcuActionExecutorError as exc:
        _copy_process_result(exc, probe)
        _bind_process_failure(
            exc,
            role="ncu_version_probe",
            command=(expected.resolved_path, "--version"),
            environment=ncu_action_environment(),
            tool_sha256=expected.executable_sha256,
        )
        raise


def _execute_opened(
    retained: immutable.RetainedExecutable,
    expected: ExpectedNcuIdentity,
    action: tuple[str, ...],
    timeout_seconds: float,
    borrowed: _BorrowedActionDescriptors,
    *,
    require_cgroup_containment: bool,
) -> NcuActionResult:
    _authorize_opened(
        retained,
        expected,
        borrowed,
        require_cgroup_containment=require_cgroup_containment,
    )
    return _run_and_reverify(
        retained,
        borrowed,
        action,
        inherited_descriptors=borrowed.inherited,
        timeout_seconds=timeout_seconds,
        max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
        timeout_code=NCU_ACTION_TIMEOUT,
        launch_code=NCU_ACTION_LAUNCH_FAILED,
        require_cgroup_containment=require_cgroup_containment,
    )


def execute_retained_ncu_version_probe(
    retained: immutable.RetainedExecutable,
    *,
    require_cgroup_containment: bool = False,
) -> NcuActionResult:
    """Run ``--version`` on one caller-owned retained NCU candidate."""

    if type(retained) is not immutable.RetainedExecutable:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "retained NCU version probe requires an exact executable capability",
        )
    if type(require_cgroup_containment) is not bool:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "cgroup containment requirement is not boolean",
        )
    try:
        retained.verify_unchanged()
        writable = os.access(
            f"/proc/self/fd/{retained.descriptor}",
            os.W_OK,
            effective_ids=True,
        )
    except (OSError, TypeError, immutable.ImmutableArtifactError) as exc:
        raise _identity_error(
            "cannot validate retained NCU version candidate", exc,
        ) from exc
    if writable:
        _fail(
            NCU_EXECUTABLE_IDENTITY_INVALID,
            "retained NCU version candidate is writable by the probe process",
        )
    borrowed = _BorrowedActionDescriptors(None, (), (), ())
    try:
        result = _run_and_reverify(
            retained,
            borrowed,
            ("--version",),
            timeout_seconds=_VERSION_TIMEOUT_SECONDS,
            max_output_bytes=_MAX_VERSION_OUTPUT_BYTES,
            timeout_code=NCU_VERSION_PROBE_FAILED,
            launch_code=NCU_VERSION_PROBE_FAILED,
            require_cgroup_containment=require_cgroup_containment,
        )
    except BaseException as exc:
        _bind_process_failure(
            exc,
            role="ncu_identity_version_probe",
            command=(str(retained.path), "--version"),
            environment=ncu_action_environment(),
            tool_sha256=retained.sha256,
        )
        raise
    return result


def execute_ncu_action(
    expected: ExpectedNcuIdentity,
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
    retained_report: immutable.RetainedRegular | None = None,
    auxiliary_descriptors: Sequence[int] = (),
    require_cgroup_containment: bool = False,
) -> NcuActionResult:
    """Run one bounded NCU action through independently retained capabilities."""
    if type(require_cgroup_containment) is not bool:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "cgroup containment requirement is not boolean",
        )
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
            retained,
            expected,
            action,
            timeout_seconds,
            borrowed,
            require_cgroup_containment=require_cgroup_containment,
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


def execute_ncu_supervised_executable(
    expected: ExpectedNcuIdentity,
    expected_executable: ExpectedExecutableIdentity,
    command: Sequence[object],
    *,
    timeout_seconds: float,
    environment: NcuCollectionEnvironment,
    working_directory: Path = Path("/"),
    auxiliary_descriptors: Sequence[int] = (),
    require_cgroup_containment: bool = False,
) -> NcuActionResult:
    """Run a descriptor-bound child with one descriptor-bound NCU authority."""

    if type(require_cgroup_containment) is not bool:
        _fail(
            NCU_ACTION_REQUEST_INVALID,
            "cgroup containment requirement is not boolean",
        )
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

    borrowed_command = tuple(
        "authorized-ncu-descriptor"
        if item is AUTHORIZED_NCU_PROC_PATH
        else (
            proc_fd_path(item.descriptor)
            if type(item) is _AuthorizedExecutableProcPath
            else str(item)
        )
        for item in template
    )
    borrowed = _validate_borrowed_descriptors(
        borrowed_command,
        None,
        auxiliary_descriptors,
    )
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
            require_cgroup_containment=require_cgroup_containment,
        )
        rendered = _render_supervised_executable_command(
            template, retained_ncu, retained_executable,
        )
        canonical_command = tuple(
            expected.resolved_path
            if item is AUTHORIZED_NCU_PROC_PATH
            else (
                item.canonical_path
                if type(item) is _AuthorizedExecutableProcPath
                else str(item)
            )
            for item in template
        )
        try:
            result = _run_command_bounded(
                rendered,
                environment=environment.as_dict(),
                working_directory=working_directory_text,
                inherited_descriptors=(
                    retained_ncu.descriptor,
                    retained_executable.descriptor,
                    *borrowed.inherited,
                ),
                timeout_seconds=timeout_seconds,
                max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
                timeout_code=NCU_ACTION_TIMEOUT,
                launch_code=NCU_ACTION_LAUNCH_FAILED,
                require_cgroup_containment=require_cgroup_containment,
            )
        except BaseException as action_error:
            _bind_process_failure(
                action_error,
                role="ncu_supervised_executable",
                command=canonical_command,
                environment=environment.as_dict(),
                tool_sha256=expected_executable.executable_sha256,
            )
            _reverify_with_supervised_executable(
                retained_ncu,
                borrowed,
                retained_executable,
                pending=action_error,
            )
            raise AssertionError("unreachable")
        try:
            _reverify_with_supervised_executable(
                retained_ncu, borrowed, retained_executable,
            )
        except NcuActionExecutorError as action_error:
            _copy_process_result(action_error, result)
            _bind_process_failure(
                action_error,
                role="ncu_supervised_executable",
                command=canonical_command,
                environment=environment.as_dict(),
                tool_sha256=expected_executable.executable_sha256,
            )
            raise
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
        error = _supervised_executable_identity_error(
            "cannot close retained supervised executable", exc,
        )
        _copy_process_result(error, result)
        _bind_process_failure(
            error,
            role="ncu_supervised_executable",
            command=canonical_command,
            environment=environment.as_dict(),
            tool_sha256=expected_executable.executable_sha256,
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
        retained_ncu.close()
    except BaseException as exc:
        error = _identity_error("cannot close retained executable", exc)
        _copy_process_result(error, result)
        _bind_process_failure(
            error,
            role="ncu_supervised_executable",
            command=canonical_command,
            environment=environment.as_dict(),
            tool_sha256=expected_executable.executable_sha256,
        )
        raise error from exc
    return result
