"""External runtime recorder for the repository-owned release ``sim-run``.

The active lane records exploratory current-simulator evidence only. It never
builds the executable, imports label artifacts, invokes a baseline producer, or
mints a claimable dashboard row without the missing workload-tier authority.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from jsonschema import ValidationError
from jsonschema.validators import validator_for
from referencing import Registry, Resource

from eval.contracts import launch_crosswalk as _launch_crosswalk
from eval.contracts._common import EvaluatorAuthorityUnavailable
from eval.contracts.model_seal import RetainedPayloadSnapshot
from eval.contracts.validation import (
    ContractValidationError,
    MatchedLabelAuthority,
    canonical_launch_key_token,
    canonical_sha256,
    load_json_file_strict,
    loads_json_strict,
    schema_document,
    validate_manifest,
    validate_report,
    validate_result,
)
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    ImmutableFile,
    canonical_json_bytes,
    inspect_immutable_directory_generation,
    open_retained_regular,
    publish_directory,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCHEMA_PATH = ROOT / "eval/schemas/sim_runtime_record.v1.schema.json"
RUNTIME_CLAIM_SCHEMA_PATH = (
    ROOT / "eval/schemas/runtime_workload_tier_claim_authority.v1.schema.json"
)
WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_PATH = (
    ROOT / "eval/schemas/workload_execution_authority.v1.schema.json"
)
RUNTIME_CLAIM_SCHEMA_SHA256 = (
    "47d8e081a745a5cead98374e9b6ec4129e04a683d3b346cf7389a264d9208557"
)
MAX_RUNTIME_CLAIM_BYTES = 4 * 1024 * 1024
MAX_CLAIM_AUTHORITY_BYTES = 16 * 1024 * 1024
MAX_ACCURACY_REPORT_BYTES = 64 * 1024 * 1024
MAX_CONTENT_MANIFEST_BYTES = 4 * 1024 * 1024
RELEASE_BINARY_RELATIVE = Path("sim/target/release/sim-run")
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_RESULT_STREAM_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_RECORD_BYTES = 16 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 1800.0
_PIPE_DRAIN_TIMEOUT_NS = 5_000_000_000
INITIAL_TIMED_REPEATS = 5
ADDITIONAL_NOISY_REPEATS = 4
CV_THRESHOLD = Fraction(1, 20)
_ENVIRONMENT_ITEMS = (
    ("LANG", "C"),
    ("LC_ALL", "C"),
    ("NO_COLOR", "1"),
    ("RUST_BACKTRACE", "0"),
    ("TZ", "UTC"),
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_RETAINED_MANIFEST_ARGUMENT_PREFIX = "--retained-manifest-fds-v1:"
_MAX_INHERITED_DESCRIPTOR = 2_147_483_647
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_CLAIM_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_CLAIM_FILE_ROLES = (
    "accuracy_report",
    "evaluator_authority",
    "runtime_record",
    "source_authority",
    "workload_execution_authority",
)
_CLAIM_BASIS = (
    "clean_committed_release_measured_admitted_tier_sealed_result_and_accuracy_v1"
)
_FIXTURE_CLAIM_REASON = "FIXTURE_ONLY_NON_AUTHORITATIVE"
_CLAIM_EVIDENCE_SUFFIX = ".evidence"
_CLAIM_PRODUCER_RELATIVE = Path("eval/bench.py")
_CLAIM_SCHEMA_RELATIVE = Path(
    "eval/schemas/runtime_workload_tier_claim_authority.v1.schema.json"
)
_CLAIM_EVIDENCE_FIXED_NAMES = {
    "runtime_record": "runtime_record.json",
    "runtime_binary": "runtime_binary",
    "runtime_manifest": "runtime_manifest.json",
    "runtime_result": "runtime_result.json",
    "workload_execution_authority": "workload_execution_authority.json",
    "evaluator_authority": "evaluator_authority.json",
    "accuracy_report": "accuracy_report.json",
    "content_manifest": "external_content_manifest.json",
    "source_authority": "source_authority.json",
    "claim_producer": "claim_producer.py",
    "claim_schema": "claim_schema.json",
    "workload_schema": "workload_authority_schema.json",
}

# Independent trust registries are immutable and intentionally empty for current
# production evidence. Caller-provided self-seals can exercise only fixture algebra.
PRODUCTION_RUNTIME_WORKLOAD_AUTHORITIES = MappingProxyType({})
PRODUCTION_RUNTIME_EVALUATOR_AUTHORITIES = MappingProxyType({})
PRODUCTION_RUNTIME_SOURCE_AUTHORITIES = MappingProxyType({})
PRODUCTION_RUNTIME_CONTENT_MANIFEST_PINS = MappingProxyType({})


class RuntimeRecorderError(RuntimeError):
    """Typed failure before a sealed runtime record can be published."""

    def __init__(
        self, code: str, detail: str, *, context: Mapping[str, Any] | None = None
    ) -> None:
        self.code = code
        self.detail = detail
        self.context = dict(context or {})
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class _StatIdentity:
    device: int
    inode: int
    mode: int
    link_count: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(slots=True)
class _RetainedFile:
    path: Path
    parent_fd: int
    name: str
    fd: int
    maximum_bytes: int
    identity: _StatIdentity
    sha256: str
    require_single_link: bool

    @property
    def size(self) -> int:
        return self.identity.size

    def revalidate(self, *, role: str) -> None:
        def observe() -> tuple[os.stat_result, os.stat_result, os.stat_result, os.stat_result]:
            reopened_parent_fd: int | None = None
            try:
                reopened_parent_fd = _open_directory_no_symlinks(
                    self.path.parent,
                    role=f"{role} parent",
                    error_code="RUNTIME_INPUT_DRIFT",
                )
                return (
                    os.fstat(self.parent_fd),
                    os.fstat(reopened_parent_fd),
                    os.stat(
                        self.name,
                        dir_fd=self.parent_fd,
                        follow_symlinks=False,
                    ),
                    os.fstat(self.fd),
                )
            except RuntimeRecorderError:
                raise
            except OSError as exc:
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_DRIFT",
                    f"{role} identity became unreadable",
                    context={"path": str(self.path), "errno": exc.errno},
                ) from exc
            finally:
                if reopened_parent_fd is not None:
                    os.close(reopened_parent_fd)

        def require_snapshot(
            retained_parent_state: os.stat_result,
            reopened_parent_state: os.stat_result,
            path_state: os.stat_result,
            descriptor_state: os.stat_result,
        ) -> None:
            if not _same_filesystem_object(
                retained_parent_state, reopened_parent_state
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_DRIFT",
                    f"{role} parent directory identity changed",
                    context={"path": str(self.path.parent)},
                )
            if not stat.S_ISREG(path_state.st_mode) or not stat.S_ISREG(
                descriptor_state.st_mode
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_DRIFT",
                    f"{role} is no longer a regular file",
                    context={"path": str(self.path)},
                )
            if _stat_identity(path_state) != self.identity:
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_DRIFT",
                    f"{role} path identity changed",
                    context={"path": str(self.path)},
                )
            if _stat_identity(descriptor_state) != self.identity:
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_DRIFT",
                    f"retained {role} descriptor identity changed",
                    context={"path": str(self.path)},
                )
            if self.require_single_link and descriptor_state.st_nlink != 1:
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_DRIFT",
                    f"{role} gained a hardlink alias",
                    context={"path": str(self.path)},
                )

        require_snapshot(*observe())
        observed = _hash_fd(self.fd, self.size, role=role)
        require_snapshot(*observe())
        if observed != self.sha256:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_DRIFT",
                f"retained {role} bytes changed",
                context={"path": str(self.path)},
            )

    def read_bytes(self, *, role: str) -> bytes:
        self.revalidate(role=role)
        payload = _read_fd(self.fd, self.size, role=role)
        self.revalidate(role=role)
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_DRIFT",
                f"retained {role} bytes changed during read",
                context={"path": str(self.path)},
            )
        return payload

    def close(self) -> None:
        os.close(self.fd)
        os.close(self.parent_fd)


@dataclass(slots=True)
class _StreamDrain:
    maximum_bytes: int
    payload: bytearray
    complete: bool = True
    error: str | None = None

    def consume(self, fd: int) -> bool:
        """Consume currently available bytes and return whether EOF was observed."""

        while True:
            try:
                chunk = os.read(fd, 64 * 1024)
            except BlockingIOError:
                return False
            except OSError as exc:  # pragma: no cover - OS stream failures are rare.
                self.complete = False
                self.error = f"OSError:{exc.errno}"
                return True
            if not chunk:
                return True
            allowance = self.maximum_bytes - len(self.payload)
            if allowance > 0:
                self.payload.extend(chunk[:allowance])
            if len(chunk) > allowance:
                self.complete = False
                return False


@dataclass(frozen=True, slots=True)
class _SpawnObservation:
    background_load: tuple[float, float, float]
    monotonic_start_ns: int
    monotonic_stop_ns: int
    user_cpu_ns: int
    system_cpu_ns: int
    peak_rss_kib: int
    return_code: int
    timed_out: bool
    stdout: bytes
    stdout_complete: bool
    stderr: bytes
    stderr_complete: bool
    capture_error: str | None


@dataclass(slots=True)
class _Publisher:
    output_path: Path
    parent_fd: int
    parent_identity: _StatIdentity
    sample_directory_name: str
    sample_directory_fd: int
    sample_directory_identity: _StatIdentity

    @classmethod
    def create(cls, output_path: str | Path) -> _Publisher:
        requested = Path(output_path)
        absolute = Path(os.path.abspath(os.fspath(requested)))
        name = absolute.name
        if name in {"", ".", ".."} or len(name.encode("utf-8")) > 180:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_PATH_INVALID", "runtime record file name is invalid"
            )
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        parent_fd = _open_directory_no_symlinks(
            absolute.parent,
            role="runtime record parent",
            error_code="RUNTIME_OUTPUT_PARENT_INVALID",
        )
        sample_name = f"{name}.samples"
        sample_fd: int | None = None
        try:
            parent_state = os.fstat(parent_fd)
            reopened_parent_fd = _open_directory_no_symlinks(
                absolute.parent,
                role="runtime record parent",
                error_code="RUNTIME_OUTPUT_PARENT_INVALID",
            )
            try:
                parent_path_state = os.fstat(reopened_parent_fd)
            finally:
                os.close(reopened_parent_fd)
            if not stat.S_ISDIR(parent_state.st_mode) or not _same_filesystem_object(
                parent_state,
                parent_path_state,
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_OUTPUT_PARENT_INVALID",
                    "runtime record parent path and descriptor disagree",
                )
            if _entry_exists(parent_fd, name) or _entry_exists(parent_fd, sample_name):
                raise RuntimeRecorderError(
                    "RUNTIME_OUTPUT_EXISTS",
                    "runtime record output generation already exists",
                    context={"path": str(absolute)},
                )
            os.mkdir(sample_name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            sample_fd = os.open(
                sample_name,
                os.O_RDONLY | os.O_CLOEXEC | nofollow | directory,
                dir_fd=parent_fd,
            )
            sample_state = os.fstat(sample_fd)
            sample_path_state = os.stat(
                sample_name, dir_fd=parent_fd, follow_symlinks=False
            )
            if not stat.S_ISDIR(sample_state.st_mode) or _stat_identity(
                sample_state
            ) != _stat_identity(sample_path_state):
                raise RuntimeRecorderError(
                    "RUNTIME_OUTPUT_PARENT_INVALID",
                    "runtime sample directory path and descriptor disagree",
                )
            parent_identity = _stat_identity(os.fstat(parent_fd))
            reopened_parent_fd = _open_directory_no_symlinks(
                absolute.parent,
                role="runtime record parent",
                error_code="RUNTIME_OUTPUT_PARENT_INVALID",
            )
            try:
                reopened_parent_identity = _stat_identity(os.fstat(reopened_parent_fd))
            finally:
                os.close(reopened_parent_fd)
            if parent_identity != reopened_parent_identity:
                raise RuntimeRecorderError(
                    "RUNTIME_OUTPUT_PARENT_INVALID",
                    "runtime record parent changed during setup",
                )
        except RuntimeRecorderError:
            if sample_fd is not None:
                os.close(sample_fd)
            os.close(parent_fd)
            raise
        except OSError as exc:
            if sample_fd is not None:
                os.close(sample_fd)
            os.close(parent_fd)
            code = (
                "RUNTIME_OUTPUT_EXISTS"
                if exc.errno == errno.EEXIST
                else "RUNTIME_OUTPUT_PUBLICATION_FAILED"
            )
            raise RuntimeRecorderError(
                code,
                "runtime output generation could not be initialized",
                context={"path": str(absolute), "errno": exc.errno},
            ) from exc
        except BaseException:
            if sample_fd is not None:
                os.close(sample_fd)
            os.close(parent_fd)
            raise
        assert sample_fd is not None
        return cls(
            output_path=absolute,
            parent_fd=parent_fd,
            parent_identity=parent_identity,
            sample_directory_name=sample_name,
            sample_directory_fd=sample_fd,
            sample_directory_identity=_stat_identity(sample_state),
        )

    def revalidate(self) -> None:
        try:
            parent_state = os.fstat(self.parent_fd)
            reopened_parent_fd = _open_directory_no_symlinks(
                self.output_path.parent,
                role="runtime record parent",
                error_code="RUNTIME_OUTPUT_DRIFT",
            )
            try:
                parent_path_state = os.fstat(reopened_parent_fd)
            finally:
                os.close(reopened_parent_fd)
            sample_state = os.fstat(self.sample_directory_fd)
            sample_path_state = os.stat(
                self.sample_directory_name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_DRIFT",
                "runtime output directory identity became unreadable",
                context={"errno": exc.errno},
            ) from exc
        if (
            self.parent_identity != _stat_identity(parent_state)
            or self.parent_identity != _stat_identity(parent_path_state)
            or self.sample_directory_identity != _stat_identity(sample_state)
            or self.sample_directory_identity != _stat_identity(sample_path_state)
        ):
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_DRIFT",
                "runtime output directory identity changed",
            )

    def write_stream(
        self,
        *,
        sample_index: int,
        kind: str,
        stream_name: str,
        payload: bytes,
        complete: bool,
    ) -> dict[str, Any]:
        suffix = "json" if stream_name == "stdout" else "bin"
        name = f"sample-{sample_index:03d}-{kind}.{stream_name}.{suffix}"
        self.revalidate()
        try:
            _write_exclusive_at(self.sample_directory_fd, name, payload)
            os.fsync(self.sample_directory_fd)
            self.sample_directory_identity = _stat_identity(
                os.fstat(self.sample_directory_fd)
            )
        except RuntimeRecorderError:
            raise
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_PUBLICATION_FAILED",
                "runtime sample artifact could not be made durable",
                context={"path": name, "errno": exc.errno},
            ) from exc
        self.revalidate()
        return {
            "path": f"{self.sample_directory_name}/{name}",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "capture_complete": complete,
        }

    def publish_record(self, document: Mapping[str, Any]) -> None:
        payload = _canonical_json_line(document)
        self.revalidate()
        _publish_exclusive_at(self.parent_fd, self.output_path.name, payload)

    def close(self) -> None:
        os.close(self.sample_directory_fd)
        os.close(self.parent_fd)


def _stat_identity(value: os.stat_result) -> _StatIdentity:
    return _StatIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        link_count=value.st_nlink,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _same_filesystem_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _retain_fd_above_stdio(fd: int, *, role: str, error_code: str) -> int:
    if fd >= 3:
        return fd
    try:
        retained = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    except OSError as exc:
        raise RuntimeRecorderError(
            error_code,
            f"{role} descriptor could not be retained above standard I/O",
            context={"errno": exc.errno},
        ) from exc
    os.close(fd)
    return retained


def _open_directory_no_symlinks(
    path: str | Path,
    *,
    role: str,
    error_code: str = "RUNTIME_INPUT_OPEN_FAILED",
) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    fd: int | None = None
    try:
        fd = os.open("/", flags)
        for component in absolute.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        state = os.fstat(fd)
        if not stat.S_ISDIR(state.st_mode):
            raise RuntimeRecorderError(
                error_code,
                f"{role} must resolve through non-symlink directories",
                context={"path": str(absolute)},
            )
        return _retain_fd_above_stdio(fd, role=role, error_code=error_code)
    except RuntimeRecorderError:
        if fd is not None:
            os.close(fd)
        raise
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise RuntimeRecorderError(
            error_code,
            f"{role} could not be opened without following symlinks",
            context={"path": str(absolute), "errno": exc.errno},
        ) from exc


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _write_exclusive_at(directory_fd: int, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_OUTPUT_PUBLICATION_FAILED",
            "runtime evidence path could not be created exclusively",
            context={"path": name, "errno": exc.errno},
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise RuntimeRecorderError(
                    "RUNTIME_OUTPUT_PUBLICATION_FAILED",
                    "runtime evidence write made no progress",
                    context={"path": name},
                )
            offset += written
        os.fsync(fd)
    except RuntimeRecorderError:
        raise
    except OSError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_OUTPUT_PUBLICATION_FAILED",
            "runtime evidence bytes could not be written durably",
            context={"path": name, "errno": exc.errno},
        ) from exc
    finally:
        os.close(fd)


def _publish_exclusive_at(directory_fd: int, name: str, payload: bytes) -> None:
    temporary = f".{name}.{secrets.token_hex(16)}.pending"
    _write_exclusive_at(directory_fd, temporary, payload)
    published = False
    cleanup_error: OSError | None = None
    try:
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_EXISTS",
                "runtime record output already exists",
                context={"path": name},
            ) from exc
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_PUBLICATION_FAILED",
                "complete runtime record could not be published atomically",
                context={"path": name, "errno": exc.errno},
            ) from exc
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError as exc:
            cleanup_error = exc
    if published and cleanup_error is not None:
        raise RuntimeRecorderError(
            "RUNTIME_OUTPUT_PUBLICATION_FAILED",
            "published runtime record retained a temporary hardlink",
            context={"path": name, "errno": cleanup_error.errno},
        ) from cleanup_error
    if published:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_PUBLICATION_FAILED",
                "published runtime record directory could not be synchronized",
                context={"path": name, "errno": exc.errno},
            ) from exc


def _hash_fd(fd: int, size: int, *, role: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        try:
            chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_READ_FAILED",
                f"retained {role} could not be hashed",
                context={"errno": exc.errno},
            ) from exc
        if not chunk:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_READ_FAILED",
                f"retained {role} ended before its admitted size",
            )
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _read_fd(fd: int, size: int, *, role: str) -> bytes:
    payload = bytearray()
    offset = 0
    while offset < size:
        try:
            chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_READ_FAILED",
                f"retained {role} could not be read",
                context={"errno": exc.errno},
            ) from exc
        if not chunk:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_READ_FAILED",
                f"retained {role} ended before its admitted size",
            )
        payload.extend(chunk)
        offset += len(chunk)
    return bytes(payload)


def _open_retained_file(
    path: str | Path,
    *,
    role: str,
    maximum_bytes: int,
    require_executable: bool = False,
    require_single_link: bool = True,
) -> _RetainedFile:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.name in {"", ".", ".."}:
        raise RuntimeRecorderError(
            "RUNTIME_INPUT_OPEN_FAILED",
            f"{role} path has no regular-file name",
            context={"path": str(absolute)},
        )
    parent_fd = _open_directory_no_symlinks(absolute.parent, role=f"{role} parent")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        fd = _retain_fd_above_stdio(
            fd,
            role=role,
            error_code="RUNTIME_INPUT_OPEN_FAILED",
        )
    except RuntimeRecorderError:
        os.close(parent_fd)
        raise
    except OSError as exc:
        os.close(parent_fd)
        raise RuntimeRecorderError(
            "RUNTIME_INPUT_OPEN_FAILED",
            f"{role} could not be opened without following a final symlink",
            context={"path": str(absolute), "errno": exc.errno},
        ) from exc
    try:
        descriptor_state = os.fstat(fd)
        path_state = os.stat(
            absolute.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(descriptor_state.st_mode) or not stat.S_ISREG(
            path_state.st_mode
        ):
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_TYPE_INVALID",
                f"{role} must be a regular file",
                context={"path": str(absolute)},
            )
        identity = _stat_identity(descriptor_state)
        if require_single_link and identity.link_count != 1:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_IDENTITY_INVALID",
                f"{role} must not have hardlink aliases",
                context={"path": str(absolute), "link_count": identity.link_count},
            )
        if _stat_identity(path_state) != identity:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_IDENTITY_INVALID",
                f"{role} path and retained descriptor disagree",
                context={"path": str(absolute)},
            )
        if identity.size <= 0 or identity.size > maximum_bytes:
            raise RuntimeRecorderError(
                "RUNTIME_INPUT_SIZE_INVALID",
                f"{role} size is outside the admitted bound",
                context={"path": str(absolute), "size_bytes": identity.size},
            )
        if require_executable and not identity.mode & 0o111:
            raise RuntimeRecorderError(
                "RUNTIME_BINARY_NOT_EXECUTABLE",
                "release sim-run has no executable mode bit",
                context={"path": str(absolute)},
            )
        digest = _hash_fd(fd, identity.size, role=role)
        retained = _RetainedFile(
            path=absolute,
            parent_fd=parent_fd,
            name=absolute.name,
            fd=fd,
            maximum_bytes=maximum_bytes,
            identity=identity,
            sha256=digest,
            require_single_link=require_single_link,
        )
        retained.revalidate(role=role)
        return retained
    except RuntimeRecorderError:
        os.close(fd)
        os.close(parent_fd)
        raise
    except OSError as exc:
        os.close(fd)
        os.close(parent_fd)
        raise RuntimeRecorderError(
            "RUNTIME_INPUT_READ_FAILED",
            f"{role} identity could not be retained",
            context={"path": str(absolute), "errno": exc.errno},
        ) from exc
    except BaseException:
        os.close(fd)
        os.close(parent_fd)
        raise


def _sha256_canonical(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_line(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _decode_object(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_RESULT_INVALID", "sim-run output is not strict UTF-8"
        ) from exc
    try:
        value = loads_json_strict(text, source=source)
    except (ContractValidationError, MemoryError, OverflowError, RecursionError) as exc:
        raise RuntimeRecorderError(
            "RUNTIME_RESULT_INVALID", "sim-run output is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeRecorderError(
            "RUNTIME_RESULT_INVALID", "sim-run output root is not an object"
        )
    return value


def _seconds_to_ns(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise RuntimeRecorderError(
            "RUNTIME_RESOURCE_USAGE_INVALID",
            "child CPU time is not a finite nonnegative value",
        )
    return int(round(value * 1_000_000_000))


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _retained_manifest_argument(manifest: _RetainedFile) -> str:
    descriptors = (manifest.fd, manifest.parent_fd)
    if (
        any(value < 3 or value > _MAX_INHERITED_DESCRIPTOR for value in descriptors)
        or descriptors[0] == descriptors[1]
    ):
        raise RuntimeRecorderError(
            "RUNTIME_DESCRIPTOR_INVALID",
            "retained manifest descriptors cannot be encoded canonically",
        )
    try:
        encoded_name = manifest.name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_INPUT_PATH_INVALID",
            "manifest leaf name is not valid UTF-8",
            context={"path": str(manifest.path)},
        ) from exc
    if not encoded_name:
        raise RuntimeRecorderError(
            "RUNTIME_INPUT_PATH_INVALID",
            "manifest leaf name is empty",
            context={"path": str(manifest.path)},
        )
    return (
        f"{_RETAINED_MANIFEST_ARGUMENT_PREFIX}{manifest.fd}:"
        f"{manifest.parent_fd}:{encoded_name.hex()}"
    )


def _parse_retained_manifest_argument(value: str) -> tuple[int, int, str] | None:
    if not value.startswith(_RETAINED_MANIFEST_ARGUMENT_PREFIX):
        return None
    fields = value[len(_RETAINED_MANIFEST_ARGUMENT_PREFIX) :].split(":")
    if len(fields) != 3:
        return None
    descriptor_fields = fields[:2]
    if any(
        not field.isascii()
        or not field.isdigit()
        or str(int(field)) != field
        or int(field) < 3
        or int(field) > _MAX_INHERITED_DESCRIPTOR
        for field in descriptor_fields
    ) or descriptor_fields[0] == descriptor_fields[1]:
        return None
    encoded_name = fields[2]
    if (
        not encoded_name
        or len(encoded_name) % 2 != 0
        or not encoded_name.isascii()
        or any(character not in "0123456789abcdef" for character in encoded_name)
    ):
        return None
    try:
        name = bytes.fromhex(encoded_name).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return None
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        or name.encode("utf-8").hex() != encoded_name
    ):
        return None
    return int(descriptor_fields[0]), int(descriptor_fields[1]), name


def _spawn_once(
    binary: _RetainedFile,
    manifest: _RetainedFile,
    *,
    working_directory: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> _SpawnObservation:
    if (
        not hasattr(os, "wait4")
        or not hasattr(os, "waitid")
        or not hasattr(os, "WNOWAIT")
        or not hasattr(os, "P_PID")
        or platform.system() != "Linux"
    ):
        raise RuntimeRecorderError(
            "RUNTIME_PLATFORM_UNSUPPORTED",
            "exact-child wait4 and non-reaping waitid supervision require Linux",
        )
    executable = f"/proc/self/fd/{binary.fd}"
    argv = [str(binary.path), _retained_manifest_argument(manifest)]
    try:
        exit_read_fd, exit_write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_SUPERVISION_UNAVAILABLE",
            "waitid notification pipe could not be created",
            context={"errno": exc.errno},
        ) from exc
    try:
        background_load = tuple(float(value) for value in os.getloadavg())
    except (AttributeError, OSError) as exc:
        os.close(exit_read_fd)
        os.close(exit_write_fd)
        raise RuntimeRecorderError(
            "RUNTIME_LOAD_OBSERVATION_UNAVAILABLE",
            "background load could not be observed before spawn",
        ) from exc
    if len(background_load) != 3 or any(
        not math.isfinite(value) or value < 0 for value in background_load
    ):
        os.close(exit_read_fd)
        os.close(exit_write_fd)
        raise RuntimeRecorderError(
            "RUNTIME_LOAD_OBSERVATION_INVALID",
            "background load observation is invalid",
        )
    timeout_ns = int(round(timeout_seconds * 1_000_000_000))
    start_ns = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            argv,
            executable=executable,
            cwd=working_directory,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(
                binary.fd,
                manifest.fd,
                manifest.parent_fd,
            ),
            start_new_session=True,
        )
    except OSError as exc:
        os.close(exit_read_fd)
        os.close(exit_write_fd)
        raise RuntimeRecorderError(
            "RUNTIME_SPAWN_FAILED",
            "release sim-run could not be spawned",
            context={"errno": exc.errno},
        ) from exc
    deadline_ns = start_ns + timeout_ns
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    waitid_errors: list[BaseException] = []

    def observe_exit_without_reaping() -> None:
        try:
            while True:
                try:
                    os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
                    break
                except InterruptedError:
                    continue
        except BaseException as exc:  # pragma: no cover - rare waitid failure.
            waitid_errors.append(exc)
        finally:
            try:
                os.write(exit_write_fd, b"x")
            except OSError:
                pass

    exit_thread = threading.Thread(
        target=observe_exit_without_reaping,
        name="sim-run-waitid-observer",
        daemon=True,
    )
    exit_thread_started = False
    try:
        os.set_blocking(stdout_fd, False)
        os.set_blocking(stderr_fd, False)
        exit_thread.start()
        exit_thread_started = True
    except (OSError, RuntimeError) as exc:
        _kill_process_group(process.pid)
        try:
            waited_pid, wait_status, _ = os.wait4(process.pid, 0)
            if waited_pid == process.pid:
                process.returncode = os.waitstatus_to_exitcode(wait_status)
        except (ChildProcessError, OSError):
            pass
        if exit_thread_started:
            exit_thread.join(timeout=5)
        os.close(exit_read_fd)
        os.close(exit_write_fd)
        process.stdout.close()
        process.stderr.close()
        raise RuntimeRecorderError(
            "RUNTIME_SUPERVISION_UNAVAILABLE",
            "waitid output supervision could not be initialized",
            context={"errno": getattr(exc, "errno", None)},
        ) from exc

    stdout_drain = _StreamDrain(
        maximum_bytes=MAX_RESULT_STREAM_BYTES,
        payload=bytearray(),
    )
    stderr_drain = _StreamDrain(
        maximum_bytes=MAX_RESULT_STREAM_BYTES,
        payload=bytearray(),
    )
    streams = {
        stdout_fd: stdout_drain,
        stderr_fd: stderr_drain,
    }
    eof_fds: set[int] = set()
    poller = select.poll()
    stream_mask = select.POLLIN | select.POLLHUP | select.POLLERR
    poller.register(exit_read_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    poller.register(stdout_fd, stream_mask)
    poller.register(stderr_fd, stream_mask)
    deadline_reached = False
    child_ready = False
    capture_failed = False
    exact_child_reaped = False
    group_termination_completed = False

    def terminate_process_group() -> None:
        nonlocal group_termination_completed
        _kill_process_group(process.pid)
        group_termination_completed = True

    try:
        while not child_ready and not capture_failed:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                deadline_reached = True
                terminate_process_group()
                break
            timeout_ms = max(1, (remaining_ns + 999_999) // 1_000_000)
            try:
                events = poller.poll(timeout_ms)
            except InterruptedError:
                continue
            for fd, _event in events:
                if fd in streams:
                    if streams[fd].consume(fd):
                        eof_fds.add(fd)
                        try:
                            poller.unregister(fd)
                        except KeyError:  # pragma: no cover - defensive only.
                            pass
                    if not streams[fd].complete:
                        capture_failed = True
                elif fd == exit_read_fd:
                    try:
                        os.read(exit_read_fd, 4096)
                    except BlockingIOError:
                        pass
                    child_ready = True
            if capture_failed:
                terminate_process_group()
                break
            if child_ready:
                # waitid used WNOWAIT, so the leader remains waitable. Terminating
                # its process group before wait4 prevents successful children from
                # leaving descendants that retain output pipes.
                terminate_process_group()
                break
            if time.monotonic_ns() >= deadline_ns:
                deadline_reached = True
                terminate_process_group()
                break

        while True:
            try:
                waited_pid, wait_status, usage = os.wait4(process.pid, 0)
                exact_child_reaped = True
                stop_ns = time.monotonic_ns()
                break
            except InterruptedError:
                continue
    except BaseException as exc:
        if not exact_child_reaped and not group_termination_completed:
            terminate_process_group()
        if not exact_child_reaped:
            try:
                os.wait4(process.pid, 0)
            except (ChildProcessError, OSError):
                pass
        exit_thread.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
        os.close(exit_read_fd)
        os.close(exit_write_fd)
        if isinstance(exc, OSError):
            raise RuntimeRecorderError(
                "RUNTIME_WAIT_FAILED",
                "exact-child supervision failed",
                context={"errno": exc.errno},
            ) from exc
        raise

    exit_thread.join(timeout=5)
    if waited_pid != process.pid or exit_thread.is_alive() or (
        waitid_errors and not (deadline_reached or capture_failed)
    ):
        process.stdout.close()
        process.stderr.close()
        os.close(exit_read_fd)
        os.close(exit_write_fd)
        raise RuntimeRecorderError(
            "RUNTIME_WAIT_FAILED",
            "waitid and wait4 did not preserve the exact child identity",
        )
    return_code = os.waitstatus_to_exitcode(wait_status)
    process.returncode = return_code
    timed_out = deadline_reached or stop_ns > deadline_ns
    try:
        poller.unregister(exit_read_fd)
    except KeyError:  # pragma: no cover - defensive only.
        pass
    os.close(exit_read_fd)
    os.close(exit_write_fd)

    drain_deadline_ns = time.monotonic_ns() + _PIPE_DRAIN_TIMEOUT_NS
    while len(eof_fds) != len(streams):
        for fd, drain in streams.items():
            if fd not in eof_fds and drain.consume(fd):
                eof_fds.add(fd)
                try:
                    poller.unregister(fd)
                except KeyError:  # pragma: no cover - defensive only.
                    pass
        if len(eof_fds) == len(streams):
            break
        remaining_ns = drain_deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            # The exact child has already been reaped, so its numeric PID/PGID is
            # reusable. Never signal by that number beyond the timed boundary.
            for fd, drain in streams.items():
                if fd not in eof_fds:
                    drain.complete = False
                    drain.error = "LingeringPipe"
            break
        try:
            poller.poll(max(1, (remaining_ns + 999_999) // 1_000_000))
        except InterruptedError:
            continue

    process.stdout.close()
    process.stderr.close()
    capture_errors = [
        value for value in (stdout_drain.error, stderr_drain.error) if value is not None
    ]
    return _SpawnObservation(
        background_load=background_load,
        monotonic_start_ns=start_ns,
        monotonic_stop_ns=stop_ns,
        user_cpu_ns=_seconds_to_ns(float(usage.ru_utime)),
        system_cpu_ns=_seconds_to_ns(float(usage.ru_stime)),
        peak_rss_kib=int(usage.ru_maxrss),
        return_code=return_code,
        timed_out=timed_out,
        stdout=bytes(stdout_drain.payload),
        stdout_complete=stdout_drain.complete,
        stderr=bytes(stderr_drain.payload),
        stderr_complete=stderr_drain.complete,
        capture_error=",".join(capture_errors) if capture_errors else None,
    )


def _available_string(value: str) -> dict[str, str]:
    return {"status": "available", "value": value}


def _available_u64(value: int) -> dict[str, Any]:
    return {"status": "available", "value": value}


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _read_optional_text(path: Path, *, maximum_bytes: int) -> str | None:
    try:
        with path.open("rb") as stream:
            payload = stream.read(maximum_bytes + 1)
    except OSError:
        return None
    if len(payload) > maximum_bytes:
        return None
    try:
        value = payload.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return None
    return value or None


def _cpu_model() -> dict[str, Any]:
    payload = _read_optional_text(Path("/proc/cpuinfo"), maximum_bytes=4 * 1024 * 1024)
    if payload is None:
        return _unavailable("cpu_model_unavailable")
    for line in payload.splitlines():
        if line.startswith("model name") and ":" in line:
            value = line.split(":", 1)[1].strip()
            if value:
                return _available_string(value)
    return _unavailable("cpu_model_unavailable")


def _physical_core_count(affinity: Sequence[int]) -> dict[str, Any]:
    identities: set[tuple[str, str]] = set()
    for cpu in affinity:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = _read_optional_text(base / "physical_package_id", maximum_bytes=128)
        core = _read_optional_text(base / "core_id", maximum_bytes=128)
        if package is None or core is None:
            return _unavailable("physical_core_topology_unavailable")
        identities.add((package, core))
    if not identities:
        return _unavailable("physical_core_topology_unavailable")
    return _available_u64(len(identities))


def _memory_bytes() -> dict[str, Any]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * page_count
    except (OSError, ValueError, TypeError):
        return _unavailable("physical_memory_unavailable")
    if total <= 0:
        return _unavailable("physical_memory_unavailable")
    return _available_u64(total)


def _cpu_governor(affinity: Sequence[int]) -> dict[str, Any]:
    values: set[str] = set()
    for cpu in affinity:
        value = _read_optional_text(
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"),
            maximum_bytes=128,
        )
        if value is None:
            return _unavailable("cpu_governor_unavailable")
        values.add(value)
    if len(values) != 1:
        return _unavailable("cpu_governor_mixed")
    return _available_string(next(iter(values)))


def _capture_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeRecorderError(
            "RUNTIME_AFFINITY_UNAVAILABLE",
            "process affinity is required by the runtime protocol",
        )
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except OSError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_AFFINITY_UNAVAILABLE",
            "process affinity could not be captured",
            context={"errno": exc.errno},
        ) from exc
    if not affinity:
        raise RuntimeRecorderError(
            "RUNTIME_AFFINITY_UNAVAILABLE", "process affinity set is empty"
        )
    return affinity


def _host_identity(manifest: Mapping[str, Any], affinity: list[int]) -> dict[str, Any]:
    logical = os.cpu_count()
    logical_value = (
        _available_u64(logical)
        if isinstance(logical, int) and logical > 0
        else _unavailable("logical_core_count_unavailable")
    )
    os_name = platform.system() or "unknown"
    kernel = platform.release() or "unknown"
    machine = platform.machine() or "unknown"
    host = {
        "hostname_sha256": hashlib.sha256(
            socket.gethostname().encode("utf-8")
        ).hexdigest(),
        "cpu_model": _cpu_model(),
        "physical_core_count": _physical_core_count(affinity),
        "logical_core_count": logical_value,
        "memory_bytes": _memory_bytes(),
        "os_name": _available_string(os_name),
        "kernel_release": _available_string(kernel),
        "machine": _available_string(machine),
        "affinity_cpus": affinity,
        "cpu_governor": _cpu_governor(affinity),
        "target_gpu": {
            "sku": _available_string(manifest["architecture"]["sku"]),
            "sm_arch": _available_string(manifest["architecture"]["arch_id"]),
            "device_architecture": _unavailable(
                "gpu_device_architecture_authority_unavailable"
            ),
            "driver_version": _unavailable("gpu_driver_not_queried_offline"),
        },
    }
    host["host_fingerprint_sha256"] = _sha256_canonical(host)
    return host


def _environment() -> tuple[list[dict[str, str]], dict[str, str], str]:
    entries = [{"name": name, "value": value} for name, value in _ENVIRONMENT_ITEMS]
    values = {name: value for name, value in _ENVIRONMENT_ITEMS}
    digest = _sha256_canonical({"environment": entries})
    return entries, values, digest


def _manifest_document(retained: _RetainedFile) -> tuple[dict[str, Any], str]:
    payload = retained.read_bytes(role="run manifest")
    try:
        document = _decode_object(payload, source=str(retained.path))
        validate_manifest(document)
    except (RuntimeRecorderError, ContractValidationError) as exc:
        raise RuntimeRecorderError(
            "RUNTIME_MANIFEST_INVALID", "run manifest v2 admission failed"
        ) from exc
    canonical = _sha256_canonical(document)
    return document, canonical


def _launch_projection(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    launch_keys = [deepcopy(launch["launch_key"]) for launch in manifest["launches"]]
    launch_keys.sort(key=canonical_launch_key_token)
    if len({canonical_launch_key_token(key) for key in launch_keys}) != len(launch_keys):
        raise RuntimeRecorderError(
            "RUNTIME_MANIFEST_INVALID", "run manifest launch keys are not unique"
        )
    digest = _sha256_canonical({"launch_keys": launch_keys})
    return launch_keys, digest


def _validate_result_binding(
    observation: _SpawnObservation,
    *,
    binary: _RetainedFile,
    manifest: _RetainedFile,
    manifest_document: Mapping[str, Any],
    manifest_canonical_sha256: str,
) -> tuple[dict[str, Any], str]:
    if observation.timed_out:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED", "release sim-run exceeded the fixed timeout"
        )
    if not observation.stdout_complete or not observation.stderr_complete:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED", "release sim-run exceeded an output bound"
        )
    if observation.capture_error is not None:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED", "release sim-run output capture failed"
        )
    if observation.return_code != 0:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "release sim-run did not exit zero",
            context={"return_code": observation.return_code},
        )
    if observation.stderr != b"":
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "release sim-run emitted stderr on a success exit",
        )
    document = _decode_object(observation.stdout, source="sim-run stdout")
    try:
        validate_result(document)
    except ContractValidationError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED", "release sim-run result contract is invalid"
        ) from exc
    if observation.stdout != _canonical_json_line(document):
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED", "release sim-run result is not canonical JSON"
        )
    build = document["build_identity"]
    if build["cargo_profile"] != "release":
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED", "sim-run result is not a release build"
        )
    if build["binary_sha256"] != binary.sha256:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result names a different executable image",
        )
    configuration = document["configuration_identity"]
    if configuration["run_manifest_sha256"] != manifest_canonical_sha256:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result does not bind the admitted manifest",
        )
    if configuration["resolved_configuration_sha256"] != _sha256_canonical(
        document["resolved_configuration"]
    ):
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result resolved-configuration digest does not match",
        )
    if document["resolved_configuration"]["replay_mode"] != manifest_document[
        "replay_mode"
    ]:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result replay mode differs from the manifest",
        )
    expected_configuration = manifest_document["configuration_evidence"]
    if (
        configuration["arch_card_sha256"]
        != expected_configuration["arch_card_sha256"]
        or configuration["pchase_card_sha256"]
        != expected_configuration["pchase_card_sha256"]
        or document["arch_card_identity"]["card_sha256"]
        != configuration["arch_card_sha256"]
    ):
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result configuration evidence differs from the manifest",
        )
    architecture = manifest_document["architecture"]
    if document["arch_card_identity"]["sm_arch"] != architecture["arch_id"] or document[
        "arch_card_identity"
    ]["sku"] != architecture["sku"]:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result architecture differs from the manifest",
        )
    expected_launches, _ = _launch_projection(manifest_document)
    observed_launches = [deepcopy(launch["launch_key"]) for launch in document["launches"]]
    observed_launches.sort(key=canonical_launch_key_token)
    if observed_launches != expected_launches:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result launch set differs from the manifest",
        )
    binary.revalidate(role="release sim-run")
    manifest.revalidate(role="run manifest")
    return document, hashlib.sha256(observation.stdout).hexdigest()


def _termination(return_code: int) -> dict[str, Any]:
    if return_code >= 0:
        return {"kind": "exit", "exit_code": return_code}
    return {"kind": "signal", "signal": -return_code}


def _is_input_drift_record(document: Mapping[str, Any]) -> bool:
    outcome = document.get("outcome")
    return (
        isinstance(outcome, Mapping)
        and outcome.get("status") == "RUNTIME_EXECUTION_FAILED"
        and outcome.get("failure_code") == "RUNTIME_INPUT_DRIFT"
        and isinstance(outcome.get("detail"), str)
        and outcome["detail"].startswith("RUNTIME_INPUT_DRIFT:")
    )


def _sample_has_success_envelope(sample: Mapping[str, Any]) -> bool:
    return (
        sample["termination"] == {"kind": "exit", "exit_code": 0}
        and not sample["timed_out"]
        and sample["stdout"]["capture_complete"]
        and sample["stderr"]["capture_complete"]
        and sample["stderr"]["size_bytes"] == 0
        and sample["stderr"]["sha256"] == _EMPTY_SHA256
    )


def _sample_stdout_identity(sample: Mapping[str, Any]) -> tuple[str, int]:
    return sample["stdout"]["sha256"], sample["stdout"]["size_bytes"]


def _protocol_expects_another_repeat(samples: Sequence[Mapping[str, Any]]) -> bool:
    count = len(samples)
    if count < 1 + INITIAL_TIMED_REPEATS:
        return True
    if count == 1 + INITIAL_TIMED_REPEATS:
        return _is_noisy([sample["wall_time_ns"] for sample in samples[1:]])
    return count < 1 + INITIAL_TIMED_REPEATS + ADDITIONAL_NOISY_REPEATS


def _sample_document(
    observation: _SpawnObservation,
    *,
    sample_index: int,
    kind: str,
    timed_repeat_index: int | None,
    publisher: _Publisher,
) -> dict[str, Any]:
    if observation.monotonic_stop_ns <= observation.monotonic_start_ns:
        raise RuntimeRecorderError(
            "RUNTIME_CLOCK_INVALID", "monotonic stop does not follow monotonic start"
        )
    stdout = publisher.write_stream(
        sample_index=sample_index,
        kind=kind,
        stream_name="stdout",
        payload=observation.stdout,
        complete=observation.stdout_complete,
    )
    stderr = publisher.write_stream(
        sample_index=sample_index,
        kind=kind,
        stream_name="stderr",
        payload=observation.stderr,
        complete=observation.stderr_complete,
    )
    load = observation.background_load
    return {
        "sample_index": sample_index,
        "kind": kind,
        "timed_repeat_index": timed_repeat_index,
        "background_load": {
            "one_minute": load[0],
            "five_minutes": load[1],
            "fifteen_minutes": load[2],
        },
        "monotonic_start_ns": observation.monotonic_start_ns,
        "monotonic_stop_ns": observation.monotonic_stop_ns,
        "wall_time_ns": observation.monotonic_stop_ns
        - observation.monotonic_start_ns,
        "user_cpu_ns": observation.user_cpu_ns,
        "system_cpu_ns": observation.system_cpu_ns,
        "peak_rss_kib": observation.peak_rss_kib,
        "termination": _termination(observation.return_code),
        "timed_out": observation.timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def _median(values: Sequence[int | Fraction]) -> Fraction:
    ordered = sorted(Fraction(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _json_number(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _summary(
    walls: Sequence[int], *, record_count: int, instruction_count: int
) -> dict[str, Any]:
    if len(walls) not in {INITIAL_TIMED_REPEATS, INITIAL_TIMED_REPEATS + ADDITIONAL_NOISY_REPEATS}:
        raise RuntimeRecorderError(
            "RUNTIME_SUMMARY_INVALID", "timed repeat count is outside the protocol"
        )
    mean = Fraction(sum(walls), len(walls))
    median = _median(walls)
    deviations = [abs(Fraction(value) - median) for value in walls]
    mad = _median(deviations)
    variance = sum((Fraction(value) - mean) ** 2 for value in walls) / (
        len(walls) - 1
    )
    standard_deviation = math.sqrt(float(variance))
    coefficient = standard_deviation / float(mean)
    records_per_second = float(Fraction(record_count * 1_000_000_000, 1) / median)
    instructions_per_second = float(
        Fraction(instruction_count * 1_000_000_000, 1) / median
    )
    return {
        "timed_repeat_count": len(walls),
        "minimum_wall_time_ns": min(walls),
        "median_wall_time_ns": _json_number(median),
        "median_absolute_deviation_ns": _json_number(mad),
        "mean_wall_time_ns": _json_number(mean),
        "sample_standard_deviation_ns": standard_deviation,
        "coefficient_of_variation": coefficient,
        "throughput_basis": "selected_replay_count_divided_by_median_external_wall_time",
        "records_per_second": records_per_second,
        "instructions_per_second": instructions_per_second,
    }


def _is_noisy(walls: Sequence[int]) -> bool:
    mean = Fraction(sum(walls), len(walls))
    variance = sum((Fraction(value) - mean) ** 2 for value in walls) / (
        len(walls) - 1
    )
    return variance > mean * mean * CV_THRESHOLD * CV_THRESHOLD


def _selected_replay_counts(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        metric = result["aggregate"]["metrics"]["smsp_inst_executed_sum"]
        value = metric["value"]
        count = value["value"]
    except (KeyError, TypeError) as exc:
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run result lacks the selected-roster replay count",
        ) from exc
    if (
        metric.get("status") != "predicted"
        or value.get("kind") != "exact"
        or value.get("semantics") != "simulator_scope"
        or type(count) is not int
        or count <= 0
        or count > 18_446_744_073_709_551_615
    ):
        raise RuntimeRecorderError(
            "RUNTIME_EXECUTION_FAILED",
            "sim-run selected-roster replay count is not a positive exact simulator-scope count",
        )
    return {
        "basis": "aggregate_exact_instructions_with_checked_issue_record_equality.v1",
        "record_count": count,
        "instruction_count": count,
    }


def _result_binding(
    result: Mapping[str, Any] | None, file_sha256: str | None, reason: str
) -> dict[str, Any]:
    if result is None or file_sha256 is None:
        return {"status": "unavailable", "reason": reason}
    diagnostics = result["diagnostics"]
    internal = (
        {
            "status": "available",
            "value": deepcopy(diagnostics["internal_phase_timings_ns"]),
            "role": "diagnostic_only_never_replaces_external_runtime",
        }
        if "internal_phase_timings_ns" in diagnostics
        else _unavailable("sim_result_internal_phase_timings_unavailable")
    )
    return {
        "status": "bound",
        "file_sha256": file_sha256,
        "result_sha256": result["result_sha256"],
        "build_identity": deepcopy(result["build_identity"]),
        "trace_identity": deepcopy(result["trace_identity"]),
        "selected_replay_counts": _selected_replay_counts(result),
        "arch_card_identity": deepcopy(result["arch_card_identity"]),
        "configuration_identity": deepcopy(result["configuration_identity"]),
        "resolved_configuration": deepcopy(result["resolved_configuration"]),
        "internal_phase_timings_ns": internal,
    }


def _workload_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_canonical_sha256: str,
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    launch_keys, launch_set_sha256 = _launch_projection(manifest)
    workload_ids = sorted({key["workload_id"] for key in launch_keys})
    return {
        "workload_ids": workload_ids,
        "workload_family": _unavailable("workload_family_authority_unavailable"),
        "input_digest": manifest_canonical_sha256,
        "launch_set_sha256": launch_set_sha256,
        "launch_count": len(launch_keys),
        "launch_keys": launch_keys,
        "trace": (
            {"status": "available", "value": deepcopy(result["trace_identity"])}
            if result is not None
            else _unavailable("successful_sim_result_unavailable")
        ),
        "configuration": (
            {
                "status": "available",
                "identity": deepcopy(result["configuration_identity"]),
                "resolved": deepcopy(result["resolved_configuration"]),
            }
            if result is not None
            else _unavailable("successful_sim_result_unavailable")
        ),
    }


def _claimability(
    result: Mapping[str, Any] | None, manifest: Mapping[str, Any]
) -> dict[str, str]:
    if manifest["scope"].startswith("fixture-only/") or any(
        launch["launch_key"]["workload_id"].startswith("fixture-only.")
        for launch in manifest["launches"]
    ):
        return {"status": "unclaimable", "reason": "SYNTHETIC_FIXTURE"}
    if result is not None and result["build_identity"]["sim_git_dirty"]:
        return {"status": "unclaimable", "reason": "DIRTY_BUILD_UNCLAIMABLE"}
    return {
        "status": "unclaimable",
        "reason": "WORKLOAD_TIER_AUTHORITY_UNAVAILABLE",
    }


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _record_runtime(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    timeout_seconds: float,
    repo_root: Path,
    spawn: Callable[..., _SpawnObservation],
) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise RuntimeRecorderError(
            "RUNTIME_TIMEOUT_INVALID", "runtime timeout must be finite and positive"
        )
    timeout_ns = int(round(timeout_seconds * 1_000_000_000))
    if timeout_ns <= 0 or timeout_ns > 18_446_744_073_709_551_615:
        raise RuntimeRecorderError(
            "RUNTIME_TIMEOUT_INVALID", "runtime timeout is outside the v1 bound"
        )
    repo_root = Path(os.path.abspath(os.fspath(repo_root)))
    release_binary = repo_root / RELEASE_BINARY_RELATIVE
    binary = _open_retained_file(
        release_binary,
        role="release sim-run",
        maximum_bytes=MAX_BINARY_BYTES,
        require_executable=True,
        require_single_link=False,
    )
    manifest = None
    publisher = None
    try:
        manifest = _open_retained_file(
            manifest_path,
            role="run manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
        manifest_document, manifest_canonical_sha256 = _manifest_document(manifest)
        manifest_argument = _retained_manifest_argument(manifest)
        affinity = _capture_affinity()
        host = _host_identity(manifest_document, affinity)
        environment_entries, environment_values, environment_sha256 = _environment()
        publisher = _Publisher.create(output_path)

        samples: list[dict[str, Any]] = []
        expected_result_bytes: bytes | None = None
        bound_result: dict[str, Any] | None = None
        bound_result_file_sha256: str | None = None
        failure_status: str | None = None
        failure_code: str | None = None
        failure_detail: str | None = None

        repeat_specs: list[tuple[str, int | None]] = [("warmup", None)]
        repeat_specs.extend(("timed", index) for index in range(INITIAL_TIMED_REPEATS))

        def run_specs(specs: Sequence[tuple[str, int | None]]) -> bool:
            nonlocal expected_result_bytes
            nonlocal bound_result
            nonlocal bound_result_file_sha256
            nonlocal failure_status
            nonlocal failure_code
            nonlocal failure_detail
            for kind, timed_index in specs:
                if _capture_affinity() != affinity:
                    failure_status = "RUNTIME_EXECUTION_FAILED"
                    failure_code = "RUNTIME_AFFINITY_DRIFT"
                    failure_detail = "process affinity changed during the runtime protocol"
                    return False
                try:
                    binary.revalidate(role="release sim-run")
                    manifest.revalidate(role="run manifest")
                    observation = spawn(
                        binary,
                        manifest,
                        working_directory=repo_root,
                        environment=environment_values,
                        timeout_seconds=timeout_seconds,
                    )
                    if _capture_affinity() != affinity:
                        raise RuntimeRecorderError(
                            "RUNTIME_AFFINITY_DRIFT",
                            "process affinity changed during a runtime repeat",
                        )
                    sample = _sample_document(
                        observation,
                        sample_index=len(samples),
                        kind=kind,
                        timed_repeat_index=timed_index,
                        publisher=publisher,
                    )
                    samples.append(sample)
                    result, result_file_sha256 = _validate_result_binding(
                        observation,
                        binary=binary,
                        manifest=manifest,
                        manifest_document=manifest_document,
                        manifest_canonical_sha256=manifest_canonical_sha256,
                    )
                except RuntimeRecorderError as exc:
                    failure_status = "RUNTIME_EXECUTION_FAILED"
                    failure_code = exc.code
                    failure_detail = f"{exc.code}: {exc.detail}"
                    return False
                if expected_result_bytes is None:
                    expected_result_bytes = observation.stdout
                    bound_result = result
                    bound_result_file_sha256 = result_file_sha256
                elif observation.stdout != expected_result_bytes:
                    bound_result = None
                    bound_result_file_sha256 = None
                    failure_status = "NONDETERMINISTIC_RESULT"
                    failure_code = "NONDETERMINISTIC_RESULT"
                    failure_detail = "release sim-run emitted non-byte-identical results"
                    return False
            return True

        completed_initial = run_specs(repeat_specs)
        summary: dict[str, Any] | None = None
        if completed_initial:
            walls = [sample["wall_time_ns"] for sample in samples if sample["kind"] == "timed"]
            assert bound_result is not None
            if _is_noisy(walls):
                extra = [
                    ("timed", INITIAL_TIMED_REPEATS + index)
                    for index in range(ADDITIONAL_NOISY_REPEATS)
                ]
                completed_initial = run_specs(extra)
                walls = [
                    sample["wall_time_ns"]
                    for sample in samples
                    if sample["kind"] == "timed"
                ]
            if completed_initial:
                selected_counts = _selected_replay_counts(bound_result)
                summary = _summary(
                    walls,
                    record_count=selected_counts["record_count"],
                    instruction_count=selected_counts["instruction_count"],
                )
                outcome = {
                    "status": "RUNTIME_NOISY" if _is_noisy(walls) else "MEASURED",
                    "summary": summary,
                }
            else:
                outcome = {
                    "status": failure_status,
                    "failure_code": failure_code,
                    "detail": failure_detail,
                    "summary": None,
                }
        else:
            outcome = {
                "status": failure_status,
                "failure_code": failure_code,
                "detail": failure_detail,
                "summary": None,
            }

        result_reason = (
            "nondeterministic_result"
            if outcome["status"] == "NONDETERMINISTIC_RESULT"
            else "no_successful_result"
            if bound_result is None
            else "result_identity_mismatch"
        )
        document: dict[str, Any] = {
            "schema_version": "sim_runtime_record.v1",
            "record_sha256": "0" * 64,
            "created_at_utc": _created_at_utc(),
            "authority_role": "exploratory_current_simulator_runtime_only",
            "timing_protocol": {
                "timing_boundary_id": (
                    "external_monotonic_immediately_before_spawn_to_wait4_return.v1"
                ),
                "clock": "time.monotonic_ns",
                "warmup_repeat_count": 1,
                "initial_timed_repeat_count": INITIAL_TIMED_REPEATS,
                "additional_repeat_count_on_noise": ADDITIONAL_NOISY_REPEATS,
                "coefficient_of_variation_threshold": float(CV_THRESHOLD),
                "warm_page_cache_policy": (
                    "one_warmup_then_natural_warm_cache_no_drop.v1"
                ),
                "background_load_observation": (
                    "os_getloadavg_immediately_before_each_spawn.v1"
                ),
                "build_timing_role": "release_build_completed_before_timing",
                "timeout_ns": timeout_ns,
            },
            "invocation": {
                "binary": {
                    "path": str(binary.path),
                    "size_bytes": binary.size,
                    "sha256": binary.sha256,
                },
                "binary_policy": "repository_release_sim_run_only",
                "executable_transport": "proc_self_fd_retained_release_inode.v1",
                "manifest_transport": (
                    "inherited_retained_manifest_and_evidence_base_fds.v1"
                ),
                "argv": [
                    str(binary.path),
                    manifest_argument,
                ],
                "environment": environment_entries,
                "environment_sha256": environment_sha256,
                "working_directory": str(repo_root),
                "affinity_cpus": affinity,
            },
            "manifest_identity": {
                "file": {
                    "path": str(manifest.path),
                    "size_bytes": manifest.size,
                    "sha256": manifest.sha256,
                },
                "canonical_sha256": manifest_canonical_sha256,
                "schema_version": 2,
                "scope": manifest_document["scope"],
            },
            "host_identity": host,
            "workload_identity": _workload_identity(
                manifest_document,
                manifest_canonical_sha256=manifest_canonical_sha256,
                result=bound_result,
            ),
            "sim_result_binding": _result_binding(
                bound_result, bound_result_file_sha256, result_reason
            ),
            "accuracy_binding": {
                "status": "unavailable",
                "reason": "production_matched_label_crosswalk_authority_unavailable",
            },
            "samples": samples,
            "claimability": _claimability(bound_result, manifest_document),
            "outcome": outcome,
        }
        document["record_sha256"] = canonical_sha256(document, "record_sha256")
        terminal_input_drift = failure_code == "RUNTIME_INPUT_DRIFT"
        validate_runtime_record(
            document,
            permit_terminal_input_drift=terminal_input_drift,
        )
        publisher.publish_record(document)
        reloaded = _load_runtime_record(
            publisher.output_path,
            repo_root=repo_root,
            permit_recorded_input_drift=terminal_input_drift,
        )
        if reloaded != document:
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_PUBLICATION_FAILED",
                "published runtime record did not replay byte-exactly",
            )
        return document
    finally:
        if publisher is not None:
            publisher.close()
        if manifest is not None:
            manifest.close()
        binary.close()


def record_runtime(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Record exploratory runtime with the repository release ``sim-run`` only."""

    return _record_runtime(
        manifest_path,
        output_path,
        timeout_seconds=timeout_seconds,
        repo_root=ROOT,
        spawn=_spawn_once,
    )


def _read_runtime_schema() -> dict[str, Any]:
    value = load_json_file_strict(RUNTIME_SCHEMA_PATH)
    if not isinstance(value, dict):
        raise RuntimeError("runtime schema root is not an object")
    return value


_RUNTIME_SCHEMA = _read_runtime_schema()
_RUNTIME_REGISTRY = Registry()
for _schema in (
    schema_document("launch_key"),
    schema_document("result"),
    _RUNTIME_SCHEMA,
):
    _RUNTIME_REGISTRY = _RUNTIME_REGISTRY.with_resource(
        _schema["$id"], Resource.from_contents(_schema)
    )
_RUNTIME_VALIDATOR_CLASS = validator_for(_RUNTIME_SCHEMA)
_RUNTIME_VALIDATOR_CLASS.check_schema(_RUNTIME_SCHEMA)
_RUNTIME_VALIDATOR = _RUNTIME_VALIDATOR_CLASS(
    _RUNTIME_SCHEMA, registry=_RUNTIME_REGISTRY
)

_RUNTIME_CLAIM_SCHEMA_BYTES = RUNTIME_CLAIM_SCHEMA_PATH.read_bytes()
if hashlib.sha256(_RUNTIME_CLAIM_SCHEMA_BYTES).hexdigest() != RUNTIME_CLAIM_SCHEMA_SHA256:
    raise RuntimeError("runtime claim schema bytes do not match the code pin")
_RUNTIME_CLAIM_SCHEMA = load_json_file_strict(RUNTIME_CLAIM_SCHEMA_PATH)
if not isinstance(_RUNTIME_CLAIM_SCHEMA, dict):
    raise RuntimeError("runtime claim schema root is not an object")
_RUNTIME_CLAIM_VALIDATOR_CLASS = validator_for(_RUNTIME_CLAIM_SCHEMA)
_RUNTIME_CLAIM_VALIDATOR_CLASS.check_schema(_RUNTIME_CLAIM_SCHEMA)
_RUNTIME_CLAIM_VALIDATOR = _RUNTIME_CLAIM_VALIDATOR_CLASS(_RUNTIME_CLAIM_SCHEMA)


def _schema_path(error: ValidationError) -> str:
    path = "$"
    for component in error.absolute_path:
        path += f"[{component}]" if isinstance(component, int) else f".{component}"
    return path


def _runtime_contract_error(detail: str, *, context: Mapping[str, Any] | None = None) -> None:
    raise RuntimeRecorderError("RUNTIME_RECORD_INVALID", detail, context=context)


def validate_runtime_record(
    document: Any,
    *,
    permit_terminal_input_drift: bool = False,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        _runtime_contract_error("runtime record root must be an object")
    try:
        _RUNTIME_VALIDATOR.validate(document)
    except ValidationError as exc:
        _runtime_contract_error(
            exc.message,
            context={"json_path": _schema_path(exc), "validator": str(exc.validator)},
        )
    expected_seal = canonical_sha256(document, "record_sha256")
    if document["record_sha256"] != expected_seal:
        _runtime_contract_error("runtime record canonical seal does not match")
    try:
        created = datetime.fromisoformat(document["created_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_RECORD_INVALID", "created_at_utc is not an ISO-8601 timestamp"
        ) from exc
    if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
        _runtime_contract_error("created_at_utc is not UTC")

    invocation = document["invocation"]
    working_directory = invocation["working_directory"]
    if working_directory != os.path.abspath(working_directory):
        _runtime_contract_error("runtime working directory is not canonical")
    expected_binary_path = str(
        Path(working_directory) / RELEASE_BINARY_RELATIVE
    )
    if invocation["binary"]["path"] != expected_binary_path:
        _runtime_contract_error("runtime binary is not repository release sim-run")
    manifest_path = document["manifest_identity"]["file"]["path"]
    if manifest_path != os.path.abspath(manifest_path):
        _runtime_contract_error("runtime manifest path is not canonical")
    expected_entries = [
        {"name": name, "value": value} for name, value in _ENVIRONMENT_ITEMS
    ]
    if invocation["environment"] != expected_entries:
        _runtime_contract_error("runtime environment is not the exact allowlist")
    if invocation["environment_sha256"] != _sha256_canonical(
        {"environment": expected_entries}
    ):
        _runtime_contract_error("runtime environment hash does not match")
    argv = invocation["argv"]
    if argv[0] != invocation["binary"]["path"]:
        _runtime_contract_error("runtime argv does not match retained inputs")
    descriptor_argument = _parse_retained_manifest_argument(argv[1])
    if descriptor_argument is None:
        _runtime_contract_error("runtime descriptor argv is not canonical")
    if descriptor_argument[2] != Path(manifest_path).name:
        _runtime_contract_error("runtime argv does not match retained inputs")
    if invocation["affinity_cpus"] != document["host_identity"]["affinity_cpus"]:
        _runtime_contract_error("host and invocation affinity disagree")
    host_unsigned = deepcopy(document["host_identity"])
    observed_host_fingerprint = host_unsigned.pop("host_fingerprint_sha256")
    if observed_host_fingerprint != _sha256_canonical(host_unsigned):
        _runtime_contract_error("host fingerprint does not match host identity")

    workload = document["workload_identity"]
    ordered_keys = sorted(workload["launch_keys"], key=canonical_launch_key_token)
    if ordered_keys != workload["launch_keys"]:
        _runtime_contract_error("runtime launch keys are not canonically ordered")
    if len({canonical_launch_key_token(key) for key in ordered_keys}) != len(ordered_keys):
        _runtime_contract_error("runtime launch keys are not unique")
    if workload["launch_count"] != len(ordered_keys):
        _runtime_contract_error("runtime launch count does not match launch keys")
    if workload["workload_ids"] != sorted(
        {key["workload_id"] for key in ordered_keys}
    ):
        _runtime_contract_error("runtime workload ids do not match launch keys")
    if workload["launch_set_sha256"] != _sha256_canonical(
        {"launch_keys": ordered_keys}
    ):
        _runtime_contract_error("runtime launch-set digest does not match")
    if workload["input_digest"] != document["manifest_identity"]["canonical_sha256"]:
        _runtime_contract_error("runtime input digest does not bind the manifest")

    binding = document["sim_result_binding"]
    if binding["status"] == "bound":
        selected_counts = binding["selected_replay_counts"]
        if selected_counts["record_count"] != selected_counts["instruction_count"]:
            _runtime_contract_error("selected replay issue and record counts disagree")
        if binding["build_identity"]["binary_sha256"] != invocation["binary"]["sha256"]:
            _runtime_contract_error("result build identity does not bind the binary")
        if (
            binding["configuration_identity"]["run_manifest_sha256"]
            != document["manifest_identity"]["canonical_sha256"]
        ):
            _runtime_contract_error("result configuration does not bind the manifest")
        if workload["trace"] != {
            "status": "available",
            "value": binding["trace_identity"],
        }:
            _runtime_contract_error("workload trace differs from the result binding")
        if workload["configuration"] != {
            "status": "available",
            "identity": binding["configuration_identity"],
            "resolved": binding["resolved_configuration"],
        }:
            _runtime_contract_error(
                "workload configuration differs from the result binding"
            )
        target_gpu = document["host_identity"]["target_gpu"]
        if (
            target_gpu["sku"]["status"] == "available"
            and target_gpu["sku"]["value"] != binding["arch_card_identity"]["sku"]
        ) or (
            target_gpu["sm_arch"]["status"] == "available"
            and target_gpu["sm_arch"]["value"]
            != binding["arch_card_identity"]["sm_arch"]
        ):
            _runtime_contract_error("host target architecture differs from the result")
    elif workload["trace"]["status"] != "unavailable" or workload[
        "configuration"
    ]["status"] != "unavailable":
        _runtime_contract_error(
            "unbound result cannot authorize trace or configuration identity"
        )

    samples = document["samples"]
    for index, sample in enumerate(samples):
        if sample["sample_index"] != index:
            _runtime_contract_error("runtime sample indexes are not contiguous")
        if sample["monotonic_stop_ns"] - sample["monotonic_start_ns"] != sample[
            "wall_time_ns"
        ]:
            _runtime_contract_error("runtime sample wall time is not stop minus start")
        if index > 0 and sample["monotonic_start_ns"] < samples[index - 1][
            "monotonic_stop_ns"
        ]:
            _runtime_contract_error("runtime samples overlap on the monotonic timeline")
        if index == 0:
            if sample["kind"] != "warmup" or sample["timed_repeat_index"] is not None:
                _runtime_contract_error("first runtime sample is not the one warm-up")
        else:
            if sample["kind"] != "timed" or sample["timed_repeat_index"] != index - 1:
                _runtime_contract_error("timed runtime sample indexes are inconsistent")
    if len(samples) > 6:
        initial_samples = samples[:6]
        if any(
            sample["termination"] != {"kind": "exit", "exit_code": 0}
            or sample["timed_out"]
            or not sample["stdout"]["capture_complete"]
            or not sample["stderr"]["capture_complete"]
            or sample["stderr"]["size_bytes"] != 0
            or sample["stderr"]["sha256"] != _EMPTY_SHA256
            for sample in initial_samples
        ) or len({sample["stdout"]["sha256"] for sample in initial_samples}) != 1:
            _runtime_contract_error(
                "additional repeats lack a successful deterministic initial protocol"
            )
        if not _is_noisy(
            [sample["wall_time_ns"] for sample in initial_samples[1:]]
        ):
            _runtime_contract_error(
                "additional runtime repeats lack a noisy initial sample set"
            )
    outcome = document["outcome"]
    recorded_input_drift = _is_input_drift_record(document)
    if outcome["status"] == "RUNTIME_EXECUTION_FAILED" and (
        outcome["detail"].startswith("RUNTIME_INPUT_DRIFT:")
        or outcome.get("failure_code") == "RUNTIME_INPUT_DRIFT"
    ) and not recorded_input_drift:
        _runtime_contract_error(
            "runtime input-drift outcome lacks a matching typed failure code"
        )
    if (
        outcome["status"] == "NONDETERMINISTIC_RESULT"
        and outcome.get("failure_code")
        not in {None, "NONDETERMINISTIC_RESULT"}
    ):
        _runtime_contract_error(
            "nondeterministic outcome has an inconsistent typed failure code"
        )
    terminal_input_drift_authorized = (
        permit_terminal_input_drift and recorded_input_drift
    )
    if outcome["status"] in {"MEASURED", "RUNTIME_NOISY"}:
        if len(samples) not in {6, 10}:
            _runtime_contract_error("completed runtime record has the wrong sample count")
        if any(not _sample_has_success_envelope(sample) for sample in samples):
            _runtime_contract_error("completed runtime record contains a failed sample")
        result_hashes = {sample["stdout"]["sha256"] for sample in samples}
        result_sizes = {sample["stdout"]["size_bytes"] for sample in samples}
        if len(result_hashes) != 1 or len(result_sizes) != 1:
            _runtime_contract_error("completed runtime results are not byte-identical")
        binding = document["sim_result_binding"]
        if binding["status"] != "bound" or binding["file_sha256"] not in result_hashes:
            _runtime_contract_error("completed runtime result binding is inconsistent")
        selected_counts = binding["selected_replay_counts"]
        if selected_counts["record_count"] != selected_counts["instruction_count"]:
            _runtime_contract_error("selected replay issue and record counts disagree")
        timed_walls = [sample["wall_time_ns"] for sample in samples[1:]]
        expected_summary = _summary(
            timed_walls,
            record_count=selected_counts["record_count"],
            instruction_count=selected_counts["instruction_count"],
        )
        if outcome["summary"] != expected_summary:
            _runtime_contract_error("runtime summary is not derived from raw samples")
        noisy = _is_noisy(timed_walls)
        if outcome["status"] == "RUNTIME_NOISY":
            if len(samples) != 10 or not noisy:
                _runtime_contract_error("RUNTIME_NOISY does not satisfy the noise gate")
        elif noisy:
            _runtime_contract_error("MEASURED outcome exceeds the noise threshold")
    else:
        if outcome["summary"] is not None:
            _runtime_contract_error("failed runtime outcome carries a summary")
        binding = document["sim_result_binding"]
        if outcome["status"] == "NONDETERMINISTIC_RESULT":
            identities = [_sample_stdout_identity(sample) for sample in samples]
            if (
                outcome["detail"]
                != "release sim-run emitted non-byte-identical results"
                or binding
                != {"status": "unavailable", "reason": "nondeterministic_result"}
                or len(samples) < 2
                or any(not _sample_has_success_envelope(sample) for sample in samples)
                or any(identity != identities[0] for identity in identities[:-1])
                or identities[-1] == identities[0]
            ):
                _runtime_contract_error(
                    "nondeterministic outcome does not stop at the first differing result"
                )
        else:
            if binding["status"] == "unavailable":
                if binding != {
                    "status": "unavailable",
                    "reason": "no_successful_result",
                } or len(samples) > 1:
                    _runtime_contract_error(
                        "execution failure has an impossible unbound sample history"
                    )
            else:
                if not samples:
                    _runtime_contract_error(
                        "execution failure binds a result without a successful sample"
                    )
                bound_hash = binding["file_sha256"]
                successful_prefix = samples if len(samples) == 1 else samples[:-1]
                if any(
                    not _sample_has_success_envelope(sample)
                    or sample["stdout"]["sha256"] != bound_hash
                    for sample in successful_prefix
                ):
                    _runtime_contract_error(
                        "execution failure does not preserve its successful result prefix"
                    )
                final_sample = samples[-1]
                final_matches_bound = (
                    _sample_has_success_envelope(final_sample)
                    and final_sample["stdout"]["sha256"] == bound_hash
                )
                if (
                    final_matches_bound
                    and not _protocol_expects_another_repeat(samples)
                    and not terminal_input_drift_authorized
                ):
                    _runtime_contract_error(
                        "execution failure follows a complete successful runtime protocol"
                    )

    if workload["workload_family"] != {
        "status": "unavailable",
        "reason": "workload_family_authority_unavailable",
    }:
        _runtime_contract_error("workload-family authority is not installed")
    claimability = document["claimability"]
    if claimability["status"] == "claimable":
        _runtime_contract_error("claimable runtime authority is not installed")
    fixture_scope = document["manifest_identity"]["scope"].startswith(
        "fixture-only/"
    ) or any(
        workload_id.startswith("fixture-only.")
        for workload_id in workload["workload_ids"]
    )
    binding = document["sim_result_binding"]
    expected_claimability_reason = (
        "SYNTHETIC_FIXTURE"
        if fixture_scope
        else "DIRTY_BUILD_UNCLAIMABLE"
        if binding["status"] == "bound"
        and binding["build_identity"]["sim_git_dirty"]
        else "WORKLOAD_TIER_AUTHORITY_UNAVAILABLE"
    )
    if claimability["reason"] != expected_claimability_reason:
        _runtime_contract_error(
            "runtime claimability is not derived from installed authority"
        )
    if document["accuracy_binding"]["status"] == "bound":
        _runtime_contract_error("production accuracy-link authority is not installed")
    return document


@dataclass(slots=True)
class _RetainedArtifact:
    name: str
    fd: int
    identity: _StatIdentity
    sha256: str

    def revalidate(self, directory_fd: int, *, role: str) -> None:
        def observe() -> tuple[os.stat_result, os.stat_result]:
            try:
                return (
                    os.stat(
                        self.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    ),
                    os.fstat(self.fd),
                )
            except OSError as exc:
                raise RuntimeRecorderError(
                    "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                    f"{role} identity became unreadable",
                    context={"path": self.name, "errno": exc.errno},
                ) from exc

        def snapshot_matches(
            path_state: os.stat_result, descriptor_state: os.stat_result
        ) -> bool:
            return (
                stat.S_ISREG(path_state.st_mode)
                and stat.S_ISREG(descriptor_state.st_mode)
                and path_state.st_nlink == 1
                and descriptor_state.st_nlink == 1
                and _stat_identity(path_state) == self.identity
                and _stat_identity(descriptor_state) == self.identity
            )

        initial_path_state, initial_descriptor_state = observe()
        if not snapshot_matches(initial_path_state, initial_descriptor_state):
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                f"{role} changed during runtime-record validation",
                context={"path": self.name},
            )
        try:
            observed_sha256 = _hash_fd(self.fd, self.identity.size, role=role)
        except RuntimeRecorderError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                f"{role} bytes became unreadable",
                context={"path": self.name},
            ) from exc
        final_path_state, final_descriptor_state = observe()
        if (
            not snapshot_matches(final_path_state, final_descriptor_state)
            or observed_sha256 != self.sha256
        ):
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                f"{role} changed during runtime-record validation",
                context={"path": self.name},
            )

    def close(self) -> None:
        os.close(self.fd)


def _open_relative_artifact(
    directory_fd: int,
    name: str,
    *,
    role: str,
    maximum_bytes: int,
) -> _RetainedArtifact:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | nofollow,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            f"{role} could not be opened without following a symlink",
            context={"path": name, "errno": exc.errno},
        ) from exc
    try:
        descriptor_state = os.fstat(fd)
        path_state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        identity = _stat_identity(descriptor_state)
        if (
            not stat.S_ISREG(descriptor_state.st_mode)
            or not stat.S_ISREG(path_state.st_mode)
            or descriptor_state.st_nlink != 1
            or path_state.st_nlink != 1
            or _stat_identity(path_state) != identity
        ):
            raise RuntimeRecorderError(
                "RUNTIME_RECORD_INVALID",
                "sample artifact must be a named regular file without hardlink aliases",
            )
        if identity.size > maximum_bytes:
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                f"{role} exceeds the retained artifact byte limit",
                context={"path": name, "size_bytes": identity.size},
            )
        digest = _hash_fd(fd, identity.size, role=role)
        return _RetainedArtifact(
            name=name,
            fd=fd,
            identity=identity,
            sha256=digest,
        )
    except BaseException:
        os.close(fd)
        raise


def _verify_named_file(
    identity: Mapping[str, Any],
    *,
    role: str,
    maximum_bytes: int,
    require_executable: bool = False,
    require_single_link: bool = True,
) -> _RetainedFile:
    retained = _open_retained_file(
        identity["path"],
        role=role,
        maximum_bytes=maximum_bytes,
        require_executable=require_executable,
        require_single_link=require_single_link,
    )
    if retained.size != identity["size_bytes"] or retained.sha256 != identity["sha256"]:
        retained.close()
        raise RuntimeRecorderError(
            "RUNTIME_INPUT_IDENTITY_MISMATCH",
            f"retained {role} differs from the runtime record",
        )
    return retained


def _validate_result_against_recorded_inputs(
    result_bytes: bytes,
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Replay result bytes against identities retained in an input-drift record."""
    result = _decode_object(result_bytes, source="retained sim-run stdout")
    try:
        validate_result(result)
    except ContractValidationError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            "retained sim-run result contract is invalid",
        ) from exc
    if result_bytes != _canonical_json_line(result):
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            "retained sim-run result is not canonical JSON",
        )
    if (
        result["build_identity"]["cargo_profile"] != "release"
        or result["build_identity"]["binary_sha256"]
        != document["invocation"]["binary"]["sha256"]
    ):
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            "retained result build identity differs from the recorded executable",
        )
    configuration = result["configuration_identity"]
    if (
        configuration["run_manifest_sha256"]
        != document["manifest_identity"]["canonical_sha256"]
        or configuration["resolved_configuration_sha256"]
        != _sha256_canonical(result["resolved_configuration"])
        or result["arch_card_identity"]["card_sha256"]
        != configuration["arch_card_sha256"]
    ):
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            "retained result configuration differs from recorded input authority",
        )
    observed_launches = [
        deepcopy(launch["launch_key"]) for launch in result["launches"]
    ]
    observed_launches.sort(key=canonical_launch_key_token)
    if observed_launches != document["workload_identity"]["launch_keys"]:
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            "retained result launch set differs from recorded input authority",
        )
    target_gpu = document["host_identity"]["target_gpu"]
    if (
        target_gpu["sku"]["status"] == "available"
        and target_gpu["sku"]["value"] != result["arch_card_identity"]["sku"]
    ) or (
        target_gpu["sm_arch"]["status"] == "available"
        and target_gpu["sm_arch"]["value"]
        != result["arch_card_identity"]["sm_arch"]
    ):
        raise RuntimeRecorderError(
            "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
            "retained result architecture differs from recorded input authority",
        )
    return result, hashlib.sha256(result_bytes).hexdigest()


def _load_runtime_record(
    path: str | Path,
    *,
    repo_root: Path,
    permit_recorded_input_drift: bool = False,
) -> dict[str, Any]:
    retained = _open_retained_file(
        path,
        role="runtime record",
        maximum_bytes=MAX_RUNTIME_RECORD_BYTES,
    )
    binary: _RetainedFile | None = None
    manifest: _RetainedFile | None = None
    sample_directory_fd: int | None = None
    retained_artifacts: list[_RetainedArtifact] = []
    try:
        payload = retained.read_bytes(role="runtime record")
        document = _decode_object(payload, source=str(retained.path))
        recorded_input_drift = (
            permit_recorded_input_drift and _is_input_drift_record(document)
        )
        validate_runtime_record(
            document,
            permit_terminal_input_drift=recorded_input_drift,
        )
        if payload != _canonical_json_line(document):
            raise RuntimeRecorderError(
                "RUNTIME_RECORD_INVALID", "runtime record is not canonical JSON"
            )

        canonical_repo_root = Path(os.path.abspath(os.fspath(repo_root)))
        if document["invocation"]["working_directory"] != str(canonical_repo_root):
            raise RuntimeRecorderError(
                "RUNTIME_RECORD_INVALID",
                "runtime record is not bound to this repository root",
            )
        if document["invocation"]["binary"]["path"] != str(
            canonical_repo_root / RELEASE_BINARY_RELATIVE
        ):
            raise RuntimeRecorderError(
                "RUNTIME_RECORD_INVALID",
                "runtime record does not name this repository release sim-run",
            )

        manifest_document: dict[str, Any] | None = None
        manifest_canonical_sha256 = document["manifest_identity"][
            "canonical_sha256"
        ]
        if not recorded_input_drift:
            binary = _verify_named_file(
                document["invocation"]["binary"],
                role="release sim-run",
                maximum_bytes=MAX_BINARY_BYTES,
                require_executable=True,
                require_single_link=False,
            )
            manifest = _verify_named_file(
                document["manifest_identity"]["file"],
                role="run manifest",
                maximum_bytes=MAX_MANIFEST_BYTES,
            )
            manifest_document, observed_manifest_canonical_sha256 = (
                _manifest_document(manifest)
            )
            manifest_identity = document["manifest_identity"]
            if (
                observed_manifest_canonical_sha256
                != manifest_identity["canonical_sha256"]
                or manifest_document["scope"] != manifest_identity["scope"]
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_IDENTITY_MISMATCH",
                    "retained manifest semantics differ from the runtime record",
                )
            launch_keys, launch_set_sha256 = _launch_projection(manifest_document)
            workload = document["workload_identity"]
            if (
                launch_keys != workload["launch_keys"]
                or launch_set_sha256 != workload["launch_set_sha256"]
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_INPUT_IDENTITY_MISMATCH",
                    "retained manifest launch set differs from the runtime record",
                )

        sample_directory_name = f"{retained.path.name}.samples"
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        try:
            sample_directory_fd = os.open(
                sample_directory_name,
                os.O_RDONLY | os.O_CLOEXEC | nofollow | directory,
                dir_fd=retained.parent_fd,
            )
            sample_directory_state = os.fstat(sample_directory_fd)
            sample_directory_path_state = os.stat(
                sample_directory_name,
                dir_fd=retained.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                "runtime sample directory could not be retained",
                context={"errno": exc.errno},
            ) from exc
        sample_directory_identity = _stat_identity(sample_directory_state)
        if (
            not stat.S_ISDIR(sample_directory_state.st_mode)
            or _stat_identity(sample_directory_path_state)
            != sample_directory_identity
        ):
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                "runtime sample directory path and descriptor disagree",
            )

        result_payload: bytes | None = None
        first_stdout_payload: bytes | None = None
        terminal_stdout_payload: bytes | None = None
        successful_prefix_bytes_identical = True
        nondeterministic_payloads: list[bytes] = []
        seen_paths: set[str] = set()
        binding = document["sim_result_binding"]
        for sample in document["samples"]:
            kind = sample["kind"]
            index = sample["sample_index"]
            for stream_name in ("stdout", "stderr"):
                artifact = sample[stream_name]
                suffix = "json" if stream_name == "stdout" else "bin"
                artifact_name = f"sample-{index:03d}-{kind}.{stream_name}.{suffix}"
                expected_path = f"{sample_directory_name}/{artifact_name}"
                if artifact["path"] != expected_path or expected_path in seen_paths:
                    raise RuntimeRecorderError(
                        "RUNTIME_RECORD_INVALID",
                        "sample artifact path does not match its sample identity",
                    )
                seen_paths.add(expected_path)
                retained_artifact = _open_relative_artifact(
                    sample_directory_fd,
                    artifact_name,
                    role=f"runtime {stream_name} artifact",
                    maximum_bytes=MAX_RESULT_STREAM_BYTES,
                )
                retained_artifacts.append(retained_artifact)
                if retained_artifact.identity.size != artifact["size_bytes"]:
                    raise RuntimeRecorderError(
                        "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                        "sample artifact size does not match the runtime record",
                    )
                if retained_artifact.sha256 != artifact["sha256"]:
                    raise RuntimeRecorderError(
                        "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                        "sample artifact hash does not match the runtime record",
                    )
                if stream_name == "stdout":
                    stdout_payload = _read_fd(
                        retained_artifact.fd,
                        retained_artifact.identity.size,
                        role="retained sim-run stdout",
                    )
                    if first_stdout_payload is None:
                        first_stdout_payload = stdout_payload
                    if index < len(document["samples"]) - 1:
                        successful_prefix_bytes_identical = (
                            successful_prefix_bytes_identical
                            and stdout_payload == first_stdout_payload
                        )
                    else:
                        terminal_stdout_payload = stdout_payload
                    if (
                        binding["status"] == "bound"
                        and retained_artifact.sha256 == binding["file_sha256"]
                        and result_payload is None
                    ):
                        result_payload = stdout_payload
                    if document["outcome"]["status"] == "NONDETERMINISTIC_RESULT":
                        nondeterministic_payloads.append(stdout_payload)

        def replay_result(result_bytes: bytes) -> tuple[dict[str, Any], str]:
            if recorded_input_drift:
                return _validate_result_against_recorded_inputs(
                    result_bytes, document
                )
            assert binary is not None and manifest is not None
            assert manifest_document is not None
            observation = _SpawnObservation(
                background_load=(0.0, 0.0, 0.0),
                monotonic_start_ns=0,
                monotonic_stop_ns=1,
                user_cpu_ns=0,
                system_cpu_ns=0,
                peak_rss_kib=0,
                return_code=0,
                timed_out=False,
                stdout=result_bytes,
                stdout_complete=True,
                stderr=b"",
                stderr_complete=True,
                capture_error=None,
            )
            return _validate_result_binding(
                observation,
                binary=binary,
                manifest=manifest,
                manifest_document=manifest_document,
                manifest_canonical_sha256=manifest_canonical_sha256,
            )

        if binding["status"] == "bound":
            if result_payload is None:
                raise RuntimeRecorderError(
                    "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                    "bound sim-run result bytes are not retained",
                )
            result_document, file_sha256 = replay_result(result_payload)
            if _result_binding(result_document, file_sha256, "unused") != binding:
                raise RuntimeRecorderError(
                    "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                    "bound sim-run result projection differs from the runtime record",
                )

        outcome_status = document["outcome"]["status"]
        if outcome_status in {"MEASURED", "RUNTIME_NOISY"}:
            if (
                not successful_prefix_bytes_identical
                or terminal_stdout_payload != first_stdout_payload
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                    "completed sim-run result bytes are not byte-identical",
                )
        elif outcome_status == "NONDETERMINISTIC_RESULT":
            if (
                len(nondeterministic_payloads) != len(document["samples"])
                or not successful_prefix_bytes_identical
                or terminal_stdout_payload == first_stdout_payload
            ):
                raise RuntimeRecorderError(
                    "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                    "nondeterministic results do not stop at the first differing bytes",
                )
            for result_variant in nondeterministic_payloads:
                replay_result(result_variant)
        elif terminal_stdout_payload is not None:
            final_sample = document["samples"][-1]
            if _sample_has_success_envelope(final_sample):
                try:
                    replay_result(terminal_stdout_payload)
                except RuntimeRecorderError:
                    pass
                else:
                    if not recorded_input_drift:
                        if binding["status"] == "unavailable":
                            raise RuntimeRecorderError(
                                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                                "execution failure terminal sample is a valid sim-run result",
                            )
                        if terminal_stdout_payload != result_payload:
                            raise RuntimeRecorderError(
                                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                                "execution failure terminal sample is a valid differing result",
                            )

        for artifact in retained_artifacts:
            artifact.revalidate(
                sample_directory_fd,
                role="retained runtime sample artifact",
            )
        try:
            final_sample_directory_state = os.fstat(sample_directory_fd)
            final_sample_directory_path_state = os.stat(
                sample_directory_name,
                dir_fd=retained.parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                "runtime sample directory became unreadable during validation",
                context={"errno": exc.errno},
            ) from exc
        if (
            _stat_identity(final_sample_directory_state) != sample_directory_identity
            or _stat_identity(final_sample_directory_path_state)
            != sample_directory_identity
        ):
            raise RuntimeRecorderError(
                "RUNTIME_SAMPLE_IDENTITY_MISMATCH",
                "runtime sample directory changed during validation",
            )
        if not recorded_input_drift:
            assert binary is not None and manifest is not None
            binary.revalidate(role="release sim-run")
            manifest.revalidate(role="run manifest")
        retained.revalidate(role="runtime record")
        return document
    finally:
        for artifact in retained_artifacts:
            artifact.close()
        if sample_directory_fd is not None:
            os.close(sample_directory_fd)
        if manifest is not None:
            manifest.close()
        if binary is not None:
            binary.close()
        retained.close()


def load_runtime_record(path: str | Path) -> dict[str, Any]:
    """Load a record bound to this checkout's fixed release executable."""

    return _load_runtime_record(path, repo_root=ROOT)


# The exploratory runtime wire above is intentionally frozen. Claimability is an
# evaluator-side, post-result decision published through this detached contract.
def _claim_contract_error(
    detail: str,
    *,
    code: str = "RUNTIME_CLAIM_INVALID",
    context: Mapping[str, Any] | None = None,
) -> None:
    raise RuntimeRecorderError(code, detail, context=context)


def _claim_exact_keys(value: object, expected: set[str], role: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _claim_contract_error(f"{role} fields are not exact")
    return value


def _claim_sha256(value: object, role: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _claim_contract_error(f"{role} is not lowercase SHA-256")
    return value


def _claim_safe_id(value: object, role: str) -> str:
    if type(value) is not str or _SAFE_CLAIM_ID_RE.fullmatch(value) is None:
        _claim_contract_error(f"{role} is not a schema-safe identifier")
    return value


def _claim_file_identity(retained: _RetainedFile) -> dict[str, Any]:
    return {
        "path": str(retained.path),
        "size_bytes": retained.size,
        "sha256": retained.sha256,
    }


def _decode_claim_object(payload: bytes, *, source: str, role: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = loads_json_strict(text, source=source)
    except (
        UnicodeDecodeError,
        ContractValidationError,
        MemoryError,
        RecursionError,
    ) as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_AUTHORITY_INVALID",
            f"{role} is not strict JSON",
        ) from exc
    if type(value) is not dict:
        _claim_contract_error(f"{role} root is not an object")
    return value


def _load_runtime_claim_evidence(
    path: str | Path, *, repo_root: Path
) -> tuple[dict[str, Any], _RetainedFile]:
    retained = _open_retained_file(
        path,
        role="runtime record",
        maximum_bytes=MAX_RUNTIME_RECORD_BYTES,
    )
    try:
        payload = retained.read_bytes(role="runtime record")
        preview = _decode_claim_object(
            payload,
            source=str(retained.path),
            role="runtime record",
        )
        recorded_input_drift = _is_input_drift_record(preview)
        document = _load_runtime_record(
            retained.path,
            repo_root=repo_root,
            permit_recorded_input_drift=recorded_input_drift,
        )
        if preview != document or payload != _canonical_json_line(document):
            _claim_contract_error("runtime record bytes changed across claim validation")
        retained.revalidate(role="runtime record")
        return document, retained
    except BaseException:
        retained.close()
        raise


def _runtime_claimability_reason(document: Mapping[str, Any]) -> str | None:
    workload = document["workload_identity"]
    fixture = document["manifest_identity"]["scope"].startswith(
        "fixture-only/"
    ) or any(
        workload_id.startswith("fixture-only.")
        for workload_id in workload["workload_ids"]
    )
    if fixture:
        return "SYNTHETIC_FIXTURE"
    binding = document["sim_result_binding"]
    if binding["status"] == "bound" and binding["build_identity"]["sim_git_dirty"]:
        return "DIRTY_BUILD_UNCLAIMABLE"
    status = document["outcome"]["status"]
    if status == "RUNTIME_NOISY":
        return "RUNTIME_NOISY"
    if status == "NONDETERMINISTIC_RESULT":
        return "NONDETERMINISTIC_RESULT"
    if status == "RUNTIME_EXECUTION_FAILED":
        if any(sample["timed_out"] for sample in document["samples"]):
            return "RUNTIME_TIMEOUT"
        return "RUNTIME_EXECUTION_FAILED"
    if status != "MEASURED" or binding["status"] != "bound":
        return "RUNTIME_PROTOCOL_PARTIAL"
    build = binding["build_identity"]
    if (
        build["cargo_profile"] != "release"
        or build["claimability"]["status"] != "claimable"
        or not re.fullmatch(r"[0-9a-f]{40}", build["sim_git_commit"])
        or build["sim_git_commit"] == "0" * 40
    ):
        return "DIRTY_BUILD_UNCLAIMABLE"
    return None


def _runtime_derived_binding(document: Mapping[str, Any]) -> dict[str, Any]:
    binding = document["sim_result_binding"]
    if binding["status"] != "bound":
        _claim_contract_error("runtime result binding is unavailable")
    workload_ids = document["workload_identity"]["workload_ids"]
    execution_ids = sorted(
        {key["matched_execution_id"] for key in document["workload_identity"]["launch_keys"]}
    )
    if len(workload_ids) != 1 or len(execution_ids) != 1:
        _claim_contract_error("claimable runtime must describe one workload execution")
    architecture = binding["arch_card_identity"]
    return {
        "status": "bound",
        "manifest_canonical_sha256": document["manifest_identity"][
            "canonical_sha256"
        ],
        "trace_identity_sha256": _sha256_canonical(binding["trace_identity"]),
        "launch_set_sha256": document["workload_identity"]["launch_set_sha256"],
        "architecture_identity_sha256": _sha256_canonical(architecture),
        "configuration_identity_sha256": _sha256_canonical(
            {
                "identity": binding["configuration_identity"],
                "resolved": binding["resolved_configuration"],
            }
        ),
        "binary_sha256": binding["build_identity"]["binary_sha256"],
        "workload_id": workload_ids[0],
        "matched_execution_id": execution_ids[0],
        "binary_architecture": architecture["sm_arch"],
        "gpu_sku": architecture["sku"],
    }


def _load_claim_result(
    runtime_record_path: Path,
    document: Mapping[str, Any],
    *,
    artifact_path_override: Path | None = None,
) -> dict[str, Any]:
    if document["outcome"]["status"] != "MEASURED" or not document["samples"]:
        _claim_contract_error("final MEASURED result bytes are unavailable")
    binding = document["sim_result_binding"]
    if binding["status"] != "bound":
        _claim_contract_error("final MEASURED result binding is unavailable")
    identities = {
        (sample["stdout"]["sha256"], sample["stdout"]["size_bytes"])
        for sample in document["samples"]
    }
    if len(identities) != 1 or next(iter(identities))[0] != binding["file_sha256"]:
        _claim_contract_error("runtime results are not all byte-identical")
    artifact = document["samples"][0]["stdout"]
    artifact_path = (
        artifact_path_override
        if artifact_path_override is not None
        else runtime_record_path.parent / artifact["path"]
    )
    retained = _open_retained_file(
        artifact_path,
        role="sealed sim-run result",
        maximum_bytes=MAX_RESULT_STREAM_BYTES,
    )
    try:
        if retained.size != artifact["size_bytes"] or retained.sha256 != artifact["sha256"]:
            _claim_contract_error("sealed sim-run result artifact identity differs")
        payload = retained.read_bytes(role="sealed sim-run result")
        result = _decode_claim_object(
            payload,
            source=str(retained.path),
            role="sealed sim-run result",
        )
        try:
            validate_result(result)
        except ContractValidationError as exc:
            raise RuntimeRecorderError(
                "RUNTIME_CLAIM_INVALID",
                "sealed sim-run result contract is invalid",
            ) from exc
        if payload != _canonical_json_line(result):
            _claim_contract_error("sealed sim-run result is not canonical JSON")
        if (
            hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
            or result["result_sha256"] != binding["result_sha256"]
        ):
            _claim_contract_error("sealed sim-run result hash differs from runtime binding")
        return result
    finally:
        retained.close()


def _load_claim_workload_authority(
    path: str | Path,
    *,
    fixture_mode: bool,
    schema_path: Path = WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_PATH,
) -> tuple[dict[str, Any], _RetainedFile]:
    retained = _open_retained_file(
        path,
        role="workload execution authority",
        maximum_bytes=MAX_CLAIM_AUTHORITY_BYTES,
    )
    try:
        payload = retained.read_bytes(role="workload execution authority")
        schema_bytes, schema_snapshot = RetainedPayloadSnapshot.read_path(
            schema_path
        )
        loader_name = (
            "load_workload_execution_authority_fixture"
            if fixture_mode
            else "load_workload_execution_authority"
        )
        loader = getattr(_launch_crosswalk, loader_name, None)
        if loader is None:
            _claim_contract_error(
                "workload execution authority loader is unavailable",
                code="RUNTIME_CLAIM_AUTHORITY_UNAVAILABLE",
            )
        try:
            authority = loader(
                payload,
                schema_bytes=schema_bytes,
                schema_snapshot=schema_snapshot,
            )
        except (EvaluatorAuthorityUnavailable, ImmutableArtifactError) as exc:
            raise RuntimeRecorderError(
                "RUNTIME_CLAIM_AUTHORITY_INVALID",
                "workload execution authority failed detached admission",
            ) from exc
        retained.revalidate(role="workload execution authority")
        return authority, retained
    except BaseException:
        retained.close()
        raise


def _require_workload_runtime_binding(
    authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> None:
    expected_keys = [
        deepcopy(row["launch_key"])
        for row in authority["expected_launch_roster"]
    ]
    expected_keys.sort(key=canonical_launch_key_token)
    if expected_keys != runtime["workload_identity"]["launch_keys"]:
        _claim_contract_error("workload authority launch roster differs from runtime")
    if (
        authority["expected_launch_count"] != runtime["workload_identity"]["launch_count"]
        or authority["workload_id"] != derived["workload_id"]
        or authority["matched_execution_id"] != derived["matched_execution_id"]
        or authority["executable_sha256"]
        != runtime["invocation"]["binary"]["sha256"]
        or authority["architecture"]
        != {
            "binary_architecture": derived["binary_architecture"],
            "gpu_sku": derived["gpu_sku"],
        }
    ):
        _claim_contract_error("workload authority cohort or architecture differs from runtime")

_SOURCE_AUTHORITY_KEYS = {
    "schema_version",
    "authority_role",
    "contract_mode",
    "source_git_commit",
    "source_clean",
    "current_source",
    "gate",
    "producer_files",
    "build",
    "authority_sha256",
}
_SOURCE_PRODUCER_FILES = (
    _CLAIM_PRODUCER_RELATIVE.as_posix(),
    _CLAIM_SCHEMA_RELATIVE.as_posix(),
)


def _validate_claim_source_authority(
    authority: dict[str, Any],
    *,
    observed_file_sha256: str,
    expected_file_sha256: str,
    runtime: Mapping[str, Any],
    producer_files: Mapping[str, Mapping[str, Any]],
    binary_identity: Mapping[str, Any],
    fixture_mode: bool,
) -> None:
    _claim_exact_keys(authority, _SOURCE_AUTHORITY_KEYS, "source authority")
    expected_mode = (
        "FIXTURE_ONLY_NON_AUTHORITATIVE"
        if fixture_mode
        else "TARGET_FREE_PRODUCTION"
    )
    if (
        authority["schema_version"] != "runtime_claim_source_authority.v1"
        or authority["authority_role"] != "detached_clean_source_gate_authority"
        or authority["contract_mode"] != expected_mode
        or authority["source_clean"] is not True
        or authority["current_source"] != "matched"
    ):
        _claim_contract_error("source authority is not current clean authority")
    commit = authority["source_git_commit"]
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _claim_contract_error("source authority commit is malformed")
    gate = _claim_exact_keys(
        authority["gate"],
        {"content_manifest_sha256", "receipt_sha256"},
        "source authority gate",
    )
    for name, value in gate.items():
        _claim_sha256(value, f"source authority gate {name}")
    observed_producers = _claim_exact_keys(
        authority["producer_files"],
        set(_SOURCE_PRODUCER_FILES),
        "source authority producer files",
    )
    for relative_path in _SOURCE_PRODUCER_FILES:
        identity = _claim_exact_keys(
            observed_producers[relative_path],
            {"size_bytes", "sha256"},
            f"source authority producer {relative_path}",
        )
        if type(identity["size_bytes"]) is not int or isinstance(
            identity["size_bytes"], bool
        ) or identity["size_bytes"] <= 0:
            _claim_contract_error("source authority producer size is invalid")
        _claim_sha256(identity["sha256"], "source authority producer hash")
    build = _claim_exact_keys(
        authority["build"],
        {"size_bytes", "sha256", "cargo_profile"},
        "source authority build",
    )
    if (
        type(build["size_bytes"]) is not int
        or isinstance(build["size_bytes"], bool)
        or build["size_bytes"] <= 0
        or build["cargo_profile"] != "release"
    ):
        _claim_contract_error("source authority build identity is invalid")
    _claim_sha256(build["sha256"], "source authority build hash")
    _claim_sha256(authority["authority_sha256"], "source authority seal")
    if authority["authority_sha256"] != canonical_sha256(
        authority, "authority_sha256"
    ):
        _claim_contract_error("source authority seal does not match")
    _claim_sha256(expected_file_sha256, "external source authority anchor")
    if observed_file_sha256 != expected_file_sha256:
        _claim_contract_error("source authority file hash does not match external pin")
    runtime_build = runtime["sim_result_binding"]["build_identity"]
    expected_build = {
        "size_bytes": binary_identity["size_bytes"],
        "sha256": binary_identity["sha256"],
        "cargo_profile": runtime_build["cargo_profile"],
    }
    if (
        authority["source_git_commit"] != runtime_build["sim_git_commit"]
        or authority["producer_files"] != dict(producer_files)
        or authority["build"] != expected_build
    ):
        _claim_contract_error(
            "source authority does not bind exact producer, schema, commit, and build bytes"
        )


def _source_authority_projection(
    authority: Mapping[str, Any],
    retained: _RetainedFile,
    *,
    file_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "bound",
        "file": deepcopy(
            dict(file_identity)
            if file_identity is not None
            else _claim_file_identity(retained)
        ),
        "authority_sha256": authority["authority_sha256"],
        "source_git_commit": authority["source_git_commit"],
        "contract_mode": authority["contract_mode"],
    }


_DETACHED_EVALUATOR_KEYS = {
    "schema_version",
    "authority_id",
    "authority_role",
    "contract_mode",
    "admission_status",
    "workload_execution_authority_model_seal_sha256",
    "result_binding",
    "prediction_binding",
    "report_binding",
    "label_authority",
    "bindings",
    "authority_sha256",
}
_EVALUATOR_BINDING_HASH_KEYS = {
    "manifest_canonical_sha256",
    "trace_identity_sha256",
    "launch_set_sha256",
    "architecture_identity_sha256",
    "configuration_identity_sha256",
}


def _load_claim_json_file(
    path: str | Path,
    *,
    role: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], _RetainedFile, bytes]:
    retained = _open_retained_file(path, role=role, maximum_bytes=maximum_bytes)
    try:
        payload = retained.read_bytes(role=role)
        document = _decode_claim_object(payload, source=str(retained.path), role=role)
        if payload != _canonical_json_line(document):
            _claim_contract_error(f"{role} is not canonical JSON")
        return document, retained, payload
    except BaseException:
        retained.close()
        raise


def _result_prediction_projection_sha256(result: Mapping[str, Any]) -> str:
    return _sha256_canonical(
        {
            "schema_version": result["schema_version"],
            "result_sha256": result["result_sha256"],
            "aggregate": result["aggregate"],
            "launches": result["launches"],
            "arch_card_identity": result["arch_card_identity"],
            "configuration_identity": result["configuration_identity"],
        }
    )


def _evaluator_prediction_binding(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    label_binding: Mapping[str, Any],
) -> dict[str, str]:
    projection_sha256 = _result_prediction_projection_sha256(result)
    report_outcome_sha256 = _sha256_canonical(report["outcome"])
    return {
        "projection_sha256": projection_sha256,
        "report_outcome_sha256": report_outcome_sha256,
        "semantic_binding_sha256": _sha256_canonical(
            {
                "result_prediction_projection_sha256": projection_sha256,
                "report_outcome_sha256": report_outcome_sha256,
                "label_authority": dict(label_binding),
            }
        ),
    }


def _validate_detached_evaluator_authority(
    authority: dict[str, Any],
    report: dict[str, Any],
    *,
    report_file_sha256: str,
    workload_authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
    derived: Mapping[str, Any],
    fixture_mode: bool,
) -> None:
    _claim_exact_keys(authority, _DETACHED_EVALUATOR_KEYS, "evaluator authority")
    expected_mode = (
        "FIXTURE_ONLY_NON_AUTHORITATIVE"
        if fixture_mode
        else "TARGET_FREE_PRODUCTION"
    )
    if (
        authority["schema_version"] != "detached_production_evaluator_authority.v1"
        or authority["authority_role"]
        != "detached_production_accuracy_report_authority"
        or authority["contract_mode"] != expected_mode
        or authority["admission_status"] != "admitted"
    ):
        _claim_contract_error("detached evaluator authority is not admitted production authority")
    _claim_safe_id(authority["authority_id"], "evaluator authority id")
    _claim_sha256(authority["authority_sha256"], "evaluator authority seal")
    if authority["authority_sha256"] != canonical_sha256(
        authority, "authority_sha256"
    ):
        _claim_contract_error("detached evaluator authority seal does not match")
    if (
        authority["workload_execution_authority_model_seal_sha256"]
        != workload_authority["model_seal_sha256"]
    ):
        _claim_contract_error("evaluator and workload authorities do not share one cohort")

    result_binding = _claim_exact_keys(
        authority["result_binding"],
        {"file_sha256", "result_sha256"},
        "evaluator result binding",
    )
    prediction_binding = _claim_exact_keys(
        authority["prediction_binding"],
        {
            "projection_sha256",
            "report_outcome_sha256",
            "semantic_binding_sha256",
        },
        "evaluator prediction binding",
    )
    report_binding = _claim_exact_keys(
        authority["report_binding"],
        {"file_sha256", "report_id", "report_sha256"},
        "evaluator report binding",
    )
    label_binding = _claim_exact_keys(
        authority["label_authority"],
        {
            "matched_labels_sha256",
            "label_set_id",
            "label_set_sha256",
            "crosswalk_id",
            "crosswalk_sha256",
        },
        "evaluator label authority",
    )
    binding_hashes = _claim_exact_keys(
        authority["bindings"],
        _EVALUATOR_BINDING_HASH_KEYS,
        "evaluator runtime bindings",
    )
    for name, value in (
        *result_binding.items(),
        *prediction_binding.items(),
        *binding_hashes.items(),
    ):
        _claim_sha256(value, f"evaluator {name}")
    _claim_sha256(report_binding["file_sha256"], "accuracy report file hash")
    _claim_sha256(report_binding["report_sha256"], "accuracy report seal")


def _validate_detached_evaluator_report_binding(
    authority: Mapping[str, Any],
    report: dict[str, Any],
    result: Mapping[str, Any],
    *,
    report_file_sha256: str,
    runtime: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> None:
    label_binding = authority["label_authority"]
    try:
        independent_labels = MatchedLabelAuthority(
            matched_labels_sha256=label_binding["matched_labels_sha256"],
            label_set_id=label_binding["label_set_id"],
            label_set_sha256=label_binding["label_set_sha256"],
            crosswalk_id=label_binding["crosswalk_id"],
            crosswalk_sha256=label_binding["crosswalk_sha256"],
        )
        validate_report(report, label_authority=independent_labels)
    except ContractValidationError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_AUTHORITY_INVALID",
            "accuracy report failed detached evaluator validation",
        ) from exc
    identities = report["identities"]
    report_id = identities["report_id"]
    if report_id.lower().startswith(("synthetic", "fixture")):
        _claim_contract_error("accuracy report is fixture-only rather than production")
    if (
        report["outcome"]["status"] != "admitted"
        or identities["simulator_git_dirty"]
        or identities["evaluator_git_dirty"]
    ):
        _claim_contract_error("accuracy report is not a clean admitted production report")
    runtime_result = runtime["sim_result_binding"]
    expected_result_binding = {
        "file_sha256": runtime_result["file_sha256"],
        "result_sha256": runtime_result["result_sha256"],
    }
    expected_report_binding = {
        "file_sha256": report_file_sha256,
        "report_id": report_id,
        "report_sha256": report["report_sha256"],
    }
    if authority["result_binding"] != expected_result_binding:
        _claim_contract_error("evaluator authority names a different sealed result")
    expected_prediction_binding = _evaluator_prediction_binding(
        result,
        report,
        label_binding,
    )
    if authority["prediction_binding"] != expected_prediction_binding:
        _claim_contract_error(
            "evaluator authority prediction semantics differ from sealed result"
        )
    if authority["report_binding"] != expected_report_binding:
        _claim_contract_error("evaluator authority names a different accuracy report")
    if (
        identities["result_file_sha256"] != runtime_result["file_sha256"]
        or identities["result_canonical_sha256"] != runtime_result["result_sha256"]
        or identities["arch_card_sha256"]
        != runtime_result["arch_card_identity"]["card_sha256"]
        or identities["binary_architecture"] != derived["binary_architecture"]
        or identities["gpu_sku"] != derived["gpu_sku"]
    ):
        _claim_contract_error("accuracy report does not bind the runtime result identity")
    expected_hash_bindings = {
        name: derived[name] for name in _EVALUATOR_BINDING_HASH_KEYS
    }
    if authority["bindings"] != expected_hash_bindings:
        _claim_contract_error("evaluator authority runtime bindings do not match")
    report_labels = identities["label_binding"]
    if any(report_labels[name] != value for name, value in label_binding.items()):
        _claim_contract_error("evaluator authority label binding differs from report")


_TIER_AUTHORITY_KEYS = {
    "tier_id",
    "admission_status",
    "classification",
    "workload_execution_authority_model_seal_sha256",
    "manifest_canonical_sha256",
    "trace_identity_sha256",
    "launch_set_sha256",
    "architecture_identity_sha256",
    "configuration_identity_sha256",
}


def _validate_claim_content_manifest_header(manifest: dict[str, Any]) -> None:
    _claim_exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_id",
            "authority_role",
            "workload_tier_authority",
            "files",
            "manifest_sha256",
        },
        "runtime claim content manifest",
    )
    if (
        manifest["schema_version"] != "runtime_claim_content_manifest.v1"
        or manifest["authority_role"] != "external_runtime_claim_content_anchor"
    ):
        _claim_contract_error("runtime claim content manifest identity is invalid")
    _claim_safe_id(manifest["manifest_id"], "content manifest id")
    _claim_sha256(manifest["manifest_sha256"], "content manifest seal")
    if manifest["manifest_sha256"] != canonical_sha256(
        manifest, "manifest_sha256"
    ):
        _claim_contract_error("runtime claim content manifest seal does not match")


def _validate_claim_content_manifest(
    manifest: dict[str, Any],
    *,
    expected_files: Mapping[str, Mapping[str, Any]],
    expected_external_sha256: str,
    observed_external_sha256: str,
    workload_authority: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_claim_content_manifest_header(manifest)
    _claim_sha256(expected_external_sha256, "external content manifest anchor")
    if expected_external_sha256 != observed_external_sha256:
        _claim_contract_error("external content manifest file hash does not match")

    files = manifest["files"]
    if (
        type(files) is not list
        or len(files) != len(_CLAIM_FILE_ROLES)
        or any(type(item) is not dict for item in files)
    ):
        _claim_contract_error("content manifest file closure is not exact and ordered")
    if [item["role"] if "role" in item else None for item in files] != list(
        _CLAIM_FILE_ROLES
    ):
        _claim_contract_error("content manifest file closure is not exact and ordered")
    observed_files: dict[str, dict[str, Any]] = {}
    for item in files:
        _claim_exact_keys(
            item,
            {"role", "path", "size_bytes", "sha256"},
            "content manifest file",
        )
        role = item["role"]
        if type(item["path"]) is not str or not item["path"]:
            _claim_contract_error("content manifest file path is invalid")
        if item["path"] != os.path.abspath(item["path"]):
            _claim_contract_error("content manifest file path is not canonical absolute")
        if (
            type(item["size_bytes"]) is not int
            or isinstance(item["size_bytes"], bool)
            or item["size_bytes"] <= 0
        ):
            _claim_contract_error("content manifest file size is invalid")
        _claim_sha256(item["sha256"], f"content manifest {role} hash")
        observed_files[role] = {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
    if observed_files != dict(expected_files):
        _claim_contract_error("external content manifest does not anchor exact evidence files")

    tier = _claim_exact_keys(
        manifest["workload_tier_authority"],
        _TIER_AUTHORITY_KEYS,
        "workload tier authority",
    )
    if tier["tier_id"] not in {"T-mid", "T-big"}:
        _claim_contract_error("workload tier is not admitted by claim schema")
    if tier["admission_status"] not in {"admitted", "unadmitted"}:
        _claim_contract_error("workload tier admission status is invalid")
    if tier["classification"] not in {"production", "fixture"}:
        _claim_contract_error("workload tier classification is invalid")
    for name in _TIER_AUTHORITY_KEYS - {
        "tier_id",
        "admission_status",
        "classification",
    }:
        _claim_sha256(tier[name], f"workload tier {name}")
    expected_tier_bindings = {
        "workload_execution_authority_model_seal_sha256": workload_authority[
            "model_seal_sha256"
        ],
        **{
            name: derived[name]
            for name in _EVALUATOR_BINDING_HASH_KEYS
        },
    }
    if any(tier[name] != value for name, value in expected_tier_bindings.items()):
        _claim_contract_error("workload tier authority does not bind exact runtime inputs")
    return tier


def _bound_workload_projection(
    authority: Mapping[str, Any],
    retained: _RetainedFile,
    *,
    file_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "bound",
        "file": deepcopy(
            dict(file_identity)
            if file_identity is not None
            else _claim_file_identity(retained)
        ),
        "model_seal_sha256": authority["model_seal_sha256"],
        "workload_id": authority["workload_id"],
        "matched_execution_id": authority["matched_execution_id"],
        "contract_mode": authority["contract_mode"],
    }


def _bound_accuracy_projection(
    evaluator: Mapping[str, Any],
    evaluator_retained: _RetainedFile,
    report: Mapping[str, Any],
    report_retained: _RetainedFile,
    *,
    evaluator_file_identity: Mapping[str, Any] | None = None,
    report_file_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "bound",
        "evaluator_authority_file": deepcopy(
            dict(evaluator_file_identity)
            if evaluator_file_identity is not None
            else _claim_file_identity(evaluator_retained)
        ),
        "evaluator_authority_id": evaluator["authority_id"],
        "evaluator_authority_sha256": evaluator["authority_sha256"],
        "report_file": deepcopy(
            dict(report_file_identity)
            if report_file_identity is not None
            else _claim_file_identity(report_retained)
        ),
        "report_id": report["identities"]["report_id"],
        "report_sha256": report["report_sha256"],
    }


def _unavailable_claim_components(reason: str) -> dict[str, dict[str, str]]:
    unavailable = {"status": "unavailable", "reason": reason}
    return {
        "source_authority": deepcopy(unavailable),
        "workload_execution_authority": deepcopy(unavailable),
        "content_manifest_anchor": deepcopy(unavailable),
        "accuracy_binding": deepcopy(unavailable),
    }


def _evaluate_bound_claim_evidence(
    *,
    runtime_path: Path,
    runtime: Mapping[str, Any],
    runtime_retained: _RetainedFile,
    workload_authority_path: str | Path,
    evaluator_authority_path: str | Path,
    accuracy_report_path: str | Path,
    content_manifest_path: str | Path,
    source_authority_path: str | Path,
    expected_content_manifest_sha256: str,
    expected_source_authority_sha256: str,
    fixture_mode: bool,
    source_file_identities: Mapping[str, Mapping[str, Any]] | None = None,
    result_artifact_path: Path | None = None,
    producer_path: Path | None = None,
    claim_schema_path: Path | None = None,
    binary_path: Path | None = None,
    workload_schema_path: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    derived = _runtime_derived_binding(runtime)
    result = _load_claim_result(
        runtime_path,
        runtime,
        artifact_path_override=result_artifact_path,
    )
    if result["result_sha256"] != runtime["sim_result_binding"]["result_sha256"]:
        _claim_contract_error("retained final result seal differs from runtime binding")

    workload, workload_retained = _load_claim_workload_authority(
        workload_authority_path,
        fixture_mode=fixture_mode,
        schema_path=(
            workload_schema_path or WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_PATH
        ),
    )
    if not fixture_mode:
        _require_registered_claim_hash(
            PRODUCTION_RUNTIME_WORKLOAD_AUTHORITIES,
            workload_retained.sha256,
            "workload execution authority",
        )
    evaluator_retained: _RetainedFile | None = None
    report_retained: _RetainedFile | None = None
    content_retained: _RetainedFile | None = None
    source_retained: _RetainedFile | None = None
    producer_retained: _RetainedFile | None = None
    claim_schema_retained: _RetainedFile | None = None
    binary_retained: _RetainedFile | None = None
    try:
        _require_workload_runtime_binding(workload, runtime, derived)
        source, source_retained, _ = _load_claim_json_file(
            source_authority_path,
            role="runtime claim source authority",
            maximum_bytes=MAX_CLAIM_AUTHORITY_BYTES,
        )
        producer_retained = _open_retained_file(
            producer_path or ROOT / _CLAIM_PRODUCER_RELATIVE,
            role="runtime claim producer",
            maximum_bytes=MAX_RUNTIME_RECORD_BYTES,
        )
        claim_schema_retained = _open_retained_file(
            claim_schema_path or ROOT / _CLAIM_SCHEMA_RELATIVE,
            role="runtime claim schema",
            maximum_bytes=MAX_RUNTIME_CLAIM_BYTES,
        )
        binary_retained = _open_retained_file(
            binary_path or Path(runtime["invocation"]["binary"]["path"]),
            role="runtime release executable",
            maximum_bytes=MAX_BINARY_BYTES,
            require_executable=(binary_path is None),
            require_single_link=False,
        )
        producer_files = {
            _CLAIM_PRODUCER_RELATIVE.as_posix(): {
                "size_bytes": producer_retained.size,
                "sha256": producer_retained.sha256,
            },
            _CLAIM_SCHEMA_RELATIVE.as_posix(): {
                "size_bytes": claim_schema_retained.size,
                "sha256": claim_schema_retained.sha256,
            },
        }
        _validate_claim_source_authority(
            source,
            observed_file_sha256=source_retained.sha256,
            expected_file_sha256=expected_source_authority_sha256,
            runtime=runtime,
            producer_files=producer_files,
            binary_identity={
                "size_bytes": binary_retained.size,
                "sha256": binary_retained.sha256,
            },
            fixture_mode=fixture_mode,
        )
        if not fixture_mode:
            _require_registered_claim_hash(
                PRODUCTION_RUNTIME_SOURCE_AUTHORITIES,
                source_retained.sha256,
                "source authority",
            )
        evaluator, evaluator_retained, _ = _load_claim_json_file(
            evaluator_authority_path,
            role="detached evaluator authority",
            maximum_bytes=MAX_CLAIM_AUTHORITY_BYTES,
        )
        report, report_retained, _ = _load_claim_json_file(
            accuracy_report_path,
            role="accuracy report",
            maximum_bytes=MAX_ACCURACY_REPORT_BYTES,
        )
        if not fixture_mode:
            _require_registered_claim_hash(
                PRODUCTION_RUNTIME_EVALUATOR_AUTHORITIES,
                evaluator_retained.sha256,
                "detached evaluator authority",
            )
        _validate_detached_evaluator_authority(
            evaluator,
            report,
            report_file_sha256=report_retained.sha256,
            workload_authority=workload,
            runtime=runtime,
            derived=derived,
            fixture_mode=fixture_mode,
        )
        _validate_detached_evaluator_report_binding(
            evaluator,
            report,
            result,
            report_file_sha256=report_retained.sha256,
            runtime=runtime,
            derived=derived,
        )
        content, content_retained, _ = _load_claim_json_file(
            content_manifest_path,
            role="runtime claim content manifest",
            maximum_bytes=MAX_CONTENT_MANIFEST_BYTES,
        )
        if not fixture_mode:
            _require_registered_claim_hash(
                PRODUCTION_RUNTIME_CONTENT_MANIFEST_PINS,
                content_retained.sha256,
                "runtime claim content manifest",
            )
        file_identities = {
            "accuracy_report": _claim_file_identity(report_retained),
            "evaluator_authority": _claim_file_identity(evaluator_retained),
            "runtime_record": _claim_file_identity(runtime_retained),
            "source_authority": _claim_file_identity(source_retained),
            "workload_execution_authority": _claim_file_identity(workload_retained),
        }
        if source_file_identities is not None:
            expected_source_roles = set(_CLAIM_FILE_ROLES) | {"content_manifest"}
            if set(source_file_identities) != expected_source_roles:
                _claim_contract_error("detached source file identity closure differs")
            observed_with_content = {
                **file_identities,
                "content_manifest": _claim_file_identity(content_retained),
            }
            for role, retained_identity in observed_with_content.items():
                source_identity = source_file_identities[role]
                if (
                    source_identity["size_bytes"] != retained_identity["size_bytes"]
                    or source_identity["sha256"] != retained_identity["sha256"]
                ):
                    _claim_contract_error(
                        f"retained {role} bytes differ from source identity"
                    )
            file_identities = {
                role: deepcopy(dict(source_file_identities[role]))
                for role in _CLAIM_FILE_ROLES
            }
        tier = _validate_claim_content_manifest(
            content,
            expected_files=file_identities,
            expected_external_sha256=expected_content_manifest_sha256,
            observed_external_sha256=content_retained.sha256,
            workload_authority=workload,
            derived=derived,
        )
        components = {
            "source_authority": _source_authority_projection(
                source,
                source_retained,
                file_identity=file_identities["source_authority"],
            ),
            "workload_execution_authority": _bound_workload_projection(
                workload,
                workload_retained,
                file_identity=file_identities["workload_execution_authority"],
            ),
            "content_manifest_anchor": {
                "status": "bound",
                "file": deepcopy(
                    dict(
                        source_file_identities.get(
                            "content_manifest",
                            _claim_file_identity(content_retained),
                        )
                    )
                )
                if source_file_identities is not None
                else _claim_file_identity(content_retained),
                "manifest_id": content["manifest_id"],
                "manifest_sha256": content["manifest_sha256"],
                "external_file_sha256": expected_content_manifest_sha256,
                "workload_tier_authority": deepcopy(tier),
            },
            "result_binding": {
                "status": "bound",
                "file_sha256": runtime["sim_result_binding"]["file_sha256"],
                "result_sha256": runtime["sim_result_binding"]["result_sha256"],
            },
            "accuracy_binding": _bound_accuracy_projection(
                evaluator,
                evaluator_retained,
                report,
                report_retained,
                evaluator_file_identity=file_identities["evaluator_authority"],
                report_file_identity=file_identities["accuracy_report"],
            ),
            "derived_binding": derived,
        }
        reason = (
            "WORKLOAD_TIER_FIXTURE"
            if tier["classification"] == "fixture"
            else "WORKLOAD_TIER_UNADMITTED"
            if tier["admission_status"] != "admitted"
            else None
        )
        for retained, role in (
            (workload_retained, "workload execution authority"),
            (evaluator_retained, "detached evaluator authority"),
            (report_retained, "accuracy report"),
            (content_retained, "runtime claim content manifest"),
            (source_retained, "runtime claim source authority"),
            (producer_retained, "runtime claim producer"),
            (claim_schema_retained, "runtime claim schema"),
            (binary_retained, "runtime release executable"),
        ):
            retained.revalidate(role=role)
        return components, reason
    finally:
        workload_retained.close()
        if evaluator_retained is not None:
            evaluator_retained.close()
        if report_retained is not None:
            report_retained.close()
        if content_retained is not None:
            content_retained.close()
        if source_retained is not None:
            source_retained.close()
        if producer_retained is not None:
            producer_retained.close()
        if claim_schema_retained is not None:
            claim_schema_retained.close()
        if binary_retained is not None:
            binary_retained.close()


def _runtime_record_claim_projection(
    runtime: Mapping[str, Any], retained: _RetainedFile
) -> dict[str, Any]:
    return {
        "status": "bound",
        "file": _claim_file_identity(retained),
        "record_sha256": runtime["record_sha256"],
        "outcome_status": runtime["outcome"]["status"],
    }


def _missing_claim_authority_reason(
    *,
    workload_execution_authority_path: str | Path | None,
    evaluator_authority_path: str | Path | None,
    accuracy_report_path: str | Path | None,
    content_manifest_path: str | Path | None,
    source_authority_path: str | Path | None,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
) -> str | None:
    if workload_execution_authority_path is None:
        return "WORKLOAD_EXECUTION_AUTHORITY_UNAVAILABLE"
    if source_authority_path is None or expected_source_authority_sha256 is None:
        return "SOURCE_AUTHORITY_UNAVAILABLE"
    if evaluator_authority_path is None:
        return "EVALUATOR_AUTHORITY_UNAVAILABLE"
    if accuracy_report_path is None:
        return "ACCURACY_REPORT_UNAVAILABLE"
    if content_manifest_path is None or expected_content_manifest_sha256 is None:
        return "CONTENT_MANIFEST_AUTHORITY_UNAVAILABLE"
    return None


def _claimability_model(
    reason: str | None, *, fixture_mode: bool
) -> dict[str, str]:
    if reason is not None:
        return {"status": "unclaimable", "reason": reason}
    if fixture_mode:
        return {"status": "unclaimable", "reason": _FIXTURE_CLAIM_REASON}
    return {"status": "claimable", "basis": _CLAIM_BASIS}


def _build_runtime_claim_document(
    *,
    runtime_path: Path,
    runtime: Mapping[str, Any],
    runtime_retained: _RetainedFile,
    workload_execution_authority_path: str | Path | None,
    evaluator_authority_path: str | Path | None,
    accuracy_report_path: str | Path | None,
    content_manifest_path: str | Path | None,
    source_authority_path: str | Path | None,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    fixture_mode: bool,
) -> dict[str, Any]:
    runtime_reason = _runtime_claimability_reason(runtime)
    if runtime_reason is not None:
        components: dict[str, Any] = {
            **_unavailable_claim_components("runtime_gate_unclaimable"),
            "result_binding": {
                "status": "unavailable",
                "reason": "runtime_gate_unclaimable",
            },
            "derived_binding": {
                "status": "unavailable",
                "reason": "runtime_gate_unclaimable",
            },
        }
        reason = runtime_reason
    else:
        result = _load_claim_result(runtime_path, runtime)
        derived = _runtime_derived_binding(runtime)
        if result["result_sha256"] != runtime["sim_result_binding"]["result_sha256"]:
            _claim_contract_error("final result seal differs from runtime record")
        missing = _missing_claim_authority_reason(
            workload_execution_authority_path=workload_execution_authority_path,
            evaluator_authority_path=evaluator_authority_path,
            accuracy_report_path=accuracy_report_path,
            content_manifest_path=content_manifest_path,
            source_authority_path=source_authority_path,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
        )
        if missing is not None:
            components = {
                **_unavailable_claim_components(missing),
                "result_binding": {
                    "status": "bound",
                    "file_sha256": runtime["sim_result_binding"]["file_sha256"],
                    "result_sha256": runtime["sim_result_binding"]["result_sha256"],
                },
                "derived_binding": derived,
            }
            reason = missing
        else:
            assert workload_execution_authority_path is not None
            assert evaluator_authority_path is not None
            assert accuracy_report_path is not None
            assert content_manifest_path is not None
            assert source_authority_path is not None
            assert expected_content_manifest_sha256 is not None
            assert expected_source_authority_sha256 is not None
            components, reason = _evaluate_bound_claim_evidence(
                runtime_path=runtime_path,
                runtime=runtime,
                runtime_retained=runtime_retained,
                workload_authority_path=workload_execution_authority_path,
                evaluator_authority_path=evaluator_authority_path,
                accuracy_report_path=accuracy_report_path,
                content_manifest_path=content_manifest_path,
                source_authority_path=source_authority_path,
                expected_content_manifest_sha256=expected_content_manifest_sha256,
                expected_source_authority_sha256=expected_source_authority_sha256,
                fixture_mode=fixture_mode,
            )
    document: dict[str, Any] = {
        "schema_version": "runtime_workload_tier_claim_authority.v1",
        "schema_sha256": RUNTIME_CLAIM_SCHEMA_SHA256,
        "claim_sha256": "0" * 64,
        "created_at_utc": _created_at_utc(),
        "authority_role": "detached_post_result_runtime_claim_authority",
        "contract_mode": (
            "FIXTURE_ONLY_NON_AUTHORITATIVE"
            if fixture_mode
            else "TARGET_FREE_PRODUCTION"
        ),
        "runtime_record_binding": _runtime_record_claim_projection(
            runtime, runtime_retained
        ),
        **components,
        "claimability": _claimability_model(reason, fixture_mode=fixture_mode),
    }
    document["claim_sha256"] = canonical_sha256(document, "claim_sha256")
    return document


def _claim_generation_projection(generation: Any) -> dict[str, Any]:
    return {
        "path": str(generation.path),
        "content_manifest_sha256": generation.content_manifest_sha256,
        "filesystem_generation_sha256": generation.filesystem_generation_sha256,
        "file_count": generation.file_count,
        "directory_count": generation.directory_count,
        "total_file_bytes": generation.total_file_bytes,
        "files": [
            {
                "path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in generation.files
        ],
    }


def _publish_claim_evidence_generation(
    output_path: str | Path,
    *,
    runtime: Mapping[str, Any],
    runtime_record_path: Path,
    workload_execution_authority_path: str | Path | None,
    evaluator_authority_path: str | Path | None,
    accuracy_report_path: str | Path | None,
    content_manifest_path: str | Path | None,
    source_authority_path: str | Path | None,
) -> dict[str, Any]:
    output = Path(os.path.abspath(os.fspath(output_path)))
    files = [
        ImmutableFile.from_path(
            _CLAIM_EVIDENCE_FIXED_NAMES["runtime_record"], runtime_record_path
        ),
        ImmutableFile.from_path(
            _CLAIM_EVIDENCE_FIXED_NAMES["runtime_binary"],
            Path(runtime["invocation"]["binary"]["path"]),
            allow_hardlinks=True,
        ),
        ImmutableFile.from_path(
            _CLAIM_EVIDENCE_FIXED_NAMES["runtime_manifest"],
            Path(runtime["manifest_identity"]["file"]["path"]),
        ),
        ImmutableFile.from_path(
            _CLAIM_EVIDENCE_FIXED_NAMES["claim_producer"],
            ROOT / _CLAIM_PRODUCER_RELATIVE,
        ),
        ImmutableFile.from_path(
            _CLAIM_EVIDENCE_FIXED_NAMES["claim_schema"],
            ROOT / _CLAIM_SCHEMA_RELATIVE,
        ),
        ImmutableFile.from_path(
            _CLAIM_EVIDENCE_FIXED_NAMES["workload_schema"],
            WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_PATH,
        ),
    ]
    for index, sample in enumerate(runtime["samples"]):
        for stream_name in ("stdout", "stderr"):
            artifact = sample[stream_name]
            source = runtime_record_path.parent / artifact["path"]
            files.append(
                ImmutableFile.from_path(
                    f"samples/{index:04d}.{stream_name}",
                    source,
                )
            )
    optional_paths = {
        "workload_execution_authority": workload_execution_authority_path,
        "evaluator_authority": evaluator_authority_path,
        "accuracy_report": accuracy_report_path,
        "content_manifest": content_manifest_path,
        "source_authority": source_authority_path,
    }
    for role, source in optional_paths.items():
        if source is not None:
            files.append(
                ImmutableFile.from_path(
                    _CLAIM_EVIDENCE_FIXED_NAMES[role], Path(source)
                )
            )
    try:
        generation_path = publish_directory(
            output.parent,
            output.name + _CLAIM_EVIDENCE_SUFFIX,
            files,
            accept_existing=False,
        )
        generation = inspect_immutable_directory_generation(
            generation_path,
            max_files=64,
            max_directories=2,
            max_total_file_bytes=(
                MAX_BINARY_BYTES
                + MAX_MANIFEST_BYTES
                + MAX_RUNTIME_RECORD_BYTES
                + 24 * MAX_RESULT_STREAM_BYTES
                + 4 * MAX_CLAIM_AUTHORITY_BYTES
                + MAX_ACCURACY_REPORT_BYTES
                + MAX_CONTENT_MANIFEST_BYTES
            ),
            max_total_relative_path_bytes=8192,
        )
    except ImmutableArtifactError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_EVIDENCE_PUBLICATION_FAILED",
            "runtime claim retained evidence generation could not be published",
        ) from exc
    return _claim_generation_projection(generation)


def _publish_runtime_claim(path: str | Path, document: Mapping[str, Any]) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.name in {"", ".", ".."}:
        _claim_contract_error("runtime claim output name is invalid")
    parent_fd = _open_directory_no_symlinks(
        absolute.parent,
        role="runtime claim output parent",
        error_code="RUNTIME_OUTPUT_PARENT_INVALID",
    )
    try:
        if _entry_exists(parent_fd, absolute.name):
            raise RuntimeRecorderError(
                "RUNTIME_OUTPUT_EXISTS",
                "runtime claim output already exists",
                context={"path": str(absolute)},
            )
        _publish_exclusive_at(parent_fd, absolute.name, _canonical_json_line(document))
    finally:
        os.close(parent_fd)
    return absolute


def _require_production_runtime_claim_authorities() -> None:
    registries = (
        PRODUCTION_RUNTIME_WORKLOAD_AUTHORITIES,
        PRODUCTION_RUNTIME_EVALUATOR_AUTHORITIES,
        PRODUCTION_RUNTIME_SOURCE_AUTHORITIES,
        PRODUCTION_RUNTIME_CONTENT_MANIFEST_PINS,
    )
    if not all(registries):
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_AUTHORITY_UNAVAILABLE",
            "production runtime claim authorities are not independently installed",
        )


def _registry_admits_sha256(
    registry: Mapping[str, Any], observed_sha256: str
) -> bool:
    for key, value in registry.items():
        candidates = (key, value)
        if any(candidate == observed_sha256 for candidate in candidates):
            return True
        if isinstance(value, Mapping) and observed_sha256 in value.values():
            return True
    return False


def _require_registered_claim_hash(
    registry: Mapping[str, Any], observed_sha256: str, role: str
) -> None:
    if not _registry_admits_sha256(registry, observed_sha256):
        _claim_contract_error(
            f"{role} is absent from the immutable production registry",
            code="RUNTIME_CLAIM_AUTHORITY_UNAVAILABLE",
        )


def _claim_runtime_impl(
    runtime_record_path: str | Path,
    output_path: str | Path,
    *,
    workload_execution_authority_path: str | Path | None,
    evaluator_authority_path: str | Path | None,
    accuracy_report_path: str | Path | None,
    content_manifest_path: str | Path | None,
    source_authority_path: str | Path | None,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    repo_root: Path,
    fixture_mode: bool,
) -> dict[str, Any]:
    runtime, runtime_retained = _load_runtime_claim_evidence(
        runtime_record_path, repo_root=repo_root
    )
    try:
        document = _build_runtime_claim_document(
            runtime_path=runtime_retained.path,
            runtime=runtime,
            runtime_retained=runtime_retained,
            workload_execution_authority_path=workload_execution_authority_path,
            evaluator_authority_path=evaluator_authority_path,
            accuracy_report_path=accuracy_report_path,
            content_manifest_path=content_manifest_path,
            source_authority_path=source_authority_path,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
            fixture_mode=fixture_mode,
        )
        document["retained_evidence_generation"] = (
            _publish_claim_evidence_generation(
                output_path,
                runtime=runtime,
                runtime_record_path=runtime_retained.path,
                workload_execution_authority_path=(
                    workload_execution_authority_path
                ),
                evaluator_authority_path=evaluator_authority_path,
                accuracy_report_path=accuracy_report_path,
                content_manifest_path=content_manifest_path,
                source_authority_path=source_authority_path,
            )
        )
        document["claim_sha256"] = canonical_sha256(document, "claim_sha256")
        validate_runtime_claim(
            document,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
            repo_root=repo_root,
        )
        published = _publish_runtime_claim(output_path, document)
    finally:
        runtime_retained.close()
    loaded = _load_runtime_claim(
        published,
        expected_content_manifest_sha256=expected_content_manifest_sha256,
        expected_source_authority_sha256=expected_source_authority_sha256,
        repo_root=repo_root,
    )
    if loaded != document:
        _claim_contract_error("published runtime claim did not replay byte-exactly")
    return document


def _translate_claim_input_failures(
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return operation()
    except RuntimeRecorderError:
        raise
    except (
        AttributeError,
        ContractValidationError,
        ImmutableArtifactError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_INVALID",
            "runtime claim input is malformed or lacks typed authority",
        ) from exc


def _claim_runtime(
    runtime_record_path: str | Path,
    output_path: str | Path,
    *,
    workload_execution_authority_path: str | Path | None,
    evaluator_authority_path: str | Path | None,
    accuracy_report_path: str | Path | None,
    content_manifest_path: str | Path | None,
    source_authority_path: str | Path | None,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    _require_production_runtime_claim_authorities()
    return _translate_claim_input_failures(
        lambda: _claim_runtime_impl(
            runtime_record_path,
            output_path,
            workload_execution_authority_path=workload_execution_authority_path,
            evaluator_authority_path=evaluator_authority_path,
            accuracy_report_path=accuracy_report_path,
            content_manifest_path=content_manifest_path,
            source_authority_path=source_authority_path,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
            repo_root=repo_root,
            fixture_mode=False,
        )
    )


def _claim_runtime_fixture(
    runtime_record_path: str | Path,
    output_path: str | Path,
    *,
    workload_execution_authority_path: str | Path | None,
    evaluator_authority_path: str | Path | None,
    accuracy_report_path: str | Path | None,
    content_manifest_path: str | Path | None,
    source_authority_path: str | Path | None,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    """Exercise claim algebra without creating production authority."""

    return _translate_claim_input_failures(
        lambda: _claim_runtime_impl(
            runtime_record_path,
            output_path,
            workload_execution_authority_path=workload_execution_authority_path,
            evaluator_authority_path=evaluator_authority_path,
            accuracy_report_path=accuracy_report_path,
            content_manifest_path=content_manifest_path,
            source_authority_path=source_authority_path,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
            repo_root=repo_root,
            fixture_mode=True,
        )
    )


def claim_runtime(
    runtime_record_path: str | Path,
    output_path: str | Path,
    *,
    workload_execution_authority_path: str | Path | None = None,
    evaluator_authority_path: str | Path | None = None,
    accuracy_report_path: str | Path | None = None,
    content_manifest_path: str | Path | None = None,
    source_authority_path: str | Path | None = None,
    content_manifest_sha256: str | None = None,
    source_authority_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish a detached post-result runtime claim assessment.

    Accuracy and label-derived evidence is opened only here, after the runtime
    record and every simulator result have already been sealed.
    """

    return _claim_runtime(
        runtime_record_path,
        output_path,
        workload_execution_authority_path=workload_execution_authority_path,
        evaluator_authority_path=evaluator_authority_path,
        accuracy_report_path=accuracy_report_path,
        content_manifest_path=content_manifest_path,
        source_authority_path=source_authority_path,
        expected_content_manifest_sha256=content_manifest_sha256,
        expected_source_authority_sha256=source_authority_sha256,
        repo_root=ROOT,
    )


_MISSING_CLAIM_REASONS = {
    "WORKLOAD_EXECUTION_AUTHORITY_UNAVAILABLE",
    "SOURCE_AUTHORITY_UNAVAILABLE",
    "EVALUATOR_AUTHORITY_UNAVAILABLE",
    "ACCURACY_REPORT_UNAVAILABLE",
    "CONTENT_MANIFEST_AUTHORITY_UNAVAILABLE",
}


def _require_claim_components(
    document: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    reason: str | None,
) -> None:
    for name, value in expected.items():
        if document[name] != value:
            _claim_contract_error(
                f"runtime claim {name} is not derived from retained evidence"
            )
    if document["claimability"] != _claimability_model(
        reason,
        fixture_mode=(document["contract_mode"] == "FIXTURE_ONLY_NON_AUTHORITATIVE"),
    ):
        _claim_contract_error(
            "runtime claimability is not derived from retained evidence"
        )


def _validate_runtime_claim_envelope(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        _claim_contract_error("runtime claim root must be an object")
    try:
        _RUNTIME_CLAIM_VALIDATOR.validate(document)
    except ValidationError as exc:
        _claim_contract_error(
            exc.message,
            context={
                "json_path": _schema_path(exc),
                "validator": str(exc.validator),
            },
        )
    if document["schema_sha256"] != RUNTIME_CLAIM_SCHEMA_SHA256:
        _claim_contract_error("runtime claim schema identity is not code-pinned")
    if document["contract_mode"] == "TARGET_FREE_PRODUCTION":
        _require_production_runtime_claim_authorities()
    if document["claim_sha256"] != canonical_sha256(document, "claim_sha256"):
        _claim_contract_error("runtime claim canonical seal does not match")
    try:
        created = datetime.fromisoformat(
            document["created_at_utc"].replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_INVALID",
            "runtime claim creation time is invalid",
        ) from exc
    if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
        _claim_contract_error("runtime claim creation time is not UTC")
    return document


def _expected_early_claim_components() -> dict[str, Any]:
    return {
        **_unavailable_claim_components("runtime_gate_unclaimable"),
        "result_binding": {
            "status": "unavailable",
            "reason": "runtime_gate_unclaimable",
        },
        "derived_binding": {
            "status": "unavailable",
            "reason": "runtime_gate_unclaimable",
        },
    }


def _expected_final_result_components(
    runtime: Mapping[str, Any], derived: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "result_binding": {
            "status": "bound",
            "file_sha256": runtime["sim_result_binding"]["file_sha256"],
            "result_sha256": runtime["sim_result_binding"]["result_sha256"],
        },
        "derived_binding": derived,
    }


def _validate_unbound_final_claim(
    document: Mapping[str, Any],
    runtime: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> None:
    evidence = (
        document["source_authority"],
        document["workload_execution_authority"],
        document["content_manifest_anchor"],
        document["accuracy_binding"],
    )
    if any(item["status"] != "unavailable" for item in evidence):
        _claim_contract_error("runtime claim authority evidence is only partially bound")
    reasons = {item["reason"] for item in evidence}
    if len(reasons) != 1:
        _claim_contract_error("missing runtime claim authorities have inconsistent typing")
    reason = next(iter(reasons))
    if reason not in _MISSING_CLAIM_REASONS:
        _claim_contract_error("missing runtime claim authority reason is invalid")
    expected = {
        **_unavailable_claim_components(reason),
        **_expected_final_result_components(runtime, derived),
    }
    _require_claim_components(document, expected, reason=reason)


def _validate_bound_final_claim(
    document: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_retained: _RetainedFile,
    *,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    result_artifact_path: Path,
    source_file_identities: Mapping[str, Mapping[str, Any]],
    generation_path: Path,
) -> None:
    if expected_content_manifest_sha256 is None:
        _claim_contract_error("claimable runtime lacks an external content-manifest pin")
    if expected_source_authority_sha256 is None:
        _claim_contract_error("claimable runtime lacks an external clean-source pin")
    source = document["source_authority"]
    workload = document["workload_execution_authority"]
    content = document["content_manifest_anchor"]
    accuracy = document["accuracy_binding"]
    if any(
        item["status"] != "bound"
        for item in (source, workload, content, accuracy)
    ):
        _claim_contract_error("runtime claim authority evidence is only partially bound")
    if content["external_file_sha256"] != expected_content_manifest_sha256:
        _claim_contract_error("serialized content-manifest pin differs from caller authority")
    expected, reason = _evaluate_bound_claim_evidence(
        runtime_path=runtime_retained.path,
        runtime=runtime,
        runtime_retained=runtime_retained,
        workload_authority_path=(
            generation_path
            / _CLAIM_EVIDENCE_FIXED_NAMES["workload_execution_authority"]
        ),
        evaluator_authority_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["evaluator_authority"]
        ),
        accuracy_report_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["accuracy_report"]
        ),
        content_manifest_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["content_manifest"]
        ),
        source_authority_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["source_authority"]
        ),
        expected_content_manifest_sha256=expected_content_manifest_sha256,
        expected_source_authority_sha256=expected_source_authority_sha256,
        fixture_mode=(document["contract_mode"] == "FIXTURE_ONLY_NON_AUTHORITATIVE"),
        source_file_identities=source_file_identities,
        result_artifact_path=result_artifact_path,
        producer_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["claim_producer"]
        ),
        claim_schema_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["claim_schema"]
        ),
        binary_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["runtime_binary"]
        ),
        workload_schema_path=(
            generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["workload_schema"]
        ),
    )
    _require_claim_components(document, expected, reason=reason)


def _validate_final_runtime_claim(
    document: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_retained: _RetainedFile,
    *,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    result_artifact_path: Path,
    source_file_identities: Mapping[str, Mapping[str, Any]],
    generation_path: Path,
) -> None:
    result = _load_claim_result(
        runtime_retained.path,
        runtime,
        artifact_path_override=result_artifact_path,
    )
    if result["result_sha256"] != runtime["sim_result_binding"]["result_sha256"]:
        _claim_contract_error("runtime claim result seal differs from retained result")
    derived = _runtime_derived_binding(runtime)
    statuses = {
        document["source_authority"]["status"],
        document["workload_execution_authority"]["status"],
        document["content_manifest_anchor"]["status"],
        document["accuracy_binding"]["status"],
    }
    if statuses == {"unavailable"}:
        _validate_unbound_final_claim(document, runtime, derived)
    elif statuses == {"bound"}:
        _validate_bound_final_claim(
            document,
            runtime,
            runtime_retained,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
            result_artifact_path=result_artifact_path,
            source_file_identities=source_file_identities,
            generation_path=generation_path,
        )
    else:
        _claim_contract_error("runtime claim authority evidence is only partially bound")


def _inspect_claim_evidence_generation(
    projection: Mapping[str, Any],
) -> tuple[Path, dict[str, Mapping[str, Any]]]:
    if type(projection) is not dict:
        _claim_contract_error("retained evidence generation is malformed")
    path = projection.get("path")
    if type(path) is not str or path != os.path.abspath(path):
        _claim_contract_error("retained evidence generation path is invalid")
    try:
        generation = inspect_immutable_directory_generation(
            Path(path),
            max_files=64,
            max_directories=2,
            max_total_file_bytes=(
                MAX_BINARY_BYTES
                + MAX_MANIFEST_BYTES
                + MAX_RUNTIME_RECORD_BYTES
                + 24 * MAX_RESULT_STREAM_BYTES
                + 4 * MAX_CLAIM_AUTHORITY_BYTES
                + MAX_ACCURACY_REPORT_BYTES
                + MAX_CONTENT_MANIFEST_BYTES
            ),
            max_total_relative_path_bytes=8192,
        )
    except ImmutableArtifactError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_EVIDENCE_INVALID",
            "retained runtime claim evidence generation is unavailable",
        ) from exc
    observed = _claim_generation_projection(generation)
    if observed != projection:
        _claim_contract_error("retained evidence generation identity differs")
    return generation.path, {
        item["path"]: item for item in projection["files"]
    }


def _read_claim_generation_file(
    generation_path: Path,
    files: Mapping[str, Mapping[str, Any]],
    relative_path: str,
    *,
    maximum_bytes: int,
) -> bytes:
    identity = files.get(relative_path)
    if type(identity) is not dict:
        _claim_contract_error(f"retained evidence lacks {relative_path}")
    try:
        with open_retained_regular(
            generation_path / relative_path,
            max_bytes=maximum_bytes,
        ) as retained:
            payload = retained.read_bytes(max_bytes=maximum_bytes)
            observed = {
                "path": relative_path,
                "size_bytes": retained.size_bytes,
                "sha256": retained.sha256,
            }
            if observed != identity:
                _claim_contract_error(
                    f"retained evidence identity differs for {relative_path}"
                )
            retained.verify_unchanged()
            return payload
    except RuntimeRecorderError:
        raise
    except ImmutableArtifactError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_EVIDENCE_INVALID",
            f"retained evidence file is unavailable: {relative_path}",
        ) from exc


def _validate_detached_runtime_result(
    result: dict[str, Any],
    *,
    result_payload: bytes,
    runtime: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_canonical_sha256: str,
    binary_sha256: str,
) -> None:
    try:
        validate_result(result)
    except ContractValidationError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_EVIDENCE_INVALID",
            "retained simulator result contract is invalid",
        ) from exc
    if result_payload != _canonical_json_line(result):
        _claim_contract_error("retained simulator result is not canonical JSON")
    if result["build_identity"]["binary_sha256"] != binary_sha256:
        _claim_contract_error("retained simulator result names a different build")
    configuration = result["configuration_identity"]
    if (
        configuration["run_manifest_sha256"] != manifest_canonical_sha256
        or configuration["resolved_configuration_sha256"]
        != _sha256_canonical(result["resolved_configuration"])
        or result["resolved_configuration"]["replay_mode"] != manifest["replay_mode"]
    ):
        _claim_contract_error("retained simulator result configuration differs")
    expected_launches, _ = _launch_projection(manifest)
    observed_launches = [
        deepcopy(item["launch_key"]) for item in result["launches"]
    ]
    observed_launches.sort(key=canonical_launch_key_token)
    if observed_launches != expected_launches:
        _claim_contract_error("retained simulator result launch set differs")
    file_sha256 = hashlib.sha256(result_payload).hexdigest()
    expected_binding = _result_binding(result, file_sha256, "result_identity_mismatch")
    if runtime["sim_result_binding"] != expected_binding:
        _claim_contract_error("retained simulator result projection differs")


def _load_detached_runtime_claim_evidence(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Mapping[str, Any]]]:
    generation_path, files = _inspect_claim_evidence_generation(
        document["retained_evidence_generation"]
    )
    runtime_payload = _read_claim_generation_file(
        generation_path,
        files,
        _CLAIM_EVIDENCE_FIXED_NAMES["runtime_record"],
        maximum_bytes=MAX_RUNTIME_RECORD_BYTES,
    )
    runtime = _decode_claim_object(
        runtime_payload,
        source=str(generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["runtime_record"]),
        role="retained runtime record",
    )
    validate_runtime_record(
        runtime,
        permit_terminal_input_drift=_is_input_drift_record(runtime),
    )
    if runtime_payload != _canonical_json_line(runtime):
        _claim_contract_error("retained runtime record is not canonical JSON")
    runtime_identity = document["runtime_record_binding"]["file"]
    if (
        len(runtime_payload) != runtime_identity["size_bytes"]
        or hashlib.sha256(runtime_payload).hexdigest() != runtime_identity["sha256"]
    ):
        _claim_contract_error("retained runtime record bytes differ from source identity")

    binary_payload = _read_claim_generation_file(
        generation_path,
        files,
        _CLAIM_EVIDENCE_FIXED_NAMES["runtime_binary"],
        maximum_bytes=MAX_BINARY_BYTES,
    )
    binary_identity = runtime["invocation"]["binary"]
    if (
        len(binary_payload) != binary_identity["size_bytes"]
        or hashlib.sha256(binary_payload).hexdigest() != binary_identity["sha256"]
    ):
        _claim_contract_error("retained runtime executable identity differs")
    manifest_payload = _read_claim_generation_file(
        generation_path,
        files,
        _CLAIM_EVIDENCE_FIXED_NAMES["runtime_manifest"],
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _decode_claim_object(
        manifest_payload,
        source=str(generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["runtime_manifest"]),
        role="retained run manifest",
    )
    try:
        validate_manifest(manifest)
    except ContractValidationError as exc:
        raise RuntimeRecorderError(
            "RUNTIME_CLAIM_EVIDENCE_INVALID",
            "retained run manifest is invalid",
        ) from exc
    manifest_identity = runtime["manifest_identity"]
    if (
        len(manifest_payload) != manifest_identity["file"]["size_bytes"]
        or hashlib.sha256(manifest_payload).hexdigest()
        != manifest_identity["file"]["sha256"]
        or _sha256_canonical(manifest) != manifest_identity["canonical_sha256"]
        or manifest["scope"] != manifest_identity["scope"]
    ):
        _claim_contract_error("retained run manifest identity differs")
    launch_keys, launch_set_sha256 = _launch_projection(manifest)
    if (
        launch_keys != runtime["workload_identity"]["launch_keys"]
        or launch_set_sha256 != runtime["workload_identity"]["launch_set_sha256"]
    ):
        _claim_contract_error("retained manifest launch identity differs")

    parsed_bound_result = False
    for index, sample in enumerate(runtime["samples"]):
        for stream_name in ("stdout", "stderr"):
            relative = f"samples/{index:04d}.{stream_name}"
            payload = _read_claim_generation_file(
                generation_path,
                files,
                relative,
                maximum_bytes=MAX_RESULT_STREAM_BYTES,
            )
            identity = sample[stream_name]
            if (
                len(payload) != identity["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != identity["sha256"]
            ):
                _claim_contract_error("retained runtime sample identity differs")
            if (
                stream_name == "stdout"
                and not parsed_bound_result
                and runtime["sim_result_binding"]["status"] == "bound"
                and identity["sha256"]
                == runtime["sim_result_binding"]["file_sha256"]
            ):
                result = _decode_claim_object(
                    payload,
                    source=str(generation_path / relative),
                    role="retained simulator result",
                )
                _validate_detached_runtime_result(
                    result,
                    result_payload=payload,
                    runtime=runtime,
                    manifest=manifest,
                    manifest_canonical_sha256=manifest_identity["canonical_sha256"],
                    binary_sha256=binary_identity["sha256"],
                )
                parsed_bound_result = True
    if runtime["sim_result_binding"]["status"] == "bound" and not parsed_bound_result:
        _claim_contract_error("retained runtime result artifact is unavailable")
    refreshed = inspect_immutable_directory_generation(
        generation_path,
        max_files=64,
        max_directories=2,
        max_total_file_bytes=document["retained_evidence_generation"][
            "total_file_bytes"
        ],
        max_total_relative_path_bytes=8192,
    )
    if _claim_generation_projection(refreshed) != document["retained_evidence_generation"]:
        _claim_contract_error("retained evidence generation changed during validation")
    return runtime, generation_path, files


def validate_runtime_claim(
    document: Any,
    *,
    expected_content_manifest_sha256: str | None = None,
    expected_source_authority_sha256: str | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Recompute claimability from the current checkout and retained files."""

    document = _validate_runtime_claim_envelope(document)
    del repo_root
    runtime_binding = document["runtime_record_binding"]
    runtime, generation_path, _files = _load_detached_runtime_claim_evidence(
        document
    )
    runtime_retained = _open_retained_file(
        generation_path / _CLAIM_EVIDENCE_FIXED_NAMES["runtime_record"],
        role="retained runtime record",
        maximum_bytes=MAX_RUNTIME_RECORD_BYTES,
    )
    try:
        expected_runtime = _runtime_record_claim_projection(runtime, runtime_retained)
        expected_runtime["file"] = deepcopy(runtime_binding["file"])
        if runtime_binding != expected_runtime:
            _claim_contract_error(
                "runtime claim record binding differs from retained bytes"
            )
        runtime_reason = _runtime_claimability_reason(runtime)
        if runtime_reason is not None:
            _require_claim_components(
                document,
                _expected_early_claim_components(),
                reason=runtime_reason,
            )
        else:
            authority_statuses = {
                document["source_authority"]["status"],
                document["workload_execution_authority"]["status"],
                document["content_manifest_anchor"]["status"],
                document["accuracy_binding"]["status"],
            }
            source_file_identities = (
                {
                    "accuracy_report": document["accuracy_binding"]["report_file"],
                    "evaluator_authority": document["accuracy_binding"][
                        "evaluator_authority_file"
                    ],
                    "runtime_record": runtime_binding["file"],
                    "source_authority": document["source_authority"]["file"],
                    "workload_execution_authority": document[
                        "workload_execution_authority"
                    ]["file"],
                    "content_manifest": document["content_manifest_anchor"]["file"],
                }
                if authority_statuses == {"bound"}
                else {}
            )
            _validate_final_runtime_claim(
                document,
                runtime,
                runtime_retained,
                expected_content_manifest_sha256=expected_content_manifest_sha256,
                expected_source_authority_sha256=expected_source_authority_sha256,
                result_artifact_path=generation_path / "samples/0000.stdout",
                source_file_identities=source_file_identities,
                generation_path=generation_path,
            )
        runtime_retained.revalidate(role="retained runtime record")
        return document
    finally:
        runtime_retained.close()


def _load_runtime_claim(
    path: str | Path,
    *,
    expected_content_manifest_sha256: str | None,
    expected_source_authority_sha256: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    retained = _open_retained_file(
        path,
        role="runtime claim",
        maximum_bytes=MAX_RUNTIME_CLAIM_BYTES,
    )
    try:
        payload = retained.read_bytes(role="runtime claim")
        document = _decode_claim_object(
            payload,
            source=str(retained.path),
            role="runtime claim",
        )
        validate_runtime_claim(
            document,
            expected_content_manifest_sha256=expected_content_manifest_sha256,
            expected_source_authority_sha256=expected_source_authority_sha256,
            repo_root=repo_root,
        )
        if payload != _canonical_json_line(document):
            _claim_contract_error("runtime claim is not canonical JSON")
        retained.revalidate(role="runtime claim")
        return document
    finally:
        retained.close()


def load_runtime_claim(
    path: str | Path,
    *,
    content_manifest_sha256: str | None = None,
    source_authority_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a detached claim against independent content and source pins."""

    return _load_runtime_claim(
        path,
        expected_content_manifest_sha256=content_manifest_sha256,
        expected_source_authority_sha256=source_authority_sha256,
        repo_root=ROOT,
    )



def _error_line(error: RuntimeRecorderError) -> str:
    return json.dumps(
        {"code": error.code, "detail": error.detail},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _claim_cli(arguments: Sequence[str]) -> int:
    (
        _command,
        runtime_path,
        output_path,
        workload_path,
        evaluator_path,
        report_path,
        content_path,
        content_sha256,
        source_path,
        source_sha256,
    ) = arguments
    optional = lambda value: None if value == "-" else value
    try:
        claim = claim_runtime(
            runtime_path,
            output_path,
            workload_execution_authority_path=optional(workload_path),
            evaluator_authority_path=optional(evaluator_path),
            accuracy_report_path=optional(report_path),
            content_manifest_path=optional(content_path),
            source_authority_path=optional(source_path),
            content_manifest_sha256=optional(content_sha256),
            source_authority_sha256=optional(source_sha256),
        )
    except RuntimeRecorderError as exc:
        sys.stderr.write(_error_line(exc))
        return 1
    message = {
        "claim_path": str(Path(output_path).absolute()),
        "claim_sha256": claim["claim_sha256"],
        "status": claim["claimability"]["status"],
    }
    line = json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    if claim["claimability"]["status"] == "claimable":
        sys.stdout.write(line)
        return 0
    sys.stderr.write(line)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 10 and arguments[0] == "claim":
        return _claim_cli(arguments)
    if len(arguments) != 3 or arguments[0] != "record":
        error = RuntimeRecorderError(
            "CLI_USAGE_FAILED",
            "usage: python -m eval.bench record MANIFEST_V2 OUTPUT_RECORD",
        )
        sys.stderr.write(_error_line(error))
        return 2
    try:
        record = record_runtime(arguments[1], arguments[2])
    except RuntimeRecorderError as exc:
        sys.stderr.write(_error_line(exc))
        return 1
    status = record["outcome"]["status"]
    message = {
        "record_path": str(Path(arguments[2]).absolute()),
        "record_sha256": record["record_sha256"],
        "status": status,
    }
    line = json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    if status == "MEASURED":
        sys.stdout.write(line)
        return 0
    sys.stderr.write(line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
