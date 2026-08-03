#!/usr/bin/env python3
"""Small fail-closed primitives for immutable external artifact publication."""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import errno
import fcntl
import functools
import hashlib
import heapq
import json
import math
import os
import resource
import stat
import sys
import threading
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, NoReturn


_CHUNK = 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_MAX_COMPONENTS = 64
_MAX_RELATIVE_BYTES = 4096
_MAX_FINAL_NAME_BYTES = 246
_MAX_REGULAR_LEAF_BYTES = 243
_SNAPSHOT_CLOSURE_THRESHOLD = 512
_INOTIFY_Q_OVERFLOW = 0x00004000
_INOTIFY_ROOT_DIRECTORY_ATTRIB = 0x40000004  # IN_ISDIR | IN_ATTRIB
_INOTIFY_ROOT_DIRECTORY_MOVED_FROM = 0x40000040  # IN_ISDIR | IN_MOVED_FROM
_INOTIFY_ROOT_DIRECTORY_MOVED_TO = 0x40000080  # IN_ISDIR | IN_MOVED_TO
_INOTIFY_ROOT_MOVE_SELF = 0x00000800  # IN_MOVE_SELF
_INOTIFY_MUTATION_EVENTS = (
    0x00000002  # IN_MODIFY
    | 0x00000004  # IN_ATTRIB
    | 0x00000008  # IN_CLOSE_WRITE
    | 0x00000040  # IN_MOVED_FROM
    | 0x00000080  # IN_MOVED_TO
    | 0x00000100  # IN_CREATE
    | 0x00000200  # IN_DELETE
    | 0x00000400  # IN_DELETE_SELF
    | 0x00000800  # IN_MOVE_SELF
    | 0x00002000  # IN_UNMOUNT
    | _INOTIFY_Q_OVERFLOW
    | 0x00008000  # IN_IGNORED
)
_INOTIFY_DIRECTORY_MASK = _INOTIFY_MUTATION_EVENTS | 0x01000000  # IN_ONLYDIR
_INOTIFY_NAMESPACE_EVENTS = (
    0x00000040  # IN_MOVED_FROM
    | 0x00000080  # IN_MOVED_TO
    | 0x00000100  # IN_CREATE
    | 0x00000200  # IN_DELETE
)
_INOTIFY_READ_BYTES = 64 * 1024
_INOTIFY_MAX_DRAIN_ATTEMPTS = 128
_INOTIFY_MAX_DRAIN_BATCHES = 64
_INOTIFY_MAX_DRAIN_BYTES = 4 * 1024 * 1024
_INOTIFY_MAX_DRAIN_EVENTS = 65_536
_INVENTORY_DIAGNOSTIC_SAMPLE = 8
_ROLLBACK_GUARD_DIAGNOSTIC_LIMIT = 128
_DESCRIPTOR_RESOURCE_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE})
_PARTIAL_SUFFIX = ".m5-partial"
_PARTIAL_SUFFIXES = frozenset({
    ".m5-partial", ".m6-partial", ".ncu-details-partial",
})
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1

# Publication failures are deliberately stable strings: callers map these
# boundaries into their own typed result contracts without weakening the
# shared filesystem checks.
DURABILITY_UNPROVEN = "DURABILITY_UNPROVEN"
VERIFY_RESOURCE_BOUND = "VERIFY_RESOURCE_BOUND"
NAMESPACE_RESERVED_COMPONENT = "NAMESPACE_RESERVED_COMPONENT"
NAMESPACE_NAME_OVERFLOW = "NAMESPACE_NAME_OVERFLOW"
NAMESPACE_PREFIX_CONFLICT = "NAMESPACE_PREFIX_CONFLICT"


class ImmutableArtifactError(RuntimeError):
    """An immutable-artifact contract or filesystem operation failed."""


class ImmutableArtifactLimitError(ImmutableArtifactError):
    """A caller-supplied bounded-read or inventory ceiling was exceeded."""


class ImmutableArtifactResourceError(ImmutableArtifactError):
    """The host cannot provide a required bounded verification resource."""


class RenameOutcomeUnproven(ImmutableArtifactError):
    """An attempted atomic rename cannot be classified without guessing."""

    def __init__(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("unproven rename outcome must retain BaseException")
        self.error = error
        super().__init__(f"RENAME_OUTCOME_UNPROVEN: {error}")


class CommittedRenameInterruption(BaseException):
    """A caller interruption observed after an authenticated rename commit."""

    def __init__(
        self,
        interruption: BaseException,
        *,
        handoff_complete: bool,
    ) -> None:
        if not isinstance(interruption, BaseException):
            raise TypeError("committed rename interruption must be BaseException")
        if type(handoff_complete) is not bool:
            raise TypeError("committed rename handoff state must be boolean")
        self.interruption = interruption
        self.handoff_complete = handoff_complete
        super().__init__(str(interruption))


def _descriptor_resource_cause(
    error: BaseException,
) -> OSError | None:
    pending: list[BaseException] = [error]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in observed:
            continue
        observed.add(identity)
        if (
            isinstance(current, OSError)
            and current.errno in _DESCRIPTOR_RESOURCE_ERRNOS
        ):
            return current
        explicit_cause = current.__cause__
        if isinstance(explicit_cause, BaseException):
            pending.append(explicit_cause)
        elif not current.__suppress_context__:
            implicit_context = current.__context__
            if isinstance(implicit_context, BaseException):
                pending.append(implicit_context)
    return None


def _translate_descriptor_resource_errors(
    label: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def translated(*args: Any, **kwargs: Any) -> Any:
            try:
                return function(*args, **kwargs)
            except ImmutableArtifactResourceError:
                raise
            except BaseException as exc:
                resource_cause = _descriptor_resource_cause(exc)
                if resource_cause is None:
                    raise
                raise ImmutableArtifactResourceError(
                    f"{VERIFY_RESOURCE_BOUND}: {label}: {resource_cause}"
                ) from exc

        return translated

    return decorate


@dataclasses.dataclass(frozen=True)
class ImmutableFile:
    """One byte-exact file to materialize in a published directory."""

    relative_path: str
    sha256: str
    size_bytes: int
    source_path: Path | None = None
    payload: bytes | None = None
    allow_hardlinks: bool = False

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)
        if not _is_sha256(self.sha256):
            raise ImmutableArtifactError("file sha256 must be lowercase 64-hex")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ImmutableArtifactError("file size_bytes must be a nonnegative integer")
        if type(self.allow_hardlinks) is not bool:
            raise ImmutableArtifactError("allow_hardlinks must be boolean")
        if (self.source_path is None) == (self.payload is None):
            raise ImmutableArtifactError("file must have exactly one source")
        if self.payload is not None:
            if len(self.payload) != self.size_bytes:
                raise ImmutableArtifactError("payload size does not match file declaration")
            if sha256_bytes(self.payload) != self.sha256:
                raise ImmutableArtifactError("payload hash does not match file declaration")

    @classmethod
    def from_bytes(cls, relative_path: str, payload: bytes) -> "ImmutableFile":
        return cls(relative_path, sha256_bytes(payload), len(payload), payload=payload)

    @classmethod
    def from_path(
        cls,
        relative_path: str,
        source_path: Path,
        *,
        allow_hardlinks: bool = False,
    ) -> "ImmutableFile":
        digest, size = sha256_file(
            source_path, allow_hardlinks=allow_hardlinks
        )
        return cls(
            relative_path,
            digest,
            size,
            source_path=Path(source_path),
            allow_hardlinks=allow_hardlinks,
        )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def safe_relative_path(value: str) -> PurePosixPath:
    """Return a byte-for-byte canonical, bounded POSIX relative path."""
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ImmutableArtifactError(f"unsafe relative path: {value!r}")
    if len(os.fsencode(value)) > _MAX_RELATIVE_BYTES:
        raise ImmutableArtifactError("relative path exceeds the bounded traversal limit")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) > _MAX_COMPONENTS
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(len(os.fsencode(part)) > 255 for part in path.parts)
        or path.as_posix() != value
    ):
        raise ImmutableArtifactError(f"noncanonical relative path: {value!r}")
    return path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a model for hashing, independently of presentation bytes."""
    _reject_nonfinite(value)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ImmutableArtifactError(f"model is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ImmutableArtifactError(f"model is not JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def model_seal(value: Mapping[str, Any], field: str = "model_seal_sha256") -> str:
    if type(value) is not dict:
        raise ImmutableArtifactError("sealed model must be an exact JSON object")
    unsigned = dict(value)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def seal_model(value: Mapping[str, Any], field: str = "model_seal_sha256") -> dict[str, Any]:
    if field in value:
        raise ImmutableArtifactError(f"unsigned model unexpectedly contains {field}")
    sealed = dict(value)
    sealed[field] = model_seal(sealed, field)
    return sealed


def verify_model_seal(value: Mapping[str, Any], field: str = "model_seal_sha256") -> None:
    observed = value.get(field)
    if not _is_sha256(observed):
        raise ImmutableArtifactError(f"{field} must be lowercase 64-hex")
    if observed != model_seal(value, field):
        raise ImmutableArtifactError(f"{field} mismatch")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ImmutableArtifactError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ImmutableArtifactError(f"non-finite JSON number: {token}")


def _reject_nonfinite(value: Any) -> None:
    if type(value) is float:
        if not math.isfinite(value):
            raise ImmutableArtifactError("non-finite JSON number")
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise ImmutableArtifactError("JSON float exceeds the exact safe range")
    if type(value) is int and abs(value) > _MAX_SAFE_JSON_INTEGER:
        raise ImmutableArtifactError("JSON integer exceeds the exact safe range")
    if type(value) is str and any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ImmutableArtifactError("JSON string contains an unpaired surrogate")
    if type(value) in {list, tuple}:
        for item in value:
            _reject_nonfinite(item)
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ImmutableArtifactError("JSON object keys must be strings")
            _reject_nonfinite(key)
            _reject_nonfinite(item)


def strict_json_loads(payload: bytes, *, label: str = "JSON") -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImmutableArtifactError(f"{label} is not UTF-8: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except ImmutableArtifactError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ImmutableArtifactError(f"invalid {label}: {exc}") from exc
    try:
        _reject_nonfinite(value)
    except RecursionError as exc:
        raise ImmutableArtifactError(f"invalid {label}: nesting exceeds safe limits") from exc
    return value


MetadataSnapshot = tuple[int, int, int, int, int, int, int]
_Snapshot = MetadataSnapshot
_TreeSnapshot = tuple[dict[str, _Snapshot], dict[str, _Snapshot]]


@dataclasses.dataclass(frozen=True)
class PinnedRegularSnapshot:
    """Bytes and seven-field identity consumed through one pinned leaf."""

    payload: bytes
    metadata: MetadataSnapshot


@dataclasses.dataclass(frozen=True)
class PinnedTreeSnapshot:
    """One stable descriptor-relative tree inventory; root is directory `.`."""

    files: tuple[tuple[str, MetadataSnapshot], ...]
    directories: tuple[tuple[str, MetadataSnapshot], ...]


def _snapshot_fields(metadata: os.stat_result) -> _Snapshot:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _consume_stable_regular(
    descriptor: int,
    consume: Callable[[bytes], None],
    *,
    allow_hardlinks: bool = False,
    max_bytes: int | None = None,
) -> tuple[int, int, MetadataSnapshot]:
    """Consume a pinned leaf through the anchored-reader state machine.

    Ownership transitions are CLOSED -> CHAIN_PINNED -> LEAF_PINNED ->
    CONSUMED -> REVERIFIED -> CLOSED.  The caller owns the descriptor until
    the surrounding ``finally`` closes it; every transition failure therefore
    preserves the same reverse-order cleanup contract.  Ancestors are bound
    by name to device/inode/type identity, while the leaf retains the complete
    seven-field snapshot for the byte-consuming interval.
    """
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ImmutableArtifactError("opened artifact is not a regular file")
    if not allow_hardlinks and before.st_nlink != 1:
        raise ImmutableArtifactError("hard-linked regular file is not immutable")
    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ImmutableArtifactError("regular snapshot byte limit is invalid")
        if before.st_size > max_bytes:
            raise ImmutableArtifactLimitError(
                f"regular file exceeds bounded snapshot limit: "
                f"{before.st_size} > {max_bytes} bytes"
            )
    remaining = before.st_size
    consumed = 0
    while remaining:
        try:
            block = os.read(descriptor, min(_CHUNK, remaining))
        except InterruptedError:
            continue
        if not block:
            raise ImmutableArtifactError("regular file truncated during snapshot read")
        consume(block)
        consumed += len(block)
        remaining -= len(block)
    while True:
        try:
            extra = os.read(descriptor, 1)
            break
        except InterruptedError:
            continue
    after = os.fstat(descriptor)
    snapshot = _snapshot_fields(before)
    if extra or consumed != before.st_size or _snapshot_fields(after) != snapshot:
        raise ImmutableArtifactError("regular file changed during snapshot read")
    return consumed, before.st_mode, snapshot


def _read_fd_snapshot(
    descriptor: int,
    *,
    allow_hardlinks: bool = False,
    max_bytes: int | None = None,
) -> tuple[bytes, MetadataSnapshot]:
    blocks: list[bytes] = []
    _size, _mode, snapshot = _consume_stable_regular(
        descriptor,
        blocks.append,
        allow_hardlinks=allow_hardlinks,
        max_bytes=max_bytes,
    )
    return b"".join(blocks), snapshot


def _read_fd(descriptor: int, *, allow_hardlinks: bool = False) -> bytes:
    payload, _snapshot = _read_fd_snapshot(
        descriptor, allow_hardlinks=allow_hardlinks,
    )
    return payload


def _hash_fd(
    descriptor: int,
    *,
    allow_hardlinks: bool = False,
    max_bytes: int | None = None,
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size, mode, _snapshot = _consume_stable_regular(
        descriptor,
        digest.update,
        allow_hardlinks=allow_hardlinks,
        max_bytes=max_bytes,
    )
    return digest.hexdigest(), size, mode


def _nofollow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if type(flag) is not int or flag == 0:
        raise ImmutableArtifactError(
            "secure no-follow filesystem operations are unavailable"
        )
    return flag


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | _nofollow_flag()
    )


def _open_dir_at(parent_descriptor: int, name: str) -> int:
    return os.open(name, _directory_flags(), dir_fd=parent_descriptor)


def _open_independent_directory(descriptor: int) -> int:
    independent = _open_dir_at(descriptor, ".")
    try:
        if _node_identity(os.fstat(independent)) != _node_identity(
            os.fstat(descriptor)
        ):
            raise ImmutableArtifactError(
                "independent directory descriptor changed during open"
            )
        return independent
    except BaseException:
        _close_descriptors(
            (independent,),
            active_error=sys.exc_info()[1],
            label="independent directory descriptor close",
        )
        raise


@contextlib.contextmanager
def _scandir_independent(descriptor: int):
    independent = _open_independent_directory(descriptor)
    try:
        with os.scandir(independent) as entries:
            yield entries
    finally:
        _close_descriptors(
            (independent,),
            active_error=sys.exc_info()[1],
            label="independent directory scan close",
        )


def _node_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _fsync_or_fail(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ImmutableArtifactError(
            f"{DURABILITY_UNPROVEN}: {label}: {exc}"
        ) from exc


def _close_descriptors(
    descriptors: Iterable[int],
    *,
    active_error: BaseException | None = None,
    label: str = "descriptor close",
) -> None:
    """Close every descriptor without hiding a primary failure.

    A close-only failure is itself a failed durability/ownership boundary and
    must not be silently converted into success.  When called from an active
    exception path, the original exception remains primary and the close fault
    is attached as diagnostic context.
    """
    close_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as exc:
            if close_error is None:
                close_error = exc
    if close_error is None:
        return
    if active_error is None:
        active_error = sys.exc_info()[1]
    if active_error is not None:
        try:
            active_error.add_note(f"{DURABILITY_UNPROVEN}: {label}: {close_error}")
        except (AttributeError, TypeError):
            pass
        return
    raise ImmutableArtifactError(
        f"{DURABILITY_UNPROVEN}: {label}: {close_error}"
    ) from close_error


def _close_descriptors_without_callback(
    descriptors: Iterable[int],
    *,
    label: str,
) -> None:
    """Release final sentinels through libc without a Python close callback."""
    libc = ctypes.CDLL(None, use_errno=True)
    close_call = libc.close
    close_call.argtypes = [ctypes.c_int]
    close_call.restype = ctypes.c_int
    close_error: OSError | None = None
    for descriptor in descriptors:
        ctypes.set_errno(0)
        if close_call(descriptor) == 0:
            continue
        error_number = ctypes.get_errno() or errno.EIO
        if close_error is None:
            close_error = OSError(
                error_number,
                f"{label}: {os.strerror(error_number)}",
            )
    if close_error is not None:
        raise ImmutableArtifactError(
            f"{DURABILITY_UNPROVEN}: {label}: {close_error}"
        ) from close_error


def _close_callables(
    closers: Iterable[Callable[[], None]], *, label: str
) -> None:
    """Run all cleanup callbacks, preserving the first active failure."""
    active_error = sys.exc_info()[1]
    close_error: BaseException | None = None
    for closer in closers:
        try:
            closer()
        except BaseException as exc:
            if close_error is None:
                close_error = exc
    if close_error is None:
        return
    if active_error is not None:
        try:
            active_error.add_note(f"{DURABILITY_UNPROVEN}: {label}: {close_error}")
        except (AttributeError, TypeError):
            pass
        return
    raise close_error


def _verify_directory_chain(descriptors: tuple[int, ...], names: tuple[str, ...]) -> None:
    if len(descriptors) != len(names) + 1:
        raise ImmutableArtifactError("invalid pinned directory chain")
    for parent, child, name in zip(
        descriptors[:-1], descriptors[1:], names, strict=True
    ):
        try:
            linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(child)
        except OSError as exc:
            raise ImmutableArtifactError("named directory component chain changed") from exc
        if (
            stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISDIR(linked.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _node_identity(linked) != _node_identity(opened)
        ):
            raise ImmutableArtifactError("named directory component chain changed")


def _verify_named_directory(
    parent_descriptor: int, name: str, descriptor: int
) -> None:
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise ImmutableArtifactError(f"named directory changed: {name}") from exc
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or _node_identity(linked) != _node_identity(opened)
    ):
        raise ImmutableArtifactError(f"named directory changed: {name}")


def _lock_directory(descriptor: int, label: str, *, blocking: bool) -> None:
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    while True:
        try:
            fcntl.flock(descriptor, operation)
            return
        except InterruptedError:
            continue
        except BlockingIOError as exc:
            raise ImmutableArtifactError(
                f"immutable publication is owned by a live publisher: {label}"
            ) from exc
        except OSError as exc:
            raise ImmutableArtifactError(
                f"cannot lock immutable publication directory {label}: {exc}"
            ) from exc


def _open_directory_chain(path: Path) -> tuple[tuple[int, ...], tuple[str, ...]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    names = tuple(absolute.parts[1:])
    if len(names) > _MAX_COMPONENTS or len(os.fsencode(os.fspath(absolute))) > _MAX_RELATIVE_BYTES:
        raise ImmutableArtifactError("absolute directory traversal exceeds bounded limits")
    descriptors = [
        os.open(os.path.sep, _directory_flags() & ~_nofollow_flag())
    ]
    try:
        for name in names:
            descriptors.append(_open_dir_at(descriptors[-1], name))
        pinned = tuple(descriptors)
        _verify_directory_chain(pinned, names)
        return pinned, names
    except BaseException:
        _close_descriptors(reversed(descriptors))
        raise


def _open_absolute_directory(path: Path) -> int:
    descriptors, _ = _open_directory_chain(path)
    retained = descriptors[-1]
    try:
        _close_descriptors(
            reversed(descriptors[:-1]), label=f"directory ancestors {path} close"
        )
    except BaseException:
        try:
            _close_descriptors(
                (retained,),
                active_error=sys.exc_info()[1],
                label=f"directory {path} retained close",
            )
        finally:
            raise
    return retained


@dataclasses.dataclass
class _PinnedDirectory:
    descriptors: tuple[int, ...]
    names: tuple[str, ...]

    @property
    def descriptor(self) -> int:
        if not self.descriptors:
            raise ImmutableArtifactError("pinned directory is closed")
        return self.descriptors[-1]

    def verify_path(self) -> None:
        _verify_directory_chain(self.descriptors, self.names)

    def close(self) -> None:
        descriptors = self.descriptors
        self.descriptors = ()
        if descriptors:
            _close_descriptors(reversed(descriptors), label="pinned directory close")


def _open_pinned_directory(path: Path) -> _PinnedDirectory:
    descriptors, names = _open_directory_chain(path)
    return _PinnedDirectory(descriptors, names)


@dataclasses.dataclass
class _PinnedRegular:
    descriptor: int
    directory_descriptors: tuple[int, ...]
    directory_names: tuple[str, ...]
    leaf_name: str
    allow_hardlinks: bool = False

    def verify_path(self) -> None:
        _verify_directory_chain(self.directory_descriptors, self.directory_names)
        try:
            linked = os.stat(
                self.leaf_name,
                dir_fd=self.directory_descriptors[-1],
                follow_symlinks=False,
            )
            opened = os.fstat(self.descriptor)
        except OSError as exc:
            raise ImmutableArtifactError("named regular-file path changed") from exc
        if (
            stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or _node_identity(linked) != _node_identity(opened)
        ):
            raise ImmutableArtifactError("named regular-file path changed")
        if (
            not self.allow_hardlinks
            and (linked.st_nlink != 1 or opened.st_nlink != 1)
        ):
            raise ImmutableArtifactError("hard-linked regular file is not immutable")
        _verify_directory_chain(self.directory_descriptors, self.directory_names)

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            directory_descriptors = self.directory_descriptors
            self.descriptor = -1
            self.directory_descriptors = ()
            _close_descriptors(
                (descriptor, *reversed(directory_descriptors)),
                label="pinned regular close",
            )


def _open_regular(
    path: Path, *, allow_hardlinks: bool = False
) -> _PinnedRegular:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.parent == absolute:
        raise ImmutableArtifactError(f"not a regular file: {absolute}")
    try:
        directory_descriptors, directory_names = _open_directory_chain(absolute.parent)
    except (OSError, ImmutableArtifactError) as exc:
        raise ImmutableArtifactError(
            f"cannot traverse regular-file ancestors for {absolute}: {exc}"
        ) from exc
    try:
        declared = os.stat(
            absolute.name,
            dir_fd=directory_descriptors[-1],
            follow_symlinks=False,
        )
    except OSError as exc:
        _close_descriptors(reversed(directory_descriptors))
        raise ImmutableArtifactError(f"cannot inspect regular file {absolute}: {exc}") from exc
    if stat.S_ISLNK(declared.st_mode) or not stat.S_ISREG(declared.st_mode):
        _close_descriptors(reversed(directory_descriptors))
        raise ImmutableArtifactError(f"not a regular file: {absolute}")
    if declared.st_nlink != 1 and not allow_hardlinks:
        _close_descriptors(reversed(directory_descriptors))
        raise ImmutableArtifactError(f"hard-linked regular file is not immutable: {absolute}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _nofollow_flag()
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(
            absolute.name, flags, dir_fd=directory_descriptors[-1]
        )
    except OSError as exc:
        _close_descriptors(reversed(directory_descriptors))
        raise ImmutableArtifactError(f"cannot open regular file {absolute}: {exc}") from exc
    pinned = _PinnedRegular(
        descriptor,
        directory_descriptors,
        directory_names,
        absolute.name,
        allow_hardlinks=allow_hardlinks,
    )
    try:
        pinned.verify_path()
    except BaseException:
        pinned.close()
        raise
    return pinned


@dataclasses.dataclass
class RetainedExecutable:
    """One no-follow executable retained across an identity-sensitive action."""

    path: Path
    sha256: str
    snapshot: _Snapshot
    _pinned: _PinnedRegular
    _monitor_state: Any = dataclasses.field(
        default=None, init=False, repr=False,
    )

    @property
    def descriptor(self) -> int:
        if self._pinned.descriptor < 0:
            raise ImmutableArtifactError("retained executable is closed")
        return self._pinned.descriptor

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def verify_unchanged(self) -> None:
        """Rebind path, inode, metadata, and bytes to the initial observation."""
        _require_retained_path_monitor_quiet(self._monitor_state)
        descriptor = self.descriptor
        self._pinned.verify_path()
        if _snapshot_fields(os.fstat(descriptor)) != self.snapshot:
            raise ImmutableArtifactError("retained executable metadata changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, _size, mode = _hash_fd(descriptor)
        if _snapshot_fields(os.fstat(descriptor)) != self.snapshot:
            raise ImmutableArtifactError("retained executable metadata changed")
        if digest != self.sha256:
            raise ImmutableArtifactError("retained executable SHA-256 changed")
        if mode & 0o111 == 0:
            raise ImmutableArtifactError("retained executable lost execute permission")
        self._pinned.verify_path()
        os.lseek(descriptor, 0, os.SEEK_SET)
        _require_retained_path_monitor_quiet(self._monitor_state)

    def close(self) -> None:
        state = self._monitor_state
        self._monitor_state = None
        closers: list[Callable[[], None]] = []
        if state is not None:
            closers.append(state.monitor.close)
        closers.append(self._pinned.close)
        _close_callables(closers, label="retained executable close")

    def __enter__(self) -> RetainedExecutable:
        if self._monitor_state is None:
            self._monitor_state = _arm_retained_path_monitor(self._pinned)
        try:
            self.verify_unchanged()
        except BaseException:
            _close_callables(
                (self.close,), label="retained executable entry close"
            )
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            try:
                self.verify_unchanged()
            except BaseException as verification_error:
                try:
                    exc.add_note(
                        "retained executable revalidation failed: "
                        f"{verification_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            self.close()
            return False
        try:
            self.verify_unchanged()
        finally:
            self.close()
        return False


@_translate_descriptor_resource_errors("retained executable open")
def open_retained_executable(path: Path) -> RetainedExecutable:
    """Open, hash, and retain one executable without following any symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    pinned, monitor_state = _open_regular_with_early_path_monitor(absolute)
    retained: RetainedExecutable | None = None
    try:
        descriptor = pinned.descriptor
        before = _snapshot_fields(os.fstat(descriptor))
        if before[2] & 0o111 == 0:
            raise ImmutableArtifactError(f"regular file is not executable: {absolute}")
        # Ancestor and leaf watches are armed before path traversal completes,
        # so a transient namespace swap cannot hide before this identity hash.
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, _size, _mode = _hash_fd(descriptor)
        after = _snapshot_fields(os.fstat(descriptor))
        if after != before:
            raise ImmutableArtifactError("executable metadata changed during identity read")
        _require_retained_path_monitor_quiet(monitor_state)
        pinned.verify_path()
        _require_retained_path_monitor_quiet(monitor_state)
        os.lseek(descriptor, 0, os.SEEK_SET)
        retained = RetainedExecutable(absolute, digest, before, pinned)
        retained._monitor_state = monitor_state
        monitor_state = None
        retained.verify_unchanged()
        return retained
    except BaseException:
        if retained is None:
            closers: list[Callable[[], None]] = []
            if monitor_state is not None:
                closers.append(monitor_state.monitor.close)
            closers.append(pinned.close)
            _close_callables(closers, label="retained executable open cleanup")
        else:
            retained.close()
        raise


@dataclasses.dataclass
class RetainedRegular:
    """One no-follow regular file retained as a caller-owned capability."""

    path: Path
    sha256: str
    size_bytes: int
    snapshot: _Snapshot
    _pinned: _PinnedRegular
    _monitor_state: Any = dataclasses.field(
        default=None, init=False, repr=False,
    )

    @property
    def descriptor(self) -> int:
        if self._pinned.descriptor < 0:
            raise ImmutableArtifactError("retained regular file is closed")
        return self._pinned.descriptor

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def read_bytes(self) -> bytes:
        """Read this retained inode while preserving path and snapshot identity."""
        _require_retained_path_monitor_quiet(self._monitor_state)
        descriptor = self.descriptor
        self._pinned.verify_path()
        if _snapshot_fields(os.fstat(descriptor)) != self.snapshot:
            raise ImmutableArtifactError("retained regular-file metadata changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = _read_fd(descriptor)
        if _snapshot_fields(os.fstat(descriptor)) != self.snapshot:
            raise ImmutableArtifactError("retained regular-file metadata changed")
        if len(payload) != self.size_bytes:
            raise ImmutableArtifactError("retained regular-file size changed")
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ImmutableArtifactError("retained regular-file SHA-256 changed")
        self._pinned.verify_path()
        os.lseek(descriptor, 0, os.SEEK_SET)
        _require_retained_path_monitor_quiet(self._monitor_state)
        return payload

    def verify_unchanged(self) -> None:
        """Rebind path, inode, metadata, size, and bytes to the initial read."""
        _require_retained_path_monitor_quiet(self._monitor_state)
        descriptor = self.descriptor
        self._pinned.verify_path()
        if _snapshot_fields(os.fstat(descriptor)) != self.snapshot:
            raise ImmutableArtifactError("retained regular-file metadata changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, size, _mode = _hash_fd(descriptor)
        if _snapshot_fields(os.fstat(descriptor)) != self.snapshot:
            raise ImmutableArtifactError("retained regular-file metadata changed")
        if size != self.size_bytes:
            raise ImmutableArtifactError("retained regular-file size changed")
        if digest != self.sha256:
            raise ImmutableArtifactError("retained regular-file SHA-256 changed")
        self._pinned.verify_path()
        os.lseek(descriptor, 0, os.SEEK_SET)
        _require_retained_path_monitor_quiet(self._monitor_state)

    def close(self) -> None:
        state = self._monitor_state
        self._monitor_state = None
        closers: list[Callable[[], None]] = []
        if state is not None:
            closers.append(state.monitor.close)
        closers.append(self._pinned.close)
        _close_callables(closers, label="retained regular-file close")

    def __enter__(self) -> RetainedRegular:
        if self._monitor_state is None:
            self._monitor_state = _arm_retained_path_monitor(self._pinned)
        try:
            self.verify_unchanged()
        except BaseException:
            _close_callables(
                (self.close,), label="retained regular-file entry close"
            )
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            try:
                self.verify_unchanged()
            except BaseException as verification_error:
                try:
                    exc.add_note(
                        "retained regular-file revalidation failed: "
                        f"{verification_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            self.close()
            return False
        try:
            self.verify_unchanged()
        finally:
            self.close()
        return False


@_translate_descriptor_resource_errors("retained regular-file open")
def open_retained_regular(path: Path) -> RetainedRegular:
    """Open, hash, and retain one regular file without following any symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    pinned, monitor_state = _open_regular_with_early_path_monitor(absolute)
    retained: RetainedRegular | None = None
    try:
        descriptor = pinned.descriptor
        before = _snapshot_fields(os.fstat(descriptor))
        # Ancestor and leaf watches already cover this first identity hash.
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, size, _mode = _hash_fd(descriptor)
        after = _snapshot_fields(os.fstat(descriptor))
        if after != before:
            raise ImmutableArtifactError(
                "regular-file metadata changed during identity read"
            )
        _require_retained_path_monitor_quiet(monitor_state)
        pinned.verify_path()
        _require_retained_path_monitor_quiet(monitor_state)
        os.lseek(descriptor, 0, os.SEEK_SET)
        retained = RetainedRegular(absolute, digest, size, before, pinned)
        retained._monitor_state = monitor_state
        monitor_state = None
        retained.verify_unchanged()
        return retained
    except BaseException:
        if retained is None:
            closers: list[Callable[[], None]] = []
            if monitor_state is not None:
                closers.append(monitor_state.monitor.close)
            closers.append(pinned.close)
            _close_callables(closers, label="retained regular-file open cleanup")
        else:
            retained.close()
        raise


def _duplicate_pinned_regular_for_close(
    pinned: _PinnedRegular,
) -> _PinnedRegular:
    """Duplicate one exact inode/ancestor capability without closing via hooks."""
    directory_descriptors: list[int] = []
    descriptor = -1
    duplicate: _PinnedRegular | None = None
    try:
        ancestor_nodes = tuple(
            _node_identity(os.fstat(value))
            for value in pinned.directory_descriptors
        )
        leaf_snapshot = _snapshot_fields(os.fstat(pinned.descriptor))
        for value in pinned.directory_descriptors:
            directory_descriptors.append(os.dup(value))
        descriptor = os.dup(pinned.descriptor)
        duplicate = _PinnedRegular(
            descriptor,
            tuple(directory_descriptors),
            pinned.directory_names,
            pinned.leaf_name,
            allow_hardlinks=pinned.allow_hardlinks,
        )
        descriptor = -1
        directory_descriptors = []
        duplicate.verify_path()
        if (
            tuple(
                _node_identity(os.fstat(value))
                for value in duplicate.directory_descriptors
            )
            != ancestor_nodes
            or _snapshot_fields(os.fstat(duplicate.descriptor))
            != leaf_snapshot
        ):
            raise ImmutableArtifactError(
                "retained regular-file close guard changed while duplicating"
            )
        return duplicate
    except BaseException:
        descriptors = tuple(
            value
            for value in (
                descriptor,
                *reversed(directory_descriptors),
            )
            if value >= 0
        )
        if duplicate is not None:
            descriptors = (
                duplicate.descriptor,
                *reversed(duplicate.directory_descriptors),
            )
            duplicate.descriptor = -1
            duplicate.directory_descriptors = ()
        if descriptors:
            _close_descriptors(
                descriptors,
                label="retained regular-file close guard cleanup",
            )
        raise


def _close_pinned_regular_without_hook(
    pinned: _PinnedRegular,
    *,
    label: str,
) -> None:
    descriptor = pinned.descriptor
    directory_descriptors = pinned.directory_descriptors
    descriptors = tuple(
        value
        for value in (descriptor, *reversed(directory_descriptors))
        if value >= 0
    )
    # close(2) may release a descriptor even when the reporting boundary raises.
    # Consume numeric ownership before the first attempt so no caller can retry
    # an uncertain number after another thread or callback has reused it.
    pinned.descriptor = -1
    pinned.directory_descriptors = ()
    if descriptors:
        _close_descriptors(descriptors, label=label)


def _close_pinned_regular_final_sentinels(
    pinned_files: tuple[_PinnedRegular, ...],
    *,
    label: str = "authenticated retained final sentinel release",
) -> None:
    """Release a complete sentinel set without another hookable close boundary."""
    descriptors = tuple(
        descriptor
        for pinned in reversed(pinned_files)
        for descriptor in (
            pinned.descriptor,
            *reversed(pinned.directory_descriptors),
        )
        if descriptor >= 0
    )
    try:
        if descriptors:
            _close_descriptors_without_callback(descriptors, label=label)
    finally:
        # Raw close(2) is attempted for every descriptor before an error is
        # reported. Never expose an uncertain descriptor number for reuse.
        for pinned in pinned_files:
            pinned.descriptor = -1
            pinned.directory_descriptors = ()


def _verify_retained_regular_with_pinned(
    retained: RetainedRegular,
    pinned: _PinnedRegular,
    *,
    monitor_state: _RetainedPathMonitorState | None,
) -> None:
    _require_retained_path_monitor_quiet(monitor_state)
    pinned.verify_path()
    descriptor = pinned.descriptor
    if _snapshot_fields(os.fstat(descriptor)) != retained.snapshot:
        raise ImmutableArtifactError("retained regular-file metadata changed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest, size, _mode = _hash_fd(descriptor)
    if _snapshot_fields(os.fstat(descriptor)) != retained.snapshot:
        raise ImmutableArtifactError("retained regular-file metadata changed")
    if size != retained.size_bytes:
        raise ImmutableArtifactError("retained regular-file size changed")
    if digest != retained.sha256:
        raise ImmutableArtifactError("retained regular-file SHA-256 changed")
    pinned.verify_path()
    os.lseek(descriptor, 0, os.SEEK_SET)
    _require_retained_path_monitor_quiet(monitor_state)


@_translate_descriptor_resource_errors(
    "authenticated retained regular-file close"
)
def close_retained_regular_files_authenticated(
    retained_files: tuple[RetainedRegular, ...],
    *,
    lock_reader: PinnedDirectoryReader,
    closure_callbacks: tuple[Callable[[], None], ...] = (),
) -> None:
    """Authenticate and close retained files while holding one publisher lock.

    Every original inode/ancestor capability is duplicated into an independent
    rollback guard before closure. Mutation monitors remain armed while the
    original ``_PinnedRegular.close`` operations and caller lock/source callbacks
    run. After monitor closure and guard authentication, a final sentinel set is
    created before the hookable guard-release boundary and authenticated again
    afterward. Failure transfers the current complete recovery set back into its
    ``RetainedRegular`` owners; success releases all final sentinels through raw
    libc ``close(2)`` without a later Python close callback.
    """
    if type(retained_files) is not tuple or not retained_files:
        raise ImmutableArtifactError(
            "authenticated retained close requires a nonempty exact tuple"
        )
    if type(lock_reader) is not PinnedDirectoryReader:
        raise ImmutableArtifactError(
            "authenticated retained close requires an exact pinned lock reader"
        )
    if any(type(retained) is not RetainedRegular for retained in retained_files):
        raise ImmutableArtifactError(
            "authenticated retained close requires exact retained regular files"
        )
    if len({id(retained) for retained in retained_files}) != len(retained_files):
        raise ImmutableArtifactError(
            "authenticated retained close rejects duplicate capabilities"
        )
    if type(closure_callbacks) is not tuple or any(
        not callable(callback) for callback in closure_callbacks
    ):
        raise ImmutableArtifactError(
            "authenticated retained close callbacks must be an exact callable tuple"
        )

    lock_reader.verify_path()
    _lock_directory(
        lock_reader.descriptor,
        "authenticated retained regular-file close",
        blocking=False,
    )
    lock_reader.verify_path()
    for retained in retained_files:
        retained.verify_unchanged()
    lock_reader.verify_path()

    originals = tuple(retained._pinned for retained in retained_files)
    guards: list[_PinnedRegular | None] = []
    try:
        for original in originals:
            guards.append(_duplicate_pinned_regular_for_close(original))
    except BaseException:
        for guard in reversed(guards):
            if guard is not None:
                _close_pinned_regular_without_hook(
                    guard, label="retained close guard construction cleanup",
                )
        raise

    def transfer_guards() -> None:
        for retained, original, guard in zip(
            retained_files, originals, guards, strict=True,
        ):
            if guard is None:
                continue
            if original.descriptor >= 0 or original.directory_descriptors:
                try:
                    _close_pinned_regular_without_hook(
                        original,
                        label="failed authenticated original capability cleanup",
                    )
                except BaseException:
                    pass
            retained._pinned = guard
        guards[:] = [None] * len(guards)

    pinned_close_errors: list[BaseException] = []
    for original in reversed(originals):
        try:
            original.close()
        except BaseException as exc:
            pinned_close_errors.append(exc)

    callback_errors: list[BaseException] = []
    for callback in closure_callbacks:
        try:
            callback()
        except BaseException as exc:
            callback_errors.append(exc)

    verification_errors: list[BaseException] = []
    for retained, guard in zip(retained_files, guards, strict=True):
        assert guard is not None
        try:
            _verify_retained_regular_with_pinned(
                retained,
                guard,
                monitor_state=retained._monitor_state,
            )
        except BaseException as exc:
            verification_errors.append(exc)

    first_phase_errors = (
        callback_errors + pinned_close_errors + verification_errors
    )
    if first_phase_errors:
        transfer_guards()
        primary = first_phase_errors[0]
        for additional in first_phase_errors[1:]:
            try:
                primary.add_note(
                    "additional authenticated retained-close failure: "
                    f"{additional}"
                )
            except (AttributeError, TypeError):
                pass
        raise primary

    monitor_close_errors: list[BaseException] = []
    for retained in retained_files:
        state = retained._monitor_state
        if state is None:
            continue
        try:
            _require_retained_path_monitor_quiet(state)
            state.monitor.close()
        except BaseException as exc:
            monitor_close_errors.append(exc)
        else:
            retained._monitor_state = None

    post_close_errors: list[BaseException] = []
    for retained, guard in zip(retained_files, guards, strict=True):
        assert guard is not None
        try:
            _verify_retained_regular_with_pinned(
                retained,
                guard,
                monitor_state=retained._monitor_state,
            )
        except BaseException as exc:
            post_close_errors.append(exc)

    final_errors = monitor_close_errors + post_close_errors
    if final_errors:
        transfer_guards()
        primary = final_errors[0]
        for additional in final_errors[1:]:
            try:
                primary.add_note(
                    "additional authenticated retained-close failure: "
                    f"{additional}"
                )
            except (AttributeError, TypeError):
                pass
        raise primary

    final_sentinels: list[_PinnedRegular | None] = []
    try:
        for guard in guards:
            assert guard is not None
            final_sentinels.append(_duplicate_pinned_regular_for_close(guard))
    except BaseException as sentinel_error:
        sentinel_cleanup_error: BaseException | None = None
        constructed_sentinels = tuple(
            sentinel
            for sentinel in final_sentinels
            if sentinel is not None
        )
        if constructed_sentinels:
            try:
                _close_pinned_regular_final_sentinels(
                    constructed_sentinels,
                    label="failed final sentinel construction cleanup",
                )
            except BaseException as exc:
                sentinel_cleanup_error = exc
        transfer_guards()
        if sentinel_cleanup_error is not None:
            try:
                sentinel_error.add_note(
                    "additional final sentinel construction cleanup failure: "
                    f"{sentinel_cleanup_error}"
                )
            except (AttributeError, TypeError):
                pass
        raise

    guard_release_errors: list[BaseException] = []
    for guard in reversed(guards):
        assert guard is not None
        try:
            _close_pinned_regular_without_hook(
                guard, label="authenticated retained close guard release",
            )
        except BaseException as exc:
            guard_release_errors.append(exc)

    guards[:] = [None] * len(guards)

    sentinel_verification_errors: list[BaseException] = []
    for retained, sentinel in zip(
        retained_files, final_sentinels, strict=True,
    ):
        assert sentinel is not None
        try:
            _verify_retained_regular_with_pinned(
                retained,
                sentinel,
                monitor_state=retained._monitor_state,
            )
        except BaseException as exc:
            sentinel_verification_errors.append(exc)

    successor_errors = guard_release_errors + sentinel_verification_errors
    if successor_errors:
        for retained, sentinel in zip(
            retained_files, final_sentinels, strict=True,
        ):
            assert sentinel is not None
            retained._pinned = sentinel
        final_sentinels[:] = [None] * len(final_sentinels)
        primary = successor_errors[0]
        for additional in successor_errors[1:]:
            try:
                primary.add_note(
                    "additional authenticated retained-close successor failure: "
                    f"{additional}"
                )
            except (AttributeError, TypeError):
                pass
        raise primary

    authenticated_sentinels = tuple(
        sentinel
        for sentinel in final_sentinels
        if sentinel is not None
    )
    try:
        _close_pinned_regular_final_sentinels(authenticated_sentinels)
    finally:
        final_sentinels[:] = [None] * len(final_sentinels)


def read_regular_snapshot(
    path: Path,
    *,
    allow_hardlinks: bool = False,
    max_bytes: int | None = None,
) -> PinnedRegularSnapshot:
    pinned = _open_regular(Path(path), allow_hardlinks=allow_hardlinks)
    try:
        payload, metadata = _read_fd_snapshot(
            pinned.descriptor,
            allow_hardlinks=allow_hardlinks,
            max_bytes=max_bytes,
        )
        pinned.verify_path()
        if _snapshot_fields(os.fstat(pinned.descriptor)) != metadata:
            raise ImmutableArtifactError("regular file changed after snapshot read")
        return PinnedRegularSnapshot(payload, metadata)
    finally:
        pinned.close()


def snapshot_regular_identity(
    path: Path, *, allow_hardlinks: bool = False,
) -> MetadataSnapshot:
    """Return one anchored, byte-free identity for an exact regular leaf."""
    pinned = _open_regular(Path(path), allow_hardlinks=allow_hardlinks)
    try:
        before = _snapshot_fields(os.fstat(pinned.descriptor))
        pinned.verify_path()
        after = _snapshot_fields(os.fstat(pinned.descriptor))
        if after != before:
            raise ImmutableArtifactError("regular file changed during identity snapshot")
        return after
    finally:
        pinned.close()


def read_regular(path: Path, *, allow_hardlinks: bool = False) -> bytes:
    return read_regular_snapshot(
        path, allow_hardlinks=allow_hardlinks,
    ).payload


def strict_json_load(path: Path) -> Any:
    return strict_json_loads(read_regular(path), label=os.fspath(path))


def sha256_file(
    path: Path, *, allow_hardlinks: bool = False
) -> tuple[str, int]:
    pinned = _open_regular(Path(path), allow_hardlinks=allow_hardlinks)
    try:
        digest, size, _ = _hash_fd(
            pinned.descriptor, allow_hardlinks=allow_hardlinks
        )
        pinned.verify_path()
        return digest, size
    finally:
        pinned.close()


def ensure_directory(path: Path, mode: int = 0o755) -> None:
    """Create/traverse an absolute directory with a pinned no-follow chain."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    names = tuple(absolute.parts[1:])
    if len(names) > _MAX_COMPONENTS or len(os.fsencode(os.fspath(absolute))) > _MAX_RELATIVE_BYTES:
        raise ImmutableArtifactError("absolute directory traversal exceeds bounded limits")
    descriptors = [
        os.open(os.path.sep, _directory_flags() & ~_nofollow_flag())
    ]
    try:
        for name in names:
            created = False
            try:
                os.mkdir(name, mode=mode, dir_fd=descriptors[-1])
                created = True
                _fsync_or_fail(descriptors[-1], f"parent after mkdir {name}")
            except FileExistsError:
                pass
            try:
                child = _open_dir_at(descriptors[-1], name)
            except OSError as exc:
                raise ImmutableArtifactError(
                    f"directory path is absent, non-directory, or symlinked: {absolute}"
                ) from exc
            descriptors.append(child)
            if created:
                _fsync_or_fail(child, f"new directory {name}")
        _verify_directory_chain(tuple(descriptors), names)
    finally:
        _close_descriptors(reversed(descriptors))


def _open_relative_directory(root: Path, parts: tuple[str, ...]) -> int:
    if parts:
        safe_relative_path(PurePosixPath(*parts).as_posix())
    return _open_absolute_directory(Path(root).joinpath(*parts))


def _open_pinned_relative_directory(
    root_descriptor: int, parts: tuple[str, ...]
) -> _PinnedDirectory:
    if parts:
        safe_relative_path(PurePosixPath(*parts).as_posix())
    descriptors = [os.dup(root_descriptor)]
    try:
        for part in parts:
            descriptors.append(_open_dir_at(descriptors[-1], part))
        pinned = _PinnedDirectory(tuple(descriptors), parts)
        pinned.verify_path()
        return pinned
    except BaseException:
        _close_descriptors(reversed(descriptors))
        raise


def _ensure_pinned_relative_directory(
    root_descriptor: int, parts: tuple[str, ...], mode: int = 0o755
) -> _PinnedDirectory:
    if parts:
        safe_relative_path(PurePosixPath(*parts).as_posix())
    descriptors = [os.dup(root_descriptor)]
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=mode, dir_fd=descriptors[-1])
                _fsync_or_fail(descriptors[-1], f"parent after mkdir {part}")
            except FileExistsError:
                pass
            descriptors.append(_open_dir_at(descriptors[-1], part))
        pinned = _PinnedDirectory(tuple(descriptors), parts)
        pinned.verify_path()
        return pinned
    except BaseException:
        _close_descriptors(reversed(descriptors))
        raise


def _open_relative_regular_at(
    root_descriptor: int, relative_path: str
) -> _PinnedRegular:
    relative = safe_relative_path(relative_path)
    directory = _open_pinned_relative_directory(
        root_descriptor, relative.parts[:-1]
    )
    try:
        declared = os.stat(
            relative.parts[-1],
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        directory.close()
        raise ImmutableArtifactError(
            f"cannot inspect anchored artifact {relative_path}: {exc}"
        ) from exc
    if (
        stat.S_ISLNK(declared.st_mode)
        or not stat.S_ISREG(declared.st_mode)
        or declared.st_nlink != 1
    ):
        directory.close()
        raise ImmutableArtifactError(
            f"anchored artifact is not a single-link regular file: {relative_path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | _nofollow_flag()
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(
            relative.parts[-1], flags, dir_fd=directory.descriptor
        )
    except OSError as exc:
        directory.close()
        raise ImmutableArtifactError(
            f"cannot open anchored artifact {relative_path}: {exc}"
        ) from exc
    pinned = _PinnedRegular(
        descriptor, directory.descriptors, directory.names, relative.parts[-1]
    )
    directory.descriptors = ()
    try:
        pinned.verify_path()
    except BaseException:
        pinned.close()
        raise
    return pinned


def _open_relative_regular(root: Path, relative_path: str) -> _PinnedRegular:
    relative = safe_relative_path(relative_path)
    try:
        return _open_regular(Path(root).joinpath(*relative.parts))
    except ImmutableArtifactError as exc:
        raise ImmutableArtifactError(f"cannot open artifact {relative_path}: {exc}") from exc


def read_relative(root: Path, relative_path: str) -> bytes:
    pinned = _open_relative_regular(Path(root), relative_path)
    try:
        payload = _read_fd(pinned.descriptor)
        pinned.verify_path()
        return payload
    finally:
        pinned.close()


@dataclasses.dataclass(frozen=True)
class _PinnedDirectoryReaderBinding:
    root: _PinnedDirectory
    path_monitor: _RetainedDirectoryPathMonitorState


class PinnedDirectoryReader:
    """Read one tree through a single pinned, continuously revalidated root."""

    @_translate_descriptor_resource_errors("pinned directory reader open")
    def __init__(self, root: Path) -> None:
        pinned, monitor = _open_directory_with_early_path_monitor(Path(root))
        self._binding: _PinnedDirectoryReaderBinding | None = (
            _PinnedDirectoryReaderBinding(pinned, monitor)
        )

    def _require_binding(self) -> _PinnedDirectoryReaderBinding:
        binding = self._binding
        if binding is None:
            raise ImmutableArtifactError("pinned directory reader is closed")
        return binding

    @property
    def _root(self) -> _PinnedDirectory:
        return self._require_binding().root

    @property
    def _path_monitor(self) -> _RetainedDirectoryPathMonitorState | None:
        binding = self._binding
        return None if binding is None else binding.path_monitor

    @property
    def descriptor(self) -> int:
        return self._root.descriptor

    def path_chain_nodes(self) -> tuple[tuple[str, int, int, int], ...]:
        """Return the ordered lexical directory chain's stable node identity."""
        self.verify_path()
        labels = (os.path.sep, *self._root.names)
        nodes: list[tuple[str, int, int, int]] = []
        for label, descriptor in zip(
            labels,
            self._root.descriptors,
            strict=True,
        ):
            metadata = os.fstat(descriptor)
            nodes.append((
                label,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
            ))
        self.verify_path()
        return tuple(nodes)

    def verify_path(self) -> None:
        self._root.verify_path()
        _require_retained_directory_path_monitor_quiet(self._path_monitor)
        self._root.verify_path()

    def seal_root_directory_mode(self) -> None:
        """Set the pinned root to 0555 while retaining path-race detection."""
        self.verify_path()
        before = os.fstat(self._root.descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise ImmutableArtifactError(
                "retained directory root is not a directory"
            )
        os.fchmod(self._root.descriptor, 0o555)
        os.fsync(self._root.descriptor)
        after = os.fstat(self._root.descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or stat.S_IFMT(before.st_mode) != stat.S_IFMT(after.st_mode)
            or before.st_nlink != after.st_nlink
            or before.st_uid != after.st_uid
            or before.st_gid != after.st_gid
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or stat.S_IMODE(after.st_mode) != 0o555
        ):
            raise ImmutableArtifactError(
                "retained directory root changed during mode seal"
            )
        self._root.verify_path()
        _require_retained_directory_root_mode_change(self._path_monitor)
        self.verify_path()
        final = _snapshot_fields(os.fstat(self._root.descriptor))
        if final != _snapshot_fields(after):
            raise ImmutableArtifactError(
                "retained directory root changed after mode seal"
            )

    @_translate_descriptor_resource_errors(
        "pinned directory reader rename transition"
    )
    def relocate_noreplace(
        self,
        *,
        source_parent: PinnedDirectoryReader,
        source_name: str,
        destination_parent: PinnedDirectoryReader,
        destination_name: str,
    ) -> None:
        """Atomically move and rebind this retained directory exactly once."""
        for role, name in (
            ("source", source_name),
            ("destination", destination_name),
        ):
            if type(name) is not str:
                raise ImmutableArtifactError(
                    f"retained directory rename {role} name must be a string"
                )
            relative = safe_relative_path(name)
            if len(relative.parts) != 1:
                raise ImmutableArtifactError(
                    f"retained directory rename {role} name is not one component"
                )
        if (
            source_parent is destination_parent
            and source_name == destination_name
        ):
            raise ImmutableArtifactError(
                "retained directory rename destination is unchanged"
            )

        old_binding = self._require_binding()
        old_root = old_binding.root
        old_state = old_binding.path_monitor
        if not old_root.names:
            raise ImmutableArtifactError(
                "retained directory reader cannot accept a root rename"
            )
        self.verify_path()
        source_parent.verify_path()
        destination_parent.verify_path()
        retained_identity = _node_identity(os.fstat(old_root.descriptor))
        source_before = os.stat(
            source_name,
            dir_fd=source_parent.descriptor,
            follow_symlinks=False,
        )
        if _node_identity(source_before) != retained_identity:
            raise ImmutableArtifactError(
                "retained directory rename source identity changed"
            )
        if (
            os.fstat(source_parent.descriptor).st_dev
            != os.fstat(destination_parent.descriptor).st_dev
        ):
            raise ImmutableArtifactError(
                "CROSS_DEVICE_RENAME: retained directory parents differ"
            )

        rename_state = _arm_directory_rename_monitor(
            source_parent.descriptor,
            source_name,
            destination_parent.descriptor,
            destination_name,
            old_root.descriptor,
        )
        replacement_root: _PinnedDirectory | None = None
        replacement_state: _RetainedDirectoryPathMonitorState | None = None
        replacement_binding: _PinnedDirectoryReaderBinding | None = None
        old_closed = False
        rename_attempted = False
        outcome_not_committed = False
        commit_authenticated = False
        rename_error: BaseException | None = None
        try:
            self.verify_path()
            source_parent.verify_path()
            destination_parent.verify_path()
            source_armed = os.stat(
                source_name,
                dir_fd=source_parent.descriptor,
                follow_symlinks=False,
            )
            if _node_identity(source_armed) != retained_identity:
                raise ImmutableArtifactError(
                    "retained directory rename source changed while arming"
                )
            _require_directory_rename_monitor_quiet(rename_state)

            rename_attempted = True
            try:
                _rename_noreplace_at(
                    source_parent.descriptor,
                    source_name,
                    destination_parent.descriptor,
                    destination_name,
                )
            except BaseException as exc:
                rename_error = exc

            try:
                source_after = _entry_metadata(
                    source_parent.descriptor, source_name,
                )
                destination_after = _entry_metadata(
                    destination_parent.descriptor, destination_name,
                )
            except BaseException as exc:
                raise RenameOutcomeUnproven(exc) from None
            source_matches = (
                source_after is not None
                and _node_identity(source_after) == retained_identity
            )
            destination_matches = (
                destination_after is not None
                and _node_identity(destination_after) == retained_identity
            )
            if source_matches and not destination_matches:
                try:
                    _require_directory_rename_monitor_quiet(rename_state)
                    self.verify_path()
                    source_parent.verify_path()
                    destination_parent.verify_path()
                    source_final = _entry_metadata(
                        source_parent.descriptor, source_name,
                    )
                    destination_final = _entry_metadata(
                        destination_parent.descriptor, destination_name,
                    )
                    if (
                        source_final is None
                        or _node_identity(source_final) != retained_identity
                        or (
                            destination_final is not None
                            and _node_identity(destination_final)
                            == retained_identity
                        )
                    ):
                        raise ImmutableArtifactError(
                            "retained directory rename noncommit outcome changed"
                        )
                    rename_state.monitor.set_close_verifier(
                        lambda: _require_directory_rename_monitor_quiet(
                            rename_state
                        )
                    )
                    rename_state.monitor.close()
                except BaseException as exc:
                    raise RenameOutcomeUnproven(exc) from None
                outcome_not_committed = True
                if rename_error is not None:
                    raise rename_error
                raise ImmutableArtifactError(
                    "retained directory rename reported success without committing"
                )
            if not destination_matches or source_after is not None:
                error = (
                    rename_error
                    if rename_error is not None
                    else ImmutableArtifactError(
                        "retained directory rename outcome is inconsistent"
                    )
                )
                raise RenameOutcomeUnproven(error) from None

            destination_parent_identity = _node_identity(
                os.fstat(destination_parent.descriptor)
            )
            destination_watches = []
            for index, descriptor in enumerate(old_root.descriptors):
                if _node_identity(os.fstat(descriptor)) != destination_parent_identity:
                    continue
                watch = (
                    old_state.protected_names[index][0]
                    if index < len(old_state.protected_names)
                    else old_state.root_watch
                )
                destination_watches.append(watch)
            if len(set(destination_watches)) > 1:
                raise RenameOutcomeUnproven(
                    ImmutableArtifactError(
                        "renamed retained directory destination parent is ambiguous"
                    )
                ) from None
            destination_entry = (
                (destination_watches[0], os.fsencode(destination_name))
                if destination_watches
                else None
            )

            try:
                _require_directory_rename_monitor_complete(rename_state)
            except BaseException as exc:
                raise RenameOutcomeUnproven(exc) from None
            commit_authenticated = True
            _require_retained_directory_root_rename(
                old_state,
                destination_entry=destination_entry,
            )

            destination_path = Path(os.path.sep).joinpath(
                *destination_parent._root.names,
                destination_name,
            )
            replacement_root, replacement_state = (
                _open_directory_with_early_path_monitor(destination_path)
            )
            if (
                _node_identity(os.fstat(replacement_root.descriptor))
                != retained_identity
            ):
                raise ImmutableArtifactError(
                    "renamed retained directory destination identity changed"
                )
            replacement_binding = _PinnedDirectoryReaderBinding(
                replacement_root,
                replacement_state,
            )
            replacement_root.verify_path()
            _require_retained_directory_path_monitor_quiet(replacement_state)
            replacement_root.verify_path()
            source_parent.verify_path()
            destination_parent.verify_path()

            rename_state.monitor.set_close_verifier(
                lambda: _require_directory_rename_monitor_quiet(rename_state)
            )
            self._binding = replacement_binding
            old_state.monitor.cancel_close_verification()
            _close_callables(
                (old_state.monitor.close, old_root.close),
                label="renamed pinned directory reader transition",
            )
            old_closed = True
            rename_state.monitor.close()
            self.verify_path()
            if rename_error is not None and not isinstance(rename_error, Exception):
                raise CommittedRenameInterruption(
                    rename_error,
                    handoff_complete=True,
                ) from rename_error
        except RenameOutcomeUnproven:
            raise
        except CommittedRenameInterruption:
            raise
        except BaseException as exc:
            binding_transferred = (
                replacement_binding is not None
                and self._binding is replacement_binding
            )
            if commit_authenticated:
                if not isinstance(exc, Exception):
                    raise CommittedRenameInterruption(
                        exc,
                        handoff_complete=binding_transferred,
                    ) from exc
                raise ImmutableArtifactError(
                    f"RENAME_HANDOFF_FAILED: {exc}"
                ) from exc
            if rename_attempted and not outcome_not_committed:
                raise RenameOutcomeUnproven(exc) from None
            raise
        finally:
            binding_transferred = (
                replacement_binding is not None
                and self._binding is replacement_binding
            )
            if rename_state.monitor.descriptor >= 0:
                if not binding_transferred:
                    rename_state.monitor.cancel_close_verification()
                _close_callables(
                    (rename_state.monitor.close,),
                    label="retained directory rename monitor cleanup",
                )
            if binding_transferred and not old_closed:
                old_state.monitor.cancel_close_verification()
                _close_callables(
                    (old_state.monitor.close, old_root.close),
                    label="interrupted pinned directory reader transition",
                )
            if not binding_transferred and replacement_root is not None:
                closers: list[Callable[[], None]] = []
                if replacement_state is not None:
                    replacement_state.monitor.cancel_close_verification()
                    closers.append(replacement_state.monitor.close)
                closers.append(replacement_root.close)
                _close_callables(
                    closers,
                    label="renamed pinned directory replacement cleanup",
                )

    @_translate_descriptor_resource_errors("pinned regular-file snapshot read")
    def read_snapshot(
        self,
        relative_path: str,
        *,
        max_bytes: int | None = None,
        expected_metadata: MetadataSnapshot | None = None,
    ) -> PinnedRegularSnapshot:
        self.verify_path()
        pinned = _open_relative_regular_at(self._root.descriptor, relative_path)
        try:
            if (
                expected_metadata is not None
                and _snapshot_fields(os.fstat(pinned.descriptor))
                != expected_metadata
            ):
                raise ImmutableArtifactError(
                    f"pinned regular-file generation changed: {relative_path}"
                )
            if max_bytes is None:
                payload, metadata = _read_fd_snapshot(pinned.descriptor)
            else:
                payload, metadata = _read_fd_snapshot(
                    pinned.descriptor, max_bytes=max_bytes,
                )
            pinned.verify_path()
            if (
                expected_metadata is not None
                and metadata != expected_metadata
            ):
                raise ImmutableArtifactError(
                    f"pinned regular-file generation changed: {relative_path}"
                )
            if _snapshot_fields(os.fstat(pinned.descriptor)) != metadata:
                raise ImmutableArtifactError(
                    "regular file changed after snapshot read"
                )
            self.verify_path()
            return PinnedRegularSnapshot(payload, metadata)
        finally:
            pinned.close()

    def read(
        self, relative_path: str, *, max_bytes: int | None = None,
    ) -> bytes:
        return self.read_snapshot(relative_path, max_bytes=max_bytes).payload

    @_translate_descriptor_resource_errors("pinned regular-file verification")
    def verify_regular(
        self, relative_path: str, expected: MetadataSnapshot,
    ) -> None:
        self.verify_path()
        pinned = _open_relative_regular_at(self._root.descriptor, relative_path)
        try:
            if _snapshot_fields(os.fstat(pinned.descriptor)) != expected:
                raise ImmutableArtifactError(
                    f"pinned regular-file generation changed: {relative_path}"
                )
            pinned.verify_path()
            if _snapshot_fields(os.fstat(pinned.descriptor)) != expected:
                raise ImmutableArtifactError(
                    f"pinned regular-file generation changed: {relative_path}"
                )
            self.verify_path()
        finally:
            pinned.close()

    @_translate_descriptor_resource_errors("pinned directory tree snapshot")
    def snapshot_tree(
        self,
        *,
        require_readonly_files: bool = False,
        require_readonly_directories: bool = False,
        max_files: int | None = None,
        max_directories: int | None = None,
        max_total_file_bytes: int | None = None,
        max_total_relative_path_bytes: int | None = None,
    ) -> PinnedTreeSnapshot:
        self.verify_path()
        snapshot = _snapshot_pinned_tree_at(
            self._root.descriptor,
            require_readonly_files=require_readonly_files,
            require_readonly_directories=require_readonly_directories,
            max_files=max_files,
            max_directories=max_directories,
            max_total_file_bytes=max_total_file_bytes,
            max_total_relative_path_bytes=max_total_relative_path_bytes,
        )
        self.verify_path()
        return snapshot

    @contextlib.contextmanager
    def retain_tree_generation(
        self,
        *,
        require_readonly_files: bool = False,
        require_readonly_directories: bool = False,
        max_files: int | None = None,
        max_directories: int | None = None,
        max_total_file_bytes: int | None = None,
        max_total_relative_path_bytes: int | None = None,
    ) -> Iterable[PinnedTreeSnapshot]:
        """Hold one monitored tree generation across caller-owned reads."""
        limits = {
            "max_files": max_files,
            "max_directories": max_directories,
            "max_total_file_bytes": max_total_file_bytes,
            "max_total_relative_path_bytes": max_total_relative_path_bytes,
        }
        monitor = _arm_directory_mutation_monitor(
            self._root.descriptor, **limits,
        )
        try:
            initial = self.snapshot_tree(
                require_readonly_files=require_readonly_files,
                require_readonly_directories=require_readonly_directories,
                **limits,
            )
            monitor.require_quiet()
            yield initial
            monitor.require_quiet()
            final = self.snapshot_tree(
                require_readonly_files=require_readonly_files,
                require_readonly_directories=require_readonly_directories,
                **limits,
            )
            if final != initial:
                raise ImmutableArtifactError(
                    "retained tree generation changed during use"
                )
            self.verify_path()
            monitor.require_quiet()
        except BaseException:
            monitor.cancel_close_verification()
            raise
        finally:
            monitor.close()

    def close(self) -> None:
        binding = self._binding
        self._binding = None
        if binding is None:
            return
        _close_callables(
            (binding.path_monitor.monitor.close, binding.root.close),
            label="pinned directory reader close",
        )

    def __enter__(self) -> "PinnedDirectoryReader":
        self.verify_path()
        return self

    def __exit__(self, _type: object, value: object, _traceback: object) -> bool:
        if value is not None:
            try:
                self.verify_path()
            except BaseException as verification_error:
                try:
                    value.add_note(
                        "pinned directory reader revalidation failed: "
                        f"{verification_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            try:
                self.close()
            except BaseException as close_error:
                try:
                    value.add_note(
                        f"pinned directory reader close failed: {close_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            return False
        try:
            self.verify_path()
        finally:
            self.close()
        return False


def write_regular_bytes_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o444,
) -> None:
    """Create one immutable regular leaf through an already-pinned parent.

    The destination is exclusive and never recovered or replaced. Failures leave
    uncertain bytes in the caller-owned staging directory for later inspection.
    """
    if type(parent_descriptor) is not int or parent_descriptor < 0:
        raise ImmutableArtifactError(
            "descriptor-bound output parent must be a nonnegative integer"
        )
    _validate_publication_name(name)
    if type(payload) is not bytes:
        raise ImmutableArtifactError("descriptor-bound output payload must be bytes")
    if type(mode) is not int or mode < 0 or mode & ~0o777 or mode & 0o222:
        raise ImmutableArtifactError(
            "descriptor-bound output mode must be a read-only permission mode"
        )
    parent_before = os.fstat(parent_descriptor)
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ImmutableArtifactError(
            "descriptor-bound output parent is not a directory"
        )

    descriptor = -1
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | _nofollow_flag(),
                0o400,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as exc:
            raise ImmutableArtifactError(
                f"descriptor-bound output already exists: {name}"
            ) from exc
        _write_all(descriptor, memoryview(payload))
        os.fchmod(descriptor, mode)
        _fsync_or_fail(descriptor, f"descriptor-bound output {name}")
        opened = os.fstat(descriptor)
        linked = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or opened.st_nlink != 1
            or linked.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != mode
            or _snapshot_fields(linked) != _snapshot_fields(opened)
        ):
            raise ImmutableArtifactError(
                f"descriptor-bound output path changed: {name}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed, snapshot = _read_fd_snapshot(descriptor)
        if observed != payload or snapshot != _snapshot_fields(opened):
            raise ImmutableArtifactError(
                f"descriptor-bound output bytes changed: {name}"
            )
        linked_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _snapshot_fields(linked_after) != snapshot:
            raise ImmutableArtifactError(
                f"descriptor-bound output changed during closure: {name}"
            )
        _fsync_or_fail(parent_descriptor, f"descriptor-bound output parent {name}")
        parent_after = os.fstat(parent_descriptor)
        if _node_identity(parent_after) != _node_identity(parent_before):
            raise ImmutableArtifactError(
                "descriptor-bound output parent identity changed"
            )
    finally:
        if descriptor >= 0:
            _close_descriptors(
                (descriptor,), label=f"descriptor-bound output {name} close"
            )


def replace_regular_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    """Durably replace one regular leaf without following any path symlink."""
    if not isinstance(payload, bytes):
        raise ImmutableArtifactError("replacement payload must be bytes")
    if type(mode) is not int or mode < 0 or mode & ~0o777:
        raise ImmutableArtifactError("replacement mode is invalid")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.parent == absolute:
        raise ImmutableArtifactError("replacement path has no regular leaf")
    safe_relative_path(absolute.name)
    if len(os.fsencode(absolute.name)) > _MAX_FINAL_NAME_BYTES:
        raise ImmutableArtifactError("replacement leaf name exceeds bounded limits")

    parent = _open_pinned_directory(absolute.parent)
    descriptor = -1
    temporary_name: str | None = None
    published = False
    try:
        parent.verify_path()
        try:
            existing = os.stat(
                absolute.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise ImmutableArtifactError(
                f"cannot inspect replacement destination {absolute}: {exc}"
            ) from exc
        if existing is not None and (
            stat.S_ISLNK(existing.st_mode)
            or not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
        ):
            raise ImmutableArtifactError(
                f"replacement destination is not a single-link regular file: {absolute}"
            )

        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | _nofollow_flag()
        )
        for _attempt in range(32):
            token = os.urandom(12).hex()
            candidate = f".{absolute.name}.{token}.partial"
            if len(os.fsencode(candidate)) > _MAX_FINAL_NAME_BYTES:
                candidate = f".summary-{token}.partial"
            try:
                descriptor = os.open(
                    candidate,
                    flags,
                    mode,
                    dir_fd=parent.descriptor,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
            except OSError as exc:
                raise ImmutableArtifactError(
                    f"cannot create replacement temporary for {absolute}: {exc}"
                ) from exc
        if descriptor < 0 or temporary_name is None:
            raise ImmutableArtifactError(
                f"cannot allocate unique replacement temporary for {absolute}"
            )

        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ImmutableArtifactError("replacement temporary is not a regular file")
        _write_all(descriptor, memoryview(payload))
        os.fchmod(descriptor, mode)
        _fsync_or_fail(descriptor, f"replacement temporary {absolute.name}")
        written = os.fstat(descriptor)
        if written.st_size != len(payload) or written.st_nlink != 1:
            raise ImmutableArtifactError("replacement temporary changed during write")
        linked = os.stat(
            temporary_name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if _node_identity(linked) != _node_identity(written):
            raise ImmutableArtifactError("replacement temporary path changed")

        os.replace(
            temporary_name,
            absolute.name,
            src_dir_fd=parent.descriptor,
            dst_dir_fd=parent.descriptor,
        )
        published = True
        _fsync_or_fail(parent.descriptor, f"replacement parent {absolute.parent}")
        named = os.stat(
            absolute.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        retained = os.fstat(descriptor)
        named_snapshot = _snapshot_fields(named)
        retained_snapshot = _snapshot_fields(retained)
        if (
            not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or _node_identity(named) != _node_identity(retained)
            or named.st_size != len(payload)
            or stat.S_IMODE(named.st_mode) != mode
        ):
            raise ImmutableArtifactError("published replacement path changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, size, observed_mode = _hash_fd(descriptor)
        if (
            digest != hashlib.sha256(payload).hexdigest()
            or size != len(payload)
            or stat.S_IMODE(observed_mode) != mode
        ):
            raise ImmutableArtifactError("published replacement bytes changed")
        _fsync_or_fail(descriptor, f"published replacement {absolute.name}")
        closed_named = os.stat(
            absolute.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if (
            _snapshot_fields(closed_named) != named_snapshot
            or _snapshot_fields(os.fstat(descriptor)) != retained_snapshot
            or _node_identity(closed_named)
            != _node_identity(os.fstat(descriptor))
        ):
            raise ImmutableArtifactError(
                "published replacement changed during content closure"
            )
        parent.verify_path()
    finally:
        active_error = sys.exc_info()[1]
        if published and active_error is not None and descriptor >= 0:
            try:
                retired_name = _retire_failed_publication_at(
                    parent.descriptor,
                    absolute.name,
                    descriptor,
                    directory=False,
                )
            except BaseException as retirement_error:
                active_error.add_note(
                    "failed replacement could not leave the active namespace: "
                    f"{retirement_error}"
                )
            else:
                active_error.add_note(
                    "failed replacement preserved outside the active namespace "
                    f"as {retired_name}"
                )
        if descriptor >= 0:
            if temporary_name is not None and not published:
                try:
                    linked = os.stat(
                        temporary_name,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                    opened = os.fstat(descriptor)
                    if _node_identity(linked) == _node_identity(opened):
                        os.unlink(temporary_name, dir_fd=parent.descriptor)
                        _fsync_or_fail(
                            parent.descriptor,
                            f"replacement temporary cleanup {absolute.name}",
                        )
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    if active_error is not None:
                        try:
                            active_error.add_note(
                                f"replacement temporary cleanup failed: {exc}"
                            )
                        except (AttributeError, TypeError):
                            pass
                    else:
                        active_error = exc
            _close_descriptors(
                (descriptor,),
                active_error=active_error,
                label="replacement temporary close",
            )
        try:
            parent.close()
        except BaseException as exc:
            if active_error is not None:
                try:
                    active_error.add_note(f"replacement parent close failed: {exc}")
                except (AttributeError, TypeError):
                    pass
            else:
                active_error = exc
        if sys.exc_info()[1] is None and active_error is not None:
            raise active_error


def retire_regular_file(
    path: Path,
    *,
    expected_payload: bytes | None = None,
    expected_snapshot: MetadataSnapshot | None = None,
) -> Path | None:
    """Atomically move one stale regular leaf to a hidden retained name.

    When ``expected_payload`` is supplied, the opened source must carry those
    exact bytes. When ``expected_snapshot`` is supplied, it must also be the
    exact seven-field file generation previously retained by the caller. A
    source-name race detected after rename is restored when the canonical name
    is still free, so a concurrent replacement is preserved rather than
    silently retired as the stale generation.
    """
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.parent == absolute:
        raise ImmutableArtifactError("retirement path has no regular leaf")
    safe_relative_path(absolute.name)
    parent = _open_pinned_directory(absolute.parent)
    pinned: _PinnedRegular | None = None
    try:
        parent.verify_path()
        try:
            present = os.stat(
                absolute.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if (
            stat.S_ISLNK(present.st_mode)
            or not stat.S_ISREG(present.st_mode)
            or present.st_nlink != 1
        ):
            raise ImmutableArtifactError(
                f"retirement source is not a single-link regular file: {absolute}"
            )
        pinned = _open_relative_regular_at(parent.descriptor, absolute.name)
        opened = os.fstat(pinned.descriptor)
        pinned.verify_path()
        if expected_snapshot is not None:
            if (
                type(expected_snapshot) is not tuple
                or len(expected_snapshot) != 7
                or any(type(value) is not int for value in expected_snapshot)
            ):
                raise ImmutableArtifactError(
                    "retirement expected snapshot must be seven integers"
                )
            if _snapshot_fields(opened) != expected_snapshot:
                raise ImmutableArtifactError(
                    f"retirement source generation changed: {absolute}"
                )
        if expected_payload is not None:
            if type(expected_payload) is not bytes:
                raise ImmutableArtifactError(
                    "retirement expected payload must be bytes"
                )
            os.lseek(pinned.descriptor, 0, os.SEEK_SET)
            observed_payload, observed_snapshot = _read_fd_snapshot(
                pinned.descriptor,
                max_bytes=len(expected_payload),
            )
            if (
                observed_payload != expected_payload
                or observed_snapshot != _snapshot_fields(opened)
                or _snapshot_fields(os.fstat(pinned.descriptor))
                != observed_snapshot
            ):
                raise ImmutableArtifactError(
                    f"retirement source payload changed: {absolute}"
                )
            pinned.verify_path()
        retired_name: str | None = None
        for _attempt in range(32):
            token = os.urandom(12).hex()
            candidate = f".{absolute.name}.{token}.retired"
            if len(os.fsencode(candidate)) > _MAX_FINAL_NAME_BYTES:
                candidate = f".artifact-{token}.retired"
            try:
                _rename_noreplace_at(
                    parent.descriptor,
                    absolute.name,
                    parent.descriptor,
                    candidate,
                )
                retired_name = candidate
                break
            except FileExistsError:
                continue
        if retired_name is None:
            raise ImmutableArtifactError(
                f"cannot allocate unique retirement name for {absolute}"
            )
        def restore_retired_entry(
            error: ImmutableArtifactError,
            *,
            retired_name: str,
            retired_snapshot: os.stat_result,
        ) -> None:
            restored_to_canonical = False
            try:
                _rename_noreplace_at(
                    parent.descriptor,
                    retired_name,
                    parent.descriptor,
                    absolute.name,
                )
                restored_to_canonical = True
                restored = os.stat(
                    absolute.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_IFMT(restored.st_mode)
                    != stat.S_IFMT(retired_snapshot.st_mode)
                    or restored.st_nlink != retired_snapshot.st_nlink
                    or _node_identity(restored)
                    != _node_identity(retired_snapshot)
                    or restored.st_size != retired_snapshot.st_size
                ):
                    raise ImmutableArtifactError(
                        "retirement race restoration changed identity"
                    )
                _fsync_or_fail(
                    parent.descriptor,
                    f"retirement race restoration {absolute.parent}",
                )
                parent.verify_path()
                error.add_note(
                    "concurrent retirement-source replacement was restored "
                    f"to {absolute}"
                )
            except BaseException as restore_error:
                if restored_to_canonical:
                    error.add_note(
                        "concurrent retirement-source replacement was restored "
                        "to the canonical name but restoration durability "
                        f"verification failed: {restore_error}"
                    )
                else:
                    error.add_note(
                        "concurrent retirement-source replacement remains "
                        f"retained as {absolute.with_name(retired_name)}: "
                        f"{restore_error}"
                    )
            raise error

        def inspect_retired_entry() -> os.stat_result:
            retired_snapshot = os.stat(
                retired_name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(retired_snapshot.st_mode)
                or retired_snapshot.st_nlink != 1
                or _node_identity(retired_snapshot) != _node_identity(opened)
                or retired_snapshot.st_size != opened.st_size
            ):
                restore_retired_entry(
                    ImmutableArtifactError(
                        "retired regular-file identity changed"
                    ),
                    retired_name=retired_name,
                    retired_snapshot=retired_snapshot,
                )
            if expected_payload is not None:
                before_read = os.fstat(pinned.descriptor)
                os.lseek(pinned.descriptor, 0, os.SEEK_SET)
                retired_payload, retired_read_snapshot = _read_fd_snapshot(
                    pinned.descriptor,
                    max_bytes=len(expected_payload),
                )
                after_read = os.fstat(pinned.descriptor)
                linked_after_read = os.stat(
                    retired_name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if (
                    retired_payload != expected_payload
                    or retired_read_snapshot != _snapshot_fields(before_read)
                    or _snapshot_fields(after_read) != retired_read_snapshot
                    or _snapshot_fields(linked_after_read)
                    != retired_read_snapshot
                ):
                    restore_retired_entry(
                        ImmutableArtifactError(
                            "retired regular-file payload changed"
                        ),
                        retired_name=retired_name,
                        retired_snapshot=linked_after_read,
                    )
                retired_snapshot = linked_after_read
            return retired_snapshot

        def salvage_unlinked_retirement() -> Path:
            retained = os.fstat(pinned.descriptor)
            retained_snapshot = _snapshot_fields(retained)
            if (
                not stat.S_ISREG(retained.st_mode)
                or retained_snapshot[0] != _snapshot_fields(opened)[0]
                or retained_snapshot[1] != _snapshot_fields(opened)[1]
                or retained_snapshot[2] != _snapshot_fields(opened)[2]
                or retained_snapshot[4] != _snapshot_fields(opened)[4]
                or retained_snapshot[5] != _snapshot_fields(opened)[5]
            ):
                raise ImmutableArtifactError(
                    "unlinked retirement source generation changed"
                )
            if expected_payload is not None:
                os.lseek(pinned.descriptor, 0, os.SEEK_SET)
                authenticated_payload, authenticated_snapshot = (
                    _read_fd_snapshot(
                        pinned.descriptor,
                        allow_hardlinks=True,
                        max_bytes=len(expected_payload),
                    )
                )
                if (
                    authenticated_payload != expected_payload
                    or authenticated_snapshot != retained_snapshot
                    or _snapshot_fields(os.fstat(pinned.descriptor))
                    != retained_snapshot
                ):
                    raise ImmutableArtifactError(
                        "unlinked retirement source payload changed"
                    )
            salvaged_name = _salvage_failed_regular_publication_at(
                parent.descriptor,
                pinned.descriptor,
            )
            return absolute.with_name(salvaged_name)

        # Identify and, when necessary, restore the exact directory entry before
        # the first post-rename durability operation.  A durability failure must
        # never strand a concurrent replacement under the hidden retirement name.
        try:
            inspect_retired_entry()
        except FileNotFoundError:
            return salvage_unlinked_retirement()
        try:
            _fsync_or_fail(
                parent.descriptor, f"retirement parent {absolute.parent}"
            )
        except BaseException as durability_error:
            try:
                os.stat(
                    retired_name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    salvaged = salvage_unlinked_retirement()
                except BaseException as salvage_error:
                    durability_error.add_note(
                        "unlinked retirement evidence salvage failed: "
                        f"{salvage_error}"
                    )
                else:
                    durability_error.add_note(
                        "unlinked retirement evidence was durably salvaged as "
                        f"{salvaged}"
                    )
            raise
        # Reauthenticate after the durability boundary so a same-inode overwrite
        # during rename/fsync cannot count as successful expected-payload cleanup.
        try:
            inspect_retired_entry()
        except FileNotFoundError:
            return salvage_unlinked_retirement()
        try:
            os.stat(
                absolute.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ImmutableArtifactError("retirement source name remains present")
        parent.verify_path()
        return absolute.with_name(retired_name)
    finally:
        active_error = sys.exc_info()[1]
        if pinned is not None:
            try:
                pinned.close()
            except BaseException as exc:
                if active_error is not None:
                    try:
                        active_error.add_note(f"retained retirement close failed: {exc}")
                    except (AttributeError, TypeError):
                        pass
                else:
                    active_error = exc
        try:
            parent.close()
        except BaseException as exc:
            if active_error is not None:
                try:
                    active_error.add_note(f"retirement parent close failed: {exc}")
                except (AttributeError, TypeError):
                    pass
            else:
                active_error = exc
        if sys.exc_info()[1] is None and active_error is not None:
            raise active_error


def salvage_retained_regular_descriptor(
    path: Path,
    descriptor: int,
    *,
    expected_payload: bytes,
    expected_snapshot: MetadataSnapshot,
) -> Path:
    """Preserve one retained regular inode after its locator was replaced.

    Link count and ctime may change when the canonical name is replaced. Every
    other generation field plus the exact bytes must still match the snapshot
    captured while this descriptor owned the canonical locator. The evidence is
    copied to a hidden durable leaf without reading from or changing the active
    name.
    """
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.parent == absolute:
        raise ImmutableArtifactError("salvage path has no regular leaf")
    safe_relative_path(absolute.name)
    if type(descriptor) is not int or descriptor < 0:
        raise ImmutableArtifactError(
            "salvage retained descriptor must be a nonnegative integer"
        )
    if type(expected_payload) is not bytes:
        raise ImmutableArtifactError("salvage expected payload must be bytes")
    if (
        type(expected_snapshot) is not tuple
        or len(expected_snapshot) != 7
        or any(type(value) is not int for value in expected_snapshot)
    ):
        raise ImmutableArtifactError(
            "salvage expected snapshot must be seven integers"
        )

    parent = _open_pinned_directory(absolute.parent)
    try:
        parent.verify_path()
        before = os.fstat(descriptor)
        current_snapshot = _snapshot_fields(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or current_snapshot[0] != expected_snapshot[0]
            or current_snapshot[1] != expected_snapshot[1]
            or current_snapshot[2] != expected_snapshot[2]
            or current_snapshot[4] != expected_snapshot[4]
            or current_snapshot[5] != expected_snapshot[5]
        ):
            raise ImmutableArtifactError(
                "salvage retained regular-file generation changed"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed_payload, observed_snapshot = _read_fd_snapshot(
            descriptor,
            allow_hardlinks=True,
            max_bytes=len(expected_payload),
        )
        if (
            observed_payload != expected_payload
            or observed_snapshot != current_snapshot
            or _snapshot_fields(os.fstat(descriptor)) != current_snapshot
        ):
            raise ImmutableArtifactError(
                "salvage retained regular-file payload changed"
            )
        retained_name = _salvage_failed_regular_publication_at(
            parent.descriptor,
            descriptor,
        )
        parent.verify_path()
        return absolute.with_name(retained_name)
    finally:
        parent.close()


def sha256_relative(root: Path, relative_path: str) -> tuple[str, int]:
    pinned = _open_relative_regular(Path(root), relative_path)
    try:
        digest, size, _ = _hash_fd(pinned.descriptor)
        pinned.verify_path()
        return digest, size
    finally:
        pinned.close()


def _write_all(descriptor: int, payload: memoryview) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise ImmutableArtifactError("short immutable-artifact write")
        offset += written


def _copy_to_descriptor(file: ImmutableFile, descriptor: int) -> None:
    digest = hashlib.sha256()
    size = 0
    if file.payload is not None:
        _write_all(descriptor, memoryview(file.payload))
        digest.update(file.payload)
        size = len(file.payload)
    else:
        assert file.source_path is not None
        source = _open_regular(
            file.source_path, allow_hardlinks=file.allow_hardlinks
        )
        try:
            def copy_block(block: bytes) -> None:
                _write_all(descriptor, memoryview(block))
                digest.update(block)

            size, _, _snapshot = _consume_stable_regular(
                source.descriptor,
                copy_block,
                allow_hardlinks=file.allow_hardlinks,
            )
            source.verify_path()
        finally:
            source.close()
    if size != file.size_bytes or digest.hexdigest() != file.sha256:
        raise ImmutableArtifactError(f"source changed while publishing {file.relative_path}")


def _entry_mode(parent_descriptor: int, name: str) -> int | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return None


def _entry_metadata(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _rename_noreplace_at(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ImmutableArtifactError("atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination_name)
    if error == errno.EXDEV:
        raise ImmutableArtifactError(
            "CROSS_DEVICE_RENAME: atomic no-replace rename crossed filesystems"
        )
    unavailable_errors = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
        getattr(errno, "ENOTSUP", errno.ENOSYS),
    }
    if error in unavailable_errors:
        raise ImmutableArtifactError(
            "atomic RENAME_NOREPLACE is unavailable: "
            f"[{error}] {os.strerror(error)}"
        )
    raise ImmutableArtifactError(
        f"atomic no-replace rename failed: [{error}] {os.strerror(error)}"
    )


def _retire_failed_publication_at(
    parent_descriptor: int,
    active_name: str,
    retained_descriptor: int,
    *,
    directory: bool,
) -> str:
    """Move one failed committed generation out of its active namespace."""
    linked = os.stat(
        active_name, dir_fd=parent_descriptor, follow_symlinks=False,
    )
    opened = os.fstat(retained_descriptor)
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        stat.S_ISLNK(linked.st_mode)
        or not expected_type(linked.st_mode)
        or not expected_type(opened.st_mode)
        or _node_identity(linked) != _node_identity(opened)
    ):
        raise ImmutableArtifactError(
            f"failed publication path cannot be retired safely: {active_name}"
        )
    kind = "directory" if directory else "regular"
    for _attempt in range(32):
        retired_name = f".failed-{kind}-{os.urandom(12).hex()}"
        try:
            _rename_noreplace_at(
                parent_descriptor,
                active_name,
                parent_descriptor,
                retired_name,
            )
        except FileExistsError:
            continue
        retired = os.stat(
            retired_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _entry_mode(parent_descriptor, active_name) is not None
            or _node_identity(retired) != _node_identity(opened)
        ):
            raise ImmutableArtifactError(
                f"failed publication retirement changed identity: {active_name}"
            )
        _fsync_or_fail(
            parent_descriptor, f"failed publication retirement {active_name}",
        )
        try:
            closed = os.stat(
                retired_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ImmutableArtifactError(
                "failed publication hidden retirement entry was unlinked"
            ) from exc
        if (
            _entry_mode(parent_descriptor, active_name) is not None
            or _node_identity(closed) != _node_identity(opened)
        ):
            raise ImmutableArtifactError(
                f"failed publication retirement changed after durability: "
                f"{active_name}"
            )
        return retired_name
    raise ImmutableArtifactResourceError(
        f"{VERIFY_RESOURCE_BOUND}: cannot allocate failed-publication name"
    )


def _salvage_failed_regular_publication_at(
    parent_descriptor: int,
    retained_descriptor: int,
) -> str:
    """Copy an unlinked failed generation to a hidden durable evidence leaf."""
    opened = os.fstat(retained_descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise ImmutableArtifactError(
            "failed regular publication descriptor is not regular"
        )

    destination_descriptor = -1
    temporary_name: str | None = None
    retired_name: str | None = None
    committed = False
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | _nofollow_flag()
        )
        for _attempt in range(32):
            token = os.urandom(12).hex()
            candidate = f".failed-regular-{token}"
            temporary = f"{candidate}.salvaging"
            try:
                destination_descriptor = os.open(
                    temporary,
                    flags,
                    0o400,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            retired_name = candidate
            temporary_name = temporary
            break
        if destination_descriptor < 0 or temporary_name is None or retired_name is None:
            raise ImmutableArtifactResourceError(
                f"{VERIFY_RESOURCE_BOUND}: cannot allocate failed-publication salvage"
            )

        digest = hashlib.sha256()

        def copy_block(block: bytes) -> None:
            digest.update(block)
            _write_all(destination_descriptor, memoryview(block))

        os.lseek(retained_descriptor, 0, os.SEEK_SET)
        size_bytes, _mode, source_snapshot = _consume_stable_regular(
            retained_descriptor,
            copy_block,
            allow_hardlinks=True,
        )
        if _snapshot_fields(os.fstat(retained_descriptor)) != source_snapshot:
            raise ImmutableArtifactError(
                "failed regular publication changed during salvage"
            )
        os.fchmod(destination_descriptor, 0o444)
        _fsync_or_fail(
            destination_descriptor, "failed regular publication salvage"
        )
        os.lseek(destination_descriptor, 0, os.SEEK_SET)
        copied_digest, copied_size, copied_mode = _hash_fd(
            destination_descriptor
        )
        if (
            copied_digest != digest.hexdigest()
            or copied_size != size_bytes
            or copied_mode & 0o222
        ):
            raise ImmutableArtifactError(
                "failed regular publication salvage bytes changed"
            )
        _fsync_or_fail(
            parent_descriptor,
            "failed regular publication salvage parent before commit",
        )
        _rename_noreplace_at(
            parent_descriptor,
            temporary_name,
            parent_descriptor,
            retired_name,
        )
        committed = True
        _fsync_or_fail(
            parent_descriptor,
            "failed regular publication salvage parent after commit",
        )
        linked = os.stat(
            retired_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_nlink != 1
            or _node_identity(linked)
            != _node_identity(os.fstat(destination_descriptor))
        ):
            raise ImmutableArtifactError(
                "failed regular publication salvage identity changed"
            )
        return retired_name
    except BaseException as exc:
        if committed and retired_name is not None:
            exc.add_note(
                "failed regular publication salvage remains outside the active "
                f"namespace as {retired_name}"
            )
        elif temporary_name is not None:
            exc.add_note(
                "incomplete failed regular publication salvage remains outside "
                f"the active namespace as {temporary_name}"
            )
        raise
    finally:
        if destination_descriptor >= 0:
            _close_descriptors(
                (destination_descriptor,),
                active_error=sys.exc_info()[1],
                label="failed regular publication salvage close",
            )


def preserve_retained_regular_file(
    retained: RetainedRegular,
    *,
    expected_payload: bytes,
) -> Path:
    """Preserve a failed retained locator without retiring a replacement.

    The initial payload/snapshot is retired when it still owns the canonical
    name. A concurrently installed name is left active while the retained inode
    is salvaged through its descriptor. If the same inode was mutated, that
    changed generation is moved out of the active namespace as evidence.
    """
    if not isinstance(retained, RetainedRegular):
        raise ImmutableArtifactError(
            "failed regular preservation requires a retained capability"
        )
    if type(expected_payload) is not bytes:
        raise ImmutableArtifactError(
            "failed regular preservation payload must be bytes"
        )
    if (
        len(expected_payload) != retained.size_bytes
        or sha256_bytes(expected_payload) != retained.sha256
    ):
        raise ImmutableArtifactError(
            "failed regular preservation payload disagrees with capability"
        )

    def preserved(path: Path) -> Path:
        state = retained._monitor_state
        if state is not None:
            # Preservation deliberately removes or supersedes the authenticated
            # path. The retained descriptor remains cleanup-only after that
            # terminal transition, so its close must not reinterpret the
            # preservation syscalls as a new integrity failure.
            state.monitor.cancel_close_verification()
        return path

    exact_error: BaseException | None = None
    try:
        retired = retire_regular_file(
            retained.path,
            expected_payload=expected_payload,
            expected_snapshot=retained.snapshot,
        )
    except BaseException as exc:
        exact_error = exc
    else:
        if retired is not None:
            return preserved(retired)

    try:
        descriptor = retained.descriptor
    except ImmutableArtifactError:
        if exact_error is not None:
            raise exact_error
        raise ImmutableArtifactError(
            "retained regular evidence disappeared before preservation"
        )

    current_error: BaseException | None = None
    try:
        retained._pinned.verify_path()
        current_snapshot = _snapshot_fields(os.fstat(descriptor))
    except BaseException:
        current_snapshot = None
    if current_snapshot is not None:
        try:
            retired = retire_regular_file(
                retained.path,
                expected_snapshot=current_snapshot,
            )
        except BaseException as exc:
            current_error = exc
        else:
            if retired is not None:
                return preserved(retired)

    try:
        return preserved(salvage_retained_regular_descriptor(
            retained.path,
            descriptor,
            expected_payload=expected_payload,
            expected_snapshot=retained.snapshot,
        ))
    except BaseException as initial_salvage_error:
        try:
            retained_name = _salvage_failed_regular_publication_at(
                retained._pinned.directory_descriptors[-1],
                descriptor,
            )
        except BaseException as current_salvage_error:
            failure = ImmutableArtifactError(
                "retained regular evidence preservation failed"
            )
            for detail in (
                exact_error,
                current_error,
                initial_salvage_error,
                current_salvage_error,
            ):
                if detail is not None:
                    failure.add_note(str(detail))
            raise failure from current_salvage_error
        return preserved(retained.path.with_name(retained_name))


def _preserve_failed_regular_publication_at(
    error: BaseException,
    parent_descriptor: int,
    active_name: str,
    retained_descriptor: int,
    *,
    evidence_descriptor: int | None = None,
) -> None:
    try:
        retired_name = _retire_failed_publication_at(
            parent_descriptor,
            active_name,
            retained_descriptor,
            directory=False,
        )
    except BaseException as retirement_error:
        try:
            salvaged_name = _salvage_failed_regular_publication_at(
                parent_descriptor,
                (
                    retained_descriptor
                    if evidence_descriptor is None
                    else evidence_descriptor
                ),
            )
        except BaseException as salvage_error:
            error.add_note(
                "failed regular publication could not leave the active "
                f"namespace: {retirement_error}"
            )
            error.add_note(
                "failed regular publication descriptor salvage failed: "
                f"{salvage_error}"
            )
        else:
            error.add_note(
                "failed regular publication path could not be retired safely: "
                f"{retirement_error}"
            )
            error.add_note(
                "failed regular publication bytes were copied outside the "
                f"active namespace as {salvaged_name}"
            )
    else:
        error.add_note(
            "failed regular publication preserved outside the active "
            f"namespace as {retired_name}"
        )


def _open_independent_parent_guard(descriptor: int) -> int:
    guard = os.open(
        ".",
        _directory_flags(),
        dir_fd=descriptor,
    )
    try:
        if _node_identity(os.fstat(guard)) != _node_identity(
            os.fstat(descriptor)
        ):
            raise ImmutableArtifactError(
                "publication parent rollback guard identity mismatch"
            )
        return guard
    except BaseException:
        _close_descriptors(
            (guard,),
            active_error=sys.exc_info()[1],
            label="publication parent rollback guard close",
        )
        raise


def _open_regular_rollback_guard(
    parent_descriptor: int,
    name: str,
    retained_descriptor: int,
) -> int:
    path_flag = getattr(os, "O_PATH", 0)
    if not path_flag:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: O_PATH rollback guard unavailable"
        )
    guard = os.open(
        name,
        path_flag
        | getattr(os, "O_CLOEXEC", 0)
        | _nofollow_flag(),
        dir_fd=parent_descriptor,
    )
    try:
        guarded = os.fstat(guard)
        retained = os.fstat(retained_descriptor)
        if (
            not stat.S_ISREG(guarded.st_mode)
            or _node_identity(guarded) != _node_identity(retained)
        ):
            raise ImmutableArtifactError(
                "regular publication rollback guard identity mismatch"
            )
        return guard
    except BaseException:
        _close_descriptors(
            (guard,),
            active_error=sys.exc_info()[1],
            label="regular publication rollback guard close",
        )
        raise


def _open_regular_evidence_guard(
    parent_descriptor: int,
    name: str,
    retained_descriptor: int,
) -> int:
    guard = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | _nofollow_flag(),
        dir_fd=parent_descriptor,
    )
    try:
        guarded = os.fstat(guard)
        retained = os.fstat(retained_descriptor)
        if (
            not stat.S_ISREG(guarded.st_mode)
            or _node_identity(guarded) != _node_identity(retained)
        ):
            raise ImmutableArtifactError(
                "regular publication evidence guard identity mismatch"
            )
        return guard
    except BaseException:
        _close_descriptors(
            (guard,),
            active_error=sys.exc_info()[1],
            label="regular publication evidence guard close",
        )
        raise


def _open_directory_rollback_guard(descriptor: int) -> int:
    path_flag = getattr(os, "O_PATH", 0)
    if not path_flag:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: O_PATH rollback guard unavailable"
        )
    guard = os.open(
        ".",
        path_flag
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | _nofollow_flag(),
        dir_fd=descriptor,
    )
    try:
        guarded = os.fstat(guard)
        retained = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(guarded.st_mode)
            or _node_identity(guarded) != _node_identity(retained)
        ):
            raise ImmutableArtifactError(
                "directory publication rollback guard identity mismatch"
            )
        return guard
    except BaseException:
        _close_descriptors(
            (guard,),
            active_error=sys.exc_info()[1],
            label="directory publication rollback guard close",
        )
        raise


_rollback_guard_state_lock = threading.Lock()
_rollback_guard_diagnostics: list[str] = []


def rollback_guard_cleanup_diagnostics() -> tuple[str, ...]:
    """Return bounded process-local diagnostics for nonfatal guard cleanup."""
    with _rollback_guard_state_lock:
        return tuple(_rollback_guard_diagnostics)


def _record_rollback_guard_diagnostic(detail: str) -> None:
    with _rollback_guard_state_lock:
        _rollback_guard_diagnostics.append(detail)
        overflow = (
            len(_rollback_guard_diagnostics)
            - _ROLLBACK_GUARD_DIAGNOSTIC_LIMIT
        )
        if overflow > 0:
            del _rollback_guard_diagnostics[:overflow]
    try:
        warnings.warn(detail, RuntimeWarning, stacklevel=3)
    except Exception:
        # Warning policy must not turn a post-retirement diagnostic into an
        # unretireable publication failure; the bounded sink remains observable.
        pass


def _close_rollback_guards(
    descriptors: Iterable[int], *, label: str,
) -> tuple[str, ...]:
    # These independent open descriptions exist so primary cleanup can fail
    # without forfeiting rollback authority. Once close() has been attempted,
    # Linux may have released the numeric descriptor even when it reports EIO.
    # Never inspect or retry that number: another thread may already own it.
    diagnostics: list[str] = []
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as exc:
            detail = (
                f"{DURABILITY_UNPROVEN}: {label}: descriptor {descriptor}: {exc}"
            )
            diagnostics.append(detail)
            _record_rollback_guard_diagnostic(detail)
    return tuple(diagnostics)


def rename_noreplace(source: Path, destination: Path) -> Path:
    """Publish one source without replacing a concurrent destination winner.

    The source is deliberately retained as evidence.  Callers that need a
    destructive move must use a higher-level transaction with an inode-bound
    cleanup policy; a pathname-only unlink would reintroduce a source-swap race.
    """
    source = Path(os.path.abspath(os.fspath(source)))
    destination = Path(os.path.abspath(os.fspath(destination)))
    if source.parent != destination.parent:
        raise ImmutableArtifactError(
            "atomic no-replace publication requires one pinned parent directory"
        )
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ImmutableArtifactError(
            f"atomic no-replace source is unavailable: {source}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ImmutableArtifactError(
            f"atomic no-replace source is a symlink: {source}"
        )
    with PinnedDirectoryReader(destination.parent) as parent:
        parent.verify_path()
        if _entry_mode(parent.descriptor, destination.name) is not None:
            raise FileExistsError(destination)
    if stat.S_ISDIR(metadata.st_mode):
        material: list[ImmutableFile] = []
        for path in sorted(source.rglob("*")):
            child_metadata = path.lstat()
            if stat.S_ISLNK(child_metadata.st_mode):
                raise ImmutableArtifactError(
                    f"atomic no-replace tree contains a symlink: {path}"
                )
            if stat.S_ISDIR(child_metadata.st_mode):
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise ImmutableArtifactError(
                    f"atomic no-replace tree contains a special file: {path}"
                )
            material.append(
                ImmutableFile.from_path(
                    path.relative_to(source).as_posix(), path,
                )
            )
        if not material:
            raise ImmutableArtifactError(
                f"atomic no-replace source directory is empty: {source}"
            )
        return publish_directory(destination.parent, destination.name, material)
    if not stat.S_ISREG(metadata.st_mode):
        raise ImmutableArtifactError(
            f"atomic no-replace source is not a regular file or directory: {source}"
        )
    return publish_regular_file(
        destination.parent,
        destination.name,
        source,
        executable=bool(metadata.st_mode & 0o111),
    )


def _partial_name(leaf_name: str) -> str:
    return f".{leaf_name}{_PARTIAL_SUFFIX}"


def _lock_partial(descriptor: int, partial_name: str) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ImmutableArtifactError(
            f"deterministic partial is owned by a live publisher: {partial_name}"
        ) from exc
    except OSError as exc:
        raise ImmutableArtifactError(
            f"cannot lock deterministic partial {partial_name}: {exc}"
        ) from exc


def _verify_locked_partial_path(
    parent_descriptor: int, partial_name: str, descriptor: int
) -> None:
    try:
        linked = os.stat(
            partial_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise ImmutableArtifactError(
            f"locked deterministic partial path changed: {partial_name}"
        ) from exc
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or linked.st_nlink != 1
        or opened.st_nlink != 1
        or _node_identity(linked) != _node_identity(opened)
    ):
        raise ImmutableArtifactError(
            f"locked deterministic partial path changed: {partial_name}"
        )


def _verify_open_file(descriptor: int, file: ImmutableFile) -> bool:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise ImmutableArtifactError("cannot rewind opened regular file") from exc
    digest, size, mode = _hash_fd(descriptor)
    return digest == file.sha256 and size == file.size_bytes and not (mode & 0o222)


def _open_exact_published_regular(
    path: Path,
    publication_descriptor: int,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_snapshot: MetadataSnapshot,
) -> RetainedRegular:
    """Retain the exact inode still owned by a completed publication."""
    retained = open_retained_regular(path)
    try:
        published_snapshot = _snapshot_fields(
            os.fstat(publication_descriptor)
        )
        if (
            published_snapshot != expected_snapshot
            or retained.snapshot != expected_snapshot
            or retained.size_bytes != expected_size_bytes
            or retained.sha256 != expected_sha256
            or _node_identity(os.fstat(retained.descriptor))
            != _node_identity(os.fstat(publication_descriptor))
        ):
            raise ImmutableArtifactError(
                "published regular-file retained capability changed"
            )
        retained.verify_unchanged()
        if (
            _snapshot_fields(os.fstat(publication_descriptor))
            != expected_snapshot
        ):
            raise ImmutableArtifactError(
                "published regular-file changed while retaining capability"
            )
        return retained
    except BaseException:
        retained.close()
        raise


def _verify_publication_parent_binding(
    pinned_parent: PinnedDirectoryReader,
    declared_parent: PinnedDirectoryReader | None,
) -> None:
    """Bind a caller-supplied parent capability to the declared parent path."""
    pinned_parent.verify_path()
    if declared_parent is None:
        return
    declared_parent.verify_path()
    if _node_identity(os.fstat(pinned_parent.descriptor)) != _node_identity(
        os.fstat(declared_parent.descriptor)
    ):
        raise ImmutableArtifactError(
            "caller-supplied pinned publication parent disagrees with parent path"
        )
    pinned_parent.verify_path()
    declared_parent.verify_path()


def _verify_declared_parent_descriptor_generation(
    parent: Path,
    expected_descriptor: int,
) -> None:
    """Rebind a lexical parent after pinned-reader monitor/descriptor closure."""
    expected = _node_identity(os.fstat(expected_descriptor))
    absolute = Path(os.path.abspath(os.fspath(parent)))
    descriptors: tuple[int, ...] = ()
    names: tuple[str, ...] = ()
    try:
        descriptors, names = _open_directory_chain(absolute)
        _verify_directory_chain(descriptors, names)
        if _node_identity(os.fstat(descriptors[-1])) != expected:
            raise ImmutableArtifactError(
                "declared publication parent generation changed during closure"
            )
        _verify_directory_chain(descriptors, names)
    finally:
        if descriptors:
            _close_descriptors(
                reversed(descriptors),
                label="declared publication parent final guard close",
            )
    try:
        linked = os.stat(absolute, follow_symlinks=False)
    except OSError as exc:
        raise ImmutableArtifactError(
            "declared publication parent generation changed during closure"
        ) from exc
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or _node_identity(linked) != expected
    ):
        raise ImmutableArtifactError(
            "declared publication parent generation changed during closure"
        )


def _open_publication_parent_binding(
    parent: Path,
    pinned_parent: PinnedDirectoryReader | None,
) -> tuple[PinnedDirectoryReader, bool, PinnedDirectoryReader | None]:
    if pinned_parent is not None and type(pinned_parent) is not PinnedDirectoryReader:
        raise ImmutableArtifactError(
            "pinned_parent must be an exact PinnedDirectoryReader"
        )
    if pinned_parent is None:
        try:
            return PinnedDirectoryReader(Path(parent)), True, None
        except (OSError, ImmutableArtifactError) as exc:
            raise ImmutableArtifactError(
                f"publication parent is unavailable: {parent}"
            ) from exc

    declared_parent: PinnedDirectoryReader | None = None
    try:
        declared_parent = PinnedDirectoryReader(Path(parent))
        _verify_publication_parent_binding(pinned_parent, declared_parent)
        return pinned_parent, False, declared_parent
    except BaseException as exc:
        if declared_parent is not None:
            _close_callables(
                (declared_parent.close,),
                label="declared publication parent binding cleanup",
            )
        if isinstance(exc, ImmutableArtifactError):
            raise
        raise ImmutableArtifactError(
            f"publication parent is unavailable: {parent}"
        ) from exc


@_translate_descriptor_resource_errors("regular-file publication")
def publish_regular_file(
    parent: Path,
    name: str,
    payload_or_source: bytes | Path | ImmutableFile,
    *,
    partial_suffix: str = _PARTIAL_SUFFIX,
    pinned_parent: PinnedDirectoryReader | None = None,
    executable: bool = False,
    before_commit: Callable[[], None] | None = None,
    after_commit: Callable[[], None] | None = None,
    _retain_capability: bool = False,
) -> Path | RetainedRegular:
    """Publish one regular file without destructive failure cleanup.

    The transaction owns only its descriptor and deterministic partial name.
    Incomplete, replaced, or otherwise uncertain partials are retained; the
    sole producer-side name removal is a verified ``RENAME_NOREPLACE`` commit.
    ``before_commit`` runs after rollback guards are armed but before that
    rename. ``after_commit`` is the final, read-only authorization after durable
    content and parent closure while rollback guards can still retire a failed
    final generation.
    """
    _validate_publication_name(name)
    if partial_suffix not in _PARTIAL_SUFFIXES:
        raise ImmutableArtifactError(
            f"{NAMESPACE_RESERVED_COMPONENT}: unsupported partial suffix"
        )
    if type(executable) is not bool:
        raise ImmutableArtifactError("executable must be boolean")
    for label, callback in (
        ("before_commit", before_commit),
        ("after_commit", after_commit),
    ):
        if callback is not None and not callable(callback):
            raise ImmutableArtifactError(f"{label} must be callable")
    if type(_retain_capability) is not bool:
        raise ImmutableArtifactError("retain capability selector must be boolean")
    if len(os.fsencode(name)) + len(os.fsencode(partial_suffix)) + 1 > 255:
        raise ImmutableArtifactError(
            f"{NAMESPACE_NAME_OVERFLOW}: regular publication partial name: {name!r}"
        )
    if isinstance(payload_or_source, ImmutableFile):
        source = payload_or_source
        material = ImmutableFile(
            name,
            source.sha256,
            source.size_bytes,
            source_path=source.source_path,
            payload=source.payload,
            allow_hardlinks=source.allow_hardlinks,
        )
    elif isinstance(payload_or_source, bytes):
        material = ImmutableFile.from_bytes(name, payload_or_source)
    elif isinstance(payload_or_source, Path):
        source = ImmutableFile.from_path(name, payload_or_source)
        material = source
    else:
        raise ImmutableArtifactError("regular-file payload must be bytes, Path, or ImmutableFile")

    pinned_parent, owns_parent, declared_parent = (
        _open_publication_parent_binding(Path(parent), pinned_parent)
    )
    descriptor: int | None = None
    lock_descriptor: int | None = None
    retained_publication: RetainedRegular | None = None
    rollback_parent = rollback_regular = rollback_evidence = -1
    committed = False
    partial_name = f".{name}{partial_suffix}"
    try:
        _verify_publication_parent_binding(pinned_parent, declared_parent)
        parent_descriptor = pinned_parent.descriptor
        if _entry_mode(parent_descriptor, name) is not None:
            raise ImmutableArtifactError(f"publication already exists: {name}")
        partial_mode = _entry_mode(parent_descriptor, partial_name)
        if partial_mode is not None:
            if not stat.S_ISREG(partial_mode) or stat.S_ISLNK(partial_mode):
                raise ImmutableArtifactError(
                    f"unrecognized partial stage entry: {partial_name}"
                )
            descriptor = os.open(
                partial_name,
                os.O_RDONLY
                | os.O_NONBLOCK
                | getattr(os, "O_CLOEXEC", 0)
                | _nofollow_flag(),
                dir_fd=parent_descriptor,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ImmutableArtifactError(
                    f"unrecognized partial stage entry: {partial_name}"
                )
            _lock_partial(descriptor, partial_name)
            if not _verify_open_file(descriptor, material):
                raise ImmutableArtifactError(
                    f"incomplete pre-existing partial is preserved: {partial_name}"
                )
            if executable and not (os.fstat(descriptor).st_mode & 0o111):
                raise ImmutableArtifactError(
                    f"pre-existing partial is not executable: {partial_name}"
                )
            _verify_locked_partial_path(parent_descriptor, partial_name, descriptor)
            _fsync_or_fail(descriptor, f"partial {partial_name}")
            lock_descriptor = descriptor
            descriptor = None
        else:
            descriptor = os.open(
                partial_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | _nofollow_flag(),
                0o400,
                dir_fd=parent_descriptor,
            )
            _lock_partial(descriptor, partial_name)
            _copy_to_descriptor(material, descriptor)
            os.fchmod(descriptor, 0o555 if executable else 0o444)
            if not _verify_open_file(descriptor, material):
                raise ImmutableArtifactError(f"immutable file mismatch: {name}")
            _fsync_or_fail(descriptor, f"partial {partial_name}")
            _verify_locked_partial_path(parent_descriptor, partial_name, descriptor)
            lock_descriptor = descriptor
            descriptor = None

        assert lock_descriptor is not None
        try:
            rollback_parent = _open_independent_parent_guard(
                pinned_parent.descriptor
            )
            rollback_regular = _open_regular_rollback_guard(
                pinned_parent.descriptor,
                partial_name,
                lock_descriptor,
            )
            rollback_evidence = _open_regular_evidence_guard(
                pinned_parent.descriptor,
                partial_name,
                lock_descriptor,
            )
        except BaseException as guard_error:
            _close_descriptors(
                tuple(
                    descriptor_value
                    for descriptor_value in (
                        rollback_evidence,
                        rollback_regular,
                        rollback_parent,
                    )
                    if descriptor_value >= 0
                ),
                active_error=guard_error,
                label=f"regular publication {name} rollback guard cleanup",
            )
            rollback_parent = rollback_regular = rollback_evidence = -1
            raise
        _verify_publication_parent_binding(pinned_parent, declared_parent)
        if before_commit is not None:
            before_commit()
        _fsync_or_fail(parent_descriptor, f"parent before {name}")
        _verify_publication_parent_binding(pinned_parent, declared_parent)
        _rename_noreplace_at(parent_descriptor, partial_name, parent_descriptor, name)
        committed = True
        _verify_publication_parent_binding(pinned_parent, declared_parent)
        _fsync_or_fail(parent_descriptor, f"parent after {name}")
        linked = _entry_metadata(parent_descriptor, name)
        opened = os.fstat(lock_descriptor)
        if linked is None:
            raise ImmutableArtifactError(f"named regular-file path changed: {name}")
        linked_snapshot = _snapshot_fields(linked)
        opened_snapshot = _snapshot_fields(opened)
        if (
            linked_snapshot != opened_snapshot
            or linked.st_nlink != 1
            or opened.st_nlink != 1
            or (executable and not (linked.st_mode & 0o111))
        ):
            raise ImmutableArtifactError(f"named regular-file path changed: {name}")
        if not _verify_open_file(lock_descriptor, material):
            raise ImmutableArtifactError(
                f"published regular-file bytes changed: {name}"
            )
        _fsync_or_fail(lock_descriptor, f"published regular file {name}")
        closed_linked = _entry_metadata(parent_descriptor, name)
        if (
            closed_linked is None
            or _snapshot_fields(closed_linked) != linked_snapshot
            or _snapshot_fields(os.fstat(lock_descriptor)) != opened_snapshot
        ):
            raise ImmutableArtifactError(
                f"published regular file changed during content closure: {name}"
            )
        _verify_publication_parent_binding(pinned_parent, declared_parent)

        cleanup_error: BaseException | None = None
        try:
            try:
                _close_descriptors(
                    (lock_descriptor,),
                    label=f"regular publication {name} close",
                )
            finally:
                lock_descriptor = None
        except BaseException as exc:
            cleanup_error = exc
        readers_to_close: list[tuple[str, PinnedDirectoryReader]] = []
        if owns_parent:
            readers_to_close.append(("publication parent", pinned_parent))
        if declared_parent is not None:
            readers_to_close.append(("declared parent binding", declared_parent))
        for reader_label, reader in readers_to_close:
            try:
                reader.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    cleanup_error.add_note(
                        f"{DURABILITY_UNPROVEN}: regular publication "
                        f"{reader_label} cleanup: {exc}"
                    )
            finally:
                if reader is declared_parent:
                    declared_parent = None
        if cleanup_error is not None:
            _preserve_failed_regular_publication_at(
                cleanup_error,
                rollback_parent,
                name,
                rollback_regular,
                evidence_descriptor=rollback_evidence,
            )
            guard_diagnostics = _close_rollback_guards(
                (rollback_evidence, rollback_regular, rollback_parent),
                label=f"regular publication {name} rollback guards",
            )
            for detail in guard_diagnostics:
                cleanup_error.add_note(detail)
            rollback_parent = rollback_regular = rollback_evidence = -1
            raise cleanup_error
        _verify_declared_parent_descriptor_generation(
            Path(parent), rollback_parent,
        )
        if _retain_capability:
            try:
                retained_publication = _open_exact_published_regular(
                    Path(parent) / name,
                    rollback_evidence,
                    expected_sha256=material.sha256,
                    expected_size_bytes=material.size_bytes,
                    expected_snapshot=opened_snapshot,
                )
            except BaseException as capability_error:
                _preserve_failed_regular_publication_at(
                    capability_error,
                    rollback_parent,
                    name,
                    rollback_regular,
                    evidence_descriptor=rollback_evidence,
                )
                guard_diagnostics = _close_rollback_guards(
                    (rollback_evidence, rollback_regular, rollback_parent),
                    label=f"regular publication {name} rollback guards",
                )
                for detail in guard_diagnostics:
                    capability_error.add_note(detail)
                rollback_parent = rollback_regular = rollback_evidence = -1
                raise
        _verify_declared_parent_descriptor_generation(
            Path(parent), rollback_parent,
        )
        if after_commit is not None:
            after_commit()
        _close_rollback_guards(
            (rollback_evidence, rollback_regular, rollback_parent),
            label=f"regular publication {name} rollback guards",
        )
        rollback_parent = rollback_regular = rollback_evidence = -1
        result: Path | RetainedRegular = (
            retained_publication
            if retained_publication is not None
            else Path(parent) / name
        )
        retained_publication = None
        return result
    except BaseException as exc:
        if committed and lock_descriptor is not None:
            _preserve_failed_regular_publication_at(
                exc, parent_descriptor, name, lock_descriptor,
            )
        elif (
            committed
            and rollback_parent >= 0
            and rollback_regular >= 0
        ):
            try:
                _preserve_failed_regular_publication_at(
                    exc,
                    rollback_parent,
                    name,
                    rollback_regular,
                    evidence_descriptor=(
                        rollback_evidence if rollback_evidence >= 0 else None
                    ),
                )
            except BaseException as preservation_error:
                try:
                    exc.add_note(
                        "failed regular publication final-path preservation: "
                        f"{preservation_error}"
                    )
                except (AttributeError, TypeError):
                    pass
        if isinstance(exc, OSError):
            raise ImmutableArtifactError(
                f"regular-file publication I/O failed: {exc}"
            ) from exc
        raise
    finally:
        closers: list[Callable[[], None]] = [
            lambda: _close_descriptors(
                tuple(
                    descriptor_value
                    for descriptor_value in (descriptor, lock_descriptor)
                    if descriptor_value is not None
                ),
                label=f"regular publication {name} close",
            )
        ]
        if retained_publication is not None:
            closers.append(retained_publication.close)
        rollback_guards = tuple(
            descriptor_value
            for descriptor_value in (
                rollback_evidence,
                rollback_regular,
                rollback_parent,
            )
            if descriptor_value >= 0
        )
        if rollback_guards:
            closers.append(
                lambda: _close_rollback_guards(
                    rollback_guards,
                    label=f"regular publication {name} rollback guards",
                )
            )
        if owns_parent:
            closers.append(pinned_parent.close)
        if declared_parent is not None:
            closers.append(declared_parent.close)
        _close_callables(closers, label=f"regular publication {name} cleanup")


def publish_retained_regular_file(
    parent: Path,
    name: str,
    payload_or_source: bytes | Path | ImmutableFile,
    *,
    partial_suffix: str = _PARTIAL_SUFFIX,
    pinned_parent: PinnedDirectoryReader | None = None,
    executable: bool = False,
    before_commit: Callable[[], None] | None = None,
    after_commit: Callable[[], None] | None = None,
) -> RetainedRegular:
    """Publish and return the exact committed locator as a live capability."""
    result = publish_regular_file(
        parent,
        name,
        payload_or_source,
        partial_suffix=partial_suffix,
        pinned_parent=pinned_parent,
        executable=executable,
        before_commit=before_commit,
        after_commit=after_commit,
        _retain_capability=True,
    )
    if not isinstance(result, RetainedRegular):  # pragma: no cover - invariant
        raise ImmutableArtifactError(
            "retained regular publication did not return a capability"
        )
    return result


def _install_staged_file_at(root_descriptor: int, file: ImmutableFile) -> None:
    """Recover/write one locked deterministic partial under one pinned stage."""
    relative = safe_relative_path(file.relative_path)
    parent = _ensure_pinned_relative_directory(
        root_descriptor, relative.parts[:-1]
    )
    parent_descriptor = parent.descriptor
    destination_name = relative.parts[-1]
    partial_name = _partial_name(destination_name)
    lock_descriptor: int | None = None
    try:
        destination_mode = _entry_mode(parent_descriptor, destination_name)
        partial_mode = _entry_mode(parent_descriptor, partial_name)
        if destination_mode is not None and partial_mode is not None:
            raise ImmutableArtifactError(
                f"ambiguous installed and partial stage entries: {file.relative_path}"
            )
        if destination_mode is not None:
            if not stat.S_ISREG(destination_mode):
                raise ImmutableArtifactError(f"stage destination is not regular: {file.relative_path}")
            return
        if partial_mode is not None:
            if not stat.S_ISREG(partial_mode) or stat.S_ISLNK(partial_mode):
                raise ImmutableArtifactError(f"unrecognized partial stage entry: {partial_name}")
            descriptor: int | None = os.open(
                partial_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | _nofollow_flag()
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            try:
                _lock_partial(descriptor, partial_name)
                complete = _verify_open_file(descriptor, file)
                _verify_locked_partial_path(parent_descriptor, partial_name, descriptor)
                if complete:
                    _fsync_or_fail(descriptor, f"partial {partial_name}")
                    lock_descriptor = descriptor
                    descriptor = None
                else:
                    raise ImmutableArtifactError(
                        f"incomplete pre-existing partial is preserved: {partial_name}"
                    )
            finally:
                if descriptor is not None:
                    _close_descriptors(
                        (descriptor,), label=f"partial {partial_name} close"
                    )
        if partial_mode is None:
            descriptor = None
            try:
                descriptor = os.open(
                    partial_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | _nofollow_flag()
                    | getattr(os, "O_NONBLOCK", 0),
                    0o444,
                    dir_fd=parent_descriptor,
                )
                _lock_partial(descriptor, partial_name)
                _copy_to_descriptor(file, descriptor)
                if os.fstat(descriptor).st_nlink != 1:
                    raise ImmutableArtifactError("hard-linked partial file is not immutable")
                os.fchmod(descriptor, 0o444)
                _fsync_or_fail(descriptor, f"partial {partial_name}")
                lock_descriptor = descriptor
                descriptor = None
            except BaseException:
                # Never unlink a failed or uncertain partial.  Its bytes remain
                # available for exact recovery or operator investigation.
                if descriptor is not None:
                    _close_descriptors(
                        (descriptor,), label=f"partial {partial_name} failure close"
                    )
                    descriptor = None
                raise
            finally:
                if descriptor is not None:
                    _close_descriptors(
                        (descriptor,), label=f"partial {partial_name} final close"
                    )
        if lock_descriptor is None:
            raise ImmutableArtifactError("deterministic partial lock was not retained")
        _verify_locked_partial_path(
            parent_descriptor, partial_name, lock_descriptor
        )
        _fsync_or_fail(parent_descriptor, f"parent before {file.relative_path}")
        _rename_noreplace_at(
            parent_descriptor,
            partial_name,
            parent_descriptor,
            destination_name,
        )
        _fsync_or_fail(parent_descriptor, f"parent after {file.relative_path}")
    finally:
        closers: list[Callable[[], None]] = []
        if lock_descriptor is not None:
            closers.append(
                lambda: _close_descriptors(
                    (lock_descriptor,), label=f"staged {file.relative_path} close"
                )
            )
        closers.append(parent.close)
        _close_callables(closers, label=f"staged {file.relative_path} cleanup")


def _install_staged_file(root: Path, file: ImmutableFile) -> None:
    pinned = _open_pinned_directory(root)
    try:
        _install_staged_file_at(pinned.descriptor, file)
        pinned.verify_path()
    finally:
        pinned.close()


def fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory(path)
    try:
        _fsync_or_fail(descriptor, f"directory {path}")
    finally:
        _close_descriptors((descriptor,), label=f"directory {path} close")


def _bounded_inventory_difference(
    expected: set[str], observed: set[str],
) -> str:
    missing_count = sum(path not in observed for path in expected)
    sample = heapq.nsmallest(
        _INVENTORY_DIAGNOSTIC_SAMPLE,
        (path for path in expected if path not in observed),
    )
    return f"count={missing_count}, sample={sample}"


def _tree_inventory_descriptor(
    root_descriptor: int,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    mismatch_label: str,
    allowed_files: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Inventory one pinned root within its permitted namespace bounds."""
    permitted_files = (
        expected_files
        if allowed_files is None
        else expected_files | allowed_files
    )
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    pending: list[tuple[str, ...]] = [()]
    while pending:
        parts = pending.pop()
        pinned = _open_pinned_relative_directory(root_descriptor, parts)
        try:
            with _scandir_independent(pinned.descriptor) as entries:
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    child_parts = (*parts, entry.name)
                    relative = PurePosixPath(*child_parts).as_posix()
                    safe_relative_path(relative)
                    if stat.S_ISLNK(metadata.st_mode):
                        raise ImmutableArtifactError(
                            f"symlink in immutable tree: {relative}"
                        )
                    if stat.S_ISDIR(metadata.st_mode):
                        if relative not in expected_directories:
                            raise ImmutableArtifactError(
                                f"{mismatch_label}: unexpected_directory={relative!r}"
                            )
                        observed_directories.add(relative)
                        pending.append(child_parts)
                    elif stat.S_ISREG(metadata.st_mode):
                        if relative not in permitted_files:
                            raise ImmutableArtifactError(
                                f"{mismatch_label}: unexpected_file={relative!r}"
                            )
                        observed_files.add(relative)
                    else:
                        raise ImmutableArtifactError(
                            f"non-regular entry in immutable tree: {relative}"
                        )
            pinned.verify_path()
        finally:
            pinned.close()
    return observed_files, observed_directories


def _tree_inventory(
    root: Path, expected_files: Iterable[str], *, mismatch_label: str,
) -> tuple[set[str], set[str]]:
    expected = set(expected_files)
    expected_directories = _expected_directories(expected)
    pinned = _open_pinned_directory(root)
    try:
        inventory = _tree_inventory_descriptor(
            pinned.descriptor,
            expected_files=expected,
            expected_directories=expected_directories,
            mismatch_label=mismatch_label,
        )
        pinned.verify_path()
        return inventory
    finally:
        pinned.close()


def _expected_directories(relative_paths: Iterable[str]) -> set[str]:
    expected: set[str] = set()
    for relative_path in relative_paths:
        parent = safe_relative_path(relative_path).parent
        while parent != PurePosixPath("."):
            expected.add(parent.as_posix())
            parent = parent.parent
    return expected


def _material_component_error(part: str, *, relative: str, leaf: bool) -> None:
    encoded = os.fsencode(part)
    if not encoded or encoded.startswith(b".") or any(
        byte < 0x20 or byte == 0x7F for byte in encoded
    ):
        raise ImmutableArtifactError(
            f"{NAMESPACE_RESERVED_COMPONENT}: {relative}"
        )
    if part.endswith(
        (".staging", ".m5-partial", ".m6-partial", ".ncu-details-partial")
    ) or ".partial-" in part:
        raise ImmutableArtifactError(
            f"{NAMESPACE_RESERVED_COMPONENT}: {relative}"
        )
    limit = _MAX_REGULAR_LEAF_BYTES if leaf else 255
    if len(encoded) > limit:
        raise ImmutableArtifactError(
            f"{NAMESPACE_NAME_OVERFLOW}: {relative}"
        )


def _validate_material_namespace(files: Iterable[ImmutableFile]) -> None:
    paths: set[str] = set()
    for file in files:
        relative = safe_relative_path(file.relative_path)
        if file.relative_path in paths:
            raise ImmutableArtifactError(
                f"duplicate publication path: {file.relative_path}"
            )
        paths.add(file.relative_path)
        for index, part in enumerate(relative.parts):
            _material_component_error(
                part,
                relative=file.relative_path,
                leaf=index == len(relative.parts) - 1,
            )
    for path in sorted(paths):
        parts = safe_relative_path(path).parts
        for index in range(1, len(parts)):
            prefix = PurePosixPath(*parts[:index]).as_posix()
            if prefix in paths:
                raise ImmutableArtifactError(
                    f"{NAMESPACE_PREFIX_CONFLICT}: {prefix} / {path}"
                )


def _material_map(files: Iterable[ImmutableFile]) -> dict[str, ImmutableFile]:
    material = tuple(files)
    _validate_material_namespace(material)
    expected: dict[str, ImmutableFile] = {}
    for file in material:
        if file.relative_path in expected:
            raise ImmutableArtifactError(
                f"duplicate publication path: {file.relative_path}"
            )
        expected[file.relative_path] = file
    return expected


@dataclasses.dataclass(frozen=True)
class _MutationDrainResult:
    events: tuple[tuple[int, int, bytes], ...]
    resource_error: ImmutableArtifactResourceError | None = None
    event_cookies: tuple[int, ...] = ()


class _DirectoryMutationMonitor:
    """Fail closed on local syscall mutations during bounded verification."""

    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        initialize = getattr(libc, "inotify_init1", None)
        add_watch = getattr(libc, "inotify_add_watch", None)
        close_on_exec = getattr(os, "O_CLOEXEC", None)
        if initialize is None or add_watch is None or not close_on_exec:
            raise ImmutableArtifactResourceError(
                f"{VERIFY_RESOURCE_BOUND}: inotify mutation monitor unavailable"
            )
        initialize.argtypes = (ctypes.c_int,)
        initialize.restype = ctypes.c_int
        add_watch.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
        add_watch.restype = ctypes.c_int
        descriptor = initialize(os.O_NONBLOCK | close_on_exec)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise ImmutableArtifactResourceError(
                f"{VERIFY_RESOURCE_BOUND}: inotify initialization failed: "
                f"[{error}] {os.strerror(error)}"
            )
        self.descriptor = descriptor
        self._add_watch = add_watch
        self._close_verifier: Callable[[], None] | None = None
        self._close_verification_cancelled = False

    def set_close_verifier(self, verifier: Callable[[], None]) -> None:
        if self.descriptor < 0:
            raise ImmutableArtifactError("inotify mutation monitor is closed")
        if self._close_verifier is not None:
            raise ImmutableArtifactError(
                "inotify mutation monitor close verifier is already configured"
            )
        self._close_verifier = verifier

    def cancel_close_verification(self) -> None:
        self._close_verification_cancelled = True

    def _watch_inode(
        self, descriptor: int, *, directory: bool,
    ) -> int:
        suffix = "/." if directory else ""
        path = os.fsencode(f"/proc/self/fd/{descriptor}{suffix}")
        mask = (
            _INOTIFY_DIRECTORY_MASK
            if directory
            else _INOTIFY_MUTATION_EVENTS
        )
        result = self._add_watch(self.descriptor, path, mask)
        if result >= 0:
            return result
        error = ctypes.get_errno()
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: inotify inode watch failed: "
            f"[{error}] {os.strerror(error)}"
        )

    def watch_directory(self, descriptor: int) -> int:
        return self._watch_inode(descriptor, directory=True)

    def watch_regular(self, descriptor: int) -> int:
        return self._watch_inode(descriptor, directory=False)

    def read_events(self) -> _MutationDrainResult:
        parsed: list[tuple[int, int, bytes]] = []
        cookies: list[int] = []
        attempts = batches = total_bytes = 0
        deferred_resource_error: ImmutableArtifactResourceError | None = None

        def remember_resource_failure(
            detail: str,
            cause: BaseException | None = None,
        ) -> None:
            nonlocal deferred_resource_error
            error = ImmutableArtifactResourceError(
                f"{VERIFY_RESOURCE_BOUND}: {detail}"
            )
            if cause is not None:
                error.__cause__ = cause
            if deferred_resource_error is None:
                deferred_resource_error = error
            else:
                deferred_resource_error.add_note(str(error))

        def resource_failure(
            detail: str,
            cause: BaseException | None = None,
        ) -> _MutationDrainResult:
            remember_resource_failure(detail, cause)
            return _MutationDrainResult(
                tuple(parsed), deferred_resource_error, tuple(cookies),
            )

        def completed() -> _MutationDrainResult:
            return _MutationDrainResult(
                tuple(parsed), deferred_resource_error, tuple(cookies),
            )

        while True:
            attempts += 1
            if attempts > _INOTIFY_MAX_DRAIN_ATTEMPTS:
                return resource_failure(
                    "inotify event drain exceeded bounded read attempts"
                )
            if (
                batches >= _INOTIFY_MAX_DRAIN_BATCHES
                or total_bytes >= _INOTIFY_MAX_DRAIN_BYTES
            ):
                return resource_failure(
                    "inotify event drain exceeded bounded bytes or batches"
                )
            try:
                events = os.read(self.descriptor, _INOTIFY_READ_BYTES)
            except BlockingIOError:
                return completed()
            except InterruptedError:
                continue
            except OSError as exc:
                return resource_failure("inotify event read failed", exc)
            if not events:
                return resource_failure("inotify monitor closed unexpectedly")
            batches += 1
            total_bytes += len(events)
            if total_bytes > _INOTIFY_MAX_DRAIN_BYTES:
                return resource_failure(
                    "inotify event drain exceeded bounded aggregate bytes"
                )
            offset = 0
            overflow_in_batch = False
            while offset < len(events):
                if len(events) - offset < 16:
                    return resource_failure("truncated inotify event header")
                watch_descriptor = int.from_bytes(
                    events[offset : offset + 4], sys.byteorder, signed=True,
                )
                mask = int.from_bytes(
                    events[offset + 4 : offset + 8], sys.byteorder,
                )
                cookie = int.from_bytes(
                    events[offset + 8 : offset + 12], sys.byteorder,
                )
                name_length = int.from_bytes(
                    events[offset + 12 : offset + 16], sys.byteorder,
                )
                event_end = offset + 16 + name_length
                if event_end > len(events):
                    return resource_failure("truncated inotify event payload")
                if mask & _INOTIFY_Q_OVERFLOW:
                    # Keep draining this already-read batch: a concrete mutation
                    # that follows the overflow marker is stronger evidence than
                    # the resource failure and must remain visible to callers.
                    remember_resource_failure(
                        "inotify event queue overflowed"
                    )
                    overflow_in_batch = True
                    offset = event_end
                    continue
                if len(parsed) >= _INOTIFY_MAX_DRAIN_EVENTS:
                    return resource_failure(
                        "inotify event drain exceeded bounded event count"
                    )
                name = events[offset + 16 : event_end].rstrip(b"\0")
                parsed.append((watch_descriptor, mask, name))
                cookies.append(cookie)
                offset = event_end
            if overflow_in_batch:
                return completed()

    def require_quiet(self) -> None:
        result = self.read_events()
        if result.events:
            self.cancel_close_verification()
            error = ImmutableArtifactError(
                "immutable tree changed during bounded verification"
            )
            if result.resource_error is not None:
                error.add_note(str(result.resource_error))
            raise error
        if result.resource_error is not None:
            self.cancel_close_verification()
            raise result.resource_error

    def close(self) -> None:
        descriptor = self.descriptor
        if descriptor < 0:
            return
        verification_error: BaseException | None = None
        verifier = self._close_verifier
        self._close_verifier = None
        try:
            # A caller may mutate protected state in the final Python-level close
            # callback after its preceding verification. Drain once more while the
            # inotify descriptor and every watch are still live so teardown cannot
            # create an unobserved terminal race window. Filtered monitors install
            # their own verifier so unrelated sibling activity remains admissible.
            if self._close_verification_cancelled:
                pass
            elif verifier is None:
                self.require_quiet()
            else:
                verifier()
        except BaseException as exc:
            verification_error = exc
        self.descriptor = -1
        _close_descriptors(
            (descriptor,),
            active_error=verification_error,
            label="inotify mutation monitor close",
        )
        if verification_error is not None:
            raise verification_error


def _raise_monitored_mutation(
    message: str,
    result: _MutationDrainResult,
) -> NoReturn:
    error = ImmutableArtifactError(message)
    if result.resource_error is not None:
        error.add_note(str(result.resource_error))
    raise error


@dataclasses.dataclass(frozen=True)
class _RetainedDirectoryPathMonitorState:
    monitor: _DirectoryMutationMonitor
    directory_watches: frozenset[int]
    protected_names: tuple[tuple[int, bytes], ...]
    root_watch: int


@dataclasses.dataclass(frozen=True)
class _RetainedPathMonitorState:
    monitor: _DirectoryMutationMonitor
    directory_watches: frozenset[int]
    protected_names: tuple[tuple[int, bytes], ...]
    regular_watch: int


@dataclasses.dataclass(frozen=True)
class _DirectoryRenameMonitorState:
    monitor: _DirectoryMutationMonitor
    source_watch: int
    destination_watch: int
    root_watch: int
    source_name: bytes
    destination_name: bytes

    @property
    def watches(self) -> frozenset[int]:
        return frozenset((
            self.source_watch,
            self.destination_watch,
            self.root_watch,
        ))


def _arm_directory_rename_monitor(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
    root_descriptor: int,
) -> _DirectoryRenameMonitorState:
    monitor = _DirectoryMutationMonitor()
    try:
        return _DirectoryRenameMonitorState(
            monitor=monitor,
            source_watch=monitor.watch_directory(source_descriptor),
            destination_watch=monitor.watch_directory(destination_descriptor),
            root_watch=monitor.watch_directory(root_descriptor),
            source_name=os.fsencode(source_name),
            destination_name=os.fsencode(destination_name),
        )
    except BaseException:
        monitor.cancel_close_verification()
        monitor.close()
        raise


def _require_directory_rename_monitor_quiet(
    state: _DirectoryRenameMonitorState,
) -> None:
    try:
        result = state.monitor.read_events()
        protected_names = {
            (state.source_watch, state.source_name),
            (state.destination_watch, state.destination_name),
        }
        for watch_descriptor, _mask, name in result.events:
            if watch_descriptor not in state.watches:
                _raise_monitored_mutation(
                    "retained directory rename observed an unknown watch",
                    result,
                )
            if (
                watch_descriptor == state.root_watch
                or not name
                or (watch_descriptor, name) in protected_names
            ):
                _raise_monitored_mutation(
                    "retained directory changed outside authorized rename",
                    result,
                )
        if result.resource_error is not None:
            raise result.resource_error
    except BaseException:
        state.monitor.cancel_close_verification()
        raise


def _require_directory_rename_monitor_complete(
    state: _DirectoryRenameMonitorState,
) -> None:
    try:
        result = state.monitor.read_events()
        if len(result.event_cookies) != len(result.events):
            raise ImmutableArtifactError(
                "retained directory rename monitor lacks event cookies"
            )
        source_event = (
            state.source_watch,
            _INOTIFY_ROOT_DIRECTORY_MOVED_FROM,
            state.source_name,
        )
        destination_event = (
            state.destination_watch,
            _INOTIFY_ROOT_DIRECTORY_MOVED_TO,
            state.destination_name,
        )
        root_event = (state.root_watch, _INOTIFY_ROOT_MOVE_SELF, b"")
        expected = {source_event, destination_event, root_event}
        protected_names = {
            (state.source_watch, state.source_name),
            (state.destination_watch, state.destination_name),
        }
        observed: dict[tuple[int, int, bytes], int] = {}
        for event, cookie in zip(
            result.events, result.event_cookies, strict=True,
        ):
            watch_descriptor, _mask, name = event
            if watch_descriptor not in state.watches:
                _raise_monitored_mutation(
                    "retained directory rename observed an unknown watch",
                    result,
                )
            if event in expected:
                if event in observed:
                    _raise_monitored_mutation(
                        "retained directory rename emitted a duplicate event",
                        result,
                    )
                observed[event] = cookie
                continue
            if (
                watch_descriptor == state.root_watch
                or not name
                or (watch_descriptor, name) in protected_names
            ):
                _raise_monitored_mutation(
                    "retained directory changed outside authorized rename",
                    result,
                )
        if result.resource_error is not None:
            raise result.resource_error
        if set(observed) != expected:
            raise ImmutableArtifactError(
                "authorized retained directory rename was not fully observed"
            )
        source_cookie = observed[source_event]
        destination_cookie = observed[destination_event]
        if (
            source_cookie == 0
            or destination_cookie == 0
            or source_cookie != destination_cookie
            or observed[root_event] != 0
        ):
            raise ImmutableArtifactError(
                "authorized retained directory rename cookie mismatch"
            )
    except BaseException:
        state.monitor.cancel_close_verification()
        raise


def _require_retained_directory_path_monitor_quiet(
    state: _RetainedDirectoryPathMonitorState | None,
) -> None:
    if state is None:
        return
    try:
        result = state.monitor.read_events()
        for watch_descriptor, _mask, name in result.events:
            if watch_descriptor not in state.directory_watches:
                _raise_monitored_mutation(
                    "retained directory monitor observed an unknown watch",
                    result,
                )
            if not name:
                _raise_monitored_mutation(
                    "retained directory ancestor changed during use",
                    result,
                )
            if any(
                watch_descriptor == protected_watch
                and name == protected_name
                for protected_watch, protected_name in state.protected_names
            ):
                _raise_monitored_mutation(
                    "retained directory ancestor changed during use",
                    result,
                )
        if result.resource_error is not None:
            raise result.resource_error
    except BaseException:
        state.monitor.cancel_close_verification()
        raise


def _require_retained_directory_root_mode_change(
    state: _RetainedDirectoryPathMonitorState | None,
) -> None:
    if state is None:
        raise ImmutableArtifactError("retained directory monitor is closed")
    try:
        if (
            state.root_watch not in state.directory_watches
            or not state.protected_names
        ):
            raise ImmutableArtifactError(
                "retained directory monitor lacks a protected root"
            )
        parent_watch, root_name = state.protected_names[-1]
        expected = {
            (state.root_watch, _INOTIFY_ROOT_DIRECTORY_ATTRIB, b""),
            (parent_watch, _INOTIFY_ROOT_DIRECTORY_ATTRIB, root_name),
        }
        result = state.monitor.read_events()
        observed: set[tuple[int, int, bytes]] = set()
        for event in result.events:
            watch_descriptor, _mask, _name = event
            if watch_descriptor not in state.directory_watches:
                _raise_monitored_mutation(
                    "retained directory monitor observed an unknown watch",
                    result,
                )
            if event in expected:
                if event in observed:
                    _raise_monitored_mutation(
                        "retained directory mode seal emitted a duplicate event",
                        result,
                    )
                observed.add(event)
                continue
            _raise_monitored_mutation(
                "retained directory changed outside authorized root mode seal",
                result,
            )
        if result.resource_error is not None:
            raise result.resource_error
        if observed != expected:
            raise ImmutableArtifactError(
                "authorized retained directory root mode seal was not fully observed"
            )
    except BaseException:
        state.monitor.cancel_close_verification()
        raise


def _require_retained_directory_root_rename(
    state: _RetainedDirectoryPathMonitorState | None,
    *,
    destination_entry: tuple[int, bytes] | None,
) -> None:
    if state is None:
        raise ImmutableArtifactError("retained directory monitor is closed")
    try:
        if (
            state.root_watch not in state.directory_watches
            or not state.protected_names
        ):
            raise ImmutableArtifactError(
                "retained directory monitor lacks a protected root"
            )
        parent_watch, root_name = state.protected_names[-1]
        expected = {
            (
                parent_watch,
                _INOTIFY_ROOT_DIRECTORY_MOVED_FROM,
                root_name,
            ),
            (state.root_watch, _INOTIFY_ROOT_MOVE_SELF, b""),
        }
        if destination_entry is not None:
            destination_watch, destination_name = destination_entry
            if destination_watch not in state.directory_watches:
                raise ImmutableArtifactError(
                    "retained directory rename destination watch is unknown"
                )
            if (destination_watch, destination_name) == (
                parent_watch,
                root_name,
            ):
                raise ImmutableArtifactError(
                    "retained directory rename destination is unchanged"
                )
            expected.add((
                destination_watch,
                _INOTIFY_ROOT_DIRECTORY_MOVED_TO,
                destination_name,
            ))

        result = state.monitor.read_events()
        observed: set[tuple[int, int, bytes]] = set()
        for event in result.events:
            watch_descriptor, _mask, name = event
            if watch_descriptor not in state.directory_watches:
                _raise_monitored_mutation(
                    "retained directory monitor observed an unknown watch",
                    result,
                )
            if event in expected:
                if event in observed:
                    _raise_monitored_mutation(
                        "retained directory root rename emitted a duplicate event",
                        result,
                    )
                observed.add(event)
                continue
            destination_name_changed = (
                destination_entry is not None
                and (watch_descriptor, name) == destination_entry
            )
            if (
                not name
                or destination_name_changed
                or any(
                    watch_descriptor == protected_watch
                    and name == protected_name
                    for protected_watch, protected_name in state.protected_names
                )
            ):
                _raise_monitored_mutation(
                    "retained directory changed outside authorized root rename",
                    result,
                )
        if result.resource_error is not None:
            raise result.resource_error
        if observed != expected:
            raise ImmutableArtifactError(
                "authorized retained directory root rename was not fully observed"
            )
    except BaseException:
        state.monitor.cancel_close_verification()
        raise



def _open_directory_with_early_path_monitor(
    path: Path,
) -> tuple[_PinnedDirectory, _RetainedDirectoryPathMonitorState]:
    """Pin one directory while protecting every name before traversal."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    names = tuple(absolute.parts[1:])
    if (
        len(names) > _MAX_COMPONENTS
        or len(os.fsencode(os.fspath(absolute))) > _MAX_RELATIVE_BYTES
    ):
        raise ImmutableArtifactError(
            "absolute directory traversal exceeds bounded limits"
        )

    monitor = _DirectoryMutationMonitor()
    descriptors: list[int] = []
    directory_watches: list[int] = []
    protected_names: list[tuple[int, bytes]] = []
    pinned: _PinnedDirectory | None = None
    transferred = False
    try:
        try:
            descriptors.append(
                os.open(os.path.sep, _directory_flags() & ~_nofollow_flag())
            )
        except OSError as exc:
            raise ImmutableArtifactError(
                f"cannot open retained directory root for {absolute}: {exc}"
            ) from exc
        directory_watches.append(
            monitor.watch_directory(descriptors[-1])
        )
        for component_name in names:
            protected_names.append((
                directory_watches[-1], os.fsencode(component_name),
            ))
            try:
                descriptors.append(
                    _open_dir_at(descriptors[-1], component_name)
                )
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise ImmutableArtifactError(
                    "cannot traverse retained directory ancestors for "
                    f"{absolute}: {exc}"
                ) from exc
            directory_watches.append(
                monitor.watch_directory(descriptors[-1])
            )
        pinned = _PinnedDirectory(tuple(descriptors), names)
        descriptors = []
        state = _RetainedDirectoryPathMonitorState(
            monitor=monitor,
            directory_watches=frozenset(directory_watches),
            protected_names=tuple(protected_names),
            root_watch=directory_watches[-1],
        )
        pinned.verify_path()
        _require_retained_directory_path_monitor_quiet(state)
        pinned.verify_path()
        monitor.set_close_verifier(
            lambda: _require_retained_directory_path_monitor_quiet(state)
        )
        transferred = True
        return pinned, state
    finally:
        if not transferred:
            monitor.cancel_close_verification()
            closers: list[Callable[[], None]] = [monitor.close]
            if pinned is not None:
                closers.append(pinned.close)
            elif descriptors:
                closers.append(
                    lambda: _close_descriptors(
                        reversed(descriptors),
                        label="early retained directory open cleanup",
                    )
                )
            _close_callables(
                closers, label="early retained directory monitor cleanup"
            )


def _arm_retained_path_monitor(
    pinned: _PinnedRegular,
) -> _RetainedPathMonitorState:
    monitor = _DirectoryMutationMonitor()
    try:
        ancestor_snapshots = tuple(
            _snapshot_fields(os.fstat(descriptor))
            for descriptor in pinned.directory_descriptors
        )
        leaf_snapshot = _snapshot_fields(os.fstat(pinned.descriptor))
        pinned.verify_path()
        directory_watches = tuple(
            monitor.watch_directory(descriptor)
            for descriptor in pinned.directory_descriptors
        )
        protected_names = tuple(
            (
                directory_watches[index],
                os.fsencode(component_name),
            )
            for index, component_name in enumerate(pinned.directory_names)
        ) + ((directory_watches[-1], os.fsencode(pinned.leaf_name)),)
        state = _RetainedPathMonitorState(
            monitor=monitor,
            directory_watches=frozenset(directory_watches),
            protected_names=protected_names,
            regular_watch=monitor.watch_regular(pinned.descriptor),
        )
        pinned.verify_path()
        if tuple(
            _snapshot_fields(os.fstat(descriptor))
            for descriptor in pinned.directory_descriptors
        ) != ancestor_snapshots or _snapshot_fields(
            os.fstat(pinned.descriptor)
        ) != leaf_snapshot:
            raise ImmutableArtifactError(
                "retained regular-file path changed while arming monitor"
            )
        _require_retained_path_monitor_quiet(state)
        monitor.set_close_verifier(
            lambda: _require_retained_path_monitor_quiet(state)
        )
        return state
    except BaseException:
        monitor.cancel_close_verification()
        monitor.close()
        raise


def _open_regular_with_early_path_monitor(
    path: Path, *, allow_hardlinks: bool = False,
) -> tuple[_PinnedRegular, _RetainedPathMonitorState]:
    """Open one regular leaf while watching every not-yet-traversed name."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.parent == absolute:
        raise ImmutableArtifactError(f"not a regular file: {absolute}")
    directory_names = tuple(absolute.parent.parts[1:])
    if (
        len(directory_names) > _MAX_COMPONENTS
        or len(os.fsencode(os.fspath(absolute.parent))) > _MAX_RELATIVE_BYTES
    ):
        raise ImmutableArtifactError(
            "absolute directory traversal exceeds bounded limits"
        )

    monitor = _DirectoryMutationMonitor()
    directory_descriptors: list[int] = []
    directory_watches: list[int] = []
    protected_names: list[tuple[int, bytes]] = []
    descriptor = -1
    pinned: _PinnedRegular | None = None
    transferred = False
    try:
        try:
            directory_descriptors.append(
                os.open(os.path.sep, _directory_flags() & ~_nofollow_flag())
            )
            directory_watches.append(
                monitor.watch_directory(directory_descriptors[-1])
            )
            for component_name in directory_names:
                protected_names.append((
                    directory_watches[-1],
                    os.fsencode(component_name),
                ))
                directory_descriptors.append(
                    _open_dir_at(directory_descriptors[-1], component_name)
                )
                directory_watches.append(
                    monitor.watch_directory(directory_descriptors[-1])
                )
        except (OSError, ImmutableArtifactError) as exc:
            raise ImmutableArtifactError(
                f"cannot traverse regular-file ancestors for {absolute}: {exc}"
            ) from exc

        protected_names.append((
            directory_watches[-1], os.fsencode(absolute.name),
        ))
        try:
            declared = os.stat(
                absolute.name,
                dir_fd=directory_descriptors[-1],
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ImmutableArtifactError(
                f"cannot inspect regular file {absolute}: {exc}"
            ) from exc
        if stat.S_ISLNK(declared.st_mode) or not stat.S_ISREG(declared.st_mode):
            raise ImmutableArtifactError(f"not a regular file: {absolute}")
        if declared.st_nlink != 1 and not allow_hardlinks:
            raise ImmutableArtifactError(
                f"hard-linked regular file is not immutable: {absolute}"
            )

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | _nofollow_flag()
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(
                absolute.name,
                flags,
                dir_fd=directory_descriptors[-1],
            )
        except OSError as exc:
            raise ImmutableArtifactError(
                f"cannot open regular file {absolute}: {exc}"
            ) from exc
        pinned = _PinnedRegular(
            descriptor,
            tuple(directory_descriptors),
            directory_names,
            absolute.name,
            allow_hardlinks=allow_hardlinks,
        )
        descriptor = -1
        directory_descriptors = []
        before = _snapshot_fields(os.fstat(pinned.descriptor))
        regular_watch = monitor.watch_regular(pinned.descriptor)
        state = _RetainedPathMonitorState(
            monitor=monitor,
            directory_watches=frozenset(directory_watches),
            protected_names=tuple(protected_names),
            regular_watch=regular_watch,
        )
        pinned.verify_path()
        if _snapshot_fields(os.fstat(pinned.descriptor)) != before:
            raise ImmutableArtifactError(
                "retained regular-file path changed while arming monitor"
            )
        _require_retained_path_monitor_quiet(state)
        monitor.set_close_verifier(
            lambda: _require_retained_path_monitor_quiet(state)
        )
        transferred = True
        return pinned, state
    finally:
        if not transferred:
            monitor.cancel_close_verification()
            closers: list[Callable[[], None]] = [monitor.close]
            if pinned is not None:
                closers.append(pinned.close)
            else:
                closers.append(
                    lambda: _close_descriptors(
                        tuple(
                            value
                            for value in (
                                descriptor,
                                *reversed(directory_descriptors),
                            )
                            if value >= 0
                        ),
                        label="early retained regular open cleanup",
                    )
                )
            _close_callables(
                closers, label="early retained regular monitor cleanup"
            )


def _require_retained_path_monitor_quiet(
    state: _RetainedPathMonitorState | None,
) -> None:
    if state is None:
        return
    try:
        result = state.monitor.read_events()
        for watch_descriptor, mask, name in result.events:
            if watch_descriptor == state.regular_watch:
                _raise_monitored_mutation(
                    "retained regular-file generation changed during use",
                    result,
                )
            if watch_descriptor not in state.directory_watches:
                _raise_monitored_mutation(
                    "retained regular-file monitor observed an unknown watch",
                    result,
                )
            if not name:
                _raise_monitored_mutation(
                    "retained regular-file ancestor changed during use",
                    result,
                )
            if any(
                watch_descriptor == protected_watch
                and name == protected_name
                for protected_watch, protected_name in state.protected_names
            ):
                if mask & _INOTIFY_NAMESPACE_EVENTS:
                    if (
                        watch_descriptor,
                        name,
                    ) == state.protected_names[-1]:
                        _raise_monitored_mutation(
                            "named regular-file path changed",
                            result,
                        )
                    _raise_monitored_mutation(
                        "retained regular-file ancestor changed during use",
                        result,
                    )
                _raise_monitored_mutation(
                    "retained regular-file generation changed during use",
                    result,
                )
        if result.resource_error is not None:
            raise result.resource_error
    except BaseException:
        state.monitor.cancel_close_verification()
        raise


def _walk_tree_descriptor(
    root_descriptor: int,
    *,
    observe_directory: Callable[[int], None] | None = None,
    max_files: int | None = None,
    max_directories: int | None = None,
    max_total_file_bytes: int | None = None,
    max_total_relative_path_bytes: int | None = None,
) -> _TreeSnapshot:
    """Inventory one tree while retaining only an O(depth) descriptor stack."""
    limits = (
        ("file", max_files),
        ("directory", max_directories),
        ("aggregate file-byte", max_total_file_bytes),
        ("aggregate relative-path-byte", max_total_relative_path_bytes),
    )
    for label, limit in limits:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ImmutableArtifactError(f"pinned tree {label} limit is invalid")
    if max_directories is not None and max_directories < 1:
        raise ImmutableArtifactError(
            "pinned tree directory limit does not admit the root"
        )

    files: dict[str, _Snapshot] = {}
    directories: dict[str, _Snapshot] = {}
    pending: list[tuple[str, ...]] = [()]
    discovered_files = 0
    discovered_directories = 1
    total_file_bytes = 0
    total_relative_path_bytes = 0
    while pending:
        parts = pending.pop()
        directory = _open_pinned_relative_directory(root_descriptor, parts)
        try:
            if observe_directory is not None:
                observe_directory(directory.descriptor)
            directories[PurePosixPath(*parts).as_posix() if parts else "."] = (
                _snapshot_fields(os.fstat(directory.descriptor))
            )
            children: list[tuple[str, os.stat_result]] = []
            with _scandir_independent(directory.descriptor) as entries:
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    child = (*parts, entry.name)
                    relative = PurePosixPath(*child).as_posix()
                    safe_relative_path(relative)
                    total_relative_path_bytes += len(os.fsencode(relative))
                    if (
                        max_total_relative_path_bytes is not None
                        and total_relative_path_bytes
                        > max_total_relative_path_bytes
                    ):
                        raise ImmutableArtifactLimitError(
                            "pinned tree exceeds bounded aggregate relative-path-"
                            f"byte limit: {total_relative_path_bytes} > "
                            f"{max_total_relative_path_bytes}"
                        )
                    if stat.S_ISLNK(metadata.st_mode):
                        raise ImmutableArtifactError(
                            f"symlink in immutable tree: {relative}"
                        )
                    if stat.S_ISDIR(metadata.st_mode):
                        discovered_directories += 1
                        if (
                            max_directories is not None
                            and discovered_directories > max_directories
                        ):
                            raise ImmutableArtifactLimitError(
                                "pinned tree exceeds bounded directory limit: "
                                f"{discovered_directories} > {max_directories}"
                            )
                    elif stat.S_ISREG(metadata.st_mode):
                        discovered_files += 1
                        if max_files is not None and discovered_files > max_files:
                            raise ImmutableArtifactLimitError(
                                "pinned tree exceeds bounded file limit: "
                                f"{discovered_files} > {max_files}"
                            )
                        total_file_bytes += metadata.st_size
                        if (
                            max_total_file_bytes is not None
                            and total_file_bytes > max_total_file_bytes
                        ):
                            raise ImmutableArtifactLimitError(
                                "pinned tree exceeds bounded aggregate file-byte "
                                f"limit: {total_file_bytes} > "
                                f"{max_total_file_bytes}"
                            )
                    else:
                        raise ImmutableArtifactError(
                            f"non-regular entry in immutable tree: {relative}"
                        )
                    children.append((entry.name, metadata))
            children.sort(key=lambda item: item[0])
            directory.verify_path()
        finally:
            directory.close()
        for name, metadata in children:
            child = (*parts, name)
            relative = PurePosixPath(*child).as_posix()
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            else:
                files[relative] = _snapshot_fields(metadata)
    return files, directories


def _require_pinned_path_set_events_quiet(
    monitor: _DirectoryMutationMonitor,
    *,
    any_watches: frozenset[int] | set[int],
    named_watches: Mapping[int, frozenset[bytes] | set[bytes]],
    self_watches: frozenset[int] | set[int],
) -> None:
    try:
        result = monitor.read_events()
        for watch_descriptor, _mask, name in result.events:
            if watch_descriptor in any_watches:
                _raise_monitored_mutation(
                    "protected input tree changed during downstream use",
                    result,
                )
            protected_names = named_watches.get(watch_descriptor)
            if protected_names is not None:
                if not name or name in protected_names:
                    _raise_monitored_mutation(
                        "protected input path changed during downstream use",
                        result,
                    )
                continue
            if watch_descriptor in self_watches:
                if not name:
                    _raise_monitored_mutation(
                        "protected input ancestor changed during downstream use",
                        result,
                    )
                continue
            _raise_monitored_mutation(
                "protected input monitor observed an unknown watch",
                result,
            )
        if result.resource_error is not None:
            raise result.resource_error
    except BaseException:
        monitor.cancel_close_verification()
        raise


class PinnedPathSetMonitor:
    """Continuously guard selected regular files and recursive input trees.

    Directory watches on the lexical ancestor chain reject self-moves while
    allowing unrelated child publication. Recursive directories reject every
    mutation. Filtered directories reject self-moves plus changes to named
    children. Optional regular paths protect either the opened generation or
    descriptor-anchored absence at the nearest existing parent.
    """

    @_translate_descriptor_resource_errors("pinned path-set monitor open")
    def __init__(
        self,
        root: Path,
        *,
        regular_paths: Iterable[str],
        recursive_directories: Iterable[str],
        optional_regular_paths: Iterable[str] = (),
        filtered_directories: Mapping[str, Iterable[str]] | None = None,
        max_files: int | None = None,
        max_directories: int | None = None,
        max_total_file_bytes: int | None = None,
        max_total_relative_path_bytes: int | None = None,
    ) -> None:
        self._reader = PinnedDirectoryReader(Path(root))
        self._root = self._reader._root
        self._owns_root = True
        self._monitor: _DirectoryMutationMonitor | None = None
        try:
            self._arm(
                regular_paths=regular_paths,
                optional_regular_paths=optional_regular_paths,
                recursive_directories=recursive_directories,
                filtered_directories=filtered_directories,
                max_files=max_files,
                max_directories=max_directories,
                max_total_file_bytes=max_total_file_bytes,
                max_total_relative_path_bytes=max_total_relative_path_bytes,
            )
        except BaseException:
            monitor = self._monitor
            if monitor is not None:
                monitor.cancel_close_verification()
            self.close()
            raise

    @classmethod
    @_translate_descriptor_resource_errors("pinned path-set monitor retention")
    def from_reader(
        cls,
        reader: PinnedDirectoryReader,
        *,
        regular_paths: Iterable[str],
        recursive_directories: Iterable[str],
        optional_regular_paths: Iterable[str] = (),
        filtered_directories: Mapping[str, Iterable[str]] | None = None,
        max_files: int | None = None,
        max_directories: int | None = None,
        max_total_file_bytes: int | None = None,
        max_total_relative_path_bytes: int | None = None,
    ) -> PinnedPathSetMonitor:
        """Arm watches from an existing pinned root without reopening its path."""
        if not isinstance(reader, PinnedDirectoryReader):
            raise TypeError("reader must be a PinnedDirectoryReader")
        reader.verify_path()
        monitor = cls.__new__(cls)
        monitor._reader = reader
        monitor._root = reader._root
        monitor._owns_root = False
        monitor._monitor = None
        try:
            monitor._arm(
                regular_paths=regular_paths,
                optional_regular_paths=optional_regular_paths,
                recursive_directories=recursive_directories,
                filtered_directories=filtered_directories,
                max_files=max_files,
                max_directories=max_directories,
                max_total_file_bytes=max_total_file_bytes,
                max_total_relative_path_bytes=max_total_relative_path_bytes,
            )
        except BaseException:
            low_level_monitor = monitor._monitor
            if low_level_monitor is not None:
                low_level_monitor.cancel_close_verification()
            monitor.close()
            raise
        return monitor

    def _verify_root_path(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is None:
            raise ImmutableArtifactError("protected input root is closed")
        reader.verify_path()

    def _arm(
        self,
        *,
        regular_paths: Iterable[str],
        recursive_directories: Iterable[str],
        optional_regular_paths: Iterable[str],
        filtered_directories: Mapping[str, Iterable[str]] | None,
        max_files: int | None,
        max_directories: int | None,
        max_total_file_bytes: int | None,
        max_total_relative_path_bytes: int | None,
    ) -> None:
        self._monitor = _DirectoryMutationMonitor()
        self._any_watches: set[int] = set()
        self._self_watches: set[int] = set()
        self._named_watches: dict[int, set[bytes]] = {}
        self._nested_directory_identities: dict[
            tuple[str, ...], tuple[int, int, int]
        ] = {}
        self._verify_root_path()
        for descriptor in self._root.descriptors:
            self._watch_self(descriptor)
        for index, component_name in enumerate(self._root.names):
            self._watch_named_children(
                self._root.descriptors[index],
                (component_name,),
            )
        for relative_path in regular_paths:
            self._watch_regular(relative_path)
        for relative_path in optional_regular_paths:
            self._watch_optional_regular(relative_path)
        for relative_path in recursive_directories:
            self._watch_recursive(
                relative_path,
                max_files=max_files,
                max_directories=max_directories,
                max_total_file_bytes=max_total_file_bytes,
                max_total_relative_path_bytes=(
                    max_total_relative_path_bytes
                ),
            )
        for relative_path, names in (filtered_directories or {}).items():
            self._watch_filtered_directory(relative_path, names)
        self._verify_root_path()
        self.verify_unchanged()
        monitor = self._monitor
        if monitor is None:
            raise ImmutableArtifactError("protected input monitor is closed")
        any_watches = frozenset(self._any_watches)
        named_watches = {
            watch: frozenset(names)
            for watch, names in self._named_watches.items()
        }
        self_watches = frozenset(self._self_watches)
        monitor.set_close_verifier(
            lambda: _require_pinned_path_set_events_quiet(
                monitor,
                any_watches=any_watches,
                named_watches=named_watches,
                self_watches=self_watches,
            )
        )

    def _remember_nested_directory(
        self, parts: tuple[str, ...], descriptor: int,
    ) -> None:
        if not parts:
            return
        observed = _node_identity(os.fstat(descriptor))
        expected = self._nested_directory_identities.get(parts)
        if expected is not None and observed != expected:
            raise ImmutableArtifactError(
                "protected input nested ancestor changed while arming: "
                f"{PurePosixPath(*parts).as_posix()}"
            )
        self._nested_directory_identities[parts] = observed

    def _remember_nested_chain(
        self,
        descriptors: tuple[int, ...],
        names: tuple[str, ...],
    ) -> None:
        if len(descriptors) != len(names) + 1:
            raise ImmutableArtifactError(
                "protected input nested ancestor chain is invalid"
            )
        for depth in range(1, len(descriptors)):
            self._remember_nested_directory(
                names[:depth], descriptors[depth],
            )

    def _verify_nested_directory_identities(self) -> None:
        for parts, expected in sorted(
            self._nested_directory_identities.items(),
            key=lambda item: (len(item[0]), item[0]),
        ):
            relative_path = PurePosixPath(*parts).as_posix()
            try:
                directory = _open_pinned_relative_directory(
                    self._root.descriptor, parts,
                )
            except (OSError, ImmutableArtifactError) as exc:
                raise ImmutableArtifactError(
                    "protected input nested ancestor changed during "
                    f"downstream use: {relative_path}"
                ) from exc
            try:
                if _node_identity(os.fstat(directory.descriptor)) != expected:
                    raise ImmutableArtifactError(
                        "protected input nested ancestor changed during "
                        f"downstream use: {relative_path}"
                    )
                directory.verify_path()
            finally:
                directory.close()

    def _watch_self(self, descriptor: int) -> None:
        self._self_watches.add(self._monitor.watch_directory(descriptor))

    def _watch_any_directory(self, descriptor: int) -> None:
        watch = self._monitor.watch_directory(descriptor)
        self._any_watches.add(watch)
        self._self_watches.discard(watch)
        self._named_watches.pop(watch, None)

    def _watch_regular_descriptor(self, descriptor: int) -> None:
        watch = self._monitor.watch_regular(descriptor)
        self._any_watches.add(watch)
        self._self_watches.discard(watch)
        self._named_watches.pop(watch, None)

    def _watch_regular(self, relative_path: str) -> None:
        pinned = _open_relative_regular_at(self._root.descriptor, relative_path)
        try:
            self._remember_nested_chain(
                pinned.directory_descriptors,
                pinned.directory_names,
            )
            self._watch_regular_descriptor(pinned.descriptor)
            pinned.verify_path()
        finally:
            pinned.close()

    def _watch_named_children(
        self, descriptor: int, names: Iterable[str],
    ) -> None:
        protected_names: set[bytes] = set()
        for name in names:
            child = safe_relative_path(name)
            if len(child.parts) != 1:
                raise ImmutableArtifactError(
                    "filtered mutation-monitor child must be one component"
                )
            protected_names.add(os.fsencode(name))
        watch = self._monitor.watch_directory(descriptor)
        if watch not in self._any_watches:
            self._self_watches.discard(watch)
            self._named_watches.setdefault(watch, set()).update(
                protected_names,
            )

    def _watch_optional_regular(self, relative_path: str) -> None:
        """Protect either one regular generation or its descriptor-bound absence."""
        relative = safe_relative_path(relative_path)
        descriptors = [os.dup(self._root.descriptor)]
        try:
            for index, part in enumerate(relative.parts[:-1]):
                self._watch_named_children(descriptors[-1], (part,))
                try:
                    descriptors.append(_open_dir_at(descriptors[-1], part))
                except FileNotFoundError:
                    return
                except OSError as exc:
                    raise ImmutableArtifactError(
                        "optional regular-file ancestor is not a no-follow "
                        f"directory: {relative_path}"
                    ) from exc
                self._remember_nested_directory(
                    relative.parts[: index + 1], descriptors[-1],
                )
            leaf_name = relative.parts[-1]
            self._watch_named_children(descriptors[-1], (leaf_name,))
            try:
                metadata = os.stat(
                    leaf_name,
                    dir_fd=descriptors[-1],
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            except OSError as exc:
                raise ImmutableArtifactError(
                    f"cannot inspect optional regular file: {relative_path}"
                ) from exc
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ImmutableArtifactError(
                    "optional artifact is not a single-link regular file: "
                    f"{relative_path}"
                )
            self._watch_regular(relative_path)
        finally:
            _close_descriptors(
                reversed(descriptors),
                label="optional regular monitor traversal close",
            )

    def _watch_recursive(
        self,
        relative_path: str,
        *,
        max_files: int | None,
        max_directories: int | None,
        max_total_file_bytes: int | None,
        max_total_relative_path_bytes: int | None,
    ) -> None:
        if relative_path == ".":
            directory = _PinnedDirectory(
                (os.dup(self._root.descriptor),), ()
            )
        else:
            relative = safe_relative_path(relative_path)
            directory = _open_pinned_relative_directory(
                self._root.descriptor, relative.parts,
            )
        try:
            self._remember_nested_chain(
                directory.descriptors, directory.names,
            )
            files, _directories = _walk_tree_descriptor(
                directory.descriptor,
                observe_directory=self._watch_any_directory,
                max_files=max_files,
                max_directories=max_directories,
                max_total_file_bytes=max_total_file_bytes,
                max_total_relative_path_bytes=max_total_relative_path_bytes,
            )
            for child_path in sorted(files):
                pinned = _open_relative_regular_at(
                    directory.descriptor, child_path,
                )
                try:
                    self._watch_regular_descriptor(pinned.descriptor)
                    pinned.verify_path()
                finally:
                    pinned.close()
            directory.verify_path()
            self._verify_root_path()
        finally:
            directory.close()

    def _watch_filtered_directory(
        self, relative_path: str, names: Iterable[str],
    ) -> None:
        relative = safe_relative_path(relative_path)
        directory = _open_pinned_relative_directory(
            self._root.descriptor, relative.parts,
        )
        try:
            self._remember_nested_chain(
                directory.descriptors, directory.names,
            )
            self._watch_named_children(directory.descriptor, names)
            directory.verify_path()
        finally:
            directory.close()

    @_translate_descriptor_resource_errors("pinned path-set monitor verification")
    def verify_unchanged(self) -> None:
        self._verify_root_path()
        self._verify_nested_directory_identities()
        monitor = self._monitor
        if monitor is None:
            raise ImmutableArtifactError("protected input monitor is closed")
        _require_pinned_path_set_events_quiet(
            monitor,
            any_watches=self._any_watches,
            named_watches=self._named_watches,
            self_watches=self._self_watches,
        )
        self._verify_nested_directory_identities()
        self._verify_root_path()

    def close(self) -> None:
        monitor = getattr(self, "_monitor", None)
        reader = getattr(self, "_reader", None)
        owns_root = getattr(self, "_owns_root", False)
        self._monitor = None
        self._reader = None
        self._root = None
        self._nested_directory_identities = {}
        self._owns_root = False
        closers: list[Callable[[], None]] = []
        if monitor is not None:
            closers.append(monitor.close)
        if reader is not None and owns_root:
            closers.append(reader.close)
        if closers:
            _close_callables(closers, label="protected input monitor close")

    def __enter__(self) -> PinnedPathSetMonitor:
        try:
            self.verify_unchanged()
        except BaseException:
            _close_callables(
                (self.close,), label="protected input monitor entry close"
            )
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            try:
                self.verify_unchanged()
            except BaseException as verification_error:
                try:
                    exc.add_note(
                        "protected input revalidation failed: "
                        f"{verification_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            try:
                self.close()
            except BaseException as close_error:
                try:
                    exc.add_note(
                        f"protected input close failed: {close_error}"
                    )
                except (AttributeError, TypeError):
                    pass
            return False
        try:
            self.verify_unchanged()
        finally:
            self.close()
        return False


def _snapshot_tree_descriptor(
    root_descriptor: int,
    *,
    max_files: int | None = None,
    max_directories: int | None = None,
    max_total_file_bytes: int | None = None,
    max_total_relative_path_bytes: int | None = None,
) -> _TreeSnapshot:
    if all(
        limit is None
        for limit in (
            max_files,
            max_directories,
            max_total_file_bytes,
            max_total_relative_path_bytes,
        )
    ):
        return _walk_tree_descriptor(root_descriptor)
    return _walk_tree_descriptor(
        root_descriptor,
        max_files=max_files,
        max_directories=max_directories,
        max_total_file_bytes=max_total_file_bytes,
        max_total_relative_path_bytes=max_total_relative_path_bytes,
    )


def _arm_directory_mutation_monitor(
    root_descriptor: int,
    *,
    max_files: int | None = None,
    max_directories: int | None = None,
    max_total_file_bytes: int | None = None,
    max_total_relative_path_bytes: int | None = None,
) -> _DirectoryMutationMonitor:
    monitor = _DirectoryMutationMonitor()
    try:
        walk_kwargs = {
            "max_files": max_files,
            "max_directories": max_directories,
            "max_total_file_bytes": max_total_file_bytes,
            "max_total_relative_path_bytes": max_total_relative_path_bytes,
        }
        if all(value is None for value in walk_kwargs.values()):
            files, _directories = _walk_tree_descriptor(
                root_descriptor, observe_directory=monitor.watch_directory,
            )
        else:
            files, _directories = _walk_tree_descriptor(
                root_descriptor,
                observe_directory=monitor.watch_directory,
                **walk_kwargs,
            )
        for relative_path in sorted(files):
            pinned = _open_relative_regular_at(root_descriptor, relative_path)
            try:
                monitor.watch_regular(pinned.descriptor)
                pinned.verify_path()
            finally:
                pinned.close()
        monitor.require_quiet()
        return monitor
    except BaseException:
        monitor.cancel_close_verification()
        monitor.close()
        raise


def _snapshot_pinned_tree_at(
    root_descriptor: int,
    *,
    require_readonly_files: bool,
    require_readonly_directories: bool,
    max_files: int | None = None,
    max_directories: int | None = None,
    max_total_file_bytes: int | None = None,
    max_total_relative_path_bytes: int | None = None,
) -> PinnedTreeSnapshot:
    limits = {
        "max_files": max_files,
        "max_directories": max_directories,
        "max_total_file_bytes": max_total_file_bytes,
        "max_total_relative_path_bytes": max_total_relative_path_bytes,
    }
    monitor = _arm_directory_mutation_monitor(root_descriptor, **limits)
    try:
        root_before = _snapshot_fields(os.fstat(root_descriptor))
        if not stat.S_ISDIR(root_before[2]):
            raise ImmutableArtifactError("pinned tree root is not a directory")
        if require_readonly_directories and root_before[2] & 0o222:
            raise ImmutableArtifactError("pinned tree root directory is writable")
        files, directories = _snapshot_tree_descriptor(
            root_descriptor, **limits,
        )
        if directories.get(".") != root_before:
            raise ImmutableArtifactError("pinned tree root changed during inventory")
        for relative_path, snapshot in files.items():
            if snapshot[3] != 1:
                raise ImmutableArtifactError(
                    f"hard-linked file in pinned tree: {relative_path}"
                )
            if require_readonly_files and snapshot[2] & 0o222:
                raise ImmutableArtifactError(
                    f"writable file in pinned tree: {relative_path}"
                )
        for relative_path, snapshot in directories.items():
            if require_readonly_directories and snapshot[2] & 0o222:
                raise ImmutableArtifactError(
                    f"writable directory in pinned tree: {relative_path}"
                )
        if _snapshot_fields(os.fstat(root_descriptor)) != root_before:
            raise ImmutableArtifactError("pinned tree root changed during inventory")
        monitor.require_quiet()
        return PinnedTreeSnapshot(
            files=tuple(sorted(files.items())),
            directories=tuple(sorted(directories.items())),
        )
    finally:
        monitor.close()


def _expected_tree_limits(
    expected: Mapping[str, ImmutableFile],
) -> dict[str, int]:
    expected_directories = _expected_directories(expected)
    return {
        "max_files": len(expected),
        "max_directories": len(expected_directories) + 1,
        "max_total_file_bytes": sum(file.size_bytes for file in expected.values()),
        "max_total_relative_path_bytes": (
            sum(len(os.fsencode(relative_path)) for relative_path in expected)
            + sum(
                len(os.fsencode(relative_path))
                for relative_path in expected_directories
            )
        ),
    }


def _available_file_descriptor_capacity() -> int | None:
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, OSError, ValueError) as exc:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: unable to inspect RLIMIT_NOFILE"
        ) from exc
    if soft_limit == resource.RLIM_INFINITY:
        return None
    if isinstance(soft_limit, bool) or not isinstance(soft_limit, int) or soft_limit < 0:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: invalid RLIMIT_NOFILE={soft_limit!r}"
        )
    try:
        with os.scandir("/proc/self/fd") as entries:
            open_descriptors = sum(1 for _entry in entries)
    except OSError as exc:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: unable to count open descriptors"
        ) from exc
    return max(0, soft_limit - open_descriptors)


def require_file_descriptor_headroom(required: int, *, label: str) -> None:
    """Fail typed when a bounded retained-capability plan cannot fit.

    This is a conservative preflight, not a reservation: callers must still
    translate a later ``EMFILE``/``ENFILE`` through the same resource taxonomy.
    """
    if type(required) is not int or required < 0:
        raise ImmutableArtifactError(
            "required file-descriptor headroom must be a nonnegative integer"
        )
    if type(label) is not str or not label or "\x00" in label:
        raise ImmutableArtifactError(
            "file-descriptor headroom label must be a nonempty string"
        )
    available = _available_file_descriptor_capacity()
    if available is not None and available < required:
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: {label}: available file descriptors "
            f"{available} < required headroom {required}"
        )


def _snapshot_descriptor_headroom(
    expected: Mapping[str, ImmutableFile],
) -> int:
    maximum_depth = max(
        (len(safe_relative_path(relative_path).parts) for relative_path in expected),
        default=1,
    )
    return maximum_depth + 32


@_translate_descriptor_resource_errors("snapshot-tree verification")
def _verify_snapshot_contents_at(
    root_descriptor: int,
    expected: dict[str, ImmutableFile],
    *,
    require_readonly_directories: bool,
) -> _TreeSnapshot:
    """Verify large trees with one hash and bounded simultaneous descriptors."""
    available_descriptors = _available_file_descriptor_capacity()
    required_descriptors = _snapshot_descriptor_headroom(expected)
    if (
        available_descriptors is not None
        and available_descriptors < required_descriptors
    ):
        raise ImmutableArtifactResourceError(
            f"{VERIFY_RESOURCE_BOUND}: available file descriptors "
            f"{available_descriptors} < required headroom {required_descriptors}"
        )
    limits = _expected_tree_limits(expected)
    expected_files = set(expected)
    expected_directories = _expected_directories(expected_files)
    root_before = _snapshot_fields(os.fstat(root_descriptor))
    if require_readonly_directories and root_before[2] & 0o222:
        raise ImmutableArtifactError("finalized publication root is writable")
    first_files, first_directories = _snapshot_tree_descriptor(
        root_descriptor, **limits,
    )
    if (
        set(first_files) != expected_files
        or set(first_directories) - {"."} != expected_directories
    ):
        raise ImmutableArtifactError("immutable tree inventory mismatch")
    for relative_directory in expected_directories:
        snapshot = first_directories[relative_directory]
        if require_readonly_directories and snapshot[2] & 0o222:
            raise ImmutableArtifactError(
                f"finalized publication directory is writable: {relative_directory}"
            )
    hashed: dict[str, _Snapshot] = {}
    for relative_path, file in expected.items():
        pinned = _open_relative_regular_at(root_descriptor, relative_path)
        try:
            if not _verify_open_file(pinned.descriptor, file):
                raise ImmutableArtifactError(
                    f"immutable file mismatch: {relative_path}"
                )
            pinned.verify_path()
            snapshot = _snapshot_fields(os.fstat(pinned.descriptor))
            if snapshot != first_files[relative_path]:
                raise ImmutableArtifactError(
                    f"immutable file changed during verification: {relative_path}"
                )
            hashed[relative_path] = snapshot
        finally:
            pinned.close()
    second_files, second_directories = _snapshot_tree_descriptor(
        root_descriptor, **limits,
    )
    if (
        set(second_files) != expected_files
        or set(second_directories) - {"."} != expected_directories
    ):
        raise ImmutableArtifactError("immutable tree inventory changed during closure")
    for relative_path, snapshot in hashed.items():
        if second_files[relative_path] != snapshot:
            raise ImmutableArtifactError(
                f"immutable file changed before verification closure: {relative_path}"
            )
        pinned = _open_relative_regular_at(root_descriptor, relative_path)
        try:
            if _snapshot_fields(os.fstat(pinned.descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"named regular-file path changed: {relative_path}"
                )
            pinned.verify_path()
        finally:
            pinned.close()
    for relative_directory, snapshot in first_directories.items():
        if relative_directory == ".":
            continue
        if second_directories[relative_directory] != snapshot:
            raise ImmutableArtifactError(
                f"immutable directory changed during verification: {relative_directory}"
            )
        directory = _open_pinned_relative_directory(
            root_descriptor, safe_relative_path(relative_directory).parts,
        )
        try:
            if _snapshot_fields(os.fstat(directory.descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"named directory path changed: {relative_directory}"
                )
            directory.verify_path()
        finally:
            directory.close()
    final_files, final_directories = _snapshot_tree_descriptor(
        root_descriptor, **limits,
    )
    if (
        set(final_files) != expected_files
        or set(final_directories) - {"."} != expected_directories
    ):
        raise ImmutableArtifactError("immutable tree inventory changed during final closure")
    for relative_path, snapshot in hashed.items():
        if final_files[relative_path] != snapshot:
            raise ImmutableArtifactError(
                f"immutable file changed before verification closure: {relative_path}"
            )
    for relative_directory, snapshot in first_directories.items():
        if final_directories[relative_directory] != snapshot:
            raise ImmutableArtifactError(
                "immutable directory changed before verification closure: "
                f"{relative_directory}"
            )
    root_after = _snapshot_fields(os.fstat(root_descriptor))
    if root_after != root_before:
        raise ImmutableArtifactError("publication root changed during verification")
    return final_files, final_directories


@_translate_descriptor_resource_errors("snapshot closure")
def _verify_snapshot_closure_at(
    root_descriptor: int,
    expected: dict[str, ImmutableFile],
    *,
    require_readonly_directories: bool,
) -> _TreeSnapshot:
    monitor = _arm_directory_mutation_monitor(
        root_descriptor, **_expected_tree_limits(expected),
    )
    try:
        snapshot = _verify_snapshot_contents_at(
            root_descriptor,
            expected,
            require_readonly_directories=require_readonly_directories,
        )
        # This nonblocking read to EAGAIN is the bounded verifier's final
        # linearization point for local write and namespace syscalls.
        monitor.require_quiet()
        return snapshot
    except BaseException:
        monitor.cancel_close_verification()
        raise
    finally:
        monitor.close()


@_translate_descriptor_resource_errors("immutable tree verification")
def _verify_published_files_at(
    root_descriptor: int,
    files: Iterable[ImmutableFile],
    *,
    require_readonly_directories: bool = True,
) -> _TreeSnapshot:
    expected = _material_map(files)
    expected_files = set(expected)
    expected_directories = _expected_directories(expected_files)
    available_descriptors = _available_file_descriptor_capacity()
    estimated_retained = (
        len(expected_files)
        + len(expected_directories)
        + _snapshot_descriptor_headroom(expected)
    )
    use_snapshot_closure = len(expected) > _SNAPSHOT_CLOSURE_THRESHOLD
    if (
        available_descriptors is not None
        and estimated_retained >= available_descriptors
    ):
        use_snapshot_closure = True
    if use_snapshot_closure:
        return _verify_snapshot_closure_at(
            root_descriptor,
            expected,
            require_readonly_directories=require_readonly_directories,
        )
    root_before = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ImmutableArtifactError("publication root descriptor is not a directory")

    def require_inventory() -> tuple[set[str], set[str]]:
        observed, observed_directories = _tree_inventory_descriptor(
            root_descriptor,
            expected_files=expected_files,
            expected_directories=expected_directories,
            mismatch_label="immutable tree inventory mismatch",
        )
        if observed != expected_files or observed_directories != expected_directories:
            raise ImmutableArtifactError(
                "immutable tree inventory mismatch: "
                "missing_files="
                f"{_bounded_inventory_difference(expected_files, observed)}, "
                "missing_directories="
                f"{_bounded_inventory_difference(expected_directories, observed_directories)}"
            )
        return observed, observed_directories

    if require_readonly_directories and root_before.st_mode & 0o222:
        raise ImmutableArtifactError("finalized publication root is writable")
    _, observed_directories = require_inventory()
    pinned_directories: list[tuple[str, int, _Snapshot]] = []
    # Retain one descriptor per file, but not one complete ancestor chain per
    # file. This keeps verification below ordinary RLIMIT_NOFILE values while
    # still pinning every hashed inode through closure.
    pinned_files: list[tuple[str, ImmutableFile, _PinnedRegular, _Snapshot]] = []
    try:
        for relative_directory in sorted(observed_directories):
            directory = _open_pinned_relative_directory(
                root_descriptor,
                safe_relative_path(relative_directory).parts,
            )
            try:
                metadata = os.fstat(directory.descriptor)
                if require_readonly_directories and metadata.st_mode & 0o222:
                    raise ImmutableArtifactError(
                        "finalized publication directory is writable: "
                        f"{relative_directory}"
                    )
                directory.verify_path()
                retained = os.dup(directory.descriptor)
                pinned_directories.append(
                    (relative_directory, retained, _snapshot_fields(metadata))
                )
            finally:
                directory.close()

        for relative_path, file in expected.items():
            pinned = _open_relative_regular_at(root_descriptor, relative_path)
            try:
                if not _verify_open_file(pinned.descriptor, file):
                    raise ImmutableArtifactError(
                        f"immutable file mismatch: {relative_path}"
                    )
                pinned.verify_path()
                snapshot = _snapshot_fields(os.fstat(pinned.descriptor))
                # The leaf descriptor remains pinned. Final path identity is
                # re-opened descriptor-relatively at closure.
                _close_descriptors(reversed(pinned.directory_descriptors))
                pinned.directory_descriptors = ()
                pinned_files.append((relative_path, file, pinned, snapshot))
            except BaseException:
                pinned.close()
                raise
        require_inventory()

        # A second stable hash closes mutations that occur after a file's first
        # hash while later files are being inspected. Metadata equality also
        # detects mutate-and-restore attempts because ctime is not restorable by
        # an unprivileged same-UID writer.
        for relative_path, file, pinned, snapshot in pinned_files:
            if _snapshot_fields(os.fstat(pinned.descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"immutable file changed after hashing: {relative_path}"
                )
            if not _verify_open_file(pinned.descriptor, file):
                raise ImmutableArtifactError(
                    f"immutable file changed after hashing: {relative_path}"
                )
            if _snapshot_fields(os.fstat(pinned.descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"immutable file changed during closing hash: {relative_path}"
                )

        require_inventory()
        for relative_path, _file, pinned, snapshot in pinned_files:
            if _snapshot_fields(os.fstat(pinned.descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"immutable file changed before verification closure: {relative_path}"
                )
            named = _open_relative_regular_at(root_descriptor, relative_path)
            try:
                if _node_identity(os.fstat(named.descriptor)) != _node_identity(
                    os.fstat(pinned.descriptor)
                ):
                    raise ImmutableArtifactError(
                        f"named regular-file path changed: {relative_path}"
                    )
                named.verify_path()
            finally:
                named.close()

        require_inventory()
        for relative_directory, descriptor, snapshot in pinned_directories:
            if _snapshot_fields(os.fstat(descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"immutable directory changed during verification: {relative_directory}"
                )
            named = _open_pinned_relative_directory(
                root_descriptor, safe_relative_path(relative_directory).parts
            )
            try:
                if _node_identity(os.fstat(named.descriptor)) != _node_identity(
                    os.fstat(descriptor)
                ):
                    raise ImmutableArtifactError(
                        f"named directory path changed: {relative_directory}"
                    )
                named.verify_path()
            finally:
                named.close()
        # Recheck every retained inode after all sequential named-path checks.
        # Replacing an earlier name unlinks or changes ctime on its retained
        # inode and is therefore detected here.
        for relative_path, _file, pinned, snapshot in pinned_files:
            if _snapshot_fields(os.fstat(pinned.descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"immutable file changed at verification closure: {relative_path}"
                )
        for relative_directory, descriptor, snapshot in pinned_directories:
            if _snapshot_fields(os.fstat(descriptor)) != snapshot:
                raise ImmutableArtifactError(
                    f"immutable directory changed at verification closure: {relative_directory}"
                )
    finally:
        _close_callables(
            [
                (lambda pinned=pinned: pinned.close())
                for _relative_path, _file, pinned, _snapshot in reversed(pinned_files)
            ]
            + [
                (
                    lambda descriptor=descriptor: _close_descriptors(
                        (descriptor,), label="verification directory close"
                    )
                )
                for _relative_directory, descriptor, _snapshot in reversed(
                    pinned_directories
                )
            ],
            label="published-tree verification cleanup",
        )
    root_after = os.fstat(root_descriptor)
    if require_readonly_directories and root_after.st_mode & 0o222:
        raise ImmutableArtifactError("finalized publication root became writable")
    if _snapshot_fields(root_after) != _snapshot_fields(root_before):
        raise ImmutableArtifactError("publication root changed during verification")
    return (
        {
            relative_path: snapshot
            for relative_path, _file, _pinned, snapshot in pinned_files
        },
        {
            ".": _snapshot_fields(root_after),
            **{
                relative_directory: snapshot
                for relative_directory, _descriptor, snapshot in pinned_directories
            },
        },
    )


def _baseline_tree_limits(
    baseline: _TreeSnapshot,
) -> dict[str, int]:
    files, directories = baseline
    if "." not in directories:
        raise ImmutableArtifactError(
            "authorization baseline omits the root directory"
        )
    return {
        "max_files": len(files),
        "max_directories": len(directories),
        "max_total_file_bytes": sum(
            metadata[4] for metadata in files.values()
        ),
        "max_total_relative_path_bytes": sum(
            len(os.fsencode(relative_path)) for relative_path in files
        ) + sum(
            len(os.fsencode(relative_path))
            for relative_path in directories
            if relative_path != "."
        ),
    }


def _verify_snapshot_against_at(
    root_descriptor: int, baseline: _TreeSnapshot,
) -> None:
    """Close a verified tree after authorization without rehashing it."""
    limits = _baseline_tree_limits(baseline)
    first = _snapshot_tree_descriptor(root_descriptor, **limits)
    if first != baseline:
        raise ImmutableArtifactError(
            "immutable tree changed after authorization"
        )
    second = _snapshot_tree_descriptor(root_descriptor, **limits)
    if second != first:
        raise ImmutableArtifactError(
            "immutable tree changed during authorization closure"
        )


def verify_published_files(
    root: Path,
    files: Iterable[ImmutableFile],
    *,
    require_readonly_directories: bool = True,
) -> None:
    material = tuple(files)
    try:
        pinned_root = _open_pinned_directory(root)
    except (OSError, ImmutableArtifactError) as exc:
        raise ImmutableArtifactError(
            f"publication root is absent, replaced, or symlinked: {root}"
        ) from exc
    try:
        _verify_published_files_at(
            pinned_root.descriptor,
            material,
            require_readonly_directories=require_readonly_directories,
        )
        pinned_root.verify_path()
    finally:
        pinned_root.close()


def _validate_publication_name(name: str) -> None:
    try:
        path = safe_relative_path(name)
    except ImmutableArtifactError as exc:
        raise ImmutableArtifactError(f"unsafe publication name: {name!r}") from exc
    encoded = os.fsencode(name)
    if (
        len(path.parts) != 1
        or len(encoded) > _MAX_FINAL_NAME_BYTES
        or not encoded
        or encoded.startswith(b".")
        or any(byte < 0x20 or byte == 0x7F for byte in encoded)
        or name.endswith(
            (".staging", ".m5-partial", ".m6-partial", ".ncu-details-partial")
        )
        or ".partial-" in name
    ):
        raise ImmutableArtifactError(
            f"{NAMESPACE_RESERVED_COMPONENT}: {name!r}"
        )


def _durability_proof_at(
    publication_root_descriptor: int,
    tree_root_descriptor: int,
    material: tuple[ImmutableFile, ...],
) -> _TreeSnapshot:
    """Prove a pre-existing winner was durably committed before acceptance."""
    expected = _material_map(material)
    expected_directories = _expected_directories(expected)
    # Hash and close the exact tree before fsyncing the inodes that will be
    # accepted.  The metadata baseline prevents a replacement between those
    # operations from turning an unverified inode into a durable winner.
    snapshot = _verify_published_files_at(tree_root_descriptor, material)
    for relative_path in sorted(expected):
        pinned = _open_relative_regular_at(
            tree_root_descriptor, relative_path,
        )
        try:
            _fsync_or_fail(pinned.descriptor, f"winner file {relative_path}")
            pinned.verify_path()
        finally:
            pinned.close()
    for relative_directory in sorted(
        expected_directories,
        key=lambda item: len(safe_relative_path(item).parts),
        reverse=True,
    ):
        directory = _open_pinned_relative_directory(
            tree_root_descriptor,
            safe_relative_path(relative_directory).parts,
        )
        try:
            _fsync_or_fail(directory.descriptor, f"winner directory {relative_directory}")
            directory.verify_path()
        finally:
            directory.close()
    _fsync_or_fail(tree_root_descriptor, "winner tree root")
    _fsync_or_fail(publication_root_descriptor, "winner publication root")
    _verify_snapshot_against_at(tree_root_descriptor, snapshot)
    return snapshot


def _accept_exact_concurrent_winner(
    root_descriptor: int,
    stage_name: str,
    final_name: str,
    material: tuple[ImmutableFile, ...],
) -> tuple[_TreeSnapshot, _PinnedDirectory] | None:
    final_mode = _entry_mode(root_descriptor, final_name)
    if final_mode is None:
        return None
    if _entry_mode(root_descriptor, stage_name) is not None:
        raise ImmutableArtifactError(
            "ambiguous final and deterministic staging directories coexist"
        )
    if not stat.S_ISDIR(final_mode) or stat.S_ISLNK(final_mode):
        raise ImmutableArtifactError("concurrent final winner is not a real directory")
    pinned_final = _open_pinned_relative_directory(
        root_descriptor, (final_name,)
    )
    transferred = False
    try:
        snapshot = _durability_proof_at(
            root_descriptor, pinned_final.descriptor, material
        )
        pinned_final.verify_path()
        if _entry_mode(root_descriptor, stage_name) is not None:
            raise ImmutableArtifactError(
                "deterministic stage reappeared beside concurrent final winner"
            )
        pinned_final.verify_path()
        # The caller owns the pinned winner through authorization and final
        # closure; do not reopen the name after this transfer.
        transferred = True
        return snapshot, pinned_final
    finally:
        if not transferred:
            pinned_final.close()


@_translate_descriptor_resource_errors("directory publication")
def publish_directory(
    output_root: Path,
    final_name: str,
    files: Iterable[ImmutableFile],
    *,
    before_commit: Callable[[Path, int], None] | None = None,
    after_commit: Callable[[Path, int], None] | None = None,
    pinned_root: PinnedDirectoryReader | None = None,
    accept_existing: bool = True,
) -> Path:
    """Publish one immutable tree through a locked, pinned deterministic stage.

    ``accept_existing`` preserves the historical idempotent exact-winner mode by
    default.  Attempt-scoped and merge publications set it to ``False`` so a
    pre-existing or racing destination generation is always a hard failure.
    ``after_commit`` runs after durable committed-tree closure while rollback
    guards can still retire this invocation's failed generation.
    """
    _validate_publication_name(final_name)
    for callback_name, callback in (
        ("before_commit", before_commit),
        ("after_commit", after_commit),
    ):
        if callback is not None and not callable(callback):
            raise ImmutableArtifactError(f"{callback_name} must be callable")
    if type(accept_existing) is not bool:
        raise ImmutableArtifactError("accept_existing must be boolean")
    material = tuple(files)
    expected = _material_map(material)
    root = Path(os.path.abspath(os.fspath(output_root)))
    owns_root = pinned_root is None
    if pinned_root is None:
        ensure_directory(root)
        pinned_root = _open_pinned_directory(root)
    else:
        pinned_root.verify_path()
    stage_name = f".{final_name}.staging"
    final = root / final_name
    pinned_stage: _PinnedDirectory | None = None
    rollback_root = rollback_stage = -1
    committed = False

    def authorize() -> None:
        if before_commit is not None:
            before_commit(root, pinned_root.descriptor)
        pinned_root.verify_path()

    def accept_winner() -> bool:
        if _entry_mode(pinned_root.descriptor, final_name) is None:
            return False
        if not accept_existing:
            raise ImmutableArtifactError(
                f"publication already exists or raced: {final_name}"
            )
        accepted = _accept_exact_concurrent_winner(
            pinned_root.descriptor, stage_name, final_name, material
        )
        if accepted is None:
            return False
        winner_snapshot, winner = accepted
        try:
            # Retain the winner descriptor across authorization so a same-byte
            # replacement cannot be accepted as a freshly verified tree.
            authorize()
            _verify_snapshot_against_at(winner.descriptor, winner_snapshot)
            if _entry_mode(pinned_root.descriptor, stage_name) is not None:
                raise ImmutableArtifactError(
                    "deterministic stage reappeared after winner closure"
                )
            _verify_named_directory(
                pinned_root.descriptor, final_name, winner.descriptor
            )
            pinned_root.verify_path()
        finally:
            winner.close()
        return True

    try:
        # The deterministic stage inode is the per-final-name ownership lock;
        # unrelated publication names never serialize on the root directory.
        pinned_root.verify_path()
        authorize()
        root_descriptor = pinned_root.descriptor
        final_mode = _entry_mode(root_descriptor, final_name)
        stage_mode = _entry_mode(root_descriptor, stage_name)
        if final_mode is not None and stage_mode is not None:
            raise ImmutableArtifactError(
                "ambiguous final and deterministic staging directories coexist"
            )
        if final_mode is not None:
            if accept_winner():
                return final
            raise ImmutableArtifactError("final publication vanished during verification")
        if stage_mode is None:
            try:
                os.mkdir(stage_name, mode=0o755, dir_fd=root_descriptor)
                _fsync_or_fail(root_descriptor, "staging parent")
            except FileExistsError:
                pass
            stage_mode = _entry_mode(root_descriptor, stage_name)
        if stage_mode is None or not stat.S_ISDIR(stage_mode) or stat.S_ISLNK(stage_mode):
            raise ImmutableArtifactError(
                "deterministic stage is not a real directory"
            )

        pinned_stage = _open_pinned_relative_directory(
            root_descriptor, (stage_name,)
        )
        _lock_directory(pinned_stage.descriptor, stage_name, blocking=False)
        pinned_stage.verify_path()
        if _entry_mode(root_descriptor, final_name) is not None:
            raise ImmutableArtifactError(
                "concurrent final appeared beside the locked deterministic stage"
            )

        expected_files = set(expected)
        expected_directories = _expected_directories(expected_files)
        recoverable_partials = {
            relative_path: PurePosixPath(
                *safe_relative_path(relative_path).parts[:-1],
                _partial_name(
                    safe_relative_path(relative_path).parts[-1]
                ),
            ).as_posix()
            for relative_path in expected_files
        }
        observed, _ = _tree_inventory_descriptor(
            pinned_stage.descriptor,
            expected_files=expected_files,
            expected_directories=expected_directories,
            mismatch_label="deterministic stage inventory mismatch",
            allowed_files=set(recoverable_partials.values()),
        )
        for relative_path, partial_path in recoverable_partials.items():
            if relative_path in observed and partial_path in observed:
                raise ImmutableArtifactError(
                    "ambiguous final and deterministic partial coexist: "
                    f"{relative_path}"
                )
        for relative_path in sorted(expected):
            file = expected[relative_path]
            if relative_path in observed:
                pinned = _open_relative_regular_at(
                    pinned_stage.descriptor, relative_path
                )
                try:
                    if not _verify_open_file(pinned.descriptor, file):
                        raise ImmutableArtifactError(
                            f"staged file collision: {relative_path}"
                        )
                    pinned.verify_path()
                finally:
                    pinned.close()
            else:
                _install_staged_file_at(pinned_stage.descriptor, file)
            pinned_stage.verify_path()

        observed, observed_directories = _tree_inventory_descriptor(
            pinned_stage.descriptor,
            expected_files=expected_files,
            expected_directories=expected_directories,
            mismatch_label="deterministic stage inventory mismatch",
        )
        if observed != expected_files or observed_directories != expected_directories:
            raise ImmutableArtifactError(
                "deterministic stage inventory mismatch: "
                "missing_files="
                f"{_bounded_inventory_difference(expected_files, observed)}, "
                "missing_directories="
                f"{_bounded_inventory_difference(expected_directories, observed_directories)}"
            )
        _verify_published_files_at(
            pinned_stage.descriptor,
            material,
            require_readonly_directories=False,
        )
        for relative_directory in sorted(
            expected_directories,
            key=lambda item: len(safe_relative_path(item).parts),
            reverse=True,
        ):
            directory = _open_pinned_relative_directory(
                pinned_stage.descriptor,
                safe_relative_path(relative_directory).parts,
            )
            try:
                os.fchmod(directory.descriptor, 0o555)
                _fsync_or_fail(directory.descriptor, f"staged directory {relative_directory}")
                directory.verify_path()
            finally:
                directory.close()
        os.fchmod(pinned_stage.descriptor, 0o555)
        _fsync_or_fail(pinned_stage.descriptor, "staged tree root")
        staged_snapshot = _verify_published_files_at(pinned_stage.descriptor, material)
        pinned_stage.verify_path()

        authorize()
        # Authorization may inspect arbitrary external state and can widen the
        # commit window.  Reclose the exact staged tree after the callback and
        # immediately before the namespace commit without rehashing it.
        _verify_snapshot_against_at(pinned_stage.descriptor, staged_snapshot)
        pinned_stage.verify_path()
        if _entry_mode(root_descriptor, final_name) is not None:
            raise ImmutableArtifactError(
                "concurrent final appeared immediately before no-replace commit"
            )
        _verify_named_directory(
            root_descriptor, stage_name, pinned_stage.descriptor
        )
        pinned_root.verify_path()
        try:
            rollback_root = _open_independent_parent_guard(root_descriptor)
            rollback_stage = _open_directory_rollback_guard(
                pinned_stage.descriptor
            )
        except BaseException as guard_error:
            _close_descriptors(
                tuple(
                    descriptor_value
                    for descriptor_value in (rollback_stage, rollback_root)
                    if descriptor_value >= 0
                ),
                active_error=guard_error,
                label=f"directory publication {final_name} rollback guard cleanup",
            )
            rollback_root = rollback_stage = -1
            raise
        _fsync_or_fail(root_descriptor, "publication parent before commit")
        pinned_root.verify_path()
        try:
            _rename_noreplace_at(
                root_descriptor, stage_name, root_descriptor, final_name
            )
            committed = True
        except (FileExistsError, ImmutableArtifactError):
            recovered_own_commit = False
            if _entry_mode(root_descriptor, stage_name) is None:
                _verify_named_directory(
                    root_descriptor, final_name, pinned_stage.descriptor
                )
                # The rename committed this exact pinned stage before the
                # helper reported an ambiguous failure. Record ownership before
                # fallback authorization so a later failure triggers retirement.
                committed = True
                recovered_own_commit = accept_winner()
            if not recovered_own_commit:
                raise
        _verify_named_directory(
            root_descriptor, final_name, pinned_stage.descriptor
        )
        pinned_root.verify_path()
        _fsync_or_fail(root_descriptor, "publication parent after commit")
        # Rehash the committed inode tree. A detected race retires this exact
        # generation to a hidden evidence name before the failure escapes.
        committed_snapshot = _verify_published_files_at(
            pinned_stage.descriptor, material
        )
        _verify_tree_rename_continuity(
            staged_snapshot,
            committed_snapshot,
            label=(
                "immutable tree child generation changed across publication commit"
            ),
        )
        authorize()
        if _entry_mode(root_descriptor, stage_name) is not None:
            raise ImmutableArtifactError(
                "deterministic stage remained after immutable publication"
            )
        # The authorization callback is an external observation boundary. It
        # may inspect or mutate same-UID state, so close the committed inode
        # tree against the pre-authorization snapshot without rehashing it.
        # Failure retires the exact committed generation outside the active
        # namespace while retaining its bytes as evidence.
        _verify_snapshot_against_at(
            pinned_stage.descriptor, committed_snapshot
        )
        if _entry_mode(root_descriptor, stage_name) is not None:
            raise ImmutableArtifactError(
                "deterministic stage reappeared after publication closure"
            )
        _verify_named_directory(
            root_descriptor, final_name, pinned_stage.descriptor
        )
        pinned_root.verify_path()
        if after_commit is not None:
            after_commit(root, pinned_root.descriptor)
            _verify_snapshot_against_at(
                pinned_stage.descriptor, committed_snapshot
            )
            if _entry_mode(root_descriptor, stage_name) is not None:
                raise ImmutableArtifactError(
                    "deterministic stage reappeared after post-commit finalization"
                )
            _verify_named_directory(
                root_descriptor, final_name, pinned_stage.descriptor
            )
            pinned_root.verify_path()

        cleanup_error: BaseException | None = None
        try:
            pinned_stage.close()
        except BaseException as exc:
            cleanup_error = exc
        if owns_root:
            try:
                pinned_root.close()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    cleanup_error.add_note(
                        f"{DURABILITY_UNPROVEN}: directory publication parent "
                        f"cleanup: {exc}"
                    )
        if cleanup_error is not None:
            try:
                retired_name = _retire_failed_publication_at(
                    rollback_root,
                    final_name,
                    rollback_stage,
                    directory=True,
                )
            except BaseException as retirement_error:
                cleanup_error.add_note(
                    "failed directory publication could not leave the active "
                    f"namespace after cleanup failure: {retirement_error}"
                )
            else:
                cleanup_error.add_note(
                    "failed directory publication preserved outside the active "
                    f"namespace as {retired_name}"
                )
            guard_diagnostics = _close_rollback_guards(
                (rollback_stage, rollback_root),
                label=f"directory publication {final_name} rollback guards",
            )
            for detail in guard_diagnostics:
                cleanup_error.add_note(detail)
            rollback_root = rollback_stage = -1
            committed = False
            pinned_stage = None
            raise cleanup_error
        _close_rollback_guards(
            (rollback_stage, rollback_root),
            label=f"directory publication {final_name} rollback guards",
        )
        rollback_root = rollback_stage = -1
        pinned_stage = None
        return final
    except BaseException as exc:
        if committed and pinned_stage is not None:
            try:
                retired_name = _retire_failed_publication_at(
                    pinned_root.descriptor,
                    final_name,
                    pinned_stage.descriptor,
                    directory=True,
                )
            except BaseException as retirement_error:
                exc.add_note(
                    "failed directory publication could not leave the active "
                    f"namespace: {retirement_error}"
                )
            else:
                exc.add_note(
                    "failed directory publication preserved outside the active "
                    f"namespace as {retired_name}"
                )
        raise
    finally:
        closers: list[Callable[[], None]] = []
        if pinned_stage is not None:
            closers.append(pinned_stage.close)
        rollback_guards = tuple(
            descriptor_value
            for descriptor_value in (rollback_stage, rollback_root)
            if descriptor_value >= 0
        )
        if rollback_guards:
            closers.append(
                lambda: _close_rollback_guards(
                    rollback_guards,
                    label=f"directory publication {final_name} rollback guards",
                )
            )
        if owns_root:
            assert pinned_root is not None
            closers.append(pinned_root.close)
        _close_callables(closers, label=f"directory publication {final_name} cleanup")


@dataclasses.dataclass(frozen=True)
class DirectoryGenerationFile:
    """One file identity in a sealed immutable directory generation."""

    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)
        if not _is_sha256(self.sha256):
            raise ImmutableArtifactError(
                "directory generation file sha256 must be lowercase 64-hex"
            )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ImmutableArtifactError(
                "directory generation file size_bytes must be a nonnegative integer"
            )


@dataclasses.dataclass(frozen=True)
class ImmutableDirectoryGeneration:
    """Portable content and host-local identity for one directory generation."""

    path: Path
    content_manifest_sha256: str
    filesystem_generation_sha256: str
    file_count: int
    directory_count: int
    total_file_bytes: int
    files: tuple[DirectoryGenerationFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ImmutableArtifactError(
                "directory generation path must be a Path"
            )
        for label, digest in (
            ("content_manifest_sha256", self.content_manifest_sha256),
            ("filesystem_generation_sha256", self.filesystem_generation_sha256),
        ):
            if not _is_sha256(digest):
                raise ImmutableArtifactError(
                    f"directory generation {label} must be lowercase 64-hex"
                )
        for label, value in (
            ("file_count", self.file_count),
            ("directory_count", self.directory_count),
            ("total_file_bytes", self.total_file_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ImmutableArtifactError(
                    f"directory generation {label} must be a nonnegative integer"
                )
        if self.directory_count < 1:
            raise ImmutableArtifactError(
                "directory generation directory_count must include the root"
            )
        if type(self.files) is not tuple or any(
            type(file) is not DirectoryGenerationFile for file in self.files
        ):
            raise ImmutableArtifactError(
                "directory generation files must be an exact identity tuple"
            )
        if self.file_count != len(self.files):
            raise ImmutableArtifactError(
                "directory generation file_count disagrees with files"
            )
        if self.total_file_bytes != sum(
            file.size_bytes for file in self.files
        ):
            raise ImmutableArtifactError(
                "directory generation total_file_bytes disagrees with files"
            )


_GENERATION_LIMIT_KEYS = (
    "max_files",
    "max_directories",
    "max_total_file_bytes",
    "max_total_relative_path_bytes",
)


def _generation_limits(
    *,
    max_files: int | None,
    max_directories: int | None,
    max_total_file_bytes: int | None,
    max_total_relative_path_bytes: int | None,
) -> dict[str, int | None]:
    return {
        "max_files": max_files,
        "max_directories": max_directories,
        "max_total_file_bytes": max_total_file_bytes,
        "max_total_relative_path_bytes": max_total_relative_path_bytes,
    }


def _generation_file_byte_limits(
    *,
    max_file_bytes: int | None,
    max_file_bytes_by_path: Mapping[str, int] | None,
) -> tuple[int | None, dict[str, int]]:
    if max_file_bytes is not None and (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes < 0
    ):
        raise ImmutableArtifactError(
            "immutable generation per-file byte limit is invalid"
        )
    if max_file_bytes_by_path is None:
        return max_file_bytes, {}
    if not isinstance(max_file_bytes_by_path, Mapping):
        raise ImmutableArtifactError(
            "immutable generation path-specific byte limits are invalid"
        )
    normalized: dict[str, int] = {}
    for relative_path, limit in max_file_bytes_by_path.items():
        safe_relative_path(relative_path)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ImmutableArtifactError(
                "immutable generation path-specific byte limit is invalid: "
                f"{relative_path!r}"
            )
        normalized[relative_path] = limit
    return max_file_bytes, normalized


def _generation_file_byte_limit(
    relative_path: str,
    *,
    max_file_bytes: int | None,
    max_file_bytes_by_path: Mapping[str, int],
) -> int | None:
    limit = max_file_bytes_by_path.get(relative_path, max_file_bytes)
    if max_file_bytes is not None and limit is not None:
        return min(max_file_bytes, limit)
    return limit


def _enforce_generation_file_byte_limits(
    files: Mapping[str, _Snapshot],
    *,
    label: str,
    max_file_bytes: int | None,
    max_file_bytes_by_path: Mapping[str, int] | None,
) -> None:
    global_limit, path_limits = _generation_file_byte_limits(
        max_file_bytes=max_file_bytes,
        max_file_bytes_by_path=max_file_bytes_by_path,
    )
    for relative_path, metadata in files.items():
        limit = _generation_file_byte_limit(
            relative_path,
            max_file_bytes=global_limit,
            max_file_bytes_by_path=path_limits,
        )
        if limit is not None and metadata[4] > limit:
            raise ImmutableArtifactLimitError(
                f"{label} file exceeds bounded per-file byte limit: "
                f"{relative_path}: {metadata[4]} > {limit}"
            )


def _filesystem_generation_sha256(
    files: Mapping[str, _Snapshot],
    directories: Mapping[str, _Snapshot],
) -> str:
    filesystem_model = {
        "schema_version": 1,
        "scope": "immutable_directory_filesystem_generation",
        "directories": [
            {
                "path": relative_path,
                "metadata": [str(value) for value in metadata],
            }
            for relative_path, metadata in sorted(directories.items())
        ],
        "files": [
            {
                "path": relative_path,
                "metadata": [str(value) for value in metadata],
            }
            for relative_path, metadata in sorted(files.items())
        ],
    }
    return hashlib.sha256(
        b"immutable-directory-filesystem-generation-v1\n"
        + canonical_json_bytes(filesystem_model)
    ).hexdigest()


def _verify_tree_rename_continuity(
    before: _TreeSnapshot,
    after: _TreeSnapshot,
    *,
    label: str,
) -> None:
    before_files, before_directories = before
    after_files, after_directories = after
    if before_files != after_files or {
        path: metadata
        for path, metadata in before_directories.items()
        if path != "."
    } != {
        path: metadata
        for path, metadata in after_directories.items()
        if path != "."
    }:
        raise ImmutableArtifactError(label)


@_translate_descriptor_resource_errors(
    "immutable directory generation inspection"
)
def _inspect_immutable_directory_generation_at(
    root_descriptor: int,
    *,
    path: Path,
    require_nonempty: bool,
    max_files: int | None,
    max_directories: int | None,
    max_total_file_bytes: int | None,
    max_total_relative_path_bytes: int | None,
    max_file_bytes: int | None = None,
    max_file_bytes_by_path: Mapping[str, int] | None = None,
) -> ImmutableDirectoryGeneration:
    """Hash one finalized tree through a pinned root and close every race."""
    if type(require_nonempty) is not bool:
        raise ImmutableArtifactError("require_nonempty must be boolean")
    max_file_bytes, max_file_bytes_by_path = _generation_file_byte_limits(
        max_file_bytes=max_file_bytes,
        max_file_bytes_by_path=max_file_bytes_by_path,
    )
    limits = _generation_limits(
        max_files=max_files,
        max_directories=max_directories,
        max_total_file_bytes=max_total_file_bytes,
        max_total_relative_path_bytes=max_total_relative_path_bytes,
    )
    monitor = _arm_directory_mutation_monitor(root_descriptor, **limits)
    try:
        tree = _snapshot_pinned_tree_at(
            root_descriptor,
            require_readonly_files=True,
            require_readonly_directories=True,
            **limits,
        )
        files = dict(tree.files)
        directories = dict(tree.directories)
        if require_nonempty and not files:
            raise ImmutableArtifactError(
                "immutable directory generation contains no regular files"
            )
        _enforce_generation_file_byte_limits(
            files,
            label="immutable generation",
            max_file_bytes=max_file_bytes,
            max_file_bytes_by_path=max_file_bytes_by_path,
        )
        for relative_path, metadata in files.items():
            if stat.S_IMODE(metadata[2]) != 0o444 or metadata[3] != 1:
                raise ImmutableArtifactError(
                    "immutable generation file is not single-link 0444: "
                    f"{relative_path}"
                )
        for relative_path, metadata in directories.items():
            if stat.S_IMODE(metadata[2]) != 0o555:
                raise ImmutableArtifactError(
                    "immutable generation directory is not 0555: "
                    f"{relative_path}"
                )

        identities: list[DirectoryGenerationFile] = []
        total_file_bytes = 0
        for relative_path, expected_metadata in sorted(files.items()):
            file_byte_limit = _generation_file_byte_limit(
                relative_path,
                max_file_bytes=max_file_bytes,
                max_file_bytes_by_path=max_file_bytes_by_path,
            )
            if max_total_file_bytes is not None:
                remaining_total_bytes = max_total_file_bytes - total_file_bytes
                file_byte_limit = (
                    remaining_total_bytes
                    if file_byte_limit is None
                    else min(file_byte_limit, remaining_total_bytes)
                )
            pinned = _open_relative_regular_at(root_descriptor, relative_path)
            try:
                if _snapshot_fields(os.fstat(pinned.descriptor)) != expected_metadata:
                    raise ImmutableArtifactError(
                        "immutable generation file changed before hash: "
                        f"{relative_path}"
                    )
                digest, size_bytes, _mode = _hash_fd(
                    pinned.descriptor,
                    max_bytes=file_byte_limit,
                )
                pinned.verify_path()
                if _snapshot_fields(os.fstat(pinned.descriptor)) != expected_metadata:
                    raise ImmutableArtifactError(
                        "immutable generation file changed during hash: "
                        f"{relative_path}"
                    )
            finally:
                pinned.close()
            identities.append(
                DirectoryGenerationFile(relative_path, digest, size_bytes)
            )
            total_file_bytes += size_bytes

        closure = _snapshot_tree_descriptor(root_descriptor, **limits)
        if closure != (files, directories):
            raise ImmutableArtifactError(
                "immutable directory generation changed during content sealing"
            )
        monitor.require_quiet()
        model = {
            "schema_version": 1,
            "scope": "immutable_directory_generation",
            "directories": sorted(directories),
            "files": [
                {
                    "path": identity.relative_path,
                    "sha256": identity.sha256,
                    "size_bytes": identity.size_bytes,
                }
                for identity in identities
            ],
        }
        content_manifest_sha256 = hashlib.sha256(
            b"immutable-directory-generation-v1\n" + canonical_json_bytes(model)
        ).hexdigest()
        filesystem_generation_sha256 = _filesystem_generation_sha256(
            files, directories,
        )
        return ImmutableDirectoryGeneration(
            path=Path(path),
            content_manifest_sha256=content_manifest_sha256,
            filesystem_generation_sha256=filesystem_generation_sha256,
            file_count=len(identities),
            directory_count=len(directories),
            total_file_bytes=total_file_bytes,
            files=tuple(identities),
        )
    finally:
        monitor.close()


@_translate_descriptor_resource_errors(
    "immutable directory generation path inspection"
)
def inspect_immutable_directory_generation(
    path: Path,
    *,
    require_nonempty: bool = True,
    max_files: int | None = None,
    max_directories: int | None = None,
    max_total_file_bytes: int | None = None,
    max_total_relative_path_bytes: int | None = None,
    max_file_bytes: int | None = None,
    max_file_bytes_by_path: Mapping[str, int] | None = None,
) -> ImmutableDirectoryGeneration:
    """Recompute one committed generation's descriptor-anchored content seal."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        root = _open_pinned_directory(absolute)
    except (ImmutableArtifactError, OSError) as exc:
        raise ImmutableArtifactError(
            f"immutable directory generation is unavailable: {absolute}"
        ) from exc
    try:
        generation = _inspect_immutable_directory_generation_at(
            root.descriptor,
            path=absolute,
            require_nonempty=require_nonempty,
            max_files=max_files,
            max_directories=max_directories,
            max_total_file_bytes=max_total_file_bytes,
            max_total_relative_path_bytes=max_total_relative_path_bytes,
            max_file_bytes=max_file_bytes,
            max_file_bytes_by_path=max_file_bytes_by_path,
        )
        root.verify_path()
        return generation
    finally:
        root.close()


@_translate_descriptor_resource_errors(
    "external directory generation freeze"
)
def _freeze_external_directory_generation_at(
    root_descriptor: int,
    *,
    path: Path,
    require_nonempty: bool,
    max_files: int | None,
    max_directories: int | None,
    max_total_file_bytes: int | None,
    max_total_relative_path_bytes: int | None,
    max_file_bytes: int | None = None,
    max_file_bytes_by_path: Mapping[str, int] | None = None,
) -> ImmutableDirectoryGeneration:
    """Validate an externally written stage, make it read-only, and seal it."""
    max_file_bytes, max_file_bytes_by_path = _generation_file_byte_limits(
        max_file_bytes=max_file_bytes,
        max_file_bytes_by_path=max_file_bytes_by_path,
    )
    limits = _generation_limits(
        max_files=max_files,
        max_directories=max_directories,
        max_total_file_bytes=max_total_file_bytes,
        max_total_relative_path_bytes=max_total_relative_path_bytes,
    )
    tree = _snapshot_pinned_tree_at(
        root_descriptor,
        require_readonly_files=False,
        require_readonly_directories=False,
        **limits,
    )
    files = dict(tree.files)
    directories = dict(tree.directories)
    if require_nonempty and not files:
        raise ImmutableArtifactError(
            "external directory generation contains no regular files"
        )
    _enforce_generation_file_byte_limits(
        files,
        label="external generation",
        max_file_bytes=max_file_bytes,
        max_file_bytes_by_path=max_file_bytes_by_path,
    )

    for relative_path, expected_metadata in sorted(files.items()):
        pinned = _open_relative_regular_at(root_descriptor, relative_path)
        try:
            if _snapshot_fields(os.fstat(pinned.descriptor)) != expected_metadata:
                raise ImmutableArtifactError(
                    "external generation file changed before finalization: "
                    f"{relative_path}"
                )
            os.fchmod(pinned.descriptor, 0o444)
            _fsync_or_fail(
                pinned.descriptor, f"external generation file {relative_path}"
            )
            pinned.verify_path()
        finally:
            pinned.close()

    for relative_path in sorted(
        (name for name in directories if name != "."),
        key=lambda value: len(safe_relative_path(value).parts),
        reverse=True,
    ):
        directory = _open_pinned_relative_directory(
            root_descriptor, safe_relative_path(relative_path).parts
        )
        try:
            os.fchmod(directory.descriptor, 0o555)
            _fsync_or_fail(
                directory.descriptor,
                f"external generation directory {relative_path}",
            )
            directory.verify_path()
        finally:
            directory.close()
    os.fchmod(root_descriptor, 0o555)
    _fsync_or_fail(root_descriptor, "external generation root")
    return _inspect_immutable_directory_generation_at(
        root_descriptor,
        path=path,
        require_nonempty=require_nonempty,
        max_file_bytes=max_file_bytes,
        max_file_bytes_by_path=max_file_bytes_by_path,
        **limits,
    )


@dataclasses.dataclass
class ExclusiveDirectoryStage:
    """A no-follow, no-reuse directory stage for an external producer.

    The stage directory stays pinned while a subprocess writes through
    ``proc_path``.  ``commit`` freezes and seals the complete tree, then performs
    one ``RENAME_NOREPLACE`` into its unique final generation.  Every failure
    preserves either the hidden stage or the already-renamed final inode tree.
    """

    output_root: Path
    final_name: str
    staging_name: str
    _root: _PinnedDirectory
    _stage: _PinnedDirectory
    _committed: bool = False
    _closed: bool = False
    _committed_generation: ImmutableDirectoryGeneration | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def descriptor(self) -> int:
        if self._closed:
            raise ImmutableArtifactError("exclusive directory stage is closed")
        return self._stage.descriptor

    @property
    def staging_path(self) -> Path:
        return self.output_root / self.staging_name

    @property
    def final_path(self) -> Path:
        return self.output_root / self.final_name

    @property
    def path(self) -> Path:
        return self.final_path if self._committed else self.staging_path

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def committed_generation(self) -> ImmutableDirectoryGeneration | None:
        """Return the exact identity already sealed by a successful commit.

        The identity remains available after descriptor cleanup so callers can
        report a later authentication or context-unwind failure without losing
        which immutable generation was renamed.
        """
        return self._committed_generation

    def verify_path(self) -> None:
        if self._closed:
            raise ImmutableArtifactError("exclusive directory stage is closed")
        self._root.verify_path()
        if self._committed:
            _verify_named_directory(
                self._root.descriptor, self.final_name, self._stage.descriptor
            )
        else:
            self._stage.verify_path()
            if _entry_mode(self._root.descriptor, self.final_name) is not None:
                raise ImmutableArtifactError(
                    "directory generation destination appeared before commit"
                )
        self._root.verify_path()

    @_translate_descriptor_resource_errors(
        "exclusive directory generation commit"
    )
    def commit(
        self,
        *,
        require_nonempty: bool = True,
        max_files: int | None = None,
        max_directories: int | None = None,
        max_total_file_bytes: int | None = None,
        max_total_relative_path_bytes: int | None = None,
        max_file_bytes: int | None = None,
        max_file_bytes_by_path: Mapping[str, int] | None = None,
        expected_files: tuple[DirectoryGenerationFile, ...] | None = None,
    ) -> ImmutableDirectoryGeneration:
        """Freeze and atomically commit this exact externally written tree.

        ``expected_files`` is an optional producer-side admission seal.  It is
        compared after the stage has been frozen and hashed but before the
        no-replace rename, so bytes that differ from the already-admitted
        identities remain hidden and cannot become the committed generation.
        """
        if self._committed:
            raise ImmutableArtifactError(
                "exclusive directory stage has already been committed"
            )
        if expected_files is not None and (
            type(expected_files) is not tuple
            or any(
                type(identity) is not DirectoryGenerationFile
                for identity in expected_files
            )
            or tuple(sorted(expected_files, key=lambda item: item.relative_path))
            != expected_files
            or len({item.relative_path for item in expected_files})
            != len(expected_files)
        ):
            raise ImmutableArtifactError(
                "expected external generation files must be a sorted exact "
                "identity tuple"
            )
        self.verify_path()
        limits = _generation_limits(
            max_files=max_files,
            max_directories=max_directories,
            max_total_file_bytes=max_total_file_bytes,
            max_total_relative_path_bytes=max_total_relative_path_bytes,
        )
        staged = _freeze_external_directory_generation_at(
            self._stage.descriptor,
            path=self.staging_path,
            require_nonempty=require_nonempty,
            max_file_bytes=max_file_bytes,
            max_file_bytes_by_path=max_file_bytes_by_path,
            **limits,
        )
        if expected_files is not None and staged.files != expected_files:
            raise ImmutableArtifactError(
                "external directory generation disagrees with the admitted "
                "file seal"
            )
        staged_tree = _snapshot_pinned_tree_at(
            self._stage.descriptor,
            require_readonly_files=True,
            require_readonly_directories=True,
            **limits,
        )
        staged_snapshot: _TreeSnapshot = (
            dict(staged_tree.files), dict(staged_tree.directories),
        )
        if _filesystem_generation_sha256(*staged_snapshot) != (
            staged.filesystem_generation_sha256
        ):
            raise ImmutableArtifactError(
                "external directory generation changed after staging seal"
            )
        self._stage.verify_path()
        if _entry_mode(self._root.descriptor, self.final_name) is not None:
            raise ImmutableArtifactError(
                "directory generation destination appeared before commit"
            )
        _fsync_or_fail(
            self._root.descriptor,
            "external generation parent before commit",
        )
        self._stage.verify_path()
        try:
            _rename_noreplace_at(
                self._root.descriptor,
                self.staging_name,
                self._root.descriptor,
                self.final_name,
            )
        except FileExistsError as exc:
            # Never accept an exact winner for an attempt-scoped generation.  If
            # the rename took effect before an injected/ambiguous error, retain
            # that final inode as non-active evidence and report the race.
            if _entry_mode(self._root.descriptor, self.staging_name) is None:
                try:
                    _verify_named_directory(
                        self._root.descriptor,
                        self.final_name,
                        self._stage.descriptor,
                    )
                except BaseException:
                    pass
                else:
                    self._committed = True
            raise ImmutableArtifactError(
                "directory generation destination race rejected"
            ) from exc
        self._committed = True
        _verify_named_directory(
            self._root.descriptor, self.final_name, self._stage.descriptor
        )
        _fsync_or_fail(
            self._root.descriptor,
            "external generation parent after commit",
        )
        renamed_tree = _snapshot_pinned_tree_at(
            self._stage.descriptor,
            require_readonly_files=True,
            require_readonly_directories=True,
            **limits,
        )
        renamed_snapshot: _TreeSnapshot = (
            dict(renamed_tree.files), dict(renamed_tree.directories),
        )
        _verify_tree_rename_continuity(
            staged_snapshot,
            renamed_snapshot,
            label="external directory child generation changed across commit",
        )
        committed = _inspect_immutable_directory_generation_at(
            self._stage.descriptor,
            path=self.final_path,
            require_nonempty=require_nonempty,
            max_file_bytes=max_file_bytes,
            max_file_bytes_by_path=max_file_bytes_by_path,
            **limits,
        )
        # The root directory's host-local ctime may advance during rename, so
        # portable content must match across the commit while the authoritative
        # filesystem-generation seal is the one recomputed after rename.
        if (
            committed.content_manifest_sha256
            != staged.content_manifest_sha256
            or committed.file_count != staged.file_count
            or committed.directory_count != staged.directory_count
            or committed.total_file_bytes != staged.total_file_bytes
            or committed.files != staged.files
        ):
            raise ImmutableArtifactError(
                "external directory generation changed across commit"
            )
        closure_tree = _snapshot_pinned_tree_at(
            self._stage.descriptor,
            require_readonly_files=True,
            require_readonly_directories=True,
            **limits,
        )
        closure_snapshot: _TreeSnapshot = (
            dict(closure_tree.files), dict(closure_tree.directories),
        )
        _verify_tree_rename_continuity(
            staged_snapshot,
            closure_snapshot,
            label=(
                "external directory child generation changed during commit closure"
            ),
        )
        self._committed_generation = committed
        self.verify_path()
        return committed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_callables(
            (self._stage.close, self._root.close),
            label=f"exclusive directory stage {self.final_name} close",
        )

    def __enter__(self) -> "ExclusiveDirectoryStage":
        try:
            self.verify_path()
        except BaseException:
            _close_callables(
                (self.close,), label="exclusive directory stage entry close"
            )
            raise
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


@_translate_descriptor_resource_errors("exclusive directory stage creation")
def create_exclusive_directory_stage(
    output_root: Path,
    final_name: str,
) -> ExclusiveDirectoryStage:
    """Create one exclusive hidden stage without reusing any prior pathname."""
    _validate_publication_name(final_name)
    staging_name = f".{final_name}.staging"
    if len(os.fsencode(staging_name)) > 255:
        raise ImmutableArtifactError(
            f"{NAMESPACE_NAME_OVERFLOW}: external generation staging name"
        )
    root_path = Path(os.path.abspath(os.fspath(output_root)))
    ensure_directory(root_path)
    root = _open_pinned_directory(root_path)
    stage: _PinnedDirectory | None = None
    try:
        root.verify_path()
        if _entry_mode(root.descriptor, final_name) is not None:
            raise ImmutableArtifactError(
                f"directory generation destination already exists: {final_name}"
            )
        if _entry_mode(root.descriptor, staging_name) is not None:
            raise ImmutableArtifactError(
                f"exclusive directory stage already exists: {staging_name}"
            )
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=root.descriptor)
        except FileExistsError as exc:
            raise ImmutableArtifactError(
                f"exclusive directory stage creation raced: {staging_name}"
            ) from exc
        stage = _open_pinned_relative_directory(root.descriptor, (staging_name,))
        _lock_directory(stage.descriptor, staging_name, blocking=False)
        os.fchmod(stage.descriptor, 0o700)
        _fsync_or_fail(stage.descriptor, "new exclusive directory stage")
        _fsync_or_fail(root.descriptor, "exclusive directory stage parent")
        stage.verify_path()
        if _entry_mode(root.descriptor, final_name) is not None:
            raise ImmutableArtifactError(
                "directory generation destination appeared during stage creation"
            )
        root.verify_path()
        result = ExclusiveDirectoryStage(
            output_root=root_path,
            final_name=final_name,
            staging_name=staging_name,
            _root=root,
            _stage=stage,
        )
        stage = None
        root = None  # type: ignore[assignment]
        return result
    finally:
        closers: list[Callable[[], None]] = []
        if stage is not None:
            closers.append(stage.close)
        if root is not None:
            closers.append(root.close)
        if closers:
            _close_callables(
                closers, label="exclusive directory stage creation cleanup"
            )
