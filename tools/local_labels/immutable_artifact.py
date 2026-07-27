#!/usr/bin/env python3
"""Small fail-closed primitives for immutable external artifact publication."""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import resource
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


_CHUNK = 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_MAX_COMPONENTS = 64
_MAX_RELATIVE_BYTES = 4096
_MAX_FINAL_NAME_BYTES = 246
_MAX_REGULAR_LEAF_BYTES = 243
_SNAPSHOT_CLOSURE_THRESHOLD = 512
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
    | 0x00004000  # IN_Q_OVERFLOW
    | 0x00008000  # IN_IGNORED
)
_INOTIFY_DIRECTORY_MASK = _INOTIFY_MUTATION_EVENTS | 0x01000000  # IN_ONLYDIR
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
    descriptor: int, *, allow_hardlinks: bool = False,
) -> tuple[bytes, MetadataSnapshot]:
    blocks: list[bytes] = []
    _size, _mode, snapshot = _consume_stable_regular(
        descriptor, blocks.append, allow_hardlinks=allow_hardlinks
    )
    return b"".join(blocks), snapshot


def _read_fd(descriptor: int, *, allow_hardlinks: bool = False) -> bytes:
    payload, _snapshot = _read_fd_snapshot(
        descriptor, allow_hardlinks=allow_hardlinks,
    )
    return payload


def _hash_fd(
    descriptor: int, *, allow_hardlinks: bool = False
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size, mode, _snapshot = _consume_stable_regular(
        descriptor, digest.update, allow_hardlinks=allow_hardlinks
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

    @property
    def descriptor(self) -> int:
        if self._pinned.descriptor < 0:
            raise ImmutableArtifactError("retained executable is closed")
        return self._pinned.descriptor

    def verify_unchanged(self) -> None:
        """Rebind path, inode, metadata, and bytes to the initial observation."""
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

    def close(self) -> None:
        self._pinned.close()

    def __enter__(self) -> RetainedExecutable:
        self.verify_unchanged()
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


def open_retained_executable(path: Path) -> RetainedExecutable:
    """Open, hash, and retain one executable without following any symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    pinned = _open_regular(absolute)
    try:
        descriptor = pinned.descriptor
        before = _snapshot_fields(os.fstat(descriptor))
        if before[2] & 0o111 == 0:
            raise ImmutableArtifactError(f"regular file is not executable: {absolute}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, _size, _mode = _hash_fd(descriptor)
        after = _snapshot_fields(os.fstat(descriptor))
        if after != before:
            raise ImmutableArtifactError("executable metadata changed during identity read")
        pinned.verify_path()
        os.lseek(descriptor, 0, os.SEEK_SET)
        return RetainedExecutable(absolute, digest, before, pinned)
    except BaseException:
        pinned.close()
        raise


@dataclasses.dataclass
class RetainedRegular:
    """One no-follow regular file retained as a caller-owned capability."""

    path: Path
    sha256: str
    size_bytes: int
    snapshot: _Snapshot
    _pinned: _PinnedRegular

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
        return payload

    def verify_unchanged(self) -> None:
        """Rebind path, inode, metadata, size, and bytes to the initial read."""
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

    def close(self) -> None:
        self._pinned.close()

    def __enter__(self) -> RetainedRegular:
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


def open_retained_regular(path: Path) -> RetainedRegular:
    """Open, hash, and retain one regular file without following any symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    pinned = _open_regular(absolute)
    try:
        descriptor = pinned.descriptor
        before = _snapshot_fields(os.fstat(descriptor))
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest, size, _mode = _hash_fd(descriptor)
        after = _snapshot_fields(os.fstat(descriptor))
        if after != before:
            raise ImmutableArtifactError(
                "regular-file metadata changed during identity read"
            )
        pinned.verify_path()
        os.lseek(descriptor, 0, os.SEEK_SET)
        return RetainedRegular(absolute, digest, size, before, pinned)
    except BaseException:
        pinned.close()
        raise


def read_regular_snapshot(
    path: Path, *, allow_hardlinks: bool = False,
) -> PinnedRegularSnapshot:
    pinned = _open_regular(Path(path), allow_hardlinks=allow_hardlinks)
    try:
        payload, metadata = _read_fd_snapshot(
            pinned.descriptor, allow_hardlinks=allow_hardlinks,
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


class PinnedDirectoryReader:
    """Read one tree through a single pinned, continuously revalidated root."""

    def __init__(self, root: Path) -> None:
        self._root = _open_pinned_directory(Path(root))

    @property
    def descriptor(self) -> int:
        return self._root.descriptor

    def verify_path(self) -> None:
        self._root.verify_path()

    def read_snapshot(self, relative_path: str) -> PinnedRegularSnapshot:
        self._root.verify_path()
        pinned = _open_relative_regular_at(self._root.descriptor, relative_path)
        try:
            payload, metadata = _read_fd_snapshot(pinned.descriptor)
            pinned.verify_path()
            if _snapshot_fields(os.fstat(pinned.descriptor)) != metadata:
                raise ImmutableArtifactError(
                    "regular file changed after snapshot read"
                )
            self._root.verify_path()
            return PinnedRegularSnapshot(payload, metadata)
        finally:
            pinned.close()

    def read(self, relative_path: str) -> bytes:
        return self.read_snapshot(relative_path).payload

    def verify_regular(
        self, relative_path: str, expected: MetadataSnapshot,
    ) -> None:
        self._root.verify_path()
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
            self._root.verify_path()
        finally:
            pinned.close()

    def snapshot_tree(
        self,
        *,
        require_readonly_files: bool = False,
        require_readonly_directories: bool = False,
    ) -> PinnedTreeSnapshot:
        self._root.verify_path()
        snapshot = _snapshot_pinned_tree_at(
            self._root.descriptor,
            require_readonly_files=require_readonly_files,
            require_readonly_directories=require_readonly_directories,
        )
        self._root.verify_path()
        return snapshot

    def close(self) -> None:
        self._root.close()

    def __enter__(self) -> "PinnedDirectoryReader":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


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
    raise ImmutableArtifactError(
        f"atomic no-replace rename failed: [{error}] {os.strerror(error)}"
    )


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


def publish_regular_file(
    parent: Path,
    name: str,
    payload_or_source: bytes | Path | ImmutableFile,
    *,
    partial_suffix: str = _PARTIAL_SUFFIX,
    pinned_parent: PinnedDirectoryReader | None = None,
    executable: bool = False,
) -> Path:
    """Publish one regular file without destructive failure cleanup.

    The transaction owns only its descriptor and deterministic partial name.
    Incomplete, replaced, or otherwise uncertain partials are retained; the
    sole producer-side name removal is a verified ``RENAME_NOREPLACE`` commit.
    """
    _validate_publication_name(name)
    if partial_suffix not in _PARTIAL_SUFFIXES:
        raise ImmutableArtifactError(
            f"{NAMESPACE_RESERVED_COMPONENT}: unsupported partial suffix"
        )
    if type(executable) is not bool:
        raise ImmutableArtifactError("executable must be boolean")
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

    owns_parent = pinned_parent is None
    if pinned_parent is None:
        try:
            pinned_parent = PinnedDirectoryReader(Path(parent))
        except (OSError, ImmutableArtifactError) as exc:
            raise ImmutableArtifactError(f"publication parent is unavailable: {parent}") from exc
    else:
        try:
            pinned_parent.verify_path()
        except (OSError, ImmutableArtifactError) as exc:
            raise ImmutableArtifactError(f"publication parent is unavailable: {parent}") from exc
    descriptor: int | None = None
    lock_descriptor: int | None = None
    partial_name = f".{name}{partial_suffix}"
    try:
        pinned_parent.verify_path()
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
        pinned_parent.verify_path()
        _fsync_or_fail(parent_descriptor, f"parent before {name}")
        pinned_parent.verify_path()
        _rename_noreplace_at(parent_descriptor, partial_name, parent_descriptor, name)
        pinned_parent.verify_path()
        _fsync_or_fail(parent_descriptor, f"parent after {name}")
        linked = _entry_metadata(parent_descriptor, name)
        opened = os.fstat(lock_descriptor)
        if (
            linked is None
            or _snapshot_fields(linked) != _snapshot_fields(opened)
            or linked.st_nlink != 1
            or opened.st_nlink != 1
            or (executable and not (linked.st_mode & 0o111))
        ):
            raise ImmutableArtifactError(f"named regular-file path changed: {name}")
        return Path(parent) / name
    except OSError as exc:
        raise ImmutableArtifactError(f"regular-file publication I/O failed: {exc}") from exc
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
        if owns_parent:
            assert pinned_parent is not None
            closers.append(pinned_parent.close)
        _close_callables(closers, label=f"regular publication {name} cleanup")


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


def _tree_inventory_descriptor(
    root_descriptor: int,
) -> tuple[set[str], set[str]]:
    """Inventory one pinned root with bounded descriptor-relative traversal."""
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    pending: list[tuple[str, ...]] = [()]
    while pending:
        parts = pending.pop()
        pinned = _open_pinned_relative_directory(root_descriptor, parts)
        try:
            with os.scandir(pinned.descriptor) as entries:
                snapshot = [
                    (entry.name, entry.stat(follow_symlinks=False).st_mode)
                    for entry in entries
                ]
            pinned.verify_path()
        finally:
            pinned.close()
        for name, mode in snapshot:
            child_parts = (*parts, name)
            relative = PurePosixPath(*child_parts).as_posix()
            safe_relative_path(relative)
            if stat.S_ISLNK(mode):
                raise ImmutableArtifactError(f"symlink in immutable tree: {relative}")
            if stat.S_ISDIR(mode):
                observed_directories.add(relative)
                pending.append(child_parts)
            elif stat.S_ISREG(mode):
                observed_files.add(relative)
            else:
                raise ImmutableArtifactError(f"non-regular entry in immutable tree: {relative}")
    return observed_files, observed_directories


def _tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    pinned = _open_pinned_directory(root)
    try:
        inventory = _tree_inventory_descriptor(pinned.descriptor)
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


def _tree_files(root: Path) -> set[str]:
    return _tree_inventory(root)[0]


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


class _DirectoryMutationMonitor:
    """Fail closed on local syscall mutations during bounded verification."""

    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        initialize = getattr(libc, "inotify_init1", None)
        add_watch = getattr(libc, "inotify_add_watch", None)
        close_on_exec = getattr(os, "O_CLOEXEC", None)
        if initialize is None or add_watch is None or not close_on_exec:
            raise ImmutableArtifactError(
                f"{VERIFY_RESOURCE_BOUND}: inotify mutation monitor unavailable"
            )
        initialize.argtypes = (ctypes.c_int,)
        initialize.restype = ctypes.c_int
        add_watch.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
        add_watch.restype = ctypes.c_int
        descriptor = initialize(os.O_NONBLOCK | close_on_exec)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise ImmutableArtifactError(
                f"{VERIFY_RESOURCE_BOUND}: inotify initialization failed: "
                f"[{error}] {os.strerror(error)}"
            )
        self.descriptor = descriptor
        self._add_watch = add_watch

    def _watch_inode(
        self, descriptor: int, *, directory: bool,
    ) -> None:
        suffix = "/." if directory else ""
        path = os.fsencode(f"/proc/self/fd/{descriptor}{suffix}")
        mask = (
            _INOTIFY_DIRECTORY_MASK
            if directory
            else _INOTIFY_MUTATION_EVENTS
        )
        result = self._add_watch(self.descriptor, path, mask)
        if result >= 0:
            return
        error = ctypes.get_errno()
        raise ImmutableArtifactError(
            f"{VERIFY_RESOURCE_BOUND}: inotify inode watch failed: "
            f"[{error}] {os.strerror(error)}"
        )

    def watch_directory(self, descriptor: int) -> None:
        self._watch_inode(descriptor, directory=True)

    def watch_regular(self, descriptor: int) -> None:
        self._watch_inode(descriptor, directory=False)

    def require_quiet(self) -> None:
        while True:
            try:
                events = os.read(self.descriptor, 64 * 1024)
            except BlockingIOError:
                return
            except InterruptedError:
                continue
            except OSError as exc:
                raise ImmutableArtifactError(
                    f"{VERIFY_RESOURCE_BOUND}: inotify event read failed"
                ) from exc
            if events:
                raise ImmutableArtifactError(
                    "immutable tree changed during bounded verification"
                )
            raise ImmutableArtifactError(
                f"{VERIFY_RESOURCE_BOUND}: inotify monitor closed unexpectedly"
            )

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            _close_descriptors(
                (descriptor,), label="inotify mutation monitor close",
            )


def _walk_tree_descriptor(
    root_descriptor: int,
    *,
    observe_directory: Callable[[int], None] | None = None,
) -> _TreeSnapshot:
    """Inventory one tree while retaining only an O(depth) descriptor stack."""
    files: dict[str, _Snapshot] = {}
    directories: dict[str, _Snapshot] = {}
    pending: list[tuple[str, ...]] = [()]
    while pending:
        parts = pending.pop()
        directory = _open_pinned_relative_directory(root_descriptor, parts)
        try:
            if observe_directory is not None:
                observe_directory(directory.descriptor)
            directories[PurePosixPath(*parts).as_posix() if parts else "."] = (
                _snapshot_fields(os.fstat(directory.descriptor))
            )
            with os.scandir(directory.descriptor) as entries:
                children = sorted(
                    (entry.name, entry.stat(follow_symlinks=False))
                    for entry in entries
                )
            directory.verify_path()
        finally:
            directory.close()
        for name, metadata in children:
            child = (*parts, name)
            relative = PurePosixPath(*child).as_posix()
            safe_relative_path(relative)
            if stat.S_ISLNK(metadata.st_mode):
                raise ImmutableArtifactError(f"symlink in immutable tree: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                files[relative] = _snapshot_fields(metadata)
            else:
                raise ImmutableArtifactError(
                    f"non-regular entry in immutable tree: {relative}"
                )
    return files, directories


def _snapshot_tree_descriptor(
    root_descriptor: int,
) -> _TreeSnapshot:
    return _walk_tree_descriptor(root_descriptor)


def _arm_directory_mutation_monitor(
    root_descriptor: int,
) -> _DirectoryMutationMonitor:
    monitor = _DirectoryMutationMonitor()
    try:
        files, _directories = _walk_tree_descriptor(
            root_descriptor, observe_directory=monitor.watch_directory,
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
        monitor.close()
        raise


def _snapshot_pinned_tree_at(
    root_descriptor: int,
    *,
    require_readonly_files: bool,
    require_readonly_directories: bool,
) -> PinnedTreeSnapshot:
    monitor = _arm_directory_mutation_monitor(root_descriptor)
    try:
        root_before = _snapshot_fields(os.fstat(root_descriptor))
        if not stat.S_ISDIR(root_before[2]):
            raise ImmutableArtifactError("pinned tree root is not a directory")
        if require_readonly_directories and root_before[2] & 0o222:
            raise ImmutableArtifactError("pinned tree root directory is writable")
        files, directories = _snapshot_tree_descriptor(root_descriptor)
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


def _verify_snapshot_contents_at(
    root_descriptor: int,
    expected: dict[str, ImmutableFile],
    *,
    require_readonly_directories: bool,
) -> _TreeSnapshot:
    """Verify large trees with one hash and bounded simultaneous descriptors."""
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, OSError, ValueError) as exc:
        raise ImmutableArtifactError(
            f"{VERIFY_RESOURCE_BOUND}: unable to inspect RLIMIT_NOFILE"
        ) from exc
    if soft_limit < 96:
        raise ImmutableArtifactError(
            f"{VERIFY_RESOURCE_BOUND}: RLIMIT_NOFILE={soft_limit}"
        )
    root_before = _snapshot_fields(os.fstat(root_descriptor))
    if require_readonly_directories and root_before[2] & 0o222:
        raise ImmutableArtifactError("finalized publication root is writable")
    first_files, first_directories = _snapshot_tree_descriptor(root_descriptor)
    expected_directories = _expected_directories(expected)
    if set(first_files) != set(expected) or set(first_directories) - {"."} != expected_directories:
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
    second_files, second_directories = _snapshot_tree_descriptor(root_descriptor)
    if set(second_files) != set(expected) or set(second_directories) - {"."} != expected_directories:
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
    final_files, final_directories = _snapshot_tree_descriptor(root_descriptor)
    if set(final_files) != set(expected) or set(final_directories) - {"."} != expected_directories:
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


def _verify_snapshot_closure_at(
    root_descriptor: int,
    expected: dict[str, ImmutableFile],
    *,
    require_readonly_directories: bool,
) -> _TreeSnapshot:
    monitor = _arm_directory_mutation_monitor(root_descriptor)
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
    finally:
        monitor.close()


def _verify_published_files_at(
    root_descriptor: int,
    files: Iterable[ImmutableFile],
    *,
    require_readonly_directories: bool = True,
) -> _TreeSnapshot:
    expected = _material_map(files)
    expected_directories = _expected_directories(expected)
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, OSError, ValueError):
        soft_limit = None
    estimated_retained = len(expected) + len(expected_directories) + 1
    use_snapshot_closure = len(expected) > _SNAPSHOT_CLOSURE_THRESHOLD
    if soft_limit is not None and estimated_retained + 32 >= soft_limit:
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
        observed, observed_directories = _tree_inventory_descriptor(root_descriptor)
        if observed != set(expected) or observed_directories != expected_directories:
            missing = sorted(set(expected) - observed)
            extra = sorted(observed - set(expected))
            missing_directories = sorted(expected_directories - observed_directories)
            extra_directories = sorted(observed_directories - expected_directories)
            raise ImmutableArtifactError(
                "immutable tree inventory mismatch: "
                f"missing={missing}, extra={extra}, "
                f"missing_directories={missing_directories}, "
                f"extra_directories={extra_directories}"
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


def _verify_snapshot_against_at(
    root_descriptor: int, baseline: _TreeSnapshot,
) -> None:
    """Close a verified tree after authorization without rehashing it."""
    first = _snapshot_tree_descriptor(root_descriptor)
    if first != baseline:
        raise ImmutableArtifactError(
            "immutable tree changed after authorization"
        )
    second = _snapshot_tree_descriptor(root_descriptor)
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


def publish_directory(
    output_root: Path,
    final_name: str,
    files: Iterable[ImmutableFile],
    *,
    before_commit: Callable[[Path, int], None] | None = None,
    pinned_root: PinnedDirectoryReader | None = None,
) -> Path:
    """Publish one immutable tree through a locked, pinned deterministic stage."""
    _validate_publication_name(final_name)
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

    def authorize() -> None:
        if before_commit is not None:
            before_commit(root, pinned_root.descriptor)
        pinned_root.verify_path()

    def accept_winner() -> bool:
        if _entry_mode(pinned_root.descriptor, final_name) is None:
            return False
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

        observed, _ = _tree_inventory_descriptor(pinned_stage.descriptor)
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
            pinned_stage.descriptor
        )
        expected_directories = _expected_directories(expected)
        if observed != set(expected) or observed_directories != expected_directories:
            missing = sorted(set(expected) - observed)
            extra = sorted(observed - set(expected))
            missing_directories = sorted(
                expected_directories - observed_directories
            )
            extra_directories = sorted(
                observed_directories - expected_directories
            )
            raise ImmutableArtifactError(
                "deterministic stage inventory mismatch: "
                f"missing={missing}, extra={extra}, "
                f"missing_directories={missing_directories}, "
                f"extra_directories={extra_directories}"
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
        _fsync_or_fail(root_descriptor, "publication parent before commit")
        pinned_root.verify_path()
        try:
            _rename_noreplace_at(
                root_descriptor, stage_name, root_descriptor, final_name
            )
        except (FileExistsError, ImmutableArtifactError):
            if _entry_mode(root_descriptor, stage_name) is None:
                _verify_named_directory(
                    root_descriptor, final_name, pinned_stage.descriptor
                )
                if accept_winner():
                    return final
            raise
        _verify_named_directory(
            root_descriptor, final_name, pinned_stage.descriptor
        )
        pinned_root.verify_path()
        _fsync_or_fail(root_descriptor, "publication parent after commit")
        # Rehash the committed inode tree.  A detected race is a typed failure;
        # the committed directory is deliberately preserved as possible unique
        # evidence rather than being rolled back or overwritten.
        committed_snapshot = _verify_published_files_at(
            pinned_stage.descriptor, material
        )
        authorize()
        if _entry_mode(root_descriptor, stage_name) is not None:
            raise ImmutableArtifactError(
                "deterministic stage remained after immutable publication"
            )
        # The authorization callback is an external observation boundary.  It
        # may inspect or mutate same-UID state, so close the committed inode
        # tree against the pre-authorization snapshot without rehashing it.
        # Preserve the committed directory as evidence if this check fails.
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
        return final
    finally:
        closers: list[Callable[[], None]] = []
        if pinned_stage is not None:
            closers.append(pinned_stage.close)
        if owns_root:
            assert pinned_root is not None
            closers.append(pinned_root.close)
        _close_callables(closers, label=f"directory publication {final_name} cleanup")
