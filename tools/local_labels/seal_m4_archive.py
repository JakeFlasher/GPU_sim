#!/usr/bin/env python3
"""Seal a completed sm_86 M4 archive without rewriting its evidence.

The capture producer intentionally leaves chunk trees writable.  This command
is the narrow transition between that producer output and ``bind-m4``: it
verifies every chunk DONE manifest, inventories one pinned no-follow tree, and
changes only filesystem modes (regular files to 0444 and directories to 0555).
It never copies, renames, truncates, or removes archive bytes.  A sealed receipt
is published outside the archive through the shared immutable publication
primitive and is idempotent on exact replay.

This module is an offline/operator-gated boundary.  Importing it performs no
filesystem action; callers must explicitly invoke :func:`seal_m4_archive`.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

try:
    from . import campaign_done_manifest as done
    from . import immutable_artifact as immutable
except ImportError:  # pragma: no cover - direct CLI execution
    import campaign_done_manifest as done  # type: ignore
    import immutable_artifact as immutable  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "m4_archive_seal_receipt.schema.json"
)
TOOL_ID = "seal_m4_archive.v1"
EXPECTED_CHUNK_COUNT = 18
DEFAULT_MATRIX_PATH = (
    REPO_ROOT / "data" / "local_labels" / "campaign"
    / "sm86_reference_capture_matrix.json"
)
DEFAULT_RECEIPT_NAME = "m4_archive_seal.json"


class M4ArchiveSealError(RuntimeError):
    """Fail-closed, machine-readable archive sealing refusal."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# Friendly aliases used by small offline callers and tests.
SealError = M4ArchiveSealError
ArchiveSealError = M4ArchiveSealError


@dataclass(frozen=True)
class _Entry:
    relative_path: str
    kind: str
    payload: bytes | None
    device: int
    inode: int
    mode: int
    nlink: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @property
    def sha256(self) -> str:
        return _sha256(self.payload or b"")

    def model(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "nlink": self.nlink,
        }

    def content_model(self) -> tuple[Any, ...]:
        """Identity used before/after mode sealing (mode is intentionally kept)."""
        return (
            self.relative_path,
            self.kind,
            self.sha256,
            self.size_bytes,
            self.device,
            self.inode,
            self.mode,
            self.nlink,
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = immutable.canonical_json_bytes(value)
    except immutable.ImmutableArtifactError as exc:
        _fail("M4_ARCHIVE_JSON", str(exc))
    return _sha256(payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail(code: str, detail: str) -> None:
    raise M4ArchiveSealError(code, detail)


def _absolute(path: Path | str, *, role: str) -> Path:
    candidate = Path(os.fspath(path))
    if not candidate.is_absolute():
        _fail("M4_ARCHIVE_PATH", f"{role} must be absolute: {candidate}")
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    if absolute != candidate or ".." in candidate.parts:
        _fail("M4_ARCHIVE_PATH", f"{role} is not canonical absolute: {candidate}")
    return absolute


def _is_under(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _real_identity(path: Path, *, role: str) -> tuple[int, int]:
    try:
        metadata = path.stat()
    except OSError as exc:
        _fail("M4_ARCHIVE_PATH", f"{role}: {path}: {exc}")
    return int(metadata.st_dev), int(metadata.st_ino)


def _external_archive_root(path: Path, *, repo_root: Path) -> Path:
    archive = _absolute(path, role="archive root")
    repository = _absolute(repo_root, role="repository root")
    try:
        archive_real = Path(os.path.realpath(os.fspath(archive)))
        repository_real = Path(os.path.realpath(os.fspath(repository)))
    except (OSError, ValueError) as exc:
        _fail("M4_ARCHIVE_PATH", str(exc))
    if _is_under(repository_real, archive_real) or _is_under(archive_real, repository_real):
        _fail(
            "M4_ARCHIVE_NOT_EXTERNAL",
            f"archive={archive_real}; repository={repository_real}",
        )
    try:
        metadata = archive.lstat()
    except OSError as exc:
        _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"{archive}: {exc}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"archive root is not a real directory: {archive}")
    return archive


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _node_equal(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev) == int(right.st_dev)
        and int(left.st_ino) == int(right.st_ino)
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _read_descriptor(descriptor: int) -> bytes:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError:
        # Some regular files may not be seekable on unusual filesystems; the
        # immutable helper's regular-file contract still requires a seekable
        # ordinary archive file, so surface the same typed boundary.
        _fail("M4_ARCHIVE_TREE_READ", "archive file is not seekable")
    chunks: list[bytes] = []
    while True:
        try:
            block = os.read(descriptor, 1 << 20)
        except InterruptedError:
            continue
        except OSError as exc:
            _fail("M4_ARCHIVE_TREE_READ", str(exc))
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _open_child_directory(parent_descriptor: int, name: str) -> int:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        opened = os.fstat(descriptor)
    except OSError as exc:
        _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"directory {name}: {exc}")
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or not _node_equal(linked, opened)
    ):
        os.close(descriptor)
        _fail("M4_ARCHIVE_TREE_CHANGED", f"directory replaced: {name}")
    return descriptor


def _open_child_file(parent_descriptor: int, name: str) -> tuple[int, os.stat_result]:
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"non-regular archive entry: {name}")
        if linked.st_nlink != 1:
            _fail("M4_ARCHIVE_HARDLINK", name)
        descriptor = os.open(name, _file_flags(), dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
    except M4ArchiveSealError:
        raise
    except OSError as exc:
        _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"file {name}: {exc}")
    if (
        stat.S_ISLNK(opened.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not _node_equal(linked, opened)
    ):
        os.close(descriptor)
        _fail("M4_ARCHIVE_TREE_CHANGED", f"file replaced: {name}")
    return descriptor, opened


def _scan_directory(
    descriptor: int,
    parts: tuple[str, ...],
    entries: list[_Entry],
) -> None:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            _fail("M4_ARCHIVE_TREE_SNAPSHOT", "opened archive node is not a directory")
        with os.scandir(descriptor) as iterator:
            children = sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in iterator),
                key=lambda row: os.fsencode(row[0]),
            )
    except M4ArchiveSealError:
        raise
    except OSError as exc:
        _fail("M4_ARCHIVE_TREE_SNAPSHOT", str(exc))

    for name, linked in children:
        child_parts = (*parts, name)
        relative = PurePosixPath(*child_parts).as_posix()
        try:
            immutable.safe_relative_path(relative)
        except immutable.ImmutableArtifactError as exc:
            _fail("M4_ARCHIVE_PATH", f"{relative}: {exc}")
        if stat.S_ISLNK(linked.st_mode):
            _fail("M4_ARCHIVE_SYMLINK", relative)
        if stat.S_ISDIR(linked.st_mode):
            child = _open_child_directory(descriptor, name)
            try:
                _scan_directory(child, child_parts, entries)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not _node_equal(current, os.fstat(child)):
                    _fail("M4_ARCHIVE_TREE_CHANGED", relative)
            except OSError as exc:
                _fail("M4_ARCHIVE_TREE_CHANGED", f"{relative}: {exc}")
            finally:
                os.close(child)
            metadata = linked
            entries.append(_Entry(
                relative_path=relative,
                kind="directory",
                payload=None,
                device=int(metadata.st_dev),
                inode=int(metadata.st_ino),
                mode=stat.S_IMODE(metadata.st_mode),
                nlink=int(metadata.st_nlink),
                size_bytes=int(metadata.st_size),
                mtime_ns=int(metadata.st_mtime_ns),
                ctime_ns=int(metadata.st_ctime_ns),
            ))
            continue
        if stat.S_ISREG(linked.st_mode):
            file_descriptor, opened = _open_child_file(descriptor, name)
            try:
                payload = _read_descriptor(file_descriptor)
                after = os.fstat(file_descriptor)
                named_after = os.stat(
                    name, dir_fd=descriptor, follow_symlinks=False,
                )
                if (
                    not _node_equal(after, opened)
                    or not _node_equal(named_after, after)
                    or after.st_size != len(payload)
                    or _snapshot(named_after) != _snapshot(after)
                ):
                    _fail("M4_ARCHIVE_TREE_CHANGED", relative)
                metadata = after
            except OSError as exc:
                _fail("M4_ARCHIVE_TREE_READ", f"{relative}: {exc}")
            finally:
                os.close(file_descriptor)
            entries.append(_Entry(
                relative_path=relative,
                kind="file",
                payload=payload,
                device=int(metadata.st_dev),
                inode=int(metadata.st_ino),
                mode=stat.S_IMODE(metadata.st_mode),
                nlink=int(metadata.st_nlink),
                size_bytes=int(metadata.st_size),
                mtime_ns=int(metadata.st_mtime_ns),
                ctime_ns=int(metadata.st_ctime_ns),
            ))
            continue
        _fail("M4_ARCHIVE_NONREGULAR", relative)

    after = os.fstat(descriptor)
    if _snapshot(before) != _snapshot(after):
        _fail("M4_ARCHIVE_TREE_CHANGED", PurePosixPath(*parts).as_posix() or ".")


def _scan_tree(
    root: Path,
    *,
    required_root_mode: int | None = None,
) -> list[_Entry]:
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            root_before = os.fstat(reader.descriptor)
            if (
                required_root_mode is not None
                and stat.S_IMODE(root_before.st_mode) != required_root_mode
            ):
                _fail(
                    "M4_ARCHIVE_MODE_SEAL",
                    f"archive root mode is "
                    f"{stat.S_IMODE(root_before.st_mode):#05o}; expected "
                    f"{required_root_mode:#05o}",
                )
            entries: list[_Entry] = []
            _scan_directory(reader.descriptor, (), entries)
            root_after = os.fstat(reader.descriptor)
            reader.verify_path()
            root_final = os.fstat(reader.descriptor)
            if (
                _snapshot(root_before) != _snapshot(root_after)
                or _snapshot(root_after) != _snapshot(root_final)
            ):
                _fail("M4_ARCHIVE_TREE_CHANGED", str(root))
            return sorted(
                entries,
                key=lambda entry: os.fsencode(entry.relative_path),
            )
    except M4ArchiveSealError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"{root}: {exc}")


def _entry_map(entries: Iterable[_Entry]) -> dict[str, _Entry]:
    result: dict[str, _Entry] = {}
    for entry in entries:
        if entry.relative_path in result:
            _fail("M4_ARCHIVE_TREE_SNAPSHOT", f"duplicate path: {entry.relative_path}")
        result[entry.relative_path] = entry
    return result


def _inventory_models(entries: Sequence[_Entry]) -> list[dict[str, Any]]:
    return [entry.model() for entry in entries]


def _inventory_hash(entries: Sequence[_Entry]) -> str:
    return _canonical_sha256(_inventory_models(entries))


def _strict_json(payload: bytes, *, role: str) -> Any:
    try:
        return immutable.strict_json_loads(payload, label=role)
    except immutable.ImmutableArtifactError as exc:
        _fail("M4_ARCHIVE_DONE_INVALID", f"{role}: {exc}")


def _load_schema() -> dict[str, Any]:
    try:
        raw = immutable.read_regular(SCHEMA_PATH)
        value = immutable.strict_json_loads(raw, label=str(SCHEMA_PATH))
        if not isinstance(value, dict):
            _fail("M4_ARCHIVE_SCHEMA", "receipt schema is not an object")
        if value.get("$id") != "simbench_bw/schemas/m4_archive_seal_receipt.schema.json":
            _fail("M4_ARCHIVE_SCHEMA", "receipt schema id mismatch")
        Draft202012Validator.check_schema(value)
        return value
    except M4ArchiveSealError:
        raise
    except (OSError, immutable.ImmutableArtifactError, json.JSONDecodeError) as exc:
        _fail("M4_ARCHIVE_SCHEMA", str(exc))


def _validate_receipt(value: Mapping[str, Any]) -> None:
    try:
        schema = _load_schema()
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    except M4ArchiveSealError:
        raise
    except Exception as exc:
        _fail("M4_ARCHIVE_RECEIPT_SCHEMA", str(exc))


def _validate_chunk_ids(values: Sequence[str]) -> tuple[str, ...]:
    if len(values) != EXPECTED_CHUNK_COUNT or len(set(values)) != len(values):
        _fail("M4_ARCHIVE_CHUNK_INVENTORY", f"expected exactly {EXPECTED_CHUNK_COUNT} unique chunks")
    result: list[str] = []
    for value in values:
        if type(value) is not str or not value:
            _fail("M4_ARCHIVE_CHUNK_INVENTORY", repr(value))
        try:
            pure = immutable.safe_relative_path(value)
        except immutable.ImmutableArtifactError as exc:
            _fail("M4_ARCHIVE_CHUNK_INVENTORY", f"{value!r}: {exc}")
        if len(pure.parts) != 1 or pure.name != value:
            _fail("M4_ARCHIVE_CHUNK_INVENTORY", repr(value))
        result.append(value)
    return tuple(sorted(result, key=os.fsencode))


def _matrix_chunk_ids(matrix_path: Path) -> tuple[str, ...]:
    try:
        payload = immutable.read_regular(matrix_path)
        matrix = _strict_json(payload, role="capture matrix")
    except M4ArchiveSealError:
        raise
    except immutable.ImmutableArtifactError as exc:
        _fail("M4_ARCHIVE_MATRIX", str(exc))
    if not isinstance(matrix, dict) or not isinstance(matrix.get("chunks"), list):
        _fail("M4_ARCHIVE_MATRIX", "matrix chunks are absent")
    values = [
        row.get("chunk_id") for row in matrix["chunks"]
        if isinstance(row, dict)
    ]
    return _validate_chunk_ids(values)


def _check_topology(
    entries: Sequence[_Entry], *, archive_root: Path, chunk_ids: Sequence[str]
) -> dict[str, _Entry]:
    by_path = _entry_map(entries)
    chunks = by_path.get("chunks")
    if chunks is None or chunks.kind != "directory":
        _fail("M4_ARCHIVE_CHUNK_INVENTORY", "missing chunks directory")
    top_level = {
        PurePosixPath(entry.relative_path).parts[0]
        for entry in entries
        if entry.relative_path
    }
    if top_level != {"chunks"}:
        _fail("M4_ARCHIVE_CHUNK_INVENTORY", f"unexpected top-level entries: {sorted(top_level)}")
    expected = set(chunk_ids)
    actual = {
        PurePosixPath(entry.relative_path).parts[1]
        for entry in entries
        if len(PurePosixPath(entry.relative_path).parts) >= 2
        and PurePosixPath(entry.relative_path).parts[0] == "chunks"
    }
    if actual != expected:
        _fail(
            "M4_ARCHIVE_CHUNK_INVENTORY",
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}",
        )
    for chunk_id in chunk_ids:
        path = f"chunks/{chunk_id}"
        entry = by_path.get(path)
        if entry is None or entry.kind != "directory":
            _fail("M4_ARCHIVE_CHUNK_INVENTORY", path)
    return by_path


def _verify_done_manifests(
    *,
    entries: Mapping[str, _Entry],
    archive_root: Path,
    chunk_ids: Sequence[str],
    repo_root: Path,
    verifier: Callable[..., list[str]] | None,
) -> list[dict[str, Any]]:
    check = done.verify_seal if verifier is None else verifier
    result: list[dict[str, Any]] = []
    for chunk_id in chunk_ids:
        relative = f"chunks/{chunk_id}/DONE.json"
        entry = entries.get(relative)
        if entry is None or entry.kind != "file" or entry.payload is None:
            _fail("M4_ARCHIVE_DONE_MISSING", relative)
        manifest = _strict_json(entry.payload, role=f"M4 DONE {chunk_id}")
        if not isinstance(manifest, dict):
            _fail("M4_ARCHIVE_DONE_INVALID", chunk_id)
        if manifest.get("chunk_id") != chunk_id:
            _fail("M4_ARCHIVE_DONE_BINDING", chunk_id)
        output_root = manifest.get("output_root")
        if isinstance(output_root, str) and output_root:
            expected = os.path.abspath(os.fspath(archive_root / "chunks" / chunk_id))
            if os.path.abspath(output_root) != expected:
                _fail("M4_ARCHIVE_DONE_BINDING", f"{chunk_id}: output_root")
        try:
            problems = check(
                manifest,
                repo_root=repo_root,
                rehash_references=True,
                require_current_head=True,
                require_clean_tree=True,
            )
        except Exception as exc:
            _fail("M4_ARCHIVE_DONE_VERIFIER_FAILURE", f"{chunk_id}: {exc}")
        if not isinstance(problems, list):
            _fail("M4_ARCHIVE_DONE_VERIFIER_FAILURE", f"{chunk_id}: invalid verifier result")
        if problems:
            _fail("M4_ARCHIVE_DONE_UNVERIFIED", f"{chunk_id}: {'; '.join(problems)}")
        result.append({
            "chunk_id": chunk_id,
            "relative_path": relative,
            "sha256": _sha256(entry.payload),
            "size_bytes": len(entry.payload),
        })
    return result


def _assert_same_tree(before: Sequence[_Entry], after: Sequence[_Entry], *, modes: bool) -> None:
    left = _entry_map(before)
    right = _entry_map(after)
    if set(left) != set(right):
        _fail("M4_ARCHIVE_TREE_CHANGED", "archive path set changed")
    for path in sorted(left, key=os.fsencode):
        a, b = left[path], right[path]
        if (
            a.kind != b.kind
            or a.sha256 != b.sha256
            or a.size_bytes != b.size_bytes
            or a.device != b.device
            or a.inode != b.inode
            or a.nlink != b.nlink
            or (modes and a.mode != b.mode)
        ):
            _fail("M4_ARCHIVE_TREE_CHANGED", path)


def _open_child_for_mode(parent: int, name: str, *, directory: bool) -> int:
    flags = _directory_flags() if directory else _file_flags()
    try:
        linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent)
        opened = os.fstat(descriptor)
    except OSError as exc:
        _fail("M4_ARCHIVE_MODE_SEAL", f"{name}: {exc}")
    if (
        stat.S_ISLNK(linked.st_mode)
        or (directory and not stat.S_ISDIR(linked.st_mode))
        or (not directory and not stat.S_ISREG(linked.st_mode))
        or not _node_equal(linked, opened)
    ):
        os.close(descriptor)
        _fail("M4_ARCHIVE_TREE_CHANGED", name)
    if not directory and opened.st_nlink != 1:
        os.close(descriptor)
        _fail("M4_ARCHIVE_HARDLINK", name)
    return descriptor


def _seal_modes_directory(
    descriptor: int,
    parts: tuple[str, ...],
    *,
    seal_self: bool = True,
) -> None:
    try:
        before = os.fstat(descriptor)
        with os.scandir(descriptor) as iterator:
            children = sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in iterator),
                key=lambda row: os.fsencode(row[0]),
            )
    except OSError as exc:
        _fail("M4_ARCHIVE_MODE_SEAL", str(exc))
    for name, linked in children:
        if stat.S_ISLNK(linked.st_mode):
            _fail("M4_ARCHIVE_SYMLINK", PurePosixPath(*(parts + (name,))).as_posix())
        child_parts = (*parts, name)
        if stat.S_ISDIR(linked.st_mode):
            child = _open_child_for_mode(descriptor, name, directory=True)
            try:
                _seal_modes_directory(child, child_parts)
                os.fchmod(child, 0o555)
                os.fsync(child)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not _node_equal(current, os.fstat(child)):
                    _fail("M4_ARCHIVE_TREE_CHANGED", PurePosixPath(*child_parts).as_posix())
            except OSError as exc:
                _fail("M4_ARCHIVE_MODE_SEAL", f"{PurePosixPath(*child_parts)}: {exc}")
            finally:
                os.close(child)
        elif stat.S_ISREG(linked.st_mode):
            child = _open_child_for_mode(descriptor, name, directory=False)
            try:
                if os.fstat(child).st_nlink != 1:
                    _fail("M4_ARCHIVE_HARDLINK", PurePosixPath(*child_parts).as_posix())
                os.fchmod(child, 0o444)
                os.fsync(child)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                opened = os.fstat(child)
                if not _node_equal(current, opened) or current.st_nlink != 1:
                    _fail("M4_ARCHIVE_TREE_CHANGED", PurePosixPath(*child_parts).as_posix())
            except OSError as exc:
                _fail("M4_ARCHIVE_MODE_SEAL", f"{PurePosixPath(*child_parts)}: {exc}")
            finally:
                os.close(child)
        else:
            _fail("M4_ARCHIVE_NONREGULAR", PurePosixPath(*child_parts).as_posix())
    try:
        if seal_self:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        after = os.fstat(descriptor)
    except OSError as exc:
        _fail("M4_ARCHIVE_MODE_SEAL", f"{PurePosixPath(*parts) or '.'}: {exc}")
    if not _node_equal(before, after):
        _fail("M4_ARCHIVE_TREE_CHANGED", PurePosixPath(*parts).as_posix() or ".")


def _seal_modes(root: Path) -> None:
    try:
        with immutable.PinnedDirectoryReader(root) as reader:
            _seal_modes_directory(reader.descriptor, (), seal_self=False)
            reader.verify_path()
            reader.seal_root_directory_mode()
    except M4ArchiveSealError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M4_ARCHIVE_MODE_SEAL", str(exc))


def _receipt_model(
    *,
    archive_root: Path,
    entries: Sequence[_Entry],
    done_manifests: Sequence[Mapping[str, Any]],
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    files = sum(entry.kind == "file" for entry in entries)
    directories = sum(entry.kind == "directory" for entry in entries)
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "m4_archive_seal_receipt",
        "tool_id": TOOL_ID,
        "status": "complete",
        "sealed_at_utc": sealed_at_utc or _now(),
        "archive_root": os.fspath(archive_root),
        "chunk_count": EXPECTED_CHUNK_COUNT,
        "file_count": files,
        "directory_count": directories,
        "chunk_done_manifests": [dict(row) for row in done_manifests],
        "inventory": _inventory_models(entries),
        "inventory_sha256": _inventory_hash(entries),
    }
    value["receipt_sha256"] = immutable.model_seal(value, "receipt_sha256")
    _validate_receipt(value)
    return value


def _receipt_path(
    archive_root: Path, receipt_path: Path | str | None,
) -> Path:
    if receipt_path is None:
        candidate = archive_root.parent / f"{archive_root.name}.{DEFAULT_RECEIPT_NAME}"
    else:
        candidate = _absolute(receipt_path, role="receipt path")
    candidate = _absolute(candidate, role="receipt path")
    if _is_under(archive_root, candidate) or _is_under(candidate, archive_root):
        _fail("M4_ARCHIVE_RECEIPT_OVERLAP", str(candidate))
    if candidate.name in {"", ".", ".."}:
        _fail("M4_ARCHIVE_PATH", str(candidate))
    return candidate


def _read_existing_receipt(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(path))
        metadata = path.stat()
        if metadata.st_nlink != 1:
            _fail("M4_ARCHIVE_RECEIPT_COLLISION", f"hard link: {path}")
        value = immutable.strict_json_load(path)
        if not isinstance(value, dict):
            _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(path))
        immutable.verify_model_seal(value, "receipt_sha256")
        _validate_receipt(value)
        return value
    except M4ArchiveSealError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", f"{path}: {exc}")


def _receipt_matches(
    value: Mapping[str, Any], *, archive_root: Path, entries: Sequence[_Entry],
    done_manifests: Sequence[Mapping[str, Any]],
) -> bool:
    return (
        value.get("archive_root") == os.fspath(archive_root)
        and value.get("chunk_count") == EXPECTED_CHUNK_COUNT
        and value.get("file_count") == sum(entry.kind == "file" for entry in entries)
        and value.get("directory_count") == sum(entry.kind == "directory" for entry in entries)
        and value.get("inventory_sha256") == _inventory_hash(entries)
        and value.get("inventory") == _inventory_models(entries)
        and value.get("chunk_done_manifests") == [dict(row) for row in done_manifests]
    )


def _receipt_publication_semantics(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"sealed_at_utc", "receipt_sha256"}
    }


def _recover_receipt_publication_candidate(
    path: Path,
    fresh: Mapping[str, Any],
) -> dict[str, Any]:
    partial = path.with_name(f".{path.name}.m6-partial")
    maximum_bytes = len(immutable.pretty_json_bytes(fresh)) + 1024
    try:
        metadata = partial.lstat()
    except FileNotFoundError:
        return dict(fresh)
    except OSError as exc:
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", f"{partial}: {exc}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(partial))
    try:
        snapshot = immutable.read_regular_snapshot(
            partial,
            max_bytes=maximum_bytes,
        )
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", f"{partial}: {exc}")
    try:
        candidate = immutable.strict_json_loads(
            snapshot.payload,
            label="M4 archive receipt publication partial",
        )
        if not isinstance(candidate, dict):
            _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(partial))
        assert isinstance(candidate, dict)
        immutable.verify_model_seal(candidate, "receipt_sha256")
        _validate_receipt(candidate)
    except M4ArchiveSealError as exc:
        if exc.code == "M4_ARCHIVE_SCHEMA":
            raise
        _fail(
            "M4_ARCHIVE_RECEIPT_COLLISION",
            f"{partial}: {exc}",
        )
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", f"{partial}: {exc}")
    if (
        _receipt_publication_semantics(candidate)
        != _receipt_publication_semantics(fresh)
    ):
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(partial))
    return candidate


def _accept_receipt_publication_winner(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    existing = _read_existing_receipt(path)
    if (
        _receipt_publication_semantics(existing)
        != _receipt_publication_semantics(expected)
    ):
        _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(path))
    return existing


def _publish_receipt(
    path: Path,
    value: Mapping[str, Any],
    *,
    before_commit: Callable[[], None] | None = None,
    after_commit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    try:
        immutable.ensure_directory(path.parent)
        immutable.publish_regular_file(
            path.parent,
            path.name,
            immutable.pretty_json_bytes(value),
            partial_suffix=".m6-partial",
            before_commit=before_commit,
            after_commit=after_commit,
        )
        return dict(value)
    except FileExistsError:
        return _accept_receipt_publication_winner(path, value)
    except immutable.ImmutableArtifactError as exc:
        if "publication already exists" in str(exc):
            return _accept_receipt_publication_winner(path, value)
        _fail("M4_ARCHIVE_RECEIPT_PUBLICATION", str(exc))


def _publish_receipt_while_guarding_archive(
    *,
    archive_root: Path,
    entries: Sequence[_Entry],
    receipt_path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    file_entries = [entry for entry in entries if entry.kind == "file"]
    directory_entries = [
        entry for entry in entries if entry.kind == "directory"
    ]
    limits = {
        "max_files": len(file_entries),
        "max_directories": len(directory_entries) + 1,
        "max_total_file_bytes": sum(
            entry.size_bytes for entry in file_entries
        ),
        "max_total_relative_path_bytes": sum(
            len(os.fsencode(entry.relative_path)) for entry in entries
        ),
    }
    monitor: immutable.PinnedPathSetMonitor | None = None
    try:
        monitor = immutable.PinnedPathSetMonitor(
            archive_root,
            regular_paths=(),
            recursive_directories=(".",),
            **limits,
        )
        guarded_entries = _scan_tree(
            archive_root,
            required_root_mode=0o555,
        )
        _assert_same_tree(entries, guarded_entries, modes=True)

        def verify_archive() -> None:
            current = monitor
            if current is None:
                _fail(
                    "M4_ARCHIVE_TREE_CHANGED",
                    "archive publication guard is closed",
                )
            assert current is not None
            try:
                current.verify_unchanged()
            except (OSError, immutable.ImmutableArtifactError) as exc:
                _fail("M4_ARCHIVE_TREE_CHANGED", str(exc))

        def finalize_archive_guard() -> None:
            nonlocal monitor
            verify_archive()
            assert monitor is not None
            monitor.close()
            monitor = None

        verify_archive()
        published = _publish_receipt(
            receipt_path,
            value,
            before_commit=verify_archive,
            after_commit=finalize_archive_guard,
        )
        if monitor is not None:
            # A concurrent exact receipt winner bypasses producer callbacks.
            finalize_archive_guard()
        return published
    except M4ArchiveSealError:
        raise
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M4_ARCHIVE_TREE_CHANGED", str(exc))
    finally:
        if monitor is not None:
            active_error = sys.exc_info()[1]
            try:
                monitor.close()
            except BaseException as close_error:
                if active_error is not None:
                    try:
                        active_error.add_note(
                            f"archive publication guard close failed: "
                            f"{close_error}"
                        )
                    except (AttributeError, TypeError):
                        pass
                else:
                    _fail("M4_ARCHIVE_TREE_CHANGED", str(close_error))


def seal_m4_archive(
    archive_root: Path | str,
    *,
    receipt_path: Path | str | None = None,
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    repo_root: Path | str = REPO_ROOT,
    expected_chunk_ids: Sequence[str] | None = None,
    done_verifier: Callable[..., list[str]] | None = None,
) -> dict[str, Any]:
    """Verify and immutably seal one external 18-chunk M4 archive.

    ``expected_chunk_ids`` is an explicit offline-fixture escape hatch.  It
    still must contain exactly 18 unique safe names; production callers leave
    it unset so the canonical capture matrix supplies the closed roster.
    ``done_verifier`` exists for deterministic fixture tests and is never
    supplied by the CLI.
    """
    root = _external_archive_root(Path(archive_root), repo_root=Path(repo_root))
    chunk_ids = (
        _validate_chunk_ids(expected_chunk_ids)
        if expected_chunk_ids is not None
        else _matrix_chunk_ids(_absolute(matrix_path, role="matrix path"))
    )
    output_receipt = _receipt_path(root, receipt_path)
    receipt_present = output_receipt.exists() or output_receipt.is_symlink()

    entries = _scan_tree(
        root,
        required_root_mode=0o555 if receipt_present else None,
    )
    by_path = _check_topology(entries, archive_root=root, chunk_ids=chunk_ids)
    done_manifests = _verify_done_manifests(
        entries=by_path,
        archive_root=root,
        chunk_ids=chunk_ids,
        repo_root=Path(repo_root),
        verifier=done_verifier,
    )

    existing: dict[str, Any] | None = None
    if receipt_present:
        existing = _read_existing_receipt(output_receipt)
        if not _receipt_matches(
            existing, archive_root=root, entries=entries,
            done_manifests=done_manifests,
        ):
            _fail("M4_ARCHIVE_RECEIPT_COLLISION", str(output_receipt))

    # No mode mutation is allowed until every DONE and every tree identity has
    # been captured.  A second scan closes the authorization-to-mutation gap.
    before = _scan_tree(
        root,
        required_root_mode=0o555 if existing is not None else None,
    )
    _assert_same_tree(entries, before, modes=True)
    if existing is None:
        # Exercise the complete receipt/schema/serialization model before any
        # mode mutation.  The final receipt is rebuilt from sealed entries.
        _receipt_model(
            archive_root=root,
            entries=before,
            done_manifests=done_manifests,
        )
        _seal_modes(root)
        sealed_entries = _scan_tree(root, required_root_mode=0o555)
        _assert_same_tree(before, sealed_entries, modes=False)
        for entry in sealed_entries:
            if entry.kind == "file" and (entry.mode != 0o444 or entry.nlink != 1):
                _fail("M4_ARCHIVE_MODE_SEAL", entry.relative_path)
            if entry.kind == "directory" and entry.mode != 0o555:
                _fail("M4_ARCHIVE_MODE_SEAL", entry.relative_path)
        fresh_value = _receipt_model(
            archive_root=root,
            entries=sealed_entries,
            done_manifests=done_manifests,
        )
        value = _recover_receipt_publication_candidate(
            output_receipt,
            fresh_value,
        )
        return _publish_receipt_while_guarding_archive(
            archive_root=root,
            entries=sealed_entries,
            receipt_path=output_receipt,
            value=value,
        )

    # Existing receipts are returned byte-for-byte semantically unchanged.  We
    # still revalidated the current tree and DONEs above, so they cannot mask a
    # modified archive.
    sealed_entries = _scan_tree(root, required_root_mode=0o555)
    _assert_same_tree(entries, sealed_entries, modes=True)
    return existing


# Common spelling used by operator wrappers.
seal_archive = seal_m4_archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        value = seal_m4_archive(
            args.archive_root,
            receipt_path=args.receipt,
            matrix_path=args.matrix,
            repo_root=args.repo_root,
        )
    except M4ArchiveSealError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"status={value['status']} archive_root={value['archive_root']} "
        f"receipt_sha256={value['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
