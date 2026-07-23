#!/usr/bin/env python3
"""Copy one quiescent supplemental SOL campaign into durable evidence storage.

The live campaign root is evidence, not a staging directory to consume.  This
publisher therefore never renames, truncates, unlinks, chmods, or otherwise
mutates the source tree.  It inventories the stopped tree, copies it into a
private destination-side partial directory, proves byte and hard-link
topology equality, runs portable offline verification, and only then performs
one same-directory ``RENAME_NOREPLACE`` into the durable namespace.

An interrupted transaction is resumable only when every already-published
entry is a complete, exact prefix of the sealed inventory.  Torn, extra, or
contradictory partial content is preserved and refused; this module never
"repairs" such evidence by deleting it.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_NAMESPACE = "supplemental_sol_campaigns"
ARCHIVE_MANIFEST = "ARCHIVE_MANIFEST.json"
ARCHIVED_CAMPAIGN = "campaign"
SOURCE_LOCK = ".sol_workload_campaign.lock"
CHUNK_LOCK = ".capture.lock"
ROOT_MARKER = ".sol_workload_capture_root.json"
ROOT_KIND = "sol_workload_capture_supplemental_v1"
ARCHIVE_KIND = "supplemental_sol_campaign_archive_v1"
TRANSACTION_PROTOCOL = "source-preserving-cross-device-copy-v1"
ARCHIVE_LOCK = ".archive.lock"
CAPACITY_RESERVE_BYTES = 4 * 1024 ** 3
CANONICAL_LIVE_BASE = Path("/var/tmp")
CANONICAL_ARCHIVE_BASE = Path("/home/jakeshea/sm86_reference_capture_r27")
CANONICAL_GPU_LOCK = Path(f"/run/user/{os.getuid()}/solbench-sm86-gpu.lock")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ArchiveError(RuntimeError):
    """A stable fail-closed archive refusal."""


@dataclasses.dataclass(frozen=True)
class ArchiveSpec:
    source_root: Path
    archive_base: Path
    source_head: str
    source_h8: str
    campaign_identity_sha256: str
    gpu_lock_path: Path
    repo_root: Path = REPO_ROOT
    test_only_allow_noncanonical_paths: bool = False

    def normalized(self) -> "ArchiveSpec":
        return dataclasses.replace(
            self,
            source_root=_absolute(self.source_root),
            archive_base=_absolute(self.archive_base),
            gpu_lock_path=_absolute(self.gpu_lock_path),
            repo_root=_absolute(self.repo_root),
        )


PortableVerifier = Callable[[Path], None]
GpuProbe = Callable[[], list[int]]


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_canonical_object(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    _require_regular(path, role)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArchiveError(f"{role} is not a regular file: {path}")
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
            if (
                before.st_dev, before.st_ino, before.st_mode, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns,
            ) != (
                after.st_dev, after.st_ino, after.st_mode, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            ):
                raise ArchiveError(f"{role} changed during read: {path}")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
        value = json.loads(payload, object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"{role} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"{role} root is not an object: {path}")
    if payload != _pretty_bytes(value):
        raise ArchiveError(f"{role} bytes are not canonical: {path}")
    return value, payload


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _assert_chain_no_symlinks(path: Path, *, leaf_may_be_missing: bool = False) -> None:
    absolute = _absolute(path)
    parts = absolute.parts
    current = Path(parts[0])
    for index, component in enumerate(parts[1:], start=1):
        current /= component
        try:
            observed = current.lstat()
        except FileNotFoundError:
            if leaf_may_be_missing and index == len(parts) - 1:
                return
            raise ArchiveError(f"required path is absent: {current}")
        if stat.S_ISLNK(observed.st_mode):
            raise ArchiveError(f"symlink path component is forbidden: {current}")


def _require_directory(path: Path, role: str) -> os.stat_result:
    _assert_chain_no_symlinks(path)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode):
        raise ArchiveError(f"{role} is not a directory: {path}")
    return observed


def _require_regular(path: Path, role: str) -> os.stat_result:
    _assert_chain_no_symlinks(path)
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode):
        raise ArchiveError(f"{role} is not a regular file: {path}")
    return observed


def _safe_relative(value: str, *, role: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ArchiveError(f"unsafe {role}: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ArchiveError(f"unsafe {role}: {value!r}")
    return relative


def _hash_file_exact(
    path: Path, expected: os.stat_result | None = None,
) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if expected is not None and (
            before.st_dev != expected.st_dev
            or before.st_ino != expected.st_ino
            or before.st_mode != expected.st_mode
            or before.st_size != expected.st_size
            or before.st_mtime_ns != expected.st_mtime_ns
            or before.st_ctime_ns != expected.st_ctime_ns
        ):
            raise ArchiveError(f"source changed before read: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise ArchiveError(f"source changed during read: {path}")
        return size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory exact bytes and the complete regular-file inode relation."""
    root_info = _require_directory(root, "campaign source root")
    if root_info.st_uid != os.geteuid():
        raise ArchiveError(f"campaign source root owner is invalid: {root}")
    root_device = root_info.st_dev
    raw: list[tuple[str, os.stat_result, str | None]] = [(".", root_info, None)]

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise ArchiveError(f"cannot inventory directory: {directory}: {exc}") from exc
        for entry in entries:
            relative = prefix / entry.name
            candidate = Path(entry.path)
            try:
                observed = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArchiveError(f"cannot stat campaign entry: {candidate}: {exc}") from exc
            if observed.st_uid != os.geteuid():
                raise ArchiveError(f"campaign entry owner is invalid: {candidate}")
            if observed.st_dev != root_device:
                raise ArchiveError(f"nested mount/device is forbidden: {candidate}")
            if stat.S_ISLNK(observed.st_mode):
                raise ArchiveError(f"symlink is forbidden: {candidate}")
            if stat.S_ISDIR(observed.st_mode):
                raw.append((relative.as_posix(), observed, None))
                visit(candidate, relative)
            elif stat.S_ISREG(observed.st_mode):
                size, digest = _hash_file_exact(candidate, observed)
                if size != observed.st_size:
                    raise ArchiveError(f"source size changed during inventory: {candidate}")
                raw.append((relative.as_posix(), observed, digest))
            else:
                raise ArchiveError(f"special file is forbidden: {candidate}")

    visit(root, PurePosixPath())
    raw.sort(key=lambda item: os.fsencode(item[0]))
    file_paths: dict[tuple[int, int], list[str]] = {}
    for relative, observed, _digest in raw:
        if stat.S_ISREG(observed.st_mode):
            file_paths.setdefault((observed.st_dev, observed.st_ino), []).append(relative)
    groups = {
        identity: f"inode-{ordinal:08d}"
        for ordinal, (identity, _paths) in enumerate(
            sorted(file_paths.items(), key=lambda item: os.fsencode(item[1][0])),
            start=1,
        )
    }
    for identity, paths in file_paths.items():
        source_nlink = next(
            observed.st_nlink for _relative, observed, _digest in raw
            if (observed.st_dev, observed.st_ino) == identity
        )
        if source_nlink != len(paths):
            raise ArchiveError(
                "regular-file hard link escapes campaign root: "
                f"{paths[0]} nlink={source_nlink} in_root={len(paths)}"
            )
    rows: list[dict[str, Any]] = []
    for relative, observed, digest in raw:
        common = {
            "path": relative,
            "mode": stat.S_IMODE(observed.st_mode),
            "source_device": observed.st_dev,
            "source_inode": observed.st_ino,
            "source_mtime_ns": observed.st_mtime_ns,
            "source_ctime_ns": observed.st_ctime_ns,
        }
        if stat.S_ISDIR(observed.st_mode):
            rows.append({**common, "type": "directory"})
        else:
            identity = (observed.st_dev, observed.st_ino)
            rows.append({
                **common,
                "type": "file",
                "size": observed.st_size,
                "sha256": digest,
                "hardlink_group": groups[identity],
                "hardlink_count": len(file_paths[identity]),
                "source_nlink": observed.st_nlink,
            })
    return rows


def _inventory_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    files = [row for row in rows if row["type"] == "file"]
    groups = {row["hardlink_group"] for row in files}
    return {
        "directories": sum(row["type"] == "directory" for row in rows),
        "files": len(files),
        "logical_bytes": sum(row["size"] for row in files),
        "physical_bytes": sum(
            members[0]["size"]
            for group in groups
            if (members := [row for row in files if row["hardlink_group"] == group])
        ),
        "unique_file_inodes": len(groups),
        "hardlink_groups": sum(
            1 for group in groups
            if next(row for row in files if row["hardlink_group"] == group)[
                "hardlink_count"
            ] > 1
        ),
    }


def _logical_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        common = {"path": row["path"], "type": row["type"], "mode": row["mode"]}
        if row["type"] == "file":
            common.update({
                "size": row["size"],
                "sha256": row["sha256"],
                "hardlink_group": row["hardlink_group"],
                "hardlink_count": row["hardlink_count"],
            })
        result.append(common)
    return result


def _evidence_inventory_sha256(rows: list[dict[str, Any]]) -> str:
    # Lock-file payloads are coordination state (typically a last-writer PID),
    # not campaign evidence.  They are still copied and byte-verified by the
    # full archive inventory, but cannot manufacture a new checkpoint identity
    # when a launcher merely reacquires the same stopped checkpoint.
    evidence = [
        row for row in _logical_inventory(rows)
        if row["path"] not in {SOURCE_LOCK, CHUNK_LOCK}
    ]
    return _canonical_hash(evidence)


def _checkpoint_id(
    spec: ArchiveSpec, *, rows: list[dict[str, Any]], frontier: dict[str, Any],
) -> tuple[str, str]:
    inventory_sha256 = _evidence_inventory_sha256(rows)
    checkpoint = _canonical_hash({
        "kind": "supplemental_sol_campaign_checkpoint_v1",
        "source_commit": spec.source_head,
        "campaign_identity_sha256": spec.campaign_identity_sha256,
        "frontier": frontier,
        "evidence_inventory_sha256": inventory_sha256,
    })
    return checkpoint, inventory_sha256


def _verify_seal(document: dict[str, Any], field: str, *, role: str) -> None:
    seal = document.get(field)
    body = {key: value for key, value in document.items() if key != field}
    if not isinstance(seal, str) or seal != _canonical_hash(body):
        raise ArchiveError(f"{role} seal mismatch")


def _load_root_marker(campaign_root: Path) -> tuple[dict[str, Any], bytes]:
    marker, payload = _load_canonical_object(
        campaign_root / ROOT_MARKER, role="supplemental campaign root marker"
    )
    if set(marker) != {
        "schema_version", "root_kind", "campaign_identity",
        "campaign_identity_sha256", "created_at_utc", "root_identity_sha256",
    } or marker.get("schema_version") != 1 or marker.get("root_kind") != ROOT_KIND:
        raise ArchiveError("supplemental campaign root marker shape is invalid")
    _verify_seal(marker, "root_identity_sha256", role="supplemental root marker")
    identity = marker.get("campaign_identity")
    if (
        not isinstance(identity, dict)
        or marker.get("campaign_identity_sha256") != _canonical_hash(identity)
    ):
        raise ArchiveError("supplemental campaign identity binding is invalid")
    observed = (campaign_root / ROOT_MARKER).lstat()
    if observed.st_uid != os.geteuid() or observed.st_nlink != 1 or observed.st_mode & 0o222:
        raise ArchiveError("supplemental campaign root marker mutability is invalid")
    return marker, payload


def _verify_campaign_identity(spec: ArchiveSpec) -> tuple[dict[str, Any], str]:
    marker, payload = _load_root_marker(spec.source_root)
    identity = marker["campaign_identity"]
    if identity.get("source_commit") != spec.source_head:
        raise ArchiveError("campaign identity source_commit mismatch")
    if marker["campaign_identity_sha256"] != spec.campaign_identity_sha256:
        raise ArchiveError("campaign identity SHA-256 mismatch")
    return marker, _sha256_bytes(payload)


def _validate_spec(spec: ArchiveSpec) -> ArchiveSpec:
    spec = spec.normalized()
    if HEX40.fullmatch(spec.source_head) is None:
        raise ArchiveError("source_head must be lowercase 40-hex")
    if spec.source_h8 != spec.source_head[:8]:
        raise ArchiveError("source_h8 must equal the first eight source_head characters")
    if HEX64.fullmatch(spec.campaign_identity_sha256) is None:
        raise ArchiveError("campaign_identity_sha256 must be lowercase 64-hex")
    source_info = _require_directory(spec.source_root, "campaign source root")
    archive_info = _require_directory(spec.archive_base, "durable archive base")
    _require_directory(spec.repo_root, "verification repository root")
    if spec.source_root.name != "capture" or not spec.source_root.parent.name.endswith(
        f"r28_{spec.source_h8}"
    ):
        raise ArchiveError("campaign source path is not bound to source_h8")
    if spec.archive_base.name != "sm86_reference_capture_r27":
        raise ArchiveError("durable archive base has the wrong namespace identity")
    if not spec.test_only_allow_noncanonical_paths:
        expected_source = Path(
            CANONICAL_LIVE_BASE
            / f"jakeshea-solbench-r28_{spec.source_h8}/capture"
        )
        if spec.source_root != expected_source:
            raise ArchiveError(
                f"production source root must be exactly {expected_source}"
            )
        if spec.archive_base != CANONICAL_ARCHIVE_BASE:
            raise ArchiveError(
                f"production archive base must be exactly {CANONICAL_ARCHIVE_BASE}"
            )
        if spec.gpu_lock_path != CANONICAL_GPU_LOCK:
            raise ArchiveError(
                f"production GPU lock must be exactly {CANONICAL_GPU_LOCK}"
            )
        if spec.repo_root != REPO_ROOT:
            raise ArchiveError(
                f"production verification checkout must be exactly {REPO_ROOT}"
            )
    if source_info.st_uid != os.geteuid() or stat.S_IMODE(source_info.st_mode) != 0o700:
        raise ArchiveError("campaign source root must be current-user private mode 0700")
    if archive_info.st_uid != os.geteuid():
        raise ArchiveError("durable archive base owner is invalid")
    if spec.source_root == spec.archive_base:
        raise ArchiveError("source and archive roots must differ")
    if spec.source_root.is_relative_to(spec.archive_base) or spec.archive_base.is_relative_to(
        spec.source_root
    ):
        raise ArchiveError("source and archive roots must not overlap")
    _require_regular(spec.source_root / SOURCE_LOCK, "campaign writer lock")
    _require_regular(spec.source_root / CHUNK_LOCK, "chunk writer lock")
    _require_regular(spec.gpu_lock_path, "GPU campaign lock")
    _verify_campaign_identity(spec)
    return spec


def _probe_gpu_pids() -> list[int]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi", "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ),
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise ArchiveError(f"GPU quiescence probe failed: {detail!r}") from exc
    result: list[int] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            result.append(int(value))
        except ValueError as exc:
            raise ArchiveError(f"GPU quiescence probe returned invalid PID: {value!r}") from exc
    return sorted(result)


def _assert_gpu_quiescent(probe: GpuProbe) -> None:
    pids = probe()
    if pids:
        raise ArchiveError(f"GPU compute processes are active: {sorted(pids)!r}")


@contextlib.contextmanager
def _exclusive_existing_lock(path: Path, role: str) -> Iterator[None]:
    observed = _require_regular(path, role)
    if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) & 0o077:
        raise ArchiveError(f"{role} owner/mode is invalid: {path}")
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError(f"{role} is active: {path}") from exc
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _available_bytes(path: Path) -> int:
    observed = os.statvfs(path)
    return observed.f_bavail * observed.f_frsize


def _remaining_copy_bytes(partial: Path, rows: list[dict[str, Any]]) -> int:
    campaign = partial / ARCHIVED_CAMPAIGN
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["type"] == "file":
            groups.setdefault(row["hardlink_group"], []).append(row)
    remaining = 0
    for members in groups.values():
        canonical = min(members, key=lambda row: os.fsencode(row["path"]))
        destination = campaign / PurePosixPath(canonical["path"])
        if not _lexists(destination):
            remaining += canonical["size"]
    return remaining


def _require_copy_capacity(partial: Path, rows: list[dict[str, Any]]) -> None:
    available = _available_bytes(partial)
    required = _remaining_copy_bytes(partial, rows)
    if available < required + CAPACITY_RESERVE_BYTES:
        raise ArchiveError(
            "insufficient durable archive capacity for remaining copy: "
            f"available={available} payload={required} "
            f"reserve={CAPACITY_RESERVE_BYTES}"
        )


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short exclusive write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_archive_namespace(spec: ArchiveSpec) -> Path:
    namespace = spec.archive_base / ARCHIVE_NAMESPACE
    if not _lexists(namespace):
        namespace.mkdir(mode=0o700)
        _fsync_directory(spec.archive_base)
    observed = _require_directory(namespace, "supplemental archive namespace")
    if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) != 0o700:
        raise ArchiveError("supplemental archive namespace must be private mode 0700")
    for entry in os.scandir(namespace):
        info = entry.stat(follow_symlinks=False)
        if entry.name == ARCHIVE_LOCK:
            if not stat.S_ISREG(info.st_mode):
                raise ArchiveError("archive lock namespace entry is invalid")
            continue
        if HEX40.fullmatch(entry.name) is not None:
            if not stat.S_ISDIR(info.st_mode):
                raise ArchiveError(f"archive revision namespace entry is invalid: {entry.path}")
            continue
        raise ArchiveError(f"unexpected supplemental archive namespace entry: {entry.path}")
    return namespace


def _ensure_revision_namespace(namespace: Path, source_head: str) -> Path:
    revision = namespace / source_head
    if not _lexists(revision):
        revision.mkdir(mode=0o700)
        _fsync_directory(namespace)
    observed = _require_directory(revision, "supplemental revision archive namespace")
    if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) != 0o700:
        raise ArchiveError("supplemental revision namespace must be private mode 0700")
    for entry in os.scandir(revision):
        info = entry.stat(follow_symlinks=False)
        if HEX64.fullmatch(entry.name) is None and re.fullmatch(
            r"\.[0-9a-f]{64}\.partial", entry.name
        ) is None:
            raise ArchiveError(f"unexpected checkpoint archive entry: {entry.path}")
        if not stat.S_ISDIR(info.st_mode):
            raise ArchiveError(f"checkpoint archive entry is not a directory: {entry.path}")
    return revision


@contextlib.contextmanager
def _archive_lock(namespace: Path) -> Iterator[None]:
    path = namespace / ARCHIVE_LOCK
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ArchiveError(f"cannot open archive transaction lock: {path}: {exc}") from exc
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) & 0o077
        ):
            raise ArchiveError("archive transaction lock is invalid")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError("another supplemental archive transaction is active") from exc
        yield
    finally:
        os.close(descriptor)


def _seal_manifest(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["archive_manifest_seal_sha256"] = _canonical_hash(body)
    return result


def _build_manifest(
    spec: ArchiveSpec, *, rows: list[dict[str, Any]], marker_sha256: str,
    checkpoint_id: str, evidence_inventory_sha256: str,
    frontier: dict[str, Any],
) -> dict[str, Any]:
    namespace = spec.archive_base / ARCHIVE_NAMESPACE
    revision = namespace / spec.source_head
    final = revision / checkpoint_id
    summary = _inventory_summary(rows)
    capacity = {
        "source_available_bytes_at_inventory": _available_bytes(spec.source_root),
        "destination_available_bytes_at_inventory": _available_bytes(namespace),
        "required_unique_payload_bytes": summary["physical_bytes"],
        "reserved_destination_bytes": CAPACITY_RESERVE_BYTES,
    }
    if capacity["destination_available_bytes_at_inventory"] < (
        capacity["required_unique_payload_bytes"]
        + capacity["reserved_destination_bytes"]
    ):
        raise ArchiveError(
            "insufficient durable archive capacity: "
            f"available={capacity['destination_available_bytes_at_inventory']} "
            f"payload={capacity['required_unique_payload_bytes']} "
            f"reserve={capacity['reserved_destination_bytes']}"
        )
    return _seal_manifest({
        "schema_version": 1,
        "kind": ARCHIVE_KIND,
        "transaction_protocol": TRANSACTION_PROTOCOL,
        "source_root": os.fspath(spec.source_root),
        "archive_base": os.fspath(spec.archive_base),
        "archive_namespace": os.fspath(namespace),
        "archive_revision_namespace": os.fspath(revision),
        "archive_root": os.fspath(final),
        "archived_campaign_relative_path": ARCHIVED_CAMPAIGN,
        "source_commit": spec.source_head,
        "source_h8": spec.source_h8,
        "checkpoint_id": checkpoint_id,
        "checkpoint_frontier": frontier,
        "evidence_inventory_sha256": evidence_inventory_sha256,
        "campaign_identity_sha256": spec.campaign_identity_sha256,
        "root_marker_sha256": marker_sha256,
        "source_device": spec.source_root.stat().st_dev,
        "archive_device": namespace.stat().st_dev,
        "source_preserved": True,
        "inventory": rows,
        "inventory_summary": summary,
        "capacity_observation": capacity,
        "required_verifications": [
            "source_portable_pre_copy",
            "source_inventory_pre_copy",
            "destination_byte_and_hardlink_topology",
            "source_inventory_post_copy",
            "destination_portable_offline",
        ],
    })


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version", "kind", "transaction_protocol", "source_root",
        "archive_base", "archive_namespace", "archive_revision_namespace",
        "archive_root",
        "archived_campaign_relative_path", "source_commit", "source_h8",
        "checkpoint_id", "checkpoint_frontier", "evidence_inventory_sha256",
        "campaign_identity_sha256", "root_marker_sha256", "source_device",
        "archive_device", "source_preserved", "inventory", "inventory_summary",
        "capacity_observation", "required_verifications",
        "archive_manifest_seal_sha256",
    }
    if set(manifest) != required:
        raise ArchiveError("archive manifest top-level keys are not closed")
    _verify_seal(manifest, "archive_manifest_seal_sha256", role="archive manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != ARCHIVE_KIND
        or manifest.get("transaction_protocol") != TRANSACTION_PROTOCOL
        or manifest.get("archived_campaign_relative_path") != ARCHIVED_CAMPAIGN
        or manifest.get("source_preserved") is not True
    ):
        raise ArchiveError("archive manifest constants are invalid")
    head = manifest.get("source_commit")
    if not isinstance(head, str) or HEX40.fullmatch(head) is None:
        raise ArchiveError("archive manifest source_commit is invalid")
    if manifest.get("source_h8") != head[:8]:
        raise ArchiveError("archive manifest source_h8 is invalid")
    for field in (
        "campaign_identity_sha256", "root_marker_sha256", "checkpoint_id",
        "evidence_inventory_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise ArchiveError(f"archive manifest {field} is invalid")
    for field in (
        "source_root", "archive_base", "archive_namespace",
        "archive_revision_namespace", "archive_root",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ArchiveError(f"archive manifest {field} is invalid")
    base = Path(manifest["archive_base"])
    namespace = base / ARCHIVE_NAMESPACE
    revision = namespace / head
    if (
        Path(manifest["archive_namespace"]) != namespace
        or Path(manifest["archive_revision_namespace"]) != revision
        or Path(manifest["archive_root"]) != revision / manifest["checkpoint_id"]
    ):
        raise ArchiveError("archive manifest destination formula is invalid")
    if manifest.get("required_verifications") != [
        "source_portable_pre_copy",
        "source_inventory_pre_copy",
        "destination_byte_and_hardlink_topology",
        "source_inventory_post_copy",
        "destination_portable_offline",
    ]:
        raise ArchiveError("archive manifest verification contract is invalid")
    rows = manifest.get("inventory")
    if not isinstance(rows, list) or not rows:
        raise ArchiveError("archive manifest inventory is empty")
    paths: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ArchiveError("archive inventory row is not an object")
        row_type = row.get("type")
        directory_keys = {
            "path", "type", "mode", "source_device", "source_inode",
            "source_mtime_ns", "source_ctime_ns",
        }
        file_keys = directory_keys | {
            "size", "sha256", "hardlink_group", "hardlink_count", "source_nlink",
        }
        if set(row) != (directory_keys if row_type == "directory" else file_keys):
            raise ArchiveError("archive inventory row shape is invalid")
        relative = row.get("path")
        if relative != ".":
            if not isinstance(relative, str):
                raise ArchiveError("archive inventory path is invalid")
            _safe_relative(relative, role="archive inventory path")
        if not isinstance(relative, str) or relative in paths:
            raise ArchiveError("archive inventory paths are invalid or duplicated")
        paths.append(relative)
        for integer in (
            "mode", "source_device", "source_inode", "source_mtime_ns", "source_ctime_ns",
        ):
            if type(row.get(integer)) is not int or row[integer] < 0:
                raise ArchiveError(f"archive inventory {integer} is invalid")
        if row_type == "file":
            if (
                type(row.get("size")) is not int or row["size"] < 0
                or not isinstance(row.get("sha256"), str)
                or HEX64.fullmatch(row["sha256"]) is None
                or not isinstance(row.get("hardlink_group"), str)
                or type(row.get("hardlink_count")) is not int
                or row["hardlink_count"] < 1
                or row.get("source_nlink") != row["hardlink_count"]
            ):
                raise ArchiveError("archive file inventory row is invalid")
            groups.setdefault(row["hardlink_group"], []).append(row)
        elif row_type != "directory":
            raise ArchiveError("archive inventory row type is invalid")
    if paths != sorted(paths, key=os.fsencode) or "." not in paths:
        raise ArchiveError("archive inventory paths are not canonical")
    for group, members in groups.items():
        identities = {(row["source_device"], row["source_inode"]) for row in members}
        if len(identities) != 1 or any(row["hardlink_count"] != len(members) for row in members):
            raise ArchiveError(f"archive hardlink group is inconsistent: {group}")
        if len({(row["size"], row["sha256"], row["mode"]) for row in members}) != 1:
            raise ArchiveError(f"archive hardlink bytes are inconsistent: {group}")
    if manifest.get("inventory_summary") != _inventory_summary(rows):
        raise ArchiveError("archive inventory summary mismatch")
    frontier = manifest.get("checkpoint_frontier")
    if not isinstance(frontier, dict) or set(frontier) != {
        "snapshot_id", "progress_seal_sha256", "event_count",
        "event_chain_head", "campaign_state",
    }:
        raise ArchiveError("archive checkpoint frontier is invalid")
    if (
        not isinstance(frontier.get("snapshot_id"), str)
        or not isinstance(frontier.get("progress_seal_sha256"), str)
        or HEX64.fullmatch(frontier["progress_seal_sha256"]) is None
        or type(frontier.get("event_count")) is not int
        or frontier["event_count"] < 0
        or (
            frontier.get("event_chain_head") is not None
            and (
                not isinstance(frontier["event_chain_head"], str)
                or HEX64.fullmatch(frontier["event_chain_head"]) is None
            )
        )
        or (
            frontier.get("campaign_state") is not None
            and not isinstance(frontier["campaign_state"], str)
        )
    ):
        raise ArchiveError("archive checkpoint frontier values are invalid")
    evidence_inventory_sha256 = _evidence_inventory_sha256(rows)
    if manifest["evidence_inventory_sha256"] != evidence_inventory_sha256:
        raise ArchiveError("archive evidence-inventory binding mismatch")
    expected_checkpoint_id = _canonical_hash({
        "kind": "supplemental_sol_campaign_checkpoint_v1",
        "source_commit": head,
        "campaign_identity_sha256": manifest["campaign_identity_sha256"],
        "frontier": frontier,
        "evidence_inventory_sha256": evidence_inventory_sha256,
    })
    if manifest["checkpoint_id"] != expected_checkpoint_id:
        raise ArchiveError("archive checkpoint identity mismatch")
    capacity = manifest.get("capacity_observation")
    if not isinstance(capacity, dict) or set(capacity) != {
        "source_available_bytes_at_inventory",
        "destination_available_bytes_at_inventory",
        "required_unique_payload_bytes",
        "reserved_destination_bytes",
    }:
        raise ArchiveError("archive capacity observation is invalid")
    if any(type(value) is not int or value < 0 for value in capacity.values()):
        raise ArchiveError("archive capacity observation values are invalid")
    if (
        capacity["required_unique_payload_bytes"]
        != manifest["inventory_summary"]["physical_bytes"]
        or capacity["reserved_destination_bytes"] != CAPACITY_RESERVE_BYTES
        or capacity["destination_available_bytes_at_inventory"]
        < capacity["required_unique_payload_bytes"] + CAPACITY_RESERVE_BYTES
    ):
        raise ArchiveError("archive capacity observation binding is invalid")


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest, _payload = _load_canonical_object(path, role="archive manifest")
    _validate_manifest(manifest)
    return manifest


def _manifest_matches_spec(
    manifest: dict[str, Any], spec: ArchiveSpec, *, checkpoint_id: str,
) -> None:
    revision = spec.archive_base / ARCHIVE_NAMESPACE / spec.source_head
    expected = {
        "source_root": os.fspath(spec.source_root),
        "archive_base": os.fspath(spec.archive_base),
        "archive_namespace": os.fspath(spec.archive_base / ARCHIVE_NAMESPACE),
        "archive_revision_namespace": os.fspath(revision),
        "archive_root": os.fspath(revision / checkpoint_id),
        "source_commit": spec.source_head,
        "source_h8": spec.source_h8,
        "checkpoint_id": checkpoint_id,
        "campaign_identity_sha256": spec.campaign_identity_sha256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ArchiveError(f"existing archive transaction mismatches {key}")


def _verify_partial_prefix(partial: Path, rows: list[dict[str, Any]]) -> None:
    entries = {entry.name: entry for entry in os.scandir(partial)}
    if set(entries) - {ARCHIVE_MANIFEST, ARCHIVED_CAMPAIGN}:
        raise ArchiveError("partial archive top-level namespace is corrupt")
    if ARCHIVE_MANIFEST not in entries:
        raise ArchiveError("partial archive manifest is missing")
    if not stat.S_ISREG(entries[ARCHIVE_MANIFEST].stat(follow_symlinks=False).st_mode):
        raise ArchiveError("partial archive manifest is not regular")
    campaign = partial / ARCHIVED_CAMPAIGN
    if not _lexists(campaign):
        return
    _require_directory(campaign, "partial archived campaign")
    expected = {row["path"]: row for row in rows}
    observed_files: dict[str, os.stat_result] = {}

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name)):
            relative = (prefix / entry.name).as_posix()
            row = expected.get(relative)
            if row is None:
                raise ArchiveError(f"partial archive contains an unexpected entry: {relative}")
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ArchiveError(f"partial archive contains a symlink: {relative}")
            if row["type"] == "directory":
                if not stat.S_ISDIR(info.st_mode):
                    raise ArchiveError(f"partial archive entry type mismatch: {relative}")
                if stat.S_IMODE(info.st_mode) not in {0o700, row["mode"]}:
                    raise ArchiveError(f"partial archive directory mode mismatch: {relative}")
                visit(Path(entry.path), prefix / entry.name)
            elif row["type"] == "file":
                if not stat.S_ISREG(info.st_mode):
                    raise ArchiveError(f"partial archive entry type mismatch: {relative}")
                size, digest = _hash_file_exact(Path(entry.path), info)
                if (
                    stat.S_IMODE(info.st_mode) != row["mode"]
                    or size != row["size"]
                    or digest != row["sha256"]
                ):
                    raise ArchiveError(f"partial archive file mismatch: {relative}")
                observed_files[relative] = info
            else:
                raise ArchiveError(f"partial archive entry type mismatch: {relative}")

    visit(campaign, PurePosixPath())
    by_group: dict[str, list[str]] = {}
    for relative in observed_files:
        by_group.setdefault(expected[relative]["hardlink_group"], []).append(relative)
    seen_inodes: dict[tuple[int, int], str] = {}
    expected_groups: dict[str, list[str]] = {}
    for row in rows:
        if row["type"] == "file":
            expected_groups.setdefault(row["hardlink_group"], []).append(row["path"])
    for group, present in by_group.items():
        canonical = sorted(expected_groups[group], key=os.fsencode)[0]
        if canonical not in present:
            raise ArchiveError(f"partial hardlink group lacks canonical member: {group}")
        identities = {
            (observed_files[path].st_dev, observed_files[path].st_ino) for path in present
        }
        if len(identities) != 1:
            raise ArchiveError(f"partial hardlink topology mismatch: {group}")
        identity = next(iter(identities))
        if identity in seen_inodes:
            raise ArchiveError(
                f"partial hardlink groups alias: {group} and {seen_inodes[identity]}"
            )
        seen_inodes[identity] = group
        if next(iter(observed_files[path].st_nlink for path in present)) != len(present):
            raise ArchiveError(f"partial hardlink escapes archive root: {group}")


def _copy_regular_file(source: Path, destination: Path, row: dict[str, Any]) -> None:
    source_fd = os.open(
        source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    destination_fd: int | None = None
    try:
        source_info = os.fstat(source_fd)
        expected_identity = (
            row["source_device"], row["source_inode"], row["mode"], row["size"],
            row["source_mtime_ns"], row["source_ctime_ns"], row["source_nlink"],
        )
        observed_identity = (
            source_info.st_dev, source_info.st_ino, stat.S_IMODE(source_info.st_mode),
            source_info.st_size, source_info.st_mtime_ns, source_info.st_ctime_ns,
            source_info.st_nlink,
        )
        if observed_identity != expected_identity:
            raise ArchiveError(f"source changed before copy: {source}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            block = os.read(source_fd, 1 << 20)
            if not block:
                break
            digest.update(block)
            copied += len(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short archive copy write")
                view = view[written:]
        if copied != row["size"] or digest.hexdigest() != row["sha256"]:
            raise ArchiveError(f"source bytes changed during copy: {source}")
        if os.fstat(source_fd).st_ctime_ns != source_info.st_ctime_ns:
            raise ArchiveError(f"source metadata changed during copy: {source}")
        os.fchmod(destination_fd, row["mode"])
        os.fsync(destination_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    _fsync_directory(destination.parent)


def _materialize_partial(spec: ArchiveSpec, partial: Path, rows: list[dict[str, Any]]) -> None:
    campaign = partial / ARCHIVED_CAMPAIGN
    if not _lexists(campaign):
        campaign.mkdir(mode=0o700)
        _fsync_directory(partial)
    expected_directories = [row for row in rows if row["type"] == "directory"]
    for row in sorted(
        (item for item in expected_directories if item["path"] != "."),
        key=lambda item: (len(PurePosixPath(item["path"]).parts), os.fsencode(item["path"])),
    ):
        destination = campaign / PurePosixPath(row["path"])
        if not _lexists(destination):
            destination.mkdir(mode=0o700)
            _fsync_directory(destination.parent)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["type"] == "file":
            groups.setdefault(row["hardlink_group"], []).append(row)
    for group in sorted(groups):
        members = sorted(groups[group], key=lambda item: os.fsencode(item["path"]))
        canonical = members[0]
        canonical_destination = campaign / PurePosixPath(canonical["path"])
        if not _lexists(canonical_destination):
            _copy_regular_file(
                spec.source_root / PurePosixPath(canonical["path"]),
                canonical_destination,
                canonical,
            )
        for row in members[1:]:
            destination = campaign / PurePosixPath(row["path"])
            if _lexists(destination):
                continue
            os.link(canonical_destination, destination, follow_symlinks=False)
            _fsync_directory(destination.parent)
    # Exact directory modes are installed only after every child is complete.
    for row in sorted(
        expected_directories,
        key=lambda item: (-len(PurePosixPath(item["path"]).parts), os.fsencode(item["path"])),
    ):
        destination = campaign if row["path"] == "." else campaign / PurePosixPath(row["path"])
        os.chmod(destination, row["mode"], follow_symlinks=False)
        _fsync_directory(destination)
    _fsync_directory(partial)


def _require_destination_inventory(campaign: Path, expected: list[dict[str, Any]]) -> None:
    observed = _inventory(campaign)
    if _logical_inventory(observed) != _logical_inventory(expected):
        raise ArchiveError("archived campaign byte or hardlink topology mismatch")


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArchiveError("atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise ArchiveError(f"archive destination already exists: {destination}")
    if error == errno.EXDEV:
        raise ArchiveError("partial and final archive paths are unexpectedly cross-device")
    raise ArchiveError(
        f"atomic archive rename failed: [{error}] {os.strerror(error)}"
    )


def _verify_event_chain(campaign_root: Path) -> list[dict[str, Any]]:
    root = campaign_root / "sol_workload_campaign/events"
    if not _lexists(root):
        return []
    _require_directory(root, "campaign event directory")
    entries = sorted(os.scandir(root), key=lambda item: os.fsencode(item.name))
    expected_names = [f"event-{index:06d}.json" for index in range(1, len(entries) + 1)]
    if [entry.name for entry in entries] != expected_names:
        raise ArchiveError("campaign event sequence is not closed")
    events: list[dict[str, Any]] = []
    prior: str | None = None
    for ordinal, entry in enumerate(entries, start=1):
        event, _payload = _load_canonical_object(Path(entry.path), role="campaign event")
        _verify_seal(event, "event_seal_sha256", role="campaign event")
        if (
            event.get("schema_version") != 1
            or event.get("kind") != "sol_workload_campaign_event"
            or event.get("event_ordinal") != ordinal
            or event.get("prior_event_seal_sha256") != prior
        ):
            raise ArchiveError(f"campaign event chain mismatch: {entry.path}")
        events.append(event)
        prior = event["event_seal_sha256"]
    return events


def _verify_progress_snapshots(
    campaign_root: Path, marker: dict[str, Any], events: list[dict[str, Any]],
) -> dict[str, Any]:
    root = campaign_root / "sol_workload_campaign/snapshots"
    _require_directory(root, "campaign progress snapshot directory")
    entries = sorted(os.scandir(root), key=lambda item: os.fsencode(item.name))
    if not entries:
        raise ArchiveError("campaign progress snapshot directory is empty")
    frontiers: list[dict[str, Any]] = []
    for entry in entries:
        if not stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode):
            raise ArchiveError(f"campaign snapshot entry is not a directory: {entry.path}")
        snapshot = Path(entry.path)
        progress, _payload = _load_canonical_object(
            snapshot / "PROGRESS.json", role="campaign progress manifest"
        )
        _verify_seal(progress, "progress_seal_sha256", role="campaign progress")
        event_count = progress.get("event_count")
        if (
            progress.get("schema_version") != 1
            or progress.get("kind") != "sol_workload_campaign_progress"
            or progress.get("snapshot_id") != entry.name
            or progress.get("root_identity") != marker
            or type(event_count) is not int
            or not 0 <= event_count <= len(events)
            or progress.get("event_chain_head")
            != (events[event_count - 1]["event_seal_sha256"] if event_count else None)
        ):
            raise ArchiveError(f"campaign progress binding mismatch: {entry.path}")
        exports = progress.get("exports")
        if not isinstance(exports, dict) or not exports:
            raise ArchiveError(f"campaign progress exports are absent: {entry.path}")
        expected_names = {"PROGRESS.json"}
        for role, reference in exports.items():
            if not isinstance(role, str) or not isinstance(reference, dict):
                raise ArchiveError("campaign progress export reference is invalid")
            if set(reference) != {"filename", "sha256", "row_count"}:
                raise ArchiveError("campaign progress export reference is not closed")
            filename = reference.get("filename")
            if not isinstance(filename, str):
                raise ArchiveError("campaign progress export filename is invalid")
            relative = _safe_relative(filename, role="campaign progress export")
            if len(relative.parts) != 1:
                raise ArchiveError("campaign progress export must be a direct child")
            if filename in expected_names:
                raise ArchiveError("campaign progress export filename is duplicated")
            expected_names.add(filename)
            document, export_payload = _load_canonical_object(
                snapshot / filename, role=f"campaign progress export {role}"
            )
            if (
                not isinstance(reference.get("sha256"), str)
                or _sha256_bytes(export_payload) != reference["sha256"]
                or type(reference.get("row_count")) is not int
                or reference["row_count"] < 0
            ):
                raise ArchiveError(f"campaign progress export hash/count mismatch: {filename}")
            if "export_seal_sha256" in document:
                _verify_seal(document, "export_seal_sha256", role=f"campaign export {role}")
            if "row_count" in document and document["row_count"] != reference["row_count"]:
                raise ArchiveError(f"campaign progress export row count mismatch: {filename}")
        observed_names = {child.name for child in os.scandir(snapshot)}
        if observed_names != expected_names:
            raise ArchiveError(f"campaign snapshot namespace mismatch: {entry.path}")
        if event_count == len(events):
            frontiers.append({
                "snapshot_id": progress["snapshot_id"],
                "progress_seal_sha256": progress["progress_seal_sha256"],
                "event_count": event_count,
                "event_chain_head": progress["event_chain_head"],
                "campaign_state": progress.get("campaign_state"),
            })
    if not frontiers:
        raise ArchiveError("no campaign progress snapshot binds the event frontier")
    if len(frontiers) != 1:
        raise ArchiveError("campaign event frontier has multiple progress snapshots")
    return frontiers[0]


def verify_portable_campaign(
    campaign_root: Path, *, repo_root: Path = REPO_ROOT,
    done_verifier: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Verify relocated low-level seals without consulting the source checkout.

    The chunk verifier is an injectable seam until the campaign controller
    publishes a stable whole-campaign verifier.  This function deliberately
    does not claim event/DONE/export reconciliation; production publication
    additionally requires that controller-level verifier in
    :func:`_default_verifier`.  The default chunk verifier is the runner's
    public DONE-v2 API in its explicitly offline/relocated mode.
    """
    campaign_root = _absolute(campaign_root)
    marker, _payload = _load_root_marker(campaign_root)
    events = _verify_event_chain(campaign_root)
    chunks_root = campaign_root / "sol_workload_chunks"
    if _lexists(chunks_root):
        _require_directory(chunks_root, "supplemental chunk directory")
        if done_verifier is None:
            from tools.local_labels import sol_workload_capture_chunk

            done_verifier = sol_workload_capture_chunk.verify_done_manifest
        for entry in sorted(os.scandir(chunks_root), key=lambda item: os.fsencode(item.name)):
            if not stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode):
                raise ArchiveError(f"supplemental chunk entry is not a directory: {entry.path}")
            final_dir = Path(entry.path)
            done, _done_payload = _load_canonical_object(
                final_dir / "DONE.json", role="supplemental chunk DONE"
            )
            try:
                done_verifier(
                    done,
                    final_dir=final_dir,
                    campaign_root=campaign_root,
                    repo_root=repo_root,
                    require_current_state=False,
                    require_final_location=True,
                    verify_git_provenance=False,
                )
            except Exception as exc:
                raise ArchiveError(f"offline supplemental DONE verification failed: {exc}") from exc
    return _verify_progress_snapshots(campaign_root, marker, events)


def _frontier_binding(campaign_root: Path) -> dict[str, Any]:
    marker, _payload = _load_root_marker(campaign_root)
    return _verify_progress_snapshots(
        campaign_root, marker, _verify_event_chain(campaign_root)
    )


def _verify_complete_campaign(root: Path, *, repo_root: Path) -> None:
    verify_portable_campaign(root, repo_root=repo_root)
    from tools.local_labels import sol_workload_capture_campaign

    reconcile = getattr(
        sol_workload_capture_campaign,
        "verify_campaign_checkpoint_offline",
        None,
    )
    if reconcile is None or not callable(reconcile):
        raise ArchiveError(
            "public offline whole-campaign reconciliation verifier is unavailable"
        )
    try:
        reconcile(root, repo_root=repo_root)
    except Exception as exc:
        raise ArchiveError(
            f"offline whole-campaign reconciliation failed: {exc}"
        ) from exc


def _default_verifier(spec: ArchiveSpec) -> PortableVerifier:
    return lambda root: _verify_complete_campaign(root, repo_root=spec.repo_root)


def _verify_final_root(final: Path, *, verifier: PortableVerifier) -> dict[str, Any]:
    observed = _require_directory(final, "final supplemental archive")
    if stat.S_IMODE(observed.st_mode) != 0o700:
        raise ArchiveError("final supplemental archive is not private mode 0700")
    names = {entry.name for entry in os.scandir(final)}
    if names != {ARCHIVE_MANIFEST, ARCHIVED_CAMPAIGN}:
        raise ArchiveError("final supplemental archive namespace is not closed")
    manifest = _load_manifest(final / ARCHIVE_MANIFEST)
    if Path(manifest["archive_root"]) != final:
        raise ArchiveError("final archive path contradicts manifest")
    if manifest["archive_device"] != final.parent.stat().st_dev:
        raise ArchiveError("final archive device contradicts manifest")
    campaign = final / ARCHIVED_CAMPAIGN
    _require_destination_inventory(campaign, manifest["inventory"])
    marker, marker_payload = _load_root_marker(campaign)
    if (
        _sha256_bytes(marker_payload) != manifest["root_marker_sha256"]
        or marker["campaign_identity_sha256"] != manifest["campaign_identity_sha256"]
        or marker["campaign_identity"].get("source_commit") != manifest["source_commit"]
    ):
        raise ArchiveError("final archived campaign identity mismatch")
    verifier(campaign)
    if _frontier_binding(campaign) != manifest["checkpoint_frontier"]:
        raise ArchiveError(
            "archived campaign frontier contradicts checkpoint manifest"
        )
    return manifest


def check_archive(
    archive_base: Path, source_head: str, checkpoint_id: str, *,
    verifier: PortableVerifier | None = None, repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    archive_base = _absolute(archive_base)
    if HEX40.fullmatch(source_head) is None:
        raise ArchiveError("source_head must be lowercase 40-hex")
    if HEX64.fullmatch(checkpoint_id) is None:
        raise ArchiveError("checkpoint_id must be lowercase 64-hex")
    final = archive_base / ARCHIVE_NAMESPACE / source_head / checkpoint_id
    effective = verifier or (
        lambda root: _verify_complete_campaign(root, repo_root=_absolute(repo_root))
    )
    return _verify_final_root(final, verifier=effective)


def archive_campaign(
    spec: ArchiveSpec, *, verifier: PortableVerifier | None = None,
    gpu_probe: GpuProbe | None = None,
) -> tuple[str, dict[str, Any]]:
    """Publish or resume the exact source-preserving archive transaction."""
    spec = _validate_spec(spec)
    from tools.local_labels import sol_workload_capture_campaign

    def refuse_unresolved_storage_receipt() -> None:
        storage_receipt = (
            sol_workload_capture_campaign.storage_fault_receipt_path(
                spec.source_root
            )
        )
        if _lexists(storage_receipt):
            raise ArchiveError(
                "campaign has an unresolved external storage-fault receipt: "
                f"{storage_receipt}"
            )

    refuse_unresolved_storage_receipt()
    effective_verifier = verifier or _default_verifier(spec)
    effective_gpu_probe = gpu_probe or _probe_gpu_pids
    writer_lock = spec.source_root / SOURCE_LOCK
    chunk_lock = spec.source_root / CHUNK_LOCK
    with _exclusive_existing_lock(writer_lock, "campaign writer lock"):
        with _exclusive_existing_lock(chunk_lock, "chunk writer lock"):
            # The campaign records its external receipt only after unwinding
            # its writer locks.  Recheck after acquiring both locks to close
            # the early-check/acquisition race, before the GPU or archive
            # namespaces can be touched.
            refuse_unresolved_storage_receipt()
            with _exclusive_existing_lock(spec.gpu_lock_path, "GPU campaign lock"):
                _assert_gpu_quiescent(effective_gpu_probe)
                namespace = _ensure_archive_namespace(spec)
                with _archive_lock(namespace):
                    revision = _ensure_revision_namespace(
                        namespace, spec.source_head
                    )
                    marker, marker_sha256 = _verify_campaign_identity(spec)
                    del marker
                    # The controller-level callback must independently
                    # reconcile retained inputs, events, DONEs, and frontier
                    # exports before its frontier participates in identity.
                    effective_verifier(spec.source_root)
                    frontier = _frontier_binding(spec.source_root)
                    current_rows = _inventory(spec.source_root)
                    checkpoint_id, evidence_inventory_sha256 = _checkpoint_id(
                        spec, rows=current_rows, frontier=frontier
                    )
                    final = revision / checkpoint_id
                    partial = revision / f".{checkpoint_id}.partial"
                    if _lexists(final):
                        manifest = _verify_final_root(final, verifier=effective_verifier)
                        _manifest_matches_spec(
                            manifest, spec, checkpoint_id=checkpoint_id
                        )
                        if (
                            manifest["checkpoint_frontier"] != frontier
                            or manifest["evidence_inventory_sha256"]
                            != evidence_inventory_sha256
                        ):
                            raise ArchiveError(
                                "existing checkpoint contradicts current evidence"
                            )
                        return "already_archived", manifest
                    if _lexists(partial):
                        partial_info = _require_directory(
                            partial, "partial supplemental archive"
                        )
                        if stat.S_IMODE(partial_info.st_mode) != 0o700:
                            raise ArchiveError(
                                "partial supplemental archive is not private mode 0700"
                            )
                        manifest = _load_manifest(partial / ARCHIVE_MANIFEST)
                        _manifest_matches_spec(
                            manifest, spec, checkpoint_id=checkpoint_id
                        )
                        if (
                            manifest["root_marker_sha256"] != marker_sha256
                            or manifest["checkpoint_frontier"] != frontier
                            or manifest["evidence_inventory_sha256"]
                            != evidence_inventory_sha256
                        ):
                            raise ArchiveError(
                                "partial archive checkpoint binding mismatch"
                            )
                        rows = manifest["inventory"]
                        if current_rows != rows:
                            raise ArchiveError(
                                "source drifted from partial archive inventory"
                            )
                        _verify_partial_prefix(partial, rows)
                    else:
                        rows = current_rows
                        manifest = _build_manifest(
                            spec, rows=rows, marker_sha256=marker_sha256,
                            checkpoint_id=checkpoint_id,
                            evidence_inventory_sha256=evidence_inventory_sha256,
                            frontier=frontier,
                        )
                        try:
                            partial.mkdir(mode=0o700)
                        except FileExistsError as exc:
                            raise ArchiveError("partial archive creation collided") from exc
                        _fsync_directory(revision)
                        _write_exclusive(
                            partial / ARCHIVE_MANIFEST, _pretty_bytes(manifest)
                        )
                        _fsync_directory(partial)
                    _assert_gpu_quiescent(effective_gpu_probe)
                    _require_copy_capacity(partial, rows)
                    _materialize_partial(spec, partial, rows)
                    _require_destination_inventory(
                        partial / ARCHIVED_CAMPAIGN, rows
                    )
                    if _inventory(spec.source_root) != rows:
                        raise ArchiveError("source drifted during archive copy")
                    effective_verifier(partial / ARCHIVED_CAMPAIGN)
                    _assert_gpu_quiescent(effective_gpu_probe)
                    _fsync_directory(partial)
                    _rename_noreplace(partial, final)
                    _fsync_directory(revision)
                    checked = _verify_final_root(
                        final, verifier=effective_verifier
                    )
                    _manifest_matches_spec(
                        checked, spec, checkpoint_id=checkpoint_id
                    )
                    if _inventory(spec.source_root) != rows:
                        raise ArchiveError(
                            "source drifted during final archive verification"
                        )
                    return "archived", checked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("archive", "check"))
    parser.add_argument("--archive-base", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-h8")
    parser.add_argument("--campaign-identity-sha256")
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "check":
            if args.checkpoint_id is None:
                raise ArchiveError("check mode requires: checkpoint_id")
            if _absolute(args.archive_base) != CANONICAL_ARCHIVE_BASE:
                raise ArchiveError(
                    f"production archive base must be exactly {CANONICAL_ARCHIVE_BASE}"
                )
            manifest = check_archive(
                args.archive_base, args.source_head, args.checkpoint_id,
                repo_root=args.repo_root,
            )
            status = "verified"
        else:
            missing = [
                name for name in (
                    "source_root", "source_h8", "campaign_identity_sha256",
                    "gpu_lock_path",
                )
                if getattr(args, name) is None
            ]
            if missing:
                raise ArchiveError(f"archive mode requires: {', '.join(missing)}")
            status, manifest = archive_campaign(ArchiveSpec(
                source_root=args.source_root,
                archive_base=args.archive_base,
                source_head=args.source_head,
                source_h8=args.source_h8,
                campaign_identity_sha256=args.campaign_identity_sha256,
                gpu_lock_path=args.gpu_lock_path,
                repo_root=args.repo_root,
            ))
    except ArchiveError as exc:
        print(f"archive_sol_workload_capture_campaign: REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": status,
        "archive_root": manifest["archive_root"],
        "source_commit": manifest["source_commit"],
        "checkpoint_id": manifest["checkpoint_id"],
        "campaign_identity_sha256": manifest["campaign_identity_sha256"],
        "inventory_summary": manifest["inventory_summary"],
        "archive_manifest_seal_sha256": manifest["archive_manifest_seal_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
