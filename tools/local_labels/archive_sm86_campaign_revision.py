#!/usr/bin/env python3
"""Archive one stopped SM86 campaign revision without discarding evidence.

The archive is an atomic namespace transition, not a cleanup operation.  Only
the revision-owned staging tree, explicitly named published chunks, and the
live module store move.  Versioned inputs, preflights, audits, and historical
evidence remain in place and are hash-bound by the sealed archive manifest.
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
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


ARCHIVE_DIR = "campaign_revision_attempts"
MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
MANIFEST_RECOVERY_NAME = "ARCHIVE_MANIFEST.recovery"
HEX40 = frozenset("0123456789abcdef")
HEX64 = HEX40
HISTORICAL_CHUNK = "portable_lcd-000-bootstrap"
RETAINED_EVIDENCE_DIRS = ("capture_defect_attempts", "preparation_failures")


class ArchiveError(RuntimeError):
    """A fail-closed archive contract violation."""


@dataclasses.dataclass(frozen=True)
class ArchiveSpec:
    campaign_root: Path
    capture_checkout: Path
    preflight_root: Path
    input_root: Path
    source_head: str
    matrix_sha256: str
    authority_sha256: str
    published_chunks: tuple[str, ...]
    staging_chunks: tuple[str, ...]

    def normalized(self) -> "ArchiveSpec":
        return dataclasses.replace(
            self,
            campaign_root=_absolute(self.campaign_root),
            capture_checkout=_absolute(self.capture_checkout),
            preflight_root=_absolute(self.preflight_root),
            input_root=_absolute(self.input_root),
            published_chunks=tuple(sorted(self.published_chunks)),
            staging_chunks=tuple(sorted(self.staging_chunks)),
        )


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"JSON root is not an object: {path}")
    return value


def _require_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(char not in HEX40 for char in value):
        raise ArchiveError(f"{label} must be lowercase {length}-hex")


def _require_name(value: str, label: str) -> None:
    candidate = Path(value)
    if (
        not value
        or candidate.name != value
        or value in {".", ".."}
        or "\\" in value
        or "/" in value
    ):
        raise ArchiveError(f"unsafe {label}: {value!r}")


def _assert_path_chain_no_symlinks(path: Path, *, must_exist: bool = True) -> None:
    path = _absolute(path)
    pending: list[Path] = []
    current = path
    while True:
        pending.append(current)
        if current.parent == current:
            break
        current = current.parent
    for index, item in enumerate(reversed(pending)):
        try:
            mode = item.lstat().st_mode
        except FileNotFoundError:
            is_leaf = index == len(pending) - 1
            if must_exist or not is_leaf:
                raise ArchiveError(f"required path is absent: {item}")
            continue
        if stat.S_ISLNK(mode):
            raise ArchiveError(f"symlink path component is forbidden: {item}")


def _require_regular(path: Path, label: str) -> None:
    _assert_path_chain_no_symlinks(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ArchiveError(f"{label} is not a regular file: {path}")


def _inventory(path: Path) -> list[dict]:
    """Return a byte inventory while rejecting symlinks and special files."""
    _assert_path_chain_no_symlinks(path)
    root_mode = path.lstat().st_mode
    if stat.S_ISREG(root_mode):
        return [{
            "path": ".",
            "type": "file",
            "mode": stat.S_IMODE(root_mode),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }]
    if not stat.S_ISDIR(root_mode):
        raise ArchiveError(f"special file is forbidden: {path}")

    rows: list[dict] = [{
        "path": ".", "type": "directory", "mode": stat.S_IMODE(root_mode),
    }]

    def visit(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(
                os.scandir(directory), key=lambda entry: os.fsencode(entry.name)
            )
        except OSError as exc:
            raise ArchiveError(f"cannot inventory {directory}: {exc}") from exc
        for entry in entries:
            child_relative = relative / entry.name
            child_path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArchiveError(f"cannot stat {child_path}: {exc}") from exc
            row = {
                "path": child_relative.as_posix(),
                "mode": stat.S_IMODE(info.st_mode),
            }
            if stat.S_ISLNK(info.st_mode):
                raise ArchiveError(f"symlink is forbidden: {child_path}")
            if stat.S_ISDIR(info.st_mode):
                row["type"] = "directory"
                rows.append(row)
                visit(child_path, child_relative)
            elif stat.S_ISREG(info.st_mode):
                row.update({
                    "type": "file",
                    "size": info.st_size,
                    "sha256": _sha256_file(child_path),
                })
                rows.append(row)
            else:
                raise ArchiveError(f"special file is forbidden: {child_path}")

    visit(path, Path())
    return rows


def _require_inventory(path: Path, expected: list[dict], label: str) -> None:
    observed = _inventory(path)
    if observed != expected:
        raise ArchiveError(f"{label} byte inventory drifted")


def _sha256_prefix(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ArchiveError(f"retained append-only file was truncated: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _require_retained_append_only(
    path: Path, expected: list[dict], component_id: str,
) -> None:
    """Verify all historical bytes while allowing new append-only evidence."""
    ledger_names = {
        "capture_defect_attempts": "capture_defect_ledger.jsonl",
        "preparation_failures": "preparation_failure_ledger.jsonl",
    }
    append_ledger = ledger_names.get(component_id)
    _assert_path_chain_no_symlinks(path)
    for row in expected:
        relative = row["path"]
        candidate = path if relative == "." else path / PurePosixPath(relative)
        _assert_path_chain_no_symlinks(candidate)
        info = candidate.lstat()
        if stat.S_IMODE(info.st_mode) != row["mode"]:
            raise ArchiveError(f"{component_id} retained mode drifted: {relative}")
        if row["type"] == "directory":
            if not stat.S_ISDIR(info.st_mode):
                raise ArchiveError(f"{component_id} retained directory drifted: {relative}")
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ArchiveError(f"{component_id} retained file drifted: {relative}")
        if relative == append_ledger:
            if info.st_size < row["size"]:
                raise ArchiveError(f"{component_id} retained ledger was truncated")
            if _sha256_prefix(candidate, row["size"]) != row["sha256"]:
                raise ArchiveError(f"{component_id} retained ledger prefix drifted")
        elif info.st_size != row["size"] or _sha256_file(candidate) != row["sha256"]:
            raise ArchiveError(f"{component_id} retained historical byte drifted: {relative}")


def _component(
    component_id: str,
    disposition: str,
    source_path: Path,
    *,
    archive_path: str | None = None,
) -> dict:
    return {
        "component_id": component_id,
        "disposition": disposition,
        "source_path": os.fspath(source_path),
        "archive_path": archive_path,
        "inventory": _inventory(source_path),
    }


def _seal_manifest(manifest: dict) -> dict:
    if "manifest_seal_sha256" in manifest:
        raise ArchiveError("manifest seal field already present")
    sealed = dict(manifest)
    sealed["manifest_seal_sha256"] = _sha256_bytes(_canonical_bytes(manifest))
    return sealed


def _verify_manifest_seal(manifest: dict) -> None:
    seal = manifest.get("manifest_seal_sha256")
    if not isinstance(seal, str):
        raise ArchiveError("archive manifest seal is missing")
    body = dict(manifest)
    del body["manifest_seal_sha256"]
    observed = _sha256_bytes(_canonical_bytes(body))
    if seal != observed:
        raise ArchiveError("archive manifest seal mismatch")


def _safe_archive_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArchiveError("archive component path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveError("archive component path is unsafe")
    return value


def _validate_manifest_structure(manifest: dict) -> None:
    required = {
        "schema_version", "manifest_kind", "transaction_protocol",
        "campaign_root", "archive_root", "capture_checkout", "input_root",
        "preflight_root", "authoritative", "reusable", "campaign_rows",
        "source_head", "capture_matrix_sha256", "roster_authority_sha256",
        "published_chunks", "staging_chunks", "components",
        "inventory_totals", "manifest_seal_sha256",
    }
    if set(manifest) != required:
        raise ArchiveError("archive manifest top-level keys are not closed")
    _require_hex(str(manifest.get("source_head")), 40, "manifest source_head")
    _require_hex(
        str(manifest.get("capture_matrix_sha256")), 64,
        "manifest capture_matrix_sha256",
    )
    _require_hex(
        str(manifest.get("roster_authority_sha256")), 64,
        "manifest roster_authority_sha256",
    )
    for name in (
        "campaign_root", "archive_root", "capture_checkout", "input_root",
        "preflight_root",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ArchiveError(f"manifest {name} is not an absolute path")
    campaign_root = Path(manifest["campaign_root"])
    source_head = manifest["source_head"]
    token = f"r28_{source_head[:8]}"
    root_formulas = {
        "archive_root": campaign_root / ARCHIVE_DIR / source_head,
        "input_root": campaign_root / "inputs" / token,
        "preflight_root": (
            Path(manifest["capture_checkout"])
            / "results" / "nvbit_capture_staging" / token
        ),
    }
    for name, expected in root_formulas.items():
        if Path(manifest[name]) != expected:
            raise ArchiveError(f"manifest {name} formula binding is invalid")
    if (
        manifest.get("authoritative") is not False
        or manifest.get("reusable") is not False
        or manifest.get("campaign_rows") != 0
        or isinstance(manifest.get("campaign_rows"), bool)
    ):
        raise ArchiveError("archive disposition must be non-authoritative/non-reusable/zero-row")
    published = manifest.get("published_chunks")
    staging = manifest.get("staging_chunks")
    if (
        not isinstance(published, list) or not published
        or not isinstance(staging, list) or not staging
        or published != sorted(set(published))
        or staging != sorted(set(staging))
    ):
        raise ArchiveError("manifest chunk lists are not sorted unique lists")
    for chunk in (*published, *staging):
        if not isinstance(chunk, str):
            raise ArchiveError("manifest chunk id is not a string")
        _require_name(chunk, "manifest chunk id")
    if not set(published).issubset(staging) or HISTORICAL_CHUNK in published:
        raise ArchiveError("manifest chunk relationship is invalid")

    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ArchiveError("manifest components is not a nonempty list")
    ids: set[str] = set()
    sources: set[str] = set()
    destinations: set[str] = set()
    file_count = 0
    byte_count = 0
    moved_count = 0
    retained_count = 0
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "component_id", "disposition", "source_path", "archive_path", "inventory",
        }:
            raise ArchiveError("manifest component shape is invalid")
        component_id = component["component_id"]
        source = component["source_path"]
        if not isinstance(component_id, str) or not component_id or component_id in ids:
            raise ArchiveError("manifest component ids are invalid or duplicated")
        if not isinstance(source, str) or not Path(source).is_absolute() or source in sources:
            raise ArchiveError("manifest component sources are invalid or duplicated")
        ids.add(component_id)
        sources.add(source)
        disposition = component["disposition"]
        if disposition == "moved":
            moved_count += 1
            destination = _safe_archive_path(component["archive_path"])
            if destination in destinations:
                raise ArchiveError("manifest archive component paths are duplicated")
            destinations.add(destination)
        elif disposition == "retained":
            retained_count += 1
            if component["archive_path"] is not None:
                raise ArchiveError("retained component has an archive path")
        else:
            raise ArchiveError("manifest component disposition is invalid")
        inventory = component["inventory"]
        if not isinstance(inventory, list) or not inventory:
            raise ArchiveError("manifest component inventory is empty")
        inventory_paths: set[str] = set()
        for row in inventory:
            if not isinstance(row, dict):
                raise ArchiveError("manifest inventory row is not an object")
            row_type = row.get("type")
            row_path = row.get("path")
            mode = row.get("mode")
            if (
                not isinstance(row_path, str) or row_path in inventory_paths
                or not isinstance(mode, int) or isinstance(mode, bool)
                or not 0 <= mode <= 0o7777
            ):
                raise ArchiveError("manifest inventory row identity is invalid")
            if row_path != ".":
                _safe_archive_path(row_path)
            inventory_paths.add(row_path)
            if row_type == "directory":
                if set(row) != {"path", "type", "mode"}:
                    raise ArchiveError("manifest directory inventory row is not closed")
            elif row_type == "file":
                if set(row) != {"path", "type", "mode", "size", "sha256"}:
                    raise ArchiveError("manifest file inventory row is not closed")
                size = row["size"]
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise ArchiveError("manifest file size is invalid")
                _require_hex(str(row["sha256"]), 64, "manifest file sha256")
                file_count += 1
                byte_count += size
            else:
                raise ArchiveError("manifest inventory row type is invalid")
        if "." not in inventory_paths:
            raise ArchiveError("manifest component inventory lacks its root")

    expected_moved = {"module_store", "staging"} | {
        f"published_chunk:{chunk}" for chunk in published
    }
    actual_moved = {
        component["component_id"] for component in components
        if component["disposition"] == "moved"
    }
    if actual_moved != expected_moved:
        raise ArchiveError("manifest moved component set is invalid")
    required_retained = {
        "input_package", "preflights",
        f"historical_chunk:{HISTORICAL_CHUNK}",
        *RETAINED_EVIDENCE_DIRS,
    }
    actual_retained = {
        component["component_id"] for component in components
        if component["disposition"] == "retained"
    }
    if not required_retained.issubset(actual_retained):
        raise ArchiveError("manifest retained component set is incomplete")
    audit_ids = actual_retained - required_retained
    if not audit_ids or any(not value.startswith("audit:") for value in audit_ids):
        raise ArchiveError("manifest retained audit component set is invalid")
    expected_bindings: dict[str, tuple[str, str | None, str]] = {
        "module_store": (
            os.fspath(campaign_root / "module_store"), "module_store", "moved",
        ),
        "staging": (
            os.fspath(campaign_root / ".staging" / source_head), "staging", "moved",
        ),
        "input_package": (manifest["input_root"], None, "retained"),
        "preflights": (manifest["preflight_root"], None, "retained"),
        f"historical_chunk:{HISTORICAL_CHUNK}": (
            os.fspath(campaign_root / "chunks" / HISTORICAL_CHUNK), None, "retained",
        ),
        "capture_defect_attempts": (
            os.fspath(campaign_root / "capture_defect_attempts"), None, "retained",
        ),
        "preparation_failures": (
            os.fspath(campaign_root / "preparation_failures"), None, "retained",
        ),
    }
    for chunk in published:
        expected_bindings[f"published_chunk:{chunk}"] = (
            os.fspath(campaign_root / "chunks" / chunk), f"chunks/{chunk}", "moved",
        )
    for audit_id in audit_ids:
        name = audit_id.removeprefix("audit:")
        _require_name(name, "manifest audit name")
        if token not in name:
            raise ArchiveError("manifest audit does not bind the source revision")
        expected_bindings[audit_id] = (
            os.fspath(campaign_root / "audits" / name), None, "retained",
        )
    observed_bindings = {
        component["component_id"]: (
            component["source_path"], component["archive_path"],
            component["disposition"],
        )
        for component in components
    }
    if observed_bindings != expected_bindings:
        raise ArchiveError("manifest component path bindings are invalid")
    expected_totals = {
        "components": len(components),
        "files": file_count,
        "bytes": byte_count,
        "moved_components": moved_count,
        "retained_components": retained_count,
    }
    if manifest.get("inventory_totals") != expected_totals:
        raise ArchiveError("manifest inventory totals mismatch")


def _validate_spec(spec: ArchiveSpec) -> ArchiveSpec:
    spec = spec.normalized()
    _require_hex(spec.source_head, 40, "source_head")
    _require_hex(spec.matrix_sha256, 64, "matrix_sha256")
    _require_hex(spec.authority_sha256, 64, "authority_sha256")
    if not spec.published_chunks:
        raise ArchiveError("published_chunks must not be empty")
    if not spec.staging_chunks:
        raise ArchiveError("staging_chunks must not be empty")
    if len(set(spec.published_chunks)) != len(spec.published_chunks):
        raise ArchiveError("published_chunks contains duplicates")
    if len(set(spec.staging_chunks)) != len(spec.staging_chunks):
        raise ArchiveError("staging_chunks contains duplicates")
    for chunk in (*spec.published_chunks, *spec.staging_chunks):
        _require_name(chunk, "chunk id")
    if not set(spec.published_chunks).issubset(spec.staging_chunks):
        raise ArchiveError("every published chunk must have a staging tree")
    if HISTORICAL_CHUNK in spec.published_chunks:
        raise ArchiveError(f"historical {HISTORICAL_CHUNK} must be retained")
    token = f"r28_{spec.source_head[:8]}"
    expected_input = spec.campaign_root / "inputs" / token
    if spec.input_root != expected_input:
        raise ArchiveError(f"input_root must be exactly {expected_input}")
    if spec.preflight_root.name != token:
        raise ArchiveError(f"preflight_root basename must be {token}")
    expected_preflight = (
        spec.capture_checkout / "results" / "nvbit_capture_staging" / token
    )
    if spec.preflight_root != expected_preflight:
        raise ArchiveError(f"preflight_root must be exactly {expected_preflight}")
    return spec


def _git_output(checkout: Path, *args: str) -> str:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(checkout), *args),
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise ArchiveError(f"capture checkout Git probe failed: {detail!r}") from exc
    return completed.stdout


def _validate_capture_checkout(spec: ArchiveSpec) -> None:
    _assert_path_chain_no_symlinks(spec.capture_checkout)
    if not spec.capture_checkout.is_dir():
        raise ArchiveError("capture_checkout is not a directory")
    top = _absolute(Path(_git_output(spec.capture_checkout, "rev-parse", "--show-toplevel").strip()))
    if top != spec.capture_checkout:
        raise ArchiveError("capture_checkout is not the Git worktree root")
    head = _git_output(spec.capture_checkout, "rev-parse", "HEAD").strip()
    if head != spec.source_head:
        raise ArchiveError(
            f"capture checkout HEAD mismatch: expected={spec.source_head} observed={head}"
        )
    dirty = _git_output(
        spec.capture_checkout, "status", "--porcelain=v1", "--untracked-files=all",
    )
    if dirty:
        raise ArchiveError(f"capture checkout is dirty: {dirty.splitlines()!r}")


def _ancestor_pids(pid: int) -> set[int]:
    result: set[int] = set()
    current = pid
    while current > 0 and current not in result:
        result.add(current)
        try:
            lines = Path(f"/proc/{current}/status").read_text().splitlines()
        except (OSError, UnicodeError):
            break
        row = next((line for line in lines if line.startswith("PPid:")), None)
        if row is None:
            break
        try:
            current = int(row.split()[1])
        except (IndexError, ValueError):
            break
    return result


def _process_rows() -> Iterator[tuple[int, tuple[str, ...]]]:
    proc = Path("/proc")
    for child in sorted(proc.iterdir(), key=lambda path: os.fsencode(path.name)):
        if not child.name.isdigit():
            continue
        try:
            raw = (child / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = tuple(
            item.decode("utf-8", "surrogateescape")
            for item in raw.rstrip(b"\0").split(b"\0")
        )
        yield int(child.name), argv


def _assert_process_quiescence(spec: ArchiveSpec) -> None:
    ignored = _ancestor_pids(os.getpid())
    path_markers = (os.fspath(spec.campaign_root), os.fspath(spec.capture_checkout))
    script_markers = {
        "sm86_capture_chunk.py", "campaign_preflight.py", "sol_reconstitution.py",
        "sol_shrink_apply.py", "eval_driver.py", "run_dataset.py",
    }
    substring_markers = ("sol_execbench", "sol-execbench")
    conflicts: list[dict] = []
    for pid, argv in _process_rows():
        if pid in ignored:
            continue
        path_hit = any(marker in arg for marker in path_markers for arg in argv)
        script_hit = any(Path(arg).name in script_markers for arg in argv)
        ncu_hit = any(Path(arg).name == "ncu" for arg in argv)
        substring_hit = any(marker in arg for marker in substring_markers for arg in argv)
        if path_hit or script_hit or ncu_hit or substring_hit:
            conflicts.append({"pid": pid, "argv": list(argv)})
    if conflicts:
        raise ArchiveError(
            "capture processes are not quiescent: "
            + json.dumps(conflicts, sort_keys=True)
        )
    try:
        probe = subprocess.run(
            (
                "nvidia-smi", "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ),
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise ArchiveError(f"GPU process quiescence probe failed: {detail!r}") from exc
    gpu_pids: list[int] = []
    for line in probe.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            gpu_pids.append(int(value))
        except ValueError as exc:
            raise ArchiveError(f"GPU process probe returned invalid PID: {value!r}") from exc
    if gpu_pids:
        raise ArchiveError(f"GPU compute processes are active: {sorted(gpu_pids)!r}")


def _validate_runtime(spec: ArchiveSpec) -> None:
    _validate_capture_checkout(spec)
    _assert_process_quiescence(spec)


def _validate_preflights(spec: ArchiveSpec) -> dict[str, str]:
    _assert_path_chain_no_symlinks(spec.preflight_root)
    if not spec.preflight_root.is_dir():
        raise ArchiveError("preflight_root is not a directory")
    hashes: dict[str, str] = {}
    entries = sorted(os.scandir(spec.preflight_root), key=lambda entry: entry.name)
    if not entries:
        raise ArchiveError("preflight_root is empty")
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        path = Path(entry.path)
        if not stat.S_ISREG(info.st_mode) or not entry.name.endswith(".json"):
            raise ArchiveError(f"preflight root must contain only JSON files: {path}")
        document = _load_json(path)
        if document.get("git_head") != spec.source_head:
            raise ArchiveError(f"preflight git_head mismatch: {path}")
        if document.get("capture_matrix_sha256") != spec.matrix_sha256:
            raise ArchiveError(f"preflight matrix mismatch: {path}")
        hashes[os.fspath(path)] = _sha256_file(path)
    return hashes


def _validate_inputs(spec: ArchiveSpec) -> tuple[Path, Path, dict]:
    _assert_path_chain_no_symlinks(spec.input_root)
    if not spec.input_root.is_dir():
        raise ArchiveError("input_root is not a directory")
    matrix_path = spec.input_root / "sm86_reference_capture_matrix.json"
    authority_path = spec.input_root / "capture_roster_authority_sm86.json"
    _require_regular(matrix_path, "input matrix")
    _require_regular(authority_path, "input authority")
    if _sha256_file(matrix_path) != spec.matrix_sha256:
        raise ArchiveError("input matrix SHA-256 mismatch")
    if _sha256_file(authority_path) != spec.authority_sha256:
        raise ArchiveError("input authority SHA-256 mismatch")
    matrix = _load_json(matrix_path)
    authority = _load_json(authority_path)
    if matrix.get("source_commit") != spec.source_head:
        raise ArchiveError("input matrix source_commit mismatch")
    if authority.get("source_commit") != spec.source_head:
        raise ArchiveError("input authority source_commit mismatch")
    matrix_authority = matrix.get("authority")
    if not isinstance(matrix_authority, dict):
        raise ArchiveError("input matrix authority binding is missing")
    if matrix_authority.get("sha256") != spec.authority_sha256:
        raise ArchiveError("input matrix authority SHA-256 binding mismatch")
    if _absolute(Path(str(matrix_authority.get("path")))) != authority_path:
        raise ArchiveError("input matrix authority path binding mismatch")
    chunks = matrix.get("chunks")
    if not isinstance(chunks, list):
        raise ArchiveError("input matrix chunks is not a list")
    matrix_chunks = {
        row.get("chunk_id") for row in chunks if isinstance(row, dict)
    }
    required = set(spec.staging_chunks) | {HISTORICAL_CHUNK}
    if not required.issubset(matrix_chunks):
        raise ArchiveError("staging/historical chunk is absent from input matrix")
    return matrix_path, authority_path, matrix


def _verify_done_semantics(manifest: dict, capture_checkout: Path) -> list[str]:
    try:
        from tools.local_labels import campaign_done_manifest
    except ModuleNotFoundError:  # direct execution from tools/local_labels
        import campaign_done_manifest  # type: ignore[no-redef]
    return campaign_done_manifest.verify_seal(
        manifest,
        repo_root=capture_checkout,
        rehash_references=True,
        require_current_head=False,
        require_clean_tree=False,
    )


def _verify_retained_ledgers(campaign_root: Path) -> None:
    try:
        from tools.local_labels import (
            archive_capture_defect_attempt,
            archive_preparation_failure,
        )
    except ModuleNotFoundError:  # direct execution from tools/local_labels
        import archive_capture_defect_attempt  # type: ignore[no-redef]
        import archive_preparation_failure  # type: ignore[no-redef]
    try:
        archive_capture_defect_attempt.load_capture_defect_ledger(
            campaign_root / "capture_defect_attempts" / "capture_defect_ledger.jsonl",
            verify_archives=True,
        )
        archive_preparation_failure.load_preparation_failure_ledger(
            campaign_root / "preparation_failures" / "preparation_failure_ledger.jsonl",
            verify_archives=True,
        )
    except Exception as exc:
        refusal_types = (
            archive_capture_defect_attempt.CaptureDefectRefusal,
            archive_preparation_failure.PreparationArchiveRefusal,
        )
        if isinstance(exc, refusal_types):
            raise ArchiveError(f"retained evidence ledger is invalid: {exc}") from exc
        raise


def _validate_published(
    spec: ArchiveSpec,
    matrix_path: Path,
    authority_path: Path,
    preflight_hashes: dict[str, str],
) -> None:
    chunks_root = spec.campaign_root / "chunks"
    _assert_path_chain_no_symlinks(chunks_root)
    observed = {
        entry.name for entry in os.scandir(chunks_root)
        if entry.is_dir(follow_symlinks=False)
    }
    expected = set(spec.published_chunks) | {HISTORICAL_CHUNK}
    if observed != expected:
        raise ArchiveError(
            f"published chunk namespace mismatch: expected={sorted(expected)!r} "
            f"observed={sorted(observed)!r}"
        )
    for entry in os.scandir(chunks_root):
        if entry.name not in observed or not entry.is_dir(follow_symlinks=False):
            raise ArchiveError(f"unexpected chunks entry: {entry.path}")
    historical_done_path = chunks_root / HISTORICAL_CHUNK / "DONE.json"
    _require_regular(historical_done_path, "historical DONE")
    historical_problems = _verify_done_semantics(
        _load_json(historical_done_path), spec.capture_checkout,
    )
    if historical_problems:
        raise ArchiveError(
            f"{historical_done_path} semantic DONE verification failed: "
            + "; ".join(historical_problems)
        )
    for chunk_id in spec.published_chunks:
        done_path = chunks_root / chunk_id / "DONE.json"
        _require_regular(done_path, "published DONE")
        done = _load_json(done_path)
        exact = {
            "chunk_id": chunk_id,
            "source_head_at_seal": spec.source_head,
            "source_commit_pre": spec.source_head,
            "source_commit_post": spec.source_head,
            "source_commit_stable": True,
            "capture_matrix_sha256": spec.matrix_sha256,
            "roster_authority_sha256": spec.authority_sha256,
        }
        for key, value in exact.items():
            if done.get(key) != value:
                raise ArchiveError(f"{done_path} has wrong {key}")
        inputs = done.get("input_paths")
        if not isinstance(inputs, dict):
            raise ArchiveError(f"{done_path} input_paths is missing")
        if _absolute(Path(str(inputs.get("capture_matrix")))) != matrix_path:
            raise ArchiveError(f"{done_path} matrix path binding mismatch")
        if _absolute(Path(str(inputs.get("roster_authority")))) != authority_path:
            raise ArchiveError(f"{done_path} authority path binding mismatch")
        preflight = _absolute(Path(str(inputs.get("preflight"))))
        preflight_text = os.fspath(preflight)
        if preflight.parent != spec.preflight_root:
            raise ArchiveError(f"{done_path} preflight path binding mismatch")
        if preflight_hashes.get(preflight_text) != done.get("preflight_sha256"):
            raise ArchiveError(f"{done_path} preflight SHA-256 binding mismatch")
        seal_problems = _verify_done_semantics(done, spec.capture_checkout)
        if seal_problems:
            raise ArchiveError(
                f"{done_path} semantic DONE verification failed: "
                + "; ".join(seal_problems)
            )


def _validate_staging(spec: ArchiveSpec) -> Path:
    staging = spec.campaign_root / ".staging" / spec.source_head
    _assert_path_chain_no_symlinks(staging)
    if not staging.is_dir():
        raise ArchiveError("revision staging root is not a directory")
    entries = sorted(os.scandir(staging), key=lambda entry: entry.name)
    observed = {entry.name for entry in entries}
    if observed != set(spec.staging_chunks):
        raise ArchiveError("staging chunk namespace mismatch")
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            raise ArchiveError(f"staging chunk is not a directory: {entry.path}")
        chunk_root = Path(entry.path)
        children = sorted(os.scandir(chunk_root), key=lambda child: child.name)
        if len(children) != 1 or children[0].name != spec.matrix_sha256:
            raise ArchiveError(f"staging matrix namespace mismatch: {chunk_root}")
        if not children[0].is_dir(follow_symlinks=False):
            raise ArchiveError(f"staging matrix root is not a directory: {children[0].path}")
        journal = Path(children[0].path) / "chunk_run_journal.jsonl"
        _require_regular(journal, "staging run journal")
        try:
            first_line = journal.open("r", encoding="utf-8").readline()
            first = json.loads(first_line, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArchiveError(f"invalid staging run journal {journal}: {exc}") from exc
        expected = {
            "chunk_id": entry.name,
            "source_commit": spec.source_head,
            "capture_matrix_sha256": spec.matrix_sha256,
        }
        if not isinstance(first, dict) or any(first.get(k) != v for k, v in expected.items()):
            raise ArchiveError(f"staging run journal binding mismatch: {journal}")
    return staging


def _matching_audits(spec: ArchiveSpec) -> list[Path]:
    root = spec.campaign_root / "audits"
    _assert_path_chain_no_symlinks(root)
    token = f"r28_{spec.source_head[:8]}"
    matches: list[Path] = []
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if token not in entry.name:
            continue
        info = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ArchiveError(f"matching audit is not a regular file: {entry.path}")
        matches.append(Path(entry.path))
    if not matches:
        raise ArchiveError(f"no retained audits match {token}")
    return matches


def build_archive_manifest(spec: ArchiveSpec) -> dict:
    """Validate all operands and construct the deterministic sealed manifest."""
    spec = _validate_spec(spec)
    _validate_runtime(spec)
    _assert_path_chain_no_symlinks(spec.campaign_root)
    if not spec.campaign_root.is_dir():
        raise ArchiveError("campaign_root is not a directory")
    matrix_path, authority_path, _matrix = _validate_inputs(spec)
    preflight_hashes = _validate_preflights(spec)
    _validate_published(spec, matrix_path, authority_path, preflight_hashes)
    staging = _validate_staging(spec)
    module_store = spec.campaign_root / "module_store"
    _assert_path_chain_no_symlinks(module_store)
    if not module_store.is_dir():
        raise ArchiveError("module_store is not a directory")
    _verify_retained_ledgers(spec.campaign_root)

    moved = [
        _component("module_store", "moved", module_store, archive_path="module_store"),
        _component("staging", "moved", staging, archive_path="staging"),
    ]
    moved.extend(
        _component(
            f"published_chunk:{chunk}", "moved",
            spec.campaign_root / "chunks" / chunk,
            archive_path=f"chunks/{chunk}",
        )
        for chunk in spec.published_chunks
    )

    retained = [
        _component("input_package", "retained", spec.input_root),
        _component("preflights", "retained", spec.preflight_root),
        _component(
            f"historical_chunk:{HISTORICAL_CHUNK}", "retained",
            spec.campaign_root / "chunks" / HISTORICAL_CHUNK,
        ),
    ]
    for name in RETAINED_EVIDENCE_DIRS:
        retained.append(_component(name, "retained", spec.campaign_root / name))
    retained.extend(
        _component(f"audit:{path.name}", "retained", path)
        for path in _matching_audits(spec)
    )
    components = sorted(moved + retained, key=lambda row: row["component_id"])
    destination = spec.campaign_root / ARCHIVE_DIR / spec.source_head
    manifest = {
        "schema_version": 1,
        "manifest_kind": "sm86_campaign_revision_archive",
        "transaction_protocol": "same-filesystem-atomic-rename-v1",
        "campaign_root": os.fspath(spec.campaign_root),
        "archive_root": os.fspath(destination),
        "capture_checkout": os.fspath(spec.capture_checkout),
        "input_root": os.fspath(spec.input_root),
        "preflight_root": os.fspath(spec.preflight_root),
        "authoritative": False,
        "reusable": False,
        "campaign_rows": 0,
        "source_head": spec.source_head,
        "capture_matrix_sha256": spec.matrix_sha256,
        "roster_authority_sha256": spec.authority_sha256,
        "published_chunks": list(spec.published_chunks),
        "staging_chunks": list(spec.staging_chunks),
        "components": components,
        "inventory_totals": {
            "components": len(components),
            "files": sum(
                1 for component in components for row in component["inventory"]
                if row["type"] == "file"
            ),
            "bytes": sum(
                row.get("size", 0)
                for component in components for row in component["inventory"]
            ),
            "moved_components": len(moved),
            "retained_components": len(retained),
        },
    }
    campaign_device = spec.campaign_root.stat().st_dev
    for component in moved:
        source = Path(component["source_path"])
        if source.stat().st_dev != campaign_device:
            raise ArchiveError(f"cross-device archive move is forbidden: {source}")
    return _seal_manifest(manifest)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _replace_torn_bootstrap_manifest(bootstrap: Path, manifest: dict) -> None:
    manifest_path = bootstrap / MANIFEST_NAME
    recovery = bootstrap / MANIFEST_RECOVERY_NAME
    if _lexists(manifest_path):
        _require_regular(manifest_path, "torn bootstrap manifest")
    if _lexists(recovery):
        _require_regular(recovery, "bootstrap recovery manifest")
        recovery.unlink()
        _fsync_dir(bootstrap)
    _write_exclusive(recovery, _canonical_bytes(manifest))
    _fsync_dir(bootstrap)
    os.replace(recovery, manifest_path)
    _fsync_dir(bootstrap)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without permitting destination replacement (Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArchiveError("atomic RENAME_NOREPLACE is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise ArchiveError(f"atomic archive destination collision: {destination}")
    if error == errno.EXDEV:
        raise ArchiveError(f"cross-device archive move is forbidden: {source}")
    raise ArchiveError(
        f"atomic archive rename failed {source} -> {destination}: "
        f"[{error}] {os.strerror(error)}"
    )


def _mkdir_component_parent(path: Path, archive_root: Path) -> None:
    if _lexists(path):
        _assert_path_chain_no_symlinks(path)
        if not path.is_dir():
            raise ArchiveError(f"archive component parent is not a directory: {path}")
        return
    if path.parent != archive_root:
        _mkdir_component_parent(path.parent, archive_root)
    path.mkdir(mode=0o755)
    _fsync_dir(path)
    _fsync_dir(path.parent)


@contextlib.contextmanager
def _campaign_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".capture.lock"
    _require_regular(lock_path, "campaign writer lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError("campaign writer lock is active") from exc
        yield
    finally:
        os.close(descriptor)


def _load_manifest(path: Path) -> dict:
    _assert_path_chain_no_symlinks(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ArchiveError("archive manifest is not a regular file")
    manifest = _load_json(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArchiveError(f"cannot read archive manifest: {exc}") from exc
    if raw != _canonical_bytes(manifest):
        raise ArchiveError("archive manifest bytes are not canonical")
    _verify_manifest_seal(manifest)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ArchiveError("unsupported archive manifest schema")
    if manifest.get("manifest_kind") != "sm86_campaign_revision_archive":
        raise ArchiveError("wrong archive manifest kind")
    if manifest.get("transaction_protocol") != "same-filesystem-atomic-rename-v1":
        raise ArchiveError("wrong archive transaction protocol")
    _validate_manifest_structure(manifest)
    return manifest


def _manifest_matches_spec(manifest: dict, spec: ArchiveSpec) -> None:
    exact = {
        "campaign_root": os.fspath(spec.campaign_root),
        "archive_root": os.fspath(spec.campaign_root / ARCHIVE_DIR / spec.source_head),
        "capture_checkout": os.fspath(spec.capture_checkout),
        "input_root": os.fspath(spec.input_root),
        "preflight_root": os.fspath(spec.preflight_root),
        "source_head": spec.source_head,
        "capture_matrix_sha256": spec.matrix_sha256,
        "roster_authority_sha256": spec.authority_sha256,
        "published_chunks": list(spec.published_chunks),
        "staging_chunks": list(spec.staging_chunks),
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise ArchiveError(f"existing transaction has mismatched {key}")


def _verify_components(
    manifest: dict,
    archive_root: Path,
    *,
    partial: bool,
    transition: bool = False,
) -> None:
    expected_top = {MANIFEST_NAME, "chunks", "module_store", "staging"}
    if _lexists(archive_root):
        _assert_path_chain_no_symlinks(archive_root)
        entries = list(os.scandir(archive_root))
        observed_top = {entry.name for entry in entries}
        namespace_ok = (
            MANIFEST_NAME in observed_top
            and (
                observed_top.issubset(expected_top)
                if partial else observed_top == expected_top
            )
        )
        if not namespace_ok:
            raise ArchiveError(
                f"archive top-level namespace mismatch: {sorted(observed_top)!r}"
            )
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if entry.name == MANIFEST_NAME:
                if not stat.S_ISREG(info.st_mode):
                    raise ArchiveError("archive manifest is not a regular file")
            elif not stat.S_ISDIR(info.st_mode):
                raise ArchiveError(f"archive namespace entry is not a directory: {entry.path}")
        chunks_root = archive_root / "chunks"
        if _lexists(chunks_root):
            expected_chunks = set(manifest["published_chunks"])
            observed_chunks = {entry.name for entry in os.scandir(chunks_root)}
            chunks_ok = (
                observed_chunks.issubset(expected_chunks)
                if partial else observed_chunks == expected_chunks
            )
            if not chunks_ok:
                raise ArchiveError("archive chunks namespace mismatch")
    for component in manifest["components"]:
        disposition = component.get("disposition")
        source = Path(component["source_path"])
        archive_path = component.get("archive_path")
        if disposition == "retained":
            if (
                partial or transition
                or component["component_id"] not in RETAINED_EVIDENCE_DIRS
            ):
                _require_inventory(
                    source, component["inventory"], component["component_id"]
                )
            else:
                _require_retained_append_only(
                    source, component["inventory"], component["component_id"]
                )
            continue
        if disposition != "moved" or not isinstance(archive_path, str):
            raise ArchiveError("archive manifest component is malformed")
        destination = archive_root / archive_path
        if partial:
            source_exists = _lexists(source)
            destination_exists = _lexists(destination)
            if source_exists == destination_exists:
                raise ArchiveError(
                    f"ambiguous transaction component state: {component['component_id']}"
                )
            selected = source if source_exists else destination
        else:
            if not _lexists(destination):
                raise ArchiveError(f"archived component is missing: {destination}")
            if transition and _lexists(source):
                raise ArchiveError(
                    f"moved source reappeared after archive: {component['component_id']}"
                )
            if component["component_id"] == "staging" and _lexists(source):
                raise ArchiveError("archived source-head staging tree reappeared")
            selected = destination
        _require_inventory(selected, component["inventory"], component["component_id"])


def _verify_archived_operand_bindings(manifest: dict, archive_root: Path) -> None:
    """Bind retained roots to immutable paths inside archived DONE manifests."""
    expected_matrix = Path(manifest["input_root"]) / "sm86_reference_capture_matrix.json"
    expected_authority = (
        Path(manifest["input_root"]) / "capture_roster_authority_sm86.json"
    )
    expected_preflight_root = Path(manifest["preflight_root"])
    for chunk_id in manifest["published_chunks"]:
        done_path = archive_root / "chunks" / chunk_id / "DONE.json"
        _require_regular(done_path, "archived published DONE")
        done = _load_json(done_path)
        inputs = done.get("input_paths")
        if not isinstance(inputs, dict):
            raise ArchiveError(f"archived DONE input_paths is missing: {done_path}")
        matrix = inputs.get("capture_matrix")
        authority = inputs.get("roster_authority")
        preflight = inputs.get("preflight")
        if not all(isinstance(value, str) for value in (matrix, authority, preflight)):
            raise ArchiveError(f"archived DONE input path is malformed: {done_path}")
        if _absolute(Path(matrix)) != expected_matrix:
            raise ArchiveError("archived DONE matrix path contradicts archive input_root")
        if _absolute(Path(authority)) != expected_authority:
            raise ArchiveError("archived DONE authority path contradicts archive input_root")
        if _absolute(Path(preflight)).parent != expected_preflight_root:
            raise ArchiveError(
                "archived DONE preflight path contradicts archive preflight_root"
            )


def _check_archive_unlocked(campaign_root: Path, source_head: str) -> dict:
    campaign_root = _absolute(campaign_root)
    _require_hex(source_head, 40, "source_head")
    archive_root = campaign_root / ARCHIVE_DIR / source_head
    _assert_path_chain_no_symlinks(archive_root)
    manifest = _load_manifest(archive_root / MANIFEST_NAME)
    if manifest.get("campaign_root") != os.fspath(campaign_root):
        raise ArchiveError("archive campaign_root binding mismatch")
    if manifest.get("archive_root") != os.fspath(archive_root):
        raise ArchiveError("archive_root binding mismatch")
    if manifest.get("source_head") != source_head:
        raise ArchiveError("archive source_head binding mismatch")
    token = f"r28_{source_head[:8]}"
    expected_audits = {
        component["component_id"].removeprefix("audit:")
        for component in manifest["components"]
        if component["component_id"].startswith("audit:")
    }
    observed_audits: set[str] = set()
    for entry in os.scandir(campaign_root / "audits"):
        if token not in entry.name:
            continue
        info = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ArchiveError(f"matching audit is not a regular file: {entry.path}")
        observed_audits.add(entry.name)
    if observed_audits != expected_audits:
        raise ArchiveError("retained old-head audit namespace drifted")
    _verify_components(manifest, archive_root, partial=False)
    _verify_archived_operand_bindings(manifest, archive_root)
    _verify_retained_ledgers(campaign_root)
    return manifest


def check_archive(campaign_root: Path, source_head: str) -> dict:
    """Verify a final archive while holding the shared campaign writer lock."""
    campaign_root = _absolute(campaign_root)
    with _campaign_lock(campaign_root):
        return _check_archive_unlocked(campaign_root, source_head)


def _ensure_same_filesystem(manifest: dict, attempts: Path, partial: Path) -> None:
    device = attempts.stat().st_dev
    for component in manifest["components"]:
        if component["disposition"] != "moved":
            continue
        source = Path(component["source_path"])
        archived = partial / component["archive_path"]
        selected = source if _lexists(source) else archived
        if selected.stat().st_dev != device:
            raise ArchiveError(f"cross-device archive move is forbidden: {selected}")


def archive_campaign(spec: ArchiveSpec) -> tuple[str, dict]:
    """Perform or resume the exact archive transaction."""
    spec = _validate_spec(spec)
    final = spec.campaign_root / ARCHIVE_DIR / spec.source_head
    partial = spec.campaign_root / ARCHIVE_DIR / f".{spec.source_head}.partial"
    bootstrap = spec.campaign_root / ARCHIVE_DIR / f".{spec.source_head}.bootstrap"
    with _campaign_lock(spec.campaign_root):
        if _lexists(final):
            manifest = _check_archive_unlocked(spec.campaign_root, spec.source_head)
            _manifest_matches_spec(manifest, spec)
            return "already_archived", manifest
        attempts = spec.campaign_root / ARCHIVE_DIR
        if _lexists(attempts):
            _assert_path_chain_no_symlinks(attempts)
            if not attempts.is_dir():
                raise ArchiveError("campaign_revision_attempts is not a directory")

        if _lexists(partial) and _lexists(bootstrap):
            raise ArchiveError("partial and bootstrap archive transactions both exist")
        if _lexists(partial):
            _validate_runtime(spec)
            _verify_retained_ledgers(spec.campaign_root)
            _assert_path_chain_no_symlinks(partial)
            manifest = _load_manifest(partial / MANIFEST_NAME)
            _manifest_matches_spec(manifest, spec)
            _verify_components(manifest, partial, partial=True)
        else:
            if _lexists(bootstrap):
                _assert_path_chain_no_symlinks(bootstrap)
                if not bootstrap.is_dir():
                    raise ArchiveError("archive bootstrap is not a directory")
                bootstrap_entries = {
                    entry.name for entry in os.scandir(bootstrap)
                }
                if not bootstrap_entries:
                    manifest = build_archive_manifest(spec)
                    _validate_runtime(spec)
                    _write_exclusive(
                        bootstrap / MANIFEST_NAME, _canonical_bytes(manifest)
                    )
                    _fsync_dir(bootstrap)
                elif (
                    MANIFEST_NAME in bootstrap_entries
                    and bootstrap_entries.issubset(
                        {MANIFEST_NAME, MANIFEST_RECOVERY_NAME}
                    )
                ):
                    _validate_runtime(spec)
                    _verify_retained_ledgers(spec.campaign_root)
                    try:
                        manifest = _load_manifest(bootstrap / MANIFEST_NAME)
                    except ArchiveError:
                        manifest = build_archive_manifest(spec)
                        _validate_runtime(spec)
                        _replace_torn_bootstrap_manifest(bootstrap, manifest)
                    else:
                        _manifest_matches_spec(manifest, spec)
                        _verify_components(manifest, bootstrap, partial=True)
                else:
                    raise ArchiveError("archive bootstrap namespace is ambiguous")
            else:
                manifest = build_archive_manifest(spec)
                _validate_runtime(spec)
                if not _lexists(attempts):
                    try:
                        attempts.mkdir(mode=0o755)
                    except FileExistsError as exc:
                        raise ArchiveError(
                            "campaign_revision_attempts creation collided"
                        ) from exc
                    _fsync_dir(spec.campaign_root)
                try:
                    bootstrap.mkdir(mode=0o755)
                except FileExistsError as exc:
                    raise ArchiveError("archive bootstrap creation collided") from exc
                _fsync_dir(bootstrap)
                _fsync_dir(attempts)
                _write_exclusive(
                    bootstrap / MANIFEST_NAME, _canonical_bytes(manifest)
                )
                _fsync_dir(bootstrap)
            _rename_noreplace(bootstrap, partial)
            _fsync_dir(attempts)

        _ensure_same_filesystem(manifest, attempts, partial)
        # Hashing a multi-GiB inventory may take time.  Re-prove both CPU and
        # GPU quiescence under the campaign lock immediately before evidence
        # begins moving.
        _validate_runtime(spec)
        for component in manifest["components"]:
            if component["disposition"] != "moved":
                continue
            source = Path(component["source_path"])
            destination = partial / component["archive_path"]
            if _lexists(destination):
                if _lexists(source):
                    raise ArchiveError(
                        f"source/destination collision: {component['component_id']}"
                    )
                _require_inventory(
                    destination, component["inventory"], component["component_id"]
                )
                continue
            if not _lexists(source):
                raise ArchiveError(f"archive source disappeared: {source}")
            _require_inventory(source, component["inventory"], component["component_id"])
            _validate_runtime(spec)
            _mkdir_component_parent(destination.parent, partial)
            _rename_noreplace(source, destination)
            _fsync_dir(destination.parent)
            _fsync_dir(source.parent)

        _verify_components(manifest, partial, partial=False, transition=True)
        _validate_runtime(spec)
        _fsync_dir(partial)
        _rename_noreplace(partial, final)
        _fsync_dir(attempts)
        return "archived", _check_archive_unlocked(
            spec.campaign_root, spec.source_head
        )


def dry_run(spec: ArchiveSpec) -> tuple[str, dict]:
    spec = _validate_spec(spec)
    final = spec.campaign_root / ARCHIVE_DIR / spec.source_head
    with _campaign_lock(spec.campaign_root):
        if _lexists(final):
            manifest = _check_archive_unlocked(spec.campaign_root, spec.source_head)
            _manifest_matches_spec(manifest, spec)
            return "already_archived", manifest
        partial = spec.campaign_root / ARCHIVE_DIR / f".{spec.source_head}.partial"
        bootstrap = spec.campaign_root / ARCHIVE_DIR / f".{spec.source_head}.bootstrap"
        if _lexists(partial) and _lexists(bootstrap):
            raise ArchiveError("partial and bootstrap archive transactions both exist")
        if _lexists(partial):
            _validate_runtime(spec)
            _verify_retained_ledgers(spec.campaign_root)
            manifest = _load_manifest(partial / MANIFEST_NAME)
            _manifest_matches_spec(manifest, spec)
            _verify_components(manifest, partial, partial=True)
            return "resumable", manifest
        if _lexists(bootstrap):
            _assert_path_chain_no_symlinks(bootstrap)
            if not bootstrap.is_dir():
                raise ArchiveError("archive bootstrap is not a directory")
            entries = {entry.name for entry in os.scandir(bootstrap)}
            if not entries:
                return "bootstrap_empty_recoverable", build_archive_manifest(spec)
            if (
                MANIFEST_NAME not in entries
                or not entries.issubset({MANIFEST_NAME, MANIFEST_RECOVERY_NAME})
            ):
                raise ArchiveError("archive bootstrap namespace is ambiguous")
            _validate_runtime(spec)
            _verify_retained_ledgers(spec.campaign_root)
            try:
                manifest = _load_manifest(bootstrap / MANIFEST_NAME)
            except ArchiveError:
                return "bootstrap_torn_recoverable", build_archive_manifest(spec)
            _manifest_matches_spec(manifest, spec)
            _verify_components(manifest, bootstrap, partial=True)
            return "bootstrap_resumable", manifest
        return "ready", build_archive_manifest(spec)


def _summary(status: str, manifest: dict) -> dict:
    return {
        "status": status,
        "archive_root": manifest["archive_root"],
        "source_head": manifest["source_head"],
        "capture_matrix_sha256": manifest["capture_matrix_sha256"],
        "manifest_seal_sha256": manifest["manifest_seal_sha256"],
        "published_chunks": manifest["published_chunks"],
        "staging_chunks": manifest["staging_chunks"],
        "inventory_totals": manifest["inventory_totals"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "archive", "check"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--capture-checkout", type=Path)
    parser.add_argument("--preflight-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--matrix-sha256")
    parser.add_argument("--authority-sha256")
    parser.add_argument("--published-chunk", action="append", default=[])
    parser.add_argument("--staging-chunk", action="append", default=[])
    return parser


def _spec_from_args(args: argparse.Namespace) -> ArchiveSpec:
    missing = [
        name for name in (
            "capture_checkout", "preflight_root", "input_root",
            "matrix_sha256", "authority_sha256",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise ArchiveError(f"{args.mode} requires: {', '.join(missing)}")
    return ArchiveSpec(
        campaign_root=args.campaign_root,
        capture_checkout=args.capture_checkout,
        preflight_root=args.preflight_root,
        input_root=args.input_root,
        source_head=args.source_head,
        matrix_sha256=args.matrix_sha256,
        authority_sha256=args.authority_sha256,
        published_chunks=tuple(args.published_chunk),
        staging_chunks=tuple(args.staging_chunk),
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "check":
            manifest = check_archive(args.campaign_root, args.source_head)
            status = "verified"
        else:
            spec = _spec_from_args(args)
            if args.mode == "dry-run":
                status, manifest = dry_run(spec)
            else:
                status, manifest = archive_campaign(spec)
    except ArchiveError as exc:
        print(f"archive_sm86_campaign_revision: REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_summary(status, manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
