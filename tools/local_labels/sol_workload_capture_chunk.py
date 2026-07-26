#!/usr/bin/env python3
"""Capture one additive SOL workload-row chunk without changing C1 authority.

The canonical 344-cell matrix remains the campaign-wide protocol authority.
The supplemental matrix is a separate, self-contained execution authority for
explicit ``(slug, source_workload_uuid, variant_id)`` rows, including rows from
problems that are not runnable members of the canonical matrix.  Native,
partial-native, and reduced-geometry claims remain sealed as distinct cohorts.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LABELS = REPO_ROOT / "tools" / "local_labels"
for _path in (REPO_ROOT, LOCAL_LABELS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_sol_workload_capture_matrix as workload_matrix  # noqa: E402
import build_sol_representative_execution_matrix as representative_execution  # noqa: E402
import campaign_preflight as preflight  # noqa: E402
import sm86_capture_chunk as canonical_capture  # noqa: E402
from tools.local_labels import metric_registry  # noqa: E402
import sol_capture_environment as sol_environment  # noqa: E402
import sol_live_workload_coverage_inventory as live_coverage  # noqa: E402
import sol_reconstitution as sol_recon  # noqa: E402
from tools.local_labels import protected_paths  # noqa: E402
from tools.cloud.trace_budgets import (  # noqa: E402
    CellBudget,
    TraceBudgetsConfig,
    load_trace_budgets,
)


DEFAULT_MATRIX = REPO_ROOT / workload_matrix.DEFAULT_OUTPUT_PATH
DEFAULT_CANONICAL_MATRIX = REPO_ROOT / workload_matrix.CANONICAL_MATRIX_PATH
DEFAULT_AUTHORITY = (
    REPO_ROOT / "data/local_labels/campaign/capture_roster_authority_sm86.json"
)
DEFAULT_PINS = LOCAL_LABELS / "campaign_protocol_pins.json"
DEFAULT_AGGREGATE_BUDGET = (
    REPO_ROOT / "data/local_labels/campaign/trace_retention_budget.json"
)
DEFAULT_CELL_BUDGETS = (
    REPO_ROOT / "workloads/simbench_bw/trace_budgets.toml"
)
DEFAULT_REGISTRY = LOCAL_LABELS / "metric_registry.json"
MATRIX_SCHEMA = REPO_ROOT / workload_matrix.SCHEMA_PATH
REPRESENTATIVE_MATRIX_SCHEMA = REPO_ROOT / representative_execution.SCHEMA_PATH
DONE_SCHEMA = (
    REPO_ROOT / "workloads/simbench_bw/schemas"
    / "sol_workload_capture_chunk_done.schema.json"
)
LIVE_COVERAGE_SCHEMA = (
    REPO_ROOT / "workloads/simbench_bw/schemas"
    / "sol_live_workload_coverage_inventory.schema.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"^problem_[0-9]{3}_[A-Za-z0-9_]+$")
GIB = 1024 ** 3
SUPPLEMENTAL_ROOT_MARKER = ".sol_workload_capture_root.json"
SUPPLEMENTAL_ROOT_KIND = "sol_workload_capture_supplemental_v1"
RETAINED_OWNER_NAMESPACE_ROOT = Path(
    "sol_workload_campaign/inputs/owner_namespaces"
)
RETAINED_OWNER_NAMESPACE_KIND = "sol_workload_retained_owner_namespace_v1"
PROTECTED_CANONICAL_CAMPAIGN_ROOTS = (
    Path("/home/jakeshea/sm86_reference_capture_r27").resolve(),
)


class WorkloadCaptureRefusal(RuntimeError):
    """Fail-closed supplemental capture refusal with a stable code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


STORAGE_CAPACITY_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
STORAGE_INTEGRITY_ERRNOS = frozenset({
    errno.EIO, errno.EROFS,
    *(
        value for value in (
            getattr(errno, "ENXIO", None), getattr(errno, "ESTALE", None),
        ) if value is not None
    ),
})
STORAGE_CAPACITY_CODE = "supplemental_storage_capacity_exhausted"
STORAGE_INTEGRITY_CODE = "supplemental_storage_integrity_failure"


def storage_refusal_from_exception(
    exc: BaseException, *, operation: str,
) -> WorkloadCaptureRefusal | None:
    """Map write/fsync/rename OS failures into the closed storage taxonomy."""
    current: BaseException | None = exc
    observed: OSError | None = None
    while current is not None:
        if isinstance(current, OSError):
            observed = current
            break
        current = current.__cause__ or current.__context__
    if observed is None:
        return None
    if observed.errno in STORAGE_CAPACITY_ERRNOS:
        code = STORAGE_CAPACITY_CODE
    elif observed.errno in STORAGE_INTEGRITY_ERRNOS or observed.errno is None:
        code = STORAGE_INTEGRITY_CODE
    else:
        return None
    detail = json.dumps({
        "operation": operation,
        "errno": observed.errno,
        "error": str(observed),
    }, sort_keys=True)
    return WorkloadCaptureRefusal(code, detail)


class _CampaignLockLease:
    """Own the canonical campaign lock outside a failing chunk frame."""

    def __init__(self) -> None:
        self._stream: Any | None = None
        self._path: Path | None = None

    def acquire(self, output_root: Path) -> None:
        if self._stream is not None:
            raise WorkloadCaptureRefusal(
                "campaign_lock_invalid",
                "supplemental chunk attempted to acquire its campaign lock twice",
            )
        stream = canonical_capture.acquire_campaign_lock(output_root)
        self._stream = stream
        self._path = output_root / ".capture.lock"

    def close(self, *, preserve_active_failure: bool) -> None:
        stream = self._stream
        path = self._path
        self._stream = None
        self._path = None
        if stream is None:
            return
        try:
            stream.close()
        except OSError as exc:
            if preserve_active_failure:
                # The capture refusal remains the authoritative first fault.
                # Closing the descriptor is still attempted exactly once.
                return
            mapped = storage_refusal_from_exception(
                exc, operation="campaign_chunk_lock_close",
            )
            if mapped is not None:
                raise mapped from exc
            raise WorkloadCaptureRefusal(
                "campaign_lock_invalid", f"{path}: close: {exc}"
            ) from exc


def _raise_output_path_error(
    exc: OSError, *, path: Path, component: str, operation: str,
) -> None:
    """Prefer the closed storage taxonomy over a generic path refusal."""
    storage_refusal = storage_refusal_from_exception(
        exc, operation=operation,
    )
    if storage_refusal is not None:
        raise storage_refusal from exc
    raise WorkloadCaptureRefusal(
        "supplemental_output_component_invalid",
        f"{path}: {component}: {exc}",
    ) from exc


def _walk_directory_nofollow(path: Path, *, create: bool) -> None:
    """Traverse/create an absolute directory using dirfds and O_NOFOLLOW."""
    if ".." in path.parts:
        raise WorkloadCaptureRefusal(
            "supplemental_output_component_invalid", str(path)
        )
    absolute = path.absolute()
    if not absolute.is_absolute():
        raise WorkloadCaptureRefusal(
            "supplemental_output_component_invalid", str(path)
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open("/", flags)
    except OSError as exc:
        _raise_output_path_error(
            exc, path=absolute, component="/",
            operation="output_directory_root_open",
        )
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    return
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    _raise_output_path_error(
                        exc, path=absolute, component=component,
                        operation="output_directory_mkdir",
                    )
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as exc:
                    _raise_output_path_error(
                        exc, path=absolute, component=component,
                        operation="output_directory_open_after_mkdir",
                    )
            except OSError as exc:
                _raise_output_path_error(
                    exc, path=absolute, component=component,
                    operation="output_directory_open",
                )
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _mkdir_nofollow(path: Path, *, exist_ok: bool) -> None:
    existed = path.exists() and not path.is_symlink()
    if path.is_symlink():
        raise WorkloadCaptureRefusal(
            "supplemental_output_component_invalid", str(path)
        )
    if existed and not exist_ok:
        raise FileExistsError(path)
    _walk_directory_nofollow(path, create=True)


def _write_bytes_new_nofollow(
    path: Path, payload: bytes | bytearray, *, mode: int, operation: str,
) -> None:
    """Create, write, and fsync one output file with local storage typing."""
    _mkdir_nofollow(path.parent, exist_ok=True)
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(path.parent, parent_flags)
        try:
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                mode, dir_fd=parent_fd,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError(None, "short exclusive output write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except FileExistsError:
        raise
    except OSError as exc:
        mapped = storage_refusal_from_exception(exc, operation=operation)
        if mapped is not None:
            raise mapped from exc
        # A short write without an errno is an integrity failure at this
        # explicit output boundary, never an input error.
        if exc.errno is None:
            raise WorkloadCaptureRefusal(
                STORAGE_INTEGRITY_CODE,
                json.dumps({
                    "operation": operation, "errno": None,
                    "error": str(exc),
                }, sort_keys=True),
            ) from exc
        raise


def _open_regular_nofollow(
    path: Path, *, role: str,
) -> tuple[int, Path, os.stat_result]:
    """Open one lexical path without following any ancestor or leaf symlink."""
    lexical = Path(path)
    if ".." in lexical.parts:
        raise WorkloadCaptureRefusal(
            "supplemental_input_path_invalid", f"{role}: {lexical}"
        )
    absolute = lexical.absolute()
    parts = absolute.parts[1:]
    if not parts:
        raise WorkloadCaptureRefusal(
            "supplemental_input_not_regular", f"{role}: {absolute}"
        )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open("/", directory_flags)
    try:
        for component in parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=descriptor,
        )
    except OSError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_input_path_invalid", f"{role}: {absolute}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    observed = os.fstat(file_descriptor)
    if not stat.S_ISREG(observed.st_mode):
        os.close(file_descriptor)
        raise WorkloadCaptureRefusal(
            "supplemental_input_not_regular", f"{role}: {absolute}"
        )
    return file_descriptor, absolute, observed


def sha256_file(path: Path) -> str:
    descriptor, _absolute, _observed = _open_regular_nofollow(
        path, role="SHA-256 input"
    )
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _output_storage_refusal(
    exc: BaseException, *, operation: str,
) -> WorkloadCaptureRefusal | None:
    """Map only a known campaign-output operation into storage taxonomy."""
    return storage_refusal_from_exception(exc, operation=operation)


def _read_output_regular_nofollow(
    path: Path, *, role: str, operation: str,
) -> tuple[bytes, os.stat_result]:
    """Read one known output artifact; preserve the first read/close fault."""
    try:
        descriptor, _absolute, observed = _open_regular_nofollow(
            path, role=role,
        )
    except BaseException as exc:
        mapped = _output_storage_refusal(exc, operation=f"{operation}_open")
        if mapped is not None:
            raise mapped from exc
        raise
    active_error: BaseException | None = None
    try:
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks), observed
    except BaseException as exc:
        active_error = exc
        mapped = _output_storage_refusal(exc, operation=f"{operation}_read")
        if mapped is not None:
            raise mapped from exc
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            if active_error is None:
                mapped = _output_storage_refusal(
                    exc, operation=f"{operation}_close",
                )
                if mapped is not None:
                    raise mapped from exc
                raise


def _sha256_output_file(path: Path, *, role: str, operation: str) -> str:
    """Hash a known output artifact without classifying unrelated inputs."""
    try:
        descriptor, _absolute, _observed = _open_regular_nofollow(
            path, role=role,
        )
    except BaseException as exc:
        mapped = _output_storage_refusal(exc, operation=f"{operation}_open")
        if mapped is not None:
            raise mapped from exc
        raise
    active_error: BaseException | None = None
    try:
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()
    except BaseException as exc:
        active_error = exc
        mapped = _output_storage_refusal(exc, operation=f"{operation}_read")
        if mapped is not None:
            raise mapped from exc
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            if active_error is None:
                mapped = _output_storage_refusal(
                    exc, operation=f"{operation}_close",
                )
                if mapped is not None:
                    raise mapped from exc
                raise


def _fsync_output_regular_nofollow(
    path: Path, *, role: str, operation: str,
) -> None:
    """Fsync one known output file without following a replaced symlink."""
    try:
        descriptor, _absolute, _observed = _open_regular_nofollow(
            path, role=role,
        )
    except BaseException as exc:
        mapped = _output_storage_refusal(exc, operation=f"{operation}_open")
        if mapped is not None:
            raise mapped from exc
        raise
    active_error: BaseException | None = None
    try:
        os.fsync(descriptor)
    except BaseException as exc:
        active_error = exc
        mapped = _output_storage_refusal(exc, operation=f"{operation}_fsync")
        if mapped is not None:
            raise mapped from exc
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            if active_error is None:
                mapped = _output_storage_refusal(
                    exc, operation=f"{operation}_close",
                )
                if mapped is not None:
                    raise mapped from exc
                raise


def _load_output_object(
    path: Path, *, role: str, operation: str,
) -> dict[str, Any]:
    payload, _observed = _read_output_regular_nofollow(
        path, role=role, operation=operation,
    )
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_input_invalid", f"{role}: {path}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_input_invalid", f"{role} root is not an object",
        )
    return value


def _output_lstat(path: Path, *, operation: str) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        mapped = _output_storage_refusal(exc, operation=operation)
        if mapped is not None:
            raise mapped from exc
        raise


def _resolve_output_path(
    path: Path, *, operation: str, strict: bool = True,
) -> Path:
    try:
        return path.resolve(strict=strict)
    except OSError as exc:
        mapped = _output_storage_refusal(exc, operation=operation)
        if mapped is not None:
            raise mapped from exc
        raise


def _output_directory_entries(
    directory: Path, *, operation: str,
) -> list[Path]:
    try:
        with os.scandir(directory) as entries:
            return [
                Path(entry.path)
                for entry in sorted(entries, key=lambda item: item.name)
            ]
    except OSError as exc:
        mapped = _output_storage_refusal(
            exc, operation=f"{operation}_scandir",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _output_tree_paths(root: Path, *, operation: str) -> list[Path]:
    """Walk a known output tree without pathlib's suppressed OSError path."""
    paths: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                observed = sorted(entries, key=lambda item: item.name)
        except OSError as exc:
            mapped = _output_storage_refusal(
                exc, operation=f"{operation}_scandir",
            )
            if mapped is not None:
                raise mapped from exc
            raise
        for entry in observed:
            path = Path(entry.path)
            paths.append(path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                mapped = _output_storage_refusal(
                    exc, operation=f"{operation}_entry_stat",
                )
                if mapped is not None:
                    raise mapped from exc
                raise
            if is_directory:
                walk(path)

    walk(root)
    return paths


def _git_blob_bytes(
    source_commit: str, repository_path: str, *, repo_root: Path = REPO_ROOT,
) -> bytes | None:
    """Return raw committed bytes, or ``None`` when the path is not tracked."""
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise WorkloadCaptureRefusal(
            "supplemental_source_commit_invalid", source_commit
        )
    environment = os.environ.copy()
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_DIR",
        "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX",
        "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "blob",
         f"{source_commit}:{repository_path}"],
        capture_output=True, check=False, env=environment,
    )
    if completed.returncode == 0:
        return completed.stdout
    missing = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e",
         f"{source_commit}:{repository_path}"],
        capture_output=True, check=False, env=environment,
    )
    if missing.returncode != 0:
        return None
    raise WorkloadCaptureRefusal(
        "supplemental_git_blob_unreadable",
        f"{source_commit}:{repository_path}: {completed.stderr[-400:]!r}",
    )


def input_file_identity(
    path: Path, *, role: str, source_commit: str,
) -> dict[str, Any]:
    """Create a relocation-safe identity and retained-owner path.

    Absolute paths, devices, and inodes are deliberately excluded.  A tracked
    input is bound to its raw Git blob at ``source_commit``.  Every other input
    is rebound into this campaign's owner namespace by exact bytes.
    """
    descriptor, absolute, observed = _open_regular_nofollow(path, role=role)
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            payload.extend(block)
    finally:
        os.close(descriptor)
    role_name = re.sub(r"[^a-z0-9_]+", "_", role.lower()).strip("_")
    if not role_name:
        raise WorkloadCaptureRefusal(
            "supplemental_input_role_invalid", role
        )
    retained = (
        Path("sol_workload_campaign/inputs/source") / source_commit
        / role_name / absolute.name
    ).as_posix()
    common = {
        "role": role_name,
        "size_bytes": observed.st_size,
        "sha256": digest.hexdigest(),
        "retained_owner_relative_path": retained,
    }
    try:
        repository_path = absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        repository_path = None
    if repository_path is not None:
        committed = _git_blob_bytes(source_commit, repository_path)
        if committed is not None:
            committed_sha = hashlib.sha256(committed).hexdigest()
            if committed != bytes(payload):
                raise WorkloadCaptureRefusal(
                    "supplemental_tracked_input_drift",
                    f"{repository_path}: live={common['sha256']} "
                    f"git_blob={committed_sha}",
                )
            return {
                **common,
                "kind": "tracked_git_blob",
                "repository_path": repository_path,
                "source_commit": source_commit,
                "git_blob_sha256": committed_sha,
            }
    return {
        **common,
        "kind": "retained_external_input",
        "owner_marker_name": SUPPLEMENTAL_ROOT_MARKER,
        "owner_relative_path": retained,
        "source_name": absolute.name,
    }


def _portable_input_reference(identity: dict[str, Any]) -> dict[str, Any]:
    """Project one runtime identity into the closed relocatable DONE union."""
    common = {
        "kind": identity.get("kind"),
        "role": identity.get("role"),
        "size_bytes": identity.get("size_bytes"),
        "sha256": identity.get("sha256"),
    }
    if identity.get("kind") == "tracked_git_blob":
        retained = identity.get("retained_owner_relative_path")
        return {
            **common,
            "repository_path": identity.get("repository_path"),
            "source_commit": identity.get("source_commit"),
            "git_blob_sha256": identity.get("git_blob_sha256"),
            "owner_relative_path": retained,
            "source_name": (
                Path(retained).name if isinstance(retained, str) else None
            ),
        }
    if identity.get("kind") == "retained_external_input":
        return {
            **common,
            "owner_relative_path": identity.get("owner_relative_path"),
            "source_name": identity.get("source_name"),
        }
    raise WorkloadCaptureRefusal(
        "supplemental_input_reference_invalid", repr(identity)
    )


def _safe_relative_path(value: Any, *, role: str) -> Path:
    path = Path(value) if isinstance(value, str) else Path(".")
    if (
        not isinstance(value, str) or not value
        or path.is_absolute() or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_path_invalid", f"{role}: {value!r}"
        )
    return path


def _read_portable_input_reference(
    reference: dict[str, Any], *, campaign_root: Path, repo_root: Path,
    source_commit: str, verify_git_provenance: bool = False,
) -> bytes:
    """Resolve one DONE input without consulting its original live path."""
    role = reference.get("role")
    if not isinstance(role, str) or not re.fullmatch(r"[a-z0-9_]+", role):
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_invalid", repr(reference)
        )
    kind = reference.get("kind")
    if kind == "tracked_git_blob":
        repository_path = _safe_relative_path(
            reference.get("repository_path"), role=role
        ).as_posix()
        if reference.get("source_commit") != source_commit:
            raise WorkloadCaptureRefusal(
                "supplemental_input_reference_commit_mismatch", role
            )
        if reference.get("git_blob_sha256") != reference.get("sha256"):
            raise WorkloadCaptureRefusal(
                "supplemental_input_reference_blob_mismatch", role
            )
        if verify_git_provenance:
            committed = _git_blob_bytes(
                source_commit, repository_path, repo_root=repo_root
            )
            if committed is None:
                raise WorkloadCaptureRefusal(
                    "supplemental_input_reference_blob_missing",
                    f"{source_commit}:{repository_path}",
                )
            if hashlib.sha256(committed).hexdigest() != reference.get(
                "git_blob_sha256"
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_input_reference_blob_mismatch", role
                )
    elif kind != "retained_external_input":
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_invalid", repr(reference)
        )
    if kind in {"tracked_git_blob", "retained_external_input"}:
        relative = _safe_relative_path(
            reference.get("owner_relative_path"), role=role
        )
        source_name = reference.get("source_name")
        if (
            not isinstance(source_name, str) or not source_name
            or source_name in {".", ".."} or "/" in source_name
            or "\\" in source_name
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_input_reference_path_invalid", role
            )
        path = campaign_root / relative
        payload, observed = _read_output_regular_nofollow(
            path, role=f"retained DONE input {role}",
            operation="retained_done_input_read",
        )
        if (
            observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
            or observed.st_mode & 0o222
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_input_reference_owner_invalid", role
            )
    if (
        type(reference.get("size_bytes")) is not int
        or len(payload) != reference["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != reference.get("sha256")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_hash_mismatch", role
        )
    return payload


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _read_immutable_campaign_payload(path: Path, *, role: str) -> bytes:
    payload, observed = _read_output_regular_nofollow(
        path, role=role, operation="immutable_campaign_output_read",
    )
    if (
        observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o222
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_owner_invalid", role
        )
    return payload


def _parse_canonical_object_payload(payload: bytes, *, role: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate key {key!r}")
            parsed[key] = value
        return parsed

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_json_invalid", f"{role}: {exc}"
        ) from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_json_invalid",
            f"{role}: noncanonical object",
        )
    return value


def _load_retained_input_owner(
    campaign_root: Path,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    """Load the relocatable campaign owner marker without source-owner access."""
    _validate_owner_directory(campaign_root)
    marker_path = campaign_root / SUPPLEMENTAL_ROOT_MARKER
    payload = _read_immutable_campaign_payload(
        marker_path, role="supplemental retained input owner"
    )
    marker = _parse_canonical_object_payload(
        payload, role="supplemental retained input owner"
    )
    if set(marker) != {
        "schema_version", "root_kind", "campaign_identity",
        "campaign_identity_sha256", "created_at_utc", "root_identity_sha256",
    } or marker.get("schema_version") != 1 or marker.get(
        "root_kind"
    ) != SUPPLEMENTAL_ROOT_KIND:
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_identity_invalid", repr(marker)
        )
    _verify_sealed(
        marker, "root_identity_sha256",
        code="supplemental_output_root_identity_invalid",
    )
    campaign_identity = marker.get("campaign_identity")
    if (
        not isinstance(campaign_identity, dict)
        or marker.get("campaign_identity_sha256")
        != canonical_sha256(campaign_identity)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_identity_invalid", repr(marker)
        )
    pin = {
        "marker_name": SUPPLEMENTAL_ROOT_MARKER,
        "marker_sha256": hashlib.sha256(payload).hexdigest(),
        "root_identity_sha256": marker["root_identity_sha256"],
        "campaign_identity_sha256": marker["campaign_identity_sha256"],
    }
    return marker, payload, pin


def _retained_external_reference(
    *, path: Path, campaign_root: Path, role: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        relative = path.absolute().relative_to(campaign_root.absolute())
    except ValueError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_input_reference_path_invalid", f"{role}: {path}"
        ) from exc
    relative = _safe_relative_path(relative.as_posix(), role=role)
    payload = _read_immutable_campaign_payload(path, role=role)
    reference = {
        "kind": "retained_external_input",
        "role": role,
        "owner_relative_path": relative.as_posix(),
        "source_name": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return reference, payload


def _retained_preflight_references(
    *, campaign_root: Path, source_commit: str, chunk_id: str,
    required_sha256: set[str],
) -> list[dict[str, Any]]:
    if not required_sha256 or any(
        SHA256_RE.fullmatch(value) is None for value in required_sha256
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_preflight_set_invalid", repr(required_sha256)
        )
    parent = (
        campaign_root / "sol_workload_campaign/inputs/preflights"
        / source_commit / chunk_id
    )
    _walk_directory_nofollow(parent, create=False)
    if parent.is_symlink() or not parent.is_dir():
        raise WorkloadCaptureRefusal(
            "supplemental_done_preflight_set_invalid", str(parent)
        )
    matches: dict[str, list[dict[str, Any]]] = {
        digest: [] for digest in required_sha256
    }
    for path in _output_directory_entries(
        parent, operation="retained_preflight_directory",
    ):
        observed = _output_lstat(
            path, operation="retained_preflight_entry_lstat",
        )
        if observed is None or stat.S_ISLNK(
            observed.st_mode
        ) or not stat.S_ISREG(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_done_preflight_set_invalid", str(path)
            )
        reference, _payload = _retained_external_reference(
            path=path, campaign_root=campaign_root, role="preflight"
        )
        digest = reference["sha256"]
        if digest in matches:
            matches[digest].append(reference)
    if any(len(rows) != 1 for rows in matches.values()):
        raise WorkloadCaptureRefusal(
            "supplemental_done_preflight_set_invalid",
            repr({key: len(value) for key, value in matches.items()}),
        )
    return sorted(
        (rows[0] for rows in matches.values()),
        key=lambda row: (row["sha256"], row["owner_relative_path"]),
    )


def _campaign_input_sources(
    *, matrix_path: Path, canonical_matrix_path: Path, authority_path: Path,
    pins_path: Path, aggregate_budget_path: Path, cell_budget_path: Path,
    registry_path: Path, full_coverage_inventory_path: Path,
    live_coverage_inventory_path: Path,
    runnability_path: Path, shrink_path: Path,
    external_sources: dict[str, Path] | None = None,
) -> dict[str, Path]:
    sources = {
        "supplemental_matrix": matrix_path,
        "canonical_matrix": canonical_matrix_path,
        "authority": authority_path,
        "pins": pins_path,
        "aggregate_budget": aggregate_budget_path,
        "cell_budgets": cell_budget_path,
        "metric_registry": registry_path,
        "full_coverage_inventory": full_coverage_inventory_path,
        "live_coverage_inventory": live_coverage_inventory_path,
        "runnability": runnability_path,
        "shrink_provenance": shrink_path,
    }
    for role, path in sorted((external_sources or {}).items()):
        if role in sources:
            raise WorkloadCaptureRefusal(
                "supplemental_campaign_input_role_duplicate", role
            )
        sources[role] = path
    return sources


def _reduced_source_role(
    source: dict[str, Any], *, input_kind: str,
) -> str:
    if input_kind not in {"workload", "definition"}:
        raise WorkloadCaptureRefusal(
            "supplemental_reduced_source_kind_invalid", input_kind
        )
    return f"reduced_{input_kind}_" + canonical_sha256({
        "name": source.get("name"), "sha256": source.get("sha256"),
    })


def _validate_external_reduced_source(
    *, root_identity: dict[str, Any],
    source: Any, logical: dict[str, Any], owner_marker: Any,
    input_kind: str,
) -> tuple[str, Path]:
    expected_fields = {
        "kind", "name", "sha256", "size_bytes", "owner_marker",
        "runtime_rebind",
    }
    expected_contract = {
        "workload": (
            "shrink_external_child_v1", "workload_file",
            "workload_file_sha256",
        ),
        "definition": (
            "shrink_external_definition_child_v1", "definition_file",
            "definition_file_sha256",
        ),
    }
    if input_kind not in expected_contract:
        raise WorkloadCaptureRefusal(
            "supplemental_reduced_source_kind_invalid", input_kind
        )
    kind, path_field, sha_field = expected_contract[input_kind]
    if (
        not isinstance(source, dict) or set(source) != expected_fields
        or source.get("kind") != kind
        or source.get("owner_marker") != owner_marker
        or source.get("runtime_rebind")
        != "campaign_retained_immutable_child"
        or source.get("name") != logical.get(path_field)
        or source.get("sha256") != logical.get(sha_field)
        or type(source.get("size_bytes")) is not int
        or source["size_bytes"] <= 0
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_reduced_source_contract_invalid", input_kind
        )
    try:
        payload = sol_recon.read_external_child(root_identity, source["name"])
        path = sol_recon.external_child_path(root_identity, source["name"])
    except sol_recon.PrerequisiteContractError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_reduced_source_unreadable",
            f"{input_kind}: {exc}",
        ) from exc
    if (
        len(payload) != source["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != source["sha256"]
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_reduced_source_hash_mismatch",
            f"{input_kind}: {source['name']}",
        )
    return _reduced_source_role(source, input_kind=input_kind), path


def validate_external_reduced_sources(
    matrix: dict[str, Any], *, shrink_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, str | None]]]:
    """Read every reduced input through its bound owner dirfd exactly once."""
    try:
        root_identity = sol_recon.validate_external_campaign_root(shrink_root)
    except sol_recon.PrerequisiteContractError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_reduced_owner_invalid", str(exc)
        ) from exc
    authority = matrix.get("reduced_proxy_authority") or {}
    owner_marker = authority.get("owner_marker")
    sources: dict[str, Path] = {}
    roles_by_variant: dict[str, dict[str, str | None]] = {}
    for variant in matrix.get("variants") or []:
        if not isinstance(variant, dict) or variant.get("cohort") != "reduced_proxy":
            continue
        logical = ((variant.get("invocation") or {}).get("logical_workload") or {})
        roles: dict[str, str | None] = {}
        source_pairs = [("workload", logical.get("external_workload_source"))]
        definition_source = logical.get("external_definition_source")
        if definition_source is not None:
            source_pairs.append(("definition", definition_source))
        else:
            roles["definition"] = None
        for input_kind, source in source_pairs:
            try:
                role, path = _validate_external_reduced_source(
                    root_identity=root_identity, source=source,
                    logical=logical, owner_marker=owner_marker,
                    input_kind=input_kind,
                )
            except WorkloadCaptureRefusal as exc:
                raise WorkloadCaptureRefusal(
                    exc.code,
                    f"variant={variant.get('variant_id')!r}; {exc.detail}",
                ) from exc
            prior = sources.get(role)
            if prior is not None and prior != path:
                raise WorkloadCaptureRefusal(
                    "supplemental_reduced_source_role_collision", role
                )
            sources[role] = path
            roles[input_kind] = role
        roles_by_variant[variant["variant_contract_sha256"]] = roles
    return sources, roles_by_variant


def _publish_retained_input(
    *, output_root: Path, source: Path, identity: dict[str, Any],
) -> None:
    relative = Path(str(identity.get("retained_owner_relative_path", "")))
    if (
        relative.is_absolute() or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_retained_input_path_invalid", str(relative)
        )
    target = output_root / relative
    descriptor, _absolute, observed = _open_regular_nofollow(
        source, role=str(identity.get("role", "campaign input"))
    )
    try:
        payload = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            payload.extend(block)
    finally:
        os.close(descriptor)
    if (
        observed.st_size != identity.get("size_bytes")
        or hashlib.sha256(payload).hexdigest() != identity.get("sha256")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_retained_input_source_drift", str(source)
        )
    _mkdir_nofollow(target.parent, exist_ok=True)
    if target.exists() or target.is_symlink():
        _validate_immutable_owned_file(
            target, expected_sha256=identity["sha256"],
            expected_size=identity["size_bytes"],
        )
        return
    _write_bytes_new_nofollow(
        target, payload, mode=0o444,
        operation="retained_campaign_input_publication",
    )
    _validate_immutable_owned_file(
        target, expected_sha256=identity["sha256"],
        expected_size=identity["size_bytes"],
    )


def retain_campaign_inputs(
    output_root: Path, *, campaign_identity: dict[str, Any],
    sources: dict[str, Path],
) -> None:
    identities = campaign_identity.get("input_file_identities")
    if not isinstance(identities, dict) or set(identities) != set(sources):
        raise WorkloadCaptureRefusal(
            "supplemental_campaign_input_set_mismatch",
            repr({"identities": sorted(identities or {}), "sources": sorted(sources)}),
        )
    for role in sorted(sources):
        _publish_retained_input(
            output_root=output_root, source=sources[role],
            identity=identities[role],
        )


def external_owner_namespace_contract(
    *, root: Path, role: str, marker_name: str,
) -> dict[str, Any]:
    """Bind every direct child of one sealed external source namespace."""
    if not re.fullmatch(r"[a-z0-9_]+", role):
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_role_invalid", role
        )
    try:
        identity = sol_recon.validate_external_campaign_root(root)
        names = sol_recon.external_root_names(identity)
        if marker_name not in names:
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_marker_missing",
                f"{role}: {marker_name}",
            )
        files = []
        for name in sorted(names):
            payload = sol_recon.read_external_child(identity, name)
            files.append({
                "name": name, "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    except WorkloadCaptureRefusal:
        raise
    except sol_recon.PrerequisiteContractError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_invalid", f"{role}: {exc}"
        ) from exc
    document = {
        "schema_version": 1,
        "kind": RETAINED_OWNER_NAMESPACE_KIND,
        "role": role,
        "marker_name": marker_name,
        "files": files,
    }
    document["namespace_seal_sha256"] = canonical_sha256(document)
    return document


def capture_owner_namespace_contracts(
    *, roots: dict[str, Path], marker_names: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if set(roots) != {"plan", "runnability", "shrink"} or set(
        marker_names
    ) != set(roots):
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_set_invalid",
            repr({"roots": sorted(roots), "markers": sorted(marker_names)}),
        )
    return {
        role: external_owner_namespace_contract(
            root=roots[role], role=role, marker_name=marker_names[role]
        )
        for role in sorted(roots)
    }


def _validate_owner_namespace_contract(
    contract: dict[str, Any], *, expected_role: str,
) -> None:
    if (
        not isinstance(contract, dict)
        or set(contract) != {
            "schema_version", "kind", "role", "marker_name", "files",
            "namespace_seal_sha256",
        }
        or contract.get("schema_version") != 1
        or contract.get("kind") != RETAINED_OWNER_NAMESPACE_KIND
        or contract.get("role") != expected_role
        or not isinstance(contract.get("marker_name"), str)
        or contract.get("namespace_seal_sha256")
        != canonical_sha256({
            key: value for key, value in contract.items()
            if key != "namespace_seal_sha256"
        })
        or not isinstance(contract.get("files"), list)
        or not contract["files"]
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_contract_invalid", expected_role
        )
    names = []
    for row in contract["files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "size_bytes", "sha256"}
            or not isinstance(row.get("name"), str)
            or row["name"] in {"", ".", ".."}
            or "/" in row["name"] or "\\" in row["name"]
            or Path(row["name"]).name != row["name"]
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] <= 0
            or SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_contract_invalid", expected_role
            )
        names.append(row["name"])
    if names != sorted(set(names)) or contract["marker_name"] not in names:
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_contract_invalid", expected_role
        )


def retain_owner_namespaces(
    output_root: Path, *, contracts: dict[str, dict[str, Any]],
    source_roots: dict[str, Path],
) -> None:
    """Copy dependency-closed, flat source namespaces into the campaign."""
    if set(contracts) != {"plan", "runnability", "shrink"} or set(
        source_roots
    ) != set(contracts):
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_set_invalid", repr(contracts)
        )
    for role in sorted(contracts):
        contract = contracts[role]
        _validate_owner_namespace_contract(contract, expected_role=role)
        try:
            source_identity = sol_recon.validate_external_campaign_root(
                source_roots[role]
            )
        except sol_recon.PrerequisiteContractError as exc:
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_invalid", f"{role}: {exc}"
            ) from exc
        target_root = output_root / RETAINED_OWNER_NAMESPACE_ROOT / role
        _mkdir_nofollow(target_root, exist_ok=True)
        try:
            target_root.chmod(0o700)
        except OSError as exc:
            mapped = storage_refusal_from_exception(
                exc, operation="retained_owner_namespace_mode",
            )
            if mapped is not None:
                raise mapped from exc
            raise
        if {
            path.name for path in _output_directory_entries(
                target_root, operation="retained_owner_namespace_entries",
            )
        } - {
            row["name"] for row in contract["files"]
        }:
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_extra", role
            )
        for row in contract["files"]:
            try:
                payload = sol_recon.read_external_child(
                    source_identity, row["name"]
                )
            except sol_recon.PrerequisiteContractError as exc:
                raise WorkloadCaptureRefusal(
                    "supplemental_owner_namespace_invalid", f"{role}: {exc}"
                ) from exc
            if (
                len(payload) != row["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != row["sha256"]
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_owner_namespace_source_drift",
                    f"{role}: {row['name']}",
                )
            target = target_root / row["name"]
            if _output_lstat(
                target, operation="retained_owner_namespace_target_lstat",
            ) is not None:
                _validate_immutable_owned_file(
                    target, expected_sha256=row["sha256"],
                    expected_size=row["size_bytes"],
                )
            else:
                _write_bytes_new_nofollow(
                    target, payload, mode=0o444,
                    operation="retained_owner_namespace_publication",
                )
        validate_retained_owner_namespaces(
            output_root, contracts={role: contract}, expected_roles={role}
        )


def validate_retained_owner_namespaces(
    output_root: Path, *, contracts: dict[str, dict[str, Any]],
    expected_roles: set[str] | None = None,
) -> dict[str, Path]:
    roles = expected_roles or {"plan", "runnability", "shrink"}
    if set(contracts) != roles:
        raise WorkloadCaptureRefusal(
            "supplemental_owner_namespace_set_invalid", repr(contracts)
        )
    result: dict[str, Path] = {}
    for role in sorted(roles):
        contract = contracts[role]
        _validate_owner_namespace_contract(contract, expected_role=role)
        root = output_root / RETAINED_OWNER_NAMESPACE_ROOT / role
        try:
            identity = sol_recon.validate_external_campaign_root(root)
            names = sol_recon.external_root_names(identity)
        except (OSError, sol_recon.PrerequisiteContractError) as exc:
            mapped = _output_storage_refusal(
                exc, operation="retained_owner_namespace_open",
            )
            if mapped is not None:
                raise mapped from exc
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_invalid", f"{role}: {exc}"
            ) from exc
        expected_names = {row["name"] for row in contract["files"]}
        if names != expected_names:
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_closure_mismatch",
                f"{role}: {sorted(names)}",
            )
        for row in contract["files"]:
            try:
                payload = sol_recon.read_external_child(identity, row["name"])
            except (OSError, sol_recon.PrerequisiteContractError) as exc:
                mapped = _output_storage_refusal(
                    exc, operation="retained_owner_namespace_child_read",
                )
                if mapped is not None:
                    raise mapped from exc
                raise WorkloadCaptureRefusal(
                    "supplemental_owner_namespace_invalid",
                    f"{role}: {exc}",
                ) from exc
            if (
                len(payload) != row["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != row["sha256"]
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_owner_namespace_hash_mismatch",
                    f"{role}: {row['name']}",
                )
        result[role] = root
    return result


def retained_reduced_input_paths(
    *, resolved: list[tuple[dict, dict, dict]],
    roles_by_variant: dict[str, dict[str, str | None]],
    campaign_identity: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    identities = campaign_identity.get("input_file_identities") or {}
    workloads: dict[str, Path] = {}
    definitions: dict[str, Path] = {}
    expected_variants = {
        variant.get("variant_contract_sha256")
        for _row, variant, _cell in resolved
        if variant.get("cohort") == "reduced_proxy"
    }
    for _row, variant, _cell in resolved:
        if variant.get("cohort") != "reduced_proxy":
            continue
        variant_sha = variant.get("variant_contract_sha256")
        logical = ((variant.get("invocation") or {}).get("logical_workload") or {})
        roles = roles_by_variant.get(variant_sha)
        if not isinstance(roles, dict) or set(roles) != {
            "workload", "definition",
        }:
            raise WorkloadCaptureRefusal(
                "supplemental_retained_reduced_identity_invalid",
                repr(variant_sha),
            )
        source_pairs = {
            "workload": logical.get("external_workload_source"),
            "definition": logical.get("external_definition_source"),
        }
        for input_kind, destination in (
            ("workload", workloads), ("definition", definitions),
        ):
            role = roles[input_kind]
            source = source_pairs[input_kind]
            if role is None:
                if source is not None or input_kind == "workload":
                    raise WorkloadCaptureRefusal(
                        "supplemental_retained_reduced_identity_invalid",
                        f"{variant_sha}: {input_kind}",
                    )
                continue
            identity = identities.get(role)
            if (
                not isinstance(source, dict)
                or not isinstance(identity, dict)
                or identity.get("sha256") != source.get("sha256")
                or identity.get("size_bytes") != source.get("size_bytes")
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_retained_reduced_identity_invalid",
                    f"{variant_sha}: {input_kind}",
                )
            relative = Path(str(
                identity.get("retained_owner_relative_path", "")
            ))
            path = output_root / relative
            _validate_immutable_owned_file(
                path, expected_sha256=source["sha256"],
                expected_size=source["size_bytes"],
            )
            destination[variant_sha] = path
    if set(workloads) != expected_variants or set(definitions) != {
        variant_sha for variant_sha in expected_variants
        if (roles := roles_by_variant.get(variant_sha)) is not None
        if roles.get("definition") is not None
    }:
        raise WorkloadCaptureRefusal(
            "supplemental_retained_reduced_set_mismatch",
            repr({
                "workloads": sorted(workloads),
                "definitions": sorted(definitions),
                "roles": sorted(roles_by_variant),
            }),
        )
    return workloads, definitions


def canonical_sha256(value: Any) -> str:
    return workload_matrix.canonical_sha256(value)


def _load_object(path: Path, *, role: str) -> dict[str, Any]:
    descriptor, absolute, _observed = _open_regular_nofollow(path, role=role)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_input_invalid", f"{role}: {absolute}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_input_invalid", f"{role} root is not an object"
        )
    return value


def lexical_input_path(value: str | os.PathLike[str]) -> Path:
    """Anchor a CLI input without resolving away symlink evidence."""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).absolute()


def assert_external_output_root_before_mutation(output_root: Path) -> None:
    """Reject protected/repository output paths without creating anything."""
    if not output_root.is_absolute():
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_not_absolute", str(output_root)
        )
    _walk_directory_nofollow(output_root, create=False)
    observed = _output_lstat(
        output_root, operation="supplemental_output_root_lstat",
    )
    if observed is not None and (
        stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_invalid", str(output_root)
        )
    resolved = _resolve_output_path(
        output_root, operation="supplemental_output_root_resolve",
        strict=False,
    )
    try:
        protected_paths.assert_write_allowed(
            resolved, repo_root=REPO_ROOT,
            operation="publish supplemental SOL capture evidence",
        )
    except protected_paths.ProtectedPathError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_protected", str(exc)
        ) from exc
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_in_repository", str(output_root)
        )
    if protected_paths.is_inside_git_worktree(resolved):
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_in_worktree", str(output_root)
        )
    _validate_owner_directory(output_root)
    for protected in PROTECTED_CANONICAL_CAMPAIGN_ROOTS:
        if (
            resolved == protected
            or resolved in protected.parents
            or protected in resolved.parents
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_output_root_aliases_canonical",
                f"output={resolved}; protected={protected}",
            )
    return


def assert_output_disjoint_from_input_owners(
    output_root: Path, *, input_roots: dict[str, Path],
) -> None:
    """Reject ancestor/descendant aliases before either namespace is mutated."""
    output = output_root.absolute().resolve(strict=False)
    for role, root in input_roots.items():
        try:
            owner = Path(root).absolute().resolve(strict=True)
        except OSError as exc:
            raise WorkloadCaptureRefusal(
                "supplemental_input_owner_invalid", f"{role}: {root}: {exc}"
            ) from exc
        if output == owner or output in owner.parents or owner in output.parents:
            raise WorkloadCaptureRefusal(
                "supplemental_output_input_owner_alias",
                f"output={output}; {role}={owner}",
            )


def _validate_owner_directory(path: Path) -> None:
    """Require the nearest existing namespace and owned root to be private."""
    cursor = path.absolute()
    metadata = _output_lstat(
        cursor, operation="supplemental_output_owner_lstat",
    )
    while metadata is None:
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
        metadata = _output_lstat(
            cursor, operation="supplemental_output_owner_lstat",
        )
    if metadata is None or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
        metadata.st_mode
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_owner_invalid", str(cursor)
        )
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_owner_invalid",
            f"{cursor}: uid={metadata.st_uid}; mode={stat.S_IMODE(metadata.st_mode):o}",
        )


def _validate_immutable_owned_file(
    path: Path, *, expected_sha256: str, expected_size: int,
) -> None:
    payload, observed = _read_output_regular_nofollow(
        path, role="retained campaign input",
        operation="retained_campaign_input_validation",
    )
    if (
        observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o222
        or observed.st_size != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_retained_input_mismatch", str(path)
        )


def _materialize_attempt_external_input(
    *, source: Path, cell_dir: Path, expected_sha256: str,
    expected_size: int, input_kind: str,
) -> Path:
    """Freeze one reduced input inside the private capture attempt.

    The campaign-retained input is opened without following any path component,
    read once, and hash-checked.  The profiler never receives that source path:
    it receives this read-only, per-attempt copy instead.  Consequently a later
    replacement of the campaign input cannot change the bytes used by prepare,
    label, trace, or control subprocesses.  The normal cell artifact inventory
    seals the copy as ``sol_capture_evidence``.
    """
    if input_kind not in {"workload", "definition"}:
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_input_kind_invalid", input_kind
        )
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_input_identity_invalid",
            f"{input_kind}: sha256={expected_sha256!r}; "
            f"size={expected_size!r}",
        )
    descriptor, _absolute, observed = _open_regular_nofollow(
        source, role=f"campaign-retained reduced {input_kind}"
    )
    try:
        payload = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            payload.extend(block)
    finally:
        os.close(descriptor)
    if (
        observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o222
        or observed.st_size != expected_size
        or len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_input_source_drift",
            f"{input_kind}: {source}",
        )

    retained_dir = cell_dir / "sol_requests" / f"retained_{input_kind}"
    _mkdir_nofollow(retained_dir, exist_ok=False)
    suffix = ".jsonl" if input_kind == "workload" else ".json"
    target = retained_dir / f"{expected_sha256}{suffix}"
    try:
        _write_bytes_new_nofollow(
            target, payload, mode=0o444,
            operation=f"attempt_{input_kind}_publication",
        )
        target.chmod(0o444, follow_symlinks=False)
        retained_dir.chmod(0o555)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation=f"attempt_{input_kind}_mode_seal",
        )
        if mapped is not None:
            raise mapped from exc
        raise
    _validate_immutable_owned_file(
        target, expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    return target.absolute()


def _materialize_attempt_workload(
    *, source: Path, cell_dir: Path, expected_sha256: str,
    expected_size: int,
) -> Path:
    return _materialize_attempt_external_input(
        source=source, cell_dir=cell_dir,
        expected_sha256=expected_sha256, expected_size=expected_size,
        input_kind="workload",
    )


def _materialize_attempt_definition(
    *, source: Path, cell_dir: Path, expected_sha256: str,
    expected_size: int,
) -> Path:
    return _materialize_attempt_external_input(
        source=source, cell_dir=cell_dir,
        expected_sha256=expected_sha256, expected_size=expected_size,
        input_kind="definition",
    )


def supplemental_campaign_identity(
    *, matrix: dict[str, Any], matrix_path: Path,
    canonical_matrix: dict[str, Any], canonical_matrix_path: Path,
    authority_path: Path, pins_path: Path, aggregate_budget_path: Path,
    cell_budget_path: Path, registry_path: Path,
    full_coverage_inventory_path: Path,
    live_coverage_inventory_path: Path,
    runnability_path: Path, shrink_path: Path,
    external_sources: dict[str, Path] | None = None,
    owner_namespaces: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_commit = matrix["source_commit"]
    sources = _campaign_input_sources(
        matrix_path=matrix_path, canonical_matrix_path=canonical_matrix_path,
        authority_path=authority_path, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path,
        cell_budget_path=cell_budget_path, registry_path=registry_path,
        full_coverage_inventory_path=full_coverage_inventory_path,
        live_coverage_inventory_path=live_coverage_inventory_path,
        runnability_path=runnability_path, shrink_path=shrink_path,
        external_sources=external_sources,
    )
    identities = {
        role: input_file_identity(
            path, role=role, source_commit=source_commit,
        )
        for role, path in sources.items()
    }
    result = {
        "source_commit": source_commit,
        "supplemental_matrix_sha256": identities[
            "supplemental_matrix"
        ]["sha256"],
        "supplemental_matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "canonical_matrix_sha256": identities["canonical_matrix"]["sha256"],
        "canonical_matrix_seal_sha256": canonical_matrix[
            "matrix_seal_sha256"
        ],
        "authority_sha256": identities["authority"]["sha256"],
        "pins_sha256": identities["pins"]["sha256"],
        "aggregate_budget_sha256": identities["aggregate_budget"]["sha256"],
        "cell_budgets_sha256": identities["cell_budgets"]["sha256"],
        "metric_registry_sha256": identities["metric_registry"]["sha256"],
        "full_coverage_inventory_sha256": identities[
            "full_coverage_inventory"
        ]["sha256"],
        "live_coverage_inventory_sha256": identities[
            "live_coverage_inventory"
        ]["sha256"],
        "runnability_sha256": identities["runnability"]["sha256"],
        "shrink_provenance_sha256": identities[
            "shrink_provenance"
        ]["sha256"],
        "input_file_identities": identities,
    }
    if owner_namespaces is not None:
        if set(owner_namespaces) != {"plan", "runnability", "shrink"}:
            raise WorkloadCaptureRefusal(
                "supplemental_owner_namespace_set_invalid",
                repr(owner_namespaces),
            )
        for role, contract in owner_namespaces.items():
            _validate_owner_namespace_contract(contract, expected_role=role)
        result["owner_namespaces"] = copy.deepcopy(owner_namespaces)
    return result


def establish_supplemental_output_root(
    output_root: Path, *, campaign_identity: dict[str, Any],
) -> dict[str, Any]:
    """Create/verify the sealed dedicated root; reject canonical-root aliasing."""
    assert_external_output_root_before_mutation(output_root)
    root_metadata = _output_lstat(
        output_root, operation="supplemental_root_establish_lstat",
    )
    if root_metadata is not None:
        entries = {
            path.name for path in _output_directory_entries(
                output_root, operation="supplemental_root_establish",
            )
        }
        if entries and SUPPLEMENTAL_ROOT_MARKER not in entries:
            raise WorkloadCaptureRefusal(
                "supplemental_output_root_identity_missing", str(output_root)
            )
        if entries.intersection({"chunks", ".staging", ".capture_staging"}):
            raise WorkloadCaptureRefusal(
                "supplemental_output_root_aliases_canonical", str(output_root)
            )
    else:
        _mkdir_nofollow(output_root, exist_ok=False)
    _validate_owner_directory(output_root)
    marker_path = output_root / SUPPLEMENTAL_ROOT_MARKER
    marker_metadata = _output_lstat(
        marker_path, operation="supplemental_root_marker_lstat",
    )
    if marker_metadata is not None:
        if stat.S_ISLNK(marker_metadata.st_mode) or not stat.S_ISREG(
            marker_metadata.st_mode
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_output_root_identity_invalid", str(marker_path)
            )
        marker_payload, marker_metadata = _read_output_regular_nofollow(
            marker_path, role="supplemental root identity",
            operation="supplemental_root_marker_read",
        )
        _validate_immutable_owned_file(
            marker_path,
            expected_sha256=hashlib.sha256(marker_payload).hexdigest(),
            expected_size=marker_metadata.st_size,
        )
        marker = _load_output_object(
            marker_path, role="supplemental root identity",
            operation="supplemental_root_marker_json",
        )
        _verify_sealed(
            marker, "root_identity_sha256",
            code="supplemental_output_root_identity_invalid",
        )
        expected = {
            "schema_version": 1,
            "root_kind": SUPPLEMENTAL_ROOT_KIND,
            "campaign_identity": campaign_identity,
            "campaign_identity_sha256": canonical_sha256(campaign_identity),
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise WorkloadCaptureRefusal(
                "supplemental_output_root_identity_invalid", repr(marker)
            )
        return marker
    marker = _sealed({
        "schema_version": 1,
        "root_kind": SUPPLEMENTAL_ROOT_KIND,
        "campaign_identity": copy.deepcopy(campaign_identity),
        "campaign_identity_sha256": canonical_sha256(campaign_identity),
        "created_at_utc": _utc_text(),
    }, "root_identity_sha256")
    _write_json_new(marker_path, marker)
    try:
        marker_path.chmod(0o444)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="output_root_identity_mode_seal",
        )
        if mapped is not None:
            raise mapped from exc
        raise
    final_marker_metadata = _output_lstat(
        marker_path, operation="supplemental_root_marker_final_lstat",
    )
    if final_marker_metadata is None:
        raise WorkloadCaptureRefusal(
            "supplemental_output_root_identity_invalid", str(marker_path)
        )
    _validate_immutable_owned_file(
        marker_path,
        expected_sha256=_sha256_output_file(
            marker_path, role="supplemental root identity",
            operation="supplemental_root_marker_hash",
        ),
        expected_size=final_marker_metadata.st_size,
    )
    return marker


def _validate_schema(value: dict, schema_path: Path, *, code: str) -> None:
    try:
        import jsonschema
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(value, schema)
    except Exception as exc:
        raise WorkloadCaptureRefusal(code, str(exc)) from exc


def validate_live_coverage_inventory(
    *, inventory_path: Path,
    inventory_root: Path, matrix: dict[str, Any], matrix_path: Path,
    runnability_path: Path, runnability_root: Path,
    shrink_path: Path, shrink_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the current 3,957-row ledger to the exact live matrix rows."""
    try:
        owner = sol_recon.validate_external_campaign_root(inventory_root)
        expected_parent = Path(owner["canonical_path"])
        if (
            not inventory_path.is_absolute()
            or inventory_path.parent != expected_parent
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_live_coverage_owner_mismatch",
                str(inventory_path),
            )
        payload = sol_recon.read_external_child(owner, inventory_path.name)
    except WorkloadCaptureRefusal:
        raise
    except sol_recon.PrerequisiteContractError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_owner_invalid", str(exc)
        ) from exc
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key: {key!r}")
            parsed[key] = value
        return parsed

    try:
        document = json.loads(
            payload, object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_json_invalid", str(exc)
        ) from exc
    if not isinstance(document, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_json_invalid", "root is not an object"
        )
    if payload != live_coverage.canonical_json_bytes(document):
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_noncanonical", str(inventory_path)
        )
    _validate_schema(
        document, LIVE_COVERAGE_SCHEMA,
        code="supplemental_live_coverage_schema_invalid",
    )
    try:
        regenerated = live_coverage.build_live_inventory(
            repo_root=REPO_ROOT,
            matrix_path=matrix_path, matrix_root=inventory_root,
            runnability_path=runnability_path,
            runnability_root=runnability_root,
            shrink_path=shrink_path, shrink_root=shrink_root,
            inventory_name=inventory_path.name,
        )
    except (
        OSError, live_coverage.LiveCoverageError,
        workload_matrix.WorkloadMatrixError,
        sol_recon.PrerequisiteContractError,
    ) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_regeneration_failed", str(exc)
        ) from exc
    if payload != live_coverage.canonical_json_bytes(regenerated):
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_regeneration_mismatch",
            repr({
                "provided_sha256": hashlib.sha256(payload).hexdigest(),
                "regenerated_sha256": hashlib.sha256(
                    live_coverage.canonical_json_bytes(regenerated)
                ).hexdigest(),
            }),
        )
    matrix_sha256 = hashlib.sha256(
        workload_matrix.serialize_matrix(matrix)
    ).hexdigest()
    declared_matrix = (document.get("inputs") or {}).get("matrix") or {}
    if (
        document.get("source_commit") != matrix.get("source_commit")
        or declared_matrix.get("name") != matrix_path.name
        or declared_matrix.get("sha256") != matrix_sha256
        or (document.get("counts") or {}).get("matrix_schedule_rows")
        != len(matrix.get("rows") or [])
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_matrix_mismatch",
            repr({
                "source_commit": document.get("source_commit"),
                "matrix": declared_matrix,
            }),
        )

    matrix_keys = [
        _row_identity(row.get("key")) for row in matrix.get("rows") or []
    ]
    inventory_keys: list[tuple[str, str, str]] = []
    for row in document.get("rows") or []:
        for key in row.get("matrix_row_keys") or []:
            inventory_keys.append(_row_identity(key))
    if (
        len(matrix_keys) != len(set(matrix_keys))
        or len(inventory_keys) != len(set(inventory_keys))
        or set(inventory_keys) != set(matrix_keys)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_live_coverage_row_closure_mismatch",
            repr({
                "matrix_rows": len(matrix_keys),
                "inventory_rows": len(inventory_keys),
            }),
        )
    return document, {
        "path": str(inventory_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "owner_identity": copy.deepcopy(owner),
        "source_commit": document["source_commit"],
        "source_workload_rows": document["counts"]["source_workload_rows"],
        "matrix_schedule_rows": len(matrix_keys),
        "source_rows_with_any_schedule": document["counts"][
            "source_rows_with_any_schedule"
        ],
        "source_rows_unscheduled_in_scope": document["counts"][
            "source_rows_unscheduled_in_scope"
        ],
    }


def validate_supplemental_matrix(
    matrix: dict[str, Any], *, matrix_path: Path,
    canonical_matrix_path: Path, authority_path: Path,
    runnability_path: Path | None = None,
    shrink_path: Path | None = None,
    runnability_root: Path | None = None,
    shrink_root: Path | None = None,
) -> None:
    """Prove schema, seal, and byte regeneration before any output mutation."""
    _validate_schema(
        matrix, MATRIX_SCHEMA, code="supplemental_matrix_schema_invalid"
    )
    unsealed = {
        key: value for key, value in matrix.items()
        if key != "matrix_seal_sha256"
    }
    expected_seal = canonical_sha256(unsealed)
    if matrix.get("matrix_seal_sha256") != expected_seal:
        raise WorkloadCaptureRefusal(
            "supplemental_matrix_seal_mismatch",
            f"expected={expected_seal}; observed={matrix.get('matrix_seal_sha256')}",
        )
    canonical_pin = matrix.get("canonical_matrix") or {}
    authority_pin = matrix.get("canonical_authority") or {}
    try:
        declared_canonical = canonical_capture._resolve_repo_path(
            canonical_pin.get("path"), "supplemental canonical matrix"
        )
        declared_authority = canonical_capture._resolve_repo_path(
            authority_pin.get("path"), "supplemental canonical authority"
        )
        if declared_canonical != canonical_matrix_path.resolve():
            raise WorkloadCaptureRefusal(
                "supplemental_canonical_matrix_path_mismatch",
                f"sealed={declared_canonical}; cli={canonical_matrix_path.resolve()}",
            )
        if declared_authority != authority_path.resolve():
            raise WorkloadCaptureRefusal(
                "supplemental_authority_path_mismatch",
                f"sealed={declared_authority}; cli={authority_path.resolve()}",
            )
        regenerated = workload_matrix.build_matrix(
            REPO_ROOT,
            canonical_matrix_path=canonical_matrix_path,
            authority_path=authority_path,
            runnability_path=runnability_path,
            shrink_path=shrink_path,
            runnability_root=runnability_root,
            shrink_root=shrink_root,
        )
        expected_bytes = workload_matrix.serialize_matrix(regenerated)
        descriptor, _absolute, _stat = _open_regular_nofollow(
            matrix_path, role="supplemental matrix bytes"
        )
        with os.fdopen(descriptor, "rb") as stream:
            observed_bytes = stream.read()
    except WorkloadCaptureRefusal:
        raise
    except (OSError, workload_matrix.WorkloadMatrixError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_matrix_regeneration_failed", str(exc)
        ) from exc
    if observed_bytes != expected_bytes:
        raise WorkloadCaptureRefusal(
            "supplemental_matrix_regeneration_mismatch", str(matrix_path)
        )


def is_representative_execution_matrix(matrix: dict[str, Any]) -> bool:
    """Return true only for the explicit additive representative profile."""
    return (
        isinstance(matrix, dict)
        and matrix.get("artifact_id") == representative_execution.ARTIFACT_ID
        and matrix.get("execution_profile")
        == representative_execution.EXECUTION_PROFILE
    )


def validate_representative_execution_matrix(
    matrix: dict[str, Any], *, matrix_path: Path,
) -> None:
    """Validate the current-HEAD authority against both retained upstream files."""
    try:
        representative_execution.validate_execution_matrix(
            matrix, repo_root=REPO_ROOT, require_clean_head=True,
            enforce_current_counts=True,
            matrix_path=matrix_path,
        )
    except representative_execution.RepresentativeExecutionError as exc:
        raise WorkloadCaptureRefusal(
            "representative_execution_matrix_invalid", str(exc)
        ) from exc
    try:
        descriptor, _absolute, _stat = _open_regular_nofollow(
            matrix_path, role="representative execution matrix bytes"
        )
        with os.fdopen(descriptor, "rb") as stream:
            observed = stream.read()
    except OSError as exc:
        raise WorkloadCaptureRefusal(
            "representative_execution_matrix_unreadable", str(exc)
        ) from exc
    if observed != representative_execution.canonical_bytes(matrix):
        raise WorkloadCaptureRefusal(
            "representative_execution_matrix_noncanonical", str(matrix_path)
        )


def validate_representative_live_coverage_inventory(
    *, inventory_path: Path, inventory_root: Path,
    matrix: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the frozen 3,420-row inventory without equating it to 216 runs."""
    try:
        owner = sol_recon.validate_external_campaign_root(inventory_root)
        expected_parent = Path(owner["canonical_path"])
        if (
            not inventory_path.is_absolute()
            or inventory_path.parent.resolve(strict=True) != expected_parent
        ):
            raise WorkloadCaptureRefusal(
                "representative_live_coverage_inventory_invalid",
                str(inventory_path),
            )
        payload = sol_recon.read_external_child(owner, inventory_path.name)
        document = json.loads(payload)
    except WorkloadCaptureRefusal:
        raise
    except (
        OSError, UnicodeError, json.JSONDecodeError,
        sol_recon.PrerequisiteContractError,
    ) as exc:
        raise WorkloadCaptureRefusal(
            "representative_live_coverage_inventory_invalid", str(exc)
        ) from exc
    if not isinstance(document, dict):
        raise WorkloadCaptureRefusal(
            "representative_live_coverage_inventory_invalid", "not an object"
        )
    _validate_schema(
        document, LIVE_COVERAGE_SCHEMA,
        code="representative_live_coverage_inventory_invalid",
    )
    expected_pin = (
        matrix.get("authority_input_pins", {}).get("coverage_inventory") or {}
    )
    expected_source_matrix = (
        matrix.get("authority_input_pins", {}).get("source_matrix") or {}
    )
    observed_sha = hashlib.sha256(payload).hexdigest()
    if (
        payload != live_coverage.canonical_json_bytes(document)
        or expected_pin.get("sha256") != observed_sha
        or document.get("source_commit") != matrix.get("authority_source_commit")
        or expected_source_matrix.get("sha256")
        != matrix.get("source_matrix", {}).get("sha256")
        or document.get("inputs", {}).get("matrix", {}).get("sha256")
        != matrix.get("source_matrix", {}).get("sha256")
    ):
        raise WorkloadCaptureRefusal(
            "representative_live_coverage_inventory_invalid",
            repr({
                "sha256": observed_sha,
                "source_commit": document.get("source_commit"),
            }),
        )
    return document, {
        "path": str(inventory_path),
        "sha256": observed_sha,
        "owner_identity": copy.deepcopy(owner),
        "source_commit": document["source_commit"],
        "source_workload_rows": document["counts"]["source_workload_rows"],
        "matrix_schedule_rows": matrix["counts"]["rows"],
        "source_rows_with_any_schedule": document["counts"][
            "source_rows_with_any_schedule"
        ],
        "source_rows_unscheduled_in_scope": document["counts"][
            "source_rows_unscheduled_in_scope"
        ],
        "execution_selection_identity_sha256": matrix[
            "representative_selection"
        ]["selection_identity_sha256"],
    }


def validate_frozen_canonical_base(
    canonical_matrix: dict[str, Any], *, canonical_matrix_path: Path,
    authority_path: Path, supplemental_matrix: dict[str, Any],
) -> None:
    """Validate frozen protocol bytes without live authority rederivation."""
    _validate_schema(
        canonical_matrix, canonical_capture.MATRIX_SCHEMA,
        code="supplemental_canonical_matrix_schema_invalid",
    )
    unsealed = {
        key: value for key, value in canonical_matrix.items()
        if key != "matrix_seal_sha256"
    }
    expected_seal = workload_matrix.canonical_source_sha256(unsealed)
    if canonical_matrix.get("matrix_seal_sha256") != expected_seal:
        raise WorkloadCaptureRefusal(
            "supplemental_canonical_matrix_seal_mismatch",
            f"expected={expected_seal}; "
            f"observed={canonical_matrix.get('matrix_seal_sha256')}",
        )
    matrix_pin = supplemental_matrix.get("canonical_matrix") or {}
    authority_pin = supplemental_matrix.get("canonical_authority") or {}
    canonical_authority = canonical_matrix.get("authority") or {}
    try:
        sealed_matrix_path = canonical_capture._resolve_repo_path(
            matrix_pin.get("path"), "supplemental canonical matrix"
        )
        sealed_authority_path = canonical_capture._resolve_repo_path(
            authority_pin.get("path"), "supplemental canonical authority"
        )
        historical_authority_path = canonical_capture._resolve_repo_path(
            canonical_authority.get("path"), "historical canonical authority"
        )
    except canonical_capture.ChunkRefusal as exc:
        raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
    canonical_sha = sha256_file(canonical_matrix_path)
    authority_sha = sha256_file(authority_path)
    authority = _load_object(authority_path, role="canonical capture authority")
    if (
        sealed_matrix_path != canonical_matrix_path.resolve()
        or matrix_pin.get("sha256") != canonical_sha
        or matrix_pin.get("source_commit") != canonical_matrix.get("source_commit")
        or matrix_pin.get("matrix_seal_sha256") != expected_seal
        or sealed_authority_path != authority_path.resolve()
        or historical_authority_path != authority_path.resolve()
        or authority_pin.get("sha256") != authority_sha
        or canonical_authority.get("sha256") != authority_sha
        or authority_pin.get("roster_id") != authority.get("roster_id")
        or authority.get("source_commit") != canonical_matrix.get("source_commit")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_frozen_canonical_binding_mismatch",
            str(canonical_matrix_path),
        )


def _validate_loaded_canonical_pin(
    expected: dict[str, Any], canonical_matrix: dict[str, Any],
    canonical_matrix_path: Path,
) -> None:
    """Validate the loaded canonical file independent of process CWD."""
    try:
        declared_path = canonical_capture._resolve_repo_path(
            expected.get("path"), "supplemental canonical matrix",
        )
    except canonical_capture.ChunkRefusal as exc:
        raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
    if (
        declared_path != canonical_matrix_path
        or expected.get("sha256") != sha256_file(canonical_matrix_path)
        or expected.get("source_commit") != canonical_matrix.get("source_commit")
        or expected.get("matrix_seal_sha256")
        != canonical_matrix.get("matrix_seal_sha256")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_canonical_matrix_mismatch", str(canonical_matrix_path)
        )


def _row_identity(key: dict[str, Any]) -> tuple[str, str, str]:
    if (
        not isinstance(key, dict)
        or set(key) != {"slug", "source_workload_uuid", "variant_id"}
        or not all(isinstance(key.get(name), str) and key[name]
                   for name in key)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_row_key_invalid", repr(key)
        )
    return key["slug"], key["source_workload_uuid"], key["variant_id"]


def _variant_cell_contract(variant: dict[str, Any]) -> dict[str, Any]:
    cohort = variant.get("cohort")
    adapter_contract = variant.get("adapter_contract")
    flashinfer_corpus = (
        adapter_contract.get("flashinfer_corpus")
        if isinstance(adapter_contract, dict) else None
    )
    contract = {
        "identity": {
            "roster_source": "sol_execbench",
            "slug": variant.get("slug"),
        },
        "category": variant.get("category"),
        "flashinfer_corpus": copy.deepcopy(flashinfer_corpus),
        "launch_scope": variant.get("launch_scope"),
        "launch_completeness_contract": copy.deepcopy(
            variant.get("launch_completeness_contract")
        ),
        "invocation": copy.deepcopy(variant.get("invocation")),
        "trace_budget": copy.deepcopy(variant.get("trace_budget")),
        "control_bits": copy.deepcopy(variant.get("control_bits")),
    }
    if cohort == "partial_native_row":
        contract["selected_diagnostic_sha256"] = variant.get(
            "selected_diagnostic_sha256"
        )
    elif cohort == "reduced_proxy":
        provenance = variant.get("reduced_provenance")
        contract["row_plan_sha256"] = (
            provenance.get("row_plan_sha256")
            if isinstance(provenance, dict) else None
        )
    elif cohort != "native":
        raise WorkloadCaptureRefusal(
            "supplemental_variant_cohort_invalid", repr(cohort)
        )
    return contract


def _synthetic_cell(
    *, row: dict[str, Any], variant: dict[str, Any],
) -> dict[str, Any]:
    """Build the additive execution cell solely from sealed variant bytes."""
    sealed_variant = copy.deepcopy(variant)
    observed_variant_sha = sealed_variant.pop(
        "variant_contract_sha256", None
    )
    if (
        not isinstance(observed_variant_sha, str)
        or canonical_sha256(sealed_variant) != observed_variant_sha
        or row.get("variant_contract_sha256") != observed_variant_sha
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_variant_seal_mismatch", repr(row.get("key"))
        )
    contract = _variant_cell_contract(variant)
    supplemental_cell_sha = variant.get(
        "supplemental_cell_contract_sha256"
    )
    if (
        not isinstance(supplemental_cell_sha, str)
        or canonical_sha256(contract) != supplemental_cell_sha
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_cell_contract_mismatch", repr(row.get("key"))
        )
    evidence = variant.get("runnability_evidence")
    invocation = variant.get("invocation")
    logical = (
        invocation.get("logical_workload")
        if isinstance(invocation, dict) else None
    )
    if variant.get("cohort") in {"native", "partial_native_row"} and (
        not isinstance(evidence, dict)
        or not isinstance(logical, dict)
        or logical.get("evaluation_evidence") != evidence
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_runnability_contract_mismatch",
            repr(row.get("key")),
        )
    if variant.get("cohort") == "partial_native_row" and (
        variant.get("selected_diagnostic_sha256")
        != row.get("row_evidence_sha256")
        or evidence.get("selected_diagnostic_sha256")
        != row.get("row_evidence_sha256")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_partial_evidence_mismatch", repr(row.get("key"))
        )
    if variant.get("cohort") == "reduced_proxy":
        provenance = variant.get("reduced_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("row_plan_sha256")
            != row.get("row_evidence_sha256")
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_reduced_evidence_mismatch",
                repr(row.get("key")),
            )
    return {
        "identity": copy.deepcopy(contract["identity"]),
        "authority_cell_sha256": supplemental_cell_sha,
        "disposition": "runnable",
        "capture_readiness": {"status": "ready", "blocking_reasons": []},
        "category": contract["category"],
        "flashinfer_corpus": copy.deepcopy(contract["flashinfer_corpus"]),
        "launch_scope": contract["launch_scope"],
        "launch_completeness_contract": copy.deepcopy(
            contract["launch_completeness_contract"]
        ),
        "invocation": copy.deepcopy(contract["invocation"]),
        "trace_budget": copy.deepcopy(contract["trace_budget"]),
        "control_bits": copy.deepcopy(contract["control_bits"]),
    }


def select_workload_chunk(
    matrix: dict[str, Any], canonical_matrix: dict[str, Any], chunk_id: str,
) -> tuple[dict[str, Any], list[tuple[dict, dict, dict]]]:
    chunks = [
        row for row in matrix.get("chunks", [])
        if isinstance(row, dict) and row.get("chunk_id") == chunk_id
    ]
    if len(chunks) != 1:
        raise WorkloadCaptureRefusal(
            "supplemental_chunk_invalid",
            f"chunk_id={chunk_id!r} matched {len(chunks)} rows",
        )
    chunk = chunks[0]
    members = chunk.get("rows")
    if (
        not isinstance(members, list)
        or not members
        or chunk.get("row_count") != len(members)
        or chunk.get("membership_sha256") != canonical_sha256(members)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_chunk_membership_invalid", repr(chunk)
        )
    rows_by_key: dict[tuple[str, str, str], dict] = {}
    for row in matrix.get("rows") or []:
        if not isinstance(row, dict):
            raise WorkloadCaptureRefusal(
                "supplemental_row_invalid", repr(row)
            )
        key = _row_identity(row.get("key"))
        if key in rows_by_key:
            raise WorkloadCaptureRefusal(
                "supplemental_row_duplicate", repr(key)
            )
        rows_by_key[key] = row
    variants_by_hash: dict[str, dict] = {}
    for variant in matrix.get("variants") or []:
        if not isinstance(variant, dict):
            raise WorkloadCaptureRefusal(
                "supplemental_variant_invalid", repr(variant)
            )
        seal = variant.get("variant_contract_sha256")
        if seal in variants_by_hash:
            raise WorkloadCaptureRefusal(
                "supplemental_variant_duplicate", repr(seal)
            )
        variants_by_hash[seal] = variant
    # Deliberately do not look rows up in canonical runnable membership.  The
    # supplemental variant is the additive execution authority; the canonical
    # matrix supplies only campaign-wide protocol/preflight inputs.
    if not isinstance(canonical_matrix.get("cells"), list):
        raise WorkloadCaptureRefusal(
            "supplemental_canonical_matrix_invalid", "cells"
        )
    resolved: list[tuple[dict, dict, dict]] = []
    seen: set[tuple[str, str, str]] = set()
    for member in members:
        key = _row_identity(member)
        if key in seen:
            raise WorkloadCaptureRefusal(
                "supplemental_chunk_member_duplicate", repr(key)
            )
        seen.add(key)
        row = rows_by_key.get(key)
        if row is None:
            raise WorkloadCaptureRefusal(
                "supplemental_chunk_member_absent", repr(key)
            )
        variant = variants_by_hash.get(row.get("variant_contract_sha256"))
        if variant is None:
            raise WorkloadCaptureRefusal(
                "supplemental_row_contract_absent", repr(key)
            )
        if (
            variant.get("slug") != key[0]
            or variant.get("variant_id") != key[2]
            or variant.get("cohort") != row.get("cohort")
            or variant.get("category") != row.get("category")
            or (variant.get("adapter_contract") or {}).get(
                "flashinfer_corpus"
            )
            != row.get("flashinfer_corpus")
            or row.get("cohort") != chunk.get("cohort")
            or variant.get("launch_scope") != row.get("launch_scope")
            or variant.get("launch_completeness_contract")
            != row.get("launch_completeness_contract")
            or variant.get("claim_scope") != row.get("claim_scope")
            or variant.get("native_predictive")
            is not row.get("native_predictive")
            or variant.get("native_shape") is not row.get("native_shape")
            or variant.get("problem_level_runnable")
            is not row.get("problem_level_runnable")
            or row.get("source_workload_uuid") != key[1]
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_row_contract_mismatch", repr(key)
            )
        cell = _synthetic_cell(row=row, variant=variant)
        resolved.append((row, variant, cell))
    return chunk, resolved


def explicit_chunk_budgets(
    resolved: list[tuple[dict, dict, dict]],
) -> TraceBudgetsConfig:
    """Construct enforcement config from explicit, hash-bound row budgets."""
    by_slug: dict[str, CellBudget] = {}
    for row, variant, cell in resolved:
        budget = variant.get("trace_budget")
        if not isinstance(budget, dict):
            raise WorkloadCaptureRefusal(
                "supplemental_trace_budget_invalid", repr(row.get("key"))
            )
        policy_path = canonical_capture._resolve_repo_path(
            budget.get("policy_path"), "SOL supplemental budget policy"
        )
        source_path = canonical_capture._resolve_repo_path(
            budget.get("trace_budgets_path"), "SOL trace budgets source"
        )
        try:
            if (
                sha256_file(policy_path) != budget.get("policy_sha256")
                or sha256_file(source_path)
                != budget.get("trace_budgets_sha256")
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_trace_budget_pin_mismatch",
                    repr(row.get("key")),
                )
            policy = _load_object(
                policy_path, role="SOL supplemental budget policy"
            )
            source = load_trace_budgets(source_path)
        except WorkloadCaptureRefusal:
            raise
        except Exception as exc:
            raise WorkloadCaptureRefusal(
                "supplemental_trace_budget_invalid", str(exc)
            ) from exc
        slug = cell["identity"]["slug"]
        explicit = source.cells_by_id().get(slug)
        explicit_sha = budget.get("explicit_slug_budget_sha256")
        representative_budget = (
            budget.get("derivation")
            == "representative_min_source_and_900s_ceiling"
        )
        if (
            budget.get("status") != "declared"
            or (
                not representative_budget
                and budget.get("derivation")
                != "explicit_slug_or_policy_default_ceiling"
            )
            or budget.get("sampling_policy") != "off"
            or budget.get("breach_disposition")
            != "hard_refusal_preserve_complete_trace"
            or budget.get("partial_trace_allowed") is not False
            or type(budget.get("max_trace_bytes")) is not int
            or type(budget.get("max_wall_time_s")) is not int
            or (
                explicit_sha is None and explicit is not None
                or explicit_sha is not None and (
                    explicit is None
                    or canonical_sha256(explicit.to_dict()) != explicit_sha
                )
            )
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_trace_budget_invalid", repr(row.get("key"))
            )
        expected_bytes = (
            explicit.max_trace_bytes
            if explicit is not None
            else policy.get("default_max_trace_bytes", 0)
        )
        expected_wall = (
            explicit.max_wall_time_s
            if explicit is not None
            else policy.get("default_max_wall_time_s", 0)
        )
        effective_wall = (
            min(expected_wall, representative_execution.REPRESENTATIVE_MAX_WALL_TIME_S)
            if representative_budget else expected_wall
        )
        effective_bytes = (
            min(
                expected_bytes,
                representative_execution.REPRESENTATIVE_MAX_TRACE_BYTES,
            )
            if representative_budget else expected_bytes
        )
        if (
            budget["max_trace_bytes"] != effective_bytes
            or budget["max_wall_time_s"] != effective_wall
            or (
                representative_budget and (
                    budget.get("source_max_wall_time_s") != expected_wall
                    or budget.get("source_max_trace_bytes") != expected_bytes
                    or budget.get("timeout_disposition")
                    != "typed_measured_infeasibility_for_trace_proxy_planning"
                    or budget.get("trace_limit_disposition")
                    != "typed_measured_infeasibility_for_trace_proxy_planning"
                )
            )
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_trace_budget_derivation_mismatch",
                repr(row.get("key")),
            )
        declared = CellBudget(
            cell_id=slug,
            max_trace_bytes=budget["max_trace_bytes"],
            max_wall_time_s=budget["max_wall_time_s"],
            sampling_policy="off",
        )
        prior = by_slug.get(slug)
        if prior is not None and prior != declared:
            raise WorkloadCaptureRefusal(
                "supplemental_trace_budget_conflict", slug
            )
        by_slug[slug] = declared
    return TraceBudgetsConfig(
        schema_version=1,
        defaults=None,
        cells=tuple(by_slug[slug] for slug in sorted(by_slug)),
    )


def verify_flashinfer_contracts(
    resolved: list[tuple[dict, dict, dict]],
) -> dict[str, Any]:
    """Rehash the complete preserved corpus and bind category-aware use."""
    try:
        proof = sol_recon.verify_flashinfer_corpus()
    except Exception as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_flashinfer_corpus_invalid", str(exc)
        ) from exc
    expected_root = (REPO_ROOT / "data/annex/flashinfer_trace").resolve()
    expected_manifest = "evidence/manifests/data_annex.sha256"
    if (
        Path(str(proof.get("root"))).resolve() != expected_root
        or proof.get("data_annex_manifest_sha256")
        != sol_recon.DATA_ANNEX_MANIFEST_SHA256
        or proof.get("filtered_manifest_sha256")
        != sol_recon.FLASHINFER_CORPUS_MANIFEST_SHA256
        or proof.get("file_count") != sol_recon.FLASHINFER_CORPUS_FILE_COUNT
        or proof.get("safetensors_count")
        != sol_recon.FLASHINFER_CORPUS_BLOB_COUNT
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_flashinfer_corpus_invalid", repr(proof)
        )
    for row, variant, cell in resolved:
        category = variant.get("category")
        if category not in sol_recon.SUBSET_MAP:
            raise WorkloadCaptureRefusal(
                "supplemental_flashinfer_contract_invalid", repr(category)
            )
        adapter_contract = variant.get("adapter_contract")
        variant_flashinfer_corpus = (
            adapter_contract.get("flashinfer_corpus")
            if isinstance(adapter_contract, dict)
            else None
        )
        expected = {
            "policy": "verified_fib_only",
            "category": category,
            "environment_variable": "FLASHINFER_TRACE_DIR",
            "injection_policy": (
                "verified_absolute_root" if category == "FIB"
                else "forbidden_remove_ambient"
            ),
            "root": "data/annex/flashinfer_trace" if category == "FIB" else None,
            "data_annex_manifest_path": expected_manifest,
            "data_annex_manifest_sha256": proof[
                "data_annex_manifest_sha256"
            ],
            "filtered_manifest_sha256": proof[
                "filtered_manifest_sha256"
            ],
            "file_count": proof["file_count"],
            "safetensors_count": proof["safetensors_count"],
        }
        if (
            row.get("category") != category
            or cell.get("category") != category
            or row.get("flashinfer_corpus") != expected
            or variant_flashinfer_corpus != expected
            or cell.get("flashinfer_corpus") != expected
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_flashinfer_contract_invalid",
                repr(row.get("key")),
            )
    return proof


def certified_row_environment(
    *, cell: dict[str, Any], certified_sol_env: dict[str, str],
    flashinfer_proof: dict[str, Any],
) -> dict[str, str]:
    """Strip ambient corpus state, then inject only the verified FIB root."""
    environment = dict(certified_sol_env)
    environment.pop("FLASHINFER_TRACE_DIR", None)
    category = cell.get("category")
    contract = cell.get("flashinfer_corpus")
    if category not in sol_recon.SUBSET_MAP or not isinstance(contract, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_flashinfer_contract_invalid", repr(category)
        )
    if category == "FIB":
        root = Path(str(flashinfer_proof.get("root"))).resolve()
        expected = (REPO_ROOT / str(contract.get("root"))).resolve()
        if (
            contract.get("injection_policy") != "verified_absolute_root"
            or root != expected or root.is_symlink() or not root.is_dir()
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_flashinfer_contract_invalid", repr(contract)
            )
        environment["FLASHINFER_TRACE_DIR"] = str(root)
    elif (
        contract.get("injection_policy") != "forbidden_remove_ambient"
        or contract.get("root") is not None
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_flashinfer_contract_invalid", repr(contract)
        )
    return environment


def flashinfer_done_proof(proof: dict[str, Any]) -> dict[str, Any]:
    document = {
        "root_repo_path": "data/annex/flashinfer_trace",
        "data_annex_manifest_path": "evidence/manifests/data_annex.sha256",
        "data_annex_manifest_sha256": proof[
            "data_annex_manifest_sha256"
        ],
        "filtered_manifest_sha256": proof["filtered_manifest_sha256"],
        "file_count": proof["file_count"],
        "safetensors_count": proof["safetensors_count"],
    }
    document["corpus_identity_sha256"] = canonical_sha256(document)
    return document


def prevalidate_supplemental_rows(
    *, resolved: list[tuple[dict, dict, dict]],
    bindings: dict[str, dict[str, Any]], cell_budgets: TraceBudgetsConfig,
    certified_sol_env: dict[str, str],
    flashinfer_proof: dict[str, Any],
    runnability_path: Path | None = None,
    reduced_workload_paths: dict[str, Path] | None = None,
    reduced_definition_paths: dict[str, Path] | None = None,
) -> None:
    """Revalidate every row contract before creating capture state."""
    for row, variant, cell in resolved:
        binding = bindings[_safe_name(row["key"])]
        try:
            canonical_capture.validate_capture_cell(
                cell,
                cell_budgets=cell_budgets,
                certified_sol_env=certified_row_environment(
                    cell=cell, certified_sol_env=certified_sol_env,
                    flashinfer_proof=flashinfer_proof,
                ),
                workload_row_binding=binding,
                supplemental_runnability_evidence=(
                    variant.get("runnability_evidence")
                    if variant.get("cohort") in {
                        "native", "partial_native_row"
                    } else None
                ),
                runnability_path=runnability_path,
                supplemental_workload_path=(
                    (reduced_workload_paths or {}).get(
                        variant.get("variant_contract_sha256")
                    )
                ),
                supplemental_definition_path=(
                    (reduced_definition_paths or {}).get(
                        variant.get("variant_contract_sha256")
                    )
                ),
            )
        except canonical_capture.ChunkRefusal as exc:
            raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc


def workload_row_binding(
    *, matrix: dict[str, Any], matrix_sha256: str, chunk_id: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    binding = {
        "schema_version": 1,
        "supplemental_matrix_sha256": matrix_sha256,
        "supplemental_matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "chunk_id": chunk_id,
        "row_key": copy.deepcopy(row["key"]),
        "cohort": row["cohort"],
        "category": row["category"],
        "flashinfer_corpus": copy.deepcopy(row["flashinfer_corpus"]),
        "launch_scope": row["launch_scope"],
        "launch_completeness_contract": copy.deepcopy(
            row["launch_completeness_contract"]
        ),
        "claim_scope": row["claim_scope"],
        "native_shape": row["native_shape"],
        "native_predictive": row["native_predictive"],
        "problem_level_runnable": row["problem_level_runnable"],
        "variant_contract_sha256": row["variant_contract_sha256"],
        "row_evidence_sha256": row["row_evidence_sha256"],
        "source_workload_index": row["source_workload_index"],
        "source_workload_uuid": row["source_workload_uuid"],
        "source_axes": copy.deepcopy(row["source_axes"]),
        "source_row_sha256": row["source_row_sha256"],
        "capture_workload_index": row["capture_workload_index"],
        "capture_axes": copy.deepcopy(row["capture_axes"]),
        "capture_row_sha256": row["capture_row_sha256"],
    }
    if is_representative_execution_matrix(matrix):
        binding["execution_profile"] = representative_execution.EXECUTION_PROFILE
    return binding


def _safe_name(key: dict[str, Any]) -> str:
    slug, _workload_uuid, _variant_id = _row_identity(key)
    if SAFE_SLUG_RE.fullmatch(slug) is None:
        raise WorkloadCaptureRefusal(
            "supplemental_slug_unsafe", repr(slug)
        )
    return f"solrow-{slug[:48]}-{canonical_sha256(key)}"


def _utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    try:
        _write_bytes_new_nofollow(
            path, _canonical_json_bytes(value), mode=0o600,
            operation="output_json_publication",
        )
    except FileExistsError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_output_exists", str(path)
        ) from exc


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    out = copy.deepcopy(value)
    out[field] = canonical_sha256(out)
    return out


def _verify_sealed(
    value: dict[str, Any], field: str, *, code: str,
) -> None:
    seal = value.get(field)
    unsealed = {key: item for key, item in value.items() if key != field}
    if not isinstance(seal, str) or seal != canonical_sha256(unsealed):
        raise WorkloadCaptureRefusal(code, repr(seal))


def _journal_start(
    *, binding: dict[str, Any], canonical_matrix_sha256: str,
    canonical_matrix_seal_sha256: str, preflight_relative_path: str,
    preflight_sha256: str, source_commit: str, admission_run_id: str,
    attempt_ordinal: int, attempt_id: str,
    prior_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return _sealed({
        "schema_version": 1,
        "kind": "sol_workload_capture_journal_start",
        "attempt_ordinal": attempt_ordinal,
        "attempt_id": attempt_id,
        "prior_attempts": copy.deepcopy(prior_attempts),
        "prior_attempts_sha256": canonical_sha256(prior_attempts),
        "binding": copy.deepcopy(binding),
        "binding_sha256": canonical_sha256(binding),
        "canonical_matrix_sha256": canonical_matrix_sha256,
        "canonical_matrix_seal_sha256": canonical_matrix_seal_sha256,
        "capture_preflight_path": preflight_relative_path,
        "capture_preflight_sha256": preflight_sha256,
        "source_commit": source_commit,
        "capture_started_at_utc": _utc_text(),
        "admission_run_id": admission_run_id,
    }, "journal_start_seal_sha256")


def _attempt_tree_inventory(attempt_dir: Path) -> dict[str, Any]:
    """Hash every preserved attempt byte without following aliases."""
    rows: list[dict[str, Any]] = []
    attempt_metadata = _output_lstat(
        attempt_dir, operation="attempt_inventory_root_lstat",
    )
    if attempt_metadata is None or stat.S_ISLNK(
        attempt_metadata.st_mode
    ) or not stat.S_ISDIR(attempt_metadata.st_mode):
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_path_invalid", str(attempt_dir)
        )
    for path in _output_tree_paths(
        attempt_dir, operation="attempt_inventory_tree_walk",
    ):
        observed = _output_lstat(
            path, operation="attempt_inventory_entry_lstat",
        )
        if observed is None:
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_path_invalid", str(path)
            )
        if stat.S_ISLNK(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_symlink_refused", str(path)
            )
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_path_invalid", str(path)
            )
        relative = path.relative_to(attempt_dir).as_posix()
        rows.append({
            "relative_path": relative,
            "sha256": _sha256_output_file(
                path, role="supplemental attempt artifact",
                operation="attempt_inventory_artifact_hash",
            ),
            "size_bytes": observed.st_size,
        })
    return {
        "file_count": len(rows),
        "logical_bytes": sum(row["size_bytes"] for row in rows),
        "tree_inventory_sha256": canonical_sha256(rows),
    }


def _make_attempt_summary(
    *, attempt_dir: Path, attempt_ordinal: int,
    start: dict[str, Any] | None, state: str,
    terminal_seal_sha256: str | None,
) -> dict[str, Any]:
    inventory = _attempt_tree_inventory(attempt_dir)
    summary = {
        "attempt_ordinal": attempt_ordinal,
        "attempt_id": start.get("attempt_id") if start is not None else None,
        "state": state,
        "journal_start_seal_sha256": (
            start.get("journal_start_seal_sha256")
            if start is not None else None
        ),
        "capture_preflight_path": (
            start.get("capture_preflight_path")
            if start is not None else None
        ),
        "capture_preflight_sha256": (
            start.get("capture_preflight_sha256")
            if start is not None else None
        ),
        "terminal_seal_sha256": terminal_seal_sha256,
        **inventory,
    }
    summary["attempt_summary_sha256"] = canonical_sha256(summary)
    return summary


def _seal_attempt_tree_untyped(attempt_dir: Path) -> None:
    """Make a retired attempt read-only after all of its bytes are durable."""
    paths = sorted(
        _output_tree_paths(
            attempt_dir, operation="attempt_mode_seal_tree_walk",
        ),
        key=lambda path: len(path.parts), reverse=True,
    )
    for path in paths:
        observed = _output_lstat(
            path, operation="attempt_mode_seal_entry_lstat",
        )
        if observed is None:
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_path_invalid", str(path)
            )
        if stat.S_ISLNK(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_symlink_refused", str(path)
            )
        if stat.S_ISREG(observed.st_mode):
            path.chmod(0o444)
        elif stat.S_ISDIR(observed.st_mode):
            path.chmod(0o555)
    attempt_dir.chmod(0o555)


def _seal_attempt_tree(attempt_dir: Path) -> None:
    try:
        _seal_attempt_tree_untyped(attempt_dir)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="attempt_tree_mode_seal",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _attempt_directories(journal_dir: Path) -> list[tuple[int, Path]]:
    journal_metadata = _output_lstat(
        journal_dir, operation="journal_directory_lstat",
    )
    if journal_metadata is None:
        return []
    if stat.S_ISLNK(journal_metadata.st_mode) or not stat.S_ISDIR(
        journal_metadata.st_mode
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_journal_path_invalid", str(journal_dir)
        )
    attempts_root = journal_dir / "attempts"
    attempts_metadata = _output_lstat(
        attempts_root, operation="journal_attempts_root_lstat",
    )
    if attempts_metadata is None:
        return []
    if stat.S_ISLNK(attempts_metadata.st_mode) or not stat.S_ISDIR(
        attempts_metadata.st_mode
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_path_invalid", str(attempts_root)
        )
    rows: list[tuple[int, Path]] = []
    for path in _output_directory_entries(
        attempts_root, operation="journal_attempts_directory",
    ):
        match = re.fullmatch(r"attempt-([0-9]{6})", path.name)
        observed = _output_lstat(
            path, operation="journal_attempt_directory_lstat",
        )
        if match is None or observed is None or stat.S_ISLNK(
            observed.st_mode
        ) or not stat.S_ISDIR(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_path_invalid", str(path)
            )
        rows.append((int(match.group(1)), path))
    ordinals = [ordinal for ordinal, _path in rows]
    if ordinals != list(range(1, len(rows) + 1)):
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_sequence_invalid", repr(ordinals)
        )
    return rows


def _scan_journal_attempts(
    *, journal_dir: Path, expected_binding: dict[str, Any],
    canonical_matrix_sha256: str,
    canonical_matrix_seal_sha256: str,
    preflight_sha256: str, source_commit: str,
) -> dict[str, Any]:
    """Validate the immutable attempt chain and recover its latest success."""
    summaries: list[dict[str, Any]] = []
    completed: tuple[dict[str, Any], dict[str, Any], Path] | None = None
    terminal_failure: dict[str, Any] | None = None
    attempt_directories = _attempt_directories(journal_dir)
    for ordinal, attempt_dir in attempt_directories:
        start_path = attempt_dir / "journal_start.json"
        start: dict[str, Any] | None = None
        start_metadata = _output_lstat(
            start_path, operation="journal_start_lstat",
        )
        if start_metadata is not None and stat.S_ISREG(
            start_metadata.st_mode
        ) and not stat.S_ISLNK(start_metadata.st_mode):
            start = _load_output_object(
                start_path, role="supplemental journal start",
                operation="journal_start_read",
            )
            _verify_sealed(
                start, "journal_start_seal_sha256",
                code="supplemental_journal_start_seal_mismatch",
            )
            expected = {
                "attempt_ordinal": ordinal,
                "prior_attempts": summaries,
                "prior_attempts_sha256": canonical_sha256(summaries),
                "binding": expected_binding,
                "binding_sha256": canonical_sha256(expected_binding),
                "canonical_matrix_sha256": canonical_matrix_sha256,
                "canonical_matrix_seal_sha256": (
                    canonical_matrix_seal_sha256
                ),
                "source_commit": source_commit,
            }
            mismatches = {
                key: {"expected": value, "observed": start.get(key)}
                for key, value in expected.items()
                if start.get(key) != value
            }
            if (
                not SHA256_RE.fullmatch(str(start.get("attempt_id", "")))
                or mismatches
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_journal_input_mismatch",
                    json.dumps(mismatches, sort_keys=True),
                )
            raw_preflight_path = start.get("capture_preflight_path")
            relative_preflight = (
                Path(raw_preflight_path)
                if isinstance(raw_preflight_path, str) else Path(".")
            )
            bound_preflight = attempt_dir / relative_preflight
            preflight_metadata = _output_lstat(
                bound_preflight, operation="journal_preflight_lstat",
            )
            if (
                relative_preflight != Path("capture_preflight.json")
                or relative_preflight.is_absolute()
                or ".." in relative_preflight.parts
                or preflight_metadata is None
                or stat.S_ISLNK(preflight_metadata.st_mode)
                or not stat.S_ISREG(preflight_metadata.st_mode)
                or not SHA256_RE.fullmatch(
                    str(start.get("capture_preflight_sha256", ""))
                )
                or _sha256_output_file(
                    bound_preflight, role="supplemental journal preflight",
                    operation="journal_preflight_hash",
                )
                != start["capture_preflight_sha256"]
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_attempt_preflight_invalid",
                    str(raw_preflight_path),
                )
        elif _output_directory_entries(
            attempt_dir, operation="journal_interrupted_attempt_entries",
        ):
            # A hard kill can occur after mkdir and before the exclusive start
            # write.  Preserve and bind those bytes, but never treat them as a
            # valid or reusable attempt.
            start = None

        complete_path = attempt_dir / "journal_complete.json"
        failure_path = attempt_dir / "journal_failure.json"
        complete_metadata = _output_lstat(
            complete_path, operation="journal_complete_lstat",
        )
        failure_metadata = _output_lstat(
            failure_path, operation="journal_failure_lstat",
        )
        if complete_metadata is not None and failure_metadata is not None:
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_terminal_conflict", str(attempt_dir)
            )
        terminal_seal: str | None = None
        if complete_metadata is not None and stat.S_ISREG(
            complete_metadata.st_mode
        ) and not stat.S_ISLNK(complete_metadata.st_mode):
            if start is None or completed is not None:
                raise WorkloadCaptureRefusal(
                    "supplemental_journal_result_mismatch", str(attempt_dir)
                )
            complete = _load_output_object(
                complete_path, role="supplemental journal complete",
                operation="journal_complete_read",
            )
            _verify_sealed(
                complete, "journal_complete_seal_sha256",
                code="supplemental_journal_complete_seal_mismatch",
            )
            result = complete.get("cell_result")
            if (
                complete.get("journal_start_seal_sha256")
                != start["journal_start_seal_sha256"]
                or complete.get("attempt_id") != start["attempt_id"]
                or not isinstance(result, dict)
                or result.get("workload_row_binding") != expected_binding
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_journal_result_mismatch", str(attempt_dir)
                )
            try:
                canonical_capture._verify_journal_artifacts(
                    result, root=attempt_dir,
                    safe_name=_safe_name(expected_binding["row_key"]),
                )
            except canonical_capture.ChunkRefusal as exc:
                if exc.code.endswith("_storage_failure"):
                    mapped = _output_storage_refusal(
                        exc, operation="journal_artifact_verification",
                    )
                    if mapped is not None:
                        raise mapped from exc
                raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
            terminal_seal = complete["journal_complete_seal_sha256"]
            state = "complete"
            completed = (start, result, attempt_dir)
        elif failure_metadata is not None and stat.S_ISREG(
            failure_metadata.st_mode
        ) and not stat.S_ISLNK(failure_metadata.st_mode):
            if start is None:
                raise WorkloadCaptureRefusal(
                    "supplemental_attempt_failure_invalid", str(attempt_dir)
                )
            failure = _load_output_object(
                failure_path, role="supplemental journal failure",
                operation="journal_failure_read",
            )
            _verify_sealed(
                failure, "journal_failure_seal_sha256",
                code="supplemental_attempt_failure_seal_mismatch",
            )
            if (
                failure.get("journal_start_seal_sha256")
                != start["journal_start_seal_sha256"]
                or failure.get("attempt_id") != start["attempt_id"]
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_attempt_failure_invalid", str(attempt_dir)
                )
            terminal_seal = failure["journal_failure_seal_sha256"]
            state = "failed"
            terminal_failure = failure
        else:
            state = "interrupted" if start is not None else "interrupted_before_start"
        summaries.append(_make_attempt_summary(
            attempt_dir=attempt_dir, attempt_ordinal=ordinal,
            start=start, state=state, terminal_seal_sha256=terminal_seal,
        ))
        if completed is not None and ordinal != len(attempt_directories):
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_after_completion", str(attempt_dir)
            )
    return {
        "attempts": summaries,
        "completed": completed,
        "terminal_failure": terminal_failure,
        "next_ordinal": len(summaries) + 1,
    }


def _load_completed_journal(
    *, journal_dir: Path, expected_binding: dict[str, Any],
    canonical_matrix_sha256: str, canonical_matrix_seal_sha256: str,
    preflight_sha256: str, source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    state = _scan_journal_attempts(
        journal_dir=journal_dir, expected_binding=expected_binding,
        canonical_matrix_sha256=canonical_matrix_sha256,
        canonical_matrix_seal_sha256=canonical_matrix_seal_sha256,
        preflight_sha256=preflight_sha256, source_commit=source_commit,
    )
    completed = state["completed"]
    if completed is None:
        return None
    return completed[0], completed[1]


def _tree_usage(root: Path) -> dict[str, int]:
    logical = 0
    physical = 0
    file_count = 0
    inodes: set[tuple[int, int]] = set()
    root_metadata = _output_lstat(root, operation="output_tree_root_lstat")
    if root_metadata is None:
        return {
            "logical_bytes": 0, "physical_bytes": 0,
            "file_count": 0, "unique_inode_count": 0,
        }
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_path_invalid", str(root)
        )
    for path in _output_tree_paths(root, operation="output_tree_usage_walk"):
        observed = _output_lstat(path, operation="output_tree_usage_lstat")
        if observed is None:
            raise WorkloadCaptureRefusal(
                "supplemental_output_path_invalid", str(path)
            )
        if stat.S_ISLNK(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_symlink_refused", str(path)
            )
        if stat.S_ISREG(observed.st_mode):
            logical += observed.st_size
            file_count += 1
            identity = (observed.st_dev, observed.st_ino)
            if identity not in inodes:
                inodes.add(identity)
                physical += observed.st_size
    return {
        "logical_bytes": logical,
        "physical_bytes": physical,
        "file_count": file_count,
        "unique_inode_count": len(inodes),
    }


def _tree_bytes(root: Path) -> int:
    """Return physical retained bytes; hard-linked content counts once."""
    return _tree_usage(root)["physical_bytes"]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="output_directory_fsync",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem directory without replacement."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_noreplace_unavailable", str(destination)
        )
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1
    ) == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_destination_exists", str(destination)
        )
    if error == errno.EXDEV:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_cross_device", str(destination)
        )
    exc = OSError(error, os.strerror(error), str(destination))
    mapped = storage_refusal_from_exception(
        exc, operation="output_directory_rename_noreplace",
    )
    if mapped is not None:
        raise mapped from exc
    raise exc


def _content_address_tree_untyped(*, source: Path, object_store: Path) -> None:
    """Deduplicate completed supplemental bytes without losing any path."""
    source_metadata = _output_lstat(
        source, operation="content_address_source_root_lstat",
    )
    if (
        source_metadata is None
        or stat.S_ISLNK(source_metadata.st_mode)
        or not stat.S_ISDIR(source_metadata.st_mode)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_path_invalid", str(source)
        )
    _mkdir_nofollow(object_store, exist_ok=True)
    store_metadata = _output_lstat(
        object_store, operation="content_address_store_root_lstat",
    )
    if (
        store_metadata is None
        or stat.S_ISLNK(store_metadata.st_mode)
        or not stat.S_ISDIR(store_metadata.st_mode)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_output_path_invalid", str(object_store)
        )
    if source_metadata.st_dev != store_metadata.st_dev:
        raise WorkloadCaptureRefusal(
            "supplemental_object_store_cross_device", str(source)
        )
    for path in _output_tree_paths(
        source, operation="content_address_source_walk",
    ):
        path_metadata = _output_lstat(
            path, operation="content_address_source_lstat",
        )
        if path_metadata is None:
            raise WorkloadCaptureRefusal(
                "supplemental_output_path_invalid", str(path)
            )
        if stat.S_ISLNK(path_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_symlink_refused", str(path)
            )
        if stat.S_ISDIR(path_metadata.st_mode):
            continue
        if not stat.S_ISREG(path_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_path_invalid", str(path)
            )
        digest = _sha256_output_file(
            path, role="content-address source artifact",
            operation="content_address_source_hash",
        )
        target_dir = object_store / digest[:2]
        _mkdir_nofollow(target_dir, exist_ok=True)
        target = target_dir / digest
        target_metadata = _output_lstat(
            target, operation="content_address_target_lstat",
        )
        if target_metadata is not None:
            if (
                stat.S_ISLNK(target_metadata.st_mode)
                or not stat.S_ISREG(target_metadata.st_mode)
                or target_metadata.st_size != path_metadata.st_size
                or _sha256_output_file(
                    target, role="content-address store artifact",
                    operation="content_address_target_hash",
                ) != digest
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_object_store_collision", str(target)
                )
            if (path_metadata.st_dev, path_metadata.st_ino) != (
                target_metadata.st_dev, target_metadata.st_ino
            ):
                replacement = path.with_name(
                    f".{path.name}.dedup-{uuid.uuid4().hex}"
                )
                os.link(target, replacement, follow_symlinks=False)
                if _sha256_output_file(
                    replacement, role="content-address replacement artifact",
                    operation="content_address_replacement_hash",
                ) != digest:
                    replacement.unlink()
                    raise WorkloadCaptureRefusal(
                        "supplemental_object_store_link_mismatch", str(path)
                    )
                os.replace(replacement, path)
                _fsync_directory(path.parent)
        else:
            try:
                os.link(path, target, follow_symlinks=False)
            except FileExistsError:
                target_metadata = _output_lstat(
                    target, operation="content_address_raced_target_lstat",
                )
                if (
                    target_metadata is None
                    or stat.S_ISLNK(target_metadata.st_mode)
                    or not stat.S_ISREG(target_metadata.st_mode)
                    or target_metadata.st_size != path_metadata.st_size
                    or _sha256_output_file(
                        target, role="content-address raced store artifact",
                        operation="content_address_raced_target_hash",
                    ) != digest
                ):
                    raise WorkloadCaptureRefusal(
                        "supplemental_object_store_collision", str(target)
                    )
            _fsync_directory(target_dir)
        path_metadata = _output_lstat(
            path, operation="content_address_linked_source_lstat",
        )
        target_metadata = _output_lstat(
            target, operation="content_address_linked_target_lstat",
        )
        if (
            path_metadata is None
            or target_metadata is None
            or not stat.S_ISREG(path_metadata.st_mode)
            or not stat.S_ISREG(target_metadata.st_mode)
            or (path_metadata.st_dev, path_metadata.st_ino) != (
                target_metadata.st_dev, target_metadata.st_ino
            )
            or _sha256_output_file(
                path, role="content-address linked source artifact",
                operation="content_address_linked_source_hash",
            ) != digest
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_object_store_link_mismatch", str(path)
            )
        path.chmod(0o444)
        _fsync_output_regular_nofollow(
            path, role="content-address linked source artifact",
            operation="content_address_linked_source_sync",
        )


def _content_address_tree(*, source: Path, object_store: Path) -> None:
    try:
        _content_address_tree_untyped(source=source, object_store=object_store)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="output_content_addressing",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _retire_failed_transient_raw_untyped(
    *, attempt_dir: Path, cell_dir: Path,
) -> dict[str, Any] | None:
    """Inventory then remove only the exact transient raw trace directory."""
    raw = cell_dir / "trace_full_raw"
    raw_metadata = _output_lstat(
        raw, operation="transient_retirement_root_lstat",
    )
    if raw_metadata is None:
        return None
    if (
        stat.S_ISLNK(raw_metadata.st_mode)
        or not stat.S_ISDIR(raw_metadata.st_mode)
        or raw.parent != cell_dir
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_failure_raw_retirement_refused", str(raw)
        )
    files: list[dict[str, Any]] = []
    tree_paths = _output_tree_paths(
        raw, operation="transient_retirement_walk",
    )
    path_metadata: dict[Path, os.stat_result] = {}
    for path in tree_paths:
        observed = _output_lstat(
            path, operation="transient_retirement_entry_lstat",
        )
        if observed is None:
            raise WorkloadCaptureRefusal(
                "supplemental_failure_raw_retirement_refused", str(path)
            )
        path_metadata[path] = observed
        if stat.S_ISLNK(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_failure_raw_retirement_refused", str(path)
            )
        if stat.S_ISREG(observed.st_mode):
            files.append({
                "relative_path": path.relative_to(raw).as_posix(),
                "sha256": _sha256_output_file(
                    path, role="transient raw artifact",
                    operation="transient_retirement_artifact_hash",
                ),
                "size_bytes": observed.st_size,
            })
        elif not stat.S_ISDIR(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_failure_raw_retirement_refused", str(path)
            )
    document = _sealed({
        "schema_version": 1,
        "kind": "sol_workload_capture_transient_raw_retirement",
        "raw_relative_path": raw.relative_to(attempt_dir).as_posix(),
        "preclean_attempt_inventory": _attempt_tree_inventory(attempt_dir),
        "removed_file_count": len(files),
        "removed_bytes": sum(row["size_bytes"] for row in files),
        "removed_files_sha256": canonical_sha256(files),
        "removed_files": files,
        "retired_at_utc": _utc_text(),
    }, "retirement_seal_sha256")
    record_path = attempt_dir / "transient_raw_retirement.json"
    _write_json_new(record_path, document)
    for path in sorted(
        tree_paths, key=lambda item: len(item.parts), reverse=True
    ):
        observed = path_metadata[path]
        if stat.S_ISREG(observed.st_mode):
            path.unlink()
        elif stat.S_ISDIR(observed.st_mode):
            path.rmdir()
    raw.rmdir()
    if _output_lstat(
        raw, operation="transient_retirement_removed_root_lstat",
    ) is not None:
        raise WorkloadCaptureRefusal(
            "supplemental_failure_raw_retirement_failed", str(raw)
        )
    return {
        "path": record_path.relative_to(attempt_dir).as_posix(),
        "sha256": _sha256_output_file(
            record_path, role="transient raw retirement record",
            operation="transient_retirement_record_hash",
        ),
        "retirement_seal_sha256": document["retirement_seal_sha256"],
        "removed_file_count": document["removed_file_count"],
        "removed_bytes": document["removed_bytes"],
        "removed_files_sha256": document["removed_files_sha256"],
    }


def _retire_failed_transient_raw(
    *, attempt_dir: Path, cell_dir: Path,
) -> dict[str, Any] | None:
    try:
        return _retire_failed_transient_raw_untyped(
            attempt_dir=attempt_dir, cell_dir=cell_dir,
        )
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="output_transient_retirement",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _hardlink_tree_untyped(source: Path, destination: Path) -> None:
    """Project immutable bytes on the same filesystem; never copy/fallback."""
    source_metadata = _output_lstat(
        source, operation="hardlink_source_root_lstat",
    )
    if (
        source_metadata is None
        or stat.S_ISLNK(source_metadata.st_mode)
        or not stat.S_ISDIR(source_metadata.st_mode)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_publication_source_invalid", str(source)
        )
    _mkdir_nofollow(destination, exist_ok=False)
    destination_metadata = _output_lstat(
        destination, operation="hardlink_destination_root_lstat",
    )
    if destination_metadata is None or not stat.S_ISDIR(
        destination_metadata.st_mode
    ) or stat.S_ISLNK(destination_metadata.st_mode):
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_invalid", str(destination)
        )
    if source_metadata.st_dev != destination_metadata.st_dev:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_cross_device", str(destination)
        )
    for path in _output_tree_paths(source, operation="hardlink_source_walk"):
        path_metadata = _output_lstat(
            path, operation="hardlink_source_entry_lstat",
        )
        if path_metadata is None:
            raise WorkloadCaptureRefusal(
                "supplemental_publication_source_invalid", str(path)
            )
        if stat.S_ISLNK(path_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_symlink_refused", str(path)
            )
        relative = path.relative_to(source)
        projected = destination / relative
        if stat.S_ISDIR(path_metadata.st_mode):
            _mkdir_nofollow(projected, exist_ok=False)
            continue
        if not stat.S_ISREG(path_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_source_invalid", str(path)
            )
        _mkdir_nofollow(projected.parent, exist_ok=True)
        os.link(path, projected, follow_symlinks=False)
        source_stat = _output_lstat(
            path, operation="hardlink_linked_source_lstat",
        )
        projected_stat = _output_lstat(
            projected, operation="hardlink_linked_destination_lstat",
        )
        if (
            source_stat is None
            or projected_stat is None
            or not stat.S_ISREG(source_stat.st_mode)
            or not stat.S_ISREG(projected_stat.st_mode)
            or (source_stat.st_dev, source_stat.st_ino)
            != (projected_stat.st_dev, projected_stat.st_ino)
            or source_stat.st_size != projected_stat.st_size
            or _sha256_output_file(
                path, role="hardlink source artifact",
                operation="hardlink_source_hash",
            ) != _sha256_output_file(
                projected, role="hardlink projected artifact",
                operation="hardlink_destination_hash",
            )
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_link_mismatch", str(projected)
            )
        projected.chmod(0o444)
        _fsync_output_regular_nofollow(
            projected, role="hardlink projected artifact",
            operation="hardlink_destination_sync",
        )
    projected_directories: list[Path] = []
    for path in _output_tree_paths(
        destination, operation="hardlink_destination_walk",
    ):
        observed = _output_lstat(
            path, operation="hardlink_destination_entry_lstat",
        )
        if observed is None or stat.S_ISLNK(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(path)
            )
        if stat.S_ISDIR(observed.st_mode):
            projected_directories.append(path)
        elif not stat.S_ISREG(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(path)
            )
    for path in sorted(
        projected_directories, key=lambda item: len(item.parts), reverse=True,
    ):
        _fsync_directory(path)
        path.chmod(0o555)
    _fsync_directory(destination)
    destination.chmod(0o555)


def _hardlink_tree(source: Path, destination: Path) -> None:
    try:
        _hardlink_tree_untyped(source, destination)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="output_hardlink_projection",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _resume_hardlink_tree_untyped(source: Path, destination: Path) -> None:
    """Finish or verify one deterministic publication projection.

    A hard kill may leave a strict prefix of the projection.  Existing bytes
    are accepted only when they are the same immutable inode as the sealed
    source.  Unexpected paths, changed bytes, symlinks, and non-regular files
    are never repaired or discarded.
    """
    source_metadata = _output_lstat(
        source, operation="resume_hardlink_source_root_lstat",
    )
    if (
        source_metadata is None
        or stat.S_ISLNK(source_metadata.st_mode)
        or not stat.S_ISDIR(source_metadata.st_mode)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_publication_source_invalid", str(source)
        )
    destination_metadata = _output_lstat(
        destination, operation="resume_hardlink_destination_root_lstat",
    )
    if destination_metadata is not None:
        if stat.S_ISLNK(destination_metadata.st_mode) or not stat.S_ISDIR(
            destination_metadata.st_mode
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(destination)
            )
    else:
        _mkdir_nofollow(destination, exist_ok=False)
        destination_metadata = _output_lstat(
            destination,
            operation="resume_hardlink_created_destination_root_lstat",
        )
        if destination_metadata is None:
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(destination)
            )
    if source_metadata.st_dev != destination_metadata.st_dev:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_cross_device", str(destination)
        )
    if source_metadata.st_mode & 0o222:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_source_mutable", str(source)
        )
    if stat.S_IMODE(destination_metadata.st_mode) not in {0o755, 0o555}:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_mode_invalid", str(destination)
        )
    source_paths = _output_tree_paths(
        source, operation="resume_hardlink_source_walk",
    )
    destination_paths = _output_tree_paths(
        destination, operation="resume_hardlink_destination_walk",
    )
    expected = {path.relative_to(source) for path in source_paths}
    observed = {path.relative_to(destination) for path in destination_paths}
    if observed - expected:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_extra",
            repr(sorted(path.as_posix() for path in observed - expected)),
        )
    for path in source_paths:
        path_metadata = _output_lstat(
            path, operation="resume_hardlink_source_entry_lstat",
        )
        if path_metadata is None:
            raise WorkloadCaptureRefusal(
                "supplemental_publication_source_invalid", str(path)
            )
        if stat.S_ISLNK(path_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_output_symlink_refused", str(path)
            )
        relative = path.relative_to(source)
        projected = destination / relative
        if stat.S_ISDIR(path_metadata.st_mode):
            if path_metadata.st_mode & 0o222:
                raise WorkloadCaptureRefusal(
                    "supplemental_publication_source_mutable", str(path)
                )
            projected_metadata = _output_lstat(
                projected, operation="resume_hardlink_projected_dir_lstat",
            )
            if projected_metadata is not None:
                if stat.S_ISLNK(
                    projected_metadata.st_mode
                ) or not stat.S_ISDIR(projected_metadata.st_mode):
                    raise WorkloadCaptureRefusal(
                        "supplemental_publication_pending_invalid",
                        str(projected),
                    )
                if stat.S_IMODE(projected_metadata.st_mode) not in {
                    0o755, 0o555,
                }:
                    raise WorkloadCaptureRefusal(
                        "supplemental_publication_pending_mode_invalid",
                        str(projected),
                    )
            else:
                _mkdir_nofollow(projected, exist_ok=False)
            continue
        if not stat.S_ISREG(path_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_source_invalid", str(path)
            )
        if path_metadata.st_mode & 0o222:
            raise WorkloadCaptureRefusal(
                "supplemental_publication_source_mutable", str(path)
            )
        _mkdir_nofollow(projected.parent, exist_ok=True)
        projected_metadata = _output_lstat(
            projected, operation="resume_hardlink_projected_file_lstat",
        )
        if projected_metadata is not None:
            if stat.S_ISLNK(
                projected_metadata.st_mode
            ) or not stat.S_ISREG(projected_metadata.st_mode):
                raise WorkloadCaptureRefusal(
                    "supplemental_publication_pending_invalid", str(projected)
                )
        else:
            os.link(path, projected, follow_symlinks=False)
        source_stat = _output_lstat(
            path, operation="resume_hardlink_linked_source_lstat",
        )
        projected_stat = _output_lstat(
            projected, operation="resume_hardlink_linked_destination_lstat",
        )
        if (
            source_stat is None
            or projected_stat is None
            or not stat.S_ISREG(source_stat.st_mode)
            or not stat.S_ISREG(projected_stat.st_mode)
            or (source_stat.st_dev, source_stat.st_ino)
            != (projected_stat.st_dev, projected_stat.st_ino)
            or source_stat.st_size != projected_stat.st_size
            or projected_stat.st_mode & 0o222
            or _sha256_output_file(
                path, role="resume hardlink source artifact",
                operation="resume_hardlink_source_hash",
            ) != _sha256_output_file(
                projected, role="resume hardlink projected artifact",
                operation="resume_hardlink_destination_hash",
            )
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_link_mismatch", str(projected)
            )
        _fsync_output_regular_nofollow(
            projected, role="resume hardlink projected artifact",
            operation="resume_hardlink_destination_sync",
        )
    completed = {
        path.relative_to(destination) for path in _output_tree_paths(
            destination, operation="resume_hardlink_completed_walk",
        )
    }
    if completed != expected:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_incomplete", str(destination)
        )
    completed_directories: list[Path] = []
    for path in (destination / relative for relative in completed):
        observed_metadata = _output_lstat(
            path, operation="resume_hardlink_completed_entry_lstat",
        )
        if observed_metadata is None or stat.S_ISLNK(
            observed_metadata.st_mode
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(path)
            )
        if stat.S_ISDIR(observed_metadata.st_mode):
            completed_directories.append(path)
        elif not stat.S_ISREG(observed_metadata.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(path)
            )
    for path in sorted(
        completed_directories, key=lambda item: len(item.parts), reverse=True,
    ):
        _fsync_directory(path)
        path.chmod(0o555)
    _fsync_directory(destination)
    destination.chmod(0o555)


def _resume_hardlink_tree(source: Path, destination: Path) -> None:
    try:
        _resume_hardlink_tree_untyped(source, destination)
    except OSError as exc:
        mapped = storage_refusal_from_exception(
            exc, operation="output_hardlink_projection_resume",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def _retention_limits(document: dict[str, Any]) -> tuple[int, int, int]:
    try:
        retention = int(document["retention_budget_gib"]) * GIB
        metadata = int(document["metadata_margin_gib"]) * GIB
        raw = int(document["raw_staging_margin_gib"]) * GIB
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_aggregate_budget_invalid", str(exc)
        ) from exc
    if retention <= 0 or metadata < 0 or raw <= 0 or metadata >= retention:
        raise WorkloadCaptureRefusal(
            "supplemental_aggregate_budget_invalid", repr(document)
        )
    return retention, metadata, raw


def validate_representative_aggregate_budget(
    matrix: dict[str, Any], aggregate_budget: dict[str, Any],
) -> None:
    """Require the external operator budget to cover every sealed row ceiling."""
    if not is_representative_execution_matrix(matrix):
        return
    required_gib = matrix["aggregate_retention_contract"][
        "required_retention_budget_gib"
    ]
    observed = aggregate_budget.get("retention_budget_gib")
    if type(observed) is not int or observed < required_gib:
        raise WorkloadCaptureRefusal(
            "representative_aggregate_retention_budget_insufficient",
            f"required_gib={required_gib}; observed_gib={observed}",
        )


def _require_live_supplemental_source_state(matrix: dict[str, Any]) -> str:
    """Require the clean current producer commit sealed by the live matrix."""
    head, clean, dirty = canonical_capture._git_state()
    if head != matrix.get("source_commit") or not clean:
        raise WorkloadCaptureRefusal(
            "supplemental_source_state_invalid",
            f"HEAD={head}, expected={matrix.get('source_commit')}, "
            f"dirty={dirty!r}",
        )
    return head


def _require_retention_capacity(
    *, output_root: Path, aggregate_budget: dict[str, Any],
    next_retained_max: int,
) -> None:
    retention, metadata, raw = _retention_limits(aggregate_budget)
    current = _tree_bytes(output_root)
    attempt_reserve = max(next_retained_max, raw)
    if current + attempt_reserve + metadata > retention:
        raise WorkloadCaptureRefusal(
            "supplemental_retention_budget_exceeded",
            f"current={current}; next_max={next_retained_max}; "
            f"raw_attempt_reserve={raw}; effective_reserve={attempt_reserve}; "
            f"metadata_reserve={metadata}; ceiling={retention}",
        )


def _verify_artifacts(
    result: dict[str, Any], *, root: Path,
) -> int:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise WorkloadCaptureRefusal(
            "supplemental_artifacts_invalid", repr(artifacts)
        )
    total = 0
    seen: set[str] = set()
    resolved_root = _resolve_output_path(
        root, operation="supplemental_artifact_root_resolve",
    )
    for row in artifacts:
        rel = row.get("relative_path") if isinstance(row, dict) else None
        if (
            not isinstance(rel, str) or not rel or rel in seen
            or Path(rel).is_absolute() or ".." in Path(rel).parts
            or not SHA256_RE.fullmatch(str(row.get("sha256", "")))
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_artifacts_invalid", repr(row)
            )
        seen.add(rel)
        candidate = root / rel
        observed = _output_lstat(
            candidate, operation="supplemental_artifact_lstat",
        )
        if observed is None or stat.S_ISLNK(
            observed.st_mode
        ) or not stat.S_ISREG(observed.st_mode):
            raise WorkloadCaptureRefusal(
                "supplemental_artifact_missing", str(candidate)
            )
        path = _resolve_output_path(
            candidate, operation="supplemental_artifact_resolve",
        )
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise WorkloadCaptureRefusal(
                "supplemental_artifact_path_escape", rel
            ) from exc
        if observed.st_size != row["size_bytes"] or _sha256_output_file(
            path, role="supplemental published artifact",
            operation="supplemental_artifact_hash",
        ) != row["sha256"]:
            raise WorkloadCaptureRefusal(
                "supplemental_artifact_hash_mismatch", rel
            )
        total += row["size_bytes"]
    return total


def validate_result_launch_completeness(result: dict[str, Any]) -> None:
    document = result.get("launch_completeness")
    if not isinstance(document, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_done_launch_completeness_invalid", "not an object"
        )
    seal = document.get("completeness_sha256")
    unsealed = {
        key: value for key, value in document.items()
        if key != "completeness_sha256"
    }
    launches = document.get("launches")
    if (
        not isinstance(seal, str) or seal != canonical_sha256(unsealed)
        or document.get("launch_scope") != "all_in_profiler_roi"
        or document.get("contract") not in (
            canonical_capture._SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT,
            canonical_capture._SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT,
        )
        or not isinstance(launches, list) or not launches
        or document.get("launch_inventory_sha256") is None
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_launch_completeness_invalid", repr(seal)
        )
    if [row.get("launch_ordinal") for row in launches] != list(
        range(len(launches))
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_launch_completeness_invalid", "ordinals"
        )
    sequence = [
        {
            "launch_ordinal": row["launch_ordinal"],
            "kernel_name": row.get("kernel_name"),
            "launch_identity_sha256": row.get("launch_identity_sha256"),
        }
        for row in launches
    ]
    trace_sequence_sha = canonical_sha256(sequence)
    if document.get("trace_launch_sequence_sha256") != trace_sequence_sha:
        raise WorkloadCaptureRefusal(
            "supplemental_done_launch_completeness_invalid", "trace sequence"
        )
    counts = {
        "inventory_launch_count": len(launches),
        "trace_complete_launch_count": sum(
            row.get("trace_complete") is True for row in launches
        ),
        "ncu_label_complete_launch_count": sum(
            row.get("ncu_label_complete") is True for row in launches
        ),
        "control_complete_launch_count": sum(
            row.get("control_complete") is True for row in launches
        ),
    }
    if any(document.get(key) != value for key, value in counts.items()):
        raise WorkloadCaptureRefusal(
            "supplemental_done_launch_completeness_invalid", repr(counts)
        )
    sequence_equal = (
        document.get("trace_ncu_sequence_equal") is True
        and document.get("ncu_replay_sequence_sha256") == trace_sequence_sha
    )
    representative = (
        document.get("contract")
        == canonical_capture._SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    if representative:
        controls_allowed = all(
            row.get("control_complete") is True
            or (
                row.get("control_outcome")
                == "typed_unavailable_no_stable_static_module"
                and row.get("control_typed_reason") == "origin_unresolved"
            )
            for row in launches
        )
        complete = (
            counts["trace_complete_launch_count"] == len(launches)
            and counts["ncu_label_complete_launch_count"] == len(launches)
            and sequence_equal and controls_allowed
        )
    else:
        complete = (
            all(
                value == len(launches) for key, value in counts.items()
                if key != "inventory_launch_count"
            ) and sequence_equal
        )
    expected_status = (
        "complete_dynamic_launch_set" if complete
        else "typed_incomplete_dynamic_launch_set"
    )
    expected_disposition = "admitted" if complete else "failed"
    expected_reason = None if complete else "supplemental_dynamic_launch_set_incomplete"
    if (
        document.get("status") != expected_status
        or result.get("disposition") != expected_disposition
        or result.get("typed_reason") != expected_reason
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_launch_completeness_invalid",
            repr({
                "status": document.get("status"),
                "disposition": result.get("disposition"),
                "typed_reason": result.get("typed_reason"),
            }),
        )


def require_exact_result_membership_order(
    membership: list[dict[str, Any]], results: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Bind DONE result order to the chunk's breadth-first row sequence."""
    expected = [_row_identity(key) for key in membership]
    observed = [
        _row_identity((result.get("workload_row_binding") or {}).get("row_key"))
        for result in results
    ]
    if observed != expected or len(set(observed)) != len(observed):
        raise WorkloadCaptureRefusal(
            "supplemental_done_membership_order_mismatch",
            repr({"expected": expected, "observed": observed}),
        )
    return observed


def _chunk_attempt_histories(
    *, stage_root: Path, results: list[dict[str, Any]],
    canonical_matrix_sha256: str,
    canonical_matrix_seal_sha256: str,
    preflight_sha256: str, source_commit: str,
) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for result in results:
        binding = result["workload_row_binding"]
        safe_name = _safe_name(binding["row_key"])
        state = _scan_journal_attempts(
            journal_dir=stage_root / "journals" / safe_name,
            expected_binding=binding,
            canonical_matrix_sha256=canonical_matrix_sha256,
            canonical_matrix_seal_sha256=canonical_matrix_seal_sha256,
            preflight_sha256=preflight_sha256, source_commit=source_commit,
        )
        completed = state["completed"]
        if completed is None:
            raise WorkloadCaptureRefusal(
                "supplemental_completed_attempt_cardinality", safe_name
            )
        selected_start, _selected_result, selected_dir = completed
        document = {
            "row_key": copy.deepcopy(binding["row_key"]),
            "selected_attempt_ordinal": selected_start["attempt_ordinal"],
            "selected_attempt_id": selected_start["attempt_id"],
            "attempts": state["attempts"],
            "attempts_sha256": canonical_sha256(state["attempts"]),
        }
        document["attempt_history_sha256"] = canonical_sha256(document)
        histories.append(document)
    return histories


def _validate_embedded_attempt_histories(
    histories: list[dict[str, Any]], *, results: list[dict[str, Any]],
    preflight_sha256s: set[str],
) -> None:
    """Validate the copied attempt chain without reopening staging paths."""
    if len(histories) != len(results):
        raise WorkloadCaptureRefusal(
            "supplemental_done_attempt_history_mismatch", "cardinality"
        )
    for history, result in zip(histories, results, strict=True):
        binding = result.get("workload_row_binding") or {}
        attempts = history.get("attempts")
        if (
            history.get("row_key") != binding.get("row_key")
            or not isinstance(attempts, list) or not attempts
            or history.get("attempts_sha256") != canonical_sha256(attempts)
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_done_attempt_history_mismatch",
                repr(binding.get("row_key")),
            )
        unsealed_history = {
            key: value for key, value in history.items()
            if key != "attempt_history_sha256"
        }
        if history.get("attempt_history_sha256") != canonical_sha256(
            unsealed_history
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_done_attempt_history_mismatch", "history seal"
            )
        selected: list[dict[str, Any]] = []
        for ordinal, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict):
                raise WorkloadCaptureRefusal(
                    "supplemental_done_attempt_history_mismatch", "attempt type"
                )
            unsealed_attempt = {
                key: value for key, value in attempt.items()
                if key != "attempt_summary_sha256"
            }
            if (
                attempt.get("attempt_ordinal") != ordinal
                or attempt.get("attempt_summary_sha256")
                != canonical_sha256(unsealed_attempt)
                or (
                    attempt.get("capture_preflight_sha256") is not None
                    and attempt.get("capture_preflight_sha256")
                    not in preflight_sha256s
                )
            ):
                raise WorkloadCaptureRefusal(
                    "supplemental_done_attempt_history_mismatch",
                    f"attempt {ordinal}",
                )
            if attempt.get("state") == "complete":
                selected.append(attempt)
        if (
            len(selected) != 1
            or history.get("selected_attempt_ordinal")
            != selected[0].get("attempt_ordinal")
            or history.get("selected_attempt_id")
            != selected[0].get("attempt_id")
            or result.get("capture_preflight_sha256")
            != selected[0].get("capture_preflight_sha256")
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_done_attempt_history_mismatch", "selected attempt"
            )


def build_done_manifest(
    *, chunk: dict[str, Any], results: list[dict[str, Any]],
    source_commit: str, matrix_path: Path, matrix: dict[str, Any],
    canonical_matrix_path: Path, canonical_matrix: dict[str, Any],
    authority_path: Path, pins_path: Path, aggregate_budget_path: Path,
    cell_budget_path: Path, registry_path: Path, preflight_path: Path,
    final_dir: Path, command: str,
    flashinfer_proof: dict[str, Any],
    stage_root: Path,
    publication_root: Path,
    campaign_root: Path | None = None,
) -> dict[str, Any]:
    del command  # Commands are machine-local execution context, not evidence identity.
    campaign_root = (
        Path(campaign_root) if campaign_root is not None
        else final_dir.parents[1]
    )
    owner_marker, _owner_payload, owner_pin = _load_retained_input_owner(
        campaign_root
    )
    campaign_identity = owner_marker["campaign_identity"]
    identities = campaign_identity.get("input_file_identities")
    if (
        campaign_identity.get("source_commit") != source_commit
        or not isinstance(identities, dict)
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "campaign identity"
        )
    input_references: dict[str, dict[str, Any]] = {}
    input_payloads: dict[str, bytes] = {}
    for role, identity in sorted(identities.items()):
        if not isinstance(identity, dict) or identity.get("role") != role:
            raise WorkloadCaptureRefusal(
                "supplemental_input_reference_invalid", repr(role)
            )
        reference = _portable_input_reference(identity)
        input_references[role] = reference
        input_payloads[role] = _read_portable_input_reference(
            reference, campaign_root=campaign_root, repo_root=REPO_ROOT,
            source_commit=source_commit,
        )
    current_preflight_reference, _current_preflight_payload = (
        _retained_external_reference(
        path=preflight_path, campaign_root=campaign_root, role="preflight"
        )
    )
    required_roles = {
        "authority", "pins", "aggregate_budget", "cell_budgets",
        "metric_registry", "full_coverage_inventory",
        "live_coverage_inventory", "runnability", "shrink_provenance",
        "supplemental_matrix", "canonical_matrix",
    }
    if not required_roles <= set(input_references):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch",
            repr(sorted(required_roles - set(input_references))),
        )
    try:
        retained_matrix = _parse_canonical_object_payload(
            input_payloads["supplemental_matrix"], role="supplemental matrix"
        )
        retained_canonical = _parse_canonical_object_payload(
            input_payloads["canonical_matrix"], role="canonical matrix"
        )
    except WorkloadCaptureRefusal:
        raise
    if retained_matrix != matrix or retained_canonical != canonical_matrix:
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "matrix data model"
        )
    membership = copy.deepcopy(chunk["rows"])
    require_exact_result_membership_order(membership, results)
    retained_bytes = sum(
        sum(item["size_bytes"] for item in result["artifacts"])
        for result in results
    )
    canonical_matrix_sha256 = input_references["canonical_matrix"]["sha256"]
    attempt_histories = _chunk_attempt_histories(
        stage_root=stage_root, results=results,
        canonical_matrix_sha256=canonical_matrix_sha256,
        canonical_matrix_seal_sha256=canonical_matrix[
            "matrix_seal_sha256"
        ],
        preflight_sha256=current_preflight_reference["sha256"],
        source_commit=source_commit,
    )
    attempt_preflight_sha256s = {
        attempt["capture_preflight_sha256"]
        for history in attempt_histories
        for attempt in history["attempts"]
        if attempt.get("capture_preflight_sha256") is not None
    }
    preflight_references = _retained_preflight_references(
        campaign_root=campaign_root, source_commit=source_commit,
        chunk_id=chunk["chunk_id"],
        required_sha256=attempt_preflight_sha256s,
    )
    _validate_embedded_attempt_histories(
        attempt_histories, results=results,
        preflight_sha256s={
            reference["sha256"] for reference in preflight_references
        },
    )
    publication_storage = _tree_usage(publication_root / "cells")
    dispositions: dict[str, int] = {}
    launch_counts = {
        "complete_dynamic_launch_set_rows": 0,
        "typed_incomplete_dynamic_launch_set_rows": 0,
        "inventory_launches": 0,
        "trace_complete_launches": 0,
        "ncu_label_complete_launches": 0,
        "control_complete_launches": 0,
    }
    for result in results:
        validate_result_launch_completeness(result)
        value = result.get("disposition")
        dispositions[value] = dispositions.get(value, 0) + 1
        completeness = result.get("launch_completeness") or {}
        status = completeness.get("status")
        if status == "complete_dynamic_launch_set":
            launch_counts["complete_dynamic_launch_set_rows"] += 1
        elif status == "typed_incomplete_dynamic_launch_set":
            launch_counts["typed_incomplete_dynamic_launch_set_rows"] += 1
        else:
            raise WorkloadCaptureRefusal(
                "supplemental_done_launch_completeness_invalid",
                repr(result.get("workload_row_binding", {}).get("row_key")),
            )
        for source, destination in (
            ("inventory_launch_count", "inventory_launches"),
            ("trace_complete_launch_count", "trace_complete_launches"),
            ("ncu_label_complete_launch_count", "ncu_label_complete_launches"),
            ("control_complete_launch_count", "control_complete_launches"),
        ):
            value = completeness.get(source)
            if type(value) is not int or value < 0:
                raise WorkloadCaptureRefusal(
                    "supplemental_done_launch_completeness_invalid", source
                )
            launch_counts[destination] += value
    manifest = {
        "schema_version": 2,
        "manifest_kind": "sol_workload_capture_chunk_done",
        "chunk_id": chunk["chunk_id"],
        "cohort": chunk["cohort"],
        "source_commit": source_commit,
        "generated_at_utc": _utc_text(),
        "publication_relative_path": (
            Path("sol_workload_chunks") / chunk["chunk_id"]
        ).as_posix(),
        "retained_input_owner": owner_pin,
        "supplemental_matrix": {
            "input_role": "supplemental_matrix",
            "sha256": input_references["supplemental_matrix"]["sha256"],
            "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        },
        "canonical_matrix": {
            "input_role": "canonical_matrix",
            "sha256": input_references["canonical_matrix"]["sha256"],
            "matrix_seal_sha256": canonical_matrix["matrix_seal_sha256"],
        },
        "flashinfer_corpus": flashinfer_done_proof(flashinfer_proof),
        "inputs": input_references,
        "preflights": preflight_references,
        "chunk_membership": membership,
        "chunk_membership_sha256": canonical_sha256(membership),
        "row_count": len(membership),
        "counts": dict(sorted(dispositions.items())),
        "launch_counts": launch_counts,
        "retained_artifact_bytes": retained_bytes,
        "publication_storage": publication_storage,
        "attempt_histories": attempt_histories,
        "cell_results": results,
    }
    manifest["done_seal_sha256"] = canonical_sha256(manifest)
    _validate_schema(
        manifest, DONE_SCHEMA, code="supplemental_done_schema_invalid"
    )
    return manifest


def verify_done_manifest(
    manifest: dict[str, Any], *, final_dir: Path,
    matrix: dict[str, Any] | None = None, matrix_path: Path | None = None,
    canonical_matrix: dict[str, Any] | None = None,
    canonical_matrix_path: Path | None = None,
    campaign_root: Path | None = None, repo_root: Path = REPO_ROOT,
    require_current_state: bool = True,
    require_final_location: bool = True,
    verify_git_provenance: bool = False,
) -> None:
    del matrix_path, canonical_matrix_path
    _validate_schema(
        manifest, DONE_SCHEMA, code="supplemental_done_schema_invalid"
    )
    _verify_sealed(
        manifest, "done_seal_sha256", code="supplemental_done_seal_mismatch"
    )
    campaign_root = (
        Path(campaign_root) if campaign_root is not None
        else final_dir.parents[1]
    )
    relative_publication = _safe_relative_path(
        manifest.get("publication_relative_path"), role="DONE publication"
    )
    expected_relative = Path("sol_workload_chunks") / manifest["chunk_id"]
    if (
        relative_publication != expected_relative
        or (
            require_final_location
            and final_dir.absolute()
            != (campaign_root / relative_publication).absolute()
        )
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_publication_path_mismatch", str(final_dir)
        )
    owner_marker, _owner_payload, owner_pin = _load_retained_input_owner(
        campaign_root
    )
    campaign_identity = owner_marker["campaign_identity"]
    if (
        manifest.get("retained_input_owner") != owner_pin
        or campaign_identity.get("source_commit") != manifest["source_commit"]
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "retained input owner"
        )
    identities = campaign_identity.get("input_file_identities")
    if not isinstance(identities, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "input identities"
        )
    expected_references = {
        role: _portable_input_reference(identity)
        for role, identity in sorted(identities.items())
        if isinstance(identity, dict)
    }
    references = manifest.get("inputs")
    if references != expected_references:
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "input references"
        )
    input_payloads = {
        role: _read_portable_input_reference(
            reference, campaign_root=campaign_root, repo_root=repo_root,
            source_commit=manifest["source_commit"],
            verify_git_provenance=verify_git_provenance,
        )
        for role, reference in sorted(references.items())
    }
    preflight_references = manifest.get("preflights")
    if not isinstance(preflight_references, list):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "preflight references"
        )
    preflight_sha256s: set[str] = set()
    preflight_paths: set[str] = set()
    expected_preflight_prefix = (
        "sol_workload_campaign", "inputs", "preflights",
        manifest["source_commit"], manifest["chunk_id"],
    )
    for reference in preflight_references:
        relative = _safe_relative_path(
            reference.get("owner_relative_path")
            if isinstance(reference, dict) else None,
            role="preflight",
        )
        if (
            not isinstance(reference, dict)
            or reference.get("kind") != "retained_external_input"
            or reference.get("role") != "preflight"
            or relative.parts[:5] != expected_preflight_prefix
            or len(relative.parts) != 6
            or reference.get("source_name") != relative.name
            or reference.get("sha256") in preflight_sha256s
            or relative.as_posix() in preflight_paths
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_done_input_mismatch", "preflight reference"
            )
        _read_portable_input_reference(
            reference, campaign_root=campaign_root, repo_root=repo_root,
            source_commit=manifest["source_commit"],
            verify_git_provenance=False,
        )
        preflight_sha256s.add(reference["sha256"])
        preflight_paths.add(relative.as_posix())
    if preflight_references != sorted(
        preflight_references,
        key=lambda row: (row["sha256"], row["owner_relative_path"]),
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "preflight reference order"
        )
    retained_matrix = _parse_canonical_object_payload(
        input_payloads["supplemental_matrix"], role="supplemental matrix"
    )
    retained_canonical = _parse_canonical_object_payload(
        input_payloads["canonical_matrix"], role="canonical matrix"
    )
    if matrix is not None and matrix != retained_matrix:
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "supplemental matrix object"
        )
    if canonical_matrix is not None and canonical_matrix != retained_canonical:
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "canonical matrix object"
        )
    matrix = retained_matrix
    canonical_matrix = retained_canonical
    try:
        schema_path = (
            representative_execution.SCHEMA_PATH
            if is_representative_execution_matrix(matrix)
            else workload_matrix.SCHEMA_PATH
        )
        supplemental_schema = json.loads(
            (repo_root / schema_path).read_text()
        )
        jsonschema.validate(matrix, supplemental_schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", f"matrix schema: {exc}"
        ) from exc
    if (
        campaign_identity.get("supplemental_matrix_sha256")
        != references["supplemental_matrix"]["sha256"]
        or campaign_identity.get("supplemental_matrix_seal_sha256")
        != matrix.get("matrix_seal_sha256")
        or campaign_identity.get("canonical_matrix_sha256")
        != references["canonical_matrix"]["sha256"]
        or campaign_identity.get("canonical_matrix_seal_sha256")
        != canonical_matrix.get("matrix_seal_sha256")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "campaign matrix pins"
        )
    chunks = [
        row for row in matrix["chunks"]
        if row["chunk_id"] == manifest["chunk_id"]
    ]
    _chunk, _resolved = select_workload_chunk(
        matrix, canonical_matrix, manifest["chunk_id"]
    )
    flashinfer = manifest.get("flashinfer_corpus")
    if not isinstance(flashinfer, dict):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "FlashInfer proof"
        )
    flashinfer_unsigned = {
        key: value for key, value in flashinfer.items()
        if key != "corpus_identity_sha256"
    }
    if (
        flashinfer.get("corpus_identity_sha256")
        != canonical_sha256(flashinfer_unsigned)
        or flashinfer.get("root_repo_path")
        != "data/annex/flashinfer_trace"
        or flashinfer.get("data_annex_manifest_path")
        != "evidence/manifests/data_annex.sha256"
        or flashinfer.get("data_annex_manifest_sha256")
        != sol_recon.DATA_ANNEX_MANIFEST_SHA256
        or flashinfer.get("filtered_manifest_sha256")
        != sol_recon.FLASHINFER_CORPUS_MANIFEST_SHA256
        or flashinfer.get("file_count") != sol_recon.FLASHINFER_CORPUS_FILE_COUNT
        or flashinfer.get("safetensors_count")
        != sol_recon.FLASHINFER_CORPUS_BLOB_COUNT
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", "FlashInfer proof"
        )
    _validate_embedded_attempt_histories(
        manifest["attempt_histories"], results=manifest["cell_results"],
        preflight_sha256s=preflight_sha256s,
    )
    if (
        len(chunks) != 1
        or manifest["chunk_membership"] != chunks[0]["rows"]
        or manifest["chunk_membership_sha256"]
        != canonical_sha256(manifest["chunk_membership"])
        or manifest["supplemental_matrix"] != {
            "input_role": "supplemental_matrix",
            "sha256": references["supplemental_matrix"]["sha256"],
            "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        }
        or manifest["canonical_matrix"] != {
            "input_role": "canonical_matrix",
            "sha256": references["canonical_matrix"]["sha256"],
            "matrix_seal_sha256": canonical_matrix["matrix_seal_sha256"],
        }
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_input_mismatch", str(final_dir)
        )
    expected_key_sequence = require_exact_result_membership_order(
        manifest["chunk_membership"], manifest["cell_results"]
    )
    expected_keys = set(expected_key_sequence)
    result_keys: set[tuple[str, str, str]] = set()
    artifact_total = 0
    dispositions: dict[str, int] = {}
    launch_counts = {
        "complete_dynamic_launch_set_rows": 0,
        "typed_incomplete_dynamic_launch_set_rows": 0,
        "inventory_launches": 0,
        "trace_complete_launches": 0,
        "ncu_label_complete_launches": 0,
        "control_complete_launches": 0,
    }
    rows_by_key = {
        _row_identity(row["key"]): row for row in matrix["rows"]
    }
    matrix_sha = references["supplemental_matrix"]["sha256"]
    for result in manifest["cell_results"]:
        validate_result_launch_completeness(result)
        binding = result.get("workload_row_binding") or {}
        key = _row_identity(binding.get("row_key"))
        if key in result_keys:
            raise WorkloadCaptureRefusal(
                "supplemental_done_result_duplicate", repr(key)
            )
        result_keys.add(key)
        matrix_row = rows_by_key.get(key)
        expected_binding = (
            workload_row_binding(
                matrix=matrix, matrix_sha256=matrix_sha,
                chunk_id=manifest["chunk_id"], row=matrix_row,
            ) if matrix_row is not None else None
        )
        if binding != expected_binding:
            raise WorkloadCaptureRefusal(
                "supplemental_done_binding_mismatch", repr(key)
            )
        artifact_total += _verify_artifacts(result, root=final_dir)
        disposition = result.get("disposition")
        dispositions[disposition] = dispositions.get(disposition, 0) + 1
        completeness = result.get("launch_completeness") or {}
        status = completeness.get("status")
        if status == "complete_dynamic_launch_set":
            launch_counts["complete_dynamic_launch_set_rows"] += 1
        elif status == "typed_incomplete_dynamic_launch_set":
            launch_counts["typed_incomplete_dynamic_launch_set_rows"] += 1
        else:
            raise WorkloadCaptureRefusal(
                "supplemental_done_launch_completeness_invalid", repr(key)
            )
        for source, destination in (
            ("inventory_launch_count", "inventory_launches"),
            ("trace_complete_launch_count", "trace_complete_launches"),
            ("ncu_label_complete_launch_count", "ncu_label_complete_launches"),
            ("control_complete_launch_count", "control_complete_launches"),
        ):
            value = completeness.get(source)
            if type(value) is not int or value < 0:
                raise WorkloadCaptureRefusal(
                    "supplemental_done_launch_completeness_invalid", repr(key)
                )
            launch_counts[destination] += value
    if (
        result_keys != expected_keys
        or manifest["row_count"] != len(expected_keys)
        or manifest["counts"] != dict(sorted(dispositions.items()))
        or manifest["launch_counts"] != launch_counts
        or manifest["retained_artifact_bytes"] != artifact_total
        or manifest["publication_storage"] != _tree_usage(final_dir / "cells")
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_done_result_mismatch", str(final_dir)
        )
    if require_current_state:
        environment = os.environ.copy()
        for name in (
            "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_DIR",
            "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX",
            "GIT_WORK_TREE",
        ):
            environment.pop(name, None)
        head_result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, env=environment,
        )
        status_result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1",
             "--untracked-files=all"],
            capture_output=True, text=True, check=False, env=environment,
        )
        head = head_result.stdout.strip()
        dirty = status_result.stdout.splitlines()
        if (
            head_result.returncode != 0 or status_result.returncode != 0
            or head != manifest["source_commit"] or dirty
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_done_source_state_invalid",
                f"head={head}; dirty={dirty!r}",
            )


def _certify_environment(
    *, supplemental_matrix: dict[str, Any], supplemental_matrix_path: Path,
    preflight_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    environment_matrix = supplemental_matrix
    if is_representative_execution_matrix(supplemental_matrix):
        environment_matrix = {
            **supplemental_matrix,
            "schema_version": 2,
            "artifact_id": sol_environment.SUPPLEMENTAL_MATRIX_ARTIFACT_ID,
        }
    try:
        probe, environment = sol_environment.certify_supplemental_sol_environment(
            environment_matrix, matrix_path=supplemental_matrix_path,
            ambient=os.environ,
        )
    except sol_environment.SolEnvironmentRefusal as exc:
        raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
    sealed = preflight_manifest["checks"]["sol_environment"]
    stable_fields = (
        "ambient", "environment_contract", "environment_contract_sha256",
        "constructed_environment_delta", "matrix_pin", "matrix_sha256",
        "observed_fingerprint", "compiler_identity", "nvcc_release",
        "nvcc_version_output",
    )
    drift = [name for name in stable_fields if probe.get(name) != sealed.get(name)]
    if drift:
        raise WorkloadCaptureRefusal(
            "SUPPLEMENTAL_SOL_ENV_PREFLIGHT_DRIFT", repr(drift)
        )
    return probe, environment


def _validate_supplemental_preflight_reference(
    *, preflight_path: Path, preflight_sha256: str,
    pins: dict[str, Any], pins_path: Path,
    aggregate_budget_path: Path,
    supplemental_matrix: dict[str, Any], supplemental_matrix_path: Path,
    supplemental_matrix_sha256: str, output_root: Path,
    expected_head: str, require_fresh: bool,
    admitted_at_utc: datetime,
) -> tuple[
    dict[str, Any], canonical_capture.ncu_executor.ExpectedNcuIdentity | None,
]:
    """Bind preflight to the live matrix; frozen canonical bytes are context."""
    return canonical_capture._validate_referenced_preflight(
        preflight_path=preflight_path, expected_sha256=preflight_sha256,
        pins=pins, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path,
        matrix=supplemental_matrix, matrix_path=supplemental_matrix_path,
        matrix_sha256=supplemental_matrix_sha256,
        output_root=output_root, expected_head=expected_head,
        require_fresh=require_fresh, admitted_at_utc=admitted_at_utc,
    )


def _capture_one(
    *, journal_dir: Path, cell: dict[str, Any], binding: dict[str, Any],
    supplemental_runnability_evidence: dict[str, Any] | None,
    canonical_matrix: dict[str, Any], pins: dict[str, Any], cell_budgets: Any,
    preflight_manifest: dict[str, Any],
    expected_ncu_identity: canonical_capture.ncu_executor.ExpectedNcuIdentity,
    preflight_path: Path,
    preflight_sha256: str, matrix_sha256: str, cell_budget_sha256: str,
    raw_staging_margin_bytes: int, source_commit: str,
    sol_environment_probe: dict[str, Any],
    certified_sol_env: dict[str, str], output_root: Path,
    aggregate_budget: dict[str, Any],
    flashinfer_proof: dict[str, Any],
    canonical_matrix_sha256: str,
    expected_sol_fingerprint_pin: dict[str, Any],
    pins_sha256: str,
    runnability_path: Path | None = None,
    supplemental_workload_path: Path | None = None,
    supplemental_definition_path: Path | None = None,
    launch_set_registry: metric_registry.MetricRegistry | None = None,
) -> dict[str, Any]:
    safe_name = _safe_name(binding["row_key"])
    _mkdir_nofollow(journal_dir, exist_ok=True)
    if journal_dir.is_symlink() or not journal_dir.is_dir():
        raise WorkloadCaptureRefusal(
            "supplemental_journal_path_invalid", str(journal_dir)
        )
    state = _scan_journal_attempts(
        journal_dir=journal_dir, expected_binding=binding,
        canonical_matrix_sha256=canonical_matrix_sha256,
        canonical_matrix_seal_sha256=canonical_matrix[
            "matrix_seal_sha256"
        ],
        preflight_sha256=preflight_sha256, source_commit=source_commit,
    )
    if state["completed"] is not None:
        return state["completed"][1]
    for _ordinal, prior_dir in _attempt_directories(journal_dir):
        _seal_attempt_tree(prior_dir)
    attempts_root = journal_dir / "attempts"
    _mkdir_nofollow(attempts_root, exist_ok=True)
    attempt_ordinal = state["next_ordinal"]
    attempt_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    attempt_dir = attempts_root / f"attempt-{attempt_ordinal:06d}"
    admission_run_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    _mkdir_nofollow(attempt_dir, exist_ok=False)
    preflight_copy = attempt_dir / "capture_preflight.json"
    preflight_descriptor, _absolute, _observed = _open_regular_nofollow(
        preflight_path, role="capture preflight"
    )
    try:
        payload = bytearray()
        while True:
            block = os.read(preflight_descriptor, 1 << 20)
            if not block:
                break
            payload.extend(block)
    finally:
        os.close(preflight_descriptor)
    if hashlib.sha256(payload).hexdigest() != preflight_sha256:
        raise WorkloadCaptureRefusal(
            "supplemental_attempt_preflight_source_drift", str(preflight_path)
        )
    _write_bytes_new_nofollow(
        preflight_copy, payload, mode=0o444,
        operation="attempt_preflight_publication",
    )
    start = _journal_start(
        binding=binding, canonical_matrix_sha256=canonical_matrix_sha256,
        canonical_matrix_seal_sha256=canonical_matrix["matrix_seal_sha256"],
        preflight_relative_path="capture_preflight.json",
        preflight_sha256=preflight_sha256,
        source_commit=source_commit, admission_run_id=admission_run_id,
        attempt_ordinal=attempt_ordinal, attempt_id=attempt_id,
        prior_attempts=state["attempts"],
    )
    _write_json_new(attempt_dir / "journal_start.json", start)
    if state["attempts"]:
        recovery = _sealed({
            "schema_version": 1,
            "kind": "sol_workload_capture_attempt_recovery",
            "attempt_ordinal": attempt_ordinal,
            "attempt_id": attempt_id,
            "journal_start_seal_sha256": start[
                "journal_start_seal_sha256"
            ],
            "prior_attempts": copy.deepcopy(state["attempts"]),
            "prior_attempts_sha256": canonical_sha256(state["attempts"]),
        }, "recovery_seal_sha256")
        _write_json_new(attempt_dir / "recovery.json", recovery)
    cell_dir = attempt_dir / "cells" / safe_name
    _mkdir_nofollow(cell_dir, exist_ok=False)
    try:
        logical = (
            ((cell.get("invocation") or {}).get("logical_workload") or {})
        )
        external_workload_source = logical.get("external_workload_source")
        external_definition_source = logical.get("external_definition_source")
        if supplemental_workload_path is not None:
            if not isinstance(external_workload_source, dict):
                raise WorkloadCaptureRefusal(
                    "supplemental_attempt_input_identity_invalid",
                    f"workload: {external_workload_source!r}",
                )
            supplemental_workload_path = _materialize_attempt_workload(
                source=supplemental_workload_path,
                cell_dir=cell_dir,
                expected_sha256=external_workload_source.get("sha256"),
                expected_size=external_workload_source.get("size_bytes"),
            )
        elif external_workload_source is not None:
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_input_identity_invalid",
                "workload source has no retained path",
            )
        if supplemental_definition_path is not None:
            if not isinstance(external_definition_source, dict):
                raise WorkloadCaptureRefusal(
                    "supplemental_attempt_input_identity_invalid",
                    f"definition: {external_definition_source!r}",
                )
            supplemental_definition_path = _materialize_attempt_definition(
                source=supplemental_definition_path,
                cell_dir=cell_dir,
                expected_sha256=external_definition_source.get("sha256"),
                expected_size=external_definition_source.get("size_bytes"),
            )
        elif external_definition_source is not None:
            raise WorkloadCaptureRefusal(
                "supplemental_attempt_input_identity_invalid",
                "definition source has no retained path",
            )
        result = canonical_capture.capture_cell(
            cell=cell, cell_dir=cell_dir, matrix=canonical_matrix, pins=pins,
            cell_budgets=cell_budgets,
            expected_tracer_sha256=(
                preflight_manifest["checks"]["injection_runtime"]
                ["tracer_so_sha256"]
            ),
            capture_provenance={
                "protocol_version": pins["measurement_protocol"]["version"],
                "source_commit": source_commit,
                "capture_matrix_sha256": matrix_sha256,
                "preflight_sha256": preflight_sha256,
                "pins_sha256": pins_sha256,
                "trace_budget_sha256": cell_budget_sha256,
                "raw_staging_margin_bytes": raw_staging_margin_bytes,
                "capture_started_at_utc": start["capture_started_at_utc"],
                "admission_run_id": admission_run_id,
                "supersedes": None,
                "prior_preparation_failures": [],
                "prior_capture_defect_attempts": [],
            },
            expected_ncu_identity=expected_ncu_identity,
            certified_sol_env=certified_row_environment(
                cell=cell, certified_sol_env=certified_sol_env,
                flashinfer_proof=flashinfer_proof,
            ),
            sol_environment_probe=sol_environment_probe,
            module_store=output_root / "sol_workload_module_store",
            expected_sol_fingerprint_pin=expected_sol_fingerprint_pin,
            workload_row_binding=binding,
            supplemental_runnability_evidence=(
                supplemental_runnability_evidence
            ),
            runnability_path=runnability_path,
            supplemental_workload_path=supplemental_workload_path,
            supplemental_definition_path=supplemental_definition_path,
            launch_set_preflight_manifest=preflight_manifest,
            launch_set_registry=launch_set_registry,
        )
        _content_address_tree(
            source=cell_dir,
            object_store=(
                output_root / "sol_workload_object_store" / "sha256"
            ),
        )
        canonical_capture._verify_journal_artifacts(
            result, root=attempt_dir, safe_name=safe_name
        )
        retention, metadata, _raw = _retention_limits(aggregate_budget)
        observed_physical = _tree_bytes(output_root)
        if observed_physical + metadata > retention:
            raise WorkloadCaptureRefusal(
                "supplemental_retention_budget_exceeded",
                f"observed={observed_physical}; metadata_reserve={metadata}; "
                f"ceiling={retention}",
            )
        complete = _sealed({
            "schema_version": 1,
            "kind": "sol_workload_capture_journal_complete",
            "attempt_ordinal": attempt_ordinal,
            "attempt_id": attempt_id,
            "journal_start_seal_sha256": start[
                "journal_start_seal_sha256"
            ],
            "prior_attempts_sha256": canonical_sha256(state["attempts"]),
            "cell_result": result,
        }, "journal_complete_seal_sha256")
        _write_json_new(attempt_dir / "journal_complete.json", complete)
    except Exception as exc:
        # Never infer an output-volume failure from a nested raw input errno.
        # Local output primitives already emit STORAGE_* directly.  The sole
        # canonical mixed producer admitted here is its explicitly typed
        # monitored-trace storage refusal, whose OSError cause is preserved.
        storage_refusal = None
        if (
            isinstance(exc, WorkloadCaptureRefusal)
            and exc.code in {STORAGE_CAPACITY_CODE, STORAGE_INTEGRITY_CODE}
        ):
            storage_refusal = exc
        elif (
            isinstance(exc, canonical_capture.ChunkRefusal)
            and (
                exc.code.endswith("_storage_monitor_failed")
                or exc.code.endswith("_storage_failure")
            )
        ):
            storage_refusal = storage_refusal_from_exception(
                exc, operation="capture_monitored_output_tree"
            )
        if storage_refusal is not None:
            # Do not attempt another write to the same failed medium.  The
            # immutable start plus any partial bytes remain an interrupted
            # attempt and are reconciled on a later successful restart.
            raise storage_refusal from exc
        evidence_before = _attempt_tree_inventory(attempt_dir)
        failure = _sealed({
            "schema_version": 1,
            "kind": "sol_workload_capture_journal_failure",
            "attempt_ordinal": attempt_ordinal,
            "attempt_id": attempt_id,
            "journal_start_seal_sha256": start[
                "journal_start_seal_sha256"
            ],
            "failed_at_utc": _utc_text(),
            "failure_code": str(getattr(exc, "code", type(exc).__name__)),
            "failure_detail": str(getattr(exc, "detail", str(exc))),
            "evidence_tree_before_failure_record": evidence_before,
            "evidence_tree_before_failure_record_sha256": canonical_sha256(
                evidence_before
            ),
        }, "journal_failure_seal_sha256")
        _write_json_new(attempt_dir / "journal_failure.json", failure)
        # Failed raw traces are evidence too.  Preserve them losslessly and
        # content-address their bytes before sealing the attempt.  Admission
        # for the next retry accounts these unique physical bytes, so repeated
        # near-ceiling failures cannot grow past the campaign budget.
        _content_address_tree(
            source=attempt_dir,
            object_store=(
                output_root / "sol_workload_object_store" / "sha256"
            ),
        )
        retention, metadata, _raw = _retention_limits(aggregate_budget)
        observed_physical = _tree_bytes(output_root)
        _seal_attempt_tree(attempt_dir)
        if observed_physical + metadata > retention:
            raise WorkloadCaptureRefusal(
                "supplemental_retention_budget_exceeded",
                f"observed={observed_physical}; metadata_reserve={metadata}; "
                f"ceiling={retention}; original_failure="
                f"{getattr(exc, 'code', type(exc).__name__)}",
            ) from exc
        if isinstance(exc, WorkloadCaptureRefusal):
            raise
        if isinstance(exc, canonical_capture.ChunkRefusal):
            raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
        raise WorkloadCaptureRefusal(
            "supplemental_internal_capture_failure",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    _seal_attempt_tree(attempt_dir)
    return result


def _completed_attempt_dir(journal_dir: Path) -> Path:
    completed = [
        attempt_dir for _ordinal, attempt_dir in _attempt_directories(journal_dir)
        if (
            (metadata := _output_lstat(
                attempt_dir / "journal_complete.json",
                operation="completed_attempt_terminal_lstat",
            )) is not None
            and stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
        )
    ]
    if len(completed) != 1:
        raise WorkloadCaptureRefusal(
            "supplemental_completed_attempt_cardinality",
            f"{journal_dir}: {len(completed)}",
        )
    return completed[0]


def _publication_pending_path(
    *, stage_root: Path, chunk_id: str, source_commit: str,
    matrix_path: Path, matrix_seal_sha256: str,
    canonical_matrix_path: Path, canonical_matrix_seal_sha256: str,
) -> Path:
    identity = {
        "schema_version": 1,
        "kind": "sol_workload_chunk_publication_v1",
        "source_commit": source_commit,
        "chunk_id": chunk_id,
        "supplemental_matrix_sha256": sha256_file(matrix_path),
        "supplemental_matrix_seal_sha256": matrix_seal_sha256,
        "canonical_matrix_sha256": sha256_file(canonical_matrix_path),
        "canonical_matrix_seal_sha256": canonical_matrix_seal_sha256,
    }
    return (
        stage_root / "publications"
        / f"publication-{canonical_sha256(identity)}"
    )


def _finalize(
    *, output_root: Path, stage_root: Path, final_dir: Path,
    chunk: dict[str, Any],
    resolved: list[tuple[dict, dict, dict]], results: dict[str, dict],
    source_commit: str, matrix_path: Path, matrix: dict[str, Any],
    canonical_matrix_path: Path, canonical_matrix: dict[str, Any],
    authority_path: Path, pins_path: Path, aggregate_budget_path: Path,
    cell_budget_path: Path, registry_path: Path, preflight_path: Path,
    flashinfer_proof: dict[str, Any],
) -> Path:
    final_metadata = _output_lstat(
        final_dir, operation="published_chunk_directory_lstat",
    )
    if final_metadata is not None:
        if stat.S_ISLNK(final_metadata.st_mode) or not stat.S_ISDIR(
            final_metadata.st_mode
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_destination_exists", str(final_dir)
            )
        published = _load_output_object(
            final_dir / "DONE.json", role="supplemental DONE",
            operation="published_done_read",
        )
        verify_done_manifest(
            published, final_dir=final_dir, matrix=matrix,
            matrix_path=matrix_path, canonical_matrix=canonical_matrix,
            canonical_matrix_path=canonical_matrix_path,
            campaign_root=output_root,
        )
        _fsync_directory(final_dir.parent)
        return final_dir
    aggregate_budget = _load_object(
        aggregate_budget_path, role="aggregate retention budget"
    )
    retention, metadata, _raw = _retention_limits(aggregate_budget)
    if _tree_bytes(output_root) + metadata > retention:
        raise WorkloadCaptureRefusal(
            "supplemental_retention_budget_exceeded",
            "campaign exceeds ceiling before publication",
        )
    publication = _publication_pending_path(
        stage_root=stage_root, chunk_id=chunk["chunk_id"],
        source_commit=source_commit, matrix_path=matrix_path,
        matrix_seal_sha256=matrix["matrix_seal_sha256"],
        canonical_matrix_path=canonical_matrix_path,
        canonical_matrix_seal_sha256=canonical_matrix[
            "matrix_seal_sha256"
        ],
    )
    _mkdir_nofollow(publication, exist_ok=True)
    if publication.is_symlink() or not publication.is_dir():
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_invalid", str(publication)
        )
    expected_safe_names = {
        _safe_name(row["key"]) for row, _variant, _cell in resolved
    }
    allowed_root_names = {"cells", "DONE.json"}
    unexpected_root = {
        path.name for path in _output_directory_entries(
            publication, operation="publication_pending_entries",
        )
    } - allowed_root_names
    if unexpected_root:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_extra",
            repr(sorted(unexpected_root)),
        )
    cells_root = publication / "cells"
    _mkdir_nofollow(cells_root, exist_ok=True)
    if cells_root.is_symlink() or not cells_root.is_dir():
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_invalid", str(cells_root)
        )
    unexpected_cells = {
        path.name for path in _output_directory_entries(
            cells_root, operation="publication_cells_entries",
        )
    } - expected_safe_names
    if unexpected_cells:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_pending_extra",
            repr(sorted(unexpected_cells)),
        )
    ordered_results: list[dict] = []
    for row, _variant, _cell in resolved:
        safe_name = _safe_name(row["key"])
        result = results[safe_name]
        source = (
            _completed_attempt_dir(stage_root / "journals" / safe_name)
            / "cells" / safe_name
        )
        destination = cells_root / safe_name
        _resume_hardlink_tree(source, destination)
        _verify_artifacts(result, root=publication)
        ordered_results.append(result)
    head, clean, dirty = canonical_capture._git_state()
    if head != source_commit or not clean:
        raise WorkloadCaptureRefusal(
            "supplemental_source_state_changed",
            f"head={head}; dirty={dirty!r}",
        )
    done_path = publication / "DONE.json"
    done_metadata = _output_lstat(
        done_path, operation="pending_done_lstat",
    )
    if done_metadata is not None:
        if stat.S_ISLNK(done_metadata.st_mode) or not stat.S_ISREG(
            done_metadata.st_mode
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_invalid", str(done_path)
            )
        manifest = _load_output_object(
            done_path, role="pending supplemental DONE",
            operation="pending_done_read",
        )
        if manifest.get("cell_results") != ordered_results:
            raise WorkloadCaptureRefusal(
                "supplemental_publication_pending_mismatch", str(done_path)
            )
    else:
        manifest = build_done_manifest(
            chunk=chunk, results=ordered_results, source_commit=source_commit,
            matrix_path=matrix_path, matrix=matrix,
            canonical_matrix_path=canonical_matrix_path,
            canonical_matrix=canonical_matrix, authority_path=authority_path,
            pins_path=pins_path, aggregate_budget_path=aggregate_budget_path,
            cell_budget_path=cell_budget_path, registry_path=registry_path,
            preflight_path=preflight_path, final_dir=final_dir,
            command=shlex.join(sys.argv),
            flashinfer_proof=flashinfer_proof,
            stage_root=stage_root,
            publication_root=publication,
            campaign_root=output_root,
        )
        _write_json_new(done_path, manifest)
    observed_after_publication = _tree_bytes(output_root)
    if observed_after_publication + metadata > retention:
        raise WorkloadCaptureRefusal(
            "supplemental_retention_budget_exceeded",
            f"observed={observed_after_publication}; "
            f"metadata_reserve={metadata}; ceiling={retention}",
        )
    verify_done_manifest(
        manifest, final_dir=publication, matrix=matrix,
        matrix_path=matrix_path, canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_matrix_path,
        campaign_root=output_root, require_final_location=False,
    )
    _mkdir_nofollow(final_dir.parent, exist_ok=True)
    if _output_lstat(
        final_dir, operation="publication_destination_lstat",
    ) is not None:
        raise WorkloadCaptureRefusal(
            "supplemental_publication_destination_exists", str(final_dir)
        )
    _rename_noreplace(publication, final_dir)
    _fsync_directory(final_dir.parent)
    published = _load_output_object(
        final_dir / "DONE.json", role="supplemental DONE",
        operation="published_done_read_after_rename",
    )
    verify_done_manifest(
        published, final_dir=final_dir, matrix=matrix,
        matrix_path=matrix_path, canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_matrix_path,
        campaign_root=output_root,
    )
    return final_dir


def _run_chunk_impl(
    args: argparse.Namespace, *, campaign_lock_lease: _CampaignLockLease,
) -> Path:
    matrix_path = lexical_input_path(args.matrix)
    canonical_matrix_path = lexical_input_path(args.canonical_matrix)
    authority_path = lexical_input_path(args.authority)
    pins_path = lexical_input_path(args.pins)
    aggregate_budget_path = lexical_input_path(args.aggregate_budget)
    cell_budget_path = lexical_input_path(args.cell_budgets)
    registry_path = lexical_input_path(args.metric_registry)
    preflight_path = lexical_input_path(args.preflight)
    runnability_path = (
        lexical_input_path(args.runnability)
        if getattr(args, "runnability", None) else None
    )
    shrink_path = (
        lexical_input_path(args.shrink_provenance)
        if getattr(args, "shrink_provenance", None) else None
    )
    runnability_root = (
        Path(args.runnability_root)
        if getattr(args, "runnability_root", None) else None
    )
    shrink_root = (
        Path(args.shrink_root)
        if getattr(args, "shrink_root", None) else None
    )
    full_coverage_inventory_path = lexical_input_path(
        getattr(args, "full_coverage_inventory", "")
    )
    live_coverage_inventory_path = lexical_input_path(
        getattr(args, "live_coverage_inventory", "")
    )
    live_coverage_inventory_root = (
        Path(args.live_coverage_inventory_root)
        if getattr(args, "live_coverage_inventory_root", None) else None
    )
    output_root = Path(args.output_root)
    retained_input_mode = bool(
        getattr(args, "retained_input_mode", False)
    )

    # All repository-bound validation happens before the first external output
    # directory is created.  The canonical helper then enforces the external
    # root and supplies the campaign-wide interprocess lock.
    if (
        not getattr(args, "full_coverage_inventory", None)
        or not getattr(args, "live_coverage_inventory", None)
        or live_coverage_inventory_root is None
        or runnability_path is None or shrink_path is None
        or runnability_root is None or shrink_root is None
    ):
        raise WorkloadCaptureRefusal(
            "campaign_live_coverage_inputs_missing",
            "historical/live inventories, runnability/shrink artifacts, and "
            "owner roots are required",
        )
    retained_campaign_identity: dict[str, Any] | None = None
    if retained_input_mode:
        owner, _payload, _pin = _load_retained_input_owner(output_root)
        retained_campaign_identity = owner["campaign_identity"]
        owner_namespaces = retained_campaign_identity.get("owner_namespaces")
        if (
            not isinstance(owner_namespaces, dict)
            or owner_namespaces != getattr(args, "owner_namespaces", None)
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_retained_owner_namespace_identity_mismatch",
                str(output_root),
            )
        retained_roots = validate_retained_owner_namespaces(
            output_root, contracts=owner_namespaces
        )
        provided = {
            "runnability": runnability_root,
            "shrink": shrink_root,
            "plan": live_coverage_inventory_root,
        }
        if any(
            Path(provided[role]).resolve(strict=True)
            != retained_roots[role].resolve(strict=True)
            for role in provided
        ):
            raise WorkloadCaptureRefusal(
                "supplemental_retained_owner_root_mismatch", repr(provided)
            )
    else:
        owner_namespaces = None
        assert_output_disjoint_from_input_owners(
            output_root,
            input_roots={
                "runnability_root": runnability_root,
                "shrink_root": shrink_root,
                "live_coverage_inventory_root": live_coverage_inventory_root,
            },
        )
    assert_external_output_root_before_mutation(output_root)
    matrix = _load_object(matrix_path, role="supplemental workload matrix")
    representative_mode = is_representative_execution_matrix(matrix)
    if representative_mode:
        validate_representative_execution_matrix(
            matrix, matrix_path=matrix_path,
        )
        _live_coverage_inventory, _live_coverage_summary = (
            validate_representative_live_coverage_inventory(
                inventory_path=live_coverage_inventory_path,
                inventory_root=live_coverage_inventory_root,
                matrix=matrix,
            )
        )
    else:
        validate_supplemental_matrix(
            matrix, matrix_path=matrix_path,
            canonical_matrix_path=canonical_matrix_path,
            authority_path=authority_path,
            runnability_path=runnability_path,
            shrink_path=shrink_path,
            runnability_root=runnability_root,
            shrink_root=shrink_root,
        )
        _live_coverage_inventory, _live_coverage_summary = (
            validate_live_coverage_inventory(
                inventory_path=live_coverage_inventory_path,
                inventory_root=live_coverage_inventory_root,
                matrix=matrix, matrix_path=matrix_path,
                runnability_path=runnability_path,
                runnability_root=runnability_root,
                shrink_path=shrink_path, shrink_root=shrink_root,
            )
        )
    input_file_identity(
        full_coverage_inventory_path, role="full coverage inventory",
        source_commit=matrix["source_commit"],
    )
    input_file_identity(
        live_coverage_inventory_path, role="live coverage inventory",
        source_commit=matrix["source_commit"],
    )
    canonical_matrix = _load_object(
        canonical_matrix_path, role="canonical capture matrix"
    )
    # Read and shape-check every remaining CLI input before mkdir/flock.
    _load_object(preflight_path, role="campaign preflight")
    pins = _load_object(pins_path, role="campaign protocol pins")
    aggregate_budget = _load_object(
        aggregate_budget_path, role="aggregate retention budget"
    )
    registry_document = _load_object(
        registry_path, role="metric registry"
    )
    try:
        launch_set_registry = metric_registry.parse_registry(
            registry_document
        )
    except metric_registry.MetricRegistryError as exc:
        raise WorkloadCaptureRefusal(
            "supplemental_metric_registry_invalid", str(exc),
        ) from exc
    for path, role in (
        (authority_path, "capture authority"),
        (cell_budget_path, "per-cell trace budgets"),
    ):
        if path.is_symlink() or not path.is_file():
            raise WorkloadCaptureRefusal(
                "supplemental_input_not_regular", f"{role}: {path}"
            )
    _validate_loaded_canonical_pin(
        matrix["canonical_matrix"], canonical_matrix, canonical_matrix_path,
    )
    validate_frozen_canonical_base(
        canonical_matrix, canonical_matrix_path=canonical_matrix_path,
        authority_path=authority_path, supplemental_matrix=matrix,
    )
    chunk, resolved = select_workload_chunk(
        matrix, canonical_matrix, args.chunk_id
    )
    flashinfer_proof = verify_flashinfer_contracts(resolved)
    head = _require_live_supplemental_source_state(matrix)

    retention, _metadata, raw_staging_margin = _retention_limits(
        aggregate_budget
    )
    validate_representative_aggregate_budget(matrix, aggregate_budget)
    matrix_sha = sha256_file(matrix_path)
    canonical_matrix_sha = sha256_file(canonical_matrix_path)
    preflight_sha = sha256_file(preflight_path)
    bindings = {
        _safe_name(row["key"]): workload_row_binding(
            matrix=matrix, matrix_sha256=matrix_sha,
            chunk_id=args.chunk_id, row=row,
        )
        for row, _variant, _cell in resolved
    }
    admitted_at = datetime.now(timezone.utc)
    try:
        # This first pass is deliberately freshness-agnostic.  It binds every
        # preflight input and enables pure environment/row validation before
        # the campaign lock or any capture directory is created.  Freshness is
        # rechecked under the lock once resume state is known.
        preflight_manifest, _unadmitted_ncu_identity = (
            _validate_supplemental_preflight_reference(
                preflight_path=preflight_path,
                preflight_sha256=preflight_sha,
                pins=pins,
                pins_path=pins_path,
                aggregate_budget_path=aggregate_budget_path,
                supplemental_matrix=matrix,
                supplemental_matrix_path=matrix_path,
                supplemental_matrix_sha256=matrix_sha,
                output_root=output_root,
                expected_head=head,
                require_fresh=False,
                admitted_at_utc=admitted_at,
            )
        )
    except canonical_capture.ChunkRefusal as exc:
        raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
    sol_probe, certified_sol_env = _certify_environment(
        supplemental_matrix=matrix,
        supplemental_matrix_path=matrix_path,
        preflight_manifest=preflight_manifest,
    )
    cell_budgets = explicit_chunk_budgets(resolved)
    external_reduced_sources, reduced_roles_by_variant = (
        validate_external_reduced_sources(matrix, shrink_root=shrink_root)
    )

    root_campaign_identity = supplemental_campaign_identity(
        matrix=matrix, matrix_path=matrix_path,
        canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_matrix_path,
        authority_path=authority_path, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path,
        cell_budget_path=cell_budget_path, registry_path=registry_path,
        full_coverage_inventory_path=full_coverage_inventory_path,
        live_coverage_inventory_path=live_coverage_inventory_path,
        runnability_path=runnability_path, shrink_path=shrink_path,
        external_sources=external_reduced_sources,
        owner_namespaces=owner_namespaces,
    )
    if (
        retained_campaign_identity is not None
        and root_campaign_identity != retained_campaign_identity
    ):
        raise WorkloadCaptureRefusal(
            "supplemental_retained_campaign_identity_mismatch",
            str(output_root),
        )
    establish_supplemental_output_root(
        output_root, campaign_identity=root_campaign_identity
    )
    retain_campaign_inputs(
        output_root, campaign_identity=root_campaign_identity,
        sources=_campaign_input_sources(
            matrix_path=matrix_path,
            canonical_matrix_path=canonical_matrix_path,
            authority_path=authority_path, pins_path=pins_path,
            aggregate_budget_path=aggregate_budget_path,
            cell_budget_path=cell_budget_path, registry_path=registry_path,
            full_coverage_inventory_path=full_coverage_inventory_path,
            live_coverage_inventory_path=live_coverage_inventory_path,
            runnability_path=runnability_path, shrink_path=shrink_path,
            external_sources=external_reduced_sources,
        ),
    )
    reduced_workload_paths, reduced_definition_paths = (
        retained_reduced_input_paths(
            resolved=resolved, roles_by_variant=reduced_roles_by_variant,
            campaign_identity=root_campaign_identity,
            output_root=output_root,
        )
    )
    prevalidate_supplemental_rows(
        resolved=resolved, bindings=bindings, cell_budgets=cell_budgets,
        certified_sol_env=certified_sol_env,
        flashinfer_proof=flashinfer_proof,
        runnability_path=runnability_path,
        reduced_workload_paths=reduced_workload_paths,
        reduced_definition_paths=reduced_definition_paths,
    )
    canonical_capture._assert_external_output_root(output_root)
    try:
        campaign_lock_lease.acquire(output_root)
    except canonical_capture.ChunkRefusal as exc:
        if exc.code.endswith("_storage_failure"):
            mapped = storage_refusal_from_exception(
                exc, operation="campaign_chunk_lock",
            )
            if mapped is not None:
                raise mapped from exc
        raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
    final_dir = output_root / "sol_workload_chunks" / args.chunk_id
    if _output_lstat(
        final_dir, operation="resume_chunk_directory_lstat",
    ) is not None:
        done = _load_output_object(
            final_dir / "DONE.json", role="supplemental DONE",
            operation="resume_done_read",
        )
        verify_done_manifest(
            done, final_dir=final_dir, matrix=matrix,
            matrix_path=matrix_path, canonical_matrix=canonical_matrix,
            canonical_matrix_path=canonical_matrix_path,
        )
        return final_dir

    stage_root = (
        output_root / ".sol_workload_staging" / head / args.chunk_id
        / matrix_sha
    )
    journals_root = stage_root / "journals"
    completed: dict[str, dict] = {}
    for safe_name, binding in bindings.items():
        loaded = _load_completed_journal(
            journal_dir=journals_root / safe_name,
            expected_binding=binding,
            canonical_matrix_sha256=canonical_matrix_sha,
            canonical_matrix_seal_sha256=canonical_matrix["matrix_seal_sha256"],
            preflight_sha256=preflight_sha, source_commit=head,
        )
        if loaded is not None:
            completed[safe_name] = loaded[1]
    all_complete = len(completed) == len(resolved)
    try:
        preflight_manifest, expected_ncu_identity = (
            _validate_supplemental_preflight_reference(
                preflight_path=preflight_path,
                preflight_sha256=preflight_sha,
                pins=pins,
                pins_path=pins_path,
                aggregate_budget_path=aggregate_budget_path,
                supplemental_matrix=matrix,
                supplemental_matrix_path=matrix_path,
                supplemental_matrix_sha256=matrix_sha,
                output_root=output_root,
                expected_head=head,
                require_fresh=not all_complete,
                admitted_at_utc=admitted_at,
            )
        )
    except canonical_capture.ChunkRefusal as exc:
        raise WorkloadCaptureRefusal(exc.code, exc.detail) from exc
    if all_complete:
        return _finalize(
            output_root=output_root, stage_root=stage_root,
            final_dir=final_dir, chunk=chunk,
            resolved=resolved, results=completed, source_commit=head,
            matrix_path=matrix_path, matrix=matrix,
            canonical_matrix_path=canonical_matrix_path,
            canonical_matrix=canonical_matrix, authority_path=authority_path,
            pins_path=pins_path, aggregate_budget_path=aggregate_budget_path,
            cell_budget_path=cell_budget_path, registry_path=registry_path,
            preflight_path=preflight_path,
            flashinfer_proof=flashinfer_proof,
        )
    stable_budget_failures: list[dict[str, Any]] = []
    stable_budget_codes = {
        "trace_target_failed_tree_limit_exceeded",
        "supplemental_full_trace_retention_ceiling_exceeded",
    }
    for row, variant, cell in resolved:
        binding = bindings[_safe_name(row["key"])]
        safe_name = _safe_name(row["key"])
        if safe_name in completed:
            continue
        budget = canonical_capture.require_cell_budget(
            cell_budgets, cell["identity"]["slug"]
        )
        prior_state = _scan_journal_attempts(
            journal_dir=journals_root / safe_name,
            expected_binding=binding,
            canonical_matrix_sha256=canonical_matrix_sha,
            canonical_matrix_seal_sha256=canonical_matrix[
                "matrix_seal_sha256"
            ],
            preflight_sha256=preflight_sha, source_commit=head,
        )
        prior_failure = prior_state.get("terminal_failure") or {}
        prior_code = prior_failure.get("failure_code")
        if prior_code in stable_budget_codes:
            stable_budget_failures.append({
                "row_key": copy.deepcopy(row["key"]),
                "failure_code": prior_code,
                "failure_detail": prior_failure.get("failure_detail"),
                "attempt_ordinal": prior_failure.get("attempt_ordinal"),
            })
            continue
        _require_retention_capacity(
            output_root=output_root, aggregate_budget=aggregate_budget,
            next_retained_max=budget.max_trace_bytes,
        )
        try:
            result = _capture_one(
                journal_dir=journals_root / safe_name, cell=cell,
                binding=binding, canonical_matrix=canonical_matrix, pins=pins,
                supplemental_runnability_evidence=(
                    variant.get("runnability_evidence")
                    if variant.get("cohort") in {
                        "native", "partial_native_row"
                    } else None
                ),
                cell_budgets=cell_budgets,
                preflight_manifest=preflight_manifest,
                expected_ncu_identity=expected_ncu_identity,
                launch_set_registry=launch_set_registry,
                preflight_path=preflight_path, preflight_sha256=preflight_sha,
                matrix_sha256=matrix_sha,
                cell_budget_sha256=canonical_sha256(variant["trace_budget"]),
                raw_staging_margin_bytes=raw_staging_margin,
                source_commit=head, sol_environment_probe=sol_probe,
                certified_sol_env=certified_sol_env, output_root=output_root,
                aggregate_budget=aggregate_budget,
                flashinfer_proof=flashinfer_proof,
                canonical_matrix_sha256=canonical_matrix_sha,
                expected_sol_fingerprint_pin=(
                    matrix["input_pins"]["sol_native_env_fingerprint"]
                ),
                pins_sha256=sha256_file(pins_path),
                runnability_path=runnability_path,
                supplemental_workload_path=reduced_workload_paths.get(
                    variant.get("variant_contract_sha256")
                ),
                supplemental_definition_path=reduced_definition_paths.get(
                    variant.get("variant_contract_sha256")
                ),
            )
        except (WorkloadCaptureRefusal, OSError) as caught:
            if isinstance(caught, OSError):
                # Output boundaries inside _capture_one type locally.  An
                # untyped raw errno may be from an immutable input and must
                # never become a campaign-volume receipt.
                raise
            exc = caught
            if exc.code not in stable_budget_codes:
                raise WorkloadCaptureRefusal(
                    exc.code,
                    json.dumps({
                        "row_key": copy.deepcopy(row["key"]),
                        "cause_detail": exc.detail,
                    }, sort_keys=True),
                ) from exc
            stable_budget_failures.append({
                "row_key": copy.deepcopy(row["key"]),
                "failure_code": exc.code,
                "failure_detail": exc.detail,
                "attempt_ordinal": prior_state["next_ordinal"],
            })
            continue
        if _tree_bytes(output_root) > retention:
            raise WorkloadCaptureRefusal(
                "supplemental_retention_budget_exceeded",
                f"observed={_tree_bytes(output_root)}; ceiling={retention}",
            )
        completed[safe_name] = result
    if stable_budget_failures:
        raise WorkloadCaptureRefusal(
            "supplemental_chunk_rows_budget_blocked",
            json.dumps(stable_budget_failures, sort_keys=True),
        )
    canonical_capture.live_capture_guard(pins)
    return _finalize(
        output_root=output_root, stage_root=stage_root,
        final_dir=final_dir, chunk=chunk,
        resolved=resolved, results=completed, source_commit=head,
        matrix_path=matrix_path, matrix=matrix,
        canonical_matrix_path=canonical_matrix_path,
        canonical_matrix=canonical_matrix, authority_path=authority_path,
        pins_path=pins_path, aggregate_budget_path=aggregate_budget_path,
        cell_budget_path=cell_budget_path, registry_path=registry_path,
        preflight_path=preflight_path,
        flashinfer_proof=flashinfer_proof,
    )


def run_chunk(args: argparse.Namespace) -> Path:
    """Run one chunk while releasing its writer lock on every exit path."""
    campaign_lock_lease = _CampaignLockLease()
    try:
        return _run_chunk_impl(
            args, campaign_lock_lease=campaign_lock_lease,
        )
    finally:
        campaign_lock_lease.close(
            preserve_active_failure=sys.exc_info()[0] is not None,
        )


def main(argv: list[str] | None = None) -> int:
    # This module is intentionally import-only.  Only the campaign controller
    # may call ``run_chunk`` because it holds the canonical host-wide GPU lock
    # for the complete campaign lifetime.  A direct CLI would create an easy
    # lock-bypass path even if all of its file inputs were otherwise valid.
    del argv
    print(
        "sol_workload_capture_chunk: REFUSED "
        "[supplemental_chunk_internal_only] invoke "
        "sol_workload_capture_campaign.py",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
