#!/usr/bin/env python3
"""Run one immutable sm_86 reference-capture chunk.

The full-roster matrix is the authority for chunk membership and invocation.
This consumer refuses stale inputs, a dirty/moved repository, an unadmitted
preflight, active competing capture processes, clock drift, binary drift,
empty/vacuously-valid traces, incomplete full-registry labels, and unsealed
artifacts.  Work is staged outside the repository and atomically renamed only
after the exact DONE manifest verifies; an existing finalized chunk is skipped
only after the same verifier rehashes it.

Direct executables and SOL-ExecBench workloads use distinct, fail-closed
adapters.  A SOL solution JSON remains a pinned workload reference rather than
being mistaken for either the profiled launcher or a static control binary.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LABELS = REPO_ROOT / "tools" / "local_labels"
NVBIT_TOOLS = REPO_ROOT / "tools" / "nvbit_tracer"
for _path in (REPO_ROOT, LOCAL_LABELS, NVBIT_TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import campaign_preflight as preflight  # noqa: E402
import archive_preparation_failure as prep_archive  # noqa: E402
import archive_capture_defect_attempt as capture_defect_archive  # noqa: E402
import capture_budget_route as budget_route  # noqa: E402
import capture_toolchain_contract as toolchain_contract  # noqa: E402
from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import ncu_action_executor as ncu_executor  # noqa: E402
from tools.local_labels import ncu_collection_receipt as collection_receipt  # noqa: E402
from tools.local_labels import ncu_details_export_persistence as details_persistence  # noqa: E402
from tools.local_labels import ncu_launch_documents  # noqa: E402
from tools.local_labels import ncu_launch_sidecars  # noqa: E402
from tools.local_labels import parse_ncu_metrics  # noqa: E402
from tools.local_labels import sidecar_schema_authority  # noqa: E402
import ncu_capture_contract  # noqa: E402
import sol_capture_environment as sol_env  # noqa: E402
import sol_definition_equivalence as sol_def  # noqa: E402
import sol_reconstitution as sol_recon  # noqa: E402
from build_sm86_capture_matrix import (  # noqa: E402
    SOL_CONTROL_OUTCOMES,
    SOL_STATIC_CONTROL_REBIND,
    tree_sha256,
    validate_matrix as rederive_capture_matrix,
)
from import_trace_contract import validate_trace_dir  # noqa: E402
from repeatability_matrix_runner import registry_unit_metrics  # noqa: E402
from tools.cloud.trace_budgets import (  # noqa: E402
    load_trace_budgets,
    require_cell_budget,
)
from tools.sass_control import extractor as sass_extractor  # noqa: E402

DEFAULT_MATRIX = (
    REPO_ROOT / "data" / "local_labels" / "campaign"
    / "sm86_reference_capture_matrix.json"
)
DEFAULT_AUTHORITY = (
    REPO_ROOT / "data" / "local_labels" / "campaign"
    / "capture_roster_authority_sm86.json"
)
DEFAULT_PINS = LOCAL_LABELS / "campaign_protocol_pins.json"
DEFAULT_AGGREGATE_BUDGET = (
    REPO_ROOT / "data" / "local_labels" / "campaign"
    / "trace_retention_budget.json"
)
DEFAULT_CELL_BUDGETS = REPO_ROOT / "workloads" / "simbench_bw" / "trace_budgets.toml"
DEFAULT_REGISTRY = LOCAL_LABELS / "metric_registry.json"
MATRIX_SCHEMA = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "sm86_capture_matrix.schema.json"
)
LABEL_SCHEMA = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "local_label_sidecar.schema.json"
)
LABEL_DRIVER = LOCAL_LABELS / "label_driver.sh"
CONTROL_EMITTER = REPO_ROOT / "tools" / "sass_control" / "emit_control_bits_sidecar.py"

_DIRECT_ADAPTERS = {
    "direct_binary", "direct_executable", "portable_direct", "simbench_direct"
}
_VENDOR_RUNTIME_CONTROL_SLUGS = frozenset({
    "vendor_gemm_cublas",
    "vendor_sgemm_cublas",
    "vendor_sgemm_strided_batched_cublas",
})
_SHA40 = frozenset("0123456789abcdef")

ROUND28_PROVENANCE_REFUSAL_CODES = frozenset({
    "PREFLIGHT_CLOCK_SKEW",
    "PREFLIGHT_EXPIRED",
    "capture_ncu_authority_invalid",
    "REFERENCED_PREFLIGHT_INVALID",
    "JOURNAL_RECOVERY_NOT_ADMITTED",
    "JOURNAL_PREFLIGHT_MISMATCH",
    "JOURNAL_PREFLIGHT_SUPERSEDABLE",
    "CELL_JOURNAL_INCOMPLETE",
    "SUPERSESSION_CROSS_DEVICE_REFUSED",
    "SUPERSESSION_DESTINATION_EXISTS",
    "SUPERSESSION_INVENTORY_INVALID",
    "SUPERSESSION_INVENTORY_MISMATCH",
    "SUPERSESSION_LEDGER_INCONSISTENT",
    "SUPERSESSION_LIMIT_EXCEEDED",
    "PREPARATION_ATTEMPT_LIMIT",
    "PREPARATION_LEDGER_INCONSISTENT",
    "PREPARATION_ARCHIVE_LEAK",
    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    "CAPTURE_DEFECT_ARCHIVE_LEAK",
    "CROSS_HEAD_ACTIVE_JOURNAL_UNRESOLVED",
    "SOL_ATTESTATION_RULE_MISMATCH",
    "SOL_ATTESTATION_STATE_MISMATCH",
    "SOL_ATTESTATION_POINTER_INELIGIBLE",
    "sol_definition_parse_error",
    "sol_definition_duplicate_key",
    "sol_definition_nonfinite",
    "sol_stage_source_copy_missing",
    "sol_stage_source_copy_mismatch",
    "sol_stage_source_mismatch",
    "CHUNK_RUN_JOURNAL_INVALID",
    "DONE_PREFLIGHT_NONUNIFORM",
    "DONE_ADMISSION_MISSING",
    "DONE_ADMISSION_INCONSISTENT",
    "FINALIZATION_JOURNALS_INCOMPLETE",
})


class ChunkRefusal(RuntimeError):
    """Fail-closed campaign refusal with a stable machine-readable code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


_STORAGE_FAILURE_ERRNOS = frozenset({
    errno.ENOSPC,
    errno.EDQUOT,
    errno.EIO,
    errno.EROFS,
    *(
        value for value in (
            getattr(errno, "ENXIO", None),
            getattr(errno, "ESTALE", None),
        ) if value is not None
    ),
})
LABEL_DRIVER_STORAGE_MARKER = "LABEL_DRIVER_STORAGE_FAILURE:"


def _storage_origin(exc: BaseException | None) -> OSError | None:
    """Return the closed storage-origin error through nested wrappers."""
    current = exc
    while current is not None:
        if isinstance(current, OSError) and current.errno in (
            _STORAGE_FAILURE_ERRNOS
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


def _is_explicit_output_storage_refusal(exc: BaseException | None) -> bool:
    return (
        isinstance(exc, ChunkRefusal)
        and (
            exc.code.endswith("_storage_failure")
            or exc.code.endswith("_storage_monitor_failed")
        )
    )


def _raise_typed_storage_failure(
    exc: OSError, *, code: str, operation: str,
) -> None:
    """Raise a canonical typed refusal only for the closed storage errnos."""
    origin = _storage_origin(exc)
    if origin is None and exc.errno is None:
        # Only callers at an explicit output mutation boundary use this
        # helper; a short/errno-less write is therefore integrity failure.
        origin = exc
    if origin is not None:
        raise ChunkRefusal(
            code, f"operation={operation}; errno={origin.errno}; {origin}",
        ) from origin


def _label_driver_storage_error(stderr: bytes) -> OSError | None:
    """Parse one explicit child output-storage marker, never arbitrary errno."""
    matches = [
        line[len(LABEL_DRIVER_STORAGE_MARKER):]
        for line in stderr.decode("utf-8", errors="replace").splitlines()
        if line.startswith(LABEL_DRIVER_STORAGE_MARKER)
    ]
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "errno", "operation"}
        or value.get("schema_version") != 1
        or value.get("errno") not in _STORAGE_FAILURE_ERRNOS
        or value.get("operation") != "label_driver_output_probe"
    ):
        return None
    return OSError(value["errno"], "label driver reported output storage failure")


def _mkdir_output(
    path: Path, *, parents: bool, exist_ok: bool,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _write_output_bytes(
    path: Path, payload: bytes, *,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    """Write+fsync one output; stop before any subsequent output on fault."""
    try:
        with path.open("wb") as stream:
            written = stream.write(payload)
            if written != len(payload):
                raise OSError(None, "short capture output write")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _write_output_bytes_exclusive(
    path: Path, payload: bytes, *,
    code: str = "capture_output_storage_failure", operation: str,
    collision_code: str | None = None,
) -> None:
    """Create+fsync one output without following or replacing an existing leaf."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    active_error: BaseException | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(None, "short exclusive capture output write")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except OSError as exc:
        active_error = exc
        if collision_code is not None and exc.errno in {errno.EEXIST, errno.ELOOP}:
            raise ChunkRefusal(collision_code, str(path)) from exc
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                if active_error is None:
                    _raise_typed_storage_failure(
                        exc, code=code, operation=f"{operation}_close",
                    )
                    raise


def _write_output_text(
    path: Path, payload: str, *,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    _write_output_bytes(
        path, payload.encode("utf-8"), code=code, operation=operation,
    )


def _replace_output(
    source: Path, destination: Path, *,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    try:
        os.replace(source, destination)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _rmtree_output(
    path: Path, *, code: str = "capture_output_storage_failure",
    operation: str,
) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _link_output(
    source: Path, destination: Path, *,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _chmod_output(
    path: Path, mode: int, *, code: str = "capture_output_storage_failure",
    operation: str,
) -> None:
    try:
        path.chmod(mode)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _unlink_output(
    path: Path, *, missing_ok: bool,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    try:
        path.unlink(missing_ok=missing_ok)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _copy_output_file(
    source: Path, destination: Path, *, exclusive: bool,
    code: str = "capture_output_storage_failure", operation: str,
) -> None:
    """Stream input to one output while keeping input read errors untyped."""
    source_stream = source.open("rb")
    active_error: BaseException | None = None
    try:
        try:
            destination_stream = destination.open("xb" if exclusive else "wb")
        except OSError as exc:
            _raise_typed_storage_failure(
                exc, code=code, operation=f"{operation}_open",
            )
            raise
        destination_error: BaseException | None = None
        try:
            while True:
                # This is an input read.  It deliberately remains outside an
                # output-error catch so EIO here cannot create a storage receipt.
                block = source_stream.read(1 << 20)
                if not block:
                    break
                try:
                    written = destination_stream.write(block)
                    if written != len(block):
                        raise OSError(None, "short capture output copy write")
                except OSError as exc:
                    _raise_typed_storage_failure(
                        exc, code=code, operation=f"{operation}_write",
                    )
                    raise
            try:
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
            except OSError as exc:
                _raise_typed_storage_failure(
                    exc, code=code, operation=f"{operation}_fsync",
                )
                raise
        except BaseException as exc:
            destination_error = exc
            raise
        finally:
            try:
                destination_stream.close()
            except OSError as exc:
                if destination_error is None:
                    _raise_typed_storage_failure(
                        exc, code=code, operation=f"{operation}_close",
                    )
                    raise
                # Never let a secondary close fault replace the first output
                # failure (or promote an active raw input-read failure).
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        try:
            source_stream.close()
        except OSError:
            if active_error is None:
                # Source close belongs to the input side and remains raw.
                raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_output_bytes(
    path: Path, *, code: str = "capture_output_storage_failure",
    operation: str,
) -> bytes:
    """Read a campaign-produced artifact with closed storage typing."""
    try:
        return path.read_bytes()
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _json_output(
    path: Path, *, code: str = "capture_output_storage_failure",
    operation: str,
) -> dict:
    value = json.loads(_read_output_bytes(
        path, code=code, operation=operation,
    ))
    if not isinstance(value, dict):
        raise ChunkRefusal("input_not_object", f"{path} root is not an object")
    return value


def _sha256_output_file(
    path: Path, *, code: str = "capture_output_storage_failure",
    operation: str,
) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise
    return digest.hexdigest()


def _output_lstat(
    path: Path, *, code: str = "capture_output_storage_failure",
    operation: str,
) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _output_tree_paths(
    root: Path, *, code: str = "capture_output_storage_failure",
    operation: str, pattern: str = "*",
) -> list[Path]:
    """Walk an output tree without pathlib's suppressed scandir failures."""
    paths: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                observed = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            _raise_typed_storage_failure(
                exc, code=code, operation=f"{operation}_scandir",
            )
            raise
        for entry in observed:
            path = Path(entry.path)
            if Path(entry.name).match(pattern):
                paths.append(path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                _raise_typed_storage_failure(
                    exc, code=code, operation=f"{operation}_entry_stat",
                )
                raise
            if is_directory:
                walk(path)

    walk(root)
    return paths


def _resolve_output_path(
    path: Path, *, code: str = "capture_output_storage_failure",
    operation: str, strict: bool = True,
) -> Path:
    try:
        return path.resolve(strict=strict)
    except OSError as exc:
        _raise_typed_storage_failure(exc, code=code, operation=operation)
        raise


def _read_owned_regular_nofollow(path: Path, *, role: str) -> tuple[bytes, os.stat_result]:
    """Read one immutable external input without following any path symlink."""
    absolute = Path(path).absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise ChunkRefusal("input_path_unreadable", f"{role}: {path}")
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open("/", directory_flags)
    descriptor: int | None = None
    try:
        for component in absolute.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
            or observed.st_mode & 0o222
        ):
            raise ChunkRefusal(
                "sol_external_workload_not_immutable", f"{role}: {absolute}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks), observed
    except OSError as exc:
        raise ChunkRefusal(
            "sol_external_workload_path_invalid", f"{role}: {absolute}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ChunkRefusal("input_not_object", f"{path} root is not an object")
    return value


def _resolve_repo_path(value: str, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ChunkRefusal("input_path_missing", f"{role} path is absent")
    path = Path(value)
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ChunkRefusal("input_path_unreadable", f"{role}: {path}: {exc}") from exc
    return resolved


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ChunkRefusal(
            "input_hash_mismatch", f"{role}: expected {expected}, observed {actual}: {path}"
        )


def _git_state(repo_root: Path = REPO_ROOT) -> tuple[str, bool, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        capture_output=True, text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "-uall"], cwd=repo_root,
        capture_output=True, text=True,
    )
    if head.returncode != 0 or status.returncode != 0:
        raise ChunkRefusal(
            "git_state_unavailable",
            f"rev-parse rc={head.returncode}; status rc={status.returncode}",
        )
    dirty = status.stdout.strip()
    return head.stdout.strip(), not dirty, dirty


def _is_sha40(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _SHA40


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA40


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00"
            if isinstance(value, str) and value.endswith("Z")
            else value
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise ChunkRefusal(
            "preflight_invalid", f"{field} is invalid: {value!r} ({exc})"
        ) from exc
    if parsed.tzinfo is None:
        raise ChunkRefusal(
            "preflight_invalid", f"{field} has no timezone: {value!r}"
        )
    return parsed.astimezone(timezone.utc)


def validate_preflight_manifest(
    manifest: dict, *, expected_head: str, expected_pins_sha256: str,
    expected_output_root: Path, expected_ncu_version_prefix: str,
    expected_budget_sha256: str | None = None,
    expected_arch: str = "sm_86", expected_sku: str = "rtx_3090_ti",
    expected_protocol: str = "v2.1",
    expected_matrix_sha256: str | None = None,
    expected_sol_fingerprint_pin: dict | None = None,
    require_fresh: bool = True,
    admitted_at_utc: datetime | None = None,
) -> ncu_executor.ExpectedNcuIdentity | None:
    problems: list[str] = []
    try:
        normalized_ncu_version = (
            toolchain_contract.normalized_ncu_version_prefix(
                expected_ncu_version_prefix
            )
        )
    except toolchain_contract.CaptureToolchainContractError as exc:
        normalized_ncu_version = None
        problems.append(f"expected NCU version authority is invalid: {exc}")
    schema_version = manifest.get("schema_version")
    schema_path: Path | None = None
    if schema_version == 3:
        schema_path = (
            REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
            / "campaign_preflight_manifest.schema.json"
        )
    elif schema_version == 2:
        schema_path = (
            REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
            / "campaign_preflight_manifest.v2.schema.json"
        )
        if require_fresh:
            problems.append(
                "schema_version 2 is historical-only and cannot authorize capture"
            )
    else:
        problems.append(f"unknown preflight schema_version: {schema_version!r}")
    if schema_path is not None:
        try:
            import jsonschema
        except Exception:
            problems.append("schema validation could not be initialized")
        else:
            try:
                schema = json.loads(schema_path.read_text())
                jsonschema.validate(manifest, schema)
            except jsonschema.ValidationError as exc:
                safe_path = ".".join(
                    str(part)
                    for part in exc.absolute_path
                    if (
                        type(part) is int
                        or (
                            type(part) is str
                            and re.fullmatch(r"[A-Za-z0-9_]{1,64}", part)
                        )
                    )
                )
                validator = (
                    exc.validator
                    if type(exc.validator) is str
                    and re.fullmatch(r"[A-Za-z0-9_]{1,64}", exc.validator)
                    else "unknown"
                )
                problems.append(
                    "schema validation failed at "
                    f"{safe_path or '<root>'}: {validator}"
                )
            except Exception:
                problems.append("schema validation could not be completed")
    if manifest.get("capture_admitted") is not True:
        problems.append("capture_admitted is not true")
    if manifest.get("working_tree_clean") is not True:
        problems.append("working_tree_clean is not true")
    if manifest.get("git_head") != expected_head:
        problems.append(
            f"git_head={manifest.get('git_head')!r} != {expected_head!r}"
        )
    if manifest.get("pins_sha256") != expected_pins_sha256:
        problems.append("pins_sha256 does not match the matrix")
    if (
        expected_matrix_sha256 is not None
        and manifest.get("capture_matrix_sha256") != expected_matrix_sha256
    ):
        problems.append("capture_matrix_sha256 does not match the matrix")
    if manifest.get("arch") != expected_arch:
        problems.append(f"arch={manifest.get('arch')!r} != {expected_arch!r}")
    if manifest.get("sku") != expected_sku:
        problems.append(f"sku={manifest.get('sku')!r} != {expected_sku!r}")
    if manifest.get("protocol_version") != expected_protocol:
        problems.append(
            f"protocol_version={manifest.get('protocol_version')!r} "
            f"!= {expected_protocol!r}"
        )
    if manifest.get("blockers") != []:
        problems.append(f"blockers is not empty: {manifest.get('blockers')!r}")
    generated_raw = manifest.get("generated_at_utc")
    captured_raw = manifest.get("captured_at_utc")
    if generated_raw != captured_raw:
        problems.append("generated_at_utc and captured_at_utc differ")
    checks = manifest.get("checks")
    if not isinstance(checks, dict):
        problems.append("checks is not an object")
    else:
        for name in preflight._BLOCKING_CHECKS:
            if (
                name == "device"
                and not require_fresh
                and name not in checks
            ):
                continue
            probe = checks.get(name)
            if not isinstance(probe, dict) or probe.get("status") != preflight.STATUS_OK:
                problems.append(f"checks.{name}.status is not ok")
        device = checks.get("device")
        device_required = {
            "status",
            "selection_policy",
            "selected_host_gpu_index",
            "nvidia_smi_device_count",
            "gpu_uuid_sha256",
        }
        if device is None:
            if require_fresh:
                problems.append("checks.device authority is absent")
        elif not isinstance(device, dict):
            problems.append("checks.device is not an object")
        else:
            if set(device) != device_required:
                problems.append("checks.device fields differ")
            if device.get("selection_policy") != (
                preflight.DEVICE_SELECTION_POLICY
            ):
                problems.append("checks.device selection policy is invalid")
            if type(device.get("selected_host_gpu_index")) is not int or (
                device.get("selected_host_gpu_index") != 0
            ):
                problems.append("checks.device selected host GPU index is invalid")
            if type(device.get("nvidia_smi_device_count")) is not int or (
                device.get("nvidia_smi_device_count") != 1
            ):
                problems.append("checks.device host GPU count is invalid")
            if not _is_sha256(device.get("gpu_uuid_sha256")):
                problems.append("checks.device GPU UUID SHA-256 is invalid")
        ncu = checks.get("ncu")
        if not isinstance(ncu, dict):
            problems.append("checks.ncu is not an object")
        else:
            version_line = ncu.get("version_line")
            resolved_path = ncu.get("resolved_path")
            launcher_path = ncu.get("launcher_resolved_path")
            launcher_target = ncu.get("launcher_target_relpath")
            if (
                normalized_ncu_version is not None
                and ncu.get("expected") != normalized_ncu_version
            ):
                problems.append("checks.ncu.expected does not match the pins")
            if ncu.get("observed") != version_line:
                problems.append("checks.ncu.observed and version_line differ")
            if (
                type(resolved_path) is not str
                or not Path(resolved_path).is_absolute()
                or os.path.normpath(resolved_path) != resolved_path
            ):
                problems.append("checks.ncu.resolved_path is not canonical absolute")
            if not _is_sha256(ncu.get("executable_sha256")):
                problems.append("checks.ncu.executable_sha256 is invalid")
            if schema_version == 3:
                if ncu.get("execution_mode") != preflight.NCU_EXECUTION_MODE:
                    problems.append("checks.ncu.execution_mode is invalid")
                if ncu.get("host_machine") != preflight.NCU_HOST_MACHINE:
                    problems.append("checks.ncu.host_machine is invalid")
                if (
                    type(launcher_path) is not str
                    or not Path(launcher_path).is_absolute()
                    or os.path.normpath(launcher_path) != launcher_path
                ):
                    problems.append(
                        "checks.ncu.launcher_resolved_path is not canonical absolute"
                    )
                if not _is_sha256(ncu.get("launcher_sha256")):
                    problems.append("checks.ncu.launcher_sha256 is invalid")
                if launcher_target != preflight.NCU_LAUNCHER_TARGET_RELPATH:
                    problems.append("checks.ncu.launcher_target_relpath is invalid")
                if type(launcher_path) is str and type(launcher_target) is str:
                    expected_inner = str(Path(launcher_path).parent / launcher_target)
                    if resolved_path != expected_inner:
                        problems.append(
                            "checks.ncu inner path does not match launcher target"
                        )
            if (
                type(version_line) is not str
                or version_line != version_line.strip()
                or normalized_ncu_version is None
                or not toolchain_contract.ncu_version_matches(
                    version_line, normalized_ncu_version,
                )
            ):
                problems.append("checks.ncu.version_line does not match the pins")
        budget_root = (checks.get("trace_budget") or {}).get("output_root")
        budget_sha = (checks.get("trace_budget") or {}).get("budget_sha256")
        if expected_budget_sha256 is not None and budget_sha != expected_budget_sha256:
            problems.append(
                f"preflight budget_sha256={budget_sha!r} != "
                f"{expected_budget_sha256!r}"
            )
        try:
            observed_root = Path(budget_root).resolve() if budget_root else None
        except OSError:
            observed_root = None
        if observed_root != expected_output_root.resolve():
                problems.append(
                    f"preflight budget output_root={budget_root!r} does not bind "
                    f"capture root {str(expected_output_root)!r}"
                )
        injection = checks.get("injection_runtime") or {}
        coverage_status, coverage_detail = preflight.full_trace_coverage_verdict(
            root_launch_count=injection.get("root_trace_launch_count"),
            grid_threadblocks=injection.get("grid_threadblock_count"),
            decoded_threadblocks=injection.get("decoded_threadblock_count"),
            threadblock_pb_count=injection.get("threadblocks_pb_count"),
            decoded_records=injection.get("decoded_record_count"),
            expected_records=injection.get("expected_insts"),
            admitted_records=injection.get("admitted_insts"),
            stats_row_count=injection.get("stats_row_count"),
            sampling_sidecar_present=injection.get("sampling_sidecar_present"),
            zero_record_threadblocks=injection.get(
                "zero_record_threadblock_count"),
        )
        if coverage_status != preflight.STATUS_OK:
            problems.append(
                f"preflight full-coverage evidence is contradictory: {coverage_detail}"
            )
        sol_check = checks.get("sol_environment") or {}
        if expected_matrix_sha256 is not None:
            if sol_check.get("matrix_sha256") != expected_matrix_sha256:
                problems.append(
                    "preflight SOL environment matrix hash does not match"
                )
            observed = sol_check.get("observed_fingerprint") or {}
            if (
                observed.get("degraded") is not False
                or observed.get("missing") != []
            ):
                problems.append("preflight SOL environment is degraded")
            compiler = sol_check.get("compiler_identity") or {}
            if compiler.get("requested") != "g++-14":
                problems.append("preflight SOL compiler is not literal g++-14")
        if expected_sol_fingerprint_pin is not None:
            if sol_check.get("matrix_pin") != expected_sol_fingerprint_pin:
                problems.append(
                    "preflight SOL fingerprint pin does not match matrix"
                )
            observed = sol_check.get("observed_fingerprint") or {}
            if observed.get("hash") != expected_sol_fingerprint_pin.get(
                "native_env_hash"
            ):
                problems.append(
                    "preflight SOL fingerprint does not match certified hash"
                )
    if problems:
        raise ChunkRefusal("preflight_invalid", "; ".join(problems))
    ncu_authority = manifest["checks"]["ncu"]
    try:
        expected_ncu_identity = ncu_executor.build_expected_ncu_identity(
            resolved_path=ncu_authority["resolved_path"],
            executable_sha256=ncu_authority["executable_sha256"],
            version_line=ncu_authority["version_line"],
        )
    except ncu_executor.NcuActionExecutorError as exc:
        raise ChunkRefusal(
            "preflight_invalid", f"checks.ncu executable authority: {exc}"
        ) from exc
    generated = _parse_utc_timestamp(
        generated_raw, field="generated_at_utc"
    )
    if not require_fresh:
        return None
    admitted = admitted_at_utc or datetime.now(timezone.utc)
    if admitted.tzinfo is None:
        raise ChunkRefusal(
            "preflight_invalid", "admitted_at_utc has no timezone"
        )
    admitted = admitted.astimezone(timezone.utc)
    future_s = (generated - admitted).total_seconds()
    if future_s > preflight.PREFLIGHT_FUTURE_SKEW_SECONDS:
        raise ChunkRefusal(
            "PREFLIGHT_CLOCK_SKEW",
            f"generated_at_utc is {future_s:.3f}s after admission; "
            f"maximum={preflight.PREFLIGHT_FUTURE_SKEW_SECONDS:.3f}s",
        )
    age_s = (admitted - generated).total_seconds()
    if age_s > preflight.PREFLIGHT_FRESHNESS_SECONDS:
        raise ChunkRefusal(
            "PREFLIGHT_EXPIRED",
            f"preflight age {age_s:.3f}s exceeds inclusive "
            f"{preflight.PREFLIGHT_FRESHNESS_SECONDS:.3f}s window",
        )
    current_device = preflight.probe_device()
    if (
        type(current_device) is not dict
        or set(current_device) != device_required
        or any(
            type(current_device.get(name)) is not type(device.get(name))
            or current_device.get(name) != device.get(name)
            for name in device_required
        )
    ):
        raise ChunkRefusal(
            "preflight_invalid",
            "current host GPU authority differs from the preflight",
        )
    return expected_ncu_identity


def validate_live_injection_hashes(manifest: dict, pins: dict) -> None:
    probe = (manifest.get("checks") or {}).get("injection_runtime") or {}
    inj = pins.get("injection_runtime") or {}
    tracer = _resolve_repo_path(inj.get("tracer_so_relpath"), "tracer .so")
    smoke = _resolve_repo_path(inj.get("smoke_binary_relpath"), "smoke binary")
    mismatches = []
    for role, path, field in (
        ("tracer .so", tracer, "tracer_so_sha256"),
        ("smoke binary", smoke, "smoke_binary_sha256"),
    ):
        actual = sha256_file(path)
        if probe.get(field) != actual:
            mismatches.append(
                f"{role}: preflight={probe.get(field)!r}, current={actual}"
            )
    if mismatches:
        raise ChunkRefusal("injection_runtime_hash_drift", "; ".join(mismatches))


def _matrix_cells_by_identity(matrix: dict) -> dict[tuple[str, str], dict]:
    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ChunkRefusal("matrix_empty", "matrix cells must be a nonempty list")
    out: dict[tuple[str, str], dict] = {}
    for index, cell in enumerate(cells):
        identity = cell.get("identity") if isinstance(cell, dict) else None
        key = (
            identity.get("roster_source") if isinstance(identity, dict) else None,
            identity.get("slug") if isinstance(identity, dict) else None,
        )
        if not all(isinstance(part, str) and part for part in key):
            raise ChunkRefusal("matrix_identity_invalid", f"cells[{index}]={key!r}")
        if key in out:
            raise ChunkRefusal("matrix_identity_duplicate", repr(key))
        out[key] = cell
    return out


def select_chunk(matrix: dict, chunk_id: str) -> tuple[dict, list[dict]]:
    chunks = matrix.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ChunkRefusal("matrix_chunks_empty", "matrix chunks must be nonempty")
    selected = [row for row in chunks if row.get("chunk_id") == chunk_id]
    if len(selected) != 1:
        raise ChunkRefusal(
            "chunk_identity_invalid", f"chunk_id {chunk_id!r} matched {len(selected)} rows"
        )
    chunk = selected[0]
    members = chunk.get("cells")
    if not isinstance(members, list) or not members:
        raise ChunkRefusal("chunk_empty", f"chunk {chunk_id!r} has no members")
    expected_membership = canonical_sha256(members)
    if chunk.get("membership_sha256") != expected_membership:
        raise ChunkRefusal(
            "chunk_membership_hash_mismatch",
            f"expected {expected_membership}, got {chunk.get('membership_sha256')}",
        )
    cells_by_id = _matrix_cells_by_identity(matrix)
    seen: set[tuple[str, str]] = set()
    resolved: list[dict] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ChunkRefusal("chunk_member_invalid", f"cells[{index}] is not an object")
        key = (member.get("roster_source"), member.get("slug"))
        if key in seen:
            raise ChunkRefusal("chunk_member_duplicate", repr(key))
        seen.add(key)
        if key not in cells_by_id:
            raise ChunkRefusal("chunk_member_absent", repr(key))
        cell = cells_by_id[key]
        if cell.get("disposition") != "runnable":
            raise ChunkRefusal("chunk_member_not_runnable", repr(key))
        readiness = cell.get("capture_readiness") or {}
        if readiness.get("status") != "ready" or readiness.get("blocking_reasons") not in ([], None):
            raise ChunkRefusal(
                "chunk_member_not_ready", f"{key!r}: {readiness!r}"
            )
        resolved.append(cell)
    return chunk, resolved


def validate_matrix_inputs(
    matrix: dict, *, matrix_path: Path, authority_path: Path, pins_path: Path,
    aggregate_budget_path: Path, cell_budget_path: Path, registry_path: Path,
) -> dict:
    try:
        import jsonschema
        jsonschema.validate(matrix, json.loads(MATRIX_SCHEMA.read_text()))
    except Exception as exc:
        raise ChunkRefusal("matrix_schema_invalid", str(exc)) from exc

    source_commit = matrix.get("source_commit")
    if not _is_sha40(source_commit):
        raise ChunkRefusal("matrix_source_commit_invalid", repr(source_commit))
    unsealed = {key: value for key, value in matrix.items()
                if key != "matrix_seal_sha256"}
    derived_matrix_seal = canonical_sha256(unsealed)
    if matrix.get("matrix_seal_sha256") != derived_matrix_seal:
        raise ChunkRefusal(
            "matrix_seal_mismatch",
            f"expected {derived_matrix_seal}, got {matrix.get('matrix_seal_sha256')}",
        )
    semantic_problems = rederive_capture_matrix(matrix, REPO_ROOT)
    if semantic_problems:
        raise ChunkRefusal(
            "matrix_semantic_rederivation_failed", "; ".join(semantic_problems)
        )

    authority_pin = matrix.get("authority") or {}
    pin_group = matrix.get("input_pins") or {}
    expected = (
        (authority_path, authority_pin, "authority"),
        (pins_path, pin_group.get("protocol_pins") or {}, "protocol pins"),
        (aggregate_budget_path, pin_group.get("aggregate_retention_budget") or {},
         "aggregate retention budget"),
        (cell_budget_path, pin_group.get("per_cell_trace_budgets") or {},
         "per-cell trace budgets"),
        (registry_path, pin_group.get("metric_registry") or {}, "metric registry"),
    )
    for actual_path, pin, role in expected:
        if pin.get("path"):
            pinned_path = _resolve_repo_path(pin["path"], role)
            if pinned_path != actual_path.resolve():
                raise ChunkRefusal(
                    "input_path_mismatch",
                    f"{role}: matrix={pinned_path}, CLI={actual_path.resolve()}",
                )
        _require_hash(actual_path, pin.get("sha256"), role)

    if (pin_group.get("per_cell_trace_budgets") or {}).get(
        "defaults_are_enforcement"
    ) is not False:
        raise ChunkRefusal(
            "budget_defaults_ambiguous", "matrix must pin defaults_are_enforcement=false"
        )

    authority = _json(authority_path)
    if authority.get("source_commit") != source_commit:
        raise ChunkRefusal(
            "authority_source_commit_mismatch",
            f"{authority.get('source_commit')!r} != {source_commit!r}",
        )
    authority_rows: dict[tuple[str, str], dict] = {}
    for row in authority.get("cells") or []:
        key = (row.get("roster_source"), row.get("slug"))
        if key in authority_rows:
            raise ChunkRefusal("authority_identity_duplicate", repr(key))
        authority_rows[key] = row
    for key, cell in _matrix_cells_by_identity(matrix).items():
        row = authority_rows.get(key)
        if row is None:
            raise ChunkRefusal("matrix_authority_member_absent", repr(key))
        actual_row_hash = canonical_sha256(row)
        if cell.get("authority_cell_sha256") != actual_row_hash:
            raise ChunkRefusal(
                "authority_cell_hash_mismatch", f"{key!r}: {actual_row_hash}"
            )
    if set(authority_rows) != set(_matrix_cells_by_identity(matrix)):
        raise ChunkRefusal("matrix_authority_set_mismatch", "identity sets differ")
    return authority


def _forwarded_nonnegative_int_flag(args: Any, flag: str) -> int | None:
    """Parse zero or one exact integer flag without accepting ambiguity."""
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ChunkRefusal("invocation_args_invalid", flag)
    values: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == flag:
            if index + 1 >= len(args):
                raise ChunkRefusal("invocation_flag_invalid", f"{flag}: missing value")
            values.append(args[index + 1])
            index += 2
            continue
        if token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ChunkRefusal("invocation_flag_invalid", f"{flag}: duplicate")
    if not values:
        return None
    if not values[0].isdigit():
        raise ChunkRefusal(
            "invocation_flag_invalid", f"{flag}: non-negative integer required"
        )
    return int(values[0])


def _require_capture_launch_contract(invocation: dict) -> dict:
    contract = invocation.get("capture_launch_contract")
    if isinstance(contract, dict) and contract.get("schema_version") == 2:
        expected = {
            "schema_version": 2,
            "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 0,
            "label_launch_index": 0,
            "label_measured_reps": 1,
            "trace_candidate_call_count": 1,
            "trace_dynamic_launch_scope": "all_in_profiler_roi",
            "trace_dynamic_launch_count_source": "runtime_inventory",
        }
        if contract != expected:
            raise ChunkRefusal(
                "capture_launch_contract_invalid", repr(contract)
            )
        return contract
    base_fields = {
        "schema_version", "label_driver_warmup_count",
        "label_effective_warmup_count", "label_launch_index",
        "label_measured_reps", "trace_launch_count",
        "label_trace_launch_index",
    }
    if not isinstance(contract, dict) or frozenset(contract) not in {
        frozenset(base_fields), frozenset(base_fields | {"trace_launch_sequence"}),
    }:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    integer_fields = base_fields - {"schema_version"}
    if contract.get("schema_version") != 1 or any(
        type(contract.get(field)) is not int or contract[field] < 0
        for field in integer_fields
    ):
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    if contract["label_measured_reps"] != 1:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    expected_count = contract["trace_launch_count"]
    if expected_count not in {1, 3}:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    if contract["label_trace_launch_index"] >= expected_count:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    sequence = contract.get("trace_launch_sequence")
    if expected_count == 1 and sequence is not None:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    if expected_count == 3:
        required_rows = [
            {
                "device_id": 0, "stream_id": 0, "kernel_id": 1,
                "kernel_name": "warm_cold_pair_kernel___0",
                "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
                "role": "cold",
            },
            {
                "device_id": 0, "stream_id": 0, "kernel_id": 2,
                "kernel_name": "warm_cold_pair_eviction_kernel___0",
                "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
                "role": "evict",
            },
            {
                "device_id": 0, "stream_id": 0, "kernel_id": 3,
                "kernel_name": "warm_cold_pair_kernel___0",
                "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
                "role": "warm",
            },
        ]
        if sequence != required_rows:
            raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    label_args = invocation.get("label_app_args")
    forwarded_warmup = _forwarded_nonnegative_int_flag(label_args, "--warmup")
    if forwarded_warmup is None:
        logical_shape = (invocation.get("logical_workload") or {}).get("value")
        effective_warmup = (
            contract["label_driver_warmup_count"]
            if invocation.get("shape_delivery") == "application_args"
            and logical_shape in {"correctness_shape", "timing_shape"}
            else 0
        )
    else:
        effective_warmup = forwarded_warmup
        if forwarded_warmup != contract["label_driver_warmup_count"]:
            raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    if effective_warmup != contract["label_effective_warmup_count"]:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    forwarded_reps = _forwarded_nonnegative_int_flag(label_args, "--reps")
    if forwarded_reps is not None and forwarded_reps != 1:
        raise ChunkRefusal("capture_launch_contract_invalid", repr(contract))
    return contract


def _validate_trace_launch_contract(invocation: dict, root: dict) -> None:
    contract = _require_capture_launch_contract(invocation)
    expected_count = contract["trace_launch_count"]
    launches = root.get("launches")
    if root.get("launch_count") != expected_count or not isinstance(launches, list) \
            or len(launches) != expected_count:
        raise ChunkRefusal(
            "trace_launch_cardinality_invalid",
            f"expected {expected_count} selected launch(es), observed "
            f"{root.get('launch_count')!r}",
        )
    expected_sequence = contract.get("trace_launch_sequence")
    if expected_sequence is not None:
        comparable = [
            {key: launch.get(key) for key in (
                "device_id", "stream_id", "kernel_id", "kernel_name",
                "grid_dim", "block_dim",
            )}
            for launch in launches
        ]
        expected_comparable = [
            {key: launch[key] for key in (
                "device_id", "stream_id", "kernel_id", "kernel_name",
                "grid_dim", "block_dim",
            )}
            for launch in expected_sequence
        ]
        if comparable != expected_comparable:
            raise ChunkRefusal(
                "trace_launch_sequence_invalid",
                f"expected={expected_comparable!r}, observed={comparable!r}",
            )


def validate_direct_cell(cell: dict, *, cell_budgets: Any) -> tuple[Path, dict, Any]:
    identity = cell["identity"]
    invocation = cell.get("invocation") or {}
    adapter_kind = invocation.get("adapter_kind")
    if adapter_kind not in _DIRECT_ADAPTERS:
        raise ChunkRefusal(
            "adapter_not_implemented",
            f"{identity}: adapter_kind={adapter_kind!r} is not a direct executable",
        )
    binary = _resolve_repo_path(invocation.get("executable_path"), "cell executable")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ChunkRefusal("cell_executable_unavailable", str(binary))
    _require_hash(binary, invocation.get("executable_sha256"), "cell executable")
    for field in ("label_app_args", "trace_app_args"):
        args = invocation.get(field)
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ChunkRefusal("invocation_args_invalid", f"{identity}: {field}")
    _require_capture_launch_contract(invocation)
    budget = require_cell_budget(cell_budgets, identity["slug"])
    matrix_budget = cell.get("trace_budget") or {}
    for field in ("max_trace_bytes", "max_wall_time_s", "sampling_policy"):
        if matrix_budget.get(field) != getattr(budget, field):
            raise ChunkRefusal(
                "cell_budget_mismatch",
                f"{identity}: matrix {field}={matrix_budget.get(field)!r} != "
                f"config {getattr(budget, field)!r}",
            )
    control = cell.get("control_bits") or {}
    expected_control_policy = (
        "post_trace_vendor_module"
        if identity.get("roster_source") == "simbench_bw"
        and identity.get("slug") in _VENDOR_RUNTIME_CONTROL_SLUGS
        else "direct_executable"
    )
    if control.get("policy") != expected_control_policy:
        raise ChunkRefusal(
            "control_policy_not_implemented", f"{identity}: {control.get('policy')!r}"
        )
    if (
        expected_control_policy == "direct_executable"
        and control.get("input_sha256") != invocation.get("executable_sha256")
    ):
        raise ChunkRefusal("control_binary_mismatch", repr(identity))
    return binary, invocation, budget


def _workload_row_count(path: Path) -> int:
    """Count the exact non-comment JSONL rows the SOL adapter consumes."""
    count = 0
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ChunkRefusal("sol_workload_unreadable", f"{path}: {exc}") from exc
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ChunkRefusal(
                "sol_workload_json_invalid", f"{path}:{line_no}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ChunkRefusal(
                "sol_workload_row_not_object", f"{path}:{line_no}"
            )
        count += 1
    if count <= 0:
        raise ChunkRefusal("sol_workload_empty", str(path))
    return count


def _shape_tokens(args: list[str]) -> list[str]:
    return [
        token for token in args
        if token in {"--shape-point", "--correctness_shape", "--timing_shape"}
        or token.startswith("--shape-point=")
    ]


_SOL_WORKLOAD_ROW_BINDING_FIELDS = frozenset({
    "schema_version",
    "supplemental_matrix_sha256",
    "supplemental_matrix_seal_sha256",
    "chunk_id",
    "row_key",
    "cohort",
    "category",
    "flashinfer_corpus",
    "launch_scope",
    "launch_completeness_contract",
    "claim_scope",
    "native_shape",
    "native_predictive",
    "problem_level_runnable",
    "variant_contract_sha256",
    "row_evidence_sha256",
    "source_workload_index",
    "source_workload_uuid",
    "source_axes",
    "source_row_sha256",
    "capture_workload_index",
    "capture_axes",
    "capture_row_sha256",
})

_SOL_REPRESENTATIVE_EXECUTION_PROFILE = (
    "representative_trace_ncu_control_typed_v1"
)
_SOL_REPRESENTATIVE_WORKLOAD_ROW_BINDING_FIELDS = (
    _SOL_WORKLOAD_ROW_BINDING_FIELDS | {"execution_profile"}
)

_SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT = {
    "inventory_policy": "requires_inventory_before_capture",
    "launch_scope": "all_in_profiler_roi",
    "trace_requirement": "every_inventory_launch_complete",
    "ncu_label_requirement": "every_inventory_launch_complete",
    "control_requirement": "every_inventory_launch_complete",
    "admission_policy": "all_three_layers_complete_for_exact_inventory",
}

_SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT = {
    "inventory_policy": "requires_inventory_before_capture",
    "launch_scope": "all_in_profiler_roi",
    "trace_requirement": "every_inventory_launch_complete",
    "ncu_label_requirement": (
        "every_inventory_launch_complete_cycles_only_binding"
    ),
    "control_requirement": (
        "complete_or_origin_unresolved_typed_unavailable"
    ),
    "admission_policy": (
        "trace_and_ncu_complete_control_typed_nonblocking"
    ),
}


def _valid_supplemental_flashinfer_binding(binding: dict[str, Any]) -> bool:
    category = binding.get("category")
    contract = binding.get("flashinfer_corpus")
    fields = {
        "policy", "category", "environment_variable", "injection_policy",
        "root", "data_annex_manifest_path", "data_annex_manifest_sha256",
        "filtered_manifest_sha256", "file_count", "safetensors_count",
    }
    if (
        category not in {"L1", "L2", "FIB", "Quant"}
        or not isinstance(contract, dict) or set(contract) != fields
        or contract.get("policy") != "verified_fib_only"
        or contract.get("category") != category
        or contract.get("environment_variable") != "FLASHINFER_TRACE_DIR"
        or contract.get("data_annex_manifest_path")
        != "evidence/manifests/data_annex.sha256"
        or contract.get("file_count") != 5244
        or contract.get("safetensors_count") != 2611
        or any(
            not isinstance(contract.get(name), str)
            or len(contract[name]) != 64
            or any(ch not in "0123456789abcdef" for ch in contract[name])
            for name in (
                "data_annex_manifest_sha256", "filtered_manifest_sha256",
            )
        )
    ):
        return False
    if category == "FIB":
        return (
            contract.get("injection_policy") == "verified_absolute_root"
            and contract.get("root") == "data/annex/flashinfer_trace"
        )
    return (
        contract.get("injection_policy") == "forbidden_remove_ambient"
        and contract.get("root") is None
    )


def _effective_literal_sol_rows(payload: bytes) -> list[tuple[bytes, Any]]:
    physical_rows = payload.split(b"\n")
    if physical_rows and physical_rows[-1] == b"":
        physical_rows.pop()
    rows: list[tuple[bytes, Any]] = []
    try:
        for raw_row in physical_rows:
            parse_row = raw_row.strip()
            if not parse_row or parse_row.startswith(b"//"):
                continue
            rows.append((raw_row, json.loads(parse_row)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ChunkRefusal(
            "sol_workload_row_binding_mismatch", str(exc)
        ) from exc
    if not rows:
        raise ChunkRefusal(
            "sol_workload_row_binding_mismatch",
            "workload contains zero effective literal JSONL rows",
        )
    return rows


def bind_supplemental_sol_workload_row(
    invocation: dict, binding: dict,
    supplemental_workload_path: Path | None = None,
    supplemental_definition_path: Path | None = None,
) -> dict:
    """Bind one additive workload-row selection without weakening C1.

    The canonical 344-cell matrix still reaches :func:`capture_cell` without
    this optional binding and therefore retains its row-zero contract.  The
    supplemental runner first revalidates that canonical cell, then calls this
    helper with a matrix-sealed row record.  We re-read the effective workload
    bytes here so the adapter index, UUID, axes, and raw-row hash cannot drift
    between supplemental validation and GPU execution.
    """
    binding_fields = set(binding) if isinstance(binding, dict) else set()
    representative = (
        binding_fields == _SOL_REPRESENTATIVE_WORKLOAD_ROW_BINDING_FIELDS
        and binding.get("execution_profile")
        == _SOL_REPRESENTATIVE_EXECUTION_PROFILE
    )
    exhaustive = binding_fields == _SOL_WORKLOAD_ROW_BINDING_FIELDS
    if not isinstance(binding, dict) or not (representative or exhaustive):
        raise ChunkRefusal(
            "sol_workload_row_binding_invalid",
            f"fields={sorted(binding) if isinstance(binding, dict) else None!r}",
        )
    sha_fields = (
        "supplemental_matrix_sha256",
        "supplemental_matrix_seal_sha256",
        "variant_contract_sha256",
        "row_evidence_sha256",
        "source_row_sha256",
        "capture_row_sha256",
    )
    if any(
        not isinstance(binding.get(field), str)
        or len(binding[field]) != 64
        or any(ch not in "0123456789abcdef" for ch in binding[field])
        for field in sha_fields
    ):
        raise ChunkRefusal(
            "sol_workload_row_binding_invalid", "one or more SHA-256 fields"
        )
    key = binding.get("row_key")
    if (
        binding.get("schema_version") != 1
        or not isinstance(binding.get("chunk_id"), str)
        or not binding["chunk_id"]
        or not isinstance(key, dict)
        or set(key) != {"slug", "source_workload_uuid", "variant_id"}
        or key.get("source_workload_uuid") != binding.get("source_workload_uuid")
        or not all(isinstance(key.get(name), str) and key[name]
                   for name in key)
        or type(binding.get("source_workload_index")) is not int
        or binding["source_workload_index"] < 0
        or type(binding.get("capture_workload_index")) is not int
        or binding["capture_workload_index"] < 0
        or not isinstance(binding.get("source_axes"), dict)
        or not isinstance(binding.get("capture_axes"), dict)
        or binding.get("launch_scope") != "all_in_profiler_roi"
        or binding.get("launch_completeness_contract")
        != (
            _SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
            if representative else _SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT
        )
        or not _valid_supplemental_flashinfer_binding(binding)
    ):
        raise ChunkRefusal(
            "sol_workload_row_binding_invalid", repr(binding)
        )
    cohort = binding.get("cohort")
    claim_logical = (
        invocation.get("logical_workload")
        if isinstance(invocation, dict) else None
    )
    claim_external_definition = (
        claim_logical.get("external_definition_source")
        if isinstance(claim_logical, dict) else None
    )
    row_bundle_changed = (
        binding.get("source_axes") != binding.get("capture_axes")
        and binding.get("source_row_sha256")
        != binding.get("capture_row_sha256")
    )
    definition_bundle_changed = (
        isinstance(claim_external_definition, dict)
        and isinstance(claim_logical.get("native_definition_file_sha256"), str)
        and claim_external_definition.get("sha256")
        == claim_logical.get("definition_file_sha256")
        and claim_external_definition.get("sha256")
        != claim_logical.get("native_definition_file_sha256")
    )
    if cohort == "native":
        if (
            key["variant_id"] != "native"
            or binding.get("claim_scope") != "native_workload_row"
            or binding.get("native_shape") is not True
            or binding.get("native_predictive") is not True
            or binding.get("problem_level_runnable") is not True
            or binding["source_workload_index"]
            != binding["capture_workload_index"]
            or binding["source_axes"] != binding["capture_axes"]
            or binding["source_row_sha256"] != binding["capture_row_sha256"]
        ):
            raise ChunkRefusal(
                "sol_workload_row_claim_invalid", repr(binding)
            )
    elif cohort == "reduced_proxy":
        if (
            len(key["variant_id"]) != 64
            or any(ch not in "0123456789abcdef" for ch in key["variant_id"])
            or binding.get("claim_scope") != "reduced_proxy_bundle_only"
            or binding.get("native_shape") is not False
            or binding.get("native_predictive") is not False
            or binding.get("problem_level_runnable") is not False
            or not (row_bundle_changed or definition_bundle_changed)
            or (
                binding["source_axes"] == binding["capture_axes"]
                and binding["source_row_sha256"]
                != binding["capture_row_sha256"]
            )
            or (
                binding["source_axes"] != binding["capture_axes"]
                and binding["source_row_sha256"]
                == binding["capture_row_sha256"]
            )
        ):
            raise ChunkRefusal(
                "sol_workload_row_claim_invalid", repr(binding)
            )
    elif cohort == "partial_native_row":
        if (
            len(key["variant_id"]) != 64
            or any(ch not in "0123456789abcdef" for ch in key["variant_id"])
            or binding.get("claim_scope")
            != "partial_native_workload_row_only"
            or binding.get("native_shape") is not True
            or binding.get("native_predictive") is not False
            or binding.get("problem_level_runnable") is not False
            or binding["source_workload_index"]
            != binding["capture_workload_index"]
            or binding["source_axes"] != binding["capture_axes"]
            or binding["source_row_sha256"] != binding["capture_row_sha256"]
        ):
            raise ChunkRefusal(
                "sol_workload_row_claim_invalid", repr(binding)
            )
    else:
        raise ChunkRefusal("sol_workload_row_claim_invalid", repr(cohort))

    rebound = copy.deepcopy(invocation)
    logical = rebound.get("logical_workload") or {}
    external_source = logical.get("external_workload_source")
    external_definition_source = logical.get("external_definition_source")
    external_payload: bytes | None = None
    external_definition_payload: bytes | None = None
    if cohort == "reduced_proxy":
        expected_external_fields = {
            "kind", "name", "sha256", "size_bytes", "owner_marker",
            "runtime_rebind",
        }
        if (
            supplemental_workload_path is None
            or not isinstance(external_source, dict)
            or set(external_source) != expected_external_fields
            or external_source.get("kind") != "shrink_external_child_v1"
            or external_source.get("name") != logical.get("workload_file")
            or external_source.get("sha256")
            != logical.get("workload_file_sha256")
            or external_source.get("runtime_rebind")
            != "campaign_retained_immutable_child"
            or type(external_source.get("size_bytes")) is not int
            or external_source["size_bytes"] <= 0
        ):
            raise ChunkRefusal(
                "sol_external_workload_rebind_invalid", repr(external_source)
            )
        workload = Path(supplemental_workload_path)
        if not workload.is_absolute():
            raise ChunkRefusal(
                "sol_external_workload_rebind_invalid", str(workload)
            )
        external_payload, observed = _read_owned_regular_nofollow(
            workload, role="retained supplemental SOL workload"
        )
        if (
            observed.st_size != external_source["size_bytes"]
            or hashlib.sha256(external_payload).hexdigest()
            != external_source["sha256"]
        ):
            raise ChunkRefusal(
                "sol_external_workload_rebind_invalid", str(workload)
            )
        definition: Path | None = None
        if external_definition_source is not None:
            if (
                supplemental_definition_path is None
                or not isinstance(external_definition_source, dict)
                or set(external_definition_source) != expected_external_fields
                or external_definition_source.get("kind")
                != "shrink_external_definition_child_v1"
                or external_definition_source.get("name")
                != logical.get("definition_file")
                or external_definition_source.get("sha256")
                != logical.get("definition_file_sha256")
                or external_definition_source.get("runtime_rebind")
                != "campaign_retained_immutable_child"
                or external_definition_source.get("owner_marker")
                != external_source.get("owner_marker")
                or type(external_definition_source.get("size_bytes")) is not int
                or external_definition_source["size_bytes"] <= 0
            ):
                raise ChunkRefusal(
                    "sol_external_definition_rebind_invalid",
                    repr(external_definition_source),
                )
            definition = Path(supplemental_definition_path)
            if not definition.is_absolute():
                raise ChunkRefusal(
                    "sol_external_definition_rebind_invalid", str(definition)
                )
            external_definition_payload, definition_stat = (
                _read_owned_regular_nofollow(
                    definition, role="retained supplemental SOL definition"
                )
            )
            try:
                definition_doc = json.loads(external_definition_payload)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ChunkRefusal(
                    "sol_external_definition_rebind_invalid",
                    f"{definition}: {exc}",
                ) from exc
            if (
                definition_stat.st_size
                != external_definition_source["size_bytes"]
                or hashlib.sha256(external_definition_payload).hexdigest()
                != external_definition_source["sha256"]
                or not isinstance(definition_doc, dict)
                or definition_doc.get("name")
                != logical.get("benchmark_definition")
            ):
                raise ChunkRefusal(
                    "sol_external_definition_rebind_invalid", str(definition)
                )
        elif supplemental_definition_path is not None:
            raise ChunkRefusal(
                "sol_external_definition_scope_invalid",
                str(supplemental_definition_path),
            )
        expected_argv = [
            "--definition", logical.get("definition_file"),
            "--workload", external_source["name"],
            "--solution", rebound.get("authority_reference_path"), "--json",
        ]
        for field in ("label_app_args", "trace_app_args"):
            if rebound.get(field) != expected_argv:
                raise ChunkRefusal(
                    "sol_external_workload_argv_invalid", repr(rebound.get(field))
                )
            rebound[field] = list(expected_argv)
            rebound[field][3] = str(workload)
            if definition is not None:
                rebound[field][1] = str(definition)
        contract = rebound.get("adapter_contract") or {}
        if contract.get("forwarded_argv") != expected_argv:
            raise ChunkRefusal(
                "sol_external_workload_argv_invalid",
                repr(contract.get("forwarded_argv")),
            )
        contract["forwarded_argv"] = list(expected_argv)
        contract["forwarded_argv"][3] = str(workload)
        if definition is not None:
            contract["forwarded_argv"][1] = str(definition)
        logical["workload_file"] = str(workload)
        if definition is not None:
            logical["definition_file"] = str(definition)
    elif (
        supplemental_workload_path is not None
        or supplemental_definition_path is not None
        or external_source is not None
        or external_definition_source is not None
    ):
        raise ChunkRefusal(
            "sol_external_workload_scope_invalid", repr(cohort)
        )
    workload = _resolve_repo_path(
        logical.get("workload_file"), "supplemental SOL workload"
    )
    if external_payload is not None:
        payload = external_payload
    else:
        try:
            payload = workload.read_bytes()
        except OSError as exc:
            raise ChunkRefusal(
                "sol_workload_row_binding_unreadable", f"{workload}: {exc}"
            ) from exc
    # Row identity is over the effective JSONL rows consumed by the adapter.
    # Hash the literal record bytes with only the LF delimiter removed; use a
    # stripped copy solely to recognize comments/blanks and parse JSON.  This
    # keeps logical workload indices stable without normalizing row identity.
    rows = _effective_literal_sol_rows(payload)
    capture_index = binding["capture_workload_index"]
    if capture_index >= len(rows):
        raise ChunkRefusal(
            "sol_workload_row_binding_mismatch",
            f"capture index {capture_index} >= row count {len(rows)}",
        )
    raw, selected = rows[capture_index]
    if (
        not isinstance(selected, dict)
        or selected.get("uuid") != binding["source_workload_uuid"]
        or selected.get("axes") != binding["capture_axes"]
        or hashlib.sha256(raw).hexdigest() != binding["capture_row_sha256"]
    ):
        raise ChunkRefusal(
            "sol_workload_row_binding_mismatch", repr(key)
        )
    target = invocation.get("capture_target")
    if (
        not isinstance(target, dict)
        or target.get("source_workload_index") != 0
        or target.get("staged_workload_index") != 0
    ):
        raise ChunkRefusal(
            "sol_workload_row_canonical_target_invalid", repr(target)
        )
    rebound["capture_target"]["source_workload_index"] = capture_index
    return rebound


def _supplemental_canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_SOL_DIAGNOSTIC_WORKLOAD_FIELDS = frozenset({
    "axes", "inputs", "uuid", "tolerance",
})
_SOL_DIAGNOSTIC_TOLERANCE_FIELDS = frozenset({
    "max_atol", "max_rtol", "required_matched_ratio", "max_error_cap",
    "allow_negative_inf",
})
_SOL_DIAGNOSTIC_TOLERANCE_DEFAULTS = {
    "max_atol": 0.01,
    "max_rtol": 0.01,
    "required_matched_ratio": 0.99,
    "max_error_cap": None,
    "allow_negative_inf": False,
}
_SOL_LEGACY_MATCH_RATIO_FIELD = "required_match_ratio"
_SOL_LEGACY_MATCH_RATIO_VALUE = 0.98


def _normalize_sol_diagnostic_workload(value: Any) -> dict[str, Any]:
    """Return the exact workload view serialized by the pinned executor.

    Source JSONL rows may omit ``tolerance`` or any of its defaulted fields,
    while the Pydantic ``Workload`` written to a diagnostic trace always
    materializes the complete ``ToleranceSpec``.  Normalize only that known
    serialization boundary.  All other workload content remains exact and
    unknown or wrongly typed fields fail closed.

    The frozen corpus also contains the historical spelling
    ``required_match_ratio: 0.98``.  The pinned model ignored that extra field
    and therefore executed with ``required_matched_ratio=0.99``.  Recognize
    only that exact legacy form and reproduce the executed value; never map it
    to 0.98 or accept another ignored field.
    """
    if type(value) is not dict:
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            "workload is not an object",
        )
    unknown = set(value) - _SOL_DIAGNOSTIC_WORKLOAD_FIELDS
    missing = {"axes", "inputs", "uuid"} - set(value)
    if unknown or missing:
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            f"workload fields missing={sorted(missing)!r} "
            f"unknown={sorted(unknown)!r}",
        )
    if (
        type(value["axes"]) is not dict
        or type(value["inputs"]) is not dict
        or type(value["uuid"]) is not str
        or not value["uuid"]
    ):
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            "axes/inputs/uuid type mismatch",
        )

    source_tolerance = value.get("tolerance", {})
    if type(source_tolerance) is not dict:
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            "tolerance is not an object",
        )
    tolerance_fields = set(source_tolerance)
    unknown_tolerance = tolerance_fields - {
        *_SOL_DIAGNOSTIC_TOLERANCE_FIELDS,
        _SOL_LEGACY_MATCH_RATIO_FIELD,
    }
    if unknown_tolerance:
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            f"unknown tolerance fields {sorted(unknown_tolerance)!r}",
        )
    if _SOL_LEGACY_MATCH_RATIO_FIELD in source_tolerance:
        legacy_value = source_tolerance[_SOL_LEGACY_MATCH_RATIO_FIELD]
        if (
            "required_matched_ratio" in source_tolerance
            or isinstance(legacy_value, bool)
            or not isinstance(legacy_value, (int, float))
            or legacy_value != _SOL_LEGACY_MATCH_RATIO_VALUE
        ):
            raise ChunkRefusal(
                "sol_native_diagnostic_workload_invalid",
                "legacy required_match_ratio is not the exact executed "
                "0.98 source form",
            )

    normalized_tolerance = dict(_SOL_DIAGNOSTIC_TOLERANCE_DEFAULTS)
    for field in _SOL_DIAGNOSTIC_TOLERANCE_FIELDS:
        if field in source_tolerance:
            normalized_tolerance[field] = source_tolerance[field]
    for field in ("max_atol", "max_rtol", "required_matched_ratio"):
        field_value = normalized_tolerance[field]
        if isinstance(field_value, bool) or not isinstance(
            field_value, (int, float)
        ):
            raise ChunkRefusal(
                "sol_native_diagnostic_workload_invalid",
                f"{field} is not numeric",
            )
        normalized_tolerance[field] = float(field_value)
    error_cap = normalized_tolerance["max_error_cap"]
    if (
        error_cap is not None
        and (
            isinstance(error_cap, bool)
            or not isinstance(error_cap, (int, float))
        )
    ):
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            "max_error_cap is not numeric or null",
        )
    if error_cap is not None:
        normalized_tolerance["max_error_cap"] = float(error_cap)
    if type(normalized_tolerance["allow_negative_inf"]) is not bool:
        raise ChunkRefusal(
            "sol_native_diagnostic_workload_invalid",
            "allow_negative_inf is not boolean",
        )
    return {
        "axes": copy.deepcopy(value["axes"]),
        "inputs": copy.deepcopy(value["inputs"]),
        "uuid": value["uuid"],
        "tolerance": normalized_tolerance,
    }


def _validate_current_sol_runnability_evidence(
    *, evidence: dict[str, Any], expected: dict[str, Any],
    binding: dict[str, Any], identity: dict[str, Any],
    observed_workloads: int, workload_path: Path,
    definition_name: str, solution_name: Any,
    runnability_path: Path | None = None,
) -> None:
    """Validate current full/partial row evidence without normalizing siblings."""
    if evidence != expected:
        raise ChunkRefusal(
            "sol_supplemental_evidence_mismatch", repr(identity)
        )
    source_kind = evidence.get("source_kind")
    cohort = binding.get("cohort")
    expected_kind = (
        "current_full_correctness" if cohort == "native"
        else "current_partial_native_row"
    )
    if source_kind != expected_kind:
        raise ChunkRefusal(
            "sol_evaluation_evidence_kind_mismatch", repr(source_kind)
        )
    if runnability_path is None:
        path = _resolve_repo_path(
            evidence.get("source_path"), "current SOL runnability evidence"
        )
        _require_hash(
            path, evidence.get("source_sha256"),
            "current SOL runnability evidence",
        )
        document = _json(path)
    else:
        path = Path(runnability_path)
        if evidence.get("source_path") != path.name:
            raise ChunkRefusal(
                "sol_supplemental_evidence_path_mismatch",
                f"expected retained basename {path.name!r}, "
                f"observed {evidence.get('source_path')!r}",
            )
        payload, _observed = _read_owned_regular_nofollow(
            path, role="retained current SOL runnability evidence"
        )
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        expected_sha256 = evidence.get("source_sha256")
        if actual_sha256 != expected_sha256:
            raise ChunkRefusal(
                "input_hash_mismatch",
                "current SOL runnability evidence: expected "
                f"{expected_sha256}, observed {actual_sha256}: {path}",
            )
        try:
            document = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ChunkRefusal(
                "input_not_object",
                f"current SOL runnability evidence: {path}: {exc}",
            ) from exc
        if not isinstance(document, dict):
            raise ChunkRefusal(
                "input_not_object", f"{path} root is not an object"
            )
    rows = [
        row for row in document.get("cells") or []
        if isinstance(row, dict) and row.get("slug") == identity.get("slug")
    ]
    if (
        len(rows) != 1
        or _supplemental_canonical_sha256(rows[0])
        != evidence.get("cell_sha256")
    ):
        raise ChunkRefusal(
            "sol_evaluation_evidence_row_mismatch", repr(identity)
        )
    evidence_row = rows[0]
    diagnostics = evidence_row.get("workload_diagnostics")
    try:
        literal_rows = _effective_literal_sol_rows(workload_path.read_bytes())
    except (OSError, ChunkRefusal) as exc:
        raise ChunkRefusal(
            "sol_native_diagnostic_bijection_invalid", str(exc)
        ) from exc
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != observed_workloads
        or len(literal_rows) != observed_workloads
        or evidence_row.get("unexpected_workload_diagnostics") != []
    ):
        raise ChunkRefusal(
            "sol_native_diagnostic_bijection_invalid", repr(identity)
        )
    source_uuids: set[str] = set()
    trace_indices: set[int] = set()
    for index, ((raw_source, source_value), diagnostic) in enumerate(
        zip(literal_rows, diagnostics)
    ):
        normalized_source = _normalize_sol_diagnostic_workload(source_value)
        source_uuid = (
            source_value.get("uuid")
            if isinstance(source_value, dict) else None
        )
        source_axes = (
            source_value.get("axes")
            if isinstance(source_value, dict) else None
        )
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("workload_index") != index
            or diagnostic.get("uuid") != source_uuid
            or diagnostic.get("axes") != source_axes
            or diagnostic.get("workload_sha256")
            != hashlib.sha256(raw_source).hexdigest()
            or not isinstance(source_uuid, str)
            or source_uuid in source_uuids
        ):
            raise ChunkRefusal(
                "sol_native_diagnostic_bijection_invalid",
                f"{identity!r}: index={index}",
            )
        source_uuids.add(source_uuid)
        if diagnostic.get("diagnostic_present") is True:
            trace_index = diagnostic.get("trace_index")
            trace = diagnostic.get("trace")
            if (
                type(trace_index) is not int or trace_index < 0
                or trace_index in trace_indices
                or not isinstance(trace, dict)
                or trace.get("workload") != normalized_source
            ):
                raise ChunkRefusal(
                    "sol_native_diagnostic_bijection_invalid",
                    f"{identity!r}: index={index}",
                )
            trace_indices.add(trace_index)
        elif any(
            diagnostic.get(field) is not None
            for field in ("trace_index", "status", "trace")
        ):
            raise ChunkRefusal(
                "sol_native_diagnostic_bijection_invalid",
                f"{identity!r}: index={index}",
            )
    if cohort == "native" and (
        evidence_row.get("returned_workloads") != observed_workloads
        or any(row.get("diagnostic_present") is not True for row in diagnostics)
        or trace_indices != set(range(observed_workloads))
    ):
        raise ChunkRefusal(
            "sol_native_diagnostic_bijection_invalid", repr(identity)
        )
    top_pins = (
        "resume_identity_sha256", "probe_protocol_sha256",
        "probe_config_sha256", "sol_execbench_package_sha256",
        "native_env_hash",
    )
    if any(document.get(name) != evidence.get(name) for name in top_pins):
        raise ChunkRefusal(
            "sol_supplemental_evidence_pin_mismatch", repr(identity)
        )
    if (
        evidence.get("diagnostics_complete")
        is not evidence_row.get("diagnostics_complete")
        or evidence.get("problem_status") != evidence_row.get("status")
        or evidence.get("problem_returncode") != evidence_row.get("returncode")
        or evidence.get("expected_workloads") != observed_workloads
        or evidence.get("expected_workloads")
        != evidence_row.get("expected_workloads")
        or evidence.get("passed_workloads")
        != evidence_row.get("passed_workloads")
    ):
        raise ChunkRefusal(
            "sol_supplemental_evidence_incomplete", repr(identity)
        )
    passed = evidence["passed_workloads"]
    if cohort == "native":
        if (
            binding.get("problem_level_runnable") is not True
            or evidence.get("diagnostics_complete") is not True
            or evidence.get("problem_status") != "runnable"
            or evidence.get("problem_returncode") != 0
            or passed != observed_workloads
            or evidence_row.get("returncode") != 0
        ):
            raise ChunkRefusal(
                "sol_native_evaluation_not_runnable", repr(identity)
            )
    elif cohort == "partial_native_row":
        if (
            binding.get("problem_level_runnable") is not False
            or type(passed) is not int
            or not (0 < passed < observed_workloads)
            or evidence.get("problem_status") == "runnable"
        ):
            raise ChunkRefusal(
                "sol_partial_native_evidence_invalid", repr(identity)
            )
        selected = {
            "selected_source_workload_index": binding[
                "source_workload_index"
            ],
            "selected_source_workload_uuid": binding[
                "source_workload_uuid"
            ],
            "selected_source_axes": binding["source_axes"],
            "selected_source_row_sha256": binding["source_row_sha256"],
            "selected_diagnostic_sha256": binding[
                "row_evidence_sha256"
            ],
        }
        if any(
            evidence.get(name) != value
            for name, value in selected.items()
        ):
            raise ChunkRefusal(
                "sol_partial_native_selection_mismatch", repr(identity)
            )
    else:
        raise ChunkRefusal(
            "sol_partial_native_evidence_invalid", repr(identity)
        )
    selected_diagnostics = [
        row for row in diagnostics
        if isinstance(row, dict)
        and row.get("workload_index")
        == binding["source_workload_index"]
        and row.get("uuid")
        == binding["source_workload_uuid"]
    ]
    try:
        raw_source, selected_source = literal_rows[
            binding["source_workload_index"]
        ]
    except IndexError as exc:
        raise ChunkRefusal(
            "sol_native_row_diagnostic_mismatch", str(exc)
        ) from exc
    diagnostic = selected_diagnostics[0] if len(selected_diagnostics) == 1 else {}
    trace = diagnostic.get("trace") if isinstance(diagnostic, dict) else None
    normalized_selected_source = _normalize_sol_diagnostic_workload(
        selected_source
    )
    trace_status = trace.get("status") if isinstance(trace, dict) else None
    if trace_status is None and isinstance(trace, dict) and isinstance(
        trace.get("evaluation"), dict
    ):
        trace_status = trace["evaluation"].get("status")
    if isinstance(trace_status, dict):
        trace_status = trace_status.get("value")
    if (
        len(selected_diagnostics) != 1
        or diagnostic.get("diagnostic_present") is not True
        or str(diagnostic.get("status", "")).upper() != "PASSED"
        or diagnostic.get("axes") != binding["source_axes"]
        or diagnostic.get("workload_sha256")
        != binding["source_row_sha256"]
        or hashlib.sha256(raw_source).hexdigest()
        != binding["source_row_sha256"]
        or not isinstance(trace, dict)
        or trace.get("workload") != normalized_selected_source
        or str(trace_status or "").upper() != "PASSED"
        or trace.get("definition") != definition_name
        or trace.get("solution") != solution_name
        or _supplemental_canonical_sha256(diagnostic)
        != binding["row_evidence_sha256"]
    ):
        raise ChunkRefusal(
            "sol_native_row_diagnostic_mismatch", repr(identity)
        )


def validate_sol_cell(
    cell: dict, *, cell_budgets: Any,
    certified_sol_env: dict[str, str] | None,
    workload_row_binding: dict[str, Any] | None = None,
    supplemental_runnability_evidence: dict[str, Any] | None = None,
    runnability_path: Path | None = None,
    supplemental_workload_path: Path | None = None,
    supplemental_definition_path: Path | None = None,
) -> tuple[Path, dict, Any, Path, dict[str, str]]:
    """Re-derive a SOL logical workload before any profiler is launched."""
    identity = cell["identity"]
    invocation = cell.get("invocation") or {}
    if identity.get("roster_source") != "sol_execbench":
        raise ChunkRefusal("adapter_source_mismatch", repr(identity))
    if invocation.get("adapter_kind") != "sol_execbench":
        raise ChunkRefusal(
            "adapter_source_mismatch",
            f"{identity}: {invocation.get('adapter_kind')!r}",
        )
    if invocation.get("shape_delivery") != "adapter_consumes_logical_shape":
        raise ChunkRefusal("sol_shape_delivery_invalid", repr(identity))
    if workload_row_binding is not None:
        invocation = bind_supplemental_sol_workload_row(
            invocation, workload_row_binding,
            supplemental_workload_path=supplemental_workload_path,
            supplemental_definition_path=supplemental_definition_path,
        )

    adapter = _resolve_repo_path(invocation.get("executable_path"), "SOL adapter")
    if not adapter.is_file() or not os.access(adapter, os.X_OK):
        raise ChunkRefusal("sol_adapter_unavailable", str(adapter))
    _require_hash(adapter, invocation.get("executable_sha256"), "SOL adapter")

    solution = _resolve_repo_path(
        invocation.get("authority_reference_path"), "SOL solution"
    )
    expected_solution_sha = invocation.get("authority_reference_sha256")
    _require_hash(solution, expected_solution_sha, "SOL solution")
    try:
        solution_doc = _json(solution)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkRefusal("sol_solution_invalid", f"{solution}: {exc}") from exc

    logical = invocation.get("logical_workload") or {}
    if logical.get("kind") != "sol_execbench_workload_set":
        raise ChunkRefusal("sol_logical_workload_kind_invalid", repr(identity))
    logical_shape = logical.get("value")
    workload_scope = logical.get("workload_scope")
    valid_shape_scope = {
        "workloads": "native",
        "reduced_representative": "reduced_representative",
    }
    if logical_shape not in valid_shape_scope:
        raise ChunkRefusal("sol_logical_shape_missing", repr(identity))
    if workload_scope != valid_shape_scope[logical_shape]:
        raise ChunkRefusal(
            "sol_workload_scope_mismatch",
            f"shape={logical_shape!r}, scope={workload_scope!r}",
        )
    definition_name = logical.get("benchmark_definition")
    if solution_doc.get("definition") != definition_name:
        raise ChunkRefusal(
            "sol_definition_mismatch",
            f"solution={solution_doc.get('definition')!r}, matrix={definition_name!r}",
        )

    benchmark_dir = _resolve_repo_path(
        logical.get("benchmark_dir"), "SOL benchmark directory"
    )
    if not benchmark_dir.is_dir():
        raise ChunkRefusal("sol_benchmark_not_directory", str(benchmark_dir))
    if benchmark_dir.name != definition_name:
        raise ChunkRefusal(
            "sol_definition_mismatch",
            f"benchmark basename={benchmark_dir.name!r}, definition={definition_name!r}",
        )
    if tree_sha256(benchmark_dir) != logical.get("benchmark_tree_sha256"):
        raise ChunkRefusal("sol_benchmark_tree_hash_mismatch", repr(identity))

    definition_path_value = logical.get("definition_file")
    definition_path = _resolve_repo_path(
        definition_path_value, "SOL benchmark definition"
    )
    external_definition_source = logical.get("external_definition_source")
    native_definition_value = logical.get(
        "native_definition_file", definition_path_value
    )
    native_definition_sha = logical.get(
        "native_definition_file_sha256", logical.get("definition_file_sha256")
    )
    native_definition_path = _resolve_repo_path(
        native_definition_value, "native SOL benchmark definition"
    )
    if native_definition_path.parent.resolve() != benchmark_dir.resolve():
        raise ChunkRefusal(
            "sol_definition_path_mismatch", str(native_definition_path)
        )
    _require_hash(
        native_definition_path, native_definition_sha,
        "native SOL benchmark definition",
    )
    if isinstance(external_definition_source, dict):
        if workload_row_binding is None:
            raise ChunkRefusal(
                "sol_external_definition_scope_invalid", repr(identity)
            )
        definition_payload, definition_stat = _read_owned_regular_nofollow(
            definition_path, role="retained supplemental SOL definition"
        )
        if (
            external_definition_source.get("kind")
            != "shrink_external_definition_child_v1"
            or external_definition_source.get("runtime_rebind")
            != "campaign_retained_immutable_child"
            or external_definition_source.get("sha256")
            != logical.get("definition_file_sha256")
            or definition_stat.st_size
            != external_definition_source.get("size_bytes")
            or hashlib.sha256(definition_payload).hexdigest()
            != logical.get("definition_file_sha256")
            or logical.get("definition_file_sha256") == native_definition_sha
        ):
            raise ChunkRefusal(
                "sol_external_definition_contract_invalid", repr(identity)
            )
        try:
            definition_doc = json.loads(definition_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ChunkRefusal(
                "sol_external_definition_contract_invalid",
                f"{definition_path}: {exc}",
            ) from exc
    else:
        if (
            external_definition_source is not None
            or definition_path != native_definition_path
            or logical.get("definition_file_sha256") != native_definition_sha
        ):
            raise ChunkRefusal(
                "sol_definition_path_mismatch", str(definition_path)
            )
        definition_doc = _json(definition_path)
    if (
        not isinstance(definition_doc, dict)
        or definition_doc.get("name") != definition_name
    ):
        observed_definition_name = (
            definition_doc.get("name")
            if isinstance(definition_doc, dict) else None
        )
        raise ChunkRefusal(
            "sol_definition_mismatch",
            f"definition.json={observed_definition_name!r}, "
            f"matrix={definition_name!r}",
        )
    expected_definition_sha = logical.get("definition_file_sha256")
    _require_hash(definition_path, expected_definition_sha, "SOL definition")

    external_source = logical.get("external_workload_source")
    if isinstance(external_source, dict):
        workload = Path(str(logical.get("workload_file", "")))
        if not workload.is_absolute():
            raise ChunkRefusal(
                "sol_external_workload_rebind_invalid", str(workload)
            )
        workload_payload, workload_stat = _read_owned_regular_nofollow(
            workload, role="retained supplemental SOL workload"
        )
        if (
            workload_stat.st_size != external_source.get("size_bytes")
            or hashlib.sha256(workload_payload).hexdigest()
            != logical.get("workload_file_sha256")
        ):
            raise ChunkRefusal(
                "sol_external_workload_rebind_invalid", str(workload)
            )
        observed_workloads = len(_effective_literal_sol_rows(workload_payload))
    else:
        workload = _resolve_repo_path(logical.get("workload_file"), "SOL workload")
        _require_hash(workload, logical.get("workload_file_sha256"), "SOL workload")
        observed_workloads = _workload_row_count(workload)
    if observed_workloads != logical.get("expected_workloads"):
        raise ChunkRefusal(
            "sol_workload_count_mismatch",
            f"observed={observed_workloads}, expected={logical.get('expected_workloads')!r}",
        )

    reduced = logical.get("reduced_workload_provenance")
    if workload_scope == "native":
        if reduced is not None:
            raise ChunkRefusal("sol_native_has_reduced_provenance", repr(identity))
        if workload.resolve() != (benchmark_dir / "workload.jsonl").resolve():
            raise ChunkRefusal("sol_native_workload_path_mismatch", str(workload))
    else:
        external_source = logical.get("external_workload_source")
        if isinstance(external_source, dict):
            if (
                reduced is not None
                or external_source.get("kind") != "shrink_external_child_v1"
                or external_source.get("runtime_rebind")
                != "campaign_retained_immutable_child"
                or external_source.get("sha256")
                != logical.get("workload_file_sha256")
                or type(external_source.get("size_bytes")) is not int
                or external_source["size_bytes"] != workload.stat().st_size
                or not isinstance(external_source.get("owner_marker"), dict)
            ):
                raise ChunkRefusal(
                    "sol_external_workload_contract_invalid", repr(identity)
                )
        else:
            if not isinstance(reduced, dict):
                raise ChunkRefusal("sol_reduced_provenance_missing", repr(identity))
            required_reduced = {
                "classification": "reduced_representative",
                "native_shape_ref": (
                    (benchmark_dir / "workload.jsonl").relative_to(REPO_ROOT).as_posix()
                ),
                "reduced_workload_ref": workload.relative_to(REPO_ROOT).as_posix(),
                "reduced_run_status": "runnable",
                "correctness_result": "passed",
            }
            for field, expected in required_reduced.items():
                if reduced.get(field) != expected:
                    raise ChunkRefusal(
                        "sol_reduced_provenance_mismatch",
                        f"{field}={reduced.get(field)!r}, expected={expected!r}",
                    )
            provenance = _resolve_repo_path(
                reduced.get("source_path"), "SOL shrink provenance"
            )
            _require_hash(
                provenance, reduced.get("source_sha256"), "SOL shrink provenance"
            )
            provenance_doc = _json(provenance)
            matching = [
                row for row in provenance_doc.get("cells") or []
                if isinstance(row, dict) and row.get("slug") == identity["slug"]
            ]
            if len(matching) != 1:
                raise ChunkRefusal(
                    "sol_reduced_provenance_row_missing", repr(identity)
                )
            if canonical_sha256(matching[0]) != reduced.get("row_sha256"):
                raise ChunkRefusal(
                    "sol_reduced_provenance_row_hash_mismatch", repr(identity)
                )
            native_runnability = _resolve_repo_path(
                reduced.get("native_runnability_source_path"),
                "SOL native runnability evidence",
            )
            _require_hash(
                native_runnability,
                reduced.get("native_runnability_source_sha256"),
                "SOL native runnability evidence",
            )
            native_doc = _json(native_runnability)
            native_rows = [
                row for row in native_doc.get("cells") or []
                if isinstance(row, dict) and row.get("slug") == identity["slug"]
            ]
            if (
                len(native_rows) != 1
                or canonical_sha256(native_rows[0])
                != reduced.get("native_runnability_row_sha256")
                or native_rows[0].get("status") != "host_capacity_oom"
                or native_rows[0].get("returncode") != 1
                or reduced.get("native_run_status") != "host_capacity_oom"
                or reduced.get("native_returncode") != 1
            ):
                raise ChunkRefusal(
                    "sol_reduced_native_evidence_mismatch", repr(identity)
                )

    evidence = logical.get("evaluation_evidence")
    if not isinstance(evidence, dict):
        raise ChunkRefusal("sol_evaluation_evidence_missing", repr(identity))
    if supplemental_runnability_evidence is not None:
        if workload_row_binding is None or workload_scope != "native":
            raise ChunkRefusal(
                "sol_supplemental_evidence_scope_invalid", repr(identity)
            )
        _validate_current_sol_runnability_evidence(
            evidence=evidence, expected=supplemental_runnability_evidence,
            binding=workload_row_binding, identity=identity,
            observed_workloads=observed_workloads,
            workload_path=workload, definition_name=definition_name,
            solution_name=solution_doc.get("name"),
            runnability_path=runnability_path,
        )
    else:
        source_kind = evidence.get("source_kind")
        expected_source_kind = (
            "native_runnability" if workload_scope == "native"
            else (
                "current_reduced_proxy_execution"
                if isinstance(logical.get("external_workload_source"), dict)
                else "reduced_shrink_evaluation"
            )
        )
        if source_kind != expected_source_kind:
            raise ChunkRefusal(
                "sol_evaluation_evidence_kind_mismatch", repr(source_kind)
            )
        expected_detail = (
            f"{observed_workloads}/{observed_workloads} workloads passed"
        )
        if (
            evidence.get("status") != "runnable"
            or evidence.get("returncode") != 0
            or evidence.get("expected_workloads") != observed_workloads
            or evidence.get("detail") != expected_detail
        ):
            raise ChunkRefusal(
                "sol_evaluation_evidence_incomplete", repr(identity)
            )
        external_evidence = (
            workload_scope == "reduced_representative"
            and isinstance(logical.get("external_workload_source"), dict)
        )
        if external_evidence:
            if (
                not isinstance(evidence.get("source_path"), str)
                or Path(evidence["source_path"]).name != evidence["source_path"]
                or not isinstance(evidence.get("cell_sha256"), str)
                or not isinstance(
                    evidence.get("cohort_execution_evidence_sha256"), str
                )
            ):
                raise ChunkRefusal(
                    "sol_evaluation_evidence_row_mismatch", repr(identity)
                )
            evidence_row = None
        else:
            evidence_path = _resolve_repo_path(
                evidence.get("source_path"), "SOL evaluation evidence"
            )
            _require_hash(
                evidence_path, evidence.get("source_sha256"),
                "SOL evaluation evidence",
            )
            evidence_doc = _json(evidence_path)
            evidence_rows = [
                row for row in evidence_doc.get("cells") or []
                if isinstance(row, dict) and row.get("slug") == identity["slug"]
            ]
            if len(evidence_rows) != 1 or canonical_sha256(
                evidence_rows[0]
            ) != evidence.get("row_sha256"):
                raise ChunkRefusal(
                    "sol_evaluation_evidence_row_mismatch", repr(identity)
                )
            evidence_row = evidence_rows[0]
        if workload_scope == "native":
            if (
                evidence_row.get("status") != "runnable"
                or evidence_row.get("returncode") != 0
                or evidence_row.get("detail") != expected_detail
            ):
                raise ChunkRefusal(
                    "sol_native_evaluation_not_runnable", repr(identity)
                )
        elif not external_evidence and (
            evidence_row.get("reduced_run_status") != "runnable"
            or evidence_row.get("correctness_result") != "passed"
            or evidence_row.get("reduced_run_detail") != expected_detail
        ):
            raise ChunkRefusal(
                "sol_reduced_evaluation_not_runnable", repr(identity)
            )

    label_args = invocation.get("label_app_args")
    trace_args = invocation.get("trace_app_args")
    launch_contract = _require_capture_launch_contract(invocation)
    if (
        workload_row_binding is None
        and launch_contract.get("schema_version") != 1
    ) or (
        workload_row_binding is not None
        and launch_contract.get("schema_version") != 2
    ):
        raise ChunkRefusal(
            "capture_launch_contract_scope_invalid", repr(identity)
        )
    contract = invocation.get("adapter_contract") or {}
    forwarded = contract.get("forwarded_argv")
    if not all(
        isinstance(args, list) and args and all(isinstance(x, str) for x in args)
        for args in (label_args, trace_args, forwarded)
    ):
        raise ChunkRefusal("sol_adapter_args_invalid", repr(identity))
    if label_args != trace_args or label_args != forwarded:
        raise ChunkRefusal("sol_adapter_args_mismatch", repr(identity))
    bad_shape_tokens = _shape_tokens(label_args)
    if bad_shape_tokens:
        raise ChunkRefusal(
            "sol_shape_token_forwarded", repr(bad_shape_tokens)
        )
    if (
        contract.get("contract_version") != 3
        or contract.get("shape_forwarded_to_target") is not False
        or contract.get("label_driver_bridge_status") != "implemented"
        or contract.get("persistent_stage_status") != "implemented"
        or contract.get("capture_request_environment")
        != "SOL_EXECBENCH_CAPTURE_REQUEST"
    ):
        raise ChunkRefusal("sol_adapter_contract_invalid", repr(identity))
    flashinfer_contract, _flashinfer_digest, flashinfer_runtime_root = (
        _sol_flashinfer_corpus_contract(invocation)
    )
    if (
        workload_row_binding is not None
        and workload_row_binding.get("flashinfer_corpus")
        != flashinfer_contract
    ):
        raise ChunkRefusal(
            "SOL_FIB_CORPUS_CONTRACT_INVALID", repr(identity)
        )
    if workload_row_binding is None:
        expected_target = {
            "source_workload_index": 0,
            "staged_workload_index": 0,
            "candidate_user_fn_calls": 1,
            "candidate_kernel_ordinal": 0,
            "capture_launches": 1,
            "profiler_range": "cudaProfilerStartStop",
            "setup_outside_profiler_range": True,
            "timing_methodology": "cuda_events",
        }
        expected_trace_cardinality = "one_pinned_candidate_kernel_launch_exact"
    else:
        expected_target = {
            "source_workload_index": workload_row_binding[
                "capture_workload_index"
            ],
            "staged_workload_index": 0,
            "selected_candidate_call_ordinal": 0,
            "candidate_user_fn_calls": 1,
            "dynamic_launch_scope": "all_in_profiler_roi",
            "dynamic_launch_count_source": "runtime_inventory",
            "profiler_range": "cudaProfilerStartStop",
            "setup_outside_profiler_range": True,
            "timing_methodology": "cuda_events",
        }
        expected_trace_cardinality = (
            "one_selected_workload_candidate_call_all_dynamic_launches_inventoried"
        )
    if invocation.get("capture_target") != expected_target:
        raise ChunkRefusal("sol_capture_target_invalid", repr(identity))
    if (
        invocation.get("evaluation_cardinality_contract")
        != "all_effective_workloads_passed_exact"
        or invocation.get("trace_cardinality_contract")
        != expected_trace_cardinality
    ):
        raise ChunkRefusal("sol_cardinality_contract_invalid", repr(identity))
    if workload_scope == "native":
        expected_argv = [
            logical["benchmark_dir"], "--solution",
            invocation["authority_reference_path"], "--json",
        ]
    else:
        expected_argv = [
            "--definition", logical["definition_file"],
            "--workload", logical["workload_file"],
            "--solution", invocation["authority_reference_path"], "--json",
        ]
    if forwarded != expected_argv:
        raise ChunkRefusal(
            "sol_adapter_argv_rederivation_mismatch",
            f"observed={forwarded!r}, expected={expected_argv!r}",
        )

    control = cell.get("control_bits") or {}
    expected_control = {
        "policy": "post_trace_module_or_typed",
        "module_discovery": "nvbit_runtime_module_inventory",
        "required_outcomes": list(SOL_CONTROL_OUTCOMES),
        "static_rebind": SOL_STATIC_CONTROL_REBIND,
    }
    if control != expected_control:
        raise ChunkRefusal(
            "sol_control_policy_invalid", repr(control)
        )
    _validated_sol_static_tools(control["static_rebind"])
    environment = contract.get("environment") or {}
    native_env_hash = contract.get("native_env_hash")
    if (
        not isinstance(native_env_hash, str) or len(native_env_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in native_env_hash)
    ):
        raise ChunkRefusal("sol_native_environment_hash_invalid", repr(identity))
    if evidence.get("native_env_hash") != native_env_hash:
        raise ChunkRefusal("sol_evaluation_environment_mismatch", repr(identity))
    cutlass = _resolve_repo_path(environment.get("CUTLASS_DIR"), "SOL CUTLASS_DIR")
    path_prepend = _resolve_repo_path(
        environment.get("PATH_prepend"), "SOL PATH prefix"
    )
    environment_keys = frozenset(environment)
    try:
        if environment_keys == sol_env.LEGACY_MATRIX_ENVIRONMENT_KEYS:
            expected_compiler_environment = (
                sol_env.legacy_certifying_environment_contract(environment)
            )
            cuda_home: Path | None = None
            cuda_bin: Path | None = None
            expected_path_prefix = [str(path_prepend)]
        elif environment_keys == sol_env.MATRIX_ENVIRONMENT_KEYS:
            expected_compiler_environment = (
                sol_env.certifying_environment_contract(environment)
            )
            cuda_home = Path(environment["CUDA_HOME"])
            cuda_bin = (cuda_home / "bin").resolve()
            expected_path_prefix = [str(cuda_bin), str(path_prepend)]
        else:
            raise sol_env.SolEnvironmentRefusal(
                "SOL_ENV_MATRIX_CONTRACT_INVALID", repr(environment)
            )
    except sol_env.SolEnvironmentRefusal as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    if (
        not cutlass.is_dir()
        or not path_prepend.is_dir()
        or (
            cuda_bin is not None
            and (
                not (cuda_bin / "nvcc").is_file()
                or not os.access(cuda_bin / "nvcc", os.X_OK)
            )
        )
    ):
        raise ChunkRefusal(
            "sol_adapter_environment_invalid",
            f"CUTLASS_DIR={cutlass}, PATH_prepend={path_prepend}, "
            f"CUDA_HOME={cuda_home}",
        )
    if not isinstance(certified_sol_env, dict):
        raise ChunkRefusal("SOL_ENV_ATTESTATION_MISSING", repr(identity))
    runtime_env = dict(certified_sol_env)
    runtime_env.pop("FLASHINFER_TRACE_DIR", None)
    if flashinfer_runtime_root is not None:
        runtime_env["FLASHINFER_TRACE_DIR"] = str(flashinfer_runtime_root)
    if (
        any(
            runtime_env.get(key) != value
            for key, value in expected_compiler_environment.items()
        )
        or runtime_env.get("CUTLASS_DIR") != str(cutlass)
        or runtime_env.get("PATH", "").split(os.pathsep)[
            :len(expected_path_prefix)
        ] != expected_path_prefix
        or runtime_env.get("PYTORCH_ALLOC_CONF")
        != "expandable_segments:True"
        or any(name in runtime_env for name in sol_env.HAZARD_VARIABLES)
        or any(name in runtime_env for name in sol_env.CAPTURE_CONTROL_VARIABLES)
    ):
        raise ChunkRefusal(
            "SOL_ENV_CONSTRUCTION_FAILED", repr(identity)
        )

    budget = require_cell_budget(cell_budgets, identity["slug"])
    matrix_budget = cell.get("trace_budget") or {}
    for field in ("max_trace_bytes", "max_wall_time_s", "sampling_policy"):
        if matrix_budget.get(field) != getattr(budget, field):
            raise ChunkRefusal(
                "cell_budget_mismatch",
                f"{identity}: matrix {field}={matrix_budget.get(field)!r} != "
                f"config {getattr(budget, field)!r}",
            )
    return adapter, invocation, budget, REPO_ROOT, runtime_env


def validate_capture_cell(
    cell: dict, *, cell_budgets: Any,
    certified_sol_env: dict[str, str] | None = None,
    workload_row_binding: dict[str, Any] | None = None,
    supplemental_runnability_evidence: dict[str, Any] | None = None,
    runnability_path: Path | None = None,
    supplemental_workload_path: Path | None = None,
    supplemental_definition_path: Path | None = None,
) -> tuple[Path, dict, Any, Path, dict[str, str] | None]:
    identity = cell.get("identity") or {}
    source = identity.get("roster_source")
    adapter_kind = (cell.get("invocation") or {}).get("adapter_kind")
    if source == "sol_execbench":
        if adapter_kind != "sol_execbench":
            raise ChunkRefusal(
                "adapter_source_mismatch", f"{source!r} + {adapter_kind!r}"
            )
        return validate_sol_cell(
            cell, cell_budgets=cell_budgets,
            certified_sol_env=certified_sol_env,
            workload_row_binding=workload_row_binding,
            supplemental_runnability_evidence=(
                supplemental_runnability_evidence
            ),
            runnability_path=runnability_path,
            supplemental_workload_path=supplemental_workload_path,
            supplemental_definition_path=supplemental_definition_path,
        )
    if source in {"portable_lcd", "simbench_bw"}:
        if adapter_kind not in _DIRECT_ADAPTERS:
            raise ChunkRefusal(
                "adapter_source_mismatch", f"{source!r} + {adapter_kind!r}"
            )
        binary, invocation, budget = validate_direct_cell(
            cell, cell_budgets=cell_budgets
        )
        return binary, invocation, budget, binary.parent, None
    raise ChunkRefusal("adapter_source_unknown", repr(source))


def live_capture_guard(pins: dict) -> dict:
    processes_pre = preflight.probe_capture_processes()
    if processes_pre.get("status") != preflight.STATUS_OK:
        raise ChunkRefusal(
            "capture_process_conflict", json.dumps(processes_pre, sort_keys=True)
        )
    clocks = preflight.wait_for_clock_state(pins)
    if clocks.get("status") != preflight.STATUS_OK:
        raise ChunkRefusal("clock_state_drift", json.dumps(clocks, sort_keys=True))
    # The wait is itself a race window, so close it with another process
    # snapshot before admitting the next profiler phase.
    processes_post = preflight.probe_capture_processes()
    if processes_post.get("status") != preflight.STATUS_OK:
        raise ChunkRefusal(
            "capture_process_conflict", json.dumps(processes_post, sort_keys=True)
        )
    return {
        "capture_processes": processes_post,
        "capture_processes_before_clock_wait": processes_pre,
        "clock_state": clocks,
    }


def _run_logged(
    command: list[object], *, cwd: Path, env: dict[str, str] | None,
    timeout_s: float, log_dir: Path, stem: str, failure_code: str,
    monitored_tree: Path | None = None,
    tree_limit_bytes: int | None = None,
    trusted_child_storage_marker: bool = False,
    supervised_ncu_identity: ncu_executor.ExpectedNcuIdentity | None = None,
    supervised_ncu_environment: ncu_executor.NcuCollectionEnvironment | None = None,
) -> tuple[float, Path, Path]:
    _mkdir_output(
        log_dir, parents=True, exist_ok=True,
        code=f"{failure_code}_storage_failure",
        operation="capture_log_directory",
    )
    stdout_path = log_dir / f"{stem}.stdout.log"
    stderr_path = log_dir / f"{stem}.stderr.log"

    def write_logs(stdout: bytes, stderr: bytes) -> None:
        _write_output_bytes(
            stdout_path, stdout,
            code=f"{failure_code}_storage_failure",
            operation="capture_stdout_log",
        )
        _write_output_bytes(
            stderr_path, stderr,
            code=f"{failure_code}_storage_failure",
            operation="capture_stderr_log",
        )

    def refuse_child_storage(stderr: bytes) -> None:
        if not trusted_child_storage_marker:
            return
        child_error = _label_driver_storage_error(stderr)
        if child_error is not None:
            raise ChunkRefusal(
                f"{failure_code}_storage_failure",
                f"operation=label_driver_output; errno={child_error.errno}",
            ) from child_error

    started = time.perf_counter()
    if timeout_s <= 0:
        raise ChunkRefusal(
            f"{failure_code}_timeout", f"non-positive timeout_s={timeout_s}"
        )
    if (monitored_tree is None) != (tree_limit_bytes is None):
        raise ChunkRefusal(
            f"{failure_code}_tree_limit_invalid",
            "monitored_tree and tree_limit_bytes must be provided together",
        )
    if tree_limit_bytes is not None and (
        type(tree_limit_bytes) is not int or tree_limit_bytes <= 0
    ):
        raise ChunkRefusal(
            f"{failure_code}_tree_limit_invalid",
            f"tree_limit_bytes={tree_limit_bytes!r}",
        )
    if (supervised_ncu_identity is None) != (
        supervised_ncu_environment is None
    ):
        raise ChunkRefusal(
            f"{failure_code}_supervision_invalid",
            "NCU identity and collection environment must be supplied together",
        )
    if supervised_ncu_identity is not None:
        if env is not None or monitored_tree is not None:
            raise ChunkRefusal(
                f"{failure_code}_supervision_invalid",
                "supervised NCU execution owns env and does not monitor a tree",
            )
        try:
            result = ncu_executor.execute_ncu_supervised_child(
                supervised_ncu_identity,
                command,
                timeout_seconds=timeout_s,
                environment=supervised_ncu_environment,
                working_directory=cwd,
            )
        except ncu_executor.NcuActionExecutorError as exc:
            refuse_child_storage(exc.stderr)
            separator = b"" if not exc.stderr or exc.stderr.endswith(b"\n") else b"\n"
            diagnostic = exc.stderr + separator + f"{exc}\n".encode()
            write_logs(exc.stdout, diagnostic)
            if exc.code == ncu_executor.NCU_ACTION_TIMEOUT:
                code = f"{failure_code}_timeout"
            elif exc.code == ncu_executor.NCU_ACTION_OUTPUT_LIMIT:
                code = f"{failure_code}_output_limit"
            elif exc.code in {
                ncu_executor.NCU_EXECUTABLE_IDENTITY_INVALID,
                ncu_executor.NCU_VERSION_PROBE_FAILED,
                ncu_executor.NCU_VERSION_LINE_MISMATCH,
                ncu_executor.NCU_IDENTITY_INPUT_INVALID,
            }:
                code = "capture_ncu_authority_invalid"
            else:
                code = failure_code
            raise ChunkRefusal(code, str(exc)) from exc
        elapsed = time.perf_counter() - started
        refuse_child_storage(result.stderr)
        write_logs(result.stdout, result.stderr)
        if result.returncode != 0:
            tail = result.stderr.decode(errors="replace")[-1200:]
            raise ChunkRefusal(
                failure_code,
                f"returncode={result.returncode}; stderr_tail={tail!r}",
            )
        return elapsed, stdout_path, stderr_path
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        write_logs(b"", f"{type(exc).__name__}: {exc}\n".encode())
        raise ChunkRefusal(
            f"{failure_code}_start_failed", f"{type(exc).__name__}: {exc}"
        ) from exc

    def terminate_group() -> tuple[bytes, bytes]:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.communicate()

    deadline = time.monotonic() + timeout_s
    timeout_error: subprocess.TimeoutExpired | None = None
    limit_detail: str | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_error = subprocess.TimeoutExpired(command, timeout_s)
                break
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                timeout_error = None
                break
            except subprocess.TimeoutExpired as exc:
                timeout_error = exc
                if monitored_tree is not None:
                    try:
                        observed_bytes = _trace_tree_bytes(monitored_tree)
                    except (OSError, ChunkRefusal) as monitor_exc:
                        stdout, stderr = terminate_group()
                        storage_origin = _storage_origin(monitor_exc)
                        if storage_origin is not None:
                            # The monitored filesystem has reported a closed
                            # capacity/integrity fault.  Keep captured process
                            # output in memory and do not attempt log writes to
                            # that same failed medium.  The supplemental outer
                            # boundary types the preserved OSError cause and
                            # records its receipt on the independent runtime fs.
                            raise ChunkRefusal(
                                f"{failure_code}_storage_monitor_failed",
                                f"errno={storage_origin.errno}; "
                                f"tree={monitored_tree}",
                            ) from storage_origin
                        refuse_child_storage(stderr or b"")
                        write_logs(stdout or b"", stderr or b"")
                        raise ChunkRefusal(
                            f"{failure_code}_tree_monitor_failed",
                            f"{type(monitor_exc).__name__}: {monitor_exc}",
                        ) from monitor_exc
                    if observed_bytes > tree_limit_bytes:
                        limit_detail = (
                            f"observed={observed_bytes}, limit={tree_limit_bytes}, "
                            f"tree={monitored_tree}"
                        )
                        break
        if timeout_error is not None:
            stdout, stderr = terminate_group()
            refuse_child_storage(stderr or b"")
            write_logs(stdout or b"", stderr or b"")
            if limit_detail is not None:
                raise ChunkRefusal(
                    f"{failure_code}_tree_limit_exceeded", limit_detail
                )
            raise ChunkRefusal(
                f"{failure_code}_timeout", f"timeout_s={timeout_s}: {command[0]}"
            ) from timeout_error
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = terminate_group()
        refuse_child_storage(stderr or b"")
        write_logs(stdout or b"", stderr or b"")
        raise ChunkRefusal(
            f"{failure_code}_timeout", f"timeout_s={timeout_s}: {command[0]}"
        ) from exc
    elapsed = time.perf_counter() - started
    observed_bytes: int | None = None
    if monitored_tree is not None:
        try:
            observed_bytes = _trace_tree_bytes(monitored_tree)
        except (OSError, ChunkRefusal) as monitor_exc:
            storage_origin = _storage_origin(monitor_exc)
            if storage_origin is not None:
                # As in the polling path above, do not write captured logs
                # after a final monitored-tree read reports a storage fault.
                raise ChunkRefusal(
                    f"{failure_code}_storage_monitor_failed",
                    f"errno={storage_origin.errno}; tree={monitored_tree}",
                ) from storage_origin
            raise
    refuse_child_storage(stderr)
    write_logs(stdout, stderr)
    if observed_bytes is not None:
        if observed_bytes > tree_limit_bytes:
            raise ChunkRefusal(
                f"{failure_code}_tree_limit_exceeded",
                f"observed={observed_bytes}, limit={tree_limit_bytes}, "
                f"tree={monitored_tree}",
            )
    if process.returncode != 0:
        tail = stderr.decode(errors="replace")[-1200:]
        raise ChunkRefusal(
            failure_code, f"returncode={process.returncode}; stderr_tail={tail!r}"
        )
    return elapsed, stdout_path, stderr_path


def _write_json_exclusive(path: Path, payload: dict) -> None:
    _mkdir_output(
        path.parent, parents=True, exist_ok=True,
        operation="capture_json_parent_directory",
    )
    if path.exists():
        raise ChunkRefusal("capture_evidence_exists", str(path))
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _write_output_text(
        temp, json.dumps(payload, indent=2, sort_keys=True) + "\n",
        operation="capture_json_temporary_write",
    )
    _replace_output(
        temp, path, operation="capture_json_exclusive_publish",
    )


def _sol_stage_entry_path(stage: Path, entry: Any, label: str) -> Path:
    if not isinstance(entry, dict):
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} not object")
    raw = entry.get("path")
    expected = entry.get("sha256")
    relative = Path(raw) if isinstance(raw, str) else Path(".")
    if (
        not isinstance(raw, str) or not raw or relative.is_absolute()
        or relative == Path(".") or ".." in relative.parts
    ):
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} path={raw!r}")
    declared = stage / raw
    observed = _output_lstat(
        declared, code="sol_phase_storage_failure",
        operation=f"sol_stage_entry_stat:{label}",
    )
    if observed is not None and stat.S_ISLNK(observed.st_mode):
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} is a symlink")
    if observed is None:
        raise ChunkRefusal("sol_stage_artifact_absent", str(declared))
    candidate = _resolve_output_path(
        declared, code="sol_phase_storage_failure",
        operation=f"sol_stage_entry_resolve:{label}",
    )
    try:
        candidate.relative_to(stage.resolve())
    except ValueError as exc:
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} escapes") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise ChunkRefusal("sol_stage_artifact_absent", str(candidate))
    actual = _sha256_output_file(
        candidate, code="sol_phase_storage_failure",
        operation=f"sol_stage_entry_hash:{label}",
    )
    if actual != expected:
        raise ChunkRefusal(
            "input_hash_mismatch",
            f"SOL stage {label}: expected {expected}, observed {actual}: {candidate}",
        )
    return candidate


def _sol_external_entry_path(
    entry: Any, label: str, *, expected_path: Path,
) -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} not exact")
    raw = entry.get("path")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} path={raw!r}")
    declared = Path(raw)
    if declared.is_symlink():
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} is a symlink")
    try:
        candidate = declared.resolve(strict=True)
    except OSError as exc:
        raise ChunkRefusal("sol_stage_artifact_absent", raw) from exc
    if candidate != expected_path.resolve() or not candidate.is_file():
        raise ChunkRefusal("sol_stage_manifest_invalid", f"{label} path drift")
    _require_hash(candidate, entry.get("sha256"), f"SOL stage {label}")
    return candidate


def _sol_manifest_sha256(value: dict) -> str:
    try:
        raw = (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ChunkRefusal("sol_stage_manifest_invalid", str(exc)) from exc
    return hashlib.sha256(raw).hexdigest()


def _sol_certifying_environment(invocation: dict) -> tuple[dict[str, str], str]:
    matrix_environment = (
        (invocation.get("adapter_contract") or {}).get("environment") or {}
    )
    try:
        keys = frozenset(matrix_environment)
        if keys == sol_env.LEGACY_MATRIX_ENVIRONMENT_KEYS:
            contract = sol_env.legacy_certifying_environment_contract(
                matrix_environment
            )
            digest = sol_env.legacy_environment_contract_sha256(contract)
        elif keys == sol_env.MATRIX_ENVIRONMENT_KEYS:
            contract = sol_env.certifying_environment_contract(matrix_environment)
            digest = sol_env.environment_contract_sha256(contract)
        else:
            raise sol_env.SolEnvironmentRefusal(
                "SOL_ENV_MATRIX_CONTRACT_INVALID", repr(matrix_environment)
            )
    except sol_env.SolEnvironmentRefusal as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    return contract, digest


def _sol_flashinfer_corpus_contract(
    invocation: dict,
) -> tuple[dict[str, Any], str, Path | None]:
    """Re-derive the exact category-scoped FlashInfer capture operand.

    The adapter performs the expensive corpus byte re-proof before it mutates
    its stage.  This runner independently closes the request/manifest shape,
    digest, and runtime-environment policy so an ambient trace directory can
    neither enter a non-FIB workload nor replace the verified FIB root.
    """
    raw = (invocation.get("adapter_contract") or {}).get(
        "flashinfer_corpus"
    )
    try:
        contract = sol_recon._validate_flashinfer_capture_contract_shape(raw)
    except (TypeError, ValueError, sol_recon.PrerequisiteContractError) as exc:
        raise ChunkRefusal(
            "SOL_FIB_CORPUS_CONTRACT_INVALID", str(exc)
        ) from exc
    normalized = copy.deepcopy(contract)
    digest = _sol_manifest_sha256(normalized)
    runtime_root = (
        (REPO_ROOT / normalized["root"]).resolve()
        if normalized["category"] == "FIB" else None
    )
    if runtime_root is not None and (
        runtime_root.is_symlink() or not runtime_root.is_dir()
    ):
        raise ChunkRefusal(
            "SOL_FIB_CORPUS_CONTRACT_INVALID", str(runtime_root)
        )
    return normalized, digest, runtime_root


def _validate_sol_prepare_manifest(
    *, cell: dict, invocation: dict, stage: Path,
) -> dict:
    manifest_path = stage / "capture_prepare_manifest.json"
    manifest = _json(manifest_path)
    definition_attestation = manifest.get("definition_equivalence")
    if (
        not isinstance(definition_attestation, dict)
        or definition_attestation.get("rule_id")
        != sol_def.CERTIFYING_RULE_ID
    ):
        raise ChunkRefusal(
            "SOL_ATTESTATION_RULE_MISMATCH",
            repr(
                definition_attestation.get("rule_id")
                if isinstance(definition_attestation, dict)
                else None
            ),
        )
    try:
        sol_def.validate_attestation_shape_and_eligibility(
            definition_attestation
        )
    except sol_def.SolDefinitionEquivalenceError as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    expected_manifest_fields = {
        "schema_version", "manifest_kind", "capture_contract_version",
        "path_mode", "stage_root", "source_inputs", "selection",
        "solution_contract", "benchmark_config", "environment_contract",
        "environment_contract_sha256", "flashinfer_corpus_contract",
        "flashinfer_corpus_contract_sha256", "artifacts", "execution",
        "stage_definition_sha256", "source_definition_sha256",
        "staged_definition_sha256", "definition_equivalence",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("capture_contract_version") != 2
        or manifest.get("manifest_kind")
        != "sol_execbench_capture_preparation"
        or manifest.get("path_mode") != "stage_relative_v1"
        or manifest.get("stage_root") != "."
        or "stage_dir" in manifest
    ):
        raise ChunkRefusal("sol_stage_manifest_invalid", repr(cell["identity"]))
    environment_contract, environment_contract_sha256 = (
        _sol_certifying_environment(invocation)
    )
    if (
        manifest.get("environment_contract") != environment_contract
        or manifest.get("environment_contract_sha256")
        != environment_contract_sha256
    ):
        raise ChunkRefusal(
            "sol_stage_environment_contract_mismatch", repr(cell["identity"])
        )
    flashinfer_contract, flashinfer_contract_sha256, _runtime_root = (
        _sol_flashinfer_corpus_contract(invocation)
    )
    if (
        manifest.get("flashinfer_corpus_contract") != flashinfer_contract
        or manifest.get("flashinfer_corpus_contract_sha256")
        != flashinfer_contract_sha256
    ):
        raise ChunkRefusal(
            "SOL_FIB_CORPUS_CONTRACT_INVALID", repr(cell["identity"])
        )
    unsigned_manifest = dict(manifest)
    claimed_stage_definition_sha256 = unsigned_manifest.pop(
        "stage_definition_sha256", None
    )
    if (
        claimed_stage_definition_sha256
        != _sol_manifest_sha256(unsigned_manifest)
    ):
        raise ChunkRefusal(
            "sol_stage_definition_hash_mismatch", repr(cell["identity"])
        )
    logical = invocation["logical_workload"]
    sources = manifest.get("source_inputs") or {}
    expected_sources = {
        "definition": (
            _resolve_repo_path(logical["definition_file"], "SOL definition"),
            logical["definition_file_sha256"],
        ),
        "workload": (
            _resolve_repo_path(logical["workload_file"], "SOL workload"),
            logical["workload_file_sha256"],
        ),
        "solution": (
            _resolve_repo_path(
                invocation["authority_reference_path"], "SOL solution"
            ),
            invocation["authority_reference_sha256"],
        ),
    }
    config_path = (
        _resolve_repo_path(logical["benchmark_dir"], "SOL benchmark directory")
        / "config.json"
    )
    if logical.get("workload_scope") == "native" and config_path.is_file():
        expected_sources["config"] = (config_path.resolve(), sha256_file(config_path))
    if set(sources) != set(expected_sources):
        raise ChunkRefusal(
            "sol_stage_source_mismatch",
            f"sources={sorted(sources)}, expected={sorted(expected_sources)}",
        )
    for name, (path, expected_sha) in expected_sources.items():
        entry = sources.get(name)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or Path(entry.get("path", "")).resolve() != path
            or entry.get("sha256") != expected_sha
        ):
            raise ChunkRefusal("sol_stage_source_mismatch", name)
        _require_hash(path, expected_sha, f"SOL source {name}")
    artifacts = manifest.get("artifacts") or {}
    expected_artifact_fields = {
        "definition", "definition_source", "workload", "solution", "adapter",
        "adapter_source",
        "certified_env_source", "eval_driver", "capture_runtime",
        "benchmark_kernel_so", "definition_equivalence_attestation",
        "solution_sources",
    }
    if set(artifacts) != expected_artifact_fields:
        raise ChunkRefusal(
            "sol_stage_manifest_invalid",
            f"artifact fields={sorted(artifacts)}",
        )
    resolved_artifacts = {
        name: _sol_stage_entry_path(stage, artifacts.get(name), name)
        for name in (
            "definition", "workload", "solution", "eval_driver",
            "capture_runtime", "benchmark_kernel_so",
        )
    }
    source_definition_path, pinned_source_definition_sha256 = (
        expected_sources["definition"]
    )
    source_copy_entry = artifacts.get("definition_source")
    if source_copy_entry is None:
        raise ChunkRefusal(
            "sol_stage_source_copy_missing",
            str(stage / "definition.source.json"),
        )
    if (
        not isinstance(source_copy_entry, dict)
        or set(source_copy_entry) != {"path", "sha256"}
        or source_copy_entry.get("path") != "definition.source.json"
    ):
        raise ChunkRefusal(
            "sol_stage_source_copy_mismatch", repr(source_copy_entry)
        )
    source_copy_path = stage / "definition.source.json"
    if (
        source_copy_path.is_symlink()
        or not source_copy_path.is_file()
    ):
        raise ChunkRefusal(
            "sol_stage_source_copy_missing", str(source_copy_path)
        )
    source_copy_sha256 = sha256_file(source_copy_path)
    if (
        source_copy_sha256 != pinned_source_definition_sha256
        or source_copy_entry.get("sha256") != source_copy_sha256
        or source_copy_path.read_bytes() != source_definition_path.read_bytes()
    ):
        raise ChunkRefusal(
            "sol_stage_source_copy_mismatch",
            f"pinned={pinned_source_definition_sha256}; "
            f"copy={source_copy_sha256}",
        )
    resolved_artifacts["definition_source"] = source_copy_path.resolve()
    attestation_entry = artifacts.get("definition_equivalence_attestation")
    if (
        not isinstance(attestation_entry, dict)
        or set(attestation_entry) != {"path", "sha256"}
        or attestation_entry.get("path")
        != "definition_equivalence_attestation.json"
    ):
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            f"attestation artifact pin={attestation_entry!r}",
        )
    attestation_path = stage / "definition_equivalence_attestation.json"
    if attestation_path.is_symlink() or not attestation_path.is_file():
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH", str(attestation_path)
        )
    attestation_sha256 = sha256_file(attestation_path)
    if attestation_entry.get("sha256") != attestation_sha256:
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "attestation artifact hash does not match its manifest pin",
        )
    resolved_artifacts[
        "definition_equivalence_attestation"
    ] = attestation_path.resolve()
    exact_stage_paths = {
        "definition": "definition.json",
        "definition_source": "definition.source.json",
        "definition_equivalence_attestation": (
            "definition_equivalence_attestation.json"
        ),
        "workload": "workload.jsonl",
        "solution": "solution.json",
        "eval_driver": "eval_driver.py",
        "capture_runtime": "_capture_runtime.py",
        "benchmark_kernel_so": "benchmark_kernel.so",
    }
    for name, expected_relpath in exact_stage_paths.items():
        if (
            not isinstance(artifacts.get(name), dict)
            or set(artifacts[name]) != {"path", "sha256"}
            or artifacts[name].get("path") != expected_relpath
        ):
            raise ChunkRefusal(
                "sol_stage_manifest_invalid", f"artifact path drift: {name}"
            )
    expected_adapter = _resolve_repo_path(
        invocation["executable_path"], "SOL adapter"
    )
    adapter_entry = artifacts.get("adapter")
    if (
        not isinstance(adapter_entry, dict)
        or set(adapter_entry) != {"path", "sha256"}
        or not isinstance(adapter_entry.get("path"), str)
        or Path(adapter_entry["path"]).resolve() != expected_adapter
        or adapter_entry.get("sha256") != invocation["executable_sha256"]
    ):
        raise ChunkRefusal("sol_stage_adapter_mismatch", repr(cell["identity"]))
    _sol_external_entry_path(
        adapter_entry, "adapter", expected_path=expected_adapter,
    )
    expected_adapter_source = (
        REPO_ROOT / "tools" / "sol_execbench" / "src" / "sol_execbench"
        / "driver" / "capture_adapter.py"
    ).resolve()
    _sol_external_entry_path(
        artifacts.get("adapter_source"), "adapter_source",
        expected_path=expected_adapter_source,
    )
    expected_certified_env_source = expected_adapter_source.with_name(
        "certified_env.py"
    )
    _sol_external_entry_path(
        artifacts.get("certified_env_source"), "certified_env_source",
        expected_path=expected_certified_env_source,
    )

    try:
        definition_comparison = sol_def.require_definition_equivalent(
            source_copy_path.read_bytes(),
            resolved_artifacts["definition"].read_bytes(),
        )
    except sol_def.SolDefinitionEquivalenceError as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    expected_attestation = definition_comparison.attestation
    try:
        ineligible_normalized_pointers = (
            sol_def.ineligible_normalized_state_pointers(
                definition_attestation, expected_attestation
            )
        )
    except sol_def.SolDefinitionEquivalenceError as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    if ineligible_normalized_pointers:
        raise ChunkRefusal(
            "SOL_ATTESTATION_POINTER_INELIGIBLE",
            f"normalized state positions outside the source/staged "
            f"eligibility set: {list(ineligible_normalized_pointers)!r}",
        )
    if (
        manifest.get("source_definition_sha256")
        != definition_comparison.source_sha256
        or definition_attestation.get("source_definition_sha256")
        != definition_comparison.source_sha256
    ):
        raise ChunkRefusal(
            "sol_stage_source_copy_mismatch",
            "manifest source_definition_sha256 does not rederive",
        )
    if (
        manifest.get("staged_definition_sha256")
        != definition_comparison.staged_sha256
        or definition_attestation.get("staged_definition_sha256")
        != definition_comparison.staged_sha256
        or definition_attestation != expected_attestation
    ):
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "staged definition SHA or equivalence attestation does not rederive",
        )
    try:
        expected_attestation_bytes = sol_def.serialize_attestation(
            expected_attestation
        )
    except sol_def.SolDefinitionEquivalenceError as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    if attestation_path.read_bytes() != expected_attestation_bytes:
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "attestation artifact bytes are not the deterministic V2 encoding",
        )
    source_solution = _json(expected_sources["solution"][0])
    staged_solution = _json(resolved_artifacts["solution"])
    if (
        staged_solution.get("name") != source_solution.get("name")
        or staged_solution.get("definition") != source_solution.get("definition")
        or staged_solution.get("sources") != source_solution.get("sources")
    ):
        raise ChunkRefusal(
            "sol_stage_source_mismatch", "staged solution identity drift"
        )
    source_specs = source_solution.get("sources")
    staged_source_entries = artifacts.get("solution_sources")
    if (
        not isinstance(source_specs, list)
        or not source_specs
        or not isinstance(staged_source_entries, list)
        or len(staged_source_entries) != len(source_specs)
    ):
        raise ChunkRefusal(
            "sol_stage_solution_sources_invalid", repr(cell["identity"])
        )
    seen_solution_paths: set[str] = set()
    for index, (source_spec, entry) in enumerate(
        zip(source_specs, staged_source_entries, strict=True)
    ):
        source_rel = source_spec.get("path") if isinstance(source_spec, dict) else None
        if (
            not isinstance(source_rel, str)
            or not source_rel
            or source_rel in seen_solution_paths
            or Path(source_rel).is_absolute()
            or ".." in Path(source_rel).parts
            or not isinstance(entry, dict)
            or set(entry) != {"relative_path", "path", "sha256"}
            or entry.get("relative_path") != source_rel
            or entry.get("path") != source_rel
        ):
            raise ChunkRefusal(
                "sol_stage_solution_sources_invalid", f"source[{index}]"
            )
        staged_source = _sol_stage_entry_path(
            stage, {"path": entry["path"], "sha256": entry["sha256"]},
            f"solution_sources[{index}]",
        )
        content = source_spec.get("content")
        if not content:
            fallback_source = expected_sources["solution"][0].parent / source_rel
            if not fallback_source.is_file() or fallback_source.is_symlink():
                raise ChunkRefusal(
                    "sol_stage_solution_sources_invalid", str(fallback_source)
                )
            content = fallback_source.read_text()
        if not isinstance(content, str) or staged_source.read_text() != content:
            raise ChunkRefusal(
                "sol_stage_solution_sources_invalid", f"content[{index}]"
            )
        seen_solution_paths.add(source_rel)

    expected_benchmark_config = {
        "warmup_runs": 10,
        "iterations": 50,
        "lock_clocks": False,
        "benchmark_reference": True,
        "seed": 200,
    }
    if "config" in expected_sources:
        config_document = _json(expected_sources["config"][0])
        if not set(config_document) <= set(expected_benchmark_config):
            raise ChunkRefusal("sol_stage_config_invalid", repr(config_document))
        expected_benchmark_config.update(config_document)
    if manifest.get("benchmark_config") != expected_benchmark_config:
        raise ChunkRefusal(
            "sol_stage_config_invalid", repr(manifest.get("benchmark_config"))
        )
    solution_spec = source_solution.get("spec") or {}
    expected_solution_contract = {
        "definition": source_solution.get("definition"),
        "solution": source_solution.get("name"),
        "languages": solution_spec.get("languages"),
        "entry_point": solution_spec.get("entry_point"),
        "destination_passing_style": True,
    }
    if manifest.get("solution_contract") != expected_solution_contract:
        raise ChunkRefusal(
            "sol_stage_solution_contract_invalid",
            repr(manifest.get("solution_contract")),
        )
    selection = manifest.get("selection") or {}
    target = invocation["capture_target"]
    if (
        set(selection) != {
            "source_workload_index", "source_workload_count",
            "staged_workload_index", "workload_uuid", "workload_axes",
            "source_row_sha256",
        }
        or selection.get("source_workload_index")
        != target["source_workload_index"]
        or selection.get("source_workload_count")
        != logical["expected_workloads"]
        or selection.get("staged_workload_index")
        != target["staged_workload_index"]
    ):
        raise ChunkRefusal("sol_stage_selection_mismatch", repr(selection))
    source_rows = [
        line.strip()
        for line in expected_sources["workload"][0].read_bytes().splitlines()
        if line.strip()
    ]
    staged_rows = [
        line.strip() for line in resolved_artifacts["workload"].read_bytes().splitlines()
        if line.strip()
    ]
    source_index = target["source_workload_index"]
    if source_index >= len(source_rows):
        raise ChunkRefusal(
            "sol_stage_workload_row_mismatch", "source workload index out of range"
        )
    selected_source_row = source_rows[source_index]
    try:
        selected_workload = json.loads(selected_source_row)
    except json.JSONDecodeError as exc:
        raise ChunkRefusal("sol_stage_workload_row_mismatch", str(exc)) from exc
    if (
        len(staged_rows) != 1
        or staged_rows[0] != selected_source_row
        or hashlib.sha256(selected_source_row).hexdigest()
        != selection.get("source_row_sha256")
        or not isinstance(selected_workload, dict)
        or selection.get("workload_uuid") != selected_workload.get("uuid")
        or selection.get("workload_axes") != selected_workload.get("axes")
    ):
        raise ChunkRefusal("sol_stage_workload_row_mismatch", repr(cell["identity"]))
    execution = manifest.get("execution") or {}
    execution_command = execution.get("command")
    if (
        not isinstance(execution_command, list)
        or len(execution_command) != 2
        or not Path(execution_command[0]).is_absolute()
        or not Path(execution_command[0]).is_file()
        or execution_command[1] != "eval_driver.py"
        or {key: value for key, value in execution.items() if key != "command"}
        != {
            "working_directory": ".",
            "timing_methodology": "cuda_events",
            "correctness_rounds": 1,
            "warmup_calls": 1,
            "profiled_candidate_calls": 1,
        }
    ):
        raise ChunkRefusal("sol_stage_execution_contract_invalid", repr(execution))
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "environment_contract": environment_contract,
        "environment_contract_sha256": environment_contract_sha256,
        "flashinfer_corpus_contract": flashinfer_contract,
        "flashinfer_corpus_contract_sha256": flashinfer_contract_sha256,
        "stage_definition_sha256": claimed_stage_definition_sha256,
        "source_definition_sha256": definition_comparison.source_sha256,
        "staged_definition_sha256": definition_comparison.staged_sha256,
        "definition_equivalence": definition_comparison.attestation,
        "definition_equivalence_attestation_path": attestation_path.resolve(),
        "definition_equivalence_attestation_sha256": attestation_sha256,
        "definition_source": source_copy_path.resolve(),
        "benchmark_kernel_so": resolved_artifacts["benchmark_kernel_so"],
        "benchmark_kernel_sha256": artifacts["benchmark_kernel_so"]["sha256"],
    }


def prepare_sol_capture(
    *, cell: dict, binary: Path, invocation: dict, cell_dir: Path,
    runtime_cwd: Path, runtime_env: dict[str, str], timeout_s: float,
) -> dict:
    request_timeout_s = int(timeout_s)
    if request_timeout_s < 1:
        raise ChunkRefusal("cell_deadline_exhausted", "SOL prepare has <1 second")
    stage = cell_dir / "sol_stage"
    request_path = cell_dir / "sol_requests" / "prepare.json"
    environment_contract, environment_contract_sha256 = (
        _sol_certifying_environment(invocation)
    )
    flashinfer_contract, flashinfer_contract_sha256, runtime_root = (
        _sol_flashinfer_corpus_contract(invocation)
    )
    if (
        any(
            runtime_env.get(key) != value
            for key, value in environment_contract.items()
        )
        or any(name in runtime_env for name in sol_env.HAZARD_VARIABLES)
    ):
        raise ChunkRefusal(
            "SOL_ENV_CONSTRUCTION_FAILED", repr(cell["identity"])
        )
    expected_runtime_root = str(runtime_root) if runtime_root is not None else None
    if runtime_env.get("FLASHINFER_TRACE_DIR") != expected_runtime_root:
        raise ChunkRefusal(
            "SOL_FIB_CORPUS_ENVIRONMENT_INVALID",
            repr({
                "category": flashinfer_contract["category"],
                "expected": expected_runtime_root,
                "observed": runtime_env.get("FLASHINFER_TRACE_DIR"),
            }),
        )
    request = {
        "schema_version": 1,
        "action": "prepare",
        "stage_dir": str(stage.resolve()),
        "source_workload_index": invocation["capture_target"][
            "source_workload_index"
        ],
        "compile_timeout_s": request_timeout_s,
        "environment_contract": environment_contract,
        "environment_contract_sha256": environment_contract_sha256,
        "flashinfer_corpus_contract": flashinfer_contract,
        "flashinfer_corpus_contract_sha256": flashinfer_contract_sha256,
    }
    _write_json_exclusive(request_path, request)
    env = dict(runtime_env)
    env["SOL_EXECBENCH_CAPTURE_REQUEST"] = str(request_path.resolve())
    _run_logged(
        [str(binary), *invocation["trace_app_args"]],
        cwd=runtime_cwd, env=env, timeout_s=timeout_s,
        log_dir=cell_dir / "logs", stem="sol_prepare",
        failure_code="sol_capture_prepare_failed",
    )
    stage_evidence = _validate_sol_prepare_manifest(
        cell=cell, invocation=invocation, stage=stage,
    )
    return {
        "stage": stage,
        "request_dir": request_path.parent,
        "base_env": dict(runtime_env),
        **stage_evidence,
    }


def sol_phase_env(
    context: dict, *, phase_tag: str, capture_kind: str, timeout_s: float,
) -> dict[str, str]:
    if capture_kind not in {"label", "trace", "control", "smoke"}:
        raise ChunkRefusal("sol_capture_kind_invalid", capture_kind)
    safe_tag = "".join(
        ch if ch.isalnum() or ch in "_-" else "_" for ch in phase_tag
    )
    if not safe_tag:
        raise ChunkRefusal("sol_capture_phase_invalid", phase_tag)
    request_timeout_s = int(timeout_s)
    if request_timeout_s < 1:
        raise ChunkRefusal(
            "cell_deadline_exhausted", f"SOL phase {phase_tag!r} has <1 second"
        )
    selection_rel = f"evidence/{safe_tag}-selection.json"
    result_rel = f"evidence/{safe_tag}-result.json"
    request_path = context["request_dir"] / f"{safe_tag}.json"
    request = {
        "schema_version": 1,
        "action": "execute",
        "stage_dir": str(context["stage"].resolve()),
        "source_workload_index": context["manifest"]["selection"][
            "source_workload_index"
        ],
        "capture_kind": capture_kind,
        "selection_manifest": selection_rel,
        "result_manifest": result_rel,
        "execution_timeout_s": request_timeout_s,
        "environment_contract": context["environment_contract"],
        "environment_contract_sha256": context[
            "environment_contract_sha256"
        ],
        "flashinfer_corpus_contract": context[
            "flashinfer_corpus_contract"
        ],
        "flashinfer_corpus_contract_sha256": context[
            "flashinfer_corpus_contract_sha256"
        ],
    }
    _write_json_exclusive(request_path, request)
    env = dict(context["base_env"])
    env["SOL_EXECBENCH_CAPTURE_REQUEST"] = str(request_path.resolve())
    return env


def validate_sol_phase_env(
    context: dict, env: dict[str, str], expected_kind: str,
) -> dict:
    if expected_kind not in {"label", "trace", "control", "smoke"}:
        raise ChunkRefusal("sol_capture_kind_invalid", expected_kind)
    raw_request_path = env.get("SOL_EXECBENCH_CAPTURE_REQUEST")
    if not isinstance(raw_request_path, str) or not raw_request_path:
        raise ChunkRefusal("sol_phase_request_invalid", repr(raw_request_path))
    request_path = Path(raw_request_path)
    request_stat = (
        _output_lstat(
            request_path, code="sol_phase_storage_failure",
            operation="sol_phase_request_stat",
        ) if request_path.is_absolute() else None
    )
    if (
        not request_path.is_absolute() or request_stat is None
        or not stat.S_ISREG(request_stat.st_mode)
    ):
        raise ChunkRefusal("sol_phase_request_invalid", str(request_path))
    try:
        _resolve_output_path(
            request_path, code="sol_phase_storage_failure",
            operation="sol_phase_request_resolve",
        ).relative_to(_resolve_output_path(
            context["request_dir"], code="sol_phase_storage_failure",
            operation="sol_phase_request_directory_resolve",
        ))
    except ValueError as exc:
        raise ChunkRefusal(
            "sol_phase_request_invalid", f"request escapes request dir: {request_path}"
        ) from exc
    request = _json_output(
        request_path, code="sol_phase_storage_failure",
        operation="sol_phase_request_read",
    )
    expected_request_fields = {
        "schema_version", "action", "stage_dir", "source_workload_index",
        "capture_kind", "selection_manifest", "result_manifest",
        "execution_timeout_s", "environment_contract",
        "environment_contract_sha256", "flashinfer_corpus_contract",
        "flashinfer_corpus_contract_sha256",
    }
    expected_source_index = context["manifest"]["selection"][
        "source_workload_index"
    ]
    if (
        set(request) != expected_request_fields
        or request.get("schema_version") != 1
        or request.get("action") != "execute"
        or request.get("capture_kind") != expected_kind
        or request.get("source_workload_index") != expected_source_index
        or request.get("environment_contract")
        != context["environment_contract"]
        or request.get("environment_contract_sha256")
        != context["environment_contract_sha256"]
        or request.get("flashinfer_corpus_contract")
        != context["flashinfer_corpus_contract"]
        or request.get("flashinfer_corpus_contract_sha256")
        != context["flashinfer_corpus_contract_sha256"]
        or type(request.get("execution_timeout_s")) is not int
        or request["execution_timeout_s"] < 1
        or not isinstance(request.get("stage_dir"), str)
        or not Path(request["stage_dir"]).is_absolute()
        or Path(request["stage_dir"]).resolve() != context["stage"].resolve()
    ):
        raise ChunkRefusal("sol_phase_request_invalid", repr(request))
    stage = context["stage"]
    evidence_paths: list[Path] = []
    for field in ("selection_manifest", "result_manifest"):
        raw = request.get(field)
        rel = Path(raw) if isinstance(raw, str) else Path(".")
        if (
            not isinstance(raw, str)
            or not raw
            or rel.is_absolute()
            or rel == Path(".")
            or ".." in rel.parts
        ):
            raise ChunkRefusal("sol_phase_request_invalid", repr(request))
        resolved = _resolve_output_path(
            stage / rel, code="sol_phase_storage_failure",
            operation=f"sol_phase_evidence_resolve:{field}",
        )
        try:
            resolved.relative_to(_resolve_output_path(
                stage, code="sol_phase_storage_failure",
                operation="sol_phase_stage_resolve",
            ))
        except ValueError as exc:
            raise ChunkRefusal("sol_phase_request_invalid", f"{field} escapes stage") from exc
        resolved_stat = _output_lstat(
            resolved, code="sol_phase_storage_failure",
            operation=f"sol_phase_evidence_stat:{field}",
        )
        if resolved_stat is None or not stat.S_ISREG(resolved_stat.st_mode):
            raise ChunkRefusal("sol_phase_evidence_absent", str(resolved))
        evidence_paths.append(resolved)
    selection_path, result_path = evidence_paths
    if selection_path == result_path:
        raise ChunkRefusal("sol_phase_request_invalid", "evidence paths alias")
    selection = _json_output(
        selection_path, code="sol_phase_storage_failure",
        operation="sol_phase_selection_read",
    )
    result = _json_output(
        result_path, code="sol_phase_storage_failure",
        operation="sol_phase_result_read",
    )
    expected_selection_fields = {
        "schema_version", "manifest_kind", "capture_contract_version",
        "capture_kind", "path_mode", "stage_root", "prepare_manifest",
        "source_workload_index", "staged_workload_index", "workload",
        "definition", "solution", "environment_contract",
        "environment_contract_sha256", "flashinfer_corpus_contract",
        "flashinfer_corpus_contract_sha256", "stage_definition_sha256",
        "benchmark_kernel_so", "eval_driver", "capture_runtime",
        "result_manifest",
    }
    expected_result_fields = {
        "schema_version", "manifest_kind", "capture_contract_version",
        "path_mode", "stage_root", "status", "capture_kind",
        "selection_manifest", "prepare_manifest", "definition", "solution",
        "source_workload_index", "staged_workload_index", "workload_uuid",
        "workload_axes", "environment_contract",
        "environment_contract_sha256", "flashinfer_corpus_contract",
        "flashinfer_corpus_contract_sha256", "stage_definition_sha256",
        "benchmark_kernel_so", "execution_order", "outside_roi", "roi",
        "correctness",
    }
    if (
        set(selection) != expected_selection_fields
        or set(result) != expected_result_fields
        or selection.get("schema_version") != 1
        or selection.get("capture_contract_version") != 2
        or selection.get("manifest_kind")
        != "sol_execbench_capture_selection"
        or selection.get("path_mode") != "stage_relative_v1"
        or selection.get("stage_root") != "."
        or result.get("schema_version") != 1
        or result.get("capture_contract_version") != 2
        or result.get("manifest_kind") != "sol_execbench_capture_result"
        or result.get("path_mode") != "stage_relative_v1"
        or result.get("stage_root") != "."
        or "stage_dir" in selection
        or "stage_dir" in result
    ):
        raise ChunkRefusal("sol_phase_not_relocatable", expected_kind)
    prepare_entry = {
        "path": "capture_prepare_manifest.json",
        "sha256": context["manifest_sha256"],
    }
    staged_module_entry = context["manifest"]["artifacts"][
        "benchmark_kernel_so"
    ]
    if (
        selection.get("capture_kind") != expected_kind
        or selection.get("source_workload_index") != expected_source_index
        or selection.get("staged_workload_index") != 0
        or selection.get("prepare_manifest") != prepare_entry
        or selection.get("workload") != context["manifest"]["selection"]
        or selection.get("definition")
        != context["manifest"]["solution_contract"]["definition"]
        or selection.get("solution")
        != context["manifest"]["solution_contract"]["solution"]
        or selection.get("environment_contract")
        != context["environment_contract"]
        or selection.get("environment_contract_sha256")
        != context["environment_contract_sha256"]
        or selection.get("flashinfer_corpus_contract")
        != context["flashinfer_corpus_contract"]
        or selection.get("flashinfer_corpus_contract_sha256")
        != context["flashinfer_corpus_contract_sha256"]
        or selection.get("stage_definition_sha256")
        != context["stage_definition_sha256"]
        or selection.get("benchmark_kernel_so") != staged_module_entry
        or selection.get("eval_driver")
        != context["manifest"]["artifacts"]["eval_driver"]
        or selection.get("capture_runtime")
        != context["manifest"]["artifacts"]["capture_runtime"]
        or selection.get("result_manifest") != request["result_manifest"]
    ):
        raise ChunkRefusal("sol_phase_selection_mismatch", expected_kind)
    selection_pin = result.get("selection_manifest") or {}
    prepare_pin = result.get("prepare_manifest") or {}
    if (
        selection_pin.get("path") != request["selection_manifest"]
        or selection_pin.get("sha256") != _sha256_output_file(
            selection_path, code="sol_phase_storage_failure",
            operation="sol_phase_selection_hash",
        )
        or prepare_pin != prepare_entry
        or result.get("status") != "captured"
        or result.get("capture_kind") != expected_kind
        or result.get("definition")
        != context["manifest"]["solution_contract"]["definition"]
        or result.get("solution")
        != context["manifest"]["solution_contract"]["solution"]
        or result.get("source_workload_index") != expected_source_index
        or result.get("staged_workload_index") != 0
        or result.get("workload_uuid")
        != context["manifest"]["selection"]["workload_uuid"]
        or result.get("workload_axes")
        != context["manifest"]["selection"]["workload_axes"]
        or result.get("environment_contract")
        != context["environment_contract"]
        or result.get("environment_contract_sha256")
        != context["environment_contract_sha256"]
        or result.get("flashinfer_corpus_contract")
        != context["flashinfer_corpus_contract"]
        or result.get("flashinfer_corpus_contract_sha256")
        != context["flashinfer_corpus_contract_sha256"]
        or result.get("stage_definition_sha256")
        != context["stage_definition_sha256"]
        or not isinstance(result.get("correctness"), dict)
    ):
        raise ChunkRefusal("sol_phase_result_invalid", expected_kind)
    expected_execution_order = [
        "correctness_outside_roi", "warmup_outside_roi", "cuda_synchronize",
        "cudaProfilerStart", "candidate_user_fn_once", "cuda_synchronize",
        "cudaProfilerStop_finally",
    ]
    if result.get("execution_order") != expected_execution_order:
        raise ChunkRefusal("sol_phase_execution_order_invalid", expected_kind)
    expected_roi = {
        "profiler_start_calls": 1,
        "candidate_user_fn_calls": 1,
        "profiler_stop_calls": 1,
        "timing_methodology": "cuda_events",
    }
    expected_outside = {
        "correctness_rounds": 1,
        "correctness_user_fn_calls": 1,
        "warmup_user_fn_calls": 1,
        "all_completed_before_profiler_start": True,
    }
    if any((result.get("roi") or {}).get(k) != v for k, v in expected_roi.items()):
        raise ChunkRefusal("sol_phase_roi_invalid", expected_kind)
    if any(
        (result.get("outside_roi") or {}).get(k) != v
        for k, v in expected_outside.items()
    ):
        raise ChunkRefusal("sol_phase_outside_roi_invalid", expected_kind)
    module = result.get("benchmark_kernel_so") or {}
    if module != staged_module_entry:
        raise ChunkRefusal("sol_phase_module_mismatch", expected_kind)
    staged_module = _sol_stage_entry_path(
        stage, module, "phase benchmark_kernel_so"
    )
    if (
        staged_module != context["benchmark_kernel_so"]
        or _sha256_output_file(
            staged_module, code="sol_phase_storage_failure",
            operation="sol_phase_module_hash",
        ) != context["benchmark_kernel_sha256"]
    ):
        raise ChunkRefusal("sol_phase_module_mismatch", expected_kind)
    return {
        "capture_kind": expected_kind,
        "selection_relative_path": (
            selection_path.relative_to(stage).as_posix()
        ),
        "selection_sha256": _sha256_output_file(
            selection_path, code="sol_phase_storage_failure",
            operation="sol_phase_selection_return_hash",
        ),
        "result_relative_path": result_path.relative_to(stage).as_posix(),
        "result_sha256": _sha256_output_file(
            result_path, code="sol_phase_storage_failure",
            operation="sol_phase_result_return_hash",
        ),
        "prepare_manifest_sha256": context["manifest_sha256"],
        "benchmark_kernel_sha256": context["benchmark_kernel_sha256"],
        "roi": result["roi"],
        "outside_roi": result["outside_roi"],
    }


class _PlannedLabelSidecars(Sequence[Path]):
    """Immutable lazy sidecar paths for large valid selector cardinalities."""

    __slots__ = ("_cell_root", "_ordinals")

    def __init__(self, cell_root: Path, ordinals: int | range) -> None:
        self._cell_root = cell_root
        self._ordinals = (
            range(ordinals) if type(ordinals) is int else ordinals
        )

    def __len__(self) -> int:
        return len(self._ordinals)

    def __getitem__(
        self, index: int | slice,
    ) -> Path | _PlannedLabelSidecars:
        if isinstance(index, slice):
            return _PlannedLabelSidecars(
                self._cell_root, self._ordinals[index],
            )
        ordinal = self._ordinals[index]
        return self._cell_root / f"label_0_{ordinal}.json"


_EXPLICIT_NCU_DRIVER_FIELDS = (
    ("ncu_target_processes", "--target-processes"),
    ("ncu_filter_mode", "--filter-mode"),
    ("ncu_replay_mode", "--replay-mode"),
    ("ncu_cache_control", "--cache-control"),
    ("ncu_clock_control", "--clock-control"),
    ("ncu_profile_from_start", "--profile-from-start"),
    ("ncu_launch_skip", "--launch-skip"),
    ("ncu_launch_count", "--launch-count"),
)
_LAUNCH_SET_FIXED_ARTIFACT_NAMES = (
    "run_0_launch_set.ncu-rep",
    "run_0_launch_set.details.csv",
    "run_0_launch_set.details.stderr",
    "run_0_launch_set.launch_metrics.json",
    "run_0_launch_set.details_export_receipt.json",
    "run_0_launch_set.launch_inventory.json",
    "run_0_launch_set.execution_identity.json",
    "label_run_manifest_0_launch_set.json",
    "label_launch_set_capture_manifest.json",
)
_LAUNCH_SET_AUTHORITY_PARTIAL_SUFFIX = ".ncu-details-partial"
_LAUNCH_SET_HANDOFF_STDERR = (
    b"label_driver: NCU_LAUNCH_SET_POSTPROCESSING_UNIMPLEMENTED: "
    b"explicit launch-set report retained without a sidecar\n"
)
_LAUNCH_SET_PERCENTAGE_METRICS = (
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
)
_PRODUCTION_LABEL_DRIVER = LABEL_DRIVER
_FIXTURE_LAUNCH_SET_EXECUTION_AUTHORITY = object()
_LAUNCH_SET_PLAN_FIELDS = frozenset({
    "command",
    "trace_inventory_cardinality",
    "driver_selector_count",
    "ncu_launch_protocol",
    "cell_root",
    "fixed_artifacts",
    "label_sidecars",
    "plan_sha256",
})


def _explicit_ncu_driver_args(protocol: dict[str, Any]) -> list[str]:
    try:
        ncu_capture_contract.validate_measurement_protocol(protocol)
    except ValueError as exc:
        raise ChunkRefusal(
            "label_launch_set_control_contract_invalid", str(exc)
        ) from exc
    expected_fields = {field for field, _option in _EXPLICIT_NCU_DRIVER_FIELDS}
    if set(protocol) != expected_fields:
        raise ChunkRefusal(
            "label_launch_set_control_contract_invalid",
            f"unexpected explicit NCU fields: {sorted(set(protocol) - expected_fields)!r}",
        )
    result: list[str] = []
    for field, option in _EXPLICIT_NCU_DRIVER_FIELDS:
        value = protocol[field]
        if field == "ncu_profile_from_start":
            value = "on" if value else "off"
        result += [option, str(value)]
    return result


def _prepare_label_capture_command(
    *, cell: dict, binary: Path, invocation: dict, cell_dir: Path,
    matrix: dict, pins: dict,
    label_launch_index_override: int | None,
    kernel_name_selector: str | None,
    artifact_tag: str,
    execution_source_commit: str | None,
    bind_supplemental_toolchain: bool,
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity | None,
    explicit_ncu_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and construct a label-driver command without side effects."""
    identity = cell["identity"]
    source_commit = (
        matrix["source_commit"]
        if execution_source_commit is None else execution_source_commit
    )
    if not _is_sha40(source_commit):
        raise ChunkRefusal(
            "label_source_commit_invalid", repr(source_commit)
        )
    labels_root = cell_dir / "labels"
    unit_metrics = tuple(
        matrix["metric_registry"]["collection"]["selected_unit_metrics"]
    )
    expected_metrics = tuple(registry_unit_metrics(matrix["arch"]))
    if set(unit_metrics) != set(expected_metrics) or len(unit_metrics) != len(
        expected_metrics
    ):
        raise ChunkRefusal(
            "matrix_metric_selection_drift",
            f"matrix selected {len(unit_metrics)} unit metrics; runtime selected "
            f"{len(expected_metrics)}",
        )
    publication_scope: dict[str, Any] | None = None
    expected_cuda_release: dict[str, int] | None = None
    expected_ncu_prefix: str | None = None
    if bind_supplemental_toolchain:
        try:
            expected_cuda_release = toolchain_contract.require_sol_cuda_release(
                pins["toolchain"]["nvcc_release"]
            )
            expected_ncu_prefix = (
                toolchain_contract.normalized_ncu_version_prefix(
                    pins["toolchain"]["ncu_version_prefix"]
                )
            )
            publication_scope = (
                toolchain_contract.cycles_only_publication_scope()
            )
            toolchain_contract.require_cycles_metric_in_collection(
                unit_metrics
            )
        except (
            KeyError,
            toolchain_contract.CaptureToolchainContractError,
        ) as exc:
            raise ChunkRefusal(
                "label_toolchain_contract_invalid", str(exc)
            ) from exc
    logical_workload = invocation.get("logical_workload") or {}
    label_shape_point = logical_workload.get("value")
    shape_delivery = invocation.get("shape_delivery")
    if not isinstance(label_shape_point, str) or not label_shape_point:
        raise ChunkRefusal(
            "shape_delivery_invalid",
            f"{identity}: {shape_delivery!r}, "
            f"logical_workload={logical_workload!r}",
        )
    if shape_delivery == "application_args":
        shape_binding = "application-args"
    elif (
        shape_delivery == "adapter_consumes_logical_shape"
        and identity.get("roster_source") == "sol_execbench"
        and invocation.get("adapter_kind") == "sol_execbench"
    ):
        shape_binding = "logical-workload"
    else:
        raise ChunkRefusal(
            "shape_delivery_invalid", f"{identity}: {shape_delivery!r}"
        )
    launch_contract = _require_capture_launch_contract(invocation)
    label_launch_index = (
        launch_contract["label_launch_index"]
        if label_launch_index_override is None else label_launch_index_override
    )
    if (
        type(label_launch_index) is not int or label_launch_index < 0
        or not isinstance(artifact_tag, str)
        or not artifact_tag
        or any(not (ch.isalnum() or ch in "_-") for ch in artifact_tag)
        or (
            kernel_name_selector is not None
            and (
                not isinstance(kernel_name_selector, str)
                or not kernel_name_selector
                or kernel_name_selector != kernel_name_selector.strip()
                or kernel_name_selector.startswith("regex:")
            )
        )
    ):
        raise ChunkRefusal(
            "label_launch_request_invalid",
            repr({
                "launch_index": label_launch_index,
                "kernel_name_selector": kernel_name_selector,
                "tag": artifact_tag,
            }),
        )
    label_driver_warmup = launch_contract["label_driver_warmup_count"]
    command = [
        str(LABEL_DRIVER), "--device", matrix["arch"],
        "--slug", identity["slug"],
        "--shape-point", label_shape_point,
        "--shape-binding", shape_binding,
        "--repeat", "0", "--launch", str(label_launch_index),
        "--binary", str(binary),
        "--warmup", str(label_driver_warmup),
    ]
    if explicit_ncu_protocol is None:
        try:
            expected_ncu_identity = ncu_executor.validate_expected_ncu_identity(
                expected_ncu_identity
            )
        except ncu_executor.NcuActionExecutorError as exc:
            raise ChunkRefusal(
                "capture_ncu_authority_invalid", str(exc)
            ) from exc
        command += [
            "--ncu", expected_ncu_identity.resolved_path,
            "--ncu-executable-sha256",
            expected_ncu_identity.executable_sha256,
            "--ncu-version-line", expected_ncu_identity.version_line,
            "--cache-control", "all", "--clock-control", "none",
        ]
    else:
        command += _explicit_ncu_driver_args(explicit_ncu_protocol)
    command += ["--out-dir", str(labels_root)]
    if bind_supplemental_toolchain:
        assert expected_cuda_release is not None
        assert expected_ncu_prefix is not None
        command += [
            "--expected-cuda-release",
            toolchain_contract.nvcc_release_text(expected_cuda_release),
            "--expected-ncu-version-prefix", expected_ncu_prefix,
            "--publication-metric-scope", "cycles_only_binding",
        ]
    if kernel_name_selector is not None:
        command += ["--kernel-name", kernel_name_selector]
    for metric in unit_metrics:
        command += ["--unit-metric", metric]
    effective_app_args = list(invocation["label_app_args"])
    if shape_binding == "logical-workload" and _shape_tokens(effective_app_args):
        raise ChunkRefusal(
            "sol_shape_token_forwarded", repr(_shape_tokens(effective_app_args))
        )
    if identity["roster_source"] == "portable_lcd":
        effective_app_args += [
            "--output_dir", str(cell_dir / "logs" / "label_workload"),
        ]
    command += ["--", *effective_app_args]
    if Path(command[0]).resolve() != LABEL_DRIVER.resolve():
        raise ChunkRefusal(
            "label_driver_identity_invalid", repr(command[0])
        )
    return {
        "command": command,
        "identity": identity,
        "source_commit": source_commit,
        "labels_root": labels_root,
        "unit_metrics": unit_metrics,
        "publication_scope": publication_scope,
        "expected_cuda_release": expected_cuda_release,
        "expected_ncu_prefix": expected_ncu_prefix,
        "label_shape_point": label_shape_point,
        "logical_workload": logical_workload,
        "launch_contract": launch_contract,
        "shape_binding": shape_binding,
        "label_launch_index": label_launch_index,
    }


def _launch_set_plan_sha256(plan: dict[str, Any]) -> str:
    model = {
        "command": plan.get("command"),
        "trace_inventory_cardinality": plan.get(
            "trace_inventory_cardinality"
        ),
        "driver_selector_count": plan.get("driver_selector_count"),
        "ncu_launch_protocol": plan.get("ncu_launch_protocol"),
        "cell_root": str(plan.get("cell_root")),
        "fixed_artifacts": [
            str(path) for path in plan.get("fixed_artifacts", ())
        ],
        "label_sidecar_count": len(plan.get("label_sidecars", ())),
    }
    return immutable.sha256_bytes(immutable.canonical_json_bytes(model))


def build_fixture_only_launch_set_plan(
    *, cell: dict, binary: Path, invocation: dict, cell_dir: Path,
    matrix: dict, pins: dict, trace_inventory_cardinality: int,
) -> dict[str, Any]:
    """Plan one supplemental launch-set command; execute and emit nothing."""
    if (
        type(trace_inventory_cardinality) is not int
        or not 1 <= trace_inventory_cardinality
        < ncu_capture_contract.MAX_NCU_SELECTOR
    ):
        raise ChunkRefusal(
            "label_launch_set_inventory_cardinality_invalid",
            repr(trace_inventory_cardinality),
        )
    identity = cell.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("roster_source") != "sol_execbench"
        or invocation.get("adapter_kind") != "sol_execbench"
        or invocation.get("shape_delivery")
        != "adapter_consumes_logical_shape"
    ):
        raise ChunkRefusal(
            "label_launch_request_invalid",
            "fixture-only launch-set plans require an SOL logical workload",
        )
    driver_selector_count = trace_inventory_cardinality + 1
    protocol = {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": driver_selector_count,
    }
    prepared = _prepare_label_capture_command(
        cell=cell, binary=binary, invocation=invocation, cell_dir=cell_dir,
        matrix=matrix, pins=pins, label_launch_index_override=0,
        kernel_name_selector=None, artifact_tag="label_launch_set",
        execution_source_commit=None, bind_supplemental_toolchain=True,
        expected_ncu_identity=None,
        explicit_ncu_protocol=protocol,
    )
    cell_root = (
        prepared["labels_root"] / matrix["arch"] / identity["slug"]
        / prepared["label_shape_point"]
    )
    plan = {
        "command": prepared["command"],
        "trace_inventory_cardinality": trace_inventory_cardinality,
        "driver_selector_count": driver_selector_count,
        "ncu_launch_protocol": protocol,
        "cell_root": cell_root,
        "fixed_artifacts": tuple(
            cell_root / name for name in _LAUNCH_SET_FIXED_ARTIFACT_NAMES
        ),
        "label_sidecars": _PlannedLabelSidecars(
            cell_root, trace_inventory_cardinality,
        ),
    }
    plan["plan_sha256"] = _launch_set_plan_sha256(plan)
    return plan


_LABEL_COLLECTION_ENVIRONMENT_NAMES = (
    "CC",
    "CUDAHOSTCXX",
    "CUDA_HOME",
    "CUTLASS_DIR",
    "CXX",
    "FLASHINFER_TRACE_DIR",
    "NVCC_CCBIN",
    "PYTORCH_ALLOC_CONF",
    "SOL_EXECBENCH_CAPTURE_REQUEST",
)
_LABEL_COLLECTION_PATH = "/usr/local/cuda-13.2/bin:/usr/bin:/bin"


def _label_collection_environment(
    runtime_env: dict[str, str] | None, *, cwd: Path,
) -> ncu_executor.NcuCollectionEnvironment:
    if runtime_env is not None and type(runtime_env) is not dict:
        raise ChunkRefusal(
            "label_collection_environment_invalid",
            "runtime environment is not an exact dictionary",
        )
    source = runtime_env or {}
    values = {
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": _LABEL_COLLECTION_PATH,
        "PWD": str(cwd),
        "PYTHONNOUSERSITE": "1",
        "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
    }
    for name in _LABEL_COLLECTION_ENVIRONMENT_NAMES:
        if name in source:
            values[name] = source[name]
    try:
        return ncu_executor.NcuCollectionEnvironment(values)
    except ncu_executor.NcuActionExecutorError as exc:
        raise ChunkRefusal("label_collection_environment_invalid", str(exc)) from exc


def _launch_set_option(command: Sequence[str], option: str) -> str:
    if command.count(option) != 1:
        raise ChunkRefusal(
            "label_launch_set_plan_invalid",
            f"launch-set command requires exactly one {option}",
        )
    index = command.index(option)
    if index + 1 >= len(command) or command[index + 1] == "--":
        raise ChunkRefusal(
            "label_launch_set_plan_invalid",
            f"launch-set command has no value for {option}",
        )
    return command[index + 1]


def _validate_fixture_launch_set_plan(
    plan: dict[str, Any], *, artifact_root: Path,
    expected_plan_sha256: str,
) -> tuple[list[str], dict[str, Any], Path, dict[str, Path]]:
    if type(plan) is not dict or set(plan) != _LAUNCH_SET_PLAN_FIELDS:
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "launch-set plan fields differ",
        )
    try:
        observed_plan_sha256 = _launch_set_plan_sha256(plan)
    except (TypeError, ValueError, RecursionError, immutable.ImmutableArtifactError) as exc:
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", f"launch-set plan is malformed: {exc}",
        ) from exc
    if (
        not _is_sha256(expected_plan_sha256)
        or plan.get("plan_sha256") != expected_plan_sha256
        or observed_plan_sha256 != expected_plan_sha256
    ):
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "launch-set plan seal differs",
        )
    command = plan.get("command")
    trace_count = plan.get("trace_inventory_cardinality")
    selector_count = plan.get("driver_selector_count")
    protocol = plan.get("ncu_launch_protocol")
    cell_root = plan.get("cell_root")
    fixed_artifacts = plan.get("fixed_artifacts")
    sidecars = plan.get("label_sidecars")
    if (
        type(command) is not list
        or not command
        or any(type(argument) is not str or "\x00" in argument for argument in command)
        or type(trace_count) is not int
        or trace_count < 1
        or type(selector_count) is not int
        or selector_count != trace_count + 1
        or type(protocol) is not dict
        or not isinstance(cell_root, Path)
        or type(fixed_artifacts) is not tuple
        or type(sidecars) is not _PlannedLabelSidecars
        or not isinstance(artifact_root, Path)
    ):
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "launch-set plan types differ",
        )
    expected_protocol = {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": selector_count,
    }
    if protocol != expected_protocol:
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "launch-set protocol differs",
        )
    _explicit_ncu_driver_args(protocol)
    if (
        command[0] != str(LABEL_DRIVER)
        or command.count("--") != 1
        or "--ncu-exec-path" in command
        or any(argument.startswith("--ncu-exec-path=") for argument in command)
        or _launch_set_option(command, "--repeat") != "0"
        or _launch_set_option(command, "--launch") != "0"
        or _launch_set_option(command, "--out-dir") != str(artifact_root)
    ):
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "launch-set command differs",
        )
    for field, option in _EXPLICIT_NCU_DRIVER_FIELDS:
        value: Any = protocol[field]
        if field == "ncu_profile_from_start":
            value = "on" if value else "off"
        if _launch_set_option(command, option) != str(value):
            raise ChunkRefusal(
                "label_launch_set_plan_invalid",
                f"launch-set command differs at {option}",
            )
    device = _launch_set_option(command, "--device")
    slug = _launch_set_option(command, "--slug")
    shape = _launch_set_option(command, "--shape-point")
    binary = Path(_launch_set_option(command, "--binary"))
    for label, component in (("device", device), ("slug", slug), ("shape", shape)):
        if (
            not component
            or component in {".", ".."}
            or "/" in component
            or "\\" in component
            or any(ord(character) < 0x20 for character in component)
        ):
            raise ChunkRefusal(
                "label_launch_set_plan_invalid",
                f"launch-set {label} path component is unsafe",
            )
    if (
        not artifact_root.is_absolute()
        or str(artifact_root).startswith("//")
        or os.path.normpath(str(artifact_root)) != str(artifact_root)
        or not cell_root.is_absolute()
        or str(cell_root).startswith("//")
        or os.path.normpath(str(cell_root)) != str(cell_root)
        or not binary.is_absolute()
        or str(binary).startswith("//")
        or os.path.normpath(str(binary)) != str(binary)
        or cell_root != artifact_root / device / slug / shape
    ):
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "launch-set paths are not canonical",
        )
    try:
        cell_relative = cell_root.relative_to(artifact_root)
    except ValueError as exc:
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "cell root escaped artifact root",
        ) from exc
    if cell_relative == Path("."):
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "cell root equals artifact root",
        )
    expected_artifacts = tuple(
        cell_root / name for name in _LAUNCH_SET_FIXED_ARTIFACT_NAMES
    )
    if fixed_artifacts != expected_artifacts:
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "fixed artifact paths differ",
        )
    if (
        sidecars._cell_root != cell_root
        or sidecars._ordinals != range(trace_count)
        or len(sidecars) != trace_count
        or sidecars[0] != cell_root / "label_0_0.json"
        or sidecars[-1] != cell_root / f"label_0_{trace_count - 1}.json"
    ):
        raise ChunkRefusal(
            "label_launch_set_plan_invalid", "planned sidecar paths differ",
        )
    return command, protocol, cell_root, {
        path.name: path for path in fixed_artifacts
    }


def _bounded_launch_set_output(payload: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tail": payload[-4096:].decode("utf-8", errors="backslashreplace"),
    }


def _validate_fixture_launch_set_manifest(
    document: dict[str, Any], *, command: Sequence[str], source_commit: str,
    expected_target_executable_sha256: str,
    expected_kernel_signature_sha256: str,
    expected_kernel_signature_source: str,
) -> None:
    required = {
        "schema_version",
        "source_commit",
        "binary_sha256",
        "kernel_signature_sha256",
        "kernel_signature_source",
        "arch",
        "slug",
        "shape_point",
        "repeat_index",
        "launch_index",
    }
    optional = set(document) - required
    if optional not in (set(), {"variant_id", "register_cap"}):
        raise ChunkRefusal(
            "label_launch_set_manifest_invalid",
            "launch-set run manifest fields differ",
        )
    expected = {
        "schema_version": 1,
        "source_commit": source_commit,
        "binary_sha256": expected_target_executable_sha256,
        "kernel_signature_sha256": expected_kernel_signature_sha256,
        "kernel_signature_source": expected_kernel_signature_source,
        "arch": _launch_set_option(command, "--device"),
        "slug": _launch_set_option(command, "--slug"),
        "shape_point": _launch_set_option(command, "--shape-point"),
        "repeat_index": 0,
        "launch_index": 0,
    }
    if (
        not required.issubset(document)
        or not _is_sha256(expected_target_executable_sha256)
        or any(document.get(name) != value for name, value in expected.items())
        or not _is_sha256(document.get("kernel_signature_sha256"))
        or type(document.get("kernel_signature_source")) is not str
        or not document["kernel_signature_source"]
        or (
            optional == {"variant_id", "register_cap"}
            and (
                type(document.get("variant_id")) is not str
                or not document["variant_id"]
                or type(document.get("register_cap")) is not int
                or document["register_cap"] < 0
            )
        )
    ):
        raise ChunkRefusal(
            "label_launch_set_manifest_invalid",
            "launch-set run manifest authority differs",
        )


def _require_fixture_launch_set_namespace(cell_root: Path) -> None:
    expected = {
        "run_0_launch_set.ncu-rep",
        "label_run_manifest_0_launch_set.json",
    }
    try:
        metadata = os.lstat(cell_root)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError(None, "launch-set cell root is not a directory")
        observed_names: list[str] = []
        with os.scandir(cell_root) as iterator:
            for entry in iterator:
                observed_names.append(entry.name)
                if len(observed_names) > len(expected):
                    break
        observed = set(observed_names)
    except OSError as exc:
        raise ChunkRefusal(
            "label_launch_set_handoff_invalid",
            f"cannot inspect fixture handoff namespace: {exc}",
        ) from exc
    if observed != expected:
        raise ChunkRefusal(
            "label_launch_set_handoff_invalid",
            f"fixture handoff namespace differs: {sorted(observed)!r}",
        )


def _require_fixture_launch_set_authority_namespace(
    cell_root: Path, *, expected_names: frozenset[str],
) -> None:
    try:
        metadata = os.lstat(cell_root)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError(None, "launch-set cell root is not a directory")
        observed_names: list[str] = []
        with os.scandir(cell_root) as iterator:
            for entry in iterator:
                observed_names.append(entry.name)
                if len(observed_names) > len(expected_names):
                    break
        observed = frozenset(observed_names)
    except OSError as exc:
        raise ncu_launch_documents.LaunchDocumentError(
            f"cannot inspect authority publication namespace: {exc}"
        ) from exc
    if observed != expected_names:
        raise ncu_launch_documents.LaunchDocumentError(
            "authority publication namespace differs: "
            f"{sorted(observed)!r}"
        )


def _persist_fixture_launch_set_authorities(
    *, cell_root: Path, artifacts: dict[str, Path], command: Sequence[str],
    protocol: dict[str, Any], expected_launch_count: int,
    expected_target_executable_sha256: str,
    expected_runtime_module_sha256s: tuple[str, ...],
    execution_identity_document: dict[str, Any],
    products: ncu_launch_documents.DetailsExportProducts,
    details_export_receipt: Path,
    expected_run_manifest_document: dict[str, Any],
    expected_run_manifest_sha256: str,
    expected_run_manifest_size_bytes: int,
    sidecar_documents: tuple[dict[str, Any], ...],
    sidecar_paths: tuple[Path, ...],
    sidecar_schema_bytes: bytes,
    sidecar_schema: dict[str, Any],
    sidecar_provenance: ncu_launch_sidecars.LaunchSidecarProvenance,
    registry: Any,
    percentage_metric_names: tuple[str, ...],
    unit_metric_names: tuple[str, ...],
) -> tuple[Path, Path, tuple[Path, ...]]:
    """Publish fixture authority documents after retained Details products."""
    execution_identity_path = artifacts[
        "run_0_launch_set.execution_identity.json"
    ]
    capture_manifest_path = artifacts[
        "label_launch_set_capture_manifest.json"
    ]
    inventory_path = artifacts["run_0_launch_set.launch_inventory.json"]
    report_path = artifacts["run_0_launch_set.ncu-rep"]
    run_manifest_path = artifacts[
        "label_run_manifest_0_launch_set.json"
    ]
    generated_product_paths = {
        "details_csv": artifacts["run_0_launch_set.details.csv"],
        "details_stderr": artifacts["run_0_launch_set.details.stderr"],
        "launch_inventory": inventory_path,
        "launch_metrics": artifacts["run_0_launch_set.launch_metrics.json"],
    }
    generated_product_bytes = {
        "details_csv": products.csv_bytes,
        "details_stderr": products.stderr_bytes,
        "launch_inventory": products.inventory_bytes,
        "launch_metrics": products.metrics_bytes,
    }
    expected_receipt_path = artifacts[
        "run_0_launch_set.details_export_receipt.json"
    ]
    expected_sidecar_paths = tuple(
        cell_root / f"label_0_{ordinal}.json"
        for ordinal in range(expected_launch_count)
    )
    try:
        parsed_sidecar_schema = (
            sidecar_schema_authority.parse_authoritative_schema_bytes(
                sidecar_schema_bytes
            )
        )
    except ValueError as exc:
        raise ncu_launch_documents.LaunchDocumentError(str(exc)) from exc
    complete_namespace = frozenset({
        *_LAUNCH_SET_FIXED_ARTIFACT_NAMES,
        *(path.name for path in expected_sidecar_paths),
    })
    prepublication_namespace = frozenset(
        _LAUNCH_SET_FIXED_ARTIFACT_NAMES
    ) - {
        execution_identity_path.name,
        capture_manifest_path.name,
    }
    if (
        details_export_receipt != expected_receipt_path
        or not _is_sha256(expected_run_manifest_sha256)
        or type(expected_run_manifest_size_bytes) is not int
        or expected_run_manifest_size_bytes <= 0
        or type(expected_run_manifest_document) is not dict
        or type(sidecar_documents) is not tuple
        or len(sidecar_documents) != expected_launch_count
        or type(sidecar_paths) is not tuple
        or sidecar_paths != expected_sidecar_paths
        or type(sidecar_schema_bytes) is not bytes
        or type(sidecar_schema) is not dict
        or parsed_sidecar_schema != sidecar_schema
        or any(
            path.parent != cell_root
            for path in (
                execution_identity_path,
                capture_manifest_path,
                report_path,
                run_manifest_path,
                *generated_product_paths.values(),
                expected_receipt_path,
            )
        )
    ):
        raise ncu_launch_documents.LaunchDocumentError(
            "fixture launch-set authority paths differ from the plan"
        )
    ncu_launch_documents.validate_launch_execution_identity_document(
        execution_identity_document,
        expected_launch_count=expected_launch_count,
        expected_target_executable_sha256=(
            expected_target_executable_sha256
        ),
        expected_runtime_module_sha256s=(
            expected_runtime_module_sha256s
        ),
    )
    ncu_launch_documents.validate_schema(
        "execution_identity", execution_identity_document,
    )
    execution_identity_bytes = immutable.pretty_json_bytes(
        execution_identity_document
    )
    launch_observation_sha256 = products.inventory_document.get(
        "observation_sha256"
    )
    execution_identity_sha256 = execution_identity_document.get(
        "identity_sha256"
    )
    capture_manifest_document = (
        ncu_launch_documents.build_launch_set_capture_manifest_document(
            expected_command=command,
            expected_launch_count=expected_launch_count,
            protocol=protocol,
            launch_observation_sha256=launch_observation_sha256,
            execution_identity_sha256=execution_identity_sha256,
        )
    )
    ncu_launch_documents.validate_schema(
        "capture_manifest", capture_manifest_document,
    )
    capture_manifest_bytes = (
        ncu_launch_documents.launch_set_capture_manifest_bytes(
            capture_manifest_document,
            expected_command=command,
            expected_launch_count=expected_launch_count,
            protocol=protocol,
            expected_launch_observation_sha256=(
                launch_observation_sha256
            ),
            expected_execution_identity_sha256=(
                execution_identity_sha256
            ),
        )
    )
    sidecar_documents = ncu_launch_sidecars.validate_launch_sidecar_documents(
        sidecar_documents,
        inventory=products.inventory_document,
        launch_metrics=products.metrics_document,
        csv_text=products.csv_bytes.decode("utf-8", errors="strict"),
        report_sha256=products.inventory_document["report_sha256"],
        expected_launch_count=expected_launch_count,
        protocol=protocol,
        registry=registry,
        percentage_metric_names=percentage_metric_names,
        unit_metric_names=unit_metric_names,
        sidecar_schema=sidecar_schema,
        provenance=sidecar_provenance,
    )
    sidecar_bytes = tuple(
        immutable.pretty_json_bytes(document)
        for document in sidecar_documents
    )

    with contextlib.ExitStack() as retained_stack:
        pinned_parent = retained_stack.enter_context(
            immutable.PinnedDirectoryReader(cell_root)
        )
        retained_sidecar_schema = retained_stack.enter_context(
            immutable.open_retained_regular(
                sidecar_schema_authority.SCHEMA_PATH
            )
        )
        if (
            retained_sidecar_schema.sha256
            != immutable.sha256_bytes(sidecar_schema_bytes)
            or retained_sidecar_schema.size_bytes != len(sidecar_schema_bytes)
            or _retained_json_model(
                retained_sidecar_schema, role="launch_set_sidecar_schema",
            ) != sidecar_schema
        ):
            raise ncu_launch_documents.LaunchDocumentError(
                "retained sidecar schema authority differs"
            )
        retained_report = retained_stack.enter_context(
            immutable.open_retained_regular(report_path)
        )
        retained_run_manifest = retained_stack.enter_context(
            immutable.open_retained_regular(run_manifest_path)
        )
        retained_products = {
            role: retained_stack.enter_context(
                immutable.open_retained_regular(path)
            )
            for role, path in generated_product_paths.items()
        }
        retained_inventory = retained_products["launch_inventory"]
        retained_receipt = retained_stack.enter_context(
            immutable.open_retained_regular(expected_receipt_path)
        )
        if (
            retained_run_manifest.sha256 != expected_run_manifest_sha256
            or retained_run_manifest.size_bytes
            != expected_run_manifest_size_bytes
            or any(
                retained_products[role].sha256
                != immutable.sha256_bytes(payload)
                or retained_products[role].size_bytes != len(payload)
                or os.fstat(retained_products[role].descriptor).st_mode & 0o222
                for role, payload in generated_product_bytes.items()
            )
            or retained_receipt.sha256
            != immutable.sha256_bytes(products.receipt_bytes)
            or retained_receipt.size_bytes != len(products.receipt_bytes)
            or os.fstat(retained_receipt.descriptor).st_mode & 0o222
        ):
            raise ncu_launch_documents.LaunchDocumentError(
                "retained launch-set authority files differ from producer bytes"
            )
        retained_run_manifest_document = _retained_json_model(
            retained_run_manifest, role="launch_set_run_manifest",
        )
        retained_inventory_document = _retained_json_model(
            retained_inventory, role="launch_set_inventory",
        )
        retained_receipt_document = _retained_json_model(
            retained_receipt, role="details_export_receipt",
        )
        if (
            retained_run_manifest_document
            != expected_run_manifest_document
            or retained_inventory_document != products.inventory_document
            or retained_receipt_document != products.receipt_document
        ):
            raise ncu_launch_documents.LaunchDocumentError(
                "retained launch-set authority models differ from producer models"
            )
        ncu_launch_documents.validate_launch_inventory_document(
            retained_inventory_document,
            expected_launch_count=expected_launch_count,
            report_sha256=retained_report.sha256,
            protocol=protocol,
        )
        _require_fixture_launch_set_authority_namespace(
            cell_root, expected_names=prepublication_namespace,
        )

        published_execution_identity = immutable.publish_regular_file(
            cell_root,
            execution_identity_path.name,
            execution_identity_bytes,
            partial_suffix=_LAUNCH_SET_AUTHORITY_PARTIAL_SUFFIX,
            pinned_parent=pinned_parent,
        )
        retained_execution_identity = retained_stack.enter_context(
            immutable.open_retained_regular(published_execution_identity)
        )
        if (
            retained_execution_identity.sha256
            != immutable.sha256_bytes(execution_identity_bytes)
            or retained_execution_identity.size_bytes
            != len(execution_identity_bytes)
            or os.fstat(retained_execution_identity.descriptor).st_mode & 0o222
        ):
            raise ncu_launch_documents.LaunchDocumentError(
                "published execution identity bytes differ"
            )
        retained_execution_document = _retained_json_model(
            retained_execution_identity, role="launch_set_execution_identity",
        )
        ncu_launch_documents.validate_launch_execution_identity_document(
            retained_execution_document,
            expected_launch_count=expected_launch_count,
            expected_target_executable_sha256=(
                expected_target_executable_sha256
            ),
            expected_runtime_module_sha256s=(
                expected_runtime_module_sha256s
            ),
        )
        ncu_launch_documents.validate_schema(
            "execution_identity", retained_execution_document,
        )
        if retained_execution_document != execution_identity_document:
            raise ncu_launch_documents.LaunchDocumentError(
                "published execution identity model differs"
            )

        retained_sidecars: list[immutable.RetainedRegular] = []
        retained_sidecar_documents: list[dict[str, Any]] = []
        published_sidecars: list[Path] = []
        for ordinal, (path, payload) in enumerate(
            zip(sidecar_paths, sidecar_bytes, strict=True)
        ):
            published_sidecar = immutable.publish_regular_file(
                cell_root,
                path.name,
                payload,
                partial_suffix=_LAUNCH_SET_AUTHORITY_PARTIAL_SUFFIX,
                pinned_parent=pinned_parent,
            )
            retained_sidecar = retained_stack.enter_context(
                immutable.open_retained_regular(published_sidecar)
            )
            if (
                retained_sidecar.sha256 != immutable.sha256_bytes(payload)
                or retained_sidecar.size_bytes != len(payload)
                or os.fstat(retained_sidecar.descriptor).st_mode & 0o222
            ):
                raise ncu_launch_documents.LaunchDocumentError(
                    f"published launch sidecar {ordinal} bytes differ"
                )
            published_sidecars.append(published_sidecar)
            retained_sidecars.append(retained_sidecar)
            retained_sidecar_documents.append(_retained_json_model(
                retained_sidecar, role=f"launch_set_sidecar_{ordinal}",
            ))
        ncu_launch_sidecars.validate_launch_sidecar_documents(
            retained_sidecar_documents,
            inventory=retained_inventory_document,
            launch_metrics=products.metrics_document,
            csv_text=products.csv_bytes.decode("utf-8", errors="strict"),
            report_sha256=retained_report.sha256,
            expected_launch_count=expected_launch_count,
            protocol=protocol,
            registry=registry,
            percentage_metric_names=percentage_metric_names,
            unit_metric_names=unit_metric_names,
            sidecar_schema=sidecar_schema,
            provenance=sidecar_provenance,
        )
        retained_sidecar_schema.verify_unchanged()

        published_capture_manifest = immutable.publish_regular_file(
            cell_root,
            capture_manifest_path.name,
            capture_manifest_bytes,
            partial_suffix=_LAUNCH_SET_AUTHORITY_PARTIAL_SUFFIX,
            pinned_parent=pinned_parent,
        )
        retained_capture_manifest = retained_stack.enter_context(
            immutable.open_retained_regular(published_capture_manifest)
        )
        if (
            retained_capture_manifest.sha256
            != immutable.sha256_bytes(capture_manifest_bytes)
            or retained_capture_manifest.size_bytes
            != len(capture_manifest_bytes)
            or os.fstat(retained_capture_manifest.descriptor).st_mode & 0o222
        ):
            raise ncu_launch_documents.LaunchDocumentError(
                "published capture manifest bytes differ"
            )
        retained_capture_document = _retained_json_model(
            retained_capture_manifest, role="launch_set_capture_manifest",
        )
        ncu_launch_documents.validate_launch_set_capture_manifest_document(
            retained_capture_document,
            expected_command=command,
            expected_launch_count=expected_launch_count,
            protocol=protocol,
            expected_launch_observation_sha256=(
                retained_inventory_document["observation_sha256"]
            ),
            expected_execution_identity_sha256=(
                retained_execution_document["identity_sha256"]
            ),
        )
        ncu_launch_documents.validate_schema(
            "capture_manifest", retained_capture_document,
        )
        if retained_capture_document != capture_manifest_document:
            raise ncu_launch_documents.LaunchDocumentError(
                "published capture manifest model differs"
            )
        for retained in (
            retained_sidecar_schema,
            retained_report,
            retained_run_manifest,
            *retained_products.values(),
            retained_receipt,
            retained_execution_identity,
            *retained_sidecars,
            retained_capture_manifest,
        ):
            retained.verify_unchanged()
        _require_fixture_launch_set_authority_namespace(
            cell_root, expected_names=complete_namespace,
        )
        pinned_parent.verify_path()
    return (
        published_execution_identity,
        published_capture_manifest,
        tuple(published_sidecars),
    )


def execute_fixture_only_launch_set_plan(
    *,
    plan: dict[str, Any],
    expected_plan_sha256: str,
    fixture_execution_authority: object,
    artifact_root: Path,
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    expected_target_executable_sha256: str,
    expected_runtime_module_sha256s: tuple[str, ...],
    expected_gpu_uuid_sha256: str,
    expected_cuda_version: str,
    expected_driver_version: str,
    expected_preflight_sha256: str,
    expected_source_commit: str,
    expected_capture_identity_sha256: str,
    expected_kernel_signature_sha256: str,
    expected_kernel_signature_source: str,
    registry: Any,
    percentage_metric_names: tuple[str, ...],
    unit_metric_names: tuple[str, ...],
    timeout_s: float,
    runtime_cwd: Path,
    runtime_env: dict[str, str] | None = None,
) -> dict[str, Path | tuple[Path, ...]]:
    """Exercise the launch-set handoff without opening a production live gate."""
    try:
        expected_ncu_identity = ncu_executor.validate_expected_ncu_identity(
            expected_ncu_identity
        )
    except ncu_executor.NcuActionExecutorError as exc:
        raise ChunkRefusal("capture_ncu_authority_invalid", str(exc)) from exc
    if (
        not _is_sha256(expected_preflight_sha256)
        or not _is_sha256(expected_gpu_uuid_sha256)
        or type(expected_cuda_version) is not str
        or not expected_cuda_version
        or "\x00" in expected_cuda_version
        or type(expected_driver_version) is not str
        or not expected_driver_version
        or "\x00" in expected_driver_version
        or not _is_sha40(expected_source_commit)
        or not _is_sha256(expected_capture_identity_sha256)
        or not _is_sha256(expected_kernel_signature_sha256)
        or type(expected_kernel_signature_source) is not str
        or not expected_kernel_signature_source
        or type(percentage_metric_names) is not tuple
        or percentage_metric_names != _LAUNCH_SET_PERCENTAGE_METRICS
        or type(unit_metric_names) is not tuple
        or fixture_execution_authority is not (
            _FIXTURE_LAUNCH_SET_EXECUTION_AUTHORITY
        )
    ):
        raise ChunkRefusal(
            "label_launch_set_authority_invalid",
            "launch-set external authority is malformed",
        )
    command, protocol, cell_root, artifacts = (
        _validate_fixture_launch_set_plan(
            plan,
            artifact_root=artifact_root,
            expected_plan_sha256=expected_plan_sha256,
        )
    )
    warmup_text = _launch_set_option(command, "--warmup")
    try:
        warmup_count = int(warmup_text, 10)
    except ValueError as exc:
        raise ChunkRefusal(
            "label_launch_set_authority_invalid",
            "launch-set warmup authority is malformed",
        ) from exc
    if warmup_count < 0 or warmup_text != str(warmup_count):
        raise ChunkRefusal(
            "label_launch_set_authority_invalid",
            "launch-set warmup authority is not canonical",
        )
    sidecar_paths = tuple(
        plan["label_sidecars"][ordinal]
        for ordinal in range(plan["trace_inventory_cardinality"])
    )
    sidecar_provenance = ncu_launch_sidecars.LaunchSidecarProvenance(
        device=_launch_set_option(command, "--device"),
        slug=_launch_set_option(command, "--slug"),
        shape_point=_launch_set_option(command, "--shape-point"),
        binary_sha256=expected_target_executable_sha256,
        profiler_version=expected_ncu_identity.version_line,
        cuda_version=expected_cuda_version,
        driver_version=expected_driver_version,
        gpu_uuid_sha256=expected_gpu_uuid_sha256,
        warmup_count=warmup_count,
    )
    try:
        execution_identity_document = (
            ncu_launch_documents.build_launch_execution_identity_document(
                target_executable_sha256=(
                    expected_target_executable_sha256
                ),
                runtime_module_sha256s=(
                    expected_runtime_module_sha256s
                ),
                expected_launch_count=plan[
                    "trace_inventory_cardinality"
                ],
            )
        )
        immutable.pretty_json_bytes(execution_identity_document)
        ncu_launch_documents.validate_schema(
            "execution_identity", execution_identity_document,
        )
        ncu_launch_documents.load_schema("capture_manifest")
        sidecar_schema_bytes, sidecar_schema = (
            sidecar_schema_authority.load_authoritative_schema()
        )
    except (
        immutable.ImmutableArtifactError,
        ncu_launch_documents.LaunchDocumentError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ChunkRefusal(
            "label_launch_set_authority_invalid",
            f"target execution identity authority is invalid: {exc}",
        ) from exc
    planned_unit_metrics = tuple(
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "--unit-metric"
    )
    planned_cuda_version = _launch_set_option(
        command, "--expected-cuda-release",
    )
    planned_ncu_version_prefix = _launch_set_option(
        command, "--expected-ncu-version-prefix",
    )
    try:
        ncu_version_matches = toolchain_contract.ncu_version_matches(
            expected_ncu_identity.version_line,
            planned_ncu_version_prefix,
        )
    except toolchain_contract.CaptureToolchainContractError as exc:
        raise ChunkRefusal(
            "label_launch_set_authority_invalid", str(exc),
        ) from exc
    if (
        planned_cuda_version != expected_cuda_version
        or not ncu_version_matches
    ):
        raise ChunkRefusal(
            "label_launch_set_authority_invalid",
            "sidecar toolchain provenance differs from the sealed command",
        )
    if (
        Path(command[0]).resolve() == _PRODUCTION_LABEL_DRIVER.resolve()
        or planned_unit_metrics != unit_metric_names
        or os.path.lexists(cell_root)
    ):
        raise ChunkRefusal(
            "label_launch_set_fixture_boundary_invalid",
            "fixture execution authority or fresh output boundary differs",
        )
    supervised_command: list[object] = list(command)
    separator = supervised_command.index("--")
    supervised_command[separator:separator] = [
        "--ncu-exec-path", ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    environment = _label_collection_environment(runtime_env, cwd=runtime_cwd)
    target_executable = Path(_launch_set_option(command, "--binary"))
    try:
        retained_target = immutable.open_retained_executable(target_executable)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise ChunkRefusal(
            "label_launch_set_target_authority_invalid",
            f"cannot retain profiled target executable: {exc}",
        ) from exc
    try:
        with retained_target:
            if retained_target.sha256 != expected_target_executable_sha256:
                raise ChunkRefusal(
                    "label_launch_set_target_authority_invalid",
                    "profiled target executable SHA-256 differs",
                )
            with immutable.PinnedDirectoryReader(artifact_root) as pinned_root:
                if os.listdir(pinned_root.descriptor):
                    raise ChunkRefusal(
                        "label_launch_set_fixture_boundary_invalid",
                        "fixture artifact root is not empty",
                    )
                result = ncu_executor.execute_ncu_supervised_child(
                    expected_ncu_identity,
                    supervised_command,
                    timeout_seconds=timeout_s,
                    environment=environment,
                    working_directory=runtime_cwd,
                )
                pinned_root.verify_path()
            retained_target.verify_unchanged()
    except ncu_executor.NcuActionExecutorError as exc:
        raise ChunkRefusal(
            f"label_launch_set_collection_{exc.code.lower()}", str(exc),
        ) from exc
    except (OSError, immutable.ImmutableArtifactError) as exc:
        raise ChunkRefusal(
            "label_launch_set_fixture_boundary_invalid",
            f"fixture artifact root is not pinned: {exc}",
        ) from exc
    if (
        result.returncode != 3
        or result.stdout != b""
        or result.stderr != _LAUNCH_SET_HANDOFF_STDERR
    ):
        raise ChunkRefusal(
            "label_launch_set_handoff_invalid",
            repr({
                "returncode": result.returncode,
                "stdout": _bounded_launch_set_output(result.stdout),
                "stderr": _bounded_launch_set_output(result.stderr),
            }),
        )
    _require_fixture_launch_set_namespace(cell_root)

    report = artifacts["run_0_launch_set.ncu-rep"]
    run_manifest = artifacts["label_run_manifest_0_launch_set.json"]
    try:
        with contextlib.ExitStack() as retained_stack:
            retained_report = retained_stack.enter_context(
                immutable.open_retained_regular(report)
            )
            retained_manifest = retained_stack.enter_context(
                immutable.open_retained_regular(run_manifest)
            )
            if retained_report.size_bytes <= 0 or retained_manifest.size_bytes <= 0:
                raise immutable.ImmutableArtifactError(
                    "launch-set handoff artifact is empty"
                )
            manifest_document = _retained_json_model(
                retained_manifest, role="launch_set_run_manifest",
            )
            _validate_fixture_launch_set_manifest(
                manifest_document,
                command=command,
                source_commit=expected_source_commit,
                expected_target_executable_sha256=(
                    expected_target_executable_sha256
                ),
                expected_kernel_signature_sha256=(
                    expected_kernel_signature_sha256
                ),
                expected_kernel_signature_source=(
                    expected_kernel_signature_source
                ),
            )
            run_manifest_sha256 = retained_manifest.sha256
            run_manifest_size_bytes = retained_manifest.size_bytes
            imported, import_error = parse_ncu_metrics._import_report_csv(
                expected_ncu_identity, report,
            )
            if import_error is not None or imported is None:
                raise ChunkRefusal(
                    "label_launch_set_import_failed",
                    repr(import_error),
                )
            if (
                imported.report_sha256 != retained_report.sha256
                or imported.report_size_bytes != retained_report.size_bytes
            ):
                raise ChunkRefusal(
                    "label_launch_set_import_failed",
                    "imported report identity differs from retained handoff",
                )
            relative = {
                name: path.relative_to(artifact_root).as_posix()
                for name, path in artifacts.items()
            }
            products = ncu_launch_documents.build_details_export_products(
                imported=imported,
                preflight_sha256=expected_preflight_sha256,
                source_commit=expected_source_commit,
                capture_identity_sha256=expected_capture_identity_sha256,
                expected_ncu_identity=expected_ncu_identity,
                report_relative_path=relative["run_0_launch_set.ncu-rep"],
                csv_relative_path=relative["run_0_launch_set.details.csv"],
                stderr_relative_path=relative[
                    "run_0_launch_set.details.stderr"
                ],
                inventory_relative_path=relative[
                    "run_0_launch_set.launch_inventory.json"
                ],
                metrics_relative_path=relative[
                    "run_0_launch_set.launch_metrics.json"
                ],
                expected_launch_count=plan["trace_inventory_cardinality"],
                protocol=protocol,
                registry=registry,
                percentage_metric_names=percentage_metric_names,
                unit_metric_names=unit_metric_names,
            )
            sidecar_documents = (
                ncu_launch_sidecars.build_launch_sidecar_documents(
                    inventory=products.inventory_document,
                    launch_metrics=products.metrics_document,
                    csv_text=products.csv_bytes.decode(
                        "utf-8", errors="strict",
                    ),
                    report_sha256=products.inventory_document[
                        "report_sha256"
                    ],
                    expected_launch_count=plan[
                        "trace_inventory_cardinality"
                    ],
                    protocol=protocol,
                    registry=registry,
                    percentage_metric_names=percentage_metric_names,
                    unit_metric_names=unit_metric_names,
                    sidecar_schema=sidecar_schema,
                    provenance=sidecar_provenance,
                )
            )
            retained_report.verify_unchanged()
            retained_manifest.verify_unchanged()
        receipt = details_persistence.persist_details_export_products(
            artifact_root=artifact_root,
            receipt_relative_path=relative[
                "run_0_launch_set.details_export_receipt.json"
            ],
            products=products,
            expected_preflight_sha256=expected_preflight_sha256,
            expected_source_commit=expected_source_commit,
            expected_capture_identity_sha256=(
                expected_capture_identity_sha256
            ),
            expected_ncu_identity=expected_ncu_identity,
            expected_launch_count=plan["trace_inventory_cardinality"],
            protocol=protocol,
            registry=registry,
            percentage_metric_names=percentage_metric_names,
            unit_metric_names=unit_metric_names,
        )
        execution_identity, capture_manifest, published_sidecars = (
            _persist_fixture_launch_set_authorities(
                cell_root=cell_root,
                artifacts=artifacts,
                command=command,
                protocol=protocol,
                expected_launch_count=plan[
                    "trace_inventory_cardinality"
                ],
                expected_target_executable_sha256=(
                    expected_target_executable_sha256
                ),
                expected_runtime_module_sha256s=(
                    expected_runtime_module_sha256s
                ),
                execution_identity_document=(
                    execution_identity_document
                ),
                products=products,
                details_export_receipt=receipt,
                expected_run_manifest_document=manifest_document,
                expected_run_manifest_sha256=run_manifest_sha256,
                expected_run_manifest_size_bytes=(
                    run_manifest_size_bytes
                ),
                sidecar_documents=sidecar_documents,
                sidecar_paths=sidecar_paths,
                sidecar_schema_bytes=sidecar_schema_bytes,
                sidecar_schema=sidecar_schema,
                sidecar_provenance=sidecar_provenance,
                registry=registry,
                percentage_metric_names=percentage_metric_names,
                unit_metric_names=unit_metric_names,
            )
        )
    except (
        OSError,
        immutable.ImmutableArtifactError,
        collection_receipt.CollectionReceiptError,
        ncu_launch_documents.LaunchDocumentError,
        details_persistence.DetailsExportPersistenceError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ChunkRefusal(
            "label_launch_set_postprocessing_failed", str(exc),
        ) from exc
    return {
        "report": report,
        "run_manifest": run_manifest,
        "details_csv": artifacts["run_0_launch_set.details.csv"],
        "details_stderr": artifacts["run_0_launch_set.details.stderr"],
        "launch_inventory": artifacts[
            "run_0_launch_set.launch_inventory.json"
        ],
        "launch_metrics": artifacts["run_0_launch_set.launch_metrics.json"],
        "details_export_receipt": receipt,
        "execution_identity": execution_identity,
        "sidecars": published_sidecars,
        "capture_manifest": capture_manifest,
    }


def _retained_json_model(
    retained: immutable.RetainedRegular, *, role: str,
) -> dict[str, Any]:
    payload = bytearray()
    offset = 0
    try:
        while offset < retained.size_bytes:
            block = os.pread(
                retained.descriptor,
                min(1 << 20, retained.size_bytes - offset),
                offset,
            )
            if not block:
                raise OSError(None, f"short retained {role} read")
            payload.extend(block)
            offset += len(block)
    except OSError as exc:
        _raise_typed_storage_failure(
            exc,
            code="label_capture_storage_failure",
            operation=f"label_collection_receipt_{role}_read",
        )
        raise
    try:
        value = immutable.strict_json_loads(
            bytes(payload), label=f"retained {role}",
        )
    except immutable.ImmutableArtifactError as exc:
        raise collection_receipt.CollectionReceiptError(str(exc)) from exc
    if type(value) is not dict:
        raise collection_receipt.CollectionReceiptError(
            f"retained {role} root is not an exact object"
        )
    return value


def _write_collection_execution_receipt(
    *,
    cell_dir: Path,
    artifact_tag: str,
    preflight_sha256: str,
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    command: Sequence[str],
    environment: ncu_executor.NcuCollectionEnvironment,
    working_directory: Path,
    report: Path,
    run_manifest: Path,
    launch_observation_path: Path,
    capture_manifest: Path,
    expected_run_manifest: dict[str, Any],
    expected_launch_observation: dict[str, Any],
    expected_capture_manifest: dict[str, Any],
    launch_observation_sha256: str,
) -> Path:
    receipt_path = (
        cell_dir / "logs" / f"{artifact_tag}_ncu_collection_receipt.json"
    )
    artifact_inputs = (
        ("ncu_report", report),
        ("label_run_manifest", run_manifest),
        ("ncu_launch_observation", launch_observation_path),
        ("label_capture_manifest", capture_manifest),
    )
    try:
        with contextlib.ExitStack() as retained_stack:
            bindings = []
            retained_by_role: dict[str, immutable.RetainedRegular] = {}
            for role, path in artifact_inputs:
                retained = retained_stack.enter_context(
                    immutable.open_retained_regular(path)
                )
                retained_by_role[role] = retained
                try:
                    relative_path = path.relative_to(cell_dir).as_posix()
                except ValueError as exc:
                    raise collection_receipt.CollectionReceiptError(
                        f"collection artifact is outside the cell root: {path}"
                    ) from exc
                bindings.append(collection_receipt.artifact_binding(
                    role=role,
                    relative_path=relative_path,
                    sha256=retained.sha256,
                    size_bytes=retained.size_bytes,
                ))
            if (
                launch_observation_sha256
                != expected_launch_observation.get("observation_sha256")
            ):
                raise collection_receipt.CollectionReceiptError(
                    "launch observation seal differs from the retained model binding"
                )
            if (
                retained_by_role["ncu_report"].sha256
                != expected_launch_observation.get("report_sha256")
            ):
                raise collection_receipt.CollectionReceiptError(
                    "retained NCU report differs from the validated launch observation"
                )
            retained_models = {
                "label_run_manifest": _retained_json_model(
                    retained_by_role["label_run_manifest"],
                    role="label_run_manifest",
                ),
                "ncu_launch_observation": _retained_json_model(
                    retained_by_role["ncu_launch_observation"],
                    role="ncu_launch_observation",
                ),
                "label_capture_manifest": _retained_json_model(
                    retained_by_role["label_capture_manifest"],
                    role="label_capture_manifest",
                ),
            }
            expected_models = {
                "label_run_manifest": expected_run_manifest,
                "ncu_launch_observation": expected_launch_observation,
                "label_capture_manifest": expected_capture_manifest,
            }
            model_mismatches = [
                role for role, expected_model in expected_models.items()
                if immutable.canonical_json_bytes(retained_models[role])
                != immutable.canonical_json_bytes(expected_model)
            ]
            if model_mismatches:
                raise collection_receipt.CollectionReceiptError(
                    "retained collection artifact models differ after validation: "
                    f"{model_mismatches}"
                )
            document = collection_receipt.build_receipt(
                preflight_sha256=preflight_sha256,
                expected_ncu_identity=expected_ncu_identity,
                command=command,
                environment=environment,
                working_directory=working_directory,
                exit_status=0,
                artifacts=bindings,
                launch_observation_sha256=launch_observation_sha256,
            )
            receipt_payload = (
                json.dumps(document, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            _write_output_bytes_exclusive(
                receipt_path,
                receipt_payload,
                code="label_capture_storage_failure",
                operation="label_collection_receipt_write",
                collision_code="label_collection_receipt_exists",
            )
            retained_receipt = retained_stack.enter_context(
                immutable.open_retained_regular(receipt_path)
            )
            if (
                retained_receipt.sha256 != hashlib.sha256(receipt_payload).hexdigest()
                or retained_receipt.size_bytes != len(receipt_payload)
            ):
                raise collection_receipt.CollectionReceiptError(
                    "published collection receipt bytes differ from the producer bytes"
                )
            reloaded = _retained_json_model(
                retained_receipt, role="ncu_collection_receipt",
            )
            collection_receipt.validate_receipt(
                reloaded,
                expected_preflight_sha256=preflight_sha256,
                expected_ncu_identity=expected_ncu_identity,
                expected_command=command,
                expected_environment=environment,
                expected_working_directory=working_directory,
                expected_exit_status=0,
                expected_artifacts=bindings,
                expected_launch_observation_sha256=launch_observation_sha256,
            )
    except (immutable.ImmutableArtifactError, collection_receipt.CollectionReceiptError) as exc:
        raise ChunkRefusal("label_collection_receipt_invalid", str(exc)) from exc
    return receipt_path


def capture_label(
    *, cell: dict, binary: Path, invocation: dict, cell_dir: Path,
    matrix: dict, pins: dict, timeout_s: float,
    preflight_sha256: str,
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    runtime_cwd: Path | None = None,
    runtime_env: dict[str, str] | None = None,
    phase_result_validator: Callable[[dict[str, str], str], dict] | None = None,
    label_launch_index_override: int | None = None,
    kernel_name_selector: str | None = None,
    artifact_tag: str = "label",
    execution_source_commit: str | None = None,
    bind_supplemental_toolchain: bool = False,
) -> dict:
    try:
        ncu_executor.validate_expected_ncu_identity(expected_ncu_identity)
    except ncu_executor.NcuActionExecutorError as exc:
        raise ChunkRefusal("capture_ncu_authority_invalid", str(exc)) from exc
    if not _is_sha256(preflight_sha256):
        raise ChunkRefusal(
            "capture_preflight_authority_invalid", repr(preflight_sha256),
        )
    prepared = _prepare_label_capture_command(
        cell=cell, binary=binary, invocation=invocation, cell_dir=cell_dir,
        matrix=matrix, pins=pins,
        label_launch_index_override=label_launch_index_override,
        kernel_name_selector=kernel_name_selector, artifact_tag=artifact_tag,
        execution_source_commit=execution_source_commit,
        bind_supplemental_toolchain=bind_supplemental_toolchain,
        expected_ncu_identity=expected_ncu_identity,
    )
    command = prepared["command"]
    supervised_command: list[object] = list(command)
    try:
        separator = supervised_command.index("--")
    except ValueError as exc:
        raise ChunkRefusal(
            "label_capture_command_invalid", "missing target argument separator",
        ) from exc
    supervised_command[separator:separator] = [
        "--ncu-exec-path", ncu_executor.AUTHORIZED_NCU_PROC_PATH,
    ]
    identity = prepared["identity"]
    source_commit = prepared["source_commit"]
    labels_root = prepared["labels_root"]
    unit_metrics = prepared["unit_metrics"]
    publication_scope = prepared["publication_scope"]
    expected_cuda_release = prepared["expected_cuda_release"]
    label_shape_point = prepared["label_shape_point"]
    logical_workload = prepared["logical_workload"]
    launch_contract = prepared["launch_contract"]
    shape_binding = prepared["shape_binding"]
    label_launch_index = prepared["label_launch_index"]
    label_cwd = runtime_cwd or REPO_ROOT
    collection_environment = _label_collection_environment(
        runtime_env, cwd=label_cwd,
    )
    elapsed, stdout_log, stderr_log = _run_logged(
        supervised_command, cwd=label_cwd, env=None,
        timeout_s=timeout_s,
        log_dir=cell_dir / "logs", stem=artifact_tag,
        failure_code="label_capture_failed",
        trusted_child_storage_marker=True,
        supervised_ncu_identity=expected_ncu_identity,
        supervised_ncu_environment=collection_environment,
    )
    phase_evidence = (
        phase_result_validator(runtime_env or {}, "label")
        if phase_result_validator is not None else None
    )
    cell_root = (
        labels_root / matrix["arch"] / identity["slug"]
        / label_shape_point
    )
    sidecar = cell_root / f"label_0_{label_launch_index}.json"
    run_manifest = cell_root / f"label_run_manifest_0_{label_launch_index}.json"
    report = cell_root / f"run_0_{label_launch_index}.ncu-rep"
    launch_observation_path = (
        cell_root / f"run_0_{label_launch_index}.launch_observation.json"
    )
    for path, role in (
        (sidecar, "label sidecar"), (run_manifest, "label run manifest"),
        (report, "NCU report"),
        (launch_observation_path, "NCU launch observation"),
    ):
        observed = _output_lstat(
            path, code="label_capture_storage_failure",
            operation=f"label_artifact_stat:{role}",
        )
        if (
            observed is None or not stat.S_ISREG(observed.st_mode)
            or observed.st_size <= 0
        ):
            raise ChunkRefusal("label_artifact_missing", f"{role}: {path}")
    label = _json_output(
        sidecar, code="label_capture_storage_failure",
        operation="label_sidecar_read",
    )
    try:
        import jsonschema
        jsonschema.validate(label, json.loads(LABEL_SCHEMA.read_text()))
    except Exception as exc:
        raise ChunkRefusal("label_schema_invalid", str(exc)) from exc
    protocol = label.get("measurement_protocol") or {}
    tc = pins["toolchain"]
    ref_clocks = pins["reference_clock_state"]
    try:
        observed_cuda_release = toolchain_contract.require_nvcc_release(
            label.get("cuda_version"), tc["nvcc_release"]
        )
        cuda_version_ok = True
    except toolchain_contract.CaptureToolchainContractError:
        observed_cuda_release = None
        cuda_version_ok = False
    checks = {
        "device": label.get("device") == matrix["arch"],
        "slug": label.get("slug") == identity["slug"],
        "shape": label.get("shape_point")
        == label_shape_point,
        "binary": label.get("binary_sha256") == invocation["executable_sha256"],
        "repeat": label.get("repeat_index") == 0,
        "launch": label.get("launch_index") == label_launch_index,
        "clock": protocol.get("ncu_clock_control") == "none",
        "cache": protocol.get("ncu_cache_control") == "all",
        "replay": protocol.get("ncu_replay_mode") == "kernel",
        "l1_flat": type(label.get("cache_hit_rate_l1tex_sector")) in (int, float),
        "l2_flat": type(label.get("cache_hit_rate_l2_sector")) in (int, float),
        "ncu_version": toolchain_contract.ncu_version_matches(
            label.get("profiler_version"), tc["ncu_version_prefix"]
        ),
        "cuda_version": cuda_version_ok,
        "driver_version": label.get("driver_version") == tc["driver_version"],
        "host_persistence": protocol.get("host_persistence_mode")
        == ref_clocks["persistence_mode"],
        "host_sm_clock": protocol.get("host_sm_clock_mhz")
        == ref_clocks["sm_clock_mhz"],
        "host_mem_clock": protocol.get("host_mem_clock_mhz")
        == ref_clocks["mem_clock_mhz"],
    }
    observed_metrics = tuple(
        row.get("ncu_metric") for row in label.get("unit_tagged_metrics") or []
    )
    checks["unit_metrics"] = (
        len(observed_metrics) == len(unit_metrics)
        and set(observed_metrics) == set(unit_metrics)
    )
    if bind_supplemental_toolchain:
        checks["publication_metric_scope"] = (
            publication_scope
            == toolchain_contract.cycles_only_publication_scope()
            and observed_metrics.count("gpc__cycles_elapsed.max") == 1
        )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ChunkRefusal("label_contract_mismatch", f"failed checks: {failed}")
    run = _json_output(
        run_manifest, code="label_capture_storage_failure",
        operation="label_run_manifest_read",
    )
    launch_observation = _json_output(
        launch_observation_path, code="label_capture_storage_failure",
        operation="label_launch_observation_read",
    )
    observation_seal = launch_observation.get("observation_sha256")
    observation_unsealed = {
        key: value for key, value in launch_observation.items()
        if key != "observation_sha256"
    }
    if (
        launch_observation.get("schema_version") != 1
        or launch_observation.get("kind")
        != "ncu_details_csv_launch_observation"
        or launch_observation.get("kernel_name_base") != "mangled"
        or launch_observation.get("requested_launch_skip")
        != label_launch_index
        or launch_observation.get("requested_launch_count") != 1
        or launch_observation.get("report_sha256") != _sha256_output_file(
            report, code="label_capture_storage_failure",
            operation="label_ncu_report_hash",
        )
        or observation_seal != canonical_sha256(observation_unsealed)
        or not isinstance(launch_observation.get("identity"), dict)
    ):
        raise ChunkRefusal(
            "label_launch_observation_invalid", repr(launch_observation)
        )
    _observation_ncu_launch_key(
        launch_observation,
        refusal_code="label_launch_identity_join_mismatch",
    )
    expected_run = {
        "source_commit": source_commit,
        "binary_sha256": invocation["executable_sha256"],
        "arch": matrix["arch"],
        "slug": identity["slug"],
        "shape_point": label_shape_point,
        "repeat_index": 0,
        "launch_index": label_launch_index,
    }
    run_mismatches = {
        key: {"expected": value, "observed": run.get(key)}
        for key, value in expected_run.items() if run.get(key) != value
    }
    if run_mismatches:
        raise ChunkRefusal(
            "label_run_manifest_mismatch", json.dumps(run_mismatches, sort_keys=True)
        )
    capture_manifest = (
        cell_dir / "logs" / f"{artifact_tag}_capture_manifest.json"
    )
    capture_manifest_document = {
        "schema_version": 1,
        "identity": identity,
        "command": command,
        "binary_sha256": invocation["executable_sha256"],
        "source_commit": source_commit,
        "selected_unit_metrics": list(unit_metrics),
        "selected_flat_metrics": matrix["metric_registry"]["collection"][
            "selected_flat_metrics"
        ],
        "toolchain_contract": {
            "expected_cuda_release": (
                expected_cuda_release
                if bind_supplemental_toolchain
                else toolchain_contract.structured_nvcc_release(
                    tc["nvcc_release"]
                )
            ),
            "observed_cuda_release": observed_cuda_release,
            "expected_ncu_version_prefix": tc["ncu_version_prefix"],
            "observed_ncu_version": label.get("profiler_version"),
        },
        "publication_metric_scope": publication_scope,
        "logical_workload_sha256": canonical_sha256(logical_workload),
        "capture_launch_contract": launch_contract,
        "requested_launch_index": label_launch_index,
        "shape_binding": shape_binding,
        "shape_forwarded_to_target": False if shape_binding == "logical-workload" else True,
        "sol_phase_evidence": phase_evidence,
        "elapsed_seconds": elapsed,
    }
    _write_output_text(
        capture_manifest,
        json.dumps(capture_manifest_document, indent=2, sort_keys=True) + "\n",
        operation="label_capture_manifest_write",
    )
    collection_execution_receipt = _write_collection_execution_receipt(
        cell_dir=cell_dir,
        artifact_tag=artifact_tag,
        preflight_sha256=preflight_sha256,
        expected_ncu_identity=expected_ncu_identity,
        command=command,
        environment=collection_environment,
        working_directory=label_cwd,
        report=report,
        run_manifest=run_manifest,
        launch_observation_path=launch_observation_path,
        capture_manifest=capture_manifest,
        expected_run_manifest=run,
        expected_launch_observation=launch_observation,
        expected_capture_manifest=capture_manifest_document,
        launch_observation_sha256=observation_seal,
    )
    return {
        "elapsed_seconds": elapsed,
        "command": command,
        "sidecar": sidecar,
        "run_manifest": run_manifest,
        "ncu_report": report,
        "ncu_launch_observation": launch_observation_path,
        "ncu_launch_observation_document": launch_observation,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "capture_manifest": capture_manifest,
        "ncu_collection_receipt": collection_execution_receipt,
        "sol_phase_evidence": phase_evidence,
        "unit_metric_count": len(unit_metrics),
        "launch_index": label_launch_index,
        "toolchain_contract": {
            "expected_cuda_release": (
                expected_cuda_release
                if bind_supplemental_toolchain
                else toolchain_contract.structured_nvcc_release(
                    tc["nvcc_release"]
                )
            ),
            "observed_cuda_release": observed_cuda_release,
            "expected_ncu_version_prefix": tc["ncu_version_prefix"],
            "observed_ncu_version": label.get("profiler_version"),
        },
        "publication_metric_scope": publication_scope,
    }


def _trace_tree_bytes(trace_dir: Path) -> int:
    total = 0
    for path in _output_tree_paths(
        trace_dir, code="trace_generated_output_storage_failure",
        operation="generated_trace_tree_walk",
    ):
        observed = _output_lstat(
            path, code="trace_generated_output_storage_failure",
            operation="generated_trace_tree_stat",
        )
        if observed is None:
            raise ChunkRefusal("trace_artifact_disappeared", str(path))
        if stat.S_ISLNK(observed.st_mode):
            raise ChunkRefusal("trace_symlink_refused", str(path))
        if stat.S_ISREG(observed.st_mode):
            total += observed.st_size
    return total


def _validate_supplemental_trace_launch_inventory(
    root: dict[str, Any], trace_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Bind every in-ROI root launch to its exact NVBit module mapping.

    Supplemental SOL rows deliberately leave ``MAX_ACCEPTED_ACTIVE_LAUNCHES``
    unset.  This sidecar is the independent producer surface that proves no
    active launch was skipped before the per-ordinal NCU replays begin.
    """
    root_launches = root.get("launches")
    count = root.get("launch_count")
    if (
        type(count) is not int or count <= 0
        or not isinstance(root_launches, list) or len(root_launches) != count
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_inventory_invalid", repr(root_launches)
        )
    source = trace_dir / "extra_info" / "traced_kernel_modules.json"
    source_stat = _output_lstat(
        source, code="trace_generated_output_storage_failure",
        operation="dynamic_launch_inventory_stat",
    )
    if (
        source_stat is None or not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_size <= 0
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_module_inventory_missing", str(source)
        )
    inventory = _json_output(
        source, code="trace_generated_output_storage_failure",
        operation="dynamic_launch_inventory_read",
    )
    expected_module_identity = {
        "stable_key": "sha256",
        "path_semantics": "path_at_capture",
        "rebind_contract": "sha256_required_basename_hint",
    }
    selection = inventory.get("selection") or {}
    mappings = inventory.get("launches")
    expected_selection = {
        "active_from_start": False,
        "max_accepted_active_launches": 0,
        "accepted_active_launches": count,
        "active_launch_attempts": count,
        "active_launches_skipped_after_limit": 0,
        "concurrent_limit_violations": 0,
        "exact_contract_satisfied": True,
        "module_mapping_count": count,
        "inactive_channel_records_dropped": 0,
    }
    if (
        inventory.get("schema_version") != 1
        or inventory.get("module_identity") != expected_module_identity
        or any(selection.get(key) != value for key, value in expected_selection.items())
        or type(selection.get("accepted_channel_records")) is not int
        or selection["accepted_channel_records"] <= 0
        or not isinstance(mappings, list) or len(mappings) != count
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_selection_incomplete",
            repr({"expected": expected_selection, "observed": selection}),
        )
    mapping_fields = {
        "accepted_launch_index", "device_id", "stream_id", "trace_launch_id",
        "kernel_name", "resolution_source", "module",
    }
    module_fields = {"status", "reason", "path_at_capture", "basename", "sha256"}
    bound: list[dict[str, Any]] = []
    for ordinal, (launch, mapping) in enumerate(
        zip(root_launches, mappings, strict=True)
    ):
        if (
            not isinstance(launch, dict) or not isinstance(mapping, dict)
            or set(mapping) != mapping_fields
            or mapping.get("accepted_launch_index") != ordinal
            or mapping.get("device_id") != launch.get("device_id")
            or mapping.get("stream_id") != launch.get("stream_id")
            or mapping.get("trace_launch_id") != launch.get("kernel_id")
            or not isinstance(mapping.get("kernel_name"), str)
            or not mapping["kernel_name"]
            or launch.get("kernel_name") != mapping["kernel_name"] + "___0"
            or not isinstance(mapping.get("resolution_source"), str)
            or not mapping["resolution_source"]
            or not isinstance(mapping.get("module"), dict)
            or set(mapping["module"]) != module_fields
        ):
            raise ChunkRefusal(
                "sol_dynamic_launch_mapping_invalid",
                repr({"ordinal": ordinal, "launch": launch, "mapping": mapping}),
            )
        module = mapping["module"]
        if module.get("status") == "stable_static_module":
            if (
                module.get("reason") != "resolved_file"
                or not isinstance(module.get("path_at_capture"), str)
                or not module["path_at_capture"]
                or not isinstance(module.get("basename"), str)
                or not module["basename"]
                or not isinstance(module.get("sha256"), str)
                or len(module["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in module["sha256"])
            ):
                raise ChunkRefusal(
                    "sol_dynamic_launch_module_identity_invalid", repr(module)
                )
        elif module.get("status") == "no_stable_static_module":
            if (
                module.get("reason") not in {
                    "origin_unresolved", "jit_or_anonymous_mapping",
                    "module_path_deleted", "module_path_missing",
                    "module_not_regular_file", "module_hash_failed",
                }
                or any(module.get(key) not in {"", None} for key in (
                    "path_at_capture", "basename", "sha256",
                ))
            ):
                raise ChunkRefusal(
                    "sol_dynamic_launch_module_identity_invalid", repr(module)
                )
        else:
            raise ChunkRefusal(
                "sol_dynamic_launch_module_identity_invalid", repr(module)
            )
        row = {
            "launch_ordinal": ordinal,
            "trace_launch": copy.deepcopy(launch),
            "module_mapping": copy.deepcopy(mapping),
        }
        row["launch_identity_sha256"] = canonical_sha256(row)
        bound.append(row)
    return bound, canonical_sha256(bound)


def _translate_generated_trace_storage(
    exc: BaseException, *, operation: str,
) -> None:
    """Type only faults from explicit generated-trace read boundaries."""
    if isinstance(exc, preflight.PreflightStorageFailure):
        origin = _storage_origin(exc)
        if origin is None:
            raise ChunkRefusal(
                "trace_generated_output_validation_failed", str(exc)
            ) from exc
    elif isinstance(exc, OSError):
        origin = exc
    else:
        return
    _raise_typed_storage_failure(
        origin, code="trace_generated_output_storage_failure",
        operation=operation,
    )


def capture_trace(
    *, cell: dict, binary: Path, invocation: dict, budget: Any,
    cell_dir: Path, pins: dict, expected_tracer_sha256: str,
    runtime_cwd: Path | None = None,
    runtime_env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    capture_provenance: dict[str, Any] | None = None,
    phase_env_factory: Callable[[str, float], dict[str, str]] | None = None,
    phase_result_validator: Callable[[dict[str, str], str], dict] | None = None,
    all_profiler_roi_launches: bool = False,
) -> dict:
    provenance = capture_provenance or {}
    raw_staging_limit = provenance.get("raw_staging_margin_bytes")
    if type(raw_staging_limit) is not int or raw_staging_limit <= 0:
        raise ChunkRefusal(
            "trace_staging_limit_missing",
            f"raw_staging_margin_bytes={raw_staging_limit!r}",
        )
    raw_trace_dir = cell_dir / "trace_full_raw"
    retained_trace_dir = cell_dir / "trace"
    _mkdir_output(
        raw_trace_dir, parents=True, exist_ok=False,
        operation="raw_trace_directory_create",
    )
    inj = pins["injection_runtime"]
    tracer_so = _resolve_repo_path(inj["tracer_so_relpath"], "tracer .so")
    _require_hash(tracer_so, expected_tracer_sha256, "tracer .so before trace")
    pb_lib = _resolve_repo_path(inj["pb_lib_relpath"], "protobuf runtime")
    env = dict(runtime_env) if runtime_env is not None else dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        f"{pb_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    )
    env["CUDA_INJECTION64_PATH"] = str(tracer_so)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(raw_trace_dir)
    env["TRACE_SAMPLING_MODE"] = "0"
    env.pop("TRACE_SAMPLING_SELECTED_BLOCKS", None)
    is_sol = cell["identity"]["roster_source"] == "sol_execbench"
    if all_profiler_roi_launches and not is_sol:
        raise ChunkRefusal(
            "all_profiler_roi_launches_invalid", repr(cell["identity"])
        )
    if is_sol:
        env["ACTIVE_FROM_START"] = "0"
        if all_profiler_roi_launches:
            env.pop("MAX_ACCEPTED_ACTIVE_LAUNCHES", None)
        else:
            env["MAX_ACCEPTED_ACTIVE_LAUNCHES"] = "1"
    else:
        env["ACTIVE_FROM_START"] = "1"
        env.pop("MAX_ACCEPTED_ACTIVE_LAUNCHES", None)
    command = [str(binary), *invocation["trace_app_args"]]
    if cell["identity"]["roster_source"] == "portable_lcd":
        command += [
            "--output_dir", str(cell_dir / "logs" / "trace_workload"),
        ]
    trace_timeout = timeout_s if timeout_s is not None else budget.max_wall_time_s
    trace_deadline = time.monotonic() + trace_timeout

    def remaining(phase: str) -> float:
        value = trace_deadline - time.monotonic()
        if value <= 0:
            raise ChunkRefusal(
                "trace_route_deadline_exhausted",
                f"{cell['identity']}: phase={phase}",
            )
        return value

    elapsed, stdout_log, stderr_log = _run_logged(
        command, cwd=runtime_cwd or binary.parent, env=env,
        timeout_s=remaining("full"),
        log_dir=cell_dir / "logs", stem="trace", failure_code="trace_target_failed",
        monitored_tree=raw_trace_dir, tree_limit_bytes=raw_staging_limit,
    )
    full_phase_evidence = (
        phase_result_validator(env, "trace")
        if phase_result_validator is not None else None
    )
    _require_hash(tracer_so, expected_tracer_sha256, "tracer .so after trace")
    root_pb = raw_trace_dir / "dynamic_trace.pb"
    stats_csv = raw_trace_dir / "stats.csv"
    try:
        root = preflight._validate_root_trace(root_pb, REPO_ROOT)
        launch_contract = _require_capture_launch_contract(invocation)
        expected_launch_count = (
            root.get("launch_count")
            if all_profiler_roi_launches else launch_contract["trace_launch_count"]
        )
        if type(expected_launch_count) is not int or expected_launch_count <= 0:
            raise ChunkRefusal(
                "sol_dynamic_launch_inventory_empty", repr(expected_launch_count)
            )
        contract = validate_trace_dir(raw_trace_dir)
        threadblocks = _output_tree_paths(
            raw_trace_dir / "threadblocks",
            code="trace_generated_output_storage_failure",
            operation="generated_trace_threadblock_walk", pattern="*.pb",
        )
        stats_details = preflight._parse_stats_admission_details(stats_csv)
        expected, admitted = stats_details["expected"], stats_details["admitted"]
        coverage = preflight._analyze_full_trace_coverage(
            raw_trace_dir, root, expected, admitted, stats_details["row_count"],
            expected_launch_count=expected_launch_count,
        )
    except (OSError, preflight.PreflightStorageFailure) as exc:
        _translate_generated_trace_storage(
            exc, operation="generated_trace_validation_read",
        )
        raise
    if root.get("status") != preflight.STATUS_OK:
        raise ChunkRefusal("trace_root_invalid", root.get("detail", "unknown"))
    if not contract.ok:
        raise ChunkRefusal(
            contract.typed_reason or "trace_contract_invalid", contract.detail or ""
        )
    if not threadblocks:
        raise ChunkRefusal(
            "trace_payload_empty", "contract walker saw zero threadblock protobufs"
        )
    if coverage.get("status") != preflight.STATUS_OK:
        raise ChunkRefusal("trace_full_coverage_invalid", coverage.get("detail", ""))
    try:
        if all_profiler_roi_launches:
            dynamic_launch_inventory, dynamic_launch_inventory_sha256 = (
                _validate_supplemental_trace_launch_inventory(root, raw_trace_dir)
            )
        else:
            _validate_trace_launch_contract(invocation, root)
            dynamic_launch_inventory = None
            dynamic_launch_inventory_sha256 = None
    except OSError as exc:
        _translate_generated_trace_storage(
            exc, operation="generated_trace_inventory_read",
        )
        raise
    if type(expected) is not int or type(admitted) is not int or expected <= 0:
        raise ChunkRefusal(
            "trace_stats_invalid", f"expected={expected!r}, admitted={admitted!r}"
        )
    if admitted <= 0 or admitted > expected:
        raise ChunkRefusal(
            "trace_admission_invalid", f"expected={expected}, admitted={admitted}"
        )
    drop_rate = (expected - admitted) / expected
    try:
        trace_bytes = _trace_tree_bytes(raw_trace_dir)
    except OSError as exc:
        _translate_generated_trace_storage(
            exc, operation="generated_trace_tree_size_read",
        )
        raise
    manifest = {
        "schema_version": 1,
        "identity": cell["identity"],
        "captured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "binary_sha256": invocation["executable_sha256"],
        "tracer_so_sha256": sha256_file(tracer_so),
        "trace_contract_id": root["trace_contract_id"],
        "memory_payload_classes": root["memory_payload_classes"],
        "capture_launch_contract": launch_contract,
        "launch_scope": (
            "all_in_profiler_roi" if all_profiler_roi_launches
            else "fixed_capture_launch_contract"
        ),
        "dynamic_launch_inventory": dynamic_launch_inventory,
        "dynamic_launch_inventory_sha256": dynamic_launch_inventory_sha256,
        "launch_count": root["launch_count"],
        "launches": root["launches"],
        "threadblocks_pb_count": len(threadblocks),
        "grid_threadblock_count": coverage["grid_threadblock_count"],
        "decoded_threadblock_count": coverage["decoded_threadblock_count"],
        "decoded_record_count": coverage["decoded_record_count"],
        "zero_record_threadblock_count": coverage["zero_record_threadblock_count"],
        "stats_row_count": coverage["stats_row_count"],
        "expected_records": expected,
        "admitted_records": admitted,
        "drop_rate": drop_rate,
        "trace_tree_bytes": trace_bytes,
        "max_trace_bytes": budget.max_trace_bytes,
        "elapsed_seconds": elapsed,
        "sol_phase_evidence": full_phase_evidence,
    }
    manifest_path = cell_dir / "trace_capture_manifest.json"
    _write_output_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        operation="trace_capture_manifest_write",
    )
    route_result = None
    route_result_path = None
    routed = trace_bytes > budget.max_trace_bytes
    if not routed:
        _replace_output(
            raw_trace_dir, retained_trace_dir,
            operation="retained_trace_directory_publish",
        )
        control_trace_dir = retained_trace_dir
    else:
        if all_profiler_roi_launches:
            # Supplemental evidence is comprehensive only when every raw
            # launch payload is retained.  Never replace an oversized complete
            # trace with metadata and call the row terminally accounted.
            raise ChunkRefusal(
                "supplemental_full_trace_retention_ceiling_exceeded",
                f"trace={trace_bytes}; ceiling={budget.max_trace_bytes}",
            )
        elif launch_contract["trace_launch_count"] != 1:
            raise ChunkRefusal(
                "multi_launch_selective_route_unsupported",
                "the full cold/evict/warm trace exceeds its 2 GiB cell budget; "
                "first-launch-only selective routing would destroy the workload contract",
            )
        else:
            required_provenance = {
                "protocol_version", "source_commit", "capture_matrix_sha256",
                "preflight_sha256", "pins_sha256", "trace_budget_sha256",
                "raw_staging_margin_bytes",
            }
            missing_provenance = sorted(required_provenance - set(provenance))
            if missing_provenance:
                raise ChunkRefusal(
                    "trace_route_provenance_missing", repr(missing_provenance)
                )
            args_contract = {
                "adapter_kind": invocation.get("adapter_kind"),
                "trace_app_args": invocation.get("trace_app_args"),
                "logical_workload": invocation.get("logical_workload"),
                "trace_cardinality_contract": invocation.get(
                    "trace_cardinality_contract"
                ),
            }
            for additive in ("capture_launch_contract", "matrix_capture_override"):
                if additive in invocation:
                    args_contract[additive] = invocation[additive]
            args_hash = canonical_sha256(args_contract)
            protocol = {
                "protocol_version": provenance["protocol_version"],
                "source_commit": provenance["source_commit"],
                "authority_cell_sha256": cell["authority_cell_sha256"],
                "capture_matrix_sha256": provenance["capture_matrix_sha256"],
                "preflight_sha256": provenance["preflight_sha256"],
                "pins_sha256": provenance["pins_sha256"],
                "trace_budget_sha256": provenance["trace_budget_sha256"],
                "capture_manifest_sha256": _sha256_output_file(
                    manifest_path, code="trace_generated_output_storage_failure",
                    operation="trace_capture_manifest_hash",
                ),
                "binary_sha256": invocation["executable_sha256"],
                "tracer_so_sha256": sha256_file(tracer_so),
                "args_hash": args_hash,
            }

            def sampling_capture(request: budget_route.CaptureRequest):
                phase_tag = request.kind
                if request.attempt is not None:
                    phase_tag += f"_{request.attempt}"
                callback_timeout = remaining(phase_tag)
                callback_env = (
                    phase_env_factory(phase_tag, callback_timeout)
                    if phase_env_factory is not None
                    else (
                        dict(runtime_env) if runtime_env is not None
                        else dict(os.environ)
                    )
                )
                callback_env["LD_LIBRARY_PATH"] = (
                    f"{pb_lib}:{callback_env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
                )
                callback_env["CUDA_INJECTION64_PATH"] = str(tracer_so)
                callback_env["USER_DEFINED_FOLDERS"] = "1"
                callback_env["TRACES_FOLDER"] = str(request.output_dir)
                mode = 1 if request.kind == "trace_sampling_counting_v2" else 2
                callback_env["TRACE_SAMPLING_MODE"] = str(mode)
                if mode == 2:
                    callback_env["TRACE_SAMPLING_SELECTED_BLOCKS"] = ",".join(
                        str(block) for block in request.selected_block_ids
                    )
                else:
                    callback_env.pop("TRACE_SAMPLING_SELECTED_BLOCKS", None)
                if is_sol:
                    callback_env["ACTIVE_FROM_START"] = "0"
                    callback_env["MAX_ACCEPTED_ACTIVE_LAUNCHES"] = "1"
                else:
                    callback_env["ACTIVE_FROM_START"] = "1"
                    callback_env.pop("MAX_ACCEPTED_ACTIVE_LAUNCHES", None)
                _mkdir_output(
                    request.output_dir, parents=True, exist_ok=False,
                    operation="trace_sampling_directory_create",
                )
                callback_command = list(command)
                if cell["identity"]["roster_source"] == "portable_lcd":
                    output_index = callback_command.index("--output_dir") + 1
                    callback_command[output_index] = str(
                        cell_dir / "logs" / f"{phase_tag}_workload"
                    )
                _run_logged(
                    callback_command, cwd=runtime_cwd or binary.parent,
                    env=callback_env, timeout_s=callback_timeout,
                    log_dir=cell_dir / "logs", stem=phase_tag,
                    failure_code="trace_sampling_target_failed",
                    monitored_tree=request.output_dir,
                    tree_limit_bytes=request.staging_remaining_bytes,
                )
                if phase_result_validator is not None:
                    phase_result_validator(callback_env, "trace")
                _require_hash(
                    tracer_so, expected_tracer_sha256,
                    f"tracer .so after {phase_tag}",
                )
                return budget_route.CaptureReceipt(
                    trace_dir=request.output_dir,
                    request_sha256=request.request_sha256,
                    binary_sha256=invocation["executable_sha256"],
                    tracer_so_sha256=sha256_file(tracer_so),
                )

            selection_limit = max(1, budget.max_trace_bytes * 3 // 4)
            config = budget_route.RouteConfig(
                selection_max_raw_bytes=selection_limit,
                staging_limit_bytes=provenance["raw_staging_margin_bytes"],
                gate_forbidden_policy="typed_unavailable",
            )
            try:
                route_result = budget_route.route_trace_budget(
                    raw_trace_dir, manifest_path, retained_trace_dir,
                    cell_id=cell["identity"]["slug"], identity=cell["identity"],
                    budget={
                        "max_trace_bytes": budget.max_trace_bytes,
                        "max_wall_time_s": budget.max_wall_time_s,
                        "sampling_policy": budget.sampling_policy,
                    },
                    protocol=protocol, config=config,
                    callbacks=budget_route.RouteCallbacks(
                        counting_capture=sampling_capture,
                        selective_capture=sampling_capture,
                    ),
                )
            except budget_route.BudgetRouteRefusal as exc:
                raise ChunkRefusal(
                    f"trace_budget_route_{exc.reason}", exc.detail or ""
                ) from exc
            route_result_path = cell_dir / "trace_budget_route.json"
            try:
                budget_route.write_route_result(route_result, route_result_path)
            except OSError as exc:
                _raise_typed_storage_failure(
                    exc, code="capture_output_storage_failure",
                    operation="trace_budget_route_result_write",
                )
                raise
            control_trace_dir = raw_trace_dir

    try:
        retained_bytes = _trace_tree_bytes(retained_trace_dir)
    except OSError as exc:
        _translate_generated_trace_storage(
            exc, operation="retained_trace_tree_size_read",
        )
        raise
    if retained_bytes > budget.max_trace_bytes:
        raise ChunkRefusal(
            "trace_retained_budget_exceeded",
            f"retained={retained_bytes}, max={budget.max_trace_bytes}",
        )
    retained_root = retained_trace_dir / "dynamic_trace.pb"
    retained_stats = retained_trace_dir / "stats.csv"
    retained_root_stat = _output_lstat(
        retained_root, code="trace_generated_output_storage_failure",
        operation="retained_trace_root_stat",
    )
    retained_stats_stat = _output_lstat(
        retained_stats, code="trace_generated_output_storage_failure",
        operation="retained_trace_stats_stat",
    )
    trace_outcome = (
        route_result.get("outcome") if route_result is not None else "admitted"
    )
    return {
        "manifest": manifest_path,
        "route_result_path": route_result_path,
        "route_result": route_result,
        "trace_dir": retained_trace_dir,
        "control_trace_dir": control_trace_dir,
        "raw_trace_dir": raw_trace_dir if routed else None,
        "root_pb": (
            retained_root if retained_root_stat is not None
            and stat.S_ISREG(retained_root_stat.st_mode) else None
        ),
        "stats_csv": (
            retained_stats if retained_stats_stat is not None
            and stat.S_ISREG(retained_stats_stat.st_mode) else None
        ),
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "routed": routed,
        "outcome": trace_outcome,
        "typed_reason": (
            route_result.get("typed_reason") if route_result is not None else None
        ),
        "retained_trace_bytes": retained_bytes,
        **manifest,
    }


def capture_control(
    *, cell: dict, binary: Path, invocation: dict, cell_dir: Path,
    timeout_s: float = 300,
) -> dict:
    out = cell_dir / "control_bits.json"
    command = [
        sys.executable, str(CONTROL_EMITTER), "--binary", str(binary),
        "--arch", "sm_86", "--out", str(out),
    ]
    elapsed, stdout_log, stderr_log = _run_logged(
        command, cwd=REPO_ROOT, env=None, timeout_s=timeout_s,
        log_dir=cell_dir / "logs", stem="control",
        failure_code="control_bits_unavailable",
    )
    payload = _json_output(
        out, code="control_bits_storage_failure",
        operation="direct_control_bits_read",
    )
    summary = payload.get("summary") or {}
    if payload.get("arch") != "sm_86":
        raise ChunkRefusal("control_arch_mismatch", repr(payload.get("arch")))
    if summary.get("total_decoded", 0) <= 0:
        raise ChunkRefusal("control_payload_empty", repr(summary))
    if summary.get("total_unknown_encoding") != 0:
        raise ChunkRefusal("control_unknown_encoding", repr(summary))
    wrapper = {
        "schema_version": 1,
        "identity": cell["identity"],
        "binary_sha256": invocation["executable_sha256"],
        "control_bits_sha256": _sha256_output_file(
            out, code="control_bits_storage_failure",
            operation="direct_control_bits_hash",
        ),
        "toolkit_version": payload.get("toolkit_version"),
        "total_decoded": summary["total_decoded"],
        "total_unknown_encoding": summary["total_unknown_encoding"],
        "elapsed_seconds": elapsed,
    }
    wrapper_path = cell_dir / "control_capture_manifest.json"
    _write_output_text(
        wrapper_path, json.dumps(wrapper, indent=2, sort_keys=True) + "\n",
        operation="control_capture_manifest_write",
    )
    return {
        "control_bits": out, "manifest": wrapper_path,
        "stdout_log": stdout_log, "stderr_log": stderr_log, **wrapper,
    }


def _runtime_control_instruction_valid(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    pc = record.get("pc_num_dec")
    encoded = record.get("encoded_instruction")
    control = record.get("control_bits")
    if type(pc) is not int or pc < 0:
        return False
    if not isinstance(encoded, list) or len(encoded) != 2:
        return False
    if not all(
        isinstance(word, str)
        and len(word) == 18
        and word.startswith("0x")
        and all(ch in "0123456789abcdefABCDEF" for ch in word[2:])
        for word in encoded
    ):
        return False
    if not isinstance(control, dict):
        return False
    exact_ints = (
        "stall_count", "id_new_read_barrier", "id_new_write_barrier",
        "wait_barrier_bits",
    )
    exact_bools = (
        "is_yield", "is_new_read_barrier", "is_new_write_barrier",
    )
    if not all(type(control.get(name)) is int for name in exact_ints):
        return False
    if not all(type(control.get(name)) is bool for name in exact_bools):
        return False
    return (
        0 <= control["stall_count"] <= 15
        and 0 <= control["id_new_read_barrier"] <= 7
        and 0 <= control["id_new_write_barrier"] <= 7
        and 0 <= control["wait_barrier_bits"] <= 63
    )


def _copy_control_evidence(source: Path, destination: Path) -> Path:
    source_stat = _output_lstat(
        source, code="sol_control_storage_failure",
        operation="sol_control_source_evidence_stat",
    )
    if (
        source_stat is None or not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_size <= 0
    ):
        raise ChunkRefusal("sol_control_evidence_invalid", str(source))
    _mkdir_output(
        destination.parent, parents=True, exist_ok=True,
        code="sol_control_storage_failure",
        operation="sol_control_evidence_parent_create",
    )
    if destination.exists() or destination.is_symlink():
        raise ChunkRefusal("sol_control_evidence_exists", str(destination))
    _copy_output_file(
        source, destination, exclusive=True,
        code="sol_control_storage_failure",
        operation="sol_control_evidence_copy",
    )
    if _sha256_output_file(
        destination, code="sol_control_storage_failure",
        operation="sol_control_destination_evidence_hash",
    ) != _sha256_output_file(
        source, code="sol_control_storage_failure",
        operation="sol_control_source_evidence_hash",
    ):
        raise ChunkRefusal("sol_control_evidence_copy_mismatch", str(source))
    return destination


def _sol_dynamic_trace_pcs(trace_dir: Path) -> set[int]:
    threadblocks = _output_tree_paths(
        trace_dir / "threadblocks", code="sol_control_storage_failure",
        operation="sol_control_threadblock_walk", pattern="*.pb",
    )
    if not threadblocks:
        raise ChunkRefusal("sol_control_dynamic_trace_empty", str(trace_dir))
    try:
        threadblock_pb2, _instruction_pb2, _address_pb2 = (
            __import__("import_trace_contract")._import_pb_modules()
        )
    except Exception as exc:
        raise ChunkRefusal("sol_control_protobuf_bindings_missing", str(exc)) from exc
    pcs: set[int] = set()
    for path in threadblocks:
        observed = _output_lstat(
            path, code="sol_control_storage_failure",
            operation="sol_control_threadblock_stat",
        )
        if observed is None or not stat.S_ISREG(observed.st_mode):
            raise ChunkRefusal("sol_control_dynamic_trace_invalid", str(path))
        message = threadblock_pb2.threadblock()
        try:
            message.ParseFromString(_read_output_bytes(
                path, code="sol_control_storage_failure",
                operation="sol_control_threadblock_read",
            ))
        except Exception as exc:
            raise ChunkRefusal("sol_control_dynamic_trace_invalid", str(path)) from exc
        for warp in message.warps.values():
            for instruction in warp.instructions:
                pcs.add(int(instruction.pc))
    if not pcs:
        raise ChunkRefusal("sol_control_dynamic_trace_empty", str(trace_dir))
    return pcs


def _preserve_sol_module(
    module: dict, *, cell_dir: Path, module_store: Path,
) -> tuple[Path, str]:
    raw_path = module.get("path_at_capture")
    expected_sha = module.get("sha256")
    expected_basename = module.get("basename")
    if not all(isinstance(value, str) and value for value in (
        raw_path, expected_sha, expected_basename,
    )):
        raise ChunkRefusal("sol_control_module_identity_invalid", repr(module))
    declared = Path(raw_path)
    if not declared.is_absolute() or declared.is_symlink():
        raise ChunkRefusal("sol_control_module_identity_invalid", raw_path)
    try:
        source = declared.resolve(strict=True)
    except OSError as exc:
        raise ChunkRefusal("sol_control_module_absent", raw_path) from exc
    if str(source) != raw_path or source.name != expected_basename:
        raise ChunkRefusal("sol_control_module_identity_invalid", raw_path)
    _require_hash(source, expected_sha, "SOL loaded module")
    try:
        relative = source.relative_to(cell_dir.resolve())
        preserved = cell_dir / relative
    except ValueError:
        safe_basename = "".join(
            ch if ch.isalnum() or ch in "._-" else "_"
            for ch in expected_basename
        )
        if not module_store.is_absolute():
            raise ChunkRefusal("sol_control_module_store_invalid", str(module_store))
        _mkdir_output(
            module_store, parents=True, exist_ok=True,
            code="sol_control_storage_failure",
            operation="sol_control_module_store_create",
        )
        if module_store.is_symlink():
            raise ChunkRefusal("sol_control_module_store_invalid", str(module_store))
        stored = module_store / f"{expected_sha}-{safe_basename}"
        if stored.exists() or stored.is_symlink():
            if stored.is_symlink() or not stored.is_file():
                raise ChunkRefusal("sol_control_module_store_invalid", str(stored))
            _require_hash(stored, expected_sha, "stored SOL loaded module")
        else:
            temp = module_store / f".{stored.name}.{uuid.uuid4().hex}.tmp"
            storage_failed = False
            try:
                _copy_output_file(
                    source, temp, exclusive=True,
                    code="sol_control_storage_failure",
                    operation="sol_control_module_store_copy",
                )
                _require_hash(temp, expected_sha, "copied SOL loaded module")
                try:
                    _link_output(
                        temp, stored, code="sol_control_storage_failure",
                        operation="sol_control_module_store_publish",
                    )
                except FileExistsError:
                    _require_hash(stored, expected_sha, "stored SOL loaded module")
            except BaseException as exc:
                storage_failed = _storage_origin(exc) is not None
                raise
            finally:
                if not storage_failed:
                    _unlink_output(
                        temp, missing_ok=True,
                        code="sol_control_storage_failure",
                        operation="sol_control_module_temp_cleanup",
                    )
        preserved = cell_dir / "control_evidence" / "modules" / stored.name
        _mkdir_output(
            preserved.parent, parents=True, exist_ok=True,
            code="sol_control_storage_failure",
            operation="sol_control_module_evidence_parent_create",
        )
        try:
            _link_output(
                stored, preserved, code="sol_control_storage_failure",
                operation="sol_control_module_evidence_link",
            )
        except FileExistsError as exc:
            raise ChunkRefusal("sol_control_evidence_exists", str(preserved)) from exc
    _require_hash(preserved, expected_sha, "preserved SOL loaded module")
    _chmod_output(
        preserved, 0o444, code="sol_control_storage_failure",
        operation="sol_control_module_evidence_chmod",
    )
    return preserved, preserved.relative_to(cell_dir).as_posix()


def _write_control_evidence_bytes(payload: bytes, destination: Path) -> Path:
    _mkdir_output(
        destination.parent, parents=True, exist_ok=True,
        code="sol_control_storage_failure",
        operation="vendor_control_evidence_parent_create",
    )
    if destination.exists() or destination.is_symlink():
        raise ChunkRefusal("vendor_control_evidence_exists", str(destination))
    temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    storage_failed = False
    try:
        _write_output_bytes(
            temp, payload, code="sol_control_storage_failure",
            operation="vendor_control_evidence_temp_write",
        )
        _link_output(
            temp, destination, code="sol_control_storage_failure",
            operation="vendor_control_evidence_publish",
        )
    except FileExistsError as exc:
        raise ChunkRefusal(
            "vendor_control_evidence_exists", str(destination)
        ) from exc
    except BaseException as exc:
        storage_failed = _storage_origin(exc) is not None
        raise
    finally:
        if not storage_failed:
            _unlink_output(
                temp, missing_ok=True, code="sol_control_storage_failure",
                operation="vendor_control_evidence_temp_cleanup",
            )
    return destination


def _validated_sol_static_tools(contract: dict) -> tuple[Path, Path]:
    if contract != SOL_STATIC_CONTROL_REBIND:
        raise ChunkRefusal(
            "sol_control_static_rebind_contract_invalid", repr(contract)
        )
    nvcc = Path(contract["nvcc_path"])
    cuobjdump = Path(contract["cuobjdump_path"])
    for role, path, expected_sha in (
        ("pinned SOL control nvcc", nvcc, contract["nvcc_sha256"]),
        (
            "pinned SOL control cuobjdump", cuobjdump,
            contract["cuobjdump_sha256"],
        ),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ChunkRefusal("sol_control_static_tool_invalid", f"{role}: {path}")
        _require_hash(path, expected_sha, role)
    return nvcc, cuobjdump


def _sol_static_rebind_controls(
    *, contract: dict, preserved_module: Path, function_name: str,
    descriptors: list[dict], dynamic_pcs: set[int], label: dict,
    evidence_dir: Path,
) -> tuple[list[dict], dict]:
    """Rebind no-binary runtime descriptors to one exact static text section."""
    nvcc, cuobjdump = _validated_sol_static_tools(contract)
    try:
        extracted = sass_extractor.extract_exact_function_text(
            preserved_module,
            function_name=function_name,
            allowed_code_arches=tuple(contract["allowed_code_arches"]),
            nvcc=str(nvcc),
            cuobjdump=str(cuobjdump),
        )
    except (sass_extractor.FunctionTextExtractionError, RuntimeError) as exc:
        raise ChunkRefusal("sol_control_static_extraction_failed", str(exc)) from exc
    records = extracted.get("records")
    raw_text = extracted.get("raw_text_bytes")
    code_arch = extracted.get("code_arch")
    section_index = extracted.get("section_index")
    section_name = extracted.get("section_name")
    expected_section_name = f"x-{function_name}.{code_arch}.elf.bin"
    section_identity_method = extracted.get(
        "section_identity_method", "list_text_exact_name",
    )
    cubin_symbol_proof = extracted.get("cubin_symbol_proof")
    exact_name_identity = (
        section_identity_method == "list_text_exact_name"
        and section_name == expected_section_name
        and cubin_symbol_proof is None
    )
    symbol_raw_identity = (
        section_identity_method == "cubin_symbol_raw_match"
        and isinstance(cubin_symbol_proof, dict)
        and set(cubin_symbol_proof) == {
            "symbol_index", "symbol_table_index", "text_section_index",
            "text_section_name", "code_arch", "cubin_catalog_index",
            "cubin_file_name", "cubin_sha256", "catalog_raw_match_count",
        }
        and cubin_symbol_proof.get("code_arch") == code_arch
        and cubin_symbol_proof.get("text_section_name")
        == f".text.{function_name}"
        and all(
            type(cubin_symbol_proof.get(key)) is int
            and cubin_symbol_proof[key] > 0
            for key in (
                "symbol_index", "symbol_table_index", "text_section_index",
                "cubin_catalog_index",
            )
        )
        and cubin_symbol_proof.get("catalog_raw_match_count") == 1
        and isinstance(cubin_symbol_proof.get("cubin_file_name"), str)
        and Path(cubin_symbol_proof["cubin_file_name"]).name
        == cubin_symbol_proof["cubin_file_name"]
        and isinstance(cubin_symbol_proof.get("cubin_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", cubin_symbol_proof["cubin_sha256"])
        is not None
        and isinstance(section_name, str)
        and re.fullmatch(
            rf"x-[A-Za-z0-9_.-]+\.{re.escape(str(code_arch))}\.elf\.bin",
            section_name,
        ) is not None
    )
    record_bytes: bytes | None = None
    record_controls_match = False
    if isinstance(records, list):
        try:
            record_bytes = b"".join(
                int(row["low_word"], 16).to_bytes(8, "little")
                + int(row["high_word"], 16).to_bytes(8, "little")
                for row in records
                if isinstance(row, dict)
            )
            record_controls_match = all(
                isinstance(row, dict)
                and row.get("control")
                == sass_extractor.decode_control(int(row["high_word"], 16))
                for row in records
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            record_bytes = None
            record_controls_match = False
    if (
        extracted.get("function_name") != function_name
        or extracted.get("toolkit_version") != contract["toolkit_version"]
        or code_arch not in contract["allowed_code_arches"]
        or extracted.get("parser_contract") != sass_extractor.PARSER_CONTRACT
        or type(section_index) is not int
        or section_index <= 0
        or not isinstance(section_name, str)
        or Path(section_name).name != section_name
        or not (exact_name_identity or symbol_raw_identity)
        or not isinstance(raw_text, bytes)
        or not raw_text
        or len(raw_text) % 16 != 0
        or extracted.get("raw_text_size_bytes") != len(raw_text)
        or extracted.get("raw_text_sha256")
        != hashlib.sha256(raw_text).hexdigest()
        or not isinstance(records, list)
        or not records
        or len(records) * 16 != len(raw_text)
        or record_bytes != raw_text
        or not record_controls_match
        or any(not isinstance(row, dict) or row.get("status") != "decoded"
               for row in records)
    ):
        raise ChunkRefusal(
            "sol_control_static_contract_mismatch",
            repr({
                "toolkit_version": extracted.get("toolkit_version"),
                "code_arch": code_arch,
                "record_count": len(records or []),
            }),
        )
    static_pcs = [row.get("pc") for row in records]
    descriptor_pcs = [
        row.get("pc_num_dec") if isinstance(row, dict) else None
        for row in descriptors
    ]
    if (
        any(type(pc) is not int or pc < 0 for pc in static_pcs)
        or len(set(static_pcs)) != len(static_pcs)
        or static_pcs != list(range(0, len(records) * 16, 16))
        or descriptor_pcs != static_pcs
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("op_code"), str)
            or not row["op_code"]
            or row.get("encoded_instruction") != []
            for row in descriptors
        )
    ):
        raise ChunkRefusal(
            "sol_control_static_descriptor_binding_invalid",
            repr({
                "static_count": len(static_pcs),
                "descriptor_count": len(descriptor_pcs),
                "static_prefix": static_pcs[:32],
                "descriptor_prefix": descriptor_pcs[:32],
            }),
        )
    missing_dynamic_pcs = sorted(dynamic_pcs - set(static_pcs))
    if missing_dynamic_pcs:
        raise ChunkRefusal(
            "sol_control_dynamic_pc_unbound", repr(missing_dynamic_pcs[:32])
        )
    report = label.get("ncu_report")
    report_stat = (
        _output_lstat(
            report, code="sol_control_storage_failure",
            operation="sol_control_ncu_report_stat",
        ) if isinstance(report, Path) else None
    )
    if (
        not isinstance(report, Path) or report_stat is None
        or not stat.S_ISREG(report_stat.st_mode)
    ):
        raise ChunkRefusal("sol_control_ncu_report_missing", repr(report))
    report_bytes = _read_output_bytes(
        report, code="sol_control_storage_failure",
        operation="sol_control_ncu_report_read",
    )
    function_bytes = function_name.encode("ascii")
    raw_occurrences = (
        report_bytes.count(raw_text) if isinstance(raw_text, bytes) else None
    )
    function_occurrences = report_bytes.count(function_bytes)
    if (
        contract.get("ncu_report_binding") != "raw_text_section_exactly_once"
        or not isinstance(raw_text, bytes)
        or raw_occurrences != 1
        or function_occurrences <= 0
    ):
        raise ChunkRefusal(
            "sol_control_ncu_report_binding_mismatch",
            repr({
                "raw_occurrences": raw_occurrences,
                "function_occurrences": function_occurrences,
            }),
        )
    instructions: list[dict] = []
    for descriptor, raw_record in zip(descriptors, records, strict=True):
        decoded = raw_record["control"]
        row = copy.deepcopy(descriptor)
        row["encoded_instruction"] = [
            raw_record["low_word"], raw_record["high_word"],
        ]
        row["control_bits"] = {
            "stall_count": decoded["stall"],
            "is_yield": bool(decoded["yield_flag"]),
            "is_new_read_barrier": decoded["read_barrier"] != 7,
            "is_new_write_barrier": decoded["write_barrier"] != 7,
            "id_new_read_barrier": decoded["read_barrier"],
            "id_new_write_barrier": decoded["write_barrier"],
            "wait_barrier_bits": decoded["wait_mask"],
            "reuse_flags": decoded["reuse"],
        }
        instructions.append(row)
    if not all(_runtime_control_instruction_valid(row) for row in instructions):
        raise ChunkRefusal("sol_control_static_instruction_invalid", function_name)
    raw_destination = evidence_dir / extracted["section_name"]
    metadata = {
        "policy": contract["policy"],
        "function_name": function_name,
        "execution_arch": "sm_86",
        "code_arch": extracted["code_arch"],
        "toolkit_version": extracted["toolkit_version"],
        "nvcc_sha256": contract["nvcc_sha256"],
        "cuobjdump_sha256": contract["cuobjdump_sha256"],
        "text_section_index": extracted["section_index"],
        "text_section_name": extracted["section_name"],
        "text_section_identity_method": section_identity_method,
        "cubin_symbol_proof": copy.deepcopy(cubin_symbol_proof),
        "raw_text_path": raw_destination.relative_to(
            evidence_dir.parent
        ).as_posix(),
        "raw_text_size_bytes": len(raw_text),
        "raw_text_sha256": hashlib.sha256(raw_text).hexdigest(),
        "static_instruction_count": len(records),
        "static_pcs_sha256": canonical_sha256(static_pcs),
        "enhanced_descriptor_pcs_sha256": canonical_sha256(descriptor_pcs),
        "ncu_report_sha256": _sha256_output_file(
            report, code="sol_control_storage_failure",
            operation="sol_control_ncu_report_hash",
        ),
        "ncu_report_raw_text_occurrences": raw_occurrences,
        "ncu_report_function_occurrences": function_occurrences,
        "parser_contract": extracted["parser_contract"],
    }
    _write_control_evidence_bytes(raw_text, raw_destination)
    return instructions, metadata


def capture_vendor_control(
    *, cell: dict, invocation: dict, label: dict, trace: dict,
    cell_dir: Path, module_store: Path,
) -> dict:
    """Bind a direct cuBLAS launch to one exact runtime-module SASS section."""
    contract = cell.get("control_bits") or {}
    if contract.get("policy") != "post_trace_vendor_module":
        raise ChunkRefusal("vendor_control_policy_invalid", repr(contract))
    trace_dir = trace.get("control_trace_dir")
    trace_dir_stat = (
        _output_lstat(
            trace_dir, code="sol_control_storage_failure",
            operation="vendor_control_trace_directory_stat",
        ) if isinstance(trace_dir, Path) else None
    )
    if (
        not isinstance(trace_dir, Path) or trace_dir_stat is None
        or not stat.S_ISDIR(trace_dir_stat.st_mode)
    ):
        raise ChunkRefusal("vendor_control_trace_dir_missing", repr(trace_dir))
    if trace.get("launch_count") != 1 or len(trace.get("launches") or []) != 1:
        raise ChunkRefusal("vendor_control_launch_cardinality_invalid", repr(trace))
    root_launch = trace["launches"][0]
    if root_launch.get("kernel_name") != contract.get("runtime_kernel"):
        raise ChunkRefusal(
            "vendor_control_runtime_kernel_mismatch", repr(root_launch)
        )

    inventory_source = trace_dir / "extra_info" / "traced_kernel_modules.json"
    inventory_stat = _output_lstat(
        inventory_source, code="sol_control_storage_failure",
        operation="vendor_control_inventory_stat",
    )
    if (
        inventory_stat is None or not stat.S_ISREG(inventory_stat.st_mode)
        or inventory_stat.st_size <= 0
    ):
        raise ChunkRefusal(
            "vendor_control_module_inventory_missing", str(inventory_source)
        )
    inventory = _json_output(
        inventory_source, code="sol_control_storage_failure",
        operation="vendor_control_inventory_read",
    )
    selection = inventory.get("selection") or {}
    expected_module_identity = {
        "stable_key": "sha256",
        "path_semantics": "path_at_capture",
        "rebind_contract": "sha256_required_basename_hint",
    }
    expected_selection = {
        "active_from_start": True,
        "max_accepted_active_launches": 0,
        "accepted_active_launches": 1,
        "active_launch_attempts": 1,
        "active_launches_skipped_after_limit": 0,
        "concurrent_limit_violations": 0,
        "exact_contract_satisfied": True,
        "module_mapping_count": 1,
        "inactive_channel_records_dropped": 0,
    }
    selection_mismatches = {
        key: {"expected": expected, "observed": selection.get(key)}
        for key, expected in expected_selection.items()
        if selection.get(key) != expected
    }
    if (
        inventory.get("schema_version") != 1
        or inventory.get("module_identity") != expected_module_identity
        or selection_mismatches
        or type(selection.get("accepted_channel_records")) is not int
        or selection.get("accepted_channel_records", 0) <= 0
    ):
        raise ChunkRefusal(
            "vendor_control_active_range_invalid",
            repr({"selection": selection, "mismatches": selection_mismatches}),
        )
    launches = inventory.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise ChunkRefusal("vendor_control_launch_mapping_invalid", repr(launches))
    mapping = launches[0]
    expected_mapping_fields = {
        "accepted_launch_index", "device_id", "stream_id", "trace_launch_id",
        "kernel_name", "resolution_source", "module",
    }
    if (
        not isinstance(mapping, dict)
        or set(mapping) != expected_mapping_fields
        or mapping.get("accepted_launch_index") != 0
        or mapping.get("device_id") != root_launch.get("device_id")
        or mapping.get("stream_id") != root_launch.get("stream_id")
        or mapping.get("trace_launch_id") != root_launch.get("kernel_id")
        or mapping.get("kernel_name") != contract.get("function_name")
        or not isinstance(mapping.get("resolution_source"), str)
        or not mapping.get("resolution_source")
    ):
        raise ChunkRefusal("vendor_control_launch_mapping_invalid", repr(mapping))
    module = mapping.get("module") or {}
    if set(module) != {"status", "reason", "path_at_capture", "basename", "sha256"}:
        raise ChunkRefusal("vendor_control_module_identity_invalid", repr(module))
    if (
        module.get("status") != contract.get("expected_inventory_status")
        or module.get("reason") != contract.get("expected_inventory_reason")
    ):
        raise ChunkRefusal("vendor_control_inventory_outcome_mismatch", repr(module))
    if contract.get("inventory_binding") == "nvbit_stable_static_module":
        expected_inventory_module = {
            "status": "stable_static_module",
            "reason": "resolved_file",
            "path_at_capture": contract.get("module_path_at_capture"),
            "basename": contract.get("module_basename"),
            "sha256": contract.get("module_sha256"),
        }
        if module != expected_inventory_module:
            raise ChunkRefusal("vendor_control_module_identity_invalid", repr(module))
    elif contract.get("inventory_binding") == "ncu_report_exact_text_section":
        if any(module.get(key) not in {"", None} for key in (
            "path_at_capture", "basename", "sha256",
        )):
            raise ChunkRefusal("vendor_control_unresolved_module_not_empty", repr(module))
    else:
        raise ChunkRefusal(
            "vendor_control_inventory_binding_invalid",
            repr(contract.get("inventory_binding")),
        )

    evidence_dir = cell_dir / "control_evidence"
    inventory_copy = _copy_control_evidence(
        inventory_source, evidence_dir / "traced_kernel_modules.json"
    )
    matrix_module = {
        "status": "stable_static_module",
        "reason": "resolved_file",
        "path_at_capture": contract.get("module_path_at_capture"),
        "basename": contract.get("module_basename"),
        "sha256": contract.get("module_sha256"),
    }
    preserved_module, preserved_module_rel = _preserve_sol_module(
        matrix_module, cell_dir=cell_dir, module_store=module_store,
    )
    nvcc = Path(str(contract.get("nvcc_path")))
    cuobjdump = Path(str(contract.get("cuobjdump_path")))
    _require_hash(nvcc, contract.get("nvcc_sha256"), "pinned control nvcc")
    _require_hash(
        cuobjdump, contract.get("cuobjdump_sha256"), "pinned control cuobjdump"
    )
    try:
        extracted = sass_extractor.extract_exact_function_text(
            preserved_module,
            function_name=contract["function_name"],
            allowed_code_arches=(contract["code_arch"],),
            nvcc=str(nvcc), cuobjdump=str(cuobjdump),
        )
    except (sass_extractor.FunctionTextExtractionError, RuntimeError) as exc:
        raise ChunkRefusal("vendor_control_static_extraction_failed", str(exc)) from exc
    extraction_checks = {
        "code_arch": contract.get("code_arch"),
        "section_index": contract.get("text_section_index"),
        "section_name": contract.get("text_section_name"),
        "raw_text_size_bytes": contract.get("raw_text_size_bytes"),
        "raw_text_sha256": contract.get("raw_text_sha256"),
        "toolkit_version": contract.get("toolkit_version"),
    }
    extraction_mismatches = {
        key: {"expected": expected, "observed": extracted.get(key)}
        for key, expected in extraction_checks.items()
        if extracted.get(key) != expected
    }
    records = extracted.get("records")
    if (
        extraction_mismatches
        or not isinstance(records, list)
        or len(records) != contract.get("static_instruction_count")
        or any(row.get("status") != "decoded" for row in records)
    ):
        raise ChunkRefusal(
            "vendor_control_static_contract_mismatch",
            repr({"mismatches": extraction_mismatches, "records": len(records or [])}),
        )
    static_pcs = [row.get("pc") for row in records]
    if (
        any(type(pc) is not int or pc < 0 for pc in static_pcs)
        or len(set(static_pcs)) != len(static_pcs)
        or static_pcs != list(range(0, len(records) * 16, 16))
        or canonical_sha256(static_pcs) != contract.get("static_pcs_sha256")
    ):
        raise ChunkRefusal("vendor_control_static_pc_invalid", repr(static_pcs[:32]))

    report = label.get("ncu_report")
    report_stat = (
        _output_lstat(
            report, code="sol_control_storage_failure",
            operation="vendor_control_ncu_report_stat",
        ) if isinstance(report, Path) else None
    )
    if (
        not isinstance(report, Path) or report_stat is None
        or not stat.S_ISREG(report_stat.st_mode)
    ):
        raise ChunkRefusal("vendor_control_ncu_report_missing", repr(report))
    report_bytes = _read_output_bytes(
        report, code="sol_control_storage_failure",
        operation="vendor_control_ncu_report_read",
    )
    raw_text = extracted.get("raw_text_bytes")
    if (
        contract.get("ncu_report_binding") != "raw_text_section_exactly_once"
        or not isinstance(raw_text, bytes)
        or report_bytes.count(raw_text) != 1
        or report_bytes.count(contract["function_name"].encode("ascii")) <= 0
    ):
        raise ChunkRefusal(
            "vendor_control_ncu_report_binding_mismatch",
            repr({
                "raw_occurrences": (
                    report_bytes.count(raw_text) if isinstance(raw_text, bytes) else None
                ),
                "function_occurrences": report_bytes.count(
                    contract["function_name"].encode("ascii")
                ),
            }),
        )

    raw_copy = _write_control_evidence_bytes(
        raw_text, evidence_dir / contract["text_section_name"]
    )
    enhanced_source = trace_dir / "extra_info" / "enhanced_execution_info.json"
    enhanced_stat = _output_lstat(
        enhanced_source, code="sol_control_storage_failure",
        operation="vendor_control_enhanced_inventory_stat",
    )
    if (
        enhanced_stat is None or not stat.S_ISREG(enhanced_stat.st_mode)
        or enhanced_stat.st_size <= 0
    ):
        raise ChunkRefusal("vendor_control_enhanced_info_missing", str(enhanced_source))
    enhanced_copy = _copy_control_evidence(
        enhanced_source, evidence_dir / "enhanced_execution_info.json"
    )
    document = _json_output(
        enhanced_copy, code="sol_control_storage_failure",
        operation="vendor_control_enhanced_copy_read",
    )
    kernels = document.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ChunkRefusal("vendor_control_kernel_inventory_invalid", repr(kernels))
    kernel = kernels[0]
    descriptors = kernel.get("instructions") if isinstance(kernel, dict) else None
    if (
        not isinstance(kernel, dict)
        or kernel.get("kernel_name") != contract.get("runtime_kernel")
        or kernel.get("architecture_version") != 86
        or kernel.get("is_captured_from_binary") is not False
        or type(kernel.get("unique_function_id")) is not int
        or kernel["unique_function_id"] <= 0
        or type(kernel.get("function_address")) is not int
        or kernel["function_address"] <= 0
        or not isinstance(descriptors, list)
        or len(descriptors) != len(records)
    ):
        raise ChunkRefusal("vendor_control_kernel_identity_invalid", repr(kernel))
    descriptor_pcs = [
        row.get("pc_num_dec") if isinstance(row, dict) else None
        for row in descriptors
    ]
    if (
        descriptor_pcs != static_pcs
        or any(
            not isinstance(row.get("op_code"), str)
            or not row["op_code"]
            or row.get("encoded_instruction") != []
            for row in descriptors
        )
    ):
        raise ChunkRefusal(
            "vendor_control_descriptor_binding_invalid", repr(descriptor_pcs[:32])
        )

    instructions: list[dict] = []
    for descriptor, raw_record in zip(descriptors, records, strict=True):
        decoded = raw_record["control"]
        row = copy.deepcopy(descriptor)
        row["encoded_instruction"] = [
            raw_record["low_word"], raw_record["high_word"],
        ]
        row["control_bits"] = {
            "stall_count": decoded["stall"],
            "is_yield": bool(decoded["yield_flag"]),
            "is_new_read_barrier": decoded["read_barrier"] != 7,
            "is_new_write_barrier": decoded["write_barrier"] != 7,
            "id_new_read_barrier": decoded["read_barrier"],
            "id_new_write_barrier": decoded["write_barrier"],
            "wait_barrier_bits": decoded["wait_mask"],
            "reuse_flags": decoded["reuse"],
        }
        instructions.append(row)
    if not all(_runtime_control_instruction_valid(row) for row in instructions):
        raise ChunkRefusal("vendor_control_instruction_invalid", contract["function_name"])
    dynamic_pcs = _sol_dynamic_trace_pcs(trace_dir)
    missing_dynamic_pcs = sorted(dynamic_pcs - set(static_pcs))
    if missing_dynamic_pcs:
        raise ChunkRefusal(
            "vendor_control_dynamic_pc_unbound", repr(missing_dynamic_pcs[:32])
        )

    outcome = contract.get("expected_outcome")
    payload = {
        "schema_version": 1,
        "arch": contract["execution_arch"],
        "identity": cell["identity"],
        "policy": "post_trace_vendor_module",
        "outcome": outcome,
        "inventory_binding": contract["inventory_binding"],
        "authority_reference_path": invocation["authority_reference_path"],
        "authority_reference_sha256": invocation["authority_reference_sha256"],
        "control_contract_sha256": canonical_sha256(contract),
        "module_inventory_path": inventory_copy.relative_to(cell_dir).as_posix(),
        "module_inventory_sha256": _sha256_output_file(
            inventory_copy, code="sol_control_storage_failure",
            operation="vendor_control_inventory_copy_hash",
        ),
        "inventory_module_status": module.get("status"),
        "inventory_module_reason": module.get("reason"),
        "inventory_module_sha256": module.get("sha256") or None,
        "matrix_pinned_module_path_at_capture": contract["module_path_at_capture"],
        "matrix_pinned_module_sha256": contract["module_sha256"],
        "preserved_module_path": preserved_module_rel,
        "preserved_module_sha256": _sha256_output_file(
            preserved_module, code="sol_control_storage_failure",
            operation="vendor_control_preserved_module_hash",
        ),
        "runtime_inventory_path": enhanced_copy.relative_to(cell_dir).as_posix(),
        "runtime_inventory_sha256": _sha256_output_file(
            enhanced_copy, code="sol_control_storage_failure",
            operation="vendor_control_runtime_inventory_hash",
        ),
        "ncu_report_sha256": _sha256_output_file(
            report, code="sol_control_storage_failure",
            operation="vendor_control_ncu_report_hash",
        ),
        "ncu_report_raw_text_occurrences": 1,
        "raw_text_path": raw_copy.relative_to(cell_dir).as_posix(),
        "raw_text_sha256": _sha256_output_file(
            raw_copy, code="sol_control_storage_failure",
            operation="vendor_control_raw_text_hash",
        ),
        "raw_text_size_bytes": _output_lstat(
            raw_copy, code="sol_control_storage_failure",
            operation="vendor_control_raw_text_stat",
        ).st_size,
        "text_section_index": extracted["section_index"],
        "text_section_name": extracted["section_name"],
        "execution_arch": contract["execution_arch"],
        "code_arch": extracted["code_arch"],
        "toolkit_version": extracted["toolkit_version"],
        "cuobjdump_sha256": contract["cuobjdump_sha256"],
        "kernels": [{
            "kernel_name": contract["runtime_kernel"],
            "function_name": contract["function_name"],
            "architecture_version": 86,
            "execution_arch": contract["execution_arch"],
            "code_arch": extracted["code_arch"],
            "control_source": "raw_static_text_section",
            "unique_function_id": kernel["unique_function_id"],
            "function_address": kernel["function_address"],
            "instructions": instructions,
            "dynamic_trace_pc_count": len(dynamic_pcs),
            "dynamic_trace_pcs_sha256": canonical_sha256(sorted(dynamic_pcs)),
            "control_pcs_sha256": canonical_sha256(static_pcs),
            "summary": {
                "instructions": len(instructions),
                "decoded": len(instructions),
                "unknown_encoding": 0,
            },
        }],
        "parser_contract": extracted["parser_contract"],
        "summary": {
            "kernel_count": 1,
            "total_decoded": len(instructions),
            "total_unknown_encoding": 0,
        },
    }
    out = cell_dir / "control_bits.json"
    _write_output_text(
        out, json.dumps(payload, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="vendor_control_bits_write",
    )
    wrapper = {
        "schema_version": 1,
        "identity": cell["identity"],
        "reference_sha256": invocation["authority_reference_sha256"],
        "control_bits_sha256": _sha256_output_file(
            out, code="sol_control_storage_failure",
            operation="vendor_control_bits_hash",
        ),
        "module_inventory_sha256": _sha256_output_file(
            inventory_copy, code="sol_control_storage_failure",
            operation="vendor_control_manifest_inventory_hash",
        ),
        "loaded_module_sha256": contract["module_sha256"],
        "preserved_module_path": preserved_module_rel,
        "policy": "post_trace_vendor_module",
        "outcome": outcome,
        "typed_reason": None,
        "total_decoded": len(instructions),
        "total_unknown_encoding": 0,
    }
    wrapper_path = cell_dir / "control_capture_manifest.json"
    _write_output_text(
        wrapper_path, json.dumps(wrapper, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="vendor_control_manifest_write",
    )
    return {
        "control_bits": out,
        "manifest": wrapper_path,
        "stdout_log": trace["stdout_log"],
        "stderr_log": trace["stderr_log"],
        **wrapper,
    }


def capture_sol_control(
    *, cell: dict, invocation: dict, label: dict, trace: dict, cell_dir: Path,
    module_store: Path,
) -> dict:
    """Seal controls for the exact accepted launch or its producer-typed absence."""
    trace_dir = trace.get("control_trace_dir")
    if not isinstance(trace_dir, Path):
        raise ChunkRefusal("sol_control_trace_dir_missing", repr(trace_dir))
    trace_dir_stat = _output_lstat(
        trace_dir, code="sol_control_storage_failure",
        operation="sol_control_trace_directory_stat",
    )
    if trace_dir_stat is None or not stat.S_ISDIR(trace_dir_stat.st_mode):
        raise ChunkRefusal("sol_control_trace_dir_missing", str(trace_dir))
    if trace.get("launch_count") != 1 or len(trace.get("launches") or []) != 1:
        raise ChunkRefusal("sol_control_launch_cardinality_invalid", repr(trace))
    root_launch = trace["launches"][0]
    inventory_source = trace_dir / "extra_info" / "traced_kernel_modules.json"
    inventory_stat = _output_lstat(
        inventory_source, code="sol_control_storage_failure",
        operation="sol_control_inventory_stat",
    )
    if (
        inventory_stat is None or not stat.S_ISREG(inventory_stat.st_mode)
        or inventory_stat.st_size <= 0
    ):
        raise ChunkRefusal("sol_control_module_inventory_missing", str(inventory_source))
    inventory = _json_output(
        inventory_source, code="sol_control_storage_failure",
        operation="sol_control_inventory_read",
    )
    expected_module_identity = {
        "stable_key": "sha256",
        "path_semantics": "path_at_capture",
        "rebind_contract": "sha256_required_basename_hint",
    }
    selection = inventory.get("selection") or {}
    launches = inventory.get("launches") or []
    selection_ints = (
        "max_accepted_active_launches", "accepted_active_launches",
        "active_launch_attempts", "active_launches_skipped_after_limit",
        "concurrent_limit_violations", "module_mapping_count",
        "inactive_channel_records_dropped", "accepted_channel_records",
    )
    if (
        inventory.get("schema_version") != 1
        or inventory.get("module_identity") != expected_module_identity
        or not all(type(selection.get(key)) is int for key in selection_ints)
        or selection.get("active_from_start") is not False
        or selection.get("max_accepted_active_launches") != 1
        or selection.get("accepted_active_launches") != 1
        or selection.get("module_mapping_count") != 1
        or selection.get("inactive_channel_records_dropped") != 0
        or selection.get("accepted_channel_records", 0) <= 0
        or selection.get("concurrent_limit_violations") != 0
        or selection.get("exact_contract_satisfied") is not True
        or selection.get("active_launch_attempts", 0) < 1
        or selection.get("active_launches_skipped_after_limit")
        != selection.get("active_launch_attempts") - 1
        or not isinstance(launches, list)
        or len(launches) != 1
    ):
        raise ChunkRefusal("sol_control_active_range_invalid", repr(inventory))
    mapping = launches[0]
    expected_mapping_fields = {
        "accepted_launch_index", "device_id", "stream_id", "trace_launch_id",
        "kernel_name", "resolution_source", "module",
    }
    if (
        not isinstance(mapping, dict)
        or set(mapping) != expected_mapping_fields
        or mapping.get("accepted_launch_index") != 0
        or mapping.get("device_id") != root_launch.get("device_id")
        or mapping.get("stream_id") != root_launch.get("stream_id")
        or mapping.get("trace_launch_id") != root_launch.get("kernel_id")
        or not isinstance(mapping.get("kernel_name"), str)
        or not mapping["kernel_name"]
        or root_launch.get("kernel_name") != mapping["kernel_name"] + "___0"
        or not isinstance(mapping.get("resolution_source"), str)
        or not mapping["resolution_source"]
    ):
        raise ChunkRefusal("sol_control_launch_mapping_invalid", repr(mapping))
    module = mapping.get("module") or {}
    if set(module) != {"status", "reason", "path_at_capture", "basename", "sha256"}:
        raise ChunkRefusal("sol_control_module_identity_invalid", repr(module))
    evidence_dir = cell_dir / "control_evidence"
    inventory_copy = _copy_control_evidence(
        inventory_source, evidence_dir / "traced_kernel_modules.json"
    )
    out = cell_dir / "control_bits.json"
    outcome = SOL_CONTROL_OUTCOMES[0]
    typed_reason = None
    kernels: list[dict] = []
    decoded = unknown = 0
    source_copy: Path | None = None
    source_sha: str | None = None
    preserved_module: Path | None = None
    preserved_module_rel: str | None = None
    control_source: str | None = None
    static_rebind: dict | None = None

    if module.get("status") == "no_stable_static_module":
        allowed_reasons = {
            "origin_unresolved", "jit_or_anonymous_mapping",
            "module_path_deleted", "module_path_missing",
            "module_not_regular_file", "module_hash_failed",
        }
        if (
            module.get("reason") not in allowed_reasons
            or module.get("sha256") not in {"", None}
        ):
            raise ChunkRefusal("sol_control_typed_module_invalid", repr(module))
        outcome = SOL_CONTROL_OUTCOMES[2]
        typed_reason = module["reason"]
    elif module.get("status") == "stable_static_module":
        if module.get("reason") != "resolved_file":
            raise ChunkRefusal("sol_control_module_identity_invalid", repr(module))
        preserved_module, preserved_module_rel = _preserve_sol_module(
            module, cell_dir=cell_dir, module_store=module_store,
        )
        source = trace_dir / "extra_info" / "enhanced_execution_info.json"
        source_stat = _output_lstat(
            source, code="sol_control_storage_failure",
            operation="sol_control_enhanced_inventory_stat",
        )
        if (
            source_stat is None or not stat.S_ISREG(source_stat.st_mode)
            or source_stat.st_size <= 0
        ):
            raise ChunkRefusal("sol_control_enhanced_info_missing", str(source))
        source_copy = _copy_control_evidence(
            source, evidence_dir / "enhanced_execution_info.json"
        )
        source_sha = _sha256_output_file(
            source_copy, code="sol_control_storage_failure",
            operation="sol_control_enhanced_copy_hash",
        )
        document = _json_output(
            source_copy, code="sol_control_storage_failure",
            operation="sol_control_enhanced_copy_read",
        )
        rows = document.get("kernels")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ChunkRefusal("sol_control_kernel_inventory_invalid", repr(rows))
        kernel = rows[0]
        descriptors = kernel.get("instructions") if isinstance(kernel, dict) else None
        if (
            not isinstance(kernel, dict)
            or kernel.get("kernel_name") != root_launch.get("kernel_name")
            or kernel.get("architecture_version") != 86
            or type(kernel.get("is_captured_from_binary")) is not bool
            or type(kernel.get("unique_function_id")) is not int
            or kernel["unique_function_id"] <= 0
            or type(kernel.get("function_address")) is not int
            or kernel["function_address"] <= 0
            or not isinstance(descriptors, list)
            or not descriptors
        ):
            raise ChunkRefusal("sol_control_kernel_identity_invalid", repr(kernel))
        if kernel["is_captured_from_binary"]:
            instructions = descriptors
            control_source = "enhanced_runtime_binary"
        else:
            dynamic_pcs = _sol_dynamic_trace_pcs(trace_dir)
            instructions, static_rebind = _sol_static_rebind_controls(
                contract=(cell.get("control_bits") or {}).get("static_rebind") or {},
                preserved_module=preserved_module,
                function_name=mapping["kernel_name"],
                descriptors=descriptors,
                dynamic_pcs=dynamic_pcs,
                label=label,
                evidence_dir=evidence_dir,
            )
            control_source = "raw_static_text_section_runtime_rebind"
            outcome = SOL_CONTROL_OUTCOMES[1]
        if not all(_runtime_control_instruction_valid(row) for row in instructions):
            raise ChunkRefusal("sol_control_instruction_invalid", repr(kernel["kernel_name"]))
        control_pcs = [row["pc_num_dec"] for row in instructions]
        if len(set(control_pcs)) != len(control_pcs):
            raise ChunkRefusal("sol_control_pc_duplicate", repr(kernel["kernel_name"]))
        if kernel["is_captured_from_binary"]:
            dynamic_pcs = _sol_dynamic_trace_pcs(trace_dir)
        missing_dynamic_pcs = sorted(dynamic_pcs - set(control_pcs))
        if missing_dynamic_pcs:
            raise ChunkRefusal(
                "sol_control_dynamic_pc_unbound", repr(missing_dynamic_pcs[:32])
            )
        decoded = len(instructions)
        kernels = [{
            "kernel_name": kernel["kernel_name"],
            "architecture_version": kernel["architecture_version"],
            "is_captured_from_binary": kernel["is_captured_from_binary"],
            "control_source": control_source,
            "unique_function_id": kernel["unique_function_id"],
            "function_address": kernel["function_address"],
            "instructions": instructions,
            "dynamic_trace_pc_count": len(dynamic_pcs),
            "dynamic_trace_pcs_sha256": canonical_sha256(sorted(dynamic_pcs)),
            "control_pcs_sha256": canonical_sha256(sorted(control_pcs)),
            "summary": {
                "instructions": decoded,
                "decoded": decoded,
                "unknown_encoding": 0,
            },
        }]
    else:
        raise ChunkRefusal("sol_control_module_status_invalid", repr(module))

    payload = {
        "schema_version": 1,
        "arch": "sm_86",
        "identity": cell["identity"],
        "policy": "post_trace_module_or_typed",
        "outcome": outcome,
        "typed_reason": typed_reason,
        "authority_reference_path": invocation["authority_reference_path"],
        "authority_reference_sha256": invocation["authority_reference_sha256"],
        "module_inventory_path": inventory_copy.relative_to(cell_dir).as_posix(),
        "module_inventory_sha256": _sha256_output_file(
            inventory_copy, code="sol_control_storage_failure",
            operation="sol_control_inventory_copy_hash",
        ),
        "runtime_inventory_path": (
            source_copy.relative_to(cell_dir).as_posix()
            if source_copy is not None else None
        ),
        "runtime_inventory_sha256": source_sha,
        "loaded_module_path_at_capture": module.get("path_at_capture"),
        "loaded_module_sha256": module.get("sha256") or None,
        "preserved_module_path": preserved_module_rel,
        "preserved_module_sha256": (
            _sha256_output_file(
                preserved_module, code="sol_control_storage_failure",
                operation="sol_control_preserved_module_hash",
            ) if preserved_module is not None else None
        ),
        "static_rebind": static_rebind,
        "kernels": kernels,
        "summary": {
            "kernel_count": len(kernels),
            "total_decoded": decoded,
            "total_unknown_encoding": unknown,
        },
    }
    _write_output_text(
        out, json.dumps(payload, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_control_bits_write",
    )
    wrapper = {
        "schema_version": 1,
        "identity": cell["identity"],
        "reference_sha256": invocation["authority_reference_sha256"],
        "control_bits_sha256": _sha256_output_file(
            out, code="sol_control_storage_failure",
            operation="sol_control_bits_hash",
        ),
        "module_inventory_sha256": _sha256_output_file(
            inventory_copy, code="sol_control_storage_failure",
            operation="sol_control_manifest_inventory_hash",
        ),
        "loaded_module_sha256": module.get("sha256") or None,
        "preserved_module_path": preserved_module_rel,
        "policy": "post_trace_module_or_typed",
        "outcome": outcome,
        "typed_reason": typed_reason,
        "total_decoded": decoded,
        "total_unknown_encoding": unknown,
    }
    wrapper_path = cell_dir / "control_capture_manifest.json"
    _write_output_text(
        wrapper_path, json.dumps(wrapper, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_control_manifest_write",
    )
    return {
        "control_bits": out,
        "manifest": wrapper_path,
        "stdout_log": trace["stdout_log"],
        "stderr_log": trace["stderr_log"],
        **wrapper,
    }


def _materialize_sol_launch_control_view(
    *, trace: dict[str, Any], inventory_row: dict[str, Any],
    launch_cell_dir: Path,
) -> dict[str, Any]:
    """Create a hash-bound one-launch view for the existing control decoder.

    The view contains hard links only to the selected launch's protobufs and
    derived JSON that names its exact producer inventory/source hashes.  The
    original all-launch trace remains the authoritative inventory.
    """
    trace_dir = trace.get("control_trace_dir")
    if not isinstance(trace_dir, Path) or trace_dir.is_symlink() or not trace_dir.is_dir():
        raise ChunkRefusal("sol_control_trace_dir_missing", repr(trace_dir))
    ordinal = inventory_row.get("launch_ordinal")
    root_launch = inventory_row.get("trace_launch")
    mapping = inventory_row.get("module_mapping")
    if type(ordinal) is not int or not isinstance(root_launch, dict) \
            or not isinstance(mapping, dict):
        raise ChunkRefusal("sol_dynamic_launch_inventory_invalid", repr(inventory_row))
    view = launch_cell_dir / "control_trace_view"
    launch_payload = (
        trace_dir / "threadblocks"
        / f"device_{root_launch['device_id']}"
        / f"stream_{root_launch['stream_id']}"
        / f"kernel_{root_launch['kernel_id']}"
    )
    launch_payload_stat = _output_lstat(
        launch_payload, code="sol_control_storage_failure",
        operation="sol_control_launch_payload_stat",
    )
    payloads = _output_tree_paths(
        launch_payload, code="sol_control_storage_failure",
        operation="sol_control_launch_payload_walk", pattern="*.pb",
    ) if (
        launch_payload_stat is not None
        and stat.S_ISDIR(launch_payload_stat.st_mode)
    ) else []
    if (
        launch_payload_stat is None
        or not stat.S_ISDIR(launch_payload_stat.st_mode) or not payloads
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_payload_missing", repr(root_launch)
        )
    view_payload = (
        view / "threadblocks"
        / f"device_{root_launch['device_id']}"
        / f"stream_{root_launch['stream_id']}"
        / f"kernel_{root_launch['kernel_id']}"
    )
    _mkdir_output(
        view_payload, parents=True, exist_ok=False,
        code="sol_control_storage_failure",
        operation="sol_control_view_payload_create",
    )
    for source in payloads:
        source_stat = _output_lstat(
            source, code="sol_control_storage_failure",
            operation="sol_control_view_payload_source_stat",
        )
        if source_stat is None or not stat.S_ISREG(source_stat.st_mode):
            raise ChunkRefusal("sol_control_dynamic_trace_invalid", str(source))
        _link_output(
            source, view_payload / source.name,
            code="sol_control_storage_failure",
            operation="sol_control_view_payload_link",
        )
    extra = view / "extra_info"
    _mkdir_output(
        extra, parents=True, exist_ok=False,
        code="sol_control_storage_failure",
        operation="sol_control_view_extra_create",
    )
    source_inventory = trace_dir / "extra_info" / "traced_kernel_modules.json"
    source_enhanced = trace_dir / "extra_info" / "enhanced_execution_info.json"
    inventory = _json_output(
        source_inventory, code="sol_control_storage_failure",
        operation="sol_control_view_source_inventory_read",
    )
    enhanced = _json_output(
        source_enhanced, code="sol_control_storage_failure",
        operation="sol_control_view_source_enhanced_read",
    )
    kernels = enhanced.get("kernels")
    matching = [
        row for row in (kernels or [])
        if isinstance(row, dict)
        and row.get("kernel_name") == root_launch.get("kernel_name")
    ]
    if len(matching) != 1:
        raise ChunkRefusal(
            "sol_control_kernel_inventory_invalid",
            repr({"launch": root_launch, "matches": len(matching)}),
        )
    one_inventory = {
        "schema_version": 1,
        "module_identity": copy.deepcopy(inventory.get("module_identity")),
        "selection": {
            "active_from_start": False,
            "max_accepted_active_launches": 1,
            "accepted_active_launches": 1,
            "active_launch_attempts": 1,
            "active_launches_skipped_after_limit": 0,
            "concurrent_limit_violations": 0,
            "exact_contract_satisfied": True,
            "module_mapping_count": 1,
            "inactive_channel_records_dropped": 0,
            "accepted_channel_records": max(
                1, int((inventory.get("selection") or {}).get(
                    "accepted_channel_records", 1
                )),
            ),
        },
        "launches": [{**copy.deepcopy(mapping), "accepted_launch_index": 0}],
    }
    _write_output_text(
        extra / "traced_kernel_modules.json",
        json.dumps(one_inventory, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_control_view_inventory_write",
    )
    one_enhanced = copy.deepcopy(enhanced)
    one_enhanced["kernels"] = matching
    _write_output_text(
        extra / "enhanced_execution_info.json",
        json.dumps(one_enhanced, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_control_view_enhanced_write",
    )
    derivation = {
        "schema_version": 1,
        "kind": "sol_launch_control_view",
        "launch_ordinal": ordinal,
        "launch_identity_sha256": inventory_row["launch_identity_sha256"],
        "source_inventory_sha256": _sha256_output_file(
            source_inventory, code="sol_control_storage_failure",
            operation="sol_control_view_source_inventory_hash",
        ),
        "source_enhanced_inventory_sha256": _sha256_output_file(
            source_enhanced, code="sol_control_storage_failure",
            operation="sol_control_view_source_enhanced_hash",
        ),
        "selected_payload_count": len(payloads),
        "selected_payload_sha256": canonical_sha256([
            {
                "name": path.name,
                "sha256": _sha256_output_file(
                    path, code="sol_control_storage_failure",
                    operation="sol_control_view_payload_hash",
                ),
                "size_bytes": _output_lstat(
                    path, code="sol_control_storage_failure",
                    operation="sol_control_view_payload_stat",
                ).st_size,
            }
            for path in payloads
        ]),
    }
    derivation["derivation_sha256"] = canonical_sha256(derivation)
    derivation_path = launch_cell_dir / "control_view_derivation.json"
    _write_output_text(
        derivation_path, json.dumps(derivation, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_control_view_derivation_write",
    )
    return {
        "launch_count": 1,
        "launches": [copy.deepcopy(root_launch)],
        "control_trace_dir": view,
        "stdout_log": trace["stdout_log"],
        "stderr_log": trace["stderr_log"],
        "derivation": derivation_path,
    }


def capture_sol_controls_all_launches(
    *, cell: dict[str, Any], invocation: dict[str, Any],
    labels_by_ordinal: dict[int, dict[str, Any]], trace: dict[str, Any],
    cell_dir: Path, module_store: Path,
) -> dict[str, Any]:
    inventory = trace.get("dynamic_launch_inventory")
    inventory_sha = trace.get("dynamic_launch_inventory_sha256")
    if (
        not isinstance(inventory, list) or not inventory
        or inventory_sha != canonical_sha256(inventory)
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_inventory_invalid", repr(inventory_sha)
        )
    launch_records: list[dict[str, Any]] = []
    total_decoded = 0
    complete_controls = 0
    for row in inventory:
        ordinal = row["launch_ordinal"]
        label = labels_by_ordinal.get(ordinal)
        if label is None:
            launch_records.append({
                "launch_ordinal": ordinal,
                "launch_identity_sha256": row["launch_identity_sha256"],
                "status": "typed_incomplete",
                "typed_reason": "ncu_label_incomplete",
                "control_outcome": None,
                "control_bits_sha256": None,
                "control_manifest_sha256": None,
            })
            continue
        if label.get("launch_index") != ordinal:
            raise ChunkRefusal(
                "sol_dynamic_launch_label_join_invalid", repr(ordinal)
            )
        ncu_identity = label.get("ncu_launch_identity") or {}
        if (
            ncu_identity.get("launch_ordinal") != ordinal
            or ncu_identity.get("launch_identity_sha256")
            != row["launch_identity_sha256"]
            or ncu_identity.get("kernel_name")
            != row["module_mapping"]["kernel_name"]
            or ncu_identity.get("ncu_report_sha256")
            != _sha256_output_file(
                label["ncu_report"], code="sol_control_storage_failure",
                operation="sol_all_launch_ncu_report_hash",
            )
        ):
            raise ChunkRefusal(
                "sol_dynamic_launch_ncu_identity_mismatch", repr(ordinal)
            )
        expected_function = row["module_mapping"]["kernel_name"]
        report = label.get("ncu_report")
        report_stat = (
            _output_lstat(
                report, code="sol_control_storage_failure",
                operation="sol_all_launch_ncu_report_stat",
            ) if isinstance(report, Path) else None
        )
        if (
            not isinstance(report, Path) or report_stat is None
            or not stat.S_ISREG(report_stat.st_mode)
        ):
            raise ChunkRefusal(
                "sol_control_ncu_report_missing", repr(report)
            )
        try:
            report_bytes = _read_output_bytes(
                report, code="sol_control_storage_failure",
                operation="sol_all_launch_ncu_report_read",
            )
            expected_function_bytes = expected_function.encode("ascii")
        except (OSError, UnicodeEncodeError) as exc:
            raise ChunkRefusal(
                "sol_control_ncu_report_unreadable", str(exc)
            ) from exc
        if report_bytes.count(expected_function_bytes) <= 0:
            raise ChunkRefusal(
                "sol_dynamic_launch_ncu_identity_mismatch",
                repr({"ordinal": ordinal, "function": expected_function}),
            )
        launch_cell_dir = (
            cell_dir / "control_launches" / f"launch_{ordinal:04d}"
        )
        control_view = launch_cell_dir / "control_trace_view"
        control: dict[str, Any] | None = None
        control_error: BaseException | None = None
        try:
            _mkdir_output(
                launch_cell_dir, parents=True, exist_ok=False,
                code="sol_control_storage_failure",
                operation="sol_control_launch_directory_create",
            )
            view = _materialize_sol_launch_control_view(
                trace=trace, inventory_row=row, launch_cell_dir=launch_cell_dir,
            )
            if view.get("control_trace_dir") != control_view:
                raise ChunkRefusal(
                    "sol_launch_control_view_invalid", repr(view)
                )
            control = capture_sol_control(
                cell=cell, invocation=invocation, label=label, trace=view,
                cell_dir=launch_cell_dir, module_store=module_store,
            )
        except BaseException as exc:
            control_error = exc
            if _is_explicit_output_storage_refusal(exc):
                storage_origin = _storage_origin(exc)
                raise ChunkRefusal(
                    "sol_control_storage_failure",
                    f"errno={getattr(storage_origin, 'errno', None)}; "
                    f"launch={ordinal}; source_code={exc.code}",
                ) from (storage_origin or exc)
            raise
        finally:
            if _storage_origin(control_error) is None:
                if control_view.exists() or control_view.is_symlink():
                    if control_view.is_symlink() or not control_view.is_dir():
                        raise ChunkRefusal(
                            "sol_launch_control_view_cleanup_refused",
                            repr(control_view),
                        )
                    _rmtree_output(
                        control_view, code="sol_control_storage_failure",
                        operation="sol_launch_control_view_cleanup",
                    )
                    if control_view.exists() or control_view.is_symlink():
                        raise ChunkRefusal(
                            "sol_launch_control_view_cleanup_failed",
                            str(control_view),
                        )
        if control is None:
            raise ChunkRefusal(
                "sol_dynamic_launch_control_missing", repr(ordinal)
            )
        control_complete = control.get("outcome") in SOL_CONTROL_OUTCOMES[:2]
        if control_complete:
            complete_controls += 1
        total_decoded += int(control.get("total_decoded", 0))
        launch_records.append({
            "launch_ordinal": ordinal,
            "launch_identity_sha256": row["launch_identity_sha256"],
            "status": "complete" if control_complete else "typed_incomplete",
            "typed_reason": None if control_complete else (
                control.get("typed_reason") or "control_unavailable"
            ),
            "control_outcome": control.get("outcome"),
            "control_bits_sha256": _sha256_output_file(
                control["control_bits"], code="sol_control_storage_failure",
                operation="sol_all_launch_control_bits_hash",
            ),
            "control_manifest_sha256": _sha256_output_file(
                control["manifest"], code="sol_control_storage_failure",
                operation="sol_all_launch_control_manifest_hash",
            ),
        })
    overall_complete = complete_controls == len(inventory)
    payload = {
        "schema_version": 1,
        "arch": "sm_86",
        "identity": cell["identity"],
        "policy": "all_profiler_roi_launches_exact_inventory",
        "launch_inventory_sha256": inventory_sha,
        "launch_count": len(inventory),
        "complete_control_count": complete_controls,
        "status": (
            "complete_dynamic_launch_set" if overall_complete
            else "typed_incomplete_dynamic_launch_set"
        ),
        "launches": launch_records,
        "summary": {
            "kernel_count": len(inventory),
            "total_decoded": total_decoded,
            "total_unknown_encoding": 0,
        },
    }
    out = cell_dir / "control_bits.json"
    _write_output_text(
        out, json.dumps(payload, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_all_launch_control_bits_write",
    )
    wrapper = {
        "schema_version": 1,
        "identity": cell["identity"],
        "reference_sha256": invocation["authority_reference_sha256"],
        "control_bits_sha256": _sha256_output_file(
            out, code="sol_control_storage_failure",
            operation="sol_all_launch_aggregate_hash",
        ),
        "module_inventory_sha256": inventory_sha,
        "loaded_module_sha256": None,
        "preserved_module_path": None,
        "policy": "all_profiler_roi_launches_exact_inventory",
        "outcome": payload["status"],
        "typed_reason": (
            None if overall_complete else "control_launch_set_incomplete"
        ),
        "total_decoded": total_decoded,
        "total_unknown_encoding": 0,
        "launch_count": len(inventory),
        "complete_control_count": complete_controls,
    }
    wrapper_path = cell_dir / "control_capture_manifest.json"
    _write_output_text(
        wrapper_path, json.dumps(wrapper, indent=2, sort_keys=True) + "\n",
        code="sol_control_storage_failure",
        operation="sol_all_launch_control_manifest_write",
    )
    return {
        "control_bits": out,
        "manifest": wrapper_path,
        "stdout_log": trace["stdout_log"],
        "stderr_log": trace["stderr_log"],
        "launch_records": launch_records,
        **wrapper,
    }


def artifact_row(path: Path, *, root: Path, artifact_kind: str) -> dict:
    observed = _output_lstat(
        path, code="cell_inventory_storage_failure",
        operation="cell_artifact_stat",
    )
    if observed is None or not stat.S_ISREG(observed.st_mode):
        raise ChunkRefusal("artifact_not_regular", str(path))
    try:
        rel = _resolve_output_path(
            path, code="cell_inventory_storage_failure",
            operation="cell_artifact_resolve",
        ).relative_to(_resolve_output_path(
            root, code="cell_inventory_storage_failure",
            operation="chunk_root_resolve",
        ))
    except ValueError as exc:
        raise ChunkRefusal("artifact_path_escape", str(path)) from exc
    return {
        "relative_path": rel.as_posix(),
        "sha256": _sha256_output_file(
            path, code="cell_inventory_storage_failure",
            operation="cell_artifact_hash",
        ),
        "size_bytes": observed.st_size, "artifact_kind": artifact_kind,
    }


def _normalize_ncu_dim(
    value: Any, *, field: str,
    refusal_code: str = "sol_dynamic_launch_ncu_identity_mismatch",
) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ChunkRefusal(
            refusal_code,
            f"{field} is not a string: {value!r}",
        )
    match = re.fullmatch(
        r"\s*(?:\(\s*)?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
        r"\s*(?:\)\s*)?",
        value,
    )
    if match is None or (value.lstrip().startswith("(") != value.rstrip().endswith(")")):
        raise ChunkRefusal(
            refusal_code,
            f"invalid {field}: {value!r}",
        )
    result = tuple(int(item) for item in match.groups())
    if any(item <= 0 for item in result):
        raise ChunkRefusal(
            refusal_code,
            f"non-positive {field}: {value!r}",
        )
    return result  # type: ignore[return-value]


def _normalize_ncu_device(
    value: Any, *,
    refusal_code: str = "sol_dynamic_launch_ncu_identity_mismatch",
) -> int:
    if not isinstance(value, str):
        raise ChunkRefusal(
            refusal_code,
            f"Device is not a string: {value!r}",
        )
    stripped = value.strip()
    if stripped.isdecimal():
        return int(stripped)
    match = re.fullmatch(r".*\((\d+)\)\s*", stripped)
    if match is None:
        raise ChunkRefusal(
            refusal_code,
            f"invalid Device: {value!r}",
        )
    return int(match.group(1))


def _trace_ncu_launch_key(
    inventory_row: dict[str, Any],
) -> tuple[str, tuple[int, int, int], tuple[int, int, int], int]:
    mapping = inventory_row.get("module_mapping") or {}
    launch = inventory_row.get("trace_launch") or {}
    name = mapping.get("kernel_name")
    block = launch.get("block_dim")
    grid = launch.get("grid_dim")
    device = launch.get("device_id")
    if (
        not isinstance(name, str) or not name
        or not isinstance(block, list) or len(block) != 3
        or not isinstance(grid, list) or len(grid) != 3
        or any(type(item) is not int or item <= 0 for item in [*block, *grid])
        or type(device) is not int or device < 0
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_ncu_identity_mismatch",
            repr(inventory_row),
        )
    return name, tuple(block), tuple(grid), device  # type: ignore[return-value]


def _observation_ncu_launch_key(
    observation: dict[str, Any], *,
    refusal_code: str = "sol_dynamic_launch_ncu_identity_mismatch",
) -> tuple[str, tuple[int, int, int], tuple[int, int, int], int]:
    identity = observation.get("identity")
    required = {"ID", "Kernel Name", "Block Size", "Grid Size", "Device"}
    columns = observation.get("identity_columns")
    if not isinstance(identity, dict):
        raise ChunkRefusal(refusal_code, repr(observation))
    name = identity.get("Kernel Name")
    if (
        observation.get("kernel_name_base") != "mangled"
        or not isinstance(columns, list)
        or not all(isinstance(column, str) for column in columns)
        or len(columns) != len(set(columns))
        or not required.issubset(columns)
        or not isinstance(identity, dict)
        or not required.issubset(identity)
        or any(
            not isinstance(identity[field], str) or not identity[field].strip()
            for field in required
        )
        or not isinstance(name, str) or not name
    ):
        raise ChunkRefusal(
            refusal_code,
            repr(observation),
        )
    return (
        name,
        _normalize_ncu_dim(
            identity.get("Block Size"), field="Block Size",
            refusal_code=refusal_code,
        ),
        _normalize_ncu_dim(
            identity.get("Grid Size"), field="Grid Size",
            refusal_code=refusal_code,
        ),
        _normalize_ncu_device(
            identity.get("Device"), refusal_code=refusal_code,
        ),
    )


def _canonical_trace_kernel_name(value: Any) -> str:
    refusal_code = "label_launch_identity_join_mismatch"
    suffix = "___0"
    if not isinstance(value, str) or not value.endswith(suffix):
        raise ChunkRefusal(refusal_code, f"invalid trace kernel name: {value!r}")
    base = value[:-len(suffix)]
    if (
        not base or base.endswith(suffix) or base.startswith("regex:")
    ):
        raise ChunkRefusal(refusal_code, f"invalid exact trace kernel name: {value!r}")
    return base


def _canonical_trace_launch_key(
    launch: Any,
) -> tuple[str, tuple[int, int, int], tuple[int, int, int], int]:
    refusal_code = "label_launch_identity_join_mismatch"
    if not isinstance(launch, dict):
        raise ChunkRefusal(refusal_code, f"trace launch is not an object: {launch!r}")
    name = _canonical_trace_kernel_name(launch.get("kernel_name"))
    block = launch.get("block_dim")
    grid = launch.get("grid_dim")
    device = launch.get("device_id")
    if (
        not isinstance(block, list) or len(block) != 3
        or not isinstance(grid, list) or len(grid) != 3
        or any(type(item) is not int or item <= 0 for item in [*block, *grid])
        or type(device) is not int or device < 0
    ):
        raise ChunkRefusal(refusal_code, repr(launch))
    return name, tuple(block), tuple(grid), device  # type: ignore[return-value]


def _label_command_kernel_selectors(command: Any) -> list[str]:
    refusal_code = "label_launch_identity_join_mismatch"
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise ChunkRefusal(refusal_code, f"invalid label command: {command!r}")
    driver_args = command[:command.index("--")] if "--" in command else command
    selectors: list[str] = []
    index = 0
    while index < len(driver_args):
        token = driver_args[index]
        if token == "--kernel-name":
            if index + 1 >= len(driver_args):
                raise ChunkRefusal(refusal_code, "--kernel-name has no value")
            selectors.append(driver_args[index + 1])
            index += 2
            continue
        if token.startswith("--kernel-name="):
            selectors.append(token.split("=", 1)[1])
        index += 1
    return selectors


def _canonical_label_kernel_selector(invocation: dict[str, Any]) -> str | None:
    """Derive a pre-label selector only from the sealed v1 launch sequence."""
    contract = _require_capture_launch_contract(invocation)
    if contract.get("schema_version") != 1:
        return None
    sequence = contract.get("trace_launch_sequence")
    if sequence is None:
        return None
    selected = contract["label_trace_launch_index"]
    names = [
        _canonical_trace_kernel_name(row.get("kernel_name"))
        if isinstance(row, dict) else _canonical_trace_kernel_name(None)
        for row in sequence
    ]
    selected_name = names[selected]
    return selected_name if names.count(selected_name) > 1 else None


def require_canonical_label_launch_identity(
    *, label: dict[str, Any], trace: dict[str, Any], invocation: dict[str, Any],
) -> dict[str, Any]:
    """Require equality of independently observed canonical NCU/NVBit launches."""
    refusal_code = "label_launch_identity_join_mismatch"
    contract = invocation.get("capture_launch_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ChunkRefusal(refusal_code, f"invalid v1 launch contract: {contract!r}")
    selected = contract.get("label_trace_launch_index")
    launches = trace.get("launches")
    if (
        type(selected) is not int or selected < 0
        or not isinstance(launches, list)
        or selected >= len(launches)
    ):
        raise ChunkRefusal(
            refusal_code,
            repr({"label_trace_launch_index": selected, "launches": launches}),
        )
    expected_key = _canonical_trace_launch_key(launches[selected])
    normalized_names = [
        _canonical_trace_kernel_name(launch.get("kernel_name"))
        if isinstance(launch, dict) else _canonical_trace_kernel_name(None)
        for launch in launches
    ]
    expected_selectors = (
        [expected_key[0]] if normalized_names.count(expected_key[0]) > 1 else []
    )
    observed_selectors = _label_command_kernel_selectors(label.get("command"))
    observation = label.get("ncu_launch_observation_document")
    if not isinstance(observation, dict):
        raise ChunkRefusal(refusal_code, repr(observation))
    observed_key = _observation_ncu_launch_key(
        observation, refusal_code=refusal_code,
    )
    if observed_key != expected_key or observed_selectors != expected_selectors:
        raise ChunkRefusal(
            refusal_code,
            repr({
                "expected_launch_key": expected_key,
                "observed_launch_key": observed_key,
                "expected_kernel_selectors": expected_selectors,
                "observed_kernel_selectors": observed_selectors,
            }),
        )
    return {
        "launch_key": expected_key,
        "kernel_selector": expected_selectors[0] if expected_selectors else None,
    }


def bind_unique_ncu_launch_identity(
    *, label: dict[str, Any], inventory_row: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind one NCU report to an independently observed NVBit launch tuple.

    ``--launch-skip`` is a request, not evidence.  The join is therefore the
    independently emitted tuple ``(mangled name, block, grid, device)``.  This
    admits repeated functions launched at different geometries while refusing
    genuinely indistinguishable repeated tuples.
    """
    ordinal = inventory_row.get("launch_ordinal")
    expected_key = _trace_ncu_launch_key(inventory_row)
    inventory_keys = [_trace_ncu_launch_key(row) for row in inventory]
    if (
        type(ordinal) is not int or ordinal < 0
        or inventory_keys.count(expected_key) != 1
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_ncu_identity_ambiguous",
            repr({"ordinal": ordinal, "launch_key": expected_key}),
        )
    expected_function = expected_key[0]
    report = label.get("ncu_report")
    report_stat = (
        _output_lstat(
            report, code="label_capture_storage_failure",
            operation="sol_ncu_identity_report_stat",
        ) if isinstance(report, Path) else None
    )
    if (
        not isinstance(report, Path) or report_stat is None
        or not stat.S_ISREG(report_stat.st_mode)
    ):
        raise ChunkRefusal("sol_control_ncu_report_missing", repr(report))
    try:
        report_bytes = _read_output_bytes(
            report, code="label_capture_storage_failure",
            operation="sol_ncu_identity_report_read",
        )
        function_bytes = expected_function.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ChunkRefusal(
            "sol_dynamic_launch_ncu_identity_mismatch", str(exc)
        ) from exc
    occurrences = report_bytes.count(function_bytes)
    observation = label.get("ncu_launch_observation_document") or {}
    observed_key = _observation_ncu_launch_key(observation)
    if (
        occurrences <= 0 or label.get("launch_index") != ordinal
        or observed_key != expected_key
        or observation.get("report_sha256") != _sha256_output_file(
            report, code="label_capture_storage_failure",
            operation="sol_ncu_identity_report_hash",
        )
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_ncu_identity_mismatch",
            repr({
                "ordinal": ordinal, "kernel_name": expected_function,
                "expected_launch_key": expected_key,
                "observed_launch_key": observed_key,
                "occurrences": occurrences,
                "label_launch_index": label.get("launch_index"),
            }),
        )
    binding = {
        "join_policy": "unique_mangled_name_block_grid_device_tuple",
        "launch_ordinal": ordinal,
        "launch_identity_sha256": inventory_row["launch_identity_sha256"],
        "kernel_name": expected_function,
        "block_dim": list(expected_key[1]),
        "grid_dim": list(expected_key[2]),
        "device_id": expected_key[3],
        "launch_tuple_inventory_occurrences": 1,
        "kernel_name_report_occurrences": occurrences,
        "ncu_report_sha256": _sha256_output_file(
            report, code="label_capture_storage_failure",
            operation="sol_ncu_identity_binding_report_hash",
        ),
        "ncu_launch_observation_sha256": observation.get(
            "observation_sha256"
        ),
    }
    binding["ncu_launch_identity_sha256"] = canonical_sha256(binding)
    label["ncu_launch_identity"] = binding
    return binding


def build_dynamic_launch_completeness(
    *, trace: dict[str, Any], labels: list[dict[str, Any]],
    label_failures: list[dict[str, Any]], control: dict[str, Any],
    launch_completeness_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join trace, NCU, and control evidence on every exact launch ordinal."""
    contract = (
        _SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT
        if launch_completeness_contract is None
        else launch_completeness_contract
    )
    if (
        contract != _SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT
        and contract != _SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    ):
        raise ChunkRefusal(
            "sol_workload_launch_completeness_contract_invalid",
            repr(contract),
        )
    inventory = trace.get("dynamic_launch_inventory")
    inventory_sha = trace.get("dynamic_launch_inventory_sha256")
    control_rows = control.get("launch_records")
    if (
        not isinstance(inventory, list) or not inventory
        or inventory_sha != canonical_sha256(inventory)
        or not isinstance(control_rows, list)
        or len(control_rows) != len(inventory)
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_completeness_join_invalid", repr(inventory_sha)
        )
    labels_by_ordinal: dict[int, dict[str, Any]] = {}
    for label in labels:
        ordinal = label.get("launch_index")
        if type(ordinal) is not int or ordinal in labels_by_ordinal:
            raise ChunkRefusal(
                "sol_dynamic_launch_label_join_invalid", repr(ordinal)
            )
        labels_by_ordinal[ordinal] = label
    failures_by_ordinal: dict[int, dict[str, Any]] = {}
    for failure in label_failures:
        ordinal = failure.get("launch_ordinal")
        if (
            type(ordinal) is not int or ordinal in failures_by_ordinal
            or ordinal in labels_by_ordinal
        ):
            raise ChunkRefusal(
                "sol_dynamic_launch_label_join_invalid", repr(ordinal)
            )
        failures_by_ordinal[ordinal] = failure
    controls_by_ordinal: dict[int, dict[str, Any]] = {}
    for row in control_rows:
        ordinal = row.get("launch_ordinal") if isinstance(row, dict) else None
        if type(ordinal) is not int or ordinal in controls_by_ordinal:
            raise ChunkRefusal(
                "sol_dynamic_launch_control_join_invalid", repr(ordinal)
            )
        controls_by_ordinal[ordinal] = row
    trace_complete = trace.get("outcome") in {None, "admitted"}
    trace_sequence = [
        {
            "launch_ordinal": row["launch_ordinal"],
            "kernel_name": row["module_mapping"]["kernel_name"],
            "launch_identity_sha256": row["launch_identity_sha256"],
        }
        for row in inventory
    ]
    trace_sequence_sha256 = canonical_sha256(trace_sequence)
    ncu_sequence: list[dict[str, Any]] = []
    joined: list[dict[str, Any]] = []
    label_complete_count = control_complete_count = 0
    for expected_ordinal, inventory_row in enumerate(inventory):
        if inventory_row.get("launch_ordinal") != expected_ordinal:
            raise ChunkRefusal(
                "sol_dynamic_launch_completeness_join_invalid",
                repr(inventory_row),
            )
        label = labels_by_ordinal.get(expected_ordinal)
        label_failure = failures_by_ordinal.get(expected_ordinal)
        control_row = controls_by_ordinal.get(expected_ordinal)
        if (
            (label is None) == (label_failure is None)
            or control_row is None
            or control_row.get("launch_identity_sha256")
            != inventory_row.get("launch_identity_sha256")
        ):
            raise ChunkRefusal(
                "sol_dynamic_launch_completeness_join_invalid",
                repr(expected_ordinal),
            )
        if (
            label_failure is not None
            and label_failure.get("typed_reason")
            != "ncu_launch_identity_ambiguous_repeated_tuple"
        ):
            raise ChunkRefusal(
                "sol_dynamic_launch_label_failure_not_terminal",
                repr(label_failure.get("typed_reason")),
            )
        label_complete = label is not None
        if label is not None:
            launch_key = _trace_ncu_launch_key(inventory_row)
            expected_identity = {
                "join_policy": "unique_mangled_name_block_grid_device_tuple",
                "launch_ordinal": expected_ordinal,
                "launch_identity_sha256": inventory_row[
                    "launch_identity_sha256"
                ],
                "kernel_name": inventory_row["module_mapping"]["kernel_name"],
                "block_dim": list(launch_key[1]),
                "grid_dim": list(launch_key[2]),
                "device_id": launch_key[3],
                "launch_tuple_inventory_occurrences": 1,
                "kernel_name_report_occurrences": (
                    label.get("ncu_launch_identity") or {}
                ).get("kernel_name_report_occurrences"),
                "ncu_report_sha256": _sha256_output_file(
                    label["ncu_report"], code="label_capture_storage_failure",
                    operation="sol_completeness_report_hash",
                ),
                "ncu_launch_observation_sha256": (
                    label.get("ncu_launch_observation_document") or {}
                ).get("observation_sha256"),
            }
            expected_identity["ncu_launch_identity_sha256"] = canonical_sha256(
                expected_identity
            )
            if label.get("ncu_launch_identity") != expected_identity:
                raise ChunkRefusal(
                    "sol_dynamic_launch_ncu_identity_mismatch",
                    repr(expected_ordinal),
                )
            ncu_sequence.append(copy.deepcopy(trace_sequence[expected_ordinal]))
        control_complete = control_row.get("status") == "complete"
        label_complete_count += int(label_complete)
        control_complete_count += int(control_complete)
        joined.append({
            "launch_ordinal": expected_ordinal,
            "launch_identity_sha256": inventory_row[
                "launch_identity_sha256"
            ],
            "kernel_name": inventory_row["module_mapping"]["kernel_name"],
            "trace_complete": trace_complete,
            "ncu_label_complete": label_complete,
            "ncu_report_sha256": (
                _sha256_output_file(
                    label["ncu_report"], code="label_capture_storage_failure",
                    operation="sol_completeness_join_report_hash",
                ) if label is not None else None
            ),
            "ncu_label_typed_reason": (
                None if label_failure is None
                else label_failure.get("typed_reason")
            ),
            "control_complete": control_complete,
            "control_outcome": control_row.get("control_outcome"),
            "control_typed_reason": control_row.get("typed_reason"),
        })
    expected_ordinals = set(range(len(inventory)))
    if (
        set(labels_by_ordinal) | set(failures_by_ordinal) != expected_ordinals
        or set(controls_by_ordinal) != expected_ordinals
    ):
        raise ChunkRefusal(
            "sol_dynamic_launch_completeness_join_invalid", "ordinal sets"
        )
    ncu_sequence_sha256 = (
        canonical_sha256(ncu_sequence)
        if len(ncu_sequence) == len(inventory) else None
    )
    sequence_equal = ncu_sequence_sha256 == trace_sequence_sha256
    representative = (
        contract == _SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    if representative:
        allowed_controls = all(
            row["control_complete"] is True
            or (
                row["control_outcome"]
                == "typed_unavailable_no_stable_static_module"
                and row["control_typed_reason"] == "origin_unresolved"
            )
            for row in joined
        )
        complete = (
            trace_complete and label_complete_count == len(inventory)
            and sequence_equal and allowed_controls
        )
    else:
        complete = (
            trace_complete and label_complete_count == len(inventory)
            and control_complete_count == len(inventory) and sequence_equal
        )
    result = {
        "schema_version": 1,
        "launch_scope": "all_in_profiler_roi",
        "contract": copy.deepcopy(contract),
        "status": (
            "complete_dynamic_launch_set" if complete
            else "typed_incomplete_dynamic_launch_set"
        ),
        "launch_inventory_sha256": inventory_sha,
        "trace_launch_sequence_sha256": trace_sequence_sha256,
        "ncu_replay_sequence_sha256": ncu_sequence_sha256,
        "trace_ncu_sequence_equal": sequence_equal,
        "inventory_launch_count": len(inventory),
        "trace_complete_launch_count": len(inventory) if trace_complete else 0,
        "ncu_label_complete_launch_count": label_complete_count,
        "control_complete_launch_count": control_complete_count,
        "launches": joined,
    }
    result["completeness_sha256"] = canonical_sha256(result)
    return result


def _remove_routed_raw_trace(trace: dict, *, cell_dir: Path) -> None:
    if not trace.get("routed"):
        return
    raw = trace.get("raw_trace_dir")
    expected = cell_dir / "trace_full_raw"
    if (
        not isinstance(raw, Path)
        or raw != expected
        or raw.is_symlink()
        or not raw.is_dir()
    ):
        raise ChunkRefusal("trace_raw_cleanup_refused", repr(raw))
    _rmtree_output(
        raw, code="trace_raw_cleanup_storage_failure",
        operation="routed_raw_trace_cleanup",
    )
    if raw.exists() or raw.is_symlink():
        raise ChunkRefusal("trace_raw_cleanup_failed", str(raw))


def inventory_cell_artifacts(
    *, cell_dir: Path, chunk_root: Path, label: dict | list[dict], trace: dict,
    control: dict, result_path: Path, disposition: str,
    typed_reason: str | None,
) -> list[dict]:
    """Hash every regular cell artifact exactly once.

    Trace allocation tables and other ``extra_info`` files affect replay even
    though they are not one of the three headline payloads.  Walking the whole
    cell directory keeps those consumer inputs inside the DONE seal.
    """
    labels = label if isinstance(label, list) else [label]
    exact_kinds: dict[Path, str] = {
        _resolve_output_path(
            trace["manifest"], code="cell_inventory_storage_failure",
            operation="trace_manifest_inventory_resolve",
        ): "trace_manifest",
        _resolve_output_path(
            control["control_bits"], code="cell_inventory_storage_failure",
            operation="control_bits_inventory_resolve",
        ): "control_bits",
        _resolve_output_path(
            control["manifest"], code="cell_inventory_storage_failure",
            operation="control_manifest_inventory_resolve",
        ): "control_manifest",
        _resolve_output_path(
            result_path, code="cell_inventory_storage_failure",
            operation="cell_result_inventory_resolve",
        ): "cell_result",
    }
    for label_row in labels:
        exact_kinds[_resolve_output_path(
            label_row["sidecar"], code="cell_inventory_storage_failure",
            operation="label_sidecar_inventory_resolve",
        )] = "label_sidecar"
        exact_kinds[_resolve_output_path(
            label_row["run_manifest"], code="cell_inventory_storage_failure",
            operation="label_run_manifest_inventory_resolve",
        )] = "label_run_manifest"
        exact_kinds[_resolve_output_path(
            label_row["ncu_report"], code="cell_inventory_storage_failure",
            operation="ncu_report_inventory_resolve",
        )] = "ncu_report"
        exact_kinds[
            _resolve_output_path(
                label_row["ncu_launch_observation"],
                code="cell_inventory_storage_failure",
                operation="ncu_observation_inventory_resolve",
            )
        ] = "ncu_launch_observation"
    if isinstance(trace.get("root_pb"), Path):
        exact_kinds[_resolve_output_path(
            trace["root_pb"], code="cell_inventory_storage_failure",
            operation="trace_root_inventory_resolve",
        )] = "trace_root"
    if isinstance(trace.get("stats_csv"), Path):
        exact_kinds[_resolve_output_path(
            trace["stats_csv"], code="cell_inventory_storage_failure",
            operation="trace_stats_inventory_resolve",
        )] = "trace_stats"
    if isinstance(trace.get("route_result_path"), Path):
        exact_kinds[_resolve_output_path(
            trace["route_result_path"], code="cell_inventory_storage_failure",
            operation="trace_route_inventory_resolve",
        )] = "trace_route_record"
    artifacts: list[dict] = []
    for path in _output_tree_paths(
        cell_dir, code="cell_inventory_storage_failure",
        operation="cell_inventory_tree_walk",
    ):
        observed = _output_lstat(
            path, code="cell_inventory_storage_failure",
            operation="cell_inventory_path_stat",
        )
        if observed is None:
            raise ChunkRefusal("artifact_not_regular", str(path))
        if stat.S_ISLNK(observed.st_mode):
            raise ChunkRefusal("artifact_not_regular", str(path))
        if not stat.S_ISREG(observed.st_mode):
            continue
        resolved = _resolve_output_path(
            path, code="cell_inventory_storage_failure",
            operation="cell_inventory_path_resolve",
        )
        kind = exact_kinds.get(resolved)
        if kind is None:
            rel = path.relative_to(cell_dir)
            if (
                rel.parts[:2] == ("trace", "threadblocks")
                and (path.name.endswith(".pb") or path.name.endswith(".pb.zst"))
            ):
                kind = "trace_threadblock"
            elif rel.parts and rel.parts[0] == "trace":
                kind = "trace_auxiliary"
            elif rel.parts and rel.parts[0] in {"sol_stage", "sol_requests"}:
                kind = "sol_capture_evidence"
            elif rel.parts and rel.parts[0] in {
                "control_evidence", "control_launches",
            }:
                kind = "control_evidence"
            elif rel.name == "sol_native_env_fingerprint.json":
                kind = "sol_capture_evidence"
            elif rel.parts and rel.parts[0] == "logs":
                # The process-only collection receipt remains an execution log
                # until the later transitive DONE contract admits it as a
                # required semantic artifact.
                kind = "execution_log"
            else:
                raise ChunkRefusal(
                    "artifact_kind_unclassified", f"{path} ({rel.as_posix()})"
                )
        artifacts.append(artifact_row(path, root=chunk_root, artifact_kind=kind))
    base_required = {
        "trace_manifest", "control_bits", "control_manifest", "cell_result",
    }
    if labels:
        base_required |= {
            "label_sidecar", "label_run_manifest", "ncu_report",
        }
    if disposition == "admitted" and typed_reason is None:
        required = base_required | {
            "trace_root", "trace_stats", "trace_threadblock",
        }
    elif disposition == "failed" and typed_reason in {
        "trace_unsupported", "supplemental_dynamic_launch_set_incomplete",
    }:
        if trace.get("routed"):
            # A certified budget route intentionally removes the raw trace
            # tree only after the retained auxiliary payload and route record
            # are sealed.  Requiring raw root/stats after that cleanup makes
            # every valid typed-unavailable result impossible to inventory.
            required = base_required | {
                "trace_auxiliary", "trace_route_record",
            }
        else:
            required = base_required | {
                "trace_root", "trace_stats", "trace_threadblock",
            }
    else:
        raise ChunkRefusal(
            "cell_disposition_invalid",
            f"disposition={disposition!r}, typed_reason={typed_reason!r}",
        )
    if trace.get("routed"):
        required.add("trace_route_record")
    if (cell_dir / "sol_stage").is_dir():
        required.add("sol_capture_evidence")
    if (
        (cell_dir / "control_evidence").is_dir()
        or (cell_dir / "control_launches").is_dir()
    ):
        required.add("control_evidence")
    observed = {row["artifact_kind"] for row in artifacts}
    missing = sorted(required - observed)
    if missing:
        raise ChunkRefusal("artifact_kind_missing", repr(missing))
    return artifacts


def capture_cell(
    *, cell: dict, cell_dir: Path, matrix: dict, pins: dict,
    cell_budgets: Any, expected_tracer_sha256: str,
    capture_provenance: dict[str, Any],
    expected_ncu_identity: ncu_executor.ExpectedNcuIdentity,
    certified_sol_env: dict[str, str] | None,
    sol_environment_probe: dict[str, Any] | None,
    module_store: Path,
    expected_sol_fingerprint_pin: dict[str, Any] | None = None,
    workload_row_binding: dict[str, Any] | None = None,
    supplemental_runnability_evidence: dict[str, Any] | None = None,
    runnability_path: Path | None = None,
    supplemental_workload_path: Path | None = None,
    supplemental_definition_path: Path | None = None,
) -> dict:
    try:
        ncu_executor.validate_expected_ncu_identity(expected_ncu_identity)
    except ncu_executor.NcuActionExecutorError as exc:
        raise ChunkRefusal("capture_ncu_authority_invalid", str(exc)) from exc
    chunk_root = cell_dir.parents[1]
    capture_started_at_utc = capture_provenance.get("capture_started_at_utc")
    admission_run_id = capture_provenance.get("admission_run_id")
    supersedes = capture_provenance.get("supersedes")
    prior_preparation_failures = capture_provenance.get(
        "prior_preparation_failures"
    )
    prior_capture_defect_attempts = capture_provenance.get(
        "prior_capture_defect_attempts"
    )
    execution_source_commit = capture_provenance.get("source_commit")
    preflight_sha256 = capture_provenance.get("preflight_sha256")
    _parse_utc_timestamp(
        capture_started_at_utc, field="capture_started_at_utc"
    )
    if (
        not isinstance(admission_run_id, str) or len(admission_run_id) != 64
        or any(ch not in "0123456789abcdef" for ch in admission_run_id)
        or (supersedes is not None and not isinstance(supersedes, dict))
        or not _preparation_backrefs_well_formed(prior_preparation_failures)
        or not _capture_defect_backrefs_well_formed(
            prior_capture_defect_attempts
        )
        or not _is_sha40(execution_source_commit)
        or not _is_sha256(preflight_sha256)
    ):
        raise ChunkRefusal(
            "capture_admission_invalid", repr(capture_provenance)
        )
    supplemental_all_launches = workload_row_binding is not None
    if supplemental_all_launches:
        identity = cell.get("identity")
        if (
            not isinstance(identity, dict)
            or identity.get("roster_source") != "sol_execbench"
        ):
            raise ChunkRefusal(
                "sol_workload_row_binding_invalid", "binding on non-SOL cell"
            )
        raise ChunkRefusal(
            "sol_supplemental_launch_set_v3_live_canary_required",
            "supplemental capture remains disabled until a pinned live canary",
        )
    binary, invocation, budget, runtime_cwd, runtime_env = validate_capture_cell(
        cell, cell_budgets=cell_budgets,
        certified_sol_env=certified_sol_env,
        workload_row_binding=workload_row_binding,
        supplemental_runnability_evidence=supplemental_runnability_evidence,
        runnability_path=runnability_path,
        supplemental_workload_path=supplemental_workload_path,
        supplemental_definition_path=supplemental_definition_path,
    )
    if workload_row_binding is not None:
        representative_execution = (
            workload_row_binding.get("execution_profile")
            == _SOL_REPRESENTATIVE_EXECUTION_PROFILE
        )
        expected_launch_contract = (
            _SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
            if representative_execution
            else _SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT
        )
        if cell.get("identity", {}).get("roster_source") != "sol_execbench":
            raise ChunkRefusal(
                "sol_workload_row_binding_invalid", "binding on non-SOL cell"
            )
        if (
            cell.get("launch_scope") != "all_in_profiler_roi"
            or cell.get("launch_completeness_contract")
            != expected_launch_contract
        ):
            raise ChunkRefusal(
                "sol_workload_launch_completeness_contract_invalid",
                repr(cell.get("launch_completeness_contract")),
            )
        _write_json_exclusive(
            cell_dir / "sol_requests" / "workload_row_binding.json",
            workload_row_binding,
        )
        flashinfer = workload_row_binding["flashinfer_corpus"]
        category = workload_row_binding["category"]
        expected_trace_root = (
            str((REPO_ROOT / flashinfer["root"]).resolve())
            if category == "FIB" else None
        )
        observed_trace_root = (
            (runtime_env or {}).get("FLASHINFER_TRACE_DIR")
        )
        if observed_trace_root != expected_trace_root:
            raise ChunkRefusal(
                "sol_workload_flashinfer_environment_mismatch",
                repr({
                    "category": category,
                    "expected": expected_trace_root,
                    "observed": observed_trace_root,
                }),
            )
        runtime_binding = {
            "schema_version": 1,
            "kind": "sol_workload_flashinfer_runtime_binding",
            "category": category,
            "flashinfer_corpus": copy.deepcopy(flashinfer),
            "environment_variable": "FLASHINFER_TRACE_DIR",
            "injected_absolute_root": observed_trace_root,
            "ambient_value_inherited": False,
        }
        runtime_binding["runtime_binding_sha256"] = canonical_sha256(
            runtime_binding
        )
        _write_json_exclusive(
            cell_dir / "sol_requests" / "flashinfer_runtime_binding.json",
            runtime_binding,
        )
    started = time.monotonic()
    deadline = started + budget.max_wall_time_s

    def remaining(phase: str) -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ChunkRefusal(
                "cell_deadline_exhausted",
                f"{cell['identity']}: phase={phase}, "
                f"max_wall_time_s={budget.max_wall_time_s}",
            )
        return seconds

    is_sol = cell["identity"]["roster_source"] == "sol_execbench"
    sol_context: dict[str, Any] | None = None
    guard_prepare = None
    label_env = runtime_env
    trace_env = runtime_env
    phase_validator = None
    phase_env_factory = None
    if is_sol:
        expected_env_hash = invocation["adapter_contract"]["native_env_hash"]
        observed_env = (
            (sol_environment_probe or {}).get("observed_fingerprint") or {}
        )
        expected_fingerprint_pin = (
            matrix["input_pins"]["sol_native_env_fingerprint"]
            if expected_sol_fingerprint_pin is None
            else expected_sol_fingerprint_pin
        )
        if (
            not isinstance(sol_environment_probe, dict)
            or sol_environment_probe.get("status") != preflight.STATUS_OK
            or observed_env.get("degraded") is not False
            or observed_env.get("missing") != []
            or observed_env.get("hash") != expected_env_hash
            or sol_environment_probe.get("matrix_pin")
            != expected_fingerprint_pin
        ):
            raise ChunkRefusal(
                "SOL_ENV_ATTESTATION_INVALID", repr(sol_environment_probe)
            )
        _write_json_exclusive(
            cell_dir / "sol_native_env_fingerprint.json",
            sol_env.build_cell_attestation(
                identity=cell["identity"], probe=sol_environment_probe,
            ),
        )
        guard_prepare = live_capture_guard(pins)
        sol_context = prepare_sol_capture(
            cell=cell, binary=binary, invocation=invocation,
            cell_dir=cell_dir, runtime_cwd=runtime_cwd,
            runtime_env=runtime_env or dict(os.environ),
            timeout_s=remaining("sol_prepare"),
        )
        if workload_row_binding is None:
            label_env = sol_phase_env(
                sol_context, phase_tag="label", capture_kind="label",
                timeout_s=remaining("sol_label_request"),
            )
        phase_validator = lambda env, kind: validate_sol_phase_env(
            sol_context, env, kind
        )
        phase_env_factory = lambda tag, timeout: sol_phase_env(
            sol_context, phase_tag=tag, capture_kind="trace", timeout_s=timeout,
        )
    guard_pre = live_capture_guard(pins)
    control_policy = (cell.get("control_bits") or {}).get("policy")
    label_failures: list[dict[str, Any]] = []
    if supplemental_all_launches:
        if control_policy != "post_trace_module_or_typed":
            raise ChunkRefusal(
                "sol_workload_control_policy_invalid", repr(control_policy)
            )
        trace_env = sol_phase_env(
            sol_context, phase_tag="trace_full_all_roi", capture_kind="trace",
            timeout_s=remaining("sol_trace_request"),
        )
        _require_hash(
            binary, invocation["executable_sha256"], "binary before trace"
        )
        trace = capture_trace(
            cell=cell, binary=binary, invocation=invocation, budget=budget,
            cell_dir=cell_dir, pins=pins,
            expected_tracer_sha256=expected_tracer_sha256,
            runtime_cwd=runtime_cwd, runtime_env=trace_env,
            timeout_s=remaining("trace"),
            capture_provenance=capture_provenance,
            phase_env_factory=phase_env_factory,
            phase_result_validator=phase_validator,
            all_profiler_roi_launches=True,
        )
        _require_hash(
            binary, invocation["executable_sha256"], "binary after trace"
        )
        labels_by_ordinal: dict[int, dict[str, Any]] = {}
        between_guards: list[dict[str, Any]] = []
        for launch_row in trace["dynamic_launch_inventory"]:
            ordinal = launch_row["launch_ordinal"]
            between_guards.append(live_capture_guard(pins))
            phase_tag = f"label_launch_{ordinal:04d}"
            launch_label_env = sol_phase_env(
                sol_context, phase_tag=phase_tag, capture_kind="label",
                timeout_s=remaining(f"sol_{phase_tag}_request"),
            )
            launch_key = _trace_ncu_launch_key(launch_row)
            repeated_tuple = sum(
                _trace_ncu_launch_key(candidate) == launch_key
                for candidate in trace["dynamic_launch_inventory"]
            ) > 1
            if repeated_tuple:
                failure = {
                    "schema_version": 1,
                    "kind": "sol_dynamic_launch_label_incomplete",
                    "launch_ordinal": ordinal,
                    "launch_identity_sha256": launch_row[
                        "launch_identity_sha256"
                    ],
                    "typed_reason": (
                        "ncu_launch_identity_ambiguous_repeated_tuple"
                    ),
                    "detail": repr(launch_key),
                }
                failure["failure_sha256"] = canonical_sha256(failure)
                _write_json_exclusive(
                    cell_dir / "logs" / f"{phase_tag}_incomplete.json",
                    failure,
                )
                label_failures.append(failure)
                between_guards.append(live_capture_guard(pins))
                continue
            label_error: BaseException | None = None
            try:
                label_result = capture_label(
                    cell=cell, binary=binary, invocation=invocation,
                    cell_dir=cell_dir, matrix=matrix, pins=pins,
                    timeout_s=remaining(phase_tag),
                    preflight_sha256=capture_provenance["preflight_sha256"],
                    expected_ncu_identity=expected_ncu_identity,
                    runtime_cwd=runtime_cwd,
                    runtime_env=launch_label_env,
                    phase_result_validator=phase_validator,
                    label_launch_index_override=ordinal,
                    artifact_tag=phase_tag,
                    execution_source_commit=execution_source_commit,
                    bind_supplemental_toolchain=True,
                )
            except ChunkRefusal as exc:
                label_error = exc
                # Capture/runtime/schema failures are retryable or integrity
                # failures at the journal layer, never terminally downgraded.
                raise
            else:
                bind_unique_ncu_launch_identity(
                    label=label_result, inventory_row=launch_row,
                    inventory=trace["dynamic_launch_inventory"],
                )
                labels_by_ordinal[ordinal] = label_result
            finally:
                try:
                    post_label_guard = live_capture_guard(pins)
                except BaseException:
                    if not _is_explicit_output_storage_refusal(label_error):
                        raise
                    # The read-only guard may still run, but it must never
                    # replace an already typed output-storage refusal.
                else:
                    between_guards.append(post_label_guard)
            _require_hash(
                binary, invocation["executable_sha256"],
                f"binary after label ordinal {ordinal}",
            )
        labels = [labels_by_ordinal[key] for key in sorted(labels_by_ordinal)]
        label = labels[0] if labels else None
        guard_between = between_guards
        control = capture_sol_controls_all_launches(
            cell=cell, invocation=invocation,
            labels_by_ordinal=labels_by_ordinal, trace=trace,
            cell_dir=cell_dir, module_store=module_store,
        )
    else:
        _require_hash(
            binary, invocation["executable_sha256"], "binary before label"
        )
        kernel_name_selector = _canonical_label_kernel_selector(invocation)
        label = capture_label(
            cell=cell, binary=binary, invocation=invocation,
            cell_dir=cell_dir, matrix=matrix, pins=pins,
            timeout_s=remaining("label"),
            preflight_sha256=capture_provenance["preflight_sha256"],
            expected_ncu_identity=expected_ncu_identity,
            runtime_cwd=runtime_cwd,
            runtime_env=label_env, phase_result_validator=phase_validator,
            kernel_name_selector=kernel_name_selector,
            execution_source_commit=execution_source_commit,
            bind_supplemental_toolchain=False,
        )
        labels = [label]
        _require_hash(
            binary, invocation["executable_sha256"], "binary after label"
        )
        # NCU and NVBit are sequential; no profiler process from the label run
        # may survive into injection.
        guard_between = live_capture_guard(pins)
        if is_sol:
            trace_env = sol_phase_env(
                sol_context, phase_tag="trace_full", capture_kind="trace",
                timeout_s=remaining("sol_trace_request"),
            )
        trace = capture_trace(
            cell=cell, binary=binary, invocation=invocation, budget=budget,
            cell_dir=cell_dir, pins=pins,
            expected_tracer_sha256=expected_tracer_sha256,
            runtime_cwd=runtime_cwd, runtime_env=trace_env,
            timeout_s=remaining("trace"),
            capture_provenance=capture_provenance,
            phase_env_factory=phase_env_factory,
            phase_result_validator=phase_validator,
        )
        _require_hash(
            binary, invocation["executable_sha256"], "binary after trace"
        )
        require_canonical_label_launch_identity(
            label=label, trace=trace, invocation=invocation,
        )
        if control_policy == "post_trace_module_or_typed":
            control = capture_sol_control(
                cell=cell, invocation=invocation, label=label, trace=trace,
                cell_dir=cell_dir, module_store=module_store,
            )
        elif control_policy == "post_trace_vendor_module":
            control = capture_vendor_control(
                cell=cell, invocation=invocation, label=label, trace=trace,
                cell_dir=cell_dir, module_store=module_store,
            )
        elif control_policy == "direct_executable":
            control = capture_control(
                cell=cell, binary=binary, invocation=invocation,
                cell_dir=cell_dir,
                timeout_s=min(300.0, remaining("control")),
            )
        else:
            raise ChunkRefusal(
                "control_policy_not_implemented", repr(control_policy)
            )
    _require_hash(binary, invocation["executable_sha256"], "binary after control")
    remaining("control_complete")
    launch_completeness = (
        build_dynamic_launch_completeness(
            trace=trace, labels=labels, label_failures=label_failures,
            control=control,
            launch_completeness_contract=(
                workload_row_binding.get("launch_completeness_contract")
                if workload_row_binding is not None else None
            ),
        )
        if supplemental_all_launches else None
    )
    _remove_routed_raw_trace(trace, cell_dir=cell_dir)
    guard_post = live_capture_guard(pins)
    trace_outcome = trace.get("outcome")
    if supplemental_all_launches:
        if launch_completeness["status"] == "complete_dynamic_launch_set":
            disposition = "admitted"
            typed_reason = None
            detail = None
        else:
            disposition = "failed"
            typed_reason = "supplemental_dynamic_launch_set_incomplete"
            detail = (
                f"trace={launch_completeness['trace_complete_launch_count']}/"
                f"{launch_completeness['inventory_launch_count']}; "
                f"labels={launch_completeness['ncu_label_complete_launch_count']}/"
                f"{launch_completeness['inventory_launch_count']}; "
                f"controls={launch_completeness['control_complete_launch_count']}/"
                f"{launch_completeness['inventory_launch_count']}"
            )
    elif trace_outcome in {None, "admitted"}:
        disposition = "admitted"
        typed_reason = None
        detail = None
    elif trace_outcome == "typed_unavailable":
        disposition = "failed"
        typed_reason = "trace_unsupported"
        detail = (
            f"certified AC-9 route method="
            f"{(trace.get('route_result') or {}).get('method')!r}; "
            f"reason={trace.get('typed_reason')!r}"
        )
    else:
        raise ChunkRefusal("trace_route_outcome_invalid", repr(trace_outcome))
    definition_equivalence = (
        sol_context["definition_equivalence"] if is_sol else None
    )
    definition_equivalence_attestation_sha256 = (
        sol_context["definition_equivalence_attestation_sha256"]
        if is_sol else None
    )
    result_path = cell_dir / "cell_capture_result.json"
    elapsed_before_inventory = time.monotonic() - started
    result = {
        "schema_version": 1,
        "identity": cell["identity"],
        "authority_cell_sha256": cell["authority_cell_sha256"],
        "disposition": disposition,
        "typed_reason": typed_reason,
        "detail": detail,
        "binary_sha256": invocation["executable_sha256"],
        "guards": {
            "prepare": guard_prepare, "pre": guard_pre,
            "between": guard_between, "post": guard_post,
        },
        "capture_preflight_sha256": capture_provenance["preflight_sha256"],
        "capture_started_at_utc": capture_started_at_utc,
        "admission_run_id": admission_run_id,
        "supersedes": supersedes,
        "prior_preparation_failures": prior_preparation_failures,
        "prior_capture_defect_attempts": prior_capture_defect_attempts,
        "definition_equivalence": definition_equivalence,
        "definition_equivalence_attestation_sha256": (
            definition_equivalence_attestation_sha256
        ),
        "label_unit_metric_count": (
            labels[0]["unit_metric_count"] if labels else 0
        ),
        "label_launch_count": len(labels),
        "trace_threadblocks_pb_count": trace["threadblocks_pb_count"],
        "trace_admitted_records": trace["admitted_records"],
        "control_decoded_instructions": control["total_decoded"],
        "control_outcome": control.get("outcome", "direct_static_controls_emitted"),
        "max_wall_time_s": budget.max_wall_time_s,
        "trace_route_outcome": trace_outcome or "admitted",
        "trace_route_result_sha256": (
            (trace.get("route_result") or {}).get("result_sha256")
        ),
        "cell_elapsed_seconds_before_inventory": elapsed_before_inventory,
        "launch_completeness": launch_completeness,
    }
    if workload_row_binding is not None:
        result["workload_row_binding"] = copy.deepcopy(workload_row_binding)
    _write_output_text(
        result_path, json.dumps(result, indent=2, sort_keys=True) + "\n",
        code="cell_result_storage_failure",
        operation="cell_capture_result_write",
    )
    artifacts = inventory_cell_artifacts(
        cell_dir=cell_dir, chunk_root=chunk_root, label=labels, trace=trace,
        control=control, result_path=result_path, disposition=disposition,
        typed_reason=typed_reason,
    )
    remaining("inventory_complete")
    captured = {
        "roster_source": cell["identity"]["roster_source"],
        "slug": cell["identity"]["slug"],
        "authority_cell_sha256": cell["authority_cell_sha256"],
        "disposition": disposition,
        "typed_reason": typed_reason,
        "detail": detail,
        "capture_preflight_sha256": capture_provenance["preflight_sha256"],
        "capture_started_at_utc": capture_started_at_utc,
        "admission_run_id": admission_run_id,
        "supersedes": supersedes,
        "prior_preparation_failures": prior_preparation_failures,
        "prior_capture_defect_attempts": prior_capture_defect_attempts,
        "definition_equivalence": definition_equivalence,
        "definition_equivalence_attestation_sha256": (
            definition_equivalence_attestation_sha256
        ),
        "launch_completeness": launch_completeness,
        "artifacts": artifacts,
    }
    if workload_row_binding is not None:
        captured["workload_row_binding"] = copy.deepcopy(
            workload_row_binding
        )
    return captured


def _assert_external_output_root(output_root: Path) -> None:
    if not output_root.is_absolute():
        raise ChunkRefusal("output_root_not_absolute", str(output_root))
    repo_root = REPO_ROOT.resolve()
    prospective = _resolve_output_path(
        output_root, code="output_root_storage_failure",
        operation="output_root_preflight_resolve", strict=False,
    )
    try:
        prospective.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise ChunkRefusal(
            "output_root_inside_repository",
            "capture artifacts must not dirty or mutate the source tree",
        )
    _mkdir_output(
        output_root, parents=True, exist_ok=True,
        code="output_root_storage_failure",
        operation="output_root_create",
    )
    resolved = _resolve_output_path(
        output_root, code="output_root_storage_failure",
        operation="output_root_resolve",
    )
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return
    raise ChunkRefusal(
        "output_root_inside_repository",
        "capture artifacts must not dirty or mutate the source tree",
    )


def acquire_campaign_lock(output_root: Path):
    """Hold one nonblocking writer lock for this external campaign root."""
    path = output_root / ".capture.lock"
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        _raise_typed_storage_failure(
            exc, code="campaign_lock_storage_failure",
            operation="campaign_chunk_lock_open",
        )
        raise ChunkRefusal("campaign_lock_invalid", str(path)) from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ChunkRefusal("campaign_lock_invalid", str(path))
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        try:
            os.close(descriptor)
        except OSError:
            # Contention is the first fault.  A cleanup-close fault must not
            # promote or replace that semantic refusal.
            pass
        raise ChunkRefusal(
            "campaign_writer_active", f"another capture runner holds {path}"
        ) from exc
    except ChunkRefusal:
        try:
            os.close(descriptor)
        except OSError:
            # Preserve the first semantic fault.
            pass
        raise
    except OSError as exc:
        try:
            _raise_typed_storage_failure(
                exc, code="campaign_lock_storage_failure",
                operation="campaign_chunk_lock_stat_or_flock",
            )
        except ChunkRefusal:
            try:
                os.close(descriptor)
            except OSError:
                # The mapped stat/flock failure is the first output fault.
                pass
            raise
        try:
            os.close(descriptor)
        except OSError:
            # The non-storage stat/flock failure is still the first fault.
            pass
        raise ChunkRefusal("campaign_lock_invalid", str(path)) from exc
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        payload = f"pid={os.getpid()}\n".encode("ascii")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(None, "short campaign chunk lock write")
        os.fsync(descriptor)
    except OSError as exc:
        try:
            _raise_typed_storage_failure(
                exc, code="campaign_lock_storage_failure",
                operation="campaign_chunk_lock_record",
            )
        except ChunkRefusal:
            try:
                os.close(descriptor)
            except OSError:
                # Preserve the first record/write/fsync storage fault.
                pass
            raise
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise ChunkRefusal("campaign_lock_invalid", str(path)) from exc
    return os.fdopen(descriptor, "r+")


def validate_resume_inputs(
    manifest: dict, *, pins_path: Path, authority_path: Path,
    aggregate_budget_path: Path, matrix_path: Path, preflight_path: Path,
) -> None:
    expected = {
        "pins_sha256": sha256_file(pins_path),
        "roster_authority_sha256": sha256_file(authority_path),
        "trace_budget_sha256": sha256_file(aggregate_budget_path),
        "capture_matrix_sha256": sha256_file(matrix_path),
        "preflight_sha256": sha256_file(preflight_path),
    }
    mismatches = [
        f"{field}: sealed={manifest.get(field)!r} current={value!r}"
        for field, value in expected.items() if manifest.get(field) != value
    ]
    if mismatches:
        raise ChunkRefusal("resume_input_mismatch", "; ".join(mismatches))


def _cell_safe_name(cell: dict) -> str:
    identity = cell.get("identity") or {}
    source = identity.get("roster_source")
    slug = identity.get("slug")
    if not all(
        isinstance(value, str) and value
        and all(ch.isalnum() or ch in "._-" for ch in value)
        for value in (source, slug)
    ):
        raise ChunkRefusal("cell_filesystem_identity_invalid", repr(identity))
    return f"{source}__{slug}"


JOURNAL_START_NAME = "journal_start.json"
JOURNAL_COMPLETE_NAME = "journal_complete.json"
SUPERSESSION_INVENTORY_NAME = "inventory.json"
SUPERSESSION_LEDGER_NAME = "supersession_ledger.jsonl"
PREPARATION_FAILURES_DIR_NAME = "preparation_failures"
PREPARATION_FAILURE_LEDGER_NAME = "preparation_failure_ledger.jsonl"
CAPTURE_DEFECT_ATTEMPTS_DIR_NAME = "capture_defect_attempts"
CAPTURE_DEFECT_LEDGER_NAME = "capture_defect_ledger.jsonl"


def _is_uuid4(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError):
        return False
    return parsed.version == 4 and str(parsed) == value


def _preparation_backrefs_well_formed(value: Any) -> bool:
    """Validate the closed journal representation before consumers use it."""
    if not isinstance(value, list):
        return False
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "ledger_entry_id", "ledger_path", "archive_path",
        }:
            return False
        entry_id = row.get("ledger_entry_id")
        ledger_path = row.get("ledger_path")
        archive_path = row.get("archive_path")
        if (
            not _is_uuid4(entry_id)
            or entry_id in seen
            or not isinstance(ledger_path, str)
            or not Path(ledger_path).is_absolute()
            or not isinstance(archive_path, str)
            or not Path(archive_path).is_absolute()
        ):
            return False
        seen.add(entry_id)
    return True


def _preparation_failure_backrefs_by_cell(
    *, output_root: Path, cells: list[dict], source_commit: str,
) -> dict[str, list[dict]]:
    ledger_path = (
        output_root.resolve()
        / prep_archive.PREPARATION_FAILURE_ROOT_NAME
        / prep_archive.PREPARATION_FAILURE_LEDGER_NAME
    )
    try:
        rows = prep_archive.load_preparation_failure_ledger(
            ledger_path, verify_archives=True,
        )
    except prep_archive.PreparationArchiveRefusal as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    out: dict[str, list[dict]] = {}
    for cell in cells:
        safe_name = _cell_safe_name(cell)
        selected_rows = [
            row for row in rows if row.get("cell_id") == safe_name
        ]
        for row in selected_rows:
            replacement_head = row.get("replacement_head")
            ancestry = subprocess.run(
                [
                    "git", "merge-base", "--is-ancestor",
                    str(replacement_head), source_commit,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if ancestry.returncode == 0:
                continue
            if ancestry.returncode == 1:
                detail = (
                    f"{safe_name} preparation failure replacement HEAD "
                    f"{replacement_head!r} is not an ancestor of "
                    f"{source_commit!r}"
                )
            else:
                diagnostic = ancestry.stderr.strip() or ancestry.stdout.strip()
                detail = (
                    f"{safe_name} preparation failure ancestry cannot be "
                    f"verified for replacement HEAD {replacement_head!r}: "
                    f"{diagnostic or f'git exit {ancestry.returncode}'}"
                )
            raise ChunkRefusal("PREPARATION_LEDGER_INCONSISTENT", detail)
        backrefs = prep_archive.preparation_failure_backrefs(
            rows, cell_id=safe_name, ledger_path=ledger_path,
        )
        if not _preparation_backrefs_well_formed(backrefs):
            raise ChunkRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"malformed preparation backrefs for {safe_name}",
            )
        out[safe_name] = backrefs
    return out


def _require_active_preparation_backrefs(
    active: dict[str, tuple[dict, dict | None]],
    expected: dict[str, list[dict]],
) -> None:
    mismatches = {
        safe_name: {
            "journal": start.get("prior_preparation_failures"),
            "ledger": expected.get(safe_name, []),
        }
        for safe_name, (start, _result) in active.items()
        if start.get("prior_preparation_failures")
        != expected.get(safe_name, [])
    }
    if mismatches:
        raise ChunkRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", repr(mismatches)
        )


def _require_preparation_attempt_available(
    *, safe_name: str, backrefs: list[dict], allow_overflow: bool,
) -> None:
    if len(backrefs) >= prep_archive.PREPARATION_FAILURE_LIMIT and not allow_overflow:
        raise ChunkRefusal(
            "PREPARATION_ATTEMPT_LIMIT",
            f"cell={safe_name}; prior_failures={len(backrefs)}; use "
            "--override-preparation-attempt-limit only after explicit escalation",
        )


def _capture_defect_backrefs_well_formed(value: Any) -> bool:
    """Validate the distinct, closed capture-defect back-reference shape."""
    if not isinstance(value, list):
        return False
    seen: set[str] = set()
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != capture_defect_archive.BACKREF_FIELDS
        ):
            return False
        entry_id = row.get("ledger_entry_id")
        ledger_path = row.get("ledger_path")
        archive_path = row.get("archive_path")
        original_head = row.get("original_head")
        if (
            not _is_uuid4(entry_id) or entry_id in seen
            or not isinstance(ledger_path, str)
            or not Path(ledger_path).is_absolute()
            or Path(ledger_path).name != CAPTURE_DEFECT_LEDGER_NAME
            or not isinstance(archive_path, str)
            or not Path(archive_path).is_absolute()
            or Path(archive_path).parent != Path(ledger_path).parent
            or Path(archive_path).name != f"capture-defect-{entry_id}"
            or not isinstance(original_head, str) or len(original_head) != 40
            or any(ch not in _SHA40 for ch in original_head)
            or row.get("reason_code") != capture_defect_archive.REASON_CODE
        ):
            return False
        seen.add(entry_id)
    return True


def _capture_defect_state(
    *, output_root: Path, cells: list[dict], source_commit: str,
) -> dict[str, list[dict]]:
    """Close the archive ledger and reject every foreign active journal.

    The scan is global to the campaign root and runs before referenced
    preflight validation or any GPU phase.  A closed archive has moved its
    source ``journals`` parent, so a ledgered source path reappearing is also a
    reuse/duplication defect rather than an admissible active journal.
    """
    ledger_path = (
        output_root.resolve() / capture_defect_archive.CAPTURE_DEFECT_ROOT_NAME
        / capture_defect_archive.CAPTURE_DEFECT_LEDGER_NAME
    )
    try:
        rows = capture_defect_archive.load_capture_defect_ledger(
            ledger_path, verify_archives=True,
        )
    except capture_defect_archive.CaptureDefectRefusal as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc

    unresolved = capture_defect_archive.unresolved_foreign_journals(
        output_root=output_root, current_head=source_commit, ledger_rows=rows,
    )
    if unresolved:
        raise ChunkRefusal(
            "CROSS_HEAD_ACTIVE_JOURNAL_UNRESOLVED", "; ".join(unresolved),
        )

    out: dict[str, list[dict]] = {}
    for cell in cells:
        safe_name = _cell_safe_name(cell)
        selected = [row for row in rows if row.get("cell_id") == safe_name]
        for row in selected:
            for field in ("original_head", "replacement_head"):
                ancestor = subprocess.run(
                    [
                        "git", "merge-base", "--is-ancestor",
                        str(row.get(field)), source_commit,
                    ],
                    cwd=REPO_ROOT, capture_output=True, text=True, check=False,
                )
                if ancestor.returncode != 0:
                    diagnostic = ancestor.stderr.strip() or ancestor.stdout.strip()
                    raise ChunkRefusal(
                        "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                        f"{safe_name} {field}={row.get(field)!r} is not an "
                        f"ancestor of {source_commit!r}: "
                        f"{diagnostic or f'git exit {ancestor.returncode}'}",
                    )
        refs = capture_defect_archive.capture_defect_backrefs(
            rows, cell_id=safe_name, ledger_path=ledger_path,
        )
        if not _capture_defect_backrefs_well_formed(refs):
            raise ChunkRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                f"malformed capture-defect backrefs for {safe_name}",
            )
        out[safe_name] = refs
    return out


def _require_active_capture_defect_backrefs(
    active: dict[str, tuple[dict, dict | None]],
    expected: dict[str, list[dict]],
) -> None:
    mismatches = {
        safe_name: {
            "journal": start.get("prior_capture_defect_attempts"),
            "ledger": expected.get(safe_name, []),
        }
        for safe_name, (start, _result) in active.items()
        if start.get("prior_capture_defect_attempts")
        != expected.get(safe_name, [])
    }
    if mismatches:
        raise ChunkRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", repr(mismatches),
        )


def _utc_text(value: datetime | None = None) -> str:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise ChunkRefusal("timestamp_invalid", repr(observed))
    return (
        observed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive_fsync(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ChunkRefusal("capture_evidence_exists", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _append_jsonl_fsync(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _read_jsonl(path: Path, *, error_code: str) -> list[dict]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ChunkRefusal(error_code, str(path))
    rows: list[dict] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as exc:
        raise ChunkRefusal(error_code, f"{path}: {exc}") from exc
    for line_no, raw in enumerate(lines, start=1):
        if not raw:
            raise ChunkRefusal(error_code, f"{path}:{line_no}: empty row")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ChunkRefusal(error_code, f"{path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ChunkRefusal(error_code, f"{path}:{line_no}: not an object")
        rows.append(row)
    return rows


def _journal_start_payload(
    *, cell: dict, matrix_sha256: str, matrix_seal_sha256: str,
    preflight_sha256: str, preflight_path: Path, source_commit: str,
    capture_started_at_utc: str, admission_run_id: str,
    supersedes: dict | None, prior_preparation_failures: list[dict],
    prior_capture_defect_attempts: list[dict],
) -> dict:
    if not _preparation_backrefs_well_formed(prior_preparation_failures):
        raise ChunkRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            "prior_preparation_failures is malformed",
        )
    if not _capture_defect_backrefs_well_formed(
        prior_capture_defect_attempts
    ):
        raise ChunkRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            "prior_capture_defect_attempts is malformed",
        )
    payload = {
        "schema_version": 2,
        "state": "capture_started",
        "identity": cell["identity"],
        "authority_cell_sha256": cell["authority_cell_sha256"],
        "source_commit": source_commit,
        "capture_matrix_sha256": matrix_sha256,
        "matrix_seal_sha256": matrix_seal_sha256,
        "capture_preflight_sha256": preflight_sha256,
        "capture_preflight_path": str(preflight_path.resolve()),
        "capture_started_at_utc": capture_started_at_utc,
        "admission_run_id": admission_run_id,
        "supersedes": supersedes,
        "prior_preparation_failures": prior_preparation_failures,
        "prior_capture_defect_attempts": prior_capture_defect_attempts,
        "definition_equivalence_rule_id": (
            sol_def.CERTIFYING_RULE_ID
            if cell["identity"]["roster_source"] == "sol_execbench"
            else None
        ),
    }
    payload["journal_start_seal_sha256"] = canonical_sha256(payload)
    return payload


def _journal_payload(*, start: dict, result: dict) -> dict:
    payload = {
        key: value for key, value in start.items()
        if key != "journal_start_seal_sha256"
    }
    payload.update({"state": "complete", "result": result})
    payload["journal_seal_sha256"] = canonical_sha256(payload)
    return payload


def _write_cell_journal(path: Path, payload: dict) -> None:
    _write_json_exclusive_fsync(path, payload)


def _verify_sealed_object(
    payload: dict, *, seal_field: str, error_code: str, path: Path,
) -> None:
    seal = payload.get(seal_field)
    unsealed = {key: value for key, value in payload.items() if key != seal_field}
    if seal != canonical_sha256(unsealed):
        raise ChunkRefusal(error_code, str(path))


def _verify_journal_artifacts(result: dict, *, root: Path, safe_name: str) -> None:
    expected_prefix = f"cells/{safe_name}/"
    artifacts = result.get("artifacts") if isinstance(result, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise ChunkRefusal("cell_journal_artifacts_empty", safe_name)
    listed: set[str] = set()
    try:
        resolved_root = _resolve_output_path(
            root, code="cell_journal_storage_failure",
            operation="cell_journal_root_resolve",
        )
    except ChunkRefusal:
        raise
    except (OSError, RuntimeError) as exc:
        raise ChunkRefusal(
            "cell_journal_artifact_unavailable", f"{root}: {exc}"
        ) from exc
    for index, row in enumerate(artifacts):
        if not isinstance(row, dict):
            raise ChunkRefusal("cell_journal_artifact_invalid", f"{safe_name}:{index}")
        rel = row.get("relative_path")
        if (
            not isinstance(rel, str) or not rel.startswith(expected_prefix)
            or ".." in Path(rel).parts or rel in listed
        ):
            raise ChunkRefusal("cell_journal_artifact_path_invalid", repr(rel))
        listed.add(rel)
        candidate = root / rel
        observed = _output_lstat(
            candidate, code="cell_journal_storage_failure",
            operation="cell_journal_artifact_lstat",
        )
        if observed is None:
            raise ChunkRefusal(
                "cell_journal_artifact_unavailable", str(candidate)
            )
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise ChunkRefusal("cell_journal_artifact_not_regular", str(candidate))
        try:
            resolved = _resolve_output_path(
                candidate, code="cell_journal_storage_failure",
                operation="cell_journal_artifact_resolve",
            )
            resolved.relative_to(resolved_root)
        except ChunkRefusal:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise ChunkRefusal(
                "cell_journal_artifact_unavailable", f"{candidate}: {exc}"
            ) from exc
        if observed.st_size != row.get("size_bytes"):
            raise ChunkRefusal("cell_journal_artifact_size_mismatch", rel)
        if _sha256_output_file(
            resolved, code="cell_journal_storage_failure",
            operation="cell_journal_artifact_hash",
        ) != row.get("sha256"):
            raise ChunkRefusal("cell_journal_artifact_hash_mismatch", rel)
    cell_dir = root / "cells" / safe_name
    actual: set[str] = set()
    symlinks: list[str] = []
    for path in _output_tree_paths(
        cell_dir, code="cell_journal_storage_failure",
        operation="cell_journal_inventory_walk",
    ):
        observed = _output_lstat(
            path, code="cell_journal_storage_failure",
            operation="cell_journal_inventory_lstat",
        )
        if observed is None:
            raise ChunkRefusal("cell_journal_artifact_unavailable", str(path))
        if stat.S_ISLNK(observed.st_mode):
            symlinks.append(str(path))
        elif stat.S_ISREG(observed.st_mode):
            actual.add(path.relative_to(root).as_posix())
    if symlinks:
        raise ChunkRefusal("cell_journal_symlink_refused", repr(symlinks))
    if actual != listed:
        raise ChunkRefusal(
            "cell_journal_inventory_mismatch",
            f"unlisted={sorted(actual - listed)!r}, absent={sorted(listed - actual)!r}",
        )


def _read_cell_journal(
    *, journal_dir: Path, cell: dict, matrix_sha256: str,
    matrix_seal_sha256: str, source_commit: str, allow_incomplete: bool,
) -> tuple[dict, dict | None]:
    if journal_dir.is_symlink() or not journal_dir.is_dir():
        raise ChunkRefusal("cell_journal_path_invalid", str(journal_dir))
    start_path = journal_dir / JOURNAL_START_NAME
    start = _json(start_path)
    _verify_sealed_object(
        start, seal_field="journal_start_seal_sha256",
        error_code="cell_journal_seal_mismatch", path=start_path,
    )
    expected = {
        "schema_version": 2,
        "state": "capture_started",
        "identity": cell["identity"],
        "authority_cell_sha256": cell["authority_cell_sha256"],
        "source_commit": source_commit,
        "capture_matrix_sha256": matrix_sha256,
        "matrix_seal_sha256": matrix_seal_sha256,
    }
    observed_rule_id = start.get("definition_equivalence_rule_id")
    if cell["identity"]["roster_source"] == "sol_execbench":
        if observed_rule_id != sol_def.CERTIFYING_RULE_ID:
            raise ChunkRefusal(
                "SOL_ATTESTATION_RULE_MISMATCH",
                f"journal start expected {sol_def.CERTIFYING_RULE_ID!r}, "
                f"observed {observed_rule_id!r}",
            )
    elif observed_rule_id is not None:
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "non-SOL journal start carries a definition-equivalence rule",
        )
    mismatches = [
        f"{field}={start.get(field)!r} expected={value!r}"
        for field, value in expected.items() if start.get(field) != value
    ]
    for field in ("capture_preflight_sha256", "admission_run_id"):
        value = start.get(field)
        if (
            not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            mismatches.append(f"{field} is not a sha256")
    preflight_path = start.get("capture_preflight_path")
    if not isinstance(preflight_path, str) or not Path(preflight_path).is_absolute():
        mismatches.append("capture_preflight_path is not absolute")
    try:
        _parse_utc_timestamp(
            start.get("capture_started_at_utc"), field="capture_started_at_utc"
        )
    except ChunkRefusal as exc:
        mismatches.append(exc.detail)
    supersedes = start.get("supersedes")
    if supersedes is not None and (
        not isinstance(supersedes, dict)
        or set(supersedes) != {
            "ledger_entry_id", "superseded_preflight_sha256", "superseded_path",
        }
    ):
        mismatches.append("supersedes is malformed")
    if not _preparation_backrefs_well_formed(
        start.get("prior_preparation_failures")
    ):
        mismatches.append("prior_preparation_failures is malformed")
    if not _capture_defect_backrefs_well_formed(
        start.get("prior_capture_defect_attempts")
    ):
        mismatches.append("prior_capture_defect_attempts is malformed")
    if mismatches:
        raise ChunkRefusal("cell_journal_input_mismatch", "; ".join(mismatches))

    complete_path = journal_dir / JOURNAL_COMPLETE_NAME
    if not complete_path.exists():
        if allow_incomplete:
            return start, None
        raise ChunkRefusal("CELL_JOURNAL_INCOMPLETE", str(journal_dir))
    complete = _json(complete_path)
    _verify_sealed_object(
        complete, seal_field="journal_seal_sha256",
        error_code="cell_journal_seal_mismatch", path=complete_path,
    )
    header = {
        key: value for key, value in start.items()
        if key != "journal_start_seal_sha256"
    }
    header["state"] = "complete"
    for field, value in header.items():
        if complete.get(field) != value:
            raise ChunkRefusal(
                "cell_journal_input_mismatch",
                f"complete {field}={complete.get(field)!r} expected={value!r}",
            )
    result = complete.get("result")
    safe_name = _cell_safe_name(cell)
    _verify_journal_artifacts(result, root=journal_dir, safe_name=safe_name)
    result_fields = {
        "capture_preflight_sha256": start["capture_preflight_sha256"],
        "capture_started_at_utc": start["capture_started_at_utc"],
        "admission_run_id": start["admission_run_id"],
        "supersedes": start["supersedes"],
        "prior_preparation_failures": start[
            "prior_preparation_failures"
        ],
        "prior_capture_defect_attempts": start[
            "prior_capture_defect_attempts"
        ],
    }
    if any(result.get(field) != value for field, value in result_fields.items()):
        raise ChunkRefusal(
            "cell_journal_result_mismatch", repr(result_fields)
        )
    cell_artifacts = [
        row for row in result.get("artifacts", [])
        if row.get("artifact_kind") == "cell_result"
    ]
    if len(cell_artifacts) != 1:
        raise ChunkRefusal("cell_journal_result_mismatch", "cell_result count")
    cell_result = _json(journal_dir / cell_artifacts[0]["relative_path"])
    if any(cell_result.get(field) != value for field, value in result_fields.items()):
        raise ChunkRefusal(
            "cell_journal_result_mismatch", "sealed cell result metadata drift"
        )
    expected_rule_id = start["definition_equivalence_rule_id"]
    definition_fields = {
        "definition_equivalence",
        "definition_equivalence_attestation_sha256",
    }
    if not definition_fields <= set(result) or not definition_fields <= set(
        cell_result
    ):
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "journal result or cell result omits definition attestation fields",
        )
    definition_equivalence = result.get("definition_equivalence")
    definition_equivalence_sha256 = result.get(
        "definition_equivalence_attestation_sha256"
    )
    if expected_rule_id is None:
        if (
            definition_equivalence is not None
            or definition_equivalence_sha256 is not None
            or any(
                cell_result.get(field) is not None
                for field in definition_fields
            )
        ):
            raise ChunkRefusal(
                "SOL_ATTESTATION_STATE_MISMATCH",
                "non-SOL evidence carries a definition attestation",
            )
        return start, result

    attestations = {
        "journal result": definition_equivalence,
        "cell result": cell_result.get("definition_equivalence"),
    }
    attestation_hashes = {
        "journal result": definition_equivalence_sha256,
        "cell result": cell_result.get(
            "definition_equivalence_attestation_sha256"
        ),
    }
    for label, attestation in attestations.items():
        try:
            sol_def.validate_attestation_shape_and_eligibility(attestation)
            serialized_attestation = sol_def.serialize_attestation(attestation)
        except sol_def.SolDefinitionEquivalenceError as exc:
            raise ChunkRefusal(exc.code, f"{label}: {exc.detail}") from exc
        if attestation.get("rule_id") != expected_rule_id:
            raise ChunkRefusal(
                "SOL_ATTESTATION_RULE_MISMATCH",
                f"{label} rule does not match journal start",
            )
        if (
            hashlib.sha256(serialized_attestation).hexdigest()
            != attestation_hashes[label]
        ):
            raise ChunkRefusal(
                "SOL_ATTESTATION_STATE_MISMATCH",
                f"{label} attestation SHA does not rederive",
            )

    stage_evidence = _validate_sol_prepare_manifest(
        cell=cell,
        invocation=cell["invocation"],
        stage=journal_dir / "cells" / safe_name / "sol_stage",
    )
    if any(
        attestation != stage_evidence["definition_equivalence"]
        or attestation_hashes[label]
        != stage_evidence["definition_equivalence_attestation_sha256"]
        for label, attestation in attestations.items()
    ):
        raise ChunkRefusal(
            "SOL_ATTESTATION_STATE_MISMATCH",
            "journal result, cell result, and sealed stage disagree",
        )
    return start, result


def _load_or_recover_cell_journal(
    *, journal_dir: Path, cell: dict, matrix_sha256: str,
    matrix_seal_sha256: str, preflight_sha256: str, source_commit: str,
    current_invocation_admitted: bool,
    original_preflight_status: str | None = None,
    current_preflight_path: Path | None = None,
) -> dict | None:
    if not journal_dir.exists():
        return None
    start, result = _read_cell_journal(
        journal_dir=journal_dir, cell=cell, matrix_sha256=matrix_sha256,
        matrix_seal_sha256=matrix_seal_sha256, source_commit=source_commit,
        allow_incomplete=False,
    )
    if not current_invocation_admitted:
        raise ChunkRefusal(
            "JOURNAL_RECOVERY_NOT_ADMITTED", str(journal_dir)
        )
    original_sha = start["capture_preflight_sha256"]
    if original_sha != preflight_sha256:
        detail = (
            f"journal={journal_dir}; original_preflight_path="
            f"{start['capture_preflight_path']}; original_sha256={original_sha}; "
            f"current_preflight_path={current_preflight_path}; "
            f"current_sha256={preflight_sha256}"
        )
        if original_preflight_status == "fresh":
            raise ChunkRefusal(
                "JOURNAL_PREFLIGHT_MISMATCH",
                detail + "; re-present the original preflight",
            )
        raise ChunkRefusal("JOURNAL_PREFLIGHT_SUPERSEDABLE", detail)
    return result


def _content_inventory(root: Path) -> list[dict]:
    if root.is_symlink() or not root.is_dir():
        raise ChunkRefusal("SUPERSESSION_INVENTORY_INVALID", str(root))
    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ChunkRefusal("SUPERSESSION_INVENTORY_INVALID", str(path))
        if (
            not path.is_file()
            or path == root / SUPERSESSION_INVENTORY_NAME
        ):
            continue
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    if not rows:
        raise ChunkRefusal("SUPERSESSION_INVENTORY_INVALID", str(root))
    return rows


def _inventory_document(rows: list[dict]) -> dict:
    return {"schema_version": 1, "files": rows}


def _verify_inventory(root: Path, expected: list[dict]) -> None:
    observed = _content_inventory(root)
    if observed != expected:
        raise ChunkRefusal(
            "SUPERSESSION_INVENTORY_MISMATCH",
            f"root={root}; expected={expected!r}; observed={observed!r}",
        )


_SUPERSESSION_LEDGER_FIELDS = {
    "schema_version", "timestamp_utc", "session_id", "chunk_id", "cell_id",
    "original_preflight_sha256", "current_preflight_sha256", "source_path",
    "destination_path", "reason_code", "operator_flag", "source_commit",
    "matrix_seal_sha256", "inventory_sha256", "ledger_entry_id",
}


def _load_supersession_ledger(path: Path) -> list[dict]:
    rows = _read_jsonl(path, error_code="SUPERSESSION_LEDGER_INCONSISTENT")
    seen: set[str] = set()
    for line_no, row in enumerate(rows, start=1):
        if set(row) != _SUPERSESSION_LEDGER_FIELDS or row.get("schema_version") != 1:
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT",
                f"{path}:{line_no}: fields/schema mismatch",
            )
        entry_id = row.get("ledger_entry_id")
        if (
            not isinstance(entry_id, str) or len(entry_id) != 64
            or any(ch not in "0123456789abcdef" for ch in entry_id)
            or entry_id in seen
        ):
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT",
                f"{path}:{line_no}: ledger_entry_id invalid",
            )
        seen.add(entry_id)
    return rows


def _validate_superseded_tree(superseded_root: Path) -> list[dict]:
    if not superseded_root.exists():
        return []
    if superseded_root.is_symlink() or not superseded_root.is_dir():
        raise ChunkRefusal(
            "SUPERSESSION_LEDGER_INCONSISTENT", str(superseded_root)
        )
    ledger = _load_supersession_ledger(
        superseded_root / SUPERSESSION_LEDGER_NAME
    )
    expected: dict[Path, dict] = {}
    for row in ledger:
        destination = Path(row["destination_path"])
        try:
            resolved = destination.resolve(strict=True)
            resolved.relative_to(superseded_root.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT",
                f"destination={destination}: {exc}",
            ) from exc
        if resolved.parent.parent != superseded_root.resolve():
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT",
                f"destination depth is invalid: {resolved}",
            )
        if resolved in expected:
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT",
                f"duplicate destination: {resolved}",
            )
        expected[resolved] = row
        inventory_path = resolved / SUPERSESSION_INVENTORY_NAME
        if (
            not inventory_path.is_file()
            or sha256_file(inventory_path) != row["inventory_sha256"]
        ):
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT",
                f"inventory mismatch: {inventory_path}",
            )
        inventory = _json(inventory_path)
        if (
            set(inventory) != {"schema_version", "files"}
            or inventory.get("schema_version") != 1
            or not isinstance(inventory.get("files"), list)
        ):
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT", str(inventory_path)
            )
        _verify_inventory(resolved, inventory["files"])
    actual: set[Path] = set()
    for group in superseded_root.iterdir():
        if group.name == SUPERSESSION_LEDGER_NAME:
            if group.is_symlink() or not group.is_file():
                raise ChunkRefusal(
                    "SUPERSESSION_LEDGER_INCONSISTENT", str(group)
                )
            continue
        if group.is_symlink() or not group.is_dir():
            raise ChunkRefusal(
                "SUPERSESSION_LEDGER_INCONSISTENT", str(group)
            )
        for cell_dir in group.iterdir():
            if cell_dir.is_symlink() or not cell_dir.is_dir():
                raise ChunkRefusal(
                    "SUPERSESSION_LEDGER_INCONSISTENT", str(cell_dir)
                )
            actual.add(cell_dir.resolve())
    if actual != set(expected):
        raise ChunkRefusal(
            "SUPERSESSION_LEDGER_INCONSISTENT",
            f"unledgered={sorted(map(str, actual-set(expected)))!r}; "
            f"missing={sorted(map(str, set(expected)-actual))!r}",
        )
    return ledger


def _active_journals(
    *, journals_root: Path, cells: list[dict], matrix_sha256: str,
    matrix_seal_sha256: str, source_commit: str,
) -> dict[str, tuple[dict, dict | None]]:
    expected = {_cell_safe_name(cell): cell for cell in cells}
    if not journals_root.exists():
        return {}
    if journals_root.is_symlink() or not journals_root.is_dir():
        raise ChunkRefusal("cell_journal_path_invalid", str(journals_root))
    observed = {path.name: path for path in journals_root.iterdir()}
    unexpected = sorted(set(observed) - set(expected))
    if unexpected:
        raise ChunkRefusal("cell_journal_unexpected", repr(unexpected))
    out: dict[str, tuple[dict, dict | None]] = {}
    for safe_name, path in observed.items():
        out[safe_name] = _read_cell_journal(
            journal_dir=path, cell=expected[safe_name],
            matrix_sha256=matrix_sha256,
            matrix_seal_sha256=matrix_seal_sha256,
            source_commit=source_commit, allow_incomplete=True,
        )
    return out


def _supersede_active_journals(
    *, journals_root: Path, superseded_root: Path,
    active: dict[str, tuple[dict, dict | None]], current_preflight_sha256: str,
    session_id: str, chunk_id: str, source_commit: str,
    matrix_seal_sha256: str, allow_overflow: bool,
) -> dict[str, dict]:
    if not active:
        return {}
    original_hashes = {
        start["capture_preflight_sha256"] for start, _result in active.values()
    }
    if len(original_hashes) != 1:
        raise ChunkRefusal(
            "DONE_PREFLIGHT_NONUNIFORM", repr(sorted(original_hashes))
        )
    original_sha = next(iter(original_hashes))
    superseded_root.mkdir(parents=True, exist_ok=True)
    if journals_root.stat().st_dev != superseded_root.stat().st_dev:
        raise ChunkRefusal(
            "SUPERSESSION_CROSS_DEVICE_REFUSED",
            f"journals={journals_root}; superseded={superseded_root}",
        )
    ledger_path = superseded_root / SUPERSESSION_LEDGER_NAME
    ledger = _load_supersession_ledger(ledger_path)
    counts: dict[str, int] = {}
    for row in ledger:
        counts[row["cell_id"]] = counts.get(row["cell_id"], 0) + 1
    over_limit = sorted(
        safe_name for safe_name in active if counts.get(safe_name, 0) >= 2
    )
    if over_limit and not allow_overflow:
        raise ChunkRefusal(
            "SUPERSESSION_LIMIT_EXCEEDED",
            f"cells={over_limit!r}; use --allow-supersession-overflow only "
            "after explicit escalation",
        )
    inventories = {
        safe_name: _content_inventory(journals_root / safe_name)
        for safe_name in sorted(active)
    }
    timestamp = datetime.now(timezone.utc)
    group_name = (
        timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        + f"__pf-{original_sha[:8]}__to-pf-{current_preflight_sha256[:8]}"
    )
    group = superseded_root / group_name
    if group.exists() or group.is_symlink():
        raise ChunkRefusal("SUPERSESSION_DESTINATION_EXISTS", str(group))
    source_paths = {
        safe_name: str((journals_root / safe_name).resolve())
        for safe_name in active
    }
    os.replace(journals_root, group)
    journals_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(superseded_root)
    _fsync_directory(journals_root.parent)

    backrefs: dict[str, dict] = {}
    operator_flag = "--supersede-stale-journals"
    if allow_overflow:
        operator_flag += " --allow-supersession-overflow"
    for safe_name in sorted(active):
        destination = group / safe_name
        _verify_inventory(destination, inventories[safe_name])
        inventory_path = destination / SUPERSESSION_INVENTORY_NAME
        _write_json_exclusive_fsync(
            inventory_path, _inventory_document(inventories[safe_name])
        )
        base_entry = {
            "schema_version": 1,
            "timestamp_utc": _utc_text(timestamp),
            "session_id": session_id,
            "chunk_id": chunk_id,
            "cell_id": safe_name,
            "original_preflight_sha256": original_sha,
            "current_preflight_sha256": current_preflight_sha256,
            "source_path": source_paths[safe_name],
            "destination_path": str(destination.resolve()),
            "reason_code": "PREFLIGHT_EXPIRED_BEFORE_DONE",
            "operator_flag": operator_flag,
            "source_commit": source_commit,
            "matrix_seal_sha256": matrix_seal_sha256,
            "inventory_sha256": sha256_file(inventory_path),
        }
        entry = {
            **base_entry,
            "ledger_entry_id": canonical_sha256(base_entry),
        }
        _append_jsonl_fsync(ledger_path, entry)
        backrefs[safe_name] = {
            "ledger_entry_id": entry["ledger_entry_id"],
            "superseded_preflight_sha256": original_sha,
            "superseded_path": entry["destination_path"],
        }
    return backrefs


def _latest_supersession_backrefs(
    superseded_root: Path, *, current_preflight_sha256: str,
) -> dict[str, dict]:
    ledger = _load_supersession_ledger(
        superseded_root / SUPERSESSION_LEDGER_NAME
    )
    out: dict[str, dict] = {}
    for row in ledger:
        if row["current_preflight_sha256"] != current_preflight_sha256:
            continue
        out[row["cell_id"]] = {
            "ledger_entry_id": row["ledger_entry_id"],
            "superseded_preflight_sha256": row[
                "original_preflight_sha256"
            ],
            "superseded_path": row["destination_path"],
        }
    return out


def _chunk_session(
    run_journal_path: Path, *, chunk_id: str, source_commit: str,
    matrix_sha256: str, matrix_seal_sha256: str,
) -> tuple[str, list[dict]]:
    rows = _read_jsonl(
        run_journal_path, error_code="CHUNK_RUN_JOURNAL_INVALID"
    )
    expected = {
        "event": "session_started", "schema_version": 1,
        "chunk_id": chunk_id, "source_commit": source_commit,
        "capture_matrix_sha256": matrix_sha256,
        "matrix_seal_sha256": matrix_seal_sha256,
    }
    if not rows:
        session_id = str(uuid.uuid4())
        first = {
            **expected, "session_id": session_id,
            "timestamp_utc": _utc_text(),
        }
        _append_jsonl_fsync(run_journal_path, first)
        return session_id, [first]
    first = rows[0]
    if any(first.get(field) != value for field, value in expected.items()):
        raise ChunkRefusal("CHUNK_RUN_JOURNAL_INVALID", repr(first))
    session_id = first.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ChunkRefusal("CHUNK_RUN_JOURNAL_INVALID", "session_id")
    seen_run_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("session_id") != session_id:
            raise ChunkRefusal(
                "CHUNK_RUN_JOURNAL_INVALID", "mixed session ids"
            )
        if row.get("chunk_id") != chunk_id:
            raise ChunkRefusal(
                "CHUNK_RUN_JOURNAL_INVALID", "mixed chunk ids"
            )
        event = row.get("event")
        if index == 0:
            continue
        if event == "capture_admission":
            required = {
                "schema_version", "event", "session_id", "chunk_id", "run_id",
                "admitted_at_utc", "preflight_path", "preflight_sha256",
                "preflight_generated_at_utc",
            }
            run_id = row.get("run_id")
            preflight_sha = row.get("preflight_sha256")
            if (
                set(row) != required or row.get("schema_version") != 1
                or not isinstance(run_id, str) or len(run_id) != 64
                or any(ch not in "0123456789abcdef" for ch in run_id)
                or run_id in seen_run_ids
                or not isinstance(preflight_sha, str) or len(preflight_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in preflight_sha)
                or not isinstance(row.get("preflight_path"), str)
                or not Path(row["preflight_path"]).is_absolute()
            ):
                raise ChunkRefusal(
                    "CHUNK_RUN_JOURNAL_INVALID", f"capture admission row {index}"
                )
            seen_run_ids.add(run_id)
            admitted = _parse_utc_timestamp(
                row.get("admitted_at_utc"), field="admitted_at_utc"
            )
            generated = _parse_utc_timestamp(
                row.get("preflight_generated_at_utc"),
                field="preflight_generated_at_utc",
            )
            age = (admitted - generated).total_seconds()
            if (
                age > preflight.PREFLIGHT_FRESHNESS_SECONDS
                or (generated - admitted).total_seconds()
                > preflight.PREFLIGHT_FUTURE_SKEW_SECONDS
            ):
                raise ChunkRefusal(
                    "CHUNK_RUN_JOURNAL_INVALID",
                    f"capture admission row {index} violates freshness",
                )
        elif event == "finalization_probe":
            required = {
                "schema_version", "event", "session_id", "chunk_id", "run_id",
                "observed_at_utc", "certifying", "sol_environment",
            }
            if (
                set(row) != required or row.get("schema_version") != 1
                or row.get("certifying") is not False
                or not isinstance(row.get("sol_environment"), dict)
            ):
                raise ChunkRefusal(
                    "CHUNK_RUN_JOURNAL_INVALID", f"finalization row {index}"
                )
            _parse_utc_timestamp(
                row.get("observed_at_utc"), field="observed_at_utc"
            )
        else:
            raise ChunkRefusal(
                "CHUNK_RUN_JOURNAL_INVALID", f"unknown event at row {index}"
            )
    return session_id, rows


def _record_capture_admission(
    run_journal_path: Path, *, session_id: str, chunk_id: str,
    run_id: str, admitted_at_utc: str, preflight_path: Path,
    preflight_sha256: str, preflight_generated_at_utc: str,
) -> dict:
    row = {
        "schema_version": 1,
        "event": "capture_admission",
        "session_id": session_id,
        "chunk_id": chunk_id,
        "run_id": run_id,
        "admitted_at_utc": admitted_at_utc,
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha256,
        "preflight_generated_at_utc": preflight_generated_at_utc,
    }
    _append_jsonl_fsync(run_journal_path, row)
    return row


def _admission_object(
    rows: list[dict], *, preflight_sha256: str,
    preflight_generated_at_utc: str,
) -> dict:
    invocations = [
        {
            "run_id": row["run_id"],
            "admitted_at_utc": row["admitted_at_utc"],
        }
        for row in rows
        if row.get("event") == "capture_admission"
        and row.get("preflight_sha256") == preflight_sha256
    ]
    if not invocations:
        raise ChunkRefusal(
            "DONE_ADMISSION_MISSING", preflight_sha256
        )
    return {
        "preflight_generated_at_utc": preflight_generated_at_utc,
        "capture_invocations": invocations,
    }


def _validate_referenced_preflight(
    *, preflight_path: Path, expected_sha256: str, pins: dict,
    pins_path: Path, aggregate_budget_path: Path, matrix: dict,
    matrix_path: Path, matrix_sha256: str, output_root: Path,
    expected_head: str, require_fresh: bool,
    admitted_at_utc: datetime | None = None,
) -> tuple[dict, ncu_executor.ExpectedNcuIdentity | None]:
    if (
        not preflight_path.is_absolute() or preflight_path.is_symlink()
        or not preflight_path.is_file()
    ):
        raise ChunkRefusal(
            "REFERENCED_PREFLIGHT_INVALID", str(preflight_path)
        )
    observed_sha = sha256_file(preflight_path)
    if observed_sha != expected_sha256:
        raise ChunkRefusal(
            "REFERENCED_PREFLIGHT_INVALID",
            f"path={preflight_path}; expected_sha256={expected_sha256}; "
            f"observed_sha256={observed_sha}",
        )
    manifest = _json(preflight_path)
    expected_ncu_identity = validate_preflight_manifest(
        manifest, expected_head=expected_head,
        expected_pins_sha256=sha256_file(pins_path),
        expected_output_root=output_root,
        expected_ncu_version_prefix=pins.get("toolchain", {}).get(
            "ncu_version_prefix"
        ),
        expected_budget_sha256=sha256_file(aggregate_budget_path),
        expected_arch=matrix["arch"], expected_sku=matrix["sku"],
        expected_protocol=pins["measurement_protocol"]["version"],
        expected_matrix_sha256=matrix_sha256,
        expected_sol_fingerprint_pin=matrix["input_pins"][
            "sol_native_env_fingerprint"
        ],
        require_fresh=require_fresh,
        admitted_at_utc=admitted_at_utc,
    )
    validate_live_injection_hashes(manifest, pins)
    return manifest, expected_ncu_identity


def _original_preflight_status(
    *, start: dict, current_preflight_sha256: str,
    current_preflight_path: Path, pins: dict,
    pins_path: Path, aggregate_budget_path: Path, matrix: dict,
    matrix_path: Path, matrix_sha256: str, output_root: Path,
    expected_head: str, admitted_at_utc: datetime,
) -> tuple[str, str]:
    original_path = Path(start["capture_preflight_path"])
    original_sha = start["capture_preflight_sha256"]
    detail = (
        f"original_path={original_path}; original_sha256={original_sha}; "
        f"current_path={current_preflight_path.resolve()}; "
        f"current_sha256={current_preflight_sha256}"
    )
    try:
        _validate_referenced_preflight(
            preflight_path=original_path, expected_sha256=original_sha,
            pins=pins, pins_path=pins_path,
            aggregate_budget_path=aggregate_budget_path, matrix=matrix,
            matrix_path=matrix_path, matrix_sha256=matrix_sha256,
            output_root=output_root, expected_head=expected_head,
            require_fresh=True, admitted_at_utc=admitted_at_utc,
        )
    except ChunkRefusal as exc:
        return "supersedable", detail + f"; original_status={exc.code}: {exc.detail}"
    return "fresh", detail


def _record_finalization_probe(
    *, run_journal_path: Path, session_id: str, chunk_id: str,
    matrix: dict, matrix_path: Path,
) -> None:
    try:
        probe = sol_env.probe_sol_environment(
            matrix, matrix_path=matrix_path, ambient=os.environ,
        )
    except Exception as exc:  # additive, explicitly non-certifying evidence
        probe = {
            "status": "failed",
            "typed_error": "SOL_ENV_FINALIZATION_PROBE_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    _append_jsonl_fsync(run_journal_path, {
        "schema_version": 1,
        "event": "finalization_probe",
        "session_id": session_id,
        "chunk_id": chunk_id,
        "run_id": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        "observed_at_utc": _utc_text(),
        "certifying": False,
        "sol_environment": probe,
    })


def _historical_chunk_matches(
    manifest: dict, *, cells: list[dict], chunk_id: str,
) -> list[str]:
    problems: list[str] = []
    if manifest.get("chunk_id") != chunk_id:
        problems.append(f"chunk_id={manifest.get('chunk_id')!r} != {chunk_id!r}")
    expected = {
        (cell["identity"]["roster_source"], cell["identity"]["slug"]):
        cell["authority_cell_sha256"] for cell in cells
    }
    observed: dict[tuple[str, str], dict] = {}
    for row in manifest.get("cell_results") or []:
        if not isinstance(row, dict):
            continue
        key = (row.get("roster_source"), row.get("slug"))
        if key in observed:
            problems.append(f"duplicate historical result {key!r}")
        observed[key] = row
    if set(observed) != set(expected):
        problems.append(
            f"historical/current membership differs: missing={sorted(set(expected)-set(observed))!r} "
            f"extra={sorted(set(observed)-set(expected))!r}"
        )
    for key in sorted(set(expected) & set(observed)):
        row = observed[key]
        if row.get("authority_cell_sha256") != expected[key]:
            problems.append(f"historical authority hash drift for {key!r}")
        terminal = (
            row.get("disposition") == "admitted"
            or (
                row.get("disposition") == "failed"
                and row.get("typed_reason") == "trace_unsupported"
            )
        )
        if not terminal:
            problems.append(
                f"historical result for {key!r} is "
                f"{row.get('disposition')!r}/{row.get('typed_reason')!r}"
            )
    return problems


def _finalize_from_journals(
    *, args: argparse.Namespace, cells: list[dict], active: dict,
    stage_root: Path, final_dir: Path, matrix: dict, matrix_path: Path,
    matrix_sha256: str, authority_path: Path, pins: dict, pins_path: Path,
    aggregate_budget_path: Path, output_root: Path, source_commit: str,
    run_journal_path: Path, run_rows: list[dict],
) -> Path:
    if set(active) != {_cell_safe_name(cell) for cell in cells} or any(
        result is None for _start, result in active.values()
    ):
        raise ChunkRefusal(
            "FINALIZATION_JOURNALS_INCOMPLETE", repr(sorted(active))
        )
    preflight_hashes = {
        start["capture_preflight_sha256"] for start, _result in active.values()
    }
    preflight_paths = {
        start["capture_preflight_path"] for start, _result in active.values()
    }
    if len(preflight_hashes) != 1 or len(preflight_paths) != 1:
        raise ChunkRefusal(
            "DONE_PREFLIGHT_NONUNIFORM",
            f"hashes={sorted(preflight_hashes)!r}; paths={sorted(preflight_paths)!r}",
        )
    preflight_sha = next(iter(preflight_hashes))
    original_preflight_path = Path(next(iter(preflight_paths)))
    original_preflight, _expected_ncu_identity = _validate_referenced_preflight(
        preflight_path=original_preflight_path,
        expected_sha256=preflight_sha, pins=pins, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path, matrix=matrix,
        matrix_path=matrix_path, matrix_sha256=matrix_sha256,
        output_root=output_root, expected_head=source_commit,
        require_fresh=False,
    )
    admission = _admission_object(
        run_rows, preflight_sha256=preflight_sha,
        preflight_generated_at_utc=original_preflight["generated_at_utc"],
    )
    invocation_ids = {
        row["run_id"] for row in admission["capture_invocations"]
    }
    for safe_name, (start, result) in active.items():
        if (
            start["admission_run_id"] not in invocation_ids
            or result["capture_preflight_sha256"] != preflight_sha
        ):
            raise ChunkRefusal(
                "DONE_ADMISSION_INCONSISTENT", safe_name
            )

    publication_root = (
        stage_root / "publications" / hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    )
    publication_root.mkdir(parents=True, exist_ok=False)
    results: list[dict] = []
    for cell in cells:
        safe_name = _cell_safe_name(cell)
        start, result = active[safe_name]
        source_cell = (stage_root / "journals" / safe_name / "cells" / safe_name)
        destination_cell = publication_root / "cells" / safe_name
        destination_cell.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_cell, destination_cell, symlinks=False)
        _verify_journal_artifacts(
            result, root=publication_root, safe_name=safe_name,
        )
        results.append(result)

    head_post, clean_post, dirty_post = _git_state()
    if head_post != source_commit or not clean_post:
        raise ChunkRefusal(
            "source_state_changed",
            f"pre={source_commit}, post={head_post}, dirty={dirty_post!r}",
        )
    import campaign_done_manifest as done  # noqa: E402
    membership = [
        {
            "roster_source": cell["identity"]["roster_source"],
            "slug": cell["identity"]["slug"],
            "authority_cell_sha256": cell["authority_cell_sha256"],
        }
        for cell in cells
    ]
    manifest = done.build_done_manifest(
        command=shlex.join(sys.argv),
        source_commit_pre=source_commit,
        source_commit_post=head_post,
        working_tree_clean=True,
        tool_versions=done._tool_versions_from_pins(pins),
        output_root=str(final_dir),
        expected_cells=len(cells),
        chunk_membership=membership,
        cell_results=results,
        pins_sha256=sha256_file(pins_path),
        roster_authority_sha256=sha256_file(authority_path),
        trace_budget_sha256=sha256_file(aggregate_budget_path),
        capture_matrix_sha256=matrix_sha256,
        matrix_seal_sha256=matrix["matrix_seal_sha256"],
        preflight_sha256=preflight_sha,
        admission=admission,
        input_paths={
            "pins": str(pins_path),
            "roster_authority": str(authority_path),
            "trace_budget": str(aggregate_budget_path),
            "capture_matrix": str(matrix_path),
            "preflight": str(original_preflight_path),
        },
        chunk_id=args.chunk_id,
        protocol_version=pins["measurement_protocol"]["version"],
        arch=pins["arch"],
        generated_at_utc=_utc_text(),
        repo_root=REPO_ROOT,
        verification_output_root=publication_root,
    )
    done._assert_schema_valid(manifest)
    problems = done.verify_seal(
        manifest, repo_root=REPO_ROOT,
        verification_output_root=publication_root,
        require_current_head=True, require_clean_tree=True,
    )
    if problems:
        raise ChunkRefusal("done_seal_invalid", "; ".join(problems))
    _write_json_exclusive_fsync(publication_root / "DONE.json", manifest)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(publication_root, final_dir)
    published_manifest = _json(final_dir / "DONE.json")
    resumed = done.verify_seal(
        published_manifest, repo_root=REPO_ROOT,
        require_current_head=True, require_clean_tree=True,
    )
    if resumed:
        raise ChunkRefusal("finalized_seal_invalid", "; ".join(resumed))
    return final_dir


def run_chunk(args: argparse.Namespace) -> Path:
    matrix_path = Path(args.matrix).resolve()
    authority_path = Path(args.authority).resolve()
    pins_path = Path(args.pins).resolve()
    aggregate_budget_path = Path(args.aggregate_budget).resolve()
    cell_budget_path = Path(args.cell_budgets).resolve()
    registry_path = Path(args.metric_registry).resolve()
    preflight_path = Path(args.preflight).resolve()
    output_root = Path(args.output_root)
    _assert_external_output_root(output_root)
    # Keep this descriptor live for the whole function. flock is released on
    # return, exception unwinding, or process exit.
    _campaign_lock = acquire_campaign_lock(output_root)

    matrix = _json(matrix_path)
    authority = validate_matrix_inputs(
        matrix, matrix_path=matrix_path, authority_path=authority_path,
        pins_path=pins_path, aggregate_budget_path=aggregate_budget_path,
        cell_budget_path=cell_budget_path, registry_path=registry_path,
    )
    chunk, cells = select_chunk(matrix, args.chunk_id)
    head_pre, clean_pre, dirty = _git_state()
    if head_pre != matrix["source_commit"] or not clean_pre:
        raise ChunkRefusal(
            "source_state_invalid",
            f"HEAD={head_pre}, expected={matrix['source_commit']}, dirty={dirty!r}",
        )
    capture_defects_by_cell = _capture_defect_state(
        output_root=output_root, cells=cells, source_commit=head_pre,
    )
    final_dir = output_root / "chunks" / args.chunk_id
    if final_dir.exists():
        import campaign_done_manifest as done  # noqa: E402
        done_path = final_dir / "DONE.json"
        if not done_path.is_file():
            raise ChunkRefusal(
                "finalized_chunk_missing_done", f"{final_dir} has no DONE.json"
            )
        resume_manifest = _json(done_path)
        resume_problems = done.verify_seal(
            resume_manifest, repo_root=REPO_ROOT, require_current_head=True,
            require_clean_tree=True,
        )
        resume_problems.extend(_historical_chunk_matches(
            resume_manifest, cells=cells, chunk_id=args.chunk_id,
        ))
        if not resume_problems:
            return final_dir
        historical_problems = done.verify_seal(
            resume_manifest, repo_root=REPO_ROOT, require_current_head=False,
            require_clean_tree=False,
        )
        historical_problems.extend(_historical_chunk_matches(
            resume_manifest, cells=cells, chunk_id=args.chunk_id,
        ))
        if not historical_problems:
            if done._is_historical_bootstrap(resume_manifest):
                return final_dir
            historical_problems.append(
                "historical non-bootstrap chunk must be revision-archived and "
                "recaptured at the current matrix HEAD"
            )
        raise ChunkRefusal(
            "finalized_chunk_resume_invalid",
            "current=" + "; ".join(resume_problems)
            + " | historical=" + "; ".join(historical_problems),
        )

    pins = _json(pins_path)
    matrix_sha = sha256_file(matrix_path)
    stage_root = (
        output_root / ".staging" / head_pre / args.chunk_id / matrix_sha
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    journals_root = stage_root / "journals"
    superseded_root = stage_root / "superseded"
    run_journal_path = stage_root / "chunk_run_journal.jsonl"
    _validate_superseded_tree(superseded_root)
    session_id, run_rows = _chunk_session(
        run_journal_path, chunk_id=args.chunk_id, source_commit=head_pre,
        matrix_sha256=matrix_sha,
        matrix_seal_sha256=matrix["matrix_seal_sha256"],
    )
    active = _active_journals(
        journals_root=journals_root, cells=cells,
        matrix_sha256=matrix_sha,
        matrix_seal_sha256=matrix["matrix_seal_sha256"],
        source_commit=head_pre,
    )
    preparation_failures_by_cell = _preparation_failure_backrefs_by_cell(
        output_root=output_root, cells=cells, source_commit=head_pre,
    )
    _require_active_preparation_backrefs(
        active, preparation_failures_by_cell,
    )
    _require_active_capture_defect_backrefs(
        active, capture_defects_by_cell,
    )
    all_complete = (
        set(active) == {_cell_safe_name(cell) for cell in cells}
        and all(result is not None for _start, result in active.values())
    )
    if all_complete:
        _record_finalization_probe(
            run_journal_path=run_journal_path, session_id=session_id,
            chunk_id=args.chunk_id, matrix=matrix, matrix_path=matrix_path,
        )
        run_rows = _read_jsonl(
            run_journal_path, error_code="CHUNK_RUN_JOURNAL_INVALID"
        )
        return _finalize_from_journals(
            args=args, cells=cells, active=active, stage_root=stage_root,
            final_dir=final_dir, matrix=matrix, matrix_path=matrix_path,
            matrix_sha256=matrix_sha, authority_path=authority_path,
            pins=pins, pins_path=pins_path,
            aggregate_budget_path=aggregate_budget_path,
            output_root=output_root, source_commit=head_pre,
            run_journal_path=run_journal_path, run_rows=run_rows,
        )

    admitted_at = datetime.now(timezone.utc)
    preflight_sha = sha256_file(preflight_path)
    preflight_manifest, expected_ncu_identity = _validate_referenced_preflight(
        preflight_path=preflight_path, expected_sha256=preflight_sha,
        pins=pins, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path, matrix=matrix,
        matrix_path=matrix_path, matrix_sha256=matrix_sha,
        output_root=output_root, expected_head=head_pre,
        require_fresh=True, admitted_at_utc=admitted_at,
    )
    try:
        sol_environment_probe, certified_sol_env = (
            sol_env.certify_sol_environment(
                matrix, matrix_path=matrix_path, ambient=os.environ,
            )
        )
    except sol_env.SolEnvironmentRefusal as exc:
        raise ChunkRefusal(exc.code, exc.detail) from exc
    sealed_sol_probe = preflight_manifest["checks"]["sol_environment"]
    stable_probe_fields = (
        "ambient", "environment_contract", "environment_contract_sha256",
        "constructed_environment_delta", "matrix_pin", "matrix_sha256",
        "observed_fingerprint", "compiler_identity", "nvcc_release",
        "nvcc_version_output",
    )
    drift = [
        name for name in stable_probe_fields
        if sol_environment_probe.get(name) != sealed_sol_probe.get(name)
    ]
    if drift:
        raise ChunkRefusal("SOL_ENV_PREFLIGHT_DRIFT", repr(drift))

    mismatched = {
        safe_name: pair for safe_name, pair in active.items()
        if pair[0]["capture_preflight_sha256"] != preflight_sha
    }
    if mismatched:
        classifications = [
            _original_preflight_status(
                start=start, current_preflight_sha256=preflight_sha,
                current_preflight_path=preflight_path,
                pins=pins, pins_path=pins_path,
                aggregate_budget_path=aggregate_budget_path, matrix=matrix,
                matrix_path=matrix_path, matrix_sha256=matrix_sha,
                output_root=output_root, expected_head=head_pre,
                admitted_at_utc=admitted_at,
            )
            for start, _result in mismatched.values()
        ]
        fresh_details = [
            detail for status, detail in classifications if status == "fresh"
        ]
        if fresh_details:
            raise ChunkRefusal(
                "JOURNAL_PREFLIGHT_MISMATCH",
                "; ".join(fresh_details)
                + "; re-present the original preflight",
            )
        supersedable_detail = "; ".join(
            detail for _status, detail in classifications
        )
        if not getattr(args, "supersede_stale_journals", False):
            raise ChunkRefusal(
                "JOURNAL_PREFLIGHT_SUPERSEDABLE", supersedable_detail
            )
        _supersede_active_journals(
            journals_root=journals_root, superseded_root=superseded_root,
            active=active, current_preflight_sha256=preflight_sha,
            session_id=session_id, chunk_id=args.chunk_id,
            source_commit=head_pre,
            matrix_seal_sha256=matrix["matrix_seal_sha256"],
            allow_overflow=getattr(args, "allow_supersession_overflow", False),
        )
        active = {}
    elif any(result is None for _start, result in active.values()):
        raise ChunkRefusal(
            "CELL_JOURNAL_INCOMPLETE",
            "an interrupted attempt under the current preflight cannot be "
            "recaptured in place; preserve it until explicit stale supersession",
        )

    # The preparation-attempt bound is a chunk admission condition.  Check
    # every journal that would be opened before the first cell can perform GPU
    # work; checking only inside the capture loop could capture an earlier cell
    # before a later cell's third-attempt refusal.
    for cell in cells:
        safe_name = _cell_safe_name(cell)
        if safe_name not in active:
            _require_preparation_attempt_available(
                safe_name=safe_name,
                backrefs=preparation_failures_by_cell.get(safe_name, []),
                allow_overflow=getattr(
                    args, "override_preparation_attempt_limit", False
                ),
            )

    run_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    admitted_at_text = _utc_text(admitted_at)
    admission_row = _record_capture_admission(
        run_journal_path, session_id=session_id, chunk_id=args.chunk_id,
        run_id=run_id, admitted_at_utc=admitted_at_text,
        preflight_path=preflight_path, preflight_sha256=preflight_sha,
        preflight_generated_at_utc=preflight_manifest["generated_at_utc"],
    )
    run_rows.append(admission_row)
    cell_budgets = load_trace_budgets(cell_budget_path)
    capture_provenance_base = {
        "protocol_version": pins["measurement_protocol"]["version"],
        "source_commit": head_pre,
        "capture_matrix_sha256": matrix_sha,
        "preflight_sha256": preflight_sha,
        "pins_sha256": sha256_file(pins_path),
        "trace_budget_sha256": sha256_file(cell_budget_path),
        "raw_staging_margin_bytes": matrix["input_pins"][
            "aggregate_retention_budget"
        ]["raw_staging_margin_bytes"],
        "admission_run_id": run_id,
    }
    supersedes_by_cell = _latest_supersession_backrefs(
        superseded_root, current_preflight_sha256=preflight_sha,
    )
    journals_root.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        safe_name = _cell_safe_name(cell)
        journal_dir = journals_root / safe_name
        result = _load_or_recover_cell_journal(
            journal_dir=journal_dir, cell=cell, matrix_sha256=matrix_sha,
            matrix_seal_sha256=matrix["matrix_seal_sha256"],
            preflight_sha256=preflight_sha, source_commit=head_pre,
            current_invocation_admitted=True,
            original_preflight_status="fresh",
            current_preflight_path=preflight_path,
        )
        if result is None:
            capture_started_at = _utc_text()
            supersedes = supersedes_by_cell.get(safe_name)
            prior_preparation_failures = preparation_failures_by_cell.get(
                safe_name, []
            )
            prior_capture_defect_attempts = capture_defects_by_cell.get(
                safe_name, []
            )
            _require_preparation_attempt_available(
                safe_name=safe_name,
                backrefs=prior_preparation_failures,
                allow_overflow=getattr(
                    args, "override_preparation_attempt_limit", False
                ),
            )
            start = _journal_start_payload(
                cell=cell, matrix_sha256=matrix_sha,
                matrix_seal_sha256=matrix["matrix_seal_sha256"],
                preflight_sha256=preflight_sha,
                preflight_path=preflight_path, source_commit=head_pre,
                capture_started_at_utc=capture_started_at,
                admission_run_id=run_id, supersedes=supersedes,
                prior_preparation_failures=prior_preparation_failures,
                prior_capture_defect_attempts=prior_capture_defect_attempts,
            )
            journal_dir.mkdir(parents=False, exist_ok=False)
            _write_cell_journal(journal_dir / JOURNAL_START_NAME, start)
            cell_dir = journal_dir / "cells" / safe_name
            cell_dir.mkdir(parents=True, exist_ok=False)
            capture_provenance = {
                **capture_provenance_base,
                "capture_started_at_utc": capture_started_at,
                "supersedes": supersedes,
                "prior_preparation_failures": prior_preparation_failures,
                "prior_capture_defect_attempts": (
                    prior_capture_defect_attempts
                ),
            }
            result = capture_cell(
                cell=cell, cell_dir=cell_dir, matrix=matrix, pins=pins,
                cell_budgets=cell_budgets,
                expected_tracer_sha256=(
                    preflight_manifest["checks"]["injection_runtime"]
                    ["tracer_so_sha256"]
                ),
                capture_provenance=capture_provenance,
                expected_ncu_identity=expected_ncu_identity,
                certified_sol_env=certified_sol_env,
                sol_environment_probe=sol_environment_probe,
                module_store=output_root / "module_store",
            )
            _verify_journal_artifacts(
                result, root=journal_dir, safe_name=safe_name,
            )
            _write_cell_journal(
                journal_dir / JOURNAL_COMPLETE_NAME,
                _journal_payload(start=start, result=result),
            )
    live_capture_guard(pins)
    active = _active_journals(
        journals_root=journals_root, cells=cells,
        matrix_sha256=matrix_sha,
        matrix_seal_sha256=matrix["matrix_seal_sha256"],
        source_commit=head_pre,
    )
    run_rows = _read_jsonl(
        run_journal_path, error_code="CHUNK_RUN_JOURNAL_INVALID"
    )
    return _finalize_from_journals(
        args=args, cells=cells, active=active, stage_root=stage_root,
        final_dir=final_dir, matrix=matrix, matrix_path=matrix_path,
        matrix_sha256=matrix_sha, authority_path=authority_path,
        pins=pins, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path,
        output_root=output_root, source_commit=head_pre,
        run_journal_path=run_journal_path, run_rows=run_rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--authority", default=str(DEFAULT_AUTHORITY))
    parser.add_argument("--pins", default=str(DEFAULT_PINS))
    parser.add_argument("--aggregate-budget", default=str(DEFAULT_AGGREGATE_BUDGET))
    parser.add_argument("--cell-budgets", default=str(DEFAULT_CELL_BUDGETS))
    parser.add_argument("--metric-registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--supersede-stale-journals", action="store_true",
        help=(
            "atomically preserve every active journal before recapturing a "
            "chunk whose original preflight is no longer usable"
        ),
    )
    parser.add_argument(
        "--allow-supersession-overflow", action="store_true",
        help=(
            "permit a third or later supersession only after explicit "
            "operator escalation"
        ),
    )
    parser.add_argument(
        "--override-preparation-attempt-limit", action="store_true",
        help=(
            "permit a third or later pre-GPU preparation attempt only after "
            "explicit operator escalation"
        ),
    )
    args = parser.parse_args(argv)
    if args.allow_supersession_overflow and not args.supersede_stale_journals:
        parser.error(
            "--allow-supersession-overflow requires "
            "--supersede-stale-journals"
        )
    try:
        final_dir = run_chunk(args)
    except ChunkRefusal as exc:
        print(f"sm86_capture_chunk: REFUSED [{exc.code}] {exc.detail}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"sm86_capture_chunk: INPUT ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"sm86_capture_chunk: SEALED {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
