"""Fail-closed guard for the blocked production evaluator import surface.

The registry is accounting, not scientific or operational authority. While the
boundary is blocked, its one retained source has an exact refusal-only AST
shape. Repository files are read once through descriptor-anchored, no-follow
snapshots. Python snapshots are required to declare UTF-8 (if they declare an
encoding at all), and the same decoded text is parsed and compiled as an AST.
Execution is available only through the retained-snapshot API, never Python's
ordinary package import machinery.

Run:  .venv/bin/python -m pytest eval/guards/test_evaluator_import_surface.py -q
"""
from __future__ import annotations

import _frozen_importlib_external
import ast
import builtins
import copy
import errno
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import marshal
import os
import py_compile
import re
import stat
import struct
import sys
import zipimport
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import CodeType, FunctionType, MethodType, ModuleType
from typing import Any, NoReturn

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_RELATIVE = "eval/guards/evaluator_import_surface_registry.v1.json"
SCHEMA_RELATIVE = "eval/guards/evaluator_import_surface_registry.v1.schema.json"
REGISTRY_SHA256 = "16571e643b464d0c2e27fced681d736e7698f9cbab6e97c303ddce959a0d2655"
ENTRYPOINT = "eval/production_evaluator/__init__.py"
ENTRYPOINT_MODULE = "eval.production_evaluator"
SCHEMA_ID = (
    "https://gpu-simulator.invalid/schemas/"
    "evaluator_import_surface_registry.v1.schema.json"
)
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+\Z")
PEP263_CODING_COOKIE = re.compile(
    r"^[ \t\f]*\#.*?coding[:=][ \t]*([-_.A-Za-z0-9]+)", re.IGNORECASE
)
HEX_SHA256 = frozenset("0123456789abcdef")
AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES = 1024 * 1024
MAX_CLASSIFICATION_ROWS = 128
MAX_CLASSIFIED_SOURCE_BYTES = 8 * 1024 * 1024
MAX_EVAL_RUNTIME_FILES = 64
MAX_EVAL_RUNTIME_ENTRIES = 128
MAX_EVAL_RUNTIME_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_CODE_GRAPH_NODES = 1_000_000
MARSHAL_STRING_TAGS = frozenset(map(ord, "tuazAZ"))
MARSHAL_INTERNED_STRING_TAGS = frozenset(map(ord, "tAZ"))
MAX_SIM_SOURCE_FILES = 4096
MAX_SIM_SOURCE_BYTES = 64 * 1024 * 1024
PRODUCTION_CLASSIFICATIONS = {"production_boundary", "production_dependency"}
EXPECTED_UNRESOLVED_AUTHORITIES = {
    "cohort_membership",
    "coverage_thresholds",
    "cycle_aggregation_semantics",
    "hardware_authority",
    "operator_approval",
    "protocol_values",
    "reviewer_approval",
    "trust_anchors",
}
EXPECTED_PROHIBITED_EVALUATORS = {
    "tools/local_labels/timing_prediction_contract.py",
    "workloads/simbench_bw/comparison_runner.py",
    "workloads/simbench_bw/compute_validation_metric.py",
    "workloads/simbench_bw/gcom_lite_cycles_evaluator.py",
    "workloads/simbench_bw/gcom_lite_slug_disjoint_scoring.py",
    "workloads/simbench_bw/per_metric_evidence_harness.py",
    "workloads/simbench_bw/timing_accuracy_runner.py",
    "workloads/simbench_bw/timing_closure.py",
    "workloads/simbench_bw/timing_cycle_predictor.py",
}
EXPECTED_NON_EVALUATORS = {
    "eval/bench.py",
    "eval/production_evaluator_entrypoint.py",
    "eval/production_manifest.py",
    "eval/profile.py",
    "sim/run/src/emit.rs",
    "sim/mem/src/l2.rs",
    "sim/sm/src/issue.rs",
    "sim/sm/src/replay.rs",
    "sim/sm/tests/machine_local_replay.rs",
    "tools/local_labels/label_driver.sh",
    "tools/local_labels/m6_determinism_campaign.py",
    "tools/local_labels/metric_registry.json",
    "tools/local_labels/parse_ncu_metrics.py",
    "tools/local_labels/repeatability_matrix_runner.py",
    "tools/local_labels/sm86_capture_chunk.py",
    "tools/local_labels/test_metric_registry.py",
    "tools/local_labels/test_schema_v5_lockstep.py",
    "workloads/simbench_bw/phase_cycle_ledger_aggregator.py",
    "workloads/simbench_bw/phase_cycle_ledger_consumer.py",
}
EXPECTED_EVALUATOR_CONTRACTS = {
    "eval/contracts/__init__.py",
    "eval/contracts/_common.py",
    "eval/contracts/_receipt_common.py",
    "eval/contracts/label_opening_receipt.py",
    "eval/contracts/launch_crosswalk.py",
    "eval/contracts/model_seal.py",
    "eval/contracts/prediction_seal_receipt.py",
    "eval/contracts/preregistration_receipt.py",
    "eval/contracts/state_lifetime.py",
    "eval/contracts/validation.py",
    "eval/production_labels.py",
}
EXPECTED_SYNTHETIC_OFFLINE_PROTOTYPE_EVALUATORS = {
    "eval/labels.py",
    "eval/production_evaluator_fixture.py",
    "eval/registry.py",
    "eval/score.py",
}
EXPECTED_EVAL_RUNTIME_SOURCES = {
    ENTRYPOINT,
    "eval/bench.py",
    "eval/production_evaluator_entrypoint.py",
    "eval/production_manifest.py",
    "eval/profile.py",
    *EXPECTED_EVALUATOR_CONTRACTS,
    *EXPECTED_SYNTHETIC_OFFLINE_PROTOTYPE_EVALUATORS,
}
SOURCE_SUFFIXES = tuple(importlib.machinery.SOURCE_SUFFIXES)
BYTECODE_SUFFIXES = tuple(importlib.machinery.BYTECODE_SUFFIXES)
_IMPORT_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)
EXTENSION_SUFFIXES = tuple(
    sorted(
        {
            *importlib.machinery.EXTENSION_SUFFIXES,
            ".dll",
            ".dylib",
            ".pyd",
            ".so",
        },
        key=lambda suffix: (-len(suffix), suffix),
    )
)
COMPILED_SUFFIXES = tuple(
    sorted(
        {*BYTECODE_SUFFIXES, *EXTENSION_SUFFIXES, ".pyo"},
        key=lambda suffix: (-len(suffix), suffix),
    )
)
SIM_SOURCE_SUFFIXES = {
    ".bash",
    ".cc",
    ".cpp",
    ".fish",
    ".h",
    ".hpp",
    ".py",
    ".rs",
    ".sh",
    ".toml",
}
SIM_PYTHON_DYNAMIC_MODULES = {"importlib", "runpy", "zipimport"}
SIM_PYTHON_PROCESS_MODULES = {"commands", "multiprocessing", "pty", "subprocess"}
# Attribute-name construction is unbounded, so reflection is rejected rather than
# treated as a string-folding problem with an incomplete accepted grammar.
SIM_PYTHON_DYNAMIC_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "import_module",
    "run_module",
    "run_path",
}
SIM_PYTHON_PROCESS_CALL_NAMES = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "create_subprocess_exec",
    "create_subprocess_shell",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "forkpty",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "run",
    "spawn",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
}
_MODULE_SPEC_TYPE = importlib.machinery.ModuleSpec
_NAMESPACE_LOADER_TYPE = importlib.machinery.NamespaceLoader
_NAMESPACE_PATH_TYPE = _frozen_importlib_external._NamespacePath
_FILE_FINDER_TYPE = importlib.machinery.FileFinder
_ZIP_IMPORTER_TYPE = zipimport.zipimporter
_PATH_FINDER_TYPE = importlib.machinery.PathFinder
_PATH_FINDER_GET_SPEC = _PATH_FINDER_TYPE._get_spec.__func__
_TRUSTED_COMPILE = builtins.compile
_TRUSTED_EXEC = builtins.exec
_TRUSTED_BUILD_CLASS = builtins.__build_class__
_TRUSTED_RUNTIME_ERROR = builtins.RuntimeError
if type(sys.path_hooks) is list:
    _TRUSTED_PATH_HOOKS: tuple[Any, ...] | None = tuple(
        list.__getitem__(sys.path_hooks, index)
        for index in range(list.__len__(sys.path_hooks))
    )
else:  # pragma: no cover - import-time hostile state fails at runtime.
    _TRUSTED_PATH_HOOKS = None


class RegistryViolation(AssertionError):
    """Raised when the evaluator registry or import boundary fails closed."""


@dataclass(frozen=True)
class FileSnapshot:
    """One immutable in-memory view read from one no-follow file descriptor."""

    relative_path: str
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int, int, int, int]

    def utf8(self, context: str) -> str:
        try:
            return self.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            _fail(f"{context} is not UTF-8: {exc}")


@dataclass(frozen=True)
class ValidatedRegistry:
    document: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    snapshots: dict[str, FileSnapshot]
    registry_snapshot: FileSnapshot
    schema_snapshot: FileSnapshot


@dataclass(frozen=True)
class EvalRuntimeInventory:
    sources: frozenset[str]
    bytecode_caches: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise RegistryViolation(message)


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and set(value).issubset(HEX_SHA256)


def _require_inventory_bounds(
    context: str,
    *,
    file_count: int,
    total_bytes: int,
    max_files: int,
    max_bytes: int,
) -> None:
    values = (file_count, total_bytes, max_files, max_bytes)
    if any(type(value) is not int or value < 0 for value in values):
        _fail(f"{context} inventory bounds require exact nonnegative integers")
    if file_count > max_files:
        _fail(
            f"{context}_file_limit_exceeded: observed={file_count}; "
            f"ceiling={max_files}"
        )
    if total_bytes > max_bytes:
        _fail(
            f"{context}_byte_limit_exceeded: observed={total_bytes}; "
            f"ceiling={max_bytes}"
        )


def _safe_relative_path(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{context} must be a non-empty repository-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(SAFE_COMPONENT.fullmatch(part) is None for part in path.parts)
    ):
        _fail(f"{context} is not a canonical safe repository-relative path: {value!r}")
    return value


def _descriptor_flags(*, directory: bool) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        _fail("platform lacks O_NOFOLLOW required by the evaluator guard")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is None:
            _fail("platform lacks O_DIRECTORY required by the evaluator guard")
        flags |= directory_flag
    else:
        nonblocking_flag = getattr(os, "O_NONBLOCK", None)
        if nonblocking_flag is None:
            _fail("platform lacks O_NONBLOCK required by the evaluator guard")
        flags |= nonblocking_flag
    return flags


def _open_root(repo_root: Path) -> int:
    try:
        return os.open(repo_root, _descriptor_flags(directory=True))
    except OSError as exc:
        _fail(f"repository root is not a no-follow directory: {repo_root}: {exc}")


def _open_directory(
    repo_root: Path,
    relative: str,
    context: str,
    *,
    missing_ok: bool = False,
) -> int | None:
    if relative:
        relative = _safe_relative_path(relative, f"{context} directory")
        components = PurePosixPath(relative).parts
    else:
        components = ()
    current = _open_root(repo_root)
    try:
        for component in components:
            try:
                child = os.open(
                    component,
                    _descriptor_flags(directory=True),
                    dir_fd=current,
                )
            except OSError as exc:
                if missing_ok and exc.errno == errno.ENOENT:
                    os.close(current)
                    return None
                _fail(
                    f"{context} directory is missing, symlinked, or non-directory: "
                    f"{relative}: {exc}"
                )
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _snapshot_file(
    repo_root: Path,
    relative: str,
    context: str,
    *,
    max_size_bytes: int = AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES,
) -> FileSnapshot:
    if type(max_size_bytes) is not int or max_size_bytes < 0:
        _fail(f"{context} snapshot bound must be an exact nonnegative integer")
    relative = _safe_relative_path(relative, f"{context} path")
    path = PurePosixPath(relative)
    parent_relative = PurePosixPath(*path.parts[:-1]).as_posix()
    if parent_relative == ".":
        parent_relative = ""
    parent = _open_directory(repo_root, parent_relative, context)
    assert parent is not None
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                path.name,
                _descriptor_flags(directory=False),
                dir_fd=parent,
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                _fail(f"{context} row is stale; file is missing: {relative}")
            _fail(f"{context} file is symlinked or unreadable: {relative}: {exc}")
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{context} is not a regular file: {relative}")
        if before.st_size > max_size_bytes:
            _fail(
                "authority_snapshot_size_limit_exceeded: "
                f"{context} file {relative} is {before.st_size} bytes; "
                f"ceiling={max_size_bytes}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                _fail(
                    f"{context} changed while its retained bytes were read: "
                    f"{relative}"
                )
            chunks.append(block)
            remaining -= len(block)
        extra = os.read(descriptor, 1)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        data = b"".join(chunks)
        if (
            extra
            or identity_before != identity_after
            or len(data) != before.st_size
        ):
            _fail(f"{context} changed while its retained bytes were read: {relative}")
        return FileSnapshot(
            relative,
            data,
            hashlib.sha256(data).hexdigest(),
            identity_before,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _path_mode(repo_root: Path, relative: str, context: str) -> int | None:
    relative = _safe_relative_path(relative, f"{context} candidate")
    path = PurePosixPath(relative)
    parent_relative = PurePosixPath(*path.parts[:-1]).as_posix()
    if parent_relative == ".":
        parent_relative = ""
    parent = _open_directory(
        repo_root,
        parent_relative,
        context,
        missing_ok=True,
    )
    if parent is None:
        return None
    try:
        try:
            metadata = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"{context} candidate is a symlink: {relative}")
        return metadata.st_mode
    finally:
        os.close(parent)


def _directory_entries(
    repo_root: Path,
    relative: str,
    context: str,
    *,
    max_entries: int = MAX_EVAL_RUNTIME_ENTRIES,
) -> dict[str, int]:
    if type(max_entries) is not int or max_entries < 0:
        _fail(f"{context} directory bound must be an exact nonnegative integer")
    descriptor = _open_directory(repo_root, relative, context)
    assert descriptor is not None
    try:
        result: dict[str, int] = {}
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if len(result) >= max_entries:
                    _fail(
                        f"{context}_directory_entry_limit_exceeded: "
                        f"ceiling={max_entries}"
                    )
                name = entry.name
                if type(name) is not str or SAFE_COMPONENT.fullmatch(name) is None:
                    _fail(f"{context} contains an unsafe entry name: {name!r}")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    joined = f"{relative}/{name}" if relative else name
                    _fail(f"{context} contains a symlink: {joined}")
                result[name] = metadata.st_mode
        return dict(sorted(result.items()))
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {token}")


def _strict_json(snapshot: FileSnapshot, context: str) -> Any:
    try:
        return json.loads(
            snapshot.utf8(context),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        _fail(f"{context} is not valid JSON: {exc}")


def _validate_with_schema(document: Any, schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = [f"{error.json_path}: {error.message}" for error in errors]
        _fail("registry schema validation failed:\n" + "\n".join(details))


def _classification_by_path(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if type(document) is not dict:
        _fail("classification registry must be an exact built-in dict")
    rows = document.get("classifications")
    if type(rows) is not list:
        _fail("classification rows must be an exact built-in list")
    _require_inventory_bounds(
        "classification_inventory",
        file_count=len(rows),
        total_bytes=0,
        max_files=MAX_CLASSIFICATION_ROWS,
        max_bytes=MAX_CLASSIFIED_SOURCE_BYTES,
    )
    if any(type(row) is not dict for row in rows):
        _fail("classification rows must be exact built-in dicts")
    paths = [row.get("path") for row in rows]
    if any(type(path) is not str for path in paths):
        _fail("classification paths must be exact built-in strings")
    if paths != sorted(paths):
        _fail("classification rows must be sorted by path")
    if len(paths) != len(set(paths)):
        _fail("classification registry contains a duplicate path")
    return {row["path"]: row for row in rows}


def _verify_blocked_package_inventory(repo_root: Path) -> None:
    package = PurePosixPath(ENTRYPOINT).parent.as_posix()
    entries = _directory_entries(repo_root, package, "blocked evaluator package")
    for name in entries:
        if name == "__pycache__" or name.endswith(COMPILED_SUFFIXES):
            _fail(f"blocked evaluator package contains compiled code: {package}/{name}")
    if set(entries) != {"__init__.py"} or not stat.S_ISREG(entries["__init__.py"]):
        _fail(
            "blocked evaluator package inventory must be exactly __init__.py; "
            f"observed={sorted(entries)!r}"
        )


def _eval_runtime_source_inventory(repo_root: Path) -> EvalRuntimeInventory:
    """Close importable evaluator Python sources outside guard and test trees."""

    sources: set[str] = set()
    bytecode_caches: list[str] = []
    compiled: list[str] = []
    excluded = {"eval/guards", "eval/tests"}
    observed_entries = 0

    def inventory_entries(relative: str, context: str) -> dict[str, int]:
        nonlocal observed_entries
        remaining = MAX_EVAL_RUNTIME_ENTRIES - observed_entries
        entries = _directory_entries(
            repo_root,
            relative,
            context,
            max_entries=max(0, remaining),
        )
        observed_entries += len(entries)
        return entries

    def inspect_bytecode_cache(relative: str) -> None:
        entries = inventory_entries(
            relative, "evaluator runtime bytecode cache"
        )
        for name, mode in entries.items():
            joined = f"{relative}/{name}"
            if not stat.S_ISREG(mode) or not name.endswith(BYTECODE_SUFFIXES):
                _fail(
                    "evaluator runtime bytecode cache contains an unexpected "
                    f"artifact: {joined}"
                )
            bytecode_caches.append(joined)

    def walk(relative: str) -> None:
        entries = inventory_entries(
            relative, "evaluator runtime inventory"
        )
        for name, mode in entries.items():
            joined = f"{relative}/{name}"
            if joined in excluded:
                continue
            if name == "__pycache__":
                if not stat.S_ISDIR(mode):
                    _fail(
                        "evaluator runtime __pycache__ candidate is not a directory: "
                        f"{joined}"
                    )
                inspect_bytecode_cache(joined)
                continue
            if stat.S_ISDIR(mode):
                walk(joined)
                continue
            if not stat.S_ISREG(mode):
                if name.endswith((*SOURCE_SUFFIXES, *COMPILED_SUFFIXES)):
                    _fail(
                        "evaluator runtime candidate is not a regular file: "
                        f"{joined}"
                    )
                continue
            if name.endswith(SOURCE_SUFFIXES):
                sources.add(joined)
            elif name.endswith(COMPILED_SUFFIXES):
                compiled.append(joined)

    walk("eval")
    if compiled:
        _fail(f"evaluator runtime contains compiled candidates: {sorted(compiled)!r}")
    _require_inventory_bounds(
        "eval_runtime_inventory",
        file_count=len(sources) + len(bytecode_caches),
        total_bytes=0,
        max_files=MAX_EVAL_RUNTIME_FILES,
        max_bytes=MAX_EVAL_RUNTIME_BYTES,
    )
    return EvalRuntimeInventory(
        frozenset(sources), tuple(sorted(bytecode_caches))
    )


def _runtime_cache_source_and_optimization(cache_path: str) -> tuple[str, int]:
    path = PurePosixPath(cache_path)
    if path.parent.name != "__pycache__":
        _fail(f"evaluator runtime bytecode cache path is malformed: {cache_path}")
    cache_tag = sys.implementation.cache_tag
    if type(cache_tag) is not str or not cache_tag:
        _fail("interpreter cache tag is unavailable")
    marker = f".{cache_tag}"
    filename = path.name
    if not filename.endswith(".pyc"):
        _fail(f"evaluator runtime bytecode cache suffix is unknown: {cache_path}")
    body = filename[: -len(".pyc")]
    marker_index = body.rfind(marker)
    if marker_index <= 0:
        _fail(f"evaluator runtime bytecode cache tag is unknown: {cache_path}")
    stem = body[:marker_index]
    trailer = body[marker_index + len(marker) :]
    if trailer == "":
        optimization = 0
    elif trailer in {".opt-1", ".opt-2"}:
        optimization = int(trailer[-1])
    else:
        _fail(
            f"evaluator runtime bytecode optimization tag is unknown: {cache_path}"
        )
    source = path.parent.parent / f"{stem}.py"
    return source.as_posix(), optimization


def _compiled_runtime_source_payload(
    repo_root: Path,
    source_path: str,
    source_snapshot: FileSnapshot,
    optimization: int,
) -> bytes:
    source_text = _strict_python_text(
        source_snapshot, "evaluator runtime bytecode source"
    )
    expected_code = _TRUSTED_COMPILE(
        source_text,
        str(repo_root / PurePosixPath(source_path)),
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=optimization,
    )
    return marshal.dumps(expected_code)


def _preflight_runtime_marshal_payload(payload: bytes) -> None:
    """Reject allocation-amplifying marshal counts before CPython decodes them."""
    if type(payload) is not bytes:
        raise TypeError("runtime marshal payload must be exact bytes")
    position = 0
    object_count = 0
    reference_count = 0
    tasks: list[tuple[str, int]] = [("object", 0)]

    def read_byte() -> int:
        nonlocal position
        if position >= len(payload):
            raise EOFError("runtime marshal payload is truncated")
        value = payload[position]
        position += 1
        return value

    def read_i32() -> int:
        nonlocal position
        end = position + 4
        if end > len(payload):
            raise EOFError("runtime marshal payload is truncated")
        value = int.from_bytes(
            payload[position:end], byteorder="little", signed=True
        )
        position = end
        return value

    def skip_bytes(length: int) -> None:
        nonlocal position
        if length < 0:
            raise ValueError("runtime marshal payload has a negative byte length")
        end = position + length
        if end > len(payload):
            raise EOFError("runtime marshal payload is truncated")
        position = end

    def bounded_count(count: int) -> int:
        if count < 0:
            raise ValueError("runtime marshal payload has a negative object count")
        if count > MAX_RUNTIME_CODE_GRAPH_NODES - object_count:
            raise MemoryError(
                "runtime marshal object count exceeds bounded graph limit"
            )
        if count > len(payload) - position:
            raise EOFError(
                "runtime marshal object count exceeds remaining payload bytes"
            )
        return count

    def schedule_sequence(count: int) -> None:
        if bounded_count(count):
            tasks.append(("sequence", count))

    def read_object(*, allow_null: bool) -> bool:
        nonlocal object_count, reference_count
        raw_tag = read_byte()
        has_reference = bool(raw_tag & 0x80)
        tag = raw_tag & 0x7F
        if tag == ord("0"):
            if has_reference or not allow_null:
                raise ValueError("runtime marshal payload has an unexpected NULL")
            return False

        if object_count >= MAX_RUNTIME_CODE_GRAPH_NODES:
            raise MemoryError(
                "runtime marshal object graph exceeds bounded node limit"
            )
        object_count += 1
        if has_reference:
            if tag == ord("r"):
                raise ValueError("runtime marshal reference cannot carry FLAG_REF")
            reference_count += 1

        if tag in map(ord, "NFTS."):
            return True
        if tag == ord("i"):
            skip_bytes(4)
            return True
        if tag == ord("I"):
            skip_bytes(8)
            return True
        if tag == ord("l"):
            digit_count = read_i32()
            skip_bytes(abs(digit_count) * 2)
            return True
        if tag == ord("f"):
            skip_bytes(read_byte())
            return True
        if tag == ord("g"):
            skip_bytes(8)
            return True
        if tag == ord("x"):
            skip_bytes(read_byte())
            skip_bytes(read_byte())
            return True
        if tag == ord("y"):
            skip_bytes(16)
            return True
        if tag in map(ord, "stuaA"):
            length = read_i32()
            skip_bytes(length)
            return True
        if tag in map(ord, "zZ"):
            skip_bytes(read_byte())
            return True
        if tag == ord("r"):
            reference_index = read_i32()
            if not 0 <= reference_index < reference_count:
                raise ValueError(
                    "runtime marshal payload has an invalid reference index"
                )
            return True
        if tag == ord(")"):
            schedule_sequence(read_byte())
            return True
        if tag in map(ord, "([<>"):
            schedule_sequence(read_i32())
            return True
        if tag == ord("{"):
            tasks.append(("dict_key", 0))
            return True
        if tag == ord("c"):
            skip_bytes(5 * 4)
            code_tasks = (
                *(("object", 0) for _ in range(8)),
                ("i32", 0),
                ("object", 0),
                ("object", 0),
            )
            tasks.extend(reversed(code_tasks))
            return True
        raise ValueError(f"runtime marshal payload has unknown tag: {tag}")

    while tasks:
        kind, remaining = tasks.pop()
        if kind == "object":
            read_object(allow_null=False)
        elif kind == "sequence":
            if remaining > 1:
                tasks.append(("sequence", remaining - 1))
            tasks.append(("object", 0))
        elif kind == "dict_key":
            if read_object(allow_null=True):
                tasks.append(("dict_value", 0))
        elif kind == "dict_value":
            tasks.append(("dict_key", 0))
            tasks.append(("object", 0))
        elif kind == "i32":
            read_i32()
        else:  # pragma: no cover - the task grammar above is closed
            raise AssertionError(f"unknown runtime marshal preflight task: {kind}")

    if position != len(payload):
        raise ValueError("runtime marshal payload has trailing bytes")


def _runtime_marshaled_string_interned(value: str) -> bool:
    """Read CPython's observable interned-string state without mutating it."""
    payload = marshal.dumps(value)
    if not payload:
        raise ValueError("marshal emitted no string tag")
    tag = payload[0] & 0x7F
    if tag not in MARSHAL_STRING_TAGS:
        raise ValueError(f"marshal emitted unexpected string tag: {tag}")
    return tag in MARSHAL_INTERNED_STRING_TAGS


def _runtime_marshaled_localsplus_payload_projections(
    payload: bytes,
) -> tuple[
    tuple[int, tuple[tuple[str, bool, int], ...], int, bytes], ...
] | None:
    """Project marshal-only locals-plus values and raw alias topology."""
    try:
        if type(payload) is not bytes:
            return None
        position = 0
        object_count = 0
        references: list[object] = []
        projections: list[
            tuple[int, tuple[tuple[str, bool, int], ...], int, bytes]
            | None
        ] = []
        opaque = object()
        unresolved = object()
        null = object()
        bytes_marker = object()
        string_marker = object()
        tuple_marker = object()

        def read_byte() -> int:
            nonlocal position
            if position >= len(payload):
                raise EOFError("runtime marshal projection is truncated")
            value = payload[position]
            position += 1
            return value

        def read_i32() -> int:
            nonlocal position
            end = position + 4
            if end > len(payload):
                raise EOFError("runtime marshal projection is truncated")
            value = int.from_bytes(
                payload[position:end], byteorder="little", signed=True
            )
            position = end
            return value

        def read_bytes(length: int) -> bytes:
            nonlocal position
            if length < 0:
                raise ValueError(
                    "runtime marshal projection has a negative byte length"
                )
            end = position + length
            if end > len(payload):
                raise EOFError("runtime marshal projection is truncated")
            value = payload[position:end]
            position = end
            return value

        def bounded_count(count: int) -> int:
            if count < 0:
                raise ValueError(
                    "runtime marshal projection has a negative object count"
                )
            if count > MAX_RUNTIME_CODE_GRAPH_NODES - object_count:
                raise MemoryError(
                    "runtime marshal projection exceeds bounded graph limit"
                )
            if count > len(payload) - position:
                raise EOFError(
                    "runtime marshal projection count exceeds remaining bytes"
                )
            return count

        def read_object(*, allow_null: bool = False) -> object:
            nonlocal object_count
            raw_tag = read_byte()
            has_reference = bool(raw_tag & 0x80)
            tag = raw_tag & 0x7F
            if tag == ord("0"):
                if has_reference or not allow_null:
                    raise ValueError(
                        "runtime marshal projection has an unexpected NULL"
                    )
                return null

            if object_count >= MAX_RUNTIME_CODE_GRAPH_NODES:
                raise MemoryError(
                    "runtime marshal projection exceeds bounded graph limit"
                )
            object_count += 1
            object_serial = object_count
            reference_slot: int | None = None
            if has_reference:
                if tag == ord("r"):
                    raise ValueError(
                        "runtime marshal projection reference carries FLAG_REF"
                    )
                reference_slot = len(references)
                references.append(unresolved)

            if tag in map(ord, "NFTS."):
                value: object = opaque
            elif tag == ord("i"):
                read_bytes(4)
                value = opaque
            elif tag == ord("I"):
                read_bytes(8)
                value = opaque
            elif tag == ord("l"):
                read_bytes(abs(read_i32()) * 2)
                value = opaque
            elif tag == ord("f"):
                read_bytes(read_byte())
                value = opaque
            elif tag == ord("g"):
                read_bytes(8)
                value = opaque
            elif tag == ord("x"):
                read_bytes(read_byte())
                read_bytes(read_byte())
                value = opaque
            elif tag == ord("y"):
                read_bytes(16)
                value = opaque
            elif tag == ord("s"):
                value = (bytes_marker, read_bytes(read_i32()), object_serial)
            elif tag in map(ord, "tuaA"):
                encoded = read_bytes(read_i32())
                value = (
                    string_marker,
                    encoded.decode(
                        "ascii" if tag in map(ord, "aA") else "utf-8",
                        errors=(
                            "strict"
                            if tag in map(ord, "aA")
                            else "surrogatepass"
                        ),
                    ),
                    tag in MARSHAL_INTERNED_STRING_TAGS,
                    object_serial,
                )
            elif tag in map(ord, "zZ"):
                encoded = read_bytes(read_byte())
                value = (
                    string_marker,
                    encoded.decode("ascii", errors="strict"),
                    tag in MARSHAL_INTERNED_STRING_TAGS,
                    object_serial,
                )
            elif tag == ord("r"):
                reference_index = read_i32()
                if not 0 <= reference_index < len(references):
                    raise ValueError(
                        "runtime marshal projection has an invalid reference"
                    )
                value = references[reference_index]
                if value is unresolved:
                    value = opaque
            elif tag in map(ord, "()"):
                count = read_byte() if tag == ord(")") else read_i32()
                items = [read_object() for _ in range(bounded_count(count))]
                value = (tuple_marker, tuple(items), object_serial)
            elif tag in map(ord, "[<>"):
                count = bounded_count(read_i32())
                for _ in range(count):
                    read_object()
                value = opaque
            elif tag == ord("{"):
                while read_object(allow_null=True) is not null:
                    read_object()
                value = opaque
            elif tag == ord("c"):
                projection_index = len(projections)
                projections.append(None)
                read_bytes(5 * 4)
                fields = [read_object() for _ in range(8)]
                read_i32()
                read_object()
                read_object()
                names = fields[3]
                kinds = fields[4]
                if (
                    type(names) is tuple
                    and len(names) == 3
                    and names[0] is tuple_marker
                    and type(names[1]) is tuple
                    and type(names[2]) is int
                    and type(kinds) is tuple
                    and len(kinds) == 3
                    and kinds[0] is bytes_marker
                    and type(kinds[1]) is bytes
                    and type(kinds[2]) is int
                ):
                    name_items = names[1]
                    if all(
                        type(item) is tuple
                        and len(item) == 4
                        and item[0] is string_marker
                        and type(item[1]) is str
                        and type(item[2]) is bool
                        and type(item[3]) is int
                        for item in name_items
                    ):
                        projections[projection_index] = (
                            names[2],
                            tuple(
                                (item[1], item[2], item[3])
                                for item in name_items
                            ),
                            kinds[2],
                            kinds[1],
                        )
                value = opaque
            else:
                raise ValueError(
                    f"runtime marshal projection has unknown tag: {tag}"
                )

            if reference_slot is not None:
                references[reference_slot] = value
            return value

        read_object()
        if position != len(payload) or any(item is None for item in projections):
            return None
        return tuple(item for item in projections if item is not None)
    except (
        EOFError,
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return None


def _runtime_marshaled_localsplus_projections(
    code: CodeType,
) -> tuple[
    tuple[int, tuple[tuple[str, bool, int], ...], int, bytes], ...
] | None:
    try:
        payload = marshal.dumps(code)
    except (MemoryError, OverflowError, RecursionError, TypeError, ValueError):
        return None
    return _runtime_marshaled_localsplus_payload_projections(payload)


def _runtime_scalar_fingerprint(
    value: object,
) -> tuple[bytes, bool] | None:
    digest = hashlib.sha256()
    if value is None:
        digest.update(b"none")
    elif value is Ellipsis:
        digest.update(b"ellipsis")
    elif type(value) is bool:
        digest.update(b"bool\x01" if value else b"bool\x00")
    elif type(value) is int:
        magnitude = abs(value)
        encoded = magnitude.to_bytes(
            max(1, (magnitude.bit_length() + 7) // 8), "big"
        )
        digest.update(b"int-")
        digest.update(b"-" if value < 0 else b"+")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    elif type(value) is bytes:
        digest.update(b"bytes-")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    elif type(value) is str:
        encoded = value.encode("utf-8", errors="surrogatepass")
        digest.update(b"str-")
        digest.update(
            b"interned-" if _runtime_marshaled_string_interned(value) else b"plain-"
        )
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    elif type(value) is float:
        digest.update(b"float-")
        digest.update(struct.pack("!d", value))
        return digest.digest(), value != value
    elif type(value) is complex:
        digest.update(b"complex-")
        digest.update(struct.pack("!d", value.real))
        digest.update(struct.pack("!d", value.imag))
        return digest.digest(), value != value
    elif not isinstance(value, (tuple, frozenset)):
        value_type = type(value)
        digest.update(b"unsupported-")
        digest.update(value_type.__module__.encode("utf-8", errors="strict"))
        digest.update(b".")
        digest.update(value_type.__qualname__.encode("utf-8", errors="strict"))
        return digest.digest(), True
    else:
        return None
    return digest.digest(), False


def _runtime_constant_fingerprints(
    values: list[object],
    *,
    max_nodes: int,
) -> tuple[list[tuple[bytes, bool]], int]:
    """Fingerprint immutable constants without trusting container iteration."""
    memo: dict[int, tuple[bytes, bool]] = {}
    visiting: set[int] = set()
    pending = [(value, False) for value in reversed(values)]
    visited_count = 0

    while pending:
        value, expanded = pending.pop()
        identity = id(value)
        if identity in memo:
            continue
        scalar = _runtime_scalar_fingerprint(value)
        if scalar is not None:
            if visited_count >= max_nodes:
                raise ValueError("runtime constant fingerprint bound exceeded")
            visited_count += 1
            memo[identity] = scalar
            continue
        assert isinstance(value, (tuple, frozenset))
        if expanded:
            child_results = [memo[id(child)] for child in value]
            if isinstance(value, frozenset):
                child_results.sort(key=lambda item: item[0])
                tag = b"frozenset-"
            else:
                tag = b"tuple-"
            digest = hashlib.sha256()
            digest.update(tag)
            digest.update(len(child_results).to_bytes(8, "big"))
            for fingerprint, _identity_dependent in child_results:
                digest.update(fingerprint)
            memo[identity] = (
                digest.digest(),
                any(result[1] for result in child_results),
            )
            visiting.remove(identity)
            continue
        if identity in visiting:
            raise ValueError("runtime constant graph is cyclic")
        if visited_count >= max_nodes:
            raise ValueError("runtime constant fingerprint bound exceeded")
        visited_count += 1
        visiting.add(identity)
        pending.append((value, True))
        pending.extend(
            (child, False)
            for child in reversed(tuple(value))
            if id(child) not in memo
        )

    return [memo[id(value)] for value in values], visited_count


def _runtime_code_matches_source(
    observed: object,
    expected: CodeType,
) -> bool:
    """Compare the observable code-object graph with deterministic bounds."""
    if isinstance(observed, CodeType):
        observed_localsplus = _runtime_marshaled_localsplus_projections(
            observed
        )
        expected_localsplus = _runtime_marshaled_localsplus_projections(
            expected
        )
        if (
            observed_localsplus is None
            or expected_localsplus is None
            or observed_localsplus != expected_localsplus
        ):
            return False
    integer_attributes = (
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_firstlineno",
    )
    graph_attributes = (
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_qualname",
        "co_linetable",
        "co_exceptiontable",
        "co_freevars",
        "co_cellvars",
    )
    pending = [(observed, expected)]
    scheduled_pairs = 1
    fingerprint_work = 0
    observed_to_expected: dict[int, int] = {}
    expected_to_observed: dict[int, int] = {}
    compared_pairs: set[tuple[int, int]] = set()

    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False

        left_id = id(left)
        right_id = id(right)
        if (
            left_id in observed_to_expected
            and observed_to_expected[left_id] != right_id
        ):
            return False
        if (
            right_id in expected_to_observed
            and expected_to_observed[right_id] != left_id
        ):
            return False
        observed_to_expected[left_id] = right_id
        expected_to_observed[right_id] = left_id

        pair = (left_id, right_id)
        if pair in compared_pairs:
            continue
        compared_pairs.add(pair)

        if isinstance(left, CodeType):
            assert isinstance(right, CodeType)
            if any(
                getattr(left, attribute) != getattr(right, attribute)
                for attribute in integer_attributes
            ):
                return False
            child_count = len(graph_attributes)
            if scheduled_pairs > MAX_RUNTIME_CODE_GRAPH_NODES - child_count:
                return False
            scheduled_pairs += child_count
            pending.extend(
                (getattr(left, attribute), getattr(right, attribute))
                for attribute in reversed(graph_attributes)
            )
            continue
        if isinstance(left, tuple):
            assert isinstance(right, tuple)
            if len(left) != len(right):
                return False
            child_count = len(left)
            if scheduled_pairs > MAX_RUNTIME_CODE_GRAPH_NODES - child_count:
                return False
            scheduled_pairs += child_count
            pending.extend(zip(reversed(left), reversed(right), strict=True))
            continue
        if isinstance(left, frozenset):
            assert isinstance(right, frozenset)
            if len(left) != len(right):
                return False
            child_count = len(left)
            if scheduled_pairs > MAX_RUNTIME_CODE_GRAPH_NODES - child_count:
                return False
            scheduled_pairs += child_count
            left_items = list(left)
            right_items = list(right)
            remaining_fingerprint_work = (
                MAX_RUNTIME_CODE_GRAPH_NODES
                - scheduled_pairs
                - fingerprint_work
            )
            if remaining_fingerprint_work < 0:
                return False
            try:
                left_fingerprints, left_work = _runtime_constant_fingerprints(
                    left_items,
                    max_nodes=remaining_fingerprint_work,
                )
                fingerprint_work += left_work
                remaining_fingerprint_work = (
                    MAX_RUNTIME_CODE_GRAPH_NODES
                    - scheduled_pairs
                    - fingerprint_work
                )
                right_fingerprints, right_work = _runtime_constant_fingerprints(
                    right_items,
                    max_nodes=remaining_fingerprint_work,
                )
                fingerprint_work += right_work
            except ValueError:
                return False

            left_identity_dependent = any(
                identity_dependent
                for _fingerprint, identity_dependent in left_fingerprints
            )
            right_identity_dependent = any(
                identity_dependent
                for _fingerprint, identity_dependent in right_fingerprints
            )
            if left_identity_dependent != right_identity_dependent:
                return False
            if not left_identity_dependent:
                child_pairs = list(
                    zip(left_items, right_items, strict=True)
                )
            else:
                left_groups: dict[bytes, list[object]] = {}
                right_groups: dict[bytes, list[object]] = {}
                for item, (fingerprint, _identity_dependent) in zip(
                    left_items, left_fingerprints, strict=True
                ):
                    left_groups.setdefault(fingerprint, []).append(item)
                for item, (fingerprint, _identity_dependent) in zip(
                    right_items, right_fingerprints, strict=True
                ):
                    right_groups.setdefault(fingerprint, []).append(item)
                if set(left_groups) != set(right_groups):
                    return False

                child_pairs = []
                for fingerprint in sorted(left_groups):
                    left_group = left_groups[fingerprint]
                    right_group = right_groups[fingerprint]
                    if len(left_group) != len(right_group):
                        return False
                    left_by_identity = {id(item): item for item in left_group}
                    right_by_identity = {id(item): item for item in right_group}
                    paired_left_to_right: dict[int, int] = {}
                    paired_right: set[int] = set()
                    group_pairs: list[tuple[object, object]] = []

                    for left_item in left_group:
                        left_identity = id(left_item)
                        mapped_right = observed_to_expected.get(left_identity)
                        if mapped_right is None:
                            continue
                        right_item = right_by_identity.get(mapped_right)
                        if right_item is None or mapped_right in paired_right:
                            return False
                        paired_left_to_right[left_identity] = mapped_right
                        paired_right.add(mapped_right)
                        group_pairs.append((left_item, right_item))

                    for right_item in right_group:
                        right_identity = id(right_item)
                        mapped_left = expected_to_observed.get(right_identity)
                        if mapped_left is None:
                            continue
                        left_item = left_by_identity.get(mapped_left)
                        if left_item is None:
                            return False
                        existing_right = paired_left_to_right.get(mapped_left)
                        if existing_right is not None:
                            if existing_right != right_identity:
                                return False
                            continue
                        if right_identity in paired_right:
                            return False
                        paired_left_to_right[mapped_left] = right_identity
                        paired_right.add(right_identity)
                        group_pairs.append((left_item, right_item))

                    remaining_left = [
                        item
                        for item in left_group
                        if id(item) not in paired_left_to_right
                    ]
                    remaining_right = [
                        item
                        for item in right_group
                        if id(item) not in paired_right
                    ]
                    group_pairs.extend(
                        zip(remaining_left, remaining_right, strict=True)
                    )
                    child_pairs.extend(group_pairs)
            pending.extend(reversed(child_pairs))
            continue
        if isinstance(left, float):
            assert isinstance(right, float)
            if struct.pack("!d", left) != struct.pack("!d", right):
                return False
            continue
        if isinstance(left, complex):
            assert isinstance(right, complex)
            if (
                struct.pack("!d", left.real) != struct.pack("!d", right.real)
                or struct.pack("!d", left.imag) != struct.pack("!d", right.imag)
            ):
                return False
            continue
        if type(left) is str:
            assert type(right) is str
            if left != right or (
                _runtime_marshaled_string_interned(left)
                != _runtime_marshaled_string_interned(right)
            ):
                return False
            continue
        if type(left) in {type(None), bool, int, bytes, type(Ellipsis)}:
            if left != right:
                return False
            continue
        return False

    return True


def _verify_runtime_bytecode_caches(
    repo_root: Path,
    inventory: EvalRuntimeInventory,
    source_snapshots: dict[str, FileSnapshot],
    *,
    max_cache_bytes: int = MAX_EVAL_RUNTIME_BYTES,
    retained_cache_snapshots: dict[str, FileSnapshot] | None = None,
) -> int:
    if type(max_cache_bytes) is not int or max_cache_bytes < 0:
        _fail(
            "evaluator runtime cache budget must be an exact nonnegative integer"
        )
    if retained_cache_snapshots is not None and (
        type(retained_cache_snapshots) is not dict
        or retained_cache_snapshots
    ):
        _fail("retained evaluator cache snapshot map must be an empty built-in dict")
    total_bytes = 0
    prepared: list[tuple[str, bytes, bytes]] = []
    for cache_path in inventory.bytecode_caches:
        source_path, optimization = _runtime_cache_source_and_optimization(cache_path)
        source_snapshot = source_snapshots.get(source_path)
        if source_snapshot is None or source_path not in inventory.sources:
            _fail(
                "evaluator runtime bytecode cache lacks its retained source: "
                f"{cache_path}"
            )
        remaining_cache_bytes = max_cache_bytes - total_bytes
        cache_snapshot = _snapshot_file(
            repo_root,
            cache_path,
            "evaluator runtime bytecode cache",
            max_size_bytes=min(
                AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES,
                remaining_cache_bytes,
            ),
        )
        if retained_cache_snapshots is not None:
            retained_cache_snapshots[cache_path] = cache_snapshot
        total_bytes += len(cache_snapshot.data)
        data = cache_snapshot.data
        if len(data) < 16 or data[:4] != importlib.util.MAGIC_NUMBER:
            _fail(
                "evaluator runtime bytecode cache header is invalid: "
                f"{cache_path}"
            )
        flags = int.from_bytes(data[4:8], byteorder="little", signed=False)
        if flags not in {0, 1, 3}:
            _fail(
                "evaluator runtime bytecode cache flags are invalid: "
                f"{cache_path}"
            )
        try:
            expected_payload = _compiled_runtime_source_payload(
                repo_root, source_path, source_snapshot, optimization
            )
            observed_payload = data[16:]
            _preflight_runtime_marshal_payload(observed_payload)
        except MemoryError as exc:
            _fail(
                "evaluator_runtime_bytecode_resource_unavailable: "
                f"{cache_path}: {type(exc).__name__}: {exc}"
            )
        except (EOFError, OverflowError, TypeError, ValueError) as exc:
            _fail(
                "evaluator runtime bytecode payload is invalid: "
                f"{cache_path}: {type(exc).__name__}: {exc}"
            )
        observed_localsplus = (
            _runtime_marshaled_localsplus_payload_projections(observed_payload)
        )
        expected_localsplus = (
            _runtime_marshaled_localsplus_payload_projections(expected_payload)
        )
        if (
            observed_localsplus is None
            or expected_localsplus is None
            or observed_localsplus != expected_localsplus
        ):
            _fail(
                "evaluator runtime bytecode differs from its retained source: "
                f"{cache_path}"
            )
        prepared.append((cache_path, observed_payload, expected_payload))

    # Compile every expected graph before decoding any observed cache. CPython's
    # marshal loader retains process-global interning/reference state, so mixing
    # observed decoding with later compilation makes equivalent optimization
    # caches compare against different runtime states.
    for cache_path, observed_payload, expected_payload in prepared:
        try:
            observed_stream = io.BytesIO(observed_payload)
            observed_code = marshal.load(observed_stream)
            if observed_stream.read(1) != b"":
                _fail(
                    "evaluator runtime bytecode payload has trailing bytes: "
                    f"{cache_path}"
                )
            expected_code = marshal.loads(expected_payload)
        except MemoryError as exc:
            _fail(
                "evaluator_runtime_bytecode_resource_unavailable: "
                f"{cache_path}: {type(exc).__name__}: {exc}"
            )
        except (EOFError, OverflowError, TypeError, ValueError) as exc:
            _fail(
                "evaluator runtime bytecode payload is invalid: "
                f"{cache_path}: {type(exc).__name__}: {exc}"
            )
        if not isinstance(expected_code, CodeType):
            _fail("trusted evaluator runtime compilation did not produce code")
        try:
            code_matches = _runtime_code_matches_source(observed_code, expected_code)
        except MemoryError as exc:
            _fail(
                "evaluator_runtime_bytecode_resource_unavailable: "
                f"{cache_path}: {type(exc).__name__}: {exc}"
            )
        if not code_matches:
            _fail(
                "evaluator runtime bytecode differs from its retained source: "
                f"{cache_path}"
            )
    return total_bytes


def _verify_runtime_snapshots_unchanged(
    repo_root: Path,
    inventory: EvalRuntimeInventory,
    source_snapshots: dict[str, FileSnapshot],
    cache_snapshots: dict[str, FileSnapshot],
) -> None:
    expected_sources = set(inventory.sources)
    expected_caches = set(inventory.bytecode_caches)
    if not expected_sources.issubset(source_snapshots):
        _fail("retained evaluator runtime source snapshot set is incomplete")
    if set(cache_snapshots) != expected_caches:
        _fail("retained evaluator runtime cache snapshot set is incomplete")

    retained = {
        path: source_snapshots[path] for path in expected_sources
    }
    retained.update(cache_snapshots)
    for path in sorted(retained):
        before = retained[path]
        try:
            after = _snapshot_file(
                repo_root,
                path,
                "terminal evaluator runtime authentication",
                max_size_bytes=len(before.data),
            )
        except RegistryViolation as exc:
            _fail(
                "evaluator runtime source/cache changed during bytecode "
                f"authentication: {path}: {exc}"
            )
        if (
            after.relative_path != before.relative_path
            or after.identity != before.identity
            or after.sha256 != before.sha256
            or after.data != before.data
        ):
            _fail(
                "evaluator runtime source/cache changed during bytecode "
                f"authentication: {path}"
            )


def _load_registry_document(
    repo_root: Path,
) -> tuple[dict[str, Any], FileSnapshot, FileSnapshot]:
    registry_snapshot = _snapshot_file(
        repo_root, REGISTRY_RELATIVE, "classification registry"
    )
    if registry_snapshot.sha256 != REGISTRY_SHA256:
        _fail(
            "classification registry hash drifted: "
            f"expected {REGISTRY_SHA256}, got {registry_snapshot.sha256}"
        )
    document = _strict_json(registry_snapshot, "classification registry")
    if type(document) is not dict:
        _fail("classification registry root must be an exact JSON object")
    canonical = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if registry_snapshot.data != canonical:
        _fail("classification registry is not deterministic sorted JSON plus LF")
    binding = document.get("schema_binding")
    if type(binding) is not dict or binding.get("path") != SCHEMA_RELATIVE:
        _fail("registry schema path binding is absent or unknown")
    schema_snapshot = _snapshot_file(repo_root, SCHEMA_RELATIVE, "registry schema")
    if (
        not _is_sha256(binding.get("sha256"))
        or schema_snapshot.sha256 != binding["sha256"]
    ):
        _fail("registry schema hash binding drifted")
    schema = _strict_json(schema_snapshot, "classification registry schema")
    if type(schema) is not dict:
        _fail("classification registry schema root must be an exact JSON object")
    if type(schema.get("$id")) is not str or schema["$id"] != SCHEMA_ID:
        _fail("classification registry schema identity is absent or unknown")
    if (
        type(schema.get("$schema")) is not str
        or schema["$schema"] != "https://json-schema.org/draft/2020-12/schema"
    ):
        _fail("classification registry schema dialect is absent or unknown")
    _validate_with_schema(document, schema)
    return document, registry_snapshot, schema_snapshot


def _reject_exact_prohibited_duplicates(
    rows: dict[str, dict[str, Any]],
) -> None:
    prohibited_hashes = {
        row["sha256"]
        for row in rows.values()
        if row["classification"] == "prohibited_evaluator"
    }
    exact_duplicates = sorted(
        path
        for path, row in rows.items()
        if row["import_policy"] == "allow" and row["sha256"] in prohibited_hashes
    )
    if exact_duplicates:
        _fail(
            "production contains an exact whole-file duplicate of prohibited bytes: "
            f"{exact_duplicates!r}"
        )


def _validate_registry(repo_root: Path = REPO_ROOT) -> ValidatedRegistry:
    document, registry_snapshot, schema_snapshot = _load_registry_document(repo_root)
    rows = _classification_by_path(document)
    if document["production_entrypoint"] != ENTRYPOINT:
        _fail("production evaluator entrypoint drifted")
    boundary_rows = {
        path
        for path, row in rows.items()
        if row["classification"] == "production_boundary"
    }
    if boundary_rows != {ENTRYPOINT}:
        _fail(
            "registry must contain exactly one production boundary row: "
            f"{sorted(boundary_rows)!r}"
        )
    prohibited = {
        path
        for path, row in rows.items()
        if row["classification"] == "prohibited_evaluator"
    }
    if prohibited != EXPECTED_PROHIBITED_EVALUATORS:
        _fail(
            "prohibited evaluator classification drifted; "
            f"missing={sorted(EXPECTED_PROHIBITED_EVALUATORS - prohibited)!r}, "
            f"unknown={sorted(prohibited - EXPECTED_PROHIBITED_EVALUATORS)!r}"
        )
    non_evaluators = {
        path
        for path, row in rows.items()
        if row["classification"] == "non_evaluator"
    }
    if non_evaluators != EXPECTED_NON_EVALUATORS:
        _fail(
            "reviewed non-evaluator classification drifted; "
            f"missing={sorted(EXPECTED_NON_EVALUATORS - non_evaluators)!r}, "
            f"unknown={sorted(non_evaluators - EXPECTED_NON_EVALUATORS)!r}"
        )
    evaluator_contracts = {
        path
        for path, row in rows.items()
        if row["classification"] == "evaluator_contract"
    }
    if evaluator_contracts != EXPECTED_EVALUATOR_CONTRACTS:
        _fail(
            "reviewed evaluator-contract classification drifted; "
            f"missing={sorted(EXPECTED_EVALUATOR_CONTRACTS - evaluator_contracts)!r}, "
            f"unknown={sorted(evaluator_contracts - EXPECTED_EVALUATOR_CONTRACTS)!r}"
        )
    prototypes = {
        path
        for path, row in rows.items()
        if row["classification"] == "synthetic_offline_prototype_evaluator"
    }
    if prototypes != EXPECTED_SYNTHETIC_OFFLINE_PROTOTYPE_EVALUATORS:
        _fail(
            "synthetic offline prototype classification drifted; "
            "missing="
            f"{sorted(EXPECTED_SYNTHETIC_OFFLINE_PROTOTYPE_EVALUATORS - prototypes)!r}, "
            "unknown="
            f"{sorted(prototypes - EXPECTED_SYNTHETIC_OFFLINE_PROTOTYPE_EVALUATORS)!r}"
        )
    runtime_inventory = _eval_runtime_source_inventory(repo_root)
    runtime_sources = set(runtime_inventory.sources)
    if runtime_sources != EXPECTED_EVAL_RUNTIME_SOURCES:
        _fail(
            "evaluator runtime source inventory is not directory-closed; "
            f"missing={sorted(EXPECTED_EVAL_RUNTIME_SOURCES - runtime_sources)!r}, "
            f"unknown={sorted(runtime_sources - EXPECTED_EVAL_RUNTIME_SOURCES)!r}"
        )
    authority_names = set(document["authorities"])
    if authority_names != EXPECTED_UNRESOLVED_AUTHORITIES:
        _fail("typed-unresolved evaluator authority field set drifted")
    if any(
        authority != {"status": "typed_unresolved"}
        for authority in document["authorities"].values()
    ):
        _fail("an evaluator authority was populated or lost its typed-unresolved state")

    allowed = {
        path for path, row in rows.items() if row["import_policy"] == "allow"
    }
    if allowed != {ENTRYPOINT}:
        _fail(
            "blocked evaluator may allow only its refusal entrypoint; "
            f"observed={sorted(allowed)!r}"
        )
    _reject_exact_prohibited_duplicates(rows)
    _verify_entrypoint_ancestor_inventory(repo_root)

    snapshots: dict[str, FileSnapshot] = {}
    classified_source_bytes = 0
    for path, row in rows.items():
        snapshot = _snapshot_file(repo_root, path, "classification")
        classified_source_bytes += len(snapshot.data)
        _require_inventory_bounds(
            "classification_inventory",
            file_count=len(rows),
            total_bytes=classified_source_bytes,
            max_files=MAX_CLASSIFICATION_ROWS,
            max_bytes=MAX_CLASSIFIED_SOURCE_BYTES,
        )
        expected = row["sha256"]
        if not _is_sha256(expected):
            _fail(f"classification has a malformed SHA-256: {path}")
        if snapshot.sha256 != expected:
            _fail(
                "classification source hash drifted: "
                f"{path}; expected {expected}, got {snapshot.sha256}"
            )
        snapshots[path] = snapshot
    runtime_source_bytes = sum(
        len(snapshots[path].data) for path in runtime_sources
    )
    _require_inventory_bounds(
        "eval_runtime_inventory",
        file_count=len(runtime_sources) + len(runtime_inventory.bytecode_caches),
        total_bytes=runtime_source_bytes,
        max_files=MAX_EVAL_RUNTIME_FILES,
        max_bytes=MAX_EVAL_RUNTIME_BYTES,
    )
    retained_cache_snapshots: dict[str, FileSnapshot] = {}
    cache_bytes = _verify_runtime_bytecode_caches(
        repo_root,
        runtime_inventory,
        snapshots,
        max_cache_bytes=MAX_EVAL_RUNTIME_BYTES - runtime_source_bytes,
        retained_cache_snapshots=retained_cache_snapshots,
    )
    runtime_bytes = runtime_source_bytes + cache_bytes
    _require_inventory_bounds(
        "eval_runtime_inventory",
        file_count=len(runtime_sources) + len(runtime_inventory.bytecode_caches),
        total_bytes=runtime_bytes,
        max_files=MAX_EVAL_RUNTIME_FILES,
        max_bytes=MAX_EVAL_RUNTIME_BYTES,
    )
    _verify_blocked_package_inventory(repo_root)
    final_runtime_inventory = _eval_runtime_source_inventory(repo_root)
    if final_runtime_inventory != runtime_inventory:
        _fail(
            "evaluator runtime inventory changed during bytecode authentication: "
            f"before={runtime_inventory!r}; after={final_runtime_inventory!r}"
        )
    _verify_runtime_snapshots_unchanged(
        repo_root,
        runtime_inventory,
        snapshots,
        retained_cache_snapshots,
    )
    return ValidatedRegistry(
        document,
        rows,
        snapshots,
        registry_snapshot,
        schema_snapshot,
    )


def _module_context(relative: str) -> tuple[str, tuple[str, ...]]:
    path = PurePosixPath(relative)
    if path.suffix != ".py":
        _fail(f"production import graph contains a non-Python source: {relative}")
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
        package = tuple(parts)
    else:
        package = tuple(parts[:-1])
    return ".".join(parts), package


def _compiled_cache_entries(
    repo_root: Path, directory: str, stem: str, context: str
) -> list[str]:
    cache = f"{directory}/__pycache__" if directory else "__pycache__"
    mode = _path_mode(repo_root, cache, context)
    if mode is None:
        return []
    if not stat.S_ISDIR(mode):
        _fail(f"{context} compiled-cache candidate is not a directory: {cache}")
    entries = _directory_entries(repo_root, cache, context)
    return [
        f"{cache}/{name}"
        for name in entries
        if name.startswith(f"{stem}.") and name.endswith(BYTECODE_SUFFIXES)
    ]


def _source_and_compiled_candidates(
    repo_root: Path,
    base: str,
    *,
    package_initializer: bool,
    context: str,
) -> tuple[list[str], list[str]]:
    stem = f"{base}/__init__" if package_initializer else base
    sources: list[str] = []
    compiled: list[str] = []
    for suffix in SOURCE_SUFFIXES:
        candidate = f"{stem}{suffix}"
        mode = _path_mode(repo_root, candidate, context)
        if mode is not None:
            if not stat.S_ISREG(mode):
                _fail(f"{context} source candidate is non-regular: {candidate}")
            sources.append(candidate)
    for suffix in COMPILED_SUFFIXES:
        candidate = f"{stem}{suffix}"
        mode = _path_mode(repo_root, candidate, context)
        if mode is not None:
            if not stat.S_ISREG(mode):
                _fail(f"{context} compiled candidate is non-regular: {candidate}")
            compiled.append(candidate)
    directory, name = stem.rsplit("/", 1) if "/" in stem else ("", stem)
    compiled.extend(_compiled_cache_entries(repo_root, directory, name, context))
    return sorted(set(sources)), sorted(set(compiled))


def _package_initializer_sources(
    repo_root: Path, package: str, context: str
) -> list[str]:
    mode = _path_mode(repo_root, package, context)
    if mode is None:
        return []
    if not stat.S_ISDIR(mode):
        _fail(f"{context} package-chain candidate is non-directory: {package}")
    sources, compiled = _source_and_compiled_candidates(
        repo_root,
        package,
        package_initializer=True,
        context=context,
    )
    if compiled:
        _fail(f"{context} found compiled package candidates: {compiled!r}")
    if len(sources) > 1:
        _fail(f"{context} found ambiguous package sources: {sources!r}")
    return sources


def _package_chain_sources(
    repo_root: Path, package: str, context: str
) -> list[str]:
    module_sources, module_compiled = _source_and_compiled_candidates(
        repo_root,
        package,
        package_initializer=False,
        context=context,
    )
    initializer_sources = _package_initializer_sources(repo_root, package, context)
    if module_compiled:
        _fail(
            f"{context} found compiled package-chain candidates: "
            f"{module_compiled!r}"
        )
    if module_sources:
        _fail(
            f"{context} found non-package source in package chain: "
            f"{module_sources!r}"
        )
    return initializer_sources


def _verify_entrypoint_ancestor_inventory(repo_root: Path) -> None:
    """Require namespace-only ancestors with no executable import candidate."""
    parts = ENTRYPOINT_MODULE.split(".")
    for length in range(1, len(parts)):
        ancestor = "/".join(parts[:length])
        context = f"blocked evaluator ancestor {ancestor!r}"
        module_sources, module_compiled = _source_and_compiled_candidates(
            repo_root,
            ancestor,
            package_initializer=False,
            context=context,
        )
        module_candidates = [*module_sources, *module_compiled]
        if module_candidates:
            _fail(
                f"{context} module candidate is forbidden: {module_candidates!r}"
            )
        mode = _path_mode(repo_root, ancestor, context)
        if mode is None or not stat.S_ISDIR(mode):
            _fail(f"{context} must be a real namespace directory")
        initializer_sources, initializer_compiled = _source_and_compiled_candidates(
            repo_root,
            ancestor,
            package_initializer=True,
            context=context,
        )
        initializer_candidates = [*initializer_sources, *initializer_compiled]
        if initializer_candidates:
            _fail(
                f"{context} initializer candidate is forbidden: "
                f"{initializer_candidates!r}"
            )


def _module_source_paths(repo_root: Path, module: str) -> set[str]:
    """Resolve every repo-local Python source or fail on non-source candidates."""
    if not module:
        return set()
    parts = module.split(".")
    if any(not part or SAFE_COMPONENT.fullmatch(part) is None for part in parts):
        return set()
    context = f"repo-local import {module!r}"
    sources: set[str] = set()
    for length in range(1, len(parts)):
        package = "/".join(parts[:length])
        sources.update(_package_chain_sources(repo_root, package, context))

    base = "/".join(parts)
    module_sources, module_compiled = _source_and_compiled_candidates(
        repo_root,
        base,
        package_initializer=False,
        context=context,
    )
    package_sources = _package_initializer_sources(repo_root, base, context)
    if module_compiled:
        _fail(f"{context} found compiled module candidates: {module_compiled!r}")
    final_sources = [*module_sources, *package_sources]
    if len(final_sources) > 1:
        _fail(f"{context} found ambiguous module/package sources: {final_sources!r}")
    sources.update(final_sources)
    return sources


def _imported_module_names(
    node: ast.Import | ast.ImportFrom, *, importer_relative: str
) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    _module, package = _module_context(importer_relative)
    if node.level:
        parents = node.level - 1
        if parents > len(package):
            _fail(
                f"relative import escapes evaluator package at "
                f"{importer_relative}:{node.lineno}"
            )
        base_parts = list(package[: len(package) - parents])
        if node.module:
            base_parts.extend(node.module.split("."))
        base = ".".join(base_parts)
    else:
        base = node.module or ""
    names = {base} if base else set()
    for alias in node.names:
        if alias.name != "*":
            names.add(".".join(part for part in (base, alias.name) if part))
    return names


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _blocked_refusal_shape_violations(tree: ast.Module, relative: str) -> list[str]:
    """Return deviations from the one admitted, non-executable refusal shape."""
    violations: list[str] = []
    if relative != ENTRYPOINT:
        return [f"{relative}: blocked boundary admits no production dependency"]
    if tree.type_ignores:
        violations.append("type-ignore directives are forbidden")
    if len(tree.body) != 6:
        return [f"top-level statement count must be 6, got {len(tree.body)}"]
    doc, future, access_gate, exports, error_type, evaluate = tree.body
    if not _is_docstring(doc):
        violations.append("first statement must be a module docstring")
    if not (
        isinstance(future, ast.ImportFrom)
        and future.module == "__future__"
        and future.level == 0
        and [(alias.name, alias.asname) for alias in future.names]
        == [("annotations", None)]
    ):
        violations.append("only 'from __future__ import annotations' is admitted")
    access_gate_ok = (
        isinstance(access_gate, ast.Raise)
        and access_gate.cause is None
        and isinstance(access_gate.exc, ast.Call)
        and isinstance(access_gate.exc.func, ast.Name)
        and access_gate.exc.func.id == "RuntimeError"
        and len(access_gate.exc.args) == 1
        and isinstance(access_gate.exc.args[0], ast.Constant)
        and access_gate.exc.args[0].value
        == (
            "ordinary package import is forbidden; use the "
            "descriptor-snapshot boundary"
        )
        and not access_gate.exc.keywords
    )
    if not access_gate_ok:
        violations.append("entrypoint must refuse every on-disk execution context")
    if not (
        isinstance(exports, ast.Assign)
        and len(exports.targets) == 1
        and isinstance(exports.targets[0], ast.Name)
        and exports.targets[0].id == "__all__"
        and isinstance(exports.value, ast.Tuple)
        and [
            element.value
            for element in exports.value.elts
            if isinstance(element, ast.Constant)
        ]
        == ["EvaluatorUnavailable", "evaluate"]
        and len(exports.value.elts) == 2
    ):
        violations.append("__all__ must expose only the refusal type and function")
    if not (
        isinstance(error_type, ast.ClassDef)
        and error_type.name == "EvaluatorUnavailable"
        and len(error_type.bases) == 1
        and isinstance(error_type.bases[0], ast.Name)
        and error_type.bases[0].id == "RuntimeError"
        and not error_type.keywords
        and not error_type.decorator_list
        and len(error_type.body) == 1
        and _is_docstring(error_type.body[0])
        and not getattr(error_type, "type_params", [])
    ):
        violations.append("EvaluatorUnavailable must be a docstring-only RuntimeError")
    evaluate_ok = (
        isinstance(evaluate, ast.FunctionDef)
        and evaluate.name == "evaluate"
        and not evaluate.decorator_list
        and not getattr(evaluate, "type_params", [])
        and not evaluate.args.posonlyargs
        and not evaluate.args.args
        and evaluate.args.vararg is None
        and not evaluate.args.kwonlyargs
        and evaluate.args.kwarg is None
        and not evaluate.args.defaults
        and not evaluate.args.kw_defaults
        and isinstance(evaluate.returns, ast.Constant)
        and evaluate.returns.value is None
        and len(evaluate.body) == 2
        and _is_docstring(evaluate.body[0])
        and isinstance(evaluate.body[1], ast.Raise)
    )
    if not evaluate_ok:
        violations.append(
            "evaluate must be a zero-argument, docstring-plus-raise refusal"
        )
    else:
        refusal = evaluate.body[1]
        assert isinstance(refusal, ast.Raise)
        exception = refusal.exc
        if not (
            refusal.cause is None
            and isinstance(exception, ast.Call)
            and isinstance(exception.func, ast.Name)
            and exception.func.id == "EvaluatorUnavailable"
            and len(exception.args) == 1
            and isinstance(exception.args[0], ast.Constant)
            and exception.args[0].value
            == (
                "production evaluator is blocked offline; required authorities are "
                "typed-unresolved"
            )
            and not exception.keywords
        ):
            violations.append(
                "evaluate must raise the one literal blocked-offline refusal"
            )
    return violations


def _strict_python_text(snapshot: FileSnapshot, context: str) -> str:
    """Decode once as UTF-8 and reject any alternate PEP 263 interpretation."""
    text = snapshot.utf8(context)
    for line in text.splitlines()[:2]:
        match = PEP263_CODING_COOKIE.match(line)
        if match is None:
            continue
        declared = match.group(1)
        normalized = declared.lower().replace("_", "-")
        if normalized not in {"utf-8", "utf8"}:
            _fail(
                f"{context} has a non-UTF-8 PEP 263 coding cookie: "
                f"{declared!r}"
            )
        break
    return text


def _parse_python_text(
    text: str, snapshot: FileSnapshot, context: str
) -> ast.Module:
    try:
        return ast.parse(text, filename=snapshot.relative_path)
    except SyntaxError as exc:
        _fail(f"{context} is not valid Python: {snapshot.relative_path}: {exc}")


def _parse_snapshot(snapshot: FileSnapshot, context: str) -> ast.Module:
    return _parse_python_text(
        _strict_python_text(snapshot, context), snapshot, context
    )


def _build_import_graph(
    document: dict[str, Any],
    repo_root: Path,
    snapshots: dict[str, FileSnapshot] | None = None,
) -> tuple[set[str], set[tuple[str, str]]]:
    _verify_entrypoint_ancestor_inventory(repo_root)
    rows = _classification_by_path(document)
    entrypoint = _safe_relative_path(
        document["production_entrypoint"], "production entrypoint"
    )
    if entrypoint not in rows:
        _fail("production entrypoint is absent from classification registry")
    retained = dict(snapshots or {})
    visited: set[str] = set()
    edges: set[tuple[str, str]] = set()
    pending = [entrypoint]
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        row = rows.get(relative)
        if row is None:
            _fail(f"unreviewed repo-local import defaults to deny: {relative}")
        if (
            row["import_policy"] != "allow"
            or row["classification"] not in PRODUCTION_CLASSIFICATIONS
            or row["review_status"] != "reviewed"
        ):
            _fail(f"classified source is denied from production evaluator: {relative}")
        snapshot = retained.get(relative)
        if snapshot is None:
            snapshot = _snapshot_file(repo_root, relative, "production import")
            retained[relative] = snapshot
        if snapshot.sha256 != row["sha256"]:
            _fail(f"production import hash differs from its retained row: {relative}")
        tree = _parse_snapshot(snapshot, "production import")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            modules = _imported_module_names(node, importer_relative=relative)
            targets: set[str] = set()
            for module in modules:
                targets.update(_module_source_paths(repo_root, module))
            for target in sorted(targets):
                if target == relative:
                    continue
                edges.add((relative, target))
                target_row = rows.get(target)
                if target_row is None:
                    _fail(
                        "unreviewed repo-local import defaults to deny: "
                        f"{relative} -> {target}"
                    )
                if target_row["import_policy"] != "allow":
                    _fail(
                        "repo-local import reaches a denied classification: "
                        f"{relative} -> {target}"
                    )
                pending.append(target)

        shape_violations = _blocked_refusal_shape_violations(tree, relative)
        if shape_violations:
            _fail(
                "blocked evaluator refusal-shape violation:\n"
                + "\n".join(shape_violations)
            )
        visited.add(relative)

    registered_allowed = {
        path for path, row in rows.items() if row["import_policy"] == "allow"
    }
    stale = sorted(registered_allowed - visited)
    if stale:
        _fail(f"allowed classification rows are stale or unreachable: {stale!r}")
    return visited, edges


def _exact_string_equal(left: Any, right: Any) -> bool:
    return (
        type(left) is str
        and type(right) is str
        and str.__eq__(left, right) is True
    )


def _exact_string_list(value: Any, context: str) -> tuple[str, ...]:
    if type(value) is not list:
        _fail(f"{context} must be an exact built-in list")
    result: list[str] = []
    for index in range(list.__len__(value)):
        item = list.__getitem__(value, index)
        if type(item) is not str:
            _fail(f"{context} contains a non-exact string entry")
        result.append(item)
    return tuple(result)


def _exact_string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _fail(f"{context} must be an exact built-in tuple")
    result: list[str] = []
    for index in range(tuple.__len__(value)):
        item = tuple.__getitem__(value, index)
        if type(item) is not str:
            _fail(f"{context} contains a non-exact string entry")
        result.append(item)
    return tuple(result)


def _require_exact_string_keys(mapping: Any, context: str) -> None:
    if type(mapping) is not dict:
        _fail(f"{context} must be an exact built-in dict")
    for key in dict.keys(mapping):
        if type(key) is not str:
            _fail(f"{context} contains a non-exact string key")


def _controlled_future_import_request(
    name: Any,
    globals_namespace: Any,
    locals_namespace: Any,
    fromlist: Any,
    level: Any,
) -> bool:
    if type(name) is not str:
        _fail("blocked evaluator import name is not an exact string")
    if type(level) is not int:
        _fail("blocked evaluator import level is not an exact integer")
    imported_names = _exact_string_tuple(
        fromlist, "blocked evaluator import fromlist"
    )
    if type(globals_namespace) is not dict or type(locals_namespace) is not dict:
        _fail("blocked evaluator import namespaces are not exact dictionaries")
    return name == "__future__" and imported_names == ("annotations",) and level == 0


def _validate_trusted_path_hooks(hooks: list[Any]) -> None:
    if _TRUSTED_PATH_HOOKS is None or len(_TRUSTED_PATH_HOOKS) != 2:
        _fail("trusted interpreter path-hook baseline is unavailable")
    if list.__len__(hooks) != 2:
        _fail("sys.path_hooks changed from the trusted interpreter state")
    zip_hook = list.__getitem__(hooks, 0)
    file_hook = list.__getitem__(hooks, 1)
    if zip_hook is not _ZIP_IMPORTER_TYPE:
        _fail("sys.path_hooks zip hook is not the exact built-in type")
    if (
        file_hook is not _TRUSTED_PATH_HOOKS[1]
        or type(file_hook) is not FunctionType
        or not _exact_string_equal(
            object.__getattribute__(file_hook, "__module__"),
            "_frozen_importlib_external",
        )
        or not _exact_string_equal(
            object.__getattribute__(file_hook, "__qualname__"),
            "FileFinder.path_hook.<locals>.path_hook_for_FileFinder",
        )
    ):
        _fail("sys.path_hooks FileFinder hook is not the exact frozen function")
    closure = object.__getattribute__(file_hook, "__closure__")
    if type(closure) is not tuple or tuple.__len__(closure) != 2:
        _fail("trusted FileFinder hook closure is malformed")
    finder_type = object.__getattribute__(
        tuple.__getitem__(closure, 0), "cell_contents"
    )
    loader_details = object.__getattribute__(
        tuple.__getitem__(closure, 1), "cell_contents"
    )
    if finder_type is not _FILE_FINDER_TYPE or type(loader_details) is not tuple:
        _fail("trusted FileFinder hook closure identity drifted")
    expected = (
        (importlib.machinery.ExtensionFileLoader, _IMPORT_EXTENSION_SUFFIXES),
        (importlib.machinery.SourceFileLoader, SOURCE_SUFFIXES),
        (importlib.machinery.SourcelessFileLoader, BYTECODE_SUFFIXES),
    )
    if tuple.__len__(loader_details) != len(expected):
        _fail("trusted FileFinder hook loader table drifted")
    for index, (expected_loader, expected_suffixes) in enumerate(expected):
        detail = tuple.__getitem__(loader_details, index)
        if type(detail) is not tuple or tuple.__len__(detail) != 2:
            _fail("trusted FileFinder hook loader row is malformed")
        loader = tuple.__getitem__(detail, 0)
        suffixes = tuple.__getitem__(detail, 1)
        if loader is not expected_loader:
            _fail("trusted FileFinder hook loader identity drifted")
        actual_suffixes = _exact_string_list(
            suffixes, "trusted FileFinder hook suffixes"
        )
        if len(actual_suffixes) != len(expected_suffixes):
            _fail("trusted FileFinder hook suffix table drifted")
        for suffix_index, expected_suffix in enumerate(expected_suffixes):
            if not _exact_string_equal(
                actual_suffixes[suffix_index], expected_suffix
            ):
                _fail("trusted FileFinder hook suffix table drifted")


def _safe_system_import_state() -> tuple[tuple[str, ...], dict[str, Any]]:
    if type(sys) is not ModuleType:
        _fail("sys binding is not the exact built-in module")
    sys_namespace = object.__getattribute__(sys, "__dict__")
    _require_exact_string_keys(sys_namespace, "sys module namespace")

    modules = dict.__getitem__(sys_namespace, "modules")
    _require_exact_string_keys(modules, "sys.modules")

    path_entries = _exact_string_list(
        dict.__getitem__(sys_namespace, "path"), "sys.path"
    )

    hooks = dict.__getitem__(sys_namespace, "path_hooks")
    if type(hooks) is not list:
        _fail("sys.path_hooks is not the exact trusted built-in list")
    _validate_trusted_path_hooks(hooks)

    importer_cache = dict.__getitem__(sys_namespace, "path_importer_cache")
    _require_exact_string_keys(importer_cache, "sys.path_importer_cache")
    for cache_key in dict.keys(importer_cache):
        finder = dict.__getitem__(importer_cache, cache_key)
        if finder is None:
            continue
        if type(finder) is _FILE_FINDER_TYPE:
            finder_state = object.__getattribute__(finder, "__dict__")
            _require_exact_string_keys(finder_state, "cached FileFinder state")
            finder_path = dict.get(finder_state, "path")
            if not _exact_string_equal(finder_path, cache_key):
                _fail("cached FileFinder path differs from its exact cache key")
            continue
        if type(finder) is _ZIP_IMPORTER_TYPE:
            _fail("zip path importer is not admitted by the evaluator boundary")
        _fail("sys.path_importer_cache contains an untrusted finder")
    return path_entries, modules


def _loader_shadow_paths(
    module_name: str, repo_root: Path, path_entries: tuple[str, ...]
) -> list[str]:
    parts = module_name.split(".")
    full_expected = repo_root / PurePosixPath(*parts)
    relative_candidates: list[tuple[PurePosixPath, Path]] = [
        (PurePosixPath(*parts[:length]), repo_root / PurePosixPath(*parts[:length]))
        for length in range(1, len(parts) + 1)
    ]
    relative_candidates.append((PurePosixPath(parts[-1]), full_expected))
    candidates: list[str] = []
    repository_parent_entries: list[str] = []
    resolved_repo_root = repo_root.resolve()
    for raw_entry in path_entries:
        entry = Path.cwd() if _exact_string_equal(raw_entry, "") else Path(raw_entry)
        try:
            entry_metadata = entry.lstat()
        except (FileNotFoundError, NotADirectoryError, OSError):
            entry_metadata = None
        if entry_metadata is not None:
            if stat.S_ISLNK(entry_metadata.st_mode):
                candidates.append(f"symlinked sys.path entry:{entry}")
            elif not stat.S_ISDIR(entry_metadata.st_mode):
                candidates.append(f"non-directory sys.path entry:{entry}")
        try:
            if entry.resolve() == resolved_repo_root:
                repository_parent_entries.append(raw_entry)
        except OSError:
            pass
        for relative, expected in relative_candidates:
            base = entry / relative
            import_candidates = [
                base,
                *(Path(f"{base}{suffix}") for suffix in SOURCE_SUFFIXES),
                *(Path(f"{base}{suffix}") for suffix in COMPILED_SUFFIXES),
            ]
            for candidate in import_candidates:
                try:
                    metadata = candidate.lstat()
                except (FileNotFoundError, NotADirectoryError, OSError):
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    candidates.append(f"symlink:{candidate}")
                    continue
                try:
                    if candidate.resolve() == expected.resolve():
                        continue
                except OSError:
                    pass
                candidates.append(str(candidate))
    if module_name == ENTRYPOINT_MODULE:
        if len(repository_parent_entries) != 1:
            candidates.append("repository-parent path is missing or aliased")
        else:
            parent_entry = repository_parent_entries[0]
            canonical_root = str(resolved_repo_root)
            if not (
                _exact_string_equal(parent_entry, "")
                or _exact_string_equal(parent_entry, canonical_root)
            ):
                candidates.append("repository-parent path uses a non-canonical alias")
    return sorted(set(candidates))


def _loaded_namespace_ancestor_violation(
    name: str,
    module: Any,
    expected: Path,
    parent_path: tuple[str, ...],
) -> str | None:
    if type(module) is not ModuleType:
        return "object is not an exact module"
    namespace = object.__getattribute__(module, "__dict__")
    try:
        _require_exact_string_keys(namespace, "ancestor module namespace")
    except RegistryViolation as exc:
        return str(exc)
    for forbidden in (
        "production_evaluator",
        "__getattr__",
        "__getattribute__",
        "__dir__",
    ):
        if dict.__contains__(namespace, forbidden):
            return f"parent has forbidden direct attribute {forbidden!r}"

    spec = dict.get(namespace, "__spec__")
    if type(spec) is not _MODULE_SPEC_TYPE:
        return "ModuleSpec is absent or non-standard"
    spec_state = object.__getattribute__(spec, "__dict__")
    try:
        _require_exact_string_keys(spec_state, "ancestor ModuleSpec state")
    except RegistryViolation as exc:
        return str(exc)
    loader = dict.get(spec_state, "loader")
    if type(loader) is not _NAMESPACE_LOADER_TYPE:
        return "loader is not the exact materialized NamespaceLoader"
    namespace_path = dict.get(spec_state, "submodule_search_locations")
    if type(namespace_path) is not _NAMESPACE_PATH_TYPE:
        return "search locations are not the exact materialized NamespacePath"
    if dict.get(namespace, "__path__") is not namespace_path:
        return "module path differs by identity from ModuleSpec search locations"
    if dict.get(namespace, "__loader__") is not loader:
        return "module loader differs by identity from ModuleSpec loader"

    loader_state = object.__getattribute__(loader, "__dict__")
    try:
        _require_exact_string_keys(loader_state, "NamespaceLoader state")
    except RegistryViolation as exc:
        return str(exc)
    if (
        dict.__len__(loader_state) != 1
        or dict.get(loader_state, "_path") is not namespace_path
    ):
        return "NamespaceLoader path identity is inconsistent"

    spec_name = dict.get(spec_state, "name")
    if not _exact_string_equal(spec_name, name):
        return "ModuleSpec name differs from the exact ancestor name"
    if not _exact_string_equal(dict.get(namespace, "__name__"), name):
        return "module name differs from the exact ancestor name"
    if not _exact_string_equal(dict.get(namespace, "__package__"), name):
        return "module package differs from the exact ancestor name"
    if (
        dict.get(spec_state, "origin") is not None
        or dict.get(spec_state, "loader_state") is not None
        or dict.get(spec_state, "_cached") is not None
        or dict.get(namespace, "__file__") is not None
        or dict.get(namespace, "__cached__") is not None
        or dict.get(spec_state, "_set_fileattr") is not False
        or dict.get(spec_state, "_initializing") is not False
    ):
        return "ModuleSpec or module has executable location state"
    pending = dict.get(spec_state, "_uninitialized_submodules")
    if type(pending) is not list or list.__len__(pending) != 0:
        return "ModuleSpec has unfinished child imports"

    path_state = object.__getattribute__(namespace_path, "__dict__")
    try:
        _require_exact_string_keys(path_state, "NamespacePath state")
    except RegistryViolation as exc:
        return str(exc)
    if not _exact_string_equal(dict.get(path_state, "_name"), name):
        return "NamespacePath name differs from the exact ancestor name"
    portions = dict.get(path_state, "_path")
    if type(portions) is not list or list.__len__(portions) != 1:
        return "NamespacePath has missing or alternate portions"
    expected_text = str(expected)
    if not _exact_string_equal(list.__getitem__(portions, 0), expected_text):
        return "NamespacePath portion differs from the descriptor-validated path"

    try:
        previous_parent = _exact_string_tuple(
            dict.get(path_state, "_last_parent_path"),
            "NamespacePath parent snapshot",
        )
    except RegistryViolation as exc:
        return str(exc)
    if len(previous_parent) != len(parent_path):
        return "NamespacePath parent snapshot is stale"
    for index, current in enumerate(parent_path):
        if not _exact_string_equal(previous_parent[index], current):
            return "NamespacePath parent snapshot is stale"

    previous_epoch = dict.get(path_state, "_last_epoch")
    current_epoch = type.__getattribute__(_NAMESPACE_PATH_TYPE, "_epoch")
    if (
        type(previous_epoch) is not int
        or type(current_epoch) is not int
        or previous_epoch != current_epoch
    ):
        return "NamespacePath cache epoch is stale"
    path_finder = dict.get(path_state, "_path_finder")
    if (
        type(path_finder) is not MethodType
        or object.__getattribute__(path_finder, "__self__") is not _PATH_FINDER_TYPE
        or object.__getattribute__(path_finder, "__func__") is not _PATH_FINDER_GET_SPEC
    ):
        return "NamespacePath finder binding is not the exact frozen PathFinder"
    return None


def _preloaded_module_conflicts(
    module_name: str,
    repo_root: Path,
    modules: dict[str, Any],
    parent_path: tuple[str, ...],
) -> list[str]:
    parts = module_name.split(".")
    ancestors = {
        ".".join(parts[:length]): repo_root / PurePosixPath(*parts[:length])
        for length in range(1, len(parts))
    }
    conflicts: list[str] = []
    for name in dict.keys(modules):
        if _exact_string_equal(name, module_name) or str.startswith(
            name, f"{module_name}."
        ):
            conflicts.append(name)
            continue
        expected = dict.get(ancestors, name)
        if expected is None:
            continue
        module = dict.__getitem__(modules, name)
        violation = _loaded_namespace_ancestor_violation(
            name, module, expected, parent_path
        )
        if violation is not None:
            conflicts.append(f"{name} ({violation})")
    return sorted(conflicts)


def _execute_authenticated_snapshot_ast(
    snapshot: FileSnapshot, *, execution_name: str
) -> dict[str, Any]:
    tree = _parse_snapshot(snapshot, "blocked evaluator execution snapshot")
    violations = _blocked_refusal_shape_violations(tree, snapshot.relative_path)
    if violations:
        _fail("blocked evaluator refusal-shape violation:\n" + "\n".join(violations))
    execution_tree = copy.deepcopy(tree)
    del execution_tree.body[2]
    ast.fix_missing_locations(execution_tree)
    code = _TRUSTED_COMPILE(
        execution_tree,
        snapshot.relative_path,
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    future_stub = ModuleType("__future__")
    future_stub.annotations = object()

    def controlled_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if _controlled_future_import_request(
            name, globals, locals, fromlist, level
        ):
            return future_stub
        _fail(f"blocked evaluator attempted an unapproved import: {name!r}")

    safe_builtins = {
        "RuntimeError": _TRUSTED_RUNTIME_ERROR,
        "__build_class__": _TRUSTED_BUILD_CLASS,
        "__import__": controlled_import,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": execution_name,
        "__package__": None,
    }
    _TRUSTED_EXEC(code, namespace, namespace)
    return namespace


def _execute_validated_blocked_snapshot(
    snapshot: FileSnapshot, *, module_name: str, repo_root: Path
) -> dict[str, Any]:
    """Execute after strict, callback-free host import-state validation."""
    if module_name == ENTRYPOINT_MODULE:
        _verify_entrypoint_ancestor_inventory(repo_root)
    path_entries, modules = _safe_system_import_state()
    shadows = _loader_shadow_paths(module_name, repo_root, path_entries)
    if shadows:
        _fail(f"hostile sys.path module shadow state: {shadows!r}")
    conflicting_modules = _preloaded_module_conflicts(
        module_name, repo_root, modules, path_entries
    )
    if conflicting_modules:
        _fail(f"hostile or preloaded sys.modules state: {conflicting_modules!r}")
    return _execute_authenticated_snapshot_ast(
        snapshot, execution_name=module_name
    )


def load_blocked_evaluator_snapshot(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate and execute the sole hash-bound descriptor snapshot.

    Ordinary package import is intentionally unsupported. This function is the
    complete blocked-boundary access API: it verifies registry/schema bindings,
    ancestor and package inventories, the transitive graph, and source shape.
    It then removes the exact validated unconditional refusal from a private AST
    copy before compilation; no public module-name value can bypass the on-disk
    refusal.
    """
    _safe_system_import_state()
    validated = _validate_registry(repo_root)
    _build_import_graph(validated.document, repo_root, validated.snapshots)
    return _execute_validated_blocked_snapshot(
        validated.snapshots[ENTRYPOINT],
        module_name=ENTRYPOINT_MODULE,
        repo_root=repo_root,
    )


def _isolated_parent_violation(name: str, module: Any) -> str | None:
    """Validate only immutable parent identity used before detaching it."""
    if type(module) is not ModuleType:
        return "object is not an exact module"
    namespace = object.__getattribute__(module, "__dict__")
    try:
        _require_exact_string_keys(namespace, "isolated parent module namespace")
    except RegistryViolation as exc:
        return str(exc)
    for forbidden in (
        "production_evaluator",
        "__getattr__",
        "__getattribute__",
        "__dir__",
    ):
        if dict.__contains__(namespace, forbidden):
            return f"parent has forbidden direct attribute {forbidden!r}"
    spec = dict.get(namespace, "__spec__")
    if type(spec) is not _MODULE_SPEC_TYPE:
        return "ModuleSpec is absent or non-standard"
    spec_state = object.__getattribute__(spec, "__dict__")
    try:
        _require_exact_string_keys(spec_state, "isolated parent ModuleSpec state")
    except RegistryViolation as exc:
        return str(exc)
    loader = dict.get(spec_state, "loader")
    path = dict.get(spec_state, "submodule_search_locations")
    if type(loader) is not _NAMESPACE_LOADER_TYPE:
        return "loader is not the exact materialized NamespaceLoader"
    if type(path) is not _NAMESPACE_PATH_TYPE:
        return "search locations are not the exact materialized NamespacePath"
    if (
        dict.get(namespace, "__loader__") is not loader
        or dict.get(namespace, "__path__") is not path
    ):
        return "parent loader or path identity is inconsistent"
    loader_state = object.__getattribute__(loader, "__dict__")
    try:
        _require_exact_string_keys(loader_state, "isolated NamespaceLoader state")
    except RegistryViolation as exc:
        return str(exc)
    if (
        dict.__len__(loader_state) != 1
        or dict.get(loader_state, "_path") is not path
    ):
        return "NamespaceLoader path identity is inconsistent"
    if (
        not _exact_string_equal(dict.get(spec_state, "name"), name)
        or not _exact_string_equal(dict.get(namespace, "__name__"), name)
        or not _exact_string_equal(dict.get(namespace, "__package__"), name)
    ):
        return "parent name state is inconsistent"
    if (
        dict.get(spec_state, "origin") is not None
        or dict.get(spec_state, "loader_state") is not None
        or dict.get(spec_state, "_cached") is not None
        or dict.get(namespace, "__file__") is not None
        or dict.get(namespace, "__cached__") is not None
    ):
        return "parent has executable location state"
    return None


def load_blocked_evaluator_snapshot_isolated(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute authenticated bytes with validated parents detached exactly.

    This test loader is used when a legitimate preloaded NamespacePath is stale
    because collection changed sys.path. It never reads or recalculates that
    path: exact parent modules are checked through direct dict state, detached
    with exact-dict operations, and restored by identity after AST execution.
    """
    _safe_system_import_state()
    validated = _validate_registry(repo_root)
    _build_import_graph(validated.document, repo_root, validated.snapshots)
    _verify_entrypoint_ancestor_inventory(repo_root)
    path_entries, modules = _safe_system_import_state()
    shadows = _loader_shadow_paths(ENTRYPOINT_MODULE, repo_root, path_entries)
    if shadows:
        _fail(f"hostile sys.path module shadow state: {shadows!r}")

    parts = ENTRYPOINT_MODULE.split(".")
    ancestor_names = [
        ".".join(parts[:length]) for length in range(1, len(parts))
    ]
    conflicts: list[str] = []
    for name in dict.keys(modules):
        if _exact_string_equal(name, ENTRYPOINT_MODULE) or str.startswith(
            name, f"{ENTRYPOINT_MODULE}."
        ):
            conflicts.append(name)
    detached: list[tuple[str, ModuleType]] = []
    for name in ancestor_names:
        if not dict.__contains__(modules, name):
            continue
        module = dict.__getitem__(modules, name)
        violation = _isolated_parent_violation(name, module)
        if violation is not None:
            conflicts.append(f"{name} ({violation})")
        else:
            assert type(module) is ModuleType
            detached.append((name, module))
    if conflicts:
        _fail(f"hostile or preloaded sys.modules state: {sorted(conflicts)!r}")

    for name, _module in reversed(detached):
        dict.__delitem__(modules, name)
    try:
        return _execute_authenticated_snapshot_ast(
            validated.snapshots[ENTRYPOINT],
            execution_name="_gpu_simulator_retained_evaluator_isolated_v1",
        )
    finally:
        for name, module in detached:
            dict.__setitem__(modules, name, module)


def _sim_snapshots(sim_root: Path) -> list[FileSnapshot]:
    excluded = {
        ("target",),
        ("trace", "pbgen", "target"),
    }
    if not sim_root.is_dir() or sim_root.is_symlink():
        _fail(f"sim source root is missing or symlinked: {sim_root}")
    snapshots: list[FileSnapshot] = []
    total_bytes = 0

    def walk_error(error: OSError) -> None:
        _fail(f"sim source inventory is unreadable: {error}")

    for current, directory_names, file_names in os.walk(
        sim_root,
        followlinks=False,
        onerror=walk_error,
    ):
        current_path = Path(current)
        current_parts = current_path.relative_to(sim_root).parts
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            relative_parts = (*current_parts, name)
            if any(relative_parts[: len(prefix)] == prefix for prefix in excluded):
                continue
            candidate = current_path / name
            if candidate.is_symlink():
                _fail(
                    "sim source tree contains a symlinked directory: "
                    f"{candidate.relative_to(sim_root)}"
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            candidate = current_path / name
            relative = candidate.relative_to(sim_root)
            if candidate.is_symlink():
                _fail(f"sim source tree contains a symlinked file: {relative}")
            if candidate.suffix not in SIM_SOURCE_SUFFIXES:
                continue
            snapshots.append(
                _snapshot_file(sim_root, relative.as_posix(), "sim source")
            )
            total_bytes += len(snapshots[-1].data)
            _require_inventory_bounds(
                "sim_source_inventory",
                file_count=len(snapshots),
                total_bytes=total_bytes,
                max_files=MAX_SIM_SOURCE_FILES,
                max_bytes=MAX_SIM_SOURCE_BYTES,
            )
    return snapshots


def _constant_string(
    expression: ast.AST, bindings: dict[str, str] | None = None
) -> str | None:
    known = bindings or {}
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.Name):
        return known.get(expression.id)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _constant_string(expression.left, known)
        right = _constant_string(expression.right, known)
        if left is None or right is None or len(left) + len(right) > 4096:
            return None
        return left + right
    if isinstance(expression, ast.JoinedStr):
        pieces: list[str] = []
        for value in expression.values:
            piece = _constant_string(value, known)
            if piece is None:
                return None
            pieces.append(piece)
        joined = "".join(pieces)
        return joined if len(joined) <= 4096 else None
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "join"
        and len(expression.args) == 1
        and not expression.keywords
        and isinstance(expression.args[0], (ast.List, ast.Tuple))
        and len(expression.args[0].elts) <= 128
    ):
        separator = _constant_string(expression.func.value, known)
        if separator is None:
            return None
        pieces = [
            _constant_string(element, known) for element in expression.args[0].elts
        ]
        if any(piece is None for piece in pieces):
            return None
        joined = separator.join(piece for piece in pieces if piece is not None)
        return joined if len(joined) <= 4096 else None
    return None


def _literal_string_bindings(tree: ast.Module) -> dict[str, str]:
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            assignments.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assignments.setdefault(node.target.id, []).append(node)

    unique = {
        name: values[0] for name, values in assignments.items() if len(values) == 1
    }
    bindings: dict[str, str] = {}
    for _ in range(min(len(unique) + 1, 64)):
        changed = False
        for name, expression in unique.items():
            value = _constant_string(expression, bindings)
            if value is not None and bindings.get(name) != value:
                bindings[name] = value
                changed = True
        if not changed:
            break
    return bindings


def _reflected_callable_name(
    expression: ast.AST, bindings: dict[str, str]
) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    if isinstance(expression, ast.Subscript):
        return _constant_string(expression.slice, bindings) or ""
    if isinstance(expression, ast.Call):
        callee = _reflected_callable_name(expression.func, bindings)
        if callee == "getattr" and len(expression.args) >= 2:
            return _constant_string(expression.args[1], bindings) or ""
    return ""


def _callable_reference_names(
    expression: ast.AST, bindings: dict[str, str]
) -> set[str]:
    if isinstance(expression, ast.Name):
        return {expression.id}
    if isinstance(expression, ast.Attribute):
        return {expression.attr}
    if isinstance(expression, ast.Call):
        callee = _reflected_callable_name(expression.func, bindings)
        if callee == "getattr" and len(expression.args) >= 2:
            attribute = _constant_string(expression.args[1], bindings)
            return {attribute} if attribute else set()
        return _callable_reference_names(expression.func, bindings)
    if isinstance(expression, ast.Subscript):
        names = _callable_reference_names(expression.value, bindings)
        key = _constant_string(expression.slice, bindings)
        if key:
            names.add(key)
        return names
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        names: set[str] = set()
        for element in expression.elts:
            names.update(_callable_reference_names(element, bindings))
        return names
    if isinstance(expression, ast.Dict):
        names = set()
        for value in expression.values:
            names.update(_callable_reference_names(value, bindings))
        return names
    if isinstance(expression, ast.IfExp):
        return _callable_reference_names(
            expression.body, bindings
        ) | _callable_reference_names(expression.orelse, bindings)
    if isinstance(expression, ast.NamedExpr):
        return _callable_reference_names(expression.value, bindings)
    if isinstance(expression, ast.Starred):
        return _callable_reference_names(expression.value, bindings)
    return set()


def _reflection_container_name(
    expression: ast.AST, bindings: dict[str, str]
) -> str:
    if isinstance(expression, ast.Attribute) and expression.attr in {
        "__dict__",
        "modules",
    }:
        return expression.attr
    if isinstance(expression, ast.Call):
        callee = _reflected_callable_name(expression.func, bindings)
        if callee in {"globals", "locals", "vars"}:
            return callee
        if callee == "getattr" and len(expression.args) >= 2:
            attribute = _constant_string(expression.args[1], bindings)
            if attribute in {"__dict__", "modules"}:
                return attribute
    return ""


def _assignment_target_alias_names(
    target: ast.AST, bindings: dict[str, str]
) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.arg):
        return {target.arg}
    if isinstance(target, ast.Starred):
        return _assignment_target_alias_names(target.value, bindings)
    if isinstance(target, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assignment_target_alias_names(element, bindings))
        return names
    if isinstance(target, ast.Attribute):
        return {target.attr}
    if isinstance(target, ast.Subscript):
        return _callable_reference_names(target.value, bindings)
    return set()


def _callable_alias_bindings(
    tree: ast.Module,
    string_bindings: dict[str, str],
    dangerous_names: set[str],
) -> dict[str, str]:
    """Resolve callable aliases, defaults, and static containers without execution."""
    relations: dict[str, set[str]] = {}

    def bind(target: ast.AST, value: ast.AST) -> None:
        sources = _callable_reference_names(value, string_bindings)
        if not sources:
            return
        for target_name in _assignment_target_alias_names(target, string_bindings):
            relations.setdefault(target_name, set()).update(sources)

    def bind_defaults(arguments: ast.arguments) -> None:
        positional = [*arguments.posonlyargs, *arguments.args]
        if arguments.defaults:
            for argument, default in zip(
                positional[-len(arguments.defaults) :], arguments.defaults
            ):
                bind(argument, default)
        for argument, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults
        ):
            if default is not None:
                bind(argument, default)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target, node.value)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in dangerous_names:
                    relations.setdefault(alias.asname or alias.name, set()).add(
                        alias.name
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            bind_defaults(node.args)

    reverse_relations: dict[str, set[str]] = {}
    for alias_name, source_names in relations.items():
        for source_name in source_names:
            reverse_relations.setdefault(source_name, set()).add(alias_name)

    resolved: dict[str, str] = {}
    pending: list[tuple[str, str]] = [
        (dangerous_name, dangerous_name)
        for dangerous_name in sorted(dangerous_names)
    ]
    index = 0
    while index < len(pending):
        source_name, canonical = pending[index]
        index += 1
        for alias_name in sorted(reverse_relations.get(source_name, set())):
            prior = resolved.get(alias_name)
            if prior is not None and prior <= canonical:
                continue
            resolved[alias_name] = canonical
            pending.append((alias_name, canonical))
    return resolved


def _resolved_dangerous_callable(
    expression: ast.AST,
    bindings: dict[str, str],
    aliases: dict[str, str],
    dangerous_names: set[str],
) -> tuple[str, str] | None:
    candidates: list[tuple[str, str]] = []
    for observed in _callable_reference_names(expression, bindings):
        canonical = aliases.get(observed, observed)
        if canonical in dangerous_names:
            candidates.append((canonical, observed))
    if not candidates:
        return None
    canonical, observed = sorted(candidates)[0]
    return observed, canonical


def _sim_python_dynamic_violations(
    tree: ast.Module, relative: str, module_name: str
) -> list[str]:
    violations: list[str] = []
    evaluator_spellings = {
        module_name,
        module_name.replace(".", "/"),
        module_name.replace(".", "::"),
        module_name.rsplit(".", 1)[-1],
    }
    bindings = _literal_string_bindings(tree)
    dynamic_aliases = _callable_alias_bindings(
        tree, bindings, SIM_PYTHON_DYNAMIC_CALL_NAMES
    )
    process_aliases = _callable_alias_bindings(
        tree, bindings, SIM_PYTHON_PROCESS_CALL_NAMES
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in SIM_PYTHON_DYNAMIC_MODULES:
                    violations.append(
                        f"{relative}:{node.lineno}: dynamic import module "
                        f"{alias.name!r}"
                    )
                if top in SIM_PYTHON_PROCESS_MODULES:
                    violations.append(
                        f"{relative}:{node.lineno}: process module {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".", 1)[0]
            if top in SIM_PYTHON_DYNAMIC_MODULES:
                violations.append(
                    f"{relative}:{node.lineno}: dynamic import module {node.module!r}"
                )
            if top in SIM_PYTHON_PROCESS_MODULES:
                violations.append(
                    f"{relative}:{node.lineno}: process module {node.module!r}"
                )
            for alias in node.names:
                if alias.name in SIM_PYTHON_DYNAMIC_CALL_NAMES:
                    violations.append(
                        f"{relative}:{node.lineno}: dynamic API import {alias.name!r}"
                    )
                if alias.name in SIM_PYTHON_PROCESS_CALL_NAMES:
                    violations.append(
                        f"{relative}:{node.lineno}: process API import {alias.name!r}"
                    )
        elif isinstance(node, ast.Call):
            dynamic_callable = _resolved_dangerous_callable(
                node.func,
                bindings,
                dynamic_aliases,
                SIM_PYTHON_DYNAMIC_CALL_NAMES,
            )
            process_callable = _resolved_dangerous_callable(
                node.func,
                bindings,
                process_aliases,
                SIM_PYTHON_PROCESS_CALL_NAMES,
            )
            if dynamic_callable is not None:
                name, dynamic_name = dynamic_callable
                alias_detail = (
                    f" alias {name!r} -> {dynamic_name!r}"
                    if name in dynamic_aliases
                    else f" {dynamic_name!r}"
                )
                violations.append(
                    f"{relative}:{node.lineno}: dynamic code/import API"
                    f"{alias_detail}"
                )
            if process_callable is not None:
                name, process_name = process_callable
                alias_detail = (
                    f" alias {name!r} -> {process_name!r}"
                    if name in process_aliases
                    else f" {process_name!r}"
                )
                violations.append(
                    f"{relative}:{node.lineno}: process execution API"
                    f"{alias_detail}"
                )
            for argument in [*node.args, *(item.value for item in node.keywords)]:
                value = _constant_string(argument, bindings)
                if value in evaluator_spellings:
                    violations.append(
                        f"{relative}:{node.lineno}: computed evaluator module "
                        f"reference {value!r}"
                    )
        elif isinstance(node, ast.Name):
            if node.id in SIM_PYTHON_DYNAMIC_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: dynamic code/import reference {node.id!r}"
                )
            if node.id in SIM_PYTHON_PROCESS_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: process execution reference {node.id!r}"
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in SIM_PYTHON_DYNAMIC_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: dynamic code/import reference {node.attr!r}"
                )
            if node.attr in SIM_PYTHON_PROCESS_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: process execution reference {node.attr!r}"
                )
        elif isinstance(node, ast.Subscript):
            container = _reflection_container_name(node.value, bindings)
            if container:
                key = _constant_string(node.slice, bindings)
                detail = f" with key {key!r}" if key is not None else ""
                violations.append(
                    f"{relative}:{node.lineno}: reflective subscript "
                    f"{container!r}{detail} is forbidden"
                )
        else:
            value = _constant_string(node, bindings)
            if value is None:
                continue
            if value in SIM_PYTHON_DYNAMIC_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: dynamic code/import name {value!r}"
                )
            if value in SIM_PYTHON_PROCESS_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: process execution name {value!r}"
                )
            if not isinstance(node, ast.Constant) and value in evaluator_spellings:
                violations.append(
                    f"{relative}:{node.lineno}: computed evaluator reference {value!r}"
                )
    return violations


def _sim_evaluator_references(sim_root: Path, module_name: str) -> list[str]:
    """Guard direct references and recognized Python loading APIs.

    Python sources reject the closed dynamic/process API vocabulary above,
    including calls fed by runtime arguments. For non-Python sources this claims
    only direct-token exclusion. It does not claim to decide arbitrary reflective,
    native, Rust, C++, or shell runtime input values.
    """
    leaf = module_name.rsplit(".", 1)[-1]
    token = re.compile(
        rf"(?<![A-Za-z0-9_])(?:{re.escape(module_name)}|"
        rf"{re.escape(module_name.replace('.', '/'))}|"
        rf"{re.escape(module_name.replace('.', '::'))}|{re.escape(leaf)})"
        rf"(?![A-Za-z0-9_])"
    )
    violations: list[str] = []
    for snapshot in _sim_snapshots(sim_root):
        is_python = PurePosixPath(snapshot.relative_path).suffix == ".py"
        if is_python:
            text = _strict_python_text(snapshot, "sim Python source")
            tree = _parse_python_text(
                text, snapshot, "sim Python source"
            )
        else:
            text = snapshot.utf8("sim source")
            tree = None
        for match in token.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(
                f"{snapshot.relative_path}:{line}: direct evaluator token "
                f"{match.group(0)!r}"
            )
        if tree is not None:
            violations.extend(
                _sim_python_dynamic_violations(
                    tree, snapshot.relative_path, module_name
                )
            )
    return sorted(set(violations))


def _write_graph_fixture(
    repo_root: Path, source: bytes
) -> tuple[dict[str, Any], dict[str, FileSnapshot]]:
    entrypoint = repo_root / ENTRYPOINT
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_bytes(source)
    snapshot = _snapshot_file(repo_root, ENTRYPOINT, "fixture entrypoint")
    document = {
        "classifications": [
            {
                "classification": "production_boundary",
                "import_policy": "allow",
                "path": ENTRYPOINT,
                "review_status": "reviewed",
                "sha256": snapshot.sha256,
            }
        ],
        "production_entrypoint": ENTRYPOINT,
    }
    return document, {ENTRYPOINT: snapshot}


def test_registry_schema_hashes_and_classifications_are_closed() -> None:
    validated = _validate_registry()
    document = validated.document
    assert document["$schema"] == SCHEMA_ID
    assert validated.rows[ENTRYPOINT]["classification"] == "production_boundary"
    assert document["default_repo_local_import_policy"] == "deny"
    assert document["boundary"] == {
        "entrypoint_access": "descriptor_snapshot_only",
        "label_value_reads": "forbidden",
        "ordinary_package_import": "forbidden",
        "score_emission": "none",
        "status": "blocked_offline",
    }


def test_registry_rewalk_rejects_terminal_runtime_cache_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_inventory = _eval_runtime_source_inventory
    inventory_calls = 0
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    planted = f"eval/__pycache__/late.{cache_tag}.pyc"

    def inventory_with_terminal_drift(repo_root: Path) -> EvalRuntimeInventory:
        nonlocal inventory_calls
        inventory_calls += 1
        inventory = real_inventory(repo_root)
        if inventory_calls == 1:
            return inventory
        return EvalRuntimeInventory(
            inventory.sources,
            tuple(sorted((*inventory.bytecode_caches, planted))),
        )

    monkeypatch.setattr(
        sys.modules[__name__],
        "_eval_runtime_source_inventory",
        inventory_with_terminal_drift,
    )

    with pytest.raises(
        RegistryViolation,
        match="runtime inventory changed during bytecode authentication",
    ):
        _validate_registry()

    assert inventory_calls == 2


@pytest.mark.parametrize(
    "mutation",
    ("same_bytes_replacement", "same_inode_content_drift"),
)
def test_terminal_runtime_snapshot_rejects_same_path_cache_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_relative = "eval/benign.py"
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_relative = f"eval/__pycache__/benign.{cache_tag}.pyc"
    source_path = tmp_path / source_relative
    cache_path = tmp_path / cache_relative
    source_path.parent.mkdir(parents=True)
    cache_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"value = 1\n")
    cache_bytes = b"authenticated-cache-bytes"
    cache_path.write_bytes(cache_bytes)
    source_snapshot = _snapshot_file(
        tmp_path,
        source_relative,
        "terminal source fixture",
    )
    cache_snapshot = _snapshot_file(
        tmp_path,
        cache_relative,
        "terminal cache fixture",
    )
    inventory = EvalRuntimeInventory(
        frozenset({source_relative}),
        (cache_relative,),
    )

    if mutation == "same_bytes_replacement":
        replacement = cache_path.with_name(".cache-replacement")
        replacement.write_bytes(cache_bytes)
        os.replace(replacement, cache_path)
        assert cache_path.stat().st_ino != cache_snapshot.identity[1]
    else:
        before_inode = cache_path.stat().st_ino
        cache_path.write_bytes(b"X" * len(cache_bytes))
        assert len(cache_path.read_bytes()) == len(cache_bytes)
        assert cache_path.stat().st_ino == before_inode

    with pytest.raises(
        RegistryViolation,
        match="runtime source/cache changed during bytecode authentication",
    ):
        _verify_runtime_snapshots_unchanged(
            tmp_path,
            inventory,
            {source_relative: source_snapshot},
            {cache_relative: cache_snapshot},
        )


def test_production_evaluator_transitive_import_graph_is_closed() -> None:
    validated = _validate_registry()
    visited, edges = _build_import_graph(
        validated.document,
        REPO_ROOT,
        validated.snapshots,
    )
    assert visited == {ENTRYPOINT}
    assert edges == set()


@pytest.fixture(scope="module", autouse=True)
def _canonical_evaluator_guard_import_state() -> Any:
    """Collapse only duplicate exact repo paths introduced during collection."""
    path_entries, modules = _safe_system_import_state()
    resolved_repository_root = REPO_ROOT.resolve()
    repository_root = str(resolved_repository_root)

    def normalize_repository_roots(
        entries: tuple[str, ...], context: str
    ) -> tuple[list[str], int]:
        normalized_entries: list[str] = []
        repository_root_count = 0
        for entry in entries:
            try:
                resolves_to_repository = (
                    Path(entry).resolve() == resolved_repository_root
                )
            except OSError:
                resolves_to_repository = False
            if resolves_to_repository:
                if not _exact_string_equal(entry, repository_root):
                    _fail(f"{context} uses an aliased repository path")
                repository_root_count += 1
                if repository_root_count > 1:
                    continue
            normalized_entries.append(entry)
        if repository_root_count == 0:
            _fail(f"{context} lacks the exact repository path")
        return normalized_entries, repository_root_count

    normalized, repository_root_count = normalize_repository_roots(
        path_entries, "evaluator guard test host"
    )

    original_shadows = _loader_shadow_paths(
        ENTRYPOINT_MODULE, REPO_ROOT, path_entries
    )
    duplicate_parent_issue = "repository-parent path is missing or aliased"
    for issue in original_shadows:
        if not _exact_string_equal(issue, duplicate_parent_issue):
            _fail(f"evaluator guard test host has an import shadow: {issue}")
    if repository_root_count == 1 and original_shadows:
        _fail("evaluator guard test host has unexpected import shadow state")

    sys_namespace = object.__getattribute__(sys, "__dict__")
    original_sys_path = dict.__getitem__(sys_namespace, "path")
    normalized_parent_path = tuple(normalized)
    if not dict.__contains__(modules, "eval"):
        try:
            dict.__setitem__(sys_namespace, "path", normalized)
            shadows = _loader_shadow_paths(
                ENTRYPOINT_MODULE, REPO_ROOT, normalized_parent_path
            )
            if shadows:
                _fail(f"normalized evaluator import shadows remain: {shadows}")
            yield
        finally:
            dict.__setitem__(sys_namespace, "path", original_sys_path)
        return

    module = dict.__getitem__(modules, "eval")
    original_violation = _loaded_namespace_ancestor_violation(
        "eval", module, REPO_ROOT / "eval", path_entries
    )
    duplicate_portions_issue = "NamespacePath has missing or alternate portions"
    stale_parent_issue = "NamespacePath parent snapshot is stale"
    if original_violation is not None and not (
        _exact_string_equal(original_violation, duplicate_portions_issue)
        or _exact_string_equal(original_violation, stale_parent_issue)
    ):
        _fail(f"collected eval namespace is hostile: {original_violation}")

    namespace = object.__getattribute__(module, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    original_portions = dict.__getitem__(path_state, "_path")
    original_parent_path = dict.__getitem__(path_state, "_last_parent_path")
    original_epoch = dict.__getitem__(path_state, "_last_epoch")
    expected_portion = str(REPO_ROOT / "eval")
    portions = _exact_string_list(original_portions, "eval namespace portions")
    if not portions:
        _fail("collected eval namespace has no canonical portion")
    for portion in portions:
        if not _exact_string_equal(portion, expected_portion):
            _fail("collected eval namespace has an alternate portion")
    parent_snapshot = _exact_string_tuple(
        original_parent_path, "eval NamespacePath parent snapshot"
    )
    normalized_snapshot, snapshot_root_count = normalize_repository_roots(
        parent_snapshot, "collected eval NamespacePath parent snapshot"
    )
    if len(portions) != snapshot_root_count:
        _fail("collected eval namespace portions do not match its parent snapshot")

    parent_matches_current = len(parent_snapshot) == len(path_entries)
    if parent_matches_current:
        for index, entry in enumerate(path_entries):
            if not _exact_string_equal(parent_snapshot[index], entry):
                parent_matches_current = False
                break
    if len(portions) != 1:
        if not _exact_string_equal(original_violation, duplicate_portions_issue):
            _fail("collected eval namespace violation is not exact duplication")
    elif not parent_matches_current:
        if not _exact_string_equal(original_violation, stale_parent_issue):
            _fail("collected eval namespace parent drift is not exact staleness")
    elif original_violation is not None:
        _fail(f"collected eval namespace is unexpectedly hostile: {original_violation}")

    # Pytest prepends these exact collection/package directories and may reinsert
    # the repository root after a namespace is materialized. Keep this closed:
    # collection topology changes must fail until the test boundary is reviewed.
    approved_collection_insertions = frozenset(
        {
            str(REPO_ROOT / "arch_profiles" / "tests"),
            str(REPO_ROOT / "eval" / "guards"),
            str(REPO_ROOT / "tools"),
            str(REPO_ROOT / "tools" / "cache_predictor"),
            str(REPO_ROOT / "tools" / "cache_replay"),
            str(REPO_ROOT / "tools" / "cloud"),
            str(REPO_ROOT / "tools" / "hrd_profiler"),
            str(REPO_ROOT / "tools" / "local_labels"),
            str(REPO_ROOT / "tools" / "nvbit_tracer"),
            str(REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tests"),
            str(REPO_ROOT / "tools" / "pchase"),
            str(REPO_ROOT / "tools" / "sass_control"),
            str(REPO_ROOT / "tools" / "sector_stream"),
            str(REPO_ROOT / "tools" / "sol_baselines_scaffold"),
            str(REPO_ROOT / "tools" / "trace_sampling"),
            str(REPO_ROOT / "tools" / "trace_sampling" / "tests"),
            str(REPO_ROOT / "workloads" / "simbench_bw"),
            str(REPO_ROOT / "workloads" / "simbench_bw" / "tests"),
        }
    )
    if repository_root_count > 1 or snapshot_root_count > 1:
        current_comparison = [
            entry
            for entry in normalized
            if not _exact_string_equal(entry, repository_root)
        ]
        snapshot_comparison = [
            entry
            for entry in normalized_snapshot
            if not _exact_string_equal(entry, repository_root)
        ]
    else:
        current_comparison = normalized
        snapshot_comparison = normalized_snapshot
    snapshot_index = 0
    for entry in current_comparison:
        if snapshot_index < len(snapshot_comparison) and _exact_string_equal(
            entry, snapshot_comparison[snapshot_index]
        ):
            snapshot_index += 1
        elif entry not in approved_collection_insertions:
            _fail("collected eval parent changed outside approved collection paths")
    if snapshot_index != len(snapshot_comparison):
        _fail("collected eval parent removed or reordered a prior path")
    current_epoch = type.__getattribute__(_NAMESPACE_PATH_TYPE, "_epoch")
    if (
        type(original_epoch) is not int
        or type(current_epoch) is not int
        or original_epoch != current_epoch
    ):
        _fail("collected eval NamespacePath cache epoch is stale")
    path_finder = dict.__getitem__(path_state, "_path_finder")
    if (
        type(path_finder) is not MethodType
        or object.__getattribute__(path_finder, "__self__") is not _PATH_FINDER_TYPE
        or object.__getattribute__(path_finder, "__func__")
        is not _PATH_FINDER_GET_SPEC
    ):
        _fail("collected eval NamespacePath finder binding is hostile")

    try:
        dict.__setitem__(sys_namespace, "path", normalized)
        dict.__setitem__(path_state, "_path", [expected_portion])
        dict.__setitem__(path_state, "_last_parent_path", normalized_parent_path)
        dict.__setitem__(path_state, "_last_epoch", current_epoch)
        violation = _loaded_namespace_ancestor_violation(
            "eval", module, REPO_ROOT / "eval", normalized_parent_path
        )
        if violation is not None:
            _fail(f"normalized eval namespace remains hostile: {violation}")
        shadows = _loader_shadow_paths(
            ENTRYPOINT_MODULE, REPO_ROOT, normalized_parent_path
        )
        if shadows:
            _fail(f"normalized evaluator import shadows remain: {shadows}")
        yield
    finally:
        dict.__setitem__(path_state, "_path", original_portions)
        dict.__setitem__(path_state, "_last_parent_path", original_parent_path)
        dict.__setitem__(path_state, "_last_epoch", original_epoch)
        dict.__setitem__(sys_namespace, "path", original_sys_path)


def _materialized_eval_parent() -> tuple[ModuleType, tuple[str, ...]]:
    importlib.import_module("eval")
    parent_path, modules = _safe_system_import_state()
    module = dict.__getitem__(modules, "eval")
    assert type(module) is ModuleType
    return module, parent_path


def test_loaded_exact_eval_namespace_is_admitted_before_snapshot_load() -> None:
    ancestor, parent_path = _materialized_eval_parent()
    violation = _loaded_namespace_ancestor_violation(
        "eval", ancestor, REPO_ROOT / "eval", parent_path
    )
    if violation is None:
        strict_namespace = load_blocked_evaluator_snapshot()
        with pytest.raises(strict_namespace["EvaluatorUnavailable"]):
            strict_namespace["evaluate"]()
    else:
        assert "stale" in violation
    namespace = load_blocked_evaluator_snapshot_isolated()
    with pytest.raises(namespace["EvaluatorUnavailable"]):
        namespace["evaluate"]()


def test_namespace_validation_rejects_custom_locations_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    spec = dict.__getitem__(namespace, "__spec__")
    spec_state = object.__getattribute__(spec, "__dict__")
    calls = {"iter": 0, "eq": 0}

    class HostileLocations:
        def __iter__(self) -> Any:
            calls["iter"] += 1
            raise AssertionError("custom locations iterated")

        def __eq__(self, other: Any) -> bool:
            calls["eq"] += 1
            raise AssertionError("custom locations compared")

    monkeypatch.setitem(
        spec_state, "submodule_search_locations", HostileLocations()
    )
    with pytest.raises(RegistryViolation, match="exact materialized NamespacePath"):
        load_blocked_evaluator_snapshot()
    assert calls == {"iter": 0, "eq": 0}


def test_namespace_validation_rejects_custom_internal_portions_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    calls = {"iter": 0, "len": 0, "getitem": 0}

    class HostilePortions(list):
        def __iter__(self) -> Any:
            calls["iter"] += 1
            raise AssertionError("custom portions iterated")

        def __len__(self) -> int:
            calls["len"] += 1
            raise AssertionError("custom portions sized")

        def __getitem__(self, index: Any) -> Any:
            calls["getitem"] += 1
            raise AssertionError("custom portions indexed")

    monkeypatch.setitem(path_state, "_path", HostilePortions())
    with pytest.raises(RegistryViolation, match="missing or alternate portions"):
        load_blocked_evaluator_snapshot()
    assert calls == {"iter": 0, "len": 0, "getitem": 0}


def test_namespace_validation_rejects_string_subclass_without_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    calls = {"eq": 0}

    class HostileString(str):
        def __eq__(self, other: Any) -> bool:
            calls["eq"] += 1
            raise AssertionError("NamespacePath string compared")

        __hash__ = str.__hash__

    monkeypatch.setitem(
        path_state, "_path", [HostileString(str(REPO_ROOT / "eval"))]
    )
    with pytest.raises(
        RegistryViolation, match="descriptor-validated path"
    ):
        load_blocked_evaluator_snapshot()
    assert calls == {"eq": 0}


def test_import_state_rejects_sys_path_subclass_without_callbacks() -> None:
    sys_namespace = object.__getattribute__(sys, "__dict__")
    original = dict.__getitem__(sys_namespace, "path")
    calls = {"iter": 0, "len": 0, "getitem": 0}

    class HostilePathList(list):
        def __iter__(self) -> Any:
            calls["iter"] += 1
            raise AssertionError("hostile sys.path iterated")

        def __len__(self) -> int:
            calls["len"] += 1
            raise AssertionError("hostile sys.path sized")

        def __getitem__(self, index: Any) -> Any:
            calls["getitem"] += 1
            raise AssertionError("hostile sys.path indexed")

    hostile = HostilePathList(list(original))
    dict.__setitem__(sys_namespace, "path", hostile)
    try:
        with pytest.raises(RegistryViolation, match="sys.path must be an exact"):
            _safe_system_import_state()
    finally:
        dict.__setitem__(sys_namespace, "path", original)
    assert calls == {"iter": 0, "len": 0, "getitem": 0}


def test_import_state_rejects_string_subclass_without_callbacks() -> None:
    sys_namespace = object.__getattribute__(sys, "__dict__")
    original = dict.__getitem__(sys_namespace, "path")
    calls = {"eq": 0, "str": 0, "fspath": 0}

    class HostileString(str):
        def __eq__(self, other: Any) -> bool:
            calls["eq"] += 1
            raise AssertionError("hostile string compared")

        def __str__(self) -> str:
            calls["str"] += 1
            raise AssertionError("hostile string rendered")

        def __fspath__(self) -> str:
            calls["fspath"] += 1
            raise AssertionError("hostile string used as a path")

        __hash__ = str.__hash__

    hostile_path = [HostileString("/planted"), *list(original)]
    dict.__setitem__(sys_namespace, "path", hostile_path)
    try:
        with pytest.raises(RegistryViolation, match="non-exact string entry"):
            _safe_system_import_state()
    finally:
        dict.__setitem__(sys_namespace, "path", original)
    assert calls == {"eq": 0, "str": 0, "fspath": 0}


def test_namespace_validation_rejects_stale_recalculating_state_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    calls = {"finder": 0}

    class HostilePathFinder:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            calls["finder"] += 1
            raise AssertionError("stale namespace recalculated")

    monkeypatch.setitem(path_state, "_last_parent_path", ("/stale",))
    monkeypatch.setitem(path_state, "_path_finder", HostilePathFinder())
    with pytest.raises(RegistryViolation, match="parent snapshot is stale"):
        load_blocked_evaluator_snapshot()
    assert calls == {"finder": 0}


def test_isolated_loader_detaches_stale_parent_without_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    calls = {"finder": 0}

    class HostilePathFinder:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            calls["finder"] += 1
            raise AssertionError("isolated loader recalculated parent")

    monkeypatch.setitem(path_state, "_last_parent_path", ("/stale",))
    monkeypatch.setitem(path_state, "_path_finder", HostilePathFinder())
    loaded = load_blocked_evaluator_snapshot_isolated()
    with pytest.raises(loaded["EvaluatorUnavailable"]):
        loaded["evaluate"]()
    assert calls == {"finder": 0}
    modules = object.__getattribute__(sys, "__dict__")["modules"]
    assert dict.__getitem__(modules, "eval") is ancestor


def test_isolated_loader_handles_legitimate_combined_order_path_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    original = object.__getattribute__(sys, "__dict__")["path"]
    monkeypatch.setattr(
        sys, "path", [str(REPO_ROOT / "eval/guards"), *list(original)]
    )
    with pytest.raises(RegistryViolation, match="parent snapshot is stale"):
        load_blocked_evaluator_snapshot()
    loaded = load_blocked_evaluator_snapshot_isolated()
    with pytest.raises(loaded["EvaluatorUnavailable"]):
        loaded["evaluate"]()
    modules = object.__getattribute__(sys, "__dict__")["modules"]
    assert dict.__getitem__(modules, "eval") is ancestor


def test_namespace_validation_rejects_stale_epoch_without_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    current_epoch = type.__getattribute__(_NAMESPACE_PATH_TYPE, "_epoch")
    monkeypatch.setitem(path_state, "_last_parent_path", parent_path)
    monkeypatch.setitem(path_state, "_last_epoch", current_epoch - 1)
    with pytest.raises(RegistryViolation, match="cache epoch is stale"):
        load_blocked_evaluator_snapshot()


def test_import_state_rejects_hostile_path_hook_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"hook": 0}

    class HostileHook:
        def __call__(self, path: Any) -> Any:
            calls["hook"] += 1
            raise AssertionError("hostile path hook called")

    original = object.__getattribute__(sys, "__dict__")["path_hooks"]
    monkeypatch.setattr(sys, "path_hooks", [*list(original), HostileHook()])
    with pytest.raises(RegistryViolation, match="sys.path_hooks changed"):
        _safe_system_import_state()
    assert calls == {"hook": 0}


def test_import_state_rejects_hostile_cached_finder_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"find_spec": 0, "eq": 0}

    class HostileFinder:
        def find_spec(self, *args: Any, **kwargs: Any) -> Any:
            calls["find_spec"] += 1
            raise AssertionError("hostile cached finder called")

        def __eq__(self, other: Any) -> bool:
            calls["eq"] += 1
            raise AssertionError("hostile cached finder compared")

    original = object.__getattribute__(sys, "__dict__")["path_importer_cache"]
    hostile_cache = dict(original)
    dict.__setitem__(hostile_cache, str(REPO_ROOT), HostileFinder())
    monkeypatch.setattr(sys, "path_importer_cache", hostile_cache)
    with pytest.raises(RegistryViolation, match="untrusted finder"):
        _safe_system_import_state()
    assert calls == {"find_spec": 0, "eq": 0}


def test_finder_form_namespace_ancestor_with_no_loader_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = REPO_ROOT / "eval"
    spec = importlib.machinery.ModuleSpec("eval", None, is_package=True)
    spec.submodule_search_locations = [str(expected)]
    module = ModuleType("eval")
    module.__spec__ = spec
    module.__loader__ = None
    module.__package__ = "eval"
    module.__file__ = None
    module.__path__ = spec.submodule_search_locations
    monkeypatch.setitem(sys.modules, "eval", module)
    with pytest.raises(
        RegistryViolation, match="exact materialized NamespaceLoader"
    ):
        load_blocked_evaluator_snapshot()


def test_retained_entrypoint_bytes_refuse_silently_without_loader(
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = load_blocked_evaluator_snapshot_isolated()
    with pytest.raises(namespace["EvaluatorUnavailable"], match="typed-unresolved"):
        namespace["evaluate"]()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_entrypoint_source_rejects_ordinary_package_import_context() -> None:
    validated = _validate_registry()
    snapshot = validated.snapshots[ENTRYPOINT]
    tree = _parse_snapshot(snapshot, "ordinary-import refusal fixture")
    code = compile(
        tree,
        snapshot.relative_path,
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    namespace = {
        "__builtins__": {
            "RuntimeError": RuntimeError,
            "__import__": builtins.__import__,
        },
        "__name__": ENTRYPOINT_MODULE,
        "__package__": "eval",
    }
    with pytest.raises(RuntimeError, match="ordinary package import is forbidden"):
        exec(code, namespace, namespace)
    assert "evaluate" not in namespace


def test_source_file_loader_cannot_spoof_snapshot_execution_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spoofed_name = "_gpu_simulator_blocked_evaluator_snapshot_v1"
    spec = importlib.util.spec_from_file_location(
        spoofed_name, REPO_ROOT / ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError, match="ordinary package import is forbidden"):
        spec.loader.exec_module(module)
    assert not hasattr(module, "evaluate")


@pytest.mark.parametrize(
    "preloaded_name", [ENTRYPOINT_MODULE, f"{ENTRYPOINT_MODULE}.planted"]
)
def test_exact_snapshot_execution_rejects_sys_modules_shadow(
    monkeypatch: pytest.MonkeyPatch, preloaded_name: str
) -> None:
    monkeypatch.setitem(sys.modules, preloaded_name, ModuleType(preloaded_name))
    with pytest.raises(RegistryViolation, match="sys.modules"):
        load_blocked_evaluator_snapshot()


def test_import_state_rejects_sys_modules_subclass_without_callbacks() -> None:
    sys_namespace = object.__getattribute__(sys, "__dict__")
    original = dict.__getitem__(sys_namespace, "modules")
    calls = {"items": 0, "iter": 0, "keys": 0, "get": 0}

    class HidingModules(dict):
        def items(self) -> Any:
            calls["items"] += 1
            return ()

        def __iter__(self) -> Any:
            calls["iter"] += 1
            return iter(())

        def keys(self) -> Any:
            calls["keys"] += 1
            return ()

        def get(self, key: Any, default: Any = None) -> Any:
            calls["get"] += 1
            return default

    hostile = HidingModules(original)
    dict.__setitem__(
        hostile, ENTRYPOINT_MODULE, ModuleType(ENTRYPOINT_MODULE)
    )
    dict.__setitem__(sys_namespace, "modules", hostile)
    caught: RegistryViolation | None = None
    try:
        try:
            load_blocked_evaluator_snapshot()
        except RegistryViolation as exc:
            caught = exc
    finally:
        dict.__setitem__(sys_namespace, "modules", original)
    assert caught is not None
    assert "sys.modules must be an exact built-in dict" in str(caught)
    assert calls == {"items": 0, "iter": 0, "keys": 0, "get": 0}


def test_parent_child_attribute_is_rejected_without_payload_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    calls = {"payload": 0}

    class HostilePayload:
        def __getattribute__(self, name: str) -> Any:
            calls["payload"] += 1
            raise AssertionError("planted parent child payload accessed")

    monkeypatch.setitem(namespace, "production_evaluator", HostilePayload())
    with pytest.raises(RegistryViolation, match="forbidden direct attribute"):
        load_blocked_evaluator_snapshot()
    assert calls == {"payload": 0}


def test_parent_child_module_is_rejected_without_sys_modules_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    modules = object.__getattribute__(sys, "__dict__")["modules"]
    assert not dict.__contains__(modules, ENTRYPOINT_MODULE)
    child = ModuleType(ENTRYPOINT_MODULE)
    monkeypatch.setitem(namespace, "production_evaluator", child)
    with pytest.raises(RegistryViolation, match="forbidden direct attribute"):
        load_blocked_evaluator_snapshot()
    assert not dict.__contains__(modules, ENTRYPOINT_MODULE)


@pytest.mark.parametrize("hook_name", ["__getattr__", "__getattribute__"])
def test_parent_module_lookup_hook_is_rejected_without_callback(
    monkeypatch: pytest.MonkeyPatch, hook_name: str
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    calls = {"lookup": 0}

    def hostile_lookup(name: str) -> Any:
        calls["lookup"] += 1
        raise AssertionError("module lookup hook called")

    monkeypatch.setitem(namespace, hook_name, hostile_lookup)
    with pytest.raises(RegistryViolation, match="forbidden direct attribute"):
        load_blocked_evaluator_snapshot()
    assert calls == {"lookup": 0}


def test_sys_path_repository_alias_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object.__getattribute__(sys, "__dict__")["path"]
    alias = f"{REPO_ROOT}/."
    monkeypatch.setattr(sys, "path", [alias, *list(original)])
    with pytest.raises(RegistryViolation, match="sys.path.*aliased"):
        load_blocked_evaluator_snapshot()


def test_ordinary_from_parent_reaches_only_unconditional_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor, _parent_path = _materialized_eval_parent()
    namespace = object.__getattribute__(ancestor, "__dict__")
    modules = object.__getattribute__(sys, "__dict__")["modules"]
    assert not dict.__contains__(namespace, "production_evaluator")
    assert not dict.__contains__(modules, ENTRYPOINT_MODULE)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    statement = _TRUSTED_COMPILE(
        "from eval import production_evaluator",
        "<ordinary-evaluator-import>",
        "exec",
        dont_inherit=True,
    )
    ordinary_namespace: dict[str, Any] = {"__builtins__": builtins.__dict__}
    with pytest.raises(RuntimeError, match="ordinary package import is forbidden"):
        _TRUSTED_EXEC(statement, ordinary_namespace, ordinary_namespace)
    assert not dict.__contains__(namespace, "production_evaluator")
    assert not dict.__contains__(modules, ENTRYPOINT_MODULE)


def test_exact_snapshot_execution_rejects_executable_ancestor_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "eval.py"
    source.write_text("PLANTED = True\n", encoding="utf-8")
    loader = importlib.machinery.SourceFileLoader("eval", str(source))
    module = ModuleType("eval")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "eval", loader, origin=str(source)
    )
    module.__loader__ = loader
    module.__package__ = "eval"
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, "eval", module)
    with pytest.raises(RegistryViolation, match="sys.modules.*eval.*loader"):
        load_blocked_evaluator_snapshot()


def test_exact_snapshot_execution_rejects_alternate_namespace_portion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importlib.import_module("eval")
    alternate = tmp_path / "eval"
    alternate.mkdir()
    modules = object.__getattribute__(sys, "__dict__")["modules"]
    module = dict.__getitem__(modules, "eval")
    namespace = object.__getattribute__(module, "__dict__")
    namespace_path = dict.__getitem__(namespace, "__path__")
    path_state = object.__getattribute__(namespace_path, "__dict__")
    monkeypatch.setitem(
        path_state, "_path", [str(REPO_ROOT / "eval"), str(alternate)]
    )
    with pytest.raises(
        RegistryViolation, match="sys.modules.*eval.*alternate portions"
    ):
        load_blocked_evaluator_snapshot()


def test_exact_snapshot_execution_rejects_sys_path_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow = tmp_path / "eval/production_evaluator"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("PLANTED = True\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    with pytest.raises(RegistryViolation, match="sys.path"):
        load_blocked_evaluator_snapshot()


def test_exact_snapshot_execution_does_not_consult_meta_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileFinder:
        calls = 0

        @classmethod
        def find_spec(cls, fullname: str, path: Any = None, target: Any = None) -> None:
            cls.calls += 1
            raise AssertionError(fullname)

    monkeypatch.setattr(sys, "meta_path", [HostileFinder(), *sys.meta_path])
    namespace = load_blocked_evaluator_snapshot_isolated()
    assert HostileFinder.calls == 0
    with pytest.raises(namespace["EvaluatorUnavailable"]):
        namespace["evaluate"]()


def test_sim_sources_have_no_direct_or_recognized_python_evaluator_path() -> None:
    violations = _sim_evaluator_references(REPO_ROOT / "sim", ENTRYPOINT_MODULE)
    assert not violations, (
        "sim/ direct references and recognized Python dynamic/process APIs "
        "must not reach evaluator code:\n" + "\n".join(violations)
    )


HOSTILE_BLOCKED_SOURCES = [
    b"from math import *\n",
    b"from builtins import __import__ as load\nload('fixture')\n",
    b"import sys\nsys.modules['fixture'] = object()\n",
    b"import sys\nsys.meta_path.insert(0, object())\n",
    b"import ctypes\nctypes.CDLL('fixture.so')\n",
    b"import posix\nposix.system('fixture')\n",
    (
        b"import types\n"
        b"code = compile('VALUE = 1', '<fixture>', 'exec')\n"
        b"types.FunctionType(code, globals())()\n"
    ),
    b"import os\nos.__dict__['system']('fixture')\n",
    b"open('data/local_labels/planted.json').read()\n",
    (
        b"name = 'eval.' + 'production_evaluator'\n"
        b"__import__(name)\n"
    ),
]


@pytest.mark.parametrize("source", HOSTILE_BLOCKED_SOURCES)
def test_blocked_refusal_shape_rejects_hostile_constructs(
    tmp_path: Path, source: bytes
) -> None:
    document, snapshots = _write_graph_fixture(tmp_path, source)
    with pytest.raises(RegistryViolation, match="refusal-shape"):
        _build_import_graph(document, tmp_path, snapshots)


def test_python_source_rejects_utf7_cookie_with_hidden_target_read(
    tmp_path: Path,
) -> None:
    live_entrypoint = _snapshot_file(REPO_ROOT, ENTRYPOINT, "live fixture source")
    hidden = (
        b"# coding: utf-7\n"
        b"# +AAo-open('data/local_labels/planted.json').read()\n"
    )
    source = hidden + live_entrypoint.data
    assert "\nopen('data/local_labels/planted.json').read()" in source.decode(
        "utf-7"
    )
    document, snapshots = _write_graph_fixture(tmp_path, source)
    with pytest.raises(
        RegistryViolation, match="non-UTF-8 PEP 263 coding cookie: 'utf-7'"
    ):
        _build_import_graph(document, tmp_path, snapshots)


def test_unreviewed_repo_local_import_defaults_to_deny(tmp_path: Path) -> None:
    document, snapshots = _write_graph_fixture(
        tmp_path, b"from . import helper\n"
    )
    helper = tmp_path / "eval/production_evaluator/helper.py"
    helper.write_text("VALUE = object()\n", encoding="utf-8")
    with pytest.raises(RegistryViolation, match="defaults to deny"):
        _build_import_graph(document, tmp_path, snapshots)


def test_registered_denied_import_remains_denied(tmp_path: Path) -> None:
    document, snapshots = _write_graph_fixture(
        tmp_path, b"from . import helper\n"
    )
    helper = tmp_path / "eval/production_evaluator/helper.py"
    helper.write_text("VALUE = object()\n", encoding="utf-8")
    helper_snapshot = _snapshot_file(
        tmp_path, "eval/production_evaluator/helper.py", "fixture"
    )
    document["classifications"].append(
        {
            "classification": "unreviewed",
            "import_policy": "deny",
            "path": "eval/production_evaluator/helper.py",
            "review_status": "provisional",
            "sha256": helper_snapshot.sha256,
        }
    )
    snapshots[helper_snapshot.relative_path] = helper_snapshot
    with pytest.raises(RegistryViolation, match="denied classification"):
        _build_import_graph(document, tmp_path, snapshots)


def test_unreachable_allowed_registry_row_is_stale(tmp_path: Path) -> None:
    live_entrypoint = _snapshot_file(REPO_ROOT, ENTRYPOINT, "live fixture source")
    document, snapshots = _write_graph_fixture(tmp_path, live_entrypoint.data)
    helper = tmp_path / "eval/production_evaluator/helper.py"
    helper.write_text("VALUE = object()\n", encoding="utf-8")
    helper_snapshot = _snapshot_file(
        tmp_path, "eval/production_evaluator/helper.py", "fixture"
    )
    document["classifications"].append(
        {
            "classification": "production_dependency",
            "import_policy": "allow",
            "path": "eval/production_evaluator/helper.py",
            "review_status": "reviewed",
            "sha256": helper_snapshot.sha256,
        }
    )
    snapshots[helper_snapshot.relative_path] = helper_snapshot
    with pytest.raises(RegistryViolation, match="stale or unreachable"):
        _build_import_graph(document, tmp_path, snapshots)


def test_entrypoint_ancestor_inventory_rejects_planted_eval_initializer(
    tmp_path: Path,
) -> None:
    package = tmp_path / "eval/production_evaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# blocked package\n", encoding="utf-8")
    (tmp_path / "eval/__init__.py").write_text(
        "TARGET_READ = True\n", encoding="utf-8"
    )
    with pytest.raises(
        RegistryViolation,
        match=r"ancestor 'eval' initializer candidate is forbidden:.*eval/__init__\.py",
    ):
        _verify_entrypoint_ancestor_inventory(tmp_path)


@pytest.mark.parametrize(
    "artifact",
    ["eval.py", "eval.pyc", "eval.so", "eval/__init__.pyc", "eval/__init__.so"],
)
def test_entrypoint_ancestor_inventory_rejects_alternate_candidates(
    tmp_path: Path, artifact: str
) -> None:
    package = tmp_path / "eval/production_evaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# blocked package\n", encoding="utf-8")
    (tmp_path / artifact).write_bytes(b"planted")
    with pytest.raises(RegistryViolation, match="ancestor.*candidate is forbidden"):
        _verify_entrypoint_ancestor_inventory(tmp_path)


@pytest.mark.parametrize(
    "candidate_kind",
    [
        "symlink_source",
        "bytecode",
        "pycache_bytecode",
        "native_extension",
        "compiled_package",
        "ambiguous",
    ],
)
def test_module_resolution_rejects_non_source_and_ambiguous_candidates(
    tmp_path: Path, candidate_kind: str
) -> None:
    package = tmp_path / "eval/production_evaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    if candidate_kind == "symlink_source":
        outside = tmp_path / "outside.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        (package / "helper.py").symlink_to(outside)
        match = "symlink"
    elif candidate_kind == "bytecode":
        (package / "helper.pyc").write_bytes(b"planted")
        match = "compiled module"
    elif candidate_kind == "pycache_bytecode":
        cache = package / "__pycache__"
        cache.mkdir()
        (cache / f"helper.{sys.implementation.cache_tag}.pyc").write_bytes(b"planted")
        match = "compiled module"
    elif candidate_kind == "native_extension":
        (package / "helper.so").write_bytes(b"planted")
        match = "compiled module"
    elif candidate_kind == "compiled_package":
        helper_package = package / "helper"
        helper_package.mkdir()
        (helper_package / "__init__.pyc").write_bytes(b"planted")
        match = "compiled package"
    elif candidate_kind == "ambiguous":
        (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        helper_package = package / "helper"
        helper_package.mkdir()
        (helper_package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
        match = "ambiguous module/package"
    else:  # pragma: no cover - closed parameter list.
        raise AssertionError(candidate_kind)
    with pytest.raises(RegistryViolation, match=match):
        _module_source_paths(tmp_path, "eval.production_evaluator.helper")


def test_module_resolution_rejects_compiled_package_chain(tmp_path: Path) -> None:
    package = tmp_path / "eval/production_evaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "eval.so").write_bytes(b"planted")
    with pytest.raises(RegistryViolation, match="compiled package-chain"):
        _module_source_paths(tmp_path, "eval.production_evaluator.helper")


def test_module_resolution_rejects_symlinked_package_chain(tmp_path: Path) -> None:
    real_eval = tmp_path / "real_eval"
    package = real_eval / "production_evaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "eval").symlink_to(real_eval, target_is_directory=True)
    with pytest.raises(RegistryViolation, match="symlink|non-directory"):
        _module_source_paths(tmp_path, "eval.production_evaluator.helper")


@pytest.mark.parametrize(
    "artifact",
    ["__pycache__", "__init__.pyc", "__init__.so", "helper.py", "helper_symlink"],
)
def test_blocked_package_rejects_compiled_extra_and_symlink_artifacts(
    tmp_path: Path, artifact: str
) -> None:
    package = tmp_path / "eval/production_evaluator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    if artifact == "__pycache__":
        (package / artifact).mkdir()
    elif artifact == "helper_symlink":
        outside = tmp_path / "outside.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        (package / "helper.py").symlink_to(outside)
    else:
        (package / artifact).write_bytes(b"planted")
    with pytest.raises(RegistryViolation, match="compiled|inventory|symlink"):
        _verify_blocked_package_inventory(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "hash_drift"])
def test_registered_source_snapshots_reject_staleness_and_hash_drift(
    tmp_path: Path, mutation: str
) -> None:
    live_entrypoint = _snapshot_file(REPO_ROOT, ENTRYPOINT, "live fixture source")
    document, _snapshots = _write_graph_fixture(tmp_path, live_entrypoint.data)
    entrypoint = tmp_path / ENTRYPOINT
    if mutation == "missing":
        entrypoint.unlink()
        with pytest.raises(RegistryViolation, match="row is stale"):
            _snapshot_file(tmp_path, ENTRYPOINT, "classification")
    else:
        entrypoint.write_text("VALUE = 'drifted'\n", encoding="utf-8")
        snapshot = _snapshot_file(tmp_path, ENTRYPOINT, "classification")
        row = document["classifications"][0]
        with pytest.raises(RegistryViolation, match="source hash drifted"):
            if snapshot.sha256 != row["sha256"]:
                _fail("classification source hash drifted")


def test_authority_snapshot_rejects_fifo_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "authority-fifo"
    os.mkfifo(tmp_path / relative)
    read_calls = 0

    def forbidden_read(_descriptor: int, _size: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("FIFO authority must fail before reading")

    monkeypatch.setattr(os, "read", forbidden_read)
    with pytest.raises(RegistryViolation, match="not a regular file"):
        _snapshot_file(tmp_path, relative, "fixture authority")

    assert read_calls == 0


def test_authority_snapshot_rejects_oversize_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "oversized-authority.bin"
    authority = tmp_path / relative
    with authority.open("wb") as stream:
        stream.truncate(AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES + 1)
    read_calls = 0

    def forbidden_read(_descriptor: int, _size: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        raise MemoryError("oversized authority must fail before reading")

    monkeypatch.setattr(os, "read", forbidden_read)
    with pytest.raises(
        RegistryViolation,
        match="authority_snapshot_size_limit_exceeded",
    ) as failure:
        _snapshot_file(tmp_path, relative, "fixture authority")

    assert read_calls == 0
    assert relative in str(failure.value)
    assert str(AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES + 1) in str(failure.value)
    assert f"ceiling={AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES}" in str(
        failure.value
    )


def test_authority_snapshot_accepts_exact_byte_ceiling(tmp_path: Path) -> None:
    relative = "bounded-authority.bin"
    authority = tmp_path / relative
    with authority.open("wb") as stream:
        stream.truncate(AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES)

    snapshot = _snapshot_file(tmp_path, relative, "fixture authority")

    assert len(snapshot.data) == AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES
    assert snapshot.sha256 == hashlib.sha256(snapshot.data).hexdigest()


def test_schema_rejects_unknown_registry_fields() -> None:
    validated = _validate_registry()
    schema = _strict_json(validated.schema_snapshot, "classification registry schema")
    planted = copy.deepcopy(validated.document)
    planted["unreviewed_authority"] = True
    with pytest.raises(RegistryViolation, match="schema validation failed"):
        _validate_with_schema(planted, schema)


def test_contract_and_prototype_sources_remain_reviewed_denied() -> None:
    validated = _validate_registry()
    denied = EXPECTED_EVALUATOR_CONTRACTS | (
        EXPECTED_SYNTHETIC_OFFLINE_PROTOTYPE_EVALUATORS
    )

    assert {
        path
        for path, row in validated.rows.items()
        if row["classification"]
        in {"evaluator_contract", "synthetic_offline_prototype_evaluator"}
    } == denied
    assert all(
        validated.rows[path]["import_policy"] == "deny"
        and validated.rows[path]["review_status"] == "reviewed"
        for path in denied
    )


def test_classification_registry_rejects_container_subclasses_without_callbacks() -> None:
    calls = {"iter": 0, "get": 0, "eq": 0}

    class HostileRows(list):
        def __iter__(self) -> Any:
            calls["iter"] += 1
            raise AssertionError("classification rows iterated")

    class HostileRow(dict):
        def get(self, key: Any, default: Any = None) -> Any:
            calls["get"] += 1
            raise AssertionError("classification row queried")

    class HostilePath(str):
        def __eq__(self, other: Any) -> bool:
            calls["eq"] += 1
            raise AssertionError("classification path compared")

        __hash__ = str.__hash__

    with pytest.raises(RegistryViolation, match="exact built-in list"):
        _classification_by_path({"classifications": HostileRows()})
    with pytest.raises(RegistryViolation, match="exact built-in dicts"):
        _classification_by_path({"classifications": [HostileRow()]})
    with pytest.raises(RegistryViolation, match="exact built-in strings"):
        _classification_by_path(
            {"classifications": [{"path": HostilePath("eval/planted.py")}]}
        )
    assert calls == {"iter": 0, "get": 0, "eq": 0}


def test_strict_registry_json_rejects_nonfinite_number() -> None:
    data = b'{"value":NaN}\n'
    snapshot = FileSnapshot(
        "fixture.json",
        data,
        hashlib.sha256(data).hexdigest(),
        (0, 0, 0, 0, 0, 0, len(data), 0, 0),
    )

    with pytest.raises(RegistryViolation, match="non-finite JSON constant"):
        _strict_json(snapshot, "fixture registry")


def test_controlled_import_requires_exact_level_and_fromlist_types() -> None:
    namespace: dict[str, Any] = {}
    assert _controlled_future_import_request(
        "__future__", namespace, namespace, ("annotations",), 0
    )

    with pytest.raises(RegistryViolation, match="exact integer"):
        _controlled_future_import_request(
            "__future__", namespace, namespace, ("annotations",), False
        )

    class HostileFromlist(tuple):
        def __iter__(self) -> Any:
            raise AssertionError("hostile fromlist iterated")

    with pytest.raises(RegistryViolation, match="exact built-in tuple"):
        _controlled_future_import_request(
            "__future__",
            namespace,
            namespace,
            HostileFromlist(("annotations",)),
            0,
        )


def test_classification_registry_enforces_row_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "MAX_CLASSIFICATION_ROWS", 1)
    document = {
        "classifications": [
            {"path": "eval/a.py"},
            {"path": "eval/b.py"},
        ]
    }

    with pytest.raises(RegistryViolation, match="classification_inventory_file_limit"):
        _classification_by_path(document)


def test_exact_whole_file_prohibited_duplicate_is_rejected() -> None:
    validated = _validate_registry()
    planted = copy.deepcopy(validated.document)
    prohibited_sha256 = validated.rows[
        "workloads/simbench_bw/gcom_lite_cycles_evaluator.py"
    ]["sha256"]
    entrypoint = next(
        row for row in planted["classifications"] if row["path"] == ENTRYPOINT
    )
    entrypoint["sha256"] = prohibited_sha256
    with pytest.raises(RegistryViolation, match="exact whole-file duplicate"):
        _reject_exact_prohibited_duplicates(_classification_by_path(planted))


def test_sim_reverse_guard_catches_split_dynamic_import(tmp_path: Path) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "name = 'eval.' + 'production_evaluator'\n__import__(name)\n",
        encoding="utf-8",
    )
    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
    assert any("dynamic code/import API" in violation for violation in violations)


def test_sim_reverse_guard_rejects_computed_reflection_import(
    tmp_path: Path,
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "import builtins as b\n"
        "b.__dict__['__im' + 'port__']("
        "'eval.production_' + 'evaluator')\n",
        encoding="utf-8",
    )
    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
    assert any("dynamic code/import API '__import__'" in item for item in violations)
    assert any("computed evaluator" in item for item in violations)


@pytest.mark.parametrize(
    "expression",
    [
        "'sys{}tem'.format('')",
        "'%s%s' % ('sys', 'tem')",
        "f\"{'sys'}tem\"",
        "'sysXtem'.replace('X', '')",
        "'xsystem'[1:]",
    ],
    ids=["format", "percent", "f-string", "replace", "slice"],
)
def test_sim_reverse_guard_rejects_reflection_without_string_folding(
    tmp_path: Path, expression: str
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        f"import os\ngetattr(os, {expression})('true')\n",
        encoding="utf-8",
    )

    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)

    assert any(
        "dynamic code/import reference 'getattr'" in item for item in violations
    )


def test_sim_reverse_guard_rejects_join_dataflow_reflection_import(
    tmp_path: Path,
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "import builtins\n"
        "key = ''.join(['__im', 'port__'])\n"
        "name = ''.join(('eval.production_', 'evaluator'))\n"
        "builtins.__dict__[key](name)\n",
        encoding="utf-8",
    )
    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
    assert any("dynamic code/import API '__import__'" in item for item in violations)
    assert any("computed evaluator module" in item for item in violations)
    assert any("reflective subscript '__dict__'" in item for item in violations)


def test_sim_reverse_guard_rejects_computed_sys_modules_key(
    tmp_path: Path,
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "import sys\n"
        "name = ''.join(['eval.production_', 'evaluator'])\n"
        "sys.modules[name] = object()\n",
        encoding="utf-8",
    )
    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
    assert any(
        "reflective subscript 'modules' with key 'eval.production_evaluator'"
        in item
        for item in violations
    )


def test_sim_reverse_guard_rejects_utf7_hidden_computed_import(
    tmp_path: Path,
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    hidden = (
        b"# coding: utf-7\n"
        b"# +AAo-b.__dict__['__im'+-'port__']("
        b"'eval.production_'+-'evaluator')\n"
    )
    assert "\nb.__dict__['__im'+'port__']" in hidden.decode("utf-7")
    (sim_root / "fixture.py").write_bytes(hidden)
    with pytest.raises(
        RegistryViolation, match="non-UTF-8 PEP 263 coding cookie: 'utf-7'"
    ):
        _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)


def test_sim_reverse_guard_rejects_runtime_dynamic_import_input(tmp_path: Path) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "def load(module_name):\n    return __import__(module_name)\n",
        encoding="utf-8",
    )
    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
    assert any("dynamic code/import" in violation for violation in violations)


def test_sim_reverse_guard_rejects_aliased_builtin_import(tmp_path: Path) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "from builtins import __import__ as load\n"
        "def resolve(module_name):\n    return load(module_name)\n",
        encoding="utf-8",
    )
    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
    assert any("dynamic API import" in violation for violation in violations)


def test_sim_reverse_guard_rejects_callable_process_alias_chain(
    tmp_path: Path,
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(
        "import os\n"
        "runner = os.system\n"
        "invoke = runner\n"
        "def execute(command):\n    return invoke(command)\n",
        encoding="utf-8",
    )

    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)

    assert any(
        "process execution API alias 'invoke' -> 'system'" in violation
        for violation in violations
    )


@pytest.mark.parametrize(
    ("source", "alias_name"),
    [
        (
            "import os\n"
            "def execute(command, runner=os.system):\n"
            "    return runner(command)\n",
            "runner",
        ),
        (
            "import os\n"
            "runner = [os.system][0]\n"
            "def execute(command):\n    return runner(command)\n",
            "runner",
        ),
        (
            "import os\n"
            "runners = {'default': os.system}\n"
            "def execute(command):\n    return runners['default'](command)\n",
            "runners",
        ),
    ],
)
def test_sim_reverse_guard_rejects_process_defaults_and_containers(
    tmp_path: Path, source: str, alias_name: str
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(source, encoding="utf-8")

    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)

    assert any(
        f"process execution API alias {alias_name!r} -> 'system'" in violation
        for violation in violations
    )


@pytest.mark.parametrize(
    ("source", "alias_name"),
    [
        (
            "import os\n"
            "runner, = [os.system]\n"
            "def execute(command):\n    return runner(command)\n",
            "runner",
        ),
        (
            "import os\n"
            "runners = {}\n"
            "runners['default'] = os.system\n"
            "def execute(command):\n    return runners['default'](command)\n",
            "runners",
        ),
        (
            "import os\n"
            "class Holder:\n    pass\n"
            "holder = Holder()\n"
            "holder.runner = os.system\n"
            "def execute(command):\n    return holder.runner(command)\n",
            "runner",
        ),
    ],
    ids=["tuple-unpack", "mapping-mutation", "attribute-assignment"],
)
def test_sim_reverse_guard_rejects_process_assignment_targets(
    tmp_path: Path, source: str, alias_name: str
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(source, encoding="utf-8")

    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)

    assert any(
        f"process execution API alias {alias_name!r} -> 'system'" in violation
        for violation in violations
    )


@pytest.mark.parametrize(
    "source",
    [
        "import os\nfor runner in [os.system]:\n    runner('true')\n",
        "import os\n[runner('true') for runner in [os.system]]\n",
        "import os\nrunners = []\nrunners.append(os.system)\nrunners[0]('true')\n",
        "import os\nrunners = dict(default=os.system)\nrunners['default']('true')\n",
        "import os\nclass Holder:\n    pass\nholder = Holder()\n"
        "setattr(holder, 'runner', os.system)\nholder.runner('true')\n",
    ],
    ids=["for-loop", "comprehension", "append", "constructor", "setattr"],
)
def test_sim_reverse_guard_rejects_process_callable_transfers(
    tmp_path: Path, source: str
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.py").write_text(source, encoding="utf-8")

    violations = _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)

    assert any(
        "process execution reference 'system'" in violation
        for violation in violations
    )


def test_eval_runtime_inventory_is_directory_closed(tmp_path: Path) -> None:
    (tmp_path / "eval/contracts").mkdir(parents=True)
    (tmp_path / "eval/guards").mkdir()
    (tmp_path / "eval/tests").mkdir()
    (tmp_path / "eval/contracts/authority.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (tmp_path / "eval/prototype.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "eval/guards/ignored.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )
    (tmp_path / "eval/tests/ignored.py").write_text(
        "VALUE = 4\n", encoding="utf-8"
    )

    assert _eval_runtime_source_inventory(tmp_path).sources == frozenset(
        {
            "eval/contracts/authority.py",
            "eval/prototype.py",
        }
    )


def test_eval_runtime_inventory_rejects_compiled_candidate(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval/planted.pyc").write_bytes(b"compiled")

    with pytest.raises(RegistryViolation, match="compiled candidates"):
        _eval_runtime_source_inventory(tmp_path)


def test_eval_runtime_inventory_authenticates_bytecode_cache(
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "benign.py"
    source_text = "VALUE = 1\n"
    source_path.write_text(source_text, encoding="utf-8")
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"benign.{cache_tag}.pyc"

    source_snapshot = _snapshot_file(
        tmp_path, "eval/benign.py", "retained runtime source"
    )
    benign_payload = _compiled_runtime_source_payload(
        tmp_path, "eval/benign.py", source_snapshot, 0
    )
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(header + benign_payload)
    inventory = _eval_runtime_source_inventory(tmp_path)
    cache_size = cache_path.stat().st_size

    assert _verify_runtime_bytecode_caches(
        tmp_path, inventory, {"eval/benign.py": source_snapshot}
    ) == cache_size

    expected_code = marshal.loads(benign_payload)
    assert isinstance(expected_code, CodeType)
    round_tripped_code = marshal.loads(marshal.dumps(expected_code))
    assert _runtime_code_matches_source(round_tripped_code, expected_code)
    wrong_filename_code = _TRUSTED_COMPILE(
        source_text,
        str(tmp_path / "elsewhere.py"),
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    assert not _runtime_code_matches_source(
        wrong_filename_code,
        expected_code,
    )

    malicious_code = _TRUSTED_COMPILE(
        "PLANTED = True\n",
        str(source_path),
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    unchecked_header = (
        importlib.util.MAGIC_NUMBER + (1).to_bytes(4, "little") + b"\0" * 8
    )
    cache_path.write_bytes(unchecked_header + marshal.dumps(malicious_code))

    with pytest.raises(RegistryViolation, match="differs from its retained source"):
        _verify_runtime_bytecode_caches(
            tmp_path, inventory, {"eval/benign.py": source_snapshot}
        )


def test_runtime_cache_rejects_hidden_localsplus_alias_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "alias.py"
    source_path.write_text("pass\n", encoding="utf-8")
    source_snapshot = _snapshot_file(
        tmp_path, "eval/alias.py", "retained runtime source"
    )

    root = _TRUSTED_COMPILE(
        "def f(x, y):\n    return x, y\n",
        "<runtime-hidden-alias>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    function = next(
        value for value in root.co_consts if isinstance(value, CodeType)
    )
    expected = function.replace(co_code=bytes(bytearray(b"  ")))
    expected_payload = marshal.dumps(expected)
    altered_payload = bytearray(expected_payload)
    direct_kinds = b"s\x02\x00\x00\x00  "
    assert altered_payload.count(direct_kinds) == 1
    kinds_offset = altered_payload.index(direct_kinds)
    altered_payload[kinds_offset : kinds_offset + len(direct_kinds)] = (
        b"r" + (1).to_bytes(4, "little")
    )
    altered_payload_bytes = bytes(altered_payload)

    expected_projection = _runtime_marshaled_localsplus_payload_projections(
        expected_payload
    )
    altered_projection = _runtime_marshaled_localsplus_payload_projections(
        altered_payload_bytes
    )
    assert expected_projection is not None
    assert altered_projection is not None
    assert expected_projection != altered_projection
    altered_code = marshal.loads(altered_payload_bytes)
    assert isinstance(altered_code, CodeType)
    assert _runtime_code_matches_source(altered_code, expected)

    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"alias.{cache_tag}.pyc"
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(header + altered_payload_bytes)
    inventory = _eval_runtime_source_inventory(tmp_path)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_compiled_runtime_source_payload",
        lambda *_args, **_kwargs: expected_payload,
    )

    with pytest.raises(
        RegistryViolation,
        match="differs from its retained source",
    ):
        _verify_runtime_bytecode_caches(
            tmp_path,
            inventory,
            {"eval/alias.py": source_snapshot},
        )


def test_runtime_code_authentication_rejects_split_local_cell_slots(
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "closure.py"
    source_text = (
        "def outer(x):\n"
        "    def inner():\n"
        "        return x\n"
        "    return inner\n"
    )
    source_path.write_text(source_text, encoding="utf-8")
    source_snapshot = _snapshot_file(
        tmp_path, "eval/closure.py", "retained runtime source"
    )
    expected_payload = _compiled_runtime_source_payload(
        tmp_path,
        "eval/closure.py",
        source_snapshot,
        0,
    )
    expected_root = marshal.loads(expected_payload)
    assert isinstance(expected_root, CodeType)
    expected_outer = next(
        item
        for item in expected_root.co_consts
        if isinstance(item, CodeType) and item.co_name == "outer"
    )

    outer_payload = bytearray(marshal.dumps(expected_outer))
    kinds_marker = b"s" + (2).to_bytes(4, "little", signed=True) + b"\x60\x20"
    kinds_offset = outer_payload.index(kinds_marker)
    names_offset = kinds_offset - 12
    assert outer_payload[names_offset : names_offset + 2] == b")\x02"
    first_name_reference = bytes(
        outer_payload[names_offset + 2 : names_offset + 7]
    )
    assert first_name_reference[0] & 0x7F == ord("r")
    outer_payload[names_offset + 1] = 3
    outer_payload[kinds_offset:kinds_offset] = first_name_reference
    kinds_offset += len(first_name_reference)
    outer_payload[kinds_offset : kinds_offset + len(kinds_marker)] = (
        b"s"
        + (3).to_bytes(4, "little", signed=True)
        + b"\x20\x20\x40"
    )
    altered_outer = marshal.loads(outer_payload)
    assert isinstance(altered_outer, CodeType)
    assert altered_outer.co_code == expected_outer.co_code
    assert altered_outer.co_nlocals == expected_outer.co_nlocals
    assert altered_outer.co_varnames == expected_outer.co_varnames
    assert altered_outer.co_cellvars == expected_outer.co_cellvars
    assert altered_outer.co_freevars == expected_outer.co_freevars
    assert not _runtime_code_matches_source(altered_outer, expected_outer)

    altered_root = expected_root.replace(
        co_consts=tuple(
            altered_outer if item is expected_outer else item
            for item in expected_root.co_consts
        )
    )
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"closure.{cache_tag}.pyc"
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(header + marshal.dumps(altered_root))
    inventory = _eval_runtime_source_inventory(tmp_path)

    with pytest.raises(
        RegistryViolation,
        match="differs from its retained source",
    ):
        _verify_runtime_bytecode_caches(
            tmp_path,
            inventory,
            {"eval/closure.py": source_snapshot},
        )


def test_eval_runtime_inventory_accepts_coexisting_optimization_caches(
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "bench.py"
    source_path.write_bytes((REPO_ROOT / "eval/bench.py").read_bytes())
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_paths = [
        cache_root / f"bench.{cache_tag}.opt-{optimization}.pyc"
        for optimization in (1, 2)
    ]
    for optimization, cache_path in zip((1, 2), cache_paths, strict=True):
        py_compile.compile(
            str(source_path),
            cfile=str(cache_path),
            dfile=str(source_path),
            doraise=True,
            optimize=optimization,
        )

    source_snapshot = _snapshot_file(
        tmp_path, "eval/bench.py", "retained runtime source"
    )
    inventory = _eval_runtime_source_inventory(tmp_path)

    assert _verify_runtime_bytecode_caches(
        tmp_path,
        inventory,
        {"eval/bench.py": source_snapshot},
    ) == sum(path.stat().st_size for path in cache_paths)


def test_runtime_code_authentication_preserves_reference_topology() -> None:
    base = _TRUSTED_COMPILE(
        "pass\n",
        "<runtime-code-topology>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    expected_shared = tuple(["shared"])
    observed_left = tuple(["shared"])
    observed_right = tuple(["shared"])
    assert observed_left is not observed_right
    expected = base.replace(co_consts=(expected_shared, expected_shared))
    round_tripped = marshal.loads(marshal.dumps(expected))
    assert isinstance(round_tripped, CodeType)
    assert _runtime_code_matches_source(round_tripped, expected)
    observed = base.replace(co_consts=(observed_left, observed_right))

    assert not _runtime_code_matches_source(observed, expected)

    expected_left = tuple(["shared"])
    expected_right = tuple(["shared"])
    assert expected_left is not expected_right
    expected = base.replace(co_consts=(expected_left, expected_right))
    observed_shared = tuple(["shared"])
    observed = base.replace(co_consts=(observed_shared, observed_shared))

    assert not _runtime_code_matches_source(observed, expected)


def test_runtime_code_authentication_preserves_frozenset_iteration_layout() -> None:
    base = _TRUSTED_COMPILE(
        "pass\n",
        "<runtime-code-frozenset>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    expected_constant = frozenset((0, 8, 16, 24, 32, 40))
    observed_constant = frozenset((0, 16, 8, 24, 40, 32))
    assert expected_constant == observed_constant
    assert tuple(expected_constant) != tuple(observed_constant)
    expected = base.replace(co_consts=(expected_constant,))
    round_tripped = marshal.loads(marshal.dumps(expected))
    assert isinstance(round_tripped, CodeType)
    assert _runtime_code_matches_source(round_tripped, expected)
    observed = base.replace(co_consts=(observed_constant,))

    assert not _runtime_code_matches_source(observed, expected)


def test_runtime_code_authentication_accepts_exact_nan_frozensets() -> None:
    base = _TRUSTED_COMPILE(
        "pass\n",
        "<runtime-code-nan-frozenset>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    nan_values = tuple(
        struct.unpack("!d", bytes.fromhex(bits))[0]
        for bits in (
            "7ff8000000000001",
            "7ff8000000000002",
            "fff8000000000001",
            "7ff8000000000001",
        )
    )
    expected = base.replace(
        co_consts=(frozenset((0, 8, 16, *nan_values)),)
    )
    payload = marshal.dumps(expected)
    loaded = [marshal.loads(payload) for _ in range(64)]
    assert all(isinstance(item, CodeType) for item in loaded)

    def layout(code: CodeType) -> tuple[object, ...]:
        constant = code.co_consts[0]
        assert isinstance(constant, frozenset)
        return tuple(
            ("float", struct.pack("!d", item))
            if isinstance(item, float)
            else item
            for item in constant
        )

    pair = next(
        (
            (left, right)
            for left in loaded
            for right in loaded
            if layout(left) != layout(right)
        ),
        None,
    )
    assert pair is not None
    assert _runtime_code_matches_source(*pair)

    changed_nan = struct.unpack(
        "!d", bytes.fromhex("7ff8000000000003")
    )[0]
    changed = base.replace(
        co_consts=(frozenset((0, 8, 16, *nan_values[:-1], changed_nan)),)
    )
    assert not _runtime_code_matches_source(changed, pair[1])


@pytest.mark.parametrize(
    "source_text",
    (
        "X = y in {1e1000 - 1e1000, 0, 1, 2}\n",
        "X = y in {(1e1000 - 1e1000,), (1,), (2,), (3,)}\n",
    ),
)
def test_runtime_code_authentication_accepts_recursive_nan_hash_layouts(
    source_text: str,
) -> None:
    payload = marshal.dumps(
        _TRUSTED_COMPILE(
            source_text,
            "<runtime-code-recursive-nan-frozenset>",
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
    )
    loaded = [marshal.loads(payload) for _ in range(64)]
    assert all(isinstance(item, CodeType) for item in loaded)

    def layout(code: CodeType) -> tuple[str, ...]:
        constant = next(
            value for value in code.co_consts if isinstance(value, frozenset)
        )
        return tuple(repr(item) for item in constant)

    pair = next(
        (
            (left, right)
            for left in loaded
            for right in loaded
            if layout(left) != layout(right)
        ),
        None,
    )
    assert pair is not None
    assert _runtime_code_matches_source(*pair)


def test_runtime_code_authentication_accepts_lone_surrogate_constant() -> None:
    expected = _TRUSTED_COMPILE(
        "X = y in {'\\ud800', 'a'}\n",
        "<runtime-code-surrogate-frozenset>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    payload = marshal.dumps(expected)
    observed = marshal.loads(payload)
    expected = marshal.loads(payload)
    assert isinstance(observed, CodeType)
    assert isinstance(expected, CodeType)
    assert any(
        isinstance(value, frozenset) and "\ud800" in value
        for value in expected.co_consts
    )

    assert _runtime_code_matches_source(observed, expected)


def test_runtime_code_authentication_preserves_string_interning() -> None:
    base = _TRUSTED_COMPILE(
        "pass\n",
        "<runtime-code-string-interning>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    value = "runtime bytecode interning fixture value"
    noninterned = (" " + value)[1:]
    interned = sys.intern(value)
    assert noninterned == interned
    assert noninterned is not interned
    assert not _runtime_marshaled_string_interned(noninterned)
    assert _runtime_marshaled_string_interned(interned)
    expected = base.replace(co_consts=(noninterned,))
    round_tripped = marshal.loads(marshal.dumps(expected))
    assert isinstance(round_tripped, CodeType)

    assert _runtime_code_matches_source(round_tripped, expected)
    assert not _runtime_code_matches_source(
        base.replace(co_consts=(interned,)),
        expected,
    )


def test_eval_runtime_inventory_rejects_marshaled_string_interning_change(
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    value = "runtime bytecode interning fixture value"
    source_path = eval_root / "benign.py"
    source_path.write_text(f"VALUE = {value!r}\n", encoding="utf-8")
    source_snapshot = _snapshot_file(
        tmp_path, "eval/benign.py", "retained runtime source"
    )
    payload = _compiled_runtime_source_payload(
        tmp_path,
        "eval/benign.py",
        source_snapshot,
        0,
    )
    encoded = value.encode("ascii")
    value_offset = payload.index(encoded)
    tag_offset = value_offset - 2
    assert payload[tag_offset] & 0x7F == ord("z")
    assert payload[tag_offset + 1] == len(encoded)
    changed_payload = bytearray(payload)
    changed_payload[tag_offset] = (
        changed_payload[tag_offset] & 0x80
    ) | ord("Z")
    expected_code = marshal.loads(payload)
    observed_code = marshal.loads(changed_payload)
    assert isinstance(expected_code, CodeType)
    assert isinstance(observed_code, CodeType)
    assert expected_code.co_consts == observed_code.co_consts
    assert not _runtime_code_matches_source(observed_code, expected_code)

    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"benign.{cache_tag}.pyc"
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(header + changed_payload)
    inventory = _eval_runtime_source_inventory(tmp_path)

    with pytest.raises(
        RegistryViolation,
        match="differs from its retained source",
    ):
        _verify_runtime_bytecode_caches(
            tmp_path,
            inventory,
            {"eval/benign.py": source_snapshot},
        )


def test_runtime_code_authentication_accepts_deep_exact_graph() -> None:
    base = _TRUSTED_COMPILE(
        "pass\n",
        "<runtime-code-depth>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    expected_constant: object = None
    observed_constant: object = None
    for _ in range(512):
        expected_constant = (expected_constant,)
        observed_constant = (observed_constant,)
    expected = base.replace(co_consts=(expected_constant,))
    observed = base.replace(co_consts=(observed_constant,))

    assert _runtime_code_matches_source(observed, expected)


def test_runtime_code_authentication_enforces_graph_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _TRUSTED_COMPILE(
        "pass\n",
        "<runtime-code-bound>",
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    monkeypatch.setattr(sys.modules[__name__], "MAX_RUNTIME_CODE_GRAPH_NODES", 1)

    assert not _runtime_code_matches_source(base, base)


def test_eval_runtime_inventory_types_marshal_allocation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "benign.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"benign.{cache_tag}.pyc"
    source_snapshot = _snapshot_file(
        tmp_path, "eval/benign.py", "retained runtime source"
    )
    payload = _compiled_runtime_source_payload(
        tmp_path, "eval/benign.py", source_snapshot, 0
    )
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(header + payload)
    inventory = _eval_runtime_source_inventory(tmp_path)

    def fail_load(_stream: object) -> NoReturn:
        raise MemoryError("fixture allocation refusal")

    monkeypatch.setattr(marshal, "load", fail_load)

    with pytest.raises(
        RegistryViolation,
        match="evaluator_runtime_bytecode_resource_unavailable.*MemoryError",
    ):
        _verify_runtime_bytecode_caches(
            tmp_path, inventory, {"eval/benign.py": source_snapshot}
        )


def test_eval_runtime_inventory_preflights_marshal_container_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "benign.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"benign.{cache_tag}.pyc"
    source_snapshot = _snapshot_file(
        tmp_path, "eval/benign.py", "retained runtime source"
    )
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(
        header + b"(" + struct.pack("<i", 0x7FFFFFFF)
    )
    inventory = _eval_runtime_source_inventory(tmp_path)

    def reject_load(_stream: object) -> NoReturn:
        raise AssertionError("marshal.load ran before bounded preflight")

    monkeypatch.setattr(marshal, "load", reject_load)

    with pytest.raises(
        RegistryViolation,
        match=(
            "evaluator_runtime_bytecode_resource_unavailable.*"
            "bounded graph limit"
        ),
    ):
        _verify_runtime_bytecode_caches(
            tmp_path, inventory, {"eval/benign.py": source_snapshot}
        )


def test_eval_runtime_inventory_enforces_aggregate_file_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval/one.py").write_text("ONE = 1\n", encoding="utf-8")
    (tmp_path / "eval/two.py").write_text("TWO = 2\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "MAX_EVAL_RUNTIME_FILES", 1)

    with pytest.raises(RegistryViolation, match="eval_runtime_inventory_file_limit"):
        _eval_runtime_source_inventory(tmp_path)


def test_eval_runtime_inventory_bounds_directory_entries_while_scanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "one.data").write_bytes(b"one")
    (eval_root / "two.data").write_bytes(b"two")
    monkeypatch.setattr(
        sys.modules[__name__], "MAX_EVAL_RUNTIME_ENTRIES", 1
    )

    with pytest.raises(
        RegistryViolation,
        match="directory_entry_limit_exceeded",
    ):
        _eval_runtime_source_inventory(tmp_path)


def test_runtime_cache_aggregate_budget_precedes_payload_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    cache_root = eval_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source_path = eval_root / "benign.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    source_snapshot = _snapshot_file(
        tmp_path, "eval/benign.py", "retained runtime source"
    )
    payload = _compiled_runtime_source_payload(
        tmp_path, "eval/benign.py", source_snapshot, 0
    )
    cache_tag = sys.implementation.cache_tag
    assert type(cache_tag) is str
    cache_path = cache_root / f"benign.{cache_tag}.pyc"
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + b"\0" * 8
    cache_path.write_bytes(header + payload)
    inventory = _eval_runtime_source_inventory(tmp_path)
    preflight_calls = 0

    def reject_preflight(_payload: bytes) -> NoReturn:
        nonlocal preflight_calls
        preflight_calls += 1
        raise AssertionError("over-budget cache reached marshal preflight")

    monkeypatch.setattr(
        sys.modules[__name__],
        "_preflight_runtime_marshal_payload",
        reject_preflight,
    )

    with pytest.raises(
        RegistryViolation,
        match="authority_snapshot_size_limit_exceeded",
    ):
        _verify_runtime_bytecode_caches(
            tmp_path,
            inventory,
            {"eval/benign.py": source_snapshot},
            max_cache_bytes=cache_path.stat().st_size - 1,
        )

    assert preflight_calls == 0


def test_sim_inventory_enforces_aggregate_byte_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    (sim_root / "fixture.rs").write_text("fn main() {}\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "MAX_SIM_SOURCE_BYTES", 1)

    with pytest.raises(RegistryViolation, match="sim_source_inventory_byte_limit"):
        _sim_snapshots(sim_root)


def test_retained_snapshot_survives_path_replacement_without_reread(
    tmp_path: Path,
) -> None:
    live_entrypoint = _snapshot_file(REPO_ROOT, ENTRYPOINT, "live fixture source")
    document, snapshots = _write_graph_fixture(tmp_path, live_entrypoint.data)
    entrypoint = tmp_path / ENTRYPOINT
    planted = tmp_path / "planted.py"
    planted.write_text("PLANTED = True\n", encoding="utf-8")
    entrypoint.unlink()
    entrypoint.symlink_to(planted)

    visited, edges = _build_import_graph(document, tmp_path, snapshots)
    assert visited == {ENTRYPOINT}
    assert edges == set()
    namespace = _execute_validated_blocked_snapshot(
        snapshots[ENTRYPOINT],
        module_name="_retained_snapshot_fixture",
        repo_root=tmp_path,
    )
    with pytest.raises(namespace["EvaluatorUnavailable"]):
        namespace["evaluate"]()
    with pytest.raises(RegistryViolation, match="symlinked or unreadable"):
        _snapshot_file(tmp_path, ENTRYPOINT, "fresh source")


def test_sim_reverse_guard_rejects_symlinked_sources(tmp_path: Path) -> None:
    sim_root = tmp_path / "sim"
    sim_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    (sim_root / "fixture.py").symlink_to(outside)
    with pytest.raises(RegistryViolation, match="symlinked file"):
        _sim_evaluator_references(sim_root, ENTRYPOINT_MODULE)
