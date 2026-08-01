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
import json
import os
import re
import stat
import sys
import zipimport
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import FunctionType, MethodType, ModuleType
from typing import Any, NoReturn

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_RELATIVE = "eval/guards/evaluator_import_surface_registry.v1.json"
SCHEMA_RELATIVE = "eval/guards/evaluator_import_surface_registry.v1.schema.json"
REGISTRY_SHA256 = "4f6d0f71d7dee12815b02432cf5b3bd17cec7a3c236561b3c1f4227de1b3fdac"
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
SIM_PYTHON_DYNAMIC_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
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


def _fail(message: str) -> NoReturn:
    raise RegistryViolation(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_SHA256)
    )


def _safe_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
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


def _snapshot_file(repo_root: Path, relative: str, context: str) -> FileSnapshot:
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
        if before.st_size > AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES:
            _fail(
                "authority_snapshot_size_limit_exceeded: "
                f"{context} file {relative} is {before.st_size} bytes; "
                f"ceiling={AUTHORITY_SNAPSHOT_MAX_SIZE_BYTES}"
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
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
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
        return FileSnapshot(relative, data, hashlib.sha256(data).hexdigest())
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
    repo_root: Path, relative: str, context: str
) -> dict[str, int]:
    descriptor = _open_directory(repo_root, relative, context)
    assert descriptor is not None
    try:
        result: dict[str, int] = {}
        for name in sorted(os.listdir(descriptor)):
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                joined = f"{relative}/{name}" if relative else name
                _fail(f"{context} contains a symlink: {joined}")
            result[name] = metadata.st_mode
        return result
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(snapshot: FileSnapshot, context: str) -> Any:
    try:
        return json.loads(
            snapshot.utf8(context), object_pairs_hook=_reject_duplicate_keys
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
    rows = document["classifications"]
    paths = [row["path"] for row in rows]
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
    canonical = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if registry_snapshot.data != canonical:
        _fail("classification registry is not deterministic sorted JSON plus LF")
    binding = document.get("schema_binding")
    if not isinstance(binding, dict) or binding.get("path") != SCHEMA_RELATIVE:
        _fail("registry schema path binding is absent or unknown")
    schema_snapshot = _snapshot_file(repo_root, SCHEMA_RELATIVE, "registry schema")
    if (
        not _is_sha256(binding.get("sha256"))
        or schema_snapshot.sha256 != binding["sha256"]
    ):
        _fail("registry schema hash binding drifted")
    schema = _strict_json(schema_snapshot, "classification registry schema")
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
    for path, row in rows.items():
        snapshot = _snapshot_file(repo_root, path, "classification")
        expected = row["sha256"]
        if not _is_sha256(expected):
            _fail(f"classification has a malformed SHA-256: {path}")
        if snapshot.sha256 != expected:
            _fail(
                "classification source hash drifted: "
                f"{path}; expected {expected}, got {snapshot.sha256}"
            )
        snapshots[path] = snapshot
    _verify_blocked_package_inventory(repo_root)
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
        if name == "__future__" and fromlist == ("annotations",) and level == 0:
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
            name = _reflected_callable_name(node.func, bindings)
            if name in SIM_PYTHON_DYNAMIC_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: dynamic code/import API {name!r}"
                )
            if name in SIM_PYTHON_PROCESS_CALL_NAMES:
                violations.append(
                    f"{relative}:{node.lineno}: process execution API {name!r}"
                )
            for argument in [*node.args, *(item.value for item in node.keywords)]:
                value = _constant_string(argument, bindings)
                if value in evaluator_spellings:
                    violations.append(
                        f"{relative}:{node.lineno}: computed evaluator module "
                        f"reference {value!r}"
                    )
        elif isinstance(node, ast.Name) and node.id in SIM_PYTHON_DYNAMIC_CALL_NAMES:
            violations.append(
                f"{relative}:{node.lineno}: dynamic code/import reference {node.id!r}"
            )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in SIM_PYTHON_DYNAMIC_CALL_NAMES
        ):
            violations.append(
                f"{relative}:{node.lineno}: dynamic code/import reference {node.attr!r}"
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


def test_production_evaluator_transitive_import_graph_is_closed() -> None:
    validated = _validate_registry()
    visited, edges = _build_import_graph(
        validated.document,
        REPO_ROOT,
        validated.snapshots,
    )
    assert visited == {ENTRYPOINT}
    assert edges == set()


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
