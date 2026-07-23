#!/usr/bin/env python3
"""Verify the sealed interrupted SOL runnability archive and v1 restore.

The one-off archive/restore transition completed on 2026-07-18.  This module is
intentionally read-only: the retained 53/201 bytes, ledger, and manifest are
historical evidence and must not be republished, rewritten, or deleted. Relocated
copies require a separate versioned path-alias authority and are refused here;
recorded original paths remain part of the fixed contract. The admitted repository
topology is likewise fixed to an owned standard ``.git`` directory with a
self-contained SHA-1 object database and loose/packed files refs; gitfiles, linked
worktrees, separate gitdirs, object alternates, and reftable need a new versioned
topology authority and fail closed here. The retained partial also predates the
closed runnability-schema root: its exact bytes contain fifteen producer
provenance fields outside that schema. Verification therefore checks the hash-pinned
schema projection, an exact allowlist for those retained fields, and rederived cell
semantics; it never upgrades or describes the immutable partial as schema-valid. The
recorded whole-worktree diff digest included unrelated dirty bytes that were not
archived, so it is retained and byte-attested rather than falsely reconstructed from
the payload. Verification requires a quiescent checkout: descriptor/stat rechecks
detect ordinary concurrent changes, but Git does not honor this verifier's advisory
file locks, so the command cannot exclude an adversarial same-UID writer after its
final observation.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import select
import stat
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_BINARY = Path("/usr/bin/git")
DEFAULT_ARCHIVE_BASE = Path("/home/jakeshea/sm86_reference_capture_r27")
SOURCE_RELATIVE_PATH = PurePosixPath(
    "data/local_labels/campaign/sol_runnability_sm86.json"
)
SCHEMA_RELATIVE_PATH = PurePosixPath(
    "workloads/simbench_bw/schemas/sol_runnability.schema.json"
)
ARCHIVE_ROOT_NAME = "interrupted_runnability_attempts"
LEDGER_NAME = "interrupted_runnability_ledger.jsonl"
LOCK_NAME = ".archive.lock"
MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
PAYLOAD_DIRECTORY = "payload"
PAYLOAD_NAME = SOURCE_RELATIVE_PATH.name
ARTIFACT_ID = "sm86_sol_runnability_interrupted_attempt_archive_v1"
ENTRY_TYPE = "sm86_sol_runnability_interrupted_attempt_ledger_v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RECORDED_SOURCE_ABSOLUTE_PATH = Path(
    "/home/jakeshea/GPU_simulator_rebase/"
    "data/local_labels/campaign/sol_runnability_sm86.json"
)
LEDGER_MAX_SIZE_BYTES = 16 * 1024
MANIFEST_MAX_SIZE_BYTES = 64 * 1024
PAYLOAD_MAX_SIZE_BYTES = 8 * 1024 * 1024
CANONICAL_MAX_SIZE_BYTES = 1024 * 1024
SCHEMA_MAX_SIZE_BYTES = 1024 * 1024
GIT_COMMAND_TIMEOUT_SECONDS = 120
GIT_CONFIG_MAX_SIZE_BYTES = 1024 * 1024
GIT_INDEX_MAX_SIZE_BYTES = 16 * 1024 * 1024
GIT_METADATA_MAX_ENTRIES = 4096
GIT_METADATA_MAX_PINNED_DESCRIPTORS = 64
GIT_SUBPROCESS_FD_HEADROOM = 8
GIT_REFERENCE_MAX_COMPONENTS = 16
GIT_SYMBOLIC_REFERENCE_MAX_CHAIN = 32
GIT_OBJECT_TREE_MAX_ENTRIES = 100_000
GIT_OBJECT_TREE_MAX_DEPTH = 4
JSONSCHEMA_VERSION = "4.26.0"
_STAT_STABLE_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
    "st_size", "st_mtime_ns", "st_ctime_ns", "st_blocks",
)

DISPOSITION: Mapping[str, bool | int] = MappingProxyType({
    "authoritative": False,
    "resumable": False,
    "reusable": False,
    "campaign_rows": 0,
})
PARTIAL_RUNNABILITY_STATUSES = (
    "runnable",
    "compile_failed",
    "correctness_failed",
    "host_capacity_oom",
    "eval_incomplete",
    "eval_error",
    "resolve_error",
    "environment_invalid",
    "prerequisite_missing",
    "eval_timeout",
)
PARTIAL_EXPECTED_BY_STATUS: Mapping[str, int] = MappingProxyType({
    "runnable": 48,
    "compile_failed": 0,
    "correctness_failed": 1,
    "host_capacity_oom": 4,
    "eval_incomplete": 0,
    "eval_error": 0,
    "resolve_error": 0,
    "environment_invalid": 0,
    "prerequisite_missing": 0,
    "eval_timeout": 0,
})
CANONICAL_EXPECTED_BY_STATUS: Mapping[str, int] = MappingProxyType({
    "runnable": 73,
    "compile_failed": 65,
    "correctness_failed": 39,
    "host_capacity_oom": 17,
    "eval_incomplete": 0,
    "eval_error": 7,
    "resolve_error": 0,
})
PARTIAL_STATUS_TO_TYPED_REASON: Mapping[str, str] = MappingProxyType({
    "compile_failed": "calibration_inconclusive",
    "correctness_failed": "correctness_failed",
    "host_capacity_oom": "host_capacity_deferred",
    "eval_incomplete": "calibration_inconclusive",
    "eval_error": "calibration_inconclusive",
    "resolve_error": "calibration_inconclusive",
    "environment_invalid": "calibration_inconclusive",
    "prerequisite_missing": "calibration_inconclusive",
    "eval_timeout": "calibration_inconclusive",
})
PARTIAL_RETAINED_UNSCHEMATIZED_FIELDS = frozenset({
    "cli_timeout_seconds",
    "data_annex_manifest_sha256",
    "flashinfer_corpus_blob_count",
    "flashinfer_corpus_error",
    "flashinfer_corpus_file_count",
    "flashinfer_corpus_manifest_sha256",
    "flashinfer_corpus_verified",
    "outer_timeout_seconds",
    "probe_config_path",
    "probe_config_sha256",
    "probe_protocol_id",
    "probe_protocol_sha256",
    "resume_identity_sha256",
    "sol_execbench_package_file_count",
    "sol_execbench_package_sha256",
})
PARTIAL_SCHEMA_VALIDATION = (
    "hash_pinned_projection_with_retained_provenance_allowlist"
)
MANIFEST_FIELDS = frozenset({
    "schema_version", "artifact_id", "attempt_id", "created_at_utc",
    "source", "canonical", "payload", "disposition", "manifest_seal_sha256",
})
SOURCE_FIELDS = frozenset({
    "repository_path", "absolute_path", "commit", "size_bytes", "sha256",
    "git_status_porcelain_v1_z_base64", "git_status_sha256",
    "git_worktree_diff_sha256", "git_index_diff_sha256",
})
CANONICAL_FIELDS = frozenset({
    "repository_path", "commit", "size_bytes", "sha256",
})
PAYLOAD_FIELDS = frozenset({"archive_relative_path", "size_bytes", "sha256"})
LEDGER_FIELDS = frozenset({
    "schema_version", "entry_type", "attempt_id", "created_at_utc",
    "archive_path", "manifest_path", "manifest_sha256", "source_commit",
    "source_sha256", "authoritative", "resumable", "reusable",
    "campaign_rows",
})
PORCELAIN_V1_STATUS_COMBINATIONS = frozenset({
    b" M", b" T", b" D", b" R", b" C",
    b"M ", b"MM", b"MT", b"MD",
    b"A ", b"AM", b"AT", b"AD",
    b"D ",
    b"R ", b"RM", b"RT", b"RD",
    b"C ", b"CM", b"CT", b"CD",
    b"T ", b"TM", b"TT", b"TD",
    b"DD", b"AU", b"UD", b"UA", b"DU", b"AA", b"UU",
    b"??", b"!!",
})

@dataclass(frozen=True)
class FixedArchiveContract:
    recorded_archive_base: Path
    recorded_source_absolute_path: Path
    source_relative_path: PurePosixPath
    schema_relative_path: PurePosixPath
    source_commit: str
    source_sha256: str
    source_size_bytes: int
    canonical_sha256: str
    canonical_size_bytes: int
    canonical_tree_mode: str
    canonical_blob_oid: str
    schema_sha256: str
    attempt_id: str
    manifest_sha256: str
    manifest_size_bytes: int
    ledger_sha256: str
    ledger_size_bytes: int
    created_at_utc: str


FIXED_CONTRACT = FixedArchiveContract(
    recorded_archive_base=DEFAULT_ARCHIVE_BASE,
    recorded_source_absolute_path=RECORDED_SOURCE_ABSOLUTE_PATH,
    source_relative_path=SOURCE_RELATIVE_PATH,
    schema_relative_path=SCHEMA_RELATIVE_PATH,
    source_commit="36c3b7042955e990e1d32a470bfab2185c75dbe2",
    source_sha256="b0ba740d9e5a170008ccb0307a6d91916705e89ca1a0f8f59dddd5dc380f2324",
    source_size_bytes=4_184_739,
    canonical_sha256="760853e5916b78ea4e233b5995f6db49d4d8942ec09d240e02f2e072ed8bbe4f",
    canonical_size_bytes=119_029,
    canonical_tree_mode="100644",
    canonical_blob_oid="648e1184cd13faf26ac9262e50b2a1bb35b902df",
    schema_sha256="1397e2e49c247ff060ed0c2141020a3512de3d7aef53eca523bbde3409dc3f2e",
    attempt_id="4de2b0208d7a11fee21ad2d24421af0d14d0391f9964c806af0caac0f1a7eb0f",
    manifest_sha256="c9a34ad5484f522152d6711f903478feab3f9dbda755f1b65326a2b50d3711ee",
    manifest_size_bytes=3_080,
    ledger_sha256="68807ce919d869b7d07e901e03e988ef4c22d96237dcd5ca276537241548ef1f",
    ledger_size_bytes=869,
    created_at_utc="2026-07-18T07:02:43.933076Z",
)


class RunnabilityArchiveRefusal(RuntimeError):
    """Stable typed refusal for fixed-archive and restoration verification."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_oid(raw: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(b"blob ")
    digest.update(str(len(raw)).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"non-canonical JSON value: {exc}",
        ) from exc
    return sha256_bytes(encoded)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")


def _load_json_bytes(raw: bytes, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role}: root is not an object",
        )
    return value


def _trusted_jsonschema_validator(
    *, refusal_code: str,
) -> tuple[type[Any], tuple[type[Exception], ...]]:
    try:
        trusted_import_roots = tuple(dict.fromkeys(
            Path(path).resolve(strict=True)
            for path in (
                sysconfig.get_path("stdlib"),
                sysconfig.get_path("platstdlib"),
                sysconfig.get_path("purelib"),
                sysconfig.get_path("platlib"),
            )
            if path
        ))
        trusted_distribution_roots = {
            Path(path).resolve(strict=True)
            for path in (
                sysconfig.get_path("purelib"),
                sysconfig.get_path("platlib"),
            )
            if path
        }
        distribution = importlib.metadata.distribution("jsonschema")
        distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
        if (
            distribution.version != JSONSCHEMA_VERSION
            or distribution_root not in trusted_distribution_roots
        ):
            raise RunnabilityArchiveRefusal(
                refusal_code,
                "jsonschema distribution/version is not the pinned trusted authority",
            )

        package_root = Path(
            distribution.locate_file("jsonschema"),
        ).resolve(strict=True)
        package_spec = importlib.util.find_spec("jsonschema")
        package_locations = (
            tuple(
                Path(location).resolve(strict=True)
                for location in package_spec.submodule_search_locations
            )
            if package_spec is not None
            and package_spec.submodule_search_locations is not None
            else ()
        )
        if (
            package_spec is None
            or package_spec.origin is None
            or Path(package_spec.origin).resolve(strict=True)
            != package_root / "__init__.py"
            or package_locations != (package_root,)
        ):
            raise RunnabilityArchiveRefusal(
                refusal_code,
                "jsonschema import resolution is outside the pinned distribution",
            )

        original_sys_path = list(sys.path)
        modules_before = frozenset(sys.modules)
        try:
            sys.path[:] = [os.fspath(root) for root in trusted_import_roots]
            importlib.invalidate_caches()
            modules = {
                "jsonschema": importlib.import_module("jsonschema"),
                "jsonschema.validators": importlib.import_module(
                    "jsonschema.validators",
                ),
                "jsonschema.exceptions": importlib.import_module(
                    "jsonschema.exceptions",
                ),
            }
        finally:
            sys.path[:] = original_sys_path
            importlib.invalidate_caches()

        expected_files = {
            "jsonschema": package_root / "__init__.py",
            "jsonschema.validators": package_root / "validators.py",
            "jsonschema.exceptions": package_root / "exceptions.py",
        }
        for name, module in modules.items():
            module_file = Path(module.__file__).resolve(strict=True)
            module_origin = Path(module.__spec__.origin).resolve(strict=True)
            if module_file != expected_files[name] or module_origin != module_file:
                raise RunnabilityArchiveRefusal(
                    refusal_code,
                    f"{name} was imported outside the pinned distribution",
                )

        for name, module in tuple(sys.modules.items()):
            if name != "jsonschema" and not name.startswith("jsonschema."):
                continue
            module_file_value = getattr(module, "__file__", None)
            module_spec = getattr(module, "__spec__", None)
            module_origin_value = getattr(module_spec, "origin", None)
            if not isinstance(module_file_value, str) or not isinstance(
                module_origin_value, str
            ):
                raise RunnabilityArchiveRefusal(
                    refusal_code,
                    f"{name} has no trusted module origin",
                )
            module_file = Path(module_file_value).resolve(strict=True)
            module_origin = Path(module_origin_value).resolve(strict=True)
            if (
                module_origin != module_file
                or not module_file.is_relative_to(package_root)
            ):
                raise RunnabilityArchiveRefusal(
                    refusal_code,
                    f"{name} was imported outside the pinned distribution",
                )

        for name in frozenset(sys.modules) - modules_before:
            module = sys.modules[name]
            module_file_value = getattr(module, "__file__", None)
            module_spec = getattr(module, "__spec__", None)
            module_origin_value = getattr(module_spec, "origin", None)
            if module_file_value is None:
                if module_origin_value not in {"built-in", "frozen"}:
                    raise RunnabilityArchiveRefusal(
                        refusal_code,
                        f"jsonschema imported {name} without a trusted origin",
                    )
                continue
            if not isinstance(module_file_value, str):
                raise RunnabilityArchiveRefusal(
                    refusal_code,
                    f"jsonschema imported {name} without a trusted file",
                )
            module_file = Path(module_file_value).resolve(strict=True)
            if not any(
                module_file == root or module_file.is_relative_to(root)
                for root in trusted_import_roots
            ):
                raise RunnabilityArchiveRefusal(
                    refusal_code,
                    f"jsonschema imported {name} outside trusted Python roots",
                )

        validator = modules["jsonschema.validators"].Draft202012Validator
        schema_error = modules["jsonschema.exceptions"].SchemaError
        validation_error = modules["jsonschema.exceptions"].ValidationError
        if (
            modules["jsonschema"].Draft202012Validator is not validator
            or validator.__module__ != "jsonschema.validators"
            or schema_error.__module__ != "jsonschema.exceptions"
            or validation_error.__module__ != "jsonschema.exceptions"
        ):
            raise RunnabilityArchiveRefusal(
                refusal_code, "jsonschema validator identity is not trusted",
            )
        return validator, (schema_error, validation_error)
    except RunnabilityArchiveRefusal:
        raise
    except Exception as exc:
        raise RunnabilityArchiveRefusal(
            refusal_code, f"trusted jsonschema unavailable: {exc}",
        ) from exc


def _validate_with_jsonschema(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    refusal_code: str,
    role: str,
) -> None:
    validator, expected_errors = _trusted_jsonschema_validator(
        refusal_code=refusal_code,
    )
    try:
        validator.check_schema(schema)
        validator(schema).validate(document)
    except expected_errors as exc:
        raise RunnabilityArchiveRefusal(
            refusal_code, f"{role}: {exc}",
        ) from exc
    except Exception as exc:
        raise RunnabilityArchiveRefusal(
            refusal_code, f"{role}: trusted validator failure: {exc}",
        ) from exc


def _is_sha(value: Any, length: int = 64) -> bool:
    return (
        type(value) is str and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _exact_mapping_equal(
    observed: Any, expected: Mapping[str, Any],
) -> bool:
    return (
        isinstance(observed, dict)
        and set(observed) == set(expected)
        and all(
            type(observed[key]) is type(expected_value)
            and observed[key] == expected_value
            for key, expected_value in expected.items()
        )
    )


def _parse_utc(value: Any, *, expected: str) -> None:
    valid = type(value) is str and value == expected and value.endswith("Z")
    if valid:
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
            valid = parsed.tzinfo == timezone.utc
        except ValueError:
            valid = False
    if not valid:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            f"created_at_utc mismatch: {value!r}",
        )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _stat_matches(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in _STAT_STABLE_FIELDS
    )


def _validate_directory_info(
    info: os.stat_result, *, role: str, require_owned: bool,
) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role} is not a directory",
        )
    if require_owned:
        safe = info.st_uid == os.geteuid() and not info.st_mode & 0o022
    else:
        safe_owner = info.st_uid in (0, os.geteuid())
        safe_writes = not info.st_mode & 0o022 or bool(info.st_mode & stat.S_ISVTX)
        safe = safe_owner and safe_writes
    if not safe:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID",
            f"{role} ownership/mode is unsafe",
        )


def _require_exact_mode(
    info: os.stat_result, expected_mode: int, *, role: str,
) -> None:
    observed_mode = stat.S_IMODE(info.st_mode)
    if observed_mode != expected_mode:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID",
            f"{role} mode drift: {observed_mode:#05o} != {expected_mode:#05o}",
        )


def _require_device(
    info: os.stat_result,
    expected_device: int,
    *,
    role: str,
    code: str = "RUNNABILITY_ARCHIVE_PATH_INVALID",
) -> None:
    if info.st_dev != expected_device:
        raise RunnabilityArchiveRefusal(
            code, f"{role} crosses a filesystem boundary",
        )


def _safe_child_name(name: str) -> bool:
    return (
        bool(name)
        and name not in {".", ".."}
        and "\0" not in name
        and Path(name).name == name
    )


def _open_absolute_directory_chain(
    path: Path, *, role: str, require_owned: bool,
) -> tuple[list[int], list[os.stat_result], list[str]]:
    absolute = path.absolute()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    infos: list[os.stat_result] = []
    names: list[str] = []
    try:
        try:
            root_descriptor = os.open(os.path.sep, flags)
            descriptors.append(root_descriptor)
            root_info = os.fstat(root_descriptor)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role}: {exc}",
            ) from exc
        _validate_directory_info(
            root_info, role="filesystem root", require_owned=False,
        )
        infos.append(root_info)
        for index, part in enumerate(absolute.parts[1:], start=1):
            component_role = f"{role} component {part!r}"
            try:
                child_descriptor = os.open(
                    part, flags, dir_fd=descriptors[-1],
                )
                descriptors.append(child_descriptor)
                child_info = os.fstat(child_descriptor)
            except OSError as exc:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_ARCHIVE_PATH_INVALID",
                    f"{component_role}: {exc}",
                ) from exc
            _validate_directory_info(
                child_info,
                role=component_role,
                require_owned=(
                    require_owned if index == len(absolute.parts) - 1 else False
                ),
            )
            names.append(part)
            infos.append(child_info)
        if len(absolute.parts) == 1 and require_owned:
            _validate_directory_info(root_info, role=role, require_owned=True)
        return descriptors, infos, names
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _verify_absolute_directory_chain(
    descriptors: Sequence[int],
    infos: Sequence[os.stat_result],
    names: Sequence[str],
    *,
    role: str,
    change_code: str,
) -> None:
    if len(descriptors) != len(infos) or len(names) + 1 != len(descriptors):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            f"{role} absolute-path binding is malformed",
        )
    for index, name in enumerate(names):
        component_role = f"{role} component {name!r}"
        try:
            observed = os.stat(
                name,
                dir_fd=descriptors[index],
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_PATH_INVALID",
                f"{component_role}: {exc}",
            ) from exc
        if not _stat_matches(infos[index + 1], observed):
            raise RunnabilityArchiveRefusal(
                change_code,
                f"{component_role} changed during absolute-path binding",
            )
    for index, (descriptor, expected) in enumerate(zip(descriptors, infos)):
        component_role = (
            "filesystem root"
            if index == 0
            else f"{role} component {names[index - 1]!r}"
        )
        try:
            observed = os.fstat(descriptor)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_PATH_INVALID",
                f"{component_role}: {exc}",
            ) from exc
        if not _stat_matches(expected, observed):
            raise RunnabilityArchiveRefusal(
                change_code,
                f"{component_role} changed during absolute-path binding",
            )


def _open_absolute_directory(
    path: Path, *, role: str, require_owned: bool = True,
) -> tuple[int, os.stat_result]:
    descriptors, infos, names = _open_absolute_directory_chain(
        path, role=role, require_owned=require_owned,
    )
    try:
        _verify_absolute_directory_chain(
            descriptors,
            infos,
            names,
            role=role,
            change_code="RUNNABILITY_ARCHIVE_PATH_INVALID",
        )
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    for descriptor in reversed(descriptors[:-1]):
        os.close(descriptor)
    return descriptors[-1], infos[-1]


def _open_owned_directory_at(
    parent_descriptor: int, name: str, *, role: str,
) -> tuple[int, os.stat_result]:
    if not _safe_child_name(name):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"unsafe {role} name: {name!r}",
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role}: {exc}",
        ) from exc
    try:
        _validate_directory_info(opened, role=role, require_owned=True)
        if not _stat_matches(before, opened):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role} changed before open",
            )
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _require_owned_directory(path: Path, *, role: str) -> os.stat_result:
    descriptor, info = _open_absolute_directory(path, role=role)
    os.close(descriptor)
    return info


def _validate_regular_info(info: os.stat_result, *, role: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role} is not regular",
        )
    special_bits = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
    if (
        info.st_uid != os.geteuid()
        or info.st_mode & (0o022 | special_bits)
        or info.st_nlink != 1
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID",
            f"{role} ownership/mode/link count is unsafe",
        )


def _open_owned_regular_at(
    parent_descriptor: int, name: str, *, role: str,
) -> tuple[int, os.stat_result]:
    if not _safe_child_name(name):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"unsafe {role} name: {name!r}",
        )
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _validate_regular_info(before, role=role)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role}: {exc}",
        ) from exc
    try:
        _validate_regular_info(opened, role=role)
        if not _stat_matches(before, opened):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role} changed before open",
            )
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _validate_observed_size_ceiling(
    info: os.stat_result, *, role: str, max_size: int,
) -> None:
    if type(max_size) is not int or max_size < 0:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"invalid {role} size ceiling",
        )
    if info.st_size > max_size:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            f"{role} exceeds maximum size: {info.st_size} > {max_size}",
        )


def _validate_read_bounds(
    info: os.stat_result,
    *,
    role: str,
    expected_size: int,
    max_size: int,
) -> None:
    if (
        type(expected_size) is not int
        or type(max_size) is not int
        or expected_size < 0
        or max_size < 0
        or expected_size > max_size
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"invalid {role} size contract",
        )
    if info.st_size > max_size:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            f"{role} exceeds maximum size: {info.st_size} > {max_size}",
        )
    if info.st_size != expected_size:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            f"{role} size mismatch: {info.st_size} != {expected_size}",
        )


def _read_regular_exact_at(
    parent_descriptor: int,
    name: str,
    *,
    role: str,
    expected_size: int,
    max_size: int,
    expected_device: int | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    descriptor, opened = _open_owned_regular_at(
        parent_descriptor, name, role=role,
    )
    try:
        if expected_device is not None:
            _require_device(opened, expected_device, role=role)
        if expected_mode is not None:
            _require_exact_mode(opened, expected_mode, role=role)
        _validate_read_bounds(
            opened, role=role, expected_size=expected_size, max_size=max_size,
        )
        payload = bytearray()
        while len(payload) < expected_size:
            chunk = os.read(descriptor, min(1 << 20, expected_size - len(payload)))
            if not chunk:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} ended early",
                )
            payload.extend(chunk)
        if os.read(descriptor, 1):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} grew while reading",
            )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_IO_FAILURE", f"{role}: {exc}",
        ) from exc
    finally:
        os.close(descriptor)
    if not _stat_matches(opened, after):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} changed while reading",
        )
    return bytes(payload), after


def _read_regular_exact(
    path: Path,
    *,
    role: str,
    expected_size: int,
    max_size: int,
    expected_device: int | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    parent_descriptor, parent_info = _open_absolute_directory(
        path.parent, role=f"{role} parent",
    )
    try:
        if expected_device is not None:
            _require_device(parent_info, expected_device, role=f"{role} parent")
        return _read_regular_exact_at(
            parent_descriptor,
            path.name,
            role=role,
            expected_size=expected_size,
            max_size=max_size,
            expected_device=expected_device,
            expected_mode=expected_mode,
        )
    finally:
        os.close(parent_descriptor)


@dataclass(frozen=True)
class GitObjectInventory:
    entry_count: int
    digest: str


@dataclass(frozen=True)
class GitMetadataGuard:
    """Pin Git metadata and detect changes through the final observation."""

    entries: tuple[tuple[int, str, os.stat_result, str], ...]
    opened: tuple[tuple[int, os.stat_result, str], ...]
    descriptors: tuple[int, ...]
    object_descriptor: int
    object_device: int
    object_inventory: GitObjectInventory
    reference_descriptor: int
    reference_device: int
    reference_inventory: GitObjectInventory

    def verify(self) -> None:
        observed_inventory = _snapshot_git_object_tree(
            self.object_descriptor,
            object_device=self.object_device,
        )
        if observed_inventory != self.object_inventory:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
                "Git object database changed during verification",
            )
        observed_references = _snapshot_git_reference_tree(
            self.reference_descriptor,
            reference_device=self.reference_device,
        )
        if observed_references != self.reference_inventory:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
                "Git loose-reference tree changed during verification",
            )
        for parent_descriptor, name, info, role in self.entries:
            _assert_entry_unchanged(
                parent_descriptor, name, info, role=role,
            )
        for descriptor, info, role in self.opened:
            _assert_opened_unchanged(descriptor, info, role=role)

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)


def _read_open_regular_bounded(
    descriptor: int,
    info: os.stat_result,
    *,
    role: str,
    max_size: int,
) -> bytes:
    _validate_observed_size_ceiling(info, role=role, max_size=max_size)
    _validate_read_bounds(
        info,
        role=role,
        expected_size=info.st_size,
        max_size=max_size,
    )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = bytearray()
        while len(payload) < info.st_size:
            chunk = os.read(
                descriptor, min(4096, info.st_size - len(payload)),
            )
            if not chunk:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
                    f"{role} ended while being pinned",
                )
            payload.extend(chunk)
        if os.read(descriptor, 1):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
                f"{role} grew while being pinned",
            )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
        ) from exc
    if not _stat_matches(info, after):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
            f"{role} changed while being pinned",
        )
    return bytes(payload)


def _require_absent_entry(
    parent_descriptor: int, name: str, *, role: str,
) -> None:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
        ) from exc
    raise RunnabilityArchiveRefusal(
        "RUNNABILITY_RESTORE_GIT_LOCKED", f"{role} exists",
    )


def _require_absent_git_authority(
    parent_descriptor: int, name: str, *, role: str,
) -> None:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
        ) from exc
    raise RunnabilityArchiveRefusal(
        "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED", f"{role} exists",
    )


def _decode_symbolic_reference(raw: bytes, *, role: str) -> str | None:
    if not raw.startswith(b"ref: "):
        return None
    raw_reference = raw[5:-1] if raw.endswith(b"\n") else b""
    reference = os.fsdecode(raw_reference)
    reference_path = PurePosixPath(reference)
    if (
        not raw_reference
        or any(byte < 0x20 or byte == 0x7f for byte in raw_reference)
        or raw != b"ref: " + os.fsencode(reference) + b"\n"
        or reference_path.is_absolute()
        or not reference_path.parts
        or reference_path.parts[0] != "refs"
        or any(part in {"", ".", ".."} for part in reference_path.parts)
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            f"unsafe {role} symbolic reference: {reference!r}",
        )
    return reference


def _stat_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(info, field) for field in _STAT_STABLE_FIELDS)


def _git_object_inventory_leaf(
    relative: str, kind: str, info: os.stat_result,
) -> bytes:
    digest = hashlib.sha256()
    for field in (
        os.fsencode(relative),
        kind.encode("ascii"),
        *(str(value).encode("ascii") for value in _stat_fingerprint(info)),
    ):
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)
    return digest.digest()


def _require_git_descriptor_capacity(
    descriptors: Sequence[int], *, additional: int = 1,
) -> None:
    if (
        type(additional) is not int
        or additional < 0
        or len(descriptors) + additional > GIT_METADATA_MAX_PINNED_DESCRIPTORS
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "Git metadata authority exceeds descriptor budget",
        )
    try:
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit == resource.RLIM_INFINITY:
            return
        admitted_open = soft_limit - GIT_SUBPROCESS_FD_HEADROOM - additional
        if admitted_open < 0:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                "process descriptor limit leaves no Git subprocess headroom",
            )
        open_count = 0
        with os.scandir("/proc/self/fd") as iterator:
            for _entry in iterator:
                open_count += 1
                if open_count > admitted_open:
                    raise RunnabilityArchiveRefusal(
                        "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                        "process descriptor limit leaves no Git subprocess headroom",
                    )
    except RunnabilityArchiveRefusal:
        raise
    except (OSError, ValueError) as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            f"cannot establish Git descriptor headroom: {exc}",
        ) from exc


def _assert_git_inventory_opened_unchanged(
    descriptor: int, expected: os.stat_result, *, role: str,
) -> None:
    try:
        observed = os.fstat(descriptor)
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
        ) from exc
    if not _stat_matches(expected, observed):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
            f"{role} changed during Git metadata inventory",
        )


def _assert_git_inventory_entry_unchanged(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    role: str,
) -> None:
    try:
        observed = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False,
        )
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
        ) from exc
    if not _stat_matches(expected, observed):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
            f"{role} changed during Git metadata inventory",
        )


def _snapshot_git_reference_tree(
    parent_descriptor: int,
    *,
    reference_device: int,
) -> GitObjectInventory:
    leaf_digests: list[bytes] = []
    count = 0

    def scan(descriptor: int, relative_parts: tuple[str, ...]) -> None:
        nonlocal count
        if len(relative_parts) >= GIT_REFERENCE_MAX_COMPONENTS - 1:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                "Git loose-reference tree exceeds admitted directory depth",
            )
        directory_role = (
            "Git loose-reference root"
            if not relative_parts
            else "Git loose-reference directory refs/" + "/".join(relative_parts)
        )
        try:
            directory_info = os.fstat(descriptor)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"{directory_role}: {exc}",
            ) from exc
        try:
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    count += 1
                    if count > GIT_METADATA_MAX_ENTRIES:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            "Git loose-reference tree exceeds admitted entry count",
                        )
                    name = entry.name
                    path_parts = (*relative_parts, name)
                    relative = "/".join(path_parts)
                    role = f"Git loose-reference entry refs/{relative!s}"
                    if not _safe_child_name(name):
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            f"{role} has an unsafe name",
                        )
                    if name.endswith(".lock"):
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_LOCKED", f"{role} exists",
                        )
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
                        ) from exc
                    if stat.S_ISDIR(info.st_mode):
                        child_descriptor, child_info = _open_owned_directory_at(
                            descriptor, name, role=role,
                        )
                        try:
                            _require_device(
                                child_info,
                                reference_device,
                                role=role,
                                code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            )
                            leaf_digests.append(
                                _git_object_inventory_leaf(
                                    relative, "directory", child_info,
                                )
                            )
                            scan(child_descriptor, path_parts)
                            _assert_git_inventory_entry_unchanged(
                                descriptor, name, child_info, role=role,
                            )
                            _assert_git_inventory_opened_unchanged(
                                child_descriptor, child_info, role=role,
                            )
                        finally:
                            os.close(child_descriptor)
                    elif stat.S_ISREG(info.st_mode):
                        file_descriptor, file_info = _open_owned_regular_at(
                            descriptor, name, role=role,
                        )
                        try:
                            _require_device(
                                file_info,
                                reference_device,
                                role=role,
                                code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            )
                            leaf_digests.append(
                                _git_object_inventory_leaf(
                                    relative, "file", file_info,
                                )
                            )
                            _assert_git_inventory_entry_unchanged(
                                descriptor, name, file_info, role=role,
                            )
                            _assert_git_inventory_opened_unchanged(
                                file_descriptor, file_info, role=role,
                            )
                        finally:
                            os.close(file_descriptor)
                    else:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            f"{role} is not an owned regular file or directory",
                        )
            _assert_git_inventory_opened_unchanged(
                descriptor, directory_info, role=directory_role,
            )
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"Git loose-reference inventory: {exc}",
            ) from exc

    scan(parent_descriptor, ())
    leaf_digests.sort()
    digest = hashlib.sha256()
    digest.update(count.to_bytes(8, "big"))
    for leaf in leaf_digests:
        digest.update(leaf)
    return GitObjectInventory(count, digest.hexdigest())


def _snapshot_git_object_tree(
    parent_descriptor: int,
    *,
    object_device: int,
) -> GitObjectInventory:
    leaf_digests: list[bytes] = []
    count = 0

    def scan(descriptor: int, relative_parts: tuple[str, ...]) -> None:
        nonlocal count
        if len(relative_parts) >= GIT_OBJECT_TREE_MAX_DEPTH:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                "Git object database exceeds admitted directory depth",
            )
        directory_role = (
            "Git object database"
            if not relative_parts
            else "Git object database directory " + "/".join(relative_parts)
        )
        try:
            directory_info = os.fstat(descriptor)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"{directory_role}: {exc}",
            ) from exc
        try:
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    count += 1
                    if count > GIT_OBJECT_TREE_MAX_ENTRIES:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            "Git object database exceeds admitted entry count",
                        )
                    name = entry.name
                    path_parts = (*relative_parts, name)
                    relative = "/".join(path_parts)
                    role = f"Git object database entry {relative!r}"
                    if name.endswith(".lock"):
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_LOCKED", f"{role} exists",
                        )
                    if path_parts in {
                        ("info", "alternates"),
                        ("info", "http-alternates"),
                    } or name.endswith(".promisor"):
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            f"{role} is an external object authority",
                        )
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
                        ) from exc
                    if stat.S_ISDIR(info.st_mode):
                        child_descriptor, child_info = _open_owned_directory_at(
                            descriptor, name, role=role,
                        )
                        try:
                            _require_device(
                                child_info,
                                object_device,
                                role=role,
                                code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            )
                            leaf_digests.append(
                                _git_object_inventory_leaf(
                                    relative, "directory", child_info,
                                )
                            )
                            scan(child_descriptor, path_parts)
                            _assert_git_inventory_entry_unchanged(
                                descriptor, name, child_info, role=role,
                            )
                            _assert_git_inventory_opened_unchanged(
                                child_descriptor, child_info, role=role,
                            )
                        finally:
                            os.close(child_descriptor)
                    elif stat.S_ISREG(info.st_mode):
                        file_descriptor, file_info = _open_owned_regular_at(
                            descriptor, name, role=role,
                        )
                        try:
                            _require_device(
                                file_info,
                                object_device,
                                role=role,
                                code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            )
                            leaf_digests.append(
                                _git_object_inventory_leaf(relative, "file", file_info)
                            )
                            _assert_git_inventory_entry_unchanged(
                                descriptor, name, file_info, role=role,
                            )
                            _assert_git_inventory_opened_unchanged(
                                file_descriptor, file_info, role=role,
                            )
                        finally:
                            os.close(file_descriptor)
                    else:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                            f"{role} is not an owned regular file or directory",
                        )
            _assert_git_inventory_opened_unchanged(
                descriptor, directory_info, role=directory_role,
            )
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"Git object database inventory: {exc}",
            ) from exc

    scan(parent_descriptor, ())
    leaf_digests.sort()
    digest = hashlib.sha256()
    digest.update(count.to_bytes(8, "big"))
    for leaf in leaf_digests:
        digest.update(leaf)
    return GitObjectInventory(count, digest.hexdigest())


def _open_git_metadata_guard(repo_descriptor: int) -> GitMetadataGuard:
    """Pin the index and symbolic-HEAD chain for the full attestation."""

    entries: list[tuple[int, str, os.stat_result, str]] = []
    opened: list[tuple[int, os.stat_result, str]] = []
    descriptors: list[int] = []
    object_inventory: GitObjectInventory | None = None
    reference_inventory: GitObjectInventory | None = None
    try:
        repo_info = os.fstat(repo_descriptor)
        git_descriptor, git_info = _open_owned_directory_at(
            repo_descriptor, ".git", role="Git metadata root",
        )
        if git_info.st_dev != repo_info.st_dev:
            os.close(git_descriptor)
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                "Git metadata root crosses a filesystem boundary",
            )
        descriptors.append(git_descriptor)
        entries.append((repo_descriptor, ".git", git_info, "Git metadata root"))
        opened.append((git_descriptor, git_info, "Git metadata root"))

        objects_descriptor, objects_info = _open_owned_directory_at(
            git_descriptor, "objects", role="Git object database",
        )
        if objects_info.st_dev != git_info.st_dev:
            os.close(objects_descriptor)
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                "Git object database crosses a filesystem boundary",
            )
        descriptors.append(objects_descriptor)
        entries.append((
            git_descriptor, "objects", objects_info, "Git object database",
        ))
        opened.append((objects_descriptor, objects_info, "Git object database"))
        object_inventory = _snapshot_git_object_tree(
            objects_descriptor,
            object_device=objects_info.st_dev,
        )

        _require_git_descriptor_capacity(descriptors)
        references_descriptor, references_info = _open_owned_directory_at(
            git_descriptor, "refs", role="Git loose-reference root",
        )
        descriptors.append(references_descriptor)
        _require_device(
            references_info,
            git_info.st_dev,
            role="Git loose-reference root",
            code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
        )
        entries.append((
            git_descriptor, "refs", references_info, "Git loose-reference root",
        ))
        opened.append((
            references_descriptor, references_info, "Git loose-reference root",
        ))
        reference_inventory = _snapshot_git_reference_tree(
            references_descriptor,
            reference_device=references_info.st_dev,
        )

        git_children = _directory_names(
            git_descriptor, role="Git metadata root",
        )
        split_indexes = sorted(
            name for name in git_children if name.startswith("sharedindex.")
        )
        if split_indexes:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                f"split Git index authority is not admitted: {split_indexes}",
            )
        for lock_name in sorted(
            name for name in git_children if name.endswith(".lock")
        ):
            _require_absent_entry(
                git_descriptor,
                lock_name,
                role=f"Git transaction lock {lock_name}",
            )
        for authority_name, role in (
            ("shallow", "shallow Git history authority"),
            ("config.worktree", "per-worktree Git config authority"),
            ("commondir", "separate Git common-directory authority"),
            ("gitdir", "linked-worktree Git backpointer authority"),
        ):
            _require_absent_git_authority(
                git_descriptor, authority_name, role=role,
            )

        for lock_name in (
            "config.lock", "index.lock", "HEAD.lock", "packed-refs.lock",
        ):
            _require_absent_entry(
                git_descriptor,
                lock_name,
                role=f"Git transaction lock {lock_name}",
            )

        config_descriptor, config_info = _open_owned_regular_at(
            git_descriptor, "config", role="Git local config",
        )
        descriptors.append(config_descriptor)
        _require_device(
            config_info,
            git_info.st_dev,
            role="Git local config",
            code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
        )
        entries.append((git_descriptor, "config", config_info, "Git local config"))
        opened.append((config_descriptor, config_info, "Git local config"))
        _read_open_regular_bounded(
            config_descriptor,
            config_info,
            role="Git local config",
            max_size=GIT_CONFIG_MAX_SIZE_BYTES,
        )

        index_descriptor, index_info = _open_owned_regular_at(
            git_descriptor, "index", role="Git index",
        )
        descriptors.append(index_descriptor)
        _require_device(
            index_info,
            git_info.st_dev,
            role="Git index",
            code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
        )
        _validate_observed_size_ceiling(
            index_info,
            role="Git index",
            max_size=GIT_INDEX_MAX_SIZE_BYTES,
        )
        _validate_read_bounds(
            index_info,
            role="Git index",
            expected_size=index_info.st_size,
            max_size=GIT_INDEX_MAX_SIZE_BYTES,
        )
        entries.append((git_descriptor, "index", index_info, "Git index"))
        opened.append((index_descriptor, index_info, "Git index"))

        head_descriptor, head_info = _open_owned_regular_at(
            git_descriptor, "HEAD", role="Git HEAD",
        )
        descriptors.append(head_descriptor)
        _require_device(
            head_info,
            git_info.st_dev,
            role="Git HEAD",
            code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
        )
        entries.append((git_descriptor, "HEAD", head_info, "Git HEAD"))
        opened.append((head_descriptor, head_info, "Git HEAD"))
        head_raw = _read_open_regular_bounded(
            head_descriptor, head_info, role="Git HEAD", max_size=1024,
        )

        packed_raw: bytes | None = None
        if "packed-refs" in git_children:
            _require_git_descriptor_capacity(descriptors)
            packed_descriptor, packed_info = _open_owned_regular_at(
                git_descriptor,
                "packed-refs",
                role="Git packed references",
            )
            descriptors.append(packed_descriptor)
            _require_device(
                packed_info,
                git_info.st_dev,
                role="Git packed references",
                code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            )
            entries.append((
                git_descriptor,
                "packed-refs",
                packed_info,
                "Git packed references",
            ))
            opened.append((
                packed_descriptor,
                packed_info,
                "Git packed references",
            ))
            packed_raw = _read_open_regular_bounded(
                packed_descriptor,
                packed_info,
                role="Git packed references",
                max_size=16 * 1024 * 1024,
            )

        reference = _decode_symbolic_reference(head_raw, role="Git HEAD")
        if reference is None:
            _decode_full_oid(head_raw, role="detached HEAD")
        else:
            visited_references: set[str] = set()
            while True:
                if (
                    reference in visited_references
                    or len(visited_references) >= GIT_SYMBOLIC_REFERENCE_MAX_CHAIN
                ):
                    raise RunnabilityArchiveRefusal(
                        "RUNNABILITY_RESTORE_GIT_FAILURE",
                        f"cyclic or overlong Git symbolic-reference chain at {reference!r}",
                    )
                visited_references.add(reference)
                reference_path = PurePosixPath(reference)
                if len(reference_path.parts) > GIT_REFERENCE_MAX_COMPONENTS:
                    raise RunnabilityArchiveRefusal(
                        "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                        f"Git reference exceeds admitted path depth: {reference!r}",
                    )
                parent_descriptor = git_descriptor
                loose_reference_available = True
                for index, part in enumerate(reference_path.parts[:-1]):
                    role = "Git reference directory " + "/".join(
                        reference_path.parts[: index + 1]
                    )
                    try:
                        os.stat(part, dir_fd=parent_descriptor, follow_symlinks=False)
                    except FileNotFoundError:
                        loose_reference_available = False
                        break
                    except OSError as exc:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_FAILURE",
                            f"{role}: {exc}",
                        ) from exc
                    _require_git_descriptor_capacity(descriptors)
                    child_descriptor, child_info = _open_owned_directory_at(
                        parent_descriptor, part, role=role,
                    )
                    descriptors.append(child_descriptor)
                    _require_device(
                        child_info,
                        git_info.st_dev,
                        role=role,
                        code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                    )
                    entries.append((parent_descriptor, part, child_info, role))
                    opened.append((child_descriptor, child_info, role))
                    parent_descriptor = child_descriptor
                reference_name = reference_path.parts[-1]
                if loose_reference_available:
                    _require_absent_entry(
                        parent_descriptor,
                        f"{reference_name}.lock",
                        role=f"Git reference transaction lock {reference}.lock",
                    )
                    try:
                        os.stat(
                            reference_name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        loose_reference_available = False
                    except OSError as exc:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_FAILURE",
                            f"Git reference {reference!r}: {exc}",
                        ) from exc
                if not loose_reference_available:
                    if packed_raw is None:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_FAILURE",
                            f"Git reference {reference!r} is neither loose nor packed",
                        )
                    packed_oid: str | None = None
                    for line in packed_raw.splitlines():
                        raw_oid, separator, packed_reference = line.partition(b" ")
                        if (
                            separator == b" "
                            and packed_reference == os.fsencode(reference)
                        ):
                            packed_oid = _decode_full_oid(
                                raw_oid, role=f"packed Git reference {reference}",
                            )
                            break
                    if packed_oid is None:
                        raise RunnabilityArchiveRefusal(
                            "RUNNABILITY_RESTORE_GIT_FAILURE",
                            f"Git reference {reference!r} is neither loose nor packed",
                        )
                    break

                _require_git_descriptor_capacity(descriptors)
                reference_descriptor, reference_info = _open_owned_regular_at(
                    parent_descriptor,
                    reference_name,
                    role=f"Git reference {reference}",
                )
                descriptors.append(reference_descriptor)
                _require_device(
                    reference_info,
                    git_info.st_dev,
                    role=f"Git reference {reference}",
                    code="RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
                )
                entries.append((
                    parent_descriptor,
                    reference_name,
                    reference_info,
                    f"Git reference {reference}",
                ))
                opened.append((
                    reference_descriptor,
                    reference_info,
                    f"Git reference {reference}",
                ))
                reference_raw = _read_open_regular_bounded(
                    reference_descriptor,
                    reference_info,
                    role=f"Git reference {reference}",
                    max_size=1024,
                )
                next_reference = _decode_symbolic_reference(
                    reference_raw, role=f"Git reference {reference}",
                )
                if next_reference is None:
                    _decode_full_oid(reference_raw, role=f"Git reference {reference}")
                    break
                reference = next_reference

        return GitMetadataGuard(
            entries=tuple(entries),
            opened=tuple(opened),
            descriptors=tuple(descriptors),
            object_descriptor=objects_descriptor,
            object_device=objects_info.st_dev,
            object_inventory=object_inventory,
            reference_descriptor=references_descriptor,
            reference_device=references_info.st_dev,
            reference_inventory=reference_inventory,
        )
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _validate_disposition(value: Any, *, role: str) -> None:
    if not isinstance(value, dict) or set(value) != set(DISPOSITION):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} disposition fields",
        )
    if (
        type(value["authoritative"]) is not bool
        or value["authoritative"] is not False
        or type(value["resumable"]) is not bool
        or value["resumable"] is not False
        or type(value["reusable"]) is not bool
        or value["reusable"] is not False
        or not _is_exact_int(value["campaign_rows"], 0)
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} disposition constants",
        )


def _validate_attempt_identity(contract: FixedArchiveContract) -> None:
    expected = canonical_sha256({
        "source_commit": contract.source_commit,
        "source_repository_path": contract.source_relative_path.as_posix(),
        "source_sha256": contract.source_sha256,
    })
    if contract.attempt_id != expected:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "attempt ID does not rederive from source identity",
        )


def _parse_porcelain_v1_z(
    raw: bytes,
) -> list[tuple[bytes, bytes, bytes | None]]:
    if not raw or not raw.endswith(b"\0"):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "manifest Git status is not a complete NUL-delimited stream",
        )
    fields = raw[:-1].split(b"\0")
    records: list[tuple[bytes, bytes, bytes | None]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        if len(field) < 4 or field[2:3] != b" ":
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "manifest Git status contains a malformed primary record",
            )
        status = field[:2]
        path = field[3:]
        if not path or status not in PORCELAIN_V1_STATUS_COMBINATIONS:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "manifest Git status contains an invalid status/path",
            )
        index += 1
        continuation: bytes | None = None
        if b"R" in status or b"C" in status:
            if index >= len(fields) or not fields[index]:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_ARCHIVE_INCONSISTENT",
                    "manifest Git rename/copy record lacks its continuation path",
                )
            continuation = fields[index]
            index += 1
        records.append((status, path, continuation))
    return records


def _validate_manifest(
    value: Mapping[str, Any], *, contract: FixedArchiveContract,
) -> None:
    _validate_attempt_identity(contract)
    if set(value) != MANIFEST_FIELDS:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest fields are not closed",
        )
    if (
        not _is_exact_int(value.get("schema_version"), 1)
        or type(value.get("artifact_id")) is not str
        or value["artifact_id"] != ARTIFACT_ID
        or value.get("attempt_id") != contract.attempt_id
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest identity constants",
        )
    _parse_utc(value.get("created_at_utc"), expected=contract.created_at_utc)
    _validate_disposition(value.get("disposition"), role="manifest")
    body = {key: item for key, item in value.items() if key != "manifest_seal_sha256"}
    if value.get("manifest_seal_sha256") != canonical_sha256(body):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest seal mismatch",
        )
    source = value.get("source")
    canonical = value.get("canonical")
    payload = value.get("payload")
    if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest source fields",
        )
    if not isinstance(canonical, dict) or set(canonical) != CANONICAL_FIELDS:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest canonical fields",
        )
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest payload fields",
        )
    relative = contract.source_relative_path.as_posix()
    expected_source = {
        "repository_path": relative,
        "absolute_path": os.fspath(
            contract.recorded_source_absolute_path.absolute()
        ),
        "commit": contract.source_commit,
        "size_bytes": contract.source_size_bytes,
        "sha256": contract.source_sha256,
    }
    if any(
        key not in source
        or type(source[key]) is not type(expected)
        or source[key] != expected
        for key, expected in expected_source.items()
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest source identity drift",
        )
    if (
        not _exact_mapping_equal(canonical, {
            "repository_path": relative,
            "commit": contract.source_commit,
            "size_bytes": contract.canonical_size_bytes,
            "sha256": contract.canonical_sha256,
        })
        or not _exact_mapping_equal(payload, {
            "archive_relative_path": f"{PAYLOAD_DIRECTORY}/{PAYLOAD_NAME}",
            "size_bytes": contract.source_size_bytes,
            "sha256": contract.source_sha256,
        })
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "canonical/payload identity drift",
        )
    for field in (
        "git_status_sha256", "git_worktree_diff_sha256",
        "git_index_diff_sha256",
    ):
        if not _is_sha(source.get(field)):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", f"manifest {field}",
            )
    if source["git_index_diff_sha256"] != EMPTY_SHA256:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "source had staged changes",
        )
    if source["git_worktree_diff_sha256"] == EMPTY_SHA256:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "modified source cannot have an empty worktree diff digest",
        )
    try:
        status_raw = base64.b64decode(
            source.get("git_status_porcelain_v1_z_base64"), validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest status encoding",
        ) from exc
    if sha256_bytes(status_raw) != source["git_status_sha256"]:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest status hash mismatch",
        )
    expected_path = os.fsencode(relative)
    status_records = _parse_porcelain_v1_z(status_raw)
    matching_statuses = [
        status
        for status, path, _continuation in status_records
        if path == expected_path
    ]
    continuation_collision = any(
        continuation == expected_path
        for _status, _path, continuation in status_records
    )
    if matching_statuses != [b" M"] or continuation_collision:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "source path is not exactly one ordinary worktree-only modification",
        )
    if any(
        status not in {b"??", b"!!"} and status[:1] != b" "
        for status, _path, _continuation in status_records
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "manifest status records contradict the empty index-diff digest",
        )


def _validate_ledger_row(
    row: Mapping[str, Any], *, contract: FixedArchiveContract,
) -> None:
    _validate_attempt_identity(contract)
    if set(row) != LEDGER_FIELDS:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "ledger fields are not closed",
        )
    if (
        not _is_exact_int(row.get("schema_version"), 1)
        or row.get("entry_type") != ENTRY_TYPE
        or row.get("attempt_id") != contract.attempt_id
        or row.get("created_at_utc") != contract.created_at_utc
        or row.get("manifest_sha256") != contract.manifest_sha256
        or row.get("source_commit") != contract.source_commit
        or row.get("source_sha256") != contract.source_sha256
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "ledger identity constants",
        )
    _validate_disposition(
        {key: row.get(key) for key in DISPOSITION}, role="ledger",
    )
    recorded_namespace = contract.recorded_archive_base / ARCHIVE_ROOT_NAME
    recorded_archive = recorded_namespace / f"attempt-{contract.attempt_id}"
    if (
        row.get("archive_path") != os.fspath(recorded_archive)
        or row.get("manifest_path") != os.fspath(recorded_archive / MANIFEST_NAME)
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", "ledger recorded path drift",
        )


def _read_contract_schema(contract: FixedArchiveContract) -> bytes:
    source_path = contract.recorded_source_absolute_path.absolute()
    repo_root = source_path
    for _part in contract.source_relative_path.parts:
        repo_root = repo_root.parent
    if (
        repo_root / Path(*contract.source_relative_path.parts)
    ).absolute() != source_path:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "recorded source path does not bind the schema repository root",
        )
    if not _is_sha(contract.source_commit, 40):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "source commit is not a full SHA-1 object ID",
        )

    repo_descriptor, repo_info = _open_absolute_directory(
        repo_root, role="historical schema repository root",
    )
    git_guard: GitMetadataGuard | None = None
    try:
        git_guard = _open_git_metadata_guard(repo_descriptor)
        _validate_git_topology(repo_root, repo_descriptor)
        resolved_source_commit = _decode_full_oid(
            _git(
                repo_root,
                ["rev-parse", "--verify", f"{contract.source_commit}^{{commit}}"],
                repo_descriptor=repo_descriptor,
            ).stdout,
            role="archive source commit",
        )
        if resolved_source_commit != contract.source_commit:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "source commit identity does not resolve to itself as a commit",
            )
        canonical_entry = _tree_entry(
            repo_root,
            contract.source_commit,
            contract.source_relative_path.as_posix(),
            repo_descriptor=repo_descriptor,
            refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
        )
        expected_canonical_entry = GitTreeEntry(
            contract.canonical_tree_mode, "blob", contract.canonical_blob_oid,
        )
        if (
            contract.canonical_tree_mode != "100644"
            or not _is_sha(contract.canonical_blob_oid, 40)
            or canonical_entry != expected_canonical_entry
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "source-commit canonical tree entry drift",
            )
        canonical_raw = _git_blob(
            repo_root,
            canonical_entry.object_id,
            repo_descriptor=repo_descriptor,
            expected_size=contract.canonical_size_bytes,
            max_size=CANONICAL_MAX_SIZE_BYTES,
            refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
        )
        if (
            sha256_bytes(canonical_raw) != contract.canonical_sha256
            or _git_blob_oid(canonical_raw) != contract.canonical_blob_oid
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "source-commit canonical byte hash or blob identity mismatch",
            )
        schema_entry = _tree_entry(
            repo_root,
            contract.source_commit,
            contract.schema_relative_path.as_posix(),
            repo_descriptor=repo_descriptor,
            refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
        )
        if schema_entry.mode != "100644" or schema_entry.object_type != "blob":
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "source-commit runnability schema is not a regular blob",
            )
        schema_raw = _git_blob(
            repo_root,
            schema_entry.object_id,
            repo_descriptor=repo_descriptor,
            max_size=SCHEMA_MAX_SIZE_BYTES,
            refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
        )
        if (
            sha256_bytes(schema_raw) != contract.schema_sha256
            or _git_blob_oid(schema_raw) != schema_entry.object_id
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "source-commit runnability schema byte hash or blob identity mismatch",
            )
        _validate_git_topology(repo_root, repo_descriptor)
        _assert_opened_unchanged(
            repo_descriptor, repo_info, role="historical schema repository root",
        )
        _assert_absolute_directory_unchanged(
            repo_root, repo_info, role="historical schema repository root",
        )
        git_guard.verify()
        return schema_raw
    finally:
        if git_guard is not None:
            git_guard.close()
        os.close(repo_descriptor)


def _derive_cell_semantics(
    cells: list[Any],
    *,
    statuses: Sequence[str],
    role: str,
    refusal_code: str,
) -> tuple[dict[str, int], frozenset[tuple[str, str, str]]]:
    by_status = {status: 0 for status in statuses}
    identities: set[tuple[str, str, str]] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"{role} contains a non-object cell",
            )
        identity_values = tuple(
            cell.get(key) for key in ("slug", "category", "language")
        )
        if not all(type(item) is str and item for item in identity_values):
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"{role} contains an invalid cell identity",
            )
        identity = (
            identity_values[0], identity_values[1], identity_values[2],
        )
        if identity in identities:
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"{role} duplicate cell identity: {identity!r}",
            )
        identities.add(identity)
        status = cell.get("status")
        if type(status) is not str or status not in by_status:
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"{role} unknown status: {status!r}",
            )
        by_status[status] += 1
    return by_status, frozenset(identities)


def _validate_partial_payload(
    raw: bytes, schema_raw: bytes,
) -> frozenset[tuple[str, str, str]]:
    value = _load_json_bytes(raw, role="interrupted runnability payload")
    schema = _load_json_bytes(schema_raw, role="hash-pinned runnability schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "hash-pinned runnability schema has no property authority",
        )
    retained_extra_fields = set(value) - set(properties)
    if retained_extra_fields != PARTIAL_RETAINED_UNSCHEMATIZED_FIELDS:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "interrupted payload unschematized root-field set drift: "
            f"missing={sorted(PARTIAL_RETAINED_UNSCHEMATIZED_FIELDS - retained_extra_fields)}; "
            f"extra={sorted(retained_extra_fields - PARTIAL_RETAINED_UNSCHEMATIZED_FIELDS)}",
        )
    schema_projection = {
        key: item for key, item in value.items() if key in properties
    }
    _validate_with_jsonschema(
        schema_projection,
        schema,
        refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
        role="interrupted payload hash-pinned schema projection is invalid",
    )

    cells = value["cells"]
    summary = value["summary"]
    by_status, identities = _derive_cell_semantics(
        cells,
        statuses=PARTIAL_RUNNABILITY_STATUSES,
        role="interrupted payload",
        refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
    )

    if (
        not _is_exact_int(value.get("schema_version"), 1)
        or value.get("artifact_id") != "sm86_sol_runnability_v1"
        or value.get("arch") != "sm_86"
        or value.get("sku") != "rtx_3090_ti"
        or not _is_exact_int(value.get("in_scope_total"), 201)
        or not _is_exact_int(value.get("probed"), len(cells))
        or not _is_exact_int(
            value.get("pending"), value["in_scope_total"] - len(cells),
        )
        or len(cells) != 53
        or not _is_exact_int(summary.get("total"), len(cells))
        or not _is_exact_int(summary.get("runnable"), by_status["runnable"])
        or not _exact_mapping_equal(summary.get("by_status"), by_status)
        or by_status != dict(PARTIAL_EXPECTED_BY_STATUS)
        or value.get("status_to_typed_reason")
        != dict(PARTIAL_STATUS_TO_TYPED_REASON)
        or by_status["runnable"] != 48
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "interrupted payload 53/201 derived observation drift",
        )
    return identities


def _directory_names(
    descriptor: int, *, role: str, max_entries: int = GIT_METADATA_MAX_ENTRIES,
) -> set[str]:
    if type(max_entries) is not int or max_entries < 0:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            f"invalid {role} entry-count contract",
        )
    names: set[str] = set()
    try:
        with os.scandir(descriptor) as iterator:
            for entry in iterator:
                if len(names) >= max_entries:
                    raise RunnabilityArchiveRefusal(
                        "RUNNABILITY_ARCHIVE_INCONSISTENT",
                        f"{role} exceeds admitted entry count {max_entries}",
                    )
                names.add(entry.name)
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_INVALID", f"{role}: {exc}",
        ) from exc
    return names


def _assert_opened_unchanged(
    descriptor: int, expected: os.stat_result, *, role: str,
) -> None:
    observed = os.fstat(descriptor)
    if not _stat_matches(expected, observed):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} changed during verification",
        )


def _assert_entry_unchanged(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    role: str,
) -> None:
    try:
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role}: {exc}",
        ) from exc
    if not _stat_matches(expected, observed):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT", f"{role} changed after verification",
        )


def _assert_absolute_directory_unchanged(
    path: Path, expected: os.stat_result, *, role: str,
) -> None:
    descriptors, infos, names = _open_absolute_directory_chain(
        path, role=role, require_owned=True,
    )
    try:
        if not _stat_matches(expected, infos[-1]):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                f"{role} path changed during verification",
            )
        _verify_absolute_directory_chain(
            descriptors,
            infos,
            names,
            role=role,
            change_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _acquire_shared_archive_lock(
    namespace_descriptor: int, *, expected_device: int,
) -> tuple[int, os.stat_result]:
    descriptor, info = _open_owned_regular_at(
        namespace_descriptor, LOCK_NAME, role="archive lock",
    )
    try:
        _require_device(info, expected_device, role="archive lock")
        _require_exact_mode(info, 0o644, role="archive lock")
        if info.st_size != 0:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "archive lock content drift",
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_LOCKED",
                "archive publisher holds the exclusive lock",
            ) from exc
        return descriptor, info
    except Exception:
        os.close(descriptor)
        raise


def _verify_archive_contract(
    archive_base: Path,
    *,
    contract: FixedArchiveContract,
    schema_raw: bytes | None = None,
    canonical_raw: bytes | None = None,
    cohort_validator: Callable[
        [frozenset[tuple[str, str, str]]], None
    ] | None = None,
) -> dict[str, Any]:
    """Verify one explicit archive contract (private test seam)."""
    archive_base = archive_base.absolute()
    recorded_base = contract.recorded_archive_base.absolute()
    if archive_base != recorded_base:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_PATH_ALIAS_REQUIRED",
            f"recorded={recorded_base}; requested={archive_base}",
        )
    base_descriptor, base_info = _open_absolute_directory(
        archive_base, role="archive base",
    )
    namespace_descriptor: int | None = None
    attempt_descriptor: int | None = None
    payload_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        namespace_descriptor, namespace_info = _open_owned_directory_at(
            base_descriptor,
            ARCHIVE_ROOT_NAME,
            role="interrupted runnability namespace",
        )
        _require_device(
            namespace_info, base_info.st_dev, role="archive namespace",
        )
        _require_exact_mode(
            namespace_info, 0o700, role="archive namespace",
        )
        lock_descriptor, lock_info = _acquire_shared_archive_lock(
            namespace_descriptor, expected_device=base_info.st_dev,
        )
        expected_namespace = {
            LEDGER_NAME,
            LOCK_NAME,
            f"attempt-{contract.attempt_id}",
        }
        if _directory_names(
            namespace_descriptor,
            role="archive namespace",
            max_entries=len(expected_namespace),
        ) != expected_namespace:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "archive namespace child set drift",
            )
        attempt_name = f"attempt-{contract.attempt_id}"
        attempt_descriptor, attempt_info = _open_owned_directory_at(
            namespace_descriptor,
            attempt_name,
            role="interrupted runnability attempt",
        )
        _require_device(
            attempt_info, base_info.st_dev, role="archive attempt",
        )
        _require_exact_mode(attempt_info, 0o700, role="archive attempt")
        if _directory_names(
            attempt_descriptor, role="archive attempt", max_entries=2,
        ) != {MANIFEST_NAME, PAYLOAD_DIRECTORY}:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "attempt child set drift",
            )
        payload_descriptor, payload_root_info = _open_owned_directory_at(
            attempt_descriptor,
            PAYLOAD_DIRECTORY,
            role="interrupted payload root",
        )
        _require_device(
            payload_root_info, base_info.st_dev, role="archive payload root",
        )
        _require_exact_mode(
            payload_root_info, 0o700, role="archive payload root",
        )
        if _directory_names(
            payload_descriptor, role="payload root", max_entries=1,
        ) != {PAYLOAD_NAME}:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "payload child set drift",
            )
        ledger_raw, ledger_info = _read_regular_exact_at(
            namespace_descriptor,
            LEDGER_NAME,
            role="interrupted runnability ledger",
            expected_size=contract.ledger_size_bytes,
            max_size=LEDGER_MAX_SIZE_BYTES,
            expected_device=base_info.st_dev,
            expected_mode=0o600,
        )
        if sha256_bytes(ledger_raw) != contract.ledger_sha256:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "ledger byte hash mismatch",
            )
        if not ledger_raw.endswith(b"\n") or ledger_raw.count(b"\n") != 1:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "ledger cardinality drift",
            )
        ledger_row = _load_json_bytes(ledger_raw[:-1], role="ledger row")
        _validate_ledger_row(ledger_row, contract=contract)
        manifest_raw, manifest_info = _read_regular_exact_at(
            attempt_descriptor,
            MANIFEST_NAME,
            role="interrupted runnability manifest",
            expected_size=contract.manifest_size_bytes,
            max_size=MANIFEST_MAX_SIZE_BYTES,
            expected_device=base_info.st_dev,
            expected_mode=0o600,
        )
        if sha256_bytes(manifest_raw) != contract.manifest_sha256:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "manifest byte hash mismatch",
            )
        manifest = _load_json_bytes(manifest_raw, role="archive manifest")
        _validate_manifest(manifest, contract=contract)
        payload_raw, payload_info = _read_regular_exact_at(
            payload_descriptor,
            PAYLOAD_NAME,
            role="interrupted runnability payload",
            expected_size=contract.source_size_bytes,
            max_size=PAYLOAD_MAX_SIZE_BYTES,
            expected_device=base_info.st_dev,
            expected_mode=0o600,
        )
        if sha256_bytes(payload_raw) != contract.source_sha256:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT", "payload byte hash mismatch",
            )
        if (schema_raw is None) != (canonical_raw is None):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "private archive fixture must provide schema and canonical bytes together",
            )
        if schema_raw is None:
            schema_raw = _read_contract_schema(contract)
        else:
            if canonical_raw is None:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_ARCHIVE_INCONSISTENT",
                    "private canonical fixture is absent",
                )
            if sha256_bytes(schema_raw) != contract.schema_sha256:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_ARCHIVE_INCONSISTENT",
                    "private schema byte hash mismatch",
                )
            if (
                contract.canonical_tree_mode != "100644"
                or not _is_sha(contract.canonical_blob_oid, 40)
                or len(canonical_raw) != contract.canonical_size_bytes
                or sha256_bytes(canonical_raw) != contract.canonical_sha256
                or _git_blob_oid(canonical_raw) != contract.canonical_blob_oid
            ):
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_ARCHIVE_INCONSISTENT",
                    "private canonical fixture identity mismatch",
                )
        partial_identities = _validate_partial_payload(payload_raw, schema_raw)

        if _directory_names(
            namespace_descriptor,
            role="archive namespace final check",
            max_entries=len(expected_namespace),
        ) != expected_namespace:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "archive namespace changed during verification",
            )
        if _directory_names(
            attempt_descriptor, role="archive attempt final check", max_entries=2,
        ) != {MANIFEST_NAME, PAYLOAD_DIRECTORY}:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "archive attempt changed during verification",
            )
        if _directory_names(
            payload_descriptor, role="payload root final check", max_entries=1,
        ) != {PAYLOAD_NAME}:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "payload root changed during verification",
            )
        _assert_entry_unchanged(
            base_descriptor,
            ARCHIVE_ROOT_NAME,
            namespace_info,
            role="archive namespace",
        )
        _assert_entry_unchanged(
            namespace_descriptor, LOCK_NAME, lock_info, role="archive lock",
        )
        _assert_entry_unchanged(
            namespace_descriptor,
            LEDGER_NAME,
            ledger_info,
            role="interrupted runnability ledger",
        )
        _assert_entry_unchanged(
            namespace_descriptor,
            attempt_name,
            attempt_info,
            role="interrupted runnability attempt",
        )
        _assert_entry_unchanged(
            attempt_descriptor,
            MANIFEST_NAME,
            manifest_info,
            role="interrupted runnability manifest",
        )
        _assert_entry_unchanged(
            attempt_descriptor,
            PAYLOAD_DIRECTORY,
            payload_root_info,
            role="interrupted payload root",
        )
        _assert_entry_unchanged(
            payload_descriptor,
            PAYLOAD_NAME,
            payload_info,
            role="interrupted runnability payload",
        )
        for descriptor, expected, role in (
            (base_descriptor, base_info, "archive base"),
            (namespace_descriptor, namespace_info, "archive namespace"),
            (attempt_descriptor, attempt_info, "archive attempt"),
            (payload_descriptor, payload_root_info, "payload root"),
            (lock_descriptor, lock_info, "archive lock"),
        ):
            _assert_opened_unchanged(descriptor, expected, role=role)
        _assert_absolute_directory_unchanged(
            archive_base, base_info, role="archive base",
        )
        if cohort_validator is None:
            return manifest
        cohort_validator(partial_identities)

        if _directory_names(
            namespace_descriptor,
            role="archive namespace post-cohort check",
            max_entries=len(expected_namespace),
        ) != expected_namespace:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "archive namespace changed during cohort verification",
            )
        if _directory_names(
            attempt_descriptor,
            role="archive attempt post-cohort check",
            max_entries=2,
        ) != {MANIFEST_NAME, PAYLOAD_DIRECTORY}:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "archive attempt changed during cohort verification",
            )
        if _directory_names(
            payload_descriptor,
            role="payload root post-cohort check",
            max_entries=1,
        ) != {PAYLOAD_NAME}:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "payload root changed during cohort verification",
            )

        final_ledger_raw, final_ledger_info = _read_regular_exact_at(
            namespace_descriptor,
            LEDGER_NAME,
            role="interrupted runnability ledger post-cohort check",
            expected_size=contract.ledger_size_bytes,
            max_size=LEDGER_MAX_SIZE_BYTES,
            expected_device=base_info.st_dev,
            expected_mode=0o600,
        )
        if (
            final_ledger_raw != ledger_raw
            or sha256_bytes(final_ledger_raw) != contract.ledger_sha256
            or not _stat_matches(final_ledger_info, ledger_info)
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "ledger changed during cohort verification",
            )
        final_manifest_raw, final_manifest_info = _read_regular_exact_at(
            attempt_descriptor,
            MANIFEST_NAME,
            role="interrupted runnability manifest post-cohort check",
            expected_size=contract.manifest_size_bytes,
            max_size=MANIFEST_MAX_SIZE_BYTES,
            expected_device=base_info.st_dev,
            expected_mode=0o600,
        )
        if (
            final_manifest_raw != manifest_raw
            or sha256_bytes(final_manifest_raw) != contract.manifest_sha256
            or not _stat_matches(final_manifest_info, manifest_info)
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "manifest changed during cohort verification",
            )
        final_payload_raw, final_payload_info = _read_regular_exact_at(
            payload_descriptor,
            PAYLOAD_NAME,
            role="interrupted runnability payload post-cohort check",
            expected_size=contract.source_size_bytes,
            max_size=PAYLOAD_MAX_SIZE_BYTES,
            expected_device=base_info.st_dev,
            expected_mode=0o600,
        )
        if (
            final_payload_raw != payload_raw
            or sha256_bytes(final_payload_raw) != contract.source_sha256
            or not _stat_matches(final_payload_info, payload_info)
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "payload changed during cohort verification",
            )

        _assert_entry_unchanged(
            base_descriptor,
            ARCHIVE_ROOT_NAME,
            namespace_info,
            role="archive namespace",
        )
        _assert_entry_unchanged(
            namespace_descriptor, LOCK_NAME, lock_info, role="archive lock",
        )
        _assert_entry_unchanged(
            namespace_descriptor,
            LEDGER_NAME,
            ledger_info,
            role="interrupted runnability ledger",
        )
        _assert_entry_unchanged(
            namespace_descriptor,
            attempt_name,
            attempt_info,
            role="interrupted runnability attempt",
        )
        _assert_entry_unchanged(
            attempt_descriptor,
            MANIFEST_NAME,
            manifest_info,
            role="interrupted runnability manifest",
        )
        _assert_entry_unchanged(
            attempt_descriptor,
            PAYLOAD_DIRECTORY,
            payload_root_info,
            role="interrupted payload root",
        )
        _assert_entry_unchanged(
            payload_descriptor,
            PAYLOAD_NAME,
            payload_info,
            role="interrupted runnability payload",
        )
        for descriptor, expected, role in (
            (base_descriptor, base_info, "archive base"),
            (namespace_descriptor, namespace_info, "archive namespace"),
            (attempt_descriptor, attempt_info, "archive attempt"),
            (payload_descriptor, payload_root_info, "payload root"),
            (lock_descriptor, lock_info, "archive lock"),
        ):
            _assert_opened_unchanged(descriptor, expected, role=role)
        _assert_absolute_directory_unchanged(
            archive_base, base_info, role="archive base",
        )
        return manifest
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        if payload_descriptor is not None:
            os.close(payload_descriptor)
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if namespace_descriptor is not None:
            os.close(namespace_descriptor)
        os.close(base_descriptor)


def verify_fixed_archive(
    archive_base: Path = DEFAULT_ARCHIVE_BASE,
) -> dict[str, Any]:
    """Verify only the immutable production archive contract."""

    return _verify_archive_contract(archive_base, contract=FIXED_CONTRACT)


def _git_environment() -> dict[str, str]:
    # Git is an evidence reader here.  Never inherit loader, tracing, prompt,
    # configuration, or executable-search state from the invoking process.
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": "/tmp",
    }
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_GRAFT_FILE"] = os.devnull
    environment["GIT_NO_LAZY_FETCH"] = "1"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_ASKPASS"] = "/bin/false"
    return environment


def _git_command(arguments: Sequence[str]) -> list[str]:
    return [
        os.fspath(GIT_BINARY),
        "--no-replace-objects",
        "--no-lazy-fetch",
        "-c", "core.useReplaceRefs=false",
        "-c", "core.commitGraph=false",
        "-c", "core.multiPackIndex=false",
        "-c", "core.fsmonitor=false",
        "-c", "core.untrackedCache=false",
        "-c", "advice.graftFileDeprecated=false",
        "-c", "diff.external=false",
        *arguments,
    ]


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    repo_descriptor: int | None = None,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    cwd: Path | str = repo_root
    pass_fds: tuple[int, ...] = ()
    if repo_descriptor is not None:
        try:
            repo_info = os.fstat(repo_descriptor)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"repository descriptor is unavailable: {exc}",
            ) from exc
        if not stat.S_ISDIR(repo_info.st_mode):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                "repository descriptor is not a directory",
            )
        cwd = f"/proc/self/fd/{repo_descriptor}"
        pass_fds = (repo_descriptor,)
    try:
        result = subprocess.run(
            _git_command(arguments),
            cwd=cwd,
            env=_git_environment(),
            capture_output=True,
            check=False,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
            pass_fds=pass_fds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", str(exc),
        ) from exc
    if result.returncode not in allowed_returncodes:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            f"git {' '.join(arguments)} returned {result.returncode}: {detail}",
        )
    return result


def _git_config_values(
    repo_root: Path,
    repo_descriptor: int,
    key: str,
) -> list[str]:
    config_path = f"/proc/self/fd/{repo_descriptor}/.git/config"
    result = _git(
        repo_root,
        [
            "config", "--file", config_path, "--no-includes", "--null",
            "--get-all", key,
        ],
        repo_descriptor=repo_descriptor,
        allowed_returncodes=frozenset({0, 1}),
    )
    if result.returncode == 1:
        if result.stdout:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"Git config returned output for absent key {key!r}",
            )
        return []
    if not result.stdout.endswith(b"\0"):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            f"Git config returned malformed values for {key!r}",
        )
    try:
        return [
            raw.decode("utf-8", errors="strict")
            for raw in result.stdout[:-1].split(b"\0")
        ]
    except UnicodeError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            f"Git config value for {key!r} is not UTF-8: {exc}",
        ) from exc


def _validate_git_local_config(repo_root: Path, repo_descriptor: int) -> None:
    """Reject config authorities outside the fixed standard repository."""

    config_path = f"/proc/self/fd/{repo_descriptor}/.git/config"
    raw = _git(
        repo_root,
        [
            "config", "--file", config_path, "--no-includes", "--null",
            "--name-only", "--list",
        ],
        repo_descriptor=repo_descriptor,
    ).stdout
    if raw and not raw.endswith(b"\0"):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            "Git local config key inventory is malformed",
        )
    try:
        keys = {
            item.decode("utf-8", errors="strict").casefold()
            for item in raw[:-1].split(b"\0")
            if item
        }
    except UnicodeError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE",
            f"Git local config key is not UTF-8: {exc}",
        ) from exc
    if any(
        key.startswith("include.") or key.startswith("includeif.")
        for key in keys
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "Git local config includes external config authority",
        )
    if "core.worktree" in keys or "extensions.worktreeconfig" in keys:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "Git local config selects a separate worktree authority",
        )
    if (
        "extensions.partialclone" in keys
        or any(
            key.startswith("remote.")
            and (
                key.endswith(".promisor")
                or key.endswith(".partialclonefilter")
            )
            for key in keys
        )
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "Git local config selects a promisor or partial-clone authority",
        )
    repository_version = _git_config_values(
        repo_root, repo_descriptor, "core.repositoryFormatVersion",
    )
    if repository_version != ["0"]:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "Git repository format version is not the admitted version 0",
        )
    ref_storage = _git_config_values(
        repo_root, repo_descriptor, "extensions.refStorage",
    )
    if ref_storage not in ([], ["files"]):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "Git local config selects an external or unsupported ref storage payload",
        )


def _validate_git_topology(repo_root: Path, repo_descriptor: int) -> None:
    """Require the fixed standard gitdir/files-ref/SHA-1 authority."""

    _validate_git_local_config(repo_root, repo_descriptor)
    raw = _git(
        repo_root,
        [
            "rev-parse",
            "--path-format=absolute",
            "--show-toplevel",
            "--absolute-git-dir",
            "--git-common-dir",
            "--show-ref-format",
            "--show-object-format=storage",
        ],
        repo_descriptor=repo_descriptor,
    ).stdout
    expected_root = os.fsencode(repo_root)
    expected_git_dir = os.fsencode(repo_root / ".git")
    if raw.splitlines() != [
        expected_root, expected_git_dir, expected_git_dir, b"files", b"sha1",
    ]:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "expected one standard worktree/.git directory with files refs and SHA-1 objects",
        )


@dataclass(frozen=True)
class GitTreeEntry:
    mode: str
    object_type: str
    object_id: str


def _decode_full_oid(raw: bytes, *, role: str, length: int = 40) -> str:
    try:
        value = raw.strip().decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
        ) from exc
    if not _is_sha(value, length):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"invalid {role} object ID: {value!r}",
        )
    return value


def _head_oid(
    repo_root: Path, *, repo_descriptor: int | None = None,
) -> str:
    object_id = _decode_full_oid(
        _git(
            repo_root,
            ["rev-parse", "--verify", "HEAD"],
            repo_descriptor=repo_descriptor,
        ).stdout,
        role="HEAD",
    )
    object_type = _git(
        repo_root,
        ["cat-file", "-t", object_id],
        repo_descriptor=repo_descriptor,
    ).stdout
    if object_type != b"commit\n":
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_TOPOLOGY_UNSUPPORTED",
            "HEAD does not directly name a commit object",
        )
    return object_id


def _tree_entry(
    repo_root: Path,
    revision: str,
    relative: str,
    *,
    repo_descriptor: int | None = None,
    refusal_code: str = "RUNNABILITY_RESTORE_CANONICAL_INVALID",
) -> GitTreeEntry:
    raw = _git(
        repo_root,
        ["ls-tree", "-z", "--full-tree", revision, "--", relative],
        repo_descriptor=repo_descriptor,
    ).stdout
    records = raw.split(b"\0")
    if len(records) != 2 or records[-1] or not records[0]:
        raise RunnabilityArchiveRefusal(
            refusal_code,
            f"expected one tree entry for {relative}",
        )
    metadata, separator, raw_path = records[0].partition(b"\t")
    fields = metadata.split(b" ")
    if separator != b"\t" or len(fields) != 3 or raw_path != os.fsencode(relative):
        raise RunnabilityArchiveRefusal(
            refusal_code,
            f"malformed tree entry for {relative}",
        )
    try:
        mode, object_type, object_id = (
            field.decode("ascii", errors="strict") for field in fields
        )
    except UnicodeError as exc:
        raise RunnabilityArchiveRefusal(
            refusal_code,
            f"non-ASCII tree entry for {relative}: {exc}",
        ) from exc
    if (
        len(mode) != 6
        or not mode.isdigit()
        or object_type not in {"blob", "tree", "commit"}
        or not _is_sha(object_id, 40)
    ):
        raise RunnabilityArchiveRefusal(
            refusal_code,
            f"invalid tree entry for {relative}",
        )
    return GitTreeEntry(mode, object_type, object_id)


def _index_entry(
    repo_root: Path,
    relative: str,
    *,
    repo_descriptor: int | None = None,
) -> GitTreeEntry:
    raw = _git(
        repo_root,
        ["ls-files", "--stage", "-v", "-z", "--", relative],
        repo_descriptor=repo_descriptor,
    ).stdout
    records = raw.split(b"\0")
    if len(records) != 2 or records[-1] or not records[0]:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            f"expected one stage-zero index entry for {relative}",
        )
    metadata, separator, raw_path = records[0].partition(b"\t")
    fields = metadata.split(b" ")
    if (
        separator != b"\t"
        or len(fields) != 4
        or fields[0] != b"H"
        or fields[3] != b"0"
        or raw_path != os.fsencode(relative)
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            "canonical index entry has assume-unchanged, skip-worktree, "
            "or other flags, is unmerged, or is malformed",
        )
    try:
        mode = fields[1].decode("ascii", errors="strict")
        object_id = fields[2].decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            f"non-ASCII index entry for {relative}: {exc}",
        ) from exc
    if len(mode) != 6 or not mode.isdigit() or not _is_sha(object_id, 40):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            f"invalid index entry for {relative}",
        )
    return GitTreeEntry(mode, "blob", object_id)


def _remaining_git_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", "Git blob reader timed out",
        )
    return remaining


def _read_git_pipe_exact(
    descriptor: int,
    size: int,
    *,
    deadline: float,
    role: str,
    refusal_code: str,
) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        try:
            poller = select.poll()
            poller.register(
                descriptor, select.POLLIN | select.POLLHUP | select.POLLERR,
            )
            events = poller.poll(
                max(1, int(_remaining_git_timeout(deadline) * 1000) + 1),
            )
            if not events:
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_RESTORE_GIT_FAILURE",
                    f"{role} timed out",
                )
            if any(event & select.POLLNVAL for _fd, event in events):
                raise RunnabilityArchiveRefusal(
                    "RUNNABILITY_RESTORE_GIT_FAILURE",
                    f"{role} pipe descriptor became invalid",
                )
            chunk = os.read(descriptor, min(1 << 20, size - len(payload)))
        except RunnabilityArchiveRefusal:
            raise
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE", f"{role}: {exc}",
            ) from exc
        if not chunk:
            raise RunnabilityArchiveRefusal(
                refusal_code, f"{role} ended early",
            )
        payload.extend(chunk)
    return bytes(payload)


def _read_git_pipe_header(
    descriptor: int, *, deadline: float, object_id: str, refusal_code: str,
) -> bytes:
    header = bytearray()
    while len(header) < 256:
        byte = _read_git_pipe_exact(
            descriptor,
            1,
            deadline=deadline,
            role=f"Git blob {object_id} header",
            refusal_code=refusal_code,
        )
        header.extend(byte)
        if byte == b"\n":
            return bytes(header)
    raise RunnabilityArchiveRefusal(
        refusal_code, f"malformed Git blob header for {object_id}",
    )


def _git_blob(
    repo_root: Path,
    object_id: str,
    *,
    repo_descriptor: int | None = None,
    expected_size: int | None = None,
    max_size: int,
    refusal_code: str = "RUNNABILITY_RESTORE_CANONICAL_INVALID",
) -> bytes:
    if (
        not _is_sha(object_id, 40)
        or type(max_size) is not int
        or max_size < 0
        or (
            expected_size is not None
            and (type(expected_size) is not int or expected_size < 0)
        )
    ):
        raise RunnabilityArchiveRefusal(
            refusal_code, "invalid Git blob read contract",
        )
    cwd: Path | str = repo_root
    pass_fds: tuple[int, ...] = ()
    if repo_descriptor is not None:
        try:
            repo_info = os.fstat(repo_descriptor)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"repository descriptor is unavailable: {exc}",
            ) from exc
        if not stat.S_ISDIR(repo_info.st_mode):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                "repository descriptor is not a directory",
            )
        cwd = f"/proc/self/fd/{repo_descriptor}"
        pass_fds = (repo_descriptor,)

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            _git_command(["cat-file", "--batch"]),
            cwd=cwd,
            env=_git_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=pass_fds,
        )
        if process.stdin is None or process.stdout is None:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                "Git blob reader did not create its bounded pipes",
            )
        process.stdin.write(object_id.encode("ascii") + b"\n")
        process.stdin.close()
        process.stdin = None
        deadline = time.monotonic() + GIT_COMMAND_TIMEOUT_SECONDS
        stdout_descriptor = process.stdout.fileno()

        header = _read_git_pipe_header(
            stdout_descriptor,
            deadline=deadline,
            object_id=object_id,
            refusal_code=refusal_code,
        )
        if not header.endswith(b"\n"):
            raise RunnabilityArchiveRefusal(
                refusal_code, f"malformed Git blob header for {object_id}",
            )
        fields = header[:-1].split(b" ")
        if len(fields) != 3:
            raise RunnabilityArchiveRefusal(
                refusal_code, f"invalid Git blob header for {object_id}",
            )
        try:
            header_oid = fields[0].decode("ascii", errors="strict")
            object_type = fields[1].decode("ascii", errors="strict")
            size_text = fields[2].decode("ascii", errors="strict")
            if (
                header_oid != object_id
                or object_type != "blob"
                or not size_text
                or not size_text.isdigit()
                or len(size_text) > 20
            ):
                raise ValueError("unexpected object identity, type, or size")
            object_size = int(size_text)
        except (UnicodeError, ValueError) as exc:
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"invalid Git blob header for {object_id}: {exc}",
            ) from exc
        if object_size > max_size:
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"Git blob {object_id} exceeds maximum size: "
                f"{object_size} > {max_size}",
            )
        if expected_size is not None and object_size != expected_size:
            raise RunnabilityArchiveRefusal(
                refusal_code,
                f"Git blob {object_id} size mismatch: "
                f"{object_size} != {expected_size}",
            )

        payload = _read_git_pipe_exact(
            stdout_descriptor,
            object_size,
            deadline=deadline,
            role=f"Git blob {object_id}",
            refusal_code=refusal_code,
        )
        delimiter = _read_git_pipe_exact(
            stdout_descriptor,
            1,
            deadline=deadline,
            role=f"Git blob {object_id} delimiter",
            refusal_code=refusal_code,
        )
        if delimiter != b"\n":
            raise RunnabilityArchiveRefusal(
                refusal_code, f"Git blob {object_id} output framing drift",
            )
        return_code = process.wait(timeout=_remaining_git_timeout(deadline))
        if return_code != 0:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"Git blob reader returned {return_code}",
            )
        try:
            extra = os.read(stdout_descriptor, 1)
        except OSError as exc:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_FAILURE",
                f"Git blob reader final output check: {exc}",
            ) from exc
        if extra:
            raise RunnabilityArchiveRefusal(
                refusal_code, f"Git blob {object_id} output framing drift",
            )
        return payload
    except RunnabilityArchiveRefusal:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_GIT_FAILURE", f"Git blob reader: {exc}",
        ) from exc
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()


def _validate_live_canonical(
    raw: bytes,
    info: os.stat_result,
    canonical_raw: bytes,
    *,
    contract: FixedArchiveContract,
) -> None:
    if (
        info.st_size != contract.canonical_size_bytes
        or sha256_bytes(raw) != contract.canonical_sha256
        or raw != canonical_raw
        or info.st_mode & 0o111
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            "working-tree runnability bytes are not canonical v1 or executable mode drifted",
        )


def _open_owned_relative_directories(
    repo_descriptor: int,
    parts: Sequence[str],
    *,
    expected_device: int,
) -> tuple[int, os.stat_result]:
    descriptor = os.dup(repo_descriptor)
    info = os.fstat(descriptor)
    try:
        _require_device(info, expected_device, role="repository root")
        for part in parts:
            child, child_info = _open_owned_directory_at(
                descriptor,
                part,
                role=f"canonical source parent {part!r}",
            )
            try:
                _require_device(
                    child_info,
                    expected_device,
                    role=f"canonical source parent {part!r}",
                )
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
            info = child_info
        return descriptor, info
    except Exception:
        os.close(descriptor)
        raise


def _validate_canonical_document(
    canonical_raw: bytes,
    schema_raw: bytes,
) -> frozenset[tuple[str, str, str]]:
    document = _load_json_bytes(canonical_raw, role="canonical runnability v1")
    schema = _load_json_bytes(schema_raw, role="historical runnability v1 schema")
    _validate_with_jsonschema(
        document,
        schema,
        refusal_code="RUNNABILITY_RESTORE_CANONICAL_INVALID",
        role="canonical runnability schema validation failed",
    )
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            "canonical document cells are not a list",
        )
    by_status, identities = _derive_cell_semantics(
        cells,
        statuses=tuple(CANONICAL_EXPECTED_BY_STATUS),
        role="canonical document",
        refusal_code="RUNNABILITY_RESTORE_CANONICAL_INVALID",
    )
    summary = document.get("summary")
    expected_status_mapping = {
        status: PARTIAL_STATUS_TO_TYPED_REASON[status]
        for status in CANONICAL_EXPECTED_BY_STATUS
        if status != "runnable"
    }
    if (
        not _is_exact_int(document.get("schema_version"), 1)
        or document.get("artifact_id") != "sm86_sol_runnability_v1"
        or document.get("arch") != "sm_86"
        or document.get("sku") != "rtx_3090_ti"
        or not _is_exact_int(document.get("in_scope_total"), 201)
        or not _is_exact_int(document.get("probed"), len(cells))
        or not _is_exact_int(document.get("pending"), 0)
        or len(cells) != 201
        or not isinstance(summary, dict)
        or not _is_exact_int(summary.get("total"), len(cells))
        or not _is_exact_int(summary.get("runnable"), by_status["runnable"])
        or not _exact_mapping_equal(summary.get("by_status"), by_status)
        or by_status != dict(CANONICAL_EXPECTED_BY_STATUS)
        or document.get("status_to_typed_reason") != expected_status_mapping
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_CANONICAL_INVALID",
            "canonical document is not the complete derived historical v1 authority",
        )
    return identities


@dataclass
class _PinnedCanonicalContract:
    repo_root: Path
    contract: FixedArchiveContract
    repo_descriptor: int
    repo_info: os.stat_result
    source_path: Path
    source_parent_descriptor: int
    source_parent_info: os.stat_result
    git_guard: GitMetadataGuard
    head: str
    expected_entry: GitTreeEntry
    canonical_raw: bytes
    schema_raw: bytes
    canonical_identities: frozenset[tuple[str, str, str]]
    live_info: os.stat_result
    finalized: bool = False
    closed: bool = False

    def verify(
        self,
        expected_partial_identities: frozenset[
            tuple[str, str, str]
        ] | None = None,
    ) -> dict[str, Any]:
        if self.closed or self.finalized:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "canonical verification guard lifecycle is invalid",
            )
        if expected_partial_identities is not None:
            _validate_cohort_identity_sets(
                expected_partial_identities, self.canonical_identities,
            )
        relative = self.contract.source_relative_path.as_posix()
        final_raw, final_info = _read_regular_exact_at(
            self.source_parent_descriptor,
            self.source_path.name,
            role="restored canonical runnability final check",
            expected_size=self.contract.canonical_size_bytes,
            max_size=CANONICAL_MAX_SIZE_BYTES,
            expected_device=self.repo_info.st_dev,
        )
        _validate_live_canonical(
            final_raw,
            final_info,
            self.canonical_raw,
            contract=self.contract,
        )
        if not _stat_matches(self.live_info, final_info):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
                "canonical source changed between verification reads",
            )
        final_index_entry = _index_entry(
            self.repo_root,
            relative,
            repo_descriptor=self.repo_descriptor,
        )
        if final_index_entry != self.expected_entry:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_CANONICAL_INVALID",
                "canonical path remains dirty or staged in the index",
            )
        if (
            _head_oid(
                self.repo_root, repo_descriptor=self.repo_descriptor,
            )
            != self.head
        ):
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_GIT_STATE_CHANGED",
                "HEAD changed during canonical verification",
            )
        _validate_git_topology(self.repo_root, self.repo_descriptor)
        _assert_entry_unchanged(
            self.source_parent_descriptor,
            self.source_path.name,
            final_info,
            role="restored canonical runnability",
        )
        _assert_opened_unchanged(
            self.source_parent_descriptor,
            self.source_parent_info,
            role="canonical source parent",
        )
        _assert_opened_unchanged(
            self.repo_descriptor,
            self.repo_info,
            role="repository root",
        )
        _assert_absolute_directory_unchanged(
            self.repo_root, self.repo_info, role="repository root",
        )
        _assert_absolute_directory_unchanged(
            self.source_path.parent,
            self.source_parent_info,
            role="canonical source parent",
        )
        result = {
            "source_commit": self.contract.source_commit,
            "canonical_sha256": self.contract.canonical_sha256,
            "canonical_size_bytes": self.contract.canonical_size_bytes,
            "restored": True,
        }
        self.git_guard.verify()
        self.finalized = True
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            os.close(self.source_parent_descriptor)
        finally:
            try:
                self.git_guard.close()
            finally:
                os.close(self.repo_descriptor)


def _open_pinned_canonical_contract(
    repo_root: Path,
    *,
    contract: FixedArchiveContract,
    source_authority_refusal_code: str = "RUNNABILITY_RESTORE_CANONICAL_INVALID",
) -> _PinnedCanonicalContract:
    repo_root = repo_root.absolute()
    relative = contract.source_relative_path.as_posix()
    schema_relative = contract.schema_relative_path.as_posix()
    source_path = repo_root / Path(*contract.source_relative_path.parts)
    recorded_source = contract.recorded_source_absolute_path.absolute()
    if source_path.absolute() != recorded_source:
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_RESTORE_PATH_ALIAS_REQUIRED",
            f"recorded={recorded_source}; requested={source_path.absolute()}",
        )
    if not _is_sha(contract.source_commit, 40):
        raise RunnabilityArchiveRefusal(
            source_authority_refusal_code,
            "source commit identity does not resolve to itself as a commit",
        )
    repo_descriptor, repo_info = _open_absolute_directory(
        repo_root, role="repository root",
    )
    source_parent_descriptor: int | None = None
    git_guard: GitMetadataGuard | None = None
    try:
        git_guard = _open_git_metadata_guard(repo_descriptor)
        _validate_git_topology(repo_root, repo_descriptor)
        source_parent_descriptor, source_parent_info = (
            _open_owned_relative_directories(
                repo_descriptor,
                contract.source_relative_path.parts[:-1],
                expected_device=repo_info.st_dev,
            )
        )
        resolved_source_commit = _decode_full_oid(
            _git(
                repo_root,
                ["rev-parse", "--verify", f"{contract.source_commit}^{{commit}}"],
                repo_descriptor=repo_descriptor,
            ).stdout,
            role="archive source commit",
        )
        if (
            not _is_sha(contract.source_commit, 40)
            or resolved_source_commit != contract.source_commit
        ):
            raise RunnabilityArchiveRefusal(
                source_authority_refusal_code,
                "source commit identity does not resolve to itself as a commit",
            )
        head = _head_oid(repo_root, repo_descriptor=repo_descriptor)
        expected_entry = GitTreeEntry(
            contract.canonical_tree_mode, "blob", contract.canonical_blob_oid,
        )
        if (
            contract.canonical_tree_mode != "100644"
            or not _is_sha(contract.canonical_blob_oid, 40)
        ):
            raise RunnabilityArchiveRefusal(
                source_authority_refusal_code,
                "canonical tree-entry contract is invalid",
            )
        if _git(
            repo_root,
            ["merge-base", "--is-ancestor", contract.source_commit, head],
            repo_descriptor=repo_descriptor,
            allowed_returncodes=frozenset({0, 1}),
        ).returncode != 0:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_CANONICAL_INVALID",
                "archival source commit is not an ancestor of pinned HEAD",
            )
        canonical_entry = _tree_entry(
            repo_root,
            contract.source_commit,
            relative,
            repo_descriptor=repo_descriptor,
            refusal_code=source_authority_refusal_code,
        )
        if canonical_entry != expected_entry:
            raise RunnabilityArchiveRefusal(
                source_authority_refusal_code,
                "source-commit canonical tree entry drift",
            )
        canonical_raw = _git_blob(
            repo_root,
            canonical_entry.object_id,
            repo_descriptor=repo_descriptor,
            expected_size=contract.canonical_size_bytes,
            max_size=CANONICAL_MAX_SIZE_BYTES,
            refusal_code=source_authority_refusal_code,
        )
        schema_entry = _tree_entry(
            repo_root,
            contract.source_commit,
            schema_relative,
            repo_descriptor=repo_descriptor,
            refusal_code=source_authority_refusal_code,
        )
        if schema_entry.mode != "100644" or schema_entry.object_type != "blob":
            raise RunnabilityArchiveRefusal(
                source_authority_refusal_code,
                "historical schema tree entry is not a regular blob",
            )
        schema_raw = _git_blob(
            repo_root,
            schema_entry.object_id,
            repo_descriptor=repo_descriptor,
            max_size=SCHEMA_MAX_SIZE_BYTES,
            refusal_code=source_authority_refusal_code,
        )
        if (
            len(canonical_raw) != contract.canonical_size_bytes
            or sha256_bytes(canonical_raw) != contract.canonical_sha256
            or _git_blob_oid(canonical_raw) != contract.canonical_blob_oid
            or sha256_bytes(schema_raw) != contract.schema_sha256
            or _git_blob_oid(schema_raw) != schema_entry.object_id
        ):
            raise RunnabilityArchiveRefusal(
                source_authority_refusal_code,
                "source-commit canonical bytes, schema hash, or schema blob identity drift",
            )
        canonical_identities = _validate_canonical_document(
            canonical_raw, schema_raw,
        )
        head_entry = _tree_entry(
            repo_root, head, relative, repo_descriptor=repo_descriptor,
        )
        if head_entry != expected_entry:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_CANONICAL_INVALID",
                "current pinned HEAD tree entry is not canonical v1",
            )
        initial_index_entry = _index_entry(
            repo_root, relative, repo_descriptor=repo_descriptor,
        )
        if initial_index_entry != expected_entry:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_RESTORE_CANONICAL_INVALID",
                "canonical path remains dirty or staged in the index",
            )
        live_raw, live_info = _read_regular_exact_at(
            source_parent_descriptor,
            source_path.name,
            role="restored canonical runnability",
            expected_size=contract.canonical_size_bytes,
            max_size=CANONICAL_MAX_SIZE_BYTES,
            expected_device=repo_info.st_dev,
        )
        _validate_live_canonical(
            live_raw, live_info, canonical_raw, contract=contract,
        )
        return _PinnedCanonicalContract(
            repo_root=repo_root,
            contract=contract,
            repo_descriptor=repo_descriptor,
            repo_info=repo_info,
            source_path=source_path,
            source_parent_descriptor=source_parent_descriptor,
            source_parent_info=source_parent_info,
            git_guard=git_guard,
            head=head,
            expected_entry=expected_entry,
            canonical_raw=canonical_raw,
            schema_raw=schema_raw,
            canonical_identities=canonical_identities,
            live_info=live_info,
        )
    except Exception:
        if source_parent_descriptor is not None:
            os.close(source_parent_descriptor)
        if git_guard is not None:
            git_guard.close()
        os.close(repo_descriptor)
        raise


def _check_canonical_contract(
    repo_root: Path,
    *,
    contract: FixedArchiveContract,
    expected_partial_identities: frozenset[
        tuple[str, str, str]
    ] | None = None,
) -> dict[str, Any]:
    """Verify one explicit canonical contract (private test seam)."""
    pinned = _open_pinned_canonical_contract(repo_root, contract=contract)
    try:
        return pinned.verify(expected_partial_identities)
    finally:
        pinned.close()


def check_canonical_restored(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify only the immutable production canonical-restoration contract."""

    return _check_canonical_contract(repo_root, contract=FIXED_CONTRACT)


def _validate_cohort_identity_sets(
    partial_identities: frozenset[tuple[str, str, str]],
    canonical_identities: frozenset[tuple[str, str, str]],
) -> None:
    if (
        len(partial_identities) != 53
        or len(canonical_identities) != 201
        or not partial_identities <= canonical_identities
    ):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "interrupted 53-cell identity set is not a subset of canonical v1",
        )


def _validate_cohort_identity_binding(
    partial_raw: bytes, canonical_raw: bytes,
) -> None:
    partial = _load_json_bytes(partial_raw, role="interrupted runnability payload")
    canonical = _load_json_bytes(canonical_raw, role="canonical runnability v1")
    partial_cells = partial.get("cells")
    canonical_cells = canonical.get("cells")
    if not isinstance(partial_cells, list) or not isinstance(canonical_cells, list):
        raise RunnabilityArchiveRefusal(
            "RUNNABILITY_ARCHIVE_INCONSISTENT",
            "cohort binding inputs do not contain cell lists",
        )
    _partial_status, partial_identities = _derive_cell_semantics(
        partial_cells,
        statuses=PARTIAL_RUNNABILITY_STATUSES,
        role="interrupted payload cohort",
        refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
    )
    _canonical_status, canonical_identities = _derive_cell_semantics(
        canonical_cells,
        statuses=tuple(CANONICAL_EXPECTED_BY_STATUS),
        role="canonical cohort",
        refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
    )
    _validate_cohort_identity_sets(partial_identities, canonical_identities)


def _verify_fixed_all(
    archive_base: Path, repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pinned = _open_pinned_canonical_contract(
        repo_root,
        contract=FIXED_CONTRACT,
        source_authority_refusal_code="RUNNABILITY_ARCHIVE_INCONSISTENT",
    )
    restored: dict[str, Any] | None = None

    def verify_restored_cohort(
        partial_identities: frozenset[tuple[str, str, str]],
    ) -> None:
        nonlocal restored
        restored = pinned.verify(partial_identities)

    try:
        manifest = _verify_archive_contract(
            archive_base,
            contract=FIXED_CONTRACT,
            schema_raw=pinned.schema_raw,
            canonical_raw=pinned.canonical_raw,
            cohort_validator=verify_restored_cohort,
        )
        if restored is None:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_INCONSISTENT",
                "aggregate archive/restoration verification did not run",
            )
        return manifest, restored
    finally:
        pinned.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("verify-archive", "check-restored", "verify-all"),
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--archive-base",
        type=Path,
        default=None,
        help="Recorded archive base; relocated roots require path-alias authority.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "verify-archive" and args.repo_root is not None:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_UNUSED_ARGUMENT",
                "verify-archive does not consume --repo-root",
            )
        if args.action == "check-restored" and args.archive_base is not None:
            raise RunnabilityArchiveRefusal(
                "RUNNABILITY_ARCHIVE_UNUSED_ARGUMENT",
                "check-restored does not consume --archive-base",
            )
        repo_root = args.repo_root or REPO_ROOT
        archive_base = args.archive_base or DEFAULT_ARCHIVE_BASE
        manifest: dict[str, Any] | None = None
        restored: dict[str, Any] | None = None
        cohort_identity_bound = False
        if args.action == "verify-archive":
            manifest = verify_fixed_archive(archive_base)
        elif args.action == "check-restored":
            restored = check_canonical_restored(repo_root)
        else:
            manifest, restored = _verify_fixed_all(archive_base, repo_root)
            cohort_identity_bound = True
    except (RunnabilityArchiveRefusal, OSError) as exc:
        if isinstance(exc, RunnabilityArchiveRefusal):
            rendered = f"{exc.code}: {exc.detail}"
        else:
            rendered = f"RUNNABILITY_ARCHIVE_IO_FAILURE: {exc}"
        print(rendered, file=sys.stderr)
        return 1
    output: dict[str, Any] = {"status": "verified"}
    if manifest is not None:
        output.update({
            "archive_verified": True,
            "attempt_id": manifest["attempt_id"],
            "source_sha256": manifest["source"]["sha256"],
            "payload_schema_validation": PARTIAL_SCHEMA_VALIDATION,
            **manifest["disposition"],
        })
    if restored is not None:
        output.update({
            "canonical_restored": True,
            "source_commit": restored["source_commit"],
            "canonical_sha256": restored["canonical_sha256"],
            "canonical_size_bytes": restored["canonical_size_bytes"],
        })
    if cohort_identity_bound:
        output["cohort_identity_bound"] = True
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_ROOT_NAME", "DEFAULT_ARCHIVE_BASE", "DISPOSITION",
    "FIXED_CONTRACT", "FixedArchiveContract", "RunnabilityArchiveRefusal",
    "SOURCE_RELATIVE_PATH", "check_canonical_restored", "sha256_bytes",
    "verify_fixed_archive",
]
