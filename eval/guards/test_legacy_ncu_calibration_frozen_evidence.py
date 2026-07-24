"""Default-on immutability gate for retired sm_86 NCU calibration evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
MANIFEST_PATH = (
    REPO_ROOT
    / "eval/guards/legacy_ncu_calibration_frozen_evidence_manifest.v1.json"
)
MANIFEST_VERSION = "legacy_ncu_calibration_frozen_evidence.v1"
MANIFEST_SHA256 = "f872e3099655d7ee8e193326ec7222e4367d4e3e3d154f99f3bac3b2fa7612ff"
BASE_DIRECTORY = "tools/sol_baselines_harness/calibration/cache_prediction"
BASELINE_PATH = f"{BASE_DIRECTORY}/sm86_baseline_predictions_v1.toml"
HISTORICAL_GIT_REVISION = "f10b2a9bd9e1aea654b52a555178c1c08f824d6c"
HISTORICAL_SOURCE_AUDIT_RECORD_COUNT = 11
HEX_SHA256 = frozenset("0123456789abcdef")
SAFE_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
FIELD_REFERENCE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])*\Z"
)

EXPECTED_ROOT_HASHES = {
    f"{BASE_DIRECTORY}/isotonic_params_sm86.toml": (
        "bd2160b13bf026b1868e8e4a8838c118b020a61991bdba6a0fcf221e7f149827"
    ),
    f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml": (
        "b99762fda7047b607a097958e0a23d9e432febe41b435728c12b9ce249bdbfcd"
    ),
    f"{BASE_DIRECTORY}/label_ledger_v1_full_partition.toml": (
        "f9962fd454fafa1036d2965387f35f39b7365c66fe153f5d8e646015bc920539"
    ),
    f"{BASE_DIRECTORY}/label_ledger_v1_sm86_expanded.toml": (
        "cbf05dd4a14bb732b2d1e3aca2e51176db69b46925e9ade4dd50664e19622b8c"
    ),
    f"{BASE_DIRECTORY}/label_ledger_v1_sm86_pilot_v1.toml": (
        "9bc5e96dbb7ad82b2a04b9f654e536cd10ed66de4276a756233f9fd1efb30c96"
    ),
    f"{BASE_DIRECTORY}/sm86_baseline_predictions_v1.toml": (
        "81c8f010e59ef66389435e9fa638b85cca504ce5dedc539afd192bbdefa72fd7"
    ),
    f"{BASE_DIRECTORY}/sm86_expanded_raw_predictions_v1.toml": (
        "5b65ebff747edcbf8bc6a9f96a25e64fa519463f0a6f4fbc27e88caa9eb2d6f0"
    ),
    f"{BASE_DIRECTORY}/sm86_geometry_sweep_scoreboard_v1.toml": (
        "da61134fda3826d2af6032650eb287295b4b620c5678b5f680993e60612a113c"
    ),
    f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml": (
        "a3be506dcd3f3aa49c732667226dd5067b5fb8aaf1879f7aada202925f81f1f3"
    ),
}
EXPECTED_CAPABILITY_LEDGER = {
    "path": f"{BASE_DIRECTORY}/sm86_predictor_capability_v1.toml",
    "sha256": "13433b0002a79b546a66f7ac18e9c0aeac2960539b6bbc7cdafd8d406568f061",
}
EXPECTED_DEPENDENCY_HASHES = {
    f"{BASE_DIRECTORY}/calibration_corpus.toml": (
        "db7a351423635b03881c07ca3bdf0211717ce1d403ce02271c312ce6db133b09"
    ),
    f"{BASE_DIRECTORY}/calibration_validation_split_v1.toml": (
        "8a53fb14357ece6da318c2c5a83b3733305dc58f980bbddfd000be917a436987"
    ),
    f"{BASE_DIRECTORY}/end_to_end_smoke_report.toml": (
        "2fee9a0eb5fbff6218c0cde6250836897e3dbeba0ca8340893743dc2c569924b"
    ),
    f"{BASE_DIRECTORY}/historical_label_split_v1_sm86_pilot_v1.toml": (
        "270a1f31054afe0b6f7b302acab04d4ffde5086e88a88d7b4148f6c717472377"
    ),
}
EXPECTED_DIRECTORY_COUNTS = {
    f"{BASE_DIRECTORY}/ncu_expanded_sidecars_sm86": 60,
    f"{BASE_DIRECTORY}/raw_pred_sidecars_sm86": 53,
    f"{BASE_DIRECTORY}/raw_pred_sweep_sidecars_sm86": 45,
}
STABLE_RETIREMENT_REASON = "legacy_ncu_calibration_retired_targets_not_features"


class FrozenEvidenceViolation(AssertionError):
    """Raised when the immutable manifest or any bound evidence drifts."""


def _fail(message: str) -> None:
    raise FrozenEvidenceViolation(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_SHA256)
    )


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{context} fields drifted; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"manifest contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_manifest(
    path: Path, *, expected_sha256: str | None, repo_root: Path
) -> dict[str, Any]:
    if not repo_root.is_absolute() or repo_root.is_symlink() or not repo_root.is_dir():
        _fail(f"manifest repository root is not a trusted directory: {repo_root}")
    if not path.is_absolute():
        _fail(f"frozen-evidence manifest path is not absolute: {path}")
    try:
        relative_path = path.relative_to(repo_root).as_posix()
    except ValueError:
        _fail(f"frozen-evidence manifest escapes its repository root: {path}")
    path = _physical_path(
        repo_root, relative_path, context="frozen-evidence manifest"
    )
    if not path.exists():
        _fail(f"frozen-evidence manifest is missing: {path}")
    if path.is_symlink() or not path.is_file():
        _fail(f"frozen-evidence manifest is not a regular file: {path}")
    raw = path.read_bytes()
    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            _fail(
                "immutable manifest hash drifted: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    try:
        text = raw.decode("utf-8")
        document = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"manifest is not strict UTF-8 JSON: {exc}")
    canonical = (
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        _fail("manifest bytes are not deterministic canonical JSON plus LF")
    return _validate_manifest(document)


def _validate_repo_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{context} must be a non-empty repository-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(SAFE_PATH_COMPONENT_RE.fullmatch(part) is None for part in path.parts)
    ):
        _fail(f"{context} is not a canonical safe repository-relative path: {value!r}")
    return value


def _validate_field_reference(value: Any, context: str) -> str:
    if not isinstance(value, str) or FIELD_REFERENCE_RE.fullmatch(value) is None:
        _fail(f"{context} is not a canonical field reference: {value!r}")
    return value


def _validate_hash_record(record: Any, context: str) -> tuple[str, str]:
    item = _exact_keys(record, {"path", "sha256"}, context)
    path = _validate_repo_relative_path(item["path"], f"{context}.path")
    sha256 = item["sha256"]
    if not _is_sha256(sha256):
        _fail(f"{context}.sha256 is not lowercase SHA-256")
    return path, sha256


def _validate_manifest(document: Any) -> dict[str, Any]:
    manifest = _exact_keys(
        document,
        {
            "base_directory",
            "capability_ledger",
            "directory_inventories",
            "hash_algorithm",
            "historical_git_dependencies",
            "manifest_version",
            "root_artifacts",
            "transitive_dependencies",
        },
        "manifest",
    )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        _fail(f"unknown manifest version: {manifest['manifest_version']!r}")
    if manifest["hash_algorithm"] != "sha256":
        _fail(f"unknown hash algorithm: {manifest['hash_algorithm']!r}")
    if manifest["base_directory"] != BASE_DIRECTORY:
        _fail(f"unknown evidence base directory: {manifest['base_directory']!r}")

    roots = manifest["root_artifacts"]
    if not isinstance(roots, list):
        _fail("manifest.root_artifacts must be an array")
    root_pairs = [
        _validate_hash_record(item, f"manifest.root_artifacts[{index}]")
        for index, item in enumerate(roots)
    ]
    if root_pairs != sorted(root_pairs):
        _fail("manifest.root_artifacts is not sorted by physical path")
    if len(root_pairs) != len({path for path, _ in root_pairs}):
        _fail("manifest.root_artifacts contains a duplicate physical path")
    if dict(root_pairs) != EXPECTED_ROOT_HASHES:
        _fail("the nine audited root artifact paths or hashes drifted")

    capability = _validate_hash_record(
        manifest["capability_ledger"], "manifest.capability_ledger"
    )
    if capability != (
        EXPECTED_CAPABILITY_LEDGER["path"],
        EXPECTED_CAPABILITY_LEDGER["sha256"],
    ):
        _fail("the audited capability-ledger path or hash drifted")

    root_paths = set(EXPECTED_ROOT_HASHES)
    historical = manifest["historical_git_dependencies"]
    if not isinstance(historical, list):
        _fail("manifest.historical_git_dependencies must be an array")
    historical_edges: list[tuple[str, str, str, str, str, str]] = []
    serialized_source_paths: list[str] = []
    repository_source_paths: list[str] = []
    for index, raw_dependency in enumerate(historical):
        context = f"manifest.historical_git_dependencies[{index}]"
        dependency = _exact_keys(
            raw_dependency,
            {
                "field",
                "referrer",
                "repository_path",
                "revision",
                "serialized_path",
                "sha256",
            },
            context,
        )
        referrer = _validate_repo_relative_path(
            dependency["referrer"], f"{context}.referrer"
        )
        if referrer != BASELINE_PATH:
            _fail(f"{context} names an unknown historical referrer: {referrer!r}")
        field = _validate_field_reference(dependency["field"], f"{context}.field")
        serialized_path = _validate_repo_relative_path(
            dependency["serialized_path"], f"{context}.serialized_path"
        )
        if not serialized_path.startswith("scripts/baselines_harness/"):
            _fail(f"{context} rewrote a serialized source path: {serialized_path!r}")
        repository_path = _validate_repo_relative_path(
            dependency["repository_path"], f"{context}.repository_path"
        )
        expected_repository_path = serialized_path.replace(
            "scripts/baselines_harness/", "tools/sol_baselines_harness/", 1
        )
        if repository_path != expected_repository_path:
            _fail(
                f"{context} has a non-canonical historical relocation: "
                f"expected {expected_repository_path!r}, got {repository_path!r}"
            )
        revision = dependency["revision"]
        if revision != HISTORICAL_GIT_REVISION:
            _fail(f"{context}.revision is not the full frozen historical revision")
        sha256 = dependency["sha256"]
        if not _is_sha256(sha256):
            _fail(f"{context}.sha256 is not lowercase SHA-256")
        historical_edges.append(
            (
                referrer,
                field,
                serialized_path,
                repository_path,
                revision,
                sha256,
            )
        )
        serialized_source_paths.append(serialized_path)
        repository_source_paths.append(repository_path)
    if len(historical_edges) != HISTORICAL_SOURCE_AUDIT_RECORD_COUNT:
        _fail(
            "manifest.historical_git_dependencies must classify exactly "
            f"{HISTORICAL_SOURCE_AUDIT_RECORD_COUNT} source audit records"
        )
    if historical_edges != sorted(historical_edges, key=lambda edge: edge[:2]):
        _fail(
            "manifest.historical_git_dependencies is not sorted by "
            "referrer/field edge"
        )
    if len(historical_edges) != len(set(historical_edges)):
        _fail("manifest.historical_git_dependencies contains a duplicate edge")
    if len(serialized_source_paths) != len(set(serialized_source_paths)):
        _fail(
            "manifest.historical_git_dependencies contains a duplicate "
            "serialized path"
        )
    if len(repository_source_paths) != len(set(repository_source_paths)):
        _fail(
            "manifest.historical_git_dependencies contains a duplicate "
            "repository path"
        )

    dependencies = manifest["transitive_dependencies"]
    if not isinstance(dependencies, list):
        _fail("manifest.transitive_dependencies must be an array")
    dependency_pairs: list[tuple[str, str]] = []
    dependency_paths: list[str] = []
    for index, raw_dependency in enumerate(dependencies):
        context = f"manifest.transitive_dependencies[{index}]"
        dependency = _exact_keys(
            raw_dependency,
            {"path", "referenced_by", "sha256"},
            context,
        )
        path = _validate_repo_relative_path(dependency["path"], f"{context}.path")
        sha256 = dependency["sha256"]
        if not _is_sha256(sha256):
            _fail(f"{context}.sha256 is not lowercase SHA-256")
        referenced_by = dependency["referenced_by"]
        if not isinstance(referenced_by, list) or not referenced_by:
            _fail(f"{context}.referenced_by must be a non-empty array")
        reference_edges: list[tuple[str, str, str]] = []
        for referrer_index, raw_referrer in enumerate(referenced_by):
            referrer_context = f"{context}.referenced_by[{referrer_index}]"
            if not isinstance(raw_referrer, dict):
                _fail(f"{referrer_context} must be an object")
            has_path = "path" in raw_referrer
            has_field = "field" in raw_referrer
            expected_keys = {"source", "path"} if has_path else {"source", "field"}
            if has_path == has_field:
                _fail(
                    f"{referrer_context} must identify exactly one path or field edge"
                )
            referrer = _exact_keys(raw_referrer, expected_keys, referrer_context)
            source = _validate_repo_relative_path(
                referrer["source"], f"{referrer_context}.source"
            )
            if source not in root_paths:
                _fail(f"{referrer_context} names an unknown referring root: {source!r}")
            if has_path:
                path_or_field = _validate_repo_relative_path(
                    referrer["path"], f"{referrer_context}.path"
                )
                if not path_or_field.startswith("scripts/baselines_harness/"):
                    _fail(
                        f"{referrer_context} rewrote a serialized legacy path: "
                        f"{path_or_field!r}"
                    )
                reference_edges.append((source, "path", path_or_field))
            else:
                path_or_field = _validate_field_reference(
                    referrer["field"], f"{referrer_context}.field"
                )
                reference_edges.append((source, "field", path_or_field))
        if reference_edges != sorted(set(reference_edges)):
            _fail(
                f"{context}.referenced_by must contain unique exact edges "
                "in lexical order"
            )
        dependency_pairs.append((path, sha256))
        dependency_paths.append(path)
    if dependency_paths != sorted(dependency_paths):
        _fail("manifest.transitive_dependencies is not sorted by physical path")
    if len(dependency_paths) != len(set(dependency_paths)):
        _fail("manifest.transitive_dependencies contains a duplicate physical path")
    if dict(dependency_pairs) != EXPECTED_DEPENDENCY_HASHES:
        _fail("the audited transitive dependency paths or hashes drifted")

    inventories = manifest["directory_inventories"]
    if not isinstance(inventories, list):
        _fail("manifest.directory_inventories must be an array")
    observed_counts: dict[str, int] = {}
    observed_directories: list[str] = []
    for index, raw_inventory in enumerate(inventories):
        context = f"manifest.directory_inventories[{index}]"
        inventory = _exact_keys(
            raw_inventory, {"directory", "file_count", "files"}, context
        )
        directory = _validate_repo_relative_path(
            inventory["directory"], f"{context}.directory"
        )
        count = inventory["file_count"]
        files = inventory["files"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            _fail(f"{context}.file_count must be a non-negative integer")
        if not isinstance(files, list):
            _fail(f"{context}.files must be an array")
        names: list[str] = []
        for file_index, raw_file in enumerate(files):
            file_context = f"{context}.files[{file_index}]"
            file_record = _exact_keys(raw_file, {"name", "sha256"}, file_context)
            name = _validate_repo_relative_path(
                file_record["name"], f"{file_context}.name"
            )
            if len(PurePosixPath(name).parts) != 1 or not name.endswith(".json"):
                _fail(f"{file_context}.name is not a plain JSON filename")
            if not _is_sha256(file_record["sha256"]):
                _fail(f"{file_context}.sha256 is not lowercase SHA-256")
            names.append(name)
        if names != sorted(set(names)):
            _fail(f"{context}.files must have unique names in lexical order")
        if count != len(files):
            _fail(f"{context}.file_count does not match its exact inventory")
        observed_counts[directory] = count
        observed_directories.append(directory)
    if observed_directories != sorted(observed_directories):
        _fail("manifest.directory_inventories is not sorted by physical path")
    if len(observed_directories) != len(set(observed_directories)):
        _fail("manifest.directory_inventories contains a duplicate physical path")
    if observed_counts != EXPECTED_DIRECTORY_COUNTS:
        _fail("the audited 60/53/45 sidecar inventory counts drifted")
    return manifest


def _physical_path(
    repo_root: Path, relative_path: str, *, context: str = "frozen evidence"
) -> Path:
    if not repo_root.is_absolute() or repo_root.is_symlink() or not repo_root.is_dir():
        _fail(f"{context} repository root is not a trusted directory: {repo_root}")
    canonical = _validate_repo_relative_path(relative_path, f"{context} path")
    current = repo_root
    for part in PurePosixPath(canonical).parts:
        current = current / part
        if current.is_symlink():
            _fail(f"{context} may not traverse a symlink: {relative_path}")
    return current


def _verify_file(repo_root: Path, relative_path: str, expected_sha256: str) -> None:
    path = _physical_path(repo_root, relative_path)
    if not path.exists():
        _fail(f"frozen evidence is missing: {relative_path}")
    if path.is_symlink() or not path.is_file():
        _fail(f"frozen evidence has an unknown non-regular type: {relative_path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        _fail(
            f"frozen evidence was tampered: {relative_path}; "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _all_single_file_records(manifest: dict[str, Any]) -> list[dict[str, str]]:
    return [
        *manifest["root_artifacts"],
        manifest["capability_ledger"],
        *[
            {"path": item["path"], "sha256": item["sha256"]}
            for item in manifest["transitive_dependencies"]
        ],
    ]


def _verify_evidence(manifest: dict[str, Any], repo_root: Path) -> None:
    for record in _all_single_file_records(manifest):
        _verify_file(repo_root, record["path"], record["sha256"])

    for inventory in manifest["directory_inventories"]:
        relative_directory = inventory["directory"]
        directory = _physical_path(repo_root, relative_directory)
        if not directory.exists():
            _fail(f"frozen evidence directory is missing: {relative_directory}")
        if directory.is_symlink() or not directory.is_dir():
            _fail(
                "frozen evidence directory has an unknown non-directory type: "
                f"{relative_directory}"
            )
        actual_names = {entry.name for entry in directory.iterdir()}
        expected_by_name = {
            record["name"]: record["sha256"] for record in inventory["files"]
        }
        expected_names = set(expected_by_name)
        if actual_names != expected_names:
            _fail(
                f"frozen evidence inventory drifted in {relative_directory}; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra_or_unknown={sorted(actual_names - expected_names)}"
            )
        for name, sha256 in expected_by_name.items():
            _verify_file(repo_root, f"{relative_directory}/{name}", sha256)


def _load_toml(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = _physical_path(repo_root, relative_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _collect_dependency_edges(
    value: Any,
    *,
    source_path: str,
    targets_by_sha256: dict[str, str],
    field_prefix: str = "",
) -> set[tuple[str, str, str, str, str]]:
    edges: set[tuple[str, str, str, str, str]] = set()
    if isinstance(value, dict):
        serialized_path = value.get("path")
        sha256 = value.get("sha256")
        if (
            isinstance(serialized_path, str)
            and isinstance(sha256, str)
            and sha256 in targets_by_sha256
        ):
            serialized_path = _validate_repo_relative_path(
                serialized_path, f"{source_path} embedded path"
            )
            edges.add(
                (
                    source_path,
                    "path",
                    serialized_path,
                    targets_by_sha256[sha256],
                    sha256,
                )
            )
        for key, child in value.items():
            child_field = f"{field_prefix}.{key}" if field_prefix else key
            if (
                key.endswith("_sha256")
                and isinstance(child, str)
                and child in targets_by_sha256
            ):
                paired_path = value.get(key.removesuffix("_sha256"))
                if isinstance(paired_path, str):
                    paired_path = _validate_repo_relative_path(
                        paired_path, f"{source_path} embedded path"
                    )
                    edges.add(
                        (
                            source_path,
                            "path",
                            paired_path,
                            targets_by_sha256[child],
                            child,
                        )
                    )
                else:
                    field = _validate_field_reference(
                        child_field, f"{source_path} embedded hash field"
                    )
                    edges.add(
                        (
                            source_path,
                            "field",
                            field,
                            targets_by_sha256[child],
                            child,
                        )
                    )
            if isinstance(child, (dict, list)):
                edges.update(
                    _collect_dependency_edges(
                        child,
                        source_path=source_path,
                        targets_by_sha256=targets_by_sha256,
                        field_prefix=child_field,
                    )
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_field = f"{field_prefix}[{index}]"
            edges.update(
                _collect_dependency_edges(
                    child,
                    source_path=source_path,
                    targets_by_sha256=targets_by_sha256,
                    field_prefix=child_field,
                )
            )
    return edges


def _source_audit_records(
    baseline: dict[str, Any], referrer: str
) -> list[tuple[str, str, str, str]]:
    records: list[tuple[str, str, str, str]] = []

    def visit(value: Any, field_prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_field = f"{field_prefix}.{key}" if field_prefix else key
                if key == "audit_hashes":
                    if not isinstance(child, list):
                        _fail(f"{referrer} {child_field} must be an audit-record array")
                    for index, raw_record in enumerate(child):
                        record_context = f"{referrer} {child_field}[{index}]"
                        record = _exact_keys(
                            raw_record, {"path", "sha256"}, record_context
                        )
                        serialized_path = _validate_repo_relative_path(
                            record["path"], f"{record_context}.path"
                        )
                        sha256 = record["sha256"]
                        if not _is_sha256(sha256):
                            _fail(f"{record_context}.sha256 is not lowercase SHA-256")
                        if serialized_path.endswith(".py"):
                            records.append(
                                (
                                    referrer,
                                    f"{child_field}[{index}]",
                                    serialized_path,
                                    sha256,
                                )
                            )
                if isinstance(child, (dict, list)):
                    visit(child, child_field)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{field_prefix}[{index}]")

    visit(baseline)
    return records


def _verify_historical_source_classification(
    manifest: dict[str, Any], loaded_roots: dict[str, dict[str, Any]]
) -> None:
    actual_records = _source_audit_records(
        loaded_roots[BASELINE_PATH], BASELINE_PATH
    )
    if len(actual_records) != HISTORICAL_SOURCE_AUDIT_RECORD_COUNT:
        _fail(
            "frozen baseline source-audit count drifted: "
            f"expected {HISTORICAL_SOURCE_AUDIT_RECORD_COUNT}, "
            f"got {len(actual_records)}"
        )
    actual = Counter(actual_records)
    classified = Counter(
        (
            dependency["referrer"],
            dependency["field"],
            dependency["serialized_path"],
            dependency["sha256"],
        )
        for dependency in manifest["historical_git_dependencies"]
    )
    if actual != classified:
        missing = sorted((actual - classified).elements())
        extraneous = sorted((classified - actual).elements())
        _fail(
            "serialized source audit records were not classified exactly once; "
            f"missing={missing}, extraneous={extraneous}"
        )


def _sanitized_git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git_cat_file(
    repo_root: Path, arguments: list[str], *, descriptor: str
) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(repo_root),
            "cat-file",
            *arguments,
        ],
        check=False,
        capture_output=True,
        env=_sanitized_git_environment(),
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        _fail(f"historical Git object is unavailable: {descriptor}: {detail}")
    return completed.stdout


def _git_cat_file_blob(repo_root: Path, revision: str, repository_path: str) -> bytes:
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not set(revision).issubset(HEX_SHA256)
    ):
        _fail(f"historical Git revision is not a full lowercase object ID: {revision!r}")
    repository_path = _validate_repo_relative_path(
        repository_path, "historical Git repository path"
    )
    object_type = _git_cat_file(
        repo_root, ["-t", revision], descriptor=revision
    ).decode("ascii", errors="replace").strip()
    if object_type != "commit":
        _fail(f"historical Git revision is not a commit: {revision}: {object_type!r}")
    descriptor = f"{revision}:{repository_path}"
    return _git_cat_file(
        repo_root, ["blob", descriptor], descriptor=descriptor
    )


def _verify_historical_git_dependencies(
    manifest: dict[str, Any],
    repo_root: Path,
    loaded_roots: dict[str, dict[str, Any]],
) -> None:
    _verify_historical_source_classification(manifest, loaded_roots)
    for dependency in manifest["historical_git_dependencies"]:
        payload = _git_cat_file_blob(
            repo_root, dependency["revision"], dependency["repository_path"]
        )
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != dependency["sha256"]:
            _fail(
                "historical Git dependency hash drifted: "
                f"{dependency['revision']}:{dependency['repository_path']}; "
                f"expected {dependency['sha256']}, got {actual_sha256}"
            )


def _verify_embedded_dependency_closure(
    manifest: dict[str, Any], repo_root: Path
) -> dict[str, dict[str, Any]]:
    loaded_roots = {
        record["path"]: _load_toml(repo_root, record["path"])
        for record in manifest["root_artifacts"]
    }
    targets_by_sha256 = {
        dependency["sha256"]: dependency["path"]
        for dependency in manifest["transitive_dependencies"]
    }
    if len(targets_by_sha256) != len(manifest["transitive_dependencies"]):
        _fail("transitive dependency hashes are not unique")
    expected_edges: set[tuple[str, str, str, str, str]] = set()
    for dependency in manifest["transitive_dependencies"]:
        for referrer in dependency["referenced_by"]:
            edge_kind = "path" if "path" in referrer else "field"
            expected_edges.add(
                (
                    referrer["source"],
                    edge_kind,
                    referrer[edge_kind],
                    dependency["path"],
                    dependency["sha256"],
                )
            )
    actual_edges: set[tuple[str, str, str, str, str]] = set()
    for source_path, loaded_root in loaded_roots.items():
        actual_edges.update(
            _collect_dependency_edges(
                loaded_root,
                source_path=source_path,
                targets_by_sha256=targets_by_sha256,
            )
        )
    if actual_edges != expected_edges:
        missing = sorted(expected_edges - actual_edges)
        extraneous = sorted(actual_edges - expected_edges)
        _fail(
            "embedded dependency edge set drifted; "
            f"missing={missing}, extraneous={extraneous}"
        )

    hashes = {
        **EXPECTED_ROOT_HASHES,
        EXPECTED_CAPABILITY_LEDGER["path"]: EXPECTED_CAPABILITY_LEDGER["sha256"],
        **EXPECTED_DEPENDENCY_HASHES,
    }
    bindings = [
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86.toml",
            "ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_pilot_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86.toml",
            "validation_split_sha256",
            f"{BASE_DIRECTORY}/historical_label_split_v1_sm86_pilot_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86.toml",
            "historical_label_split_sha256",
            f"{BASE_DIRECTORY}/historical_label_split_v1_sm86_pilot_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_expanded.toml",
            "seed_ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_pilot_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_expanded_raw_predictions_v1.toml",
            "expanded_ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_expanded.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_geometry_sweep_scoreboard_v1.toml",
            "capability_ledger_sha256",
            f"{BASE_DIRECTORY}/sm86_predictor_capability_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml",
            "expanded_ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_expanded.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml",
            "raw_predictions_sha256",
            f"{BASE_DIRECTORY}/sm86_expanded_raw_predictions_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml",
            "seed_ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_pilot_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml",
            "capability_ledger_sha256",
            f"{BASE_DIRECTORY}/sm86_predictor_capability_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml",
            "sweep_scoreboard_sha256",
            f"{BASE_DIRECTORY}/sm86_geometry_sweep_scoreboard_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            "isotonic_params_sha256",
            f"{BASE_DIRECTORY}/isotonic_params_sm86_expanded.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            "expanded_ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_expanded.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            "raw_predictions_sha256",
            f"{BASE_DIRECTORY}/sm86_expanded_raw_predictions_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            "seed_ledger_sha256",
            f"{BASE_DIRECTORY}/label_ledger_v1_sm86_pilot_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            "capability_ledger_sha256",
            f"{BASE_DIRECTORY}/sm86_predictor_capability_v1.toml",
        ),
        (
            f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            "sweep_scoreboard_sha256",
            f"{BASE_DIRECTORY}/sm86_geometry_sweep_scoreboard_v1.toml",
        ),
    ]
    for source_path, meta_field, target_path in bindings:
        actual = loaded_roots[source_path].get("meta", {}).get(meta_field)
        expected = hashes[target_path]
        if actual != expected:
            _fail(
                f"embedded hash binding drifted: {source_path} meta.{meta_field}; "
                f"expected {expected}, got {actual!r}"
            )

    baseline = loaded_roots[f"{BASE_DIRECTORY}/sm86_baseline_predictions_v1.toml"]
    baseline_meta = baseline.get("meta", {})
    if baseline_meta.get("source_smoke_report") != (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "end_to_end_smoke_report.toml"
    ) or baseline_meta.get("source_smoke_report_sha256") != hashes[
        f"{BASE_DIRECTORY}/end_to_end_smoke_report.toml"
    ]:
        _fail("baseline smoke-report serialized path/hash binding drifted")
    if baseline_meta.get("source_ledger") != (
        "scripts/baselines_harness/calibration/cache_prediction/"
        "label_ledger_v1_full_partition.toml"
    ) or baseline_meta.get("source_ledger_sha256") != hashes[
        f"{BASE_DIRECTORY}/label_ledger_v1_full_partition.toml"
    ]:
        _fail("baseline source-ledger serialized path/hash binding drifted")
    return loaded_roots


def _copy_manifest_evidence(
    manifest: dict[str, Any], source_root: Path, destination_root: Path
) -> None:
    for record in _all_single_file_records(manifest):
        source = source_root / record["path"]
        destination = destination_root / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for inventory in manifest["directory_inventories"]:
        directory = destination_root / inventory["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        for record in inventory["files"]:
            shutil.copyfile(
                source_root / inventory["directory"] / record["name"],
                directory / record["name"],
            )


def _import_retirement_contract():
    tools = str(TOOLS_ROOT)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )


@pytest.mark.campaign_evidence
def test_frozen_evidence_manifest_and_historical_bytes_are_exact() -> None:
    manifest = _load_manifest(
        MANIFEST_PATH, expected_sha256=MANIFEST_SHA256, repo_root=REPO_ROOT
    )
    _verify_evidence(manifest, REPO_ROOT)
    loaded_roots = _verify_embedded_dependency_closure(manifest, REPO_ROOT)
    _verify_historical_git_dependencies(manifest, REPO_ROOT, loaded_roots)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "is missing"),
        ("extra", "extra_or_unknown"),
        ("tampered", "was tampered"),
        ("unknown_type", "unknown non-regular type"),
    ],
)
def test_frozen_evidence_gate_fails_closed_on_evidence_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest = _load_manifest(
        MANIFEST_PATH, expected_sha256=MANIFEST_SHA256, repo_root=REPO_ROOT
    )
    sandbox = tmp_path / "repo"
    _copy_manifest_evidence(manifest, REPO_ROOT, sandbox)
    first_inventory = manifest["directory_inventories"][0]
    first_sidecar = (
        sandbox
        / first_inventory["directory"]
        / first_inventory["files"][0]["name"]
    )
    if mutation == "missing":
        (sandbox / manifest["root_artifacts"][0]["path"]).unlink()
    elif mutation == "extra":
        (sandbox / first_inventory["directory"] / "unknown_sidecar.json").write_text(
            "{}\n", encoding="utf-8"
        )
    elif mutation == "tampered":
        first_sidecar.write_bytes(first_sidecar.read_bytes() + b"\n")
    elif mutation == "unknown_type":
        first_sidecar.unlink()
        first_sidecar.mkdir()
    else:  # pragma: no cover - the closed parameter list makes this unreachable.
        raise AssertionError(mutation)
    with pytest.raises(FrozenEvidenceViolation, match=message):
        _verify_evidence(manifest, sandbox)


@pytest.mark.parametrize("mutation", ["unknown_version", "missing_field", "unknown_field"])
def test_frozen_evidence_manifest_schema_fails_closed(mutation: str) -> None:
    manifest = _load_manifest(
        MANIFEST_PATH, expected_sha256=MANIFEST_SHA256, repo_root=REPO_ROOT
    )
    planted = copy.deepcopy(manifest)
    if mutation == "unknown_version":
        planted["manifest_version"] = "legacy_ncu_calibration_frozen_evidence.v2"
    elif mutation == "missing_field":
        del planted["hash_algorithm"]
    elif mutation == "unknown_field":
        planted["unrecognized_evidence"] = []
    else:  # pragma: no cover - the closed parameter list makes this unreachable.
        raise AssertionError(mutation)
    with pytest.raises(FrozenEvidenceViolation):
        _validate_manifest(planted)


@pytest.mark.parametrize("mutation", ["missing_referrer", "extraneous_referrer"])
def test_embedded_dependency_edges_reject_missing_or_extraneous_referrers(
    mutation: str,
) -> None:
    manifest = _load_manifest(
        MANIFEST_PATH, expected_sha256=MANIFEST_SHA256, repo_root=REPO_ROOT
    )
    planted = copy.deepcopy(manifest)
    dependency = next(
        item
        for item in planted["transitive_dependencies"]
        if item["path"].endswith("historical_label_split_v1_sm86_pilot_v1.toml")
    )
    if mutation == "missing_referrer":
        dependency["referenced_by"].pop()
    elif mutation == "extraneous_referrer":
        dependency["referenced_by"].append(
            {
                "path": (
                    "scripts/baselines_harness/calibration/cache_prediction/"
                    "historical_label_split_v1_sm86_pilot_v1.toml"
                ),
                "source": f"{BASE_DIRECTORY}/sm86_validation_report_v1.toml",
            }
        )
    else:  # pragma: no cover - the closed parameter list makes this unreachable.
        raise AssertionError(mutation)
    with pytest.raises(FrozenEvidenceViolation, match="dependency edge set drifted"):
        _verify_embedded_dependency_closure(planted, REPO_ROOT)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_historical_source_audit_records_are_classified_exactly_once(
    mutation: str,
) -> None:
    manifest = _load_manifest(
        MANIFEST_PATH, expected_sha256=MANIFEST_SHA256, repo_root=REPO_ROOT
    )
    loaded_roots = _verify_embedded_dependency_closure(manifest, REPO_ROOT)
    planted = copy.deepcopy(manifest)
    if mutation == "missing":
        planted["historical_git_dependencies"].pop()
    elif mutation == "duplicate":
        planted["historical_git_dependencies"].append(
            copy.deepcopy(planted["historical_git_dependencies"][0])
        )
    else:  # pragma: no cover - the closed parameter list makes this unreachable.
        raise AssertionError(mutation)
    with pytest.raises(FrozenEvidenceViolation, match="classified exactly once"):
        _verify_historical_source_classification(planted, loaded_roots)


@pytest.mark.parametrize("symlink_location", ["manifest", "ancestor"])
def test_frozen_evidence_manifest_symlinks_fail_closed(
    tmp_path: Path, symlink_location: str
) -> None:
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    real_manifest = real_directory / MANIFEST_PATH.name
    shutil.copyfile(MANIFEST_PATH, real_manifest)
    if symlink_location == "manifest":
        planted_manifest = sandbox / MANIFEST_PATH.name
        planted_manifest.symlink_to(real_manifest)
    elif symlink_location == "ancestor":
        linked_directory = sandbox / "linked"
        linked_directory.symlink_to(real_directory, target_is_directory=True)
        planted_manifest = linked_directory / MANIFEST_PATH.name
    else:  # pragma: no cover - the closed parameter list makes this unreachable.
        raise AssertionError(symlink_location)
    with pytest.raises(FrozenEvidenceViolation, match="symlink"):
        _load_manifest(
            planted_manifest,
            expected_sha256=MANIFEST_SHA256,
            repo_root=sandbox,
        )


@pytest.mark.parametrize("symlink_location", ["evidence", "ancestor"])
def test_frozen_evidence_symlinks_fail_closed(
    tmp_path: Path, symlink_location: str
) -> None:
    manifest = _load_manifest(
        MANIFEST_PATH, expected_sha256=MANIFEST_SHA256, repo_root=REPO_ROOT
    )
    record = manifest["root_artifacts"][0]
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    planted = sandbox / record["path"]
    if symlink_location == "evidence":
        planted.parent.mkdir(parents=True)
        planted.symlink_to(REPO_ROOT / record["path"])
    elif symlink_location == "ancestor":
        linked_ancestor = sandbox / PurePosixPath(record["path"]).parts[0]
        linked_ancestor.symlink_to(
            REPO_ROOT / PurePosixPath(record["path"]).parts[0],
            target_is_directory=True,
        )
    else:  # pragma: no cover - the closed parameter list makes this unreachable.
        raise AssertionError(symlink_location)
    with pytest.raises(FrozenEvidenceViolation, match="symlink"):
        _verify_file(sandbox, record["path"], record["sha256"])


def test_frozen_evidence_manifest_duplicate_json_keys_fail_closed(
    tmp_path: Path,
) -> None:
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    planted_manifest = sandbox / "duplicate.json"
    planted_manifest.write_bytes(
        b'{"manifest_version":"first","manifest_version":"second"}\n'
    )
    with pytest.raises(FrozenEvidenceViolation, match="duplicate JSON key"):
        _load_manifest(
            planted_manifest, expected_sha256=None, repo_root=sandbox
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        "/absolute/path",
        "../escape",
        "safe/../../escape",
        "safe//noncanonical",
        "safe/./noncanonical",
        "safe\\windows_escape",
        "C:/windows_absolute",
        "safe/control\x00byte",
        "safe/newline\npath",
        "safe/space path",
    ],
)
def test_unsafe_repository_paths_fail_closed(unsafe_path: str) -> None:
    with pytest.raises(FrozenEvidenceViolation, match="repository-relative path"):
        _validate_repo_relative_path(unsafe_path, "planted path")


def test_historical_git_reads_ignore_replace_refs_and_git_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "init", "--quiet", "--object-format=sha1", str(repository)],
        check=True,
        capture_output=True,
        env=_sanitized_git_environment(),
        timeout=30,
    )

    def git_object(arguments: list[str], payload: bytes | None = None) -> str:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repository), *arguments],
            input=payload,
            check=True,
            capture_output=True,
            env=_sanitized_git_environment(),
            timeout=30,
        )
        return completed.stdout.decode("ascii").strip()

    def commit_with_payload(payload: bytes, message: str) -> str:
        blob = git_object(["hash-object", "-w", "--stdin"], payload)
        tree = git_object(
            ["mktree"], f"100644 blob {blob}\tsource.py\n".encode("ascii")
        )
        return git_object(
            [
                "-c",
                "user.name=Frozen Evidence Test",
                "-c",
                "user.email=frozen-evidence@example.invalid",
                "commit-tree",
                tree,
                "-m",
                message,
            ]
        )

    original_payload = b"original historical bytes\n"
    replacement_payload = b"planted replacement bytes\n"
    original_commit = commit_with_payload(original_payload, "original")
    replacement_commit = commit_with_payload(replacement_payload, "replacement")
    git_object(["replace", original_commit, replacement_commit])

    unsanitized_environment = _sanitized_git_environment()
    unsanitized_environment.pop("GIT_NO_REPLACE_OBJECTS")
    virtual_read = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "cat-file",
            "blob",
            f"{original_commit}:source.py",
        ],
        check=True,
        capture_output=True,
        env=unsanitized_environment,
        timeout=30,
    ).stdout
    assert virtual_read == replacement_payload

    for variable in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    ):
        monkeypatch.setenv(variable, str(tmp_path / f"planted-{variable.lower()}"))
    assert _git_cat_file_blob(repository, original_commit, "source.py") == original_payload


def test_missing_frozen_evidence_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FrozenEvidenceViolation, match="manifest is missing"):
        _load_manifest(
            tmp_path / "missing.v1.json",
            expected_sha256=MANIFEST_SHA256,
            repo_root=tmp_path,
        )


def test_central_retirement_contract_import_is_safe(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(TOOLS_ROOT)!r})\n"
        "from sol_baselines_harness.calibration.cache_prediction.retirement import "
        "LEGACY_NCU_CALIBRATION_RETIRED, LegacyCalibrationRetiredError, "
        "refuse_legacy_ncu_calibration, retired_legacy_ncu_calibration\n"
        f"assert LEGACY_NCU_CALIBRATION_RETIRED == {STABLE_RETIREMENT_REASON!r}\n"
    )
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_central_retirement_error_uses_one_stable_reason() -> None:
    retirement = _import_retirement_contract()
    assert retirement.LEGACY_NCU_CALIBRATION_RETIRED == STABLE_RETIREMENT_REASON
    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        retirement.refuse_legacy_ncu_calibration("unit_test_surface")
    error = caught.value
    assert error.reason == STABLE_RETIREMENT_REASON
    assert error.surface == "unit_test_surface"
    assert str(error) == f"{STABLE_RETIREMENT_REASON}: unit_test_surface"


def test_central_retirement_guard_preserves_help_metadata_and_refuses() -> None:
    retirement = _import_retirement_contract()
    body_calls: list[int] = []

    @retirement.retired_legacy_ncu_calibration("decorated_unit_test_surface")
    def retired_action(value: int = 7) -> int:
        """Sentinel help text."""
        body_calls.append(value)
        return value

    assert retired_action.__name__ == "retired_action"
    assert retired_action.__doc__ == "Sentinel help text."
    assert str(inspect.signature(retired_action)) == "(value: 'int' = 7) -> 'int'"
    assert not hasattr(retired_action, "__wrapped__")
    assert inspect.unwrap(retired_action) is retired_action
    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        retired_action(9)
    assert caught.value.reason == STABLE_RETIREMENT_REASON
    assert caught.value.surface == "decorated_unit_test_surface"
    assert body_calls == []
