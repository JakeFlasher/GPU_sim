#!/usr/bin/env python3
"""Build a deterministic, fail-closed SOL workload coverage inventory.

The capture roster is problem-granular, while a SOL problem's ``workload.jsonl``
usually contains many workload rows and the capture matrix selects exactly one
of them.  This report keeps those units separate.  It also distinguishes the
typed reason in the unified roster from the underlying SOL disposition
(architecture, compiler, evaluation, correctness, or host capacity), and it
checks whether a selected row in a reduced workload actually differs from the
same source index in the native workload.

No timestamp or host-local probe enters the report.  All inputs are committed
repository artifacts and benchmark workload files.  Cross-file pins, identity
sets, counts, statuses, reduced-workload UUID order, and matrix selections are
validated before any report is returned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


class CoverageInventoryError(ValueError):
    """An input is malformed or the committed evidence does not reconcile."""


@dataclass(frozen=True)
class InventoryInputs:
    authority: Path = Path(
        "data/local_labels/campaign/capture_roster_authority_sm86.json")
    matrix: Path = Path(
        "data/local_labels/campaign/sm86_reference_capture_matrix.json")
    runnability: Path = Path(
        "data/local_labels/campaign/sol_runnability_sm86.json")
    shrink: Path = Path(
        "data/local_labels/campaign/sol_shrink_provenance_sm86.json")
    port_scope: Path = Path(
        "tools/sol_baselines_harness/calibration/sm86_port_scope_v1.toml")
    benchmark_root: Path = Path("tools/sol_execbench/data/benchmark")


DEFAULT_INPUTS = InventoryInputs()

_CATEGORY_DIRS = {
    "FIB": "FlashInfer-Bench",
    "L1": "L1",
    "L2": "L2",
    "Quant": "Quant",
}
_CATEGORIES = tuple(sorted(_CATEGORY_DIRS))
_CANONICAL_SLUG_RE = re.compile(r"problem_[A-Za-z0-9_]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUN_STATUSES = frozenset({
    "runnable",
    "compile_failed",
    "correctness_failed",
    "host_capacity_oom",
    "eval_incomplete",
    "eval_error",
    "resolve_error",
})
_SHRINK_CLASSIFICATIONS = frozenset({
    "shape_adjusted",
    "reduced_representative",
    "native_captured_via_replay_mode",
    "unavailable::oom_shape_gated",
    "correctness_failed",
    "calibration_inconclusive",
})
_FINAL_DISPOSITIONS = (
    "architecture_excluded",
    "capacity_excluded",
    "compiler_excluded",
    "correctness_excluded",
    "evaluation_excluded",
    "native_runnable",
    "reduced_runnable",
)
_EXCLUSION_BUCKETS = (
    "architecture",
    "capacity",
    "compiler",
    "correctness",
    "evaluation",
)


def canonical_json_bytes(value: Any) -> bytes:
    """The canonical report and committed-artifact JSON representation."""
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2,
                       sort_keys=True)
            + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoverageInventoryError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> Any:
    raise CoverageInventoryError(f"non-finite JSON number: {value}")


def _parse_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except CoverageInventoryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageInventoryError(f"{label}: invalid JSON: {exc}") from exc


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CoverageInventoryError(f"{label}: unreadable: {path}: {exc}") from exc
    value = _parse_json(payload, label)
    if not isinstance(value, dict):
        raise CoverageInventoryError(f"{label}: JSON root must be an object")
    if payload != canonical_json_bytes(value):
        raise CoverageInventoryError(f"{label}: JSON bytes are not canonical")
    return value, payload


def _resolve_input(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoverageInventoryError(message)


def _require_sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{label}: expected lowercase SHA-256")
    return value


def _canonical_slug(directory_name: str) -> str:
    full = "problem_" + directory_name
    match = _CANONICAL_SLUG_RE.match(full)
    if match is None:
        raise CoverageInventoryError(
            f"benchmark directory cannot form canonical slug: {directory_name!r}")
    return match.group(0)


def _repo_path(repo_root: Path, relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative,
             f"{label}: repository-relative path must be a nonempty string")
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts,
             f"{label}: unsafe repository-relative path: {relative!r}")
    resolved = (repo_root / path).resolve()
    _require(resolved == repo_root or repo_root in resolved.parents,
             f"{label}: path escapes repository: {relative!r}")
    return resolved


def _load_workload_rows(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise CoverageInventoryError(f"{label}: unreadable: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    uuids: set[str] = set()
    for line_number, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(b"//"):
            continue
        value = _parse_json(stripped, f"{label}:{line_number}")
        _require(isinstance(value, dict),
                 f"{label}:{line_number}: workload row must be an object")
        uuid = value.get("uuid")
        _require(isinstance(uuid, str) and uuid,
                 f"{label}:{line_number}: workload row needs a nonempty uuid")
        _require(uuid not in uuids,
                 f"{label}:{line_number}: duplicate workload uuid: {uuid}")
        uuids.add(uuid)
        rows.append(value)
    _require(bool(rows), f"{label}: workload file has no effective rows")
    return rows


def _load_benchmark_index(repo_root: Path, root: Path) -> dict[str, dict[str, Any]]:
    _require(root.is_dir(), f"benchmark root missing: {root}")
    observed_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    expected_dirs = set(_CATEGORY_DIRS.values())
    _require(observed_dirs == expected_dirs,
             "benchmark category directories mismatch: "
             f"observed={sorted(observed_dirs)!r} expected={sorted(expected_dirs)!r}")

    problems: dict[str, dict[str, Any]] = {}
    for category in _CATEGORIES:
        category_dir = root / _CATEGORY_DIRS[category]
        for problem_dir in sorted(path for path in category_dir.iterdir()
                                  if path.is_dir()):
            slug = _canonical_slug(problem_dir.name)
            _require(slug not in problems,
                     f"canonical benchmark slug collision: {slug}")
            workload_path = problem_dir / "workload.jsonl"
            rows = _load_workload_rows(workload_path, f"native workload {slug}")
            problems[slug] = {
                "category": category,
                "directory": _display_path(repo_root, problem_dir),
                "workload_path": _display_path(repo_root, workload_path),
                "workload_sha256": _sha256(workload_path.read_bytes()),
                "rows": rows,
            }
    return problems


def _load_port_scope(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = tomllib.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CoverageInventoryError(f"port scope invalid: {path}: {exc}") from exc
    _require(isinstance(value, dict), "port scope root must be a table")
    return value, payload


def _validate_port_scope(
        scope: dict[str, Any], problems: dict[str, dict[str, Any]]) -> tuple[
            dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(scope.get("schema_version") == 1,
             "port scope schema_version must be 1")
    meta = scope.get("meta")
    in_scope = scope.get("in_scope")
    excluded_rows = scope.get("excluded")
    _require(isinstance(meta, dict) and isinstance(in_scope, dict)
             and isinstance(excluded_rows, list), "port scope shape is invalid")
    _require(set(in_scope) == {"L1", "L2", "FIB"},
             f"port scope in_scope categories invalid: {sorted(in_scope)!r}")

    included: dict[str, dict[str, Any]] = {}
    for category in ("L1", "L2", "FIB"):
        slugs = in_scope[category]
        _require(isinstance(slugs, list),
                 f"port scope {category} list is invalid")
        for slug in slugs:
            _require(isinstance(slug, str) and slug not in included,
                     f"port scope duplicate/invalid in-scope slug: {slug!r}")
            _require(slug in problems and problems[slug]["category"] == category,
                     f"port scope in-scope identity mismatch: {category}:{slug}")
            included[slug] = {"category": category}

    excluded: dict[str, dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for index, row in enumerate(excluded_rows):
        _require(isinstance(row, dict),
                 f"port scope excluded[{index}] is not an object")
        slug, category, reason = (row.get("slug"), row.get("category"),
                                  row.get("reason"))
        _require(isinstance(slug, str) and slug not in included
                 and slug not in excluded,
                 f"port scope duplicate/invalid excluded slug: {slug!r}")
        _require(category in {"FIB", "Quant"}
                 and slug in problems and problems[slug]["category"] == category,
                 f"port scope excluded identity mismatch: {category}:{slug}")
        _require(reason in {"fp8_e4m3_contract", "nvfp4_e2m1_contract"},
                 f"port scope unsupported exclusion reason: {reason!r}")
        excluded[slug] = {"category": category, "reason": reason}
        reason_counts[reason] += 1
        category_counts[category] += 1

    _require(set(included) | set(excluded) == set(problems)
             and not (set(included) & set(excluded)),
             "port scope does not partition the benchmark problem set")
    _require(meta.get("total_problems") == len(problems)
             and meta.get("in_scope_count") == len(included)
             and meta.get("excluded_count") == len(excluded),
             "port scope meta problem counts do not reconcile")
    _require(meta.get("in_scope_by_category") == dict(sorted(Counter(
        row["category"] for row in included.values()).items())),
        "port scope in_scope_by_category does not reconcile")
    _require(meta.get("excluded_by_category") == dict(sorted(category_counts.items())),
             "port scope excluded_by_category does not reconcile")
    _require(meta.get("excluded_by_reason") == dict(sorted(reason_counts.items())),
             "port scope excluded_by_reason does not reconcile")
    return included, excluded


def _authority_counts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(cell.get("resolution_status") for cell in cells)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    for cell in cells:
        source = cell.get("roster_source")
        status = cell.get("resolution_status")
        _require(isinstance(source, str) and status in {
            "runnable", "pending_build", "pending_reconstitution",
            "pending_shrink", "excluded",
        }, f"authority cell has invalid source/status: {source!r}/{status!r}")
        sources[source][status] += 1
    return {
        "universe": len(cells),
        "runnable": statuses["runnable"],
        "pending_build": statuses["pending_build"],
        "pending_reconstitution": statuses["pending_reconstitution"],
        "pending_shrink": statuses["pending_shrink"],
        "excluded": statuses["excluded"],
        "by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(sources.items())
        },
    }


def _validate_runnability(
        runnability: dict[str, Any], included: dict[str, dict[str, Any]],
        problems: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _require(runnability.get("schema_version") == 1
             and runnability.get("artifact_id") == "sm86_sol_runnability_v1",
             "SOL runnability identity mismatch")
    native_env_hash = _require_sha(
        runnability.get("native_env_hash"), "SOL runnability native_env_hash")
    cells = runnability.get("cells")
    _require(isinstance(cells, list), "SOL runnability cells must be a list")
    by_slug: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for index, row in enumerate(cells):
        _require(isinstance(row, dict),
                 f"SOL runnability cells[{index}] is not an object")
        slug, category, status = (row.get("slug"), row.get("category"),
                                  row.get("status"))
        _require(isinstance(slug, str) and slug not in by_slug,
                 f"SOL runnability duplicate/invalid slug: {slug!r}")
        _require(slug in included and category == included[slug]["category"],
                 f"SOL runnability identity mismatch: {category}:{slug}")
        _require(status in _RUN_STATUSES,
                 f"SOL runnability status invalid for {slug}: {status!r}")
        _require(row.get("expected_workloads") == len(problems[slug]["rows"]),
                 f"SOL runnability workload count mismatch for {slug}")
        by_slug[slug] = row
        status_counts[status] += 1
    _require(set(by_slug) == set(included),
             "SOL runnability cells do not equal the port-scope in-scope set")
    summary = runnability.get("summary")
    expected_summary = {
        "total": len(cells),
        "runnable": status_counts["runnable"],
        "by_status": {
            status: status_counts[status] for status in sorted(_RUN_STATUSES)
        },
    }
    _require(summary == expected_summary,
             f"SOL runnability summary mismatch: {summary!r} != {expected_summary!r}")
    _require(runnability.get("in_scope_total") == len(cells)
             and runnability.get("probed") == len(cells)
             and runnability.get("pending") == 0,
             "SOL runnability probe accounting is incomplete")
    mapping = runnability.get("status_to_typed_reason")
    expected_mapping = {
        "compile_failed": "calibration_inconclusive",
        "correctness_failed": "correctness_failed",
        "eval_error": "calibration_inconclusive",
        "eval_incomplete": "calibration_inconclusive",
        "host_capacity_oom": "host_capacity_deferred",
        "resolve_error": "calibration_inconclusive",
    }
    _require(mapping == expected_mapping,
             "SOL runnability status_to_typed_reason mismatch")
    # Return the hash too without weakening the row map type.
    by_slug["__native_env_hash__"] = {"value": native_env_hash}
    return by_slug


def _compare_workload_rows(
        native_rows: list[dict[str, Any]], reduced_rows: list[dict[str, Any]],
        *, source_index: int, staged_index: int, label: str) -> dict[str, Any]:
    _require(len(native_rows) == len(reduced_rows),
             f"{label}: native/reduced workload row counts differ")
    native_uuids = [row.get("uuid") for row in native_rows]
    reduced_uuids = [row.get("uuid") for row in reduced_rows]
    _require(native_uuids == reduced_uuids,
             f"{label}: reduced workload UUID order differs from native")
    _require(isinstance(source_index, int) and not isinstance(source_index, bool)
             and 0 <= source_index < len(native_rows),
             f"{label}: source_workload_index is out of range")
    _require(isinstance(staged_index, int) and not isinstance(staged_index, bool)
             and 0 <= staged_index < len(reduced_rows),
             f"{label}: staged_workload_index is out of range")
    changed = sum(native != reduced
                  for native, reduced in zip(native_rows, reduced_rows))
    native_selected = native_rows[source_index]
    reduced_selected = reduced_rows[staged_index]
    return {
        "native_workload_rows": len(native_rows),
        "reduced_workload_rows": len(reduced_rows),
        "changed_workload_rows": changed,
        "unchanged_workload_rows": len(native_rows) - changed,
        "source_workload_index": source_index,
        "staged_workload_index": staged_index,
        "selected_native_uuid": native_selected["uuid"],
        "selected_reduced_uuid": reduced_selected["uuid"],
        "selected_row_differs_from_native": native_selected != reduced_selected,
    }


def _validate_shrink(
        repo_root: Path, shrink: dict[str, Any], run_rows: dict[str, dict[str, Any]],
        problems: dict[str, dict[str, Any]]) -> tuple[
            dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(shrink.get("schema_version") == 1
             and shrink.get("artifact_id") == "sm86_sol_shrink_provenance_v1",
             "SOL shrink identity mismatch")
    native_env_hash = run_rows["__native_env_hash__"]["value"]
    _require(shrink.get("native_env_hash") == native_env_hash,
             "SOL shrink native_env_hash differs from runnability")
    cells = shrink.get("cells")
    _require(isinstance(cells, list), "SOL shrink cells must be a list")
    by_slug: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(cells):
        _require(isinstance(row, dict),
                 f"SOL shrink cells[{index}] is not an object")
        slug = row.get("slug")
        _require(isinstance(slug, str) and slug not in by_slug
                 and slug in run_rows and slug in problems,
                 f"SOL shrink duplicate/unknown slug: {slug!r}")
        _require(run_rows[slug].get("status") == "host_capacity_oom",
                 f"SOL shrink row is not a native host-capacity cell: {slug}")
        _require(row.get("category") == problems[slug]["category"],
                 f"SOL shrink category mismatch for {slug}")
        classification = row.get("classification")
        _require(classification in _SHRINK_CLASSIFICATIONS,
                 f"SOL shrink classification invalid for {slug}: {classification!r}")
        _require(isinstance(row.get("native_predictive"), bool),
                 f"SOL shrink native_predictive invalid for {slug}")
        native_ref = row.get("native_shape_ref")
        _require(native_ref == problems[slug]["workload_path"],
                 f"SOL shrink native_shape_ref mismatch for {slug}")
        reduced_ref = row.get("reduced_workload_ref")
        if reduced_ref is None:
            _require(row.get("reduced_run_status") == "no_safe_axis",
                     f"SOL shrink null reduced path is not no_safe_axis for {slug}")
        else:
            reduced_path = _repo_path(repo_root, reduced_ref,
                                      f"SOL shrink reduced_workload_ref {slug}")
            reduced_rows = _load_workload_rows(
                reduced_path, f"reduced workload {slug}")
            comparisons[slug] = _compare_workload_rows(
                problems[slug]["rows"], reduced_rows,
                source_index=0, staged_index=0, label=slug)
            comparisons[slug]["reduced_workload_path"] = reduced_ref
            comparisons[slug]["reduced_workload_sha256"] = _sha256(
                reduced_path.read_bytes())
        by_slug[slug] = row
    host_capacity = {
        slug for slug, row in run_rows.items()
        if slug != "__native_env_hash__" and row.get("status") == "host_capacity_oom"
    }
    _require(set(by_slug) == host_capacity,
             "SOL shrink cells do not equal native host-capacity cells")
    return by_slug, comparisons


def _final_disposition(
        slug: str, *, excluded_scope: dict[str, dict[str, Any]],
        run_rows: dict[str, dict[str, Any]],
        shrink_rows: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    if slug in excluded_scope:
        return "architecture_excluded", "arch_gate"
    status = run_rows[slug]["status"]
    if status == "runnable":
        return "native_runnable", None
    if status == "compile_failed":
        return "compiler_excluded", "calibration_inconclusive"
    if status in {"eval_error", "eval_incomplete", "resolve_error"}:
        return "evaluation_excluded", "calibration_inconclusive"
    if status == "correctness_failed":
        return "correctness_excluded", "correctness_failed"
    _require(status == "host_capacity_oom" and slug in shrink_rows,
             f"SOL disposition cannot be derived for {slug}: {status!r}")
    classification = shrink_rows[slug]["classification"]
    if classification in {"shape_adjusted", "reduced_representative"}:
        return "reduced_runnable", None
    if classification == "native_captured_via_replay_mode":
        return "native_runnable", None
    if classification == "unavailable::oom_shape_gated":
        return "capacity_excluded", "host_capacity_deferred"
    if classification == "correctness_failed":
        return "correctness_excluded", "correctness_failed"
    _require(classification == "calibration_inconclusive",
             f"unsupported shrink disposition for {slug}: {classification!r}")
    reduced_status = shrink_rows[slug].get("reduced_run_status")
    if reduced_status == "compile_failed":
        return "compiler_excluded", "calibration_inconclusive"
    return "evaluation_excluded", "calibration_inconclusive"


def _validate_authority(
        authority: dict[str, Any], problems: dict[str, dict[str, Any]],
        excluded_scope: dict[str, dict[str, Any]],
        run_rows: dict[str, dict[str, Any]],
        shrink_rows: dict[str, dict[str, Any]]) -> tuple[
            dict[str, dict[str, Any]], dict[str, tuple[str, str | None]]]:
    _require(authority.get("schema_version") == 1
             and authority.get("roster_id") == "sm86_capture_roster_authority_v1",
             "capture authority identity mismatch")
    _require(authority.get("arch") == "sm_86"
             and authority.get("accounting_identity_holds") is True,
             "capture authority architecture/accounting mismatch")
    cells = authority.get("cells")
    _require(isinstance(cells, list), "capture authority cells must be a list")
    _require(authority.get("counts") == _authority_counts(cells),
             "capture authority counts do not reconcile")
    all_cells: dict[tuple[str, str], dict[str, Any]] = {}
    for index, cell in enumerate(cells):
        _require(isinstance(cell, dict),
                 f"capture authority cells[{index}] is not an object")
        identity = (cell.get("roster_source"), cell.get("slug"))
        _require(all(isinstance(part, str) for part in identity)
                 and identity not in all_cells,
                 f"capture authority duplicate/invalid identity: {identity!r}")
        all_cells[identity] = cell

    sol = {slug: cell for (source, slug), cell in all_cells.items()
           if source == "sol_execbench"}
    _require(set(sol) == set(problems),
             "capture authority SOL identities do not equal benchmark identities")
    dispositions: dict[str, tuple[str, str | None]] = {}
    for slug, source in problems.items():
        disposition, typed_reason = _final_disposition(
            slug, excluded_scope=excluded_scope, run_rows=run_rows,
            shrink_rows=shrink_rows)
        cell = sol[slug]
        _require(cell.get("category") == source["category"],
                 f"capture authority category mismatch for {slug}")
        expected_resolution = (
            "runnable" if disposition in {"native_runnable", "reduced_runnable"}
            else "excluded")
        _require(cell.get("resolution_status") == expected_resolution,
                 f"capture authority resolution mismatch for {slug}")
        _require(cell.get("exclusion_reason") == typed_reason,
                 f"capture authority exclusion reason mismatch for {slug}")
        if disposition == "native_runnable":
            _require(cell.get("shape_point") == "workloads",
                     f"native runnable shape_point mismatch for {slug}")
        elif disposition == "reduced_runnable":
            _require(cell.get("shape_point") in {
                "shape_adjusted", "reduced_representative"},
                f"reduced runnable shape_point mismatch for {slug}")
        dispositions[slug] = (disposition, typed_reason)
    return all_cells, dispositions


def _validate_pins(
        repo_root: Path, inputs: InventoryInputs,
        authority: dict[str, Any], authority_bytes: bytes,
        matrix: dict[str, Any], run_bytes: bytes, shrink_bytes: bytes,
        scope_bytes: bytes) -> None:
    authority_pin = matrix.get("authority")
    _require(isinstance(authority_pin, dict)
             and authority_pin.get("sha256") == _sha256(authority_bytes)
             and authority_pin.get("roster_id") == authority.get("roster_id"),
             "matrix authority pin mismatch")
    _require(authority_pin.get("path") == _display_path(
        repo_root, _resolve_input(repo_root, inputs.authority)),
        "matrix authority path mismatch")
    source_pins = authority.get("source_pins")
    _require(isinstance(source_pins, dict)
             and source_pins.get("sol_runnability_sha256") == _sha256(run_bytes)
             and source_pins.get("sol_port_scope_sha256") == _sha256(scope_bytes),
             "authority SOL source pin mismatch")
    matrix_pins = matrix.get("input_pins")
    _require(isinstance(matrix_pins, dict), "matrix input_pins missing")
    for field, path, payload in (
        ("sol_runnability", inputs.runnability, run_bytes),
        ("sol_shrink_provenance", inputs.shrink, shrink_bytes),
    ):
        pin = matrix_pins.get(field)
        _require(isinstance(pin, dict)
                 and pin.get("path") == _display_path(
                     repo_root, _resolve_input(repo_root, path))
                 and pin.get("sha256") == _sha256(payload),
                 f"matrix {field} pin mismatch")


def _validate_matrix(
        matrix: dict[str, Any], authority_cells: dict[tuple[str, str], dict[str, Any]],
        dispositions: dict[str, tuple[str, str | None]],
        problems: dict[str, dict[str, Any]],
        shrink_comparisons: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _require(matrix.get("schema_version") == 1
             and matrix.get("matrix_id") == "sm86_reference_capture_matrix_v1",
             "capture matrix identity mismatch")
    cells = matrix.get("cells")
    _require(isinstance(cells, list), "capture matrix cells must be a list")
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for index, cell in enumerate(cells):
        _require(isinstance(cell, dict) and isinstance(cell.get("identity"), dict),
                 f"capture matrix cells[{index}] shape invalid")
        identity = (cell["identity"].get("roster_source"),
                    cell["identity"].get("slug"))
        _require(all(isinstance(part, str) for part in identity)
                 and identity not in by_identity,
                 f"capture matrix duplicate/invalid identity: {identity!r}")
        _require(identity in authority_cells,
                 f"capture matrix identity absent from authority: {identity!r}")
        expected = authority_cells[identity].get("resolution_status")
        _require(cell.get("disposition") == expected,
                 f"capture matrix disposition mismatch: {identity!r}")
        by_identity[identity] = cell
    _require(set(by_identity) == set(authority_cells),
             "capture matrix identity set differs from authority")

    basics = matrix.get("counts")
    _require(isinstance(basics, dict)
             and basics.get("universe") == len(cells)
             and basics.get("runnable") == sum(
                 cell.get("disposition") == "runnable" for cell in cells)
             and basics.get("excluded") == sum(
                 cell.get("disposition") == "excluded" for cell in cells),
             "capture matrix counts do not reconcile")

    selections: dict[str, dict[str, Any]] = {}
    for slug, (disposition, _reason) in dispositions.items():
        matrix_cell = by_identity[("sol_execbench", slug)]
        if disposition not in {"native_runnable", "reduced_runnable"}:
            _require("invocation" not in matrix_cell,
                     f"excluded SOL cell unexpectedly has invocation: {slug}")
            continue
        invocation = matrix_cell.get("invocation")
        _require(isinstance(invocation, dict),
                 f"runnable SOL cell lacks invocation: {slug}")
        capture_target = invocation.get("capture_target")
        logical = invocation.get("logical_workload")
        _require(isinstance(capture_target, dict) and isinstance(logical, dict),
                 f"SOL capture selection missing for {slug}")
        source_index = capture_target.get("source_workload_index")
        staged_index = capture_target.get("staged_workload_index")
        _require(source_index == 0 and staged_index == 0,
                 f"SOL capture target must select source/staged index 0: {slug}")
        _require(capture_target.get("candidate_user_fn_calls") == 1
                 and capture_target.get("candidate_kernel_ordinal") == 0
                 and capture_target.get("capture_launches") == 1,
                 f"SOL capture target cardinality mismatch for {slug}")
        native_rows = problems[slug]["rows"]
        _require(logical.get("expected_workloads") == len(native_rows),
                 f"matrix expected_workloads mismatch for {slug}")
        if disposition == "native_runnable":
            _require(logical.get("workload_scope") == "native"
                     and logical.get("workload_file") == problems[slug]["workload_path"],
                     f"native workload selection mismatch for {slug}")
            _require(logical.get("workload_file_sha256")
                     == problems[slug]["workload_sha256"],
                     f"native workload SHA-256 mismatch for {slug}")
            selections[slug] = {
                "source_workload_index": source_index,
                "staged_workload_index": staged_index,
                "selected_workload_rows": 1,
                "unselected_workload_rows": len(native_rows) - 1,
                "selected_native_uuid": native_rows[source_index]["uuid"],
            }
        else:
            _require(logical.get("workload_scope") == "reduced_representative",
                     f"reduced workload scope mismatch for {slug}")
            _require(slug in shrink_comparisons,
                     f"reduced runnable lacks reduced workload comparison: {slug}")
            comparison = dict(shrink_comparisons[slug])
            _require(logical.get("workload_file") == comparison["reduced_workload_path"],
                     f"reduced matrix workload path mismatch for {slug}")
            _require(logical.get("workload_file_sha256")
                     == comparison["reduced_workload_sha256"],
                     f"reduced matrix workload SHA-256 mismatch for {slug}")
            comparison["source_workload_index"] = source_index
            comparison["staged_workload_index"] = staged_index
            comparison["selected_workload_rows"] = 1
            comparison["unselected_workload_rows"] = (
                comparison["reduced_workload_rows"] - 1)
            selections[slug] = comparison
    return selections


def _empty_category_counts() -> dict[str, dict[str, int]]:
    return {category: {"problem_count": 0, "workload_row_count": 0}
            for category in _CATEGORIES}


def _aggregate_problem_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = _empty_category_counts()
    for row in rows:
        category = row["category"]
        categories[category]["problem_count"] += 1
        categories[category]["workload_row_count"] += row["workload_row_count"]
    return {
        "problem_count": len(rows),
        "workload_row_count": sum(row["workload_row_count"] for row in rows),
        "by_category": categories,
    }


def build_inventory(
        repo_root: Path = REPO_ROOT,
        inputs: InventoryInputs = DEFAULT_INPUTS) -> dict[str, Any]:
    """Validate all inputs and return the complete deterministic inventory."""
    repo_root = Path(repo_root).resolve()
    paths = {name: _resolve_input(repo_root, getattr(inputs, name)) for name in (
        "authority", "matrix", "runnability", "shrink", "port_scope",
        "benchmark_root",
    )}
    authority, authority_bytes = _load_canonical_json(
        paths["authority"], "capture authority")
    matrix, matrix_bytes = _load_canonical_json(paths["matrix"], "capture matrix")
    runnability, run_bytes = _load_canonical_json(
        paths["runnability"], "SOL runnability")
    shrink, shrink_bytes = _load_canonical_json(paths["shrink"], "SOL shrink")
    scope, scope_bytes = _load_port_scope(paths["port_scope"])
    problems = _load_benchmark_index(repo_root, paths["benchmark_root"])
    included, excluded_scope = _validate_port_scope(scope, problems)
    run_rows = _validate_runnability(runnability, included, problems)
    shrink_rows, shrink_comparisons = _validate_shrink(
        repo_root, shrink, run_rows, problems)
    authority_cells, dispositions = _validate_authority(
        authority, problems, excluded_scope, run_rows, shrink_rows)
    _validate_pins(
        repo_root, inputs, authority, authority_bytes, matrix,
        run_bytes, shrink_bytes, scope_bytes)
    selections = _validate_matrix(
        matrix, authority_cells, dispositions, problems, shrink_comparisons)

    problem_rows: list[dict[str, Any]] = []
    for slug in sorted(problems, key=lambda item: (problems[item]["category"], item)):
        source = problems[slug]
        disposition, typed_reason = dispositions[slug]
        run_row = None if slug in excluded_scope else run_rows[slug]
        shrink_row = shrink_rows.get(slug)
        entry: dict[str, Any] = {
            "authority_resolution_status": (
                "runnable" if disposition in {"native_runnable", "reduced_runnable"}
                else "excluded"),
            "category": source["category"],
            "disposition": disposition,
            "native_runnability_status": (
                None if run_row is None else run_row["status"]),
            "slug": slug,
            "typed_exclusion_reason": typed_reason,
            "workload_row_count": len(source["rows"]),
        }
        if slug in excluded_scope:
            entry["architecture_contract_reason"] = excluded_scope[slug]["reason"]
        if shrink_row is not None:
            comparison = shrink_comparisons.get(slug)
            entry["shrink_evidence"] = {
                "classification": shrink_row["classification"],
                "native_predictive": shrink_row["native_predictive"],
                "reduced_run_status": shrink_row.get("reduced_run_status"),
                "reduced_workload": comparison,
            }
        if slug in selections:
            entry["capture_selection"] = selections[slug]
        problem_rows.append(entry)

    by_disposition_rows = {
        disposition: [row for row in problem_rows
                      if row["disposition"] == disposition]
        for disposition in _FINAL_DISPOSITIONS
    }
    by_disposition = {
        disposition: _aggregate_problem_rows(rows)
        for disposition, rows in by_disposition_rows.items()
    }

    native_rows = by_disposition_rows["native_runnable"]
    native_selection = {
        "problem_count": len(native_rows),
        "workload_row_count": sum(row["workload_row_count"] for row in native_rows),
        "selected_workload_row_count": len(native_rows),
        "unselected_workload_row_count": sum(
            row["workload_row_count"] - 1 for row in native_rows),
        "source_workload_index_counts": dict(sorted(Counter(
            str(row["capture_selection"]["source_workload_index"])
            for row in native_rows).items())),
    }

    all_shrink_comparisons = list(shrink_comparisons.values())
    reduced_rows = by_disposition_rows["reduced_runnable"]
    reduced_selection_rows = [row["capture_selection"] for row in reduced_rows]
    reduced_evidence = {
        "native_host_capacity_problem_count": len(shrink_rows),
        "no_reduced_workload_file_problem_count": (
            len(shrink_rows) - len(shrink_comparisons)),
        "reduced_workload_file_problem_count": len(shrink_comparisons),
        "native_workload_row_count": sum(
            row["native_workload_rows"] for row in all_shrink_comparisons),
        "reduced_workload_row_count": sum(
            row["reduced_workload_rows"] for row in all_shrink_comparisons),
        "changed_workload_row_count": sum(
            row["changed_workload_rows"] for row in all_shrink_comparisons),
        "unchanged_workload_row_count": sum(
            row["unchanged_workload_rows"] for row in all_shrink_comparisons),
        "capture_selected_problem_count": len(reduced_selection_rows),
        "capture_selected_row_differs_from_native_count": sum(
            row["selected_row_differs_from_native"]
            for row in reduced_selection_rows),
        "capture_selected_row_matches_native_count": sum(
            not row["selected_row_differs_from_native"]
            for row in reduced_selection_rows),
        "selected_row_comparison_basis": "parsed_json_value_at_matrix_indices",
        "by_classification": dict(sorted(Counter(
            row["classification"] for row in shrink_rows.values()).items())),
    }

    exclusion_map = {
        "architecture": "architecture_excluded",
        "capacity": "capacity_excluded",
        "compiler": "compiler_excluded",
        "correctness": "correctness_excluded",
        "evaluation": "evaluation_excluded",
    }
    exclusion_buckets = {
        bucket: by_disposition[disposition]
        for bucket, disposition in exclusion_map.items()
    }
    typed_excluded_count = sum(
        bucket["problem_count"] for bucket in exclusion_buckets.values())

    benchmark_categories = _empty_category_counts()
    for source in problems.values():
        category = source["category"]
        benchmark_categories[category]["problem_count"] += 1
        benchmark_categories[category]["workload_row_count"] += len(source["rows"])

    workload_manifest = [
        {"path": problems[slug]["workload_path"],
         "sha256": problems[slug]["workload_sha256"]}
        for slug in sorted(problems, key=lambda item: problems[item]["workload_path"])
    ]
    input_records = {}
    for name, payload in (
        ("authority", authority_bytes),
        ("matrix", matrix_bytes),
        ("runnability", run_bytes),
        ("shrink", shrink_bytes),
        ("port_scope", scope_bytes),
    ):
        input_records[name] = {
            "path": _display_path(repo_root, paths[name]),
            "sha256": _sha256(payload),
        }
    input_records["benchmark_workloads"] = {
        "file_count": len(workload_manifest),
        "manifest_sha256": _sha256(canonical_json_bytes(workload_manifest)),
        "root": _display_path(repo_root, paths["benchmark_root"]),
    }

    unified_counts = authority["counts"]
    _require(typed_excluded_count == unified_counts["by_source"]["sol_execbench"][
        "excluded"], "SOL typed-excluded aggregate does not reconcile")
    _require(sum(item["problem_count"] for item in by_disposition.values())
             == len(problems), "SOL problem dispositions do not reconcile")
    _require(sum(item["workload_row_count"] for item in by_disposition.values())
             == sum(len(source["rows"]) for source in problems.values()),
             "SOL workload-row dispositions do not reconcile")

    return {
        "arch": authority["arch"],
        "artifact_id": "sm86_sol_workload_coverage_inventory_v1",
        "definitions": {
            "problem": "one canonical SOL benchmark directory and one SOL authority cell",
            "selected_workload_row": (
                "the matrix capture_target source/staged index; one row per runnable problem"),
            "typed_excluded": (
                "a problem-level accounted non-runnable disposition, not a claim that every "
                "workload is architecturally incompatible"),
            "workload_row": "one effective JSON object line in workload.jsonl",
        },
        "exclusion_buckets": exclusion_buckets,
        "inputs": input_records,
        "schema_version": 1,
        "sku": authority.get("sku"),
        "sol_problem_dispositions": {
            "by_disposition": by_disposition,
            "problems": problem_rows,
            "runnable_problem_count": (
                by_disposition["native_runnable"]["problem_count"]
                + by_disposition["reduced_runnable"]["problem_count"]),
            "total_problem_count": len(problems),
            "typed_excluded_problem_count": typed_excluded_count,
        },
        "unified_roster": unified_counts,
        "workload_rows": {
            "by_category": benchmark_categories,
            "by_disposition": {
                key: value["workload_row_count"]
                for key, value in by_disposition.items()
            },
            "native_runnable_capture_selection": native_selection,
            "reduced_workload_evidence": reduced_evidence,
            "total_workload_row_count": sum(
                len(source["rows"]) for source in problems.values()),
        },
    }


def publish_new_bytes(path: Path, payload: bytes) -> None:
    """Exclusively publish immutable inventory bytes without symlink follows."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise CoverageInventoryError(
            f"published inventory path must be absolute and lexical: {path}"
        )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open("/", directory_flags)
    try:
        for component in path.parts[1:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                child = os.open(component, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise CoverageInventoryError(
                    f"published inventory parent is unsafe: {path}: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        try:
            output_fd = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o444,
                dir_fd=descriptor,
            )
        except OSError as exc:
            raise CoverageInventoryError(
                f"published inventory already exists or is unsafe: {path}: {exc}"
            ) from exc
        try:
            view = memoryview(payload)
            while view:
                written = os.write(output_fd, view)
                if written <= 0:
                    raise CoverageInventoryError(
                        f"short published inventory write: {path}"
                    )
                view = view[written:]
            os.fsync(output_fd)
            observed = os.fstat(output_fd)
            if not stat.S_ISREG(observed.st_mode):
                raise CoverageInventoryError(
                    f"published inventory is not regular: {path}"
                )
        finally:
            os.close(output_fd)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed_payload = path.read_bytes()
    if observed_payload != payload:
        raise CoverageInventoryError(
            f"published inventory byte verification failed: {path}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out", default="-",
        help="canonical JSON output path, or '-' for stdout (default)")
    parser.add_argument(
        "--publish-new", type=Path,
        help=(
            "exclusive absolute external path for immutable baseline "
            "publication; refuses overwrite and every symlink ancestor"
        ),
    )
    args = parser.parse_args(argv)
    try:
        payload = canonical_json_bytes(build_inventory(args.repo_root))
        if args.publish_new is not None and args.out != "-":
            raise CoverageInventoryError(
                "--publish-new and a non-default --out are mutually exclusive"
            )
        if args.publish_new is not None:
            publish_new_bytes(args.publish_new, payload)
            print(
                f"published {args.publish_new} "
                f"sha256={hashlib.sha256(payload).hexdigest()}"
            )
        elif args.out == "-":
            sys.stdout.buffer.write(payload)
        else:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
    except (CoverageInventoryError, OSError) as exc:
        print(f"sol coverage inventory: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
