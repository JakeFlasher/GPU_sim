#!/usr/bin/env python3
"""Audit the exact runnable SOL definition set under the certifying V2 rule."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SOL_EXECBENCH_SRC = REPO_ROOT / "tools" / "sol_execbench" / "src"
for import_root in (REPO_ROOT, SOL_EXECBENCH_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sol_execbench.driver import capture_adapter  # noqa: E402
from sol_execbench.driver.problem_packager import ProblemPackager  # noqa: E402
from tools.local_labels import build_sm86_capture_matrix as matrix_builder  # noqa: E402
from tools.local_labels import sol_definition_equivalence as equiv  # noqa: E402


EXPECTED_ROWS = 74
EXPECTED_NORMALIZED_ROWS = 31
EXPECTED_NORMALIZED_POINTERS = 212
EXPECTED_FULL_MATRIX_ROWS = 344
H1_HEAD = "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a"
H1_MATRIX_SHA256 = (
    "fd3a1868845ceac309e718da6519f4779824b7fd69eddf1bfa0e7e7847d4e297"
)
H1_MATRIX_SEAL_SHA256 = (
    "544938301f40a46713dd056af9d349c4f59848179212ef4a7826f12d9418d447"
)
H1_MATRIX_PATH = Path(
    "/home/jakeshea/sm86_reference_capture_r27/inputs/r28_9b8e0e40/"
    "sm86_reference_capture_matrix.json"
)
PROBLEM_070_ROW_ID = (
    "sol_execbench::problem_070_mamba2_fused_intra_chunk_diagonal_computation"
)
PROBLEM_070_SOURCE_SHA256 = (
    "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
)
PROBLEM_070_STAGED_SHA256 = (
    "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387"
)
SOL_EQUIVALENCE_EVIDENCE_ROOT = (
    REPO_ROOT / "evidence" / "round28" / "sol_equivalence"
)
DEFAULT_BLOCKING_ARTIFACT = (
    SOL_EQUIVALENCE_EVIDENCE_ROOT
    / "sol_definition_equivalence_audit_r28_v1_blocking.json"
)
DEFAULT_PRECOMMIT_ARTIFACT = (
    SOL_EQUIVALENCE_EVIDENCE_ROOT
    / "sol_definition_equivalence_audit_r28_precommit_h1.json"
)
DEFAULT_CERTIFYING_ARTIFACT = (
    SOL_EQUIVALENCE_EVIDENCE_ROOT
    / "sol_definition_equivalence_audit_r28.json"
)
BLOCKING_ARTIFACT_SHA256 = (
    "b6dd22ad6e7a87033374262aa8cd6a8a3ea97ac901a5e1c266a3a6d7341cdcd0"
)
_SHA256_MANIFEST_LINE_RE = re.compile(r"([0-9a-f]{64})  (.+)")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class AuditRefusal(RuntimeError):
    """Fail-closed audit refusal that must prevent a passing artifact."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_utc(value: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise AuditRefusal("AUDIT_TIMESTAMP_INVALID", repr(value))
    try:
        dt.datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise AuditRefusal("AUDIT_TIMESTAMP_INVALID", value) from exc
    return value


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _git_clean(repo_root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    )
    return output == ""


def _verify_repository_fixture(
    path: Path,
    *,
    expected_sha256: str,
    repo_root: Path = REPO_ROOT,
    label: str = "repository evidence fixture",
) -> str:
    """Verify a fixture using Git-reproducible authority, not POSIX mode bits."""

    root = repo_root.resolve()
    fixture = path.expanduser().resolve()
    try:
        relative = fixture.relative_to(root).as_posix()
    except ValueError as exc:
        raise AuditRefusal(
            "AUDIT_FIXTURE_UNTRACKED", f"{label} is outside repository: {fixture}"
        ) from exc

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tracked.returncode != 0 or relative not in tracked.stdout.splitlines():
        raise AuditRefusal(
            "AUDIT_FIXTURE_UNTRACKED", f"{label} is not Git-tracked: {relative}"
        )

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
        cwd=root,
        text=True,
    )
    if status:
        raise AuditRefusal(
            "AUDIT_FIXTURE_DIRTY", f"{label} has local changes: {status.strip()}"
        )

    try:
        observed_sha256 = sha256_file(fixture)
    except OSError as exc:
        raise AuditRefusal(
            "AUDIT_FIXTURE_HASH_MISMATCH", f"{label} cannot be hashed: {exc}"
        ) from exc
    if observed_sha256 != expected_sha256:
        raise AuditRefusal(
            "AUDIT_FIXTURE_HASH_MISMATCH",
            f"{label}: expected={expected_sha256} observed={observed_sha256}",
        )

    manifest = root / "evidence" / "MANIFEST.sha256"
    manifest_relative = "evidence/MANIFEST.sha256"
    if not manifest.is_file():
        raise AuditRefusal(
            "AUDIT_FIXTURE_MANIFEST_MISSING",
            f"{label} manifest is absent: {manifest_relative}",
        )
    manifest_tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", manifest_relative],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    manifest_status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            manifest_relative,
        ],
        cwd=root,
        text=True,
    )
    if (
        manifest_tracked.returncode != 0
        or manifest_relative not in manifest_tracked.stdout.splitlines()
        or manifest_status
    ):
        raise AuditRefusal(
            "AUDIT_FIXTURE_MANIFEST_INVALID",
            f"{label} manifest is untracked or dirty: {manifest_status.strip()}",
        )

    try:
        manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditRefusal(
            "AUDIT_FIXTURE_MANIFEST_INVALID", f"cannot read {manifest}: {exc}"
        ) from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(manifest_lines, start=1):
        if not line:
            continue
        match = _SHA256_MANIFEST_LINE_RE.fullmatch(line)
        if match is None or match.group(2) in entries:
            raise AuditRefusal(
                "AUDIT_FIXTURE_MANIFEST_INVALID",
                f"invalid or duplicate entry at {manifest_relative}:{line_number}",
            )
        entries[match.group(2)] = match.group(1)
    if relative not in entries:
        raise AuditRefusal(
            "AUDIT_FIXTURE_MANIFEST_MISSING",
            f"{label} has no manifest entry: {relative}",
        )
    if entries[relative] != expected_sha256:
        raise AuditRefusal(
            "AUDIT_FIXTURE_MANIFEST_MISMATCH",
            f"{label}: manifest={entries[relative]} expected={expected_sha256}",
        )
    return observed_sha256


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditRefusal("AUDIT_INPUT_INVALID", f"{label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditRefusal("AUDIT_INPUT_INVALID", f"{label} is not an object")
    return value


def _repo_file(repo_root: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise AuditRefusal("AUDIT_INPUT_INVALID", f"{label} path is absent")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditRefusal("AUDIT_INPUT_INVALID", f"unsafe {label} path: {raw_path}")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise AuditRefusal("AUDIT_INPUT_INVALID", f"{label} escapes repository") from exc
    if not resolved.is_file():
        raise AuditRefusal("AUDIT_INPUT_INVALID", f"missing {label}: {resolved}")
    return resolved


def _runnable_sol_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    cells = matrix.get("cells")
    if not isinstance(cells, list):
        raise AuditRefusal("AUDIT_MATRIX_INVALID", "cells is not an array")
    rows = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("disposition") == "runnable"
        and (cell.get("identity") or {}).get("roster_source") == "sol_execbench"
    ]
    if len(rows) != EXPECTED_ROWS:
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH",
            f"expected {EXPECTED_ROWS} runnable SOL rows, observed {len(rows)}",
        )
    row_ids = [
        f"sol_execbench::{(row.get('identity') or {}).get('slug')}" for row in rows
    ]
    if len(set(row_ids)) != EXPECTED_ROWS or any(
        row_id.endswith("::None") for row_id in row_ids
    ):
        raise AuditRefusal("AUDIT_MEMBERSHIP_MISMATCH", "row IDs are not unique")
    return [row for _, row in sorted(zip(row_ids, rows), key=lambda item: item[0])]


def _full_matrix_row_ids(matrix: dict[str, Any]) -> list[str]:
    cells = matrix.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_FULL_MATRIX_ROWS:
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH",
            f"expected {EXPECTED_FULL_MATRIX_ROWS} matrix cells",
        )
    row_ids: list[str] = []
    for index, cell in enumerate(cells):
        identity = cell.get("identity") if isinstance(cell, dict) else None
        source = identity.get("roster_source") if isinstance(identity, dict) else None
        slug = identity.get("slug") if isinstance(identity, dict) else None
        if not isinstance(source, str) or not source or not isinstance(slug, str) or not slug:
            raise AuditRefusal(
                "AUDIT_MEMBERSHIP_MISMATCH", f"invalid matrix identity at {index}"
            )
        row_ids.append(f"{source}::{slug}")
    row_ids.sort()
    if len(set(row_ids)) != EXPECTED_FULL_MATRIX_ROWS:
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH", "full matrix row IDs are not unique"
        )
    return row_ids


def _serialize_definition(
    repo_root: Path, cell: dict[str, Any], path: Path,
) -> bytes:
    """Serialize through the real no-compile/no-execute ProblemPackager path."""

    invocation = cell.get("invocation") or {}
    logical = invocation.get("logical_workload") or {}
    workload_path = _repo_file(
        repo_root, logical.get("workload_file"), label="SOL workload"
    )
    solution_path = _repo_file(
        repo_root,
        invocation.get("authority_reference_path"),
        label="SOL solution",
    )
    source_index = (invocation.get("capture_target") or {}).get(
        "source_workload_index"
    )
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise AuditRefusal(
            "AUDIT_INPUT_INVALID", "SOL source workload index is invalid"
        )
    config_path = (
        _repo_file(
            repo_root,
            f"{logical.get('benchmark_dir')}/config.json",
            label="SOL benchmark config",
        )
        if logical.get("workload_scope") == "native"
        and (repo_root / str(logical.get("benchmark_dir")) / "config.json").is_file()
        else None
    )
    definition = capture_adapter._load_definition(path)
    workload, _raw_row, _row_count = capture_adapter._select_workload(
        workload_path, source_index
    )
    solution = capture_adapter._load_solution(solution_path)
    config = capture_adapter._load_config(config_path)
    with tempfile.TemporaryDirectory(prefix="sol-def-audit-") as temporary:
        stage = Path(temporary)
        packager = ProblemPackager(
            definition=definition,
            workloads=[workload],
            solution=solution,
            config=config,
            output_dir=stage,
            keep_output_dir=True,
        )
        staged_bytes = (stage / "definition.json").read_bytes()
        del packager
    return staged_bytes


def _audit_row(repo_root: Path, cell: dict[str, Any]) -> dict[str, Any]:
    identity = cell.get("identity") or {}
    row_id = f"sol_execbench::{identity.get('slug')}"
    logical = (cell.get("invocation") or {}).get("logical_workload") or {}
    source_relative = logical.get("definition_file")
    source_path = _repo_file(
        repo_root, source_relative, label=f"{row_id} definition"
    )
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != logical.get("definition_file_sha256"):
        raise AuditRefusal(
            "AUDIT_SOURCE_PIN_MISMATCH",
            f"{row_id}: matrix={logical.get('definition_file_sha256')} "
            f"actual={source_sha256}",
        )
    staged_bytes = _serialize_definition(repo_root, cell, source_path)
    comparison = equiv.compare_definition_bytes(source_bytes, staged_bytes)
    return {
        "identity": {
            "roster_source": "sol_execbench",
            "slug": identity.get("slug"),
        },
        "row_id": row_id,
        "source_path": source_relative,
        "source_definition_sha256": comparison.source_sha256,
        "staged_definition_sha256": comparison.staged_sha256,
        "top_level_field_states": comparison.top_level_field_states,
        "axes_container_state": comparison.axes_container_state,
        "axis_description_states": comparison.axis_description_states,
        "tensor_container_states": comparison.tensor_container_states,
        "tensor_description_states": comparison.tensor_description_states,
        "normalized_pointers": list(comparison.normalized_pointers),
        "residual_differences": list(comparison.difference_pointers),
        "verdict": "equal" if comparison.equal else "divergent",
    }


def _reference_membership(reference_path: Path) -> tuple[list[str], list[str]]:
    reference = _load_object(reference_path, label="membership reference")
    if (
        reference.get("artifact_kind")
        != "sol_definition_equivalence_v2_precommit_audit"
        or reference.get("certifying") is not False
        or reference.get("rule_id") != equiv.CERTIFYING_RULE_ID
        or reference.get("verdict") != "pass"
        or reference.get("summary")
        != {
            "rows": 74,
            "unique_rows": 74,
            "equal": 74,
            "divergent": 0,
            "normalized_rows": 31,
            "total_normalized_pointers": 212,
            "total_residual_differences": 0,
        }
    ):
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH",
            "membership reference is not the passing non-certifying H1 audit",
        )
    rows = reference.get("rows")
    if not isinstance(rows, list):
        raise AuditRefusal("AUDIT_MEMBERSHIP_MISMATCH", "reference rows absent")
    row_ids = [row.get("row_id") for row in rows if isinstance(row, dict)]
    if len(row_ids) != len(rows) or any(not isinstance(value, str) for value in row_ids):
        raise AuditRefusal("AUDIT_MEMBERSHIP_MISMATCH", "invalid reference row IDs")
    if (
        len(row_ids) != EXPECTED_ROWS
        or len(set(row_ids)) != EXPECTED_ROWS
        or row_ids != sorted(row_ids)
        or reference.get("membership_sha256") != canonical_sha256(row_ids)
    ):
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH",
            "membership reference row set/order/hash is invalid",
        )
    full_membership = reference.get("full_membership")
    full_row_ids = (
        full_membership.get("row_ids")
        if isinstance(full_membership, dict) else None
    )
    if (
        not isinstance(full_row_ids, list)
        or len(full_row_ids) != EXPECTED_FULL_MATRIX_ROWS
        or any(not isinstance(value, str) for value in full_row_ids)
        or full_row_ids != sorted(full_row_ids)
        or len(set(full_row_ids)) != EXPECTED_FULL_MATRIX_ROWS
        or full_membership.get("sha256") != canonical_sha256(full_row_ids)
    ):
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH",
            "membership reference full 344-row set/order/hash is invalid",
        )
    return row_ids, full_row_ids


def build_audit(
    *,
    matrix_path: Path,
    expected_head: str,
    expected_matrix_sha256: str,
    blocking_artifact_path: Path = DEFAULT_BLOCKING_ARTIFACT,
    certifying: bool,
    created_at_utc: str | None = None,
    membership_reference: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build and fully validate one V2 audit object without writing it."""

    repo_root = repo_root.resolve()
    matrix_path = matrix_path.expanduser().resolve()
    blocking_artifact_path = blocking_artifact_path.expanduser().resolve()
    created = _validate_utc(created_at_utc or _utc_now())
    head = _git_head(repo_root)
    if head != expected_head:
        raise AuditRefusal(
            "AUDIT_HEAD_MISMATCH", f"expected={expected_head} observed={head}"
        )
    if certifying and membership_reference is None:
        raise AuditRefusal(
            "AUDIT_MEMBERSHIP_MISMATCH",
            "certifying audit requires the passing H1 membership reference",
        )
    if certifying and not _git_clean(repo_root):
        raise AuditRefusal("AUDIT_TREE_DIRTY", "certifying audit requires a clean tree")
    matrix_sha256 = sha256_file(matrix_path)
    if matrix_sha256 != expected_matrix_sha256:
        raise AuditRefusal(
            "AUDIT_MATRIX_HASH_MISMATCH",
            f"expected={expected_matrix_sha256} observed={matrix_sha256}",
        )
    matrix = _load_object(matrix_path, label="matrix")
    matrix_unsealed = {
        key: value for key, value in matrix.items() if key != "matrix_seal_sha256"
    }
    observed_matrix_seal = matrix_builder.canonical_sha256(matrix_unsealed)
    if matrix.get("matrix_seal_sha256") != observed_matrix_seal:
        raise AuditRefusal(
            "AUDIT_MATRIX_SEAL_MISMATCH",
            f"claimed={matrix.get('matrix_seal_sha256')} "
            f"observed={observed_matrix_seal}",
        )
    if matrix.get("source_commit") != head:
        raise AuditRefusal(
            "AUDIT_HEAD_MISMATCH",
            f"matrix={matrix.get('source_commit')} repository={head}",
        )
    matrix_problems = matrix_builder.validate_matrix(matrix, repo_root=repo_root)
    if matrix_problems:
        raise AuditRefusal(
            "AUDIT_MATRIX_INVALID",
            json.dumps(matrix_problems, sort_keys=True),
        )

    blocking_sha256 = _verify_repository_fixture(
        blocking_artifact_path,
        expected_sha256=BLOCKING_ARTIFACT_SHA256,
        repo_root=repo_root,
        label="preserved V1 blocking artifact",
    )
    blocking = _load_object(blocking_artifact_path, label="blocking artifact")
    if (
        blocking.get("rule_id") != equiv.NONCERTIFYING_V1_RULE_ID
        or blocking.get("summary")
        != {
            "divergent": 31,
            "equal": 43,
            "rows": 74,
            "total_difference_pointers": 212,
            "unique_rows": 74,
        }
        or blocking.get("verdict") != "blocking_divergence_preserved"
    ):
        raise AuditRefusal(
            "AUDIT_BLOCKING_ARTIFACT_MISMATCH",
            "preserved V1 blocking artifact has unexpected identity or semantics",
        )
    blocking_rows_raw = blocking.get("rows")
    if not isinstance(blocking_rows_raw, list) or len(blocking_rows_raw) != EXPECTED_ROWS:
        raise AuditRefusal(
            "AUDIT_BLOCKING_ARTIFACT_MISMATCH",
            "preserved V1 blocking artifact row set is invalid",
        )
    blocking_rows = {
        row.get("row_id"): row for row in blocking_rows_raw if isinstance(row, dict)
    }
    if len(blocking_rows) != EXPECTED_ROWS or None in blocking_rows:
        raise AuditRefusal(
            "AUDIT_BLOCKING_ARTIFACT_MISMATCH",
            "preserved V1 blocking artifact row identities are invalid",
        )

    comparator_path = repo_root / "tools/local_labels/sol_definition_equivalence.py"
    packager_path = (
        repo_root
        / "tools/sol_execbench/src/sol_execbench/driver/problem_packager.py"
    )
    comparator_sha256 = sha256_file(comparator_path)
    packager_sha256 = sha256_file(packager_path)
    rows = [_audit_row(repo_root, cell) for cell in _runnable_sol_rows(matrix)]
    row_ids = [row["row_id"] for row in rows]
    full_row_ids = _full_matrix_row_ids(matrix)
    for row in rows:
        prior = blocking_rows.get(row["row_id"])
        if (
            not isinstance(prior, dict)
            or row["source_definition_sha256"]
            != prior.get("source_definition_sha256")
            or row["staged_definition_sha256"]
            != prior.get("staged_definition_sha256")
            or row["normalized_pointers"] != prior.get("difference_pointers")
        ):
            raise AuditRefusal(
                "AUDIT_BLOCKING_ARTIFACT_MISMATCH",
                f"{row['row_id']} does not close the preserved V1 evidence",
            )
    if membership_reference is not None:
        reference_ids, reference_full_ids = _reference_membership(
            membership_reference.expanduser().resolve()
        )
        if row_ids != reference_ids or full_row_ids != reference_full_ids:
            raise AuditRefusal(
                "AUDIT_MEMBERSHIP_MISMATCH",
                "SOL-74 or full-344 row IDs differ from the H1 reference",
            )

    divergent_rows = [row for row in rows if row["residual_differences"]]
    normalized_rows = [row for row in rows if row["normalized_pointers"]]
    normalized_pointer_count = sum(
        len(row["normalized_pointers"]) for row in rows
    )
    residual_count = sum(len(row["residual_differences"]) for row in rows)
    if divergent_rows:
        preview = {
            row["row_id"]: row["residual_differences"] for row in divergent_rows[:5]
        }
        raise AuditRefusal(
            "AUDIT_UNPERMITTED_DIVERGENCE",
            json.dumps(preview, sort_keys=True),
        )
    if (
        len(normalized_rows) != EXPECTED_NORMALIZED_ROWS
        or normalized_pointer_count != EXPECTED_NORMALIZED_POINTERS
    ):
        raise AuditRefusal(
            "AUDIT_NORMALIZATION_COUNT_MISMATCH",
            f"normalized_rows={len(normalized_rows)} "
            f"normalized_pointers={normalized_pointer_count}",
        )

    controls = [row for row in rows if row["row_id"] == PROBLEM_070_ROW_ID]
    if len(controls) != 1:
        raise AuditRefusal("AUDIT_CONTROL_MISMATCH", "problem-070 row absent")
    control = controls[0]
    if (
        control["source_definition_sha256"] != PROBLEM_070_SOURCE_SHA256
        or control["staged_definition_sha256"] != PROBLEM_070_STAGED_SHA256
        or control["residual_differences"]
        or control["normalized_pointers"]
        or control["top_level_field_states"].get("op_type")
        != {
            "source": "missing",
            "staged": "null",
            "source_normalized": True,
            "staged_normalized": False,
        }
        or any(
            state.get("source_normalized") or state.get("staged_normalized")
            for state in control["axis_description_states"].values()
        )
    ):
        raise AuditRefusal(
            "AUDIT_CONTROL_MISMATCH", json.dumps(control, sort_keys=True)
        )

    _verify_repository_fixture(
        blocking_artifact_path,
        expected_sha256=BLOCKING_ARTIFACT_SHA256,
        repo_root=repo_root,
        label="preserved V1 blocking artifact",
    )
    if (
        _git_head(repo_root) != head
        or sha256_file(matrix_path) != matrix_sha256
        or sha256_file(comparator_path) != comparator_sha256
        or sha256_file(packager_path) != packager_sha256
    ):
        raise AuditRefusal(
            "AUDIT_INPUT_CHANGED",
            "HEAD, matrix, blocking evidence, comparator, or packager changed "
            "during the audit",
        )
    if certifying and not _git_clean(repo_root):
        raise AuditRefusal(
            "AUDIT_TREE_DIRTY", "tree changed during the certifying audit"
        )
    artifact = {
        "schema_version": 1,
        "artifact_kind": (
            "sol_definition_equivalence_v2_certifying_audit"
            if certifying
            else "sol_definition_equivalence_v2_precommit_audit"
        ),
        "certifying": certifying,
        "created_at_utc": created,
        "git_head": head,
        "rule_id": equiv.CERTIFYING_RULE_ID,
        "matrix": {
            "path": str(matrix_path),
            "file_sha256": matrix_sha256,
            "matrix_seal_sha256": matrix.get("matrix_seal_sha256"),
        },
        "comparator_module": {
            "path": comparator_path.relative_to(repo_root).as_posix(),
            "sha256": comparator_sha256,
        },
        "packager_module": {
            "path": packager_path.relative_to(repo_root).as_posix(),
            "sha256": packager_sha256,
        },
        "prior_blocking_artifact": {
            "path": str(blocking_artifact_path),
            "sha256": blocking_sha256,
        },
        "membership_sha256": canonical_sha256(row_ids),
        "full_membership": {
            "row_ids": full_row_ids,
            "sha256": canonical_sha256(full_row_ids),
        },
        "problem_070_control": {
            "row_id": PROBLEM_070_ROW_ID,
            "source_definition_sha256": control["source_definition_sha256"],
            "staged_definition_sha256": control["staged_definition_sha256"],
            "verdict": "equal",
        },
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "unique_rows": len(set(row_ids)),
            "equal": len(rows) - len(divergent_rows),
            "divergent": len(divergent_rows),
            "normalized_rows": len(normalized_rows),
            "total_normalized_pointers": normalized_pointer_count,
            "total_residual_differences": residual_count,
        },
        "verdict": "pass",
    }
    return artifact


def audit_bytes(artifact: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without the POSIX rename-overwrite race."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise AuditRefusal(
            "AUDIT_ATOMIC_PUBLICATION_UNAVAILABLE",
            "renameat2(RENAME_NOREPLACE) is unavailable",
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno == errno.EEXIST:
        raise AuditRefusal("AUDIT_OUTPUT_EXISTS", str(destination))
    raise AuditRefusal(
        "AUDIT_ATOMIC_PUBLICATION_FAILED",
        f"{source} -> {destination}: {os.strerror(observed_errno)}",
    )


def write_audit_atomic(path: Path, artifact: dict[str, Any]) -> None:
    """Publish a new audit without overwriting prior evidence."""

    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise AuditRefusal("AUDIT_OUTPUT_EXISTS", str(output))
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise AuditRefusal("AUDIT_OUTPUT_EXISTS", str(temporary))
    try:
        with temporary.open("xb") as handle:
            handle.write(audit_bytes(artifact))
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_publication_context(artifact: dict[str, Any]) -> None:
    """Recheck every time-varying input immediately before publication."""

    comparator = REPO_ROOT / artifact["comparator_module"]["path"]
    packager = REPO_ROOT / artifact["packager_module"]["path"]
    matrix = Path(artifact["matrix"]["path"])
    blocking = Path(artifact["prior_blocking_artifact"]["path"])
    blocking_sha256 = _verify_repository_fixture(
        blocking,
        expected_sha256=BLOCKING_ARTIFACT_SHA256,
        repo_root=REPO_ROOT,
        label="preserved V1 blocking artifact",
    )
    if (
        _git_head(REPO_ROOT) != artifact.get("git_head")
        or sha256_file(matrix) != artifact["matrix"]["file_sha256"]
        or sha256_file(comparator) != artifact["comparator_module"]["sha256"]
        or sha256_file(packager) != artifact["packager_module"]["sha256"]
        or blocking_sha256 != artifact["prior_blocking_artifact"]["sha256"]
    ):
        raise AuditRefusal(
            "AUDIT_INPUT_CHANGED", "audit inputs changed before publication"
        )
    if artifact.get("certifying") is True and not _git_clean(REPO_ROOT):
        raise AuditRefusal(
            "AUDIT_TREE_DIRTY", "certifying publication requires a clean tree"
        )


def _validate_cli_role(args: argparse.Namespace) -> None:
    current_head_certifying = getattr(args, "certifying_current_head", False)
    observed_output = args.output.expanduser().resolve()
    if current_head_certifying:
        expected_output = (
            REPO_ROOT
            / "results"
            / "nvbit_capture_staging"
            / f"r28_{args.expected_head[:8]}"
            / "sol_definition_equivalence_audit.json"
        ).resolve()
        if not args.certifying or observed_output != expected_output:
            raise AuditRefusal(
                "AUDIT_OUTPUT_ROLE_MISMATCH",
                "current-HEAD certifying output must be the ignored, "
                "HEAD-versioned campaign-staging path "
                f"{expected_output}; observed {observed_output}",
            )
    else:
        expected_output = (
            DEFAULT_CERTIFYING_ARTIFACT
            if args.certifying
            else DEFAULT_PRECOMMIT_ARTIFACT
        )
        expected_output = expected_output.resolve()
        if observed_output != expected_output:
            raise AuditRefusal(
                "AUDIT_OUTPUT_ROLE_MISMATCH",
                f"expected {expected_output}, observed {observed_output}",
            )
    if args.blocking_artifact.expanduser().resolve() != DEFAULT_BLOCKING_ARTIFACT.resolve():
        raise AuditRefusal(
            "AUDIT_BLOCKING_ARTIFACT_MISMATCH",
            f"blocking artifact must be {DEFAULT_BLOCKING_ARTIFACT}",
        )
    if args.certifying:
        if args.membership_reference is None:
            raise AuditRefusal(
                "AUDIT_MEMBERSHIP_MISMATCH",
                "certifying audit requires --membership-reference",
            )
        if (
            args.membership_reference.expanduser().resolve()
            != DEFAULT_PRECOMMIT_ARTIFACT.resolve()
        ):
            raise AuditRefusal(
                "AUDIT_MEMBERSHIP_MISMATCH",
                f"membership reference must be {DEFAULT_PRECOMMIT_ARTIFACT}",
            )
    elif args.membership_reference is not None:
        raise AuditRefusal(
            "AUDIT_OUTPUT_ROLE_MISMATCH",
            "the non-certifying H1 audit cannot consume a membership reference",
        )
    elif (
        args.matrix.expanduser().resolve() != H1_MATRIX_PATH.resolve()
        or args.expected_head != H1_HEAD
        or args.expected_matrix_sha256 != H1_MATRIX_SHA256
    ):
        raise AuditRefusal(
            "AUDIT_OUTPUT_ROLE_MISMATCH",
            "the non-certifying H1 role requires the adjudicated "
            f"HEAD={H1_HEAD}, matrix={H1_MATRIX_PATH}, "
            f"sha256={H1_MATRIX_SHA256}",
        )
    elif _load_object(
        args.matrix.expanduser().resolve(), label="H1 matrix"
    ).get("matrix_seal_sha256") != H1_MATRIX_SEAL_SHA256:
        raise AuditRefusal(
            "AUDIT_OUTPUT_ROLE_MISMATCH",
            f"the H1 matrix seal must be {H1_MATRIX_SEAL_SHA256}",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocking-artifact", type=Path, default=DEFAULT_BLOCKING_ARTIFACT)
    certifying = parser.add_mutually_exclusive_group()
    certifying.add_argument("--certifying", action="store_true")
    certifying.add_argument(
        "--certifying-current-head",
        action="store_true",
        help=(
            "publish a current-HEAD certifying audit to the exact ignored "
            "results/nvbit_capture_staging/r28_<HEAD8> role"
        ),
    )
    parser.add_argument("--created-at-utc")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-matrix-sha256", required=True)
    parser.add_argument("--membership-reference", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.certifying_current_head:
        args.certifying = True
    try:
        _validate_cli_role(args)
        artifact = build_audit(
            matrix_path=args.matrix,
            blocking_artifact_path=args.blocking_artifact,
            certifying=args.certifying,
            created_at_utc=args.created_at_utc,
            expected_head=args.expected_head,
            expected_matrix_sha256=args.expected_matrix_sha256,
            membership_reference=args.membership_reference,
        )
        _validate_publication_context(artifact)
        write_audit_atomic(args.output, artifact)
    except (AuditRefusal, equiv.SolDefinitionEquivalenceError) as exc:
        print(f"audit_sol_definition_equivalence: REFUSED [{getattr(exc, 'code', 'AUDIT_ERROR')}] {getattr(exc, 'detail', str(exc))}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output.expanduser().resolve()),
                "sha256": sha256_file(args.output.expanduser().resolve()),
                "summary": artifact["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
