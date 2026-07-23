#!/usr/bin/env python3
"""Audit immutable capture chunks against one current runnable matrix.

Each historical DONE is verified against the source commit and input bundle it
actually sealed.  Only after that proof succeeds are its cell identities joined
to the current matrix.  This permits a campaign to span clean commits without
rewriting old evidence, while still rejecting duplicate, stale, extra, or
non-admitted runnable outcomes.  A current matrix that pins the SOL native
environment also upgrades the join contract: un-attested historical SOL rows
cannot count, while unrelated pre-attestation rows remain valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import campaign_done_manifest as done  # noqa: E402
import build_sm86_capture_matrix as matrix_builder  # noqa: E402
import archive_preparation_failure as preparation_archive  # noqa: E402
import archive_capture_defect_attempt as capture_defect_archive  # noqa: E402


MATRIX_SCHEMA = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "sm86_capture_matrix.schema.json"
)
EXPECTED_UNIVERSE = 344
EXPECTED_RUNNABLE = 122
EXPECTED_EXCLUDED = 222
EXPECTED_CHUNKS = 18
EXPECTED_BLOCKED = 0
EXPECTED_EXPLICIT_BUDGETS = 122
HISTORICAL_BOOTSTRAP_DONE_SHA256 = (
    "3e64d122b741d362b2945775ae73ed1200d5828e4ff273ff09f98e0bc82e2564"
)
HISTORICAL_BOOTSTRAP_CHUNK_ID = "portable_lcd-000-bootstrap"
HISTORICAL_BOOTSTRAP_IDENTITY = ("portable_lcd", "silu_fwd_bf16_sm80")


class AccountingRefusal(ValueError):
    """Typed fail-closed refusal raised before any DONE file is opened."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AccountingRefusal(
            "MATRIX_READ_FAILED", f"cannot hash matrix {path}: {exc}",
        ) from exc
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise AccountingRefusal(
            "MATRIX_REGENERATION_HEAD_UNAVAILABLE",
            f"cannot invoke git in {repo_root}: {exc}",
        ) from exc
    head = proc.stdout.strip()
    if proc.returncode != 0 or len(head) != 40:
        raise AccountingRefusal(
            "MATRIX_REGENERATION_HEAD_UNAVAILABLE",
            proc.stderr.strip() or f"invalid git HEAD {head!r}",
        )
    return head


def _unsealed_matrix(matrix: dict) -> dict:
    return {
        key: value for key, value in matrix.items()
        if key != "matrix_seal_sha256"
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _resolve_authority_path(matrix: dict, repo_root: Path) -> Path:
    raw = (matrix.get("authority") or {}).get("path")
    if not isinstance(raw, str) or not raw:
        raise AccountingRefusal(
            "MATRIX_REGENERATION_INPUT_INVALID", "authority.path is absent",
        )
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _assert_exact_cardinality(matrix: dict) -> None:
    cells = matrix.get("cells")
    chunks = matrix.get("chunks")
    counts = matrix.get("counts")
    if not isinstance(cells, list) or not isinstance(chunks, list) or not isinstance(
        counts, dict
    ):
        raise AccountingRefusal(
            "MATRIX_CARDINALITY_MISMATCH",
            "cells, chunks, and counts must have their canonical container types",
        )
    dispositions = Counter(
        row.get("disposition") if isinstance(row, dict) else None for row in cells
    )
    declared = {
        "universe": counts.get("universe"),
        "runnable": counts.get("runnable"),
        "excluded": counts.get("excluded"),
        "chunks": counts.get("chunks"),
    }
    observed = {
        "universe": len(cells),
        "runnable": dispositions.get("runnable", 0),
        "excluded": dispositions.get("excluded", 0),
        "chunks": len(chunks),
    }
    expected = {
        "universe": EXPECTED_UNIVERSE,
        "runnable": EXPECTED_RUNNABLE,
        "excluded": EXPECTED_EXCLUDED,
        "chunks": EXPECTED_CHUNKS,
    }
    unknown = {
        key: count for key, count in dispositions.items()
        if key not in {"runnable", "excluded"} and count
    }
    if declared != expected or observed != expected or unknown:
        raise AccountingRefusal(
            "MATRIX_CARDINALITY_MISMATCH",
            f"expected={expected!r} declared={declared!r} "
            f"observed={observed!r} unknown_dispositions={unknown!r}",
        )

    runnable = [
        row for row in cells
        if isinstance(row, dict) and row.get("disposition") == "runnable"
    ]
    blocked = [
        row.get("identity") for row in runnable
        if not isinstance(row.get("capture_readiness"), dict)
        or row["capture_readiness"].get("status") != "ready"
        or row["capture_readiness"].get("blocking_reasons") != []
    ]
    if (
        counts.get("capture_blocked") != EXPECTED_BLOCKED
        or counts.get("capture_ready") != EXPECTED_RUNNABLE
        or blocked
    ):
        raise AccountingRefusal(
            "MATRIX_READINESS_BLOCKER",
            f"declared_blocked={counts.get('capture_blocked')!r} "
            f"declared_ready={counts.get('capture_ready')!r} "
            f"blocked_identities={blocked!r}",
        )

    budgeted = [
        row for row in runnable
        if isinstance(row.get("trace_budget"), dict)
        and row["trace_budget"].get("status") == "declared"
        and type(row["trace_budget"].get("max_trace_bytes")) is int
        and row["trace_budget"]["max_trace_bytes"] > 0
        and type(row["trace_budget"].get("max_wall_time_s")) is int
        and row["trace_budget"]["max_wall_time_s"] > 0
    ]
    if (
        len(budgeted) != EXPECTED_EXPLICIT_BUDGETS
        or counts.get("budget_declared") != EXPECTED_EXPLICIT_BUDGETS
        or counts.get("budget_missing") != 0
    ):
        raise AccountingRefusal(
            "MATRIX_BUDGET_COVERAGE_INVALID",
            f"expected={EXPECTED_EXPLICIT_BUDGETS} observed={len(budgeted)} "
            f"declared={counts.get('budget_declared')!r} "
            f"missing={counts.get('budget_missing')!r}",
        )


def admit_matrix(
    matrix: dict, *, matrix_path: Path, repo_root: Path = REPO_ROOT,
    loaded_matrix_sha256: str | None = None,
) -> dict:
    """Fully admit current matrix authority before any DONE is opened."""
    try:
        import jsonschema
        schema = json.loads(MATRIX_SCHEMA.read_text())
        jsonschema.validate(matrix, schema)
    except Exception as exc:  # noqa: BLE001 - schema boundary is fail-closed
        raise AccountingRefusal("MATRIX_SCHEMA_INVALID", str(exc)) from exc

    claimed_seal = matrix.get("matrix_seal_sha256")
    observed_seal = matrix_builder.canonical_sha256(_unsealed_matrix(matrix))
    if claimed_seal != observed_seal:
        raise AccountingRefusal(
            "MATRIX_SEAL_MISMATCH",
            f"claimed={claimed_seal!r} observed={observed_seal}",
        )

    semantic_problems = matrix_builder.validate_matrix(matrix, repo_root=repo_root)
    if semantic_problems:
        raise AccountingRefusal(
            "MATRIX_VALIDATE_FAILED", "; ".join(semantic_problems),
        )

    recorded_head = matrix.get("source_commit")
    actual_head = _git_head(repo_root)
    if recorded_head != actual_head:
        raise AccountingRefusal(
            "MATRIX_REGENERATION_HEAD_MISMATCH",
            f"matrix source_commit={recorded_head!r} current HEAD={actual_head!r}",
        )
    authority_path = _resolve_authority_path(matrix, repo_root)
    try:
        regenerated = matrix_builder.build_matrix(
            repo_root, require_head_match=True, authority_path=authority_path,
        )
    except Exception as exc:  # noqa: BLE001 - generator refuses with rich detail
        raise AccountingRefusal(
            "MATRIX_REGENERATION_FAILED", f"{type(exc).__name__}: {exc}",
        ) from exc
    supplied_payload = _unsealed_matrix(matrix)
    regenerated_payload = _unsealed_matrix(regenerated)
    if _canonical_json_bytes(supplied_payload) != _canonical_json_bytes(
        regenerated_payload
    ):
        raise AccountingRefusal(
            "MATRIX_REGENERATION_MISMATCH",
            "canonical payload differs: supplied="
            f"{matrix_builder.canonical_sha256(supplied_payload)} regenerated="
            f"{matrix_builder.canonical_sha256(regenerated_payload)}",
        )

    _assert_exact_cardinality(matrix)
    final_matrix_sha256 = _sha256_file(matrix_path)
    if (
        loaded_matrix_sha256 is not None
        and final_matrix_sha256 != loaded_matrix_sha256
    ):
        raise AccountingRefusal(
            "MATRIX_FILE_CHANGED_DURING_ADMISSION",
            f"before={loaded_matrix_sha256} after={final_matrix_sha256}",
        )
    return {
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": final_matrix_sha256,
        "matrix_seal_sha256": claimed_seal,
        "matrix_source_commit": recorded_head,
        "accounting_tool_head": actual_head,
    }


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} is not a JSON object")
    return value


def _matrix_runnable(matrix: dict) -> dict[tuple[str, str], str]:
    rows = matrix.get("cells")
    if not isinstance(rows, list) or not rows:
        raise ValueError("capture matrix cells must be a nonempty array")
    runnable: dict[tuple[str, str], str] = {}
    all_ids: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        identity = row.get("identity") if isinstance(row, dict) else None
        source = identity.get("roster_source") if isinstance(identity, dict) else None
        slug = identity.get("slug") if isinstance(identity, dict) else None
        key = (source, slug)
        if not all(isinstance(part, str) and part for part in key):
            raise ValueError(f"matrix cells[{index}] has invalid identity {key!r}")
        if key in all_ids:
            raise ValueError(f"matrix has duplicate identity {key!r}")
        all_ids.add(key)
        if row.get("disposition") == "runnable":
            authority_sha = row.get("authority_cell_sha256")
            if not isinstance(authority_sha, str) or len(authority_sha) != 64:
                raise ValueError(f"matrix runnable {key!r} has invalid authority hash")
            runnable[key] = authority_sha
    if not runnable:
        raise ValueError("capture matrix has zero runnable cells")
    return runnable


def _matrix_chunk_members(
    matrix: dict,
) -> dict[str, set[tuple[str, str]]]:
    indexed: dict[str, set[tuple[str, str]]] = {}
    for chunk in matrix.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        indexed[chunk_id] = {
            (row.get("roster_source"), row.get("slug"))
            for row in chunk.get("cells") or [] if isinstance(row, dict)
        }
    return indexed


def _path_mentions_superseded(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    normalized = raw.replace("\\", "/")
    return "superseded" in PurePosixPath(normalized).parts


def _path_mentions_preparation_failures(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    normalized = raw.replace("\\", "/")
    return done.PREPARATION_FAILURE_ROOT_NAME in PurePosixPath(normalized).parts


def _path_mentions_capture_defects(raw: Any) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    normalized = raw.replace("\\", "/")
    return done.CAPTURE_DEFECT_ROOT_NAME in PurePosixPath(normalized).parts


def _artifact_resolves_under_superseded(output_root: Any, relative_path: Any) -> bool:
    if (
        not isinstance(output_root, str) or not Path(output_root).is_absolute()
        or not isinstance(relative_path, str) or not relative_path
    ):
        return False
    try:
        candidate = (Path(output_root) / relative_path).resolve(strict=False)
        superseded_root = (Path(output_root) / "superseded").resolve(strict=False)
        candidate.relative_to(superseded_root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _artifact_resolves_under_preparation_failures(
    output_root: Any, relative_path: Any,
) -> bool:
    if (
        not isinstance(output_root, str) or not Path(output_root).is_absolute()
        or not isinstance(relative_path, str) or not relative_path
    ):
        return False
    try:
        candidate = (Path(output_root) / relative_path).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return done.PREPARATION_FAILURE_ROOT_NAME in candidate.parts


def _artifact_resolves_under_capture_defects(
    output_root: Any, relative_path: Any,
) -> bool:
    if (
        not isinstance(output_root, str) or not Path(output_root).is_absolute()
        or not isinstance(relative_path, str) or not relative_path
    ):
        return False
    try:
        candidate = (Path(output_root) / relative_path).resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return done.CAPTURE_DEFECT_ROOT_NAME in candidate.parts


def _historical_bootstrap_is_allowlisted(
    done_path: Path, manifest: dict,
) -> bool:
    if manifest.get("chunk_id") != HISTORICAL_BOOTSTRAP_CHUNK_ID:
        return False
    results = manifest.get("cell_results")
    if not isinstance(results, list) or len(results) != 1:
        return False
    row = results[0]
    if not isinstance(row, dict) or (
        row.get("roster_source"), row.get("slug")
    ) != HISTORICAL_BOOTSTRAP_IDENTITY:
        return False
    try:
        byte_sha256 = hashlib.sha256(done_path.read_bytes()).hexdigest()
    except OSError:
        return False
    return (
        byte_sha256 == HISTORICAL_BOOTSTRAP_DONE_SHA256
        and done._is_historical_bootstrap(manifest)
    )


def _cell_result_document(manifest: dict, row: dict) -> tuple[dict | None, str | None]:
    output = manifest.get("output_root")
    if not isinstance(output, str) or not Path(output).is_absolute():
        return None, "DONE_CELL_RESULT_INVALID: output_root is not absolute"
    artifacts = row.get("artifacts")
    matches = [
        artifact for artifact in artifacts or []
        if isinstance(artifact, dict) and artifact.get("artifact_kind") == "cell_result"
    ] if isinstance(artifacts, list) else []
    if len(matches) != 1:
        return None, (
            "DONE_CELL_RESULT_INVALID: expected exactly one cell_result artifact; "
            f"found {len(matches)}"
        )
    rel = matches[0].get("relative_path")
    if not isinstance(rel, str) or not rel or _path_mentions_superseded(rel):
        return None, f"DONE_SUPERSEDED_LEAK: invalid cell_result path {rel!r}"
    if _path_mentions_preparation_failures(rel):
        return None, f"PREPARATION_ARCHIVE_LEAK: invalid cell_result path {rel!r}"
    if _path_mentions_capture_defects(rel):
        return None, f"CAPTURE_DEFECT_ARCHIVE_LEAK: invalid cell_result path {rel!r}"
    path = Path(output) / rel
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return None, f"DONE_CELL_RESULT_INVALID: cannot read {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"DONE_CELL_RESULT_INVALID: {path} is not an object"
    return value, None


def _load_supersession_ledger(
    manifest: dict, *, result_documents: list[tuple[dict, dict]],
) -> list[str]:
    """Join sealed active back-references to persistent staging ledgers.

    Superseded attempts live below the runner's external staging root, not the
    final published output root.  Consequently the only authoritative route to
    a ledger is the absolute ``superseded_path`` sealed in an active cell
    result.  Different cells may name different persistent staging roots.
    """
    active_refs = [
        (row, result.get("supersedes"))
        for row, result in result_documents
        if result.get("supersedes") is not None
    ]
    if not active_refs:
        return []
    problems: list[str] = []
    grouped: dict[Path, list[tuple[dict, dict, Path]]] = {}
    ref_fields = {
        "ledger_entry_id", "superseded_preflight_sha256", "superseded_path",
    }
    for row, ref in active_refs:
        if not isinstance(ref, dict) or set(ref) != ref_fields:
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: active supersedes object "
                "fields differ from contract"
            )
            continue
        destination_raw = ref.get("superseded_path")
        if not isinstance(destination_raw, str) or not Path(
            destination_raw
        ).is_absolute():
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: active superseded_path "
                "must be absolute"
            )
            continue
        try:
            destination = Path(destination_raw).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: active superseded_path is "
                f"unavailable: {exc}"
            )
            continue
        if str(destination) != destination_raw or not destination.is_dir():
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: active superseded_path is "
                "not a canonical directory"
            )
            continue
        superseded_root = destination.parent.parent
        if superseded_root.name != "superseded":
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: active superseded_path does "
                "not have superseded/<attempt>/<cell> shape"
            )
            continue
        ledger_path = superseded_root / "supersession_ledger.jsonl"
        grouped.setdefault(ledger_path, []).append((row, ref, destination))

    required = {
        "schema_version", "timestamp_utc", "session_id", "chunk_id", "cell_id",
        "original_preflight_sha256", "current_preflight_sha256", "source_path",
        "destination_path", "reason_code", "operator_flag", "source_commit",
        "matrix_seal_sha256", "inventory_sha256", "ledger_entry_id",
    }
    for ledger_path, refs in sorted(grouped.items(), key=lambda item: str(item[0])):
        superseded_root = ledger_path.parent
        try:
            lines = ledger_path.read_text().splitlines()
        except (OSError, UnicodeError) as exc:
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: cannot read back-referenced "
                f"ledger {ledger_path}: {exc}"
            )
            continue
        if not lines:
            problems.append(
                f"SUPERSESSION_LEDGER_INCONSISTENT: ledger is empty: {ledger_path}"
            )
            continue
        entries: dict[str, dict] = {}
        ledger_destinations: set[Path] = set()
        for line_number, raw in enumerate(lines, 1):
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: "
                    f"{ledger_path} line {line_number} is invalid JSON: {exc}"
                )
                continue
            if not isinstance(entry, dict) or set(entry) != required:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: "
                    f"{ledger_path} line {line_number} fields differ from contract"
                )
                continue
            entry_id = entry.get("ledger_entry_id")
            if not isinstance(entry_id, str) or not entry_id or entry_id in entries:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: "
                    f"{ledger_path} line {line_number} id is absent/duplicate"
                )
                continue
            entries[entry_id] = entry
            expected_common = {
                "schema_version": 1,
                "chunk_id": manifest.get("chunk_id"),
                "reason_code": "PREFLIGHT_EXPIRED_BEFORE_DONE",
                "source_commit": manifest.get("source_commit_pre"),
                "matrix_seal_sha256": manifest.get("matrix_seal_sha256"),
            }
            mismatches = [
                f"{field}={entry.get(field)!r} expected={expected!r}"
                for field, expected in expected_common.items()
                if entry.get(field) != expected
            ]
            for field in (
                "timestamp_utc", "session_id", "cell_id", "operator_flag",
            ):
                if not isinstance(entry.get(field), str) or not entry[field]:
                    mismatches.append(f"{field} must be a nonempty string")
            timestamp_problems: list[str] = []
            done._parse_utc_timestamp(
                entry.get("timestamp_utc"),
                f"{ledger_path} line {line_number} timestamp_utc",
                timestamp_problems,
            )
            problems.extend(
                f"SUPERSESSION_LEDGER_INCONSISTENT: {problem}"
                for problem in timestamp_problems
            )
            for field in (
                "original_preflight_sha256", "current_preflight_sha256",
                "source_commit", "matrix_seal_sha256", "inventory_sha256",
            ):
                expected_length = 40 if field == "source_commit" else 64
                value = entry.get(field)
                if (
                    not isinstance(value, str) or len(value) != expected_length
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    mismatches.append(f"{field} is not a lowercase hex digest")
            source_raw = entry.get("source_path")
            if not isinstance(source_raw, str) or not Path(source_raw).is_absolute():
                mismatches.append("source_path must be absolute")
            destination_raw = entry.get("destination_path")
            if not isinstance(destination_raw, str) or not Path(
                destination_raw
            ).is_absolute():
                mismatches.append("destination_path must be absolute")
                destination = None
            else:
                try:
                    destination = Path(destination_raw).resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    mismatches.append(f"destination_path is unavailable: {exc}")
                    destination = None
                else:
                    if (
                        str(destination) != destination_raw
                        or not destination.is_dir()
                        or destination.parent.parent != superseded_root
                    ):
                        mismatches.append(
                            "destination_path is noncanonical or outside its "
                            "back-referenced superseded root"
                        )
                    else:
                        ledger_destinations.add(destination)
            if mismatches:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: "
                    f"{ledger_path} line {line_number} mismatch: {mismatches!r}"
                )
            if destination is not None:
                inventory_problems: list[str] = []
                done._verify_superseded_inventory(
                    destination, expected_sha256=entry.get("inventory_sha256"),
                    label=f"{ledger_path} line {line_number}",
                    problems=inventory_problems,
                )
                problems.extend(
                    f"SUPERSESSION_LEDGER_INCONSISTENT: {problem}"
                    for problem in inventory_problems
                )

        evidence_destinations: set[Path] = set()
        try:
            attempt_entries = [
                child for child in superseded_root.iterdir()
                if child.name != "supersession_ledger.jsonl"
            ]
        except OSError as exc:
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: cannot scan persistent "
                f"superseded root {superseded_root}: {exc}"
            )
            attempt_entries = []
        for attempt in attempt_entries:
            if not attempt.is_dir() or attempt.is_symlink():
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: persistent superseded "
                    f"root contains non-directory attempt {attempt}"
                )
                continue
            try:
                children = list(attempt.iterdir())
            except OSError as exc:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: cannot scan attempt "
                    f"{attempt}: {exc}"
                )
                continue
            for child in children:
                if child.is_dir() and not child.is_symlink():
                    evidence_destinations.add(child.resolve())
                else:
                    problems.append(
                        "SUPERSESSION_LEDGER_INCONSISTENT: attempt contains "
                        f"non-directory child {child}"
                    )
        if evidence_destinations != ledger_destinations:
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: ledger destinations differ "
                f"from preserved attempts under {superseded_root}: "
                f"ledger={sorted(map(str, ledger_destinations))!r} "
                f"evidence={sorted(map(str, evidence_destinations))!r}"
            )

        referenced_ids = {
            ref.get("ledger_entry_id") for _row, ref, _destination in refs
            if isinstance(ref.get("ledger_entry_id"), str)
        }
        for destination in evidence_destinations:
            try:
                json_paths = sorted(destination.rglob("*.json"))
            except OSError:
                continue
            for path in json_paths:
                if path.name == "inventory.json":
                    continue
                try:
                    value = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError, UnicodeError):
                    continue
                stack = [value]
                while stack:
                    current = stack.pop()
                    if isinstance(current, dict):
                        supersedes = current.get("supersedes")
                        if isinstance(supersedes, dict) and isinstance(
                            supersedes.get("ledger_entry_id"), str
                        ):
                            referenced_ids.add(supersedes["ledger_entry_id"])
                        stack.extend(current.values())
                    elif isinstance(current, list):
                        stack.extend(current)
        if referenced_ids != set(entries):
            problems.append(
                "SUPERSESSION_LEDGER_INCONSISTENT: ledger/back-reference ids "
                f"differ for {ledger_path}: ledger={sorted(entries)!r} "
                f"references={sorted(referenced_ids)!r}"
            )

        for row, ref, destination in refs:
            entry = entries.get(ref.get("ledger_entry_id"))
            if entry is None:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: dangling active "
                    f"ledger_entry_id {ref.get('ledger_entry_id')!r}"
                )
                continue
            expected = {
                "original_preflight_sha256": ref.get(
                    "superseded_preflight_sha256"
                ),
                "current_preflight_sha256": manifest.get("preflight_sha256"),
                "destination_path": str(destination),
                "chunk_id": manifest.get("chunk_id"),
                "cell_id": f"{row.get('roster_source')}__{row.get('slug')}",
                "matrix_seal_sha256": manifest.get("matrix_seal_sha256"),
                "source_commit": manifest.get("source_commit_pre"),
            }
            mismatches = [
                f"{field}={entry.get(field)!r} expected={value!r}"
                for field, value in expected.items() if entry.get(field) != value
            ]
            if mismatches:
                problems.append(
                    "SUPERSESSION_LEDGER_INCONSISTENT: active back-reference "
                    f"does not match {ledger_path}: {mismatches!r}"
                )
    return problems


def _current_matrix_manifest_problems(
    *, matrix: dict, matrix_sha256: str, done_path: Path, manifest: dict,
) -> tuple[list[str], int, int, int]:
    """Enforce Round-28 current-authority join after historical seal proof."""
    problems: list[str] = []
    superseded_leaks = 0
    preparation_archive_leaks = 0
    capture_defect_archive_leaks = 0
    allowlisted_bootstrap = _historical_bootstrap_is_allowlisted(done_path, manifest)
    chunks = _matrix_chunk_members(matrix)
    chunk_id = manifest.get("chunk_id")
    expected_members = chunks.get(chunk_id)
    results = manifest.get("cell_results")
    safe_results = results if isinstance(results, list) else []
    observed_members = {
        (row.get("roster_source"), row.get("slug"))
        for row in safe_results if isinstance(row, dict)
    }
    if (
        expected_members is None
        or observed_members != expected_members
        or len(safe_results) != len(expected_members)
    ):
        problems.append(
            "DONE_CHUNK_MEMBERSHIP_MISMATCH: "
            f"chunk={chunk_id!r} expected={sorted(expected_members or set())!r} "
            f"observed={sorted(observed_members)!r}"
        )

    if not allowlisted_bootstrap:
        head_fields = {
            manifest.get("source_commit_pre"), manifest.get("source_commit_post"),
            manifest.get("source_head_at_seal"),
        }
        if head_fields != {matrix.get("source_commit")}:
            problems.append(
                "DONE_HEAD_MISMATCH: DONE source commits do not equal current "
                f"matrix source_commit={matrix.get('source_commit')!r}"
            )
        if manifest.get("capture_matrix_sha256") != matrix_sha256:
            problems.append(
                "DONE_MATRIX_MISMATCH: DONE capture_matrix_sha256="
                f"{manifest.get('capture_matrix_sha256')!r} current={matrix_sha256!r}"
            )
        if manifest.get("matrix_seal_sha256") != matrix.get("matrix_seal_sha256"):
            problems.append(
                "DONE_MATRIX_MISMATCH: DONE matrix seal does not equal current matrix"
            )

    try:
        resolved_done_path = str(done_path.resolve(strict=False))
    except (OSError, RuntimeError):
        resolved_done_path = str(done_path)
    output_raw = manifest.get("output_root")
    try:
        resolved_output = (
            str(Path(output_raw).resolve(strict=False))
            if isinstance(output_raw, str) else None
        )
    except (OSError, RuntimeError):
        resolved_output = output_raw
    if any(_path_mentions_superseded(path) for path in (
        str(done_path), resolved_done_path, output_raw, resolved_output,
    )):
        superseded_leaks += 1
        problems.append("DONE_SUPERSEDED_LEAK: DONE/output_root is under superseded/")
    if any(_path_mentions_preparation_failures(path) for path in (
        str(done_path), resolved_done_path, output_raw, resolved_output,
    )):
        preparation_archive_leaks += 1
        problems.append(
            "PREPARATION_ARCHIVE_LEAK: DONE/output_root is under "
            f"{done.PREPARATION_FAILURE_ROOT_NAME}/"
        )
    if any(_path_mentions_capture_defects(path) for path in (
        str(done_path), resolved_done_path, output_raw, resolved_output,
    )):
        capture_defect_archive_leaks += 1
        problems.append(
            "CAPTURE_DEFECT_ARCHIVE_LEAK: DONE/output_root is under "
            f"{done.CAPTURE_DEFECT_ROOT_NAME}/"
        )
    result_documents: list[tuple[dict, dict]] = []
    for index, row in enumerate(safe_results):
        if not isinstance(row, dict):
            continue
        for artifact in row.get("artifacts") or []:
            if isinstance(artifact, dict) and (
                _path_mentions_superseded(artifact.get("relative_path"))
                or _artifact_resolves_under_superseded(
                    manifest.get("output_root"), artifact.get("relative_path"),
                )
            ):
                superseded_leaks += 1
                problems.append(
                    "DONE_SUPERSEDED_LEAK: "
                    f"cell_results[{index}] publishes {artifact.get('relative_path')!r}"
                )
            if isinstance(artifact, dict) and (
                _path_mentions_preparation_failures(artifact.get("relative_path"))
                or _artifact_resolves_under_preparation_failures(
                    manifest.get("output_root"), artifact.get("relative_path"),
                )
            ):
                preparation_archive_leaks += 1
                problems.append(
                    "PREPARATION_ARCHIVE_LEAK: "
                    f"cell_results[{index}] publishes "
                    f"{artifact.get('relative_path')!r}"
                )
            if isinstance(artifact, dict) and (
                _path_mentions_capture_defects(artifact.get("relative_path"))
                or _artifact_resolves_under_capture_defects(
                    manifest.get("output_root"), artifact.get("relative_path"),
                )
            ):
                capture_defect_archive_leaks += 1
                problems.append(
                    "CAPTURE_DEFECT_ARCHIVE_LEAK: "
                    f"cell_results[{index}] publishes "
                    f"{artifact.get('relative_path')!r}"
                )
        if allowlisted_bootstrap:
            continue
        document, document_problem = _cell_result_document(manifest, row)
        if document_problem:
            problems.append(document_problem)
            if document_problem.startswith("DONE_SUPERSEDED_LEAK"):
                superseded_leaks += 1
            if document_problem.startswith("PREPARATION_ARCHIVE_LEAK"):
                preparation_archive_leaks += 1
            if document_problem.startswith("CAPTURE_DEFECT_ARCHIVE_LEAK"):
                capture_defect_archive_leaks += 1
            continue
        assert document is not None
        result_documents.append((row, document))
        if document.get("capture_preflight_sha256") != manifest.get(
            "preflight_sha256"
        ):
            problems.append(
                "DONE_PREFLIGHT_MISMATCH: "
                f"cell_results[{index}] capture_preflight_sha256 differs from DONE"
            )
    if not allowlisted_bootstrap:
        problems.extend(_load_supersession_ledger(
            manifest, result_documents=result_documents,
        ))
        problems.extend(done.verify_manifest_preparation_failure_closure(
            manifest, result_documents=result_documents,
        ))
        problems.extend(done.verify_manifest_capture_defect_closure(
            manifest, result_documents=result_documents,
        ))
    return (
        problems, superseded_leaks, preparation_archive_leaks,
        capture_defect_archive_leaks,
    )


def audit_loaded_manifests(
    matrix: dict, manifests: list[tuple[Path, dict, list[str]]], *,
    matrix_sha256: str | None = None,
) -> dict:
    """Pure current-matrix join over already-verified historical manifests."""
    runnable = _matrix_runnable(matrix)
    observations: dict[tuple[str, str], list[dict]] = {}
    seal_failures: list[dict] = []
    chunks: list[dict] = []
    superseded_leaks: list[dict] = []
    preparation_archive_leaks: list[dict] = []
    capture_defect_archive_leaks: list[dict] = []
    for done_path, manifest, verification_problems in sorted(
        manifests, key=lambda row: row[0].as_posix()
    ):
        current_matrix_attestation_problems: list[str] = []
        current_matrix_definition_attestation_problems: list[str] = []
        current_matrix_join_problems: list[str] = []
        chunk_superseded_leaks = 0
        chunk_preparation_archive_leaks = 0
        chunk_capture_defect_archive_leaks = 0
        if not verification_problems and isinstance(manifest, dict):
            current_matrix_attestation_problems = (
                done.verify_manifest_sol_native_env_attestations(manifest, matrix)
            )
            if matrix_sha256 is not None:
                current_matrix_definition_attestation_problems = (
                    done.verify_manifest_sol_definition_attestations(
                        manifest, matrix,
                    )
                )
                (
                    current_matrix_join_problems,
                    chunk_superseded_leaks,
                    chunk_preparation_archive_leaks,
                    chunk_capture_defect_archive_leaks,
                ) = (
                    _current_matrix_manifest_problems(
                        matrix=matrix, matrix_sha256=matrix_sha256,
                        done_path=done_path, manifest=manifest,
                    )
                )
        combined_problems = [
            *verification_problems,
            *(
                f"current matrix SOL env attestation: {problem}"
                for problem in current_matrix_attestation_problems
            ),
            *(
                f"current matrix SOL definition attestation: {problem}"
                for problem in current_matrix_definition_attestation_problems
            ),
            *current_matrix_join_problems,
        ]
        chunk_id = manifest.get("chunk_id") if isinstance(manifest, dict) else None
        chunks.append({
            "chunk_id": chunk_id,
            "done_path": str(done_path),
            "source_commit": manifest.get("source_commit_pre")
            if isinstance(manifest, dict) else None,
            "verification_problem_count": len(combined_problems),
            "current_matrix_attestation_problem_count": len(
                current_matrix_attestation_problems
            ),
            "current_matrix_definition_attestation_problem_count": len(
                current_matrix_definition_attestation_problems
            ),
            "current_matrix_join_problem_count": len(current_matrix_join_problems),
            "superseded_leak_count": chunk_superseded_leaks,
            "preparation_archive_leak_count": chunk_preparation_archive_leaks,
            "capture_defect_archive_leak_count": (
                chunk_capture_defect_archive_leaks
            ),
        })
        if chunk_superseded_leaks:
            superseded_leaks.append({
                "done_path": str(done_path),
                "count": chunk_superseded_leaks,
            })
        if chunk_preparation_archive_leaks:
            preparation_archive_leaks.append({
                "done_path": str(done_path),
                "count": chunk_preparation_archive_leaks,
            })
        if chunk_capture_defect_archive_leaks:
            capture_defect_archive_leaks.append({
                "done_path": str(done_path),
                "count": chunk_capture_defect_archive_leaks,
            })
        if combined_problems:
            seal_failures.append({
                "done_path": str(done_path),
                "problems": combined_problems,
            })
            continue
        for row in manifest.get("cell_results") or []:
            if not isinstance(row, dict):
                continue
            key = (row.get("roster_source"), row.get("slug"))
            observations.setdefault(key, []).append({
                "done_path": str(done_path),
                "chunk_id": chunk_id,
                "source_commit": manifest.get("source_commit_pre"),
                "authority_cell_sha256": row.get("authority_cell_sha256"),
                "disposition": row.get("disposition"),
                "typed_reason": row.get("typed_reason"),
                "capture_preflight_sha256": row.get(
                    "capture_preflight_sha256",
                    manifest.get("preflight_sha256"),
                ),
            })

    duplicate = [
        {"roster_source": key[0], "slug": key[1], "observations": rows}
        for key, rows in sorted(observations.items()) if len(rows) > 1
    ]
    extra = [
        {"roster_source": key[0], "slug": key[1], "observations": rows}
        for key, rows in sorted(observations.items()) if key not in runnable
    ]
    stale = []
    invalid_outcomes = []
    admitted: set[tuple[str, str]] = set()
    typed_unavailable: set[tuple[str, str]] = set()
    for key, expected_sha in sorted(runnable.items()):
        rows = observations.get(key, [])
        if len(rows) != 1:
            continue
        row = rows[0]
        if row["authority_cell_sha256"] != expected_sha:
            stale.append({
                "roster_source": key[0], "slug": key[1],
                "current_authority_cell_sha256": expected_sha,
                "observation": row,
            })
            continue
        if row["disposition"] == "failed" and row.get(
            "typed_reason"
        ) == "trace_unsupported":
            typed_unavailable.add(key)
            continue
        if row["disposition"] != "admitted":
            invalid_outcomes.append({
                "roster_source": key[0], "slug": key[1], "observation": row,
            })
            continue
        admitted.add(key)
    accounted = admitted | typed_unavailable
    missing = [
        {"roster_source": key[0], "slug": key[1]}
        for key in sorted(set(runnable) - accounted)
    ]
    dispositions = Counter(
        rows[0].get("disposition")
        for key, rows in observations.items()
        if key in runnable and len(rows) == 1
    )
    integrity_ok = not any((
        seal_failures, duplicate, extra, stale, invalid_outcomes,
        superseded_leaks, preparation_archive_leaks,
        capture_defect_archive_leaks,
    ))
    complete = (
        integrity_ok
        and not missing
        and not duplicate
        and not extra
        and not superseded_leaks
        and not preparation_archive_leaks
        and not capture_defect_archive_leaks
        and len(accounted) == len(runnable)
    )
    return {
        "schema_version": 1,
        "matrix_source_commit": matrix.get("source_commit"),
        "counts": {
            "current_runnable": len(runnable),
            "verified_chunks": len(chunks) - len(seal_failures),
            "seal_failed_chunks": len(seal_failures),
            "admitted_unique_current": len(admitted),
            "typed_unavailable_unique_current": len(typed_unavailable),
            "accounted_unique_current": len(accounted),
            "missing_current": len(missing),
            "duplicate_current": len(duplicate),
            "extra_or_excluded": len(extra),
            "unexpected": len(extra),
            "stale_authority": len(stale),
            "invalid_outcome_current": len(invalid_outcomes),
            "superseded_leaks": sum(row["count"] for row in superseded_leaks),
            "preparation_archive_leaks": sum(
                row["count"] for row in preparation_archive_leaks
            ),
            "capture_defect_archive_leaks": sum(
                row["count"] for row in capture_defect_archive_leaks
            ),
            "observed_dispositions": dict(sorted(dispositions.items())),
        },
        "integrity_ok": integrity_ok,
        "complete": complete,
        "chunks": chunks,
        "seal_failures": seal_failures,
        "duplicates": duplicate,
        "extra_or_excluded": extra,
        "unexpected": extra,
        "stale_authority": stale,
        "invalid_outcomes": invalid_outcomes,
        "superseded_leaks": superseded_leaks,
        "preparation_archive_leaks": preparation_archive_leaks,
        "capture_defect_archive_leaks": capture_defect_archive_leaks,
        "missing": missing,
    }


def audit_campaign(
    *, matrix_path: Path, output_root: Path, repo_root: Path = REPO_ROOT,
) -> dict:
    loaded_matrix_sha256 = _sha256_file(matrix_path)
    try:
        matrix = _load_object(matrix_path, "capture matrix")
    except ValueError as exc:
        raise AccountingRefusal("MATRIX_READ_FAILED", str(exc)) from exc
    matrix_admission = admit_matrix(
        matrix, matrix_path=matrix_path, repo_root=repo_root,
        loaded_matrix_sha256=loaded_matrix_sha256,
    )
    preparation_ledger = (
        output_root / done.PREPARATION_FAILURE_ROOT_NAME
        / preparation_archive.PREPARATION_FAILURE_LEDGER_NAME
    )
    try:
        preparation_archive.load_preparation_failure_ledger(
            preparation_ledger, verify_archives=True,
        )
    except preparation_archive.PreparationArchiveRefusal as exc:
        # Preserve the shared loader's typed FS-PRE/FS-POST recognition and
        # exact path-bearing detail at the campaign-global pre-join gate.
        raise AccountingRefusal(exc.code, exc.detail) from exc
    capture_defect_ledger = (
        output_root / done.CAPTURE_DEFECT_ROOT_NAME
        / capture_defect_archive.CAPTURE_DEFECT_LEDGER_NAME
    )
    try:
        capture_defect_rows = capture_defect_archive.load_capture_defect_ledger(
            capture_defect_ledger, verify_archives=True,
        )
    except capture_defect_archive.CaptureDefectRefusal as exc:
        raise AccountingRefusal(exc.code, exc.detail) from exc
    unresolved = capture_defect_archive.unresolved_foreign_journals(
        output_root=output_root,
        current_head=str(matrix.get("source_commit", "")),
        ledger_rows=capture_defect_rows,
    )
    if unresolved:
        raise AccountingRefusal(
            "CROSS_HEAD_ACTIVE_JOURNAL_UNRESOLVED", "; ".join(unresolved),
        )
    chunks_root = output_root / "chunks"
    manifests: list[tuple[Path, dict, list[str]]] = []
    if chunks_root.exists():
        for chunk_dir in sorted(chunks_root.iterdir()):
            if not chunk_dir.is_dir() or chunk_dir.is_symlink():
                continue
            done_path = chunk_dir / "DONE.json"
            try:
                manifest = _load_object(done_path, "DONE")
            except ValueError as exc:
                manifests.append((done_path, {}, [str(exc)]))
                continue
            problems = done.verify_seal(
                manifest, repo_root=repo_root, rehash_references=True,
                require_current_head=False, require_clean_tree=False,
            )
            manifests.append((done_path, manifest, problems))
    result = audit_loaded_manifests(
        matrix, manifests, matrix_sha256=matrix_admission["matrix_sha256"],
    )
    final_matrix_sha256 = _sha256_file(matrix_path)
    if final_matrix_sha256 != matrix_admission["matrix_sha256"]:
        raise AccountingRefusal(
            "MATRIX_FILE_CHANGED_DURING_ACCOUNTING",
            f"admitted={matrix_admission['matrix_sha256']} "
            f"final={final_matrix_sha256}",
        )
    result.update(matrix_admission)
    result["output_root"] = str(output_root.resolve())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_campaign(
            matrix_path=Path(args.matrix), output_root=Path(args.output_root),
        )
    except AccountingRefusal as exc:
        print(
            f"campaign_accounting: {exc.code}: {exc.detail}", file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(f"campaign_accounting: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        if out.exists():
            print(f"campaign_accounting: refusing existing output {out}", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    else:
        print(payload, end="")
    if not result["integrity_ok"]:
        return 1
    if args.require_complete and not result["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
