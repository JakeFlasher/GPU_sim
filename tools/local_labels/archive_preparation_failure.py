#!/usr/bin/env python3
"""Atomically preserve a sealed, pre-GPU SOL preparation failure.

This is deliberately separate from capture-journal supersession.  It verifies
the immutable Round-28 preparation attempt, moves the complete ``journals``
parent on the same filesystem, seals its inventory and copied chunk context,
then appends one durable, non-authoritative ledger row.

Importing this module has no filesystem side effects.  The public ledger and
inventory helpers are shared by the runner, DONE verifier, and accounting.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARATION_FAILURE_ROOT_NAME = "preparation_failures"
PREPARATION_FAILURE_LEDGER_NAME = "preparation_failure_ledger.jsonl"
PREPARATION_FAILURE_INVENTORY_NAME = "inventory.json"
PREPARATION_FAILURE_ENTRY_SCHEMA = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "preparation_failure_ledger_entry.schema.json"
)

RECONSTRUCTION_RECORD_SCHEMAS = {
    "legacy_comparator_reproduction.json": (
        REPO_ROOT / "workloads/simbench_bw/schemas"
        / "legacy_comparator_reproduction.schema.json"
    ),
    "runner_refusal_reconstruction.json": (
        REPO_ROOT / "workloads/simbench_bw/schemas"
        / "runner_refusal_reconstruction.schema.json"
    ),
    "phase_reconstruction.json": (
        REPO_ROOT / "workloads/simbench_bw/schemas"
        / "phase_reconstruction.schema.json"
    ),
    "gpu_process_observation_late.json": (
        REPO_ROOT / "workloads/simbench_bw/schemas"
        / "gpu_process_observation_late.schema.json"
    ),
    "preparation_failure_no_gpu_attestation_v2.json": (
        REPO_ROOT / "workloads/simbench_bw/schemas"
        / "preparation_failure_no_gpu_attestation_v2.schema.json"
    ),
}

ROUND28_SOURCE_INVENTORY_SHA256 = (
    "ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36"
)
ROUND28_SOURCE_FILE_COUNT = 22
ROUND28_SOURCE_TOTAL_BYTES = 3_855_797
SOURCE_INVENTORY_FORMAT = (
    "c_locale_sorted_relative_path_tab_size_bytes_tab_sha256_lf_v1"
)

ROUND28_ORIGINAL_HEAD = "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a"
ROUND28_MATRIX_FILE_SHA256 = (
    "fd3a1868845ceac309e718da6519f4779824b7fd69eddf1bfa0e7e7847d4e297"
)
ROUND28_MATRIX_SEAL_SHA256 = (
    "544938301f40a46713dd056af9d349c4f59848179212ef4a7826f12d9418d447"
)
ROUND28_PREFLIGHT_SHA256 = (
    "3810d2b29a06163f41d6aab73375b273e87cf462a7ad22032088802eef85b63a"
)
ROUND28_ADMISSION_RUN_ID = (
    "0ff22c451824e77fa4ec20392866c50a7356b6de4e1e986ba788bd1b5ee1a658"
)
ROUND28_SESSION_ID = "f10de1cc-612e-43e3-b51f-ce1c9985e281"
ROUND28_RLCR_SESSION_ID = "b41e0614-a371-4041-b2ed-0d0aecfda548"
ROUND28_CHUNK_ID = "sol_execbench-000-bootstrap"
ROUND28_CELL_ID = (
    "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation"
)
ROUND28_PINNED_SOURCE_DEFINITION_SHA256 = (
    "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
)
ROUND28_STAGED_DEFINITION_SHA256 = (
    "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387"
)
ROUND28_STAGED_MODULE_SHA256 = (
    "017ec9c8c977078c141e8d7d4fb58a67501550aa5540c220dbd97a94b8e1ee96"
)
ROUND28_H2_HEAD = "823f2d60c9bc5c5e78d21ff0c74f3e48ceaa1abc"
ROUND28_H2_COMMITTED_AT_UTC = "2026-07-10T20:15:42Z"
ROUND28_H3_SUBJECT = (
    "fix(local_labels): rebind prep070 refusal authority to legacy "
    "9b8e0e40 comparator"
)
ROUND28_PREPARE_MANIFEST_SHA256 = (
    "d9c2fd5e55a9edf860ff23c2b2b018e4c63800adb47d26174ff664c0bf20c7e7"
)
ROUND28_JOURNAL_START_SHA256 = (
    "6b6dcce8cb7f4a21385c6e52c693f7c398258877ec702090438ee67bea8faa4f"
)
ROUND28_PREPARE_STDOUT_SHA256 = (
    "ca3249b38c8c5dc3c8d428aed3aa8c650978d629903640befb46517aed7d8cdc"
)
ROUND28_PREPARE_STDOUT_MTIME_UTC = "2026-07-10T13:34:52.699087829Z"
ROUND28_PREPARE_STDERR_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
ROUND28_SOURCE_DEFINITION_PATH = (
    REPO_ROOT / "tools/sol_execbench/data/benchmark/L1"
    / "070_mamba2_fused_intra_chunk_diagonal_computation/definition.json"
)
ROUND28_MANIFEST_JOURNAL_RELATIVE_PATH = (
    f"cells/{ROUND28_CELL_ID}/sol_stage/capture_prepare_manifest.json"
)
ROUND28_STAGED_DEFINITION_JOURNAL_RELATIVE_PATH = (
    f"cells/{ROUND28_CELL_ID}/sol_stage/definition.json"
)
ROUND28_LEGACY_COMPARATOR_PATH = "tools/local_labels/sm86_capture_chunk.py"
ROUND28_LEGACY_COMPARATOR_FILE_SHA256 = (
    "1dfbe3ec5ed19d6dff234616141fbd3c601adc88d20709860055380a47ea5447"
)
ROUND28_LEGACY_COMPARATOR_EXCERPT_SHA256 = (
    "cbc95368ce58e4f7454e4f39a87a85c04fce2baa83546ca5af08eb38025d124c"
)
ROUND28_LEGACY_COMPARATOR_LINE_SPAN = (1306, 1317)
ROUND28_LEGACY_RULE_ID = "SOL_DEF_LEGACY_COMPARE_9B8E0E40"
ROUND28_LEGACY_AUTHORITY_LITERAL = (
    "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_"
    "sealed_bytes"
)
ROUND28_AMENDMENT_ID = "ANNEX_C_AMEND_R28_PREP070_RECON_V2"
ROUND28_CORRECTION_ID = "R28_PREP070_RECON_V2_CORR1"
ROUND28_CORRECTION_SHA256 = (
    "01b57749e29627fee4a84d6ecb491744aa9d6e1e10adf3ddf865ddbabd9f0196"
)
ROUND28_CORRECTION2_ID = "R28_PREP070_RECON_V2_CORR2"
ROUND28_CORRECTION2_SHA256 = (
    "e039a9a23883869fb9cdba207540667ea439201a3eacadc77b5fceb5e1b4230b"
)
ROUND28_ANNEX_C_SHA256 = (
    "b9bc56d06ddbed14ca28df02012077a533c9b846a90f2f08ffdcfc9c6c52b568"
)
ROUND28_V1_AUDIT_SHA256 = (
    "b6dd22ad6e7a87033374262aa8cd6a8a3ea97ac901a5e1c266a3a6d7341cdcd0"
)
ROUND28_V2_H1_AUDIT_SHA256 = (
    "ea51b0c6da89f5d339cab365ed9bfc69cb642a44dd73dc2fabe9d5dafafcb252"
)
ROUND28_RLCR_DIR = Path(
    "/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/"
    ".humanize/rlcr/2026-07-08_11-39-13"
)
ROUND28_EVIDENCE_ROOT = (
    REPO_ROOT / "evidence/round28/preparation_failures/problem-070"
)
ROUND28_EVIDENCE_MANIFEST = REPO_ROOT / "evidence/MANIFEST.sha256"
ROUND28_PATH_ALIASES = REPO_ROOT / "evidence/path_aliases.v1.json"
ROUND28_RECONSTRUCTION_ROOT = ROUND28_EVIDENCE_ROOT / "prep070_reconstruction"
ROUND28_SESSION_LOG = ROUND28_EVIDENCE_ROOT / "session_log.jsonl"
ROUND28_CORRECTED_ADJUDICATION = (
    ROUND28_EVIDENCE_ROOT / "adjudication_r28_prep070_reconstruction_v2.md"
)
ROUND28_SUPERSEDED_ADJUDICATION = (
    ROUND28_EVIDENCE_ROOT
    / "adjudication_r28_prep070_reconstruction_superseded_v1.md"
)
ROUND28_CORRECTION_ADJUDICATION = (
    ROUND28_EVIDENCE_ROOT
    / "adjudication_r28_prep070_reproduction_contract_correction.md"
)
ROUND28_CORRECTION2_ADJUDICATION = (
    ROUND28_EVIDENCE_ROOT
    / "adjudication_r28_post_move_failure_state_correction.md"
)
ROUND28_STAGE_EQUIVALENCE_ADJUDICATION = (
    ROUND28_RLCR_DIR / "adjudication_r28_stage_equivalence.md"
)
ROUND28_STAGE_EQUIVALENCE_SOURCE = (
    ROUND28_EVIDENCE_ROOT / "adjudication_r28_stage_equivalence.md"
)
ROUND28_PROBLEM_PACKAGER_SOURCE = (
    REPO_ROOT / "tools/sol_execbench/src/sol_execbench/driver"
    / "problem_packager.py"
)
ROUND28_RECONSTRUCTION_RECORD_ORDER = (
    "legacy_comparator_reproduction.json",
    "runner_refusal_reconstruction.json",
    "phase_reconstruction.json",
    "gpu_process_observation_late.json",
    "preparation_failure_no_gpu_attestation_v2.json",
)
ROUND28_RECONSTRUCTION_CONTEXT_FILES = (
    *ROUND28_RECONSTRUCTION_RECORD_ORDER,
    "legacy_comparator_source_excerpt.py.txt",
    "gpu_process_observation_late.raw.txt",
)
ROUND28_AUTHORITY_BASIS = (
    "annex_c_adjudication",
    ROUND28_LEGACY_AUTHORITY_LITERAL,
)
CONTEXT_MODE_CONTEMPORANEOUS = "contemporaneous"
CONTEXT_MODE_RECONSTRUCTED = "reconstructed"
ROUND28_CAMPAIGN_ROOT = Path("/home/jakeshea/sm86_reference_capture_r27")
ROUND28_PREPARATION_FAILURES_ROOT = (
    ROUND28_CAMPAIGN_ROOT / PREPARATION_FAILURE_ROOT_NAME
)
ROUND28_JOURNALS_ROOT = (
    ROUND28_CAMPAIGN_ROOT / ".staging" / ROUND28_ORIGINAL_HEAD
    / ROUND28_CHUNK_ID / ROUND28_MATRIX_FILE_SHA256 / "journals"
)
PREPARATION_FAILURE_REASON = "STAGE_SEMANTIC_VALIDATION_FAILED"
PREPARATION_FAILURE_RUNNER_REFUSAL = "sol_stage_source_mismatch"
PREPARATION_FAILURE_OPERATOR_FLAG = "--archive-preparation-failure"
PREPARATION_FAILURE_OVERRIDE_FLAG = "--override-preparation-attempt-limit"
PREPARATION_FAILURE_LIMIT = 2
DEFECT_SUMMARY = "comparator lacked absent≡null for optional fields"

PREPARATION_FAILURE_ENTRY_FIELDS = frozenset({
    "schema_version", "entry_id", "entry_type", "created_at_utc",
    "rlcr_session_id", "rlcr_dir", "campaign_round", "chunk_id",
    "cell_id", "reason_code", "runner_refusal_code", "original_head",
    "original_matrix_file_sha256", "original_matrix_seal_sha256",
    "original_preflight_sha256", "original_admission_run_id",
    "original_session_id", "replacement_head", "source_journal_path",
    "archive_path", "inventory_file", "inventory_sha256",
    "no_gpu_attestation", "pinned_source_definition_sha256",
    "staged_definition_sha256", "staged_module_sha256", "defect_summary",
    "adjudication_ref", "operator_flag",
})
PREPARATION_FAILURE_RECONSTRUCTED_ENTRY_FIELDS = frozenset({
    *PREPARATION_FAILURE_ENTRY_FIELDS,
    "supersedes", "reconstruction_records", "authority",
})
PREPARATION_FAILURE_BACKREF_FIELDS = frozenset({
    "ledger_entry_id", "ledger_path", "archive_path",
})
INVENTORY_FIELDS = frozenset({
    "schema_version", "source_inventory_format", "source_inventory_sha256",
    "source_file_count", "source_total_bytes", "files",
    "chunk_context_files",
})
FILE_INVENTORY_FIELDS = frozenset({"relative_path", "size_bytes", "sha256"})
CONTEXT_INVENTORY_FIELDS = frozenset({
    "relative_path", "source_path", "size_bytes", "sha256",
})

# These patterns describe *execution evidence*.  The staged eval_driver.py and
# prepare request are permitted because packaging them does not execute them.
FORBIDDEN_ARTIFACT_PATTERNS = (
    "**/sol_stage/evidence/*-selection.json",
    "**/sol_stage/evidence/*-result.json",
    "**/sol_requests/label*.json",
    "**/sol_requests/trace*.json",
    "**/sol_requests/control*.json",
    "**/sol_requests/smoke*.json",
    "**/logs/sol_label*",
    "**/logs/label*",
    "**/logs/trace*",
    "**/logs/*ncu*",
    "**/logs/*nvbit*",
    "**/logs/*control*",
    "**/*.ncu-rep",
    "**/*.ncu-report",
    "**/ncu/**",
    "**/nvbit/**",
    "**/threadblocks/**",
    "**/dynamic_trace.pb",
    "**/stats.csv",
    "**/*decoded*record*",
    "**/*control*capture*",
    "**/*control*evidence*",
    "**/*accepted*launch*",
    "**/*launch*record*",
)

_NO_GPU_EVIDENCE_FIELDS = {
    "journal_state": {
        "evidence_path", "sha256", "state", "journal_complete_present",
    },
    "prepare_stdout": {
        "evidence_path", "sha256",
        "contains_only_validated_prepare_manifest",
    },
    "prepare_stderr": {"evidence_path", "sha256", "empty"},
    "forbidden_artifact_scan": {
        "evidence_path", "sha256", "checked_patterns", "matches",
        "scanned_file_count",
    },
    "runner_refusal": {
        "evidence_path", "sha256", "runner_refusal_code", "refused_at_utc",
    },
    "phase_log": {
        "evidence_path", "sha256", "last_phase", "post_prepare_phases",
    },
    "problem_packager_execute": {
        "evidence_path", "sha256", "source_head", "function", "citation",
        "writes_driver_and_returns_command_without_execution",
    },
    "gpu_process_observation": {
        "evidence_path", "sha256", "remaining_gpu_processes",
    },
}


class PreparationArchiveRefusal(RuntimeError):
    """Stable typed refusal shared across archive producers and consumers."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8"))


def _utc_text(value: datetime | None = None) -> str:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "timestamp is timezone-naive",
        )
    return observed.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label} is not UTC Z time",
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: {exc}",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label} is not UTC",
        )
    return parsed


def _is_sha(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str) and len(value) == length
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_uuid4(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _safe_relative(value: Any, *, label: str) -> Path:
    path = Path(value) if isinstance(value, str) else Path(".")
    if (
        not isinstance(value, str) or not value or path.is_absolute()
        or path == Path(".") or ".." in path.parts
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label} is not safe-relative",
        )
    return path


def _json_object(path: Path, *, code: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationArchiveRefusal(code, f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationArchiveRefusal(code, f"{path}: root is not object")
    return value


def _validate_json_schema(
    value: Mapping[str, Any], *, schema_path: Path, code: str,
) -> None:
    try:
        import jsonschema

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except (ImportError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationArchiveRefusal(code, f"{schema_path}: {exc}") from exc
    except Exception as exc:
        if exc.__class__.__module__.split(".", 1)[0] != "jsonschema":
            raise
        raise PreparationArchiveRefusal(code, f"{schema_path}: {exc}") from exc


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("TO_BE_SEALED")
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    return False


def _git_bytes(repo_root: Path, *args: str, code: str) -> bytes:
    proc = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise PreparationArchiveRefusal(code, f"git {' '.join(args)}: {detail}")
    return proc.stdout


def _git_text(repo_root: Path, *args: str, code: str) -> str:
    try:
        return _git_bytes(repo_root, *args, code=code).decode("utf-8").strip()
    except UnicodeError as exc:
        raise PreparationArchiveRefusal(code, f"git {' '.join(args)}: {exc}") from exc


def _stat_mtime_utc(path: Path) -> str:
    observed_ns = path.stat().st_mtime_ns
    seconds, nanoseconds = divmod(observed_ns, 1_000_000_000)
    prefix = datetime.fromtimestamp(seconds, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    return f"{prefix}.{nanoseconds:09d}Z"


def _record_time() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _write_compact_json_exclusive_fsync(
    path: Path, payload: Mapping[str, Any], *, code: str,
) -> None:
    if _contains_placeholder(payload):
        raise PreparationArchiveRefusal(code, f"placeholder remains in {path}")
    raw = (json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PreparationArchiveRefusal(code, f"exists: {path}") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH", f"exists: {path}",
        )
    raw = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _append_jsonl_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def content_inventory(root: Path) -> list[dict]:
    """Return a non-vacuous, symlink-free inventory sorted like LC_ALL=C."""
    if root.is_symlink() or not root.is_dir():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", str(root),
        )
    rows: list[dict] = []
    for path in sorted(root.rglob("*"), key=lambda p: os.fsencode(
        p.relative_to(root).as_posix()
    )):
        if path.is_symlink():
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_SOURCE_MISSING", f"symlink: {path}",
            )
        if path.is_file():
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    if not rows:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", f"empty inventory: {root}",
        )
    return rows


def source_inventory_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize the immutable C-locale Round-28 inventory pin domain."""
    ordered = sorted(rows, key=lambda row: os.fsencode(str(row["relative_path"])))
    return b"".join(
        f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n".encode(
            "utf-8"
        )
        for row in ordered
    )


def source_inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(source_inventory_bytes(rows))


def _validate_file_rows(rows: Any, *, context: bool) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "inventory rows are empty/malformed",
        )
    fields = CONTEXT_INVENTORY_FIELDS if context else FILE_INVENTORY_FIELDS
    observed: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != fields:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"inventory row {index} has non-closed fields",
            )
        rel = _safe_relative(row.get("relative_path"), label=f"row {index}")
        rel_text = rel.as_posix()
        if rel_text in seen or (context and rel.parts[0] != "chunk_context"):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"inventory row {index} path is duplicate/outside context",
            )
        if (
            type(row.get("size_bytes")) is not int or row["size_bytes"] < 0
            or not _is_sha(row.get("sha256"))
            or (context and (
                not isinstance(row.get("source_path"), str)
                or not row["source_path"]
            ))
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"inventory row {index} metadata is invalid",
            )
        seen.add(rel_text)
        observed.append(dict(row))
    expected = sorted(observed, key=lambda row: os.fsencode(row["relative_path"]))
    if observed != expected:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "inventory rows are not C-sorted",
        )
    return observed


def load_archive_inventory(path: Path) -> dict:
    inventory = _json_object(path, code="PREPARATION_LEDGER_INCONSISTENT")
    if set(inventory) != INVENTORY_FIELDS or inventory.get("schema_version") != 1:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"closed inventory fields: {path}",
        )
    files = _validate_file_rows(inventory.get("files"), context=False)
    contexts = _validate_file_rows(
        inventory.get("chunk_context_files"), context=True,
    )
    if (
        inventory.get("source_inventory_format") != SOURCE_INVENTORY_FORMAT
        or inventory.get("source_inventory_sha256")
        != source_inventory_sha256(files)
        or inventory.get("source_file_count") != len(files)
        or inventory.get("source_total_bytes")
        != sum(row["size_bytes"] for row in files)
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"source inventory seal: {path}",
        )
    return {**inventory, "files": files, "chunk_context_files": contexts}


def verify_preparation_archive(archive_path: Path, inventory: dict) -> None:
    """Rehash both the moved journal and copied context, closing the tree."""
    if archive_path.is_symlink() or not archive_path.is_dir():
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"archive absent: {archive_path}",
        )
    files = inventory.get("files") or []
    journal_candidates = [
        child for child in archive_path.iterdir()
        if child.name not in {PREPARATION_FAILURE_INVENTORY_NAME, "chunk_context"}
    ]
    if (
        len(journal_candidates) != 1 or journal_candidates[0].is_symlink()
        or not journal_candidates[0].is_dir()
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            "archive must contain exactly one regular journal directory",
        )
    try:
        actual_files = content_inventory(journal_candidates[0])
    except PreparationArchiveRefusal as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", exc.detail,
        ) from exc
    if actual_files != files:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "archive journal inventory mismatch",
        )
    expected_context = {
        row["relative_path"]: {
            key: row[key] for key in FILE_INVENTORY_FIELDS
        }
        for row in inventory.get("chunk_context_files") or []
    }
    context_root = archive_path / "chunk_context"
    actual_context: dict[str, dict] = {}
    if context_root.is_symlink() or not context_root.is_dir():
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "chunk_context is absent/invalid",
        )
    for path in sorted(context_root.rglob("*")):
        if path.is_symlink():
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"context symlink: {path}",
            )
        if path.is_file():
            rel = path.relative_to(archive_path).as_posix()
            actual_context[rel] = {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if actual_context != expected_context:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "chunk context inventory mismatch",
        )


def _validate_no_gpu_attestation(value: Any, *, label: str) -> None:
    fields = {
        "journal_state", "prepare_stdout", "prepare_stderr",
        "forbidden_artifact_scan", "runner_refusal", "phase_log",
        "problem_packager_execute", "gpu_process_observation",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label} fields are not closed",
        )
    evidence_paths: set[str] = set()
    for name, evidence in value.items():
        if (
            not isinstance(evidence, dict)
            or set(evidence) != _NO_GPU_EVIDENCE_FIELDS[name]
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}.{name} fields are not closed",
            )
        if name == "forbidden_artifact_scan":
            if (
                evidence.get("checked_patterns") != list(FORBIDDEN_ARTIFACT_PATTERNS)
                or evidence.get("matches") != []
            ):
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT", f"{label}.{name} is not clean",
                )
        evidence_path = evidence.get("evidence_path")
        evidence_sha = evidence.get("sha256")
        if (
            not isinstance(evidence_path, str) or not Path(evidence_path).is_absolute()
            or evidence_path in evidence_paths or not _is_sha(evidence_sha)
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}.{name} lacks absolute evidence path/hash",
            )
        evidence_paths.add(evidence_path)
    expected_scalars = {
        "journal_state": {
            "state": "capture_started", "journal_complete_present": False,
        },
        "prepare_stdout": {
            "contains_only_validated_prepare_manifest": True,
        },
        "prepare_stderr": {"empty": True},
        "runner_refusal": {
            "runner_refusal_code": PREPARATION_FAILURE_RUNNER_REFUSAL,
        },
        "phase_log": {
            "last_phase": "prepare_validate", "post_prepare_phases": [],
        },
        "problem_packager_execute": {
            "function": "ProblemPackager.execute",
            "writes_driver_and_returns_command_without_execution": True,
        },
        "gpu_process_observation": {"remaining_gpu_processes": []},
    }
    for name, expected in expected_scalars.items():
        evidence = value[name]
        if any(evidence.get(key) != expected_value for key, expected_value in expected.items()):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}.{name} semantic assertion mismatch",
            )
    if (
        type(value["forbidden_artifact_scan"].get("scanned_file_count")) is not int
        or value["forbidden_artifact_scan"]["scanned_file_count"] <= 0
        or not isinstance(value["runner_refusal"].get("refused_at_utc"), str)
        or not isinstance(value["problem_packager_execute"].get("citation"), str)
        or not value["problem_packager_execute"]["citation"]
        or not _is_sha(value["problem_packager_execute"].get("source_head"), 40)
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label} no-GPU metadata invalid",
        )
    _parse_utc(
        value["runner_refusal"]["refused_at_utc"],
        label=f"{label}.runner_refusal.refused_at_utc",
    )


def _validate_no_gpu_attestation_v2(value: Any, *, label: str) -> None:
    fields = {
        "record_type", "record_version", "amendment_id", "cell", "chunk",
        "attestation", "certifying_elements", "journal_forensic_absence_scan",
        "non_certifying_elements", "late_observation_reference",
        "reproduction_reference", "campaign_rows",
        "capture_supersession_attempts_consumed",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: v2 fields not closed",
        )
    expected = {
        "record_type": "preparation_failure_no_gpu_attestation",
        "record_version": 2,
        "amendment_id": ROUND28_AMENDMENT_ID,
        "cell": ROUND28_CELL_ID,
        "chunk": ROUND28_CHUNK_ID,
        "certifying_elements": [
            "annex_c_adjudication", ROUND28_LEGACY_AUTHORITY_LITERAL,
            "journal_forensic_absence_scan",
        ],
        "non_certifying_elements": ["gpu_process_observation_late"],
        "campaign_rows": 0,
        "capture_supersession_attempts_consumed": 0,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: v2 constants drift",
        )
    scan = value.get("journal_forensic_absence_scan")
    if scan != {
        "forbidden_artifact_scan": [],
        "journal_complete_present": False,
        "prepare_stderr_sha256": ROUND28_PREPARE_STDERR_SHA256,
        "prepare_stderr_empty": True,
        "inventory_files": ROUND28_SOURCE_FILE_COUNT,
        "inventory_bytes": ROUND28_SOURCE_TOTAL_BYTES,
        "inventory_domain_sha256": ROUND28_SOURCE_INVENTORY_SHA256,
    }:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: forensic scan drift",
        )
    late = value.get("late_observation_reference")
    reproduction = value.get("reproduction_reference")
    if (
        not isinstance(late, dict)
        or set(late) != {"file", "sha256", "certifying"}
        or late.get("file") != "gpu_process_observation_late.json"
        or not _is_sha(late.get("sha256")) or late.get("certifying") is not False
        or not isinstance(reproduction, dict)
        or set(reproduction) != {"file", "sha256"}
        or reproduction.get("file") != "legacy_comparator_reproduction.json"
        or not _is_sha(reproduction.get("sha256"))
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: references drift",
        )


def _validate_reconstructed_entry_extensions(
    row: Mapping[str, Any], *, label: str,
) -> None:
    expected_tuple = {
        "rlcr_session_id": ROUND28_RLCR_SESSION_ID,
        "chunk_id": ROUND28_CHUNK_ID,
        "cell_id": ROUND28_CELL_ID,
        "original_head": ROUND28_ORIGINAL_HEAD,
        "original_matrix_file_sha256": ROUND28_MATRIX_FILE_SHA256,
        "original_matrix_seal_sha256": ROUND28_MATRIX_SEAL_SHA256,
        "original_preflight_sha256": ROUND28_PREFLIGHT_SHA256,
        "original_admission_run_id": ROUND28_ADMISSION_RUN_ID,
        "original_session_id": ROUND28_SESSION_ID,
        "pinned_source_definition_sha256": (
            ROUND28_PINNED_SOURCE_DEFINITION_SHA256
        ),
        "staged_definition_sha256": ROUND28_STAGED_DEFINITION_SHA256,
        "staged_module_sha256": ROUND28_STAGED_MODULE_SHA256,
        "rlcr_dir": str(ROUND28_RLCR_DIR.resolve()),
        "source_journal_path": str(
            ROUND28_JOURNALS_ROOT.resolve() / ROUND28_CELL_ID
        ),
        "adjudication_ref": str(
            (ROUND28_RLCR_DIR / "adjudication_r28_stage_equivalence.md").resolve()
        ),
        "operator_flag": PREPARATION_FAILURE_OPERATOR_FLAG,
    }
    drift = [
        field for field, expected in expected_tuple.items()
        if row.get(field) != expected
    ]
    if drift:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"{label}: reconstructed incident tuple drift: {drift}",
        )
    if row.get("supersedes") is not None:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: supersedes not null",
        )
    authority = row.get("authority")
    authority_fields = {
        "amendment_id", "supersession_status",
        "corrected_adjudication_sha256", "superseded_adjudication_sha256",
        "correction_id", "correction_adjudication_sha256",
    }
    if not isinstance(authority, dict) or set(authority) != authority_fields:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: authority fields",
        )
    if (
        authority.get("amendment_id") != ROUND28_AMENDMENT_ID
        or authority.get("supersession_status") != "superseded_by_v2"
        or authority.get("correction_id") != ROUND28_CORRECTION_ID
        or authority.get("correction_adjudication_sha256")
        != ROUND28_CORRECTION_SHA256
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: authority constants",
        )
    authority_shas = [
        authority.get("corrected_adjudication_sha256"),
        authority.get("superseded_adjudication_sha256"),
        authority.get("correction_adjudication_sha256"),
    ]
    if not all(_is_sha(value) for value in authority_shas) or len(
        set(authority_shas)
    ) != 3:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: authority SHA relation",
        )
    records = row.get("reconstruction_records")
    expected_paths = [
        f"chunk_context/reconstructed/{name}"
        for name in ROUND28_RECONSTRUCTION_RECORD_ORDER
    ]
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: record cardinality",
        )
    for index, (record, expected_path) in enumerate(zip(records, expected_paths)):
        if (
            not isinstance(record, dict) or set(record) != {"path", "sha256"}
            or record.get("path") != expected_path
            or not _is_sha(record.get("sha256"))
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: reconstruction record {index}",
            )
    _validate_no_gpu_attestation_v2(
        row.get("no_gpu_attestation"), label=f"{label}.no_gpu_attestation",
    )


def _verify_reconstructed_session_log(
    row: Mapping[str, Any], *, records: Mapping[str, Mapping[str, Any]],
) -> None:
    authority = row["authority"]
    rlcr_dir = Path(row["rlcr_dir"])
    log_path = rlcr_dir / "session_log.jsonl"
    expected = [
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_prep070_reproduction_contract_correction.md",
            "sha256": authority["correction_adjudication_sha256"],
            "status": "binding_supplement",
            "supplements_amendment_id": ROUND28_AMENDMENT_ID,
            "correction_id": ROUND28_CORRECTION_ID,
        },
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_post_move_failure_state_correction.md",
            "sha256": ROUND28_CORRECTION2_SHA256,
            "status": "binding_supplement",
            "supplements_amendment_id": ROUND28_AMENDMENT_ID,
            "correction_id": ROUND28_CORRECTION2_ID,
        },
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_prep070_reconstruction_superseded_v1.md",
            "sha256": authority["superseded_adjudication_sha256"],
            "status": "superseded_non_authoritative",
            "superseded_by_sha256": authority["corrected_adjudication_sha256"],
            "defect": (
                "false_certifying_premise:"
                "deterministic_v1_comparator_over_sealed_bytes"
            ),
        },
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_prep070_reconstruction_v2.md",
            "sha256": authority["corrected_adjudication_sha256"],
            "status": "binding",
            "supersedes_sha256": authority["superseded_adjudication_sha256"],
            "amendment_id": ROUND28_AMENDMENT_ID,
            "h3": row["replacement_head"],
        },
    ]
    record_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in row["reconstruction_records"]
    }
    expected.extend([
        _record_event(
            "legacy_comparator_source_excerpt.py.txt",
            records["runner_refusal_reconstruction.json"]
            ["legacy_comparator_identity"]["line_span_sha256"],
        ),
        _record_event(
            "legacy_comparator_reproduction.json",
            record_hashes["legacy_comparator_reproduction.json"],
        ),
        _record_event(
            "runner_refusal_reconstruction.json",
            record_hashes["runner_refusal_reconstruction.json"],
        ),
        _record_event(
            "phase_reconstruction.json",
            record_hashes["phase_reconstruction.json"],
        ),
        _record_event(
            "gpu_process_observation_late.raw.txt",
            records["gpu_process_observation_late.json"]["raw_output_sha256"],
        ),
        _record_event(
            "gpu_process_observation_late.json",
            record_hashes["gpu_process_observation_late.json"],
        ),
        _record_event(
            "preparation_failure_no_gpu_attestation_v2.json",
            record_hashes["preparation_failure_no_gpu_attestation_v2.json"],
        ),
        {
            "event": "preparation_failure_archived",
            "entry_id": row["entry_id"],
            "archive_path": row["archive_path"],
            "inventory_sha256": row["inventory_sha256"],
        },
    ])
    observed = _load_session_log(log_path)
    if len(observed) < len(expected) or observed[:len(expected)] != expected:
        raise PreparationArchiveRefusal(
            "RLCR_SESSION_LOG_INCONSISTENT",
            f"archived prefix mismatch: rows={len(observed)}",
        )


def _verify_reconstructed_archive_entry(
    row: Mapping[str, Any], *, archive: Path, inventory: Mapping[str, Any],
    label: str, require_archive_log_event: bool,
) -> None:
    _refuse_reconstruction_contamination(archive)
    records: dict[str, dict] = {}
    for item, name in zip(
        row["reconstruction_records"], ROUND28_RECONSTRUCTION_RECORD_ORDER,
    ):
        path = archive / item["path"]
        if (
            path.is_symlink() or not path.is_file()
            or sha256_file(path) != item["sha256"]
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: reconstruction record hash drift: {path}",
            )
        value = _json_object(path, code="PREPARATION_LEDGER_INCONSISTENT")
        try:
            _validate_reconstruction_record(name, value)
        except PreparationArchiveRefusal as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", exc.detail,
            ) from exc
        records[name] = value
    embedded = row["no_gpu_attestation"]
    if records["preparation_failure_no_gpu_attestation_v2.json"] != embedded:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"{label}: embedded/standalone attestation inequality",
        )
    runner = records["runner_refusal_reconstruction.json"]
    phase = records["phase_reconstruction.json"]
    gpu = records["gpu_process_observation_late.json"]
    reproduction = records["legacy_comparator_reproduction.json"]
    authority = row["authority"]
    if (
        runner["authority_citations"]["corrected_adjudication_sha256"]
        != authority["corrected_adjudication_sha256"]
        or runner["authority_citations"]["superseded_adjudication_sha256"]
        != authority["superseded_adjudication_sha256"]
        or reproduction["tool"]["head"] != row["replacement_head"]
        or runner["authority_basis"] != list(ROUND28_AUTHORITY_BASIS)
        or phase["phases"][2]["evidence_basis"]
        != list(ROUND28_AUTHORITY_BASIS)
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: authority cross-link drift",
        )
    cutoff = _parse_utc(
        ROUND28_H2_COMMITTED_AT_UTC, label="ROUND28_H2_COMMITTED_AT_UTC",
    )
    observed_times: list[datetime] = []
    for name, value in (
        ("executed_at", reproduction["executed_at"]),
        ("runner reconstructed_at", runner["reconstructed_at"]),
        ("phase reconstructed_at", phase["reconstructed_at"]),
        ("observed_at", gpu["observed_at"]),
    ):
        parsed = _parse_utc(value, label=name)
        if parsed <= cutoff:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: {name} does not postdate H2",
            )
        observed_times.append(parsed)
    if observed_times != sorted(observed_times):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"{label}: reconstructed event chronology drift",
        )
    legacy_blob = _git_bytes(
        REPO_ROOT, "show",
        f"{ROUND28_ORIGINAL_HEAD}:{ROUND28_LEGACY_COMPARATOR_PATH}",
        code="PREPARATION_LEDGER_INCONSISTENT",
    )
    start, end = ROUND28_LEGACY_COMPARATOR_LINE_SPAN
    legacy_excerpt = b"".join(legacy_blob.splitlines(keepends=True)[start - 1:end])
    tool_blob = _git_bytes(
        REPO_ROOT, "show",
        f"{row['replacement_head']}:tools/local_labels/legacy_compare_9b8e0e40.py",
        code="PREPARATION_LEDGER_INCONSISTENT",
    )
    command_argv = reproduction["command_argv"]
    excerpt_path = archive / "chunk_context/reconstructed/legacy_comparator_source_excerpt.py.txt"
    raw_path = archive / "chunk_context/reconstructed/gpu_process_observation_late.raw.txt"
    scratch_source = archive / "chunk_context/reconstructed/scratch/source_definition.json"
    scratch_staged = archive / "chunk_context/reconstructed/scratch/staged_definition.json"
    if (
        sha256_bytes(legacy_blob) != ROUND28_LEGACY_COMPARATOR_FILE_SHA256
        or sha256_bytes(legacy_excerpt) != ROUND28_LEGACY_COMPARATOR_EXCERPT_SHA256
        or runner["legacy_comparator_identity"]["file_sha256"]
        != ROUND28_LEGACY_COMPARATOR_FILE_SHA256
        or runner["legacy_comparator_identity"]["line_span_sha256"]
        != ROUND28_LEGACY_COMPARATOR_EXCERPT_SHA256
        or reproduction["tool"]["sha256"] != sha256_bytes(tool_blob)
        or reproduction["tool"]["head"] != row["replacement_head"]
        or command_argv[10] != reproduction["tool"]["head"]
        or sha256_file(excerpt_path) != ROUND28_LEGACY_COMPARATOR_EXCERPT_SHA256
        or sha256_file(raw_path) != gpu["raw_output_sha256"]
        or sha256_file(scratch_source) != ROUND28_PINNED_SOURCE_DEFINITION_SHA256
        or sha256_file(scratch_staged) != ROUND28_STAGED_DEFINITION_SHA256
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: reconstructed context drift",
        )
    record_hashes = {
        name: next(
            item["sha256"] for item in row["reconstruction_records"]
            if Path(item["path"]).name == name
        )
        for name in ROUND28_RECONSTRUCTION_RECORD_ORDER
    }
    if (
        runner["rederivation"]["reproduction_record_sha256"]
        != record_hashes["legacy_comparator_reproduction.json"]
        or embedded["reproduction_reference"]["sha256"]
        != record_hashes["legacy_comparator_reproduction.json"]
        or embedded["late_observation_reference"]["sha256"]
        != record_hashes["gpu_process_observation_late.json"]
        or inventory.get("source_inventory_sha256")
        != ROUND28_SOURCE_INVENTORY_SHA256
        or inventory.get("source_file_count") != ROUND28_SOURCE_FILE_COUNT
        or inventory.get("source_total_bytes") != ROUND28_SOURCE_TOTAL_BYTES
        or _forbidden_matches(inventory.get("files") or [])
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: reconstructed seals drift",
        )
    if require_archive_log_event:
        _verify_reconstructed_session_log(row, records=records)


def validate_preparation_failure_entry(
    row: Mapping[str, Any], *, ledger_path: Path, line_number: int,
    verify_archive: bool = False,
) -> None:
    label = f"{ledger_path}:{line_number}"
    observed_fields = set(row)
    reconstructed = observed_fields == PREPARATION_FAILURE_RECONSTRUCTED_ENTRY_FIELDS
    contemporaneous = observed_fields == PREPARATION_FAILURE_ENTRY_FIELDS
    if not contemporaneous and not reconstructed:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: non-closed fields",
        )
    if (
        row.get("schema_version") != 1
        or row.get("entry_type") != "preparation_failure_archive"
        or not _is_uuid4(row.get("entry_id"))
        or row.get("campaign_round") != 28
        or row.get("reason_code") != PREPARATION_FAILURE_REASON
        or row.get("runner_refusal_code") != PREPARATION_FAILURE_RUNNER_REFUSAL
        or row.get("defect_summary") != DEFECT_SUMMARY
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}: constants/UUID invalid",
        )
    _parse_utc(row.get("created_at_utc"), label=f"{label}.created_at_utc")
    for field in (
        "original_matrix_file_sha256", "original_matrix_seal_sha256",
        "original_preflight_sha256", "original_admission_run_id",
        "inventory_sha256", "pinned_source_definition_sha256",
        "staged_definition_sha256", "staged_module_sha256",
    ):
        if not _is_sha(row.get(field)):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}.{field} is not sha256",
            )
    for field in ("original_head", "replacement_head"):
        if not _is_sha(row.get(field), 40):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}.{field} is not git SHA",
            )
    for field in (
        "rlcr_dir", "source_journal_path", "archive_path", "inventory_file",
        "adjudication_ref",
    ):
        value = row.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}.{field} not absolute",
            )
    for field in (
        "rlcr_session_id", "chunk_id", "cell_id", "original_session_id",
        "operator_flag",
    ):
        if not isinstance(row.get(field), str) or not row[field]:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}.{field} empty",
            )
    allowed_flags = {
        PREPARATION_FAILURE_OPERATOR_FLAG,
        f"{PREPARATION_FAILURE_OPERATOR_FLAG} {PREPARATION_FAILURE_OVERRIDE_FLAG}",
    }
    if row.get("operator_flag") not in allowed_flags:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{label}.operator_flag invalid",
        )
    if reconstructed:
        _validate_reconstructed_entry_extensions(row, label=label)
    else:
        _validate_no_gpu_attestation(row.get("no_gpu_attestation"), label=label)
        if row["no_gpu_attestation"]["problem_packager_execute"].get(
            "source_head"
        ) != row.get("original_head"):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: ProblemPackager source HEAD is not original_head",
            )
    _validate_json_schema(
        row, schema_path=PREPARATION_FAILURE_ENTRY_SCHEMA,
        code="PREPARATION_LEDGER_INCONSISTENT",
    )
    if verify_archive:
        archive = Path(row["archive_path"])
        inventory_path = Path(row["inventory_file"])
        if inventory_path != archive / PREPARATION_FAILURE_INVENTORY_NAME:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}: inventory path drift",
            )
        if not inventory_path.is_file() or sha256_file(inventory_path) != row[
            "inventory_sha256"
        ]:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}: inventory hash drift",
            )
        inventory = load_archive_inventory(inventory_path)
        verify_preparation_archive(archive, inventory)
        if reconstructed:
            _verify_reconstructed_archive_entry(
                row, archive=archive, inventory=inventory, label=label,
                require_archive_log_event=True,
            )
            return
        for name, evidence in row["no_gpu_attestation"].items():
            evidence_path = Path(evidence["evidence_path"])
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"{label}.{name} evidence is not a regular file",
                )
            try:
                evidence_path.resolve(strict=True).relative_to(archive.resolve())
            except (OSError, RuntimeError, ValueError) as exc:
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"{label}.{name} evidence escapes archive: {exc}",
                ) from exc
            if sha256_file(evidence_path) != evidence["sha256"]:
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"{label}.{name} evidence hash drift",
                )
        attestation = row["no_gpu_attestation"]
        journal_start_path = Path(attestation["journal_state"]["evidence_path"])
        journal_start = _json_object(
            journal_start_path, code="PREPARATION_LEDGER_INCONSISTENT",
        )
        try:
            _verify_start_seal(journal_start, path=journal_start_path)
        except PreparationArchiveRefusal as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", exc.detail,
            ) from exc
        if (
            journal_start.get("state") != "capture_started"
            or (journal_start_path.parent / "journal_complete.json").exists()
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: archived journal is not incomplete capture_started",
            )
        stderr_path = Path(attestation["prepare_stderr"]["evidence_path"])
        if stderr_path.read_bytes() != b"":
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{label}: stderr is nonempty",
            )
        stdout_path = Path(attestation["prepare_stdout"]["evidence_path"])
        prepare_manifest_path = (
            journal_start_path.parent / "cells" / str(row["cell_id"])
            / "sol_stage" / "capture_prepare_manifest.json"
        )
        try:
            stdout_value = json.loads(stdout_path.read_text(encoding="utf-8"))
            prepare_manifest = _json_object(
                prepare_manifest_path, code="PREPARATION_LEDGER_INCONSISTENT",
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: prepare stream/manifest unreadable: {exc}",
            ) from exc
        if stdout_value != prepare_manifest:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: prepare stdout is not exactly the sealed manifest",
            )
        refusal_path = Path(attestation["runner_refusal"]["evidence_path"])
        phase_path = Path(attestation["phase_log"]["evidence_path"])
        gpu_path = Path(attestation["gpu_process_observation"]["evidence_path"])
        try:
            refusal = _validate_refusal_record(refusal_path)
            phase = _validate_phase_log(phase_path)
            gpu_observation = _validate_gpu_observation(gpu_path)
            _validate_packager_execute(
                Path(attestation["problem_packager_execute"]["evidence_path"])
                .read_bytes()
            )
        except PreparationArchiveRefusal as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", exc.detail,
            ) from exc
        if refusal["refused_at_utc"] != phase.get("refused_at_utc"):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: refusal/phase timestamps differ",
            )
        if _parse_utc(
            gpu_observation["observed_at_utc"], label="observed_at_utc",
        ) < _parse_utc(refusal["refused_at_utc"], label="refused_at_utc"):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: GPU-process observation predates refusal",
            )
        if row.get("cell_id") == ROUND28_CELL_ID:
            historical_expected = {
                "original_head": ROUND28_ORIGINAL_HEAD,
                "original_matrix_file_sha256": ROUND28_MATRIX_FILE_SHA256,
                "original_matrix_seal_sha256": ROUND28_MATRIX_SEAL_SHA256,
                "original_preflight_sha256": ROUND28_PREFLIGHT_SHA256,
                "original_admission_run_id": ROUND28_ADMISSION_RUN_ID,
                "original_session_id": ROUND28_SESSION_ID,
                "rlcr_session_id": ROUND28_RLCR_SESSION_ID,
                "chunk_id": ROUND28_CHUNK_ID,
                "pinned_source_definition_sha256": (
                    ROUND28_PINNED_SOURCE_DEFINITION_SHA256
                ),
                "staged_definition_sha256": ROUND28_STAGED_DEFINITION_SHA256,
                "staged_module_sha256": ROUND28_STAGED_MODULE_SHA256,
            }
            drift = [
                field for field, expected in historical_expected.items()
                if row.get(field) != expected
            ]
            if (
                drift
                or inventory.get("source_inventory_sha256")
                != ROUND28_SOURCE_INVENTORY_SHA256
                or inventory.get("source_file_count") != ROUND28_SOURCE_FILE_COUNT
                or inventory.get("source_total_bytes")
                != ROUND28_SOURCE_TOTAL_BYTES
            ):
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"{label}: preserved Round-28 attempt pin drift: {drift}",
                )
        forbidden = _forbidden_matches(inventory["files"])
        if (
            forbidden
            or attestation["forbidden_artifact_scan"]["scanned_file_count"]
            != len(inventory["files"])
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{label}: forbidden scan no longer rederives",
            )


def load_preparation_failure_ledger(
    path: Path, *, verify_archives: bool = False,
) -> list[dict]:
    """Load and strictly validate an append-only preparation ledger."""
    if path.parent.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"ledger parent is a symlink: {path.parent}",
        )
    if not path.exists():
        if verify_archives and path.parent.exists():
            try:
                unexpected = sorted(
                    str(child) for child in path.parent.iterdir()
                    if child.name != path.name
                )
            except OSError as exc:
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT", f"{path.parent}: {exc}",
                ) from exc
            if unexpected:
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"archive evidence exists without ledger: {unexpected}",
                )
        return []
    if path.is_symlink() or not path.is_file():
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", str(path),
        )
    try:
        raw_bytes = path.read_bytes()
        if raw_bytes and not raw_bytes.endswith(b"\n"):
            line_number = raw_bytes.count(b"\n") + 1
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{path}:{line_number}: missing final LF",
            )
        text = raw_bytes.decode("utf-8")
        lines = text.split("\n")[:-1] if raw_bytes else []
    except PreparationArchiveRefusal:
        raise
    except (OSError, UnicodeError) as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"{path}: {exc}",
        ) from exc
    rows: list[dict] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        if not raw:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{path}:{line_number}: blank",
            )
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{path}:{line_number}: {exc}",
            ) from exc
        if not isinstance(row, dict):
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{path}:{line_number}: not object",
            )
        validate_preparation_failure_entry(
            row, ledger_path=path, line_number=line_number,
            verify_archive=verify_archives,
        )
        if row["entry_id"] in seen:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"{path}:{line_number}: duplicate entry_id",
            )
        seen.add(row["entry_id"])
        rows.append(row)
    if not rows:
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", f"ledger is empty: {path}",
        )
    if verify_archives:
        root = path.parent.resolve()
        expected: set[Path] = set()
        for line_number, row in enumerate(rows, start=1):
            archive = Path(row["archive_path"])
            try:
                resolved = archive.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"{path}:{line_number}: archive unavailable: {exc}",
                ) from exc
            if resolved.parent != root or resolved in expected:
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"{path}:{line_number}: archive path outside root/duplicate",
                )
            expected.add(resolved)
        actual: set[Path] = set()
        try:
            children = list(root.iterdir())
        except OSError as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT", f"{root}: {exc}",
            ) from exc
        for child in children:
            if child.name == path.name:
                if child.is_symlink() or not child.is_file():
                    raise PreparationArchiveRefusal(
                        "PREPARATION_LEDGER_INCONSISTENT",
                        f"ledger is not a regular file: {child}",
                    )
                continue
            if child.is_symlink() or not child.is_dir():
                raise PreparationArchiveRefusal(
                    "PREPARATION_LEDGER_INCONSISTENT",
                    f"unexpected preparation-failure root entry: {child}",
                )
            actual.add(child.resolve())
        if actual != expected:
            raise PreparationArchiveRefusal(
                "PREPARATION_LEDGER_INCONSISTENT",
                f"ledger/archive set mismatch: "
                f"unledgered={sorted(map(str, actual - expected))!r}; "
                f"missing={sorted(map(str, expected - actual))!r}",
            )
    return rows


def preparation_failure_backrefs(
    rows: Sequence[Mapping[str, Any]], *, cell_id: str, ledger_path: Path,
) -> list[dict]:
    """Map ledger ``entry_id`` values to the journal back-reference shape."""
    resolved_ledger = ledger_path.resolve()
    selected = [row for row in rows if row.get("cell_id") == cell_id]
    return [{
        "ledger_entry_id": row["entry_id"],
        "ledger_path": str(resolved_ledger),
        "archive_path": row["archive_path"],
    } for row in selected]


def _verify_start_seal(start: dict, *, path: Path) -> None:
    claimed = start.get("journal_start_seal_sha256")
    unsigned = {
        key: value for key, value in start.items()
        if key != "journal_start_seal_sha256"
    }
    if claimed != canonical_sha256(unsigned):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", f"journal seal mismatch: {path}",
        )


def _forbidden_matches(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    matches: list[dict] = []
    for row in rows:
        rel = str(row["relative_path"])
        lowered = rel.lower()
        for pattern in FORBIDDEN_ARTIFACT_PATTERNS:
            if fnmatch.fnmatchcase(lowered, pattern.lower()):
                matches.append({"relative_path": rel, "pattern": pattern})
    return matches


def _validate_refusal_record(path: Path) -> dict:
    value = _json_object(path, code="PREPARATION_ARCHIVE_NOT_PREGPU")
    if (
        value.get("schema_version") != 1
        or value.get("runner_refusal_code") != PREPARATION_FAILURE_RUNNER_REFUSAL
        or value.get("phase") != "prepare_validate"
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"refusal record: {path}",
        )
    try:
        _parse_utc(value.get("refused_at_utc"), label="refused_at_utc")
    except PreparationArchiveRefusal as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", exc.detail,
        ) from exc
    return value


def _validate_phase_log(path: Path) -> dict:
    value = _json_object(path, code="PREPARATION_ARCHIVE_NOT_PREGPU")
    phases = value.get("phases")
    if value.get("schema_version") != 1 or not isinstance(phases, list) or not phases:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"phase log malformed: {path}",
        )
    allowed = {"prepare", "prepare_validate"}
    names = []
    for row in phases:
        name = row.get("phase") if isinstance(row, dict) else None
        if name not in allowed:
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_NOT_PREGPU", f"post-prepare phase: {name!r}",
            )
        names.append(name)
    if (
        names[-1] != "prepare_validate"
        or not isinstance(phases[-1], dict)
        or phases[-1].get("status") != "refused"
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", "semantic refusal phase absent",
        )
    return value


def _validate_gpu_observation(path: Path) -> dict:
    value = _json_object(path, code="PREPARATION_ARCHIVE_NOT_PREGPU")
    if (
        value.get("schema_version") != 1
        or value.get("remaining_gpu_processes") != []
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"GPU observation: {path}",
        )
    try:
        _parse_utc(value.get("observed_at_utc"), label="observed_at_utc")
    except PreparationArchiveRefusal as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", exc.detail,
        ) from exc
    return value


def _validate_packager_execute(source: bytes) -> tuple[int, int]:
    try:
        tree = ast.parse(source.decode("utf-8"))
    except (UnicodeError, SyntaxError) as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"ProblemPackager source: {exc}",
        ) from exc
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "execute"
    ]
    if len(functions) != 1:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", "ProblemPackager.execute not unique",
        )
    function = functions[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    writes_driver = any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "write_text"
        for call in calls
    )
    executes = any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "subprocess"
        and call.func.attr in {
            "run", "Popen", "call", "check_call", "check_output",
        }
        for call in calls
    )
    returns_command = any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.List)
        for node in ast.walk(function)
    )
    if not writes_driver or executes or not returns_command:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU",
            "ProblemPackager.execute does not satisfy static pre-GPU contract",
        )
    return function.lineno, function.end_lineno or function.lineno


def _verify_packager_source_at_head(
    source_path: Path, *, original_head: str,
    require_repo_relative: bool = False,
) -> bytes:
    try:
        relative = source_path.resolve(strict=True).relative_to(REPO_ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        if require_repo_relative:
            raise PreparationArchiveRefusal(
                "SOL_PREP_RECON_ALLOWLIST_VIOLATION",
                f"ProblemPackager source is not repository-relative: {source_path}",
            ) from exc
        # Fixture callers may supply an isolated source file.  The production
        # contemporaneous path may supply an isolated source file.  Reconstructed
        # mode requires the exact in-repository source pinned by its allowlist.
        return source_path.read_bytes()
    observed = source_path.read_bytes()
    proc = subprocess.run(
        ["git", "show", f"{original_head}:{relative.as_posix()}"],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if proc.returncode != 0 or proc.stdout != observed:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU",
            "ProblemPackager source does not match the original HEAD",
        )
    return observed


@dataclass(frozen=True)
class ArchiveRequest:
    journals_root: Path
    preparation_failures_root: Path
    original_matrix_path: Path
    chunk_run_journal: Path
    runner_refusal_record: Path | None
    phase_log: Path | None
    gpu_process_observation: Path | None
    adjudication_ref: Path
    rlcr_dir: Path
    replacement_head: str
    problem_packager_source: Path
    expected_inventory_sha256: str = ROUND28_SOURCE_INVENTORY_SHA256
    expected_file_count: int = ROUND28_SOURCE_FILE_COUNT
    expected_total_bytes: int = ROUND28_SOURCE_TOTAL_BYTES
    original_head: str = ROUND28_ORIGINAL_HEAD
    original_matrix_file_sha256: str = ROUND28_MATRIX_FILE_SHA256
    original_matrix_seal_sha256: str = ROUND28_MATRIX_SEAL_SHA256
    original_preflight_sha256: str = ROUND28_PREFLIGHT_SHA256
    original_admission_run_id: str = ROUND28_ADMISSION_RUN_ID
    original_session_id: str = ROUND28_SESSION_ID
    rlcr_session_id: str = ROUND28_RLCR_SESSION_ID
    chunk_id: str = ROUND28_CHUNK_ID
    cell_id: str = ROUND28_CELL_ID
    pinned_source_definition_sha256: str = (
        ROUND28_PINNED_SOURCE_DEFINITION_SHA256
    )
    staged_definition_sha256: str = ROUND28_STAGED_DEFINITION_SHA256
    staged_module_sha256: str = ROUND28_STAGED_MODULE_SHA256
    override_preparation_attempt_limit: bool = False
    created_at_utc: datetime | None = None
    context_mode: str = CONTEXT_MODE_CONTEMPORANEOUS
    reconstruction_root: Path = ROUND28_RECONSTRUCTION_ROOT
    session_log_path: Path = ROUND28_SESSION_LOG
    corrected_adjudication_ref: Path = ROUND28_CORRECTED_ADJUDICATION
    superseded_adjudication_ref: Path = ROUND28_SUPERSEDED_ADJUDICATION
    correction_adjudication_ref: Path = ROUND28_CORRECTION_ADJUDICATION
    correction2_adjudication_ref: Path = ROUND28_CORRECTION2_ADJUDICATION
    stage_equivalence_adjudication_source: Path = (
        ROUND28_STAGE_EQUIVALENCE_SOURCE
    )


def _write_bytes_exclusive_fsync(
    path: Path, raw: bytes, *, code: str, mode: int = 0o600,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as exc:
        raise PreparationArchiveRefusal(code, f"exists: {path}") from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _single_journal(request: ArchiveRequest) -> Path:
    if request.journals_root.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            f"journals root is a symlink: {request.journals_root}",
        )
    journals = request.journals_root.resolve()
    if journals.name != "journals" or not journals.is_dir():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", str(journals),
        )
    children = sorted(journals.iterdir())
    if (
        len(children) != 1 or children[0].name != request.cell_id
        or children[0].is_symlink() or not children[0].is_dir()
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            f"journals must contain exactly {request.cell_id}",
        )
    return children[0]


def _refuse_reconstruction_contamination(*roots: Path) -> None:
    retired = {
        "runner_refusal_record.json", "phase_log.json",
        "gpu_process_observation.json",
    }
    matches: list[str] = []
    for root in roots:
        if root.exists() and root.is_dir():
            matches.extend(
                str(path) for path in root.rglob("*")
                if path.name in retired
            )
    if matches:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_JOURNAL_CONTAMINATION",
            f"retired contemporaneous records: {sorted(matches)}",
        )


def _verify_reconstructed_allowlist(request: ArchiveRequest) -> None:
    expected = {
        "context_mode": CONTEXT_MODE_RECONSTRUCTED,
        "expected_inventory_sha256": ROUND28_SOURCE_INVENTORY_SHA256,
        "expected_file_count": ROUND28_SOURCE_FILE_COUNT,
        "expected_total_bytes": ROUND28_SOURCE_TOTAL_BYTES,
        "original_head": ROUND28_ORIGINAL_HEAD,
        "original_matrix_file_sha256": ROUND28_MATRIX_FILE_SHA256,
        "original_matrix_seal_sha256": ROUND28_MATRIX_SEAL_SHA256,
        "original_preflight_sha256": ROUND28_PREFLIGHT_SHA256,
        "original_admission_run_id": ROUND28_ADMISSION_RUN_ID,
        "original_session_id": ROUND28_SESSION_ID,
        "rlcr_session_id": ROUND28_RLCR_SESSION_ID,
        "chunk_id": ROUND28_CHUNK_ID,
        "cell_id": ROUND28_CELL_ID,
        "pinned_source_definition_sha256": (
            ROUND28_PINNED_SOURCE_DEFINITION_SHA256
        ),
        "staged_definition_sha256": ROUND28_STAGED_DEFINITION_SHA256,
        "staged_module_sha256": ROUND28_STAGED_MODULE_SHA256,
        "override_preparation_attempt_limit": False,
    }
    drift = [
        field for field, expected_value in expected.items()
        if getattr(request, field) != expected_value
    ]
    path_expected = {
        "journals_root": ROUND28_JOURNALS_ROOT,
        "preparation_failures_root": ROUND28_PREPARATION_FAILURES_ROOT,
        "rlcr_dir": ROUND28_RLCR_DIR,
        "adjudication_ref": ROUND28_STAGE_EQUIVALENCE_ADJUDICATION,
        "problem_packager_source": ROUND28_PROBLEM_PACKAGER_SOURCE,
        "reconstruction_root": ROUND28_RECONSTRUCTION_ROOT,
        "session_log_path": ROUND28_SESSION_LOG,
        "corrected_adjudication_ref": ROUND28_CORRECTED_ADJUDICATION,
        "superseded_adjudication_ref": ROUND28_SUPERSEDED_ADJUDICATION,
        "correction_adjudication_ref": ROUND28_CORRECTION_ADJUDICATION,
        "correction2_adjudication_ref": ROUND28_CORRECTION2_ADJUDICATION,
        "stage_equivalence_adjudication_source": (
            ROUND28_STAGE_EQUIVALENCE_SOURCE
        ),
    }
    for field, expected_path in path_expected.items():
        try:
            if getattr(request, field).resolve() != expected_path.resolve():
                drift.append(field)
        except (OSError, RuntimeError):
            drift.append(field)
    if drift:
        raise PreparationArchiveRefusal(
            "SOL_PREP_RECON_ALLOWLIST_VIOLATION",
            f"reconstructed incident tuple drift: {sorted(set(drift))}",
        )


def _verify_reconstruction_journal(
    request: ArchiveRequest,
) -> tuple[Path, list[dict], Path, Path, Path]:
    journal = _single_journal(request)
    _refuse_reconstruction_contamination(journal)
    start_path = journal / "journal_start.json"
    complete_path = journal / "journal_complete.json"
    if complete_path.exists() or complete_path.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"completion exists: {complete_path}",
        )
    if sha256_file(start_path) != ROUND28_JOURNAL_START_SHA256:
        raise PreparationArchiveRefusal(
            "SOL_PREP_RECON_ALLOWLIST_VIOLATION", "journal-start SHA drift",
        )
    start = _json_object(start_path, code="PREPARATION_ARCHIVE_SOURCE_MISSING")
    _verify_start_seal(start, path=start_path)
    expected_start = {
        "state": "capture_started",
        "source_commit": ROUND28_ORIGINAL_HEAD,
        "capture_matrix_sha256": ROUND28_MATRIX_FILE_SHA256,
        "matrix_seal_sha256": ROUND28_MATRIX_SEAL_SHA256,
        "capture_preflight_sha256": ROUND28_PREFLIGHT_SHA256,
        "admission_run_id": ROUND28_ADMISSION_RUN_ID,
        "supersedes": None,
    }
    if any(start.get(key) != value for key, value in expected_start.items()):
        raise PreparationArchiveRefusal(
            "SOL_PREP_RECON_ALLOWLIST_VIOLATION", "journal-start tuple drift",
        )
    manifest = journal / ROUND28_MANIFEST_JOURNAL_RELATIVE_PATH
    staged = journal / ROUND28_STAGED_DEFINITION_JOURNAL_RELATIVE_PATH
    stdout = journal / "cells" / ROUND28_CELL_ID / "logs/sol_prepare.stdout.log"
    if sha256_file(manifest) != ROUND28_PREPARE_MANIFEST_SHA256:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", "prepare-manifest-hash mismatch",
        )
    manifest_value = _json_object(manifest, code="SOL_LEGACY_REPRO_INPUT_DRIFT")
    source_inputs = manifest_value.get("source_inputs")
    definition = (
        source_inputs.get("definition") if isinstance(source_inputs, dict) else None
    )
    if (
        not isinstance(definition, dict)
        or definition.get("path") != str(ROUND28_SOURCE_DEFINITION_PATH)
        or definition.get("sha256") != ROUND28_PINNED_SOURCE_DEFINITION_SHA256
    ):
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", "prepare-manifest pin mismatch",
        )
    try:
        if (
            ROUND28_SOURCE_DEFINITION_PATH.is_symlink()
            or not ROUND28_SOURCE_DEFINITION_PATH.is_file()
            or sha256_file(ROUND28_SOURCE_DEFINITION_PATH)
            != ROUND28_PINNED_SOURCE_DEFINITION_SHA256
            or sha256_file(staged) != ROUND28_STAGED_DEFINITION_SHA256
        ):
            raise PreparationArchiveRefusal(
                "SOL_LEGACY_REPRO_INPUT_DRIFT",
                "external-source/staged-definition hash drift",
            )
    except OSError as exc:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", f"missing-file: {exc}",
        ) from exc
    if sha256_file(stdout) != ROUND28_PREPARE_STDOUT_SHA256:
        raise PreparationArchiveRefusal(
            "SOL_PREP_RECON_ALLOWLIST_VIOLATION", "prepare-stdout SHA drift",
        )
    stderr = stdout.with_name("sol_prepare.stderr.log")
    if sha256_file(stderr) != ROUND28_PREPARE_STDERR_SHA256:
        raise PreparationArchiveRefusal(
            "SOL_PREP_RECON_ALLOWLIST_VIOLATION", "prepare-stderr SHA drift",
        )
    rows = content_inventory(journal)
    observed_pin = source_inventory_sha256(rows)
    if (
        observed_pin != ROUND28_SOURCE_INVENTORY_SHA256
        or len(rows) != ROUND28_SOURCE_FILE_COUNT
        or sum(row["size_bytes"] for row in rows) != ROUND28_SOURCE_TOTAL_BYTES
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
            f"pin={observed_pin}; files={len(rows)}; "
            f"bytes={sum(row['size_bytes'] for row in rows)}",
        )
    forbidden = _forbidden_matches(rows)
    if forbidden:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", json.dumps(forbidden, sort_keys=True),
        )
    return journal, rows, manifest, staged, stdout


def _verify_h3_identity(request: ArchiveRequest) -> str:
    code = "SOL_LEGACY_REPRO_HEAD_MISMATCH"
    status = _git_text(REPO_ROOT, "status", "--porcelain", "-uall", code=code)
    head = _git_text(REPO_ROOT, "rev-parse", "HEAD", code=code)
    parent = _git_text(REPO_ROOT, "rev-parse", "HEAD^", code=code)
    subject = _git_text(REPO_ROOT, "show", "-s", "--format=%s", "HEAD", code=code)
    committed_at = _git_text(
        REPO_ROOT, "show", "-s", "--format=%cI", "HEAD", code=code,
    )
    try:
        committed = datetime.fromisoformat(committed_at).astimezone(timezone.utc)
        cutoff = _parse_utc(
            ROUND28_H2_COMMITTED_AT_UTC, label="ROUND28_H2_COMMITTED_AT_UTC",
        )
    except (ValueError, PreparationArchiveRefusal) as exc:
        raise PreparationArchiveRefusal(code, f"commit time: {exc}") from exc
    if (
        status or head != request.replacement_head or parent != ROUND28_H2_HEAD
        or subject != ROUND28_H3_SUBJECT or committed <= cutoff
    ):
        raise PreparationArchiveRefusal(
            code,
            f"status={status!r}; head={head}; requested={request.replacement_head}; "
            f"parent={parent}; subject={subject!r}; committed_at={committed_at}",
        )
    return head


def _load_session_log(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise PreparationArchiveRefusal(
            "RLCR_SESSION_LOG_INCONSISTENT", f"missing/invalid: {path}",
        )
    try:
        raw_bytes = path.read_bytes()
        if raw_bytes and not raw_bytes.endswith(b"\n"):
            line_number = raw_bytes.count(b"\n") + 1
            raise PreparationArchiveRefusal(
                "RLCR_SESSION_LOG_INCONSISTENT",
                f"{path}:{line_number}: missing final LF",
            )
        text = raw_bytes.decode("utf-8")
        lines = text.split("\n")[:-1] if raw_bytes else []
        rows = [json.loads(line) for line in lines]
    except PreparationArchiveRefusal:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationArchiveRefusal(
            "RLCR_SESSION_LOG_INCONSISTENT", f"{path}: {exc}",
        ) from exc
    if not rows or len(rows) != len(lines) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise PreparationArchiveRefusal(
            "RLCR_SESSION_LOG_INCONSISTENT", f"empty/non-object row: {path}",
        )
    return rows


def _evidence_manifest_hashes(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_INVALID",
            f"missing/non-regular evidence manifest: {path}",
        )
    try:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_INVALID",
            f"manifest is outside the repository: {path}",
        ) from exc
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if tracked.returncode != 0:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_UNTRACKED", str(path),
        )
    dirty = subprocess.run(
        [
            "git", "status", "--porcelain=v1", "--untracked-files=all",
            "--", relative.as_posix(),
        ],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if dirty.returncode != 0:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_INVALID",
            dirty.stderr.decode("utf-8", errors="replace").strip(),
        )
    if dirty.stdout:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_DIRTY", str(path),
        )
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_INVALID", f"{path}: {exc}",
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        digest, separator, name = line.partition("  ")
        if (
            not separator or not _is_sha(digest) or not name
            or name in entries
        ):
            raise PreparationArchiveRefusal(
                "PREP_ARCHIVE_EVIDENCE_MANIFEST_INVALID",
                f"{path}:{line_number}: malformed/duplicate entry",
            )
        entries[name] = digest
    return entries


def _repository_evidence_hash(path: Path, manifest: Mapping[str, str]) -> str:
    if path.is_symlink() or not path.is_file():
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_PATH_INVALID",
            f"missing/non-regular evidence fixture: {path}",
        )
    try:
        relative = path.resolve(strict=True).relative_to(REPO_ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_PATH_INVALID",
            f"fixture is outside the repository: {path}",
        ) from exc
    relative_text = relative.as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_text],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if tracked.returncode != 0:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_UNTRACKED", relative_text,
        )
    dirty = subprocess.run(
        [
            "git", "status", "--porcelain=v1", "--untracked-files=all",
            "--", relative_text,
        ],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if dirty.returncode != 0:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_PATH_INVALID",
            dirty.stderr.decode("utf-8", errors="replace").strip(),
        )
    if dirty.stdout:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_DIRTY", relative_text,
        )
    expected = manifest.get(relative_text)
    if expected is None:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_MANIFEST_MISSING", relative_text,
        )
    observed = sha256_file(path)
    if observed != expected:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_HASH_MISMATCH",
            f"{relative_text}: expected={expected}; observed={observed}",
        )
    return observed


def _verify_stage_adjudication_alias(request: ArchiveRequest) -> None:
    aliases_path = ROUND28_PATH_ALIASES
    if aliases_path.is_symlink() or not aliases_path.is_file():
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_ALIAS_INVALID", str(aliases_path),
        )
    try:
        relative = aliases_path.resolve(strict=True).relative_to(REPO_ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_ALIAS_INVALID", str(aliases_path),
        ) from exc
    for command, code in (
        (["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
         "PREP_ARCHIVE_EVIDENCE_ALIAS_UNTRACKED"),
        (["git", "status", "--porcelain=v1", "--untracked-files=all", "--",
          relative.as_posix()], "PREP_ARCHIVE_EVIDENCE_ALIAS_DIRTY"),
    ):
        proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, check=False)
        if proc.returncode != 0 or (command[1] == "status" and proc.stdout):
            raise PreparationArchiveRefusal(code, str(aliases_path))
    value = _json_object(
        aliases_path, code="PREP_ARCHIVE_EVIDENCE_ALIAS_INVALID",
    )
    historical = str(request.adjudication_ref.absolute())
    expected = request.stage_equivalence_adjudication_source.resolve()
    matches: list[Path] = []
    for row in value.get("aliases", []):
        if not isinstance(row, dict) or row.get("requires_sha256_match") is not True:
            continue
        source = row.get("historical")
        target = row.get("canonical")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        suffix: str | None = None
        if row.get("match") == "exact" and historical == source:
            suffix = ""
        elif row.get("match") == "prefix" and historical.startswith(source):
            suffix = historical[len(source):]
        if suffix is not None:
            matches.append((REPO_ROOT / target / suffix).resolve())
    if matches != [expected]:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_ALIAS_MISMATCH",
            f"historical={historical}; resolved={list(map(str, matches))}; "
            f"expected={expected}",
        )
    historical_path = Path(historical)
    if historical_path.is_symlink() or not historical_path.is_file():
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_ALIAS_SOURCE_INVALID",
            f"missing/non-regular historical adjudication: {historical_path}",
        )
    try:
        historical_sha256 = sha256_file(historical_path)
        canonical_sha256 = sha256_file(expected)
    except OSError as exc:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_ALIAS_SOURCE_INVALID",
            f"unreadable alias endpoint: {exc}",
        ) from exc
    if historical_sha256 != canonical_sha256:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_ALIAS_HASH_MISMATCH",
            f"historical={historical_sha256}; canonical={canonical_sha256}",
        )


def _adjudication_hashes(request: ArchiveRequest) -> dict[str, str]:
    _verify_stage_adjudication_alias(request)
    refs = {
        "corrected": request.corrected_adjudication_ref,
        "superseded": request.superseded_adjudication_ref,
        "correction": request.correction_adjudication_ref,
        "correction2": request.correction2_adjudication_ref,
        "stage_equivalence": request.stage_equivalence_adjudication_source,
    }
    manifest = _evidence_manifest_hashes(ROUND28_EVIDENCE_MANIFEST)
    observed = {
        name: _repository_evidence_hash(path, manifest)
        for name, path in refs.items()
    }
    if observed["stage_equivalence"] != ROUND28_ANNEX_C_SHA256:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_EVIDENCE_HASH_MISMATCH",
            "stage-equivalence adjudication does not match Annex C authority",
        )
    if (
        observed["correction"] != ROUND28_CORRECTION_SHA256
        or observed["correction2"] != ROUND28_CORRECTION2_SHA256
        or len(set(observed.values())) != len(observed)
    ):
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID",
            f"adjudication SHA drift/equality: {observed}",
        )
    return observed


def _required_session_prefix(
    request: ArchiveRequest, *, h3: str, adjudications: Mapping[str, str],
) -> list[dict]:
    return [
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_prep070_reproduction_contract_correction.md",
            "sha256": ROUND28_CORRECTION_SHA256,
            "status": "binding_supplement",
            "supplements_amendment_id": ROUND28_AMENDMENT_ID,
            "correction_id": ROUND28_CORRECTION_ID,
        },
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_post_move_failure_state_correction.md",
            "sha256": ROUND28_CORRECTION2_SHA256,
            "status": "binding_supplement",
            "supplements_amendment_id": ROUND28_AMENDMENT_ID,
            "correction_id": ROUND28_CORRECTION2_ID,
        },
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_prep070_reconstruction_superseded_v1.md",
            "sha256": adjudications["superseded"],
            "status": "superseded_non_authoritative",
            "superseded_by_sha256": adjudications["corrected"],
            "defect": (
                "false_certifying_premise:"
                "deterministic_v1_comparator_over_sealed_bytes"
            ),
        },
        {
            "event": "adjudication_saved",
            "artifact": "adjudication_r28_prep070_reconstruction_v2.md",
            "sha256": adjudications["corrected"],
            "status": "binding",
            "supersedes_sha256": adjudications["superseded"],
            "amendment_id": ROUND28_AMENDMENT_ID,
            "h3": h3,
        },
    ]


def _record_event(relative_path: str, sha256: str) -> dict:
    return {
        "event": "reconstruction_record_created",
        "artifact": relative_path,
        "sha256": sha256,
    }


def _verify_session_log_prefix(
    request: ArchiveRequest, *, h3: str, adjudications: Mapping[str, str],
    required_record_count: int,
) -> list[dict]:
    rows = _load_session_log(request.session_log_path)
    expected = _required_session_prefix(
        request, h3=h3, adjudications=adjudications,
    )
    record_names = (
        "legacy_comparator_source_excerpt.py.txt",
        "legacy_comparator_reproduction.json",
        "runner_refusal_reconstruction.json",
        "phase_reconstruction.json",
        "gpu_process_observation_late.raw.txt",
        "gpu_process_observation_late.json",
        "preparation_failure_no_gpu_attestation_v2.json",
    )
    for name in record_names[:required_record_count]:
        path = request.reconstruction_root / name
        if path.is_symlink() or not path.is_file():
            raise PreparationArchiveRefusal(
                "RLCR_SESSION_LOG_INCONSISTENT", f"logged artifact missing: {path}",
            )
        expected.append(_record_event(name, sha256_file(path)))
    if rows != expected:
        raise PreparationArchiveRefusal(
            "RLCR_SESSION_LOG_INCONSISTENT",
            f"required prefix mismatch: rows={len(rows)} expected={len(expected)}",
        )
    return rows


def _append_reconstruction_record_event(
    request: ArchiveRequest, *, name: str, h3: str,
    adjudications: Mapping[str, str], prior_count: int,
) -> None:
    _verify_session_log_prefix(
        request, h3=h3, adjudications=adjudications,
        required_record_count=prior_count,
    )
    path = request.reconstruction_root / name
    _append_jsonl_fsync(
        request.session_log_path, _record_event(name, sha256_file(path)),
    )


def _copy_readonly_verified(source: Path, destination: Path, expected: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", f"missing-file: {source}",
        )
    before = sha256_file(source)
    if before != expected:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", f"hash-drift: {source}",
        )
    _write_bytes_exclusive_fsync(
        destination, source.read_bytes(), code="SOL_LEGACY_REPRO_INPUT_DRIFT",
        mode=0o400,
    )
    os.chmod(destination, 0o444)
    if sha256_file(source) != expected or sha256_file(destination) != expected:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", f"copy-hash-drift: {source}",
        )


def _validate_reconstruction_record(name: str, value: Mapping[str, Any]) -> None:
    if _contains_placeholder(value):
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID", f"placeholder in {name}",
        )
    schema = RECONSTRUCTION_RECORD_SCHEMAS[name]
    _validate_json_schema(
        value, schema_path=schema, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )


def _legacy_source_identity() -> tuple[bytes, str, str]:
    code = "SOL_LEGACY_REPRO_SOURCE_DRIFT"
    blob = _git_bytes(
        REPO_ROOT, "show",
        f"{ROUND28_ORIGINAL_HEAD}:{ROUND28_LEGACY_COMPARATOR_PATH}",
        code=code,
    )
    blob_sha = sha256_bytes(blob)
    start, end = ROUND28_LEGACY_COMPARATOR_LINE_SPAN
    excerpt = b"".join(blob.splitlines(keepends=True)[start - 1:end])
    excerpt_sha = sha256_bytes(excerpt)
    if (
        blob_sha != ROUND28_LEGACY_COMPARATOR_FILE_SHA256
        or excerpt_sha != ROUND28_LEGACY_COMPARATOR_EXCERPT_SHA256
        or b"def normalized_definition" not in excerpt
        or b"sol_stage_source_mismatch" not in excerpt
    ):
        raise PreparationArchiveRefusal(
            code,
            f"blob={blob_sha}; excerpt={excerpt_sha}; "
            f"lines={start}-{end}",
        )
    return excerpt, blob_sha, excerpt_sha


def _verify_reproduction_record(
    record: Mapping[str, Any], *, h3: str, expected_argv_tail: list[str],
    t_pre: datetime, t_post: datetime,
) -> None:
    _validate_reconstruction_record("legacy_comparator_reproduction.json", record)
    tool = record.get("tool")
    argv = record.get("command_argv")
    if not isinstance(tool, dict) or tool.get("head") != h3:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_HEAD_MISMATCH", "record tool.head mismatch",
        )
    current = _git_text(
        REPO_ROOT, "rev-parse", "HEAD", code="SOL_LEGACY_REPRO_HEAD_MISMATCH",
    )
    if current != h3:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_HEAD_MISMATCH",
            f"record={h3}; current={current}",
        )
    tool_path = REPO_ROOT / "tools/local_labels/legacy_compare_9b8e0e40.py"
    disk_sha = sha256_file(tool_path)
    blob_sha = sha256_bytes(_git_bytes(
        REPO_ROOT, "show",
        f"{h3}:tools/local_labels/legacy_compare_9b8e0e40.py",
        code="SOL_LEGACY_REPRO_TOOL_DRIFT",
    ))
    if tool.get("sha256") != disk_sha or disk_sha != blob_sha:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_TOOL_DRIFT",
            f"record={tool.get('sha256')}; disk={disk_sha}; blob={blob_sha}",
        )
    try:
        executed = _parse_utc(record.get("executed_at"), label="executed_at")
        cutoff = _parse_utc(
            ROUND28_H2_COMMITTED_AT_UTC, label="ROUND28_H2_COMMITTED_AT_UTC",
        )
    except PreparationArchiveRefusal as exc:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_TIME_INVALID", exc.detail,
        ) from exc
    lower = t_pre.astimezone(timezone.utc).timestamp() - 2.0
    upper = t_post.astimezone(timezone.utc).timestamp() + 2.0
    if not lower <= executed.timestamp() <= upper or executed <= cutoff:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_TIME_INVALID",
            f"executed={record.get('executed_at')}; pre={_utc_text(t_pre)}; "
            f"post={_utc_text(t_post)}",
        )
    if (
        not isinstance(argv, list) or len(argv) != 13
        or not isinstance(argv[0], str)
        or not argv[0].endswith("tools/local_labels/legacy_compare_9b8e0e40.py")
        or argv[1:] != expected_argv_tail
        or argv[10] != h3
    ):
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_ARGV_MISMATCH", repr(argv),
        )


def _load_reconstructed_records(
    request: ArchiveRequest,
) -> dict[str, dict]:
    root = request.reconstruction_root
    _refuse_reconstruction_contamination(root)
    expected_files = {
        *ROUND28_RECONSTRUCTION_CONTEXT_FILES,
    }
    expected_scratch = {"source_definition.json", "staged_definition.json"}
    if root.is_symlink() or not root.is_dir():
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID", f"construction root: {root}",
        )
    actual_top = {path.name for path in root.iterdir()}
    if actual_top != expected_files | {"scratch"}:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID",
            f"construction site fields drift: {sorted(actual_top)}",
        )
    scratch = root / "scratch"
    if scratch.is_symlink() or not scratch.is_dir() or {
        path.name for path in scratch.iterdir()
    } != expected_scratch:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID", "scratch file set drift",
        )
    for name, expected_sha in (
        ("source_definition.json", ROUND28_PINNED_SOURCE_DEFINITION_SHA256),
        ("staged_definition.json", ROUND28_STAGED_DEFINITION_SHA256),
    ):
        path = scratch / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise PreparationArchiveRefusal(
                "SOL_LEGACY_REPRO_INPUT_DRIFT", f"scratch drift: {path}",
            )
    records: dict[str, dict] = {}
    for name in ROUND28_RECONSTRUCTION_RECORD_ORDER:
        path = root / name
        value = _json_object(path, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID")
        _validate_reconstruction_record(name, value)
        records[name] = value
    excerpt = root / "legacy_comparator_source_excerpt.py.txt"
    raw = root / "gpu_process_observation_late.raw.txt"
    if (
        sha256_file(excerpt) != ROUND28_LEGACY_COMPARATOR_EXCERPT_SHA256
        or records["runner_refusal_reconstruction.json"]
        ["legacy_comparator_identity"]["line_span_sha256"]
        != sha256_file(excerpt)
        or records["gpu_process_observation_late.json"]["raw_output_sha256"]
        != sha256_file(raw)
    ):
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID", "cited context hash drift",
        )
    reproduction_sha = sha256_file(root / "legacy_comparator_reproduction.json")
    late_sha = sha256_file(root / "gpu_process_observation_late.json")
    runner = records["runner_refusal_reconstruction.json"]
    attestation = records["preparation_failure_no_gpu_attestation_v2.json"]
    if (
        runner["rederivation"]["reproduction_record_sha256"] != reproduction_sha
        or attestation["reproduction_reference"]["sha256"] != reproduction_sha
        or attestation["late_observation_reference"]["sha256"] != late_sha
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "record cross-reference drift",
        )
    return records


def build_reconstructed_context(request: ArchiveRequest) -> dict[str, dict]:
    """Construct the closed problem-070 records without moving the journal."""
    _verify_reconstructed_allowlist(request)
    h3 = _verify_h3_identity(request)
    adjudications = _adjudication_hashes(request)
    _verify_session_log_prefix(
        request, h3=h3, adjudications=adjudications, required_record_count=0,
    )
    journal, original_rows, manifest_path, staged_path, stdout_path = (
        _verify_reconstruction_journal(request)
    )
    root = request.reconstruction_root
    if root.exists() or root.is_symlink():
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID", f"construction site exists: {root}",
        )
    root.mkdir(parents=True, mode=0o700)
    scratch = root / "scratch"
    scratch.mkdir(mode=0o700)

    manifest = _json_object(manifest_path, code="SOL_LEGACY_REPRO_INPUT_DRIFT")
    source_inputs = manifest.get("source_inputs")
    definition = (
        source_inputs.get("definition") if isinstance(source_inputs, dict) else None
    )
    expected_source_text = str(ROUND28_SOURCE_DEFINITION_PATH)
    if (
        not isinstance(definition, dict)
        or definition.get("sha256") != ROUND28_PINNED_SOURCE_DEFINITION_SHA256
        or definition.get("path") != expected_source_text
    ):
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_INPUT_DRIFT", "prepare-manifest definition pin mismatch",
        )
    _copy_readonly_verified(
        ROUND28_SOURCE_DEFINITION_PATH, scratch / "source_definition.json",
        ROUND28_PINNED_SOURCE_DEFINITION_SHA256,
    )
    _copy_readonly_verified(
        staged_path, scratch / "staged_definition.json",
        ROUND28_STAGED_DEFINITION_SHA256,
    )
    if (
        content_inventory(journal) != original_rows
        or source_inventory_sha256(content_inventory(journal))
        != ROUND28_SOURCE_INVENTORY_SHA256
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
            "journal inventory changed during scratch construction",
        )

    excerpt, blob_sha, excerpt_sha = _legacy_source_identity()
    excerpt_path = root / "legacy_comparator_source_excerpt.py.txt"
    _write_bytes_exclusive_fsync(
        excerpt_path, excerpt, code="SOL_LEGACY_REPRO_SOURCE_DRIFT",
    )
    _append_reconstruction_record_event(
        request, name=excerpt_path.name, h3=h3, adjudications=adjudications,
        prior_count=0,
    )

    root_rel = root.relative_to(REPO_ROOT).as_posix()
    source_rel = f"{root_rel}/scratch/source_definition.json"
    staged_rel = f"{root_rel}/scratch/staged_definition.json"
    emit_rel = f"{root_rel}/legacy_comparator_reproduction.json"
    argv_tail = [
        "--source", source_rel,
        "--staged", staged_rel,
        "--expect-source-sha", ROUND28_PINNED_SOURCE_DEFINITION_SHA256,
        "--expect-staged-sha", ROUND28_STAGED_DEFINITION_SHA256,
        "--tool-head", h3,
        "--emit", emit_rel,
    ]
    t_pre = datetime.now(timezone.utc)
    proc = subprocess.run(
        [sys.executable, "-m", "tools.local_labels.legacy_compare_9b8e0e40",
         *argv_tail],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    t_post = datetime.now(timezone.utc)
    if proc.returncode != 0:
        known = (
            "SOL_LEGACY_REPRO_INPUT_DRIFT", "SOL_LEGACY_REPRO_PARSE_ERROR",
            "SOL_LEGACY_REPRO_HEAD_INVALID", "SOL_LEGACY_REPRO_EMIT_EXISTS",
        )
        code = next((item for item in known if item in proc.stderr), None)
        raise PreparationArchiveRefusal(
            code or "PREP_ARCHIVE_RECON_SCHEMA_INVALID", proc.stderr.strip(),
        )
    expected_stdout = f"legacy_compare_9b8e0e40: EMITTED {emit_rel}\n"
    if proc.stdout != expected_stdout or proc.stderr:
        raise PreparationArchiveRefusal(
            "SOL_LEGACY_REPRO_ARGV_MISMATCH",
            f"stdout={proc.stdout!r}; stderr={proc.stderr!r}",
        )
    reproduction_path = root / "legacy_comparator_reproduction.json"
    reproduction = _json_object(
        reproduction_path, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )
    _verify_reproduction_record(
        reproduction, h3=h3, expected_argv_tail=argv_tail,
        t_pre=t_pre, t_post=t_post,
    )
    _append_reconstruction_record_event(
        request, name=reproduction_path.name, h3=h3,
        adjudications=adjudications, prior_count=1,
    )

    reproduction_sha = sha256_file(reproduction_path)
    runner = {
        "record_type": "runner_refusal_reconstruction",
        "amendment_id": ROUND28_AMENDMENT_ID,
        "reconstruction_late": True,
        "contemporaneous": False,
        "reconstructed_at": _record_time(),
        "cell": ROUND28_CELL_ID,
        "chunk": ROUND28_CHUNK_ID,
        "original_head": ROUND28_ORIGINAL_HEAD,
        "session": ROUND28_SESSION_ID,
        "matrix_file_sha256": ROUND28_MATRIX_FILE_SHA256,
        "preflight_sha256": ROUND28_PREFLIGHT_SHA256,
        "historical_refusal": {
            "code": PREPARATION_FAILURE_RUNNER_REFUSAL,
            "message": "staged definition differs from pinned source",
            "reason": PREPARATION_FAILURE_REASON,
        },
        "sealed_inputs": {
            "source_definition_sha256": ROUND28_PINNED_SOURCE_DEFINITION_SHA256,
            "staged_definition_sha256": ROUND28_STAGED_DEFINITION_SHA256,
            "staged_module_sha256": ROUND28_STAGED_MODULE_SHA256,
        },
        "legacy_comparator_identity": {
            "rule_id": ROUND28_LEGACY_RULE_ID,
            "head": ROUND28_ORIGINAL_HEAD,
            "path": ROUND28_LEGACY_COMPARATOR_PATH,
            "file_sha256": blob_sha,
            "line_span": list(ROUND28_LEGACY_COMPARATOR_LINE_SPAN),
            "line_span_sha256": excerpt_sha,
            "function": "normalized_definition",
            "source_excerpt_file": excerpt_path.name,
            "semantics": (
                "For top-level fields hf_id, op_type, custom_inputs_entrypoint: "
                "a PRESENT empty-string value is rewritten to null; missing keys "
                "are not inserted and are never equated with null or empty string; "
                "all other content is compared by strict dict equality; inequality "
                "raises ChunkRefusal sol_stage_source_mismatch."
            ),
        },
        "rederivation": {
            "equal": False,
            "differing_top_level_keys": ["op_type"],
            "difference_pointers": ["/op_type"],
            "source_op_type_state": "missing",
            "staged_op_type_state": "null",
            "maps_to_refusal_code": PREPARATION_FAILURE_RUNNER_REFUSAL,
            "reproduction_record": reproduction_path.name,
            "reproduction_record_sha256": reproduction_sha,
        },
        "authority_basis": list(ROUND28_AUTHORITY_BASIS),
        "authority_citations": {
            "annex_c_adjudication_sha256": ROUND28_ANNEX_C_SHA256,
            "corrected_adjudication_sha256": adjudications["corrected"],
            "superseded_adjudication_sha256": adjudications["superseded"],
            "supersession_status": "superseded_by_v2",
        },
        "non_certifying_context": {
            "sol_def_equiv_v1": {
                "verdict": "equal", "difference_pointers": [],
                "artifact_sha256": ROUND28_V1_AUDIT_SHA256,
            },
            "sol_def_equiv_v2": {
                "verdict": "equal", "audit_sha256": ROUND28_V2_H1_AUDIT_SHA256,
            },
            "note": (
                "V1/V2 equality is the intended repair semantics and is never "
                "a basis for the historical refusal."
            ),
        },
    }
    runner_path = root / "runner_refusal_reconstruction.json"
    _validate_reconstruction_record(runner_path.name, runner)
    _write_compact_json_exclusive_fsync(
        runner_path, runner, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )
    _append_reconstruction_record_event(
        request, name=runner_path.name, h3=h3, adjudications=adjudications,
        prior_count=2,
    )

    stdout_mtime = _stat_mtime_utc(stdout_path)
    if stdout_mtime != ROUND28_PREPARE_STDOUT_MTIME_UTC:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_INTERVAL_INVALID",
            f"stdout mtime={stdout_mtime}",
        )
    phase = {
        "record_type": "phase_reconstruction",
        "amendment_id": ROUND28_AMENDMENT_ID,
        "interval_type": "non_certifying_reconstruction",
        "reconstruction_late": True,
        "contemporaneous": False,
        "reconstructed_at": _record_time(),
        "cell": ROUND28_CELL_ID,
        "chunk": ROUND28_CHUNK_ID,
        "anchors": [
            {
                "name": "prepare_stdout_mtime", "value": stdout_mtime,
                "basis": "filesystem_stat_mtime",
                "evidence_sha256": ROUND28_PREPARE_STDOUT_SHA256,
                "certainty": "exact_observed",
            },
            {
                "name": "journal_start", "value": None,
                "basis": "journal_start_sha256",
                "evidence_sha256": ROUND28_JOURNAL_START_SHA256,
                "certainty": "content_sealed_time_unknown",
            },
        ],
        "phases": [
            {
                "phase": "admission", "occurred": True, "start": None,
                "end": None, "certainty": "unknown_interval",
                "evidence_basis": [
                    f"admission_run:{ROUND28_ADMISSION_RUN_ID}"
                ],
            },
            {
                "phase": "prepare_execution", "occurred": True,
                "start": None, "end": stdout_mtime,
                "certainty": "end_bounded_by_mtime",
                "evidence_basis": [
                    f"prepare_stdout_sha256:{ROUND28_PREPARE_STDOUT_SHA256}"
                ],
            },
            {
                "phase": "stage_semantic_validation_refusal", "occurred": True,
                "start": None, "end": None, "certainty": "unknown_interval",
                "evidence_basis": list(ROUND28_AUTHORITY_BASIS),
                "refusal_code": PREPARATION_FAILURE_RUNNER_REFUSAL,
                "reason": PREPARATION_FAILURE_REASON,
            },
            {
                "phase": "gpu_execution", "occurred": False, "start": None,
                "end": None, "certainty": "certified_absent",
                "evidence_basis": [
                    "journal_forensic_absence_scan", "annex_c_adjudication"
                ],
            },
            {
                "phase": "finalization", "occurred": False, "start": None,
                "end": None, "certainty": "certified_absent",
                "evidence_basis": ["journal_complete_json_absent"],
            },
        ],
    }
    phase_path = root / "phase_reconstruction.json"
    _validate_reconstruction_record(phase_path.name, phase)
    _write_compact_json_exclusive_fsync(
        phase_path, phase, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )
    _append_reconstruction_record_event(
        request, name=phase_path.name, h3=h3, adjudications=adjudications,
        prior_count=3,
    )

    query = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if query.returncode != 0:
        raise PreparationArchiveRefusal(
            "PREP_ARCHIVE_RECON_SCHEMA_INVALID",
            query.stderr.decode("utf-8", errors="replace").strip(),
        )
    raw_path = root / "gpu_process_observation_late.raw.txt"
    _write_bytes_exclusive_fsync(
        raw_path, query.stdout, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )
    _append_reconstruction_record_event(
        request, name=raw_path.name, h3=h3, adjudications=adjudications,
        prior_count=4,
    )
    gpu = {
        "record_type": "gpu_process_observation_late",
        "amendment_id": ROUND28_AMENDMENT_ID,
        "observed_at": _record_time(),
        "certifying": False,
        "late": True,
        "method": "nvidia-smi_readonly_query",
        "command": (
            "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory "
            "--format=csv"
        ),
        "raw_output_file": raw_path.name,
        "raw_output_sha256": sha256_file(raw_path),
        "interpretation": (
            "Late observation only; asserts nothing about the historical "
            "2026-07-10 attempt."
        ),
    }
    gpu_path = root / "gpu_process_observation_late.json"
    _validate_reconstruction_record(gpu_path.name, gpu)
    _write_compact_json_exclusive_fsync(
        gpu_path, gpu, code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )
    _append_reconstruction_record_event(
        request, name=gpu_path.name, h3=h3, adjudications=adjudications,
        prior_count=5,
    )

    attestation = {
        "record_type": "preparation_failure_no_gpu_attestation",
        "record_version": 2,
        "amendment_id": ROUND28_AMENDMENT_ID,
        "cell": ROUND28_CELL_ID,
        "chunk": ROUND28_CHUNK_ID,
        "attestation": (
            "No GPU execution occurred in the historical problem-070 "
            "preparation attempt; the attempt terminated at pre-GPU "
            "staged-definition semantic validation."
        ),
        "certifying_elements": [
            "annex_c_adjudication", ROUND28_LEGACY_AUTHORITY_LITERAL,
            "journal_forensic_absence_scan",
        ],
        "journal_forensic_absence_scan": {
            "forbidden_artifact_scan": [],
            "journal_complete_present": False,
            "prepare_stderr_sha256": ROUND28_PREPARE_STDERR_SHA256,
            "prepare_stderr_empty": True,
            "inventory_files": ROUND28_SOURCE_FILE_COUNT,
            "inventory_bytes": ROUND28_SOURCE_TOTAL_BYTES,
            "inventory_domain_sha256": ROUND28_SOURCE_INVENTORY_SHA256,
        },
        "non_certifying_elements": ["gpu_process_observation_late"],
        "late_observation_reference": {
            "file": gpu_path.name, "sha256": sha256_file(gpu_path),
            "certifying": False,
        },
        "reproduction_reference": {
            "file": reproduction_path.name, "sha256": reproduction_sha,
        },
        "campaign_rows": 0,
        "capture_supersession_attempts_consumed": 0,
    }
    attestation_path = root / "preparation_failure_no_gpu_attestation_v2.json"
    _validate_reconstruction_record(attestation_path.name, attestation)
    _write_compact_json_exclusive_fsync(
        attestation_path, attestation,
        code="PREP_ARCHIVE_RECON_SCHEMA_INVALID",
    )
    _append_reconstruction_record_event(
        request, name=attestation_path.name, h3=h3,
        adjudications=adjudications, prior_count=6,
    )
    records = _load_reconstructed_records(request)
    _verify_session_log_prefix(
        request, h3=h3, adjudications=adjudications,
        required_record_count=7,
    )
    return records


def _context_row(source: Path, *, relative_name: str) -> dict:
    if source.is_symlink() or not source.is_file():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", str(source),
        )
    return {
        "relative_path": f"chunk_context/{relative_name}",
        "source_path": str(source.resolve()),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _copy_context(rows: Sequence[Mapping[str, Any]], archive: Path) -> None:
    context_root = archive / "chunk_context"
    context_root.mkdir(mode=0o700, exist_ok=False)
    for row in rows:
        source_text = row["source_path"]
        if source_text.startswith("git:"):
            continue
        source = Path(source_text)
        destination = archive / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
                f"chunk context destination exists: {destination}",
            )
        shutil.copyfile(source, destination)
        if (
            destination.stat().st_size != row["size_bytes"]
            or sha256_file(destination) != row["sha256"]
            or sha256_file(source) != row["sha256"]
        ):
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
                f"chunk context copy drift: {source}",
            )
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
        _fsync_directory(destination.parent)
    _fsync_directory(context_root)


def _same_device(source: Path, destination: Path) -> bool:
    """Separate helper so the cross-device refusal is directly testable."""
    return source.stat().st_dev == destination.stat().st_dev


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move one archive without replacing raced evidence."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
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
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno == errno.EEXIST:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            f"destination exists: {destination}",
        )
    raise PreparationArchiveRefusal(
        "PREPARATION_ARCHIVE_SOURCE_MISSING",
        f"atomic rename {source} -> {destination}: "
        f"{os.strerror(observed_errno)}",
    )


def _acquire_campaign_lock(campaign_root: Path):
    """Serialize the authorized archive CLI with capture and peer archives."""

    path = campaign_root / ".capture.lock"
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"campaign writer already holds {path}",
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"archive_pid={os.getpid()}\n")
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def archive_preparation_failure(request: ArchiveRequest) -> dict:
    """Verify, move, and durably ledger one preparation failure."""
    if request.context_mode not in {
        CONTEXT_MODE_CONTEMPORANEOUS, CONTEXT_MODE_RECONSTRUCTED,
    }:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            f"unknown context mode: {request.context_mode}",
        )
    reconstructed_records: dict[str, dict] | None = None
    reconstructed_adjudications: dict[str, str] | None = None
    h3: str | None = None
    if request.context_mode == CONTEXT_MODE_RECONSTRUCTED:
        _verify_reconstructed_allowlist(request)
        h3 = _verify_h3_identity(request)
        reconstructed_adjudications = _adjudication_hashes(request)
        _verify_session_log_prefix(
            request, h3=h3, adjudications=reconstructed_adjudications,
            required_record_count=7,
        )
        reconstructed_records = _load_reconstructed_records(request)
    elif any(
        value is None for value in (
            request.runner_refusal_record, request.phase_log,
            request.gpu_process_observation,
        )
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            "contemporaneous context records are required",
        )
    if request.journals_root.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            f"journals root is a symlink: {request.journals_root}",
        )
    journals = request.journals_root.resolve()
    if journals.name != "journals" or journals.is_symlink() or not journals.is_dir():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", str(journals),
        )
    children = sorted(journals.iterdir())
    if (
        len(children) != 1 or children[0].name != request.cell_id
        or children[0].is_symlink() or not children[0].is_dir()
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING",
            f"journals must contain exactly {request.cell_id}",
        )
    journal = children[0]
    start_path = journal / "journal_start.json"
    complete_path = journal / "journal_complete.json"
    if complete_path.exists() or complete_path.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"completion exists: {complete_path}",
        )
    start = _json_object(start_path, code="PREPARATION_ARCHIVE_SOURCE_MISSING")
    _verify_start_seal(start, path=start_path)
    expected_start = {
        "schema_version": 2,
        "state": "capture_started",
        "source_commit": request.original_head,
        "capture_matrix_sha256": request.original_matrix_file_sha256,
        "matrix_seal_sha256": request.original_matrix_seal_sha256,
        "capture_preflight_sha256": request.original_preflight_sha256,
        "admission_run_id": request.original_admission_run_id,
    }
    drift = [
        f"{key}={start.get(key)!r}" for key, value in expected_start.items()
        if start.get(key) != value
    ]
    if drift:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "; ".join(drift),
        )
    if start.get("supersedes") is not None:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "supersedes must be null",
        )
    if (
        request.original_matrix_path.is_symlink()
        or not request.original_matrix_path.is_file()
        or sha256_file(request.original_matrix_path)
        != request.original_matrix_file_sha256
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "matrix file pin mismatch",
        )
    matrix = _json_object(
        request.original_matrix_path, code="PREPARATION_ARCHIVE_SOURCE_MISSING",
    )
    if matrix.get("matrix_seal_sha256") != request.original_matrix_seal_sha256:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "matrix seal mismatch",
        )
    preflight_path = Path(str(start.get("capture_preflight_path", "")))
    if (
        not preflight_path.is_absolute() or preflight_path.is_symlink()
        or not preflight_path.is_file()
        or sha256_file(preflight_path) != request.original_preflight_sha256
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "preflight pin mismatch",
        )

    run_rows: list[dict] = []
    try:
        raw_run_rows = request.chunk_run_journal.read_text().splitlines()
        run_rows = [json.loads(raw) for raw in raw_run_rows]
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", f"run journal: {exc}",
        ) from exc
    if (
        not run_rows or not all(isinstance(row, dict) for row in run_rows)
        or run_rows[0].get("session_id") != request.original_session_id
        or run_rows[0].get("chunk_id") != request.chunk_id
        or run_rows[0].get("source_commit") != request.original_head
        or run_rows[0].get("capture_matrix_sha256")
        != request.original_matrix_file_sha256
        or run_rows[0].get("matrix_seal_sha256")
        != request.original_matrix_seal_sha256
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "run journal session mismatch",
        )
    admissions = [
        row for row in run_rows if row.get("event") == "capture_admission"
        and row.get("run_id") == request.original_admission_run_id
        and row.get("preflight_sha256") == request.original_preflight_sha256
    ]
    if len(admissions) != 1 or any(
        row.get("session_id") != request.original_session_id for row in run_rows
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "admission binding mismatch",
        )

    # The immutable V2 pin was computed from the single journal root, not the
    # enclosing journals/ parent; preserve that exact relative-path domain.
    rows = content_inventory(journal)
    observed_pin = source_inventory_sha256(rows)
    if (
        observed_pin != request.expected_inventory_sha256
        or len(rows) != request.expected_file_count
        or sum(row["size_bytes"] for row in rows) != request.expected_total_bytes
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
            f"pin={observed_pin}; files={len(rows)}; "
            f"bytes={sum(row['size_bytes'] for row in rows)}",
        )
    forbidden = _forbidden_matches(rows)
    if forbidden:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", json.dumps(forbidden, sort_keys=True),
        )

    stdout_rel = (
        Path("cells") / request.cell_id / "logs" / "sol_prepare.stdout.log"
    )
    stderr_rel = stdout_rel.with_name("sol_prepare.stderr.log")
    manifest_rel = (
        Path("cells") / request.cell_id / "sol_stage"
        / "capture_prepare_manifest.json"
    )
    staged_definition_rel = manifest_rel.with_name("definition.json")
    staged_module_rel = manifest_rel.with_name("benchmark_kernel.so")
    stdout_path, stderr_path = journal / stdout_rel, journal / stderr_rel
    manifest_path = journal / manifest_rel
    prepare_manifest = _json_object(
        manifest_path, code="PREPARATION_ARCHIVE_NOT_PREGPU",
    )
    try:
        stdout_json = json.loads(stdout_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"prepare stdout: {exc}",
        ) from exc
    try:
        stderr_bytes = stderr_path.read_bytes()
    except OSError as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", f"prepare stderr: {exc}",
        ) from exc
    if stdout_json != prepare_manifest or stderr_bytes != b"":
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_NOT_PREGPU", "prepare streams are not manifest/empty",
        )
    sources = prepare_manifest.get("source_inputs")
    source_definition_entry = (
        sources.get("definition") if isinstance(sources, dict) else None
    )
    source_definition_path = Path(
        source_definition_entry.get("path", "")
        if isinstance(source_definition_entry, dict) else ""
    )
    source_definition_claim = (
        source_definition_entry.get("sha256")
        if isinstance(source_definition_entry, dict) else None
    )
    try:
        pinned_source_ok = (
            source_definition_path.is_absolute()
            and not source_definition_path.is_symlink()
            and source_definition_path.is_file()
            and source_definition_claim == request.pinned_source_definition_sha256
            and sha256_file(source_definition_path)
            == request.pinned_source_definition_sha256
            and sha256_file(journal / staged_definition_rel)
            == request.staged_definition_sha256
            and sha256_file(journal / staged_module_rel)
            == request.staged_module_sha256
        )
    except OSError:
        pinned_source_ok = False
    if not pinned_source_ok:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "definition/module pin mismatch",
        )
    refusal: dict | None = None
    phase_log: dict | None = None
    gpu_observation: dict | None = None
    if request.context_mode == CONTEXT_MODE_CONTEMPORANEOUS:
        assert request.runner_refusal_record is not None
        assert request.phase_log is not None
        assert request.gpu_process_observation is not None
        refusal = _validate_refusal_record(request.runner_refusal_record)
        phase_log = _validate_phase_log(request.phase_log)
        gpu_observation = _validate_gpu_observation(
            request.gpu_process_observation
        )
        if refusal["refused_at_utc"] != phase_log.get("refused_at_utc"):
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_NOT_PREGPU",
                "refusal/phase timestamps differ",
            )
        if _parse_utc(
            gpu_observation["observed_at_utc"], label="observed_at_utc",
        ) < _parse_utc(refusal["refused_at_utc"], label="refused_at_utc"):
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_NOT_PREGPU",
                "GPU-process observation predates the semantic refusal",
            )
    else:
        exact_journal, exact_rows, _, _, _ = _verify_reconstruction_journal(
            request
        )
        if exact_journal != journal or exact_rows != rows:
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
                "reconstructed journal verification disagrees",
            )
    packager_source = _verify_packager_source_at_head(
        request.problem_packager_source, original_head=request.original_head,
        require_repo_relative=(
            request.context_mode == CONTEXT_MODE_RECONSTRUCTED
        ),
    )
    execute_start, execute_end = _validate_packager_execute(packager_source)

    if request.preparation_failures_root.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            f"preparation-failure root is a symlink: "
            f"{request.preparation_failures_root}",
        )
    prep_root = request.preparation_failures_root.resolve()
    prep_root.mkdir(parents=True, exist_ok=True)
    if not _same_device(journals, prep_root):
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_CROSS_DEVICE",
            f"journals={journals}; preparation_failures={prep_root}",
        )
    ledger_path = prep_root / PREPARATION_FAILURE_LEDGER_NAME
    existing = load_preparation_failure_ledger(
        ledger_path, verify_archives=True,
    )
    if request.context_mode == CONTEXT_MODE_RECONSTRUCTED and any(
        row.get("cell_id") == ROUND28_CELL_ID
        and set(row) == PREPARATION_FAILURE_RECONSTRUCTED_ENTRY_FIELDS
        for row in existing
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT",
            "the allowlisted reconstructed incident is already sealed",
        )
    cell_attempts = [row for row in existing if row["cell_id"] == request.cell_id]
    if (
        len(cell_attempts) >= PREPARATION_FAILURE_LIMIT
        and not request.override_preparation_attempt_limit
    ):
        raise PreparationArchiveRefusal(
            "PREPARATION_ATTEMPT_LIMIT",
            f"cell={request.cell_id}; attempts={len(cell_attempts)}; "
            f"requires {PREPARATION_FAILURE_OVERRIDE_FLAG}",
        )
    if any(row["source_journal_path"] == str(journal) for row in existing):
        raise PreparationArchiveRefusal(
            "PREPARATION_LEDGER_INCONSISTENT", "source journal already archived",
        )

    created_at = request.created_at_utc or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", "created_at_utc is timezone-naive",
        )
    created_at = created_at.astimezone(timezone.utc)
    group_name = (
        created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        + f"__head-{request.original_head[:8]}"
        + f"__pf-{request.original_preflight_sha256[:8]}"
        + f"__{request.chunk_id}"
    )
    archive = prep_root / group_name
    if archive.exists() or archive.is_symlink():
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_SOURCE_MISSING", f"destination exists: {archive}",
        )
    if request.context_mode == CONTEXT_MODE_CONTEMPORANEOUS:
        if request.rlcr_dir.is_symlink() or not request.rlcr_dir.is_dir():
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_SOURCE_MISSING",
                f"RLCR dir: {request.rlcr_dir}",
            )
        try:
            request.adjudication_ref.resolve(strict=True).relative_to(
                request.rlcr_dir.resolve(strict=True)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_SOURCE_MISSING",
                f"adjudication ref is outside the RLCR dir: {exc}",
            ) from exc
        adjudication_source = request.adjudication_ref
    else:
        # The ledger retains its historical absolute RLCR identity.  Reads use
        # the committed, hash-pinned fixture so reconstructed validation remains
        # self-contained after the local .humanize tree is removed.
        adjudication_source = request.stage_equivalence_adjudication_source

    context_sources: list[tuple[Path, str]] = [
        (request.chunk_run_journal, "chunk_run_journal.jsonl"),
        (adjudication_source, request.adjudication_ref.name),
    ]
    if request.context_mode == CONTEXT_MODE_CONTEMPORANEOUS:
        assert request.runner_refusal_record is not None
        assert request.phase_log is not None
        assert request.gpu_process_observation is not None
        context_sources.extend([
            (request.runner_refusal_record, "runner_refusal_record.json"),
            (request.phase_log, "phase_log.json"),
            (request.gpu_process_observation, "gpu_process_observation.json"),
        ])
    else:
        for path in sorted(
            request.reconstruction_root.rglob("*"),
            key=lambda item: os.fsencode(
                item.relative_to(request.reconstruction_root).as_posix()
            ),
        ):
            if path.is_symlink():
                raise PreparationArchiveRefusal(
                    "PREP_ARCHIVE_RECON_SCHEMA_INVALID",
                    f"construction context symlink: {path}",
                )
            if path.is_file():
                relative = path.relative_to(request.reconstruction_root).as_posix()
                context_sources.append(
                    (path, f"reconstructed/{relative}")
                )
    context_rows = [
        _context_row(path, relative_name=name) for path, name in context_sources
    ]
    packager_row = {
        "relative_path": "chunk_context/problem_packager.py",
        "source_path": str(request.problem_packager_source.resolve()),
        "size_bytes": len(packager_source),
        "sha256": sha256_bytes(packager_source),
    }
    context_rows.append(packager_row)
    context_rows.sort(key=lambda row: os.fsencode(row["relative_path"]))
    inventory = {
        "schema_version": 1,
        "source_inventory_format": SOURCE_INVENTORY_FORMAT,
        "source_inventory_sha256": observed_pin,
        "source_file_count": len(rows),
        "source_total_bytes": sum(row["size_bytes"] for row in rows),
        "files": rows,
        "chunk_context_files": context_rows,
    }

    source_journal_path = str(journal)
    _rename_noreplace(journals, archive)
    _fsync_directory(prep_root)
    _fsync_directory(journals.parent)
    try:
        moved_rows = content_inventory(archive / request.cell_id)
    except PreparationArchiveRefusal as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH", exc.detail,
        ) from exc
    if moved_rows != rows or source_inventory_sha256(moved_rows) != observed_pin:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH", "post-move inventory drift",
        )
    inventory_path = archive / PREPARATION_FAILURE_INVENTORY_NAME
    _write_json_exclusive_fsync(inventory_path, inventory)
    _copy_context(context_rows, archive)
    packager_destination = archive / packager_row["relative_path"]
    try:
        verify_preparation_archive(archive, inventory)
    except PreparationArchiveRefusal as exc:
        raise PreparationArchiveRefusal(
            "PREPARATION_ARCHIVE_INVENTORY_MISMATCH", exc.detail,
        ) from exc

    if request.context_mode == CONTEXT_MODE_CONTEMPORANEOUS:
        assert refusal is not None
        moved_start = archive / request.cell_id / "journal_start.json"
        moved_stdout = archive / request.cell_id / stdout_rel
        moved_stderr = archive / request.cell_id / stderr_rel
        copied_refusal = archive / "chunk_context/runner_refusal_record.json"
        copied_phase = archive / "chunk_context/phase_log.json"
        copied_gpu = archive / "chunk_context/gpu_process_observation.json"
        no_gpu_attestation = {
            "journal_state": {
                "evidence_path": str(moved_start.resolve()),
                "sha256": sha256_file(moved_start),
                "state": "capture_started",
                "journal_complete_present": False,
            },
            "prepare_stdout": {
                "evidence_path": str(moved_stdout.resolve()),
                "sha256": sha256_file(moved_stdout),
                "contains_only_validated_prepare_manifest": True,
            },
            "prepare_stderr": {
                "evidence_path": str(moved_stderr.resolve()),
                "sha256": sha256_file(moved_stderr),
                "empty": True,
            },
            "forbidden_artifact_scan": {
                "evidence_path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path),
                "checked_patterns": list(FORBIDDEN_ARTIFACT_PATTERNS),
                "matches": [],
                "scanned_file_count": len(rows),
            },
            "runner_refusal": {
                "evidence_path": str(copied_refusal.resolve()),
                "sha256": sha256_file(copied_refusal),
                "runner_refusal_code": PREPARATION_FAILURE_RUNNER_REFUSAL,
                "refused_at_utc": refusal["refused_at_utc"],
            },
            "phase_log": {
                "evidence_path": str(copied_phase.resolve()),
                "sha256": sha256_file(copied_phase),
                "last_phase": "prepare_validate",
                "post_prepare_phases": [],
            },
            "problem_packager_execute": {
                "evidence_path": str(packager_destination.resolve()),
                "sha256": sha256_file(packager_destination),
                "source_head": request.original_head,
                "function": "ProblemPackager.execute",
                "citation": (
                    f"tools/sol_execbench/src/sol_execbench/driver/"
                    f"problem_packager.py:{execute_start}-{execute_end}"
                ),
                "writes_driver_and_returns_command_without_execution": True,
            },
            "gpu_process_observation": {
                "evidence_path": str(copied_gpu.resolve()),
                "sha256": sha256_file(copied_gpu),
                "remaining_gpu_processes": [],
            },
        }
    else:
        assert reconstructed_records is not None
        no_gpu_attestation = reconstructed_records[
            "preparation_failure_no_gpu_attestation_v2.json"
        ]
    operator_flag = PREPARATION_FAILURE_OPERATOR_FLAG
    if request.override_preparation_attempt_limit:
        operator_flag += f" {PREPARATION_FAILURE_OVERRIDE_FLAG}"
    entry = {
        "schema_version": 1,
        "entry_id": str(uuid.uuid4()),
        "entry_type": "preparation_failure_archive",
        "created_at_utc": _utc_text(created_at),
        "rlcr_session_id": request.rlcr_session_id,
        "rlcr_dir": str(request.rlcr_dir.absolute()),
        "campaign_round": 28,
        "chunk_id": request.chunk_id,
        "cell_id": request.cell_id,
        "reason_code": PREPARATION_FAILURE_REASON,
        "runner_refusal_code": PREPARATION_FAILURE_RUNNER_REFUSAL,
        "original_head": request.original_head,
        "original_matrix_file_sha256": request.original_matrix_file_sha256,
        "original_matrix_seal_sha256": request.original_matrix_seal_sha256,
        "original_preflight_sha256": request.original_preflight_sha256,
        "original_admission_run_id": request.original_admission_run_id,
        "original_session_id": request.original_session_id,
        "replacement_head": request.replacement_head,
        "source_journal_path": source_journal_path,
        "archive_path": str(archive.resolve()),
        "inventory_file": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "no_gpu_attestation": no_gpu_attestation,
        "pinned_source_definition_sha256": (
            request.pinned_source_definition_sha256
        ),
        "staged_definition_sha256": request.staged_definition_sha256,
        "staged_module_sha256": request.staged_module_sha256,
        "defect_summary": DEFECT_SUMMARY,
        "adjudication_ref": str(request.adjudication_ref.absolute()),
        "operator_flag": operator_flag,
    }
    if request.context_mode == CONTEXT_MODE_RECONSTRUCTED:
        assert reconstructed_adjudications is not None
        entry.update({
            "supersedes": None,
            "reconstruction_records": [
                {
                    "path": f"chunk_context/reconstructed/{name}",
                    "sha256": sha256_file(
                        archive / f"chunk_context/reconstructed/{name}"
                    ),
                }
                for name in ROUND28_RECONSTRUCTION_RECORD_ORDER
            ],
            "authority": {
                "amendment_id": ROUND28_AMENDMENT_ID,
                "supersession_status": "superseded_by_v2",
                "corrected_adjudication_sha256": (
                    reconstructed_adjudications["corrected"]
                ),
                "superseded_adjudication_sha256": (
                    reconstructed_adjudications["superseded"]
                ),
                "correction_id": ROUND28_CORRECTION_ID,
                "correction_adjudication_sha256": (
                    reconstructed_adjudications["correction"]
                ),
            },
        })
    validate_preparation_failure_entry(
        entry, ledger_path=ledger_path, line_number=len(existing) + 1,
        verify_archive=False,
    )
    if request.context_mode == CONTEXT_MODE_RECONSTRUCTED:
        _verify_reconstructed_archive_entry(
            entry, archive=archive, inventory=inventory,
            label=f"{ledger_path}:{len(existing) + 1}",
            require_archive_log_event=False,
        )
        _verify_session_log_prefix(
            request, h3=str(h3), adjudications=reconstructed_adjudications,
            required_record_count=7,
        )
    else:
        validate_preparation_failure_entry(
            entry, ledger_path=ledger_path, line_number=len(existing) + 1,
            verify_archive=True,
        )
    _append_jsonl_fsync(ledger_path, entry)
    if request.context_mode == CONTEXT_MODE_RECONSTRUCTED:
        _append_jsonl_fsync(request.session_log_path, {
            "event": "preparation_failure_archived",
            "entry_id": entry["entry_id"],
            "archive_path": entry["archive_path"],
            "inventory_sha256": entry["inventory_sha256"],
        })
    # Close the newly appended row and all historical rows before success.
    load_preparation_failure_ledger(ledger_path, verify_archives=True)
    return entry


def _current_head_clean() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "-uall"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=False,
    )
    return head.stdout.strip(), head.returncode == 0 and status.returncode == 0 and not status.stdout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-preparation-failure", action="store_true")
    parser.add_argument("--build-reconstructed-context", action="store_true")
    parser.add_argument(
        "--context-mode",
        choices=(CONTEXT_MODE_CONTEMPORANEOUS, CONTEXT_MODE_RECONSTRUCTED),
        default=CONTEXT_MODE_CONTEMPORANEOUS,
    )
    parser.add_argument("--journals-root", required=True, type=Path)
    parser.add_argument("--preparation-failures-root", required=True, type=Path)
    parser.add_argument("--original-matrix", required=True, type=Path)
    parser.add_argument("--chunk-run-journal", required=True, type=Path)
    parser.add_argument("--runner-refusal-record", type=Path)
    parser.add_argument("--phase-log", type=Path)
    parser.add_argument("--gpu-process-observation", type=Path)
    parser.add_argument("--replacement-head", required=True)
    parser.add_argument(
        "--rlcr-dir", type=Path,
        default=ROUND28_RLCR_DIR,
    )
    parser.add_argument(
        "--adjudication-ref", type=Path,
        default=ROUND28_STAGE_EQUIVALENCE_ADJUDICATION,
    )
    parser.add_argument(
        "--stage-equivalence-adjudication-source", type=Path,
        default=ROUND28_STAGE_EQUIVALENCE_SOURCE,
    )
    parser.add_argument(
        "--problem-packager-source", type=Path,
        default=ROUND28_PROBLEM_PACKAGER_SOURCE,
    )
    parser.add_argument(
        "--reconstruction-root", type=Path,
        default=ROUND28_RECONSTRUCTION_ROOT,
    )
    parser.add_argument(
        "--session-log", type=Path, default=ROUND28_SESSION_LOG,
    )
    parser.add_argument(
        "--corrected-adjudication-ref", type=Path,
        default=ROUND28_CORRECTED_ADJUDICATION,
    )
    parser.add_argument(
        "--superseded-adjudication-ref", type=Path,
        default=ROUND28_SUPERSEDED_ADJUDICATION,
    )
    parser.add_argument(
        "--correction-adjudication-ref", type=Path,
        default=ROUND28_CORRECTION_ADJUDICATION,
    )
    parser.add_argument(
        "--correction2-adjudication-ref", type=Path,
        default=ROUND28_CORRECTION2_ADJUDICATION,
    )
    parser.add_argument("--override-preparation-attempt-limit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.archive_preparation_failure == args.build_reconstructed_context
    ):
        print(
            "archive_preparation_failure: REFUSED "
            "[PREPARATION_ARCHIVE_FLAG_REQUIRED] "
            "choose exactly one explicit build/archive action",
            file=sys.stderr,
        )
        return 1
    if (
        args.journals_root.resolve() != ROUND28_JOURNALS_ROOT
        or args.preparation_failures_root.resolve()
        != ROUND28_PREPARATION_FAILURES_ROOT
    ):
        print(
            "archive_preparation_failure: REFUSED "
            "[PREPARATION_ARCHIVE_SOURCE_MISSING] "
            "the Round-28 archival CLI requires the adjudicated fixed source "
            "and preparation-failure roots",
            file=sys.stderr,
        )
        return 1
    if args.context_mode != CONTEXT_MODE_RECONSTRUCTED:
        head, clean = _current_head_clean()
        if head != args.replacement_head or not clean:
            print(
                "archive_preparation_failure: REFUSED "
                "[PREPARATION_ARCHIVE_SOURCE_MISSING] "
                f"replacement HEAD/clean-tree mismatch: head={head} clean={clean}",
                file=sys.stderr,
            )
            return 1
    request = ArchiveRequest(
        journals_root=args.journals_root,
        preparation_failures_root=args.preparation_failures_root,
        original_matrix_path=args.original_matrix,
        chunk_run_journal=args.chunk_run_journal,
        runner_refusal_record=args.runner_refusal_record,
        phase_log=args.phase_log,
        gpu_process_observation=args.gpu_process_observation,
        adjudication_ref=args.adjudication_ref,
        rlcr_dir=args.rlcr_dir,
        replacement_head=args.replacement_head,
        problem_packager_source=args.problem_packager_source,
        override_preparation_attempt_limit=args.override_preparation_attempt_limit,
        context_mode=args.context_mode,
        reconstruction_root=args.reconstruction_root,
        session_log_path=args.session_log,
        corrected_adjudication_ref=args.corrected_adjudication_ref,
        superseded_adjudication_ref=args.superseded_adjudication_ref,
        correction_adjudication_ref=args.correction_adjudication_ref,
        correction2_adjudication_ref=args.correction2_adjudication_ref,
        stage_equivalence_adjudication_source=(
            args.stage_equivalence_adjudication_source
        ),
    )
    campaign_lock = None
    try:
        campaign_lock = _acquire_campaign_lock(ROUND28_CAMPAIGN_ROOT)
        if args.build_reconstructed_context:
            if args.context_mode != CONTEXT_MODE_RECONSTRUCTED:
                raise PreparationArchiveRefusal(
                    "SOL_PREP_RECON_ALLOWLIST_VIOLATION",
                    "the builder requires --context-mode reconstructed",
                )
            entry = build_reconstructed_context(request)
        else:
            entry = archive_preparation_failure(request)
    except PreparationArchiveRefusal as exc:
        print(
            f"archive_preparation_failure: REFUSED [{exc.code}] {exc.detail}",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            "archive_preparation_failure: REFUSED "
            f"[PREPARATION_ARCHIVE_SOURCE_MISSING] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if campaign_lock is not None:
            campaign_lock.close()
    print(json.dumps(entry, sort_keys=True))
    return 0


__all__ = [
    "ArchiveRequest",
    "CONTEXT_INVENTORY_FIELDS",
    "FILE_INVENTORY_FIELDS",
    "FORBIDDEN_ARTIFACT_PATTERNS",
    "INVENTORY_FIELDS",
    "PREPARATION_FAILURE_BACKREF_FIELDS",
    "PREPARATION_FAILURE_ENTRY_FIELDS",
    "PREPARATION_FAILURE_ENTRY_SCHEMA",
    "PREPARATION_FAILURE_INVENTORY_NAME",
    "PREPARATION_FAILURE_LEDGER_NAME",
    "PREPARATION_FAILURE_LIMIT",
    "PREPARATION_FAILURE_ROOT_NAME",
    "PreparationArchiveRefusal",
    "ROUND28_SOURCE_INVENTORY_SHA256",
    "ROUND28_JOURNALS_ROOT",
    "ROUND28_PREPARATION_FAILURES_ROOT",
    "SOURCE_INVENTORY_FORMAT",
    "archive_preparation_failure",
    "content_inventory",
    "load_archive_inventory",
    "load_preparation_failure_ledger",
    "preparation_failure_backrefs",
    "source_inventory_bytes",
    "source_inventory_sha256",
    "validate_preparation_failure_entry",
    "verify_preparation_archive",
]


if __name__ == "__main__":
    raise SystemExit(main())
