#!/usr/bin/env python3
"""Preserve the fixed cross-HEAD H3 capture-driver defect attempt.

The archived attempt is execution evidence, but it is permanently
non-authoritative: it contributes no campaign row and none of its generated
artifacts may be reused.  The CLI recognizes only the audited H3 incident.
Importing this module is side-effect free; the strict ledger reader is shared
by the capture runner, DONE verifier, and campaign accounting.
"""
from __future__ import annotations

import argparse
import fcntl
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
CAPTURE_DEFECT_ROOT_NAME = "capture_defect_attempts"
CAPTURE_DEFECT_LEDGER_NAME = "capture_defect_ledger.jsonl"
CAPTURE_DEFECT_LEDGER_PENDING_NAME = "capture_defect_ledger.jsonl.pending"
CAPTURE_DEFECT_INVENTORY_NAME = "inventory.json"
CAPTURE_DEFECT_OWNER_SUFFIX = ".owner.json"
CAPTURE_DEFECT_ENTRY_SCHEMA = (
    REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
    / "capture_defect_ledger_entry.schema.json"
)
INVENTORY_FORMAT = "c_locale_sorted_relative_path_tab_size_bytes_tab_sha256_lf_v1"

FIXED_ORIGINAL_HEAD = "9bfc1fc27fd24f062c7a1691e3a7d7b8b0b963c2"
FIXED_MATRIX_FILE_SHA256 = (
    "e725709cdd7f4302befd123d391dcdffa233b40c37120b3226e39f56062c412d"
)
FIXED_MATRIX_SEAL_SHA256 = (
    "dabb26187c9617d5fd08c7ba6737af969f9202d7f7d3d164ad0a83407726ea5d"
)
FIXED_PREFLIGHT_SHA256 = (
    "b918531281b48ac3358b6a1e866cd4fa3da4c26babdcc4cf29412b4c833af898"
)
FIXED_SESSION_ID = "88f01256-168a-4880-ac52-204014fe0040"
FIXED_ADMISSION_RUN_ID = (
    "e76313a5791db2c911a99113e9ca917dc354e2b32ee619d6e6d5660297dd3f35"
)
FIXED_CHUNK_ID = "sol_execbench-000-bootstrap"
FIXED_CELL_ID = (
    "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation"
)
FIXED_JOURNAL_FILE_COUNT = 32
FIXED_JOURNAL_TOTAL_BYTES = 3_888_602
FIXED_JOURNAL_INVENTORY_SHA256 = (
    "b6569f61caf98a29e2b1a7711208f88938658f3d27582b4062fac1bf95fcbc5a"
)
FIXED_STAGE_FILE_COUNT = 33
FIXED_STAGE_TOTAL_BYTES = 3_889_540
FIXED_STAGE_INVENTORY_SHA256 = (
    "2bf29fafa773052fc2c2bb5b4b982effb7c1f6ff16813c0d80af03eb41ab4d6f"
)
FIXED_CHUNK_RUN_JOURNAL_SHA256 = (
    "5f7ab4ba11a8d1aabed9d30b19077613f22d9c29e1c16d239c67fa8d5c447713"
)
FIXED_JOURNAL_START_SHA256 = (
    "e8cd2eb364a269c0886e8c37564723bc7efb7c7e6c6bb96c0a030f5ec40b9e01"
)
FIXED_LABEL_RESULT_SHA256 = (
    "05732f22bb839cb4c533609a97900bb21e4e9b12e62ba714665fe4833bd71d0a"
)
FIXED_EVAL_DRIVER_SHA256 = (
    "0e68033a46a449d514e56551eccd7a729d56daa3ca7e278de5f251f73501c655"
)
FIXED_PREPARE_MANIFEST_SHA256 = (
    "764c4b99a9f77b7e952dc32e38bd4a1f418a6985da1dc59a14931e703e1b49e2"
)
FIXED_SELECTION_SHA256 = (
    "a1cacb6fb823a6788ff321f298bfb00d4222fef7c511485d21976ae2edd61cba"
)
FIXED_EMPTY_JOURNAL_DIRECTORY = (
    "cells/"
    f"{FIXED_CELL_ID}/labels/sm_86/"
    "problem_070_mamba2_fused_intra_chunk_diagonal_computation/workloads"
)
REASON_CODE = "CAPTURE_DRIVER_INTEGRITY_FALSE_POSITIVE_BEFORE_ROI"
DEFECT_SUMMARY = (
    "capture mode intentionally bound time_runnable to None while the integrity "
    "snapshot still required the canonical timing function"
)
OPERATOR_FLAG = "--archive-fixed-h3-capture-defect"

FILE_ROW_FIELDS = frozenset({"relative_path", "size_bytes", "sha256"})
CONTEXT_ROW_FIELDS = frozenset({
    "archive_relative_path", "source_path", "size_bytes", "sha256",
})
BACKREF_FIELDS = frozenset({
    "ledger_entry_id", "ledger_path", "archive_path", "original_head",
    "reason_code",
})
ENTRY_FIELDS = frozenset({
    "schema_version", "entry_id", "entry_type", "created_at_utc",
    "original_head", "replacement_head", "chunk_id", "cell_id",
    "original_matrix_file_sha256", "original_matrix_seal_sha256",
    "original_preflight_sha256", "original_session_id",
    "original_admission_run_id", "source_stage_path",
    "source_journals_path", "archive_path", "inventory_file",
    "inventory_sha256", "journal_inventory_sha256", "journal_file_count",
    "journal_total_bytes", "stage_inventory_sha256", "stage_file_count",
    "stage_total_bytes", "chunk_run_journal_sha256", "journal_start_sha256",
    "label_result_sha256", "eval_driver_sha256", "prepare_manifest_sha256",
    "selection_sha256", "reason_code", "defect_summary", "authoritative",
    "reusable", "campaign_rows", "operator_flag",
})
INVENTORY_FIELDS = frozenset({
    "schema_version", "inventory_format", "journal", "stage_before_move",
    "chunk_context_files",
})
SEALED_INVENTORY_FIELDS = frozenset({
    "file_count", "total_bytes", "domain_sha256", "files",
})


class CaptureDefectRefusal(RuntimeError):
    """Stable typed refusal for producer and all consumers."""

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
    except (AttributeError, ValueError):
        return False
    return parsed.version == 4 and str(parsed) == value


def _utc_text(value: datetime | None = None) -> str:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "created_at_utc is timezone-naive",
        )
    return observed.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _parse_utc(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label} is not UTC Z time",
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: {exc}",
        ) from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label} is not UTC",
        )


def _json_object(path: Path, *, code: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureDefectRefusal(code, f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureDefectRefusal(code, f"{path}: root is not object")
    return value


def _validate_schema(value: Mapping[str, Any], *, code: str) -> None:
    try:
        import jsonschema

        schema = json.loads(CAPTURE_DEFECT_ENTRY_SCHEMA.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ImportError) as exc:
        raise CaptureDefectRefusal(code, f"{CAPTURE_DEFECT_ENTRY_SCHEMA}: {exc}") from exc
    except Exception as exc:
        if exc.__class__.__module__.split(".", 1)[0] != "jsonschema":
            raise
        raise CaptureDefectRefusal(code, str(exc)) from exc


def content_inventory(
    root: Path, *, expected_empty_directories: Sequence[str] = (),
) -> list[dict]:
    """Return a non-vacuous, symlink-free, byte-sorted file inventory.

    Empty directories are outside the file-domain hash, so callers must bind
    their complete relative-path set explicitly.  The fixed H3 attempt has one
    audited empty workloads directory; every other empty directory is refused.
    """
    if root.is_symlink() or not root.is_dir():
        raise CaptureDefectRefusal("CAPTURE_DEFECT_SOURCE_MISMATCH", str(root))
    rows: list[dict] = []
    empty_directories: list[str] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: os.fsencode(item.relative_to(root).as_posix()),
    ):
        if path.is_symlink():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH", f"symlink: {path}",
            )
        if path.is_dir():
            try:
                if not any(path.iterdir()):
                    empty_directories.append(path.relative_to(root).as_posix())
            except OSError as exc:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_SOURCE_MISMATCH", f"directory: {path}: {exc}",
                ) from exc
        elif path.is_file():
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        else:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH",
                f"special filesystem node is outside inventory domain: {path}",
            )
    expected_empty = sorted(
        expected_empty_directories, key=lambda value: os.fsencode(value),
    )
    if empty_directories != expected_empty:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH",
            "empty-directory topology mismatch: "
            f"observed={empty_directories!r}; expected={expected_empty!r}",
        )
    if not rows:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"empty inventory: {root}",
        )
    return rows


def inventory_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: os.fsencode(str(row["relative_path"])))
    return b"".join(
        f"{row['relative_path']}\t{row['size_bytes']}\t{row['sha256']}\n".encode()
        for row in ordered
    )


def inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(inventory_bytes(rows))


def _sealed_inventory(rows: list[dict]) -> dict:
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "domain_sha256": inventory_sha256(rows),
        "files": rows,
    }


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_ARCHIVE_EXISTS", str(path),
        ) from exc
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _copy_exclusive_fsync(source: Path, destination: Path) -> dict:
    if source.is_symlink() or not source.is_file():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"context is not regular: {source}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_ARCHIVE_EXISTS", str(destination),
        ) from exc
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb", closefd=False) as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
    finally:
        os.close(fd)
    _fsync_directory(destination.parent)
    return {
        "archive_relative_path": destination.as_posix(),
        "source_path": str(source.resolve()),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _append_jsonl_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class IncidentContract:
    original_head: str = FIXED_ORIGINAL_HEAD
    matrix_file_sha256: str = FIXED_MATRIX_FILE_SHA256
    matrix_seal_sha256: str = FIXED_MATRIX_SEAL_SHA256
    preflight_sha256: str = FIXED_PREFLIGHT_SHA256
    session_id: str = FIXED_SESSION_ID
    admission_run_id: str = FIXED_ADMISSION_RUN_ID
    chunk_id: str = FIXED_CHUNK_ID
    cell_id: str = FIXED_CELL_ID
    journal_file_count: int = FIXED_JOURNAL_FILE_COUNT
    journal_total_bytes: int = FIXED_JOURNAL_TOTAL_BYTES
    journal_inventory_sha256: str = FIXED_JOURNAL_INVENTORY_SHA256
    stage_file_count: int = FIXED_STAGE_FILE_COUNT
    stage_total_bytes: int = FIXED_STAGE_TOTAL_BYTES
    stage_inventory_sha256: str = FIXED_STAGE_INVENTORY_SHA256
    chunk_run_journal_sha256: str = FIXED_CHUNK_RUN_JOURNAL_SHA256
    journal_start_sha256: str = FIXED_JOURNAL_START_SHA256
    label_result_sha256: str = FIXED_LABEL_RESULT_SHA256
    eval_driver_sha256: str = FIXED_EVAL_DRIVER_SHA256
    prepare_manifest_sha256: str = FIXED_PREPARE_MANIFEST_SHA256
    selection_sha256: str = FIXED_SELECTION_SHA256


FIXED_CONTRACT = IncidentContract()


@dataclass(frozen=True)
class ArchiveRequest:
    output_root: Path
    matrix_path: Path
    preflight_path: Path
    replacement_head: str
    contract: IncidentContract = FIXED_CONTRACT
    created_at_utc: datetime | None = None
    entry_id: str | None = None


def _source_paths(request: ArchiveRequest) -> tuple[Path, Path, Path]:
    contract = request.contract
    stage = (
        request.output_root.resolve() / ".staging" / contract.original_head
        / contract.chunk_id / contract.matrix_file_sha256
    )
    return stage, stage / "journals", stage / "chunk_run_journal.jsonl"


def _expected_empty_directories(
    contract: IncidentContract, *, stage_scope: bool,
) -> tuple[str, ...]:
    if contract != FIXED_CONTRACT:
        return ()
    relative = FIXED_EMPTY_JOURNAL_DIRECTORY
    if stage_scope:
        relative = f"journals/{contract.cell_id}/{relative}"
    return (relative,)


def _require_directory_chain(root: Path, target: Path, *, label: str) -> None:
    """Reject symlinks/non-directories at every lexical ancestor."""
    root = root.absolute()
    target = target.absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH",
            f"{label} escapes root: {target}",
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink() or not cursor.is_dir():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH",
                f"{label} ancestor is not a regular directory: {cursor}",
            )


def _require_replacement_contract(request: ArchiveRequest) -> None:
    if request.contract != FIXED_CONTRACT:
        return
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=False,
    )
    observed_head = head.stdout.strip() if head.returncode == 0 else ""
    if observed_head != request.replacement_head:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH",
            f"replacement_head={request.replacement_head!r}; HEAD={observed_head!r}",
        )
    if not _replacement_commit_has_repair(request.replacement_head):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH",
            "replacement HEAD does not contain the capture-only time_runnable exclusion",
        )


def _replacement_commit_has_repair(commit: str) -> bool:
    replacement = subprocess.run(
        [
            "git", "show",
            f"{commit}:tools/sol_execbench/src/sol_execbench/"
            "driver/templates/eval_driver.py",
        ],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    required = (
        b'_CRITICAL_NAMES = [\n    "compute_error_stats",',
        b'if _CAPTURE_CONTRACT is None:\n    _CRITICAL_NAMES.insert(0, "time_runnable")',
    )
    return replacement.returncode == 0 and not any(
        token not in replacement.stdout for token in required
    )


def _require_fixed_source_paths(request: ArchiveRequest) -> None:
    if request.contract != FIXED_CONTRACT:
        return
    inputs = request.output_root.resolve() / "inputs" / "r28_9bfc1fc2"
    expected = {
        request.matrix_path: inputs / "sm86_reference_capture_matrix.json",
        request.preflight_path: inputs / "preflight_sol_bootstrap.json",
    }
    for supplied, required in expected.items():
        if supplied.is_symlink() or not supplied.is_file():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH", f"source path: {supplied}",
            )
        if supplied.resolve() != required.resolve(strict=False):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH",
                f"noncanonical fixed source path: {supplied}; expected={required}",
            )
        boundary = request.output_root.resolve()
        cursor = supplied.parent
        while cursor != boundary:
            if cursor.is_symlink():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_SOURCE_MISMATCH",
                    f"source path has symlink ancestor: {cursor}",
                )
            if boundary not in cursor.parents:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_SOURCE_MISMATCH",
                    f"source path escapes output root: {supplied}",
                )
            cursor = cursor.parent


def _verify_incident(request: ArchiveRequest) -> tuple[Path, Path, list[dict], list[dict]]:
    _require_replacement_contract(request)
    _require_fixed_source_paths(request)
    contract = request.contract
    stage, journals, run_journal = _source_paths(request)
    _require_directory_chain(request.output_root, stage, label="stage")
    expected_suffix = Path(
        ".staging", contract.original_head, contract.chunk_id,
        contract.matrix_file_sha256,
    )
    if Path(*stage.parts[-4:]) != expected_suffix:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"stage path drift: {stage}",
        )
    if journals.is_symlink() or not journals.is_dir():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"journals unavailable: {journals}",
        )
    children = sorted(journals.iterdir())
    if (
        len(children) != 1 or children[0].name != contract.cell_id
        or children[0].is_symlink() or not children[0].is_dir()
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH",
            f"journals must contain exactly {contract.cell_id}",
        )
    journal = children[0]
    if (journal / "journal_complete.json").exists():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "attempt is already complete",
        )
    journal_rows = content_inventory(
        journal,
        expected_empty_directories=_expected_empty_directories(
            contract, stage_scope=False,
        ),
    )
    stage_rows = content_inventory(
        stage,
        expected_empty_directories=_expected_empty_directories(
            contract, stage_scope=True,
        ),
    )
    observed = (
        len(journal_rows), sum(row["size_bytes"] for row in journal_rows),
        inventory_sha256(journal_rows), len(stage_rows),
        sum(row["size_bytes"] for row in stage_rows), inventory_sha256(stage_rows),
    )
    expected = (
        contract.journal_file_count, contract.journal_total_bytes,
        contract.journal_inventory_sha256, contract.stage_file_count,
        contract.stage_total_bytes, contract.stage_inventory_sha256,
    )
    if observed != expected:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_INVENTORY_MISMATCH",
            f"observed={observed!r}; expected={expected!r}",
        )

    journal_start = journal / "journal_start.json"
    cell = journal / "cells" / contract.cell_id
    label_result = cell / "sol_stage" / "evidence" / "label-result.json"
    selection = cell / "sol_stage" / "evidence" / "label-selection.json"
    eval_driver = cell / "sol_stage" / "eval_driver.py"
    prepare_manifest = cell / "sol_stage" / "capture_prepare_manifest.json"
    exact_hashes = {
        run_journal: contract.chunk_run_journal_sha256,
        journal_start: contract.journal_start_sha256,
        label_result: contract.label_result_sha256,
        selection: contract.selection_sha256,
        eval_driver: contract.eval_driver_sha256,
        prepare_manifest: contract.prepare_manifest_sha256,
        request.matrix_path: contract.matrix_file_sha256,
        request.preflight_path: contract.preflight_sha256,
    }
    drift = [
        f"{path}={sha256_file(path) if path.is_file() else 'absent'}"
        for path, expected_hash in exact_hashes.items()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash
    ]
    if drift:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "; ".join(drift),
        )

    start = _json_object(journal_start, code="CAPTURE_DEFECT_SOURCE_MISMATCH")
    start_seal = start.get("journal_start_seal_sha256")
    if start_seal != canonical_sha256({
        key: value for key, value in start.items()
        if key != "journal_start_seal_sha256"
    }):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "journal start seal mismatch",
        )
    start_expected = {
        "schema_version": 2,
        "state": "capture_started",
        "source_commit": contract.original_head,
        "capture_matrix_sha256": contract.matrix_file_sha256,
        "matrix_seal_sha256": contract.matrix_seal_sha256,
        "capture_preflight_sha256": contract.preflight_sha256,
        "admission_run_id": contract.admission_run_id,
    }
    if any(start.get(key) != value for key, value in start_expected.items()):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "journal start identity drift",
        )
    matrix = _json_object(
        request.matrix_path, code="CAPTURE_DEFECT_SOURCE_MISMATCH",
    )
    if matrix.get("matrix_seal_sha256") != contract.matrix_seal_sha256:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "matrix seal drift",
        )
    try:
        run_rows = [json.loads(raw) for raw in run_journal.read_text().splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"run journal: {exc}",
        ) from exc
    if len(run_rows) != 2 or not all(isinstance(row, dict) for row in run_rows):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "run journal cardinality",
        )
    session, admission = run_rows
    if (
        session.get("event") != "session_started"
        or session.get("source_commit") != contract.original_head
        or session.get("session_id") != contract.session_id
        or session.get("chunk_id") != contract.chunk_id
        or session.get("capture_matrix_sha256") != contract.matrix_file_sha256
        or session.get("matrix_seal_sha256") != contract.matrix_seal_sha256
        or admission.get("event") != "capture_admission"
        or admission.get("session_id") != contract.session_id
        or admission.get("chunk_id") != contract.chunk_id
        or admission.get("run_id") != contract.admission_run_id
        or admission.get("preflight_sha256") != contract.preflight_sha256
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "run journal identity drift",
        )

    result = _json_object(label_result, code="CAPTURE_DEFECT_SOURCE_MISMATCH")
    _require_false_positive_before_roi(
        result, code="CAPTURE_DEFECT_SOURCE_MISMATCH",
    )
    try:
        original_driver = subprocess.run(
            [
                "git", "show",
                f"{contract.original_head}:tools/sol_execbench/src/sol_execbench/driver/templates/eval_driver.py",
            ],
            cwd=REPO_ROOT, capture_output=True, check=False,
        )
    except OSError as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"git show: {exc}",
        ) from exc
    if (
        original_driver.returncode != 0
        or sha256_bytes(original_driver.stdout) != contract.eval_driver_sha256
        or b"else:\n    time_runnable = None" not in original_driver.stdout
        or b'_CRITICAL_NAMES = [\n    "time_runnable",' not in original_driver.stdout
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", "original capture-driver defect no longer re-proves",
        )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", contract.original_head, request.replacement_head],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if ancestry.returncode != 0:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH",
            f"replacement HEAD does not descend from original: {ancestry.stderr.strip()}",
        )
    return stage, journal, journal_rows, stage_rows


def _require_false_positive_before_roi(value: Mapping[str, Any], *, code: str) -> None:
    if (
        value.get("status") != "reward_hack_detected"
        or value.get("error") != {
            "detail": "Eval driver integrity violated: 'time_runnable' has been monkey-patched",
            "exception_type": "RewardHackDetected",
            "type": "reward_hack_detected",
        }
        or value.get("outside_roi") != {
            "all_completed_before_profiler_start": False,
            "correctness_rounds": 1,
            "correctness_user_fn_calls": 1,
            "warmup_user_fn_calls": 0,
        }
        or value.get("roi") != {
            "candidate_user_fn_calls": 0,
            "latency_ms": None,
            "profiler_start_calls": 0,
            "profiler_start_status": None,
            "profiler_stop_calls": 0,
            "profiler_stop_status": None,
            "timing_methodology": "cuda_events",
        }
    ):
        raise CaptureDefectRefusal(
            code, "label result is not the audited pre-ROI false positive",
        )


def _validate_file_rows(rows: Any) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "inventory files are empty",
        )
    observed: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FILE_ROW_FIELDS:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"file row {index} fields",
            )
        rel = row.get("relative_path")
        if (
            not isinstance(rel, str) or not rel or Path(rel).is_absolute()
            or ".." in Path(rel).parts or rel in seen
            or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0
            or not _is_sha(row.get("sha256"))
        ):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"file row {index} invalid",
            )
        seen.add(rel)
        observed.append(dict(row))
    if observed != sorted(observed, key=lambda row: os.fsencode(row["relative_path"])):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "file rows are not byte-sorted",
        )
    return observed


def _validate_sealed_inventory(value: Any, *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != SEALED_INVENTORY_FIELDS:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label} fields",
        )
    files = _validate_file_rows(value.get("files"))
    if value != _sealed_inventory(files):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label} seal mismatch",
        )
    return {**value, "files": files}


def load_archive_inventory(path: Path) -> dict:
    value = _json_object(path, code="CAPTURE_DEFECT_LEDGER_INCONSISTENT")
    if (
        set(value) != INVENTORY_FIELDS or value.get("schema_version") != 1
        or value.get("inventory_format") != INVENTORY_FORMAT
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"inventory fields: {path}",
        )
    journal = _validate_sealed_inventory(value.get("journal"), label="journal")
    stage = _validate_sealed_inventory(
        value.get("stage_before_move"), label="stage_before_move",
    )
    contexts = value.get("chunk_context_files")
    if not isinstance(contexts, list) or len(contexts) != 3:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "chunk context cardinality",
        )
    seen: set[str] = set()
    normalized: list[dict] = []
    for index, row in enumerate(contexts):
        if not isinstance(row, dict) or set(row) != CONTEXT_ROW_FIELDS:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"context row {index} fields",
            )
        rel = row.get("archive_relative_path")
        if (
            not isinstance(rel, str) or Path(rel).is_absolute()
            or Path(rel).parts[:1] != ("chunk_context",) or ".." in Path(rel).parts
            or rel in seen or not isinstance(row.get("source_path"), str)
            or not Path(row["source_path"]).is_absolute()
            or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0
            or not _is_sha(row.get("sha256"))
        ):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"context row {index} invalid",
            )
        seen.add(rel)
        normalized.append(dict(row))
    if normalized != sorted(normalized, key=lambda row: os.fsencode(row["archive_relative_path"])):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "context rows are not byte-sorted",
        )
    return {
        **value, "journal": journal, "stage_before_move": stage,
        "chunk_context_files": normalized,
    }


def verify_capture_defect_archive(
    archive: Path, inventory: Mapping[str, Any], *,
    expected_empty_directories: Sequence[str] = (),
) -> None:
    if archive.is_symlink() or not archive.is_dir():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"archive absent: {archive}",
        )
    expected_children = {
        "journals", "chunk_context", CAPTURE_DEFECT_INVENTORY_NAME,
    }
    if {child.name for child in archive.iterdir()} != expected_children:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archive child set mismatch",
        )
    journals = archive / "journals"
    children = sorted(journals.iterdir()) if journals.is_dir() else []
    if len(children) != 1 or children[0].is_symlink() or not children[0].is_dir():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived journals cardinality",
        )
    actual_journal = content_inventory(
        children[0], expected_empty_directories=expected_empty_directories,
    )
    if actual_journal != inventory["journal"]["files"]:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived journal drift",
        )
    actual_context: dict[str, dict] = {}
    context_root = archive / "chunk_context"
    if context_root.is_symlink() or not context_root.is_dir():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "chunk_context invalid",
        )
    expected_context_names = {
        "chunk_run_journal.jsonl", "capture_matrix.json",
        "capture_preflight.json",
    }
    direct_context = list(context_root.iterdir())
    if (
        {path.name for path in direct_context} != expected_context_names
        or any(path.is_symlink() or not path.is_file() for path in direct_context)
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            "chunk_context must contain exactly three regular files",
        )
    for path in sorted(context_root.rglob("*")):
        if path.is_symlink():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"context symlink: {path}",
            )
        if path.is_file():
            rel = path.relative_to(archive).as_posix()
            actual_context[rel] = {
                "archive_relative_path": rel,
                "source_path": next(
                    (row["source_path"] for row in inventory["chunk_context_files"]
                     if row["archive_relative_path"] == rel),
                    "",
                ),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        elif not path.is_dir():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                f"context special filesystem node: {path}",
            )
    expected_context = {
        row["archive_relative_path"]: row
        for row in inventory["chunk_context_files"]
    }
    if actual_context != expected_context:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "chunk context drift",
        )
    run_rows = [
        {
            "relative_path": "chunk_run_journal.jsonl",
            "size_bytes": row["size_bytes"], "sha256": row["sha256"],
        }
        for row in inventory["chunk_context_files"]
        if Path(row["archive_relative_path"]).name == "chunk_run_journal.jsonl"
    ]
    reconstructed_stage = [
        *run_rows,
        *({
            "relative_path": f"journals/{children[0].name}/{row['relative_path']}",
            "size_bytes": row["size_bytes"], "sha256": row["sha256"],
        } for row in actual_journal),
    ]
    reconstructed_stage.sort(key=lambda row: os.fsencode(row["relative_path"]))
    if reconstructed_stage != inventory["stage_before_move"]["files"]:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "stage reconstruction drift",
        )


def _verify_archived_incident_bindings(
    row: Mapping[str, Any], *, archive: Path,
) -> None:
    journal = archive / "journals" / str(row["cell_id"])
    cell = journal / "cells" / str(row["cell_id"])
    bound = {
        journal / "journal_start.json": row["journal_start_sha256"],
        cell / "sol_stage" / "evidence" / "label-result.json": (
            row["label_result_sha256"]
        ),
        cell / "sol_stage" / "evidence" / "label-selection.json": (
            row["selection_sha256"]
        ),
        cell / "sol_stage" / "eval_driver.py": row["eval_driver_sha256"],
        cell / "sol_stage" / "capture_prepare_manifest.json": (
            row["prepare_manifest_sha256"]
        ),
        archive / "chunk_context" / "chunk_run_journal.jsonl": (
            row["chunk_run_journal_sha256"]
        ),
        archive / "chunk_context" / "capture_matrix.json": (
            row["original_matrix_file_sha256"]
        ),
        archive / "chunk_context" / "capture_preflight.json": (
            row["original_preflight_sha256"]
        ),
    }
    drift = [
        str(path) for path, expected in bound.items()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected
    ]
    if drift:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"archived incident bound-file drift: {drift}",
        )
    start = _json_object(
        journal / "journal_start.json",
        code="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    )
    if start.get("journal_start_seal_sha256") != canonical_sha256({
        key: value for key, value in start.items()
        if key != "journal_start_seal_sha256"
    }):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived journal seal drift",
        )
    capture_started = start.get("capture_started_at_utc")
    if capture_started is not None:
        _parse_utc(
            capture_started,
            label="archived journal_start.capture_started_at_utc",
        )
        if row.get("created_at_utc") != capture_started:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                "ledger timestamp does not match archived journal start",
            )
    expected_start = {
        "schema_version": 2,
        "state": "capture_started",
        "source_commit": row["original_head"],
        "capture_matrix_sha256": row["original_matrix_file_sha256"],
        "matrix_seal_sha256": row["original_matrix_seal_sha256"],
        "capture_preflight_sha256": row["original_preflight_sha256"],
        "admission_run_id": row["original_admission_run_id"],
    }
    if any(start.get(key) != value for key, value in expected_start.items()):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived journal identity drift",
        )
    result = _json_object(
        cell / "sol_stage" / "evidence" / "label-result.json",
        code="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    )
    _require_false_positive_before_roi(
        result, code="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    )
    matrix = _json_object(
        archive / "chunk_context" / "capture_matrix.json",
        code="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    )
    if matrix.get("matrix_seal_sha256") != row["original_matrix_seal_sha256"]:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived matrix seal drift",
        )
    run_path = archive / "chunk_context" / "chunk_run_journal.jsonl"
    try:
        run_rows = [json.loads(raw) for raw in run_path.read_text().splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"archived run journal: {exc}",
        ) from exc
    if len(run_rows) != 2:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived run journal cardinality",
        )
    session, admission = run_rows
    if (
        session.get("event") != "session_started"
        or session.get("source_commit") != row["original_head"]
        or session.get("session_id") != row["original_session_id"]
        or session.get("chunk_id") != row["chunk_id"]
        or session.get("capture_matrix_sha256")
        != row["original_matrix_file_sha256"]
        or session.get("matrix_seal_sha256")
        != row["original_matrix_seal_sha256"]
        or admission.get("event") != "capture_admission"
        or admission.get("session_id") != row["original_session_id"]
        or admission.get("chunk_id") != row["chunk_id"]
        or admission.get("run_id") != row["original_admission_run_id"]
        or admission.get("preflight_sha256") != row["original_preflight_sha256"]
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "archived run-journal binding drift",
        )


def validate_capture_defect_entry(
    row: Mapping[str, Any], *, ledger_path: Path, line_number: int,
    verify_archive: bool = False,
    expected_contract: IncidentContract = FIXED_CONTRACT,
) -> None:
    label = f"{ledger_path}:{line_number}"
    if set(row) != ENTRY_FIELDS:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: fields are not closed",
        )
    if (
        row.get("schema_version") != 1
        or row.get("entry_type") != "capture_defect_attempt_archive"
        or not _is_uuid4(row.get("entry_id"))
        or row.get("reason_code") != REASON_CODE
        or row.get("defect_summary") != DEFECT_SUMMARY
        or row.get("authoritative") is not False
        or row.get("reusable") is not False
        or row.get("campaign_rows") != 0
        or row.get("operator_flag") != OPERATOR_FLAG
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: constants invalid",
        )
    _parse_utc(row.get("created_at_utc"), label=f"{label}.created_at_utc")
    contract_fields = {
        "original_head": expected_contract.original_head,
        "chunk_id": expected_contract.chunk_id,
        "cell_id": expected_contract.cell_id,
        "original_matrix_file_sha256": expected_contract.matrix_file_sha256,
        "original_matrix_seal_sha256": expected_contract.matrix_seal_sha256,
        "original_preflight_sha256": expected_contract.preflight_sha256,
        "original_session_id": expected_contract.session_id,
        "original_admission_run_id": expected_contract.admission_run_id,
        "journal_inventory_sha256": expected_contract.journal_inventory_sha256,
        "journal_file_count": expected_contract.journal_file_count,
        "journal_total_bytes": expected_contract.journal_total_bytes,
        "stage_inventory_sha256": expected_contract.stage_inventory_sha256,
        "stage_file_count": expected_contract.stage_file_count,
        "stage_total_bytes": expected_contract.stage_total_bytes,
        "chunk_run_journal_sha256": expected_contract.chunk_run_journal_sha256,
        "journal_start_sha256": expected_contract.journal_start_sha256,
        "label_result_sha256": expected_contract.label_result_sha256,
        "eval_driver_sha256": expected_contract.eval_driver_sha256,
        "prepare_manifest_sha256": expected_contract.prepare_manifest_sha256,
        "selection_sha256": expected_contract.selection_sha256,
    }
    drift = [
        field for field, expected in contract_fields.items()
        if row.get(field) != expected
    ]
    if drift:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"{label}: incident contract drift: {drift}",
        )
    for field in ("original_head", "replacement_head"):
        if not _is_sha(row.get(field), 40):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}.{field}",
            )
    if (
        expected_contract == FIXED_CONTRACT
        and not _replacement_commit_has_repair(str(row["replacement_head"]))
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"{label}: replacement commit lacks the bound repair",
        )
    for field in (
        "original_matrix_file_sha256", "original_matrix_seal_sha256",
        "original_preflight_sha256", "original_admission_run_id",
        "inventory_sha256", "journal_inventory_sha256",
        "stage_inventory_sha256", "chunk_run_journal_sha256",
        "journal_start_sha256", "label_result_sha256", "eval_driver_sha256",
        "prepare_manifest_sha256", "selection_sha256",
    ):
        if not _is_sha(row.get(field)):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}.{field}",
            )
    for field in (
        "source_stage_path", "source_journals_path", "archive_path",
        "inventory_file",
    ):
        value = row.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}.{field}",
            )
    if (
        not isinstance(row.get("original_session_id"), str)
        or not isinstance(row.get("chunk_id"), str)
        or not isinstance(row.get("cell_id"), str)
        or any(
            type(row.get(field)) is not int or row[field] <= 0
            for field in (
                "journal_file_count", "journal_total_bytes",
                "stage_file_count", "stage_total_bytes",
            )
        )
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: scalar fields",
        )
    _validate_schema(row, code="CAPTURE_DEFECT_LEDGER_INCONSISTENT")
    archive = Path(row["archive_path"])
    inventory_path = Path(row["inventory_file"])
    if (
        archive.parent != ledger_path.parent.resolve()
        or inventory_path != archive / CAPTURE_DEFECT_INVENTORY_NAME
        or archive.name != f"capture-defect-{row['entry_id']}"
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: path closure",
        )
    if verify_archive:
        if (
            inventory_path.is_symlink() or not inventory_path.is_file()
            or sha256_file(inventory_path) != row["inventory_sha256"]
        ):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: inventory hash",
            )
        inventory = load_archive_inventory(inventory_path)
        expected = {
            "journal": (
                row["journal_file_count"], row["journal_total_bytes"],
                row["journal_inventory_sha256"],
            ),
            "stage_before_move": (
                row["stage_file_count"], row["stage_total_bytes"],
                row["stage_inventory_sha256"],
            ),
        }
        for name, expected_tuple in expected.items():
            value = inventory[name]
            if (
                value["file_count"], value["total_bytes"], value["domain_sha256"]
            ) != expected_tuple:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: {name} pins",
                )
        context_hashes = {
            Path(item["archive_relative_path"]).name: item["sha256"]
            for item in inventory["chunk_context_files"]
        }
        if context_hashes != {
            "chunk_run_journal.jsonl": row["chunk_run_journal_sha256"],
            "capture_matrix.json": row["original_matrix_file_sha256"],
            "capture_preflight.json": row["original_preflight_sha256"],
        }:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{label}: context pins",
            )
        verify_capture_defect_archive(
            archive, inventory,
            expected_empty_directories=_expected_empty_directories(
                expected_contract, stage_scope=False,
            ),
        )
        _verify_archived_incident_bindings(row, archive=archive)


def load_capture_defect_ledger(
    path: Path, *, verify_archives: bool = False,
    expected_contract: IncidentContract = FIXED_CONTRACT,
) -> list[dict]:
    """Strictly load the append-only capture-defect ledger and archive set."""
    if path.parent.is_symlink():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"symlink root: {path.parent}",
        )
    if not path.exists():
        if verify_archives and path.parent.exists():
            extras = sorted(str(child) for child in path.parent.iterdir())
            if extras:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    f"archive exists without ledger: {extras}",
                )
        return []
    if path.is_symlink() or not path.is_file():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", str(path),
        )
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{path}: {exc}",
        ) from exc
    if raw and not raw.endswith(b"\n"):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{path}: missing final LF",
        )
    rows: list[dict] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{path}:{line_number}: {exc}",
            ) from exc
        if not isinstance(row, dict):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"{path}:{line_number}: not object",
            )
        validate_capture_defect_entry(
            row, ledger_path=path.resolve(), line_number=line_number,
            verify_archive=verify_archives,
            expected_contract=expected_contract,
        )
        if row["entry_id"] in seen:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "duplicate entry_id",
            )
        seen.add(row["entry_id"])
        rows.append(row)
    if not rows:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"ledger is empty: {path}",
        )
    if len(rows) != 1:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"fixed incident ledger must contain exactly one row: {len(rows)}",
        )
    if verify_archives:
        expected = {Path(row["archive_path"]) for row in rows}
        actual: set[Path] = set()
        for child in path.parent.iterdir():
            if child.name == path.name:
                continue
            if child.is_symlink() or not child.is_dir():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    f"unexpected capture-defect root entry: {child}",
                )
            actual.add(child)
        if actual != expected:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                f"ledger/archive mismatch: expected={expected}; actual={actual}",
            )
    return rows


def capture_defect_backrefs(
    rows: Sequence[Mapping[str, Any]], *, cell_id: str, ledger_path: Path,
) -> list[dict]:
    selected = [row for row in rows if row.get("cell_id") == cell_id]
    return [{
        "ledger_entry_id": row["entry_id"],
        "ledger_path": str(ledger_path.resolve()),
        "archive_path": row["archive_path"],
        "original_head": row["original_head"],
        "reason_code": row["reason_code"],
    } for row in selected]


def unresolved_foreign_journals(
    *, output_root: Path, current_head: str,
    ledger_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return every foreign active journal not closed by an atomic archive."""
    staging = output_root.resolve() / ".staging"
    if not staging.exists():
        return []
    if staging.is_symlink() or not staging.is_dir():
        return [f"staging root is not a regular directory: {staging}"]
    unresolved: list[str] = []

    def children(path: Path, *, level: str) -> list[Path] | None:
        if path.is_symlink() or not path.is_dir():
            unresolved.append(f"invalid {level} directory {path}")
            return None
        try:
            return sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
        except OSError as exc:
            unresolved.append(f"unreadable {level} directory {path}: {exc}")
            return None

    head_dirs = children(staging, level="staging") or []
    for head_dir in head_dirs:
        chunk_dirs = children(head_dir, level="HEAD")
        if chunk_dirs is None:
            continue
        for chunk_dir in chunk_dirs:
            matrix_dirs = children(chunk_dir, level="chunk")
            if matrix_dirs is None:
                continue
            for matrix_dir in matrix_dirs:
                if matrix_dir.is_symlink() or not matrix_dir.is_dir():
                    # Matrix-stage directories also contain the regular
                    # chunk_run_journal.jsonl.  Other regular children are not
                    # journal ancestors and are safely ignored.
                    if matrix_dir.name != "chunk_run_journal.jsonl":
                        unresolved.append(
                            f"invalid matrix directory {matrix_dir}"
                        )
                    continue
                journals = matrix_dir / "journals"
                if not journals.exists() and not journals.is_symlink():
                    continue
                active_children = children(journals, level="journals")
                if active_children is None:
                    continue
                invalid_children = [
                    child for child in active_children
                    if child.is_symlink() or not child.is_dir()
                ]
                if invalid_children:
                    unresolved.extend(
                        f"invalid journal child {child}" for child in invalid_children
                    )
                    continue
                if head_dir.name == current_head or not active_children:
                    continue
                matching = [
                    row for row in ledger_rows
                    if Path(str(row.get("source_journals_path", "")))
                    == journals.resolve()
                ]
                if matching:
                    unresolved.append(
                        "ledgered source path reappeared after atomic archive: "
                        f"{journals}"
                    )
                else:
                    unresolved.append(f"unledgered foreign journals: {journals}")
    return unresolved


def _acquire_campaign_lock(output_root: Path):
    path = output_root / ".capture.lock"
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"campaign lock held: {path}",
        ) from exc
    return stream


def _stable_entry_id(request: ArchiveRequest) -> str:
    if request.entry_id is not None:
        if not _is_uuid4(request.entry_id):
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH", "entry_id is not UUID4",
            )
        return request.entry_id
    material = "\0".join((
        request.contract.original_head,
        request.contract.matrix_file_sha256,
        request.contract.cell_id,
    )).encode()
    return str(uuid.UUID(bytes=hashlib.sha256(material).digest()[:16], version=4))


def _fault_point(_name: str) -> None:
    """Test-only crash seam; production execution is a no-op."""


def _owner_marker_payload(request: ArchiveRequest, *, entry_id: str) -> dict:
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "original_head": request.contract.original_head,
        "replacement_head": request.replacement_head,
        "chunk_id": request.contract.chunk_id,
        "cell_id": request.contract.cell_id,
        "matrix_file_sha256": request.contract.matrix_file_sha256,
    }


def _require_owner_marker(
    request: ArchiveRequest, *, entry_id: str, marker: Path,
) -> None:
    if marker.is_symlink() or not marker.is_file():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"pre-rename owner marker is invalid: {marker}",
        )
    if _json_object(
        marker, code="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    ) != _owner_marker_payload(request, entry_id=entry_id):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"pre-rename owner marker drift: {marker}",
        )


def _remove_authenticated_premove_scratch(
    request: ArchiveRequest, *, stage: Path, defect_root: Path, archive: Path,
    ledger_path: Path, pending_path: Path, marker: Path, entry_id: str,
) -> None:
    """Remove only marker-authenticated scratch while source journals exist.

    This is the sole recovery path for a process death before the
    evidence-bearing rename.  The marker is fsynced before any copy, so an
    empty context, a completed prefix, or a partially written current file is
    still unambiguously invocation-owned.  No other namespace is accepted.
    """
    if defect_root.is_symlink() or not defect_root.is_dir():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"invalid pre-rename scratch root: {defect_root}",
        )
    _require_owner_marker(request, entry_id=entry_id, marker=marker)
    root_children = list(defect_root.iterdir())
    expected_root_names = {marker.name}
    archive_exists = archive.exists() or archive.is_symlink()
    if archive_exists:
        expected_root_names.add(archive.name)
    if (
        {child.name for child in root_children} != expected_root_names
        or ledger_path.exists() or ledger_path.is_symlink()
        or pending_path.exists() or pending_path.is_symlink()
    ):
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
            f"pre-rename scratch root is not exclusive: {root_children}",
        )
    if archive_exists:
        if archive.is_symlink() or not archive.is_dir():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                f"pre-rename archive is invalid: {archive}",
            )
        archive_children = list(archive.iterdir())
        child_names = {child.name for child in archive_children}
        if not child_names <= {"chunk_context"}:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                f"pre-rename scratch shape mismatch: {archive_children}",
            )
        if "chunk_context" in child_names:
            context_root = archive / "chunk_context"
            if context_root.is_symlink() or not context_root.is_dir():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    f"pre-rename context is invalid: {context_root}",
                )
            ordered_sources = [
                ("chunk_run_journal.jsonl", stage / "chunk_run_journal.jsonl"),
                ("capture_matrix.json", request.matrix_path),
                ("capture_preflight.json", request.preflight_path),
            ]
            context_children = list(context_root.iterdir())
            present = {child.name for child in context_children}
            expected_prefix = {
                name for name, _source in ordered_sources[:len(present)]
            }
            if present != expected_prefix:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    f"pre-rename context set mismatch: {context_children}",
                )
            for index, (name, source) in enumerate(ordered_sources[:len(present)]):
                copied = context_root / name
                if (
                    copied.is_symlink() or not copied.is_file()
                    or source.is_symlink() or not source.is_file()
                ):
                    raise CaptureDefectRefusal(
                        "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                        f"pre-rename context node is invalid: {copied}",
                    )
                copied_raw = copied.read_bytes()
                source_raw = source.read_bytes()
                if copied_raw != source_raw[:len(copied_raw)] or (
                    index < len(present) - 1 and copied_raw != source_raw
                ):
                    raise CaptureDefectRefusal(
                        "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                        f"pre-rename context byte mismatch: {copied}",
                    )
        shutil.rmtree(archive)
    marker.unlink()
    _fsync_directory(defect_root)


def _created_at_for_archive(request: ArchiveRequest, archive: Path) -> str:
    start = _json_object(
        archive / "journals" / request.contract.cell_id / "journal_start.json",
        code="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    )
    value = start.get("capture_started_at_utc")
    if value is not None:
        _parse_utc(value, label="journal_start.capture_started_at_utc")
        return str(value)
    if request.created_at_utc is not None:
        return _utc_text(request.created_at_utc)
    raise CaptureDefectRefusal(
        "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
        "journal_start.capture_started_at_utc is absent",
    )


def _context_rows_from_archive(
    request: ArchiveRequest, *, stage: Path, archive: Path,
) -> list[dict]:
    sources = {
        "chunk_run_journal.jsonl": stage / "chunk_run_journal.jsonl",
        "capture_matrix.json": request.matrix_path,
        "capture_preflight.json": request.preflight_path,
    }
    rows: list[dict] = []
    for name, source in sources.items():
        path = archive / "chunk_context" / name
        if path.is_symlink() or not path.is_file():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", f"context absent: {path}",
            )
        rows.append({
            "archive_relative_path": f"chunk_context/{name}",
            "source_path": str(source.resolve(strict=False)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return sorted(rows, key=lambda row: os.fsencode(row["archive_relative_path"]))


def _materialize_or_verify_inventory(
    request: ArchiveRequest, *, stage: Path, archive: Path,
) -> tuple[dict, Path]:
    contract = request.contract
    journal = archive / "journals" / contract.cell_id
    journal_rows = content_inventory(
        journal,
        expected_empty_directories=_expected_empty_directories(
            contract, stage_scope=False,
        ),
    )
    contexts = _context_rows_from_archive(request, stage=stage, archive=archive)
    run = next(
        row for row in contexts
        if Path(row["archive_relative_path"]).name == "chunk_run_journal.jsonl"
    )
    stage_rows = [{
        "relative_path": "chunk_run_journal.jsonl",
        "size_bytes": run["size_bytes"], "sha256": run["sha256"],
    }, *({
        "relative_path": f"journals/{contract.cell_id}/{row['relative_path']}",
        "size_bytes": row["size_bytes"], "sha256": row["sha256"],
    } for row in journal_rows)]
    stage_rows.sort(key=lambda row: os.fsencode(row["relative_path"]))
    inventory = {
        "schema_version": 1,
        "inventory_format": INVENTORY_FORMAT,
        "journal": _sealed_inventory(journal_rows),
        "stage_before_move": _sealed_inventory(stage_rows),
        "chunk_context_files": contexts,
    }
    observed = (
        len(journal_rows), sum(row["size_bytes"] for row in journal_rows),
        inventory_sha256(journal_rows), len(stage_rows),
        sum(row["size_bytes"] for row in stage_rows), inventory_sha256(stage_rows),
    )
    expected = (
        contract.journal_file_count, contract.journal_total_bytes,
        contract.journal_inventory_sha256, contract.stage_file_count,
        contract.stage_total_bytes, contract.stage_inventory_sha256,
    )
    if observed != expected:
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_INVENTORY_MISMATCH",
            f"post-move observed={observed!r}; expected={expected!r}",
        )
    inventory_path = archive / CAPTURE_DEFECT_INVENTORY_NAME
    if inventory_path.exists() or inventory_path.is_symlink():
        if inventory_path.is_symlink() or not inventory_path.is_file():
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", str(inventory_path),
            )
        if load_archive_inventory(inventory_path) != inventory:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "existing inventory drift",
            )
    else:
        _write_json_exclusive_fsync(inventory_path, inventory)
        _fault_point("after_inventory_fsync")
    verify_capture_defect_archive(
        archive, inventory,
        expected_empty_directories=_expected_empty_directories(
            contract, stage_scope=False,
        ),
    )
    return inventory, inventory_path


def _entry_payload(
    request: ArchiveRequest, *, entry_id: str, created: str, stage: Path,
    archive: Path, inventory_path: Path,
) -> dict:
    contract = request.contract
    return {
        "schema_version": 1,
        "entry_id": entry_id,
        "entry_type": "capture_defect_attempt_archive",
        "created_at_utc": created,
        "original_head": contract.original_head,
        "replacement_head": request.replacement_head,
        "chunk_id": contract.chunk_id,
        "cell_id": contract.cell_id,
        "original_matrix_file_sha256": contract.matrix_file_sha256,
        "original_matrix_seal_sha256": contract.matrix_seal_sha256,
        "original_preflight_sha256": contract.preflight_sha256,
        "original_session_id": contract.session_id,
        "original_admission_run_id": contract.admission_run_id,
        "source_stage_path": str(stage),
        "source_journals_path": str(stage / "journals"),
        "archive_path": str(archive.resolve()),
        "inventory_file": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "journal_inventory_sha256": contract.journal_inventory_sha256,
        "journal_file_count": contract.journal_file_count,
        "journal_total_bytes": contract.journal_total_bytes,
        "stage_inventory_sha256": contract.stage_inventory_sha256,
        "stage_file_count": contract.stage_file_count,
        "stage_total_bytes": contract.stage_total_bytes,
        "chunk_run_journal_sha256": contract.chunk_run_journal_sha256,
        "journal_start_sha256": contract.journal_start_sha256,
        "label_result_sha256": contract.label_result_sha256,
        "eval_driver_sha256": contract.eval_driver_sha256,
        "prepare_manifest_sha256": contract.prepare_manifest_sha256,
        "selection_sha256": contract.selection_sha256,
        "reason_code": REASON_CODE,
        "defect_summary": DEFECT_SUMMARY,
        "authoritative": False,
        "reusable": False,
        "campaign_rows": 0,
        "operator_flag": OPERATOR_FLAG,
    }


def _ledger_bytes(row: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode()


def _publish_ledger_atomically(
    *, ledger_path: Path, pending_path: Path, row: Mapping[str, Any],
) -> None:
    raw = _ledger_bytes(row)
    if ledger_path.exists():
        if ledger_path.is_symlink() or not ledger_path.is_file() \
                or ledger_path.read_bytes() != raw:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "published ledger drift",
            )
        if pending_path.exists():
            if pending_path.is_symlink() or not pending_path.is_file() \
                    or pending_path.read_bytes() != raw:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "orphan pending drift",
                )
            pending_path.unlink()
            _fsync_directory(pending_path.parent)
        return
    if pending_path.exists():
        if pending_path.is_symlink() or not pending_path.is_file() \
                or pending_path.read_bytes() != raw:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "pending ledger drift",
            )
    else:
        fd = os.open(pending_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        _fsync_directory(pending_path.parent)
    _fault_point("after_ledger_temp_fsync")
    os.rename(pending_path, ledger_path)
    _fsync_directory(ledger_path.parent)
    _fault_point("after_ledger_publish")


def archive_capture_defect_attempt(request: ArchiveRequest) -> dict:
    """Verify, atomically move, and crash-resumably ledger the fixed H3 attempt."""
    output_root = request.output_root.resolve()
    if output_root.is_symlink() or not output_root.is_dir():
        raise CaptureDefectRefusal(
            "CAPTURE_DEFECT_SOURCE_MISMATCH", f"output root: {output_root}",
        )
    lock = _acquire_campaign_lock(output_root)
    archive: Path | None = None
    journals: Path | None = None
    owner_marker: Path | None = None
    journals_moved = False
    archive_owned_by_invocation = False
    marker_owned_by_invocation = False
    try:
        contract = request.contract
        _require_replacement_contract(request)
        _require_fixed_source_paths(request)
        stage, journals, _run_journal = _source_paths(request)
        _require_directory_chain(output_root, stage, label="stage")
        entry_id = _stable_entry_id(request)
        archive_name = f"capture-defect-{entry_id}"
        defect_root = output_root / CAPTURE_DEFECT_ROOT_NAME
        archive = defect_root / archive_name
        ledger_path = defect_root / CAPTURE_DEFECT_LEDGER_NAME
        pending_path = defect_root / CAPTURE_DEFECT_LEDGER_PENDING_NAME
        owner_marker = defect_root / f"{archive_name}{CAPTURE_DEFECT_OWNER_SUFFIX}"

        source_available = journals.exists() or journals.is_symlink()
        recovery_available = archive.exists() or archive.is_symlink()
        marker_available = owner_marker.exists() or owner_marker.is_symlink()

        # Verify the still-live incident before creating any archive output.
        # A post-rename retry instead proves the immutable destination below.
        if source_available:
            if recovery_available or marker_available:
                _remove_authenticated_premove_scratch(
                    request, stage=stage, defect_root=defect_root,
                    archive=archive, ledger_path=ledger_path,
                    pending_path=pending_path, marker=owner_marker,
                    entry_id=entry_id,
                )
                recovery_available = False
                marker_available = False
            _verify_incident(request)
        elif not recovery_available:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_SOURCE_MISMATCH",
                f"journals unavailable and no resumable archive: {journals}",
            )

        if defect_root.exists() or defect_root.is_symlink():
            if defect_root.is_symlink() or not defect_root.is_dir():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT", str(defect_root),
                )
        else:
            defect_root.mkdir(mode=0o700)
            _fsync_directory(output_root)

        if source_available:
            if ledger_path.exists() or ledger_path.is_symlink() \
                    or pending_path.exists() or pending_path.is_symlink():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    "ledger state exists before the evidence-bearing rename",
                )
            extras = list(defect_root.iterdir())
            if extras:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    f"unexpected pre-archive entries: {extras}",
                )
            _write_json_exclusive_fsync(
                owner_marker,
                _owner_marker_payload(request, entry_id=entry_id),
            )
            marker_owned_by_invocation = True
            archive.mkdir(mode=0o700)
            archive_owned_by_invocation = True
            _fsync_directory(defect_root)
            context_root = archive / "chunk_context"
            context_root.mkdir(mode=0o700)
            _fsync_directory(archive)
            for source, name in (
                (stage / "chunk_run_journal.jsonl", "chunk_run_journal.jsonl"),
                (request.matrix_path, "capture_matrix.json"),
                (request.preflight_path, "capture_preflight.json"),
            ):
                _copy_exclusive_fsync(source, context_root / name)
            destination = archive / "journals"
            if destination.exists() or destination.is_symlink():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_ARCHIVE_EXISTS", str(destination),
                )
            if journals.stat().st_dev != defect_root.stat().st_dev:
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_CROSS_DEVICE", f"{journals} -> {defect_root}",
                )
            _fault_point("before_journals_rename")
            os.rename(journals, destination)
            _fault_point("after_journals_rename_before_flag")
            journals_moved = True
            _fsync_directory(stage)
            _fsync_directory(archive)
            owner_marker.unlink()
            marker_owned_by_invocation = False
            _fsync_directory(defect_root)
            _fault_point("after_journals_rename")
        else:
            if archive.is_symlink() or not archive.is_dir():
                raise CaptureDefectRefusal(
                    "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                    f"resumable archive is invalid: {archive}",
                )
            if marker_available:
                _require_owner_marker(
                    request, entry_id=entry_id, marker=owner_marker,
                )
                if (
                    ledger_path.exists() or ledger_path.is_symlink()
                    or pending_path.exists() or pending_path.is_symlink()
                    or not (archive / "journals").is_dir()
                    or (archive / "journals").is_symlink()
                ):
                    raise CaptureDefectRefusal(
                        "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                        "post-rename owner marker has inconsistent state",
                    )
                expected_children = {archive.name, owner_marker.name}
                if {child.name for child in defect_root.iterdir()} \
                        != expected_children:
                    raise CaptureDefectRefusal(
                        "CAPTURE_DEFECT_LEDGER_INCONSISTENT",
                        "post-rename owner marker root is not exclusive",
                    )
                owner_marker.unlink()
                _fsync_directory(defect_root)
            # A published ledger plus a reappeared source is rejected above;
            # recovery never recreates or modifies the moved journal tree.
            journals_moved = True

        inventory, inventory_path = _materialize_or_verify_inventory(
            request, stage=stage, archive=archive,
        )
        del inventory
        created = _created_at_for_archive(request, archive)
        row = _entry_payload(
            request, entry_id=entry_id, created=created, stage=stage,
            archive=archive, inventory_path=inventory_path,
        )
        validate_capture_defect_entry(
            row, ledger_path=ledger_path.resolve(), line_number=1,
            verify_archive=True, expected_contract=contract,
        )
        _publish_ledger_atomically(
            ledger_path=ledger_path, pending_path=pending_path, row=row,
        )
        observed = load_capture_defect_ledger(
            ledger_path, verify_archives=True, expected_contract=contract,
        )
        if observed != [row]:
            raise CaptureDefectRefusal(
                "CAPTURE_DEFECT_LEDGER_INCONSISTENT", "published row drift",
            )
        return row
    except BaseException:
        # Before the evidence-bearing rename, this invocation owns every byte
        # beneath its UUID-named directory and may remove that scratch output
        # to keep a retry possible.  Once journals moved, never roll back or
        # delete anything: an interrupted post-move state is forensic evidence.
        destination_has_evidence = (
            archive is not None
            and ((archive / "journals").exists()
                 or (archive / "journals").is_symlink())
        )
        source_was_removed = journals is not None and not (
            journals.exists() or journals.is_symlink()
        )
        if (
            archive is not None and archive_owned_by_invocation
            and not journals_moved
            and not destination_has_evidence and not source_was_removed
            and archive.exists()
        ):
            shutil.rmtree(archive)
            _fsync_directory(archive.parent)
        if (
            marker_owned_by_invocation and not journals_moved
            and owner_marker is not None
            and owner_marker.exists() and not owner_marker.is_symlink()
        ):
            owner_marker.unlink()
            _fsync_directory(owner_marker.parent)
        raise
    finally:
        lock.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(OPERATOR_FLAG, action="store_true", required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/home/jakeshea/sm86_reference_capture_r27"),
    )
    parser.add_argument("--replacement-head")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    replacement_head = args.replacement_head
    if replacement_head is None:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False,
        )
        replacement_head = proc.stdout.strip() if proc.returncode == 0 else ""
    inputs = output_root / "inputs" / "r28_9bfc1fc2"
    request = ArchiveRequest(
        output_root=output_root,
        matrix_path=inputs / "sm86_reference_capture_matrix.json",
        preflight_path=inputs / "preflight_sol_bootstrap.json",
        replacement_head=replacement_head,
    )
    try:
        row = archive_capture_defect_attempt(request)
    except CaptureDefectRefusal as exc:
        print(f"{exc.code}: {exc.detail}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "archived_non_authoritative_reuse_forbidden",
        "entry_id": row["entry_id"],
        "archive_path": row["archive_path"],
        "campaign_rows": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArchiveRequest", "BACKREF_FIELDS", "CAPTURE_DEFECT_LEDGER_NAME",
    "CAPTURE_DEFECT_ROOT_NAME", "CaptureDefectRefusal", "DEFECT_SUMMARY",
    "FIXED_CONTRACT", "IncidentContract", "REASON_CODE",
    "archive_capture_defect_attempt", "capture_defect_backrefs",
    "content_inventory", "inventory_sha256", "load_capture_defect_ledger",
    "unresolved_foreign_journals", "verify_capture_defect_archive",
]
