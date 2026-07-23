#!/usr/bin/env python3
"""Resume and reconcile the complete additive SOL workload-row campaign.

This controller never changes the canonical 344-cell campaign.  It owns the
dedicated supplemental root, walks the sealed breadth-first chunk sequence,
creates one immutable live preflight per chunk attempt, and publishes immutable
whole-matrix progress snapshots with claim-safe cohort exports.

The process holds ``/run/user/$UID/solbench-sm86-gpu.lock`` for its complete
lifetime.  A launcher may pass an already-locked descriptor through
``SOLBENCH_SM86_GPU_LOCK_FD``; the descriptor is accepted only when it resolves
to that exact canonical path and still carries the nonblocking exclusive lock.
Use ``--max-new-chunks N`` for an internal, deterministic pilot boundary;
external timeout/kill is not a valid pilot control.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import uuid
import stat
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LABELS = REPO_ROOT / "tools/local_labels"
for _path in (REPO_ROOT, LOCAL_LABELS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import sol_workload_capture_chunk as chunk_capture  # noqa: E402
import sol_workload_coverage_inventory as coverage_inventory_builder  # noqa: E402


EVENT_DIR = "sol_workload_campaign/events"
SNAPSHOT_DIR = "sol_workload_campaign/snapshots"
PREFLIGHT_DIR = "sol_workload_campaign/preflights"
GLOBAL_GPU_LOCK_PATH = Path(
    f"/run/user/{os.getuid()}/solbench-sm86-gpu.lock"
)
INHERITED_GPU_LOCK_FD_ENV = "SOLBENCH_SM86_GPU_LOCK_FD"
STORAGE_FAULT_ROOT = GLOBAL_GPU_LOCK_PATH.parent


class CampaignRefusal(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _global_gpu_lock():
    """Hold the canonical host-wide GPU campaign lock for this process."""
    inherited = os.environ.get(INHERITED_GPU_LOCK_FD_ENV)
    if inherited is not None:
        try:
            inherited_fd = int(inherited)
            if inherited_fd < 0:
                raise ValueError(inherited)
            observed_path = Path(
                os.readlink(f"/proc/self/fd/{inherited_fd}")
            ).resolve()
            if observed_path != GLOBAL_GPU_LOCK_PATH.resolve():
                raise ValueError(str(observed_path))
            descriptor = os.dup(inherited_fd)
        except (OSError, ValueError) as exc:
            raise CampaignRefusal(
                "campaign_global_gpu_lock_invalid", str(inherited)
            ) from exc
    else:
        try:
            descriptor = os.open(
                GLOBAL_GPU_LOCK_PATH,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise CampaignRefusal(
                "campaign_global_gpu_lock_invalid", str(GLOBAL_GPU_LOCK_PATH)
            ) from exc
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise CampaignRefusal(
            "campaign_global_gpu_lock_invalid", str(GLOBAL_GPU_LOCK_PATH)
        )
    stream = os.fdopen(descriptor, "r+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise CampaignRefusal(
            "campaign_global_gpu_lock_active", str(GLOBAL_GPU_LOCK_PATH)
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def storage_fault_receipt_path(output_root: Path) -> Path:
    identity = hashlib.sha256(
        os.fsencode(output_root.absolute())
    ).hexdigest()
    return STORAGE_FAULT_ROOT / f"solbench-sm86-storage-fault-{identity}.json"


# Backward-compatible internal spelling used by the focused safety tests.
_storage_fault_receipt_path = storage_fault_receipt_path


def _record_storage_fault(
    *, output_root: Path, refusal: chunk_capture.WorkloadCaptureRefusal,
    storage_state: dict[str, int], operation: str,
) -> Path:
    """Record a storage stop on the separate runtime filesystem."""
    receipt = _storage_fault_receipt_path(output_root)
    document = {
        "schema_version": 1,
        "kind": "solbench_sm86_storage_fault",
        "campaign_root_sha256": hashlib.sha256(
            os.fsencode(output_root.absolute())
        ).hexdigest(),
        "failure_code": refusal.code,
        "failure_detail_sha256": hashlib.sha256(
            refusal.detail.encode("utf-8", errors="surrogateescape")
        ).hexdigest(),
        "operation": operation,
        "storage_state": storage_state,
    }
    payload = _canonical_pretty_json(document)
    if receipt.exists():
        if receipt.is_symlink() or not receipt.is_file() or receipt.read_bytes() != payload:
            raise CampaignRefusal(
                "campaign_storage_fault_receipt_conflict", str(receipt)
            )
        return receipt
    descriptor = os.open(
        receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short external storage-fault receipt write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return receipt


def _check_storage_fault_receipt(output_root: Path) -> None:
    receipt = _storage_fault_receipt_path(output_root)
    if not receipt.exists():
        return
    if receipt.is_symlink() or not receipt.is_file():
        raise CampaignRefusal(
            "campaign_storage_fault_receipt_invalid", str(receipt)
        )
    try:
        payload = receipt.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignRefusal(
            "campaign_storage_fault_receipt_invalid", str(receipt)
        ) from exc
    expected_root = hashlib.sha256(
        os.fsencode(output_root.absolute())
    ).hexdigest()
    if (
        not isinstance(document, dict)
        or set(document) != {
            "schema_version", "kind", "campaign_root_sha256",
            "failure_code", "failure_detail_sha256", "operation",
            "storage_state",
        }
        or document.get("schema_version") != 1
        or document.get("kind") != "solbench_sm86_storage_fault"
        or document.get("campaign_root_sha256") != expected_root
        or document.get("failure_code") not in {
            chunk_capture.STORAGE_CAPACITY_CODE,
            chunk_capture.STORAGE_INTEGRITY_CODE,
        }
        or chunk_capture.SHA256_RE.fullmatch(
            str(document.get("failure_detail_sha256", ""))
        ) is None
        or document.get("operation") != "campaign_mutation_or_reconciliation"
        or not isinstance(document.get("storage_state"), dict)
        or set(document["storage_state"]) != {
            "available_bytes", "available_inodes",
        }
        or any(
            type(value) is not int or value < 0
            for value in document["storage_state"].values()
        )
        or payload != _canonical_pretty_json(document)
    ):
        raise CampaignRefusal(
            "campaign_storage_fault_receipt_invalid", str(receipt)
        )
    prior = document.get("storage_state")
    state_path = output_root if output_root.exists() else output_root.parent
    current = _storage_state(state_path)
    if document.get("failure_code") == chunk_capture.STORAGE_INTEGRITY_CODE:
        raise CampaignRefusal(
            "campaign_storage_integrity_reconciliation_required", str(receipt)
        )
    if (
        current.get("available_bytes", 0) <= prior.get("available_bytes", 0)
        and current.get("available_inodes", 0)
        <= prior.get("available_inodes", 0)
    ):
        raise CampaignRefusal(
            "campaign_storage_state_unchanged", str(receipt)
        )
    receipt.unlink()
    directory = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _row_identity(key: dict[str, Any]) -> tuple[str, str, str]:
    return chunk_capture._row_identity(key)


def validate_whole_matrix_closure(matrix: dict[str, Any]) -> dict[str, Any]:
    """Prove exact variant/row/chunk closure before orchestration."""
    rows = matrix.get("rows")
    variants = matrix.get("variants")
    chunks = matrix.get("chunks")
    counts = matrix.get("counts")
    if not all(isinstance(value, list) for value in (rows, variants, chunks)):
        raise CampaignRefusal("campaign_matrix_closure_invalid", "lists absent")
    if (
        not isinstance(counts, dict)
        or counts.get("rows") != len(rows)
        or counts.get("variants") != len(variants)
        or counts.get("chunks") != len(chunks)
    ):
        raise CampaignRefusal(
            "campaign_matrix_count_mismatch", repr(counts)
        )
    variants_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in variants:
        key = (variant.get("slug"), variant.get("variant_id"))
        if key in variants_by_key or not all(
            isinstance(value, str) and value for value in key
        ):
            raise CampaignRefusal(
                "campaign_variant_closure_invalid", repr(key)
            )
        variants_by_key[key] = variant
    problem_count = len({key[0] for key in variants_by_key})
    variant_cohorts = Counter(
        variant.get("cohort") for variant in variants
    )
    if (
        counts.get("problems") != problem_count
        or counts.get("native_variants") != variant_cohorts["native"]
        or counts.get("partial_native_variants")
        != variant_cohorts["partial_native_row"]
        or counts.get("reduced_variants") != variant_cohorts["reduced_proxy"]
    ):
        raise CampaignRefusal(
            "campaign_matrix_count_mismatch", repr(counts)
        )
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _row_identity(row.get("key"))
        variant = variants_by_key.get((key[0], key[2]))
        if (
            key in rows_by_key or variant is None
            or row.get("variant_contract_sha256")
            != variant.get("variant_contract_sha256")
            or row.get("cohort") != variant.get("cohort")
            or row.get("category") != variant.get("category")
            or row.get("claim_scope") != variant.get("claim_scope")
        ):
            raise CampaignRefusal(
                "campaign_row_variant_closure_invalid", repr(key)
            )
        rows_by_key[key] = row
    seen_chunks: set[str] = set()
    scheduled: list[tuple[str, str, str]] = []
    chunks_by_row: dict[tuple[str, str, str], str] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        members = chunk.get("rows")
        if (
            not isinstance(chunk_id, str) or not chunk_id
            or chunk_id in seen_chunks or not isinstance(members, list)
            or chunk.get("row_count") != len(members)
            or chunk.get("membership_sha256")
            != chunk_capture.canonical_sha256(members)
        ):
            raise CampaignRefusal(
                "campaign_chunk_closure_invalid", repr(chunk_id)
            )
        seen_chunks.add(chunk_id)
        for member in members:
            key = _row_identity(member)
            row = rows_by_key.get(key)
            if (
                row is None or key in chunks_by_row
                or row.get("cohort") != chunk.get("cohort")
            ):
                raise CampaignRefusal(
                    "campaign_chunk_membership_invalid", repr(key)
                )
            chunks_by_row[key] = chunk_id
            scheduled.append(key)
    if set(scheduled) != set(rows_by_key) or len(scheduled) != len(rows_by_key):
        raise CampaignRefusal(
            "campaign_chunk_partition_invalid",
            f"scheduled={len(scheduled)} rows={len(rows_by_key)}",
        )
    row_cohorts = Counter(row["cohort"] for row in rows)
    chunk_cohorts = Counter(chunk["cohort"] for chunk in chunks)
    if (
        counts.get("native_rows") != row_cohorts["native"]
        or counts.get("partial_native_rows")
        != row_cohorts["partial_native_row"]
        or counts.get("reduced_proxy_rows") != row_cohorts["reduced_proxy"]
        or sum(row_cohorts.values()) != len(rows)
        or sum(chunk_cohorts.values()) != len(chunks)
    ):
        raise CampaignRefusal(
            "campaign_matrix_count_mismatch", repr(counts)
        )
    return {
        "rows_by_key": rows_by_key,
        "variants_by_key": variants_by_key,
        "chunks_by_row": chunks_by_row,
        "scheduled_keys": scheduled,
        "problem_count": problem_count,
        "variant_cohorts": dict(variant_cohorts),
        "row_cohorts": dict(row_cohorts),
        "chunk_cohorts": dict(chunk_cohorts),
    }


def _regenerate_full_coverage_inventory(
    *, repo_root: Path,
) -> dict[str, Any]:
    """Regenerate the complete inventory from the exact campaign inputs.

    Count checks alone are not evidence: a fabricated inventory can preserve
    235/3,957 while changing individual problem dispositions.  The builder is
    therefore run again over the certified, clean capture checkout and the
    tracked historical v1 input bundle.  Additive live-v2 runnability/shrink
    inputs drive the supplemental matrix but cannot rewrite this baseline
    accounting universe.  Callers compare canonical bytes, not just counts.
    """
    try:
        return coverage_inventory_builder.build_inventory(
            repo_root, coverage_inventory_builder.DEFAULT_INPUTS
        )
    except coverage_inventory_builder.CoverageInventoryError as exc:
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_regeneration_failed", str(exc)
        ) from exc


def validate_full_coverage_inventory(
    document: dict[str, Any], *, regenerated: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the separate 235-problem/3,957-row accounting universe."""
    problems = document.get("sol_problem_dispositions") or {}
    rows = document.get("workload_rows") or {}
    buckets = document.get("exclusion_buckets") or {}
    if (
        document.get("schema_version") != 1
        or document.get("artifact_id")
        != "sm86_sol_workload_coverage_inventory_v1"
        or problems.get("total_problem_count") != 235
        or len(problems.get("problems") or []) != 235
        or rows.get("total_workload_row_count") != 3957
        or sum((rows.get("by_disposition") or {}).values()) != 3957
        or (buckets.get("architecture") or {}).get("problem_count") != 34
        or (buckets.get("architecture") or {}).get("workload_row_count") != 537
    ):
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_invalid", repr({
                "problems": problems.get("total_problem_count"),
                "rows": rows.get("total_workload_row_count"),
            }),
        )
    payload = coverage_inventory_builder.canonical_json_bytes(document)
    if regenerated is not None:
        regenerated_payload = coverage_inventory_builder.canonical_json_bytes(
            regenerated
        )
        if payload != regenerated_payload:
            raise CampaignRefusal(
                "campaign_full_coverage_inventory_byte_mismatch",
                repr({
                    "provided_sha256": hashlib.sha256(payload).hexdigest(),
                    "regenerated_sha256": hashlib.sha256(
                        regenerated_payload
                    ).hexdigest(),
                }),
            )
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "problem_count": 235,
        "workload_row_count": 3957,
        "architecture_excluded_problems": 34,
        "architecture_excluded_rows": 537,
    }


def _event_paths(output_root: Path) -> list[Path]:
    root = output_root / EVENT_DIR
    root_metadata = chunk_capture._output_lstat(
        root, operation="campaign_event_root_lstat",
    )
    if root_metadata is None:
        return []
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise CampaignRefusal("campaign_event_sequence_invalid", str(root))
    paths = [
        path for path in chunk_capture._output_directory_entries(
            root, operation="campaign_event_directory",
        )
        if path.match("event-*.json")
    ]
    if [path.name for path in paths] != [
        f"event-{ordinal:06d}.json" for ordinal in range(1, len(paths) + 1)
    ]:
        raise CampaignRefusal("campaign_event_sequence_invalid", str(root))
    return paths


def load_event_chain(output_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    prior_seal: str | None = None
    for ordinal, path in enumerate(_event_paths(output_root), start=1):
        event = chunk_capture._load_output_object(
            path, role="campaign event", operation="campaign_event_read",
        )
        chunk_capture._verify_sealed(
            event, "event_seal_sha256", code="campaign_event_seal_invalid"
        )
        if (
            event.get("schema_version") != 1
            or event.get("kind") != "sol_workload_campaign_event"
            or event.get("event_ordinal") != ordinal
            or event.get("prior_event_seal_sha256") != prior_seal
        ):
            raise CampaignRefusal("campaign_event_chain_invalid", str(path))
        events.append(event)
        prior_seal = event["event_seal_sha256"]
    return events


def append_event(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    events = load_event_chain(output_root)
    event = chunk_capture._sealed({
        "schema_version": 1,
        "kind": "sol_workload_campaign_event",
        "event_ordinal": len(events) + 1,
        "prior_event_seal_sha256": (
            events[-1]["event_seal_sha256"] if events else None
        ),
        "recorded_at_utc": chunk_capture._utc_text(),
        **copy.deepcopy(payload),
    }, "event_seal_sha256")
    path = (
        output_root / EVENT_DIR
        / f"event-{event['event_ordinal']:06d}.json"
    )
    chunk_capture._write_json_new(path, event)
    return event


def _done_completion_payload(
    *, output_root: Path, chunk: dict[str, Any], done: dict[str, Any],
) -> dict[str, Any]:
    done_path = (
        output_root / "sol_workload_chunks" / chunk["chunk_id"] / "DONE.json"
    )
    selected_ordinals = [
        history["selected_attempt_ordinal"]
        for history in done["attempt_histories"]
    ]
    return {
        "chunk_id": chunk["chunk_id"],
        "cohort": chunk["cohort"],
        "outcome": "completed",
        "retry_ordinal": max(selected_ordinals),
        "preflight_sha256s": sorted({
            reference["sha256"] for reference in done["preflights"]
        }),
        "done_relative_path": (
            Path("sol_workload_chunks") / chunk["chunk_id"] / "DONE.json"
        ).as_posix(),
        "done_sha256": chunk_capture._sha256_output_file(
            done_path, role="published supplemental DONE",
            operation="campaign_done_hash",
        ),
        "failure_code": None,
        "failure_detail": None,
        "row_outcomes": [],
    }


def _ensure_completed_chunk_event(
    *, output_root: Path, chunk: dict[str, Any], done: dict[str, Any],
) -> dict[str, Any]:
    """Append exactly one DONE-derived completion event after a kill window."""
    expected = _done_completion_payload(
        output_root=output_root, chunk=chunk, done=done
    )
    events = load_event_chain(output_root)
    completed = [
        event for event in events
        if event.get("chunk_id") == chunk["chunk_id"]
        and event.get("outcome") == "completed"
    ]
    if len(completed) > 1:
        raise CampaignRefusal(
            "campaign_completion_event_duplicate", chunk["chunk_id"]
        )
    if completed:
        event = completed[0]
        observed = {key: event.get(key) for key in expected}
        if observed != expected or any(
            later.get("chunk_id") == chunk["chunk_id"]
            for later in events[event["event_ordinal"]:]
        ):
            raise CampaignRefusal(
                "campaign_completion_event_conflict", chunk["chunk_id"]
            )
        return event
    return append_event(output_root, expected)


def _recover_published_completion_events(
    *, output_root: Path, matrix: dict[str, Any], matrix_path: Path,
    canonical_matrix: dict[str, Any], canonical_matrix_path: Path,
) -> None:
    """Close DONE→event kill windows without invoking capture work."""
    for chunk in matrix["chunks"]:
        final_dir = output_root / "sol_workload_chunks" / chunk["chunk_id"]
        final_metadata = chunk_capture._output_lstat(
            final_dir, operation="campaign_published_chunk_lstat",
        )
        if final_metadata is None:
            continue
        done = chunk_capture._load_output_object(
            final_dir / "DONE.json", role="published supplemental DONE",
            operation="campaign_published_done_read",
        )
        chunk_capture.verify_done_manifest(
            done, final_dir=final_dir, matrix=matrix,
            matrix_path=matrix_path, canonical_matrix=canonical_matrix,
            canonical_matrix_path=canonical_matrix_path,
            campaign_root=output_root,
        )
        _ensure_completed_chunk_event(
            output_root=output_root, chunk=chunk, done=done
        )


def _latest_chunk_events(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        chunk_id = event.get("chunk_id")
        if isinstance(chunk_id, str):
            latest[chunk_id] = event
    return latest


def _latest_row_events(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str, str], tuple[dict[str, Any], dict[str, Any]]]:
    latest: dict[
        tuple[str, str, str], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    for event in events:
        outcomes = event.get("row_outcomes", [])
        if not isinstance(outcomes, list):
            raise CampaignRefusal(
                "campaign_event_row_outcomes_invalid",
                repr(event.get("event_ordinal")),
            )
        seen: set[tuple[str, str, str]] = set()
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                raise CampaignRefusal(
                    "campaign_event_row_outcomes_invalid", repr(outcome)
                )
            key = _row_identity(outcome.get("row_key"))
            if key in seen or outcome.get("status") not in {
                "budget_blocked", "retry_exhausted", "storage_blocked",
                "storage_integrity_blocked", "integrity_failed",
                "measured_infeasible_trace_capture",
            }:
                raise CampaignRefusal(
                    "campaign_event_row_outcomes_invalid", repr(outcome)
                )
            seen.add(key)
            latest[key] = (event, outcome)
    return latest


def _canonical_pretty_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _campaign_evidence_storage_usage(output_root: Path) -> dict[str, int]:
    """Measure retained evidence without self-counting progress snapshots.

    Snapshots describe campaign state; they are not capture evidence.  Counting
    an earlier snapshot while constructing the next manifest made identical
    reconciliation non-idempotent because PROGRESS.json changed solely due to
    its own prior publication.  All snapshot and pending-snapshot bytes are
    excluded, while symlinks remain a hard refusal everywhere.
    """
    logical = 0
    physical = 0
    file_count = 0
    inodes: set[tuple[int, int]] = set()
    root_metadata = chunk_capture._output_lstat(
        output_root, operation="campaign_usage_root_lstat",
    )
    if root_metadata is None:
        return {
            "logical_bytes": 0,
            "physical_bytes": 0,
            "file_count": 0,
            "unique_inode_count": 0,
        }
    snapshot_parent = output_root / SNAPSHOT_DIR
    operational_locks = {
        output_root / ".capture.lock",
        output_root / ".sol_workload_campaign.lock",
    }
    for path in chunk_capture._output_tree_paths(
        output_root, operation="campaign_usage_tree_walk",
    ):
        observed = chunk_capture._output_lstat(
            path, operation="campaign_usage_entry_lstat",
        )
        if observed is None:
            raise CampaignRefusal("campaign_output_path_invalid", str(path))
        if stat.S_ISLNK(observed.st_mode):
            raise CampaignRefusal(
                "campaign_output_symlink_refused", str(path)
            )
        try:
            path.relative_to(snapshot_parent)
        except ValueError:
            pass
        else:
            continue
        if not stat.S_ISREG(observed.st_mode):
            continue
        if path in operational_locks:
            continue
        logical += observed.st_size
        file_count += 1
        identity = (observed.st_dev, observed.st_ino)
        if identity not in inodes:
            inodes.add(identity)
            physical += observed.st_size
    return {
        "logical_bytes": logical,
        "physical_bytes": physical,
        "file_count": file_count,
        "unique_inode_count": len(inodes),
    }


def _publish_snapshot_atomic_untyped(
    *, snapshot_root: Path, documents: dict[str, dict[str, Any]],
) -> None:
    parent = snapshot_root.parent
    chunk_capture._mkdir_nofollow(parent, exist_ok=True)
    expected = {
        name: _canonical_pretty_json(document)
        for name, document in documents.items()
    }

    def verify_complete(root: Path) -> bool:
        metadata = chunk_capture._output_lstat(
            root, operation="campaign_snapshot_root_lstat",
        )
        if metadata is None or stat.S_ISLNK(
            metadata.st_mode
        ) or not stat.S_ISDIR(metadata.st_mode):
            raise CampaignRefusal("campaign_snapshot_path_invalid", str(root))
        names = {
            entry.name for entry in chunk_capture._output_directory_entries(
                root, operation="campaign_snapshot_entries",
            )
        }
        if names != set(expected):
            return False
        for name, payload in expected.items():
            path = root / name
            observed, _metadata = chunk_capture._read_output_regular_nofollow(
                path, role="campaign snapshot document",
                operation="campaign_snapshot_document_read",
            )
            if observed != payload:
                raise CampaignRefusal(
                    "campaign_snapshot_content_mismatch", str(path)
                )
        return True

    if chunk_capture._output_lstat(
        snapshot_root, operation="campaign_snapshot_final_lstat",
    ) is not None:
        if verify_complete(snapshot_root):
            return
        raise CampaignRefusal(
            "campaign_snapshot_incomplete_final", str(snapshot_root)
        )
    pending = parent / f".{snapshot_root.name}.pending"
    chunk_capture._mkdir_nofollow(pending, exist_ok=True)
    observed_names = {
        entry.name for entry in chunk_capture._output_directory_entries(
            pending, operation="campaign_snapshot_pending_entries",
        )
    }
    unexpected = observed_names - set(expected)
    if unexpected:
        raise CampaignRefusal(
            "campaign_snapshot_pending_extra", repr(sorted(unexpected))
        )
    for name, payload in expected.items():
        path = pending / name
        if chunk_capture._output_lstat(
            path, operation="campaign_snapshot_pending_document_lstat",
        ) is not None:
            observed, _metadata = chunk_capture._read_output_regular_nofollow(
                path, role="pending campaign snapshot document",
                operation="campaign_snapshot_pending_document_read",
            )
            if observed != payload:
                raise CampaignRefusal(
                    "campaign_snapshot_pending_mismatch", str(path)
                )
            continue
        chunk_capture._write_json_new(path, documents[name])
        path.chmod(0o444)
    if not verify_complete(pending):
        raise CampaignRefusal(
            "campaign_snapshot_pending_incomplete", str(pending)
        )
    pending_fd = os.open(pending, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(pending_fd)
    finally:
        os.close(pending_fd)
    os.replace(pending, snapshot_root)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _publish_snapshot_atomic(
    *, snapshot_root: Path, documents: dict[str, dict[str, Any]],
) -> None:
    """Publish one snapshot with storage typing at its mutation boundary."""
    try:
        _publish_snapshot_atomic_untyped(
            snapshot_root=snapshot_root, documents=documents,
        )
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_snapshot_publication",
        )
        if mapped is not None:
            raise mapped from exc
        raise


def reconcile_campaign(
    *, matrix: dict[str, Any], matrix_path: Path,
    canonical_matrix: dict[str, Any], canonical_matrix_path: Path,
    output_root: Path, publish: bool = True,
    full_coverage_inventory: dict[str, Any] | None = None,
    full_coverage_inventory_path: Path | None = None,
    live_coverage_inventory: dict[str, Any] | None = None,
    live_coverage_inventory_summary: dict[str, Any] | None = None,
    campaign_identity: dict[str, Any] | None = None,
    offline: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize exact whole-matrix state and three claim-safe exports."""
    closure = validate_whole_matrix_closure(matrix)
    if full_coverage_inventory is None:
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_missing",
            "explicit current-input inventory is required",
        )
    full_inventory = copy.deepcopy(full_coverage_inventory)
    full_inventory_summary = validate_full_coverage_inventory(full_inventory)
    if (
        full_coverage_inventory_path is None
        or full_coverage_inventory_path.is_symlink()
        or not full_coverage_inventory_path.is_file()
        or chunk_capture.sha256_file(full_coverage_inventory_path)
        != full_inventory_summary["sha256"]
    ):
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_invalid",
            str(full_coverage_inventory_path),
        )
    full_inventory_summary["input_role"] = "full_coverage_inventory"
    representative_mode = chunk_capture.is_representative_execution_matrix(
        matrix
    )
    expected_inventory_commit = (
        matrix.get("authority_source_commit")
        if representative_mode else matrix.get("source_commit")
    )
    if (
        live_coverage_inventory is None
        or live_coverage_inventory_summary is None
        or live_coverage_inventory.get("source_commit")
        != expected_inventory_commit
        or live_coverage_inventory_summary.get("matrix_schedule_rows")
        != len(matrix.get("rows") or [])
    ):
        raise CampaignRefusal(
            "campaign_live_coverage_inventory_invalid",
            repr(live_coverage_inventory_summary),
        )
    live_inventory = copy.deepcopy(live_coverage_inventory)
    live_inventory_summary = copy.deepcopy(
        live_coverage_inventory_summary
    )
    live_inventory_summary.pop("path", None)
    live_inventory_summary.pop("owner_identity", None)
    live_inventory_summary["input_role"] = "live_coverage_inventory"
    if campaign_identity is None:
        raise CampaignRefusal(
            "campaign_root_identity_missing", str(output_root)
        )
    if offline:
        root_identity, _owner_payload, _owner_pin = (
            chunk_capture._load_retained_input_owner(output_root)
        )
        if root_identity.get("campaign_identity") != campaign_identity:
            raise CampaignRefusal(
                "campaign_retained_identity_mismatch", str(output_root)
            )
    else:
        root_identity = chunk_capture.establish_supplemental_output_root(
            output_root, campaign_identity=campaign_identity
        )
    events = load_event_chain(output_root)
    latest_rows = _latest_row_events(events)
    done_results: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    for chunk in matrix["chunks"]:
        final_dir = output_root / "sol_workload_chunks" / chunk["chunk_id"]
        if chunk_capture._output_lstat(
            final_dir, operation="campaign_reconcile_chunk_lstat",
        ) is None:
            continue
        done = chunk_capture._load_output_object(
            final_dir / "DONE.json", role="supplemental chunk DONE",
            operation="campaign_reconcile_done_read",
        )
        chunk_capture.verify_done_manifest(
            done, final_dir=final_dir, matrix=matrix,
            matrix_path=matrix_path, canonical_matrix=canonical_matrix,
            canonical_matrix_path=canonical_matrix_path,
            campaign_root=output_root, repo_root=repo_root,
            require_current_state=not offline,
            verify_git_provenance=False,
        )
        for result in done["cell_results"]:
            key = _row_identity(result["workload_row_binding"]["row_key"])
            if key in done_results:
                raise CampaignRefusal(
                    "campaign_done_result_duplicate", repr(key)
                )
            done_results[key] = (chunk["chunk_id"], result)
    cohort_rows: dict[str, list[dict[str, Any]]] = {
        "native": [], "partial_native_row": [], "reduced_proxy": [],
    }
    status_counts: Counter[str] = Counter()
    all_rows: list[dict[str, Any]] = []
    for key in closure["scheduled_keys"]:
        row = closure["rows_by_key"][key]
        chunk_id = closure["chunks_by_row"][key]
        completed = done_results.get(key)
        row_event = latest_rows.get(key)
        if completed is not None:
            result = completed[1]
            status = (
                "admitted"
                if result["disposition"] == "admitted"
                else "typed_incomplete"
            )
            result_binding = {
                "chunk_done_relative_path": (
                    Path("sol_workload_chunks") / chunk_id / "DONE.json"
                ).as_posix(),
                "cell_result_sha256": chunk_capture.canonical_sha256(result),
                "typed_reason": result.get("typed_reason"),
                "disposition": result["disposition"],
                "launch_completeness": copy.deepcopy(
                    result["launch_completeness"]
                ),
            }
        elif row_event is not None:
            event, outcome = row_event
            status = outcome["status"]
            result_binding = {
                "campaign_event_ordinal": event["event_ordinal"],
                "failure_code": outcome.get("failure_code"),
                "attempt_ordinal": outcome.get("attempt_ordinal"),
            }
        else:
            status = "unattempted"
            result_binding = None
        export_row = {
            "key": copy.deepcopy(row["key"]),
            "chunk_id": chunk_id,
            "cohort": row["cohort"],
            "category": row["category"],
            "claim_scope": row["claim_scope"],
            "native_shape": row["native_shape"],
            "native_predictive": row["native_predictive"],
            "problem_level_runnable": row["problem_level_runnable"],
            "variant_contract_sha256": row["variant_contract_sha256"],
            "row_evidence_sha256": row["row_evidence_sha256"],
            "source_workload_index": row["source_workload_index"],
            "source_workload_uuid": row["source_workload_uuid"],
            "source_axes": copy.deepcopy(row["source_axes"]),
            "source_row_sha256": row["source_row_sha256"],
            "capture_workload_index": row["capture_workload_index"],
            "capture_axes": copy.deepcopy(row["capture_axes"]),
            "capture_row_sha256": row["capture_row_sha256"],
            "flashinfer_corpus": copy.deepcopy(row["flashinfer_corpus"]),
            "launch_scope": row["launch_scope"],
            "launch_completeness_contract": copy.deepcopy(
                row["launch_completeness_contract"]
            ),
            "status": status,
            "result_binding": result_binding,
        }
        export_row["row_status_sha256"] = chunk_capture.canonical_sha256(
            export_row
        )
        status_counts[status] += 1
        all_rows.append(export_row)
        cohort_rows[row["cohort"]].append(export_row)
    if len(all_rows) != matrix["counts"]["rows"]:
        raise CampaignRefusal(
            "campaign_reconciliation_row_count_invalid", str(len(all_rows))
        )
    snapshot_binding = {
        "event_count": len(events),
        "event_chain_head": (
            events[-1]["event_seal_sha256"] if events else None
        ),
        "matrix_sha256": chunk_capture.sha256_file(matrix_path),
        "row_state_sha256": chunk_capture.canonical_sha256(all_rows),
        "full_coverage_inventory_sha256": full_inventory_summary["sha256"],
        "live_coverage_inventory_sha256": live_inventory_summary["sha256"],
    }
    snapshot_id = (
        f"snapshot-{len(events):06d}-"
        f"{chunk_capture.canonical_sha256(snapshot_binding)}"
    )
    snapshot_root = output_root / SNAPSHOT_DIR / snapshot_id
    exports: dict[str, dict[str, Any]] = {}
    export_names = {
        "native": "full_native_rows.json",
        "partial_native_row": "partial_native_rows.json",
        "reduced_proxy": "reduced_proxy_rows.json",
    }
    for cohort, filename in export_names.items():
        document = chunk_capture._sealed({
            "schema_version": 1,
            "kind": "sol_workload_campaign_cohort_export",
            "cohort": cohort,
            "claim_policy": matrix["claim_policy"],
            "matrix_sha256": chunk_capture.sha256_file(matrix_path),
            "row_count": len(cohort_rows[cohort]),
            "rows": cohort_rows[cohort],
        }, "export_seal_sha256")
        payload = (
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        ).encode()
        exports[cohort] = {
            "filename": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "row_count": len(cohort_rows[cohort]),
            "document": document,
        }
    coverage_payload = coverage_inventory_builder.canonical_json_bytes(
        full_inventory
    )
    exports["full_coverage_inventory"] = {
        "filename": "full_coverage_inventory.json",
        "sha256": hashlib.sha256(coverage_payload).hexdigest(),
        "row_count": full_inventory_summary["workload_row_count"],
        "document": full_inventory,
    }
    live_coverage_payload = (
        chunk_capture.live_coverage.canonical_json_bytes(live_inventory)
    )
    if hashlib.sha256(live_coverage_payload).hexdigest() != (
        live_inventory_summary["sha256"]
    ):
        raise CampaignRefusal(
            "campaign_live_coverage_inventory_invalid", "hash mismatch"
        )
    exports["live_coverage_inventory"] = {
        "filename": "live_coverage_inventory.json",
        "sha256": live_inventory_summary["sha256"],
        "row_count": live_inventory["counts"]["source_workload_rows"],
        "document": live_inventory,
    }
    nonterminal = {
        "unattempted", "budget_blocked", "retry_exhausted",
        "storage_blocked", "storage_integrity_blocked", "integrity_failed",
    }
    if nonterminal.intersection(status_counts):
        state = "incomplete"
    elif status_counts.get("typed_incomplete", 0):
        state = "terminal_accounted_with_typed_incomplete"
    elif status_counts.get("measured_infeasible_trace_capture", 0):
        state = "terminal_accounted_with_measured_infeasibility"
    else:
        state = "fully_admitted"
    cohort_accounting: dict[str, Any] = {}
    for cohort in ("native", "partial_native_row", "reduced_proxy"):
        cohort_variants = [
            variant for variant in matrix["variants"]
            if variant["cohort"] == cohort
        ]
        cohort_chunks = [
            chunk for chunk in matrix["chunks"] if chunk["cohort"] == cohort
        ]
        cohort_status = Counter(
            row["status"] for row in cohort_rows[cohort]
        )
        cohort_accounting[cohort] = {
            "problems": len({variant["slug"] for variant in cohort_variants}),
            "variants": len(cohort_variants),
            "rows": len(cohort_rows[cohort]),
            "chunks": len(cohort_chunks),
            "status_counts": dict(sorted(cohort_status.items())),
        }
    manifest = chunk_capture._sealed({
        "schema_version": 1,
        "kind": "sol_workload_campaign_progress",
        "campaign_state": state,
        "snapshot_id": snapshot_id,
        "root_identity": root_identity,
        "supplemental_matrix": {
            "input_role": "supplemental_matrix",
            "sha256": chunk_capture.sha256_file(matrix_path),
            "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        },
        "canonical_matrix": {
            "input_role": "canonical_matrix",
            "sha256": chunk_capture.sha256_file(canonical_matrix_path),
            "matrix_seal_sha256": canonical_matrix["matrix_seal_sha256"],
        },
        "event_count": len(events),
        "event_chain_head": (
            events[-1]["event_seal_sha256"] if events else None
        ),
        "exact_closure": {
            "problems": closure["problem_count"],
            "variants": len(matrix["variants"]),
            "rows": len(matrix["rows"]),
            "chunks": len(matrix["chunks"]),
        },
        "cohort_accounting": cohort_accounting,
        "status_counts": dict(sorted(status_counts.items())),
        "full_coverage_inventory": full_inventory_summary,
        "live_coverage_inventory": live_inventory_summary,
        "exports": {
            cohort: {
                key: value for key, value in record.items()
                if key != "document"
            }
            for cohort, record in exports.items()
        },
        "storage_usage": _campaign_evidence_storage_usage(output_root),
        "generated_at_utc": (
            events[-1]["recorded_at_utc"]
            if events else root_identity["created_at_utc"]
        ),
    }, "progress_seal_sha256")
    documents = {
        record["filename"]: record["document"]
        for record in exports.values()
    }
    documents["PROGRESS.json"] = manifest
    if publish:
        for record in exports.values():
            if hashlib.sha256(
                _canonical_pretty_json(record["document"])
            ).hexdigest() != record["sha256"]:
                raise CampaignRefusal(
                    "campaign_export_hash_mismatch", record["filename"]
                )
        _publish_snapshot_atomic(
            snapshot_root=snapshot_root, documents=documents
        )
    return {
        "snapshot_root": snapshot_root, "manifest": manifest,
        "documents": documents,
    }


def _retained_campaign_input(
    *, campaign_root: Path, campaign_identity: dict[str, Any], role: str,
    repo_root: Path,
) -> tuple[Path, bytes]:
    identities = campaign_identity.get("input_file_identities")
    identity = identities.get(role) if isinstance(identities, dict) else None
    if not isinstance(identity, dict) or identity.get("role") != role:
        raise CampaignRefusal("campaign_retained_inputs_invalid", role)
    reference = chunk_capture._portable_input_reference(identity)
    payload = chunk_capture._read_portable_input_reference(
        reference, campaign_root=campaign_root, repo_root=repo_root,
        source_commit=campaign_identity["source_commit"],
        verify_git_provenance=False,
    )
    relative = chunk_capture._safe_relative_path(
        reference["owner_relative_path"], role=role
    )
    return campaign_root / relative, payload


def _decode_retained_json(payload: bytes, *, role: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignRefusal(
            "campaign_retained_input_json_invalid", f"{role}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CampaignRefusal(
            "campaign_retained_input_json_invalid", f"{role}: root"
        )
    return value


def _offline_live_inventory_summary(
    *, inventory: dict[str, Any], payload: bytes,
    matrix: dict[str, Any], repo_root: Path,
) -> dict[str, Any]:
    try:
        schema = json.loads(
            (repo_root / chunk_capture.LIVE_COVERAGE_SCHEMA.relative_to(
                chunk_capture.REPO_ROOT
            )).read_text()
        )
        jsonschema.validate(inventory, schema)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise CampaignRefusal(
            "campaign_live_coverage_inventory_invalid", str(exc)
        ) from exc
    if payload != chunk_capture.live_coverage.canonical_json_bytes(inventory):
        raise CampaignRefusal(
            "campaign_live_coverage_inventory_invalid", "noncanonical bytes"
        )
    matrix_keys = [
        _row_identity(row.get("key")) for row in matrix.get("rows") or []
    ]
    inventory_keys = [
        _row_identity(key)
        for row in inventory.get("rows") or []
        for key in row.get("matrix_row_keys") or []
    ]
    if (
        inventory.get("source_commit") != matrix.get("source_commit")
        or len(matrix_keys) != len(set(matrix_keys))
        or len(inventory_keys) != len(set(inventory_keys))
        or set(inventory_keys) != set(matrix_keys)
    ):
        raise CampaignRefusal(
            "campaign_live_coverage_inventory_invalid", "row closure"
        )
    counts = inventory["counts"]
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_commit": inventory["source_commit"],
        "source_workload_rows": counts["source_workload_rows"],
        "matrix_schedule_rows": len(matrix_keys),
        "source_rows_with_any_schedule": counts[
            "source_rows_with_any_schedule"
        ],
        "source_rows_unscheduled_in_scope": counts[
            "source_rows_unscheduled_in_scope"
        ],
    }


def _assert_no_absolute_evidence_strings(value: Any, *, role: str) -> None:
    if isinstance(value, str) and value.startswith("/"):
        raise CampaignRefusal(
            "campaign_absolute_evidence_path_forbidden", f"{role}: {value}"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_absolute_evidence_strings(
                item, role=f"{role}/{key}"
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_absolute_evidence_strings(
                item, role=f"{role}/{index}"
            )


_EVENT_BASE_FIELDS = {
    "schema_version", "kind", "event_ordinal", "prior_event_seal_sha256",
    "recorded_at_utc", "event_seal_sha256",
}
_FAILURE_EVENT_FIELDS = {
    "chunk_id", "cohort", "outcome", "retry_ordinal",
    "preflight_owner_relative_path", "preflight_sha256", "failure_code",
    "failure_detail_sha256", "row_outcomes", "storage_state",
}
_FAILURE_OUTCOMES = {
    "chunk_failed", "budget_blocked", "campaign_budget_blocked",
    "integrity_blocked", "storage_blocked", "storage_integrity_blocked",
    "measured_infeasible",
}


def _storage_state(path: Path) -> dict[str, int]:
    observed = os.statvfs(path)
    return {
        "available_bytes": observed.f_bavail * observed.f_frsize,
        "available_inodes": observed.f_favail,
    }


def _validate_offline_event_completion_closure(
    *, campaign_root: Path, matrix: dict[str, Any], events: list[dict[str, Any]],
) -> None:
    chunks = {chunk["chunk_id"]: chunk for chunk in matrix["chunks"]}
    rows_to_chunks = {
        _row_identity(row_key): chunk["chunk_id"]
        for chunk in matrix["chunks"] for row_key in chunk["rows"]
    }
    completion_events: dict[str, list[dict[str, Any]]] = {
        chunk_id: [] for chunk_id in chunks
    }
    for event in events:
        chunk_id = event.get("chunk_id")
        chunk = chunks.get(chunk_id)
        if chunk is None or event.get("cohort") != chunk["cohort"]:
            raise CampaignRefusal(
                "campaign_event_chunk_invalid", repr(chunk_id)
            )
        outcome = event.get("outcome")
        if outcome == "completed":
            final_dir = campaign_root / "sol_workload_chunks" / chunk_id
            final_metadata = chunk_capture._output_lstat(
                final_dir, operation="offline_done_directory_lstat",
            )
            if final_metadata is None or stat.S_ISLNK(
                final_metadata.st_mode
            ) or not stat.S_ISDIR(final_metadata.st_mode):
                raise CampaignRefusal(
                    "campaign_completion_event_without_done", chunk_id
                )
            done = chunk_capture._load_output_object(
                final_dir / "DONE.json", role="offline supplemental DONE",
                operation="offline_done_read",
            )
            expected = _done_completion_payload(
                output_root=campaign_root, chunk=chunk, done=done
            )
            if (
                set(event) != _EVENT_BASE_FIELDS | set(expected)
                or {key: event.get(key) for key in expected} != expected
            ):
                raise CampaignRefusal(
                    "campaign_completion_event_conflict", chunk_id
                )
            completion_events[chunk_id].append(event)
            continue
        if outcome not in _FAILURE_OUTCOMES or set(event) != (
            _EVENT_BASE_FIELDS | _FAILURE_EVENT_FIELDS
        ):
            raise CampaignRefusal(
                "campaign_event_shape_invalid", repr(event.get("event_ordinal"))
            )
        expected_status = {
            "chunk_failed": "retry_exhausted",
            "budget_blocked": "budget_blocked",
            "campaign_budget_blocked": None,
            "integrity_blocked": "integrity_failed",
            "storage_blocked": "storage_blocked",
            "storage_integrity_blocked": "storage_integrity_blocked",
            "measured_infeasible": "measured_infeasible_trace_capture",
        }[outcome]
        storage_state = event.get("storage_state")
        if outcome in {"storage_blocked", "storage_integrity_blocked"}:
            if (
                not isinstance(storage_state, dict)
                or set(storage_state) != {"available_bytes", "available_inodes"}
                or any(type(value) is not int or value < 0 for value in storage_state.values())
            ):
                raise CampaignRefusal(
                    "campaign_event_shape_invalid", repr(event["event_ordinal"])
                )
        elif storage_state is not None:
            raise CampaignRefusal(
                "campaign_event_shape_invalid", repr(event["event_ordinal"])
            )
        row_outcomes = event.get("row_outcomes")
        if not isinstance(row_outcomes, list):
            raise CampaignRefusal(
                "campaign_event_row_outcomes_invalid", repr(event["event_ordinal"])
            )
        seen_rows: set[tuple[str, str, str]] = set()
        for row in row_outcomes:
            if not isinstance(row, dict) or set(row) != {
                "row_key", "status", "failure_code", "attempt_ordinal",
            }:
                raise CampaignRefusal(
                    "campaign_event_row_outcomes_invalid", repr(row)
                )
            key = _row_identity(row["row_key"])
            if (
                key in seen_rows or rows_to_chunks.get(key) != chunk_id
                or row.get("status") != expected_status
                or not isinstance(row.get("failure_code"), str)
                or (
                    row.get("attempt_ordinal") is not None
                    and (
                        type(row["attempt_ordinal"]) is not int
                        or row["attempt_ordinal"] < 1
                    )
                )
            ):
                raise CampaignRefusal(
                    "campaign_event_row_outcomes_invalid", repr(row)
                )
            seen_rows.add(key)
        if expected_status is None and row_outcomes:
            raise CampaignRefusal(
                "campaign_event_row_outcomes_invalid", repr(event["event_ordinal"])
            )
        relative = chunk_capture._safe_relative_path(
            event.get("preflight_owner_relative_path"), role="event preflight"
        )
        expected_prefix = (
            "sol_workload_campaign", "inputs", "preflights",
            matrix["source_commit"], chunk_id,
        )
        preflight_path = campaign_root / relative
        if (
            relative.parts[:5] != expected_prefix
            or len(relative.parts) != 6
            or event.get("preflight_sha256")
            != chunk_capture._sha256_output_file(
                preflight_path, role="retained campaign preflight",
                operation="offline_preflight_hash",
            )
            or not isinstance(event.get("failure_code"), str)
            or chunk_capture.SHA256_RE.fullmatch(
                str(event.get("failure_detail_sha256", ""))
            ) is None
        ):
            raise CampaignRefusal(
                "campaign_event_shape_invalid", repr(event["event_ordinal"])
            )
    for chunk_id, chunk in chunks.items():
        final_dir = campaign_root / "sol_workload_chunks" / chunk_id
        final_metadata = chunk_capture._output_lstat(
            final_dir, operation="offline_done_cardinality_lstat",
        )
        has_done = (
            final_metadata is not None
            and stat.S_ISDIR(final_metadata.st_mode)
            and not stat.S_ISLNK(final_metadata.st_mode)
        )
        completed = completion_events[chunk_id]
        if has_done != (len(completed) == 1):
            raise CampaignRefusal(
                "campaign_done_completion_event_cardinality",
                f"{chunk_id}: done={has_done}; events={len(completed)}",
            )
        if completed and any(
            event.get("chunk_id") == chunk_id
            for event in events[completed[0]["event_ordinal"]:]
        ):
            raise CampaignRefusal(
                "campaign_completion_event_conflict", chunk_id
            )


def verify_campaign_checkpoint_offline(
    campaign_root: Path, *, repo_root: Path = REPO_ROOT,
) -> None:
    """Byte-rederive the unique frontier using retained campaign bytes only."""
    campaign_root = Path(campaign_root)
    marker, _marker_payload, _marker_pin = (
        chunk_capture._load_retained_input_owner(campaign_root)
    )
    campaign_identity = marker["campaign_identity"]
    loaded: dict[str, tuple[Path, bytes, dict[str, Any]]] = {}
    for role in (
        "supplemental_matrix", "canonical_matrix",
        "full_coverage_inventory", "live_coverage_inventory",
    ):
        path, payload = _retained_campaign_input(
            campaign_root=campaign_root, campaign_identity=campaign_identity,
            role=role, repo_root=repo_root,
        )
        loaded[role] = (
            path, payload, _decode_retained_json(payload, role=role)
        )
    matrix_path, matrix_payload, matrix = loaded["supplemental_matrix"]
    canonical_path, _canonical_payload, canonical_matrix = loaded[
        "canonical_matrix"
    ]
    full_path, full_payload, full_inventory = loaded[
        "full_coverage_inventory"
    ]
    _live_path, live_payload, live_inventory = loaded[
        "live_coverage_inventory"
    ]
    if matrix_payload != chunk_capture.workload_matrix.serialize_matrix(matrix):
        raise CampaignRefusal(
            "campaign_retained_input_json_invalid", "supplemental_matrix bytes"
        )
    for document, label in (
        (matrix, "supplemental_matrix"),
        (canonical_matrix, "canonical_matrix"),
    ):
        seal = document.get("matrix_seal_sha256")
        unsigned = {
            key: value for key, value in document.items()
            if key != "matrix_seal_sha256"
        }
        if seal != chunk_capture.canonical_sha256(unsigned):
            raise CampaignRefusal(
                "campaign_retained_input_seal_invalid", label
            )
    full_summary = validate_full_coverage_inventory(full_inventory)
    if full_payload != coverage_inventory_builder.canonical_json_bytes(
        full_inventory
    ) or full_summary["sha256"] != hashlib.sha256(full_payload).hexdigest():
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_invalid", "retained bytes"
        )
    live_summary = _offline_live_inventory_summary(
        inventory=live_inventory, payload=live_payload,
        matrix=matrix, repo_root=repo_root,
    )
    rederived = reconcile_campaign(
        matrix=matrix, matrix_path=matrix_path,
        canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_path,
        output_root=campaign_root, publish=False,
        full_coverage_inventory=full_inventory,
        full_coverage_inventory_path=full_path,
        live_coverage_inventory=live_inventory,
        live_coverage_inventory_summary=live_summary,
        campaign_identity=campaign_identity, offline=True,
        repo_root=repo_root,
    )
    events = load_event_chain(campaign_root)
    _validate_offline_event_completion_closure(
        campaign_root=campaign_root, matrix=matrix, events=events
    )
    expected_root = rederived["snapshot_root"]
    event_count = rederived["manifest"]["event_count"]
    snapshot_parent = campaign_root / SNAPSHOT_DIR
    prefix = f"snapshot-{event_count:06d}-"
    snapshot_metadata = chunk_capture._output_lstat(
        snapshot_parent, operation="checkpoint_snapshot_parent_lstat",
    )
    snapshot_entries = (
        chunk_capture._output_directory_entries(
            snapshot_parent, operation="checkpoint_snapshot_parent_entries",
        )
        if snapshot_metadata is not None
        and stat.S_ISDIR(snapshot_metadata.st_mode)
        and not stat.S_ISLNK(snapshot_metadata.st_mode)
        else []
    )
    candidates = [
        path for path in snapshot_entries
        if path.name.startswith(prefix) and not path.name.startswith(".")
    ]
    pending = [
        path for path in snapshot_entries
        if path.name.startswith(f".{prefix}")
    ]
    if candidates != [expected_root] or pending:
        raise CampaignRefusal(
            "campaign_checkpoint_frontier_not_unique",
            repr({"candidates": candidates, "pending": pending}),
        )
    expected_payloads = {
        name: _canonical_pretty_json(document)
        for name, document in rederived["documents"].items()
    }
    observed_names = {
        path.name for path in chunk_capture._output_directory_entries(
            expected_root, operation="checkpoint_frontier_entries",
        )
    }
    if observed_names != set(expected_payloads):
        raise CampaignRefusal(
            "campaign_checkpoint_frontier_mismatch", repr(observed_names)
        )
    for name, payload in expected_payloads.items():
        path = expected_root / name
        observed, _metadata = chunk_capture._read_output_regular_nofollow(
            path, role="campaign checkpoint frontier",
            operation="checkpoint_frontier_document_read",
        )
        if observed != payload:
            raise CampaignRefusal(
                "campaign_checkpoint_frontier_mismatch", str(path)
            )
    _assert_no_absolute_evidence_strings(events, role="events")
    _assert_no_absolute_evidence_strings(
        rederived["documents"], role="frontier"
    )


def _orchestrator_lock(output_root: Path):
    path = output_root / ".sol_workload_campaign.lock"
    root_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)

    def close_after_fault(descriptor: int) -> None:
        try:
            os.close(descriptor)
        except OSError:
            # The caller is already handling the first fault.  Cleanup never
            # replaces it with a secondary close error.
            pass

    try:
        root_fd = os.open(output_root, root_flags)
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_orchestrator_lock_open",
        )
        if mapped is not None:
            raise mapped from exc
        raise CampaignRefusal("campaign_lock_invalid", str(path)) from exc
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
    except OSError as exc:
        close_after_fault(root_fd)
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_orchestrator_lock_open",
        )
        if mapped is not None:
            raise mapped from exc
        raise CampaignRefusal("campaign_lock_invalid", str(path)) from exc
    try:
        os.close(root_fd)
    except OSError as exc:
        close_after_fault(descriptor)
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_orchestrator_lock_root_close",
        )
        if mapped is not None:
            raise mapped from exc
        raise CampaignRefusal("campaign_lock_invalid", str(path)) from exc
    try:
        observed = os.fstat(descriptor)
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_orchestrator_lock_stat",
        )
        close_after_fault(descriptor)
        if mapped is not None:
            raise mapped from exc
        raise CampaignRefusal("campaign_lock_invalid", str(path)) from exc
    if not stat.S_ISREG(observed.st_mode):
        close_after_fault(descriptor)
        raise CampaignRefusal("campaign_lock_invalid", str(path))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        close_after_fault(descriptor)
        raise CampaignRefusal("campaign_orchestrator_active", str(path)) from exc
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_orchestrator_lock_flock",
        )
        close_after_fault(descriptor)
        if mapped is not None:
            raise mapped from exc
        raise CampaignRefusal("campaign_lock_invalid", str(path)) from exc
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        payload = f"pid={os.getpid()}\n".encode("ascii")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(None, "short campaign orchestrator lock write")
        os.fsync(descriptor)
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_orchestrator_lock_record",
        )
        close_after_fault(descriptor)
        if mapped is not None:
            raise mapped from exc
        raise CampaignRefusal("campaign_lock_invalid", str(path)) from exc
    return os.fdopen(descriptor, "r+")


def _new_preflight(
    args: argparse.Namespace, *, source_commit: str, chunk_id: str,
    retry_ordinal: int, matrix_path: Path,
) -> Path:
    token = uuid.uuid4().hex
    path = (
        REPO_ROOT / "results/nvbit_capture_staging/sol_workload_campaign"
        / source_commit / chunk_id
        / f"attempt-{retry_ordinal:03d}-{token}.json"
    )
    smoke_root = (
        Path(args.output_root) / "sol_workload_campaign/preflight_smoke" / token
    )
    chunk_capture._mkdir_nofollow(path.parent, exist_ok=True)
    chunk_capture._mkdir_nofollow(smoke_root.parent, exist_ok=True)
    command = [
        sys.executable, str(LOCAL_LABELS / "campaign_preflight.py"),
        "--output", str(path), "--pins", str(args.pins),
        "--trace-budget", str(args.aggregate_budget),
        "--matrix", str(matrix_path),
        "--output-root", str(args.output_root),
        "--smoke-temp-root", str(smoke_root),
    ]
    clean_env = dict(os.environ)
    for name in (
        "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
        "CUDA_HOME", "CC", "CXX", "CUDAHOSTCXX",
        "FLASHINFER_TRACE_DIR", "CUDA_INJECTION64_PATH",
        "SOL_EXECBENCH_CAPTURE_REQUEST", "SOL_EXECBENCH_CAPTURE_MODE",
        "SOL_EXECBENCH_CAPTURE_PREPARE_MANIFEST",
        "SOL_EXECBENCH_CAPTURE_SELECTION_MANIFEST",
        "SOL_EXECBENCH_CAPTURE_SELECTION_SHA256",
        "SOL_EXECBENCH_CAPTURE_RESULT_MANIFEST",
        "SOL_EXECBENCH_CAPTURE_SOURCE_WORKLOAD_INDEX",
    ):
        clean_env.pop(name, None)
    try:
        local_git_env = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"], cwd=REPO_ROOT,
            env=clean_env, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError as exc:
        raise CampaignRefusal(
            "campaign_preflight_environment_failed", str(exc)
        ) from exc
    for name in local_git_env:
        clean_env.pop(name.strip(), None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    outcome = subprocess.run(
        command, cwd=REPO_ROOT, env=clean_env,
        capture_output=True, text=True, check=False,
    )
    storage_marker = chunk_capture.preflight.parse_preflight_storage_marker(
        outcome.stderr
    )
    if storage_marker is not None:
        campaign_output_operations = {
            "injection_smoke_failure_output_probe",
            "injection_smoke_tree",
            "injection_smoke_root_read",
            "injection_smoke_stats_read",
            "injection_smoke_payload_read",
            "injection_smoke_contract_read",
            "injection_smoke_cleanup",
            "injection_smoke_payload_tree_stat",
            "injection_smoke_payload_tree_walk",
            "injection_smoke_sidecar_stat",
            "injection_smoke_root_stat",
            "injection_smoke_stats_stat",
            "injection_smoke_threadblocks_stat",
            "injection_smoke_threadblocks_walk",
            "injection_smoke_root_create",
            "injection_smoke_traces_create",
            "injection_smoke_workload_output_create",
            "injection_smoke_root_hash",
            "injection_smoke_stats_hash",
        }
        if storage_marker.get("operation") not in campaign_output_operations:
            # The preflight manifest itself is emitted into repo-local staging,
            # not the campaign volume.  Keep that failure semantic/local so
            # run_campaign cannot publish a false campaign-volume receipt.
            raise CampaignRefusal(
                "campaign_preflight_local_storage_failed",
                json.dumps(storage_marker, sort_keys=True),
            )
        code = (
            chunk_capture.STORAGE_CAPACITY_CODE
            if storage_marker["kind"]
            == chunk_capture.preflight.PREFLIGHT_STORAGE_CAPACITY
            else chunk_capture.STORAGE_INTEGRITY_CODE
        )
        raise chunk_capture.WorkloadCaptureRefusal(
            code,
            json.dumps({
                "operation": "campaign_preflight_child",
                "child": storage_marker,
            }, sort_keys=True),
        )
    if outcome.returncode != 0 or not path.is_file():
        raise CampaignRefusal(
            "campaign_preflight_failed",
            f"returncode={outcome.returncode}; stderr={outcome.stderr.strip()}",
        )
    payload = path.read_bytes()
    staging_sha256 = hashlib.sha256(payload).hexdigest()
    # Journals must outlive a disposable/linked capture checkout.  Publish the
    # exact preflight bytes once into the immutable external evidence namespace
    # and pass only that durable path to the chunk runner.  The repo staging
    # source remains as an audit copy but is never needed for resume/verify.
    external_path = (
        Path(args.output_root) / "sol_workload_campaign" / "inputs"
        / "preflights" / source_commit / chunk_id / path.name
    )
    chunk_capture._mkdir_nofollow(external_path.parent, exist_ok=True)
    parent_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_fd = os.open(external_path.parent, parent_flags)
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_preflight_publication_parent_open",
        )
        if mapped is not None:
            raise mapped from exc
        raise
    try:
        descriptor = os.open(
            external_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=parent_fd,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short preflight publication write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
    except FileExistsError as exc:
        raise CampaignRefusal(
            "campaign_preflight_publication_exists", str(external_path)
        ) from exc
    except OSError as exc:
        mapped = chunk_capture.storage_refusal_from_exception(
            exc, operation="campaign_preflight_publication",
        )
        if mapped is not None:
            raise mapped from exc
        raise
    finally:
        os.close(parent_fd)
    path.chmod(0o444)
    external_metadata = chunk_capture._output_lstat(
        external_path, operation="campaign_preflight_published_lstat",
    )
    if external_metadata is None or stat.S_ISLNK(
        external_metadata.st_mode
    ) or not stat.S_ISREG(external_metadata.st_mode):
        raise CampaignRefusal(
            "campaign_preflight_publication_mismatch", str(external_path)
        )
    external_payload, _observed = (
        chunk_capture._read_output_regular_nofollow(
            external_path, role="published campaign preflight",
            operation="campaign_preflight_published_read",
        )
    )
    if (
        hashlib.sha256(external_payload).hexdigest() != staging_sha256
        or external_payload != payload
    ):
        raise CampaignRefusal(
            "campaign_preflight_publication_mismatch", str(external_path)
        )
    return external_path


_RETRYABLE_CODES = frozenset({
    "cell_deadline_exhausted", "trace_route_deadline_exhausted",
    "trace_target_failed", "trace_target_failed_timeout",
    "label_capture_failed", "label_capture_failed_timeout",
    "sol_capture_prepare_failed", "sol_capture_prepare_failed_timeout",
})
_ROW_BUDGET_CODES = frozenset({
    "trace_target_failed_tree_limit_exceeded",
    "supplemental_full_trace_retention_ceiling_exceeded",
    "supplemental_chunk_rows_budget_blocked",
})
_CAMPAIGN_BUDGET_CODES = frozenset({
    "supplemental_retention_budget_exceeded",
})
_STORAGE_CAPACITY_CODES = frozenset({chunk_capture.STORAGE_CAPACITY_CODE})
_STORAGE_INTEGRITY_CODES = frozenset({chunk_capture.STORAGE_INTEGRITY_CODE})


def classify_capture_failure(code: str) -> str:
    """Closed disposition over codes actually emitted by capture producers."""
    if code in _RETRYABLE_CODES:
        return "retryable"
    if code in _ROW_BUDGET_CODES:
        return "row_budget_blocked"
    if code in _CAMPAIGN_BUDGET_CODES:
        return "campaign_budget_blocked"
    if code in _STORAGE_CAPACITY_CODES:
        return "storage_blocked"
    if code in _STORAGE_INTEGRITY_CODES:
        return "storage_integrity_blocked"
    return "integrity_blocked"


def _failure_row_outcomes(
    *, code: str, detail: str, classification: str, terminal_retry: bool,
) -> list[dict[str, Any]]:
    if code == "supplemental_chunk_rows_budget_blocked":
        try:
            records = json.loads(detail)
        except json.JSONDecodeError as exc:
            raise CampaignRefusal(
                "campaign_row_failure_detail_invalid", detail
            ) from exc
        if not isinstance(records, list):
            raise CampaignRefusal(
                "campaign_row_failure_detail_invalid", detail
            )
        return [
            {
                "row_key": copy.deepcopy(record["row_key"]),
                "status": (
                    "measured_infeasible_trace_capture"
                    if classification == "representative_measured_infeasible"
                    else "budget_blocked"
                ),
                "failure_code": record.get("failure_code"),
                "attempt_ordinal": record.get("attempt_ordinal"),
            }
            for record in records if isinstance(record, dict)
        ]
    if classification == "campaign_budget_blocked":
        return []
    try:
        envelope = json.loads(detail)
    except json.JSONDecodeError:
        return []
    row_key = envelope.get("row_key") if isinstance(envelope, dict) else None
    if not isinstance(row_key, dict):
        return []
    if classification == "retryable" and not terminal_retry:
        return []
    status = {
        "retryable": "retry_exhausted",
        "storage_blocked": "storage_blocked",
        "storage_integrity_blocked": "storage_integrity_blocked",
        "representative_measured_infeasible": "measured_infeasible_trace_capture",
    }.get(classification, "integrity_failed")
    return [{
        "row_key": copy.deepcopy(row_key),
        "status": status,
        "failure_code": code,
        "attempt_ordinal": None,
    }]


def _load_or_complete_retained_owner_roots(
    *, output_root: Path, args: argparse.Namespace,
    namespace_contracts: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Finish only a marker-bound interrupted namespace retention bootstrap."""
    try:
        return chunk_capture.validate_retained_owner_namespaces(
            output_root, contracts=namespace_contracts
        )
    except chunk_capture.WorkloadCaptureRefusal as retained_error:
        original_roots = {
            "plan": Path(args.live_coverage_inventory_root),
            "runnability": Path(args.runnability_root),
            "shrink": Path(args.shrink_root),
        }
        marker_names = {
            role: namespace_contracts[role].get("marker_name")
            for role in namespace_contracts
        }
        try:
            observed_contracts = chunk_capture.capture_owner_namespace_contracts(
                roots=original_roots, marker_names=marker_names
            )
        except chunk_capture.WorkloadCaptureRefusal:
            raise retained_error
        if observed_contracts != namespace_contracts:
            raise CampaignRefusal(
                "campaign_original_owner_namespace_drift", str(output_root)
            ) from retained_error
        chunk_capture.retain_owner_namespaces(
            output_root, contracts=namespace_contracts,
            source_roots=original_roots,
        )
        return chunk_capture.validate_retained_owner_namespaces(
            output_root, contracts=namespace_contracts
        )


def _ordered_unpublished_chunks(
    matrix: dict[str, Any], *, output_root: Path,
    max_new_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Return the bounded attempt frontier in sealed matrix order.

    The bound is applied before any chunk is attempted.  A failed, blocked, or
    retry-exhausted chunk therefore consumes one pilot slot just like a
    successfully published chunk; failures can never make a bounded pilot walk
    farther into the matrix.
    """
    unpublished = [
        chunk for chunk in matrix["chunks"]
        if chunk_capture._output_lstat(
            output_root / "sol_workload_chunks" / chunk["chunk_id"],
            operation="campaign_chunk_frontier_lstat",
        ) is None
    ]
    return (
        unpublished if max_new_chunks is None
        else unpublished[:max_new_chunks]
    )


def _run_campaign_locked(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    max_new_chunks = getattr(args, "max_new_chunks", None)
    if (
        max_new_chunks is not None
        and (type(max_new_chunks) is not int or max_new_chunks < 1)
    ):
        raise CampaignRefusal(
            "campaign_max_new_chunks_invalid", repr(max_new_chunks)
        )
    retained_restart = chunk_capture._output_lstat(
        output_root / chunk_capture.SUPPLEMENTAL_ROOT_MARKER,
        operation="campaign_retained_marker_lstat",
    ) is not None
    retained_campaign_identity: dict[str, Any] | None = None
    if retained_restart:
        owner, _payload, _pin = chunk_capture._load_retained_input_owner(
            output_root
        )
        retained_campaign_identity = owner["campaign_identity"]
        namespace_contracts = retained_campaign_identity.get(
            "owner_namespaces"
        )
        if not isinstance(namespace_contracts, dict):
            raise CampaignRefusal(
                "campaign_retained_owner_namespaces_missing", str(output_root)
            )
        # A root marker can be durable one instruction before all retained
        # owner children.  Complete only that exact sealed bootstrap.  Existing
        # corrupt or extra retained bytes are never repaired.
        retained_roots = _load_or_complete_retained_owner_roots(
            output_root=output_root, args=args,
            namespace_contracts=namespace_contracts,
        )
        identities = retained_campaign_identity.get("input_file_identities")
        if not isinstance(identities, dict):
            raise CampaignRefusal(
                "campaign_retained_inputs_invalid", str(output_root)
            )

        def retained_owner_child(role: str, owner_role: str) -> Path:
            identity = identities.get(role)
            source_name = (
                identity.get("source_name")
                if isinstance(identity, dict) else None
            )
            if (
                not isinstance(source_name, str) or not source_name
                or source_name in {".", ".."} or "/" in source_name
                or "\\" in source_name
            ):
                raise CampaignRefusal(
                    "campaign_retained_inputs_invalid", role
                )
            return retained_roots[owner_role] / source_name

        runnability_root = retained_roots["runnability"]
        shrink_root = retained_roots["shrink"]
        live_inventory_root = retained_roots["plan"]
        matrix_path = retained_owner_child("supplemental_matrix", "plan")
        live_inventory_path = retained_owner_child(
            "live_coverage_inventory", "plan"
        )
        runnability_path = retained_owner_child("runnability", "runnability")
        shrink_path = retained_owner_child("shrink_provenance", "shrink")
    else:
        runnability_root = Path(args.runnability_root)
        shrink_root = Path(args.shrink_root)
        live_inventory_root = Path(args.live_coverage_inventory_root)
        chunk_capture.assert_output_disjoint_from_input_owners(
            output_root,
            input_roots={
                "runnability_root": runnability_root,
                "shrink_root": shrink_root,
                "live_coverage_inventory_root": live_inventory_root,
            },
        )
        matrix_path = chunk_capture.lexical_input_path(args.matrix)
        live_inventory_path = chunk_capture.lexical_input_path(
            args.live_coverage_inventory
        )
        runnability_path = chunk_capture.lexical_input_path(args.runnability)
        shrink_path = chunk_capture.lexical_input_path(args.shrink_provenance)
    canonical_matrix_path = chunk_capture.lexical_input_path(
        args.canonical_matrix
    )
    matrix = chunk_capture._load_object(matrix_path, role="supplemental matrix")
    canonical_matrix = chunk_capture._load_object(
        canonical_matrix_path, role="canonical matrix"
    )
    if not args.runnability or not args.shrink_provenance:
        raise CampaignRefusal(
            "campaign_coverage_inputs_missing",
            "--runnability and --shrink-provenance are required",
        )
    full_inventory_path = chunk_capture.lexical_input_path(
        args.full_coverage_inventory
    )
    if (
        not full_inventory_path.is_absolute()
        or full_inventory_path.is_symlink()
        or not full_inventory_path.is_file()
    ):
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_missing",
            str(full_inventory_path),
        )
    try:
        full_inventory = json.loads(full_inventory_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_invalid", str(exc)
        ) from exc
    if (
        not isinstance(full_inventory, dict)
        or full_inventory_path.read_bytes()
        != coverage_inventory_builder.canonical_json_bytes(full_inventory)
    ):
        raise CampaignRefusal(
            "campaign_full_coverage_inventory_invalid",
            "inventory bytes are not canonical",
        )
    head, clean, dirty = chunk_capture.canonical_capture._git_state()
    if head != matrix.get("source_commit") or not clean:
        raise CampaignRefusal(
            "campaign_source_state_invalid",
            f"HEAD={head}; expected={matrix.get('source_commit')}; dirty={dirty!r}",
        )
    regenerated_full_inventory = _regenerate_full_coverage_inventory(
        repo_root=REPO_ROOT,
    )
    validate_full_coverage_inventory(
        full_inventory, regenerated=regenerated_full_inventory
    )
    representative_mode = chunk_capture.is_representative_execution_matrix(
        matrix
    )
    if representative_mode:
        chunk_capture.validate_representative_execution_matrix(
            matrix, matrix_path=matrix_path,
        )
        live_inventory, live_inventory_summary = (
            chunk_capture.validate_representative_live_coverage_inventory(
                inventory_path=live_inventory_path,
                inventory_root=live_inventory_root,
                matrix=matrix,
            )
        )
    else:
        chunk_capture.validate_supplemental_matrix(
            matrix, matrix_path=matrix_path,
            canonical_matrix_path=canonical_matrix_path,
            authority_path=chunk_capture.lexical_input_path(args.authority),
            runnability_path=runnability_path,
            shrink_path=shrink_path,
            runnability_root=runnability_root,
            shrink_root=shrink_root,
        )
        live_inventory, live_inventory_summary = (
            chunk_capture.validate_live_coverage_inventory(
                inventory_path=live_inventory_path,
                inventory_root=live_inventory_root,
                matrix=matrix, matrix_path=matrix_path,
                runnability_path=runnability_path,
                runnability_root=runnability_root,
                shrink_path=shrink_path,
                shrink_root=shrink_root,
            )
        )
    validate_whole_matrix_closure(matrix)
    authority_path = chunk_capture.lexical_input_path(args.authority)
    pins_path = chunk_capture.lexical_input_path(args.pins)
    aggregate_budget_path = chunk_capture.lexical_input_path(
        args.aggregate_budget
    )
    cell_budget_path = chunk_capture.lexical_input_path(args.cell_budgets)
    registry_path = chunk_capture.lexical_input_path(args.metric_registry)
    external_reduced_sources, _reduced_roles = (
        chunk_capture.validate_external_reduced_sources(
            matrix, shrink_root=shrink_root
        )
    )
    owner_roots = {
        "plan": live_inventory_root,
        "runnability": runnability_root,
        "shrink": shrink_root,
    }
    marker_names = {
        "plan": chunk_capture.live_coverage.PLAN_OWNER_FILENAME,
        "runnability": matrix["runnability_authority"]["owner_marker"][
            "name"
        ],
        "shrink": matrix["reduced_proxy_authority"]["owner_marker"]["name"],
    }
    owner_namespaces = chunk_capture.capture_owner_namespace_contracts(
        roots=owner_roots, marker_names=marker_names
    )
    if (
        retained_campaign_identity is not None
        and retained_campaign_identity.get("owner_namespaces")
        != owner_namespaces
    ):
        raise CampaignRefusal(
            "campaign_retained_owner_namespace_identity_mismatch",
            str(output_root),
        )
    campaign_identity = chunk_capture.supplemental_campaign_identity(
        matrix=matrix, matrix_path=matrix_path,
        canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_matrix_path,
        authority_path=authority_path, pins_path=pins_path,
        aggregate_budget_path=aggregate_budget_path,
        cell_budget_path=cell_budget_path, registry_path=registry_path,
        full_coverage_inventory_path=full_inventory_path,
        live_coverage_inventory_path=live_inventory_path,
        runnability_path=runnability_path, shrink_path=shrink_path,
        external_sources=external_reduced_sources,
        owner_namespaces=owner_namespaces,
    )
    if (
        retained_campaign_identity is not None
        and campaign_identity != retained_campaign_identity
    ):
        raise CampaignRefusal(
            "campaign_retained_identity_mismatch", str(output_root)
        )
    chunk_capture.establish_supplemental_output_root(
        output_root, campaign_identity=campaign_identity
    )
    chunk_capture.retain_campaign_inputs(
        output_root, campaign_identity=campaign_identity,
        sources=chunk_capture._campaign_input_sources(
            matrix_path=matrix_path,
            canonical_matrix_path=canonical_matrix_path,
            authority_path=authority_path, pins_path=pins_path,
            aggregate_budget_path=aggregate_budget_path,
            cell_budget_path=cell_budget_path, registry_path=registry_path,
            full_coverage_inventory_path=full_inventory_path,
            live_coverage_inventory_path=live_inventory_path,
            runnability_path=runnability_path, shrink_path=shrink_path,
            external_sources=external_reduced_sources,
        ),
    )
    if not retained_restart:
        chunk_capture.retain_owner_namespaces(
            output_root, contracts=owner_namespaces,
            source_roots=owner_roots,
        )
    retained_roots = chunk_capture.validate_retained_owner_namespaces(
        output_root, contracts=owner_namespaces
    )
    identities = campaign_identity["input_file_identities"]

    def retained_child(role: str, owner_role: str) -> Path:
        identity = identities[role]
        source_name = identity.get("source_name")
        if not isinstance(source_name, str):
            raise CampaignRefusal("campaign_retained_inputs_invalid", role)
        return retained_roots[owner_role] / source_name

    # From this point onward every owner-bound read uses the retained replica,
    # including the first run.  Original owner deletion therefore cannot alter
    # chunk execution, reconciliation, or restart behavior.
    matrix_path = retained_child("supplemental_matrix", "plan")
    live_inventory_path = retained_child("live_coverage_inventory", "plan")
    runnability_path = retained_child("runnability", "runnability")
    shrink_path = retained_child("shrink_provenance", "shrink")
    live_inventory_root = retained_roots["plan"]
    runnability_root = retained_roots["runnability"]
    shrink_root = retained_roots["shrink"]
    _lock = _orchestrator_lock(output_root)
    _recover_published_completion_events(
        output_root=output_root, matrix=matrix, matrix_path=matrix_path,
        canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_matrix_path,
    )
    latest_snapshot = reconcile_campaign(
        matrix=matrix, matrix_path=matrix_path,
        canonical_matrix=canonical_matrix,
        canonical_matrix_path=canonical_matrix_path,
        output_root=output_root,
        full_coverage_inventory=full_inventory,
        full_coverage_inventory_path=full_inventory_path,
        live_coverage_inventory=live_inventory,
        live_coverage_inventory_summary=live_inventory_summary,
        campaign_identity=campaign_identity,
    )["snapshot_root"]
    for chunk in _ordered_unpublished_chunks(
        matrix, output_root=output_root, max_new_chunks=max_new_chunks,
    ):
        final_dir = output_root / "sol_workload_chunks" / chunk["chunk_id"]
        prior_chunk_event = _latest_chunk_events(
            load_event_chain(output_root)
        ).get(chunk["chunk_id"])
        if prior_chunk_event is not None and prior_chunk_event.get(
            "outcome"
        ) in {"storage_blocked", "storage_integrity_blocked"}:
            if prior_chunk_event.get("storage_state") == _storage_state(
                output_root
            ):
                if prior_chunk_event["outcome"] == "storage_integrity_blocked":
                    raise CampaignRefusal(
                        prior_chunk_event["failure_code"],
                        "storage state is unchanged since the integrity failure",
                    )
                return latest_snapshot
        admission_storage_state = _storage_state(output_root)
        for retry in range(args.max_chunk_attempts):
            try:
                preflight_path = _new_preflight(
                    args, source_commit=matrix["source_commit"],
                    chunk_id=chunk["chunk_id"], retry_ordinal=retry + 1,
                    matrix_path=matrix_path,
                )
            except OSError:
                # Raw OS errors are not sufficient to identify the failed
                # filesystem.  Output mutation boundaries type their own
                # storage failures; an untyped error remains semantic/input.
                raise
            run_args = SimpleNamespace(
                matrix=str(matrix_path), canonical_matrix=str(canonical_matrix_path),
                authority=str(chunk_capture.lexical_input_path(args.authority)),
                pins=str(chunk_capture.lexical_input_path(args.pins)),
                aggregate_budget=str(chunk_capture.lexical_input_path(
                    args.aggregate_budget
                )),
                cell_budgets=str(chunk_capture.lexical_input_path(
                    args.cell_budgets
                )),
                metric_registry=str(chunk_capture.lexical_input_path(
                    args.metric_registry
                )),
                runnability=str(runnability_path),
                runnability_root=str(runnability_root),
                shrink_provenance=str(shrink_path),
                shrink_root=str(shrink_root),
                full_coverage_inventory=str(full_inventory_path),
                live_coverage_inventory=str(live_inventory_path),
                live_coverage_inventory_root=str(live_inventory_root),
                preflight=str(preflight_path), chunk_id=chunk["chunk_id"],
                output_root=str(output_root),
                retained_input_mode=True,
                owner_namespaces=owner_namespaces,
            )
            try:
                result = chunk_capture.run_chunk(run_args)
            except (chunk_capture.WorkloadCaptureRefusal, OSError) as caught:
                if isinstance(caught, OSError):
                    # See above: never infer campaign-volume failure from a
                    # raw errno that could have originated in a repo input.
                    raise
                exc = caught
                classification = classify_capture_failure(exc.code)
                if representative_mode and (
                    exc.code.endswith("_timeout")
                    or exc.code in {
                        "cell_deadline_exhausted",
                        "trace_route_deadline_exhausted",
                        "supplemental_chunk_rows_budget_blocked",
                    }
                ):
                    classification = "representative_measured_infeasible"
                if classification in {
                    "storage_blocked", "storage_integrity_blocked",
                }:
                    # The campaign filesystem has just failed.  Do not append
                    # an event, publish a snapshot, finalize, or perform any
                    # other write to that same medium.  The outer boundary
                    # records a typed receipt on the independent runtime fs.
                    raise exc
                outcome = {
                    "retryable": "chunk_failed",
                    "row_budget_blocked": "budget_blocked",
                    "campaign_budget_blocked": "campaign_budget_blocked",
                    "integrity_blocked": "integrity_blocked",
                    "representative_measured_infeasible": "measured_infeasible",
                }[classification]
                row_outcomes = _failure_row_outcomes(
                    code=exc.code, detail=exc.detail,
                    classification=classification,
                    terminal_retry=(retry + 1 >= args.max_chunk_attempts),
                )
                append_event(output_root, {
                    "chunk_id": chunk["chunk_id"],
                    "cohort": chunk["cohort"],
                    "outcome": outcome,
                    "retry_ordinal": retry + 1,
                    "preflight_owner_relative_path": (
                        preflight_path.relative_to(output_root).as_posix()
                    ),
                    "preflight_sha256": chunk_capture._sha256_output_file(
                        preflight_path, role="published campaign preflight",
                        operation="campaign_failure_preflight_hash",
                    ),
                    "failure_code": exc.code,
                    "failure_detail_sha256": hashlib.sha256(
                        exc.detail.encode("utf-8", errors="surrogateescape")
                    ).hexdigest(),
                    "row_outcomes": row_outcomes,
                    "storage_state": None,
                })
                latest_snapshot = reconcile_campaign(
                    matrix=matrix, matrix_path=matrix_path,
                    canonical_matrix=canonical_matrix,
                    canonical_matrix_path=canonical_matrix_path,
                    output_root=output_root,
                    full_coverage_inventory=full_inventory,
                    full_coverage_inventory_path=full_inventory_path,
                    live_coverage_inventory=live_inventory,
                    live_coverage_inventory_summary=live_inventory_summary,
                    campaign_identity=campaign_identity,
                )["snapshot_root"]
                if outcome == "integrity_blocked":
                    raise CampaignRefusal(exc.code, exc.detail) from exc
                if outcome == "campaign_budget_blocked":
                    return latest_snapshot
                if outcome == "budget_blocked":
                    break
                if outcome == "measured_infeasible":
                    # This matrix is now measured-infeasible at its sealed
                    # scale.  The event binds the exact retained journal
                    # detail by failure_detail_sha256; continuing the same
                    # multi-row chunk cannot publish exact membership.  A
                    # certified proxy requires a rebuilt matrix and fresh root.
                    return latest_snapshot
                if retry + 1 < args.max_chunk_attempts:
                    continue
                break
            done = chunk_capture._load_output_object(
                result / "DONE.json", role="published supplemental DONE",
                operation="campaign_completed_done_read",
            )
            _ensure_completed_chunk_event(
                output_root=output_root, chunk=chunk, done=done
            )
            latest_snapshot = reconcile_campaign(
                matrix=matrix, matrix_path=matrix_path,
                canonical_matrix=canonical_matrix,
                canonical_matrix_path=canonical_matrix_path,
                output_root=output_root,
                full_coverage_inventory=full_inventory,
                full_coverage_inventory_path=full_inventory_path,
                live_coverage_inventory=live_inventory,
                live_coverage_inventory_summary=live_inventory_summary,
                campaign_identity=campaign_identity,
            )["snapshot_root"]
            break
    return latest_snapshot


def _storage_state_for_fault(output_root: Path) -> dict[str, int]:
    """Best-effort state for an external receipt; never writes output_root."""
    state_path = output_root if output_root.exists() else output_root.parent
    try:
        return _storage_state(state_path)
    except OSError:
        return {"available_bytes": 0, "available_inodes": 0}


def run_campaign(args: argparse.Namespace) -> Path:
    """Run under the global lock and one fail-closed storage boundary."""
    output_root = Path(args.output_root)
    chunk_capture.assert_external_output_root_before_mutation(output_root)
    _gpu_lock = _global_gpu_lock()
    _check_storage_fault_receipt(output_root)
    try:
        return _run_campaign_locked(args)
    except (
        CampaignRefusal, chunk_capture.WorkloadCaptureRefusal, OSError,
    ) as exc:
        code = getattr(exc, "code", None)
        if code in {
            chunk_capture.STORAGE_CAPACITY_CODE,
            chunk_capture.STORAGE_INTEGRITY_CODE,
        }:
            storage_refusal = chunk_capture.WorkloadCaptureRefusal(
                code, str(getattr(exc, "detail", exc))
            )
        else:
            # A semantic refusal may retain an OSError as its diagnostic
            # cause (for example EAGAIN on an already-held orchestrator lock,
            # EEXIST for an immutable preflight collision, or ENOENT while
            # reading an input).  Those are not campaign-medium failures and
            # must never poison the external storage receipt.
            storage_refusal = None
        if storage_refusal is None:
            raise
        try:
            receipt = _record_storage_fault(
                output_root=output_root, refusal=storage_refusal,
                storage_state=_storage_state_for_fault(output_root),
                operation="campaign_mutation_or_reconciliation",
            )
        except OSError as receipt_error:
            raise CampaignRefusal(
                "campaign_storage_fault_receipt_write_failed",
                f"original={storage_refusal}; receipt={receipt_error}",
            ) from receipt_error
        raise CampaignRefusal(
            storage_refusal.code,
            f"{storage_refusal.detail}; external_receipt={receipt}",
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", default=str(chunk_capture.DEFAULT_MATRIX))
    parser.add_argument(
        "--canonical-matrix", default=str(chunk_capture.DEFAULT_CANONICAL_MATRIX)
    )
    parser.add_argument("--authority", default=str(chunk_capture.DEFAULT_AUTHORITY))
    parser.add_argument("--pins", default=str(chunk_capture.DEFAULT_PINS))
    parser.add_argument(
        "--aggregate-budget", default=str(chunk_capture.DEFAULT_AGGREGATE_BUDGET)
    )
    parser.add_argument(
        "--cell-budgets", default=str(chunk_capture.DEFAULT_CELL_BUDGETS)
    )
    parser.add_argument(
        "--metric-registry", default=str(chunk_capture.DEFAULT_REGISTRY)
    )
    parser.add_argument("--runnability")
    parser.add_argument("--runnability-root", required=True)
    parser.add_argument("--shrink-provenance")
    parser.add_argument("--shrink-root", required=True)
    parser.add_argument("--full-coverage-inventory", required=True)
    parser.add_argument("--live-coverage-inventory", required=True)
    parser.add_argument("--live-coverage-inventory-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-chunk-attempts", type=int, default=3)
    parser.add_argument(
        "--max-new-chunks", type=int,
        help=(
            "attempt at most this many previously-unpublished chunks in exact "
            "matrix order (failures consume a slot), then return a valid "
            "incomplete checkpoint"
        ),
    )
    args = parser.parse_args(argv)
    if args.max_chunk_attempts < 1:
        parser.error("--max-chunk-attempts must be positive")
    if args.max_new_chunks is not None and args.max_new_chunks < 1:
        parser.error("--max-new-chunks must be positive")
    try:
        snapshot = run_campaign(args)
    except (CampaignRefusal, chunk_capture.WorkloadCaptureRefusal) as exc:
        print(
            f"sol_workload_capture_campaign: REFUSED "
            f"[{getattr(exc, 'code', type(exc).__name__)}] {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"sol_workload_capture_campaign: PROGRESS {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
