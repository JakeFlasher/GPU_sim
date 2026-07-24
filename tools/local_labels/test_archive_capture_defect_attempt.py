from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.local_labels import archive_capture_defect_attempt as archive


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=archive.REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _synthetic_driver() -> bytes:
    # Archive mechanics bind this fixture through the contract hash below; they
    # do not depend on historical evaluator Git object authority.
    return (
        b"else:\n    time_runnable = None\n"
        b'_CRITICAL_NAMES = [\n    "time_runnable",\n]\n'
    )


@pytest.fixture(autouse=True)
def _synthetic_driver_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run
    expected_object = (
        f"{archive.FIXED_ORIGINAL_HEAD}:tools/sol_execbench/src/"
        "sol_execbench/driver/templates/eval_driver.py"
    )
    payload = _synthetic_driver()
    replacement_head = real_run(
        ["git", "rev-parse", "HEAD"],
        cwd=archive.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_ancestry = [
        "git",
        "merge-base",
        "--is-ancestor",
        archive.FIXED_ORIGINAL_HEAD,
        replacement_head,
    ]

    def run(command, *args, **kwargs):
        command_list = list(command)
        if command_list == ["git", "show", expected_object]:
            return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")
        if command_list == expected_ancestry:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[archive.ArchiveRequest, Path]:
    output = tmp_path / "campaign"
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    matrix = inputs / "matrix.json"
    preflight = inputs / "preflight.json"
    matrix_seal = "a" * 64
    _write_json(matrix, {"matrix_seal_sha256": matrix_seal})
    _write_json(preflight, {"kind": "test_preflight"})
    matrix_sha = archive.sha256_file(matrix)
    preflight_sha = archive.sha256_file(preflight)
    base = replace(
        archive.FIXED_CONTRACT,
        matrix_file_sha256=matrix_sha,
        matrix_seal_sha256=matrix_seal,
        preflight_sha256=preflight_sha,
        session_id="0b258641-4877-4ea8-8d76-0bc8680bd006",
        admission_run_id="b" * 64,
        cell_id="sol_execbench__synthetic_capture_defect",
    )
    stage = (
        output / ".staging" / base.original_head / base.chunk_id
        / base.matrix_file_sha256
    )
    journal = stage / "journals" / base.cell_id
    cell = journal / "cells" / base.cell_id
    driver = cell / "sol_stage" / "eval_driver.py"
    driver.parent.mkdir(parents=True)
    driver.write_bytes(_synthetic_driver())
    _write_json(cell / "sol_stage" / "capture_prepare_manifest.json", {})
    _write_json(cell / "sol_stage" / "evidence" / "label-selection.json", {})
    _write_json(cell / "sol_stage" / "evidence" / "label-result.json", {
        "status": "reward_hack_detected",
        "error": {
            "detail": "Eval driver integrity violated: 'time_runnable' has been monkey-patched",
            "exception_type": "RewardHackDetected",
            "type": "reward_hack_detected",
        },
        "outside_roi": {
            "all_completed_before_profiler_start": False,
            "correctness_rounds": 1,
            "correctness_user_fn_calls": 1,
            "warmup_user_fn_calls": 0,
        },
        "roi": {
            "candidate_user_fn_calls": 0,
            "latency_ms": None,
            "profiler_start_calls": 0,
            "profiler_start_status": None,
            "profiler_stop_calls": 0,
            "profiler_stop_status": None,
            "timing_methodology": "cuda_events",
        },
    })
    start = {
        "schema_version": 2,
        "state": "capture_started",
        "identity": {"roster_source": "sol_execbench", "slug": "synthetic"},
        "source_commit": base.original_head,
        "capture_matrix_sha256": base.matrix_file_sha256,
        "matrix_seal_sha256": base.matrix_seal_sha256,
        "capture_preflight_sha256": base.preflight_sha256,
        "capture_preflight_path": str(preflight.resolve()),
        "capture_started_at_utc": "2026-07-13T00:00:00.000000Z",
        "admission_run_id": base.admission_run_id,
        "prior_preparation_failures": [],
        "supersedes": None,
    }
    start["journal_start_seal_sha256"] = archive.canonical_sha256(start)
    _write_json(journal / "journal_start.json", start)
    run_rows = [
        {
            "schema_version": 1,
            "event": "session_started",
            "session_id": base.session_id,
            "chunk_id": base.chunk_id,
            "source_commit": base.original_head,
            "capture_matrix_sha256": base.matrix_file_sha256,
            "matrix_seal_sha256": base.matrix_seal_sha256,
        },
        {
            "schema_version": 1,
            "event": "capture_admission",
            "session_id": base.session_id,
            "chunk_id": base.chunk_id,
            "run_id": base.admission_run_id,
            "preflight_sha256": base.preflight_sha256,
        },
    ]
    run_journal = stage / "chunk_run_journal.jsonl"
    run_journal.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in run_rows
    ))
    journal_rows = archive.content_inventory(journal)
    stage_rows = archive.content_inventory(stage)
    contract = replace(
        base,
        journal_file_count=len(journal_rows),
        journal_total_bytes=sum(row["size_bytes"] for row in journal_rows),
        journal_inventory_sha256=archive.inventory_sha256(journal_rows),
        stage_file_count=len(stage_rows),
        stage_total_bytes=sum(row["size_bytes"] for row in stage_rows),
        stage_inventory_sha256=archive.inventory_sha256(stage_rows),
        chunk_run_journal_sha256=archive.sha256_file(run_journal),
        journal_start_sha256=archive.sha256_file(journal / "journal_start.json"),
        label_result_sha256=archive.sha256_file(
            cell / "sol_stage" / "evidence" / "label-result.json"
        ),
        eval_driver_sha256=archive.sha256_file(driver),
        prepare_manifest_sha256=archive.sha256_file(
            cell / "sol_stage" / "capture_prepare_manifest.json"
        ),
        selection_sha256=archive.sha256_file(
            cell / "sol_stage" / "evidence" / "label-selection.json"
        ),
    )
    request = archive.ArchiveRequest(
        output_root=output,
        matrix_path=matrix,
        preflight_path=preflight,
        replacement_head=_head(),
        contract=contract,
        created_at_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
        entry_id="f1bd571a-33f8-4596-b177-8f6af83bc679",
    )
    return request, stage


def _premove_scratch(
    request: archive.ArchiveRequest, stage: Path, *, state: str = "full",
) -> Path:
    path = (
        request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
        / f"capture-defect-{archive._stable_entry_id(request)}"
    )
    path.parent.mkdir(parents=True)
    marker = path.parent / f"{path.name}{archive.CAPTURE_DEFECT_OWNER_SUFFIX}"
    _write_json(
        marker,
        archive._owner_marker_payload(
            request, entry_id=archive._stable_entry_id(request),
        ),
    )
    context = path / "chunk_context"
    context.mkdir(parents=True)
    sources = [
        (stage / "chunk_run_journal.jsonl", "chunk_run_journal.jsonl"),
        (request.matrix_path, "capture_matrix.json"),
        (request.preflight_path, "capture_preflight.json"),
    ]
    count = {"empty": 0, "one": 1, "partial": 2, "full": 3}[state]
    for index, (source, name) in enumerate(sources[:count]):
        shutil.copyfile(source, context / name)
        if state == "partial" and index == count - 1:
            raw = (context / name).read_bytes()
            (context / name).write_bytes(raw[:max(1, len(raw) // 2)])
    return path


def test_archive_moves_once_and_closes_non_authoritative_ledger(tmp_path: Path):
    request, stage = _fixture(tmp_path)
    row = archive.archive_capture_defect_attempt(request)
    assert not (stage / "journals").exists()
    assert row["authoritative"] is False
    assert row["reusable"] is False
    assert row["campaign_rows"] == 0
    ledger = (
        request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
        / archive.CAPTURE_DEFECT_LEDGER_NAME
    )
    rows = archive.load_capture_defect_ledger(
        ledger, verify_archives=True, expected_contract=request.contract,
    )
    assert rows == [row]
    assert archive.capture_defect_backrefs(
        rows, cell_id=request.contract.cell_id, ledger_path=ledger,
    ) == [{
        "ledger_entry_id": row["entry_id"],
        "ledger_path": str(ledger.resolve()),
        "archive_path": row["archive_path"],
        "original_head": request.contract.original_head,
        "reason_code": archive.REASON_CODE,
    }]
    assert archive.unresolved_foreign_journals(
        output_root=request.output_root,
        current_head=request.replacement_head,
        ledger_rows=rows,
    ) == []


def test_inventory_tamper_refuses_before_any_move(tmp_path: Path):
    request, stage = _fixture(tmp_path)
    label = next((stage / "journals").rglob("label-result.json"))
    label.write_text("{}\n")
    with pytest.raises(
        archive.CaptureDefectRefusal, match="CAPTURE_DEFECT_INVENTORY_MISMATCH"
    ):
        archive.archive_capture_defect_attempt(request)
    assert (stage / "journals").is_dir()
    defect_root = request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
    assert not defect_root.exists()


def test_missing_driver_git_authority_refuses_before_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, stage = _fixture(tmp_path)
    authority_run = subprocess.run

    def missing_driver(command, *args, **kwargs):
        if list(command)[:2] == ["git", "show"]:
            return subprocess.CompletedProcess(command, 128, stdout=b"", stderr=b"missing")
        return authority_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", missing_driver)
    with pytest.raises(
        archive.CaptureDefectRefusal,
        match="original capture-driver defect no longer re-proves",
    ):
        archive.archive_capture_defect_attempt(request)
    assert (stage / "journals").is_dir()
    assert not (request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME).exists()


def test_missing_ancestry_git_authority_refuses_before_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, stage = _fixture(tmp_path)
    authority_run = subprocess.run

    def missing_ancestry(command, *args, **kwargs):
        if list(command)[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(command, 128, stdout="", stderr="missing")
        return authority_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", missing_ancestry)
    with pytest.raises(
        archive.CaptureDefectRefusal,
        match="replacement HEAD does not descend from original",
    ):
        archive.archive_capture_defect_attempt(request)
    assert (stage / "journals").is_dir()
    assert not (request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME).exists()


def test_pre_rename_failure_removes_only_invocation_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    request, stage = _fixture(tmp_path)
    real_copy = archive._copy_exclusive_fsync
    calls = 0

    def fail_second(source: Path, destination: Path) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise archive.CaptureDefectRefusal("SYNTHETIC", "before rename")
        return real_copy(source, destination)

    monkeypatch.setattr(archive, "_copy_exclusive_fsync", fail_second)
    with pytest.raises(archive.CaptureDefectRefusal, match="SYNTHETIC"):
        archive.archive_capture_defect_attempt(request)
    assert (stage / "journals").is_dir()
    defect_root = request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
    assert defect_root.is_dir()
    assert list(defect_root.iterdir()) == []


@pytest.mark.parametrize("state", ["empty", "one", "partial", "full"])
def test_exact_process_restart_residue_before_rename_is_recovered(
    tmp_path: Path, state: str,
):
    request, stage = _fixture(tmp_path)
    scratch = _premove_scratch(request, stage, state=state)
    source_before = archive.inventory_bytes(
        archive.content_inventory(stage / "journals" / request.contract.cell_id)
    )
    row = archive.archive_capture_defect_attempt(request)
    assert Path(row["archive_path"]) == scratch.resolve()
    assert not (stage / "journals").exists()
    assert archive.inventory_bytes(archive.content_inventory(
        scratch / "journals" / request.contract.cell_id,
    )) == source_before


def test_before_rename_fault_seam_keeps_source_and_cleans_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    request, stage = _fixture(tmp_path)

    def crash(name: str) -> None:
        if name == "before_journals_rename":
            raise RuntimeError(name)

    monkeypatch.setattr(archive, "_fault_point", crash)
    with pytest.raises(RuntimeError, match="before_journals_rename"):
        archive.archive_capture_defect_attempt(request)
    assert (stage / "journals").is_dir()
    defect_root = request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
    assert defect_root.is_dir()
    assert list(defect_root.iterdir()) == []


@pytest.mark.parametrize("kind", ["extra", "symlink", "fifo"])
def test_premove_restart_refuses_nonexact_or_special_scratch(
    tmp_path: Path, kind: str,
):
    request, stage = _fixture(tmp_path)
    scratch = _premove_scratch(request, stage)
    unexpected = scratch / "unexpected"
    if kind == "extra":
        unexpected.write_text("not invocation-owned\n")
    elif kind == "symlink":
        unexpected.symlink_to(scratch / "chunk_context", target_is_directory=True)
    else:
        os.mkfifo(unexpected)
    with pytest.raises(
        archive.CaptureDefectRefusal, match="pre-rename scratch shape mismatch",
    ):
        archive.archive_capture_defect_attempt(request)
    assert (stage / "journals").is_dir()
    assert scratch.is_dir()


def test_post_rename_failure_never_rolls_back_or_deletes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    request, stage = _fixture(tmp_path)
    real_write = archive._write_json_exclusive_fsync

    def fail_inventory(path: Path, payload: dict) -> None:
        if path.name == archive.CAPTURE_DEFECT_INVENTORY_NAME:
            raise archive.CaptureDefectRefusal("SYNTHETIC", "after rename")
        real_write(path, payload)

    monkeypatch.setattr(archive, "_write_json_exclusive_fsync", fail_inventory)
    with pytest.raises(archive.CaptureDefectRefusal, match="SYNTHETIC"):
        archive.archive_capture_defect_attempt(request)
    assert not (stage / "journals").exists()
    defect_root = request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
    archives = [path for path in defect_root.iterdir() if path.is_dir()]
    assert len(archives) == 1
    assert (archives[0] / "journals" / request.contract.cell_id).is_dir()
    assert not (defect_root / archive.CAPTURE_DEFECT_LEDGER_NAME).exists()


def test_unledgered_and_reappeared_foreign_journals_are_both_unresolved(
    tmp_path: Path,
):
    request, stage = _fixture(tmp_path)
    unresolved = archive.unresolved_foreign_journals(
        output_root=request.output_root,
        current_head=request.replacement_head,
        ledger_rows=[],
    )
    assert len(unresolved) == 1
    assert "unledgered foreign journals" in unresolved[0]
    row = archive.archive_capture_defect_attempt(request)
    ledger = Path(row["archive_path"]).parent / archive.CAPTURE_DEFECT_LEDGER_NAME
    rows = archive.load_capture_defect_ledger(
        ledger, verify_archives=True, expected_contract=request.contract,
    )
    source = Path(row["source_journals_path"])
    source.mkdir(parents=True)
    (source / "reused").mkdir()
    unresolved = archive.unresolved_foreign_journals(
        output_root=request.output_root,
        current_head=request.replacement_head,
        ledger_rows=rows,
    )
    assert len(unresolved) == 1
    assert "reappeared after atomic archive" in unresolved[0]


def test_loader_rejects_archive_byte_drift(tmp_path: Path):
    request, _stage = _fixture(tmp_path)
    row = archive.archive_capture_defect_attempt(request)
    archived_driver = next(Path(row["archive_path"]).rglob("eval_driver.py"))
    archived_driver.write_bytes(archived_driver.read_bytes() + b"# drift\n")
    ledger = Path(row["archive_path"]).parent / archive.CAPTURE_DEFECT_LEDGER_NAME
    with pytest.raises(
        archive.CaptureDefectRefusal, match="CAPTURE_DEFECT_LEDGER_INCONSISTENT"
    ):
        archive.load_capture_defect_ledger(
            ledger, verify_archives=True, expected_contract=request.contract,
        )


@pytest.mark.parametrize("field", ["entry_id", "created_at_utc"])
def test_ledger_identity_is_bound_to_archive_and_journal_start(
    tmp_path: Path, field: str,
):
    request, _stage = _fixture(tmp_path)
    row = archive.archive_capture_defect_attempt(request)
    ledger = Path(row["archive_path"]).parent / archive.CAPTURE_DEFECT_LEDGER_NAME
    tampered = dict(row)
    tampered[field] = (
        str(uuid.uuid4()) if field == "entry_id"
        else "2026-07-13T00:00:01.000000Z"
    )
    ledger.write_bytes(archive._ledger_bytes(tampered))
    with pytest.raises(
        archive.CaptureDefectRefusal, match="CAPTURE_DEFECT_LEDGER_INCONSISTENT",
    ):
        archive.load_capture_defect_ledger(
            ledger, verify_archives=True, expected_contract=request.contract,
        )


@pytest.mark.parametrize("fault", [
    "after_journals_rename_before_flag",
    "after_journals_rename",
    "after_inventory_fsync",
    "after_ledger_temp_fsync",
    "after_ledger_publish",
])
def test_exact_retry_recovers_every_post_rename_crash_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
):
    request, stage = _fixture(tmp_path)
    crashed = False

    def crash_once(name: str) -> None:
        nonlocal crashed
        if name == fault and not crashed:
            crashed = True
            raise RuntimeError(f"synthetic crash at {name}")

    monkeypatch.setattr(archive, "_fault_point", crash_once)
    with pytest.raises(RuntimeError, match=fault):
        archive.archive_capture_defect_attempt(request)
    assert not (stage / "journals").exists()
    archive_path = (
        request.output_root / archive.CAPTURE_DEFECT_ROOT_NAME
        / f"capture-defect-{request.entry_id}"
    )
    moved = archive_path / "journals" / request.contract.cell_id
    before = archive.inventory_bytes(archive.content_inventory(moved))

    monkeypatch.setattr(archive, "_fault_point", lambda _name: None)
    row = archive.archive_capture_defect_attempt(request)
    after = archive.inventory_bytes(archive.content_inventory(moved))
    assert after == before
    assert Path(row["archive_path"]) == archive_path.resolve()
    ledger = archive_path.parent / archive.CAPTURE_DEFECT_LEDGER_NAME
    assert archive.load_capture_defect_ledger(
        ledger, verify_archives=True, expected_contract=request.contract,
    ) == [row]
    assert not (
        archive_path.parent / archive.CAPTURE_DEFECT_LEDGER_PENDING_NAME
    ).exists()


def test_strict_ledger_rejects_second_fixed_incident_row(tmp_path: Path):
    request, _stage = _fixture(tmp_path)
    row = archive.archive_capture_defect_attempt(request)
    ledger = Path(row["archive_path"]).parent / archive.CAPTURE_DEFECT_LEDGER_NAME
    duplicate = {**row, "entry_id": str(uuid.uuid4())}
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
    with pytest.raises(
        archive.CaptureDefectRefusal, match="path closure",
    ):
        archive.load_capture_defect_ledger(
            ledger, verify_archives=False, expected_contract=request.contract,
        )


def test_foreign_scan_rejects_symlink_ancestor_even_for_current_head(
    tmp_path: Path,
):
    output = tmp_path / "campaign"
    outside = tmp_path / "outside"
    (outside / "chunk" / "matrix" / "journals").mkdir(parents=True)
    staging = output / ".staging"
    staging.mkdir(parents=True)
    (staging / "current").symlink_to(outside, target_is_directory=True)
    unresolved = archive.unresolved_foreign_journals(
        output_root=output, current_head="current", ledger_rows=[],
    )
    assert len(unresolved) == 1
    assert "invalid HEAD directory" in unresolved[0]
