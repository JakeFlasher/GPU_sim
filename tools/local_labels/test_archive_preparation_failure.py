from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import archive_preparation_failure as ap


SHA = "a" * 64
HEAD = "b" * 40
NEW_HEAD = "c" * 40
SESSION = "fixture-session"
RUN_ID = "d" * 64
CELL = "sol_execbench__fixture_problem"
CHUNK = "sol_execbench-000-bootstrap"
STAMP = "2026-07-10T13:35:00.000000Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_snapshot(root: Path) -> list[tuple[str, str, int, str | None]]:
    """Seal the observable tree so refusal retries are proven non-mutating."""
    if not root.exists():
        return []
    rows: list[tuple[str, str, int, str | None]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            rows.append((relative, "file", path.stat().st_mode, _sha(path)))
        elif path.is_dir():
            rows.append((relative, "directory", path.stat().st_mode, None))
        else:
            rows.append((relative, "other", path.lstat().st_mode, None))
    return rows


def _assert_no_failure_marker(root: Path) -> None:
    assert not any(
        "FAILURE" in path.name.upper()
        for path in root.rglob("*")
    )


def _fixture(
    tmp_path: Path, *, prep_root: Path | None = None,
    suffix: str = "one", created_at: datetime | None = None,
) -> ap.ArchiveRequest:
    root = tmp_path / f"attempt-{suffix}"
    journals = root / "journals"
    journal = journals / CELL
    cell_dir = journal / "cells" / CELL
    stage = cell_dir / "sol_stage"
    logs = cell_dir / "logs"
    stage.mkdir(parents=True)
    logs.mkdir()

    source_definition = tmp_path / "inputs" / "definition.json"
    source_definition.parent.mkdir(exist_ok=True)
    if not source_definition.exists():
        source_definition.write_text('{"name":"fixture"}\n')
    staged_definition = stage / "definition.json"
    staged_definition.write_text('{"name":"fixture","op_type":null}\n')
    staged_module = stage / "benchmark_kernel.so"
    staged_module.write_bytes(b"fixture-module")
    manifest = {
        "manifest_kind": "sol_execbench_capture_preparation",
        "source_inputs": {
            "definition": {
                "path": str(source_definition.resolve()),
                "sha256": _sha(source_definition),
            }
        },
    }
    _write_json(stage / "capture_prepare_manifest.json", manifest)
    (logs / "sol_prepare.stdout.log").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    (logs / "sol_prepare.stderr.log").write_bytes(b"")

    preflight = tmp_path / "inputs" / "preflight.json"
    if not preflight.exists():
        _write_json(preflight, {"fixture": "preflight"})
    preflight_sha = _sha(preflight)
    matrix = tmp_path / "inputs" / f"matrix-{suffix}.json"
    matrix_payload = {"matrix_seal_sha256": SHA, "fixture": suffix}
    _write_json(matrix, matrix_payload)
    matrix_sha = _sha(matrix)
    start = {
        "schema_version": 2,
        "state": "capture_started",
        "identity": {"roster_source": "sol_execbench", "slug": "fixture_problem"},
        "authority_cell_sha256": "e" * 64,
        "source_commit": HEAD,
        "capture_matrix_sha256": matrix_sha,
        "matrix_seal_sha256": SHA,
        "capture_preflight_sha256": preflight_sha,
        "capture_preflight_path": str(preflight.resolve()),
        "capture_started_at_utc": STAMP,
        "admission_run_id": RUN_ID,
        "supersedes": None,
    }
    start["journal_start_seal_sha256"] = ap.canonical_sha256(start)
    _write_json(journal / "journal_start.json", start)

    # The fixture preserves the same non-vacuous 22-file shape as production.
    for index in range(16):
        path = stage / "support" / f"file-{index:02}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"support-{index}\n")
    rows = ap.content_inventory(journal)
    assert len(rows) == 22

    run_journal = root / "chunk_run_journal.jsonl"
    run_rows = [
        {
            "schema_version": 1,
            "event": "session_started",
            "session_id": SESSION,
            "chunk_id": CHUNK,
            "source_commit": HEAD,
            "capture_matrix_sha256": matrix_sha,
            "matrix_seal_sha256": SHA,
            "timestamp_utc": STAMP,
        },
        {
            "schema_version": 1,
            "event": "capture_admission",
            "session_id": SESSION,
            "chunk_id": CHUNK,
            "run_id": RUN_ID,
            "admitted_at_utc": STAMP,
            "preflight_path": str(preflight.resolve()),
            "preflight_sha256": preflight_sha,
            "preflight_generated_at_utc": STAMP,
        },
    ]
    run_journal.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in run_rows
    ))
    refusal = root / "runner_refusal_record.json"
    _write_json(refusal, {
        "schema_version": 1,
        "runner_refusal_code": "sol_stage_source_mismatch",
        "refused_at_utc": STAMP,
        "phase": "prepare_validate",
    })
    phase_log = root / "phase_log.json"
    _write_json(phase_log, {
        "schema_version": 1,
        "refused_at_utc": STAMP,
        "phases": [
            {"phase": "prepare", "status": "complete"},
            {"phase": "prepare_validate", "status": "refused"},
        ],
    })
    gpu = root / "gpu_process_observation.json"
    _write_json(gpu, {
        "schema_version": 1,
        "observed_at_utc": STAMP,
        "remaining_gpu_processes": [],
    })
    adjudication = tmp_path / "adjudication.md"
    if not adjudication.exists():
        adjudication.write_text("binding fixture adjudication\n")
    packager = tmp_path / "problem_packager.py"
    if not packager.exists():
        packager.write_text(
            "class ProblemPackager:\n"
            "    def execute(self):\n"
            "        self.output.write_text('driver')\n"
            "        return ['python', 'eval_driver.py']\n"
        )
    return ap.ArchiveRequest(
        journals_root=journals,
        preparation_failures_root=prep_root or tmp_path / "preparation_failures",
        original_matrix_path=matrix,
        chunk_run_journal=run_journal,
        runner_refusal_record=refusal,
        phase_log=phase_log,
        gpu_process_observation=gpu,
        adjudication_ref=adjudication,
        rlcr_dir=tmp_path,
        replacement_head=NEW_HEAD,
        problem_packager_source=packager,
        expected_inventory_sha256=ap.source_inventory_sha256(rows),
        expected_file_count=len(rows),
        expected_total_bytes=sum(row["size_bytes"] for row in rows),
        original_head=HEAD,
        original_matrix_file_sha256=matrix_sha,
        original_matrix_seal_sha256=SHA,
        original_preflight_sha256=preflight_sha,
        original_admission_run_id=RUN_ID,
        original_session_id=SESSION,
        rlcr_session_id="fixture-rlcr",
        chunk_id=CHUNK,
        cell_id=CELL,
        pinned_source_definition_sha256=_sha(source_definition),
        staged_definition_sha256=_sha(staged_definition),
        staged_module_sha256=_sha(staged_module),
        session_log_path=tmp_path / "session_log.jsonl",
        created_at_utc=created_at or datetime(
            2026, 7, 10, 14, 0, int(suffix[-1]) if suffix[-1].isdigit() else 0,
            tzinfo=timezone.utc,
        ),
    )


def test_legacy_inventory_domain_is_c_sorted_size_before_hash():
    rows = [
        {"relative_path": "z", "size_bytes": 2, "sha256": "b" * 64},
        {"relative_path": "a", "size_bytes": 1, "sha256": "a" * 64},
    ]
    expected = (
        f"a\t1\t{'a' * 64}\n"
        f"z\t2\t{'b' * 64}\n"
    ).encode()
    assert ap.source_inventory_bytes(rows) == expected
    assert ap.source_inventory_sha256(rows) == hashlib.sha256(expected).hexdigest()


def test_archive_happy_path_is_atomic_closed_and_schema_valid(tmp_path):
    request = _fixture(tmp_path)
    source_context = request.chunk_run_journal.read_bytes()
    entry = ap.archive_preparation_failure(request)

    assert not request.journals_root.exists()
    archive = Path(entry["archive_path"])
    assert archive.parent == request.preparation_failures_root.resolve()
    assert (archive / "chunk_context/chunk_run_journal.jsonl").read_bytes() == source_context
    assert request.chunk_run_journal.read_bytes() == source_context
    inventory = ap.load_archive_inventory(Path(entry["inventory_file"]))
    assert inventory["source_file_count"] == 22
    assert inventory["source_inventory_sha256"] == request.expected_inventory_sha256
    assert inventory["source_total_bytes"] == request.expected_total_bytes
    ap.verify_preparation_archive(archive, inventory)

    ledger_path = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    rows = ap.load_preparation_failure_ledger(ledger_path, verify_archives=True)
    assert rows == [entry]
    assert uuid.UUID(entry["entry_id"]).version == 4
    assert entry["reason_code"] == "STAGE_SEMANTIC_VALIDATION_FAILED"
    assert entry["runner_refusal_code"] == "sol_stage_source_mismatch"
    assert entry["operator_flag"] == "--archive-preparation-failure"
    assert entry["no_gpu_attestation"]["forbidden_artifact_scan"]["matches"] == []
    assert ap.preparation_failure_backrefs(
        rows, cell_id=CELL, ledger_path=ledger_path,
    ) == [{
        "ledger_entry_id": entry["entry_id"],
        "ledger_path": str(ledger_path.resolve()),
        "archive_path": entry["archive_path"],
    }]
    schema = json.loads(ap.PREPARATION_FAILURE_ENTRY_SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(entry)


def test_no_gpu_nested_evidence_fields_are_closed_in_runtime_and_schema(
    tmp_path
):
    request = _fixture(tmp_path)
    entry = ap.archive_preparation_failure(request)
    entry["no_gpu_attestation"]["journal_state"]["unsealed_extra"] = True
    ledger = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.validate_preparation_failure_entry(
            entry, ledger_path=ledger, line_number=1, verify_archive=False,
        )
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"

    schema = json.loads(ap.PREPARATION_FAILURE_ENTRY_SCHEMA.read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(entry)


def test_atomic_archive_move_never_replaces_a_raced_destination(
    tmp_path, monkeypatch
):
    request = _fixture(tmp_path)
    original_rename = ap._rename_noreplace

    def raced_rename(source, destination):
        destination.mkdir()
        (destination / "raced-evidence").write_text("preserve me\n")
        original_rename(source, destination)

    monkeypatch.setattr(ap, "_rename_noreplace", raced_rename)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)

    assert caught.value.code == "PREPARATION_ARCHIVE_SOURCE_MISSING"
    assert request.journals_root.is_dir()
    raced = next(
        child for child in request.preparation_failures_root.iterdir()
        if child.is_dir()
    )
    assert (raced / "raced-evidence").read_text() == "preserve me\n"


def test_archive_cli_lock_serializes_capture_and_peer_archives(tmp_path):
    first = ap._acquire_campaign_lock(tmp_path)
    try:
        with pytest.raises(ap.PreparationArchiveRefusal) as caught:
            ap._acquire_campaign_lock(tmp_path)
        assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    finally:
        first.close()

    resumed = ap._acquire_campaign_lock(tmp_path)
    resumed.close()


@pytest.mark.parametrize(
    "relative_path",
    [
        "cells/x/sol_stage/evidence/label-result.json",
        "cells/x/profile.ncu-rep",
        "cells/x/nvbit/dynamic_trace.pb",
        "cells/x/control_capture.json",
        "cells/x/accepted_launch_record.json",
    ],
)
def test_forbidden_execution_artifacts_refuse_before_move(tmp_path, relative_path):
    request = _fixture(tmp_path)
    forbidden = request.journals_root / CELL / relative_path
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("forbidden\n")
    rows = ap.content_inventory(request.journals_root / CELL)
    request = replace(
        request,
        expected_inventory_sha256=ap.source_inventory_sha256(rows),
        expected_file_count=len(rows),
        expected_total_bytes=sum(row["size_bytes"] for row in rows),
    )
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_NOT_PREGPU"
    assert request.journals_root.is_dir()


def test_inventory_pin_mismatch_refuses_without_move(tmp_path):
    request = replace(_fixture(tmp_path), expected_inventory_sha256="f" * 64)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_INVENTORY_MISMATCH"
    assert request.journals_root.is_dir()


def test_explicit_archive_flag_is_required_before_any_mutation(capsys):
    rc = ap.main([
        "--journals-root", "/does/not/exist/journals",
        "--preparation-failures-root", "/does/not/exist/preparation_failures",
        "--original-matrix", "/does/not/exist/matrix.json",
        "--chunk-run-journal", "/does/not/exist/run.jsonl",
        "--runner-refusal-record", "/does/not/exist/refusal.json",
        "--phase-log", "/does/not/exist/phases.json",
        "--gpu-process-observation", "/does/not/exist/gpu.json",
        "--replacement-head", NEW_HEAD,
    ])
    assert rc == 1
    assert "PREPARATION_ARCHIVE_FLAG_REQUIRED" in capsys.readouterr().err


def test_post_prepare_phase_refuses_before_move(tmp_path):
    request = _fixture(tmp_path)
    phase = json.loads(request.phase_log.read_text())
    phase["phases"].append({"phase": "label", "status": "started"})
    _write_json(request.phase_log, phase)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_NOT_PREGPU"
    assert request.journals_root.is_dir()


def test_nonempty_prepare_stderr_refuses_before_move(tmp_path):
    request = _fixture(tmp_path)
    stderr = (
        request.journals_root / CELL / "cells" / CELL / "logs"
        / "sol_prepare.stderr.log"
    )
    stderr.write_text("unexpected execution output\n")
    rows = ap.content_inventory(request.journals_root / CELL)
    request = replace(
        request,
        expected_inventory_sha256=ap.source_inventory_sha256(rows),
        expected_total_bytes=sum(row["size_bytes"] for row in rows),
    )
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_NOT_PREGPU"
    assert request.journals_root.is_dir()


def test_cross_device_refuses_without_move(tmp_path, monkeypatch):
    request = _fixture(tmp_path)
    monkeypatch.setattr(ap, "_same_device", lambda _a, _b: False)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_CROSS_DEVICE"
    assert request.journals_root.is_dir()


def test_post_move_inventory_mismatch_appends_no_ledger(tmp_path, monkeypatch):
    request = _fixture(tmp_path)
    original = ap.content_inventory
    calls = 0

    def drift_on_second(root):
        nonlocal calls
        calls += 1
        rows = original(root)
        if calls == 2:
            rows[0] = {**rows[0], "sha256": "0" * 64}
        return rows

    monkeypatch.setattr(ap, "content_inventory", drift_on_second)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_INVENTORY_MISMATCH"
    ledger = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    assert not ledger.exists()
    assert not request.journals_root.exists()
    archives = [
        path for path in request.preparation_failures_root.iterdir()
        if path.is_dir()
    ]
    assert len(archives) == 1
    archive = archives[0]
    assert [path.name for path in archive.iterdir()] == [CELL]
    assert not (archive / ap.PREPARATION_FAILURE_INVENTORY_NAME).exists()
    assert not (archive / "chunk_context").exists()
    _assert_no_failure_marker(request.preparation_failures_root)

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert "archive evidence exists without ledger" in caught.value.detail

    before_retry = _tree_snapshot(request.preparation_failures_root)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_SOURCE_MISSING"
    assert _tree_snapshot(request.preparation_failures_root) == before_retry


@pytest.mark.parametrize(
    "failure_site",
    [
        "post_move_content_inventory",
        "inventory_json_write",
        "copy_context",
        "verify_archive",
    ],
)
def test_fs_pre_failures_are_structural_unledgered_and_fail_closed(
    tmp_path, monkeypatch, failure_site,
):
    request = _fixture(tmp_path)
    renamed = False
    original_rename = ap._rename_noreplace

    def tracked_rename(source, destination):
        nonlocal renamed
        original_rename(source, destination)
        renamed = True

    monkeypatch.setattr(ap, "_rename_noreplace", tracked_rename)
    if failure_site == "post_move_content_inventory":
        original_inventory = ap.content_inventory

        def post_move_drift(root):
            rows = original_inventory(root)
            if renamed:
                rows[0] = {**rows[0], "sha256": "0" * 64}
            return rows

        monkeypatch.setattr(ap, "content_inventory", post_move_drift)
    elif failure_site == "inventory_json_write":
        def refuse_inventory_write(*_args, **_kwargs):
            raise ap.PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
                "injected inventory.json write failure",
            )

        monkeypatch.setattr(ap, "_write_json_exclusive_fsync", refuse_inventory_write)
    elif failure_site == "copy_context":
        def refuse_context_copy(*_args, **_kwargs):
            raise ap.PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
                "injected chunk context copy failure",
            )

        monkeypatch.setattr(ap, "_copy_context", refuse_context_copy)
    else:
        def refuse_archive_verification(*_args, **_kwargs):
            raise ap.PreparationArchiveRefusal(
                "PREPARATION_ARCHIVE_INVENTORY_MISMATCH",
                "injected archive verification failure",
            )

        monkeypatch.setattr(
            ap, "verify_preparation_archive", refuse_archive_verification,
        )

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(request)
    assert caught.value.code == "PREPARATION_ARCHIVE_INVENTORY_MISMATCH"
    assert renamed
    assert not request.journals_root.exists()
    ledger = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    assert not ledger.exists()
    assert not request.session_log_path.exists()
    _assert_no_failure_marker(request.preparation_failures_root)

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert "archive evidence exists without ledger" in caught.value.detail


def test_archive_module_defines_no_post_move_marker_or_refusal_code():
    source = Path(ap.__file__).read_text(encoding="utf-8")
    assert "POST_MOVE_FAILURE" not in source
    assert "PREP_ARCHIVE_POST_MOVE" not in source


def test_cli_wraps_raw_post_move_oserror_without_creating_marker(
    tmp_path, monkeypatch, capsys,
):
    request = _fixture(tmp_path)
    monkeypatch.setattr(
        ap, "ROUND28_JOURNALS_ROOT", request.journals_root.resolve(),
    )
    monkeypatch.setattr(
        ap, "ROUND28_PREPARATION_FAILURES_ROOT",
        request.preparation_failures_root.resolve(),
    )
    monkeypatch.setattr(ap, "_current_head_clean", lambda: (NEW_HEAD, True))
    lock = (tmp_path / "fixture.capture.lock").open("w+")
    monkeypatch.setattr(ap, "_acquire_campaign_lock", lambda _root: lock)

    def refuse_with_raw_oserror(_request):
        raise OSError("injected post-move write failure")

    monkeypatch.setattr(ap, "archive_preparation_failure", refuse_with_raw_oserror)
    rc = ap.main([
        "--archive-preparation-failure",
        "--journals-root", str(request.journals_root),
        "--preparation-failures-root", str(request.preparation_failures_root),
        "--original-matrix", str(request.original_matrix_path),
        "--chunk-run-journal", str(request.chunk_run_journal),
        "--runner-refusal-record", str(request.runner_refusal_record),
        "--phase-log", str(request.phase_log),
        "--gpu-process-observation", str(request.gpu_process_observation),
        "--replacement-head", NEW_HEAD,
    ])

    assert rc == 1
    assert (
        "archive_preparation_failure: REFUSED "
        "[PREPARATION_ARCHIVE_SOURCE_MISSING] "
        "OSError: injected post-move write failure"
    ) in capsys.readouterr().err
    assert not list(tmp_path.rglob("POST_MOVE_FAILURE.json"))


def test_third_preparation_archive_requires_distinct_override(tmp_path):
    prep_root = tmp_path / "preparation_failures"
    first = _fixture(tmp_path, prep_root=prep_root, suffix="1")
    second = _fixture(tmp_path, prep_root=prep_root, suffix="2")
    third = _fixture(tmp_path, prep_root=prep_root, suffix="3")
    ap.archive_preparation_failure(first)
    ap.archive_preparation_failure(second)
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.archive_preparation_failure(third)
    assert caught.value.code == "PREPARATION_ATTEMPT_LIMIT"
    assert third.journals_root.is_dir()
    entry = ap.archive_preparation_failure(replace(
        third, override_preparation_attempt_limit=True,
    ))
    assert entry["operator_flag"].endswith("--override-preparation-attempt-limit")


def test_ledger_loader_rejects_truncation_and_archive_drift(tmp_path):
    request = _fixture(tmp_path)
    entry = ap.archive_preparation_failure(request)
    ledger = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    ledger.write_text(ledger.read_text() + "{\n")
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    ledger.write_text(json.dumps(entry, sort_keys=True) + "\n")
    Path(entry["archive_path"], "chunk_context", "phase_log.json").write_text("{}\n")
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"


def test_ledger_loader_rejects_complete_final_record_without_lf(tmp_path):
    request = _fixture(tmp_path)
    entry = ap.archive_preparation_failure(request)
    ledger = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    ledger.write_bytes(json.dumps(entry, sort_keys=True).encode("utf-8"))

    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger)

    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
    assert caught.value.detail == f"{ledger}:1: missing final LF"


def test_bidirectional_closure_rejects_archive_without_ledger(tmp_path):
    root = tmp_path / "preparation_failures"
    (root / "unledgered-attempt").mkdir(parents=True)
    ledger = root / ap.PREPARATION_FAILURE_LEDGER_NAME
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"


def test_non_vacuous_ledger_loader_rejects_existing_empty_file(tmp_path):
    root = tmp_path / "preparation_failures"
    root.mkdir()
    ledger = root / ap.PREPARATION_FAILURE_LEDGER_NAME
    ledger.write_bytes(b"")
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"


def test_bidirectional_closure_rejects_extra_archive_directory(tmp_path):
    request = _fixture(tmp_path)
    ap.archive_preparation_failure(request)
    ledger = request.preparation_failures_root / ap.PREPARATION_FAILURE_LEDGER_NAME
    (request.preparation_failures_root / "unledgered-attempt").mkdir()
    with pytest.raises(ap.PreparationArchiveRefusal) as caught:
        ap.load_preparation_failure_ledger(ledger, verify_archives=True)
    assert caught.value.code == "PREPARATION_LEDGER_INCONSISTENT"
