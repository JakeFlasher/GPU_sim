"""Adversarial tests for the source-preserving supplemental SOL archive."""

from __future__ import annotations

import fcntl
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tools.local_labels import archive_sol_workload_capture_campaign as archive
from tools.local_labels import sol_workload_capture_campaign as campaign
from tools.local_labels import test_sol_workload_capture_campaign as campaign_fixture


HEAD = "a" * 40
H8 = HEAD[:8]


def _write_json(path: Path, value: dict[str, Any], *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive._pretty_bytes(value))
    path.chmod(mode)


def _sealed(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: archive._canonical_hash(body)}


def _make_case(base: Path, *, source_parent: Path | None = None) -> dict[str, Any]:
    source_parent = source_parent or base
    campaign_parent = source_parent / f"jakeshea-solbench-r28_{H8}"
    source = campaign_parent / "capture"
    source.mkdir(parents=True)
    source.chmod(0o700)
    writer_lock = source / archive.SOURCE_LOCK
    writer_lock.write_text("pid=stopped\n", encoding="utf-8")
    writer_lock.chmod(0o600)
    chunk_lock = source / archive.CHUNK_LOCK
    chunk_lock.write_text("pid=stopped\n", encoding="utf-8")
    chunk_lock.chmod(0o600)
    identity = {
        "source_commit": HEAD,
        "supplemental_matrix_sha256": "b" * 64,
        "input_file_identities": {},
    }
    marker = _sealed({
        "schema_version": 1,
        "root_kind": archive.ROOT_KIND,
        "campaign_identity": identity,
        "campaign_identity_sha256": archive._canonical_hash(identity),
        "created_at_utc": "2026-07-16T00:00:00Z",
    }, "root_identity_sha256")
    _write_json(source / archive.ROOT_MARKER, marker)

    event = _sealed({
        "schema_version": 1,
        "kind": "sol_workload_campaign_event",
        "event_ordinal": 1,
        "prior_event_seal_sha256": None,
        "recorded_at_utc": "2026-07-16T00:00:01Z",
        "outcome": "completed",
    }, "event_seal_sha256")
    _write_json(
        source / "sol_workload_campaign/events/event-000001.json", event
    )
    export = _sealed({
        "schema_version": 1,
        "kind": "sol_workload_campaign_cohort_export",
        "cohort": "native",
        "row_count": 0,
        "rows": [],
    }, "export_seal_sha256")
    export_payload = archive._pretty_bytes(export)
    snapshot_id = "snapshot-000001-" + "c" * 64
    progress = _sealed({
        "schema_version": 1,
        "kind": "sol_workload_campaign_progress",
        "snapshot_id": snapshot_id,
        "root_identity": marker,
        "event_count": 1,
        "event_chain_head": event["event_seal_sha256"],
        "exports": {
            "native": {
                "filename": "full_native_rows.json",
                "sha256": archive._sha256_bytes(export_payload),
                "row_count": 0,
            }
        },
    }, "progress_seal_sha256")
    snapshot = source / "sol_workload_campaign/snapshots" / snapshot_id
    _write_json(snapshot / "full_native_rows.json", export)
    _write_json(snapshot / "PROGRESS.json", progress)

    done = {
        "schema_version": 2,
        "kind": "synthetic_done_for_archive_callback_test",
        "chunk_id": "native-000001",
    }
    _write_json(
        source / "sol_workload_chunks/native-000001/DONE.json", done
    )
    evidence = source / "capture_evidence"
    evidence.mkdir()
    primary = evidence / "trace.bin"
    primary.write_bytes(b"trace-payload\0" * 17)
    primary.chmod(0o440)
    os.link(primary, evidence / "trace.alias.bin")
    failed = source / "failed_attempts/attempt-0001/stderr.txt"
    failed.parent.mkdir(parents=True)
    failed.write_text("typed failure retained\n", encoding="utf-8")

    archive_base = base / "sm86_reference_capture_r27"
    archive_base.mkdir()
    gpu_lock = base / "solbench-sm86-gpu.lock"
    gpu_lock.write_text("", encoding="utf-8")
    gpu_lock.chmod(0o600)
    spec = archive.ArchiveSpec(
        source_root=source,
        archive_base=archive_base,
        source_head=HEAD,
        source_h8=H8,
        campaign_identity_sha256=marker["campaign_identity_sha256"],
        gpu_lock_path=gpu_lock,
        test_only_allow_noncanonical_paths=True,
    )
    return {
        "source": source,
        "archive_base": archive_base,
        "gpu_lock": gpu_lock,
        "marker": marker,
        "spec": spec,
    }


def _portable_verifier(calls: list[Path] | None = None):
    def done_verifier(
        manifest: dict[str, Any], *, final_dir: Path, campaign_root: Path,
        repo_root: Path, require_current_state: bool,
        require_final_location: bool, verify_git_provenance: bool,
    ) -> None:
        assert manifest["schema_version"] == 2
        assert final_dir == (
            campaign_root / "sol_workload_chunks" / manifest["chunk_id"]
        )
        assert repo_root == archive.REPO_ROOT
        assert require_current_state is False
        assert require_final_location is True
        assert verify_git_provenance is False

    def verify(root: Path) -> None:
        if calls is not None:
            calls.append(root)
        archive.verify_portable_campaign(root, done_verifier=done_verifier)

    return verify


def _advance_case_to_second_checkpoint(case: dict[str, Any]) -> None:
    source = case["source"]
    first_event, _payload = archive._load_canonical_object(
        source / "sol_workload_campaign/events/event-000001.json",
        role="fixture event",
    )
    second_event = _sealed({
        "schema_version": 1,
        "kind": "sol_workload_campaign_event",
        "event_ordinal": 2,
        "prior_event_seal_sha256": first_event["event_seal_sha256"],
        "recorded_at_utc": "2026-07-16T00:00:02Z",
        "outcome": "completed",
    }, "event_seal_sha256")
    _write_json(
        source / "sol_workload_campaign/events/event-000002.json",
        second_event,
    )
    _write_json(
        source / "sol_workload_chunks/native-000002/DONE.json",
        {
            "schema_version": 2,
            "kind": "synthetic_done_for_archive_callback_test",
            "chunk_id": "native-000002",
        },
    )
    snapshots = source / "sol_workload_campaign/snapshots"
    first_snapshot = next(iter(sorted(snapshots.iterdir())))
    first_progress, _progress_payload = archive._load_canonical_object(
        first_snapshot / "PROGRESS.json", role="fixture progress"
    )
    second_id = "snapshot-000002-" + "e" * 64
    body = {
        key: value for key, value in first_progress.items()
        if key != "progress_seal_sha256"
    }
    body.update({
        "snapshot_id": second_id,
        "event_count": 2,
        "event_chain_head": second_event["event_seal_sha256"],
        "campaign_state": "incomplete",
    })
    second_progress = _sealed(body, "progress_seal_sha256")
    export, _export_payload = archive._load_canonical_object(
        first_snapshot / "full_native_rows.json", role="fixture export"
    )
    second_snapshot = snapshots / second_id
    _write_json(second_snapshot / "full_native_rows.json", export)
    _write_json(second_snapshot / "PROGRESS.json", second_progress)


def _run(case: dict[str, Any], **kwargs):
    return archive.archive_campaign(
        case["spec"], verifier=kwargs.pop("verifier", _portable_verifier()),
        gpu_probe=kwargs.pop("gpu_probe", lambda: []), **kwargs,
    )


def _make_real_offline_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Relocate the runner's retained-byte fixture into archive path shape."""
    fixture = campaign_fixture._materialize_offline_checkpoint_fixture(
        tmp_path / "offline-fixture", monkeypatch
    )
    marker, _payload = archive._load_root_marker(fixture)
    head = marker["campaign_identity"]["source_commit"]
    h8 = head[:8]
    source = tmp_path / f"jakeshea-solbench-r28_{h8}" / "capture"
    shutil.copytree(fixture, source)
    source.chmod(0o700)
    shutil.rmtree(fixture)
    archive_base = tmp_path / "sm86_reference_capture_r27"
    archive_base.mkdir()
    gpu_lock = tmp_path / "solbench-sm86-gpu.lock"
    gpu_lock.write_bytes(b"")
    gpu_lock.chmod(0o600)
    spec = archive.ArchiveSpec(
        source_root=source,
        archive_base=archive_base,
        source_head=head,
        source_h8=h8,
        campaign_identity_sha256=marker["campaign_identity_sha256"],
        gpu_lock_path=gpu_lock,
        test_only_allow_noncanonical_paths=True,
    )
    return {
        "source": source,
        "archive_base": archive_base,
        "gpu_lock": gpu_lock,
        "marker": marker,
        "spec": spec,
        "head": head,
    }


def _revision_root(case: dict[str, Any]) -> Path:
    return case["archive_base"] / archive.ARCHIVE_NAMESPACE / HEAD


def _partials(case: dict[str, Any]) -> list[Path]:
    revision = _revision_root(case)
    return sorted(revision.glob(".*.partial")) if revision.is_dir() else []


def _checkpoint_for_case(case: dict[str, Any]) -> str:
    _portable_verifier()(case["source"])
    rows = archive._inventory(case["source"])
    checkpoint, _inventory_sha = archive._checkpoint_id(
        case["spec"], rows=rows,
        frontier=archive._frontier_binding(case["source"]),
    )
    return checkpoint


def test_positive_copy_is_source_preserving_idempotent_and_hardlink_exact(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    before = archive._inventory(case["source"])
    calls: list[Path] = []
    status, manifest = _run(case, verifier=_portable_verifier(calls))
    assert status == "archived"
    assert archive._inventory(case["source"]) == before
    final = Path(manifest["archive_root"])
    copied = final / "campaign/capture_evidence"
    assert (copied / "trace.bin").stat().st_ino == (
        copied / "trace.alias.bin"
    ).stat().st_ino
    assert stat.S_IMODE((copied / "trace.bin").stat().st_mode) == 0o440
    archived_marker = final / "campaign" / archive.ROOT_MARKER
    assert stat.S_IMODE(archived_marker.stat().st_mode) == 0o444
    assert archived_marker.stat().st_nlink == 1
    assert (final / "campaign/failed_attempts/attempt-0001/stderr.txt").is_file()
    checked = archive.check_archive(
        case["archive_base"], HEAD, manifest["checkpoint_id"],
        verifier=_portable_verifier(calls),
    )
    assert checked == manifest
    retry_status, retry_manifest = _run(
        case, verifier=_portable_verifier(calls)
    )
    assert retry_status == "already_archived"
    assert retry_manifest == manifest
    assert archive._inventory(case["source"]) == before
    assert case["source"] in calls
    assert final / "campaign" in calls


def test_real_offline_reconciliation_survives_archive_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_real_offline_case(tmp_path, monkeypatch)
    before = archive._inventory(case["source"])
    status, manifest = archive.archive_campaign(
        case["spec"], gpu_probe=lambda: []
    )
    assert status == "archived"
    assert archive._inventory(case["source"]) == before
    archived_campaign = Path(manifest["archive_root"]) / "campaign"
    campaign.verify_campaign_checkpoint_offline(archived_campaign)
    checked = archive.check_archive(
        case["archive_base"], case["head"], manifest["checkpoint_id"]
    )
    assert checked == manifest


def test_real_offline_reconciliation_rejects_self_sealed_forged_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_real_offline_case(tmp_path, monkeypatch)
    snapshot = next(
        (case["source"] / campaign.SNAPSHOT_DIR).iterdir()
    )
    progress_path = snapshot / "PROGRESS.json"
    progress, _payload = archive._load_canonical_object(
        progress_path, role="real offline fixture progress"
    )
    unsigned = {
        key: value for key, value in progress.items()
        if key != "progress_seal_sha256"
    }
    assert unsigned["campaign_state"] == "incomplete"
    unsigned["campaign_state"] = "fully_admitted"
    progress_path.chmod(0o600)
    _write_json(
        progress_path, _sealed(unsigned, "progress_seal_sha256")
    )
    before = archive._inventory(case["source"])
    with pytest.raises(
        archive.ArchiveError,
        match="offline whole-campaign reconciliation failed",
    ):
        archive.archive_campaign(case["spec"], gpu_probe=lambda: [])
    assert archive._inventory(case["source"]) == before
    revision = (
        case["archive_base"] / archive.ARCHIVE_NAMESPACE / case["head"]
    )
    assert not list(revision.glob(".*.partial"))


def test_final_byte_tamper_is_refused(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    _status, manifest = _run(case)
    target = Path(manifest["archive_root"]) / "campaign/failed_attempts/attempt-0001/stderr.txt"
    target.chmod(0o600)
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="byte or hardlink topology"):
        archive.check_archive(
            case["archive_base"], HEAD, manifest["checkpoint_id"],
            verifier=_portable_verifier(),
        )


def test_final_check_directly_binds_manifest_to_actual_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _status, manifest = _run(case)
    contradictory = dict(manifest["checkpoint_frontier"])
    contradictory["event_count"] += 1
    monkeypatch.setattr(
        archive, "_frontier_binding", lambda _campaign: contradictory
    )
    with pytest.raises(archive.ArchiveError, match="frontier contradicts"):
        archive.check_archive(
            case["archive_base"], HEAD, manifest["checkpoint_id"],
            verifier=_portable_verifier(),
        )


def test_same_head_pilot_and_resumed_checkpoint_coexist_immutably(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    first_status, first = _run(case)
    assert first_status == "archived"
    first_root = Path(first["archive_root"])
    first_inventory = archive._inventory(first_root / "campaign")
    _advance_case_to_second_checkpoint(case)
    second_status, second = _run(case)
    assert second_status == "archived"
    assert first["checkpoint_id"] != second["checkpoint_id"]
    assert Path(first["archive_revision_namespace"]) == Path(
        second["archive_revision_namespace"]
    )
    assert first_root.is_dir() and Path(second["archive_root"]).is_dir()
    assert archive._inventory(first_root / "campaign") == first_inventory
    assert archive.check_archive(
        case["archive_base"], HEAD, first["checkpoint_id"],
        verifier=_portable_verifier(),
    ) == first
    assert archive.check_archive(
        case["archive_base"], HEAD, second["checkpoint_id"],
        verifier=_portable_verifier(),
    ) == second
    retry_status, retry = _run(case)
    assert retry_status == "already_archived"
    assert retry == second


def test_source_symlink_is_refused_before_partial_creation(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    os.symlink("failed_attempts", case["source"] / "alias")
    with pytest.raises(archive.ArchiveError, match="symlink is forbidden"):
        _run(case)
    assert not _partials(case)


def test_source_special_file_is_refused_before_partial_creation(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    os.mkfifo(case["source"] / "capture.pipe")
    with pytest.raises(archive.ArchiveError, match="special file is forbidden"):
        _run(case)
    assert not _partials(case)


def test_source_hardlink_escaping_root_is_refused(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    outside = tmp_path / "outside-hardlink"
    os.link(case["source"] / "capture_evidence/trace.bin", outside)
    with pytest.raises(archive.ArchiveError, match="hard link escapes campaign root"):
        _run(case)
    assert outside.read_bytes() == (
        case["source"] / "capture_evidence/trace.bin"
    ).read_bytes()


def test_active_campaign_writer_lock_is_refused(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    descriptor = os.open(case["source"] / archive.SOURCE_LOCK, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(archive.ArchiveError, match="campaign writer lock is active"):
            _run(case)
    finally:
        os.close(descriptor)


def test_active_gpu_lock_is_refused(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    descriptor = os.open(case["gpu_lock"], os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(archive.ArchiveError, match="GPU campaign lock is active"):
            _run(case)
    finally:
        os.close(descriptor)


def test_active_chunk_writer_lock_is_refused(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    descriptor = os.open(case["source"] / archive.CHUNK_LOCK, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(archive.ArchiveError, match="chunk writer lock is active"):
            _run(case)
    finally:
        os.close(descriptor)


def test_unresolved_external_storage_fault_refuses_before_archive_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    campaign._record_storage_fault(
        output_root=case["source"],
        refusal=campaign.chunk_capture.WorkloadCaptureRefusal(
            campaign.chunk_capture.STORAGE_INTEGRITY_CODE,
            "injected archive gate",
        ),
        storage_state={"available_bytes": 1, "available_inodes": 1},
        operation="campaign_mutation_or_reconciliation",
    )
    with pytest.raises(
        archive.ArchiveError, match="unresolved external storage-fault receipt",
    ):
        archive.archive_campaign(
            case["spec"], verifier=_portable_verifier(), gpu_probe=lambda: []
        )
    assert not (
        case["archive_base"] / archive.ARCHIVE_NAMESPACE
    ).exists()


def test_storage_receipt_race_is_refused_inside_both_source_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    real_lock = archive._exclusive_existing_lock
    injected = False

    @contextmanager
    def racing_lock(path: Path, role: str):
        nonlocal injected
        with real_lock(path, role):
            if role == "chunk writer lock" and not injected:
                injected = True
                campaign._record_storage_fault(
                    output_root=case["source"],
                    refusal=campaign.chunk_capture.WorkloadCaptureRefusal(
                        campaign.chunk_capture.STORAGE_INTEGRITY_CODE,
                        "injected after early archive receipt check",
                    ),
                    storage_state={
                        "available_bytes": 1, "available_inodes": 1,
                    },
                    operation="campaign_mutation_or_reconciliation",
                )
            yield

    monkeypatch.setattr(archive, "_exclusive_existing_lock", racing_lock)
    with pytest.raises(
        archive.ArchiveError, match="unresolved external storage-fault receipt",
    ):
        archive.archive_campaign(
            case["spec"], verifier=_portable_verifier(), gpu_probe=lambda: []
        )
    assert injected is True
    assert not (
        case["archive_base"] / archive.ARCHIVE_NAMESPACE
    ).exists()


def test_active_gpu_process_probe_is_refused_without_destination_mutation(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(archive.ArchiveError, match="GPU compute processes are active"):
        _run(case, gpu_probe=lambda: [99123])
    assert not (case["archive_base"] / archive.ARCHIVE_NAMESPACE).exists()


def test_insufficient_destination_capacity_refuses_before_partial_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    observed = iter((10 * 1024 ** 3, archive.CAPACITY_RESERVE_BYTES - 1))
    monkeypatch.setattr(archive, "_available_bytes", lambda _path: next(observed))
    with pytest.raises(archive.ArchiveError, match="insufficient durable archive capacity"):
        _run(case)
    assert not _partials(case)


def test_complete_partial_prefix_resumes_without_recopying_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    original = archive._copy_regular_file
    copied: list[Path] = []

    def interrupt(source: Path, destination: Path, row: dict[str, Any]) -> None:
        original(source, destination, row)
        copied.append(destination)
        raise RuntimeError("simulated power loss after fsynced file")

    monkeypatch.setattr(archive, "_copy_regular_file", interrupt)
    with pytest.raises(RuntimeError, match="simulated power loss"):
        _run(case)
    partial = _partials(case)[0]
    assert partial.is_dir()
    assert copied[0].is_file()
    first_inode = copied[0].stat().st_ino
    monkeypatch.setattr(archive, "_copy_regular_file", original)
    status, manifest = _run(case)
    assert status == "archived"
    final_first = (
        Path(manifest["archive_root"]) / copied[0].relative_to(partial)
    )
    assert final_first.stat().st_ino == first_inode


def test_corrupt_partial_is_preserved_and_never_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    original = archive._copy_regular_file
    copied: list[Path] = []

    def interrupt(source: Path, destination: Path, row: dict[str, Any]) -> None:
        original(source, destination, row)
        copied.append(destination)
        raise RuntimeError("interrupt")

    monkeypatch.setattr(archive, "_copy_regular_file", interrupt)
    with pytest.raises(RuntimeError, match="interrupt"):
        _run(case)
    monkeypatch.setattr(archive, "_copy_regular_file", original)
    copied[0].chmod(0o600)
    copied[0].write_bytes(b"corrupt partial")
    corrupt = copied[0].read_bytes()
    with pytest.raises(archive.ArchiveError, match="partial archive file mismatch"):
        _run(case)
    assert copied[0].read_bytes() == corrupt
    assert case["source"].is_dir()


def test_partial_extra_entry_collision_is_preserved_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    original = archive._copy_regular_file

    def interrupt(source: Path, destination: Path, row: dict[str, Any]) -> None:
        original(source, destination, row)
        raise RuntimeError("interrupt")

    monkeypatch.setattr(archive, "_copy_regular_file", interrupt)
    with pytest.raises(RuntimeError):
        _run(case)
    monkeypatch.setattr(archive, "_copy_regular_file", original)
    partial = _partials(case)[0]
    extra = partial / "unexpected"
    extra.write_text("do not delete\n", encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="top-level namespace is corrupt"):
        _run(case)
    assert extra.read_text(encoding="utf-8") == "do not delete\n"


def test_existing_final_collision_is_never_overwritten(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    namespace = case["archive_base"] / archive.ARCHIVE_NAMESPACE
    namespace.mkdir(mode=0o700)
    revision = namespace / HEAD
    revision.mkdir(mode=0o700)
    final = revision / _checkpoint_for_case(case)
    final.mkdir(mode=0o700)
    sentinel = final / "sentinel"
    sentinel.write_text("owned by another transaction\n", encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="namespace is not closed"):
        _run(case)
    assert sentinel.read_text(encoding="utf-8") == "owned by another transaction\n"


def test_progress_export_tamper_fails_portable_offline_verification(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    snapshot_root = next(
        (case["source"] / "sol_workload_campaign/snapshots").iterdir()
    )
    export = snapshot_root / "full_native_rows.json"
    export.chmod(0o600)
    export.write_text("{}\n", encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="export hash/count mismatch"):
        _portable_verifier()(case["source"])


def test_multiple_self_sealed_frontier_snapshots_are_refused(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    snapshots = case["source"] / "sol_workload_campaign/snapshots"
    original = next(snapshots.iterdir())
    progress, _payload = archive._load_canonical_object(
        original / "PROGRESS.json", role="fixture progress"
    )
    body = {
        key: value for key, value in progress.items()
        if key != "progress_seal_sha256"
    }
    body["snapshot_id"] = "snapshot-000001-" + "d" * 64
    forged = _sealed(body, "progress_seal_sha256")
    duplicate = snapshots / body["snapshot_id"]
    _write_json(duplicate / "PROGRESS.json", forged)
    export, _export_payload = archive._load_canonical_object(
        original / "full_native_rows.json", role="fixture export"
    )
    _write_json(duplicate / "full_native_rows.json", export)
    with pytest.raises(archive.ArchiveError, match="multiple progress snapshots"):
        _portable_verifier()(case["source"])


def test_failed_whole_campaign_reconciliation_prevents_partial_creation(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)

    def reject(root: Path) -> None:
        _portable_verifier()(root)
        raise archive.ArchiveError("forged self-sealed campaign checkpoint")

    with pytest.raises(archive.ArchiveError, match="forged self-sealed"):
        _run(case, verifier=reject)
    assert not _partials(case)


def test_production_verifier_refuses_when_public_reconciliation_api_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    from tools.local_labels import sol_workload_capture_campaign

    monkeypatch.setattr(
        archive, "verify_portable_campaign", lambda _root, repo_root: {}
    )
    monkeypatch.delattr(
        sol_workload_capture_campaign,
        "verify_campaign_checkpoint_offline",
        raising=False,
    )
    with pytest.raises(archive.ArchiveError, match="reconciliation verifier is unavailable"):
        archive._verify_complete_campaign(
            case["source"], repo_root=archive.REPO_ROOT
        )


def test_cross_device_source_uses_copy_and_preserves_source(tmp_path: Path) -> None:
    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("/dev/shm is unavailable for the cross-device proof")
    with tempfile.TemporaryDirectory(prefix="sol-archive-test-", dir=shm) as raw:
        source_parent = Path(raw)
        if source_parent.stat().st_dev == tmp_path.stat().st_dev:
            pytest.skip("test environment does not expose a second filesystem")
        case = _make_case(tmp_path, source_parent=source_parent)
        before = archive._inventory(case["source"])
        status, manifest = _run(case)
        assert status == "archived"
        assert manifest["source_device"] != manifest["archive_device"]
        assert archive._inventory(case["source"]) == before
        assert Path(manifest["archive_root"]).stat().st_dev == tmp_path.stat().st_dev


def test_h8_and_campaign_identity_are_both_mandatory(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(archive.ArchiveError, match="source_h8"):
        _run(case | {"spec": dataclasses_replace(case["spec"], source_h8="deadbeef")})
    wrong = dataclasses_replace(case["spec"], campaign_identity_sha256="d" * 64)
    with pytest.raises(archive.ArchiveError, match="identity SHA-256 mismatch"):
        _run(case | {"spec": wrong})


def test_production_archive_and_gpu_paths_are_not_caller_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    monkeypatch.setattr(archive, "CANONICAL_LIVE_BASE", tmp_path)
    monkeypatch.setattr(archive, "CANONICAL_ARCHIVE_BASE", case["archive_base"])
    monkeypatch.setattr(archive, "CANONICAL_GPU_LOCK", case["gpu_lock"])
    production = dataclasses_replace(
        case["spec"], test_only_allow_noncanonical_paths=False
    )
    assert archive._validate_spec(production) == production.normalized()
    wrong_gpu = dataclasses_replace(
        production, gpu_lock_path=tmp_path / "caller-selected-gpu.lock"
    )
    (tmp_path / "caller-selected-gpu.lock").write_text("", encoding="utf-8")
    (tmp_path / "caller-selected-gpu.lock").chmod(0o600)
    with pytest.raises(archive.ArchiveError, match="production GPU lock"):
        archive._validate_spec(wrong_gpu)
    wrong_archive_root = tmp_path / "other" / "sm86_reference_capture_r27"
    wrong_archive_root.mkdir(parents=True)
    wrong_archive = dataclasses_replace(
        production, archive_base=wrong_archive_root
    )
    with pytest.raises(archive.ArchiveError, match="production archive base"):
        archive._validate_spec(wrong_archive)


def dataclasses_replace(spec: archive.ArchiveSpec, **changes: Any) -> archive.ArchiveSpec:
    values = {
        field.name: getattr(spec, field.name)
        for field in spec.__dataclass_fields__.values()
    }
    values.update(changes)
    return archive.ArchiveSpec(**values)
