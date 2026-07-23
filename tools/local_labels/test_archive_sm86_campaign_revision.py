from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from tools.local_labels import archive_sm86_campaign_revision as archive


HEAD = "5c3f216b6daa866b26fb9ff4d3957135e1ba0764"
PUBLISHED = ("portable_lcd-001", "sol_execbench-000-bootstrap")
STAGING = (*PUBLISHED, "sol_execbench-001")


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(archive, "_assert_process_quiescence", lambda _spec: None)
    monkeypatch.setattr(archive, "_validate_capture_checkout", lambda _spec: None)
    monkeypatch.setattr(archive, "_verify_done_semantics", lambda _done, _checkout: [])
    monkeypatch.setattr(archive, "_verify_retained_ledgers", lambda _root: None)
    root = tmp_path / "campaign"
    root.mkdir()
    capture_checkout = tmp_path / "capture-checkout"
    capture_checkout.mkdir()
    (root / ".capture.lock").write_text("stopped\n")
    (root / "module_store").mkdir()
    module = root / "module_store" / "module.bin"
    module.write_bytes(b"module-bytes")

    for name in archive.RETAINED_EVIDENCE_DIRS:
        directory = root / name
        directory.mkdir()
        (directory / "evidence.json").write_text(f'{{"kind":"{name}"}}\n')
        ledger_name = (
            "capture_defect_ledger.jsonl"
            if name == "capture_defect_attempts"
            else "preparation_failure_ledger.jsonl"
        )
        (directory / ledger_name).write_text('{"historical":true}\n')

    historical = root / "chunks" / archive.HISTORICAL_CHUNK
    historical.mkdir(parents=True)
    (historical / "DONE.json").write_text('{"historical":true}\n')

    token = f"r28_{HEAD[:8]}"
    inputs = root / "inputs" / token
    inputs.mkdir(parents=True)
    authority_path = inputs / "capture_roster_authority_sm86.json"
    authority_sha = _write_json(authority_path, {
        "schema_version": 1,
        "source_commit": HEAD,
        "cells": [],
    })
    matrix_path = inputs / "sm86_reference_capture_matrix.json"
    matrix_sha = _write_json(matrix_path, {
        "schema_version": 1,
        "source_commit": HEAD,
        "authority": {
            "path": os.fspath(authority_path),
            "sha256": authority_sha,
        },
        "chunks": [
            {"chunk_id": chunk, "cells": []}
            for chunk in (archive.HISTORICAL_CHUNK, *STAGING)
        ],
    })
    (inputs / "accounting_initial.json").write_text('{"accounting":true}\n')

    preflights = (
        capture_checkout / "results" / "nvbit_capture_staging" / token
    )
    preflights.mkdir(parents=True)
    preflight_hashes: dict[str, str] = {}
    for chunk in STAGING:
        path = preflights / f"preflight_{chunk}.json"
        preflight_hashes[chunk] = _write_json(path, {
            "schema_version": 1,
            "git_head": HEAD,
            "capture_matrix_sha256": matrix_sha,
            "chunk_id": chunk,
        })

    for chunk in PUBLISHED:
        destination = root / "chunks" / chunk
        destination.mkdir()
        (destination / "cells").mkdir()
        (destination / "cells" / "evidence.bin").write_bytes(chunk.encode())
        _write_json(destination / "DONE.json", {
            "schema_version": 2,
            "chunk_id": chunk,
            "source_head_at_seal": HEAD,
            "source_commit_pre": HEAD,
            "source_commit_post": HEAD,
            "source_commit_stable": True,
            "capture_matrix_sha256": matrix_sha,
            "roster_authority_sha256": authority_sha,
            "preflight_sha256": preflight_hashes[chunk],
            "input_paths": {
                "capture_matrix": os.fspath(matrix_path),
                "roster_authority": os.fspath(authority_path),
                "preflight": os.fspath(preflights / f"preflight_{chunk}.json"),
            },
        })

    staging = root / ".staging" / HEAD
    for chunk in STAGING:
        matrix_root = staging / chunk / matrix_sha
        matrix_root.mkdir(parents=True)
        (matrix_root / "chunk_run_journal.jsonl").write_text(json.dumps({
            "schema_version": 1,
            "event": "session_started",
            "chunk_id": chunk,
            "source_commit": HEAD,
            "capture_matrix_sha256": matrix_sha,
        }, sort_keys=True) + "\n")
    linked_module = staging / "sol_execbench-001" / matrix_sha / "module.bin"
    os.link(module, linked_module)

    audits = root / "audits"
    audits.mkdir()
    (audits / f"accounting.{token}.initial.json").write_text('{"ok":true}\n')
    (audits / f"run.{token}.log").write_text("stopped cleanly\n")
    (audits / "unrelated.json").write_text('{"retain":true}\n')

    spec = archive.ArchiveSpec(
        campaign_root=root,
        capture_checkout=capture_checkout,
        preflight_root=preflights,
        input_root=inputs,
        source_head=HEAD,
        matrix_sha256=matrix_sha,
        authority_sha256=authority_sha,
        published_chunks=PUBLISHED,
        staging_chunks=STAGING,
    )
    return {
        "root": root,
        "inputs": inputs,
        "preflights": preflights,
        "matrix": matrix_path,
        "module": module,
        "linked_module": linked_module,
        "spec": spec,
    }


def test_dry_run_is_read_only_and_contract_is_exact(case):
    status, manifest = archive.dry_run(case["spec"])
    assert status == "ready"
    assert case["module"].is_file()
    assert (case["root"] / ".staging" / HEAD).is_dir()
    assert not (case["root"] / archive.ARCHIVE_DIR).exists()
    assert manifest["source_head"] == HEAD
    assert manifest["published_chunks"] == sorted(PUBLISHED)
    assert manifest["staging_chunks"] == sorted(STAGING)
    dispositions = {
        row["component_id"]: row["disposition"]
        for row in manifest["components"]
    }
    assert dispositions["module_store"] == "moved"
    assert dispositions["staging"] == "moved"
    assert dispositions["input_package"] == "retained"
    assert dispositions["preflights"] == "retained"
    assert dispositions[f"historical_chunk:{archive.HISTORICAL_CHUNK}"] == "retained"


def test_preflight_root_must_be_checkout_local(case, tmp_path: Path):
    wrong_root = tmp_path / f"r28_{HEAD[:8]}"
    wrong = archive.dataclasses.replace(case["spec"], preflight_root=wrong_root)
    with pytest.raises(archive.ArchiveError, match="preflight_root must be exactly"):
        archive.dry_run(wrong)


def test_archive_moves_only_revision_owned_components_and_is_idempotent(case):
    retained = {
        "inputs": archive._inventory(case["inputs"]),
        "preflights": archive._inventory(case["preflights"]),
        "historical": archive._inventory(
            case["root"] / "chunks" / archive.HISTORICAL_CHUNK
        ),
        "audits": {
            path.name: path.read_bytes()
            for path in (case["root"] / "audits").iterdir()
        },
    }
    status, manifest = archive.archive_campaign(case["spec"])
    assert status == "archived"
    destination = case["root"] / archive.ARCHIVE_DIR / HEAD
    assert not (case["root"] / ".staging" / HEAD).exists()
    assert not case["module"].exists()
    for chunk in PUBLISHED:
        assert not (case["root"] / "chunks" / chunk).exists()
        assert (destination / "chunks" / chunk / "DONE.json").is_file()
    assert (case["root"] / "chunks" / archive.HISTORICAL_CHUNK).is_dir()
    assert archive._inventory(case["inputs"]) == retained["inputs"]
    assert archive._inventory(case["preflights"]) == retained["preflights"]
    assert archive._inventory(
        case["root"] / "chunks" / archive.HISTORICAL_CHUNK
    ) == retained["historical"]
    assert {
        path.name: path.read_bytes()
        for path in (case["root"] / "audits").iterdir()
    } == retained["audits"]
    archived_module = destination / "module_store" / "module.bin"
    archived_link = destination / "staging" / "sol_execbench-001" / case["spec"].matrix_sha256 / "module.bin"
    assert archived_module.stat().st_ino == archived_link.stat().st_ino
    assert archive.check_archive(case["root"], HEAD) == manifest
    retry_status, retry_manifest = archive.archive_campaign(case["spec"])
    assert retry_status == "already_archived"
    assert retry_manifest == manifest


@pytest.mark.parametrize("moved_prefix", range(5))
def test_every_partial_move_prefix_resumes(case, moved_prefix):
    manifest = archive.build_archive_manifest(case["spec"])
    partial = case["root"] / archive.ARCHIVE_DIR / f".{HEAD}.partial"
    partial.mkdir(parents=True)
    (partial / archive.MANIFEST_NAME).write_bytes(archive._canonical_bytes(manifest))
    moved = [
        component for component in manifest["components"]
        if component["disposition"] == "moved"
    ]
    assert len(moved) == 4
    for component in moved[:moved_prefix]:
        source = Path(component["source_path"])
        destination = partial / component["archive_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, destination)
    status, observed = archive.archive_campaign(case["spec"])
    assert status == "archived"
    assert observed == manifest
    assert (case["root"] / archive.ARCHIVE_DIR / HEAD / "module_store").is_dir()


def test_partial_transaction_rejects_source_destination_collision(case):
    manifest = archive.build_archive_manifest(case["spec"])
    partial = case["root"] / archive.ARCHIVE_DIR / f".{HEAD}.partial"
    partial.mkdir(parents=True)
    (partial / archive.MANIFEST_NAME).write_bytes(archive._canonical_bytes(manifest))
    shutil.copytree(case["root"] / "module_store", partial / "module_store")
    with pytest.raises(archive.ArchiveError, match="ambiguous transaction component state"):
        archive.dry_run(case["spec"])


def test_symlink_anywhere_in_moved_evidence_is_refused(case):
    os.symlink("module.bin", case["root"] / "module_store" / "alias.bin")
    with pytest.raises(archive.ArchiveError, match="symlink is forbidden"):
        archive.dry_run(case["spec"])


def test_symlink_in_any_ancestor_is_refused(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "leaf").write_text("x")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(archive.ArchiveError, match="symlink path component"):
        archive._assert_path_chain_no_symlinks(alias / "leaf")


def test_special_file_is_refused(case):
    os.mkfifo(case["root"] / "module_store" / "pipe")
    with pytest.raises(archive.ArchiveError, match="special file is forbidden"):
        archive.dry_run(case["spec"])


def test_cross_device_move_is_refused(case, monkeypatch: pytest.MonkeyPatch):
    module_store = case["root"] / "module_store"
    real_stat = Path.stat

    def fake_stat(path: Path, *args, **kwargs):
        observed = real_stat(path, *args, **kwargs)
        if path == module_store:
            fields = list(observed)
            fields[2] = observed.st_dev + 1
            return os.stat_result(fields)
        return observed

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(archive.ArchiveError, match="cross-device archive move"):
        archive.dry_run(case["spec"])


def test_semantic_done_failure_is_refused(case, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        archive, "_verify_done_semantics", lambda _done, _checkout: ["sealed artifact drift"]
    )
    with pytest.raises(archive.ArchiveError, match="semantic DONE verification failed"):
        archive.dry_run(case["spec"])


def test_retained_ledger_semantic_failure_is_refused(
    case, monkeypatch: pytest.MonkeyPatch,
):
    def fail(_root):
        raise archive.ArchiveError("retained evidence ledger is invalid")

    monkeypatch.setattr(archive, "_verify_retained_ledgers", fail)
    with pytest.raises(archive.ArchiveError, match="retained evidence ledger is invalid"):
        archive.dry_run(case["spec"])


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("preflight_head", "preflight git_head mismatch"),
        ("done_matrix", "has wrong capture_matrix_sha256"),
        ("stage_matrix_namespace", "staging matrix namespace mismatch"),
        ("unexpected_chunk", "published chunk namespace mismatch"),
    ],
)
def test_repository_binding_defects_are_refused(case, mutation, message):
    if mutation == "preflight_head":
        path = next(case["preflights"].iterdir())
        document = json.loads(path.read_text())
        document["git_head"] = "0" * 40
        _write_json(path, document)
    elif mutation == "done_matrix":
        path = case["root"] / "chunks" / PUBLISHED[0] / "DONE.json"
        document = json.loads(path.read_text())
        document["capture_matrix_sha256"] = "0" * 64
        _write_json(path, document)
    elif mutation == "stage_matrix_namespace":
        source = case["root"] / ".staging" / HEAD / STAGING[0] / case["spec"].matrix_sha256
        os.rename(source, source.parent / ("0" * 64))
    else:
        (case["root"] / "chunks" / "foreign-001").mkdir()
    with pytest.raises(archive.ArchiveError, match=message):
        archive.dry_run(case["spec"])


def test_seal_and_retained_inventory_are_checked_after_archive(case):
    archive.archive_campaign(case["spec"])
    destination = case["root"] / archive.ARCHIVE_DIR / HEAD
    manifest_path = destination / archive.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["source_head"] = "0" * 40
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(archive._canonical_bytes(manifest))
    with pytest.raises(archive.ArchiveError, match="manifest seal mismatch"):
        archive.check_archive(case["root"], HEAD)


def test_empty_bootstrap_is_safely_recovered(case):
    bootstrap = (
        case["root"] / archive.ARCHIVE_DIR / f".{HEAD}.bootstrap"
    )
    bootstrap.mkdir(parents=True)
    status, _manifest = archive.archive_campaign(case["spec"])
    assert status == "archived"
    assert not bootstrap.exists()


def test_complete_bootstrap_manifest_is_safely_resumed(case):
    manifest = archive.build_archive_manifest(case["spec"])
    bootstrap = (
        case["root"] / archive.ARCHIVE_DIR / f".{HEAD}.bootstrap"
    )
    bootstrap.mkdir(parents=True)
    (bootstrap / archive.MANIFEST_NAME).write_bytes(archive._canonical_bytes(manifest))
    status, observed = archive.archive_campaign(case["spec"])
    assert status == "archived"
    assert observed == manifest


def test_truncated_bootstrap_manifest_is_rebuilt_from_unmoved_sources(case):
    bootstrap = (
        case["root"] / archive.ARCHIVE_DIR / f".{HEAD}.bootstrap"
    )
    bootstrap.mkdir(parents=True)
    (bootstrap / archive.MANIFEST_NAME).write_bytes(b'{"schema_version":')
    status, manifest = archive.archive_campaign(case["spec"])
    assert status == "archived"
    assert archive.check_archive(case["root"], HEAD) == manifest


def test_atomic_rename_never_replaces_destination(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("source")
    destination.write_text("destination")
    with pytest.raises(archive.ArchiveError, match="destination collision"):
        archive._rename_noreplace(source, destination)
    assert source.read_text() == "source"
    assert destination.read_text() == "destination"


def test_final_archive_allows_new_revision_global_sources_but_not_old_staging(case):
    _status, manifest = archive.archive_campaign(case["spec"])
    (case["root"] / "module_store").mkdir()
    (case["root"] / "module_store" / "new-module.bin").write_bytes(b"new")
    for chunk in PUBLISHED:
        fresh = case["root"] / "chunks" / chunk
        fresh.mkdir()
        (fresh / "new-revision.txt").write_text("new\n")
    assert archive.check_archive(case["root"], HEAD) == manifest
    retry_status, retry_manifest = archive.archive_campaign(case["spec"])
    assert retry_status == "already_archived"
    assert retry_manifest == manifest
    old_staging = case["root"] / ".staging" / HEAD
    old_staging.mkdir(parents=True)
    with pytest.raises(archive.ArchiveError, match="staging tree reappeared"):
        archive.check_archive(case["root"], HEAD)


def test_final_archive_rejects_extra_archived_chunk(case):
    archive.archive_campaign(case["spec"])
    final = case["root"] / archive.ARCHIVE_DIR / HEAD
    (final / "chunks" / "unexpected-001").mkdir()
    with pytest.raises(archive.ArchiveError, match="archive chunks namespace mismatch"):
        archive.check_archive(case["root"], HEAD)


def test_idempotent_retry_rejects_wrong_requested_operands(case):
    archive.archive_campaign(case["spec"])
    wrong = archive.dataclasses.replace(case["spec"], matrix_sha256="0" * 64)
    with pytest.raises(archive.ArchiveError, match="mismatched capture_matrix_sha256"):
        archive.archive_campaign(wrong)


def test_retained_evidence_drift_is_rejected(case):
    archive.archive_campaign(case["spec"])
    (case["inputs"] / "accounting_initial.json").write_text('{"changed":true}\n')
    with pytest.raises(archive.ArchiveError, match="input_package byte inventory drifted"):
        archive.check_archive(case["root"], HEAD)


@pytest.mark.parametrize(
    ("directory", "ledger"),
    (
        ("capture_defect_attempts", "capture_defect_ledger.jsonl"),
        ("preparation_failures", "preparation_failure_ledger.jsonl"),
    ),
)
def test_global_retained_ledgers_allow_append_and_extras_but_bind_history(
    case, directory, ledger,
):
    _status, manifest = archive.archive_campaign(case["spec"])
    root = case["root"] / directory
    ledger_path = root / ledger
    original = ledger_path.read_bytes()
    with ledger_path.open("ab") as stream:
        stream.write(b'{"new":true}\n')
    new_evidence = root / "new-attempt"
    new_evidence.mkdir()
    (new_evidence / "inventory.json").write_text('{"new":true}\n')
    assert archive.check_archive(case["root"], HEAD) == manifest
    ledger_path.write_bytes(b"X" + original[1:] + b'{"new":true}\n')
    with pytest.raises(archive.ArchiveError, match="retained ledger prefix drifted"):
        archive.check_archive(case["root"], HEAD)


def test_added_matching_old_head_audit_is_rejected(case):
    archive.archive_campaign(case["spec"])
    token = f"r28_{HEAD[:8]}"
    (case["root"] / "audits" / f"late.{token}.json").write_text('{}\n')
    with pytest.raises(archive.ArchiveError, match="audit namespace drifted"):
        archive.check_archive(case["root"], HEAD)


def test_active_campaign_lock_is_refused(case):
    lock = (case["root"] / ".capture.lock").open("r+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(archive.ArchiveError, match="campaign writer lock is active"):
            archive.dry_run(case["spec"])
    finally:
        lock.close()


def test_public_check_holds_campaign_lock(case):
    archive.archive_campaign(case["spec"])
    lock = (case["root"] / ".capture.lock").open("r+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(archive.ArchiveError, match="campaign writer lock is active"):
            archive.check_archive(case["root"], HEAD)
    finally:
        lock.close()


def test_special_campaign_lock_file_is_refused_without_opening(case):
    lock = case["root"] / ".capture.lock"
    lock.unlink()
    os.mkfifo(lock)
    with pytest.raises(archive.ArchiveError, match="campaign writer lock is not a regular"):
        archive.dry_run(case["spec"])


def test_wrapped_ncu_process_is_refused(monkeypatch: pytest.MonkeyPatch):
    spec = types.SimpleNamespace(
        campaign_root=Path("/campaign"), capture_checkout=Path("/checkout")
    )
    monkeypatch.setattr(archive, "_ancestor_pids", lambda _pid: {1, 2})
    monkeypatch.setattr(
        archive, "_process_rows",
        lambda: iter([(9001, ("env", "CUDA_VISIBLE_DEVICES=0", "/usr/bin/ncu"))]),
    )
    with pytest.raises(archive.ArchiveError, match="capture processes are not quiescent"):
        archive._assert_process_quiescence(spec)


def test_gpu_compute_pid_and_probe_failure_are_refused(monkeypatch: pytest.MonkeyPatch):
    spec = types.SimpleNamespace(
        campaign_root=Path("/campaign"), capture_checkout=Path("/checkout")
    )
    monkeypatch.setattr(archive, "_ancestor_pids", lambda _pid: {1, 2})
    monkeypatch.setattr(archive, "_process_rows", lambda: iter(()))
    monkeypatch.setattr(
        archive.subprocess, "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="4242\n", stderr=""),
    )
    with pytest.raises(archive.ArchiveError, match="GPU compute processes are active"):
        archive._assert_process_quiescence(spec)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="driver unavailable")

    monkeypatch.setattr(archive.subprocess, "run", fail)
    with pytest.raises(archive.ArchiveError, match="GPU process quiescence probe failed"):
        archive._assert_process_quiescence(spec)


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", os.fspath(checkout), *args), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def test_capture_checkout_requires_exact_head_and_clean_tree(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.name", "Archive Test")
    _git(checkout, "config", "user.email", "archive@example.invalid")
    (checkout / "tracked.txt").write_text("clean\n")
    _git(checkout, "add", "tracked.txt")
    _git(checkout, "commit", "-q", "-m", "fixture")
    head = _git(checkout, "rev-parse", "HEAD")
    spec = types.SimpleNamespace(capture_checkout=checkout, source_head=head)
    archive._validate_capture_checkout(spec)
    wrong = types.SimpleNamespace(capture_checkout=checkout, source_head="0" * 40)
    with pytest.raises(archive.ArchiveError, match="HEAD mismatch"):
        archive._validate_capture_checkout(wrong)
    (checkout / "tracked.txt").write_text("dirty\n")
    with pytest.raises(archive.ArchiveError, match="capture checkout is dirty"):
        archive._validate_capture_checkout(spec)


def test_capture_checkout_git_probes_disable_optional_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    head = "a" * 40
    outputs = iter((f"{checkout}\n", f"{head}\n", ""))

    def fake_run(*args, **kwargs):
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        assert not any(
            key.startswith("GIT_") and key != "GIT_OPTIONAL_LOCKS"
            for key in kwargs["env"]
        )
        return types.SimpleNamespace(stdout=next(outputs), stderr="")

    monkeypatch.setattr(archive.subprocess, "run", fake_run)
    archive._validate_capture_checkout(types.SimpleNamespace(
        capture_checkout=checkout, source_head=head,
    ))


@pytest.mark.parametrize("target", ("matrix", "done", "journal"))
def test_pre_read_symlink_checks_refuse_authority_files(case, target):
    if target == "matrix":
        path = case["matrix"]
    elif target == "done":
        path = case["root"] / "chunks" / PUBLISHED[0] / "DONE.json"
    else:
        path = (
            case["root"] / ".staging" / HEAD / STAGING[0]
            / case["spec"].matrix_sha256 / "chunk_run_journal.jsonl"
        )
    payload = path.read_bytes()
    replacement = path.with_name(path.name + ".payload")
    replacement.write_bytes(payload)
    path.unlink()
    path.symlink_to(replacement.name)
    with pytest.raises(archive.ArchiveError, match="symlink path component"):
        archive.dry_run(case["spec"])


def test_resealed_manifest_cannot_redirect_component(case):
    archive.archive_campaign(case["spec"])
    manifest_path = (
        case["root"] / archive.ARCHIVE_DIR / HEAD / archive.MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text())
    for component in manifest["components"]:
        if component["component_id"] == "module_store":
            component["source_path"] = os.fspath(case["root"] / "other-store")
            break
    del manifest["manifest_seal_sha256"]
    manifest = archive._seal_manifest(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(archive._canonical_bytes(manifest))
    with pytest.raises(archive.ArchiveError, match="component path bindings"):
        archive.check_archive(case["root"], HEAD)


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("input", "input_root formula binding"),
        ("preflight", "preflight_root formula binding"),
        ("coupled_preflight", "DONE preflight path contradicts"),
    ),
)
def test_resealed_manifest_cannot_redirect_paired_retained_roots(
    case, kind, message,
):
    archive.archive_campaign(case["spec"])
    final = case["root"] / archive.ARCHIVE_DIR / HEAD
    manifest_path = final / archive.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if kind == "input":
        redirected = case["root"].parent / "redirected-input"
        shutil.copytree(case["inputs"], redirected)
        manifest["input_root"] = os.fspath(redirected)
        component_id = "input_package"
    elif kind == "preflight":
        redirected = case["root"].parent / "redirected-preflight"
        shutil.copytree(case["preflights"], redirected)
        manifest["preflight_root"] = os.fspath(redirected)
        component_id = "preflights"
    else:
        redirected_checkout = case["root"].parent / "redirected-checkout"
        redirected = (
            redirected_checkout / "results" / "nvbit_capture_staging"
            / f"r28_{HEAD[:8]}"
        )
        redirected.parent.mkdir(parents=True)
        shutil.copytree(case["preflights"], redirected)
        manifest["capture_checkout"] = os.fspath(redirected_checkout)
        manifest["preflight_root"] = os.fspath(redirected)
        component_id = "preflights"
    for component in manifest["components"]:
        if component["component_id"] == component_id:
            component["source_path"] = os.fspath(redirected)
            break
    del manifest["manifest_seal_sha256"]
    manifest = archive._seal_manifest(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(archive._canonical_bytes(manifest))
    with pytest.raises(archive.ArchiveError, match=message):
        archive.check_archive(case["root"], HEAD)


def test_manifest_schema_version_rejects_boolean_true(case):
    archive.archive_campaign(case["spec"])
    manifest_path = (
        case["root"] / archive.ARCHIVE_DIR / HEAD / archive.MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = True
    del manifest["manifest_seal_sha256"]
    manifest = archive._seal_manifest(manifest)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(archive._canonical_bytes(manifest))
    with pytest.raises(archive.ArchiveError, match="unsupported archive manifest schema"):
        archive.check_archive(case["root"], HEAD)


def test_manifest_binds_non_authoritative_zero_row_disposition(case):
    _status, manifest = archive.dry_run(case["spec"])
    assert manifest["authoritative"] is False
    assert manifest["reusable"] is False
    assert manifest["campaign_rows"] == 0


def test_cli_check_needs_no_source_operands(case, capsys):
    archive.archive_campaign(case["spec"])
    assert archive.main([
        "check", "--campaign-root", os.fspath(case["root"]),
        "--source-head", HEAD,
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    assert payload["source_head"] == HEAD
