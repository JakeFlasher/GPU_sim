"""CPU-only safety contracts for supplemental SOL campaign reconciliation."""
from __future__ import annotations

import errno
import json
import hashlib
import os
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from tools.local_labels import sol_workload_capture_campaign as campaign
from tools.local_labels import sol_workload_capture_chunk as chunk
from tools.local_labels import test_sol_workload_capture_matrix as matrix_fixture


@pytest.fixture
def _permit_offline_supplemental_campaign(monkeypatch):
    """Exercise post-gate campaign internals without a production bypass."""
    monkeypatch.setattr(campaign, "_require_disabled_collect_gate", lambda _args: None)


def _write_disabled_gate_matrix(path: Path) -> Path:
    path.write_text(json.dumps({
        "variants": [{
            "slug": "problem_fixture",
            "variant_id": "native",
            "invocation": {"adapter_kind": "sol_execbench"},
        }],
        "rows": [{
            "key": {"slug": "problem_fixture", "variant_id": "native"},
        }],
    }) + "\n")
    return path


def test_supplemental_campaign_gate_refuses_before_any_output_or_gpu_effect(
    tmp_path, monkeypatch,
):
    matrix = _write_disabled_gate_matrix(tmp_path / "matrix.json")
    output = tmp_path / "must-not-exist"

    def unexpected(*_args, **_kwargs):
        pytest.fail("supplemental campaign crossed the live-canary gate")

    monkeypatch.setattr(
        campaign.chunk_capture,
        "assert_external_output_root_before_mutation",
        unexpected,
    )
    monkeypatch.setattr(campaign, "_global_gpu_lock", unexpected)
    monkeypatch.setattr(campaign, "_check_storage_fault_receipt", unexpected)
    monkeypatch.setattr(campaign, "_run_campaign_locked", unexpected)
    monkeypatch.setattr(campaign.subprocess, "run", unexpected)

    with pytest.raises(campaign.CampaignRefusal) as caught:
        campaign.run_campaign(Namespace(
            matrix=str(matrix), output_root=str(output),
        ))
    assert caught.value.code == (
        "sol_supplemental_launch_set_v3_live_canary_required"
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["variants"][0]["invocation"].update(
            adapter_kind="direct"
        ),
        lambda document: document["variants"].append(
            json.loads(json.dumps(document["variants"][0]))
        ),
        lambda document: document["rows"][0]["key"].update(
            variant_id="missing"
        ),
    ],
)
def test_supplemental_campaign_gate_refuses_mixed_or_ambiguous_scope(
    tmp_path, mutate,
):
    matrix = _write_disabled_gate_matrix(tmp_path / "matrix.json")
    document = json.loads(matrix.read_text())
    mutate(document)
    matrix.write_text(json.dumps(document) + "\n")
    with pytest.raises(campaign.CampaignRefusal) as caught:
        campaign.run_campaign(Namespace(
            matrix=str(matrix), output_root=str(tmp_path / "output"),
        ))
    assert caught.value.code == "campaign_supplemental_source_scope_invalid"


def test_supplemental_campaign_gate_preserves_malformed_input_refusal(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text("{malformed\n")
    with pytest.raises(campaign.chunk_capture.WorkloadCaptureRefusal) as caught:
        campaign.run_campaign(Namespace(
            matrix=str(matrix), output_root=str(tmp_path / "output"),
        ))
    assert caught.value.code == "supplemental_input_invalid"


def _materialize_offline_checkpoint_fixture(tmp_path, monkeypatch) -> Path:
    """Create a real retained, honestly-incomplete frontier for archive tests."""
    variants, rows, *_ = matrix_fixture._native_fixture(
        tmp_path / "matrix-fixture", statuses=("PASSED",)
    )
    matrix = matrix_fixture._matrix_document(variants, rows)
    canonical_unsigned = {
        "schema_version": 1, "source_commit": matrix["source_commit"],
        "fixture": "offline-frontier",
    }
    canonical = {
        **canonical_unsigned,
        "matrix_seal_sha256": chunk.canonical_sha256(canonical_unsigned),
    }
    full_inventory = {"fixture": "full-inventory"}
    live_inventory = {
        "source_commit": matrix["source_commit"],
        "counts": {
            "source_workload_rows": 1,
            "source_rows_with_any_schedule": 1,
            "source_rows_unscheduled_in_scope": 0,
        },
        "fixture": "live-inventory",
    }
    source = tmp_path / "checkpoint-sources"
    source.mkdir()
    documents = {
        "supplemental_matrix": (
            source / "matrix.json", chunk.workload_matrix.serialize_matrix(matrix)
        ),
        "canonical_matrix": (
            source / "canonical.json", chunk._canonical_json_bytes(canonical)
        ),
        "full_coverage_inventory": (
            source / "full.json",
            campaign.coverage_inventory_builder.canonical_json_bytes(
                full_inventory
            ),
        ),
        "live_coverage_inventory": (
            source / "live.json",
            chunk.live_coverage.canonical_json_bytes(live_inventory),
        ),
    }
    source_paths = {}
    identities = {}
    for role, (path, payload) in documents.items():
        path.write_bytes(payload)
        source_paths[role] = path
        identities[role] = chunk.input_file_identity(
            path, role=role, source_commit=matrix["source_commit"]
        )
    campaign_identity = {
        "source_commit": matrix["source_commit"],
        "supplemental_matrix_sha256": identities[
            "supplemental_matrix"
        ]["sha256"],
        "supplemental_matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "canonical_matrix_sha256": identities["canonical_matrix"]["sha256"],
        "canonical_matrix_seal_sha256": canonical["matrix_seal_sha256"],
        "input_file_identities": identities,
    }
    root = tmp_path / "checkpoint"
    chunk.establish_supplemental_output_root(
        root, campaign_identity=campaign_identity
    )
    chunk.retain_campaign_inputs(
        root, campaign_identity=campaign_identity, sources=source_paths
    )
    for lock_name in (".capture.lock", ".sol_workload_campaign.lock"):
        lock = root / lock_name
        lock.write_text("fixture lock\n")
        lock.chmod(0o600)

    retained_paths = {
        role: root / identity["retained_owner_relative_path"]
        for role, identity in identities.items()
    }
    full_payload = documents["full_coverage_inventory"][1]
    full_summary = {
        "sha256": hashlib.sha256(full_payload).hexdigest(),
        "problem_count": 235, "workload_row_count": 3957,
        "architecture_excluded_problems": 34,
        "architecture_excluded_rows": 537,
    }
    live_payload = documents["live_coverage_inventory"][1]
    live_summary = {
        "sha256": hashlib.sha256(live_payload).hexdigest(),
        "source_commit": matrix["source_commit"],
        "source_workload_rows": 1, "matrix_schedule_rows": len(rows),
        "source_rows_with_any_schedule": 1,
        "source_rows_unscheduled_in_scope": 0,
    }
    monkeypatch.setattr(
        campaign, "validate_full_coverage_inventory",
        lambda _document: dict(full_summary),
    )
    monkeypatch.setattr(
        campaign, "_offline_live_inventory_summary",
        lambda **_kwargs: dict(live_summary),
    )
    campaign.reconcile_campaign(
        matrix=matrix, matrix_path=retained_paths["supplemental_matrix"],
        canonical_matrix=canonical,
        canonical_matrix_path=retained_paths["canonical_matrix"],
        output_root=root, full_coverage_inventory=full_inventory,
        full_coverage_inventory_path=retained_paths[
            "full_coverage_inventory"
        ],
        live_coverage_inventory=live_inventory,
        live_coverage_inventory_summary=live_summary,
        campaign_identity=campaign_identity,
    )
    shutil.rmtree(source)
    return root


def _owner_namespace_fixture(tmp_path: Path):
    roots: dict[str, Path] = {}
    markers = {
        "plan": ".sol-live-coverage-plan-owner-v1.json",
        "runnability": ".sol-runnability-owner-v1.json",
        "shrink": ".sol-shrink-owner-v1.json",
    }
    children = {
        "plan": ("matrix.json", "live-inventory.json"),
        "runnability": ("runnability.json",),
        "shrink": ("provenance.json", "checkpoint.json", "variant.jsonl"),
    }
    for role in sorted(markers):
        root = tmp_path / f"source-{role}"
        root.mkdir(mode=0o700)
        roots[role] = root
        for name in (markers[role], *children[role]):
            path = root / name
            path.write_bytes(f"{role}:{name}\n".encode())
            path.chmod(0o444)
    contracts = chunk.capture_owner_namespace_contracts(
        roots=roots, marker_names=markers
    )
    args = Namespace(
        live_coverage_inventory_root=str(roots["plan"]),
        runnability_root=str(roots["runnability"]),
        shrink_root=str(roots["shrink"]),
    )
    return roots, contracts, args


def _key(index: int) -> dict[str, str]:
    return {
        "slug": f"problem_{index:03d}_test",
        "source_workload_uuid": f"{index:08x}-1234-5123-8123-123456789abc",
        "variant_id": "native",
    }


def test_row_events_never_project_one_chunk_failure_onto_siblings():
    first, second = _key(1), _key(2)
    events = [{
        "event_ordinal": 1,
        "chunk_id": "sol-workload-native-001",
        "row_outcomes": [{
            "row_key": first,
            "status": "budget_blocked",
            "failure_code": "trace_target_failed_tree_limit_exceeded",
            "attempt_ordinal": 1,
        }],
    }]
    latest = campaign._latest_row_events(events)
    assert campaign._row_identity(first) in latest
    assert campaign._row_identity(second) not in latest


def test_row_failure_detail_is_exact_and_retry_is_terminal_only():
    key = _key(3)
    detail = json.dumps([{
        "row_key": key,
        "failure_code": "trace_target_failed_tree_limit_exceeded",
        "failure_detail": "ceiling",
        "attempt_ordinal": 2,
    }])
    assert campaign._failure_row_outcomes(
        code="supplemental_chunk_rows_budget_blocked", detail=detail,
        classification="row_budget_blocked", terminal_retry=True,
    ) == [{
        "row_key": key,
        "status": "budget_blocked",
        "failure_code": "trace_target_failed_tree_limit_exceeded",
        "attempt_ordinal": 2,
    }]
    retry_detail = json.dumps({"row_key": key, "cause_detail": "timeout"})
    assert campaign._failure_row_outcomes(
        code="label_capture_failed_timeout", detail=retry_detail,
        classification="retryable", terminal_retry=False,
    ) == []
    assert campaign._failure_row_outcomes(
        code="label_capture_failed_timeout", detail=retry_detail,
        classification="retryable", terminal_retry=True,
    )[0]["status"] == "retry_exhausted"


def test_snapshot_publication_is_atomic_idempotent_and_resume_safe(tmp_path):
    parent = tmp_path / "snapshots"
    root = parent / "snapshot-000001-deadbeef"
    documents = {
        "full_native_rows.json": {"rows": [1]},
        "PROGRESS.json": {"state": "incomplete"},
    }
    campaign._publish_snapshot_atomic(
        snapshot_root=root, documents=documents
    )
    first = {
        path.name: path.read_bytes() for path in root.iterdir()
    }
    campaign._publish_snapshot_atomic(
        snapshot_root=root, documents=documents
    )
    assert {path.name: path.read_bytes() for path in root.iterdir()} == first

    resumed = parent / "snapshot-000002-cafebabe"
    pending = parent / f".{resumed.name}.pending"
    pending.mkdir()
    partial = pending / "full_native_rows.json"
    partial.write_bytes(campaign._canonical_pretty_json(documents[
        "full_native_rows.json"
    ]))
    partial.chmod(0o444)
    campaign._publish_snapshot_atomic(
        snapshot_root=resumed, documents=documents
    )
    assert resumed.is_dir() and not pending.exists()
    assert {path.name for path in resumed.iterdir()} == set(documents)


def _rerender_offline_checkpoint(
    root: Path, live_summary: dict,
) -> dict:
    marker = json.loads(
        (root / chunk.SUPPLEMENTAL_ROOT_MARKER).read_text()
    )
    campaign_identity = marker["campaign_identity"]
    identities = campaign_identity["input_file_identities"]

    def retained(role: str) -> Path:
        return root / identities[role]["retained_owner_relative_path"]

    matrix_path = retained("supplemental_matrix")
    canonical_path = retained("canonical_matrix")
    full_path = retained("full_coverage_inventory")
    live_path = retained("live_coverage_inventory")
    return campaign.reconcile_campaign(
        matrix=json.loads(matrix_path.read_text()), matrix_path=matrix_path,
        canonical_matrix=json.loads(canonical_path.read_text()),
        canonical_matrix_path=canonical_path,
        output_root=root, publish=False,
        full_coverage_inventory=json.loads(full_path.read_text()),
        full_coverage_inventory_path=full_path,
        live_coverage_inventory=json.loads(live_path.read_text()),
        live_coverage_inventory_summary=live_summary,
        campaign_identity=campaign_identity, offline=True,
    )


def _checkpoint_live_summary(root: Path) -> dict:
    snapshot = next((root / campaign.SNAPSHOT_DIR).iterdir())
    summary = json.loads((snapshot / "PROGRESS.json").read_text())[
        "live_coverage_inventory"
    ]
    summary.pop("input_role")
    return summary


def test_progress_snapshot_ignores_live_inventory_owner_location(
    tmp_path, monkeypatch,
):
    root = _materialize_offline_checkpoint_fixture(tmp_path, monkeypatch)
    snapshot = next((root / campaign.SNAPSHOT_DIR).iterdir())
    expected = {
        path.name: path.read_bytes() for path in snapshot.iterdir()
    }
    stable_summary = _checkpoint_live_summary(root)

    for label, inode in (("original-owner", 101), ("retained-owner", 202)):
        summary = {
            **stable_summary,
            "path": str(tmp_path / label / "inventory.json"),
            "owner_identity": {
                "canonical_path": str(tmp_path / label),
                "device": 1,
                "inode": inode,
                "mode": 0o700,
                "uid": os.getuid(),
            },
        }
        rendered = _rerender_offline_checkpoint(root, summary)
        observed = {
            name: campaign._canonical_pretty_json(document)
            for name, document in rendered["documents"].items()
        }
        assert observed == expected
        assert "path" not in rendered["manifest"]["live_coverage_inventory"]
        assert (
            "owner_identity"
            not in rendered["manifest"]["live_coverage_inventory"]
        )


def test_progress_snapshot_retains_stable_live_inventory_summary_changes(
    tmp_path, monkeypatch,
):
    root = _materialize_offline_checkpoint_fixture(tmp_path, monkeypatch)
    stable_summary = _checkpoint_live_summary(root)
    baseline = _rerender_offline_checkpoint(root, stable_summary)
    changed_summary = dict(stable_summary)
    changed_summary["source_rows_with_any_schedule"] += 1
    changed = _rerender_offline_checkpoint(root, changed_summary)

    assert baseline["snapshot_root"] == changed["snapshot_root"]
    assert (
        campaign._canonical_pretty_json(baseline["manifest"])
        != campaign._canonical_pretty_json(changed["manifest"])
    )
    assert (
        baseline["manifest"]["live_coverage_inventory"][
            "source_rows_with_any_schedule"
        ]
        != changed["manifest"]["live_coverage_inventory"][
            "source_rows_with_any_schedule"
        ]
    )


def test_snapshot_resume_refuses_corrupt_pending_bytes(tmp_path):
    root = tmp_path / "snapshots" / "snapshot-000001-deadbeef"
    pending = root.parent / f".{root.name}.pending"
    pending.mkdir(parents=True)
    (pending / "PROGRESS.json").write_text("{}\n")
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_snapshot_pending_mismatch",
    ):
        campaign._publish_snapshot_atomic(
            snapshot_root=root,
            documents={"PROGRESS.json": {"state": "incomplete"}},
        )


def test_campaign_storage_usage_does_not_count_its_own_snapshots(tmp_path):
    evidence = tmp_path / "sol_workload_chunks" / "chunk-001"
    evidence.mkdir(parents=True)
    (evidence / "DONE.json").write_bytes(b"retained evidence\n")
    before = campaign._campaign_evidence_storage_usage(tmp_path)

    snapshot = tmp_path / campaign.SNAPSHOT_DIR / "snapshot-000001-test"
    snapshot.mkdir(parents=True)
    (snapshot / "PROGRESS.json").write_bytes(b"first snapshot\n")
    pending = snapshot.parent / ".snapshot-000002-test.pending"
    pending.mkdir()
    (pending / "PROGRESS.json").write_bytes(b"partial snapshot\n")
    assert campaign._campaign_evidence_storage_usage(tmp_path) == before


def test_output_root_cannot_alias_input_owner(tmp_path):
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    with pytest.raises(
        chunk.WorkloadCaptureRefusal,
        match="supplemental_output_input_owner_alias",
    ):
        chunk.assert_output_disjoint_from_input_owners(
            owner / "campaign", input_roots={"shrink_root": owner}
        )
    child = owner / "input"
    child.mkdir(mode=0o700)
    with pytest.raises(
        chunk.WorkloadCaptureRefusal,
        match="supplemental_output_input_owner_alias",
    ):
        chunk.assert_output_disjoint_from_input_owners(
            owner, input_roots={"shrink_root": child}
        )


def test_global_gpu_lock_is_exact_exclusive_and_accepts_inherited_fd(
    tmp_path, monkeypatch,
):
    lock_path = tmp_path / "solbench-sm86-gpu.lock"
    monkeypatch.setattr(campaign, "GLOBAL_GPU_LOCK_PATH", lock_path)
    monkeypatch.delenv(campaign.INHERITED_GPU_LOCK_FD_ENV, raising=False)
    first = campaign._global_gpu_lock()
    try:
        with pytest.raises(
            campaign.CampaignRefusal, match="campaign_global_gpu_lock_active",
        ):
            campaign._global_gpu_lock()
        monkeypatch.setenv(
            campaign.INHERITED_GPU_LOCK_FD_ENV, str(first.fileno())
        )
        inherited = campaign._global_gpu_lock()
        inherited.close()
    finally:
        first.close()

    wrong = tmp_path / "wrong.lock"
    wrong.write_text("")
    wrong.chmod(0o600)
    with wrong.open("r+") as stream:
        monkeypatch.setenv(
            campaign.INHERITED_GPU_LOCK_FD_ENV, str(stream.fileno())
        )
        with pytest.raises(
            campaign.CampaignRefusal, match="campaign_global_gpu_lock_invalid",
        ):
            campaign._global_gpu_lock()


def test_bounded_pilot_counts_only_new_chunks_in_exact_matrix_order(tmp_path):
    chunks = [
        {"chunk_id": f"chunk-{index:03d}", "cohort": "native"}
        for index in range(1, 4)
    ]
    output = tmp_path / "campaign"
    (output / "sol_workload_chunks/chunk-002").mkdir(parents=True)
    assert [
        row["chunk_id"] for row in campaign._ordered_unpublished_chunks(
            {"chunks": chunks}, output_root=output, max_new_chunks=1,
        )
    ] == ["chunk-001"]
    assert [
        row["chunk_id"] for row in campaign._ordered_unpublished_chunks(
            {"chunks": chunks}, output_root=output, max_new_chunks=2,
        )
    ] == ["chunk-001", "chunk-003"]
    assert [
        row["chunk_id"] for row in campaign._ordered_unpublished_chunks(
            {"chunks": chunks}, output_root=output,
        )
    ] == ["chunk-001", "chunk-003"]


def test_bounded_failure_and_resume_never_reach_beyond_selected_frontier(
    tmp_path,
):
    matrix = {
        "chunks": [
            {"chunk_id": f"chunk-{index:03d}", "cohort": "native"}
            for index in range(1, 4)
        ]
    }
    output = tmp_path / "campaign"
    output.mkdir()

    # A failure in this bounded invocation can only occur in chunk-001: the
    # loop is given a one-element list before capture starts.  With no DONE,
    # restart selects the same chunk again rather than silently advancing.
    first_attempt = campaign._ordered_unpublished_chunks(
        matrix, output_root=output, max_new_chunks=1,
    )
    assert [row["chunk_id"] for row in first_attempt] == ["chunk-001"]
    restarted = campaign._ordered_unpublished_chunks(
        matrix, output_root=output, max_new_chunks=1,
    )
    assert [row["chunk_id"] for row in restarted] == ["chunk-001"]

    # Only publication of chunk-001 advances the next bounded invocation.
    (output / "sol_workload_chunks/chunk-001").mkdir(parents=True)
    after_publication = campaign._ordered_unpublished_chunks(
        matrix, output_root=output, max_new_chunks=1,
    )
    assert [row["chunk_id"] for row in after_publication] == ["chunk-002"]


def test_storage_capacity_fault_records_external_receipt_and_gates_restart(
    tmp_path, monkeypatch, _permit_offline_supplemental_campaign,
):
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    monkeypatch.setattr(
        campaign, "GLOBAL_GPU_LOCK_PATH", runtime / "solbench-sm86-gpu.lock"
    )
    state = {"available_bytes": 100, "available_inodes": 20}
    monkeypatch.setattr(campaign, "_storage_state", lambda _path: dict(state))
    calls = []

    def fail_once(_args):
        calls.append("run")
        raise campaign.chunk_capture.WorkloadCaptureRefusal(
            campaign.chunk_capture.STORAGE_CAPACITY_CODE,
            "injected locally typed campaign medium full",
        )

    monkeypatch.setattr(campaign, "_run_campaign_locked", fail_once)
    args = Namespace(output_root=str(output))
    with pytest.raises(
        campaign.CampaignRefusal,
        match=chunk.STORAGE_CAPACITY_CODE,
    ):
        campaign.run_campaign(args)
    receipt = campaign._storage_fault_receipt_path(output)
    document = json.loads(receipt.read_text())
    assert document["failure_code"] == chunk.STORAGE_CAPACITY_CODE
    assert document["storage_state"] == state
    assert calls == ["run"]

    with pytest.raises(
        campaign.CampaignRefusal, match="campaign_storage_state_unchanged",
    ):
        campaign.run_campaign(args)
    assert calls == ["run"]

    state["available_bytes"] += 1
    expected = output / "progress"
    monkeypatch.setattr(
        campaign, "_run_campaign_locked", lambda _args: expected
    )
    assert campaign.run_campaign(args) == expected
    assert not receipt.exists()


def test_storage_integrity_receipt_requires_explicit_reconciliation(
    tmp_path, monkeypatch, _permit_offline_supplemental_campaign,
):
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    monkeypatch.setattr(
        campaign, "GLOBAL_GPU_LOCK_PATH", runtime / "solbench-sm86-gpu.lock"
    )
    state = {"available_bytes": 100, "available_inodes": 20}
    monkeypatch.setattr(campaign, "_storage_state", lambda _path: dict(state))
    calls = []

    def fail_integrity(_args):
        calls.append("run")
        raise campaign.chunk_capture.WorkloadCaptureRefusal(
            campaign.chunk_capture.STORAGE_INTEGRITY_CODE,
            "injected locally typed campaign medium I/O failure",
        )

    monkeypatch.setattr(campaign, "_run_campaign_locked", fail_integrity)
    args = Namespace(output_root=str(output))
    with pytest.raises(
        campaign.CampaignRefusal,
        match=chunk.STORAGE_INTEGRITY_CODE,
    ):
        campaign.run_campaign(args)
    state["available_bytes"] += 10_000
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_storage_integrity_reconciliation_required",
    ):
        campaign.run_campaign(args)
    assert calls == ["run"]


@pytest.mark.parametrize(
    "error_number",
    [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_raw_input_storage_errno_never_creates_output_storage_receipt(
    tmp_path, monkeypatch, error_number, _permit_offline_supplemental_campaign,
):
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    monkeypatch.setattr(campaign, "_global_gpu_lock", lambda: object())

    def fail_input(_args):
        raise OSError(error_number, "injected immutable input read failure")

    monkeypatch.setattr(campaign, "_run_campaign_locked", fail_input)
    with pytest.raises(OSError) as caught:
        campaign.run_campaign(Namespace(output_root=str(output)))
    assert caught.value.errno == error_number
    assert not campaign._storage_fault_receipt_path(output).exists()


def _storage_receipt_fixture(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    monkeypatch.setattr(campaign, "_global_gpu_lock", lambda: object())
    monkeypatch.setattr(
        campaign, "_storage_state",
        lambda _path: {"available_bytes": 10, "available_inodes": 10},
    )
    return output


def _inject_output_open_fault(monkeypatch, target: Path, error_number: int):
    # The campaign loader intentionally supports direct script execution and
    # therefore owns the chunk-module identity used by its production calls.
    # Patch that exact object so the injected errno reaches the campaign
    # output boundary instead of a second package-qualified module instance.
    chunk_module = campaign.chunk_capture
    real_open = chunk_module._open_regular_nofollow

    def fail_open(path, *, role):
        if path == target:
            refusal = chunk_module.WorkloadCaptureRefusal(
                "supplemental_input_path_invalid", f"{role}: injected",
            )
            refusal.__cause__ = OSError(
                error_number, "injected campaign-output read failure",
            )
            raise refusal
        return real_open(path, role=role)

    monkeypatch.setattr(chunk_module, "_open_regular_nofollow", fail_open)


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_event_output_read_storage_fault_creates_receipt(
    tmp_path, monkeypatch, error_number, _permit_offline_supplemental_campaign,
):
    output = _storage_receipt_fixture(tmp_path, monkeypatch)
    campaign.append_event(output, {
        "chunk_id": "chunk-000001", "cohort": "native",
        "outcome": "integrity_blocked", "row_outcomes": [],
    })
    event_path = output / campaign.EVENT_DIR / "event-000001.json"
    _inject_output_open_fault(monkeypatch, event_path, error_number)
    monkeypatch.setattr(
        campaign, "_run_campaign_locked",
        lambda _args: campaign.load_event_chain(output),
    )
    with pytest.raises(campaign.CampaignRefusal, match=chunk.STORAGE_INTEGRITY_CODE):
        campaign.run_campaign(Namespace(output_root=str(output)))
    receipt = campaign._storage_fault_receipt_path(output)
    assert json.loads(receipt.read_text())["failure_code"] == (
        chunk.STORAGE_INTEGRITY_CODE
    )


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_published_done_output_read_storage_fault_creates_receipt(
    tmp_path, monkeypatch, error_number, _permit_offline_supplemental_campaign,
):
    output = _storage_receipt_fixture(tmp_path, monkeypatch)
    done_path = output / "sol_workload_chunks/chunk-001/DONE.json"
    done_path.parent.mkdir(parents=True)
    done_path.write_text("{}\n")
    _inject_output_open_fault(monkeypatch, done_path, error_number)

    def recover(_args):
        campaign._recover_published_completion_events(
            output_root=output,
            matrix={"chunks": [{"chunk_id": "chunk-001", "cohort": "native"}]},
            matrix_path=tmp_path / "matrix.json",
            canonical_matrix={}, canonical_matrix_path=tmp_path / "canonical.json",
        )

    monkeypatch.setattr(campaign, "_run_campaign_locked", recover)
    with pytest.raises(campaign.CampaignRefusal, match=chunk.STORAGE_INTEGRITY_CODE):
        campaign.run_campaign(Namespace(output_root=str(output)))
    receipt = campaign._storage_fault_receipt_path(output)
    assert json.loads(receipt.read_text())["failure_code"] == (
        chunk.STORAGE_INTEGRITY_CODE
    )


@pytest.mark.parametrize(
    "error_number", [errno.EIO, getattr(errno, "ESTALE", errno.EIO)],
)
def test_checkpoint_output_read_storage_fault_creates_receipt(
    tmp_path, monkeypatch, error_number, _permit_offline_supplemental_campaign,
):
    output = _storage_receipt_fixture(tmp_path, monkeypatch)
    snapshot = output / campaign.SNAPSHOT_DIR / "snapshot-000001-test"
    snapshot.mkdir(parents=True)
    document = {"schema_version": 1}
    checkpoint = snapshot / "PROGRESS.json"
    checkpoint.write_bytes(campaign._canonical_pretty_json(document))
    _inject_output_open_fault(monkeypatch, checkpoint, error_number)
    monkeypatch.setattr(
        campaign, "_run_campaign_locked",
        lambda _args: campaign._publish_snapshot_atomic(
            snapshot_root=snapshot, documents={"PROGRESS.json": document},
        ),
    )
    with pytest.raises(campaign.CampaignRefusal, match=chunk.STORAGE_INTEGRITY_CODE):
        campaign.run_campaign(Namespace(output_root=str(output)))
    receipt = campaign._storage_fault_receipt_path(output)
    assert json.loads(receipt.read_text())["failure_code"] == (
        chunk.STORAGE_INTEGRITY_CODE
    )


def test_event_output_fsync_eio_types_locally_and_creates_receipt(
    tmp_path, monkeypatch, _permit_offline_supplemental_campaign,
):
    output = _storage_receipt_fixture(tmp_path, monkeypatch)
    real_fsync = campaign.chunk_capture.os.fsync
    injected = False

    def fail_first_fsync(descriptor):
        nonlocal injected
        if not injected:
            injected = True
            raise OSError(errno.EIO, "injected event fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(campaign.chunk_capture.os, "fsync", fail_first_fsync)
    monkeypatch.setattr(
        campaign, "_run_campaign_locked",
        lambda _args: campaign.append_event(output, {
            "chunk_id": "chunk-000001", "cohort": "native",
            "outcome": "integrity_blocked", "row_outcomes": [],
        }),
    )
    with pytest.raises(
        campaign.CampaignRefusal,
        match=campaign.chunk_capture.STORAGE_INTEGRITY_CODE,
    ):
        campaign.run_campaign(Namespace(output_root=str(output)))
    receipt = campaign._storage_fault_receipt_path(output)
    assert json.loads(receipt.read_text())["failure_code"] == (
        campaign.chunk_capture.STORAGE_INTEGRITY_CODE
    )


def test_snapshot_output_rename_eio_types_locally_and_creates_receipt(
    tmp_path, monkeypatch, _permit_offline_supplemental_campaign,
):
    output = _storage_receipt_fixture(tmp_path, monkeypatch)
    real_replace = campaign.os.replace
    injected = False

    def fail_first_replace(source, destination):
        nonlocal injected
        if not injected:
            injected = True
            raise OSError(errno.EIO, "injected snapshot rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(campaign.os, "replace", fail_first_replace)

    def publish(_args):
        campaign._publish_snapshot_atomic(
            snapshot_root=output / campaign.SNAPSHOT_DIR / "snapshot-000001-test",
            documents={"PROGRESS.json": {"schema_version": 1}},
        )

    monkeypatch.setattr(campaign, "_run_campaign_locked", publish)
    with pytest.raises(
        campaign.CampaignRefusal,
        match=campaign.chunk_capture.STORAGE_INTEGRITY_CODE,
    ):
        campaign.run_campaign(Namespace(output_root=str(output)))
    receipt = campaign._storage_fault_receipt_path(output)
    assert json.loads(receipt.read_text())["failure_code"] == (
        campaign.chunk_capture.STORAGE_INTEGRITY_CODE
    )


@pytest.mark.parametrize(
    ("refusal", "expected_code"),
    [
        (
            campaign.CampaignRefusal(
                "campaign_orchestrator_active", "already locked"
            ),
            "campaign_orchestrator_active",
        ),
        (
            campaign.CampaignRefusal(
                "campaign_preflight_publication_exists", "immutable collision"
            ),
            "campaign_preflight_publication_exists",
        ),
        (
            chunk.WorkloadCaptureRefusal(
                "supplemental_input_path_invalid", "missing input"
            ),
            "supplemental_input_path_invalid",
        ),
    ],
)
def test_semantic_os_backed_refusals_never_create_storage_receipt(
    tmp_path, monkeypatch, refusal, expected_code,
    _permit_offline_supplemental_campaign,
):
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    monkeypatch.setattr(
        campaign, "GLOBAL_GPU_LOCK_PATH", runtime / "solbench-sm86-gpu.lock"
    )
    # Preserve the diagnostic OS cause that previously triggered the false
    # recursive storage classification.
    refusal.__cause__ = BlockingIOError(errno.EAGAIN, "injected semantic cause")

    def fail_semantically(_args):
        raise refusal

    monkeypatch.setattr(campaign, "_run_campaign_locked", fail_semantically)
    with pytest.raises(type(refusal)) as caught:
        campaign.run_campaign(Namespace(output_root=str(output)))
    assert getattr(caught.value, "code") == expected_code
    assert not campaign._storage_fault_receipt_path(output).exists()


def test_bounded_pilot_cli_rejects_nonpositive_limit():
    with pytest.raises(SystemExit):
        campaign.main([
            "--runnability-root", "/tmp/runnability",
            "--shrink-root", "/tmp/shrink",
            "--full-coverage-inventory", "/tmp/full.json",
            "--live-coverage-inventory", "/tmp/live.json",
            "--live-coverage-inventory-root", "/tmp/plan",
            "--output-root", "/tmp/output",
            "--max-new-chunks", "0",
        ])


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("capacity", chunk.STORAGE_CAPACITY_CODE),
        ("integrity", chunk.STORAGE_INTEGRITY_CODE),
    ],
)
def test_preflight_child_storage_marker_maps_before_generic_failure(
    tmp_path, monkeypatch, kind, expected_code,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "campaign"
    output.mkdir()
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "LOCAL_LABELS", repo / "tools/local_labels")
    supplemental_matrix = repo / "supplemental.json"
    supplemental_matrix.write_bytes(b'{"artifact_id":"live-supplemental"}\n')
    marker = chunk.preflight.PreflightStorageFailure(
        kind,
        error_number=(errno.ENOSPC if kind == "capacity" else errno.EIO),
        operation="injection_smoke_tree",
        detail="injected child storage failure",
    ).marker()
    outcomes = iter([
        Namespace(stdout="", stderr="", returncode=0),
        Namespace(stdout="", stderr=marker + "\n", returncode=3),
    ])
    commands = []
    child_matrix_hashes = []
    real_run = campaign.subprocess.run

    def run(command, *_args, **_kwargs):
        commands.append(command)
        if "--matrix" in command:
            matrix_arg = Path(command[command.index("--matrix") + 1])
            child_matrix_hashes.append(
                hashlib.sha256(matrix_arg.read_bytes()).hexdigest()
            )
        return next(outcomes)

    monkeypatch.setattr(
        campaign.subprocess, "run", run,
    )
    args = Namespace(
        output_root=str(output),
        pins=str(repo / "pins.json"),
        aggregate_budget=str(repo / "budget.json"),
        matrix=str(repo / "supplemental.json"),
        canonical_matrix=str(repo / "matrix.json"),
    )
    with pytest.raises(campaign.chunk_capture.WorkloadCaptureRefusal) as caught:
        campaign._new_preflight(
            args, source_commit="a" * 40, chunk_id="chunk-000001",
            retry_ordinal=1, matrix_path=supplemental_matrix,
        )
    assert caught.value.code == expected_code
    assert json.loads(caught.value.detail)["child"]["kind"] == kind
    preflight_command = commands[1]
    matrix_index = preflight_command.index("--matrix") + 1
    assert preflight_command[matrix_index] == str(supplemental_matrix)
    assert str(repo / "matrix.json") not in preflight_command
    assert child_matrix_hashes == [
        hashlib.sha256(supplemental_matrix.read_bytes()).hexdigest()
    ]


def test_retained_restart_preflight_uses_retained_matrix_after_source_deletion(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "campaign"
    output.mkdir()
    original = tmp_path / "original-owner" / "supplemental.json"
    original.parent.mkdir()
    matrix_bytes = b'{"artifact_id":"retained-live-supplemental"}\n'
    original.write_bytes(matrix_bytes)
    retained = output / "retained/plan/supplemental.json"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(matrix_bytes)
    original.unlink()
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "LOCAL_LABELS", repo / "tools/local_labels")
    marker = chunk.preflight.PreflightStorageFailure(
        "integrity", error_number=errno.EIO,
        operation="injection_smoke_tree", detail="stop after command audit",
    ).marker()
    child_observation = {}
    calls = 0

    def run(command, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Namespace(stdout="", stderr="", returncode=0)
        matrix_arg = Path(command[command.index("--matrix") + 1])
        child_observation.update(
            path=matrix_arg,
            sha256=hashlib.sha256(matrix_arg.read_bytes()).hexdigest(),
            command=command,
        )
        return Namespace(stdout="", stderr=marker + "\n", returncode=3)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    args = Namespace(
        output_root=str(output), pins=str(repo / "pins.json"),
        aggregate_budget=str(repo / "budget.json"), matrix=str(original),
        canonical_matrix=str(repo / "canonical.json"),
    )
    with pytest.raises(campaign.chunk_capture.WorkloadCaptureRefusal):
        campaign._new_preflight(
            args, source_commit="a" * 40, chunk_id="chunk-000001",
            retry_ordinal=1, matrix_path=retained,
        )
    assert child_observation["path"] == retained
    assert child_observation["sha256"] == hashlib.sha256(matrix_bytes).hexdigest()
    assert str(original) not in child_observation["command"]
    assert str(args.canonical_matrix) not in child_observation["command"]


def test_repo_local_preflight_manifest_storage_marker_creates_no_campaign_receipt(
    tmp_path, monkeypatch, _permit_offline_supplemental_campaign,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "campaign"
    output.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "LOCAL_LABELS", repo / "tools/local_labels")
    monkeypatch.setattr(campaign, "STORAGE_FAULT_ROOT", runtime)
    monkeypatch.setattr(campaign, "_global_gpu_lock", lambda: object())
    marker = chunk.preflight.PreflightStorageFailure(
        chunk.preflight.PREFLIGHT_STORAGE_INTEGRITY,
        error_number=errno.EIO,
        operation="preflight_manifest_emission",
        detail="injected repo-local staging write failure",
    ).marker()
    outcomes = iter([
        Namespace(stdout="", stderr="", returncode=0),
        Namespace(stdout="", stderr=marker + "\n", returncode=3),
    ])
    real_run = campaign.subprocess.run
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda *_args, **_kwargs: next(outcomes)
    )
    args = Namespace(
        output_root=str(output),
        pins=str(repo / "pins.json"),
        aggregate_budget=str(repo / "budget.json"),
        matrix=str(repo / "supplemental.json"),
        canonical_matrix=str(repo / "matrix.json"),
    )
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_preflight_local_storage_failed",
    ) as caught:
        campaign._new_preflight(
            args, source_commit="a" * 40, chunk_id="chunk-000001",
            retry_ordinal=1, matrix_path=repo / "supplemental.json",
        )
    local_failure = caught.value
    monkeypatch.setattr(campaign.subprocess, "run", real_run)
    monkeypatch.setattr(
        campaign, "_run_campaign_locked",
        lambda _args: (_ for _ in ()).throw(local_failure),
    )
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_preflight_local_storage_failed",
    ):
        campaign.run_campaign(Namespace(output_root=str(output)))
    assert not campaign._storage_fault_receipt_path(output).exists()


@pytest.mark.parametrize("boundary", ["open", "record"])
def test_orchestrator_lock_storage_boundaries_are_typed(
    tmp_path, monkeypatch, boundary,
):
    output = tmp_path / "campaign"
    output.mkdir()
    if boundary == "open":
        real_open = campaign.os.open

        def fail_lock(path, *args, **kwargs):
            if os.fspath(path) == ".sol_workload_campaign.lock":
                raise OSError(errno.ENOSPC, "injected orchestrator lock open")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(campaign.os, "open", fail_lock)
        expected = chunk.STORAGE_CAPACITY_CODE
    else:
        monkeypatch.setattr(
            campaign.os, "fsync",
            lambda _fd: (_ for _ in ()).throw(
                OSError(errno.EIO, "injected orchestrator lock fsync")
            ),
        )
        expected = chunk.STORAGE_INTEGRITY_CODE
    with pytest.raises(campaign.chunk_capture.WorkloadCaptureRefusal) as caught:
        campaign._orchestrator_lock(output)
    assert caught.value.code == expected


def test_orchestrator_lock_primary_fault_wins_over_close_fault(
    tmp_path, monkeypatch,
):
    output = tmp_path / "campaign"
    output.mkdir()
    real_open = campaign.os.open
    real_close = campaign.os.close
    real_fstat = campaign.os.fstat
    lock_descriptor = None

    def track_open(path, *args, **kwargs):
        nonlocal lock_descriptor
        descriptor = real_open(path, *args, **kwargs)
        if os.fspath(path) == ".sol_workload_campaign.lock":
            lock_descriptor = descriptor
        return descriptor

    def fail_primary_stat(descriptor):
        if descriptor == lock_descriptor:
            raise OSError(errno.ENOSPC, "injected primary lock stat fault")
        return real_fstat(descriptor)

    def fail_secondary_close(descriptor):
        if descriptor == lock_descriptor:
            raise OSError(errno.EIO, "injected secondary lock close fault")
        return real_close(descriptor)

    monkeypatch.setattr(campaign.os, "open", track_open)
    monkeypatch.setattr(campaign.os, "fstat", fail_primary_stat)
    monkeypatch.setattr(campaign.os, "close", fail_secondary_close)
    try:
        with pytest.raises(campaign.chunk_capture.WorkloadCaptureRefusal) as caught:
            campaign._orchestrator_lock(output)
        assert caught.value.code == campaign.chunk_capture.STORAGE_CAPACITY_CODE
        assert json.loads(caught.value.detail)["errno"] == errno.ENOSPC
    finally:
        if lock_descriptor is not None:
            real_close(lock_descriptor)


def test_retained_owner_bootstrap_resumes_partial_copy_then_needs_no_sources(
    tmp_path,
):
    roots, contracts, args = _owner_namespace_fixture(tmp_path)
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    partial_root = output / chunk.RETAINED_OWNER_NAMESPACE_ROOT / "plan"
    partial_root.mkdir(parents=True, mode=0o700)
    first = contracts["plan"]["files"][0]
    partial = partial_root / first["name"]
    partial.write_bytes((roots["plan"] / first["name"]).read_bytes())
    partial.chmod(0o444)

    retained = campaign._load_or_complete_retained_owner_roots(
        output_root=output, args=args, namespace_contracts=contracts
    )
    assert chunk.validate_retained_owner_namespaces(
        output, contracts=contracts
    ) == retained
    for root in roots.values():
        shutil.rmtree(root)
    assert campaign._load_or_complete_retained_owner_roots(
        output_root=output, args=args, namespace_contracts=contracts
    ) == retained


def test_retained_owner_bootstrap_never_repairs_corrupt_existing_child(
    tmp_path,
):
    roots, contracts, args = _owner_namespace_fixture(tmp_path)
    output = tmp_path / "campaign"
    output.mkdir(mode=0o700)
    target_root = output / chunk.RETAINED_OWNER_NAMESPACE_ROOT / "plan"
    target_root.mkdir(parents=True, mode=0o700)
    first = contracts["plan"]["files"][0]
    target = target_root / first["name"]
    target.write_bytes(b"corrupt retained bytes\n")
    target.chmod(0o444)
    before = target.read_bytes()

    with pytest.raises(
        campaign.chunk_capture.WorkloadCaptureRefusal,
        match="supplemental_retained_input_mismatch",
    ):
        campaign._load_or_complete_retained_owner_roots(
            output_root=output, args=args, namespace_contracts=contracts
        )
    assert target.read_bytes() == before


def test_published_done_kill_window_recovers_one_completion_event_idempotently(
    tmp_path, monkeypatch,
):
    output = tmp_path / "campaign"
    done_path = output / "sol_workload_chunks/chunk-001/DONE.json"
    done_path.parent.mkdir(parents=True)
    done = {
        "attempt_histories": [
            {"selected_attempt_ordinal": 1},
            {"selected_attempt_ordinal": 2},
        ],
        "preflights": [
            {"sha256": "a" * 64}, {"sha256": "b" * 64},
        ],
    }
    done_path.write_bytes(chunk._canonical_json_bytes(done))
    matrix = {
        "chunks": [{"chunk_id": "chunk-001", "cohort": "native"}],
    }
    verified = []
    monkeypatch.setattr(
        campaign.chunk_capture, "verify_done_manifest",
        lambda manifest, **_kwargs: verified.append(manifest),
    )
    campaign._recover_published_completion_events(
        output_root=output, matrix=matrix, matrix_path=tmp_path / "matrix",
        canonical_matrix={}, canonical_matrix_path=tmp_path / "canonical",
    )
    events = campaign.load_event_chain(output)
    assert len(events) == 1
    assert events[0]["outcome"] == "completed"
    assert events[0]["retry_ordinal"] == 2
    assert events[0]["preflight_sha256s"] == ["a" * 64, "b" * 64]
    assert events[0]["done_relative_path"] == (
        "sol_workload_chunks/chunk-001/DONE.json"
    )
    assert verified == [done]

    campaign._recover_published_completion_events(
        output_root=output, matrix=matrix, matrix_path=tmp_path / "matrix",
        canonical_matrix={}, canonical_matrix_path=tmp_path / "canonical",
    )
    assert len(campaign.load_event_chain(output)) == 1
    assert verified == [done, done]


def test_published_done_recovery_refuses_conflicting_completion_event(tmp_path):
    output = tmp_path / "campaign"
    done_path = output / "sol_workload_chunks/chunk-001/DONE.json"
    done_path.parent.mkdir(parents=True)
    done = {
        "attempt_histories": [{"selected_attempt_ordinal": 1}],
        "preflights": [{"sha256": "a" * 64}],
    }
    done_path.write_bytes(chunk._canonical_json_bytes(done))
    event = campaign._ensure_completed_chunk_event(
        output_root=output,
        chunk={"chunk_id": "chunk-001", "cohort": "native"}, done=done,
    )
    event_path = output / campaign.EVENT_DIR / "event-000001.json"
    event["done_sha256"] = "f" * 64
    unsigned = {
        key: value for key, value in event.items()
        if key != "event_seal_sha256"
    }
    event["event_seal_sha256"] = chunk.canonical_sha256(unsigned)
    event_path.chmod(0o600)
    event_path.write_bytes(chunk._canonical_json_bytes(event))
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_completion_event_conflict",
    ):
        campaign._ensure_completed_chunk_event(
            output_root=output,
            chunk={"chunk_id": "chunk-001", "cohort": "native"}, done=done,
        )


def test_offline_checkpoint_rederives_after_relocation_and_source_deletion(
    tmp_path, monkeypatch,
):
    root = _materialize_offline_checkpoint_fixture(tmp_path, monkeypatch)
    campaign.verify_campaign_checkpoint_offline(root)
    relocated = tmp_path / "relocated-checkpoint"
    shutil.copytree(root, relocated)
    shutil.rmtree(root)
    campaign.verify_campaign_checkpoint_offline(relocated)

    frontier = next(
        path for path in (relocated / campaign.SNAPSHOT_DIR).iterdir()
        if path.name.startswith("snapshot-")
    )
    progress = json.loads((frontier / "PROGRESS.json").read_text())
    assert progress["campaign_state"] == "incomplete"
    campaign._assert_no_absolute_evidence_strings(
        progress, role="test progress"
    )


def test_offline_checkpoint_rejects_orphan_completion_event(
    tmp_path, monkeypatch,
):
    root = _materialize_offline_checkpoint_fixture(tmp_path, monkeypatch)
    marker, _payload, _pin = chunk._load_retained_input_owner(root)
    matrix_identity = marker["campaign_identity"]["input_file_identities"][
        "supplemental_matrix"
    ]
    matrix = json.loads(
        (root / matrix_identity["retained_owner_relative_path"]).read_text()
    )
    chunk_row = matrix["chunks"][0]
    campaign.append_event(root, {
        "chunk_id": chunk_row["chunk_id"], "cohort": chunk_row["cohort"],
        "outcome": "completed", "retry_ordinal": 1,
        "preflight_sha256s": ["a" * 64],
        "done_relative_path": (
            f"sol_workload_chunks/{chunk_row['chunk_id']}/DONE.json"
        ),
        "done_sha256": "b" * 64, "failure_code": None,
        "failure_detail": None, "row_outcomes": [],
    })
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_completion_event_without_done",
    ):
        campaign.verify_campaign_checkpoint_offline(root)


def test_offline_checkpoint_rejects_unknown_event_shape(
    tmp_path, monkeypatch,
):
    root = _materialize_offline_checkpoint_fixture(tmp_path, monkeypatch)
    marker, _payload, _pin = chunk._load_retained_input_owner(root)
    matrix_identity = marker["campaign_identity"]["input_file_identities"][
        "supplemental_matrix"
    ]
    matrix = json.loads(
        (root / matrix_identity["retained_owner_relative_path"]).read_text()
    )
    chunk_row = matrix["chunks"][0]
    campaign.append_event(root, {
        "chunk_id": chunk_row["chunk_id"], "cohort": chunk_row["cohort"],
        "outcome": "invented_outcome", "retry_ordinal": 1,
        "row_outcomes": [],
    })
    with pytest.raises(
        campaign.CampaignRefusal, match="campaign_event_shape_invalid",
    ):
        campaign.verify_campaign_checkpoint_offline(root)


def test_offline_event_refuses_cross_chunk_row_outcome(tmp_path):
    first, second = _key(1), _key(2)
    matrix = {
        "source_commit": "a" * 40,
        "chunks": [
            {"chunk_id": "chunk-a", "cohort": "native", "rows": [first]},
            {"chunk_id": "chunk-b", "cohort": "native", "rows": [second]},
        ],
    }
    event = {
        "schema_version": 1, "kind": "sol_workload_campaign_event",
        "event_ordinal": 1, "prior_event_seal_sha256": None,
        "recorded_at_utc": "2026-07-16T00:00:00.000000Z",
        "event_seal_sha256": "b" * 64,
        "chunk_id": "chunk-a", "cohort": "native",
        "outcome": "budget_blocked", "retry_ordinal": 1,
        "preflight_owner_relative_path": (
            "sol_workload_campaign/inputs/preflights/" + "a" * 40
            + "/chunk-a/preflight.json"
        ),
        "preflight_sha256": "c" * 64,
        "failure_code": "supplemental_chunk_rows_budget_blocked",
        "failure_detail_sha256": "d" * 64,
        "storage_state": None,
        "row_outcomes": [{
            "row_key": second, "status": "budget_blocked",
            "failure_code": "trace_target_failed_tree_limit_exceeded",
            "attempt_ordinal": 1,
        }],
    }
    with pytest.raises(
        campaign.CampaignRefusal,
        match="campaign_event_row_outcomes_invalid",
    ):
        campaign._validate_offline_event_completion_closure(
            campaign_root=tmp_path, matrix=matrix, events=[event]
        )
