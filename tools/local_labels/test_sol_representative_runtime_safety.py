"""CPU-only safety gates for the representative SOL runtime boundary."""
from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.local_labels import build_sol_representative_execution_matrix as execution
from tools.local_labels import sm86_capture_chunk as sm86
from tools.local_labels import sol_workload_capture_campaign as campaign
from tools.local_labels import sol_workload_capture_chunk as capture
from tools.local_labels import test_sol_workload_capture_chunk as capture_fixtures


def _representative_matrix(*, chunks: list[dict] | None = None) -> dict:
    source_matrix_sha = "3" * 64
    return {
        "artifact_id": execution.ARTIFACT_ID,
        "execution_profile": execution.EXECUTION_PROFILE,
        "source_commit": "2" * 40,
        "authority_source_commit": "1" * 40,
        "matrix_seal_sha256": "4" * 64,
        "source_matrix": {"sha256": source_matrix_sha},
        "authority_input_pins": {
            "coverage_inventory": {"sha256": "0" * 64},
            "source_matrix": {"sha256": source_matrix_sha},
        },
        "representative_selection": {
            "selection_identity_sha256": "5" * 64,
        },
        "counts": {"rows": 216},
        "chunks": copy.deepcopy(chunks or []),
        "runnability_authority": {"owner_marker": {"name": ".owner.json"}},
        "reduced_proxy_authority": {
            "owner_marker": {"name": ".owner.json"},
        },
    }


def test_dispatch_and_frozen_inventory_bind_216_to_3420(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _representative_matrix()
    assert capture.is_representative_execution_matrix(matrix)
    for mutation in (
        {"artifact_id": "sm86_sol_workload_capture_matrix_live_v2"},
        {"execution_profile": None},
        {"execution_profile": execution.EXECUTION_PROFILE + "_open"},
    ):
        candidate = copy.deepcopy(matrix)
        candidate.update(mutation)
        assert not capture.is_representative_execution_matrix(candidate)

    root = tmp_path / "frozen-plan"
    root.mkdir()
    inventory_path = root / "sol_live_workload_coverage_inventory.v1.json"
    document = {
        "source_commit": matrix["authority_source_commit"],
        "inputs": {"matrix": {"sha256": matrix["source_matrix"]["sha256"]}},
        "counts": {
            "source_workload_rows": 3957,
            "in_scope_rows": 3420,
            "source_rows_with_any_schedule": 216,
            "source_rows_unscheduled_in_scope": 3204,
        },
    }
    payload = capture.live_coverage.canonical_json_bytes(document)
    inventory_path.write_bytes(payload)
    matrix["authority_input_pins"]["coverage_inventory"]["sha256"] = (
        hashlib.sha256(payload).hexdigest()
    )
    owner = {"canonical_path": str(root.resolve()), "owner": "fixture"}
    monkeypatch.setattr(
        capture.sol_recon, "validate_external_campaign_root", lambda _root: owner,
    )
    monkeypatch.setattr(
        capture.sol_recon,
        "read_external_child",
        lambda _owner, name: (root / name).read_bytes(),
    )
    schema_calls: list[tuple[dict, Path]] = []
    monkeypatch.setattr(
        capture,
        "_validate_schema",
        lambda value, schema, **_kwargs: schema_calls.append((value, schema)),
    )

    observed, summary = capture.validate_representative_live_coverage_inventory(
        inventory_path=inventory_path, inventory_root=root, matrix=matrix,
    )
    assert schema_calls == [(document, capture.LIVE_COVERAGE_SCHEMA)]
    assert observed["counts"]["in_scope_rows"] == 3420
    assert summary["source_workload_rows"] == 3957
    assert summary["matrix_schedule_rows"] == 216
    assert summary["source_rows_with_any_schedule"] == 216

    stale = copy.deepcopy(matrix)
    stale["source_matrix"]["sha256"] = "6" * 64
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="representative_live_coverage_inventory_invalid",
    ):
        capture.validate_representative_live_coverage_inventory(
            inventory_path=inventory_path, inventory_root=root, matrix=stale,
        )


def test_sm86_binder_requires_exact_closed_execution_profile(
    tmp_path: Path,
) -> None:
    workload_path, rows, values = capture_fixtures._literal_rows(tmp_path)
    invocation, source_binding = capture_fixtures._binding(
        path=workload_path, rows=rows, values=values,
    )
    row = {
        name: copy.deepcopy(value)
        for name, value in source_binding.items()
        if name not in {
            "schema_version", "supplemental_matrix_sha256",
            "supplemental_matrix_seal_sha256", "chunk_id",
        }
    }
    row["key"] = row.pop("row_key")
    row["launch_completeness_contract"] = copy.deepcopy(
        sm86._SOL_REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    matrix = _representative_matrix()
    binding = capture.workload_row_binding(
        matrix=matrix, matrix_sha256="7" * 64,
        chunk_id="sol-representative-native-000", row=row,
    )
    assert binding["execution_profile"] == execution.EXECUTION_PROFILE
    rebound = sm86.bind_supplemental_sol_workload_row(invocation, binding)
    assert rebound["capture_target"]["source_workload_index"] == 1

    invalid_bindings: list[dict] = []
    missing = copy.deepcopy(binding)
    missing.pop("execution_profile")
    invalid_bindings.append(missing)
    wrong = copy.deepcopy(binding)
    wrong["execution_profile"] = execution.EXECUTION_PROFILE + "_wrong"
    invalid_bindings.append(wrong)
    opened = copy.deepcopy(binding)
    opened["profile_extension"] = "not-allowed"
    invalid_bindings.append(opened)
    for invalid in invalid_bindings:
        with pytest.raises(sm86.ChunkRefusal, match="sol_workload_row_binding_invalid"):
            sm86.bind_supplemental_sol_workload_row(invocation, invalid)


def _representative_completeness(
    *, outcome: str = "typed_unavailable_no_stable_static_module",
    reason: str = "origin_unresolved",
) -> dict:
    launch = {
        "launch_ordinal": 0,
        "launch_identity_sha256": "8" * 64,
        "kernel_name": "fixture_kernel",
        "trace_complete": True,
        "ncu_label_complete": True,
        "ncu_report_sha256": "9" * 64,
        "ncu_label_typed_reason": None,
        "control_complete": False,
        "control_outcome": outcome,
        "control_typed_reason": reason,
    }
    sequence = [{
        "launch_ordinal": 0,
        "kernel_name": "fixture_kernel",
        "launch_identity_sha256": "8" * 64,
    }]
    sequence_sha = capture.canonical_sha256(sequence)
    document = {
        "schema_version": 1,
        "launch_scope": "all_in_profiler_roi",
        "contract": copy.deepcopy(
            execution.REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
        ),
        "status": "complete_dynamic_launch_set",
        "launch_inventory_sha256": "a" * 64,
        "trace_launch_sequence_sha256": sequence_sha,
        "ncu_replay_sequence_sha256": sequence_sha,
        "trace_ncu_sequence_equal": True,
        "inventory_launch_count": 1,
        "trace_complete_launch_count": 1,
        "ncu_label_complete_launch_count": 1,
        "control_complete_launch_count": 0,
        "launches": [launch],
    }
    document["completeness_sha256"] = capture.canonical_sha256(document)
    return document


def test_done_admits_only_the_exact_origin_unresolved_control_token() -> None:
    capture.validate_result_launch_completeness({
        "launch_completeness": _representative_completeness(),
        "disposition": "admitted",
        "typed_reason": None,
    })
    for outcome, reason in (
        ("typed_unavailable_no_stable_static_module", "jit_or_anonymous_mapping"),
        ("typed_unavailable", "origin_unresolved"),
        ("control_probe_failed", "origin_unresolved"),
    ):
        with pytest.raises(
            capture.WorkloadCaptureRefusal,
            match="supplemental_done_launch_completeness_invalid",
        ):
            capture.validate_result_launch_completeness({
                "launch_completeness": _representative_completeness(
                    outcome=outcome, reason=reason,
                ),
                "disposition": "admitted",
                "typed_reason": None,
            })


def _campaign_chunks() -> list[dict]:
    rows: list[dict] = []
    cohort_sizes = (
        ("native", execution.ROWS_PER_CHUNK + 1),
        ("reduced_proxy", 1),
        ("partial_native_row", 1),
    )
    ordinal = 0
    for cohort, count in cohort_sizes:
        for _ in range(count):
            ordinal += 1
            rows.append({
                "cohort": cohort,
                "key": {
                    "slug": f"problem_{ordinal:03d}_fixture",
                    "source_workload_uuid": f"uuid-{ordinal}",
                    "variant_id": "native" if cohort == "native" else "b" * 64,
                },
            })
    return execution._interleaved_chunks(rows)


def _run_fake_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_first: bool,
) -> tuple[Path, dict[str, list]]:
    # The campaign imports this dependency as a script-local module, so patch
    # the exact object it dispatches through rather than a second package alias.
    runtime_capture = campaign.chunk_capture
    chunks = _campaign_chunks()
    matrix = _representative_matrix(chunks=chunks)
    canonical_matrix = {"matrix_seal_sha256": "c" * 64}
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    plan_root = tmp_path / "plan"
    runnability_root = tmp_path / "runnability"
    shrink_root = tmp_path / "shrink"
    for root in (plan_root, runnability_root, shrink_root):
        root.mkdir()
    paths = {
        "matrix": plan_root / "execution.json",
        "live": plan_root / "coverage.json",
        "runnability": runnability_root / "runnability.json",
        "shrink": shrink_root / "shrink.json",
        "canonical": tmp_path / "canonical.json",
        "authority": tmp_path / "authority.json",
        "pins": tmp_path / "pins.json",
        "budget": tmp_path / "budget.json",
        "cells": tmp_path / "cells.json",
        "registry": tmp_path / "registry.json",
        "full": tmp_path / "full.json",
    }
    for name, path in paths.items():
        if name == "full":
            path.write_bytes(
                campaign.coverage_inventory_builder.canonical_json_bytes({})
            )
        else:
            path.write_text("{}\n")

    monkeypatch.setattr(
        runtime_capture, "_output_lstat", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_capture, "assert_output_disjoint_from_input_owners",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_capture, "lexical_input_path", lambda value: Path(value),
    )
    monkeypatch.setattr(
        runtime_capture,
        "_load_object",
        lambda _path, **kwargs: (
            matrix
            if kwargs.get("role") == "supplemental matrix"
            else canonical_matrix
        ),
    )
    monkeypatch.setattr(
        runtime_capture.canonical_capture, "_git_state",
        lambda: (matrix["source_commit"], True, []),
    )
    monkeypatch.setattr(
        campaign, "_regenerate_full_coverage_inventory", lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        campaign, "validate_full_coverage_inventory", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runtime_capture, "validate_representative_execution_matrix",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_capture,
        "validate_representative_live_coverage_inventory",
        lambda **_kwargs: (
            {"source_commit": matrix["authority_source_commit"]},
            {"matrix_schedule_rows": 216, "sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        campaign, "validate_whole_matrix_closure", lambda _matrix: {},
    )
    monkeypatch.setattr(
        runtime_capture, "validate_external_reduced_sources",
        lambda *_args, **_kwargs: ({}, {}),
    )
    owner_namespaces = {"plan": {}, "runnability": {}, "shrink": {}}
    monkeypatch.setattr(
        runtime_capture, "capture_owner_namespace_contracts",
        lambda **_kwargs: owner_namespaces,
    )
    campaign_identity = {
        "owner_namespaces": owner_namespaces,
        "input_file_identities": {
            "supplemental_matrix": {"source_name": paths["matrix"].name},
            "live_coverage_inventory": {"source_name": paths["live"].name},
            "runnability": {"source_name": paths["runnability"].name},
            "shrink_provenance": {"source_name": paths["shrink"].name},
        },
    }
    monkeypatch.setattr(
        runtime_capture, "supplemental_campaign_identity",
        lambda **_kwargs: campaign_identity,
    )
    monkeypatch.setattr(
        runtime_capture, "establish_supplemental_output_root",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runtime_capture, "_campaign_input_sources", lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        runtime_capture, "retain_campaign_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_capture, "retain_owner_namespaces",
        lambda *_args, **_kwargs: None,
    )
    retained_roots = {
        "plan": plan_root, "runnability": runnability_root,
        "shrink": shrink_root,
    }
    monkeypatch.setattr(
        runtime_capture, "validate_retained_owner_namespaces",
        lambda *_args, **_kwargs: retained_roots,
    )
    monkeypatch.setattr(campaign, "_orchestrator_lock", lambda _root: object())
    monkeypatch.setattr(
        campaign, "_recover_published_completion_events",
        lambda **_kwargs: None,
    )
    observations: dict[str, list] = {
        "children": [], "preflights": [], "events": [],
        "completed": [], "reconciliations": [],
    }

    def fake_reconcile(**_kwargs):
        ordinal = len(observations["reconciliations"])
        snapshot = tmp_path / f"snapshot-{ordinal}"
        observations["reconciliations"].append(snapshot)
        return {"snapshot_root": snapshot}

    monkeypatch.setattr(campaign, "reconcile_campaign", fake_reconcile)
    monkeypatch.setattr(campaign, "load_event_chain", lambda _root: [])
    monkeypatch.setattr(campaign, "_latest_chunk_events", lambda _events: {})
    monkeypatch.setattr(
        campaign, "_storage_state",
        lambda _root: {"available_bytes": 10**12, "available_inodes": 10**6},
    )

    def fake_preflight(_args, *, chunk_id, retry_ordinal, **_kwargs):
        observations["preflights"].append((chunk_id, retry_ordinal))
        path = output_root / f"preflight-{chunk_id}-{retry_ordinal}.json"
        path.write_text("{}\n")
        return path

    monkeypatch.setattr(campaign, "_new_preflight", fake_preflight)

    def fake_child(args):
        observations["children"].append(args.chunk_id)
        if fail_first:
            detail = json.dumps({
                "row_key": {
                    "slug": "problem_001_fixture",
                    "source_workload_uuid": "uuid-1",
                    "variant_id": "native",
                },
                "cause_detail": "timeout_s=900",
            })
            raise runtime_capture.WorkloadCaptureRefusal(
                "trace_target_failed_timeout", detail,
            )
        return tmp_path / f"result-{args.chunk_id}"

    monkeypatch.setattr(runtime_capture, "run_chunk", fake_child)
    monkeypatch.setattr(
        runtime_capture, "_load_output_object", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        campaign,
        "_ensure_completed_chunk_event",
        lambda **kwargs: observations["completed"].append(
            kwargs["chunk"]["chunk_id"]
        ),
    )
    monkeypatch.setattr(
        campaign, "append_event",
        lambda _root, event: observations["events"].append(event),
    )

    args = Namespace(
        output_root=str(output_root), max_new_chunks=3, max_chunk_attempts=4,
        runnability_root=str(runnability_root), shrink_root=str(shrink_root),
        live_coverage_inventory_root=str(plan_root),
        matrix=str(paths["matrix"]), live_coverage_inventory=str(paths["live"]),
        runnability=str(paths["runnability"]),
        shrink_provenance=str(paths["shrink"]),
        canonical_matrix=str(paths["canonical"]),
        full_coverage_inventory=str(paths["full"]),
        authority=str(paths["authority"]), pins=str(paths["pins"]),
        aggregate_budget=str(paths["budget"]),
        cell_budgets=str(paths["cells"]),
        metric_registry=str(paths["registry"]),
    )
    return campaign._run_campaign_locked(args), observations


def test_representative_timeout_is_one_attempt_and_stops_after_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, observed = _run_fake_campaign(
        tmp_path, monkeypatch, fail_first=True,
    )
    first_chunk = _campaign_chunks()[0]["chunk_id"]
    assert observed["children"] == [first_chunk]
    assert observed["preflights"] == [(first_chunk, 1)]
    assert len(observed["reconciliations"]) == 2
    assert result == observed["reconciliations"][-1]
    assert len(observed["events"]) == 1
    event = observed["events"][0]
    assert event["outcome"] == "measured_infeasible"
    assert event["retry_ordinal"] == 1
    assert event["row_outcomes"][0]["status"] == (
        "measured_infeasible_trace_capture"
    )


def test_three_chunk_pilot_runs_native_reduced_partial_in_child_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = _campaign_chunks()
    assert [chunk["cohort"] for chunk in chunks[:3]] == [
        "native", "reduced_proxy", "partial_native_row",
    ]
    assert chunks[3]["cohort"] == "native"
    result, observed = _run_fake_campaign(
        tmp_path, monkeypatch, fail_first=False,
    )
    expected = [chunk["chunk_id"] for chunk in chunks[:3]]
    assert observed["children"] == expected
    assert [chunk_id for chunk_id, _retry in observed["preflights"]] == expected
    assert observed["completed"] == expected
    assert len(observed["reconciliations"]) == 4
    assert result == observed["reconciliations"][-1]


def test_representative_aggregate_budget_refuses_below_224_gib() -> None:
    matrix = _representative_matrix()
    matrix["aggregate_retention_contract"] = {
        "required_retention_budget_gib": 224,
    }
    capture.validate_representative_aggregate_budget(
        matrix, {"retention_budget_gib": 224},
    )
    with pytest.raises(
        capture.WorkloadCaptureRefusal,
        match="representative_aggregate_retention_budget_insufficient",
    ):
        capture.validate_representative_aggregate_budget(
            matrix, {"retention_budget_gib": 223},
        )
