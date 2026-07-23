"""CPU-only integration gates for the representative SOL campaign profile."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.local_labels import build_sol_representative_execution_matrix as execution
from tools.local_labels import campaign_preflight
from tools.local_labels import sm86_capture_chunk as canonical_capture
from tools.local_labels import sol_workload_capture_campaign as campaign
from tools.local_labels import sol_workload_capture_chunk as chunk
from tools.local_labels import test_sol_workload_capture_matrix as source_fixture


def _representative_completeness(*, typed_reason: str = "origin_unresolved") -> dict:
    launches = [{
        "launch_ordinal": 0,
        "launch_identity_sha256": "1" * 64,
        "kernel_name": "kernel",
        "trace_complete": True,
        "ncu_label_complete": True,
        "ncu_report_sha256": "2" * 64,
        "ncu_label_typed_reason": None,
        "control_complete": False,
        "control_outcome": "typed_unavailable_no_stable_static_module",
        "control_typed_reason": typed_reason,
    }]
    sequence = [{
        "launch_ordinal": 0,
        "kernel_name": "kernel",
        "launch_identity_sha256": "1" * 64,
    }]
    sequence_sha = chunk.canonical_sha256(sequence)
    document = {
        "schema_version": 1,
        "launch_scope": "all_in_profiler_roi",
        "contract": copy.deepcopy(
            execution.REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
        ),
        "status": "complete_dynamic_launch_set",
        "launch_inventory_sha256": "3" * 64,
        "trace_launch_sequence_sha256": sequence_sha,
        "ncu_replay_sequence_sha256": sequence_sha,
        "trace_ncu_sequence_equal": True,
        "inventory_launch_count": 1,
        "trace_complete_launch_count": 1,
        "ncu_label_complete_launch_count": 1,
        "control_complete_launch_count": 0,
        "launches": launches,
    }
    document["completeness_sha256"] = chunk.canonical_sha256(document)
    return document


def test_representative_done_admits_exact_origin_unresolved_control_only():
    result = {
        "launch_completeness": _representative_completeness(),
        "disposition": "admitted",
        "typed_reason": None,
    }
    chunk.validate_result_launch_completeness(result)

    invalid = copy.deepcopy(result)
    invalid["launch_completeness"] = _representative_completeness(
        typed_reason="jit_or_anonymous_mapping"
    )
    with pytest.raises(
        chunk.WorkloadCaptureRefusal,
        match="supplemental_done_launch_completeness_invalid",
    ):
        chunk.validate_result_launch_completeness(invalid)


def test_exhaustive_done_still_rejects_typed_control_as_complete():
    document = _representative_completeness()
    document["contract"] = copy.deepcopy(
        canonical_capture._SOL_ALL_LAUNCH_COMPLETENESS_CONTRACT
    )
    document["completeness_sha256"] = chunk.canonical_sha256({
        key: value for key, value in document.items()
        if key != "completeness_sha256"
    })
    with pytest.raises(chunk.WorkloadCaptureRefusal):
        chunk.validate_result_launch_completeness({
            "launch_completeness": document,
            "disposition": "admitted",
            "typed_reason": None,
        })


def test_preflight_routes_representative_matrix_without_canonical_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    matrix = {
        "schema_version": 1,
        "artifact_id": execution.ARTIFACT_ID,
        "execution_profile": execution.EXECUTION_PROFILE,
    }
    matrix_path = tmp_path / "representative.json"
    matrix_path.write_text(json.dumps(matrix))
    calls: list[tuple[dict, Path]] = []
    monkeypatch.setattr(
        campaign_preflight.representative_execution,
        "validate_execution_matrix",
        lambda observed, **kwargs: calls.append(
            (observed, kwargs["matrix_path"])
        ),
    )
    monkeypatch.setattr(
        campaign_preflight.sol_environment,
        "probe_supplemental_sol_environment",
        lambda observed, **_kwargs: {
            "status": "ok", "observed_artifact": observed["artifact_id"],
        },
    )
    result = campaign_preflight.probe_sol_native_environment(matrix_path)
    assert calls == [(matrix, matrix_path)]
    assert result["status"] == "ok"
    assert result["observed_artifact"] == (
        campaign_preflight.sol_environment.SUPPLEMENTAL_MATRIX_ARTIFACT_ID
    )
    assert result["execution_profile"] == execution.EXECUTION_PROFILE


def test_representative_matrix_dispatch_is_canonical_and_never_calls_3420_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    matrix = {
        "artifact_id": execution.ARTIFACT_ID,
        "execution_profile": execution.EXECUTION_PROFILE,
    }
    path = tmp_path / "execution.json"
    path.write_bytes(execution.canonical_bytes(matrix))
    calls: list[tuple[dict, Path]] = []
    monkeypatch.setattr(
        chunk.representative_execution, "validate_execution_matrix",
        lambda observed, **kwargs: calls.append(
            (observed, kwargs["matrix_path"])
        ),
    )
    monkeypatch.setattr(
        chunk, "validate_supplemental_matrix",
        lambda *_args, **_kwargs: pytest.fail("exhaustive regeneration called"),
    )
    chunk.validate_representative_execution_matrix(matrix, matrix_path=path)
    assert calls == [(matrix, path)]


def test_representative_aggregate_budget_requires_sealed_224_gib():
    matrix = {
        "artifact_id": execution.ARTIFACT_ID,
        "execution_profile": execution.EXECUTION_PROFILE,
        "aggregate_retention_contract": {"required_retention_budget_gib": 224},
    }
    chunk.validate_representative_aggregate_budget(
        matrix, {"retention_budget_gib": 224}
    )
    with pytest.raises(
        chunk.WorkloadCaptureRefusal,
        match="representative_aggregate_retention_budget_insufficient",
    ):
        chunk.validate_representative_aggregate_budget(
            matrix, {"retention_budget_gib": 223}
        )
    # The exhaustive matrix never acquires the representative requirement.
    chunk.validate_representative_aggregate_budget(
        {"artifact_id": "sm86_sol_workload_capture_matrix_live_v2"},
        {"retention_budget_gib": 200},
    )


def test_representative_timeout_and_trace_limit_are_one_typed_terminal_outcome():
    row_key = {
        "slug": "problem_001_unit",
        "source_workload_uuid": "uuid",
        "variant_id": "native",
    }
    detail = json.dumps({"row_key": row_key, "cause_detail": "timeout_s=900"})
    outcomes = campaign._failure_row_outcomes(
        code="trace_target_failed_timeout", detail=detail,
        classification="representative_measured_infeasible",
        terminal_retry=False,
    )
    assert outcomes == [{
        "row_key": row_key,
        "status": "measured_infeasible_trace_capture",
        "failure_code": "trace_target_failed_timeout",
        "attempt_ordinal": None,
    }]
    assert "measured_infeasible" in campaign._FAILURE_OUTCOMES


def test_first_three_campaign_slots_are_cross_cohort_and_cohort_pure(
    tmp_path: Path,
):
    rows = []
    for cohort, start, count in (
        ("native", 0, 18), ("reduced_proxy", 100, 17),
        ("partial_native_row", 200, 3),
    ):
        for index in range(start, start + count):
            rows.append({
                "cohort": cohort,
                "key": {
                    "slug": f"problem_{index:03d}_unit",
                    "source_workload_uuid": f"uuid-{index}",
                    "variant_id": "native" if cohort == "native" else "a" * 64,
                },
            })
    chunks = execution._interleaved_chunks(rows)
    selected = campaign._ordered_unpublished_chunks(
        {"chunks": chunks}, output_root=tmp_path, max_new_chunks=3,
    )
    assert [item["cohort"] for item in selected] == [
        "native", "reduced_proxy", "partial_native_row",
    ]
    for item in selected:
        assert all(
            next(row for row in rows if row["key"] == member)["cohort"]
            == item["cohort"]
            for member in item["rows"]
        )


def test_no_gpu_matrix_closure_chunk_selection_and_binding_consumability(
    tmp_path: Path,
):
    variants, rows, _record, _runnability, _path = (
        source_fixture._native_fixture(tmp_path, statuses=("PASSED",))
    )
    source_variant = variants[0]
    source_row = rows[0]
    execution_variant = execution.rebuild_execution_variant(
        source_variant, authority_commit="1" * 40,
        execution_commit="2" * 40, carry_forward_sha256="3" * 64,
    )
    execution_row = execution.rebuild_execution_row(
        source_row, variant=execution_variant,
        authority_row_sha256=execution.canonical_sha256(source_row),
        execution_commit="2" * 40,
    )
    chunks = execution._interleaved_chunks([execution_row])
    matrix = {
        "artifact_id": execution.ARTIFACT_ID,
        "execution_profile": execution.EXECUTION_PROFILE,
        "matrix_seal_sha256": "4" * 64,
        "variants": [execution_variant],
        "rows": [execution_row],
        "chunks": chunks,
        "counts": {
            "problems": 1, "variants": 1, "rows": 1, "chunks": 1,
            "native_variants": 1, "partial_native_variants": 0,
            "reduced_variants": 0, "native_rows": 1,
            "partial_native_rows": 0, "reduced_proxy_rows": 0,
        },
    }
    closure = campaign.validate_whole_matrix_closure(matrix)
    assert closure["scheduled_keys"] == [
        chunk._row_identity(execution_row["key"])
    ]
    selected_chunk, resolved = chunk.select_workload_chunk(
        matrix, {"cells": []}, chunks[0]["chunk_id"]
    )
    assert selected_chunk == chunks[0]
    assert len(resolved) == 1
    binding = chunk.workload_row_binding(
        matrix=matrix, matrix_sha256="5" * 64,
        chunk_id=chunks[0]["chunk_id"], row=execution_row,
    )
    assert binding["execution_profile"] == execution.EXECUTION_PROFILE
    assert binding["launch_completeness_contract"] == (
        execution.REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
