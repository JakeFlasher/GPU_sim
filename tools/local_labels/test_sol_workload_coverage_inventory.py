"""Focused tests for the deterministic SOL workload coverage inventory."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import sol_workload_coverage_inventory as inv  # noqa: E402


@pytest.fixture(scope="module")
def inventory() -> dict:
    return inv.build_inventory(_REPO)


def test_unified_and_sol_problem_units_are_distinct_and_reconcile(inventory):
    assert inventory["unified_roster"] == {
        "universe": 344,
        "runnable": 122,
        "pending_build": 0,
        "pending_reconstitution": 0,
        "pending_shrink": 0,
        "excluded": 222,
        "by_source": {
            "portable_lcd": {"runnable": 17},
            "simbench_bw": {"runnable": 31, "excluded": 61},
            "sol_execbench": {"runnable": 74, "excluded": 161},
        },
    }
    sol = inventory["sol_problem_dispositions"]
    assert sol["total_problem_count"] == 235
    assert sol["runnable_problem_count"] == 74
    assert sol["typed_excluded_problem_count"] == 161
    assert len(sol["problems"]) == 235
    assert len({row["slug"] for row in sol["problems"]}) == 235


def test_all_3957_workload_rows_are_accounted_by_category_and_disposition(
        inventory):
    rows = inventory["workload_rows"]
    assert rows["total_workload_row_count"] == 3957
    assert rows["by_category"] == {
        "FIB": {"problem_count": 26, "workload_row_count": 660},
        "L1": {"problem_count": 94, "workload_row_count": 1480},
        "L2": {"problem_count": 82, "workload_row_count": 1299},
        "Quant": {"problem_count": 33, "workload_row_count": 518},
    }
    assert rows["by_disposition"] == {
        "architecture_excluded": 537,
        "capacity_excluded": 205,
        "compiler_excluded": 1149,
        "correctness_excluded": 772,
        "evaluation_excluded": 112,
        "native_runnable": 1166,
        "reduced_runnable": 16,
    }
    assert sum(rows["by_disposition"].values()) == 3957


def test_exclusion_buckets_separate_arch_capacity_compiler_and_correctness(
        inventory):
    buckets = inventory["exclusion_buckets"]
    assert {
        name: (bucket["problem_count"], bucket["workload_row_count"])
        for name, bucket in buckets.items()
    } == {
        "architecture": (34, 537),
        "capacity": (14, 205),
        "compiler": (65, 1149),
        "correctness": (41, 772),
        "evaluation": (7, 112),
    }
    assert sum(bucket["problem_count"] for bucket in buckets.values()) == 161


def test_native_runnable_selection_counts_selected_and_unselected_rows(inventory):
    selected = inventory["workload_rows"]["native_runnable_capture_selection"]
    assert selected == {
        "problem_count": 73,
        "workload_row_count": 1166,
        "selected_workload_row_count": 73,
        "unselected_workload_row_count": 1093,
        "source_workload_index_counts": {"0": 73},
    }
    assert (selected["selected_workload_row_count"]
            + selected["unselected_workload_row_count"]
            == selected["workload_row_count"])


def test_reduced_file_does_not_mislabel_native_equal_selected_row(inventory):
    evidence = inventory["workload_rows"]["reduced_workload_evidence"]
    assert evidence == {
        "native_host_capacity_problem_count": 17,
        "no_reduced_workload_file_problem_count": 1,
        "reduced_workload_file_problem_count": 16,
        "native_workload_row_count": 236,
        "reduced_workload_row_count": 236,
        "changed_workload_row_count": 191,
        "unchanged_workload_row_count": 45,
        "capture_selected_problem_count": 1,
        "capture_selected_row_differs_from_native_count": 0,
        "capture_selected_row_matches_native_count": 1,
        "selected_row_comparison_basis": "parsed_json_value_at_matrix_indices",
        "by_classification": {
            "correctness_failed": 2,
            "reduced_representative": 1,
            "unavailable::oom_shape_gated": 14,
        },
    }
    reduced = [
        row for row in inventory["sol_problem_dispositions"]["problems"]
        if row["disposition"] == "reduced_runnable"
    ]
    assert [row["slug"] for row in reduced] == [
        "problem_028_gqa_rotary_attention_core_backward"]
    selection = reduced[0]["capture_selection"]
    assert selection["changed_workload_rows"] == 11
    assert selection["unchanged_workload_rows"] == 5
    assert selection["source_workload_index"] == 0
    assert selection["staged_workload_index"] == 0
    assert selection["selected_row_differs_from_native"] is False
    assert selection["selected_native_uuid"] == selection["selected_reduced_uuid"]


def test_report_bytes_are_canonical_and_deterministic(inventory):
    first = inv.canonical_json_bytes(inventory)
    second = inv.canonical_json_bytes(inv.build_inventory(_REPO))
    assert first == second
    assert first.endswith(b"\n") and not first.endswith(b"\n\n")
    assert json.loads(first) == inventory
    assert b"generated_at" not in first


def test_duplicate_json_keys_and_reduced_uuid_reordering_fail_closed():
    with pytest.raises(inv.CoverageInventoryError, match="duplicate JSON object key"):
        inv._parse_json(b'{"a": 1, "a": 2}', "fixture")
    with pytest.raises(inv.CoverageInventoryError, match="non-finite JSON number"):
        inv._parse_json(b'{"a": NaN}', "fixture")

    native = [{"uuid": "a", "axes": {"batch": 8}},
              {"uuid": "b", "axes": {"batch": 4}}]
    reordered = [{"uuid": "b", "axes": {"batch": 4}},
                 {"uuid": "a", "axes": {"batch": 8}}]
    with pytest.raises(inv.CoverageInventoryError, match="UUID order"):
        inv._compare_workload_rows(
            native, reordered, source_index=0, staged_index=0, label="fixture")


def test_selected_row_comparison_uses_values_at_the_two_matrix_indices():
    native = [{"uuid": "a", "axes": {"batch": 1}},
              {"uuid": "b", "axes": {"batch": 8}}]
    reduced = [{"uuid": "a", "axes": {"batch": 1}},
               {"uuid": "b", "axes": {"batch": 1}}]
    row_zero = inv._compare_workload_rows(
        native, reduced, source_index=0, staged_index=0, label="fixture")
    row_one = inv._compare_workload_rows(
        native, reduced, source_index=1, staged_index=1, label="fixture")
    assert row_zero["changed_workload_rows"] == 1
    assert row_zero["selected_row_differs_from_native"] is False
    assert row_one["selected_row_differs_from_native"] is True


def test_changed_runnability_bytes_cannot_bypass_cross_file_pin(tmp_path):
    source = _REPO / inv.DEFAULT_INPUTS.runnability
    changed = json.loads(source.read_text())
    changed["native_env_degraded"] = not changed["native_env_degraded"]
    altered = tmp_path / "sol_runnability_sm86.json"
    altered.write_bytes(inv.canonical_json_bytes(changed))
    inputs = replace(inv.DEFAULT_INPUTS, runnability=altered)
    with pytest.raises(inv.CoverageInventoryError,
                       match="authority SOL source pin mismatch"):
        inv.build_inventory(_REPO, inputs)


def test_matrix_selection_other_than_bound_index_zero_fails_closed(tmp_path):
    source = _REPO / inv.DEFAULT_INPUTS.matrix
    changed = copy.deepcopy(json.loads(source.read_text()))
    cell = next(
        row for row in changed["cells"]
        if row["identity"]["roster_source"] == "sol_execbench"
        and row["disposition"] == "runnable"
    )
    cell["invocation"]["capture_target"]["source_workload_index"] = 1
    altered = tmp_path / "sm86_reference_capture_matrix.json"
    altered.write_bytes(inv.canonical_json_bytes(changed))
    inputs = replace(inv.DEFAULT_INPUTS, matrix=altered)
    with pytest.raises(inv.CoverageInventoryError,
                       match="must select source/staged index 0"):
        inv.build_inventory(_REPO, inputs)
