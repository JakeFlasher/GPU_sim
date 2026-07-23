"""Contracts for the additive 216-row SOL representative trace census."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import build_sol_representative_capture_matrix as rep


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / rep.SCHEMA_PATH).read_text())
SHA = "a" * 64


def _diagnostic(
    *, uuid: str, index: int, axes: dict, row_sha: str, latency_ms: float,
) -> dict:
    return {
        "workload_index": index,
        "source_line": index + 1,
        "uuid": uuid,
        "axes": axes,
        "workload_sha256": row_sha,
        "diagnostic_present": True,
        "trace_index": index,
        "status": "PASSED",
        "trace": {
            "evaluation": {
                "status": "PASSED",
                "performance": {"latency_ms": latency_ms},
            },
            "workload": {"uuid": uuid, "axes": axes},
        },
    }


def _variant(*, slug: str, variant_id: str, cohort: str) -> dict:
    native = cohort == "native"
    partial = cohort == "partial_native_row"
    return {
        "slug": slug,
        "variant_id": variant_id,
        "cohort": cohort,
        "native_shape": native or partial,
        "native_predictive": native,
        "variant_contract_sha256": SHA,
        **({"reduced_provenance": {}} if not native else {}),
    }


def _footprint(value: int) -> dict:
    return {
        "model": "declared_tensor_io_footprint_v1",
        "definition_path": "definition.json",
        "definition_sha256": SHA,
        "aggregate_declared_tensor_bits": value,
        "peak_declared_tensor_bits": value,
        "aggregate_declared_tensor_elements": value,
        "resolved_axis_volume": max(1, value),
        "scalar_control_tuple": [],
        "tensor_count": 1,
    }


def _footprint_index(matrix: dict, values: dict[str, int] | None = None) -> dict:
    values = values or {}
    return {
        (
            row["key"]["slug"], row["key"]["source_workload_uuid"],
            row["key"]["variant_id"],
        ): _footprint(values.get(row["key"]["source_workload_uuid"], 1))
        for row in matrix["rows"]
    }


def _row(
    *, slug: str, uuid: str, source_index: int, capture_index: int,
    source_sha: str, capture_sha: str, source_axes: dict, capture_axes: dict,
    cohort: str, variant_id: str, evidence_sha: str,
) -> dict:
    native = cohort == "native"
    return {
        "key": {
            "slug": slug,
            "source_workload_uuid": uuid,
            "variant_id": variant_id,
        },
        "category": "L1",
        "cohort": cohort,
        "claim_scope": (
            "native_workload_row" if native else "reduced_proxy_bundle_only"
        ),
        "native_shape": native,
        "native_predictive": native,
        "problem_level_runnable": native,
        "variant_contract_sha256": SHA,
        "row_evidence_sha256": evidence_sha,
        "source_workload_index": source_index,
        "source_workload_uuid": uuid,
        "source_axes": source_axes,
        "source_row_sha256": source_sha,
        "capture_workload_index": capture_index,
        "capture_axes": capture_axes,
        "capture_row_sha256": capture_sha,
    }


def _problem(slug: str, *, reason=None, rows: int = 1) -> dict:
    return {
        "slug": slug,
        "category": "L1",
        "architecture_reason": reason,
        "source_workload_row_count": rows,
    }


def test_minimum_definition_footprint_native_row_is_selected_not_latency():
    slug = "problem_native"
    diagnostics = [
        _diagnostic(uuid="slow", index=0, axes={"n": 1}, row_sha="1" * 64,
                    latency_ms=9.0),
        _diagnostic(uuid="fast", index=1, axes={"n": 8}, row_sha="2" * 64,
                    latency_ms=2.0),
    ]
    rows = [
        _row(
            slug=slug,
            uuid=diagnostic["uuid"],
            source_index=index,
            capture_index=index,
            source_sha=diagnostic["workload_sha256"],
            capture_sha=diagnostic["workload_sha256"],
            source_axes=diagnostic["axes"],
            capture_axes=diagnostic["axes"],
            cohort="native",
            variant_id="native",
            evidence_sha=rep.source_contract_sha256(diagnostic),
        )
        for index, diagnostic in enumerate(diagnostics)
    ]
    matrix = {"variants": [_variant(slug=slug, variant_id="native", cohort="native")],
              "rows": rows}
    runnability = {"cells": [{"slug": slug, "status": "runnable",
                               "workload_diagnostics": diagnostics}]}
    selections, exclusions = rep.select_representatives(
        matrix=matrix,
        inventory={"problems": [_problem(slug)]},
        runnability=runnability,
        shrink={"cells": []},
        footprint_index=_footprint_index(matrix, {"slow": 1, "fast": 8}),
        enforce_current_counts=False,
    )
    assert exclusions == []
    assert len(selections) == 1
    assert selections[0]["source_workload_uuid"] == "slow"
    assert selections[0]["selection_class"] == "minimum_footprint_passing_native"
    assert selections[0]["selection_evidence"]["observed_latency_ms"] == 9.0
    assert selections[0]["selection_rank"]["aggregate_declared_tensor_bits"] == 1
    assert selections[0]["native_scale_claim_eligible"] is True


def test_equal_native_latency_uses_index_then_identity_tiebreak():
    slug = "problem_tie"
    diagnostics = [
        _diagnostic(uuid="z", index=0, axes={"n": 9}, row_sha="1" * 64,
                    latency_ms=2.0),
        _diagnostic(uuid="a", index=1, axes={"n": 1}, row_sha="2" * 64,
                    latency_ms=2.0),
    ]
    rows = [
        _row(
            slug=slug, uuid=item["uuid"], source_index=index,
            capture_index=index, source_sha=item["workload_sha256"],
            capture_sha=item["workload_sha256"], source_axes=item["axes"],
            capture_axes=item["axes"], cohort="native", variant_id="native",
            evidence_sha=rep.source_contract_sha256(item),
        )
        for index, item in enumerate(diagnostics)
    ]
    selections, _ = rep.select_representatives(
        matrix={"variants": [_variant(slug=slug, variant_id="native", cohort="native")],
                "rows": list(reversed(rows))},
        inventory={"problems": [_problem(slug)]},
        runnability={"cells": [{"slug": slug, "status": "runnable",
                                 "workload_diagnostics": diagnostics}]},
        shrink={"cells": []},
        footprint_index=_footprint_index({"rows": rows}),
        enforce_current_counts=False,
    )
    assert selections[0]["source_workload_uuid"] == "z"


def _reduced_case() -> tuple[dict, dict, dict, dict]:
    slug = "problem_oom"
    native_diag = _diagnostic(
        uuid="native-partial", index=0, axes={"n": 1}, row_sha="1" * 64,
        latency_ms=0.5,
    )
    partial_variant = _variant(
        slug=slug, variant_id="partial", cohort="partial_native_row"
    )
    # The matrix also contains a faster partial-native row.  Proxy precedence
    # must still choose only from explicit reduced rows.
    partial_variant["selected_diagnostic_sha256"] = (
        rep.source_contract_sha256(native_diag)
    )
    partial_row = _row(
        slug=slug, uuid="native-partial", source_index=0, capture_index=0,
        source_sha="1" * 64, capture_sha="1" * 64,
        source_axes={"n": 1}, capture_axes={"n": 1},
        cohort="native", variant_id="partial",
        evidence_sha=rep.source_contract_sha256(native_diag),
    )
    partial_row.update({
        "cohort": "partial_native_row",
        "claim_scope": "partial_native_workload_row_only",
        "native_predictive": False,
        "problem_level_runnable": False,
    })
    variants = [partial_variant]
    rows = [partial_row]
    diagnostics = []
    plans = []
    for index, (uuid, latency) in enumerate((("proxy-slow", 8.0), ("proxy-fast", 3.0))):
        source_sha = str(index + 2) * 64
        capture_sha = str(index + 4) * 64
        variant_id = str(index + 6) * 64
        plan = {
            "slug": slug,
            "source_workload_index": index + 1,
            "source_workload_uuid": uuid,
            "variant_id": variant_id,
            "planning_status": "proxy_planned",
            "claim_scope": "reduced_shape_only",
            "native_predictive": False,
            "reduced_run_status": "runnable",
            "correctness_result": "passed",
            "native_row_sha256": source_sha,
            "reduced_row_sha256": capture_sha,
        }
        diagnostic = _diagnostic(
            uuid=uuid, index=index, axes={"n": index + 1}, row_sha=capture_sha,
            latency_ms=latency,
        )
        variant = _variant(slug=slug, variant_id=variant_id, cohort="reduced_proxy")
        variant["reduced_provenance"] = {
            "row_plan_sha256": rep.source_contract_sha256(plan)
        }
        row = _row(
            slug=slug, uuid=uuid, source_index=index + 1, capture_index=index,
            source_sha=source_sha, capture_sha=capture_sha,
            source_axes={"n": 100 + index}, capture_axes={"n": index + 1},
            cohort="reduced_proxy", variant_id=variant_id,
            evidence_sha=rep.source_contract_sha256(plan),
        )
        variants.append(variant)
        rows.append(row)
        plans.append(plan)
        diagnostics.append(diagnostic)
    matrix = {"variants": variants, "rows": rows}
    inventory = {"problems": [_problem(slug, rows=3)]}
    runnability = {"cells": [{"slug": slug, "status": "host_capacity_oom",
                               "workload_diagnostics": [native_diag]}]}
    shrink = {"cells": [{
        "slug": slug,
        "classification": "reduced_representative",
        "claim_scope": "reduced_shape_only",
        "native_predictive": False,
        "reduced_run_status": "runnable",
        "correctness_result": "passed",
        "row_plans": plans,
        "cohort_execution_evidence": {"workload_diagnostics": diagnostics},
    }]}
    return matrix, inventory, runnability, shrink


def test_reduced_anchor_keeps_minimum_footprint_partial_native_companion():
    matrix, inventory, runnability, shrink = _reduced_case()
    selections, _ = rep.select_representatives(
        matrix=matrix,
        inventory=inventory,
        runnability=runnability,
        shrink=shrink,
        footprint_index=_footprint_index(
            matrix,
            {"native-partial": 1, "proxy-slow": 8, "proxy-fast": 3},
        ),
        enforce_current_counts=False,
    )
    assert len(selections) == 2
    selected = next(row for row in selections if row["selection_role"] == "application_coverage_anchor")
    companion = next(row for row in selections if row["selection_role"] == "native_comparison_companion")
    assert selected["source_workload_uuid"] == "proxy-fast"
    assert selected["selection_class"] == "minimum_footprint_passing_reduced_proxy"
    assert selected["scale_modified"] is True
    assert selected["native_predictive"] is False
    assert selected["native_scale_claim_eligible"] is False
    assert selected["claim_scope"] == "reduced_proxy_bundle_only"
    assert companion["source_workload_uuid"] == "native-partial"
    assert companion["cohort"] == "partial_native_row"
    assert companion["native_scale_claim_eligible"] is True
    assert companion["problem_level_native_claim_eligible"] is False


def test_reduced_proxy_cannot_be_relabelled_native_predictive():
    matrix, inventory, runnability, shrink = _reduced_case()
    reduced = next(row for row in matrix["rows"] if row["cohort"] == "reduced_proxy")
    reduced["native_predictive"] = True
    with pytest.raises(rep.RepresentativeMatrixError, match="reduced-proxy selection contract"):
        rep.select_representatives(
            matrix=matrix, inventory=inventory, runnability=runnability,
            shrink=shrink, footprint_index=_footprint_index(matrix),
            enforce_current_counts=False,
        )


def test_architecture_inapplicable_is_typed_and_never_scheduled():
    selections, exclusions = rep.select_representatives(
        matrix={"variants": [], "rows": []},
        inventory={"problems": [
            _problem("problem_fp8", reason="fp8_e4m3_contract", rows=19),
            _problem("problem_nvfp4", reason="nvfp4_e2m1_contract", rows=16),
        ]},
        runnability={"cells": []},
        shrink={"cells": []},
        footprint_index={},
        enforce_current_counts=False,
    )
    assert selections == []
    assert [item["reason"] for item in exclusions] == [
        "fp8_e4m3_contract", "nvfp4_e2m1_contract"
    ]
    assert sum(item["source_workload_rows"] for item in exclusions) == 35


def test_unknown_architecture_reason_fails_closed():
    with pytest.raises(rep.RepresentativeMatrixError, match="untyped"):
        rep.select_representatives(
            matrix={"variants": [], "rows": []},
            inventory={"problems": [_problem("problem_bad", reason="oom")]},
            runnability={"cells": []},
            shrink={"cells": []},
            footprint_index={},
            enforce_current_counts=False,
        )


def test_definition_footprint_resolves_axes_dtypes_and_scalar_controls():
    definition = {
        "axes": {
            "B": {"type": "var"},
            "C": {"type": "const", "value": 2},
            "X": {"type": "expr", "expression": "B * C"},
        },
        "inputs": {
            "data": {"shape": ["B", "C"], "dtype": "float16"},
            "alpha": {"shape": None, "dtype": "int32"},
        },
        "outputs": {
            "result": {"shape": ["X"], "dtype": "float4_e2m1fn_x2"},
        },
    }
    footprint = rep.definition_footprint(
        definition=definition,
        capture_axes={"B": 3, "alpha": 7},
        definition_path="fixture/definition.json",
        definition_sha256=SHA,
    )
    assert footprint == {
        "model": "declared_tensor_io_footprint_v1",
        "definition_path": "fixture/definition.json",
        "definition_sha256": SHA,
        "aggregate_declared_tensor_bits": 152,
        "peak_declared_tensor_bits": 96,
        "aggregate_declared_tensor_elements": 13,
        "resolved_axis_volume": 36,
        "scalar_control_tuple": [["alpha", 7]],
        "tensor_count": 3,
    }


@pytest.mark.parametrize("axes, message", [
    ({"alpha": 1}, "variable axis is unbound"),
    ({"B": 1, "unknown": 2}, "not a declared scalar input"),
    ({"B": -1}, "capture axis is invalid"),
])
def test_definition_footprint_rejects_unbound_or_open_axes(axes, message):
    definition = {
        "axes": {"B": {"type": "var"}},
        "inputs": {"alpha": {"shape": None, "dtype": "int32"}},
        "outputs": {"result": {"shape": ["B"], "dtype": "float16"}},
    }
    with pytest.raises(rep.RepresentativeMatrixError, match=message):
        rep.definition_footprint(
            definition=definition, capture_axes=axes,
            definition_path="fixture.json", definition_sha256=SHA,
        )


def test_definition_footprint_rejects_unknown_dtype():
    with pytest.raises(rep.RepresentativeMatrixError, match="dtype is unsupported"):
        rep.definition_footprint(
            definition={
                "axes": {},
                "inputs": {},
                "outputs": {"x": {"shape": None, "dtype": "complex64"}},
            },
            capture_axes={}, definition_path="fixture.json", definition_sha256=SHA,
        )


def _complete_outcome(control: dict) -> dict:
    return {
        "nvbit_trace": {"status": "complete"},
        "ncu_profile": {"status": "complete", "metric_scope": "cycles_only_binding"},
        "control": control,
    }


@pytest.mark.parametrize("control", [
    {"status": "complete", "reason": None},
    {
        "status": "typed_unavailable_no_stable_static_module",
        "reason": "origin_unresolved",
    },
])
def test_trace_and_ncu_are_admitted_with_complete_or_typed_control(control):
    rep.validate_representative_capture_outcome(_complete_outcome(control))


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("control"),
    lambda value: value["nvbit_trace"].update(status="partial"),
    lambda value: value["ncu_profile"].update(metric_scope="all_metrics"),
    lambda value: value["control"].update(status="unavailable"),
    lambda value: value["control"].update(reason="unknown"),
    lambda value: value.update(extra=True),
])
def test_missing_partial_or_open_capture_outcomes_fail_closed(mutation):
    value = _complete_outcome({
        "status": "typed_unavailable_no_stable_static_module",
        "reason": "origin_unresolved",
    })
    mutation(value)
    with pytest.raises(rep.RepresentativeMatrixError):
        rep.validate_representative_capture_outcome(value)


def test_trace_contract_is_closed_by_schema():
    validator_schema = {
        "$schema": SCHEMA["$schema"],
        "$defs": SCHEMA["$defs"],
        "$ref": "#/$defs/traceContract",
    }
    jsonschema.validate(rep.REPRESENTATIVE_CAPTURE_CONTRACT, validator_schema)
    for field, unsafe in (
        ("control_outcome_required", False),
        ("control_unavailability_invalidates_trace_ncu", True),
        ("partial_trace_allowed", True),
    ):
        mutated = copy.deepcopy(rep.REPRESENTATIVE_CAPTURE_CONTRACT)
        mutated[field] = unsafe
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(mutated, validator_schema)


def test_chunking_is_deterministic_and_exactly_once():
    selections = [
        {
            "slug": f"problem_{index:03d}",
            "source_workload_uuid": f"uuid-{index:03d}",
            "source_matrix_row_key": {"variant_id": "native"},
            "cohort": "native" if index < 17 else (
                "reduced_proxy" if index < 32 else "partial_native_row"
            ),
        }
        for index in range(33)
    ]
    chunks = rep._chunks(selections)
    assert [len(chunk["selection_ids"]) for chunk in chunks] == [16, 1, 15, 1]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "sol-representative-native-000", "sol-representative-native-001",
        "sol-representative-reduced-proxy-000",
        "sol-representative-partial-native-row-000",
    ]
    flattened = [item for chunk in chunks for item in chunk["selection_ids"]]
    assert {item["slug"] for item in flattened} == {
        item["slug"] for item in selections
    }


def test_schema_and_default_artifact_are_distinct_from_full_matrix():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    assert rep.DEFAULT_OUTPUT_PATH.as_posix() == (
        "data/local_labels/campaign/sol_representative_capture_matrix.v1.json"
    )
    assert rep.DEFAULT_OUTPUT_PATH != rep.full_builder.DEFAULT_OUTPUT_PATH
    assert rep.SCHEMA_PATH != rep.full_builder.SCHEMA_PATH
