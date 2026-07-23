"""Fixture-only contracts for the supplemental SOL workload-row matrix."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import build_sol_workload_capture_matrix as wm
from tools.local_labels import sol_workload_coverage_inventory as historical
from tools.local_labels import sol_live_workload_coverage_inventory as live


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / wm.SCHEMA_PATH).read_text())
SHA = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _policy() -> dict:
    return {
        "schema_version": 1,
        "policy_id": "sm86_sol_supplemental_unsampled_v1",
        "default_max_trace_bytes": 17179869184,
        "default_max_wall_time_s": 7200,
        "sampling_policy": "off",
        "derivation": "explicit_slug_or_policy_default_ceiling",
        "breach_disposition": "hard_refusal_preserve_complete_trace",
        "partial_trace_allowed": False,
        "source_basis": "supplemental_raw_staging_ceiling_full_trace_retention",
        "path": wm.SUPPLEMENTAL_BUDGET_POLICY_PATH.as_posix(),
        "sha256": "b" * 64,
        "trace_budgets_path": wm.TRACE_BUDGETS_PATH.as_posix(),
        "trace_budgets_sha256": "c" * 64,
    }


def _control() -> dict:
    return {
        "policy": "post_trace_module_or_typed",
        "module_discovery": "nvbit_runtime_module_inventory",
        "required_outcomes": [
            "exact_accepted_launch_runtime_binary_controls_emitted",
            "exact_accepted_launch_static_rebind_controls_emitted",
            "typed_unavailable_no_stable_static_module",
        ],
        "static_rebind": {
            "policy": "preserved_module_exact_function",
            "toolkit_version": "V13.2",
            "allowed_code_arches": ["sm_80", "sm_86"],
            "nvcc_path": "/usr/local/cuda-13.2/bin/nvcc",
            "nvcc_sha256": "d" * 64,
            "cuobjdump_path": "/usr/local/cuda-13.2/bin/cuobjdump",
            "cuobjdump_sha256": "e" * 64,
            "ncu_report_binding": "raw_text_section_exactly_once",
        },
    }


def _flashinfer_contracts() -> dict[str, dict]:
    base = {
        "policy": "verified_fib_only",
        "environment_variable": "FLASHINFER_TRACE_DIR",
        "data_annex_manifest_path": "evidence/manifests/data_annex.sha256",
        "data_annex_manifest_sha256": "6" * 64,
        "filtered_manifest_sha256": "7" * 64,
        "file_count": 5244,
        "safetensors_count": 2611,
    }
    return {
        category: {
            **base,
            "category": category,
            "injection_policy": (
                "verified_absolute_root" if category == "FIB"
                else "forbidden_remove_ambient"
            ),
            "root": (
                "data/annex/flashinfer_trace" if category == "FIB" else None
            ),
        }
        for category in wm.SUBSET_BY_CATEGORY
    }


def _native_fixture(
    tmp_path: Path, *, statuses: tuple[str, ...] = ("PASSED", "PASSED"),
    diagnostics_complete: bool = True,
    command_present: bool = True,
) -> tuple[list[dict], list[dict], dict, dict, Path]:
    slug = "problem_999_fixture"
    category = "L1"
    language = "python"
    definition_name = "fixture_definition"
    solution_name = "fixture_solution"
    benchmark_rel = (
        wm.SOL_BENCHMARK_ROOT / wm.SUBSET_BY_CATEGORY[category] / definition_name
    )
    benchmark = tmp_path / benchmark_rel
    benchmark.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "uuid": f"00000000-0000-4000-8000-{index:012d}",
            "axes": {"n": 8 - index},
            "inputs": {"value": {"type": "random"}},
            "tolerance": (
                {"max_atol": 1e-4, "max_rtol": 2e-3}
                if index % 2 == 0 else
                {
                    "max_atol": 3e-4,
                    "max_rtol": 4e-3,
                    "required_match_ratio": 0.98,
                }
            ),
        }
        for index in range(len(statuses))
    ]
    workload = benchmark / "workload.jsonl"
    workload.write_bytes(b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n"
        for row in rows
    ))
    _write_json(benchmark / "definition.json", {"name": definition_name})
    solution_rel = (
        wm.SOL_BASELINES_ROOT / category / slug / language / "solution.json"
    )
    _write_json(
        tmp_path / solution_rel,
        {"name": solution_name, "definition": definition_name},
    )
    executable = tmp_path / wm.SOL_EXECUTABLE_PATH
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    runnability_path = tmp_path / wm.RUNNABILITY_PATH
    _write_json(runnability_path, {"fixture": True})
    parsed = wm._workload_rows(workload)
    present_count = len(rows) if diagnostics_complete else len(rows) - 1
    diagnostics = []
    for index, (row, source, status) in enumerate(zip(rows, parsed, statuses)):
        present = index < present_count
        trace_workload = copy.deepcopy(row)
        trace_tolerance = trace_workload["tolerance"]
        trace_tolerance.pop("required_match_ratio", None)
        trace_tolerance.setdefault("required_matched_ratio", 0.99)
        trace_tolerance.setdefault("max_error_cap", None)
        trace_tolerance.setdefault("allow_negative_inf", False)
        diagnostics.append({
            "workload_index": index,
            "source_line": source["source_line"],
            "uuid": source["source_workload_uuid"],
            "axes": copy.deepcopy(source["source_axes"]),
            "workload_sha256": source["source_row_sha256"],
            "diagnostic_present": present,
            "trace_index": index if present else None,
            "status": status if present else None,
            "trace": ({
                "workload": trace_workload,
                "definition": definition_name,
                "solution": solution_name,
                "evaluation": {"status": status},
            } if present else None),
        })
    passed = sum(
        row["diagnostic_present"] and row["status"].lower() == "passed"
        for row in diagnostics
    )
    stdout = "fixture stdout\n"
    stderr = ""
    all_passed = passed == len(rows) and diagnostics_complete
    record = {
        "slug": slug,
        "category": category,
        "language": language,
        "status": "runnable" if all_passed else "correctness_failed",
        "returncode": 0 if all_passed else 1,
        "detail": (
            f"{len(rows)}/{len(rows)} workloads passed"
            if all_passed else f"{passed}/{len(rows)} workloads passed"
        ),
        "command": [
            "tools/sol_execbench/.venv/bin/python",
            "-m", "sol_execbench.cli.main",
            "--definition", (benchmark_rel / "definition.json").as_posix(),
            "--workload", (benchmark_rel / "workload.jsonl").as_posix(),
            "--solution", solution_rel.as_posix(),
            "--json",
            "--config", wm.PROBE_CONFIG_PATH.as_posix(),
            "--timeout", "1500",
        ],
        "stdout": stdout,
        "stdout_sha256": wm.sha256_bytes(stdout.encode()),
        "stderr": stderr,
        "stderr_sha256": wm.sha256_bytes(stderr.encode()),
        "expected_workloads": len(rows),
        "returned_workloads": present_count,
        "passed_workloads": passed,
        "diagnostics_complete": diagnostics_complete,
        "workload_diagnostics": diagnostics,
        "unexpected_workload_diagnostics": [],
    }
    if not command_present:
        record["command"] = None
    runnability = {
        "native_env_hash": "1" * 64,
        "resume_identity_sha256": "2" * 64,
        "probe_protocol_sha256": "3" * 64,
        "probe_config_sha256": "4" * 64,
        "sol_execbench_package_sha256": "5" * 64,
    }
    variants, matrix_rows = wm._native_problem_contract(
        repo_root=tmp_path,
        record=record,
        roster_row={"slug": slug, "category": category, "language": language},
        runnability=runnability,
        runnability_path=runnability_path,
        policy=_policy(),
        explicit_budgets={},
        control_bits=_control(),
        flashinfer_contracts=_flashinfer_contracts(),
    )
    return variants, matrix_rows, record, runnability, runnability_path


def _matrix_document(variants: list[dict], rows: list[dict]) -> dict:
    chunks = wm._chunks(sorted(rows, key=wm._row_sort_key))
    document = {
        "schema_version": 2,
        "artifact_id": "sm86_sol_workload_capture_matrix_live_v2",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "source_commit": "1" * 40,
        "input_pins": {
            "sol_native_env_fingerprint": {
                "path": wm.LIVE_NATIVE_ENV_FINGERPRINT_PATH.as_posix(),
                "sha256": SHA,
                "native_env_hash": "1" * 64,
            },
        },
        "canonical_matrix": {
            "path": "canonical.json", "sha256": SHA,
            "source_commit": "1" * 40, "matrix_seal_sha256": SHA,
        },
        "canonical_authority": {
            "path": "authority.json", "sha256": SHA, "roster_id": "fixture",
        },
        "runnability_authority": {
            "path": "sol_runnability_sm86.json", "sha256": SHA,
            "owner_marker": {
                "name": ".sol-runnability-owner-v1.json",
                "sha256": SHA, "owner_seal_sha256": SHA,
            },
            "roster_path": wm.ROSTER_PATH.as_posix(), "roster_sha256": SHA,
            "roster_rows": 201, "native_env_hash": "1" * 64,
            "probe_protocol_sha256": "3" * 64,
            "probe_config_sha256": "4" * 64,
            "sol_execbench_package_sha256": "5" * 64,
            "data_annex_manifest_sha256": SHA,
            "flashinfer_corpus_manifest_sha256": SHA,
            "flashinfer_corpus_file_count": 5244,
            "flashinfer_corpus_blob_count": 2611,
            "flashinfer_corpus_verified": True,
            "resume_identity_sha256": "2" * 64,
        },
        "supplemental_budget_policy": _policy(),
        "reduced_proxy_authority": {
            "path": "sol_shrink_provenance_sm86.json",
            "sha256": SHA,
            "owner_marker": {
                "name": ".sol-shrink-owner-v1.json",
                "sha256": SHA, "owner_seal_sha256": SHA,
            },
            "provenance_contract_version": 3,
        },
        "claim_policy": {
            "native_cohort": "native_workload_rows_only",
            "partial_native_cohort": (
                "passed_native_rows_only_no_problem_level_runnability_claim"
            ),
            "reduced_proxy_cohort": wm.REDUCED_PROXY_COHORT_POLICY,
            "cross_cohort_aggregation": "forbidden_without_explicit_stratification",
            "canonical_344_cell_matrix_impact": "none_additive_supplement_only",
        },
        "chunk_policy": {
            "rows_per_chunk": 16, "cohorts_may_mix": False,
            "ordering": "representatives_across_all_cohorts_before_depth",
            "tier_order": ["representative", "depth"],
        },
        "counts": {
            "problems": 1,
            "variants": len(variants),
            "native_variants": sum(v["cohort"] == "native" for v in variants),
            "partial_native_variants": sum(
                v["cohort"] == "partial_native_row" for v in variants
            ),
            "reduced_variants": sum(
                v["cohort"] == "reduced_proxy" for v in variants
            ),
            "rows": len(rows),
            "native_rows": sum(r["cohort"] == "native" for r in rows),
            "partial_native_rows": sum(
                r["cohort"] == "partial_native_row" for r in rows
            ),
            "reduced_proxy_rows": sum(
                r["cohort"] == "reduced_proxy" for r in rows
            ),
            "chunks": len(chunks),
        },
        "variants": variants,
        "rows": sorted(rows, key=wm._row_sort_key),
        "chunks": chunks,
    }
    document["matrix_seal_sha256"] = wm.canonical_sha256(document)
    return document


def _reduced_fixture(
    tmp_path: Path, monkeypatch, *, definition_proxy: bool = False,
    equal_workload: bool = False, mutation=None,
) -> tuple[list[dict], list[dict], dict]:
    slug = "problem_997_geometry_fixture"
    category = "L1"
    language = "python"
    definition_name = "geometry_fixture"
    benchmark_rel = (
        wm.SOL_BENCHMARK_ROOT / wm.SUBSET_BY_CATEGORY[category] / definition_name
    )
    benchmark = tmp_path / benchmark_rel
    benchmark.mkdir(parents=True)
    native_value = {
        "uuid": "00000000-0000-4000-8000-000000000000",
        "axes": {"n": 16},
    }
    reduced_value = {
        "uuid": native_value["uuid"], "axes": {"n": 8},
    }
    if equal_workload:
        reduced_value = copy.deepcopy(native_value)
    native_payload = json.dumps(native_value, separators=(",", ":")).encode()
    reduced_payload = json.dumps(reduced_value, separators=(",", ":")).encode()
    native_path = benchmark / "workload.jsonl"
    native_path.write_bytes(native_payload + b"\n")
    definition_path = benchmark / "definition.json"
    native_definition = {
        "name": definition_name,
        "axes": {"num_experts": {"type": "const", "value": 256}},
        "inputs": {"x": {"shape": ["num_experts"], "dtype": "float32"}},
    }
    _write_json(definition_path, native_definition)
    reduced_rel = Path("fixture.jsonl")
    reduced_root = tmp_path / "reduced-owner"
    reduced_root.mkdir()
    reduced_path = reduced_root / reduced_rel
    reduced_path.write_bytes(reduced_payload + b"\n")
    definition_delta = []
    reduced_definition_ref = None
    reduced_definition_sha256 = None
    effective_definition_sha256 = wm.sha256_file(definition_path)
    if definition_proxy:
        reduced_definition_ref = "fixture.definition.json"
        reduced_definition = copy.deepcopy(native_definition)
        reduced_definition["axes"]["num_experts"]["value"] = 64
        reduced_definition_path = reduced_root / reduced_definition_ref
        _write_json(reduced_definition_path, reduced_definition)
        reduced_definition_sha256 = wm.sha256_file(reduced_definition_path)
        effective_definition_sha256 = reduced_definition_sha256
        definition_delta = [{
            "json_pointer": "/axes/num_experts/value",
            "from": 256,
            "to": 64,
        }]
    solution_rel = (
        wm.SOL_BASELINES_ROOT / category / slug / language / "solution.json"
    )
    solution_path = tmp_path / solution_rel
    _write_json(
        solution_path,
        {"name": "geometry_solution", "definition": definition_name},
    )
    executable = tmp_path / wm.SOL_EXECUTABLE_PATH
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    model_id = "fixture_geometry_model_v1"
    native_sha = wm.sha256_bytes(native_payload)
    reduced_sha = wm.sha256_bytes(reduced_payload)
    variant_id = wm.producer_variant_id(
        slug=slug, source_uuid=native_value["uuid"],
        native_sha256=native_sha, reduced_sha256=reduced_sha,
        native_definition_sha256=wm.sha256_file(definition_path),
        reduced_definition_sha256=reduced_definition_sha256,
        model_id=model_id,
    )
    plan = {
        "slug": slug,
        "source_workload_uuid": native_value["uuid"],
        "source_workload_index": 0,
        "variant_id": variant_id,
        "planning_status": "proxy_planned",
        "claim_scope": "reduced_shape_only",
        "native_predictive": False,
        "row_sha256_scope": "exact_json_line_bytes_excluding_lf_only",
        "native_row_sha256": native_sha,
        "reduced_row_sha256": reduced_sha,
        "native_axes": native_value["axes"],
        "reduced_axes": reduced_value["axes"],
        "axes_delta": [{"axis": "n", "from": 16, "to": 8, "factor": 2.0}],
        "definition_ref": (benchmark_rel / "definition.json").as_posix(),
        "definition_sha256": wm.sha256_file(definition_path),
        "reduced_definition_ref": reduced_definition_ref,
        "reduced_definition_sha256": reduced_definition_sha256,
        "definition_delta": copy.deepcopy(definition_delta),
        "model": {"model_id": model_id},
        "reduced_run_status": "runnable",
        "correctness_result": "passed",
    }
    if equal_workload:
        plan["axes_delta"] = []
    effective_definition_ref = (
        reduced_definition_ref
        if reduced_definition_ref is not None
        else (benchmark_rel / "definition.json").as_posix()
    )
    evidence_inputs = {
        "definition_input": {
            "path": effective_definition_ref,
            "sha256": effective_definition_sha256,
        },
        "workload_input": {
            "path": reduced_rel.as_posix(),
            "sha256": wm.sha256_file(reduced_path),
        },
        "solution_input": {
            "path": solution_rel.as_posix(),
            "sha256": wm.sha256_file(solution_path),
        },
    }
    cell = {
        "slug": slug,
        "provenance_contract_version": 3,
        "classification": "reduced_representative",
        "claim_scope": "reduced_shape_only",
        "native_predictive": False,
        "reduced_run_status": "runnable",
        "correctness_result": "passed",
        "definition_ref": (benchmark_rel / "definition.json").as_posix(),
        "definition_sha256": wm.sha256_file(definition_path),
        "reduced_definition_ref": reduced_definition_ref,
        "reduced_definition_sha256": reduced_definition_sha256,
        "definition_delta": copy.deepcopy(definition_delta),
        "reduced_workload_ref": reduced_rel.name,
        "reduced_workload_sha256": wm.sha256_file(reduced_path),
        "native_shape_ref": (benchmark_rel / "workload.jsonl").as_posix(),
        "native_workload_sha256": wm.sha256_file(native_path),
        "source_row_count": 1,
        "cohort_id": "6" * 64,
        "row_plans": [plan],
        "cohort_execution_evidence": {
            **evidence_inputs,
            "detail": "1/1 workloads passed",
        },
    }
    provenance = {
        "provenance_contract_version": 3,
        "publication_state": "complete",
        "native_env_hash": "1" * 64,
        "reduced_output_owner": {
            "name": ".sol-reduced-output-owner-v1.json",
            "sha256": "2" * 64,
            "owner_seal_sha256": "3" * 64,
        },
        "cells": [cell],
    }
    if mutation is not None:
        mutation(provenance, reduced_root)
    provenance_path = reduced_root / "sol_shrink_provenance_sm86.json"
    _write_json(provenance_path, provenance)
    root_identity = {"canonical_path": str(reduced_root)}
    monkeypatch.setattr(
        wm.shrink, "validate_external_reduced_output_root",
        lambda path: root_identity if path == reduced_root else None,
    )
    monkeypatch.setattr(
        wm.shrink, "_external_output_path",
        lambda _identity, name: reduced_root / name,
    )
    monkeypatch.setattr(
        wm.shrink, "_read_bound_direct_child",
        lambda _identity, name: (reduced_root / name).read_bytes(),
    )
    variants, rows = wm._reduced_variants(
        repo_root=tmp_path, provenance=provenance,
        provenance_path=provenance_path,
        provenance_sha256=wm.sha256_file(provenance_path),
        reduced_output_root=reduced_root,
        roster={slug: {"slug": slug, "category": category, "language": language}},
        policy=_policy(), explicit_budgets={}, control_bits=_control(),
        flashinfer_contracts=_flashinfer_contracts(),
    )
    return variants, rows, provenance


def test_full_native_requires_all_uuid_diagnostics_passed(tmp_path):
    variants, rows, _record, _runnability, _path = _native_fixture(tmp_path)
    assert len(variants) == 1
    assert len(rows) == 2
    variant = variants[0]
    assert (variant["cohort"], variant["variant_id"]) == ("native", "native")
    assert variant["problem_level_runnable"] is True
    assert variant["native_predictive"] is True
    invocation = variant["invocation"]
    logical = invocation["logical_workload"]
    assert logical["value"] == "workloads"
    assert logical["benchmark_tree_sha256"] == wm.tree_sha256(
        tmp_path / logical["benchmark_dir"]
    )
    assert invocation["evaluation_cardinality_contract"] == (
        "all_effective_workloads_passed_exact"
    )
    assert invocation["trace_cardinality_contract"] == (
        wm.SUPPLEMENTAL_TRACE_CARDINALITY_CONTRACT
    )
    assert invocation["capture_launch_contract"] == (
        wm.SUPPLEMENTAL_CAPTURE_LAUNCH_CONTRACT
    )
    assert invocation["capture_target"] == wm.SUPPLEMENTAL_CAPTURE_TARGET
    assert variant["adapter_contract"]["environment"] == wm.EXACT_SOL_ENVIRONMENT
    unsigned = dict(variant)
    seal = unsigned.pop("variant_contract_sha256")
    assert seal == wm.canonical_sha256(unsigned)
    assert all(row["source_row_sha256"] == row["capture_row_sha256"] for row in rows)


def test_supplemental_contract_does_not_claim_one_dynamic_kernel_launch(tmp_path):
    variants, _rows, *_ = _native_fixture(tmp_path)
    invocation = variants[0]["invocation"]
    launch_contract = invocation["capture_launch_contract"]
    capture_target = invocation["capture_target"]
    assert launch_contract["trace_candidate_call_count"] == 1
    assert launch_contract["trace_dynamic_launch_count_source"] == (
        "runtime_inventory"
    )
    assert launch_contract["trace_dynamic_launch_scope"] == "all_in_profiler_roi"
    assert capture_target["candidate_user_fn_calls"] == 1
    assert capture_target["dynamic_launch_count_source"] == "runtime_inventory"
    assert capture_target["dynamic_launch_scope"] == "all_in_profiler_roi"
    forbidden = {
        "trace_launch_count", "label_trace_launch_index",
        "capture_launches", "candidate_kernel_ordinal",
    }
    assert forbidden.isdisjoint(launch_contract)
    assert forbidden.isdisjoint(capture_target)
    serialized = json.dumps(invocation, sort_keys=True)
    assert "one_pinned_candidate_kernel_launch_exact" not in serialized


def test_partial_native_admits_only_passed_row_without_problem_claim(tmp_path):
    variants, rows, record, _runnability, _path = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED")
    )
    assert record["diagnostics_complete"] is True
    assert record["returned_workloads"] == record["expected_workloads"] == 2
    assert len(variants) == len(rows) == 1
    variant, row = variants[0], rows[0]
    assert variant["cohort"] == row["cohort"] == "partial_native_row"
    assert variant["variant_id"] == row["key"]["variant_id"]
    assert len(variant["variant_id"]) == 64
    assert variant["claim_scope"] == "partial_native_workload_row_only"
    assert variant["problem_level_runnable"] is False
    assert variant["native_predictive"] is False
    evidence = variant["runnability_evidence"]
    assert evidence["diagnostics_complete"] is True
    assert evidence["expected_workloads"] == 2
    assert evidence["passed_workloads"] == 1
    assert evidence["selected_source_workload_uuid"] == row["source_workload_uuid"]
    assert evidence["selected_diagnostic_sha256"] == row["row_evidence_sha256"]


def test_incomplete_problem_retains_independently_bound_passed_row(tmp_path):
    variants, rows, record, _runnability, _path = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED"), diagnostics_complete=False
    )
    assert record["diagnostics_complete"] is False
    assert len(variants) == len(rows) == 1
    assert variants[0]["cohort"] == "partial_native_row"
    assert variants[0]["problem_level_runnable"] is False
    assert variants[0]["runnability_evidence"]["diagnostics_complete"] is False


def test_commandless_zero_passed_prerequisite_cell_is_typed_omission(tmp_path):
    variants, rows, record, _runnability, _path = _native_fixture(
        tmp_path, statuses=("FAILED", "FAILED"), command_present=False
    )
    assert record["passed_workloads"] == 0
    assert record["command"] is None
    assert variants == []
    assert rows == []


def test_trace_index_must_biject_all_source_rows(tmp_path):
    variants, _rows, record, runnability, runnability_path = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED")
    )
    assert variants
    record["workload_diagnostics"][1]["trace_index"] = 0
    with pytest.raises(
        wm.WorkloadMatrixError, match="diagnostic completeness mismatch"
    ):
        wm._native_problem_contract(
            repo_root=tmp_path, record=record,
            roster_row={
                "slug": "problem_999_fixture", "category": "L1",
                "language": "python",
            },
            runnability=runnability, runnability_path=runnability_path,
            policy=_policy(), explicit_budgets={}, control_bits=_control(),
            flashinfer_contracts=_flashinfer_contracts(),
        )


def test_executor_effective_workload_matches_pinned_tolerance_semantics():
    source = {
        "uuid": "12345678-1234-5123-8123-123456789abc",
        "axes": {"n": 1},
        "inputs": {"value": {"type": "custom"}},
        "tolerance": {
            "max_atol": 1e-5,
            "max_rtol": 0.05,
            "required_match_ratio": 0.98,
        },
    }
    expected = copy.deepcopy(source)
    expected["tolerance"] = {
        "max_atol": 1e-5,
        "max_rtol": 0.05,
        "required_matched_ratio": 0.99,
        "max_error_cap": None,
        "allow_negative_inf": False,
    }
    assert wm._executor_effective_workload(source) == expected
    assert source["tolerance"]["required_match_ratio"] == 0.98


def test_executor_effective_workload_adds_absent_tolerance_defaults():
    source = {
        "uuid": "12345678-1234-5123-8123-123456789abc",
        "axes": {"n": 1},
        "inputs": {"value": {"type": "random"}},
    }
    assert wm._executor_effective_workload(source)["tolerance"] == {
        "max_atol": 1e-2,
        "max_rtol": 1e-2,
        "required_matched_ratio": 0.99,
        "max_error_cap": None,
        "allow_negative_inf": False,
    }


def test_executor_effective_workload_rejects_ambiguous_ratio_spellings():
    source = {
        "uuid": "12345678-1234-5123-8123-123456789abc",
        "axes": {"n": 1},
        "inputs": {"value": {"type": "random"}},
        "tolerance": {
            "required_match_ratio": 0.98,
            "required_matched_ratio": 0.99,
        },
    }
    with pytest.raises(wm.WorkloadMatrixError, match="ambiguous matched-ratio"):
        wm._executor_effective_workload(source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_atol", 9.0),
        ("required_matched_ratio", 0.5),
        ("unexpected", True),
    ],
)
def test_trace_workload_effective_view_tamper_fails_closed(
    tmp_path, field, value,
):
    variants, _rows, record, runnability, runnability_path = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED")
    )
    assert variants
    trace_workload = record["workload_diagnostics"][0]["trace"]["workload"]
    if field == "unexpected":
        trace_workload[field] = value
    else:
        trace_workload["tolerance"][field] = value
    with pytest.raises(
        wm.WorkloadMatrixError, match="diagnostic trace binding mismatch",
    ):
        wm._native_problem_contract(
            repo_root=tmp_path, record=record,
            roster_row={
                "slug": "problem_999_fixture", "category": "L1",
                "language": "python",
            },
            runnability=runnability, runnability_path=runnability_path,
            policy=_policy(), explicit_budgets={}, control_bits=_control(),
            flashinfer_contracts=_flashinfer_contracts(),
        )


def test_native_contract_accepts_run_certified_command_and_rejects_legacy_shape(
    tmp_path,
):
    variants, _rows, record, runnability, runnability_path = _native_fixture(
        tmp_path
    )
    assert variants
    # The historical positional benchmark CLI is not what run_certified seals.
    record["command"] = [
        record["command"][0], "-m", "sol_execbench.cli.main",
        str((tmp_path / "tools/sol_execbench/data/benchmark/L1/fixture_definition").resolve()),
        "--solution", record["command"][8], "--json",
        "--config", record["command"][11], "--timeout", "1500",
    ]
    with pytest.raises(wm.WorkloadMatrixError, match="probe command binding mismatch"):
        wm._native_problem_contract(
            repo_root=tmp_path, record=record,
            roster_row={
                "slug": "problem_999_fixture", "category": "L1",
                "language": "python",
            },
            runnability=runnability, runnability_path=runnability_path,
            policy=_policy(), explicit_budgets={}, control_bits=_control(),
            flashinfer_contracts=_flashinfer_contracts(),
        )


def test_raw_jsonl_identity_drops_lf_only(tmp_path):
    row = {
        "uuid": "12345678-1234-5123-8123-123456789abc",
        "axes": {"n": 1},
    }
    raw = b"  " + json.dumps(row).encode() + b" \r"
    path = tmp_path / "workload.jsonl"
    path.write_bytes(b"// ignored\r\n" + raw + b"\n")
    parsed = wm._workload_rows(path)
    assert len(parsed) == 1
    assert parsed[0]["source_line"] == 2
    assert parsed[0]["source_row_sha256"] == wm.sha256_bytes(raw)
    assert parsed[0]["source_value"] == row
    assert parsed[0]["source_row_sha256"] != wm.sha256_bytes(raw.strip())


def test_hex32_uuid_and_empty_axes_are_exact_valid_identifiers(tmp_path):
    row = {"uuid": "0123456789abcdef0123456789abcdef", "axes": {}}
    path = tmp_path / "workload.jsonl"
    path.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    parsed = wm._workload_rows(path)
    assert parsed[0]["source_workload_uuid"] == row["uuid"]
    assert parsed[0]["source_axes"] == {}
    jsonschema.validate(row["uuid"], SCHEMA["$defs"]["uuid"])
    for invalid in (
        row["uuid"].upper(), "g" * 32, "0" * 31, "arbitrary-id",
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, SCHEMA["$defs"]["uuid"])


def test_duplicate_or_malformed_workload_identity_fails_closed(tmp_path):
    row = {"uuid": "12345678-1234-5123-8123-123456789abc", "axes": {"n": 1}}
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(wm.WorkloadMatrixError, match="duplicate workload UUID"):
        wm._workload_rows(duplicate)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps({"uuid": row["uuid"], "axes": {"n": True}}))
    with pytest.raises(wm.WorkloadMatrixError, match="integer-valued"):
        wm._workload_rows(malformed)


def test_real_corpus_uuid_union_and_scope_tripwires():
    benchmark_root = REPO_ROOT / historical.DEFAULT_INPUTS.benchmark_root
    problems = historical._load_benchmark_index(REPO_ROOT, benchmark_root)
    rfc_count = 0
    lowercase_hex32_count = 0
    row_counts: dict[str, int] = {}
    for slug, problem in problems.items():
        rows = wm._workload_rows(REPO_ROOT / problem["workload_path"])
        row_counts[slug] = len(rows)
        for row in rows:
            identity = row["source_workload_uuid"]
            assert wm.UUID_RE.fullmatch(identity) is not None
            if len(identity) == 32:
                assert identity == identity.lower()
                assert all(ch in "0123456789abcdef" for ch in identity)
                lowercase_hex32_count += 1
            else:
                rfc_count += 1
    scope, _payload = historical._load_port_scope(
        REPO_ROOT / historical.DEFAULT_INPUTS.port_scope
    )
    included, excluded = historical._validate_port_scope(scope, problems)
    assert (len(problems), sum(row_counts.values())) == (235, 3957)
    assert (rfc_count, lowercase_hex32_count) == (3928, 29)
    assert (len(excluded), sum(row_counts[slug] for slug in excluded)) == (
        34, 537
    )
    assert (len(included), sum(row_counts[slug] for slug in included)) == (
        201, 3420
    )


def test_unsampled_budget_never_weakens_explicit_slug_ceiling():
    explicit = {
        "problem_999_fixture": {
            "cell_id": "problem_999_fixture",
            "max_trace_bytes": 3_000_000_000,
            "max_wall_time_s": 8000,
            "sampling_policy": "legacy-value-is-not-inherited",
        }
    }
    budget = wm._trace_budget(
        "problem_999_fixture", policy=_policy(), explicit=explicit
    )
    assert budget["max_trace_bytes"] == 3_000_000_000
    assert budget["max_wall_time_s"] == 8000
    assert budget["sampling_policy"] == "off"
    assert budget["partial_trace_allowed"] is False
    assert budget["explicit_slug_budget_sha256"] == wm.canonical_sha256(
        explicit["problem_999_fixture"]
    )


def test_reduced_proxy_is_built_from_v3_provenance_not_canonical_cell(
    tmp_path, monkeypatch,
):
    variants, rows, provenance = _reduced_fixture(tmp_path, monkeypatch)
    assert len(variants) == len(rows) == 1
    variant, row = variants[0], rows[0]
    assert variant["cohort"] == row["cohort"] == "reduced_proxy"
    assert variant["claim_scope"] == row["claim_scope"] == (
        wm.REDUCED_PROXY_CLAIM_SCOPE
    )
    assert variant["variant_id"] == row["key"]["variant_id"]
    assert variant["native_predictive"] is False
    assert variant["native_shape"] is False
    assert variant["problem_level_runnable"] is False
    assert "canonical_authority_cell_sha256" not in variant
    assert "canonical_invocation_sha256" not in variant
    invocation = variant["invocation"]
    assert invocation["logical_workload"]["value"] == "reduced_representative"
    assert "--workload" in invocation["trace_app_args"]
    assert invocation["logical_workload"]["evaluation_evidence"][
        "cell_sha256"
    ] == wm.canonical_sha256(provenance["cells"][0])
    assert row["source_workload_index"] == row["capture_workload_index"] == 0
    assert row["source_row_sha256"] != row["capture_row_sha256"]
    assert row["launch_completeness_contract"] == wm.LAUNCH_COMPLETENESS_CONTRACT


def test_definition_proxy_binds_external_definition_and_allows_equal_row(
    tmp_path, monkeypatch,
):
    variants, rows, provenance = _reduced_fixture(
        tmp_path, monkeypatch, definition_proxy=True, equal_workload=True,
    )
    assert len(variants) == len(rows) == 1
    variant, row = variants[0], rows[0]
    cell = provenance["cells"][0]
    logical = variant["invocation"]["logical_workload"]
    assert variant["native_predictive"] is False
    assert variant["claim_scope"] == row["claim_scope"] == (
        wm.REDUCED_PROXY_CLAIM_SCOPE
    )
    assert row["source_row_sha256"] == row["capture_row_sha256"]
    assert logical["definition_file"] == cell["reduced_definition_ref"]
    assert logical["definition_file_sha256"] == cell[
        "reduced_definition_sha256"
    ]
    assert logical["native_definition_file"] == cell["definition_ref"]
    assert logical["native_definition_file_sha256"] == cell[
        "definition_sha256"
    ]
    assert logical["external_definition_source"] == {
        "kind": "shrink_external_definition_child_v1",
        "name": cell["reduced_definition_ref"],
        "sha256": cell["reduced_definition_sha256"],
        "size_bytes": (tmp_path / "reduced-owner" /
                       cell["reduced_definition_ref"]).stat().st_size,
        "owner_marker": provenance["reduced_output_owner"],
        "runtime_rebind": "campaign_retained_immutable_child",
    }
    assert variant["adapter_contract"]["forwarded_argv"][1] == cell[
        "reduced_definition_ref"
    ]
    reduced_provenance = variant["reduced_provenance"]
    assert reduced_provenance["native_definition_file_sha256"] == cell[
        "definition_sha256"
    ]
    assert reduced_provenance["reduced_definition_file_sha256"] == cell[
        "reduced_definition_sha256"
    ]
    assert reduced_provenance["definition_delta"] == cell["definition_delta"]
    plan = cell["row_plans"][0]
    assert variant["variant_id"] == wm.producer_variant_id(
        slug=cell["slug"], source_uuid=plan["source_workload_uuid"],
        native_sha256=plan["native_row_sha256"],
        reduced_sha256=plan["reduced_row_sha256"],
        native_definition_sha256=cell["definition_sha256"],
        reduced_definition_sha256=cell["reduced_definition_sha256"],
        model_id=plan["model"]["model_id"],
    )
    document = _matrix_document(variants, rows)
    jsonschema.validate(document, SCHEMA)
    missing_native = copy.deepcopy(document)
    del missing_native["variants"][0]["invocation"]["logical_workload"][
        "native_definition_file_sha256"
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_native, SCHEMA)
    malformed_delta = copy.deepcopy(document)
    malformed_delta["variants"][0]["reduced_provenance"][
        "definition_delta"
    ][0]["undeclared"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(malformed_delta, SCHEMA)
    stale_claim = copy.deepcopy(document)
    stale_claim["variants"][0]["claim_scope"] = (
        "reduced_proxy_workload_row_only"
    )
    stale_claim["rows"][0]["claim_scope"] = (
        "reduced_proxy_workload_row_only"
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(stale_claim, SCHEMA)


def test_equal_workload_without_definition_change_is_not_a_proxy(
    tmp_path, monkeypatch,
):
    with pytest.raises(wm.WorkloadMatrixError, match="row binding mismatch"):
        _reduced_fixture(
            tmp_path, monkeypatch, definition_proxy=False, equal_workload=True,
        )


def test_definition_proxy_hash_drift_fails_closed(tmp_path, monkeypatch):
    def mutation(_provenance, root):
        (root / "fixture.definition.json").write_text("{}\n")

    with pytest.raises(
        wm.WorkloadMatrixError, match="reduced definition binding mismatch",
    ):
        _reduced_fixture(
            tmp_path, monkeypatch, definition_proxy=True, equal_workload=True,
            mutation=mutation,
        )


@pytest.mark.parametrize("change_name", [False, True])
def test_definition_proxy_rejects_undeclared_or_identity_mutation(
    tmp_path, monkeypatch, change_name,
):
    def mutation(provenance, root):
        cell = provenance["cells"][0]
        plan = cell["row_plans"][0]
        path = root / cell["reduced_definition_ref"]
        definition = json.loads(path.read_text())
        if change_name:
            definition["name"] = "wrong_definition"
        else:
            definition["inputs"]["x"]["dtype"] = "float16"
        _write_json(path, definition)
        digest = wm.sha256_file(path)
        cell["reduced_definition_sha256"] = digest
        plan["reduced_definition_sha256"] = digest
        cell["cohort_execution_evidence"]["definition_input"]["sha256"] = digest
        plan["variant_id"] = wm.producer_variant_id(
            slug=cell["slug"], source_uuid=plan["source_workload_uuid"],
            native_sha256=plan["native_row_sha256"],
            reduced_sha256=plan["reduced_row_sha256"],
            native_definition_sha256=cell["definition_sha256"],
            reduced_definition_sha256=digest,
            model_id=plan["model"]["model_id"],
        )

    with pytest.raises(
        wm.WorkloadMatrixError, match="reduced definition contract mismatch",
    ):
        _reduced_fixture(
            tmp_path, monkeypatch, definition_proxy=True, equal_workload=True,
            mutation=mutation,
        )


def test_reduced_proxy_schema_and_row_identity_are_closed(tmp_path, monkeypatch):
    variants, rows, _provenance = _reduced_fixture(tmp_path, monkeypatch)
    document = _matrix_document(variants, rows)
    jsonschema.validate(document, SCHEMA)
    mutated = copy.deepcopy(document)
    mutated["variants"][0]["native_predictive"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, SCHEMA)
    mutated = copy.deepcopy(document)
    mutated["rows"][0]["launch_completeness_contract"][
        "ncu_label_requirement"
    ] = "ordinal_zero_only"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, SCHEMA)


def test_schema_accepts_full_and_partial_contracts(tmp_path):
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    full, full_rows, *_ = _native_fixture(tmp_path / "full")
    partial, partial_rows, *_ = _native_fixture(
        tmp_path / "partial", statuses=("PASSED", "FAILED")
    )
    document = _matrix_document(full + partial, full_rows + partial_rows)
    # Use distinct fixture slugs so the synthetic top-level key set is unique.
    partial_variant = document["variants"][1]
    partial_variant["slug"] = "problem_998_partial_fixture"
    for row in document["rows"]:
        if row["cohort"] == "partial_native_row":
            row["key"]["slug"] = partial_variant["slug"]
    document["chunks"] = wm._chunks(sorted(document["rows"], key=wm._row_sort_key))
    document["counts"]["problems"] = 2
    document["counts"]["chunks"] = len(document["chunks"])
    document["matrix_seal_sha256"] = wm.canonical_sha256({
        key: value for key, value in document.items() if key != "matrix_seal_sha256"
    })
    jsonschema.validate(document, SCHEMA)


def test_schema_allows_partial_with_independently_bound_passed_row(tmp_path):
    variants, rows, *_ = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED")
    )
    document = _matrix_document(variants, rows)
    document["variants"][0]["runnability_evidence"][
        "diagnostics_complete"
    ] = False
    document["variants"][0]["invocation"]["logical_workload"][
        "evaluation_evidence"
    ]["diagnostics_complete"] = False
    jsonschema.validate(document, SCHEMA)


def test_chunks_are_single_cohort_and_cover_keys_once(tmp_path):
    native, native_rows, *_ = _native_fixture(tmp_path / "native")
    partial, partial_rows, *_ = _native_fixture(
        tmp_path / "partial", statuses=("PASSED", "FAILED")
    )
    del native, partial
    rows = sorted(native_rows + partial_rows, key=wm._row_sort_key)
    chunks = wm._chunks(rows)
    scheduled = [key for chunk in chunks for key in chunk["rows"]]
    assert {wm.canonical_sha256(key) for key in scheduled} == {
        wm.canonical_sha256(row["key"]) for row in rows
    }
    assert {chunk["cohort"] for chunk in chunks} == {
        "native", "partial_native_row"
    }
    for chunk in chunks:
        assert chunk["membership_sha256"] == wm.canonical_sha256(chunk["rows"])
        assert 1 <= chunk["row_count"] <= 16


def test_chunk_schedule_is_breadth_first_across_problem_variants():
    rows = []
    for slug in ("problem_001_first", "problem_002_second"):
        for depth in range(3):
            rows.append({
                "cohort": "native",
                "key": {
                    "slug": slug,
                    "source_workload_uuid": (
                        f"00000000-0000-4000-8000-{depth:012d}"
                    ),
                    "variant_id": "native",
                },
            })
    chunks = wm._chunks(sorted(rows, key=wm._row_sort_key))
    scheduled = [key for chunk in chunks for key in chunk["rows"]]
    assert [(key["slug"], key["source_workload_uuid"][-1]) for key in scheduled] == [
        ("problem_001_first", "0"), ("problem_002_second", "0"),
        ("problem_001_first", "1"), ("problem_002_second", "1"),
        ("problem_001_first", "2"), ("problem_002_second", "2"),
    ]


def test_representative_chunks_interleave_cohorts_before_any_depth_chunk():
    rows = []
    for cohort, slug, count in (
        ("native", "problem_001_native", 2),
        ("partial_native_row", "problem_002_partial", 1),
        ("reduced_proxy", "problem_003_reduced", 1),
    ):
        for depth in range(count):
            rows.append({
                "cohort": cohort,
                "key": {
                    "slug": slug,
                    "source_workload_uuid": (
                        f"00000000-0000-4000-8000-{depth:012d}"
                    ),
                    "variant_id": "native" if cohort == "native" else "a" * 64,
                },
            })
    chunks = wm._chunks(sorted(rows, key=wm._row_sort_key))
    assert [(chunk["tier"], chunk["cohort"]) for chunk in chunks] == [
        ("representative", "native"),
        ("representative", "partial_native_row"),
        ("representative", "reduced_proxy"),
        ("depth", "native"),
    ]


def test_completed_authority_validation_delegates_current_contract(monkeypatch):
    calls = {}
    roster = {
        f"problem_{index:03d}_fixture": {
            "slug": f"problem_{index:03d}_fixture",
            "category": "L1", "language": "python",
        }
        for index in range(201)
    }
    document = {"outer_timeout_seconds": 2000}
    monkeypatch.setattr(wm.sr, "native_env_fingerprint", lambda: {"hash": SHA})

    def contract(**kwargs):
        calls["contract"] = kwargs
        return {"current": True}

    def validate(artifact, **kwargs):
        calls["validate"] = (artifact, kwargs)

    monkeypatch.setattr(wm.sr, "build_probe_contract", contract)
    monkeypatch.setattr(wm.sr, "validate_completed_runnability", validate)
    wm._validate_completed_runnability(
        document, repo_root=REPO_ROOT, roster=roster
    )
    assert calls["contract"]["native_env_hash"] == SHA
    assert calls["contract"]["outer_timeout"] == 2000
    assert calls["contract"]["target_cells"] == list(roster.values())
    assert calls["validate"] == (
        document,
        {"expected_contract": {"current": True}, "expected_cells": list(roster.values())},
    )


def test_runnability_exact_bytes_use_shared_loader(tmp_path, monkeypatch):
    path = tmp_path / "runnability.json"
    artifact = {"outer_timeout_seconds": 2000, "fixture": True}
    _write_json(path, artifact)
    roster = {"problem_000_fixture": {"slug": "problem_000_fixture"}}
    contract = {"current": True}
    monkeypatch.setattr(
        wm, "_validate_completed_runnability",
        lambda candidate, **kwargs: contract,
    )
    calls = []

    def load(observed_path, **kwargs):
        calls.append((observed_path, kwargs))
        return copy.deepcopy(artifact), wm.sha256_file(path)

    monkeypatch.setattr(wm.sr, "load_completed_runnability", load)
    monkeypatch.setattr(wm.sr, "validate_completed_runnability", lambda *_a, **_k: None)
    owner_root = tmp_path / "owner"
    owner_root.mkdir()
    observed, payload_sha, observed_contract = wm._load_current_runnability(
        path, repo_root=tmp_path, roster=roster,
        external_campaign_root=owner_root,
    )
    assert observed == artifact
    assert payload_sha == wm.sha256_file(path)
    assert observed_contract == contract
    assert calls == [(path, {
        "expected_contract": None,
        "require_external_owner": True,
        "external_campaign_root": owner_root,
    })]


def test_matrix_serialization_and_variant_id_are_deterministic(tmp_path):
    first, rows, *_ = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED")
    )
    second, second_rows, *_ = _native_fixture(
        tmp_path, statuses=("PASSED", "FAILED")
    )
    assert first == second
    assert rows == second_rows
    document = _matrix_document(first, rows)
    assert wm.serialize_matrix(document).endswith(b"\n")
    assert wm.serialize_matrix(document) == wm.serialize_matrix(copy.deepcopy(document))


def test_live_fingerprint_pin_and_variant_closure_are_exact(tmp_path):
    pin = wm._live_native_env_pin(REPO_ROOT)
    assert pin["path"] == wm.LIVE_NATIVE_ENV_FINGERPRINT_PATH.as_posix()
    assert pin["native_env_hash"] == wm.sr.CERTIFIED_NATIVE_ENV_HASH
    assert pin["sha256"] == wm.sha256_file(
        REPO_ROOT / wm.LIVE_NATIVE_ENV_FINGERPRINT_PATH
    )

    variants, _rows, *_ = _native_fixture(tmp_path / "source")
    wm._validate_variant_native_env_closure(
        variants, expected_hash="1" * 64,
    )
    for target in ("variant", "adapter", "invocation"):
        stale = copy.deepcopy(variants)
        if target == "variant":
            stale[0]["native_env_hash"] = "f" * 64
        elif target == "adapter":
            stale[0]["adapter_contract"]["native_env_hash"] = "f" * 64
        else:
            stale[0]["invocation"]["adapter_contract"][
                "native_env_hash"
            ] = "f" * 64
        with pytest.raises(
            wm.WorkloadMatrixError,
            match="does not close over the live fingerprint",
        ):
            wm._validate_variant_native_env_closure(
                stale, expected_hash="1" * 64,
            )


def test_live_source_commit_is_distinct_from_frozen_canonical_provenance(
    monkeypatch,
):
    live = "a" * 40
    monkeypatch.setattr(wm.sr, "source_head", lambda: live)
    monkeypatch.setattr(wm.sr, "source_worktree_state", lambda: [])
    assert wm._live_source_commit(
        {"source_commit": live}, {"source_commit": live}
    ) == live
    # The frozen canonical commit is deliberately not an operand here; it is
    # retained only inside document["canonical_matrix"]["source_commit"].


@pytest.mark.parametrize("wrong_role", ["runnability", "shrink", "head"])
def test_live_source_commit_refuses_any_authority_disagreement(
    monkeypatch, wrong_role,
):
    live = "a" * 40
    run_commit = live
    shrink_commit = live
    head = live
    if wrong_role == "runnability":
        run_commit = "b" * 40
    elif wrong_role == "shrink":
        shrink_commit = "b" * 40
    else:
        head = "b" * 40
    monkeypatch.setattr(wm.sr, "source_head", lambda: head)
    monkeypatch.setattr(wm.sr, "source_worktree_state", lambda: [])
    with pytest.raises(
        wm.WorkloadMatrixError, match="source commits differ",
    ):
        wm._live_source_commit(
            {"source_commit": run_commit},
            {"source_commit": shrink_commit},
        )


def test_live_source_commit_refuses_dirty_producer(monkeypatch):
    live = "a" * 40
    monkeypatch.setattr(wm.sr, "source_head", lambda: live)

    def dirty():
        raise wm.sr.PrerequisiteContractError("live source is dirty")

    monkeypatch.setattr(wm.sr, "source_worktree_state", dirty)
    with pytest.raises(
        wm.sr.PrerequisiteContractError, match="live source is dirty",
    ):
        wm._live_source_commit(
            {"source_commit": live}, {"source_commit": live}
        )


def test_complete_fixture_emit_and_check_are_byte_exact(tmp_path):
    full, full_rows, *_ = _native_fixture(tmp_path / "source")
    document = _matrix_document(full, full_rows)
    jsonschema.validate(document, SCHEMA)
    output = tmp_path / "out" / "matrix.json"
    output.parent.mkdir()
    live.initialize_plan_owner(
        plan_root=output.parent, source_commit=document["source_commit"],
        matrix_name=output.name, inventory_name="live-inventory.json",
        runnability_owner=document["runnability_authority"]["owner_marker"],
        shrink_owner=document["reduced_proxy_authority"]["owner_marker"],
    )
    assert wm._deliver_matrix(document, emit=output) == 0
    assert output.read_bytes() == wm.serialize_matrix(document)
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(
        wm.WorkloadMatrixError, match="emit namespace is not empty"
    ):
        wm._deliver_matrix(document, emit=output)
    assert wm._deliver_matrix(document, check=output) == 0
    output.chmod(0o600)
    output.write_bytes(output.read_bytes() + b"\n")
    with pytest.raises(
        wm.WorkloadMatrixError, match="emit namespace is not empty"
    ):
        wm._deliver_matrix(document, emit=output)
    output.chmod(0o444)
    assert wm._deliver_matrix(document, check=output) == 1


def test_matrix_delivery_refuses_wrong_plan_filename_before_poisoning(tmp_path):
    full, full_rows, *_ = _native_fixture(tmp_path / "source")
    document = _matrix_document(full, full_rows)
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    live.initialize_plan_owner(
        plan_root=root, source_commit=document["source_commit"],
        matrix_name="matrix.json", inventory_name="live-inventory.json",
        runnability_owner=document["runnability_authority"]["owner_marker"],
        shrink_owner=document["reduced_proxy_authority"]["owner_marker"],
    )
    with pytest.raises(wm.WorkloadMatrixError, match="marker contract mismatch"):
        wm._deliver_matrix(document, emit=root / "wrong.json")
    assert {path.name for path in root.iterdir()} == {
        live.PLAN_OWNER_FILENAME
    }


def test_matrix_delivery_detects_extra_child_injected_during_publish(
    tmp_path, monkeypatch,
):
    full, full_rows, *_ = _native_fixture(tmp_path / "source")
    document = _matrix_document(full, full_rows)
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    output = root / "matrix.json"
    live.initialize_plan_owner(
        plan_root=root, source_commit=document["source_commit"],
        matrix_name=output.name, inventory_name="live-inventory.json",
        runnability_owner=document["runnability_authority"]["owner_marker"],
        shrink_owner=document["reduced_proxy_authority"]["owner_marker"],
    )
    original_publish = wm.sr.publish_external_child

    def publish_with_extra(identity, name, payload, **kwargs):
        result = original_publish(identity, name, payload, **kwargs)
        if name == output.name:
            original_publish(identity, "unexpected.json", b"{}\n")
        return result

    monkeypatch.setattr(wm.sr, "publish_external_child", publish_with_extra)
    with pytest.raises(
        wm.WorkloadMatrixError, match="namespace changed during matrix publication"
    ):
        wm._deliver_matrix(document, emit=output)


def test_matrix_delivery_detects_marker_replacement_during_publish(
    tmp_path, monkeypatch,
):
    full, full_rows, *_ = _native_fixture(tmp_path / "source")
    document = _matrix_document(full, full_rows)
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    output = root / "matrix.json"
    live.initialize_plan_owner(
        plan_root=root, source_commit=document["source_commit"],
        matrix_name=output.name, inventory_name="live-inventory.json",
        runnability_owner=document["runnability_authority"]["owner_marker"],
        shrink_owner=document["reduced_proxy_authority"]["owner_marker"],
    )
    marker_path = root / live.PLAN_OWNER_FILENAME
    original_publish = wm.sr.publish_external_child

    def publish_then_replace_marker(identity, name, payload, **kwargs):
        result = original_publish(identity, name, payload, **kwargs)
        if name == output.name:
            marker_path.chmod(0o600)
            marker_path.write_bytes(marker_path.read_bytes() + b"\n")
            marker_path.chmod(0o444)
        return result

    monkeypatch.setattr(
        wm.sr, "publish_external_child", publish_then_replace_marker
    )
    with pytest.raises(
        wm.WorkloadMatrixError, match="owner changed during matrix publication"
    ):
        wm._deliver_matrix(document, emit=output)


def test_matrix_check_detects_matrix_replacement_during_read(
    tmp_path, monkeypatch,
):
    full, full_rows, *_ = _native_fixture(tmp_path / "source")
    document = _matrix_document(full, full_rows)
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)
    output = root / "matrix.json"
    live.initialize_plan_owner(
        plan_root=root, source_commit=document["source_commit"],
        matrix_name=output.name, inventory_name="live-inventory.json",
        runnability_owner=document["runnability_authority"]["owner_marker"],
        shrink_owner=document["reduced_proxy_authority"]["owner_marker"],
    )
    assert wm._deliver_matrix(document, emit=output) == 0
    original_read = wm.sr.read_external_child
    matrix_reads = 0

    def read_then_replace_matrix(identity, name, **kwargs):
        nonlocal matrix_reads
        payload = original_read(identity, name, **kwargs)
        if name == output.name:
            matrix_reads += 1
            if matrix_reads == 1:
                output.chmod(0o600)
                output.write_bytes(payload + b"\n")
                output.chmod(0o444)
        return payload

    monkeypatch.setattr(wm.sr, "read_external_child", read_then_replace_matrix)
    with pytest.raises(
        wm.WorkloadMatrixError,
        match="owner, matrix, or namespace changed during matrix check",
    ):
        wm._deliver_matrix(document, check=output)


def test_explicit_external_live_inputs_are_regular_byte_bound_files(tmp_path):
    external = tmp_path / "live" / "canonical.json"
    _write_json(external, {"sealed": True})
    observed = wm._explicit_input_file(
        REPO_ROOT, external, wm.CANONICAL_MATRIX_PATH,
        label="external canonical",
    )
    assert observed == external.resolve()
    assert wm.sha256_file(observed) == wm.sha256_file(external)
    with pytest.raises(wm.WorkloadMatrixError, match="repository-relative"):
        wm._repo_file(REPO_ROOT, str(external), label="implicit external")
    link = tmp_path / "linked.json"
    link.symlink_to(external)
    with pytest.raises(wm.WorkloadMatrixError, match="symbolic link"):
        wm._explicit_input_file(
            REPO_ROOT, link, wm.CANONICAL_MATRIX_PATH,
            label="external canonical",
        )
