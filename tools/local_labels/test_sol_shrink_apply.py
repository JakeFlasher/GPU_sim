"""Tests for the SOL shrink application — pure parts, no GPU."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import sol_shrink_apply as sa  # noqa: E402
from tools.local_labels import sol_reconstitution as sr  # noqa: E402

_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "sol_shrink_provenance_artifact.schema.json").read_text()
)
_EXTERNAL_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "sol_shrink_provenance_artifact.external.v3.schema.json").read_text()
)


def _cell(slug: str, category: str = "L1") -> dict:
    return {
        "slug": slug,
        "category": category,
        "language": "cuda_cpp",
        "status": sr.HOST_CAPACITY_OOM,
    }


def _bind_native_diagnostics(
    cell: dict, benchmark: Path, statuses: list[str],
) -> dict:
    identities = sr._load_workload_identities(benchmark)
    assert len(identities) == len(statuses)
    diagnostics = []
    for identity, requested in zip(identities, statuses, strict=True):
        if requested == "passed":
            status, log = "PASSED", ""
        elif requested == "oom":
            status, log = "RUNTIME_ERROR", "CUDA out of memory"
        elif requested == "correctness_failed":
            status, log = "FAILED", "numerical mismatch"
        else:
            raise AssertionError(f"unknown planted native status: {requested}")
        trace = {
            "workload": {"uuid": identity["uuid"], "axes": identity["axes"]},
            "evaluation": {"status": status, "log": log},
        }
        diagnostics.append({
            **identity,
            "diagnostic_present": True,
            "trace_index": identity["workload_index"],
            "status": status,
            "trace": trace,
        })
    return {
        **cell,
        "diagnostics_complete": True,
        "unexpected_workload_diagnostics": [],
        "workload_diagnostics": diagnostics,
    }


def _benchmark(
        root: Path, slug: str, *, constants: dict[str, int],
        axes: dict[str, int], source_uuid: str = "source-u") -> tuple[Path, bytes]:
    root.mkdir(parents=True)
    definition_axes = {
        **{key: {"type": "const", "value": value}
           for key, value in constants.items()},
        **{key: {"type": "var"} for key in axes},
    }
    (root / "definition.json").write_text(json.dumps({
        "name": slug.removeprefix("problem_"),
        "axes": definition_axes,
    }, indent=2) + "\n")
    # Deliberate surrounding spaces prove native-row hashing binds exact source
    # bytes rather than a normalized reserialization.
    source = ("  " + json.dumps({"uuid": source_uuid, "axes": axes},
                                 sort_keys=True) + "  ").encode()
    (root / "workload.jsonl").write_bytes(source + b"\n")
    return root, source


def _multirow_benchmark(
    root: Path, slug: str, *, constants: dict[str, int],
    axes_rows: list[dict[str, int]],
) -> tuple[Path, list[bytes]]:
    benchmark, _ = _benchmark(
        root, slug, constants=constants, axes=axes_rows[0], source_uuid="source-0"
    )
    rows = [
        json.dumps(
            {"uuid": f"source-{index}", "axes": axes}, sort_keys=True,
        ).encode()
        for index, axes in enumerate(axes_rows)
    ]
    (benchmark / "workload.jsonl").write_bytes(
        b"".join(row + b"\n" for row in rows)
    )
    return benchmark, rows


def _attention_benchmark(tmp_path: Path) -> tuple[dict, Path, bytes]:
    slug = "problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward"
    benchmark, source = _benchmark(
        tmp_path / "benchmark", slug,
        constants={"num_heads": 128, "head_dim": 128},
        axes={"batch_size": 1, "seq_len_q": 4096, "seq_len_k": 4096})
    return _bind_native_diagnostics(_cell(slug), benchmark, ["oom"]), benchmark, source


def test_geometry_planner_binds_identity_bytes_axes_and_model(tmp_path):
    cell, benchmark, source = _attention_benchmark(tmp_path)
    before = (benchmark / "workload.jsonl").read_bytes()
    plan = sa.plan_proxy_cohort(cell, benchmark)
    assert (benchmark / "workload.jsonl").read_bytes() == before
    assert plan["source_row_count"] == 1
    row = plan["row_plans"][0]
    assert row["planning_status"] == "proxy_planned"
    assert (row["slug"], row["source_workload_uuid"], row["variant_id"]) == (
        cell["slug"], "source-u", row["variant_id"])
    assert len(row["variant_id"]) == 64
    assert row["native_row_sha256"] == hashlib.sha256(source).hexdigest()
    assert row["reduced_row_sha256"] != row["native_row_sha256"]
    assert row["native_axes"] != row["reduced_axes"]
    assert row["axes_delta"] == [
        {"axis": "seq_len_k", "from": 4096, "to": 1024, "factor": 4.0},
        {"axis": "seq_len_q", "from": 4096, "to": 1024, "factor": 4.0},
    ]
    assert row["model"]["model_id"] == (
        "sol_reviewed_oom_proxy.v2:p060_sequence"
    )
    assert row["static_floor"] is None
    assert set(row["geometry"]) == {
        "classification", "fit", "regime_flags", "reviewed_policy"}
    assert row["native_predictive"] is False
    assert row["claim_scope"] == "reduced_shape_only"


def test_geometry_planner_hashes_exact_row_bytes_excluding_lf_only(tmp_path):
    cell, benchmark, source = _attention_benchmark(tmp_path)
    exact = source + b"\r"
    (benchmark / "workload.jsonl").write_bytes(exact + b"\n")
    cell = _bind_native_diagnostics(_cell(cell["slug"]), benchmark, ["oom"])
    row = sa.plan_proxy_cohort(cell, benchmark)["row_plans"][0]
    assert row["row_sha256_scope"] == sa.ROW_SHA256_SCOPE
    assert row["native_row_sha256"] == hashlib.sha256(exact).hexdigest()
    assert row["native_row_sha256"] != hashlib.sha256(exact.strip()).hexdigest()


@pytest.mark.parametrize(
    ("slug", "native", "expected"),
    [
        ("problem_001_attention_softmax_dropout_value_matmul_backward",
         {"batch_size": 4, "seq_len_q": 1024, "seq_len_kv": 2048},
         {"batch_size": 1, "seq_len_q": 1024, "seq_len_kv": 2048}),
        ("problem_013_fused_residual_rms_norm_backward",
         {"batch_size": 64, "seq_len": 8192},
         {"batch_size": 16, "seq_len": 8192}),
        ("problem_018_fused_rope_with_qk_norm_and_kv_cache_update",
         {"batch_size": 8, "seq_len": 256, "cache_len": 1024},
         {"batch_size": 4, "seq_len": 256, "cache_len": 1024}),
        ("problem_026_video_patch_embedding_projection",
         {"batch_size": 1, "height": 224, "width": 224},
         {"batch_size": 1, "height": 128, "width": 128}),
        ("problem_044_moe_expert_computation", {"num_tokens": 2048},
         {"num_tokens": 2048}),
        ("problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward",
         {"batch_size": 1, "seq_len_q": 4096, "seq_len_k": 4096},
         {"batch_size": 1, "seq_len_q": 1024, "seq_len_k": 1024}),
        ("problem_067_flash_attention_gqa_ultralong",
         {"batch_size": 2, "seq_len": 8192},
         {"batch_size": 2, "seq_len": 2048}),
        ("problem_076_batched_expert_forward",
         {"batch_size": 4, "seq_len": 293},
         {"batch_size": 4, "seq_len": 293}),
        ("problem_002_decoder_layer_full_block",
         {"batch_size": 2, "seq_len": 8192},
         {"batch_size": 1, "seq_len": 4096}),
        ("problem_005_swiglu_mlp_backward",
         {"batch_size": 64, "seq_len": 128},
         {"batch_size": 16, "seq_len": 128}),
        ("problem_012_moe_expert_batched_execution_with_capacity_factor",
         {"num_tokens": 3089}, {"num_tokens": 772}),
        ("problem_024_moe_expert_parallel_execution",
         {"num_tokens": 2447}, {"num_tokens": 612}),
        ("problem_025_moe_expert_parallel_execution_backward",
         {"num_tokens": 2080}, {"num_tokens": 130}),
        ("problem_026_moe_expert_parallel_execution_with_weighted_aggregation",
         {"batch_size": 2, "seq_len": 449},
         {"batch_size": 1, "seq_len": 225}),
        ("problem_028_gqa_rotary_attention_core_backward",
         {"batch_size": 64, "seq_len": 853},
         {"batch_size": 4, "seq_len": 853}),
        ("problem_029_moe_sparse_routing_and_dispatch",
         {"batch_size": 32, "seq_len": 256},
         {"batch_size": 8, "seq_len": 256}),
        ("problem_034_vision_language_cross_attention_fusion",
         {"batch_size": 1, "lang_seq_len": 8192, "vision_seq_len": 12544},
         {"batch_size": 16, "lang_seq_len": 512, "vision_seq_len": 784}),
        ("problem_047_moe_training_token_repeat_and_expert_computation",
         {"batch_seq_len": 8192}, {"batch_seq_len": 8192}),
        ("problem_055_audio_encoder_conv_positional_layer_stack",
         {"batch_size": 80}, {"batch_size": 1}),
        ("problem_059_decoder_layer_full_block",
         {"batch_size": 1, "seq_len": 8192},
         {"batch_size": 8, "seq_len": 1024}),
        ("problem_065_sparse_expert_dispatch_and_combine",
         {"num_tokens": 16384}, {"num_tokens": 16384}),
        ("problem_070_basic_transformer_block",
         {"batch_size": 2, "spatial_seq_len": 4093, "encoder_seq_len": 77},
         {"batch_size": 8, "spatial_seq_len": 511, "encoder_seq_len": 77}),
        ("problem_077_sam_hq_vision_encoder_window_partition_attention",
         {"batch_size": 1, "height": 1024, "width": 1024},
         {"batch_size": 1, "height": 224, "width": 224}),
    ],
)
def test_all_reviewed_oom_families_have_exact_proxy_contracts(
    slug, native, expected,
):
    assert sa._policy_reduced_axes(slug, native) == expected


def test_reviewed_policy_registry_is_closed_to_23_families_and_8_definitions():
    assert len(sa.MODEL_SPECS) == 23
    assert sum("definition_axis" in row for row in sa.MODEL_SPECS.values()) == 8
    assert len({row["policy"] for row in sa.MODEL_SPECS.values()}) == 23


@pytest.mark.parametrize(
    ("slug", "native_experts", "topk", "description", "expected_pointers"),
    [
        ("problem_044_moe_expert_computation", 256, 8, "generic",
         ["/axes/num_experts/value", "/reference"]),
        ("problem_076_batched_expert_forward", 128, 8, "generic",
         ["/axes/num_experts/value"]),
        ("problem_012_moe_expert_batched_execution_with_capacity_factor",
         160, 8, "generic", ["/axes/num_experts/value"]),
        ("problem_024_moe_expert_parallel_execution", 256, 8,
         "Parallel execution of 256 routed MoE experts",
         ["/axes/n_routed_experts/value", "/description"]),
        ("problem_025_moe_expert_parallel_execution_backward", 256, 8,
         "generic", ["/axes/n_routed_experts/value"]),
        ("problem_026_moe_expert_parallel_execution_with_weighted_aggregation",
         16, 2, "parallel computation across all 16 experts",
         ["/axes/num_experts/value", "/description"]),
        ("problem_047_moe_training_token_repeat_and_expert_computation",
         256, 8, "generic", ["/axes/num_experts/value"]),
        ("problem_065_sparse_expert_dispatch_and_combine", 128, 4,
         "generic", ["/axes/num_local_experts/value"]),
    ],
)
def test_definition_proxy_changes_only_reviewed_json_pointers(
    slug, native_experts, topk, description, expected_pointers,
):
    spec = sa.MODEL_SPECS[slug]
    definition = {
        "name": slug.removeprefix("problem_"),
        "description": description,
        "axes": {
            spec["definition_axis"]: {"type": "const", "value": native_experts},
            "num_experts_per_tok": {"type": "const", "value": topk},
            "immutable": {"type": "const", "value": 777},
        },
        "inputs": [{"name": "x", "dtype": "float16"}],
        "outputs": [{"name": "y", "dtype": "float16"}],
    }
    reference_replacements = spec.get("text_replacements", {}).get(
        "/reference", []
    )
    definition["reference"] = "\n".join(
        pair[0] for pair in reference_replacements
    ) or "generic dynamic reference"
    payload, delta = sa._definition_variant(definition, spec)
    reduced = json.loads(payload)
    assert [row["json_pointer"] for row in delta] == sorted(expected_pointers)
    assert reduced["name"] == definition["name"]
    assert reduced["inputs"] == definition["inputs"]
    assert reduced["outputs"] == definition["outputs"]
    assert reduced["axes"]["immutable"] == definition["axes"]["immutable"]
    assert reduced["axes"][spec["definition_axis"]]["value"] == (
        spec["definition_value"]
    )


def test_unregistered_family_is_typed_needs_model(tmp_path):
    slug = "problem_016_rope_inverse_frequency_computation"
    benchmark, _ = _benchmark(
        tmp_path / "benchmark", slug, constants={}, axes={})
    cell = _bind_native_diagnostics(_cell(slug), benchmark, ["oom"])
    row = sa.plan_proxy_cohort(cell, benchmark)["row_plans"][0]
    assert row["planning_status"] == "needs_model"
    assert row["planning_reason"] == "family_model_not_registered"
    assert row["reduced_row_sha256"] is None


def test_empty_axes_row_is_valid_and_typed_by_model_availability(tmp_path):
    slug = "problem_016_rope_inverse_frequency_computation"
    benchmark, source = _benchmark(
        tmp_path / "benchmark", slug, constants={}, axes={},
        source_uuid="0123456789abcdef0123456789abcdef",
    )
    cell = _bind_native_diagnostics(_cell(slug), benchmark, ["oom"])
    row = sa.plan_proxy_cohort(cell, benchmark)["row_plans"][0]
    assert row["native_axes"] == {}
    assert row["native_row_sha256"] == hashlib.sha256(source).hexdigest()
    assert row["planning_status"] in {"needs_model", "geometry_unavailable"}


def test_reviewed_family_rejects_unrecognized_native_geometry(tmp_path):
    slug = "problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward"
    benchmark, _ = _benchmark(
        tmp_path / "benchmark", slug,
        constants={"num_heads": 128, "head_dim": 128},
        axes={"batch_size": 1, "seq_len_q": 128, "seq_len_k": 128})
    cell = _bind_native_diagnostics(_cell(slug), benchmark, ["oom"])
    with pytest.raises(ValueError, match="no mapping"):
        sa.plan_proxy_cohort(cell, benchmark)


def test_moe_static_floor_uses_external_definition_proxy(tmp_path):
    slug = "problem_044_moe_expert_computation"
    benchmark, _ = _benchmark(
        tmp_path / "benchmark", slug,
        constants={
            "num_experts": 256, "num_experts_per_tok": 8,
            "hidden_size": 7168, "intermediate_size": 2048,
        }, axes={"num_tokens": 4096})
    definition_path = benchmark / "definition.json"
    definition = json.loads(definition_path.read_text())
    definition["reference"] = "\n".join(
        pair[0] for pair in sa.MODEL_SPECS[slug]["text_replacements"][
            "/reference"
        ]
    )
    definition_path.write_text(json.dumps(definition, indent=2) + "\n")
    cell = _bind_native_diagnostics(_cell(slug), benchmark, ["oom"])
    row = sa.plan_proxy_cohort(cell, benchmark)["row_plans"][0]
    assert row["planning_status"] == "proxy_planned"
    assert row["axes_delta"] == []
    assert row["native_row_sha256"] == row["reduced_row_sha256"]
    assert [item["json_pointer"] for item in row["definition_delta"]] == [
        "/axes/num_experts/value", "/reference",
    ]
    assert row["definition_sha256"] != row["reduced_definition_sha256"]


def test_definition_proxy_preserves_semantic_input_member_order(tmp_path):
    slug = "problem_044_moe_expert_computation"
    benchmark, _ = _benchmark(
        tmp_path / "benchmark", slug,
        constants={
            "num_experts": 256, "num_experts_per_tok": 8,
            "hidden_size": 7168, "intermediate_size": 2048,
        }, axes={"num_tokens": 4096})
    definition_path = benchmark / "definition.json"
    definition = json.loads(definition_path.read_text())
    definition["inputs"] = {
        "hidden_states": {"dtype": "float16"},
        "gate_proj_weights": {"dtype": "float16"},
        "up_proj_weights": {"dtype": "float16"},
        "down_proj_weights": {"dtype": "float16"},
        "topk_ids": {"dtype": "int64"},
        "topk_weights": {"dtype": "float32"},
    }
    definition["reference"] = "\n".join(
        pair[0] for pair in sa.MODEL_SPECS[slug]["text_replacements"][
            "/reference"
        ]
    )

    reduced_bytes, _ = sa._definition_variant(
        definition, sa.MODEL_SPECS[slug]
    )

    assert reduced_bytes is not None
    reduced = json.loads(reduced_bytes)
    assert list(reduced) == list(definition)
    assert list(reduced["inputs"]) == list(definition["inputs"])
    assert list(reduced["axes"]) == list(definition["axes"])
    assert reduced["axes"]["num_experts"]["value"] == 32


def test_mixed_passed_and_oom_rows_plan_only_the_oom_row(tmp_path):
    slug = "problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward"
    benchmark, _ = _multirow_benchmark(
        tmp_path / "benchmark", slug,
        constants={"num_heads": 128, "head_dim": 128},
        axes_rows=[
            {"batch_size": 1, "seq_len_q": 128, "seq_len_k": 128},
            {"batch_size": 1, "seq_len_q": 4096, "seq_len_k": 4096},
        ],
    )
    cell = _bind_native_diagnostics(
        _cell(slug), benchmark, ["passed", "oom"]
    )
    plans = sa.plan_proxy_cohort(cell, benchmark)["row_plans"]
    assert len(plans) == 2
    assert plans[0]["planning_status"] == "no_proxy_needed"
    assert plans[0]["planning_reason"] == "native_workload_passed"
    assert plans[0]["reduced_row_sha256"] is None
    assert plans[1]["planning_status"] == "proxy_planned"
    assert plans[1]["reduced_row_sha256"] is not None


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("workload_index", 9),
        ("source_line", 9),
        ("uuid", "wrong-uuid"),
        ("axes", {"batch_size": 999}),
        ("workload_sha256", "0" * 64),
        ("workload_sha256_scope", "normalized_json"),
    ],
)
def test_native_diagnostic_identity_mismatch_is_rejected(
    tmp_path, field, replacement,
):
    cell, benchmark, _ = _attention_benchmark(tmp_path)
    cell["workload_diagnostics"][0][field] = replacement
    with pytest.raises(ValueError, match=f"identity mismatch.*{field}"):
        sa.plan_proxy_cohort(cell, benchmark)


def test_correctness_failed_row_is_excluded_from_shrink_planning(tmp_path):
    slug = "problem_060_fused_attention_qk_matmul_scale_mask_softmax_backward"
    benchmark, _ = _multirow_benchmark(
        tmp_path / "benchmark", slug,
        constants={"num_heads": 128, "head_dim": 128},
        axes_rows=[
            {"batch_size": 1, "seq_len_q": 4096, "seq_len_k": 4096},
            {"batch_size": 1, "seq_len_q": 4096, "seq_len_k": 4096},
        ],
    )
    cell = _bind_native_diagnostics(
        _cell(slug), benchmark, ["correctness_failed", "oom"]
    )
    plans = sa.plan_proxy_cohort(cell, benchmark)["row_plans"]
    assert plans[0]["planning_status"] == "not_shrink_eligible"
    assert plans[0]["planning_reason"] == (
        "native_workload_not_host_capacity:correctness_failed"
    )
    assert plans[0]["model"] is None
    assert plans[0]["reduced_row_sha256"] is None
    assert plans[1]["planning_status"] == "proxy_planned"


def test_passed_row_cannot_be_erased_by_missing_family_model(
    tmp_path, monkeypatch,
):
    slug = "problem_016_rope_inverse_frequency_computation"
    benchmark, _ = _multirow_benchmark(
        tmp_path / "benchmark", slug, constants={},
        axes_rows=[
            {},
            {},
        ],
    )
    cell = _bind_native_diagnostics(
        _cell(slug), benchmark, ["passed", "oom"]
    )
    monkeypatch.setattr(
        sa.sr, "resolve_paths",
        lambda _unused: (Path("runner"), Path("solution"), benchmark),
    )
    monkeypatch.setattr(
        sa, "_run_reduced",
        lambda *_args, **_kwargs: pytest.fail(
            "no-proxy/needs-model rows reached the GPU runner"
        ),
    )
    record = sa.apply_one(cell, timeout=2000, probe_contract={})
    assert record["classification"] == sa.CLASS_NEEDS_MODEL
    assert record["source_row_count"] == len(record["row_plans"]) == 2
    assert record["row_plans"][0]["planning_status"] == "no_proxy_needed"
    assert record["row_plans"][0]["planning_reason"] == "native_workload_passed"
    assert record["row_plans"][1]["planning_status"] == "needs_model"
    assert record["row_plans"][1]["planning_reason"] == (
        "family_model_not_registered"
    )
    jsonschema.validate(_v2_artifact(record), _EXTERNAL_SCHEMA)
    assert sa._record_is_resumable(record, slug)
    erased = copy.deepcopy(record)
    del erased["row_plans"][0]
    assert not sa._record_is_resumable(erased, slug)


# ---- shrink classification (pure, fail-closed) ----------------------------- #
def test_reduced_pass_is_reduced_representative_not_shape_adjusted():
    # Conservative: a mechanical shrink that passes correctness is
    # reduced_representative (regime not geometry-verified), NOT shape_adjusted.
    cls, _ = sa.classify_shrink_result(reduced_ran=True, status=sr.RUNNABLE, has_safe_axis=True)
    assert cls == sa.CLASS_REDUCED
    assert cls != sa.CLASS_SHAPE_ADJUSTED


def test_still_oom_is_unavailable():
    cls, _ = sa.classify_shrink_result(reduced_ran=True, status=sr.HOST_CAPACITY_OOM,
                                       has_safe_axis=True)
    assert cls == sa.CLASS_UNAVAILABLE


def test_no_safe_axis_is_unavailable():
    # Genuinely shape-gated (nothing to reduce) -> capacity/shape verdict.
    cls, _ = sa.classify_shrink_result(reduced_ran=False, status=sr.RUNNABLE, has_safe_axis=False)
    assert cls == sa.CLASS_UNAVAILABLE


def test_reduced_correctness_fail_is_correctness_not_capacity():
    # THE BUG Codex found: a reduced-shape correctness failure is NOT a capacity
    # deferral — it must be correctness_failed, never unavailable::oom_shape_gated.
    cls, _ = sa.classify_shrink_result(reduced_ran=True, status=sr.CORRECTNESS_FAILED,
                                       has_safe_axis=True)
    assert cls == sa.CLASS_CORRECTNESS_FAILED
    assert cls != sa.CLASS_UNAVAILABLE


def test_reduced_compile_fail_is_calibration_inconclusive():
    cls, _ = sa.classify_shrink_result(reduced_ran=True, status=sr.COMPILE_FAILED,
                                       has_safe_axis=True)
    assert cls == sa.CLASS_CALIBRATION_INCONCLUSIVE


def test_reduced_eval_error_is_calibration_inconclusive():
    cls, _ = sa.classify_shrink_result(reduced_ran=False, status=sr.EVAL_ERROR,
                                       has_safe_axis=True)
    assert cls == sa.CLASS_CALIBRATION_INCONCLUSIVE


# ---- target selection + artifact schema ------------------------------------ #
def test_host_capacity_slugs_filters_oom_only():
    runnability = {"cells": [
        {"slug": "a", "status": sr.HOST_CAPACITY_OOM},
        {"slug": "b", "status": sr.COMPILE_FAILED},
        {"slug": "c", "status": sr.HOST_CAPACITY_OOM},
    ]}
    got = {c["slug"] for c in sa.host_capacity_slugs(runnability)}
    assert got == {"a", "c"}
    assert sa.host_capacity_slugs(None) == []


def _full_cell(**over):
    base = {
        "slug": "problem_x", "category": "L1", "language": "cuda_cpp",
        "classification": "reduced_representative", "detail": "ok",
        "native_predictive": False, "native_shape_ref": "…/workload.jsonl",
        "shrink_vector": [{"axis": "batch_size", "from": 8, "to": 4}],
        "reduced_workload_ref": "…/reduced.jsonl", "reduced_run_status": "runnable",
        "reduced_run_detail": "passed", "correctness_result": "passed",
        "replay_mode_check": "not_supported_by_sol_execbench_cli",
        "l2_side_verdict": "not_geometry_verified",
        "roofline_class_verdict": "not_geometry_verified",
        "occupancy_wave_verdict": "not_geometry_verified", "halvings": 1,
    }
    base.update(over)
    return base


def _artifact(cells):
    return {"schema_version": 1, "artifact_id": sa.HISTORICAL_ARTIFACT_ID,
            "arch": "sm_86",
            "native_env_hash": "a" * 64, "cells": cells,
            "prior_host_capacity_reconciliation": []}


def test_shrink_artifact_schema_valid():
    art = _artifact([
        _full_cell(),
        _full_cell(slug="problem_y", classification="unavailable::oom_shape_gated",
                   reduced_run_status="host_capacity_oom", correctness_result="not_reached"),
    ])
    jsonschema.validate(art, _SCHEMA)


def test_shrink_artifact_schema_rejects_bad_classification():
    import pytest
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_artifact([_full_cell(classification="made_up")]), _SCHEMA)


def test_shrink_artifact_schema_rejects_missing_verdict_fields():
    # A cell missing the structured verdict provenance must be rejected.
    import pytest
    thin = {"slug": "z", "classification": "reduced_representative"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_artifact([thin]), _SCHEMA)


@pytest.mark.parametrize("field", ["native_shape_ref", "reduced_run_status",
                                   "reduced_run_detail", "reduced_workload_ref"])
def test_shrink_artifact_schema_requires_native_reduced_fields(field):
    # Removing any core native/reduced-run provenance field must FAIL validation
    # (the machine-checkable evidence contract, not merely optional properties).
    cell = _full_cell()
    del cell[field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_artifact([cell]), _SCHEMA)


def test_null_reduced_workload_ref_with_ran_status_is_rejected():
    # A null reduced_workload_ref is legal ONLY for a no-reduced-run status; a
    # status that ran a reduced workload (e.g. host_capacity_oom) must carry one.
    cell = _full_cell(classification="unavailable::oom_shape_gated",
                      reduced_run_status="host_capacity_oom",
                      reduced_workload_ref=None, correctness_result="not_reached")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_artifact([cell]), _SCHEMA)


def test_null_reduced_workload_ref_with_no_safe_axis_is_valid():
    # The legitimate null case: nothing was run because there is no safe axis.
    cell = _full_cell(classification="unavailable::oom_shape_gated",
                      reduced_run_status="no_safe_axis", reduced_workload_ref=None,
                      reduced_run_detail="no safe axis", correctness_result="not_reached")
    jsonschema.validate(_artifact([cell]), _SCHEMA)  # must not raise


def test_committed_shrink_provenance_is_schema_valid():
    # The committed evidence package must satisfy the enforced schema.
    p = (_REPO / "data" / "local_labels" / "campaign"
         / "sol_shrink_provenance_sm86.json")
    if not p.is_file():
        pytest.skip("no committed shrink provenance artifact")
    jsonschema.validate(json.loads(p.read_text()), _SCHEMA)


def test_reconciliation_covers_all_23_prior_deferred():
    # The reconciliation must account for every prior host-capacity cell.
    entries = json.loads(sa.DEFERRED_JSON.read_text())["entries"]
    assert len(entries) == 23
    runnability = {"cells": [{"slug": e["slug"], "status": "compile_failed"}
                             for e in entries]}
    recon = sa.prior_host_capacity_reconciliation(runnability, {})
    assert {r["slug"] for r in recon} == {e["slug"] for e in entries}
    assert all(r["disposition"] in ("shrunk", "retyped_by_native_sweep") for r in recon)


def _v2_artifact(cell):
    return {
        "schema_version": 1,
        "artifact_id": sa.ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": "a" * 64,
        "provenance_contract_version": 3,
        "publication_state": "complete",
        "shrink_contract_version": 1,
        "source_commit": "e" * 40,
        "source_worktree_state": [],
        "source_worktree_state_sha256": sr._canonical_sha256([]),
        "runnability_name": "sol_runnability.external.v2.json",
        "runnability_sha256": "1" * 64,
        "runnability_resume_identity_sha256": "2" * 64,
        "runnability_external_campaign_owner": {
            "name": sr.RUNNABILITY_OWNER_MARKER_NAME,
            "sha256": "e" * 64,
            "owner_seal_sha256": "f" * 64,
        },
        "probe_contract_sha256": "3" * 64,
        "producer_sha256": "4" * 64,
        "calculator_sha256": "5" * 64,
        "provenance_schema_sha256": "6" * 64,
        "artifact_schema_sha256": "7" * 64,
        "model_registry_sha256": "8" * 64,
        "model_threshold_sha256": "9" * 64,
        "target_count": 1,
        "target_set_sha256": "a" * 64,
        "targets": [{
            "category": cell["category"], "slug": cell["slug"],
            "language": cell["language"],
        }],
        "source_closure_file_count": 1,
        "source_closure_sha256": "b" * 64,
        "source_closure": [{"path": "source", "sha256": "c" * 64}],
        "provenance_output_name": "shrink.external.v3.json",
        "checkpoint_output_name": "shrink.external.v3.json.partial",
        "shrink_plan_identity_sha256": "0" * 64,
        "reduced_output_owner": {
            "name": sa.EXTERNAL_OWNER_MARKER_NAME,
            "sha256": "1" * 64,
            "owner_seal_sha256": "2" * 64,
        },
        "shrink_resume_identity_sha256": "d" * 64,
        "external_child_manifest": [],
        "cells": [cell],
        "prior_host_capacity_reconciliation": [],
    }


def _validate_external_def(definition: str, instance: dict) -> None:
    jsonschema.validate(
        instance,
        {
            "$schema": _EXTERNAL_SCHEMA["$schema"],
            "$defs": _EXTERNAL_SCHEMA["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
    )


def _definition_proxy_row() -> dict:
    return {
        "axes_delta": [],
        "claim_scope": "reduced_shape_only",
        "correctness_result": "passed",
        "definition_ref": (
            "tools/sol_execbench/data/benchmark/L1/problem_x/definition.json"
        ),
        "definition_sha256": "1" * 64,
        "definition_delta": [{
            "json_pointer": "/axes/num_experts/value",
            "from": 256,
            "to": 32,
        }],
        "execution_evidence": {},
        "family": "moe_expert_computation",
        "geometry": {},
        "model": {},
        "native_axes": {"num_tokens": 4096},
        "native_predictive": False,
        "native_row_sha256": "2" * 64,
        "planning_reason": None,
        "planning_status": "proxy_planned",
        "reduced_axes": {"num_tokens": 4096},
        "reduced_definition_ref": "L1__problem_x__definition__abc.json",
        "reduced_definition_sha256": "3" * 64,
        "reduced_row_sha256": "2" * 64,
        "reduced_run_detail": "correct",
        "reduced_run_status": "runnable",
        "row_sha256_scope": "exact_json_line_bytes_excluding_lf_only",
        "slug": "problem_x",
        "source_workload_index": 0,
        "source_workload_uuid": "source-u",
        "static_floor": {},
        "variant_id": "4" * 64,
    }


def _definition_proxy_cell() -> dict:
    row = _definition_proxy_row()
    return {
        "claim_scope": "reduced_shape_only",
        "classification": "reduced_representative",
        "cohort_id": "5" * 64,
        "cohort_execution_evidence": {},
        "correctness_result": "passed",
        "category": "L1",
        "definition_ref": row["definition_ref"],
        "definition_sha256": row["definition_sha256"],
        "definition_delta": copy.deepcopy(row["definition_delta"]),
        "detail": "reduced definition passed",
        "halvings": None,
        "l2_side_verdict": "per_variant_geometry",
        "language": "cuda_cpp",
        "native_predictive": False,
        "native_shape_ref": None,
        "native_workload_sha256": "6" * 64,
        "occupancy_wave_verdict": "per_variant_geometry",
        "provenance_contract_version": 3,
        "reduced_run_detail": "correct",
        "reduced_run_status": "runnable",
        "reduced_definition_ref": row["reduced_definition_ref"],
        "reduced_definition_sha256": row["reduced_definition_sha256"],
        "reduced_workload_ref": "L1__problem_x__cohort__def.jsonl",
        "reduced_workload_sha256": "7" * 64,
        "replay_mode_check": "per_variant_geometry",
        "roofline_class_verdict": "per_variant_geometry",
        "row_plans": [row],
        "shrink_vector": [],
        "slug": "problem_x",
        "source_row_count": 1,
    }


def test_external_v3_schema_accepts_definition_only_proxy_identity():
    row = _definition_proxy_row()
    assert row["axes_delta"] == []
    assert row["native_axes"] == row["reduced_axes"]
    assert row["native_row_sha256"] == row["reduced_row_sha256"]
    _validate_external_def("rowPlan", row)
    _validate_external_def("v3Cell", _definition_proxy_cell())


def test_external_v3_schema_accepts_workload_only_proxy_identity():
    row = _definition_proxy_row()
    row.update({
        "axes_delta": [{"axis": "num_tokens", "from": 4096, "to": 512}],
        "definition_delta": [],
        "reduced_axes": {"num_tokens": 512},
        "reduced_definition_ref": None,
        "reduced_definition_sha256": None,
        "reduced_row_sha256": "8" * 64,
    })
    _validate_external_def("rowPlan", row)


def test_external_v3_schema_binds_cell_definition_tuple_to_its_rows():
    cell = _definition_proxy_cell()
    cell.update({
        "definition_delta": [],
        "reduced_definition_ref": None,
        "reduced_definition_sha256": None,
    })
    with pytest.raises(jsonschema.ValidationError):
        _validate_external_def("v3Cell", cell)

    cell = _definition_proxy_cell()
    cell["row_plans"][0].update({
        "axes_delta": [{"axis": "num_tokens", "from": 4096, "to": 512}],
        "definition_delta": [],
        "reduced_axes": {"num_tokens": 512},
        "reduced_definition_ref": None,
        "reduced_definition_sha256": None,
        "reduced_row_sha256": "8" * 64,
    })
    with pytest.raises(jsonschema.ValidationError):
        _validate_external_def("v3Cell", cell)


@pytest.mark.parametrize(
    "mutation",
    [
        {"reduced_definition_ref": None},
        {"reduced_definition_sha256": None},
        {"definition_delta": []},
        {"reduced_definition_ref": None, "reduced_definition_sha256": None},
    ],
)
@pytest.mark.parametrize("definition", ["rowPlan", "v3Cell"])
def test_external_v3_schema_rejects_incoherent_definition_proxy_tuple(
    definition, mutation,
):
    instance = (
        _definition_proxy_row() if definition == "rowPlan"
        else _definition_proxy_cell()
    )
    instance.update(copy.deepcopy(mutation))
    with pytest.raises(jsonschema.ValidationError):
        _validate_external_def(definition, instance)


@pytest.mark.parametrize(
    "unsafe_ref",
    [
        "/absolute/definition.json",
        "../definition.json",
        "proxy/../definition.json",
        "proxy\\definition.json",
        "proxy//definition.json",
        "proxy/",
    ],
)
def test_external_v3_schema_rejects_unsafe_reduced_definition_ref(unsafe_ref):
    row = _definition_proxy_row()
    row["reduced_definition_ref"] = unsafe_ref
    with pytest.raises(jsonschema.ValidationError):
        _validate_external_def("rowPlan", row)


@pytest.mark.parametrize(
    "bad_delta",
    [
        [{"json_pointer": "axes/num_experts/value", "from": 256, "to": 32}],
        [{
            "json_pointer": "/axes/num_experts/value",
            "from": 256,
            "to": 32,
            "reason": "extra field",
        }],
        [{"json_pointer": "/axes/~2bad/value", "from": 256, "to": 32}],
    ],
)
def test_external_v3_schema_closes_definition_delta_objects(bad_delta):
    row = _definition_proxy_row()
    row["definition_delta"] = bad_delta
    with pytest.raises(jsonschema.ValidationError):
        _validate_external_def("rowPlan", row)


def test_external_v3_schema_requires_absent_definition_tuple_without_proxy():
    row = _definition_proxy_row()
    row.update({
        "correctness_result": "not_applicable",
        "definition_delta": [],
        "execution_evidence": None,
        "planning_reason": "native_workload_passed",
        "planning_status": "no_proxy_needed",
        "reduced_axes": None,
        "reduced_definition_ref": None,
        "reduced_definition_sha256": None,
        "reduced_row_sha256": None,
        "reduced_run_detail": None,
        "reduced_run_status": None,
    })
    _validate_external_def("rowPlan", row)

    row.update({
        "definition_delta": [{
            "json_pointer": "/axes/num_experts/value",
            "from": 256,
            "to": 32,
        }],
        "reduced_definition_ref": "definition.json",
        "reduced_definition_sha256": "3" * 64,
    })
    with pytest.raises(jsonschema.ValidationError):
        _validate_external_def("rowPlan", row)


def _passing_execution(cell: dict, workload: Path) -> dict:
    identities = sr._load_workload_identities(workload.parent, workload)
    stdout = "\n".join(json.dumps({
        "definition": "test",
        "workload": {"uuid": row["uuid"], "axes": row["axes"], "inputs": {}},
        "solution": "baseline",
        "evaluation": {"status": "PASSED"},
    }) for row in identities) + "\n"
    file_identity = {"path": "test-input", "sha256": "e" * 64}
    input_state = {
        name: {
            "path": file_identity["path"], "state": "present",
            "sha256": file_identity["sha256"], "detail": None,
        }
        for name in ("definition", "workload", "solution")
    }
    return sr._finish_record(
        {
            "slug": cell["slug"], "category": cell["category"],
            "language": cell["language"], "prior_done": None,
            "prior_deferred_correctness": None, "execution_kind": "proxy",
            "definition_input": file_identity, "workload_input": file_identity,
            "solution_input": file_identity,
            "execution_contract_sha256": "f" * 64,
            "execution_input_state": input_state,
            "execution_input_state_sha256": sr._canonical_sha256(input_state),
        },
        identities, status=sr.RUNNABLE, detail="correct",
        command=[
            str(sr.SOL_VENV_PYTHON), "-m", "sol_execbench.cli.main",
            "--definition", "definition", "--workload", str(workload),
            "--solution", "solution", "--json", "--config",
            str(sr.PROBE_CONFIG), "--timeout", "1500",
        ],
        returncode=0, stdout=stdout, stderr="", outer_timeout_seconds=2000,
    )


def test_portable_external_execution_evidence_removes_machine_local_root(tmp_path):
    root = tmp_path / "reduced-output"
    root.mkdir(mode=0o700)
    root_identity = sa.validate_external_reduced_output_root(root)
    name = "L1__problem_x__cohort__abc.jsonl"
    payload = b'{"uuid":"u","axes":{"n":1}}\n'
    path = sr.publish_external_child(root_identity, name, payload)
    digest = hashlib.sha256(payload).hexdigest()
    state = {
        "definition": {
            "path": "definition.json", "state": "present",
            "sha256": "a" * 64, "detail": None,
        },
        "workload": {
            "path": path.as_posix(), "state": "present",
            "sha256": digest, "detail": None,
        },
        "solution": {
            "path": "solution.json", "state": "present",
            "sha256": "b" * 64, "detail": None,
        },
    }
    evidence = {
        "workload_input": {"path": path.as_posix(), "sha256": digest},
        "execution_input_state": state,
        "execution_input_state_sha256": sr._canonical_sha256(state),
        "command": [
            "tools/sol_execbench/.venv/bin/python", "-m",
            "sol_execbench.cli.main", "--definition", "definition.json",
            "--workload", path.as_posix(), "--solution", "solution.json",
            "--json", "--config",
            "tools/local_labels/sol_full_correctness_config.v1.json",
            "--timeout", "1500",
        ],
    }

    portable = sa._portable_external_execution_evidence(
        evidence, root_identity, name
    )

    assert portable["workload_input"]["path"] == name
    assert portable["execution_input_state"]["workload"]["path"] == name
    assert portable["command"][6] == name
    assert portable["execution_input_state_sha256"] == sr._canonical_sha256(
        portable["execution_input_state"]
    )
    assert root.as_posix() not in json.dumps(portable, sort_keys=True)
    assert evidence["workload_input"]["path"] == path.as_posix()


def test_apply_one_emits_only_changed_correctness_passed_proxy_rows(
        tmp_path, monkeypatch):
    cell, benchmark, _ = _attention_benchmark(tmp_path / "source")
    native_before = (benchmark / "workload.jsonl").read_bytes()
    monkeypatch.setattr(sa.sr, "resolve_paths",
                        lambda unused: (Path("runner"), Path("solution"), benchmark))
    monkeypatch.setattr(sa, "PROXY_VARIANT_ROOT", tmp_path / "variants")
    monkeypatch.setattr(sa, "REDUCED_WORKLOAD_ROOT", tmp_path / "cohorts")
    calls = []

    def pass_correctness(
        unused_cell, workload, unused_timeout, unused_contract,
        unused_definition=None,
    ):
        calls.append(Path(workload))
        return _passing_execution(unused_cell, Path(workload))

    monkeypatch.setattr(sa, "_run_reduced", pass_correctness)
    record = sa.apply_one(cell, timeout=2000, probe_contract={})

    assert len(calls) == 2  # one isolated variant, then the combined cohort
    assert (benchmark / "workload.jsonl").read_bytes() == native_before
    assert record["classification"] == sa.CLASS_REDUCED
    assert record["claim_scope"] == "reduced_shape_only"
    assert record["native_predictive"] is False
    assert record["correctness_result"] == "passed"
    assert record["reduced_run_status"] == sr.RUNNABLE
    assert record["reduced_workload_ref"] is not None
    row = record["row_plans"][0]
    assert row["correctness_result"] == "passed"
    assert row["reduced_run_status"] == sr.RUNNABLE
    assert row["axes_delta"]
    assert row["native_row_sha256"] != row["reduced_row_sha256"]
    cohort = Path(record["reduced_workload_ref"])
    payload = cohort.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["reduced_workload_sha256"]
    effective = [line for line in payload.splitlines() if line.strip()]
    assert len(effective) == 1
    assert hashlib.sha256(effective[0]).hexdigest() == row["reduced_row_sha256"]
    jsonschema.validate(_v2_artifact(record), _EXTERNAL_SCHEMA)
    for field, nested in (
        ("cohort_execution_evidence", None),
        ("execution_evidence", "row_plans"),
    ):
        missing = copy.deepcopy(record)
        if nested is None:
            del missing[field]
        else:
            del missing[nested][0][field]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(_v2_artifact(missing), _EXTERNAL_SCHEMA)


def test_proxy_execution_evidence_rejects_uuid_or_hash_drift(tmp_path):
    cell, benchmark, _ = _attention_benchmark(tmp_path)
    plan = sa.plan_proxy_cohort(cell, benchmark)
    row = plan["row_plans"][0]
    workload = tmp_path / "proxy.jsonl"
    workload.write_bytes(plan["_proxy_rows"][row["variant_id"]] + b"\n")
    evidence = _passing_execution(cell, workload)
    expected = [(row["source_workload_uuid"], row["reduced_row_sha256"])]
    sa._assert_execution_evidence(evidence, expected)
    wrong_uuid = copy.deepcopy(evidence)
    wrong_uuid["workload_diagnostics"][0]["uuid"] = "different"
    with pytest.raises(ValueError, match="bijective"):
        sa._assert_execution_evidence(wrong_uuid, expected)
    wrong_hash = copy.deepcopy(evidence)
    wrong_hash["workload_diagnostics"][0]["workload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bijective"):
        sa._assert_execution_evidence(wrong_hash, expected)


def test_apply_one_needs_model_never_runs_or_labels_reduced(tmp_path, monkeypatch):
    slug = "problem_016_rope_inverse_frequency_computation"
    benchmark, _ = _benchmark(
        tmp_path / "benchmark", slug, constants={}, axes={})
    cell = _bind_native_diagnostics(_cell(slug), benchmark, ["oom"])
    monkeypatch.setattr(sa.sr, "resolve_paths",
                        lambda unused: (Path("runner"), Path("solution"), benchmark))

    def must_not_run(*unused_args, **unused_kwargs):
        raise AssertionError("needs-model row reached GPU correctness runner")

    monkeypatch.setattr(sa, "_run_reduced", must_not_run)
    record = sa.apply_one(cell, timeout=2000, probe_contract={})
    assert record["classification"] == sa.CLASS_NEEDS_MODEL
    assert record["reduced_workload_ref"] is None
    assert record["correctness_result"] == "not_reached"
    jsonschema.validate(_v2_artifact(record), _EXTERNAL_SCHEMA)


def test_apply_all_rejects_partial_mode_before_any_read_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sa.sr, "native_env_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("partial mode reached environment probe"),
    )
    out = tmp_path / "canonical.json"
    with pytest.raises(sr.PrerequisiteContractError, match="partial shrink mode"):
        sa.apply_all(
            out, slugs={"problem_x"}, timeout=2000,
            runnability_path=tmp_path / "runnability.json",
            reduced_output_root=tmp_path,
        )
    assert not out.exists()


def test_apply_all_validates_complete_runnability_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sa.sr, "native_env_fingerprint",
        lambda: {"hash": sr.CERTIFIED_NATIVE_ENV_HASH, "degraded": False},
    )
    monkeypatch.setattr(sa.sr, "in_scope_cells", lambda: [{
        "category": "L1", "slug": "problem_x", "language": "cuda_cpp"
    }])
    monkeypatch.setattr(sa.sr, "build_probe_contract", lambda **_kwargs: {})
    monkeypatch.setattr(
        sa.sr, "load_completed_runnability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sr.PrerequisiteContractError("planted partial runnability")
        ),
    )
    monkeypatch.setattr(
        sa, "apply_one",
        lambda *_args, **_kwargs: pytest.fail("mutation reached apply_one"),
    )
    root = tmp_path / "external"
    root.mkdir(mode=0o700)
    out = root / "canonical.json"
    with pytest.raises(sr.PrerequisiteContractError, match="partial runnability"):
        sa.apply_all(
            out, slugs=None, timeout=2000,
            runnability_path=tmp_path / "runnability.json",
            reduced_output_root=root,
        )
    assert not out.exists()


def test_shrink_resume_identity_binds_exact_source_closure(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    benchmark = tmp_path / "benchmark"
    baseline.mkdir()
    benchmark.mkdir()
    solution = baseline / "solution.json"
    solution.write_text("{}\n")
    source = benchmark / "workload.jsonl"
    original_source = '{"uuid":"u","axes":{"n":1}}\n'
    source.write_text(original_source)
    deferred = tmp_path / "tools" / "sol_baselines_harness" / "deferred.json"
    deferred.parent.mkdir(parents=True)
    deferred.write_text('{"entries":[]}\n')
    cell = {"category": "L1", "slug": "problem_x", "language": "cuda_cpp"}
    monkeypatch.setattr(sa, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sa, "DEFERRED_JSON", deferred)
    monkeypatch.setattr(
        sa.sr, "resolve_paths", lambda _cell: (baseline, solution, benchmark)
    )
    run_root = tmp_path / "run-root"
    run_root.mkdir(mode=0o700)
    run_path = run_root / "runnability.json"
    run_payload = b"{}\n"
    run_path.write_bytes(run_payload)
    run_path.chmod(0o444)
    reduced_root = tmp_path / "reduced-root"
    reduced_root.mkdir(mode=0o700)
    out = reduced_root / "shrink.json"
    runnability = {
        "native_env_hash": sr.CERTIFIED_NATIVE_ENV_HASH,
        "resume_identity_sha256": "a" * 64,
        "external_output_name": run_path.name,
        "external_campaign_owner": {
            "name": sr.RUNNABILITY_OWNER_MARKER_NAME,
            "sha256": "c" * 64,
            "owner_seal_sha256": "d" * 64,
        },
        "source_commit": "e" * 40,
    }
    probe_contract = {"probe_protocol_id": "test"}
    monkeypatch.setattr(sa.sr, "source_worktree_state", lambda: [])
    monkeypatch.setattr(sa, "_source_head", lambda: "e" * 40)
    monkeypatch.setattr(
        sa, "_bind_external_owner",
        lambda *_args, **_kwargs: {
            "name": sa.EXTERNAL_OWNER_MARKER_NAME,
            "sha256": "f" * 64,
            "owner_seal_sha256": "0" * 64,
        },
    )
    first = sa.build_shrink_contract(
        runnability_sha256=hashlib.sha256(run_payload).hexdigest(),
        runnability=runnability, probe_contract=probe_contract, targets=[cell],
        runnability_path=run_path, reduced_output_root=reduced_root,
        provenance_output_path=out,
    )
    source.write_text('{"uuid":"u","axes":{"n":2}}\n')
    second = sa.build_shrink_contract(
        runnability_sha256=hashlib.sha256(run_payload).hexdigest(),
        runnability=runnability, probe_contract=probe_contract, targets=[cell],
        runnability_path=run_path, reduced_output_root=reduced_root,
        provenance_output_path=out,
    )
    assert first["source_closure_sha256"] != second["source_closure_sha256"]
    assert first["shrink_resume_identity_sha256"] != second[
        "shrink_resume_identity_sha256"
    ]
    assert {
        "path": "tools/sol_baselines_harness/deferred.json",
        "sha256": hashlib.sha256(b'{"entries":[]}\n').hexdigest(),
    } in first["source_closure"]

    source.write_text(original_source)
    deferred.write_text('{"entries":[{"slug":"problem_x"}]}\n')
    third = sa.build_shrink_contract(
        runnability_sha256=hashlib.sha256(run_payload).hexdigest(),
        runnability=runnability, probe_contract=probe_contract, targets=[cell],
        runnability_path=run_path, reduced_output_root=reduced_root,
        provenance_output_path=out,
    )
    assert first["source_closure_sha256"] != third["source_closure_sha256"]
    assert first["shrink_resume_identity_sha256"] != third[
        "shrink_resume_identity_sha256"
    ]


def test_shrink_resume_rejects_cell_outside_exact_current_target_set(monkeypatch):
    contract = {
        "shrink_resume_identity_sha256": "a" * 64,
        "targets": [{
            "category": "L1", "slug": "current_slug", "language": "cuda_cpp"
        }],
    }
    checkpoint = {
        "schema_version": 1,
        "artifact_id": sa.ARTIFACT_ID,
        "arch": "sm_86", "sku": "rtx_3090_ti",
        "provenance_contract_version": 3,
        "publication_state": "checkpoint",
        **contract,
        "external_child_manifest": [],
        "cells": [{"slug": "stale_slug"}],
        "prior_host_capacity_reconciliation": [],
    }
    monkeypatch.setattr(sa, "_record_is_resumable", lambda *_a, **_k: True)
    with pytest.raises(sr.PrerequisiteContractError, match="out-of-contract"):
        sa._resumable_checkpoint_records(
            checkpoint, shrink_contract=contract
        )


def test_shrink_resume_rejects_duplicate_valid_slug(monkeypatch):
    duplicate = {"slug": "current_slug", "evidence": "independently-valid"}
    contract = {
        "shrink_resume_identity_sha256": "a" * 64,
        "targets": [{
            "category": "L1", "slug": "current_slug", "language": "cuda_cpp",
        }],
    }
    checkpoint = {
        "schema_version": 1,
        "artifact_id": sa.ARTIFACT_ID,
        "arch": "sm_86", "sku": "rtx_3090_ti",
        "provenance_contract_version": 3,
        "publication_state": "checkpoint",
        **contract,
        "external_child_manifest": [],
        "cells": [duplicate, copy.deepcopy(duplicate)],
        "prior_host_capacity_reconciliation": [],
    }
    monkeypatch.setattr(sa, "_record_is_resumable", lambda *_args, **_kwargs: True)
    with pytest.raises(sr.PrerequisiteContractError, match="duplicate"):
        sa._resumable_checkpoint_records(
            checkpoint, shrink_contract=contract
        )


def test_completed_shrink_validator_rederives_contract_and_exact_cells(
    tmp_path, monkeypatch
):
    target = {"category": "L1", "slug": "problem_x", "language": "cuda_cpp"}
    contract = {
        "shrink_contract_version": 1,
        "targets": [target],
        "target_count": 1,
        "target_set_sha256": "a" * 64,
    }
    runnability = {
        "native_env_hash": sr.CERTIFIED_NATIVE_ENV_HASH,
        "cells": [{**target, "status": sr.HOST_CAPACITY_OOM}],
    }
    artifact = {
        "schema_version": 1, "artifact_id": sa.ARTIFACT_ID,
        "arch": "sm_86", "sku": "rtx_3090_ti",
        "native_env_hash": sr.CERTIFIED_NATIVE_ENV_HASH,
        "provenance_contract_version": 3, "publication_state": "complete",
        **contract,
        "runnability_name": "runnability.json",
        "provenance_output_name": "shrink.json",
        "external_child_manifest": [],
        "cells": [{"slug": "problem_x"}],
        "prior_host_capacity_reconciliation": [],
    }
    monkeypatch.setattr(sa.sr, "validate_completed_runnability", lambda *_a, **_k: None)
    monkeypatch.setattr(sa, "build_shrink_contract", lambda **_kwargs: contract)
    monkeypatch.setattr(sa.jsonschema, "validate", lambda *_a, **_k: None)
    monkeypatch.setattr(sa, "_record_is_resumable", lambda *args, **kwargs: True)
    monkeypatch.setattr(sa, "_external_child_manifest", lambda *_a, **_k: [])
    monkeypatch.setattr(
        sa, "prior_host_capacity_reconciliation", lambda *_args: []
    )
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    run_path = run_root / "runnability.json"
    run_path.write_text("{}\n")
    run_path.chmod(0o444)
    reduced_root = tmp_path / "reduced"
    reduced_root.mkdir(mode=0o700)
    provenance_path = reduced_root / "shrink.json"
    sa.validate_completed_shrink_provenance(
        artifact, runnability=runnability, runnability_sha256="b" * 64,
        probe_contract={}, targets=[target], runnability_path=run_path,
        provenance_path=provenance_path, reduced_output_root=reduced_root,
    )
    with pytest.raises(sr.PrerequisiteContractError, match="host-capacity"):
        sa.validate_completed_shrink_provenance(
            artifact, runnability=runnability, runnability_sha256="b" * 64,
            probe_contract={}, targets=[], runnability_path=run_path,
            provenance_path=provenance_path, reduced_output_root=reduced_root,
        )
    missing = copy.deepcopy(artifact)
    missing["cells"] = []
    with pytest.raises(sr.PrerequisiteContractError, match="target set mismatch"):
        sa.validate_completed_shrink_provenance(
            missing, runnability=runnability, runnability_sha256="b" * 64,
            probe_contract={}, targets=[target], runnability_path=run_path,
            provenance_path=provenance_path, reduced_output_root=reduced_root,
        )


def test_reduced_evidence_invariant_rejects_unchanged_or_unverified(tmp_path):
    cell, benchmark, _ = _attention_benchmark(tmp_path)
    plan = sa.plan_proxy_cohort(cell, benchmark)
    row = plan["row_plans"][0]
    payload = plan["_proxy_rows"][row["variant_id"]]

    unchanged = copy.deepcopy(row)
    unchanged["reduced_axes"] = unchanged["native_axes"]
    with pytest.raises(ValueError, match="unchanged"):
        sa._assert_proxy_row_evidence(unchanged, payload, require_passed=False)

    with pytest.raises(ValueError, match="unverified"):
        sa._assert_proxy_row_evidence(row, payload, require_passed=True)

    drifted_identity = copy.deepcopy(row)
    drifted_identity["variant_id"] = "0" * 64
    with pytest.raises(ValueError, match="exact bundle"):
        sa._assert_proxy_row_evidence(
            drifted_identity, payload, require_passed=False
        )


@pytest.mark.parametrize("mutation", ["native_predictive", "empty_delta", "pending"])
def test_v2_schema_rejects_unsafe_reduced_claim(tmp_path, monkeypatch, mutation):
    cell, benchmark, _ = _attention_benchmark(tmp_path / "source")
    monkeypatch.setattr(sa.sr, "resolve_paths",
                        lambda unused: (Path("runner"), Path("solution"), benchmark))
    monkeypatch.setattr(sa, "PROXY_VARIANT_ROOT", tmp_path / "variants")
    monkeypatch.setattr(sa, "REDUCED_WORKLOAD_ROOT", tmp_path / "cohorts")
    monkeypatch.setattr(
        sa, "_run_reduced",
        lambda cell_arg, workload, *_args, **_kwargs: _passing_execution(
            cell_arg, Path(workload)
        ))
    record = sa.apply_one(cell, timeout=2000, probe_contract={})
    unsafe = copy.deepcopy(record)
    if mutation == "native_predictive":
        unsafe["native_predictive"] = True
    elif mutation == "empty_delta":
        unsafe["row_plans"][0]["axes_delta"] = []
    else:
        unsafe["row_plans"][0]["correctness_result"] = "pending"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_v2_artifact(unsafe), _EXTERNAL_SCHEMA)
