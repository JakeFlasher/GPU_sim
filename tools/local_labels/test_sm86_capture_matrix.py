"""Focused structural and semantic tests for the sm_86 capture matrix."""
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

from tools.local_labels import build_sm86_capture_matrix as bcm  # noqa: E402
from tools.local_labels import sm86_capture_chunk as cc  # noqa: E402

_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "sm86_capture_matrix.schema.json").read_text()
)
_COMMITTED = (
    _REPO / "data" / "local_labels" / "campaign"
    / "sm86_reference_capture_matrix.json"
)


@pytest.fixture(scope="module")
def matrix() -> dict:
    return bcm.build_matrix(_REPO, require_head_match=False)


def _committed_matrix() -> dict:
    return json.loads(_COMMITTED.read_text())


def test_legacy_two_argument_validation_keeps_live_rederivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _committed_matrix()
    calls: list[dict] = []

    def fake_build_matrix(*args, **kwargs):
        calls.append(kwargs)
        return copy.deepcopy(matrix)

    monkeypatch.setattr(bcm, "build_matrix", fake_build_matrix)
    assert bcm.validate_matrix(matrix, _REPO) == []
    assert calls == [{
        "require_head_match": False,
        "authority_path": _REPO / bcm.AUTHORITY_PATH,
    }]


def test_frozen_v1_validation_does_not_rederive_live_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _committed_matrix()

    def fail_live_rebuild(*args, **kwargs):
        raise AssertionError("frozen validation called live build_matrix")

    monkeypatch.setattr(bcm, "build_matrix", fail_live_rebuild)
    assert bcm.validate_matrix(
        matrix,
        _REPO,
        validation_mode="frozen_v1",
    ) == []

    registry_path = _REPO / bcm.METRIC_REGISTRY_PATH
    registry_payload = registry_path.read_bytes()
    registry_document = json.loads(registry_payload)
    assert bcm.validate_matrix(
        matrix,
        _REPO,
        registry_document=registry_document,
        registry_payload=registry_payload,
        registry_sha256=hashlib.sha256(registry_payload).hexdigest(),
        validation_mode="frozen_v1",
    ) == []


@pytest.mark.parametrize(
    "operands",
    [
        {"registry_document": {}},
        {"registry_payload": b"{}"},
        {"registry_sha256": "0" * 64},
    ],
)
def test_frozen_v1_rejects_partial_registry_operands(operands: dict) -> None:
    problems = bcm.validate_matrix(
        _committed_matrix(),
        _REPO,
        validation_mode="frozen_v1",
        **operands,
    )
    assert "registry_operands_incomplete" in problems


def test_frozen_v1_rejects_unknown_validation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_live_rebuild(*args, **kwargs):
        raise AssertionError("unknown validation mode reached live builder")

    monkeypatch.setattr(bcm, "build_matrix", fail_live_rebuild)
    problems = bcm.validate_matrix(
        _committed_matrix(),
        _REPO,
        validation_mode="ambient_successor",
    )
    assert problems == ["validation_mode_unknown"]


def test_frozen_v1_rejects_registry_payload_hash_mismatch() -> None:
    matrix = _committed_matrix()
    registry_path = _REPO / bcm.METRIC_REGISTRY_PATH
    payload = registry_path.read_bytes()
    document = json.loads(payload)
    problems = bcm.validate_matrix(
        matrix,
        _REPO,
        validation_mode="frozen_v1",
        registry_document=document,
        registry_payload=payload,
        registry_sha256="0" * 64,
    )
    assert "registry_payload_sha256" in problems


def test_frozen_v1_rejects_registry_document_payload_mismatch() -> None:
    matrix = _committed_matrix()
    registry_path = _REPO / bcm.METRIC_REGISTRY_PATH
    payload = registry_path.read_bytes()
    document = json.loads(payload)
    document["registry_version"] += 1
    problems = bcm.validate_matrix(
        matrix,
        _REPO,
        validation_mode="frozen_v1",
        registry_document=document,
        registry_payload=payload,
        registry_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert "registry_document_payload_mismatch" in problems


def test_frozen_v1_rejects_type_drifting_registry_document() -> None:
    matrix = _committed_matrix()
    registry_payload = (_REPO / bcm.METRIC_REGISTRY_PATH).read_bytes()
    registry_document = json.loads(registry_payload)
    registry_document["registry_version"] = float(registry_document["registry_version"])
    problems = bcm.validate_matrix(
        matrix,
        _REPO,
        validation_mode="frozen_v1",
        registry_document=registry_document,
        registry_payload=registry_payload,
        registry_sha256=hashlib.sha256(registry_payload).hexdigest(),
    )
    assert "registry_document_payload_mismatch" in problems


def test_frozen_v1_rejects_valid_registry_with_wrong_projection_or_pin() -> None:
    registry_payload = (_REPO / bcm.METRIC_REGISTRY_PATH).read_bytes()
    registry_document = json.loads(registry_payload)
    digest = hashlib.sha256(registry_payload).hexdigest()

    projection_mutation = _committed_matrix()
    projection_mutation["metric_registry"]["metric_count"] += 1
    problems = bcm.validate_matrix(
        projection_mutation,
        _REPO,
        validation_mode="frozen_v1",
        registry_document=registry_document,
        registry_payload=registry_payload,
        registry_sha256=digest,
    )
    assert "metric_registry_projection" in problems

    pin_mutation = _committed_matrix()
    pin_mutation["input_pins"]["metric_registry"]["path"] = "other-registry.json"
    problems = bcm.validate_matrix(
        pin_mutation,
        _REPO,
        validation_mode="frozen_v1",
        registry_document=registry_document,
        registry_payload=registry_payload,
        registry_sha256=digest,
    )
    assert "registry_pin_path" in problems


def test_frozen_v1_rejects_non_object_registry_pins_without_crashing() -> None:
    matrix = _committed_matrix()
    matrix["input_pins"] = []
    registry_payload = (_REPO / bcm.METRIC_REGISTRY_PATH).read_bytes()
    registry_document = json.loads(registry_payload)
    problems = bcm.validate_matrix(
        matrix,
        _REPO,
        validation_mode="frozen_v1",
        registry_document=registry_document,
        registry_payload=registry_payload,
        registry_sha256=hashlib.sha256(registry_payload).hexdigest(),
    )
    assert "registry_pin_missing" in problems


def test_frozen_v1_rejects_malformed_registry_payload_typed() -> None:
    payload = b'{"registry_version": ' + (b"9" * 5000) + b"}"
    problems = bcm.validate_matrix(
        _committed_matrix(),
        _REPO,
        validation_mode="frozen_v1",
        registry_document={},
        registry_payload=payload,
        registry_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert "registry_payload_json" in problems


def _reseal(matrix: dict) -> None:
    matrix["matrix_seal_sha256"] = bcm.canonical_sha256({
        key: value for key, value in matrix.items()
        if key != "matrix_seal_sha256"
    })


def test_schema_is_valid_and_generated_matrix_conforms(matrix):
    jsonschema.Draft202012Validator.check_schema(_SCHEMA)
    jsonschema.validate(matrix, _SCHEMA)
    assert bcm.validate_matrix(matrix, _REPO) == []


def test_committed_artifact_is_exact_deterministic_rebuild(matrix):
    assert _COMMITTED.is_file(), "generate the committed matrix artifact"
    committed = json.loads(_COMMITTED.read_text())
    assert committed == matrix
    assert bcm.build_matrix(_REPO, require_head_match=False) == matrix


def test_external_authority_path_is_pinned_and_rederivable(tmp_path):
    external = tmp_path / "capture_roster_authority_sm86.json"
    external.write_bytes((_REPO / bcm.AUTHORITY_PATH).read_bytes())
    generated = bcm.build_matrix(
        _REPO, require_head_match=False, authority_path=external,
    )
    assert generated["authority"]["path"] == str(external.resolve())
    jsonschema.validate(generated, _SCHEMA)
    assert bcm.validate_matrix(generated, _REPO) == []


def test_exact_authority_partition_and_tuple_identity(matrix):
    assert matrix["counts"]["universe"] == 344 == len(matrix["cells"])
    assert matrix["counts"]["runnable"] == 122
    assert matrix["counts"]["excluded"] == 222
    identities = [
        (cell["identity"]["roster_source"], cell["identity"]["slug"])
        for cell in matrix["cells"]
    ]
    assert len(identities) == len(set(identities))
    assert {cell["disposition"] for cell in matrix["cells"]} == {"runnable", "excluded"}
    assert matrix["authority"]["sha256"] == bcm.sha256_file(
        _REPO / bcm.AUTHORITY_PATH
    )


def test_chunks_are_bounded_exact_one_time_runnable_partition(matrix):
    chunks = matrix["chunks"]
    assert chunks[0]["chunk_id"] == "portable_lcd-000-bootstrap"
    assert chunks[0]["cells"] == [{
        "roster_source": "portable_lcd", "slug": "silu_fwd_bf16_sm80",
    }]
    sol_bootstrap = next(
        chunk for chunk in chunks
        if chunk["chunk_id"] == "sol_execbench-000-bootstrap"
    )
    assert sol_bootstrap["cells"] == [{
        "roster_source": "sol_execbench", "slug": bcm.SOL_BOOTSTRAP_SLUG,
    }]
    assert len(chunks) == 18
    assert matrix["chunking"]["sol_bootstrap_identity"] == sol_bootstrap["cells"][0]
    assert all(1 <= len(chunk["cells"]) <= bcm.MAX_CHUNK_CELLS for chunk in chunks)
    members = [
        (identity["roster_source"], identity["slug"])
        for chunk in chunks for identity in chunk["cells"]
    ]
    runnable = {
        (cell["identity"]["roster_source"], cell["identity"]["slug"])
        for cell in matrix["cells"] if cell["disposition"] == "runnable"
    }
    excluded = {
        (cell["identity"]["roster_source"], cell["identity"]["slug"])
        for cell in matrix["cells"] if cell["disposition"] == "excluded"
    }
    assert len(members) == len(set(members)) == 122
    assert set(members) == runnable
    assert not (set(members) & excluded)
    for chunk in chunks:
        assert chunk["membership_sha256"] == bcm.canonical_sha256(chunk["cells"])


def test_portable_label_and_trace_args_are_distinct_and_bounded(matrix):
    portable = [
        cell for cell in matrix["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "portable_lcd"
    ]
    assert len(portable) == 17
    for cell in portable:
        inv = cell["invocation"]
        assert inv["adapter_kind"] == "direct_binary"
        assert inv["label_app_args"][-4:] == ["--warmup", "1", "--reps", "1"]
        assert inv["trace_app_args"][-4:] == ["--warmup", "0", "--reps", "1"]
        assert inv["label_app_args"][:-4] == inv["trace_app_args"][:-4]
        assert inv["trace_cardinality_contract"] == "runner_common_single_launch_v1"
        assert inv["capture_launch_contract"] == {
            "schema_version": 1,
            "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 1,
            "label_launch_index": 1,
            "label_measured_reps": 1,
            "trace_launch_count": 1,
            "label_trace_launch_index": 0,
        }
        assert cell["control_bits"] == {
            "policy": "direct_executable",
            "input_path": inv["executable_path"],
            "input_sha256": inv["executable_sha256"],
        }


def test_simbench_capture_overrides_and_launch_contracts_are_closed(matrix):
    simbench = [
        cell for cell in matrix["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "simbench_bw"
    ]
    assert len(simbench) == 31
    by_slug = {cell["identity"]["slug"]: cell for cell in simbench}
    assert set(bcm.SIMBENCH_CAPTURE_OVERRIDES) == {
        "cp_async_shared_tile_copy", "dram_streaming_dot_product",
        "shared_bank_conflict_stride", "vendor_gemm_cublas",
        "vendor_sgemm_cublas",
        "warm_cold_pair_relaunch",
    }
    ordinary = set(by_slug) - set(bcm.SIMBENCH_CAPTURE_OVERRIDES)
    assert len(ordinary) == 25
    for slug in ordinary:
        inv = by_slug[slug]["invocation"]
        assert inv["label_app_args"] == inv["trace_app_args"]
        assert inv["trace_cardinality_contract"] == "authority_args_exact"
        assert "matrix_capture_override" not in inv
        assert inv["capture_launch_contract"] == {
            "schema_version": 1,
            "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 0,
            "label_launch_index": 0,
            "label_measured_reps": 1,
            "trace_launch_count": 1,
            "label_trace_launch_index": 0,
        }

    for slug, pinned in bcm.SIMBENCH_CAPTURE_OVERRIDES.items():
        inv = by_slug[slug]["invocation"]
        override = inv["matrix_capture_override"]
        assert set(override) == {
            "schema_version", "reason", "authority_cell_sha256",
            "authority_shape_point", "authority_app_args_sha256",
            "capture_shape_point", "label_app_args_sha256",
            "trace_app_args_sha256", "shape_file_path", "shape_file_sha256",
        }
        assert override["reason"] == pinned["reason"]
        assert override["authority_cell_sha256"] == pinned["authority_cell_sha256"]
        assert override["authority_app_args_sha256"] == pinned["authority_app_args_sha256"]
        assert override["label_app_args_sha256"] == pinned["label_app_args_sha256"]
        assert override["trace_app_args_sha256"] == pinned["trace_app_args_sha256"]
        assert override["shape_file_sha256"] == pinned["shape_file_sha256"]
        assert inv["label_app_args"] == pinned["label_app_args"]
        assert inv["trace_app_args"] == pinned["trace_app_args"]
        assert inv["logical_workload"]["value"] == pinned["capture_shape_point"]

    warm = by_slug["warm_cold_pair_relaunch"]["invocation"]
    assert warm["trace_cardinality_contract"] == "warm_cold_pair_three_launch_v1"
    assert warm["capture_launch_contract"]["trace_launch_count"] == 3
    assert [
        row["role"] for row in warm["capture_launch_contract"]["trace_launch_sequence"]
    ] == ["cold", "evict", "warm"]
    assert warm["capture_launch_contract"]["label_launch_index"] == 0

    vendor_bf16 = by_slug["vendor_gemm_cublas"]["invocation"]
    assert vendor_bf16["label_app_args"] == [
        "--m", "320", "--n", "320", "--k", "320", "--seed", "42",
        "--shape-point", "trace_s16816_compatible_m320",
    ]
    assert vendor_bf16["trace_app_args"] == vendor_bf16["label_app_args"]
    assert vendor_bf16["executable_sha256"] == (
        "919a9461e1decd9df27437056ab8172da648e811a938cfc8a9f9c269ed07be15"
    )
    assert vendor_bf16["logical_workload"] == {
        "kind": "shape_point", "value": "trace_s16816_compatible_m320",
    }
    vendor_override = vendor_bf16["matrix_capture_override"]
    assert vendor_override["schema_version"] == 1
    assert vendor_override["authority_shape_point"] == "trace_00"
    assert vendor_override["capture_shape_point"] == (
        "trace_s16816_compatible_m320"
    )
    assert vendor_override["shape_file_path"] == (
        "workloads/simbench_bw/shapes/vendor_gemm_cublas.toml"
    )
    assert vendor_override["reason"] == (
        "authority_shape_selects_h3_tracer_incompatible_movm_kernel"
    )
    assert vendor_bf16["capture_launch_contract"]["trace_launch_count"] == 1

    for slug, expected in bcm.VENDOR_RUNTIME_CONTROL_CONTRACTS.items():
        assert by_slug[slug]["control_bits"] == expected
        for key, value in bcm.STATIC_CONTROL_TOOLCHAIN.items():
            assert expected[key] == value
    assert by_slug["vendor_gemm_cublas"]["control_bits"]["inventory_binding"] == (
        "nvbit_stable_static_module"
    )
    assert by_slug["vendor_sgemm_cublas"]["control_bits"]["inventory_binding"] == (
        "ncu_report_exact_text_section"
    )
    assert by_slug["vendor_sgemm_cublas"]["control_bits"]["expected_outcome"] == (
        "matrix_pinned_runtime_module_controls_emitted"
    )
    strided = by_slug["vendor_sgemm_strided_batched_cublas"]["control_bits"]
    assert strided["inventory_binding"] == "ncu_report_exact_text_section"
    assert strided["runtime_kernel"] == "ampere_sgemm_128x128_nn___0"
    assert strided["raw_text_sha256"] == (
        "3744493afe93632f667f166e1168dc8047c56ebd938296f0b46cd9f57b3ed213"
    )
    assert strided["static_instruction_count"] == 1208
    for slug in set(bcm.SIMBENCH_CAPTURE_OVERRIDES) - {"warm_cold_pair_relaunch"}:
        assert by_slug[slug]["invocation"]["trace_cardinality_contract"] == (
            "matrix_capture_args_override_v1"
        )


def test_sol_static_control_toolchain_is_closed_and_host_verified():
    bcm._verify_static_control_toolchain(bcm.SOL_STATIC_CONTROL_REBIND)
    drifted = copy.deepcopy(bcm.SOL_STATIC_CONTROL_REBIND)
    drifted["allowed_code_arches"].reverse()
    with pytest.raises(bcm.MatrixBuildError, match="rebind contract drift"):
        bcm._verify_static_control_toolchain(drifted)


def test_strided_batched_vendor_control_dispatches_post_trace(matrix):
    cell = next(
        row for row in matrix["cells"]
        if row["identity"] == {
            "roster_source": "simbench_bw",
            "slug": "vendor_sgemm_strided_batched_cublas",
        }
    )
    budgets = cc.load_trace_budgets(cc.DEFAULT_CELL_BUDGETS)
    binary, invocation, _ = cc.validate_direct_cell(
        cell, cell_budgets=budgets,
    )
    assert binary == _REPO / invocation["executable_path"]

    wrong = copy.deepcopy(cell)
    wrong["control_bits"] = {
        "policy": "direct_executable",
        "input_path": invocation["executable_path"],
        "input_sha256": invocation["executable_sha256"],
    }
    with pytest.raises(cc.ChunkRefusal, match="control_policy_not_implemented"):
        cc.validate_direct_cell(wrong, cell_budgets=budgets)


def test_sol_bridge_binds_exact_native_and_reduced_workloads(matrix):
    sol = [
        cell for cell in matrix["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    ]
    assert len(sol) == 74
    native = [
        cell for cell in sol
        if cell["invocation"]["logical_workload"]["workload_scope"] == "native"
    ]
    reduced = [
        cell for cell in sol
        if cell["invocation"]["logical_workload"]["workload_scope"]
        == bcm.SOL_REDUCED_SHAPE
    ]
    assert len(native) == 73
    assert [cell["identity"]["slug"] for cell in reduced] == [
        "problem_028_gqa_rotary_attention_core_backward",
    ]
    for cell in sol:
        inv = cell["invocation"]
        logical = inv["logical_workload"]
        adapter = inv["adapter_contract"]
        assert inv["adapter_kind"] == "sol_execbench"
        assert inv["authority_reference_path"].endswith("/solution.json")
        assert inv["executable_path"] == bcm.SOL_EXECBENCH_PATH.as_posix()
        assert inv["authority_reference_path"] != inv["executable_path"]
        assert inv["target_processes"] == "all"
        assert logical["expected_workloads"] > 0
        evidence = logical["evaluation_evidence"]
        assert evidence["status"] == "runnable"
        assert evidence["returncode"] == 0
        assert evidence["expected_workloads"] == logical["expected_workloads"]
        assert evidence["detail"] == (
            f"{logical['expected_workloads']}/{logical['expected_workloads']} "
            "workloads passed"
        )
        assert adapter["forwarded_argv"] == inv["trace_app_args"] == inv["label_app_args"]
        assert adapter["contract_version"] == 3
        assert adapter["shape_forwarded_to_target"] is False
        assert adapter["label_driver_bridge_status"] == "implemented"
        assert adapter["persistent_stage_status"] == "implemented"
        assert adapter["capture_request_environment"] == (
            "SOL_EXECBENCH_CAPTURE_REQUEST"
        )
        assert adapter["environment"] == {
            "CUTLASS_DIR": bcm.SOL_CUTLASS_PATH.as_posix(),
            "PATH_prepend": bcm.SOL_EXECBENCH_PATH.parent.as_posix(),
            "NVCC_CCBIN": "g++-14",
        }
        assert inv["evaluation_cardinality_contract"] == (
            "all_effective_workloads_passed_exact"
        )
        assert inv["trace_cardinality_contract"] == (
            "one_pinned_candidate_kernel_launch_exact"
        )
        assert inv["capture_launch_contract"] == {
            "schema_version": 1,
            "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 0,
            "label_launch_index": 0,
            "label_measured_reps": 1,
            "trace_launch_count": 1,
            "label_trace_launch_index": 0,
        }
        assert inv["capture_target"] == {
            "source_workload_index": 0,
            "staged_workload_index": 0,
            "candidate_user_fn_calls": 1,
            "candidate_kernel_ordinal": 0,
            "capture_launches": 1,
            "profiler_range": "cudaProfilerStartStop",
            "setup_outside_profiler_range": True,
            "timing_methodology": "cuda_events",
        }
        assert not any(
            token == "--shape-point" or token.startswith("--shape-point=")
            for token in adapter["forwarded_argv"]
        )
        assert cell["control_bits"] == {
            "policy": "post_trace_module_or_typed",
            "module_discovery": "nvbit_runtime_module_inventory",
            "required_outcomes": list(bcm.SOL_CONTROL_OUTCOMES),
            "static_rebind": bcm.SOL_STATIC_CONTROL_REBIND,
        }
        assert "input_path" not in cell["control_bits"]
        assert "sol_label_shape_bridge_not_implemented" not in (
            cell["capture_readiness"]["blocking_reasons"]
        )

    for cell in native:
        inv = cell["invocation"]
        logical = inv["logical_workload"]
        assert logical["value"] == bcm.SOL_NATIVE_SHAPE
        assert logical["workload_file"] == f"{logical['benchmark_dir']}/workload.jsonl"
        assert "reduced_workload_provenance" not in logical
        assert logical["evaluation_evidence"]["source_kind"] == "native_runnability"
        assert inv["trace_app_args"] == [
            logical["benchmark_dir"],
            "--solution", inv["authority_reference_path"],
            "--json",
        ]

    cell = reduced[0]
    inv = cell["invocation"]
    logical = inv["logical_workload"]
    provenance = logical["reduced_workload_provenance"]
    assert logical["value"] == bcm.SOL_REDUCED_SHAPE
    assert logical["workload_file"] == provenance["reduced_workload_ref"]
    assert logical["workload_file"] == (
        "data/local_labels/campaign/sol_reduced_workloads/L2/"
        "problem_028_gqa_rotary_attention_core_backward__h3.jsonl"
    )
    assert provenance["source_path"] == bcm.SOL_SHRINK_PROVENANCE_PATH.as_posix()
    assert provenance["source_sha256"] == matrix["input_pins"][
        "sol_shrink_provenance"
    ]["sha256"]
    assert provenance["native_run_status"] == "host_capacity_oom"
    assert logical["evaluation_evidence"]["source_kind"] == (
        "reduced_shrink_evaluation"
    )
    assert inv["trace_app_args"] == [
        "--definition", logical["definition_file"],
        "--workload", logical["workload_file"],
        "--solution", inv["authority_reference_path"],
        "--json",
    ]


def test_sol_logical_workload_hashes_and_counts_rederive(matrix):
    sol = [
        cell for cell in matrix["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    ]
    assert matrix["input_pins"]["sol_shrink_provenance"] == {
        "path": bcm.SOL_SHRINK_PROVENANCE_PATH.as_posix(),
        "sha256": bcm.sha256_file(_REPO / bcm.SOL_SHRINK_PROVENANCE_PATH),
    }
    assert matrix["input_pins"]["sol_runnability"] == {
        "path": bcm.SOL_RUNNABILITY_PATH.as_posix(),
        "sha256": bcm.sha256_file(_REPO / bcm.SOL_RUNNABILITY_PATH),
    }
    fingerprint = json.loads(
        (_REPO / bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH).read_text()
    )
    assert matrix["input_pins"]["sol_native_env_fingerprint"] == {
        "path": bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH.as_posix(),
        "sha256": bcm.sha256_file(
            _REPO / bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH
        ),
        "native_env_hash": fingerprint["native_env_hash"],
    }
    assert fingerprint["native_env_hash"] == bcm.compute_native_env_hash(
        fingerprint["components"]
    )
    assert fingerprint["components"]["nvcc_ccbin"] == (
        bcm.SOL_PINNED_NVCC_CCBIN
    )
    for cell in sol:
        inv = cell["invocation"]
        logical = inv["logical_workload"]
        solution = _REPO / inv["authority_reference_path"]
        definition = _REPO / logical["definition_file"]
        benchmark = _REPO / logical["benchmark_dir"]
        workload = _REPO / logical["workload_file"]
        assert bcm.sha256_file(solution) == inv["authority_reference_sha256"]
        assert bcm.sha256_file(definition) == logical["definition_file_sha256"]
        assert bcm.tree_sha256(benchmark) == logical["benchmark_tree_sha256"]
        assert bcm.sha256_file(workload) == logical["workload_file_sha256"]
        assert bcm._count_workloads(workload) == logical["expected_workloads"]
        assert json.loads(solution.read_text())["definition"] == logical[
            "benchmark_definition"
        ]
        assert json.loads(definition.read_text())["name"] == logical[
            "benchmark_definition"
        ]


def test_sol_native_env_contract_rederives_exact_certified_fingerprint():
    contract = bcm._sol_native_env_contract(_REPO)
    artifact = json.loads(
        (_REPO / bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH).read_text()
    )
    assert contract == {
        "source_sha256": bcm.sha256_file(
            _REPO / bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH
        ),
        "native_env_hash": artifact["native_env_hash"],
        "components": artifact["components"],
    }
    assert contract["native_env_hash"] == bcm._sol_shrink_contract(
        _REPO
    )["native_env_hash"]
    assert contract["native_env_hash"] == bcm._sol_runnability_contract(
        _REPO
    )["native_env_hash"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("hash_drift", "fingerprint hash mismatch"),
        ("degraded", "fingerprint is degraded"),
        ("component_set", "component set mismatch"),
        ("compiler_spelling", "compiler pin mismatch"),
    ],
)
def test_sol_native_env_contract_rejects_drift(monkeypatch, mutation, error):
    source = (_REPO / bcm.SOL_NATIVE_ENV_FINGERPRINT_PATH).resolve()
    doc = json.loads(source.read_text())
    if mutation == "hash_drift":
        doc["components"]["cudnn_version"] = "tampered"
    elif mutation == "degraded":
        doc["components"]["cudnn_version"] = "unknown"
        doc["missing_components"] = ["cudnn_version"]
        doc["degraded"] = True
        doc["native_env_hash"] = bcm.compute_native_env_hash(doc["components"])
    elif mutation == "component_set":
        del doc["components"]["c_compiler"]
    elif mutation == "compiler_spelling":
        doc["components"]["nvcc_ccbin"] = "/usr/bin/g++-14"
        doc["native_env_hash"] = bcm.compute_native_env_hash(doc["components"])
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
    real_load = bcm._load_json

    def mutated_load(path: Path) -> dict:
        if path.resolve() == source:
            return copy.deepcopy(doc)
        return real_load(path)

    monkeypatch.setattr(bcm, "_load_json", mutated_load)
    with pytest.raises(bcm.MatrixBuildError, match=error):
        bcm._sol_native_env_contract(_REPO)


def test_sol_invocation_rejects_authority_fingerprint_disagreement():
    authority = json.loads((_REPO / bcm.AUTHORITY_PATH).read_text())
    cell = next(
        row for row in authority["cells"]
        if row["roster_source"] == "sol_execbench"
        and row["resolution_status"] == "runnable"
    )
    with pytest.raises(
        bcm.MatrixBuildError, match="fingerprint/native environment mismatch"
    ):
        bcm._sol_invocation(
            cell,
            _REPO,
            native_env_contract={"native_env_hash": "f" * 64},
        )


def test_full_registry_identity_has_no_silent_metric_omission(matrix):
    reg = matrix["metric_registry"]
    assert reg["metric_count"] == 61 == len(reg["metrics"])
    assert reg["metric_identity_sha256"] == bcm.canonical_sha256(reg["metrics"])
    keys = [metric["key"] for metric in reg["metrics"]]
    assert len(keys) == len(set(keys))
    selected = [m for m in reg["metrics"] if m["capture_status"] == "selected"]
    omitted = reg["collection"]["not_collected_registry_metrics"]
    assert len(selected) == reg["collection"]["selected_registry_metric_count"]
    assert len(selected) + len(omitted) == 61
    assert matrix["input_pins"]["metric_registry"]["sha256"] == bcm.sha256_file(
        _REPO / bcm.METRIC_REGISTRY_PATH
    )


def test_all_runnable_budgets_are_explicit_and_fit_retention_contract(matrix):
    assert matrix["input_pins"]["per_cell_trace_budgets"]["defaults_are_enforcement"] is False
    runnable = [cell for cell in matrix["cells"] if cell["disposition"] == "runnable"]
    missing = [cell for cell in runnable if cell["trace_budget"]["status"] == "missing"]
    declared = [cell for cell in runnable if cell["trace_budget"]["status"] == "declared"]
    assert missing == []
    assert len(declared) == len(runnable) == 122
    assert matrix["counts"]["budget_missing"] == 0
    assert matrix["counts"]["capture_ready"] == 122
    assert matrix["counts"]["capture_blocked"] == 0
    aggregate = matrix["input_pins"]["aggregate_retention_budget"]
    assert aggregate["declared_runnable_max_trace_bytes"] == sum(
        cell["trace_budget"]["max_trace_bytes"] for cell in runnable
    )
    assert aggregate["retained_headroom_bytes"] == (
        aggregate["retention_ceiling_bytes"]
        - aggregate["declared_runnable_max_trace_bytes"]
        - aggregate["metadata_margin_bytes"]
    )
    assert aggregate["retained_headroom_bytes"] >= 0
    assert aggregate["preflight_required_free_bytes"] == (
        aggregate["retention_ceiling_bytes"]
        + aggregate["raw_staging_margin_bytes"]
    )
    silu = next(
        cell for cell in runnable
        if cell["identity"] == {
            "roster_source": "portable_lcd", "slug": "silu_fwd_bf16_sm80",
        }
    )
    assert silu["trace_budget"]["status"] == "declared"
    assert silu["capture_readiness"]["status"] == "ready"


def test_histogram_wall_envelope_is_measured_and_cell_specific(matrix):
    portable = {
        cell["identity"]["slug"]: cell
        for cell in matrix["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "portable_lcd"
    }
    histogram = portable["ac4_global_atomic_histogram"]["trace_budget"]
    assert histogram == {
        "source_path": bcm.TRACE_BUDGETS_PATH.as_posix(),
        "source_sha256": bcm.sha256_file(_REPO / bcm.TRACE_BUDGETS_PATH),
        "status": "declared",
        "max_trace_bytes": 2147483648,
        "max_wall_time_s": 5400,
        "sampling_policy": "off",
    }
    assert {
        row["trace_budget"]["max_wall_time_s"]
        for slug, row in portable.items()
        if slug != "ac4_global_atomic_histogram"
    } == {3600}


def test_sol_wall_envelopes_are_measured_and_cell_specific(matrix):
    sol = {
        cell["identity"]["slug"]: cell
        for cell in matrix["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    }
    problem_027 = sol[
        "problem_027_grouped_query_attention_with_yarn_rope_and_qk_norm"
    ]["trace_budget"]
    assert problem_027 == {
        "source_path": bcm.TRACE_BUDGETS_PATH.as_posix(),
        "source_sha256": bcm.sha256_file(_REPO / bcm.TRACE_BUDGETS_PATH),
        "status": "declared",
        "max_trace_bytes": 1073741824,
        "max_wall_time_s": 10800,
        "sampling_policy": "off",
    }
    problem_039 = sol[
        "problem_039_adaptive_instance_normalization_2d_backward"
    ]["trace_budget"]
    assert problem_039 == {
        "source_path": bcm.TRACE_BUDGETS_PATH.as_posix(),
        "source_sha256": bcm.sha256_file(_REPO / bcm.TRACE_BUDGETS_PATH),
        "status": "declared",
        "max_trace_bytes": 1073741824,
        "max_wall_time_s": 21600,
        "sampling_policy": "off",
    }
    problem_040 = sol[
        "problem_040_altup_predict_correction_cycle_backward"
    ]["trace_budget"]
    assert problem_040 == {
        "source_path": bcm.TRACE_BUDGETS_PATH.as_posix(),
        "source_sha256": bcm.sha256_file(_REPO / bcm.TRACE_BUDGETS_PATH),
        "status": "declared",
        "max_trace_bytes": 1073741824,
        "max_wall_time_s": 18000,
        "sampling_policy": "off",
    }
    measured_exceptions = {
        "problem_027_grouped_query_attention_with_yarn_rope_and_qk_norm",
        "problem_039_adaptive_instance_normalization_2d_backward",
        "problem_040_altup_predict_correction_cycle_backward",
    }
    assert {
        row["trace_budget"]["max_wall_time_s"]
        for slug, row in sol.items()
        if slug not in measured_exceptions
    } == {7200}


def test_authority_guard_rejects_zero_mismatch_pending_and_stale(matrix):
    authority = json.loads((_REPO / bcm.AUTHORITY_PATH).read_text())
    empty = copy.deepcopy(authority)
    empty["cells"] = []
    empty["counts"]["universe"] = 0
    with pytest.raises(bcm.MatrixBuildError, match="zero cells"):
        bcm._authority_invariants(empty, repo_root=_REPO, require_head_match=False)

    mismatch = copy.deepcopy(authority)
    mismatch["counts"]["universe"] -= 1
    with pytest.raises(bcm.MatrixBuildError, match="universe mismatch"):
        bcm._authority_invariants(mismatch, repo_root=_REPO, require_head_match=False)

    pending = copy.deepcopy(authority)
    runnable = next(c for c in pending["cells"] if c["resolution_status"] == "runnable")
    runnable["resolution_status"] = "pending_build"
    pending["counts"]["runnable"] -= 1
    pending["counts"]["pending_build"] += 1
    with pytest.raises(bcm.MatrixBuildError, match="pending rows"):
        bcm._authority_invariants(pending, repo_root=_REPO, require_head_match=False)

    stale = copy.deepcopy(authority)
    stale["source_commit"] = "0" * 40
    with pytest.raises(bcm.MatrixBuildError, match="stale authority"):
        bcm._authority_invariants(stale, repo_root=_REPO, require_head_match=True)


def test_sol_bridge_rejects_missing_shape_and_stale_solution_hash():
    authority = json.loads((_REPO / bcm.AUTHORITY_PATH).read_text())
    native = next(
        cell for cell in authority["cells"]
        if cell["roster_source"] == "sol_execbench"
        and cell["resolution_status"] == "runnable"
        and cell["shape_point"] == bcm.SOL_NATIVE_SHAPE
    )
    contract = bcm._sol_shrink_contract(_REPO)

    missing_shape = copy.deepcopy(native)
    del missing_shape["shape_point"]
    with pytest.raises(bcm.MatrixBuildError, match="unsupported/missing workload shape"):
        bcm._sol_invocation(
            missing_shape, _REPO, shrink_contract=contract,
        )

    stale_solution = copy.deepcopy(native)
    stale_solution["binary_sha256"] = "0" * 64
    with pytest.raises(bcm.MatrixBuildError, match="solution hash mismatch"):
        bcm._sol_invocation(
            stale_solution, _REPO, shrink_contract=contract,
        )


def test_sol_bridge_rejects_wrong_definition_binding(monkeypatch):
    authority = json.loads((_REPO / bcm.AUTHORITY_PATH).read_text())
    native = next(
        cell for cell in authority["cells"]
        if cell["roster_source"] == "sol_execbench"
        and cell["resolution_status"] == "runnable"
        and cell["shape_point"] == bcm.SOL_NATIVE_SHAPE
    )
    contract = bcm._sol_shrink_contract(_REPO)
    real_load = bcm._load_json
    solution_doc = real_load(_REPO / native["binary"])
    definition_path = (
        _REPO / bcm.SOL_BENCHMARK_ROOT
        / bcm.SOL_SUBSET_MAP[native["category"]]
        / solution_doc["definition"] / "definition.json"
    ).resolve()

    def wrong_definition(path: Path) -> dict:
        value = real_load(path)
        if path.resolve() == definition_path:
            value = copy.deepcopy(value)
            value["name"] = "wrong_definition"
        return value

    monkeypatch.setattr(bcm, "_load_json", wrong_definition)
    with pytest.raises(bcm.MatrixBuildError, match="solution/benchmark definition mismatch"):
        bcm._sol_invocation(native, _REPO, shrink_contract=contract)


def test_sol_bridge_rejects_reduced_to_native_misroute_and_fake_shape_token():
    authority = json.loads((_REPO / bcm.AUTHORITY_PATH).read_text())
    reduced = next(
        cell for cell in authority["cells"]
        if cell["roster_source"] == "sol_execbench"
        and cell["slug"] == "problem_028_gqa_rotary_attention_core_backward"
    )
    contract = bcm._sol_shrink_contract(_REPO)
    misrouted = copy.deepcopy(reduced)
    misrouted["shape_point"] = bcm.SOL_NATIVE_SHAPE
    with pytest.raises(bcm.MatrixBuildError, match="misrouted from shrink provenance"):
        bcm._sol_invocation(misrouted, _REPO, shrink_contract=contract)

    with pytest.raises(bcm.MatrixBuildError, match="fabricated shape token"):
        bcm._validate_sol_forwarded_argv(
            ["benchmark", "--shape-point", "workloads"],
            ident={"roster_source": "sol_execbench", "slug": "fake"},
        )


def test_schema_and_semantics_reject_missing_or_faked_invocation(matrix):
    missing = copy.deepcopy(matrix)
    row = next(cell for cell in missing["cells"] if cell["disposition"] == "runnable")
    del row["invocation"]
    _reseal(missing)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, _SCHEMA)

    fake_sol = copy.deepcopy(matrix)
    sol = next(
        cell for cell in fake_sol["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    )
    sol["control_bits"] = {
        "policy": "direct_executable",
        "input_path": sol["invocation"]["authority_reference_path"],
        "input_sha256": sol["invocation"]["authority_reference_sha256"],
    }
    _reseal(fake_sol)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(fake_sol, _SCHEMA)

    missing_static_rebind = copy.deepcopy(matrix)
    sol = next(
        cell for cell in missing_static_rebind["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    )
    del sol["control_bits"]["static_rebind"]
    _reseal(missing_static_rebind)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_static_rebind, _SCHEMA)

    for slug in bcm.VENDOR_RUNTIME_CONTROL_CONTRACTS:
        fake_vendor = copy.deepcopy(matrix)
        vendor = next(
            cell for cell in fake_vendor["cells"]
            if cell["identity"] == {
                "roster_source": "simbench_bw", "slug": slug,
            }
        )
        vendor["control_bits"] = {
            "policy": "direct_executable",
            "input_path": vendor["invocation"]["executable_path"],
            "input_sha256": vendor["invocation"]["executable_sha256"],
        }
        _reseal(fake_vendor)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(fake_vendor, _SCHEMA)

    direct_sol = copy.deepcopy(matrix)
    sol = next(
        cell for cell in direct_sol["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    )
    invocation = sol["invocation"]
    invocation["adapter_kind"] = "direct_binary"
    invocation["shape_delivery"] = "application_args"
    invocation["trace_cardinality_contract"] = "authority_args_exact"
    invocation["logical_workload"] = {"kind": "shape_point", "value": "workloads"}
    del invocation["adapter_contract"]
    _reseal(direct_sol)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(direct_sol, _SCHEMA)

    missing_override = copy.deepcopy(matrix)
    overridden = next(
        cell for cell in missing_override["cells"]
        if cell["identity"]["slug"] == "cp_async_shared_tile_copy"
    )
    del overridden["invocation"]["matrix_capture_override"]
    _reseal(missing_override)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_override, _SCHEMA)

    extra_override = copy.deepcopy(matrix)
    ordinary = next(
        cell for cell in extra_override["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "simbench_bw"
        and cell["identity"]["slug"] not in bcm.SIMBENCH_CAPTURE_OVERRIDES
    )
    ordinary["invocation"]["matrix_capture_override"] = copy.deepcopy(
        overridden.get("invocation", {}).get("matrix_capture_override")
        or next(
            cell for cell in matrix["cells"]
            if cell["identity"]["slug"] == "cp_async_shared_tile_copy"
        )["invocation"]["matrix_capture_override"]
    )
    _reseal(extra_override)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(extra_override, _SCHEMA)

    bad_sequence = copy.deepcopy(matrix)
    warm = next(
        cell for cell in bad_sequence["cells"]
        if cell["identity"]["slug"] == "warm_cold_pair_relaunch"
    )
    warm["invocation"]["capture_launch_contract"]["trace_launch_sequence"].reverse()
    _reseal(bad_sequence)
    assert "matrix_content_stale_or_tampered" in bcm.validate_matrix(
        bad_sequence, _REPO
    )

    mislabeled_vendor_shape = copy.deepcopy(matrix)
    vendor = next(
        cell for cell in mislabeled_vendor_shape["cells"]
        if cell["identity"] == {
            "roster_source": "simbench_bw", "slug": "vendor_gemm_cublas",
        }
    )
    vendor["invocation"]["logical_workload"]["value"] = "trace_00"
    _reseal(mislabeled_vendor_shape)
    assert "matrix_content_stale_or_tampered" in bcm.validate_matrix(
        mislabeled_vendor_shape, _REPO
    )

    missing_fingerprint_pin = copy.deepcopy(matrix)
    del missing_fingerprint_pin["input_pins"]["sol_native_env_fingerprint"]
    _reseal(missing_fingerprint_pin)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_fingerprint_pin, _SCHEMA)

    missing_compiler_pin = copy.deepcopy(matrix)
    sol = next(
        cell for cell in missing_compiler_pin["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    )
    del sol["invocation"]["adapter_contract"]["environment"]["NVCC_CCBIN"]
    _reseal(missing_compiler_pin)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_compiler_pin, _SCHEMA)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", "ambient_module_guess"),
        ("allowed_code_arches", ["sm_86", "sm_80"]),
        ("ncu_report_binding", "function_name_only"),
        ("nvcc_sha256", "0" * 64),
        ("cuobjdump_sha256", "f" * 64),
    ],
)
def test_schema_rejects_static_rebind_contract_mutations(matrix, field, value):
    mutated = copy.deepcopy(matrix)
    sol = next(
        cell for cell in mutated["cells"]
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    )
    sol["control_bits"]["static_rebind"][field] = value
    _reseal(mutated)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated, _SCHEMA)


def test_resealed_pin_or_chunk_tamper_is_detected(matrix):
    stale = copy.deepcopy(matrix)
    stale["authority"]["sha256"] = "f" * 64
    _reseal(stale)
    assert "matrix_content_stale_or_tampered" in bcm.validate_matrix(stale, _REPO)

    stale_fingerprint = copy.deepcopy(matrix)
    stale_fingerprint["input_pins"]["sol_native_env_fingerprint"][
        "sha256"
    ] = "e" * 64
    _reseal(stale_fingerprint)
    assert "matrix_content_stale_or_tampered" in bcm.validate_matrix(
        stale_fingerprint, _REPO
    )

    duplicate = copy.deepcopy(matrix)
    duplicate["chunks"][1]["cells"][0] = copy.deepcopy(duplicate["chunks"][0]["cells"][0])
    duplicate["chunks"][1]["membership_sha256"] = bcm.canonical_sha256(
        duplicate["chunks"][1]["cells"]
    )
    duplicate["chunking"]["runnable_membership_sha256"] = bcm.canonical_sha256([
        member for chunk in duplicate["chunks"] for member in chunk["cells"]
    ])
    _reseal(duplicate)
    problems = bcm.validate_matrix(duplicate, _REPO)
    assert "chunk_membership_uniqueness" in problems
    assert "chunk_runnable_set_equality" in problems
