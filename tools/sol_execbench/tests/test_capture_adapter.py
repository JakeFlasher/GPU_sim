# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from sol_execbench.core.bench.reward_hack import (
    RewardHackDetected,
    check_eval_integrity,
    snapshot_critical_functions,
)
from sol_execbench.driver import capture_adapter as adapter


CERTIFIED_ENVIRONMENT = {
    "CUDA_HOME": "/usr/local/cuda-13.2",
    "NVCC_CCBIN": "g++-14",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
}
LEGACY_CERTIFIED_ENVIRONMENT = {"NVCC_CCBIN": "g++-14"}
CERTIFIED_ENVIRONMENT_SHA256 = adapter.environment_contract_sha256(
    CERTIFIED_ENVIRONMENT
)
LEGACY_CERTIFIED_ENVIRONMENT_SHA256 = (
    adapter.legacy_environment_contract_sha256(LEGACY_CERTIFIED_ENVIRONMENT)
)
ADMITTED_AMBIENT = {"PATH": "/usr/bin", "TEST_SENTINEL": "preserved"}
NON_FIB_CORPUS = {
    "policy": "verified_fib_only",
    "category": "L1",
    "environment_variable": "FLASHINFER_TRACE_DIR",
    "injection_policy": "forbidden_remove_ambient",
    "root": None,
    "data_annex_manifest_path": "evidence/manifests/data_annex.sha256",
    "data_annex_manifest_sha256": "6" * 64,
    "filtered_manifest_sha256": "7" * 64,
    "file_count": 5244,
    "safetensors_count": 2611,
}
NON_FIB_CORPUS_SHA256 = hashlib.sha256(
    (json.dumps(NON_FIB_CORPUS, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()


@pytest.fixture(autouse=True)
def _fixture_corpus_verifier(monkeypatch):
    """Keep adapter fixtures local; shared corpus bytes have separate tests."""
    def verify(contract, *, repo_root):
        assert repo_root == adapter._REPO_ROOT
        if contract != NON_FIB_CORPUS:
            raise adapter.PrerequisiteContractError("fixture corpus contract drift")
        return None

    monkeypatch.setattr(adapter, "verify_flashinfer_capture_contract", verify)
    for name in (
        "prepare_capture_stage", "execute_capture_stage", "validate_capture_stage",
    ):
        original = getattr(adapter, name)

        def with_fixture_contract(*args, __original=original, **kwargs):
            kwargs.setdefault(
                "flashinfer_corpus_contract", copy_json(NON_FIB_CORPUS)
            )
            return __original(*args, **kwargs)

        monkeypatch.setattr(adapter, name, with_fixture_contract)


def copy_json(value):
    return json.loads(json.dumps(value))


def _eval_template_critical_names(capture_contract):
    template_path = Path(adapter.__file__).parent / "templates" / "eval_driver.py"
    parsed = ast.parse(template_path.read_text())
    assignment_index = next(
        index
        for index, node in enumerate(parsed.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CRITICAL_NAMES"
            for target in node.targets
        )
    )
    critical_names_nodes = parsed.body[assignment_index : assignment_index + 2]
    assert isinstance(critical_names_nodes[1], ast.If)
    namespace = {"_CAPTURE_CONTRACT": capture_contract}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=critical_names_nodes, type_ignores=[])
            ),
            str(template_path),
            "exec",
        ),
        namespace,
    )
    return namespace["_CRITICAL_NAMES"]


def _eval_template_correctness_loops():
    template_path = Path(adapter.__file__).parent / "templates" / "eval_driver.py"
    parsed = ast.parse(template_path.read_text())
    workload_loop = next(
        node
        for node in parsed.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_workload"
    )
    correctness_loop = next(
        node
        for node in workload_loop.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_round"
    )
    return template_path, workload_loop, correctness_loop


def _build_template_compiler_helpers(
    *, environment=None, nvcc="/usr/local/cuda-13.2/bin/nvcc",
    torch_cuda_home="/usr/local/cuda-13.2",
):
    template_path = Path(adapter.__file__).parent / "templates" / "build_ext.py"
    parsed = ast.parse(template_path.read_text())
    selected = []
    for node in parsed.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "CERTIFIED_ENVIRONMENT"
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "_validated_compiler_binding", "_certified_cuda_cflags",
                "_prepare_torch_extension_environment",
            }
        ):
            selected.append(node)
    namespace = {
        "os": SimpleNamespace(
            environ=dict(CERTIFIED_ENVIRONMENT if environment is None else environment),
            access=os.access,
            X_OK=os.X_OK,
        ),
        "Path": Path,
        "shutil": SimpleNamespace(
            which=lambda name: nvcc if name == "nvcc" else "/usr/bin/g++-14"
        ),
        "ext": SimpleNamespace(CUDA_HOME=torch_cuda_home),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(template_path),
            "exec",
        ),
        namespace,
    )
    return namespace


def test_build_template_emits_one_validated_ccbin_and_rejects_overrides():
    helpers = _build_template_compiler_helpers()
    bind = helpers["_certified_cuda_cflags"]
    assert bind(["-O3"]) == ["-O3", "-ccbin", "g++-14"]
    for override in (
        "-ccbin", "-ccbin=g++-15", "--compiler-bindir",
        "--compiler-bindir=/usr/bin/g++-15",
    ):
        with pytest.raises(RuntimeError, match="may not override"):
            bind([override])
    helpers["_prepare_torch_extension_environment"](host_cxx="g++-14")
    assert "CC" not in helpers["os"].environ
    assert helpers["os"].environ["CXX"] == "g++-14"
    assert helpers["os"].environ["CUDAHOSTCXX"] == "g++-14"


def test_torch_nvcc_command_assembles_exactly_one_certified_ccbin(
    tmp_path, monkeypatch,
):
    torch_extension = pytest.importorskip("torch.utils.cpp_extension")
    helpers = _build_template_compiler_helpers()
    cuda_flags = helpers["_certified_cuda_cflags"](["-O3"])
    helpers["_prepare_torch_extension_environment"](host_cxx="g++-14")
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("CXX", "g++-14")
    monkeypatch.setenv("CUDAHOSTCXX", "g++-14")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")
    ninja = tmp_path / "build.ninja"
    torch_extension._write_ninja_file_to_build_library(
        path=str(ninja),
        name="fixture",
        sources=[str(tmp_path / "fixture.cu")],
        extra_cflags=[],
        extra_cuda_cflags=cuda_flags,
        extra_sycl_cflags=[],
        extra_ldflags=[],
        extra_include_paths=[],
        with_cuda=True,
        with_sycl=False,
        is_standalone=False,
    )
    cuda_line = next(
        line for line in ninja.read_text().splitlines()
        if line.startswith("cuda_cflags = ")
    )
    assert cuda_line.count("-ccbin") == 1
    assert "-ccbin g++-14" in cuda_line
    assert "-ccbin gcc-14" not in cuda_line


def test_build_template_refuses_cuda_home_or_nvcc_resolution_drift():
    helpers = _build_template_compiler_helpers(
        environment={**CERTIFIED_ENVIRONMENT, "CUDA_HOME": "/opt/cuda"}
    )
    with pytest.raises(RuntimeError, match="not certified"):
        helpers["_certified_cuda_cflags"]([])

    helpers = _build_template_compiler_helpers(nvcc="/opt/cuda/bin/nvcc")
    with pytest.raises(RuntimeError, match="does not resolve"):
        helpers["_certified_cuda_cflags"]([])


def test_real_console_script_loads_shared_comparator_without_ambient_pythonpath(
    tmp_path,
):
    repo = Path(__file__).resolve().parents[3]
    executable = repo / "tools/sol_execbench/.venv/bin/sol-execbench"
    observed = subprocess.run(
        [str(executable), "--help"],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert observed.returncode == 0, observed.stderr
    assert "Usage:" in observed.stdout


def _write_inputs(
    root: Path, *, dps: bool = True, rows: int = 2,
    tensor_descriptions: bool = True,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    definition = root / "definition.json"
    workload = root / "workload.jsonl"
    solution = root / "solution.json"
    executable = root / "sol-execbench"
    definition.write_text(
        json.dumps(
            {
                "name": "capture_test",
                "axes": {"n": {"type": "var", "description": None}},
                "inputs": {
                    "x": {
                        "shape": ["n"],
                        "dtype": "float32",
                        **(
                            {"description": None}
                            if tensor_descriptions else {}
                        ),
                    }
                },
                "outputs": {
                    "y": {
                        "shape": ["n"],
                        "dtype": "float32",
                        **(
                            {"description": None}
                            if tensor_descriptions else {}
                        ),
                    }
                },
                "reference": "def run(x):\n    return x\n",
                "description": None,
            }
        )
    )
    workload_rows = [
        {
            "uuid": f"workload-{index}",
            "axes": {"n": index + 1},
            "inputs": {"x": {"type": "random"}},
        }
        for index in range(rows)
    ]
    workload.write_text("".join(json.dumps(row) + "\n" for row in workload_rows))
    solution.write_text(
        json.dumps(
            {
                "name": "capture_solution",
                "definition": "capture_test",
                "author": "test",
                "spec": {
                    "languages": ["cuda_cpp"],
                    "target_hardware": ["B200"],
                    "entry_point": "main.cpp::run",
                    "destination_passing_style": dps,
                    "binding": "torch",
                },
                "sources": [
                    {
                        "path": "main.cpp",
                        "content": "void capture_test_source() {}\n",
                    }
                ],
            }
        )
    )
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return {
        "definition": definition,
        "workload": workload,
        "solution": solution,
        "executable": executable,
    }


class _Compiler:
    def __init__(self):
        self.calls = 0
        self.environments = []

    def __call__(self, command, *, cwd, **kwargs):
        self.calls += 1
        self.environments.append(dict(kwargs["env"]))
        assert command[-1] == "build_ext.py"
        assert {
            key: kwargs["env"].get(key) for key in CERTIFIED_ENVIRONMENT
        } == CERTIFIED_ENVIRONMENT
        assert "NVCC_PREPEND_FLAGS" not in kwargs["env"]
        assert "NVCC_APPEND_FLAGS" not in kwargs["env"]
        (Path(cwd) / "benchmark_kernel.so").write_bytes(b"compiled-once")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _successful_execution(
    stage: Path,
    selection_relative: Path,
    result_relative: Path,
    *,
    source_index: int,
):
    def run(command, *, cwd, env, **kwargs):
        assert command == [adapter.sys.executable, str(stage / "eval_driver.py")]
        assert Path(cwd) == stage
        assert env[adapter.CAPTURE_MODE_ENV] == "1"
        assert {
            key: env.get(key) for key in CERTIFIED_ENVIRONMENT
        } == CERTIFIED_ENVIRONMENT
        assert "NVCC_PREPEND_FLAGS" not in env
        assert "NVCC_APPEND_FLAGS" not in env
        selection_path = stage / selection_relative
        result_path = stage / result_relative
        selection_sha = adapter.sha256_file(selection_path)
        selection = json.loads(selection_path.read_text())
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_kind": "sol_execbench_capture_result",
                    "capture_contract_version": 2,
                    "path_mode": "stage_relative_v1",
                    "stage_root": ".",
                    "status": "captured",
                    "capture_kind": selection["capture_kind"],
                    "selection_manifest": {
                        "path": selection_relative.as_posix(),
                        "sha256": selection_sha,
                    },
                    "prepare_manifest": selection["prepare_manifest"],
                    "definition": selection["definition"],
                    "solution": selection["solution"],
                    "source_workload_index": source_index,
                    "staged_workload_index": 0,
                    "workload_uuid": selection["workload"]["workload_uuid"],
                    "environment_contract": selection["environment_contract"],
                    "environment_contract_sha256": selection[
                        "environment_contract_sha256"
                    ],
                    "flashinfer_corpus_contract": selection[
                        "flashinfer_corpus_contract"
                    ],
                    "flashinfer_corpus_contract_sha256": selection[
                        "flashinfer_corpus_contract_sha256"
                    ],
                    "stage_definition_sha256": selection[
                        "stage_definition_sha256"
                    ],
                    "benchmark_kernel_so": selection["benchmark_kernel_so"],
                    "outside_roi": {
                        "correctness_rounds": 1,
                        "correctness_user_fn_calls": 1,
                        "warmup_user_fn_calls": 1,
                        "all_completed_before_profiler_start": True,
                    },
                    "roi": {
                        "profiler_start_calls": 1,
                        "candidate_user_fn_calls": 1,
                        "profiler_stop_calls": 1,
                        "timing_methodology": "cuda_events",
                    },
                    "correctness": {},
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _prepare(
    tmp_path: Path, *, index: int = 1, rows: int = 2,
    tensor_descriptions: bool = True,
):
    inputs = _write_inputs(
        tmp_path / "inputs",
        rows=rows,
        tensor_descriptions=tensor_descriptions,
    )
    stage = tmp_path / "persistent-stage"
    compiler = _Compiler()
    outcome = adapter.prepare_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=stage,
        source_workload_index=index,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=compiler,
    )
    return inputs, stage, compiler, outcome


def test_prepare_stages_one_exact_selected_workload_and_all_hashes(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, compiler, outcome = _prepare(tmp_path)

    assert compiler.calls == 1
    expected_ambient = dict(ADMITTED_AMBIENT)
    expected_ambient["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    assert compiler.environments == [
        adapter.build_certified_env(CERTIFIED_ENVIRONMENT, expected_ambient)
    ]
    assert outcome["status"] == "prepared"
    assert stage.is_dir()
    manifest = json.loads((stage / adapter.PREPARE_MANIFEST_NAME).read_text())
    assert manifest["capture_contract_version"] == 2
    assert manifest["selection"] == {
        "source_workload_index": 1,
        "source_workload_count": 2,
        "staged_workload_index": 0,
        "workload_uuid": "workload-1",
        "workload_axes": {"n": 2},
        "source_row_sha256": hashlib.sha256(
            (stage / "workload.jsonl").read_bytes().strip()
        ).hexdigest(),
    }
    assert len((stage / "workload.jsonl").read_text().splitlines()) == 1
    assert json.loads((stage / "workload.jsonl").read_text())["uuid"] == "workload-1"
    assert manifest["execution"]["timing_methodology"] == "cuda_events"
    assert manifest["execution"]["profiled_candidate_calls"] == 1
    assert manifest["environment_contract"] == CERTIFIED_ENVIRONMENT
    assert (
        manifest["environment_contract_sha256"]
        == CERTIFIED_ENVIRONMENT_SHA256
    )
    stage_without_digest = dict(manifest)
    stage_digest = stage_without_digest.pop("stage_definition_sha256")
    assert stage_digest == adapter._stage_definition_sha256(stage_without_digest)
    assert manifest["path_mode"] == "stage_relative_v1"
    assert manifest["stage_root"] == "."
    assert "stage_dir" not in manifest
    assert (stage / "definition.source.json").read_bytes() == inputs[
        "definition"
    ].read_bytes()
    assert manifest["source_definition_sha256"] == adapter.sha256_file(
        inputs["definition"]
    )
    assert manifest["staged_definition_sha256"] == adapter.sha256_file(
        stage / "definition.json"
    )
    assert manifest["definition_equivalence"]["rule_id"] == "SOL_DEF_EQUIV_V2"
    assert manifest["definition_equivalence"]["result"] == "equal"
    assert manifest["definition_equivalence"]["top_level_field_states"][
        "op_type"
    ] == {
        "source": "missing",
        "staged": "null",
        "source_normalized": True,
        "staged_normalized": False,
    }
    attestation_path = stage / "definition_equivalence_attestation.json"
    assert attestation_path.read_bytes() == adapter.serialize_attestation(
        manifest["definition_equivalence"]
    )
    for name in (
        "definition",
        "definition_source",
        "definition_equivalence_attestation",
        "workload",
        "solution",
        "eval_driver",
        "capture_runtime",
        "benchmark_kernel_so",
    ):
        entry = manifest["artifacts"][name]
        path = Path(entry["path"])
        assert not path.is_absolute() and (stage / path).is_file()
        assert adapter.sha256_file(stage / path) == entry["sha256"]
    for name in ("adapter", "adapter_source", "certified_env_source"):
        entry = manifest["artifacts"][name]
        path = Path(entry["path"])
        assert path.is_absolute() and path.is_file()
        assert adapter.sha256_file(path) == entry["sha256"]
    assert manifest["artifacts"]["solution_sources"]
    assert adapter.validate_capture_stage(
        stage,
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        source_workload_index=1,
        environment_contract=CERTIFIED_ENVIRONMENT,
    ) == manifest


def test_prepare_certifies_direct_tensor_description_materialization(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, compiler, outcome = _prepare(
        tmp_path, tensor_descriptions=False
    )
    assert compiler.calls == 1
    manifest = outcome["manifest"]
    source = json.loads(inputs["definition"].read_text())
    staged = json.loads((stage / "definition.json").read_text())
    assert "description" not in source["inputs"]["x"]
    assert "description" not in source["outputs"]["y"]
    assert staged["inputs"]["x"]["description"] is None
    assert staged["outputs"]["y"]["description"] is None
    attestation = manifest["definition_equivalence"]
    assert attestation["tensor_description_states"] == {
        "/inputs/x/description": {
            "source": "missing",
            "staged": "null",
            "source_normalized": True,
            "staged_normalized": False,
        },
        "/outputs/y/description": {
            "source": "missing",
            "staged": "null",
            "source_normalized": True,
            "staged_normalized": False,
        },
    }
    assert adapter.validate_capture_stage(
        stage,
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        source_workload_index=1,
        environment_contract=CERTIFIED_ENVIRONMENT,
    ) == manifest


def test_compile_environment_recovers_fingerprinted_base_from_request_transport():
    base = adapter.build_certified_env(
        CERTIFIED_ENVIRONMENT,
        {
            **ADMITTED_AMBIENT,
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
    )
    transported = {
        **base,
        adapter.CAPTURE_REQUEST_ENV: "/tmp/prepare-request.json",
    }
    assert adapter._certified_subprocess_env(
        CERTIFIED_ENVIRONMENT, transported
    ) == base


def test_compile_environment_strips_all_stale_transport_and_fixes_allocator():
    stale = {
        **ADMITTED_AMBIENT,
        "PYTORCH_ALLOC_CONF": "conflicting-value",
        **{name: "stale" for name in adapter.CAPTURE_CONTROL_ENVIRONMENTS},
    }
    observed = adapter._certified_subprocess_env(
        CERTIFIED_ENVIRONMENT, stale,
    )
    assert observed["PYTORCH_ALLOC_CONF"] == adapter.PYTORCH_ALLOC_CONF
    assert not set(adapter.CAPTURE_CONTROL_ENVIRONMENTS).intersection(observed)


def test_execution_transport_is_added_only_after_certified_base_recovery():
    transport = {
        adapter.CAPTURE_MODE_ENV: "1",
        "SOL_EXECBENCH_CAPTURE_SELECTION_SHA256": "a" * 64,
    }
    observed = adapter._certified_subprocess_env(
        CERTIFIED_ENVIRONMENT,
        {**ADMITTED_AMBIENT, adapter.CAPTURE_MODE_ENV: "stale"},
        transport=transport,
    )
    assert observed["PYTORCH_ALLOC_CONF"] == adapter.PYTORCH_ALLOC_CONF
    assert {name: observed[name] for name in transport} == transport


def test_prepare_reuses_verified_stage_without_compiling_again(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, compiler, first = _prepare(tmp_path)

    second = adapter.prepare_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=stage,
        source_workload_index=1,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=lambda *args, **kwargs: pytest.fail("verified stage must not recompile"),
    )

    assert compiler.calls == 1
    assert first["manifest"] == second["manifest"]
    assert second["status"] == "reused"


def test_prepare_supports_single_row_reduced_workload(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, compiler, outcome = _prepare(tmp_path, index=0, rows=1)
    reduced = tmp_path / "reduced_representative.jsonl"
    reduced.write_bytes(inputs["workload"].read_bytes())
    reduced_stage = tmp_path / "reduced-stage"

    reduced_outcome = adapter.prepare_capture_stage(
        definition_file=inputs["definition"],
        workload_file=reduced,
        solution_file=inputs["solution"],
        staging_dir=reduced_stage,
        source_workload_index=0,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=compiler,
    )

    assert outcome["manifest"]["selection"]["source_workload_count"] == 1
    assert reduced_outcome["manifest"]["source_inputs"]["workload"]["path"] == str(
        reduced.resolve()
    )
    assert reduced_outcome["manifest"]["selection"]["workload_uuid"] == "workload-0"


@pytest.mark.parametrize(
    "workload_relative",
    [
        "tools/sol_execbench/data/benchmark/L2/"
        "028_gqa_rotary_attention_core_backward/workload.jsonl",
        "data/local_labels/campaign/sol_reduced_workloads/L2/"
        "problem_028_gqa_rotary_attention_core_backward__h3.jsonl",
    ],
    ids=["native", "reduced_representative"],
)
def test_problem_028_production_native_and_reduced_stage_without_gpu(
    tmp_path, monkeypatch, workload_relative
):
    repo = Path(__file__).resolve().parents[3]
    definition = (
        repo
        / "tools/sol_execbench/data/benchmark/L2/"
        "028_gqa_rotary_attention_core_backward/definition.json"
    )
    solution = (
        repo
        / "workloads/sol_execbench_baselines/L2/"
        "problem_028_gqa_rotary_attention_core_backward/cuda_cpp/solution.json"
    )
    workload = repo / workload_relative
    executable = repo / "tools/sol_execbench/.venv/bin/sol-execbench"
    compiler = _Compiler()
    monkeypatch.setattr(
        "sol_execbench.driver.problem_packager._get_local_sm", lambda: "sm_86"
    )

    outcome = adapter.prepare_capture_stage(
        definition_file=definition,
        workload_file=workload,
        solution_file=solution,
        staging_dir=tmp_path / "stage",
        source_workload_index=0,
        adapter_executable=executable,
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=compiler,
    )

    assert compiler.calls == 1
    assert outcome["manifest"]["solution_contract"]["definition"] == (
        "028_gqa_rotary_attention_core_backward"
    )
    assert outcome["manifest"]["selection"]["source_workload_count"] >= 1
    assert outcome["manifest"]["selection"]["staged_workload_index"] == 0


def test_problem_070_real_packager_emits_exact_v2_attestation_without_gpu(
    tmp_path, monkeypatch
):
    repo = Path(__file__).resolve().parents[3]
    benchmark = (
        repo
        / "tools/sol_execbench/data/benchmark/L1/"
        "070_mamba2_fused_intra_chunk_diagonal_computation"
    )
    definition = benchmark / "definition.json"
    workload = benchmark / "workload.jsonl"
    solution = (
        repo
        / "workloads/sol_execbench_baselines/L1/"
        "problem_070_mamba2_fused_intra_chunk_diagonal_computation/"
        "cuda_cpp/solution.json"
    )
    executable = repo / "tools/sol_execbench/.venv/bin/sol-execbench"
    source_document = json.loads(definition.read_text())
    assert "op_type" not in source_document
    assert adapter.sha256_file(definition) == (
        "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
    )
    compiler = _Compiler()
    monkeypatch.setattr(
        "sol_execbench.driver.problem_packager._get_local_sm", lambda: "sm_86"
    )

    outcome = adapter.prepare_capture_stage(
        definition_file=definition,
        workload_file=workload,
        solution_file=solution,
        staging_dir=tmp_path / "problem-070-stage",
        source_workload_index=0,
        adapter_executable=executable,
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=compiler,
    )

    manifest = outcome["manifest"]
    stage = tmp_path / "problem-070-stage"
    assert compiler.calls == 1
    assert manifest["source_definition_sha256"] == (
        "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa"
    )
    assert manifest["staged_definition_sha256"] == (
        "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387"
    )
    attestation = manifest["definition_equivalence"]
    assert attestation["rule_id"] == "SOL_DEF_EQUIV_V2"
    assert attestation["top_level_field_states"]["op_type"] == {
        "source": "missing",
        "staged": "null",
        "source_normalized": True,
        "staged_normalized": False,
    }
    assert not any(
        state["source_normalized"] or state["staged_normalized"]
        for state in attestation["axis_description_states"].values()
    )
    assert (stage / "definition.source.json").read_bytes() == definition.read_bytes()
    assert (stage / "definition_equivalence_attestation.json").read_bytes() == (
        adapter.serialize_attestation(attestation)
    )
    assert adapter.validate_capture_stage(
        stage,
        definition_file=definition,
        workload_file=workload,
        solution_file=solution,
        adapter_executable=executable,
        source_workload_index=0,
        environment_contract=CERTIFIED_ENVIRONMENT,
    ) == manifest


def test_definition_attestation_bytes_are_deterministic_across_preparations(tmp_path):
    observed = []
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "inputs").mkdir(parents=True)
        _, stage, _, _ = _prepare(root)
        observed.append((stage / "definition_equivalence_attestation.json").read_bytes())
    assert observed[0] == observed[1]


@pytest.mark.parametrize(
    ("tamper", "code"),
    [
        ("source_copy_missing", "sol_stage_source_copy_missing"),
        ("source_copy_bytes", "sol_stage_source_copy_mismatch"),
        ("wrong_rule", "SOL_ATTESTATION_RULE_MISMATCH"),
        ("wrong_state", "SOL_ATTESTATION_STATE_MISMATCH"),
        ("ineligible_pointer", "SOL_ATTESTATION_POINTER_INELIGIBLE"),
        ("ineligible_tensor_pointer", "SOL_ATTESTATION_POINTER_INELIGIBLE"),
        ("nondeterministic_attestation", "SOL_ATTESTATION_STATE_MISMATCH"),
    ],
)
def test_stage_revalidation_refuses_source_or_attestation_tamper(
    tmp_path, tamper, code
):
    (tmp_path / "inputs").mkdir()
    _, stage, _, _ = _prepare(tmp_path)
    manifest_path = stage / adapter.PREPARE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if tamper == "source_copy_missing":
        (stage / "definition.source.json").unlink()
    elif tamper == "source_copy_bytes":
        (stage / "definition.source.json").write_bytes(b"{}")
    elif tamper == "wrong_rule":
        manifest["definition_equivalence"]["rule_id"] = "SOL_DEF_EQUIV_V1"
    elif tamper == "wrong_state":
        manifest["definition_equivalence"]["top_level_field_states"]["op_type"][
            "source_normalized"
        ] = False
    elif tamper == "ineligible_pointer":
        manifest["definition_equivalence"]["axis_description_states"][
            "outside_both_definitions"
        ] = {
            "source": "null",
            "staged": "null",
            "source_normalized": False,
            "staged_normalized": False,
        }
        attestation_path = stage / "definition_equivalence_attestation.json"
        attestation_path.write_bytes(
            adapter.serialize_attestation(manifest["definition_equivalence"])
        )
        manifest["artifacts"]["definition_equivalence_attestation"]["sha256"] = (
            adapter.sha256_file(attestation_path)
        )
    elif tamper == "ineligible_tensor_pointer":
        manifest["definition_equivalence"]["tensor_description_states"][
            "/inputs/outside_both_definitions/description"
        ] = {
            "source": "null",
            "staged": "null",
            "source_normalized": False,
            "staged_normalized": False,
        }
        attestation_path = stage / "definition_equivalence_attestation.json"
        attestation_path.write_bytes(
            adapter.serialize_attestation(manifest["definition_equivalence"])
        )
        manifest["artifacts"]["definition_equivalence_attestation"]["sha256"] = (
            adapter.sha256_file(attestation_path)
        )
    else:
        attestation_path = stage / "definition_equivalence_attestation.json"
        attestation_path.write_text(
            json.dumps(manifest["definition_equivalence"], ensure_ascii=False) + "\n"
        )
        manifest["artifacts"]["definition_equivalence_attestation"]["sha256"] = (
            adapter.sha256_file(attestation_path)
        )
    if tamper in {
        "wrong_rule", "wrong_state", "ineligible_pointer",
        "ineligible_tensor_pointer",
        "nondeterministic_attestation",
    }:
        unsigned = dict(manifest)
        unsigned.pop("stage_definition_sha256")
        manifest["stage_definition_sha256"] = adapter._stage_definition_sha256(
            unsigned
        )
        manifest_path.write_bytes(adapter._canonical_json_bytes(manifest))

    with pytest.raises(adapter.CaptureAdapterError) as caught:
        adapter.validate_capture_stage(stage)
    assert caught.value.code == code


@pytest.mark.parametrize("index", [-1, 2])
def test_invalid_pinned_index_fails_before_compile(tmp_path, index):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir)
    compiler = _Compiler()

    with pytest.raises(adapter.CaptureAdapterError, match="workload_index"):
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=index,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ=ADMITTED_AMBIENT,
            run=compiler,
        )

    assert compiler.calls == 0


@pytest.mark.parametrize(
    ("payload_factory", "code"),
    [
        (
            lambda original: original.replace(
                b'{"name": "capture_test"',
                b'{"name": "capture_test", "name": "capture_test"',
                1,
            ),
            "sol_definition_duplicate_key",
        ),
        (
            lambda original: original.replace(b"{", b'{"rogue":NaN,', 1),
            "sol_definition_nonfinite",
        ),
        (lambda _original: b"[]", "sol_definition_parse_error"),
    ],
)
def test_strict_definition_parse_refuses_before_compile(
    tmp_path, payload_factory, code
):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir)
    inputs["definition"].write_bytes(
        payload_factory(inputs["definition"].read_bytes())
    )
    compiler = _Compiler()

    with pytest.raises(adapter.CaptureAdapterError) as caught:
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=0,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ=ADMITTED_AMBIENT,
            run=compiler,
        )
    assert caught.value.code == code
    assert compiler.calls == 0
    assert not (tmp_path / "stage").exists()


def test_zero_workloads_never_vacuously_prepares(tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir, rows=0)
    compiler = _Compiler()

    with pytest.raises(adapter.CaptureAdapterError, match="workload_set_empty"):
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=0,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ=ADMITTED_AMBIENT,
            run=compiler,
        )
    assert compiler.calls == 0


def test_non_dps_solution_fails_before_compile(tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir, dps=False)
    compiler = _Compiler()

    with pytest.raises(adapter.CaptureAdapterError, match="capture_solution_not_dps"):
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=0,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ=ADMITTED_AMBIENT,
            run=compiler,
        )
    assert compiler.calls == 0


@pytest.mark.parametrize(
    "ambient",
    [
        {"PATH": "/usr/bin", "NVCC_CCBIN": "/usr/bin/g++-15"},
        {"PATH": "/usr/bin", "NVCC_CCBIN": "/usr/bin/g++-14"},
        {"PATH": "/usr/bin", "NVCC_PREPEND_FLAGS": "-Xcompiler"},
        {"PATH": "/usr/bin", "NVCC_APPEND_FLAGS": ""},
    ],
)
def test_prepare_rejects_unadmitted_ambient_before_compiler_runs(tmp_path, ambient):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir)
    compiler = _Compiler()

    with pytest.raises(adapter.CaptureAdapterError, match="SOL_ENV_"):
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=0,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ=ambient,
            run=compiler,
        )

    assert compiler.calls == 0
    assert not (tmp_path / "stage").exists()


def test_prepare_requires_exact_certified_environment_contract(tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir)
    compiler = _Compiler()

    with pytest.raises(adapter.CaptureAdapterError) as caught:
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=0,
            adapter_executable=inputs["executable"],
            environment_contract={"NVCC_CCBIN": "g++-15"},
            environ=ADMITTED_AMBIENT,
            run=compiler,
        )

    assert caught.value.code == "SOL_ENV_CONTRACT_INVALID"
    assert compiler.calls == 0


def test_adapter_explicitly_dispatches_exact_legacy_and_live_contracts():
    legacy, legacy_digest = adapter._environment_contract(
        LEGACY_CERTIFIED_ENVIRONMENT
    )
    assert legacy == LEGACY_CERTIFIED_ENVIRONMENT
    assert legacy_digest == LEGACY_CERTIFIED_ENVIRONMENT_SHA256
    assert legacy_digest == (
        "d959a1a89d2f3aa3e7c351ba59788c016404bbc77e9da428123188ada775af74"
    )
    legacy_environment = adapter._certified_subprocess_env(
        legacy, ADMITTED_AMBIENT
    )
    assert legacy_environment["NVCC_CCBIN"] == "g++-14"
    assert not (
        set(CERTIFIED_ENVIRONMENT) - {"NVCC_CCBIN"}
    ).intersection(legacy_environment)

    live, live_digest = adapter._environment_contract(CERTIFIED_ENVIRONMENT)
    assert live == CERTIFIED_ENVIRONMENT
    assert live_digest == CERTIFIED_ENVIRONMENT_SHA256
    live_environment = adapter._certified_subprocess_env(
        live, ADMITTED_AMBIENT
    )
    assert {
        key: live_environment[key] for key in CERTIFIED_ENVIRONMENT
    } == CERTIFIED_ENVIRONMENT

    for malformed in (
        {**LEGACY_CERTIFIED_ENVIRONMENT, "CC": "gcc-14"},
        {**CERTIFIED_ENVIRONMENT, "EXTRA": "forbidden"},
    ):
        with pytest.raises(adapter.CaptureAdapterError) as caught:
            adapter._environment_contract(malformed)
        assert caught.value.code == "SOL_ENV_CONTRACT_INVALID"


def test_corpus_refusal_happens_before_stage_or_compiler_mutation(
    tmp_path, monkeypatch,
):
    inputs = _write_inputs(tmp_path / "inputs")
    compiler = _Compiler()

    def refuse(_contract, *, repo_root):
        raise adapter.PrerequisiteContractError("planted corpus hash mismatch")

    monkeypatch.setattr(adapter, "verify_flashinfer_capture_contract", refuse)
    with pytest.raises(adapter.CaptureAdapterError) as caught:
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=tmp_path / "stage",
            source_workload_index=0,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            flashinfer_corpus_contract={**NON_FIB_CORPUS, "file_count": 1},
            environ=ADMITTED_AMBIENT,
            run=compiler,
        )
    assert caught.value.code == "SOL_FIB_CORPUS_CONTRACT_INVALID"
    assert compiler.calls == 0
    assert not (tmp_path / "stage").exists()


def test_verified_fib_contract_injects_exact_root_and_seals_manifest(
    tmp_path, monkeypatch,
):
    corpus = tmp_path / "verified-corpus"
    corpus.mkdir()
    contract = {
        **NON_FIB_CORPUS,
        "category": "FIB",
        "injection_policy": "verified_absolute_root",
        "root": "data/annex/flashinfer_trace",
    }
    monkeypatch.setattr(
        adapter, "verify_flashinfer_capture_contract",
        lambda observed, *, repo_root: corpus.resolve()
        if observed == contract else pytest.fail("unexpected FIB contract"),
    )
    inputs = _write_inputs(tmp_path / "inputs")
    compiler = _Compiler()
    outcome = adapter.prepare_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=tmp_path / "stage",
        source_workload_index=0,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        flashinfer_corpus_contract=contract,
        environ={**ADMITTED_AMBIENT, "FLASHINFER_TRACE_DIR": "/ambient"},
        run=compiler,
    )
    assert compiler.environments[0]["FLASHINFER_TRACE_DIR"] == str(corpus.resolve())
    assert outcome["manifest"]["flashinfer_corpus_contract"] == contract
    assert outcome["manifest"]["flashinfer_corpus_contract_sha256"] == (
        adapter.sha256_bytes(adapter._canonical_json_bytes(contract))
    )


def test_stale_source_or_staged_module_fails_closed(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, _, _ = _prepare(tmp_path)
    inputs["solution"].write_text(inputs["solution"].read_text() + "\n")
    with pytest.raises(adapter.CaptureAdapterError, match="hash_mismatch"):
        adapter.validate_capture_stage(stage)


def test_staged_binary_drift_fails_closed(tmp_path):
    (tmp_path / "inputs").mkdir()
    _, stage, _, _ = _prepare(tmp_path)
    (stage / "benchmark_kernel.so").write_bytes(b"drifted")
    with pytest.raises(adapter.CaptureAdapterError, match="hash_mismatch"):
        adapter.validate_capture_stage(stage)


def test_manifest_workload_claim_is_rederived_on_consumer_path(tmp_path):
    (tmp_path / "inputs").mkdir()
    _, stage, _, _ = _prepare(tmp_path)
    manifest_path = stage / adapter.PREPARE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["selection"]["source_workload_index"] = 0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(adapter.CaptureAdapterError, match="stage definition digest"):
        adapter.validate_capture_stage(stage)


def test_environment_contract_digest_drift_fails_stage_reuse(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, compiler, _ = _prepare(tmp_path)
    manifest_path = stage / adapter.PREPARE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["environment_contract_sha256"] = "0" * 64
    without_stage_digest = dict(manifest)
    without_stage_digest.pop("stage_definition_sha256")
    manifest["stage_definition_sha256"] = adapter._stage_definition_sha256(
        without_stage_digest
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        adapter.CaptureAdapterError, match="environment contract digest"
    ):
        adapter.prepare_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=stage,
            source_workload_index=1,
            adapter_executable=inputs["executable"],
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ=ADMITTED_AMBIENT,
            run=lambda *args, **kwargs: pytest.fail("drift must not recompile"),
        )

    assert compiler.calls == 1


def test_stage_definition_digest_drift_fails_validation(tmp_path):
    (tmp_path / "inputs").mkdir()
    _, stage, _, _ = _prepare(tmp_path)
    manifest_path = stage / adapter.PREPARE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["stage_definition_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(adapter.CaptureAdapterError, match="stage definition digest"):
        adapter.validate_capture_stage(
            stage, environment_contract=CERTIFIED_ENVIRONMENT
        )


def test_execute_emits_selection_and_validates_success_result(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, _, _ = _prepare(tmp_path)
    selection_relative = Path("evidence/trace-selection.json")
    result_relative = Path("evidence/trace-result.json")
    selection_path = stage / selection_relative
    result_path = stage / result_relative

    outcome = adapter.execute_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=stage,
        source_workload_index=1,
        capture_kind="trace",
        selection_manifest=selection_relative,
        result_manifest=result_relative,
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=_successful_execution(
            stage, selection_relative, result_relative, source_index=1
        ),
    )

    selection = json.loads(selection_path.read_text())
    assert selection["capture_kind"] == "trace"
    assert selection["capture_contract_version"] == 2
    assert selection["source_workload_index"] == 1
    assert selection["workload"]["workload_uuid"] == "workload-1"
    assert selection["result_manifest"] == result_relative.as_posix()
    assert selection["environment_contract"] == CERTIFIED_ENVIRONMENT
    assert selection["environment_contract_sha256"] == CERTIFIED_ENVIRONMENT_SHA256
    assert selection["stage_definition_sha256"] == outcome["result"][
        "stage_definition_sha256"
    ]
    assert "stage_dir" not in selection
    assert selection["benchmark_kernel_so"] == outcome["selection"][
        "benchmark_kernel_so"
    ]
    assert outcome["status"] == "captured"
    assert outcome["result"]["capture_contract_version"] == 2

    malformed = dict(outcome["result"])
    malformed["outside_roi"] = dict(malformed["outside_roi"])
    malformed["outside_roi"]["correctness_rounds"] = 0
    result_path.write_text(json.dumps(malformed))
    with pytest.raises(adapter.CaptureAdapterError, match="outside-ROI"):
        adapter._validate_capture_result(
            result_path,
            stage=stage,
            selection_path=selection_path,
            selection_relative=selection_relative.as_posix(),
            selection_sha256=adapter.sha256_file(selection_path),
        )


def test_execute_rejects_ambient_conflict_before_evidence_or_subprocess(tmp_path):
    (tmp_path / "inputs").mkdir()
    inputs, stage, _, _ = _prepare(tmp_path)
    selection_relative = Path("evidence/conflict-selection.json")
    result_relative = Path("evidence/conflict-result.json")

    with pytest.raises(adapter.CaptureAdapterError) as caught:
        adapter.execute_capture_stage(
            definition_file=inputs["definition"],
            workload_file=inputs["workload"],
            solution_file=inputs["solution"],
            staging_dir=stage,
            source_workload_index=1,
            capture_kind="trace",
            selection_manifest=selection_relative,
            result_manifest=result_relative,
            environment_contract=CERTIFIED_ENVIRONMENT,
            environ={"PATH": "/usr/bin", "NVCC_CCBIN": "/usr/bin/g++-15"},
            run=lambda *args, **kwargs: pytest.fail("conflict must not execute"),
        )

    assert caught.value.code == "SOL_ENV_AMBIENT_CONFLICT"
    assert not (stage / selection_relative).exists()
    assert not (stage / result_relative).exists()


def test_stage_and_evidence_survive_arbitrary_parent_renames(tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir)
    old_parent = tmp_path / "campaign.staging"
    old_parent.mkdir()
    old_stage = old_parent / "cell"
    compiler = _Compiler()
    adapter.prepare_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=old_stage,
        source_workload_index=1,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=compiler,
    )

    current_parent = tmp_path / "chunks" / "chunk-000"
    current_parent.parent.mkdir()
    old_parent.rename(current_parent)
    current_stage = current_parent / "cell"
    manifest = adapter.validate_capture_stage(current_stage)
    assert manifest["stage_root"] == "."
    assert "stage_dir" not in manifest

    reused = adapter.prepare_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=current_stage,
        source_workload_index=1,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=lambda *args, **kwargs: pytest.fail("renamed stage must be reused"),
    )
    assert reused["status"] == "reused"
    assert compiler.calls == 1

    selection_relative = Path("evidence/label-selection.json")
    result_relative = Path("evidence/label-result.json")
    adapter.execute_capture_stage(
        definition_file=inputs["definition"],
        workload_file=inputs["workload"],
        solution_file=inputs["solution"],
        staging_dir=current_stage,
        source_workload_index=1,
        capture_kind="label",
        selection_manifest=selection_relative,
        result_manifest=result_relative,
        adapter_executable=inputs["executable"],
        environment_contract=CERTIFIED_ENVIRONMENT,
        environ=ADMITTED_AMBIENT,
        run=_successful_execution(
            current_stage, selection_relative, result_relative, source_index=1
        ),
    )

    sealed_parent = tmp_path / "sealed" / "chunk-000"
    sealed_parent.parent.mkdir()
    current_parent.rename(sealed_parent)
    sealed_stage = sealed_parent / "cell"
    evidence = adapter.validate_capture_evidence(
        sealed_stage,
        selection_manifest=selection_relative,
        result_manifest=result_relative,
    )
    assert evidence["result"]["status"] == "captured"
    serialized = json.dumps(evidence, sort_keys=True)
    assert str(old_parent) not in serialized
    assert str(current_parent) not in serialized


def test_request_contract_uses_absolute_stage_and_relative_evidence(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action": "execute",
                "stage_dir": str((tmp_path / "stage").resolve()),
                "source_workload_index": 0,
                "environment_contract": CERTIFIED_ENVIRONMENT,
                "environment_contract_sha256": CERTIFIED_ENVIRONMENT_SHA256,
                "flashinfer_corpus_contract": NON_FIB_CORPUS,
                "flashinfer_corpus_contract_sha256": NON_FIB_CORPUS_SHA256,
                "capture_kind": "label",
                "selection_manifest": "evidence/selection.json",
                "result_manifest": "evidence/result.json",
                "execution_timeout_s": 37,
            }
        )
    )
    loaded = adapter.load_capture_request(request)
    assert loaded["capture_kind"] == "label"
    assert loaded["environment_contract"] == CERTIFIED_ENVIRONMENT
    assert loaded["environment_contract_sha256"] == CERTIFIED_ENVIRONMENT_SHA256

    invalid = json.loads(request.read_text())
    invalid["environment_contract_sha256"] = "0" * 64
    request.write_text(json.dumps(invalid))
    with pytest.raises(adapter.CaptureAdapterError, match="does not rederive"):
        adapter.load_capture_request(request)

    invalid = json.loads(request.read_text())
    invalid_contract = {"NVCC_CCBIN": "g++-15"}
    invalid["environment_contract"] = invalid_contract
    invalid["environment_contract_sha256"] = hashlib.sha256(
        (json.dumps(invalid_contract, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    request.write_text(json.dumps(invalid))
    with pytest.raises(adapter.CaptureAdapterError) as caught:
        adapter.load_capture_request(request)
    assert caught.value.code == "SOL_ENV_CONTRACT_INVALID"

    invalid = json.loads(request.read_text())
    invalid["environment_contract"] = CERTIFIED_ENVIRONMENT
    invalid["environment_contract_sha256"] = CERTIFIED_ENVIRONMENT_SHA256
    invalid["selection_manifest"] = str((tmp_path / "selection.json").resolve())
    request.write_text(json.dumps(invalid))
    with pytest.raises(adapter.CaptureAdapterError, match="stage-relative"):
        adapter.load_capture_request(request)

    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "action": "prepare",
                "stage_dir": "relative",
                "source_workload_index": 0,
                "environment_contract": CERTIFIED_ENVIRONMENT,
                "environment_contract_sha256": CERTIFIED_ENVIRONMENT_SHA256,
                "flashinfer_corpus_contract": NON_FIB_CORPUS,
                "flashinfer_corpus_contract_sha256": NON_FIB_CORPUS_SHA256,
                "compile_timeout_s": 37,
            }
        )
    )
    with pytest.raises(adapter.CaptureAdapterError, match="absolute"):
        adapter.load_capture_request(request)


@pytest.mark.parametrize(
    ("action", "timeout_field", "timeout_value", "target_name"),
    [
        ("prepare", "compile_timeout_s", 37, "prepare_capture_stage"),
        ("execute", "execution_timeout_s", 43, "execute_capture_stage"),
    ],
)
def test_request_dispatch_binds_contract_and_request_timeout(
    tmp_path, monkeypatch, action, timeout_field, timeout_value, target_name
):
    request = {
        "schema_version": 1,
        "action": action,
        "stage_dir": str((tmp_path / "stage").resolve()),
        "source_workload_index": 0,
        "environment_contract": CERTIFIED_ENVIRONMENT,
        "environment_contract_sha256": CERTIFIED_ENVIRONMENT_SHA256,
        "flashinfer_corpus_contract": NON_FIB_CORPUS,
        "flashinfer_corpus_contract_sha256": NON_FIB_CORPUS_SHA256,
        timeout_field: timeout_value,
    }
    if action == "execute":
        request.update(
            {
                "capture_kind": "trace",
                "selection_manifest": "evidence/selection.json",
                "result_manifest": "evidence/result.json",
            }
        )
    request_path = tmp_path / f"{action}.json"
    request_path.write_text(json.dumps(request))
    seen = {}

    def fake_target(**kwargs):
        seen.update(kwargs)
        return {"status": "prepared" if action == "prepare" else "captured"}

    monkeypatch.setattr(adapter, target_name, fake_target)
    outcome = adapter.handle_capture_request(
        request_file=request_path,
        definition_file=tmp_path / "definition.json",
        workload_file=tmp_path / "workload.jsonl",
        solution_file=tmp_path / "solution.json",
        config_file=None,
        compile_timeout=999,
        timeout=999,
        adapter_executable=tmp_path / "sol-execbench",
        environ=ADMITTED_AMBIENT,
    )

    assert outcome["action"] == action
    assert seen["environment_contract"] == CERTIFIED_ENVIRONMENT
    assert seen["environ"] == ADMITTED_AMBIENT
    if action == "prepare":
        assert seen["compile_timeout"] == timeout_value
    else:
        assert seen["timeout"] == timeout_value


def test_existing_cli_uses_request_env_without_changing_application_argv(
    tmp_path, monkeypatch
):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    inputs = _write_inputs(inputs_dir)
    request = tmp_path / "prepare-request.json"
    request.write_text("{}")
    seen = {}
    cli_main = importlib.import_module("sol_execbench.cli.main")

    def fake_handle(**kwargs):
        seen.update(kwargs)
        return {
            "action": "prepare",
            "status": "prepared",
            "manifest": {"manifest_kind": "sol_execbench_capture_preparation"},
        }

    monkeypatch.setattr(cli_main, "handle_capture_request", fake_handle)
    result = CliRunner().invoke(
        cli_main.cli,
        [
            "--definition",
            str(inputs["definition"]),
            "--workload",
            str(inputs["workload"]),
            "--solution",
            str(inputs["solution"]),
            "--json",
        ],
        env={adapter.CAPTURE_REQUEST_ENV: str(request)},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["manifest_kind"] == (
        "sol_execbench_capture_preparation"
    )
    assert seen["request_file"] == request
    assert seen["definition_file"] == inputs["definition"]
    assert seen["workload_file"] == inputs["workload"]
    assert seen["solution_file"] == inputs["solution"]


def test_capture_integrity_accepts_unchanged_none_timing_binding():
    critical_names = _eval_template_critical_names(capture_contract={})
    assert "time_runnable" not in critical_names

    namespace = {name: object() for name in critical_names}
    namespace["time_runnable"] = None
    snapshot = snapshot_critical_functions(namespace, critical_names)

    check_eval_integrity(snapshot, namespace)


def test_capture_integrity_still_rejects_shared_guard_replacement():
    critical_names = _eval_template_critical_names(capture_contract={})
    assert "check_lazy_outputs" in critical_names
    namespace = {name: object() for name in critical_names}
    snapshot = snapshot_critical_functions(namespace, critical_names)
    namespace["check_lazy_outputs"] = object()

    with pytest.raises(
        RewardHackDetected,
        match="'check_lazy_outputs' has been monkey-patched",
    ):
        check_eval_integrity(snapshot, namespace)


def test_canonical_integrity_still_rejects_timing_replacement():
    critical_names = _eval_template_critical_names(capture_contract=None)
    assert critical_names[0] == "time_runnable"
    namespace = {name: object() for name in critical_names}
    snapshot = snapshot_critical_functions(namespace, critical_names)
    namespace["time_runnable"] = object()

    with pytest.raises(
        RewardHackDetected,
        match="'time_runnable' has been monkey-patched",
    ):
        check_eval_integrity(snapshot, namespace)


def test_eval_template_validates_selection_before_any_cuda_query():
    template = (
        Path(adapter.__file__).parent / "templates" / "eval_driver.py"
    ).read_text()
    assert template.index("_CAPTURE_CONTRACT = _load_capture_contract()") < template.index(
        '_device = "cuda:0" if torch.cuda.is_available() else "cpu"'
    )
    contract_body = template.split("def _load_capture_contract():", 1)[1].split(
        "_CAPTURE_CONTRACT = _load_capture_contract()", 1
    )[0]
    assert 'prepare.get("capture_contract_version") != 2' in contract_body
    assert 'os.environ.get(name) != value' in contract_body
    assert 'shutil.which("nvcc")' in contract_body
    assert "capture stage definition digest mismatch" in contract_body
    capture_body = template.split("def _run_capture_mode():", 1)[1].split(
        "if _CAPTURE_CONTRACT is not None:", 1
    )[0]
    assert "time_runnable(" not in capture_body
    assert '"timing_methodology": "cuda_events"' in capture_body
    assert "profile_candidate_once(" in capture_body


def test_eval_template_releases_previous_round_before_generating_inputs():
    _, _, correctness_loop = _eval_template_correctness_loops()

    cleanup = correctness_loop.body[0]
    assert isinstance(cleanup, ast.If)
    assert ast.unparse(cleanup.test) == "_round > 0"
    assert [
        node.targets[0].id
        for node in cleanup.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ] == ["_inputs", "_ref_outputs", "_user_outputs"]
    cleanup_source = ast.unparse(cleanup)
    assert "gc.collect" not in cleanup_source
    assert "empty_cache" not in cleanup_source

    generation = correctness_loop.body[1]
    assert isinstance(generation, ast.Try)
    assignment = generation.body[0]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "_inputs"
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "gen_inputs"


def test_eval_template_round_cleanup_lifetime_and_final_input_behavior():
    template_path, _, correctness_loop = _eval_template_correctness_loops()
    exercised_loop = copy.deepcopy(correctness_loop)
    exercised_loop.body = exercised_loop.body[:2] + ast.parse(
        """
_ref_outputs = _LifetimeTracker(f"ref{_round}")
_user_outputs = _LifetimeTracker(f"user{_round}")
"""
    ).body
    module = ast.fix_missing_locations(
        ast.Module(body=[exercised_loop], type_ignores=[])
    )

    released = []
    generation_snapshots = []

    class _LifetimeTracker:
        def __init__(self, name):
            self.name = name

        def __del__(self):
            released.append(self.name)

    def fake_gen_inputs(*args, **kwargs):
        generation_snapshots.append(tuple(released))
        return _LifetimeTracker(f"inputs{len(generation_snapshots) - 1}")

    namespace = {
        "_inputs": _LifetimeTracker("initial_inputs"),
        "_ref_outputs": _LifetimeTracker("initial_ref"),
        "_user_outputs": _LifetimeTracker("initial_user"),
        "_LifetimeTracker": _LifetimeTracker,
        "gen_inputs": fake_gen_inputs,
        "definition": object(),
        "_workload": object(),
        "_device": "cpu",
        "_safe_tensors": {},
        "_custom_inputs_fn": None,
    }
    exec(compile(module, str(template_path), "exec"), namespace)

    # Round 0 performs no pre-generation cleanup: every existing sentinel is
    # still live while its gen_inputs() right-hand side is evaluated.
    assert generation_snapshots[0] == ()
    # Before round 1 generation, both the initial sentinels and every round-0
    # container have been released by the real template cleanup block.
    assert set(generation_snapshots[1]) == {
        "initial_inputs",
        "initial_ref",
        "initial_user",
        "inputs0",
        "ref0",
        "user0",
    }
    # There is no cleanup after the final correctness round. Its input is the
    # one deliberately retained for the subsequent timing phase.
    assert namespace["_inputs"].name == "inputs9"


def test_eval_template_passes_surviving_final_round_inputs_to_timing():
    _, workload_loop, correctness_loop = _eval_template_correctness_loops()
    loop_index = workload_loop.body.index(correctness_loop)
    timing_index, timing_call = next(
        (index, call)
        for index, statement in enumerate(
            workload_loop.body[loop_index + 1 :], start=loop_index + 1
        )
        for call in ast.walk(statement)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "time_runnable"
    )
    assert isinstance(timing_call.args[1], ast.Name)
    assert timing_call.args[1].id == "_inputs"

    before_timing = workload_loop.body[loop_index + 1 : timing_index + 1]
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_inputs"
        and isinstance(node.ctx, (ast.Store, ast.Del))
        for statement in before_timing
        for node in ast.walk(statement)
    )
