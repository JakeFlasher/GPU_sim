from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import sol_trace_scale_proxy as proxy


def _json_bytes(value, *, ordered=False):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=not ordered,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


@pytest.fixture
def source(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    solution_dir = tmp_path / "baseline" / "cuda_cpp"
    benchmark.mkdir()
    solution_dir.mkdir(parents=True)
    definition = {
        "name": "001_fixture",
        "axes": {
            "batch_size": {"type": "var", "description": "batch"},
            "seq_len": {"type": "var", "description": "sequence"},
            "hidden_size": {"type": "const", "value": 128},
            "tokens": {"type": "expr", "expression": "batch_size * seq_len"},
        },
        "inputs": {
            "x": {"shape": ["batch_size", "seq_len", "hidden_size"], "dtype": "float32"}
        },
        "outputs": {
            "y": {"shape": ["batch_size", "seq_len", "hidden_size"], "dtype": "float32"}
        },
        "reference": "def run(x):\n    return x\n",
    }
    definition_bytes = _json_bytes(definition, ordered=True)
    (benchmark / "definition.json").write_bytes(definition_bytes)
    rows = [
        {
            "uuid": "row-large",
            "axes": {"batch_size": 8, "seq_len": 1024},
            "inputs": {"x": {"type": "random"}},
            "tolerance": {"max_atol": 0.01, "max_rtol": 0.01},
        },
        {
            "uuid": "row-smallest",
            "axes": {"batch_size": 1, "seq_len": 256},
            "inputs": {"x": {"type": "random"}},
            "tolerance": {"max_atol": 0.01, "max_rtol": 0.01},
        },
    ]
    row_bytes = [_json_bytes(row)[:-1] for row in rows]
    workload_bytes = b"\n".join(row_bytes) + b"\n"
    (benchmark / "workload.jsonl").write_bytes(workload_bytes)
    solution = solution_dir / "solution.json"
    solution.write_bytes(_json_bytes({"definition": "001_fixture", "solution": "fixture"}))
    target = {
        "axis_updates": {"seq_len": 32},
        "category": "FIB",
        "evidence_sha256": "e" * 64,
        "language": "cuda_cpp",
        "native_definition_sha256": hashlib.sha256(definition_bytes).hexdigest(),
        "native_row_sha256": hashlib.sha256(row_bytes[1]).hexdigest(),
        "partial_trace_allowed": False,
        "reviewed_definition_policy": None,
        "sampling_policy": "off",
        "slug": "problem_001_fixture",
        "smallest_native_workload": True,
        "source_workload_index": 1,
        "source_workload_uuid": "row-smallest",
        "trace_failure_reason": "trace_target_failed_timeout",
    }
    cell = {
        "category": "FIB",
        "slug": "problem_001_fixture",
        "language": "cuda_cpp",
        "prior_done": True,
        "prior_deferred_correctness": False,
    }
    return {
        "benchmark": benchmark,
        "solution": solution,
        "definition": definition,
        "definition_bytes": definition_bytes,
        "workload_bytes": workload_bytes,
        "row": rows[1],
        "row_bytes": row_bytes[1],
        "target": target,
        "cell": cell,
    }


def _candidate(source):
    return proxy.plan_candidate(
        cell=source["cell"],
        benchmark_dir=source["benchmark"],
        solution_path=source["solution"],
        target=source["target"],
    )


def _passing_certification(candidate):
    evidence = {
        "diagnostics_complete": True,
        "status": "runnable",
        "workload_diagnostics": [
            {"diagnostic_present": True, "status": "PASSED"}
        ],
    }
    return {
        "method": "sol_execbench_full_correctness",
        "status": "runnable",
        "correctness_result": "passed",
        "workload_sha256": candidate["reduced_workload"]["sha256"],
        "definition_sha256": candidate["reduced_definition"]["sha256"],
        "solution_sha256": candidate["solution"]["sha256"],
        "evidence_sha256": proxy.sha256_bytes(proxy.canonical_json_bytes(evidence)),
        "execution_evidence": evidence,
    }


def test_plan_is_read_only_separate_and_preserves_every_non_axis_contract(source):
    before = {
        path: path.read_bytes()
        for path in (
            source["benchmark"] / "definition.json",
            source["benchmark"] / "workload.jsonl",
            source["solution"],
        )
    }
    candidate = _candidate(source)
    assert {path: path.read_bytes() for path in before} == before
    assert candidate["claim_scope"] == "trace_scale_proxy_only"
    assert candidate["native_predictive"] is False
    assert candidate["native_definition"]["sha256"] == hashlib.sha256(
        source["definition_bytes"]
    ).hexdigest()
    assert candidate["native_workload_file"]["sha256"] == hashlib.sha256(
        source["workload_bytes"]
    ).hexdigest()
    assert candidate["native_workload_row"] == {
        "sha256": hashlib.sha256(source["row_bytes"]).hexdigest(),
        "size": len(source["row_bytes"]),
        "sha256_scope": "exact_json_line_bytes_excluding_lf_only",
    }
    assert candidate["reduced_definition"]["name"].endswith(".definition.json")
    assert candidate["_definition_bytes"] == source["definition_bytes"]
    reduced = json.loads(candidate["_workload_bytes"])
    assert reduced["axes"] == {"batch_size": 1, "seq_len": 32}
    assert {k: v for k, v in reduced.items() if k != "axes"} == {
        k: v for k, v in source["row"].items() if k != "axes"
    }
    assert candidate["axes_delta"] == [
        {"axis": "seq_len", "from": 256, "to": 32}
    ]
    assert candidate["definition_delta"] == []


def test_plan_targets_is_exact_complete_and_roster_bound(source):
    resolver = lambda _cell: (
        source["solution"].parent,
        source["solution"],
        source["benchmark"],
    )
    candidates = proxy.plan_targets(
        [source["target"]], cells=[source["cell"]], resolver=resolver,
    )
    assert [row["slug"] for row in candidates] == [source["cell"]["slug"]]
    with pytest.raises(proxy.TraceScaleProxyError, match="duplicate problem"):
        proxy.plan_targets(
            [source["target"], source["target"]],
            cells=[source["cell"]], resolver=resolver,
        )
    outside = copy.deepcopy(source["target"])
    outside["slug"] = "problem_999_outside"
    with pytest.raises(proxy.TraceScaleProxyError, match="outside the sm86"):
        proxy.plan_targets([outside], cells=[source["cell"]], resolver=resolver)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trace_failure_reason", "trace_target_failed", "typed unsampled-capacity"),
        ("sampling_policy", "adaptive", "sampling disabled"),
        ("partial_trace_allowed", True, "allowed a partial trace"),
        ("smallest_native_workload", False, "not the measured smallest"),
    ],
)
def test_target_never_infers_trace_infeasibility(source, field, value, message):
    target = copy.deepcopy(source["target"])
    target[field] = value
    with pytest.raises(proxy.TraceScaleProxyError, match=message):
        proxy.plan_candidate(
            cell=source["cell"], benchmark_dir=source["benchmark"],
            solution_path=source["solution"], target=target,
        )


@pytest.mark.parametrize("axis", ["hidden_size", "tokens", "invented"])
def test_only_definition_declared_variable_axes_may_change(source, axis):
    target = copy.deepcopy(source["target"])
    target["axis_updates"] = {axis: 1}
    with pytest.raises(proxy.TraceScaleProxyError, match="definition-declared var"):
        proxy.plan_candidate(
            cell=source["cell"], benchmark_dir=source["benchmark"],
            solution_path=source["solution"], target=target,
        )


@pytest.mark.parametrize("value", [0, -1, True, 256, 512])
def test_axis_update_must_be_a_strict_positive_reduction(source, value):
    target = copy.deepcopy(source["target"])
    target["axis_updates"] = {"seq_len": value}
    with pytest.raises(proxy.TraceScaleProxyError, match="positive integer|does not reduce"):
        proxy.plan_candidate(
            cell=source["cell"], benchmark_dir=source["benchmark"],
            solution_path=source["solution"], target=target,
        )


def test_native_row_and_definition_hashes_are_hard_bindings(source):
    for field in ("native_row_sha256", "native_definition_sha256"):
        target = copy.deepcopy(source["target"])
        target[field] = "0" * 64
        with pytest.raises(proxy.TraceScaleProxyError, match="differ"):
            proxy.plan_candidate(
                cell=source["cell"], benchmark_dir=source["benchmark"],
                solution_path=source["solution"], target=target,
            )


def test_source_order_cannot_override_definition_footprint_rank_zero(source):
    target = copy.deepcopy(source["target"])
    first = json.loads(source["workload_bytes"].splitlines()[0])
    first_bytes = source["workload_bytes"].splitlines()[0]
    target.update(
        source_workload_index=0,
        source_workload_uuid=first["uuid"],
        native_row_sha256=hashlib.sha256(first_bytes).hexdigest(),
        axis_updates={"seq_len": 32},
    )
    with pytest.raises(proxy.TraceScaleProxyError, match="definition-aware smallest"):
        proxy.plan_candidate(
            cell=source["cell"], benchmark_dir=source["benchmark"],
            solution_path=source["solution"], target=target,
        )


def test_footprint_resolves_expr_dtypes_and_declared_scalar_extras():
    definition = {
        "axes": {
            "n": {"type": "var"},
            "c": {"type": "const", "value": 3},
            "twice": {"type": "expr", "expression": "n * 2"},
        },
        "inputs": {
            "x": {"shape": ["n", "c"], "dtype": "float16"},
            "flag": {"shape": None, "dtype": "bool"},
        },
        "outputs": {"y": {"shape": ["twice"], "dtype": "int8"}},
    }
    footprint = proxy.execution_footprint(
        definition, {"n": 2, "flag": 1},
        source_index=4, source_uuid="row", variant="native",
    )
    assert footprint == {
        "aggregate_tensor_bits": 136,
        "peak_tensor_bits": 96,
        "aggregate_elements": 11,
        "resolved_axes_product": 24,
        "extra_scalar_axes": [["flag", 1]],
        "source_workload_index": 4,
        "source_workload_uuid": "row",
        "variant": "native",
        "key_sha256": footprint["key_sha256"],
    }
    assert len(footprint["key_sha256"]) == 64


def test_footprint_rejects_undeclared_extra_and_unsafe_expression():
    definition = {
        "axes": {
            "n": {"type": "var"},
            "bad": {"type": "expr", "expression": "__import__('os')"},
        },
        "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
    }
    with pytest.raises(proxy.TraceScaleProxyError, match="extra workload axis"):
        proxy.execution_footprint(
            definition, {"n": 2, "invented": 1},
            source_index=0, source_uuid="row",
        )
    with pytest.raises(proxy.TraceScaleProxyError, match="unsupported shape expression"):
        proxy.execution_footprint(
            definition, {"n": 2}, source_index=0, source_uuid="row",
        )


def test_unregistered_definition_constant_edit_is_refused(source):
    target = copy.deepcopy(source["target"])
    target["reviewed_definition_policy"] = "invented_policy"
    with pytest.raises(proxy.TraceScaleProxyError, match="no registered slug"):
        proxy.plan_candidate(
            cell=source["cell"], benchmark_dir=source["benchmark"],
            solution_path=source["solution"], target=target,
        )


def test_authority_requires_exact_runnable_correctness_evidence(source):
    candidate = _candidate(source)
    artifact, children = proxy.build_authority(
        source_commit="a" * 40,
        target_manifest_sha256="b" * 64,
        candidates=[candidate],
        validator=_passing_certification,
    )
    assert artifact["proxy_count"] == 1
    assert artifact["claim_scope"] == "trace_scale_proxy_only"
    assert artifact["native_predictive"] is False
    assert artifact["proxies"][0]["validation"]["correctness_result"] == "passed"
    assert set(children) == {
        candidate["reduced_definition"]["name"],
        candidate["reduced_workload"]["name"],
    }
    schema = json.loads(proxy.SCHEMA_PATH.read_text())
    jsonschema.validate(artifact, schema)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "host_capacity_oom", "not runnable"),
        ("correctness_result", "failed", "did not pass"),
        ("workload_sha256", "0" * 64, "different workload"),
        ("definition_sha256", "0" * 64, "different definition"),
        ("solution_sha256", "0" * 64, "different solution"),
        ("evidence_sha256", "0" * 64, "evidence hash mismatch"),
    ],
)
def test_failed_or_misbound_certification_never_publishes(
    source, field, value, message,
):
    candidate = _candidate(source)

    def invalid(row):
        result = _passing_certification(row)
        result[field] = value
        return result

    with pytest.raises(proxy.TraceScaleProxyError, match=message):
        proxy.build_authority(
            source_commit="a" * 40,
            target_manifest_sha256="b" * 64,
            candidates=[candidate],
            validator=invalid,
        )


def test_publish_writes_only_separate_immutable_children_and_authority(
    source, tmp_path,
):
    candidate = _candidate(source)
    artifact, children = proxy.build_authority(
        source_commit="a" * 40,
        target_manifest_sha256="b" * 64,
        candidates=[candidate],
        validator=_passing_certification,
    )
    output = tmp_path / "external"
    output.mkdir(mode=0o700)
    authority = proxy.publish_authority(
        output_root=output,
        output_name="trace-proxies.json",
        artifact=artifact,
        children=children,
    )
    assert authority == output / "trace-proxies.json"
    assert set(path.name for path in output.iterdir()) == {
        "trace-proxies.json", *children,
    }
    for name, payload in children.items():
        assert (output / name).read_bytes() == payload
    assert json.loads(authority.read_bytes()) == artifact
    with pytest.raises(proxy.TraceScaleProxyError, match="not empty"):
        proxy.publish_authority(
            output_root=output,
            output_name="second.json",
            artifact=artifact,
            children=children,
        )


def test_planning_and_authority_builder_do_not_cross_the_live_gpu_boundary(
    source, monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("live SOL runner called")

    monkeypatch.setattr(proxy.sr, "run_certified", forbidden)
    candidate = _candidate(source)
    artifact, _ = proxy.build_authority(
        source_commit="a" * 40,
        target_manifest_sha256="b" * 64,
        candidates=[candidate],
        validator=_passing_certification,
    )
    assert artifact["proxy_count"] == 1


def test_explicit_sol_boundary_binds_exact_temp_bytes_without_a_gpu(
    source, monkeypatch,
):
    candidate = _candidate(source)
    observed = {}

    def fake_run(cell, *, timeout, probe_contract, workload_path, definition_path):
        observed.update(
            cell=cell,
            timeout=timeout,
            probe_contract=probe_contract,
            workload=workload_path.read_bytes(),
            definition=definition_path.read_bytes(),
        )
        return {
            "status": "runnable",
            "diagnostics_complete": True,
            "workload_diagnostics": [
                {"diagnostic_present": True, "status": "PASSED"}
            ],
        }

    monkeypatch.setattr(proxy.sr, "run_certified", fake_run)
    result = proxy.certify_with_sol(
        candidate, probe_contract={"contract": "fixture"}, timeout=17,
    )
    proxy.validate_certification(candidate, result)
    assert observed == {
        "cell": source["cell"],
        "timeout": 17,
        "probe_contract": {"contract": "fixture"},
        "workload": candidate["_workload_bytes"],
        "definition": candidate["_definition_bytes"],
    }


def test_explicit_sol_boundary_rejects_incomplete_diagnostic_even_if_cell_says_runnable(
    source, monkeypatch,
):
    candidate = _candidate(source)
    monkeypatch.setattr(
        proxy.sr,
        "run_certified",
        lambda *_args, **_kwargs: {
            "status": "runnable",
            "diagnostics_complete": False,
            "workload_diagnostics": [],
        },
    )
    result = proxy.certify_with_sol(candidate, probe_contract={})
    assert result["correctness_result"] == "failed"
    with pytest.raises(proxy.TraceScaleProxyError, match="did not pass"):
        proxy.validate_certification(candidate, result)
