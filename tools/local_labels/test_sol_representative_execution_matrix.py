"""Contracts for clean-HEAD SOL representative execution projection."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import build_sol_representative_execution_matrix as execution


REPO_ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
OLD_LAUNCH = {
    "inventory_policy": "requires_inventory_before_capture",
    "launch_scope": "all_in_profiler_roi",
    "trace_requirement": "every_inventory_launch_complete",
    "ncu_label_requirement": "every_inventory_launch_complete",
    "control_requirement": "every_inventory_launch_complete",
    "admission_policy": "all_three_layers_complete_for_exact_inventory",
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        check=True,
    )
    return result.stdout.strip()


def _source_budget(seconds: int = 7200) -> dict:
    return {
        "status": "declared",
        "policy_path": "config/policy.json",
        "policy_sha256": SHA,
        "trace_budgets_path": "config/budgets.toml",
        "trace_budgets_sha256": SHA,
        "explicit_slug_budget_sha256": None,
        "derivation": "explicit_slug_or_policy_default_ceiling",
        "max_trace_bytes": 1024,
        "max_wall_time_s": seconds,
        "sampling_policy": "off",
        "breach_disposition": "hard_refusal_preserve_complete_trace",
        "partial_trace_allowed": False,
    }


def _source_variant(*, cohort: str = "native", seconds: int = 7200) -> dict:
    slug = "problem_001_unit"
    variant: dict = {
        "slug": slug,
        "category": "L1",
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(OLD_LAUNCH),
        "workload_scope": "native" if cohort != "reduced_proxy" else "reduced_representative",
        "variant_id": "native" if cohort == "native" else "b" * 64,
        "cohort": cohort,
        "claim_scope": {
            "native": "native_workload_row",
            "partial_native_row": "partial_native_workload_row_only",
            "reduced_proxy": "reduced_proxy_bundle_only",
        }[cohort],
        "native_predictive": cohort == "native",
        "native_shape": cohort != "reduced_proxy",
        "problem_level_runnable": cohort == "native",
        "source_workload_file": (
            "tools/sol_execbench/data/benchmark/L1/unit/workload.jsonl"
        ),
        "source_workload_file_sha256": SHA,
        "source_workload_count": 1,
        "definition_file": (
            "tools/sol_execbench/data/benchmark/L1/unit/definition.json"
        ),
        "definition_file_sha256": SHA,
        "solution_file": "inputs/solution.json",
        "solution_file_sha256": SHA,
        "adapter_contract": {"flashinfer_corpus": None},
        "adapter_contract_sha256": SHA,
        "capture_target": {},
        "native_env_hash": SHA,
        "invocation": {"executable_path": "runtime/sol", "executable_sha256": SHA},
        "trace_budget": _source_budget(seconds),
        "control_bits": {},
        "supplemental_cell_contract_sha256": SHA,
    }
    if cohort == "partial_native_row":
        variant["selected_diagnostic_sha256"] = SHA
        variant["runnability_evidence"] = {}
    elif cohort == "reduced_proxy":
        variant["reduced_provenance"] = {"row_plan_sha256": SHA}
    else:
        variant["runnability_evidence"] = {}
    # The source cell seal is not exercised by rebuild; the source variant seal is.
    variant["variant_contract_sha256"] = execution.canonical_sha256(variant)
    return variant


def _row(*, cohort: str, index: int) -> dict:
    variant = _source_variant(cohort=cohort)
    row = {
        "key": {
            "slug": f"problem_{index:03d}_unit",
            "source_workload_uuid": f"uuid-{cohort}-{index}",
            "variant_id": variant["variant_id"],
        },
        "cohort": cohort,
        "category": "L1",
        "flashinfer_corpus": None,
        "launch_scope": "all_in_profiler_roi",
        "launch_completeness_contract": copy.deepcopy(OLD_LAUNCH),
        "claim_scope": variant["claim_scope"],
        "native_predictive": variant["native_predictive"],
        "native_shape": variant["native_shape"],
        "problem_level_runnable": variant["problem_level_runnable"],
        "variant_contract_sha256": variant["variant_contract_sha256"],
        "row_evidence_sha256": SHA,
        "source_workload_index": index,
        "source_workload_uuid": f"uuid-{cohort}-{index}",
        "source_axes": {"n": 1},
        "source_row_sha256": SHA,
        "capture_workload_index": index,
        "capture_axes": {"n": 1},
        "capture_row_sha256": SHA,
    }
    return row


def _execution_authority_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict, dict, bytes, dict, bytes]:
    """Build a one-row authority in a disposable Git repository."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    payloads = {
        "tools/sol_execbench/data/benchmark/L1/unit/workload.jsonl": b"{}\n",
        "tools/sol_execbench/data/benchmark/L1/unit/definition.json": b"{}\n",
        "inputs/solution.json": b"{}\n",
        "config/policy.json": b"{}\n",
        "config/budgets.toml": b"x=1\n",
        "config/env.json": b"{}\n",
        "tool.py": b"tool\n",
    }
    for relative, payload in payloads.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "authority and execution")
    commit = _git(repo, "rev-parse", "HEAD")

    variant = _source_variant()
    for field, relative in (
        (
            "source_workload_file",
            "tools/sol_execbench/data/benchmark/L1/unit/workload.jsonl",
        ),
        (
            "definition_file",
            "tools/sol_execbench/data/benchmark/L1/unit/definition.json",
        ),
        ("solution_file", "inputs/solution.json"),
    ):
        variant[field] = relative
        variant[field + "_sha256"] = execution.sha256_bytes(payloads[relative])
    variant["trace_budget"]["policy_path"] = "config/policy.json"
    variant["trace_budget"]["policy_sha256"] = execution.sha256_bytes(
        payloads["config/policy.json"]
    )
    variant["trace_budget"]["trace_budgets_path"] = "config/budgets.toml"
    variant["trace_budget"]["trace_budgets_sha256"] = execution.sha256_bytes(
        payloads["config/budgets.toml"]
    )
    variant["supplemental_cell_contract_sha256"] = execution.canonical_sha256(
        execution._variant_cell_contract(variant)
    )
    variant.pop("variant_contract_sha256")
    variant["variant_contract_sha256"] = execution.canonical_sha256(variant)

    row = _row(cohort="native", index=0)
    row["key"]["slug"] = variant["slug"]
    row["variant_contract_sha256"] = variant["variant_contract_sha256"]
    source_matrix = {
        "source_commit": commit,
        "input_pins": {"sol_native_env_fingerprint": {
            "path": "config/env.json",
            "sha256": execution.sha256_bytes(payloads["config/env.json"]),
        }},
        "canonical_matrix": {"fixture": "canonical_matrix"},
        "canonical_authority": {"fixture": "canonical_authority"},
        "runnability_authority": {"fixture": "runnability_authority"},
        "supplemental_budget_policy": {"fixture": "budget"},
        "reduced_proxy_authority": {"fixture": "reduced"},
        "claim_policy": {"fixture": "claim"},
        "variants": [variant],
        "rows": [row],
    }
    source_matrix["matrix_seal_sha256"] = execution.source_builder.canonical_sha256(
        source_matrix
    )
    source_payload = execution.canonical_bytes(source_matrix)
    selection_identity = "9" * 64
    representative = {
        "artifact_id": "sm86_sol_representative_capture_matrix_v1",
        "source_commit": commit,
        "input_pins": {"source_matrix": {
            "sha256": execution.sha256_bytes(source_payload),
        }},
        "selection_receipt": {
            "selection_identity_sha256": selection_identity,
        },
        "counts": {
            "application_coverage_anchors": 1,
            "native_comparison_companions": 0,
            "architecture_inapplicable_problems": 0,
            "architecture_inapplicable_rows": 0,
        },
        "architecture_exclusions": [],
        "selections": [{
            "source_matrix_row_key": copy.deepcopy(row["key"]),
            "source_matrix_row_sha256": execution.canonical_sha256(row),
            "source_matrix_variant_sha256": variant["variant_contract_sha256"],
        }],
    }
    representative["matrix_seal_sha256"] = execution.selector.canonical_sha256(
        representative
    )
    representative_payload = execution.canonical_bytes(representative)

    schema_path = repo / "schema.json"
    schema_path.write_text(json.dumps({"type": "object"}))
    monkeypatch.setattr(execution, "SCHEMA_PATH", Path("schema.json"))
    monkeypatch.setattr(execution, "EXECUTION_TOOL_PATHS", ("tool.py",))
    monkeypatch.setattr(execution, "EXECUTION_TOOL_PREFIXES", ())
    monkeypatch.setattr(execution, "PINNED_AUTHORITY_COMMIT", commit)
    monkeypatch.setattr(
        execution, "PINNED_SOURCE_MATRIX_SHA256",
        execution.sha256_bytes(source_payload),
    )
    monkeypatch.setattr(
        execution, "PINNED_SOURCE_MATRIX_SEAL_SHA256",
        source_matrix["matrix_seal_sha256"],
    )
    monkeypatch.setattr(
        execution, "PINNED_REPRESENTATIVE_SELECTION_SHA256",
        execution.sha256_bytes(representative_payload),
    )
    monkeypatch.setattr(
        execution, "PINNED_REPRESENTATIVE_SELECTION_SEAL_SHA256",
        representative["matrix_seal_sha256"],
    )
    monkeypatch.setattr(
        execution, "PINNED_SELECTION_IDENTITY_SHA256", selection_identity,
    )
    document = execution.build_execution_matrix(
        source_matrix=source_matrix,
        source_matrix_payload=source_payload,
        representative=representative,
        representative_payload=representative_payload,
        repo_root=repo,
        execution_commit=commit,
        require_clean_head=False,
        enforce_current_counts=False,
    )
    return (
        repo, document, source_matrix, source_payload,
        representative, representative_payload,
    )


def _reseal_document(document: dict) -> None:
    document["matrix_seal_sha256"] = execution.canonical_sha256({
        key: value for key, value in document.items()
        if key != "matrix_seal_sha256"
    })


def test_schema_is_valid_and_default_artifact_is_additive():
    schema = json.loads((REPO_ROOT / execution.SCHEMA_PATH).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert set(schema["$defs"]["runtimeInput"]["properties"]["kind"]["enum"]) == {
        "runtime_executable",
        "retained_reduced_workload",
        "retained_reduced_definition",
        "native_workload_behind_reduced_proxy",
        "native_definition_behind_reduced_proxy",
        "sol_benchmark_workload",
        "sol_benchmark_definition",
        "sol_benchmark_tree",
    }
    assert execution.DEFAULT_OUTPUT_PATH.name == (
        "sol_representative_execution_matrix.live.v1.json"
    )
    assert execution.DEFAULT_OUTPUT_PATH != execution.selector.DEFAULT_OUTPUT_PATH
    assert execution.ARTIFACT_ID.endswith("representative_execution_matrix_live_v1")


@pytest.mark.parametrize(
    ("source_seconds", "expected"), [(7200, 900), (900, 900), (125, 125)],
)
def test_representative_budget_caps_without_sampling_or_partial_trace(
    source_seconds: int, expected: int,
):
    budget = execution._representative_budget(_source_budget(source_seconds))
    assert budget["source_max_wall_time_s"] == source_seconds
    assert budget["max_wall_time_s"] == expected
    assert budget["source_max_trace_bytes"] == 1024
    assert budget["max_trace_bytes"] == 1024
    assert budget["sampling_policy"] == "off"
    assert budget["partial_trace_allowed"] is False
    assert budget["timeout_disposition"] == (
        "typed_measured_infeasibility_for_trace_proxy_planning"
    )
    assert budget["trace_limit_disposition"] == (
        "typed_measured_infeasibility_for_trace_proxy_planning"
    )


def test_reduced_external_children_are_not_misclassified_as_git_inputs():
    variant = _source_variant(cohort="reduced_proxy")
    variant["source_workload_file"] = "retained-reduced.jsonl"
    variant["source_workload_file_sha256"] = "1" * 64
    variant["definition_file"] = "retained-definition.json"
    variant["definition_file_sha256"] = "2" * 64
    variant["reduced_provenance"].update({
        "native_workload_file": (
            "tools/sol_execbench/data/benchmark/L1/unit/native-workload.jsonl"
        ),
        "native_workload_file_sha256": "3" * 64,
        "native_definition_file": (
            "tools/sol_execbench/data/benchmark/L1/unit/native-definition.json"
        ),
        "native_definition_file_sha256": "4" * 64,
    })
    variant["invocation"]["logical_workload"] = {
        "external_workload_source": {
            "name": "retained-reduced.jsonl", "sha256": "1" * 64,
        },
        "external_definition_source": {
            "name": "retained-definition.json", "sha256": "2" * 64,
        },
    }
    matrix = {"input_pins": {"sol_native_env_fingerprint": {
        "path": "config/env.json", "sha256": "5" * 64,
    }}}

    static = execution._static_input_pins([variant], matrix)
    assert "retained-reduced.jsonl" not in static
    assert "retained-definition.json" not in static
    assert not any("workload" in path or "definition" in path for path in static)
    external = execution._external_runtime_pins([variant])
    assert {(row["kind"], row["name"], row["sha256"]) for row in external} >= {
        ("retained_reduced_workload", "retained-reduced.jsonl", "1" * 64),
        ("retained_reduced_definition", "retained-definition.json", "2" * 64),
        (
            "native_workload_behind_reduced_proxy",
            "tools/sol_execbench/data/benchmark/L1/unit/native-workload.jsonl",
            "3" * 64,
        ),
        (
            "native_definition_behind_reduced_proxy",
            "tools/sol_execbench/data/benchmark/L1/unit/native-definition.json",
            "4" * 64,
        ),
    }


def test_rebuild_reseals_variant_and_keeps_source_receipts():
    source = _source_variant(seconds=7200)
    rebuilt = execution.rebuild_execution_variant(
        source, authority_commit="1" * 40, execution_commit="2" * 40,
        carry_forward_sha256="3" * 64,
    )
    assert rebuilt["authority_variant_contract_sha256"] == source[
        "variant_contract_sha256"
    ]
    assert rebuilt["source_trace_budget"] == source["trace_budget"]
    assert rebuilt["trace_budget"]["max_wall_time_s"] == 900
    assert rebuilt["launch_completeness_contract"] == (
        execution.REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    assert rebuilt["variant_contract_sha256"] != source[
        "variant_contract_sha256"
    ]
    unsigned = {key: value for key, value in rebuilt.items()
                if key != "variant_contract_sha256"}
    assert execution.canonical_sha256(unsigned) == rebuilt[
        "variant_contract_sha256"
    ]
    assert execution.canonical_sha256(execution._variant_cell_contract(rebuilt)) == (
        rebuilt["supplemental_cell_contract_sha256"]
    )
    assert execution._recover_authority_variant(rebuilt) == source


def test_resealed_execution_variant_cannot_rewrite_authority_content():
    source = _source_variant(seconds=7200)
    rebuilt = execution.rebuild_execution_variant(
        source, authority_commit="1" * 40, execution_commit="2" * 40,
        carry_forward_sha256="3" * 64,
    )
    rebuilt["source_workload_count"] = 2
    rebuilt["variant_contract_sha256"] = execution.canonical_sha256({
        key: value for key, value in rebuilt.items()
        if key != "variant_contract_sha256"
    })
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="does not reverse to its authority seal",
    ):
        execution._recover_authority_variant(rebuilt)


def test_rebuild_reseals_row_against_execution_variant():
    source_variant = _source_variant()
    source_row = _row(cohort="native", index=0)
    source_row["key"]["slug"] = source_variant["slug"]
    source_row["variant_contract_sha256"] = source_variant[
        "variant_contract_sha256"
    ]
    row_sha = execution.canonical_sha256(source_row)
    variant = execution.rebuild_execution_variant(
        source_variant, authority_commit="1" * 40,
        execution_commit="2" * 40, carry_forward_sha256="3" * 64,
    )
    row = execution.rebuild_execution_row(
        source_row, variant=variant, authority_row_sha256=row_sha,
        execution_commit="2" * 40,
    )
    assert row["authority_row_contract_sha256"] == row_sha
    assert row["authority_variant_contract_sha256"] == source_variant[
        "variant_contract_sha256"
    ]
    assert row["variant_contract_sha256"] == variant[
        "variant_contract_sha256"
    ]
    assert row["launch_completeness_contract"] == (
        execution.REPRESENTATIVE_LAUNCH_COMPLETENESS_CONTRACT
    )
    assert execution._recover_authority_row(
        row, execution_variant=variant
    ) == source_row


def test_execution_row_cannot_rewrite_authority_content_behind_receipt():
    source_variant = _source_variant()
    source_row = _row(cohort="native", index=0)
    source_row["key"]["slug"] = source_variant["slug"]
    source_row["variant_contract_sha256"] = source_variant[
        "variant_contract_sha256"
    ]
    variant = execution.rebuild_execution_variant(
        source_variant, authority_commit="1" * 40,
        execution_commit="2" * 40, carry_forward_sha256="3" * 64,
    )
    row = execution.rebuild_execution_row(
        source_row, variant=variant,
        authority_row_sha256=execution.canonical_sha256(source_row),
        execution_commit="2" * 40,
    )
    row["capture_axes"] = {"n": 2}
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="does not reverse to its authority seal",
    ):
        execution._recover_authority_row(row, execution_variant=variant)


def test_chunk_order_is_cohort_pure_and_first_pilot_is_cross_cohort():
    rows = [
        *[_row(cohort="native", index=index) for index in range(18)],
        *[_row(cohort="reduced_proxy", index=100 + index) for index in range(17)],
        *[_row(cohort="partial_native_row", index=200 + index) for index in range(3)],
    ]
    chunks = execution._interleaved_chunks(rows)
    assert [chunk["cohort"] for chunk in chunks[:3]] == [
        "native", "reduced_proxy", "partial_native_row",
    ]
    assert [chunk["cohort"] for chunk in chunks[3:]] == [
        "native", "reduced_proxy",
    ]
    assert [chunk["ordinal"] for chunk in chunks] == list(
        range(1, len(chunks) + 1)
    )
    for chunk in chunks:
        assert 1 <= chunk["row_count"] <= 16
        assert chunk["membership_sha256"] == execution.canonical_sha256(
            chunk["rows"]
        )
        assert {row["slug"].split("_")[1] for row in chunk["rows"]}


def test_carry_forward_proof_rejects_semantic_drift_and_repins_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    for relative, payload in {
        "tools/sol_execbench/data/benchmark/L1/unit/workload.jsonl": b"{}\n",
        "tools/sol_execbench/data/benchmark/L1/unit/definition.json": b"{}\n",
        "inputs/solution.json": b"{}\n",
        "config/policy.json": b"{}\n",
        "config/budgets.toml": b"x=1\n",
        "config/env.json": b"{}\n",
        "tool.py": b"old\n",
    }.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "authority")
    authority = _git(repo, "rev-parse", "HEAD")

    variant = _source_variant()
    for field, path in (
        (
            "source_workload_file",
            "tools/sol_execbench/data/benchmark/L1/unit/workload.jsonl",
        ),
        (
            "definition_file",
            "tools/sol_execbench/data/benchmark/L1/unit/definition.json",
        ),
        ("solution_file", "inputs/solution.json"),
    ):
        variant[field] = path
        variant[field + "_sha256"] = execution.sha256_bytes((repo / path).read_bytes())
    variant["trace_budget"]["policy_sha256"] = execution.sha256_bytes(
        (repo / "config/policy.json").read_bytes()
    )
    variant["trace_budget"]["trace_budgets_sha256"] = execution.sha256_bytes(
        (repo / "config/budgets.toml").read_bytes()
    )
    matrix = {"input_pins": {"sol_native_env_fingerprint": {
        "path": "config/env.json",
        "sha256": execution.sha256_bytes((repo / "config/env.json").read_bytes()),
    }}}
    monkeypatch.setattr(execution, "EXECUTION_TOOL_PATHS", ("tool.py",))
    monkeypatch.setattr(execution, "EXECUTION_TOOL_PREFIXES", ())
    (repo / "tool.py").write_bytes(b"new\n")
    _git(repo, "add", "tool.py")
    _git(repo, "commit", "-qm", "execution")
    current = _git(repo, "rev-parse", "HEAD")
    proof = execution.build_carry_forward_proof(
        repo_root=repo, authority_commit=authority, execution_commit=current,
        variants=[variant], source_matrix=matrix,
    )
    assert {row["disposition"] for row in proof["semantic_inputs"]} == {
        "byte_identical_carry_forward"
    }
    assert proof["execution_tools"][0]["disposition"] == (
        "deliberately_updated_and_repinned"
    )
    unsigned = {key: value for key, value in proof.items()
                if key != "proof_sha256"}
    assert proof["proof_sha256"] == execution.canonical_sha256(unsigned)

    (repo / "inputs/solution.json").write_bytes(b'{"changed":true}\n')
    _git(repo, "add", "inputs/solution.json")
    _git(repo, "commit", "-qm", "semantic drift")
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="semantic input changed",
    ):
        execution.build_carry_forward_proof(
            repo_root=repo, authority_commit=authority,
            execution_commit=_git(repo, "rev-parse", "HEAD"),
            variants=[variant], source_matrix=matrix,
        )


def test_validator_rejects_resealed_row_and_variant_projection_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    (
        repo, document, source_matrix, source_payload,
        representative, representative_payload,
    ) = _execution_authority_fixture(tmp_path, monkeypatch)

    row_tamper = copy.deepcopy(document)
    row_tamper["rows"][0]["capture_axes"] = {"n": 12}
    _reseal_document(row_tamper)
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="execution rows are not the exact deterministic upstream projection",
    ):
        execution.validate_execution_matrix(
            row_tamper,
            repo_root=repo,
            require_clean_head=False,
            enforce_current_counts=False,
            source_matrix=source_matrix,
            source_matrix_payload=source_payload,
            representative=representative,
            representative_payload=representative_payload,
        )

    variant_tamper = copy.deepcopy(document)
    variant = variant_tamper["variants"][0]
    variant["invocation"]["trace_app_args"] = ["--mutated"]
    variant["supplemental_cell_contract_sha256"] = execution.canonical_sha256(
        execution._variant_cell_contract(variant)
    )
    variant["variant_contract_sha256"] = execution.canonical_sha256({
        key: value for key, value in variant.items()
        if key != "variant_contract_sha256"
    })
    row = variant_tamper["rows"][0]
    row["variant_contract_sha256"] = variant["variant_contract_sha256"]
    row["execution_row_contract_sha256"] = execution.canonical_sha256({
        "authority_row_contract_sha256": row[
            "authority_row_contract_sha256"
        ],
        "execution_commit": variant_tamper["source_commit"],
        "execution_profile": execution.EXECUTION_PROFILE,
        "execution_variant_contract_sha256": variant[
            "variant_contract_sha256"
        ],
        "row_key": row["key"],
    })
    _reseal_document(variant_tamper)
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="execution variants are not the exact deterministic upstream projection",
    ):
        execution.validate_execution_matrix(
            variant_tamper,
            repo_root=repo,
            require_clean_head=False,
            enforce_current_counts=False,
            source_matrix=source_matrix,
            source_matrix_payload=source_payload,
            representative=representative,
            representative_payload=representative_payload,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda proof: proof["semantic_inputs"][0].__setitem__(
            "authority_blob", "f" * 40
        ),
        lambda proof: proof["execution_tools"][0].__setitem__(
            "authority_sha256", "e" * 64
        ),
        lambda proof: proof["execution_tools"][0].__setitem__(
            "execution_blob", "d" * 40
        ),
        lambda proof: proof["execution_tools"][0].__setitem__(
            "disposition", "deliberately_updated_and_repinned"
        ),
        lambda proof: proof["execution_tools"].pop(),
        lambda proof: proof["execution_tools"].append(
            {"path": "extra.py", "role": "execution_tool"}
        ),
    ],
)
def test_validator_rejects_resealed_false_carry_forward_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate,
):
    (
        repo, document, source_matrix, source_payload,
        representative, representative_payload,
    ) = _execution_authority_fixture(tmp_path, monkeypatch)
    candidate = copy.deepcopy(document)
    mutate(candidate["carry_forward_proof"])
    proof = candidate["carry_forward_proof"]
    proof["proof_sha256"] = execution.canonical_sha256({
        key: value for key, value in proof.items() if key != "proof_sha256"
    })
    _reseal_document(candidate)
    with pytest.raises(execution.RepresentativeExecutionError):
        execution.validate_execution_matrix(
            candidate,
            repo_root=repo,
            require_clean_head=False,
            enforce_current_counts=False,
            source_matrix=source_matrix,
            source_matrix_payload=source_payload,
            representative=representative,
            representative_payload=representative_payload,
        )


def test_runtime_validation_requires_exact_non_symlink_upstream_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    (
        repo, document, _source_matrix, source_payload,
        _representative, representative_payload,
    ) = _execution_authority_fixture(tmp_path, monkeypatch)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    matrix_path = bundle / "execution.json"
    matrix_path.write_bytes(execution.canonical_bytes(document))
    source_path = bundle / document["source_matrix"]["name"]
    selection_path = bundle / document["representative_selection"]["name"]
    source_path.write_bytes(source_payload)
    selection_path.write_bytes(representative_payload)

    execution.validate_execution_matrix(
        document,
        repo_root=repo,
        require_clean_head=False,
        enforce_current_counts=False,
        matrix_path=matrix_path,
    )

    source_path.unlink()
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="cannot read retained source matrix sibling",
    ):
        execution.validate_execution_matrix(
            document,
            repo_root=repo,
            require_clean_head=False,
            enforce_current_counts=False,
            matrix_path=matrix_path,
        )
    source_path.write_bytes(source_payload)
    selection_path.write_bytes(representative_payload + b"\n")
    with pytest.raises(
        execution.RepresentativeExecutionError,
        match="representative selection is not the pinned 216-row authority",
    ):
        execution.validate_execution_matrix(
            document,
            repo_root=repo,
            require_clean_head=False,
            enforce_current_counts=False,
            matrix_path=matrix_path,
        )
