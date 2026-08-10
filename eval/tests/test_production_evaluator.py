from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from eval import production_evaluator_fixture as fixture_evaluator
from eval import production_labels
from eval import score as score_module
from eval.contracts.validation import (
    canonical_launch_key_token,
    canonical_sha256,
    loads_json_strict,
)
from eval.production_evaluator_fixture import (
    FixtureAlgebraRequest,
    FixtureLaunchFamily,
    evaluate_non_authoritative_fixture_algebra,
)
from eval.tests.test_production_labels import _make_case
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    publish_retained_regular_file,
    sha256_bytes,
)

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = Path(__file__).parent / "fixtures/evaluator_mvp"
_RESULT_TEMPLATE = _FIXTURE_DIR / "sim_result.v2.json"
_ENTRYPOINT = _ROOT / "eval/production_evaluator_entrypoint.py"


@dataclass(frozen=True)
class _Material:
    crosswalk: production_labels.VerifiedProductionCrosswalk
    crosswalk_bytes: bytes
    labels_bytes: bytes
    authority_bytes: bytes
    authority_pin: production_labels.ProductionAuthorityPin
    result_bytes: bytes


def _seal_result(model: dict[str, Any]) -> bytes:
    model["result_sha256"] = canonical_sha256(model, "result_sha256")
    return canonical_json_bytes(model)


def _result_for_launch_keys(keys: list[dict[str, Any]]) -> bytes:
    model = loads_json_strict(
        _RESULT_TEMPLATE.read_text(encoding="utf-8"), source="result template"
    )
    base_launch = model["launches"][0]
    base_diagnostic = model["diagnostics"]["per_launch"][0]
    model["launches"] = []
    model["diagnostics"]["per_launch"] = []
    for key in keys:
        launch = copy.deepcopy(base_launch)
        launch["launch_key"] = copy.deepcopy(key)
        model["launches"].append(launch)
        diagnostic = copy.deepcopy(base_diagnostic)
        diagnostic["launch_key"] = copy.deepcopy(key)
        model["diagnostics"]["per_launch"].append(diagnostic)
    count = len(keys)
    model["aggregate"]["launch_count"] = count
    for metric in model["aggregate"]["metrics"].values():
        if metric["status"] != "predicted":
            continue
        value = metric["value"]
        if value["kind"] == "exact":
            value["value"] *= count
        else:
            value["numerator"] *= count
            value["denominator"] *= count
    return _seal_result(model)


def _production_pin(authority: dict[str, Any], authority_bytes: bytes):
    source = authority["source_label_identity"]
    return production_labels.ProductionAuthorityPin(
        authority_file_sha256=sha256_bytes(authority_bytes),
        producer_content_manifest_sha256=authority["producer_content_manifest"][
            "content_manifest_sha256"
        ],
        label_set_sha256=source["label_set_sha256"],
        source_manifest_sha256=source["source_manifest_sha256"],
        crosswalk_file_sha256=authority["crosswalk_identity"][
            "crosswalk_file_sha256"
        ],
        builder_sha256=authority["builder_identity"]["builder_sha256"],
    )


@pytest.fixture(scope="module")
def material(tmp_path_factory: pytest.TempPathFactory) -> _Material:
    root = tmp_path_factory.mktemp("production-evaluator-material")
    case = _make_case(root)
    artifacts = production_labels.build_production_matched_label_artifacts(
        source=case.source,
        crosswalk=case.crosswalk,
        repeats=case.repeats,
    )
    keys = [launch.launch_key.to_model() for launch in case.crosswalk.launches]
    return _Material(
        crosswalk=case.crosswalk,
        crosswalk_bytes=(root / "crosswalk.json").read_bytes(),
        labels_bytes=artifacts.matched_labels_bytes,
        authority_bytes=artifacts.authority_bytes,
        authority_pin=_production_pin(
            artifacts.authority_document, artifacts.authority_bytes
        ),
        result_bytes=_result_for_launch_keys(keys),
    )


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _request(
    tmp_path: Path,
    material: _Material,
    *,
    result_bytes: bytes | None = None,
    labels_bytes: bytes | None = None,
    authority_bytes: bytes | None = None,
    authority_pin: production_labels.ProductionAuthorityPin | None = None,
) -> FixtureAlgebraRequest:
    result = _write(
        tmp_path / "result.json",
        material.result_bytes if result_bytes is None else result_bytes,
    )
    labels = _write(
        tmp_path / "labels.json",
        material.labels_bytes if labels_bytes is None else labels_bytes,
    )
    authority = _write(
        tmp_path / "authority.json",
        material.authority_bytes if authority_bytes is None else authority_bytes,
    )
    launches = tuple(
        FixtureLaunchFamily(
            canonical_launch_key_token(launch.launch_key.to_model()),
            launch.kernel_family,
        )
        for launch in material.crosswalk.launches
    )
    return FixtureAlgebraRequest(
        result_path=result,
        labels_path=labels,
        label_authority_path=authority,
        label_authority_pin=(
            material.authority_pin if authority_pin is None else authority_pin
        ),
        executable_sha256=material.crosswalk.executable_sha256,
        binary_architecture=material.crosswalk.binary_architecture,
        gpu_sku=material.crosswalk.gpu_sku,
        launches=launches,
    )


def test_package_boundary_is_exact_refusal_and_no_ast_loader_exists() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import eval.production_evaluator"],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "ordinary package import is forbidden" in completed.stderr
    standalone = subprocess.run(
        [sys.executable, "-c", "import eval.production_evaluator_entrypoint"],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert standalone.returncode != 0
    assert "source-pinned CLI" in standalone.stderr
    assert "_load_explicit_fixture_runtime" not in globals()
    assert "_load_explicit_fixture_facade" not in globals()


def test_fixture_module_has_no_production_admission_surface() -> None:
    forbidden = {
        "evaluate",
        "evaluate_production",
        "authenticate_production_evaluator_build_identity",
        "ProductionEvaluationRequest",
        "publish_retained_regular_file",
    }
    assert forbidden.isdisjoint(vars(fixture_evaluator))
    score_forbidden = {
        "AuthenticatedEvaluatorBuildIdentity",
        "score_authenticated_snapshot",
        "_score_authenticated_snapshot",
        "_mint_authenticated_scoring_capability",
        "_mint_authenticated_evaluator_build_identity",
        "_build_synthetic_accuracy_report",
    }
    assert score_forbidden.isdisjoint(vars(score_module))
    assert score_forbidden.isdisjoint(score_module.__all__)
    assert set(score_module.__all__) == {
        "EvaluatorBuildIdentity",
        "EvaluatorScoringError",
        "KNOWN_SYNTHETIC_BUILD_IDENTITY_FILE_SHA256",
        "ResultSnapshot",
        "load_result_snapshot",
        "load_synthetic_evaluator_build_identity",
        "main",
        "score_synthetic_fixture",
    }
    assert all(hasattr(score_module, name) for name in score_module.__all__)
    assert set(fixture_evaluator.__all__) == {
        "FixtureAlgebraRequest",
        "FixtureLaunchFamily",
        "evaluate_non_authoritative_fixture_algebra",
    }
    source = Path(fixture_evaluator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.FunctionDef)
        and ("report" in node.name or "receipt" in node.name)
        for node in ast.walk(tree)
    )


def test_score_private_identity_builder_is_intrinsically_fixture_only() -> None:
    fixture_id = "evaluator-mvp-canary-v1"
    result = score_module.load_result_snapshot(_RESULT_TEMPLATE)
    labels = score_module.load_synthetic_matched_labels(
        _FIXTURE_DIR / "matched_labels.v1.json",
        _FIXTURE_DIR / "synthetic_matched_label_authority.v1.json",
        authority_pin=score_module.known_synthetic_authority_pin(fixture_id),
    )
    pinned = score_module.load_synthetic_evaluator_build_identity(
        _FIXTURE_DIR / "synthetic_evaluator_build_identity.v1.json",
        fixture_id=fixture_id,
    )
    forged_clean = score_module.EvaluatorBuildIdentity(
        evaluator_git_commit=pinned.evaluator_git_commit,
        evaluator_git_dirty=False,
        fixture_id=pinned.fixture_id,
        identity_sha256=pinned.identity_sha256,
    )
    with pytest.raises(
        score_module.EvaluatorScoringError,
        match="fixture-bound and unclaimable",
    ):
        score_module._report_identities(
            result,
            result.validated_document(),
            labels,
            forged_clean,
            score_module.load_metric_registry(),
        )
    with pytest.raises(TypeError):
        score_module._report_identities(
            result,
            result.validated_document(),
            labels,
            pinned,
            score_module.load_metric_registry(),
            report_id="production:forged",
        )


def test_positive_fixture_algebra_is_explicitly_unclaimable(
    tmp_path: Path,
    material: _Material,
) -> None:
    outcome = evaluate_non_authoritative_fixture_algebra(
        _request(tmp_path, material)
    )
    assert outcome["contract_mode"] == "FIXTURE_ONLY_NON_AUTHORITATIVE_ALGEBRA"
    assert outcome["claimability"] == {
        "status": "unclaimable",
        "reason": "fixture_only_non_authoritative_algebra",
    }
    assert not ({"admission", "identities", "report_sha256"} & set(outcome))
    assert outcome["coverage"]["scored_cells"] == 14
    assert outcome["coverage"]["unavailable_cells"] == 2
    rows = outcome["per_metric_family"]
    assert len(rows) == 8
    assert {row["kernel_family"] for row in rows} == {"vector"}
    cycles = next(row for row in rows if row["metric_key"] == "exec_cycles_gpc_max")
    assert cycles["n_scored"] == 0
    assert cycles["n_unavailable"] == 2


def _run_m1_cli(mode: str, *, source_pin: str | None = None):
    pin = sha256_bytes(_ENTRYPOINT.read_bytes()) if source_pin is None else source_pin
    completed = subprocess.run(
        [sys.executable, str(_ENTRYPOINT), pin, mode],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    model = json.loads(completed.stdout)
    return completed, model


@pytest.mark.parametrize("mode", ("evaluate", "evaluate_production"))
def test_source_pinned_cli_blocks_m1_before_missing_caller_paths(mode: str) -> None:
    completed, model = _run_m1_cli(mode)
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert model["code"] == "COHORT_AUTHORITY_UNAVAILABLE"
    assert "input" not in model["detail"].lower()


def test_standalone_entrypoint_rejects_wrong_source_pin_first() -> None:
    completed, model = _run_m1_cli("evaluate", source_pin="0" * 64)
    assert completed.returncode == 1
    assert model["code"] == "EVALUATOR_ENTRYPOINT_SOURCE_PIN_MISMATCH"


@pytest.mark.parametrize("mode", ("missing", "duplicate", "cross-namespace"))
def test_fixture_result_roster_fails_before_label_open(
    tmp_path: Path,
    material: _Material,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    keys = [launch.launch_key.to_model() for launch in material.crosswalk.launches]
    if mode == "missing":
        keys = keys[:1]
    elif mode == "duplicate":
        keys = [keys[0], copy.deepcopy(keys[0])]
    else:
        keys[0] = {**keys[0], "workload_id": "foreign-workload"}
    request = _request(
        tmp_path,
        material,
        result_bytes=_result_for_launch_keys(keys),
    )

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("labels opened before result roster closure")

    monkeypatch.setattr(fixture_evaluator, "load_production_matched_labels", forbidden)
    with pytest.raises((ImmutableArtifactError, ValueError)):
        evaluate_non_authoritative_fixture_algebra(request)


@pytest.mark.parametrize(
    "mode",
    (
        "target-leakage",
        "cycles-promotion",
        "executable",
        "card-config",
        "diagnostic-order",
    ),
)
def test_fixture_result_identity_and_cycle_failures_precede_labels(
    tmp_path: Path,
    material: _Material,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    result = loads_json_strict(material.result_bytes, source="sealed result")
    if mode == "target-leakage":
        result["target"] = {"gpc__cycles_elapsed.max": 1}
    elif mode == "cycles-promotion":
        promoted = {
            "status": "predicted",
            "value": {"kind": "exact", "semantics": "simulator_scope", "value": 1},
        }
        result["aggregate"]["metrics"]["exec_cycles_gpc_max"] = copy.deepcopy(
            promoted
        )
        for launch in result["launches"]:
            launch["metrics"]["exec_cycles_gpc_max"] = copy.deepcopy(promoted)
    elif mode == "executable":
        result["build_identity"]["binary_sha256"] = "e" * 64
    elif mode == "card-config":
        result["configuration_identity"]["arch_card_sha256"] = "d" * 64
    else:
        result["diagnostics"]["per_launch"].reverse()
    request = _request(tmp_path, material, result_bytes=_seal_result(result))

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("labels opened before result identity closure")

    monkeypatch.setattr(fixture_evaluator, "load_production_matched_labels", forbidden)
    with pytest.raises((ImmutableArtifactError, ValueError)):
        evaluate_non_authoritative_fixture_algebra(request)


def _reauthorize_labels(
    material: _Material,
    labels: dict[str, Any],
) -> tuple[bytes, bytes, production_labels.ProductionAuthorityPin]:
    labels["matched_labels_sha256"] = canonical_sha256(
        labels,
        "matched_labels_sha256",
    )
    labels_bytes = canonical_json_bytes(labels)
    authority = loads_json_strict(
        material.authority_bytes,
        source="production label authority",
    )

    authority["matched_labels"] = {
        **authority["matched_labels"],
        "file_sha256": sha256_bytes(labels_bytes),
        "size_bytes": len(labels_bytes),
        "canonical_seal_sha256": labels["matched_labels_sha256"],
    }
    authority["authority_sha256"] = canonical_sha256(
        authority,
        "authority_sha256",
    )
    authority_bytes = canonical_json_bytes(authority)
    return labels_bytes, authority_bytes, _production_pin(authority, authority_bytes)


def test_fixture_family_substitution_is_rejected(
    tmp_path: Path,
    material: _Material,
) -> None:
    labels = loads_json_strict(material.labels_bytes, source="production labels")
    labels["launches"][0]["kernel_family"] = "matrix"
    labels_bytes, authority_bytes, pin = _reauthorize_labels(material, labels)
    request = _request(
        tmp_path,
        material,
        labels_bytes=labels_bytes,
        authority_bytes=authority_bytes,
        authority_pin=pin,
    )
    with pytest.raises(ImmutableArtifactError, match="launch-family projection"):
        evaluate_non_authoritative_fixture_algebra(request)


def test_fixture_unit_substitution_is_rejected(
    tmp_path: Path,
    material: _Material,
) -> None:
    labels = loads_json_strict(material.labels_bytes, source="production labels")
    labels["launches"][0]["metrics"][0]["unit"] = "cycles"
    labels_bytes, authority_bytes, pin = _reauthorize_labels(material, labels)
    request = _request(
        tmp_path,
        material,
        labels_bytes=labels_bytes,
        authority_bytes=authority_bytes,
        authority_pin=pin,
    )
    with pytest.raises(ValueError):
        evaluate_non_authoritative_fixture_algebra(request)


def test_fixture_monkeypatch_surface_cannot_reach_production_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "evaluate_production",
        "_evaluate_production_retained",
        "_require_production_authority_preflight",
        "authenticate_production_evaluator_build_identity",
        "_mint_evaluator_build_identity",
    ):
        assert not hasattr(fixture_evaluator, name)
        with pytest.raises(AttributeError):
            monkeypatch.setattr(fixture_evaluator, name, lambda: None)


@pytest.mark.parametrize("mutation", ("unlink", "replace"))
def test_retained_receipt_primitive_detects_generation_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "receipt.json"
    retained = publish_retained_regular_file(
        tmp_path,
        path.name,
        canonical_json_bytes({"fixture": "receipt"}),
    )
    with pytest.raises(ImmutableArtifactError):
        with retained:
            payload = retained.read_bytes()
            if mutation == "unlink":
                path.unlink()
            else:
                replacement = tmp_path / "replacement.json"
                replacement.write_bytes(payload)
                os.replace(replacement, path)
            retained.verify_unchanged()


def test_standalone_is_preflight_only_without_admission_builders() -> None:
    source = _ENTRYPOINT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_symbols = {
        "evaluate",
        "evaluate_production",
        "evaluate_fixture",
        "_evaluate_production_retained",
        "_evaluate_fixture_retained",
        "_mint_evaluator_build_identity",
        "_publish_label_opening_receipt",
        "_score_lane_neutral_algebra",
        "authenticate_production_crosswalk",
        "load_production_matched_labels",
    }
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert forbidden_symbols.isdisjoint(defined)
    assert "accuracy_report.v1" not in source
    assert '"status": "admitted"' not in source
    assert "report_sha256" not in source
    assert "publish_retained_regular_file" not in source
    assert "from eval.score" not in source
    assert "from eval.production_labels" not in source


def test_direct_entrypoint_exec_leaves_no_production_admission_surface(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(_ENTRYPOINT),
    }
    monkeypatch.setattr(sys, "argv", [str(_ENTRYPOINT)])
    with pytest.raises(SystemExit) as stopped:
        exec(
            compile(_ENTRYPOINT.read_bytes(), str(_ENTRYPOINT), "exec"),
            namespace,
        )
    assert stopped.value.code == 2
    capsys.readouterr()

    forbidden = {
        "evaluate",
        "evaluate_production",
        "evaluate_fixture",
        "authenticate_production_evaluator_build_identity",
        "authenticate_production_crosswalk",
        "load_workload_execution_authority",
        "load_production_matched_labels",
        "_evaluate_production_retained",
        "_evaluate_fixture_retained",
        "_mint_evaluator_build_identity",
        "_publish_label_opening_receipt",
        "_score_lane_neutral_algebra",
        "crosswalk_contract",
    }
    assert forbidden.isdisjoint(namespace)
    assert not any(
        callable(value) and ("report" in name or "receipt" in name)
        for name, value in namespace.items()
    )
