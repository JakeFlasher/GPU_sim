from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOOLS_ROOT = _REPO_ROOT / "tools"
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from sol_baselines_harness.calibration.cache_prediction import (  # noqa: E402
    accuracy_diagnostic,
    end_to_end_smoke,
)
from sol_baselines_harness.calibration.cache_prediction.retirement import (  # noqa: E402
    LEGACY_NCU_CALIBRATION_RETIRED,
    LegacyCalibrationRetiredError,
)


def _measured_record(
    spec: end_to_end_smoke.SmokeCellSpec,
) -> end_to_end_smoke.CapturedSmokeRecord:
    return end_to_end_smoke.CapturedSmokeRecord(
        cell_name=spec.cell_name,
        predictor_bucket=spec.predictor_bucket,
        status="measured",
        ncu_l1_hit_rate=spec.ncu_l1_hit_rate,
        ncu_l2_hit_rate=spec.ncu_l2_hit_rate,
        predicted_l1_hit_rate=0.25,
        predicted_l2_hit_rate=0.50,
        l1_abs_pp_error=abs(0.25 - spec.ncu_l1_hit_rate) * 100.0,
        l2_abs_pp_error=abs(0.50 - spec.ncu_l2_hit_rate) * 100.0,
        l1_sectors_hit=25,
        l1_sectors_total=100,
        l2_sectors_hit=50,
        l2_sectors_total=100,
        launch_skip_count=spec.launch_skip_count,
        warmup_protocol="test",
        cache_initial_state="cold",
        measured_frame_uid=0,
        total_frame_count=1,
        predicted_l1_calibrated=None,
        predicted_l2_calibrated=None,
        l1_calibrated_abs_pp_error=None,
        l2_calibrated_abs_pp_error=None,
        l1_rel_err=None,
        l2_rel_err=None,
        tier2_trace_path=None,
        typed_reason=None,
        typed_message=None,
        sidecar_sha256=None,
        binary_sha256=spec.binary_sha256,
        kernel_signature_sha256=spec.kernel_signature_sha256,
    )


def test_same_cell_labels_cannot_influence_published_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvbit_so = tmp_path / "metric_extractor.so"
    nvbit_so.write_bytes(b"test metric extractor")

    loocv_labels: list[tuple[float, float]] = []
    forbidden_publication_side_effects: list[str] = []

    monkeypatch.setattr(
        end_to_end_smoke,
        "resolve_cache_geometry",
        lambda **_kwargs: ({}, None),
    )
    monkeypatch.setattr(
        end_to_end_smoke,
        "_measure_one_cell",
        lambda spec, **_kwargs: _measured_record(spec),
    )
    monkeypatch.setattr(
        end_to_end_smoke,
        "_git_head_sha",
        lambda _repo_root: "a" * 40,
    )

    def fake_run_loocv(calibration_inputs):
        calibration_input = tuple(calibration_inputs)[0]
        loocv_labels.append(
            (
                calibration_input.ncu_l1_label,
                calibration_input.ncu_l2_label,
            )
        )
        residual = SimpleNamespace(
            cell_name=calibration_input.cell_name,
            predicted_l1_calibrated=calibration_input.ncu_l1_label,
            predicted_l2_calibrated=calibration_input.ncu_l2_label,
            l1_residual_pp=0.0,
            l2_residual_pp=0.0,
            ncu_l1_label=calibration_input.ncu_l1_label,
            ncu_l2_label=calibration_input.ncu_l2_label,
        )
        return SimpleNamespace(
            per_cell_residuals=(residual,),
            admitted_cell_count=1,
            mean_abs_l1_pp_error=0.0,
            mean_abs_l2_pp_error=0.0,
            random_seed=0,
        )

    monkeypatch.setattr(end_to_end_smoke, "run_loocv", fake_run_loocv)
    monkeypatch.setattr(
        end_to_end_smoke,
        "evaluate_tiered_gate",
        lambda _result: SimpleNamespace(overall_passed=True),
    )
    monkeypatch.setattr(
        end_to_end_smoke,
        "_count_loocv_residuals_per_cohort",
        lambda _result: (
            end_to_end_smoke.COVERAGE_FLOOR_SMOKE,
            end_to_end_smoke.COVERAGE_FLOOR_SMOKE,
        ),
    )
    monkeypatch.setattr(
        end_to_end_smoke,
        "_gate_report_to_rows",
        lambda *_args, **_kwargs: (None, ()),
    )

    class ForbiddenCalibrator:
        def __init__(self) -> None:
            forbidden_publication_side_effects.append("calibrator_constructed")

    def forbidden_surface_write(*_args, **_kwargs):
        forbidden_publication_side_effects.append("surface_written")
        raise AssertionError("retired publication path performed a write")

    monkeypatch.setattr(
        end_to_end_smoke,
        "IsotonicCalibrator",
        ForbiddenCalibrator,
    )
    monkeypatch.setattr(
        end_to_end_smoke,
        "emit_cache_prediction_surface_entry",
        forbidden_surface_write,
    )

    def run_with_labels(
        l1_label: float,
        l2_label: float,
        output_name: str,
    ) -> end_to_end_smoke.EndToEndSmokeReport:
        spec = end_to_end_smoke.SmokeCellSpec(
            cell_name="problem_001_test",
            binary_path=tmp_path / "unused-binary",
            binary_sha256="b" * 64,
            launch_skip_count=0,
            predictor_bucket=end_to_end_smoke.PREDICTOR_BUCKET_FRIENDLY,
            ncu_l1_hit_rate=l1_label,
            ncu_l2_hit_rate=l2_label,
            ncu_l1_sectors_total=100,
            ncu_l1_sectors_hit=25,
            ncu_l2_sectors_total=100,
            ncu_l2_sectors_hit=50,
            arch="sm_86",
            ncu_metric_version="test",
            shape_point="test",
            source_commit="c" * 40,
            kernel_signature_sha256="d" * 64,
            cache_initial_state="cold",
        )
        return end_to_end_smoke.run_end_to_end_smoke(
            (spec,),
            out_dir=tmp_path / output_name,
            nvbit_so=nvbit_so,
        )

    first = run_with_labels(0.20, 0.40, "first")
    second = run_with_labels(0.80, 0.90, "second")

    # Held-out LOOCV diagnostics remain active and see the labels.
    assert loocv_labels == [(0.20, 0.40), (0.80, 0.90)]
    assert first.per_cell_records[0].predicted_l1_calibrated == 0.20
    assert second.per_cell_records[0].predicted_l1_calibrated == 0.80

    # Publication is label-independent because no calibrated scalar or
    # artifact is published at all: both runs emit the same typed outcome.
    first_surface = first.cache_prediction_surface_entries
    second_surface = second.cache_prediction_surface_entries
    assert first_surface == second_surface
    assert len(first_surface) == 1
    assert first_surface[0].status == "unavailable"
    assert first_surface[0].waiver_reason == LEGACY_NCU_CALIBRATION_RETIRED
    assert first_surface[0].artifact_path is None
    assert first_surface[0].bundles_written == 0
    assert forbidden_publication_side_effects == []
    assert not (tmp_path / "first" / "cache_prediction_bundles").exists()
    assert not (tmp_path / "second" / "cache_prediction_bundles").exists()


@pytest.mark.parametrize(
    ("option_name", "expected_surface"),
    (
        (
            "expanded_ncu_ledger_path",
            "accuracy_diagnostic.expanded_ncu_ledger_path",
        ),
        (
            "expanded_validation_report_path",
            "accuracy_diagnostic.expanded_validation_report_path",
        ),
    ),
)
def test_expanded_accuracy_inputs_fail_closed_before_side_effects(
    option_name: str,
    expected_surface: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_effects: list[str] = []

    def forbidden_git_lookup(*_args, **_kwargs):
        side_effects.append("git_lookup")
        raise AssertionError("guard ran after git lookup")

    def forbidden_partition_walk(*_args, **_kwargs):
        side_effects.append("partition_walk")
        raise AssertionError("guard ran after partition walk")

    monkeypatch.setattr(
        accuracy_diagnostic,
        "_resolve_implementation_commit_sha_from_git",
        forbidden_git_lookup,
    )
    monkeypatch.setattr(
        accuracy_diagnostic,
        "enumerate_canonical_solbench_problems",
        forbidden_partition_walk,
    )
    monkeypatch.setattr(Path, "is_file", forbidden_partition_walk)
    monkeypatch.setattr(Path, "read_bytes", forbidden_partition_walk)
    monkeypatch.setattr(Path, "read_text", forbidden_partition_walk)

    with pytest.raises(LegacyCalibrationRetiredError) as exc_info:
        accuracy_diagnostic.build_accuracy_diagnostic_report(
            repo_root=tmp_path,
            **{option_name: tmp_path / "label-selected-input.toml"},
        )

    assert exc_info.value.reason == LEGACY_NCU_CALIBRATION_RETIRED
    assert exc_info.value.surface == expected_surface
    assert side_effects == []


_DECORATED_RETIREMENT_INVENTORY = {
    ("calibration_sweep.py", "run_calibration_sweep"):
        "sol_baselines_harness.calibration.cache_prediction."
        "calibration_sweep.run_calibration_sweep",
    ("calibration_sweep.py", "serialize_ledger_to_toml"):
        "sol_baselines_harness.calibration.cache_prediction."
        "calibration_sweep.serialize_ledger_to_toml",
    ("end_to_end_smoke.py", "_emit_cache_prediction_surface_for_admitted_cohort"):
        "end_to_end_smoke.same_cell_calibrated_publication",
    ("harvest_warm_cold_slugs.py", "_run_legacy_workflow"):
        "sol_baselines_harness.calibration.cache_prediction."
        "harvest_warm_cold_slugs.main",
    ("run_ncu_harvest_batch.py", "run_batch_harvest"):
        "sol_baselines_harness.calibration.cache_prediction."
        "run_ncu_harvest_batch.run_batch_harvest",
    ("run_ncu_harvest_batch.py", "run_label_stability_rerun"):
        "sol_baselines_harness.calibration.cache_prediction."
        "run_ncu_harvest_batch.run_label_stability_rerun",
    ("sm120_l2_associativity_probe/sweep_associativity.py", "sweep_one_candidate"):
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_l2_associativity_probe.sweep_associativity.sweep_one_candidate",
    ("sm86_expanded/emit_expanded_ncu_ledger.py", "harvest_cell"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_ncu_ledger.harvest_cell",
    ("sm86_expanded/emit_expanded_ncu_ledger.py", "run_harvest"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_ncu_ledger.run_harvest",
    ("sm86_expanded/emit_expanded_ncu_ledger.py", "emit_ledger"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_ncu_ledger.emit_ledger",
    ("sm86_expanded/emit_expanded_raw_predictions.py", "harvest_raw_prediction"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_raw_predictions.harvest_raw_prediction",
    ("sm86_expanded/emit_expanded_raw_predictions.py", "run_raw_harvest"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_raw_predictions.run_raw_harvest",
    ("sm86_expanded/emit_expanded_raw_predictions.py", "emit_table"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_raw_predictions.emit_table",
    ("sm86_expanded/emit_expanded_validation_report.py", "emit"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
        "emit_expanded_validation_report.emit",
    ("sm86_geometry_sweep/harvest_sweep_traces.py", "harvest_cell"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
        "harvest_sweep_traces.harvest_cell",
    ("sm86_geometry_sweep/harvest_sweep_traces.py", "run_harvest"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
        "harvest_sweep_traces.run_harvest",
    ("sm86_geometry_sweep/run_sweep.py", "evaluate_assoc"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
        "run_sweep.evaluate_assoc",
    ("sm86_geometry_sweep/run_sweep.py", "emit_scoreboard"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
        "run_sweep.emit_scoreboard",
    ("sm86_geometry_sweep/run_sweep.py", "sweep"):
        "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
        "run_sweep.sweep",
    ("sm86_pilot/emit_variant_dispatch_snapshot.py", "main"):
        "sol_baselines_harness.calibration.cache_prediction."
        "sm86_pilot.emit_variant_dispatch_snapshot.main",
    ("sm86_pilot/persist_fitted_sidecar.py", "main"):
        "sol_baselines_harness.calibration.cache_prediction."
        "sm86_pilot.persist_fitted_sidecar.main",
    (
        "sm120_coverage_expansion/drive_pilot_predictor.py",
        "drive_pilot_predictor_cohort",
    ):
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_coverage_expansion.drive_pilot_predictor."
        "drive_pilot_predictor_cohort",
    (
        "sm120_coverage_expansion/drive_pilot_predictor.py",
        "emit_predictor_predictions",
    ):
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_coverage_expansion.drive_pilot_predictor."
        "emit_predictor_predictions",
}

_DIRECT_FIRST_REFUSAL_INVENTORY = {
    ("ncu_harvester.py", "_run_checked_ncu"),
    ("ncu_harvester.py", "collect_ncu_label_provenance"),
    ("ncu_harvester.py", "harvest_ncu_label_with_provenance"),
    ("ncu_harvester.py", "harvest_ncu_label"),
    ("label_smoke.py", "run_label_smoke"),
    ("ncu_capability_probe.py", "harvest_sm120_with_preflight"),
    ("sm120_coverage_expansion/ncu_repeatability.py", "run_pilot_ncu_repeatability"),
    ("sm120_coverage_expansion/ncu_repeatability.py", "emit_ncu_repeatability_status"),
    ("sm120_coverage_expansion/ncu_repeatability.py", "live_ncu_run_fn"),
    ("sm86_pilot/emit_ncu_3run_sidecars.py", "_harvest_3run_full"),
    ("sm86_pilot/emit_ncu_3run_sidecars.py", "_harvest_cell"),
    ("sm86_pilot/emit_ncu_3run_sidecars.py", "_write_sidecar"),
    ("sm86_pilot/emit_ncu_3run_sidecars.py", "_update_manifest"),
    ("sm86_pilot/emit_ncu_3run_sidecars.py", "_update_ledger"),
    ("sm86_pilot/emit_ncu_3run_sidecars.py", "main"),
    ("sm86_pilot/emit_pilot_ledger.py", "main"),
    ("sm86_pilot/emit_tier2_sidecars.py", "main"),
    ("accuracy_diagnostic.py", "predictor_callback_from_smoke_records"),
    ("accuracy_diagnostic.py", "_sm86_expanded_predictor_callback"),
    ("sm86_expanded/dense_reuse_l2_calibrator.py", "_fit_iso"),
    ("sm86_expanded/dense_reuse_l2_calibrator.py", "fit_segmented"),
    ("sm86_expanded/dense_reuse_l2_calibrator.py", "loocv_l2_abs_errors"),
    ("sm86_expanded/dense_reuse_l2_calibrator.py", "is_identity"),
    ("sm86_geometry_sweep/run_sweep.py", "select_winner"),
}

_CLI_PARSE_THEN_REFUSAL_INVENTORY = {
    (
        "sm120_coverage_expansion/emit_pilot_predictor_predictions.py",
        "main",
    ):
        "sol_baselines_harness.calibration.cache_prediction."
        "sm120_coverage_expansion.emit_pilot_predictor_predictions.main",
}


def _first_executable_statement(function: ast.FunctionDef) -> ast.stmt:
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    assert body, function.name
    return body[0]


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _is_refusal_call(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "refuse_legacy_ncu_calibration"
    )


def _is_parse_args_assignment(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "parse_args"
    )


def test_decorated_retirement_inventory_is_complete_and_self_guarded() -> None:
    calibration_root = Path(__file__).parent
    actual: dict[tuple[str, str], str] = {}
    for source in sorted(calibration_root.rglob("*.py")):
        relative = source.relative_to(calibration_root).as_posix()
        for function in _top_level_functions(source).values():
            decorators = [
                decorator
                for decorator in function.decorator_list
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "retired_legacy_ncu_calibration"
                )
            ]
            if not decorators:
                continue
            assert len(decorators) == 1
            decorator = decorators[0]
            surface = ast.literal_eval(decorator.args[0])
            first = _first_executable_statement(function)
            assert _is_refusal_call(first), (relative, function.name)
            body_surface = ast.literal_eval(first.value.args[0])  # type: ignore[union-attr]
            assert body_surface == surface
            actual[(relative, function.name)] = surface

    assert actual == _DECORATED_RETIREMENT_INVENTORY


def test_direct_retirement_inventory_refuses_as_first_statement() -> None:
    calibration_root = Path(__file__).parent
    for relative, function_name in sorted(_DIRECT_FIRST_REFUSAL_INVENTORY):
        function = _top_level_functions(calibration_root / relative)[function_name]
        assert _is_refusal_call(
            _first_executable_statement(function)
        ), (relative, function_name)


def test_retired_cli_refuses_immediately_after_help_safe_parse() -> None:
    calibration_root = Path(__file__).parent
    allowed_pre_refusal_calls = {"ArgumentParser", "add_argument", "parse_args"}
    for (relative, function_name), surface in sorted(
        _CLI_PARSE_THEN_REFUSAL_INVENTORY.items()
    ):
        function = _top_level_functions(calibration_root / relative)[function_name]
        parse_indices = [
            index
            for index, statement in enumerate(function.body)
            if _is_parse_args_assignment(statement)
        ]
        assert len(parse_indices) == 1, (relative, function_name)
        refusal_index = parse_indices[0] + 1
        refusal = function.body[refusal_index]
        assert _is_refusal_call(refusal), (relative, function_name)
        assert ast.literal_eval(refusal.value.args[0]) == surface  # type: ignore[union-attr]
        assert not any(
            isinstance(statement, (ast.Import, ast.ImportFrom))
            for statement in function.body[:refusal_index]
        )
        calls_before_refusal = [
            node
            for statement in function.body[:refusal_index]
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ]
        assert calls_before_refusal
        assert all(
            isinstance(call.func, ast.Attribute)
            and call.func.attr in allowed_pre_refusal_calls
            for call in calls_before_refusal
        )


def test_callback_injection_builders_guard_before_builder_work() -> None:
    calibration_root = Path(__file__).parent
    for relative, function_name in (
        ("accuracy_diagnostic.py", "_build_per_problem_row"),
        ("accuracy_diagnostic.py", "_build_cohort_cell_payloads"),
        ("accuracy_diagnostic.py", "build_accuracy_diagnostic_report"),
        ("intermediate_baseline.py", "emit_intermediate_baseline"),
    ):
        function = _top_level_functions(calibration_root / relative)[function_name]
        first = _first_executable_statement(function)
        assert isinstance(first, ast.If), (relative, function_name)
        assert isinstance(first.test, ast.Compare)
        assert isinstance(first.test.left, ast.Name)
        assert first.test.left.id == "predictor_callback"
        assert _is_refusal_call(first.body[0])


def test_retirement_decorator_does_not_use_wraps_bypass_metadata() -> None:
    retirement_path = Path(__file__).parent / "retirement.py"
    tree = ast.parse(retirement_path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "wraps" not in imported_names
    assert "update_wrapper" not in imported_names
