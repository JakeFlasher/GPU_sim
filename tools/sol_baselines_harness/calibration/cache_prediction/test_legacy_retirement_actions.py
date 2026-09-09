"""Focused retirement guards for legacy SOL NCU-calibration actions."""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOOLS_ROOT = _REPO_ROOT / "tools"
_REASON = "legacy_ncu_calibration_retired_targets_not_features"

_CALIBRATION_SWEEP = (
    "sol_baselines_harness.calibration.cache_prediction.calibration_sweep"
)
_HARVEST_BATCH = (
    "sol_baselines_harness.calibration.cache_prediction.run_ncu_harvest_batch"
)
_WARM_COLD = (
    "sol_baselines_harness.calibration.cache_prediction.harvest_warm_cold_slugs"
)
_EXPANDED_NCU = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
    "emit_expanded_ncu_ledger"
)
_EXPANDED_RAW = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
    "emit_expanded_raw_predictions"
)
_EXPANDED_REPORT = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
    "emit_expanded_validation_report"
)
_GEOMETRY_HARVEST = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
    "harvest_sweep_traces"
)
_GEOMETRY_SWEEP = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_geometry_sweep."
    "run_sweep"
)
_END_TO_END = (
    "sol_baselines_harness.calibration.cache_prediction.end_to_end_smoke"
)
_SM120_ASSOC_SWEEP = (
    "sol_baselines_harness.calibration.cache_prediction."
    "sm120_l2_associativity_probe.sweep_associativity"
)
_SM120_ASSOC_FIT = (
    "sol_baselines_harness.calibration.cache_prediction."
    "sm120_l2_associativity_probe.persist_fitted_sidecar"
)
_VARIANT_SNAPSHOT = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_pilot."
    "emit_variant_dispatch_snapshot"
)
_PILOT_FIT = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_pilot."
    "persist_fitted_sidecar"
)
_NCU_HARVESTER = (
    "sol_baselines_harness.calibration.cache_prediction.ncu_harvester"
)
_LABEL_SMOKE = (
    "sol_baselines_harness.calibration.cache_prediction.label_smoke"
)
_NCU_CAPABILITY = (
    "sol_baselines_harness.calibration.cache_prediction.ncu_capability_probe"
)
_REPEATABILITY = (
    "sol_baselines_harness.calibration.cache_prediction."
    "sm120_coverage_expansion.ncu_repeatability"
)
_REPEATABILITY_CLI = (
    "sol_baselines_harness.calibration.cache_prediction."
    "sm120_coverage_expansion.emit_pilot_ncu_repeatability"
)
_PILOT_PREDICTOR = (
    "sol_baselines_harness.calibration.cache_prediction."
    "sm120_coverage_expansion.drive_pilot_predictor"
)
_PILOT_PREDICTOR_CLI = (
    "sol_baselines_harness.calibration.cache_prediction."
    "sm120_coverage_expansion.emit_pilot_predictor_predictions"
)
_NCU_3RUN = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_pilot."
    "emit_ncu_3run_sidecars"
)
_PILOT_LEDGER = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_pilot."
    "emit_pilot_ledger"
)
_TIER2_SIDECARS = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_pilot."
    "emit_tier2_sidecars"
)
_ACCURACY_DIAGNOSTIC = (
    "sol_baselines_harness.calibration.cache_prediction.accuracy_diagnostic"
)
_INTERMEDIATE_BASELINE = (
    "sol_baselines_harness.calibration.cache_prediction.intermediate_baseline"
)
_DENSE_REUSE_CALIBRATOR = (
    "sol_baselines_harness.calibration.cache_prediction.sm86_expanded."
    "dense_reuse_l2_calibrator"
)

Action = Callable[[object, Path], object]


def _run_calibration_sweep(module: object, root: Path) -> object:
    return module.run_calibration_sweep(  # type: ignore[attr-defined]
        [],
        ncu_metric_version="retired",
        host_arch="sm_86",
        host_descriptor_hash="retired",
        ncu_version="retired",
        driver_version="retired",
    )


def _serialize_ledger(module: object, root: Path) -> object:
    return module.serialize_ledger_to_toml(  # type: ignore[attr-defined]
        object(), root / "must-not-be-written.toml"
    )


def _run_batch_harvest(module: object, root: Path) -> object:
    return module.run_batch_harvest(  # type: ignore[attr-defined]
        repo_root=root,
        ncu_metric_version="retired",
        host_arch="sm_86",
        host_descriptor_hash="retired",
        ncu_version="retired",
        driver_version="retired",
        output_ledger_path=root / "must-not-be-written.toml",
    )


def _run_stability_rerun(module: object, root: Path) -> object:
    return module.run_label_stability_rerun(  # type: ignore[attr-defined]
        repo_root=root,
        existing_ledger_path=root / "ledger.toml",
        ncu_metric_version="retired",
        host_arch="sm_86",
        host_descriptor_hash="retired",
        ncu_version="retired",
        driver_version="retired",
    )


def _warm_cold_main(module: object, root: Path) -> object:
    return module.main([])  # type: ignore[attr-defined]


def _harvest_ncu_cell(module: object, root: Path) -> object:
    return module.harvest_cell("cell", root / "probe")  # type: ignore[attr-defined]


def _run_harvest(module: object, root: Path) -> object:
    return module.run_harvest(root)  # type: ignore[attr-defined]


def _emit_ledger(module: object, root: Path) -> object:
    return module.emit_ledger(root)  # type: ignore[attr-defined]


def _harvest_raw_cell(module: object, root: Path) -> object:
    return module.harvest_raw_prediction(  # type: ignore[attr-defined]
        "cell", root / "probe", root / "work"
    )


def _run_raw_harvest(module: object, root: Path) -> object:
    return module.run_raw_harvest(root)  # type: ignore[attr-defined]


def _emit_table(module: object, root: Path) -> object:
    return module.emit_table(root)  # type: ignore[attr-defined]


def _harvest_geometry_cell(module: object, root: Path) -> object:
    return module.harvest_cell(  # type: ignore[attr-defined]
        "cell", root / "probe", root / "work"
    )


def _evaluate_assoc(module: object, root: Path) -> object:
    return module.evaluate_assoc(root, 16)  # type: ignore[attr-defined]


def _emit_scoreboard(module: object, root: Path) -> object:
    return module.emit_scoreboard(root)  # type: ignore[attr-defined]


def _sweep(module: object, root: Path) -> object:
    return module.sweep(root)  # type: ignore[attr-defined]


def _emit_report(module: object, root: Path) -> object:
    return module.emit(root)  # type: ignore[attr-defined]


def _emit_same_cell_surface(module: object, root: Path) -> object:
    return module._emit_cache_prediction_surface_for_admitted_cohort(  # type: ignore[attr-defined]
        (),
        {},
        (),
        SimpleNamespace(per_cell_residuals=()),
        bundle_dir=root / "bundles",
        out_dir=root,
    )


def _sweep_sm120_candidate(module: object, root: Path) -> object:
    return module.sweep_one_candidate(  # type: ignore[attr-defined]
        16,
        cell_traces={},
        ledger_entries={},
        split_by_slug={},
    )


def _emit_variant_snapshot(module: object, root: Path) -> object:
    return module.main(snapshot_path=root / "snapshot.toml")  # type: ignore[attr-defined]


def _persist_pilot_fit(module: object, root: Path) -> object:
    return module.main(output_root=root)  # type: ignore[attr-defined]


def _drive_pilot_predictor(module: object, root: Path) -> object:
    return module.drive_pilot_predictor_cohort(  # type: ignore[attr-defined]
        [],
        ncu_status_path=root / "must-not-read-ncu-status.toml",
        tier2_status_path=root / "must-not-read-tier2-status.toml",
        harvest_output_root=root / "must-not-read-harvest",
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def _emit_pilot_predictor(module: object, root: Path) -> object:
    return module.emit_predictor_predictions(  # type: ignore[attr-defined]
        [],
        output=root / "must-not-write-predictor-predictions.toml",
        captured_at_utc="2026-01-01T00:00:00+00:00",
        ncu_status_path=root / "must-not-read-ncu-status.toml",
        tier2_status_path=root / "must-not-read-tier2-status.toml",
        harvest_output_root=root / "must-not-read-harvest",
    )


_ACTIONS: tuple[tuple[str, str, Action], ...] = (
    (
        _CALIBRATION_SWEEP,
        f"{_CALIBRATION_SWEEP}.run_calibration_sweep",
        _run_calibration_sweep,
    ),
    (
        _CALIBRATION_SWEEP,
        f"{_CALIBRATION_SWEEP}.serialize_ledger_to_toml",
        _serialize_ledger,
    ),
    (
        _HARVEST_BATCH,
        f"{_HARVEST_BATCH}.run_batch_harvest",
        _run_batch_harvest,
    ),
    (
        _HARVEST_BATCH,
        f"{_HARVEST_BATCH}.run_label_stability_rerun",
        _run_stability_rerun,
    ),
    (_WARM_COLD, f"{_WARM_COLD}.main", _warm_cold_main),
    (_EXPANDED_NCU, f"{_EXPANDED_NCU}.harvest_cell", _harvest_ncu_cell),
    (_EXPANDED_NCU, f"{_EXPANDED_NCU}.run_harvest", _run_harvest),
    (_EXPANDED_NCU, f"{_EXPANDED_NCU}.emit_ledger", _emit_ledger),
    (
        _EXPANDED_RAW,
        f"{_EXPANDED_RAW}.harvest_raw_prediction",
        _harvest_raw_cell,
    ),
    (_EXPANDED_RAW, f"{_EXPANDED_RAW}.run_raw_harvest", _run_raw_harvest),
    (_EXPANDED_RAW, f"{_EXPANDED_RAW}.emit_table", _emit_table),
    (
        _GEOMETRY_HARVEST,
        f"{_GEOMETRY_HARVEST}.harvest_cell",
        _harvest_geometry_cell,
    ),
    (_GEOMETRY_HARVEST, f"{_GEOMETRY_HARVEST}.run_harvest", _run_harvest),
    (_GEOMETRY_SWEEP, f"{_GEOMETRY_SWEEP}.evaluate_assoc", _evaluate_assoc),
    (_GEOMETRY_SWEEP, f"{_GEOMETRY_SWEEP}.emit_scoreboard", _emit_scoreboard),
    (_GEOMETRY_SWEEP, f"{_GEOMETRY_SWEEP}.sweep", _sweep),
    (_EXPANDED_REPORT, f"{_EXPANDED_REPORT}.emit", _emit_report),
    (
        _END_TO_END,
        "end_to_end_smoke.same_cell_calibrated_publication",
        _emit_same_cell_surface,
    ),
    (
        _SM120_ASSOC_SWEEP,
        f"{_SM120_ASSOC_SWEEP}.sweep_one_candidate",
        _sweep_sm120_candidate,
    ),
    (
        _VARIANT_SNAPSHOT,
        f"{_VARIANT_SNAPSHOT}.main",
        _emit_variant_snapshot,
    ),
    (_PILOT_FIT, f"{_PILOT_FIT}.main", _persist_pilot_fit),
    (
        _PILOT_PREDICTOR,
        f"{_PILOT_PREDICTOR}.drive_pilot_predictor_cohort",
        _drive_pilot_predictor,
    ),
    (
        _PILOT_PREDICTOR,
        f"{_PILOT_PREDICTOR}.emit_predictor_predictions",
        _emit_pilot_predictor,
    ),
)


@pytest.mark.parametrize(
    ("module_name", "surface", "invoke"),
    _ACTIONS,
    ids=[surface.rsplit(".", 1)[-1] for _, surface, _ in _ACTIONS],
)
def test_legacy_actions_refuse_before_side_effects(
    module_name: str,
    surface: str,
    invoke: Action,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(module_name)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    calibrator = importlib.import_module(
        "sol_baselines_harness.metrics.isotonic_calibrator"
    ).IsotonicCalibrator

    side_effects: list[str] = []

    def forbidden_side_effect(*args: object, **kwargs: object) -> None:
        side_effects.append("called")
        raise AssertionError("legacy action reached a side-effect boundary")

    monkeypatch.setattr(subprocess, "run", forbidden_side_effect)
    monkeypatch.setattr(Path, "mkdir", forbidden_side_effect)
    monkeypatch.setattr(Path, "read_bytes", forbidden_side_effect)
    monkeypatch.setattr(Path, "read_text", forbidden_side_effect)
    monkeypatch.setattr(Path, "write_text", forbidden_side_effect)
    monkeypatch.setattr(Path, "write_bytes", forbidden_side_effect)
    monkeypatch.setattr(Path, "unlink", forbidden_side_effect)

    for name in (
        "_run_checked_ncu",
        "harvest_ncu_label_with_provenance",
        "predict_from_launch_frame",
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden_side_effect)

    monkeypatch.setattr(calibrator, "fit", forbidden_side_effect)

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        invoke(module, tmp_path)

    assert caught.value.reason == _REASON
    assert caught.value.surface == surface
    assert str(caught.value) == f"{_REASON}: {surface}"
    assert side_effects == []
    assert list(tmp_path.iterdir()) == []


_DECORATED_FUNCTIONS: tuple[tuple[str, str], ...] = (
    (_CALIBRATION_SWEEP, "run_calibration_sweep"),
    (_CALIBRATION_SWEEP, "serialize_ledger_to_toml"),
    (_HARVEST_BATCH, "run_batch_harvest"),
    (_HARVEST_BATCH, "run_label_stability_rerun"),
    (_WARM_COLD, "_run_legacy_workflow"),
    (_EXPANDED_NCU, "harvest_cell"),
    (_EXPANDED_NCU, "run_harvest"),
    (_EXPANDED_NCU, "emit_ledger"),
    (_EXPANDED_RAW, "harvest_raw_prediction"),
    (_EXPANDED_RAW, "run_raw_harvest"),
    (_EXPANDED_RAW, "emit_table"),
    (_GEOMETRY_HARVEST, "harvest_cell"),
    (_GEOMETRY_HARVEST, "run_harvest"),
    (_GEOMETRY_SWEEP, "evaluate_assoc"),
    (_GEOMETRY_SWEEP, "emit_scoreboard"),
    (_GEOMETRY_SWEEP, "sweep"),
    (_EXPANDED_REPORT, "emit"),
    (_END_TO_END, "_emit_cache_prediction_surface_for_admitted_cohort"),
    (_SM120_ASSOC_SWEEP, "sweep_one_candidate"),
    (_VARIANT_SNAPSHOT, "main"),
    (_PILOT_FIT, "main"),
    (_PILOT_PREDICTOR, "drive_pilot_predictor_cohort"),
    (_PILOT_PREDICTOR, "emit_predictor_predictions"),
)


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    _DECORATED_FUNCTIONS,
    ids=[f"{module.rsplit('.', 1)[-1]}.{name}" for module, name in _DECORATED_FUNCTIONS],
)
def test_retired_decorators_expose_no_wrapped_body(
    module_name: str,
    function_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    function = getattr(importlib.import_module(module_name), function_name)

    assert not hasattr(function, "__wrapped__")
    assert inspect.unwrap(function) is function
    assert function.__name__ == function_name
    assert inspect.signature(function)


DirectAction = Callable[[object, Path, Callable[..., object]], object]


def _repeatability_cli_main(
    module: object,
    root: Path,
    forbidden: Callable[..., object],
) -> object:
    del root, forbidden
    previous_argv = sys.argv
    sys.argv = [
        "emit_pilot_ncu_repeatability.py",
        "--captured-at-utc",
        "2026-01-01T00:00:00+00:00",
    ]
    try:
        return module.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = previous_argv


def _pilot_predictor_cli_main(
    module: object,
    root: Path,
    forbidden: Callable[..., object],
) -> object:
    del root, forbidden
    previous_argv = sys.argv
    sys.argv = [
        "emit_pilot_predictor_predictions.py",
        "--captured-at-utc",
        "2026-01-01T00:00:00+00:00",
    ]
    try:
        return module.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = previous_argv


def _sm120_assoc_sweep_cli_main(
    module: object,
    root: Path,
    forbidden: Callable[..., object],
) -> object:
    del forbidden
    previous_argv = sys.argv
    sys.argv = [
        "sweep_associativity.py",
        "--traces-root",
        str(root / "must-not-read-traces"),
        "--output-dir",
        str(root / "must-not-write-scoreboard"),
    ]
    try:
        return module.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = previous_argv


def _sm120_assoc_fit_cli_main(
    module: object,
    root: Path,
    forbidden: Callable[..., object],
) -> object:
    del forbidden
    previous_argv = sys.argv
    sys.argv = [
        "persist_fitted_sidecar.py",
        "--traces-root",
        str(root / "must-not-read-traces"),
    ]
    try:
        return module.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = previous_argv


def _l2_cells(module: object) -> list[object]:
    return [
        module.L2Cell("a", "predictor_friendly", 1.0, 0.1, 0.2),  # type: ignore[attr-defined]
        module.L2Cell("b", "predictor_hostile", 5.0, 0.4, 0.8),  # type: ignore[attr-defined]
    ]


_DIRECT_ACTIONS: tuple[tuple[str, str, str, DirectAction], ...] = (
    (
        _NCU_HARVESTER,
        "_run_checked_ncu",
        f"{_NCU_HARVESTER}._run_checked_ncu",
        lambda _m, _r, _f: _m._run_checked_ncu(("ncu",)),
    ),
    (
        _NCU_HARVESTER,
        "collect_ncu_label_provenance",
        f"{_NCU_HARVESTER}.collect_ncu_label_provenance",
        lambda _m, _r, _f: _m.collect_ncu_label_provenance(("ncu",), "csv"),
    ),
    (
        _NCU_HARVESTER,
        "harvest_ncu_label_with_provenance",
        f"{_NCU_HARVESTER}.harvest_ncu_label_with_provenance",
        lambda _m, _r, _f: _m.harvest_ncu_label_with_provenance(
            _r / "binary", launch_skip_count=1, _runner=_f,
        ),
    ),
    (
        _NCU_HARVESTER,
        "harvest_ncu_label",
        f"{_NCU_HARVESTER}.harvest_ncu_label",
        lambda _m, _r, _f: _m.harvest_ncu_label(
            _r / "binary", launch_skip_count=1, _runner=_f,
        ),
    ),
    (
        _LABEL_SMOKE,
        "run_label_smoke",
        f"{_LABEL_SMOKE}.run_label_smoke",
        lambda _m, _r, _f: _m.run_label_smoke(
            (), output_path=_r / "smoke.json", _runner=_f,
        ),
    ),
    (
        _NCU_CAPABILITY,
        "harvest_sm120_with_preflight",
        f"{_NCU_CAPABILITY}.harvest_sm120_with_preflight",
        lambda _m, _r, _f: _m.harvest_sm120_with_preflight(
            _r / "binary", launch_skip_count=1, probe_path=_r / "probe.toml",
        ),
    ),
    (
        _REPEATABILITY,
        "run_pilot_ncu_repeatability",
        f"{_REPEATABILITY}.run_pilot_ncu_repeatability",
        lambda _m, _r, _f: _m.run_pilot_ncu_repeatability(
            [("slug", "cuda_cpp")], run_fn=_f,
        ),
    ),
    (
        _REPEATABILITY,
        "emit_ncu_repeatability_status",
        f"{_REPEATABILITY}.emit_ncu_repeatability_status",
        lambda _m, _r, _f: _m.emit_ncu_repeatability_status(
            [("slug", "cuda_cpp")],
            output=_r / "repeatability.toml",
            captured_at_utc="2026-01-01T00:00:00+00:00",
            run_fn=_f,
        ),
    ),
    (
        _REPEATABILITY,
        "live_ncu_run_fn",
        f"{_REPEATABILITY}.live_ncu_run_fn",
        lambda _m, _r, _f: _m.live_ncu_run_fn(
            slug="slug", language="cuda_cpp",
        ),
    ),
    (
        _REPEATABILITY_CLI,
        "main",
        f"{_REPEATABILITY_CLI}.main",
        _repeatability_cli_main,
    ),
    (
        _PILOT_PREDICTOR_CLI,
        "main",
        f"{_PILOT_PREDICTOR_CLI}.main",
        _pilot_predictor_cli_main,
    ),
    (
        _SM120_ASSOC_SWEEP,
        "main",
        f"{_SM120_ASSOC_SWEEP}.main",
        _sm120_assoc_sweep_cli_main,
    ),
    (
        _SM120_ASSOC_FIT,
        "main",
        f"{_SM120_ASSOC_FIT}.main",
        _sm120_assoc_fit_cli_main,
    ),
    (
        _NCU_3RUN,
        "_harvest_3run_full",
        f"{_NCU_3RUN}._harvest_3run_full",
        lambda _m, _r, _f: _m._harvest_3run_full("slug", "cuda_cpp"),
    ),
    (
        _NCU_3RUN,
        "_harvest_cell",
        f"{_NCU_3RUN}._harvest_cell",
        lambda _m, _r, _f: _m._harvest_cell("slug"),
    ),
    (
        _NCU_3RUN,
        "_write_sidecar",
        f"{_NCU_3RUN}._write_sidecar",
        lambda _m, _r, _f: _m._write_sidecar("slug", {}),
    ),
    (
        _NCU_3RUN,
        "_update_manifest",
        f"{_NCU_3RUN}._update_manifest",
        lambda _m, _r, _f: _m._update_manifest("slug", "0" * 64, {}),
    ),
    (
        _NCU_3RUN,
        "_update_ledger",
        f"{_NCU_3RUN}._update_ledger",
        lambda _m, _r, _f: _m._update_ledger({}),
    ),
    (
        _NCU_3RUN,
        "main",
        f"{_NCU_3RUN}.main",
        lambda _m, _r, _f: _m.main(("slug",)),
    ),
    (
        _PILOT_LEDGER,
        "main",
        f"{_PILOT_LEDGER}.main",
        lambda _m, _r, _f: _m.main(output_root=_r / "pilot-ledger"),
    ),
    (
        _TIER2_SIDECARS,
        "main",
        f"{_TIER2_SIDECARS}.main",
        lambda _m, _r, _f: _m.main(("slug",)),
    ),
    (
        _ACCURACY_DIAGNOSTIC,
        "predictor_callback_from_smoke_records",
        "accuracy_diagnostic.predictor_callback_from_smoke_records",
        lambda _m, _r, _f: _m.predictor_callback_from_smoke_records(()),
    ),
    (
        _ACCURACY_DIAGNOSTIC,
        "_sm86_expanded_predictor_callback",
        "accuracy_diagnostic._sm86_expanded_predictor_callback",
        lambda _m, _r, _f: _m._sm86_expanded_predictor_callback(
            _r / "expanded.toml", _f,
        ),
    ),
    (
        _ACCURACY_DIAGNOSTIC,
        "_build_per_problem_row",
        "accuracy_diagnostic._build_per_problem_row.predictor_callback",
        lambda _m, _r, _f: _m._build_per_problem_row(
            object(),
            baselines_root=_r,
            ledger_data={},
            denominator_floor=0.01,
            predictor_callback=_f,
        ),
    ),
    (
        _ACCURACY_DIAGNOSTIC,
        "_build_cohort_cell_payloads",
        "accuracy_diagnostic._build_cohort_cell_payloads.predictor_callback",
        lambda _m, _r, _f: _m._build_cohort_cell_payloads(
            ledger_data={},
            denominator_floor=0.01,
            predictor_callback=_f,
        ),
    ),
    (
        _ACCURACY_DIAGNOSTIC,
        "build_accuracy_diagnostic_report",
        "accuracy_diagnostic.predictor_callback",
        lambda _m, _r, _f: _m.build_accuracy_diagnostic_report(
            repo_root=_r, predictor_callback=_f,
        ),
    ),
    (
        _INTERMEDIATE_BASELINE,
        "emit_intermediate_baseline",
        "intermediate_baseline.predictor_callback",
        lambda _m, _r, _f: _m.emit_intermediate_baseline(
            _r / "baseline.toml", predictor_callback=_f,
        ),
    ),
    (
        _DENSE_REUSE_CALIBRATOR,
        "_fit_iso",
        "sm86_expanded.dense_reuse_l2_calibrator._fit_iso",
        lambda _m, _r, _f: _m._fit_iso([1.0, 2.0], [0.1, 0.2]),
    ),
    (
        _DENSE_REUSE_CALIBRATOR,
        "fit_segmented",
        "sm86_expanded.dense_reuse_l2_calibrator.fit_segmented",
        lambda _m, _r, _f: _m.fit_segmented(_l2_cells(_m)),
    ),
    (
        _DENSE_REUSE_CALIBRATOR,
        "loocv_l2_abs_errors",
        "sm86_expanded.dense_reuse_l2_calibrator.loocv_l2_abs_errors",
        lambda _m, _r, _f: _m.loocv_l2_abs_errors(_l2_cells(_m)),
    ),
    (
        _DENSE_REUSE_CALIBRATOR,
        "is_identity",
        "sm86_expanded.dense_reuse_l2_calibrator.is_identity",
        lambda _m, _r, _f: _m.is_identity(_l2_cells(_m)),
    ),
    (
        _GEOMETRY_SWEEP,
        "select_winner",
        "sm86_geometry_sweep.run_sweep.select_winner",
        lambda _m, _r, _f: _m.select_winner([
            {
                "all_gates_pass": True,
                "friendly_l1_mape": 0.0,
                "friendly_l2_mape": 0.0,
                "hostile_l1_mape": 0.0,
                "hostile_l2_mape": 0.0,
                "associativity_l2": 16,
            }
        ]),
    ),
)


@pytest.mark.parametrize(
    ("module_name", "target_name", "surface", "invoke"),
    _DIRECT_ACTIONS,
    ids=[surface.rsplit(".", 1)[-1] for _, _, surface, _ in _DIRECT_ACTIONS],
)
def test_direct_legacy_actions_refuse_before_injected_side_effects(
    module_name: str,
    target_name: str,
    surface: str,
    invoke: DirectAction,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(module_name)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    project_calibrator = importlib.import_module(
        "sol_baselines_harness.metrics.isotonic_calibrator"
    ).IsotonicCalibrator
    sklearn_calibrator = importlib.import_module(
        "sklearn.isotonic"
    ).IsotonicRegression
    side_effects: list[str] = []

    def forbidden_side_effect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("retirement guard ran after a side effect")

    target = getattr(module, target_name)
    for name in (
        "_build_invocation",
        "_probe_ncu_cli_path",
        "_probe_ncu_cli_sha256",
        "_probe_ncu_version_stdout",
        "_probe_nvidia_smi_device_info",
        "preflight_for_arch",
        "harvest_ncu_label",
        "harvest_ncu_label_with_provenance",
        "run_pilot_ncu_repeatability",
        "emit_ncu_repeatability_status",
        "live_ncu_run_fn",
        "_harvest_3run_full",
        "_harvest_cell",
        "_write_sidecar",
        "_update_manifest",
        "_update_ledger",
        "predictor_callback_from_smoke_records",
        "_sm86_expanded_predictor_callback",
        "_build_per_problem_row",
        "_build_cohort_cell_payloads",
        "build_accuracy_diagnostic_report",
        "_fit_iso",
        "fit_segmented",
    ):
        if name != target_name and hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden_side_effect)

    monkeypatch.setattr(subprocess, "run", forbidden_side_effect)
    for method in (
        "mkdir",
        "read_bytes",
        "read_text",
        "unlink",
        "write_bytes",
        "write_text",
    ):
        monkeypatch.setattr(Path, method, forbidden_side_effect)

    monkeypatch.setattr(project_calibrator, "fit", forbidden_side_effect)
    monkeypatch.setattr(sklearn_calibrator, "fit", forbidden_side_effect)

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        invoke(module, tmp_path, forbidden_side_effect)

    assert caught.value.reason == _REASON
    assert caught.value.surface == surface
    assert side_effects == []
    assert list(tmp_path.iterdir()) == []
    assert getattr(module, target_name) is target


def test_read_only_and_trace_only_helpers_remain_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    sweep_module = importlib.import_module(_GEOMETRY_SWEEP)
    trace_module = importlib.import_module(_GEOMETRY_HARVEST)
    calibration_module = importlib.import_module(_CALIBRATION_SWEEP)
    harvester_module = importlib.import_module(_NCU_HARVESTER)
    capability_module = importlib.import_module(_NCU_CAPABILITY)
    repeatability_module = importlib.import_module(_REPEATABILITY)
    expanded_report_module = importlib.import_module(_EXPANDED_REPORT)
    pilot_ledger_module = importlib.import_module(_PILOT_LEDGER)

    assert not hasattr(calibration_module.load_ledger_from_toml, "__wrapped__")
    assert not hasattr(sweep_module.build_records_at_assoc, "__wrapped__")
    assert not hasattr(trace_module.dense_reuse_feature, "__wrapped__")

    frame = SimpleNamespace(
        records=(
            SimpleNamespace(sector_ids=(1, 2, 1)),
            SimpleNamespace(sector_ids=(2, 3)),
        ),
        truncation=None,
    )
    assert trace_module.dense_reuse_feature(frame) == {
        "distinct_sectors": 3,
        "total_sector_accesses": 5,
        "reuse_factor": 5 / 3,
        "truncated": False,
    }
    argv = harvester_module._build_invocation(
        "counter-free-binary", launch_skip_count=1,
    )
    assert argv[0] == "ncu"
    assert argv[-1] == "counter-free-binary"
    assert harvester_module.validate_ncu_label_provenance({
        "ncu_cli_path": "/counter-free/ncu",
        "ncu_cli_sha256": "0" * 64,
        "ncu_version_stdout": "counter-free",
        "driver_version": "counter-free",
        "gpu_device_name": "counter-free",
        "compute_capability": "counter-free",
        "clock_policy_observed": "none",
        "command_argv": ("ncu", "--help"),
        "raw_report_sha256": "1" * 64,
    }).clock_policy_observed == "none"
    probe = capability_module.load_capability_probe(
        _REPO_ROOT
        / "tools" / "sol_baselines_harness"
        / "calibration" / "cache_prediction"
        / "sm120_ncu_capability_probe.toml"
    )
    assert probe.host_arch == "sm_120"
    assert "[meta]" in repeatability_module.render_ncu_repeatability_status_toml(
        captured_at_utc="2026-01-01T00:00:00+00:00",
        rows=[],
    )
    assert expanded_report_module.winning_associativity(_REPO_ROOT) == 4
    assert "[meta]" in pilot_ledger_module._serialize_ledger({
        "meta": {"schema_version": 1},
        "entries": [],
    })
    assert "[manifest_meta]" in pilot_ledger_module._serialize_split({
        "manifest_meta": {"schema_version": 1},
        "splits": [],
    })


def test_canonical_smoke_report_refuses_before_mkdir_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(_END_TO_END)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    report_path = (
        _REPO_ROOT
        / "tools" / "sol_baselines_harness"
        / "calibration" / "cache_prediction"
        / "end_to_end_smoke_report.toml"
    )
    retained = report_path.read_bytes()
    side_effects: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("frozen report reached mkdir/write")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        module.dump_smoke_report_toml(object(), report_path)

    assert caught.value.reason == _REASON
    assert caught.value.surface == (
        "end_to_end_smoke.dump_smoke_report_toml.frozen_output"
    )
    assert side_effects == []
    assert report_path.read_bytes() == retained


@pytest.mark.parametrize("symlink_parent", (False, True), ids=("file", "parent"))
def test_smoke_report_symlink_routes_refuse_without_writes(
    symlink_parent: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(_END_TO_END)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    retained_path = tmp_path / "retained" / "report.toml"
    retained_path.parent.mkdir()
    retained_path.write_bytes(b"frozen report sentinel\n")
    if symlink_parent:
        output_parent = tmp_path / "output-link"
        output_parent.symlink_to(retained_path.parent, target_is_directory=True)
        output_path = output_parent / "new-report.toml"
    else:
        output_path = tmp_path / "report-link.toml"
        output_path.symlink_to(retained_path)

    side_effects: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("symlink route reached mkdir/write")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        module.dump_smoke_report_toml(object(), output_path)

    assert caught.value.reason == _REASON
    assert side_effects == []
    assert retained_path.read_bytes() == b"frozen report sentinel\n"


def test_manifest_bound_smoke_scratch_refuses_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(_END_TO_END)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    side_effects: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("manifest-bound scratch reached capture/write")

    monkeypatch.setattr(module, "_measure_one_cell", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        module.run_end_to_end_smoke(
            (),
            out_dir=(
                Path("/home/jakeshea/.solbench_round15_tmp")
                / "solbench_end_to_end_smoke"
            ),
        )

    assert caught.value.reason == _REASON
    assert caught.value.surface == (
        "end_to_end_smoke.run_end_to_end_smoke.frozen_output"
    )
    assert side_effects == []


def test_symlinked_smoke_scratch_refuses_before_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(_END_TO_END)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    output_link = tmp_path / "smoke-scratch-link"
    output_link.symlink_to(
        Path.home() / ".solbench_round15_tmp" / "solbench_end_to_end_smoke",
        target_is_directory=True,
    )
    side_effects: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("symlinked scratch reached capture/write")

    monkeypatch.setattr(module, "_measure_one_cell", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        module.run_end_to_end_smoke((), out_dir=output_link)

    assert caught.value.reason == _REASON
    assert caught.value.surface == (
        "end_to_end_smoke.run_end_to_end_smoke.frozen_output"
    )
    assert side_effects == []


def test_smoke_cli_rejects_frozen_report_before_ledger_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(_TOOLS_ROOT))
    module = importlib.import_module(_END_TO_END)
    retirement = importlib.import_module(
        "sol_baselines_harness.calibration.cache_prediction.retirement"
    )
    canonical_report = (
        _REPO_ROOT
        / "tools" / "sol_baselines_harness"
        / "calibration" / "cache_prediction"
        / "end_to_end_smoke_report.toml"
    )
    side_effects: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        side_effects.append("called")
        raise AssertionError("CLI read or wrote before frozen-output refusal")

    monkeypatch.setattr(module, "load_ledger_from_toml", forbidden)
    monkeypatch.setattr(module, "run_end_to_end_smoke", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "end_to_end_smoke.py",
            "--output",
            str(canonical_report),
            "--scratch-dir",
            str(tmp_path / "fresh-scratch"),
        ],
    )

    with pytest.raises(retirement.LegacyCalibrationRetiredError) as caught:
        module._main_cli()

    assert caught.value.reason == _REASON
    assert caught.value.surface == "end_to_end_smoke._main_cli.report_output"
    assert side_effects == []


_HELP_SCRIPTS = (
    Path("tools/sol_baselines_harness/calibration/cache_prediction/")
    / "end_to_end_smoke.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/")
    / "harvest_warm_cold_slugs.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_expanded/")
    / "emit_expanded_ncu_ledger.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_expanded/")
    / "emit_expanded_raw_predictions.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_expanded/")
    / "emit_expanded_validation_report.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_geometry_sweep/")
    / "harvest_sweep_traces.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_geometry_sweep/")
    / "run_sweep.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot/")
    / "emit_ncu_3run_sidecars.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot/")
    / "emit_pilot_ledger.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot/")
    / "emit_tier2_sidecars.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot/")
    / "emit_variant_dispatch_snapshot.py",
    Path("tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot/")
    / "persist_fitted_sidecar.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_coverage_expansion/"
    )
    / "emit_diagnostics.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_coverage_expansion/"
    )
    / "emit_ladder.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_coverage_expansion/"
    )
    / "emit_pilot_ncu_repeatability.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_coverage_expansion/"
    )
    / "emit_pilot_predictor_predictions.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_coverage_expansion/"
    )
    / "emit_pilot_tier2_harvest_status.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_l2_associativity_probe/"
    )
    / "persist_fitted_sidecar.py",
    Path(
        "tools/sol_baselines_harness/calibration/cache_prediction/"
        "sm120_l2_associativity_probe/"
    )
    / "sweep_associativity.py",
)


@pytest.mark.parametrize("relative_script", _HELP_SCRIPTS, ids=lambda path: path.stem)
def test_retired_cli_help_remains_available(
    relative_script: Path,
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(_REPO_ROOT / relative_script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert _REASON not in result.stdout
    assert _REASON not in result.stderr
