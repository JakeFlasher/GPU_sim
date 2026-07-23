"""Focused regression tests for the sm_120 template safety boundary.

The tests intentionally use temporary manifests/paths and monkeypatch producer
entry points. They prove that sm_120 requests become typed-unavailable before
training, bundle emission, trace-sidecar regeneration, or comparison scoring,
while the sm_86 route remains available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.local_labels import build_predictor_bundle_for_cell as bundle_builder
from tools.local_labels import fit_calibration_head_v4_sm120 as fitter
from tools.local_labels import regenerate_v5_trace_sidecars as sidecars
from tools.local_labels import sm120_activation_bundle_producer as producer
from tools.local_labels.sm120_template_safety import (
    SM120_TEMPLATE_UNAVAILABLE_REASON,
    canonical_arch,
)


def test_arch_aliases_cannot_bypass_template_gate() -> None:
    assert canonical_arch("sm_86") == "sm_86"
    for token in ("sm_120", "sm120", "sm_120-laptop", "sm_120a", "sm120f"):
        assert canonical_arch(token) == "sm_120"


def test_bundle_builder_returns_unavailable_before_stage_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bundle_builder,
        "legacy_match_scan",
        lambda **_: pytest.fail("sm_120 must not scan historical bundles"),
    )
    monkeypatch.setattr(
        bundle_builder,
        "fresh_regen_via_walker",
        lambda **_: pytest.fail("sm_120 must not invoke fresh regeneration"),
    )

    result = bundle_builder.build_predictor_bundle_for_cell(
        arch="sm_120",
        slug="historical_kernel",
        shape_point="trace_xl",
    )

    assert result["status"] == "typed_unavailable"
    assert result["result_kind"] == "typed_unavailable"
    assert result["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert result["bundle_admission_path"] == "unavailable"
    assert result["output_path"] is None


def test_sm86_bundle_builder_still_enters_active_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bundle_builder,
        "_label_5tuple_for_cell",
        lambda **_: {"slug": "active", "shape_point": "trace_xl"},
    )
    monkeypatch.setattr(
        bundle_builder,
        "legacy_match_scan",
        lambda **_: calls.append("legacy") or {
            "admitted": True,
            "path": "data/local_labels/sweeps/active.json",
        },
    )

    result = bundle_builder.build_predictor_bundle_for_cell(
        arch="sm_86",
        slug="active",
        shape_point="trace_xl",
    )

    assert calls == ["legacy"]
    assert result["status"] == "admitted"
    assert result["bundle_admission_path"] == "legacy_match"


def test_sm120_producer_main_is_non_mutating_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        producer,
        "_query_l2_cache_size_bytes",
        lambda **_: pytest.fail("CUDA geometry query must not run"),
    )
    output_root = tmp_path / "would_be_output"
    rc = producer.main(["--output-root", str(output_root)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc != 0
    assert payload["result_kind"] == "typed_unavailable"
    assert payload["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert not output_root.exists()


def test_sm120_fitter_does_not_import_training_dependencies_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fitter.fch,
        "_require_numpy",
        lambda: pytest.fail("sm_120 fitter must not initialize NumPy"),
    )
    head_path = tmp_path / "head.json"
    eval_path = tmp_path / "evaluation.json"

    result_head, result_eval, summary = fitter.fit_and_emit_v4(
        roster_path=tmp_path / "missing-roster.toml",
        admitted_path=tmp_path / "missing-labels.json",
        bundles_root=tmp_path / "missing-bundles",
        head_output_path=head_path,
        eval_output_path=eval_path,
    )

    assert result_head is None
    assert result_eval is None
    assert summary["result_kind"] == "typed_unavailable"
    assert summary["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert not head_path.exists()
    assert not eval_path.exists()


def _write_manifest(path: Path, arch: str) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "arch": arch,
                        "slug": "kernel",
                        "shape_point": "trace_xl",
                        "suggested_operator_action": (
                            "REGENERATE_SECTOR_STREAM_FROM_TRACE"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_sm120_sidecar_regeneration_skips_producer_and_preserves_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    _write_manifest(manifest, "sm_120")
    monkeypatch.setattr(
        sidecars,
        "_resolve_trace_dir",
        lambda **_: pytest.fail("sm_120 must not resolve a production trace"),
    )
    monkeypatch.setattr(
        sidecars,
        "_regenerate_sector_stream",
        lambda **_: pytest.fail("sm_120 must not invoke a sidecar producer"),
    )

    report = sidecars.regenerate_manifest_regenerate_rows(
        manifest_path=manifest,
        report_path=report_path,
        dry_run=False,
    )

    assert report["counts"]["typed_unavailable"] == 1
    row = report["per_cell_reports"][0]
    assert row["status"] == "typed_unavailable"
    assert row["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert row["output_path"] is None


def test_sm86_sidecar_regeneration_dry_run_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    _write_manifest(manifest, "sm_86")
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "dynamic_trace.pb").write_bytes(b"trace")
    monkeypatch.setattr(
        sidecars,
        "_resolve_trace_dir",
        lambda **_: trace_dir,
    )

    report = sidecars.regenerate_manifest_regenerate_rows(
        manifest_path=manifest,
        report_path=report_path,
        dry_run=True,
    )

    assert report["counts"]["dry_run_skipped"] == 1
    assert report["per_cell_reports"][0]["status"] == "dry_run_skipped"


def test_headline_runner_routes_sm120_to_typed_unavailable() -> None:
    pytest.importorskip("jsonschema")
    from tools.local_labels import run_headline_cohort_predict_compare as runner

    route = runner._route_predictor_stage(
        headline_cohort=[("kernel", "trace_xl", "sm_120")],
        target_arch="sm_120",
    )
    assert route.outcome == "sm120_template_unavailable"
    assert route.diagnostic == SM120_TEMPLATE_UNAVAILABLE_REASON
    result = runner._sm120_template_unavailable_result(
        headline_cohort=[("kernel", "trace_xl", "sm_120")],
        target_arch="sm_120",
    )
    assert result["result_kind"] == "typed_unavailable"
    assert result["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON


def test_headline_runner_sm86_route_stays_legacy() -> None:
    pytest.importorskip("jsonschema")
    from tools.local_labels import run_headline_cohort_predict_compare as runner

    route = runner._route_predictor_stage(
        headline_cohort=[("kernel", "trace_xl", "sm_86")],
        target_arch="sm_86",
    )
    assert route.outcome == "legacy"


def test_headline_runner_sm120_alias_stops_before_manifest_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("jsonschema")
    from tools.local_labels import run_headline_cohort_predict_compare as runner

    monkeypatch.setattr(
        runner.cr,
        "run_comparison",
        lambda *args, **kwargs: pytest.fail("comparison must not run"),
    )
    output_dir = tmp_path / "runner-output"
    missing_manifest = tmp_path / "missing-manifest.json"
    missing_sidecar = tmp_path / "missing-calibration.json"
    rc = runner.main([
        "--output-dir", str(output_dir),
        "--target-arch", "sm120-laptop",
        "--cohort-manifest", str(missing_manifest),
        "--calibration-evaluation-sidecar", str(missing_sidecar),
    ])

    captured = capsys.readouterr()
    assert rc != 0
    assert SM120_TEMPLATE_UNAVAILABLE_REASON in captured.err
    assert "target_arch_unlabeled_extrapolation" in captured.err
    assert not output_dir.exists()


def test_v5_transfer_gate_blocks_explicit_sm120_before_head_read(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = pytest.importorskip(
        "tools.local_labels.run_v5_active_path_comparison",
    )
    output_dir = tmp_path / "v5-output"
    rc = module.main([
        "--target-arch", "sm_120",
        "--v5-head", str(tmp_path / "missing-head.json"),
        "--sm-120-head-out", str(output_dir / "sm120.json"),
        "--sm-86-head-out", str(output_dir / "sm86.json"),
        "--comparison-out", str(output_dir / "comparison.json"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    payload = json.loads(captured.out)
    assert payload["result_kind"] == "typed_unavailable"
    assert payload["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert payload["comparison_typed_reason"] == (
        "target_arch_unlabeled_extrapolation"
    )
    assert not output_dir.exists()


def test_v5_transfer_gate_detects_sm120_rows_without_target_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = pytest.importorskip(
        "tools.local_labels.run_v5_active_path_comparison",
    )
    monkeypatch.setattr(
        module,
        "_fit_arch_head",
        lambda *args, **kwargs: pytest.fail("sm120 fit must not run"),
    )
    head_path = tmp_path / "historical-head.json"
    head_path.write_text(
        json.dumps({
            "target_arch": "sm_86",
            "cohort_scope": {
                "admitted": [{
                    "arch": "sm_120-laptop",
                    "slug": "historical",
                    "shape_point": "timing_shape",
                }],
            },
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "v5-output"
    rc = module.main([
        "--v5-head", str(head_path),
        "--sm-120-head-out", str(output_dir / "sm120.json"),
        "--sm-86-head-out", str(output_dir / "sm86.json"),
        "--comparison-out", str(output_dir / "comparison.json"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    payload = json.loads(captured.out)
    assert payload["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert not output_dir.exists()


def test_v5_fitter_explicit_sm120_stops_before_input_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = pytest.importorskip(
        "tools.local_labels.fit_monotonic_calibration_head_v5",
    )
    monkeypatch.setattr(
        module,
        "_load_label_cells",
        lambda *args, **kwargs: pytest.fail("sm120 inputs must not load"),
    )
    output_path = tmp_path / "v5-head.json"
    result = module.fit_and_emit_v5(
        label_admitted_path=tmp_path / "missing-labels.json",
        bundle_paths=[tmp_path / "missing-bundle.json"],
        family_buckets_path=tmp_path / "missing-families.toml",
        output_path=output_path,
        arch="sm_120a",
    )

    assert result["result_kind"] == "typed_unavailable"
    assert result["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert result["fit_status"] == "sm120_template_unavailable"
    assert not output_path.exists()


def test_v5_fitter_joint_sm120_cohort_is_typed_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = pytest.importorskip(
        "tools.local_labels.fit_monotonic_calibration_head_v5",
    )
    monkeypatch.setattr(
        module,
        "_build_cell_records",
        lambda *args, **kwargs: pytest.fail("joint sm120 fit must not run"),
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"cells": [{"arch": "sm_120", "slug": "k"}]}),
        encoding="utf-8",
    )
    output_path = tmp_path / "v5-head.json"
    result = module.fit_and_emit_v5(
        label_admitted_path=labels,
        bundle_paths=[],
        family_buckets_path=tmp_path / "missing-families.toml",
        output_path=output_path,
        arch=None,
    )

    assert result["result_kind"] == "typed_unavailable"
    assert result["typed_reason"] == SM120_TEMPLATE_UNAVAILABLE_REASON
    assert not output_path.exists()


def test_v5_transfer_architecture_probe_preserves_pure_sm86_lane() -> None:
    module = pytest.importorskip(
        "tools.local_labels.run_v5_active_path_comparison",
    )
    assert module._v5_head_contains_sm120({
        "target_arch": "sm_86",
        "cohort_scope": {"admitted": [{"arch": "sm_86"}]},
    }) is False


def test_sm120_calibration_override_cannot_bypass_gate() -> None:
    pytest.importorskip("jsonschema")
    from tools.local_labels import run_headline_cohort_predict_compare as runner
    from tools.local_labels.sm120_template_safety import Sm120TemplateUnavailable

    with pytest.raises(Sm120TemplateUnavailable) as exc_info:
        runner._resolve_calibration_evaluation_sidecar_path(
            explicit_override=Path("historical-v4.json"),
            target_arch="sm_120-laptop",
            repo_root=Path("."),
        )
    assert exc_info.value.typed_reason == SM120_TEMPLATE_UNAVAILABLE_REASON
