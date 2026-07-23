"""L1 calibration head + L1 closure: scoring, bootstrap CI, admission gating.

Positive: synthetic L1 predictions + labels fit a bounded-[0,1] L1 head instance,
are scored by the comparison runner with a bootstrap CI, and emit the L1 closure
artifact; the frozen L2 head is left byte-identical. Negative: an admitted cell
missing its L1 label fails admission into the L1 head fit, and a non-scalar L1
prediction is typed-skipped rather than scored.

Reproducibility: the fit/score/closure path uses the v5 monotonic model-class
hierarchy, whose primary class is a sklearn IsotonicRegression. In an environment
with NONE of sklearn/pygam/lightgbm/xgboost the fit emits
``fit_status="indeterminate_dependencies_absent"`` and the closure cannot
rehydrate the head — so the FIT-DEPENDENT tests skip explicitly when no model
package is importable (Codex Round-3 review blocking #1 / required #6). The
committed-artifact provenance test
(:func:`test_committed_l1_artifacts_provenance_without_sklearn`) runs
UNCONDITIONALLY: it validates the committed head shape + frozen-L2 sha + the
committed closure's head provenance and floor WITHOUT requiring sklearn.

Host-independent: synthetic fixtures + pytest tmp_path; no GPU, no driver.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "workloads" / "simbench_bw",
           REPO_ROOT / "tools" / "local_labels"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_l1_calibration_head as fl1  # noqa: E402
import emit_l1_closure_artifacts as el1  # noqa: E402
from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402

_DATA = REPO_ROOT / "data" / "local_labels"
# Byte-pin of the committed v5 (L2) head: the L1 fit must leave it
# untouched. Updated in lockstep whenever the v5 head is LEGITIMATELY
# regenerated (most recently: the demotion-round regeneration from
# committed HEAD, which only moved the source_commit stamp).
_FROZEN_L2_HEAD_SHA = (
    "d28665f6a6379139f245114ba6ce11f5d9e0d3cf87aca39301ef695cba83a83c"
)
_NCU_VER = "ncu_2026.1.x"
_FAMILY_BUCKETS = _DATA / "sm120_cohort_family_buckets.toml"
_CONTRACT = REPO_ROOT / "workloads" / "simbench_bw" / "comparison_contract.toml"


def _any_model_class_available() -> bool:
    """True iff at least one registered model class can be fit on this host.

    The L1 fit + the closure's head rehydration both require an importable model
    class (sklearn IR primary, then pygam / lightgbm / xgboost). When none is
    importable the fit emits indeterminate and the closure cannot rehydrate, so
    the fit-dependent tests skip rather than assert fit_status=="ok"."""
    probe = mch.probe_dependency_availability()
    return any(v.get("available") for v in probe.values())


_MODEL_DEP = pytest.mark.skipif(
    not _any_model_class_available(),
    reason=(
        "no calibration model class (sklearn/pygam/lightgbm/xgboost) importable; "
        "the L1 fit/score path is indeterminate without one. The committed-"
        "artifact provenance test covers the no-dependency environment."
    ),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mk_pred_cell(slug, *, result_kind, predicted_hit_rate=None,
                  typed_reason=None, idx=0):
    return {
        "slug": slug,
        "shape_point": "timing_shape",
        "arch": "sm_86",
        "binary_sha256": f"{idx:064x}",
        "source_commit": f"{idx:040x}",
        "kernel_signature_sha256": f"{idx + 100:064x}",
        "ncu_metric_version": _NCU_VER,
        "result_kind": result_kind,
        "cache_layer": "L1TEX",
        "predicted_hit_rate": predicted_hit_rate,
        "typed_reason": typed_reason,
        "evidence": {},
        "input_digests": {
            "classifier_manifest_sha256": f"{idx + 1:064x}",
            "dynamic_trace_pb_sha256": f"{idx + 2:064x}",
            "threadblock_proto_set_sha256": f"{idx + 3:064x}",
        },
    }


def _mk_label_cell(slug, *, l1_pct, l1tex_label_status="mean_present", idx=0):
    return {
        "slug": slug,
        "shape_point": "timing_shape",
        "binary_sha256": f"{idx:064x}",
        "source_commit": f"{idx:040x}",
        "kernel_signature_sha256": f"{idx + 100:064x}",
        "arch": "sm_86",
        "profiler_version": _NCU_VER,
        "ncu_metric_name": "lts__t_sector_hit_rate.pct",
        "cache_hit_rate_l2_sector_pct": 50.0,
        "cache_hit_rate_l1tex_sector_pct": l1_pct,
        "l1tex_label_status": l1tex_label_status,
        "evidence_kind": "live",
    }


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _fit_head(tmp_path, preds, labels, out_name="sm_86_l1_head.json") -> dict:
    pred_path = _write_json(tmp_path / "preds.json", preds)
    label_path = _write_json(tmp_path / "labels.json", labels)
    out_path = tmp_path / out_name
    fl1.fit_and_emit_l1(
        label_admitted_path=label_path,
        l1_predictions_path=pred_path,
        family_buckets_path=_FAMILY_BUCKETS,
        output_path=out_path,
        arch="sm_86",
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


@_MODEL_DEP
def test_l1_head_is_bounded_instance_and_leaves_l2_head_byte_identical(tmp_path):
    preds = {
        "schema_version": 2, "generator": "test", "cohort_size": 3,
        "cells": [
            _mk_pred_cell("k_a", result_kind="scalar", predicted_hit_rate=0.30, idx=1),
            _mk_pred_cell("k_b", result_kind="scalar", predicted_hit_rate=0.70, idx=2),
            _mk_pred_cell("k_c", result_kind="typed_unavailable",
                          typed_reason="predictor_l1_sass_provenance_missing", idx=3),
        ],
    }
    labels = {"schema_version": 5, "cells": [
        _mk_label_cell("k_a", l1_pct=31.0, idx=1),
        _mk_label_cell("k_b", l1_pct=69.0, idx=2),
        _mk_label_cell("k_c", l1_pct=88.0, idx=3),
    ]}
    head = _fit_head(tmp_path, preds, labels)
    assert head["head_kind"] == "l1_sector_hit_rate"
    assert head["cache_layer"] == "L1TEX"
    assert head["head_scope"] == "standalone_single_arch_l1"
    assert head["fit_status"] == "ok"
    # Two scalar cells admitted; the typed_unavailable cell is skipped with its
    # propagated root-cause reason (NOT silently dropped).
    assert head["cohort_scope"]["admitted_count"] == 2
    skip_reasons = {s["typed_reason"] for s in head["cohort_scope"]["skipped"]}
    assert "predictor_l1_sass_provenance_missing" in skip_reasons
    # Frozen L2 head untouched by the L1 fit.
    assert _sha(_DATA / "sm_86_v5_head.json") == _FROZEN_L2_HEAD_SHA


def test_l1_head_skips_admitted_cell_missing_its_l1_label(tmp_path):
    """NEGATIVE: a label cell whose L1 label is absent fails admission into the
    L1 head fit (it is typed-skipped, never scored on a fabricated label). This
    runs WITHOUT a model dependency: the admission gate fires before any fit, so
    admitted_count==0 regardless of fit_status (ok or indeterminate)."""
    preds = {"schema_version": 2, "generator": "test", "cohort_size": 1, "cells": [
        _mk_pred_cell("k_missing", result_kind="scalar", predicted_hit_rate=0.5, idx=7),
    ]}
    labels = {"schema_version": 5, "cells": [
        _mk_label_cell("k_missing", l1_pct=None,
                       l1tex_label_status="mean_absent_label_missing", idx=7),
    ]}
    head = _fit_head(tmp_path, preds, labels, out_name="head.json")
    assert head["cohort_scope"]["admitted_count"] == 0
    skip_reasons = {s["typed_reason"] for s in head["cohort_scope"]["skipped"]}
    # The mean_absent status is surfaced (not the label-absent fallthrough)
    # because the status gate fires first; either typed skip is acceptable as
    # long as the cell is NOT admitted.
    assert skip_reasons & {
        "l1tex_label_status_mean_absent_label_missing",
        "label_cache_hit_rate_l1tex_sector_pct_absent",
    }


def test_build_l1tex_bundles_tags_cache_layer_and_arms():
    """Pure transform — no model dependency."""
    preds = {"cells": [
        _mk_pred_cell("k_a", result_kind="scalar", predicted_hit_rate=0.3, idx=1),
        _mk_pred_cell("k_c", result_kind="typed_unavailable",
                      typed_reason="predictor_l1_sass_provenance_missing", idx=3),
    ]}
    bundles = el1.build_l1tex_bundles_from_predictions(preds)
    assert len(bundles) == 2
    assert all(b["cache_layer"] == "L1TEX" for b in bundles)
    scalar = next(b for b in bundles if b["result_kind"] == "scalar")
    assert scalar["predicted_hit_rate"] == 0.3
    assert scalar["ci_low"] == scalar["ci_high"] == 0.3
    assert scalar["ci_half_width"] == 0.0
    tu = next(b for b in bundles if b["result_kind"] == "typed_unavailable")
    assert tu["typed_reason"] == "predictor_l1_sass_provenance_missing"
    assert "predicted_hit_rate" not in tu


@_MODEL_DEP
def test_l1_closure_is_head_calibrated_and_scored(tmp_path):
    """POSITIVE: the closure LOADS the fitted L1 head, calibrates the walker
    predictions through it, and scores the calibrated L1TEX bundles with a
    bootstrap CI; the closure records head provenance and the calibrated values
    differ from the raw walker values (proving the head is load-bearing)."""
    preds = {"schema_version": 2, "generator": "test", "cohort_size": 3, "cells": [
        _mk_pred_cell("k_a", result_kind="scalar", predicted_hit_rate=0.30, idx=1),
        _mk_pred_cell("k_b", result_kind="scalar", predicted_hit_rate=0.70, idx=2),
        _mk_pred_cell("k_c", result_kind="typed_unavailable",
                      typed_reason="predictor_l1_sass_provenance_missing", idx=3),
    ]}
    labels = {"schema_version": 5, "cells": [
        _mk_label_cell("k_a", l1_pct=31.0, idx=1),
        _mk_label_cell("k_b", l1_pct=69.0, idx=2),
        _mk_label_cell("k_c", l1_pct=88.0, idx=3),
    ]}
    pred_path = _write_json(tmp_path / "preds.json", preds)
    label_path = _write_json(tmp_path / "labels.json", labels)
    head_path = tmp_path / "sm_86_l1_head.json"
    fl1.fit_and_emit_l1(
        label_admitted_path=label_path,
        l1_predictions_path=pred_path,
        family_buckets_path=_FAMILY_BUCKETS,
        output_path=head_path,
        arch="sm_86",
    )
    out_path = tmp_path / "l1_closure.json"
    out = el1.emit_l1_closure(
        l1_head_path=head_path,
        l1_predictions_path=pred_path,
        label_admitted_path=label_path,
        contract_path=_CONTRACT,
        output_path=out_path,
        family_buckets_path=_FAMILY_BUCKETS,
        arch="sm_86",
    )
    per_arch = out["per_arch_diagnostics"]["per_arch"]["sm_86"]
    l1 = per_arch["relative_error_stats_l1tex"]
    assert l1["n_cells"] == 3
    assert l1["n_scalar"] == 2
    assert l1["n_typed_unavailable"] == 1
    ci = per_arch["relative_error_geomean_ci"]
    assert ci["bootstrap_resamples"] == 2000
    assert ci["ci_low"] <= ci["ci_high"]
    assert "coverage_gate_verdict" in per_arch
    # Head provenance present + the head actually transformed at least one value.
    prov = out["l1_head_provenance"]
    assert prov["l1_head_fit_status"] == "ok"
    assert str(prov["l1_head_path"]) == str(head_path)
    assert prov["calibrated_cells"], "must list calibrated cells"
    assert any(
        abs(c["raw_walker_prediction"] - c["calibrated_prediction"]) > 1e-9
        for c in prov["calibrated_cells"]
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["closure_artifact_kind"] == "l1_sector_hit_rate_closure"
    assert written["closure_cache_layer"] == "L1TEX"


@_MODEL_DEP
def test_l1_closure_fails_closed_on_invalid_head(tmp_path):
    """NEGATIVE: a head artifact that fails the load-bearing contract
    (fit_status != ok) makes the closure FAIL CLOSED — no silent raw fallback."""
    preds = {"schema_version": 2, "generator": "test", "cohort_size": 1, "cells": [
        _mk_pred_cell("k_a", result_kind="scalar", predicted_hit_rate=0.3, idx=1),
    ]}
    labels = {"schema_version": 5, "cells": [
        _mk_label_cell("k_a", l1_pct=31.0, idx=1),
    ]}
    pred_path = _write_json(tmp_path / "preds.json", preds)
    label_path = _write_json(tmp_path / "labels.json", labels)
    # A head with the wrong fit_status must be rejected.
    bad_head = {
        "calibration_head_version": 5,
        "head_kind": "l1_sector_hit_rate",
        "cache_layer": "L1TEX",
        "fit_status": "indeterminate_dependencies_absent",
    }
    bad_head_path = _write_json(tmp_path / "bad_head.json", bad_head)
    with pytest.raises(el1.L1HeadValidationError):
        el1.emit_l1_closure(
            l1_head_path=bad_head_path,
            l1_predictions_path=pred_path,
            label_admitted_path=label_path,
            contract_path=_CONTRACT,
            output_path=tmp_path / "out.json",
            family_buckets_path=_FAMILY_BUCKETS,
            arch="sm_86",
        )


def test_load_and_validate_l1_head_rejects_wrong_fields(tmp_path):
    """Pure validation — no model dependency. Each wrong required field is
    rejected by the load-bearing contract gate."""
    base = {
        "calibration_head_version": 5,
        "head_kind": "l1_sector_hit_rate",
        "cache_layer": "L1TEX",
        "fit_status": "ok",
    }
    # Good head validates.
    good = _write_json(tmp_path / "good.json", base)
    assert el1.load_and_validate_l1_head(good)["fit_status"] == "ok"
    # Each corruption is rejected.
    for field, bad in [
        ("calibration_head_version", 4),
        ("head_kind", "l2_sector_hit_rate"),
        ("cache_layer", "L2"),
        ("fit_status", "empty_cohort_no_admitted_cells"),
    ]:
        corrupt = dict(base)
        corrupt[field] = bad
        p = _write_json(tmp_path / f"bad_{field}.json", corrupt)
        with pytest.raises(el1.L1HeadValidationError):
            el1.load_and_validate_l1_head(p)
    # Missing file is rejected.
    with pytest.raises(el1.L1HeadValidationError):
        el1.load_and_validate_l1_head(tmp_path / "does_not_exist.json")


def test_committed_l1_artifacts_provenance_without_sklearn():
    """sklearn-free committed-artifact validation (Codex required #6).

    Runs UNCONDITIONALLY (no model dependency): validates the committed L1 head
    shape, the frozen-L2 byte identity, and the committed L1 closure's head
    provenance + coverage floor by reading JSON only — no fit, no rehydration."""
    head_path = _DATA / "sm_86_l1_head.json"
    assert head_path.is_file()
    head = json.loads(head_path.read_text(encoding="utf-8"))
    assert head["calibration_head_version"] == 5
    assert head["head_kind"] == "l1_sector_hit_rate"
    assert head["cache_layer"] == "L1TEX"
    assert head["arch"] == "sm_86"
    assert head["fit_status"] == "ok"
    assert head["cohort_scope"]["admitted_count"] >= 5

    # Frozen L2 head byte-identical.
    assert _sha(_DATA / "sm_86_v5_head.json") == _FROZEN_L2_HEAD_SHA

    # Committed closure: top-level admitted, floor met, head provenance present.
    closure_path = _DATA / "l1_closure_comparison_sm_86.json"
    assert closure_path.is_file()
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    assert closure["result_kind"] == "admitted"
    pa = closure["per_arch_diagnostics"]["per_arch"]["sm_86"]
    verdict = pa["coverage_gate_verdict"]
    assert verdict["meets_floor"] is True
    assert verdict["headline_count_live"] >= verdict["floor"]
    assert pa["relative_error_stats_l1tex"]["n_scalar"] >= 5
    prov = closure["l1_head_provenance"]
    assert prov["l1_head_fit_status"] == "ok"
    assert str(prov["l1_head_path"]).endswith("sm_86_l1_head.json")
    # The committed closure proves the head transformed at least one value.
    assert any(
        abs(c["raw_walker_prediction"] - c["calibrated_prediction"]) > 1e-9
        for c in prov["calibrated_cells"]
    )
