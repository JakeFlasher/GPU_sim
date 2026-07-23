"""Tests for the calibration spine: parameter grid + staged selection, the
monotone residual provenance guard, the accuracy gate (positive + negative),
the falsification report (positive + negative), and the sm_120-only scope of the
emitted artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # noqa: E402
import parameter_grid as pg  # noqa: E402
import residual as residual_mod  # noqa: E402
import accuracy_gate  # noqa: E402
import falsification  # noqa: E402
import cohort  # noqa: E402
import calibrate  # noqa: E402

_L2 = 32 * 1024 * 1024


class ParameterGridTest(unittest.TestCase):
    def test_grid_within_pre_registered_cap(self):
        grid = pg.grid_for_device(_L2)
        self.assertLessEqual(len(grid), pg.GRID_SIZE_CAP)
        self.assertEqual(pg.GRID_SIZE_CAP, 75)

    def test_grid_is_a_frozen_tuple_of_configs(self):
        grid = pg.grid_for_device(_L2)
        self.assertIsInstance(grid, tuple)
        for c in grid:
            self.assertIsInstance(c, scr.CacheReplayConfig)

    def test_grid_expansion_beyond_cap_is_rejected(self):
        # Enlarge the slice axis so the enumerated grid exceeds the cap; the
        # size guard must reject it (27 stage-one + 38*5 stage-two > 75).
        with mock.patch.object(pg, "SLICE_COUNTS", tuple(range(2, 40))):
            with self.assertRaises(ValueError):
                pg.grid_for_device(_L2)

    def test_staged_cv_converges_to_simplest_under_inert_grid(self):
        # When every config predicts identically (an inert grid), the
        # simpler-within-two-percent tie-break must select the simplest config:
        # one slice, smallest associativity, LRU, capacity scale 1.0.
        grid = pg.grid_for_device(_L2)
        labels = [0.5] * 10
        buckets = (["tensor_compute"] * 3 + ["activation_norm"] * 3
                   + ["memory_stress"] * 2 + ["other"] * 2)

        def predict_fn(cfg):
            return [0.7] * 10  # inert: identical for every config

        def cv_score_fn(preds, labs, fbs):
            return abs(preds[0] - 0.5)

        sel = pg.select_config(
            grid, predict_fn=predict_fn, labels=labels,
            family_buckets=buckets, cv_score_fn=cv_score_fn,
        )
        self.assertEqual(sel.selected.num_slices, 1)
        self.assertEqual(sel.selected.associativity, min(pg.ASSOCIATIVITIES))
        self.assertEqual(sel.selected.replacement_policy, "LRU")
        self.assertEqual(sel.selected.capacity_scale, 1.0)

    def test_stage_two_reanchors_on_noncanonical_stage_one_winner(self):
        # The staged protocol must re-anchor stage two on the ACTUAL stage-one
        # winner. Here stage one favours a NON-canonical config (0.75, 8, SRRIP),
        # and a multi-slice config wins only if it inherits that anchor.
        grid = pg.grid_for_device(_L2)
        labels = [0.5] * 6
        buckets = ["tensor_compute"] * 6
        winner_anchor = (0.75, 8, "SRRIP")
        winning_sig = (0.75, 8, "SRRIP", max(pg.SLICE_COUNTS), "knuth_mult")

        def cv_score_fn(preds, labs, fbs):
            return preds[0]  # the prediction IS the score (lower better)

        def predict_fn(cfg):
            sig = pg._config_signature(cfg)
            if sig == winning_sig:
                return [0.01] * 6                      # best overall
            anchor = (cfg.capacity_scale, cfg.associativity, cfg.replacement_policy)
            if cfg.num_slices == 1 and anchor == winner_anchor:
                return [0.05] * 6                      # stage-one winner
            if cfg.num_slices == 1:
                return [0.50] * 6                      # other stage-one configs
            return [0.40] * 6                          # non-winning multi-slice

        sel = pg.select_config(
            grid, predict_fn=predict_fn, labels=labels,
            family_buckets=buckets, cv_score_fn=cv_score_fn,
        )
        # stage two must have re-anchored on the non-canonical stage-one winner
        self.assertEqual(
            (sel.stage_two_anchor.capacity_scale,
             sel.stage_two_anchor.associativity,
             sel.stage_two_anchor.replacement_policy),
            winner_anchor,
        )
        self.assertEqual(pg._config_signature(sel.selected), winning_sig)
        self.assertLessEqual(sel.n_candidates_evaluated, pg.GRID_SIZE_CAP)

    def test_stratified_folds_are_deterministic(self):
        buckets = ["a", "a", "b", "b", "c", "c"]
        f1 = pg.stratified_folds(buckets, n_folds=3)
        f2 = pg.stratified_folds(buckets, n_folds=3)
        self.assertEqual(f1, f2)


class ResidualProvenanceTest(unittest.TestCase):
    def test_one_dimensional_provenance_admitted(self):
        prov = {"free_dimensions": 1, "feature": residual_mod.RESIDUAL_FEATURE}
        residual_mod.validate_residual_provenance(prov)  # must not raise

    def test_more_than_one_free_dimension_rejected(self):
        with self.assertRaises(ValueError):
            residual_mod.validate_residual_provenance({"free_dimensions": 2})

    def test_missing_free_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            residual_mod.validate_residual_provenance({"feature": "x"})

    def test_monotone_fit_is_increasing(self):
        head = residual_mod.fit_monotone_residual(
            [0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.2, 0.6, 0.7, 1.0])
        preds = head.predict([0.1, 0.5, 0.9])
        self.assertLessEqual(preds[0], preds[1])
        self.assertLessEqual(preds[1], preds[2])
        self.assertEqual(head.provenance()["free_dimensions"], 1)


class AccuracyGateTest(unittest.TestCase):
    def test_passing_predictor_admitted(self):
        verdict = accuracy_gate.evaluate_accuracy_gate(
            full_cohort_median_abs=0.04,
            full_cohort_geomean_rel=0.10,
            holdout_geomean_rel=0.20,
            per_bucket_p90={
                "tensor_compute": {"n": 9, "p90_absolute_error": 0.10},
                "activation_norm": {"n": 7, "p90_absolute_error": 0.12},
            },
        )
        self.assertTrue(verdict["passed"])

    def test_median_absolute_error_too_high_fails(self):
        verdict = accuracy_gate.evaluate_accuracy_gate(
            full_cohort_median_abs=0.10,
            full_cohort_geomean_rel=0.10,
            holdout_geomean_rel=0.20,
            per_bucket_p90={"tensor_compute": {"n": 9, "p90_absolute_error": 0.10}},
        )
        self.assertFalse(verdict["passed"])

    def test_per_bucket_p90_too_high_fails(self):
        verdict = accuracy_gate.evaluate_accuracy_gate(
            full_cohort_median_abs=0.04,
            full_cohort_geomean_rel=0.10,
            holdout_geomean_rel=0.20,
            per_bucket_p90={"memory_stress": {"n": 8, "p90_absolute_error": 0.20}},
        )
        self.assertFalse(verdict["passed"])

    def test_small_bucket_not_evaluated(self):
        verdict = accuracy_gate.evaluate_accuracy_gate(
            full_cohort_median_abs=0.04,
            full_cohort_geomean_rel=0.10,
            holdout_geomean_rel=0.20,
            per_bucket_p90={"other": {"n": 3, "p90_absolute_error": 0.99}},
        )
        self.assertTrue(verdict["passed"])


def _synthetic_bundle(*, replay_medabs, nug_medabs, v5_beat, nug_beat,
                      fully_assoc, most_set_assoc):
    return {
        "top_level_admission_scope": "target_arch=sm_120",
        "arches": ["sm_120"],
        "aggregate_metrics": {
            "replay_prediction": {"median_absolute_error": replay_medabs,
                                  "geomean_relative_error": 0.4},
            "replay_calibrated_prediction": {"median_absolute_error": replay_medabs,
                                             "geomean_relative_error": 0.3},
            "nugteren_prediction": {"median_absolute_error": nug_medabs,
                                    "geomean_relative_error": 0.41},
            "v5_prediction": {"median_absolute_error": 0.075,
                              "geomean_relative_error": 0.07},
        },
        "replay_vs_v5_beat_fraction": v5_beat,
        "replay_vs_rdh_baseline_beat_fraction": nug_beat,
        "holdout_geomean": 0.3,
        "per_family_bucket_p90_calibrated": {
            "tensor_compute": {"n": 9, "p90_absolute_error": 0.3},
        },
        "grid_config_medabs": {
            "fully_associative_single_slice": fully_assoc,
            "most_set_associative": most_set_assoc,
            "min": min(fully_assoc, most_set_assoc),
            "max": max(fully_assoc, most_set_assoc),
        },
        "inert_grid_evidence": {"max_grid_prediction_spread": 0.0},
        "rows": [
            {"replay_prediction": 0.9, "label": 0.1},
            {"replay_prediction": 0.9, "label": 0.9},
        ],
    }


class FalsificationTest(unittest.TestCase):
    def test_inert_grid_refutes_hypothesis_and_closes_engineering(self):
        bundle = _synthetic_bundle(
            replay_medabs=0.27, nug_medabs=0.34, v5_beat=0.27, nug_beat=0.63,
            fully_assoc=0.27, most_set_assoc=0.27,
        )
        report = falsification.build_falsification_report(bundle)
        falsification.validate_falsification_report(report)
        self.assertEqual(report["closure_disjunct"], "engineering_closure")
        self.assertEqual(report["hypothesis"]["verdict"], "refuted")
        self.assertIn("accuracy_gate", report["missed_gates"])

    def test_strong_set_associative_contribution_confirms(self):
        bundle = _synthetic_bundle(
            replay_medabs=0.20, nug_medabs=0.34, v5_beat=0.27, nug_beat=0.63,
            fully_assoc=0.30, most_set_assoc=0.20,  # ~33% improvement
        )
        report = falsification.build_falsification_report(bundle)
        self.assertEqual(report["hypothesis"]["verdict"], "confirmed")

    def test_missing_failure_mode_rejected(self):
        bundle = _synthetic_bundle(
            replay_medabs=0.27, nug_medabs=0.34, v5_beat=0.27, nug_beat=0.63,
            fully_assoc=0.27, most_set_assoc=0.27,
        )
        report = falsification.build_falsification_report(bundle)
        del report["failure_mode_decomposition"]["trace_fidelity"]
        with self.assertRaises(ValueError):
            falsification.validate_falsification_report(report)

    def test_cross_arch_precondition_tristate(self):
        # 2026-07-06 re-founding: tri-state (false | true | 'pending') is
        # accepted; anything else is rejected.
        bundle = _synthetic_bundle(
            replay_medabs=0.27, nug_medabs=0.34, v5_beat=0.27, nug_beat=0.63,
            fully_assoc=0.27, most_set_assoc=0.27,
        )
        report = falsification.build_falsification_report(bundle)
        report["cross_arch_precondition"] = True
        falsification.validate_falsification_report(report)  # must not raise
        report["cross_arch_precondition"] = "pending"
        falsification.validate_falsification_report(report)  # must not raise
        report["cross_arch_precondition"] = "verified"
        with self.assertRaises(ValueError):
            falsification.validate_falsification_report(report)

    def test_indeterminate_verdict_is_not_a_valid_closure(self):
        bundle = _synthetic_bundle(
            replay_medabs=0.27, nug_medabs=0.34, v5_beat=0.27, nug_beat=0.63,
            fully_assoc=0.30, most_set_assoc=0.255,  # ~15% -> indeterminate
        )
        report = falsification.build_falsification_report(bundle)
        self.assertEqual(report["hypothesis"]["verdict"], "indeterminate")
        with self.assertRaises(ValueError):
            falsification.validate_falsification_report(report)


def _write_pb(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _synthetic_cell(trace_dir: Path) -> "cohort.Cell":
    return cohort.Cell(
        slug="probe", shape_point="trace_x", family_bucket="other",
        label=0.5, l1_label=None, workload="simbench_bw", trace_dir=trace_dir,
    )


class StreamCacheTraceIdentityTest(unittest.TestCase):
    """The decoded-stream `.npz` cache must be keyed by trace CONTENT identity,
    not by cell id alone -- otherwise a regenerated/swapped trace (same
    slug/shape) is a stale cache HIT replaying the wrong line_ids/sector_offsets.
    The fixtures use the real on-disk `trace_<shape>/threadblocks/*.pb` shape the
    decoder consumes (BL-20260527: mirror the real artifact shape)."""

    def test_cache_path_changes_when_trace_pb_content_changes(self):
        with tempfile.TemporaryDirectory() as td:
            trace_dir = Path(td) / "trace_x"
            pb = trace_dir / "threadblocks" / "launch0" / "tb0.pb"
            _write_pb(pb, b"AAAA")
            cell = _synthetic_cell(trace_dir)
            p1 = cohort._cache_path(cell)
            self.assertIn("__", p1.name)  # a trace fingerprint is embedded
            # idempotent: same content -> same cache file
            self.assertEqual(p1, cohort._cache_path(cell))
            # regeneration: same slug/shape, different bytes (different size)
            # -> the cache path MUST change (a miss, not a stale hit)
            _write_pb(pb, b"BBBBBBBB")
            p2 = cohort._cache_path(cell)
            self.assertNotEqual(
                p1, p2,
                "stream cache path must change when the trace .pb content changes",
            )
            # swap: an added .pb file changes the structural manifest
            _write_pb(trace_dir / "threadblocks" / "launch0" / "tb1.pb", b"CCCC")
            p3 = cohort._cache_path(cell)
            self.assertNotEqual(p2, p3)

    def test_empty_trace_dir_uses_sentinel_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            cell = _synthetic_cell(Path(td))
            self.assertTrue(cohort._cache_path(cell).name.endswith("__empty.npz"))


class PredictionCacheCohortIdentityTest(unittest.TestCase):
    """The higher-level prediction cache (config-signature -> per-cell
    prediction) must invalidate when any cell's trace changes; it is namespaced
    by a cohort trace fingerprint."""

    def test_cohort_fingerprint_and_pred_cache_path_track_trace_changes(self):
        with tempfile.TemporaryDirectory() as td:
            trace_dir = Path(td) / "trace_x"
            pb = trace_dir / "threadblocks" / "launch0" / "tb0.pb"
            _write_pb(pb, b"AAAA")
            cell = _synthetic_cell(trace_dir)
            fp1 = cohort.cohort_fingerprint([cell])
            path1 = calibrate._prediction_cache_path([cell])
            self.assertIn(fp1, path1.name)
            self.assertEqual(path1, calibrate._prediction_cache_path([cell]))
            # change the trace -> fingerprint + cache file must change
            _write_pb(pb, b"BBBBBBBB")
            fp2 = cohort.cohort_fingerprint([cell])
            path2 = calibrate._prediction_cache_path([cell])
            self.assertNotEqual(fp1, fp2)
            self.assertNotEqual(path1, path2)


class FalsificationBenchmarkOverrideTest(unittest.TestCase):
    """`falsification.main` must honor `--benchmark <path>`, not silently load
    the hard-coded default artifact."""

    def test_main_honors_benchmark_override(self):
        bundle = _synthetic_bundle(
            replay_medabs=0.27, nug_medabs=0.34, v5_beat=0.27, nug_beat=0.63,
            fully_assoc=0.27, most_set_assoc=0.27,
        )
        with tempfile.TemporaryDirectory() as td:
            comp = Path(td) / "comparison.json"
            comp.write_text(json.dumps(bundle))
            # Distinctive override: the committed DEFAULT benchmark records
            # passed=false (the throughput bar is missed on the 30-cell scope),
            # so a captured passed=true proves main read THIS path, not the
            # hard-coded default.
            bench = Path(td) / "custom_benchmark.json"
            bench.write_text(json.dumps({"passed": True, "note": "override"}))
            captured: dict = {}
            real_build = falsification.build_falsification_report

            def _spy(bundle_arg, **kw):
                captured["benchmark"] = kw.get("benchmark")
                captured["throughput_passed"] = kw.get("throughput_passed")
                return real_build(bundle_arg, **kw)

            with mock.patch.object(falsification, "build_falsification_report", _spy), \
                 mock.patch.object(falsification.artifact_io, "validate_against_schema"), \
                 mock.patch.object(falsification.artifact_io, "atomic_write_json"):
                rc = falsification.main(
                    ["--comparison", str(comp), "--benchmark", str(bench)]
                )
            self.assertEqual(rc, 0)
            self.assertIsNotNone(captured["benchmark"])
            self.assertTrue(captured["benchmark"]["passed"])
            self.assertTrue(captured["throughput_passed"])


class EmittedArtifactScopeTest(unittest.TestCase):
    """The emitted comparison + results artifacts must be sm_120-only."""

    def _load(self, name):
        import json
        p = _REPO_ROOT / "data/local_labels" / name
        if not p.exists():
            self.skipTest(f"{name} not emitted yet")
        return json.loads(p.read_text())

    def test_comparison_bundle_is_sm120_only(self):
        b = self._load("cache_replay_comparison.json")
        self.assertEqual(b["top_level_admission_scope"], "target_arch=sm_120")
        self.assertEqual(b["arches"], ["sm_120"])

    def test_results_artifact_is_sm120_only(self):
        r = self._load("cache_replay_results.json")
        self.assertEqual(r["top_level_admission_scope"], "target_arch=sm_120")
        self.assertEqual(r["arches"], ["sm_120"])
        for cell in r["results"]:
            self.assertEqual(cell["join_key"]["arch"], "sm_120")

    def test_falsification_report_is_terminal_and_decisive(self):
        rep = self._load("cache_replay_falsification.json")
        self.assertTrue(rep["terminal_on_sm120_host"])
        self.assertFalse(rep["cross_arch_precondition"])
        falsification.validate_falsification_report(rep)


if __name__ == "__main__":
    unittest.main()
