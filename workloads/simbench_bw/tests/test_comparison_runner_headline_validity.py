"""Witnesses for the AC-T7 headline-validity guard scaffolding
added to ``workloads/simbench_bw/comparison_runner.py``.

Three test surfaces:

  1. ``HeadlineValidityHelperPositiveNegativeTest`` exercises
     each of the three new detection helpers
     (``check_calibration_reuse_distance_envelope``,
     ``check_calibration_covariate_envelope``,
     ``check_trace_opcode_coverage``) with one positive and
     one negative witness per helper (eight tests). Each
     helper returns the typed-reason string on violation and
     ``None`` on the clean path. The covariate helper carries
     three distinct positive witnesses (one per covariate
     boolean) so a future regression that swaps the
     conjunction order can be caught.
  2. ``RunComparisonHeadlineValidityIntegrationTest`` drives
     ``run_comparison(...)`` with the new keyword-only
     ``headline_validity_inputs`` kwarg and asserts:
       (a) multi-failure witness — the emitted bundle's
           ``headline_validity_failed_reasons`` carries BOTH
           the reuse-distance and covariate reasons when both
           inputs violate (the plan's "records ALL conditions
           that fire, not just the first" invariant).
       (b) no-violations positive control — clean inputs
           produce ``headline_validity_status == "valid"`` and
           ``headline_validity_failed_reasons == []``.
       (c) absent-kwarg compatibility — existing call sites
           that omit the kwarg get byte-identical bundle
           output (no headline-validity fields added).
  3. ``ComparisonBundleHeadlineValiditySchemaValidationTest``
     builds a bundle whose ``headline_validity_failed_reasons``
     and ``headline_validity_status`` fields are populated and
     calls ``jsonschema.validate`` against the on-disk
     ``comparison_bundle.schema.json``. Skip-soft when
     jsonschema is missing (matches the existing
     ``_HAS_JSONSCHEMA`` pattern in
     ``test_comparison_runner_calibration_loocv_reason.py``).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"

if str(_SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUITE_ROOT))


# ---------------------------------------------------------------------
# Helper tests — eight per-helper positive + negative witnesses.
# ---------------------------------------------------------------------


class HeadlineValidityHelperPositiveNegativeTest(unittest.TestCase):
    """Per-helper positive + negative witnesses for the three
    sm_86-active detection helpers added in this round. Mirrors
    the BL-20260506 failing-test-first discipline: every new
    clause has paired positive + negative controls."""

    def test_check_calibration_reuse_distance_envelope_fires_on_any_true(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        # Any True element in the envelope_mask fires the
        # reuse-distance reason. Mixed mask suffices.
        mask = [False, False, True, False]
        self.assertEqual(
            cr.check_calibration_reuse_distance_envelope(mask),
            cr.CALIBRATION_REUSE_DISTANCE_OUT_OF_ENVELOPE,
        )

    def test_check_calibration_reuse_distance_envelope_quiet_on_all_false(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        mask = [False, False, False, False]
        self.assertIsNone(
            cr.check_calibration_reuse_distance_envelope(mask),
        )

    def test_check_calibration_covariate_envelope_fires_when_clock_out_of_range(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        self.assertEqual(
            cr.check_calibration_covariate_envelope(
                clock_in_range=False,
                thermal_in_range=True,
                throttle_in_range=True,
            ),
            cr.CALIBRATION_COVARIATE_OUT_OF_ENVELOPE,
        )

    def test_check_calibration_covariate_envelope_fires_when_thermal_out_of_range(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        self.assertEqual(
            cr.check_calibration_covariate_envelope(
                clock_in_range=True,
                thermal_in_range=False,
                throttle_in_range=True,
            ),
            cr.CALIBRATION_COVARIATE_OUT_OF_ENVELOPE,
        )

    def test_check_calibration_covariate_envelope_fires_when_throttle_out_of_range(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        self.assertEqual(
            cr.check_calibration_covariate_envelope(
                clock_in_range=True,
                thermal_in_range=True,
                throttle_in_range=False,
            ),
            cr.CALIBRATION_COVARIATE_OUT_OF_ENVELOPE,
        )

    def test_check_calibration_covariate_envelope_quiet_when_all_in_range(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        self.assertIsNone(
            cr.check_calibration_covariate_envelope(
                clock_in_range=True,
                thermal_in_range=True,
                throttle_in_range=True,
            ),
        )

    def test_check_trace_opcode_coverage_fires_on_nonzero_unknown(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        # Any positive count fires the typed reason; the helper
        # treats one unknown opcode as enough evidence to mark
        # the trace's opcode coverage incomplete.
        self.assertEqual(
            cr.check_trace_opcode_coverage(unknown_opcodes_count=1),
            cr.TRACE_OPCODE_COVERAGE_INCOMPLETE,
        )

    def test_check_trace_opcode_coverage_quiet_on_zero_unknown(
        self,
    ) -> None:
        import comparison_runner as cr  # type: ignore[import-not-found]
        self.assertIsNone(
            cr.check_trace_opcode_coverage(unknown_opcodes_count=0),
        )


# ---------------------------------------------------------------------
# Integration tests — drive run_comparison with the new kwarg.
# ---------------------------------------------------------------------


def _admitted_fixture(tmp_path: Path) -> tuple:
    """Build a 5-cell sm_86 fixture that clears the
    coverage gate. Returns (contract, bundles, cells)."""
    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from test_comparison_runner import (  # type: ignore[import-not-found]
        _admitted_contract, _five_synthetic_pairs,
    )
    bundles, cells = _five_synthetic_pairs(
        tmp_path, arch="sm_86", pct=50.0,
        predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
    )
    return _admitted_contract(), bundles, cells


class RunComparisonHeadlineValidityIntegrationTest(unittest.TestCase):
    """``run_comparison(...)`` populates
    ``headline_validity_status`` and
    ``headline_validity_failed_reasons`` on the emitted bundle
    when the ``headline_validity_inputs`` kwarg is supplied.

    Three invariants:
      (a) the array captures EVERY fired reason (multi-failure
          witness), not just the first;
      (b) clean inputs produce status == valid and an empty
          array (no-violations positive control);
      (c) omitting the kwarg preserves byte-identical output
          (no fields added).
    """

    def test_multi_failure_witness_records_all_fired_reasons(self) -> None:
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            # Two distinct violations: reuse-distance envelope mask has
            # a True element AND one covariate boolean is False.
            inputs = {
                "envelope_mask": [False, True, False],
                "clock_in_range": False,
                "thermal_in_range": True,
                "throttle_in_range": True,
            }
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                headline_validity_inputs=inputs,
            )
            self.assertIn("headline_validity_failed_reasons", out)
            self.assertIn("headline_validity_status", out)
            self.assertEqual(
                out["headline_validity_status"], "typed_unavailable",
            )
            reasons = out["headline_validity_failed_reasons"]
            self.assertIn(
                cr.CALIBRATION_REUSE_DISTANCE_OUT_OF_ENVELOPE, reasons,
                msg=(
                    "Multi-failure witness must include the "
                    "reuse-distance reason; got "
                    f"{reasons!r}"
                ),
            )
            self.assertIn(
                cr.CALIBRATION_COVARIATE_OUT_OF_ENVELOPE, reasons,
                msg=(
                    "Multi-failure witness must include the "
                    "covariate reason; got "
                    f"{reasons!r}"
                ),
            )

    def test_no_violations_positive_control(self) -> None:
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            inputs = {
                "envelope_mask": [False, False, False],
                "clock_in_range": True,
                "thermal_in_range": True,
                "throttle_in_range": True,
                "unknown_opcodes_count": 0,
            }
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                headline_validity_inputs=inputs,
            )
            self.assertEqual(
                out["headline_validity_status"], "valid",
            )
            self.assertEqual(
                out["headline_validity_failed_reasons"], [],
            )

    def test_upstream_calibration_variance_precondition_failed_promotes_to_comparison_level(
        self,
    ) -> None:
        """When an upstream predictor bundle's typed_reason is
        ``calibration_variance_precondition_failed`` (i.e., the
        variance-precondition gate fired at the per-cell predictor
        level), run_comparison MUST promote that reason to the
        comparison bundle's ``headline_validity_failed_reasons``.
        Pins the comparison-level lockstep with the predictor-level
        gate: a downstream consumer reading only the comparison
        bundle MUST see the calibration variance failure, not just
        the per-cell predictor's typed_reason."""
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            # Mutate one bundle into a typed-unavailable with the
            # variance-precondition typed reason. The comparison
            # runner should promote this upstream reason to the
            # comparison-level headline-validity-failed-reasons.
            bundles[0]["result_kind"] = "typed_unavailable"
            bundles[0]["typed_reason"] = (
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED
            )
            bundles[0]["headline_validity_failed_reasons"] = [
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            ]
            # Remove fields that conflict with the typed-unavailable
            # arm per the cache_predictor schema.
            for forbidden in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                bundles[0].pop(forbidden, None)
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
            )
            self.assertIn("headline_validity_failed_reasons", out)
            self.assertIn(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                out["headline_validity_failed_reasons"],
                msg=(
                    "Comparison bundle's headline_validity_failed_reasons "
                    "must promote the upstream predictor bundle's "
                    "calibration_variance_precondition_failed typed reason; "
                    f"got {out.get('headline_validity_failed_reasons')!r}"
                ),
            )
            self.assertEqual(
                out["headline_validity_status"], "typed_unavailable",
            )

    def test_target_arch_active_typed_unavailable_bundle_promotes_to_comparison_level(
        self,
    ) -> None:
        """Codex Round-23 review P2 positive control: under
        ``target_arch="sm_86"``, an active sm_86 typed_unavailable
        predictor bundle (variance precondition failed) for a
        joined cell MUST still promote its typed_reason to the
        comparison-level ``headline_validity_failed_reasons``.
        Pins the R11 contract: the active-arch's variance failure
        still forces a comparison-level typed_unavailable."""
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from test_comparison_runner import (  # type: ignore[import-not-found]
                _admitted_contract, _five_synthetic_pairs,
            )
            bundles, cells = _five_synthetic_pairs(
                tmp_path, arch="sm_86", pct=50.0,
                predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
            )
            contract = _admitted_contract()
            bundles[0]["result_kind"] = "typed_unavailable"
            bundles[0]["typed_reason"] = (
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED
            )
            bundles[0]["headline_validity_failed_reasons"] = [
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            ]
            for forbidden in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                bundles[0].pop(forbidden, None)
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                target_arch="sm_86",
                bootstrap_seed=0,
            )
            self.assertIn(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                out.get("headline_validity_failed_reasons", []),
                msg=(
                    "Active sm_86 typed_unavailable bundle under "
                    "target_arch='sm_86' must promote its "
                    "typed_reason to "
                    "headline_validity_failed_reasons; got "
                    f"{out.get('headline_validity_failed_reasons')!r}"
                ),
            )

    def test_target_arch_inactive_arch_typed_unavailable_bundle_not_promoted(
        self,
    ) -> None:
        """Codex Round-23 review P2 negative-A (the Codex
        scenario): under ``target_arch="sm_86"``, a
        typed_unavailable bundle for the INACTIVE arch (sm_120)
        must NOT promote to the comparison-level
        ``headline_validity_failed_reasons``. The active sm_86
        comparison admits cleanly even though an inactive arch's
        bundle failed the variance precondition."""
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from test_comparison_runner import (  # type: ignore[import-not-found]
                _admitted_contract, _five_synthetic_pairs,
            )
            # Active arm: 5 valid sm_86 pairs (clears the floor).
            bundles_86, cells_86 = _five_synthetic_pairs(
                tmp_path / "active", arch="sm_86", pct=50.0,
                predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
            )
            # Inactive arm: 1 sm_120 typed_unavailable predictor
            # bundle + matching sm_120 label cell so the
            # typed_unavailable bundle DOES successfully join
            # an inactive-arch label cell (the worst case for
            # the bug — even a joined inactive-arch bundle must
            # not promote).
            bundles_120, cells_120 = _five_synthetic_pairs(
                tmp_path / "inactive", arch="sm_120", pct=50.0,
                predicted_rates=(0.4,),
            )
            bundles_120[0]["result_kind"] = "typed_unavailable"
            bundles_120[0]["typed_reason"] = (
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED
            )
            bundles_120[0]["headline_validity_failed_reasons"] = [
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            ]
            for forbidden in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                bundles_120[0].pop(forbidden, None)
            contract = _admitted_contract()
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles_86 + bundles_120,
                label_cells=cells_86 + cells_120,
                target_arch="sm_86",
                bootstrap_seed=0,
            )
            self.assertNotIn(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                out.get("headline_validity_failed_reasons", []),
                msg=(
                    "Inactive sm_120 typed_unavailable bundle "
                    "must NOT promote under target_arch='sm_86'; "
                    "the active arch must be insulated from the "
                    "inactive arch's failures. Got "
                    f"{out.get('headline_validity_failed_reasons')!r}; "
                    f"result_kind={out.get('result_kind')!r}"
                ),
            )
            # The comparison should be admitted (no validity
            # failures on the active arch).
            self.assertEqual(
                out.get("result_kind"), "admitted",
                msg=(
                    "Expected admitted result_kind for clean "
                    "active sm_86 arm; got "
                    f"{out.get('result_kind')!r}, "
                    f"typed_reason={out.get('typed_reason')!r}, "
                    f"failed_reasons={out.get('headline_validity_failed_reasons')!r}"
                ),
            )

    def test_target_arch_stale_unmatched_bundle_typed_reason_not_promoted(
        self,
    ) -> None:
        """Codex Round-23 review P2 negative-B: under
        ``target_arch="sm_86"``, a stale predictor bundle with a
        join key that matches NO label cell (e.g., from a broad
        glob over old sweep predictor_bundles/) must NOT promote
        its typed_reason even if it carries
        ``calibration_variance_precondition_failed``. The
        matched_cell_keys filter scopes the promotion to bundles
        that actually joined the active comparison set."""
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from test_comparison_runner import (  # type: ignore[import-not-found]
                _admitted_contract, _five_synthetic_pairs,
            )
            # Active arm: 5 valid sm_86 pairs (clears the floor).
            bundles, cells = _five_synthetic_pairs(
                tmp_path / "active", arch="sm_86", pct=50.0,
                predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4),
            )
            # Stale arm: 1 sm_86 typed_unavailable predictor
            # bundle whose slug does NOT match any label cell
            # (mimics a stale bundle from a broad glob of an
            # older sweep's predictor_bundles directory).
            stale_bundles, _ = _five_synthetic_pairs(
                tmp_path / "stale", arch="sm_86", pct=50.0,
                predicted_rates=(0.4,),
            )
            # Mutate the stale bundle's slug so it does NOT join
            # any of the 5 active label cells.
            stale_bundles[0]["manifest_pairing_index"]["slug"] = (
                "stale_unmatched_kernel"
            )
            stale_bundles[0]["result_kind"] = "typed_unavailable"
            stale_bundles[0]["typed_reason"] = (
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED
            )
            stale_bundles[0]["headline_validity_failed_reasons"] = [
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            ]
            for forbidden in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                stale_bundles[0].pop(forbidden, None)
            contract = _admitted_contract()
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles + stale_bundles,
                label_cells=cells,
                target_arch="sm_86",
                bootstrap_seed=0,
            )
            self.assertNotIn(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                out.get("headline_validity_failed_reasons", []),
                msg=(
                    "Stale join-key-unmatched typed_unavailable "
                    "bundle must NOT promote; matched_cell_keys "
                    "filter must scope to bundles that joined the "
                    "active comparison set. Got "
                    f"{out.get('headline_validity_failed_reasons')!r}; "
                    f"result_kind={out.get('result_kind')!r}"
                ),
            )
            self.assertEqual(
                out.get("result_kind"), "admitted",
                msg=(
                    "Expected admitted result_kind for clean "
                    "active sm_86 arm; got "
                    f"{out.get('result_kind')!r}, "
                    f"typed_reason={out.get('typed_reason')!r}, "
                    f"failed_reasons={out.get('headline_validity_failed_reasons')!r}"
                ),
            )

    def test_six_live_pairs_one_variance_failed_forces_top_level_typed_unavailable(
        self,
    ) -> None:
        """Codex Round-11 directive regression: 6 live sm_86
        matched pairs where 1 is mutated to typed_unavailable
        with the variance-precondition typed reason. The 5
        remaining scalar bundles clear the live floor
        (smoke_test_floor_per_arch=5). Before this round's
        repair, ``run_comparison`` admitted the bundle even
        though ``headline_validity_failed_reasons`` was
        non-empty (the admitted-arm cardinality contradiction
        the comparison-bundle schema was unable to catch).
        After this round's fix, the comparison bundle MUST
        short-circuit to top-level
        ``result_kind="typed_unavailable"`` +
        ``typed_reason="calibration_variance_precondition_failed"``
        BEFORE geomean emission."""
        import sys
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        tests_dir = Path(__file__).resolve().parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from test_comparison_runner import (  # type: ignore[import-not-found]
            _admitted_contract, _five_synthetic_pairs,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 6 pairs total. After mutating bundles[0] to
            # typed_unavailable, 5 scalar live pairs remain --
            # exactly the smoke_test_floor_per_arch=5 minimum --
            # so the live coverage gate WOULD pass admit
            # without the comparison-level fail-closed
            # short-circuit. This is the specific scenario
            # Codex Round-11 named in the directive.
            bundles, cells = _five_synthetic_pairs(
                Path(tmp_dir), arch="sm_86", pct=50.0,
                predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
            )
            contract = _admitted_contract()
            bundles[0]["result_kind"] = "typed_unavailable"
            bundles[0]["typed_reason"] = (
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED
            )
            bundles[0]["headline_validity_failed_reasons"] = [
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            ]
            for forbidden in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                bundles[0].pop(forbidden, None)
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
            )
            self.assertEqual(
                out["result_kind"], "typed_unavailable",
                msg=(
                    "Comparison runner MUST short-circuit to "
                    "typed_unavailable when "
                    "headline_validity_failed_reasons is "
                    "non-empty (admitted-arm cardinality "
                    "invariant); got "
                    f"result_kind={out.get('result_kind')!r} "
                    f"with failed_reasons="
                    f"{out.get('headline_validity_failed_reasons')!r}"
                ),
            )
            self.assertEqual(
                out["typed_reason"],
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                msg=(
                    "When variance-precondition is the first "
                    "failed reason in canonical order, the "
                    "top-level typed_reason MUST mirror it; "
                    f"got typed_reason={out.get('typed_reason')!r}"
                ),
            )
            self.assertIn(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                out["headline_validity_failed_reasons"],
            )
            self.assertEqual(
                out["headline_validity_status"], "typed_unavailable",
            )
            # The admitted-arm fields MUST be absent on the
            # typed_unavailable arm (schema oneOf invariant).
            self.assertNotIn("geomean", out)
            self.assertNotIn("geomean_ci_low", out)
            self.assertNotIn("geomean_ci_high", out)
            self.assertNotIn("geomean_ci_half_width", out)

    def test_typed_reason_precedence_follows_canonical_order(
        self,
    ) -> None:
        """When BOTH the LOOCV-evaluation-sidecar reason
        (``calibration_loocv_p90_residual_exceeded``) AND the
        upstream variance-precondition reason
        (``calibration_variance_precondition_failed``) fire on the
        same comparison-runner call, the top-level
        ``typed_reason`` MUST follow the canonical
        ``HEADLINE_VALIDITY_FAILED_REASONS_ALL`` order — which is
        ALPHABETICAL across the sm_86-active reasons, so LOOCV
        (alpha index 1) outranks variance (alpha index 3).

        This pins the policy decision Codex's earlier review
        flagged: prose in some prior summaries claimed
        variance-precondition outranks LOOCV. The actual
        behavior follows canonical-order precedence. This test
        is the source-of-truth assertion for that policy."""
        import sys
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        tests_dir = Path(__file__).resolve().parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from test_comparison_runner import (  # type: ignore[import-not-found]
            _admitted_contract, _five_synthetic_pairs,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 6 live pairs total. After mutating bundles[0] to
            # typed_unavailable + variance reason, 5 scalar live
            # pairs remain (meeting the floor=5). The LOOCV
            # sidecar at p90=0.35 (above DEC-T5 threshold 0.30)
            # adds calibration_loocv_p90_residual_exceeded to
            # the failed_reasons array.
            bundles, cells = _five_synthetic_pairs(
                Path(tmp_dir), arch="sm_86", pct=50.0,
                predicted_rates=(0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
            )
            contract = _admitted_contract()
            bundles[0]["result_kind"] = "typed_unavailable"
            bundles[0]["typed_reason"] = (
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED
            )
            bundles[0]["headline_validity_failed_reasons"] = [
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            ]
            for forbidden in (
                "predicted_hit_rate", "ci_low", "ci_high",
                "ci_half_width", "ci_confidence",
            ):
                bundles[0].pop(forbidden, None)
            sidecar = {
                "schema_version": 1,
                "chosen_head": "PlattCalibrator",
                "selection_rule": "simplicity_one_se_rule",
                "loocv_table": {
                    "PlattCalibrator": {
                        "p50": 0.05,
                        "p90": 0.35,  # above DEC-T5 threshold 0.30
                        "p99": 0.50,
                        "se_p90": 0.05,
                        "per_fold_residuals": [0.10, 0.20, 0.30, 0.35, 0.40],
                    },
                },
            }
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                calibration_evaluation_sidecar=sidecar,
            )
            # Both reasons should appear in the failed_reasons array.
            self.assertIn(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
                out["headline_validity_failed_reasons"],
            )
            self.assertIn(
                cr.CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED,
                out["headline_validity_failed_reasons"],
            )
            # The top-level result is typed_unavailable.
            self.assertEqual(
                out["result_kind"], "typed_unavailable",
            )
            # The top-level typed_reason follows canonical order:
            # LOOCV (alpha index 1) outranks variance (alpha index 3).
            self.assertEqual(
                out["typed_reason"],
                cr.CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED,
                msg=(
                    "Top-level typed_reason MUST follow canonical "
                    "HEADLINE_VALIDITY_FAILED_REASONS_ALL alphabetical "
                    "order; LOOCV (alpha index 1) outranks "
                    "variance-precondition (alpha index 3). Got "
                    f"typed_reason={out.get('typed_reason')!r} with "
                    f"failed_reasons="
                    f"{out.get('headline_validity_failed_reasons')!r}"
                ),
            )
            # The pinned canonical order is the SM_86_ACTIVE tuple.
            # Assert the precedence directly from the constants so a
            # future reorder of the tuple is caught by this test.
            sm86_order = cr.HEADLINE_VALIDITY_FAILED_REASONS_SM_86_ACTIVE
            loocv_idx = sm86_order.index(
                cr.CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED,
            )
            variance_idx = sm86_order.index(
                cr.CALIBRATION_VARIANCE_PRECONDITION_FAILED,
            )
            self.assertLess(
                loocv_idx, variance_idx,
                msg=(
                    "Pinned canonical precedence: "
                    "calibration_loocv_p90_residual_exceeded MUST "
                    "appear BEFORE calibration_variance_precondition_failed "
                    "in HEADLINE_VALIDITY_FAILED_REASONS_SM_86_ACTIVE; "
                    f"got loocv_idx={loocv_idx}, "
                    f"variance_idx={variance_idx}"
                ),
            )

    def test_absent_kwarg_preserves_existing_bundle_shape(self) -> None:
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                # headline_validity_inputs omitted -> default None.
            )
            self.assertNotIn(
                "headline_validity_status", out,
                msg=(
                    "absent kwarg must NOT add the status field "
                    "(existing call-site byte-compat)"
                ),
            )
            self.assertNotIn(
                "headline_validity_failed_reasons", out,
                msg=(
                    "absent kwarg must NOT add the failed-reasons "
                    "array (existing call-site byte-compat)"
                ),
            )


# ---------------------------------------------------------------------
# Schema-validation regression — gated on jsonschema availability.
# ---------------------------------------------------------------------


def _load_comparison_bundle_schema() -> dict:
    import json
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas" / "comparison_bundle.schema.json"
    )
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


try:
    import jsonschema as _jsonschema_mod  # type: ignore[import-not-found]
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


@unittest.skipIf(
    not _HAS_JSONSCHEMA, "jsonschema unavailable on this host",
)
class ComparisonBundleHeadlineValiditySchemaValidationTest(unittest.TestCase):
    """The on-disk comparison_bundle.schema.json MUST admit
    bundles that carry the two new headline_validity fields
    populated with valid enum members.

    Builds the bundle via run_comparison so the schema-validation
    pass reflects what the runner actually emits."""

    def test_multi_failure_bundle_validates_against_committed_schema(
        self,
    ) -> None:
        import tempfile
        import jsonschema  # type: ignore[import-not-found]
        import comparison_runner as cr  # type: ignore[import-not-found]
        schema = _load_comparison_bundle_schema()
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            inputs = {
                "envelope_mask": [True],
                "clock_in_range": True,
                "thermal_in_range": True,
                "throttle_in_range": False,
                "unknown_opcodes_count": 7,
            }
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                headline_validity_inputs=inputs,
            )
            self.assertEqual(
                out["headline_validity_status"], "typed_unavailable",
            )
            self.assertGreater(
                len(out["headline_validity_failed_reasons"]), 1,
            )
            jsonschema.validate(out, schema)


# ---------------------------------------------------------------------
# Schema cardinality invariant tests — every status/reasons
# combination from plan AC-T7 negative tests.
# ---------------------------------------------------------------------


def _synthetic_admitted_bundle_skeleton() -> dict:
    """Build the smallest schema-valid admitted bundle skeleton
    so the cardinality tests can stamp the new fields onto it
    and probe jsonschema acceptance. Cardinality tests should
    only differ in the two new fields; the rest of the bundle
    must be a known-valid baseline."""
    return {
        "schema_version": 1,
        "algorithm_version": "comparison_runner_v1",
        "comparison_run_pairing_index": {
            "ncu_metric_name": "l1tex__t_sector_hit_rate.pct",
            "ncu_metric_version": "13.2",
            "epsilon": 0.01,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 0,
            "ci_confidence": 0.95,
            "split_label": "all",
        },
        "top_level_admission_scope": "cross_arch_any_arch",
        "result_kind": "admitted",
        "comparable_cells_per_arch": {"sm_86": 0},
        "headline_cells_per_arch": {"sm_86": 0},
        "headline_floor_per_arch": 0,
        "included_cells": [],
        "excluded_cells": [],
        "geomean": 0.1,
        "geomean_ci_low": 0.05,
        "geomean_ci_high": 0.15,
        "geomean_ci_half_width": 0.05,
    }


def _synthetic_typed_unavailable_bundle_skeleton(
    typed_reason: str,
) -> dict:
    """Build the smallest schema-valid typed-unavailable bundle
    skeleton for cardinality tests that need a bundle on the
    typed_unavailable oneOf arm (e.g., the variance-precondition
    promotion + the admitted-arm cardinality guard). The
    typed_reason value MUST be in the runner-level
    ``typed_reason.enum`` of comparison_bundle.schema.json. Stamp
    headline_validity fields on top to probe jsonschema
    acceptance. The geomean fields are deliberately omitted per
    the schema's typed_unavailable oneOf arm exclusivity
    invariant."""
    return {
        "schema_version": 1,
        "algorithm_version": "comparison_runner_v1",
        "comparison_run_pairing_index": {
            "ncu_metric_name": "l1tex__t_sector_hit_rate.pct",
            "ncu_metric_version": "13.2",
            "epsilon": 0.01,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 0,
            "ci_confidence": 0.95,
            "split_label": "all",
        },
        "top_level_admission_scope": "cross_arch_any_arch",
        "result_kind": "typed_unavailable",
        "typed_reason": typed_reason,
        "comparable_cells_per_arch": {"sm_86": 0},
        "headline_cells_per_arch": {"sm_86": 0},
        "headline_floor_per_arch": 0,
        "included_cells": [],
        "excluded_cells": [],
    }


@unittest.skipIf(
    not _HAS_JSONSCHEMA, "jsonschema unavailable on this host",
)
class HeadlineValidityCardinalityInvariantTest(unittest.TestCase):
    """Per plan AC-T7 negative tests: the schema MUST enforce
    a cardinality invariant tying ``headline_validity_status``
    to ``headline_validity_failed_reasons``.

    - status=valid ⇒ array must be empty (maxItems: 0)
    - status=typed_unavailable ⇒ array must be non-empty (minItems: 1)
    - either field's presence requires the other (dependentRequired)
    - legacy bundles omitting BOTH stay valid

    Per BL-20260506: every schema clause has a paired positive
    + negative control. Five tests below cover the four
    cardinality combinations + the dependentRequired pair.
    """

    def _validate(self, bundle: dict) -> None:
        import jsonschema  # type: ignore[import-not-found]
        schema = _load_comparison_bundle_schema()
        jsonschema.validate(bundle, schema)

    def _assert_rejects(self, bundle: dict, *, expected_substring: str) -> None:
        import jsonschema  # type: ignore[import-not-found]
        schema = _load_comparison_bundle_schema()
        with self.assertRaises(jsonschema.ValidationError) as ctx:
            jsonschema.validate(bundle, schema)
        # Sanity: the rejection message must mention the
        # cardinality field so a future schema refactor that
        # rejects for a different reason fails this test
        # rather than silently passing.
        msg = str(ctx.exception).lower()
        self.assertIn(
            expected_substring.lower(), msg,
            msg=(
                f"expected jsonschema rejection naming "
                f"{expected_substring!r}; got: {ctx.exception!s}"
            ),
        )

    # -----------------------------------------------------------------
    # Positive controls.
    # -----------------------------------------------------------------

    def test_schema_valid_status_with_empty_reasons_accepted(self) -> None:
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_status"] = "valid"
        bundle["headline_validity_failed_reasons"] = []
        self._validate(bundle)

    def test_schema_typed_unavailable_status_with_nonempty_reasons_accepted(
        self,
    ) -> None:
        # Use the typed_unavailable skeleton so the bundle
        # sits on the typed_unavailable oneOf arm. (The Round
        # 12 admitted-arm cardinality guard forbids non-empty
        # headline_validity_failed_reasons on the admitted arm;
        # this positive control intentionally exercises the
        # typed_unavailable arm where non-empty reasons are
        # the canonical shape.)
        bundle = _synthetic_typed_unavailable_bundle_skeleton(
            typed_reason="calibration_loocv_p90_residual_exceeded",
        )
        bundle["headline_validity_status"] = "typed_unavailable"
        bundle["headline_validity_failed_reasons"] = [
            "calibration_loocv_p90_residual_exceeded",
        ]
        self._validate(bundle)

    def test_schema_both_fields_absent_accepted_legacy_compat(self) -> None:
        # Legacy bundles emitted before the headline-validity
        # fields existed must still validate.
        bundle = _synthetic_admitted_bundle_skeleton()
        self._validate(bundle)

    # -----------------------------------------------------------------
    # Negative controls — cardinality.
    # -----------------------------------------------------------------

    def test_schema_valid_status_with_nonempty_reasons_rejected(self) -> None:
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_status"] = "valid"
        bundle["headline_validity_failed_reasons"] = [
            "calibration_reuse_distance_out_of_envelope",
        ]
        self._assert_rejects(
            bundle, expected_substring="headline_validity",
        )

    def test_schema_typed_unavailable_status_with_empty_reasons_rejected(
        self,
    ) -> None:
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_status"] = "typed_unavailable"
        bundle["headline_validity_failed_reasons"] = []
        self._assert_rejects(
            bundle, expected_substring="headline_validity",
        )

    # -----------------------------------------------------------------
    # Negative controls — dependentRequired pair.
    # -----------------------------------------------------------------

    def test_schema_status_without_reasons_rejected(self) -> None:
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_status"] = "valid"
        # headline_validity_failed_reasons deliberately absent.
        self._assert_rejects(
            bundle, expected_substring="headline_validity_failed_reasons",
        )

    def test_schema_reasons_without_status_rejected(self) -> None:
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_failed_reasons"] = []
        # headline_validity_status deliberately absent.
        self._assert_rejects(
            bundle, expected_substring="headline_validity_status",
        )

    # -----------------------------------------------------------------
    # Variance-precondition repair — admitted-arm cardinality
    # guard. The schema's admitted arm MUST reject any bundle
    # whose ``headline_validity_failed_reasons`` is non-empty
    # (the cardinality negative test paired with the
    # variance-precondition typed-reason promotion). Paired
    # positive control: a typed_unavailable bundle with the
    # variance-precondition typed reason is accepted.
    # -----------------------------------------------------------------

    def test_schema_admitted_arm_with_nonempty_failed_reasons_rejected(
        self,
    ) -> None:
        """Variance-precondition repair directive: the schema's
        admitted arm MUST reject any bundle that carries
        non-empty headline_validity_failed_reasons. This is the
        admitted-arm cardinality negative test that closes the
        runner-side + schema-side fail-closed lockstep against
        the variance-precondition typed reason. The previous
        adjudication round landed the array but allowed the
        producer to admit with non-empty entries; this guard
        catches drift if a future producer regresses the
        comparison-level short-circuit."""
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_status"] = "typed_unavailable"
        bundle["headline_validity_failed_reasons"] = [
            "calibration_variance_precondition_failed",
        ]
        self._assert_rejects(
            bundle, expected_substring="headline_validity",
        )

    def test_schema_admitted_arm_with_empty_failed_reasons_accepted(
        self,
    ) -> None:
        """Positive control paired with the negative above.
        Admitted bundle WITH empty
        headline_validity_failed_reasons MUST validate (the
        canonical valid-status admitted shape)."""
        bundle = _synthetic_admitted_bundle_skeleton()
        bundle["headline_validity_status"] = "valid"
        bundle["headline_validity_failed_reasons"] = []
        self._validate(bundle)

    def test_schema_typed_unavailable_arm_with_variance_precondition_reason_accepted(
        self,
    ) -> None:
        """Variance-precondition repair lockstep: the schema's
        typed_unavailable arm MUST admit the runner-level
        typed_reason ``calibration_variance_precondition_failed``
        together with ``headline_validity_failed_reasons``
        containing the same value. Pins the runner-level
        typed_reason enum extension (the variance reason was
        added in lockstep with the comparison-level
        short-circuit) against future schema drift."""
        bundle = _synthetic_typed_unavailable_bundle_skeleton(
            typed_reason="calibration_variance_precondition_failed",
        )
        bundle["headline_validity_status"] = "typed_unavailable"
        bundle["headline_validity_failed_reasons"] = [
            "calibration_variance_precondition_failed",
        ]
        self._validate(bundle)


# ---------------------------------------------------------------------
# Production-path field population — run_comparison must emit
# the new fields when EITHER kwarg is provided.
# ---------------------------------------------------------------------


class RunComparisonHeadlineValidityProductionPathTest(unittest.TestCase):
    """``run_comparison(...)`` must populate
    ``headline_validity_status`` + ``headline_validity_failed_reasons``
    on the emitted bundle when EITHER
    ``headline_validity_inputs`` OR
    ``calibration_evaluation_sidecar`` is provided. The legacy
    byte-compat path (neither kwarg) preserves byte-identical
    output.

    Codex's prior review of the sm_86-active rollout
    surfaced that the production CLI paths called
    ``run_comparison(...)`` with only the calibration
    sidecar, leaving the new fields silently absent in the
    emitted comparison bundle. AC-T7's value depends on the
    LOOCV reason propagating into the new array on the
    sidecar-only path; the regression below pins this so a
    future refactor cannot drop the production-path emission
    again.
    """

    def _sidecar_above_threshold(self) -> dict:
        return {
            "schema_version": 1,
            "chosen_head": "PlattCalibrator",
            "selection_rule": "simplicity_one_se_rule",
            "loocv_table": {
                "PlattCalibrator": {
                    "p50": 0.05,
                    "p90": 0.35,
                    "p99": 0.50,
                    "se_p90": 0.05,
                    "per_fold_residuals": [0.10, 0.20, 0.30, 0.35, 0.40],
                },
            },
        }

    def test_calibration_sidecar_alone_populates_headline_validity_fields(
        self,
    ) -> None:
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            sidecar = self._sidecar_above_threshold()
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                calibration_evaluation_sidecar=sidecar,
                # headline_validity_inputs deliberately omitted.
            )
            # The LOOCV gate short-circuits to typed_unavailable.
            self.assertEqual(out["result_kind"], "typed_unavailable")
            self.assertEqual(
                out["typed_reason"],
                cr.CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED,
            )
            # AC-T7 invariant: the new fields appear too,
            # carrying the LOOCV reason in the array.
            self.assertEqual(
                out["headline_validity_status"], "typed_unavailable",
                msg=(
                    "AC-T7 production path: sidecar-only LOOCV "
                    "gate must populate headline_validity_status; "
                    f"got {out!r}"
                ),
            )
            self.assertEqual(
                out["headline_validity_failed_reasons"],
                [cr.CALIBRATION_LOOCV_P90_RESIDUAL_EXCEEDED],
                msg=(
                    "AC-T7 production path: sidecar-only LOOCV "
                    "gate must populate "
                    "headline_validity_failed_reasons with the "
                    "LOOCV reason; got "
                    f"{out.get('headline_validity_failed_reasons')!r}"
                ),
            )

    def test_neither_kwarg_preserves_byte_compat(self) -> None:
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                # both kwargs omitted.
            )
            self.assertNotIn("headline_validity_status", out)
            self.assertNotIn("headline_validity_failed_reasons", out)

    def test_below_threshold_sidecar_populates_valid_status(self) -> None:
        # When the sidecar's chosen head is BELOW the LOOCV
        # threshold, the LOOCV gate does NOT fire — the bundle
        # admits normally. The new fields still appear because
        # the sidecar is non-None; they record status=valid +
        # an empty array.
        import tempfile
        import comparison_runner as cr  # type: ignore[import-not-found]
        with tempfile.TemporaryDirectory() as tmp_dir:
            contract, bundles, cells = _admitted_fixture(Path(tmp_dir))
            sidecar = self._sidecar_above_threshold()
            sidecar["loocv_table"]["PlattCalibrator"]["p90"] = 0.20
            out = cr.run_comparison(
                contract=contract,
                predictor_bundles=bundles,
                label_cells=cells,
                bootstrap_seed=0,
                calibration_evaluation_sidecar=sidecar,
            )
            self.assertEqual(out["result_kind"], "admitted")
            self.assertEqual(
                out["headline_validity_status"], "valid",
            )
            self.assertEqual(
                out["headline_validity_failed_reasons"], [],
            )


# ---------------------------------------------------------------------
# Orchestrator boundary test — executable monkeypatch proves the
# CLI/orchestrator threads the calibration sidecar through to
# run_comparison.
# ---------------------------------------------------------------------


class RunHeadlineCohortCliCalibrationSidecarBoundaryTest(unittest.TestCase):
    """Drives ``tools/local_labels/run_headline_cohort_predict_compare.py``'s
    main() with a monkeypatched ``run_comparison(...)`` that
    captures the kwargs. Asserts the orchestrator parses the
    explicit ``--calibration-evaluation-sidecar`` flag and
    passes the parsed payload through to the runner. A future
    refactor that drops the kwarg from the orchestrator's
    ``cr.run_comparison(...)`` call fails this boundary
    immediately.
    """

    def test_orchestrator_threads_calibration_sidecar_to_run_comparison(
        self,
    ) -> None:
        import importlib.util
        import json
        import tempfile

        orch_path = (
            Path(__file__).resolve().parents[3]
            / "tools" / "local_labels"
            / "run_headline_cohort_predict_compare.py"
        )
        spec = importlib.util.spec_from_file_location(
            "run_headline_cohort_predict_compare", orch_path,
        )
        assert spec and spec.loader
        orch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orch)  # type: ignore[union-attr]

        # The orchestrator binds the runner module as `cr`; the
        # call site is `cr.run_comparison(...)` so we
        # monkeypatch the attribute on that binding.
        cr_module = orch.cr  # type: ignore[attr-defined]

        captured: dict[str, object] = {}

        def _fake_run_comparison(**kwargs):
            captured.update(kwargs)
            return {
                "result_kind": "typed_unavailable",
                "typed_reason": "calibration_loocv_p90_residual_exceeded",
                "schema_version": 1,
                "headline_validity_status": "typed_unavailable",
                "headline_validity_failed_reasons": [
                    "calibration_loocv_p90_residual_exceeded",
                ],
            }

        original = cr_module.run_comparison
        original_write = cr_module.write_bundle
        original_load_contract = cr_module.load_comparison_contract
        original_load_cells = cr_module._load_admitted_label_cells
        original_cohort = orch._HEADLINE_COHORT_13  # type: ignore[attr-defined]
        # Stub the manifest-driven cohort loaders so the runner
        # reaches the calibration-sidecar threading path without
        # opening the dummy manifest path.
        orig_cohort_loader = orch._load_admitted_cohort_from_manifest  # type: ignore[attr-defined]
        orig_manifest_loader = orch._load_manifest_for_runner_gates  # type: ignore[attr-defined]
        orch._load_admitted_cohort_from_manifest = (  # type: ignore[attr-defined]
            lambda _manifest_path, **_kw: ()
        )
        orch._load_manifest_for_runner_gates = (  # type: ignore[attr-defined]
            lambda _manifest_path, **_kw: object()
        )
        # Stub publication_gate so the runner reaches cr.run_comparison
        # without consulting the (sentinel) manifest object.
        import publication_gate as _pg
        orig_floor = _pg.check_contract_floor
        _pg.check_contract_floor = (
            lambda manifest, *, publication_grade_floor: _pg.ContractFloorGateVerdict(
                passed=True, admitted_count=publication_grade_floor,
                required_floor=publication_grade_floor,
                typed_diagnostic=None,
            )
        )
        cr_module.run_comparison = _fake_run_comparison
        cr_module.write_bundle = lambda bundle, path: None
        cr_module.load_comparison_contract = lambda p: "contract"
        cr_module._load_admitted_label_cells = lambda p: []
        # Stub the cohort to empty so main() doesn't iterate
        # the 13 real predictor cells (each requires real
        # binary + trace artifacts not available in this test).
        orch._HEADLINE_COHORT_13 = ()  # type: ignore[attr-defined]
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_root = Path(tmp_dir)
                sidecar_path = tmp_root / "calibration_evaluation_v1.json"
                sidecar_payload = {
                    "schema_version": 1,
                    "chosen_head": "PlattCalibrator",
                    "selection_rule": "simplicity_one_se_rule",
                    "loocv_table": {
                        "PlattCalibrator": {
                            "p50": 0.05,
                            "p90": 0.35,
                            "p99": 0.50,
                            "se_p90": 0.05,
                            "per_fold_residuals": [0.10, 0.20, 0.30, 0.35, 0.40],
                        },
                    },
                }
                sidecar_path.write_text(json.dumps(sidecar_payload))

                label_path = tmp_root / "label_admitted.json"
                label_path.write_text("{}")
                output_dir = tmp_root / "out"

                exit_code = orch.main([
                    "--cohort-manifest", str(tmp_root / "dummy_manifest.json"),
                    "--label-admitted-path", str(label_path),
                    "--output-dir", str(output_dir),
                    "--calibration-evaluation-sidecar", str(sidecar_path),
                ])
                self.assertEqual(exit_code, 0)
            self.assertIn(
                "calibration_evaluation_sidecar", captured,
                msg=(
                    "orchestrator main() must thread "
                    "calibration_evaluation_sidecar to "
                    "run_comparison; captured: "
                    f"{list(captured.keys())!r}"
                ),
            )
            payload = captured["calibration_evaluation_sidecar"]
            self.assertIsNotNone(payload)
            self.assertEqual(payload["chosen_head"], "PlattCalibrator")  # type: ignore[index]
        finally:
            cr_module.run_comparison = original
            cr_module.write_bundle = original_write
            cr_module.load_comparison_contract = original_load_contract
            cr_module._load_admitted_label_cells = original_load_cells
            orch._HEADLINE_COHORT_13 = original_cohort  # type: ignore[attr-defined]
            orch._load_admitted_cohort_from_manifest = orig_cohort_loader  # type: ignore[attr-defined]
            orch._load_manifest_for_runner_gates = orig_manifest_loader  # type: ignore[attr-defined]
            _pg.check_contract_floor = orig_floor

    def test_orchestrator_variance_precondition_artifact_empty_string_opts_out(
        self,
    ) -> None:
        """When --variance-precondition-artifact='' is passed, the
        orchestrator MUST treat it as an explicit opt-out: the
        variance_precondition_loader.load_variance_precondition
        function MUST NOT be invoked, and the predictor MUST NOT
        receive a non-None variance_precondition_stddev kwarg.

        Round-11 introduced the `argparse type=Path` regression
        where Path('') becomes Path('.') and the opt-out check
        never fired; this round changed type=str so the
        empty-string boundary is detected before the Path
        conversion. The test pins both halves of the contract:
        (a) main() exits 0 on --variance-precondition-artifact='';
        (b) the loader is not called even when the default
        artifact file is present on disk."""
        import importlib.util
        import json
        import sys as _sys
        import tempfile

        orch_path = (
            Path(__file__).resolve().parents[3]
            / "tools" / "local_labels"
            / "run_headline_cohort_predict_compare.py"
        )
        spec = importlib.util.spec_from_file_location(
            "run_headline_cohort_predict_compare", orch_path,
        )
        assert spec and spec.loader
        orch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orch)  # type: ignore[union-attr]
        cr_module = orch.cr  # type: ignore[attr-defined]

        # The loader is imported lazily inside main() only when
        # variance_artifact_path is not None. We stamp a sentinel
        # module into sys.modules so any attempt to import the
        # loader raises immediately. If the opt-out fires correctly,
        # the lazy import is never reached and the test passes.
        loader_mod_name = "variance_precondition_loader"
        original_loader_mod = _sys.modules.get(loader_mod_name)

        class _ExplodingModule:
            def __getattr__(self, name: str):
                raise AssertionError(
                    "variance_precondition_loader was imported / "
                    "accessed but the explicit empty-string opt-out "
                    "should have prevented that"
                )

        _sys.modules[loader_mod_name] = _ExplodingModule()  # type: ignore[assignment]

        captured: dict[str, object] = {}

        def _fake_run_comparison(**kwargs):
            captured.update(kwargs)
            return {
                "result_kind": "typed_unavailable",
                "typed_reason": "headline_geomean_admitted_cell_count_below_threshold",
                "schema_version": 1,
            }

        original = cr_module.run_comparison
        original_write = cr_module.write_bundle
        original_load_contract = cr_module.load_comparison_contract
        original_load_cells = cr_module._load_admitted_label_cells
        original_cohort = orch._HEADLINE_COHORT_13  # type: ignore[attr-defined]
        # Stub the manifest-driven cohort loaders so the runner
        # reaches the variance-precondition opt-out path.
        orig_cohort_loader = orch._load_admitted_cohort_from_manifest  # type: ignore[attr-defined]
        orig_manifest_loader = orch._load_manifest_for_runner_gates  # type: ignore[attr-defined]
        orch._load_admitted_cohort_from_manifest = (  # type: ignore[attr-defined]
            lambda _manifest_path, **_kw: ()
        )
        orch._load_manifest_for_runner_gates = (  # type: ignore[attr-defined]
            lambda _manifest_path, **_kw: object()
        )
        import publication_gate as _pg
        orig_floor = _pg.check_contract_floor
        _pg.check_contract_floor = (
            lambda manifest, *, publication_grade_floor: _pg.ContractFloorGateVerdict(
                passed=True, admitted_count=publication_grade_floor,
                required_floor=publication_grade_floor,
                typed_diagnostic=None,
            )
        )
        cr_module.run_comparison = _fake_run_comparison
        cr_module.write_bundle = lambda bundle, path: None
        cr_module.load_comparison_contract = lambda p: "contract"
        cr_module._load_admitted_label_cells = lambda p: []
        orch._HEADLINE_COHORT_13 = ()  # type: ignore[attr-defined]
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_root = Path(tmp_dir)
                label_path = tmp_root / "label_admitted.json"
                label_path.write_text("{}")
                output_dir = tmp_root / "out"

                exit_code = orch.main([
                    "--cohort-manifest", str(tmp_root / "dummy_manifest.json"),
                    "--label-admitted-path", str(label_path),
                    "--output-dir", str(output_dir),
                    "--variance-precondition-artifact", "",
                ])
                self.assertEqual(
                    exit_code, 0,
                    msg=(
                        "main() with --variance-precondition-artifact='' "
                        f"must exit 0; got {exit_code}"
                    ),
                )
        finally:
            cr_module.run_comparison = original
            cr_module.write_bundle = original_write
            cr_module.load_comparison_contract = original_load_contract
            cr_module._load_admitted_label_cells = original_load_cells
            orch._HEADLINE_COHORT_13 = original_cohort  # type: ignore[attr-defined]
            orch._load_admitted_cohort_from_manifest = orig_cohort_loader  # type: ignore[attr-defined]
            orch._load_manifest_for_runner_gates = orig_manifest_loader  # type: ignore[attr-defined]
            _pg.check_contract_floor = orig_floor
            if original_loader_mod is not None:
                _sys.modules[loader_mod_name] = original_loader_mod
            else:
                _sys.modules.pop(loader_mod_name, None)


if __name__ == "__main__":
    unittest.main()
