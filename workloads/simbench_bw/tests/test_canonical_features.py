"""Tests for the canonical pre-calibration feature assembler + active path.

Covers the canonical-features contract positive + negative for the monotonic calibration head's wire
contract:

  - Schema admits the canonical_pre_calibration_features block when present;
    rejects bundles where the block is present but missing any of the 6 fields;
    rejects bundles whose footprint_ratio is outside [0, 100] or whose
    fraction-domain features are outside [0, 1].
  - Schema continues to admit historical bundles that lack the block
    (additive widening per the wire-extension policy + the lockstep
    bitlesson BL-20260526-schema-additionalproperties-false-requires-lockstep
    + BL-20260527-optional-schema-extension-not-enforcement).
  - The 30 committed sm_120 bundles continue to validate after the schema
    additions (real-artifact regression per
    BL-20260527-test-fixture-shape-must-mirror-real-shape).
  - assemble_canonical_features computes each of the 6 features from the
    legacy sidecar shapes (sharing_locality / hrd_profile / sector_stream
    record iterable) byte-equivalently to the existing
    test_l2_high_footprint_correction._ac5_read_per_cell_evidence helper.
  - The active path's predict_active_path entry rejects per-slug factor
    kwargs (factor / factor_override / per_slug_factor /
    l2_reduction_factor / l2_high_footprint_factor / gridDim_penalty_alpha)
    with ActivePathDomainMismatchError.
  - The active path rejects bundles whose canonical_pre_calibration_features
    block is absent or incomplete with ValueError.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jsonschema

from workloads.simbench_bw import canonical_features as cf


_SCHEMA_PATH = (
    _REPO_ROOT / "workloads/simbench_bw/schemas/cache_predictor.schema.json"
)
_SM120_BUNDLES_DIR = (
    _REPO_ROOT
    / "data/local_labels/sweeps/sm120_cohort_bundles_20260526T154829Z"
    / "predictor_bundles"
)


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _minimal_scalar_bundle(**overrides: Any) -> dict:
    """Build a minimal schema-valid scalar bundle suitable for negative-control
    mutation. The bundle's shape mirrors the real producer output (top-level
    cache_layer; manifest_pairing_index identity; input_sidecar_digests.sector_stream
    present) so consumer-side gates do not over-fire on test fixtures."""
    bundle = {
        "schema_version": 1,
        "algorithm_version": "smith_eklund_lru_v1",
        "manifest_pairing_index": {
            "slug": "test_slug",
            "shape_point": "test_shape",
            "binary_sha256": "0" * 64,
            "source_commit": "b6372fddd6078896b831cbecc0cf1be8724c2ecb",
            "kernel_signature_sha256": "1" * 64,
        },
        "input_sidecar_digests": {
            "sector_stream": {
                "path": "trace_test/sector_stream/sector_stream.jsonl",
                "sha256": "a" * 64,
            },
        },
        "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "cache_model_capacity_bytes": 33554432,
        "sector_size_bytes": 32,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 0,
        "epsilon": 0.01,
        "result_kind": "scalar",
        "cache_layer": "L2",
        "predicted_hit_rate": 0.5,
        "ci_low": 0.45,
        "ci_high": 0.55,
        "ci_half_width": 0.05,
        "ci_confidence": 0.95,
        "evidence": {
            "predicted_hit_rate_uncorrected": 0.5,
        },
    }
    bundle.update(overrides)
    return bundle


def _full_feature_block() -> dict[str, float]:
    """A schema-valid feature block with all 6 fields in their canonical ranges."""
    return {
        "raw_prediction": 0.5,
        "footprint_ratio": 2.5,
        "long_rd_tail_mass": 0.1,
        "intra_lane_reuse": 0.7,
        "global_share": 1.0,
        "gridDim_density": 4.0,
    }


class SchemaAdmitsHistoricalBundleTest(unittest.TestCase):
    """The schema's additive widening for canonical_pre_calibration_features
    MUST remain backwards-compatible with historical bundles that lack the
    block."""

    def test_minimal_bundle_without_canonical_block_admits(self) -> None:
        schema = _load_schema()
        bundle = _minimal_scalar_bundle()
        self.assertNotIn("canonical_pre_calibration_features", bundle)
        jsonschema.validate(instance=bundle, schema=schema)

    def test_committed_sm120_bundles_still_validate(self) -> None:
        """Real-artifact regression per
        BL-20260527-test-fixture-shape-must-mirror-real-shape: the 30
        committed sm_120 predictor bundles MUST continue to validate against
        the extended schema."""
        if not _SM120_BUNDLES_DIR.is_dir():
            self.skipTest(
                "sm120 cohort bundles directory absent on this host"
            )
        bundle_paths = sorted(_SM120_BUNDLES_DIR.glob("sm_120__*.json"))
        if not bundle_paths:
            self.skipTest("no sm_120 bundles under the cohort dir")
        schema = _load_schema()
        for path in bundle_paths:
            with self.subTest(path=path.name):
                bundle = json.loads(path.read_text(encoding="utf-8"))
                jsonschema.validate(instance=bundle, schema=schema)


class SchemaCanonicalFeaturesBlockTest(unittest.TestCase):
    """Positive + negative tests for the canonical_pre_calibration_features
    schema arm."""

    def test_bundle_with_full_canonical_block_admits(self) -> None:
        schema = _load_schema()
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = _full_feature_block()
        jsonschema.validate(instance=bundle, schema=schema)

    def test_bundle_missing_one_canonical_feature_fails(self) -> None:
        schema = _load_schema()
        for missing_feature in cf.CANONICAL_FEATURE_NAMES:
            with self.subTest(missing_feature=missing_feature):
                features = _full_feature_block()
                del features[missing_feature]
                bundle = _minimal_scalar_bundle()
                bundle["canonical_pre_calibration_features"] = features
                with self.assertRaises(jsonschema.ValidationError) as cm:
                    jsonschema.validate(instance=bundle, schema=schema)
                # Diagnostic should name the missing field.
                self.assertIn(missing_feature, str(cm.exception))

    def test_bundle_with_unexpected_canonical_feature_fails(self) -> None:
        schema = _load_schema()
        features = _full_feature_block()
        features["unexpected_extra"] = 0.5
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bundle, schema=schema)

    def test_footprint_ratio_negative_fails(self) -> None:
        schema = _load_schema()
        features = _full_feature_block()
        features["footprint_ratio"] = -0.5
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bundle, schema=schema)

    def test_footprint_ratio_above_one_hundred_fails(self) -> None:
        schema = _load_schema()
        features = _full_feature_block()
        features["footprint_ratio"] = 100.5
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bundle, schema=schema)

    def test_intra_lane_reuse_above_one_fails(self) -> None:
        schema = _load_schema()
        features = _full_feature_block()
        features["intra_lane_reuse"] = 1.01
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bundle, schema=schema)

    def test_gridDim_density_negative_fails(self) -> None:
        schema = _load_schema()
        features = _full_feature_block()
        features["gridDim_density"] = -1.0
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bundle, schema=schema)

    def test_gridDim_density_can_be_large(self) -> None:
        schema = _load_schema()
        features = _full_feature_block()
        features["gridDim_density"] = 1000.0
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = features
        # No upper bound on gridDim_density; should admit.
        jsonschema.validate(instance=bundle, schema=schema)


class MonotonicConstraintSignsTest(unittest.TestCase):
    """the monotonic-constraint-sign table sign table is the binding contract for the v5 head."""

    def test_canonical_feature_count_is_six(self) -> None:
        self.assertEqual(len(cf.CANONICAL_FEATURE_NAMES), 6)

    def test_constraint_signs_match_ac21(self) -> None:
        self.assertEqual(
            cf.MONOTONIC_CONSTRAINT_SIGNS,
            {
                "raw_prediction": +1,
                "footprint_ratio": -1,
                "long_rd_tail_mass": -1,
                "intra_lane_reuse": +1,
                "global_share": 0,
                "gridDim_density": -1,
            },
        )

    def test_canonical_feature_names_match_constraint_keys(self) -> None:
        self.assertEqual(
            cf.CANONICAL_FEATURE_NAMES,
            tuple(cf.MONOTONIC_CONSTRAINT_SIGNS.keys()),
        )


class ComputeFootprintRatioTest(unittest.TestCase):
    def test_basic_ratio(self) -> None:
        # 10000 sectors * 32 bytes / 320000 bytes = 1.0
        self.assertEqual(
            cf.compute_footprint_ratio(
                unique_sector_count=10000,
                cache_model_capacity_bytes=320000,
            ),
            1.0,
        )

    def test_zero_unique_sectors(self) -> None:
        self.assertEqual(
            cf.compute_footprint_ratio(
                unique_sector_count=0,
                cache_model_capacity_bytes=1024,
            ),
            0.0,
        )

    def test_negative_unique_sectors_raises(self) -> None:
        with self.assertRaises(ValueError):
            cf.compute_footprint_ratio(
                unique_sector_count=-1,
                cache_model_capacity_bytes=1024,
            )

    def test_zero_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            cf.compute_footprint_ratio(
                unique_sector_count=100,
                cache_model_capacity_bytes=0,
            )


class ComputeLongRdTailMassTest(unittest.TestCase):
    def test_empty_bins_returns_zero(self) -> None:
        self.assertEqual(
            cf.compute_long_rd_tail_mass(hrd_profile={"bins": []}),
            0.0,
        )

    def test_no_sector_layer_returns_zero(self) -> None:
        hrd = {
            "bins": [
                {
                    "hierarchy_layer": "allocation",
                    "rd_sector_units_min": 5000,
                    "observation_count": 1000,
                },
            ],
        }
        self.assertEqual(cf.compute_long_rd_tail_mass(hrd_profile=hrd), 0.0)

    def test_long_rd_threshold_inclusive(self) -> None:
        # Two sector bins: one BELOW threshold (3000), one AT threshold (4096).
        # Long-RD tail mass = 200 / (800 + 200) = 0.2.
        hrd = {
            "bins": [
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 3000,
                    "observation_count": 800,
                },
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 4096,
                    "observation_count": 200,
                },
            ],
        }
        self.assertAlmostEqual(
            cf.compute_long_rd_tail_mass(hrd_profile=hrd),
            0.2,
        )

    def test_custom_threshold(self) -> None:
        hrd = {
            "bins": [
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 100,
                    "observation_count": 500,
                },
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 200,
                    "observation_count": 500,
                },
            ],
        }
        self.assertEqual(
            cf.compute_long_rd_tail_mass(
                hrd_profile=hrd, threshold_sectors=150,
            ),
            0.5,
        )


class ComputeIntraLaneReuseTest(unittest.TestCase):
    def test_intra_lane_present(self) -> None:
        sl = {
            "grounded": [
                {"class_name": "intra_lane", "fraction": 0.42},
                {"class_name": "intra_warp", "fraction": 0.0},
            ],
        }
        self.assertEqual(cf.compute_intra_lane_reuse(sharing_locality=sl), 0.42)

    def test_intra_lane_absent_returns_zero(self) -> None:
        sl = {
            "grounded": [
                {"class_name": "intra_warp", "fraction": 0.0},
            ],
        }
        self.assertEqual(cf.compute_intra_lane_reuse(sharing_locality=sl), 0.0)

    def test_no_grounded_returns_zero(self) -> None:
        self.assertEqual(cf.compute_intra_lane_reuse(sharing_locality={}), 0.0)


class ComputeGlobalShareTest(unittest.TestCase):
    def test_precomputed_pair(self) -> None:
        self.assertEqual(
            cf.compute_global_share(global_count=3, total_count=10),
            0.3,
        )

    def test_precomputed_zero_total(self) -> None:
        self.assertEqual(
            cf.compute_global_share(global_count=0, total_count=0),
            0.0,
        )

    def test_iterable_scan(self) -> None:
        records = [
            {"memory_op_class": "GLOBAL"},
            {"memory_op_class": "SHARED"},
            {"memory_op_class": "GLOBAL"},
            {"memory_op_class": "LOCAL"},
        ]
        self.assertEqual(
            cf.compute_global_share(sector_stream_records=records),
            0.5,
        )

    def test_global_count_exceeds_total_raises(self) -> None:
        with self.assertRaises(ValueError):
            cf.compute_global_share(global_count=5, total_count=3)

    def test_negative_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            cf.compute_global_share(global_count=-1, total_count=10)

    def test_neither_form_raises(self) -> None:
        with self.assertRaises(ValueError):
            cf.compute_global_share()


class ComputeGridDimDensityTest(unittest.TestCase):
    def test_none_grid_volume_returns_zero(self) -> None:
        self.assertEqual(
            cf.compute_gridDim_density(grid_volume=None, device=None),
            0.0,
        )

    def test_sm120_device_uses_sm_120_count(self) -> None:
        # 2600 / 26 = 100.0 (sm_count 26 = MEASURED on the live 5060; the
        # guessed 50 was corrected in the 2026-07-06 pivot baseline)
        self.assertEqual(
            cf.compute_gridDim_density(
                grid_volume=2600,
                device="NVIDIA GeForce RTX 5060 Laptop GPU",
            ),
            100.0,
        )

    def test_sm86_device_uses_sm_86_count(self) -> None:
        # 8400 / 84 = 100.0
        self.assertEqual(
            cf.compute_gridDim_density(
                grid_volume=8400,
                device="NVIDIA GeForce RTX 3090 Ti",
            ),
            100.0,
        )

    def test_negative_grid_raises(self) -> None:
        with self.assertRaises(ValueError):
            cf.compute_gridDim_density(grid_volume=-1, device=None)


class AssembleCanonicalFeaturesTest(unittest.TestCase):
    """End-to-end on synthetic + real sidecars."""

    def _synthetic_inputs(self) -> dict[str, Any]:
        bundle = _minimal_scalar_bundle(
            cache_model_capacity_bytes=320_000,
            cache_model_grid_volume=5000,
        )
        bundle["evidence"]["predicted_hit_rate_uncorrected"] = 0.4
        sharing_locality = {
            "grounded": [
                {"class_name": "intra_lane", "fraction": 0.6},
            ],
        }
        hrd_profile = {
            "bins": [
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 4096,
                    "observation_count": 300,
                },
                {
                    "hierarchy_layer": "sector",
                    "rd_sector_units_min": 100,
                    "observation_count": 700,
                },
            ],
        }
        return {
            "predictor_bundle": bundle,
            "sharing_locality": sharing_locality,
            "hrd_profile": hrd_profile,
            "unique_sector_count": 10000,
            "global_share_value": 0.95,
        }

    def test_assemble_produces_all_six_features(self) -> None:
        result = cf.assemble_canonical_features(**self._synthetic_inputs())
        self.assertEqual(
            set(result.keys()), set(cf.CANONICAL_FEATURE_NAMES),
        )
        self.assertAlmostEqual(result["raw_prediction"], 0.4)
        self.assertAlmostEqual(result["footprint_ratio"], 1.0)
        self.assertAlmostEqual(result["long_rd_tail_mass"], 0.3)
        self.assertAlmostEqual(result["intra_lane_reuse"], 0.6)
        self.assertAlmostEqual(result["global_share"], 0.95)
        # 5000 blocks / 26 SMs (the 5060's measured sm_count)
        self.assertAlmostEqual(result["gridDim_density"], 5000 / 26)

    def test_assemble_requires_predicted_hit_rate_uncorrected(self) -> None:
        inputs = self._synthetic_inputs()
        del inputs["predictor_bundle"]["evidence"][
            "predicted_hit_rate_uncorrected"
        ]
        with self.assertRaises(ValueError) as cm:
            cf.assemble_canonical_features(**inputs)
        self.assertIn("predicted_hit_rate_uncorrected", str(cm.exception))

    def test_assemble_scans_sector_stream_records_when_no_value(self) -> None:
        inputs = self._synthetic_inputs()
        del inputs["global_share_value"]
        inputs["sector_stream_records"] = [
            {"memory_op_class": "GLOBAL"},
            {"memory_op_class": "GLOBAL"},
            {"memory_op_class": "SHARED"},
        ]
        result = cf.assemble_canonical_features(**inputs)
        self.assertAlmostEqual(result["global_share"], 2.0 / 3.0)

    def test_assemble_validates_each_feature_range(self) -> None:
        # Build inputs that produce footprint_ratio above 100 to trigger range fail.
        inputs = self._synthetic_inputs()
        inputs["unique_sector_count"] = 10_000_000  # huge
        # footprint_ratio = 10_000_000 * 32 / 320_000 = 1000.0 -> out of range.
        with self.assertRaises(ValueError) as cm:
            cf.assemble_canonical_features(**inputs)
        self.assertIn("footprint_ratio", str(cm.exception))

    def test_assemble_real_sm120_cell_if_present(self) -> None:
        """Real-artifact regression: assemble features for a real scalar
        sm_120 cell whose sharing_locality + hrd_profile sidecars are
        committed. Skips when the artifacts are absent on this host."""
        # carveout_sensitivity_probe is a scalar bundle (predicted_hit_rate_uncorrected
        # is present in evidence) with committed sidecars for sharing_locality
        # and hrd_profile under workloads/simbench_bw/results/.
        slug = "carveout_sensitivity_probe"
        bundle_path = (
            _SM120_BUNDLES_DIR
            / f"sm_120__{slug}__trace_l2_accuracy_xl.json"
        )
        if not bundle_path.is_file():
            self.skipTest(f"sm_120 {slug} bundle absent")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("result_kind") != "scalar":
            self.skipTest(
                f"bundle result_kind={bundle.get('result_kind')!r}; "
                "real-artifact assemble test requires a scalar bundle"
            )
        results_root = (
            _REPO_ROOT / "workloads/simbench_bw/results"
            / slug / "rtx_5060_laptop" / "sm_120"
            / "trace_trace_l2_accuracy_xl"
        )
        sl_path = (
            results_root / "sharing_locality" / "sharing_locality.json"
        )
        hrd_path = results_root / "hrd_profile" / "hrd_profile.json"
        if not (sl_path.is_file() and hrd_path.is_file()):
            self.skipTest("sharing_locality or hrd_profile sidecar absent")
        sharing_locality = json.loads(sl_path.read_text(encoding="utf-8"))
        hrd_profile = json.loads(hrd_path.read_text(encoding="utf-8"))
        result = cf.assemble_canonical_features(
            predictor_bundle=bundle,
            sharing_locality=sharing_locality,
            hrd_profile=hrd_profile,
            unique_sector_count=10_000,  # placeholder; range-only assertion
            global_share_value=1.0,
        )
        self.assertEqual(set(result.keys()), set(cf.CANONICAL_FEATURE_NAMES))
        # Each value must be in its declared range.
        self.assertGreaterEqual(result["raw_prediction"], 0.0)
        self.assertLessEqual(result["raw_prediction"], 1.0)
        self.assertGreaterEqual(result["footprint_ratio"], 0.0)
        self.assertLessEqual(result["footprint_ratio"], 100.0)
        for n in (
            "long_rd_tail_mass",
            "intra_lane_reuse",
            "global_share",
        ):
            self.assertGreaterEqual(result[n], 0.0)
            self.assertLessEqual(result[n], 1.0)
        self.assertGreaterEqual(result["gridDim_density"], 0.0)


class ActivePathRejectsLegacyFactorKwargsTest(unittest.TestCase):
    """The active path's contract: per-slug factor kwargs raise
    ActivePathDomainMismatchError because the 3-stage multiplicative
    correction chain is retired."""

    def _bundle_with_features(self) -> dict:
        bundle = _minimal_scalar_bundle()
        bundle["canonical_pre_calibration_features"] = _full_feature_block()
        return bundle

    def _identity_head(self) -> Any:
        return lambda features: features["raw_prediction"]

    def test_admits_clean_call(self) -> None:
        result = cf.predict_active_path(
            bundle_with_features=self._bundle_with_features(),
            head_predict_fn=self._identity_head(),
        )
        self.assertAlmostEqual(result["calibrated_hit_rate"], 0.5)
        self.assertEqual(
            set(result["features_consumed"].keys()),
            set(cf.CANONICAL_FEATURE_NAMES),
        )

    def test_rejects_factor_kwarg(self) -> None:
        with self.assertRaises(cf.ActivePathDomainMismatchError) as cm:
            cf.predict_active_path(
                bundle_with_features=self._bundle_with_features(),
                head_predict_fn=self._identity_head(),
                factor=0.6,
            )
        self.assertEqual(cm.exception.offending_kwarg, "factor")
        self.assertEqual(cm.exception.value, 0.6)
        diag = cm.exception.to_json()
        self.assertEqual(diag["exception"], "ActivePathDomainMismatchError")
        self.assertEqual(diag["offending_kwarg"], "factor")

    def test_rejects_factor_override_kwarg(self) -> None:
        with self.assertRaises(cf.ActivePathDomainMismatchError):
            cf.predict_active_path(
                bundle_with_features=self._bundle_with_features(),
                head_predict_fn=self._identity_head(),
                factor_override=0.22,
            )

    def test_rejects_per_slug_factor_kwarg(self) -> None:
        with self.assertRaises(cf.ActivePathDomainMismatchError):
            cf.predict_active_path(
                bundle_with_features=self._bundle_with_features(),
                head_predict_fn=self._identity_head(),
                per_slug_factor=0.54,
            )

    def test_rejects_l2_reduction_factor_kwarg(self) -> None:
        with self.assertRaises(cf.ActivePathDomainMismatchError):
            cf.predict_active_path(
                bundle_with_features=self._bundle_with_features(),
                head_predict_fn=self._identity_head(),
                l2_reduction_factor=0.88,
            )

    def test_rejects_l2_high_footprint_factor_kwarg(self) -> None:
        with self.assertRaises(cf.ActivePathDomainMismatchError):
            cf.predict_active_path(
                bundle_with_features=self._bundle_with_features(),
                head_predict_fn=self._identity_head(),
                l2_high_footprint_factor=0.26,
            )

    def test_rejects_gridDim_penalty_alpha_kwarg(self) -> None:
        with self.assertRaises(cf.ActivePathDomainMismatchError):
            cf.predict_active_path(
                bundle_with_features=self._bundle_with_features(),
                head_predict_fn=self._identity_head(),
                gridDim_penalty_alpha=0.110,
            )


class ActivePathRejectsMissingBlockTest(unittest.TestCase):
    """The active path fails closed when the canonical block is absent or
    incomplete."""

    def _const_head(self) -> Any:
        # Lambda body intentionally ignores the features dict; the
        # const_head fixture only exercises the wrapping path, not the
        # head's calibration logic.
        return lambda _: 0.5

    def test_missing_block_raises_value_error(self) -> None:
        bundle = _minimal_scalar_bundle()
        with self.assertRaises(ValueError) as cm:
            cf.predict_active_path(
                bundle_with_features=bundle,
                head_predict_fn=self._const_head(),
            )
        self.assertIn("canonical_pre_calibration_features", str(cm.exception))

    def test_block_missing_one_feature_raises(self) -> None:
        bundle = _minimal_scalar_bundle()
        features = _full_feature_block()
        del features["intra_lane_reuse"]
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(ValueError) as cm:
            cf.predict_active_path(
                bundle_with_features=bundle,
                head_predict_fn=self._const_head(),
            )
        self.assertIn("intra_lane_reuse", str(cm.exception))

    def test_block_unexpected_feature_raises(self) -> None:
        bundle = _minimal_scalar_bundle()
        features = _full_feature_block()
        features["sneaky_extra"] = 0.1
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(ValueError) as cm:
            cf.predict_active_path(
                bundle_with_features=bundle,
                head_predict_fn=self._const_head(),
            )
        self.assertIn("sneaky_extra", str(cm.exception))

    def test_block_out_of_range_feature_raises(self) -> None:
        bundle = _minimal_scalar_bundle()
        features = _full_feature_block()
        features["footprint_ratio"] = -1.0
        bundle["canonical_pre_calibration_features"] = features
        with self.assertRaises(ValueError) as cm:
            cf.predict_active_path(
                bundle_with_features=bundle,
                head_predict_fn=self._const_head(),
            )
        self.assertIn("footprint_ratio", str(cm.exception))


class ActivePathSourceDecouplingTest(unittest.TestCase):
    """The active path module MUST NOT import from cache_predictor.py because
    that module owns the retired 3-stage multiplicative correction chain
    (gridDim penalty + per-slug L1/L2 factors). Per the canonical-features contract's positive test:
    'no per-slug factor table import survives in the active path'.

    Source-text scan rather than runtime introspection because the import-time
    decoupling is the design contract; a runtime probe could miss a lazy
    `import workloads.simbench_bw.cache_predictor` deep in a code path.
    """

    def test_canonical_features_does_not_import_cache_predictor(self) -> None:
        path = (
            _REPO_ROOT / "workloads/simbench_bw/canonical_features.py"
        )
        src = path.read_text(encoding="utf-8")
        # Strip docstrings + comments would be ideal, but a simple
        # substring scan is sufficient: the legacy module name is unique
        # and any import would carry it verbatim. Doc-mentions live as
        # plain references (e.g., "cache_predictor.py") not as imports.
        # We forbid only the import forms.
        forbidden_import_forms = (
            "from workloads.simbench_bw.cache_predictor",
            "from workloads.simbench_bw import cache_predictor",
            "import workloads.simbench_bw.cache_predictor",
            "from cache_predictor ",
            "from .cache_predictor",
            "from . import cache_predictor",
        )
        for form in forbidden_import_forms:
            with self.subTest(form=form):
                self.assertNotIn(
                    form, src,
                    f"canonical_features.py imports cache_predictor via "
                    f"forbidden form {form!r}; the active path is "
                    f"contract-decoupled from the retired 3-stage chain",
                )

    def test_canonical_features_does_not_reference_legacy_factor_constants(self) -> None:
        """Source-level scan: the active path MUST NOT reference the
        retired legacy correction constants by name. A bare-name reference
        (even in a comment) is admitted ONLY if it is inside the
        ``_FORBIDDEN_FACTOR_KWARGS`` set definition or its associated
        documentation, because that set IS the rejection contract."""
        path = (
            _REPO_ROOT / "workloads/simbench_bw/canonical_features.py"
        )
        src = path.read_text(encoding="utf-8")
        legacy_constants = (
            "_L2_REDUCTION_FACTORS",
            "_L2_HIGH_FOOTPRINT_LOW_REUSE_CORRECTION_FACTORS",
            "_GRIDDIM_PENALTY_ALPHA",
        )
        for name in legacy_constants:
            with self.subTest(name=name):
                self.assertNotIn(
                    name, src,
                    f"canonical_features.py references legacy constant "
                    f"{name!r}; the active path must not couple to the "
                    f"retired correction stack",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
