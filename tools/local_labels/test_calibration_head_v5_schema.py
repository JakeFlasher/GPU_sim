"""Schema validator tests for the v5 monotonic calibration head artifact.

Covers the goal tracker's v5 head provenance contract row + the missing
v5 schema/negative-test surface:

  - Positive: the committed data/local_labels/calibration_head_v5.json
    validates cleanly against the new schema.
  - Negative: synthetic fixtures that drop composite_score_formula,
    that flip a monotonic constraint sign, OR that set kind=per_bucket
    with n_train<8 all fail validation.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import jsonschema


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads/simbench_bw/schemas/calibration_head_v5.schema.json"
)
_ARTIFACT_PATH = _REPO_ROOT / "data/local_labels/calibration_head_v5.json"


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _load_artifact() -> dict:
    return json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))


class V5HeadSchemaPositiveTest(unittest.TestCase):
    def test_committed_v5_head_validates(self) -> None:
        if not _ARTIFACT_PATH.is_file():
            self.skipTest("calibration_head_v5.json absent on this host")
        artifact = _load_artifact()
        schema = _load_schema()
        jsonschema.validate(instance=artifact, schema=schema)

    def test_schema_self_references_drop_zero(self) -> None:
        """Schema metadata fields exist + are non-empty."""
        schema = _load_schema()
        self.assertEqual(schema["schema_version"], 1)
        self.assertIn("required", schema)


class V5HeadSchemaNegativeTest(unittest.TestCase):
    def _base_artifact(self) -> dict:
        """Build a synthetic, schema-valid artifact for mutation. Mirrors the
        real artifact's shape per BL-20260527-test-fixture-shape-must-mirror-real-shape."""
        return {
            "schema_version": 1,
            "calibration_head_version": 5,
            "source_commit": "0" * 40,
            "dependency_availability": {
                "isotonic_regression_on_frozen_projection": {
                    "available": True, "version": "1.8.0",
                    "import_path": "sklearn.isotonic", "package": "sklearn",
                },
            },
            "dependency_versions": {"sklearn": "1.8.0"},
            "composite_score_formula": {
                "primary": "mean_cv_geomean_relative_error",
                "secondary_tiebreaker": "median_absolute_hit_rate_error",
                "tertiary": "log_loss_eps_1e_minus_6_fractional_bernoulli",
                "simpler_model_tiebreaker_scope": "primary_metric_only_within_2pct_diff",
                "simpler_model_tiebreaker_threshold_fraction": "0.02",
            },
            "feature_schema": {
                "canonical_feature_names": [
                    "raw_prediction", "footprint_ratio", "long_rd_tail_mass",
                    "intra_lane_reuse", "global_share", "gridDim_density",
                ],
                "monotonic_constraints_by_feature": {
                    "raw_prediction": 1,
                    "footprint_ratio": -1,
                    "long_rd_tail_mass": -1,
                    "intra_lane_reuse": 1,
                    "global_share": 0,
                    "gridDim_density": -1,
                },
            },
            "fit_status": "ok",
            "cohort_scope": {
                "admitted_count": 1,
                "skipped_count": 0,
                "admitted": [
                    {
                        "arch": "sm_120",
                        "slug": "carveout_sensitivity_probe",
                        "shape_point": "trace_l2_accuracy_xl",
                        "family_bucket": "memory_stress",
                        "label": 0.97,
                    },
                ],
                "skipped": [],
            },
            "model_class_per_bucket": {
                "sm_120/memory_stress": "isotonic_regression_on_frozen_projection__pooled_fallback_with_one_hot",
            },
            "monotonic_constraints_by_feature": {
                "sm_120/memory_stress": {
                    "raw_prediction": 1, "footprint_ratio": -1,
                    "long_rd_tail_mass": -1, "family_bucket_one_hot": 0,
                },
            },
            "fold_assignments": {"['sm_120', 'carveout_sensitivity_probe', 'trace_l2_accuracy_xl']": 0},
            "selected_hyperparameters": {
                "isotonic_regression_on_frozen_projection": {
                    "increasing": True, "out_of_bounds": "clip",
                },
            },
            "training_cell_ids": {
                "sm_120/memory_stress": [["sm_120", "carveout_sensitivity_probe", "trace_l2_accuracy_xl"]],
            },
            "per_bucket_provenance": {
                "by_bucket": {},
                "pooled_fallback_present": True,
                "pooled_used_features": [
                    "raw_prediction", "footprint_ratio", "long_rd_tail_mass",
                    "family_bucket_one_hot",
                ],
            },
        }

    def test_base_artifact_is_valid(self) -> None:
        jsonschema.validate(
            instance=self._base_artifact(), schema=_load_schema(),
        )

    def test_missing_composite_score_formula_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["composite_score_formula"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("composite_score_formula", str(cm.exception))

    def test_missing_dependency_versions_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["dependency_versions"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("dependency_versions", str(cm.exception))

    def test_monotonic_constraint_sign_drift_fails(self) -> None:
        artifact = self._base_artifact()
        # Flip footprint_ratio's sign to +1 (canonical is -1).
        artifact["feature_schema"]["monotonic_constraints_by_feature"][
            "footprint_ratio"
        ] = 1
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=artifact, schema=_load_schema())

    def test_per_bucket_with_n_train_below_eight_fails(self) -> None:
        artifact = self._base_artifact()
        artifact["per_bucket_provenance"] = {
            "by_bucket": {
                "sm_120/memory_stress": {
                    "kind": "per_bucket",
                    "n_train": 5,
                    "training_cell_ids": [],
                },
            },
            "pooled_fallback_present": False,
            "pooled_used_features": [],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=artifact, schema=_load_schema())

    def test_missing_model_class_per_bucket_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["model_class_per_bucket"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("model_class_per_bucket", str(cm.exception))

    def test_missing_fold_assignments_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["fold_assignments"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("fold_assignments", str(cm.exception))

    def test_missing_selected_hyperparameters_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["selected_hyperparameters"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("selected_hyperparameters", str(cm.exception))

    def test_missing_training_cell_ids_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["training_cell_ids"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("training_cell_ids", str(cm.exception))

    def test_missing_per_bucket_provenance_fails(self) -> None:
        artifact = self._base_artifact()
        del artifact["per_bucket_provenance"]
        with self.assertRaises(jsonschema.ValidationError) as cm:
            jsonschema.validate(instance=artifact, schema=_load_schema())
        self.assertIn("per_bucket_provenance", str(cm.exception))

    def test_indeterminate_dependencies_absent_fit_status_admits(self) -> None:
        """The dependency-absent arm emits a schema-valid artifact so review
        environments without sklearn / pygam / lightgbm / xgboost can re-run
        the fitter without crashing."""
        artifact = self._base_artifact()
        artifact["fit_status"] = "indeterminate_dependencies_absent"
        artifact["dependency_absence_reason"] = (
            "No model class from the registered hierarchy is importable on this host."
        )
        # Should validate cleanly.
        jsonschema.validate(instance=artifact, schema=_load_schema())

    def test_pooled_fallback_does_not_require_n_train_floor(self) -> None:
        """The per-bucket minimum-N rule is ONLY enforced when kind=per_bucket.
        A pooled_fallback bucket with n_train<8 is the explicit expected case
        when the bucket has too few cells for a per-bucket fit."""
        artifact = self._base_artifact()
        artifact["per_bucket_provenance"] = {
            "by_bucket": {
                "sm_120/memory_stress": {
                    "kind": "pooled_fallback",
                    "n_train": 2,
                    "training_cell_ids": [],
                    "pooled_bias": 0.0,
                },
            },
            "pooled_fallback_present": True,
            "pooled_used_features": [
                "raw_prediction",
                "footprint_ratio",
                "long_rd_tail_mass",
                "family_bucket_one_hot",
            ],
        }
        # No exception expected.
        jsonschema.validate(instance=artifact, schema=_load_schema())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
