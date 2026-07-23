"""Tests for the advisory cache-replay validation artifact.

Covers:
  * building + emitting the advisory robustness artifact over the 30 sm_120
    cells, then validating the emitted artifact against the draft-2020-12 JSON
    schema (positive);
  * a NEGATIVE test that a payload missing the bootstrap-CI block FAILS the JSON
    schema validator;
  * a NEGATIVE test that an unflagged-unstable config payload is rejected by the
    exposed ``validate_validation_artifact`` guard, plus the absent-bootstrap-CI
    rejection by the same guard;
  * determinism of the stratified split and the bootstrap CI under the fixed
    seed;
  * the advisory framing (advisory=true, gates_closure=false) is preserved.

The build path replays the real decoded streams (cached on disk), so it is a
real exercise of the mechanistic predictor over the cohort, not a synthetic
stub.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import validation as validation_mod  # noqa: E402

_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_validation.schema.json"
_COMPARISON_PATH = _REPO_ROOT / "data/local_labels/cache_replay_comparison.json"


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _schema_validator() -> jsonschema.Draft202012Validator:
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


class SchemaIsWellFormedTest(unittest.TestCase):
    def test_schema_is_draft_2020_12_and_closed(self):
        schema = _load_schema()
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertFalse(schema["additionalProperties"])
        # the schema itself must be a valid draft-2020-12 schema
        jsonschema.Draft202012Validator.check_schema(schema)


class _BuiltArtifactMixin:
    """Builds the artifact once per test class (the cohort decode is cached on
    disk, but replaying still costs a few seconds; share it across tests)."""

    _artifact: dict | None = None

    @classmethod
    def _build_once(cls) -> dict:
        if cls._artifact is None:
            if not _COMPARISON_PATH.exists():
                raise unittest.SkipTest(
                    "comparison bundle not emitted yet; run calibrate first"
                )
            try:
                cls._artifact = validation_mod.build_validation_artifact(use_cache=True)
            except ValueError as exc:
                # Typed skip, not a silent pass: the live-artifact build
                # replays the 30-cell sm_120 cohort from decoded NVBit
                # traces. Hosts that carry the curated comparison bundle but
                # NOT the bulk trace corpus (it stayed in the
                # non_root_gpu_profiler_5060_branch legacy tree; see
                # data/annex/SALVAGE_MANIFEST.md) decode zero admitted cells
                # and sklearn raises the 0-sample ValueError. Anything else
                # re-raises.
                if "0 sample" in str(exc):
                    raise unittest.SkipTest(
                        "sm_120 cohort trace corpus not present on this host; "
                        "live-artifact validation tests need the Tree-B traces"
                    ) from exc
                raise
        return cls._artifact


class EmittedArtifactTest(_BuiltArtifactMixin, unittest.TestCase):
    def test_emitted_artifact_is_schema_valid(self):
        artifact = self._build_once()
        validator = _schema_validator()
        errors = sorted(validator.iter_errors(artifact), key=str)
        self.assertEqual(
            errors, [], msg="; ".join(e.message for e in errors),
        )

    def test_emitted_artifact_passes_exposed_validator(self):
        artifact = self._build_once()
        validation_mod.validate_validation_artifact(artifact)  # must not raise

    def test_artifact_is_advisory_and_does_not_gate_closure(self):
        artifact = self._build_once()
        self.assertTrue(artifact["advisory"])
        self.assertFalse(artifact["gates_closure"])

    def test_cohort_shape_is_thirty_cells(self):
        artifact = self._build_once()
        self.assertEqual(artifact["cohort_size"], 30)
        self.assertEqual(sum(artifact["family_bucket_counts"].values()), 30)

    def test_all_cells_flagged_stable_under_inert_grid(self):
        artifact = self._build_once()
        stab = artifact["per_cell_stability"]
        self.assertTrue(stab["grid_is_inert"])
        self.assertTrue(stab["all_cells_stable"])
        self.assertEqual(stab["n_unstable"], 0)
        self.assertEqual(stab["n_cells"], artifact["cohort_size"])
        self.assertTrue(all(stab["per_cell_stable"].values()))

    def test_bootstrap_ci_brackets_point_geomean(self):
        artifact = self._build_once()
        ci = artifact["bootstrap_ci"]
        self.assertLessEqual(ci["ci_low"], ci["geomean"] + 1e-9)
        self.assertGreaterEqual(ci["ci_high"], ci["geomean"] - 1e-9)
        self.assertEqual(ci["n_resamples"], validation_mod.N_BOOTSTRAP)
        self.assertEqual(ci["seed"], validation_mod.SEED)

    def test_advisory_thresholds_report_value_and_band(self):
        artifact = self._build_once()
        adv = artifact["advisory_thresholds"]
        for key in ("loocv_vs_holdout_gap", "per_slug_loso_headline_change"):
            block = adv[key]
            self.assertIn("value", block)
            self.assertIn("advisory_max", block)
            self.assertIsInstance(block["within_advisory_band"], bool)
            self.assertEqual(
                block["within_advisory_band"],
                block["value"] <= block["advisory_max"],
            )

    def test_emit_writes_schema_valid_file(self):
        # full emit path: write atomically, reload, schema-validate the file.
        # Write to a tmp path (NOT the tracked artifact) so running the test
        # never re-stamps the committed artifact's source_commit.
        import tempfile
        artifact = self._build_once()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cache_replay_validation.json"
            validation_mod.artifact_io.atomic_write_json(out, artifact)
            on_disk = json.loads(out.read_text())
        self.assertEqual(on_disk["source_commit"], artifact["source_commit"])
        validator = _schema_validator()
        errors = sorted(validator.iter_errors(on_disk), key=str)
        self.assertEqual(errors, [], msg="; ".join(e.message for e in errors))
        # provenance stamps required by the artifact contract
        self.assertEqual(on_disk["schema_version"], 1)
        self.assertTrue(on_disk["source_commit"])


class NegativeSchemaTest(_BuiltArtifactMixin, unittest.TestCase):
    def test_missing_bootstrap_ci_block_fails_schema(self):
        artifact = copy.deepcopy(self._build_once())
        del artifact["bootstrap_ci"]
        validator = _schema_validator()
        self.assertFalse(validator.is_valid(artifact))
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(artifact)

    def test_gates_closure_true_fails_schema(self):
        artifact = copy.deepcopy(self._build_once())
        artifact["gates_closure"] = True
        validator = _schema_validator()
        self.assertFalse(validator.is_valid(artifact))

    def test_extra_top_level_key_fails_closed_schema(self):
        artifact = copy.deepcopy(self._build_once())
        artifact["unexpected_field"] = 1
        validator = _schema_validator()
        self.assertFalse(validator.is_valid(artifact))


class NegativeExposedValidatorTest(_BuiltArtifactMixin, unittest.TestCase):
    def test_unflagged_unstable_config_is_rejected(self):
        artifact = copy.deepcopy(self._build_once())
        stab = artifact["per_cell_stability"]
        # flip one cell to unstable while leaving all_cells_stable=True ->
        # an unflagged-unstable config the exposed validator must reject.
        first_cell = next(iter(stab["per_cell_stable"]))
        stab["per_cell_stable"][first_cell] = False
        self.assertTrue(stab["all_cells_stable"])
        with self.assertRaises(ValueError):
            validation_mod.validate_validation_artifact(artifact)

    def test_unstable_count_mismatch_is_rejected(self):
        artifact = copy.deepcopy(self._build_once())
        stab = artifact["per_cell_stability"]
        first_cell = next(iter(stab["per_cell_stable"]))
        stab["per_cell_stable"][first_cell] = False
        # acknowledge instability but leave n_unstable inconsistent (0).
        stab["all_cells_stable"] = False
        stab["n_unstable"] = 0
        with self.assertRaises(ValueError):
            validation_mod.validate_validation_artifact(artifact)

    def test_absent_bootstrap_ci_block_rejected_by_validator(self):
        artifact = copy.deepcopy(self._build_once())
        del artifact["bootstrap_ci"]
        with self.assertRaises(ValueError):
            validation_mod.validate_validation_artifact(artifact)

    def test_gates_closure_true_rejected_by_validator(self):
        artifact = copy.deepcopy(self._build_once())
        artifact["gates_closure"] = True
        with self.assertRaises(ValueError):
            validation_mod.validate_validation_artifact(artifact)


class DeterminismTest(unittest.TestCase):
    def test_stratified_split_is_deterministic(self):
        if not _COMPARISON_PATH.exists():
            self.skipTest("comparison bundle not emitted yet")
        import cohort as cohort_mod  # noqa: WPS433

        cells = cohort_mod.load_cohort()
        if not cells:
            self.skipTest(
                "sm_120 cohort trace corpus not present on this host; "
                "see data/annex/SALVAGE_MANIFEST.md"
            )
        s1 = validation_mod.deterministic_stratified_split(cells)
        s2 = validation_mod.deterministic_stratified_split(cells)
        self.assertEqual(s1, s2)
        # every cell assigned to exactly one of the two splits
        self.assertEqual(set(s1.values()), {"calibration", "holdout"})

    def test_bootstrap_ci_is_seed_deterministic(self):
        preds = [0.6, 0.7, 0.55, 0.9, 0.4, 0.65]
        labels = [0.5, 0.8, 0.5, 0.85, 0.45, 0.6]
        a = validation_mod.bootstrap_geomean_ci(preds, labels)
        b = validation_mod.bootstrap_geomean_ci(preds, labels)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_low"], a["ci_high"])

    def test_reconstruct_selected_config_uses_only_constructor_kwargs(self):
        if not _COMPARISON_PATH.exists():
            self.skipTest("comparison bundle not emitted yet")
        bundle = json.loads(_COMPARISON_PATH.read_text())
        config = validation_mod.reconstruct_selected_config(bundle)
        sel = bundle["grid_provenance"]["selected_config"]
        self.assertEqual(config.capacity_bytes, sel["capacity_bytes"])
        self.assertEqual(config.associativity, sel["associativity"])
        self.assertEqual(config.replacement_policy, sel["replacement_policy"])
        self.assertEqual(config.num_slices, sel["num_slices"])
        self.assertEqual(config.slice_hash_id, sel["slice_hash_id"])
        self.assertEqual(config.capacity_scale, sel["capacity_scale"])


if __name__ == "__main__":
    unittest.main()
