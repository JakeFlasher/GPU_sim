"""Tests for the L1 sector-hit-ratio diagnostic surface.

Covers (a) the emitted diagnostic artifact is schema-valid (draft-2020-12) and
honestly marked ``load_bearing == false``; (b) a NEGATIVE guard test that a
synthetic payload claiming ``load_bearing == true`` WITHOUT meeting the
median-absolute-error bar plus the bypass-provenance requirement is REJECTED by
``validate_l1_diagnostic``; and (c) a POSITIVE guard test that a synthetic
payload that meets the bar AND carries bypass provenance with
``load_bearing == true`` is ACCEPTED.

The diagnostic is expected to be NOT load-bearing: a faithful L1 model would
need per-SM / per-CTA scheduling plus .ca/.cg cache-op eligibility and explicit
bypass semantics, none of which are modeled in this coarse single-L1 surface.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import l1_diagnostic as l1d  # noqa: E402

_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_l1_diagnostic.schema.json"


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text())
    # The schema itself must be a valid draft-2020-12 schema.
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _accepting_payload() -> dict:
    """A minimal synthetic payload that meets the promotion bar AND carries
    bypass provenance with load_bearing=true (the POSITIVE-guard fixture)."""
    return {
        "schema_version": 1,
        "algorithm_version": "cache_replay_l1_diagnostic_v1",
        "source_commit": "deadbeef",
        "arch": "sm_120",
        "sku": "rtx_5060_laptop",
        "cache_layer": "L1TEX",
        "load_bearing": True,
        "diagnostic_median_absolute_error": 0.05,
        "load_bearing_provenance": {
            "load_bearing_medabs_bar": l1d.L1_LOAD_BEARING_MEDABS_BAR,
            "meets_accuracy_bar": True,
            "bypass_provenance_present": True,
            "l1_capacity_bytes": l1d.L1_DIAGNOSTIC_CAPACITY_BYTES,
            "associativity": l1d.L1_DIAGNOSTIC_ASSOCIATIVITY,
            "replacement": l1d.L1_DIAGNOSTIC_REPLACEMENT,
            "num_slices": l1d.L1_DIAGNOSTIC_NUM_SLICES,
            "assumption_note": "synthetic positive fixture with bypass modeled",
        },
        "cache_parameters": l1d._l1_diagnostic_config().provenance(),
        "n_cells": 0,
        "rows": [],
    }


class EmittedDiagnosticArtifactTest(unittest.TestCase):
    """The diagnostic surface, built over the real cohort, must be schema-valid
    and honestly marked not-load-bearing."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.payload = l1d.build_l1_diagnostic(use_cache=True)
        # Emit atomically to a TMP path (NOT the tracked artifact) so running the
        # test never re-stamps the committed artifact's source_commit.
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._on_disk_path = Path(cls._tmpdir.name) / "cache_replay_l1_diagnostic.json"
        l1d.artifact_io.atomic_write_json(cls._on_disk_path, cls.payload)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_emitted_payload_is_schema_valid(self):
        _validator().validate(self.payload)

    def test_persisted_artifact_round_trips_schema(self):
        on_disk = json.loads(self._on_disk_path.read_text())
        _validator().validate(on_disk)

    def test_surface_is_marked_not_load_bearing(self):
        # The honest, expected outcome: the coarse single-L1 diagnostic does not
        # meet the promotion bar, so it must not be load-bearing.
        self.assertFalse(self.payload["load_bearing"])
        prov = self.payload["load_bearing_provenance"]
        self.assertFalse(prov["bypass_provenance_present"])

    def test_provenance_records_documented_assumptions(self):
        prov = self.payload["load_bearing_provenance"]
        self.assertEqual(prov["l1_capacity_bytes"], 128 * 1024)
        self.assertEqual(prov["associativity"], 4)
        self.assertEqual(prov["replacement"], "LRU")
        self.assertEqual(prov["num_slices"], 1)
        self.assertEqual(prov["load_bearing_medabs_bar"], 0.10)
        self.assertIsInstance(prov["assumption_note"], str)
        self.assertTrue(prov["assumption_note"])

    def test_stamped_with_commit_and_schema_version(self):
        self.assertEqual(self.payload["schema_version"], 1)
        self.assertIn("source_commit", self.payload)
        self.assertEqual(self.payload["arch"], "sm_120")

    def test_every_row_has_an_l1_label_and_is_l1tex_layer(self):
        # Only cells that HAVE an l1_label contribute to the diagnostic.
        for row in self.payload["rows"]:
            self.assertEqual(row["join_key"]["cache_layer"], "L1TEX")
            self.assertEqual(row["join_key"]["arch"], "sm_120")
            self.assertGreaterEqual(row["l1_ground_truth_hit_ratio"], 0.0)
            self.assertLessEqual(row["l1_ground_truth_hit_ratio"], 1.0)

    def test_emitted_payload_passes_load_bearing_scope_guard(self):
        # A not-load-bearing surface is always admitted by the guard.
        l1d.validate_l1_diagnostic(self.payload)  # must not raise


class LoadBearingScopeGuardTest(unittest.TestCase):
    """The load-bearing-scope guard: load_bearing=true is admitted only with the
    accuracy bar met AND bypass provenance present."""

    def test_negative_load_bearing_without_bar_is_rejected(self):
        # Claims load_bearing=true but misses the <=0.10 bar AND has no bypass
        # provenance -- exactly the dishonest-promotion case the guard refuses.
        payload = _accepting_payload()
        payload["load_bearing"] = True
        payload["diagnostic_median_absolute_error"] = 0.42
        payload["load_bearing_provenance"]["meets_accuracy_bar"] = False
        payload["load_bearing_provenance"]["bypass_provenance_present"] = False
        with self.assertRaises(ValueError):
            l1d.validate_l1_diagnostic(payload)

    def test_negative_bar_met_but_no_bypass_provenance_is_rejected(self):
        # Meets the accuracy bar but lacks bypass provenance -> still rejected.
        payload = _accepting_payload()
        payload["load_bearing"] = True
        payload["diagnostic_median_absolute_error"] = 0.05
        payload["load_bearing_provenance"]["bypass_provenance_present"] = False
        with self.assertRaises(ValueError):
            l1d.validate_l1_diagnostic(payload)

    def test_negative_bypass_present_but_bar_missed_is_rejected(self):
        # Carries bypass provenance but the median absolute error exceeds the
        # bar -> still rejected.
        payload = _accepting_payload()
        payload["load_bearing"] = True
        payload["diagnostic_median_absolute_error"] = 0.25
        payload["load_bearing_provenance"]["bypass_provenance_present"] = True
        with self.assertRaises(ValueError):
            l1d.validate_l1_diagnostic(payload)

    def test_positive_bar_met_and_bypass_present_is_accepted(self):
        # Meets the <=0.10 bar AND has bypass provenance present with
        # load_bearing=true -> accepted.
        payload = _accepting_payload()
        l1d.validate_l1_diagnostic(payload)  # must not raise

    def test_positive_not_load_bearing_is_always_accepted(self):
        payload = _accepting_payload()
        payload["load_bearing"] = False
        # Even with a bad medabs and no bypass provenance, not-load-bearing is
        # always admitted.
        payload["diagnostic_median_absolute_error"] = 0.9
        payload["load_bearing_provenance"]["bypass_provenance_present"] = False
        payload["load_bearing_provenance"]["meets_accuracy_bar"] = False
        l1d.validate_l1_diagnostic(payload)  # must not raise

    def test_accepting_fixture_is_schema_valid(self):
        # The positive fixture is also a structurally valid diagnostic payload.
        _validator().validate(_accepting_payload())


if __name__ == "__main__":
    unittest.main()
