"""Closed-schema + production-validator coverage for the sampled-fidelity artifact.

The positive case loads the committed artifact at
``data/local_labels/cache_replay_sampled_fidelity.json`` and asserts it validates
against the closed draft-2020-12 schema with zero errors and passes the
PRODUCTION validate-before-emit guard
(``sampled_replay.validate_sampled_fidelity_report``). The negative cases flip
individual preserved-verdict flags while leaving the fidelity gate marked passed;
the PRODUCTION validator (the same one the emit path calls) must reject them — a
verdict flip is not allowed to sail through the gate.
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

import sampled_replay as sr  # noqa: E402

_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_sampled_fidelity.schema.json"
_ARTIFACT_PATH = _REPO_ROOT / "data" / "local_labels" / "cache_replay_sampled_fidelity.json"


def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(_SCHEMA_PATH.read_text()))


def _load_artifact() -> dict:
    return json.loads(_ARTIFACT_PATH.read_text())


class SampledFidelitySchemaTest(unittest.TestCase):
    def test_committed_artifact_validates_and_passes_production_validator(self):
        payload = _load_artifact()
        errors = list(_validator().iter_errors(payload))
        self.assertEqual(errors, [], [e.message for e in errors])
        # The PRODUCTION validate-before-emit guard accepts the real artifact.
        sr.validate_sampled_fidelity_report(payload)

    def _flipped_passing_copy(self, criterion: str) -> dict:
        payload = copy.deepcopy(_load_artifact())
        payload["ac4_verdict_preserved"][criterion] = False
        payload["fidelity_gate"]["passed"] = True
        return payload

    def test_geomean_flip_rejected_by_production_validator(self):
        with self.assertRaises(ValueError):
            sr.validate_sampled_fidelity_report(self._flipped_passing_copy("geomean"))

    def test_holdout_flip_rejected_by_production_validator(self):
        with self.assertRaises(ValueError):
            sr.validate_sampled_fidelity_report(
                self._flipped_passing_copy("holdout_geomean"))

    def test_per_bucket_flip_rejected_by_production_validator(self):
        with self.assertRaises(ValueError):
            sr.validate_sampled_fidelity_report(
                self._flipped_passing_copy("per_bucket_p90"))

    def test_inconsistent_preservation_summary_rejected(self):
        # gate.ac4_verdict_preserved must equal the conjunction of the flags.
        payload = copy.deepcopy(_load_artifact())
        payload["ac4_verdict_preserved"]["median"] = False
        # leave fidelity_gate.ac4_verdict_preserved True (now inconsistent) and
        # passed False so the flip-rule alone would not catch it.
        payload["fidelity_gate"]["passed"] = False
        payload["fidelity_gate"]["ac4_verdict_preserved"] = True
        with self.assertRaises(ValueError):
            sr.validate_sampled_fidelity_report(payload)


if __name__ == "__main__":
    unittest.main()
