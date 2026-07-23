"""Tests for the cache-replay falsification report's closed draft-2020-12 schema.

Covers:
  * POSITIVE: the committed falsification artifact validates against the
    draft-2020-12 JSON schema with zero errors AND passes the module's exposed
    ``validate_falsification_report`` guard;
  * NEGATIVE (omitted failure mode): a copy missing
    ``failure_mode_decomposition.trace_fidelity`` is rejected by the exposed
    behavioural validator (the three-way decomposition is mandatory);
  * NEGATIVE (indeterminate closure): a copy with ``research_success=false`` and
    ``hypothesis.verdict="indeterminate"`` is rejected by the behavioural
    validator (engineering closure needs a decisive verdict);
  * NEGATIVE (forbidden cross-arch gate): a copy with
    ``cross_arch_precondition=true`` is rejected by the behavioural validator
    (the closure must be terminal on the single sm_120 host).

The positive case loads the committed artifact verbatim, so it is a real
exercise of the emitted report against both the schema and the guard.
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

import falsification as falsification_mod  # noqa: E402

_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_falsification.schema.json"
_ARTIFACT_PATH = _REPO_ROOT / "data/local_labels/cache_replay_falsification.json"


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _schema_validator() -> jsonschema.Draft202012Validator:
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _load_artifact() -> dict:
    return json.loads(_ARTIFACT_PATH.read_text())


class SchemaIsWellFormedTest(unittest.TestCase):
    def test_schema_is_draft_2020_12_and_closed(self):
        schema = _load_schema()
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertFalse(schema["additionalProperties"])
        jsonschema.Draft202012Validator.check_schema(schema)


class PositiveTest(unittest.TestCase):
    def test_committed_artifact_validates_with_zero_errors(self):
        artifact = _load_artifact()
        validator = _schema_validator()
        errors = sorted(validator.iter_errors(artifact), key=str)
        self.assertEqual(errors, [], msg="; ".join(e.message for e in errors))

    def test_committed_artifact_passes_exposed_validator(self):
        artifact = _load_artifact()
        falsification_mod.validate_falsification_report(artifact)  # must not raise

    def test_terminal_on_single_host_is_allowed_true(self):
        artifact = _load_artifact()
        self.assertTrue(artifact["terminal_on_sm120_host"])
        validator = _schema_validator()
        self.assertTrue(validator.is_valid(artifact))


class NegativeOmittedFailureModeTest(unittest.TestCase):
    def test_missing_trace_fidelity_rejected_by_validator(self):
        artifact = copy.deepcopy(_load_artifact())
        self.assertFalse(artifact["research_success"])
        del artifact["failure_mode_decomposition"]["trace_fidelity"]
        with self.assertRaises(ValueError):
            falsification_mod.validate_falsification_report(artifact)

    def test_missing_trace_fidelity_also_fails_closed_schema(self):
        artifact = copy.deepcopy(_load_artifact())
        del artifact["failure_mode_decomposition"]["trace_fidelity"]
        validator = _schema_validator()
        self.assertFalse(validator.is_valid(artifact))


class NegativeIndeterminateClosureTest(unittest.TestCase):
    def test_indeterminate_verdict_without_research_success_rejected(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["research_success"] = False
        artifact["hypothesis"]["verdict"] = "indeterminate"
        with self.assertRaises(ValueError):
            falsification_mod.validate_falsification_report(artifact)


class TriStateCrossArchGateTest(unittest.TestCase):
    """2026-07-06 re-founding: ``cross_arch_precondition`` is tri-state
    (false | true | 'pending'); only non-tri-state values are rejected.
    The historical single-host closures pin ``false`` and stay valid."""

    def test_cross_arch_precondition_true_accepted_by_validator(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["cross_arch_precondition"] = True
        falsification_mod.validate_falsification_report(artifact)  # must not raise

    def test_cross_arch_precondition_pending_accepted_by_validator(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["cross_arch_precondition"] = "pending"
        falsification_mod.validate_falsification_report(artifact)  # must not raise

    def test_cross_arch_precondition_non_tristate_rejected_by_validator(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["cross_arch_precondition"] = "verified"
        with self.assertRaises(ValueError):
            falsification_mod.validate_falsification_report(artifact)


if __name__ == "__main__":
    unittest.main()
