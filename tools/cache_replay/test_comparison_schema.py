"""Tests for the per-cell baseline comparison artifact JSON schema.

The comparison artifact records, for every cohort cell, the calibrated and
uncalibrated cache-replay predictions alongside the full set of baseline
predictions (v5, Nugteren, uncalibrated walker, v4 Platt, cohort mean, and
always-hit). The plan requires that a missing baseline column for ANY cell
must fail the schema, so every baseline column is required and typed as a
number in the row schema.

Covers:
  * POSITIVE: the committed comparison artifact validates against the
    draft-2020-12 JSON schema with zero validation errors;
  * NEGATIVE: a deep-copy of the committed artifact with one baseline column
    deleted from one row FAILS the validator (the explicit plan requirement);
  * NEGATIVE: an extra unexpected top-level key fails (closed schema).

The schema is checked as a well-formed draft-2020-12 schema before use.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import jsonschema

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

_SCHEMA_PATH = _HERE / "schemas" / "cache_replay_comparison.schema.json"
_COMPARISON_PATH = _REPO_ROOT / "data/local_labels/cache_replay_comparison.json"

# Every per-cell baseline column the row schema must require as a typed number.
_BASELINE_COLUMNS = (
    "v5_prediction",
    "nugteren_prediction",
    "uncalibrated_walker_prediction",
    "v4_platt_prediction",
    "cohort_mean_prediction",
    "always_hit_prediction",
)


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _schema_validator() -> jsonschema.Draft202012Validator:
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _load_artifact() -> dict:
    return json.loads(_COMPARISON_PATH.read_text())


class SchemaIsWellFormedTest(unittest.TestCase):
    def test_schema_is_draft_2020_12_and_closed(self):
        schema = _load_schema()
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertFalse(schema["additionalProperties"])
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_row_schema_requires_every_baseline_column(self):
        schema = _load_schema()
        required = set(schema["$defs"]["row"]["required"])
        for column in _BASELINE_COLUMNS:
            self.assertIn(column, required)
            self.assertEqual(
                schema["$defs"]["row"]["properties"][column]["type"], "number",
            )


class CommittedArtifactTest(unittest.TestCase):
    def test_artifact_exists(self):
        self.assertTrue(
            _COMPARISON_PATH.exists(),
            msg=f"committed comparison artifact not found at {_COMPARISON_PATH}",
        )

    def test_committed_artifact_validates_with_zero_errors(self):
        artifact = _load_artifact()
        validator = _schema_validator()
        errors = sorted(validator.iter_errors(artifact), key=str)
        self.assertEqual(errors, [], msg="; ".join(e.message for e in errors))

    def test_every_row_carries_all_baseline_columns(self):
        artifact = _load_artifact()
        self.assertTrue(artifact["rows"])
        for row in artifact["rows"]:
            for column in _BASELINE_COLUMNS:
                self.assertIn(column, row)
                self.assertIsInstance(row[column], (int, float))


class NegativeSchemaTest(unittest.TestCase):
    def test_missing_baseline_column_in_one_row_fails(self):
        validator = _schema_validator()
        # Deleting any single baseline column from any single cell must fail.
        for column in _BASELINE_COLUMNS:
            artifact = copy.deepcopy(_load_artifact())
            del artifact["rows"][0][column]
            self.assertFalse(
                validator.is_valid(artifact),
                msg=f"deleting {column!r} from a row should fail validation",
            )
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(artifact)

    def test_extra_top_level_key_fails_closed_schema(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["unexpected_field"] = 1
        validator = _schema_validator()
        self.assertFalse(validator.is_valid(artifact))
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(artifact)

    def test_extra_key_inside_a_row_fails_closed_schema(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["rows"][0]["unexpected_row_field"] = 0
        validator = _schema_validator()
        self.assertFalse(validator.is_valid(artifact))

    def test_29_row_cohort_fails_exact_full_cohort_cardinality(self):
        # Exact full-cohort cardinality: trimming to 29 rows (and editing
        # cohort_size to match) must still fail — the cohort is fixed at 30.
        artifact = copy.deepcopy(_load_artifact())
        artifact["rows"] = artifact["rows"][:29]
        artifact["cohort_size"] = 29
        self.assertFalse(_schema_validator().is_valid(artifact))

    def test_cohort_size_other_than_30_fails(self):
        artifact = copy.deepcopy(_load_artifact())
        artifact["cohort_size"] = 31
        self.assertFalse(_schema_validator().is_valid(artifact))


if __name__ == "__main__":
    unittest.main()
