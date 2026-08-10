from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from eval.contracts.validation import load_json_file_strict


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATHS = (
    ROOT / "eval" / "schemas" / "model_seal.v1.schema.json",
    ROOT / "eval" / "schemas" / "launch_crosswalk.v1.schema.json",
    ROOT / "eval" / "schemas" / "state_lifetime.v1.schema.json",
    ROOT / "eval" / "schemas" / "preregistration_receipt.v1.schema.json",
    ROOT / "eval" / "schemas" / "prediction_seal_receipt.v1.schema.json",
    ROOT / "eval" / "schemas" / "label_opening_receipt.v1.schema.json",
    ROOT / "eval" / "schemas" / "workload_execution_authority.v1.schema.json",
    ROOT / "eval" / "schemas" / "launch_crosswalk.v2.schema.json",
    ROOT / "eval" / "schemas" / "production_ncu_repeat_manifest.v1.schema.json",
    ROOT / "eval" / "schemas" / "production_matched_label_authority.v1.schema.json",
    ROOT / "eval" / "schemas" / "runtime_workload_tier_claim_authority.v1.schema.json",
    ROOT / "eval" / "schemas" / "sim_profile_record.v1.schema.json",
    ROOT / "eval" / "schemas" / "production_sim_manifest_authority.v1.schema.json",
)
FIXTURE_SCHEMA_PATHS = SCHEMA_PATHS[1:6]
EVALUATION_BOUNDARY_SCHEMA_PATHS = SCHEMA_PATHS[1:6]
_FORBIDDEN_METRIC_PROPERTIES = {
    "metric_value",
    "metric_values",
    "observed_value",
    "predicted_value",
    "target_value",
    "score",
}


def test_all_evaluator_schemas_are_duplicate_rejecting_closed_draft_2020_12() -> None:
    for path in SCHEMA_PATHS:
        schema = load_json_file_strict(path)
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        _assert_closed_objects(schema, pointer="#")
        _assert_local_references(schema)
        _assert_no_numeric_metric_property_surface(schema)


def test_fixture_schemas_expose_no_admitted_or_acceptance_state() -> None:
    for path in FIXTURE_SCHEMA_PATHS:
        payload = path.read_bytes()
        assert b'"ADMITTED"' not in payload, path
        assert b'"ACCEPTED"' not in payload, path


def test_every_scientific_input_array_has_exact_order_and_closed_tail() -> None:
    expected = [
        "CYCLE_MAPPING",
        "CYCLE_COMPOSITION",
        "COHORT",
        "PROTOCOL_NUMERIC_SETTINGS",
        "COVERAGE",
        "TRUST",
        "OPERATOR_APPROVAL",
    ]
    for path in EVALUATION_BOUNDARY_SCHEMA_PATHS:
        schema = load_json_file_strict(path)
        arrays = list(_scientific_input_arrays(schema))
        assert arrays, path
        for array in arrays:
            assert array.get("minItems") == len(expected), path
            assert array.get("maxItems") == len(expected), path
            assert array.get("items") is False, path
            prefix = array.get("prefixItems")
            assert isinstance(prefix, list) and len(prefix) == len(expected), path
            observed = [
                item["properties"]["name"]["const"]
                for item in prefix
            ]
            assert observed == expected, path


def _scientific_input_arrays(value: Any):
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            scientific_inputs = properties.get("scientific_inputs")
            if isinstance(scientific_inputs, dict):
                yield scientific_inputs
        for child in value.values():
            yield from _scientific_input_arrays(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scientific_input_arrays(child)


def _assert_closed_objects(value: Any, *, pointer: str) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False, pointer
        for key, child in value.items():
            _assert_closed_objects(child, pointer=f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_closed_objects(child, pointer=f"{pointer}/{index}")


def _assert_local_references(value: Any) -> None:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None:
            assert isinstance(reference, str) and reference.startswith("#/")
        for child in value.values():
            _assert_local_references(child)
    elif isinstance(value, list):
        for child in value:
            _assert_local_references(child)


def _assert_no_numeric_metric_property_surface(value: Any) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            assert _FORBIDDEN_METRIC_PROPERTIES.isdisjoint(properties)
        for child in value.values():
            _assert_no_numeric_metric_property_surface(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_numeric_metric_property_surface(child)
