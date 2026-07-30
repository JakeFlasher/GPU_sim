import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest


RUNNER_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = RUNNER_ROOT / "schema"
FIXTURE_ROOT = RUNNER_ROOT / "tests" / "fixtures" / "addr_carry_canary_bundle"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(document: dict, seal_field: str) -> dict:
    document.pop(seal_field, None)
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document[seal_field] = hashlib.sha256(canonical).hexdigest()
    return document


def _sealed_error(
    *, code: str, category: str, exit_code: int, message: str = "typed error"
) -> dict:
    return _reseal(
        {
            "schema_version": "sim_error.v1",
            "code": code,
            "category": category,
            "exit_code": exit_code,
            "message": message,
        },
        "error_sha256",
    )


def _object_paths(value: object, path: tuple[object, ...] = ()):
    if isinstance(value, dict):
        yield path
        for key, child in value.items():
            yield from _object_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _object_paths(child, (*path, index))


def _resolve_path(value: object, path: tuple[object, ...]) -> object:
    current = value
    for component in path:
        current = current[component]
    return current


def test_runner_schemas_are_valid_and_accept_canonical_fixtures() -> None:
    manifest_schema = _load(SCHEMA_ROOT / "sim_run_manifest.v1.schema.json")
    result_schema = _load(SCHEMA_ROOT / "sim_result.v1.schema.json")
    error_schema = _load(SCHEMA_ROOT / "sim_error.v1.schema.json")
    for schema in (manifest_schema, result_schema, error_schema):
        jsonschema.Draft7Validator.check_schema(schema)

    jsonschema.validate(
        _load(FIXTURE_ROOT / "run_manifest.json"), manifest_schema
    )
    jsonschema.validate(
        _load(FIXTURE_ROOT / "result.golden.json"), result_schema
    )
    jsonschema.validate(
        _sealed_error(
            code="configuration_unavailable",
            category="prerequisite_missing",
            exit_code=2,
            message="typed prerequisite is absent",
        ),
        error_schema,
    )
    jsonschema.validate(
        _sealed_error(
            code="configuration_rejected",
            category="failure",
            exit_code=1,
            message="supplied card value is contradictory",
        ),
        error_schema,
    )


def test_manifest_schema_rejects_unknown_and_target_like_inputs() -> None:
    schema = _load(SCHEMA_ROOT / "sim_run_manifest.v1.schema.json")
    manifest = _load(FIXTURE_ROOT / "run_manifest.json")
    contaminated = copy.deepcopy(manifest)
    contaminated["labels"] = {"forbidden_target_value": 1}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(contaminated, schema)

    bad_path = copy.deepcopy(manifest)
    bad_path["launches"][0]["launch_evidence_manifest_path"] = "../escape.json"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad_path, schema)


def test_result_schema_pins_registry_and_blocks_binding_cycles() -> None:
    schema = _load(SCHEMA_ROOT / "sim_result.v1.schema.json")
    result = _load(FIXTURE_ROOT / "result.golden.json")

    predicted_cycles = copy.deepcopy(result)
    predicted_cycles["emissions"][-1]["outcome"] = {
        "status": "predicted",
        "value": {
            "kind": "exact",
            "semantics": "simulator_scope",
            "value": 1,
        },
    }
    _reseal(predicted_cycles, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(predicted_cycles, schema)

    omitted_cycles = copy.deepcopy(result)
    omitted_cycles["emissions"][-1] = copy.deepcopy(
        omitted_cycles["emissions"][0]
    )
    _reseal(omitted_cycles, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(omitted_cycles, schema)

    duplicate_cycles = copy.deepcopy(result)
    duplicate_cycles["emissions"].append(
        copy.deepcopy(duplicate_cycles["emissions"][-1])
    )
    _reseal(duplicate_cycles, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(duplicate_cycles, schema)


def test_result_schema_rejects_aggregate_cycles_and_nested_targets() -> None:
    schema = _load(SCHEMA_ROOT / "sim_result.v1.schema.json")
    result = _load(FIXTURE_ROOT / "result.golden.json")

    aggregate_cycles = copy.deepcopy(result)
    aggregate_cycles["aggregate"]["cycles"] = 1
    _reseal(aggregate_cycles, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(aggregate_cycles, schema)

    admitted_labels = copy.deepcopy(result)
    admitted_labels["launches"][0]["admitted_identity"]["labels"] = {
        "gpc__cycles_elapsed.max": 1
    }
    _reseal(admitted_labels, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(admitted_labels, schema)

    diagnostic_target = copy.deepcopy(result)
    diagnostic_target["launches"][0]["timing_diagnostics"][
        "target_cycles"
    ] = 1
    _reseal(diagnostic_target, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(diagnostic_target, schema)

    emission_target = copy.deepcopy(result)
    emission_target["emissions"][0]["outcome"]["target_value"] = 1
    _reseal(emission_target, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(emission_target, schema)

    configuration_label = copy.deepcopy(result)
    configuration_label["resolved_configuration"]["issue"]["ncu_label"] = 1
    _reseal(configuration_label, "result_sha256")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(configuration_label, schema)


def test_result_schema_closes_every_golden_object_to_target_injection() -> None:
    schema = _load(SCHEMA_ROOT / "sim_result.v1.schema.json")
    result = _load(FIXTURE_ROOT / "result.golden.json")
    validator = jsonschema.Draft7Validator(schema)
    object_paths = list(_object_paths(result))
    assert len(object_paths) >= 50

    for path in object_paths:
        contaminated = copy.deepcopy(result)
        target = _resolve_path(contaminated, path)
        assert isinstance(target, dict)
        target["forbidden_target_value"] = 1
        _reseal(contaminated, "result_sha256")
        assert not validator.is_valid(contaminated), path


def test_error_schema_binds_code_category_and_exit_code() -> None:
    schema = _load(SCHEMA_ROOT / "sim_error.v1.schema.json")
    contradictory_documents = [
        _sealed_error(
            code="configuration_unavailable",
            category="failure",
            exit_code=1,
        ),
        _sealed_error(
            code="configuration_rejected",
            category="prerequisite_missing",
            exit_code=2,
        ),
        _sealed_error(
            code="run_manifest_read",
            category="failure",
            exit_code=2,
        ),
        _sealed_error(
            code="launch_evidence_rejected",
            category="prerequisite_missing",
            exit_code=1,
        ),
    ]
    for document in contradictory_documents:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)
