from __future__ import annotations

import json
from pathlib import Path

import pytest
jsonschema = pytest.importorskip("jsonschema")

from preflight_sm86_ncu_campaign import preflight_campaign, summarize_repetitions, validate_run_records


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads((ROOT / "docs/sm86_ncu_accuracy_protocol_v1.json").read_text())


def _fixture(tmp_path: Path):
    protocol = _protocol()
    paths = {}
    for field in ("trace", "control", "arch_card", "pchase_card"):
        path = tmp_path / field
        path.write_text("fixture")
        paths[field] = str(path)
    manifest = {
        "schema": "reification_prediction_manifest_v1",
        "target_contract": {
            "metric": "gpc__cycles_elapsed.max",
            "endpoint": "ncu_grid_completion_semaphore",
            "clock_domain": "gpc_cycles",
            "cache_policy": "all",
            "launch_identity": "sm86_accuracy_campaign_v1",
            "repetitions": 5,
            "hardware": {
                "gpu_uuid": "GPU-fixture",
                "compute_capability": "8.6",
                "sm_clock_mhz": 1350,
                "device_index": 0,
            },
        },
        "rows": [{"workload": "w", "family": "normalization", "split": "calibration", **paths}],
    }
    binding = {"workload": "w", **paths, "launch_identity": "sm86_accuracy_campaign_v1"}
    prediction = {
        "schema": "reification_prediction_v2",
        "source_bindings": [binding],
        "target_contract": manifest["target_contract"],
        "rows": [{
            "workload": "w", "family": "normalization", "split": "calibration", "status": "predicted",
            "prediction": {"point": 10, "lower": 9, "upper": 11},
            "point_basis": "fixture",
            "observer_projection": {
                "target_metric": "gpc__cycles_elapsed.max", "role": "fixture", "conversion": "cycles",
                "simulator_lower_cycles": 9, "simulator_upper_cycles": 11,
                "predicted_lower_cycles": 9, "predicted_upper_cycles": 11,
            },
        }],
    }
    identity = {
        "schema": "sm86_ncu_identity_v1", "status": "independently_verified", "eligible_for_accuracy": True,
        "measurement_contract": protocol["measurement_contract"],
        "hardware": {"gpu_name": "NVIDIA GeForce RTX 3090 Ti", "gpu_uuid": "GPU-fixture", "compute_capability": "8.6", "driver_version": "615.71.09", "device_index": 0, "graphics_clock_mhz": 1350, "memory_clock_mhz": 10501, "power_state": "P0"},
        "build": {"working_tree_dirty": False},
        "metric_availability": {"available": True},
    }
    return protocol, identity, manifest, prediction


def test_preflight_ready_fixture_has_no_collection_side_effect(tmp_path: Path):
    protocol, identity, manifest, prediction = _fixture(tmp_path)
    report = preflight_campaign(protocol, identity, manifest, prediction=prediction)
    assert report["ready_for_collection"] is True
    assert report["collection_performed"] is False
    assert report["repetition_policy"]["minimum"] == 5


def test_preflight_requires_prediction_before_collection(tmp_path: Path):
    protocol, identity, manifest, _prediction = _fixture(tmp_path)
    report = preflight_campaign(protocol, identity, manifest)
    assert report["ready_for_collection"] is False
    assert report["prediction_status"] == "not_checked"


def test_preflight_labels_identity_or_contract_mismatch(tmp_path: Path):
    protocol, identity, manifest, prediction = _fixture(tmp_path)
    identity["status"] = "mismatch"
    identity["eligible_for_accuracy"] = False
    report = preflight_campaign(protocol, identity, manifest, prediction=prediction)
    assert report["ready_for_collection"] is False
    assert "identity_ineligible" in report["mismatches"]


def test_preflight_reports_observation_present_before_collection(tmp_path: Path):
    protocol, identity, manifest, prediction = _fixture(tmp_path)
    observation = tmp_path / "observation.json"
    observation.write_text("{}")
    report = preflight_campaign(protocol, identity, manifest, prediction=prediction, observation_path=observation)
    assert report["ready_for_collection"] is False
    assert "observation_already_present_before_collection" in report["mismatches"]


def test_repetition_summary_reports_all_samples_and_iqr():
    report = summarize_repetitions([10, 14, 12, 11, 13])
    assert report["count"] == 5
    assert report["median"] == 12
    assert report["iqr"] == 2
    assert report["all_samples"] == [10, 11, 12, 13, 14]


def test_preflight_report_schema_is_valid():
    schema = json.loads((ROOT / "docs/sm86_ncu_campaign_preflight_v1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)


def test_run_records_require_termination_and_timeout_compliance():
    result = validate_run_records(
        [{"returncode": 0, "timed_out": False, "elapsed_seconds": 2, "value": value} for value in [10, 11, 12, 13, 14]],
        minimum=5,
        timeout_seconds=10,
    )
    assert result["termination_ok"] is True
    assert result["repetitions"]["median"] == 12
    failed = validate_run_records(
        [{"returncode": 1, "timed_out": False, "elapsed_seconds": 20, "value": 10}],
        minimum=5,
        timeout_seconds=10,
    )
    assert failed["termination_ok"] is False
    assert failed["repetitions"] is None
