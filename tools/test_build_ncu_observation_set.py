from __future__ import annotations

from pathlib import Path

import pytest

from build_ncu_observation_set import build_observation_set


CONTRACT = {
    "metric": "gpc__cycles_elapsed.max",
    "endpoint": "ncu_grid_completion_semaphore",
    "clock_domain": "gpc_cycles",
    "cache_policy": "all",
    "launch_identity": "fixture",
    "repetitions": 2,
    "hardware": {
        "gpu_uuid": "GPU-fixture",
        "compute_capability": "8.6",
        "sm_clock_mhz": 1350,
        "device_index": 0,
    },
}


def _csv(value: int) -> str:
    return f'"ID","gpc__cycles_elapsed.max"\n"","cycle"\n"0","{value}"\n'


def test_builder_keeps_two_repetitions_and_row_identity(tmp_path: Path):
    workload = tmp_path / "w"
    workload.mkdir()
    (workload / "ncu_0_csv.stdout").write_text(_csv(17))
    (workload / "ncu_1_csv.stdout").write_text(_csv(19))
    document = build_observation_set(
        {
            "schema": "reification_prediction_manifest_v1",
            "target_contract": CONTRACT,
            "rows": [{"workload": "w", "family": "f", "split": "held_out"}],
        },
        tmp_path,
        source="fixture",
    )
    assert [row["value"] for row in document["rows"]] == [17.0, 19.0]
    assert all(row["hardware"] == CONTRACT["hardware"] for row in document["rows"])
    assert document["identity_status"] == "manifest_labeled_historical"
    assert all(
        row["identity_status"] == "manifest_labeled_historical"
        for row in document["rows"]
    )


def test_builder_rejects_wrong_repetition_count(tmp_path: Path):
    workload = tmp_path / "w"
    workload.mkdir()
    (workload / "ncu_0_csv.stdout").write_text(_csv(17))
    with pytest.raises(ValueError, match="NCU CSV files"):
        build_observation_set(
            {
                "schema": "reification_prediction_manifest_v1",
                "target_contract": CONTRACT,
                "rows": [{"workload": "w", "family": "f", "split": "held_out"}],
            },
            tmp_path,
        )
