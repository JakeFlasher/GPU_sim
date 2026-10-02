from __future__ import annotations

from pathlib import Path

import pytest

from run_prediction_manifest import run_manifest


CONTRACT = {
    "metric": "gpc__cycles_elapsed.max",
    "endpoint": "ncu_grid_completion_semaphore",
    "clock_domain": "gpc_cycles",
    "cache_policy": "launch",
    "launch_identity": "fixture_launch",
    "repetitions": 1,
    "hardware": {
        "gpu_uuid": "GPU-fixture",
        "compute_capability": "8.6",
        "sm_clock_mhz": 1350,
        "device_index": 0,
    },
}


def _row(workload: str, family: str, split: str) -> dict[str, str]:
    return {
        "workload": workload,
        "family": family,
        "split": split,
        "trace": "trace",
        "control": "control",
        "arch_card": "arch",
        "pchase_card": "pchase",
    }


def test_manifest_rejects_family_crossing_splits_before_running_simulator(tmp_path: Path):
    manifest = {
        "schema": "reification_prediction_manifest_v1",
        "target_contract": CONTRACT,
        "rows": [_row("a", "same-family", "calibration"), _row("b", "same-family", "held_out")],
    }
    with pytest.raises(ValueError, match="family .* crosses prediction splits"):
        run_manifest(manifest, tmp_path / "missing-sim-run", tmp_path / "profile", 1.0)


def test_manifest_rejects_missing_target_contract_before_running_simulator(tmp_path: Path):
    manifest = {
        "schema": "reification_prediction_manifest_v1",
        "rows": [_row("a", "family", "held_out")],
    }
    with pytest.raises(ValueError, match="target_contract"):
        run_manifest(manifest, tmp_path / "missing-sim-run", tmp_path / "profile", 1.0)
