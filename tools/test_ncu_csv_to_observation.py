from __future__ import annotations

from pathlib import Path

from ncu_csv_to_observation import parse_ncu_csv


def test_ncu_csv_adapter_skips_profiler_preamble_and_units(tmp_path: Path):
    path = tmp_path / "ncu.csv"
    path.write_text(
        "==PROF== Connected\n"
        '"ID","gpc__cycles_elapsed.max"\n'
        '"","cycle"\n'
        '"0","5270"\n'
    )
    document = parse_ncu_csv(
        path,
        "gpc__cycles_elapsed.max",
        "canary",
        "ncu_test",
        endpoint="ncu_grid_completion_semaphore",
        clock_domain="gpc_cycles",
        cache_policy="launch",
        launch_identity="fixture_launch",
        gpu_uuid="GPU-fixture",
        compute_capability="8.6",
        sm_clock_mhz=1350,
        device_index=0,
    )
    assert document["schema"] == "ncu_observation_v2"
    assert document["target_contract"]["metric"] == "gpc__cycles_elapsed.max"
    assert document["rows"] == [{
        "workload": "canary",
        "value": 5270.0,
        "metric": "gpc__cycles_elapsed.max",
        "endpoint": "ncu_grid_completion_semaphore",
        "clock_domain": "gpc_cycles",
        "hardware": {
            "gpu_uuid": "GPU-fixture",
            "compute_capability": "8.6",
            "sm_clock_mhz": 1350,
            "device_index": 0,
        },
        "source": "ncu_test",
        "identity_status": "caller_labeled_unverified",
    }]
