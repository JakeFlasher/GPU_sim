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
    document = parse_ncu_csv(path, "gpc__cycles_elapsed.max", "canary", "ncu_test")
    assert document["schema"] == "ncu_observation_v1"
    assert document["rows"] == [{
        "workload": "canary",
        "value": 5270.0,
        "metric": "gpc__cycles_elapsed.max",
        "source": "ncu_test",
    }]
