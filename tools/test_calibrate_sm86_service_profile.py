from __future__ import annotations

import pytest

from calibrate_sm86_service_profile import fit_profile


def test_fit_profile_uses_probe_only_service_observations() -> None:
    profile = fit_profile(
        {
            "schema": "sm86_service_probe_v1",
            "probes": [
                {"pool": "alu", "events": 10, "cycles": 20, "latency_cycles": 2},
                {"pool": "alu", "events": 10, "cycles": 30, "latency_cycles": 3},
                {"pool": "lsu", "events": 4, "cycles": 40, "latency_cycles": 10},
            ],
        },
        "test_probe_fit",
    )
    assert profile["pools"]["alu"]["initiation_interval_cycles"] == 2
    assert profile["pools"]["alu"]["latency_lower_cycles"] == 2
    assert profile["pools"]["alu"]["latency_upper_cycles"] == 3
    assert profile["pools"]["lsu"]["capacity"] == 1


def test_fit_profile_rejects_mixed_pool_capacity() -> None:
    with pytest.raises(ValueError):
        fit_profile(
            {
                "schema": "sm86_service_probe_v1",
                "probes": [
                    {"pool": "alu", "events": 1, "cycles": 1, "capacity": 1},
                    {"pool": "alu", "events": 1, "cycles": 1, "capacity": 2},
                ],
            },
            "bad",
        )
