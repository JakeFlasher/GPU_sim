from __future__ import annotations

import pytest

from calibrate_sm86_service_profile import evaluate_profile_on_holdout, fit_profile


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


def test_v2_requires_separate_identifiable_probe_modes_and_marks_capacity_slope() -> None:
    profile = fit_profile(
        {
            "schema": "sm86_service_probe_v2",
            "probes": [
                {"pool": "alu", "mode": "dependent_latency", "events": 8, "cycles": 32},
                {
                    "pool": "alu",
                    "mode": "independent_throughput",
                    "events": 16,
                    "cycles": 48,
                    "capacity": 2,
                },
                {
                    "pool": "alu",
                    "mode": "queue_capacity",
                    "events": 16,
                    "cycles": 48,
                    "capacity": 2,
                    "queue_depth": 8,
                },
                {"pool": "alu", "mode": "occupancy_sweep", "events": 8, "cycles": 32, "occupancy": 1},
                {"pool": "alu", "mode": "contention_sweep", "events": 8, "cycles": 36, "contenders": 2},
            ],
        },
        "test_probe_fit_v2",
    )
    assert profile["pools"]["alu"]["capacity_already_applied"] is True
    assert profile["pools"]["alu"]["initiation_interval_cycles"] == 3
    assert profile["pools"]["alu"]["latency_lower_cycles"] == 4


def test_v2_rejects_an_aggregate_only_pool() -> None:
    with pytest.raises(ValueError, match="missing probe modes"):
        fit_profile(
            {
                "schema": "sm86_service_probe_v2",
                "probes": [
                    {
                        "pool": "alu",
                        "mode": "independent_throughput",
                        "events": 16,
                        "cycles": 48,
                        "capacity": 2,
                    }
                ],
            },
            "bad_v2",
        )


def test_v2_holdout_scores_only_identifiable_probe_modes() -> None:
    train = {
        "schema": "sm86_service_probe_v2",
        "probes": [
            {"pool": "alu", "mode": "dependent_latency", "events": 8, "cycles": 32},
            {"pool": "alu", "mode": "independent_throughput", "events": 16, "cycles": 48, "capacity": 2},
            {"pool": "alu", "mode": "queue_capacity", "events": 16, "cycles": 48, "capacity": 2, "queue_depth": 8},
            {"pool": "alu", "mode": "occupancy_sweep", "events": 8, "cycles": 32, "occupancy": 1},
            {"pool": "alu", "mode": "contention_sweep", "events": 8, "cycles": 36, "contenders": 2},
        ],
    }
    profile = fit_profile(train, "train")
    holdout = {
        "schema": "sm86_service_probe_v2",
        "probes": [
            {"pool": "alu", "mode": "dependent_latency", "events": 4, "cycles": 16},
            {"pool": "alu", "mode": "independent_throughput", "events": 8, "cycles": 24, "capacity": 2},
            {"pool": "alu", "mode": "occupancy_sweep", "events": 8, "cycles": 24, "occupancy": 2},
        ],
    }
    report = evaluate_profile_on_holdout(profile, holdout)
    assert report["identifiable_count"] == 2
    assert report["covered_count"] == 2
    assert report["mean_absolute_error_cycles"] == 0
    assert report["observations"][2]["status"] == "unmodeled"
