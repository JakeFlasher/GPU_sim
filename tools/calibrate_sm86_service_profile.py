#!/usr/bin/env python3
"""Fit an abstract service-rate profile from independent probe observations.

``sm86_service_probe_v1`` remains a compatibility format for old probe
documents, but it still requires an explicit dependent latency; aggregate
cycles/events never identify that quantity. New work should use
``sm86_service_probe_v2`` and provide explicit
dependent-chain latency and independent-stream throughput observations. Queue
depth/capacity, occupancy, and contention observations are retained as
interventions, but the current profile has no queue equation; they are
reported as unmodeled rather than being called calibrated state. Dependent
latency must be supplied explicitly and is never inferred from aggregate
throughput. The fitter never accepts NCU scores or workload identity fields.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def fit_profile(raw: Any, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") not in {
        "sm86_service_probe_v1",
        "sm86_service_probe_v2",
    }:
        raise ValueError("input schema must be sm86_service_probe_v1 or sm86_service_probe_v2")
    if raw["schema"] == "sm86_service_probe_v2":
        return _fit_v2(raw, source)
    return _fit_v1(raw, source)


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _fit_v1(raw: dict[str, Any], source: str) -> dict[str, Any]:
    probes = raw.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("probes must be a nonempty list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise ValueError(f"probe {index} must be an object")
        pool = probe.get("pool")
        if not isinstance(pool, str) or not pool.strip():
            raise ValueError(f"probe {index}.pool must be nonempty")
        events = probe.get("events")
        cycles = probe.get("cycles")
        events = _positive_int(events, f"probe {index}.events")
        cycles = _positive_int(cycles, f"probe {index}.cycles")
        if "latency_cycles" not in probe:
            raise ValueError(
                f"probe {index}.latency_cycles is required; aggregate cycles/events cannot identify dependent latency"
            )
        latency = _positive_int(probe["latency_cycles"], f"probe {index}.latency_cycles")
        capacity = probe.get("capacity", 1)
        capacity = _positive_int(capacity, f"probe {index}.capacity")
        grouped.setdefault(pool, []).append(
            {"events": events, "cycles": cycles, "latency": latency, "capacity": capacity}
        )

    pools: dict[str, Any] = {}
    for pool, entries in sorted(grouped.items()):
        intervals = [max(1, (entry["cycles"] + entry["events"] - 1) // entry["events"]) for entry in entries]
        latencies = [entry["latency"] for entry in entries]
        capacities = {entry["capacity"] for entry in entries}
        if len(capacities) != 1:
            raise ValueError(f"probe capacity differs for pool {pool}")
        pools[pool] = {
            "capacity": capacities.pop(),
            "initiation_interval_cycles": max(1, round(statistics.median(intervals))),
            "initiation_interval_upper_cycles": max(intervals),
            "capacity_already_applied": False,
            "latency_lower_cycles": min(latencies),
            "latency_upper_cycles": max(latencies),
        }
    return {
        "schema_version": 1,
        "source": source,
        "pools": pools,
        "ncu_start_padding_cycles": None,
        "ncu_end_padding_cycles": None,
        "unmodeled_dimensions": [],
    }


V2_MODES = {
    "dependent_latency",
    "independent_throughput",
    "queue_capacity",
    "occupancy_sweep",
    "contention_sweep",
}


def _fit_v2(raw: dict[str, Any], source: str) -> dict[str, Any]:
    probes = raw.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("probes must be a nonempty list")
    grouped: dict[str, dict[str, list[dict[str, int]]]] = {}
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise ValueError(f"probe {index} must be an object")
        pool = probe.get("pool")
        if not isinstance(pool, str) or not pool.strip():
            raise ValueError(f"probe {index}.pool must be nonempty")
        mode = probe.get("mode")
        if mode not in V2_MODES:
            raise ValueError(f"probe {index}.mode must be one of {sorted(V2_MODES)}")
        events = _positive_int(probe.get("events"), f"probe {index}.events")
        cycles = _positive_int(probe.get("cycles"), f"probe {index}.cycles")
        entry: dict[str, int] = {"events": events, "cycles": cycles}
        if mode in {"independent_throughput", "queue_capacity"}:
            entry["capacity"] = _positive_int(
                probe.get("capacity"), f"probe {index}.capacity"
            )
        if mode == "dependent_latency":
            if "latency_cycles" not in probe:
                raise ValueError(
                    f"probe {index}.latency_cycles is required for dependent_latency"
                )
            entry["latency"] = _positive_int(
                probe["latency_cycles"], f"probe {index}.latency_cycles"
            )
        if mode == "queue_capacity":
            entry["queue_depth"] = _positive_int(
                probe.get("queue_depth"), f"probe {index}.queue_depth"
            )
        if mode == "occupancy_sweep":
            entry["occupancy"] = _positive_int(
                probe.get("occupancy"), f"probe {index}.occupancy"
            )
        if mode == "contention_sweep":
            entry["contenders"] = _positive_int(
                probe.get("contenders"), f"probe {index}.contenders"
            )
        grouped.setdefault(pool, {}).setdefault(mode, []).append(entry)

    pools: dict[str, Any] = {}
    for pool, modes in sorted(grouped.items()):
        required = {"dependent_latency", "independent_throughput"}
        missing = required - modes.keys()
        if missing:
            raise ValueError(f"pool {pool} is missing probe modes: {sorted(missing)}")
        queue_entries = modes.get("queue_capacity", [])
        capacities = {
            entry["capacity"]
            for entry in [*modes["independent_throughput"], *queue_entries]
            if "capacity" in entry
        }
        throughput_capacities = {
            entry["capacity"] for entry in modes["independent_throughput"]
        }
        if len(throughput_capacities) != 1 or (
            capacities and throughput_capacities != capacities
        ):
            raise ValueError(f"pool {pool} has no consistent throughput-capacity witness")
        throughput_intervals = [
            max(1, (entry["cycles"] + entry["events"] - 1) // entry["events"])
            for entry in modes["independent_throughput"]
        ]
        median_interval = max(1, round(statistics.median(throughput_intervals)))
        if max(throughput_intervals) - min(throughput_intervals) > max(1, median_interval // 2):
            raise ValueError(f"pool {pool} throughput probes are not identifiable")
        latencies = [entry["latency"] for entry in modes["dependent_latency"]]
        pools[pool] = {
            "capacity": capacities.pop(),
            # Throughput probes report an already capacity-adjusted slope;
            # projection must not divide it by capacity again.
            "capacity_already_applied": True,
            "initiation_interval_cycles": median_interval,
            "initiation_interval_upper_cycles": max(throughput_intervals),
            "latency_lower_cycles": min(latencies),
            "latency_upper_cycles": max(latencies),
        }
    return {
        "schema_version": 1,
        "source": source,
        "pools": pools,
        "ncu_start_padding_cycles": None,
        "ncu_end_padding_cycles": None,
            "unmodeled_dimensions": ["queue_depth", "occupancy", "contention"],
    }


def evaluate_profile_on_holdout(profile: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Score identifiable probe modes on a separate v2 probe document.

    This is intentionally a probe-only check. Queue capacity, occupancy, and
    contention rows are reported as unmodeled rather than converted into a
    guessed timing correction.
    """
    if not isinstance(raw, dict) or raw.get("schema") != "sm86_service_probe_v2":
        raise ValueError("holdout schema must be sm86_service_probe_v2")
    probes = raw.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("holdout probes must be a nonempty list")
    observations: list[dict[str, Any]] = []
    absolute_errors: list[int] = []
    covered = 0
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise ValueError(f"holdout probe {index} must be an object")
        pool = probe.get("pool")
        mode = probe.get("mode")
        if not isinstance(pool, str) or mode not in V2_MODES:
            raise ValueError(f"holdout probe {index} has invalid pool or mode")
        rate = profile.get("pools", {}).get(pool)
        if not isinstance(rate, dict):
            raise ValueError(f"holdout pool {pool!r} is absent from the fitted profile")
        if mode == "dependent_latency":
            events = _positive_int(probe.get("events"), f"holdout probe {index}.events")
            cycles = _positive_int(probe.get("cycles"), f"holdout probe {index}.cycles")
            if "latency_cycles" not in probe:
                raise ValueError(
                    f"holdout probe {index}.latency_cycles is required for dependent_latency"
                )
            observed = _positive_int(
                probe["latency_cycles"],
                f"holdout probe {index}.latency_cycles",
            )
            lower = rate["latency_lower_cycles"]
            upper = rate["latency_upper_cycles"]
            error = 0 if lower <= observed <= upper else min(abs(observed - lower), abs(observed - upper))
            covered += int(error == 0)
            absolute_errors.append(error)
            observations.append({"index": index, "mode": mode, "observed": observed, "error": error})
        elif mode == "independent_throughput":
            events = _positive_int(probe.get("events"), f"holdout probe {index}.events")
            cycles = _positive_int(probe.get("cycles"), f"holdout probe {index}.cycles")
            observed = (cycles + events - 1) // events
            predicted_lower = rate["initiation_interval_cycles"]
            predicted_upper = rate.get("initiation_interval_upper_cycles") or predicted_lower
            covered_interval = predicted_lower <= observed <= predicted_upper
            error = (
                0
                if covered_interval
                else min(abs(observed - predicted_lower), abs(observed - predicted_upper))
            )
            covered += int(covered_interval)
            absolute_errors.append(error)
            observations.append(
                {
                    "index": index,
                    "mode": mode,
                    "observed": observed,
                    "predicted_lower": predicted_lower,
                    "predicted_upper": predicted_upper,
                    "covered": covered_interval,
                    "error": error,
                }
            )
        else:
            observations.append({
                "index": index,
                "mode": mode,
                "status": "unmodeled",
                "reason": "queue_depth_not_in_profile_equation"
                if mode == "queue_capacity"
                else "probe_dimension_not_in_profile_equation",
                "queue_depth": probe.get("queue_depth") if mode == "queue_capacity" else None,
            })
    return {
        "schema": "sm86_service_holdout_v1",
        "observations": observations,
        "identifiable_count": len(absolute_errors),
        "covered_count": covered,
        "interval_coverage": covered / len(absolute_errors) if absolute_errors else None,
        "mean_absolute_error_cycles": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", default="independent_probe_fit_v1")
    args = parser.parse_args()
    raw = json.loads(args.probe_json.read_text())
    profile = fit_profile(raw, args.source)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
