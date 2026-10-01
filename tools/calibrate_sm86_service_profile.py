#!/usr/bin/env python3
"""Fit an abstract service-rate profile from independent probe observations.

The input intentionally contains only probe observations (pool, event count,
cycles, and optional latency).  It does not accept NCU score rows or workload
identity fields, so it cannot silently turn a scored workload into a feature.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def fit_profile(raw: Any, source: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != "sm86_service_probe_v1":
        raise ValueError("input schema must be sm86_service_probe_v1")
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
        if not isinstance(events, int) or events <= 0:
            raise ValueError(f"probe {index}.events must be positive")
        if not isinstance(cycles, int) or cycles <= 0:
            raise ValueError(f"probe {index}.cycles must be positive")
        latency = probe.get("latency_cycles", max(1, cycles // events))
        if not isinstance(latency, int) or latency <= 0:
            raise ValueError(f"probe {index}.latency_cycles must be positive")
        capacity = probe.get("capacity", 1)
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"probe {index}.capacity must be positive")
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
            "latency_lower_cycles": min(latencies),
            "latency_upper_cycles": max(latencies),
        }
    return {
        "schema_version": 1,
        "source": source,
        "pools": pools,
        "ncu_start_padding_cycles": None,
        "ncu_end_padding_cycles": None,
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
