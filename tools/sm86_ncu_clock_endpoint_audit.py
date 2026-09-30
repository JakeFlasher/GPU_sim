#!/usr/bin/env python3
"""Audit NCU-local frequency/cycle/duration endpoint consistency.

This is deliberately an observation audit, not a predictor: NCU duration/GPC
labels and NCU frequency counters are never fed into simulator fitting. It
quantifies how closely NCU's own per-kernel frequency metric agrees with the
ratio of NCU GPC cycles to NCU duration, then emits the residual as endpoint
uncertainty. The relation does not establish CUDA-event, CUPTI, host-timer, or
simulator clock equivalence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time


def num(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    with (args.campaign / "performance.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("status") == "completed"]
    observations = []
    for row in rows:
        duration = num(row, "ncu_duration_ns")
        cycles = num(row, "ncu_gpc_cycles")
        frequency_ghz = num(row, "gpc__cycles_elapsed.avg.per_second")
        if not all(value is not None and value > 0 for value in (duration, cycles, frequency_ghz)):
            continue
        observed_mhz = frequency_ghz * 1000.0
        implied_mhz = cycles / duration * 1000.0
        relative_error = (implied_mhz - observed_mhz) / observed_mhz
        observations.append({
            "workload": row["workload"],
            "ncu_duration_ns_score_only": duration,
            "ncu_gpc_cycles_score_only": cycles,
            "ncu_gpc_frequency_ghz_score_only": frequency_ghz,
            "implied_frequency_mhz": implied_mhz,
            "relative_frequency_residual": relative_error,
            "absolute_frequency_residual_percent": abs(relative_error) * 100.0,
        })
    residuals = [item["absolute_frequency_residual_percent"] for item in observations]
    report = {
        "schema": "sm86_ncu_clock_endpoint_audit_v1",
        "campaign": str(args.campaign),
        "status": "observation_only",
        "workload_count": len(rows),
        "clock_observation_count": len(observations),
        "ncu_labels_score_only": True,
        "metrics": {
            "median_absolute_frequency_residual_percent": statistics.median(residuals) if residuals else None,
            "p90_absolute_frequency_residual_percent": (
                sorted(residuals)[max(0, int(math.ceil(0.9 * len(residuals))) - 1)]
                if residuals else None
            ),
            "max_absolute_frequency_residual_percent": max(residuals) if residuals else None,
            "median_ncu_gpc_frequency_ghz": statistics.median(
                [item["ncu_gpc_frequency_ghz_score_only"] for item in observations]
            ) if observations else None,
        },
        "observations": observations,
        "runtime_seconds": time.perf_counter() - start,
        "interpretation": [
            "NCU-local frequency/cycle/duration relation only",
            "does not prove CUDA-event/CUPTI/host/device/simulator clock equivalence",
            "does not expose physical pipe, issue-slot, cache-owner, read-from, or DRAM-completion state",
            "NCU fields are score-only and are not model features or tuning inputs",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"campaign": report["campaign"], "workload_count": report["workload_count"], "clock_observation_count": report["clock_observation_count"], "metrics": report["metrics"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
