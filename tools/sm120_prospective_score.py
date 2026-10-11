#!/usr/bin/env python3
"""Score the frozen prospective cohort without relabelling targets as inputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    timing = json.loads(args.timing.read_text(encoding="utf-8"))
    full = {json.loads(line)["capture_name"]: json.loads(line) for line in args.full.read_text(encoding="utf-8").splitlines() if line.strip()}
    fast = {json.loads(line)["capture_name"]: json.loads(line) for line in args.fast.read_text(encoding="utf-8").splitlines() if line.strip()}
    timing_by_workload = {row.get("workload"): row for row in timing.get("rows", [])}
    rows = []
    errors = []
    for candidate in cohort.get("rows", []):
        name = candidate.get("workload")
        full_record = full.get(name, {})
        fast_record = fast.get(name, {})
        sim = full_record.get("simulator", {})
        fast_sim = fast_record.get("simulator", {})
        ncu_errors = []
        for error in sim.get("ncu_metric_errors", []) or []:
            value = error.get("absolute_relative_error")
            if isinstance(value, (int, float)):
                ncu_errors.append(abs(float(value)))
                errors.append(abs(float(value)))
        host_ns = timing_by_workload.get(name, {}).get("host_elapsed_ns")
        event_rows = timing_by_workload.get(name, {}).get("sm120_event_markers", []) or []
        event_ms = float(event_rows[0].get("elapsed_ms")) if event_rows else None
        full_wall_ns = sim.get("sim_wall_ns_replay") or (float(sim.get("wall_ms", 0.0)) * 1_000_000.0 if sim.get("wall_ms") else None)
        fast_wall_ns = fast_sim.get("sim_wall_ns_replay") or (float(fast_sim.get("wall_ms", 0.0)) * 1_000_000.0 if fast_sim.get("wall_ms") else None)
        rows.append(
            {
                "family": candidate.get("family"),
                "workload": name,
                "role": candidate.get("role"),
                "trace_status": "observed" if full_record.get("trace", {}).get("dynamic_trace") else "unavailable",
                "ncu_status": "observed" if full_record.get("ncu", {}).get("rows") else "unavailable",
                "ncu_metric_rows": len(ncu_errors),
                "ncu_median_absolute_relative_error": statistics.median(ncu_errors) if ncu_errors else None,
                "host_inclusive_over_full_replay": host_ns / full_wall_ns if host_ns and full_wall_ns else None,
                "host_inclusive_over_fast_frontier": host_ns / fast_wall_ns if host_ns and fast_wall_ns else None,
                "cuda_event_ms": event_ms,
                "primary_cuda_event_speedup": event_ms / (float(full_wall_ns) / 1_000_000.0) if event_ms and full_wall_ns else None,
                "fast_cuda_event_speedup": event_ms / (float(fast_wall_ns) / 1_000_000.0) if event_ms and fast_wall_ns else None,
                "primary_speedup_status": "available_source_level_cuda_event_marker" if event_ms else "unavailable_no_uninstrumented_cuda_event_marker",
                "coverage": "partial_async_visibility_unobserved",
                "abstention": "retain_diagnostic_only_for_unavailable_cuda_event_or_visibility",
            }
        )
    result = {
        "schema_version": 1,
        "model": "sm120_prospective_score_v1",
        "cohort_status": cohort.get("status"),
        "rows": rows,
        "aggregate": {
            "family_count": len({row["family"] for row in rows}),
            "workload_count": len(rows),
            "ncu_metric_rows": len(errors),
            "ncu_median_absolute_relative_error": statistics.median(errors) if errors else None,
            "ncu_max_absolute_relative_error": max(errors) if errors else None,
            "primary_cuda_event_speedup_status": "available_source_level_cuda_event_markers"
            if any(row.get("cuda_event_ms") is not None for row in rows)
            else "unavailable_for_all_prospective_rows",
        },
        "target_leakage": "none; NCU rows are read only after trace-only replay outputs are produced",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
