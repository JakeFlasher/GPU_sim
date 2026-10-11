#!/usr/bin/env python3
"""Compute explicit trace-reuse/amortized fast-frontier accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--repeated-timing", type=Path, required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fast = {json.loads(line)["capture_name"]: json.loads(line) for line in args.fast.read_text(encoding="utf-8").splitlines() if line.strip()}
    timing = json.loads(args.repeated_timing.read_text(encoding="utf-8"))
    rows = []
    for item in timing.get("rows", []):
        name = item.get("workload")
        record = fast.get(name, {})
        sim = record.get("simulator", {})
        event_rows = item.get("sm120_event_markers", []) or []
        event_ms = float(event_rows[0]["elapsed_ms"]) / max(1, args.reps) if event_rows else None
        process_ms = float(sim.get("wall_ms", 0.0))
        rows.append(
            {
                "workload": name,
                "reps": args.reps,
                "cuda_event_ms_per_iteration": event_ms,
                "fast_frontier_one_input_ms": process_ms,
                "amortized_fast_frontier_ms_per_iteration": process_ms / max(1, args.reps),
                "amortized_cuda_event_speedup": event_ms / (process_ms / max(1, args.reps)) if event_ms and process_ms else None,
                "scope": "trace/index reuse accounting; one model input reused across repeated identical launches",
                "not_a_claim": "does not model repeated GPU state evolution or cache warm-up",
            }
        )
    result = {
        "schema_version": 1,
        "model": "sm120_fast_frontier_amortized_reuse_v1",
        "reps": args.reps,
        "rows": rows,
        "reuse_assumption": "immutable trace/static index reused for identical launches",
        "gpu_state_evolution": "unmodeled",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "reps": args.reps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
