#!/usr/bin/env python3
"""Collect a compact independent NCU profile for the prospective cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


METRICS = ",".join(
    [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.avg",
        "smsp__inst_executed.sum",
        "lts__t_sectors_lookup_hit.sum",
        "lts__t_sectors_lookup_miss.sum",
        "dram__sectors_op_read.sum",
        "dram__sectors_op_write.sum",
    ]
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda-13.2/bin/ncu"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--repo", type=Path, default=Path("/root/GPU_simulator_rebase"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in manifest.get("candidates", []):
        workload = str(candidate["workload"])
        log_file = args.out_root / f"{workload}.raw.csv"
        command = [
            str(args.ncu),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
            "--target-processes",
            "all",
            "--metrics",
            METRICS,
            "--log-file",
            str(log_file),
            str(candidate["binary_path"]),
            *candidate.get("command", []),
        ]
        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(command, cwd=args.repo, capture_output=True, text=True, timeout=args.timeout, check=False)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "")
            timed_out = True
        rows.append(
            {
                "family": candidate.get("family"),
                "workload": workload,
                "command": command,
                "returncode": completed.returncode,
                "timed_out": timed_out,
                "elapsed_ns": time.perf_counter_ns() - started,
                "raw_csv": str(log_file),
                "raw_csv_present": log_file.is_file() and log_file.stat().st_size > 0,
                "stdout_tail": (completed.stdout or "")[-4000:],
                "stderr_tail": (completed.stderr or "")[-4000:],
            }
        )
    result = {"schema_version": 1, "model": "sm120_prospective_ncu_collection_v1", "metrics": METRICS.split(","), "rows": rows, "target_used_as_prediction_input": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "profiles_present": sum(row["raw_csv_present"] for row in rows), "timed_out": sum(row["timed_out"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
