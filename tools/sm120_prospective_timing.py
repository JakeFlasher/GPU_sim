#!/usr/bin/env python3
"""Run prospective SM120 candidates without NVBit/NCU and record timing."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


RUNTIME_RE = re.compile(r"Avg\s+runtime\s*:?[ \t]*([0-9.eE+-]+)\s*ms", re.I)
EVENT_RE = re.compile(r"SM120_CUDA_EVENT\s+label=([^\s]+)\s+elapsed_ms=([0-9.eE+-]+)", re.I)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/root/GPU_simulator_rebase"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--event-reps", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.event_reps > 1:
        env["SM120_EVENT_REPS"] = str(args.event_reps)
    rows = []
    for candidate in manifest.get("candidates", []):
        binary = Path(candidate["binary_path"])
        command = [str(binary), *candidate.get("command", [])]
        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                command,
                cwd=args.repo,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "")
            timed_out = True
        elapsed_ns = time.perf_counter_ns() - started
        stdout = completed.stdout or ""
        runtimes = [float(match.group(1)) for match in RUNTIME_RE.finditer(stdout)]
        event_rows = [{"label": match.group(1), "elapsed_ms": float(match.group(2))} for match in EVENT_RE.finditer(stdout)]
        rows.append(
            {
                "family": candidate.get("family"),
                "workload": candidate.get("workload"),
                "command": command,
                "returncode": completed.returncode,
                "timed_out": timed_out,
                "binary_present": binary.is_file(),
                "host_elapsed_ns": elapsed_ns,
                "cuda_event_runtime_ms": runtimes,
                "cuda_event_runtime_status": "observed_program_marker"
                if runtimes or event_rows
                else "unavailable_program_emits_no_cuda_event_marker",
                "sm120_event_markers": event_rows,
                "stdout": stdout[-12000:],
                "stderr": (completed.stderr or "")[-12000:],
                "scope": "one_uninstrumented_process_invocation",
            }
        )
    result = {
        "schema_version": 1,
        "model": "sm120_prospective_uninstrumented_timing_v1",
        "status": "observed",
        "manifest": str(args.manifest),
        "rows": rows,
        "prediction_inputs": "none",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "completed": sum(row["returncode"] == 0 for row in rows), "timed_out": sum(row["timed_out"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
