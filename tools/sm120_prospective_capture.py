#!/usr/bin/env python3
"""Capture the prospective cohort with the existing SM120 NVBit producer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/root/GPU_simulator_rebase"))
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracer = args.repo / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"
    pb_lib = args.repo / ".toolchain/pb/lib"
    rows = []
    for candidate in manifest.get("candidates", []):
        destination = args.work_root / "prospective" / "captures" / str(candidate["workload"])
        destination.mkdir(parents=True, exist_ok=True)
        binary = Path(candidate["binary_path"])
        command = [str(binary), *candidate.get("command", [])]
        env = dict(os.environ)
        env.update(
            {
                "CUDA_INJECTION64_PATH": str(tracer),
                "USER_DEFINED_FOLDERS": "1",
                "TRACES_FOLDER": str(destination),
                "ACTIVE_FROM_START": "1",
                "NVBIT_ALPHA_NO_HASH": "1",
                "SM120_MEMORY_TENSOR_ONLY": "1",
                "SM120_CAPTURE_OPERANDS": "0",
                "SM120_ALLOW_UNREGISTERED_GLOBALS": "0",
                "SKIP_ENHANCED_TRACER": "1",
            }
        )
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(pb_lib), "/usr/local/cuda-13.2/targets/x86_64-linux/lib", env.get("LD_LIBRARY_PATH", "")]
        )
        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(command, cwd=args.repo, env=env, capture_output=True, text=True, timeout=args.timeout, check=False)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "")
            timed_out = True
        rows.append(
            {
                "family": candidate.get("family"),
                "workload": candidate.get("workload"),
                "command": command,
                "capture_dir": str(destination),
                "returncode": completed.returncode,
                "timed_out": timed_out,
                "elapsed_ns": time.perf_counter_ns() - started,
                "dynamic_trace_present": (destination / "dynamic_trace.pb").is_file(),
                "stdout_tail": (completed.stdout or "")[-8000:],
                "stderr_tail": (completed.stderr or "")[-8000:],
            }
        )
    result = {"schema_version": 1, "model": "sm120_prospective_nvbit_capture_v1", "rows": rows, "capture_count": len(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"capture_count": len(rows), "trace_count": sum(row["dynamic_trace_present"] for row in rows), "timed_out": sum(row["timed_out"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
