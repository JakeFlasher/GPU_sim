#!/usr/bin/env python3
"""Run the installed CUPTI PC-sampling collector for one SM120 workload.

The CUDA toolkit's continuous collector is the primary mechanism.  Driver or
SKU refusal is emitted as a typed observation limitation and does not fabricate
PC rows or redirect a process-level sample to launch zero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_RUNNER = Path("/usr/local/cuda-13.2/extras/CUPTI/samples/pc_sampling_continuous/libpc_sampling_continuous.pl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
            "/usr/local/cuda-13.2/extras/CUPTI/lib64",
            str(args.runner.parent),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    args.data.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.runner),
        "--collection-mode",
        "2",
        "--sampling-period",
        "7",
        "--file-name",
        args.data.name,
        "--app",
        args.app,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=args.data.parent,
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
    data_present = args.data.is_file() and args.data.stat().st_size > 0
    status = "observed_file" if data_present else "unavailable_tool_or_sku"
    result = {
        "schema_version": 1,
        "model": "sm120_cupti_pc_sampling_collection_v1",
        "status": status,
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "command": command,
        "data_path": str(args.data),
        "data_present": data_present,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "fallback": "aggregate NCU observations only; no PC identity inferred" if not data_present else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "returncode": completed.returncode, "data_present": data_present}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
