#!/usr/bin/env python3
"""Measure the real MRC/service executable paths on the local corpus.

This records wall time, maximum resident set size and output size for the
trace-driven tools.  It is a performance microbenchmark, not a test or
verification gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def run_timed(name: str, command: list[str], output: Path, stats_path: Path) -> dict[str, Any]:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        ["/usr/bin/time", "-f", "%e %M", "-o", str(stats_path), *command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )
    wall = time.perf_counter() - started
    elapsed_s = None
    max_rss_kb = None
    try:
        fields = stats_path.read_text(encoding="utf-8").strip().split()
        if len(fields) >= 2:
            elapsed_s = float(fields[0])
            max_rss_kb = int(fields[1])
    except (OSError, ValueError):
        pass
    record = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds_process": wall,
        "elapsed_seconds_time": elapsed_s,
        "max_rss_kb": max_rss_kb,
        "output": str(output),
        "output_bytes": output.stat().st_size if output.is_file() else None,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
    return record


def modeled_references(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        int(launch.get("sampled_sector_references", 0) or 0)
        for capture in payload.get("captures", [])
        for launch in capture.get("launches", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path("/root/GPU_simulator_rebase")
    python = str(repo / "tools/sol_execbench/.venv/bin/python")
    mrc_tool = str(repo / "tools/sm120_mrc_model.py")
    service_tool = str(repo / "tools/sm120_memory_service_model.py")
    audit = args.work_root / "p8_audit"
    audit.mkdir(parents=True, exist_ok=True)
    retained_mrc = audit / "perf_retained_mrc.json"
    prospective_mrc = audit / "perf_prospective_mrc.json"
    retained_service = audit / "perf_retained_service.json"
    prospective_service = audit / "perf_prospective_service.json"
    commands = [
        (
            "retained_mrc_all",
            [python, mrc_tool, "--work-root", str(args.work_root), "--route", "all", "--max-references", "250000", "--out", str(retained_mrc)],
            retained_mrc,
        ),
        (
            "prospective_mrc_all",
            [python, mrc_tool, "--work-root", str(args.work_root / "p4_run_clean"), "--route", "all", "--max-references", "250000", "--out", str(prospective_mrc)],
            prospective_mrc,
        ),
        (
            "retained_service_with_mrc",
            [python, service_tool, "--features", str(args.work_root / "f3_trace_features_all_v2.json"), "--mrc", str(retained_mrc), "--mrc-scenario", "instrumented_receive_order", "--replay", str(args.work_root / "f1_scope/all_results.jsonl"), "--out", str(retained_service)],
            retained_service,
        ),
        (
            "prospective_service_with_mrc",
            [python, service_tool, "--features", str(args.work_root / "prospective/p1/features/trace_features.json"), "--mrc", str(prospective_mrc), "--mrc-scenario", "instrumented_receive_order", "--replay", str(args.work_root / "p4_run_clean/p1/full.jsonl"), "--out", str(prospective_service)],
            prospective_service,
        ),
    ]
    records = []
    for index, (name, command, output) in enumerate(commands):
        record = run_timed(name, command, output, audit / f"perf_{index}.time")
        if name.endswith("mrc_all") and output.is_file():
            refs = modeled_references(output)
            record["modeled_references"] = refs
            elapsed = record.get("elapsed_seconds_time") or record.get("wall_seconds_process")
            record["modeled_refs_per_second"] = refs / elapsed if elapsed else None
        records.append(record)
    result = {
        "schema_version": 1,
        "model": "sm120_mrc_service_perf_benchmark_v1",
        "records": records,
        "interpretation": "real local process wall time and maximum RSS; no hardware-cache or prediction-accuracy claim",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "returncodes": [r["returncode"] for r in records]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
