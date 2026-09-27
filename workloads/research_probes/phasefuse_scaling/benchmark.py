#!/usr/bin/env python3
"""Run the bounded PhaseFuse descriptor-scaling matrix.

This is an analyzer/contract experiment, not a native GPU simulation. Each
case is a short child process so wait4 supplies independent user/system time
and peak RSS even on hosts without /usr/bin/time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "sim/target/release/phasefuse-scale"
SIZES = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000)
REGION_LENGTHS = (1, 2, 8, 64)
RESIDENCIES = ("complete", "unknown")
ORDERS = ("sorted", "shuffled")


def run_once(command: list[str], timeout: float) -> dict[str, object]:
    started = time.perf_counter_ns()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    status = None
    usage = None
    deadline = time.monotonic() + timeout
    while usage is None:
        waited, candidate_status, candidate_usage = os.wait4(process.pid, os.WNOHANG)
        if waited == process.pid:
            status, usage = candidate_status, candidate_usage
            break
        if time.monotonic() >= deadline:
            process.kill()
            _, status, usage = os.wait4(process.pid, 0)
            stdout = process.stdout.read() if process.stdout is not None else b""
            stderr = process.stderr.read() if process.stderr is not None else b""
            process.returncode = os.waitstatus_to_exitcode(status)
            return {
                "exit_code": os.waitstatus_to_exitcode(status),
                "timed_out": True,
                "wall_ns": time.perf_counter_ns() - started,
                "user_cpu_ns": int(usage.ru_utime * 1e9),
                "sys_cpu_ns": int(usage.ru_stime * 1e9),
                "peak_rss_kib": int(usage.ru_maxrss),
                "stderr": stderr.decode("utf-8", errors="replace")[-2000:],
            }
        time.sleep(0.001)
    stdout = process.stdout.read() if process.stdout is not None else b""
    stderr = process.stderr.read() if process.stderr is not None else b""
    process.returncode = os.waitstatus_to_exitcode(status)
    exit_code = os.waitstatus_to_exitcode(status)
    row: dict[str, object] = {
        "exit_code": exit_code,
        "timed_out": False,
        "wall_ns": time.perf_counter_ns() - started,
        "user_cpu_ns": int(usage.ru_utime * 1e9),
        "sys_cpu_ns": int(usage.ru_stime * 1e9),
        "peak_rss_kib": int(usage.ru_maxrss),
        "stderr": stderr.decode("utf-8", errors="replace")[-2000:],
    }
    if exit_code == 0:
        try:
            payload = json.loads(stdout)
            if not isinstance(payload, dict):
                raise ValueError("child JSON is not an object")
            row["payload"] = payload
        except (json.JSONDecodeError, ValueError) as error:
            row["parse_error"] = str(error)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phasefuse_scale.jsonl"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 10:
        parser.error("--repeats must be in 1..10")
    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    rows: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for n in SIZES:
            for region_len in REGION_LENGTHS:
                for residency in RESIDENCIES:
                    for order in ORDERS:
                        for repeat in range(args.repeats):
                            command = [
                                str(binary),
                                "--n",
                                str(n),
                                "--region-len",
                                str(region_len),
                                "--residency",
                                residency,
                                "--order",
                                order,
                                "--seed",
                                str(args.seed),
                            ]
                            result = run_once(command, args.timeout)
                            payload = result.pop("payload", None)
                            row = {
                                "case_key": f"n{n}-r{region_len}-{residency}-{order}-seed{args.seed}",
                                "n": n,
                                "region_len": region_len,
                                "residency": residency,
                                "order": order,
                                "repeat": repeat,
                                "timeout_s": args.timeout,
                                **result,
                            }
                            if isinstance(payload, dict):
                                row["contract"] = payload
                            rows.append(row)
                            stream.write(json.dumps(row, sort_keys=True) + "\n")
                            stream.flush()
    aggregates = []
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["case_key"]), []).append(row)
    for key, group in sorted(groups.items()):
        good = [row for row in group if row.get("exit_code") == 0 and not row.get("timed_out")]
        aggregates.append(
            {
                "case_key": key,
                "repeats": len(group),
                "successful_repeats": len(good),
                "wall_ns_median": statistics.median(row["wall_ns"] for row in good) if good else None,
                "peak_rss_kib_max": max((row["peak_rss_kib"] for row in good), default=None),
                "terminal_state_matches": all(
                    row.get("contract", {}).get("terminal_state_matches") is True for row in good
                ),
            }
        )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"schema": "phasefuse_scale_results_v1", "aggregates": aggregates}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
