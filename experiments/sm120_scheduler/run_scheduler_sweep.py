#!/usr/bin/env python3
"""Run the SM120 scheduler intervention matrix and retain raw observations.

This harness has no correctness oracle.  It records command, return status,
CUDA-event output and the `%clock64`/placement observations emitted by the
timing-only probe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def add(cases: list[dict], name: str, mode: str, chains: int, unroll: int, rounds: int, warps: int, mask: int, blocks: int, reps: int = 3, warmups: int = 2) -> None:
    cases.append({"name": name, "argv": [mode, str(chains), str(unroll), str(rounds), str(warps), str(mask), str(blocks), str(reps), str(warmups)]})


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for mode in ("fma", "mad", "add", "mix", "warp_mix"):
        for chains in (1, 2, 4, 8, 16):
            rounds = 8192 // chains
            add(cases, f"fixedwork_{mode}_k{chains}", mode, chains, 16, rounds, 1, 1, 1)
    for warps in (1, 2, 4, 8, 16, 32):
        mask = 0xFFFFFFFF if warps == 32 else (1 << warps) - 1
        add(cases, f"warp_density_w{warps}", "fma", 4, 16, 2048, warps, mask, 1)
    for mask, label in ((1, "w0"), (2, "w1"), (4, "w2"), (8, "w3"), (0x55, "even"), (0xAA, "odd")):
        add(cases, f"placement_{label}_allocated8", "fma", 4, 16, 2048, 8, mask, 1)
    for blocks in (1, 2, 4, 8, 16, 24):
        add(cases, f"block_density_b{blocks}", "fma", 4, 16, 2048, 4, 0xF, blocks)
    for warps in (1, 2, 4, 8, 16, 32):
        mask = 0xFFFFFFFF if warps == 32 else (1 << warps) - 1
        add(cases, f"barrier_w{warps}", "barrier", 1, 16, 128, warps, mask, 1)
    for unroll, rounds in ((64, 64), (256, 16), (1024, 4)):
        add(cases, f"footprint_u{unroll}", "fma", 4, unroll, rounds, 1, 1, 1)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in build_cases():
        started = time.perf_counter_ns()
        try:
            p = subprocess.run([str(args.binary), *case["argv"]], capture_output=True, text=True, timeout=args.timeout, check=False)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            p = subprocess.CompletedProcess([str(args.binary), *case["argv"]], 124, exc.stdout or "", exc.stderr or "")
            timed_out = True
        observations = []
        for line in (p.stdout or "").splitlines():
            if line.startswith("{"):
                try:
                    observations.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        record = {
            "name": case["name"],
            "argv": case["argv"],
            "returncode": p.returncode,
            "timed_out": timed_out,
            "elapsed_ns": time.perf_counter_ns() - started,
            "observations": observations,
            "stdout_tail": (p.stdout or "")[-2000:],
            "stderr_tail": (p.stderr or "")[-2000:],
        }
        (args.out / f"{case['name']}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(record)
        print(case["name"], p.returncode, len(observations), flush=True)
    result = {"schema_version": 1, "model": "sm120_scheduler_intervention_sweep_v1", "case_count": len(rows), "rows": rows}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "observed_cases": sum(bool(r["observations"]) for r in rows), "returncodes": sorted({r["returncode"] for r in rows})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
