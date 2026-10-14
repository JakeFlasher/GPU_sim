#!/usr/bin/env python3
"""Evaluate the fixed SM120 GPC front-end envelope post-replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from jsonl_io import open_jsonl_reader
from sm120_replay_bench import join_ncu_rows, parse_ncu_raw


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def family(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def dist(values: list[float]) -> dict:
    values.sort()
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "p90": values[int(0.90 * (len(values) - 1))] if values else None,
    }


def summarize(path: Path, profiles: Path) -> dict:
    families = defaultdict(lambda: {"raw_gpc": [], "effective_gpc": [], "raw_ipc": [], "effective_ipc": []})
    totals = {"captures": 0, "launches": 0, "better_gpc": 0, "worse_gpc": 0}
    with open_jsonl_reader(path) as stream:
        for line in stream:
            capture = json.loads(line)
            name = str(capture.get("canonical_capture", ""))
            ncu_rows = parse_ncu_raw(profiles / name / "raw.csv")
            launches = capture.get("launches", [])
            totals["captures"] += 1
            totals["launches"] += len(launches)
            for join in join_ncu_rows(launches, ncu_rows):
                li, ni = join.get("launch_index"), join.get("ncu_row_index")
                if li is None or ni is None:
                    continue
                launch = launches[li]
                target = ncu_rows[ni].get("metrics", {})
                timing = launch.get("timing") or {}
                completion = timing.get("completion_frontier") or {}
                raw = completion.get("lower_bound_cycles", completion.get("cycles"))
                if not finite(raw):
                    continue
                counters = launch.get("counters") or {}
                lines = int((counters.get("front_end") or {}).get("compulsory_instruction_read_sectors", 0))
                envelope = 3900 + (3 * lines + 1) // 2
                effective = float(raw) + envelope
                target_value = target.get("gpc__cycles_elapsed.max")
                target_value = target_value.get("value") if isinstance(target_value, dict) else None
                if not finite(target_value) or target_value <= 0:
                    continue
                raw_error = abs(float(raw) - target_value) / target_value
                effective_error = abs(effective - target_value) / target_value
                f = families[family(name)]
                f["raw_gpc"].append(raw_error)
                f["effective_gpc"].append(effective_error)
                totals["better_gpc"] += int(effective_error < raw_error)
                totals["worse_gpc"] += int(effective_error > raw_error)
                # The elapsed-IPC projection uses the same adjusted GPC
                # denominator and the carded 170-SM population.
                inst = ((launch.get("trace_metrics") or {}).get("instructions") or {}).get("warp_instructions")
                ipc_target = target.get("sm__inst_executed.avg.per_cycle_elapsed")
                ipc_target = ipc_target.get("value") if isinstance(ipc_target, dict) else None
                if finite(inst) and finite(ipc_target) and ipc_target > 0:
                    raw_ipc = float(inst) / (float(raw) * 170.0) if raw else None
                    effective_ipc = float(inst) / (effective * 170.0)
                    if raw_ipc is not None:
                        f["raw_ipc"].append(abs(raw_ipc - ipc_target) / ipc_target)
                    f["effective_ipc"].append(abs(effective_ipc - ipc_target) / ipc_target)
    rendered = {}
    for name, f in sorted(families.items()):
        rendered[name] = {key: dist(value) for key, value in f.items()}
    return {
        "scope": "post-replay fixed alpha front-end envelope; NCU not replay input",
        "envelope": "3900 + ceil(1.5 * compulsory instruction sectors)",
        "source": str(path),
        "profiles": str(profiles),
        "totals": totals,
        "families": rendered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.full, args.profiles)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SM120 effective GPC front-end envelope",
        "",
        "Post-replay projection using a fixed source-derived alpha envelope; NCU was not a replay input.",
        "",
        "| family | raw GPC APE median/P90 | effective GPC APE median/P90 | raw IPC-elapsed APE median/P90 | effective IPC-elapsed APE median/P90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in result["families"].items():
        lines.append(
            f"| {name} | {value['raw_gpc']['median']} / {value['raw_gpc']['p90']} | "
            f"{value['effective_gpc']['median']} / {value['effective_gpc']['p90']} | "
            f"{value['raw_ipc']['median']} / {value['raw_ipc']['p90']} | "
            f"{value['effective_ipc']['median']} / {value['effective_ipc']['p90']} |"
        )
    lines.extend(
        [
            "",
            f"Launches improved: {result['totals']['better_gpc']}; worsened: {result['totals']['worse_gpc']}.",
            "The envelope is an effective observation projection, not a physical instruction-cache or clock claim.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
