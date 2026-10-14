#!/usr/bin/env python3
"""Family-stratify post-replay cycle residuals and trace-side features."""

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


def summarize(values: list[float]) -> dict:
    values.sort()
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "p90": values[int(0.90 * (len(values) - 1))] if values else None,
    }


def report(path: Path, profiles: Path) -> dict:
    groups: dict[str, dict[str, list[float] | int]] = defaultdict(
        lambda: {
            "gpc_ape": [],
            "sm_ape": [],
            "gpc_abs": [],
            "sm_abs": [],
            "launches": 0,
            "tma_load_bytes": 0,
            "tma_store_bytes": 0,
            "tma_launches": 0,
            "async_queries": 0,
        }
    )
    totals = {"captures": 0, "launches": 0}
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
                model_gpc = completion.get("lower_bound_cycles", completion.get("cycles"))
                if model_gpc is None:
                    model_gpc = timing.get("terminal_modeled_event_time")
                sm_frontier = timing.get("sm_active_terminal_frontier") or {}
                model_sm = sm_frontier.get("cycles")
                def target_value(key):
                    value = target.get(key)
                    return value.get("value") if isinstance(value, dict) else None
                gpc = target_value("gpc__cycles_elapsed.max")
                sm = target_value("sm__cycles_active.max")
                group = groups[family(name)]
                group["launches"] += 1
                metrics = launch.get("trace_metrics") or {}
                group["tma_load_bytes"] += int(metrics.get("tma_sidecar_load_payload_bytes", 0))
                group["tma_store_bytes"] += int(metrics.get("tma_sidecar_store_payload_bytes", 0))
                if metrics.get("tma_sidecar_load_records", 0) or metrics.get("tma_sidecar_store_records", 0):
                    group["tma_launches"] += 1
                group["async_queries"] += int(metrics.get("sync_sidecar_phase_check_records", 0))
                if finite(model_gpc) and finite(gpc) and gpc > 0:
                    group["gpc_abs"].append(abs(float(model_gpc) - float(gpc)))
                    group["gpc_ape"].append(abs(float(model_gpc) - float(gpc)) / float(gpc))
                if finite(model_sm) and finite(sm) and sm > 0:
                    group["sm_abs"].append(abs(float(model_sm) - float(sm)))
                    group["sm_ape"].append(abs(float(model_sm) - float(sm)) / float(sm))
    rendered = {}
    for name, group in sorted(groups.items()):
        rendered[name] = {
            "launches": group["launches"],
            "tma_load_bytes": group["tma_load_bytes"],
            "tma_store_bytes": group["tma_store_bytes"],
            "tma_launches": group["tma_launches"],
            "async_queries": group["async_queries"],
            "gpc_absolute_error": summarize(group["gpc_abs"]),
            "gpc_ape": summarize(group["gpc_ape"]),
            "sm_absolute_error": summarize(group["sm_abs"]),
            "sm_ape": summarize(group["sm_ape"]),
        }
    return {
        "scope": "post-replay cycle residuals with trace-side feature totals; NCU never replay input",
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
    result = report(args.full, args.profiles)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SM120 cycle residuals by trace family",
        "",
        "Post-replay comparison; NCU values were not replay inputs.",
        "",
        "| family | launches | TMA load bytes | TMA store bytes | async queries | GPC APE median/P90 | SM APE median/P90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["families"].items():
        g, s = value["gpc_ape"], value["sm_ape"]
        lines.append(
            f"| {name} | {value['launches']} | {value['tma_load_bytes']} | {value['tma_store_bytes']} | {value['async_queries']} | "
            f"{g['median']} / {g['p90']} | {s['median']} / {s['p90']} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
