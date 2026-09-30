#!/usr/bin/env python3
"""Compare launch-local CUPTI state surfaces after removing warmup buffers."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


FEATURES = ("no_issue_fraction", "pressure_fraction", "long_scoreboard_fraction", "imc_miss_fraction", "short_scoreboard_fraction", "wait_fraction")
REASONS = ("long_scoreboard", "imc_miss", "short_scoreboard", "wait", "branch_resolving", "drain", "math_pipe_throttle", "mio_throttle", "dispatch_stall", "no_instructions")


def _perf(path):
    with (path / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


def _summary(root, workload):
    paths = sorted((root / workload).glob("decode_*.summary.json"))
    if not paths:
        raise RuntimeError(f"missing CUPTI summary for {workload}")
    # One decoded file is expected; aggregate if a caller supplied several.
    buffers = []
    for path in paths:
        buffers.extend(json.loads(path.read_text()).get("buffers", []))
    return sorted(buffers, key=lambda item: int(item.get("buffer", 0)))


def _buffer_feature(item):
    states = item.get("scheduler_state_samples", {})
    no_issue = item.get("scheduler_no_issue_samples", {})
    total = sum(float(value) for value in states.values())
    no_total = sum(float(value) for value in no_issue.values())
    pressure = sum(float(states.get(key, 0.0)) for key in REASONS)
    return {
        "no_issue_fraction": no_total / max(total, 1.0),
        "pressure_fraction": pressure / max(total, 1.0),
        "long_scoreboard_fraction": float(states.get("long_scoreboard", 0.0)) / max(total, 1.0),
        "imc_miss_fraction": float(states.get("imc_miss", 0.0)) / max(total, 1.0),
        "short_scoreboard_fraction": float(states.get("short_scoreboard", 0.0)) / max(total, 1.0),
        "wait_fraction": float(states.get("wait", 0.0)) / max(total, 1.0),
        "samples": float(item.get("samples", 0.0)),
        "records": float(item.get("pc_records", 0.0)),
        "correlation_ids": item.get("correlation_ids", []),
    }


def _aggregate(buffers):
    values = [_buffer_feature(item) for item in buffers]
    weights = [max(item["samples"], 1.0) for item in values]
    result = {feature: float(np.average([item[feature] for item in values], weights=weights)) for feature in FEATURES}
    result.update({"samples": sum(item["samples"] for item in values), "records": sum(item["records"] for item in values), "buffers": len(values)})
    return result


def run(args):
    started = time.perf_counter()
    cold_perf, warm_perf = _perf(args.cold_campaign), _perf(args.warm_campaign)
    names = sorted(set(cold_perf) & set(warm_perf))
    rows = []
    for name in names:
        cold_buffers = _summary(args.cold_cupti, name)
        warm_buffers = _summary(args.warm_cupti, name)
        if args.warmup_buffers < 0 or len(warm_buffers) <= args.warmup_buffers:
            raise RuntimeError(f"warmup split leaves no measured buffers for {name}")
        measured_warm = warm_buffers[args.warmup_buffers:]
        cold = _aggregate(cold_buffers)
        warm = _aggregate(measured_warm)
        item = {"workload": name, "cold": cold, "warm_measured": warm, "warmup_buffers_discarded": args.warmup_buffers, "ncu": {}}
        for feature in FEATURES:
            item["ncu"][feature] = {"cold": cold[feature], "warm": warm[feature], "abs_delta": abs(warm[feature] - cold[feature])}
        for target in ("ncu_gpc_cycles", "ncu_duration_ns"):
            c, w = float(cold_perf[name][target]), float(warm_perf[name][target])
            item["ncu"][target] = {"cold": c, "warm": w, "warm_cold_ratio": w / c}
        rows.append(item)
    summary = {"count": len(rows), "warmup_buffers_discarded": args.warmup_buffers, "feature_abs_delta_median": {}, "feature_abs_delta_p90": {}, "ncu_ratios": {}}
    for feature in FEATURES:
        values = [row["ncu"][feature]["abs_delta"] for row in rows]
        summary["feature_abs_delta_median"][feature] = float(np.median(values)); summary["feature_abs_delta_p90"][feature] = float(np.quantile(values, .90))
    for target in ("ncu_gpc_cycles", "ncu_duration_ns"):
        values = [row["ncu"][target]["warm_cold_ratio"] for row in rows]
        summary["ncu_ratios"][target] = {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
    correlations = {}
    for feature in FEATURES:
        delta = np.asarray([row["ncu"][feature]["abs_delta"] for row in rows])
        correlations[feature] = {}
        for target in ("ncu_gpc_cycles", "ncu_duration_ns"):
            ratio = np.asarray([row["ncu"][target]["warm_cold_ratio"] for row in rows])
            correlations[feature][target] = float(np.corrcoef(delta, ratio)[0, 1]) if np.std(delta) > 0 and np.std(ratio) > 0 else None
    report = {"schema": "sm86_native_cupti_launch_phase_analysis_v1", "cold_cupti": str(args.cold_cupti), "warm_cupti": str(args.warm_cupti), "cold_campaign": str(args.cold_campaign), "warm_campaign": str(args.warm_campaign), "summary": summary, "feature_ratio_correlations": correlations, "rows": rows, "model_seconds": time.perf_counter()-started, "interpretation": "warmup buffers are removed by correlation/range position; remaining CUPTI states remain aggregate sampled observations"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2)+"\n"); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--cold-cupti',type=Path,required=True); parser.add_argument('--warm-cupti',type=Path,required=True); parser.add_argument('--cold-campaign',type=Path,required=True); parser.add_argument('--warm-campaign',type=Path,required=True); parser.add_argument('--warmup-buffers',type=int,default=8); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); report=run(args); print(json.dumps({'summary':report['summary'],'output':str(args.output)},indent=2))


if __name__ == '__main__': main()
