#!/usr/bin/env python3
"""Compare cold/warm native CUPTI PC surfaces with post-hoc NCU ratios."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np


PHASE_FEATURES = (
    "cupti_no_issue_fraction", "cupti_pressure_fraction",
    "cupti_state_long_scoreboard", "cupti_state_imc_miss",
    "cupti_state_short_scoreboard", "cupti_state_wait",
    "cupti_state_drain", "cupti_state_no_instructions",
    "cupti_state_math_pipe_throttle", "cupti_state_mio_throttle",
)


def _perf(path):
    with (path / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


def _feature(summary):
    state = {key: 0.0 for key in ("long_scoreboard", "imc_miss", "short_scoreboard", "wait", "drain", "no_instructions", "math_pipe_throttle", "mio_throttle")}
    no_issue = {key: 0.0 for key in state}
    unknown = 0.0
    for item in summary.get("summaries", []):
        for key, value in item.get("scheduler_state_samples", {}).items():
            if key in state:
                state[key] += float(value)
            else:
                unknown += float(value)
        for key, value in item.get("scheduler_no_issue_samples", {}).items():
            if key in no_issue:
                no_issue[key] += float(value)
            else:
                unknown += float(value)
    total = sum(state.values()) + unknown
    return {
        "cupti_no_issue_fraction": sum(no_issue.values()) / max(total, 1.0),
        "cupti_pressure_fraction": sum(state[key] for key in state if key not in ("no_instructions",)) / max(total, 1.0),
        **{f"cupti_state_{key}": value / max(total, 1.0) for key, value in state.items()},
    }


def _load_cupti(root):
    report = json.loads((root / "summary.json").read_text())
    return {row["workload"]: _feature(row) for row in report.get("workloads", []) if row.get("status") == "completed"}


def run(args):
    started = time.perf_counter()
    cold = _load_cupti(args.cold_cupti)
    warm = _load_cupti(args.warm_cupti)
    cold_perf = _perf(args.cold_campaign)
    warm_perf = _perf(args.warm_campaign)
    names = sorted(set(cold) & set(warm) & set(cold_perf) & set(warm_perf))
    if not names:
        raise RuntimeError("no paired rows")
    rows = []
    for name in names:
        item = {"workload": name, "cupti": {}, "ncu": {}}
        for feature in PHASE_FEATURES:
            cold_value, warm_value = cold[name].get(feature, 0.0), warm[name].get(feature, 0.0)
            item["cupti"][feature] = {"cold": cold_value, "warm": warm_value, "abs_delta": abs(warm_value - cold_value)}
        for target in ("ncu_gpc_cycles", "ncu_duration_ns"):
            c, w = float(cold_perf[name][target]), float(warm_perf[name][target])
            item["ncu"][target] = {"cold": c, "warm": w, "warm_cold_ratio": w / c}
        rows.append(item)
    summary = {"count": len(rows), "feature_abs_delta_median": {}, "feature_abs_delta_p90": {}, "ratios": {}}
    for feature in PHASE_FEATURES:
        values = [row["cupti"][feature]["abs_delta"] for row in rows]
        summary["feature_abs_delta_median"][feature] = float(np.median(values))
        summary["feature_abs_delta_p90"][feature] = float(np.quantile(values, .90))
    for target in ("ncu_gpc_cycles", "ncu_duration_ns"):
        ratios = np.asarray([row["ncu"][target]["warm_cold_ratio"] for row in rows])
        summary["ratios"][target] = {"median": float(np.median(ratios)), "min": float(np.min(ratios)), "max": float(np.max(ratios))}
    correlations = {}
    for feature in PHASE_FEATURES:
        delta = np.asarray([row["cupti"][feature]["abs_delta"] for row in rows])
        correlations[feature] = {}
        for target in ("ncu_gpc_cycles", "ncu_duration_ns"):
            ratio = np.asarray([row["ncu"][target]["warm_cold_ratio"] for row in rows])
            correlations[feature][target] = float(np.corrcoef(delta, ratio)[0, 1]) if np.std(delta) > 0 and np.std(ratio) > 0 else None
    report = {
        "schema": "sm86_native_cupti_phase_analysis_v1",
        "cold_cupti": str(args.cold_cupti), "warm_cupti": str(args.warm_cupti),
        "cold_campaign": str(args.cold_campaign), "warm_campaign": str(args.warm_campaign),
        "target_labels_used": "post-hoc ratio comparison only; no fitting or routing",
        "summary": summary, "feature_ratio_correlations": correlations, "rows": rows,
        "model_seconds": time.perf_counter() - started,
        "interpretation": "CUPTI state deltas are an independent phase descriptor; they do not prove cache ownership/read-from or complete writeback",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-cupti", type=Path, required=True)
    parser.add_argument("--warm-cupti", type=Path, required=True)
    parser.add_argument("--cold-campaign", type=Path, required=True)
    parser.add_argument("--warm-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"summary": report["summary"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
