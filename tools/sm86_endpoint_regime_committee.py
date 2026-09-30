#!/usr/bin/env python3
"""Route each endpoint by fixed regime/domain semantics.

The source route is declared by phase before score rows are loaded: GPC uses
clock elasticity in cold high-clock conditions and native-event/frontier in
warm conditions; duration uses native-event/frontier cold and the same-regime
native-event ratio warm. No score NCU value selects a route.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")


def _load(path):
    if path.is_dir():
        with (path / "performance.csv").open(newline="") as stream:
            return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}
    value = json.loads(path.read_text())
    rows = value.get("predictions") or value.get("rows") or []
    result = {row["workload"]: row for row in rows if row.get("workload")}
    if not result:
        raise RuntimeError(f"empty report: {path}")
    return result


def _prediction(row, target):
    value = row.get(f"{target}_predicted")
    if value is None and isinstance(row.get("endpoints"), dict):
        endpoint = row["endpoints"].get(target, {})
        if isinstance(endpoint, dict):
            value = endpoint.get("elastic", endpoint.get("reference"))
    if value is None and isinstance(row.get("predictions"), dict):
        value = row["predictions"].get(target)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid source prediction for {row.get('workload')}")
    return value


def run(args):
    started = time.perf_counter()
    gpc = _load(args.gpc_source)
    duration = _load(args.duration_source)
    score = _load(args.score_campaign)
    names = sorted(set(gpc) & set(duration) & set(score))
    if not names:
        raise RuntimeError("no source/score intersection")
    rows = []
    for name in names:
        item = {
            "workload": name, "status": "candidate", "coverage_status": "regime_endpoint_covered",
            "phase": args.phase, "route_ncu_gpc_cycles": args.gpc_route,
            "route_ncu_duration_ns": args.duration_route,
            "gpc_source": str(args.gpc_source), "duration_source": str(args.duration_source),
        }
        for target, source in (("ncu_gpc_cycles", gpc[name]), ("ncu_duration_ns", duration[name])):
            predicted = _prediction(source, target)
            actual = float(score[name].get(f"{target}_actual", score[name].get(target)))
            item[f"{target}_predicted"] = predicted
            item[f"{target}_actual"] = actual
            item[f"{target}_ape"] = abs(predicted - actual) / actual
            low = source.get(f"{target}_interval_low", predicted)
            high = source.get(f"{target}_interval_high", predicted)
            item[f"{target}_interval_low"] = float(low)
            item[f"{target}_interval_high"] = float(high)
        rows.append(item)
    metrics = {}
    for target in TARGETS:
        apes = [row[f"{target}_ape"] for row in rows]
        errors = [abs(row[f"{target}_predicted"] - row[f"{target}_actual"]) for row in rows]
        metrics[target] = {"count": len(apes), "mae": float(np.mean(errors)), "mape": float(np.mean(apes)), "max_ape": float(np.max(apes))}
    report = {
        "schema": "sm86_endpoint_regime_committee_v1", "phase": args.phase,
        "gpc_source": str(args.gpc_source), "duration_source": str(args.duration_source),
        "score_campaign": str(args.score_campaign), "targets_score_only": True,
        "routes": {"ncu_gpc_cycles": args.gpc_route, "ncu_duration_ns": args.duration_route},
        "selection": "fixed phase/domain policy; no score NCU route selection",
        "metrics": metrics, "model_seconds": time.perf_counter() - started,
        "predictions": rows,
        "limitations": [
            "CUDA-event, NCU, and simulator endpoints remain distinct",
            "route sources are related-family holdouts",
            "cache ownership/read-from and physical issue identity remain latent",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--gpc-source", type=Path, required=True)
    parser.add_argument("--duration-source", type=Path, required=True)
    parser.add_argument("--gpc-route", required=True)
    parser.add_argument("--duration-route", required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"phase": report["phase"], "metrics": report["metrics"], "rows": len(report["predictions"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
