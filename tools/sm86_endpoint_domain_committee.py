#!/usr/bin/env python3
"""Combine endpoint predictors with a fixed target-domain policy.

The policy is deliberately mechanism-agnostic and score-label independent:
cycle-domain GPC predictions stay on the replay/frontier model, while elapsed
duration uses the same-regime CUDA-event-to-NCU crosswalk.  The latter is an
independent host/device observation and is not treated as proof that CUDA
events, NCU semaphore completion, CUPTI, and simulator clocks are identical.

Only prediction fields are consumed from the input reports.  Actual NCU values
are loaded once, after routing, solely to report the final live campaign error.
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


def _load(path: Path) -> dict[str, dict]:
    if path.is_dir():
        with (path / "performance.csv").open(newline="") as stream:
            return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}
    value = json.loads(path.read_text())
    rows = value.get("predictions") or value.get("rows") or []
    result = {}
    for row in rows:
        name = row.get("workload")
        if name:
            result[name] = row
    if not result:
        raise RuntimeError(f"empty prediction report: {path}")
    return result


def _prediction(row: dict, target: str) -> float:
    key = f"{target}_predicted"
    value = row.get(key)
    if value is None and isinstance(row.get("endpoints"), dict):
        endpoint = row["endpoints"].get(target, {})
        # Clock-elasticity reports retain several named hypotheses.  The
        # committee consumes only the explicitly requested elastic endpoint.
        if isinstance(endpoint, dict) and "elastic" in endpoint:
            value = endpoint["elastic"]
    if value is None and "predictions" in row:
        value = row["predictions"].get(target)
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"invalid {target} prediction for {row.get('workload')}")
    return value


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    baseline = _load(args.baseline)
    event = _load(args.event_ratio)
    score = _load(args.score_campaign)
    names = sorted(set(baseline) & set(event) & set(score))
    if not names:
        raise RuntimeError("no workload intersection among reports")
    results = []
    for name in names:
        gpc_route = (
            "clock_elasticity"
            if isinstance(baseline[name].get("endpoints"), dict)
            and isinstance(baseline[name]["endpoints"].get("ncu_gpc_cycles"), dict)
            and "elastic" in baseline[name]["endpoints"].get("ncu_gpc_cycles", {})
            else "replay_frontier"
        )
        item = {
            "workload": name,
            "status": "candidate",
            "coverage_status": "domain_committee_covered",
            "route_ncu_gpc_cycles": gpc_route,
            "route_ncu_duration_ns": "native_cuda_event_crosswalk",
            "baseline_source": str(args.baseline),
            "event_ratio_source": str(args.event_ratio),
        }
        for target, source in (("ncu_gpc_cycles", baseline[name]), ("ncu_duration_ns", event[name])):
            predicted = _prediction(source, target)
            actual = score[name].get(f"{target}_actual")
            if actual is None:
                actual = score[name].get(target)
            actual = float(actual)
            item[f"{target}_predicted"] = predicted
            item[f"{target}_actual"] = actual
            item[f"{target}_ape"] = abs(predicted - actual) / actual if actual > 0 else None
            # Preserve source interval if available; otherwise expose a
            # conservative point interval. The interval is not confidence.
            low = source.get(f"{target}_interval_low", predicted)
            high = source.get(f"{target}_interval_high", predicted)
            item[f"{target}_interval_low"] = float(low)
            item[f"{target}_interval_high"] = float(high)
        results.append(item)
    metrics = {}
    for target in TARGETS:
        apes = [row[f"{target}_ape"] for row in results if row[f"{target}_ape"] is not None]
        metrics[target] = {
            "count": len(apes),
            "mae": float(np.mean([abs(row[f"{target}_predicted"] - row[f"{target}_actual"]) for row in results])),
            "mape": float(np.mean(apes)) if apes else None,
            "max_ape": float(np.max(apes)) if apes else None,
        }
    report = {
        "schema": "sm86_endpoint_domain_committee_v1",
        "baseline": str(args.baseline),
        "event_ratio": str(args.event_ratio),
        "score_campaign": str(args.score_campaign),
        "targets_score_only": True,
        "policy": {
            "ncu_gpc_cycles": "baseline replay/frontier prediction",
            "ncu_duration_ns": "same-regime native CUDA-event endpoint ratio",
            "selection": "fixed endpoint-domain semantics; no score-row NCU route selection",
        },
        "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "projection_only_speedup_vs_replay_wall": None,
        "predictions": results,
        "limitations": [
            "CUDA event and NCU duration are crosswalked, not proven identical clocks",
            "GPC remains an aggregate NCU cycle target with no exact physical issue attribution",
            "cache ownership/read-from/complete DRAM semantics remain latent",
            "source reports may be related-family holdouts rather than independent generalization",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--event-ratio", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"metrics": report["metrics"], "rows": len(report["predictions"]), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
