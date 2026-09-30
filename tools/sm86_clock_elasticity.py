#!/usr/bin/env python3
"""Two-component clock response, fitted only on paired calibration workloads.

Separate fit/predict/score commands avoid reading target labels in inference.
Readiness exposures are simulator features, NOT silicon wall-time fractions.
The model is a constrained response hypothesis; no endpoint identity follows.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")
RIDGE = 0.01  # fixed before the prospective campaign, not score-selected


def positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name}: expected a finite positive number, got {value!r}")
    return result


def read_rows(campaign, cohort=None):
    with (campaign / "performance.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {r["workload"]: r for r in rows if r["status"] == "completed"
            and (cohort is None or r["workload"].split("__", 1)[0] == cohort)}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def exposure(campaign, workload):
    path = campaign / workload / "replay_0.stdout"
    if not path.is_file():
        return [0.5, 0.5], "missing_replay_exposures"
    text = path.read_text()
    value = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
    launches = value.get("launches", [])
    if len(launches) != 1:
        return [0.5, 0.5], "not_one_replay_launch"
    counts = launches[0].get("timing", {}).get("readiness", {}).get("constraint_exposure_cycles", {})
    if not counts or sum(counts.values()) <= 0:
        return [0.5, 0.5], "missing_replay_exposures"
    if any(not math.isfinite(float(v)) or v < 0 for v in counts.values()):
        raise ValueError(f"invalid exposure population in {path}")
    # Global/async waits are only a covariate: L1/L2 hits can scale with SM
    # frequency, so the fitted memory coefficient is not forced to one.
    memory = sum(float(counts.get(k, 0)) for k in
                 ("dependence_counter_l1tex", "async_source"))
    total = float(sum(counts.values()))
    fraction = memory / total
    return [fraction, 1.0 - fraction], "replay_exposure_mixture"


def box_fit(x, y):
    """Solve two-variable ridge LS on [0,1]^2 by enumerating active faces."""
    prior = np.array([1.0, 0.0])
    best = None
    for state in itertools.product((-1, 0, 1), repeat=2):
        free = [i for i, s in enumerate(state) if s == -1]
        fixed = [i for i, s in enumerate(state) if s != -1]
        beta = np.array([0.0 if s == -1 else float(s) for s in state])
        if free:
            a = x[:, free]
            rhs = y - x[:, fixed] @ beta[fixed] if fixed else y
            beta[free] = np.linalg.solve(a.T @ a + RIDGE * np.eye(len(free)),
                                         a.T @ rhs + RIDGE * prior[free])
        if np.any(beta < 0) or np.any(beta > 1):
            continue
        loss = float(np.sum((x @ beta - y) ** 2) + RIDGE * np.sum((beta - prior) ** 2))
        if best is None or loss < best[0]:
            best = (loss, beta)
    return best[1]


def response(target, ratio, share):
    if target == "ncu_duration_ns":
        return ratio + (1.0 - ratio) * share
    return 1.0 + (1.0 / ratio - 1.0) * share


def fit(args):
    start = time.perf_counter()
    low = read_rows(args.low_campaign, args.calibration_cohort)
    high = read_rows(args.high_campaign, args.calibration_cohort)
    if not low or set(low) != set(high):
        raise ValueError("calibration needs completed paired workloads in both campaigns")
    workloads = sorted(low)
    features, sources = zip(*(exposure(args.low_campaign, w) for w in workloads))
    x = np.array(features)
    f0 = positive(args.anchor_clock_mhz, "anchor clock")
    f1 = positive(args.calibration_clock_mhz, "calibration clock")
    ratio = f0 / f1
    if abs(ratio - 1) < 1e-6:
        raise ValueError("clock response is unidentifiable at identical frequencies")
    targets = {}
    for target in TARGETS:
        observed = np.array([positive(high[w][target], target) / positive(low[w][target], target)
                             for w in workloads])
        offset = ratio if target == "ncu_duration_ns" else 1.0
        slope = 1.0 - ratio if target == "ncu_duration_ns" else 1.0 / ratio - 1.0
        beta = box_fit(slope * x, observed - offset)
        predicted = offset + slope * (x @ beta)
        residual = np.abs(predicted - observed)
        physical = sorted((response(target, ratio, 0), response(target, ratio, 1)))
        targets[target] = {"coefficients": beta.tolist(),
                           "calibration_ratio_residual_q90": float(np.quantile(residual, .9)),
                           "outside_response_cone_count": int(np.sum((observed < physical[0]) | (observed > physical[1]))),
                           "observed_ratios": observed.tolist(),
                           "fitted_ratios": predicted.tolist()}
    out = {"schema": "sm86_clock_elasticity_v1", "created_unix_ns": time.time_ns(),
           "calibration_cohort": args.calibration_cohort, "workloads": workloads,
           "low_campaign": str(args.low_campaign), "high_campaign": str(args.high_campaign),
           "anchor_clock_mhz": f0, "calibration_clock_mhz": f1, "ridge": RIDGE,
           "features": ["memory_wait_exposure_fraction", "other_exposure_fraction"],
           "feature_values": features, "feature_sources": sources, "targets": targets,
           "fit_seconds": time.perf_counter() - start,
           "limitations": ["campaign-level frequencies are not launch-synchronized",
                           "memory frequency assumed unchanged; fixed waits not physically identified",
                           "cold and cache-control-none require separate calibrations",
                           "GPC and duration coefficients fit separately; no common-clock assertion"]}
    write(args.output, out)
    return out


def predict(args):
    start = time.perf_counter()
    model = json.loads(args.model.read_text())
    reference = json.loads(args.reference.read_text())
    # Only *_predicted and interval fields are inputs. Reference actuals are
    # deliberately NOT propagated; score() must load them separately.
    f = positive(args.target_clock_mhz, "target clock")
    ratio = model["anchor_clock_mhz"] / f
    rows = []
    for source in reference["predictions"]:
        w = source["workload"]
        if w in model["workloads"]:
            raise ValueError(f"score workload is present in calibration: {w}")
        x, provenance = exposure(args.replay_campaign, w)
        row = {"workload": w, "status": "candidate" if provenance == "replay_exposure_mixture" else "candidate_fallback",
               "exposure_source": provenance, "exposure_features": x, "endpoints": {}}
        for target in TARGETS:
            base = positive(source[f"{target}_predicted"], "reference prediction")
            share = float(np.dot(x, model["targets"][target]["coefficients"]))
            factor = response(target, ratio, share)
            width = model["targets"][target]["calibration_ratio_residual_q90"]
            low_factor, high_factor = sorted((response(target, ratio, 0), response(target, ratio, 1)))
            row["endpoints"][target] = {
                "reference": base, "elastic": base * factor,
                "full_core_scaling": base * ratio if target == "ncu_duration_ns" else base,
                "fixed_wait_share_hypothesis": share, "response_factor": factor,
                "response_scenario_interval": [base * low_factor, base * high_factor],
                "calibration_residual_interval": [max(0.0, base * (factor - width)), base * (factor + width)]}
        rows.append(row)
    out = {"schema": "sm86_clock_elasticity_predictions_v1", "created_unix_ns": time.time_ns(),
           "model": str(args.model), "reference": str(args.reference),
           "replay_campaign": str(args.replay_campaign), "target_clock_mhz_assumption": f,
           "target_NCU_labels_consumed": False, "predictions": rows,
           "adapter_seconds_including_feature_load": time.perf_counter() - start,
           "runtime_scope": "excludes production of reference predictions and replay input features",
           "limitations": model["limitations"]}
    write(args.output, out)
    return out


def score(args):
    predicted = json.loads(args.predictions.read_text())
    measured = read_rows(args.score_campaign)
    rows = []
    for p in predicted["predictions"]:
        if p["workload"] not in measured:
            raise ValueError(f"missing completed score observation: {p['workload']}")
        row = {"workload": p["workload"], "endpoints": {}}
        for target in TARGETS:
            actual = positive(measured[p["workload"]][target], "NCU observation")
            row["endpoints"][target] = {"actual": actual, **p["endpoints"][target]}
        rows.append(row)
    metrics = {}
    for target in TARGETS:
        a = np.array([r["endpoints"][target]["actual"] for r in rows])
        metrics[target] = {}
        for mode in ("reference", "elastic", "full_core_scaling"):
            p = np.array([r["endpoints"][target][mode] for r in rows])
            metrics[target][mode] = {"mape": float(np.mean(np.abs(p - a) / a)),
                                     "max_ape": float(np.max(np.abs(p - a) / a))}
    out = {"schema": "sm86_clock_elasticity_score_v1", "predictions": str(args.predictions),
           "score_campaign": str(args.score_campaign), "workload_count": len(rows),
           "metrics": metrics, "rows": rows,
           "scope": "prospective temporal repetition of previously studied workloads; not new-family generalization"}
    write(args.output, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("fit")
    p.add_argument("--low-campaign", type=Path, required=True)
    p.add_argument("--high-campaign", type=Path, required=True)
    p.add_argument("--calibration-cohort", choices=("base", "large"), required=True)
    p.add_argument("--anchor-clock-mhz", type=float, required=True)
    p.add_argument("--calibration-clock-mhz", type=float, required=True)
    p = subs.add_parser("predict")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--replay-campaign", type=Path, required=True)
    p.add_argument("--target-clock-mhz", type=float, required=True)
    p = subs.add_parser("score")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--score-campaign", type=Path, required=True)
    for p in subs.choices.values():
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.command](args)
    print(json.dumps({"output": str(args.output), "metrics": result.get("metrics"),
                      "targets": result.get("targets"),
                      "adapter_seconds": result.get("adapter_seconds_including_feature_load")}, indent=2))


if __name__ == "__main__":
    main()
