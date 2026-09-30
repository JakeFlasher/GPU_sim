#!/usr/bin/env python3
"""Project NCU endpoints from independent native CUPTI PC-state surfaces.

The CUPTI campaign is an independent observation lane.  Features include
function/PC coverage, correlation IDs, all-cycle scheduler states, and the
explicit no-issue subset.  Selected/not-selected are preserved as states but
are excluded from the semantic stall-pressure aggregate because NVIDIA
documents them as scheduler states rather than true stalls.  No NCU counter
from a score row is used as a feature.

The fit is family-disjoint.  Rows outside the feature hull receive a widened
candidate interval rather than an OOD refusal.  This is an endpoint projection,
not physical pipe, issue-slot, or cache-owner reconstruction.
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
PRESSURE = (
    "long_scoreboard", "imc_miss", "short_scoreboard", "wait",
    "branch_resolving", "drain", "misc", "math_pipe_throttle",
    "mio_throttle", "dispatch_stall", "no_instructions",
)
STATES = PRESSURE + ("selected", "not_selected")
FEATURE_PROFILES = {
    "full": None,
    # Fixed low-dimensional surface chosen from the semantic contract, not
    # from score-row NCU labels. It retains frontier scale plus the strongest
    # independent scheduler/cache pressure descriptors.
    "four": (
        "log_sim_event", "log_sim_envelope", "cupti_no_issue_fraction",
        "cupti_state_long_scoreboard", "cupti_state_imc_miss",
    ),
}


def _float(row, key, default=0.0):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _family(name):
    return name.split("__", 1)[0]


def _perf(campaign):
    with (campaign / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


def _cupti(campaign, discard_buffers=0):
    report = json.loads((campaign / "summary.json").read_text())
    output = {}
    for row in report.get("workloads", []):
        name = row.get("workload")
        if not name or row.get("status") != "completed":
            continue
        state = {key: 0.0 for key in STATES}
        no_issue = {key: 0.0 for key in STATES}
        unknown_state = unknown_no_issue = 0.0
        samples = records = dropped = buffers = qualified_pcs = 0.0
        correlations = set()
        functions = set()
        selected_buffers = []
        for summary in row.get("summaries", []):
            selected_buffers.extend(summary.get("buffers", []))
            if discard_buffers <= 0:
                samples += float(summary.get("total_samples", 0.0))
                records += float(summary.get("pc_record_count", 0.0))
                dropped += float(summary.get("dropped_samples", 0.0))
                buffers += float(summary.get("buffer_count", 0.0))
                qualified_pcs += float(summary.get("qualified_pc_count", summary.get("decoded_pc_count", 0.0)))
                for item in summary.get("buffers", []):
                    correlations.update(item.get("correlation_ids", []))
                for function, values in summary.get("functions", {}).items():
                    functions.add(function)
                    for key, value in values.get("scheduler_state_samples", {}).items():
                        if key in state:
                            state[key] += float(value)
                        else:
                            unknown_state += float(value)
                    for key, value in values.get("scheduler_no_issue_samples", {}).items():
                        if key in no_issue:
                            no_issue[key] += float(value)
                        else:
                            unknown_no_issue += float(value)
        if discard_buffers > 0:
            selected_buffers = selected_buffers[discard_buffers:]
            buffers = float(len(selected_buffers))
            for item in selected_buffers:
                samples += float(item.get("samples", 0.0))
                records += float(item.get("pc_records", 0.0))
                dropped += float(item.get("dropped", 0.0))
                correlations.update(item.get("correlation_ids", []))
                for key, value in item.get("scheduler_state_samples", {}).items():
                    if key in state:
                        state[key] += float(value)
                    else:
                        unknown_state += float(value)
                for key, value in item.get("scheduler_no_issue_samples", {}).items():
                    if key in no_issue:
                        no_issue[key] += float(value)
                    else:
                        unknown_no_issue += float(value)
        state_total = sum(state.values())
        no_issue_total = sum(no_issue.values())
        state_total_with_unknown = state_total + unknown_state
        no_issue_total_with_unknown = no_issue_total + unknown_no_issue
        feature = {
            "log_cupti_samples": math.log1p(samples),
            "log_cupti_records": math.log1p(records),
            "log_cupti_pc_coverage": math.log1p(qualified_pcs),
            "log_cupti_buffers": math.log1p(buffers),
            "log_cupti_functions": math.log1p(len(functions)),
            "log_cupti_correlations": math.log1p(len(correlations)),
            "cupti_drop_fraction": dropped / max(samples, 1.0),
            "cupti_no_issue_fraction": no_issue_total / max(state_total_with_unknown, 1.0),
            "cupti_unknown_state_fraction": unknown_state / max(state_total_with_unknown, 1.0),
            "cupti_unknown_no_issue_fraction": unknown_no_issue / max(no_issue_total_with_unknown, 1.0),
        }
        for key in STATES:
            feature[f"cupti_state_{key}"] = state[key] / max(state_total_with_unknown, 1.0)
            feature[f"cupti_no_issue_{key}"] = no_issue[key] / max(no_issue_total_with_unknown, 1.0)
        feature["cupti_pressure_fraction"] = sum(state[key] for key in PRESSURE) / max(state_total_with_unknown, 1.0)
        feature["cupti_pressure_no_issue_fraction"] = sum(no_issue[key] for key in PRESSURE) / max(no_issue_total_with_unknown, 1.0)
        output[name] = {"features": feature, "samples": samples, "records": records, "dropped": dropped,
                        "unknown_state_samples": unknown_state, "unknown_no_issue_samples": unknown_no_issue,
                        "qualified_pcs": qualified_pcs, "correlations": sorted(correlations),
                        "functions": sorted(functions)}
    if not output:
        raise RuntimeError(f"empty CUPTI campaign: {campaign}")
    return output


def _features(row, cupti):
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    issue = max(_float(row, "sim_last_issue_cycles"), 0.0)
    instructions = max(_float(row, "sim_warp_instructions"), 1.0)
    return {
        "log_sim_event": math.log1p(event),
        "log_sim_envelope": math.log1p(envelope),
        "log_sim_instructions": math.log1p(instructions),
        "sim_tail_per_instruction": max(event - issue, 0.0) / instructions,
        **cupti["features"],
    }


def _fit(x, y, ridge):
    scale = np.sqrt(np.mean(x[:, 1:] * x[:, 1:], axis=0))
    scale[scale == 0.0] = 1.0
    xs = x.copy()
    xs[:, 1:] /= scale
    beta = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    return beta, scale


def run(args):
    started = time.perf_counter()
    train_perf = _perf(args.train_campaign)
    score_perf = _perf(args.score_campaign)
    train_cupti = _cupti(args.train_cupti, args.discard_buffers)
    score_cupti = _cupti(args.score_cupti, args.discard_buffers)
    train_names = sorted(set(train_perf) & set(train_cupti))
    score_names = sorted(set(score_perf) & set(score_cupti))
    train_names = [name for name in train_names if _family(name) != args.holdout_family]
    score_names = [name for name in score_names if _family(name) == args.holdout_family]
    if not train_names or not score_names:
        raise RuntimeError("empty family-disjoint train or score cohort")
    train_maps = [_features(train_perf[name], train_cupti[name]) for name in train_names]
    score_maps = [_features(score_perf[name], score_cupti[name]) for name in score_names]
    names = sorted(train_maps[0])
    if any(sorted(item) != names for item in train_maps + score_maps):
        raise RuntimeError("CUPTI feature schema differs")
    profile = FEATURE_PROFILES[args.feature_profile]
    if profile is not None:
        missing = [name for name in profile if name not in names]
        if missing:
            raise RuntimeError(f"feature profile missing fields: {missing}")
        names = list(profile)
    x_train = np.asarray([[1.0] + [item[name] for name in names] for item in train_maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in names] for item in score_maps], dtype=float)
    predictions, intervals, coeffs, residual_q90 = {}, {}, {}, {}
    for target in TARGETS:
        y = np.asarray([_float(train_perf[name], target) for name in train_names], dtype=float)
        if np.any(y <= 0.0):
            raise RuntimeError(f"invalid train target {target}")
        beta, scale = _fit(x_train, np.log(y), args.ridge)
        train_pred = np.exp((x_train / np.r_[1.0, scale]) @ beta)
        score_pred = np.exp((x_score / np.r_[1.0, scale]) @ beta)
        residual_q90[target] = float(np.quantile(np.abs(train_pred - y), 0.90))
        predictions[target] = np.maximum(score_pred, 0.0)
        coeffs[target] = beta.tolist()
        intervals[target] = (
            np.maximum(score_pred - residual_q90[target], 0.0),
            score_pred + residual_q90[target],
        )
    feature_scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    feature_scale[feature_scale == 0.0] = 1.0
    distances = np.asarray([float(np.min(np.linalg.norm((x_train[:, 1:] - row[1:]) / feature_scale, axis=1))) for row in x_score])
    factor = 1.0 + np.maximum(distances - args.ood_threshold, 0.0) / max(args.ood_threshold, 1.0)
    rows = []
    for index, name in enumerate(score_names):
        item = {
            "workload": name,
            "status": "candidate" if distances[index] <= args.ood_threshold else "candidate_fallback",
            "coverage_status": "native_cupti_surface_covered" if distances[index] <= args.ood_threshold else "native_cupti_surface_fallback_covered",
            "feature_distance": float(distances[index]), "fallback_distance_factor": float(factor[index]),
            "cupti_observation": score_cupti[name],
        }
        for target in TARGETS:
            actual = _float(score_perf[name], target, float("nan"))
            predicted = float(predictions[target][index])
            width = residual_q90[target] * factor[index]
            item[f"{target}_predicted"] = predicted
            item[f"{target}_actual"] = actual if math.isfinite(actual) else None
            item[f"{target}_ape"] = abs(predicted - actual) / actual if math.isfinite(actual) and actual > 0 else None
            item[f"{target}_interval_low"] = max(predicted - width, 0.0)
            item[f"{target}_interval_high"] = predicted + width
        rows.append(item)
    metrics = {}
    for target in TARGETS:
        apes = [item[f"{target}_ape"] for item in rows if item[f"{target}_ape"] is not None]
        metrics[target] = {"count": len(apes), "mape": float(np.mean(apes)), "max_ape": float(np.max(apes))}
    report = {
        "schema": "sm86_native_cupti_surface_projection_v1",
        "train_campaign": str(args.train_campaign), "score_campaign": str(args.score_campaign),
        "train_cupti": str(args.train_cupti), "score_cupti": str(args.score_cupti),
        "discard_buffers": args.discard_buffers,
        "holdout_family": args.holdout_family, "train_workloads": train_names, "score_workloads": score_names,
        "target_labels_score_only": True, "cupti_labels_used_as_features": True,
        "feature_names": names, "ridge": args.ridge, "ood_threshold": args.ood_threshold,
        "model": {
            "features": "replay frontiers plus native CUPTI PC/scheduler-state surface",
            "feature_profile": args.feature_profile,
            "selected_not_selected": "retained as states, excluded from pressure aggregate",
            "no_issue": "separate subset, never added to all-cycle states",
            "fit": "log NCU endpoint ridge on opposite workload family",
            "interval": "training residual q90 widened by feature distance; not confidence",
            "coverage": "candidate or candidate_fallback for every row",
        },
        "residual_q90": residual_q90, "coefficients": coeffs, "metrics": metrics,
        "model_seconds": time.perf_counter() - started,
        "predictions": rows,
        "limitations": [
            "CUPTI PC sampling is aggregate sampled scheduler state, not a physical pipe or issue-slot trace",
            "sample fractions are not complete cycle fractions and selected/not_selected are not true stalls",
            "warmup campaign provides phase-local observation but does not prove full-cache ownership or flush semantics",
            "family-disjoint related workloads are not architecture-wide generalization",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-campaign", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--train-cupti", type=Path, required=True)
    parser.add_argument("--score-cupti", type=Path, required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.25)
    parser.add_argument("--feature-profile", choices=tuple(FEATURE_PROFILES), default="full")
    parser.add_argument("--discard-buffers", type=int, default=0,
                        help="discard this many leading CUPTI buffers per workload (warmup phase)")
    parser.add_argument("--ood-threshold", type=float, default=3.0)
    args = parser.parse_args()
    if args.discard_buffers < 0:
        parser.error("--discard-buffers must be nonnegative")
    if not math.isfinite(args.ridge) or args.ridge <= 0 or not math.isfinite(args.ood_threshold) or args.ood_threshold <= 0:
        parser.error("ridge and OOD threshold must be positive and finite")
    report = run(args)
    print(json.dumps({"holdout_family": report["holdout_family"], "metrics": report["metrics"], "rows": len(report["predictions"]), "model_seconds": report["model_seconds"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
