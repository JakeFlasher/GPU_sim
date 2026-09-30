#!/usr/bin/env python3
"""Project endpoints from CUPTI PC states joined to semantic opcode groups.

The join uses PC offsets from CUPTI and the retained cuobjdump/control sidecar.
It can say that sampled long-scoreboard or branch states occurred on a
semantic memory/math/control group; it cannot identify a physical pipe,
sub-unit, or issue slot. NCU labels are used only for opposite-family fitting
and post-prediction score reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from sm86_ncu_projection_surrogate import _opcode_group

TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")
GROUPS = ("memory", "tensor", "math", "sfu", "control", "special")


def _float(row, key, default=0.0):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _perf(campaign):
    with (campaign / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


def _control(campaign, workload):
    data = json.loads((campaign / workload / "control.json").read_text())
    mapping = {}
    for function, records in data.get("functions", {}).items():
        for record in records:
            mapping[(function, int(record.get("pc", 0)))] = _opcode_group(str(record.get("opcode_text", "UNKNOWN")))
    return mapping


def _summary_files(cupti, workload):
    return sorted((cupti / workload).glob("decode_*.summary.json"))


def _surface(cupti, campaign, workload):
    mapping = _control(campaign, workload)
    state = {group: {key: 0.0 for key in ("long_scoreboard", "imc_miss", "short_scoreboard", "wait", "branch_resolving", "math_pipe_throttle", "mio_throttle", "drain", "no_instructions")} for group in GROUPS}
    records = samples = unknown = 0.0
    for path in _summary_files(cupti, workload):
        report = json.loads(path.read_text())
        samples += float(report.get("total_samples", 0.0))
        for record in report.get("pc_records", []):
            group = mapping.get((record.get("function", ""), int(record.get("pc_offset", 0))), "special")
            if group not in state:
                group = "special"
            records += 1.0
            for name, value in record.get("stall_reason_samples", {}).items():
                reason = name.removeprefix("smsp__pcsamp_warps_issue_stalled_")
                reason = reason.removesuffix("_not_issued")
                if reason in state[group]:
                    state[group][reason] += float(value)
                else:
                    unknown += float(value)
    all_state = {key: sum(state[group][key] for group in GROUPS) for key in next(iter(state.values()))}
    total = sum(all_state.values())
    feature = {
        "log_cupti_samples": math.log1p(samples),
        "log_cupti_pc_records": math.log1p(records),
        "cupti_unknown_fraction": unknown / max(total + unknown, 1.0),
    }
    # Fixed semantic joins: memory scoreboard, tensor/math pipe pressure,
    # control branch state, and SFU/MIO pressure. No physical-route naming.
    for group, prefix in (("memory", "memory"), ("tensor", "tensor"), ("math", "math"), ("control", "control"), ("sfu", "sfu")):
        denom = max(sum(state[group].values()), 1.0)
        feature[f"{prefix}_long_fraction"] = state[group]["long_scoreboard"] / denom
        feature[f"{prefix}_imc_fraction"] = state[group]["imc_miss"] / denom
        feature[f"{prefix}_short_fraction"] = state[group]["short_scoreboard"] / denom
        feature[f"{prefix}_wait_fraction"] = state[group]["wait"] / denom
        feature[f"{prefix}_branch_fraction"] = state[group]["branch_resolving"] / denom
        feature[f"{prefix}_math_pipe_fraction"] = state[group]["math_pipe_throttle"] / denom
        feature[f"{prefix}_mio_fraction"] = state[group]["mio_throttle"] / denom
    feature["memory_sample_weight"] = sum(state["memory"].values()) / max(total, 1.0)
    feature["tensor_sample_weight"] = sum(state["tensor"].values()) / max(total, 1.0)
    feature["control_sample_weight"] = sum(state["control"].values()) / max(total, 1.0)
    return {"features": feature, "records": records, "samples": samples, "state": state, "unknown": unknown}


def _features(row, surface):
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    return {
        "log_sim_event": math.log1p(event),
        "log_sim_envelope": math.log1p(envelope),
        "log_sim_instructions": math.log1p(max(_float(row, "sim_warp_instructions"), 1.0)),
        **surface["features"],
    }


def _fit(x, y, ridge):
    scale = np.sqrt(np.mean(x[:, 1:] * x[:, 1:], axis=0))
    scale[scale == 0.0] = 1.0
    xs = x.copy(); xs[:, 1:] /= scale
    return np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y), scale


def run(args):
    started = time.perf_counter()
    train_perf, score_perf = _perf(args.train_campaign), _perf(args.score_campaign)
    train_names = [name for name in sorted(train_perf) if name.split("__", 1)[0] != args.holdout_family]
    score_names = [name for name in sorted(score_perf) if name.split("__", 1)[0] == args.holdout_family]
    train_surface = {name: _surface(args.train_cupti, args.train_campaign, name) for name in train_names}
    score_surface = {name: _surface(args.score_cupti, args.score_campaign, name) for name in score_names}
    train_maps = [_features(train_perf[name], train_surface[name]) for name in train_names]
    score_maps = [_features(score_perf[name], score_surface[name]) for name in score_names]
    all_names = sorted(train_maps[0])
    selected = [name for name in all_names if name in {
        "log_sim_event", "log_sim_envelope", "log_sim_instructions",
        "memory_long_fraction", "memory_imc_fraction", "memory_short_fraction",
        "tensor_math_pipe_fraction", "tensor_long_fraction", "control_branch_fraction",
        "sfu_mio_fraction", "cupti_unknown_fraction", "memory_sample_weight",
    }]
    x_train = np.asarray([[1.0] + [item[name] for name in selected] for item in train_maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in selected] for item in score_maps], dtype=float)
    predictions, intervals, coeffs, residuals = {}, {}, {}, {}
    for target in TARGETS:
        y = np.asarray([_float(train_perf[name], target) for name in train_names], dtype=float)
        beta, scale = _fit(x_train, np.log(y), args.ridge)
        train_pred = np.exp((x_train / np.r_[1.0, scale]) @ beta)
        score_pred = np.exp((x_score / np.r_[1.0, scale]) @ beta)
        residuals[target] = float(np.quantile(np.abs(train_pred - y), 0.90))
        predictions[target] = score_pred
        intervals[target] = (np.maximum(score_pred - residuals[target], 0.0), score_pred + residuals[target])
        coeffs[target] = beta.tolist()
    feature_scale = np.sqrt(np.mean(x_train[:, 1:] ** 2, axis=0)); feature_scale[feature_scale == 0] = 1
    distances = np.asarray([float(np.min(np.linalg.norm((x_train[:, 1:] - row[1:]) / feature_scale, axis=1))) for row in x_score])
    factor = 1.0 + np.maximum(distances - args.ood_threshold, 0.0) / max(args.ood_threshold, 1.0)
    rows=[]
    for i,name in enumerate(score_names):
        item={"workload":name,"status":"candidate" if distances[i] <= args.ood_threshold else "candidate_fallback","coverage_status":"cupti_opcode_surface_covered","feature_distance":float(distances[i]),"semantic_surface":score_surface[name]}
        for target in TARGETS:
            actual=_float(score_perf[name],target,float('nan')); pred=float(predictions[target][i]); width=residuals[target]*factor[i]
            item[f"{target}_predicted"]=pred; item[f"{target}_actual"]=actual if math.isfinite(actual) else None; item[f"{target}_ape"]=abs(pred-actual)/actual if math.isfinite(actual) and actual>0 else None; item[f"{target}_interval_low"]=max(pred-width,0.0); item[f"{target}_interval_high"]=pred+width
        rows.append(item)
    metrics={}
    for target in TARGETS:
        a=[row[f"{target}_ape"] for row in rows if row[f"{target}_ape"] is not None]; metrics[target]={"count":len(a),"mape":float(np.mean(a)),"max_ape":float(np.max(a))}
    report={"schema":"sm86_cupti_opcode_surface_projection_v1","train_campaign":str(args.train_campaign),"score_campaign":str(args.score_campaign),"train_cupti":str(args.train_cupti),"score_cupti":str(args.score_cupti),"holdout_family":args.holdout_family,"target_labels_score_only":True,"feature_names":selected,"ridge":args.ridge,"metrics":metrics,"model_seconds":time.perf_counter()-started,"predictions":rows,"limitations":["PC-to-opcode join is semantic only; it does not expose physical route or issue slot","CUPTI records are sampled states, not complete dynamic instruction timing","cache ownership/read-from and unified timing remain unresolved"] ,"coefficients":coeffs,"training_residual_q90":residuals}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--train-campaign',type=Path,required=True); parser.add_argument('--score-campaign',type=Path,required=True); parser.add_argument('--train-cupti',type=Path,required=True); parser.add_argument('--score-cupti',type=Path,required=True); parser.add_argument('--holdout-family',choices=('base','large'),required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--ridge',type=float,default=0.01); parser.add_argument('--ood-threshold',type=float,default=3.0); args=parser.parse_args(); report=run(args); print(json.dumps({"holdout_family":report['holdout_family'],"metrics":report['metrics'],"output":str(args.output)},indent=2))


if __name__ == '__main__': main()
