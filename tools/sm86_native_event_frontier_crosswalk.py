#!/usr/bin/env python3
"""Family-disjoint native-event/frontier crosswalk for SM86 endpoints.

The native CUDA event is an independent endpoint observation. Replay frontiers
provide scale/context, while NCU labels from only the opposite workload family
fit a log(NCU/native-event) response. No NCU score counter is a feature for its
own held-out workload. This is a measurement crosswalk, not a unified-clock or
physical scheduler model.
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


def _float(row, key, default=0.0):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _rows(path):
    with (path / "performance.csv").open(newline="") as stream:
        return {row["workload"]: row for row in csv.DictReader(stream) if row.get("status") == "completed"}


def _features(row):
    return {
        "log_native_event_ns": math.log(max(_float(row, "native_mean_ms") * 1_000_000.0, 1.0)),
        "log_sim_event": math.log1p(max(_float(row, "sim_event_cycles"), 1.0)),
        "log_sim_envelope": math.log1p(max(_float(row, "sim_gpc_envelope_cycles"), 1.0)),
        "log_sim_instructions": math.log1p(max(_float(row, "sim_warp_instructions"), 1.0)),
        "sim_tail_per_instruction": max(_float(row, "sim_event_cycles") - _float(row, "sim_last_issue_cycles"), 0.0) / max(_float(row, "sim_warp_instructions"), 1.0),
    }


def _fit(x, y, ridge):
    scale = np.sqrt(np.mean(x[:, 1:] * x[:, 1:], axis=0)); scale[scale == 0] = 1
    z = x.copy(); z[:, 1:] /= scale
    beta = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ y)
    return beta, scale


def run(args):
    started = time.perf_counter()
    train_all, score_all = _rows(args.train_campaign), _rows(args.score_campaign)
    train_names = [name for name in sorted(train_all) if name.split("__", 1)[0] != args.holdout_family]
    score_names = [name for name in sorted(score_all) if name.split("__", 1)[0] == args.holdout_family]
    if not train_names or not score_names:
        raise RuntimeError("empty family-disjoint cohort")
    names = ["log_sim_event", "log_sim_envelope", "log_sim_instructions", "sim_tail_per_instruction"]
    x_train = np.asarray([[1.0] + [_features(train_all[name])[key] for key in names] for name in train_names], dtype=float)
    x_score = np.asarray([[1.0] + [_features(score_all[name])[key] for key in names] for name in score_names], dtype=float)
    predictions, intervals, coefficients, residual_q90 = {}, {}, {}, {}
    for target in TARGETS:
        native_train = np.asarray([max(_float(train_all[name], "native_mean_ms") * 1_000_000.0, 1.0) for name in train_names])
        native_score = np.asarray([max(_float(score_all[name], "native_mean_ms") * 1_000_000.0, 1.0) for name in score_names])
        y = np.asarray([_float(train_all[name], target) for name in train_names])
        beta, scale = _fit(x_train, np.log(y / native_train), args.ridge)
        train_pred = native_train * np.exp((x_train / np.r_[1.0, scale]) @ beta)
        score_pred = native_score * np.exp((x_score / np.r_[1.0, scale]) @ beta)
        q90 = float(np.quantile(np.abs(train_pred - y), .90))
        predictions[target] = score_pred; residual_q90[target] = q90; coefficients[target] = beta.tolist()
        intervals[target] = (np.maximum(score_pred - q90, 0.0), score_pred + q90)
    scale = np.sqrt(np.mean(x_train[:, 1:] ** 2, axis=0)); scale[scale == 0] = 1
    distance = np.asarray([float(np.min(np.linalg.norm((x_train[:, 1:] - row[1:]) / scale, axis=1))) for row in x_score])
    factor = 1.0 + np.maximum(distance - args.ood_threshold, 0.0) / max(args.ood_threshold, 1.0)
    rows=[]
    for i,name in enumerate(score_names):
        item={"workload":name,"status":"candidate" if distance[i] <= args.ood_threshold else "candidate_fallback","coverage_status":"native_event_frontier_covered","feature_distance":float(distance[i]),"native_event_ns":_float(score_all[name],"native_mean_ms")*1_000_000.0}
        for target in TARGETS:
            actual=_float(score_all[name],target,float('nan')); pred=float(predictions[target][i]); width=residual_q90[target]*factor[i]
            item[f"{target}_predicted"]=pred; item[f"{target}_actual"]=actual if math.isfinite(actual) else None; item[f"{target}_ape"]=abs(pred-actual)/actual if math.isfinite(actual) and actual>0 else None; item[f"{target}_interval_low"]=max(pred-width,0.0); item[f"{target}_interval_high"]=pred+width
        rows.append(item)
    metrics={}
    for target in TARGETS:
        a=[row[f"{target}_ape"] for row in rows if row[f"{target}_ape"] is not None]; metrics[target]={"count":len(a),"mae":float(np.mean([abs(row[f"{target}_predicted"]-row[f"{target}_actual"]) for row in rows])),"mape":float(np.mean(a)),"max_ape":float(np.max(a))}
    report={"schema":"sm86_native_event_frontier_crosswalk_v1","train_campaign":str(args.train_campaign),"score_campaign":str(args.score_campaign),"holdout_family":args.holdout_family,"train_workloads":train_names,"score_workloads":score_names,"target_labels_score_only":True,"feature_names":names,"ridge":args.ridge,"ood_threshold":args.ood_threshold,"model":"log(NCU/native CUDA-event) response from replay frontier features; ridge fixed before score","residual_q90":residual_q90,"coefficients":coefficients,"metrics":metrics,"model_seconds":time.perf_counter()-started,"predictions":rows,"limitations":["CUDA event and NCU endpoints remain distinct","family-disjoint related workloads are not architecture-wide generalization","physical pipe, cache ownership/read-from, and unified timing remain unresolved"]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--train-campaign',type=Path,required=True); parser.add_argument('--score-campaign',type=Path,required=True); parser.add_argument('--holdout-family',choices=('base','large'),required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--ridge',type=float,default=1.0); parser.add_argument('--ood-threshold',type=float,default=3.0); args=parser.parse_args();
    if args.ridge<=0 or not math.isfinite(args.ridge): parser.error('--ridge must be positive and finite')
    report=run(args); print(json.dumps({'holdout_family':report['holdout_family'],'metrics':report['metrics'],'rows':len(report['predictions']),'output':str(args.output)},indent=2))


if __name__ == '__main__': main()
