#!/usr/bin/env python3
"""Add opposite-cohort clock/cache scenario spread without changing points.

Calibration uses only the other cohort across four prior campaigns. Base/large
is a cohort split, not unrelated kernel-family generalization. Bounds are
empirical scenarios, not confidence intervals.
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
COHORTS = ("base", "large")


def cohort(name):
    return name.split("__", 1)[0]


def index_rows(rows, source):
    result = {}
    for row in rows:
        name = row.get("workload")
        if not isinstance(name, str) or not name:
            raise ValueError(f"missing workload in {source}")
        if name in result:
            raise ValueError(f"duplicate workload {name!r} in {source}")
        result[name] = row
    return result


def read_campaign(path):
    with (path / "performance.csv").open(newline="") as stream:
        return index_rows((r for r in csv.DictReader(stream)
                           if r.get("status") == "completed"), path)


def read_predictions(path):
    value = json.loads(path.read_text())
    rows = value.get("predictions")
    if not isinstance(rows, list):
        rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"prediction list missing in {path}")
    return index_rows(rows, path)


def finite(value, context, positive=False):
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"invalid {context}: {value!r}")
    return result


def summarize(rows):
    metrics = {}
    for target in TARGETS:
        actual = np.asarray([r[f"{target}_actual"] for r in rows])
        low = np.asarray([r[f"{target}_interval_low"] for r in rows])
        high = np.asarray([r[f"{target}_interval_high"] for r in rows])
        points = np.asarray([r[f"{target}_predicted"] for r in rows])
        width = (high - low) / actual
        covered = (low <= actual) & (actual <= high)
        metrics[target] = {
            "count": len(rows), "covered_count": int(covered.sum()),
            "coverage": float(covered.mean()),
            "point_mape": float((abs(points - actual) / actual).mean()),
            "mean_relative_width": float(width.mean()),
            "median_relative_width": float(np.median(width)),
            "max_relative_width": float(width.max()),
        }
    return metrics


def run(args):
    started = time.perf_counter()
    paths = (args.low_cold, args.low_warm, args.high_cold, args.high_warm)
    campaigns = [read_campaign(path) for path in paths]
    score = read_campaign(args.score_campaign)
    predictions = read_predictions(args.predictions)
    eligible = sorted(set.intersection(*(set(c) for c in campaigns)))
    score_names = sorted(name for name in score
                         if args.holdout_family == "all" or cohort(name) == args.holdout_family)
    if not score_names:
        raise ValueError("empty score cohort")
    missing = set(score_names) - predictions.keys()
    if missing:
        raise ValueError(f"source predictions missing score rows: {sorted(missing)}")

    folds = {}
    for heldout in sorted({cohort(name) for name in score_names}):
        if heldout not in COHORTS:
            raise ValueError(f"unknown cohort {heldout!r}; expected base or large")
        train = [name for name in eligible
                 if cohort(name) in COHORTS and cohort(name) != heldout]
        selected = [name for name in score_names if cohort(name) == heldout]
        if not train or set(train) & set(selected):
            raise ValueError(f"{heldout}: missing or overlapping calibration rows")
        ratios = {}
        for target in TARGETS:
            interaction = []
            for name in train:
                lc, lw, hc, hw = [finite(c[name][target], f"{name}/{target}", True)
                                  for c in campaigns]
                interaction.append(finite((hw / hc) / (lw / lc), "interaction", True))
            ratios[target] = {
                "p10": float(np.quantile(interaction, .10)),
                "median": float(np.median(interaction)),
                "p90": float(np.quantile(interaction, .90)),
                "min": min(interaction), "max": max(interaction),
            }
        folds[heldout] = {"calibration_names": train, "score_names": selected,
                          "interaction_ratios": ratios}

    # Prediction pass: do not read any actual endpoint from score rows here.
    rows = []
    for name in score_names:
        source = predictions[name]
        fold = folds[cohort(name)]
        item = {"workload": name, "status": source.get("status", "candidate"),
                "holdout_cohort": cohort(name),
                "interaction_interval_source": "opposite_cohort_prior_campaigns",
                "interaction_calibration_count": len(fold["calibration_names"])}
        for target in TARGETS:
            point = finite(source[f"{target}_predicted"], f"{name}/{target} point", True)
            low = finite(source[f"{target}_interval_low"], f"{name}/{target} low")
            high = finite(source[f"{target}_interval_high"], f"{name}/{target} high")
            if not 0 <= low <= point <= high:
                raise ValueError(f"{name}/{target}: source bounds must contain the point")
            q = fold["interaction_ratios"][target]
            item[f"{target}_predicted"] = point
            item[f"{target}_base_interval_low"] = low
            item[f"{target}_base_interval_high"] = high
            item[f"{target}_interval_low"] = finite(low * min(1., q["p10"] / q["median"]), "expanded low")
            item[f"{target}_interval_high"] = finite(high * max(1., q["p90"] / q["median"]), "expanded high")
        rows.append(item)

    # Score pass: each workload appears exactly once, in its own held-out fold.
    for row in rows:
        for target in TARGETS:
            row[f"{target}_actual"] = finite(score[row["workload"]][target], target, True)
    report = {
        "schema": "sm86_interaction_interval_envelope_v2",
        "source_predictions": str(args.predictions),
        "score_campaign": str(args.score_campaign),
        "calibration_campaigns": dict(zip(("low_cold", "low_warm", "high_cold", "high_warm"), map(str, paths))),
        "holdout_family": args.holdout_family, "folds": folds,
        "coverage": {"requested_rows": len(score_names), "predicted_rows": len(rows)},
        "metrics": summarize(rows), "rows": rows,
        "prediction_and_scoring_seconds": time.perf_counter() - started,
        "point_predictions_changed": False,
        "target_labels_used": "score values only in final metric pass; calibration labels from opposite cohort",
        "limitations": ["base/large cohorts may contain related kernels",
                        "p10/p90 scenarios have no formal coverage guarantee",
                        "v1 accidentally scored calibration-cohort rows; pooled 30-row summaries are superseded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("low-cold", "low-warm", "high-cold", "high-warm", "predictions", "score-campaign", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--holdout-family", choices=(*COHORTS, "all"), required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"metrics": result["metrics"], "coverage": result["coverage"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
