#!/usr/bin/env python3
"""Validate observational NCU intervention comparisons.

This validator checks repetition preservation, prediction-before-observation
identity, and availability of matching mechanism counters. It deliberately
never emits a causal hardware claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prediction_contract import validate_prediction_document


def validate_comparison(plan: dict[str, Any], comparison: dict[str, Any], docs_root: Path) -> dict[str, Any]:
    if plan.get("schema") != "sm86_ncu_intervention_plan_v1":
        raise ValueError("invalid intervention plan schema")
    if comparison.get("schema") != "sm86_accuracy_intervention_comparison_v1":
        raise ValueError("invalid intervention comparison schema")
    if comparison.get("causal_claim") is not False:
        raise ValueError("intervention comparison must not claim causality")
    rows = {row["workload"]: row for row in comparison.get("rows", [])}
    baseline_prediction = json.loads((docs_root / plan["baseline_prediction"]).read_text())
    validate_prediction_document(baseline_prediction)
    baseline_rows = {row["workload"]: row for row in baseline_prediction["rows"]}
    summaries = []
    for contrast in plan["contrasts"]:
        prediction_path = docs_root / contrast["prediction"]
        prediction = json.loads(prediction_path.read_text())
        validate_prediction_document(prediction)
        if prediction.get("prediction_stage") != "pre_observation":
            raise ValueError(f"{contrast['id']} prediction is not pre-observation")
        workload_set = set(contrast["workloads"])
        direction_counts: dict[str, dict[str, int]] = {}
        mechanism_available = True
        simulator_invariant = True
        prediction_rows = {row["workload"]: row for row in prediction["rows"]}
        for workload in workload_set:
            base_row = baseline_rows.get(workload)
            variant_row = prediction_rows.get(workload)
            if base_row is None or variant_row is None or base_row.get("prediction") != variant_row.get("prediction"):
                simulator_invariant = False
            row = rows.get(workload)
            if row is None or contrast["id"].replace("_all_workloads", "") not in row["contrasts"]:
                # Comparison keys are deliberately compact and may omit the
                # suffix used by the plan id.
                key = "cache_none" if contrast["variable"] == "cache_policy" else "clock_base"
                if row is None or key not in row["contrasts"]:
                    raise ValueError(f"missing comparison row for {contrast['id']}:{workload}")
                intervention = row["contrasts"][key]
            else:
                intervention = row["contrasts"][contrast["id"].replace("_all_workloads", "")]
            if len(intervention["samples"]["gpc__cycles_elapsed.max"]) != contrast["repetitions"]:
                raise ValueError(f"{contrast['id']}:{workload} repetition count mismatch")
            for metric, direction in intervention["direction_vs_all_cache"].items():
                direction_counts.setdefault(metric, Counter())[direction] += 1
                if metric != "gpc__cycles_elapsed.max" and not intervention["samples"].get(metric):
                    mechanism_available = False
        # Simulator predictions are intentionally expected to be independent of
        # post-prediction NCU intervention values in this diagnostic program.
        summaries.append({
            "id": contrast["id"],
            "variable": contrast["variable"],
            "workload_count": len(workload_set),
            "target_direction_counts": dict(direction_counts.get("gpc__cycles_elapsed.max", {})),
            "mechanism_counter_directions": {metric: dict(counts) for metric, counts in direction_counts.items() if metric != "gpc__cycles_elapsed.max"},
            "mechanism_counters_available": mechanism_available,
            "simulator_projection_invariant": simulator_invariant,
            "consistency_status": "consistent_observational_contrast" if simulator_invariant and mechanism_available else "inconsistent_refuse_causal_claim",
            "causal_claim": False,
            "interpretation": "observational_direction_and_metric_consistency_only",
        })
    comparison["contrast_summaries"] = summaries
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--docs-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_comparison(json.loads(args.plan.read_text()), json.loads(args.comparison.read_text()), args.docs_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "valid", "contrasts": len(result["contrast_summaries"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
