#!/usr/bin/env python3
"""Join model predictions with independent NCU observations after prediction.

The prediction document is intentionally forbidden from containing observed
targets.  The join key is workload identity only; split membership is supplied
by the prediction/development manifest and is never selected using NCU error.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from prediction_contract import compare_target_contracts, validate_target_contract


SPLITS = {"calibration", "development", "held_out"}


def _partial_order(group: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    """Return only interval-disjoint ordering relations.

    Open or overlapping intervals are deliberately indeterminate; lexical
    order is not evidence of a model ranking.
    """
    ordered_pairs: list[list[str]] = []
    indeterminate_pairs: list[list[str]] = []
    for index, left in enumerate(group):
        for right in group[index + 1 :]:
            left_upper = left.get("upper")
            right_upper = right.get("upper")
            if isinstance(left_upper, (int, float)) and left_upper < right["lower"]:
                ordered_pairs.append([left["intervention"], right["intervention"]])
            elif isinstance(right_upper, (int, float)) and right_upper < left["lower"]:
                ordered_pairs.append([right["intervention"], left["intervention"]])
            else:
                indeterminate_pairs.append([left["intervention"], right["intervention"]])
    return {"ordered_pairs": ordered_pairs, "indeterminate_pairs": indeterminate_pairs}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def evaluate_prediction_splits(predictions: dict[str, Any], observations: dict[str, Any]) -> dict[str, Any]:
    if predictions.get("schema") != "reification_prediction_v2":
        raise ValueError("prediction schema must be reification_prediction_v2")
    if observations.get("schema") != "ncu_observation_v2":
        raise ValueError("observation schema must be ncu_observation_v2")
    prediction_contract = validate_target_contract(
        predictions.get("target_contract"), "prediction", require_hardware=True
    )
    observation_contract = validate_target_contract(
        observations.get("target_contract"), "observation", require_hardware=True
    )
    compare_target_contracts(prediction_contract, observation_contract)
    prediction_rows = predictions.get("rows")
    observation_rows = observations.get("rows")
    if not isinstance(prediction_rows, list) or not isinstance(observation_rows, list):
        raise ValueError("prediction and observation rows must be arrays")
    if not prediction_rows:
        raise ValueError("prediction rows must be nonempty")
    observed_forbidden = {"ncu", "observed", "observation", "target", "target_value"}
    predictions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    workload_families: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    for index, row in enumerate(prediction_rows):
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {index} must be an object")
        if observed_forbidden.intersection(row):
            raise ValueError("prediction rows must not contain observed target fields")
        key = row.get("workload")
        split = row.get("split")
        if not isinstance(key, str) or not key or split not in SPLITS:
            raise ValueError(f"prediction row {index} needs workload and split")
        family = row.get("family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"prediction row {index} needs a nonempty family")
        prior_family = workload_families.setdefault(key, family)
        if prior_family != family:
            raise ValueError(f"workload {key!r} appears with multiple families")
        prior_split = family_splits.setdefault(family, split)
        if prior_split != split:
            raise ValueError(f"family {family!r} crosses prediction splits")
        intervention = row.get("intervention", "default")
        if not isinstance(intervention, str) or not intervention:
            raise ValueError(f"prediction row {index}.intervention must be nonempty")
        identity = (key, intervention)
        if identity in predictions_by_key:
            raise ValueError(f"duplicate prediction workload/intervention {identity!r}")
        predictions_by_key[identity] = row
    observations_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(observation_rows):
        if not isinstance(row, dict) or not isinstance(row.get("workload"), str):
            raise ValueError(f"observation row {index} needs workload")
        key = row["workload"]
        for field in ("metric", "endpoint", "clock_domain", "hardware"):
            if field in row and row[field] != observation_contract[field]:
                raise ValueError(f"observation row {index}.{field} disagrees with target_contract")
        observations_by_key[key].append({
            "value": _finite(row.get("value"), f"observation row {index}.value"),
            "metric": row.get("metric", observation_contract["metric"]),
            "endpoint": row.get("endpoint", observation_contract["endpoint"]),
            "clock_domain": row.get("clock_domain", observation_contract["clock_domain"]),
            "hardware": row.get("hardware", observation_contract["hardware"]),
        })
    for key, rows in observations_by_key.items():
        if len(rows) != observation_contract["repetitions"]:
            raise ValueError(
                f"observation workload {key!r} has {len(rows)} rows; "
                f"target contract declares {observation_contract['repetitions']} repetitions"
            )

    joined = []
    for (key, intervention), row in predictions_by_key.items():
        status = row.get("status", "predicted")
        if status not in {"predicted", "refused"}:
            raise ValueError(f"prediction {key!r} has unsupported status {status!r}")
        if status == "refused":
            joined.append({"workload": key, "intervention": intervention, "split": row["split"], "status": "refused"})
            continue
        prediction = row.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"prediction {key!r} lacks prediction interval")
        point = _finite(prediction.get("point"), f"prediction {key}.point")
        lower = _finite(prediction.get("lower"), f"prediction {key}.lower")
        upper_raw = prediction.get("upper")
        upper = None if upper_raw is None else _finite(upper_raw, f"prediction {key}.upper")
        if upper is not None and lower > upper:
            raise ValueError(f"prediction {key!r} interval is inverted")
        if point < lower or (upper is not None and point > upper):
            raise ValueError(f"prediction {key!r} point is outside its interval")
        if key not in observations_by_key:
            raise ValueError(f"prediction {key!r} has no independent observation")
        observations_for_workload = observations_by_key[key]
        observation = observations_for_workload[0]
        observed = statistics.median(row["value"] for row in observations_for_workload)
        joined.append({
            "workload": key,
            "intervention": intervention,
            "split": row["split"],
            "status": "predicted",
            "point": point,
            "point_basis": row.get("point_basis", "unspecified"),
            "lower": lower,
            "upper": upper,
            "observed": observed,
            "observation_samples": len(observations_for_workload),
            "observation_metric": observation["metric"],
            "observation_endpoint": observation["endpoint"],
            "observation_clock_domain": observation["clock_domain"],
            "observation_hardware": observation["hardware"],
            "covered": lower <= observed and (upper is None or observed <= upper),
            "width": upper - lower if upper is not None else None,
            "ape_pct": abs(point - observed) / abs(observed) * 100 if observed != 0 else None,
            "runtime_ms": row.get("runtime_ms"),
            "memory_bytes": row.get("memory_bytes"),
            "step_reduction": row.get("step_reduction"),
        })

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_split[row["split"]].append(row)
    split_reports = {}
    for split in sorted(SPLITS):
        rows = by_split.get(split, [])
        predicted = [row for row in rows if row["status"] == "predicted"]
        refused = [row for row in rows if row["status"] == "refused"]
        widths = [row["width"] for row in predicted if row["width"] is not None]
        errors = [row["ape_pct"] for row in predicted if row["ape_pct"] is not None]
        by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predicted:
            by_workload[row["workload"]].append(row)
        variant_rankings = {
            workload: _partial_order(group)
            for workload, group in sorted(by_workload.items())
            if len(group) > 1
        }
        finite = [row for row in predicted if row["upper"] is not None]
        open_rows = [row for row in predicted if row["upper"] is None]
        split_reports[split] = {
            "rows": len(rows),
            "predicted": len(predicted),
            "refused": len(refused),
            "refusal_rate": len(refused) / len(rows) if rows else None,
            # Open intervals are refusal/uncertainty, not calibrated coverage.
            "interval_coverage": sum(row["covered"] for row in finite) / len(finite) if finite else None,
            "finite_interval_coverage": sum(row["covered"] for row in finite) / len(finite) if finite else None,
            "finite_interval_rows": len(finite),
            "open_interval_rows": len(open_rows),
            "open_interval_rate": len(open_rows) / len(predicted) if predicted else None,
            "mean_interval_width": statistics.mean(widths) if widths else None,
            "mean_ape_pct": statistics.mean(errors) if errors else None,
            "mean_runtime_ms": statistics.mean([row["runtime_ms"] for row in predicted if isinstance(row.get("runtime_ms"), (int, float))]) if any(isinstance(row.get("runtime_ms"), (int, float)) for row in predicted) else None,
            "mean_memory_bytes": statistics.mean([row["memory_bytes"] for row in predicted if isinstance(row.get("memory_bytes"), (int, float))]) if any(isinstance(row.get("memory_bytes"), (int, float)) for row in predicted) else None,
            "mean_step_reduction": statistics.mean([row["step_reduction"] for row in predicted if isinstance(row.get("step_reduction"), (int, float))]) if any(isinstance(row.get("step_reduction"), (int, float)) for row in predicted) else None,
            "variant_rankings": variant_rankings,
        }
    return {
        "schema": "reification_prediction_evaluation_v2",
        "target_join": "independent_ncu_after_prediction",
        "target_contract": prediction_contract,
        "split_reports": split_reports,
        "joined_rows": joined,
        "selection_used_observation_error": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("observations", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_prediction_splits(
        json.loads(args.predictions.read_text()),
        json.loads(args.observations.read_text()),
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
