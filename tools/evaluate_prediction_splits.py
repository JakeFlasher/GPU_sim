#!/usr/bin/env python3
"""Join model predictions with independent NCU observations after prediction.

The prediction document is intentionally forbidden from containing observed
targets.  The join key is workload identity only; split membership is supplied
by the prediction/development manifest and is never selected using NCU error.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = {"calibration", "development", "held_out"}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def evaluate_prediction_splits(predictions: dict[str, Any], observations: dict[str, Any]) -> dict[str, Any]:
    if predictions.get("schema") != "reification_prediction_v1":
        raise ValueError("prediction schema must be reification_prediction_v1")
    if observations.get("schema") != "ncu_observation_v1":
        raise ValueError("observation schema must be ncu_observation_v1")
    prediction_rows = predictions.get("rows")
    observation_rows = observations.get("rows")
    if not isinstance(prediction_rows, list) or not isinstance(observation_rows, list):
        raise ValueError("prediction and observation rows must be arrays")
    observed_forbidden = {"ncu", "observed", "observation", "target", "target_value"}
    predictions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(prediction_rows):
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {index} must be an object")
        if observed_forbidden.intersection(row):
            raise ValueError("prediction rows must not contain observed target fields")
        key = row.get("workload")
        split = row.get("split")
        if not isinstance(key, str) or not key or split not in SPLITS:
            raise ValueError(f"prediction row {index} needs workload and split")
        intervention = row.get("intervention", "default")
        if not isinstance(intervention, str) or not intervention:
            raise ValueError(f"prediction row {index}.intervention must be nonempty")
        identity = (key, intervention)
        if identity in predictions_by_key:
            raise ValueError(f"duplicate prediction workload/intervention {identity!r}")
        predictions_by_key[identity] = row
    observations_by_key: dict[str, float] = {}
    for index, row in enumerate(observation_rows):
        if not isinstance(row, dict) or not isinstance(row.get("workload"), str):
            raise ValueError(f"observation row {index} needs workload")
        key = row["workload"]
        if key in observations_by_key:
            raise ValueError(f"duplicate observation workload {key!r}")
        observations_by_key[key] = _finite(row.get("value"), f"observation row {index}.value")

    joined = []
    for (key, intervention), row in predictions_by_key.items():
        status = row.get("status", "predicted")
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
        if key not in observations_by_key:
            raise ValueError(f"prediction {key!r} has no independent observation")
        observed = observations_by_key[key]
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
            "covered": lower <= observed and (upper is None or observed <= upper),
            "width": upper - lower if upper is not None else None,
            "ape_pct": abs(point - observed) / abs(observed) * 100 if observed != 0 else None,
            "runtime_ms": row.get("runtime_ms"),
            "memory_bytes": row.get("memory_bytes"),
            "compression_ratio": row.get("compression_ratio"),
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
            workload: [
                row["intervention"]
                for row in sorted(group, key=lambda item: (item["point"], item["intervention"]))
            ]
            for workload, group in sorted(by_workload.items())
            if len(group) > 1
        }
        split_reports[split] = {
            "rows": len(rows),
            "predicted": len(predicted),
            "refused": len(refused),
            "refusal_rate": len(refused) / len(rows) if rows else None,
            "interval_coverage": sum(row["covered"] for row in predicted) / len(predicted) if predicted else None,
            "mean_interval_width": statistics.mean(widths) if widths else None,
            "mean_ape_pct": statistics.mean(errors) if errors else None,
            "mean_runtime_ms": statistics.mean([row["runtime_ms"] for row in predicted if isinstance(row.get("runtime_ms"), (int, float))]) if any(isinstance(row.get("runtime_ms"), (int, float)) for row in predicted) else None,
            "mean_memory_bytes": statistics.mean([row["memory_bytes"] for row in predicted if isinstance(row.get("memory_bytes"), (int, float))]) if any(isinstance(row.get("memory_bytes"), (int, float)) for row in predicted) else None,
            "mean_compression_ratio": statistics.mean([row["compression_ratio"] for row in predicted if isinstance(row.get("compression_ratio"), (int, float))]) if any(isinstance(row.get("compression_ratio"), (int, float)) for row in predicted) else None,
            "variant_rankings": variant_rankings,
        }
    return {
        "schema": "reification_prediction_evaluation_v1",
        "target_join": "independent_ncu_after_prediction",
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
