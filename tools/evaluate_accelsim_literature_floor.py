#!/usr/bin/env python3
"""Evaluate local evidence against the registered Accel-Sim target manifest.

The tool deliberately distinguishes a registered literature target from a
score. It refuses to mark attainment when the local table is incomplete,
clock-invalid, or uses a different observable/endpoint. It never feeds target
values into the simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA = "accelsim_literature_floor_evaluation_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _holdout_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def evaluate(manifest_path: Path, holdout_path: Path | None) -> dict[str, Any]:
    manifest = _load(manifest_path)
    rows = _holdout_rows(holdout_path) if holdout_path else []
    records: list[dict[str, Any]] = []
    for target in manifest["records"]:
        result: dict[str, Any] = {
            "target_id": target["id"],
            "metric_kind": target["metric_kind"],
            "threshold_type": target["threshold_type"],
            "threshold": target["accuracy_floor"]
            if target["metric_kind"] == "accuracy"
            else target["speed_floor"],
            "comparison_status": target["comparison_status"],
            "attainment_status": "not_evaluable_unmatched",
            "ground_truth": False,
        }
        # The retained local CSV is intentionally marked incomplete/unlocked in
        # the repository ledger. Its frontier is not an admitted NCU elapsed
        # endpoint, so refusing here is the scientifically correct result.
        if target["metric_kind"] == "accuracy" and rows:
            eligible = [
                row
                for row in rows
                if row.get("completion_status") == "complete"
                and row.get("gpc_abs_error_pct") not in {None, ""}
            ]
            if eligible:
                result["local_observation"] = {
                    "source": str(holdout_path),
                    "rows": len(eligible),
                    "mean_error_pct": mean(float(row["gpc_abs_error_pct"]) for row in eligible),
                    "status": "diagnostic_only_unadmitted_clock_or_endpoint",
                }
                result["attainment_status"] = "not_evaluable_unadmitted_local_protocol"
        records.append(result)
    return {
        "schema": SCHEMA,
        "status": "ok",
        "scoring": False,
        "ground_truth_lane": "local_ncu",
        "manifest": str(manifest_path),
        "local_holdout": None if holdout_path is None else str(holdout_path),
        "records": records,
        "scientific_boundary": (
            "Published aggregates register accuracy ceilings and throughput/speed floors; "
            "they are not per-kernel labels. Current development data is not admitted "
            "for attainment until clocks, endpoint, workload mapping, and tail semantics pass."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.manifest, args.holdout)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
