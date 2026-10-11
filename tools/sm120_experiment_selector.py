#!/usr/bin/env python3
"""Rank the smallest SM120 interventions by model disagreement per cost."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    features = json.loads(args.features.read_text(encoding="utf-8"))
    candidates = []
    for capture in features.get("captures", []):
        for launch in capture.get("launches", []):
            scenarios = launch.get("locality_scenarios") or {}
            values = {
                name: float((scenario.get("capacity_sectors", {}).get("512", {}) or {}).get("misses", 0) or 0)
                for name, scenario in scenarios.items()
            }
            if not values:
                continue
            disagreement = max(values.values()) - min(values.values())
            demand = max(1, int(launch.get("address_records", 0) or 0))
            candidates.append(
                {
                    "capture_name": capture.get("capture_name"),
                    "kernel_id": launch.get("kernel_id"),
                    "probe": "warp_cta_order_permutation",
                    "hypotheses": ["instrumented_receive_order", "reverse_warp_cta_order", "warm_second_pass"],
                    "disagreement": disagreement,
                    "estimated_cost": 1.0 + math.log2(demand),
                    "priority": disagreement / (1.0 + math.log2(demand)),
                    "observable": "logical_LRU_miss_curve_and_sector_service",
                }
            )
    candidates.sort(key=lambda row: (-row["priority"], row["capture_name"], row["kernel_id"]))
    result = {
        "schema_version": 1,
        "model": "sm120_disagreement_per_cost_selector_v1",
        "candidate_count": len(candidates),
        "ranking": candidates,
        "next_probe": candidates[0] if candidates else None,
        "target_values_consumed": False,
        "selection_semantics": "model-scenario disagreement divided by estimated collection cost; not entropy from a constant priority list",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "next_probe": result["next_probe"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
