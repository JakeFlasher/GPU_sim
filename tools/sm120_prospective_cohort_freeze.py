#!/usr/bin/env python3
"""Freeze prospective cohort roles before any model/NCU scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    timing = json.loads(args.timing.read_text(encoding="utf-8"))
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    ncu = json.loads(args.ncu.read_text(encoding="utf-8"))
    rows = []
    timing_by_workload = {row.get("workload"): row for row in timing.get("rows", [])}
    capture_by_workload = {row.get("workload"): row for row in capture.get("rows", [])}
    ncu_by_workload = {row.get("workload"): row for row in ncu.get("rows", [])}
    for candidate in inventory.get("candidates", []):
        workload = candidate.get("workload")
        rows.append(
            {
                "family": candidate.get("family"),
                "workload": workload,
                "role": "evaluation_only",
                "seen_by_model_before_freeze": False,
                "timing_record": timing_by_workload.get(workload, {}),
                "capture_record": capture_by_workload.get(workload, {}),
                "ncu_record": ncu_by_workload.get(workload, {}),
                "scoring_target_available": bool(ncu_by_workload.get(workload, {}).get("raw_csv_present")),
            }
        )
    result = {
        "schema_version": 1,
        "model": "sm120_prospective_evaluation_cohort_v1",
        "status": "frozen_before_scoring",
        "development_corpus_excluded": True,
        "retrospective_relabeling": False,
        "candidate_count": len(rows),
        "rows": rows,
        "scoring_rule": "predictions are generated before reading prospective NCU rows; unavailable targets remain unavailable",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(rows), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
