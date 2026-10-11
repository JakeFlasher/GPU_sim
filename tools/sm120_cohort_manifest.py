#!/usr/bin/env python3
"""Declare development/calibration/future-evaluation workload cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    captures = sorted(
        path.name
        for path in (args.work_root / "captures").iterdir()
        if path.is_dir() and (path / "dynamic_trace.pb").is_file()
    )
    families = {}
    for name in captures:
        family = name.split("_sm120", 1)[0]
        families.setdefault(family, []).append(name)
    result = {
        "schema_version": 1,
        "model": "sm120_family_cohort_manifest_v1",
        "development": {"captures": captures, "families": sorted(families)},
        "mechanism_calibration": {"captures": [], "families": [], "status": "not_yet_collected"},
        "future_evaluation": {"captures": [], "families": [], "status": "prospective_population_required"},
        "retrospective_holdout_claim": "none",
        "shared_binary_note": "shape variants are grouped by workload family; current retained corpus remains development data",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"development_captures": len(captures), "development_families": len(families)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
