#!/usr/bin/env python3
"""Emit a conservative shape/occupancy interval from live NCU anchors.

This is an observation-envelope helper, not a timing oracle.  It refuses
extrapolation outside the calibrated shape range and never consumes the target
NCU measurement while constructing the interval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--shape", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        row for row in csv.DictReader(args.matrix.open())
        if row.get("role") == "independent_calibration"
    ]
    rows.sort(key=lambda row: int(row["shape"]))
    lower = [row for row in rows if int(row["shape"]) <= args.shape]
    upper = [row for row in rows if int(row["shape"]) >= args.shape]
    if not lower or not upper:
        raise SystemExit("shape lies outside independent calibration range")
    lo = lower[-1]
    hi = upper[0]
    result = {
        "schema": "sm86_shape_interval_envelope_v1",
        "target_shape": args.shape,
        "lower_anchor": lo,
        "upper_anchor": hi,
        "duration_ns_interval": [float(lo["ncu_duration_ns"]), float(hi["ncu_duration_ns"])],
        "gpc_cycles_interval": [float(lo["ncu_gpc_cycles"]), float(hi["ncu_gpc_cycles"])],
        "extrapolated": int(lo["shape"]) == int(hi["shape"]) and int(lo["shape"]) != args.shape,
        "uncertainty": "interval_between_independent_shape_anchors",
        "target_ncu_is_not_an_input": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
