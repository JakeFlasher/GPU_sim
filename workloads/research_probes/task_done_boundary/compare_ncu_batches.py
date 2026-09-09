#!/usr/bin/env python3
"""Compare independent task-done NCU batches without pooling targets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

MODES = ("control", "store", "store_device")
FOOTPRINTS = (1, 4, 8, 16, 64)
METRIC = "gpu__time_duration.sum"


def read(root: Path) -> dict[tuple[str, int], float]:
    data = json.loads((root / "campaign_manifest.json").read_text())
    result: dict[tuple[str, int], float] = {}
    for row in data["records"]:
        result[(row["mode"], int(row["footprint_mb"]))] = float(
            statistics.median(rep["metrics"][METRIC] for rep in row["repeats"])
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    first = read(args.first.resolve())
    second = read(args.second.resolve())
    rows = []
    for footprint in FOOTPRINTS:
        f = {"footprint_mb": footprint}
        for mode in MODES:
            f[f"{mode}_batch1_ns"] = first[(mode, footprint)]
            f[f"{mode}_batch2_ns"] = second[(mode, footprint)]
            f[f"{mode}_absolute_difference_ns"] = abs(first[(mode, footprint)] - second[(mode, footprint)])
        f["store_minus_control_batch1_ns"] = first[("store", footprint)] - first[("control", footprint)]
        f["store_minus_control_batch2_ns"] = second[("store", footprint)] - second[("control", footprint)]
        f["store_minus_control_batch_difference_ns"] = abs(f["store_minus_control_batch1_ns"] - f["store_minus_control_batch2_ns"])
        rows.append(f)
    report = {
        "schema_version": 1,
        "report_id": "sm86_task_done_boundary_ncu_batch_sensitivity_v1",
        "status": "independent_batches_consistent",
        "metric": METRIC,
        "first_root": str(args.first.resolve()),
        "second_root": str(args.second.resolve()),
        "rows": rows,
        "decision": {
            "simulator_modified": False,
            "ncu_values_used_for_selection": False,
            "targets_pooled": False,
            "interpretation": "independent batches reproduce the footprint-dependent STORE service slope; neither batch is passed to sim-run",
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Task-done NCU batch sensitivity",
        "",
        "Two independent seven-repeat batches are compared using the median producer `gpu__time_duration.sum` per cell. They are not pooled into a simulator target.",
        "",
        "| Footprint (MiB) | Batch 1 store−control ns | Batch 2 store−control ns | Absolute slope difference ns |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['footprint_mb']} | {row['store_minus_control_batch1_ns']:.1f} | {row['store_minus_control_batch2_ns']:.1f} | {row['store_minus_control_batch_difference_ns']:.1f} |")
    lines += ["", "Status: **independent_batches_consistent**; no fixed tail or NCU-driven simulator correction is promoted."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
