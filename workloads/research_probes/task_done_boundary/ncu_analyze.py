#!/usr/bin/env python3
"""Reduce the task-done NCU matrix and test for a transferable tail relation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

MODES = ("control", "store", "store_device")
FOOTPRINTS = (1, 4, 8, 16, 64)
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "dram__sectors_write.sum",
    "lts__t_sectors_srcunit_tex_lookup_miss.sum",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = load(root / "campaign_manifest.json")
    if manifest["protocol"]["repeats"] < 3:
        raise ValueError("NCU matrix needs at least three repeats")
    rows: dict[tuple[str, int], dict[str, float]] = {}
    invalid: list[str] = []
    for record in manifest["records"]:
        key = (str(record["mode"]), int(record["footprint_mb"]))
        repeats = record["repeats"]
        if len(repeats) != manifest["protocol"]["repeats"]:
            raise ValueError(f"{key}: repeat count mismatch")
        medians: dict[str, float] = {}
        for metric in METRICS:
            values = [float(rep["metrics"][metric]) for rep in repeats]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                invalid.append(f"{key}/{metric}")
            medians[metric] = float(statistics.median(values))
        rows[key] = medians
    if invalid:
        raise ValueError(f"invalid NCU samples: {invalid[:5]}")

    comparisons: list[dict[str, Any]] = []
    for footprint in FOOTPRINTS:
        control = rows[("control", footprint)]
        store = rows[("store", footprint)]
        device = rows[("store_device", footprint)]
        item: dict[str, Any] = {"footprint_mb": footprint}
        for metric in METRICS:
            item[f"store_minus_control_{metric}"] = store[metric] - control[metric]
            item[f"device_minus_store_{metric}"] = device[metric] - store[metric]
        comparisons.append(item)

    # A scalar retirement tail would appear as a stable store/control difference
    # in the endpoint metrics.  Report variation and sign changes; never fit it.
    endpoint = [row["store_minus_control_gpu__time_duration.sum"] for row in comparisons]
    active = [row["store_minus_control_sm__cycles_active.max"] for row in comparisons]
    variation = max(endpoint) - min(endpoint)
    sign_change = min(endpoint) < 0 < max(endpoint)
    report = {
        "schema_version": 1,
        "report_id": "sm86_task_done_boundary_ncu_analysis_v1",
        "status": "negative_no_scalar_retirement_tail" if variation > 100 or sign_change else "inconclusive_tail_slope",
        "manifest": str((root / "campaign_manifest.json").relative_to(Path.cwd())),
        "rows": [
            {"mode": mode, "footprint_mb": footprint, "median": rows[(mode, footprint)]}
            for mode in MODES for footprint in FOOTPRINTS
        ],
        "comparisons": comparisons,
        "endpoint_variation": {
            "gpu_time_duration_store_minus_control_range_ns": variation,
            "sm_cycles_active_store_minus_control_range_cycles": max(active) - min(active),
            "gpu_time_duration_sign_change": sign_change,
        },
        "decision": {
            "fixed_tail_promoted": False,
            "simulator_modified": False,
            "ncu_values_used_for_selection": False,
            "reason": "producer endpoint deltas vary across footprint and include service/visibility effects; the matrix does not identify a transferable post-issue scalar",
        },
    }
    (root / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SM86 task-done boundary NCU analysis",
        "",
        f"Status: **{report['status']}**",
        "",
        "Seven-repeat NCU producer-kernel measurements were collected for control, STORE, and STORE+device-fence modes after a write-thrash/consumer/copy sequence. NCU values are comparison-only.",
        "",
        "| Footprint (MiB) | Store−control `gpu__time_duration` ns | Store−control `sm__cycles_active` cycles | Device−store duration ns | Device−store active cycles |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['footprint_mb']} | {row['store_minus_control_gpu__time_duration.sum']:.1f} | {row['store_minus_control_sm__cycles_active.max']:.1f} | {row['device_minus_store_gpu__time_duration.sum']:.1f} | {row['device_minus_store_sm__cycles_active.max']:.1f} |"
        )
    lines += [
        "",
        f"The store/control duration delta range is {variation:.1f} ns and sign_change={sign_change}; no fixed physical-retirement tail is promoted.",
        "",
        "The matching same-stream consumer and host-copy endpoints remain separate from the SM-active metric. A true task-done model needs explicit writeback/coherence/visibility state and an independently matching endpoint.",
    ]
    (root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root)
    print(json.dumps({"status": report["status"], "output": str(args.root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
