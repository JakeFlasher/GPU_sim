#!/usr/bin/env python3
"""Reduce the task-done boundary probe without fitting a simulator tail."""

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
    "producer_event_ms",
    "total_event_ms",
    "copy_event_ms",
    "host_total_ns",
    "producer_to_consumer_ns",
    "producer_block_global_ns",
    "thrash_block_global_ns",
    "consumer_block_global_ns",
)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    manifest = load(root / "campaign_manifest.json")
    if manifest.get("protocol", {}).get("simulator_runs") != 0:
        raise ValueError("probe manifest unexpectedly contains simulator runs")
    if manifest.get("protocol", {}).get("ncu_values_used_for_selection") is not False:
        raise ValueError("probe manifest does not prove NCU isolation")
    rows: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        for footprint in FOOTPRINTS:
            key = f"{mode}_{footprint}MiB"
            record = next(
                (item for item in manifest.get("records", [])
                 if item.get("mode") == mode and item.get("footprint_mb") == footprint),
                None,
            )
            if record is None:
                raise ValueError(f"missing manifest record {key}")
            trials = record.get("trials")
            if not isinstance(trials, list) or len(trials) != manifest["protocol"]["trials"]:
                raise ValueError(f"{key}: trial count mismatch")
            if not all(trial.get("correct") is True for trial in trials):
                raise ValueError(f"{key}: correctness failure")
            summaries: dict[str, float] = {}
            for metric in METRICS:
                values = [float(trial[metric]) for trial in trials]
                if not all(math.isfinite(value) and value >= 0 for value in values):
                    raise ValueError(f"{key}/{metric}: non-finite value")
                summaries[metric] = median(values)
            rows[key] = {
                "mode": mode,
                "footprint_mb": footprint,
                "trials": len(trials),
                "median": summaries,
            }

    comparisons: list[dict[str, Any]] = []
    for footprint in FOOTPRINTS:
        control = rows[f"control_{footprint}MiB"]["median"]
        store = rows[f"store_{footprint}MiB"]["median"]
        device = rows[f"store_device_{footprint}MiB"]["median"]
        comparison: dict[str, Any] = {"footprint_mb": footprint}
        for metric in METRICS:
            comparison[f"store_minus_control_{metric}"] = store[metric] - control[metric]
            comparison[f"device_minus_store_{metric}"] = device[metric] - store[metric]
            comparison[f"device_minus_control_{metric}"] = device[metric] - control[metric]
        comparisons.append(comparison)

    # Infer the timer quantum from the observed globaltimer fields.  A fixed
    # physical tail would require a stable, positive store-minus-control
    # endpoint delta across the footprint domain and a matching fence relation;
    # variation of more than two timer quanta (or a sign change) is evidence
    # against that scalar interpretation.  This is descriptive only, never a
    # parameter selector.
    copy_deltas = [row["store_minus_control_copy_event_ms"] for row in comparisons]
    total_deltas = [row["store_minus_control_total_event_ms"] for row in comparisons]
    globaltimer_values: list[int] = []
    for row in rows.values():
        for trial in manifest["records"]:
            if trial.get("mode") == row["mode"] and trial.get("footprint_mb") == row["footprint_mb"]:
                globaltimer_values.extend(
                    int(item["producer_block_global_ns"])
                    for item in trial["trials"]
                )
                break
    positive_steps = [
        right - left
        for left, right in zip(sorted(set(globaltimer_values)), sorted(set(globaltimer_values))[1:])
        if right > left
    ]
    quantum_ns = positive_steps[0] if positive_steps else 0
    for step in positive_steps[1:]:
        quantum_ns = math.gcd(quantum_ns, step)
    quantum_ms = quantum_ns / 1_000_000.0
    variation_threshold_ms = max(2.0 * quantum_ms, 0.001)
    nonlinear = (
        max(copy_deltas) - min(copy_deltas) > variation_threshold_ms
        or max(total_deltas) - min(total_deltas) > variation_threshold_ms
        or (min(copy_deltas) < 0 < max(copy_deltas))
        or (min(total_deltas) < 0 < max(total_deltas))
    )
    report = {
        "schema_version": 1,
        "report_id": "sm86_task_done_boundary_analysis_v1",
        "status": "negative_no_scalar_retirement_tail" if nonlinear else "inconclusive_tail_slope",
        "manifest": {
            "path": str((root / "campaign_manifest.json").relative_to(Path.cwd())),
            "size_bytes": (root / "campaign_manifest.json").stat().st_size,
        },
        "rows": list(rows.values()),
        "comparisons": comparisons,
        "decision": {
            "fixed_tail_promoted": False,
            "ncu_values_used_for_selection": False,
            "simulator_modified": False,
            "inferred_globaltimer_quantum_ns": quantum_ns,
            "variation_threshold_ms": variation_threshold_ms,
            "reason": "endpoint deltas include dispatch, thrash, visibility, and copy-engine service and vary with footprint; they do not identify a transferable post-issue writeback constant",
        },
    }
    output_json = root / "analysis.json"
    output_md = root / "analysis.md"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SM86 task-done boundary analysis",
        "",
        f"Status: **{report['status']}**",
        "",
        "The probe compares control, ordinary STORE, and STORE+device-fence producers followed by a write-thrash kernel, a same-stream consumer, and an asynchronous host copy. It is comparison-only and does not select simulator parameters.",
        "",
        "| Footprint (MiB) | Store−control copy ms | Store−control total ms | Device−store copy ms | Device−store total ms |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['footprint_mb']} | {row['store_minus_control_copy_event_ms']:.6f} | {row['store_minus_control_total_event_ms']:.6f} | {row['device_minus_store_copy_event_ms']:.6f} | {row['device_minus_store_total_event_ms']:.6f} |"
        )
    lines += [
        "",
        "No fixed physical-retirement tail is promoted. The endpoint includes multiple causal stages; a true task-done model needs explicit writeback/coherence/visibility state and a matching independent hardware endpoint.",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "output": str(args.root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
