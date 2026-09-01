#!/usr/bin/env python3
"""Replay the independent six-row cohort and score the single-kernel timing crosswalk."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes.sm86_ncu_demo_benchmark import (
    analyze_l2_lookup_calibration as common,
)
from workloads.research_probes.sm86_ncu_demo_benchmark import validate_spec


CalibrationError = common.CalibrationError
NCU_SM_ACTIVE = "sm__cycles_active.max"
NCU_GPC_ELAPSED = "gpc__cycles_elapsed.max"
NCU_DURATION = "gpu__time_duration.sum"
NCU_INSTRUCTIONS = "smsp__inst_executed.sum"


def _ncu_median(raw_root: Path, workload_id: str, metric: str) -> tuple[float, list[float]]:
    values: list[float] = []
    repeat = 1
    while True:
        path = raw_root / "ncu" / f"rep-{repeat}" / workload_id / "normalized.json"
        if not path.is_file():
            break
        record = common._load(path)["metrics"][metric]
        if record.get("status") != "measured":
            raise CalibrationError(f"{workload_id}: NCU metric unavailable: {metric}")
        value = float(record["value"])
        if not math.isfinite(value) or value <= 0:
            raise CalibrationError(f"{workload_id}: NCU metric invalid: {metric}")
        values.append(value)
        repeat += 1
    if len(values) < 3:
        raise CalibrationError(f"{workload_id}: at least three NCU repeats are required")
    return float(statistics.median(values)), values


def _ape(predicted: float, observed: float) -> float:
    return abs(predicted - observed) / observed * 100.0


def _timing(document: Mapping[str, Any]) -> tuple[float, float, float]:
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise CalibrationError("simulator output must contain exactly one launch")
    timing = launches[0].get("timing")
    counters = launches[0].get("counters")
    if not isinstance(timing, Mapping) or not isinstance(counters, Mapping):
        raise CalibrationError("simulator timing or counters are missing")
    issue = float(timing["diagnostic_last_issue_cycles"])
    terminal = float(timing["terminal_modeled_event_time"])
    instructions = float(counters["instructions_issued"])
    if min(issue, terminal, instructions) <= 0:
        raise CalibrationError("simulator timing values must be positive")
    return issue, terminal, instructions


def build_report(
    campaign_root: Path,
    predecessor_root: Path,
    output_root: Path,
    simulator: Path,
    crosswalk_path: Path,
    repeats: int,
) -> dict[str, Any]:
    campaign = common._load(campaign_root / "campaign_manifest.json")
    spec_record = campaign.get("inputs", {}).get("spec")
    metrics_record = campaign.get("inputs", {}).get("metrics")
    if not isinstance(spec_record, Mapping) or not isinstance(metrics_record, Mapping):
        raise CalibrationError("campaign input identities are missing")
    spec_path = REPO_ROOT / str(spec_record["path"])
    metrics_path = REPO_ROOT / str(metrics_record["path"])
    if (
        common._sha256(spec_path) != spec_record["sha256"]
        or common._sha256(metrics_path) != metrics_record["sha256"]
    ):
        raise CalibrationError("campaign input identity drift")
    spec = common._load(spec_path)
    metrics = common._load(metrics_path)
    profile = common._load(HERE / "host_profile.json")
    validate_spec.validate_spec(spec, metrics, profile)
    crosswalk = common._load(crosswalk_path)
    validate_spec.validate_timing_crosswalk(crosswalk)
    active_envelope = float(crosswalk["sm_active_projection"]["fixed_envelope_cycles"])
    profiler_envelope = float(
        crosswalk["gpc_elapsed_projection"]["profiler_envelope_cycles"]
    )
    sm_clock_mhz = float(crosswalk["calibration"]["gpu"]["sm_clock_mhz"])

    raw_root = campaign_root / "evidence" / "raw"
    rows: list[dict[str, Any]] = []
    for workload in spec["workloads"]:
        workload_id = str(workload["id"])
        predecessor = common._load(
            predecessor_root / "simulator" / workload_id / "replay-1.json"
        )
        candidate, walls, artifacts = common._run_one(
            simulator,
            raw_root,
            workload_id,
            repeats,
            output_root / "simulator" / workload_id,
        )
        if candidate != predecessor:
            raise CalibrationError(
                f"{workload_id}: timing crosswalk changed simulator output"
            )
        ncu_sm, ncu_sm_repeats = _ncu_median(raw_root, workload_id, NCU_SM_ACTIVE)
        ncu_gpc, ncu_gpc_repeats = _ncu_median(raw_root, workload_id, NCU_GPC_ELAPSED)
        ncu_duration, ncu_duration_repeats = _ncu_median(
            raw_root, workload_id, NCU_DURATION
        )
        ncu_instructions, ncu_instruction_repeats = _ncu_median(
            raw_root, workload_id, NCU_INSTRUCTIONS
        )
        raw_issue, raw_terminal, simulator_instructions = _timing(candidate)
        if simulator_instructions != ncu_instructions:
            raise CalibrationError(f"{workload_id}: instruction conservation differs")
        projected_sm = raw_issue + active_envelope
        projected_gpc = raw_terminal + active_envelope + profiler_envelope
        projected_ns = projected_gpc * 1000.0 / sm_clock_mhz
        rows.append(
            {
                "workload_id": workload_id,
                "ncu": {
                    "sm_active_cycles": ncu_sm,
                    "sm_active_repeat_values": ncu_sm_repeats,
                    "gpc_elapsed_cycles": ncu_gpc,
                    "gpc_elapsed_repeat_values": ncu_gpc_repeats,
                    "duration_ns": ncu_duration,
                    "duration_repeat_values": ncu_duration_repeats,
                    "instructions": ncu_instructions,
                    "instruction_repeat_values": ncu_instruction_repeats,
                },
                "simulator": {
                    "raw_issue_cycles": raw_issue,
                    "projected_sm_active_cycles": projected_sm,
                    "raw_terminal_cycles": raw_terminal,
                    "projected_gpc_elapsed_cycles": projected_gpc,
                    "projected_duration_ns": projected_ns,
                    "instructions": simulator_instructions,
                },
                "ape_pct": {
                    "raw_issue": _ape(raw_issue, ncu_sm),
                    "projected_sm_active": _ape(projected_sm, ncu_sm),
                    "raw_terminal": _ape(raw_terminal, ncu_gpc),
                    "projected_gpc_elapsed": _ape(projected_gpc, ncu_gpc),
                    "projected_runtime": _ape(projected_ns, ncu_duration),
                    "projected_instruction_throughput": _ape(
                        simulator_instructions / projected_sm,
                        ncu_instructions / ncu_sm,
                    ),
                },
                "simulator_repeat_wall_seconds": walls,
                "simulator_outputs": artifacts,
                "simulator_repeats_byte_identical": True,
                "simulator_output_equal_to_predecessor": True,
            }
        )

    names = (
        "raw_issue",
        "projected_sm_active",
        "raw_terminal",
        "projected_gpc_elapsed",
        "projected_runtime",
        "projected_instruction_throughput",
    )
    summary = {
        name: {
            "coverage": len(rows),
            "mape_pct": statistics.fmean(float(row["ape_pct"][name]) for row in rows),
            "maximum_ape_pct": max(float(row["ape_pct"][name]) for row in rows),
        }
        for name in names
    }
    return {
        "schema_version": 1,
        "milestone": "sm86_single_kernel_timing_crosswalk_calibration",
        "status": "calibration_only_not_gate",
        "source_campaign": {
            "path": common._portable(campaign_root / "campaign_manifest.json"),
            "sha256": common._sha256(campaign_root / "campaign_manifest.json"),
        },
        "crosswalk": {
            "path": common._portable(crosswalk_path),
            "sha256": common._sha256(crosswalk_path),
            "document": crosswalk,
        },
        "simulator": {
            "path": common._portable(simulator),
            "sha256": common._sha256(simulator),
            "repeats_per_workload": repeats,
        },
        "scientific_boundary": {
            "scored_gate_rows_used": False,
            "simulator_output_changed": False,
            "raw_frontiers_preserved": True,
            "multi_launch_ranges_supported": False,
        },
        "rows": rows,
        "summary": summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=HERE / "timing_crosswalk.json")
    parser.add_argument("--simulator", type=Path, default=Path("sim/target/release/sim-run"))
    parser.add_argument("--simulator-repeats", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not 3 <= args.simulator_repeats <= 16:
            raise CalibrationError("--simulator-repeats must be within 3..16")
        report = build_report(
            args.campaign.resolve(),
            args.predecessor.resolve(),
            args.output.resolve(),
            args.simulator.resolve(),
            args.crosswalk.resolve(),
            args.simulator_repeats,
        )
        common._write(args.output.resolve() / "report.json", report)
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    except (
        CalibrationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"timing crosswalk calibration: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
