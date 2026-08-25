#!/usr/bin/env python3
"""Validate and reduce the sm_86 branch-free IBuffer timeline campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_front_end_ibuffer_depth_20260818_v1"
FUNCTION = "_Z22ibuffer_timeline_probePyPj"
EXPANDED_WIDTHS = {4, 5, 8, 12}
TRANSFER_VARIANTS = ("baseline_s1", "stall4_s1_i1", "yield_s1_i1")
REQUIRED_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__warps_issue_stalled_no_instruction.sum",
    "smsp__warps_issue_stalled_branch_resolving.sum",
    "smsp__sass_branch_targets_threads_uniform.sum",
    "smsp__sass_branch_targets_threads_divergent.sum",
)


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("ibuffer_order_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_order_analysis()
require = ORDER.require
read_json = ORDER.read_json
write_json = ORDER.write_json
wide_ncu = ORDER.wide_ncu
measurement = ORDER.measurement
source_summary = ORDER.source_summary
expected_physical_slots = ORDER.expected_physical_slots


def validate_static(root: Path) -> dict[str, Any]:
    names = ["front_end_ibuffer_depth_native"]
    names.extend(
        ["front_end_ibuffer_depth_baseline_s1"]
        + [f"front_end_ibuffer_depth_stall4_s1_i{index}" for index in range(8)]
        + [f"front_end_ibuffer_depth_yield_s1_i{index}" for index in range(8)]
    )
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        require(sidecar["summary"]["total_decoded"] == 72, f"{name}: decoded controls")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown controls")
        require(set(sidecar["functions"]) == {FUNCTION}, f"{name}: function")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require("REG:28" in resources, f"{name}: registers")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spills")
    patch = read_json(root / "metadata/front_end_ibuffer_depth.patch.json")
    require(patch["measured_instructions"] == 32, "measured instruction count")
    require(patch["measured_pc_first"] == 0x20 and patch["measured_pc_last"] == 0x210, "measured PCs")
    require(patch["no_measured_branch"] is True, "branch-free treatment")
    for base_stall in (0, 1):
        baseline = f"baseline_s{base_stall}"
        for kind in ("stall4", "yield"):
            for index in range(8):
                key = f"{baseline}_to_{kind}_s{base_stall}_i{index}"
                require(patch["pair_differences"][key] == 1, f"{key}: one-byte control change")
    projection = read_json(root / "metadata/operand_projection_audit.json")
    require(
        (projection["files"], projection["records"], projection["projected"], projection["gaps"])
        == (18, 1296, 54, 0),
        "local operand projection",
    )
    corpus = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (153, 48784, 18426, 0),
        "retained operand projection",
    )
    return {
        "binaries": 18,
        "decoded_instructions": 1296,
        "registers": 28,
        "spill_free": True,
        "local_projection": {
            key: projection[key]
            for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
        "retained_projection": {
            key: corpus[key]
            for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
    }


def median_timeline(payload: dict[str, Any]) -> list[list[float]]:
    timelines = payload["timeline_cycles"]
    return [
        [
            statistics.median(trial[warp][index] for trial in timelines)
            for index in range(9)
        ]
        for warp in range(payload["warps"])
    ]


def measured_span(timeline: list[list[float]]) -> float:
    return max(row[index] for row in timeline for index in range(8)) - min(
        row[index] for row in timeline for index in range(8)
    )


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata/matrix.json")["cells"]
    require(len(cells) == 32 and len({entry["key"] for entry in cells}) == 32, "matrix")
    calibration = read_json(root / "calibration/summary.json")
    require(calibration["status"] == "literal_stall_zero_rejected_from_main_matrix", "stall-zero calibration")
    static = validate_static(root)

    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live = {}
    timelines = {}
    replay = {}
    trace_records = 0
    ncu_profiles = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    maximum_timeline_range = 0
    for entry in cells:
        key = entry["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True and payload["warps"] == entry["warps"], f"{key}: correctness")
        require(payload["instructions"] == 32 and payload["timestamps_per_warp"] == 9, f"{key}: timeline shape")
        require(len(payload["timeline_cycles"]) == 7, f"{key}: trials")
        maximum_timeline_range = max(
            maximum_timeline_range,
            max(
                max(trial[warp][index] for trial in payload["timeline_cycles"])
                - min(trial[warp][index] for trial in payload["timeline_cycles"])
                for warp in range(entry["warps"])
                for index in range(9)
            ),
        )
        timelines[key] = median_timeline(payload)

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 7 if entry["warps"] in EXPANDED_WIDTHS else 3
        require(len(repeat_paths) == expected_repeats, f"{key}: NCU repeats")
        ncu_profiles += len(repeat_paths)
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in REQUIRED_METRICS:
            require(all(metric in row for row in rows), f"{key}: {metric}")
        require(all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows), f"{key}: issue ratio")
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        live[key] = {
            "active_cycles": active_median,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            **{
                metric: statistics.median(row[metric] for row in rows)
                for metric in REQUIRED_METRICS[1:]
            },
        }

        capture = root / "nvbit" / key
        traced = measurement(capture / "stdout.txt")
        require(traced["correct"] is True and traced["warps"] == entry["warps"], f"{key}: trace result")
        for diagnostic in ("unsupported_memory_path", "unmapped_memory_range", "cross_allocation_lane_span"):
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: block")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {
            int(warp.instructions[0].warpid_sm)
            for warp in block.warps.values()
            if warp.instructions
        }
        require(slots == expected_physical_slots(entry["warps"]), f"{key}: physical slots")
        trace_records += records

        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: replay determinism")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: records")
        replay[key] = launch

    require(ncu_profiles == 144, "NCU profile count")
    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 7, "SourceCounters count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}

    width_rows = []
    for warps in (1, 4, 5, 8, 12, 16):
        baseline = f"w{warps}_baseline_s1"
        baseline_span = measured_span(timelines[baseline])
        row = {"warps": warps, "baseline_span": baseline_span}
        for treatment, variant in (("stall4", "stall4_s1_i1"), ("yield", "yield_s1_i1")):
            key = f"w{warps}_{variant}"
            row[f"{treatment}_span"] = measured_span(timelines[key])
            row[f"{treatment}_span_delta"] = measured_span(timelines[key]) - baseline_span
            row[f"{treatment}_active_delta"] = live[key]["active_cycles"] - live[baseline]["active_cycles"]
            row[f"{treatment}_no_instruction_delta"] = (
                live[key]["smsp__warps_issue_stalled_no_instruction.sum"]
                - live[baseline]["smsp__warps_issue_stalled_no_instruction.sum"]
            )
            row[f"{treatment}_wait_delta"] = (
                live[key]["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"]
                - live[baseline]["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"]
            )
            row[f"{treatment}_replay_delta"] = (
                replay[key]["timing"]["diagnostic_last_issue_cycles"]
                - replay[baseline]["timing"]["diagnostic_last_issue_cycles"]
            )
        width_rows.append(row)

    index_rows = []
    baseline_span = measured_span(timelines["w16_baseline_s1"])
    for kind in ("stall4", "yield"):
        for index in range(8):
            key = f"w16_{kind}_s1_i{index}"
            index_rows.append(
                {
                    "treatment": kind,
                    "index": index,
                    "opcode": "CS2R" if index in {0, 4} else "NOP",
                    "span": measured_span(timelines[key]),
                    "span_delta": measured_span(timelines[key]) - baseline_span,
                    "active_delta": live[key]["active_cycles"] - live["w16_baseline_s1"]["active_cycles"],
                    "no_instruction_delta": (
                        live[key]["smsp__warps_issue_stalled_no_instruction.sum"]
                        - live["w16_baseline_s1"]["smsp__warps_issue_stalled_no_instruction.sum"]
                    ),
                }
            )

    candidates = read_json(root / "metadata/front_end_candidates.json")
    decision = candidates["decision"]
    require(decision["best_capacity"] == 2 and decision["absolute_best"]["capacity"] == 2, "capacity-two candidate")
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 32,
            "correctness_executions": 224,
            "ncu_profiles": ncu_profiles,
            "source_counters_profiles": 7,
            "nvbit_captures": 32,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 32,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "maximum_timeline_range_cycles": maximum_timeline_range,
            "all_issue_ratios_one": True,
        },
        "static": static,
        "calibration": {
            "literal_stall_zero": "hardware_correct_but_nvdisasm_invalid_and_pathologically_slow",
            "legal_baseline": "stall_one_yield_zero",
        },
        "width_transfer": width_rows,
        "index_sweep": index_rows,
        "source_counters": sources,
        "candidate": {
            "tested": len(candidates["candidates"]),
            "treatment_best": {
                key: decision[key]
                for key in (
                    "best_capacity",
                    "best_instruction_miss_latency",
                    "best_prefetch_depth",
                    "best_treatment_delta_mae",
                    "best_absolute_timeline_mae",
                    "best_event_order_accuracy",
                )
            },
            "absolute_best": decision["absolute_best"],
            "reference_capacity3_prefetch8": decision["reference_capacity3_prefetch8"],
        },
        "decision": {
            "status": "capacity_two_survives_but_ibuffer_front_end_candidate_rejected",
            "activate_ibuffer_capacity": False,
            "evidence": [
                "all W16 index treatments and width-transfer timelines select capacity two under in-flight reservation semantics",
                "capacity two also provides the lowest absolute timeline MAE",
                "the treatment-best and absolute-best candidates require incompatible instruction-cache parameters",
                "exact CLOCK event-order accuracy remains below fifteen percent",
                "the capacity-three/prefetch-eight reference shape is worse on both treatment and event-order metrics",
            ],
            "next_slice": "returning atomic dependency completion and same-line serialization",
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    candidate = summary["candidate"]
    lines = [
        "# sm_86 branch-free IBuffer timeline",
        "",
        "## Decision",
        "",
        "Capacity two is the only surviving IBuffer candidate, but the complete",
        "front-end state is rejected. Both treatment-delta and absolute CLOCK",
        "objectives choose capacity two once in-flight fetches reserve entries.",
        "However, their instruction-cache parameters disagree, exact event-order",
        "accuracy remains below 15%, and the paper's capacity-3/prefetch-8 shape",
        "does not transfer. No IBuffer capacity enters production replay.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells and {campaign['correctness_executions']} correctness executions",
        f"- {campaign['ncu_profiles']} NCU profiles, {campaign['source_counters_profiles']} SourceCounters profiles, and {campaign['nvbit_captures']} NVBit captures",
        f"- {campaign['trace_records']:,} dynamic records and {campaign['byte_identical_replay_pairs']} byte-identical replay pairs",
        f"- maximum active-cycle CV {campaign['maximum_active_cycle_cv_pct']:.3f}%, MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%, and CLOCK range across repeats {campaign['maximum_timeline_range_cycles']} cycles",
        "- 18 spill-free 28-register cubins, 1,296 decoded controls, 54 projections, zero gaps",
        "- retained corpus: 153 sidecars, 48,784 records, 18,426 projections, zero gaps",
        "- every NCU profile has issue ratio 1.0",
        "",
        "## Width transfer: index-1 treatment",
        "",
        "| W | Baseline span | Stall span Δ | Yield span Δ | Stall active Δ | Yield active Δ | Stall no-inst Δ | Yield no-inst Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["width_transfer"]:
        lines.append(
            f"| {row['warps']} | {row['baseline_span']:.0f} | {row['stall4_span_delta']:+.0f} | "
            f"{row['yield_span_delta']:+.0f} | {row['stall4_active_delta']:+.0f} | "
            f"{row['yield_active_delta']:+.0f} | {row['stall4_no_instruction_delta']:+.0f} | "
            f"{row['yield_no_instruction_delta']:+.0f} |"
        )
    lines.extend(
        [
            "",
            "## W16 control-index sweep",
            "",
            "| Treatment | Index | Opcode | CLOCK span Δ | NCU-active Δ | No-instruction Δ |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["index_sweep"]:
        lines.append(
            f"| {row['treatment']} | {row['index']} | {row['opcode']} | {row['span_delta']:+.0f} | "
            f"{row['active_delta']:+.0f} | {row['no_instruction_delta']:+.0f} |"
        )
    treatment = candidate["treatment_best"]
    absolute = candidate["absolute_best"]
    reference = candidate["reference_capacity3_prefetch8"]
    lines.extend(
        [
            "",
            "## Candidate boundary",
            "",
            f"The 280-candidate sweep spans capacities 1–8, 4–40-cycle instruction-line misses, and prefetch depths 0/1/2/4/8. The treatment optimum is capacity {treatment['best_capacity']}, miss {treatment['best_instruction_miss_latency']}, prefetch {treatment['best_prefetch_depth']}, with delta MAE {treatment['best_treatment_delta_mae']:.3f} cycles and event-order accuracy {treatment['best_event_order_accuracy'] * 100:.1f}%.",
            f"The absolute optimum is also capacity {absolute['capacity']} but instead needs miss {absolute['instruction_miss_latency']} and prefetch {absolute['prefetch_depth']}; its absolute MAE is {absolute['absolute_timeline_mae']:.3f} cycles and delta MAE {absolute['treatment_delta_mae']:.3f}.",
            f"The capacity-3/prefetch-8 reference reaches absolute MAE {reference['absolute_timeline_mae']:.3f}, delta MAE {reference['treatment_delta_mae']:.3f}, and only {reference['event_order_accuracy'] * 100:.1f}% exact event order.",
            "",
            "The incompatible cache fits and poor event order prevent promotion;",
            "selecting one cache parameter set solely because it preserves capacity",
            "two would be overfitting this microprobe.",
            "",
            "## Next slice",
            "",
            "Advance to returning-atomic dependency completion and same-line",
            "serialization. Retain capacity two as a front-end candidate for future",
            "instruction-fetch observability, not as active architecture state.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = reduce(args.root.resolve())
        analysis = args.root.resolve() / "analysis"
        write_json(analysis / "summary.json", summary)
        (analysis / "README.md").write_text(render(summary), encoding="utf-8")
        print(analysis / "README.md")
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"IBuffer timeline analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
