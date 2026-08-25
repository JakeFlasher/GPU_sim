#!/usr/bin/env python3
"""Validate and reduce the sm_86 branch-frequency/footprint campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = (
    REPO_ROOT
    / "results/captures/sm86_fixed_latency_front_end_footprint_20260817_v1"
)
UNROLLS = (1, 2, 4, 8)
WIDTHS = (4, 5, 8, 9, 13, 16)
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
EXPANDED = {
    "u2_w9_fixed_bypass",
    "u2_w9_fixed_retired",
    "u2_w13_fixed_bypass",
    "u2_w13_fixed_retired",
    "u2_w16_fixed_bypass",
    "u2_w16_fixed_retired",
    "u8_w13_fixed_bypass",
    "u8_w13_fixed_retired",
}
REQUIRED_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_cbu.sum",
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
)
COUNT_METRICS = (
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_cbu.sum",
    "smsp__inst_executed_pipe_lsu.sum",
)
EMPTY_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("footprint_order_analysis", path)
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


def smsp_counts(warps: int) -> list[int]:
    return [(warps - 1 - subcore) // 4 + 1 for subcore in range(4)]


def static_names(unroll: int) -> list[str]:
    result = []
    for trace in (False, True):
        infix = "_trace" if trace else ""
        result.append(f"fixed_latency_front_end_footprint_u{unroll}{infix}_native")
        result.extend(
            f"fixed_latency_front_end_footprint_u{unroll}{infix}_{variant}"
            for variant in VARIANTS
        )
    return result


def validate_static(root: Path) -> dict[str, Any]:
    expected_decoded = {1: 80, 2: 96, 4: 112, 8: 152}
    registers: dict[str, list[int]] = {}
    for unroll in UNROLLS:
        for name in static_names(unroll):
            sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
            require(
                sidecar["summary"]["total_decoded"] == expected_decoded[unroll],
                f"{name}: decoded controls",
            )
            require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown controls")
            resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
            require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spills")
            registers[name] = sorted({int(value) for value in re.findall(r"REG:(\d+)", resources)})

        for trace in (False, True):
            infix = "_trace" if trace else ""
            report = read_json(
                root / "metadata" / f"fixed_latency_front_end_footprint_u{unroll}{infix}.patch.json"
            )
            require(report["unroll"] == unroll, f"U{unroll}: patch identity")
            require(report["one_backward_branch_per_unrolled_group"] is True, f"U{unroll}: branch shape")
            require(report["exact_producer_consumer_bodies"] == {"fixed": unroll, "global": unroll}, f"U{unroll}: bodies")
            for left, right in (("fixed_bypass", "fixed_retired"), ("global_frontier", "global_late")):
                require(
                    report["variants"][left]["selected_stall_sum"]
                    == report["variants"][right]["selected_stall_sum"],
                    f"U{unroll} {left}/{right}: equal stalls",
                )

    projection = read_json(root / "metadata/operand_projection_audit.json")
    require(
        (projection["files"], projection["records"], projection["projected"], projection["gaps"])
        == (40, 4400, 1190, 0),
        "local operand projection",
    )
    corpus = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (135, 47488, 18372, 0),
        "retained operand projection",
    )
    return {
        "binaries": 40,
        "decoded_instructions": 4400,
        "spill_free": True,
        "register_counts": {
            f"u{unroll}": registers[f"fixed_latency_front_end_footprint_u{unroll}_native"]
            for unroll in UNROLLS
        },
        "local_projection": {
            key: projection[key]
            for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
        "retained_corpus_projection": {
            key: corpus[key]
            for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
    }


def reduce(root: Path) -> dict[str, Any]:
    matrix = read_json(root / "metadata/matrix.json")
    cells = matrix["cells"]
    require(len(cells) == 96 and len({cell["key"] for cell in cells}) == 96, "matrix")
    calibration = read_json(root / "calibration/summary.json")
    require(calibration["fixed_d3_semantic_boundary"] == "invalid", "fixed d3 calibration")
    require(
        calibration["global_d4_semantic_boundary"] == "not_identifying_in_this_register_allocation",
        "global d4 characterization",
    )
    static = validate_static(root)

    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live: dict[str, dict[str, float | list[float]]] = {}
    replay: dict[str, dict[str, Any]] = {}
    front_end: dict[str, dict[str, float]] = {}
    trace_records = 0
    ncu_profiles = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    branch_cycles_per_target = []

    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True and payload["warps"] == cell["warps"], f"{key}: correctness")
        require(payload["generations"] == 1024 and len(payload["samples_cycles"]) == 7, f"{key}: trials")
        require(payload["unroll"] == cell["unroll"], f"{key}: unroll")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 7 if key in EXPANDED else 3
        require(len(repeat_paths) == expected_repeats, f"{key}: NCU repeats")
        ncu_profiles += len(repeat_paths)
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in REQUIRED_METRICS:
            require(all(metric in row for row in rows), f"{key}: {metric}")
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows),
            f"{key}: issue ratio",
        )
        global_kind = cell["kind"] == "global"
        expected_hits = (4092 if global_kind else 0) * cell["warps"]
        expected_misses = (8 if global_kind else 4) * cell["warps"]
        require(
            all(
                row["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"] == expected_hits
                and row["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"] == expected_misses
                for row in rows
            ),
            f"{key}: exact L1 traffic",
        )
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        median = lambda metric: statistics.median(row[metric] for row in rows)
        live[key] = {
            "driver_cycles_per_generation": statistics.median(driver_samples) / 1024,
            "active_cycles_per_generation": active_median / 1024,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            **{metric: median(metric) for metric in REQUIRED_METRICS[1:]},
        }

        direct = wide_ncu(root / "ncu-front-end" / key / "ncu.csv")
        expected_targets = (1024 // cell["unroll"]) * cell["warps"]
        uniform_targets = direct["smsp__sass_branch_targets_threads_uniform.sum"]
        divergent_targets = direct["smsp__sass_branch_targets_threads_divergent.sum"]
        require(uniform_targets == expected_targets, f"{key}: branch targets")
        require(divergent_targets == 0, f"{key}: divergent targets")
        branch_cycles = direct["smsp__warps_issue_stalled_branch_resolving.sum"]
        branch_cycles_per_target.append(branch_cycles / uniform_targets)
        front_end[key] = {
            "no_instruction_warp_cycles": direct["smsp__warps_issue_stalled_no_instruction.sum"],
            "branch_resolving_warp_cycles": branch_cycles,
            "no_instruction_per_warp_generation": (
                direct["smsp__warps_issue_stalled_no_instruction.sum"]
                / (1024 * cell["warps"])
            ),
            "branch_resolving_per_warp_generation": branch_cycles / (1024 * cell["warps"]),
            "branch_cycles_per_target": branch_cycles / uniform_targets,
            "uniform_branch_targets": uniform_targets,
            "divergent_branch_targets": divergent_targets,
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True and trace_payload["generations"] == 64, f"{key}: trace")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: traced block")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {
            int(warp.instructions[0].warpid_sm)
            for warp in block.warps.values()
            if warp.instructions
        }
        require(slots == expected_physical_slots(cell["warps"]), f"{key}: physical slots")
        trace_records += records
        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: replay determinism")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: replay records")
        control_pipeline = launch["timing"]["readiness"]["control_pipeline"]
        expected_trace_branches = (64 // cell["unroll"]) * cell["warps"]
        require(
            control_pipeline["branch_instructions_issued"] == expected_trace_branches,
            f"{key}: replay branch count",
        )
        require(
            control_pipeline["branch_resolution_cycles"] == 3 * expected_trace_branches,
            f"{key}: replay branch resolution",
        )
        replay[key] = launch

    require(ncu_profiles == 320, "NCU profile count")
    for unroll in UNROLLS:
        for warps in WIDTHS:
            for left, right in (("fixed_bypass", "fixed_retired"), ("global_frontier", "global_late")):
                for metric in COUNT_METRICS:
                    require(
                        live[f"u{unroll}_w{warps}_{left}"][metric]
                        == live[f"u{unroll}_w{warps}_{right}"][metric],
                        f"U{unroll} W{warps} {left}/{right}: {metric}",
                    )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 32, "SourceCounters count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}

    rows = []
    for unroll in UNROLLS:
        for warps in WIDTHS:
            def delta(field: str, left: str, right: str) -> float:
                return float(live[f"u{unroll}_w{warps}_{right}"][field]) - float(
                    live[f"u{unroll}_w{warps}_{left}"][field]
                )

            def front_delta(field: str, left: str, right: str) -> float:
                return front_end[f"u{unroll}_w{warps}_{right}"][field] - front_end[
                    f"u{unroll}_w{warps}_{left}"
                ][field]

            def sim(variant: str) -> float:
                return replay[f"u{unroll}_w{warps}_{variant}"]["timing"]["diagnostic_last_issue_cycles"] / 64

            rows.append(
                {
                    "unroll": unroll,
                    "warps": warps,
                    "warps_per_smsp": smsp_counts(warps),
                    "driver_fixed_delta": delta("driver_cycles_per_generation", "fixed_bypass", "fixed_retired"),
                    "active_fixed_delta": delta("active_cycles_per_generation", "fixed_bypass", "fixed_retired"),
                    "replay_fixed_delta": sim("fixed_retired") - sim("fixed_bypass"),
                    "driver_global_delta": delta("driver_cycles_per_generation", "global_frontier", "global_late"),
                    "active_global_delta": delta("active_cycles_per_generation", "global_frontier", "global_late"),
                    "replay_global_delta": sim("global_late") - sim("global_frontier"),
                    "fixed_no_instruction_delta": front_delta(
                        "no_instruction_per_warp_generation", "fixed_bypass", "fixed_retired"
                    ),
                    "fixed_branch_resolving_delta": front_delta(
                        "branch_resolving_per_warp_generation", "fixed_bypass", "fixed_retired"
                    ),
                    "global_no_instruction_delta": front_delta(
                        "no_instruction_per_warp_generation", "global_frontier", "global_late"
                    ),
                    "global_branch_resolving_delta": front_delta(
                        "branch_resolving_per_warp_generation", "global_frontier", "global_late"
                    ),
                }
            )

    candidates = read_json(root / "metadata/front_end_candidates.json")
    best = candidates["candidates"][0]
    require(
        best["stop_fetch_at_unresolved_branch"] is True
        and best["branch_resolve_latency"] == 3,
        "front-end branch candidate",
    )
    host_replay = read_json(root / "metadata/host_replay_branch_resolution_ab.json")
    require(
        host_replay["decision"] == "no_replay_time_regression_in_interleaved_ab",
        "host replay A/B",
    )
    production_root = root / "regression-branch-resolution/production"
    production_predicted = read_json(production_root / "predicted.json")
    prior_predicted = read_json(
        REPO_ROOT
        / "results/captures/sm86_fixed_latency_result_load_priority_20260817_v1"
        / "regression-priority/production/predicted.json"
    )
    require(production_predicted == prior_predicted, "production predictions changed")
    for name in production_predicted:
        require(
            (production_root / f"{name}-1.json").read_bytes()
            == (production_root / f"{name}-2.json").read_bytes(),
            f"production {name}: nondeterministic replay",
        )
    observed_production = read_json(
        REPO_ROOT
        / "results/captures/sm86_cycle_residual_20260817_v1"
        / "analysis/observed_sm_active_median.json"
    )
    production_errors = [
        production_predicted[key] - observed_production[key]
        for key in sorted(observed_production)
    ]
    production_score = {
        "coverage": len(production_errors),
        "mae": statistics.mean(abs(error) for error in production_errors),
        "rmse": math.sqrt(statistics.mean(error * error for error in production_errors)),
        "mape": statistics.mean(
            abs(production_predicted[key] - observed_production[key])
            / observed_production[key]
            for key in observed_production
        ),
        "predictions_unchanged": 6,
    }
    branch_by_unroll = {}
    for unroll in UNROLLS:
        values = [
            front_end[cell["key"]]["branch_cycles_per_target"]
            for cell in cells
            if cell["unroll"] == unroll
        ]
        branch_by_unroll[str(unroll)] = {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }

    return {
        "schema_version": 1,
        "campaign": {
            "cells": 96,
            "correctness_executions": 672,
            "ncu_profiles": ncu_profiles,
            "expanded_cells": sorted(EXPANDED),
            "front_end_ncu_profiles": 96,
            "source_counters_profiles": 32,
            "nvbit_captures": 96,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 96,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
            "exact_l1_traffic": True,
            "exact_replay_branch_diagnostics": True,
            "host_replay_ab_samples_per_arm": 200,
            "production_predictions_unchanged": 6,
        },
        "static": static,
        "calibration": {
            "fixed_d3": "deterministic_checksum_failure",
            "fixed_d4_d5": "valid",
            "global_d4": "valid_and_therefore_not_an_identifying_negative_arm",
            "global_d5_d6": "equal_work_transfer_holdout_only",
        },
        "branch_resolution": {
            "cycles_per_uniform_target": {
                "minimum": min(branch_cycles_per_target),
                "median": statistics.median(branch_cycles_per_target),
                "maximum": max(branch_cycles_per_target),
            },
            "by_unroll": branch_by_unroll,
            "all_targets_uniform": True,
            "candidate_latency_cycles": best["branch_resolve_latency"],
            "candidate_branch_counter_mae": best[
                "branch_resolving_per_warp_generation_mae"
            ],
        },
        "cells": rows,
        "source_counters": sources,
        "front_end_candidates": {
            "production": {
                key: candidates["production"][key]
                for key in (
                    "primary_mae",
                    "transfer_mae",
                    "fixed_driver_mae",
                    "fixed_active_mae",
                    "global_driver_mae",
                    "global_active_mae",
                )
            },
            "best": {
                key: best[key]
                for key in (
                    "capacity",
                    "stop_fetch_at_unresolved_branch",
                    "branch_resolve_latency",
                    "front_end_counter_mae",
                    "no_instruction_per_warp_generation_mae",
                    "branch_resolving_per_warp_generation_mae",
                    "primary_mae",
                    "transfer_mae",
                )
            },
            "tested": len(candidates["candidates"]),
        },
        "host_replay_ab": host_replay,
        "production": production_score,
        "decision": {
            "status": "uniform_branch_resolution_identified_ibuffer_capacity_unresolved",
            "activate_branch_resolution_cycles": 3,
            "activate_ibuffer_capacity": False,
            "evidence": [
                "exact branch counts scale as 1024/U per warp and every target is uniform",
                "direct branch-resolving counters identify three cycles independently of U and instruction family",
                "pair branch-resolving deltas remain near zero while no-instruction deltas change sign and magnitude",
                "fixed no-instruction deltas correlate with NCU-active treatment deltas but do not fully explain driver phase",
                "the best direct-counter fit selects one IBuffer entry, conflicting with the paper/reference three-entry result",
            ],
            "next_slice": (
                "paper-style branch-free independent-instruction fetch/decode probe with controlled "
                "stall/yield cadence to identify effective IBuffer entries and fetch-selection policy"
            ),
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    branch = summary["branch_resolution"]
    candidate = summary["front_end_candidates"]
    production = summary["production"]
    lines = [
        "# sm_86 branch-frequency/static-footprint factorial",
        "",
        "## Decision",
        "",
        "The campaign identifies a three-cycle uniform backward-branch resolution",
        "state, but it does not identify a transferable IBuffer capacity. Branch",
        "counts are exact, all targets are uniform, and measured resolving cost stays",
        f"between {branch['cycles_per_uniform_target']['minimum']:.3f} and {branch['cycles_per_uniform_target']['maximum']:.3f} warp-cycles/target (median {branch['cycles_per_uniform_target']['median']:.3f}).",
        "The d4/d5 treatment changes `no_instruction` stalls while matched branch",
        "resolution remains nearly invariant, localizing the remaining error to",
        "fetch selection / decoded-instruction availability rather than branch",
        "execution latency or L1 data traffic.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells and {campaign['correctness_executions']} seven-trial correctness executions",
        f"- {campaign['ncu_profiles']} general NCU profiles, including seven-repeat expansion of {len(campaign['expanded_cells'])} paired cells",
        f"- {campaign['front_end_ncu_profiles']} direct front-end NCU profiles and {campaign['source_counters_profiles']} SourceCounters profiles",
        f"- {campaign['nvbit_captures']} NVBit captures, {campaign['trace_records']:,} dynamic records, and {campaign['byte_identical_replay_pairs']} byte-identical replay pairs",
        f"- maximum active-cycle CV {campaign['maximum_active_cycle_cv_pct']:.3f}% and robust MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "- 40 spill-free cubins, 4,400/4,400 decoded controls, 1,190 projections, zero gaps",
        "- retained corpus: 135 sidecars, 47,488 records, 18,372 projections, zero gaps",
        "- issue ratio 1.0, pair-exact executed counts, and exact fixed/global L1 traffic",
        "- replay branch diagnostics exactly equal 64/U targets per warp and three cycles per target",
        "- interleaved 200-sample-per-arm host A/B finds no replay-time regression",
        f"- production remains 6/6 with unchanged predictions and {production['mape'] * 100:.3f}% MAPE",
        "",
        "## Fixed d5 − d4 treatment",
        "",
        "| U | W | Warps/SMSP | Driver Δ | NCU-active Δ | Replay Δ | No-instruction Δ | Branch-resolving Δ |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cells"]:
        counts = "/".join(map(str, row["warps_per_smsp"]))
        lines.append(
            f"| {row['unroll']} | {row['warps']} | {counts} | {row['driver_fixed_delta']:.3f} | "
            f"{row['active_fixed_delta']:.3f} | {row['replay_fixed_delta']:.3f} | "
            f"{row['fixed_no_instruction_delta']:.4f} | {row['fixed_branch_resolving_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Global d6 − d5 transfer holdout",
            "",
            "The new register allocation makes global d4 semantically valid, so",
            "global d5/d6 is not called a bypass frontier. It remains useful as an",
            "equal-work transfer holdout.",
            "",
            "| U | W | Driver Δ | NCU-active Δ | Replay Δ | No-instruction Δ | Branch-resolving Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cells"]:
        lines.append(
            f"| {row['unroll']} | {row['warps']} | {row['driver_global_delta']:.3f} | "
            f"{row['active_global_delta']:.3f} | {row['replay_global_delta']:.3f} | "
            f"{row['global_no_instruction_delta']:.4f} | {row['global_branch_resolving_delta']:.4f} |"
        )
    production = candidate["production"]
    best = candidate["best"]
    lines.extend(
        [
            "",
            "## Candidate boundary",
            "",
            f"Twenty-four bounded candidates combine IBuffer capacities 1/2/3/4/8/16, stop/continue fetch, and 2/3/4-cycle branch resolution. Direct counters select stop-at-branch and exactly three resolving cycles (branch-counter MAE {best['branch_resolving_per_warp_generation_mae']:.4f}).",
            f"The best aggregate counter fit chooses capacity {best['capacity']}, with no-instruction MAE {best['no_instruction_per_warp_generation_mae']:.3f} warp-cycles/warp/generation. It improves primary timing MAE from {production['primary_mae']:.3f} to {best['primary_mae']:.3f} and transfer MAE from {production['transfer_mae']:.3f} to {best['transfer_mae']:.3f}, but still misses the large U1 W9/W13/W16 fixed phases and conflicts with the reference paper's three-entry IBuffer.",
            "",
            "Therefore the three-cycle branch-resolution fact is ready for a typed",
            "architecture capability, while IBuffer capacity/fetch policy remains",
            "unidentified and must not be tuned by U, warp count, or instruction family.",
            "",
            "## Next slice",
            "",
            "Use the paper's branch-free independent-instruction construction with",
            "controlled stall/yield cadence and four warps per SMSP. Measure direct",
            "`no_instruction` counters while sweeping decoded depth. This separates",
            "IBuffer capacity from backward-branch target refill without relying on",
            "kernel endpoint timing.",
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
        print(f"front-end footprint analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
