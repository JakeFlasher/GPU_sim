#!/usr/bin/env python3
"""Validate and reduce the dense sm_86 producer-shift campaign."""

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
DEFAULT_ROOT = (
    REPO_ROOT
    / "results/captures/sm86_fixed_latency_producer_shift_20260817_v1"
)
WIDTHS = tuple(range(4, 17))
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
SOURCE_WIDTHS = {4, 5, 6, 8, 9, 12, 13, 16}
REQUIRED_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__inst_issued_per_issue_active.ratio",
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
    "smsp__inst_executed_pipe_lsu.sum",
)
EMPTY_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("dense_shift_order_analysis", path)
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


def validate_static(root: Path) -> dict[str, Any]:
    names = [
        f"fixed_latency_producer_shift_{'trace_' if trace else ''}{variant}"
        for trace in (False, True)
        for variant in VARIANTS
    ]
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        require(sidecar["summary"]["total_decoded"] == 104, f"{name}: decoded")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(resources.count("REG:10") == 1, f"{name}: canary registers")
        require(resources.count("REG:14") == 1, f"{name}: fixed registers")
        require(resources.count("REG:16") == 1, f"{name}: global registers")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spills")
    projection = read_json(root / "metadata/operand_projection_audit.json")
    require((projection["files"], projection["records"], projection["projected"], projection["gaps"]) == (8, 832, 104, 0), "local projection")
    corpus = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require((corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"]) == (95, 43088, 17182, 0), "corpus projection")
    return {
        "binaries": 8,
        "decoded_instructions": 832,
        "spill_free": True,
        "local_projection": {key: projection[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
        "retained_corpus_projection": {key: corpus[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
    }


def stage(warps: int) -> str:
    if warps == 4:
        return "one_warp_per_smsp_baseline"
    if warps <= 8:
        return "second_warp_partition_fill"
    if warps <= 12:
        return "third_warp_partition_fill"
    return "fourth_warp_partition_fill"


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata/matrix.json")["cells"]
    require(len(cells) == 52 and len({cell["key"] for cell in cells}) == 52, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings
    bindings = load_dynamic_trace_bindings()
    live = {}
    replay = {}
    trace_records = 0
    ncu_profiles = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    expanded_cells = []
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True and payload["warps"] == cell["warps"], f"{key}: correctness")
        require(payload["generations"] == 1025 and len(payload["samples_cycles"]) == 7, f"{key}: trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]
        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 7 if key == "w10_global_late" else 3
        require(len(repeat_paths) == expected_repeats, f"{key}: NCU repeats")
        if expected_repeats == 7:
            expanded_cells.append(key)
        ncu_profiles += len(repeat_paths)
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in REQUIRED_METRICS:
            require(all(metric in row for row in rows), f"{key}: {metric}")
        require(all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows), f"{key}: issue ratio")
        global_kind = cell["kind"] == "global"
        expected_hits = (4096 if global_kind else 0) * cell["warps"]
        expected_misses = (8 if global_kind else 4) * cell["warps"]
        require(all(
            row["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"] == expected_hits
            and row["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"] == expected_misses
            for row in rows
        ), f"{key}: exact L1 traffic")
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        median = lambda metric: statistics.median(row[metric] for row in rows)
        live[key] = {
            "driver_cycles_per_generation": statistics.median(driver_samples) / 1025,
            "active_cycles_per_generation": active_median / 1025,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            **{metric: median(metric) for metric in REQUIRED_METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True and trace_payload["generations"] == 65, f"{key}: trace")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: block")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {int(warp.instructions[0].warpid_sm) for warp in block.warps.values() if warp.instructions}
        require(slots == expected_physical_slots(cell["warps"]), f"{key}: slots")
        trace_records += records
        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: replay determinism")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: records")
        replay[key] = launch

    require(ncu_profiles == 160, "NCU profile count")
    require(expanded_cells == ["w10_global_late"], "adaptive expansion")
    for warps in WIDTHS:
        for left, right in (("fixed_bypass", "fixed_retired"), ("global_frontier", "global_late")):
            for metric in COUNT_METRICS:
                require(live[f"w{warps}_{left}"][metric] == live[f"w{warps}_{right}"][metric], f"W{warps} {left}/{right}: {metric}")
    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 32, "SourceCounters count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}

    rows = []
    for warps in WIDTHS:
        def delta(field: str, left: str, right: str) -> float:
            return live[f"w{warps}_{right}"][field] - live[f"w{warps}_{left}"][field]
        def sim(variant: str) -> float:
            return replay[f"w{warps}_{variant}"]["timing"]["diagnostic_last_issue_cycles"] / 65
        rows.append({
            "warps": warps,
            "stage": stage(warps),
            "warps_per_smsp": smsp_counts(warps),
            "driver_fixed_delta": delta("driver_cycles_per_generation", "fixed_bypass", "fixed_retired"),
            "active_fixed_delta": delta("active_cycles_per_generation", "fixed_bypass", "fixed_retired"),
            "replay_fixed_delta": sim("fixed_retired") - sim("fixed_bypass"),
            "driver_global_delta": delta("driver_cycles_per_generation", "global_frontier", "global_late"),
            "active_global_delta": delta("active_cycles_per_generation", "global_frontier", "global_late"),
            "replay_global_delta": sim("global_late") - sim("global_frontier"),
            "fixed_long_scoreboard_delta": delta("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "fixed_bypass", "fixed_retired"),
            "global_long_scoreboard_delta": delta("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "global_frontier", "global_late"),
        })

    candidates = read_json(root / "metadata/front_end_candidates.json")
    require(candidates["decision"]["status"] == "all_reference_inspired_front_end_candidates_rejected", "candidate decision")
    require(len(candidates["candidates"]) == 10, "candidate count")
    best = candidates["candidates"][0]
    require(best["capacity"] == 4 and best["stop_fetch_at_unresolved_branch"] is False, "best candidate identity")
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 52,
            "correctness_executions": 364,
            "ncu_profiles": ncu_profiles,
            "expanded_cells": expanded_cells,
            "source_counters_profiles": 32,
            "nvbit_captures": 52,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 52,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
            "exact_l1_traffic": True,
        },
        "static": static,
        "widths": rows,
        "source_counters": sources,
        "front_end_candidates": {
            "production_driver_mae": candidates["production"]["driver_mae"],
            "production_active_mae": candidates["production"]["active_mae"],
            "best": {key: best[key] for key in ("capacity", "stop_fetch_at_unresolved_branch", "driver_mae", "active_mae", "fixed_driver_mae", "global_driver_mae")},
            "tested": len(candidates["candidates"]),
        },
        "decision": {
            "status": "generic_shift_localized_to_per_smsp_front_end_phase_but_model_unidentified",
            "activate_front_end_timing": False,
            "evidence": [
                "W5 already changes fixed timing when only SMSP0 receives a second warp",
                "W5-W8 fixed NCU deltas stay near +1.42 cycles per generation as remaining SMSPs fill",
                "global deltas change sign across partial/full second-, third-, and fourth-warp cohorts",
                "dispatch and scoreboard deltas are near zero in most fixed cells, excluding result-bank and dependency causes",
                "the reference-inspired 3-entry IBuffer is not the best complete fit",
            ],
            "candidate_rejection": (
                "the best 4-entry continue-through-branch candidate reduces delta MAE but "
                "overprices fixed W9-W12 and misses global W13-W16; choosing capacities by "
                "cohort or instruction family would be a protocol/warp-count correction"
            ),
            "next_slice": "branch-free body-length and instruction-footprint factorial for front-end refill/branch coupling",
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    candidates = summary["front_end_candidates"]
    lines = [
        "# sm_86 dense producer-shift localization",
        "",
        "## Decision",
        "",
        "The W8+ residual is localized to a per-SMSP instruction-frontier phase,",
        "not a result queue, register bank, or all-partition occupancy threshold.",
        "It begins at W5 when only SMSP0 receives a second warp, evolves as the",
        "other partitions fill, and changes sign between fixed-only and global-load",
        "loops. No front-end timing is promoted yet.",
        "",
        "Ten reference-inspired candidates model one fetch/decode per SMSP per",
        "cycle, IBuffer capacities 1/2/3/4/8, and stop/continue behavior at an",
        "unresolved backward branch. The best capacity-4 continue candidate improves",
        f"driver delta MAE from {candidates['production_driver_mae']:.3f} to {candidates['best']['driver_mae']:.3f} and NCU-active MAE from {candidates['production_active_mae']:.3f} to {candidates['best']['active_mae']:.3f}, but it does not transfer across cohorts/families.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells and {campaign['correctness_executions']} correctness executions",
        f"- {campaign['ncu_profiles']} NCU profiles; one adaptive expansion (`w10_global_late`)",
        f"- {campaign['source_counters_profiles']} SourceCounters profiles, {campaign['nvbit_captures']} NVBit captures, and {campaign['trace_records']:,} replay records",
        f"- maximum active-cycle CV {campaign['maximum_active_cycle_cv_pct']:.3f}% and maximum MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "- 8 spill-free cubins, 832/832 controls, 104 projections, zero gaps",
        "- retained corpus: 95 sidecars, 43,088 records, 17,182 projections, zero gaps",
        "- exact pair counts, issue ratio 1.0, exact L1 traffic, and deterministic replay",
        "",
        "## Dense width transfer",
        "",
        "| W | Warps/SMSP | Driver fixed Δ | NCU fixed Δ | Replay fixed Δ | Driver global Δ | NCU global Δ | Replay global Δ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["widths"]:
        counts = "/".join(map(str, row["warps_per_smsp"]))
        lines.append(
            f"| {row['warps']} | {counts} | {row['driver_fixed_delta']:.3f} | "
            f"{row['active_fixed_delta']:.3f} | {row['replay_fixed_delta']:.3f} | "
            f"{row['driver_global_delta']:.3f} | {row['active_global_delta']:.3f} | "
            f"{row['replay_global_delta']:.3f} |"
        )
    lines.extend([
        "",
        "## Candidate boundary",
        "",
        "The paper/reference configuration uses a three-entry per-warp IBuffer,",
        "fetch/decode width one, and an eight-instruction stream prefetcher. That",
        "shape improves some high-occupancy global cells but reverses the fixed",
        "second-warp cohort and overprices the third-warp cohort. Capacity four",
        "matches fixed W5-W8 and W13-W16 more closely, yet predicts about +3.94",
        "cycles/generation for fixed W9-W12 and misses the positive global W13-W16",
        "regime. No single state machine transfers.",
        "",
        "## Next slice",
        "",
        "Use branch-free unrolled bodies with several static instruction footprints",
        "and matched backward-branch controls. This will separate IBuffer refill and",
        "instruction-cache-line/prefetch effects from branch redirection before any",
        "front-end state enters production replay.",
        "",
    ])
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
        print(f"producer-shift analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
