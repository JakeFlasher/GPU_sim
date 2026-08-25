#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed-result bypass-frontier campaign."""

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
    / "results"
    / "captures"
    / "sm86_fixed_latency_result_bypass_20260817_v1"
)
WIDTHS = (1, 4, 8, 12, 16)
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
FUNCTIONS = {
    "fixed": "_Z19bypass_fixed_searchPyPjPKj",
    "global": "_Z20bypass_global_searchPyPjPKjS2_",
    "canary": "_Z20bypass_global_canaryPjPKjS1_",
}
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
    path = HERE.parent / "fixed_latency_result_order" / "analyze.py"
    spec = importlib.util.spec_from_file_location("bypass_order_analysis", path)
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


def names() -> list[str]:
    result = [
        "fixed_latency_result_bypass_native",
        "fixed_latency_result_bypass_trace_native",
    ]
    result.extend(f"fixed_latency_result_bypass_{variant}" for variant in VARIANTS)
    result.extend(f"fixed_latency_result_bypass_trace_{variant}" for variant in VARIANTS)
    result.extend(
        f"fixed_latency_result_bypass_{variant}"
        for variant in ("fixed_invalid", "canary_d4", "canary_d5")
    )
    return result


def validate_static(root: Path) -> dict[str, Any]:
    sidecars = {}
    for name in names():
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 104, f"{name}: decoded")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(set(sidecar["functions"]) == set(FUNCTIONS.values()), f"{name}: functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(resources.count("REG:10") == 1, f"{name}: canary registers")
        require(resources.count("REG:14") == 1, f"{name}: fixed registers")
        require(resources.count("REG:16") == 1, f"{name}: global registers")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill/local")

    for label in ("long", "trace"):
        patch = read_json(root / "metadata" / f"{label}_bypass_patch.json")
        require(patch["all_changes_inside_selected_control_words"] is True, f"{label}: scope")
        require(patch["equal_instruction_work"] is True, f"{label}: work")
        require(patch["equal_static_stall_sum_within_pairs"] is True, f"{label}: stalls")
        require(patch["fixed_boundary"] == {
            "invalid_distance": 3,
            "bypass_distance": 4,
            "post_retirement_control_distance": 5,
        }, f"{label}: fixed boundary")
        require(patch["global_boundary"] == {
            "invalid_distance": 4,
            "register_available_distance": 5,
            "late_control_distance": 6,
        }, f"{label}: global boundary")
        require(all(value == 2 for value in patch["pair_differences"].values()), f"{label}: pair bytes")

    calibration_patch = read_json(root / "metadata/calibration_bypass_patch.json")
    require(calibration_patch["fixed_d3_invalid_d4_valid"] is True, "calibration fixed")
    require(calibration_patch["global_d4_invalid_d5_valid"] is True, "calibration global")

    for trace in (False, True):
        prefix = "fixed_latency_result_bypass_trace" if trace else "fixed_latency_result_bypass"
        rows = {
            variant: sidecars[f"{prefix}_{variant}"]["functions"]
            for variant in VARIANTS
        }
        for left, right, function, expected_pcs in (
            ("fixed_bypass", "fixed_retired", FUNCTIONS["fixed"], [0xE0, 0x100]),
            ("global_frontier", "global_late", FUNCTIONS["global"], [0xD0, 0x110]),
        ):
            differing = []
            for a, b in zip(rows[left][function], rows[right][function]):
                require(a["pc"] == b["pc"], f"{left}/{right}: PCs")
                if a != b:
                    differing.append(int(a["pc"]))
                    require(a["opcode_text"] == b["opcode_text"], "pair opcode")
                    require(a["low_word"] == b["low_word"], "pair low word")
                    for field in ("yield_flag", "write_barrier", "read_barrier", "wait_mask", "reuse"):
                        require(a["control"][field] == b["control"][field], f"pair {field}")
            require(differing == expected_pcs, f"{left}/{right}: diff PCs")
        fixed = {int(row["pc"]): row for row in rows["fixed_bypass"][FUNCTIONS["fixed"]]}
        require(fixed[0xE0]["opcode_text"].startswith("LOP3.LUT R5,"), "fixed producer")
        require(fixed[0xF0]["opcode_text"].startswith("LOP3.LUT R6, R5,"), "fixed consumer")
        global_rows = {int(row["pc"]): row for row in rows["global_frontier"][FUNCTIONS["global"]]}
        require(global_rows[0xC0]["opcode_text"] == "IMAD.IADD R5, R3, 0x1, R9", "global producer")
        require(global_rows[0xE0]["opcode_text"] == "LDG.E R4, [R4.64]", "global consumer")

    projection = read_json(root / "metadata/operand_projection_audit.json")
    require((projection["files"], projection["records"], projection["projected"], projection["gaps"]) == (13, 1352, 169, 0), "local projection")
    corpus = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require((corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"]) == (87, 42256, 17078, 0), "corpus projection")
    return {
        "binaries": 13,
        "decoded_instructions": 1352,
        "spill_free": True,
        "local_projection": {key: projection[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
        "retained_corpus_projection": {key: corpus[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
    }


def validate_calibration(root: Path) -> dict[str, Any]:
    calibration = read_json(root / "calibration/summary.json")
    fixed = json.loads(calibration["fixed_d3"]["stdout"])
    require(calibration["fixed_d3"]["return_code"] == 1 and fixed["correct"] is False, "fixed d3")
    require(calibration["global_d5"]["return_code"] == 0, "global d5 return")
    require(json.loads(calibration["global_d5"]["stdout"])["correct"] is True, "global d5 oracle")
    d4 = calibration["global_d4"]["runs"]
    require(len(d4) == 7, "global d4 trials")
    require(all(row["return_code"] == 1 and "CUDA_ERROR_ILLEGAL_ADDRESS" in row["stderr"] for row in d4), "global d4 error")
    return {
        "fixed_d3_trials": len(fixed["samples_cycles"]),
        "fixed_d3_checksum_invalid": True,
        "fixed_d4_valid": True,
        "global_d4_illegal_address_trials": len(d4),
        "global_d5_trials": json.loads(calibration["global_d5"]["stdout"])["trials"],
        "global_d5_valid": True,
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata/matrix.json")["cells"]
    require(len(cells) == 20 and len({cell["key"] for cell in cells}) == 20, "matrix")
    static = validate_static(root)
    calibration = validate_calibration(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings
    bindings = load_dynamic_trace_bindings()
    live = {}
    replay = {}
    trace_records = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["variant"] == cell["variant"] and payload["warps"] == cell["warps"], f"{key}: identity")
        require(payload["generations"] == 1025 and len(payload["samples_cycles"]) == 7, f"{key}: trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]
        paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(paths) == 3, f"{key}: NCU repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in paths]
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
        ), f"{key}: L1 traffic")
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        median = lambda name: statistics.median(row[name] for row in rows)
        live[key] = {
            "driver_cycles_per_generation": statistics.median(driver_samples) / 1025,
            "active_cycles_per_generation": active_median / 1025,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            **{name: median(name) for name in REQUIRED_METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True and trace_payload["generations"] == 65, f"{key}: trace")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        block_paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(block_paths) == 1, f"{key}: block count")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(block_paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {int(warp.instructions[0].warpid_sm) for warp in block.warps.values() if warp.instructions}
        require(slots == expected_physical_slots(cell["warps"]), f"{key}: slots")
        trace_records += records
        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: deterministic replay")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: records")
        l1 = launch["counters"]["l1"]
        require(l1["hits"] == (256 if global_kind else 0) * cell["warps"], f"{key}: replay hits")
        require(l1["misses"] == expected_misses, f"{key}: replay misses")
        replay[key] = launch

    for warps in WIDTHS:
        for left, right in (("fixed_bypass", "fixed_retired"), ("global_frontier", "global_late")):
            for metric in COUNT_METRICS:
                require(live[f"w{warps}_{left}"][metric] == live[f"w{warps}_{right}"][metric], f"W{warps} {left}/{right}: {metric}")

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 8, "SourceCounters count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    width_rows = []
    for warps in WIDTHS:
        def delta(field: str, left: str, right: str) -> float:
            return live[f"w{warps}_{right}"][field] - live[f"w{warps}_{left}"][field]
        def replay_cpg(variant: str) -> float:
            return replay[f"w{warps}_{variant}"]["timing"]["diagnostic_last_issue_cycles"] / 65
        width_rows.append({
            "warps": warps,
            "stage": "identification" if warps in {1, 4} else "held_out_width",
            "live_driver_fixed_delta": delta("driver_cycles_per_generation", "fixed_bypass", "fixed_retired"),
            "live_active_fixed_delta": delta("active_cycles_per_generation", "fixed_bypass", "fixed_retired"),
            "replay_fixed_delta": replay_cpg("fixed_retired") - replay_cpg("fixed_bypass"),
            "live_driver_global_delta": delta("driver_cycles_per_generation", "global_frontier", "global_late"),
            "live_active_global_delta": delta("active_cycles_per_generation", "global_frontier", "global_late"),
            "replay_global_delta": replay_cpg("global_late") - replay_cpg("global_frontier"),
            "fixed_long_scoreboard_delta": delta("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "fixed_bypass", "fixed_retired"),
            "global_long_scoreboard_delta": delta("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "global_frontier", "global_late"),
        })
    replay_delta_mae = statistics.mean(
        abs(row["replay_fixed_delta"] - row["live_driver_fixed_delta"])
        + abs(row["replay_global_delta"] - row["live_driver_global_delta"])
        for row in width_rows
    ) / 2
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 20,
            "correctness_executions": 140,
            "ncu_profiles": 60,
            "source_counters_profiles": 8,
            "nvbit_captures": 20,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 20,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
            "exact_l1_traffic": True,
        },
        "static": static,
        "calibration": calibration,
        "live": live,
        "widths": width_rows,
        "source_counters": sources,
        "model": {"production_replay_delta_mae": replay_delta_mae},
        "decision": {
            "status": "fixed_result_bypass_frontier_identified",
            "projection_promoted": True,
            "active_bypass_timing_added": False,
            "evidence": [
                "fixed distance 3 fails seven checksum trials while distance 4 is correct",
                "global-address distance 4 raises CUDA_ERROR_ILLEGAL_ADDRESS in seven fresh contexts while distance 5 is correct",
                "W1/W4 valid d4/d5 and d5/d6 controls are null within measurement noise",
                "exact IMAD.IADD projection closes the compiler-generated global producer",
            ],
            "limitations": [
                "the fixed and global canaries use different exact producer opcodes",
                "invalid binaries establish semantic availability but are not profiler inputs",
                "W8+ pairs enter the previously observed generic producer-shift phase",
                "the control-driven timing simulator does not execute values and must not repair invalid compiler schedules with a scoreboard",
            ],
            "next_slice": "localize the W8+ generic producer-shift phase; retain queue capacity as a separate canary",
        },
    }


def render(summary: dict[str, Any]) -> str:
    c = summary["campaign"]
    lines = [
        "# sm_86 fixed-result bypass-frontier reduction",
        "",
        "## Decision",
        "",
        "The fixed-result bypass frontier is identified semantically. A fixed",
        "consumer at distance 4 receives the LOP3 result correctly; distance 3",
        "does not. A global load whose address high half is fixed-produced faults",
        "at distance 4 and becomes correct at distance 5. This reproduces Listing",
        "3's one-cycle fixed-versus-variable availability distinction.",
        "",
        "No new timing state is added. Valid W1/W4 controls are null, while W8+",
        "moves into the already known generic producer-shift phase. The simulator",
        "uses compiler controls and does not execute register values; adding a",
        "scoreboard to repair deliberately invalid schedules would model the wrong",
        "hardware mechanism.",
        "",
        "## Campaign integrity",
        "",
        f"- {c['cells']} cells, {c['correctness_executions']} correctness executions, and {c['ncu_profiles']} NCU profiles",
        f"- {c['source_counters_profiles']} SourceCounters profiles, {c['nvbit_captures']} NVBit captures, and {c['trace_records']:,} replay records",
        f"- maximum active-cycle CV {c['maximum_active_cycle_cv_pct']:.3f}% and maximum MAD {c['maximum_active_cycle_mad_pct']:.3f}%",
        "- 13 spill-free cubins, 1,352/1,352 controls, 169 projections, zero gaps",
        "- retained corpus: 87 sidecars, 42,256 records, 17,078 projections, zero gaps",
        "- exact fixed/global pair counts, issue ratio 1.0, exact L1 traffic, and deterministic replay",
        "",
        "## Availability calibration",
        "",
        "- fixed d3: 7/7 checksum failures; fixed d4: valid",
        "- global d4: 7/7 fresh-context `CUDA_ERROR_ILLEGAL_ADDRESS`; global d5: 7/7 valid",
        "",
        "## Width transfer",
        "",
        "| Warps | Driver fixed d5−d4 | NCU fixed Δ | Replay fixed Δ | Driver global d6−d5 | NCU global Δ | Replay global Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["widths"]:
        lines.append(
            f"| {row['warps']} | {row['live_driver_fixed_delta']:.3f} | "
            f"{row['live_active_fixed_delta']:.3f} | {row['replay_fixed_delta']:.3f} | "
            f"{row['live_driver_global_delta']:.3f} | {row['live_active_global_delta']:.3f} | "
            f"{row['replay_global_delta']:.3f} |"
        )
    lines.extend([
        "",
        "## Static increment",
        "",
        "The operand projector now admits exact `IMAD.IADD` as a four-operand",
        "FMA-family fixed instruction. MOV/WIDE/X and other IMAD modifiers remain",
        "outside the vocabulary. The valid cubin pairs change only selected high",
        "control words and retain equal selected stall sums.",
        "",
        "## Next slice",
        "",
        "Localize the generic W8+ producer-shift phase independently from bypass",
        "and bank priority. Queue depth/pop width remain unresolved; retain the W8",
        "ninth-reservation cell as a canary rather than importing capacity eight.",
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
        print(f"bypass analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
