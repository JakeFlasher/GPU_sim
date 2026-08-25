#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed-result reservation-capacity campaign."""

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
    REPO_ROOT / "results" / "captures" / "sm86_fixed_latency_result_capacity_20260817_v1"
)
WIDTHS = (1, 4, 8)
DEPTHS = (7, 8, 9)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 64
FUNCTION = "_Z22result_capacity_searchPyPjPKj"
REQUIRED_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_fma_type_fp16.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
)
EMPTY_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order" / "analyze.py"
    spec = importlib.util.spec_from_file_location("capacity_order_analysis", path)
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
    names = [
        "fixed_latency_result_capacity_native",
        "fixed_latency_result_capacity_trace_native",
    ]
    names.extend(f"fixed_latency_result_capacity_b{depth}" for depth in DEPTHS)
    names.extend(f"fixed_latency_result_capacity_trace_b{depth}" for depth in DEPTHS)
    sidecars = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 72, f"{name}: decoded")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(set(sidecar["functions"]) == {FUNCTION}, f"{name}: function")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(resources.count("REG:28") == 1, f"{name}: registers")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill/local")

    for label in ("long", "trace"):
        patch = read_json(root / "metadata" / f"{label}_capacity_patch.json")
        require(patch["all_changes_inside_selected_control_words"] is True, f"{label}: scope")
        require(patch["equal_instruction_work"] is True, f"{label}: work")
        require(patch["equal_static_stall_sum"] is True, f"{label}: stall sum")
        require(patch["capacity_signature"] == "b9 slower than b8 while b8 equals b7", f"{label}: signature")
        require(len(patch["reservation_run"]) == 17, f"{label}: run")
        require([row["reservation_depth_before_gap"] for row in patch["variants"].values()] == [7, 8, 9], f"{label}: depths")

    for trace in (False, True):
        prefix = "fixed_latency_result_capacity_trace" if trace else "fixed_latency_result_capacity"
        rows_by_depth = {
            depth: sidecars[f"{prefix}_b{depth}"]["functions"][FUNCTION]
            for depth in DEPTHS
        }
        for left, right in ((7, 8), (8, 9)):
            a = rows_by_depth[left]
            b = rows_by_depth[right]
            require(len(a) == len(b) == 72, f"{prefix}: rows")
            differing = []
            for x, y in zip(a, b):
                require(x["pc"] == y["pc"], f"{prefix}: PCs")
                if x != y:
                    differing.append(int(x["pc"]))
                    require(x["opcode_text"] == y["opcode_text"], f"{prefix}: opcode")
                    require(x["low_word"] == y["low_word"], f"{prefix}: low word")
                    for field in ("yield_flag", "write_barrier", "read_barrier", "wait_mask", "reuse"):
                        require(x["control"][field] == y["control"][field], f"{prefix}: {field}")
            require(len(differing) == 2, f"{prefix} b{left}/b{right}: movable controls")
        for depth, rows in rows_by_depth.items():
            by_pc = {int(row["pc"]): row for row in rows}
            require(by_pc[0x1A0]["control"]["stall"] == 15, f"b{depth}: drain")
            require(by_pc[0x1E0]["control"]["stall"] == 1, f"b{depth}: gate")
            run = [row for row in rows if 0x1B0 <= int(row["pc"]) <= 0x2C0 and int(row["pc"]) != 0x1E0]
            require(len(run) == 17, f"b{depth}: selected run")
            require(sum(row["control"]["stall"] == 15 for row in run) == 1, f"b{depth}: gap count")
            require(run[depth - 1]["control"]["stall"] == 15, f"b{depth}: gap ordinal")
            require(all(row["control"]["yield_flag"] == 0 for row in run), f"b{depth}: yield")
            require(all(row["control"]["reuse"] == 0 for row in run), f"b{depth}: reuse")
            require(any(row["opcode_text"].startswith("HMMA.16816.F32 ") for row in run), f"b{depth}: anchor")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(
        (projection["files"], projection["records"], projection["projected"], projection["gaps"])
        == (8, 576, 208, 0),
        "local projection",
    )
    corpus = read_json(root / "metadata" / "retained_corpus_operand_projection_audit.json")
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (64, 40424, 16805, 0),
        "corpus projection",
    )
    return {
        "binaries": 8,
        "decoded_instructions": 576,
        "registers": 28,
        "barriers": 1,
        "spill_free": True,
        "reservation_run_instructions": 17,
        "movable_depths": [7, 8, 9],
        "local_projection": {key: projection[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
        "retained_corpus_projection": {key: corpus[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata/matrix.json")["cells"]
    require(len(cells) == 9 and len({cell["key"] for cell in cells}) == 9, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live = {}
    replay = {}
    trace_records = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    candidate_outputs_identical = 0
    restored_outputs_identical = 0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["depth"] == cell["depth"] and payload["warps"] == cell["warps"], f"{key}: identity")
        require(payload["generations"] == LONG_GENERATIONS and len(payload["samples_cycles"]) == 7, f"{key}: trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]
        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) == 7, f"{key}: NCU repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in REQUIRED_METRICS:
            require(all(metric in row for row in rows), f"{key}: {metric}")
        require(all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows), f"{key}: issue ratio")
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv); maximum_mad = max(maximum_mad, mad)
        median = lambda name: statistics.median(row[name] for row in rows)
        live[key] = {
            "active_cycles": active_median,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            "driver_cycles": statistics.median(driver_samples),
            **{name: median(name) for name in REQUIRED_METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True and trace_payload["generations"] == TRACE_GENERATIONS, f"{key}: trace")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: blocks")
        block = bindings.threadblock_pb2.threadblock(); block.ParseFromString(paths[0].read_bytes())
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
        for directory, counter_name in (
            ("simulator-capacity8", "candidate"),
            ("simulator-restored", "restored"),
        ):
            candidate_first = root / directory / key / "replay-1.json"
            candidate_second = root / directory / key / "replay-2.json"
            require(candidate_first.read_bytes() == candidate_second.read_bytes(), f"{key}: {counter_name} determinism")
            require(candidate_first.read_bytes() == first.read_bytes(), f"{key}: {counter_name} changed replay")
            if counter_name == "candidate":
                candidate_outputs_identical += 2
            else:
                restored_outputs_identical += 2

    rows_summary = []
    for warps in WIDTHS:
        values = {depth: live[f"w{warps}_b{depth}"] for depth in DEPTHS}
        launches = {depth: replay[f"w{warps}_b{depth}"] for depth in DEPTHS}
        for name in (
            "smsp__inst_issued.sum", "smsp__inst_executed.sum",
            "smsp__inst_executed_pipe_alu.sum", "smsp__inst_executed_pipe_fma.sum",
            "smsp__inst_executed_pipe_fma_type_fp16.sum",
        ):
            require(len({values[depth][name] for depth in DEPTHS}) == 1, f"W{warps}: {name}")
        require(launches[7]["counters"] == launches[8]["counters"] == launches[9]["counters"], f"W{warps}: replay counters")
        live_cpg = {depth: values[depth]["active_cycles"] / LONG_GENERATIONS for depth in DEPTHS}
        sim_cpg = {depth: launches[depth]["timing"]["diagnostic_last_issue_cycles"] / TRACE_GENERATIONS for depth in DEPTHS}
        rows_summary.append(
            {
                "warps": warps,
                "stage": "identification" if warps == 1 else ("subcore_transfer" if warps == 4 else "multiwarp_transfer"),
                "live_cycles_per_generation": live_cpg,
                "replay_cycles_per_generation": sim_cpg,
                "live_delta_b8_minus_b7": live_cpg[8] - live_cpg[7],
                "live_delta_b9_minus_b8": live_cpg[9] - live_cpg[8],
                "replay_delta_b8_minus_b7": sim_cpg[8] - sim_cpg[7],
                "replay_delta_b9_minus_b8": sim_cpg[9] - sim_cpg[8],
                "dispatch_b7_b8_b9": [
                    values[depth]["smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"]
                    for depth in DEPTHS
                ],
                "wait_b7_b8_b9": [
                    values[depth]["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"]
                    for depth in DEPTHS
                ],
                "math_b7_b8_b9": [
                    values[depth]["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"]
                    for depth in DEPTHS
                ],
                "short_scoreboard_b7_b8_b9": [
                    values[depth]["smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"]
                    for depth in DEPTHS
                ],
            }
        )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 6, "SourceCounters count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    w1 = next(row for row in rows_summary if row["warps"] == 1)
    w4 = next(row for row in rows_summary if row["warps"] == 4)
    w8 = next(row for row in rows_summary if row["warps"] == 8)
    strict_signature = (
        abs(w1["live_delta_b9_minus_b8"]) <= 0.01
        and abs(w4["live_delta_b9_minus_b8"]) <= 0.01
        and w8["live_delta_b9_minus_b8"] >= 1.0
        and abs(w8["live_delta_b8_minus_b7"]) <= 0.25
    )
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 9,
            "correctness_executions": 63,
            "ncu_profiles": 63,
            "source_counters_profiles": 6,
            "nvbit_captures": 9,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 9,
            "candidate_outputs_identical_to_production": candidate_outputs_identical,
            "restored_outputs_identical_to_production": restored_outputs_identical,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
        },
        "static": static,
        "live": live,
        "replay": replay,
        "widths": rows_summary,
        "source_counters": sources,
        "decision": {
            "status": "multiwarp_ninth_reservation_boundary_identified_candidate_only",
            "strict_capacity_signature_observed": strict_signature,
            "activate_result_queue_timing": False,
            "evidence": [
                "W1 and W4 b9-minus-b8 are null within 0.001 cycles per generation",
                "W8 b8-minus-b7 is 0.164 and b9-minus-b8 is 1.126 cycles per generation",
                "W8 dispatch pressure rises at b9 while short-scoreboard remains unchanged",
                "production replay predicts b9-minus-b8 as -1.969 cycles per generation",
                "the capacity-8/pop-1 candidate changes none of 18 replay outputs",
            ],
            "limitations": [
                "the boundary requires two warps per modeled SMSP",
                "b8 is not an exact null relative to b7",
                "the construction contains an admitted tensor anchor and CTA barrier",
                "capacity and one-pop policy have not transferred to non-tensor predecessors",
            ],
            "next_slice": "fixed-result versus L1TEX writeback priority; queue capacity remains unidentified",
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        "# sm_86 fixed-result reservation-capacity feasibility reduction",
        "",
        "## Decision",
        "",
        "The campaign identifies a candidate ninth-reservation boundary only when two",
        "warps occupy each modeled SMSP. W1 and W4 are null, while W8 has a small",
        "b8−b7 step and a dominant b9−b8 step. This is consistent with a shared",
        "eight-slot issue-reservation queue, but is not sufficient to activate it:",
        "the construction includes Tensor and BAR work and has not transferred.",
        "",
        "Production replay has no reservation queue and predicts the W8 b9 step with",
        "the wrong sign. An experimental capacity-8/pop-1 implementation changed none",
        "of 18 replay outputs, so it was removed without predecessor transfer.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells, {campaign['correctness_executions']} correctness executions, and {campaign['ncu_profiles']} NCU profiles",
        f"- {campaign['source_counters_profiles']} SourceCounters profiles, {campaign['nvbit_captures']} NVBit captures, and {campaign['trace_records']:,} replay records",
        "- eight spill-free cubins, 576/576 controls, 208 local projections, zero gaps",
        "- retained 64-sidecar corpus: 40,424 records, 16,805 projections, zero gaps",
        "- equal work/counts, issue ratio 1.0, and deterministic replay throughout",
        "- 18 candidate and 18 restored outputs byte-identical to production",
        f"- maximum raw CV {campaign['maximum_active_cycle_cv_pct']:.3f}%; maximum MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "",
        "## Reservation-depth deltas",
        "",
        "| Warps | Stage | Live b8−b7 | Live b9−b8 | Replay b8−b7 | Replay b9−b8 | Dispatch b7/b8/b9 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["widths"]:
        dispatch = "/".join(f"{value:.3f}" for value in row["dispatch_b7_b8_b9"])
        lines.append(
            f"| {row['warps']} | {row['stage']} | {row['live_delta_b8_minus_b7']:.3f} | "
            f"{row['live_delta_b9_minus_b8']:.3f} | {row['replay_delta_b8_minus_b7']:.3f} | "
            f"{row['replay_delta_b9_minus_b8']:.3f} | {dispatch} |"
        )
    lines.extend(
        [
            "",
            "## Static boundary",
            "",
            "All variants share one instruction stream. A common stall-15 BAR drains",
            "before a 17-result reservation run containing one admitted HMMA.16816",
            "anchor. Run controls are stall 1/yield 0/reuse 0 except one stall 15",
            "moved after reservation 7, 8, or 9. Static stall sums and terminal issue",
            "time are equal absent dynamic resource pressure.",
            "",
            "## Next slice",
            "",
            "Queue capacity remains unidentified: the reference reservation placement has",
            "no effect under the admitted timing. Move to the independently observable",
            "fixed-result versus L1TEX writeback-priority slice while retaining W8 b9 as",
            "a future candidate canary.",
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
        print(f"capacity analysis: {error}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
