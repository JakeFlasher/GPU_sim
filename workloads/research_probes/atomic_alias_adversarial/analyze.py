#!/usr/bin/env python3
"""Validate and reduce the sm_86 adversarial atomic-alias campaign."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_atomic_alias_adversarial_20260818_v1"


def load_previous_analysis():
    path = HERE.parent / "atomic_completion_serialization/analyze.py"
    spec = importlib.util.spec_from_file_location("alias_previous_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREVIOUS = load_previous_analysis()
require = PREVIOUS.require
read_json = PREVIOUS.read_json
write_json = PREVIOUS.write_json
wide_ncu = PREVIOUS.wide_ncu
measurement = PREVIOUS.measurement
expected_physical_slots = PREVIOUS.expected_physical_slots
source_stalls = PREVIOUS.source_stalls


def address_multiset(entry: dict[str, Any]) -> list[int]:
    word_bytes = entry["width"] // 8
    if entry["pattern"] == "cross_warp":
        return [0] * 32
    groups = 32 // entry["alias"]
    stride = word_bytes if entry["pattern"] == "dense" else entry["stride_bytes"]
    addresses = []
    for lane in range(32):
        group = lane % groups if entry["scattered"] else lane // entry["alias"]
        addresses.append(group * stride)
    return addresses


def work_packages(entry: dict[str, Any], granularity: int) -> int:
    regions: dict[int, Counter[int]] = defaultdict(Counter)
    width = entry["width"] // 8
    for address in address_multiset(entry):
        first = address // granularity
        last = (address + width - 1) // granularity
        for region in range(first, last + 1):
            regions[region][address] += 1
    return sum(max(counts.values()) for counts in regions.values())


def median_max(payload: dict[str, Any]) -> float:
    return statistics.median(max(trial) for trial in payload["samples_cycles"])


def ape(predicted: float, observed: float) -> float:
    return abs(predicted / observed - 1.0) * 100.0


def validate_static(root: Path) -> dict[str, Any]:
    expected = {
        "ret32": (48, 16, "_Z18atomic_alias_probePyS_PjPKj", "ATOMG"),
        "red32": (40, 12, "_Z18atomic_alias_probePyS_PjPKj", "RED"),
        "ret64": (48, 18, "_Z18atomic_alias_probePyS_S_PKj", "ATOMG"),
        "red64": (40, 14, "_Z18atomic_alias_probePyS_S_PKj", "RED"),
    }
    decoded = 0
    for variant, (instructions, registers, function, opcode) in expected.items():
        for trace in (False, True):
            infix = "_trace" if trace else ""
            name = f"atomic_alias_adversarial{infix}_{variant}"
            sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
            require(sidecar["summary"]["total_decoded"] == instructions, f"{name}: controls")
            require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: decode gaps")
            require(set(sidecar["functions"]) == {function}, f"{name}: function")
            rows = sidecar["functions"][function]
            atomics = [row for row in rows if row["opcode_text"].startswith(("ATOMG", "RED"))]
            require(len(atomics) == 1 and atomics[0]["opcode_text"].startswith(opcode), f"{name}: atomic opcode")
            require(sum("SR_CLOCKLO" in row["opcode_text"] for row in rows) == 2, f"{name}: CLOCK bounds")
            require(any(row["opcode_text"].startswith("BAR.SYNC") for row in rows), f"{name}: pre-timing barrier")
            resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
            require(f"REG:{registers}" in resources, f"{name}: registers")
            require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spills")
            decoded += instructions
    local = read_json(root / "metadata/operand_projection_audit.json")
    retained = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require((local["files"], local["records"], local["projected"], local["gaps"]) == (8, 352, 36, 0), "local projection")
    require((retained["files"], retained["records"], retained["projected"], retained["gaps"]) == (177, 49_792, 18_544, 0), "retained projection")
    return {
        "binaries": 8,
        "decoded_instructions": decoded,
        "spill_free": True,
        "local_projection": {key: local[key] for key in ("files", "records", "projected", "gaps")},
        "retained_projection": {key: retained[key] for key in ("files", "records", "projected", "gaps")},
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata/matrix.json")["cells"]
    require(len(cells) == 138 and len({entry["key"] for entry in cells}) == 138, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live: dict[str, dict[str, float]] = {}
    clock: dict[str, float] = {}
    replay_clock: dict[str, float] = {}
    trace_records = 0
    ncu_profiles = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    maximum_clock_range = 0
    maximum_trace_range = 0
    required_metrics = (
        "sm__cycles_active.max",
        "smsp__inst_issued_per_issue_active.ratio",
        "smsp__inst_executed_op_generic_atom_dot_alu.sum",
        "smsp__inst_executed_op_global_red.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
        "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum",
        "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
        "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
        "lts__t_requests_srcunit_tex_op_red.sum",
        "lts__t_sectors_op_atom.sum",
        "lts__t_sectors_op_red.sum",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    )

    for entry in cells:
        key = entry["key"]
        sectors_per_instruction = work_packages(entry, 32)
        l2_requests_per_instruction = work_packages(entry, 128)
        long = read_json(root / "correctness" / key / "result.json")["measurement"]
        trace = read_json(root / "trace-correctness" / key / "result.json")["measurement"]
        for payload, generations, label in ((long, 1024, "long"), (trace, 64, "trace")):
            require(payload["correct"] is True, f"{key}: {label} correctness")
            require(payload["warps"] == entry["warps"], f"{key}: {label} warps")
            require(payload["generations"] == generations, f"{key}: {label} generations")
            require(len(payload["samples_cycles"]) == 7, f"{key}: {label} trials")
        maximum_clock_range = max(maximum_clock_range, max(max(x) for x in long["samples_cycles"]) - min(max(x) for x in long["samples_cycles"]))
        maximum_trace_range = max(maximum_trace_range, max(max(x) for x in trace["samples_cycles"]) - min(max(x) for x in trace["samples_cycles"]))
        clock[key] = median_max(trace)

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 3 if entry["warps"] == 16 else 1
        require(len(repeat_paths) == expected_repeats, f"{key}: NCU repeats")
        ncu_profiles += len(repeat_paths)
        ncu = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in required_metrics:
            require(all(metric in row for row in ncu), f"{key}: {metric}")
        require(all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in ncu), f"{key}: issue ratio")
        med = {metric: statistics.median(row[metric] for row in ncu) for metric in required_metrics}
        if len(ncu) > 1:
            samples = [row["sm__cycles_active.max"] for row in ncu]
            center = statistics.median(samples)
            maximum_cv = max(maximum_cv, statistics.stdev(samples) / statistics.mean(samples) * 100)
            maximum_mad = max(maximum_mad, statistics.median(abs(value - center) for value in samples) / center * 100)
        warp_instructions = entry["warps"] * 1024
        operation = "atom" if entry["returning"] else "red"
        opposite = "red" if entry["returning"] else "atom"
        instruction_metric = "smsp__inst_executed_op_generic_atom_dot_alu.sum" if entry["returning"] else "smsp__inst_executed_op_global_red.sum"
        opposite_metric = "smsp__inst_executed_op_global_red.sum" if entry["returning"] else "smsp__inst_executed_op_generic_atom_dot_alu.sum"
        require(med[instruction_metric] == warp_instructions and med[opposite_metric] == 0, f"{key}: operation count")
        require(med[f"l1tex__t_requests_pipe_lsu_mem_global_op_{operation}.sum"] == warp_instructions, f"{key}: L1 requests")
        require(med[f"l1tex__t_requests_pipe_lsu_mem_global_op_{opposite}.sum"] == 0, f"{key}: opposite requests")
        expected_sectors = warp_instructions * sectors_per_instruction
        expected_l2_requests = warp_instructions * l2_requests_per_instruction
        require(med[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{operation}.sum"] == expected_sectors, f"{key}: sectors")
        require(med[f"lts__t_sectors_op_{operation}.sum"] == expected_sectors, f"{key}: L2 sectors")
        l2_request_metric = "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum" if entry["returning"] else "lts__t_requests_srcunit_tex_op_red.sum"
        require(med[l2_request_metric] == expected_l2_requests, f"{key}: L2 requests")
        live[key] = {
            "active_cycles": med["sm__cycles_active.max"],
            "wavefronts": med[f"l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_{operation}.sum"],
            "l2_sectors": med[f"lts__t_sectors_op_{operation}.sum"],
            "l2_requests": med[l2_request_metric],
            "long_scoreboard": med["smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"],
            "sectors_per_instruction": sectors_per_instruction,
            "l2_requests_per_instruction": l2_requests_per_instruction,
        }

        capture = root / "nvbit" / key
        require(measurement(capture / "stdout.txt")["correct"] is True, f"{key}: trace correctness")
        for diagnostic in ("unsupported_memory_path", "unmapped_memory_range", "cross_allocation_lane_span"):
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: one CTA")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {int(warp.instructions[0].warpid_sm) for warp in block.warps.values() if warp.instructions}
        require(slots == expected_physical_slots(entry["warps"]), f"{key}: physical slots")
        trace_records += records
        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: deterministic replay")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full" and launch["counters"]["records_seen"] == records, f"{key}: replay coverage")
        expected_trace_sectors = entry["warps"] * 64 * sectors_per_instruction
        l1 = launch["counters"]["l1"]
        replay_sectors = l1["bypass_sectors"] if entry["returning"] else l1["reduction_sectors"]
        require(replay_sectors == expected_trace_sectors, f"{key}: replay sectors")
        clock_diag = launch["timing"]["readiness"]["clock_issue"]
        require(clock_diag["reads_issued"] == entry["warps"] * 2, f"{key}: clock reads")
        require(clock_diag["warps_with_interval"] == entry["warps"], f"{key}: clock intervals")
        replay_clock[key] = clock_diag["maximum_first_to_last_issue_cycles"]

    require(ncu_profiles == 278, "NCU profile total")
    sources = sorted((root / "ncu-source").glob("*/source.csv"))
    returns = sorted((root / "ncu-return").glob("*/ncu.csv"))
    require(len(sources) == 34, "SourceCounters total")
    require(len(returns) == 12, "return-path NCU total")
    source_summary = {path.parent.name: source_stalls(path) for path in sources}

    dense_w16_red = [entry for entry in cells if not entry["returning"] and entry["stage"] in {"dense_alias_width_transfer", "word_width_transfer"} and entry["warps"] == 16 and not entry["scattered"]]
    nonspecial = [entry for entry in dense_w16_red if entry["alias"] < 32]
    long_intercept = statistics.median(live[e["key"]]["active_cycles"] - live[e["key"]]["l2_sectors"] - live[e["key"]]["l2_requests"] for e in nonspecial)
    trace_intercept = statistics.median(clock[e["key"]] - 64 * e["warps"] * work_packages(e, 32) - 64 * e["warps"] * work_packages(e, 128) for e in nonspecial)

    def red_work(entry: dict[str, Any], generations: int) -> float:
        sectors = generations * entry["warps"] * work_packages(entry, 32)
        requests = generations * entry["warps"] * work_packages(entry, 128)
        if entry["alias"] == 32 and entry["pattern"] in {"dense", "cross_warp"}:
            requests /= 8
        return sectors + requests

    red_long_errors = [ape(long_intercept + red_work(e, 1024), live[e["key"]]["active_cycles"]) for e in dense_w16_red]
    red_trace_errors = [ape(trace_intercept + red_work(e, 64), clock[e["key"]]) for e in dense_w16_red]
    red_transfer = [entry for entry in cells if not entry["returning"] and entry["stage"] in {"dense_alias_width_transfer", "word_width_transfer"} and not entry["scattered"] and entry["warps"] > 1]
    red_floor_errors = []
    for entry in red_transfer:
        base = clock[f"w1_{entry['variant']}_dense_a{entry['alias']}_g"]
        red_floor_errors.append(ape(max(base, red_work(entry, 64)), clock[entry["key"]]))

    old_rows = [entry for entry in cells if entry["width"] == 32 and entry["pattern"] == "dense" and not entry["scattered"] and entry["warps"] > 1]
    old_errors = []
    for entry in old_rows:
        base = clock[f"w1_{entry['variant']}_dense_a{entry['alias']}_g"]
        transactions = entry["warps"] * 64 * work_packages(entry, 32)
        prediction = max(base, transactions * (2.0 if entry["returning"] else 1.125) + (268 if entry["returning"] else 0))
        old_errors.append(ape(prediction, clock[entry["key"]]))

    pair_deltas = []
    for returning in (True, False):
        op = "ret" if returning else "red"
        for alias in (1, 2, 4, 8, 16, 32):
            grouped = clock[f"w16_{op}32_dense_a{alias}_g"]
            scattered = clock[f"w16_{op}32_dense_a{alias}_s"]
            pair_deltas.append({"operation": op, "alias": alias, "delta_cycles": scattered - grouped, "relative_pct": (scattered / grouped - 1) * 100})

    rows = []
    for entry in cells:
        key = entry["key"]
        rows.append({
            "key": key,
            "stage": entry["stage"],
            "warps": entry["warps"],
            "width": entry["width"],
            "operation": entry["operation"],
            "alias": entry["alias"],
            "pattern": entry["pattern"],
            "stride_bytes": entry["stride_bytes"],
            "scattered": entry["scattered"],
            "clock_cycles": clock[key],
            "replay_clock_cycles": replay_clock[key],
            "replay_ape_pct": ape(replay_clock[key], clock[key]),
            **live[key],
        })
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 138,
            "long_correctness_executions": 966,
            "trace_correctness_executions": 966,
            "ncu_profiles": ncu_profiles,
            "source_counters_profiles": 34,
            "return_path_profiles": 12,
            "nvbit_captures": 138,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 138,
            "maximum_w16_active_cycle_cv_pct": maximum_cv,
            "maximum_w16_active_cycle_mad_pct": maximum_mad,
            "maximum_long_clock_range_cycles": maximum_clock_range,
            "maximum_trace_clock_range_cycles": maximum_trace_range,
        },
        "static": static,
        "cells": rows,
        "source_counters": source_summary,
        "old_sector_candidate": {"status": "rejected", "mape_pct": statistics.mean(old_errors), "max_ape_pct": max(old_errors)},
        "red_request_sector_model": {
            "status": "survives_in_sm_but_not_activated_without_multi_sm_mapping",
            "ordinary_rule": "cycles = intercept + L2 sectors + L2 requests",
            "full_warp_single_word_rule": "cycles = intercept + L2 sectors + L2 requests / 8",
            "long_intercept_cycles": long_intercept,
            "long_w16_mape_pct": statistics.mean(red_long_errors),
            "long_w16_max_ape_pct": max(red_long_errors),
            "trace_intercept_cycles": trace_intercept,
            "trace_w16_mape_pct": statistics.mean(red_trace_errors),
            "trace_w16_max_ape_pct": max(red_trace_errors),
            "trace_transfer_floor_mape_pct": statistics.mean(red_floor_errors),
            "trace_transfer_floor_max_ape_pct": max(red_floor_errors),
        },
        "lane_assignment_pairs": pair_deltas,
        "decision": {
            "status": "request_topology_and_clock_interval_promoted_timing_withheld",
            "promoted": [
                "exact ATOMG/RED 32-byte transaction sectors across alias degree and word width",
                "exact ATOMG/RED L2 request topology reconstructed from compact sectors",
                "exact first-to-last CLOCKLO issue interval diagnostic",
            ],
            "rejected": [
                "operation-only atomic sector service rates",
                "the prior two-cycle ATOMG and nine-over-eight-cycle RED sector candidate",
                "an extra lane-permutation correction",
            ],
            "deferred": [
                "ATOMG return-path timing, especially 64-bit alias-16",
                "global atomic-unit address mapping and cross-SM service",
            ],
            "next_slice": "multi-SM atomic-unit mapping and shared service",
        },
    }


def render(summary: dict[str, Any]) -> str:
    c = summary["campaign"]
    red = summary["red_request_sector_model"]
    old = summary["old_sector_candidate"]
    rows = {row["key"]: row for row in summary["cells"]}
    lines = [
        "# sm_86 adversarial atomic alias and request topology",
        "",
        "## Decision",
        "",
        "Promote exact atomic sector and L2-request topology plus exact CLOCK-boundary",
        "issue diagnostics. Reject the prior operation-only sector-rate timing model.",
        "No atomic timing queue is activated: RED is now identified inside one SM, but",
        "ATOMG return behavior and cross-SM atomic-unit mapping remain unresolved.",
        "",
        "## Campaign integrity",
        "",
        f"- {c['cells']} cells; {c['long_correctness_executions']} long and {c['trace_correctness_executions']} trace-length correctness executions",
        f"- {c['ncu_profiles']} general NCU, {c['source_counters_profiles']} SourceCounters, and {c['return_path_profiles']} return-path profiles",
        f"- {c['nvbit_captures']} NVBit captures, {c['trace_records']:,} records, and {c['byte_identical_replay_pairs']} byte-identical replay pairs",
        f"- maximum W16 NCU CV {c['maximum_w16_active_cycle_cv_pct']:.3f}% and MAD {c['maximum_w16_active_cycle_mad_pct']:.3f}%",
        "- eight spill-free static cubins; every address treatment uses one invariant cubin per operation/width",
        "",
        "## Dense 32-bit alias ladder, trace CLOCK cycles",
        "",
        "| Alias | RET W1 | RET W4 | RET W8 | RET W16 | RED W1 | RED W4 | RED W8 | RED W16 | Sectors/inst | L2 req/inst |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in (1, 2, 4, 8, 16, 32):
        ret = [rows[f"w{w}_ret32_dense_a{alias}_g"] for w in (1, 4, 8, 16)]
        red_rows = [rows[f"w{w}_red32_dense_a{alias}_g"] for w in (1, 4, 8, 16)]
        lines.append(
            f"| {alias} | " + " | ".join(f"{row['clock_cycles']:.0f}" for row in ret + red_rows)
            + f" | {ret[0]['sectors_per_instruction']:.0f} | {ret[0]['l2_requests_per_instruction']:.0f} |"
        )
    lines.extend([
        "",
        "Aliases 1/2/4 all generate four sectors, but one/two/four L2 requests.",
        "Their RED timing separates accordingly, directly falsifying a sector-only queue.",
        "Grouped/scattered pairs preserve the address multiset and remain effectively null,",
        "so lane ordering needs no correction.",
        "",
        "## Resolved in-SM RED law",
        "",
        "For alias degrees below 32, RED active cycles are one cycle per L2 sector",
        "plus one cycle per L2 request and a small static intercept. Full-warp",
        "single-word RED has an eight-request-per-cycle fast path:",
        "",
        f"- long W16 MAPE {red['long_w16_mape_pct']:.4f}% (max {red['long_w16_max_ape_pct']:.4f}%)",
        f"- trace W16 MAPE {red['trace_w16_mape_pct']:.3f}% (max {red['trace_w16_max_ape_pct']:.3f}%)",
        f"- W4/W8/W16 latency-floor transfer MAPE {red['trace_transfer_floor_mape_pct']:.3f}% (max {red['trace_transfer_floor_max_ape_pct']:.3f}%)",
        "",
        "The earlier two-cycle ATOMG / 9-over-8-cycle RED sector model is rejected:",
        f"its expanded dense-ladder MAPE is {old['mape_pct']:.3f}% and maximum error {old['max_ape_pct']:.3f}%.",
        "The 9/8 endpoint was an accidental identity: one sector cycle plus one-eighth",
        "of a full-broadcast request cycle.",
        "",
        "## Returning path",
        "",
        "ATOMG does not obey the RED fast path because every lane observes an old value.",
        "Return-path counters track L2 sectors and LSU writeback activity, and the 64-bit",
        "alias-16 case is slower than alias-32. A scalar return constant would therefore",
        "be another non-transferable correction. ATOMG timing remains inactive.",
        "",
        "## Replay changes",
        "",
        "Replay now retains exact alias-expanded sectors, reconstructs exact L2 request",
        "count per atomic instruction, rejects incoherent request/sector pairs, and reports",
        "the maximum first-to-last `CS2R ..., SR_CLOCKLO` issue interval. These changes",
        "are timing-neutral and remove two confounds from future queue work.",
        "",
        "## Next slice",
        "",
        "The surviving RED law is local to the one-CTA campaign. The next experiment",
        "uses 1–168 one-warp CTAs and fine address spacing to identify distributed L2",
        "atomic-unit mapping and cross-SM service before any queue can be activated.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        summary = reduce(args.root.resolve())
        write_json(args.root.resolve() / "analysis/summary.json", summary)
        path = args.root.resolve() / "analysis/README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(summary), encoding="utf-8")
        print(path)
    except (OSError, RuntimeError, json.JSONDecodeError, csv.Error) as error:
        print(f"adversarial atomic analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
