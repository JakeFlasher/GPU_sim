#!/usr/bin/env python3
"""Validate and reduce the sm_86 atomic completion/serialization campaign."""

from __future__ import annotations

import argparse
import csv
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
    / "results/captures/sm86_atomic_completion_serialization_20260818_v1"
)
FUNCTION = "_Z12atomic_probePyS_Pj"
LAYOUT_SECTORS = {
    "lane_line": 32,
    "lane_word": 4,
    "warp_word": 32,
    "global_word": 1,
}
LAYOUT_WAVEFRONTS = {
    "lane_line": 8,
    "lane_word": 1,
    "warp_word": 32,
    "global_word": 1,
}
LAYOUT_L2_REQUESTS = {
    "lane_line": 32,
    "lane_word": 1,
    "warp_word": 32,
    "global_word": 1,
}
REQUIRED_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__inst_executed_op_generic_atom_dot_alu.sum",
    "smsp__inst_executed_op_global_red.sum",
    "sm__sass_inst_executed_op_global_atom.sum",
    "sm__sass_inst_executed_op_global_red.sum",
    "smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
    "lts__t_requests_srcunit_tex_op_atom_dot_alu_lookup_hit.sum",
    "lts__t_requests_srcunit_tex_op_red.sum",
    "lts__t_requests_srcunit_tex_op_red_lookup_hit.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
)


def load_predecessor_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("atomic_result_order_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_predecessor_analysis()
require = ORDER.require
read_json = ORDER.read_json
write_json = ORDER.write_json
wide_ncu = ORDER.wide_ncu
measurement = ORDER.measurement
expected_physical_slots = ORDER.expected_physical_slots


def validate_static(root: Path) -> dict[str, Any]:
    decoded = 0
    for variant in (
        "ret_lane_line",
        "ret_lane_word",
        "ret_warp_word",
        "ret_global_word",
        "red_lane_line",
        "red_lane_word",
        "red_warp_word",
        "red_global_word",
    ):
        expected_instructions = 48 if variant == "ret_global_word" else 40
        expected_registers = (
            18 if variant == "ret_global_word" else 14 if variant.startswith("ret_") else 12
        )
        for trace in (False, True):
            infix = "_trace" if trace else ""
            name = f"atomic_completion_serialization{infix}_{variant}"
            sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
            require(
                sidecar["summary"]["total_decoded"] == expected_instructions,
                f"{name}: decoded controls",
            )
            require(
                sidecar["summary"]["total_unknown_encoding"] == 0,
                f"{name}: unknown controls",
            )
            require(set(sidecar["functions"]) == {FUNCTION}, f"{name}: function")
            instructions = sidecar["functions"][FUNCTION]
            atomic = [
                entry
                for entry in instructions
                if entry["opcode_text"].startswith(("ATOMG", "RED"))
            ]
            require(len(atomic) == 1, f"{name}: one atomic instruction")
            expected_opcode = "ATOMG" if variant.startswith("ret_") else "RED"
            require(
                atomic[0]["opcode_text"].startswith(expected_opcode),
                f"{name}: {expected_opcode}",
            )
            resources = (root / "sass" / f"{name}.resources.txt").read_text(
                encoding="utf-8"
            )
            require(f"REG:{expected_registers}" in resources, f"{name}: registers")
            require(
                not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources),
                f"{name}: spills",
            )
            decoded += expected_instructions

    local = read_json(root / "metadata/operand_projection_audit.json")
    retained = read_json(root / "metadata/retained_corpus_operand_projection_audit.json")
    require(
        (local["files"], local["records"], local["projected"], local["gaps"])
        == (16, 656, 82, 0),
        "local operand projection",
    )
    require(
        (retained["files"], retained["records"], retained["projected"], retained["gaps"])
        == (169, 49_440, 18_508, 0),
        "retained operand projection",
    )
    return {
        "binaries": 16,
        "decoded_instructions": decoded,
        "spill_free": True,
        "primary_registers": {"ATOMG": 14, "RED": 12},
        "compiler_aggregated_global_return_registers": 18,
        "local_projection": {
            key: local[key] for key in ("files", "records", "projected", "gaps")
        },
        "retained_projection": {
            key: retained[key]
            for key in ("files", "records", "projected", "gaps")
        },
    }


def source_stalls(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: empty SourceCounters report")
    documents = [dict(zip(rows[1], row)) for row in rows[2:]]
    result: dict[str, Any] = {}
    for short, column in (
        ("long_scoreboard", "stall_long_sb (Not Issued)"),
        ("lg_throttle", "stall_lg (Not Issued)"),
        ("mio_throttle", "stall_mio (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
    ):
        values = []
        for row in documents:
            try:
                count = int(float(row.get(column) or 0))
            except ValueError:
                count = 0
            if count:
                values.append((count, row.get("Source", "").strip(), row.get("Address", "")))
        values.sort(reverse=True)
        result[short] = {
            "samples": sum(value[0] for value in values),
            "top": [
                {"samples": value, "source": source, "address": address}
                for value, source, address in values[:5]
            ],
        }
    return result


def median_max_cycles(payload: dict[str, Any]) -> float:
    return statistics.median(max(trial) for trial in payload["samples_cycles"])


def relative_error(predicted: float, observed: float) -> float:
    return abs(predicted / observed - 1.0) * 100.0


def fit_service_candidate(
    cells: list[dict[str, Any]], hardware: dict[str, float]
) -> dict[str, Any]:
    primary = [entry for entry in cells if entry["layout"] != "global_word"]
    calibration = {
        entry["key"]: hardware[entry["key"]]
        for entry in cells
        if entry["warps"] == 1
    }
    train = [entry for entry in primary if entry["warps"] in {4, 8}]
    holdout = [entry for entry in primary if entry["warps"] == 16]

    def predict(entry: dict[str, Any], atom_eighths: int, red_eighths: int, tail: int) -> float:
        base = calibration[f"w1_{entry['variant']}"]
        transactions = 64 * entry["warps"] * LAYOUT_SECTORS[entry["layout"]]
        eighths = atom_eighths if entry["returning"] else red_eighths
        service = math.ceil(transactions * eighths / 8)
        if entry["returning"]:
            service += tail
        return max(base, service)

    best = None
    for atom_eighths in range(12, 25):
        for red_eighths in range(6, 17):
            for return_tail in range(200, 361, 4):
                errors = [
                    relative_error(
                        predict(entry, atom_eighths, red_eighths, return_tail),
                        hardware[entry["key"]],
                    )
                    for entry in train
                ]
                score = statistics.mean(errors)
                candidate = (score, atom_eighths, red_eighths, return_tail)
                if best is None or candidate < best:
                    best = candidate
    assert best is not None
    train_mape, atom_eighths, red_eighths, return_tail = best

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
        errors = [
            relative_error(
                predict(entry, atom_eighths, red_eighths, return_tail),
                hardware[entry["key"]],
            )
            for entry in rows
        ]
        return {"mape_pct": statistics.mean(errors), "max_ape_pct": max(errors)}

    return {
        "status": "strong_offline_candidate_not_activated",
        "calibration": "W1 per static variant",
        "training_widths": [4, 8],
        "holdout_width": 16,
        "atomic_return_service_cycles_per_sector": atom_eighths / 8,
        "reduction_service_cycles_per_sector": red_eighths / 8,
        "atomic_return_tail_cycles": return_tail,
        "training": summarize(train),
        "holdout": summarize(holdout),
        "all_primary_transfer": summarize(
            [entry for entry in primary if entry["warps"] > 1]
        ),
        "predictions": {
            entry["key"]: predict(entry, atom_eighths, red_eighths, return_tail)
            for entry in cells
            if entry["warps"] > 1
        },
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata/matrix.json")["cells"]
    require(len(cells) == 32 and len({entry["key"] for entry in cells}) == 32, "matrix")
    static = validate_static(root)

    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    hardware_trace: dict[str, float] = {}
    replay_cycles: dict[str, float] = {}
    live: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    trace_records = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    maximum_correctness_range = 0
    maximum_trace_range = 0

    for entry in cells:
        key = entry["key"]
        expected_operations = entry["warps"] * 32 * 1024
        expected_trace_operations = entry["warps"] * 32 * 64
        expected_warp_instructions = entry["warps"] * 1024
        expected_trace_sectors = (
            entry["warps"] * 64 * LAYOUT_SECTORS[entry["layout"]]
        )

        long_payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        trace_payload = read_json(root / "trace-correctness" / key / "result.json")[
            "measurement"
        ]
        for payload, generations, operations, label in (
            (long_payload, 1024, expected_operations, "long"),
            (trace_payload, 64, expected_trace_operations, "trace"),
        ):
            require(payload["correct"] is True, f"{key}: {label} correctness")
            require(payload["warps"] == entry["warps"], f"{key}: {label} warps")
            require(payload["generations"] == generations, f"{key}: {label} generations")
            require(payload["operations"] == operations, f"{key}: {label} operations")
            require(len(payload["samples_cycles"]) == 7, f"{key}: {label} trials")
        maximum_correctness_range = max(
            maximum_correctness_range,
            max(max(trial) for trial in long_payload["samples_cycles"])
            - min(max(trial) for trial in long_payload["samples_cycles"]),
        )
        maximum_trace_range = max(
            maximum_trace_range,
            max(max(trial) for trial in trace_payload["samples_cycles"])
            - min(max(trial) for trial in trace_payload["samples_cycles"]),
        )
        hardware_trace[key] = median_max_cycles(trace_payload)

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) == 3, f"{key}: NCU repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in REQUIRED_METRICS:
            require(all(metric in row for row in rows), f"{key}: {metric}")
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows),
            f"{key}: issue ratio",
        )
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        medians = {
            metric: statistics.median(row[metric] for row in rows)
            for metric in REQUIRED_METRICS[1:]
        }

        operation = "atom" if entry["returning"] else "red"
        other = "red" if entry["returning"] else "atom"
        require(
            medians[
                "smsp__inst_executed_op_generic_atom_dot_alu.sum"
                if entry["returning"]
                else "smsp__inst_executed_op_global_red.sum"
            ]
            == expected_warp_instructions,
            f"{key}: operation instruction count",
        )
        require(
            medians[
                "smsp__inst_executed_op_global_red.sum"
                if entry["returning"]
                else "smsp__inst_executed_op_generic_atom_dot_alu.sum"
            ]
            == 0,
            f"{key}: ATOMG/RED separation",
        )
        require(
            medians[f"l1tex__t_requests_pipe_lsu_mem_global_op_{operation}.sum"]
            == expected_warp_instructions,
            f"{key}: L1TEX request count",
        )
        require(
            medians[f"l1tex__t_requests_pipe_lsu_mem_global_op_{other}.sum"] == 0,
            f"{key}: opposite L1TEX request count",
        )
        expected_sectors = expected_warp_instructions * LAYOUT_SECTORS[entry["layout"]]
        expected_wavefronts = (
            expected_warp_instructions * LAYOUT_WAVEFRONTS[entry["layout"]]
        )
        expected_l2_requests = (
            expected_warp_instructions * LAYOUT_L2_REQUESTS[entry["layout"]]
        )
        require(
            medians[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{operation}.sum"]
            == expected_sectors,
            f"{key}: atomic sector transactions",
        )
        require(
            medians[f"l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_{operation}.sum"]
            == expected_wavefronts,
            f"{key}: atomic wavefronts",
        )
        l2_metric = (
            "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum"
            if entry["returning"]
            else "lts__t_requests_srcunit_tex_op_red.sum"
        )
        require(medians[l2_metric] == expected_l2_requests, f"{key}: L2 atomic requests")
        live[key] = {
            "clock_cycles": median_max_cycles(long_payload),
            "active_cycles": active_median,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            "atomic_warp_instructions": expected_warp_instructions,
            "l1tex_requests": expected_warp_instructions,
            "l1tex_sectors": expected_sectors,
            "l1tex_wavefronts": expected_wavefronts,
            "l2_requests": expected_l2_requests,
            "long_scoreboard_per_issue_active": medians[
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
            ],
        }

        capture = root / "nvbit" / key
        traced = measurement(capture / "stdout.txt")
        require(traced["correct"] is True, f"{key}: traced correctness")
        for diagnostic in (
            "unsupported_memory_path",
            "unmapped_memory_range",
            "cross_allocation_lane_span",
        ):
            require(
                read_json(capture / "extra_info" / f"{diagnostic}.json") == [],
                f"{key}: {diagnostic}",
            )
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: one threadblock")
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
        require(launch["coverage"]["kind"] == "full", f"{key}: replay coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: replay records")
        l1 = launch["counters"]["l1"]
        require(l1 is not None, f"{key}: L1 counters")
        observed_trace_sectors = (
            l1["bypass_sectors"] if entry["returning"] else l1["reduction_sectors"]
        )
        require(
            observed_trace_sectors == expected_trace_sectors,
            f"{key}: replay atomic sector projection",
        )
        replay_cycles[key] = launch["timing"]["diagnostic_last_issue_cycles"]
        replay[key] = {
            "records": records,
            "atomic_sector_transactions": observed_trace_sectors,
            "last_issue_cycles": replay_cycles[key],
        }

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 8, "SourceCounters count")
    sources = {path.parent.name: source_stalls(path) for path in source_paths}
    for variant in ("ret_lane_line", "ret_lane_word", "ret_warp_word"):
        top = sources[f"w16_{variant}"]["long_scoreboard"]["top"]
        require(top and "IADD3" in top[0]["source"], f"{variant}: return consumer stall")
    for variant in ("red_lane_line", "red_warp_word"):
        top = sources[f"w16_{variant}"]["long_scoreboard"]["top"]
        require(top and "BRA" in top[0]["source"], f"{variant}: read-barrier branch stall")

    current_primary_transfer = [
        entry for entry in cells if entry["layout"] != "global_word" and entry["warps"] > 1
    ]
    current_errors = [
        relative_error(replay_cycles[entry["key"]], hardware_trace[entry["key"]])
        for entry in current_primary_transfer
    ]
    candidate = fit_service_candidate(cells, hardware_trace)

    rows = []
    for entry in cells:
        key = entry["key"]
        rows.append(
            {
                "key": key,
                "warps": entry["warps"],
                "operation": entry["operation"],
                "layout": entry["layout"],
                "trace_clock_cycles": hardware_trace[key],
                "replay_last_issue_cycles": replay_cycles[key],
                "replay_ape_pct": relative_error(replay_cycles[key], hardware_trace[key]),
                "l1tex_sectors_per_instruction": LAYOUT_SECTORS[entry["layout"]],
                "l1tex_wavefronts_per_instruction": LAYOUT_WAVEFRONTS[entry["layout"]],
                "l2_requests_per_instruction": LAYOUT_L2_REQUESTS[entry["layout"]],
                "long_scoreboard_per_issue_active": live[key][
                    "long_scoreboard_per_issue_active"
                ],
                "candidate_cycles": candidate["predictions"].get(key),
            }
        )

    return {
        "schema_version": 1,
        "campaign": {
            "cells": 32,
            "long_correctness_executions": 224,
            "trace_correctness_executions": 224,
            "ncu_profiles": 96,
            "source_counters_profiles": 8,
            "nvbit_captures": 32,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 32,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "maximum_long_clock_range_cycles": maximum_correctness_range,
            "maximum_trace_clock_range_cycles": maximum_trace_range,
            "all_issue_ratios_one": True,
        },
        "static": static,
        "cells": rows,
        "source_counters": sources,
        "current_replay": {
            "primary_transfer_mape_pct": statistics.mean(current_errors),
            "primary_transfer_max_ape_pct": max(current_errors),
            "atomic_sector_projection": "activated_and_exact_on_all_32_cells",
            "timing_service": "not_activated",
        },
        "service_candidate": candidate,
        "decision": {
            "status": "sector_projection_promoted_service_candidate_deferred",
            "promoted": [
                "global atomic alias-aware 32-byte sector transaction projection",
                "returning ATOMG and non-returning RED remain distinct request semantics",
            ],
            "rejected": [
                "a generic same-word or same-line serialization penalty",
                "the compiler-aggregated global-word shape as a causal contention treatment",
            ],
            "deferred": [
                "per-SM atomic sector service queue",
                "dynamic RED read-barrier completion from atomic service",
                "ATOMG response tail after atomic sector service",
            ],
            "evidence": [
                "warp_word and lane_line have equal 32-sector traffic and equal cycles despite opposite aliasing",
                "lane_word has four sectors and is much faster at width",
                "global_word is compiler warp aggregation with one predicated atomic lane and is transfer-only",
                "SourceCounters place ATOMG delay on its dependent IADD3 and sector-heavy RED delay on the read-barrier branch",
                "the W4/W8-fitted service candidate transfers to W16 at low error but partial alias degree and exact replay interval are not yet identified",
            ],
            "next_slice": (
                "compiler-resistant runtime address tables, partial alias degree sweep, "
                "exact simulator CLOCK-boundary timing, and adversarial cross-check of "
                "the 2-cycle ATOMG / 9-over-8-cycle RED sector-service candidate"
            ),
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    current = summary["current_replay"]
    candidate = summary["service_candidate"]
    by_key = {entry["key"]: entry for entry in summary["cells"]}
    lines = [
        "# sm_86 atomic completion and sector serialization",
        "",
        "## Decision",
        "",
        "Promote alias-aware atomic sector transaction formation, but do not yet",
        "activate an atomic timing queue. A warp instruction whose 32 lanes alias",
        "one word generates 32 serialized 32-byte atomic sectors; 32 distinct words",
        "inside one 128-byte line generate four sectors. Replay previously collapsed",
        "the first case to one unique sector and now reproduces every campaign sector",
        "count exactly.",
        "",
        "The measurements reject a generic same-address penalty beyond those sector",
        "transactions: compiler-resistant `warp_word` and `lane_line` have opposite",
        "aliasing but identical 32-sector traffic and essentially identical cycles.",
        "The full-CTA global-word shape is compiler warp aggregation and is transfer-only.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells; {campaign['long_correctness_executions']} long and {campaign['trace_correctness_executions']} trace-length correctness executions",
        f"- {campaign['ncu_profiles']} NCU profiles, {campaign['source_counters_profiles']} SourceCounters profiles, and {campaign['nvbit_captures']} NVBit captures",
        f"- {campaign['trace_records']:,} dynamic records and {campaign['byte_identical_replay_pairs']} byte-identical replay pairs",
        f"- maximum active-cycle CV {campaign['maximum_active_cycle_cv_pct']:.3f}% and MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "- 16 spill-free cubins, 656 decoded instructions, 82 operand projections, zero local gaps",
        "- retained corpus: 169 sidecars, 49,440 records, 18,508 projections, zero gaps",
        "- every NCU profile has issue ratio 1.0 and exact ATOMG-versus-RED classification",
        "",
        "## Exact transaction formation",
        "",
        "| Layout | Lane relation | L1TEX sectors / instruction | T wavefronts / instruction | L2 requests / instruction |",
        "|---|---|---:|---:|---:|",
        "| `lane_line` | distinct word and 128 B line per lane | 32 | 8 | 32 |",
        "| `lane_word` | distinct word, four sectors in one line | 4 | 1 | 1 |",
        "| `warp_word` | all 32 lanes alias one word | 32 | 32 | 32 |",
        "| `global_word` | compiler elects one lane per warp | 1 | 1 | 1 |",
        "",
        "## Trace-length timing (64 generations)",
        "",
        "| Operation/layout | W1 | W4 | W8 | W16 | W16 long-scoreboard ratio | Current replay W16 | Candidate W16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for operation in ("ret", "red"):
        for layout in ("lane_line", "lane_word", "warp_word", "global_word"):
            variant = f"{operation}_{layout}"
            cells = [by_key[f"w{warps}_{variant}"] for warps in (1, 4, 8, 16)]
            candidate_w16 = cells[-1]["candidate_cycles"]
            lines.append(
                f"| `{operation.upper()} {layout}` | "
                + " | ".join(f"{entry['trace_clock_cycles']:.0f}" for entry in cells)
                + f" | {cells[-1]['long_scoreboard_per_issue_active']:.2f}"
                + f" | {cells[-1]['replay_last_issue_cycles']:.0f}"
                + f" | {candidate_w16:.0f} |"
            )
    lines.extend(
        [
            "",
            "The current replay has no atomic service/backpressure queue. On the 18",
            f"primary W4/W8/W16 transfer cells its timing MAPE is {current['primary_transfer_mape_pct']:.3f}%",
            f"(maximum {current['primary_transfer_max_ape_pct']:.3f}%). The values compare",
            "hardware's CLOCK-bounded loop with replay's last-issue horizon and therefore",
            "remain a directional diagnostic, not a promoted scalar predictor.",
            "",
            "## Offline service candidate",
            "",
            "W1 supplies a per-static-variant latency baseline. A deterministic sector",
            "server is then fit only on W4/W8 and evaluated on W16:",
            "",
            f"- returning ATOMG: {candidate['atomic_return_service_cycles_per_sector']:.3f} cycles per sector plus a {candidate['atomic_return_tail_cycles']}-cycle response tail",
            f"- non-returning RED: {candidate['reduction_service_cycles_per_sector']:.3f} cycles per sector",
            f"- W4/W8 training MAPE {candidate['training']['mape_pct']:.3f}%, max {candidate['training']['max_ape_pct']:.3f}%",
            f"- W16 holdout MAPE {candidate['holdout']['mape_pct']:.3f}%, max {candidate['holdout']['max_ape_pct']:.3f}%",
            "",
            "This candidate is deliberately not active. The campaign has only alias",
            "degree 1 and 32 endpoints, its global-word arm is compiler-transformed,",
            "and simulator timing does not yet expose the exact two CLOCK boundary PCs.",
            "",
            "## Source attribution",
            "",
            "Returning ATOMG stalls sample at the dependent `IADD3` that waits on its",
            "write barrier. Sector-heavy RED stalls sample at the loop `BRA` that waits",
            "on RED's read barrier. LG and MIO throttle are effectively zero; this rules",
            "out relabeling the observed delay as a generic LSU issue throttle.",
            "",
            "## Next adversarial slice",
            "",
            "Use runtime-loaded address tables to hold SASS constant while sweeping alias",
            "degrees 1/2/4/8/16/32, add exact replay issue timestamps at the two CLOCK PCs,",
            "check 32-bit and 64-bit atomics plus address permutations, and retest the",
            "2-cycle ATOMG / 9-over-8-cycle RED service rates before activating timing.",
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
    root = args.root.resolve()
    try:
        summary = reduce(root)
        write_json(root / "analysis/summary.json", summary)
        report = render(summary)
        (root / "analysis").mkdir(parents=True, exist_ok=True)
        (root / "analysis/README.md").write_text(report, encoding="utf-8")
        print(root / "analysis/README.md")
    except (OSError, RuntimeError, json.JSONDecodeError, csv.Error) as error:
        print(f"atomic analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
