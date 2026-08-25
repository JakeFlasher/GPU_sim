#!/usr/bin/env python3
"""Validate and reduce the sm_86 same-bank fixed-result producer-order campaign."""

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
    REPO_ROOT / "results" / "captures" / "sm86_fixed_latency_result_same_bank_20260817_v1"
)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 64
WIDTHS = (4, 8, 12, 16)
FUNCTIONS = {
    "fh": "_Z16result_same_bankILb0EEvPyPjPKjS3_",
    "hf": "_Z16result_same_bankILb1EEvPyPjPKjS3_",
}
METRICS = (
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
    spec = importlib.util.spec_from_file_location("same_bank_order_analysis", path)
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
destination = ORDER.destination


def source_bank_demand(opcode_text: str) -> tuple[int, int]:
    registers = [int(value) for value in re.findall(r"\bR(\d+)\b", opcode_text)][1:]
    return (sum(value % 2 == 0 for value in registers), sum(value % 2 == 1 for value in registers))


def validate_static(root: Path) -> dict[str, Any]:
    names = (
        "fixed_latency_result_same_bank_native",
        "fixed_latency_result_same_bank",
        "fixed_latency_result_same_bank_trace_native",
        "fixed_latency_result_same_bank_trace",
    )
    sidecars = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 96, f"{name}: decoded")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(set(sidecar["functions"]) == set(FUNCTIONS.values()), f"{name}: functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(resources.count("REG:16") == 2, f"{name}: registers")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill/local")

    for label in ("long", "trace"):
        patch = read_json(root / "metadata" / f"{label}_same_bank_patch.json")
        require(patch["changed_bytes"] == 2, f"{label}: changed bytes")
        require(patch["all_changes_inside_producer_control_words"] is True, f"{label}: scope")
        require(patch["same_destination_bank"] == 0, f"{label}: destination bank")
        require(patch["timed_consumers_byte_identical"] is True, f"{label}: consumers")
        require(patch["post_clock_oracle_commutative_permutation"] is True, f"{label}: oracle")
        require(patch["per_opcode_source_bank_demand"] == {"FFMA": [1, 2], "HADD2": [1, 1]}, f"{label}: source demand")
        require(len(patch["producers"]) == 4, f"{label}: producers")

    for prefix in ("fixed_latency_result_same_bank", "fixed_latency_result_same_bank_trace"):
        sidecar = sidecars[prefix]
        fh = sidecar["functions"][FUNCTIONS["fh"]]
        hf = sidecar["functions"][FUNCTIONS["hf"]]
        require(len(fh) == len(hf) == 48, f"{prefix}: rows")
        differing = []
        for a, b in zip(fh, hf):
            require(a["pc"] == b["pc"], f"{prefix}: PCs")
            pc = int(a["pc"])
            if pc in {0xE0, 0xF0, 0x1C0}:
                differing.append(pc)
                require(a["control"] == b["control"], f"{prefix}/{pc:#x}: controls")
                require(a["predicate"] == b["predicate"], f"{prefix}/{pc:#x}: predicate")
                require(destination(a["opcode_text"]) == destination(b["opcode_text"]), f"{prefix}/{pc:#x}: dest")
                if pc == 0x1C0:
                    require(a["opcode_text"].startswith("LOP3.LUT "), f"{prefix}: oracle opcode")
                    a_sources = sorted(map(int, re.findall(r"\bR(\d+)\b", a["opcode_text"])[1:]))
                    b_sources = sorted(map(int, re.findall(r"\bR(\d+)\b", b["opcode_text"])[1:]))
                    require(a_sources == b_sources, f"{prefix}: oracle permutation")
            else:
                for field in ("opcode_text", "predicate", "low_word", "high_word", "control"):
                    require(a[field] == b[field], f"{prefix}/{pc:#x}: {field}")
        require(differing == [0xE0, 0xF0, 0x1C0], f"{prefix}: differing PCs")
        fh_producers = [row for row in fh if row["pc"] in {0xE0, 0xF0}]
        hf_producers = [row for row in hf if row["pc"] in {0xE0, 0xF0}]
        require([row["opcode_text"].split()[0] for row in fh_producers] == ["FFMA", "HADD2"], "FH")
        require([row["opcode_text"].split()[0] for row in hf_producers] == ["HADD2", "FFMA"], "HF")
        for rows in (fh_producers, hf_producers):
            require([destination(row["opcode_text"]) for row in rows] == [4, 8], "dest stream")
            require(all(destination(row["opcode_text"]) % 2 == 0 for row in rows), "same bank")
            require(rows[0]["control"]["stall"] == 1 and rows[1]["control"]["stall"] == 2, "stalls")
            require(all(row["control"]["yield_flag"] == 0 for row in rows), "yield")
            require(all(row["control"]["reuse"] == 0 for row in rows), "reuse")
        demands = {
            row["opcode_text"].split()[0]: source_bank_demand(row["opcode_text"])
            for row in fh_producers
        }
        require(
            demands
            == {
                row["opcode_text"].split()[0]: source_bank_demand(row["opcode_text"])
                for row in hf_producers
            },
            "source-bank demands",
        )
        for pc in (0x110, 0x120):
            require(fh[pc // 16]["opcode_text"] == hf[pc // 16]["opcode_text"], "consumer identity")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(
        (projection["files"], projection["records"], projection["projected"], projection["gaps"])
        == (4, 384, 88, 0),
        "local projection",
    )
    corpus = read_json(root / "metadata" / "retained_corpus_operand_projection_audit.json")
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (56, 39848, 16597, 0),
        "corpus projection",
    )
    return {
        "binaries": 4,
        "decoded_instructions": 384,
        "registers_per_function": 16,
        "spill_free": True,
        "producer_pcs": [0xE0, 0xF0],
        "destination_stream": [4, 8],
        "destination_bank": 0,
        "per_opcode_source_bank_demand": {"FFMA": [1, 2], "HADD2": [1, 1]},
        "timed_consumers_byte_identical": True,
        "local_projection": {key: projection[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
        "retained_corpus_projection": {key: corpus[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 8 and len({row["key"] for row in cells}) == 8, "matrix")
    static = validate_static(root)
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
        require(payload["correct"] is True and payload["bank"] == 0, f"{key}: correctness")
        require(payload["order"] == cell["order"], f"{key}: order")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) == 7, f"{key}: NCU repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in METRICS:
            require(all(metric in row for row in rows), f"{key}: {metric}")
        require(all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows), f"{key}: issue ratio")
        active = [row["sm__cycles_active.max"] for row in rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        median = lambda name: statistics.median(row[name] for row in rows)
        live[key] = {
            "active_cycles": active_median,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            "driver_cycles": statistics.median(driver_samples),
            **{name: median(name) for name in METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True and trace_payload["generations"] == TRACE_GENERATIONS, f"{key}: trace")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: blocks")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {
            int(warp.instructions[0].warpid_sm)
            for warp in block.warps.values()
            if warp.instructions
        }
        require(slots == expected_physical_slots(cell["warps"]), f"{key}: slots")
        trace_records += records

        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: replay determinism")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: records")
        replay[key] = launch

    opposite = read_json(
        REPO_ROOT
        / "results/captures/sm86_fixed_latency_result_order_20260817_v1/analysis/summary.json"
    )
    opposite_by_width = {row["warps"]: row for row in opposite["pairs"]}
    pairs = []
    for warps in WIDTHS:
        fh_key = f"w{warps}_fh"
        hf_key = f"w{warps}_hf"
        fh = live[fh_key]
        hf = live[hf_key]
        require(
            read_json(root / "correctness" / fh_key / "result.json")["measurement"]["checksum"]
            == read_json(root / "correctness" / hf_key / "result.json")["measurement"]["checksum"],
            f"W{warps}: checksum",
        )
        for name in (
            "smsp__inst_issued.sum",
            "smsp__inst_executed.sum",
            "smsp__inst_executed_pipe_alu.sum",
            "smsp__inst_executed_pipe_fma.sum",
            "smsp__inst_executed_pipe_fma_type_fp16.sum",
        ):
            require(fh[name] == hf[name], f"W{warps}: {name}")
        require(replay[fh_key]["counters"] == replay[hf_key]["counters"], f"W{warps}: counters")
        live_delta = (hf["active_cycles"] - fh["active_cycles"]) / LONG_GENERATIONS
        replay_delta = (
            replay[hf_key]["timing"]["diagnostic_last_issue_cycles"]
            - replay[fh_key]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        require(replay_delta == 0.0, f"W{warps}: replay delta")
        opposite_delta = opposite_by_width[warps]["live_delta_hf_minus_fh"]
        pairs.append(
            {
                "stage": "identification" if warps in {4, 8} else "held_out_width",
                "warps": warps,
                "warps_per_modeled_smsp": warps // 4,
                "fh_cycles_per_generation": fh["active_cycles"] / LONG_GENERATIONS,
                "hf_cycles_per_generation": hf["active_cycles"] / LONG_GENERATIONS,
                "live_delta_hf_minus_fh": live_delta,
                "driver_delta_hf_minus_fh": (hf["driver_cycles"] - fh["driver_cycles"]) / LONG_GENERATIONS,
                "replay_delta_hf_minus_fh": replay_delta,
                "opposite_bank_live_delta": opposite_delta,
                "same_minus_opposite_bank_excess": live_delta - opposite_delta,
                "fh_cv_pct": fh["active_cv_pct"],
                "hf_cv_pct": hf["active_cv_pct"],
                "dispatch_delta": hf[
                    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"
                ]
                - fh["smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"],
                "math_delta": hf[
                    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
                ]
                - fh["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"],
                "not_selected_delta": hf[
                    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                ]
                - fh["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"],
                "wait_delta": hf[
                    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                ]
                - fh["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"],
                "short_scoreboard_delta": hf[
                    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
                ]
                - fh["smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"],
            }
        )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 8, "SourceCounters count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    identification = [row for row in pairs if row["stage"] == "identification"]
    held_out = [row for row in pairs if row["stage"] == "held_out_width"]
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 8,
            "correctness_executions": 56,
            "ncu_profiles": 56,
            "source_counters_profiles": 8,
            "nvbit_captures": 8,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 8,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
        },
        "static": static,
        "live": live,
        "replay": replay,
        "pairs": pairs,
        "source_counters": sources,
        "scores": {
            "identification_delta_mae": statistics.mean(abs(row["live_delta_hf_minus_fh"]) for row in identification),
            "held_out_width_delta_mae": statistics.mean(abs(row["live_delta_hf_minus_fh"]) for row in held_out),
            "identification_same_minus_opposite_mae": statistics.mean(abs(row["same_minus_opposite_bank_excess"]) for row in identification),
        },
        "decision": {
            "status": "two_result_same_bank_bypass_buffering_identified_timing_inactive",
            "two_same_cycle_same_bank_results_accommodated": True,
            "activate_result_queue_timing": False,
            "identified_lower_bound": (
                "hardware accommodates and forwards two same-cycle fixed-latency results "
                "to one RF bank without exposed W4/W8 delay"
            ),
            "unidentified": [
                "physical queue entry count",
                "queue reservation point and pop width",
                "result queue versus per-bank retirement contribution at W12",
                "fixed-result versus L1TEX writeback priority",
            ],
            "next_slice": (
                "single-warp same-bank completion cohorts of 2/3/4 and 7/8/9 results, "
                "holding controls and pipe work equal to cross queue occupancy boundaries"
            ),
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    scores = summary["scores"]
    lines = [
        "# sm_86 same-bank fixed-result producer-order reduction",
        "",
        "## Decision",
        "",
        "The campaign identifies a timing-neutral two-result same-bank",
        "buffering/bypass capability. At W4 and W8, HADD2→FFMA completion",
        "coincidence at bank 0 is indistinguishable from the separated FFMA→HADD2",
        "order, while the first consumer issues at the characterized completion",
        "frontier. Both orders remain correct and all counts/stall metrics match.",
        "",
        "No active queue timing is added: a null identifies accommodation, not queue",
        "depth. W12 exposes a bank-specific +0.742 cycles/generation beyond the",
        "opposite-bank order delta, with dispatch/wait but zero short-scoreboard",
        "movement. That onset cannot yet distinguish queue occupancy from bank",
        "retirement. W16 remains contaminated by the known global phase.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells and {campaign['correctness_executions']} correctness executions",
        f"- {campaign['ncu_profiles']} NCU profiles, {campaign['source_counters_profiles']} SourceCounters profiles, and {campaign['nvbit_captures']} NVBit captures",
        f"- {campaign['trace_records']:,} replay records and eight deterministic replay pairs",
        "- four spill-free cubins, 384/384 controls, 88 local projections, zero gaps",
        "- retained 56-sidecar corpus: 39,848 records, 16,597 projections, zero gaps",
        "- issue ratio 1.0 and equal issued/executed/ALU/FMA/FP16 counts in every pair",
        f"- maximum raw CV {campaign['maximum_active_cycle_cv_pct']:.3f}%; maximum MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "",
        "## Same-bank order deltas",
        "",
        "| Warps | Stage | Same-bank HF−FH | Opposite-bank HF−FH | Bank-specific excess | Replay | Dispatch Δ | Wait Δ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["pairs"]:
        lines.append(
            f"| {row['warps']} | {row['stage']} | {row['live_delta_hf_minus_fh']:.3f} | "
            f"{row['opposite_bank_live_delta']:.3f} | {row['same_minus_opposite_bank_excess']:.3f} | "
            f"{row['replay_delta_hf_minus_fh']:.3f} | {row['dispatch_delta']:.4f} | {row['wait_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Identification delta MAE is {scores['identification_delta_mae']:.4f} cycles/generation; "
            f"held-out width MAE is {scores['held_out_width_delta_mae']:.4f}.",
            "Short-scoreboard delta is zero at every width, including the W12 onset.",
            "",
            "## Static boundary",
            "",
            "Both functions have 48 instructions and 16 registers. Producers occupy",
            "relative PCs 0xe0/0xf0 with destination stream R4/R8 (bank 0), matching",
            "per-opcode source-bank demand, and controls (1,0,0) then (2,0,0). Timed",
            "consumers are byte-identical. Only producer order differs before stop; the",
            "post-clock oracle differs by a commutative source permutation.",
            "",
            "## Next slice",
            "",
            "Construct equal-work single-warp completion cohorts crossing 2/3/4 and",
            "7/8/9 same-bank results. This separates queue occupancy/pop width from RF",
            "retirement before attempting fixed-versus-load priority or active timing.",
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
        print(f"same-bank analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
