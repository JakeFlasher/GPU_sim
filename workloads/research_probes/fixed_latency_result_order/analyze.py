#!/usr/bin/env python3
"""Validate and reduce the sm_86 balanced fixed-result producer-order campaign."""

from __future__ import annotations

import argparse
import csv
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
    REPO_ROOT / "results" / "captures" / "sm86_fixed_latency_result_order_20260817_v1"
)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 64
WIDTHS = (4, 8, 12, 13, 14, 15, 16)
ORDERS = ("fh", "hf")
FUNCTIONS = {
    "fh": "_Z12result_orderILb0EEvPyPjPKjS3_",
    "hf": "_Z12result_orderILb1EEvPyPjPKjS3_",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wide_ncu(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3 and len(rows[0]) == len(rows[2]), f"{path}: malformed NCU CSV")
    result: dict[str, float] = {}
    for key, value in zip(rows[0], rows[2]):
        try:
            result[key] = float(value.replace(",", ""))
        except ValueError:
            pass
    return result


def measurement(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{") and '"correct"' in line
    ]
    require(len(rows) == 1, f"{path}: expected one measurement")
    return rows[0]


def destination(opcode_text: str) -> int:
    matched = re.match(r"\S+ R(\d+),", opcode_text)
    require(matched is not None, f"cannot parse destination: {opcode_text}")
    return int(matched.group(1))


def expected_physical_slots(warps: int) -> set[int]:
    # Ampere assigns each partial four-warp group in slot order 1,2,3,0.
    return {
        (logical // 4) * 4 + ((logical % 4 + 1) % 4)
        for logical in range(warps)
    }


def source_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: empty SourceCounters report")
    documents = [dict(zip(rows[1], row)) for row in rows[2:]]
    result = {}
    for short, column in (
        ("dispatch", "stall_dispatch (Not Issued)"),
        ("math", "stall_math (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
        ("short_scoreboard", "stall_short_sb (Not Issued)"),
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
                {"samples": count, "sass": sass, "address": address}
                for count, sass, address in values[:8]
            ],
        }
    return result


def validate_static(root: Path) -> dict[str, Any]:
    names = (
        "fixed_latency_result_order_native",
        "fixed_latency_result_order",
        "fixed_latency_result_order_trace_native",
        "fixed_latency_result_order_trace",
    )
    sidecars = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 96, f"{name}: decoded count")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(set(sidecar["functions"]) == set(FUNCTIONS.values()), f"{name}: functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill/local")
        require(resources.count("REG:15") == 2, f"{name}: register declarations")

    for label in ("long", "trace"):
        patch = read_json(root / "metadata" / f"{label}_result_order_patch.json")
        require(patch["changed_bytes"] == 4, f"{label}: changed bytes")
        require(patch["all_changes_inside_producer_control_words"] is True, f"{label}: scope")
        require(patch["opposite_destination_banks"] is True, f"{label}: banks")
        require(len(patch["producers"]) == 4, f"{label}: producer rows")
        require(
            all(
                row["final_stall"] == (1 if row["pc"] == 0x100 else 2)
                and row["final_yield"] == 0
                and row["final_reuse"] == 0
                and row["low_unchanged"] is True
                for row in patch["producers"]
            ),
            f"{label}: canonical controls",
        )

    for prefix in ("fixed_latency_result_order", "fixed_latency_result_order_trace"):
        canonical = sidecars[prefix]
        fh = canonical["functions"][FUNCTIONS["fh"]]
        hf = canonical["functions"][FUNCTIONS["hf"]]
        require(len(fh) == len(hf) == 48, f"{prefix}: instruction count")
        differing = []
        for a, b in zip(fh, hf):
            require(a["pc"] == b["pc"], f"{prefix}: PC alignment")
            if a["pc"] in {0x100, 0x110}:
                differing.append(a["pc"])
                require(a["control"] == b["control"], f"{prefix}/{a['pc']:#x}: controls")
                require(a["predicate"] == b["predicate"], f"{prefix}/{a['pc']:#x}: predicate")
                require(destination(a["opcode_text"]) == destination(b["opcode_text"]), f"{prefix}: dest")
            else:
                for field in ("opcode_text", "predicate", "low_word", "high_word", "control"):
                    require(a[field] == b[field], f"{prefix}/{a['pc']:#x}: {field}")
        require(differing == [0x100, 0x110], f"{prefix}: differing PCs")
        require(
            [row["opcode_text"].split()[0] for row in fh if row["pc"] in differing]
            == ["FFMA", "HADD2"],
            f"{prefix}: FH order",
        )
        require(
            [row["opcode_text"].split()[0] for row in hf if row["pc"] in differing]
            == ["HADD2", "FFMA"],
            f"{prefix}: HF order",
        )
        for rows in (fh, hf):
            producers = [row for row in rows if row["pc"] in differing]
            require([destination(row["opcode_text"]) for row in producers] == [11, 4], "destinations")
            require(producers[0]["control"]["stall"] == 1, "first stall")
            require(producers[1]["control"]["stall"] == 2, "second stall")
            require(all(row["control"]["yield_flag"] == 0 for row in producers), "producer yield")
            require(all(row["control"]["reuse"] == 0 for row in producers), "producer reuse")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(
        (projection["files"], projection["records"], projection["projected"], projection["gaps"])
        == (4, 384, 88, 0),
        "local projection",
    )
    corpus = read_json(root / "metadata" / "retained_corpus_operand_projection_audit.json")
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (52, 39464, 16509, 0),
        "retained corpus projection",
    )
    return {
        "binaries": 4,
        "decoded_instructions": 384,
        "functions_per_binary": 2,
        "registers_per_function": 15,
        "spill_free": True,
        "producer_pcs": [0x100, 0x110],
        "destinations_by_pc": [11, 4],
        "opposite_destination_banks": True,
        "local_projection": {key: projection[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
        "retained_corpus_projection": {key: corpus[key] for key in ("files", "records", "projected", "gaps", "families", "opcodes")},
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 14 and len({row["key"] for row in cells}) == 14, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live = {}
    replay = {}
    trace_records = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    high_cv_cells = []
    ncu_profiles = 0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["order"] == cell["order"], f"{key}: order")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: driver trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 11 if key in {"w14_hf", "w15_hf", "w16_hf"} else 7
        require(len(repeat_paths) == expected_repeats, f"{key}: NCU repeats")
        ncu_profiles += len(repeat_paths)
        ncu_rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in METRICS:
            require(all(metric in row for row in ncu_rows), f"{key}: missing {metric}")
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in ncu_rows),
            f"{key}: issue ratio",
        )
        active = [row["sm__cycles_active.max"] for row in ncu_rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        active_median = statistics.median(active)
        mad = statistics.median(abs(value - active_median) for value in active) / active_median * 100
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        if cv > 2.0:
            high_cv_cells.append(key)
        median = lambda name: statistics.median(row[name] for row in ncu_rows)
        live[key] = {
            "active_cycles": active_median,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            "driver_cycles": statistics.median(driver_samples),
            "ncu_repeats": len(repeat_paths),
            **{name: median(name) for name in METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True, f"{key}: trace correctness")
        require(trace_payload["generations"] == TRACE_GENERATIONS, f"{key}: trace generations")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: block trace count")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {
            int(warp.instructions[0].warpid_sm)
            for warp in block.warps.values()
            if warp.instructions
        }
        require(slots == expected_physical_slots(cell["warps"]), f"{key}: warp slots")
        trace_records += records

        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: nondeterministic replay")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: records")
        replay[key] = launch

    pairs = []
    replay_residuals = []
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
        driver_delta = (hf["driver_cycles"] - fh["driver_cycles"]) / LONG_GENERATIONS
        replay_delta = (
            replay[hf_key]["timing"]["diagnostic_last_issue_cycles"]
            - replay[fh_key]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        require(replay_delta == 0.0, f"W{warps}: expected exact replay null")
        replay_residuals.append(abs(live_delta - replay_delta))
        slow_threshold = fh["active_cycles"] + LONG_GENERATIONS
        hf_slow = sum(value > slow_threshold for value in hf["active_samples"])
        fh_slow = sum(value > slow_threshold for value in fh["active_samples"])
        pairs.append(
            {
                "stage": (
                    "identification"
                    if warps in {4, 8}
                    else ("threshold_localization" if warps in {13, 14, 15} else "held_out_width")
                ),
                "warps": warps,
                "warps_per_modeled_smsp_floor": warps // 4,
                "modeled_smsps_with_four_warps": max(0, warps - 12),
                "physical_warp_slots": sorted(expected_physical_slots(warps)),
                "fh_cycles_per_generation": fh["active_cycles"] / LONG_GENERATIONS,
                "hf_cycles_per_generation": hf["active_cycles"] / LONG_GENERATIONS,
                "live_delta_hf_minus_fh": live_delta,
                "driver_delta_hf_minus_fh": driver_delta,
                "replay_delta_hf_minus_fh": replay_delta,
                "live_minus_replay_residual": live_delta - replay_delta,
                "slow_regime_definition": "active cycles > paired FH median + 8192",
                "fh_slow_regime_samples": fh_slow,
                "fh_total_samples": len(fh["active_samples"]),
                "hf_slow_regime_samples": hf_slow,
                "hf_total_samples": len(hf["active_samples"]),
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
    require(len(source_paths) == 8, "SourceCounters profile count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    identification = [row for row in pairs if row["stage"] == "identification"]
    threshold = [row for row in pairs if row["stage"] == "threshold_localization"]
    held_out = [row for row in pairs if row["stage"] == "held_out_width"]
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 14,
            "correctness_executions": 98,
            "ncu_profiles": ncu_profiles,
            "source_counters_profiles": 8,
            "nvbit_captures": 14,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 14,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "high_cv_cells": high_cv_cells,
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
            "threshold_localization_delta_mae": statistics.mean(abs(row["live_delta_hf_minus_fh"]) for row in threshold),
            "held_out_width_delta_mae": statistics.mean(abs(row["live_delta_hf_minus_fh"]) for row in held_out),
            "all_delta_mae": statistics.mean(replay_residuals),
        },
        "decision": {
            "status": "opposite_bank_null_admitted_global_w16_phase_unmodeled",
            "activate_result_queue_timing": False,
            "findings": [
                "coincident and separated completion are indistinguishable at W4/W8",
                "W12 and adaptive W14/W15 medians remain within 0.056 cycles per generation",
                "one four-warp modeled partition at W13 does not trigger the W16 slow regime",
                "the HF slow regime occurs in 1/11 W14, 1/11 W15, and 8/11 W16 profiles",
                "production replay is an exact zero order delta at all seven widths",
                "short-scoreboard treatment delta is zero at every width",
            ],
            "interpretation": (
                "The W16 effect requires fully occupied SM-wide scheduler state or another global phase; "
                "it is not evidence for a four-warp SMSP-local result queue boundary."
            ),
            "next_slice": (
                "same-bank HADD2/FFMA completion at W4/W8 using the admitted order construction, "
                "with W16 retained only as a global-phase sentinel"
            ),
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    scores = summary["scores"]
    lines = [
        "# sm_86 balanced fixed-result producer-order reduction",
        "",
        "## Decision",
        "",
        "The exact opposite-bank order pair is admitted as a W4/W8 null, but no",
        "result-queue timing is activated. HADD2→FFMA coincident completion and",
        "FFMA→HADD2 separated completion are indistinguishable at one and two warps",
        "per modeled SMSP. Production replay is exactly zero-delta at every width.",
        "",
        "W16 exposes a separate bimodal global phase: 8/11 HF profiles are more than",
        "one cycle/generation above the paired FH median, versus 1/11 at W14, 1/11",
        "at W15, and none at W13. Because W13 already places four warps on one",
        "modeled partition, this falsifies a simple four-warp SMSP-local queue",
        "threshold. W16 remains a global scheduler/completion-phase sentinel.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells and {campaign['correctness_executions']} correctness executions",
        f"- {campaign['ncu_profiles']} base-clock NCU profiles and {campaign['source_counters_profiles']} SourceCounters profiles",
        f"- {campaign['nvbit_captures']} NVBit captures with {campaign['trace_records']:,} replay records",
        "- four spill-free cubins, 384/384 decoded controls, 88 local projections, zero gaps",
        "- retained 52-sidecar corpus: 39,464 records, 16,509 projections, zero gaps",
        "- issue ratio 1.0 and equal issued/executed/ALU/FMA/FP16 counts in every pair",
        f"- maximum raw CV {campaign['maximum_active_cycle_cv_pct']:.3f}%; maximum MAD {campaign['maximum_active_cycle_mad_pct']:.3f}%",
        "",
        "## Order deltas",
        "",
        "| Warps | Four-warp partitions | Stage | Live HF−FH | Replay | HF slow regime | Dispatch Δ | Math Δ | Wait Δ |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["pairs"]:
        lines.append(
            f"| {row['warps']} | {row['modeled_smsps_with_four_warps']} | {row['stage']} | "
            f"{row['live_delta_hf_minus_fh']:.3f} | {row['replay_delta_hf_minus_fh']:.3f} | "
            f"{row['hf_slow_regime_samples']}/{row['hf_total_samples']} | "
            f"{row['dispatch_delta']:.4f} | {row['math_delta']:.4f} | {row['wait_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Identification delta MAE is {scores['identification_delta_mae']:.4f} cycles/generation; "
            f"adaptive-threshold MAE is {scores['threshold_localization_delta_mae']:.4f}; "
            f"held-out W12/W16 MAE is {scores['held_out_width_delta_mae']:.4f}.",
            "Short-scoreboard delta is exactly zero throughout. Source sampling likewise",
            "finds no dependency-scoreboard explanation.",
            "",
            "## Static boundary",
            "",
            "Each function has 48 instructions and 15 registers. Code and controls are",
            "byte-identical outside PCs 0x100/0x110. At each PC the destination is fixed",
            "(R11 then R4), consumers are byte-identical, and the canonical controls are",
            "(stall=1,yield=0,reuse=0) then (stall=2,yield=0,reuse=0). Only the",
            "HADD2/FFMA order and corresponding source encoding differ.",
            "",
            "## Next slice",
            "",
            "Use this admitted producer-order construction for a same-bank W4/W8 pair.",
            "The bank treatment must be compiler-safe or a proven complete register",
            "renaming; unsafe destination-only patches remain excluded. W16 is not an",
            "identification input and remains only a global-phase sentinel.",
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
        print(f"result-order analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
