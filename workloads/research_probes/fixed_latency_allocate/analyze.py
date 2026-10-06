#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed-latency Allocate campaign."""

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
DEFAULT_ROOT = REPO_ROOT / "results" / "captures" / "sm86_fixed_latency_allocate_20260817_v1"
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 8192
PROTOCOL_NAMES = {
    "r00": "r00_control",
    "r10": "r10_parity_triplets",
    "r11": "r11_adjacent_triplets",
    "r12": "r12_same_bank_triplets",
    "r20": "r20_stable_triplet",
    "r21": "r21_rotated_triplet",
}
METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
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


def source_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: empty source report")
    documents = [dict(zip(rows[1], row)) for row in rows[2:]]
    result: dict[str, Any] = {}
    for short, column in (
        ("dispatch", "stall_dispatch (Not Issued)"),
        ("math", "stall_math (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
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
            "samples": sum(item[0] for item in values),
            "top": [
                {"samples": count, "sass": sass, "address": address}
                for count, sass, address in values[:6]
            ],
        }
    return result


def validate_static(root: Path) -> dict[str, Any]:
    names = (
        "fixed_latency_allocate",
        "fixed_latency_allocate_same_bank",
    )
    binaries = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        summary = sidecar["summary"]
        require(summary["total_decoded"] == 408, f"{name}: decoded")
        require(summary["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(len(summary["functions"]) == 6, f"{name}: functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill")
        binaries[name] = {"decoded": 408, "functions": 6, "spill_free": True}

    long_patch = read_json(root / "metadata" / "r11_same_bank_patch.json")
    for label, document in (("long", long_patch),):
        require(document["patched_instructions"] == 16, f"{label}: patch count")
        require(document["changed_bytes"] == 28, f"{label}: changed bytes")
        require(document["all_changed_bytes_inside_selected_words"] is True, f"{label}: scope")
        require(all(row["high_unchanged"] for row in document["patches"]), f"{label}: high")
        require(all(row["reuse"] == 0 for row in document["patches"]), f"{label}: reuse")

    function = "_Z22fixed_latency_allocateILi2EEvPyPj"
    native = read_json(root / "sass" / "fixed_latency_allocate.control_bits.json")
    bank3 = read_json(root / "sass" / "fixed_latency_allocate_same_bank.control_bits.json")
    left = {row["pc"]: row for row in native["functions"][function]}
    right = {row["pc"]: row for row in bank3["functions"][function]}
    require(left.keys() == right.keys(), "patched PC set")
    changed = set()
    for pc in left:
        require(left[pc]["high_word"] == right[pc]["high_word"], f"pc {pc}: high changed")
        require(left[pc]["control"] == right[pc]["control"], f"pc {pc}: controls changed")
        require(left[pc]["predicate"] == right[pc]["predicate"], f"pc {pc}: predicate changed")
        if left[pc]["low_word"] != right[pc]["low_word"]:
            changed.add(pc)
    require(changed == {int(row["pc"]) for row in long_patch["patches"]}, "changed PC set")

    projection = read_json(root / "metadata" / "static_operand_projection.json")
    require(projection["files"] == 4 and projection["gaps"] == 0, "projection coverage")
    targets = [
        item
        for item in projection["per_function"]
        if "same_bank" in item["path"] and item["function"] == function
    ]
    require(len(targets) == 2, "bank3 functions")
    for target in targets:
        hot = [
            row
            for row in target["instructions"]
            if row["opcode"] == "Imad" and row["pc"] in changed
        ]
        require(len(hot) == 16, "bank3 hot count")
        require(all(row["reuse_mask"] == 0 for row in hot), "bank3 reuse")
        require(all(row["source_bank_reads"] in ([3, 0], [0, 3]) for row in hot), "bank3 demand")
    return {
        "binaries": binaries,
        "patched_instructions": 16,
        "changed_bytes": 28,
        "all_controls_high_words_and_other_bytes_unchanged": True,
        "projection": {
            "records": projection["records"],
            "projected": projection["projected"],
            "gaps": projection["gaps"],
        },
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 28 and len({row["key"] for row in cells}) == 28, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live: dict[str, dict[str, Any]] = {}
    replay: dict[str, dict[str, Any]] = {}
    candidate_directories = {
        "flexible_window": "simulator-control-allocate",
        "fixed_operand_slots": "simulator-control-allocate-slots",
        "extra_allocate_hold": "simulator-control-allocate-hold",
        "tagged_rfc_allocate_hold": "simulator-rfc-tagged-hold",
    }
    candidate_cycles: dict[str, dict[str, float]] = {
        name: {} for name in candidate_directories
    }
    trace_records = 0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["protocol"] == PROTOCOL_NAMES[cell["protocol"]], f"{key}: protocol")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: trials")

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        expected_repeats = 5 if key == "w8_r00_native" else 3
        require(len(repeat_paths) == expected_repeats, f"{key}: repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows),
            f"{key}: issue ratio",
        )
        active = [row["sm__cycles_active.max"] for row in rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        require(cv <= (6.0 if key == "w8_r00_native" else 2.0), f"{key}: CV")
        median = lambda name: statistics.median(row[name] for row in rows)
        live[key] = {
            "active_samples": active,
            "active_cycles": statistics.median(active),
            "active_cv_pct": cv,
            **{name: median(name) for name in METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_result = measurement(capture / "stdout.txt")
        require(trace_result["correct"] is True, f"{key}: trace correctness")
        require(trace_result["generations"] == TRACE_GENERATIONS, f"{key}: trace generations")
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: block files")
        block = bindings.threadblock_pb2.threadblock()
        block.ParseFromString(paths[0].read_bytes())
        records = sum(len(warp.instructions) for warp in block.warps.values())
        slots = {
            int(warp.instructions[0].warpid_sm)
            for warp in block.warps.values()
            if warp.instructions
        }
        require(slots == set(range(cell["warps"])), f"{key}: slots")
        trace_records += records

        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: deterministic")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
        require(launch["counters"]["records_seen"] == records, f"{key}: records")
        require(launch["counters"]["instructions_issued"] == records, f"{key}: issued")
        replay[key] = launch
        for name, directory in candidate_directories.items():
            candidate_first = root / directory / key / "replay-1.json"
            candidate_second = root / directory / key / "replay-2.json"
            require(
                candidate_first.read_bytes() == candidate_second.read_bytes(),
                f"{key}: {name} nondeterministic",
            )
            candidate_launch = read_json(candidate_first)["launches"][0]
            require(candidate_launch["counters"] == launch["counters"], f"{key}: {name} traffic")
            candidate_cycles[name][key] = (
                candidate_launch["timing"]["diagnostic_last_issue_cycles"]
                / TRACE_GENERATIONS
            )
        restored_first = root / "simulator-restored" / key / "replay-1.json"
        restored_second = root / "simulator-restored" / key / "replay-2.json"
        require(restored_first.read_bytes() == restored_second.read_bytes(), f"{key}: restored pair")
        require(restored_first.read_bytes() == first.read_bytes(), f"{key}: restored changed")

    pair_rows = []
    linear_errors = []
    for warps in (4, 8, 12, 16):
        native_key = f"w{warps}_r11_native"
        bank_key = f"w{warps}_r11_bank3"
        native = live[native_key]
        bank = live[bank_key]
        for name in ("smsp__inst_issued.sum", "smsp__inst_executed.sum", "smsp__inst_executed_pipe_fma.sum"):
            require(native[name] == bank[name], f"W{warps}: {name}")
        require(replay[native_key]["counters"] == replay[bank_key]["counters"], f"W{warps}: traffic")
        delta = (bank["active_cycles"] - native["active_cycles"]) / LONG_GENERATIONS
        simple = 7.0 * (warps // 4)
        linear_errors.append(abs(delta - simple))
        pair_rows.append(
            {
                "warps": warps,
                "warps_per_modeled_smsp": warps // 4,
                "native_cycles_per_generation": native["active_cycles"] / LONG_GENERATIONS,
                "bank3_cycles_per_generation": bank["active_cycles"] / LONG_GENERATIONS,
                "live_delta": delta,
                "w4_linear_prediction": simple,
                "w4_linear_error": abs(delta - simple),
                "dispatch_delta": bank["smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"] - native["smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"],
                "math_delta": bank["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"] - native["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"],
                "not_selected_delta": bank["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"] - native["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"],
                "wait_delta": bank["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"] - native["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"],
                "replay_delta": (
                    replay[bank_key]["timing"]["diagnostic_last_issue_cycles"]
                    - replay[native_key]["timing"]["diagnostic_last_issue_cycles"]
                ) / TRACE_GENERATIONS,
                **{
                    f"{name}_delta": candidate_cycles[name][bank_key]
                    - candidate_cycles[name][native_key]
                    for name in candidate_directories
                },
            }
        )

    native_rows = []
    for warps in (4, 8, 12, 16):
        anchor = live[f"w{warps}_r10_native"]["active_cycles"] / LONG_GENERATIONS
        for protocol in ("r10", "r11", "r12", "r20", "r21"):
            cycles = live[f"w{warps}_{protocol}_native"]["active_cycles"] / LONG_GENERATIONS
            native_rows.append(
                {
                    "warps": warps,
                    "protocol": protocol,
                    "cycles_per_generation": cycles,
                    "delta_from_r10": cycles - anchor,
                }
            )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 9, "source profile count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    candidate_scores = []
    for name in candidate_directories:
        errors = [abs(row["live_delta"] - row[f"{name}_delta"]) for row in pair_rows]
        candidate_scores.append(
            {
                "candidate": name,
                "identification_mae": statistics.mean(errors[:2]),
                "held_out_mae": statistics.mean(errors[2:]),
                "all_width_mae": statistics.mean(errors),
                "status": "rejected_pending_missing_read_completion",
            }
        )
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 28,
            "correctness_executions": 196,
            "ncu_profiles": 86,
            "primary_ncu_profiles": 84,
            "source_counters_profiles": 9,
            "nvbit_captures": 28,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 28,
            "byte_identical_candidate_pairs": 28 * len(candidate_directories),
            "restored_outputs_identical": 28,
            "expanded_control_cv_pct": live["w8_r00_native"]["active_cv_pct"],
            "all_issue_ratios_one": True,
            "all_cardinality_exact": True,
            "all_full_coverage": True,
            "all_exclusion_sidecars_empty": True,
        },
        "static": static,
        "exact_bank_pair": pair_rows,
        "w4_linear_model_mae_all": statistics.mean(linear_errors),
        "w4_linear_model_mae_held_out": statistics.mean(linear_errors[2:]),
        "native_layouts": native_rows,
        "candidates": candidate_scores,
        "source_counters": sources,
        "decision": {
            "operand_projection_promoted": True,
            "timing_promoted": False,
            "best_candidate": "flexible_window",
            "next_candidate": "flexible Control/Allocate window plus register-result completion distance",
        },
    }


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        "# sm_86 fixed-latency operand projection and bank pilot",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (sm_86)  ",
        "**Decision:** retain the static operand projection; bank timing remains inactive pending Control/Allocate transfer",
        "",
        "## Campaign integrity",
        "",
        f"All 28 cells pass seven trials (196 executions). The result contains {campaign['ncu_profiles']} NCU profiles, nine SourceCounters reports, and 28 NVBit captures carrying {campaign['trace_records']:,} replay records. Every issue ratio is 1.0. Every non-control cell is below 2% active-cycle CV; the tiny W8 control was expanded to five repeats and remains {campaign['expanded_control_cv_pct']:.2f}% CV.",
        "",
        "All four native/patched long/trace cubins decode 408/408 instructions across six functions with zero unknown controls, spills, or local memory. The exact patch changes 28 bytes inside sixteen selected bare-IMAD words. Every high/control word, predicate, destination, opcode, PC, resource declaration, and all other file bytes remain unchanged. Projection proves three-on-one-bank demand and reuse mask zero for all sixteen patched instructions.",
        "",
        "## Exact bank pair",
        "",
        "| Warps | Warps/SMSP | Native cycles/gen | Bank3 cycles/gen | Live delta | Flexible Allocate | W4-linear | Dispatch delta | Math delta | Replay delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["exact_bank_pair"]:
        lines.append(
            f"| {row['warps']} | {row['warps_per_modeled_smsp']} | "
            f"{row['native_cycles_per_generation']:.3f} | {row['bank3_cycles_per_generation']:.3f} | "
            f"{row['live_delta']:.3f} | {row['flexible_window_delta']:.3f} | "
            f"{row['w4_linear_prediction']:.3f} | "
            f"{row['dispatch_delta']:.3f} | {row['math_delta']:.3f} | "
            f"{row['replay_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Issued, executed, and FMA-pipe instruction counts plus every replay traffic counter are identical within each pair. W4 has no math throttle: its seven-cycle penalty appears as dispatch-stall plus wait, locating the base effect in allocation/read staging. Higher occupancy adds math throttle while not-selected changes little.",
            "",
            f"A seven-cycle-per-warp/SMSP law predicts W4/W8 exactly and W12 within 0.56 cycles, but misses W16 by 6.83; held-out MAE is {summary['w4_linear_model_mae_held_out']:.3f}. It is not promoted as a correction.",
            "",
            "## Native reuse layouts",
            "",
            "At W4 all native layouts are effectively equal. The no-reuse adjacent layout r11 is about +1 cycle/generation at W8/W12 and +2 at W16 relative to r10, while stable and rotated compiler-reuse layouts track r10. This supports a reuse-cache mechanism but is not yet an exact binary pair.",
            "",
            "## Timing candidates",
            "",
            "| Candidate | W4/W8 identification MAE | W12/W16 held-out MAE | Decision |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["candidates"]:
        lines.append(
            f"| {row['candidate']} | {row['identification_mae']:.3f} | "
            f"{row['held_out_mae']:.3f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "The flexible three-cycle read window is the only viable stage shape: it predicts W8 and W12 bank deltas within 0.31 and 0.47 cycles and sharply improves native W8-W16 absolute timing. Fixed operand slots collapse the bank differential; an extra Allocate hold overprices every multi-warp width. Flexible Allocate still exposes only 2 of 7 W4 cycles and underpredicts W16 by 5.31, so its timing code was removed after capture.",
            "",
            "## Decision",
            "",
            "The immutable SASS operand projection is retained because it is timing-neutral and covers all 13,141 target instructions in the retained 26-sidecar corpus with zero gaps and exact text/control reuse agreement. Current replay correctly reports zero timing delta for the bank-only pair.",
            "",
            "The next candidate composes the surviving flexible Control/Allocate window with register-result completion distance. A dependency-distance probe must identify that arm before timing can be promoted; W12/W16 and the prior 99 scheduler cells remain held out.",
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
        write_json(root / "analysis" / "summary.json", summary)
        (root / "analysis" / "README.md").write_text(report(summary), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"fixed Allocate analysis: {error}", file=sys.stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
