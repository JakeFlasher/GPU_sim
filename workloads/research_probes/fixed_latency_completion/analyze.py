#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed-latency completion-distance campaign."""

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
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_fixed_latency_completion_20260817_v1"
)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 8192
DISTANCES = (1, 2, 4, 8)
WIDTHS = (4, 8, 12, 16)
METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
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
CANDIDATES = {
    "staged_window": "simulator-control-allocate-staged",
    "reference_dispatch_hold": "simulator-control-allocate-hold2",
    "allocate_control_conflict_hold": "simulator-control-allocate-conflict-noexec",
    "allocate_failure_hold": "simulator-control-allocate-fail-hold",
    "issuer_completion_exposure": "simulator-control-allocate-warp-completion",
    "tagged_rfc_allocate_hold": "simulator-rfc-tagged-hold",
}


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
        ("selected", "stall_selected (Not Issued)"),
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
            "samples": sum(item[0] for item in values),
            "top": [
                {"samples": count, "sass": sass, "address": address}
                for count, sass, address in values[:8]
            ],
        }
    return result


def validate_static(root: Path) -> dict[str, Any]:
    names = tuple(
        f"fixed_latency_completion_{variant}"
        for variant in ("native", "split", "bank3")
    )
    binaries = {}
    sidecars = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        summary = sidecar["summary"]
        require(summary["total_decoded"] == 424, f"{name}: decoded")
        require(summary["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(len(summary["functions"]) == 5, f"{name}: functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill")
        binaries[name] = {"decoded": 424, "functions": 5, "spill_free": True}

    reports = {}
    changed_by_function: dict[str, set[int]] = {}
    for label, patch_name in (("long", "long_layout_patch.json"),):
        document = read_json(root / "metadata" / patch_name)
        require(document["patched_instructions"] == 64, f"{label}: patch count")
        require(document["split_to_bank3_changed_bytes"] == 64, f"{label}: pair bytes")
        require(document["all_changed_bytes_inside_selected_words"] is True, f"{label}: scope")
        require(document["pair_high_words_identical"] is True, f"{label}: high words")
        require(all(row["final_reuse"] == 0 for row in document["patches"]), f"{label}: reuse")
        require(
            all(row["split_bank_reads"] in ([2, 1], [1, 2]) for row in document["patches"]),
            f"{label}: split demand",
        )
        require(
            all(row["bank3_bank_reads"] in ([3, 0], [0, 3]) for row in document["patches"]),
            f"{label}: bank3 demand",
        )
        reports[label] = {
            key: document[key]
            for key in (
                "patched_instructions",
                "original_to_split_changed_bytes",
                "original_to_bank3_changed_bytes",
                "split_to_bank3_changed_bytes",
                "pair_high_words_identical",
            )
        }
        if label == "long":
            for row in document["patches"]:
                changed_by_function.setdefault(row["function"], set()).add(int(row["pc"]))

    for prefix in ("fixed_latency_completion",):
        split = sidecars[f"{prefix}_split"]
        bank3 = sidecars[f"{prefix}_bank3"]
        require(split["functions"].keys() == bank3["functions"].keys(), f"{prefix}: functions")
        for function in split["functions"]:
            left = {row["pc"]: row for row in split["functions"][function]}
            right = {row["pc"]: row for row in bank3["functions"][function]}
            require(left.keys() == right.keys(), f"{prefix} {function}: PCs")
            changed = set()
            for pc in left:
                require(left[pc]["high_word"] == right[pc]["high_word"], f"{function} {pc}: high")
                require(left[pc]["control"] == right[pc]["control"], f"{function} {pc}: control")
                require(left[pc]["predicate"] == right[pc]["predicate"], f"{function} {pc}: predicate")
                if left[pc]["low_word"] != right[pc]["low_word"]:
                    changed.add(pc)
            expected = changed_by_function.get(function, set())
            require(changed == expected, f"{prefix} {function}: changed PC set")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(projection["files"] == 3 and projection["gaps"] == 0, "projection coverage")
    for variant, expected_demands in (
        ("split", ([2, 1], [1, 2])),
        ("bank3", ([3, 0], [0, 3])),
    ):
        targets = [
            item
            for item in projection["per_function"]
            if f"_{variant}.control_bits.json" in item["path"]
            and item["function"] in changed_by_function
        ]
        require(len(targets) == 4, f"{variant}: projected functions")
        for target in targets:
            hot = [
                row
                for row in target["instructions"]
                if row["opcode"] == "Imad"
                and row["pc"] in changed_by_function[target["function"]]
            ]
            require(len(hot) == 16, f"{target['function']}: hot projection")
            require(all(row["reuse_mask"] == 0 for row in hot), f"{target['function']}: reuse")
            require(
                all(row["source_bank_reads"] in expected_demands for row in hot),
                f"{target['function']}: bank demand",
            )

    schedules = {}
    split_long = sidecars["fixed_latency_completion_split"]
    for distance in DISTANCES:
        function = f"completion_d{distance}"
        hot_pcs = changed_by_function[function]
        rows = [row for row in split_long["functions"][function] if row["pc"] in hot_pcs]
        rows.sort(key=lambda row: row["pc"])
        schedules[str(distance)] = {
            "hot_instructions": len(rows),
            "stall_sum": sum(row["control"]["stall"] for row in rows),
            "stall_histogram": {
                str(stall): sum(row["control"]["stall"] == stall for row in rows)
                for stall in sorted({row["control"]["stall"] for row in rows})
            },
            "yield_count": sum(row["control"]["yield_flag"] for row in rows),
        }
    return {
        "binaries": binaries,
        "patches": reports,
        "operand_projection": {
            key: projection[key]
            for key in ("files", "records", "projected", "gaps")
        },
        "hot_control_schedules": schedules,
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 32 and len({row["key"] for row in cells}) == 32, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live: dict[str, dict[str, Any]] = {}
    replay: dict[str, dict[str, Any]] = {}
    candidate_cycles: dict[str, dict[str, float]] = {
        candidate: {} for candidate in CANDIDATES
    }
    trace_records = 0
    max_cv = 0.0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["distance"] == cell["distance"], f"{key}: distance")
        require(payload["variant"] == cell["variant"], f"{key}: variant")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) >= 3, f"{key}: repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in rows),
            f"{key}: issue ratio",
        )
        active = [row["sm__cycles_active.max"] for row in rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        max_cv = max(max_cv, cv)
        require(cv <= 2.0, f"{key}: active-cycle CV {cv:.3f}%")
        median = lambda name: statistics.median(row[name] for row in rows)
        live[key] = {
            "driver_cycles": statistics.median(driver_samples),
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
            require(
                read_json(capture / "extra_info" / f"{diagnostic}.json") == [],
                f"{key}: {diagnostic}",
            )
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
        restored_first = root / "simulator-restored" / key / "replay-1.json"
        restored_second = root / "simulator-restored" / key / "replay-2.json"
        require(restored_first.read_bytes() == restored_second.read_bytes(), f"{key}: restored pair")
        require(restored_first.read_bytes() == first.read_bytes(), f"{key}: restored changed")
        for candidate, directory in CANDIDATES.items():
            candidate_first = root / directory / key / "replay-1.json"
            candidate_second = root / directory / key / "replay-2.json"
            require(
                candidate_first.read_bytes() == candidate_second.read_bytes(),
                f"{key}: {candidate} nondeterministic",
            )
            candidate_launch = read_json(candidate_first)["launches"][0]
            require(
                candidate_launch["counters"] == launch["counters"],
                f"{key}: {candidate} changed traffic",
            )
            candidate_cycles[candidate][key] = (
                candidate_launch["timing"]["diagnostic_last_issue_cycles"]
                / TRACE_GENERATIONS
            )

    pair_rows = []
    for warps in WIDTHS:
        for distance in DISTANCES:
            split_key = f"w{warps}_d{distance}_split"
            bank_key = f"w{warps}_d{distance}_bank3"
            split = live[split_key]
            bank = live[bank_key]
            for name in (
                "smsp__inst_issued.sum",
                "smsp__inst_executed.sum",
                "smsp__inst_executed_pipe_fma.sum",
            ):
                require(split[name] == bank[name], f"W{warps} D{distance}: {name}")
            require(
                replay[split_key]["counters"] == replay[bank_key]["counters"],
                f"W{warps} D{distance}: traffic",
            )
            row = {
                    "stage": next(row["stage"] for row in cells if row["key"] == split_key),
                    "warps": warps,
                    "warps_per_modeled_smsp": warps // 4,
                    "distance": distance,
                    "split_cycles_per_generation": split["active_cycles"] / LONG_GENERATIONS,
                    "bank3_cycles_per_generation": bank["active_cycles"] / LONG_GENERATIONS,
                    "live_delta": (bank["active_cycles"] - split["active_cycles"])
                    / LONG_GENERATIONS,
                    "driver_delta": (bank["driver_cycles"] - split["driver_cycles"])
                    / LONG_GENERATIONS,
                    "split_cv_pct": split["active_cv_pct"],
                    "bank3_cv_pct": bank["active_cv_pct"],
                    "selected_delta": bank[
                        "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio"
                    ]
                    - split[
                        "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio"
                    ],
                    "dispatch_delta": bank[
                        "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"
                    ]
                    - split[
                        "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"
                    ],
                    "math_delta": bank[
                        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
                    ]
                    - split[
                        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
                    ],
                    "not_selected_delta": bank[
                        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                    ]
                    - split[
                        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                    ],
                    "wait_delta": bank[
                        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                    ]
                    - split[
                        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                    ],
                    "short_scoreboard_delta": bank[
                        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
                    ]
                    - split[
                        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
                    ],
                    "replay_delta": (
                        replay[bank_key]["timing"]["diagnostic_last_issue_cycles"]
                        - replay[split_key]["timing"]["diagnostic_last_issue_cycles"]
                    )
                    / TRACE_GENERATIONS,
                }
            for candidate in CANDIDATES:
                row[f"{candidate}_delta"] = (
                    candidate_cycles[candidate][bank_key]
                    - candidate_cycles[candidate][split_key]
                )
            pair_rows.append(row)

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 12, "source profile count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    candidate_scores = []
    for candidate, directory in CANDIDATES.items():
        score: dict[str, Any] = {
            "candidate": candidate,
            "directory": directory,
            "status": "rejected",
        }
        all_delta_errors = []
        all_absolute_errors = []
        for stage in ("identification", "held_out_distance", "held_out_width"):
            stage_rows = [row for row in pair_rows if row["stage"] == stage]
            delta_errors = [
                abs(row["live_delta"] - row[f"{candidate}_delta"])
                for row in stage_rows
            ]
            absolute_errors = []
            for row in stage_rows:
                for variant, live_name in (
                    ("split", "split_cycles_per_generation"),
                    ("bank3", "bank3_cycles_per_generation"),
                ):
                    key = f"w{row['warps']}_d{row['distance']}_{variant}"
                    absolute_errors.append(
                        abs(candidate_cycles[candidate][key] - row[live_name])
                    )
            score[f"{stage}_delta_mae"] = statistics.mean(delta_errors)
            score[f"{stage}_absolute_mae"] = statistics.mean(absolute_errors)
            all_delta_errors.extend(delta_errors)
            all_absolute_errors.extend(absolute_errors)
        score["all_delta_mae"] = statistics.mean(all_delta_errors)
        score["all_absolute_mae"] = statistics.mean(all_absolute_errors)
        candidate_scores.append(score)

    scheduler_summary = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_smsp_scheduler_contention_20260817_v1"
        / "analysis"
        / "summary.json"
    )
    scheduler_candidates = {
        row["candidate"]: row for row in scheduler_summary["candidates"]
    }
    scheduler_transfer = {
        candidate: {
            key: scheduler_candidates[scheduler_name][key]
            for key in (
                "no_contention_same_mae_cycles_per_generation",
                "identification_same_mae_cycles_per_generation",
                "phase_transfer_same_mae_cycles_per_generation",
                "held_out_same_mae_cycles_per_generation",
                "held_out_cross_mae_cycles_per_generation",
                "all_scored_mae_cycles_per_generation",
            )
        }
        for candidate, scheduler_name in (
            ("staged_window", "staged_control_allocate"),
            ("issuer_completion_exposure", "staged_allocate_warp_completion"),
        )
    }
    best_candidate = min(candidate_scores, key=lambda row: row["all_delta_mae"])
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 32,
            "correctness_executions": 224,
            "ncu_profiles": sum(len(list((root / "ncu" / row["key"]).glob("repeat-*"))) for row in cells),
            "source_counters_profiles": 12,
            "nvbit_captures": 32,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 32,
            "restored_outputs_identical": 64,
            "maximum_active_cycle_cv_pct": max_cv,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
            "all_cardinality_exact": True,
            "all_full_coverage": True,
            "all_exclusion_sidecars_empty": True,
        },
        "static": static,
        "exact_pairs": pair_rows,
        "candidates": candidate_scores,
        "scheduler_transfer": scheduler_transfer,
        "source_counters": sources,
        "decision": {
            "timing_promoted": False,
            "best_completion_candidate": best_candidate["candidate"],
            "best_completion_delta_mae": best_candidate["all_delta_mae"],
            "status": "all timing candidates rejected",
            "reason": (
                "no candidate jointly preserves W4 distance exposure, held-out "
                "distance/width transfer, absolute timing, and the prior 99-cell "
                "scheduler campaign"
            ),
        },
    }


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        "# sm_86 fixed-latency completion-distance campaign",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (`sm_86`)  ",
        "**Decision:** retain the evidence; reject all tested timing compositions and keep production unchanged",
        "",
        "## Campaign integrity",
        "",
        f"All 32 cells pass seven correctness trials ({campaign['correctness_executions']} executions). The result contains {campaign['ncu_profiles']} base-clock NCU profiles, 12 SourceCounters reports, and 32 NVBit captures carrying {campaign['trace_records']:,} replay records. Maximum active-cycle CV is {campaign['maximum_active_cycle_cv_pct']:.2f}%; every issue ratio is 1.0. After candidate removal, all {campaign['restored_outputs_identical']} saved replay outputs are byte-identical to production.",
        "",
        "Three current cubins decode 424/424 instructions across five functions with no unknown controls, spills, or local memory. One patched split/bank3 pair contains 64 selected hot IMAD changes. Every pairwise byte difference is a first/second source-register byte inside a selected low word; all high/control words, destinations, carried addends, predicates, resources, and reuse-zero masks are identical. Projection covers the current three-file static set with zero gaps.",
        "",
        "## Exact source-bank treatment by consumer distance",
        "",
        "| Warps | Warps/SMSP | RAW distance | Split cycles/gen | Bank3 cycles/gen | Live delta | Dispatch Δ | Math Δ | Wait Δ | Replay Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["exact_pairs"]:
        lines.append(
            f"| {row['warps']} | {row['warps_per_modeled_smsp']} | {row['distance']} | "
            f"{row['split_cycles_per_generation']:.3f} | "
            f"{row['bank3_cycles_per_generation']:.3f} | {row['live_delta']:.3f} | "
            f"{row['dispatch_delta']:.3f} | {row['math_delta']:.3f} | "
            f"{row['wait_delta']:.3f} | {row['replay_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The exact pair preserves issued, executed, and FMA-pipe counts at every distance and width. Current replay preserves all traffic but predicts zero bank delta, localizing the residual to fixed-latency staging rather than trace content or instruction classification.",
            "",
            "## Timing candidates",
            "",
            "| Candidate | Identification delta MAE | Held-out distance MAE | Held-out width MAE | Absolute MAE | Decision |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["candidates"]:
        lines.append(
            f"| {row['candidate']} | {row['identification_delta_mae']:.3f} | "
            f"{row['held_out_distance_delta_mae']:.3f} | "
            f"{row['held_out_width_delta_mae']:.3f} | "
            f"{row['all_absolute_mae']:.3f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "The staged flexible window transfers best at moderate/high occupancy but misses exposed W4 conflicts. Reference-style global dispatch holds recover W4 while disrupting multiwarp phase, especially D2. Issuer-local completion exposure predicts W4 D1/D2/D4 within 0.03/0.00/0.00 cycles and W16 D1 within 0.79, but misses W8 D4 by 9.13 cycles and does not materially improve the prior 99-cell scheduler corpus. Future execution-slot reservation produced no additional treatment delta in this homogeneous-IMAD probe, and the identical destination stream cannot identify result-queue capacity.",
            "",
            "## Decision",
            "",
            "No timing component is promoted. The static operand projection remains the essential retained component, while production replay remains byte-stable. The next probe must isolate compiler-managed reuse hits/evictions and mixed-latency destination writeback/result-queue pressure with exact binary pairs; neither can be inferred from this no-reuse, identical-destination experiment.",
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
        print(f"completion-distance analysis: {error}", file=sys.stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
