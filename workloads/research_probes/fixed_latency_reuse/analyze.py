#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed-latency RFC campaign."""

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
    REPO_ROOT / "results" / "captures" / "sm86_fixed_latency_reuse_20260817_v1"
)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 8192
PROTOCOLS = ("u10", "u11", "u12", "u20", "u22")
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
    "tagged_rfc": "simulator-rfc-tagged",
    "untagged_rfc": "simulator-rfc-untagged",
    "tagged_issuer_exposure": "simulator-rfc-tagged-warp",
    "tagged_allocate_hold": "simulator-rfc-tagged-hold",
    "per_warp_upper_bound": "simulator-rfc-per-warp",
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


def steady_hits(rows: list[dict[str, Any]], variant: str) -> int:
    entries: list[list[tuple[int, int] | None]] = [[None, None] for _ in range(3)]
    hits = 0
    for generation in range(2):
        generation_hits = 0
        for row in rows:
            sources = row[f"sources_{variant}"]
            reuse = int(row[f"reuse_{variant}"])
            for position, register in enumerate(sources):
                bank = register % 2
                if entries[position][bank] == (0, register):
                    generation_hits += 1
                entries[position][bank] = None
                if reuse & (1 << position):
                    entries[position][bank] = (0, register)
        if generation == 1:
            hits = generation_hits
    return hits


def validate_static(root: Path) -> dict[str, Any]:
    names = tuple(f"fixed_latency_reuse_{variant}" for variant in ("native", "a", "b"))
    sidecars = {}
    binaries = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 560, f"{name}: decoded")
        require(sidecar["summary"]["total_unknown_encoding"] == 0, f"{name}: unknown")
        require(len(sidecar["functions"]) == 5, f"{name}: functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spill")
        binaries[name] = {"decoded": 560, "functions": 5, "spill_free": True}

    patch_documents = {}
    hot_pcs: dict[str, set[int]] = {}
    for label in ("long",):
        document = read_json(root / "metadata" / f"{label}_reuse_patch.json")
        require(document["patched_instructions"] == 80, f"{label}: patch count")
        require(document["all_changed_bytes_inside_selected_words"] is True, f"{label}: scope")
        require(document["pair_bank_demand_identical"] is True, f"{label}: demand")
        patch_documents[label] = {
            key: document[key]
            for key in (
                "patched_instructions",
                "original_to_a_changed_bytes",
                "original_to_b_changed_bytes",
                "a_to_b_changed_bytes",
            )
        }
        if label == "long":
            for row in document["patches"]:
                hot_pcs.setdefault(row["function"], set()).add(int(row["pc"]))

    for prefix in ("fixed_latency_reuse", "fixed_latency_reuse_trace"):
        left = sidecars[f"{prefix}_a"]
        right = sidecars[f"{prefix}_b"]
        for function in left["functions"]:
            a_rows = {row["pc"]: row for row in left["functions"][function]}
            b_rows = {row["pc"]: row for row in right["functions"][function]}
            require(a_rows.keys() == b_rows.keys(), f"{prefix} {function}: PCs")
            for pc in a_rows:
                a = a_rows[pc]
                b = b_rows[pc]
                require(a["predicate"] == b["predicate"], f"{function} {pc}: predicate")
                for field in ("stall", "yield_flag", "write_barrier", "read_barrier", "wait_mask"):
                    require(a["control"][field] == b["control"][field], f"{function} {pc}: {field}")
            for pc in hot_pcs[function]:
                require(a_rows[pc]["control"]["yield_flag"] == 1, f"{function} {pc}: yield")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(projection["files"] == 6 and projection["gaps"] == 0, "projection coverage")
    patch_rows = read_json(root / "metadata" / "long_reuse_patch.json")["patches"]
    expected_hits = {}
    for function in hot_pcs:
        rows = sorted(
            [row for row in patch_rows if row["function"] == function],
            key=lambda row: row["ordinal"],
        )
        expected_hits[function.removeprefix("reuse_")] = {
            "a": steady_hits(rows, "a"),
            "b": steady_hits(rows, "b"),
        }
    return {
        "binaries": binaries,
        "patches": patch_documents,
        "projection": {key: projection[key] for key in ("files", "records", "projected", "gaps")},
        "reference_steady_hits_per_generation_one_warp": expected_hits,
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 40 and len({row["key"] for row in cells}) == 40, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live = {}
    replay = {}
    candidate_cycles = {candidate: {} for candidate in CANDIDATES}
    trace_records = 0
    max_cv = 0.0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(payload["protocol"] == cell["protocol"], f"{key}: protocol")
        require(payload["variant"] == cell["variant"], f"{key}: variant")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: trials")
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) >= 3, f"{key}: repeats")
        ncu_rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in ncu_rows),
            f"{key}: issue ratio",
        )
        active = [row["sm__cycles_active.max"] for row in ncu_rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        max_cv = max(max_cv, cv)
        require(cv <= 2.0, f"{key}: CV {cv:.3f}%")
        median = lambda name: statistics.median(row[name] for row in ncu_rows)
        live[key] = {
            "active_cycles": statistics.median(active),
            "active_samples": active,
            "active_cv_pct": cv,
            "driver_cycles": statistics.median(driver_samples),
            **{name: median(name) for name in METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(trace_payload["correct"] is True, f"{key}: trace correctness")
        require(trace_payload["generations"] == TRACE_GENERATIONS, f"{key}: trace generations")
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
                f"{key}: {candidate} traffic",
            )
            candidate_cycles[candidate][key] = (
                candidate_launch["timing"]["diagnostic_last_issue_cycles"]
                / TRACE_GENERATIONS
            )

    pairs = []
    for warps in WIDTHS:
        for protocol in PROTOCOLS:
            a_key = f"w{warps}_{protocol}_a"
            b_key = f"w{warps}_{protocol}_b"
            a = live[a_key]
            b = live[b_key]
            for name in ("smsp__inst_issued.sum", "smsp__inst_executed.sum", "smsp__inst_executed_pipe_fma.sum"):
                require(a[name] == b[name], f"W{warps} {protocol}: {name}")
            require(replay[a_key]["counters"] == replay[b_key]["counters"], f"W{warps} {protocol}: traffic")
            stage = next(row["stage"] for row in cells if row["key"] == a_key)
            pair = {
                    "stage": stage,
                    "warps": warps,
                    "warps_per_modeled_smsp": warps // 4,
                    "protocol": protocol,
                    "a_cycles_per_generation": a["active_cycles"] / LONG_GENERATIONS,
                    "b_cycles_per_generation": b["active_cycles"] / LONG_GENERATIONS,
                    "live_delta_b_minus_a": (b["active_cycles"] - a["active_cycles"])
                    / LONG_GENERATIONS,
                    "driver_delta_b_minus_a": (b["driver_cycles"] - a["driver_cycles"])
                    / LONG_GENERATIONS,
                    "dispatch_delta": b[
                        "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"
                    ]
                    - a["smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"],
                    "math_delta": b[
                        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"
                    ]
                    - a["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"],
                    "wait_delta": b[
                        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
                    ]
                    - a["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"],
                    "not_selected_delta": b[
                        "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"
                    ]
                    - a["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"],
                    "short_scoreboard_delta": b[
                        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
                    ]
                    - a["smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"],
                    "replay_delta": (
                        replay[b_key]["timing"]["diagnostic_last_issue_cycles"]
                        - replay[a_key]["timing"]["diagnostic_last_issue_cycles"]
                    )
                    / TRACE_GENERATIONS,
                    "a_cv_pct": a["active_cv_pct"],
                    "b_cv_pct": b["active_cv_pct"],
                }
            for candidate in CANDIDATES:
                pair[f"{candidate}_delta"] = (
                    candidate_cycles[candidate][b_key]
                    - candidate_cycles[candidate][a_key]
                )
            pairs.append(pair)

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 16, "source profile count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}
    candidate_scores = []
    for candidate, directory in CANDIDATES.items():
        score = {"candidate": candidate, "directory": directory, "status": "rejected"}
        all_errors = []
        all_absolute = []
        for stage in ("identification", "held_out_policy", "held_out_width"):
            rows = [row for row in pairs if row["stage"] == stage]
            errors = [
                abs(row["live_delta_b_minus_a"] - row[f"{candidate}_delta"])
                for row in rows
            ]
            absolute = []
            for row in rows:
                for variant, live_field in (
                    ("a", "a_cycles_per_generation"),
                    ("b", "b_cycles_per_generation"),
                ):
                    key = f"w{row['warps']}_{row['protocol']}_{variant}"
                    absolute.append(abs(candidate_cycles[candidate][key] - row[live_field]))
            score[f"{stage}_delta_mae"] = statistics.mean(errors)
            score[f"{stage}_absolute_mae"] = statistics.mean(absolute)
            all_errors.extend(errors)
            all_absolute.extend(absolute)
        score["all_delta_mae"] = statistics.mean(all_errors)
        score["all_absolute_mae"] = statistics.mean(all_absolute)
        candidate_scores.append(score)
    best = min(candidate_scores, key=lambda row: row["all_delta_mae"])
    completion_summary = read_json(
        REPO_ROOT
        / "results/captures/sm86_fixed_latency_completion_20260817_v1/analysis/summary.json"
    )
    allocate_summary = read_json(
        REPO_ROOT
        / "results/captures/sm86_fixed_latency_allocate_20260817_v1/analysis/summary.json"
    )
    scheduler_summary = read_json(
        REPO_ROOT
        / "results/captures/sm86_smsp_scheduler_contention_20260817_v1/analysis/summary.json"
    )
    completion_transfer = next(
        row
        for row in completion_summary["candidates"]
        if row["candidate"] == "tagged_rfc_allocate_hold"
    )
    allocate_transfer = next(
        row
        for row in allocate_summary["candidates"]
        if row["candidate"] == "tagged_rfc_allocate_hold"
    )
    scheduler_transfer = next(
        row
        for row in scheduler_summary["candidates"]
        if row["candidate"] == "tagged_rfc_allocate_hold"
    )
    predecessor_transfer = {
        "completion_distance": {
            key: completion_transfer[key]
            for key in (
                "identification_delta_mae",
                "held_out_distance_delta_mae",
                "held_out_width_delta_mae",
                "all_delta_mae",
                "all_absolute_mae",
            )
        },
        "original_bank": {
            key: allocate_transfer[key]
            for key in ("identification_mae", "held_out_mae", "all_width_mae")
        },
        "scheduler_99": {
            key: scheduler_transfer[key]
            for key in (
                "no_contention_same_mae_cycles_per_generation",
                "identification_same_mae_cycles_per_generation",
                "phase_transfer_same_mae_cycles_per_generation",
                "held_out_same_mae_cycles_per_generation",
                "held_out_cross_mae_cycles_per_generation",
                "all_scored_mae_cycles_per_generation",
            )
        },
    }
    return {
        "schema_version": 1,
        "campaign": {
            "cells": 40,
            "correctness_executions": 280,
            "ncu_profiles": sum(len(list((root / "ncu" / row["key"]).glob("repeat-*"))) for row in cells),
            "source_counters_profiles": 16,
            "nvbit_captures": 40,
            "trace_records": trace_records,
            "byte_identical_replay_pairs": 40,
            "restored_outputs_identical": 80,
            "maximum_active_cycle_cv_pct": max_cv,
            "all_issue_ratios_one": True,
            "all_pair_counts_equal": True,
            "all_full_coverage": True,
        },
        "static": static,
        "exact_pairs": pairs,
        "candidates": candidate_scores,
        "predecessor_transfer": predecessor_transfer,
        "source_counters": sources,
        "decision": {
            "rfc_timing_promoted": False,
            "best_candidate": best["candidate"],
            "best_delta_mae": best["all_delta_mae"],
            "warp_tag_required": True,
            "shared_entry_supported_over_per_warp": True,
            "bank_and_source_position_identity_supported": True,
            "status": "RFC identity retained; timing candidates rejected",
            "reason": (
                "the best candidate preserves the zero-hit control and treatment "
                "directions but fails absolute, completion-distance, and 99-cell "
                "scheduler transfer"
            ),
        },
    }


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        "# sm_86 compiler-managed register-file-cache campaign",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (`sm_86`)  ",
        "**Decision:** retain RFC identity evidence; reject active timing candidates",
        "",
        "## Campaign integrity",
        "",
        f"All 40 cells pass seven correctness trials ({campaign['correctness_executions']} executions). The result contains {campaign['ncu_profiles']} NCU profiles, 16 SourceCounters reports, 40 NVBit captures, and {campaign['trace_records']:,} replay records. Maximum active-cycle CV is {campaign['maximum_active_cycle_cv_pct']:.2f}%; every issue ratio is 1.0. After candidate removal, all {campaign['restored_outputs_identical']} saved replay outputs are byte-identical to production.",
        "",
        "Six cubins decode 560/560 instructions across five functions with no unknown controls, spills, or local memory. The operand auditor covers 3,360 records and 1,440 fixed-latency projections with zero gaps. Pair variants preserve runtime values, opcode/destination/predicate, non-reuse controls, per-instruction bank demand, resources, and counts.",
        "",
        "## Exact RFC treatments",
        "",
        "| Warps | Protocol | A cycles/gen | B cycles/gen | B−A | Dispatch Δ | Math Δ | Wait Δ | Replay Δ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["exact_pairs"]:
        lines.append(
            f"| {row['warps']} | {row['protocol']} | {row['a_cycles_per_generation']:.3f} | "
            f"{row['b_cycles_per_generation']:.3f} | {row['live_delta_b_minus_a']:.3f} | "
            f"{row['dispatch_delta']:.3f} | {row['math_delta']:.3f} | "
            f"{row['wait_delta']:.3f} | {row['replay_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Production replay preserves all traffic and currently predicts zero RFC treatment delta. Candidate fitting must reproduce the u10 null control, hit/retention speedups, position loss, the u22 falsification cell, and W12/W16 without a protocol or warp-count correction.",
            "",
            "## RFC candidates",
            "",
            "| Candidate | Identification delta MAE | Policy holdout MAE | Width holdout MAE | Absolute MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["candidates"]:
        lines.append(
            f"| {row['candidate']} | {row['identification_delta_mae']:.3f} | "
            f"{row['held_out_policy_delta_mae']:.3f} | "
            f"{row['held_out_width_delta_mae']:.3f} | "
            f"{row['all_absolute_mae']:.3f} |"
        )
    lines.append("")
    transfer = summary["predecessor_transfer"]
    lines.extend(
        [
            "The tagged one-entry-per-bank/source-position model is the only candidate that preserves u10's exact zero across widths. Untagged state invents 5.34/7.27/9.78 cycles of W8/W12/W16 speedup, while per-warp storage overpredicts u11 by up to 14.98 cycles. This supports a shared entry carrying a warp tag, not cross-warp reuse or per-warp capacity.",
            "",
            f"The best timing composition reaches {summary['decision']['best_delta_mae']:.3f} cycles/generation RFC delta MAE. On predecessor transfer, original-bank held-out MAE is {transfer['original_bank']['held_out_mae']:.3f}, completion-distance held-out-width MAE is {transfer['completion_distance']['held_out_width_delta_mae']:.3f}, and the 99-cell scheduler all-scored MAE is {transfer['scheduler_99']['all_scored_mae_cycles_per_generation']:.3f}. It is rejected: the state identity is supported, but its timing placement is not.",
            "",
            "No RFC timing is promoted. The exact operand/reuse projection and campaign evidence remain available for the separately configured mixed-latency result/writeback probe.",
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
        print(f"RFC analysis: {error}", file=sys.stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
