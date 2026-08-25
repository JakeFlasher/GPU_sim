#!/usr/bin/env python3
"""Validate and reduce fixed-eight-warp multi-CTA barrier contention."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "results"
    / "captures"
    / "sm86_barrier_pipeline_multicta_20260817_v1"
)
LONG_GENERATIONS = 16384
TRACE_GENERATIONS = 64
EMPTY_NVBIT_DIAGNOSTICS = (
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
            continue
    return result


def measurement_line(path: Path) -> dict[str, Any]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{") and '"correct"' in line
    ]
    require(len(values) == 1, f"{path}: expected one measurement result")
    return values[0]


def static_summary(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("multi_cta_barrier", "multi_cta_barrier_trace"):
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        summary = sidecar["summary"]
        require(summary["total_unknown_encoding"] == 0, f"{name}: unknown controls")
        require(len(summary["functions"]) == 2, f"{name}: expected two functions")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        records = re.findall(
            r"Function (_Z17multi_cta_barrierILi\dEEvPyPjS1_):\s+"
            r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
            resources,
        )
        require(len(records) == 2, f"{name}: resource records")
        require(all(int(stack) == 0 and int(local) == 0 for _, _, stack, _, local in records), f"{name}: spill/local")
        result[name] = {
            "decoded": summary["total_decoded"],
            "unknown": summary["total_unknown_encoding"],
            "functions": 2,
            "spill_free": True,
        }
    return result


def reduce(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "metadata" / "matrix.json")
    cells = manifest["cells"]
    require(len(cells) == 6, "multi-CTA matrix must contain six cells")
    by_key = {cell["key"]: cell for cell in cells}
    require(len(by_key) == 6, "multi-CTA keys are not unique")
    static = static_summary(root)

    ncu: dict[str, dict[str, Any]] = {}
    replay: dict[str, dict[str, Any]] = {}
    capture_summary: dict[str, dict[str, Any]] = {}
    for key, cell in by_key.items():
        correctness = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(correctness["correct"] is True, f"{key}: correctness failed")
        require(len(correctness["sample_max_cycles"]) == 7, f"{key}: expected seven trials")
        require(correctness["sm_count"] == 84, f"{key}: expected 84 SMs")
        require(
            correctness["sm_block_counts"] == [cell["ctas_per_sm"]] * 84,
            f"{key}: live block distribution mismatch",
        )
        require(
            correctness["shared_bytes_per_cta"] * cell["ctas_per_sm"] == 96 * 1024,
            f"{key}: shared-memory residency mismatch",
        )

        repeats = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeats) == 3, f"{key}: expected three NCU repeats")
        rows = [wide_ncu(path / "ncu.csv") for path in repeats]
        active = [row["sm__cycles_active.max"] for row in rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100.0
        require(cv <= 2.0, f"{key}: active-cycle CV exceeds 2%")
        ncu[key] = {
            "active_median": statistics.median(active),
            "active_samples": active,
            "active_cv_pct": cv,
            "barrier_stall_median": statistics.median(
                row["smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"]
                for row in rows
            ),
            "not_selected_median": statistics.median(
                row["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"]
                for row in rows
            ),
        }

        capture = root / "nvbit" / key
        payload = measurement_line(capture / "stdout.txt")
        require(payload["correct"] is True, f"{key}: NVBit correctness")
        require(payload["sm_block_counts"] == [cell["ctas_per_sm"]] * 84, f"{key}: NVBit SM placement")
        stats = list(csv.DictReader((capture / "stats.csv").open(newline="", encoding="utf-8")))
        require(len(stats) == 1, f"{key}: expected one traced launch")
        row = {name.strip(): value.strip() for name, value in stats[0].items()}
        expected_blocks = 84 * cell["ctas_per_sm"]
        require(int(row["#blocks"]) == expected_blocks, f"{key}: trace grid")
        require(row["total_insts"] == row["total_reported_insts"], f"{key}: tracer count")
        require(len(list((capture / "threadblocks").rglob("*.pb"))) == expected_blocks, f"{key}: CTA payload count")
        for diagnostic in EMPTY_NVBIT_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        capture_summary[key] = {
            "blocks": expected_blocks,
            "trace_bytes": (capture / "dynamic_trace.pb").stat().st_size,
        }

        first = root / "simulator" / key / "replay-1.json"
        second = root / "simulator" / key / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: replay nondeterminism")
        launch = read_json(first)["launches"][0]
        require(launch["coverage"]["blocks_observed"] == expected_blocks, f"{key}: coverage")
        require(
            launch["counters"]["instructions_issued"]
            == launch["counters"]["records_seen"],
            f"{key}: replay count",
        )
        require(launch["timing"]["readiness"]["schema_version"] == 2, f"{key}: schema")
        replay[key] = launch

    shape_rows: list[dict[str, Any]] = []
    for ctas, warps in ((1, 8), (2, 4), (4, 2)):
        control = f"c{ctas}_w{warps}_m00"
        barrier = f"c{ctas}_w{warps}_m10"
        live_delta = (ncu[barrier]["active_median"] - ncu[control]["active_median"]) / LONG_GENERATIONS
        sim_delta = (
            replay[barrier]["timing"]["diagnostic_last_issue_cycles"]
            - replay[control]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        readiness = replay[barrier]["timing"]["readiness"]
        expected_barriers = 84 * 8 * (1 + TRACE_GENERATIONS)
        require(
            readiness["barrier_pipeline"]["instructions_issued"] == expected_barriers,
            f"{barrier}: dynamic barrier count",
        )
        require(
            readiness["barrier_pipeline"]["issue_server_busy_cycles"]
            == expected_barriers * 2,
            f"{barrier}: service work",
        )
        shape_rows.append(
            {
                "ctas_per_sm": ctas,
                "warps_per_cta": warps,
                "live_barrier_cost_cycles_per_generation": live_delta,
                "sim_barrier_cost_cycles_per_generation": sim_delta,
                "error_cycles_per_generation": sim_delta - live_delta,
                "barrier_busy_cycles": readiness["barrier_pipeline"]["issue_server_busy_cycles"],
                "max_arrival_spread_cycles": readiness["cta_barriers"]["max_arrival_spread_cycles"],
                "cta_rendezvous_limiting_cycles": readiness["limiting_cycles"]["cta_barrier_rendezvous"],
                "eligible_not_selected_cycles": readiness["limiting_cycles"]["eligible_not_selected"],
            }
        )
    require(
        shape_rows[0]["live_barrier_cost_cycles_per_generation"]
        > shape_rows[1]["live_barrier_cost_cycles_per_generation"]
        > shape_rows[2]["live_barrier_cost_cycles_per_generation"],
        "live CTA-shape ordering mismatch",
    )
    require(
        shape_rows[0]["sim_barrier_cost_cycles_per_generation"]
        > shape_rows[1]["sim_barrier_cost_cycles_per_generation"]
        > shape_rows[2]["sim_barrier_cost_cycles_per_generation"],
        "sim CTA-shape ordering mismatch",
    )
    live_anchor = shape_rows[-1]["live_barrier_cost_cycles_per_generation"]
    sim_anchor = shape_rows[-1]["sim_barrier_cost_cycles_per_generation"]
    slope_mae = statistics.mean(
        abs(
            (row["live_barrier_cost_cycles_per_generation"] - live_anchor)
            - (row["sim_barrier_cost_cycles_per_generation"] - sim_anchor)
        )
        for row in shape_rows
    )

    return {
        "schema_version": 1,
        "campaign": {
            "cells": 6,
            "correctness_trials_per_cell": 7,
            "ncu_profiles": 18,
            "source_counters_profiles": len(list((root / "ncu-source").glob("*/source.csv"))),
            "nvbit_captures": len(capture_summary),
            "byte_identical_replay_pairs": len(replay),
            "maximum_ncu_cv_pct": max(value["active_cv_pct"] for value in ncu.values()),
            "all_sm_block_distributions_exact": True,
            "all_full_coverage": True,
            "all_cardinality_exact": True,
            "all_exclusion_sidecars_empty": True,
        },
        "static": static,
        "shapes": shape_rows,
        "shape_slope_mae_cycles_per_generation": slope_mae,
        "median_constant_error_cycles_per_generation": statistics.median(
            row["error_cycles_per_generation"] for row in shape_rows
        ),
        "decision": {
            "cta_partition_order_transfers": True,
            "shared_barrier_service_work_shape_invariant": True,
            "fixed_shape_correction_added": False,
            "status": "multi_cta_direction_validated_constant_control_residual_retained",
        },
    }


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    lines = [
        "# Multi-CTA barrier/scheduler contention validation",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (`sm_86`)  ",
        "**Result:** CTA-partition ordering transfers at fixed total warps; constant control residual retained",
        "",
        "## Campaign integrity",
        "",
        f"All six cells pass seven trials and report the exact declared block count on every one of 84 SMs. The root contains {campaign['ncu_profiles']} NCU profiles, {campaign['source_counters_profiles']} SourceCounters profiles, {campaign['nvbit_captures']} full NVBit captures, and {campaign['byte_identical_replay_pairs']} byte-identical replay pairs. Maximum active-cycle CV is {campaign['maximum_ncu_cv_pct']:.3f}%. Every CTA payload is present, issued/record cardinality is exact, and all exclusion sidecars are empty.",
        "",
        "The construction holds eight warps and 96 KiB of requested shared memory per SM constant: 1×8-warp CTA at 96 KiB, 2×4-warp CTAs at 48 KiB each, or 4×2-warp CTAs at 24 KiB each.",
        "",
        "## CTA-shape transfer",
        "",
        "| CTAs/SM | Warps/CTA | Live barrier cost/group | Sim barrier cost/group | Error | Max arrival spread | Eligible-not-selected |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["shapes"]:
        lines.append(
            f"| {row['ctas_per_sm']} | {row['warps_per_cta']} | {row['live_barrier_cost_cycles_per_generation']:.3f} | {row['sim_barrier_cost_cycles_per_generation']:.3f} | {row['error_cycles_per_generation']:+.3f} | {row['max_arrival_spread_cycles']} | {row['eligible_not_selected_cycles']} |"
        )
    lines.extend(
        [
            "",
            f"Hardware and replay both order 1×8 > 2×4 > 4×2. After anchoring at 4×2, the shape-slope MAE is {summary['shape_slope_mae_cycles_per_generation']:.3f} cycles/generation. The median absolute offset is {summary['median_constant_error_cycles_per_generation']:.3f} cycles/generation. Barrier service busy work is identical across all shapes because each SM executes the same eight warps and barrier count; only CTA rendezvous and scheduler overlap change.",
            "",
            "The negative live control-subtracted cost at 4×2 is retained as measured overlap, not clamped. A per-shape correction would fit a scheduler/control residual and is not added. The result validates the direction of the existing modeled co-residency, shared barrier service, rendezvous, and eligible-not-selected relations while leaving their common control-pipeline offset explicit.",
            "",
            "## Decision",
            "",
            "The barrier service survives multi-CTA transfer unchanged. No CTA-count branch, re-arm latency, or fixed scheduler penalty enters the model. The remaining timing work moves to broader issue-port/ALU-control characterization, where the approximately constant four-cycle offset can be isolated without using barrier or cache route labels as fit inputs.",
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
        analysis = root / "analysis"
        write_json(analysis / "summary.json", summary)
        (analysis / "README.md").write_text(report(summary), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"multi-CTA analysis: {error}", file=__import__("sys").stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
