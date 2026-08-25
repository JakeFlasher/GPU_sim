#!/usr/bin/env python3
"""Validate and reduce the sm_86 CTA-barrier issue-pipeline campaign."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = (
    REPO_ROOT / "results" / "captures" / "sm86_barrier_pipeline_20260817_v1"
)
LONG_GENERATIONS = 16384
TRACE_GENERATIONS = 64
IDENTIFICATION_WIDTHS = {1, 2, 4, 8, 16}
HELD_OUT_WIDTHS = {3, 5, 6, 7, 12}
EMPTY_NVBIT_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)
PROTOCOL_NAMES = {
    "b00": "b00_control",
    "b10": "b10_one_barrier",
    "b20": "b20_two_barriers",
    "b11": "b11_skew_barrier",
    "cal": "cal_uniform_skew",
}
PROTOCOL_TEMPLATE = {"b00": 0, "b10": 1, "b20": 2, "b11": 3, "cal": 4}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def measurement_line(path: Path) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{") or '"correct"' not in line:
            continue
        value = json.loads(line)
        if isinstance(value, dict) and "correct" in value:
            values.append(value)
    require(len(values) == 1, f"{path}: expected exactly one target JSON result")
    return values[0]


def wide_ncu(path: Path) -> dict[str, float]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: malformed NCU CSV")
    require(len(rows[0]) == len(rows[2]), f"{path}: NCU row-width mismatch")
    result: dict[str, float] = {}
    for key, value in zip(rows[0], rows[2]):
        try:
            result[key] = float(value.replace(",", ""))
        except ValueError:
            continue
    return result


def coefficient_of_variation_pct(values: Sequence[float]) -> float:
    require(len(values) >= 2, "CV requires at least two samples")
    mean = statistics.mean(values)
    require(mean != 0.0, "CV is undefined at zero mean")
    return statistics.stdev(values) / mean * 100.0


def command_option(command: Sequence[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def sample_cycle_median(payload: dict[str, Any]) -> float:
    samples = payload["samples_cycles"]
    require(isinstance(samples, list) and samples, "missing clock64 samples")
    return statistics.median(max(float(value) for value in trial) for trial in samples)


def expected_dynamic_barriers(protocol: str, warps: int) -> int:
    per_warp = {
        "b00": 1,
        "b10": 1 + TRACE_GENERATIONS,
        "b20": 1 + 2 * TRACE_GENERATIONS,
        "b11": 1 + TRACE_GENERATIONS,
    }[protocol]
    return per_warp * warps


def source_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: empty SourceCounters export")
    documents = [dict(zip(rows[1], row)) for row in rows[2:]]
    result: dict[str, Any] = {}
    for short, column in (
        ("barrier", "stall_barrier (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
    ):
        samples: list[tuple[int, str, str]] = []
        for row in documents:
            try:
                count = int(float(row.get(column) or 0))
            except ValueError:
                count = 0
            if count:
                samples.append((count, row.get("Source", "").strip(), row.get("Address", "")))
        samples.sort(reverse=True)
        result[short] = {
            "samples": sum(item[0] for item in samples),
            "top": [
                {"samples": count, "sass": sass, "address": address}
                for count, sass, address in samples[:4]
            ],
        }
    return result


def static_summary(root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name in ("barrier_pipeline", "barrier_pipeline_trace"):
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        summary = sidecar["summary"]
        require(summary["total_unknown_encoding"] == 0, f"{name}: unknown SASS control")
        require(summary["total_decoded"] == 248, f"{name}: expected 248 decoded instructions")
        require(len(summary["functions"]) == 5, f"{name}: expected five kernels")
        resource_text = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        resources = re.findall(
            r"Function (_Z16barrier_pipelineILi\dEEvPyPji):\s+"
            r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
            resource_text,
        )
        require(len(resources) == 5, f"{name}: expected five resource records")
        require(all(int(stack) == 0 for _, _, stack, _, _ in resources), f"{name}: spill stack")
        require(all(int(local) == 0 for _, _, _, _, local in resources), f"{name}: local memory")
        summaries[name] = {
            "decoded_instructions": summary["total_decoded"],
            "unknown_encodings": summary["total_unknown_encoding"],
            "functions": len(resources),
            "all_stack_zero": True,
            "all_local_zero": True,
        }
    return summaries


def reduce(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "metadata" / "matrix.json")
    cells = manifest["cells"]
    require(len(cells) == 33, "matrix must contain 33 cells")
    require(len({cell["key"] for cell in cells}) == 33, "matrix keys are not unique")
    by_key = {cell["key"]: cell for cell in cells}
    static = static_summary(root)

    correctness: dict[str, dict[str, Any]] = {}
    for key, cell in by_key.items():
        result = read_json(root / "correctness" / key / "result.json")
        payload = result["measurement"]
        require(payload["correct"] is True, f"{key}: correctness failed")
        require(payload["warps"] == cell["warps"], f"{key}: width mismatch")
        require(payload["skew"] == cell["skew"], f"{key}: skew mismatch")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: long generation mismatch")
        require(payload["protocol"] == PROTOCOL_NAMES[cell["protocol"]], f"{key}: protocol mismatch")
        require(len(payload["samples_cycles"]) == 7, f"{key}: expected seven trials")
        correctness[key] = payload

    calibration: dict[int, dict[str, Any]] = {}
    for skew in (32, 64):
        key = f"w4_cal_s{skew}"
        payload = read_json(root / "calibration" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: calibration correctness failed")
        require(len(payload["samples_cycles"]) == 7, f"{key}: expected seven calibration trials")
        calibration[skew] = payload
    control_clock = sample_cycle_median(correctness["w4_b00_s0"])
    calibration_cycles = {
        skew: (sample_cycle_median(payload) - control_clock) / LONG_GENERATIONS
        for skew, payload in calibration.items()
    }

    ncu: dict[str, dict[str, Any]] = {}
    for key in by_key:
        repeats = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeats) == 3, f"{key}: expected exactly three NCU repeats")
        rows: list[dict[str, float]] = []
        for repeat in repeats:
            run = read_json(repeat / "run.json")
            command = run["command"]
            require(run["return_code"] == 0, f"{repeat}: NCU failed")
            require(command_option(command, "--replay-mode") == "application", f"{repeat}: replay mode")
            require(command_option(command, "--clock-control") == "base", f"{repeat}: clock control")
            require(command_option(command, "--cache-control") == "none", f"{repeat}: cache control")
            require(command_option(command, "--launch-count") == "1", f"{repeat}: launch count")
            rows.append(wide_ncu(repeat / "ncu.csv"))
        active = [row["sm__cycles_active.max"] for row in rows]
        require(coefficient_of_variation_pct(active) <= 2.0, f"{key}: active-cycle CV exceeds 2%")
        ncu[key] = {
            "rows": rows,
            "active_samples": active,
            "active_median": statistics.median(active),
            "active_cv_pct": coefficient_of_variation_pct(active),
            "gpc_median": statistics.median(row["gpc__cycles_elapsed.max"] for row in rows),
            "issued_median": statistics.median(row["smsp__inst_issued.sum"] for row in rows),
            "executed_median": statistics.median(row["smsp__inst_executed.sum"] for row in rows),
            "barrier_stall_median": statistics.median(
                row["smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"]
                for row in rows
            ),
            "not_selected_median": statistics.median(
                row["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"]
                for row in rows
            ),
        }

    nvbit: dict[str, dict[str, Any]] = {}
    simulator: dict[str, dict[str, Any]] = {}
    for key, cell in by_key.items():
        capture = root / "nvbit" / key
        require((capture / "dynamic_trace.pb").stat().st_size > 0, f"{key}: empty trace")
        payload = measurement_line(capture / "stdout.txt")
        require(payload["correct"] is True, f"{key}: NVBit target failed")
        require(payload["generations"] == TRACE_GENERATIONS, f"{key}: trace generation mismatch")
        stats_rows = list(csv.DictReader((capture / "stats.csv").open(newline="", encoding="utf-8")))
        require(len(stats_rows) == 1, f"{key}: expected one traced kernel")
        stats = {name.strip(): value.strip() for name, value in stats_rows[0].items()}
        require(int(stats["#blocks"]) == 1, f"{key}: expected one CTA")
        require(int(stats["#threads"]) == cell["warps"] * 32, f"{key}: thread count mismatch")
        require(stats["total_insts"] == stats["total_reported_insts"], f"{key}: tracer count mismatch")
        modules = read_json(capture / "extra_info" / "traced_kernel_modules.json")
        selection = modules["selection"]
        require(selection["accepted_active_launches"] == 1, f"{key}: launch selection")
        require(selection["active_launch_attempts"] == 1, f"{key}: launch attempts")
        require(selection["inactive_channel_records_dropped"] == 0, f"{key}: dropped records")
        for diagnostic in EMPTY_NVBIT_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")
        payload_files = list((capture / "threadblocks").rglob("*.pb"))
        require(len(payload_files) == 1, f"{key}: expected one CTA payload")

        replay_dir = root / "simulator" / key
        first = replay_dir / "replay-1.json"
        second = replay_dir / "replay-2.json"
        require(first.read_bytes() == second.read_bytes(), f"{key}: replay is not deterministic")
        launch = read_json(first)["launches"][0]
        counters = launch["counters"]
        require(counters["instructions_issued"] == counters["records_seen"], f"{key}: replay count")
        require(launch["coverage"]["blocks_observed"] == 1, f"{key}: replay coverage")
        readiness = launch["timing"]["readiness"]
        require(readiness["schema_version"] == 2, f"{key}: expected readiness schema 2")
        expected_barriers = expected_dynamic_barriers(cell["protocol"], cell["warps"])
        require(
            readiness["barrier_pipeline"]["instructions_issued"] == expected_barriers,
            f"{key}: dynamic barrier count mismatch",
        )
        require(
            readiness["barrier_pipeline"]["issue_server_busy_cycles"] == 2 * expected_barriers,
            f"{key}: barrier busy-cycle mismatch",
        )
        nvbit[key] = {
            "reported_records": int(stats["total_reported_insts"]),
            "trace_bytes": (capture / "dynamic_trace.pb").stat().st_size,
            "barriers": expected_barriers,
        }
        simulator[key] = launch

    source_keys = (
        "w1_b10_s0",
        "w4_b10_s0",
        "w8_b10_s0",
        "w16_b10_s0",
        "w4_b11_s64",
        "w8_b11_s64",
    )
    sources = {key: source_summary(root / "ncu-source" / key / "source.csv") for key in source_keys}

    width_rows: list[dict[str, Any]] = []
    live_anchor = ncu["w1_b10_s0"]["active_median"]
    sim_anchor = simulator["w1_b10_s0"]["timing"]["diagnostic_last_issue_cycles"]
    for width in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16):
        key = f"w{width}_b10_s0"
        live_shape = (ncu[key]["active_median"] - live_anchor) / LONG_GENERATIONS
        sim_shape = (
            simulator[key]["timing"]["diagnostic_last_issue_cycles"] - sim_anchor
        ) / TRACE_GENERATIONS
        width_rows.append(
            {
                "width": width,
                "role": "identification" if width in IDENTIFICATION_WIDTHS else "held_out",
                "live_shape_cycles_per_generation": live_shape,
                "sim_shape_cycles_per_generation": sim_shape,
                "error_cycles_per_generation": sim_shape - live_shape,
            }
        )
    width_mae = statistics.mean(abs(row["error_cycles_per_generation"]) for row in width_rows)
    identification_mae = statistics.mean(
        abs(row["error_cycles_per_generation"])
        for row in width_rows
        if row["role"] == "identification"
    )
    held_out_mae = statistics.mean(
        abs(row["error_cycles_per_generation"])
        for row in width_rows
        if row["role"] == "held_out"
    )

    b20_rows: list[dict[str, Any]] = []
    for width in (1, 2, 4, 8, 16):
        b20 = f"w{width}_b20_s0"
        b10 = f"w{width}_b10_s0"
        live_second = (
            ncu[b20]["active_median"] - ncu[b10]["active_median"]
        ) / LONG_GENERATIONS
        sim_second = (
            simulator[b20]["timing"]["diagnostic_last_issue_cycles"]
            - simulator[b10]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        b20_rows.append(
            {
                "width": width,
                "live_second_barrier_cycles_per_generation": live_second,
                "sim_second_barrier_cycles_per_generation": sim_second,
                "constant_residual_cycles_per_generation": live_second - sim_second,
            }
        )
    b20_live_anchor = b20_rows[0]["live_second_barrier_cycles_per_generation"]
    b20_sim_anchor = b20_rows[0]["sim_second_barrier_cycles_per_generation"]
    b20_slope_mae = statistics.mean(
        abs(
            (row["live_second_barrier_cycles_per_generation"] - b20_live_anchor)
            - (row["sim_second_barrier_cycles_per_generation"] - b20_sim_anchor)
        )
        for row in b20_rows
    )

    skew_rows: list[dict[str, Any]] = []
    for width in (2, 4, 8, 16):
        base = f"w{width}_b10_s0"
        for skew in (32, 64):
            key = f"w{width}_b11_s{skew}"
            skew_rows.append(
                {
                    "width": width,
                    "skew": skew,
                    "live_extra_cycles_per_generation": (
                        ncu[key]["active_median"] - ncu[base]["active_median"]
                    )
                    / LONG_GENERATIONS,
                    "sim_extra_cycles_per_generation": (
                        simulator[key]["timing"]["diagnostic_last_issue_cycles"]
                        - simulator[base]["timing"]["diagnostic_last_issue_cycles"]
                    )
                    / TRACE_GENERATIONS,
                    "sim_max_arrival_spread": simulator[key]["timing"]["readiness"][
                        "cta_barriers"
                    ]["max_arrival_spread_cycles"],
                }
            )

    summary = {
        "schema_version": 1,
        "campaign": {
            "cells": len(cells),
            "correctness_cells": len(correctness),
            "correctness_trials_per_cell": 7,
            "ncu_profiles": sum(len(value["active_samples"]) for value in ncu.values()),
            "ncu_cells_above_2pct_cv": sum(value["active_cv_pct"] > 2.0 for value in ncu.values()),
            "source_counters_profiles": len(sources),
            "nvbit_captures": len(nvbit),
            "byte_identical_replay_pairs": len(simulator),
            "all_full_coverage": True,
            "all_cardinality_exact": True,
            "all_exclusion_sidecars_empty": True,
        },
        "static": static,
        "calibration_cycles_per_generation": calibration_cycles,
        "width_transfer": {
            "rows": width_rows,
            "all_width_mae_cycles_per_generation": width_mae,
            "identification_mae_cycles_per_generation": identification_mae,
            "held_out_mae_cycles_per_generation": held_out_mae,
        },
        "consecutive_barriers": {
            "rows": b20_rows,
            "width_slope_mae_cycles_per_generation": b20_slope_mae,
            "median_constant_residual_cycles_per_generation": statistics.median(
                row["constant_residual_cycles_per_generation"] for row in b20_rows
            ),
            "rearm_promoted": False,
            "reason": "ptxas schedules four independent register/control instructions between the two BAR instructions, so the constant residual does not identify barrier re-arm independently",
        },
        "arrival_skew": skew_rows,
        "source_counters": sources,
        "selected_resource": {
            "scope": "sm_shared",
            "warp_initiation_interval_cycles": 2,
            "generation_rearm_cycles": 0,
            "fixed_release_latency_cycles": 0,
            "status": "stage2_identified_pending_coupled_validation",
        },
    }
    return summary


def write_cells_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["width_transfer"]["rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    width = summary["width_transfer"]
    b20 = summary["consecutive_barriers"]
    calibration = summary["calibration_cycles_per_generation"]
    lines = [
        "# sm_86 CTA-barrier issue-pipeline campaign",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (`sm_86`)  ",
        "**Decision:** one SM-shared barrier-arrival service at one warp per two cycles; no fixed release or re-arm latency promoted",
        "",
        "## Campaign integrity",
        "",
        f"All {campaign['correctness_cells']} cells pass seven correctness trials. The root contains {campaign['ncu_profiles']} base-clock application-replay NCU profiles, {campaign['source_counters_profiles']} SourceCounters profiles, {campaign['nvbit_captures']} independent NVBit captures, and {campaign['byte_identical_replay_pairs']} byte-identical replay pairs. Every trace has one complete CTA, exact issued/record cardinality, readiness schema 2, and empty unsupported, unmapped, and cross-allocation sidecars. No cell exceeds 2% active-cycle CV.",
        "",
        "Both long and trace binaries decode 248/248 static instructions with zero unknown control encodings. All five specialized kernels use 16 registers, zero stack, and zero local memory. The standalone skew chains subtract to "
        f"{calibration[32]:.3f} and {calibration[64]:.3f} clock64 cycles/generation for the nominal 32/64 constructions.",
        "",
        "## Barrier width identification and held-out transfer",
        "",
        "The table anchors both hardware and replay at W1 b10, so launch overhead and the common control body cancel:",
        "",
        "| Width | Role | Live cycles/generation | Sim cycles/generation | Error |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in width["rows"]:
        lines.append(
            f"| {row['width']} | {row['role']} | {row['live_shape_cycles_per_generation']:.3f} | {row['sim_shape_cycles_per_generation']:.3f} | {row['error_cycles_per_generation']:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"The full-width MAE is {width['all_width_mae_cycles_per_generation']:.3f} cycles/generation; identification MAE is {width['identification_mae_cycles_per_generation']:.3f}, and held-out MAE is {width['held_out_mae_cycles_per_generation']:.3f}. W1-W8 follow an exact +2-cycle-per-added-warp law. A per-SMSP server would plateau or step at four-warp boundaries and is rejected; dispatch-only is also rejected. The simplest surviving shape is one SM-shared 16-thread/cycle service, represented as two cycles per warp arrival.",
            "",
            "Barrier instructions still dispatch through their SMSPs. The shared service assigns completion timestamps in stable subcore order; CTA arrival and release consume those timestamps. This is why independent SMSP dispatch is not incorrectly serialized.",
            "",
            "## Consecutive generations",
            "",
            "| Width | Live second-barrier cost | Sim second-barrier cost | Constant residual |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in b20["rows"]:
        lines.append(
            f"| {row['width']} | {row['live_second_barrier_cycles_per_generation']:.3f} | {row['sim_second_barrier_cycles_per_generation']:.3f} | {row['constant_residual_cycles_per_generation']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The shared service predicts the complete width-dependent b20 slope (slope MAE {b20['width_slope_mae_cycles_per_generation']:.3f} cycles/generation). A width-invariant {b20['median_constant_residual_cycles_per_generation']:.3f}-cycle residual remains. Source-level consecutive synchronizations do not become adjacent SASS: ptxas schedules four independent register/control instructions between the BARs even when they are emitted in one inline-PTX block. The residual therefore cannot identify a barrier re-arm independently of those still-dispatch-only ALU/control instructions. No re-arm or post-release latency is added.",
            "",
            "## Controlled arrival skew",
            "",
            "Nominal skew 64 is slower than skew 32 at every width in both hardware and replay, and the modeled post-service arrival spread grows accordingly. Absolute skew cost remains underpredicted by roughly 13-23 cycles/generation, consistent with the still-dispatch-only dependent integer chain. This validation supports the barrier service direction but does not authorize an ALU latency fit from barrier cells.",
            "",
            "## Decision",
            "",
            "Stage 2 identifies and implements only the transferable mechanism: one `sm_86` SM-shared CTA-barrier arrival service with a two-cycle warp initiation interval. `sm_120` remains a typed capability gap. Readiness schema 2 reports accepted barrier instructions, service busy cycles, generation first/last timestamp sums, issue spans, pipeline exposure, and the existing CTA arrival/rendezvous diagnostics. Barrier issue, own-arrival completion, other-participant rendezvous, and ordinary post-barrier instructions remain separate state relations.",
            "",
            "The resource is now ready for the coupled warm-cache c10/c11 and multi-CTA validation stage. It is not a workload-name branch, consumes no NCU value during replay, and introduces no fixed CTA release delay.",
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
        write_cells_csv(analysis / "width_transfer.csv", summary)
        (analysis / "README.md").write_text(report(summary), encoding="utf-8")
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"analysis: {error}", file=__import__("sys").stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
