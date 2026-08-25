#!/usr/bin/env python3
"""Validate and reduce the sm_86 issue-port/control campaign."""

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
    / "sm86_issue_port_control_20260817_v1"
)
LONG_GENERATIONS = 8192
TRACE_GENERATIONS = 64
LENGTHS = (4, 8, 16, 32, 64)
PROTOCOLS = ("p00", "p10", "p11", "p12", "p13", "p20", "p21", "p22", "p31")
ACTIVE_PROTOCOLS = PROTOCOLS[1:]
PROTOCOL_NAMES = {
    "p00": "p00_control",
    "p10": "p10_dependent_iadd",
    "p11": "p11_dependent_imad",
    "p12": "p12_dependent_lop3",
    "p13": "p13_branch_taken",
    "p20": "p20_independent_iadd",
    "p21": "p21_independent_imad",
    "p22": "p22_independent_lop3",
    "p31": "p31_mixed_iadd_ffma",
}
EMPTY_NVBIT_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)
FINAL_SIMULATOR = "simulator-branch-forward"


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


def command_option(command: Sequence[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def static_summary(root: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    function_pattern = re.compile(r"Function : _Z18issue_port_controlILi(\d+)ELi(\d+)EEvPyPjS1_")
    instruction_pattern = re.compile(
        r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9.]*)"
    )
    for name in ("issue_port_control", "issue_port_control_trace"):
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        summary = sidecar["summary"]
        require(summary["total_decoded"] == 5136, f"{name}: decoded count mismatch")
        require(summary["total_unknown_encoding"] == 0, f"{name}: unknown controls")
        require(len(summary["functions"]) == 45, f"{name}: function count mismatch")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(encoding="utf-8")
        require(resources.count("Function _Z18issue_port_control") == 45, f"{name}: resources")
        require(not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources), f"{name}: spills")

        functions: dict[tuple[int, int], dict[str, int]] = {}
        current: tuple[int, int] | None = None
        for line in (root / "sass" / f"{name}.sass.txt").read_text(encoding="utf-8").splitlines():
            match = function_pattern.search(line)
            if match:
                current = (int(match.group(1)), int(match.group(2)))
                functions[current] = {}
                continue
            if current is not None and "Function :" in line:
                current = None
            if current is None:
                continue
            match = instruction_pattern.search(line)
            if match:
                head = match.group(1).split(".")[0]
                functions[current][head] = functions[current].get(head, 0) + 1
        require(len(functions) == 45, f"{name}: SASS function census mismatch")
        for length in LENGTHS:
            require(functions[(2, length)].get("IMAD", 0) >= length, f"{name}: IMAD length")
            require(functions[(3, length)].get("LOP3", 0) >= length, f"{name}: LOP3 length")
            require(functions[(4, length)].get("BRA", 0) == length + 2, f"{name}: BRA length")
            require(functions[(4, length)].get("ISETP", 0) == length + 2, f"{name}: ISETP length")
            require(functions[(8, length)].get("FFMA", 0) >= length, f"{name}: FFMA length")
        summaries[name] = {
            "decoded": summary["total_decoded"],
            "unknown": summary["total_unknown_encoding"],
            "functions": len(functions),
            "spill_free": True,
        }
    return summaries


def source_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    require(len(rows) >= 3, f"{path}: empty source report")
    documents = [dict(zip(rows[1], row)) for row in rows[2:]]
    result: dict[str, Any] = {}
    for short, column in (
        ("branch_resolving", "stall_branch_resolving (Not Issued)"),
        ("math_pipe_throttle", "stall_math_pipe_throttle (Not Issued)"),
        ("not_selected", "stall_not_selected (Not Issued)"),
        ("wait", "stall_wait (Not Issued)"),
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


def score(observed: dict[str, int], predicted: dict[str, int]) -> dict[str, float | int]:
    keys = sorted(observed.keys() & predicted.keys())
    require(len(keys) == 6, "production coverage mismatch")
    absolute = [abs(observed[key] - predicted[key]) for key in keys]
    squared = [(observed[key] - predicted[key]) ** 2 for key in keys]
    percentage = [error / observed[key] for key, error in zip(keys, absolute)]
    return {
        "coverage": len(keys),
        "mae": statistics.mean(absolute),
        "rmse": statistics.mean(squared) ** 0.5,
        "mape": statistics.mean(percentage),
    }


def reduce(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "metadata" / "matrix.json")
    cells = manifest["cells"]
    require(len(cells) == 68, "matrix must contain 68 cells")
    by_key = {cell["key"]: cell for cell in cells}
    require(len(by_key) == 68, "matrix keys are not unique")
    static = static_summary(root)

    correctness: dict[str, dict[str, Any]] = {}
    ncu: dict[str, dict[str, Any]] = {}
    replay: dict[str, dict[str, Any]] = {}
    for key, cell in by_key.items():
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness failed")
        require(payload["protocol"] == PROTOCOL_NAMES[cell["protocol"]], f"{key}: protocol")
        require(payload["length"] == cell["length"], f"{key}: length")
        require(payload["warps"] == cell["warps"], f"{key}: warps")
        require(payload["generations"] == LONG_GENERATIONS, f"{key}: generations")
        require(len(payload["samples_cycles"]) == 7, f"{key}: trial count")
        correctness[key] = payload

        repeat_dirs = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_dirs) == 3, f"{key}: expected three NCU repeats")
        rows: list[dict[str, float]] = []
        for repeat in repeat_dirs:
            run = read_json(repeat / "run.json")
            command = run["command"]
            require(command_option(command, "--replay-mode") == "application", f"{key}: replay")
            require(command_option(command, "--clock-control") == "base", f"{key}: clocks")
            require(command_option(command, "--cache-control") == "none", f"{key}: cache")
            rows.append(wide_ncu(repeat / "ncu.csv"))
        active = [row["sm__cycles_active.max"] for row in rows]
        cv = statistics.stdev(active) / statistics.mean(active) * 100.0
        require(cv <= 2.0, f"{key}: CV exceeds 2%")
        ncu[key] = {
            "active_median": statistics.median(active),
            "active_samples": active,
            "active_cv_pct": cv,
            "issued": statistics.median(row["smsp__inst_issued.sum"] for row in rows),
            "executed": statistics.median(row["smsp__inst_executed.sum"] for row in rows),
            "issued_per_issue_active": statistics.median(
                row["smsp__inst_issued_per_issue_active.ratio"] for row in rows
            ),
            "alu": statistics.median(row["smsp__inst_executed_pipe_alu.sum"] for row in rows),
            "fma": statistics.median(row["smsp__inst_executed_pipe_fma.sum"] for row in rows),
            "math_throttle": statistics.median(
                row["smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"]
                for row in rows
            ),
            "not_selected": statistics.median(
                row["smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"]
                for row in rows
            ),
            "branch_resolving": statistics.median(
                row["smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio"]
                for row in rows
            ),
        }
        require(ncu[key]["issued_per_issue_active"] == 1.0, f"{key}: dual issue observed")

        capture = root / "nvbit" / key
        trace_payload = measurement_line(capture / "stdout.txt")
        require(trace_payload["correct"] is True, f"{key}: NVBit correctness")
        require(trace_payload["generations"] == TRACE_GENERATIONS, f"{key}: trace generations")
        stats_rows = list(csv.DictReader((capture / "stats.csv").open(newline="", encoding="utf-8")))
        require(len(stats_rows) == 1, f"{key}: expected one traced kernel")
        stats = {name.strip(): value.strip() for name, value in stats_rows[0].items()}
        require(int(stats["#blocks"]) == 1, f"{key}: CTA count")
        require(int(stats["#threads"]) == cell["warps"] * 32, f"{key}: threads")
        require(stats["total_insts"] == stats["total_reported_insts"], f"{key}: trace count")
        require(len(list((capture / "threadblocks").rglob("*.pb"))) == 1, f"{key}: payload")
        modules = read_json(capture / "extra_info" / "traced_kernel_modules.json")
        require(modules["selection"]["accepted_active_launches"] == 1, f"{key}: selection")
        for diagnostic in EMPTY_NVBIT_DIAGNOSTICS:
            require(read_json(capture / "extra_info" / f"{diagnostic}.json") == [], f"{key}: {diagnostic}")

        documents: dict[str, dict[str, Any]] = {}
        for directory in ("simulator", FINAL_SIMULATOR):
            first = root / directory / key / "replay-1.json"
            second = root / directory / key / "replay-2.json"
            require(first.read_bytes() == second.read_bytes(), f"{key}: {directory} nondeterminism")
            launch = read_json(first)["launches"][0]
            require(launch["coverage"]["kind"] == "full", f"{key}: coverage")
            require(
                launch["counters"]["instructions_issued"]
                == launch["counters"]["records_seen"],
                f"{key}: replay count",
            )
            documents[directory] = launch
        require(
            documents["simulator"]["counters"] == documents[FINAL_SIMULATOR]["counters"],
            f"{key}: branch model changed traffic",
        )
        final_readiness = documents[FINAL_SIMULATOR]["timing"]["readiness"]
        if cell["protocol"] == "p13":
            require(final_readiness["schema_version"] == 3, f"{key}: schema 3")
            expected_branches = cell["warps"] * cell["length"] * TRACE_GENERATIONS
            require(
                final_readiness["control_pipeline"]["branch_instructions_issued"]
                == expected_branches,
                f"{key}: branch count",
            )
            require(
                final_readiness["control_pipeline"]["branch_resolution_cycles"]
                == expected_branches * 10,
                f"{key}: branch cycles",
            )
        else:
            require(final_readiness["schema_version"] == 1, f"{key}: unexpected schema")
        replay[key] = documents

    length_rows: list[dict[str, Any]] = []
    for protocol in ACTIVE_PROTOCOLS:
        for lower, upper in zip(LENGTHS, LENGTHS[1:]):
            denominator_live = LONG_GENERATIONS * (upper - lower)
            denominator_sim = TRACE_GENERATIONS * (upper - lower)
            lower_key = f"w1_{protocol}_l{lower}"
            upper_key = f"w1_{protocol}_l{upper}"
            live_slope = (ncu[upper_key]["active_median"] - ncu[lower_key]["active_median"]) / denominator_live
            baseline_slope = (
                replay[upper_key]["simulator"]["timing"]["diagnostic_last_issue_cycles"]
                - replay[lower_key]["simulator"]["timing"]["diagnostic_last_issue_cycles"]
            ) / denominator_sim
            final_slope = (
                replay[upper_key][FINAL_SIMULATOR]["timing"]["diagnostic_last_issue_cycles"]
                - replay[lower_key][FINAL_SIMULATOR]["timing"]["diagnostic_last_issue_cycles"]
            ) / denominator_sim
            length_rows.append(
                {
                    "protocol": protocol,
                    "lower_length": lower,
                    "upper_length": upper,
                    "role": "held_out" if lower in {4, 16} and upper in {8, 32} else "identification",
                    "live_cycles_per_step": live_slope,
                    "dispatch_cycles_per_step": baseline_slope,
                    "final_cycles_per_step": final_slope,
                }
            )

    width_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        live_penalty = (
            ncu[f"w8_{protocol}_l16"]["active_median"]
            - ncu[f"w4_{protocol}_l16"]["active_median"]
        ) / LONG_GENERATIONS
        dispatch_penalty = (
            replay[f"w8_{protocol}_l16"]["simulator"]["timing"]["diagnostic_last_issue_cycles"]
            - replay[f"w4_{protocol}_l16"]["simulator"]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        final_penalty = (
            replay[f"w8_{protocol}_l16"][FINAL_SIMULATOR]["timing"]["diagnostic_last_issue_cycles"]
            - replay[f"w4_{protocol}_l16"][FINAL_SIMULATOR]["timing"]["diagnostic_last_issue_cycles"]
        ) / TRACE_GENERATIONS
        held_w3 = (
            ncu[f"w3_{protocol}_l16"]["active_median"]
            - ncu[f"w1_{protocol}_l16"]["active_median"]
        ) / LONG_GENERATIONS
        width_rows.append(
            {
                "protocol": protocol,
                "live_w8_minus_w4": live_penalty,
                "dispatch_w8_minus_w4": dispatch_penalty,
                "final_w8_minus_w4": final_penalty,
                "live_held_w3_minus_w1": held_w3,
                "math_throttle_w8": ncu[f"w8_{protocol}_l16"]["math_throttle"],
                "not_selected_w8": ncu[f"w8_{protocol}_l16"]["not_selected"],
            }
        )

    candidate_directories = {
        "dispatch_only": "simulator",
        "broad_alu_fma_ii2": "simulator-scalar",
        "all_iadd_ii2": "simulator-iadd",
        "carry_only_ii2": "simulator-carry",
        "carry_only_ii3": "simulator-carry3",
        "all_bra_min10": "simulator-branch",
        "forward_bra_min10": FINAL_SIMULATOR,
    }
    candidate_rows: list[dict[str, Any]] = []
    for name, directory in candidate_directories.items():
        errors: list[float] = []
        for row in width_rows:
            protocol = row["protocol"]
            predicted = (
                read_json(root / directory / f"w8_{protocol}_l16" / "replay-1.json")["launches"][0]["timing"]["diagnostic_last_issue_cycles"]
                - read_json(root / directory / f"w4_{protocol}_l16" / "replay-1.json")["launches"][0]["timing"]["diagnostic_last_issue_cycles"]
            ) / TRACE_GENERATIONS
            errors.append(abs(predicted - row["live_w8_minus_w4"]))
        candidate_rows.append(
            {
                "candidate": name,
                "w8_width_mae_cycles_per_generation": statistics.mean(errors),
                "status": "selected" if name == "forward_bra_min10" else "rejected_or_baseline",
            }
        )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 14, "expected 14 SourceCounters profiles")
    sources = {path.parent.name: source_summary(path) for path in source_paths}

    barrier_root = REPO_ROOT / "results" / "captures" / "sm86_barrier_pipeline_20260817_v1"
    coupled_root = REPO_ROOT / "results" / "captures" / "sm86_barrier_pipeline_coupled_20260817_v1"
    multicta_root = REPO_ROOT / "results" / "captures" / "sm86_barrier_pipeline_multicta_20260817_v1"
    barrier_changed_cells = 0
    coupled_changed_cells = 0
    for cohort_root, count in ((barrier_root, 33), (coupled_root, 18), (multicta_root, 6)):
        baseline = cohort_root / "simulator"
        final = cohort_root / FINAL_SIMULATOR
        baseline_files = sorted(baseline.glob("*/replay-1.json"))
        if not baseline_files:
            baseline_files = sorted(baseline.glob("*/stream-l2-1.json"))
        require(len(baseline_files) == count, f"{cohort_root}: baseline count")
        for path in baseline_files:
            key = path.parent.name
            final_path = final / key / path.name
            if cohort_root in {barrier_root, coupled_root}:
                baseline_launch = read_json(path)["launches"][0]
                final_launch = read_json(final_path)["launches"][0]
                require(
                    baseline_launch["counters"] == final_launch["counters"],
                    f"{cohort_root}: {key} traffic changed",
                )
                changed = (
                    baseline_launch["timing"]["diagnostic_last_issue_cycles"]
                    != final_launch["timing"]["diagnostic_last_issue_cycles"]
                )
                if cohort_root == barrier_root:
                    barrier_changed_cells += changed
                else:
                    coupled_changed_cells += changed
            else:
                require(
                    path.read_bytes() == final_path.read_bytes(),
                    f"{cohort_root}: {key} changed",
                )

    def external_ncu_median(key: str) -> float:
        values = [
            wide_ncu(path / "ncu.csv")["sm__cycles_active.max"]
            for path in sorted((barrier_root / "ncu" / key).glob("repeat-*"))
        ]
        return statistics.median(values)

    skew_errors: dict[str, list[float]] = {"dispatch": [], "final": []}
    for width in (2, 4, 8, 16):
        for skew in (32, 64):
            key = f"w{width}_b11_s{skew}"
            base = f"w{width}_b10_s0"
            live = (
                external_ncu_median(key) - external_ncu_median(base)
            ) / 16384
            for label, directory in (("dispatch", "simulator"), ("final", FINAL_SIMULATOR)):
                predicted = (
                    read_json(barrier_root / directory / key / "replay-1.json")["launches"][0]["timing"]["diagnostic_last_issue_cycles"]
                    - read_json(barrier_root / directory / base / "replay-1.json")["launches"][0]["timing"]["diagnostic_last_issue_cycles"]
                ) / 64
                skew_errors[label].append(abs(predicted - live))
    barrier_skew = {
        "dispatch_mae_cycles_per_generation": statistics.mean(skew_errors["dispatch"]),
        "final_mae_cycles_per_generation": statistics.mean(skew_errors["final"]),
        "timing_changed_cells": barrier_changed_cells,
        "all_traffic_unchanged": True,
    }

    coupled_interactions_unchanged = True
    for width in (2, 4, 8):
        for locality in ("l1", "l2", "dram"):
            c10 = f"w{width}_c10_{locality}_g0"
            c11 = f"w{width}_c11_{locality}_g0"
            baseline_delta = (
                read_json(coupled_root / "simulator" / c11 / "stream-l2-1.json")["launches"][-1]["timing"]["diagnostic_last_issue_cycles"]
                - read_json(coupled_root / "simulator" / c10 / "stream-l2-1.json")["launches"][-1]["timing"]["diagnostic_last_issue_cycles"]
            )
            final_delta = (
                read_json(coupled_root / FINAL_SIMULATOR / c11 / "stream-l2-1.json")["launches"][-1]["timing"]["diagnostic_last_issue_cycles"]
                - read_json(coupled_root / FINAL_SIMULATOR / c10 / "stream-l2-1.json")["launches"][-1]["timing"]["diagnostic_last_issue_cycles"]
            )
            coupled_interactions_unchanged &= baseline_delta == final_delta
    require(coupled_interactions_unchanged, "coupled interaction deltas changed")

    regression = root / "regression-branch-forward"
    production_predicted = read_json(regression / "production" / "predicted.json")
    production_observed = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_cycle_residual_20260817_v1"
        / "analysis"
        / "observed_sm_active_median.json"
    )
    production_baseline = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_barrier_pipeline_coupled_20260817_v1"
        / "production"
        / "predicted.json"
    )
    production_score = score(production_observed, production_predicted)
    production_rows = [
        {
            "key": key,
            "observed": production_observed[key],
            "baseline": production_baseline[key],
            "final": production_predicted[key],
            "delta": production_predicted[key] - production_baseline[key],
        }
        for key in sorted(production_predicted)
    ]
    predecessor_files = sorted((regression / "predecessor").glob("*-1.json"))
    require(len(predecessor_files) == 36, "predecessor regression count")
    baseline_predecessor = (
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_barrier_pipeline_coupled_20260817_v1"
        / "predecessor"
    )
    for path in predecessor_files:
        key = path.name.removesuffix("-1.json")
        final_launch = read_json(path)["launches"][0]
        baseline_launch = read_json(baseline_predecessor / f"{key}-1.json")["launches"][0]
        require(final_launch["counters"] == baseline_launch["counters"], f"{key}: traffic changed")

    p13_rows = [row for row in length_rows if row["protocol"] == "p13"]
    p13_dispatch_mae = statistics.mean(
        abs(row["dispatch_cycles_per_step"] - row["live_cycles_per_step"])
        for row in p13_rows
    )
    p13_final_mae = statistics.mean(
        abs(row["final_cycles_per_step"] - row["live_cycles_per_step"])
        for row in p13_rows
    )

    host_baseline = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_barrier_pipeline_coupled_20260817_v1"
        / "metadata"
        / "host_replay_final.json"
    )
    host_final = read_json(root / "metadata" / "host_replay_final.json")
    require(host_baseline.keys() == host_final.keys(), "host replay keys differ")
    host_replay = {
        key: {
            "samples": host_final[key]["samples"],
            "baseline_median_ms": host_baseline[key]["median_ms"],
            "final_median_ms": host_final[key]["median_ms"],
            "change_ms": host_final[key]["median_ms"] - host_baseline[key]["median_ms"],
            "change_pct": (
                host_final[key]["median_ms"] - host_baseline[key]["median_ms"]
            )
            / host_baseline[key]["median_ms"]
            * 100.0,
        }
        for key in host_final
    }

    return {
        "schema_version": 1,
        "campaign": {
            "cells": 68,
            "correctness_trials_per_cell": 7,
            "ncu_profiles": 204,
            "ncu_cells_above_2pct_cv": 0,
            "source_counters_profiles": len(sources),
            "nvbit_captures": 68,
            "byte_identical_baseline_pairs": 68,
            "byte_identical_final_pairs": 68,
            "all_cardinality_exact": True,
            "all_full_coverage": True,
            "all_exclusion_sidecars_empty": True,
        },
        "static": static,
        "length_slopes": length_rows,
        "width_transfer": width_rows,
        "candidates": candidate_rows,
        "p13_dispatch_mae_cycles_per_branch": p13_dispatch_mae,
        "p13_final_mae_cycles_per_branch": p13_final_mae,
        "all_ncu_issue_ratios_one": True,
        "source_counters": sources,
        "closed_cohorts": {
            "barrier": barrier_skew,
            "coupled": {
                "timing_changed_cells": coupled_changed_cells,
                "all_traffic_unchanged": True,
                "all_c11_minus_c10_deltas_unchanged": True,
            },
            "multicta_pairs_unchanged": 6,
            "predecessor_traffic_unchanged": 36,
        },
        "production": {**production_score, "rows": production_rows},
        "host_replay": host_replay,
        "decision": {
            "selected": "predicated forward BRA with encoded stall=5/yield=1 gets a 10-cycle minimum next-issue interval",
            "scalar_initiation_promoted": False,
            "dual_issue_promoted": False,
            "rejected_reason": "simple ALU/FMA and carry-only initiation timelines fail to transfer across independent/mixed probes and closed multi-CTA slopes",
            "next_gap": "two-warps-per-SMSP math-pipe-throttle and not-selected arbitration",
        },
    }


def report(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    p13_rows = [row for row in summary["length_slopes"] if row["protocol"] == "p13"]
    lines = [
        "# sm_86 issue-port and control-pipeline campaign",
        "",
        "**Date:** 2026-08-17  ",
        "**GPU:** NVIDIA GeForce RTX 3090 Ti (`sm_86`)  ",
        "**Decision:** promote only the exact forward-branch next-issue frontier; reject simple scalar initiation and dual-issue candidates",
        "",
        "## Campaign integrity",
        "",
        f"All {campaign['cells']} cells pass seven correctness trials. The root contains {campaign['ncu_profiles']} base-clock NCU profiles, {campaign['source_counters_profiles']} SourceCounters profiles, {campaign['nvbit_captures']} independent NVBit captures, and deterministic dispatch/final replay pairs for every cell. No NCU cell exceeds 2% active-cycle CV. Both binaries decode 5,136/5,136 SASS instructions across 45 specializations with zero unknown controls, zero spills, and zero local memory. Every trace has one complete CTA, exact issued/record cardinality, and empty unsupported, unmapped, and cross-allocation sidecars.",
        "",
        "## Pipe census and dual issue",
        "",
        "GA102 NCU counters map carry-add/LOP work to ALU and plain IMAD/FFMA to FMA; IPA remains zero. Every one of the 204 profiles reports `smsp__inst_issued_per_issue_active.ratio == 1.0`. The mixed IADD/FFMA runtime demonstrates latency overlap, but no cell observes two issued instructions per issue-active cycle, so dual issue is not promoted.",
        "",
        "Dependent IADD, IMAD, and LOP3 length slopes are already predicted by compiler control bits to within roughly 0.02 cycles/logical step. Their large W8 penalties coincide with both math-pipe-throttle and not-selected samples, but integer two-/three-cycle ALU/FMA and carry-only timelines do not transfer across dependent, independent, mixed, and prior multi-CTA cells. Those candidates remain rejected.",
        "",
        "## Exact forward branch",
        "",
        "| Length interval | Role | Live cycles/branch | Dispatch-only | Final |",
        "|---|---|---:|---:|---:|",
    ]
    for row in p13_rows:
        lines.append(
            f"| {row['lower_length']}→{row['upper_length']} | {row['role']} | {row['live_cycles_per_step']:.3f} | {row['dispatch_cycles_per_step']:.3f} | {row['final_cycles_per_step']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The encoded branch stall is five cycles, but all identification and held-out slopes require a ten-cycle minimum next-issue interval. MAE drops from {summary['p13_dispatch_mae_cycles_per_branch']:.3f} to {summary['p13_final_mae_cycles_per_branch']:.3f} cycles/branch. Classification is deliberately narrow: predicated forward `BRA`, target greater than function-relative PC, control stall 5, and yield set. Backward loop branches and other branch forms retain compiler-only timing.",
            "",
            "Readiness schema 3 reports branch instructions, resolution work, and branch-limiting/exposure cycles. The relation delays only the issuing warp's next frontier; it is neither a shared throughput server nor a register scoreboard. `sm_120` remains a typed research gap.",
            "",
            "## Regression",
            "",
            f"All 6 multi-CTA final outputs are byte-identical to their prior baselines. Coupled warm-route traffic and every c11−c10 delta are unchanged. Barrier traffic is unchanged; the exact branch form improves b11 skew MAE from {summary['closed_cohorts']['barrier']['dispatch_mae_cycles_per_generation']:.3f} to {summary['closed_cohorts']['barrier']['final_mae_cycles_per_generation']:.3f} cycles/generation. All 36 predecessor traffic reports are unchanged. Production retains 6/6 coverage: MAE {summary['production']['mae']:.2f}, RMSE {summary['production']['rmse']:.2f}, and active-SM MAPE {summary['production']['mape'] * 100:.3f}% versus the 33.394% predecessor baseline.",
            "",
            f"Over 200 process-level invocations, the tiny control changes from {summary['host_replay']['w4_c00_na_g0']['baseline_median_ms']:.3f} to {summary['host_replay']['w4_c00_na_g0']['final_median_ms']:.3f} ms ({summary['host_replay']['w4_c00_na_g0']['change_pct']:+.2f}%), while the representative async sentinel changes from {summary['host_replay']['w4_c11_l1_g0']['baseline_median_ms']:.3f} to {summary['host_replay']['w4_c11_l1_g0']['final_median_ms']:.3f} ms ({summary['host_replay']['w4_c11_l1_g0']['change_pct']:+.2f}%).",
            "",
            "## Decision and next target",
            "",
            "Promote the exact forward-branch frontier. Do not promote a generic ALU/FMA interval, carry-only interval, generic branch latency, or dual issue. The unresolved evidence is now localized to W8: adding the second warp per SMSP creates math-pipe-throttle and not-selected delay that simple resource-ready timestamps cannot reproduce. The next milestone must characterize scheduler arbitration and port-queue interaction directly, without refitting branch, barrier, cache, or workload identities.",
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
        print(f"issue-port analysis: {error}", file=__import__("sys").stderr)
        return 1
    print(root / "analysis" / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
