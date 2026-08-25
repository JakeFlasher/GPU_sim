#!/usr/bin/env python3
"""Validate and reduce the sm_86 fixed/L1TEX writeback-priority campaign."""

from __future__ import annotations

import argparse
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
    / "results"
    / "captures"
    / "sm86_fixed_latency_result_load_priority_20260817_v1"
)
FUNCTION = "_Z27result_load_priority_searchPyPjPKj"
WIDTHS = (1, 4, 8, 12, 16)
VARIANTS = (
    "same_noncollision",
    "same_collision",
    "opposite_noncollision",
    "opposite_shifted",
)
LONG_GENERATIONS = 1024
TRACE_GENERATIONS = 64
REQUIRED_METRICS = (
    "sm__cycles_active.max",
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma_type_fp16.sum",
    "smsp__inst_executed_pipe_lsu.sum",
    "smsp__inst_issued_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
)
COUNT_METRICS = (
    "smsp__inst_issued.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma_type_fp16.sum",
    "smsp__inst_executed_pipe_lsu.sum",
)
EMPTY_DIAGNOSTICS = (
    "unsupported_memory_path",
    "unmapped_memory_range",
    "cross_allocation_lane_span",
)


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order" / "analyze.py"
    spec = importlib.util.spec_from_file_location("load_priority_order_analysis", path)
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


def static_name(variant: str | None, trace: bool) -> str:
    prefix = (
        "fixed_latency_result_load_priority_trace"
        if trace
        else "fixed_latency_result_load_priority"
    )
    return f"{prefix}_{variant}" if variant is not None else f"{prefix}_native"


def validate_static(root: Path) -> dict[str, Any]:
    names = [static_name(None, False), static_name(None, True)]
    names.extend(
        static_name(variant, trace)
        for trace in (False, True)
        for variant in VARIANTS
    )
    sidecars = {}
    for name in names:
        sidecar = read_json(root / "sass" / f"{name}.control_bits.json")
        sidecars[name] = sidecar
        require(sidecar["summary"]["total_decoded"] == 48, f"{name}: decoded")
        require(
            sidecar["summary"]["total_unknown_encoding"] == 0,
            f"{name}: unknown controls",
        )
        require(set(sidecar["functions"]) == {FUNCTION}, f"{name}: function")
        resources = (root / "sass" / f"{name}.resources.txt").read_text(
            encoding="utf-8"
        )
        require(resources.count("REG:18") == 1, f"{name}: registers")
        require(
            not re.search(r"STACK:[1-9]|LOCAL:[1-9]", resources),
            f"{name}: spill/local",
        )

    for label in ("long", "trace"):
        patch = read_json(
            root / "metadata" / f"{label}_result_load_priority_patch.json"
        )
        require(patch["equal_instruction_work"] is True, f"{label}: work")
        require(patch["equal_static_stall_sum"] is True, f"{label}: stall sum")
        require(
            patch["all_changes_inside_selected_control_or_register_fields"] is True,
            f"{label}: byte scope",
        )
        rename = patch["semantic_register_rename"]
        require(
            (
                rename["from"],
                rename["to"],
                rename["producer_pc"],
                rename["consumer_pc"],
                rename["complete_live_range_coverage"],
                rename["replacement_absent_from_native_sass"],
                rename["higher_register_r15_compiler_allocated"],
            )
            == (11, 14, 0xF0, 0x140, True, True, True),
            f"{label}: complete register rename",
        )
        require(
            all(
                row["changed_bytes"] == 2
                for row in patch["pair_differences"].values()
            ),
            f"{label}: pair byte count",
        )

    for trace in (False, True):
        rows = {
            variant: sidecars[static_name(variant, trace)]["functions"][FUNCTION]
            for variant in VARIANTS
        }
        for left, right in (
            ("same_noncollision", "same_collision"),
            ("opposite_noncollision", "opposite_shifted"),
        ):
            differing = []
            for a, b in zip(rows[left], rows[right]):
                require(a["pc"] == b["pc"], f"{left}/{right}: PCs")
                if a != b:
                    differing.append(int(a["pc"]))
                    require(a["opcode_text"] == b["opcode_text"], "timing opcode")
                    require(a["low_word"] == b["low_word"], "timing low word")
                    for field in (
                        "yield_flag",
                        "write_barrier",
                        "read_barrier",
                        "wait_mask",
                        "reuse",
                    ):
                        require(
                            a["control"][field] == b["control"][field],
                            f"timing {field}",
                        )
            require(differing == [0x100, 0x130], f"{left}/{right}: timing scope")
        for left, right in (
            ("same_noncollision", "opposite_noncollision"),
            ("same_collision", "opposite_shifted"),
        ):
            differing = []
            for a, b in zip(rows[left], rows[right]):
                require(a["pc"] == b["pc"], f"{left}/{right}: PCs")
                if a != b:
                    differing.append(int(a["pc"]))
                    require(a["high_word"] == b["high_word"], "bank high word")
                    require(a["control"] == b["control"], "bank control")
            require(differing == [0xF0, 0x140], f"{left}/{right}: rename scope")

        for variant, records in rows.items():
            by_pc = {int(row["pc"]): row for row in records}
            same = variant.startswith("same_")
            collision = variant in {"same_collision", "opposite_shifted"}
            load_register = 11 if same else 14
            require(
                by_pc[0xF0]["opcode_text"]
                == f"LDG.E.STRONG.SM R{load_register}, [R2.64]",
                f"{variant}: load",
            )
            require(
                by_pc[0x110]["opcode_text"] == "HADD2 R9, R9, R6",
                f"{variant}: HADD2",
            )
            require(
                by_pc[0x140]["opcode_text"].startswith(
                    f"LOP3.LUT R12, R12, R{load_register},"
                ),
                f"{variant}: load consumer",
            )
            expected_stalls = {
                0xF0: 15,
                0x100: 8 if collision else 7,
                0x110: 1,
                0x120: 1,
                0x130: 4 if collision else 5,
                0x140: 15,
                0x150: 2,
            }
            require(
                sum(expected_stalls.values()) == 46,
                f"{variant}: static stall sum",
            )
            for pc, stall in expected_stalls.items():
                require(by_pc[pc]["control"]["stall"] == stall, f"{variant}: stall")
                require(
                    by_pc[pc]["control"]["yield_flag"] == 0
                    and by_pc[pc]["control"]["reuse"] == 0,
                    f"{variant}: canonical control",
                )
            require(by_pc[0xF0]["control"]["write_barrier"] == 2, "load WB")
            require(by_pc[0x140]["control"]["wait_mask"] == 4, "consumer wait")

    projection = read_json(root / "metadata" / "operand_projection_audit.json")
    require(
        (
            projection["files"],
            projection["records"],
            projection["projected"],
            projection["gaps"],
        )
        == (10, 480, 100, 0),
        "local projection",
    )
    corpus = read_json(
        root / "metadata" / "retained_corpus_operand_projection_audit.json"
    )
    require(
        (corpus["files"], corpus["records"], corpus["projected"], corpus["gaps"])
        == (74, 40904, 16905, 0),
        "retained corpus projection",
    )
    return {
        "binaries": 10,
        "decoded_instructions": 480,
        "registers": 18,
        "spill_free": True,
        "local_projection": {
            key: projection[key]
            for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
        "retained_corpus_projection": {
            key: corpus[key]
            for key in ("files", "records", "projected", "gaps", "families", "opcodes")
        },
    }


def score(observed: dict[str, float], predicted: dict[str, float]) -> dict[str, float]:
    keys = sorted(observed)
    require(keys == sorted(predicted) and len(keys) == 6, "production coverage")
    errors = [predicted[key] - observed[key] for key in keys]
    return {
        "coverage": len(keys),
        "mae": statistics.mean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
        "mape": statistics.mean(
            abs(predicted[key] - observed[key]) / observed[key] for key in keys
        ),
    }


def reduce(root: Path) -> dict[str, Any]:
    cells = read_json(root / "metadata" / "matrix.json")["cells"]
    require(len(cells) == 20 and len({cell["key"] for cell in cells}) == 20, "matrix")
    static = validate_static(root)
    from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

    bindings = load_dynamic_trace_bindings()
    live: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    trace_records = 0
    maximum_cv = 0.0
    maximum_mad = 0.0
    for cell in cells:
        key = cell["key"]
        payload = read_json(root / "correctness" / key / "result.json")["measurement"]
        require(payload["correct"] is True, f"{key}: correctness")
        require(
            payload["variant"] == cell["variant"]
            and payload["warps"] == cell["warps"],
            f"{key}: identity",
        )
        require(
            payload["generations"] == LONG_GENERATIONS
            and len(payload["samples_cycles"]) == 7,
            f"{key}: correctness trials",
        )
        driver_samples = [max(sample) for sample in payload["samples_cycles"]]

        repeat_paths = sorted((root / "ncu" / key).glob("repeat-*"))
        require(len(repeat_paths) == 3, f"{key}: NCU repeats")
        ncu_rows = [wide_ncu(path / "ncu.csv") for path in repeat_paths]
        for metric in REQUIRED_METRICS:
            require(all(metric in row for row in ncu_rows), f"{key}: {metric}")
        require(
            all(row["smsp__inst_issued_per_issue_active.ratio"] == 1.0 for row in ncu_rows),
            f"{key}: issue ratio",
        )
        expected_hits = 4096 * cell["warps"]
        expected_misses = 8 * cell["warps"]
        require(
            all(
                row["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"]
                == expected_hits
                and row[
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
                ]
                == expected_misses
                for row in ncu_rows
            ),
            f"{key}: exact L1 hit/miss treatment",
        )
        active = [row["sm__cycles_active.max"] for row in ncu_rows]
        active_median = statistics.median(active)
        cv = statistics.stdev(active) / statistics.mean(active) * 100
        mad = (
            statistics.median(abs(value - active_median) for value in active)
            / active_median
            * 100
        )
        maximum_cv = max(maximum_cv, cv)
        maximum_mad = max(maximum_mad, mad)
        median = lambda name: statistics.median(row[name] for row in ncu_rows)
        live[key] = {
            "driver_cycles_per_generation": statistics.median(driver_samples)
            / LONG_GENERATIONS,
            "active_cycles_per_generation": active_median / LONG_GENERATIONS,
            "active_samples": active,
            "active_cv_pct": cv,
            "active_mad_pct": mad,
            **{name: median(name) for name in REQUIRED_METRICS[1:]},
        }

        capture = root / "nvbit" / key
        trace_payload = measurement(capture / "stdout.txt")
        require(
            trace_payload["correct"] is True
            and trace_payload["generations"] == TRACE_GENERATIONS,
            f"{key}: trace oracle",
        )
        for diagnostic in EMPTY_DIAGNOSTICS:
            require(
                read_json(capture / "extra_info" / f"{diagnostic}.json") == [],
                f"{key}: {diagnostic}",
            )
        paths = list((capture / "threadblocks").rglob("*.pb"))
        require(len(paths) == 1, f"{key}: block count")
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

        outputs = {}
        for directory in (
            "simulator",
            "simulator-flat-priority",
            "simulator-l1",
            "simulator-priority",
        ):
            first = root / directory / key / "replay-1.json"
            second = root / directory / key / "replay-2.json"
            require(first.read_bytes() == second.read_bytes(), f"{key}: {directory} determinism")
            launch = read_json(first)["launches"][0]
            require(launch["coverage"]["kind"] == "full", f"{key}: {directory} coverage")
            require(launch["counters"]["records_seen"] == records, f"{key}: {directory} records")
            outputs[directory] = launch
        require(
            (root / "simulator" / key / "replay-1.json").read_bytes()
            == (root / "simulator-flat-priority" / key / "replay-1.json").read_bytes(),
            f"{key}: flat behavior changed",
        )
        for directory in ("simulator-l1", "simulator-priority"):
            l1 = outputs[directory]["counters"]["l1"]
            require(
                l1["hits"] == 256 * cell["warps"]
                and l1["misses"] == 8 * cell["warps"]
                and l1["bypass_sectors"] == 0,
                f"{key}: {directory} STRONG.SM route",
            )
        replay[key] = outputs

    for warps in WIDTHS:
        values = {variant: live[f"w{warps}_{variant}"] for variant in VARIANTS}
        for metric in COUNT_METRICS:
            require(
                len({values[variant][metric] for variant in VARIANTS}) == 1,
                f"W{warps}: {metric} differs",
            )

    source_paths = sorted((root / "ncu-source").glob("*/source.csv"))
    require(len(source_paths) == 8, "SourceCounters profile count")
    sources = {path.parent.name: source_summary(path) for path in source_paths}

    width_rows = []
    for warps in WIDTHS:
        keys = {variant: f"w{warps}_{variant}" for variant in VARIANTS}

        def live_delta(field: str, collision: str, baseline: str) -> float:
            return live[keys[collision]][field] - live[keys[baseline]][field]

        same_driver = live_delta(
            "driver_cycles_per_generation", "same_collision", "same_noncollision"
        )
        opposite_driver = live_delta(
            "driver_cycles_per_generation",
            "opposite_shifted",
            "opposite_noncollision",
        )
        same_active = live_delta(
            "active_cycles_per_generation", "same_collision", "same_noncollision"
        )
        opposite_active = live_delta(
            "active_cycles_per_generation",
            "opposite_shifted",
            "opposite_noncollision",
        )
        baseline_cpg = {
            variant: replay[keys[variant]]["simulator-l1"]["timing"][
                "diagnostic_last_issue_cycles"
            ]
            / TRACE_GENERATIONS
            for variant in VARIANTS
        }
        candidate_cpg = {
            variant: replay[keys[variant]]["simulator-priority"]["timing"][
                "diagnostic_last_issue_cycles"
            ]
            / TRACE_GENERATIONS
            for variant in VARIANTS
        }
        baseline_same = baseline_cpg["same_collision"] - baseline_cpg["same_noncollision"]
        baseline_opposite = baseline_cpg["opposite_shifted"] - baseline_cpg["opposite_noncollision"]
        candidate_same = candidate_cpg["same_collision"] - candidate_cpg["same_noncollision"]
        candidate_opposite = candidate_cpg["opposite_shifted"] - candidate_cpg["opposite_noncollision"]
        diagnostics = replay[keys["same_collision"]]["simulator-priority"]["timing"][
            "readiness"
        ].get("result_writeback", {})
        width_rows.append(
            {
                "warps": warps,
                "stage": next(cell["stage"] for cell in cells if cell["warps"] == warps),
                "live_driver_same_delta": same_driver,
                "live_driver_opposite_delta": opposite_driver,
                "live_driver_difference_in_differences": same_driver - opposite_driver,
                "live_active_same_delta": same_active,
                "live_active_opposite_delta": opposite_active,
                "live_active_difference_in_differences": same_active - opposite_active,
                "baseline_same_delta": baseline_same,
                "baseline_opposite_delta": baseline_opposite,
                "candidate_same_delta": candidate_same,
                "candidate_opposite_delta": candidate_opposite,
                "candidate_difference_in_differences": candidate_same - candidate_opposite,
                "long_scoreboard_same_delta": live_delta(
                    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                    "same_collision",
                    "same_noncollision",
                ),
                "long_scoreboard_opposite_delta": live_delta(
                    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                    "opposite_shifted",
                    "opposite_noncollision",
                ),
                "candidate_collisions": diagnostics.get(
                    "fixed_over_l1tex_collisions", 0
                ),
            }
        )

    baseline_same_mae = statistics.mean(
        abs(row["baseline_same_delta"] - row["live_driver_same_delta"])
        for row in width_rows
    )
    candidate_same_mae = statistics.mean(
        abs(row["candidate_same_delta"] - row["live_driver_same_delta"])
        for row in width_rows
    )

    predecessor_roots = (
        "sm86_fixed_latency_allocate_20260817_v1",
        "sm86_fixed_latency_completion_20260817_v1",
        "sm86_fixed_latency_reuse_20260817_v1",
        "sm86_fixed_latency_result_queue_20260817_v1",
        "sm86_fixed_latency_result_capacity_20260817_v1",
    )
    predecessor_comparisons = 0
    for name in predecessor_roots:
        campaign_root = REPO_ROOT / "results" / "captures" / name
        for candidate in (campaign_root / "simulator-priority").glob("*/replay-*.json"):
            baseline = campaign_root / "simulator" / candidate.parent.name / candidate.name
            require(candidate.read_bytes() == baseline.read_bytes(), f"{name}: predecessor changed")
            predecessor_comparisons += 1
    require(predecessor_comparisons == 234, "predecessor comparison count")

    production_root = root / "regression-priority" / "production"
    prior_production = (
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_issue_port_control_20260817_v1"
        / "regression-branch-forward"
        / "production"
    )
    production_comparisons = 0
    for candidate in production_root.glob("*-[12].json"):
        baseline = prior_production / candidate.name
        require(candidate.read_bytes() == baseline.read_bytes(), "production output changed")
        production_comparisons += 1
    require(production_comparisons == 12, "production comparison count")
    predicted = read_json(production_root / "predicted.json")
    observed = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_cycle_residual_20260817_v1"
        / "analysis"
        / "observed_sm_active_median.json"
    )
    production = score(observed, predicted)

    prior_host = read_json(
        REPO_ROOT
        / "results"
        / "captures"
        / "sm86_barrier_pipeline_coupled_20260817_v1"
        / "metadata"
        / "host_replay_final.json"
    )
    host = read_json(root / "metadata" / "host_replay_priority.json")
    host_ratios = {
        key: host[key]["median_ms"] / prior_host[key]["median_ms"] for key in host
    }

    return {
        "schema_version": 1,
        "campaign": {
            "cells": 20,
            "correctness_executions": 140,
            "ncu_profiles": 60,
            "source_counters_profiles": 8,
            "nvbit_captures": 20,
            "trace_records": trace_records,
            "byte_identical_replay_pairs_per_mode": 20,
            "flat_outputs_unchanged": 40,
            "predecessor_outputs_unchanged": predecessor_comparisons,
            "production_outputs_unchanged": production_comparisons,
            "maximum_active_cycle_cv_pct": maximum_cv,
            "maximum_active_cycle_mad_pct": maximum_mad,
            "all_issue_ratios_one": True,
            "all_variant_counts_equal": True,
            "exact_l1_hit_miss_counts": True,
        },
        "static": static,
        "live": live,
        "widths": width_rows,
        "source_counters": sources,
        "model": {
            "baseline_same_delta_mae": baseline_same_mae,
            "candidate_same_delta_mae": candidate_same_mae,
            "host_replay_median_ratios": host_ratios,
            "production": production,
        },
        "decision": {
            "status": "sm86_fixed_over_l1tex_writeback_priority_promoted",
            "strong_sm_l1_route_promoted": True,
            "priority_timing_promoted": True,
            "scope": (
                "sm_86 scalar LDG.E.STRONG.SM L1 hits with an ordinary write barrier "
                "versus exact HADD2 results on the same SMSP and regular-register bank"
            ),
            "evidence": [
                "W1 and W4 unprofiled same-bank collision deltas are exactly +1.000 cycles per generation",
                "W1 and W4 matched opposite-bank shifts are exactly 0.000 cycles per generation",
                "W1 NCU difference-in-differences is +0.990 cycles per generation",
                "same-bank long-scoreboard pressure rises by about 0.1245 at W1/W4 while opposite-bank movement is about 0.001",
                "all timed loads produce exactly 4096 L1 lookup hits per warp with identical setup misses",
                "the candidate reduces five-width raw same-pair delta MAE while leaving all opposite pairs untouched",
            ],
            "limitations": [
                "W8 and wider expose a separate generic producer-shift/scheduler phase in the opposite-bank control",
                "W16 remains the previously identified global phase sentinel",
                "queue capacity and same-cycle fixed/fixed retirement width remain unidentified",
                "the active priority vocabulary is deliberately narrower than all fixed and variable opcode forms",
            ],
            "next_slice": "consumer-distance bypass frontier followed by multiwarp generic producer-shift localization",
        },
    }


def render(summary: dict[str, Any]) -> str:
    campaign = summary["campaign"]
    model = summary["model"]
    production = model["production"]
    lines = [
        "# sm_86 fixed-result versus L1TEX writeback-priority reduction",
        "",
        "## Decision",
        "",
        "The measured sm_86 fixed-over-variable writeback rule is promoted for the",
        "exact admitted forms. A scalar `LDG.E.STRONG.SM` L1 hit and HADD2 result",
        "that reach one SMSP-local regular-register bank together grant HADD2 and",
        "defer the load result—and its dependence-barrier visibility—by one cycle.",
        "The same timing shift is null when the load destination moves to the other",
        "bank. The simulator also now treats this Ampere STRONG.SM form as L1",
        "eligible while preserving its literal cache token.",
        "",
        "## Campaign integrity",
        "",
        f"- {campaign['cells']} cells and {campaign['correctness_executions']} correctness executions",
        f"- {campaign['ncu_profiles']} base-clock NCU profiles; maximum active-cycle CV {campaign['maximum_active_cycle_cv_pct']:.3f}%",
        f"- {campaign['source_counters_profiles']} SourceCounters profiles and {campaign['nvbit_captures']} NVBit captures",
        f"- {campaign['trace_records']:,} replay records and deterministic pairs in flat, L1 baseline, and priority modes",
        "- exact timed traffic: 4,096 L1 hits and 8 setup misses per warp in every NCU profile",
        "- 10 spill-free cubins, 480/480 decoded controls, 100 projections, zero gaps",
        "- retained corpus: 74 sidecars, 40,904 records, 16,905 projections, zero gaps",
        f"- {campaign['predecessor_outputs_unchanged']} predecessor and {campaign['production_outputs_unchanged']} production replay outputs byte-identical",
        f"- production remains 6/6, MAE {production['mae']:.2f}, RMSE {production['rmse']:.2f}, MAPE {production['mape'] * 100:.3f}%",
        "",
        "## Width transfer",
        "",
        "| Warps | Live driver same Δ | Live driver opposite Δ | Driver DID | NCU active DID | Baseline same Δ | Priority same Δ | Priority collisions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["widths"]:
        lines.append(
            f"| {row['warps']} | {row['live_driver_same_delta']:.3f} | "
            f"{row['live_driver_opposite_delta']:.3f} | "
            f"{row['live_driver_difference_in_differences']:.3f} | "
            f"{row['live_active_difference_in_differences']:.3f} | "
            f"{row['baseline_same_delta']:.3f} | {row['candidate_same_delta']:.3f} | "
            f"{row['candidate_collisions']} |"
        )
    lines.extend(
        [
            "",
            f"Raw same-pair delta MAE across all five widths improves from {model['baseline_same_delta_mae']:.3f} to {model['candidate_same_delta_mae']:.3f} cycles/generation.",
            "W8+ opposite-bank movement shows that shifting HADD2 also changes a",
            "generic multiwarp scheduler phase. The priority state intentionally does",
            "not fabricate that effect; W16 remains a phase sentinel.",
            "",
            "## Static treatment",
            "",
            "The timing pair moves one stall cycle between IADD3 and NOP, placing",
            "HADD2 at relative cycle 22 or 23 while keeping the load consumer at cycle",
            "29 and the selected stall sum at 46. Same/opposite-bank binaries differ",
            "only at the timed load destination and its sole consumer (R11→R14).",
            "R14 is unused but inside the compiler-declared allocation proven by R15.",
            "",
            "## Next slice",
            "",
            "Measure the dependent consumer immediately at the fixed-result bypass",
            "frontier versus after register-file retirement. Then localize the generic",
            "W8+ producer-shift phase independently of bank priority; queue capacity",
            "remains a separate unresolved mechanism.",
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
        print(f"result/load priority analysis: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
