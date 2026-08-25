#!/usr/bin/env python3
"""Reduce the sm_86 memory-retirement service campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from functools import cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import campaign
from campaign import DEFAULT_OUTPUT, configured_cells, source_cells


GPC = "gpc__cycles_elapsed.max"
GPC_AVG = "gpc__cycles_elapsed.avg"
GPC_MIN = "gpc__cycles_elapsed.min"
GPC_SUM = "gpc__cycles_elapsed.sum"
SM_ACTIVE = "sm__cycles_active.max"
INST = "smsp__inst_executed.sum"


def median(values: Iterable[float]) -> float:
    material = list(values)
    if not material:
        raise RuntimeError("empty median")
    return float(statistics.median(material))


def mean(values: Iterable[float]) -> float:
    material = list(values)
    if not material:
        raise RuntimeError("empty mean")
    return float(statistics.fmean(material))


def cv(values: Sequence[float]) -> float:
    center = mean(values)
    return 0.0 if len(values) < 2 or center == 0 else statistics.stdev(values) / center


def mae(predicted: Iterable[float], observed: Iterable[float]) -> float:
    return mean(abs(p - o) for p, o in zip(predicted, observed, strict=True))


def maximum_error(predicted: Iterable[float], observed: Iterable[float]) -> float:
    return max(abs(p - o) for p, o in zip(predicted, observed, strict=True))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ncu_row(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("ID") or "").strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one NCU row, got {len(rows)}")
    result: dict[str, float] = {}
    for key, raw in rows[0].items():
        if raw is None or not raw.strip():
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            result[key] = value
    return result


def observed(root: Path, key: str) -> tuple[dict[str, float], dict[str, float]]:
    paths = sorted((root / "ncu" / key).glob("repeat-*/ncu.csv"))
    if len(paths) not in {3, 5, 7}:
        raise RuntimeError(f"{key}: expected three, five, or seven NCU CSVs, got {len(paths)}")
    rows = [ncu_row(path) for path in paths]
    keys = set.intersection(*(set(row) for row in rows))
    medians = {name: median(row[name] for row in rows) for name in keys}
    cvs = {name: cv([row[name] for row in rows]) for name in keys}
    return medians, cvs


@cache
def replay(root: Path, directory: str, key: str) -> dict:
    cells = {cell.key: cell for cell in configured_cells()}
    try:
        cell = cells[key]
    except KeyError as error:
        raise RuntimeError(f"unknown replay cell: {key}") from error
    first = root / directory / key / "replay-1.json"
    second = root / directory / key / "replay-2.json"
    if directory == "simulator":
        carded = False
    elif directory == "simulator-carded-dram":
        carded = True
    else:
        raise RuntimeError(f"unknown replay directory: {directory}")
    trace = root / "nvbit" / key
    dependencies = [
        campaign.REPO_ROOT / "sim/target/release/sim-run",
        root / "sass/memory_retirement_service_trace.control_bits.json",
        campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json",
        campaign.REPO_ROOT / "data/pchase/sm_86_static_card.json",
        trace,
    ]
    expectation = campaign.REPLAY.ReplayExpectation(
        launches=campaign.REPLAY.expected_launches_from_trace(
            trace, grid=(cell.blocks, 1, 1), block=(32, 1, 1)
        ),
        carded_dram=carded,
    )
    try:
        document = campaign.REPLAY.load_replay_pair(
            first, second, dependencies, expectation
        )
    except RuntimeError as error:
        raise RuntimeError(f"{directory}/{key}: {error}") from error
    if len(document["launches"]) != 1:
        raise RuntimeError(f"{directory}/{key}: expected one launch")
    return document["launches"][0]


def validate_integrity(root: Path) -> dict[str, object]:
    cells = configured_cells()
    correctness = 0
    records = 0
    ncu_profiles = 0
    for cell in cells:
        result = json.loads(
            (root / "correctness" / cell.key / "result.json").read_text(encoding="utf-8")
        )["measurement"]
        if result["correct"] is not True:
            raise RuntimeError(f"{cell.key}: incorrect result")
        correctness += int(result["trials"])
        observed(root, cell.key)
        ncu_profiles += len(list((root / "ncu" / cell.key).glob("repeat-*/ncu.csv")))

        trace = root / "nvbit" / cell.key
        if (trace / "dynamic_trace.pb").stat().st_size <= 0:
            raise RuntimeError(f"{cell.key}: empty trace")
        with (trace / "stats.csv").open(encoding="utf-8", newline="") as handle:
            rows = [
                {name.strip(): value.strip() for name, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        if len(rows) != 1:
            raise RuntimeError(f"{cell.key}: trace launch count mismatch")
        total = int(rows[0]["total_insts"])
        if total != int(rows[0]["total_reported_insts"]):
            raise RuntimeError(f"{cell.key}: trace cardinality mismatch")
        records += total
        extra = trace / "extra_info"
        for name in (
            "unsupported_memory_path.json",
            "unmapped_memory_range.json",
            "cross_allocation_lane_span.json",
        ):
            if json.loads((extra / name).read_text(encoding="utf-8")) != []:
                raise RuntimeError(f"{cell.key}: nonempty {name}")
        modules = json.loads((extra / "traced_kernel_modules.json").read_text(encoding="utf-8"))
        selection = modules["selection"]
        if (
            selection["accepted_active_launches"] != 1
            or selection["accepted_channel_records"] != total
            or selection["exact_contract_satisfied"] is not True
        ):
            raise RuntimeError(f"{cell.key}: tracer selection mismatch")
        for directory in ("simulator", "simulator-carded-dram"):
            launch = replay(root, directory, cell.key)
            if launch["counters"]["instructions_issued"] != launch["counters"]["records_seen"]:
                raise RuntimeError(f"{directory}/{cell.key}: issue/record mismatch")

    source_reports = sum(
        (root / "ncu-source" / cell.key / "ncu-report.ncu-rep").is_file()
        for cell in source_cells()
    )
    if source_reports != len(source_cells()):
        raise RuntimeError("SourceCounters coverage is incomplete")

    sidecar = json.loads(
        (root / "sass/memory_retirement_service_trace.control_bits.json").read_text(
            encoding="utf-8"
        )
    )
    statuses = [entry["status"] for entries in sidecar["functions"].values() for entry in entries]
    if len(sidecar["functions"]) != 12 or any(status != "decoded" for status in statuses):
        raise RuntimeError("static sidecar is incomplete")

    return {
        "matrix_cells": len(cells),
        "correctness_executions": correctness,
        "ncu_profiles": ncu_profiles,
        "source_counter_profiles": source_reports,
        "nvbit_captures": len(cells),
        "nvbit_records": records,
        "simulator_replay_pairs": len(cells),
        "carded_dram_replay_pairs": len(cells),
        "static_functions": len(sidecar["functions"]),
        "static_instructions": len(statuses),
        "static_unknown_encodings": sum(status != "decoded" for status in statuses),
    }


def build_rows(root: Path) -> list[dict[str, object]]:
    raw: dict[str, tuple[dict[str, float], dict[str, float], dict, dict]] = {}
    for cell in configured_cells():
        values, cvs = observed(root, cell.key)
        raw[cell.key] = (
            values,
            cvs,
            replay(root, "simulator", cell.key),
            replay(root, "simulator-carded-dram", cell.key),
        )

    def control_key(cell) -> str:
        return (
            f"b{cell.blocks}_control_{cell.fence}_line_stream_"
            f"d{cell.depth}_l32"
        )

    rows: list[dict[str, object]] = []
    for cell in configured_cells():
        values, cvs, launch, carded = raw[cell.key]
        control_values = raw[control_key(cell)][0]
        timing = launch["timing"]
        carded_timing = carded["timing"]
        memory_issue = timing["readiness"].get("global_memory_issue", {})
        gpc_minus_sm = values[GPC] - values[SM_ACTIVE]
        control_envelope = control_values[GPC] - control_values[SM_ACTIVE]
        l1 = launch["counters"]["l1"]
        l2 = launch["counters"]["l2"]
        rows.append(
            {
                "cell": cell.key,
                "stage": cell.stage,
                "tail": cell.tail,
                "fence": cell.fence,
                "layout": cell.layout,
                "blocks": cell.blocks,
                "depth": cell.depth,
                "active_lanes": cell.active_lanes,
                "ncu_gpc_elapsed": values[GPC],
                "ncu_sm_active": values[SM_ACTIVE],
                "ncu_gpc_max_minus_sm_active_max": gpc_minus_sm,
                "ncu_gpc_max_minus_sm_active_max_delta": gpc_minus_sm
                - control_envelope,
                "ncu_gpc_tail_delta": values[GPC] - control_values[GPC],
                "ncu_sm_active_tail_delta": values[SM_ACTIVE] - control_values[SM_ACTIVE],
                "ncu_gpc_instance_count": values[GPC_SUM] / values[GPC_AVG],
                "ncu_gpc_instance_spread": values[GPC] - values[GPC_MIN],
                "ncu_gpc_cv": cvs[GPC],
                "ncu_sm_active_cv": cvs[SM_ACTIVE],
                "ncu_inst_executed": values[INST],
                "ncu_l1tex_ld_requests": values.get(
                    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", 0.0
                ),
                "ncu_l1tex_st_requests": values.get(
                    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", 0.0
                ),
                "ncu_l1tex_red_requests": values.get(
                    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", 0.0
                ),
                "ncu_l1tex_ld_sectors": values.get(
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 0.0
                ),
                "ncu_l1tex_st_sectors": values.get(
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", 0.0
                ),
                "ncu_l1tex_red_sectors": values.get(
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum", 0.0
                ),
                "ncu_l1tex_ld_wavefronts": values.get(
                    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum", 0.0
                ),
                "ncu_l1tex_st_wavefronts": values.get(
                    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_st.sum", 0.0
                ),
                "ncu_l1tex_red_wavefronts": values.get(
                    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum", 0.0
                ),
                "ncu_stall_drain_ratio": values.get(
                    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio", 0.0
                ),
                "ncu_stall_membar_ratio": values.get(
                    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio", 0.0
                ),
                "sim_last_issue": timing["diagnostic_last_issue_cycles"],
                "sim_last_system_event": timing[
                    "last_processed_modeled_system_event_time"
                ],
                "sim_terminal_lower_bound": timing["terminal_modeled_event_time"],
                "sim_completion_status": timing["completion_frontier"]["status"],
                "sim_modeled_drain": timing["terminal_modeled_event_time"]
                - timing["diagnostic_last_issue_cycles"],
                "sim_global_load_instructions": memory_issue.get("load_instructions", 0),
                "sim_global_store_instructions": memory_issue.get("store_instructions", 0),
                "sim_global_reduction_instructions": memory_issue.get(
                    "reduction_instructions", 0
                ),
                "sim_global_load_sectors": memory_issue.get("load_sectors", 0),
                "sim_global_store_issue_sectors": memory_issue.get("store_sectors", 0),
                "sim_global_reduction_issue_sectors": memory_issue.get(
                    "reduction_sectors", 0
                ),
                "sim_global_load_wavefronts": memory_issue.get(
                    "load_ideal_wavefronts", 0
                ),
                "sim_global_store_wavefronts": memory_issue.get(
                    "store_ideal_wavefronts", 0
                ),
                "sim_global_reduction_wavefronts": memory_issue.get(
                    "reduction_ideal_wavefronts", 0
                ),
                "sim_global_memory_issue_span_max": memory_issue.get(
                    "max_issue_span_cycles", 0
                ),
                "sim_store_sectors": l1["store_sectors"],
                "sim_reduction_sectors": l1["reduction_sectors"],
                "sim_l2_hits": l2["hits"],
                "sim_l2_misses": l2["misses"],
                "sim_carded_last_issue": carded_timing["diagnostic_last_issue_cycles"],
                "sim_carded_terminal": carded_timing["terminal_modeled_event_time"],
            }
        )
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    no_fence = [row for row in rows if row["fence"] == "none" and row["tail"] != "control"]
    candidates: dict[str, object] = {}
    for tail in ("load", "store", "red"):
        tail_rows = [row for row in no_fence if row["tail"] == tail]
        training = [
            row
            for row in tail_rows
            if row["blocks"] == 1
            and row["layout"] == "line_stream"
            and row["active_lanes"] == 32
            and row["depth"] in {1, 4}
        ]
        held_out = [row for row in tail_rows if row not in training]
        constant = median(
            float(row["ncu_gpc_max_minus_sm_active_max_delta"]) for row in training
        )
        observed_values = [
            float(row["ncu_gpc_max_minus_sm_active_max_delta"]) for row in held_out
        ]
        baseline = [0.0] * len(held_out)
        fixed = [constant] * len(held_out)
        candidates[tail] = {
            "training_constant_cycles": constant,
            "training_cells": len(training),
            "held_out_cells": len(held_out),
            "zero_tail_mae_cycles": mae(baseline, observed_values),
            "fixed_tail_mae_cycles": mae(fixed, observed_values),
            "fixed_tail_max_error_cycles": maximum_error(fixed, observed_values),
            "held_out_extrema_delta_min_cycles": min(observed_values),
            "held_out_extrema_delta_max_cycles": max(observed_values),
        }

    depth: dict[str, list[dict[str, float]]] = {}
    for tail in ("load", "store", "red"):
        depth[tail] = []
        for blocks in (1, 84):
            for value in (1, 4, 16, 64):
                row = next(
                    row
                    for row in rows
                    if row["tail"] == tail
                    and row["fence"] == "none"
                    and row["layout"] == "line_stream"
                    and row["blocks"] == blocks
                    and row["depth"] == value
                    and row["active_lanes"] == 32
                )
                depth[tail].append(
                    {
                        "blocks": blocks,
                        "depth": value,
                        "gpc_max_minus_sm_active_max_delta": float(
                            row["ncu_gpc_max_minus_sm_active_max_delta"]
                        ),
                        "gpc_tail_delta": float(row["ncu_gpc_tail_delta"]),
                        "sm_active_tail_delta": float(row["ncu_sm_active_tail_delta"]),
                    }
                )

    fences: dict[str, list[dict[str, float | str]]] = {}
    for tail in ("load", "store", "red"):
        fences[tail] = []
        for blocks in (1, 84):
            no_fence_row = next(
                row
                for row in rows
                if row["tail"] == tail
                and row["blocks"] == blocks
                and row["fence"] == "none"
                and row["layout"] == "line_stream"
                and row["depth"] == 16
                and row["active_lanes"] == 32
            )
            for fence in ("device", "system"):
                row = next(
                    row
                    for row in rows
                    if row["tail"] == tail
                    and row["blocks"] == blocks
                    and row["fence"] == fence
                )
                fences[tail].append(
                    {
                        "blocks": blocks,
                        "fence": fence,
                        "gpc_change_vs_none": float(row["ncu_gpc_elapsed"])
                        - float(no_fence_row["ncu_gpc_elapsed"]),
                        "sm_active_change_vs_none": float(row["ncu_sm_active"])
                        - float(no_fence_row["ncu_sm_active"]),
                        "gpc_max_minus_sm_active_max_delta": float(
                            row["ncu_gpc_max_minus_sm_active_max_delta"]
                        ),
                    }
                )

    topology_mismatches = []
    wavefront_replays = []
    system_fence_hidden_store_work = []
    for row in rows:
        if row["tail"] == "load":
            triples = (
                ("instructions", row["sim_global_load_instructions"], row["ncu_l1tex_ld_requests"]),
                ("sectors", row["sim_global_load_sectors"], row["ncu_l1tex_ld_sectors"]),
                ("wavefronts", row["sim_global_load_wavefronts"], row["ncu_l1tex_ld_wavefronts"]),
            )
        elif row["tail"] == "store":
            triples = (
                ("instructions", row["sim_global_store_instructions"], row["ncu_l1tex_st_requests"]),
                ("sectors", row["sim_global_store_issue_sectors"], row["ncu_l1tex_st_sectors"]),
                ("wavefronts", row["sim_global_store_wavefronts"], row["ncu_l1tex_st_wavefronts"]),
            )
        elif row["tail"] == "red":
            triples = (
                ("instructions", row["sim_global_reduction_instructions"], row["ncu_l1tex_red_requests"]),
                ("sectors", row["sim_global_reduction_issue_sectors"], row["ncu_l1tex_red_sectors"]),
                ("wavefronts", row["sim_global_reduction_wavefronts"], row["ncu_l1tex_red_wavefronts"]),
            )
        else:
            continue
        for name, simulated, ncu in triples:
            simulated_value = float(simulated)
            ncu_value = float(ncu)
            if row["tail"] == "store" and row["fence"] == "system":
                if name == "instructions":
                    system_fence_hidden_store_work.append(
                        {
                            "cell": row["cell"],
                            "extra_requests": ncu_value - simulated_value,
                        }
                    )
                continue
            if name == "wavefronts":
                if ncu_value < simulated_value:
                    topology_mismatches.append(
                        {
                            "cell": row["cell"],
                            "field": "ideal_wavefront_floor",
                            "sim": simulated,
                            "ncu": ncu,
                        }
                    )
                elif ncu_value > simulated_value:
                    wavefront_replays.append(
                        {
                            "cell": row["cell"],
                            "ideal_wavefronts": simulated,
                            "ncu_output_wavefronts": ncu,
                            "extra_replay_wavefronts": ncu_value - simulated_value,
                        }
                    )
            elif simulated_value != ncu_value:
                topology_mismatches.append(
                    {"cell": row["cell"], "field": name, "sim": simulated, "ncu": ncu}
                )

    return {
        "gpc_instances_median": median(float(row["ncu_gpc_instance_count"]) for row in rows),
        "gpc_instance_spread_max_cycles": max(
            float(row["ncu_gpc_instance_spread"]) for row in rows
        ),
        "ncu_gpc_cv_max_pct": 100.0 * max(float(row["ncu_gpc_cv"]) for row in rows),
        "ncu_sm_active_cv_max_pct": 100.0
        * max(float(row["ncu_sm_active_cv"]) for row in rows),
        "candidate_transfer": candidates,
        "depth": depth,
        "fences": fences,
        "carded_dram_changed_any_timing": any(
            row["sim_last_issue"] != row["sim_carded_last_issue"]
            or row["sim_terminal_lower_bound"] != row["sim_carded_terminal"]
            for row in rows
        ),
        "global_memory_topology_mismatches": topology_mismatches,
        "output_wavefront_replay_cells": wavefront_replays,
        "system_fence_hidden_store_work": system_fence_hidden_store_work,
    }


def markdown(integrity: Mapping[str, object], summary: Mapping[str, object]) -> str:
    replay_cell = summary["output_wavefront_replay_cells"][0]
    hidden_store_work = sorted(
        summary["system_fence_hidden_store_work"],
        key=lambda item: float(item["extra_requests"]),
    )
    candidates = summary["candidate_transfer"]
    lines = [
        "# sm_86 memory-retirement service campaign",
        "",
        "## Outcome",
        "",
        "The campaign promotes trace-derived scalar-global instruction, sector,",
        "ideal-wavefront, and issue-span diagnostics as readiness schema 6. It",
        "rejects fixed load/store/RED retirement delays and does not activate a",
        "memory service: deep B84 line streams expose a nonlinear, multimodal",
        "backlog that depends on issue overlap, topology, internal replay, and",
        "atomic many-region scheduling.",
        "",
        "## Coverage",
        "",
        f"- {integrity['matrix_cells']} cells and {integrity['correctness_executions']} correct executions.",
        f"- {integrity['ncu_profiles']} NCU profiles and {integrity['source_counter_profiles']} SourceCounters profiles.",
        f"- {integrity['nvbit_captures']} NVBit captures with {integrity['nvbit_records']:,} records.",
        f"- {integrity['simulator_replay_pairs']} ordinary and {integrity['carded_dram_replay_pairs']} carded deterministic replay pairs.",
        f"- {integrity['static_functions']} functions and {integrity['static_instructions']} decoded instructions; zero unknown encodings.",
        f"- Maximum GPC/SM-active CV after targeted expansion to seven repeats is {summary['ncu_gpc_cv_max_pct']:.3f}% / {summary['ncu_sm_active_cv_max_pct']:.3f}%; the retained saturation modes remain multimodal.",
        "- NCU application replay may collect GPC and SM-active groups on different",
        "  executions; their matched extrema difference is not co-temporal.",
        "",
        "## Trace-derived work boundary",
        "",
        f"Issued instruction and sector topology has {len(summary['global_memory_topology_mismatches'])} mismatches across every load/store/RED cell.",
        "Ordinary sectors pack four per ideal wavefront; RED additionally preserves",
        "same-sector alias multiplicity. NCU output equals this ideal everywhere",
        f"except {replay_cell['cell']}, which emits {replay_cell['extra_replay_wavefronts']:,.0f} extra replay",
        "wavefronts. That internal replay is a measured service effect, not something",
        "lane addresses can predict. `MEMBAR.SC.SYS` also produces one hidden store",
        f"request per CTA through its compiler epilogue ({hidden_store_work[0]['extra_requests']:,.0f} at B1, {hidden_store_work[-1]['extra_requests']:,.0f} at B84), so",
        "system-fence traffic cannot be reconstructed as only the source STG stream.",
        "",
        "## Fixed-tail transfer candidates",
        "",
        "Constants are trained only on B1, depths 1/4, full-lane line streams;",
        "all CTA, deep, lane, layout, alias, and reuse cells are held out.",
        "",
        "| tail | trained cycles | held-out cells | zero-tail MAE | fixed-tail MAE | max error | held-out range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tail in ("load", "store", "red"):
        item = summary["candidate_transfer"][tail]
        lines.append(
            f"| {tail} | {item['training_constant_cycles']:.1f} | {item['held_out_cells']} | "
            f"{item['zero_tail_mae_cycles']:.1f} | {item['fixed_tail_mae_cycles']:.1f} | "
            f"{item['fixed_tail_max_error_cycles']:.1f} | "
            f"{item['held_out_extrema_delta_min_cycles']:.1f}…{item['held_out_extrema_delta_max_cycles']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Depth transfer",
            "",
            "The extrema-difference column subtracts independently reduced GPC and",
            "SM maxima. It is a useful matched response, but not a temporal interval.",
            "",
            "| tail | CTAs | depth | GPC-max minus SM-active-max delta | live GPC tail delta | live SM-active delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tail in ("load", "store", "red"):
        for item in summary["depth"][tail]:
            lines.append(
                f"| {tail} | {item['blocks']} | {item['depth']} | "
                f"{item['gpc_max_minus_sm_active_max_delta']:.1f} | {item['gpc_tail_delta']:.1f} | "
                f"{item['sm_active_tail_delta']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Explicit fence discriminator",
            "",
            "| tail | CTAs | fence | GPC change vs none | SM-active change vs none | extrema-difference delta |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for tail in ("load", "store", "red"):
        for item in summary["fences"][tail]:
            lines.append(
                f"| {tail} | {item['blocks']} | {item['fence']} | "
                f"{item['gpc_change_vs_none']:.1f} | {item['sm_active_change_vs_none']:.1f} | "
                f"{item['gpc_max_minus_sm_active_max_delta']:.1f} |"
            )
    lines.extend(
        [
            "",
            "Full per-cell topology, traffic, timing, and completion-status values are",
            "in `cells.csv`; machine-readable reductions are in `summary.json`.",
            "",
            "## Decision",
            "",
            "- Keep load-only terminal response events active and typed complete, but",
            "  do not add a fixed extrema correction: the held-out matched difference",
            f"  ranges from {candidates['load']['held_out_extrema_delta_min_cycles']:,.0f} to {candidates['load']['held_out_extrema_delta_max_cycles']:,.0f} cycles and is not a temporal drain measurement.",
            "- Keep store and RED retirement incomplete. Their B1-trained constants",
            "  are worse than zero-tail baselines on held-out topology, with maximum",
            f"  errors of {candidates['store']['fixed_tail_max_error_cycles']:,.1f} and {candidates['red']['fixed_tail_max_error_cycles']:,.1f} cycles.",
            "- Promote only readiness schema 6 work/arrival diagnostics. They expose",
            "  the exact ideal work needed by a future service without changing issue",
            "  or terminal time.",
            "- Treat the B84 depth-64 extra store wavefronts and B84 deep RED phase as",
            "  separate internal-replay/many-region scheduling gaps. Neither is",
            "  repaired with a campaign-specific tail penalty.",
            "- Explicit fences change both GPC elapsed and SM active work, which is",
            "  consistent with overlap moving into active execution. The extrema",
            "  difference cannot locate that work on a post-active timeline, and",
            "  fences remain non-interchangeable with the task-boundary flush.",
            "",
            "No privileged command was required.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.output_root.resolve()
    integrity = validate_integrity(root)
    rows = build_rows(root)
    summary = summarize(rows)
    output = root / "analysis"
    write_csv(output / "cells.csv", rows)
    write_json(output / "summary.json", {"integrity": integrity, "results": summary})
    (output / "README.md").write_text(markdown(integrity, summary), encoding="utf-8")
    print(output / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
