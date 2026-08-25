#!/usr/bin/env python3
"""Reduce the sm_86 L1TEX arrival/output-replay service campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from functools import cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import campaign
from campaign import (
    CANDIDATE_FILE,
    DEFAULT_OUTPUT,
    configured_cells,
    l2_request_cells,
    source_cells,
)


GPC = "gpc__cycles_elapsed.max"
GPC_AVG = "gpc__cycles_elapsed.avg"
GPC_MIN = "gpc__cycles_elapsed.min"
GPC_SUM = "gpc__cycles_elapsed.sum"
SM_ACTIVE = "sm__cycles_active.max"


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
        raise RuntimeError(f"{path}: expected one NCU row")
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


def observation(root: Path, key: str) -> tuple[dict[str, float], dict[str, float], int]:
    paths = sorted((root / "ncu" / key).glob("repeat-*/ncu.csv"))
    if len(paths) not in {3, 5, 7}:
        raise RuntimeError(f"{key}: expected 3/5/7 NCU rows, got {len(paths)}")
    rows = [ncu_row(path) for path in paths]
    keys = set.intersection(*(set(row) for row in rows))
    return (
        {key: median(row[key] for row in rows) for key in keys},
        {key: cv([row[key] for row in rows]) for key in keys},
        len(rows),
    )


@cache
def replay(root: Path, directory: str, key: str) -> dict:
    cells = {cell.key: cell for cell in configured_cells()}
    try:
        cell = cells[key]
    except KeyError as error:
        raise RuntimeError(f"unknown replay cell: {key}") from error
    first = root / directory / key / "replay-1.json"
    second = root / directory / key / "replay-2.json"
    trace = root / "nvbit" / key
    dependencies = [
        campaign.REPO_ROOT / "sim/target/release/sim-run",
        root / "sass/l1tex_arrival_replay_service_trace.control_bits.json",
        campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json",
        campaign.REPO_ROOT / "data/pchase/sm_86_static_card.json",
        trace,
    ]
    expectation_kwargs: dict[str, object] = {}
    if directory == "simulator":
        carded = False
    elif directory == "simulator-carded-dram":
        carded = True
    elif directory == "simulator-memory-service-candidates":
        carded = False
        expectation_kwargs["service_candidates"] = (
            campaign.REPLAY.candidate_configurations(CANDIDATE_FILE)
        )
        dependencies.extend(
            [CANDIDATE_FILE, root / "metadata/memory_service_candidates.json"]
        )
    else:
        raise RuntimeError(f"unknown replay directory: {directory}")
    expectation = campaign.REPLAY.ReplayExpectation(
        launches=campaign.REPLAY.expected_launches_from_trace(
            trace,
            grid=(cell.blocks, 1, 1),
            block=(cell.warps * 32, 1, 1),
        ),
        carded_dram=carded,
        **expectation_kwargs,
    )
    try:
        document = campaign.REPLAY.load_replay_pair(
            first, second, dependencies, expectation
        )
    except RuntimeError as error:
        raise RuntimeError(f"{directory}/{key}: {error}") from error
    launches = document["launches"]
    if len(launches) != 1:
        raise RuntimeError(f"{directory}/{key}: expected one launch")
    return launches[0]


def l2_observation(root: Path, key: str) -> dict[str, float]:
    path = root / "ncu-l2-requests" / key / "ncu.csv"
    return ncu_row(path)


def validate_integrity(root: Path) -> dict[str, object]:
    correctness = 0
    ncu_profiles = 0
    records = 0
    for cell in configured_cells():
        result = json.loads(
            (root / "correctness" / cell.key / "result.json").read_text(encoding="utf-8")
        )["measurement"]
        if result["correct"] is not True or result["ideal_wavefronts"] != cell.ideal_wavefronts:
            raise RuntimeError(f"{cell.key}: correctness/work mismatch")
        correctness += int(result["trials"])
        _, _, repeats = observation(root, cell.key)
        ncu_profiles += repeats

        trace = root / "nvbit" / cell.key
        if (trace / "dynamic_trace.pb").stat().st_size <= 0:
            raise RuntimeError(f"{cell.key}: empty trace")
        with (trace / "stats.csv").open(encoding="utf-8", newline="") as handle:
            stats = [
                {name.strip(): value.strip() for name, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        if len(stats) != 1:
            raise RuntimeError(f"{cell.key}: trace launch mismatch")
        total = int(stats[0]["total_insts"])
        if total != int(stats[0]["total_reported_insts"]):
            raise RuntimeError(f"{cell.key}: trace record mismatch")
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
        candidate_launch = replay(root, "simulator-memory-service-candidates", cell.key)
        candidate_readiness = candidate_launch["timing"]["readiness"]
        if candidate_readiness["schema_version"] not in {7, 8}:
            raise RuntimeError(f"{cell.key}: candidate replay did not emit readiness schema 7/8")
        candidates = candidate_readiness.get("memory_service_candidates", [])
        if len(candidates) != 16:
            raise RuntimeError(f"{cell.key}: expected 16 candidate results, got {len(candidates)}")
        ordinary_launch = replay(root, "simulator", cell.key)
        stripped_candidate = copy.deepcopy(candidate_launch)
        stripped_readiness = stripped_candidate["timing"]["readiness"]
        stripped_readiness.pop("memory_service_candidates")
        stripped_readiness["schema_version"] = ordinary_launch["timing"]["readiness"][
            "schema_version"
        ]
        if stripped_candidate != ordinary_launch:
            raise RuntimeError(f"{cell.key}: observational candidates changed replay output")

    source_count = sum(
        (root / "ncu-source" / cell.key / "ncu-report.ncu-rep").is_file()
        for cell in source_cells()
    )
    if source_count != len(source_cells()):
        raise RuntimeError("SourceCounters coverage incomplete")
    l2_request_count = sum(
        (root / "ncu-l2-requests" / cell.key / "ncu-report.ncu-rep").is_file()
        for cell in l2_request_cells()
    )
    if l2_request_count != len(l2_request_cells()):
        raise RuntimeError("focused L2-request coverage incomplete")

    sidecar = json.loads(
        (root / "sass/l1tex_arrival_replay_service_trace.control_bits.json").read_text(
            encoding="utf-8"
        )
    )
    statuses = [entry["status"] for entries in sidecar["functions"].values() for entry in entries]
    if len(sidecar["functions"]) != 4 or any(status != "decoded" for status in statuses):
        raise RuntimeError("static sidecar incomplete")

    return {
        "matrix_cells": len(configured_cells()),
        "correctness_executions": correctness,
        "ncu_profiles": ncu_profiles,
        "source_counter_profiles": source_count,
        "l2_request_profiles": l2_request_count,
        "nvbit_captures": len(configured_cells()),
        "nvbit_records": records,
        "simulator_replay_pairs": len(configured_cells()),
        "carded_dram_replay_pairs": len(configured_cells()),
        "memory_service_candidate_replay_pairs": len(configured_cells()),
        "candidate_replays_with_base_output_changes": 0,
        "static_functions": len(sidecar["functions"]),
        "static_instructions": len(statuses),
        "static_unknown_encodings": sum(status != "decoded" for status in statuses),
    }


def build_rows(root: Path) -> list[dict[str, object]]:
    raw: dict[str, tuple[dict[str, float], dict[str, float], dict, dict, dict]] = {}
    for cell in configured_cells():
        values, cvs, _ = observation(root, cell.key)
        raw[cell.key] = (
            values,
            cvs,
            replay(root, "simulator", cell.key),
            replay(root, "simulator-carded-dram", cell.key),
            replay(root, "simulator-memory-service-candidates", cell.key),
        )

    def control_key(cell) -> str:
        return (
            f"b{cell.blocks}_w{cell.warps}_control_{cell.layout}_"
            f"d{cell.depth}_l{cell.active_lanes}_g{cell.gap_steps}"
        )

    rows: list[dict[str, object]] = []
    for cell in configured_cells():
        values, cvs, launch, carded, candidate_launch = raw[cell.key]
        control_values, _, control_launch, _, control_candidate_launch = raw[control_key(cell)]
        timing = launch["timing"]
        control_timing = control_launch["timing"]
        memory = timing["readiness"].get("global_memory_issue", {})
        if cell.tail == "load":
            request_metric = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
            sector_metric = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
            wave_metric = "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum"
            conflict_metric = "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum"
            sim_instructions = memory.get("load_instructions", 0)
            sim_sectors = memory.get("load_sectors", 0)
            sim_ideal = memory.get("load_ideal_wavefronts", 0)
            sim_unique_lines = memory.get("load_unique_lines", 0)
            sim_atomic_requests = 0
        elif cell.tail == "store":
            request_metric = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
            sector_metric = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
            wave_metric = "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_st.sum"
            conflict_metric = "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_st.sum"
            sim_instructions = memory.get("store_instructions", 0)
            sim_sectors = memory.get("store_sectors", 0)
            sim_ideal = memory.get("store_ideal_wavefronts", 0)
            sim_unique_lines = memory.get("store_unique_lines", 0)
            sim_atomic_requests = 0
        elif cell.tail == "red":
            request_metric = "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"
            sector_metric = "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum"
            wave_metric = "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum"
            conflict_metric = "l1tex__t_set_conflicts_pipe_lsu_mem_global_op_red.sum"
            sim_instructions = memory.get("reduction_instructions", 0)
            sim_sectors = memory.get("reduction_sectors", 0)
            sim_ideal = memory.get("reduction_ideal_wavefronts", 0)
            sim_unique_lines = memory.get("reduction_unique_lines", 0)
            sim_atomic_requests = memory.get("reduction_l2_requests", 0)
        else:
            request_metric = sector_metric = wave_metric = conflict_metric = ""
            sim_instructions = sim_sectors = sim_ideal = sim_unique_lines = 0
            sim_atomic_requests = 0

        output_wavefronts = values.get(wave_metric, 0.0) if wave_metric else 0.0
        control_envelope = control_values[GPC] - control_values[SM_ACTIVE]
        row: dict[str, object] = {
                "cell": cell.key,
                "stage": cell.stage,
                "tail": cell.tail,
                "layout": cell.layout,
                "blocks": cell.blocks,
                "warps": cell.warps,
                "depth": cell.depth,
                "active_lanes": cell.active_lanes,
                "gap_steps": cell.gap_steps,
                "ideal_wavefronts_configured": cell.ideal_wavefronts,
                "ideal_wavefronts_per_nominal_sm": cell.ideal_wavefronts_per_nominal_sm,
                "ncu_gpc_elapsed": values[GPC],
                "ncu_sm_active": values[SM_ACTIVE],
                "ncu_gpc_tail_delta": values[GPC] - control_values[GPC],
                "ncu_sm_active_tail_delta": values[SM_ACTIVE] - control_values[SM_ACTIVE],
                "ncu_gpc_max_minus_sm_active_max_delta": (values[GPC] - values[SM_ACTIVE])
                - control_envelope,
                "ncu_requests": values.get(request_metric, 0.0) if request_metric else 0.0,
                "ncu_sectors": values.get(sector_metric, 0.0) if sector_metric else 0.0,
                "ncu_output_wavefronts": output_wavefronts,
                "ncu_extra_replay_wavefronts": output_wavefronts - float(sim_ideal),
                "ncu_set_conflicts": values.get(conflict_metric, 0.0) if conflict_metric else 0.0,
                "ncu_lsuin_requests": values.get("l1tex__lsuin_requests.sum", 0.0),
                "ncu_gpc_instances": values[GPC_SUM] / values[GPC_AVG],
                "ncu_gpc_instance_spread": values[GPC] - values[GPC_MIN],
                "ncu_gpc_cv": cvs[GPC],
                "ncu_sm_active_cv": cvs[SM_ACTIVE],
                "sim_instructions": sim_instructions,
                "sim_sectors": sim_sectors,
                "sim_ideal_wavefronts": sim_ideal,
                "sim_unique_lines": sim_unique_lines,
                "sim_unique_lines_per_nominal_sm": float(sim_unique_lines) / 84.0,
                "sim_atomic_l2_requests": sim_atomic_requests,
                "sim_issue_span_max": memory.get("max_issue_span_cycles", 0),
                "sim_last_issue": timing["diagnostic_last_issue_cycles"],
                "sim_terminal": timing["terminal_modeled_event_time"],
                "sim_issue_tail_delta": timing["diagnostic_last_issue_cycles"]
                - control_timing["diagnostic_last_issue_cycles"],
                "sim_terminal_tail_delta": timing["terminal_modeled_event_time"]
                - control_timing["terminal_modeled_event_time"],
                "sim_completion_status": timing["completion_frontier"]["status"],
                "sim_carded_last_issue": carded["timing"]["diagnostic_last_issue_cycles"],
                "sim_carded_terminal": carded["timing"]["terminal_modeled_event_time"],
            }
        candidate_timing = candidate_launch["timing"]
        control_candidate_timing = control_candidate_launch["timing"]
        candidate_results = {
            int(item["candidate"]["candidate_id"]): item
            for item in candidate_timing["readiness"]["memory_service_candidates"]
        }
        control_results = {
            int(item["candidate"]["candidate_id"]): item
            for item in control_candidate_timing["readiness"]["memory_service_candidates"]
        }
        if candidate_results.keys() != control_results.keys():
            raise RuntimeError(f"{cell.key}: candidate/control configuration mismatch")
        tail_terminal = int(candidate_timing["terminal_modeled_event_time"] or 0)
        control_terminal = int(control_candidate_timing["terminal_modeled_event_time"] or 0)
        for candidate_id in sorted(candidate_results):
            item = candidate_results[candidate_id]
            control_item = control_results[candidate_id]
            if item["candidate"] != control_item["candidate"]:
                raise RuntimeError(f"{cell.key}: candidate {candidate_id} configuration mismatch")
            prefix = f"candidate_{candidate_id}"
            terminal = int(item["terminal_completion_time"])
            control_candidate_terminal = int(control_item["terminal_completion_time"])
            row[f"{prefix}_terminal"] = terminal
            row[f"{prefix}_busy_cycles"] = int(item["busy_cycles"])
            row[f"{prefix}_queue_delay_cycles"] = int(item["queue_delay_cycles"])
            row[f"{prefix}_predicted_tail_delta"] = max(tail_terminal, terminal) - max(
                control_terminal, control_candidate_terminal
            )
            row[f"{prefix}_configuration"] = json.dumps(
                item["candidate"], sort_keys=True, separators=(",", ":")
            )
        rows.append(row)
    return rows


def summarize(root: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    topology_mismatches = []
    for row in rows:
        if row["tail"] == "control":
            continue
        for field, sim_field, ncu_field in (
            ("instructions", "sim_instructions", "ncu_requests"),
            ("sectors", "sim_sectors", "ncu_sectors"),
        ):
            if float(row[sim_field]) != float(row[ncu_field]):
                topology_mismatches.append(
                    {
                        "cell": row["cell"],
                        "field": field,
                        "sim": row[sim_field],
                        "ncu": row[ncu_field],
                    }
                )
        if float(row["ncu_output_wavefronts"]) < float(row["sim_ideal_wavefronts"]):
            topology_mismatches.append(
                {
                    "cell": row["cell"],
                    "field": "ideal_wavefront_floor",
                    "sim": row["sim_ideal_wavefronts"],
                    "ncu": row["ncu_output_wavefronts"],
                }
            )

    l2_request_mismatches = []
    for cell in l2_request_cells():
        row = next(row for row in rows if row["cell"] == cell.key)
        ncu = l2_observation(root, cell.key)
        if cell.tail == "load":
            observed_requests = ncu["lts__t_requests_srcunit_tex_op_read.sum"]
            simulated_requests = float(row["sim_unique_lines"])
        elif cell.tail == "store":
            observed_requests = ncu["lts__t_requests_srcunit_tex_op_write.sum"]
            simulated_requests = float(row["sim_unique_lines"])
        else:
            observed_requests = ncu["lts__t_requests_srcunit_tex_op_red.sum"]
            simulated_requests = float(row["sim_atomic_l2_requests"])
        if observed_requests != simulated_requests:
            l2_request_mismatches.append(
                {
                    "cell": cell.key,
                    "sim": simulated_requests,
                    "ncu": observed_requests,
                }
            )

    def selected(**criteria) -> list[Mapping[str, object]]:
        return [
            row
            for row in rows
            if all(row[name] == value for name, value in criteria.items())
        ]

    fixed_concurrency: dict[str, list[dict[str, float]]] = {}
    for tail in ("load", "store", "red"):
        fixed_concurrency[tail] = []
        for warps in (1, 2, 4, 8):
            row = selected(stage="fixed_ideal_concurrency", tail=tail, warps=warps)[0]
            fixed_concurrency[tail].append(
                {
                    "warps": warps,
                    "gpc_tail_delta": float(row["ncu_gpc_tail_delta"]),
                    "sm_active_tail_delta": float(row["ncu_sm_active_tail_delta"]),
                    "gpc_minus_sm_active_extrema_delta": float(
                        row["ncu_gpc_max_minus_sm_active_max_delta"]
                    ),
                    "extra_replay_wavefronts": float(row["ncu_extra_replay_wavefronts"]),
                    "sim_issue_span": float(row["sim_issue_span_max"]),
                }
            )

    cta_shape: dict[str, list[dict[str, float]]] = {}
    for tail in ("load", "store", "red"):
        cta_shape[tail] = []
        shape_keys = ((84, 8), (168, 4), (336, 2))
        for blocks, warps in shape_keys:
            row = next(
                row
                for row in rows
                if row["tail"] == tail
                and row["blocks"] == blocks
                and row["warps"] == warps
                and row["depth"] == 8
                and row["gap_steps"] == 0
                and row["layout"] == "line"
                and row["active_lanes"] == 32
            )
            cta_shape[tail].append(
                {
                    "blocks": blocks,
                    "warps": warps,
                    "ctas_per_nominal_sm": blocks // 84,
                    "gpc_tail_delta": float(row["ncu_gpc_tail_delta"]),
                    "extra_replay_wavefronts": float(row["ncu_extra_replay_wavefronts"]),
                    "sim_issue_span": float(row["sim_issue_span_max"]),
                }
            )

    skew: dict[str, list[dict[str, float]]] = {}
    for tail in ("load", "store"):
        skew[tail] = []
        for warps, depth, gaps in ((8, 8, (0, 16, 64, 256)), (4, 16, (0, 64, 256))):
            for gap in gaps:
                row = next(
                    row
                    for row in rows
                    if row["tail"] == tail
                    and row["blocks"] == 84
                    and row["warps"] == warps
                    and row["depth"] == depth
                    and row["gap_steps"] == gap
                    and row["layout"] == "line"
                    and row["active_lanes"] == 32
                )
                skew[tail].append(
                    {
                        "warps": warps,
                        "gap": gap,
                        "gpc_tail_delta": float(row["ncu_gpc_tail_delta"]),
                        "extra_replay_wavefronts": float(row["ncu_extra_replay_wavefronts"]),
                        "sim_issue_span": float(row["sim_issue_span_max"]),
                    }
                )

    topology: dict[str, list[dict[str, float | str]]] = {}
    for tail in ("load", "store"):
        topology[tail] = []
        candidates = [
            row
            for row in rows
            if row["tail"] == tail
            and row["blocks"] == 84
            and row["warps"] == 8
            and row["gap_steps"] == 0
            and row["ideal_wavefronts_per_nominal_sm"] == 512
        ]
        for row in candidates:
            topology[tail].append(
                {
                    "layout": str(row["layout"]),
                    "lanes": int(row["active_lanes"]),
                    "depth": int(row["depth"]),
                    "gpc_tail_delta": float(row["ncu_gpc_tail_delta"]),
                    "extra_replay_wavefronts": float(row["ncu_extra_replay_wavefronts"]),
                    "sim_issue_span": float(row["sim_issue_span_max"]),
                    "ideal_wavefronts_per_nominal_sm": float(
                        row["ideal_wavefronts_per_nominal_sm"]
                    ),
                    "unique_lines_per_nominal_sm": float(
                        row["sim_unique_lines_per_nominal_sm"]
                    ),
                }
            )

    def solve_three(matrix: list[list[float]], rhs: list[float]) -> list[float]:
        augmented = [row[:] + [value] for row, value in zip(matrix, rhs, strict=True)]
        for column in range(3):
            pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot][column]) < 1e-12:
                raise RuntimeError("singular work-surface candidate")
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            scale = augmented[column][column]
            augmented[column] = [value / scale for value in augmented[column]]
            for row in range(3):
                if row == column:
                    continue
                factor = augmented[row][column]
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
        return [augmented[index][3] for index in range(3)]

    work_surface_candidates: dict[str, object] = {}
    for tail in ("load", "store"):
        training = [
            row
            for row in rows
            if row["tail"] == tail
            and (
                row["stage"] == "w4_work_scaling"
                or row["stage"] == "fixed_ideal_word_topology"
            )
            and row["gap_steps"] == 0
        ]
        design = [
            [
                1.0,
                float(row["ideal_wavefronts_per_nominal_sm"]),
                float(row["sim_unique_lines_per_nominal_sm"]),
            ]
            for row in training
        ]
        target = [float(row["ncu_gpc_tail_delta"]) for row in training]
        normal = [
            [sum(vector[i] * vector[j] for vector in design) for j in range(3)]
            for i in range(3)
        ]
        rhs = [sum(vector[i] * value for vector, value in zip(design, target, strict=True)) for i in range(3)]
        coefficients = solve_three(normal, rhs)

        def predict(row: Mapping[str, object]) -> float:
            return (
                coefficients[0]
                + coefficients[1] * float(row["ideal_wavefronts_per_nominal_sm"])
                + coefficients[2] * float(row["sim_unique_lines_per_nominal_sm"])
            )

        held_out = [
            row for row in rows if row["tail"] == tail and row not in training
        ]
        work_surface_candidates[tail] = {
            "coefficients_intercept_ideal_wavefront_unique_line": coefficients,
            "training_cells": len(training),
            "held_out_cells": len(held_out),
            "training_mae": mae(
                [predict(row) for row in training],
                [float(row["ncu_gpc_tail_delta"]) for row in training],
            ),
            "held_out_mae": mae(
                [predict(row) for row in held_out],
                [float(row["ncu_gpc_tail_delta"]) for row in held_out],
            ),
            "held_out_max_error": max(
                abs(predict(row) - float(row["ncu_gpc_tail_delta"]))
                for row in held_out
            ),
            "held_out_by_stage": {
                stage: {
                    "cells": len(stage_rows),
                    "mae": mae(
                        [predict(row) for row in stage_rows],
                        [float(row["ncu_gpc_tail_delta"]) for row in stage_rows],
                    ),
                    "max_error": max(
                        abs(predict(row) - float(row["ncu_gpc_tail_delta"]))
                        for row in stage_rows
                    ),
                }
                for stage in sorted({str(row["stage"]) for row in held_out})
                if (
                    stage_rows := [
                        row for row in held_out if row["stage"] == stage
                    ]
                )
            },
        }

    candidate_document = json.loads(
        (root / "metadata/memory_service_candidates.json").read_text(encoding="utf-8")
    )
    if candidate_document.get("schema_version") != 1:
        raise RuntimeError("memory-service candidate document schema mismatch")
    candidate_configs = candidate_document["candidates"]
    candidate_scores: dict[str, dict[str, object]] = {"store": {}, "red": {}}
    operation_tail = {"store": "store", "reduction": "red"}
    calibration_stages = {
        "store": {"w4_work_scaling", "w1_work_scaling"},
        "red": {"w4_work_scaling"},
    }

    def score_candidate(
        candidate_id: int, material: Sequence[Mapping[str, object]]
    ) -> dict[str, float | int]:
        predicted = [
            float(row[f"candidate_{candidate_id}_predicted_tail_delta"])
            for row in material
        ]
        observed = [float(row["ncu_gpc_tail_delta"]) for row in material]
        errors = [value - target for value, target in zip(predicted, observed, strict=True)]
        return {
            "cells": len(material),
            "mae": mean(abs(error) for error in errors),
            "max_error": max(abs(error) for error in errors),
            "mean_signed_error": mean(errors),
        }

    for config in candidate_configs:
        operation = config["operation"]
        if operation not in operation_tail:
            continue
        tail = operation_tail[operation]
        candidate_id = int(config["candidate_id"])
        field = f"candidate_{candidate_id}_configuration"
        if any(json.loads(str(row[field])) != config for row in rows):
            raise RuntimeError(
                f"candidate {candidate_id} replay configuration differs from retained snapshot"
            )
        operation_rows = [row for row in rows if row["tail"] == tail]
        calibration = [
            row for row in operation_rows if row["stage"] in calibration_stages[tail]
        ]
        held_out = [row for row in operation_rows if row not in calibration]
        candidate_scores[tail][str(candidate_id)] = {
            "configuration": config,
            "calibration": score_candidate(candidate_id, calibration),
            "held_out": score_candidate(candidate_id, held_out),
            "all": score_candidate(candidate_id, operation_rows),
            "held_out_by_stage": {
                stage: score_candidate(
                    candidate_id,
                    [row for row in held_out if row["stage"] == stage],
                )
                for stage in sorted({str(row["stage"]) for row in held_out})
            },
        }

    selected_candidates: dict[str, object] = {}
    frozen_ids = {"store": "2", "red": "10"}
    for tail in ("store", "red"):
        best_mae = min(
            float(item["calibration"]["mae"])
            for item in candidate_scores[tail].values()
        )
        tied_ids = sorted(
            int(candidate_id)
            for candidate_id, item in candidate_scores[tail].items()
            if math.isclose(
                float(item["calibration"]["mae"]),
                best_mae,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        selected_id = frozen_ids[tail]
        selected = candidate_scores[tail][selected_id]
        if int(selected_id) not in tied_ids:
            raise RuntimeError(f"frozen {tail} candidate is not calibration-optimal")
        operation_rows = [row for row in rows if row["tail"] == tail]
        held_out = [
            row
            for row in operation_rows
            if row["stage"] not in calibration_stages[tail]
        ]
        selected_candidates[tail] = {
            "candidate_id": int(selected_id),
            "calibration_tied_candidate_ids": tied_ids,
            **selected,
            "baseline_sim_terminal_held_out": {
                "cells": len(held_out),
                "mae": mae(
                    [float(row["sim_terminal_tail_delta"]) for row in held_out],
                    [float(row["ncu_gpc_tail_delta"]) for row in held_out],
                ),
                "max_error": max(
                    abs(
                        float(row["sim_terminal_tail_delta"])
                        - float(row["ncu_gpc_tail_delta"])
                    )
                    for row in held_out
                ),
            },
        }

    scope_pairs = (
        ("store_line_equal_per_server", 2, 6),
        ("store_line_equal_full_sm_aggregate", 2, 7),
        ("store_ideal_equal_per_server", 4, 8),
        ("red_line_equal_per_server", 10, 14),
        ("red_line_equal_full_sm_aggregate", 10, 15),
        ("red_ideal_equal_per_server", 12, 16),
    )
    config_by_id = {int(config["candidate_id"]): config for config in candidate_configs}
    scope_pair_discriminators = {}
    for label, left, right in scope_pairs:
        tail = operation_tail[config_by_id[left]["operation"]]
        operation_rows = [row for row in rows if row["tail"] == tail]
        worst = max(
            operation_rows,
            key=lambda row: abs(
                float(row[f"candidate_{left}_predicted_tail_delta"])
                - float(row[f"candidate_{right}_predicted_tail_delta"])
            ),
        )
        scope_pair_discriminators[label] = {
            "candidate_ids": [left, right],
            "tail": tail,
            "maximum_absolute_prediction_delta": abs(
                float(worst[f"candidate_{left}_predicted_tail_delta"])
                - float(worst[f"candidate_{right}_predicted_tail_delta"])
            ),
            "worst_cell": worst["cell"],
        }

    return {
        "topology_mismatches": topology_mismatches,
        "l2_request_mismatches": l2_request_mismatches,
        "gpc_instances_median": median(float(row["ncu_gpc_instances"]) for row in rows),
        "gpc_instance_spread_max_cycles": max(float(row["ncu_gpc_instance_spread"]) for row in rows),
        "ncu_gpc_cv_max_pct": 100.0 * max(float(row["ncu_gpc_cv"]) for row in rows),
        "ncu_sm_active_cv_max_pct": 100.0 * max(float(row["ncu_sm_active_cv"]) for row in rows),
        "fixed_ideal_concurrency": fixed_concurrency,
        "cta_shape": cta_shape,
        "arrival_skew": skew,
        "fixed_ideal_topology": topology,
        "work_surface_candidates": work_surface_candidates,
        "memory_service_candidates": candidate_scores,
        "selected_memory_service_candidates": selected_candidates,
        "scope_pair_discriminators": scope_pair_discriminators,
        "carded_dram_changed_any_timing": any(
            row["sim_last_issue"] != row["sim_carded_last_issue"]
            or row["sim_terminal"] != row["sim_carded_terminal"]
            for row in rows
        ),
    }


def markdown(integrity: Mapping[str, object], result: Mapping[str, object]) -> str:
    selected_store = result["selected_memory_service_candidates"]["store"]
    selected_red = result["selected_memory_service_candidates"]["red"]
    topology = {
        (tail, str(item["layout"]), int(item["lanes"]), int(item["depth"])): item
        for tail, items in result["fixed_ideal_topology"].items()
        for item in items
    }
    store_line_items = [
        item
        for (tail, layout, _lanes, _depth), item in topology.items()
        if tail == "store" and layout == "line"
    ]
    store_word = next(
        item
        for (tail, layout, _lanes, _depth), item in topology.items()
        if tail == "store" and layout == "word"
    )
    store_line_spread = max(float(item["gpc_tail_delta"]) for item in store_line_items) - min(
        float(item["gpc_tail_delta"]) for item in store_line_items
    )
    skew = {
        (tail, int(item["warps"]), int(item["gap"])): item
        for tail, items in result["arrival_skew"].items()
        for item in items
    }
    store_skew_start = skew[("store", 8, 0)]
    store_skew_end = skew[("store", 8, 256)]
    store_word_error = selected_store["held_out_by_stage"][
        "fixed_ideal_word_topology"
    ]["max_error"]
    maximum_scope_divergence = max(
        float(item["maximum_absolute_prediction_delta"])
        for item in result["scope_pair_discriminators"].values()
    )
    store_config = selected_store["configuration"]
    red_config = selected_red["configuration"]
    fixed_concurrency = {
        (tail, int(item["warps"])): item
        for tail, items in result["fixed_ideal_concurrency"].items()
        for item in items
    }
    load_concurrency_ratio = (
        float(fixed_concurrency[("load", 1)]["gpc_tail_delta"])
        / float(fixed_concurrency[("load", 8)]["gpc_tail_delta"])
    )
    store_concurrency_ratio = (
        float(fixed_concurrency[("store", 8)]["gpc_tail_delta"])
        / float(fixed_concurrency[("store", 1)]["gpc_tail_delta"])
    )
    lines = [
        "# sm_86 L1TEX arrival/output-replay service campaign",
        "",
        "## Outcome",
        "",
        "The campaign promotes exact unique-128-byte-line and atomic-request work",
        "into readiness schema 6 and promotes a bounded, observational streaming",
        "candidate evaluator as schema 7, but does not promote a service rate. Equal",
        "ideal wavefront work does not imply equal time: load latency hiding, unique",
        "line pressure, warp concurrency, and arrival skew are independent dimensions.",
        "Output replay is a phase response rather than a scalar delay cause.",
        "",
        "## Coverage",
        "",
        f"- {integrity['matrix_cells']} cells and {integrity['correctness_executions']} correct executions.",
        f"- {integrity['ncu_profiles']} NCU profiles, {integrity['source_counter_profiles']} SourceCounters profiles, and {integrity['l2_request_profiles']} focused L2-request profiles.",
        f"- {integrity['nvbit_captures']} NVBit captures with {integrity['nvbit_records']:,} records.",
        f"- {integrity['simulator_replay_pairs']} ordinary and {integrity['carded_dram_replay_pairs']} carded replay pairs.",
        f"- {integrity['memory_service_candidate_replay_pairs']} bounded-candidate replay pairs; stripping the schema-7 candidate payload (currently carried by schema 8 when operand barriers are present) leaves {integrity['candidate_replays_with_base_output_changes']} base-output changes.",
        "- Every embedded candidate configuration matches the retained metadata snapshot.",
        f"- {integrity['static_functions']} functions and {integrity['static_instructions']} decoded instructions; zero unknown encodings.",
        f"- Maximum retained GPC/SM-active CV after targeted seven-repeat expansion is {result['ncu_gpc_cv_max_pct']:.3f}% / {result['ncu_sm_active_cv_max_pct']:.3f}%; high-variance line-stream phases remain explicit.",
        f"- Trace-derived instruction/sector/ideal-wavefront and focused L2-request topology have {len(result['topology_mismatches']) + len(result['l2_request_mismatches'])} mismatches.",
        "- NCU application replay collects metric groups on different executions;",
        "  cross-metric associations are replicated cell-level evidence, not",
        "  co-temporal mediation.",
        "- Template-specialized target/control kernels differ in SASS and register",
        "  count, so their subtraction is a whole-kernel marginal cost.",
        "",
        "## Fixed 512 ideal wavefronts per nominal SM",
        "",
        "| tail | resident warps | live GPC tail delta | live SM-active delta | GPC-max minus SM-active-max extrema delta | extra output replay wavefronts | sim issue span |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tail in ("load", "store", "red"):
        for item in result["fixed_ideal_concurrency"][tail]:
            lines.append(
                f"| {tail} | {item['warps']} | {item['gpc_tail_delta']:.1f} | "
                f"{item['sm_active_tail_delta']:.1f} | "
                f"{item['gpc_minus_sm_active_extrema_delta']:.1f} | "
                f"{item['extra_replay_wavefronts']:.1f} | {item['sim_issue_span']:.1f} |"
            )
    lines.extend(
        [
            "",
            "The extrema delta subtracts maxima that may belong to different GPC/SM",
            "instances. It is directional context only, can be negative, and is not an",
            "observed interval after the last active SM.",
            "",
            "## CTA partition at fixed eight warps / 512 ideal wavefronts per nominal SM",
            "",
            "| tail | CTA shape | live GPC tail delta | extra replay wavefronts | sim issue span |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for tail in ("load", "store", "red"):
        for item in result["cta_shape"][tail]:
            lines.append(
                f"| {tail} | {item['ctas_per_nominal_sm']}×{item['warps']} | "
                f"{item['gpc_tail_delta']:.1f} | {item['extra_replay_wavefronts']:.1f} | "
                f"{item['sim_issue_span']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Fixed ideal work and unique-line topology",
            "",
            "| tail | layout | lanes | depth | unique lines/SM | live GPC tail delta | extra replay wavefronts | sim issue span |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tail in ("load", "store"):
        for item in result["fixed_ideal_topology"][tail]:
            lines.append(
                f"| {tail} | {item['layout']} | {item['lanes']} | {item['depth']} | "
                f"{item['unique_lines_per_nominal_sm']:.0f} | {item['gpc_tail_delta']:.1f} | "
                f"{item['extra_replay_wavefronts']:.1f} | {item['sim_issue_span']:.1f} |"
            )
    lines.extend(
        [
            "",
            f"Store line arms stay within {store_line_spread:,.0f} cycles while preserving {store_line_items[0]['ideal_wavefronts_per_nominal_sm']:,.0f} ideal",
            f"wavefronts and {store_line_items[0]['unique_lines_per_nominal_sm']:,.0f} unique lines/SM despite 4–32 active lanes. The",
            f"word arm keeps {store_word['ideal_wavefronts_per_nominal_sm']:,.0f} ideal wavefronts, reduces unique lines to {store_word['unique_lines_per_nominal_sm']:,.0f}, and",
            f"falls to {store_word['gpc_tail_delta']:,.0f} cycles. Focused L2 requests exactly equal projected unique",
            "lines for load/store and exact atomic requests for RED.",
            "",
            "## Arrival-skew discriminator",
            "",
            "| tail | warps | gap steps/warp | live GPC tail delta | extra replay wavefronts | sim issue span |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tail in ("load", "store"):
        for item in result["arrival_skew"][tail]:
            lines.append(
                f"| {tail} | {item['warps']} | {item['gap']} | "
                f"{item['gpc_tail_delta']:.1f} | {item['extra_replay_wavefronts']:.1f} | "
                f"{item['sim_issue_span']:.1f} |"
            )
    lines.extend(
        [
            "",
            "Staggering W8 store arrivals from gap 0 to 256 reduces the tail from",
            f"{store_skew_start['gpc_tail_delta']:,.0f} to {store_skew_end['gpc_tail_delta']:,.0f} cycles while extra output replay rises from {store_skew_start['extra_replay_wavefronts']:,.0f} to",
            f"{store_skew_end['extra_replay_wavefronts']:,.0f} wavefronts. Output replay therefore cannot be priced as a positive",
            "per-wavefront tail penalty. The dependent gap also changes the generated",
            "STORE payload, so compression sensitivity prevents a timing-only causal claim.",
            "",
            "## Ideal-wavefront + unique-line work surfaces",
            "",
            "Each affine surface is trained only on W4 line work scaling plus the",
            "word-coalesced point, then tested on all other concurrency, CTA, skew,",
            "and lane-topology cells.",
            "",
            "| tail | training MAE | held-out MAE | held-out max error |",
            "|---|---:|---:|---:|",
        ]
    )
    for tail in ("load", "store"):
        item = result["work_surface_candidates"][tail]
        lines.append(
            f"| {tail} | {item['training_mae']:.1f} | {item['held_out_mae']:.1f} | "
            f"{item['held_out_max_error']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Streaming service candidates at exact issue timestamps",
            "",
            "Sixteen fixed-capacity candidates were evaluated online without adding",
            "events or changing scheduling: eight STORE and eight RED hypotheses spanning",
            "SM/SMSP scope and unique-line/ideal-wavefront work. Selection used only the",
            "W1/W4 work-scaling arms; concurrency, CTA transfer, arrival skew, and topology",
            "arms were held out structurally. This is a retrospective stress test over a",
            "previously observed corpus, not a prospective validation set.",
            "",
            "| tail | selected candidate | service hypothesis | calibration MAE | held-out MAE | held-out max | prior terminal held-out MAE |",
            "|---|---:|---|---:|---:|---:|---:|",
            f"| store | {selected_store['candidate_id']} | {store_config['scope']}-shared, {store_config['unique_line_cycles']} cycles / {store_config['unique_line_units']} unique line | {selected_store['calibration']['mae']:.1f} | {selected_store['held_out']['mae']:.1f} | {selected_store['held_out']['max_error']:.1f} | {selected_store['baseline_sim_terminal_held_out']['mae']:.1f} |",
            f"| red | {selected_red['candidate_id']} | {red_config['scope']}-shared, {red_config['unique_line_cycles']} cycles / {red_config['unique_line_units']} unique line | {selected_red['calibration']['mae']:.1f} | {selected_red['held_out']['mae']:.1f} | {selected_red['held_out']['max_error']:.1f} | {selected_red['baseline_sim_terminal_held_out']['mae']:.1f} |",
            "",
            f"Calibration does not identify the named work basis: STORE candidates {selected_store['calibration_tied_candidate_ids']} and RED candidates {selected_red['calibration_tied_candidate_ids']} tie exactly. IDs {selected_store['candidate_id']}/{selected_red['candidate_id']} are frozen representatives, not uniquely selected mechanisms.",
            f"Candidate {selected_store['candidate_id']} follows the observed STORE skew response instead of treating",
            "replay as a scalar penalty, but the payload confound and retrospective",
            "selection prevent a timing-only causal interpretation.",
            f"The word-coalesced holdout remains its worst topology family ({store_word_error:,.0f} cycles),",
            "so the line rate is not yet architecture-card material. Rate-normalized SMSP",
            f"variants diverge by as much as {maximum_scope_divergence:,.0f} cycles when a CTA shape leaves some",
            "subcores underused, rejecting the tested independent-per-SMSP completion rate. Multiple",
            "SM-scoped RED work bases remain observationally tied on this all-line corpus; a",
            "RED topology transfer is required to identify the actual work basis.",
            "",
            "Full skew, topology, traffic, and replay rows are in `cells.csv`;",
            "machine-readable reductions are in `summary.json`.",
            "",
            "## Decision",
            "",
            "- Promote exact unique-line counts and atomic L2-request work as additive",
            "  schema 6 diagnostics; focused NCU LTS requests validate them with zero",
            "  mismatches.",
            "- Do not activate a work-only, per-SMSP-only, or SM-shared fixed-rate",
            f"  service. Fixed work changes load by {load_concurrency_ratio:.2f}× across W1–W8 and store by",
            f"  {store_concurrency_ratio:.2f}× in the opposite direction.",
            "- Do not turn output replay into delay. CTA shapes have near-stable time",
            "  with different replay counts, and skew increases replay while sharply",
            "  reducing time.",
            "- Retain actual per-instruction arrival sequencing as the next requirement.",
            "  The schema-7 evaluator now does this online with fixed memory and zero",
            f"  scheduling effect; all {integrity['memory_service_candidate_replay_pairs']} stripped outputs exactly match ordinary replay.",
            f"- Retain the {store_config['unique_line_cycles']}-cycle STORE-line and {red_config['unique_line_cycles']}-cycle RED-line rates only as",
            "  prospective hypotheses. RED work basis is unidentified, STORE has a",
            f"  {store_word_error:,.0f}-cycle word-topology miss, and neither rate has crossed an unseen",
            "  address-topology/atomic campaign.",
            "- Reject the tested rate-normalized independent per-SMSP completion",
            "  server. This does not identify the scope of a distinct admission queue;",
            "  the prospective successor retains SM-shared and per-SMSP admission",
            "  candidates feeding an SM-shared downstream service.",
            "- Store/RED retirement remains typed incomplete and numeric GPC emission",
            "  remains unavailable.",
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
    result = summarize(root, rows)
    output = root / "analysis"
    write_csv(output / "cells.csv", rows)
    write_json(output / "summary.json", {"integrity": integrity, "results": result})
    (output / "README.md").write_text(markdown(integrity, result), encoding="utf-8")
    print(output / "README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
