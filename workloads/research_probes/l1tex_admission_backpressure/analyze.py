#!/usr/bin/env python3
"""Reduce the prospective sm_86 L1TEX admission/backpressure campaign."""

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
    DEFAULT_OUTPUT,
    METRICS,
    cell_document,
    configured_cells,
    measurement_matches_cell,
    ncu_profile_command,
    ncu_source_command,
    source_cells,
)


GPC = "gpc__cycles_elapsed.max"
GPC_AVG = "gpc__cycles_elapsed.avg"
GPC_MIN = "gpc__cycles_elapsed.min"
GPC_SUM = "gpc__cycles_elapsed.sum"
SM_ACTIVE = "sm__cycles_active.max"
LG = "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"
MEMBAR = "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio"
DRAIN = "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio"
LONG = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
MIO = "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
SOURCE_STALL_FIELDS = (
    "stall_lg (Not Issued)",
    "stall_membar (Not Issued)",
    "stall_long_sb (Not Issued)",
    "stall_dispatch (Not Issued)",
    "stall_drain (Not Issued)",
)


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
    return mean(abs(left - right) for left, right in zip(predicted, observed, strict=True))


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


def ncu_raw_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("ID") or "").strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one NCU row, got {len(rows)}")
    return {key: raw for key, raw in rows[0].items() if key is not None and raw is not None}


def ncu_row(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in ncu_raw_row(path).items():
        if not raw.strip():
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            result[key] = value
    return result


def observation(root: Path, key: str) -> dict[str, object]:
    paths = sorted((root / "ncu" / key).glob("repeat-*/ncu.csv"))
    if len(paths) not in {3, 5, 7}:
        raise RuntimeError(f"{key}: expected 3/5/7 NCU rows, got {len(paths)}")
    rows = [ncu_row(path) for path in paths]
    keys = set.intersection(*(set(row) for row in rows))
    return {
        "median": {metric: median(row[metric] for row in rows) for metric in keys},
        "cv": {metric: cv([row[metric] for row in rows]) for metric in keys},
        "minimum": {metric: min(row[metric] for row in rows) for metric in keys},
        "maximum": {metric: max(row[metric] for row in rows) for metric in keys},
        "balanced_three_median": {
            metric: median(row[metric] for row in rows[:3]) for metric in keys
        },
        "repeat_count": len(rows),
    }


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
        root / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        campaign.REPO_ROOT / "arch_profiles/cards/sm_86.json",
        campaign.REPO_ROOT / "data/pchase/sm_86_static_card.json",
        trace,
    ]
    launches = campaign.REPLAY.expected_launches_from_trace(
        trace,
        grid=(cell.blocks, 1, 1),
        block=(cell.warps * 32, 1, 1),
    )
    expectation_kwargs: dict[str, object] = {}
    if directory == "simulator-memory-service-candidates":
        expectation_kwargs["service_candidates"] = (
            campaign.REPLAY.candidate_configurations(campaign.CANDIDATE_FILE)
        )
        dependencies.extend(
            [campaign.CANDIDATE_FILE, root / "metadata/memory_service_candidates.json"]
        )
    elif directory == "simulator-memory-admission-candidates":
        expectation_kwargs["admission_candidates"] = (
            campaign.REPLAY.candidate_configurations(
                campaign.ADMISSION_CANDIDATE_FILE
            )
        )
        dependencies.extend(
            [
                campaign.ADMISSION_CANDIDATE_FILE,
                root / "metadata/memory_admission_candidates.json",
            ]
        )
    elif directory.startswith("simulator-memory-admission-feedback-"):
        try:
            candidate_id = int(directory.rsplit("-", 1)[1])
            source = campaign.FEEDBACK_CANDIDATE_FILES[candidate_id]
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"unknown replay directory: {directory}") from error
        candidates = campaign.REPLAY.candidate_configurations(
            source, require_exactly_one=True
        )
        expectation_kwargs["feedback_candidate"] = candidates[0]
        dependencies.extend(
            [source, root / f"metadata/memory_admission_feedback_{candidate_id}.json"]
        )
    elif directory != "simulator":
        raise RuntimeError(f"unknown replay directory: {directory}")
    expectation = campaign.REPLAY.ReplayExpectation(
        launches=launches,
        carded_dram=False,
        **expectation_kwargs,
    )
    try:
        document = campaign.REPLAY.load_replay_pair(
            first, second, dependencies, expectation
        )
    except RuntimeError as error:
        raise RuntimeError(f"{directory}/{key}: {error}") from error
    for repeat in (1, 2):
        stderr = root / directory / key / f"replay-{repeat}.stderr.txt"
        if not stderr.is_file() or stderr.stat().st_size != 0:
            raise RuntimeError(f"{directory}/{key}: replay {repeat} stderr is nonempty")
    launches = document["launches"]
    if len(launches) != 1:
        raise RuntimeError(f"{directory}/{key}: expected one launch")
    return launches[0]


def native_measurement(root: Path, key: str) -> dict:
    return json.loads(
        (root / "correctness" / key / "result.json").read_text(encoding="utf-8")
    )["measurement"]


def native_timing_observation(root: Path, key: str) -> dict[str, float]:
    paths = sorted((root / "native-timing" / key).glob("repeat-*/result.json"))
    if len(paths) != 7:
        raise RuntimeError(f"{key}: expected seven independent native timing runs")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(
        row.get("correct") is not True
        or row.get("trials") != 1
        or row.get("l2_eviction_bytes") != 32 * 1024 * 1024
        for row in rows
    ):
        raise RuntimeError(f"{key}: invalid independent native timing run")
    issue = [float(row["issue_cycles_median"]) for row in rows]
    fence = [float(row["fence_cycles_median"]) for row in rows]
    return {
        "issue_cycles_median": median(issue),
        "fence_cycles_median": median(fence),
        "issue_cycles_cv": cv(issue),
        "fence_cycles_cv": cv(fence),
        "issue_cycles_min": min(issue),
        "issue_cycles_max": max(issue),
        "fence_cycles_min": min(fence),
        "fence_cycles_max": max(fence),
        "repeat_count": len(rows),
    }


def control_key(cell) -> str:
    return (
        f"b{cell.blocks}_w{cell.warps}_control_{cell.layout}_"
        f"d{cell.depth}_g{cell.gap_steps}_{cell.pattern}"
    )


def operation_metrics(tail: str) -> tuple[str, str, str, str]:
    if tail == "store":
        return (
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_st.sum",
            "lts__t_requests_srcunit_tex_op_write.sum",
        )
    if tail == "red":
        return (
            "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
            "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red.sum",
            "lts__t_requests_srcunit_tex_op_red.sum",
        )
    if tail == "atom":
        return (
            "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
            "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum",
            "lts__t_requests_srcunit_tex_op_atom_dot_alu.sum",
        )
    return ("", "", "", "")


def memory_diagnostics(launch: Mapping[str, object], tail: str) -> dict[str, int]:
    memory = launch["timing"]["readiness"].get("global_memory_issue", {})
    prefix = {"store": "store", "red": "reduction", "atom": "atomic"}.get(tail)
    if prefix is None:
        return {
            "instructions": 0,
            "operand_read_barrier_instructions": 0,
            "sectors": 0,
            "ideal_wavefronts": 0,
            "unique_lines": 0,
            "l2_requests": 0,
        }
    return {
        "instructions": int(memory.get(f"{prefix}_instructions", 0)),
        "operand_read_barrier_instructions": int(
            memory.get(f"{prefix}_operand_read_barrier_instructions", 0)
        ),
        "sectors": int(memory.get(f"{prefix}_sectors", 0)),
        "ideal_wavefronts": int(memory.get(f"{prefix}_ideal_wavefronts", 0)),
        "unique_lines": int(memory.get(f"{prefix}_unique_lines", 0)),
        "l2_requests": int(memory.get(f"{prefix}_l2_requests", 0)),
    }


def subtract(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    return {key: int(left[key]) - int(right[key]) for key in left}


def l2_source_diagnostics(launch: Mapping[str, object], tail: str) -> dict[str, int]:
    lane_name = "bypass" if tail == "atom" else "store"
    lane = launch["counters"].get("l2_by_source", {}).get(lane_name, {})
    sectors = int(lane.get("sectors", 0))
    hits = int(lane.get("hits", 0))
    return {"sectors": sectors, "hits": hits, "misses": sectors - hits}


def run_document(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{path}: invalid run document") from error
    if document.get("return_code") != 0 or not isinstance(document.get("command"), list):
        raise RuntimeError(f"{path}: unsuccessful or malformed retained run")
    return document


def validate_command(path: Path, expected: Sequence[str]) -> None:
    command = run_document(path)["command"]
    if command != list(expected):
        raise RuntimeError(f"{path}: retained command/cell binding mismatch")


def expected_matrix_cell(cell) -> dict[str, object]:
    return {
        **cell_document(cell),
        "selected_warps_per_block": cell.selected_warps_per_block,
        "memory_instructions": cell.memory_instructions,
        "unique_lines_per_instruction": cell.unique_lines_per_instruction,
        "ideal_wavefronts_per_instruction": cell.ideal_wavefronts_per_instruction,
        "sectors_per_instruction": cell.sectors_per_instruction,
        "atomic_requests_per_instruction": cell.atomic_requests_per_instruction,
    }


def validate_ncu_binding(path: Path, cell) -> tuple[int, int]:
    row = ncu_raw_row(path)
    expected_tail_id = {"control": 0, "store": 1, "red": 2, "atom": 3}[cell.tail]
    expected_kernel = f"admission_backpressure_tail<{expected_tail_id}>"
    checks = {
        "launch__grid_size": cell.blocks,
        "launch__block_size": cell.warps * 32,
        "launch__sm_count": 84,
        "device__attribute_compute_capability_major": 8,
        "device__attribute_compute_capability_minor": 6,
    }
    for field, expected in checks.items():
        try:
            observed = int(float(row[field].replace(",", "")))
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"{path}: missing NCU launch binding {field}") from error
        if observed != expected:
            raise RuntimeError(
                f"{path}: NCU launch binding {field}={observed}, expected {expected}"
            )
    if expected_kernel not in row.get("Kernel Name", ""):
        raise RuntimeError(f"{path}: NCU kernel/tail binding mismatch")
    try:
        replay_passes = int(float(row["profiler__replayer_passes"]))
        l2_bytes = int(float(row["device__attribute_l2_cache_size"]))
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{path}: missing NCU replay/cache provenance") from error
    return replay_passes, l2_bytes


def normalized_sass(text: str) -> str:
    return " ".join(text.replace(",", " ").split())


def validate_static_claims(sidecar: Mapping[str, object]) -> None:
    functions = sidecar["functions"]
    for tail_id, producer_prefix, barrier_field, barrier_id in (
        (1, "STG.E.STRONG.SM", "read_barrier", 0),
        (2, "RED.E.ADD.STRONG.GPU", "read_barrier", 0),
        (3, "ATOMG.E.ADD.STRONG.GPU", "write_barrier", 2),
    ):
        function = f"_Z27admission_backpressure_tailILi{tail_id}EEvPjPyS1_iiiiii"
        instructions = functions[function]
        producers = [
            (index, instruction)
            for index, instruction in enumerate(instructions)
            if str(instruction["opcode_text"]).startswith(producer_prefix)
        ]
        if len(producers) != 1:
            raise RuntimeError(f"static producer identification failed for tail {tail_id}")
        index, producer = producers[0]
        if int(producer["control"][barrier_field]) != barrier_id:
            raise RuntimeError(f"static producer barrier mismatch for tail {tail_id}")
        if not any(
            int(instruction["control"]["wait_mask"]) & (1 << barrier_id)
            for instruction in instructions[index + 1 :]
        ):
            raise RuntimeError(f"static producer wait mismatch for tail {tail_id}")

    for function, instructions in functions.items():
        opcodes = [str(instruction["opcode_text"]) for instruction in instructions]
        membar = next(
            (index for index, opcode in enumerate(opcodes) if opcode == "MEMBAR.SC.GPU"),
            None,
        )
        cctl = next(
            (
                index
                for index, opcode in enumerate(opcodes)
                if membar is not None and index > membar and opcode == "CCTL.IVALL"
            ),
            None,
        )
        final_clock = next(
            (
                index
                for index, opcode in enumerate(opcodes)
                if cctl is not None and index > cctl and opcode.startswith("CS2R ")
            ),
            None,
        )
        if membar is None or cctl is None or final_clock is None:
            raise RuntimeError(f"{function}: final fence/CLOCK sequence mismatch")


def validate_dynamic_static_join(root: Path, cell, sidecar: Mapping[str, object]) -> int:
    path = root / "nvbit" / cell.key / "extra_info/enhanced_execution_info.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    kernels = document.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise RuntimeError(f"{cell.key}: enhanced trace kernel coverage mismatch")
    kernel = kernels[0]
    tail_id = {"control": 0, "store": 1, "red": 2, "atom": 3}[cell.tail]
    function = f"_Z27admission_backpressure_tailILi{tail_id}EEvPjPyS1_iiiiii"
    if (
        kernel.get("architecture_version") != 86
        or not str(kernel.get("kernel_name", "")).startswith(function)
    ):
        raise RuntimeError(f"{cell.key}: enhanced trace kernel identity mismatch")
    dynamic = kernel.get("instructions")
    static = sidecar["functions"].get(function)
    if not isinstance(dynamic, list) or not isinstance(static, list) or len(dynamic) != len(static):
        raise RuntimeError(f"{cell.key}: dynamic/static instruction coverage mismatch")
    for dynamic_instruction, static_instruction in zip(dynamic, static, strict=True):
        predicate = None
        if dynamic_instruction.get("is_predicated"):
            predicate = "@" + ("!" if dynamic_instruction.get("is_predicate_negate") else "")
            predicate += "UP" if dynamic_instruction.get("is_uniform_predicate") else "P"
            predicate += str(dynamic_instruction.get("predicate_register"))
        operands = [
            str(operand.get("operand_string", ""))
            for operand in dynamic_instruction.get("operands", [])
        ]
        opcode_text = str(dynamic_instruction.get("op_code", ""))
        if operands:
            opcode_text += " " + ", ".join(operands)
        if (
            int(dynamic_instruction.get("pc_num_dec", -1)) != int(static_instruction["pc"])
            or predicate != static_instruction.get("predicate")
            or normalized_sass(opcode_text)
            != normalized_sass(str(static_instruction["opcode_text"]))
        ):
            raise RuntimeError(f"{cell.key}: dynamic/static SASS join mismatch")
    return len(dynamic)


def feedback_partitions(
    material: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    calibration = [
        row
        for row in material
        if row["stage"] == "burst_depth" and int(row["depth"]) <= 16
    ]
    held_out = [row for row in material if row not in calibration]
    if len(calibration) != 5 or len(held_out) != 20:
        raise RuntimeError("feedback selection split must contain 5 calibration/20 holdout cells")
    return calibration, held_out


def count_changed_outcomes(
    rows: Sequence[Mapping[str, object]],
    left_hit: str,
    left_miss: str,
    right_hit: str,
    right_miss: str,
) -> int:
    return sum(
        int(row[left_hit]) != int(row[right_hit])
        or int(row[left_miss]) != int(row[right_miss])
        for row in rows
    )


def validate_integrity(root: Path) -> dict[str, object]:
    cells = configured_cells()
    matrix = json.loads((root / "metadata/matrix.json").read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1 or matrix.get("cells") != [
        expected_matrix_cell(cell) for cell in cells
    ]:
        raise RuntimeError("retained matrix differs from configured cells")
    if matrix.get("ncu_metrics") != list(METRICS):
        raise RuntimeError("retained NCU metric configuration mismatch")
    if matrix.get("source_counter_cells") != [cell.key for cell in source_cells()]:
        raise RuntimeError("retained SourceCounters cell configuration mismatch")

    sidecar = json.loads(
        (root / "sass/l1tex_admission_backpressure_trace.control_bits.json").read_text(
            encoding="utf-8"
        )
    )
    statuses = [
        entry["status"]
        for entries in sidecar["functions"].values()
        for entry in entries
    ]
    if len(sidecar["functions"]) != 4 or any(status != "decoded" for status in statuses):
        raise RuntimeError("static sidecar incomplete")
    validate_static_claims(sidecar)
    long_sass = root / "sass/l1tex_admission_backpressure.sass.txt"
    trace_sass = root / "sass/l1tex_admission_backpressure_trace.sass.txt"
    if long_sass.read_bytes() != trace_sass.read_bytes():
        raise RuntimeError("retained long/trace static SASS differs")
    resource_lines = (
        root / "sass/l1tex_admission_backpressure_trace.resources.txt"
    ).read_text(encoding="utf-8").splitlines()
    function_resources = [
        resource_lines[index + 1].strip()
        for index, line in enumerate(resource_lines[:-1])
        if line.strip().startswith("Function ")
    ]
    if len(function_resources) != 4 or any(
        "STACK:0" not in line or "LOCAL:0" not in line
        for line in function_resources
    ):
        raise RuntimeError("retained static resources are not spill-free")

    correctness_executions = 0
    ncu_profiles = 0
    ncu_application_passes = 0
    ncu_replay_pass_counts: set[int] = set()
    ncu_l2_sizes: set[int] = set()
    records = 0
    native_timing_executions = 0
    dynamic_static_instructions = 0
    unresolved_module_origins = 0
    for cell in cells:
        correctness = root / "correctness" / cell.key
        correctness_document = json.loads(
            (correctness / "result.json").read_text(encoding="utf-8")
        )
        measurement = correctness_document.get("measurement")
        if correctness_document.get("cell") != cell_document(cell) or not measurement_matches_cell(
            measurement, cell, trials=3
        ):
            raise RuntimeError(f"{cell.key}: native correctness/cell binding failed")
        validate_command(
            correctness / "run.json", cell.target(trace=False, trials=3)
        )
        correctness_executions += int(measurement["trials"])

        native_paths = sorted(
            (root / "native-timing" / cell.key).glob("repeat-*/result.json")
        )
        if len(native_paths) != 7:
            raise RuntimeError(f"{cell.key}: incomplete native timing repeats")
        for result_path in native_paths:
            retained = json.loads(result_path.read_text(encoding="utf-8"))
            if not measurement_matches_cell(retained, cell, trials=1):
                raise RuntimeError(f"{cell.key}: native timing/cell binding failed")
            validate_command(
                result_path.parent / "run.json", cell.target(trace=False, trials=1)
            )
        native = native_timing_observation(root, cell.key)
        if not math.isfinite(native["issue_cycles_median"]) or not math.isfinite(
            native["fence_cycles_median"]
        ):
            raise RuntimeError(f"{cell.key}: non-finite native timing")
        native_timing_executions += len(native_paths)

        paths = sorted((root / "ncu" / cell.key).glob("repeat-*/ncu.csv"))
        if len(paths) not in {3, 5, 7}:
            raise RuntimeError(f"{cell.key}: incomplete NCU repeats")
        for path in paths:
            destination = path.parent
            actual_command = run_document(destination / "run.json")["command"]
            if not actual_command:
                raise RuntimeError(f"{destination}: empty NCU command")
            validate_command(
                destination / "run.json",
                ncu_profile_command(str(actual_command[0]), destination, cell),
            )
            replay_passes, l2_bytes = validate_ncu_binding(path, cell)
            ncu_replay_pass_counts.add(replay_passes)
            ncu_l2_sizes.add(l2_bytes)
            ncu_application_passes += replay_passes
        ncu_profiles += len(paths)

        trace = root / "nvbit" / cell.key
        if (trace / "dynamic_trace.pb").stat().st_size <= 0:
            raise RuntimeError(f"{cell.key}: empty trace")
        capture = json.loads((trace / "capture.json").read_text(encoding="utf-8"))
        if capture.get("cell") != cell_document(cell):
            raise RuntimeError(f"{cell.key}: NVBit capture/cell binding mismatch")
        validate_command(trace / "run.json", cell.target(trace=True, trials=1))
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
        launches = modules.get("launches", [])
        if len(launches) != 1:
            raise RuntimeError(f"{cell.key}: tracer module launch mismatch")
        unresolved_module_origins += launches[0].get("resolution_source") == "origin_unresolved"
        dynamic_static_instructions += validate_dynamic_static_join(root, cell, sidecar)

        ordinary = replay(root, "simulator", cell.key)
        candidate = replay(root, "simulator-memory-service-candidates", cell.key)
        admission_candidate = replay(
            root, "simulator-memory-admission-candidates", cell.key
        )
        if ordinary["counters"]["instructions_issued"] != ordinary["counters"]["records_seen"]:
            raise RuntimeError(f"{cell.key}: simulator issue/record mismatch")
        readiness = candidate["timing"]["readiness"]
        if readiness["schema_version"] != 8 or len(readiness["memory_service_candidates"]) != 16:
            raise RuntimeError(f"{cell.key}: candidate schema/count mismatch")
        stripped = copy.deepcopy(candidate)
        stripped_readiness = stripped["timing"]["readiness"]
        stripped_readiness.pop("memory_service_candidates")
        stripped_readiness["schema_version"] = ordinary["timing"]["readiness"]["schema_version"]
        if stripped != ordinary:
            raise RuntimeError(f"{cell.key}: candidate evaluation changed base replay")
        admission_readiness = admission_candidate["timing"]["readiness"]
        if (
            admission_readiness["schema_version"] != 9
            or len(admission_readiness["memory_admission_candidates"]) != 16
        ):
            raise RuntimeError(f"{cell.key}: admission candidate schema/count mismatch")
        stripped = copy.deepcopy(admission_candidate)
        stripped_readiness = stripped["timing"]["readiness"]
        stripped_readiness.pop("memory_admission_candidates")
        stripped_readiness["schema_version"] = ordinary["timing"]["readiness"][
            "schema_version"
        ]
        if stripped != ordinary:
            raise RuntimeError(
                f"{cell.key}: admission candidate evaluation changed base replay"
            )
        for feedback_id in (7, 12, 14):
            feedback = replay(
                root,
                f"simulator-memory-admission-feedback-{feedback_id}",
                cell.key,
            )
            feedback_readiness = feedback["timing"]["readiness"]
            if feedback_readiness["schema_version"] != 10:
                raise RuntimeError(
                    f"{cell.key}: feedback candidate {feedback_id} schema mismatch"
                )
            if (
                int(
                    feedback_readiness["memory_admission_feedback"]["candidate"][
                        "candidate_id"
                    ]
                )
                != feedback_id
            ):
                raise RuntimeError(
                    f"{cell.key}: feedback candidate {feedback_id} identity mismatch"
                )

    source_count = 0
    for cell in source_cells():
        destination = root / "ncu-source" / cell.key
        report = destination / "ncu-report.ncu-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("SourceCounters coverage incomplete")
        actual_command = run_document(destination / "run.json")["command"]
        if not actual_command:
            raise RuntimeError(f"{destination}: empty SourceCounters command")
        validate_command(
            destination / "run.json",
            ncu_source_command(str(actual_command[0]), destination, cell),
        )
        validate_ncu_binding(destination / "ncu.csv", cell)
        source_count += 1
    if ncu_replay_pass_counts != {9} or ncu_l2_sizes != {6 * 1024 * 1024}:
        raise RuntimeError("NCU replay-pass or L2-size provenance changed")
    return {
        "matrix_cells": len(cells),
        "correctness_executions": correctness_executions,
        "native_timing_executions": native_timing_executions,
        "ncu_profiles": ncu_profiles,
        "ncu_application_replay_passes": ncu_application_passes,
        "ncu_replay_passes_per_profile": sorted(ncu_replay_pass_counts),
        "ncu_l2_bytes": next(iter(ncu_l2_sizes)),
        "l2_eviction_sweep_bytes": 32 * 1024 * 1024,
        "l2_eviction_to_reported_capacity_ratio": (32 * 1024 * 1024)
        / next(iter(ncu_l2_sizes)),
        "source_counter_profiles": source_count,
        "nvbit_captures": len(cells),
        "nvbit_records": records,
        "nvbit_unresolved_module_origins": unresolved_module_origins,
        "dynamic_static_joined_instructions": dynamic_static_instructions,
        "simulator_replay_pairs": len(cells),
        "candidate_replay_pairs": len(cells),
        "candidate_base_output_changes": 0,
        "admission_candidate_replay_pairs": len(cells),
        "admission_candidate_base_output_changes": 0,
        "feedback_candidate_replay_pairs": len(cells) * 3,
        "static_long_trace_sass_identical": True,
        "static_spill_free_functions": len(function_resources),
        "static_functions": len(sidecar["functions"]),
        "static_instructions": len(statuses),
        "static_unknown_encodings": sum(status != "decoded" for status in statuses),
    }


def build_rows(root: Path) -> list[dict[str, object]]:
    raw = {}
    for cell in configured_cells():
        raw[cell.key] = (
            observation(root, cell.key),
            native_timing_observation(root, cell.key),
            replay(root, "simulator", cell.key),
            replay(root, "simulator-memory-service-candidates", cell.key),
            replay(root, "simulator-memory-admission-candidates", cell.key),
            {
                candidate_id: replay(
                    root,
                    f"simulator-memory-admission-feedback-{candidate_id}",
                    cell.key,
                )
                for candidate_id in (7, 12, 14)
            },
        )

    rows: list[dict[str, object]] = []
    for cell in configured_cells():
        (
            observed,
            native,
            launch,
            candidate_launch,
            admission_candidate_launch,
            feedback_launches,
        ) = raw[cell.key]
        (
            control_observed,
            control_native,
            control_launch,
            control_candidate,
            control_admission_candidate,
            control_feedback_launches,
        ) = raw[control_key(cell)]
        values = observed["median"]
        cvs = observed["cv"]
        control_values = control_observed["median"]
        request_metric, sector_metric, wave_metric, lts_metric = operation_metrics(cell.tail)
        ncu_requests = values.get(request_metric, 0.0) if request_metric else 0.0
        ncu_sectors = values.get(sector_metric, 0.0) if sector_metric else 0.0
        ncu_wavefronts = values.get(wave_metric, 0.0) if wave_metric else 0.0
        ncu_lts_requests = values.get(lts_metric, 0.0) if lts_metric else 0.0
        control_requests = control_values.get(request_metric, 0.0) if request_metric else 0.0
        control_sectors = control_values.get(sector_metric, 0.0) if sector_metric else 0.0
        control_wavefronts = control_values.get(wave_metric, 0.0) if wave_metric else 0.0
        control_lts_requests = control_values.get(lts_metric, 0.0) if lts_metric else 0.0
        sim_memory = subtract(
            memory_diagnostics(launch, cell.tail),
            memory_diagnostics(control_launch, cell.tail),
        )
        sim_l2_source = subtract(
            l2_source_diagnostics(launch, cell.tail),
            l2_source_diagnostics(control_launch, cell.tail),
        )
        expected_sectors = cell.memory_instructions * cell.sectors_per_instruction
        expected_ideal = cell.memory_instructions * cell.ideal_wavefronts_per_instruction
        expected_lines = cell.memory_instructions * cell.unique_lines_per_instruction
        expected_lts = cell.memory_instructions * (
            cell.atomic_requests_per_instruction
            if cell.tail in {"red", "atom"}
            else cell.unique_lines_per_instruction
        )

        candidate_results = {
            int(item["candidate"]["candidate_id"]): item
            for item in candidate_launch["timing"]["readiness"]["memory_service_candidates"]
        }
        control_candidate_results = {
            int(item["candidate"]["candidate_id"]): item
            for item in control_candidate["timing"]["readiness"]["memory_service_candidates"]
        }
        admission_candidate_results = {
            int(item["candidate"]["candidate_id"]): item
            for item in admission_candidate_launch["timing"]["readiness"][
                "memory_admission_candidates"
            ]
        }
        control_admission_candidate_results = {
            int(item["candidate"]["candidate_id"]): item
            for item in control_admission_candidate["timing"]["readiness"][
                "memory_admission_candidates"
            ]
        }
        tail_terminal = int(candidate_launch["timing"]["terminal_modeled_event_time"] or 0)
        control_terminal = int(control_candidate["timing"]["terminal_modeled_event_time"] or 0)

        row: dict[str, object] = {
            "cell": cell.key,
            "stage": cell.stage,
            "tail": cell.tail,
            "layout": cell.layout,
            "blocks": cell.blocks,
            "warps": cell.warps,
            "depth": cell.depth,
            "gap_steps": cell.gap_steps,
            "pattern": cell.pattern,
            "active_lanes": cell.active_lanes,
            "selected_warps_per_block": cell.selected_warps_per_block,
            "expected_memory_instructions": cell.memory_instructions,
            "expected_operand_read_barrier_instructions": (
                cell.memory_instructions if cell.tail in {"store", "red"} else 0
            ),
            "expected_sectors": expected_sectors,
            "expected_ideal_wavefronts": expected_ideal,
            "expected_dynamic_unique_lines": expected_lines,
            "expected_lts_requests": expected_lts,
            "native_issue_cycles": native["issue_cycles_median"],
            "native_fence_cycles": native["fence_cycles_median"],
            "native_issue_cv": native["issue_cycles_cv"],
            "native_fence_cv": native["fence_cycles_cv"],
            "native_repeat_count": native["repeat_count"],
            "control_native_repeat_count": control_native["repeat_count"],
            "native_issue_delta": int(native["issue_cycles_median"])
            - int(control_native["issue_cycles_median"]),
            "native_issue_delta_repeat_envelope_low": int(native["issue_cycles_min"])
            - int(control_native["issue_cycles_max"]),
            "native_issue_delta_repeat_envelope_high": int(native["issue_cycles_max"])
            - int(control_native["issue_cycles_min"]),
            "native_fence_delta": int(native["fence_cycles_median"])
            - int(control_native["fence_cycles_median"]),
            "native_fence_delta_repeat_envelope_low": int(native["fence_cycles_min"])
            - int(control_native["fence_cycles_max"]),
            "native_fence_delta_repeat_envelope_high": int(native["fence_cycles_max"])
            - int(control_native["fence_cycles_min"]),
            "native_issue_plus_fence_delta": (
                int(native["issue_cycles_median"])
                + int(native["fence_cycles_median"])
                - int(control_native["issue_cycles_median"])
                - int(control_native["fence_cycles_median"])
            ),
            "ncu_gpc_elapsed": values[GPC],
            "ncu_sm_active": values[SM_ACTIVE],
            "ncu_repeat_count": observed["repeat_count"],
            "control_ncu_repeat_count": control_observed["repeat_count"],
            "ncu_gpc_delta": values[GPC] - control_values[GPC],
            "ncu_gpc_delta_balanced_three": observed["balanced_three_median"][GPC]
            - control_observed["balanced_three_median"][GPC],
            "ncu_gpc_delta_repeat_envelope_low": observed["minimum"][GPC]
            - control_observed["maximum"][GPC],
            "ncu_gpc_delta_repeat_envelope_high": observed["maximum"][GPC]
            - control_observed["minimum"][GPC],
            "ncu_sm_active_delta": values[SM_ACTIVE] - control_values[SM_ACTIVE],
            "ncu_sm_active_delta_balanced_three": observed["balanced_three_median"][
                SM_ACTIVE
            ]
            - control_observed["balanced_three_median"][SM_ACTIVE],
            "ncu_sm_active_delta_repeat_envelope_low": observed["minimum"][SM_ACTIVE]
            - control_observed["maximum"][SM_ACTIVE],
            "ncu_sm_active_delta_repeat_envelope_high": observed["maximum"][SM_ACTIVE]
            - control_observed["minimum"][SM_ACTIVE],
            "ncu_gpc_max_minus_sm_active_max_delta": (values[GPC] - values[SM_ACTIVE])
            - (control_values[GPC] - control_values[SM_ACTIVE]),
            "ncu_requests_delta": ncu_requests - control_requests,
            "ncu_sectors_delta": ncu_sectors - control_sectors,
            "ncu_output_wavefronts_delta": ncu_wavefronts - control_wavefronts,
            "ncu_lts_requests_delta": ncu_lts_requests - control_lts_requests,
            "ncu_lg_throttle": values.get(LG, 0.0),
            "ncu_lg_throttle_delta": values.get(LG, 0.0) - control_values.get(LG, 0.0),
            "ncu_membar": values.get(MEMBAR, 0.0),
            "ncu_membar_delta": values.get(MEMBAR, 0.0) - control_values.get(MEMBAR, 0.0),
            "ncu_drain": values.get(DRAIN, 0.0),
            "ncu_drain_delta": values.get(DRAIN, 0.0) - control_values.get(DRAIN, 0.0),
            "ncu_long_scoreboard": values.get(LONG, 0.0),
            "ncu_long_scoreboard_delta": values.get(LONG, 0.0) - control_values.get(LONG, 0.0),
            "ncu_mio_throttle": values.get(MIO, 0.0),
            "ncu_gpc_cv": cvs[GPC],
            "ncu_sm_active_cv": cvs[SM_ACTIVE],
            "ncu_gpc_instances": values[GPC_SUM] / values[GPC_AVG],
            "ncu_gpc_instance_spread": values[GPC] - values[GPC_MIN],
            "sim_instructions_delta": sim_memory["instructions"],
            "sim_operand_read_barrier_instructions_delta": sim_memory[
                "operand_read_barrier_instructions"
            ],
            "sim_sectors_delta": sim_memory["sectors"],
            "sim_ideal_wavefronts_delta": sim_memory["ideal_wavefronts"],
            "sim_unique_lines_delta": sim_memory["unique_lines"],
            "sim_l2_requests_delta": sim_memory["l2_requests"],
            "sim_l2_source_sectors_delta": sim_l2_source["sectors"],
            "sim_l2_source_hits_delta": sim_l2_source["hits"],
            "sim_l2_source_misses_delta": sim_l2_source["misses"],
            "sim_last_issue_delta": launch["timing"]["diagnostic_last_issue_cycles"]
            - control_launch["timing"]["diagnostic_last_issue_cycles"],
            "sim_terminal_delta": launch["timing"]["terminal_modeled_event_time"]
            - control_launch["timing"]["terminal_modeled_event_time"],
            "sim_clock_interval": launch["timing"]["readiness"].get("clock_issue", {}).get(
                "maximum_first_to_last_issue_cycles", 0
            ),
        }
        for candidate_id in sorted(candidate_results):
            item = candidate_results[candidate_id]
            control_item = control_candidate_results[candidate_id]
            if item["candidate"] != control_item["candidate"]:
                raise RuntimeError(
                    f"{cell.key}: candidate {candidate_id} tail/control configuration mismatch"
                )
            terminal = int(item["terminal_completion_time"])
            control_candidate_terminal = int(control_item["terminal_completion_time"])
            row[f"candidate_{candidate_id}_terminal"] = terminal
            row[f"candidate_{candidate_id}_busy_cycles"] = int(item["busy_cycles"])
            row[f"candidate_{candidate_id}_queue_delay_cycles"] = int(item["queue_delay_cycles"])
            row[f"candidate_{candidate_id}_predicted_delta"] = max(
                tail_terminal, terminal
            ) - max(control_terminal, control_candidate_terminal)
            row[f"candidate_{candidate_id}_configuration"] = json.dumps(
                item["candidate"], sort_keys=True, separators=(",", ":")
            )
        for candidate_id in sorted(admission_candidate_results):
            item = admission_candidate_results[candidate_id]
            control_item = control_admission_candidate_results[candidate_id]
            if item["candidate"] != control_item["candidate"]:
                raise RuntimeError(
                    f"{cell.key}: admission candidate {candidate_id} tail/control configuration mismatch"
                )
            service_terminal = int(item["terminal_service_completion_time"])
            control_service_terminal = int(control_item["terminal_service_completion_time"])
            row[f"admission_candidate_{candidate_id}_predicted_cumulative_warp_issue_delta"] = int(
                item["maximum_warp_admission_delay_cycles"]
            ) - int(control_item["maximum_warp_admission_delay_cycles"])
            row[f"admission_candidate_{candidate_id}_predicted_issue_delta"] = int(
                item["maximum_admission_delay_cycles"]
            ) - int(control_item["maximum_admission_delay_cycles"])
            row[f"admission_candidate_{candidate_id}_predicted_terminal_delta"] = max(
                tail_terminal, service_terminal
            ) - max(control_terminal, control_service_terminal)
            for field in (
                "instructions_observed",
                "l2_hit_sectors_observed",
                "l2_miss_sectors_observed",
                "service_busy_cycles",
                "downstream_queue_delay_cycles",
                "queue_full_instructions",
                "total_admission_delay_cycles",
                "maximum_admission_delay_cycles",
                "maximum_warp_admission_delay_cycles",
                "terminal_operand_read_clear_time",
                "terminal_service_completion_time",
            ):
                row[f"admission_candidate_{candidate_id}_{field}"] = int(item[field])
            for field in (
                "instructions_observed",
                "l2_hit_sectors_observed",
                "l2_miss_sectors_observed",
                "service_busy_cycles",
                "downstream_queue_delay_cycles",
                "queue_full_instructions",
                "total_admission_delay_cycles",
            ):
                row[f"admission_candidate_{candidate_id}_{field}_delta"] = int(
                    item[field]
                ) - int(control_item[field])
            row[f"admission_candidate_{candidate_id}_configuration"] = json.dumps(
                item["candidate"], sort_keys=True, separators=(",", ":")
            )
        for feedback_id, feedback_launch in feedback_launches.items():
            control_feedback = control_feedback_launches[feedback_id]
            feedback_readiness = feedback_launch["timing"]["readiness"]
            control_feedback_readiness = control_feedback["timing"]["readiness"]
            item = feedback_readiness["memory_admission_feedback"]
            control_item = control_feedback_readiness["memory_admission_feedback"]
            if item["candidate"] != control_item["candidate"]:
                raise RuntimeError(
                    f"{cell.key}: feedback candidate {feedback_id} control configuration mismatch"
                )
            feedback_l2 = subtract(
                l2_source_diagnostics(feedback_launch, cell.tail),
                l2_source_diagnostics(control_feedback, cell.tail),
            )
            row[f"feedback_{feedback_id}_configuration"] = json.dumps(
                item["candidate"], sort_keys=True, separators=(",", ":")
            )
            row[f"feedback_{feedback_id}_sim_clock_delta"] = int(
                feedback_readiness.get("clock_issue", {}).get(
                    "maximum_first_to_last_issue_cycles", 0
                )
            ) - int(
                control_feedback_readiness.get("clock_issue", {}).get(
                    "maximum_first_to_last_issue_cycles", 0
                )
            )
            row[f"feedback_{feedback_id}_sim_last_issue_delta"] = int(
                feedback_launch["timing"]["diagnostic_last_issue_cycles"]
            ) - int(control_feedback["timing"]["diagnostic_last_issue_cycles"])
            row[f"feedback_{feedback_id}_sim_terminal_delta"] = int(
                feedback_launch["timing"]["terminal_modeled_event_time"] or 0
            ) - int(control_feedback["timing"]["terminal_modeled_event_time"] or 0)
            row[f"feedback_{feedback_id}_memory_admission_queue_cycles_delta"] = int(
                feedback_readiness["limiting_cycles"].get("memory_admission_queue", 0)
            ) - int(
                control_feedback_readiness["limiting_cycles"].get(
                    "memory_admission_queue", 0
                )
            )
            row[f"feedback_{feedback_id}_instructions_admitted_delta"] = int(
                item["instructions_admitted"]
            ) - int(control_item["instructions_admitted"])
            row[f"feedback_{feedback_id}_l2_hit_sectors_observed_delta"] = int(
                item["l2_hit_sectors_observed"]
            ) - int(control_item["l2_hit_sectors_observed"])
            row[f"feedback_{feedback_id}_l2_miss_sectors_observed_delta"] = int(
                item["l2_miss_sectors_observed"]
            ) - int(control_item["l2_miss_sectors_observed"])
            row[f"feedback_{feedback_id}_service_busy_cycles_delta"] = int(
                item["service_busy_cycles"]
            ) - int(control_item["service_busy_cycles"])
            row[f"feedback_{feedback_id}_terminal_service_completion_time"] = int(
                item["terminal_service_completion_time"]
            )
            row[f"feedback_{feedback_id}_l2_source_sectors_delta"] = feedback_l2[
                "sectors"
            ]
            row[f"feedback_{feedback_id}_l2_source_hits_delta"] = feedback_l2["hits"]
            row[f"feedback_{feedback_id}_l2_source_misses_delta"] = feedback_l2[
                "misses"
            ]
        rows.append(row)
    return rows


def score(rows: Sequence[Mapping[str, object]], predicted: str, observed: str) -> dict[str, object]:
    pairs = [(row, float(row[predicted]) - float(row[observed])) for row in rows]
    errors = [error for _, error in pairs]
    worst, worst_error = max(pairs, key=lambda item: abs(item[1]))
    return {
        "cells": len(rows),
        "mae": mean(abs(error) for error in errors),
        "max_error": max(abs(error) for error in errors),
        "mean_signed_error": mean(errors),
        "worst_cell": str(worst["cell"]),
        "worst_signed_error": worst_error,
    }


def score_observation_envelope(
    rows: Sequence[Mapping[str, object]], predicted: str, low: str, high: str
) -> dict[str, object]:
    lower_errors = []
    upper_errors = []
    for row in rows:
        prediction = float(row[predicted])
        lower = float(row[low])
        upper = float(row[high])
        if lower > upper:
            raise RuntimeError(f"{row['cell']}: inverted observation envelope")
        lower_errors.append(
            0.0
            if lower <= prediction <= upper
            else min(abs(prediction - lower), abs(prediction - upper))
        )
        upper_errors.append(max(abs(prediction - lower), abs(prediction - upper)))
    return {
        "cells": len(rows),
        "mae_lower_bound": mean(lower_errors),
        "mae_upper_bound": mean(upper_errors),
    }


def source_stall_samples(root: Path, key: str) -> dict[str, list[dict[str, object]]]:
    path = root / "ncu-source" / key / "source.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        handle.readline()
        rows = list(csv.DictReader(handle))
    result = {}
    for field in SOURCE_STALL_FIELDS:
        material = []
        for row in rows:
            raw = (row.get(field) or "").strip()
            try:
                samples = float(raw.replace(",", ""))
            except ValueError:
                continue
            if samples <= 0:
                continue
            material.append(
                {
                    "address": (row.get("Address") or "").strip(),
                    "sass": (row.get("Source") or "").strip(),
                    "samples": samples,
                }
            )
        result[field] = sorted(
            material, key=lambda item: float(item["samples"]), reverse=True
        )
    return result


def source_attribution(root: Path, key: str) -> dict[str, list[dict[str, object]]]:
    return {
        field: samples[:5]
        for field, samples in source_stall_samples(root, key).items()
    }


def summarize(root: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    topology_mismatches = []
    lts_request_differences = []
    for row in rows:
        if row["tail"] == "control":
            continue
        for label, observed, expected in (
            ("ncu_requests", row["ncu_requests_delta"], row["expected_memory_instructions"]),
            ("ncu_sectors", row["ncu_sectors_delta"], row["expected_sectors"]),
            ("sim_instructions", row["sim_instructions_delta"], row["expected_memory_instructions"]),
            (
                "sim_operand_read_barrier_instructions",
                row["sim_operand_read_barrier_instructions_delta"],
                row["expected_operand_read_barrier_instructions"],
            ),
            ("sim_sectors", row["sim_sectors_delta"], row["expected_sectors"]),
            ("sim_ideal_wavefronts", row["sim_ideal_wavefronts_delta"], row["expected_ideal_wavefronts"]),
            ("sim_unique_lines", row["sim_unique_lines_delta"], row["expected_dynamic_unique_lines"]),
            ("sim_l2_source_sectors", row["sim_l2_source_sectors_delta"], row["expected_sectors"]),
        ):
            if float(observed) != float(expected):
                topology_mismatches.append(
                    {"cell": row["cell"], "field": label, "observed": observed, "expected": expected}
                )
        if float(row["ncu_lts_requests_delta"]) != float(row["expected_lts_requests"]):
            lts_request_differences.append(
                {
                    "cell": row["cell"],
                    "observed": row["ncu_lts_requests_delta"],
                    "dynamic_line_or_atomic_requests": row["expected_lts_requests"],
                    "difference": float(row["ncu_lts_requests_delta"])
                    - float(row["expected_lts_requests"]),
                }
            )
        if row["tail"] in {"red", "atom"} and float(row["sim_l2_requests_delta"]) != float(
            row["expected_lts_requests"]
        ):
            topology_mismatches.append(
                {
                    "cell": row["cell"],
                    "field": "sim_l2_requests",
                    "observed": row["sim_l2_requests_delta"],
                    "expected": row["expected_lts_requests"],
                }
            )

    def selected(**criteria) -> list[Mapping[str, object]]:
        return [row for row in rows if all(row[key] == value for key, value in criteria.items())]

    burst = {
        tail: [
            {
                "depth": int(row["depth"]),
                "native_issue_delta": float(row["native_issue_delta"]),
                "native_fence_delta": float(row["native_fence_delta"]),
                "ncu_gpc_delta": float(row["ncu_gpc_delta"]),
                "ncu_sm_active_delta": float(row["ncu_sm_active_delta"]),
                "lg_throttle_delta": float(row["ncu_lg_throttle_delta"]),
                "membar_delta": float(row["ncu_membar_delta"]),
                "long_scoreboard_delta": float(row["ncu_long_scoreboard_delta"]),
            }
            for row in sorted(selected(stage="burst_depth", tail=tail), key=lambda item: int(item["depth"]))
        ]
        for tail in ("store", "red", "atom")
    }
    spacing = {
        tail: [
            {
                "gap_steps": int(row["gap_steps"]),
                "native_issue_delta": float(row["native_issue_delta"]),
                "native_fence_delta": float(row["native_fence_delta"]),
                "ncu_gpc_delta": float(row["ncu_gpc_delta"]),
                "lg_throttle_delta": float(row["ncu_lg_throttle_delta"]),
            }
            for row in sorted(
                [
                    row
                    for row in rows
                    if row["tail"] == tail
                    and row["blocks"] == 84
                    and row["warps"] == 8
                    and row["layout"] == "line_unique"
                    and row["depth"] == 32
                    and row["pattern"] == "all"
                    and row["gap_steps"] in {0, 1, 4, 16, 64}
                ],
                key=lambda item: int(item["gap_steps"]),
            )
        ]
        for tail in ("store", "red", "atom")
    }
    topology = {
        tail: [
            {
                "layout": str(row["layout"]),
                "native_issue_delta": float(row["native_issue_delta"]),
                "native_fence_delta": float(row["native_fence_delta"]),
                "ncu_gpc_delta": float(row["ncu_gpc_delta"]),
                "ncu_lts_requests_delta": float(row["ncu_lts_requests_delta"]),
                "ncu_output_wavefronts_delta": float(row["ncu_output_wavefronts_delta"]),
                "sim_l2_hits": float(row["sim_l2_source_hits_delta"]),
                "sim_l2_misses": float(row["sim_l2_source_misses_delta"]),
            }
            for row in rows
            if row["tail"] == tail
            and row["blocks"] == 84
            and row["warps"] == 8
            and row["depth"] == 32
            and row["gap_steps"] == 0
            and row["pattern"] == "all"
        ]
        for tail in ("store", "red", "atom")
    }
    sparse = {
        tail: [
            {
                "pattern": str(row["pattern"]),
                "selected_warps": int(row["selected_warps_per_block"]),
                "native_issue_delta": float(row["native_issue_delta"]),
                "native_fence_delta": float(row["native_fence_delta"]),
                "ncu_gpc_delta": float(row["ncu_gpc_delta"]),
            }
            for row in rows
            if row["tail"] == tail
            and row["blocks"] == 84
            and row["warps"] == 8
            and row["depth"] == 32
            and row["gap_steps"] == 0
            and row["layout"] == "line_unique"
            and row["pattern"] in {"all", "one_per_smsp", "same_smsp", "two_smsps"}
        ]
        for tail in ("store", "red", "atom")
    }
    cta = {
        tail: [
            {
                "blocks": int(row["blocks"]),
                "warps": int(row["warps"]),
                "native_issue_delta": float(row["native_issue_delta"]),
                "native_fence_delta": float(row["native_fence_delta"]),
                "ncu_gpc_delta": float(row["ncu_gpc_delta"]),
            }
            for row in rows
            if row["tail"] == tail
            and row["layout"] == "line_unique"
            and row["depth"] == 32
            and row["gap_steps"] == 0
            and row["pattern"] == "all"
            and (row["blocks"], row["warps"]) in {(84, 8), (168, 4), (336, 2)}
        ]
        for tail in ("store", "red", "atom")
    }

    candidate_scores: dict[str, dict[str, object]] = {"store": {}, "red": {}}
    candidate_transfer = {}
    retained_candidate_document = json.loads(
        (root / "metadata/memory_service_candidates.json").read_text(encoding="utf-8")
    )
    retained_configs = {
        int(config["candidate_id"]): config
        for config in retained_candidate_document["candidates"]
    }
    frozen_ids = {"store": 2, "red": 10}
    for tail, operation in (("store", "store"), ("red", "reduction")):
        material = [row for row in rows if row["tail"] == tail]
        candidate_ids = []
        for candidate_id in range(1, 17):
            field = f"candidate_{candidate_id}_configuration"
            config = json.loads(str(material[0][field]))
            if config["operation"] != operation:
                continue
            if any(json.loads(str(row[field])) != config for row in rows):
                raise RuntimeError(
                    f"candidate {candidate_id} configuration differs across replay cells"
                )
            if retained_configs.get(candidate_id) != config:
                raise RuntimeError(
                    f"candidate {candidate_id} replay differs from retained configuration snapshot"
                )
            candidate_ids.append(candidate_id)
            candidate_scores[tail][str(candidate_id)] = {
                "configuration": config,
                "gpc": score(
                    material,
                    f"candidate_{candidate_id}_predicted_delta",
                    "ncu_gpc_delta",
                ),
                "native_issue_plus_fence": score(
                    material,
                    f"candidate_{candidate_id}_predicted_delta",
                    "native_issue_plus_fence_delta",
                ),
                "by_stage": {
                    stage: score(
                        [row for row in material if row["stage"] == stage],
                        f"candidate_{candidate_id}_predicted_delta",
                        "ncu_gpc_delta",
                    )
                    for stage in sorted({str(row["stage"]) for row in material})
                },
            }
        selected_id = frozen_ids[tail]
        if selected_id not in candidate_ids:
            raise RuntimeError(f"frozen {tail} candidate is absent")
        candidate_transfer[tail] = {
            "candidate_id": selected_id,
            **candidate_scores[tail][str(selected_id)],
            "raw_sim_terminal_gpc": score(
                material, "sim_terminal_delta", "ncu_gpc_delta"
            ),
        }

    admission_candidate_scores: dict[str, dict[str, object]] = {
        "store": {},
        "red": {},
    }
    selected_admission_candidates: dict[str, object] = {}
    retained_admission_document = json.loads(
        (root / "metadata/memory_admission_candidates.json").read_text(encoding="utf-8")
    )
    retained_admission_configs = {
        int(config["candidate_id"]): config
        for config in retained_admission_document["candidates"]
    }
    for tail, operation, candidate_ids in (
        ("store", "store", range(1, 9)),
        ("red", "reduction", range(9, 17)),
    ):
        material = [row for row in rows if row["tail"] == tail]
        calibration, held_out = feedback_partitions(material)
        for candidate_id in candidate_ids:
            field = f"admission_candidate_{candidate_id}_configuration"
            config = json.loads(str(material[0][field]))
            if config["operation"] != operation:
                raise RuntimeError(
                    f"admission candidate {candidate_id} has wrong operation"
                )
            if any(json.loads(str(row[field])) != config for row in rows):
                raise RuntimeError(
                    f"admission candidate {candidate_id} differs across replay cells"
                )
            if retained_admission_configs.get(candidate_id) != config:
                raise RuntimeError(
                    f"admission candidate {candidate_id} differs from retained snapshot"
                )
            for row in material:
                if int(row[f"admission_candidate_{candidate_id}_instructions_observed_delta"]) != int(
                    row["expected_memory_instructions"]
                ):
                    raise RuntimeError(
                        f"{row['cell']}: admission candidate {candidate_id} instruction mismatch"
                    )
                if int(row[f"admission_candidate_{candidate_id}_l2_hit_sectors_observed_delta"]) != int(
                    row["sim_l2_source_hits_delta"]
                ) or int(
                    row[f"admission_candidate_{candidate_id}_l2_miss_sectors_observed_delta"]
                ) != int(row["sim_l2_source_misses_delta"]):
                    raise RuntimeError(
                        f"{row['cell']}: admission candidate {candidate_id} L2 outcome mismatch"
                    )
            issue_field = f"admission_candidate_{candidate_id}_predicted_issue_delta"
            cumulative_field = (
                f"admission_candidate_{candidate_id}_predicted_cumulative_warp_issue_delta"
            )
            terminal_field = f"admission_candidate_{candidate_id}_predicted_terminal_delta"
            admission_candidate_scores[tail][str(candidate_id)] = {
                "configuration": config,
                "issue_calibration": score(calibration, issue_field, "native_issue_delta"),
                "issue_held_out": score(held_out, issue_field, "native_issue_delta"),
                "issue_all": score(material, issue_field, "native_issue_delta"),
                "cumulative_warp_issue_all": score(
                    material, cumulative_field, "native_issue_delta"
                ),
                "terminal_gpc_all": score(material, terminal_field, "ncu_gpc_delta"),
                "terminal_gpc_by_stage": {
                    stage: score(
                        [row for row in material if row["stage"] == stage],
                        terminal_field,
                        "ncu_gpc_delta",
                    )
                    for stage in sorted({str(row["stage"]) for row in material})
                },
                "by_stage": {
                    stage: score(
                        [row for row in material if row["stage"] == stage],
                        issue_field,
                        "native_issue_delta",
                    )
                    for stage in sorted({str(row["stage"]) for row in material})
                },
            }
        best = min(
            float(item["issue_calibration"]["mae"])
            for item in admission_candidate_scores[tail].values()
        )
        tied = sorted(
            int(candidate_id)
            for candidate_id, item in admission_candidate_scores[tail].items()
            if math.isclose(
                float(item["issue_calibration"]["mae"]),
                best,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        selected_id = tied[0]
        selected_admission_candidates[tail] = {
            "candidate_id": selected_id,
            "calibration_tied_candidate_ids": tied,
            **admission_candidate_scores[tail][str(selected_id)],
        }

    feedback_scores: dict[str, object] = {}
    feedback_baselines: dict[str, object] = {}
    expected_feedback_ids = {
        "store": {int(selected_admission_candidates["store"]["candidate_id"])},
        "red": set(
            int(candidate_id)
            for candidate_id in selected_admission_candidates["red"][
                "calibration_tied_candidate_ids"
            ]
        ),
    }
    if expected_feedback_ids != {"store": {7}, "red": {12, 14}}:
        raise RuntimeError(
            "schema-10 feedback IDs do not match schema-9 calibration selection"
        )
    for tail, candidate_ids in (("store", (7,)), ("red", (12, 14))):
        material = [row for row in rows if row["tail"] == tail]
        calibration, held_out = feedback_partitions(material)
        feedback_baselines[tail] = {
            "last_issue": score(material, "sim_last_issue_delta", "native_issue_delta"),
            "last_issue_calibration": score(
                calibration, "sim_last_issue_delta", "native_issue_delta"
            ),
            "last_issue_held_out": score(
                held_out, "sim_last_issue_delta", "native_issue_delta"
            ),
            "last_issue_held_out_repeat_envelope": score_observation_envelope(
                held_out,
                "sim_last_issue_delta",
                "native_issue_delta_repeat_envelope_low",
                "native_issue_delta_repeat_envelope_high",
            ),
            "terminal_gpc": score(material, "sim_terminal_delta", "ncu_gpc_delta"),
            "terminal_gpc_calibration": score(
                calibration, "sim_terminal_delta", "ncu_gpc_delta"
            ),
            "terminal_gpc_held_out": score(
                held_out, "sim_terminal_delta", "ncu_gpc_delta"
            ),
            "terminal_gpc_held_out_repeat_envelope": score_observation_envelope(
                held_out,
                "sim_terminal_delta",
                "ncu_gpc_delta_repeat_envelope_low",
                "ncu_gpc_delta_repeat_envelope_high",
            ),
            "terminal_gpc_held_out_balanced_three": score(
                held_out, "sim_terminal_delta", "ncu_gpc_delta_balanced_three"
            ),
        }
        feedback_scores[tail] = {}
        for candidate_id in candidate_ids:
            config = json.loads(
                str(material[0][f"feedback_{candidate_id}_configuration"])
            )
            snapshot = json.loads(
                (
                    root
                    / f"metadata/memory_admission_feedback_{candidate_id}.json"
                ).read_text(encoding="utf-8")
            )["candidates"][0]
            if config != snapshot or config != retained_admission_configs[candidate_id] or any(
                json.loads(str(row[f"feedback_{candidate_id}_configuration"]))
                != config
                for row in rows
            ):
                raise RuntimeError(
                    f"feedback candidate {candidate_id} configuration mismatch"
                )
            for row in material:
                if int(row[f"feedback_{candidate_id}_instructions_admitted_delta"]) != int(
                    row["expected_memory_instructions"]
                ):
                    raise RuntimeError(
                        f"{row['cell']}: feedback candidate {candidate_id} instruction mismatch"
                    )
            feedback_scores[tail][str(candidate_id)] = {
                "configuration": config,
                "clock_issue": score(
                    material,
                    f"feedback_{candidate_id}_sim_clock_delta",
                    "native_issue_delta",
                ),
                "clock_issue_calibration": score(
                    calibration,
                    f"feedback_{candidate_id}_sim_clock_delta",
                    "native_issue_delta",
                ),
                "clock_issue_held_out": score(
                    held_out,
                    f"feedback_{candidate_id}_sim_clock_delta",
                    "native_issue_delta",
                ),
                "last_issue": score(
                    material,
                    f"feedback_{candidate_id}_sim_last_issue_delta",
                    "native_issue_delta",
                ),
                "last_issue_calibration": score(
                    calibration,
                    f"feedback_{candidate_id}_sim_last_issue_delta",
                    "native_issue_delta",
                ),
                "last_issue_held_out": score(
                    held_out,
                    f"feedback_{candidate_id}_sim_last_issue_delta",
                    "native_issue_delta",
                ),
                "last_issue_held_out_repeat_envelope": score_observation_envelope(
                    held_out,
                    f"feedback_{candidate_id}_sim_last_issue_delta",
                    "native_issue_delta_repeat_envelope_low",
                    "native_issue_delta_repeat_envelope_high",
                ),
                "terminal_gpc": score(
                    material,
                    f"feedback_{candidate_id}_sim_terminal_delta",
                    "ncu_gpc_delta",
                ),
                "terminal_gpc_calibration": score(
                    calibration,
                    f"feedback_{candidate_id}_sim_terminal_delta",
                    "ncu_gpc_delta",
                ),
                "terminal_gpc_held_out": score(
                    held_out,
                    f"feedback_{candidate_id}_sim_terminal_delta",
                    "ncu_gpc_delta",
                ),
                "terminal_gpc_held_out_repeat_envelope": score_observation_envelope(
                    held_out,
                    f"feedback_{candidate_id}_sim_terminal_delta",
                    "ncu_gpc_delta_repeat_envelope_low",
                    "ncu_gpc_delta_repeat_envelope_high",
                ),
                "terminal_gpc_held_out_balanced_three": score(
                    held_out,
                    f"feedback_{candidate_id}_sim_terminal_delta",
                    "ncu_gpc_delta_balanced_three",
                ),
                "by_stage": {
                    stage: {
                        "clock_issue": score(
                            [row for row in material if row["stage"] == stage],
                            f"feedback_{candidate_id}_sim_clock_delta",
                            "native_issue_delta",
                        ),
                        "last_issue": score(
                            [row for row in material if row["stage"] == stage],
                            f"feedback_{candidate_id}_sim_last_issue_delta",
                            "native_issue_delta",
                        ),
                        "terminal_gpc": score(
                            [row for row in material if row["stage"] == stage],
                            f"feedback_{candidate_id}_sim_terminal_delta",
                            "ncu_gpc_delta",
                        ),
                    }
                    for stage in sorted({str(row["stage"]) for row in material})
                },
                "aggregate_store_source_changed_cells": count_changed_outcomes(
                    material,
                    f"feedback_{candidate_id}_l2_source_hits_delta",
                    f"feedback_{candidate_id}_l2_source_misses_delta",
                    "sim_l2_source_hits_delta",
                    "sim_l2_source_misses_delta",
                ),
                "candidate_observer_changed_cells": count_changed_outcomes(
                    material,
                    f"feedback_{candidate_id}_l2_hit_sectors_observed_delta",
                    f"feedback_{candidate_id}_l2_miss_sectors_observed_delta",
                    f"admission_candidate_{candidate_id}_l2_hit_sectors_observed_delta",
                    f"admission_candidate_{candidate_id}_l2_miss_sectors_observed_delta",
                ),
                "candidate_observer_vs_aggregate_mismatches": count_changed_outcomes(
                    material,
                    f"feedback_{candidate_id}_l2_hit_sectors_observed_delta",
                    f"feedback_{candidate_id}_l2_miss_sectors_observed_delta",
                    f"feedback_{candidate_id}_l2_source_hits_delta",
                    f"feedback_{candidate_id}_l2_source_misses_delta",
                ),
                "candidate_observer_negative_hit_cells": sum(
                    int(row[f"feedback_{candidate_id}_l2_hit_sectors_observed_delta"]) < 0
                    for row in material
                ),
                "candidate_observer_negative_miss_cells": sum(
                    int(row[f"feedback_{candidate_id}_l2_miss_sectors_observed_delta"])
                    < 0
                    for row in material
                ),
                "target_operation_cache_attribution": (
                    "unresolved_common_store_instrumentation"
                    if tail == "store"
                    else "operation_specific"
                ),
                "target_operation_changed_cells": (
                    None
                    if tail == "store"
                    else count_changed_outcomes(
                        material,
                        f"feedback_{candidate_id}_l2_hit_sectors_observed_delta",
                        f"feedback_{candidate_id}_l2_miss_sectors_observed_delta",
                        f"admission_candidate_{candidate_id}_l2_hit_sectors_observed_delta",
                        f"admission_candidate_{candidate_id}_l2_miss_sectors_observed_delta",
                    )
                ),
            }

    source_samples = {
        cell.key: source_stall_samples(root, cell.key) for cell in source_cells()
    }
    source_totals = {
        field: sum(
            float(item["samples"])
            for cell_samples in source_samples.values()
            for item in cell_samples[field]
        )
        for field in SOURCE_STALL_FIELDS
    }
    if source_totals["stall_lg (Not Issued)"] != 0:
        raise RuntimeError("SourceCounters unexpectedly attribute LG-throttle samples")

    noncontrol_rows = [row for row in rows if row["tail"] != "control"]
    adaptive_shift_worst = max(
        noncontrol_rows,
        key=lambda row: abs(
            float(row["ncu_gpc_delta"])
            - float(row["ncu_gpc_delta_balanced_three"])
        ),
    )
    return {
        "topology_mismatches": topology_mismatches,
        "lts_request_differences": lts_request_differences,
        "high_variance_cells": [
            {
                "cell": row["cell"],
                "gpc_cv_pct": 100.0 * float(row["ncu_gpc_cv"]),
                "sm_active_cv_pct": 100.0 * float(row["ncu_sm_active_cv"]),
            }
            for row in sorted(
                [
                    row
                    for row in rows
                    if float(row["ncu_gpc_cv"]) > 0.02
                    or float(row["ncu_sm_active_cv"]) > 0.02
                ],
                key=lambda item: max(float(item["ncu_gpc_cv"]), float(item["ncu_sm_active_cv"])),
                reverse=True,
            )
        ],
        "ncu_gpc_cv_max_pct": 100.0 * max(float(row["ncu_gpc_cv"]) for row in rows),
        "ncu_sm_active_cv_max_pct": 100.0 * max(float(row["ncu_sm_active_cv"]) for row in rows),
        "native_issue_cv_max_pct": 100.0 * max(float(row["native_issue_cv"]) for row in rows),
        "native_fence_cv_max_pct": 100.0 * max(float(row["native_fence_cv"]) for row in rows),
        "trial_uncertainty": {
            "ncu_repeat_count_distribution": {
                str(repeats): sum(int(row["ncu_repeat_count"]) == repeats for row in rows)
                for repeats in (3, 5, 7)
            },
            "tail_control_repeat_count_mismatches": sum(
                int(row["ncu_repeat_count"])
                != int(row["control_ncu_repeat_count"])
                for row in rows
            ),
            "ncu_gpc_balanced_three_shift_max_cycles": abs(
                float(adaptive_shift_worst["ncu_gpc_delta"])
                - float(adaptive_shift_worst["ncu_gpc_delta_balanced_three"])
            ),
            "ncu_gpc_balanced_three_shift_worst_cell": adaptive_shift_worst["cell"],
            "ncu_gpc_repeat_envelope_max_width_cycles": max(
                float(row["ncu_gpc_delta_repeat_envelope_high"])
                - float(row["ncu_gpc_delta_repeat_envelope_low"])
                for row in noncontrol_rows
            ),
            "ncu_gpc_repeat_envelope_zero_crossing_cells": sum(
                float(row["ncu_gpc_delta_repeat_envelope_low"]) <= 0
                <= float(row["ncu_gpc_delta_repeat_envelope_high"])
                for row in noncontrol_rows
            ),
            "native_issue_repeat_envelope_max_width_cycles": max(
                float(row["native_issue_delta_repeat_envelope_high"])
                - float(row["native_issue_delta_repeat_envelope_low"])
                for row in noncontrol_rows
            ),
            "native_issue_repeat_envelope_zero_crossing_cells": sum(
                float(row["native_issue_delta_repeat_envelope_low"]) <= 0
                <= float(row["native_issue_delta_repeat_envelope_high"])
                for row in noncontrol_rows
            ),
            "native_fence_repeat_envelope_max_width_cycles": max(
                float(row["native_fence_delta_repeat_envelope_high"])
                - float(row["native_fence_delta_repeat_envelope_low"])
                for row in noncontrol_rows
            ),
            "native_fence_repeat_envelope_zero_crossing_cells": sum(
                float(row["native_fence_delta_repeat_envelope_low"]) <= 0
                <= float(row["native_fence_delta_repeat_envelope_high"])
                for row in noncontrol_rows
            ),
            "interpretation": (
                "repeat envelopes are observed min/max sensitivity bounds, not "
                "confidence intervals"
            ),
        },
        "gpc_instances_median": median(float(row["ncu_gpc_instances"]) for row in rows),
        "gpc_instance_spread_max_cycles": max(float(row["ncu_gpc_instance_spread"]) for row in rows),
        "burst_depth": burst,
        "issue_spacing": spacing,
        "address_topology": topology,
        "sparse_smsp": sparse,
        "cta_transfer": cta,
        "frozen_candidate_transfer": candidate_transfer,
        "frozen_candidate_scores": candidate_scores,
        "memory_admission_candidate_scores": admission_candidate_scores,
        "selected_memory_admission_candidates": selected_admission_candidates,
        "memory_admission_feedback_scores": feedback_scores,
        "memory_admission_feedback_baselines": feedback_baselines,
        "feedback_selection_split": {
            "calibration": "burst_depth with depth <= 16",
            "calibration_cells_per_operation": 5,
            "held_out_cells_per_operation": 20,
            "holdout_semantics": (
                "excluded from schema-9 candidate-ID selection; the hardware corpus "
                "itself predates that retrospective selection"
            ),
        },
        "source_attribution_totals": source_totals,
        "source_attribution": {
            key: {field: samples[:5] for field, samples in cell_samples.items()}
            for key, cell_samples in source_samples.items()
        },
    }


def markdown(integrity: Mapping[str, object], result: Mapping[str, object]) -> str:
    burst = {
        (tail, int(item["depth"])): item
        for tail, items in result["burst_depth"].items()
        for item in items
    }
    spacing = {
        (tail, int(item["gap_steps"])): item
        for tail, items in result["issue_spacing"].items()
        for item in items
    }
    topology = {
        (tail, str(item["layout"])): item
        for tail, items in result["address_topology"].items()
        for item in items
    }
    sparse = {
        (tail, str(item["pattern"])): item
        for tail, items in result["sparse_smsp"].items()
        for item in items
    }
    cta_spans = {
        tail: max(float(item["ncu_gpc_delta"]) for item in items)
        - min(float(item["ncu_gpc_delta"]) for item in items)
        for tail, items in result["cta_transfer"].items()
    }
    high_cv_above_five = sum(
        max(float(item["gpc_cv_pct"]), float(item["sm_active_cv_pct"])) > 5.0
        for item in result["high_variance_cells"]
    )
    reuse_rows = []
    for tail in ("store", "red", "atom"):
        unique = topology[(tail, "line_unique")]
        reused = topology[(tail, "line_reuse")]
        reuse_rows.append(
            f"| {tail} | {unique['ncu_gpc_delta']:,.0f} | {reused['ncu_gpc_delta']:,.0f} | "
            f"{unique['sim_l2_misses']:,.0f} | {reused['sim_l2_misses']:,.0f} / "
            f"{reused['sim_l2_hits']:,.0f} |"
        )
    same_region_store = topology[("store", "warp_word_shared_region")]
    sparse_differences = {
        tail: abs(
            float(sparse[(tail, "one_per_smsp")]["ncu_gpc_delta"])
            - float(sparse[(tail, "two_smsps")]["ncu_gpc_delta"])
        )
        for tail in ("store", "red", "atom")
    }
    frozen = result["frozen_candidate_transfer"]
    selected_admission = result["selected_memory_admission_candidates"]
    feedback_scores = result["memory_admission_feedback_scores"]
    feedback_baselines = result["memory_admission_feedback_baselines"]
    uncertainty = result["trial_uncertainty"]
    source_totals = result["source_attribution_totals"]
    lines = [
        "# sm_86 L1TEX admission/backpressure campaign",
        "",
        "## Outcome",
        "",
        "The campaign rejects promotion of the predecessor's completion-only",
        "STORE/RED line rates. It identifies the missing causal boundary instead:",
        "compiler read-dependence barriers on STG/RED represent operand-consumption",
        "and queue-admission readiness, while returning ATOMG uses a result write",
        "barrier. Downstream L2 tag residency changes that admission pressure even",
        "when dynamic L1TEX work and LTS request counts are identical.",
        "",
        "Readiness schema 8 now attributes scalar-global operand-read producers and",
        "consumer delays by operation. Direct JSON also exposes the already-modeled",
        "L2 source-sector/hit split. Neither change alters scheduling or timing.",
        "Schema 9 additionally evaluates finite SM/per-SMSP admission queues against",
        "exact per-instruction L2 hit/miss outcomes without changing base replay.",
        "",
        "## Coverage",
        "",
        f"- {integrity['matrix_cells']} cells and {integrity['correctness_executions']} native correct executions.",
        f"- {integrity['native_timing_executions']} independent 32-MiB-sweep-preconditioned native CLOCK executions.",
        f"- The sweep is {integrity['l2_eviction_to_reported_capacity_ratio']:.2f}x NCU's reported {integrity['ncu_l2_bytes'] / (1024 * 1024):.0f}-MiB L2 capacity. It is a reproducible precondition, not direct proof that every target line was cold.",
        f"- {integrity['ncu_profiles']} NCU profiles and {integrity['source_counter_profiles']} SourceCounters profiles. Each metric profile reports {integrity['ncu_replay_passes_per_profile'][0]} application-replay passes ({integrity['ncu_application_replay_passes']:,} application executions total), so cross-metric values are not a joint single-launch observation.",
        f"- {integrity['nvbit_captures']} NVBit captures with {integrity['nvbit_records']:,} records.",
        f"- {integrity['simulator_replay_pairs']} ordinary and {integrity['candidate_replay_pairs']} frozen-candidate replay pairs.",
        f"- {integrity['admission_candidate_replay_pairs']} finite-admission candidate replay pairs with {integrity['admission_candidate_base_output_changes']} base-output changes.",
        f"- {integrity['feedback_candidate_replay_pairs']} timing-active single-candidate feedback replay pairs.",
        "- Every embedded candidate configuration matches its retained metadata snapshot; each schema-10 singleton also exactly matches the same-ID schema-9 candidate.",
        f"- {integrity['static_functions']} spill-free functions, {integrity['static_instructions']} decoded instructions, and zero unknown encodings.",
        f"- Long/trace static SASS dumps are identical, and {integrity['dynamic_static_joined_instructions']:,} per-capture decoded PC/opcode/operand entries join to the sidecar.",
        f"- Maximum GPC/SM-active CV: {result['ncu_gpc_cv_max_pct']:.3f}% / {result['ncu_sm_active_cv_max_pct']:.3f}%.",
        f"- Maximum native issue/fence-sequence CV: {result['native_issue_cv_max_pct']:.3f}% / {result['native_fence_cv_max_pct']:.3f}%.",
        f"- NCU repeat counts 3/5/7 cover {uncertainty['ncu_repeat_count_distribution']['3']}/{uncertainty['ncu_repeat_count_distribution']['5']}/{uncertainty['ncu_repeat_count_distribution']['7']} cells; {uncertainty['tail_control_repeat_count_mismatches']} tail/control rows use unequal counts. Restricting every cell to repeats 1-3 shifts the worst GPC delta by {uncertainty['ncu_gpc_balanced_three_shift_max_cycles']:,.0f} cycles ({uncertainty['ncu_gpc_balanced_three_shift_worst_cell']}).",
        f"- Observed repeat-range delta envelopes span at most {uncertainty['ncu_gpc_repeat_envelope_max_width_cycles']:,.0f} GPC, {uncertainty['native_issue_repeat_envelope_max_width_cycles']:,.0f} native issue, and {uncertainty['native_fence_repeat_envelope_max_width_cycles']:,.0f} native fence cycles; they cross zero in {uncertainty['ncu_gpc_repeat_envelope_zero_crossing_cells']}/{uncertainty['native_issue_repeat_envelope_zero_crossing_cells']}/{uncertainty['native_fence_repeat_envelope_zero_crossing_cells']} non-control cells. These are sensitivity bounds, not confidence intervals.",
        f"- Exact trace/NCU instruction-sector/read-barrier/L2-source topology mismatches: {len(result['topology_mismatches'])}; {len(result['lts_request_differences'])} STORE LTS request differences remain measured downstream write-combining behavior.",
        f"- {high_cv_above_five} cells retain a GPC or SM-active CV above 5% after their retained 3/5/7 repeats; the maximum remains {result['ncu_gpc_cv_max_pct']:.3f}% / {result['ncu_sm_active_cv_max_pct']:.3f}% and is reported as a phase response.",
        f"- NVBit module origin is unresolved in all {integrity['nvbit_unresolved_module_origins']} retained captures. The decoded PC/opcode/operand join does not contain encoded instruction bytes or prove captured control-bit identity.",
        "- Raw NCU/NVBit/native run records bind every cell to its exact command and arguments, but do not retain a content identity for the executed host binary. The later static dump therefore mitigates, but cannot close, capture-binary provenance.",
        "",
        "## Static and SourceCounters identification",
        "",
        "STG and RED carry read-dependence barrier 0. The subsequent address-register",
        "consumer/post-memory wait uses mask bit 0.",
        "Returning ATOMG instead carries write barrier 2 for its returned value.",
        f"Focused SourceCounters place {source_totals['stall_lg (Not Issued)']:,.0f} samples in `stall_lg`; the dominant {source_totals['stall_long_sb (Not Issued)']:,.0f}/{source_totals['stall_membar (Not Issued)']:,.0f} samples",
        "appear as long-scoreboard at the post-memory wait and MEMBAR at the compiler's",
        "ERRBAR following `MEMBAR.SC.GPU`. The final CLOCK also follows CCTL.IVALL,",
        "so this is a composite matched-control fence interval, not pure retirement.",
        f"Drain samples total {source_totals['stall_drain (Not Issued)']:,.0f} because the fence precedes EXIT.",
        "",
        "## Burst depth",
        "",
        "| tail | depth | native issue delta | native fence-sequence delta | NCU GPC delta | NCU SM-active delta | LG throttle delta | MEMBAR stall delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tail in ("store", "red", "atom"):
        for item in result["burst_depth"][tail]:
            lines.append(
                f"| {tail} | {item['depth']} | {item['native_issue_delta']:.1f} | "
                f"{item['native_fence_delta']:.1f} | {item['ncu_gpc_delta']:.1f} | "
                f"{item['ncu_sm_active_delta']:.1f} | {item['lg_throttle_delta']:.3f} | "
                f"{item['membar_delta']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The sweep-preconditioned native follow-up supersedes the original warm CLOCK",
            "labels. STORE/RED fence-sequence pressure is already material at depth 2",
            f"({burst[('store', 2)]['native_fence_delta']:,.0f}/{burst[('red', 2)]['native_fence_delta']:,.0f} cycles) and grows through depth 8 ({burst[('store', 8)]['native_fence_delta']:,.0f}/{burst[('red', 8)]['native_fence_delta']:,.0f}). STORE issue",
            f"becomes positive at depth 8 ({burst[('store', 8)]['native_issue_delta']:+,.0f}); RED issue remains near control until",
            f"depth 16 ({burst[('red', 16)]['native_issue_delta']:+,.0f}). The old depth-4 claim is not stable under sweep",
            "preconditioning and is consistent with L2 priming; it never identified",
            "queue capacity. ATOMG remains",
            "different: its returning-value dependency places almost all cost in issue.",
            "",
            "## Reuse and L2-tag discriminator",
            "",
            "At W8 depth 32, line-unique and line-reuse execute the same number of",
            "memory instructions and expose the same L1TEX sectors, ideal/output",
            "wavefronts, and LTS requests. Only inter-instruction address reuse changes:",
            "",
            "| tail | unique-line GPC | reused-line GPC | unique L2 miss sectors | reused L2 miss/hit sectors |",
            "|---|---:|---:|---:|---:|",
            *reuse_rows,
            "",
            "A dynamic-line, sector, wavefront, or LTS-request server cannot explain",
            "these ratios. The simulator's existing shared L2 predicts the exact cold",
            "versus tag-hit partition, making that split an essential candidate input.",
            f"The same-region STORE arm also collapses {same_region_store['sim_l2_hits'] + same_region_store['sim_l2_misses']:,.0f} dynamic line touches to",
            f"about {same_region_store['ncu_lts_requests_delta']:,.0f} observed LTS writes, exposing downstream write combining rather",
            "than a trace-topology error.",
            "",
            "## Spacing, SMSP placement, and CTA transfer",
            "",
            "Sixty-four dependent IMADs between memory instructions reduce STORE",
            f"issue+fence cost from {spacing[('store', 0)]['native_issue_delta'] + spacing[('store', 0)]['native_fence_delta']:,.0f} to {spacing[('store', 64)]['native_issue_delta'] + spacing[('store', 64)]['native_fence_delta']:,.0f} cycles and RED from {spacing[('red', 0)]['native_issue_delta'] + spacing[('red', 0)]['native_fence_delta']:,.0f} to {spacing[('red', 64)]['native_issue_delta'] + spacing[('red', 64)]['native_fence_delta']:,.0f};",
            f"ATOMG remains at {spacing[('atom', 64)]['native_issue_delta'] + spacing[('atom', 64)]['native_fence_delta']:,.0f} because each returned value is consumed. Gap IMADs",
            "also change the dependent STORE payload, so compression sensitivity is a",
            "retained confound rather than a timing-only intervention. Four",
            "selected warps spread one-per-SMSP versus two-per-two-SMSPs preserve total",
            f"work: STORE, RED, and ATOMG GPC deltas differ by only {sparse_differences['store']:,.0f}/{sparse_differences['red']:,.0f}/{sparse_differences['atom']:,.0f} cycles.",
            "The stabilized data does not identify an SMSP-local admission scope.",
            "",
            "At fixed 256 selected warp-operations per nominal SM, the observed 1×8,",
            f"2×4, and 4×2 medians span {cta_spans['store']:,.0f} STORE, {cta_spans['red']:,.0f} RED, and {cta_spans['atom']:,.0f} ATOMG GPC cycles.",
            "Several cells are phase-variable, so this is no detected large CTA effect,",
            "not an equivalence bound.",
            "",
            "## Frozen predecessor candidates",
            "",
            f"The {frozen['store']['configuration']['unique_line_cycles']}-cycle STORE-line and {frozen['red']['configuration']['unique_line_cycles']}-cycle RED-line candidates were frozen before",
            "this prospective corpus and replayed without refitting.",
            "",
            "| tail | candidate | prospective GPC MAE | prospective max error | raw simulator terminal MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for tail in ("store", "red"):
        item = result["frozen_candidate_transfer"][tail]
        lines.append(
            f"| {tail} | {item['candidate_id']} | {item['gpc']['mae']:.1f} | "
            f"{item['gpc']['max_error']:.1f} | {item['raw_sim_terminal_gpc']['mae']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The predecessor calibration had exact work-basis ties. Prospective topology",
            "scores for the tied candidates are:",
            "",
            "| tail | candidate | scope/work basis | GPC MAE | maximum error |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for tail, candidate_ids in (("store", (2, 4, 5)), ("red", (10, 12, 13, 15))):
        for candidate_id in candidate_ids:
            item = result["frozen_candidate_scores"][tail][str(candidate_id)]
            config = item["configuration"]
            if config["ideal_wavefront_cycles"] and config["unique_line_cycles"]:
                basis = "hybrid"
            elif config["ideal_wavefront_cycles"]:
                basis = "ideal-wavefront"
            else:
                basis = "unique-line"
            lines.append(
                f"| {tail} | {candidate_id} | {config['scope']} {basis} | "
                f"{item['gpc']['mae']:.1f} | {item['gpc']['max_error']:.1f} |"
            )
    lines.extend(
        [
            "",
            "STORE unique-line and hybrid remain near-tied prospectively; ideal-only is",
            "weaker. RED topology strongly favors unique-line over ideal/hybrid, but even",
            "the winning candidates fail unseen reuse: STORE overpredicts line-reuse by",
            f"{frozen['store']['gpc']['max_error']:,.0f} cycles and RED by {frozen['red']['gpc']['max_error']:,.0f}. Their good aggregate MAE came from a dominant",
            "cold-stream cohort, not a transferable completion mechanism. No rate advances.",
            "",
            "## Finite hit/miss-aware admission candidates",
            "",
            "Sixteen candidates declared before schema-9 replay cross STORE/RED, SM/per-SMSP queue scope,",
            "and capacities 4/8/16/32. Queue capacity counts waiting warp instructions;",
            "a slot is released when the shared FIFO downstream server starts service.",
            "Hit sectors cost one cycle and miss sectors retain the predecessor's",
            "10/16-cycle STORE/RED work rates. The evaluator preserves actual issue and",
            "cache order and is therefore diagnostic, not a counterfactual execution.",
            "The hardware corpus predates this candidate grid. The five shallow-burst",
            "cells select IDs; the other 20 cells per operation are strict selection",
            "holdouts, not prospectively unobserved measurements.",
            "",
            "| tail | selected candidate | scope | capacity | calibration issue MAE | held-out issue MAE | held-out max | cumulative-warp MAE | hit/miss terminal GPC MAE |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tail in ("store", "red"):
        item = selected_admission[tail]
        config = item["configuration"]
        lines.append(
            f"| {tail} | {item['candidate_id']} | {config['scope']} | "
            f"{config['queue_capacity_instructions']} | {item['issue_calibration']['mae']:.1f} | "
            f"{item['issue_held_out']['mae']:.1f} | {item['issue_held_out']['max_error']:.1f} | "
            f"{item['cumulative_warp_issue_all']['mae']:.1f} | "
            f"{item['terminal_gpc_all']['mae']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Shallow burst calibration selects STORE {selected_admission['store']['candidate_id']} ({selected_admission['store']['configuration']['scope']}, capacity {selected_admission['store']['configuration']['queue_capacity_instructions']}) and RED {selected_admission['red']['candidate_id']} ({selected_admission['red']['configuration']['scope']}, capacity {selected_admission['red']['configuration']['queue_capacity_instructions']}); RED ties {selected_admission['red']['calibration_tied_candidate_ids']}.",
            f"Maximum single-instruction admission delay transfers poorly at deep burst: held-out errors reach {selected_admission['store']['issue_held_out']['max_error']:,.0f} STORE and {selected_admission['red']['issue_held_out']['max_error']:,.0f} RED cycles.",
            f"Summing open-loop delays per warp is invalid: all-cell MAE expands to {selected_admission['store']['cumulative_warp_issue_all']['mae']:,.0f}/{selected_admission['red']['cumulative_warp_issue_all']['mae']:,.0f} cycles because delayed demand was not rescheduled.",
            f"The exact hit/miss service frontier is materially better than the frozen line-only completion model ({selected_admission['store']['terminal_gpc_all']['mae']:,.1f} STORE and {selected_admission['red']['terminal_gpc_all']['mae']:,.1f} RED GPC MAE), but it is identical across queue capacities and cannot identify admission scope.",
            "No queue capacity, scope, or timing rule is promoted.",
            "",
            "## Feedback-coupled replay",
            "",
            "Schema 10 runs one frozen candidate at a time. Candidate issue reserves",
            "queue space and yields the SM; ordered L2 accesses price service before the",
            "next SM wake. A full queue becomes an actual readiness constraint, shifting",
            "later issue and allowing the executable L2 order to change.",
            "",
            "The issue columns below compare diagnostic last issue to diagnostic last",
            "issue. CLOCK-interval scores are reported separately after the table.",
            "",
            "| tail | candidate | scope/capacity | calibration baseline issue MAE | calibration feedback issue MAE | held-out baseline issue MAE | held-out feedback issue MAE | held-out baseline terminal MAE | held-out feedback terminal MAE |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tail, candidate_ids in (("store", (7,)), ("red", (12, 14))):
        for candidate_id in candidate_ids:
            item = feedback_scores[tail][str(candidate_id)]
            config = item["configuration"]
            lines.append(
                f"| {tail} | {candidate_id} | {config['scope']}/{config['queue_capacity_instructions']} | "
                f"{feedback_baselines[tail]['last_issue_calibration']['mae']:.1f} | "
                f"{item['last_issue_calibration']['mae']:.1f} | "
                f"{feedback_baselines[tail]['last_issue_held_out']['mae']:.1f} | "
                f"{item['last_issue_held_out']['mae']:.1f} | "
                f"{feedback_baselines[tail]['terminal_gpc_held_out']['mae']:.1f} | "
                f"{item['terminal_gpc_held_out']['mae']:.1f} |"
            )
    lines.extend(
        [
            "",
            "On the strict 20-cell selection holdout, feedback closes most of the",
            f"missing last-issue magnitude: STORE falls from {feedback_baselines['store']['last_issue_held_out']['mae']:,.1f} to {feedback_scores['store']['7']['last_issue_held_out']['mae']:,.1f}, while RED falls from {feedback_baselines['red']['last_issue_held_out']['mae']:,.1f} to {feedback_scores['red']['12']['last_issue_held_out']['mae']:,.1f}/{feedback_scores['red']['14']['last_issue_held_out']['mae']:,.1f}.",
            f"The separate held-out feedback CLOCK MAEs are {feedback_scores['store']['7']['clock_issue_held_out']['mae']:,.1f} STORE and {feedback_scores['red']['12']['clock_issue_held_out']['mae']:,.1f}/{feedback_scores['red']['14']['clock_issue_held_out']['mae']:,.1f} RED.",
            f"Held-out terminal GPC MAE falls from {feedback_baselines['store']['terminal_gpc_held_out']['mae']:,.1f} to {feedback_scores['store']['7']['terminal_gpc_held_out']['mae']:,.1f} STORE and from {feedback_baselines['red']['terminal_gpc_held_out']['mae']:,.1f} to {feedback_scores['red']['12']['terminal_gpc_held_out']['mae']:,.1f}/{feedback_scores['red']['14']['terminal_gpc_held_out']['mae']:,.1f} RED. Deep burst remains underpredicted by roughly 23–28k issue cycles, so the frozen rates/capacities still do not transfer.",
            f"Using only the common first three NCU repeats preserves that held-out terminal result: STORE {feedback_baselines['store']['terminal_gpc_held_out_balanced_three']['mae']:,.1f} to {feedback_scores['store']['7']['terminal_gpc_held_out_balanced_three']['mae']:,.1f}, RED {feedback_baselines['red']['terminal_gpc_held_out_balanced_three']['mae']:,.1f} to {feedback_scores['red']['12']['terminal_gpc_held_out_balanced_three']['mae']:,.1f}/{feedback_scores['red']['14']['terminal_gpc_held_out_balanced_three']['mae']:,.1f}.",
            f"Even the observed min/max repeat envelopes leave the held-out MAE bands separated. Issue is STORE {feedback_baselines['store']['last_issue_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_baselines['store']['last_issue_held_out_repeat_envelope']['mae_upper_bound']:,.1f} baseline versus {feedback_scores['store']['7']['last_issue_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_scores['store']['7']['last_issue_held_out_repeat_envelope']['mae_upper_bound']:,.1f} feedback, and RED {feedback_baselines['red']['last_issue_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_baselines['red']['last_issue_held_out_repeat_envelope']['mae_upper_bound']:,.1f} versus {feedback_scores['red']['12']['last_issue_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_scores['red']['12']['last_issue_held_out_repeat_envelope']['mae_upper_bound']:,.1f}/{feedback_scores['red']['14']['last_issue_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_scores['red']['14']['last_issue_held_out_repeat_envelope']['mae_upper_bound']:,.1f}.",
            f"Terminal is STORE {feedback_baselines['store']['terminal_gpc_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_baselines['store']['terminal_gpc_held_out_repeat_envelope']['mae_upper_bound']:,.1f} versus {feedback_scores['store']['7']['terminal_gpc_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_scores['store']['7']['terminal_gpc_held_out_repeat_envelope']['mae_upper_bound']:,.1f}, and RED {feedback_baselines['red']['terminal_gpc_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_baselines['red']['terminal_gpc_held_out_repeat_envelope']['mae_upper_bound']:,.1f} versus {feedback_scores['red']['12']['terminal_gpc_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_scores['red']['12']['terminal_gpc_held_out_repeat_envelope']['mae_upper_bound']:,.1f}/{feedback_scores['red']['14']['terminal_gpc_held_out_repeat_envelope']['mae_lower_bound']:,.1f}–{feedback_scores['red']['14']['terminal_gpc_held_out_repeat_envelope']['mae_upper_bound']:,.1f}; these remain sensitivity bounds, not confidence intervals.",
            "",
            "Cache attribution must distinguish the candidate-operation observer from",
            "the aggregate Store-source lane:",
            "",
            "| tail | candidate | aggregate Store-source changed cells | candidate observer changed cells | observer/source mismatches | target-operation conclusion |",
            "|---|---:|---:|---:|---:|---|",
            f"| store | 7 | {feedback_scores['store']['7']['aggregate_store_source_changed_cells']} | {feedback_scores['store']['7']['candidate_observer_changed_cells']} | {feedback_scores['store']['7']['candidate_observer_vs_aggregate_mismatches']} | unresolved: target and instrumentation are both STORE |",
            f"| red | 12 | {feedback_scores['red']['12']['aggregate_store_source_changed_cells']} | {feedback_scores['red']['12']['candidate_observer_changed_cells']} | {feedback_scores['red']['12']['candidate_observer_vs_aggregate_mismatches']} | {feedback_scores['red']['12']['target_operation_changed_cells']} RED target cells changed |",
            f"| red | 14 | {feedback_scores['red']['14']['aggregate_store_source_changed_cells']} | {feedback_scores['red']['14']['candidate_observer_changed_cells']} | {feedback_scores['red']['14']['candidate_observer_vs_aggregate_mismatches']} | {feedback_scores['red']['14']['target_operation_changed_cells']} RED target cells changed |",
            "",
            f"For STORE, the candidate observer changes in {feedback_scores['store']['7']['candidate_observer_changed_cells']} cells, while {feedback_scores['store']['7']['candidate_observer_negative_hit_cells']}/{feedback_scores['store']['7']['candidate_observer_negative_miss_cells']} cells have negative hit/miss deltas after control subtraction. Negative counts cannot be physical target-only outcomes. For RED, the operation-specific observer changes in zero cells while the aggregate lane changes in {feedback_scores['red']['12']['aggregate_store_source_changed_cells']}/{feedback_scores['red']['14']['aggregate_store_source_changed_cells']} cells because later instrumentation stores share that lane.",
            "The retained data therefore show that feedback changes downstream",
            "instrumentation-store cache order, but do not identify a target STORE",
            "cache-outcome change and show no target RED cache-outcome change.",
            "",
            "## Decision and next component",
            "",
            "- Promote the schema-10 single-candidate feedback mechanism as an opt-in",
            "  research lane; within replay, activating feedback improves strict-holdout",
            "  issue and terminal predictions.",
            "- Do not promote candidates 7, 12, or 14 as architecture defaults: deep",
            "  burst, CTA transfer, and spacing residuals remain material.",
            "- Retain schema 9 for timing-neutral differential checks and schema 8 for",
            "  operation-specific compiler-barrier attribution.",
            "- Keep STORE/RED completion typed incomplete and GPC emission unavailable.",
            "",
            "The next bounded discriminator holds the feedback mechanism fixed and",
            "crosses downstream miss rate/service width using shallow depths for",
            "calibration and deep/reuse/spacing/CTA cells as holdouts. Any rate must",
            "improve both issue and terminal transfer without relying on aggregate",
            "instrumentation-store cache changes or an operation-specific intercept.",
            "",
            "Full spacing, topology, sparse-SMSP, CTA-transfer, stall, native CLOCK,",
            "and simulator rows are in `cells.csv`; machine-readable reductions are in",
            "`summary.json`.",
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
