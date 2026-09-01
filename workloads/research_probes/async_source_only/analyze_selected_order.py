#!/usr/bin/env python3
"""Reduce the SM86 selected asynchronous-copy instrumentation-order campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)

DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_selected_order_trace_20260831_v1"
)
PREDECESSOR_ROOT = (
    REPO_ROOT
    / "results/captures/sm86_async_cross_warp_queue_overlap_20260831_v1"
)
DEFAULT_AGGREGATE = (
    REPO_ROOT / "data/local_labels/sm86_async_selected_order_trace_report.json"
)
BINARY = HERE / "async_cross_warp_queue_overlap"
CONTROL = PREDECESSOR_ROOT / "static/control_bits.json"
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ROUTES = ("l1", "l2")
ARMS = ("queued", "drained")
REPETITIONS = (1, 2)
GROUPS = 64
EXACT_NCU = {
    "smsp__inst_executed_op_ldgsts.sum": 2048.0,
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum": 32768.0,
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": 8192.0,
}
ORDER_DIAGNOSTIC_FIELDS = (
    "selected_events",
    "issue_events",
    "commit_events",
    "wait_events",
    "issue_events_with_other_warp_committed_groups",
    "other_warp_committed_groups_sum",
    "maximum_other_warp_committed_groups",
    "protocol_contradictions",
)


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OrderedEvent:
    token: int
    kind: str
    warp_id: int
    sm_id: int
    function_unique_id: int
    vpc: int
    active_mask: int
    predicate_mask: int


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisError(f"{path}: cannot read strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected one JSON object")
    return value


def require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{context}: expected object")
    return value


def exact_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{context}: expected integer")
    return value


def finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{context}: expected number")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{context}: expected finite number")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AnalysisError(f"{path}: cannot hash: {error}") from error
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise AnalysisError(f"{path}: cannot resolve artifact: {error}") from error
    if not resolved.is_file() or stat.st_size <= 0:
        raise AnalysisError(f"{resolved}: missing or empty artifact")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": stat.st_size,
    }


def protobuf_closure(trace: Path) -> dict[str, str]:
    closure = {"dynamic_trace.pb": sha256(trace / "dynamic_trace.pb")}
    payloads = sorted((trace / "threadblocks").rglob("*.pb"))
    if not payloads:
        raise AnalysisError(f"{trace}: no threadblock protobuf payloads")
    for path in payloads:
        closure[str(path.relative_to(trace))] = sha256(path)
    return closure


def stage_cells(stage: str) -> Iterable[tuple[int, str, str, int]]:
    if stage == "w4":
        for repetition in REPETITIONS:
            for route in ROUTES:
                for arm in ARMS:
                    yield repetition, route, arm, 4
        return
    if stage == "holdout":
        for repetition in REPETITIONS:
            for arm in ARMS:
                yield repetition, "l2", arm, 8
        return
    raise AnalysisError(f"unknown selected-order stage {stage!r}")


def cell_key(route: str, arm: str, active_warps: int) -> str:
    return f"{route}_{arm}_w{active_warps}_g64"


def cell_directory(base: Path, stage: str, repetition: int, key: str) -> Path:
    return base / stage / f"repeat-{repetition}" / key


def expected_counts(active_warps: int) -> dict[str, int]:
    issue = GROUPS * active_warps * 8
    commit = GROUPS * ((active_warps - 1) * 4 + 1)
    wait = GROUPS * active_warps
    return {
        "issue": issue,
        "commit": commit,
        "wait": wait,
        "selected": issue + commit + wait,
    }


def validate_reused_evidence() -> dict[str, Any]:
    static = read_json(PREDECESSOR_ROOT / "static/summary.json")
    retained_binary = require_object(static.get("binary"), "retained binary")
    current_binary = file_identity(BINARY)
    if current_binary["sha256"] != retained_binary.get("sha256"):
        raise AnalysisError("current queue-overlap binary differs from retained exact-work evidence")
    control = read_json(CONTROL)
    control_summary = require_object(control.get("summary"), "control summary")
    if control.get("arch") != "sm_86" or control_summary.get("total_unknown_encoding") != 0:
        raise AnalysisError("retained control sidecar is not a complete sm_86 decode")

    ncu: dict[str, Any] = {}
    for route in ROUTES:
        route_values: dict[str, Mapping[str, Any]] = {}
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            summary = read_json(PREDECESSOR_ROOT / "ncu" / key / "summary.json")
            if (
                summary.get("route") != route
                or summary.get("arm") != arm
                or summary.get("binary", {}).get("sha256") != current_binary["sha256"]
            ):
                raise AnalysisError(f"{key}: retained NCU identity differs")
            metrics = require_object(summary.get("metrics"), f"{key}: NCU metrics")
            for metric, expected in EXACT_NCU.items():
                if finite_number(metrics.get(metric), f"{key}: {metric}") != expected:
                    raise AnalysisError(f"{key}: retained exact NCU work differs")
            route_values[arm] = metrics
            ncu[key] = {
                "report_sha256": summary.get("report", {}).get("sha256"),
                "exact_work": {metric: metrics[metric] for metric in EXACT_NCU},
            }
        if dict(route_values["queued"]) != dict(route_values["drained"]):
            raise AnalysisError(f"{route}: retained queued/drained NCU maps differ")
    return {
        "passed": True,
        "binary": current_binary,
        "source": static.get("source"),
        "control": file_identity(CONTROL),
        "static_summary": file_identity(PREDECESSOR_ROOT / "static/summary.json"),
        "decoded_instructions": control_summary.get("total_decoded"),
        "reused_ncu_reports": ncu,
    }


def last_target_result(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AnalysisError(f"{path}: cannot read target output: {error}") from error
    for line in reversed(lines):
        try:
            value = json.loads(
                line, parse_constant=_reject_constant, object_pairs_hook=_unique_object
            )
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict) and value.get("workload") == "async_cross_warp_queue_overlap":
            return value
    raise AnalysisError(f"{path}: no queue-overlap target result")


def validate_capture_metadata(
    trace: Path,
    *,
    stage: str,
    repetition: int,
    route: str,
    arm: str,
    active_warps: int,
    binary_sha: str,
) -> dict[str, Any]:
    key = cell_key(route, arm, active_warps)
    complete = read_json(trace / "complete.json")
    binding = read_json(trace / "campaign_binding.json")
    environment = read_json(trace / "environment.json")
    if complete.get("complete") is not True:
        raise AnalysisError(f"{trace}: capture is incomplete")
    expected_cell = f"selected_order_{stage}_r{repetition}_{key}"
    if binding.get("cell") != expected_cell or complete.get("key") != expected_cell:
        raise AnalysisError(f"{trace}: capture cell identity differs")
    if (
        binding.get("binary", {}).get("sha256") != binary_sha
        or complete.get("binary_sha256") != binary_sha
        or binding.get("async_order_enabled") is not True
        or binding.get("repetition") != repetition
        or binding.get("mode") != "ncu"
    ):
        raise AnalysisError(f"{trace}: order-enabled capture binding differs")
    logical = require_object(environment.get("environment"), f"{trace}: environment")
    effective = require_object(
        environment.get("effective_environment"), f"{trace}: effective environment"
    )
    if (
        environment.get("async_order_enabled") is not True
        or logical.get("SIMBENCH_NVBIT_ASYNC_ORDER") != "1"
        or effective.get("SIMBENCH_NVBIT_ASYNC_ORDER") != "1"
    ):
        raise AnalysisError(f"{trace}: order environment is absent")

    expected_args = [
        str(BINARY.resolve()),
        "--groups",
        "64",
        "--active-warps",
        str(active_warps),
        "--route",
        route,
        "--mode",
        "endpoint",
        "--arm",
        arm,
        "--markers",
        "on",
    ]
    argv = binding.get("argv")
    if not isinstance(argv, list) or [str(item) for item in argv] != expected_args:
        raise AnalysisError(f"{trace}: target argv differs")
    result = last_target_result(trace / "stdout.txt")
    expected_result = {
        "workload": "async_cross_warp_queue_overlap",
        "mode": "endpoint",
        "route": route,
        "arm": arm,
        "active_warps": active_warps,
        "groups": GROUPS,
        "copies_per_group": 8,
        "background_groups_per_epoch": 4,
        "correct": True,
        "result_store_after_intervals": True,
    }
    for field, expected in expected_result.items():
        if result.get(field) != expected:
            raise AnalysisError(f"{trace}: target {field} differs")
    try:
        with (trace / "stats.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"{trace}: cannot read stats.csv: {error}") from error
    if len(rows) != 1:
        raise AnalysisError(f"{trace}: expected one stats row")
    row = {str(key).strip(): str(value).strip() for key, value in rows[0].items()}
    total = exact_integer(int(row["total_insts"]), f"{trace}: total instructions")
    reported = exact_integer(
        int(row["total_reported_insts"]), f"{trace}: reported instructions"
    )
    if total <= 0 or total != reported:
        raise AnalysisError(f"{trace}: dynamic instruction stream is incomplete")
    if "queue_overlap_endpoint_kernel" not in row.get("kernel mangled name", ""):
        raise AnalysisError(f"{trace}: wrong traced kernel")
    for reason in (
        "unsupported_memory_path",
        "unmapped_memory_range",
        "cross_allocation_lane_span",
    ):
        path = trace / "extra_info" / f"{reason}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnalysisError(f"{path}: cannot read exclusion sidecar: {error}") from error
        if value != []:
            raise AnalysisError(f"{path}: exclusion sidecar is nonempty")
    return {
        "instructions": total,
        "trace_sha256": sha256(trace / "dynamic_trace.pb"),
        "protobuf_closure": protobuf_closure(trace),
        "tracer_sha256": binding.get("tracer", {}).get("sha256"),
    }


def classify_static_instruction(instruction: Mapping[str, Any]) -> str | None:
    opcode = instruction.get("op_code")
    if not isinstance(opcode, str):
        raise AnalysisError("enhanced instruction has no opcode")
    operands_raw = instruction.get("operands")
    if not isinstance(operands_raw, list):
        raise AnalysisError("enhanced instruction has no operand list")
    operands = []
    for operand in operands_raw:
        if not isinstance(operand, Mapping) or not isinstance(
            operand.get("operand_string"), str
        ):
            raise AnalysisError("enhanced instruction has malformed operand")
        operands.append(str(operand["operand_string"]))
    if opcode.startswith("LDGSTS"):
        return "issue"
    if opcode == "LDGDEPBAR":
        return "commit"
    if opcode == "DEPBAR.LE" and operands and operands[0] == "SB0":
        if len(operands) != 2 or operands[1] != "0x0":
            raise AnalysisError(
                "queue-overlap selected-order assay contains a nonzero SB0 wait"
            )
        return "wait"
    return None


def static_selected_map(trace: Path) -> dict[tuple[int, int], str]:
    value = read_json(trace / "extra_info/enhanced_execution_info.json")
    kernels = value.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise AnalysisError(f"{trace}: expected one enhanced kernel")
    selected: dict[tuple[int, int], str] = {}
    for kernel in kernels:
        if not isinstance(kernel, Mapping):
            raise AnalysisError(f"{trace}: malformed enhanced kernel")
        function_id = exact_integer(
            kernel.get("unique_function_id"), f"{trace}: function id"
        )
        instructions = kernel.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            raise AnalysisError(f"{trace}: enhanced kernel has no instructions")
        for instruction in instructions:
            if not isinstance(instruction, Mapping):
                raise AnalysisError(f"{trace}: malformed enhanced instruction")
            kind = classify_static_instruction(instruction)
            if kind is None:
                continue
            pc = exact_integer(instruction.get("pc_num_dec"), f"{trace}: static PC")
            key = (function_id, pc)
            if key in selected:
                raise AnalysisError(f"{trace}: duplicate selected static PC {key}")
            selected[key] = kind
    if set(selected.values()) != {"issue", "commit", "wait"}:
        raise AnalysisError(f"{trace}: static map lacks a selected event class")
    return selected


def parse_order_events(trace: Path) -> list[OrderedEvent]:
    bindings = load_dynamic_trace_bindings()
    descriptor = bindings.instruction_pb2.instruction.DESCRIPTOR
    if "instrumentation_order" not in descriptor.fields_by_name:
        raise AnalysisError("tracked protobuf binding has no instrumentation_order field")
    selected = static_selected_map(trace)
    payloads = sorted((trace / "threadblocks").rglob("*.pb"))
    if len(payloads) != 1:
        raise AnalysisError(f"{trace}: expected one threadblock payload, got {len(payloads)}")
    threadblock = bindings.threadblock_pb2.threadblock()
    try:
        threadblock.ParseFromString(payloads[0].read_bytes())
    except (OSError, ValueError) as error:
        raise AnalysisError(f"{payloads[0]}: cannot parse protobuf: {error}") from error
    block_id = threadblock.block_id
    if (int(block_id.x), int(block_id.y), int(block_id.z)) != (0, 0, 0):
        raise AnalysisError(f"{trace}: unexpected block identity")

    events: list[OrderedEvent] = []
    for warp_id in sorted(threadblock.warps.keys()):
        warp = threadblock.warps[warp_id]
        prior_token = 0
        for instruction in warp.instructions:
            key = (int(instruction.function_unique_id), int(instruction.vpc))
            kind = selected.get(key)
            ordered = instruction.HasField("instrumentation_order")
            if ordered != (kind is not None):
                raise AnalysisError(
                    f"{trace}: instrumentation-order/static selection mismatch "
                    f"at warp={warp_id} function={key[0]} vpc={key[1]}"
                )
            if kind is None:
                continue
            for field in (
                "active_mask",
                "predicate_mask",
                "warpid_tb",
                "warpid_sm",
                "sm_id",
                "vpc",
            ):
                if not instruction.HasField(field):
                    raise AnalysisError(f"{trace}: selected instruction lacks {field}")
            token = int(instruction.instrumentation_order)
            if token <= prior_token:
                raise AnalysisError(f"{trace}: warp {warp_id} tokens are not increasing")
            prior_token = token
            if int(instruction.warpid_tb) != int(warp_id):
                raise AnalysisError(f"{trace}: warp map/instruction identity differs")
            active_mask = int(instruction.active_mask)
            predicate_mask = int(instruction.predicate_mask)
            if active_mask != 0xFFFFFFFF or predicate_mask != 0xFFFFFFFF:
                raise AnalysisError(f"{trace}: selected instruction is not full-warp active")
            events.append(
                OrderedEvent(
                    token=token,
                    kind=kind,
                    warp_id=int(warp_id),
                    sm_id=int(instruction.sm_id),
                    function_unique_id=key[0],
                    vpc=key[1],
                    active_mask=active_mask,
                    predicate_mask=predicate_mask,
                )
            )
    return events


def chunks(values: Sequence[OrderedEvent], width: int) -> list[Sequence[OrderedEvent]]:
    if len(values) % width:
        raise AnalysisError(f"event stream length {len(values)} is not divisible by {width}")
    return [values[index : index + width] for index in range(0, len(values), width)]


def expected_order_diagnostic(events: Sequence[OrderedEvent]) -> dict[str, int]:
    open_group: dict[int, bool] = {}
    committed: dict[int, int] = {}
    issue_with_other = 0
    population_sum = 0
    maximum_population = 0
    contradictions = 0
    counts = {"issue": 0, "commit": 0, "wait": 0}
    for event in sorted(events, key=lambda item: item.token):
        counts[event.kind] += 1
        open_group.setdefault(event.warp_id, False)
        committed.setdefault(event.warp_id, 0)
        if event.kind == "issue":
            other = sum(
                population
                for warp_id, population in committed.items()
                if warp_id != event.warp_id
            )
            if other:
                issue_with_other += 1
            population_sum += other
            maximum_population = max(maximum_population, other)
            open_group[event.warp_id] = True
        elif event.kind == "commit":
            if not open_group[event.warp_id]:
                contradictions += 1
            else:
                committed[event.warp_id] += 1
                open_group[event.warp_id] = False
        else:
            if open_group[event.warp_id]:
                contradictions += 1
            committed[event.warp_id] = 0
    contradictions += sum(open_group.values()) + sum(committed.values())
    return {
        "selected_events": len(events),
        "issue_events": counts["issue"],
        "commit_events": counts["commit"],
        "wait_events": counts["wait"],
        "issue_events_with_other_warp_committed_groups": issue_with_other,
        "other_warp_committed_groups_sum": population_sum,
        "maximum_other_warp_committed_groups": maximum_population,
        "protocol_contradictions": contradictions,
    }


def validate_event_relations(
    events: Sequence[OrderedEvent], *, arm: str, active_warps: int
) -> dict[str, Any]:
    counts = expected_counts(active_warps)
    observed_counts = {
        "issue": sum(event.kind == "issue" for event in events),
        "commit": sum(event.kind == "commit" for event in events),
        "wait": sum(event.kind == "wait" for event in events),
        "selected": len(events),
    }
    if observed_counts != counts:
        raise AnalysisError(f"selected event counts {observed_counts} != {counts}")
    tokens = sorted(event.token for event in events)
    if tokens != list(range(1, counts["selected"] + 1)):
        raise AnalysisError("instrumentation-order tokens are not exactly dense from one")
    sm_ids = {event.sm_id for event in events}
    if len(sm_ids) != 1:
        raise AnalysisError(f"selected events span SM ids {sorted(sm_ids)}")
    by_warp: dict[int, list[OrderedEvent]] = {}
    for event in events:
        by_warp.setdefault(event.warp_id, []).append(event)
    if set(by_warp) != set(range(active_warps)):
        raise AnalysisError(
            f"selected warp ids {sorted(by_warp)} != {list(range(active_warps))}"
        )

    background_pattern = ("issue", "issue", "commit") * 4 + ("wait",)
    target_pattern = ("issue",) * 8 + ("commit", "wait")
    background_epochs: dict[int, list[Sequence[OrderedEvent]]] = {}
    for warp_id in range(active_warps - 1):
        epochs = chunks(by_warp[warp_id], len(background_pattern))
        if len(epochs) != GROUPS:
            raise AnalysisError(f"background warp {warp_id} epoch count differs")
        if any(tuple(event.kind for event in epoch) != background_pattern for epoch in epochs):
            raise AnalysisError(f"background warp {warp_id} selected pattern differs")
        background_epochs[warp_id] = epochs
    target = active_warps - 1
    target_epochs = chunks(by_warp[target], len(target_pattern))
    if len(target_epochs) != GROUPS or any(
        tuple(event.kind for event in epoch) != target_pattern
        for epoch in target_epochs
    ):
        raise AnalysisError("target selected pattern differs")

    commit_before_target = 0
    queued_signatures = 0
    drained_signatures = 0
    expected_background_commits = (active_warps - 1) * 4
    for epoch_index in range(GROUPS):
        target_first_issue = target_epochs[epoch_index][0].token
        target_wait = target_epochs[epoch_index][-1].token
        background_commits = [
            event.token
            for warp_id in range(active_warps - 1)
            for event in background_epochs[warp_id][epoch_index]
            if event.kind == "commit"
        ]
        background_waits = [
            background_epochs[warp_id][epoch_index][-1].token
            for warp_id in range(active_warps - 1)
        ]
        if len(background_commits) != expected_background_commits:
            raise AnalysisError(f"epoch {epoch_index}: background commit count differs")
        commits_before = all(token < target_first_issue for token in background_commits)
        target_before_waits = all(target_wait < token for token in background_waits)
        waits_before_target = all(token < target_first_issue for token in background_waits)
        commit_before_target += commits_before
        queued_signatures += commits_before and target_before_waits and not waits_before_target
        drained_signatures += commits_before and waits_before_target and not target_before_waits
    expected_queued = GROUPS if arm == "queued" else 0
    expected_drained = GROUPS if arm == "drained" else 0
    if (
        commit_before_target != GROUPS
        or queued_signatures != expected_queued
        or drained_signatures != expected_drained
    ):
        raise AnalysisError(
            f"{arm}: causal classifier differs: commits={commit_before_target}, "
            f"queued={queued_signatures}, drained={drained_signatures}"
        )
    diagnostic = expected_order_diagnostic(events)
    if diagnostic["protocol_contradictions"] != 0:
        raise AnalysisError("selected event stream contains an async protocol contradiction")
    return {
        "counts": counts,
        "dense_tokens": True,
        "single_sm": next(iter(sm_ids)),
        "per_warp_program_order": True,
        "background_commits_per_epoch": expected_background_commits,
        "commit_before_target_epochs": commit_before_target,
        "queued_classified_epochs": queued_signatures,
        "drained_classified_epochs": drained_signatures,
        "classified_epochs": queued_signatures + drained_signatures,
        "expected_replay_diagnostic": diagnostic,
    }


def readiness(value: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    launches = value.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{context}: expected one replay launch")
    launch = require_object(launches[0], f"{context}: launch")
    timing = require_object(launch.get("timing"), f"{context}: timing")
    return require_object(timing.get("readiness"), f"{context}: readiness")


def normalized_order_replay(value: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    target = readiness(normalized, context)
    if not isinstance(target, dict):
        raise AnalysisError(f"{context}: mutable readiness projection unavailable")
    if target.get("schema_version") != 21:
        raise AnalysisError(f"{context}: ordered readiness schema is not 21")
    if not isinstance(target.pop("async_instrumentation_order", None), dict):
        raise AnalysisError(f"{context}: ordered diagnostic is absent")
    target["schema_version"] = 20
    return normalized


def validate_replay(
    root: Path,
    *,
    stage: str,
    repetition: int,
    route: str,
    arm: str,
    active_warps: int,
    binary_sha: str,
    expected_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    key = cell_key(route, arm, active_warps)
    trace = cell_directory(root / "nvbit", stage, repetition, key)
    stripped = cell_directory(root / "order-stripped", stage, repetition, key)
    directory = cell_directory(root / "simulator", stage, repetition, key)
    command = read_json(directory / "command.json")
    if (
        command.get("stage") != stage
        or command.get("repetition") != repetition
        or command.get("route") != route
        or command.get("arm") != arm
        or command.get("active_warps") != active_warps
        or command.get("groups") != GROUPS
        or command.get("binary", {}).get("sha256") != binary_sha
        or command.get("sim_run", {}).get("sha256") != sha256(SIM_RUN)
        or command.get("trace_protobuf_closure") != protobuf_closure(trace)
        or command.get("stripped_protobuf_closure") != protobuf_closure(stripped)
    ):
        raise AnalysisError(f"{directory}: replay binding differs")
    stripped_record = read_json(stripped / "order_stripped.json")
    if (
        stripped_record.get("source_protobuf_closure") != protobuf_closure(trace)
        or stripped_record.get("derived_protobuf_closure") != protobuf_closure(stripped)
        or stripped_record.get("cleared_instrumentation_order_fields")
        != expected_counts(active_warps)["selected"]
    ):
        raise AnalysisError(f"{stripped}: stripped-trace binding differs")

    first_path = directory / "replay-1.json"
    second_path = directory / "replay-2.json"
    baseline_path = directory / "stripped.json"
    first_sha = sha256(first_path)
    if first_sha != sha256(second_path):
        raise AnalysisError(f"{directory}: replay twins differ")
    ordered = read_json(first_path)
    baseline = read_json(baseline_path)
    ordered_readiness = readiness(ordered, f"{key}: ordered")
    if ordered_readiness.get("schema_version") != 21:
        raise AnalysisError(f"{key}: ordered readiness schema is not 21")
    observed = require_object(
        ordered_readiness.get("async_instrumentation_order"),
        f"{key}: instrumentation order",
    )
    if set(observed) != set(ORDER_DIAGNOSTIC_FIELDS):
        raise AnalysisError(f"{key}: order diagnostic fields differ")
    if dict(observed) != dict(expected_diagnostic):
        raise AnalysisError(
            f"{key}: replay diagnostic {dict(observed)} != {dict(expected_diagnostic)}"
        )
    baseline_readiness = readiness(baseline, f"{key}: stripped")
    if baseline_readiness.get("schema_version") != 20 or (
        "async_instrumentation_order" in baseline_readiness
    ):
        raise AnalysisError(f"{key}: stripped replay is not schema-20 observation-free")
    if normalized_order_replay(ordered, key) != baseline:
        raise AnalysisError(f"{key}: order observation changed normalized replay output")
    return {
        "ordered_replay_sha256": first_sha,
        "stripped_replay_sha256": sha256(baseline_path),
        "twins_deterministic": True,
        "normalized_output_equal": True,
        "diagnostic": dict(observed),
    }


def validate_stage(
    root: Path, stage: str, reused: Mapping[str, Any]
) -> dict[str, Any]:
    binary = require_object(reused.get("binary"), "reused binary")
    binary_sha = str(binary.get("sha256"))
    cells: dict[str, Any] = {}
    for repetition, route, arm, active_warps in stage_cells(stage):
        key = cell_key(route, arm, active_warps)
        qualified = f"repeat-{repetition}/{key}"
        trace = cell_directory(root / "nvbit", stage, repetition, key)
        metadata = validate_capture_metadata(
            trace,
            stage=stage,
            repetition=repetition,
            route=route,
            arm=arm,
            active_warps=active_warps,
            binary_sha=binary_sha,
        )
        events = parse_order_events(trace)
        relations = validate_event_relations(
            events, arm=arm, active_warps=active_warps
        )
        replay = validate_replay(
            root,
            stage=stage,
            repetition=repetition,
            route=route,
            arm=arm,
            active_warps=active_warps,
            binary_sha=binary_sha,
            expected_diagnostic=relations["expected_replay_diagnostic"],
        )
        cells[qualified] = {
            "repetition": repetition,
            "route": route,
            "arm": arm,
            "active_warps": active_warps,
            "groups": GROUPS,
            "capture": metadata,
            "order": relations,
            "replay": replay,
        }
    expected_cells = 8 if stage == "w4" else 4
    if len(cells) != expected_cells:
        raise AnalysisError(f"{stage}: validated cell count differs")
    return {
        "passed": True,
        "stage": stage,
        "cell_count": len(cells),
        "all_epochs_classified": True,
        "all_replays_deterministic": True,
        "all_normalized_outputs_equal": True,
        "cells": cells,
    }


def validate_default_compatibility(root: Path) -> dict[str, Any]:
    ordered = root / "compat/one_warp_ordered"
    default = root / "compat/one_warp_default"
    ordered_binding = read_json(ordered / "campaign_binding.json")
    default_binding = read_json(default / "campaign_binding.json")
    if ordered_binding.get("async_order_enabled") is not True:
        raise AnalysisError("one-warp ordered canary did not enable tag 14")
    if default_binding.get("async_order_enabled") is not False:
        raise AnalysisError("one-warp default canary unexpectedly enabled tag 14")
    bindings = load_dynamic_trace_bindings()
    ordered_payloads = sorted((ordered / "threadblocks").rglob("*.pb"))
    default_payloads = sorted((default / "threadblocks").rglob("*.pb"))
    if len(ordered_payloads) != 1 or len(default_payloads) != 1:
        raise AnalysisError("one-warp compatibility canary payload count differs")
    ordered_threadblock = bindings.threadblock_pb2.threadblock()
    ordered_threadblock.ParseFromString(ordered_payloads[0].read_bytes())
    ordered_tokens = sorted(
        int(instruction.instrumentation_order)
        for warp in ordered_threadblock.warps.values()
        for instruction in warp.instructions
        if instruction.HasField("instrumentation_order")
    )
    if ordered_tokens != list(range(1, 385)):
        raise AnalysisError("one-warp ordered canary token contract differs")
    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(default_payloads[0].read_bytes())
    default_fields = sum(
        instruction.HasField("instrumentation_order")
        for warp in threadblock.warps.values()
        for instruction in warp.instructions
    )
    if default_fields != 0:
        raise AnalysisError("default-disabled canary emitted tag 14")
    compatibility_results = []
    for path in (ordered / "stdout.txt", default / "stdout.txt"):
        result = None
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, dict)
                and candidate.get("workload") == "async_wait_prefix_order"
            ):
                result = candidate
                break
        if result is None:
            raise AnalysisError(f"{path}: no wait-prefix compatibility result")
        compatibility_results.append(result)
    ordered_result, default_result = compatibility_results
    for result in (ordered_result, default_result):
        if result.get("correct") is not True or result.get("epochs") != 64:
            raise AnalysisError("one-warp compatibility target result differs")
    return {
        "passed": True,
        "ordered_selected_events": len(ordered_tokens),
        "ordered_tokens_dense": True,
        "default_instrumentation_order_fields": default_fields,
        "ordered_trace_sha256": sha256(ordered / "dynamic_trace.pb"),
        "default_trace_sha256": sha256(default / "dynamic_trace.pb"),
        "independent_capture_bytes_not_compared": True,
    }


def holdout_paths(root: Path) -> list[Path]:
    return [
        root / "nvbit/holdout",
        root / "order-stripped/holdout",
        root / "simulator/holdout",
    ]


def build_smoke_authorization(
    root: Path = DEFAULT_ROOT, *, require_holdout_absent: bool = True
) -> dict[str, Any]:
    root = root.resolve()
    if require_holdout_absent:
        present = [str(path) for path in holdout_paths(root) if path.exists()]
        if present:
            raise AnalysisError(f"holdout exists before W4 authorization: {present}")
    reused = validate_reused_evidence()
    w4 = validate_stage(root, "w4", reused)
    return {
        "schema_version": 1,
        "kind": "sm86_async_selected_order_smoke_authorization",
        "status": "w4_selected_order_accepted",
        "root": str(root),
        "w4_passed": True,
        "holdout_authorized": True,
        "timing_promotion_authorized": False,
        "binary_sha256": reused["binary"]["sha256"],
        "control_sha256": reused["control"]["sha256"],
        "sim_run_sha256": sha256(SIM_RUN),
        "counts": {
            "w4_nvbit_traces": 8,
            "w4_ordered_replay_outputs": 16,
            "w4_stripped_replay_outputs": 8,
            "w4_classified_epochs": 8 * GROUPS,
        },
        "w4": w4,
    }


def build_report(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    authorization_path = root / "analysis/smoke-authorization.json"
    retained_authorization = read_json(authorization_path)
    authorization = build_smoke_authorization(root, require_holdout_absent=False)
    if retained_authorization != authorization:
        raise AnalysisError("retained W4 authorization differs from current evidence")
    reused = validate_reused_evidence()
    compatibility = validate_default_compatibility(root)
    holdout = validate_stage(root, "holdout", reused)
    return {
        "schema_version": 1,
        "assay": "sm86_async_selected_order_trace_v1",
        "status": "complete_selected_order_trace_fidelity",
        "decision": (
            "The timing-free instrumentation-order witness exactly preserves and "
            "classifies all forced W4 and W8 queued/drained causal relations. "
            "Observe-only schema 21 is deterministic and byte-equal to schema 20 "
            "after removing its diagnostic; no simulator timing is authorized."
        ),
        "root": str(root),
        "timing_promotion_authorized": False,
        "queue_parameter_promotion_authorized": False,
        "typed_tail_change_authorized": False,
        "scientific_scope": "instrumentation_order_not_silicon_service_order",
        "reused_evidence": reused,
        "default_disabled_compatibility": compatibility,
        "smoke_authorization_sha256": sha256(authorization_path),
        "w4": authorization["w4"],
        "holdout": holdout,
        "counts": {
            "reused_ncu_reports": 4,
            "compatibility_traces": 2,
            "nvbit_traces": 12,
            "ordered_replay_outputs": 24,
            "stripped_replay_outputs": 12,
            "classified_epochs": 12 * GROUPS,
            "w4_selected_events": 8 * expected_counts(4)["selected"],
            "w8_selected_events": 4 * expected_counts(8)["selected"],
        },
        "criteria": {
            "w4_exact_counts_and_relations": True,
            "w4_arm_classification_512_of_512": True,
            "w8_exact_counts_and_relations": True,
            "w8_arm_classification_256_of_256": True,
            "tokens_dense_and_per_warp_ordered": True,
            "default_disabled_emits_zero_order_fields": True,
            "schema21_observe_only": True,
            "ordered_replay_twins_deterministic": True,
            "normalized_schema20_outputs_equal": True,
            "no_live_or_simulator_timing_gate": True,
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    counts = require_object(report.get("counts"), "report counts")
    return f"""# SM86 selected asynchronous-copy instrumentation-order result

Status: `{report['status']}`

The opt-in tracer field carries one dense, timing-free order token for each
selected LDGSTS issue, group commit, and SB0 wait. It classifies every forced
queued/drained epoch in the W4 identification smoke and the unopened W8
transfer without reading a silicon or simulator cycle value.

| Stage | Traces | Classified epochs | Selected events |
|---|---:|---:|---:|
| W4 L1/L2, two repeats | 8 | 512 | {counts['w4_selected_events']} |
| W8 L2 holdout, two repeats | 4 | 256 | {counts['w8_selected_events']} |

All ordered replay twins are deterministic. Removing only the schema-21 order
diagnostic and normalizing its schema number reproduces the order-stripped
schema-20 output exactly, including timing, traffic, prior diagnostics, and
typed tails.

The deterministic one-warp producer canary carries 384 dense order fields when
enabled and zero when default-disabled. Raw bytes from independent live
captures are not compared because placement is itself captured state.

This establishes tracer and replay fidelity only. Instrumentation/channel
order is not independent silicon issue, source-completion, destination-service,
or wait-completion order, so no timing or queue parameter is promoted.
"""


def write_or_validate(path: Path, payload: bytes, validate_only: bool) -> None:
    if validate_only:
        try:
            retained = path.read_bytes()
        except OSError as error:
            raise AnalysisError(f"{path}: cannot read retained output: {error}") from error
        if retained != payload:
            raise AnalysisError(f"{path}: retained output differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("smoke", "complete"), default="complete")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        if args.stage == "smoke":
            report = build_smoke_authorization(root)
            output = args.output or root / "analysis/smoke-authorization.json"
            encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
            write_or_validate(output, encoded, args.validate_only)
        else:
            report = build_report(root)
            output = args.output or root / "analysis/report.json"
            output_md = args.output_md or root / "analysis/report.md"
            encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
            rendered = markdown(report).encode()
            write_or_validate(output, encoded, args.validate_only)
            write_or_validate(output_md, rendered, args.validate_only)
            write_or_validate(args.aggregate_output, encoded, args.validate_only)
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"selected-order analysis: {error}", file=sys.stderr)
        return 1
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
