#!/usr/bin/env python3
"""Score branch-free per-warp IBuffer capacities against CLOCK timelines."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = REPO_ROOT / "results/captures/sm86_front_end_ibuffer_depth_20260818_v1"
FUNCTION = "_Z22ibuffer_timeline_probePyPj"
MEASURED_CLOCK_PCS = tuple(range(0x20, 0x220, 0x40))


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("ibuffer_candidate_order_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_order_analysis()
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

bindings = load_dynamic_trace_bindings()


@functools.cache
def load_cell(
    root_text: str, key: str
) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]], dict[int, int]]:
    root = Path(root_text)
    variant = key.split("_", 1)[1]
    sidecar = json.loads(
        (root / "sass" / f"front_end_ibuffer_depth_{variant}.control_bits.json").read_text(encoding="utf-8")
    )
    static = {int(row["pc"]): row for row in sidecar["functions"][FUNCTION]}
    paths = list((root / "nvbit" / key / "threadblocks").rglob("*.pb"))
    if len(paths) != 1:
        raise RuntimeError(f"{key}: expected one traced block")
    block = bindings.threadblock_pb2.threadblock()
    block.ParseFromString(paths[0].read_bytes())
    streams = {}
    logical_by_slot = {}
    for warp in block.warps.values():
        if warp.instructions:
            slot = int(warp.instructions[0].warpid_sm)
            streams[slot] = [int(inst.pc) for inst in warp.instructions]
            logical_by_slot[slot] = int(warp.instructions[0].warpid_tb)
    return static, streams, logical_by_slot


def simulate(
    root: Path,
    key: str,
    capacity: int,
    instruction_miss_latency: int,
    prefetch_depth: int,
) -> dict[int, list[int]]:
    static, streams, logical_by_slot = load_cell(str(root), key)
    result: dict[int, list[int]] = {}
    for subcore in range(4):
        slots = sorted(slot for slot in streams if slot % 4 == subcore)
        if not slots:
            continue
        state = {
            slot: {
                "issue": 0,
                "fetch": 0,
                "valid": set(),
                "eligible": 0,
            }
            for slot in slots
        }
        clocks = {slot: [] for slot in slots}
        issue_greedy = slots[-1]
        fetch_greedy = slots[-1]
        fetch_decode_latch: tuple[int, int, int] | None = None
        responses: list[tuple[int, int, int, int]] = []
        response_sequence = 0
        clock_pipeline_ready = 0
        instruction_lines: dict[int, int] = {}
        time = 0
        while not all(state[slot]["issue"] == len(streams[slot]) for slot in slots):
            ready = []
            for slot in slots:
                warp = state[slot]
                if (
                    warp["issue"] >= len(streams[slot])
                    or warp["issue"] not in warp["valid"]
                    or time < warp["eligible"]
                ):
                    continue
                pc = streams[slot][warp["issue"]]
                if static[pc]["opcode_text"].startswith("CS2R") and time < clock_pipeline_ready:
                    continue
                ready.append(slot)
            chosen = issue_greedy if issue_greedy in ready else (max(ready) if ready else None)
            if chosen is not None:
                warp = state[chosen]
                pc = streams[chosen][warp["issue"]]
                control = static[pc]["control"]
                warp["valid"].remove(warp["issue"])
                warp["issue"] += 1
                warp["eligible"] = time + max(
                    1,
                    int(control["stall"]),
                    2 if control["yield_flag"] else 1,
                )
                if static[pc]["opcode_text"].startswith("CS2R"):
                    clock_pipeline_ready = time + 2
                    if pc in MEASURED_CLOCK_PCS:
                        clocks[chosen].append(time)
                if not control["yield_flag"]:
                    issue_greedy = chosen

            if fetch_decode_latch is not None and fetch_decode_latch[0] <= time:
                _, slot, instruction_index = fetch_decode_latch
                state[slot]["valid"].add(instruction_index)
                fetch_decode_latch = None
            if fetch_decode_latch is None:
                ready_responses = [entry for entry in responses if entry[0] <= time]
                if ready_responses:
                    response = min(ready_responses)
                    responses.remove(response)
                    _, _, slot, instruction_index = response
                    fetch_decode_latch = (time + 1, slot, instruction_index)

            order = [fetch_greedy] + [
                slot for slot in sorted(slots, reverse=True) if slot != fetch_greedy
            ]
            for slot in order:
                warp = state[slot]
                if (
                    warp["fetch"] >= len(streams[slot])
                    or warp["fetch"] - warp["issue"] >= capacity
                ):
                    continue
                instruction_index = warp["fetch"]
                pc = streams[slot][instruction_index]
                line = pc // 128
                ready_at = instruction_lines.get(line)
                if ready_at is None:
                    ready_at = time + instruction_miss_latency
                    instruction_lines[line] = ready_at
                    for offset in range(1, prefetch_depth + 1):
                        prefetched = line + offset
                        instruction_lines.setdefault(
                            prefetched,
                            time + instruction_miss_latency + offset,
                        )
                response_sequence += 1
                responses.append(
                    (max(time, ready_at), response_sequence, slot, instruction_index)
                )
                warp["fetch"] += 1
                break
            fetch_greedy = issue_greedy
            time += 1
            if time > 100_000:
                raise RuntimeError(f"candidate made no progress: {key} SMSP{subcore}")
        for slot, values in clocks.items():
            if len(values) != 8:
                raise RuntimeError(f"{key}: slot {slot} produced {len(values)} measured clocks")
            result[logical_by_slot[slot]] = values
    return result


def observed(root: Path, key: str) -> dict[int, list[float]]:
    payload = json.loads((root / "correctness" / key / "result.json").read_text(encoding="utf-8"))[
        "measurement"
    ]
    timelines = payload["timeline_cycles"]
    return {
        warp: [
            statistics.median(trial[warp][index] for trial in timelines)
            for index in range(8)
        ]
        for warp in range(payload["warps"])
    }


def normalized(timeline: dict[int, list[int | float]]) -> dict[int, list[float]]:
    origin = min(value for values in timeline.values() for value in values)
    return {warp: [float(value - origin) for value in values] for warp, values in timeline.items()}


def mean_absolute_error(left: list[float], right: list[float]) -> float:
    return statistics.mean(abs(a - b) for a, b in zip(left, right))


def reduce(root: Path) -> dict[str, Any]:
    cells = json.loads((root / "metadata/matrix.json").read_text(encoding="utf-8"))["cells"]
    keys = [entry["key"] for entry in cells]
    hardware = {key: normalized(observed(root, key)) for key in keys}
    baselines = {entry["warps"]: f"w{entry['warps']}_baseline_s1" for entry in cells if entry["variant"] == "baseline_s1"}
    candidates = []
    for capacity in range(1, 9):
      for instruction_miss_latency in (4, 8, 12, 16, 24, 32, 40):
       for prefetch_depth in (0, 1, 2, 4, 8):
        predicted = {
            key: normalized(
                simulate(
                    root,
                    key,
                    capacity,
                    instruction_miss_latency,
                    prefetch_depth,
                )
            )
            for key in keys
        }
        absolute_errors = []
        delta_errors = []
        event_order_matches = 0
        event_order_total = 0
        by_cell = {}
        for entry in cells:
            key = entry["key"]
            baseline_key = baselines[entry["warps"]]
            cell_absolute = []
            cell_delta = []
            for warp in hardware[key]:
                for index in range(8):
                    cell_absolute.append(
                        abs(predicted[key][warp][index] - hardware[key][warp][index])
                    )
                    if key != baseline_key:
                        hardware_delta = hardware[key][warp][index] - hardware[baseline_key][warp][index]
                        predicted_delta = predicted[key][warp][index] - predicted[baseline_key][warp][index]
                        cell_delta.append(abs(predicted_delta - hardware_delta))
            absolute_errors.extend(cell_absolute)
            delta_errors.extend(cell_delta)
            hardware_order = [
                (warp, index)
                for _, warp, index in sorted(
                    (value, warp, index)
                    for warp, values in hardware[key].items()
                    for index, value in enumerate(values)
                )
            ]
            predicted_order = [
                (warp, index)
                for _, warp, index in sorted(
                    (value, warp, index)
                    for warp, values in predicted[key].items()
                    for index, value in enumerate(values)
                )
            ]
            matches = sum(left == right for left, right in zip(hardware_order, predicted_order))
            event_order_matches += matches
            event_order_total += len(hardware_order)
            by_cell[key] = {
                "absolute_mae": statistics.mean(cell_absolute),
                "delta_mae": statistics.mean(cell_delta) if cell_delta else 0.0,
                "event_order_matches": matches,
                "events": len(hardware_order),
                "predicted": predicted[key],
            }
        candidates.append(
            {
                "capacity": capacity,
                "instruction_miss_latency": instruction_miss_latency,
                "prefetch_depth": prefetch_depth,
                "absolute_timeline_mae": statistics.mean(absolute_errors),
                "treatment_delta_mae": statistics.mean(delta_errors),
                "event_order_accuracy": event_order_matches / event_order_total,
                "event_order_matches": event_order_matches,
                "events": event_order_total,
                "identification_absolute_mae": statistics.mean(
                    value["absolute_mae"]
                    for name, value in by_cell.items()
                    if name.startswith("w16_")
                ),
                "identification_delta_mae": statistics.mean(
                    value["delta_mae"]
                    for name, value in by_cell.items()
                    if name.startswith("w16_") and "baseline" not in name
                ),
                "transfer_absolute_mae": statistics.mean(
                    value["absolute_mae"]
                    for name, value in by_cell.items()
                    if not name.startswith("w16_")
                ),
                "transfer_delta_mae": statistics.mean(
                    value["delta_mae"]
                    for name, value in by_cell.items()
                    if not name.startswith("w16_") and "baseline" not in name
                ),
                "cells": by_cell,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["treatment_delta_mae"],
            row["absolute_timeline_mae"],
            -row["event_order_accuracy"],
        )
    )
    absolute_best = min(candidates, key=lambda row: row["absolute_timeline_mae"])
    reference = min(
        (
            row
            for row in candidates
            if row["capacity"] == 3 and row["prefetch_depth"] == 8
        ),
        key=lambda row: (row["treatment_delta_mae"], row["absolute_timeline_mae"]),
    )
    best_cells = candidates[0]["cells"]
    compact_candidates = [
        {key: value for key, value in row.items() if key != "cells"}
        for row in candidates
    ]
    return {
        "schema_version": 1,
        "model": {
            "capacities": list(range(1, 9)),
            "smsps": 4,
            "fetch_decode_width_per_smsp": 1,
            "fetch_decode_latency_cycles": 1,
            "issue_policy": "compiler_guided_greedy_then_youngest",
            "fetch_policy": "last_issued_then_youngest_with_space",
            "clock_initiation_interval_cycles": 2,
            "instruction_line_bytes": 128,
            "instruction_miss_latencies": [4, 8, 12, 16, 24, 32, 40],
            "prefetch_depths": [0, 1, 2, 4, 8],
            "prefetch_issue_rate_lines_per_cycle": 1,
        },
        "hardware": hardware,
        "candidates": compact_candidates,
        "best_predictions": best_cells,
        "decision": {
            "best_capacity": candidates[0]["capacity"],
            "best_instruction_miss_latency": candidates[0]["instruction_miss_latency"],
            "best_prefetch_depth": candidates[0]["prefetch_depth"],
            "best_treatment_delta_mae": candidates[0]["treatment_delta_mae"],
            "best_absolute_timeline_mae": candidates[0]["absolute_timeline_mae"],
            "best_event_order_accuracy": candidates[0]["event_order_accuracy"],
            "absolute_best": {
                key: absolute_best[key]
                for key in (
                    "capacity",
                    "instruction_miss_latency",
                    "prefetch_depth",
                    "absolute_timeline_mae",
                    "treatment_delta_mae",
                    "event_order_accuracy",
                )
            },
            "reference_capacity3_prefetch8": {
                key: reference[key]
                for key in (
                    "instruction_miss_latency",
                    "absolute_timeline_mae",
                    "treatment_delta_mae",
                    "event_order_accuracy",
                )
            },
            "status": "capacity_two_dominates_but_full_front_end_candidate_rejected",
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--out", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.root.resolve()
    output = args.out or root / "metadata/front_end_candidates.json"
    try:
        summary = reduce(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"IBuffer candidates: {error}", file=sys.stderr)
        return 1
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
