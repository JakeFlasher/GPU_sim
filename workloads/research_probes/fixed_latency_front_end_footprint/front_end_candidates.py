#!/usr/bin/env python3
"""Score reference-style fetch/decode candidates on the footprint factorial."""

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
DEFAULT_ROOT = (
    REPO_ROOT
    / "results/captures/sm86_fixed_latency_front_end_footprint_20260817_v1"
)
UNROLLS = (1, 2, 4, 8)
WIDTHS = (4, 5, 8, 9, 13, 16)
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
FUNCTIONS = {
    "fixed": "_Z22footprint_fixed_searchPyPjPKj",
    "global": "_Z23footprint_global_searchPyPjPKjS2_",
}


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("footprint_candidate_order_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = load_order_analysis()
wide_ncu = ORDER.wide_ncu
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings

bindings = load_dynamic_trace_bindings()


@functools.cache
def load_cell(root_text: str, key: str) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    root = Path(root_text)
    parts = key.split("_")
    unroll = int(parts[0][1:])
    variant = "_".join(parts[2:])
    sidecar = json.loads(
        (
            root
            / "sass"
            / f"fixed_latency_front_end_footprint_u{unroll}_trace_{variant}.control_bits.json"
        ).read_text(encoding="utf-8")
    )
    function = FUNCTIONS["fixed" if variant.startswith("fixed") else "global"]
    static = {int(row["pc"]): row for row in sidecar["functions"][function]}
    paths = list((root / "nvbit" / key / "threadblocks").rglob("*.pb"))
    if len(paths) != 1:
        raise RuntimeError(f"{key}: expected one traced block")
    block = bindings.threadblock_pb2.threadblock()
    block.ParseFromString(paths[0].read_bytes())
    streams = {}
    for warp in block.warps.values():
        if warp.instructions:
            slot = int(warp.instructions[0].warpid_sm)
            streams[slot] = [int(inst.pc) for inst in warp.instructions]
    return static, streams


def simulate(
    root: Path,
    key: str,
    *,
    capacity: int,
    stop_at_branch: bool,
    branch_resolve_latency: int,
) -> dict[str, int]:
    static, streams = load_cell(str(root), key)
    subcore_cycles = []
    total_no_instruction = 0
    total_branch_resolving = 0
    for subcore in range(4):
        slots = sorted(slot for slot in streams if slot % 4 == subcore)
        if not slots:
            continue
        state = {
            slot: {
                "issue": 0,
                "fetch": 0,
                "decoded": 0,
                "eligible": 0,
                "barrier": [0] * 6,
                "unresolved_branch": False,
                "branch_ready": 0,
            }
            for slot in slots
        }
        issue_greedy = slots[-1]
        fetch_greedy = slots[-1]
        fetch_decode_latch: tuple[int, int] | None = None
        time = 0
        last_issue = 0
        while not all(state[slot]["issue"] == len(streams[slot]) for slot in slots):
            for slot in slots:
                warp = state[slot]
                if warp["unresolved_branch"] and warp["branch_ready"] and time >= warp["branch_ready"]:
                    warp["unresolved_branch"] = False
                    warp["branch_ready"] = 0
                if warp["branch_ready"] > time:
                    total_branch_resolving += 1
                elif (
                    warp["issue"] < len(streams[slot])
                    and warp["issue"] >= warp["decoded"]
                    and time >= warp["eligible"]
                ):
                    pc = streams[slot][warp["issue"]]
                    control = static[pc]["control"]
                    dependency = max(
                        (
                            warp["barrier"][index]
                            for index in range(6)
                            if control["wait_mask"] & (1 << index)
                        ),
                        default=0,
                    )
                    if time >= dependency:
                        total_no_instruction += 1
            ready = []
            for slot in slots:
                warp = state[slot]
                if (
                    warp["issue"] >= len(streams[slot])
                    or warp["issue"] >= warp["decoded"]
                    or time < warp["eligible"]
                ):
                    continue
                pc = streams[slot][warp["issue"]]
                control = static[pc]["control"]
                dependency = max(
                    (
                        warp["barrier"][index]
                        for index in range(6)
                        if control["wait_mask"] & (1 << index)
                    ),
                    default=0,
                )
                if time >= dependency:
                    ready.append(slot)
            chosen = issue_greedy if issue_greedy in ready else (max(ready) if ready else None)
            if chosen is not None:
                warp = state[chosen]
                pc = streams[chosen][warp["issue"]]
                row = static[pc]
                control = row["control"]
                warp["issue"] += 1
                last_issue = time
                warp["eligible"] = time + max(
                    1,
                    int(control["stall"]),
                    2 if control["yield_flag"] else 1,
                )
                write_barrier = int(control["write_barrier"])
                if write_barrier < 6:
                    latency = 34 if row["opcode_text"].startswith("LDG") else 25
                    warp["barrier"][write_barrier] = max(
                        warp["barrier"][write_barrier], time + latency
                    )
                read_barrier = int(control["read_barrier"])
                if read_barrier < 6:
                    warp["barrier"][read_barrier] = max(
                        warp["barrier"][read_barrier], time + 8
                    )
                if row["opcode_text"].startswith("BRA"):
                    if stop_at_branch:
                        warp["branch_ready"] = time + 1 + branch_resolve_latency
                    else:
                        warp["unresolved_branch"] = False
                if not control["yield_flag"]:
                    issue_greedy = chosen

            if fetch_decode_latch is not None and fetch_decode_latch[0] <= time:
                _, slot = fetch_decode_latch
                state[slot]["decoded"] += 1
                fetch_decode_latch = None
            if fetch_decode_latch is None:
                order = [fetch_greedy] + [
                    slot for slot in sorted(slots, reverse=True) if slot != fetch_greedy
                ]
                for slot in order:
                    warp = state[slot]
                    in_flight = int(
                        fetch_decode_latch is not None
                        and fetch_decode_latch[1] == slot
                    )
                    if (
                        warp["fetch"] >= len(streams[slot])
                        or warp["decoded"] - warp["issue"] + in_flight >= capacity
                        or warp["unresolved_branch"]
                    ):
                        continue
                    pc = streams[slot][warp["fetch"]]
                    warp["fetch"] += 1
                    if stop_at_branch and static[pc]["opcode_text"].startswith("BRA"):
                        warp["unresolved_branch"] = True
                    fetch_decode_latch = (time + 1, slot)
                    break
            fetch_greedy = issue_greedy
            time += 1
            if time > 1_000_000:
                raise RuntimeError(f"candidate made no progress: {key} subcore {subcore}")
        subcore_cycles.append(last_issue + 1)
    return {
        "cycles": max(subcore_cycles),
        "no_instruction_warp_cycles": total_no_instruction,
        "branch_resolving_warp_cycles": total_branch_resolving,
    }


def observed(root: Path) -> dict[str, dict[str, float]]:
    result = {}
    for unroll in UNROLLS:
        for warps in WIDTHS:
            values = {}
            for variant in VARIANTS:
                key = f"u{unroll}_w{warps}_{variant}"
                payload = json.loads(
                    (root / "correctness" / key / "result.json").read_text(encoding="utf-8")
                )["measurement"]
                driver = statistics.median(
                    max(sample) for sample in payload["samples_cycles"]
                ) / 1024
                rows = [
                    wide_ncu(path / "ncu.csv")
                    for path in sorted((root / "ncu" / key).glob("repeat-*"))
                ]
                active = statistics.median(
                    row["sm__cycles_active.max"] for row in rows
                ) / 1024
                replay = json.loads(
                    (root / "simulator" / key / "replay-1.json").read_text(encoding="utf-8")
                )["launches"][0]["timing"]["diagnostic_last_issue_cycles"] / 64
                front_end = wide_ncu(root / "ncu-front-end" / key / "ncu.csv")
                values[variant] = {
                    "driver": driver,
                    "active": active,
                    "replay": replay,
                    "no_instruction_per_warp_generation": (
                        front_end["smsp__warps_issue_stalled_no_instruction.sum"]
                        / (1024 * warps)
                    ),
                    "branch_resolving_per_warp_generation": (
                        front_end["smsp__warps_issue_stalled_branch_resolving.sum"]
                        / (1024 * warps)
                    ),
                }
            result[f"u{unroll}_w{warps}"] = {
                "driver_fixed": values["fixed_retired"]["driver"] - values["fixed_bypass"]["driver"],
                "active_fixed": values["fixed_retired"]["active"] - values["fixed_bypass"]["active"],
                "production_fixed": values["fixed_retired"]["replay"] - values["fixed_bypass"]["replay"],
                "driver_global": values["global_late"]["driver"] - values["global_frontier"]["driver"],
                "active_global": values["global_late"]["active"] - values["global_frontier"]["active"],
                "production_global": values["global_late"]["replay"] - values["global_frontier"]["replay"],
                "front_end": values,
            }
    return result


def mae(predicted: list[float], target: list[float]) -> float:
    return statistics.mean(abs(a - b) for a, b in zip(predicted, target))


def reduce(root: Path) -> dict[str, Any]:
    truth = observed(root)
    keys = [f"u{unroll}_w{warps}" for unroll in UNROLLS for warps in WIDTHS]
    targets = {
        name: [truth[key][name] for key in keys]
        for name in ("driver_fixed", "active_fixed", "driver_global", "active_global")
    }
    production = {
        family: [truth[key][f"production_{family}"] for key in keys]
        for family in ("fixed", "global")
    }
    front_end_targets = {
        metric: [
            truth[key]["front_end"][variant][metric]
            for key in keys
            for variant in VARIANTS
        ]
        for metric in (
            "no_instruction_per_warp_generation",
            "branch_resolving_per_warp_generation",
        )
    }
    rows = []
    candidate_shapes = [(False, 0)] + [
        (True, latency) for latency in (2, 3, 4)
    ]
    for stop_at_branch, branch_resolve_latency in candidate_shapes:
        for capacity in (1, 2, 3, 4, 8, 16):
            predicted = {"fixed": [], "global": []}
            predicted_no_instruction = []
            predicted_branch_resolving = []
            by_cell = {}
            for unroll in UNROLLS:
                for warps in WIDTHS:
                    simulations = {
                        variant: simulate(
                            root,
                            f"u{unroll}_w{warps}_{variant}",
                            capacity=capacity,
                            stop_at_branch=stop_at_branch,
                            branch_resolve_latency=branch_resolve_latency,
                        )
                        for variant in VARIANTS
                    }
                    fixed = (
                        simulations["fixed_retired"]["cycles"]
                        - simulations["fixed_bypass"]["cycles"]
                    ) / 64
                    global_delta = (
                        simulations["global_late"]["cycles"]
                        - simulations["global_frontier"]["cycles"]
                    ) / 64
                    predicted["fixed"].append(fixed)
                    predicted["global"].append(global_delta)
                    front_end_predictions = {}
                    for variant in VARIANTS:
                        no_instruction = (
                            simulations[variant]["no_instruction_warp_cycles"]
                            / (64 * warps)
                        )
                        branch_resolving = (
                            simulations[variant]["branch_resolving_warp_cycles"]
                            / (64 * warps)
                        )
                        predicted_no_instruction.append(no_instruction)
                        predicted_branch_resolving.append(branch_resolving)
                        front_end_predictions[variant] = {
                            "no_instruction_per_warp_generation": no_instruction,
                            "branch_resolving_per_warp_generation": branch_resolving,
                        }
                    by_cell[f"u{unroll}_w{warps}"] = {
                        "fixed": fixed,
                        "global": global_delta,
                        "front_end": front_end_predictions,
                    }
            row = {
                "capacity": capacity,
                "stop_fetch_at_unresolved_branch": stop_at_branch,
                "branch_resolve_latency": branch_resolve_latency,
                "fixed_driver_mae": mae(predicted["fixed"], targets["driver_fixed"]),
                "fixed_active_mae": mae(predicted["fixed"], targets["active_fixed"]),
                "global_driver_mae": mae(predicted["global"], targets["driver_global"]),
                "global_active_mae": mae(predicted["global"], targets["active_global"]),
                "no_instruction_per_warp_generation_mae": mae(
                    predicted_no_instruction,
                    front_end_targets["no_instruction_per_warp_generation"],
                ),
                "branch_resolving_per_warp_generation_mae": mae(
                    predicted_branch_resolving,
                    front_end_targets["branch_resolving_per_warp_generation"],
                ),
                "predictions": by_cell,
            }
            row["primary_mae"] = statistics.mean(
                (row["fixed_driver_mae"], row["fixed_active_mae"])
            )
            row["transfer_mae"] = statistics.mean(
                (row["global_driver_mae"], row["global_active_mae"])
            )
            row["front_end_counter_mae"] = statistics.mean(
                (
                    row["no_instruction_per_warp_generation_mae"],
                    row["branch_resolving_per_warp_generation_mae"],
                )
            )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["front_end_counter_mae"],
            row["primary_mae"],
            row["transfer_mae"],
        )
    )
    production_metrics = {
        "fixed_driver_mae": mae(production["fixed"], targets["driver_fixed"]),
        "fixed_active_mae": mae(production["fixed"], targets["active_fixed"]),
        "global_driver_mae": mae(production["global"], targets["driver_global"]),
        "global_active_mae": mae(production["global"], targets["active_global"]),
    }
    production_metrics["primary_mae"] = statistics.mean(
        (production_metrics["fixed_driver_mae"], production_metrics["fixed_active_mae"])
    )
    production_metrics["transfer_mae"] = statistics.mean(
        (production_metrics["global_driver_mae"], production_metrics["global_active_mae"])
    )
    return {
        "schema_version": 1,
        "model": {
            "per_warp_ibuffer_entries": [1, 2, 3, 4, 8, 16],
            "fetch_decode_width_per_smsp": 1,
            "fetch_decode_latency_cycles": 1,
            "fetch_policy": "greedy_then_highest_warp_id",
            "issue_policy": "greedy_then_highest_ready_warp_id",
            "branch_fetch_policies": ["stop", "continue"],
            "branch_resolve_latencies": [2, 3, 4],
            "l1_hit_latency_cycles": 34,
            "non_memory_write_barrier_latency_cycles": 25,
        },
        "observed": truth,
        "production": {
            **production_metrics,
            "predictions": {
                key: {
                    "fixed": production["fixed"][index],
                    "global": production["global"][index],
                }
                for index, key in enumerate(keys)
            },
        },
        "candidates": rows,
        "decision": {
            "status": "reference_candidate_scored_on_branch_footprint_factorial",
            "best_capacity": rows[0]["capacity"],
            "best_branch_policy": (
                "stop" if rows[0]["stop_fetch_at_unresolved_branch"] else "continue"
            ),
            "best_branch_resolve_latency": rows[0]["branch_resolve_latency"],
            "best_front_end_counter_mae": rows[0]["front_end_counter_mae"],
            "best_primary_mae": rows[0]["primary_mae"],
            "best_transfer_mae": rows[0]["transfer_mae"],
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
        print(f"front-end footprint candidates: {error}", file=sys.stderr)
        return 1
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
