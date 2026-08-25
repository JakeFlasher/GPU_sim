#!/usr/bin/env python3
"""Score bounded reference-inspired fetch/decode/IBuffer candidates."""

from __future__ import annotations

import argparse
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
    / "results/captures/sm86_fixed_latency_producer_shift_20260817_v1"
)
WIDTHS = tuple(range(4, 17))
VARIANTS = ("fixed_bypass", "fixed_retired", "global_frontier", "global_late")
FUNCTIONS = {
    "fixed": "_Z19bypass_fixed_searchPyPjPKj",
    "global": "_Z20bypass_global_searchPyPjPKjS2_",
}


def load_order_analysis():
    path = HERE.parent / "fixed_latency_result_order/analyze.py"
    spec = importlib.util.spec_from_file_location("shift_candidate_order_analysis", path)
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


def load_cell(root: Path, key: str) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    variant = key.split("_", 1)[1]
    sidecar = json.loads(
        (root / "sass" / f"fixed_latency_producer_shift_trace_{variant}.control_bits.json").read_text()
    )
    function = FUNCTIONS["fixed" if variant.startswith("fixed") else "global"]
    static = {int(row["pc"]): row for row in sidecar["functions"][function]}
    path = next((root / "nvbit" / key / "threadblocks").rglob("*.pb"))
    block = bindings.threadblock_pb2.threadblock()
    block.ParseFromString(path.read_bytes())
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
) -> int:
    static, streams = load_cell(root, key)
    subcore_cycles = []
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
            }
            for slot in slots
        }
        issue_greedy = slots[-1]
        fetch_greedy = slots[-1]
        fetch_decode_latch: tuple[int, int] | None = None
        time = 0
        last_issue = 0
        while not all(state[slot]["issue"] == len(streams[slot]) for slot in slots):
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
                    warp["unresolved_branch"] = False
                if not control["yield_flag"]:
                    issue_greedy = chosen

            # Reference stage order is Issue -> Decode -> Fetch.
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
                    if (
                        warp["fetch"] >= len(streams[slot])
                        or warp["decoded"] - warp["issue"] >= capacity
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
    return max(subcore_cycles)


def observed(root: Path) -> dict[str, dict[str, float]]:
    result = {}
    for warps in WIDTHS:
        values = {}
        for variant in VARIANTS:
            key = f"w{warps}_{variant}"
            payload = json.loads((root / "correctness" / key / "result.json").read_text())[
                "measurement"
            ]
            driver = statistics.median(max(sample) for sample in payload["samples_cycles"]) / 1025
            rows = [
                wide_ncu(path / "ncu.csv")
                for path in sorted((root / "ncu" / key).glob("repeat-*"))
            ]
            active = statistics.median(row["sm__cycles_active.max"] for row in rows) / 1025
            replay = json.loads(
                (root / "simulator" / key / "replay-1.json").read_text()
            )["launches"][0]["timing"]["diagnostic_last_issue_cycles"] / 65
            values[variant] = (driver, active, replay)
        result[str(warps)] = {
            "driver_fixed": values["fixed_retired"][0] - values["fixed_bypass"][0],
            "active_fixed": values["fixed_retired"][1] - values["fixed_bypass"][1],
            "production_fixed": values["fixed_retired"][2] - values["fixed_bypass"][2],
            "driver_global": values["global_late"][0] - values["global_frontier"][0],
            "active_global": values["global_late"][1] - values["global_frontier"][1],
            "production_global": values["global_late"][2] - values["global_frontier"][2],
        }
    return result


def mae(predicted: list[float], target: list[float]) -> float:
    return statistics.mean(abs(a - b) for a, b in zip(predicted, target))


def reduce(root: Path) -> dict[str, Any]:
    truth = observed(root)
    targets = {
        name: [truth[str(warps)][name] for warps in WIDTHS]
        for name in ("driver_fixed", "active_fixed", "driver_global", "active_global")
    }
    production = {
        "fixed": [truth[str(warps)]["production_fixed"] for warps in WIDTHS],
        "global": [truth[str(warps)]["production_global"] for warps in WIDTHS],
    }
    rows = []
    for stop_at_branch in (True, False):
        for capacity in (1, 2, 3, 4, 8):
            predicted_fixed = []
            predicted_global = []
            by_width = {}
            for warps in WIDTHS:
                fixed = (
                    simulate(root, f"w{warps}_fixed_retired", capacity=capacity, stop_at_branch=stop_at_branch)
                    - simulate(root, f"w{warps}_fixed_bypass", capacity=capacity, stop_at_branch=stop_at_branch)
                ) / 65
                global_delta = (
                    simulate(root, f"w{warps}_global_late", capacity=capacity, stop_at_branch=stop_at_branch)
                    - simulate(root, f"w{warps}_global_frontier", capacity=capacity, stop_at_branch=stop_at_branch)
                ) / 65
                predicted_fixed.append(fixed)
                predicted_global.append(global_delta)
                by_width[str(warps)] = {"fixed": fixed, "global": global_delta}
            rows.append(
                {
                    "capacity": capacity,
                    "stop_fetch_at_unresolved_branch": stop_at_branch,
                    "driver_mae": statistics.mean(
                        (
                            mae(predicted_fixed, targets["driver_fixed"]),
                            mae(predicted_global, targets["driver_global"]),
                        )
                    ),
                    "active_mae": statistics.mean(
                        (
                            mae(predicted_fixed, targets["active_fixed"]),
                            mae(predicted_global, targets["active_global"]),
                        )
                    ),
                    "fixed_driver_mae": mae(predicted_fixed, targets["driver_fixed"]),
                    "global_driver_mae": mae(predicted_global, targets["driver_global"]),
                    "fixed_active_mae": mae(predicted_fixed, targets["active_fixed"]),
                    "global_active_mae": mae(predicted_global, targets["active_global"]),
                    "predictions": by_width,
                }
            )
    rows.sort(key=lambda row: (row["driver_mae"], row["active_mae"]))
    production_driver_mae = statistics.mean(
        (
            mae(production["fixed"], targets["driver_fixed"]),
            mae(production["global"], targets["driver_global"]),
        )
    )
    production_active_mae = statistics.mean(
        (
            mae(production["fixed"], targets["active_fixed"]),
            mae(production["global"], targets["active_global"]),
        )
    )
    return {
        "schema_version": 1,
        "model": {
            "per_smsp_ibuffer_entries": [1, 2, 3, 4, 8],
            "fetch_decode_width": 1,
            "fetch_decode_latency_cycles": 1,
            "fetch_policy": "greedy_then_highest_warp_id",
            "issue_policy": "greedy_then_highest_ready_warp_id",
            "branch_fetch_policies": ["stop", "continue"],
            "l1_hit_latency_cycles": 34,
            "non_memory_write_barrier_latency_cycles": 25,
        },
        "observed": truth,
        "production": {
            "driver_mae": production_driver_mae,
            "active_mae": production_active_mae,
            "predictions": {
                str(warps): {
                    "fixed": production["fixed"][index],
                    "global": production["global"][index],
                }
                for index, warps in enumerate(WIDTHS)
            },
        },
        "candidates": rows,
        "decision": {
            "status": "all_reference_inspired_front_end_candidates_rejected",
            "best_capacity": rows[0]["capacity"],
            "best_branch_policy": (
                "stop" if rows[0]["stop_fetch_at_unresolved_branch"] else "continue"
            ),
            "best_driver_mae": rows[0]["driver_mae"],
            "best_active_mae": rows[0]["active_mae"],
            "reason": (
                "no one IBuffer/branch policy transfers the fixed and global signs "
                "across second-, third-, and fourth-warp SMSP cohorts"
            ),
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
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"front-end candidates: {error}", file=sys.stderr)
        return 1
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
