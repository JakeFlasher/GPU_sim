#!/usr/bin/env python3
"""Stage-centric interval projection for fast SM86 endpoint estimates.

This is a conservative, metric-targeted alternative to detailed replay.  It
borrows two ideas from public analytical simulators: (1) GCStack/GCScaler's
separation of base, memory, compute, synchronization, control, and idle
intervals, and (2) recent stage-centric GPU models' max-plus composition of
compute, memory, synchronization, and writeback stages.  The implementation
does not pretend that the semantic classes are physical SM86 pipes.

The input contract is deliberately narrow:

* sparse replay frontiers (issue/event/GPC-envelope/tail),
* static SASS opcode populations and control metadata from the captured trace,
* dynamic trace population and coalesced-sector summaries, and
* independent-cohort NCU labels used only to learn an endpoint efficiency
  surface.

The output always contains a finite candidate or candidate_fallback estimate.
Distance widens an interval; it never drops a workload as OOD.  This tool is
an endpoint projection/committee member, not a native scheduler/cache/ABI
simulator.  NCU score labels are read only for final error reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from sm86_frontier_fallback import fit_frontier_fallback
from sm86_ncu_projection_surrogate import (
    TARGETS,
    _float,
    _flash_bwd_archetype_projection,
    _load_archetype_matrix,
    _load_shape_specs,
    _shape_spec_for_workload,
    fit_ridge,
    load_rows,
)


SM_CLOCK_MHZ = 1350.0
SEMANTIC_STAGES = ("compute", "memory", "control", "sync", "tail")


def _opcode_class(opcode: str) -> str:
    """Map an opcode to a semantic demand class, never to a claimed pipe."""
    head = opcode.upper().split(".", 1)[0]
    if head.startswith(("LD", "ST", "ATOM", "RED", "LDS", "STS", "CP", "TEX", "TLD")):
        return "memory"
    if head.startswith(("HMMA", "IMMA", "DMMA", "MMA", "WGMMA")):
        return "compute"
    if head.startswith(("BRA", "BRX", "BSSY", "BSYNC", "SSY", "EXIT", "RET", "PBK", "BRK")):
        return "control"
    if head.startswith(("BAR", "MEMBAR", "DEPBAR", "WARPSYNC", "VOTE", "SHFL")):
        return "sync"
    if head.startswith(("MUFU", "RCP", "RSQ", "SIN", "COS", "EX2", "LG2")):
        return "compute"
    return "compute"


def _read_stats(path: Path) -> dict[str, float]:
    """Read stats with the legacy leading-space column spelling tolerated."""
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        raw = next(csv.DictReader(stream), None) or {}
    return {str(key).strip(): _float({"v": str(value)}, "v") for key, value in raw.items()}


def stage_features(
    campaign: Path,
    row: dict[str, str],
    dynamic_trace: bool = False,
    dynamic_sample: int = 0,
) -> dict[str, float]:
    """Extract semantic stage demands from trace metadata and replay frontiers."""
    root = campaign / row["workload"] / "trace"
    info = json.loads((root / "extra_info" / "enhanced_execution_info.json").read_text())
    kernels = info.get("kernels", [])
    if not kernels:
        raise RuntimeError(f"trace has no kernel metadata: {root}")
    instructions = kernels[0].get("instructions", [])
    static_count = max(len(instructions), 1)
    class_counts = {name: 0.0 for name in SEMANTIC_STAGES if name not in {"tail"}}
    stall_sum = 0.0
    stall_max = 0.0
    barrier_count = 0.0
    wait_bits = 0.0
    max_registers = 0.0
    for instruction in instructions:
        class_counts[_opcode_class(str(instruction.get("op_code", "UNKNOWN")))] += 1.0
        control = instruction.get("control_bits", {})
        stall = _float({"v": str(control.get("stall_count", 0))}, "v")
        stall_sum += stall
        stall_max = max(stall_max, stall)
        barrier_count += float(bool(control.get("is_new_read_barrier", False)))
        barrier_count += float(bool(control.get("is_new_write_barrier", False)))
        wait_bits += float(int(control.get("wait_barrier_bits", 0) or 0).bit_count())
        max_registers = max(
            max_registers,
            _float({"v": str(instruction.get("register_usage", {}).get("num_regular", 0))}, "v"),
        )

    stats = _read_stats(root / "stats.csv")
    dynamic_insts = max(stats.get("total_insts", 0.0), 0.0)
    blocks = max(stats.get("#blocks", 0.0), 1.0)
    threads = max(stats.get("#threads", 0.0), 1.0)
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    issue = max(_float(row, "sim_last_issue_cycles"), 1.0)
    work = max(_float(row, "sim_warp_instructions"), 1.0)

    # Dynamic trace counts are cheaper and more reliable than the optional
    # stats total on older campaign files.  Keep both in the feature vector;
    # missing dynamic stats simply collapse to the static/work proxy.
    trace_records = 0.0
    trace_memory_records = 0.0
    trace_memory_bytes = 0.0
    trace_sectors = 0.0
    trace_warps = 0.0
    trace_blocks = 0.0
    dynamic_classes = {name: 0.0 for name in ("memory", "compute", "control", "sync")}
    try:
        if not dynamic_trace:
            raise StopIteration
        # Import lazily so the stage model remains a standalone fast path.
        from sm86_ncu_projection_surrogate import _load_threadblock_binding

        threadblock = _load_threadblock_binding()
        static_groups = {
            int(instruction.get("pc_num_dec", 0)):
            _opcode_class(str(instruction.get("op_code", "UNKNOWN")))
            for instruction in instructions
        }
        paths = sorted((root / "threadblocks").rglob("*.pb"))
        if dynamic_sample > 0:
            paths = paths[:dynamic_sample]
        for path in paths:
            block = threadblock()
            block.ParseFromString(path.read_bytes())
            trace_blocks += 1.0
            for warp in block.warps.values():
                trace_warps += 1.0
                for dynamic in warp.instructions:
                    trace_records += 1.0
                    name = static_groups.get(dynamic.pc, "compute")
                    dynamic_classes[name] += 1.0
                    mask = (dynamic.active_mask or 0) & (dynamic.predicate_mask or 0)
                    active = mask.bit_count()
                    if dynamic.addresses:
                        trace_memory_records += 1.0
                        for address in dynamic.addresses:
                            width = max(int(address.data_width), 1)
                            trace_memory_bytes += width * active
                            if address.compression_format == 0:
                                values = [value for value in address.addrs if value >= 0]
                                trace_sectors += len({value >> 5 for value in values})
                            elif address.compression_format == 1:
                                values = [address.base_address + lane * address.stride for lane in range(active)]
                                trace_sectors += len({value >> 5 for value in values})
                            else:
                                trace_sectors += float(bool(active))
    except (OSError, ValueError, TypeError, StopIteration):
        # The fallback still has static/frontier terms.  Preserve a bounded
        # candidate instead of turning a partial trace into a refusal.
        pass

    dynamic_scale = max(trace_records, dynamic_insts, work, 1.0)
    compute = max(dynamic_classes["compute"], work * class_counts["compute"] / static_count, 1.0)
    memory = max(
        dynamic_classes["memory"],
        trace_memory_records,
        work * class_counts["memory"] / static_count,
        1.0,
    )
    control = max(
        dynamic_classes["control"] + dynamic_classes["sync"],
        work * (class_counts["control"] + class_counts["sync"]) / static_count,
        1.0,
    )
    sync = max(barrier_count * max(trace_warps, 1.0), wait_bits * max(trace_warps, 1.0), 1.0)
    tail = max(envelope - event, event - issue, 1.0)
    return {
        "log_event": math.log(event),
        "log_envelope": math.log(envelope),
        "log_issue": math.log(issue),
        "log_work": math.log(work),
        "log_compute": math.log(compute),
        "log_memory": math.log(memory),
        "log_control": math.log(control),
        "log_sync": math.log(sync),
        "log_tail": math.log(tail),
        "envelope_over_event": envelope / event,
        "tail_over_event": tail / event,
        "work_over_event": work / event,
        "compute_share": compute / dynamic_scale,
        "memory_share": memory / dynamic_scale,
        "control_share": control / dynamic_scale,
        "sync_share": sync / dynamic_scale,
        "trace_memory_density": trace_memory_records / max(trace_records, 1.0),
        "trace_sector_density": trace_sectors / max(trace_memory_records, 1.0),
        "trace_bytes_per_record": trace_memory_bytes / max(trace_memory_records, 1.0),
        "trace_warps_per_block": trace_warps / max(trace_blocks, blocks, 1.0),
        "blocks_per_wave_proxy": blocks / max(trace_warps / max(trace_blocks, 1.0), 1.0),
        "static_memory_fraction": class_counts["memory"] / static_count,
        "static_compute_fraction": class_counts["compute"] / static_count,
        "static_control_fraction": class_counts["control"] / static_count,
        "static_sync_fraction": class_counts["sync"] / static_count,
        "static_stall_mean": stall_sum / static_count,
        "static_stall_max": stall_max,
        "static_barrier_density": barrier_count / static_count,
        "static_wait_density": wait_bits / static_count,
        "max_regular_registers": max_registers,
        "log_blocks": math.log(blocks),
        "log_threads": math.log(threads),
    }


def _base(row: dict[str, str], target: str) -> float:
    event = max(_float(row, "sim_event_cycles"), 1.0)
    envelope = max(_float(row, "sim_gpc_envelope_cycles"), event)
    return envelope if target == "ncu_gpc_cycles" else event * (1000.0 / SM_CLOCK_MHZ)


def _metric(report_rows: list[dict], target: str) -> dict[str, float]:
    actual = np.asarray([float(item[f"{target}_actual"]) for item in report_rows], dtype=float)
    prediction = np.asarray([float(item[f"{target}_predicted"]) for item in report_rows], dtype=float)
    return {
        "mae": float(np.mean(np.abs(prediction - actual))),
        "mape": float(np.mean(np.abs(prediction - actual) / np.maximum(actual, 1.0))),
        "max_abs_error": float(np.max(np.abs(prediction - actual))),
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    train_rows = [row for row in load_rows(args.train_campaign) if row["workload"].split("__", 1)[0] != args.holdout_family]
    score_rows = [row for row in load_rows(args.score_campaign) if row["workload"].split("__", 1)[0] == args.holdout_family]
    if not train_rows or not score_rows:
        raise RuntimeError("empty stage-model train or score partition")

    train_maps = [
        stage_features(args.train_campaign, row, args.dynamic_trace, args.dynamic_sample)
        for row in train_rows
    ]
    score_maps = [
        stage_features(args.score_campaign, row, args.dynamic_trace, args.dynamic_sample)
        for row in score_rows
    ]
    names = sorted(train_maps[0])
    if any(sorted(item) != names for item in train_maps + score_maps):
        raise RuntimeError("stage feature schema differs")
    x_train = np.asarray([[1.0] + [item[name] for name in names] for item in train_maps], dtype=float)
    x_score = np.asarray([[1.0] + [item[name] for name in names] for item in score_maps], dtype=float)

    predictions: dict[str, np.ndarray] = {}
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    coefficients: dict[str, list[float]] = {}
    residual_q90: dict[str, float] = {}
    train_bases: dict[str, np.ndarray] = {}
    score_bases: dict[str, np.ndarray] = {}
    for target in TARGETS:
        y = np.asarray([_float(row, target) for row in train_rows], dtype=float)
        train_base = np.asarray([_base(row, target) for row in train_rows], dtype=float)
        score_base = np.asarray([_base(row, target) for row in score_rows], dtype=float)
        # Log-efficiency regression keeps the learned state dimensionless.  A
        # max-plus base already supplies a legal lower frontier; the learned
        # overlap/idle/ABI/cache residual is the only fitted quantity.
        beta, scale = fit_ridge(x_train, np.log(np.maximum(y, 1.0) / np.maximum(train_base, 1.0)), args.ridge)
        train_pred = train_base * np.exp((x_train / scale) @ beta)
        score_pred = score_base * np.exp((x_score / scale) @ beta)
        residual_q90[target] = float(np.quantile(np.abs(train_pred - y), 0.90))
        predictions[target] = np.maximum(score_pred, 0.0)
        coefficients[target] = beta.tolist()
        train_bases[target] = train_base
        score_bases[target] = score_base

    # A monotone endpoint crosswalk is not asserted to be a common clock.  It
    # is only a learned duration/GPC consistency term from the independent
    # cohort, blended in log space to stop one endpoint from drifting.
    gpc = np.asarray([_float(row, "ncu_gpc_cycles") for row in train_rows], dtype=float)
    duration = np.asarray([_float(row, "ncu_duration_ns") for row in train_rows], dtype=float)
    cross_beta, cross_scale = fit_ridge(x_train, np.log(np.maximum(duration, 1.0) / np.maximum(gpc, 1.0)), args.ridge)
    cross_train = train_bases["ncu_gpc_cycles"] * np.exp((x_train / cross_scale) @ cross_beta)
    cross_score = predictions["ncu_gpc_cycles"] * np.exp((x_score / cross_scale) @ cross_beta)
    # Use the crosswalk only as an explicit interval/consistency companion;
    # retaining direct duration as the center avoids overfitting a clock map.
    cross_duration = np.sqrt(np.maximum(predictions["ncu_duration_ns"], 1.0) * np.maximum(cross_score, 1.0))
    predictions["ncu_duration_ns"] = cross_duration
    direct_duration_beta, direct_duration_scale = fit_ridge(
        x_train,
        np.log(np.maximum(duration, 1.0) / np.maximum(train_bases["ncu_duration_ns"], 1.0)),
        args.ridge,
    )
    direct_train_duration = train_bases["ncu_duration_ns"] * np.exp(
        (x_train / direct_duration_scale) @ direct_duration_beta
    )
    cross_train_duration = np.sqrt(np.maximum(direct_train_duration, 1.0) * np.maximum(cross_train, 1.0))
    # The direct surface and the crosswalk are both independently fit; only
    # their residual is used for interval width.
    residual_q90["ncu_duration_ns"] = float(np.quantile(np.abs(cross_train_duration - duration), 0.90))

    # Nearest independent stage state controls coverage provenance and interval
    # width.  It is not a binary OOD gate.
    feature_scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    feature_scale[feature_scale == 0.0] = 1.0
    train_norm = x_train[:, 1:] / feature_scale
    score_norm = x_score[:, 1:] / feature_scale
    distances = np.asarray([
        float(np.min(np.linalg.norm(train_norm - item, axis=1))) for item in score_norm
    ])
    distance_factor = 1.0 + np.minimum(distances / 3.0, 8.0)
    for target in TARGETS:
        width = residual_q90[target] * distance_factor
        intervals[target] = (
            np.maximum(predictions[target] - width, 0.0),
            predictions[target] + width,
        )

    archetype_results: list[dict | None] = [None] * len(score_rows)
    if args.archetype_matrix is not None or args.shape_specs is not None:
        if args.archetype_matrix is None or args.shape_specs is None:
            raise RuntimeError("both --archetype-matrix and --shape-specs are required")
        matrix = _load_archetype_matrix(args.archetype_matrix)
        specs = _load_shape_specs(args.shape_specs)
        for index, row in enumerate(score_rows):
            result = _flash_bwd_archetype_projection(
                row["workload"], _shape_spec_for_workload(specs, row["workload"]), matrix
            )
            archetype_results[index] = result
            if result is None or result.get("status") != "candidate":
                continue
            for target in TARGETS:
                predictions[target][index] = float(result["predictions"][target])
                low, high = result["intervals"][target]
                intervals[target] = (intervals[target][0], intervals[target][1])
                intervals[target][0][index] = float(low)
                intervals[target][1][index] = float(high)

    output_rows = []
    for index, row in enumerate(score_rows):
        exact = archetype_results[index] is not None and archetype_results[index].get("status") == "candidate"
        item = {
            "workload": row["workload"],
            "status": "candidate" if exact or distances[index] <= args.coverage_distance else "candidate_fallback",
            "coverage_status": "exact_shape" if exact else "stage_frontier_covered",
            "projection_path": "independent_same_shape_archetype" if exact else "stage_max_plus_efficiency_surface",
            "feature_distance": float(distances[index]),
            "distance_factor": float(distance_factor[index]),
            "stage_features": {name: float(score_maps[index][name]) for name in names},
        }
        if not exact and distances[index] > args.coverage_distance:
            item["coverage_status"] = "stage_frontier_fallback"
        if archetype_results[index] is not None:
            item["archetype_projection_status"] = archetype_results[index].get("status")
            item["archetype_reason"] = archetype_results[index].get("reason")
        for target in TARGETS:
            actual = _float(row, target, float("nan"))
            predicted = float(predictions[target][index])
            low, high = intervals[target]
            item[f"{target}_actual"] = actual
            item[f"{target}_predicted"] = predicted
            item[f"{target}_abs_error"] = abs(predicted - actual)
            item[f"{target}_ape"] = abs(predicted - actual) / actual if math.isfinite(actual) and actual > 0 else None
            item[f"{target}_interval_low"] = float(low[index])
            item[f"{target}_interval_high"] = float(high[index])
        output_rows.append(item)

    report = {
        "schema": "sm86_stage_interval_projection_v1",
        "train_campaign": str(args.train_campaign),
        "score_campaign": str(args.score_campaign),
        "holdout_family": args.holdout_family,
        "feature_names": names,
        "targets_are_score_only": True,
        "model": {
            "stage_demand": "semantic compute/memory/control/sync populations plus trace sectors",
            "frontier": "max(sim_gpc_envelope,event) for GPC; event/SM_clock for duration",
            "fit": "independent-cohort log endpoint/frontier efficiency surface",
            "composition": "max-plus frontier with learned overlap/idle/cache/ABI residual",
            "clock": "duration/GPC log crosswalk is consistency metadata, not a common-clock claim",
            "coverage": "candidate or candidate_fallback for every score row; interval width grows with stage distance",
        },
        "metrics": {target: _metric(output_rows, target) for target in TARGETS},
        "model_seconds": time.perf_counter() - started,
        "score_replay_wall_seconds_sum": float(sum(_float(row, "replay_wall_seconds") for row in score_rows)),
        "projection_only_speedup_vs_replay_wall": float(sum(_float(row, "replay_wall_seconds") for row in score_rows) / max(time.perf_counter() - started, 1e-12)),
        "residual_q90": residual_q90,
        "predictions": output_rows,
        "coefficients": coefficients,
        "crosswalk_coefficients": cross_beta.tolist(),
        "coverage_summary": {
            "workload_count": len(output_rows),
            "status_counts": {
                status: sum(1 for item in output_rows if item["status"] == status)
                for status in ("candidate", "candidate_fallback")
            },
        },
        "limitations": [
            "semantic stage classes are not physical SM86 pipe identities",
            "cache features are traffic/phase surrogates, not owner/read-from/DRAM proofs",
            "ABI and endpoint effects remain latent residuals",
            "cohort holdout is related-family evidence, not independent silicon generalization",
            "projection time excludes capture and trace production",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-campaign", type=Path, required=True)
    parser.add_argument("--score-campaign", type=Path, required=True)
    parser.add_argument("--holdout-family", choices=("base", "large"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archetype-matrix", type=Path)
    parser.add_argument("--shape-specs", type=Path)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--coverage-distance", type=float, default=3.0)
    parser.add_argument(
        "--dynamic-trace",
        action="store_true",
        help="walk protobufs for richer traffic features (slower; default is metadata-only)",
    )
    parser.add_argument(
        "--dynamic-sample",
        type=int,
        default=0,
        help="when --dynamic-trace is set, inspect only this many threadblocks per workload",
    )
    args = parser.parse_args()
    if args.ridge <= 0 or not math.isfinite(args.ridge):
        parser.error("--ridge must be finite and positive")
    if args.dynamic_sample < 0:
        parser.error("--dynamic-sample must be nonnegative")
    report = run(args)
    print(json.dumps({
        "holdout_family": report["holdout_family"],
        "score_workloads": len(report["predictions"]),
        "metrics": report["metrics"],
        "model_seconds": report["model_seconds"],
        "projection_only_speedup_vs_replay_wall": report["projection_only_speedup_vs_replay_wall"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
