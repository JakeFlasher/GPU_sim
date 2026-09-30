#!/usr/bin/env python3
"""Fast, holdout-only projection of SM86 trace features to NCU metrics.

This is deliberately a metric surrogate, not a hardware-mechanism simulator.
It consumes only trace/control/grid features and never uses NCU counters as
features. NCU labels are used only for a cohort-disjoint training partition;
the other cohort is scored as a fresh holdout. The optional
``frontier-archetype`` mode adds independent shape-anchor labels behind a
strict same-configuration bracket for a named workload family. The cohorts may
contain related kernel families, so this is not independent-generalization
proof. The projection is a candidate for an NCU-oriented fast path, not a
replacement for native replay or a claim about physical pipes/cache ownership.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


TARGETS = ("ncu_gpc_cycles", "ncu_duration_ns")
FORBIDDEN_FEATURE_PREFIXES = ("ncu_", "smsp__inst_executed_pipe_")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _family(workload: str) -> str:
    return workload.split("__", 1)[0]


def _opcode_group(opcode: str) -> str:
    head = opcode.upper().split(".", 1)[0]
    if head.startswith(("LD", "ST", "ATOM", "RED", "LDS", "STS", "CP")):
        return "memory"
    if head.startswith(("HMMA", "IMMA", "DMMA", "WGMMA", "MMA")):
        return "tensor"
    if head.startswith(("MUFU", "RCP", "RSQ", "SIN", "COS", "EX2", "LG2")):
        return "sfu"
    if head.startswith(("BRA", "BRX", "BSSY", "BSYNC", "SSY", "BAR", "EXIT", "RET")):
        return "control"
    if head in {"NOP", "CS2R", "S2R", "S2UR", "ULDC"}:
        return "special"
    return "math"


_PB_READY = False
_THREADBLOCK = None


def _load_threadblock_binding():
    global _PB_READY, _THREADBLOCK
    if _PB_READY:
        return _THREADBLOCK
    binding_root = Path(__file__).resolve().parents[1] / "workloads" / "simbench_bw" / "_pb_generated"
    sys.path.insert(0, str(binding_root))
    import threadblock_pb2  # type: ignore[import-not-found]

    _THREADBLOCK = threadblock_pb2.threadblock
    _PB_READY = True
    return _THREADBLOCK


def _trace_dynamic_features(root: Path, static_groups: dict[int, str]) -> dict[str, float]:
    """Walk captured threadblock protobufs without invoking replay timing."""
    threadblock = _load_threadblock_binding()
    groups = {name: 0.0 for name in ("memory", "tensor", "sfu", "control", "special", "math")}
    records = active_lanes = memory_records = memory_bytes = sectors = 0.0
    warps = blocks = 0.0
    for path in sorted((root / "threadblocks").rglob("*.pb")):
        block = threadblock()
        block.ParseFromString(path.read_bytes())
        blocks += 1.0
        for warp in block.warps.values():
            warps += 1.0
            for instruction in warp.instructions:
                records += 1.0
                mask = (instruction.active_mask or 0) & (instruction.predicate_mask or 0)
                active = mask.bit_count()
                active_lanes += active
                groups[static_groups.get(instruction.pc, "special")] += 1.0
                if instruction.addresses:
                    memory_records += 1.0
                    for address in instruction.addresses:
                        width = max(int(address.data_width), 1)
                        memory_bytes += width * active
                        if address.compression_format == 0:
                            values = [value for value in address.addrs if value >= 0]
                            sectors += len({value >> 5 for value in values})
                        elif address.compression_format == 1:
                            values = [address.base_address + lane * address.stride for lane in range(active)]
                            sectors += len({value >> 5 for value in values})
                        else:
                            # Base-delta records carry enough information for a
                            # conservative one-sector lower bound without a
                            # second semantic reconstruction pass.
                            sectors += float(bool(active))
    return {
        "trace_records": records,
        "trace_active_lanes": active_lanes,
        "trace_memory_records": memory_records,
        "trace_memory_bytes": memory_bytes,
        "trace_memory_sectors": sectors,
        "trace_warps": warps,
        "trace_blocks": blocks,
        **{f"trace_dynamic_{name}": value for name, value in groups.items()},
    }


def trace_features(campaign: Path, workload: str) -> tuple[dict[str, float], float]:
    """Extract cheap features from trace metadata, not the timing replay."""
    root = campaign / workload / "trace"
    info_path = root / "extra_info" / "enhanced_execution_info.json"
    stats_path = root / "stats.csv"
    info = json.loads(info_path.read_text())
    with stats_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not info.get("kernels"):
        raise RuntimeError(f"missing trace metadata for {workload}")
    stat = rows[0]
    kernel = info["kernels"][0]
    instructions = kernel.get("instructions", [])
    static_count = max(len(instructions), 1)
    dynamic_insts = _float(stat, "total_insts")
    blocks = _float(stat, "#blocks")
    threads = _float(stat, "#threads")
    groups = {name: 0.0 for name in ("memory", "tensor", "sfu", "control", "special", "math")}
    stall_sum = 0.0
    stall_max = 0.0
    wait_count = 0.0
    barrier_sets = 0.0
    max_regular = 0.0
    max_uniform = 0.0
    for instruction in instructions:
        groups[_opcode_group(str(instruction.get("op_code", "UNKNOWN")))] += 1.0
        control = instruction.get("control_bits", {})
        stall = float(control.get("stall_count", 0) or 0)
        stall_sum += stall
        stall_max = max(stall_max, stall)
        wait_count += (int(control.get("wait_barrier_bits", 0) or 0).bit_count())
        barrier_sets += int(bool(control.get("is_new_read_barrier", False)))
        barrier_sets += int(bool(control.get("is_new_write_barrier", False)))
        registers = instruction.get("register_usage", {})
        max_regular = max(max_regular, float(registers.get("num_regular", 0) or 0))
        max_uniform = max(max_uniform, float(registers.get("num_uniform", 0) or 0))

    features: dict[str, float] = {
        "log_dynamic_insts": math.log1p(max(dynamic_insts, 0.0)),
        "log_blocks": math.log1p(max(blocks, 0.0)),
        "log_threads": math.log1p(max(threads, 0.0)),
        "static_count": float(static_count),
        "dynamic_insts_per_block": dynamic_insts / max(blocks, 1.0),
        "threads_per_block": threads / max(blocks, 1.0),
        "stall_sum_per_inst": stall_sum / static_count,
        "stall_max": stall_max,
        "wait_bits_per_inst": wait_count / static_count,
        "barrier_sets_per_inst": barrier_sets / static_count,
        "max_regular": max_regular,
        "max_uniform": max_uniform,
    }
    for name, count in groups.items():
        dynamic_group = dynamic_insts * count / static_count
        features[f"log_dynamic_{name}"] = math.log1p(max(dynamic_group, 0.0))
        features[f"dynamic_{name}_per_block"] = dynamic_group / max(blocks, 1.0)
        features[f"fraction_{name}"] = count / static_count
    static_groups = {
        int(instruction.get("pc_num_dec", 0)): _opcode_group(str(instruction.get("op_code", "UNKNOWN")))
        for instruction in instructions
    }
    features.update(_trace_dynamic_features(root, static_groups))
    features["log_trace_records"] = math.log1p(features["trace_records"])
    features["log_trace_memory_sectors"] = math.log1p(features["trace_memory_sectors"])
    features["trace_memory_bytes_per_record"] = features["trace_memory_bytes"] / max(
        features["trace_memory_records"], 1.0
    )
    if any(name.startswith(prefix) for name in features for prefix in FORBIDDEN_FEATURE_PREFIXES):
        raise RuntimeError("feature set accidentally contains a forbidden NCU field")
    return features, dynamic_insts


def load_rows(campaign: Path) -> list[dict[str, str]]:
    with (campaign / "performance.csv").open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("status") == "completed"]


def _trace_stats_shape(campaign: Path | None, workload: str) -> dict[str, float]:
    if campaign is None:
        return {}
    path = campaign / workload / "trace" / "stats.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        return {}
    blocks = _float(row, "#blocks")
    threads = _float(row, "#threads")
    instructions = _float(row, "total_insts")
    return {
        "log_trace_blocks": math.log1p(blocks),
        "trace_threads_per_block": threads / max(blocks, 1.0),
        "log_trace_insts_per_block": math.log1p(instructions / max(blocks, 1.0)),
    }


def frontier_features(row: dict[str, str], campaign: Path | None = None) -> dict[str, float]:
    """Features emitted by the sparse replay/frontier path, never by NCU."""
    event = _float(row, "sim_event_cycles")
    envelope = _float(row, "sim_gpc_envelope_cycles")
    issue = _float(row, "sim_last_issue_cycles")
    instructions = _float(row, "sim_warp_instructions")
    return {
        "sim_event_cycles": event,
        "sim_envelope_delta": max(envelope - event, 0.0),
        "sim_tail_delta": max(event - issue, 0.0),
        "log_sim_warp_instructions": math.log1p(max(instructions, 0.0)),
        **_trace_stats_shape(campaign, row["workload"]),
    }


def frontier_normalized_features(row: dict[str, str]) -> dict[str, float]:
    """Scale frontier work by dynamic population before archetype matching."""
    event = _float(row, "sim_event_cycles")
    envelope = _float(row, "sim_gpc_envelope_cycles")
    issue = _float(row, "sim_last_issue_cycles")
    instructions = max(_float(row, "sim_warp_instructions"), 1.0)
    return {
        "log_sim_warp_instructions": math.log1p(instructions),
        "event_cycles_per_instruction": event / instructions,
        "envelope_delta_per_instruction": max(envelope - event, 0.0) / instructions,
        "tail_cycles_per_instruction": max(event - issue, 0.0) / instructions,
    }


def static_stage_signature(campaign: Path, workload: str) -> dict[str, float]:
    """Return a cheap static semantic signature for independent neighbor transfer.

    This deliberately reads only captured instruction metadata.  It does not
    walk protobufs, read NCU counters, or map a class to a physical pipe.
    """
    root = campaign / workload / "trace"
    info = json.loads((root / "extra_info" / "enhanced_execution_info.json").read_text())
    kernels = info.get("kernels", [])
    if not kernels:
        raise RuntimeError(f"trace has no kernel metadata: {root}")
    instructions = kernels[0].get("instructions", [])
    static_count = max(len(instructions), 1)
    groups = {name: 0.0 for name in ("memory", "tensor", "sfu", "control", "special", "math")}
    barriers = wait_bits = stall_sum = 0.0
    max_regular = max_uniform = 0.0
    for instruction in instructions:
        groups[_opcode_group(str(instruction.get("op_code", "UNKNOWN")))] += 1.0
        control = instruction.get("control_bits", {})
        barriers += float(bool(control.get("is_new_read_barrier", False)))
        barriers += float(bool(control.get("is_new_write_barrier", False)))
        wait_bits += float(int(control.get("wait_barrier_bits", 0) or 0).bit_count())
        stall_sum += _float({"v": str(control.get("stall_count", 0))}, "v")
        registers = instruction.get("register_usage", {})
        max_regular = max(max_regular, _float({"v": str(registers.get("num_regular", 0))}, "v"))
        max_uniform = max(max_uniform, _float({"v": str(registers.get("num_uniform", 0))}, "v"))
    result = {f"fraction_{name}": value / static_count for name, value in groups.items()}
    result.update({
        "barrier_density": barriers / static_count,
        "wait_density": wait_bits / static_count,
        "stall_mean": stall_sum / static_count,
        "max_regular": max_regular,
        "max_uniform": max_uniform,
    })
    return result


def _neighbor_transfer(
    train_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    train_campaign: Path,
    score_campaign: Path,
    target: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict an endpoint ratio from independent static nearest neighbors."""
    train_maps = [static_stage_signature(train_campaign, row["workload"]) for row in train_rows]
    score_maps = [static_stage_signature(score_campaign, row["workload"]) for row in score_rows]
    names = sorted(train_maps[0])
    if any(sorted(item) != names for item in train_maps + score_maps):
        raise RuntimeError("neighbor static signature schema differs")
    x_train = np.asarray([[item[name] for name in names] for item in train_maps], dtype=float)
    x_score = np.asarray([[item[name] for name in names] for item in score_maps], dtype=float)
    scale = np.sqrt(np.mean(x_train * x_train, axis=0))
    scale[scale == 0.0] = 1.0
    x_train /= scale
    x_score /= scale
    base_train = np.asarray([
        max(_float(row, "sim_gpc_envelope_cycles"), _float(row, "sim_event_cycles"))
        if target == "ncu_gpc_cycles"
        else max(_float(row, "sim_event_cycles"), 1.0) * (1000.0 / 1350.0)
        for row in train_rows
    ], dtype=float)
    base_score = np.asarray([
        max(_float(row, "sim_gpc_envelope_cycles"), _float(row, "sim_event_cycles"))
        if target == "ncu_gpc_cycles"
        else max(_float(row, "sim_event_cycles"), 1.0) * (1000.0 / 1350.0)
        for row in score_rows
    ], dtype=float)
    ratios = np.asarray([
        _float(row, target) / max(base, 1.0)
        for row, base in zip(train_rows, base_train)
    ], dtype=float)
    predictions = []
    distances = []
    neighbors = max(1, min(int(k), len(train_rows)))
    for item in x_score:
        distance = np.linalg.norm(x_train - item, axis=1)
        indices = np.argsort(distance)[:neighbors]
        weights = 1.0 / (distance[indices] + 0.05)
        predictions.append(base_score[len(predictions)] * float(np.sum(weights * ratios[indices]) / np.sum(weights)))
        distances.append(float(distance[indices[0]]))
    return np.asarray(predictions, dtype=float), np.asarray(distances, dtype=float)


def _load_shape_specs(path: Path) -> dict:
    """Load shape metadata separately from the NCU target campaign.

    Shape metadata is an input contract (the launch dimensions), not a
    measurement label.  Keeping it in a sidecar prevents the archetype path
    from reverse-engineering a target shape from its NCU output.
    """
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"shape spec must be an object: {path}")
    return value


def _load_archetype_matrix(path: Path) -> list[dict[str, str]]:
    """Load independent archetype anchors with NCU labels.

    The caller is responsible for passing a matrix whose rows are independent
    calibration launches.  Rows explicitly labelled as score targets are
    rejected here so a target observation cannot silently enter the fit.
    """
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty archetype matrix: {path}")
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        if "score_target" in role or role == "target":
            raise RuntimeError("archetype matrix contains a score target row")
        for key in ("BH", "Sq", "Sk", "D", *TARGETS):
            if key in TARGETS:
                value = _float(row, key)
            else:
                value = _float(row, key)
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite archetype field {key}")
    return rows


def _shape_spec_for_workload(specs: dict, workload: str) -> dict | None:
    target = specs.get("target")
    if isinstance(target, dict) and target.get("workload") == workload:
        return target
    for candidate in specs.get("workloads", []):
        if isinstance(candidate, dict) and candidate.get("workload") == workload:
            return candidate
    return None


def _flash_bwd_archetype_projection(
    workload: str,
    shape_spec: dict | None,
    matrix: list[dict[str, str]],
) -> dict | None:
    """Interpolate a target only inside an independent same-shape bracket.

    The matching dimensions are deliberately strict.  A different batch/head
    configuration can have nearly identical instruction populations but a
    different NCU duration, so an unmatched shape falls back to the generic
    frontier/trace path rather than being silently interpolated here.
    """
    if shape_spec is None:
        return None
    archetype = str(shape_spec.get("archetype", "flash_attn_bwd_bf16_sm80"))
    required = ("BH", "Sq", "Sk", "D")
    if any(key not in shape_spec for key in required):
        return {
            "archetype": archetype,
            "status": "unmatched_shape",
            "reason": "shape_spec_missing_dimension",
        }
    target = {key: _float({key: str(shape_spec[key])}, key) for key in required}
    anchors = []
    for row in matrix:
        if str(row.get("archetype", archetype)) != archetype:
            continue
        # Matrix loader rejects target-labelled rows; retain the explicit
        # role check as a defense against hand-edited alpha artifacts.
        if "score_target" in str(row.get("role", "")).lower():
            continue
        if all(_float(row, key) == target[key] for key in ("BH", "Sq", "D")):
            anchors.append(row)
    anchors.sort(key=lambda row: _float(row, "Sk"))
    if not anchors:
        return {
            "archetype": archetype,
            "status": "unmatched_shape",
            "reason": "no_matching_BH_Sq_D_anchor",
        }
    exact = [row for row in anchors if _float(row, "Sk") == target["Sk"]]
    if exact:
        row = exact[0]
        return {
            "archetype": archetype,
            "status": "candidate",
            "path": "independent_exact_anchor",
            "feature_distance": 0.0,
            "anchor_low": row.get("name", "exact"),
            "anchor_high": row.get("name", "exact"),
            "predictions": {target_name: _float(row, target_name) for target_name in TARGETS},
            "intervals": {
                target_name: (_float(row, target_name), _float(row, target_name))
                for target_name in TARGETS
            },
        }
    # With three or more independent anchors, fit a low-order local response
    # surface instead of throwing away the extra shape evidence and using only
    # the nearest pair.  This remains interpolation-only: the target must lie
    # inside the observed Sk hull, and the target label is never in anchors.
    if len(anchors) >= 3:
        min_sk = _float(anchors[0], "Sk")
        max_sk = _float(anchors[-1], "Sk")
        if not (min_sk <= target["Sk"] <= max_sk):
            return {
                "archetype": archetype,
                "status": "unmatched_shape",
                "reason": "target_outside_matching_Sk_hull",
                "feature_distance": 1.0,
            }
        anchor_sk = np.asarray([_float(row, "Sk") for row in anchors], dtype=float)
        # Center/scale before fitting to make the polynomial conditioning
        # explicit for long-sequence shapes.
        center = float(np.mean(anchor_sk))
        scale = max(float(np.ptp(anchor_sk)), 1.0)
        normalized_sk = (anchor_sk - center) / scale
        target_sk = (target["Sk"] - center) / scale
        predictions = {}
        intervals = {}
        for target_name in TARGETS:
            values = np.asarray([_float(row, target_name) for row in anchors], dtype=float)
            coefficients = np.polyfit(normalized_sk, values, deg=2)
            predictions[target_name] = float(np.polyval(coefficients, target_sk))
            intervals[target_name] = (float(np.min(values)), float(np.max(values)))
        return {
            "archetype": archetype,
            "status": "candidate",
            "path": "independent_same_shape_quadratic_lattice",
            "feature_distance": min(
                abs(target["Sk"] - min_sk), abs(max_sk - target["Sk"])
            ) / max(max_sk - min_sk, 1.0),
            "polynomial_degree": 2,
            "anchor_count": len(anchors),
            "anchor_low": anchors[0].get("name", "lower"),
            "anchor_high": anchors[-1].get("name", "upper"),
            "anchor_low_Sk": min_sk,
            "anchor_high_Sk": max_sk,
            "predictions": predictions,
            "intervals": intervals,
        }
    lower = [row for row in anchors if _float(row, "Sk") < target["Sk"]]
    upper = [row for row in anchors if _float(row, "Sk") > target["Sk"]]
    if not lower or not upper:
        return {
            "archetype": archetype,
            "status": "unmatched_shape",
            "reason": "target_outside_matching_Sk_bracket",
            "feature_distance": 1.0,
        }
    low = lower[-1]
    high = upper[0]
    low_sk = _float(low, "Sk")
    high_sk = _float(high, "Sk")
    fraction = (target["Sk"] - low_sk) / max(high_sk - low_sk, 1.0)
    predictions = {}
    intervals = {}
    for target_name in TARGETS:
        low_value = _float(low, target_name)
        high_value = _float(high, target_name)
        predictions[target_name] = low_value + fraction * (high_value - low_value)
        # The bracket is intentionally conservative: it is an envelope of
        # observed independent endpoints, not a confidence interval claim.
        intervals[target_name] = (min(low_value, high_value), max(low_value, high_value))
    return {
        "archetype": archetype,
        "status": "candidate",
        "path": "independent_same_shape_linear_bracket",
        "feature_distance": min(fraction, 1.0 - fraction),
        "interpolation_fraction": fraction,
        "anchor_low": low.get("name", "lower"),
        "anchor_high": high.get("name", "upper"),
        "anchor_low_Sk": low_sk,
        "anchor_high_Sk": high_sk,
        "predictions": predictions,
        "intervals": intervals,
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[0] = 1.0
    scale[scale == 0.0] = 1.0
    xs = x / scale
    gram = xs.T @ xs + ridge * np.eye(xs.shape[1])
    beta = np.linalg.solve(gram, xs.T @ y)
    return beta, scale


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    train_rows = load_rows(args.train_campaign)
    score_rows = load_rows(args.score_campaign)
    train_rows = [row for row in train_rows if _family(row["workload"]) != args.holdout_family]
    score_rows = [row for row in score_rows if _family(row["workload"]) == args.holdout_family]
    if not train_rows or not score_rows:
        raise RuntimeError("empty train or score family partition")

    feature_names: list[str] | None = None
    train_features = []
    score_features = []
    for row in train_rows:
        feature_map = (
            frontier_features(row, args.train_campaign)
            if args.feature_mode in ("frontier", "frontier-archetype")
            else frontier_normalized_features(row)
            if args.feature_mode == "frontier-normalized"
            else trace_features(args.train_campaign, row["workload"])[0]
        )
        feature_names = feature_names or sorted(feature_map)
        train_features.append([1.0] + [feature_map[name] for name in feature_names])
    for row in score_rows:
        feature_map = (
            frontier_features(row, args.score_campaign)
            if args.feature_mode in ("frontier", "frontier-archetype")
            else frontier_normalized_features(row)
            if args.feature_mode == "frontier-normalized"
            else trace_features(args.score_campaign, row["workload"])[0]
        )
        if sorted(feature_map) != feature_names:
            raise RuntimeError("train/score feature schema differs")
        score_features.append([1.0] + [feature_map[name] for name in feature_names])

    x_train = np.asarray(train_features, dtype=float)
    x_score = np.asarray(score_features, dtype=float)
    predictions: dict[str, np.ndarray] = {}
    coefficients: dict[str, list[float]] = {}
    prediction_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    calibration_residual_q90: dict[str, float] = {}
    cache_transfer_ratios: dict[str, float] = {}
    neighbor_predictions: dict[str, np.ndarray] = {}
    neighbor_distances: np.ndarray | None = None
    if args.cache_transfer_campaign is not None:
        transfer_rows = load_rows(args.cache_transfer_campaign)
        by_workload = {row["workload"]: row for row in transfer_rows}
        for target in TARGETS:
            ratios = [
                _float(by_workload[row["workload"]], target) / _float(row, target)
                for row in train_rows
                if row["workload"] in by_workload and _float(row, target) > 0
            ]
            if not ratios:
                raise RuntimeError("cache-transfer calibration has no training-family overlap")
            cache_transfer_ratios[target] = float(np.median(ratios))
    for target in TARGETS:
        y_train = np.asarray([_float(row, target) for row in train_rows], dtype=float)
        if args.feature_mode in ("frontier", "frontier-normalized", "frontier-archetype"):
            beta, scale = fit_ridge(x_train, y_train, args.ridge)
            predictions[target] = (x_score / scale) @ beta
            train_pred = (x_train / scale) @ beta
        else:
            # Log-target fitting keeps the direct trace projection stable
            # across kernels whose work differs by orders of magnitude.
            beta, scale = fit_ridge(x_train, np.log1p(y_train), args.ridge)
            predictions[target] = np.expm1((x_score / scale) @ beta)
            train_pred = np.expm1((x_train / scale) @ beta)
        if target in cache_transfer_ratios:
            predictions[target] *= cache_transfer_ratios[target]
            train_pred *= cache_transfer_ratios[target]
        residual_q90 = float(np.quantile(np.abs(train_pred - y_train), 0.90))
        calibration_residual_q90[target] = residual_q90
        prediction_intervals[target] = (
            np.maximum(predictions[target] - residual_q90, 0.0),
            predictions[target] + residual_q90,
        )
        coefficients[target] = beta.tolist()

    # Optional independent nearest-stage residual transfer.  It is disabled
    # by default so the promoted projector is byte-for-byte behaviorally
    # unchanged.  Training labels come only from the independent cohort; score
    # NCU values are never consulted before the final report rows.
    neighbor_alphas = {
        "ncu_gpc_cycles": float(args.neighbor_correction_gpc),
        "ncu_duration_ns": float(args.neighbor_correction_duration),
    }
    if any(value > 0.0 for value in neighbor_alphas.values()):
        for target in TARGETS:
            if neighbor_alphas[target] <= 0.0:
                continue
            neighbor, distances = _neighbor_transfer(
                train_rows,
                score_rows,
                args.train_campaign,
                args.score_campaign,
                target,
                args.neighbor_k,
            )
            neighbor_predictions[target] = neighbor
            neighbor_distances = distances
            predictions[target] = (
                (1.0 - neighbor_alphas[target]) * predictions[target]
                + neighbor_alphas[target] * neighbor
            )

    feature_scale = np.sqrt(np.mean(x_train[:, 1:] * x_train[:, 1:], axis=0))
    feature_scale[feature_scale == 0.0] = 1.0
    normalized_train = x_train[:, 1:] / feature_scale
    normalized_score = x_score[:, 1:] / feature_scale
    feature_distances = np.asarray([
        float(np.min(np.linalg.norm(normalized_train - row, axis=1)))
        for row in normalized_score
    ])
    ood_status = [
        "candidate" if distance <= args.ood_threshold else "candidate_fallback"
        for distance in feature_distances
    ]
    # Feature distance is a useful uncertainty diagnostic, not a reason to
    # drop a workload.  Keep a finite prediction for every row and widen the
    # residual interval when the row is outside the local feature hull.
    distance_factor = 1.0 + np.maximum(feature_distances - args.ood_threshold, 0.0) / max(args.ood_threshold, 1.0)
    for target in TARGETS:
        predictions[target] = np.maximum(predictions[target], 0.0)
        width = calibration_residual_q90[target] * distance_factor
        prediction_intervals[target] = (
            np.maximum(predictions[target] - width, 0.0),
            predictions[target] + width,
        )

    # Optional hierarchical override.  The generic frontier fit remains the
    # baseline for every row; only a workload with a strict independent shape
    # bracket receives the archetype prediction.  This keeps the generic hull
    # distance visible while avoiding silent shape extrapolation for a known
    # family.
    archetype_results: list[dict | None] = [None] * len(score_rows)
    if args.feature_mode == "frontier-archetype":
        if args.archetype_matrix is None or args.shape_specs is None:
            raise RuntimeError(
                "frontier-archetype requires --archetype-matrix and --shape-specs"
            )
        archetype_matrix = _load_archetype_matrix(args.archetype_matrix)
        shape_specs = _load_shape_specs(args.shape_specs)
        for index, row in enumerate(score_rows):
            spec = _shape_spec_for_workload(shape_specs, row["workload"])
            archetype_results[index] = _flash_bwd_archetype_projection(
                row["workload"], spec, archetype_matrix
            )
            result = archetype_results[index]
            if result is None or result.get("status") != "candidate":
                continue
            for target in TARGETS:
                transfer = (
                    cache_transfer_ratios.get(target, 1.0)
                    if args.archetype_cache_transfer == "apply"
                    else 1.0
                )
                predictions[target][index] = float(result["predictions"][target]) * transfer
                low, high = result["intervals"][target]
                prediction_intervals[target][0][index] = float(low) * transfer
                prediction_intervals[target][1][index] = float(high) * transfer

    rows = []
    metrics = {}
    for index, row in enumerate(score_rows):
        item = {"workload": row["workload"]}
        item["feature_distance"] = float(feature_distances[index])
        item["projection_status"] = ood_status[index]
        item["status"] = ood_status[index]
        item["coverage_status"] = (
            "fallback_covered" if ood_status[index] == "candidate_fallback" else "frontier_covered"
        )
        item["fallback_distance_factor"] = float(distance_factor[index])
        if neighbor_distances is not None:
            item["neighbor_stage_distance"] = float(neighbor_distances[index])
        for target in TARGETS:
            if target in neighbor_predictions:
                item[f"{target}_neighbor_predicted"] = float(neighbor_predictions[target][index])
                item[f"{target}_neighbor_blend_alpha"] = neighbor_alphas[target]
        result = archetype_results[index]
        if result is not None:
            item["generic_projection_status"] = ood_status[index]
            item["archetype"] = result.get("archetype")
            item["archetype_projection_status"] = result.get("status")
            item["archetype_projection_path"] = result.get("path")
            item["archetype_feature_distance"] = result.get("feature_distance")
            item["archetype_reason"] = result.get("reason")
            item["archetype_anchor_low"] = result.get("anchor_low")
            item["archetype_anchor_high"] = result.get("anchor_high")
            if result.get("status") == "candidate":
                item["projection_status"] = "candidate"
                item["status"] = "candidate"
                item["projection_path"] = result.get("path")
                item["coverage_status"] = "exact_shape"
        for target in TARGETS:
            actual = _float(row, target)
            predicted = float(predictions[target][index])
            item[f"{target}_actual"] = actual
            item[f"{target}_predicted"] = predicted
            item[f"{target}_abs_error"] = abs(predicted - actual)
            item[f"{target}_ape"] = abs(predicted - actual) / actual if actual else None
            low, high = prediction_intervals[target]
            item[f"{target}_interval_low"] = float(low[index])
            item[f"{target}_interval_high"] = float(high[index])
        rows.append(item)
    for target in TARGETS:
        actual = np.asarray([item[f"{target}_actual"] for item in rows])
        predicted = np.asarray([item[f"{target}_predicted"] for item in rows])
        metrics[target] = {
            "mae": float(np.mean(np.abs(predicted - actual))),
            "mape": float(np.mean(np.abs(predicted - actual) / actual)),
            "max_abs_error": float(np.max(np.abs(predicted - actual))),
        }

    elapsed = time.perf_counter() - started
    replay_seconds = sum(_float(row, "replay_wall_seconds") for row in score_rows)
    report = {
        "schema": "sm86_ncu_projection_surrogate_v2",
        "train_campaign": str(args.train_campaign),
        "score_campaign": str(args.score_campaign),
        "holdout_family": args.holdout_family,
        "feature_mode": args.feature_mode,
        "cache_transfer_campaign": str(args.cache_transfer_campaign) if args.cache_transfer_campaign else None,
        "cache_transfer_ratios": cache_transfer_ratios,
        "train_workloads": [row["workload"] for row in train_rows],
        "score_workloads": [row["workload"] for row in score_rows],
        "feature_names": feature_names,
        "feature_contract": (
            "population-normalized sparse frontiers only; no NCU or pipe counters"
            if args.feature_mode == "frontier-normalized"
            else "sparse replay frontiers plus independent same-shape archetype anchors; no NCU or pipe counters"
            if args.feature_mode == "frontier-archetype"
            else "sparse replay frontiers only; no NCU or pipe counters"
            if args.feature_mode == "frontier"
            else "trace extra_info + trace stats only; no NCU or pipe counters"
        ),
        "targets": list(TARGETS),
        "ridge": args.ridge,
        "ood_threshold": args.ood_threshold,
        "neighbor_correction_gpc": args.neighbor_correction_gpc,
        "neighbor_correction_duration": args.neighbor_correction_duration,
        "neighbor_k": args.neighbor_k,
        "coverage_policy": "every score row receives a candidate or candidate_fallback prediction; no OOD refusal",
        "calibration_residual_q90": calibration_residual_q90,
        "metrics": metrics,
        "model_seconds": elapsed,
        "score_replay_wall_seconds_sum": replay_seconds,
        "projection_only_speedup_vs_replay_wall": replay_seconds / elapsed if elapsed else None,
        "predictions": rows,
        "coefficients": coefficients,
        "limitations": [
            "projection-only timing excludes NVBit capture and trace production",
            "holdout is cohort-disjoint but small and may contain related kernels",
            "NCU labels are independent targets and never model features",
            "surrogate predicts measurement metrics, not physical pipe/cache state",
        ],
    }
    if any(value > 0.0 for value in neighbor_alphas.values()):
        report["limitations"].append(
            "neighbor blend is an optional sensitivity path; alpha must be fixed from a training-only policy before using it as a promoted score"
        )
    if args.feature_mode == "frontier-archetype":
        report["archetype_contract"] = {
            "matrix": str(args.archetype_matrix),
            "shape_specs": str(args.shape_specs),
            "target_labels_excluded_from_matrix": True,
            "resolved_rows": sum(
                1
                for result in archetype_results
                if result and result.get("status") == "candidate"
            ),
            "fallback_rows": sum(
                1
                for result in archetype_results
                if result and result.get("status") != "candidate"
            ),
            "unmatched_rows": sum(1 for result in archetype_results if result is None),
            "interpolation_rule": (
                "strict BH/Sq/D match; linear Sk interpolation only inside "
                "independent bracket"
            ),
            "archetype_cache_transfer": args.archetype_cache_transfer,
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
    parser.add_argument(
        "--feature-mode",
        choices=("trace", "frontier", "frontier-normalized", "frontier-archetype"),
        default="trace",
    )
    parser.add_argument(
        "--archetype-matrix",
        type=Path,
        help="independent shape-anchor matrix used by frontier-archetype",
    )
    parser.add_argument(
        "--shape-specs",
        type=Path,
        help="shape metadata sidecar used by frontier-archetype",
    )
    parser.add_argument(
        "--archetype-cache-transfer",
        choices=("apply", "skip"),
        default="apply",
        help=(
            "apply the disjoint-family warm/cold ratio to archetype labels, "
            "or skip it when the archetype matrix is already in the score domain"
        ),
    )
    parser.add_argument(
        "--cache-transfer-campaign",
        type=Path,
        help="optional disjoint-family warm/cold calibration campaign; applies median target ratios",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ood-threshold", type=float, default=3.0)
    parser.add_argument(
        "--neighbor-correction-gpc",
        type=float,
        default=0.0,
        help="optional independent static-neighbor blend for GPC (default disabled)",
    )
    parser.add_argument(
        "--neighbor-correction-duration",
        type=float,
        default=0.0,
        help="optional independent static-neighbor blend for duration (default disabled)",
    )
    parser.add_argument("--neighbor-k", type=int, default=1)
    args = parser.parse_args()
    for name in ("neighbor_correction_gpc", "neighbor_correction_duration"):
        value = getattr(args, name)
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            parser.error(f"--{name.replace('_', '-')} must be within [0,1]")
    if args.neighbor_k <= 0:
        parser.error("--neighbor-k must be positive")
    report = run(args)
    print(json.dumps({
        "holdout_family": report["holdout_family"],
        "score_workloads": len(report["score_workloads"]),
        "metrics": report["metrics"],
        "model_seconds": report["model_seconds"],
        "projection_only_speedup_vs_replay_wall": report["projection_only_speedup_vs_replay_wall"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
