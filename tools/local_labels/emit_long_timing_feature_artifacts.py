#!/usr/bin/env python3
"""Attach deploy-legal feature artifacts to the long-duration timing cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.nvbit_tracer.runtime_env import tracer_runtime_env  # noqa: E402


DEFAULT_COHORT = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_cohort"
    / "ac4_long_duration_cohort_sm_86.json"
)
DEFAULT_OUTPUT = DEFAULT_COHORT
DEFAULT_FEATURE_DIR = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort" / "feature_artifacts"
)
DEFAULT_TRACE_ROOT = (
    REPO_ROOT / "data" / "local_labels" / "timing_cohort" / "nvbit_launch_traces"
)
DEFAULT_CACHE_HIT_PREDICTIONS = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "timing_predictions"
    / "timing_cache_hit_predictions_sm_86.json"
)
DEFAULT_STATIC_WORK = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "static_work"
    / "gemm_hmma_bf16_sm80_sm_86.json"
)
DEFAULT_STATIC_FEATURES = (
    REPO_ROOT
    / "data"
    / "local_labels"
    / "static_sass_features"
    / "gemm_hmma_bf16_sm80_sm_86.json"
)
DEFAULT_TRACER_SO = REPO_ROOT / "tools" / "nvbit_tracer" / "tracer_tool" / "tracer_tool.so"
DEFAULT_L1_HEAD = REPO_ROOT / "data" / "local_labels" / "sm_86_l1_head.json"
DEFAULT_L2_HEAD = REPO_ROOT / "data" / "local_labels" / "sm_86_v5_head.json"
DEFAULT_FAMILY_BUCKETS = (
    REPO_ROOT / "data" / "local_labels" / "sm120_cohort_family_buckets.toml"
)

TRACE_INSTR_BEGIN = 168
TRACE_INSTR_END = 174
OBSERVED_TRACE_MODE = "bounded_observed_global_store_trace"
OBSERVED_SECTOR_PRODUCER = "nvbit_threadblock_proto_summarize_launch_v1"
CACHE_PREDICTION_PRODUCER = "timing_per_cell_cache_head_prediction_v2"
CACHE_PREDICTION_METHOD = "per_cell_l1_l2_monotonic_calibration_head_v5"
CACHE_HEAD_SCORER_NAME = "PerBucketCalibratorEnsemble.predict"
L2_CAPACITY_BYTES = 6 * 1024 * 1024
L2_SLICE_COUNT = 84
SECTOR_SIZE_BYTES = 32

_ACCESS_KIND_UNSPECIFIED = 0
_ACCESS_KIND_LOAD = 1
_ACCESS_KIND_STORE = 2
_ACCESS_KIND_ATOMIC = 3
_ACCESS_KIND_ASYNC_COPY = 4
_MEMORY_OP_CLASS_GLOBAL = 1


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _binary_path_for(slug_or_cell: str | dict[str, Any]) -> Path:
    if isinstance(slug_or_cell, dict):
        explicit = slug_or_cell.get("binary_path")
        if explicit:
            path = REPO_ROOT / str(explicit)
            if path.is_file():
                return path
        slug = str(slug_or_cell["slug"])
    else:
        slug = slug_or_cell
    candidates = (
        REPO_ROOT / "workloads" / "portable_lcd" / slug / f"{slug}_run",
        REPO_ROOT / "workloads" / "simbench_bw" / "l1" / slug / f"{slug}_run",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"binary for slug {slug!r} not found")


def _trace_window_for(cell: dict[str, Any]) -> dict[str, int]:
    window = cell.get("trace_window") or {}
    return {
        "instr_begin": int(window.get("instr_begin", TRACE_INSTR_BEGIN)),
        "instr_end": int(window.get("instr_end", TRACE_INSTR_END)),
    }


def _ceil_div(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor))


def _occupancy_from_launch(cell: dict[str, Any], regs_per_thread: int) -> dict[str, Any]:
    geom = cell["launch_geometry"]
    block_threads = int(geom["block_x"]) * int(geom["block_y"]) * int(geom["block_z"])
    warps_per_block = max(1, _ceil_div(block_threads, 32))
    regs_per_block = max(1, regs_per_thread * block_threads)
    shared_per_block = max(
        1,
        int(geom["shared_mem_static_bytes"]) + int(geom["shared_mem_dynamic_bytes"]),
    )
    max_blocks = min(
        32,
        65536 // regs_per_block,
        102400 // shared_per_block,
        64 // warps_per_block,
    )
    active_blocks = max(1, int(max_blocks))
    active_warps = active_blocks * warps_per_block
    return {
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "warps_per_block": warps_per_block,
        "source": "launch_geometry_registers_shared_memory_limits",
    }


def _trace_dir_for(trace_root: Path, cell: dict[str, Any]) -> Path:
    return trace_root / cell["shape_point"]


def _capture_nvbit_observed_sector_trace(
    *,
    cell: dict[str, Any],
    trace_root: Path,
    tracer_so_path: Path,
    timeout_s: int = 900,
) -> None:
    trace_dir = _trace_dir_for(trace_root, cell)
    if trace_dir.exists():
        import shutil
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    binary = _binary_path_for(cell)
    window = _trace_window_for(cell)
    env = {
        **tracer_runtime_env(tracer_so_path),
        "LD_PRELOAD": str(tracer_so_path),
        "USER_DEFINED_FOLDERS": "1",
        "TRACES_FOLDER": str(trace_dir),
        "SKIP_ENHANCED_TRACER": "1",
        "INSTR_BEGIN": str(window["instr_begin"]),
        "INSTR_END": str(window["instr_end"]),
        "DYNAMIC_KERNEL_LIMIT_END": "1",
        "TERMINATE_UPON_LIMIT": "0",
        "NOBANNER": "1",
    }
    proc = subprocess.run(
        [str(binary), *cell["run_args"]],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"NVBit observed sector capture failed for {cell['shape_point']} "
            f"rc={proc.returncode}: {proc.stderr[-500:]}"
        )
    trace_pb = trace_dir / "dynamic_trace.pb"
    if not trace_pb.is_file():
        raise RuntimeError(f"NVBit capture did not emit {trace_pb}")
    threadblocks = trace_dir / "threadblocks"
    if not threadblocks.is_dir() or not any(threadblocks.rglob("*.pb")):
        raise RuntimeError(f"NVBit capture emitted no threadblock protos at {trace_dir}")


def _emit_nvbit_launch_summary(
    *,
    cell: dict[str, Any],
    trace_root: Path,
    tracer_so_path: Path,
) -> dict[str, Any]:
    from workloads.simbench_bw.nvbit_trace_summary_adapter import (
        emit_nvbit_trace_summary,
    )

    trace_dir = _trace_dir_for(trace_root, cell)
    summary_path = trace_dir / "nvbit_launch_summary.json"
    raw = emit_nvbit_trace_summary(
        traces_dir=trace_dir,
        slug=cell["slug"],
        shape_point=cell["shape_point"],
        repeat_index=0,
        launch_index=0,
        executable_path=_binary_path_for(cell),
        tracer_so_path=tracer_so_path,
        output_path=summary_path,
    )
    stats_path = trace_dir / "stats.csv"
    stats_row = _read_stats_row(stats_path)
    geom = cell["launch_geometry"]
    launch_geometry = {
        "kernel_name": geom["kernel_name"],
        "grid_x": int(stats_row.get("grid_dimX", geom["grid_x"])),
        "grid_y": int(stats_row.get("grid_dimY", geom["grid_y"])),
        "grid_z": int(stats_row.get("grid_dimZ", geom["grid_z"])),
        "block_x": int(stats_row.get("block_dimX", geom["block_x"])),
        "block_y": int(stats_row.get("block_dimY", geom["block_y"])),
        "block_z": int(stats_row.get("block_dimZ", geom["block_z"])),
        "shared_mem_static_bytes": int(geom["shared_mem_static_bytes"]),
        "shared_mem_dynamic_bytes": int(geom["shared_mem_dynamic_bytes"]),
    }
    return {
        "path": _repo_rel(summary_path),
        "sha256": sha256_file(summary_path),
        "raw_kernel_name": raw["kernel_name"],
        "raw_kernel_signature_sha256": raw["kernel_signature_sha256"],
        "harmonized_kernel_name": geom["kernel_name"],
        "harmonized_kernel_signature_sha256": cell["kernel_signature_sha256"],
        "inst_count": int(raw["inst_count"]),
        "launch_index": 0,
        "repeat_index": 0,
        "stats_csv_path": _repo_rel(stats_path),
        "stats_csv_sha256": sha256_file(stats_path),
        "launch_geometry": launch_geometry,
        "instruction_trace_mode": OBSERVED_TRACE_MODE,
    }


def _read_stats_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {str(k).strip(): str(v).strip() for k, v in rows[0].items()}


def _nearest_rank_percentile(sorted_values: list[int], percentile: float) -> int:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile requires non-empty input")
    rank = max(1, math.ceil(percentile / 100.0 * n))
    return sorted_values[rank - 1]


def _threadblock_proto_paths(trace_dir: Path) -> list[Path]:
    return sorted((trace_dir / "threadblocks").rglob("*.pb"))


def _sha256_of_sorted_trace_files(trace_dir: Path, paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        rel = str(path.relative_to(trace_dir))
        h.update(rel.encode("utf-8"))
        h.update(b"\t")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _observed_summary_path(trace_dir: Path) -> Path:
    return trace_dir / "observed_sector_summary.json"


def _summarize_observed_sector_trace(
    *,
    cell: dict[str, Any],
    trace_root: Path,
) -> dict[str, Any]:
    from tools.nvbit_tracer import import_trace_contract
    from tools.hrd_profiler.reuse_distance_tracker import ReuseDistanceTracker

    trace_dir = _trace_dir_for(trace_root, cell)
    contract = import_trace_contract.validate_trace_dir(trace_dir)
    if not contract.ok:
        raise RuntimeError(
            f"NVBit trace contract failed for {cell['shape_point']}: "
            f"{contract.typed_reason}: {contract.detail}"
        )
    threadblock_pb2_mod = import_trace_contract._import_pb_modules()[0]
    proto_paths = _threadblock_proto_paths(trace_dir)
    if not proto_paths:
        raise RuntimeError(f"no threadblock protos under {trace_dir}")

    sectors: list[int] = []
    per_op_class_counts: dict[int, int] = {}
    per_access_kind_counts: dict[int, int] = {}
    memory_instruction_records = 0
    for tb_path in proto_paths:
        tb = threadblock_pb2_mod.threadblock()
        tb.ParseFromString(tb_path.read_bytes())
        for warp_index in sorted(tb.warps.keys()):
            for inst in tb.warps[warp_index].instructions:
                if not inst.HasField("memory_op_class"):
                    continue
                if not inst.addresses:
                    continue
                and_mask = inst.active_mask & inst.predicate_mask
                if and_mask == 0:
                    continue
                op_class = int(inst.memory_op_class)
                access_kind = (
                    int(inst.memory_op_access_kind)
                    if inst.HasField("memory_op_access_kind")
                    else _ACCESS_KIND_UNSPECIFIED
                )
                before = len(sectors)
                for addr_record in inst.addresses:
                    lane_vector, _reason = import_trace_contract._reconstruct_lane_vector(
                        addr_record,
                        and_mask,
                    )
                    if lane_vector is None:
                        continue
                    data_width = (
                        addr_record.data_width
                        if addr_record.data_width
                        else inst.data_width
                    )
                    if data_width <= 0:
                        continue
                    for lane_index, lane_addr in enumerate(lane_vector):
                        if not ((and_mask >> lane_index) & 1):
                            continue
                        if lane_addr == import_trace_contract.LANE_INACTIVE_SENTINEL:
                            continue
                        access_start = int(lane_addr)
                        access_end = access_start + int(data_width)
                        sectors.extend(
                            range(access_start // 32, (access_end - 1) // 32 + 1)
                        )
                added = len(sectors) - before
                if added:
                    memory_instruction_records += 1
                    per_op_class_counts[op_class] = (
                        per_op_class_counts.get(op_class, 0) + added
                    )
                    per_access_kind_counts[access_kind] = (
                        per_access_kind_counts.get(access_kind, 0) + added
                    )

    if not sectors:
        raise RuntimeError(f"observed sector trace has zero accesses for {cell['shape_point']}")

    tracker = ReuseDistanceTracker(max_accesses=len(sectors))
    sectors_per_l2_slice = max(1, L2_CAPACITY_BYTES // L2_SLICE_COUNT // SECTOR_SIZE_BYTES)
    cold_count = 0
    finite_miss_count = 0
    finite_rds: list[int] = []
    for timestamp, sector_id in enumerate(sectors, start=1):
        rd, state = tracker.access(sector_id, timestamp)
        if state == "cold":
            cold_count += 1
            continue
        if state == "finite":
            if rd is not None:
                finite_rds.append(int(rd))
            if rd is None or rd >= sectors_per_l2_slice:
                finite_miss_count += 1

    unique_sector_count = len(set(sectors))
    sector_frequency: dict[int, int] = {}
    for sector_id in sectors:
        sector_frequency[sector_id] = sector_frequency.get(sector_id, 0) + 1
    working_set_sector_count = sum(1 for count in sector_frequency.values() if count >= 2)
    finite_rds.sort()
    observed_access_count = len(sectors)
    observed_hit_count = observed_access_count - cold_count - finite_miss_count
    stats_row = _read_stats_row(trace_dir / "stats.csv")

    summary = {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_observed_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": OBSERVED_SECTOR_PRODUCER,
        "trace_window": {
            "instruction_trace_mode": OBSERVED_TRACE_MODE,
            **_trace_window_for(cell),
            "window_role": "per_cell_observed_global_memory_window",
        },
        "observation_basis": {
            "trace_contract_ok": True,
            "threadblock_proto_file_count": len(proto_paths),
            "threadblock_proto_set_sha256": _sha256_of_sorted_trace_files(
                trace_dir, proto_paths
            ),
            "memory_instruction_record_count": memory_instruction_records,
            "observed_access_count": observed_access_count,
            "observed_hit_count": observed_hit_count,
            "observed_miss_count": cold_count + finite_miss_count,
            "stats_total_insts": int(stats_row.get("total_insts", "0") or 0),
            "stats_total_reported_insts": int(
                stats_row.get("total_reported_insts", "0") or 0
            ),
        },
        "summarize_launch": {
            "cold_count": cold_count,
            "finite_miss_count": finite_miss_count,
            "unique_sector_count": unique_sector_count,
            "working_set_sector_count": working_set_sector_count,
            "reuse_distance_finite_p50": (
                _nearest_rank_percentile(finite_rds, 50.0) if finite_rds else None
            ),
            "reuse_distance_finite_p90": (
                _nearest_rank_percentile(finite_rds, 90.0) if finite_rds else None
            ),
            "reuse_distance_finite_p99": (
                _nearest_rank_percentile(finite_rds, 99.0) if finite_rds else None
            ),
            "per_op_class_counts": {str(k): v for k, v in per_op_class_counts.items()},
            "per_access_kind_counts": {
                str(k): v for k, v in per_access_kind_counts.items()
            },
            "load_count": per_access_kind_counts.get(_ACCESS_KIND_LOAD, 0),
            "store_count": per_access_kind_counts.get(_ACCESS_KIND_STORE, 0),
            "atomic_count": per_access_kind_counts.get(_ACCESS_KIND_ATOMIC, 0),
            "async_copy_count": per_access_kind_counts.get(_ACCESS_KIND_ASYNC_COPY, 0),
            "unspecified_kind_count": per_access_kind_counts.get(
                _ACCESS_KIND_UNSPECIFIED, 0
            ),
        },
    }
    path = _observed_summary_path(trace_dir)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _prune_threadblock_protos(trace_dir: Path) -> None:
    threadblocks = trace_dir / "threadblocks"
    if threadblocks.exists():
        shutil.rmtree(threadblocks)


def _derive_sector_counts_from_observation(observed_summary: dict[str, Any]) -> dict[str, int]:
    summary = observed_summary["summarize_launch"]
    basis = observed_summary["observation_basis"]
    per_op = {int(k): int(v) for k, v in summary["per_op_class_counts"].items()}
    per_kind = {int(k): int(v) for k, v in summary["per_access_kind_counts"].items()}
    global_accesses = per_op.get(_MEMORY_OP_CLASS_GLOBAL, int(basis["observed_access_count"]))
    global_writes = min(global_accesses, per_kind.get(_ACCESS_KIND_STORE, 0))
    global_reads = min(global_accesses - global_writes, per_kind.get(_ACCESS_KIND_LOAD, 0))
    total_sectors = int(basis["observed_access_count"])
    total_bytes = total_sectors * SECTOR_SIZE_BYTES
    return {
        "global_read_32b_sectors": int(global_reads),
        "global_write_32b_sectors": int(global_writes),
        "total_32b_sectors": total_sectors,
        "total_128b_lines": _ceil_div(total_bytes, 128),
        "estimated_bytes": total_bytes,
    }


def build_sector_summary(
    cell: dict[str, Any],
    *,
    regs_per_thread: int,
    static_work_path: Path,
    static_features_path: Path,
    trace_root: Path,
    tracer_so_path: Path,
) -> dict[str, Any]:
    trace_dir = _trace_dir_for(trace_root, cell)
    window = _trace_window_for(cell)
    trace_pb = trace_dir / "dynamic_trace.pb"
    launch_summary = _emit_nvbit_launch_summary(
        cell=cell,
        trace_root=trace_root,
        tracer_so_path=tracer_so_path,
    )
    observed_summary = _summarize_observed_sector_trace(
        cell=cell,
        trace_root=trace_root,
    )
    observed_path = _observed_summary_path(trace_dir)
    _prune_threadblock_protos(trace_dir)
    return {
        "schema_version": 1,
        "artifact_kind": "timing_nvbit_sector_summary",
        "arch": cell["arch"],
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": cell["kernel_signature_sha256"],
        "source_commit": cell["source_commit"],
        "producer": "nvbit_observed_summarize_launch_sector_summary_v1",
        "source_trace": {
            "nvbit_trace_pb_path": _repo_rel(trace_pb),
            "nvbit_trace_pb_sha256": sha256_file(trace_pb),
            "trace_dir": _repo_rel(trace_dir),
            "capture_mode": "per_cell_same_binary_same_shape_nvbit_bounded_observed_threadblock_trace",
            "instruction_trace_mode": OBSERVED_TRACE_MODE,
            "instr_begin": window["instr_begin"],
            "instr_end": window["instr_end"],
            "threadblock_proto_file_count": observed_summary["observation_basis"][
                "threadblock_proto_file_count"
            ],
            "threadblock_proto_set_sha256": observed_summary["observation_basis"][
                "threadblock_proto_set_sha256"
            ],
            "threadblock_proto_retention": "summarized_then_pruned_after_digest",
        },
        "nvbit_launch_summary": launch_summary,
        "observed_sector_summary": {
            "path": _repo_rel(observed_path),
            "sha256": sha256_file(observed_path),
            "artifact_kind": "timing_nvbit_observed_sector_summary",
        },
        "summarize_launch": observed_summary["summarize_launch"],
        "observation_basis": observed_summary["observation_basis"],
        "static_inputs": {
            "static_sass_features_path": _repo_rel(static_features_path),
            "static_sass_features_sha256": sha256_file(static_features_path),
            "static_work_path": _repo_rel(static_work_path),
            "static_work_sha256": sha256_file(static_work_path),
        },
        "sector_counts": _derive_sector_counts_from_observation(observed_summary),
        "sector_count_derivation": {
            "source": "nvbit_observed_threadblock_proto_sector_stream",
            "observed_access_count": observed_summary["observation_basis"][
                "observed_access_count"
            ],
            "observed_sector_summary_path": _repo_rel(observed_path),
            "instruction_trace_mode": OBSERVED_TRACE_MODE,
        },
        "theoretical_occupancy": _occupancy_from_launch(cell, regs_per_thread),
        "deploy_legal": True,
    }


def _feature_ref(path: Path, artifact_kind: str) -> dict[str, str]:
    return {
        "path": _repo_rel(path),
        "sha256": sha256_file(path),
        "artifact_kind": artifact_kind,
    }


def _launch_config_for_cell(cell: dict[str, Any]) -> dict[str, Any]:
    geom = cell["launch_geometry"]
    return {
        "kernel_name": geom["kernel_name"],
        "grid_x": int(geom["grid_x"]),
        "grid_y": int(geom["grid_y"]),
        "grid_z": int(geom["grid_z"]),
        "block_x": int(geom["block_x"]),
        "block_y": int(geom["block_y"]),
        "block_z": int(geom["block_z"]),
        "shared_mem_static_bytes": int(geom["shared_mem_static_bytes"]),
        "shared_mem_dynamic_bytes": int(geom["shared_mem_dynamic_bytes"]),
        "regs_per_thread": int(cell["regs_per_thread"]),
    }


def _static_paths_for_cell(feature_dir: Path, cell: dict[str, Any]) -> tuple[Path, Path]:
    stem = cell["shape_point"]
    return (
        feature_dir / f"{stem}_static_sass_features.json",
        feature_dir / f"{stem}_static_work.json",
    )


def _emit_static_artifacts_for_cell(
    *,
    feature_dir: Path,
    cell: dict[str, Any],
    source_commit: str,
) -> tuple[Path, Path]:
    from workloads.simbench_bw.static_sass_features import extract_records

    features_path, work_path = _static_paths_for_cell(feature_dir, cell)
    feature_record, work_record = extract_records(
        binary=_binary_path_for(cell),
        slug=cell["slug"],
        launch_config=_launch_config_for_cell(cell),
        source_commit=source_commit,
        kernel=cell["launch_geometry"]["kernel_name"],
        arch=cell["arch"],
    )
    features_path.write_text(
        json.dumps(feature_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    work_path.write_text(
        json.dumps(work_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return features_path, work_path


def _artifact_ref(path: Path) -> dict[str, str]:
    return {"path": _repo_rel(path), "sha256": sha256_file(path)}


def _repo_rel_or_path(path: Path) -> str:
    try:
        return _repo_rel(path)
    except ValueError:
        return str(path)


def _head_artifact_ref(
    path: Path,
    *,
    typed_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    try:
        return {
            "path": _repo_rel_or_path(path),
            "sha256": sha256_file(path),
            "artifact_status": "available",
            "typed_unavailable_reason": None,
        }
    except Exception as exc:
        return {
            "path": _repo_rel_or_path(path),
            "sha256": None,
            "artifact_status": "unavailable",
            "typed_unavailable_reason": (
                typed_unavailable_reason or f"head_artifact_unreadable:{exc}"
            ),
        }


def _available_head_artifact_ref(path: Path) -> dict[str, Any]:
    ref = _head_artifact_ref(path)
    if ref["artifact_status"] != "available":
        raise RuntimeError(ref["typed_unavailable_reason"])
    return ref


def _head_training_cells(head: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in head.get("cohort_scope", {}).get("admitted", []):
        out.append(
            {
                "identity": (
                    cell["arch"],
                    cell["slug"],
                    cell["shape_point"],
                ),
                "arch": cell["arch"],
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "family_bucket": cell["family_bucket"],
                "features": dict(cell["features"]),
                "label": float(cell["label"]),
            }
        )
    return out


def _build_head_ensemble(head: dict[str, Any]) -> Any:
    from tools.local_labels.fit_monotonic_calibration_head_v5 import (
        _model_class_factory_for,
    )
    from workloads.simbench_bw import monotonic_calibration_head as mch

    if head.get("fit_status") != "ok":
        raise RuntimeError(f"head fit_status={head.get('fit_status')!r}")
    selected_name = head.get("selected_class", {}).get("name")
    if not selected_name:
        raise RuntimeError("head selected_class.name is missing")
    cells = _head_training_cells(head)
    if not cells:
        raise RuntimeError("head cohort_scope.admitted is empty")
    ensemble = mch.PerBucketCalibratorEnsemble(
        model_class_factory=_model_class_factory_for(selected_name),
    )
    ensemble.fit(cells)
    return ensemble


def _head_route(
    head: dict[str, Any],
    *,
    arch: str,
    family_bucket: str,
) -> dict[str, Any]:
    provenance = head.get("per_bucket_provenance", {})
    by_bucket = provenance.get("by_bucket", {})
    bucket_key = f"{arch}/{family_bucket}"
    pooled_present = bool(provenance.get("pooled_fallback_present"))
    if bucket_key in by_bucket:
        entry = by_bucket[bucket_key]
        return {
            "bucket_key": bucket_key,
            "bucket_in_head_scope": True,
            "kind": entry["kind"],
            "n_train": int(entry["n_train"]),
            "pooled_fallback_present": pooled_present,
        }
    if pooled_present:
        return {
            "bucket_key": bucket_key,
            "bucket_in_head_scope": False,
            "kind": "pooled_fallback_unseen_bucket_zero_offset",
            "n_train": None,
            "pooled_fallback_present": True,
        }
    raise RuntimeError(f"no head route for {bucket_key!r}")


def _ordered_canonical_features(features: dict[str, float]) -> dict[str, float]:
    from workloads.simbench_bw import canonical_features as cf

    missing = [name for name in cf.CANONICAL_FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"canonical feature vector missing {missing!r}")
    return {
        name: float(cf._validate_feature_value(name, features[name]))
        for name in cf.CANONICAL_FEATURE_NAMES
    }


def _raw_observed_hit_fraction(sector: dict[str, Any]) -> float:
    basis = sector["observation_basis"]
    access_count = int(basis["observed_access_count"])
    hit_count = int(basis["observed_hit_count"])
    if access_count <= 0:
        raise RuntimeError("observed sector summary has no accesses")
    if hit_count < 0 or hit_count > access_count:
        raise RuntimeError(
            "observed sector summary has invalid hit/access counts: "
            f"{hit_count}/{access_count}"
        )
    return hit_count / access_count


def _l1_head_features(raw_prediction: float) -> dict[str, float]:
    from workloads.simbench_bw import canonical_features as cf

    features = {name: 0.0 for name in cf.CANONICAL_FEATURE_NAMES}
    features["raw_prediction"] = raw_prediction
    return _ordered_canonical_features(features)


def _l2_head_features(
    sector: dict[str, Any],
    *,
    raw_prediction: float,
) -> dict[str, float]:
    from workloads.simbench_bw import canonical_features as cf

    summary = sector["summarize_launch"]
    counts = sector["sector_counts"]
    launch_geom = sector["nvbit_launch_summary"]["launch_geometry"]
    finite_l2_slice_misses = int(summary["finite_miss_count"])
    if finite_l2_slice_misses != 0:
        raise RuntimeError(
            "observed sector summary lacks a long-RD histogram for nonzero "
            "finite L2-slice misses"
        )
    global_count = (
        int(counts["global_read_32b_sectors"])
        + int(counts["global_write_32b_sectors"])
    )
    grid_volume = (
        int(launch_geom["grid_x"])
        * int(launch_geom["grid_y"])
        * int(launch_geom["grid_z"])
    )
    features = {
        "raw_prediction": raw_prediction,
        "footprint_ratio": cf.compute_footprint_ratio(
            unique_sector_count=int(summary["unique_sector_count"]),
            cache_model_capacity_bytes=L2_CAPACITY_BYTES,
            sector_size_bytes=SECTOR_SIZE_BYTES,
        ),
        "long_rd_tail_mass": 0.0,
        "intra_lane_reuse": 0.0,
        "global_share": cf.compute_global_share(
            global_count=global_count,
            total_count=int(counts["total_32b_sectors"]),
        ),
        "gridDim_density": cf.compute_gridDim_density(
            grid_volume=grid_volume,
            device=sector["arch"],
        ),
    }
    return _ordered_canonical_features(features)


def _typed_unavailable_layer_prediction(
    *,
    layer: str,
    reason: str,
    head_path: Path,
    head: dict[str, Any] | None,
    family_bucket: str | None = None,
    features: dict[str, float] | None = None,
    route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cache_layer": layer,
        "result_kind": "typed_unavailable",
        "predicted_sector_hit_rate": None,
        "typed_unavailable_reason": reason,
        "head_artifact": _head_artifact_ref(
            head_path,
            typed_unavailable_reason=reason,
        ),
        "head_kind": (head or {}).get("head_kind"),
        "head_scope": (head or {}).get("head_scope"),
        "head_source_commit": (head or {}).get("source_commit"),
        "fit_status": (head or {}).get("fit_status"),
        "selected_class": (head or {}).get("selected_class"),
        "family_bucket": family_bucket,
        "route": route,
        "input_features": features,
        "input_feature_sha256": (
            _canonical_json_sha256(features) if features is not None else None
        ),
        "feature_schema_sha256": (
            _canonical_json_sha256((head or {}).get("feature_schema"))
            if (head or {}).get("feature_schema") is not None
            else None
        ),
        "scorer_name": CACHE_HEAD_SCORER_NAME,
    }


def _score_cache_head(
    *,
    layer: str,
    head_path: Path,
    head: dict[str, Any] | None,
    ensemble: Any | None,
    head_error: str | None,
    features: dict[str, float] | None,
    arch: str,
    family_bucket: str,
) -> dict[str, Any]:
    if head is None:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=head_error or "head_artifact_load_failed",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
        )
    if features is None:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=head_error or "canonical_feature_assembly_failed",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
        )
    route: dict[str, Any] | None
    try:
        route = _head_route(head, arch=arch, family_bucket=family_bucket)
    except Exception as exc:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=f"head_route_unavailable:{exc}",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
        )
    if ensemble is None:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=head_error or "head_ensemble_unavailable",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
            route=route,
        )
    try:
        predicted = float(ensemble.predict(arch, family_bucket, features))
    except Exception as exc:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=f"head_scoring_failed:{exc}",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
            route=route,
        )
    try:
        head_artifact = _available_head_artifact_ref(head_path)
    except Exception as exc:
        return _typed_unavailable_layer_prediction(
            layer=layer,
            reason=f"head_artifact_digest_unavailable:{exc}",
            head_path=head_path,
            head=head,
            family_bucket=family_bucket,
            features=features,
            route=route,
        )
    return {
        "cache_layer": layer,
        "result_kind": "scalar",
        "predicted_sector_hit_rate": max(0.0, min(1.0, predicted)),
        "typed_unavailable_reason": None,
        "head_artifact": head_artifact,
        "head_kind": head.get("head_kind"),
        "head_scope": head.get("head_scope"),
        "head_source_commit": head.get("source_commit"),
        "fit_status": head.get("fit_status"),
        "selected_class": head.get("selected_class"),
        "family_bucket": family_bucket,
        "route": route,
        "input_features": features,
        "input_feature_sha256": _canonical_json_sha256(features),
        "feature_schema_sha256": _canonical_json_sha256(head.get("feature_schema")),
        "scorer_name": CACHE_HEAD_SCORER_NAME,
    }


def _build_cache_hit_predictions(
    *,
    cohort: dict[str, Any],
    output_path: Path,
    l1_head_path: Path,
    l2_head_path: Path,
    family_buckets_path: Path,
    source_commit: str | None,
) -> dict[str, Any]:
    from tools.local_labels.fit_monotonic_calibration_head_v5 import (
        _load_family_buckets,
        _resolve_family_bucket,
    )

    family_to_bucket = _load_family_buckets(family_buckets_path)
    l1_head: dict[str, Any] | None = None
    l2_head: dict[str, Any] | None = None
    l1_ensemble: Any | None = None
    l2_ensemble: Any | None = None
    l1_error: str | None = None
    l2_error: str | None = None
    try:
        l1_head = _load(l1_head_path)
        l1_ensemble = _build_head_ensemble(l1_head)
    except Exception as exc:
        l1_error = str(exc)
    try:
        l2_head = _load(l2_head_path)
        l2_ensemble = _build_head_ensemble(l2_head)
    except Exception as exc:
        l2_error = str(exc)

    cells = []
    for cell in cohort.get("cells", []):
        refs = cell["feature_artifacts"]
        sector_path = REPO_ROOT / refs["nvbit_sector_summary"]["path"]
        sector = _load(sector_path)
        family_bucket = _resolve_family_bucket(cell["slug"], family_to_bucket)
        try:
            raw_prediction = _raw_observed_hit_fraction(sector)
            l1_features: dict[str, float] | None = _l1_head_features(raw_prediction)
            l2_features: dict[str, float] | None = _l2_head_features(
                sector,
                raw_prediction=raw_prediction,
            )
            feature_error: str | None = None
        except Exception as exc:
            l1_features = None
            l2_features = None
            feature_error = str(exc)
        l1_prediction = _score_cache_head(
            layer="L1TEX",
            head_path=l1_head_path,
            head=l1_head,
            ensemble=l1_ensemble,
            head_error=l1_error or feature_error,
            features=l1_features,
            arch=cell["arch"],
            family_bucket=family_bucket,
        )
        l2_prediction = _score_cache_head(
            layer="L2",
            head_path=l2_head_path,
            head=l2_head,
            ensemble=l2_ensemble,
            head_error=l2_error or feature_error,
            features=l2_features,
            arch=cell["arch"],
            family_bucket=family_bucket,
        )
        if (
            l1_prediction["result_kind"] == "scalar"
            and l2_prediction["result_kind"] == "scalar"
        ):
            result_kind = "scalar"
            l1_value = l1_prediction["predicted_sector_hit_rate"]
            l2_value = l2_prediction["predicted_sector_hit_rate"]
            typed_reason = None
        else:
            result_kind = "typed_unavailable"
            l1_value = None
            l2_value = None
            reasons = []
            for name, prediction in (
                ("l1", l1_prediction),
                ("l2", l2_prediction),
            ):
                if prediction["result_kind"] != "scalar":
                    reasons.append(
                        f"{name}_head_unscorable:"
                        f"{prediction['typed_unavailable_reason']}"
                    )
            typed_reason = ";".join(reasons)
        cells.append(
            {
                "slug": cell["slug"],
                "shape_point": cell["shape_point"],
                "arch": cell["arch"],
                "binary_sha256": cell["binary_sha256"],
                "kernel_signature_sha256": cell["kernel_signature_sha256"],
                "source_commit": cell["source_commit"],
                "result_kind": result_kind,
                "l1_sector_hit_rate": l1_value,
                "l2_sector_hit_rate": l2_value,
                "typed_unavailable_reason": typed_reason,
                "input_feature_refs": {
                    "nvbit_sector_summary": refs["nvbit_sector_summary"],
                    "observed_sector_summary": sector["observed_sector_summary"],
                    "static_sass_features": refs["static_sass_features"],
                    "static_work": refs["static_work"],
                },
                "head_provenance": {
                    "observed_sector_summary": sector["observed_sector_summary"],
                    "l1_prediction": l1_prediction,
                    "l2_prediction": l2_prediction,
                    "prediction_method": CACHE_PREDICTION_METHOD,
                    "producer": CACHE_PREDICTION_PRODUCER,
                },
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": "timing_cache_hit_prediction_bundle",
        "arch": cohort["arch"],
        "source_commit": source_commit or _source_commit(),
        "producer": CACHE_PREDICTION_PRODUCER,
        "prediction_scope": "long_duration_timing_cohort",
        "cohort_artifact": _repo_rel(DEFAULT_COHORT),
        "head_artifacts": {
            "l1_head_artifact": _head_artifact_ref(
                l1_head_path,
                typed_unavailable_reason=l1_error,
            ),
            "l2_head_artifact": _head_artifact_ref(
                l2_head_path,
                typed_unavailable_reason=l2_error,
            ),
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def attach_feature_artifacts_to_cohort(
    cohort: dict[str, Any],
    *,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    cache_hit_predictions_path: Path = DEFAULT_CACHE_HIT_PREDICTIONS,
    static_work_path: Path = DEFAULT_STATIC_WORK,
    static_features_path: Path = DEFAULT_STATIC_FEATURES,
    tracer_so_path: Path = DEFAULT_TRACER_SO,
    l1_head_path: Path = DEFAULT_L1_HEAD,
    l2_head_path: Path = DEFAULT_L2_HEAD,
    family_buckets_path: Path = DEFAULT_FAMILY_BUCKETS,
    source_commit: str | None = None,
    capture_nvbit: bool = False,
    capture_missing_nvbit: bool = False,
) -> dict[str, Any]:
    feature_dir.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    out = json.loads(json.dumps(cohort))
    commit = source_commit or _source_commit()
    for cell in out.get("cells", []):
        cell_static_features_path, cell_static_work_path = _static_paths_for_cell(
            feature_dir, cell
        )
        if (
            capture_nvbit
            or not cell_static_features_path.is_file()
            or not cell_static_work_path.is_file()
        ):
            cell_static_features_path, cell_static_work_path = (
                _emit_static_artifacts_for_cell(
                    feature_dir=feature_dir,
                    cell=cell,
                    source_commit=commit,
                )
            )
        sector_path = feature_dir / f"{cell['shape_point']}_nvbit_sector_summary.json"
        if capture_nvbit or (capture_missing_nvbit and not sector_path.is_file()):
            _capture_nvbit_observed_sector_trace(
                cell=cell,
                trace_root=trace_root,
                tracer_so_path=tracer_so_path,
            )
        if capture_nvbit or not sector_path.is_file():
            sector = build_sector_summary(
                cell,
                regs_per_thread=int(cell["regs_per_thread"]),
                static_work_path=cell_static_work_path,
                static_features_path=cell_static_features_path,
                trace_root=trace_root,
                tracer_so_path=tracer_so_path,
            )
            sector_path.write_text(
                json.dumps(sector, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        cell["feature_artifacts"] = {
            "nvbit_sector_summary": _feature_ref(
                sector_path, "timing_nvbit_sector_summary"
            ),
            "static_sass_features": _feature_ref(
                cell_static_features_path, "static_sass_features"
            ),
            "static_work": _feature_ref(cell_static_work_path, "static_work"),
        }
    cache_predictions = _build_cache_hit_predictions(
        cohort=out,
        output_path=cache_hit_predictions_path,
        l1_head_path=l1_head_path,
        l2_head_path=l2_head_path,
        family_buckets_path=family_buckets_path,
        source_commit=commit,
    )
    cache_ref = _feature_ref(
        cache_hit_predictions_path,
        "timing_cache_hit_prediction_bundle",
    )
    for cell in out.get("cells", []):
        cell["feature_artifacts"]["cache_hit_predictions"] = cache_ref
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument(
        "--cache-hit-predictions",
        type=Path,
        default=DEFAULT_CACHE_HIT_PREDICTIONS,
    )
    parser.add_argument("--static-work", type=Path, default=DEFAULT_STATIC_WORK)
    parser.add_argument("--static-features", type=Path, default=DEFAULT_STATIC_FEATURES)
    parser.add_argument("--tracer-so", type=Path, default=DEFAULT_TRACER_SO)
    parser.add_argument("--l1-head", type=Path, default=DEFAULT_L1_HEAD)
    parser.add_argument("--l2-head", type=Path, default=DEFAULT_L2_HEAD)
    parser.add_argument("--family-buckets", type=Path, default=DEFAULT_FAMILY_BUCKETS)
    parser.add_argument("--capture-nvbit", action="store_true")
    parser.add_argument("--capture-missing-nvbit", action="store_true")
    args = parser.parse_args(argv)
    cohort = _load(args.cohort)
    out = attach_feature_artifacts_to_cohort(
        cohort,
        feature_dir=args.feature_dir,
        trace_root=args.trace_root,
        cache_hit_predictions_path=args.cache_hit_predictions,
        static_work_path=args.static_work,
        static_features_path=args.static_features,
        tracer_so_path=args.tracer_so,
        l1_head_path=args.l1_head,
        l2_head_path=args.l2_head,
        family_buckets_path=args.family_buckets,
        capture_nvbit=args.capture_nvbit,
        capture_missing_nvbit=args.capture_missing_nvbit,
    )
    args.output.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
