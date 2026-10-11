#!/usr/bin/env python3
"""Declare a prospective SM120 cohort without relabelling old captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATES = [
    {
        "family": "async_shared",
        "workload": "cp_async_shared_tile_copy",
        "binary": "workloads/simbench_bw/l1/cp_async_shared_tile_copy/cp_async_shared_tile_copy_run",
        "command": ["--rows", "64", "--cols", "64", "--seed", "7", "--shape-point", "prospective_64"],
        "mechanisms": ["LDGSTS", "shared_visibility", "wait_group"],
        "evidence": ["cuda_event", "NVBit", "NCU"],
    },
    {
        "family": "dram_stream",
        "workload": "dram_streaming_dot_product",
        "binary": "workloads/simbench_bw/l1/dram_streaming_dot_product/dram_streaming_dot_product_run",
        "command": ["--n", "65536", "--passes", "1", "--blocks", "16", "--threads", "128", "--warmup", "0", "--reps", "1", "--seed", "11", "--shape-point", "prospective_64k"],
        "mechanisms": ["global_load", "DRAM_service", "offered_ILP"],
        "evidence": ["cuda_event", "NVBit", "NCU"],
    },
    {
        "family": "shared_banks",
        "workload": "shared_bank_conflict_stride",
        "binary": "workloads/simbench_bw/l1/shared_bank_conflict_stride/shared_bank_conflict_stride_run",
        "command": ["--rows", "64", "--cols", "64", "--stride", "1", "--seed", "13", "--shape-point", "prospective_stride1"],
        "mechanisms": ["shared_wavefront", "bank_conflict", "warp_issue"],
        "evidence": ["cuda_event", "NVBit", "NCU"],
    },
    {
        "family": "control_flow",
        "workload": "warp_divergence_reconvergence",
        "binary": "workloads/simbench_bw/l1/warp_divergence_reconvergence/warp_divergence_reconvergence_run",
        "command": ["--warps", "4", "--branches", "2", "--iterations", "4", "--seed", "17", "--shape-point", "prospective_divergence"],
        "mechanisms": ["branch", "reconvergence", "front_end"],
        "evidence": ["cuda_event", "NVBit", "NCU"],
    },
    {
        "family": "tma_2d",
        "workload": "tma_tensor_copy_2d",
        "binary": "workloads/simbench_bw/l1/tma_tensor_copy_2d/tma_tensor_copy_2d_run",
        "command": ["--rows", "128", "--cols", "128", "--seed", "19", "--shape-point", "prospective_tma2d"],
        "mechanisms": ["UTMA", "mbarrier", "TMA_visibility"],
        "evidence": ["cuda_event", "NVBit", "NCU"],
    },
    {
        "family": "global_width",
        "workload": "global_vectorized_128b_copy",
        "binary": "workloads/simbench_bw/l1/global_vectorized_128b_copy/global_vectorized_128b_copy_run",
        "command": ["--samples", "4096", "--bins", "4", "--seed", "23", "--shape-point", "prospective_vec128"],
        "mechanisms": ["global_route", "sector_demand", "vector_width"],
        "evidence": ["cuda_event", "NVBit", "NCU"],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/GPU_simulator_rebase"))
    parser.add_argument("--work-root", type=Path, default=Path("/dev/shm/sm120_rtx_5090_work"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for item in CANDIDATES:
        binary = args.repo / item["binary"]
        rows.append({
            **item,
            "binary_path": str(binary),
            "binary_present": binary.is_file(),
            "capture_destination": str(args.work_root / "prospective" / item["workload"]),
            "role": "prospective_evaluation_only",
            "old_capture_overlap": False,
        })
    result = {
        "schema_version": 1,
        "model": "sm120_prospective_cohort_inventory_v1",
        "status": "inventory_complete",
        "role": "future_evaluation_only_before_new_observations",
        "development_corpus_excluded": True,
        "candidate_count": len(rows),
        "candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(rows), "binaries_present": sum(row["binary_present"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
