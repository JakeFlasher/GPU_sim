#!/usr/bin/env python3
"""Build the deterministic sm_86 full-roster reference-capture matrix.

The roster authority decides *which* cells are runnable.  This producer binds
that decision to exact capture invocations, metric-registry identity, budgets,
and bounded resumable chunks.  It is intentionally fail-closed:

* the authority must be non-empty, account exactly for every cell, contain no
  pending disposition, and be stamped at the current source commit;
* every runnable reference/executable must exist and match its authority hash;
* per-cell trace limits are never inferred from the aggregate retention ceiling
  or from the TOML defaults template;
* SOL solution JSON is a reference input, not an executable or static control
  binary.  SOL rows carry the exact sol-execbench adapter contract and require a
  post-trace loaded-module control-bit outcome (or a typed unavailable result).

The output has no timestamp.  Given identical inputs and host adapter files it
is byte-stable, and ``validate_matrix`` re-derives it rather than trusting
self-reported counts or seals.

Usage:
  .venv/bin/python tools/local_labels/build_sm86_capture_matrix.py \
      --out data/local_labels/campaign/sm86_reference_capture_matrix.json
  .venv/bin/python tools/local_labels/build_sm86_capture_matrix.py \
      --check data/local_labels/campaign/sm86_reference_capture_matrix.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cloud.trace_budgets import load_trace_budgets  # noqa: E402
from tools.sol_baselines_harness.native_env import (  # noqa: E402
    NATIVE_ENV_COMPONENT_ORDER,
    compute_native_env_hash,
    missing_components,
)

ARCH = "sm_86"
SKU = "rtx_3090_ti"
MATRIX_ID = "sm86_reference_capture_matrix_v1"
EXPECTED_UNIVERSE = 344
SOURCE_ORDER = ("portable_lcd", "simbench_bw", "sol_execbench")
FINAL_DISPOSITIONS = frozenset({"runnable", "excluded"})
PENDING_DISPOSITIONS = frozenset({
    "pending_build", "pending_reconstitution", "pending_shrink",
})
MAX_CHUNK_CELLS = 8

AUTHORITY_PATH = Path("data/local_labels/campaign/capture_roster_authority_sm86.json")
PROTOCOL_PINS_PATH = Path("tools/local_labels/campaign_protocol_pins.json")
RETENTION_BUDGET_PATH = Path("data/local_labels/campaign/trace_retention_budget.json")
TRACE_BUDGETS_PATH = Path("workloads/simbench_bw/trace_budgets.toml")
METRIC_REGISTRY_PATH = Path("tools/local_labels/metric_registry.json")
FROZEN_V1_MATRIX_PATH = Path(
    "data/local_labels/campaign/sm86_reference_capture_matrix.json"
)
FROZEN_V1_MATRIX_SHA256 = (
    "af29d72e955cf42e281036190f2d55d72af0af92a632c78c8d159073a072fb0e"
)
FROZEN_V1_MATRIX_SEAL_SHA256 = (
    "6376da9721c75bb9299e84a9d11bf607bc21f4cbaa30da25b672c86f6757a5e1"
)
FROZEN_V1_SOURCE_COMMIT = "124ce358733f97ba6e0af808d10ab69ffe4d5a78"
FROZEN_V1_REGISTRY_SHA256 = (
    "be1f7893cf653800c69f6e39a4d22abfae4d67c10197e7d5f9bd46c6900c2423"
)
FROZEN_V1_METRIC_IDENTITY_SHA256 = (
    "13f1bdfd8155cb71db62cc30aea41bc7fc6af6447d1251d9ea4f0efd4a8396a6"
)
VALIDATION_MODES = frozenset({"live", "frozen_v1"})
SOL_EXECBENCH_PATH = Path("tools/sol_execbench/.venv/bin/sol-execbench")
SOL_BENCHMARK_ROOT = Path("tools/sol_execbench/data/benchmark")
SOL_BASELINES_ROOT = Path("workloads/sol_execbench_baselines")
SOL_CUTLASS_PATH = Path("third_party/cutlass")
SOL_SHRINK_PROVENANCE_PATH = Path(
    "data/local_labels/campaign/sol_shrink_provenance_sm86.json"
)
SOL_RUNNABILITY_PATH = Path(
    "data/local_labels/campaign/sol_runnability_sm86.json"
)
SOL_NATIVE_ENV_FINGERPRINT_PATH = Path(
    "data/local_labels/campaign/sol_native_env_fingerprint_sm86.json"
)
SOL_SUBSET_MAP = {"L1": "L1", "L2": "L2", "FIB": "FlashInfer-Bench", "Quant": "Quant"}
SOL_NATIVE_SHAPE = "workloads"
SOL_REDUCED_SHAPE = "reduced_representative"
SOL_BOOTSTRAP_SLUG = "problem_070_mamba2_fused_intra_chunk_diagonal_computation"
SOL_PINNED_NVCC_CCBIN = "g++-14"

# A stable runtime module whose enhanced-tracer row was populated through the
# tracer's no-binary parser path still has authoritative static SASS bytes.  In
# that case the capture consumer rebinds the runtime function/PC inventory to
# one exact text section from the preserved, SHA-bound module.  Keep the tools
# and supported code arches in the matrix so this is an explicit campaign
# contract rather than an ambient PATH dependency.
STATIC_CONTROL_TOOLCHAIN = {
    "toolkit_version": "V13.2",
    "nvcc_path": "/usr/local/cuda-13.2/bin/nvcc",
    "nvcc_sha256": (
        "1b32c567e18698a94d78f18e571d7a213cfb461a7422505120e88748d5150136"
    ),
    "cuobjdump_path": "/usr/local/cuda-13.2/bin/cuobjdump",
    "cuobjdump_sha256": (
        "903c5e1324669c1222b13ede0fc7226f48f7db355669f2912f0236031c56e1bf"
    ),
}
SOL_STATIC_CONTROL_REBIND = {
    "policy": "preserved_module_exact_function",
    "allowed_code_arches": ["sm_80", "sm_86"],
    "ncu_report_binding": "raw_text_section_exactly_once",
    **STATIC_CONTROL_TOOLCHAIN,
}
SOL_CONTROL_OUTCOMES = (
    "exact_accepted_launch_runtime_binary_controls_emitted",
    "exact_accepted_launch_static_rebind_controls_emitted",
    "typed_unavailable_no_stable_static_module",
)

# label_driver.sh always requests these four fallback counters in addition to
# the two selected flat percentage metrics.  They are supporting parser
# observations, not entries in the 61-row metric registry.
DRIVER_SUPPORTING_RAW = (
    "l1tex__t_sectors_lookup_hit.sum",
    "l1tex__t_sectors_lookup_miss.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
)

# The frozen roster remains the authority for runnable membership and binary
# identity.  These rows are execution-only corrections discovered by live
# sm_86 probes: two authority shapes exceed the device's dynamic shared-memory
# launch envelope, two binaries otherwise expose the wrong launch to NCU/NVBit,
# one cuBLAS shape selects a kernel that the frozen H3 tracer cannot parse, and
# one workload is intentionally a cold/evict/warm three-launch sequence.
# Every operand is pinned so an authority, shape, or argument change fails
# closed instead of silently inheriting a stale override.
SIMBENCH_CAPTURE_OVERRIDES = {
    "cp_async_shared_tile_copy": {
        "reason": "authority_shape_dynamic_shared_memory_unlaunchable",
        "authority_cell_sha256": "0f9853adcb12be5663e5ea0511103232e6a375b852d8a8263521750494082e91",
        "authority_shape_point": "trace_00",
        "authority_app_args_sha256": "18e5e874bd3a8a0387cc0d835f4c856bc95b294471a4446809f7a73c10e98a7a",
        "capture_shape_point": "trace_l2_accuracy_xl",
        "label_app_args_sha256": "3e05be5979af5769d551f1512a942635a9e8809566a27ba21919da032803653b",
        "trace_app_args_sha256": "3e05be5979af5769d551f1512a942635a9e8809566a27ba21919da032803653b",
        "label_app_args": [
            "--rows", "64", "--cols", "64", "--seed", "42",
            "--shape-point", "trace_l2_accuracy_xl",
        ],
        "trace_app_args": [
            "--rows", "64", "--cols", "64", "--seed", "42",
            "--shape-point", "trace_l2_accuracy_xl",
        ],
        "shape_file_sha256": "83075ad60f9e6303a0dac2af1cae38cbae43c53b3c985dc6d379c783c64c3193",
        "trace_cardinality_contract": "matrix_capture_args_override_v1",
    },
    "shared_bank_conflict_stride": {
        "reason": "authority_shape_dynamic_shared_memory_unlaunchable",
        "authority_cell_sha256": "855e4054fdddaed1c93db91e963bf46e65882aaf07139575f601b04bdf826957",
        "authority_shape_point": "trace_00",
        "authority_app_args_sha256": "be449b950e217ba40002990a4dda59ec4ce067a22e33fb7e27e91fd90143f767",
        "capture_shape_point": "trace_l2_accuracy_xl",
        "label_app_args_sha256": "9c83716a9fe43cf43e1dabcfce0f161f41f3d2f8c81c558dc5c3716b49f238b5",
        "trace_app_args_sha256": "9c83716a9fe43cf43e1dabcfce0f161f41f3d2f8c81c558dc5c3716b49f238b5",
        "label_app_args": [
            "--rows", "32", "--cols", "16", "--stride", "4",
            "--seed", "42", "--shape-point", "trace_l2_accuracy_xl",
        ],
        "trace_app_args": [
            "--rows", "32", "--cols", "16", "--stride", "4",
            "--seed", "42", "--shape-point", "trace_l2_accuracy_xl",
        ],
        "shape_file_sha256": "e7097c29a9474314e2d6ddc3939ff5127d885d179dadbf8ad84c8913a1fb40ab",
        "trace_cardinality_contract": "matrix_capture_args_override_v1",
    },
    "dram_streaming_dot_product": {
        "reason": "authority_args_multi_launch_default",
        "authority_cell_sha256": "e1a66776ba76826ca6ae799464401eb0599ddd08c44f97eaee3e8e04614aa976",
        "authority_shape_point": "trace_00",
        "authority_app_args_sha256": "d2944826d0e3f6c5eef0fe0ddcb9729a3fa149a3892bd6d332d6fd31d8667d26",
        "capture_shape_point": "trace_00",
        "label_app_args_sha256": "72604cbad1c5eb8c85289958ae68da440494bb6b12c91629671c411424444975",
        "trace_app_args_sha256": "d4b8e29f661b6469e47392c1ff8358023a227dc5893d755c2707c7f9f2b6c2c0",
        "label_app_args": [
            "--n", "64", "--passes", "1", "--blocks", "4",
            "--threads", "32", "--seed", "42", "--shape-point",
            "trace_00", "--warmup", "1", "--reps", "1",
        ],
        "trace_app_args": [
            "--n", "64", "--passes", "1", "--blocks", "4",
            "--threads", "32", "--seed", "42", "--shape-point",
            "trace_00", "--warmup", "0", "--reps", "1",
        ],
        "shape_file_sha256": "6325443e0cb12eff6c905e4c0b1a91c9db356a4c9280c2cd5452e297f149148f",
        "trace_cardinality_contract": "matrix_capture_args_override_v1",
        "label_launch_index": 1,
        "label_effective_warmup_count": 1,
    },
    "vendor_gemm_cublas": {
        # The authority's 32x32x32 BF16 shape dispatches to
        # cutlass_80_wmma_tensorop_s161616 and emits MOVM.16.MT88.  The
        # grandfathered H3 tracer classifies MOVM as TENSOR_CORE_OP and then
        # applies an Mx8xK parser, aborting on that valid MOVM modifier.  A live
        # pinned-tracer probe established 320x320x320 as the smallest practical
        # low-grid dispatch tested here that selects ampere_s16816 instead:
        # exactly 25/25 CTAs, 107400 decoded records, zero producer drops, and
        # 5.6 MiB raw trace.  This execution-only override keeps the BF16
        # cuBLAS workload, binary, frozen authority, and shape-file bytes fixed.
        "reason": "authority_shape_selects_h3_tracer_incompatible_movm_kernel",
        "authority_cell_sha256": "1e84352627f783b1d28fb3684cab96a61fa10cbdbb9c0e0b8844b027c1dc6142",
        "authority_shape_point": "trace_00",
        "authority_app_args_sha256": "db8a0731684b481fb0113e8f2f4b61e9bde73b1ee0361deca7f2de9a44ceedd7",
        "capture_shape_point": "trace_s16816_compatible_m320",
        "label_app_args_sha256": "4ed5773f1ff4e0a9fa17a4a8dbf3c82d03f51c05c934bb3cf4609c0d563c2da4",
        "trace_app_args_sha256": "4ed5773f1ff4e0a9fa17a4a8dbf3c82d03f51c05c934bb3cf4609c0d563c2da4",
        "label_app_args": [
            "--m", "320", "--n", "320", "--k", "320", "--seed", "42",
            "--shape-point", "trace_s16816_compatible_m320",
        ],
        "trace_app_args": [
            "--m", "320", "--n", "320", "--k", "320", "--seed", "42",
            "--shape-point", "trace_s16816_compatible_m320",
        ],
        "shape_file_sha256": "11978b5e83445fa03afa372b1769150f7efad49b30e89d264c173dbf5fcdabe1",
        "trace_cardinality_contract": "matrix_capture_args_override_v1",
    },
    "vendor_sgemm_cublas": {
        "reason": "label_driver_implicit_warmup_changes_launch_identity",
        "authority_cell_sha256": "328127e1ea186daea64cbc09527c954c5ab6bdb3453e8c38df0fd0122bcd294a",
        "authority_shape_point": "timing_shape",
        "authority_app_args_sha256": "6e103572b19ca3c52a41581cd9378423729fa6fe94f33da53e046992ec0ae587",
        "capture_shape_point": "timing_shape",
        "label_app_args_sha256": "f97d82f373c5a44d98a68cff5c9904b9352f00789555ff7d2bc6eb0d89572412",
        "trace_app_args_sha256": "8084ed10ecef2f5c5c5e7b9db4e47a4d9446e98ef1f6afb705c37a38d7b53bc8",
        "label_app_args": [
            "--m", "32", "--n", "32", "--k", "32", "--seed", "42",
            "--timing_shape", "--warmup", "1", "--reps", "1",
        ],
        "trace_app_args": [
            "--m", "32", "--n", "32", "--k", "32", "--seed", "42",
            "--timing_shape", "--warmup", "0", "--reps", "1",
        ],
        "shape_file_sha256": "d9f0e8db2a065ff1f90e4a556de4483dd4e56a9de80103734bd77ef0e998443c",
        "trace_cardinality_contract": "matrix_capture_args_override_v1",
        "label_launch_index": 1,
        "label_effective_warmup_count": 1,
    },
    "warm_cold_pair_relaunch": {
        "reason": "authority_workload_requires_cold_evict_warm_sequence",
        "authority_cell_sha256": "d26dac6a880cc3238a853409b58a8b1ef2c5dd642d730b02858a620f2ca14625",
        "authority_shape_point": "trace_00",
        "authority_app_args_sha256": "b2a11f936ddd3ee7658f9e7c3cc2aa190c9a656b8d28f8e69a885e4c5557b7a8",
        "capture_shape_point": "trace_00",
        "label_app_args_sha256": "b2a11f936ddd3ee7658f9e7c3cc2aa190c9a656b8d28f8e69a885e4c5557b7a8",
        "trace_app_args_sha256": "b2a11f936ddd3ee7658f9e7c3cc2aa190c9a656b8d28f8e69a885e4c5557b7a8",
        "label_app_args": [
            "--allocation-bytes", "524288", "--accesses", "256",
            "--eviction-bytes", "1048576", "--warps", "32", "--seed",
            "42", "--shape-point", "trace_00",
        ],
        "trace_app_args": [
            "--allocation-bytes", "524288", "--accesses", "256",
            "--eviction-bytes", "1048576", "--warps", "32", "--seed",
            "42", "--shape-point", "trace_00",
        ],
        "shape_file_sha256": "06a40de82b0cf4c04c759be42f77a87256faa37056a37d2ccf97241aa877f4a1",
        "trace_cardinality_contract": "warm_cold_pair_three_launch_v1",
        "trace_launch_sequence": [
            {
                "device_id": 0, "stream_id": 0, "kernel_id": 1,
                "kernel_name": "warm_cold_pair_kernel___0",
                "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
                "role": "cold",
            },
            {
                "device_id": 0, "stream_id": 0, "kernel_id": 2,
                "kernel_name": "warm_cold_pair_eviction_kernel___0",
                "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
                "role": "evict",
            },
            {
                "device_id": 0, "stream_id": 0, "kernel_id": 3,
                "kernel_name": "warm_cold_pair_kernel___0",
                "grid_dim": [32, 1, 1], "block_dim": [32, 1, 1],
                "role": "warm",
            },
        ],
    },
}


# The three cuBLAS launchers contain only host-side adapter code; their accepted
# kernels execute from the pinned libcublasLt runtime module.  Static control
# evidence therefore comes from one exact raw SASS text section selected only
# after the live NCU/NVBit captures bind the runtime kernel.  The FP32 launches'
# frozen tracer cannot resolve their module origins, so each same-cell NCU report
# must contain its exact raw section once before the pinned module fallback is
# admitted.  These are closed, host-pinned campaign contracts, not ambient
# library discovery rules.
VENDOR_RUNTIME_CONTROL_CONTRACTS = {
    "vendor_gemm_cublas": {
        "policy": "post_trace_vendor_module",
        "module_discovery": "nvbit_runtime_module_inventory",
        "selection_contract": "active_from_start_single_launch_exact",
        "runtime_kernel": (
            "ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_nn___0"
        ),
        "function_name": "ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_nn",
        "execution_arch": "sm_86",
        "code_arch": "sm_80",
        "inventory_binding": "nvbit_stable_static_module",
        "expected_outcome": "runtime_controls_emitted",
        "expected_inventory_status": "stable_static_module",
        "expected_inventory_reason": "resolved_file",
        "module_path_at_capture": (
            "/opt/cuda/targets/x86_64-linux/lib/libcublasLt.so.13.5.1.27"
        ),
        "module_basename": "libcublasLt.so.13.5.1.27",
        "module_sha256": (
            "db5c748bc095423401029c00294fb8eb3c721c75c363f3d8b744b7c274504ae0"
        ),
        "text_section_index": 41835,
        "text_section_name": (
            "x-ampere_s16816gemm_bf16_64x64_ldg8_stages_64x5_nn.sm_80.elf.bin"
        ),
        "raw_text_size_bytes": 11136,
        "raw_text_sha256": (
            "1dad71996bb6f7a022d3f0ddc282980445a96a49b24b3db28b0e3686e1492cd8"
        ),
        "static_instruction_count": 696,
        "static_pcs_sha256": (
            "785952ecd701998345ae8c30f90563202cecf6a38af574743f7249d472c801c9"
        ),
        "ncu_report_binding": "raw_text_section_exactly_once",
        "toolkit_version": "V13.2",
        "nvcc_path": "/usr/local/cuda-13.2/bin/nvcc",
        "nvcc_sha256": (
            "1b32c567e18698a94d78f18e571d7a213cfb461a7422505120e88748d5150136"
        ),
        "cuobjdump_path": "/usr/local/cuda-13.2/bin/cuobjdump",
        "cuobjdump_sha256": (
            "903c5e1324669c1222b13ede0fc7226f48f7db355669f2912f0236031c56e1bf"
        ),
    },
    "vendor_sgemm_cublas": {
        "policy": "post_trace_vendor_module",
        "module_discovery": "nvbit_runtime_module_inventory",
        "selection_contract": "active_from_start_single_launch_exact",
        "runtime_kernel": "ampere_sgemm_128x32_nn___0",
        "function_name": "ampere_sgemm_128x32_nn",
        "execution_arch": "sm_86",
        "code_arch": "sm_80",
        "inventory_binding": "ncu_report_exact_text_section",
        "expected_outcome": "matrix_pinned_runtime_module_controls_emitted",
        "expected_inventory_status": "no_stable_static_module",
        "expected_inventory_reason": "origin_unresolved",
        "module_path_at_capture": (
            "/opt/cuda/targets/x86_64-linux/lib/libcublasLt.so.13.5.1.27"
        ),
        "module_basename": "libcublasLt.so.13.5.1.27",
        "module_sha256": (
            "db5c748bc095423401029c00294fb8eb3c721c75c363f3d8b744b7c274504ae0"
        ),
        "text_section_index": 41599,
        "text_section_name": "x-ampere_sgemm_128x32_nn.sm_80.elf.bin",
        "raw_text_size_bytes": 9984,
        "raw_text_sha256": (
            "de3847a9b4933456201ac7b5eefe849c0b2a21ced67a6d68bd4fdaf92a64a54c"
        ),
        "static_instruction_count": 624,
        "static_pcs_sha256": (
            "92ed1ee86a68ad09a0643b57d0627992efcbf91f6089bfd01dad9d09019c7d28"
        ),
        "ncu_report_binding": "raw_text_section_exactly_once",
        "toolkit_version": "V13.2",
        "nvcc_path": "/usr/local/cuda-13.2/bin/nvcc",
        "nvcc_sha256": (
            "1b32c567e18698a94d78f18e571d7a213cfb461a7422505120e88748d5150136"
        ),
        "cuobjdump_path": "/usr/local/cuda-13.2/bin/cuobjdump",
        "cuobjdump_sha256": (
            "903c5e1324669c1222b13ede0fc7226f48f7db355669f2912f0236031c56e1bf"
        ),
    },
    "vendor_sgemm_strided_batched_cublas": {
        "policy": "post_trace_vendor_module",
        "module_discovery": "nvbit_runtime_module_inventory",
        "selection_contract": "active_from_start_single_launch_exact",
        "runtime_kernel": "ampere_sgemm_128x128_nn___0",
        "function_name": "ampere_sgemm_128x128_nn",
        "execution_arch": "sm_86",
        "code_arch": "sm_80",
        "inventory_binding": "ncu_report_exact_text_section",
        "expected_outcome": "matrix_pinned_runtime_module_controls_emitted",
        "expected_inventory_status": "no_stable_static_module",
        "expected_inventory_reason": "origin_unresolved",
        "module_path_at_capture": (
            "/opt/cuda/targets/x86_64-linux/lib/libcublasLt.so.13.5.1.27"
        ),
        "module_basename": "libcublasLt.so.13.5.1.27",
        "module_sha256": (
            "db5c748bc095423401029c00294fb8eb3c721c75c363f3d8b744b7c274504ae0"
        ),
        "text_section_index": 41591,
        "text_section_name": "x-ampere_sgemm_128x128_nn.sm_80.elf.bin",
        "raw_text_size_bytes": 19328,
        "raw_text_sha256": (
            "3744493afe93632f667f166e1168dc8047c56ebd938296f0b46cd9f57b3ed213"
        ),
        "static_instruction_count": 1208,
        "static_pcs_sha256": (
            "7aed0eaad0c35a4669f40a51dc383553323c6786cea3e180bc2aac4d8b2bc290"
        ),
        "ncu_report_binding": "raw_text_section_exactly_once",
        "toolkit_version": "V13.2",
        "nvcc_path": "/usr/local/cuda-13.2/bin/nvcc",
        "nvcc_sha256": (
            "1b32c567e18698a94d78f18e571d7a213cfb461a7422505120e88748d5150136"
        ),
        "cuobjdump_path": "/usr/local/cuda-13.2/bin/cuobjdump",
        "cuobjdump_sha256": (
            "903c5e1324669c1222b13ede0fc7226f48f7db355669f2912f0236031c56e1bf"
        ),
    },
}


class MatrixBuildError(ValueError):
    """A fail-closed matrix invariant was not satisfied."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_static_control_toolchain(contract: dict) -> None:
    if contract != SOL_STATIC_CONTROL_REBIND:
        raise MatrixBuildError("SOL static-control rebind contract drift")
    for role, path_field, sha_field in (
        ("pinned static-control nvcc", "nvcc_path", "nvcc_sha256"),
        ("pinned static-control cuobjdump", "cuobjdump_path", "cuobjdump_sha256"),
    ):
        raw_path = contract[path_field]
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise MatrixBuildError(f"{role} is not a regular absolute file: {raw_path}")
        actual = sha256_file(path)
        expected = contract[sha_field]
        if actual != expected:
            raise MatrixBuildError(
                f"{role} hash mismatch: {raw_path} "
                f"expected={expected} actual={actual}"
            )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_fingerprint(value: Any) -> tuple:
    """Return a type-sensitive fingerprint for already-decoded JSON values."""
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return ("float", value.hex())
    if type(value) is str:
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_strict_json_fingerprint(item) for item in value))
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise TypeError("JSON object keys must be strings")
        return (
            "object",
            tuple(
                (key, _strict_json_fingerprint(item))
                for key, item in sorted(value.items())
            ),
        )
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _strict_json_equal(left: Any, right: Any) -> bool:
    try:
        return _strict_json_fingerprint(left) == _strict_json_fingerprint(right)
    except (TypeError, ValueError):
        return False


def _repo_rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _portable_path_ref(path: Path, repo_root: Path) -> str:
    """Prefer a repo-relative pin; preserve an external live-session path."""
    try:
        return _repo_rel(path, repo_root)
    except ValueError:
        return str(path.resolve())


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise MatrixBuildError(f"cannot resolve git HEAD: {proc.stderr.strip()}")
    head = proc.stdout.strip()
    if len(head) != 40:
        raise MatrixBuildError(f"invalid git HEAD {head!r}")
    return head


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixBuildError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixBuildError(f"JSON root must be an object: {path}")
    return value


def tree_sha256(path: Path) -> str:
    """Hash a directory as sorted (relative path, file SHA) pairs."""
    if not path.is_dir():
        raise MatrixBuildError(f"required directory absent: {path}")
    rows = []
    for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if child.is_symlink():
            rows.append({
                "path": child.relative_to(path).as_posix(),
                "symlink": os.readlink(child),
            })
        elif child.is_file():
            rows.append({
                "path": child.relative_to(path).as_posix(),
                "sha256": sha256_file(child),
            })
    if not rows:
        raise MatrixBuildError(f"required directory is empty: {path}")
    return canonical_sha256(rows)


def _identity(cell: dict) -> dict:
    source = cell.get("roster_source")
    slug = cell.get("slug")
    if source not in SOURCE_ORDER or not isinstance(slug, str) or not slug:
        raise MatrixBuildError(f"invalid authority identity: {(source, slug)!r}")
    return {"roster_source": source, "slug": slug}


def _authority_invariants(authority: dict, *, repo_root: Path,
                          require_head_match: bool) -> None:
    if authority.get("arch") != ARCH or authority.get("sku") != SKU:
        raise MatrixBuildError("authority arch/sku mismatch")
    if not authority.get("accounting_identity_holds"):
        raise MatrixBuildError("authority accounting identity is not true")
    cells = authority.get("cells")
    counts = authority.get("counts")
    if not isinstance(cells, list) or not cells:
        raise MatrixBuildError("authority has zero cells")
    if not isinstance(counts, dict):
        raise MatrixBuildError("authority counts absent")
    universe = counts.get("universe")
    if universe != EXPECTED_UNIVERSE or universe != len(cells):
        raise MatrixBuildError(
            f"authority universe mismatch: declared={universe!r} rows={len(cells)} "
            f"expected={EXPECTED_UNIVERSE}"
        )
    seen: set[tuple[str, str]] = set()
    observed = {status: 0 for status in FINAL_DISPOSITIONS | PENDING_DISPOSITIONS}
    for cell in cells:
        ident = _identity(cell)
        key = (ident["roster_source"], ident["slug"])
        if key in seen:
            raise MatrixBuildError(f"duplicate authority identity: {key!r}")
        seen.add(key)
        status = cell.get("resolution_status")
        if status not in observed:
            raise MatrixBuildError(f"unknown authority disposition {status!r}: {key!r}")
        observed[status] += 1
    for status, n in observed.items():
        if counts.get(status) != n:
            raise MatrixBuildError(
                f"authority {status} count mismatch: declared={counts.get(status)!r} actual={n}"
            )
    pending = sum(observed[s] for s in PENDING_DISPOSITIONS)
    if pending:
        raise MatrixBuildError(f"authority still has {pending} pending rows")
    if observed["runnable"] <= 0 or observed["excluded"] <= 0:
        raise MatrixBuildError("authority runnable/excluded partition is vacuous")
    source_commit = authority.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise MatrixBuildError("authority source_commit is invalid")
    if require_head_match and source_commit != _git_head(repo_root):
        raise MatrixBuildError(
            f"stale authority source_commit={source_commit} head={_git_head(repo_root)}"
        )


def _metric_contract(registry: dict) -> dict:
    metrics = registry.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise MatrixBuildError("metric registry is empty")
    if registry.get("registry_version") is None or registry.get("label_schema_version") is None:
        raise MatrixBuildError("metric registry version pins absent")
    identities = []
    seen_keys: set[str] = set()
    seen_ncu: set[str] = set()
    selected_flat = []
    selected_unit = []
    not_collected = []
    for metric in metrics:
        key = metric.get("key")
        ncu = metric.get("ncu_metric")
        if not isinstance(key, str) or not key or key in seen_keys:
            raise MatrixBuildError(f"invalid/duplicate metric key {key!r}")
        if not isinstance(ncu, str) or not ncu or ncu in seen_ncu:
            raise MatrixBuildError(f"invalid/duplicate NCU metric {ncu!r}")
        seen_keys.add(key)
        seen_ncu.add(ncu)
        per_arch = (metric.get("per_arch_availability") or {}).get(ARCH)
        availability = metric.get("availability")
        carrier = metric.get("carrier")
        if availability != "excluded" and per_arch == "available":
            capture_status = "selected"
            if carrier == "flat_field":
                selected_flat.append(ncu)
            elif carrier == "unit_tagged_metric":
                selected_unit.append(ncu)
            else:
                raise MatrixBuildError(f"unknown carrier for selected metric {key!r}: {carrier!r}")
        elif per_arch == "requires_runtime_carveout":
            capture_status = "requires_runtime_carveout"
            not_collected.append({"key": key, "reason": capture_status})
        elif per_arch != "available":
            capture_status = "unavailable_on_sm86"
            not_collected.append({"key": key, "reason": capture_status})
        else:
            capture_status = "static_derived_not_ncu_collected"
            not_collected.append({"key": key, "reason": capture_status})
        identities.append({
            "key": key,
            "ncu_metric": ncu,
            "unit": metric.get("unit"),
            "carrier": carrier,
            "availability": availability,
            "sm86_availability": per_arch,
            "capture_status": capture_status,
        })
    identities.sort(key=lambda row: row["key"])
    selected_flat.sort()
    selected_unit.sort()
    not_collected.sort(key=lambda row: row["key"])
    return {
        "registry_version": registry["registry_version"],
        "label_schema_version": registry["label_schema_version"],
        "metric_count": len(identities),
        "metric_identity_sha256": canonical_sha256(identities),
        "metrics": identities,
        "collection": {
            "selected_registry_metric_count": len(selected_flat) + len(selected_unit),
            "selected_flat_metrics": selected_flat,
            "selected_unit_metrics": selected_unit,
            "supporting_raw_metrics": list(DRIVER_SUPPORTING_RAW),
            "not_collected_registry_metrics": not_collected,
        },
    }


def _budget_record(slug: str, *, budget_by_id: dict, source_sha: str) -> dict:
    base = {
        "source_path": TRACE_BUDGETS_PATH.as_posix(),
        "source_sha256": source_sha,
    }
    # require_cell_budget() keys this canonical source by the campaign cell_id.
    # The full runnable matrix additionally enforces global slug uniqueness, so
    # an explicit row cannot silently bind to two source suites.
    budget = budget_by_id.get(slug)
    if budget is None:
        return {
            **base,
            "status": "missing",
            "reason": "trace_budget_missing",
        }
    return {
        **base,
        "status": "declared",
        "max_trace_bytes": budget.max_trace_bytes,
        "max_wall_time_s": budget.max_wall_time_s,
        "sampling_policy": budget.sampling_policy,
    }


def _readiness(trace_budget: dict) -> dict:
    blockers = []
    if trace_budget["status"] != "declared":
        blockers.append("trace_budget_missing")
    elif trace_budget["sampling_policy"] != "off":
        # Fixed-design reference capture requires an explicitly unsampled row.
        blockers.append("legacy_sampling_policy_not_authorized")
    return {
        "status": "ready" if not blockers else "blocked",
        "blocking_reasons": blockers,
    }


def _exact_nonnegative_arg(args: list[str], flag: str) -> int:
    positions = [index for index, token in enumerate(args) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        raise MatrixBuildError(f"capture override requires exactly one {flag}")
    value = args[positions[0] + 1]
    if not value.isdigit():
        raise MatrixBuildError(f"capture override {flag} must be non-negative integer")
    return int(value)


def _direct_invocation(cell: dict, repo_root: Path) -> tuple[dict, dict]:
    source = cell["roster_source"]
    rel = cell.get("binary")
    expected_sha = cell.get("binary_sha256")
    args = cell.get("app_args")
    shape = cell.get("shape_point")
    if not isinstance(rel, str) or not rel or not isinstance(expected_sha, str):
        raise MatrixBuildError(f"runnable direct row lacks binary identity: {_identity(cell)}")
    if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
        raise MatrixBuildError(f"runnable direct row lacks app args: {_identity(cell)}")
    if not isinstance(shape, str) or not shape:
        raise MatrixBuildError(f"runnable direct row lacks shape: {_identity(cell)}")
    path = repo_root / rel
    if not path.is_file() or not os.access(path, os.X_OK):
        raise MatrixBuildError(f"runnable executable absent/not executable: {rel}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise MatrixBuildError(
            f"runnable executable hash mismatch: {rel} expected={expected_sha} actual={actual_sha}"
        )
    label_args = list(args)
    trace_args = list(args)
    capture_shape = shape
    trace_override = "authority_args_exact"
    matrix_override = None
    launch_contract = {
        "schema_version": 1,
        "label_driver_warmup_count": 1,
        "label_effective_warmup_count": 0,
        "label_launch_index": 0,
        "label_measured_reps": 1,
        "trace_launch_count": 1,
        "label_trace_launch_index": 0,
    }
    if source == "portable_lcd":
        if "--warmup" in trace_args or "--reps" in trace_args:
            raise MatrixBuildError(f"portable authority already overrides trace cardinality: {_identity(cell)}")
        label_args += ["--warmup", "1", "--reps", "1"]
        trace_args += ["--warmup", "0", "--reps", "1"]
        trace_override = "runner_common_single_launch_v1"
        launch_contract.update({
            "label_effective_warmup_count": 1,
            "label_launch_index": 1,
        })
    elif source == "simbench_bw" and cell["slug"] in SIMBENCH_CAPTURE_OVERRIDES:
        override = SIMBENCH_CAPTURE_OVERRIDES[cell["slug"]]
        observed_cell_sha = canonical_sha256(cell)
        observed_args_sha = canonical_sha256(args)
        if observed_cell_sha != override["authority_cell_sha256"]:
            raise MatrixBuildError(
                f"capture override authority cell drift: {_identity(cell)} "
                f"expected={override['authority_cell_sha256']} actual={observed_cell_sha}"
            )
        if shape != override["authority_shape_point"]:
            raise MatrixBuildError(
                f"capture override authority shape drift: {_identity(cell)} "
                f"expected={override['authority_shape_point']!r} actual={shape!r}"
            )
        if observed_args_sha != override["authority_app_args_sha256"]:
            raise MatrixBuildError(
                f"capture override authority args drift: {_identity(cell)} "
                f"expected={override['authority_app_args_sha256']} actual={observed_args_sha}"
            )
        shape_file = Path("workloads/simbench_bw/shapes") / f"{cell['slug']}.toml"
        expected_shape_sha = override["shape_file_sha256"]
        if cell.get("shape_file_sha256") != expected_shape_sha:
            raise MatrixBuildError(
                f"capture override authority shape-file pin drift: {_identity(cell)}"
            )
        actual_shape_sha = sha256_file(repo_root / shape_file)
        if actual_shape_sha != expected_shape_sha:
            raise MatrixBuildError(
                f"capture override live shape-file drift: {shape_file} "
                f"expected={expected_shape_sha} actual={actual_shape_sha}"
            )
        label_args = list(override["label_app_args"])
        trace_args = list(override["trace_app_args"])
        label_args_sha = canonical_sha256(label_args)
        trace_args_sha = canonical_sha256(trace_args)
        if label_args_sha != override["label_app_args_sha256"]:
            raise MatrixBuildError(
                f"capture override label args definition drift: {_identity(cell)}"
            )
        if trace_args_sha != override["trace_app_args_sha256"]:
            raise MatrixBuildError(
                f"capture override trace args definition drift: {_identity(cell)}"
            )
        capture_shape = override["capture_shape_point"]
        trace_override = override["trace_cardinality_contract"]
        launch_contract.update({
            "label_effective_warmup_count": override.get(
                "label_effective_warmup_count", 0
            ),
            "label_launch_index": override.get("label_launch_index", 0),
        })
        sequence = override.get("trace_launch_sequence")
        if sequence is not None:
            warps = _exact_nonnegative_arg(trace_args, "--warps")
            eviction_bytes = _exact_nonnegative_arg(trace_args, "--eviction-bytes")
            if warps <= 0 or eviction_bytes <= 0:
                raise MatrixBuildError(
                    "warm/cold capture override requires positive warps and eviction bytes"
                )
            derived_sequence = [
                {
                    "device_id": 0, "stream_id": 0, "kernel_id": index + 1,
                    "kernel_name": kernel_name,
                    "grid_dim": [warps, 1, 1], "block_dim": [32, 1, 1],
                    "role": role,
                }
                for index, (kernel_name, role) in enumerate((
                    ("warm_cold_pair_kernel___0", "cold"),
                    ("warm_cold_pair_eviction_kernel___0", "evict"),
                    ("warm_cold_pair_kernel___0", "warm"),
                ))
            ]
            if sequence != derived_sequence:
                raise MatrixBuildError(
                    "warm/cold trace launch sequence does not derive from capture args"
                )
            launch_contract["trace_launch_count"] = len(sequence)
            launch_contract["trace_launch_sequence"] = copy.deepcopy(sequence)
        matrix_override = {
            "schema_version": 1,
            "reason": override["reason"],
            "authority_cell_sha256": observed_cell_sha,
            "authority_shape_point": shape,
            "authority_app_args_sha256": observed_args_sha,
            "capture_shape_point": capture_shape,
            "label_app_args_sha256": label_args_sha,
            "trace_app_args_sha256": trace_args_sha,
            "shape_file_path": shape_file.as_posix(),
            "shape_file_sha256": actual_shape_sha,
        }
    invocation = {
        "adapter_kind": "direct_binary",
        "authority_reference_path": rel,
        "authority_reference_sha256": expected_sha,
        "executable_path": rel,
        "executable_sha256": actual_sha,
        "label_app_args": label_args,
        "trace_app_args": trace_args,
        "trace_cardinality_contract": trace_override,
        "capture_launch_contract": launch_contract,
        "logical_workload": {"kind": "shape_point", "value": capture_shape},
        "shape_delivery": "application_args",
        # label_driver.sh uses --target-processes all for every capture. Direct
        # binaries have no child process, but the matrix records the literal
        # profiler contract rather than a narrower behavioral approximation.
        "target_processes": "all",
    }
    if matrix_override is not None:
        invocation["matrix_capture_override"] = matrix_override
    vendor_control = VENDOR_RUNTIME_CONTROL_CONTRACTS.get(cell["slug"])
    if source == "simbench_bw" and vendor_control is not None:
        for role, raw_path, expected_sha in (
            (
                "vendor runtime module",
                vendor_control["module_path_at_capture"],
                vendor_control["module_sha256"],
            ),
            ("pinned nvcc", vendor_control["nvcc_path"], vendor_control["nvcc_sha256"]),
            (
                "pinned cuobjdump",
                vendor_control["cuobjdump_path"],
                vendor_control["cuobjdump_sha256"],
            ),
        ):
            pinned_path = Path(raw_path)
            if (
                not pinned_path.is_absolute()
                or pinned_path.is_symlink()
                or not pinned_path.is_file()
            ):
                raise MatrixBuildError(f"{role} is not a pinned regular file: {raw_path}")
            observed_sha = sha256_file(pinned_path)
            if observed_sha != expected_sha:
                raise MatrixBuildError(
                    f"{role} hash mismatch: {raw_path} "
                    f"expected={expected_sha} actual={observed_sha}"
                )
        control = copy.deepcopy(vendor_control)
    else:
        control = {
            "policy": "direct_executable",
            "input_path": rel,
            "input_sha256": actual_sha,
        }
    return invocation, control


def _count_workloads(path: Path) -> int:
    if not path.is_file():
        raise MatrixBuildError(f"SOL workload file absent: {path}")
    count = 0
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise MatrixBuildError(f"cannot read SOL workload file {path}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            workload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MatrixBuildError(
                f"invalid SOL workload JSON {path}:{lineno}: {exc}"
            ) from exc
        if not isinstance(workload, dict):
            raise MatrixBuildError(
                f"SOL workload row must be an object: {path}:{lineno}"
            )
        count += 1
    return count


def _repo_relative_file(rel: str, repo_root: Path, *, label: str) -> Path:
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise MatrixBuildError(f"{label} must be repo-relative: {rel!r}")
    path = (repo_root / candidate).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise MatrixBuildError(f"{label} escapes repository: {rel!r}") from exc
    if not path.is_file():
        raise MatrixBuildError(f"{label} absent: {rel}")
    return path


def _sol_shrink_contract(repo_root: Path) -> dict:
    source = repo_root / SOL_SHRINK_PROVENANCE_PATH
    doc = _load_json(source)
    if (
        doc.get("schema_version") != 1
        or doc.get("artifact_id") != "sm86_sol_shrink_provenance_v1"
        or doc.get("arch") != ARCH
        or doc.get("sku") != SKU
    ):
        raise MatrixBuildError("SOL shrink provenance identity mismatch")
    native_env_hash = doc.get("native_env_hash")
    if (
        not isinstance(native_env_hash, str)
        or len(native_env_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in native_env_hash)
    ):
        raise MatrixBuildError("SOL shrink provenance native environment is invalid")
    rows = doc.get("cells")
    if not isinstance(rows, list) or not rows:
        raise MatrixBuildError("SOL shrink provenance cells are absent")
    by_slug = {}
    for row in rows:
        if not isinstance(row, dict):
            raise MatrixBuildError("SOL shrink provenance row must be an object")
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug:
            raise MatrixBuildError("SOL shrink provenance row lacks slug")
        if slug in by_slug:
            raise MatrixBuildError(f"duplicate SOL shrink provenance slug: {slug}")
        by_slug[slug] = row
    return {
        "source_sha256": sha256_file(source),
        "native_env_hash": native_env_hash,
        "rows_by_slug": by_slug,
    }


def _sol_runnability_contract(repo_root: Path) -> dict:
    source = repo_root / SOL_RUNNABILITY_PATH
    doc = _load_json(source)
    if (
        doc.get("schema_version") != 1
        or doc.get("artifact_id") != "sm86_sol_runnability_v1"
        or doc.get("arch") != ARCH
        or doc.get("sku") != SKU
    ):
        raise MatrixBuildError("SOL runnability identity mismatch")
    native_env_hash = doc.get("native_env_hash")
    if (
        not isinstance(native_env_hash, str) or len(native_env_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in native_env_hash)
    ):
        raise MatrixBuildError("SOL runnability native environment is invalid")
    rows = doc.get("cells")
    if not isinstance(rows, list) or not rows:
        raise MatrixBuildError("SOL runnability cells are absent")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            raise MatrixBuildError("SOL runnability row identity is invalid")
        if row["slug"] in indexed:
            raise MatrixBuildError(f"duplicate SOL runnability slug: {row['slug']}")
        indexed[row["slug"]] = row
    return {
        "source_sha256": sha256_file(source),
        "native_env_hash": native_env_hash,
        "rows_by_slug": indexed,
    }


def _sol_native_env_contract(repo_root: Path) -> dict:
    """Validate and rederive the immutable SOL native-environment pin."""
    source = repo_root / SOL_NATIVE_ENV_FINGERPRINT_PATH
    doc = _load_json(source)
    if (
        doc.get("schema_version") != 1
        or doc.get("fingerprint_id") != "sm86_sol_native_env_fingerprint_v1"
        or doc.get("arch") != ARCH
        or doc.get("sku") != SKU
    ):
        raise MatrixBuildError("SOL native environment fingerprint identity mismatch")
    components = doc.get("components")
    if not isinstance(components, dict) or set(components) != set(
        NATIVE_ENV_COMPONENT_ORDER
    ):
        raise MatrixBuildError(
            "SOL native environment fingerprint component set mismatch"
        )
    if not all(isinstance(components[name], str) for name in NATIVE_ENV_COMPONENT_ORDER):
        raise MatrixBuildError(
            "SOL native environment fingerprint components must be strings"
        )
    derived_missing = missing_components(components)
    if (
        doc.get("degraded") is not False
        or doc.get("missing_components") != derived_missing
        or derived_missing
    ):
        raise MatrixBuildError("SOL native environment fingerprint is degraded")
    native_env_hash = doc.get("native_env_hash")
    derived_hash = compute_native_env_hash(components)
    if native_env_hash != derived_hash:
        raise MatrixBuildError(
            "SOL native environment fingerprint hash mismatch: "
            f"declared={native_env_hash!r} derived={derived_hash}"
        )
    if components.get("nvcc_ccbin") != SOL_PINNED_NVCC_CCBIN:
        raise MatrixBuildError(
            "SOL native environment compiler pin mismatch: "
            f"declared={components.get('nvcc_ccbin')!r} "
            f"required={SOL_PINNED_NVCC_CCBIN!r}"
        )
    return {
        "source_sha256": sha256_file(source),
        "native_env_hash": native_env_hash,
        "components": copy.deepcopy(components),
    }


def _validate_sol_forwarded_argv(argv: list[str], *, ident: dict) -> None:
    if not argv or not all(isinstance(token, str) and token for token in argv):
        raise MatrixBuildError(f"SOL forwarded argv is invalid: {ident}")
    if any(
        token == "--shape-point" or token.startswith("--shape-point=")
        for token in argv
    ):
        raise MatrixBuildError(f"SOL forwarded argv contains a fabricated shape token: {ident}")


def _sol_invocation(cell: dict, repo_root: Path, *,
                    shrink_contract: dict | None = None,
                    runnability_contract: dict | None = None,
                    native_env_contract: dict | None = None) -> tuple[dict, dict]:
    ident = _identity(cell)
    solution_rel = cell.get("binary")
    expected_sha = cell.get("binary_sha256")
    category = cell.get("category")
    native_env_hash = cell.get("native_env_hash")
    shape = cell.get("shape_point")
    if not isinstance(solution_rel, str) or not isinstance(expected_sha, str):
        raise MatrixBuildError(f"SOL row lacks solution identity: {ident}")
    if (
        category not in SOL_SUBSET_MAP
        or not isinstance(native_env_hash, str)
        or len(native_env_hash) != 64
    ):
        raise MatrixBuildError(f"SOL row lacks category/native environment: {ident}")
    if shape not in {SOL_NATIVE_SHAPE, SOL_REDUCED_SHAPE}:
        raise MatrixBuildError(f"SOL row has unsupported/missing workload shape: {ident}")
    native_env_contract = native_env_contract or _sol_native_env_contract(repo_root)
    if native_env_contract.get("native_env_hash") != native_env_hash:
        raise MatrixBuildError(
            f"SOL fingerprint/native environment mismatch: {ident}"
        )
    solution = _repo_relative_file(solution_rel, repo_root, label="SOL solution")
    actual_solution_sha = sha256_file(solution)
    if actual_solution_sha != expected_sha:
        raise MatrixBuildError(
            f"SOL solution hash mismatch: {solution_rel} "
            f"expected={expected_sha} actual={actual_solution_sha}"
        )
    expected_prefix = SOL_BASELINES_ROOT / category / ident["slug"]
    try:
        solution.relative_to(repo_root / expected_prefix)
    except ValueError as exc:
        raise MatrixBuildError(f"SOL solution path/category mismatch: {ident}") from exc
    solution_doc = _load_json(solution)
    definition = solution_doc.get("definition")
    if not isinstance(definition, str) or not definition or "/" in definition or ".." in definition:
        raise MatrixBuildError(f"SOL solution definition invalid: {ident}")
    benchmark_rel = SOL_BENCHMARK_ROOT / SOL_SUBSET_MAP[category] / definition
    benchmark = repo_root / benchmark_rel
    definition_path = benchmark / "definition.json"
    if not definition_path.is_file():
        raise MatrixBuildError(f"SOL benchmark definition absent: {benchmark_rel}")
    definition_doc = _load_json(definition_path)
    if definition_doc.get("name") != definition:
        raise MatrixBuildError(
            f"SOL solution/benchmark definition mismatch: {ident}"
        )
    native_workload = benchmark / "workload.jsonl"
    native_workload_rel = _repo_rel(native_workload, repo_root)

    shrink_contract = shrink_contract or _sol_shrink_contract(repo_root)
    if shrink_contract.get("native_env_hash") != native_env_hash:
        raise MatrixBuildError(f"SOL shrink/native environment mismatch: {ident}")
    shrink_row = (shrink_contract.get("rows_by_slug") or {}).get(ident["slug"])
    reduced_provenance = None
    if shape == SOL_NATIVE_SHAPE:
        if shrink_row is not None:
            raise MatrixBuildError(
                f"SOL native workload is misrouted from shrink provenance: {ident}"
            )
        workload_scope = "native"
        workload = native_workload
        argv = [benchmark_rel.as_posix(), "--solution", solution_rel, "--json"]
    else:
        if not isinstance(shrink_row, dict):
            raise MatrixBuildError(f"SOL reduced workload lacks shrink provenance: {ident}")
        required_reduced = {
            "slug": ident["slug"],
            "category": category,
            "classification": SOL_REDUCED_SHAPE,
            "native_shape_ref": native_workload_rel,
            "reduced_run_status": "runnable",
            "correctness_result": "passed",
        }
        for key, expected in required_reduced.items():
            if shrink_row.get(key) != expected:
                raise MatrixBuildError(
                    f"SOL reduced provenance {key} mismatch: {ident}"
                )
        reduced_rel = shrink_row.get("reduced_workload_ref")
        if not isinstance(reduced_rel, str) or not reduced_rel:
            raise MatrixBuildError(f"SOL reduced workload reference absent: {ident}")
        workload = _repo_relative_file(
            reduced_rel, repo_root, label="SOL reduced workload",
        )
        reduced_root = (
            repo_root / "data/local_labels/campaign/sol_reduced_workloads" / category
        ).resolve()
        try:
            workload.relative_to(reduced_root)
        except ValueError as exc:
            raise MatrixBuildError(
                f"SOL reduced workload/category mismatch: {ident}"
            ) from exc
        workload_scope = SOL_REDUCED_SHAPE
        argv = [
            "--definition", _repo_rel(definition_path, repo_root),
            "--workload", reduced_rel,
            "--solution", solution_rel,
            "--json",
        ]
        reduced_provenance = {
            "source_path": SOL_SHRINK_PROVENANCE_PATH.as_posix(),
            "source_sha256": shrink_contract["source_sha256"],
            "row_sha256": canonical_sha256(shrink_row),
            "classification": SOL_REDUCED_SHAPE,
            "native_shape_ref": native_workload_rel,
            "reduced_workload_ref": reduced_rel,
            "reduced_run_status": "runnable",
            "correctness_result": "passed",
        }

    expected_workloads = _count_workloads(workload)
    if expected_workloads <= 0:
        raise MatrixBuildError(f"SOL workload has zero rows: {_repo_rel(workload, repo_root)}")
    runnability_contract = runnability_contract or _sol_runnability_contract(repo_root)
    if runnability_contract.get("native_env_hash") != native_env_hash:
        raise MatrixBuildError(f"SOL runnability/native environment mismatch: {ident}")
    runnability_row = (runnability_contract.get("rows_by_slug") or {}).get(
        ident["slug"]
    )
    if not isinstance(runnability_row, dict):
        raise MatrixBuildError(f"SOL runnable row lacks empirical evidence: {ident}")
    common_evidence = {
        "slug": ident["slug"], "category": category,
        "expected_workloads": expected_workloads,
    }
    for field, expected in common_evidence.items():
        if runnability_row.get(field) != expected:
            raise MatrixBuildError(
                f"SOL runnability {field} mismatch for {ident}: "
                f"{runnability_row.get(field)!r} != {expected!r}"
            )
    if shape == SOL_NATIVE_SHAPE:
        if runnability_row.get("status") != "runnable" or runnability_row.get(
            "returncode"
        ) != 0:
            raise MatrixBuildError(f"SOL native row is not empirically runnable: {ident}")
        detail = runnability_row.get("detail")
        if detail != f"{expected_workloads}/{expected_workloads} workloads passed":
            raise MatrixBuildError(f"SOL runnability completeness mismatch: {ident}")
        evaluation_evidence = {
            "source_kind": "native_runnability",
            "source_path": SOL_RUNNABILITY_PATH.as_posix(),
            "source_sha256": runnability_contract["source_sha256"],
            "row_sha256": canonical_sha256(runnability_row),
            "native_env_hash": runnability_contract["native_env_hash"],
            "status": "runnable",
            "returncode": 0,
            "expected_workloads": expected_workloads,
            "detail": detail,
        }
    else:
        if (
            runnability_row.get("status") != "host_capacity_oom"
            or runnability_row.get("returncode") != 1
            or reduced_provenance is None
            or shrink_row.get("reduced_run_detail")
            != f"{expected_workloads}/{expected_workloads} workloads passed"
        ):
            raise MatrixBuildError(f"SOL reduced/native evidence mismatch: {ident}")
        reduced_provenance.update({
            "native_runnability_source_path": SOL_RUNNABILITY_PATH.as_posix(),
            "native_runnability_source_sha256": runnability_contract["source_sha256"],
            "native_runnability_row_sha256": canonical_sha256(runnability_row),
            "native_run_status": "host_capacity_oom",
            "native_returncode": 1,
        })
        evaluation_evidence = {
            "source_kind": "reduced_shrink_evaluation",
            "source_path": SOL_SHRINK_PROVENANCE_PATH.as_posix(),
            "source_sha256": shrink_contract["source_sha256"],
            "row_sha256": canonical_sha256(shrink_row),
            "native_env_hash": shrink_contract["native_env_hash"],
            "status": "runnable",
            "returncode": 0,
            "expected_workloads": expected_workloads,
            "detail": shrink_row["reduced_run_detail"],
        }
    adapter = repo_root / SOL_EXECBENCH_PATH
    if not adapter.is_file() or not os.access(adapter, os.X_OK):
        raise MatrixBuildError(f"SOL adapter executable absent/not executable: {SOL_EXECBENCH_PATH}")
    adapter_sha = sha256_file(adapter)
    _validate_sol_forwarded_argv(argv, ident=ident)
    logical_workload = {
        "kind": "sol_execbench_workload_set",
        "value": shape,
        "workload_scope": workload_scope,
        "benchmark_definition": definition,
        "definition_file": _repo_rel(definition_path, repo_root),
        "definition_file_sha256": sha256_file(definition_path),
        "benchmark_dir": benchmark_rel.as_posix(),
        "benchmark_tree_sha256": tree_sha256(benchmark),
        "workload_file": _repo_rel(workload, repo_root),
        "workload_file_sha256": sha256_file(workload),
        "expected_workloads": expected_workloads,
        "evaluation_evidence": evaluation_evidence,
    }
    if reduced_provenance is not None:
        logical_workload["reduced_workload_provenance"] = reduced_provenance
    invocation = {
        "adapter_kind": "sol_execbench",
        "authority_reference_path": solution_rel,
        "authority_reference_sha256": expected_sha,
        "executable_path": SOL_EXECBENCH_PATH.as_posix(),
        "executable_sha256": adapter_sha,
        "label_app_args": argv,
        "trace_app_args": argv,
        "evaluation_cardinality_contract": "all_effective_workloads_passed_exact",
        "trace_cardinality_contract": "one_pinned_candidate_kernel_launch_exact",
        "capture_launch_contract": {
            "schema_version": 1,
            "label_driver_warmup_count": 1,
            "label_effective_warmup_count": 0,
            "label_launch_index": 0,
            "label_measured_reps": 1,
            "trace_launch_count": 1,
            "label_trace_launch_index": 0,
        },
        "capture_target": {
            "source_workload_index": 0,
            "staged_workload_index": 0,
            "candidate_user_fn_calls": 1,
            "candidate_kernel_ordinal": 0,
            "capture_launches": 1,
            "profiler_range": "cudaProfilerStartStop",
            "setup_outside_profiler_range": True,
            "timing_methodology": "cuda_events",
        },
        "logical_workload": logical_workload,
        "shape_delivery": "adapter_consumes_logical_shape",
        "target_processes": "all",
        "adapter_contract": {
            "contract_version": 3,
            "forwarded_argv": argv,
            "environment": {
                "CUTLASS_DIR": SOL_CUTLASS_PATH.as_posix(),
                "PATH_prepend": SOL_EXECBENCH_PATH.parent.as_posix(),
                "NVCC_CCBIN": SOL_PINNED_NVCC_CCBIN,
            },
            "native_env_hash": native_env_hash,
            "shape_forwarded_to_target": False,
            "label_driver_bridge_status": "implemented",
            "persistent_stage_status": "implemented",
            "capture_request_environment": "SOL_EXECBENCH_CAPTURE_REQUEST",
            "label_driver_bridge_rule": (
                "forward adapter_contract.forwarded_argv exactly without appending "
                "a synthetic --shape-point token"
            ),
        },
    }
    control = {
        "policy": "post_trace_module_or_typed",
        "module_discovery": "nvbit_runtime_module_inventory",
        "required_outcomes": list(SOL_CONTROL_OUTCOMES),
        "static_rebind": copy.deepcopy(SOL_STATIC_CONTROL_REBIND),
    }
    return invocation, control


def _cell_record(cell: dict, *, repo_root: Path, budget_by_id: dict,
                 budget_sha: str, sol_shrink_contract: dict,
                 sol_runnability_contract: dict,
                 sol_native_env_contract: dict) -> dict:
    ident = _identity(cell)
    status = cell["resolution_status"]
    record = {
        "identity": ident,
        "authority_cell_sha256": canonical_sha256(cell),
        "disposition": status,
    }
    if status == "excluded":
        reason = cell.get("exclusion_reason")
        detail = cell.get("exclusion_detail")
        if not isinstance(reason, str) or not reason or not isinstance(detail, str) or not detail:
            raise MatrixBuildError(f"excluded row lacks typed exclusion: {ident}")
        record.update({
            "capture_readiness": {"status": "not_applicable", "blocking_reasons": []},
            "exclusion": {"reason": reason, "detail": detail},
        })
        return record
    if status != "runnable":
        raise MatrixBuildError(f"non-final disposition reached matrix: {status!r} {ident}")
    if cell.get("binary_present") is not True:
        raise MatrixBuildError(f"runnable authority row is not live-resolved: {ident}")
    budget = _budget_record(
        ident["slug"], budget_by_id=budget_by_id, source_sha=budget_sha,
    )
    if ident["roster_source"] == "sol_execbench":
        invocation, control = _sol_invocation(
            cell, repo_root, shrink_contract=sol_shrink_contract,
            runnability_contract=sol_runnability_contract,
            native_env_contract=sol_native_env_contract,
        )
    else:
        invocation, control = _direct_invocation(cell, repo_root)
    record.update({
        "capture_readiness": _readiness(budget),
        "trace_budget": budget,
        "invocation": invocation,
        "control_bits": control,
    })
    return record


def _chunks(cells: list[dict], protocol_pins: dict) -> tuple[dict, list[dict]]:
    runnable = [c for c in cells if c["disposition"] == "runnable"]
    by_source = {
        source: sorted(
            (c for c in runnable if c["identity"]["roster_source"] == source),
            key=lambda c: c["identity"]["slug"],
        )
        for source in SOURCE_ORDER
    }
    smoke_path = ((protocol_pins.get("injection_runtime") or {}).get("smoke_binary_relpath"))
    bootstrap = None
    if isinstance(smoke_path, str) and smoke_path:
        for cell in by_source["portable_lcd"]:
            if cell["invocation"]["executable_path"] == smoke_path:
                bootstrap = cell
                break
    if bootstrap is None:
        raise MatrixBuildError("pinned injection smoke cell is not a runnable portable matrix row")
    by_source["portable_lcd"].remove(bootstrap)
    sol_bootstrap = next(
        (
            cell for cell in by_source["sol_execbench"]
            if cell["identity"]["slug"] == SOL_BOOTSTRAP_SLUG
        ),
        None,
    )
    if sol_bootstrap is None:
        raise MatrixBuildError("pinned SOL bridge bootstrap is not runnable")
    by_source["sol_execbench"].remove(sol_bootstrap)
    groups: list[tuple[str, list[dict], str]] = [
        ("portable_lcd", [bootstrap], "portable_bootstrap"),
    ]
    for source in SOURCE_ORDER:
        if source == "sol_execbench":
            groups.append((source, [sol_bootstrap], "sol_bootstrap"))
        rows = by_source[source]
        for start in range(0, len(rows), MAX_CHUNK_CELLS):
            groups.append((source, rows[start:start + MAX_CHUNK_CELLS], "regular"))
    chunks = []
    source_ordinals = {source: 0 for source in SOURCE_ORDER}
    for source, members, kind in groups:
        if not members:
            continue
        if kind == "portable_bootstrap":
            chunk_id = "portable_lcd-000-bootstrap"
        elif kind == "sol_bootstrap":
            chunk_id = "sol_execbench-000-bootstrap"
        else:
            source_ordinals[source] += 1
            chunk_id = f"{source}-{source_ordinals[source]:03d}"
        identities = [copy.deepcopy(c["identity"]) for c in members]
        chunks.append({
            "chunk_id": chunk_id,
            "roster_source": source,
            "ordinal": len(chunks),
            "membership_sha256": canonical_sha256(identities),
            "cells": identities,
        })
    flattened = [ident for chunk in chunks for ident in chunk["cells"]]
    return ({
        "source_order": list(SOURCE_ORDER),
        "ordering": (
            "pinned_portable_bootstrap_then_source_order_with_sol_bootstrap_"
            "before_regular_sol_slug_lexicographic"
        ),
        "max_cells_per_regular_chunk": MAX_CHUNK_CELLS,
        "bootstrap_identity": copy.deepcopy(bootstrap["identity"]),
        "sol_bootstrap_identity": copy.deepcopy(sol_bootstrap["identity"]),
        "runnable_membership_sha256": canonical_sha256(flattened),
    }, chunks)


def build_matrix(repo_root: Path = REPO_ROOT, *,
                 require_head_match: bool = True,
                 authority_path: Path | None = None) -> dict:
    repo_root = repo_root.resolve()
    authority_path = (authority_path or (repo_root / AUTHORITY_PATH)).resolve()
    protocol_path = repo_root / PROTOCOL_PINS_PATH
    retention_path = repo_root / RETENTION_BUDGET_PATH
    budgets_path = repo_root / TRACE_BUDGETS_PATH
    registry_path = repo_root / METRIC_REGISTRY_PATH
    shrink_path = repo_root / SOL_SHRINK_PROVENANCE_PATH
    runnability_path = repo_root / SOL_RUNNABILITY_PATH
    native_env_path = repo_root / SOL_NATIVE_ENV_FINGERPRINT_PATH

    authority = _load_json(authority_path)
    _authority_invariants(
        authority, repo_root=repo_root, require_head_match=require_head_match,
    )
    protocol = _load_json(protocol_path)
    retention = _load_json(retention_path)
    registry = _load_json(registry_path)
    sol_shrink_contract = _sol_shrink_contract(repo_root)
    sol_runnability_contract = _sol_runnability_contract(repo_root)
    sol_native_env_contract = _sol_native_env_contract(repo_root)
    certified_native_env_hash = sol_native_env_contract["native_env_hash"]
    if sol_shrink_contract["native_env_hash"] != certified_native_env_hash:
        raise MatrixBuildError(
            "SOL shrink provenance/native fingerprint hash mismatch"
        )
    if sol_runnability_contract["native_env_hash"] != certified_native_env_hash:
        raise MatrixBuildError(
            "SOL runnability/native fingerprint hash mismatch"
        )
    if protocol.get("arch") != ARCH or protocol.get("sku") != SKU:
        raise MatrixBuildError("protocol pins arch/sku mismatch")
    _verify_static_control_toolchain(SOL_STATIC_CONTROL_REBIND)
    if retention.get("arch") != ARCH or retention.get("sku") != SKU:
        raise MatrixBuildError("aggregate retention budget arch/sku mismatch")
    retention_gib = retention.get("retention_budget_gib")
    if not isinstance(retention_gib, int) or isinstance(retention_gib, bool) or retention_gib <= 0:
        raise MatrixBuildError("aggregate retention budget is absent/non-positive")
    metadata_margin_gib = retention.get("metadata_margin_gib")
    raw_staging_margin_gib = retention.get("raw_staging_margin_gib")
    for field, value in (
        ("metadata_margin_gib", metadata_margin_gib),
        ("raw_staging_margin_gib", raw_staging_margin_gib),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MatrixBuildError(
                f"aggregate retention budget {field} is absent/non-positive"
            )
    budget_config = load_trace_budgets(budgets_path)
    budget_by_id = budget_config.cells_by_id()
    budget_sha = sha256_file(budgets_path)

    cells = [
        _cell_record(
            cell, repo_root=repo_root, budget_by_id=budget_by_id,
            budget_sha=budget_sha, sol_shrink_contract=sol_shrink_contract,
            sol_runnability_contract=sol_runnability_contract,
            sol_native_env_contract=sol_native_env_contract,
        )
        for cell in sorted(
            authority["cells"],
            key=lambda c: (SOURCE_ORDER.index(c["roster_source"]), c["slug"]),
        )
    ]
    identities = [
        (c["identity"]["roster_source"], c["identity"]["slug"]) for c in cells
    ]
    if len(identities) != len(set(identities)):
        raise MatrixBuildError("matrix identity collision")
    observed_overrides = {
        cell["identity"]["slug"] for cell in cells
        if cell.get("disposition") == "runnable"
        and cell["identity"]["roster_source"] == "simbench_bw"
        and "matrix_capture_override" in (cell.get("invocation") or {})
    }
    if observed_overrides != set(SIMBENCH_CAPTURE_OVERRIDES):
        raise MatrixBuildError(
            "SimBench capture override set drift: "
            f"expected={sorted(SIMBENCH_CAPTURE_OVERRIDES)!r} "
            f"actual={sorted(observed_overrides)!r}"
        )
    sol_runnable = [
        cell for cell in cells
        if cell["disposition"] == "runnable"
        and cell["identity"]["roster_source"] == "sol_execbench"
    ]
    native_sol = [
        cell for cell in sol_runnable
        if cell["invocation"]["logical_workload"]["workload_scope"] == "native"
    ]
    reduced_sol = [
        cell for cell in sol_runnable
        if cell["invocation"]["logical_workload"]["workload_scope"]
        == SOL_REDUCED_SHAPE
    ]
    if len(native_sol) != 73 or len(reduced_sol) != 1:
        raise MatrixBuildError(
            "SOL logical workload partition must be exactly 73 native and 1 reduced"
        )
    if reduced_sol[0]["identity"]["slug"] != "problem_028_gqa_rotary_attention_core_backward":
        raise MatrixBuildError("unexpected SOL reduced representative identity")
    chunking, chunks = _chunks(cells, protocol)
    runnable = [c for c in cells if c["disposition"] == "runnable"]
    excluded = [c for c in cells if c["disposition"] == "excluded"]
    declared = [c for c in runnable if c["trace_budget"]["status"] == "declared"]
    ready = [c for c in runnable if c["capture_readiness"]["status"] == "ready"]
    gib = 1024 ** 3
    declared_runnable_max_trace_bytes = sum(
        c["trace_budget"]["max_trace_bytes"] for c in declared
    )
    retention_ceiling_bytes = retention_gib * gib
    metadata_margin_bytes = metadata_margin_gib * gib
    raw_staging_margin_bytes = raw_staging_margin_gib * gib
    retained_headroom_bytes = (
        retention_ceiling_bytes
        - declared_runnable_max_trace_bytes
        - metadata_margin_bytes
    )
    if retained_headroom_bytes < 0:
        raise MatrixBuildError(
            "declared runnable trace ceilings plus metadata margin exceed the "
            f"aggregate retention ceiling: traces={declared_runnable_max_trace_bytes} "
            f"metadata={metadata_margin_bytes} ceiling={retention_ceiling_bytes}"
        )
    by_source = {}
    for source in SOURCE_ORDER:
        rows = [c for c in cells if c["identity"]["roster_source"] == source]
        by_source[source] = {
            "universe": len(rows),
            "runnable": sum(c["disposition"] == "runnable" for c in rows),
            "excluded": sum(c["disposition"] == "excluded" for c in rows),
            "capture_ready": sum(
                c["disposition"] == "runnable"
                and c["capture_readiness"]["status"] == "ready" for c in rows
            ),
        }

    matrix = {
        "schema_version": 1,
        "matrix_id": MATRIX_ID,
        "arch": ARCH,
        "sku": SKU,
        "source_commit": authority["source_commit"],
        "authority": {
            "path": _portable_path_ref(authority_path, repo_root),
            "sha256": sha256_file(authority_path),
            "roster_id": authority["roster_id"],
        },
        "input_pins": {
            "protocol_pins": {
                "path": PROTOCOL_PINS_PATH.as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "aggregate_retention_budget": {
                "path": RETENTION_BUDGET_PATH.as_posix(),
                "sha256": sha256_file(retention_path),
                "retention_budget_gib": retention_gib,
                "metadata_margin_gib": metadata_margin_gib,
                "raw_staging_margin_gib": raw_staging_margin_gib,
                "declared_runnable_max_trace_bytes": (
                    declared_runnable_max_trace_bytes
                ),
                "retention_ceiling_bytes": retention_ceiling_bytes,
                "metadata_margin_bytes": metadata_margin_bytes,
                "retained_headroom_bytes": retained_headroom_bytes,
                "raw_staging_margin_bytes": raw_staging_margin_bytes,
                "preflight_required_free_bytes": (
                    retention_ceiling_bytes + raw_staging_margin_bytes
                ),
            },
            "per_cell_trace_budgets": {
                "path": TRACE_BUDGETS_PATH.as_posix(),
                "sha256": budget_sha,
                "defaults_are_enforcement": False,
            },
            "metric_registry": {
                "path": METRIC_REGISTRY_PATH.as_posix(),
                "sha256": sha256_file(registry_path),
            },
            "sol_shrink_provenance": {
                "path": SOL_SHRINK_PROVENANCE_PATH.as_posix(),
                "sha256": sha256_file(shrink_path),
            },
            "sol_runnability": {
                "path": SOL_RUNNABILITY_PATH.as_posix(),
                "sha256": sha256_file(runnability_path),
            },
            "sol_native_env_fingerprint": {
                "path": SOL_NATIVE_ENV_FINGERPRINT_PATH.as_posix(),
                "sha256": sha256_file(native_env_path),
                "native_env_hash": certified_native_env_hash,
            },
        },
        "metric_registry": _metric_contract(registry),
        "counts": {
            "universe": len(cells),
            "runnable": len(runnable),
            "excluded": len(excluded),
            "capture_ready": len(ready),
            "capture_blocked": len(runnable) - len(ready),
            "budget_declared": len(declared),
            "budget_missing": len(runnable) - len(declared),
            "chunks": len(chunks),
            "by_source": by_source,
        },
        "chunking": chunking,
        "chunks": chunks,
        "cells": cells,
    }
    matrix["matrix_seal_sha256"] = canonical_sha256(matrix)
    return matrix


def _validate_frozen_registry_contract(
    matrix: dict,
    *,
    registry_document: dict | None,
    registry_payload: bytes | None,
    registry_sha256: str | None,
) -> list[str]:
    """Validate an explicitly supplied frozen registry projection.

    The all-``None`` form is intentional: callers that only have the frozen
    matrix must still be able to validate it without rebuilding against live
    host inputs.  Callers that have the source registry must supply all three
    operands so the projection is checked against the exact payload they
    already pinned.
    """
    supplied = (registry_document, registry_payload, registry_sha256)
    if all(operand is None for operand in supplied):
        return []
    if any(operand is None for operand in supplied):
        return ["registry_operands_incomplete"]
    if not isinstance(registry_document, dict):
        return ["registry_document_invalid"]
    if not isinstance(registry_payload, bytes):
        return ["registry_payload_invalid"]
    if (
        not isinstance(registry_sha256, str)
        or len(registry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in registry_sha256)
    ):
        return ["registry_sha256_invalid"]
    observed_digest = hashlib.sha256(registry_payload).hexdigest()
    if observed_digest != registry_sha256:
        return ["registry_payload_sha256"]
    try:
        parsed = json.loads(registry_payload)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return ["registry_payload_json"]
    if not _strict_json_equal(parsed, registry_document):
        return ["registry_document_payload_mismatch"]
    input_pins = matrix.get("input_pins")
    if not isinstance(input_pins, dict):
        return ["registry_pin_missing"]
    pin = input_pins.get("metric_registry")
    if not isinstance(pin, dict):
        return ["registry_pin_missing"]
    if pin.get("path") != METRIC_REGISTRY_PATH.as_posix():
        return ["registry_pin_path"]
    if pin.get("sha256") != registry_sha256:
        return ["registry_pin_sha256"]
    try:
        projection = _metric_contract(registry_document)
    except Exception:  # noqa: BLE001 -- return a typed validation result
        return ["registry_projection_invalid"]
    if not _strict_json_equal(matrix.get("metric_registry"), projection):
        return ["metric_registry_projection"]
    return []


def _validate_frozen_v1_authority(matrix: dict, repo_root: Path) -> list[str]:
    """Validate the checked-in v1 matrix without ambient matrix generation."""
    authority_path = repo_root / FROZEN_V1_MATRIX_PATH
    if authority_path.is_symlink() or not authority_path.is_file():
        return ["frozen_v1_authority_file"]
    try:
        if sha256_file(authority_path) != FROZEN_V1_MATRIX_SHA256:
            return ["frozen_v1_authority_sha256"]
        expected = _load_json(authority_path)
    except Exception:  # noqa: BLE001 -- return a typed validation result
        return ["frozen_v1_authority_json"]
    if not _strict_json_equal(matrix, expected):
        return ["matrix_content_stale_or_tampered"]
    if matrix.get("matrix_seal_sha256") != FROZEN_V1_MATRIX_SEAL_SHA256:
        return ["frozen_v1_matrix_seal"]
    if matrix.get("source_commit") != FROZEN_V1_SOURCE_COMMIT:
        return ["frozen_v1_source_commit"]
    input_pins = matrix.get("input_pins")
    if not isinstance(input_pins, dict):
        return ["frozen_v1_registry_pin"]
    registry_pin = input_pins.get("metric_registry")
    if not isinstance(registry_pin, dict):
        return ["frozen_v1_registry_pin"]
    if registry_pin.get("path") != METRIC_REGISTRY_PATH.as_posix():
        return ["frozen_v1_registry_path"]
    if registry_pin.get("sha256") != FROZEN_V1_REGISTRY_SHA256:
        return ["frozen_v1_registry_sha256"]
    metric_registry = matrix.get("metric_registry")
    if not isinstance(metric_registry, dict):
        return ["frozen_v1_metric_identity"]
    if metric_registry.get("metric_identity_sha256") != FROZEN_V1_METRIC_IDENTITY_SHA256:
        return ["frozen_v1_metric_identity"]
    return []


def validate_matrix(
    matrix: dict,
    repo_root: Path = REPO_ROOT,
    *,
    validation_mode: str = "live",
    registry_document: dict | None = None,
    registry_payload: bytes | None = None,
    registry_sha256: str | None = None,
) -> list[str]:
    """Validate matrix semantics using either live or explicit frozen inputs."""
    problems = []
    seal = matrix.get("matrix_seal_sha256")
    unsealed = {k: v for k, v in matrix.items() if k != "matrix_seal_sha256"}
    if seal != canonical_sha256(unsealed):
        problems.append("matrix_seal_sha256")
    if not isinstance(validation_mode, str) or validation_mode not in VALIDATION_MODES:
        return list(dict.fromkeys(problems + ["validation_mode_unknown"]))
    registry_operands = (
        registry_document,
        registry_payload,
        registry_sha256,
    )
    if validation_mode == "live":
        if any(operand is not None for operand in registry_operands):
            return list(dict.fromkeys(problems + ["registry_operands_live_mode"]))
        authority_ref = (matrix.get("authority") or {}).get("path")
        authority_path = Path(authority_ref) if isinstance(authority_ref, str) else None
        if authority_path is not None and not authority_path.is_absolute():
            authority_path = Path(repo_root) / authority_path
        try:
            expected = build_matrix(
                repo_root, require_head_match=False, authority_path=authority_path,
            )
        except Exception as exc:  # noqa: BLE001 -- return a typed validation result
            return problems + [f"rederive_failed:{type(exc).__name__}:{exc}"]
        if matrix != expected:
            problems.append("matrix_content_stale_or_tampered")
    else:
        problems.extend(_validate_frozen_v1_authority(matrix, repo_root))
        problems.extend(
            _validate_frozen_registry_contract(
                matrix,
                registry_document=registry_document,
                registry_payload=registry_payload,
                registry_sha256=registry_sha256,
            )
        )

    cells = matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        problems.append("cells_nonempty")
        return list(dict.fromkeys(problems))
    identities = [
        (c.get("identity", {}).get("roster_source"), c.get("identity", {}).get("slug"))
        for c in cells if isinstance(c, dict)
    ]
    if len(identities) != len(cells) or len(set(identities)) != len(identities):
        problems.append("cell_identity_uniqueness")
    runnable = {
        ident for ident, cell in zip(identities, cells)
        if isinstance(cell, dict) and cell.get("disposition") == "runnable"
    }
    excluded = {
        ident for ident, cell in zip(identities, cells)
        if isinstance(cell, dict) and cell.get("disposition") == "excluded"
    }
    chunks = matrix.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        problems.append("chunks_nonempty")
        return list(dict.fromkeys(problems))
    flattened = []
    chunk_ids = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            problems.append("chunk_object")
            continue
        chunk_ids.append(chunk.get("chunk_id"))
        members = chunk.get("cells")
        if not isinstance(members, list) or not members:
            problems.append("chunk_members_nonempty")
            continue
        if len(members) > MAX_CHUNK_CELLS:
            problems.append("chunk_size_bound")
        if chunk.get("membership_sha256") != canonical_sha256(members):
            problems.append("chunk_membership_sha256")
        flattened.extend(
            (m.get("roster_source"), m.get("slug")) for m in members
            if isinstance(m, dict)
        )
    if len(chunk_ids) != len(set(chunk_ids)):
        problems.append("chunk_id_uniqueness")
    if len(flattened) != len(set(flattened)):
        problems.append("chunk_membership_uniqueness")
    if set(flattened) != runnable:
        problems.append("chunk_runnable_set_equality")
    if set(flattened) & excluded:
        problems.append("excluded_in_chunk")
    if matrix.get("chunking", {}).get("runnable_membership_sha256") != canonical_sha256(
        [m for chunk in chunks for m in (chunk.get("cells") or [])]
    ):
        problems.append("runnable_membership_sha256")
    return list(dict.fromkeys(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out")
    mode.add_argument("--check")
    parser.add_argument(
        "--authority", default=str(AUTHORITY_PATH),
        help="authority input; may be external for a clean live-capture session",
    )
    args = parser.parse_args(argv)
    authority_path = Path(args.authority)
    if not authority_path.is_absolute():
        authority_path = REPO_ROOT / authority_path
    try:
        # Emission is only allowed from the authority's stamped source commit.
        # Validation after the generated artifact is committed compares exact
        # pinned inputs instead; requiring HEAD equality there would make every
        # generated artifact become uncheckable merely because its own commit
        # advanced HEAD.
        matrix = build_matrix(
            require_head_match=bool(args.out), authority_path=authority_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"build_sm86_capture_matrix: build failed: {exc}", file=sys.stderr)
        return 2
    target_arg = args.out or args.check
    target = Path(target_arg)
    if not target.is_absolute():
        target = REPO_ROOT / target
    if args.check:
        try:
            existing = _load_json(target)
        except MatrixBuildError as exc:
            print(f"build_sm86_capture_matrix: check failed: {exc}", file=sys.stderr)
            return 2
        problems = validate_matrix(existing)
        if existing != matrix and "matrix_content_stale_or_tampered" not in problems:
            problems.append("matrix_content_stale_or_tampered")
        if problems:
            print("build_sm86_capture_matrix: INVALID " + ", ".join(problems), file=sys.stderr)
            return 1
        print(f"capture matrix: VALID {target}")
        return 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"build_sm86_capture_matrix: write failed: {exc}", file=sys.stderr)
        return 2
    c = matrix["counts"]
    print(
        f"capture matrix: universe={c['universe']} runnable={c['runnable']} "
        f"excluded={c['excluded']} ready={c['capture_ready']} "
        f"blocked={c['capture_blocked']} chunks={c['chunks']} -> {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
