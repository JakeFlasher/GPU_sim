"""SimBench-BW MMA microbench reference family.

HMMA / QMMA / IMMA / sparse tensor-core microbenches. Reference is an FP32
matmul of the same MxNxK shape for GEMM-like rows, or a seeded tensor echo
for dep-chain / saturation / memory microbenches.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor
from _low_precision_quant import (
    quantize_fp4_e2m1_to_fp32_representable,
    quantize_fp6_e2m3_to_fp32_representable,
    quantize_fp6_e3m2_to_fp32_representable,
    quantize_fp8_e4m3_to_fp32_representable,
    quantize_fp8_e5m2_to_fp32_representable,
)


FAMILY_NAME = "mma_microbench"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "fp32_fma_dep_chain",
    "bf16_hmma_gemm_traceable",
    "tf32_hmma_gemm_traceable",
    "fp16_hmma_gemm_traceable",
    "wgmma_bf16_saturation",
    "wgmma_fp8_saturation",
    "fp8_e4m3_qmma_dep_chain",
    "fp8_e5m2_saturation",
    "fp6_e3m2_mma_dep_chain",
    "fp6_e2m3_saturation",
    "fp4_e2m1_mma_dep_chain",
    "nvfp4_blockscale_gemm",
    "int8_imma_gemm_traceable",
    "sparse_2to4_bf16_gemm",
    "sparse_2to4_int8_gemm",
    "sparse_2to4_fp8_gemm",
    "shared_bank_conflict_stride",
    "l2_thrash_pointer_chase",
    "dram_stream_stride_mix",
    "dram_streaming_dot_product",
    "global_atomic_histogram",
    "warp_divergence_reconvergence",
    "register_spill_local_mem",
    # R44 / R46 / R47 / R48 microbench-suite-coverage criterion slugs.
    # Each row's host driver emits an identity round-trip that the
    # branch in `build_reference` reproduces exactly; the
    # generic axes.values() fallback would not match the host's
    # printed JSON output shape.
    "working_set_capacity_sweep",
    "warm_cold_pair_relaunch",
    "l2_prefetch_hint_sweep",
    "l2_persisting_window_sweep",
    "carveout_sensitivity_probe",
    # Identity round-trip cells whose host drivers print a row-major
    # [samples][bins] copy of the seeded input (out = in). Their seed_tensor
    # LCG is byte-identical to deterministic_seeded_tensor, so the (samples,
    # bins) reference matches the printed output exactly. The atomics ATOM
    # partners share the same identity-round-trip `out`; the vector-width family
    # members (32b / 64b / 128b) are global copies.
    "global_atomic_return_used",
    "global_atomic_max_return_used",
    "global_atomic_max_histogram",
    "global_atomic_and_return_used",
    "global_atomic_and_histogram",
    "global_vectorized_128b_copy",
    "global_vectorized_64b_copy",
    "global_scalar_32b_copy",
    # Local-memory family members. Their host drivers print the same
    # row-major [samples][bins] identity copy (out = in); the local-memory
    # traffic (forced by dynamic indexing / volatile semantics) folds into a
    # scratch sink that is NOT part of `out`.
    "local_dynamic_index_array",
    "local_volatile_array",
)


def _fp32_matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            aip = a[i][p]
            for j in range(n):
                out[i][j] += aip * b[p][j]
    return out


def _quantize_matrix_fp4_e2m1(matrix: list[list[float]]) -> list[list[float]]:
    """Apply FP4 E2M1 quantization element-wise to a 2-D matrix. Each
    output element is the input rounded to the nearest FP4 E2M1
    representable value (returned as FP32). Matches the host driver's
    `_quantize_fp4_e2m1_nibble` + `_dequantize_fp4_e2m1_nibble` round-
    trip in `l1/fp4_e2m1_mma_dep_chain/main.cu`. R3 addition.
    """
    return [quantize_fp4_e2m1_to_fp32_representable(row) for row in matrix]


def _quantize_matrix_fp6_e2m3(matrix: list[list[float]]) -> list[list[float]]:
    """Apply FP6 E2M3 quantization element-wise to a 2-D matrix. Each
    output element is the input rounded to the nearest FP6 E2M3
    representable value (returned as FP32). Matches the host driver's
    `_quantize_fp6_e2m3_code` + `_dequantize_fp6_e2m3_code` round-trip
    in `l1/fp6_e2m3_saturation/main.cu`. R5 addition.
    """
    return [quantize_fp6_e2m3_to_fp32_representable(row) for row in matrix]


def _quantize_matrix_fp6_e3m2(matrix: list[list[float]]) -> list[list[float]]:
    """Apply FP6 E3M2 quantization element-wise to a 2-D matrix. Each
    output element is the input rounded to the nearest FP6 E3M2
    representable value (returned as FP32). Matches the host driver's
    `_quantize_fp6_e3m2_code` + `_dequantize_fp6_e3m2_code` round-trip
    in `l1/fp6_e3m2_mma_dep_chain/main.cu`. R6 addition.
    """
    return [quantize_fp6_e3m2_to_fp32_representable(row) for row in matrix]


def _quantize_matrix_fp8_e4m3(matrix: list[list[float]]) -> list[list[float]]:
    """Apply FP8 E4M3 quantization element-wise to a 2-D matrix. Each
    output element is the input rounded to the nearest FP8 E4M3
    representable value (returned as FP32). Matches the host driver's
    `_quantize_fp8_e4m3_byte` + `_dequantize_fp8_e4m3_byte` round-trip
    in `l1/sparse_2to4_fp8_gemm/main.cu`. R7 addition.
    """
    return [quantize_fp8_e4m3_to_fp32_representable(row) for row in matrix]


def _quantize_matrix_fp8_e5m2(matrix: list[list[float]]) -> list[list[float]]:
    """Apply FP8 E5M2 quantization element-wise to a 2-D matrix. Each
    output element is the input rounded to the nearest FP8 E5M2
    representable value (returned as FP32). Matches the host driver's
    `__nv_fp8_e5m2` cast + dequantize round-trip in
    `l1/fp8_e5m2_saturation/main.cu`. R20 addition (per Codex
    Round-19-review blocking finding #2).
    """
    return [quantize_fp8_e5m2_to_fp32_representable(row) for row in matrix]


def _round_half_away_from_zero(x: float) -> int:
    """Round to nearest integer, breaking ties away from zero.

    Matches C99 `lroundf` semantics — the rounding mode the host driver
    uses at `l1/sparse_2to4_int8_gemm/main.cu:28-34` (`lroundf(q)`).
    Python's built-in `round()` uses banker's rounding (round-half-to-
    even), which disagrees with `lroundf` at every half-tie:
    `round(0.5)==0` vs `lroundf(0.5)==1`, `round(-0.5)==0` vs
    `lroundf(-0.5)==-1`. The R14 fix (per Codex Round-13-review
    mainline-gap #1) replaces `round()` in `_quantize_matrix_int8` with
    this helper so the Python reference matches the host bit-exactly
    at INT8 boundaries.

    `math.floor(x + 0.5)` for x ≥ 0 and `math.ceil(x - 0.5)` for x < 0
    is the standard portable formulation.
    """
    if x >= 0.0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def _quantize_matrix_int8(matrix: list[list[float]], scale: float = 64.0) -> list[list[float]]:
    """Apply INT8 round-trip quantization element-wise: scale → round
    half-away-from-zero (C99 `lroundf` semantics) → clamp to [-128, 127]
    → divide by scale. Matches the host driver's `quantize_int8(v,
    SCALE=64)` + the kernel's `(float)d * dequant_scale` pattern in
    `l1/sparse_2to4_int8_gemm/main.cu` bit-exactly (R14 fix; Round
    13 used Python `round()` which disagreed with `lroundf` at half-
    tie boundaries). The reference matmul of these quantized values
    models what the IMMA.SP.16864.S8.S8 kernel actually computes
    (modulo the INT32 accumulation precision, which is exact for
    K ≤ ~16M).
    """
    out = []
    for row in matrix:
        new_row = []
        for v in row:
            q = _round_half_away_from_zero(v * scale)
            if q > 127:
                q = 127
            elif q < -128:
                q = -128
            new_row.append(q / scale)
        out.append(new_row)
    return out


def _apply_2to4_mask(a: list[list[float]], m: int, k: int) -> list[list[float]]:
    """Deterministic 2:4 sparsity mask — keep K-indices {0, 1} of every
    4-group, zero K-indices {2, 3}.

    This is the simplest deterministic 2:4 pattern (no dependence on
    input values), and it's the pattern the host driver / kernel
    for `sparse_2to4_bf16_gemm` pack their compressed A buffer
    against. Requires `k % 4 == 0` so the final group is full.
    """
    if k % 4 != 0:
        raise ValueError(f"sparse_2to4 requires k % 4 == 0 (got k={k})")
    out = [row[:] for row in a]
    for i in range(m):
        for ks in range(0, k, 4):
            out[i][ks + 2] = 0.0
            out[i][ks + 3] = 0.0
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    # R47 microbench-suite-coverage criterion slugs land explicit
    # shape branches so the reference matches the host driver's
    # printed JSON output shape exactly. The generic axes.values()
    # fallback at the bottom of this function builds an N-D tensor
    # from the axes ordering, which does not match these slugs'
    # actual host output shapes.
    if slug == "working_set_capacity_sweep":
        # Host driver prints (warps, 32) 2D: each warp dumps its
        # 32-lane row. The kernel writes `out[i] = in[i %
        # working_set_floats]` for i in 0..warps*32; the reference
        # is the (warps, 32)-shape seeded tensor.
        warps = int(axes["warps"])
        return deterministic_seeded_tensor((warps, 32), seed, low=-1.0, high=1.0)
    if slug == "warm_cold_pair_relaunch":
        # Host driver prints `min(allocation_floats, 64)` 1D from
        # the warm-launch output; the kernel writes `out[i] = in[i]`
        # so the reference is the leading slice of the seeded tensor.
        allocation_bytes = int(axes["allocation_bytes"])
        allocation_floats = max(allocation_bytes // 4, 32)
        report = min(allocation_floats, 64)
        return deterministic_seeded_tensor((report,), seed, low=-1.0, high=1.0)
    if slug == "l2_prefetch_hint_sweep":
        # Host driver prints (accesses, warps) 2D; the kernel
        # writes `out[i] = in[i]` for i in 0..accesses*warps; the
        # reference is the (accesses, warps)-shape seeded tensor.
        accesses = int(axes["accesses"])
        warps = int(axes["warps"])
        return deterministic_seeded_tensor(
            (accesses, warps), seed, low=-1.0, high=1.0,
        )
    if slug == "l2_persisting_window_sweep":
        # Host driver prints (warps, 32) 2D from the persistence-
        # window kernel. Same identity round-trip pattern as
        # working_set_capacity_sweep.
        warps = int(axes["warps"])
        return deterministic_seeded_tensor((warps, 32), seed, low=-1.0, high=1.0)
    if slug == "carveout_sensitivity_probe":
        # Host driver prints (warps, 32) 2D from the carveout-
        # sensitivity kernel. Same identity round-trip pattern as
        # working_set_capacity_sweep / l2_persisting_window_sweep.
        warps = int(axes["warps"])
        return deterministic_seeded_tensor((warps, 32), seed, low=-1.0, high=1.0)
    if slug in (
        "global_atomic_return_used",
        "global_atomic_max_return_used",
        "global_atomic_max_histogram",
        "global_atomic_and_return_used",
        "global_atomic_and_histogram",
        "global_vectorized_128b_copy",
        "global_vectorized_64b_copy",
        "global_scalar_32b_copy",
        "local_dynamic_index_array",
        "local_volatile_array",
    ):
        # Host drivers print a row-major (samples, bins) 2D identity copy of
        # the seeded input (out[s*bins+b] = in[s*bins+b]) over the
        # samples*bins-float buffer. The drivers seed with the SAME LCG as
        # deterministic_seeded_tensor (main.cu::seed_tensor: state=seed*
        # 2654435761+1, then state=state*1103515245+12345, value=lo+(hi-lo)*
        # u32/2**32), so the (samples, bins) seeded tensor reproduces the
        # printed output exactly. (global_atomic_return_used's atomic traffic
        # escapes to a scratch `histo` buffer that is NOT part of `out`.)
        samples = int(axes["samples"])
        bins = int(axes["bins"])
        return deterministic_seeded_tensor(
            (samples, bins), seed, low=-1.0, high=1.0,
        )
    if slug == "dram_streaming_dot_product":
        # Host driver emits a SCALAR dot product (under the `output`
        # field): total = passes * sum_i a[i]*b[i] over two `n`-float
        # buffers a (seed) and b (seed+1) in [-1, 1]. The driver seeds
        # those buffers with the SAME LCG as deterministic_seeded_tensor
        # (main.cu::seed_tensor: state=seed*2654435761+1, then
        # state=state*1103515245+12345, value=lo+(hi-lo)*u32/2**32), and
        # the kernel re-traverses both buffers `passes` times accumulating
        # a[i]*b[i] (dram_streaming_dot_product.cu). So the reference is
        # that analytic scalar dot product — NOT the generic seeded tensor
        # the axes fallback would return. The (FP32, fma_chain,
        # mma_microbench) tolerance triplet absorbs the GPU block-reduction
        # accumulation-order divergence from this sequential sum.
        n = int(axes["n"])
        passes = int(axes["passes"])
        a = deterministic_seeded_tensor((n,), seed, low=-1.0, high=1.0)
        b = deterministic_seeded_tensor((n,), seed + 1, low=-1.0, high=1.0)
        return float(passes) * float(sum(a[i] * b[i] for i in range(n)))
    if "m" in axes and "n" in axes and "k" in axes:
        m, n, k = int(axes["m"]), int(axes["n"]), int(axes["k"])
        a = deterministic_seeded_tensor((m, k), seed, low=-0.5, high=0.5)
        b = deterministic_seeded_tensor((k, n), seed + 1, low=-0.5, high=0.5)
        if slug == "sparse_2to4_bf16_gemm":
            a = _apply_2to4_mask(a, m, k)
        elif slug == "fp4_e2m1_mma_dep_chain":
            # R3: this row's host driver pre-quantizes A and B to
            # FP4 E2M1 packed bytes (kind::f8f6f4 layout). The reference
            # must apply the same quantization so the FP32 matmul models
            # what the on-device kernel actually computes (modulo the
            # FP4*FP4→FP32 hardware multiply rounding, which the
            # (FP4_E2M1, fma_chain, mma_microbench) tolerance triplet
            # absorbs at rtol=atol=0.15).
            a = _quantize_matrix_fp4_e2m1(a)
            b = _quantize_matrix_fp4_e2m1(b)
        elif slug == "fp6_e2m3_saturation":
            # R5: this row's host driver pre-quantizes A and B to
            # FP6 E2M3 packed bytes (kind::f8f6f4 .e2m3 layout, low 6
            # bits [5:0]). The reference applies the same quantization
            # so the FP32 matmul models what the on-device kernel
            # actually computes (modulo the FP6*FP6→FP32 hardware
            # multiply rounding, which the (FP6_E2M3, fma_chain,
            # mma_microbench) tolerance triplet absorbs at
            # rtol=atol=0.1, max_ulp=4).
            a = _quantize_matrix_fp6_e2m3(a)
            b = _quantize_matrix_fp6_e2m3(b)
        elif slug == "fp6_e3m2_mma_dep_chain":
            # R6: same byte layout as e2m3 (low 6 bits [5:0]),
            # different value table (bias=3, max=28.0). Tolerance
            # triplet (FP6_E3M2, fma_chain, mma_microbench) at
            # rtol=atol=0.1, max_ulp=4.
            a = _quantize_matrix_fp6_e3m2(a)
            b = _quantize_matrix_fp6_e3m2(b)
        elif slug == "sparse_2to4_fp8_gemm":
            # R7: this row's host driver applies the 2:4 keep mask
            # to A (kept K-indices {4g+0, 4g+1}) and quantizes both
            # masked A and dense B to FP8 E4M3 packed bytes (kind::f8f6f4
            # .e4m3 layout, all 8 bits [7:0] used; no padding mask).
            # The reference applies the same mask + quantization. The
            # (FP8_E4M3, fma_chain, mma_microbench) tolerance triplet at
            # rtol=atol=0.05, max_ulp=8 absorbs the FP8*FP8→FP32
            # hardware-multiply rounding.
            a = _apply_2to4_mask(a, m, k)
            a = _quantize_matrix_fp8_e4m3(a)
            b = _quantize_matrix_fp8_e4m3(b)
        elif slug == "sparse_2to4_int8_gemm":
            # R13: this row's host driver applies the 2:4 keep mask
            # to A (kept K-indices {4g+0, 4g+1}; host main.cu lines
            # 67-75) and INT8-quantizes both masked A and dense B at
            # SCALE=64 (host main.cu line 53). The kernel reads
            # IMMA.SP.16864.S8.S8 + scales the INT32 accumulator by
            # `1/(SCALE*SCALE) = 1/4096`. The reference applies the
            # same mask + INT8 round-trip quantization so the FP32
            # matmul models what the kernel actually computes.
            a = _apply_2to4_mask(a, m, k)
            a = _quantize_matrix_int8(a, scale=64.0)
            b = _quantize_matrix_int8(b, scale=64.0)
        elif slug == "fp8_e4m3_qmma_dep_chain":
            # R19: this row's host driver pre-quantizes A and B to
            # FP8 E4M3 via `__nv_fp8_e4m3` cast (host main.cu lines
            # 49-54) before H2D. The kernel reads QMMA.16832.F32.E4M3.E4M3
            # + accumulates in FP32. Without this elif the reference
            # would compute FP32 matmul on un-quantized FP32 operands
            # while the kernel sees FP8-rounded operands — same R13
            # sparse_int8 defect class. The (FP8_E4M3, fma_chain,
            # mma_microbench) tolerance triplet at rtol=atol=0.05,
            # max_ulp=8 absorbs the FP8*FP8→FP32 hardware-multiply
            # rounding only after the reference applies the same
            # FP8 round-trip the host applied.
            a = _quantize_matrix_fp8_e4m3(a)
            b = _quantize_matrix_fp8_e4m3(b)
        elif slug == "wgmma_fp8_saturation":
            # R19: this Hopper sm_90a row's host driver pre-quantizes
            # A and B to FP8 E4M3 via `__nv_fp8_e4m3` cast (host main.cu
            # lines 60-65) before H2D. The kernel issues
            # wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 (cubin:
            # QGMMA.64x64x32.F32.E4M3.E4M3). Same R13 defect class
            # as fp8_e4m3_qmma_dep_chain above: reference must apply the
            # same FP8 E4M3 round-trip the host applies for the
            # (FP8_E4M3, fma_chain, mma_microbench) tolerance triplet to
            # be meaningfully consumed on hardware. R20 follow-up
            # fixed the host runner to actually emit the kernel's d_C
            # output (was: explicitly overwriting d_C with a host
            # un-quantized FP32 matmul, defeating the R19 dispatch
            # — Codex Round-19-review blocking finding #1).
            a = _quantize_matrix_fp8_e4m3(a)
            b = _quantize_matrix_fp8_e4m3(b)
        elif slug == "fp8_e5m2_saturation":
            # R20: this row's host driver pre-quantizes A and B to
            # FP8 E5M2 via `__nv_fp8_e5m2` cast (host main.cu lines
            # 49-54) before H2D. The kernel reads QMMA.16832 with
            # FP8 E5M2 operands. Same R13 / R19 defect class:
            # without this elif the reference would compute FP32 matmul
            # on un-quantized FP32 operands while the kernel sees FP8
            # E5M2-rounded operands — Codex Round-19-review blocking
            # finding #2. The (FP8_E5M2, fma_chain, mma_microbench)
            # tolerance triplet absorbs the FP8*FP8→FP32 hardware
            # multiply rounding only after the reference applies the
            # same FP8 E5M2 round-trip the host applied.
            a = _quantize_matrix_fp8_e5m2(a)
            b = _quantize_matrix_fp8_e5m2(b)
        return _fp32_matmul(a, b)
    shape = tuple(int(v) for v in axes.values() if isinstance(v, int))
    if not shape:
        shape = (64,)
    return deterministic_seeded_tensor(shape, seed, low=-1.0, high=1.0)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
