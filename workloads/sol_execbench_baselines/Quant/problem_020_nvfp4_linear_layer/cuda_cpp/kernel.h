#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Stage 1: take scaled FP32 values (already clipped to [-6, 6] by the
// host quant prepass) and emit packed FP4 nibbles using the
// reference's exact threshold table.
//
//   scaled_x      : [M, K] FP32 contiguous
//   packed_out    : [M, K/2] uint8 contiguous (low nibble = even k)
void problem_020_fp4_pack_launcher(
    const torch::Tensor& scaled_x,
    torch::Tensor packed_out,
    cudaStream_t stream);

// Stage 2: remap a plain ``[D, K/16]`` FP8 e4m3 scale tensor (the
// shape produced by the reference's ``BlockWiseScalerNVFP4.quantize``)
// into the CUTLASS sm_120 dense block-scaled GEMM's interleaved
// scale-factor layout.
//
// The layout is the closed form derived from
// ``Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA`` for shape
// ``(M, N, K, 1)`` with K=8192, scale block size 16:
//
//   offset = (d % 32) * 16
//          + ((d / 32) % 4) * 4
//          + (d / 128) * (K_blocks * 128)
//          + (kb % 4)
//          + (kb / 4) * K_blocks
//
// Both SFA and SFB use this formula (D = M for SFA, N for SFB).
// The destination buffer is sized to ``D * K_blocks`` FP8 e4m3
// elements (= the plain element count); only the index pattern
// differs.
//
//   plain_scales : [D, K_blocks] FP8 e4m3
//   sfa_out      : [D * K_blocks] FP8 e4m3
void problem_020_remap_scales_launcher(
    const torch::Tensor& plain_scales,
    torch::Tensor sfa_out,
    int D,
    int K_blocks,
    cudaStream_t stream);

// Stage 3: dispatch the CUTLASS sm_120 dense block-scaled NVFP4 GEMM
// (matches example ``79a_blackwell_geforce_nvfp4_bf16_gemm`` in the
// local CUTLASS v4.4.1 tree under ``cutlass/``).
//
//   x_fp4         : [M, K/2] uint8 packed E2M1 (CUTLASS A operand)
//   x_sfa         : [M*K/16] FP8 e4m3, CUTLASS interleaved SFA layout
//   w_fp4         : [N, K/2] uint8 packed E2M1 (CUTLASS B operand)
//   w_sfb         : [N*K/16] FP8 e4m3, CUTLASS interleaved SFB layout
//   y_bf16        : [M, N] BF16 destination (CUTLASS D operand)
//   global_scale  : x_global_decode * w_global_decode (FP32 scalar,
//                   applied as the epilogue's ``alpha``).
void problem_020_fp4_gemm_launcher(
    const torch::Tensor& x_fp4,
    const torch::Tensor& x_sfa,
    const torch::Tensor& w_fp4,
    const torch::Tensor& w_sfb,
    torch::Tensor y_bf16,
    float global_scale,
    cudaStream_t stream);
