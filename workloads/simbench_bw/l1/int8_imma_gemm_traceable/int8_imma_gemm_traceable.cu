// slug: int8_imma_gemm_traceable
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// INT8 IMMA microbench via warp-level
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX is the
// contract enforcement; cubin must contain IMMA SASS.
//
// Per-thread fragment layout (m16n8k32, .row x .col):
//   lane = 0..31; groupID = lane / 4; tidg = lane & 3.
//   A (4 x b32 per thread, 16 INT8):
//     a0: row=groupID,   k=[tidg*4 + 0 .. tidg*4 + 3]
//     a1: row=groupID+8, k=[tidg*4 + 0 .. tidg*4 + 3]
//     a2: row=groupID,   k=[tidg*4 + 16 .. tidg*4 + 19]
//     a3: row=groupID+8, k=[tidg*4 + 16 .. tidg*4 + 19]
//   B (2 x b32 per thread, 8 INT8; B is .col):
//     b0: col=groupID, k=[tidg*4 + 0 .. tidg*4 + 3]
//     b1: col=groupID, k=[tidg*4 + 16 .. tidg*4 + 19]
//   D (4 x s32 per thread): row=groupID/+8, col=tidg*2/+1.
//
// Output is dequantized in-kernel: C[m, n] = D_int32[m, n] * (a_scale * b_scale)
// where a_scale, b_scale are the host-driver quantization scales such that
// (raw_fp32 * scale) round to int8. The host driver passes both scales as
// kernel arguments so the FP32 output matches references/mma_microbench.py::_fp32_matmul
// up to INT8 quantization noise.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void int8_imma_gemm_traceable_kernel(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B, float *__restrict__ C,
    int32_t M, int32_t N, int32_t K, float dequant_scale)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    const int a_row0_base = (tile_m * 16 + groupID    ) * K;
    const int a_row1_base = (tile_m * 16 + groupID + 8) * K;
    const int b_col_base  = (tile_n * 8  + groupID    ) * K;

    for (int k0 = 0; k0 < K; k0 += 32) {
        uint32_t a0 = *reinterpret_cast<const uint32_t *>(A + a_row0_base + k0 + tidg * 4);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(A + a_row1_base + k0 + tidg * 4);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(A + a_row0_base + k0 + tidg * 4 + 16);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(A + a_row1_base + k0 + tidg * 4 + 16);
        uint32_t b0 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg * 4);
        uint32_t b1 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg * 4 + 16);

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8 + tidg * 2;
    C[c_row0 * N + c_col + 0] = (float)d0 * dequant_scale;
    C[c_row0 * N + c_col + 1] = (float)d1 * dequant_scale;
    C[c_row1 * N + c_col + 0] = (float)d2 * dequant_scale;
    C[c_row1 * N + c_col + 1] = (float)d3 * dequant_scale;
}

extern "C" void int8_imma_gemm_traceable_launch(
    const int8_t *d_A, const int8_t *d_B, float *d_C,
    int M, int N, int K, float dequant_scale, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    int8_imma_gemm_traceable_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, dequant_scale);
}
