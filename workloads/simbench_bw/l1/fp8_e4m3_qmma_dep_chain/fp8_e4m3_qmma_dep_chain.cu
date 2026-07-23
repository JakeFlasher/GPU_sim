// slug: fp8_e4m3_qmma_dep_chain
// tier: P
// arch_gate: sm_90a sm_120
//
// FP8 E4M3 dependent-QMMA chain via warp-level
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX is
// the contract enforcement; cubin must contain QMMA E4M3 SASS.
//
// Per-thread fragment layout (m16n8k32, .row x .col):
//   lane = 0..31; groupID = lane / 4; tidg = lane % 4.
//   A (4 x b32 per thread, 16 FP8E4M3):
//     a0: row=groupID,   k=[tidg*4 + 0 .. tidg*4 + 3]
//     a1: row=groupID+8, k=[tidg*4 + 0 .. tidg*4 + 3]
//     a2: row=groupID,   k=[tidg*4 + 16 .. tidg*4 + 19]
//     a3: row=groupID+8, k=[tidg*4 + 16 .. tidg*4 + 19]
//   B (2 x b32 per thread, 8 FP8E4M3; B is .col):
//     b0: col=groupID, k=[tidg*4 + 0 .. tidg*4 + 3]
//     b1: col=groupID, k=[tidg*4 + 16 .. tidg*4 + 19]
//   D (4 x f32 per thread): row=groupID/+8, col=tidg*2/+1.
//
// Launch: grid = (M/16, N/8, 1), block = (32, 1, 1).

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void fp8_e4m3_qmma_dep_chain_kernel(
    const __nv_fp8_e4m3 *__restrict__ A, const __nv_fp8_e4m3 *__restrict__ B, float *__restrict__ C,
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

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
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8 + tidg * 2;
    C[c_row0 * N + c_col + 0] = d0;
    C[c_row0 * N + c_col + 1] = d1;
    C[c_row1 * N + c_col + 0] = d2;
    C[c_row1 * N + c_col + 1] = d3;
}

extern "C" void fp8_e4m3_qmma_dep_chain_launch(
    const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    fp8_e4m3_qmma_dep_chain_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
