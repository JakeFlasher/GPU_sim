// slug: tf32_hmma_gemm_traceable
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// TF32 HMMA microbench via warp-level
//   mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
//
// Per BL-20260424-kernel-contract-vs-shape-match the inline PTX
// is the contract enforcement; cubin must contain HMMA TF32 SASS.
// TF32 is FP32 with the low 13 mantissa bits zeroed; the host
// driver applies the rounding before H2D, so the kernel reads
// raw 32-bit registers as TF32 literals.
//
// Per-thread fragment layout (m16n8k8, .row x .col):
//   lane = 0..31; groupID = lane / 4; tidg = lane % 4.
//   A (4 x b32 per thread, 4 TF32):
//     a0: row=groupID,   k=tidg
//     a1: row=groupID+8, k=tidg
//     a2: row=groupID,   k=tidg+4
//     a3: row=groupID+8, k=tidg+4
//   B (2 x b32 per thread, 2 TF32; B is .col so K is the leading dim):
//     b0: col=groupID, k=tidg
//     b1: col=groupID, k=tidg+4
//   D (4 x f32 per thread): same as the m16n8k16 BF16 variant.
//
// Launch: grid = (M/16, N/8, 1), block = (32, 1, 1).
// Requires M % 16 == 0, N % 8 == 0, K % 8 == 0.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void tf32_hmma_gemm_traceable_kernel(
    const float *__restrict__ A, // [M, K] row-major; values are FP32 rounded to TF32 by the host.
    const float *__restrict__ B, // [N, K] col-major; values are FP32 rounded to TF32 by the host.
    float       *__restrict__ C, // [M, N] row-major.
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

    for (int k0 = 0; k0 < K; k0 += 8) {
        uint32_t a0 = *reinterpret_cast<const uint32_t *>(A + a_row0_base + k0 + tidg);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(A + a_row1_base + k0 + tidg);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(A + a_row0_base + k0 + tidg + 4);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(A + a_row1_base + k0 + tidg + 4);
        uint32_t b0 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg);
        uint32_t b1 = *reinterpret_cast<const uint32_t *>(B + b_col_base + k0 + tidg + 4);

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
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

extern "C" void tf32_hmma_gemm_traceable_launch(
    const float *d_A, const float *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    tf32_hmma_gemm_traceable_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
