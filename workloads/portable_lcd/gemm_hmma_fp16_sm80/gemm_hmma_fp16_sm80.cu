// slug: gemm_hmma_fp16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// FP16 HMMA microbench via warp-level
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// Sister kernel of `gemm_hmma_bf16_sm80`. Same single-warp CTA,
// same 16x8 output tile, same cp.async-staged shared-memory
// pipeline; only the MMA datatype changes from BF16 to FP16.
// Pairing the two kernels gives the suite tensor-core opcode-mix
// coverage on both Ampere FP16/BF16 HMMA encodings — the SASS
// mnemonic narrows from the BF16 sibling's `HMMA.16816.F32.BF16`
// to `HMMA.16816.F32.F16` (this kernel) on sm_80+ targets.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int K_TILE = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
} // namespace

extern "C" __global__ void gemm_hmma_fp16_sm80_kernel(
    const __half *__restrict__ A, // [M, K] row-major
    const __half *__restrict__ B, // [N, K] col-major (B is .col)
    float        *__restrict__ C, // [M, N] row-major
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    __shared__ __half sA[16 * K_TILE];
    __shared__ __half sB[8  * K_TILE];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const int gA_row_base = tile_m * 16;
    const int gB_row_base = tile_n * 8;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        // Stage A: 32 cp.async ops, 16 bytes each, covering 16x16 FP16.
        {
            const int row     = lane >> 1;
            const int col     = (lane & 1) << 3;
            const int gmem_el = (gA_row_base + row) * K + k0 + col;
            const int smem_el = row * K_TILE + col;
            uint32_t s_addr = cp_async_smem_addr(&sA[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"(A + gmem_el));
        }

        // Stage B: first 16 lanes load 8x16 FP16 = 256 bytes / 16 bytes per op = 16 ops.
        if (lane < 16) {
            const int row     = lane >> 1;
            const int col     = (lane & 1) << 3;
            const int gmem_el = (gB_row_base + row) * K + k0 + col;
            const int smem_el = row * K_TILE + col;
            uint32_t s_addr = cp_async_smem_addr(&sB[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"(B + gmem_el));
        }

        asm volatile("cp.async.commit_group;\n" :: );
        asm volatile("cp.async.wait_all;\n"     :: );
        __syncthreads();

        const int a_row0 = groupID;
        const int a_row1 = groupID + 8;
        const int b_row  = groupID;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(&sA[a_row0 * K_TILE + tidg * 2 + 0]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(&sA[a_row1 * K_TILE + tidg * 2 + 0]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t *>(&sA[a_row0 * K_TILE + tidg * 2 + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t *>(&sA[a_row1 * K_TILE + tidg * 2 + 8]);
        const uint32_t b0 = *reinterpret_cast<const uint32_t *>(&sB[b_row  * K_TILE + tidg * 2 + 0]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t *>(&sB[b_row  * K_TILE + tidg * 2 + 8]);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

        __syncthreads();
    }

    const int c_row0 = tile_m * 16 + groupID;
    const int c_row1 = tile_m * 16 + groupID + 8;
    const int c_col  = tile_n * 8  + tidg * 2;
    C[c_row0 * N + c_col + 0] = d0;
    C[c_row0 * N + c_col + 1] = d1;
    C[c_row1 * N + c_col + 0] = d2;
    C[c_row1 * N + c_col + 1] = d3;
}

extern "C" void gemm_hmma_fp16_sm80_launch(
    const __half *d_A, const __half *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    gemm_hmma_fp16_sm80_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
