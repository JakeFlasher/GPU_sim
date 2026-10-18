// slug: gemm_hmma_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 HMMA microbench via warp-level
//   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// LCD-portable variant of workloads/simbench_bw/l1/bf16_hmma_gemm_traceable
// adding cp.async-staged shared-memory loading so the SASS contains both a
// CTA-level synchronization barrier (BAR.SYNC, the __syncthreads() lowering)
// and an async-copy dependency barrier (LDGDEPBAR, the cp.async.wait_all
// lowering). Together these populate the SASS regression material the
// phase-injection's `is_bar` substring-match bug needs.
//
// Single-warp CTA. Each warp computes one [16x8] output tile. cp.async
// stages a [16xK_TILE] BF16 A-tile and a [8xK_TILE] BF16 B-tile (B is col-
// major: sB[col][k]) into shared memory per K iteration; __syncthreads()
// before fragment loads + after MMA ensures shared-memory reuse safety.
//
// Per-thread fragment layout (m16n8k16, .row x .col):
//   lane = 0..31; groupID = lane / 4; tidg = lane % 4.
//   A (4 x b32 per thread, 8 BF16):
//     a0: row=groupID,   k=[tidg*2 + 0 .. tidg*2 + 1]
//     a1: row=groupID+8, k=[tidg*2 + 0 .. tidg*2 + 1]
//     a2: row=groupID,   k=[tidg*2 + 8 .. tidg*2 + 9]
//     a3: row=groupID+8, k=[tidg*2 + 8 .. tidg*2 + 9]
//   B (2 x b32 per thread, 4 BF16; B is .col so K is the leading dim):
//     b0: col=groupID, k=[tidg*2 + 0 .. tidg*2 + 1]
//     b1: col=groupID, k=[tidg*2 + 8 .. tidg*2 + 9]
//
// Launch: grid = (M/16, N/8, 1), block = (32, 1, 1).
// Requires M % 16 == 0, N % 8 == 0, K % 16 == 0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int K_TILE = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
} // namespace

extern "C" __global__ void gemm_hmma_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ A, // [M, K] row-major
    const __nv_bfloat16 *__restrict__ B, // [N, K] col-major (B is .col)
    float               *__restrict__ C, // [M, N] row-major
    int32_t M, int32_t N, int32_t K)
{
    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int lane    = threadIdx.x;
    const int groupID = lane >> 2;
    const int tidg    = lane & 3;

    __shared__ __nv_bfloat16 sA[16 * K_TILE];
    __shared__ __nv_bfloat16 sB[8  * K_TILE];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const int gA_row_base = tile_m * 16;
    const int gB_row_base = tile_n * 8;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        // Stage A: lanes 0..31 each load 16 bytes (8 BF16) of the A tile.
        // Layout: lane -> (row = lane / 2, col = (lane % 2) * 8). Covers
        // all 16x16 = 256 BF16 elements with 32 16-byte cp.async copies.
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

        // Stage B: only lanes 0..15 participate (B tile is half the size of A).
        if (lane < 16) {
            const int row     = lane >> 1;        // 0..7
            const int col     = (lane & 1) << 3;  // 0 or 8
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
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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

extern "C" void gemm_hmma_bf16_sm80_launch(
    const __nv_bfloat16 *d_A, const __nv_bfloat16 *d_B, float *d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid((M + 15) / 16, (N + 7) / 8, 1);
    dim3 block(32, 1, 1);
    gemm_hmma_bf16_sm80_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
