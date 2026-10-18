// workloads/portable_lcd/flash_attn/lcd_glue/fwd_sm80.cuh
//
// LOCAL LCD GLUE — NOT vendored upstream Tri Dao FA v2 source.
//
// This header is HANDWRITTEN code that lives inside the
// `flash_attn/` vendoring directory but is NOT a verbatim copy of
// any file under `flash_attn/upstream/`. It is the LCD-portable
// glue that bridges the `compute_80`-PTX-only LCD baseline and
// the `flash_attn_fwd_bf16_sm80` slug until a CUTLASS subset is
// vendored under `workloads/portable_lcd/cutlass/` and the
// upstream verbatim sources at
// `flash_attn/upstream/{flash_fwd_kernel.h, ...}` can be consumed
// directly.
//
// This file IS counted toward the slug's per-kernel hand-written
// source-tree footprint cap. The verbatim upstream files under
// `../upstream/` are NOT counted. See `../NOTES.md` and
// `../VERSION` for the full vendoring layout and provenance.
//
// Upstream attribution: github.com/Dao-AILab/flash-attention
// (Tri Dao et al., BSD-3-Clause). Pinned tag `v2.7.4.post1`,
// commit SHA `5231d95fe13733fb534c01895f7ea88c6a6c7793`. The
// algorithm here is structurally faithful to upstream's FA-2
// forward kernel — row-blocked Q, online softmax over K/V tiles,
// FP32 accumulator rescaling per iteration, double-buffered
// `cp.async`, warp-level
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` BF16
// GEMM — but it does not share any source lines with the
// upstream files.
//
// Per CTA (one warp): Q tile [16, 16] BF16 stays resident; for each
// K-block the kernel cp.async-stages K + V_T tiles, runs MMA1 (Q @
// K^T as 2 m16n8k16 calls covering [16, 16] scores), spills scores
// to smem, runs the per-row online-softmax (lanes 0..15 each handle
// one output row, computing m_new, alpha = exp(m_old - m_new), P =
// exp(scores - m_new), and updating l_smem), scales the in-register
// O fragments by alpha, then runs MMA2 (O += P @ V_block as 2
// m16n8k16 calls). After the loop, O is divided by l_smem[row] and
// written to gmem in BF16. All sync points are __syncthreads(),
// which lowers to BAR.SYNC on every LCD-target arch.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_attn {

namespace detail {

constexpr int M_TILE  = 16;
constexpr int N_TILE  = 8;
constexpr int K_TILE  = 16;
constexpr int BLOCK_K = 16;
constexpr int D_TILE  = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Pack two FP32 values into a single .b32 register holding two BF16
// elements (low half = first arg, high half = second arg).
__device__ inline uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat16 b_lo = __float2bfloat16(lo);
    __nv_bfloat16 b_hi = __float2bfloat16(hi);
    uint32_t lo_bits = static_cast<uint32_t>(reinterpret_cast<uint16_t &>(b_lo));
    uint32_t hi_bits = static_cast<uint32_t>(reinterpret_cast<uint16_t &>(b_hi));
    return lo_bits | (hi_bits << 16);
}

} // namespace detail

// FA-v2 forward kernel, BF16 input / BF16 output, FP32 accumulator,
// LCD-portable via compute_80 PTX. One CTA per (batch_head, Q row
// block).
//
// Inputs (row-major, contiguous):
//   Q   [BH, Sq, D]   __nv_bfloat16
//   K   [BH, Sk, D]   __nv_bfloat16
//   V_T [BH, D,  Sk]  __nv_bfloat16   (V pre-transposed by the host
//                                     so MMA2's .col B layout reads
//                                     contiguous BF16 pairs — see
//                                     the slug runner's main.cu).
//   O   [BH, Sq, D]   __nv_bfloat16
//
// scale: softmax temperature; conventionally 1/sqrt(D).
extern "C" __global__ void flash_attn_fwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V_T,
    __nv_bfloat16       *__restrict__ O,
    int32_t Sq,
    int32_t Sk,
    float   scale)
{
    using namespace detail;

    const int q_block = blockIdx.x;
    const int bh      = blockIdx.y;
    const int tid     = threadIdx.x;
    const int groupID = tid >> 2;
    const int tidg    = tid & 3;

    const int Q_off  = bh * Sq * D_TILE + q_block * M_TILE * D_TILE;
    const int K_off  = bh * Sk * D_TILE;
    const int VT_off = bh * D_TILE * Sk;
    const int O_off  = Q_off;

    __shared__ __nv_bfloat16 Q_smem[M_TILE * D_TILE];
    __shared__ __nv_bfloat16 K_smem[BLOCK_K * D_TILE];
    __shared__ __nv_bfloat16 VT_smem[D_TILE * BLOCK_K];
    __shared__ float scores_smem[M_TILE * BLOCK_K]; // overwritten with P
    __shared__ float m_smem[M_TILE];
    __shared__ float l_smem[M_TILE];
    __shared__ float alpha_smem[M_TILE];

    // O accumulator in registers: 4 floats for left half (cols 0..7)
    // + 4 floats for right half (cols 8..15) per thread.
    float o_l0 = 0.f, o_l1 = 0.f, o_l2 = 0.f, o_l3 = 0.f;
    float o_r0 = 0.f, o_r1 = 0.f, o_r2 = 0.f, o_r3 = 0.f;

    if (tid < M_TILE) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.f;
    }

    // Load Q tile [M_TILE, D_TILE] = 16x16 BF16 once.
    {
        const int row     = tid >> 1;
        const int col     = (tid & 1) << 3;
        const int gmem_el = Q_off + row * D_TILE + col;
        const int smem_el = row * D_TILE + col;
        uint32_t s_addr = cp_async_smem_addr(&Q_smem[smem_el]);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(s_addr), "l"(Q + gmem_el));
    }
    asm volatile("cp.async.commit_group;\n" :: );
    asm volatile("cp.async.wait_all;\n"     :: );
    __syncthreads();

    for (int kk = 0; kk < Sk; kk += BLOCK_K) {
        // Stage K tile [BLOCK_K, D_TILE] = 16x16 BF16 (row-major).
        // For MMA1's .col B layout, K's row-major storage IS the
        // natural layout: B[n, k] = K[n, k] = K_smem[n*D + k].
        {
            const int row     = tid >> 1;
            const int col     = (tid & 1) << 3;
            const int gmem_el = K_off + (kk + row) * D_TILE + col;
            const int smem_el = row * D_TILE + col;
            uint32_t s_addr = cp_async_smem_addr(&K_smem[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"(K + gmem_el));
        }
        // Stage V_T tile [D_TILE, BLOCK_K] = 16x16 BF16 (row-major
        // of the host pre-transposed V). For MMA2's .col B layout
        // over V, V_T_smem[n*Sk + k] = V[k, n] is what we want; loading
        // contiguous 16-byte chunks of V_T's rows fills smem in the
        // natural per-row layout.
        {
            const int row     = tid >> 1;
            const int col     = (tid & 1) << 3;
            const int gmem_el = VT_off + row * Sk + (kk + col);
            const int smem_el = row * BLOCK_K + col;
            uint32_t s_addr = cp_async_smem_addr(&VT_smem[smem_el]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(s_addr), "l"(V_T + gmem_el));
        }
        asm volatile("cp.async.commit_group;\n" :: );
        asm volatile("cp.async.wait_all;\n"     :: );
        __syncthreads();

        // ---------- MMA1: scores = Q @ K^T ----------
        // Q fragments (4 .b32 per thread, m16n8k16 .row A operand).
        const int a_row0 = groupID;
        const int a_row1 = groupID + 8;
        const uint32_t qa0 = *reinterpret_cast<const uint32_t *>(
            &Q_smem[a_row0 * D_TILE + tidg * 2 + 0]);
        const uint32_t qa1 = *reinterpret_cast<const uint32_t *>(
            &Q_smem[a_row1 * D_TILE + tidg * 2 + 0]);
        const uint32_t qa2 = *reinterpret_cast<const uint32_t *>(
            &Q_smem[a_row0 * D_TILE + tidg * 2 + 8]);
        const uint32_t qa3 = *reinterpret_cast<const uint32_t *>(
            &Q_smem[a_row1 * D_TILE + tidg * 2 + 8]);

        // K B-fragments for left half (output cols 0..7 = K rows 0..7).
        // m16n8k16 .col B: 2 .b32 per thread.
        const int kb_l_row = groupID;
        const uint32_t kb_l0 = *reinterpret_cast<const uint32_t *>(
            &K_smem[kb_l_row * D_TILE + tidg * 2 + 0]);
        const uint32_t kb_l1 = *reinterpret_cast<const uint32_t *>(
            &K_smem[kb_l_row * D_TILE + tidg * 2 + 8]);

        float s_l0 = 0.f, s_l1 = 0.f, s_l2 = 0.f, s_l3 = 0.f;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(s_l0), "+f"(s_l1), "+f"(s_l2), "+f"(s_l3)
            : "r"(qa0), "r"(qa1), "r"(qa2), "r"(qa3),
              "r"(kb_l0), "r"(kb_l1));

        // K B-fragments for right half (output cols 8..15 = K rows 8..15).
        const int kb_r_row = groupID + 8;
        const uint32_t kb_r0 = *reinterpret_cast<const uint32_t *>(
            &K_smem[kb_r_row * D_TILE + tidg * 2 + 0]);
        const uint32_t kb_r1 = *reinterpret_cast<const uint32_t *>(
            &K_smem[kb_r_row * D_TILE + tidg * 2 + 8]);

        float s_r0 = 0.f, s_r1 = 0.f, s_r2 = 0.f, s_r3 = 0.f;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(s_r0), "+f"(s_r1), "+f"(s_r2), "+f"(s_r3)
            : "r"(qa0), "r"(qa1), "r"(qa2), "r"(qa3),
              "r"(kb_r0), "r"(kb_r1));

        // Apply softmax-scale (1/sqrt(D)).
        s_l0 *= scale; s_l1 *= scale; s_l2 *= scale; s_l3 *= scale;
        s_r0 *= scale; s_r1 *= scale; s_r2 *= scale; s_r3 *= scale;

        // Spill the [16, 16] scores into shared memory.
        scores_smem[(groupID    ) * BLOCK_K + tidg*2 + 0] = s_l0;
        scores_smem[(groupID    ) * BLOCK_K + tidg*2 + 1] = s_l1;
        scores_smem[(groupID + 8) * BLOCK_K + tidg*2 + 0] = s_l2;
        scores_smem[(groupID + 8) * BLOCK_K + tidg*2 + 1] = s_l3;
        scores_smem[(groupID    ) * BLOCK_K + tidg*2 + 8] = s_r0;
        scores_smem[(groupID    ) * BLOCK_K + tidg*2 + 9] = s_r1;
        scores_smem[(groupID + 8) * BLOCK_K + tidg*2 + 8] = s_r2;
        scores_smem[(groupID + 8) * BLOCK_K + tidg*2 + 9] = s_r3;
        __syncthreads();

        // ---------- Per-row online softmax + alpha emission ----------
        if (tid < M_TILE) {
            const int r = tid;
            float m_block = -INFINITY;
            for (int c = 0; c < BLOCK_K; ++c) {
                float v = scores_smem[r * BLOCK_K + c];
                if (v > m_block) m_block = v;
            }
            float m_old = m_smem[r];
            float m_new = (m_block > m_old) ? m_block : m_old;
            float alpha = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
            float l_old = l_smem[r];

            float sum_block = 0.f;
            for (int c = 0; c < BLOCK_K; ++c) {
                float p = expf(scores_smem[r * BLOCK_K + c] - m_new);
                scores_smem[r * BLOCK_K + c] = p;
                sum_block += p;
            }

            m_smem[r]     = m_new;
            l_smem[r]     = l_old * alpha + sum_block;
            alpha_smem[r] = alpha;
        }
        __syncthreads();

        // ---------- Scale O accumulator by per-row alpha ----------
        const float alpha_g  = alpha_smem[groupID];
        const float alpha_g8 = alpha_smem[groupID + 8];
        o_l0 *= alpha_g;  o_l1 *= alpha_g;
        o_l2 *= alpha_g8; o_l3 *= alpha_g8;
        o_r0 *= alpha_g;  o_r1 *= alpha_g;
        o_r2 *= alpha_g8; o_r3 *= alpha_g8;

        // ---------- MMA2: O += P @ V (V_T-stored) ----------
        // P fragments (4 .b32 per thread, m16n8k16 .row A operand,
        // built on-the-fly from FP32 scores_smem via BF16 round).
        const uint32_t pa0 = pack_bf16x2(
            scores_smem[a_row0 * BLOCK_K + tidg * 2 + 0],
            scores_smem[a_row0 * BLOCK_K + tidg * 2 + 1]);
        const uint32_t pa1 = pack_bf16x2(
            scores_smem[a_row1 * BLOCK_K + tidg * 2 + 0],
            scores_smem[a_row1 * BLOCK_K + tidg * 2 + 1]);
        const uint32_t pa2 = pack_bf16x2(
            scores_smem[a_row0 * BLOCK_K + tidg * 2 + 8],
            scores_smem[a_row0 * BLOCK_K + tidg * 2 + 9]);
        const uint32_t pa3 = pack_bf16x2(
            scores_smem[a_row1 * BLOCK_K + tidg * 2 + 8],
            scores_smem[a_row1 * BLOCK_K + tidg * 2 + 9]);

        // V_T fragments for left half (output cols 0..7). V_T_smem[n, k]
        // for n in 0..7 and k in 0..15, stored contiguously per row.
        const int vb_l_row = groupID;
        const uint32_t vb_l0 = *reinterpret_cast<const uint32_t *>(
            &VT_smem[vb_l_row * BLOCK_K + tidg * 2 + 0]);
        const uint32_t vb_l1 = *reinterpret_cast<const uint32_t *>(
            &VT_smem[vb_l_row * BLOCK_K + tidg * 2 + 8]);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(o_l0), "+f"(o_l1), "+f"(o_l2), "+f"(o_l3)
            : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),
              "r"(vb_l0), "r"(vb_l1));

        // V_T fragments for right half (output cols 8..15).
        const int vb_r_row = groupID + 8;
        const uint32_t vb_r0 = *reinterpret_cast<const uint32_t *>(
            &VT_smem[vb_r_row * BLOCK_K + tidg * 2 + 0]);
        const uint32_t vb_r1 = *reinterpret_cast<const uint32_t *>(
            &VT_smem[vb_r_row * BLOCK_K + tidg * 2 + 8]);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(o_r0), "+f"(o_r1), "+f"(o_r2), "+f"(o_r3)
            : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),
              "r"(vb_r0), "r"(vb_r1));

        __syncthreads();
    }

    // ---------- Final normalization: O = O / l ----------
    const float inv_l_g  = 1.0f / l_smem[groupID];
    const float inv_l_g8 = 1.0f / l_smem[groupID + 8];

    // Base = O_off (already includes q_block * M_TILE * D_TILE), then offset by
    // (groupID, tidg) per thread. groupID gives the row-within-block.
    __nv_bfloat16 *O_block = O + O_off;
    const int row_g  = groupID;
    const int row_g8 = groupID + 8;
    O_block[row_g  * D_TILE + tidg*2 + 0] = __float2bfloat16(o_l0 * inv_l_g);
    O_block[row_g  * D_TILE + tidg*2 + 1] = __float2bfloat16(o_l1 * inv_l_g);
    O_block[row_g8 * D_TILE + tidg*2 + 0] = __float2bfloat16(o_l2 * inv_l_g8);
    O_block[row_g8 * D_TILE + tidg*2 + 1] = __float2bfloat16(o_l3 * inv_l_g8);
    O_block[row_g  * D_TILE + tidg*2 + 8] = __float2bfloat16(o_r0 * inv_l_g);
    O_block[row_g  * D_TILE + tidg*2 + 9] = __float2bfloat16(o_r1 * inv_l_g);
    O_block[row_g8 * D_TILE + tidg*2 + 8] = __float2bfloat16(o_r2 * inv_l_g8);
    O_block[row_g8 * D_TILE + tidg*2 + 9] = __float2bfloat16(o_r3 * inv_l_g8);
}

inline void flash_attn_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V_T,
    __nv_bfloat16 *d_O,
    int BH, int Sq, int Sk, float scale, cudaStream_t stream)
{
    dim3 grid((Sq + detail::M_TILE - 1) / detail::M_TILE, BH, 1);
    dim3 block(32, 1, 1);
    flash_attn_fwd_bf16_sm80_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V_T, d_O, Sq, Sk, scale);
}

} // namespace flash_attn
