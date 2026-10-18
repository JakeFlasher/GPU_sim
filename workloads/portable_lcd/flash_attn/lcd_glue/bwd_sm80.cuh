// workloads/portable_lcd/flash_attn/lcd_glue/bwd_sm80.cuh
//
// FA-2 backward (softmax-recompute). Per CTA = one Q row block of 16
// rows, one warp. phase 1: forward online softmax filling (m, l).
// phase 2 sweep 1: full-row row_dot[q] = sum_{n in [0, Sk)}
// dP[q, n] * P[q, n]. phase 2 sweep 2: dS = (dP - row_dot) * P *
// scale; dV / dK accumulated via atomicAdd, dQ via in-register MMA.
// Sister to fwd_sm80.cuh. Upstream attribution: Dao-AILab FA
// v2.7.4.post1 (BSD-3); this file is local LCD glue, not vendored.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_attn_bwd {
namespace detail {

constexpr int M_TILE  = 16;
constexpr int N_TILE  = 8;
constexpr int K_TILE  = 16;
constexpr int BLOCK_K = 16;
constexpr int D_TILE  = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ inline uint32_t pack_bf16_pair(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t lo = static_cast<uint32_t>(reinterpret_cast<uint16_t &>(a));
    uint32_t hi = static_cast<uint32_t>(reinterpret_cast<uint16_t &>(b));
    return lo | (hi << 16);
}

__device__ inline void stage16(__nv_bfloat16 *smem, int smem_idx,
                               const __nv_bfloat16 *gmem_base, int gmem_idx) {
    uint32_t s = cp_async_smem_addr(&smem[smem_idx]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(gmem_base + gmem_idx));
}

// Stage K_smem from K and VT_smem from V_T for the current K block;
// fence with cp.async.commit_group + wait_all + __syncthreads.
__device__ inline void stage_K_VT(__nv_bfloat16 *K_smem, __nv_bfloat16 *VT_smem,
                                  const __nv_bfloat16 *K, const __nv_bfloat16 *V_T,
                                  int K_off, int VT_off, int kk, int Sk, int tid) {
    const int row = tid >> 1, col = (tid & 1) << 3;
    stage16(K_smem,  row * D_TILE  + col, K,   K_off  + (kk + row) * D_TILE + col);
    stage16(VT_smem, row * BLOCK_K + col, V_T, VT_off + row * Sk + (kk + col));
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" :: );
    __syncthreads();
}

// Q @ K^T via 2 m16n8k16 calls, applies softmax-scale, spills the
// [16, 16] scores buffer to smem.
__device__ inline void recompute_scores(const __nv_bfloat16 *Q_smem, const __nv_bfloat16 *K_smem,
                                        float *scores_smem, int groupID, int tidg, float scale) {
    const int a_row0 = groupID, a_row1 = groupID + 8;
    const uint32_t qa0 = *reinterpret_cast<const uint32_t *>(&Q_smem[a_row0 * D_TILE + tidg * 2 + 0]);
    const uint32_t qa1 = *reinterpret_cast<const uint32_t *>(&Q_smem[a_row1 * D_TILE + tidg * 2 + 0]);
    const uint32_t qa2 = *reinterpret_cast<const uint32_t *>(&Q_smem[a_row0 * D_TILE + tidg * 2 + 8]);
    const uint32_t qa3 = *reinterpret_cast<const uint32_t *>(&Q_smem[a_row1 * D_TILE + tidg * 2 + 8]);
    const uint32_t kb_l0 = *reinterpret_cast<const uint32_t *>(&K_smem[groupID       * D_TILE + tidg * 2 + 0]);
    const uint32_t kb_l1 = *reinterpret_cast<const uint32_t *>(&K_smem[groupID       * D_TILE + tidg * 2 + 8]);
    const uint32_t kb_r0 = *reinterpret_cast<const uint32_t *>(&K_smem[(groupID + 8) * D_TILE + tidg * 2 + 0]);
    const uint32_t kb_r1 = *reinterpret_cast<const uint32_t *>(&K_smem[(groupID + 8) * D_TILE + tidg * 2 + 8]);

    float s_l0=0.f, s_l1=0.f, s_l2=0.f, s_l3=0.f, s_r0=0.f, s_r1=0.f, s_r2=0.f, s_r3=0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(s_l0), "+f"(s_l1), "+f"(s_l2), "+f"(s_l3)
        : "r"(qa0), "r"(qa1), "r"(qa2), "r"(qa3), "r"(kb_l0), "r"(kb_l1));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(s_r0), "+f"(s_r1), "+f"(s_r2), "+f"(s_r3)
        : "r"(qa0), "r"(qa1), "r"(qa2), "r"(qa3), "r"(kb_r0), "r"(kb_r1));

    const int g0 = a_row0 * BLOCK_K + tidg*2, g1 = a_row1 * BLOCK_K + tidg*2;
    scores_smem[g0+0] = s_l0*scale; scores_smem[g0+1] = s_l1*scale;
    scores_smem[g1+0] = s_l2*scale; scores_smem[g1+1] = s_l3*scale;
    scores_smem[g0+8] = s_r0*scale; scores_smem[g0+9] = s_r1*scale;
    scores_smem[g1+8] = s_r2*scale; scores_smem[g1+9] = s_r3*scale;
}

// Recompute scores -> P -> dP for the current K block, used by both
// phase 2 sweeps. Caller must __syncthreads before and after.
__device__ inline void recompute_P_dP(
    const __nv_bfloat16 *Q_smem, const __nv_bfloat16 *K_smem, const __nv_bfloat16 *VT_smem,
    const __nv_bfloat16 *dO_smem, float *scores_smem, float *P_smem, float *dP_smem,
    const float *m_smem, const float *l_smem, int groupID, int tidg, int tid, float scale) {
    recompute_scores(Q_smem, K_smem, scores_smem, groupID, tidg, scale);
    __syncthreads();

    if (tid < M_TILE) {
        const int r = tid;
        const float m = m_smem[r], inv_l = 1.0f / l_smem[r];
        for (int c = 0; c < BLOCK_K; ++c)
            P_smem[r * BLOCK_K + c] = expf(scores_smem[r * BLOCK_K + c] - m) * inv_l;
    }
    __syncthreads();

    const int a_row0 = groupID, a_row1 = groupID + 8;
    const uint32_t doa0 = *reinterpret_cast<const uint32_t *>(&dO_smem[a_row0 * D_TILE + tidg * 2 + 0]);
    const uint32_t doa1 = *reinterpret_cast<const uint32_t *>(&dO_smem[a_row1 * D_TILE + tidg * 2 + 0]);
    const uint32_t doa2 = *reinterpret_cast<const uint32_t *>(&dO_smem[a_row0 * D_TILE + tidg * 2 + 8]);
    const uint32_t doa3 = *reinterpret_cast<const uint32_t *>(&dO_smem[a_row1 * D_TILE + tidg * 2 + 8]);

    auto pack_vt = [&](int n, int k) -> uint32_t {
        return pack_bf16_pair(VT_smem[k * BLOCK_K + n], VT_smem[(k + 1) * BLOCK_K + n]);
    };
    const uint32_t vb_l0 = pack_vt(groupID,     tidg * 2 + 0);
    const uint32_t vb_l1 = pack_vt(groupID,     tidg * 2 + 8);
    const uint32_t vb_r0 = pack_vt(groupID + 8, tidg * 2 + 0);
    const uint32_t vb_r1 = pack_vt(groupID + 8, tidg * 2 + 8);

    float dp_l0=0.f, dp_l1=0.f, dp_l2=0.f, dp_l3=0.f, dp_r0=0.f, dp_r1=0.f, dp_r2=0.f, dp_r3=0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(dp_l0), "+f"(dp_l1), "+f"(dp_l2), "+f"(dp_l3)
        : "r"(doa0), "r"(doa1), "r"(doa2), "r"(doa3), "r"(vb_l0), "r"(vb_l1));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(dp_r0), "+f"(dp_r1), "+f"(dp_r2), "+f"(dp_r3)
        : "r"(doa0), "r"(doa1), "r"(doa2), "r"(doa3), "r"(vb_r0), "r"(vb_r1));

    const int g0 = a_row0 * BLOCK_K + tidg*2, g1 = a_row1 * BLOCK_K + tidg*2;
    dP_smem[g0+0] = dp_l0; dP_smem[g0+1] = dp_l1;
    dP_smem[g1+0] = dp_l2; dP_smem[g1+1] = dp_l3;
    dP_smem[g0+8] = dp_r0; dP_smem[g0+9] = dp_r1;
    dP_smem[g1+8] = dp_r2; dP_smem[g1+9] = dp_r3;
}

} // namespace detail

extern "C" __global__ void flash_attn_bwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ Q,    // [BH, Sq, D]
    const __nv_bfloat16 *__restrict__ K,    // [BH, Sk, D]
    const __nv_bfloat16 *__restrict__ V_T,  // [BH, D, Sk]
    const __nv_bfloat16 *__restrict__ dO,   // [BH, Sq, D]
    __nv_bfloat16       *__restrict__ dQ,   // [BH, Sq, D]
    __nv_bfloat16       *__restrict__ dK,   // [BH, Sk, D]   atomic-add
    __nv_bfloat16       *__restrict__ dV,   // [BH, Sk, D]   atomic-add
    int32_t Sq, int32_t Sk, float scale)
{
    using namespace detail;

    const int q_block = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
    const int groupID = tid >> 2, tidg = tid & 3;

    const int Q_off  = bh * Sq * D_TILE + q_block * M_TILE * D_TILE;
    const int K_off  = bh * Sk * D_TILE;
    const int VT_off = bh * D_TILE * Sk;
    const int dO_off = Q_off, dQ_off = Q_off, dK_off = K_off, dV_off = K_off;

    __shared__ __nv_bfloat16 Q_smem[M_TILE * D_TILE];
    __shared__ __nv_bfloat16 K_smem[BLOCK_K * D_TILE];
    __shared__ __nv_bfloat16 VT_smem[D_TILE * BLOCK_K];
    __shared__ __nv_bfloat16 dO_smem[M_TILE * D_TILE];
    __shared__ float scores_smem[M_TILE * BLOCK_K];
    __shared__ float P_smem[M_TILE * BLOCK_K];
    __shared__ float dP_smem[M_TILE * BLOCK_K];
    __shared__ float dS_smem[M_TILE * BLOCK_K];
    __shared__ float m_smem[M_TILE];
    __shared__ float l_smem[M_TILE];

    // dQ register accumulator: 4 fp32 left + 4 fp32 right per thread.
    float dQ_l0=0.f, dQ_l1=0.f, dQ_l2=0.f, dQ_l3=0.f, dQ_r0=0.f, dQ_r1=0.f, dQ_r2=0.f, dQ_r3=0.f;

    // ---------------- Stage Q (resident) ----------------
    {
        const int row = tid >> 1, col = (tid & 1) << 3;
        stage16(Q_smem, row * D_TILE + col, Q, Q_off + row * D_TILE + col);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" :: );
    __syncthreads();

    if (tid < M_TILE) { m_smem[tid] = -INFINITY; l_smem[tid] = 0.f; }
    __syncthreads();

    // =================== PHASE 1: forward softmax sweep ===================
    for (int kk = 0; kk < Sk; kk += BLOCK_K) {
        {
            const int row = tid >> 1, col = (tid & 1) << 3;
            stage16(K_smem, row * D_TILE + col, K, K_off + (kk + row) * D_TILE + col);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" :: );
        __syncthreads();

        recompute_scores(Q_smem, K_smem, scores_smem, groupID, tidg, scale);
        __syncthreads();

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
            float sum_block = 0.f;
            for (int c = 0; c < BLOCK_K; ++c) sum_block += expf(scores_smem[r * BLOCK_K + c] - m_new);
            m_smem[r] = m_new;
            l_smem[r] = l_smem[r] * alpha + sum_block;
        }
        __syncthreads();
    }

    // Stage dO once (resident across both phase 2 sweeps).
    {
        const int row = tid >> 1, col = (tid & 1) << 3;
        stage16(dO_smem, row * D_TILE + col, dO, dO_off + row * D_TILE + col);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" :: );

    // dS_smem[0..M_TILE) is the row_dot accumulator during sweep 1.
    if (tid < M_TILE) dS_smem[tid] = 0.f;
    __syncthreads();

    // ============== PHASE 2 SWEEP 1: full-row row_dot accumulation ==============
    for (int kk = 0; kk < Sk; kk += BLOCK_K) {
        stage_K_VT(K_smem, VT_smem, K, V_T, K_off, VT_off, kk, Sk, tid);
        recompute_P_dP(Q_smem, K_smem, VT_smem, dO_smem,
                       scores_smem, P_smem, dP_smem,
                       m_smem, l_smem, groupID, tidg, tid, scale);
        __syncthreads();

        if (tid < M_TILE) {
            const int r = tid;
            float rd = 0.f;
            for (int c = 0; c < BLOCK_K; ++c) rd += dP_smem[r * BLOCK_K + c] * P_smem[r * BLOCK_K + c];
            dS_smem[r] += rd;
        }
        __syncthreads();
    }

    // Cache row_dot[r] into per-lane register; lane r in [0, M_TILE) owns row r.
    float row_dot_reg = (tid < M_TILE) ? dS_smem[tid] : 0.f;
    __syncthreads();

    // ============== PHASE 2 SWEEP 2: dS, dV, dK, dQ ==============
    for (int kk = 0; kk < Sk; kk += BLOCK_K) {
        stage_K_VT(K_smem, VT_smem, K, V_T, K_off, VT_off, kk, Sk, tid);
        recompute_P_dP(Q_smem, K_smem, VT_smem, dO_smem,
                       scores_smem, P_smem, dP_smem,
                       m_smem, l_smem, groupID, tidg, tid, scale);
        __syncthreads();

        // dS[r, c] = (dP[r, c] - row_dot[r]) * P[r, c] * scale
        if (tid < M_TILE) {
            const int r = tid;
            for (int c = 0; c < BLOCK_K; ++c)
                dS_smem[r * BLOCK_K + c] = (dP_smem[r * BLOCK_K + c] - row_dot_reg) * P_smem[r * BLOCK_K + c] * scale;
        }
        __syncthreads();

        // dV[K, :] += P^T @ dO_block, dK[K, :] += dS^T @ Q (atomicAdd, 8 elems/thread).
        for (int idx = tid; idx < BLOCK_K * D_TILE; idx += 32) {
            const int n = idx / D_TILE, d = idx % D_TILE;
            float acc_v = 0.f, acc_k = 0.f;
            for (int q = 0; q < M_TILE; ++q) {
                const float dOqd = __bfloat162float(dO_smem[q * D_TILE + d]);
                const float Qqd  = __bfloat162float(Q_smem [q * D_TILE + d]);
                acc_v += P_smem [q * BLOCK_K + n] * dOqd;
                acc_k += dS_smem[q * BLOCK_K + n] * Qqd;
            }
            atomicAdd(&dV[dV_off + (kk + n) * D_TILE + d], __float2bfloat16(acc_v));
            atomicAdd(&dK[dK_off + (kk + n) * D_TILE + d], __float2bfloat16(acc_k));
        }

        // dQ_block += dS @ K_block (MMA, FP32->BF16 on the fly).
        const int a_row0 = groupID, a_row1 = groupID + 8;
        auto pack_ds = [&](int row, int col_lo) {
            return pack_bf16_pair(__float2bfloat16(dS_smem[row * BLOCK_K + col_lo + 0]),
                                  __float2bfloat16(dS_smem[row * BLOCK_K + col_lo + 1]));
        };
        const uint32_t dsa0 = pack_ds(a_row0, tidg * 2 + 0);
        const uint32_t dsa1 = pack_ds(a_row1, tidg * 2 + 0);
        const uint32_t dsa2 = pack_ds(a_row0, tidg * 2 + 8);
        const uint32_t dsa3 = pack_ds(a_row1, tidg * 2 + 8);

        auto pack_k = [&](int n, int k_lo) {
            return pack_bf16_pair(K_smem[k_lo * D_TILE + n], K_smem[(k_lo + 1) * D_TILE + n]);
        };
        const uint32_t kb_l0 = pack_k(groupID,     tidg * 2 + 0);
        const uint32_t kb_l1 = pack_k(groupID,     tidg * 2 + 8);
        const uint32_t kb_r0 = pack_k(groupID + 8, tidg * 2 + 0);
        const uint32_t kb_r1 = pack_k(groupID + 8, tidg * 2 + 8);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(dQ_l0), "+f"(dQ_l1), "+f"(dQ_l2), "+f"(dQ_l3)
            : "r"(dsa0), "r"(dsa1), "r"(dsa2), "r"(dsa3), "r"(kb_l0), "r"(kb_l1));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(dQ_r0), "+f"(dQ_r1), "+f"(dQ_r2), "+f"(dQ_r3)
            : "r"(dsa0), "r"(dsa1), "r"(dsa2), "r"(dsa3), "r"(kb_r0), "r"(kb_r1));

        __syncthreads();
    }

    // ---------- Write dQ from registers to gmem ----------
    __nv_bfloat16 *dQ_block = dQ + dQ_off;
    const int row_g = groupID, row_g8 = groupID + 8;
    dQ_block[row_g  * D_TILE + tidg*2 + 0] = __float2bfloat16(dQ_l0);
    dQ_block[row_g  * D_TILE + tidg*2 + 1] = __float2bfloat16(dQ_l1);
    dQ_block[row_g8 * D_TILE + tidg*2 + 0] = __float2bfloat16(dQ_l2);
    dQ_block[row_g8 * D_TILE + tidg*2 + 1] = __float2bfloat16(dQ_l3);
    dQ_block[row_g  * D_TILE + tidg*2 + 8] = __float2bfloat16(dQ_r0);
    dQ_block[row_g  * D_TILE + tidg*2 + 9] = __float2bfloat16(dQ_r1);
    dQ_block[row_g8 * D_TILE + tidg*2 + 8] = __float2bfloat16(dQ_r2);
    dQ_block[row_g8 * D_TILE + tidg*2 + 9] = __float2bfloat16(dQ_r3);
}

inline void flash_attn_bwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V_T,
    const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dQ, __nv_bfloat16 *d_dK, __nv_bfloat16 *d_dV,
    int BH, int Sq, int Sk, float scale, cudaStream_t stream)
{
    dim3 grid((Sq + detail::M_TILE - 1) / detail::M_TILE, BH, 1);
    dim3 block(32, 1, 1);
    flash_attn_bwd_bf16_sm80_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V_T, d_dO, d_dQ, d_dK, d_dV, Sq, Sk, scale);
}

} // namespace flash_attn_bwd
