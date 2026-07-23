// slug: flash_attn_bwd_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Scaled-dot-product attention BACKWARD pass (non-causal): given Q, K,
// V in BF16 and an upstream output gradient dO in BF16, compute dQ
// (gradient of the attention output w.r.t. Q). Output shape matches Q.
//
// Non-causal: the row's slug does not contain "causal" so the
// references/attention.py family resolver picks the non-causal branch
// (see attention.py:95 `causal = "causal" in slug and ...`). The
// causal-masked backward variant is `flash_attn_bwd_bf16_causal` when
// that row lands in a later milestone.
//
// Math (per (batch, head) block; scale = 1/sqrt(head_dim)):
//   P[i][j]   = softmax_row_i(Q @ K^T)[j]                (FP32 weights)
//   dP[i][j]  = <dO[i], V[j]>                             (FP32 gradient to P)
//   row_dot[i] = sum_j P[i][j] * dP[i][j]                 (softmax-backward scalar)
//   ds[i][j]  = P[i][j] * (dP[i][j] - row_dot[i])         (FP32 score gradient)
//   dQ[i][d]  = (sum_j ds[i][j] * K[j][d]) * scale        (FP32, cast back to BF16)
//
// Matches the Round-17-extended `references/attention.py::_compute_dq`
// path for slug "flash_attn_bwd_bf16". The row is non-causal: the
// reference's causal-branch selector at attention.py:95 checks
// `"causal" in slug`, and `"causal"` is not a substring of
// `"flash_attn_bwd_bf16"`, so the reference runs non-causal. The
// tolerance is `(BF16, atomic_add, attention)` (rtol/atol=0.025,
// max_ulp=128) — the row is `deterministic="ND"` so
// `run_suite.py::_infer_reduction_kind` routes it through the widened
// atomic-add band.
//
// Grid: one block per (query_row, head, batch). Block: head_dim threads,
// each owning one dimension of dQ[qi]. Thread 0 serially computes P[qi,:],
// dP[qi,:], the row_dot scalar, and ds[qi,:] into shared memory; then
// all threads cooperatively compute dQ[qi, d] = (sum_j ds[j] * K[j,d]) * scale.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void flash_attn_bwd_bf16_kernel(
    const __nv_bfloat16 *__restrict__ q,   // [batch, heads, seq, head_dim]
    const __nv_bfloat16 *__restrict__ k,   // [batch, heads, seq, head_dim]
    const __nv_bfloat16 *__restrict__ v,   // [batch, heads, seq, head_dim]
    const __nv_bfloat16 *__restrict__ dO,  // [batch, heads, seq, head_dim]
    __nv_bfloat16 *__restrict__ dq,        // [batch, heads, seq, head_dim]
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float scale)
{
    const int qi = blockIdx.x;
    const int h  = blockIdx.y;
    const int ba = blockIdx.z;
    const int d  = threadIdx.x;
    if (d >= head_dim) return;

    // shared layout: [ P[seq] | ds[seq] ]  both FP32
    extern __shared__ float shared_buf[];
    float *shared_P  = shared_buf;
    float *shared_ds = shared_buf + seq;

    if (d == 0) {
        // Pass 1a: scores[j] = <Q[qi], K[j]> * scale (non-causal, all j).
        float m = -FLT_MAX;
        for (int j = 0; j < seq; ++j) {
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = __bfloat162float(q[(((ba * heads) + h) * seq + qi) * head_dim + dk]);
                float kv = __bfloat162float(k[(((ba * heads) + h) * seq +  j) * head_dim + dk]);
                s += qv * kv;
            }
            shared_P[j] = s * scale;
            if (shared_P[j] > m) m = shared_P[j];
        }
        // Pass 1b: softmax-normalize P[qi, :] row-stably.
        float z = 0.0f;
        for (int j = 0; j < seq; ++j) {
            shared_P[j] = __expf(shared_P[j] - m);
            z += shared_P[j];
        }
        if (z > 0.0f) {
            for (int j = 0; j < seq; ++j) shared_P[j] /= z;
        }
        // Pass 2: dP[j] = <dO[qi], V[j]>; row_dot = sum_j P[j] * dP[j];
        //         ds[j] = P[j] * (dP[j] - row_dot).
        // dP is not stored globally — we reuse the row_dot scalar and
        // inline-fuse dP into ds during the second scan.
        // First, compute dP[j] into shared_ds temporarily and row_dot.
        float row_dot = 0.0f;
        for (int j = 0; j < seq; ++j) {
            float dp = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float dov = __bfloat162float(dO[(((ba * heads) + h) * seq + qi) * head_dim + dk]);
                float vv  = __bfloat162float(v[(((ba * heads) + h) * seq +  j) * head_dim + dk]);
                dp += dov * vv;
            }
            shared_ds[j] = dp;   // temporarily holds dP[j]
            row_dot += shared_P[j] * dp;
        }
        // Finalize ds[j] = P[j] * (dP[j] - row_dot).
        for (int j = 0; j < seq; ++j) {
            shared_ds[j] = shared_P[j] * (shared_ds[j] - row_dot);
        }
    }
    __syncthreads();

    // Pass 3 (parallel across d): dQ[qi, d] = (sum_j ds[j] * K[j, d]) * scale.
    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
        float kv = __bfloat162float(k[(((ba * heads) + h) * seq + j) * head_dim + d]);
        acc += shared_ds[j] * kv;
    }
    acc *= scale;
    dq[(((ba * heads) + h) * seq + qi) * head_dim + d] = __float2bfloat16(acc);
}

extern "C" void flash_attn_bwd_bf16_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dq,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid(seq, heads, batch);
    dim3 block(head_dim);
    // shared: P[seq] + ds[seq]
    size_t shmem = (size_t)(2 * seq) * sizeof(float);
    flash_attn_bwd_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_dO, d_dq, heads, seq, head_dim, scale);
}
