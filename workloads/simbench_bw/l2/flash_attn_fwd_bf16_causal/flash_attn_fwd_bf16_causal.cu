// slug: flash_attn_fwd_bf16_causal
// tier: P
// arch_gate: sm_90a sm_120
//
// Causal scaled-dot-product attention forward. BF16 Q / K / V inputs,
// FP32 softmax + accumulate, BF16 output. Matches
// `references/attention.py::build_reference` for slug
// "flash_attn_fwd_bf16_causal" (causal=True).
//
// Scalar BF16->FP32->BF16 arithmetic — same pattern as R13
// `moe_grouped_gemm_bf16`. The Hopper FA-3 / wgmma / warp-specialized
// producer-consumer path lands as a later-round upgrade; R15
// targets correctness parity with the FP32 Python reference under the
// `(BF16, tree_sum, attention)` tolerance.
//
// Grid: one block per (query_row, head, batch). Block: head_dim
// threads, each owning one output dimension.  Row-wise softmax is
// driven serially by thread 0 into shared memory so the lane-0 scan
// matches the reference's row-major softmax.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void flash_attn_fwd_bf16_causal_kernel(
    const __nv_bfloat16 *__restrict__ q,   // [batch, heads, seq, head_dim]
    const __nv_bfloat16 *__restrict__ k,   // [batch, heads, seq, head_dim]
    const __nv_bfloat16 *__restrict__ v,   // [batch, heads, seq, head_dim]
    __nv_bfloat16 *__restrict__ out,       // [batch, heads, seq, head_dim]
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float scale)
{
    const int qi = blockIdx.x;      // query row
    const int h  = blockIdx.y;      // head
    const int ba = blockIdx.z;      // batch
    const int d  = threadIdx.x;     // output dim
    if (d >= head_dim) return;

    extern __shared__ float shared_scores[];  // [seq]

    // Pass 1 (lane 0): compute scores[j] = <q[qi, :], k[j, :]> * scale,
    // apply causal mask (j > qi -> -inf), then row-wise softmax with the
    // standard row-max stability trick.
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < seq; ++j) {
            if (j > qi) {
                shared_scores[j] = -FLT_MAX;
                continue;
            }
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = __bfloat162float(q[(((ba * heads) + h) * seq + qi) * head_dim + dk]);
                float kv = __bfloat162float(k[(((ba * heads) + h) * seq +  j) * head_dim + dk]);
                s += qv * kv;
            }
            s *= scale;
            shared_scores[j] = s;
            if (s > m) m = s;
        }
        float z = 0.0f;
        for (int j = 0; j < seq; ++j) {
            if (shared_scores[j] == -FLT_MAX) {
                shared_scores[j] = 0.0f;
            } else {
                shared_scores[j] = __expf(shared_scores[j] - m);
                z += shared_scores[j];
            }
        }
        if (z > 0.0f) {
            for (int j = 0; j < seq; ++j) shared_scores[j] /= z;
        }
    }
    __syncthreads();

    // Pass 2: out[ba, h, qi, d] = sum_j probs[j] * v[ba, h, j, d].
    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
        float vv = __bfloat162float(v[(((ba * heads) + h) * seq + j) * head_dim + d]);
        acc += shared_scores[j] * vv;
    }
    out[(((ba * heads) + h) * seq + qi) * head_dim + d] = __float2bfloat16(acc);
}

extern "C" void flash_attn_fwd_bf16_causal_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid(seq, heads, batch);
    dim3 block(head_dim);
    size_t shmem = (size_t)seq * sizeof(float);
    flash_attn_fwd_bf16_causal_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_out, heads, seq, head_dim, scale);
}
