// slug: flash_attn_fwd_bf16_noncausal
// tier: P
// arch_gate: sm_90a sm_120
//
// Non-causal scaled-dot-product attention forward. Identical to the
// causal variant minus the `if (j > qi)` mask. Matches
// references/attention.py::_sdpa(..., causal=False) for this slug
// under (BF16, tree_sum, attention) tolerance.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void flash_attn_fwd_bf16_noncausal_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    __nv_bfloat16 *__restrict__ out,
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

    extern __shared__ float shared_scores[];

    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < seq; ++j) {
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
            shared_scores[j] = __expf(shared_scores[j] - m);
            z += shared_scores[j];
        }
        if (z > 0.0f) {
            for (int j = 0; j < seq; ++j) shared_scores[j] /= z;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
        float vv = __bfloat162float(v[(((ba * heads) + h) * seq + j) * head_dim + d]);
        acc += shared_scores[j] * vv;
    }
    out[(((ba * heads) + h) * seq + qi) * head_dim + d] = __float2bfloat16(acc);
}

extern "C" void flash_attn_fwd_bf16_noncausal_launch(
    const __nv_bfloat16 *d_q, const __nv_bfloat16 *d_k, const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid(seq, heads, batch);
    dim3 block(head_dim);
    size_t shmem = (size_t)seq * sizeof(float);
    flash_attn_fwd_bf16_noncausal_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_out, heads, seq, head_dim, scale);
}
