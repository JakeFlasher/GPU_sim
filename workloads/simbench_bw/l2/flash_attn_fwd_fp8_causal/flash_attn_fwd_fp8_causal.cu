// slug: flash_attn_fwd_fp8_causal
// tier: P
// arch_gate: sm_90a sm_120
//
// Causal scaled-dot-product attention forward in FP8 E4M3.
// Q / K / V / output are stored as __nv_fp8_e4m3; softmax + accumulate
// are in FP32. Matches `references/attention.py::build_reference` for
// slug "flash_attn_fwd_fp8_causal" (causal=True) under the
// `(FP8_E4M3, tree_sum, attention)` tolerance (rtol/atol=0.05, max_ulp=8).
//
// Same scalar two-pass structure as R15 `flash_attn_fwd_bf16_causal`.
// FP8 E4M3 has ~3 bits of mantissa precision; with correctness-shape
// Q/K/V drawn from [-0.5, 0.5) and scale = 1/sqrt(head_dim)=1/8, per-row
// softmax scores end up well within E4M3's ~448 absolute max, and the
// output stays in [-0.5, 0.5) which quantizes cleanly.
//
// Grid: one block per (query_row, head, batch). Block: head_dim threads,
// each owning one output dim. Thread 0 serially computes the row's
// scores + softmax into shared memory (causal mask at j > qi); then all
// threads cooperate on the output accumulation.

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void flash_attn_fwd_fp8_causal_kernel(
    const __nv_fp8_e4m3 *__restrict__ q,   // [batch, heads, seq, head_dim]
    const __nv_fp8_e4m3 *__restrict__ k,   // [batch, heads, seq, head_dim]
    const __nv_fp8_e4m3 *__restrict__ v,   // [batch, heads, seq, head_dim]
    __nv_fp8_e4m3 *__restrict__ out,       // [batch, heads, seq, head_dim]
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

    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < seq; ++j) {
            if (j > qi) {
                shared_scores[j] = -FLT_MAX;
                continue;
            }
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = (float)q[(((ba * heads) + h) * seq + qi) * head_dim + dk];
                float kv = (float)k[(((ba * heads) + h) * seq +  j) * head_dim + dk];
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

    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
        float vv = (float)v[(((ba * heads) + h) * seq + j) * head_dim + d];
        acc += shared_scores[j] * vv;
    }
    out[(((ba * heads) + h) * seq + qi) * head_dim + d] = (__nv_fp8_e4m3)acc;
}

extern "C" void flash_attn_fwd_fp8_causal_launch(
    const __nv_fp8_e4m3 *d_q,
    const __nv_fp8_e4m3 *d_k,
    const __nv_fp8_e4m3 *d_v,
    __nv_fp8_e4m3 *d_out,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid(seq, heads, batch);
    dim3 block(head_dim);
    size_t shmem = (size_t)seq * sizeof(float);
    flash_attn_fwd_fp8_causal_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_out, heads, seq, head_dim, scale);
}
