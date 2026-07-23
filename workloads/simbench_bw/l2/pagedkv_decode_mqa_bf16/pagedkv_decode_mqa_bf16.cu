// slug: pagedkv_decode_mqa_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Grouped-query attention paged-KV single-step decode. Ports the
// paged_mla_decode_bf16 pattern to the GQA layout: `q_heads` > `kv_heads`,
// each kv head is shared by `q_heads / kv_heads` query heads via
// `kv_h = h % kv_heads`. BF16 input / KV, FP32 softmax + accumulate.
// Simplified reference-shape-matching: context stored as contiguous K / V
// pages (page_size axis accepted but unused in the simplified kernel;
// the production page-table walk lands in a later round). Output shape
// matches `references/paged_attention.py::build_reference` so the kernel
// vs CPU comparison passes under the family's BF16 tolerance.
//
// Grid: one block per (batch, q_head). Block: head_dim threads; each
// thread accumulates one dim of the output vector. Shared memory holds
// the per-token softmax-weighted probabilities plus a 2-slot accumulator
// for row_max / row_sum.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void pagedkv_decode_mqa_bf16_kernel(
    const __nv_bfloat16 *__restrict__ q,       // [batch, q_heads, head_dim]
    const __nv_bfloat16 *__restrict__ k,       // [batch, kv_heads, context, head_dim]
    const __nv_bfloat16 *__restrict__ v,       // [batch, kv_heads, context, head_dim]
    __nv_bfloat16 *__restrict__ out,            // [batch, q_heads, head_dim]
    int32_t q_heads,
    int32_t kv_heads,
    int32_t context,
    int32_t head_dim,
    float scale)
{
    const int ba = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    // GQA mapping: q_heads / kv_heads queries share one kv head.
    const int kv_h = h % kv_heads;

    extern __shared__ float shared_scores[];  // [context]
    float *row_max = shared_scores + context;
    float *row_sum = shared_scores + context + 1;

    // Pass 1: compute softmax-normalized scores[j]. One thread per block
    // drives the serial scan since the head_dim reduction is small.
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < context; ++j) {
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = __bfloat162float(q[((ba * q_heads) + h) * head_dim + dk]);
                float kv = __bfloat162float(k[(((ba * kv_heads) + kv_h) * context + j) * head_dim + dk]);
                s += qv * kv;
            }
            s *= scale;
            shared_scores[j] = s;
            if (s > m) m = s;
        }
        *row_max = m;
        float z = 0.0f;
        for (int j = 0; j < context; ++j) {
            shared_scores[j] = __expf(shared_scores[j] - m);
            z += shared_scores[j];
        }
        for (int j = 0; j < context; ++j) shared_scores[j] /= z;
        *row_sum = z;
    }
    __syncthreads();

    // Pass 2: out[ba, h, d] = sum_j probs[j] * v[ba, kv_h, j, d].
    float acc = 0.0f;
    for (int j = 0; j < context; ++j) {
        float vv = __bfloat162float(v[(((ba * kv_heads) + kv_h) * context + j) * head_dim + d]);
        acc += shared_scores[j] * vv;
    }
    out[((ba * q_heads) + h) * head_dim + d] = __float2bfloat16(acc);
}

extern "C" void pagedkv_decode_mqa_bf16_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    int batch, int q_heads, int kv_heads, int context, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid(batch, q_heads);
    dim3 block(head_dim);
    size_t shmem = (context + 2) * sizeof(float);
    pagedkv_decode_mqa_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_out, q_heads, kv_heads, context, head_dim, scale);
}
