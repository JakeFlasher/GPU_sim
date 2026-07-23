// slug: ragged_attention_prefill_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Variable-length packed-tokens attention prefill (task19). BF16 Q / K / V
// inputs, FP32 softmax + accumulate, BF16 output. Segmentation is real:
// each query row i attends only to keys in its own segment s, where s
// is the index such that `cu_seqlens[s] <= i < cu_seqlens[s+1]`. No
// cross-segment attention. Matches the segmented branch of
// `references/attention.py::build_reference` for slug
// "ragged_attention_prefill_bf16" (non-causal, since "causal" is not
// in the slug name; num_seqs > 1 triggers the segmented path).
//
// Grid: one block per (query_row, head). Block: head_dim threads.
// Thread 0 binary-scans `cu_seqlens` to find the query's segment,
// serially computes per-segment scores + row-wise softmax into shared
// memory, then all threads cooperatively accumulate the output.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void ragged_attention_prefill_bf16_kernel(
    const __nv_bfloat16 *__restrict__ q,       // [heads, total_tokens, head_dim]
    const __nv_bfloat16 *__restrict__ k,       // [heads, total_tokens, head_dim]
    const __nv_bfloat16 *__restrict__ v,       // [heads, total_tokens, head_dim]
    __nv_bfloat16 *__restrict__ out,           // [heads, total_tokens, head_dim]
    const int32_t *__restrict__ cu_seqlens,    // [num_seqs + 1]
    int32_t heads,
    int32_t total_tokens,
    int32_t head_dim,
    int32_t num_seqs,
    float scale)
{
    const int qi = blockIdx.x;      // query row within total_tokens
    const int h  = blockIdx.y;      // head
    const int d  = threadIdx.x;     // output dim
    if (d >= head_dim) return;

    // Find the segment s that contains qi. num_seqs is small (tens),
    // so a linear scan is fine. Produces [seg_lo, seg_hi) bounds.
    int seg_lo = 0;
    int seg_hi = total_tokens;
    for (int s = 0; s < num_seqs; ++s) {
        if (qi >= cu_seqlens[s] && qi < cu_seqlens[s + 1]) {
            seg_lo = cu_seqlens[s];
            seg_hi = cu_seqlens[s + 1];
            break;
        }
    }

    extern __shared__ float shared_scores[];  // [max_seg_len]; here sized to total_tokens for simplicity

    // Pass 1 (lane 0): compute scores[j] = <q[qi, :], k[j, :]> * scale
    // for j in [seg_lo, seg_hi), then row-wise softmax. j outside the
    // segment is never touched.
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = seg_lo; j < seg_hi; ++j) {
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = __bfloat162float(q[(h * total_tokens + qi) * head_dim + dk]);
                float kv = __bfloat162float(k[(h * total_tokens +  j) * head_dim + dk]);
                s += qv * kv;
            }
            s *= scale;
            shared_scores[j - seg_lo] = s;
            if (s > m) m = s;
        }
        float z = 0.0f;
        int seg_len = seg_hi - seg_lo;
        for (int j = 0; j < seg_len; ++j) {
            shared_scores[j] = __expf(shared_scores[j] - m);
            z += shared_scores[j];
        }
        if (z > 0.0f) {
            for (int j = 0; j < seg_len; ++j) shared_scores[j] /= z;
        }
    }
    __syncthreads();

    // Pass 2: out[h, qi, d] = sum_{j in segment} probs[j] * v[h, j, d].
    float acc = 0.0f;
    for (int j = seg_lo; j < seg_hi; ++j) {
        float vv = __bfloat162float(v[(h * total_tokens + j) * head_dim + d]);
        acc += shared_scores[j - seg_lo] * vv;
    }
    out[(h * total_tokens + qi) * head_dim + d] = __float2bfloat16(acc);
}

extern "C" void ragged_attention_prefill_bf16_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v,
    __nv_bfloat16 *d_out,
    const int32_t *d_cu_seqlens,
    int heads, int total_tokens, int head_dim, int num_seqs,
    float scale, cudaStream_t stream)
{
    dim3 grid(total_tokens, heads, 1);
    dim3 block(head_dim);
    // Shared memory sized to the worst-case segment length (<= total_tokens).
    // Per-block we only touch `seg_len` entries, so oversizing is harmless.
    size_t shmem = (size_t)total_tokens * sizeof(float);
    ragged_attention_prefill_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_out, d_cu_seqlens, heads, total_tokens, head_dim, num_seqs, scale);
}
