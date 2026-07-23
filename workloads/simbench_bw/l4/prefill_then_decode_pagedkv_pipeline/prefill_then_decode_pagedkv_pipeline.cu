// slug: prefill_then_decode_pagedkv_pipeline
// tier: P
// arch_gate: sm_90a sm_120
//
// Prefill-then-decode pipeline reading from a paged KV cache. The
// kernel consumes the L4 boundary state declared in
// `references/paged_attention.py::_prefill_then_decode_boundary_state`:
//
//   - kv_data_blocks    [num_blocks, block_size, kv_heads, head_dim]  (seed + 500)
//   - block_table       [batch, max_blocks_per_sequence]              sequential per-sequence
//   - q (decode query)  [batch, q_heads, head_dim]                    (seed + 0)
//
// Output: `[batch, q_heads, head_dim]` = SDPA(q, K, V) under GQA
// (kv_h = q_h % kv_heads), where K and V are reconstructed from
// kv_data_blocks by walking the block_table per
// `references/paged_attention.py::_prefill_then_decode_pipeline_reference`.
// (The reference aliases K and V to the same paged tensor; the kernel
// does the same so byte-level coherence is preserved.)
//
// BF16 storage on q + KV + output; FP32 internal compute. Tolerance:
// (BF16, tree_sum, paged_attention) = rtol = atol = 0.01.
//
// Grid: one block per (batch, q_head). Block: `head_dim` threads;
// thread d accumulates out[b, h, d]. Shared memory carries the
// `[context]` softmax scratch.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>


extern "C" __global__ void prefill_then_decode_pagedkv_pipeline_kernel(
    const __nv_bfloat16 *__restrict__ q,                // [batch, q_heads, head_dim]
    const __nv_bfloat16 *__restrict__ kv_data_blocks,   // [num_blocks, block_size, kv_heads, head_dim]
    const int32_t       *__restrict__ block_table,      // [batch, max_blocks_per_sequence]
    __nv_bfloat16 *__restrict__ out,                     // [batch, q_heads, head_dim]
    int32_t q_heads,
    int32_t kv_heads,
    int32_t context,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks_per_sequence,
    float   scale)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const int kv_h = h % kv_heads;

    extern __shared__ float smem[];
    float *scores = smem;   // [context]

    // ---- Pass 1: thread 0 computes scores + stable softmax ----
    if (d == 0) {
        float m = -FLT_MAX;
        for (int p = 0; p < context; ++p) {
            const int block_idx = block_table[b * max_blocks_per_sequence + (p / block_size)];
            const int offset = p % block_size;
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                const float qv = __bfloat162float(q[((b * q_heads) + h) * head_dim + dk]);
                const size_t kv_pos = ((((size_t)block_idx * block_size + offset) * kv_heads) + kv_h) * head_dim + dk;
                const float kv = __bfloat162float(kv_data_blocks[kv_pos]);
                s += qv * kv;
            }
            s *= scale;
            scores[p] = s;
            if (s > m) m = s;
        }
        float z = 0.0f;
        for (int p = 0; p < context; ++p) {
            scores[p] = __expf(scores[p] - m);
            z += scores[p];
        }
        for (int p = 0; p < context; ++p) scores[p] /= z;
    }
    __syncthreads();

    // ---- Pass 2: V-weighted accumulate; thread d owns out[b, h, d] ----
    float acc = 0.0f;
    for (int p = 0; p < context; ++p) {
        const int block_idx = block_table[b * max_blocks_per_sequence + (p / block_size)];
        const int offset = p % block_size;
        const size_t kv_pos = ((((size_t)block_idx * block_size + offset) * kv_heads) + kv_h) * head_dim + d;
        const float vv = __bfloat162float(kv_data_blocks[kv_pos]);
        acc += scores[p] * vv;
    }
    out[((b * q_heads) + h) * head_dim + d] = __float2bfloat16(acc);
}


extern "C" void prefill_then_decode_pagedkv_pipeline_launch(
    const __nv_bfloat16 *d_q,
    const __nv_bfloat16 *d_kv_data_blocks,
    const int32_t       *d_block_table,
    __nv_bfloat16 *d_out,
    int batch, int q_heads, int kv_heads,
    int context, int head_dim, int block_size,
    int max_blocks_per_sequence,
    float scale, cudaStream_t stream)
{
    dim3 grid(batch, q_heads);
    dim3 block(head_dim);
    size_t shmem = (size_t)context * sizeof(float);
    prefill_then_decode_pagedkv_pipeline_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_kv_data_blocks, d_block_table, d_out,
        q_heads, kv_heads, context, head_dim, block_size,
        max_blocks_per_sequence, scale);
}
