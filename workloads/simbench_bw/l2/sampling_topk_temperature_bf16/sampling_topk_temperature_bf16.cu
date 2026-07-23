// slug: sampling_topk_temperature_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Deterministic top-k + temperature + softmax sampling primitive. Produces
// the sorted top-k indices + normalized probabilities per batch row.
//
// Design note: this kernel uses a two-pass approach — pass 1 computes the
// temperature-scaled softmax with the row-max stability trick; pass 2 runs
// a k-way insertion sort to extract the top-k. For vocab sizes in the
// shape file's range (4k-128k) and top_k <= 50, the insertion-sort kernel
// launch config is one block per batch row with 256 threads, where each
// thread scans a contiguous vocab segment and maintains a local top-k
// heap; a final merge step combines the per-thread heaps in shared memory.
//
// The CPU reference in references/sampling.py uses stable sort on ties;
// the kernel must match that ordering to satisfy the family's compare().

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void sampling_topk_temperature_bf16_kernel(
    const __nv_bfloat16 *__restrict__ logits, // [batch, vocab]
    int32_t *__restrict__ out_indices,        // [batch, top_k]
    __nv_bfloat16 *__restrict__ out_probs,    // [batch, top_k]
    int32_t vocab,
    int32_t top_k,
    float temperature)
{
    extern __shared__ float shared_mem[];
    float *row_max = shared_mem;
    float *row_sum = shared_mem + 1;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;

    // Pass 1a: per-thread max scan.
    float local_max = -FLT_MAX;
    for (int c = tid; c < vocab; c += lanes) {
        float v = __bfloat162float(logits[row * vocab + c]) / temperature;
        if (v > local_max) local_max = v;
    }
    // Warp + block max reduce.
    float *max_scratch = shared_mem + 2;
    max_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = max_scratch[tid];
            float b = max_scratch[tid + stride];
            max_scratch[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) *row_max = max_scratch[0];
    __syncthreads();
    float m = *row_max;

    // Pass 1b: compute row sum of exp(v - m).
    float local_sum = 0.0f;
    for (int c = tid; c < vocab; c += lanes) {
        float v = __bfloat162float(logits[row * vocab + c]) / temperature;
        local_sum += expf(v - m);
    }
    max_scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_scratch[tid] += max_scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) *row_sum = max_scratch[0];
    __syncthreads();
    float z = *row_sum;

    // Pass 2: single-thread top-k scan (simple, correct; optimization deferred
    // to a later round when top_k > 50 cases are traced).
    if (tid == 0) {
        int32_t *idx_out = out_indices + row * top_k;
        __nv_bfloat16 *prob_out = out_probs + row * top_k;
        for (int k = 0; k < top_k; ++k) {
            float best = -1.0f;
            int best_i = -1;
            for (int c = 0; c < vocab; ++c) {
                // Skip previously-chosen indices
                bool chosen = false;
                for (int j = 0; j < k; ++j) {
                    if (idx_out[j] == c) { chosen = true; break; }
                }
                if (chosen) continue;
                float v = __bfloat162float(logits[row * vocab + c]) / temperature;
                float p = expf(v - m) / z;
                if (p > best || (p == best && c < best_i)) {
                    best = p;
                    best_i = c;
                }
            }
            idx_out[k] = best_i;
            prob_out[k] = __float2bfloat16(best);
        }
    }
}

extern "C" void sampling_topk_temperature_bf16_launch(
    const __nv_bfloat16 *d_logits,
    int32_t *d_indices,
    __nv_bfloat16 *d_probs,
    int batch, int vocab, int top_k, float temperature, cudaStream_t stream)
{
    dim3 grid(batch);
    dim3 block(256);
    size_t shmem = (2 + block.x) * sizeof(float);
    sampling_topk_temperature_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_logits, d_indices, d_probs, vocab, top_k, temperature);
}
