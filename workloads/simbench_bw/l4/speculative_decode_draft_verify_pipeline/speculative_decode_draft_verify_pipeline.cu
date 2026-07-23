// slug: speculative_decode_draft_verify_pipeline
// tier: P
// arch_gate: sm_90a sm_120
//
// Speculative decoding draft-verify pipeline. The kernel consumes the
// draft model's logits (boundary-state offset = seed + 700 in the
// host driver) and the verifier's target-model logits (offset = seed
// + 900) and produces the per-position rejection-sampling-weighted
// target probabilities:
//
//   draft_probs[v]  = softmax(draft_logits)[v]
//   target_probs[v] = softmax(target_logits)[v]
//   accept[v]       = min(1, target_probs[v] / max(draft_probs[v], eps))
//   out[v]          = target_probs[v] * accept[v]
//
// Output shape matches references/attention.py::build_reference for
// `speculative_decode_draft_verify_pipeline`: [batch, draft_length,
// vocab]. BF16 storage on the logits + output, FP32 softmax + accept
// math internally. rtol = atol = 0.025 under (BF16, atomic_add,
// attention).
//
// Grid: one block per (batch, draft_length); threads stride over the
// vocab dim. Shared memory carries 4 block-scope reductions (draft
// max + sum, target max + sum) plus a 2 * blockDim.x scratch for
// the per-thread partials.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>


extern "C" __global__ void speculative_decode_draft_verify_pipeline_kernel(
    const __nv_bfloat16 *__restrict__ draft_logits,   // [batch, dlen, vocab]
    const __nv_bfloat16 *__restrict__ target_logits,  // [batch, dlen, vocab]
    __nv_bfloat16 *__restrict__ out,                   // [batch, dlen, vocab]
    int32_t batch,
    int32_t dlen,
    int32_t vocab)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (b >= batch || s >= dlen) return;

    const __nv_bfloat16 *d_row = draft_logits  + (size_t)(b * dlen + s) * vocab;
    const __nv_bfloat16 *t_row = target_logits + (size_t)(b * dlen + s) * vocab;
    __nv_bfloat16 *o_row       = out           + (size_t)(b * dlen + s) * vocab;

    extern __shared__ float smem[];
    float *block_scratch = smem;                  // [4]
    float *thread_max = smem + 4;                 // [nthreads]
    float *thread_sum = smem + 4 + nthreads;      // [nthreads]

    // ---- Stable softmax pass over draft_logits ----
    float local_max = -FLT_MAX;
    for (int v = tid; v < vocab; v += nthreads) {
        float l = __bfloat162float(d_row[v]);
        if (l > local_max) local_max = l;
    }
    thread_max[tid] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int t = 0; t < nthreads; ++t) if (thread_max[t] > m) m = thread_max[t];
        block_scratch[0] = m;
    }
    __syncthreads();
    const float draft_m = block_scratch[0];

    float local_sum = 0.0f;
    for (int v = tid; v < vocab; v += nthreads) {
        local_sum += __expf(__bfloat162float(d_row[v]) - draft_m);
    }
    thread_sum[tid] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float z = 0.0f;
        for (int t = 0; t < nthreads; ++t) z += thread_sum[t];
        block_scratch[1] = z;
    }
    __syncthreads();
    const float draft_z = block_scratch[1];

    // ---- Stable softmax pass over target_logits ----
    local_max = -FLT_MAX;
    for (int v = tid; v < vocab; v += nthreads) {
        float l = __bfloat162float(t_row[v]);
        if (l > local_max) local_max = l;
    }
    thread_max[tid] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int t = 0; t < nthreads; ++t) if (thread_max[t] > m) m = thread_max[t];
        block_scratch[2] = m;
    }
    __syncthreads();
    const float target_m = block_scratch[2];

    local_sum = 0.0f;
    for (int v = tid; v < vocab; v += nthreads) {
        local_sum += __expf(__bfloat162float(t_row[v]) - target_m);
    }
    thread_sum[tid] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float z = 0.0f;
        for (int t = 0; t < nthreads; ++t) z += thread_sum[t];
        block_scratch[3] = z;
    }
    __syncthreads();
    const float target_z = block_scratch[3];

    // ---- Per-vocab acceptance + output ----
    const float eps = 1e-12f;
    for (int v = tid; v < vocab; v += nthreads) {
        float draft_prob  = __expf(__bfloat162float(d_row[v]) - draft_m)  / draft_z;
        float target_prob = __expf(__bfloat162float(t_row[v]) - target_m) / target_z;
        float a = fminf(1.0f, target_prob / fmaxf(draft_prob, eps));
        o_row[v] = __float2bfloat16(target_prob * a);
    }
}


extern "C" void speculative_decode_draft_verify_pipeline_launch(
    const __nv_bfloat16 *d_draft_logits,
    const __nv_bfloat16 *d_target_logits,
    __nv_bfloat16 *d_out,
    int batch, int dlen, int vocab,
    cudaStream_t stream)
{
    const int threads = 128;
    dim3 grid(batch, dlen);
    dim3 block(threads);
    size_t shmem = (4 + 2 * threads) * sizeof(float);
    speculative_decode_draft_verify_pipeline_kernel<<<grid, block, shmem, stream>>>(
        d_draft_logits, d_target_logits, d_out, batch, dlen, vocab);
}
