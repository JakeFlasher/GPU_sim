// slug: moe_layer_residual_norm_pipeline
// tier: P
// arch_gate: sm_90a sm_120
//
// MoE-layer pipeline that fuses the canonical 5-stage L4 boundary
// referenced by `references/grouped_gemm.py::build_reference` for slug
// "moe_layer_residual_norm_pipeline":
//
//   1. Per-token softmax over the [tokens, experts] logits.
//   2. Top-k expert selection with stable lower-index tie-breaking
//      and re-normalized routing weights summing to 1.
//   3. Per-expert linear projection of x[t] through w_experts[eid]
//      ([hidden, intermediate]); accumulated weighted sum across the
//      top-k slots into out[t].
//   4. Residual add `out[t] += x[t]` (hidden == intermediate by row
//      contract; the reference refuses other shapes with ValueError).
//   5. Per-token RMSNorm over the hidden dim with a seeded
//      `gamma[hidden]` multiplier (seed + 200).
//
// BF16 storage on every tensor; FP32 internal compute (this is the row's
// "MIXED" precision band per `tolerances.toml::(MIXED, atomic_add,
// grouped_gemm)` = rtol = atol = 0.125 — the wide band absorbs the
// reduction-ordering nondeterminism declared in `roster.toml`).
//
// Grid: one block per token. Threads stride over `intermediate`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define MOE_LAYER_RMS_EPS 1e-5f


extern "C" __global__ void moe_layer_residual_norm_pipeline_kernel(
    const __nv_bfloat16 *__restrict__ logits,    // [tokens, experts]
    const __nv_bfloat16 *__restrict__ x,         // [tokens, hidden]
    const __nv_bfloat16 *__restrict__ w_experts, // [experts, hidden, intermediate]
    const __nv_bfloat16 *__restrict__ gamma,     // [hidden]
    __nv_bfloat16 *__restrict__ out,             // [tokens, hidden] (== [tokens, intermediate])
    int32_t tokens,
    int32_t experts,
    int32_t top_k,
    int32_t hidden,
    int32_t intermediate)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= tokens) return;

    // Shared-memory layout (single dynamic block):
    //   [experts]               exp_cache (softmax exponentials)
    //   [top_k]                 topk_idx_f (expert IDs as floats; cast to int when read)
    //   [top_k]                 topk_w
    //   [hidden]                x_cache
    //   [intermediate]          acc (per-column post-residual accumulator,
    //                           re-used as the RMSNorm-input buffer)
    //   [1]                     mean_square
    extern __shared__ float smem[];
    float *exp_cache  = smem;
    float *topk_idx_f = smem + experts;
    float *topk_w     = smem + experts + top_k;
    float *x_cache    = smem + experts + 2 * top_k;
    float *acc        = smem + experts + 2 * top_k + hidden;
    float *mean_square = smem + experts + 2 * top_k + hidden + intermediate;

    // ---- stage 1: per-token softmax + top-k (thread 0 only) ----
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int e = 0; e < experts; ++e) {
            float l = __bfloat162float(logits[t * experts + e]);
            if (l > m) m = l;
        }
        float z = 0.0f;
        for (int e = 0; e < experts; ++e) {
            float ev = __expf(__bfloat162float(logits[t * experts + e]) - m);
            exp_cache[e] = ev;
            z += ev;
        }
        // Stable top-k: pick highest-prob unselected expert; tie-break
        // to lowest index. Mirrors `references/grouped_gemm.py::_router_topk`
        // which sorts by `key=lambda i: (-probs[i], i)`.
        for (int k = 0; k < top_k; ++k) {
            int   best_idx = -1;
            float best_p   = -FLT_MAX;
            for (int e = 0; e < experts; ++e) {
                bool already = false;
                for (int j = 0; j < k; ++j) {
                    if ((int)topk_idx_f[j] == e) { already = true; break; }
                }
                if (already) continue;
                float p = exp_cache[e] / z;
                if (p > best_p || (p == best_p && (best_idx == -1 || e < best_idx))) {
                    best_p   = p;
                    best_idx = e;
                }
            }
            topk_idx_f[k] = (float)best_idx;
            topk_w[k]     = best_p;
        }
        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k) wsum += topk_w[k];
        if (wsum > 0.0f) {
            for (int k = 0; k < top_k; ++k) topk_w[k] /= wsum;
        }
    }
    __syncthreads();

    // ---- stage 2: cache x[t] in shared memory ----
    for (int k = tid; k < hidden; k += blockDim.x) {
        x_cache[k] = __bfloat162float(x[t * hidden + k]);
    }
    __syncthreads();

    // ---- stage 3: weighted-expert matmul + ----
    // ---- stage 4: residual add ----
    for (int c = tid; c < intermediate; c += blockDim.x) {
        float my_acc = 0.0f;
        for (int k_slot = 0; k_slot < top_k; ++k_slot) {
            const int   eid = (int)topk_idx_f[k_slot];
            const float w   = topk_w[k_slot];
            float partial = 0.0f;
            const __nv_bfloat16 *we = w_experts + (size_t)eid * hidden * intermediate;
            for (int k = 0; k < hidden; ++k) {
                float xv = x_cache[k];
                float wv = __bfloat162float(we[(size_t)k * intermediate + c]);
                partial += xv * wv;
            }
            my_acc += w * partial;
        }
        my_acc += x_cache[c];   // residual; hidden == intermediate by row contract
        acc[c] = my_acc;
    }
    __syncthreads();

    // ---- stage 5: RMSNorm ----
    if (tid == 0) {
        float ss = 0.0f;
        for (int c = 0; c < intermediate; ++c) ss += acc[c] * acc[c];
        mean_square[0] = ss / (float)hidden;
    }
    __syncthreads();
    const float rstd = rsqrtf(mean_square[0] + MOE_LAYER_RMS_EPS);
    for (int c = tid; c < intermediate; c += blockDim.x) {
        const float gv = __bfloat162float(gamma[c]);
        out[(size_t)t * intermediate + c] = __float2bfloat16(acc[c] * rstd * gv);
    }
}


extern "C" void moe_layer_residual_norm_pipeline_launch(
    const __nv_bfloat16 *d_logits,
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_w_experts,
    const __nv_bfloat16 *d_gamma,
    __nv_bfloat16 *d_out,
    int tokens, int experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    // Heuristic: cap threads at intermediate, but at least 32 (one warp)
    // so the syncthreads paths are not pathological. block.x is bounded
    // by the device's max threads-per-block (1024 on sm_120).
    int threads = intermediate;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;
    dim3 grid(tokens);
    dim3 block(threads);
    size_t shmem = (size_t)(experts + 2 * top_k + hidden + intermediate + 1) * sizeof(float);
    moe_layer_residual_norm_pipeline_kernel<<<grid, block, shmem, stream>>>(
        d_logits, d_x, d_w_experts, d_gamma, d_out,
        tokens, experts, top_k, hidden, intermediate);
}
