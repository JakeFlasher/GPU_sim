// slug: moe_router_topk_softmax
// tier: P
// arch_gate: sm_90a sm_120
//
// MoE router: per-token softmax over expert logits, then pick the top-k
// indices with stable tie-breaking (lower index wins) and emit the
// re-normalized weights. BF16 logits, FP32 softmax + top-k, BF16 outputs.
//
// Output matches the Python reference in references/grouped_gemm.py so
// kernel vs CPU comparison passes under the family tolerance.
//
// Grid: one block per token. Each block runs a single-thread top-k scan
// (serial; vocab=experts is small, typically <= 256 for MoE routing).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void moe_router_topk_softmax_kernel(
    const __nv_bfloat16 *__restrict__ logits,  // [tokens, experts]
    int32_t *__restrict__ out_idx,             // [tokens, top_k]
    __nv_bfloat16 *__restrict__ out_weights,   // [tokens, top_k]
    int32_t experts,
    int32_t top_k)
{
    const int token = blockIdx.x;
    const int tid = threadIdx.x;

    // Single-thread top-k scan per token. Serial is fine at experts <= 256.
    if (tid != 0) return;

    float m = -FLT_MAX;
    for (int e = 0; e < experts; ++e) {
        float v = __bfloat162float(logits[token * experts + e]);
        if (v > m) m = v;
    }
    // Row sum of exp(v - m) for softmax normalizer
    float z = 0.0f;
    for (int e = 0; e < experts; ++e) {
        z += __expf(__bfloat162float(logits[token * experts + e]) - m);
    }
    // Top-k selection over softmax probs with stable tie-breaking
    int32_t *idx_out = out_idx + token * top_k;
    __nv_bfloat16 *w_out = out_weights + token * top_k;
    for (int k = 0; k < top_k; ++k) {
        float best = -1.0f;
        int best_i = -1;
        for (int e = 0; e < experts; ++e) {
            bool chosen = false;
            for (int j = 0; j < k; ++j) {
                if (idx_out[j] == e) { chosen = true; break; }
            }
            if (chosen) continue;
            float v = __bfloat162float(logits[token * experts + e]);
            float p = __expf(v - m) / z;
            if (p > best || (p == best && e < best_i)) {
                best = p;
                best_i = e;
            }
        }
        idx_out[k] = best_i;
        w_out[k] = __float2bfloat16(best);
    }
    // Re-normalize top-k weights to sum to 1 (MoE convention)
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k) wsum += __bfloat162float(w_out[k]);
    if (wsum > 0.0f) {
        for (int k = 0; k < top_k; ++k) {
            w_out[k] = __float2bfloat16(__bfloat162float(w_out[k]) / wsum);
        }
    }
}

extern "C" void moe_router_topk_softmax_launch(
    const __nv_bfloat16 *d_logits,
    int32_t *d_idx,
    __nv_bfloat16 *d_weights,
    int tokens, int experts, int top_k, cudaStream_t stream)
{
    dim3 grid(tokens);
    dim3 block(32);
    moe_router_topk_softmax_kernel<<<grid, block, 0, stream>>>(
        d_logits, d_idx, d_weights, experts, top_k);
}
