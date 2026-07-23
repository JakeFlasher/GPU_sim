// slug: moe_grouped_gemm_fp8
// tier: P
// arch_gate: sm_90a sm_120
//
// FP8 E4M3 MoE forward: per-token softmax top-k routing over expert
// logits, then accumulate each top-k expert's FP8 matmul output
// weighted by the renormalized routing probability. Direct FP8 port
// of R13 `moe_grouped_gemm_bf16`. Matches
// `references/grouped_gemm.py::build_reference` for slug
// "moe_grouped_gemm_fp8": FP32 logits + routing (to avoid
// quantization-flip ties — the R13 MoE-tie precedent);
// FP8 x / w_experts / output; FP32 accumulation internally.
//
// Grid: one block per token. Block: 256 threads striding over the
// intermediate dim. Thread 0 computes the top-k routing (small serial
// scan over `experts`). Output:
//   out[t, c] = sum_{k in top_k} w[t, k] * dot(x[t, :], w_experts[e[t,k], :, c])

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

constexpr int kMaxTopK = 16;

extern "C" __global__ void moe_grouped_gemm_fp8_kernel(
    const float         *__restrict__ logits,     // [tokens, experts]  — FP32 so routing matches reference
    const __nv_fp8_e4m3 *__restrict__ x,          // [tokens, hidden]
    const __nv_fp8_e4m3 *__restrict__ w_experts,  // [experts, hidden, intermediate]
    __nv_fp8_e4m3 *__restrict__ out,              // [tokens, intermediate]
    int32_t experts,
    int32_t top_k,
    int32_t hidden,
    int32_t intermediate)
{
    const int token = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ int   s_idx[kMaxTopK];
    __shared__ float s_w  [kMaxTopK];

    // Stage 1: top-k routing with stable tie-breaking + renormalize,
    // computed entirely in FP32 so BF16/FP8 quantization cannot flip
    // tied experts (R13 precedent — see moe_grouped_gemm_bf16.cu).
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int e = 0; e < experts; ++e) {
            float v = logits[token * experts + e];
            if (v > m) m = v;
        }
        float z = 0.0f;
        for (int e = 0; e < experts; ++e) {
            z += __expf(logits[token * experts + e] - m);
        }
        for (int k = 0; k < top_k; ++k) {
            float best = -1.0f;
            int best_i = -1;
            for (int e = 0; e < experts; ++e) {
                bool chosen = false;
                for (int j = 0; j < k; ++j) {
                    if (s_idx[j] == e) { chosen = true; break; }
                }
                if (chosen) continue;
                float v = logits[token * experts + e];
                float p = __expf(v - m) / z;
                if (p > best || (p == best && e < best_i)) {
                    best = p;
                    best_i = e;
                }
            }
            s_idx[k] = best_i;
            s_w[k]   = best;
        }
        // Renormalize top-k weights to sum to 1 (standard MoE convention).
        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k) wsum += s_w[k];
        if (wsum > 0.0f) {
            for (int k = 0; k < top_k; ++k) s_w[k] /= wsum;
        }
    }
    __syncthreads();

    // Stage 2: per-intermediate-column accumulation over the top-k experts.
    for (int c = tid; c < intermediate; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int eid = s_idx[k];
            float partial = 0.0f;
            const __nv_fp8_e4m3 *w_row = w_experts + ((size_t)eid * hidden * intermediate);
            for (int h = 0; h < hidden; ++h) {
                float xv = (float)x[token * hidden + h];
                float wv = (float)w_row[h * intermediate + c];
                partial += xv * wv;
            }
            acc += s_w[k] * partial;
        }
        out[token * intermediate + c] = (__nv_fp8_e4m3)acc;
    }
}

extern "C" void moe_grouped_gemm_fp8_launch(
    const float         *d_logits,
    const __nv_fp8_e4m3 *d_x,
    const __nv_fp8_e4m3 *d_w_experts,
    __nv_fp8_e4m3 *d_out,
    int tokens, int experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    dim3 grid(tokens);
    dim3 block(256);
    moe_grouped_gemm_fp8_kernel<<<grid, block, 0, stream>>>(
        d_logits, d_x, d_w_experts, d_out, experts, top_k, hidden, intermediate);
}
