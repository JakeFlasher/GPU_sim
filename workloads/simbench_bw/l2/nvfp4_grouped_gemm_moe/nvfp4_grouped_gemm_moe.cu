// slug: nvfp4_grouped_gemm_moe
// tier: B
// arch_gate: sm_120a
//
// ROUND 39 STATUS: BF16 stand-in for the NVFP4 sm_120a MoE grouped
// GEMM. The row is marked `status = "unsupported"` in roster.toml
// until a real `mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale`
// PTX path lands (see `l1/nvfp4_blockscale_gemm/` for the MMA
// template). The committed tree here compiles for the sm_120a
// arch-gate and produces numerically compatible output against
// `references/grouped_gemm.py::build_reference`, but it does NOT
// exercise the declared NVFP4 hardware contract — which is why the
// row is unsupported rather than scored. Codex R38-review
// rejected BF16 stand-ins as delivery evidence for row #53.
//
// Underlying math (placeholder): per-token softmax top-k routing over
// expert logits, then accumulate each top-k expert's BF16 matmul
// output weighted by the renormalized routing probability. Matches
// `references/grouped_gemm.py::build_reference` structurally; NVFP4
// block-scale quantization is NOT applied.
//
// Grid: one block per token. Block: 256 threads striding over the
// intermediate dim. Thread 0 computes the top-k routing (small serial
// scan over `experts`). Output:
//   out[t, c] = sum_{k in top_k} w[t, k] * dot(x[t, :], w_experts[e[t,k], :, c])

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

constexpr int kMaxTopK = 16;

extern "C" __global__ void nvfp4_grouped_gemm_moe_kernel(
    const float         *__restrict__ logits,     // [tokens, experts]  — FP32 so routing matches reference
    const __nv_bfloat16 *__restrict__ x,          // [tokens, hidden]
    const __nv_bfloat16 *__restrict__ w_experts,  // [experts, hidden, intermediate]
    __nv_bfloat16 *__restrict__ out,              // [tokens, intermediate]
    int32_t experts,
    int32_t top_k,
    int32_t hidden,
    int32_t intermediate)
{
    const int token = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ int   s_idx[kMaxTopK];
    __shared__ float s_w  [kMaxTopK];

    // Stage 1: top-k routing with stable tie-breaking + renormalize.
    // Single-thread scan is fine since experts is small (<=64 for MoE).
    // Logits are FP32 (not BF16) so ties match the Python reference bit-for-bit;
    // BF16 logits quantization can collapse two close experts onto the same
    // representable value and flip which index wins a tie.
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
            const __nv_bfloat16 *w_row = w_experts + ((size_t)eid * hidden * intermediate);
            for (int h = 0; h < hidden; ++h) {
                float xv = __bfloat162float(x[token * hidden + h]);
                float wv = __bfloat162float(w_row[h * intermediate + c]);
                partial += xv * wv;
            }
            acc += s_w[k] * partial;
        }
        out[token * intermediate + c] = __float2bfloat16(acc);
    }
}

extern "C" void nvfp4_grouped_gemm_moe_launch(
    const float         *d_logits,
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_w_experts,
    __nv_bfloat16 *d_out,
    int tokens, int experts, int top_k, int hidden, int intermediate,
    cudaStream_t stream)
{
    dim3 grid(tokens);
    dim3 block(256);
    nvfp4_grouped_gemm_moe_kernel<<<grid, block, 0, stream>>>(
        d_logits, d_x, d_w_experts, d_out, experts, top_k, hidden, intermediate);
}
