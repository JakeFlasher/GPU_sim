#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// One custom CUDA kernel (silu_mul_bf16) paired with at:: ops for the
// MoE dispatch pipeline. The custom kernel handles the SwiGLU
// activation fusion `silu(gate_out) * up_out` in bf16 — at:: would
// otherwise launch two separate elementwise kernels (silu and mul)
// per expert dispatch, and there are up to 128 expert iterations
// per launch.
//
// Everything else dispatches through PyTorch CUDA ops:
//   - gate matmul, expert gate / up / down matmuls — at::matmul
//   - softmax + topk + (optional) normalize — at::softmax, at::topk
//   - one-hot expert mask + nonzero + index_select +
//     index_add_ — PyTorch's CUDA primitives on the current stream.

namespace {

constexpr int BLOCK_THREADS = 256;

__device__ inline float silu_f(float v) {
    return v / (1.0f + __expf(-v));
}

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
silu_mul_bf16_kernel(
    const __nv_bfloat162* __restrict__ gate,
    const __nv_bfloat162* __restrict__ up,
    __nv_bfloat162* __restrict__ out,
    long num_pairs) {

    // The reference computes `silu_gate = gate_out * sigmoid(gate_out)`
    // (bf16) and then `intermediate = silu_gate * up_out` (bf16). Both
    // intermediates round to bf16 before the next multiply. Match the
    // reference's lower-precision intermediates by rounding the
    // `gate * sigmoid(gate)` product to bf16 before multiplying by up.
    long stride = static_cast<long>(blockDim.x) * gridDim.x;
    long i_start = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (long i = i_start; i < num_pairs; i += stride) {
        float2 g_f2 = __bfloat1622float2(__ldg(gate + i));
        float2 u_f2 = __bfloat1622float2(__ldg(up + i));
        float sg0 = 1.0f / (1.0f + __expf(-g_f2.x));
        float sg1 = 1.0f / (1.0f + __expf(-g_f2.y));
        __nv_bfloat162 silu_bf = __float22bfloat162_rn(
            make_float2(g_f2.x * sg0, g_f2.y * sg1));
        float2 silu_f2 = __bfloat1622float2(silu_bf);
        out[i] = __float22bfloat162_rn(
            make_float2(silu_f2.x * u_f2.x, silu_f2.y * u_f2.y));
    }
}

}  // namespace

void silu_mul_bf16_kernel_launch(
    const at::BFloat16* gate,
    const at::BFloat16* up,
    at::BFloat16* out,
    long num_elems,
    cudaStream_t stream) {

    const long num_pairs = num_elems / 2;
    constexpr int blocks_per_sm = 32;
    constexpr int num_sms = 84;
    long num_blocks = std::min<long>(
        static_cast<long>(blocks_per_sm) * num_sms,
        (num_pairs + BLOCK_THREADS - 1) / BLOCK_THREADS);
    if (num_blocks <= 0) num_blocks = 1;
    dim3 grid(static_cast<unsigned int>(num_blocks));
    dim3 block(BLOCK_THREADS);
    silu_mul_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(gate),
        reinterpret_cast<const __nv_bfloat162*>(up),
        reinterpret_cast<__nv_bfloat162*>(out),
        num_pairs);
}

void problem_008_moe_sparse_routing_and_dispatch_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gate_weight,
    const torch::Tensor& expert_gate_proj,
    const torch::Tensor& expert_up_proj,
    const torch::Tensor& expert_down_proj,
    bool norm_topk_prob,
    torch::Tensor& output,
    cudaStream_t stream) {

    constexpr int top_k = 8;
    const int num_experts = static_cast<int>(gate_weight.size(0));

    const int B = static_cast<int>(hidden_states.size(0));
    const int S = static_cast<int>(hidden_states.size(1));
    const int H = static_cast<int>(hidden_states.size(2));
    const long num_tokens = static_cast<long>(B) * static_cast<long>(S);

    // Flatten hidden_states to 2D.
    auto hidden_flat = hidden_states.view({num_tokens, H});

    // (a) routing logits.
    auto router_logits = at::matmul(hidden_flat, gate_weight.t());
    // (b) softmax in fp32, cast back to bf16.
    auto routing_weights =
        at::softmax(router_logits.to(at::kFloat), 1).to(at::kBFloat16);
    // (c) topk.
    auto topk_result = at::topk(routing_weights, top_k, /*dim=*/-1);
    auto routing_weights_topk = std::get<0>(topk_result);  // [N, top_k] bf16
    auto selected_experts = std::get<1>(topk_result);      // [N, top_k] int64

    // (d) normalize topk weights if requested.
    if (norm_topk_prob) {
        auto sum_w = routing_weights_topk.sum(-1, /*keepdim=*/true);
        routing_weights_topk =
            routing_weights_topk / (sum_w + 1e-9);
    }

    // (e) zero-initialize accumulator.
    auto final_hidden = torch::zeros({num_tokens, H}, hidden_flat.options());

    // (f) expert mask, shape [num_experts, top_k, N].
    auto expert_mask = at::one_hot(selected_experts, num_experts)
        .permute({2, 1, 0});

    // (g) per-expert dispatch loop.
    for (int e = 0; e < num_experts; ++e) {
        auto where_2d = at::nonzero(expert_mask[e]);  // [n_hits, 2]
        const long n_hits = where_2d.size(0);
        if (n_hits == 0) continue;

        auto idx = where_2d.select(/*dim=*/1, 0);     // top-k positions
        auto top_x = where_2d.select(/*dim=*/1, 1);   // token indices

        auto current_state = hidden_flat.index_select(0, top_x);  // [n_hits, H]
        auto gate_w_e = expert_gate_proj[e];  // [I, H]
        auto up_w_e = expert_up_proj[e];      // [I, H]
        auto down_w_e = expert_down_proj[e];  // [H, I]

        auto gate_out = at::matmul(current_state, gate_w_e.t());  // [n_hits, I]
        auto up_out = at::matmul(current_state, up_w_e.t());      // [n_hits, I]

        // Fused SiLU * up via the custom kernel.
        auto intermediate = torch::empty_like(gate_out);
        silu_mul_bf16_kernel_launch(
            gate_out.data_ptr<at::BFloat16>(),
            up_out.data_ptr<at::BFloat16>(),
            intermediate.data_ptr<at::BFloat16>(),
            static_cast<long>(gate_out.numel()),
            stream);

        auto expert_out = at::matmul(intermediate, down_w_e.t());  // [n_hits, H]

        // Weight by routing probability: routing_weights_topk[top_x, idx]
        // indexed as a flat gather, then broadcast over hidden dim.
        auto weights = routing_weights_topk.index({top_x, idx}).unsqueeze(-1);
        auto current_hidden = expert_out * weights;

        final_hidden.index_add_(0, top_x, current_hidden);
    }

    // (h) reshape and copy into DPS output.
    auto reshaped = final_hidden.reshape({B, S, H});
    output.copy_(reshaped);

    (void)stream;
}
