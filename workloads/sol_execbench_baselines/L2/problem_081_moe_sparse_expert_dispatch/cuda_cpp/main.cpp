#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Sparse MoE expert computation with internal sigmoid-based top-k routing,
// group-wise expert selection, and a dense shared expert. The router scores
// every token over the routed experts, applies a score-correction bias, selects
// the active group and then the top-k experts, normalizes and scales the
// weights, dispatches tokens to per-expert SwiGLU MLPs, accumulates the routed
// contribution, and finally adds a dense SwiGLU shared expert. Reproduces the
// reference run() op-for-op via pure ATen / cuBLAS dispatch -- no custom device
// kernels.
//
// Pinned axes from the problem definition:
//   hidden_size           = 4096
//   moe_intermediate_size = 1408
//   n_routed_experts      = 128
//   num_experts_per_tok   = 8
//   n_group               = 1
//   topk_group            = 1
//   shared_intermediate   = 1408
//   norm_topk_prob        = true
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states           : [num_tokens, hidden_size]                          bfloat16
//     router_weight           : [n_routed_experts, hidden_size]                     bfloat16
//     expert_gate_weights     : [n_routed_experts, moe_intermediate_size, hidden_size] bfloat16
//     expert_up_weights       : [n_routed_experts, moe_intermediate_size, hidden_size] bfloat16
//     expert_down_weights     : [n_routed_experts, hidden_size, moe_intermediate_size] bfloat16
//     shared_gate_weight      : [shared_intermediate, hidden_size]                 bfloat16
//     shared_up_weight        : [shared_intermediate, hidden_size]                 bfloat16
//     shared_down_weight      : [hidden_size, shared_intermediate]                 bfloat16
//     e_score_correction_bias : [n_routed_experts]                                 float32
//     routed_scaling_factor   : scalar                                             float32
//   output (bfloat16, DPS pre-alloc, written in place):
//     output : [num_tokens, hidden_size]
//
// Router and selection the reference computes (all fp32 until the weight cast):
//   router_logits   = linear(hidden_states.float(), router_weight.float())
//   scores          = sigmoid(router_logits)
//   scores_choice   = scores + e_score_correction_bias.unsqueeze(0)
//   group_scores    = scores_choice.view(-1, n_group, experts_per_group)
//   group_agg       = group_scores.topk(2, dim=-1)[0].sum(dim=-1)
//   group_idx       = topk(group_agg, k=topk_group, dim=-1, sorted=False)[1]
//   group_mask      = zeros_like(group_agg); group_mask.scatter_(1, group_idx, 1)
//   score_mask      = group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group)
//                                              .reshape(-1, n_routed_experts)
//   scores_choice   = scores_choice.masked_fill(~score_mask.bool(), 0.0)
//   topk_w, topk_i  = topk(scores_choice, k=num_experts_per_tok, dim=-1, sorted=False)
//   topk_w          = topk_w / (topk_w.sum(-1, keepdim) + 1e-20)
//   topk_w          = topk_w * routed_scaling_factor
//   topk_w          = topk_w.to(bf16)
//
// Routed path, per expert e over the fixed expert count:
//   expert_mask = one_hot(topk_i, n_routed_experts).permute(2, 0, 1)
//                 -> [n_routed_experts, num_tokens, num_experts_per_tok]
//   token_idx, weight_idx = where(expert_mask[e])
//   process e only when token_idx.numel() > 0 (the reference's if-guard)
//   expert_weights = topk_w[token_idx, weight_idx]
//   expert_input   = hidden_states[token_idx]                       (bf16)
//   gate_output    = silu(linear(expert_input, gate_w))             (bf16)
//   up_output      = linear(expert_input, up_w)                     (bf16)
//   intermediate   = gate_output * up_output                        (bf16)
//   expert_output  = linear(intermediate, down_w)                   (bf16)
//   weighted       = expert_output * expert_weights.unsqueeze(-1)   (bf16)
//   final_hidden_states.index_add_(0, token_idx, weighted)
//
// Shared path (every token):
//   shared_gate_out = silu(linear(hidden_states, shared_gate_weight)) (bf16)
//   shared_up_out   = linear(hidden_states, shared_up_weight)         (bf16)
//   shared_inter    = shared_gate_out * shared_up_out                 (bf16)
//   shared_output   = linear(shared_inter, shared_down_weight)        (bf16)
//   output          = final_hidden_states + shared_output             (bf16)
//
// Precision mirrors the reference exactly. The entire routing path runs in fp32:
// the router linear is computed on fp32-upcast inputs and weights, the sigmoid,
// the fp32 score-correction bias add, the group aggregation / selection, the
// masked_fill, the top-k selection, and the normalization all stay in fp32; the
// fp32 routed_scaling_factor scalar multiplies in fp32, and only then are the
// per-token weights cast to bf16. Every expert-MLP and shared-expert projection
// runs in bf16 just as the reference's F.linear calls do -- no fp32 upcast on
// the compute path -- and both SwiGLU gates use the canonical F.silu
// (x * sigmoid(x)). The one_hot / permute / where construction reproduces the
// reference's exact (token_idx, weight_idx) ordering that drives index_add_,
// and the numel > 0 guard skips experts with no routed tokens. The host loop
// runs over the fixed expert count read from router_weight.size(0) and issues
// work only on the current PyTorch stream.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_weight,
    const torch::Tensor& expert_gate_weights,
    const torch::Tensor& expert_up_weights,
    const torch::Tensor& expert_down_weights,
    const torch::Tensor& shared_gate_weight,
    const torch::Tensor& shared_up_weight,
    const torch::Tensor& shared_down_weight,
    const torch::Tensor& e_score_correction_bias,
    double routed_scaling_factor,
    torch::Tensor output
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(router_weight, torch::kBFloat16, "router_weight");
    check_dtype_cuda(expert_gate_weights, torch::kBFloat16, "expert_gate_weights");
    check_dtype_cuda(expert_up_weights, torch::kBFloat16, "expert_up_weights");
    check_dtype_cuda(expert_down_weights, torch::kBFloat16, "expert_down_weights");
    check_dtype_cuda(shared_gate_weight, torch::kBFloat16, "shared_gate_weight");
    check_dtype_cuda(shared_up_weight, torch::kBFloat16, "shared_up_weight");
    check_dtype_cuda(shared_down_weight, torch::kBFloat16, "shared_down_weight");
    check_dtype_cuda(e_score_correction_bias, torch::kFloat32, "e_score_correction_bias");
    check_dtype_cuda(output, torch::kBFloat16, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(router_weight.dim() == 2, "router_weight must be 2D, got ", router_weight.dim(), "D");
    TORCH_CHECK(expert_gate_weights.dim() == 3, "expert_gate_weights must be 3D, got ", expert_gate_weights.dim(), "D");
    TORCH_CHECK(expert_up_weights.dim() == 3, "expert_up_weights must be 3D, got ", expert_up_weights.dim(), "D");
    TORCH_CHECK(expert_down_weights.dim() == 3, "expert_down_weights must be 3D, got ", expert_down_weights.dim(), "D");
    TORCH_CHECK(shared_gate_weight.dim() == 2, "shared_gate_weight must be 2D, got ", shared_gate_weight.dim(), "D");
    TORCH_CHECK(shared_up_weight.dim() == 2, "shared_up_weight must be 2D, got ", shared_up_weight.dim(), "D");
    TORCH_CHECK(shared_down_weight.dim() == 2, "shared_down_weight must be 2D, got ", shared_down_weight.dim(), "D");
    TORCH_CHECK(e_score_correction_bias.dim() == 1, "e_score_correction_bias must be 1D, got ", e_score_correction_bias.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t n_routed_experts = router_weight.size(0);
    constexpr int64_t kNumExpertsPerTok = 8;
    constexpr int64_t kNGroup = 1;
    constexpr int64_t kTopkGroup = 1;
    const int64_t experts_per_group = n_routed_experts / kNGroup;

    // Router logits in fp32 (reference upcasts both inputs and weights).
    auto router_logits =
        at::linear(hidden_states.to(torch::kFloat32), router_weight.to(torch::kFloat32));

    // Sigmoid scores plus the fp32 score-correction bias.
    auto scores = at::sigmoid(router_logits);
    auto scores_for_choice = scores + e_score_correction_bias.unsqueeze(0);

    // Group-wise selection: aggregate each group's top-2 scores, pick the top
    // group(s), and broadcast the group mask back over all experts.
    auto group_scores = scores_for_choice.view({-1, kNGroup, experts_per_group});
    auto group_topk2 = at::topk(group_scores, /*k=*/2, /*dim=*/-1,
                                /*largest=*/true, /*sorted=*/true);
    auto group_scores_agg =
        std::get<0>(group_topk2).sum(std::vector<int64_t>{-1});  // [num_tokens, n_group]

    auto group_idx = std::get<1>(at::topk(group_scores_agg, /*k=*/kTopkGroup,
                                          /*dim=*/-1, /*largest=*/true,
                                          /*sorted=*/false));
    auto group_mask = at::zeros_like(group_scores_agg);
    group_mask.scatter_(/*dim=*/1, group_idx, 1);

    auto score_mask = group_mask.unsqueeze(-1)
                          .expand({-1, kNGroup, experts_per_group})
                          .reshape({-1, n_routed_experts});
    scores_for_choice =
        scores_for_choice.masked_fill(~score_mask.to(torch::kBool), 0.0);

    // Top-k expert selection, fp32 normalization, fp32 scaling, then bf16 cast.
    auto topk = at::topk(scores_for_choice, /*k=*/kNumExpertsPerTok, /*dim=*/-1,
                         /*largest=*/true, /*sorted=*/false);
    auto topk_weights = std::get<0>(topk);   // [num_tokens, num_experts_per_tok] fp32
    auto topk_indices = std::get<1>(topk);   // [num_tokens, num_experts_per_tok] int64

    auto denominator = topk_weights.sum(std::vector<int64_t>{-1}, /*keepdim=*/true) + 1e-20;
    topk_weights = topk_weights / denominator;
    topk_weights = topk_weights * routed_scaling_factor;
    topk_weights = topk_weights.to(torch::kBFloat16);

    // Routed accumulator initialized to zero (reference: zeros_like(hidden_states)).
    auto final_hidden_states = at::zeros_like(hidden_states);

    // expert_mask: [n_routed_experts, num_tokens, num_experts_per_tok].
    auto expert_mask =
        at::one_hot(topk_indices, n_routed_experts).permute({2, 0, 1});

    for (int64_t expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
        // where on the [num_tokens, num_experts_per_tok] slice returns
        // {token indices, slot indices} in row-major order -- matching the
        // reference's (token_indices, weight_indices) unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto token_indices = nz[0];   // token indices
        auto weight_indices = nz[1];  // slot indices

        if (token_indices.numel() > 0) {
            // Per-expert routing weights and gathered tokens.
            auto expert_weights =
                topk_weights.index({token_indices, weight_indices});
            auto expert_input =
                hidden_states.index_select(/*dim=*/0, token_indices);

            // Per-expert projection weights.
            auto gate_w = expert_gate_weights[expert_idx];   // [intermediate, hidden]
            auto up_w = expert_up_weights[expert_idx];       // [intermediate, hidden]
            auto down_w = expert_down_weights[expert_idx];   // [hidden, intermediate]

            // SwiGLU expert MLP in bf16: F.silu(gate) * up, then down.
            auto gate_output = at::silu(at::linear(expert_input, gate_w));
            auto up_output = at::linear(expert_input, up_w);
            auto intermediate = gate_output * up_output;
            auto expert_output = at::linear(intermediate, down_w);

            // Weight by the per-(token, slot) bf16 routing weight, accumulate.
            auto weighted_output = expert_output * expert_weights.unsqueeze(-1);
            final_hidden_states.index_add_(/*dim=*/0, token_indices, weighted_output);
        }
    }

    // Shared expert evaluated densely on every token (bf16 SwiGLU).
    auto shared_gate_output = at::silu(at::linear(hidden_states, shared_gate_weight));
    auto shared_up_output = at::linear(hidden_states, shared_up_weight);
    auto shared_intermediate = shared_gate_output * shared_up_output;
    auto shared_output = at::linear(shared_intermediate, shared_down_weight);

    // Combine routed and shared experts.
    auto result = final_hidden_states + shared_output;

    TORCH_CHECK(output.sizes() == result.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", result.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Sparse MoE expert dispatch with sigmoid top-k routing and a shared "
        "expert. Pure ATen dispatch: fp32 router linear -> sigmoid -> fp32 "
        "score-correction bias -> group-wise selection (top-2 aggregate, top "
        "group, broadcast mask, masked_fill) -> top-k selection -> fp32 "
        "normalization and fp32 routed-scaling-factor scaling cast back to "
        "bf16; a per-expert host loop over the fixed expert count gathers each "
        "expert's tokens (one_hot / permute / where / index_select under a "
        "numel guard), runs the bf16 SwiGLU MLP, weights by the bf16 routing "
        "weight, and accumulates via index_add_; a dense bf16 SwiGLU shared "
        "expert is added to every token, reproduced op-for-op from the "
        "reference. Destination-passing style (output written in place).");
}
