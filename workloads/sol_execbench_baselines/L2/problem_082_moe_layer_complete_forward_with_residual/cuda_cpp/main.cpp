#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Complete MoE layer forward pass: sigmoid-scored hierarchical-group router
// selecting the top-8 of 160 routed experts, SwiGLU expert MLPs with weighted
// aggregation, plus a shared expert that processes every token, with the two
// branches summed into a single output. Reproduces the reference run()
// op-for-op via pure ATen / cuBLAS dispatch -- no custom device kernels.
//
// Pinned axes from the problem definition:
//   hidden_size           = 5120
//   moe_intermediate_size = 1536
//   n_routed_experts      = 160
//   num_experts_per_tok   = 8
//   n_group               = 1
//   topk_group            = 1
//   shared_intermediate   = 1536  (moe_intermediate_size * 1)
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states            : [batch_seq_len, hidden_size]                 bfloat16
//     router_weight            : [n_routed_experts, hidden_size]              bfloat16
//     e_score_correction_bias  : [n_routed_experts]                           float32
//     expert_gate_projs        : [n_routed_experts, moe_inter, hidden_size]   bfloat16
//     expert_up_projs          : [n_routed_experts, moe_inter, hidden_size]   bfloat16
//     expert_down_projs        : [n_routed_experts, hidden_size, moe_inter]   bfloat16
//     shared_gate_proj_weight  : [shared_intermediate, hidden_size]           bfloat16
//     shared_up_proj_weight    : [shared_intermediate, hidden_size]           bfloat16
//     shared_down_proj_weight  : [hidden_size, shared_intermediate]           bfloat16
//     routed_scaling_factor    : scalar                                       float32
//     norm_topk_prob           : scalar                                       bool
//   output (bfloat16, DPS pre-alloc, written in place):
//     output : [batch_seq_len, hidden_size]
//
// The two trailing scalar inputs carry shape None in the definition, so the
// evaluator marshals them as plain Python scalars rather than tensors; they
// are received here by value as a double and a bool.
//
// Math the reference computes:
//   router_logits     = linear(hidden_states.float(), router_weight.float())   (fp32)
//   scores            = sigmoid(router_logits)                                  (fp32)
//   scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)           (fp32)
//   -- hierarchical group masking with n_group = topk_group = 1 selects the one
//      and only group, so the expert mask leaves every score in place; the
//      reconstruction below mirrors the reference op sequence exactly so it
//      stays faithful if the group axes ever change.
//   topk_weights, topk_indices = topk(scores_for_choice, k=8, sorted=False)     (fp32)
//   if norm_topk_prob:
//       topk_weights = topk_weights / (topk_weights.sum(-1, keepdim) + 1e-20)   (fp32)
//   topk_weights = topk_weights * routed_scaling_factor                         (fp32)
//   final = zeros_like(hidden_states, dtype=topk_weights.dtype)                 (fp32 accum)
//   expert_mask = one_hot(topk_indices, 160).permute(2, 0, 1)
//                 -> [160, batch_seq_len, num_experts_per_tok]
//   per expert e in range(160):
//     token_indices, weight_indices = where(expert_mask[e])   (token index, slot index)
//     skip e when no token routes to it (numel guard)
//     expert_input  = hidden_states[token_indices]                             (bf16)
//     sel_weights   = topk_weights[token_indices, weight_indices]              (fp32)
//     gate_out      = linear(expert_input, expert_gate_projs[e])               (bf16)
//     up_out        = linear(expert_input, expert_up_projs[e])                 (bf16)
//     intermediate  = silu(gate_out) * up_out                                  (bf16)
//     expert_out    = linear(intermediate, expert_down_projs[e])               (bf16)
//     weighted      = expert_out * sel_weights.unsqueeze(-1)                    (fp32, bf16*fp32)
//     final.index_add_(0, token_indices, weighted)
//   routed_output = final.to(bf16)
//   shared branch over all tokens, every projection in bf16:
//     shared_out = linear(silu(linear(hidden_states, shared_gate)) *
//                         linear(hidden_states, shared_up), shared_down)        (bf16)
//   output = routed_output + shared_output                                      (bf16)
//
// Precision mirrors the reference exactly. The router runs entirely in fp32:
// hidden states and router weight are upcast before the gate linear, the
// sigmoid / bias add / group selection / top-k / normalization / scaling all
// stay fp32, and the accumulator inherits the fp32 top-k weight dtype. The
// expert-MLP projections and the SwiGLU SiLU gate run in bf16, just as the
// reference's F.linear / F.silu calls do on the bf16 expert weights and bf16
// gathered tokens. The only place fp32 re-enters the expert path is the
// per-(token, slot) weight multiply: a bf16 expert output times an fp32
// routing weight promotes to fp32, which lands back in the fp32 accumulator
// before the single .to(bf16) narrowing that yields routed_output. The shared
// expert is bf16 end to end. The output is the bf16 sum of the routed and
// shared branches. The host loop runs over the fixed routed-expert count and
// issues work only on the current PyTorch stream.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_weight,
    const torch::Tensor& e_score_correction_bias,
    const torch::Tensor& expert_gate_projs,
    const torch::Tensor& expert_up_projs,
    const torch::Tensor& expert_down_projs,
    const torch::Tensor& shared_gate_proj_weight,
    const torch::Tensor& shared_up_proj_weight,
    const torch::Tensor& shared_down_proj_weight,
    double routed_scaling_factor,
    bool norm_topk_prob,
    torch::Tensor output
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(router_weight, torch::kBFloat16, "router_weight");
    check_dtype_cuda(e_score_correction_bias, torch::kFloat32, "e_score_correction_bias");
    check_dtype_cuda(expert_gate_projs, torch::kBFloat16, "expert_gate_projs");
    check_dtype_cuda(expert_up_projs, torch::kBFloat16, "expert_up_projs");
    check_dtype_cuda(expert_down_projs, torch::kBFloat16, "expert_down_projs");
    check_dtype_cuda(shared_gate_proj_weight, torch::kBFloat16, "shared_gate_proj_weight");
    check_dtype_cuda(shared_up_proj_weight, torch::kBFloat16, "shared_up_proj_weight");
    check_dtype_cuda(shared_down_proj_weight, torch::kBFloat16, "shared_down_proj_weight");
    check_dtype_cuda(output, torch::kBFloat16, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(router_weight.dim() == 2, "router_weight must be 2D, got ", router_weight.dim(), "D");
    TORCH_CHECK(e_score_correction_bias.dim() == 1, "e_score_correction_bias must be 1D, got ", e_score_correction_bias.dim(), "D");
    TORCH_CHECK(expert_gate_projs.dim() == 3, "expert_gate_projs must be 3D, got ", expert_gate_projs.dim(), "D");
    TORCH_CHECK(expert_up_projs.dim() == 3, "expert_up_projs must be 3D, got ", expert_up_projs.dim(), "D");
    TORCH_CHECK(expert_down_projs.dim() == 3, "expert_down_projs must be 3D, got ", expert_down_projs.dim(), "D");
    TORCH_CHECK(shared_gate_proj_weight.dim() == 2, "shared_gate_proj_weight must be 2D, got ", shared_gate_proj_weight.dim(), "D");
    TORCH_CHECK(shared_up_proj_weight.dim() == 2, "shared_up_proj_weight must be 2D, got ", shared_up_proj_weight.dim(), "D");
    TORCH_CHECK(shared_down_proj_weight.dim() == 2, "shared_down_proj_weight must be 2D, got ", shared_down_proj_weight.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t n_routed_experts = router_weight.size(0);
    constexpr int64_t kNumExpertsPerTok = 8;
    constexpr int64_t kNGroup = 1;
    constexpr int64_t kTopkGroup = 1;

    // Router forward and sigmoid scoring, in fp32.
    auto router_logits =
        at::linear(hidden_states.to(torch::kFloat32), router_weight.to(torch::kFloat32));
    auto scores = at::sigmoid(router_logits);
    auto scores_for_choice = scores + e_score_correction_bias.unsqueeze(0);

    // Hierarchical group-based selection. With n_group = topk_group = 1 the
    // group mask admits the single group whole, but the op sequence mirrors
    // the reference so it stays faithful under different group axes.
    const int64_t experts_per_group = n_routed_experts / kNGroup;
    auto group_scores_reshaped =
        scores_for_choice.view({-1, kNGroup, experts_per_group});
    auto group_scores =
        std::get<0>(group_scores_reshaped.topk(2, /*dim=*/-1)).sum(/*dim=*/-1);
    auto group_idx =
        std::get<1>(at::topk(group_scores, /*k=*/kTopkGroup, /*dim=*/-1,
                             /*largest=*/true, /*sorted=*/false));
    auto group_mask = at::zeros_like(group_scores);
    group_mask.scatter_(/*dim=*/1, group_idx, 1);
    auto score_mask = group_mask.unsqueeze(-1)
                          .expand({-1, kNGroup, experts_per_group})
                          .reshape({-1, n_routed_experts});
    scores_for_choice =
        scores_for_choice.masked_fill(~score_mask.to(torch::kBool), 0.0);

    // Top-k expert selection within the selected groups, fp32 weights.
    auto topk = at::topk(scores_for_choice, /*k=*/kNumExpertsPerTok, /*dim=*/-1,
                         /*largest=*/true, /*sorted=*/false);
    auto topk_weights = std::get<0>(topk);    // [T, num_experts_per_tok] fp32
    auto topk_indices = std::get<1>(topk);    // [T, num_experts_per_tok] int64

    // Optional fp32 normalization of the routing weights.
    if (norm_topk_prob) {
        auto denominator =
            topk_weights.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20;
        topk_weights = topk_weights / denominator;
    }

    // Apply the fp32 routing scaling factor.
    topk_weights = topk_weights * routed_scaling_factor;

    // Accumulator inherits the fp32 top-k weight dtype (zeros_like with the
    // routing-weight dtype in the reference).
    auto final_hidden_states =
        at::zeros_like(hidden_states, hidden_states.options().dtype(torch::kFloat32));

    // expert_mask: [n_routed_experts, batch_seq_len, num_experts_per_tok].
    auto expert_mask =
        at::one_hot(topk_indices, n_routed_experts).permute({2, 0, 1});

    for (int64_t expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
        // where on the [batch_seq_len, num_experts_per_tok] slice returns
        // {token indices, slot indices} in row-major order -- matching the
        // reference's (token_indices, weight_indices) unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto token_indices = nz[0];    // token indices
        auto weight_indices = nz[1];   // slot indices

        if (token_indices.numel() == 0) {
            continue;
        }

        // Per-expert projection weights.
        auto gate_w = expert_gate_projs[expert_idx];   // [moe_inter, hidden]
        auto up_w = expert_up_projs[expert_idx];        // [moe_inter, hidden]
        auto down_w = expert_down_projs[expert_idx];    // [hidden, moe_inter]

        // Gather this expert's tokens (bf16, no upcast on the expert path) and
        // the fp32 per-(token, slot) routing weight.
        auto expert_input = hidden_states.index_select(/*dim=*/0, token_indices);
        auto sel_weights = topk_weights.index({token_indices, weight_indices});

        // SwiGLU expert MLP in bf16, matching the reference's F.linear / F.silu.
        auto gate_output = at::linear(expert_input, gate_w);
        auto up_output = at::linear(expert_input, up_w);
        auto intermediate = at::silu(gate_output) * up_output;
        auto expert_output = at::linear(intermediate, down_w);

        // Weight by the fp32 routing weight (bf16 * fp32 -> fp32) and scatter
        // into the fp32 accumulator.
        auto weighted_output = expert_output * sel_weights.unsqueeze(-1);
        final_hidden_states.index_add_(/*dim=*/0, token_indices, weighted_output);
    }

    // Narrow the fp32 routed accumulation back to bf16.
    auto routed_output = final_hidden_states.to(torch::kBFloat16);

    // Shared expert over every token, bf16 end to end.
    auto shared_gate_output = at::linear(hidden_states, shared_gate_proj_weight);
    auto shared_up_output = at::linear(hidden_states, shared_up_proj_weight);
    auto shared_intermediate = at::silu(shared_gate_output) * shared_up_output;
    auto shared_output = at::linear(shared_intermediate, shared_down_proj_weight);

    // Combine routed and shared branches in bf16.
    auto combined = routed_output + shared_output;

    TORCH_CHECK(output.sizes() == combined.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", combined.sizes());
    output.copy_(combined);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Complete MoE layer forward with residual. Pure ATen dispatch: fp32 "
        "sigmoid router with hierarchical group selection and top-8 expert "
        "choice (fp32 normalization and scaling); per-expert host loop over "
        "the fixed routed-expert count gathers each expert's tokens (one_hot / "
        "permute / where / index_select), runs the SwiGLU expert MLP in bf16, "
        "weights by the fp32 per-token routing weight into an fp32 accumulator "
        "(narrowed to bf16), and adds the bf16 shared-expert branch, "
        "reproduced op-for-op from the reference. Destination-passing style "
        "(single bf16 output written in place).");
}
