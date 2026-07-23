#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Backward pass for a complete MoE layer with a shared expert (GLM-4.5-Air
// style). Computes gradients through the router (sigmoid + top-k selection)
// and the shared expert with SwiGLU.
//
// Reference (problem_080_moe_complete_layer_with_shared_expert_backward): the
// analytic backward written out by hand as explicit tensor algebra (no
// torch.autograd). It deliberately covers ONLY the shared-expert and router
// paths: the 128 routed-expert weight gradients are omitted by the reference
// (they would require passing all 128*3 weight tensors and their saved
// activations), so there is no per-expert loop and the routed contribution is
// approximated through an isotropic straight-through estimator on the routing
// weights. This port reproduces that omission and that estimator exactly.
//
// Precision profile (matched exactly, per definition.json per-tensor dtypes):
//   - The shared-expert activation-gradient chain runs in bfloat16 (bf16 @ bf16
//     stays bf16), matching the reference's untouched dtypes there.
//   - Every weight gradient is accumulated in float32 via explicit .to(float32)
//     matmuls, then cast back to bfloat16 (grad_shared_expert_{down,up,gate}).
//   - The SiLU derivative is computed in float32 (the reference promotes the
//     gate output and grad to float32), then the gate-output grad is cast back
//     to bfloat16.
//   - The router path keeps router_logits / scores grads in float32; the
//     hidden-state contribution from the router uses a bf16 matmul
//     (grad_router_logits.to(bf16) @ router_weight), while grad_router_weight
//     stays float32 (matching its float32 output dtype).
//   No casts are added beyond the reference's explicit .to() calls.
//
// Per definition.json (insertion order):
//   inputs:
//     grad_output               : [batch_seq_len, hidden_size]              bf16
//     hidden_states             : [batch_seq_len, hidden_size]              bf16
//     router_weight             : [n_routed_experts, hidden_size]           bf16
//     e_score_correction_bias   : [n_routed_experts]                        fp32
//     router_logits             : [batch_seq_len, n_routed_experts]         fp32
//     scores                    : [batch_seq_len, n_routed_experts]         fp32
//     topk_indices              : [batch_seq_len, num_experts_per_tok]      int64
//     topk_weights              : [batch_seq_len, num_experts_per_tok]      fp32
//     score_mask                : [batch_seq_len, n_routed_experts]         fp32
//     shared_expert_gate_weight : [moe_intermediate_size, hidden_size]      bf16
//     shared_expert_up_weight   : [moe_intermediate_size, hidden_size]      bf16
//     shared_expert_down_weight : [hidden_size, moe_intermediate_size]      bf16
//     shared_gate_output        : [batch_seq_len, moe_intermediate_size]    bf16
//     shared_up_output          : [batch_seq_len, moe_intermediate_size]    bf16
//     shared_activated          : [batch_seq_len, moe_intermediate_size]    bf16
//   outputs (DPS pre-alloc, written in place; insertion order):
//     grad_hidden_states             : [batch_seq_len, hidden_size]           bf16
//     grad_router_weight             : [n_routed_experts, hidden_size]        fp32
//     grad_shared_expert_gate_weight : [moe_intermediate_size, hidden_size]   bf16
//     grad_shared_expert_up_weight   : [moe_intermediate_size, hidden_size]   bf16
//     grad_shared_expert_down_weight : [hidden_size, moe_intermediate_size]   bf16
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// @ / .t() matmuls route through the same cuBLAS GEMM the reference's torch
// ops use; the SwiGLU / SiLU vjp, the routing-weight normalization quotient
// rule, the scatter_add over the int64 top-k indices, the sigmoid derivative,
// and every cast match the reference exactly. The routed-expert weight
// gradients are intentionally not produced (the reference omits them).

static void check_cuda_dtype(
    const torch::Tensor& t, const std::string& name, torch::ScalarType expected) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == expected,
                name, " must be ", expected, ", got ", t.scalar_type());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_weight,
    const torch::Tensor& e_score_correction_bias,
    const torch::Tensor& router_logits,
    const torch::Tensor& scores,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_weights,
    const torch::Tensor& score_mask,
    const torch::Tensor& shared_expert_gate_weight,
    const torch::Tensor& shared_expert_up_weight,
    const torch::Tensor& shared_expert_down_weight,
    const torch::Tensor& shared_gate_output,
    const torch::Tensor& shared_up_output,
    const torch::Tensor& shared_activated,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_router_weight,
    torch::Tensor grad_shared_expert_gate_weight,
    torch::Tensor grad_shared_expert_up_weight,
    torch::Tensor grad_shared_expert_down_weight
) {
    check_cuda_dtype(grad_output, "grad_output", torch::kBFloat16);
    check_cuda_dtype(hidden_states, "hidden_states", torch::kBFloat16);
    check_cuda_dtype(router_weight, "router_weight", torch::kBFloat16);
    check_cuda_dtype(router_logits, "router_logits", torch::kFloat32);
    check_cuda_dtype(scores, "scores", torch::kFloat32);
    check_cuda_dtype(topk_indices, "topk_indices", torch::kLong);
    check_cuda_dtype(topk_weights, "topk_weights", torch::kFloat32);
    check_cuda_dtype(score_mask, "score_mask", torch::kFloat32);
    check_cuda_dtype(shared_expert_gate_weight, "shared_expert_gate_weight", torch::kBFloat16);
    check_cuda_dtype(shared_expert_up_weight, "shared_expert_up_weight", torch::kBFloat16);
    check_cuda_dtype(shared_expert_down_weight, "shared_expert_down_weight", torch::kBFloat16);
    check_cuda_dtype(shared_gate_output, "shared_gate_output", torch::kBFloat16);
    check_cuda_dtype(shared_up_output, "shared_up_output", torch::kBFloat16);
    check_cuda_dtype(shared_activated, "shared_activated", torch::kBFloat16);

    TORCH_CHECK(hidden_states.dim() == 2,
                "hidden_states must be 2D [batch_seq_len, hidden_size], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_seq_len = hidden_states.size(0);
    const int64_t n_routed_experts = 128;
    const bool norm_topk_prob = true;
    const double routed_scaling_factor = 1.0;

    // Initialize gradients.
    auto grad_hidden_states_result = at::zeros_like(hidden_states);

    // Gradient flows through addition: split to routed and shared paths.
    auto grad_shared_output = grad_output.clone();

    // ===== Backward through shared expert =====
    // Gradient through shared_expert_down: output = down(activated).
    auto grad_shared_activated = at::matmul(grad_shared_output, shared_expert_down_weight);

    // grad_shared_expert_down_weight = grad_shared_output^T @ shared_activated,
    // accumulated in float32 then cast back to bfloat16.
    auto grad_shared_expert_down_weight_result =
        at::matmul(grad_shared_output.t().to(torch::kFloat32),
                   shared_activated.to(torch::kFloat32));
    grad_shared_expert_down_weight_result =
        grad_shared_expert_down_weight_result.to(torch::kBFloat16);

    // Gradient through SwiGLU: activated = silu(gate) * up.
    auto grad_shared_gate_silu = grad_shared_activated * shared_up_output;
    auto grad_shared_up_output = grad_shared_activated * at::silu(shared_gate_output);

    // Gradient through silu: silu(x) = x * sigmoid(x), derivative in float32.
    auto sigmoid_gate = at::sigmoid(shared_gate_output.to(torch::kFloat32));
    auto shared_gate_float = shared_gate_output.to(torch::kFloat32);
    auto grad_shared_gate_output = grad_shared_gate_silu.to(torch::kFloat32) *
        (sigmoid_gate * (1.0 + shared_gate_float * (1.0 - sigmoid_gate)));
    auto grad_shared_gate_output_bf16 = grad_shared_gate_output.to(torch::kBFloat16);

    // Gradient through shared_expert_up and shared_expert_gate.
    auto grad_hidden_from_shared_up = at::matmul(grad_shared_up_output, shared_expert_up_weight);
    auto grad_hidden_from_shared_gate = at::matmul(grad_shared_gate_output_bf16, shared_expert_gate_weight);

    // Weight grads accumulate in float32, cast back to bfloat16.
    auto grad_shared_expert_up_weight_result =
        at::matmul(grad_shared_up_output.t().to(torch::kFloat32),
                   hidden_states.to(torch::kFloat32));
    grad_shared_expert_up_weight_result =
        grad_shared_expert_up_weight_result.to(torch::kBFloat16);

    auto grad_shared_expert_gate_weight_result =
        at::matmul(grad_shared_gate_output_bf16.t().to(torch::kFloat32),
                   hidden_states.to(torch::kFloat32));
    grad_shared_expert_gate_weight_result =
        grad_shared_expert_gate_weight_result.to(torch::kBFloat16);

    grad_hidden_states_result =
        grad_hidden_states_result + grad_hidden_from_shared_up + grad_hidden_from_shared_gate;

    // ===== Backward through routing =====
    // The routed expert outputs are not saved (would need 128*3 weights and
    // activations), so the reference approximates the routing-weight gradient
    // with an isotropic straight-through estimator: grad_w_k = ||grad_output||^2
    // / num_experts_per_tok, broadcast across the top-k slots.
    const int64_t num_experts_per_tok = topk_weights.size(-1);

    auto grad_output_f32 = grad_output.to(torch::kFloat32);
    auto grad_norm_sq = (grad_output_f32 * grad_output_f32).sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_topk_weights = grad_norm_sq.expand_as(topk_weights) / static_cast<double>(num_experts_per_tok);

    // Gradient through routing weight normalization and scaling.
    torch::Tensor grad_topk_weights_before_norm;
    if (norm_topk_prob) {
        // w_norm = w / sum(w) * routed_scaling_factor.
        auto topk_weights_unnorm = topk_weights / routed_scaling_factor;
        auto denominator = topk_weights_unnorm.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20;
        auto grad_topk_weights_unnorm = grad_topk_weights / routed_scaling_factor;

        // Quotient rule: d/dw_i (w_i / S) = (S - w_i) / S^2.
        auto sum_grad =
            (grad_topk_weights_unnorm * topk_weights_unnorm).sum(/*dim=*/-1, /*keepdim=*/true)
            / denominator;
        grad_topk_weights_before_norm = (grad_topk_weights_unnorm - sum_grad) / denominator;
    } else {
        grad_topk_weights_before_norm = grad_topk_weights / routed_scaling_factor;
    }

    // Gradient through top-k selection (sparse): scatter the per-slot grads into
    // the full expert axis at the chosen indices.
    auto grad_scores_for_choice = at::zeros(
        {batch_seq_len, n_routed_experts},
        hidden_states.options().dtype(torch::kFloat32));
    grad_scores_for_choice.scatter_add_(/*dim=*/1, topk_indices, grad_topk_weights_before_norm);

    // Gradient through masking (only selected groups receive gradient).
    grad_scores_for_choice = grad_scores_for_choice * score_mask;

    // Gradient through score correction (bias is non-trainable; only scores).
    auto grad_scores = grad_scores_for_choice;

    // Gradient through sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
    auto grad_router_logits = grad_scores * scores * (1.0 - scores);

    // Gradient through the router linear projection.
    auto grad_hidden_from_router = at::matmul(grad_router_logits.to(torch::kBFloat16), router_weight);

    // grad_router_weight = grad_router_logits^T @ hidden_states, in float32.
    auto grad_router_weight_result =
        at::matmul(grad_router_logits.t(), hidden_states.to(torch::kFloat32));

    grad_hidden_states_result = grad_hidden_states_result + grad_hidden_from_router;

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_hidden_states_result.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states shape ", grad_hidden_states_result.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_router_weight_result.sizes() == grad_router_weight.sizes(),
                "grad_router_weight shape ", grad_router_weight_result.sizes(),
                " does not match pre-allocated ", grad_router_weight.sizes());
    TORCH_CHECK(grad_shared_expert_gate_weight_result.sizes() == grad_shared_expert_gate_weight.sizes(),
                "grad_shared_expert_gate_weight shape ", grad_shared_expert_gate_weight_result.sizes(),
                " does not match pre-allocated ", grad_shared_expert_gate_weight.sizes());
    TORCH_CHECK(grad_shared_expert_up_weight_result.sizes() == grad_shared_expert_up_weight.sizes(),
                "grad_shared_expert_up_weight shape ", grad_shared_expert_up_weight_result.sizes(),
                " does not match pre-allocated ", grad_shared_expert_up_weight.sizes());
    TORCH_CHECK(grad_shared_expert_down_weight_result.sizes() == grad_shared_expert_down_weight.sizes(),
                "grad_shared_expert_down_weight shape ", grad_shared_expert_down_weight_result.sizes(),
                " does not match pre-allocated ", grad_shared_expert_down_weight.sizes());

    grad_hidden_states.copy_(grad_hidden_states_result);
    grad_router_weight.copy_(grad_router_weight_result);
    grad_shared_expert_gate_weight.copy_(grad_shared_expert_gate_weight_result);
    grad_shared_expert_up_weight.copy_(grad_shared_expert_up_weight_result);
    grad_shared_expert_down_weight.copy_(grad_shared_expert_down_weight_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for a MoE layer with shared expert (router + shared "
        "paths only; routed-expert weight grads omitted as in the reference). "
        "Pure ATen / cuBLAS dispatch: the shared-expert SwiGLU vjp, the SiLU "
        "derivative, the routing-weight normalization quotient rule, the "
        "scatter_add over int64 top-k indices, and the sigmoid derivative "
        "reproduced op-for-op from the reference. Mixed bf16/fp32 precision "
        "per definition.json: weight grads accumulate in float32 and cast back "
        "to bfloat16; grad_router_weight stays float32. Destination-passing "
        "style (5 outputs).");
}
