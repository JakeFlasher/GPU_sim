#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Backward pass for MoE expert-parallel SwiGLU dispatch (the analytic backward
// of the expert-parallel forward). Reproduces the reference run() op-for-op
// via pure ATen / cuBLAS dispatch -- no custom device kernels. The reference
// recomputes each expert's forward activations (gate / up projections, SiLU,
// the gate*up SwiGLU product, down projection) and walks the gradients back
// through the weighted scatter-add aggregation, the down projection, the
// element-wise SwiGLU multiply, the SiLU nonlinearity, and the gate / up
// projections, then scatters the input gradient back to the original token
// rows.
//
// Dimensions from the selected problem definition:
//   hidden_size           = 4096
//   moe_intermediate_size = 2048
//   n_routed_experts      = selected definition (projection-weight dim 0)
//   num_experts_per_tok   = 8
//
// Per definition.json (insertion order):
//   inputs (all float32 except topk_indices):
//     grad_output    : [num_tokens, hidden_size]                              float32
//     hidden_states  : [num_tokens, hidden_size]                              float32
//     topk_indices   : [num_tokens, num_experts_per_tok]                      int64
//     topk_weights   : [num_tokens, num_experts_per_tok]                      float32
//     gate_weights   : [n_routed_experts, moe_intermediate_size, hidden_size] float32
//     up_weights     : [n_routed_experts, moe_intermediate_size, hidden_size] float32
//     down_weights   : [n_routed_experts, hidden_size, moe_intermediate_size] float32
//   outputs (all float32, DPS pre-alloc, written in place):
//     grad_hidden_states : [num_tokens, hidden_size]
//     grad_topk_weights  : [num_tokens, num_experts_per_tok]
//     grad_gate_weights  : [n_routed_experts, moe_intermediate_size, hidden_size]
//     grad_up_weights    : [n_routed_experts, moe_intermediate_size, hidden_size]
//     grad_down_weights  : [n_routed_experts, hidden_size, moe_intermediate_size]
//
// Math the reference computes, per expert e over the fixed expert count:
//   expert_mask = one_hot(topk_indices, n_routed_experts).permute(2, 0, 1)
//                 -> [n_routed_experts, num_tokens, num_experts_per_tok]
//   token_idx, weight_idx = where(expert_mask[e])   (token index, slot index)
//   skip e when no token routes to it (numel guard)
//   expert_input = hidden_states[token_idx]                         [n_e, hidden]
//   ew_vals      = topk_weights[token_idx, weight_idx]              [n_e]
//   gate_pre_act = linear(expert_input, gate_w[e])                 [n_e, inter]
//   gate_output  = silu(gate_pre_act)
//   up_output    = linear(expert_input, up_w[e])                   [n_e, inter]
//   intermediate = gate_output * up_output
//   expert_output= linear(intermediate, down_w[e])                 [n_e, hidden]
//   grad_weighted_output = grad_output[token_idx]                  [n_e, hidden]
//   grad_expert_output   = grad_weighted_output * ew_vals[:, None]
//   grad_expert_weights  = (grad_weighted_output * expert_output).sum(-1)
//   grad_topk_weights[token_idx, weight_idx] += grad_expert_weights
//   grad_intermediate    = linear(grad_expert_output, down_w[e].t())
//   grad_down_weights[e] += grad_expert_output.t() @ intermediate
//   grad_gate_output     = grad_intermediate * up_output
//   grad_up_output       = grad_intermediate * gate_output
//   sigmoid_gate         = sigmoid(gate_pre_act)
//   silu_grad            = sigmoid_gate * (1 + gate_pre_act * (1 - sigmoid_gate))
//   grad_gate_pre_act    = grad_gate_output * silu_grad
//   grad_expert_input_gate = linear(grad_gate_pre_act, gate_w[e].t())
//   grad_gate_weights[e] += grad_gate_pre_act.t() @ expert_input
//   grad_expert_input_up   = linear(grad_up_output, up_w[e].t())
//   grad_up_weights[e]   += grad_up_output.t() @ expert_input
//   grad_expert_input    = grad_expert_input_gate + grad_expert_input_up
//   grad_hidden_states.index_add_(0, token_idx, grad_expert_input)
//
// Precision profile (matched exactly): every tensor is float32 and the
// reference stays in float32 throughout -- there are no narrowing cast points.
// The per-expert weight slice is upcast to fp32 inside the loop so the resident
// footprint stays at one expert's weights rather than the whole
// [n_routed_experts, ...] stack; because the inputs are already fp32 this is a
// no-op copy that preserves bit-identical results. The one_hot / permute / where
// construction reproduces the reference's exact (token_idx, weight_idx)
// ordering that drives index_add_ and the routing-weight gather. The
// accumulate index_put_ into grad_topk_weights mirrors the reference's
// advanced-index += (the (token, slot) pairs are unique within an expert, so
// the accumulation is exact). The host loop runs over the fixed expert count
// read from gate_weights.size(0) (iterating every expert and skipping the empty
// ones yields the identical set of accumulations) and issues work only on the
// current PyTorch stream.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_weights,
    const torch::Tensor& gate_weights,
    const torch::Tensor& up_weights,
    const torch::Tensor& down_weights,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_topk_weights,
    torch::Tensor grad_gate_weights,
    torch::Tensor grad_up_weights,
    torch::Tensor grad_down_weights
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(hidden_states, "hidden_states");
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(topk_indices.dtype() == torch::kInt64,
                "topk_indices must be int64, got ", topk_indices.dtype());
    check_fp32_cuda(topk_weights, "topk_weights");
    check_fp32_cuda(gate_weights, "gate_weights");
    check_fp32_cuda(up_weights, "up_weights");
    check_fp32_cuda(down_weights, "down_weights");
    check_fp32_cuda(grad_hidden_states, "grad_hidden_states");
    check_fp32_cuda(grad_topk_weights, "grad_topk_weights");
    check_fp32_cuda(grad_gate_weights, "grad_gate_weights");
    check_fp32_cuda(grad_up_weights, "grad_up_weights");
    check_fp32_cuda(grad_down_weights, "grad_down_weights");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D, got ", grad_output.dim(), "D");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2D, got ", topk_indices.dim(), "D");
    TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D, got ", topk_weights.dim(), "D");
    TORCH_CHECK(gate_weights.dim() == 3, "gate_weights must be 3D, got ", gate_weights.dim(), "D");
    TORCH_CHECK(up_weights.dim() == 3, "up_weights must be 3D, got ", up_weights.dim(), "D");
    TORCH_CHECK(down_weights.dim() == 3, "down_weights must be 3D, got ", down_weights.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    const int64_t n_routed_experts = gate_weights.size(0);

    // Output gradients initialized to zero (reference: zeros_like).
    auto grad_hidden = at::zeros_like(hidden_states);
    auto grad_topk_w = at::zeros_like(topk_weights);
    auto grad_gate_w = at::zeros_like(gate_weights);
    auto grad_up_w = at::zeros_like(up_weights);
    auto grad_down_w = at::zeros_like(down_weights);

    // expert_mask: [n_routed_experts, num_tokens, num_experts_per_tok].
    auto expert_mask =
        at::one_hot(topk_indices, n_routed_experts).permute({2, 0, 1});

    for (int64_t expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
        // where on the [num_tokens, num_experts_per_tok] slice returns
        // {token indices, slot indices} in row-major order -- matching the
        // reference's (token_indices, weight_indices) unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto token_indices = nz[0];   // token indices
        auto weight_indices = nz[1];  // routing-slot indices

        if (token_indices.numel() == 0) {
            continue;
        }

        // Per-expert weight slices upcast to fp32 (no-op copy on already-fp32
        // inputs; keeps the resident footprint to one expert's weights).
        auto gate_w = gate_weights[expert_idx].to(torch::kFloat32);  // [inter, hidden]
        auto up_w = up_weights[expert_idx].to(torch::kFloat32);      // [inter, hidden]
        auto down_w = down_weights[expert_idx].to(torch::kFloat32);  // [hidden, inter]

        // Gather this expert's tokens and routing weights.
        auto expert_input = hidden_states.index_select(/*dim=*/0, token_indices);  // [n_e, hidden]
        auto expert_weights_vals = topk_weights.index({token_indices, weight_indices});  // [n_e]

        // Recompute forward activations for this expert.
        auto gate_pre_act = at::linear(expert_input, gate_w);   // [n_e, inter]
        auto gate_output = at::silu(gate_pre_act);              // [n_e, inter]
        auto up_output = at::linear(expert_input, up_w);        // [n_e, inter]
        auto intermediate = gate_output * up_output;            // [n_e, inter]
        auto expert_output = at::linear(intermediate, down_w);  // [n_e, hidden]

        // Gather gradient from output (reverse of scatter-add).
        auto grad_weighted_output = grad_output.index_select(/*dim=*/0, token_indices);  // [n_e, hidden]

        // Gradient through the per-token routing-weight multiply.
        auto grad_expert_output = grad_weighted_output * expert_weights_vals.unsqueeze(-1);  // [n_e, hidden]
        auto grad_expert_weights = (grad_weighted_output * expert_output).sum(/*dim=*/-1);   // [n_e]

        // Accumulate gradient for topk_weights at the routed (token, slot) cells.
        grad_topk_w.index_put_({token_indices, weight_indices}, grad_expert_weights, /*accumulate=*/true);

        // Gradient through the down projection.
        auto grad_intermediate = at::linear(grad_expert_output, down_w.t());  // [n_e, inter]
        grad_down_w[expert_idx] += at::matmul(grad_expert_output.t(), intermediate);  // [hidden, inter]

        // Gradient through the element-wise SwiGLU multiply.
        auto grad_gate_output = grad_intermediate * up_output;  // [n_e, inter]
        auto grad_up_output = grad_intermediate * gate_output;  // [n_e, inter]

        // Gradient through SiLU: silu(x) = x * sigmoid(x),
        // d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
        auto sigmoid_gate = at::sigmoid(gate_pre_act);
        auto silu_grad = sigmoid_gate * (1.0 + gate_pre_act * (1.0 - sigmoid_gate));
        auto grad_gate_pre_act = grad_gate_output * silu_grad;  // [n_e, inter]

        // Gradient through the gate projection.
        auto grad_expert_input_gate = at::linear(grad_gate_pre_act, gate_w.t());  // [n_e, hidden]
        grad_gate_w[expert_idx] += at::matmul(grad_gate_pre_act.t(), expert_input);  // [inter, hidden]

        // Gradient through the up projection.
        auto grad_expert_input_up = at::linear(grad_up_output, up_w.t());  // [n_e, hidden]
        grad_up_w[expert_idx] += at::matmul(grad_up_output.t(), expert_input);  // [inter, hidden]

        // Combine the two contributions to the expert input.
        auto grad_expert_input = grad_expert_input_gate + grad_expert_input_up;  // [n_e, hidden]

        // Scatter gradient back to the original token rows (reverse of gather).
        grad_hidden.index_add_(/*dim=*/0, token_indices, grad_expert_input);
    }

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_hidden_states.sizes() == grad_hidden.sizes(),
                "grad_hidden_states shape ", grad_hidden_states.sizes(),
                " does not match computed ", grad_hidden.sizes());
    TORCH_CHECK(grad_topk_weights.sizes() == grad_topk_w.sizes(),
                "grad_topk_weights shape ", grad_topk_weights.sizes(),
                " does not match computed ", grad_topk_w.sizes());
    TORCH_CHECK(grad_gate_weights.sizes() == grad_gate_w.sizes(),
                "grad_gate_weights shape ", grad_gate_weights.sizes(),
                " does not match computed ", grad_gate_w.sizes());
    TORCH_CHECK(grad_up_weights.sizes() == grad_up_w.sizes(),
                "grad_up_weights shape ", grad_up_weights.sizes(),
                " does not match computed ", grad_up_w.sizes());
    TORCH_CHECK(grad_down_weights.sizes() == grad_down_w.sizes(),
                "grad_down_weights shape ", grad_down_weights.sizes(),
                " does not match computed ", grad_down_w.sizes());

    grad_hidden_states.copy_(grad_hidden);
    grad_topk_weights.copy_(grad_topk_w);
    grad_gate_weights.copy_(grad_gate_w);
    grad_up_weights.copy_(grad_up_w);
    grad_down_weights.copy_(grad_down_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for MoE expert-parallel SwiGLU dispatch (expert count "
        "derived from the projection tensors). Pure ATen dispatch: per-expert "
        "host loop over the fixed expert count gathers each expert's tokens "
        "(one_hot / permute / where / "
        "index_select), recomputes the forward activations (gate / up "
        "projections, SiLU, SwiGLU product, down projection) in float32, then "
        "walks the analytic backward (routing-weight multiply, down-projection "
        "grad, SwiGLU-multiply grad, SiLU derivative, gate / up projection "
        "grads) and scatters the input gradient with index_add_, reproduced "
        "op-for-op from the reference. Five outputs (destination-passing "
        "style): grad_hidden_states, grad_topk_weights, grad_gate_weights, "
        "grad_up_weights, grad_down_weights, native float32 throughout.");
}
