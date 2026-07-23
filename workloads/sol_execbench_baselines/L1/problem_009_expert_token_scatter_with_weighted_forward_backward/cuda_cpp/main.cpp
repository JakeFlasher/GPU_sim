#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for expert token scatter with weighted forward (MoE SwiGLU).
//
// Reference (problem_009_expert_token_scatter_with_weighted_forward_backward):
// the analytic backward written out by hand as explicit tensor algebra
// (NO torch.autograd). Every grad is computed in float32 and cast back to
// bfloat16 for the five destination buffers. Reproduced op-for-op below.
//
// Per definition.json (insertion order):
//   inputs:
//     grad_output      : [batch_seq_len, hidden_dim]   bfloat16
//     token_indices    : [num_tokens]                  int64
//     selected_tokens  : [num_tokens, hidden_dim]      bfloat16
//     w1_output        : [num_tokens, ffn_dim]         bfloat16
//     gate_output      : [num_tokens, ffn_dim]         bfloat16  (PRECOMPUTED silu(w1_output))
//     up_output        : [num_tokens, ffn_dim]         bfloat16
//     gated_output     : [num_tokens, ffn_dim]         bfloat16  (PRECOMPUTED gate*up)
//     expert_output    : [num_tokens, hidden_dim]      bfloat16
//     selected_weights : [num_tokens]                  bfloat16
//     w1_weight        : [ffn_dim, hidden_dim]         bfloat16
//     w2_weight        : [hidden_dim, ffn_dim]         bfloat16
//     w3_weight        : [ffn_dim, hidden_dim]         bfloat16
//   outputs (all bfloat16, DPS pre-alloc, written in place):
//     grad_hidden_states   : [batch_seq_len, hidden_dim]
//     grad_routing_weights : [batch_seq_len]
//     grad_w1_weight       : [ffn_dim, hidden_dim]
//     grad_w2_weight       : [hidden_dim, ffn_dim]
//     grad_w3_weight       : [ffn_dim, hidden_dim]
//
// Forward chain the backward inverts (gate = silu branch, up = w3 branch):
//   w1_output       = selected_tokens @ w1_weight.T
//   gate_output     = silu(w1_output)
//   up_output       = selected_tokens @ w3_weight.T
//   gated_output    = gate_output * up_output
//   expert_output   = gated_output @ w2_weight.T
//   weighted_output = expert_output * selected_weights.unsqueeze(-1)
//   then scattered: full[token_indices] = weighted_output  (grad gathers/scatters)
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// .t().mm() / .mm() calls route through the SAME cuBLAS GEMM the reference's
// torch ops use (bf16 inputs upcast to fp32 here, fp32 accum), so they match
// the reference's float32 algebra. The gather is grad_output_f[token_indices]
// (index_select on dim 0); the two scatters write into ZERO-INIT full-size
// fp32 tensors via index_put_ with accumulate=false -- the reference uses
// plain assignment `t[token_indices] = ...` (index_copy semantics, NOT
// index_add), and token_indices are unique by construction, so the
// non-accumulating scatter is the faithful op. The SiLU derivative and every
// .to(float32)/.to(bfloat16) cast point match the reference exactly.
// gate_output / up_output / gated_output are consumed as precomputed inputs
// (never recomputed).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& token_indices,
    const torch::Tensor& selected_tokens,
    const torch::Tensor& w1_output,
    const torch::Tensor& gate_output,
    const torch::Tensor& up_output,
    const torch::Tensor& gated_output,
    const torch::Tensor& expert_output,
    const torch::Tensor& selected_weights,
    const torch::Tensor& w1_weight,
    const torch::Tensor& w2_weight,
    const torch::Tensor& w3_weight,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_routing_weights,
    torch::Tensor grad_w1_weight,
    torch::Tensor grad_w2_weight,
    torch::Tensor grad_w3_weight
) {
    check_bf16_cuda(grad_output, "grad_output");
    check_bf16_cuda(selected_tokens, "selected_tokens");
    check_bf16_cuda(w1_output, "w1_output");
    check_bf16_cuda(gate_output, "gate_output");
    check_bf16_cuda(up_output, "up_output");
    check_bf16_cuda(gated_output, "gated_output");
    check_bf16_cuda(expert_output, "expert_output");
    check_bf16_cuda(selected_weights, "selected_weights");
    check_bf16_cuda(w1_weight, "w1_weight");
    check_bf16_cuda(w2_weight, "w2_weight");
    check_bf16_cuda(w3_weight, "w3_weight");
    check_bf16_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_cuda(grad_routing_weights, "grad_routing_weights");
    check_bf16_cuda(grad_w1_weight, "grad_w1_weight");
    check_bf16_cuda(grad_w2_weight, "grad_w2_weight");
    check_bf16_cuda(grad_w3_weight, "grad_w3_weight");

    TORCH_CHECK(token_indices.is_cuda(), "token_indices must be a CUDA tensor");
    TORCH_CHECK(token_indices.dtype() == torch::kInt64,
                "token_indices must be int64, got ", token_indices.dtype());

    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D, got ", grad_output.dim(), "D");
    TORCH_CHECK(token_indices.dim() == 1, "token_indices must be 1D, got ", token_indices.dim(), "D");
    TORCH_CHECK(selected_tokens.dim() == 2, "selected_tokens must be 2D, got ", selected_tokens.dim(), "D");
    TORCH_CHECK(w1_output.dim() == 2, "w1_output must be 2D, got ", w1_output.dim(), "D");
    TORCH_CHECK(gate_output.dim() == 2, "gate_output must be 2D, got ", gate_output.dim(), "D");
    TORCH_CHECK(up_output.dim() == 2, "up_output must be 2D, got ", up_output.dim(), "D");
    TORCH_CHECK(gated_output.dim() == 2, "gated_output must be 2D, got ", gated_output.dim(), "D");
    TORCH_CHECK(expert_output.dim() == 2, "expert_output must be 2D, got ", expert_output.dim(), "D");
    TORCH_CHECK(selected_weights.dim() == 1, "selected_weights must be 1D, got ", selected_weights.dim(), "D");
    TORCH_CHECK(w1_weight.dim() == 2, "w1_weight must be 2D, got ", w1_weight.dim(), "D");
    TORCH_CHECK(w2_weight.dim() == 2, "w2_weight must be 2D, got ", w2_weight.dim(), "D");
    TORCH_CHECK(w3_weight.dim() == 2, "w3_weight must be 2D, got ", w3_weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    const auto kF32 = torch::kFloat32;
    const auto kBF16 = torch::kBFloat16;

    const int64_t batch_seq_len = grad_output.size(0);
    const int64_t hidden_dim = grad_output.size(1);

    // Convert to float32 for numerical stability (mirrors the reference).
    auto grad_output_f = grad_output.to(kF32);
    auto selected_tokens_f = selected_tokens.to(kF32);
    auto w1_output_f = w1_output.to(kF32);
    auto gate_output_f = gate_output.to(kF32);
    auto up_output_f = up_output.to(kF32);
    auto gated_output_f = gated_output.to(kF32);
    auto expert_output_f = expert_output.to(kF32);
    auto selected_weights_f = selected_weights.to(kF32);
    auto w1_weight_f = w1_weight.to(kF32);
    auto w2_weight_f = w2_weight.to(kF32);
    auto w3_weight_f = w3_weight.to(kF32);

    // Stage 1: gather gradients from the scattered positions.
    //   grad_weighted_output = grad_output_f[token_indices]   -> (num_tokens, hidden_dim)
    auto grad_weighted_output = grad_output_f.index_select(/*dim=*/0, token_indices);

    // Stage 2: backprop through routing weight multiplication.
    //   weighted_output = expert_output * selected_weights.unsqueeze(-1)
    //   grad_selected_weights = (grad_weighted_output * expert_output).sum(dim=-1)
    auto grad_selected_weights =
        (grad_weighted_output * expert_output_f).sum(/*dim=*/-1);  // (num_tokens,)

    // Scatter to full routing_weights tensor (zero-init, then assign at indices).
    //   grad_routing_weights = zeros(batch_seq_len)
    //   grad_routing_weights[token_indices] = grad_selected_weights
    auto grad_routing_weights_f =
        at::zeros({batch_seq_len}, grad_output_f.options());
    grad_routing_weights_f.index_put_({token_indices}, grad_selected_weights);

    // Gradient w.r.t. expert_output.
    //   grad_expert_output = grad_weighted_output * selected_weights.unsqueeze(-1)
    auto grad_expert_output =
        grad_weighted_output * selected_weights_f.unsqueeze(-1);  // (num_tokens, hidden_dim)

    // Stage 3: backprop through w2 linear layer.
    //   expert_output = gated_output @ w2_weight.T
    //   grad_w2_weight = grad_expert_output.T @ gated_output   -> (hidden_dim, ffn_dim)
    auto grad_w2_weight_f = grad_expert_output.t().mm(gated_output_f);

    //   grad_gated_output = grad_expert_output @ w2_weight     -> (num_tokens, ffn_dim)
    auto grad_gated_output = grad_expert_output.mm(w2_weight_f);

    // Stage 4: backprop through element-wise gating (gated_output = gate_output * up_output).
    auto grad_gate_output = grad_gated_output * up_output_f;  // (num_tokens, ffn_dim)
    auto grad_up_output = grad_gated_output * gate_output_f;  // (num_tokens, ffn_dim)

    // Stage 5: backprop through w3 linear layer (up branch).
    //   up_output = selected_tokens @ w3_weight.T
    //   grad_w3_weight = grad_up_output.T @ selected_tokens    -> (ffn_dim, hidden_dim)
    auto grad_w3_weight_f = grad_up_output.t().mm(selected_tokens_f);
    //   grad_selected_tokens_w3 = grad_up_output @ w3_weight   -> (num_tokens, hidden_dim)
    auto grad_selected_tokens_w3 = grad_up_output.mm(w3_weight_f);

    // Stage 6: backprop through SiLU activation and w1 linear layer (gate branch).
    //   gate_output = silu(w1_output) = w1_output * sigmoid(w1_output)
    //   d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    auto sigmoid_w1 = at::sigmoid(w1_output_f);
    auto grad_w1_output =
        grad_gate_output * (sigmoid_w1 * (1.0 + w1_output_f * (1.0 - sigmoid_w1)));  // (num_tokens, ffn_dim)

    //   w1_output = selected_tokens @ w1_weight.T
    //   grad_w1_weight = grad_w1_output.T @ selected_tokens    -> (ffn_dim, hidden_dim)
    auto grad_w1_weight_f = grad_w1_output.t().mm(selected_tokens_f);
    //   grad_selected_tokens_w1 = grad_w1_output @ w1_weight   -> (num_tokens, hidden_dim)
    auto grad_selected_tokens_w1 = grad_w1_output.mm(w1_weight_f);

    // Stage 7: accumulate gradients for selected_tokens (reference order: w1 + w3).
    auto grad_selected_tokens = grad_selected_tokens_w1 + grad_selected_tokens_w3;  // (num_tokens, hidden_dim)

    // Stage 8: scatter gradients back to the full hidden_states tensor
    // (zero-init, then assign at indices).
    //   grad_hidden_states = zeros(batch_seq_len, hidden_dim)
    //   grad_hidden_states[token_indices] = grad_selected_tokens
    auto grad_hidden_states_f =
        at::zeros({batch_seq_len, hidden_dim}, grad_output_f.options());
    grad_hidden_states_f.index_put_({token_indices}, grad_selected_tokens);

    // Cast each result back to bfloat16 and write into the DPS buffers.
    auto grad_hidden_states_bf16 = grad_hidden_states_f.to(kBF16);
    auto grad_routing_weights_bf16 = grad_routing_weights_f.to(kBF16);
    auto grad_w1_weight_bf16 = grad_w1_weight_f.to(kBF16);
    auto grad_w2_weight_bf16 = grad_w2_weight_f.to(kBF16);
    auto grad_w3_weight_bf16 = grad_w3_weight_f.to(kBF16);

    TORCH_CHECK(grad_hidden_states_bf16.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states result shape ", grad_hidden_states_bf16.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_routing_weights_bf16.sizes() == grad_routing_weights.sizes(),
                "grad_routing_weights result shape ", grad_routing_weights_bf16.sizes(),
                " does not match pre-allocated ", grad_routing_weights.sizes());
    TORCH_CHECK(grad_w1_weight_bf16.sizes() == grad_w1_weight.sizes(),
                "grad_w1_weight result shape ", grad_w1_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_w1_weight.sizes());
    TORCH_CHECK(grad_w2_weight_bf16.sizes() == grad_w2_weight.sizes(),
                "grad_w2_weight result shape ", grad_w2_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_w2_weight.sizes());
    TORCH_CHECK(grad_w3_weight_bf16.sizes() == grad_w3_weight.sizes(),
                "grad_w3_weight result shape ", grad_w3_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_w3_weight.sizes());

    grad_hidden_states.copy_(grad_hidden_states_bf16);
    grad_routing_weights.copy_(grad_routing_weights_bf16);
    grad_w1_weight.copy_(grad_w1_weight_bf16);
    grad_w2_weight.copy_(grad_w2_weight_bf16);
    grad_w3_weight.copy_(grad_w3_weight_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for expert token scatter with weighted forward "
        "(MoE SwiGLU). Pure ATen / cuBLAS dispatch: analytic grad algebra "
        "(gather grad_output by token_indices, routing-weight grad, w2 / "
        "gating / w3 / SiLU+w1 grad-matmuls, zero-init scatter back via "
        "index_put_) reproduced op-for-op from the reference, fp32 compute "
        "with bf16 in/out, precomputed gate_output / up_output / gated_output "
        "consumed directly. Destination-passing style (5 outputs).");
}
