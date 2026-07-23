#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for an SSM output projection with gated multiplication.
//
// Reference (079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward):
// the analytic backward written out by hand as explicit tensor algebra (no
// torch.autograd). Forward was output = (ssm_output * gate_activated) @ weight.T
// + bias, with gate_activated = SiLU(gate) when use_silu_gate else gate. The
// backward reproduced here computes:
//   grad_weight        = grad_output_2d.T @ gated_output_2d        [hidden_dim, expanded_dim]
//   grad_bias          = grad_output_2d.sum(dim=0)                 [hidden_dim]
//   grad_gated_output  = (grad_output_2d @ weight) reshaped to [B, S, expanded_dim]
//   grad_ssm_output    = grad_gated_output * gate_activated
//   grad_gate_activated= grad_gated_output * ssm_output
//   grad_gate          = grad_gate_activated * SiLU'(gate)         (use_silu_gate)
//                      = grad_gate_activated                       (otherwise)
// where SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
//
// The harness-facing slug is the canonical problem_079_ImageNet_83 (the
// benchmark dir name truncated at the first non-identifier character); the
// real benchmark data dir is carried in solution.json.definition
// ("079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward") so the
// evaluator resolves the workloads.
//
// Per definition.json (insertion order):
//   inputs:
//     grad_output    : [batch_size, seq_len, hidden_dim]   bf16
//     ssm_output     : [batch_size, seq_len, expanded_dim] bf16
//     gate           : [batch_size, seq_len, expanded_dim] bf16
//     weight         : [hidden_dim, expanded_dim]          bf16
//     bias           : [hidden_dim]                        bf16  (unused by the backward)
//     gate_activated : [batch_size, seq_len, expanded_dim] bf16
//     gated_output   : [batch_size, seq_len, expanded_dim] bf16
//     use_silu_gate  : scalar bool
//   outputs (all bf16, DPS pre-alloc, written in place):
//     grad_ssm_output : [batch_size, seq_len, expanded_dim]
//     grad_gate       : [batch_size, seq_len, expanded_dim]
//     grad_weight     : [hidden_dim, expanded_dim]
//     grad_bias       : [hidden_dim]
//
// Implementation: pure ATen dispatch, no custom CUDA kernels. The reference
// operates in bfloat16 end-to-end (no .float() anywhere), so this stays in
// bfloat16 to match its precision profile exactly -- the matmuls route through
// the same cuBLAS path the reference's @ uses.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& ssm_output,
    const torch::Tensor& gate,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& gate_activated,
    const torch::Tensor& gated_output,
    bool use_silu_gate,
    torch::Tensor grad_ssm_output,
    torch::Tensor grad_gate,
    torch::Tensor grad_weight,
    torch::Tensor grad_bias
) {
    check_bf16_cuda(grad_output, "grad_output");
    check_bf16_cuda(ssm_output, "ssm_output");
    check_bf16_cuda(gate, "gate");
    check_bf16_cuda(weight, "weight");
    check_bf16_cuda(gate_activated, "gate_activated");
    check_bf16_cuda(gated_output, "gated_output");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must be 3D [B, S, hidden_dim]");
    (void)bias;  // the analytic backward does not use the forward bias

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    const int64_t batch_size = grad_output.size(0);
    const int64_t seq_len = grad_output.size(1);
    const int64_t hidden_dim = grad_output.size(2);
    const int64_t expanded_dim = ssm_output.size(2);

    auto grad_output_2d = grad_output.reshape({-1, hidden_dim});
    auto gated_output_2d = gated_output.reshape({-1, expanded_dim});

    auto grad_weight_result = at::matmul(grad_output_2d.t(), gated_output_2d);
    auto grad_bias_result = grad_output_2d.sum(/*dim=*/0);

    auto grad_gated_output_2d = at::matmul(grad_output_2d, weight);
    auto grad_gated_output =
        grad_gated_output_2d.view({batch_size, seq_len, expanded_dim});

    auto grad_ssm_output_result = grad_gated_output * gate_activated;
    auto grad_gate_activated = grad_gated_output * ssm_output;

    torch::Tensor grad_gate_result;
    if (use_silu_gate) {
        auto sigmoid_gate = at::sigmoid(gate);
        auto silu_grad = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate));
        grad_gate_result = grad_gate_activated * silu_grad;
    } else {
        grad_gate_result = grad_gate_activated;
    }

    TORCH_CHECK(grad_ssm_output_result.sizes() == grad_ssm_output.sizes(),
                "grad_ssm_output shape mismatch");
    TORCH_CHECK(grad_gate_result.sizes() == grad_gate.sizes(),
                "grad_gate shape mismatch");
    TORCH_CHECK(grad_weight_result.sizes() == grad_weight.sizes(),
                "grad_weight shape mismatch");
    TORCH_CHECK(grad_bias_result.sizes() == grad_bias.sizes(),
                "grad_bias shape mismatch");

    grad_ssm_output.copy_(grad_ssm_output_result);
    grad_gate.copy_(grad_gate_result);
    grad_weight.copy_(grad_weight_result);
    grad_bias.copy_(grad_bias_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Backward for SSM output projection with gated multiply; ATen dispatch, "
          "native bfloat16; destination-passing style (4 outputs).");
}
