#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Backward pass for the fused FFN block with tanh-GELU activation, residual
// connection, and layer normalization.
//
// Reference (problem_042_ffn_gelu_projection_fused_backward): the analytic
// backward written out by hand as explicit tensor algebra (NO torch.autograd).
// Everything stays in the native float32 dtype the reference uses, so there
// are no cast points to track. Reproduced op-for-op below.
//
// Forward chain the backward inverts:
//   fc1_output      = hidden_states @ fc1_weight.T
//   gelu_output     = 0.5 * fc1_output * (1 + tanh(inner))            (tanh GELU)
//                     with inner = sqrt(2/pi) * (fc1_output + 0.044715 * fc1_output^3)
//   fc2_output      = gelu_output @ fc2_weight.T
//   residual_output = fc2_output + hidden_states
//   normalized      = (residual_output - mean) / sqrt(var + eps)
//   output          = normalized * ln_weight + ln_bias
//
// Per definition.json (insertion order):
//   inputs  (all float32):
//     grad_output     : [batch_size, seq_len, hidden_size]
//     hidden_states   : [batch_size, seq_len, hidden_size]
//     fc1_weight      : [intermediate_size, hidden_size]
//     fc1_output      : [batch_size, seq_len, intermediate_size]   (saved)
//     gelu_output     : [batch_size, seq_len, intermediate_size]   (saved)
//     fc2_weight      : [hidden_size, intermediate_size]
//     residual_output : [batch_size, seq_len, hidden_size]         (saved)
//     normalized      : [batch_size, seq_len, hidden_size]         (saved)
//     var             : [batch_size, seq_len, 1]                   (saved)
//     ln_weight       : [hidden_size]
//     eps             : scalar (layer-norm epsilon)
//   outputs (all float32, DPS pre-alloc, written in place):
//     grad_hidden_states : [batch_size, seq_len, hidden_size]
//     grad_fc1_weight    : [intermediate_size, hidden_size]
//     grad_fc1_bias      : [intermediate_size]
//     grad_fc2_weight    : [hidden_size, intermediate_size]
//     grad_fc2_bias      : [hidden_size]
//     grad_ln_weight     : [hidden_size]
//     grad_ln_bias       : [hidden_size]
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// .t().mm() / .mm() calls route through the same cuBLAS GEMM the reference's
// torch ops use; the layer-norm backward algebra, the tanh-GELU derivative,
// and every reshape / bias-sum orientation match the reference exactly. The
// saved forward intermediates (fc1_output, gelu_output, normalized, var,
// ln_weight) are consumed directly and never recomputed.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_output,
    const torch::Tensor& gelu_output,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& residual_output,
    const torch::Tensor& normalized,
    const torch::Tensor& var,
    const torch::Tensor& ln_weight,
    double eps,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_fc1_weight,
    torch::Tensor grad_fc1_bias,
    torch::Tensor grad_fc2_weight,
    torch::Tensor grad_fc2_bias,
    torch::Tensor grad_ln_weight,
    torch::Tensor grad_ln_bias
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(fc1_weight, "fc1_weight");
    check_fp32_cuda(fc1_output, "fc1_output");
    check_fp32_cuda(gelu_output, "gelu_output");
    check_fp32_cuda(fc2_weight, "fc2_weight");
    check_fp32_cuda(residual_output, "residual_output");
    check_fp32_cuda(normalized, "normalized");
    check_fp32_cuda(var, "var");
    check_fp32_cuda(ln_weight, "ln_weight");
    check_fp32_cuda(grad_hidden_states, "grad_hidden_states");
    check_fp32_cuda(grad_fc1_weight, "grad_fc1_weight");
    check_fp32_cuda(grad_fc1_bias, "grad_fc1_bias");
    check_fp32_cuda(grad_fc2_weight, "grad_fc2_weight");
    check_fp32_cuda(grad_fc2_bias, "grad_fc2_bias");
    check_fp32_cuda(grad_ln_weight, "grad_ln_weight");
    check_fp32_cuda(grad_ln_bias, "grad_ln_bias");

    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must be 3D [B, S, hidden_size], got ", grad_output.dim(), "D");
    TORCH_CHECK(fc1_weight.dim() == 2,
                "fc1_weight must be 2D [intermediate_size, hidden_size], got ", fc1_weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    const int64_t H = grad_output.size(2);
    const int64_t I = fc1_weight.size(0);

    // ----- Backward through layer norm -----
    // output = normalized * ln_weight + ln_bias
    // Gradient w.r.t. ln_weight: sum over batch and sequence.
    auto grad_ln_weight_result = (grad_output * normalized).sum(/*dim=*/std::vector<int64_t>{0, 1});
    // Gradient w.r.t. ln_bias: sum over batch and sequence.
    auto grad_ln_bias_result = grad_output.sum(/*dim=*/std::vector<int64_t>{0, 1});

    // Gradient w.r.t. normalized.
    auto grad_normalized = grad_output * ln_weight;

    // Backward through normalization: normalized = (x - mean) / sqrt(var + eps).
    auto std = at::sqrt(var + eps);
    auto grad_normalized_mean = grad_normalized.mean(/*dim=*/-1, /*keepdim=*/true);
    auto grad_normalized_normalized_mean =
        (grad_normalized * normalized).mean(/*dim=*/-1, /*keepdim=*/true);
    auto grad_residual_output = (1.0 / std) *
        (grad_normalized - grad_normalized_mean - normalized * grad_normalized_normalized_mean);

    // ----- Backward through residual addition -----
    // residual_output = fc2_output + hidden_states
    auto grad_fc2_output = grad_residual_output;
    auto grad_residual = grad_residual_output;

    // ----- Backward through FC2 (contraction layer) -----
    // fc2_output = gelu_output @ fc2_weight.T + fc2_bias
    auto grad_fc2_bias_result = grad_fc2_output.sum(/*dim=*/std::vector<int64_t>{0, 1});

    auto grad_fc2_output_reshaped = grad_fc2_output.reshape({-1, H});
    auto gelu_output_reshaped = gelu_output.reshape({-1, I});
    auto grad_fc2_weight_result = grad_fc2_output_reshaped.t().mm(gelu_output_reshaped);

    auto grad_gelu_output = at::matmul(grad_fc2_output, fc2_weight);

    // ----- Backward through tanh-GELU activation -----
    const double sqrt_2_over_pi = 0.7978845608028654;
    const double coeff = 0.044715;

    auto x = fc1_output;
    auto x_cubed = x * x * x;
    auto tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    auto tanh_out = at::tanh(tanh_arg);

    auto dtanh_arg_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
    auto sech_sq = 1.0 - tanh_out * tanh_out;

    auto gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * dtanh_arg_dx;
    auto grad_fc1_output = grad_gelu_output * gelu_grad;

    // ----- Backward through FC1 (expansion layer) -----
    // fc1_output = hidden_states @ fc1_weight.T + fc1_bias
    auto grad_fc1_bias_result = grad_fc1_output.sum(/*dim=*/std::vector<int64_t>{0, 1});

    auto grad_fc1_output_reshaped = grad_fc1_output.reshape({-1, I});
    auto hidden_states_reshaped = hidden_states.reshape({-1, H});
    auto grad_fc1_weight_result = grad_fc1_output_reshaped.t().mm(hidden_states_reshaped);

    auto grad_hidden_states_fc1 = at::matmul(grad_fc1_output, fc1_weight);

    // ----- Combine gradients for hidden_states -----
    auto grad_hidden_states_result = grad_hidden_states_fc1 + grad_residual;

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_hidden_states_result.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states shape ", grad_hidden_states_result.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_fc1_weight_result.sizes() == grad_fc1_weight.sizes(),
                "grad_fc1_weight shape ", grad_fc1_weight_result.sizes(),
                " does not match pre-allocated ", grad_fc1_weight.sizes());
    TORCH_CHECK(grad_fc1_bias_result.sizes() == grad_fc1_bias.sizes(),
                "grad_fc1_bias shape ", grad_fc1_bias_result.sizes(),
                " does not match pre-allocated ", grad_fc1_bias.sizes());
    TORCH_CHECK(grad_fc2_weight_result.sizes() == grad_fc2_weight.sizes(),
                "grad_fc2_weight shape ", grad_fc2_weight_result.sizes(),
                " does not match pre-allocated ", grad_fc2_weight.sizes());
    TORCH_CHECK(grad_fc2_bias_result.sizes() == grad_fc2_bias.sizes(),
                "grad_fc2_bias shape ", grad_fc2_bias_result.sizes(),
                " does not match pre-allocated ", grad_fc2_bias.sizes());
    TORCH_CHECK(grad_ln_weight_result.sizes() == grad_ln_weight.sizes(),
                "grad_ln_weight shape ", grad_ln_weight_result.sizes(),
                " does not match pre-allocated ", grad_ln_weight.sizes());
    TORCH_CHECK(grad_ln_bias_result.sizes() == grad_ln_bias.sizes(),
                "grad_ln_bias shape ", grad_ln_bias_result.sizes(),
                " does not match pre-allocated ", grad_ln_bias.sizes());

    grad_hidden_states.copy_(grad_hidden_states_result);
    grad_fc1_weight.copy_(grad_fc1_weight_result);
    grad_fc1_bias.copy_(grad_fc1_bias_result);
    grad_fc2_weight.copy_(grad_fc2_weight_result);
    grad_fc2_bias.copy_(grad_fc2_bias_result);
    grad_ln_weight.copy_(grad_ln_weight_result);
    grad_ln_bias.copy_(grad_ln_bias_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for the fused FFN block with tanh-GELU, residual, and "
        "layer norm. Pure ATen / cuBLAS dispatch: the analytic grad algebra "
        "(layer-norm backward, residual split, FC2 contraction, tanh-GELU "
        "derivative, FC1 expansion, hidden-state accumulation) reproduced "
        "op-for-op from the reference, native float32 throughout, saved "
        "forward intermediates consumed directly. Destination-passing style "
        "(7 outputs).");
}
