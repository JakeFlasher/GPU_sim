#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <cmath>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. Per definition.json:
//   grad_output         : [batch_size, seq_len, hidden_size=4096] bf16
//   gate                : scalar bf16 (zero-dim tensor)
//   hidden_states       : [batch_size, seq_len, hidden_size=4096] bf16
//   mask                : [batch_size, seq_len, 1] bf16
//   grad_residual       : [batch_size, seq_len, hidden_size]      bf16 (DPS)
//   grad_hidden_states  : [batch_size, seq_len, hidden_size]      bf16 (DPS)
//   grad_gate           : scalar bf16                              (DPS)
//
// hidden_size = 4096 is pinned; batch_size and seq_len are
// workload-variable.
void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& gate,
    const torch::Tensor& hidden_states,
    const torch::Tensor& mask,
    torch::Tensor grad_residual,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_gate
) {
    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must be 3D [B, S, H]");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, H]");
    TORCH_CHECK(mask.dim() == 3,
                "mask must be 3D [B, S, 1]");
    TORCH_CHECK(grad_residual.dim() == 3,
                "grad_residual must be 3D");
    TORCH_CHECK(grad_hidden_states.dim() == 3,
                "grad_hidden_states must be 3D");
    TORCH_CHECK(gate.dim() == 0,
                "gate must be a zero-dim scalar tensor, got dim=",
                gate.dim());
    TORCH_CHECK(grad_gate.dim() == 0,
                "grad_gate must be a zero-dim scalar tensor, got dim=",
                grad_gate.dim());

    constexpr int64_t kHiddenSize = 4096;
    TORCH_CHECK(grad_output.size(2) == kHiddenSize,
                "hidden_size must be ", kHiddenSize,
                " for problem_061, got ", grad_output.size(2));
    TORCH_CHECK(hidden_states.sizes() == grad_output.sizes(),
                "hidden_states shape must match grad_output");
    TORCH_CHECK(mask.size(0) == grad_output.size(0) &&
                mask.size(1) == grad_output.size(1) &&
                mask.size(2) == 1,
                "mask must have shape [B, S, 1] matching grad_output");
    TORCH_CHECK(grad_residual.sizes() == grad_output.sizes(),
                "grad_residual shape must match grad_output");
    TORCH_CHECK(grad_hidden_states.sizes() == grad_output.sizes(),
                "grad_hidden_states shape must match grad_output");

    check_bf16_contiguous_cuda(grad_output, "grad_output");
    check_bf16_contiguous_cuda(gate, "gate");
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(mask, "mask");
    check_bf16_contiguous_cuda(grad_residual, "grad_residual");
    check_bf16_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_contiguous_cuda(grad_gate, "grad_gate");

    // Host-side scalar evaluation of tanh(gate) and sech²(gate).
    // Gate is a zero-dim BF16 tensor; read its single element to
    // the host as fp32, evaluate the two constants once per
    // kernel invocation, and pass them as float kernel arguments.
    // `item<float>()` performs the scalar device-to-host read directly.  The
    // previous `gate.to(kFloat32).item<float>()` materialized a one-element
    // ATen CUDA conversion kernel inside the profiler gate, which made the
    // native vertical depend on a libtorch control sidecar for no semantic
    // benefit.
    const float gate_f32 = gate.item<float>();
    const float tanh_g = std::tanh(gate_f32);
    const float sech2 = 1.0f - tanh_g * tanh_g;

    // Per-row partial-sum buffer for the grad_gate reduction.
    const int64_t num_rows = grad_output.size(0) * grad_output.size(1);
    auto row_partials = at::empty(
        {num_rows}, grad_output.options().dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Kernel A: per-row grad_hidden_states + row partial sums.
    problem_061_grad_hidden_launcher(
        grad_output, hidden_states, mask,
        tanh_g, grad_residual, grad_hidden_states, row_partials, stream);

    // Kernel B: single-CTA reduction of row_partials -> grad_gate.
    problem_061_grad_gate_reduce_launcher(
        row_partials, sech2, grad_gate, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16 tanh-gated residual-addition backward: identity copy "
        "for grad_residual, per-element transform for "
        "grad_hidden_states, two-pass reduction for the scalar "
        "grad_gate. hidden_size=4096 pinned.");
}
