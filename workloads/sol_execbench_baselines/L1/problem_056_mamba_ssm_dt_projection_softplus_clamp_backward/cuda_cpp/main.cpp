#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
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
//   grad_output     : [batch_size, seq_len, num_heads=40] bf16
//   dt_with_bias    : [batch_size, seq_len, num_heads]     bf16
//   dt_activated    : [batch_size, seq_len, num_heads]     bf16
//   time_step_min   : scalar fp32
//   time_step_max   : scalar fp32
//   grad_dt         : [batch_size, seq_len, num_heads]     bf16 (DPS)
//   grad_dt_bias    : [num_heads]                          bf16 (DPS)
//
// num_heads = 40 is pinned; batch_size and seq_len are
// workload-variable.
void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& dt_with_bias,
    const torch::Tensor& dt_activated,
    double time_step_min,
    double time_step_max,
    torch::Tensor grad_dt,
    torch::Tensor grad_dt_bias
) {
    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must be 3D [B, S, H]");
    TORCH_CHECK(dt_with_bias.dim() == 3,
                "dt_with_bias must be 3D [B, S, H]");
    TORCH_CHECK(dt_activated.dim() == 3,
                "dt_activated must be 3D [B, S, H]");
    TORCH_CHECK(grad_dt.dim() == 3, "grad_dt must be 3D");
    TORCH_CHECK(grad_dt_bias.dim() == 1,
                "grad_dt_bias must be 1D [H]");

    constexpr int64_t kNumHeads = 40;
    TORCH_CHECK(grad_output.size(2) == kNumHeads,
                "num_heads must be ", kNumHeads,
                " for problem_056, got ", grad_output.size(2));
    TORCH_CHECK(dt_with_bias.sizes() == grad_output.sizes(),
                "dt_with_bias shape must match grad_output");
    TORCH_CHECK(dt_activated.sizes() == grad_output.sizes(),
                "dt_activated shape must match grad_output");
    TORCH_CHECK(grad_dt.sizes() == grad_output.sizes(),
                "grad_dt shape must match grad_output");
    TORCH_CHECK(grad_dt_bias.size(0) == kNumHeads,
                "grad_dt_bias must have num_heads elements");

    check_bf16_contiguous_cuda(grad_output, "grad_output");
    check_bf16_contiguous_cuda(dt_with_bias, "dt_with_bias");
    check_bf16_contiguous_cuda(dt_activated, "dt_activated");
    check_bf16_contiguous_cuda(grad_dt, "grad_dt");
    check_bf16_contiguous_cuda(grad_dt_bias, "grad_dt_bias");

    const float ts_min = static_cast<float>(time_step_min);
    const float ts_max = static_cast<float>(time_step_max);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_056_mamba_softplus_clamp_backward_launcher(
        grad_output, dt_with_bias, dt_activated,
        ts_min, ts_max,
        grad_dt, grad_dt_bias, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16 Mamba SSM dt-projection softplus + clamp backward: "
        "per-head CTA computes per-element grad_dt and "
        "fp32-summed-then-bf16-cast grad_dt_bias. num_heads=40 pinned.");
}
