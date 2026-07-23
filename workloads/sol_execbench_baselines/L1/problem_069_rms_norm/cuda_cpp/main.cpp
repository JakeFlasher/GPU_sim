#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

// Tensor-shape validators shared across input checks.
static void check_bf16_contiguous(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. The definition declares:
//   hidden_states: [batch_size, seq_len, hidden_size] bf16
//   residual    : [batch_size, seq_len, hidden_size] bf16
//   weight      : [hidden_size]                       bf16
//   eps         : scalar fp32
//   output      : [batch_size, seq_len, hidden_size] bf16  (DPS, pre-alloc)
//
// hidden_size is constant at 8192 per definition.json.
void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps,
    torch::Tensor output
) {
    // Note on the eps argument type: definition.json declares eps with
    // shape=null and dtype=float32. The upstream eval driver passes it
    // as a Python float, which PyBind11 binds as ``double`` (the safe
    // host-side scalar type). We accept the wider type and downcast on
    // entry to the kernel launcher; the kernel itself does its FP32
    // arithmetic in float.
    TORCH_CHECK(
        hidden_states.dim() == 3,
        "hidden_states must be 3D [batch, seq, hidden]; got ",
        hidden_states.dim(), "D");
    TORCH_CHECK(
        residual.sizes() == hidden_states.sizes(),
        "residual shape must match hidden_states");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");

    const int64_t hidden_size = hidden_states.size(2);
    TORCH_CHECK(
        hidden_size == 8192,
        "hidden_size must be 8192 for problem_069, got ", hidden_size);
    TORCH_CHECK(
        weight.size(0) == hidden_size,
        "weight must have size ", hidden_size, ", got ", weight.size(0));

    check_bf16_contiguous(hidden_states, "hidden_states");
    check_bf16_contiguous(residual, "residual");
    check_bf16_contiguous(weight, "weight");
    check_bf16_contiguous(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_069_rms_norm_launcher(
        hidden_states,
        residual,
        weight,
        static_cast<float>(eps),
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused residual-add + RMSNorm; hidden_size=8192; BF16 inputs / output, "
        "FP32 accumulator; destination-passing style.");
}
