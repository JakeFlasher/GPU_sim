#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. definition.json declares:
//   x      : [batch_size, seq_len, intermediate_size] fp32
//   output : [batch_size, seq_len, intermediate_size] fp32 (DPS)
//
// intermediate_size is constant at 4096 per definition.json;
// batch_size and seq_len are variable.
void run(
    const torch::Tensor& x,
    torch::Tensor output
) {
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, seq, intermediate]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    TORCH_CHECK(
        x.sizes() == output.sizes(),
        "output shape must match x");

    check_fp32_contiguous_cuda(x, "x");
    check_fp32_contiguous_cuda(output, "output");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_025_video_latent_gelu_launcher(x, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "FP32 GELU (tanh approximation) element-wise on video latent "
        "tensors. Destination-passing style; float4-vectorized "
        "load/store along the contiguous last dim.");
}
