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

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. Per definition.json:
//   grad_cos        : [batch_size, seq_len, head_dim] bf16
//   grad_sin        : [batch_size, seq_len, head_dim] bf16
//   idx_theta       : [batch_size, seq_len, head_dim] fp32
//   grad_idx_theta  : [batch_size, seq_len, head_dim] fp32 (DPS)
//
// batch_size, seq_len, and head_dim are all workload-variable
// (no pinned constants for this slug).
void run(
    const torch::Tensor& grad_cos,
    const torch::Tensor& grad_sin,
    const torch::Tensor& idx_theta,
    torch::Tensor grad_idx_theta
) {
    TORCH_CHECK(grad_cos.dim() == 3,
                "grad_cos must be 3D [B, S, D]");
    TORCH_CHECK(grad_sin.dim() == 3,
                "grad_sin must be 3D [B, S, D]");
    TORCH_CHECK(idx_theta.dim() == 3,
                "idx_theta must be 3D [B, S, D]");
    TORCH_CHECK(grad_idx_theta.dim() == 3,
                "grad_idx_theta must be 3D [B, S, D]");
    TORCH_CHECK(grad_cos.sizes() == idx_theta.sizes(),
                "grad_cos shape must match idx_theta");
    TORCH_CHECK(grad_sin.sizes() == idx_theta.sizes(),
                "grad_sin shape must match idx_theta");
    TORCH_CHECK(grad_idx_theta.sizes() == idx_theta.sizes(),
                "grad_idx_theta shape must match idx_theta");

    check_bf16_contiguous_cuda(grad_cos, "grad_cos");
    check_bf16_contiguous_cuda(grad_sin, "grad_sin");
    check_fp32_contiguous_cuda(idx_theta, "idx_theta");
    check_fp32_contiguous_cuda(grad_idx_theta, "grad_idx_theta");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_090_batched_2d_rope_backward_launcher(
        grad_cos, grad_sin, idx_theta, grad_idx_theta, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "BF16/fp32 batched 2D RoPE position-encoding backward: "
        "computes grad_idx_theta = -grad_cos.f32 * sin(idx_theta) "
        "+ grad_sin.f32 * cos(idx_theta) per element. No reductions.");
}
