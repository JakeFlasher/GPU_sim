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

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. definition.json declares:
//   freqs             : [batch_size, seq_len, half_head_dim] fp32
//   attention_scaling : scalar fp32
//   cos               : [batch_size, seq_len, head_dim] bf16 (DPS)
//   sin               : [batch_size, seq_len, head_dim] bf16 (DPS)
//
// head_dim is constant at 128 per definition.json; half_head_dim
// is the derived axis head_dim // 2 = 64.
void run(
    const torch::Tensor& freqs,
    double attention_scaling,
    torch::Tensor cos,
    torch::Tensor sin
) {
    TORCH_CHECK(freqs.dim() == 3, "freqs must be 3D [batch, seq, half_head_dim]");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D");

    const int64_t half_head_dim = freqs.size(2);
    const int64_t head_dim = cos.size(2);
    TORCH_CHECK(
        head_dim == 2 * half_head_dim,
        "head_dim must equal 2 * half_head_dim; got head_dim=",
        head_dim, " half_head_dim=", half_head_dim);
    TORCH_CHECK(
        cos.size(0) == freqs.size(0) && cos.size(1) == freqs.size(1),
        "cos shape must match freqs leading dims");
    TORCH_CHECK(
        sin.size(0) == freqs.size(0) && sin.size(1) == freqs.size(1),
        "sin shape must match freqs leading dims");
    TORCH_CHECK(
        sin.size(2) == head_dim,
        "sin head_dim must match cos head_dim");

    check_fp32_contiguous_cuda(freqs, "freqs");
    check_bf16_contiguous_cuda(cos, "cos");
    check_bf16_contiguous_cuda(sin, "sin");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_012_fused_cos_sin_launcher(
        freqs,
        static_cast<float>(attention_scaling),
        cos,
        sin,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused cos/sin embedding generation for RoPE: "
        "emb = cat(freqs, freqs); cos = bf16(cos(emb) * scale); "
        "sin = bf16(sin(emb) * scale). Destination-passing style.");
}
