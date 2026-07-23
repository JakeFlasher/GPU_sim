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

// Destination-passing-style entry. Per definition.json:
//   grad_q_embed : [batch_size, num_heads,    seq_len, head_dim=128] fp32
//   grad_k_embed : [batch_size, num_kv_heads, seq_len, head_dim]      fp32
//   q            : [batch_size, num_heads,    seq_len, head_dim]      fp32
//   k            : [batch_size, num_kv_heads, seq_len, head_dim]      fp32
//   cos          : [batch_size, seq_len, head_dim]                    fp32
//   sin          : [batch_size, seq_len, head_dim]                    fp32
//   grad_q       : same shape as q  (DPS)
//   grad_k       : same shape as k  (DPS)
//   grad_cos     : [batch_size, seq_len, head_dim]                    fp32 (DPS)
//   grad_sin     : [batch_size, seq_len, head_dim]                    fp32 (DPS)
//
// head_dim = 128 is pinned. num_heads, num_kv_heads, seq_len, and
// batch_size are workload-variable. The kernel set runs three
// launches on the current PyTorch CUDA stream:
//   1. problem_068_grad_x_launcher(grad_q_embed, cos, sin, grad_q)
//   2. problem_068_grad_x_launcher(grad_k_embed, cos, sin, grad_k)
//   3. problem_068_grad_cos_sin_launcher(grad_q_embed, grad_k_embed,
//      q, k, grad_cos, grad_sin)
// Kernels 1 and 2 share the same per-element template (head axis is
// determined by the input tensor's size(1)). Kernel 3 reduces over
// both H and H_kv inside a single CTA per (b, s).
void run(
    const torch::Tensor& grad_q_embed,
    const torch::Tensor& grad_k_embed,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_cos,
    torch::Tensor grad_sin
) {
    TORCH_CHECK(grad_q_embed.dim() == 4,
                "grad_q_embed must be 4D [B, H, S, D]");
    TORCH_CHECK(grad_k_embed.dim() == 4,
                "grad_k_embed must be 4D [B, H_kv, S, D]");
    TORCH_CHECK(q.dim() == 4, "q must be 4D");
    TORCH_CHECK(k.dim() == 4, "k must be 4D");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D [B, S, D]");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D [B, S, D]");
    TORCH_CHECK(grad_q.dim() == 4, "grad_q must be 4D");
    TORCH_CHECK(grad_k.dim() == 4, "grad_k must be 4D");
    TORCH_CHECK(grad_cos.dim() == 3, "grad_cos must be 3D");
    TORCH_CHECK(grad_sin.dim() == 3, "grad_sin must be 3D");

    constexpr int64_t kHeadDim = 128;
    TORCH_CHECK(grad_q_embed.size(3) == kHeadDim,
                "head_dim must be 128 for problem_068, got ",
                grad_q_embed.size(3));
    TORCH_CHECK(grad_k_embed.size(3) == kHeadDim,
                "grad_k_embed head_dim must be 128");
    TORCH_CHECK(q.sizes() == grad_q_embed.sizes(),
                "q shape must match grad_q_embed");
    TORCH_CHECK(k.sizes() == grad_k_embed.sizes(),
                "k shape must match grad_k_embed");
    TORCH_CHECK(grad_q.sizes() == grad_q_embed.sizes(),
                "grad_q shape must match grad_q_embed");
    TORCH_CHECK(grad_k.sizes() == grad_k_embed.sizes(),
                "grad_k shape must match grad_k_embed");
    TORCH_CHECK(cos.size(2) == kHeadDim && sin.size(2) == kHeadDim,
                "cos/sin last dim must be head_dim=128");
    TORCH_CHECK(cos.size(0) == grad_q_embed.size(0) &&
                cos.size(1) == grad_q_embed.size(2),
                "cos must have shape [B, S, D] matching grad_q_embed");
    TORCH_CHECK(sin.size(0) == grad_q_embed.size(0) &&
                sin.size(1) == grad_q_embed.size(2),
                "sin must have shape [B, S, D] matching grad_q_embed");
    TORCH_CHECK(grad_cos.sizes() == cos.sizes(),
                "grad_cos shape must match cos");
    TORCH_CHECK(grad_sin.sizes() == sin.sizes(),
                "grad_sin shape must match sin");
    TORCH_CHECK(grad_q_embed.size(0) == grad_k_embed.size(0) &&
                grad_q_embed.size(2) == grad_k_embed.size(2),
                "Q and K must agree on batch_size and seq_len");

    check_fp32_contiguous_cuda(grad_q_embed, "grad_q_embed");
    check_fp32_contiguous_cuda(grad_k_embed, "grad_k_embed");
    check_fp32_contiguous_cuda(q, "q");
    check_fp32_contiguous_cuda(k, "k");
    check_fp32_contiguous_cuda(cos, "cos");
    check_fp32_contiguous_cuda(sin, "sin");
    check_fp32_contiguous_cuda(grad_q, "grad_q");
    check_fp32_contiguous_cuda(grad_k, "grad_k");
    check_fp32_contiguous_cuda(grad_cos, "grad_cos");
    check_fp32_contiguous_cuda(grad_sin, "grad_sin");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_068_grad_x_launcher(grad_q_embed, cos, sin, grad_q, stream);
    problem_068_grad_x_launcher(grad_k_embed, cos, sin, grad_k, stream);
    problem_068_grad_cos_sin_launcher(
        grad_q_embed, grad_k_embed, q, k, grad_cos, grad_sin, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 fused RoPE-application backward: grad_q / grad_k via "
        "per-element rotate-half-with-cos/sin (two launches sharing "
        "one launcher); grad_cos / grad_sin via per-(b, s) reduction "
        "over both H and H_kv (one launch). head_dim=128 pinned.");
}
