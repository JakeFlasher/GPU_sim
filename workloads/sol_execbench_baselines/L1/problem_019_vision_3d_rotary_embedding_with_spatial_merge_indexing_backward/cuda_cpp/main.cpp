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
//   grad_q_embed     : [seq_len, num_heads=16, head_dim=128] fp32
//   grad_k_embed     : [seq_len, num_heads, head_dim]        fp32
//   q                : [seq_len, num_heads, head_dim]        fp32
//   k                : [seq_len, num_heads, head_dim]        fp32
//   embeddings       : [seq_len, head_dim]                   fp32
//   grad_q           : same as q  (DPS)
//   grad_k           : same as k  (DPS)
//   grad_embeddings  : same as embeddings (DPS)
//
// head_dim = 128 and num_heads = 16 are pinned. seq_len is
// workload-variable.
void run(
    const torch::Tensor& grad_q_embed,
    const torch::Tensor& grad_k_embed,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& embeddings,
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_embeddings
) {
    TORCH_CHECK(grad_q_embed.dim() == 3,
                "grad_q_embed must be 3D [S, H, D]");
    TORCH_CHECK(grad_k_embed.dim() == 3,
                "grad_k_embed must be 3D [S, H, D]");
    TORCH_CHECK(q.dim() == 3, "q must be 3D [S, H, D]");
    TORCH_CHECK(k.dim() == 3, "k must be 3D [S, H, D]");
    TORCH_CHECK(embeddings.dim() == 2,
                "embeddings must be 2D [S, D]");
    TORCH_CHECK(grad_q.dim() == 3, "grad_q must be 3D");
    TORCH_CHECK(grad_k.dim() == 3, "grad_k must be 3D");
    TORCH_CHECK(grad_embeddings.dim() == 2,
                "grad_embeddings must be 2D");

    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumHeads = 16;
    TORCH_CHECK(grad_q_embed.size(2) == kHeadDim,
                "head_dim must be ", kHeadDim,
                " for problem_019, got ", grad_q_embed.size(2));
    TORCH_CHECK(grad_q_embed.size(1) == kNumHeads,
                "num_heads must be ", kNumHeads,
                " for problem_019, got ", grad_q_embed.size(1));
    TORCH_CHECK(q.sizes() == grad_q_embed.sizes(),
                "q shape must match grad_q_embed");
    TORCH_CHECK(k.sizes() == grad_k_embed.sizes(),
                "k shape must match grad_k_embed");
    TORCH_CHECK(grad_q.sizes() == grad_q_embed.sizes(),
                "grad_q shape must match grad_q_embed");
    TORCH_CHECK(grad_k.sizes() == grad_k_embed.sizes(),
                "grad_k shape must match grad_k_embed");
    TORCH_CHECK(embeddings.size(0) == grad_q_embed.size(0) &&
                embeddings.size(1) == kHeadDim,
                "embeddings must be [seq_len, head_dim]");
    TORCH_CHECK(grad_embeddings.sizes() == embeddings.sizes(),
                "grad_embeddings shape must match embeddings");
    TORCH_CHECK(grad_q_embed.size(0) == grad_k_embed.size(0),
                "Q and K must agree on seq_len");

    check_fp32_contiguous_cuda(grad_q_embed, "grad_q_embed");
    check_fp32_contiguous_cuda(grad_k_embed, "grad_k_embed");
    check_fp32_contiguous_cuda(q, "q");
    check_fp32_contiguous_cuda(k, "k");
    check_fp32_contiguous_cuda(embeddings, "embeddings");
    check_fp32_contiguous_cuda(grad_q, "grad_q");
    check_fp32_contiguous_cuda(grad_k, "grad_k");
    check_fp32_contiguous_cuda(grad_embeddings, "grad_embeddings");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_019_grad_x_launcher(grad_q_embed, embeddings, grad_q, stream);
    problem_019_grad_x_launcher(grad_k_embed, embeddings, grad_k, stream);
    problem_019_grad_emb_launcher(
        grad_q_embed, grad_k_embed, q, k, embeddings,
        grad_embeddings, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 Vision 3D RoPE backward: per-element grad_q / grad_k "
        "(two launches of one launcher) + per-(s) reduction "
        "grad_embeddings (one launch). head_dim=128, num_heads=16 "
        "pinned.");
}
