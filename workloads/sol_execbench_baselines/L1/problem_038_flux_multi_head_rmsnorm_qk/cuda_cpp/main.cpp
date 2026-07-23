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
//   query      : [batch_size, seq_len, num_heads=48, head_dim=128] fp32
//   key        : [batch_size, seq_len, num_heads=48, head_dim=128] fp32
//   weight_q   : [num_heads=48, head_dim=128]  fp32
//   weight_k   : [num_heads=48, head_dim=128]  fp32
//   eps        : scalar fp32
//   query_norm : [batch_size, seq_len, num_heads, head_dim] fp32 (DPS)
//   key_norm   : [batch_size, seq_len, num_heads, head_dim] fp32 (DPS)
//
// head_dim and num_heads are pinned. The same kernel runs twice:
// once for (query, weight_q, query_norm), once for
// (key, weight_k, key_norm).
void run(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_k,
    double eps,
    torch::Tensor query_norm,
    torch::Tensor key_norm
) {
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B, S, H, D]");
    TORCH_CHECK(key.dim() == 4, "key must be 4D [B, S, H, D]");
    TORCH_CHECK(weight_q.dim() == 2, "weight_q must be 2D [H, D]");
    TORCH_CHECK(weight_k.dim() == 2, "weight_k must be 2D [H, D]");
    TORCH_CHECK(query_norm.dim() == 4, "query_norm must be 4D");
    TORCH_CHECK(key_norm.dim() == 4, "key_norm must be 4D");

    const int64_t head_dim = query.size(3);
    const int64_t num_heads = query.size(2);
    TORCH_CHECK(head_dim == 128,
                "head_dim must be 128 for problem_038, got ", head_dim);
    TORCH_CHECK(num_heads == 48,
                "num_heads must be 48 for problem_038, got ", num_heads);
    TORCH_CHECK(key.size(3) == head_dim,
                "key head_dim must match query head_dim");
    TORCH_CHECK(key.size(2) == num_heads,
                "key num_heads must match query num_heads");
    TORCH_CHECK(key.size(1) == query.size(1),
                "key seq_len must match query seq_len");
    TORCH_CHECK(key.size(0) == query.size(0),
                "key batch_size must match query batch_size");
    TORCH_CHECK(weight_q.size(0) == num_heads &&
                weight_q.size(1) == head_dim,
                "weight_q must be [num_heads, head_dim]");
    TORCH_CHECK(weight_k.size(0) == num_heads &&
                weight_k.size(1) == head_dim,
                "weight_k must be [num_heads, head_dim]");
    TORCH_CHECK(query_norm.sizes() == query.sizes(),
                "query_norm shape must match query");
    TORCH_CHECK(key_norm.sizes() == key.sizes(),
                "key_norm shape must match key");

    check_fp32_contiguous_cuda(query, "query");
    check_fp32_contiguous_cuda(key, "key");
    check_fp32_contiguous_cuda(weight_q, "weight_q");
    check_fp32_contiguous_cuda(weight_k, "weight_k");
    check_fp32_contiguous_cuda(query_norm, "query_norm");
    check_fp32_contiguous_cuda(key_norm, "key_norm");

    const float eps_f32 = static_cast<float>(eps);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_038_flux_multi_head_rmsnorm_qk_launcher(
        query, weight_q, query_norm, eps_f32, stream);
    problem_038_flux_multi_head_rmsnorm_qk_launcher(
        key, weight_k, key_norm, eps_f32, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 per-head RMSNorm-QK: normalizes query and key tensors "
        "over head_dim with per-head learnable scales. Two distinct "
        "DPS outputs (query_norm, key_norm). num_heads=48, "
        "head_dim=128 pinned.");
}
