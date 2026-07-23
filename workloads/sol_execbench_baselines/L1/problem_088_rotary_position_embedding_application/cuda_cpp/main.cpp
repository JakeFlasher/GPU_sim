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
//   query          : [batch_size, num_heads,    seq_len, head_dim=128]  fp32
//   key            : [batch_size, num_kv_heads, seq_len, head_dim=128]  fp32
//   cos            : [seq_len, head_dim]  fp32
//   sin            : [seq_len, head_dim]  fp32
//   query_rotated  : [batch_size, num_heads,    seq_len, head_dim]  fp32 (DPS)
//   key_rotated    : [batch_size, num_kv_heads, seq_len, head_dim]  fp32 (DPS)
//
// head_dim is pinned at 128. The same kernel runs twice: once for
// (query, query_rotated), once for (key, key_rotated). num_heads and
// num_kv_heads are workload-variable.
void run(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    torch::Tensor query_rotated,
    torch::Tensor key_rotated
) {
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B, H, S, D]");
    TORCH_CHECK(key.dim() == 4, "key must be 4D [B, H_kv, S, D]");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2D [S, D]");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D [S, D]");
    TORCH_CHECK(query_rotated.dim() == 4, "query_rotated must be 4D");
    TORCH_CHECK(key_rotated.dim() == 4, "key_rotated must be 4D");

    const int64_t head_dim = query.size(3);
    TORCH_CHECK(head_dim == 128,
                "head_dim must be 128 for problem_088, got ", head_dim);
    TORCH_CHECK(key.size(3) == head_dim,
                "key head_dim must match query head_dim");
    TORCH_CHECK(cos.size(1) == head_dim &&
                sin.size(1) == head_dim,
                "cos/sin last dim must be head_dim");
    TORCH_CHECK(query_rotated.sizes() == query.sizes(),
                "query_rotated shape must match query");
    TORCH_CHECK(key_rotated.sizes() == key.sizes(),
                "key_rotated shape must match key");
    TORCH_CHECK(cos.size(0) == query.size(2) &&
                sin.size(0) == query.size(2),
                "cos/sin seq_len must match query seq_len");
    TORCH_CHECK(key.size(2) == query.size(2),
                "key seq_len must match query seq_len");
    TORCH_CHECK(key.size(0) == query.size(0),
                "key batch_size must match query batch_size");

    check_fp32_contiguous_cuda(query, "query");
    check_fp32_contiguous_cuda(key, "key");
    check_fp32_contiguous_cuda(cos, "cos");
    check_fp32_contiguous_cuda(sin, "sin");
    check_fp32_contiguous_cuda(query_rotated, "query_rotated");
    check_fp32_contiguous_cuda(key_rotated, "key_rotated");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_088_rotary_position_embedding_application_launcher(
        query, cos, sin, query_rotated, stream);
    problem_088_rotary_position_embedding_application_launcher(
        key, cos, sin, key_rotated, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 RoPE application: rotates query and key tensors by "
        "precomputed cos/sin tables. Two distinct DPS outputs "
        "(query_rotated, key_rotated). head_dim=128 pinned.");
}
