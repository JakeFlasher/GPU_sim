#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json:
//   position_ids       : [batch, seq_len]                int64
//   inv_freq           : [half_head_dim]                 float32  (llama3-scaled)
//   attention_scaling  : scalar                          float32
//   cos_sin (output)   : [batch, seq_len, head_dim, 2]   bfloat16 (DPS)
// head_dim is pinned at 128, so half_head_dim = 64.

void run(
    torch::Tensor position_ids,
    torch::Tensor inv_freq,
    double attention_scaling,
    torch::Tensor cos_sin) {

    TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
    TORCH_CHECK(inv_freq.is_cuda(),     "inv_freq must be a CUDA tensor");
    TORCH_CHECK(cos_sin.is_cuda(),      "cos_sin must be a CUDA tensor");

    TORCH_CHECK(
        position_ids.dtype() == torch::kInt64,
        "position_ids must be int64");
    TORCH_CHECK(
        inv_freq.dtype() == torch::kFloat32,
        "inv_freq must be float32");
    TORCH_CHECK(
        cos_sin.dtype() == torch::kBFloat16,
        "cos_sin must be bfloat16");

    TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");
    TORCH_CHECK(inv_freq.is_contiguous(),     "inv_freq must be contiguous");
    TORCH_CHECK(cos_sin.is_contiguous(),      "cos_sin must be contiguous");

    TORCH_CHECK(position_ids.dim() == 2, "position_ids must be 2D");
    TORCH_CHECK(inv_freq.dim() == 1,     "inv_freq must be 1D");
    TORCH_CHECK(cos_sin.dim() == 4,      "cos_sin must be 4D");

    TORCH_CHECK(
        inv_freq.size(0) == 64,
        "inv_freq must have 64 elements (head_dim=128/2), got ",
        inv_freq.size(0));
    TORCH_CHECK(
        cos_sin.size(2) == 128 && cos_sin.size(3) == 2,
        "cos_sin must have shape [batch, seq, 128, 2]");
    TORCH_CHECK(
        cos_sin.size(0) == position_ids.size(0) &&
        cos_sin.size(1) == position_ids.size(1),
        "cos_sin batch/seq dims must match position_ids");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_011_rotary_position_embedding_launcher(
        position_ids,
        inv_freq,
        static_cast<float>(attention_scaling),
        cos_sin,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "RoPE cos/sin embedding computation; one CTA per (batch, seq) "
        "pair, half_head_dim threads each writing a duplicated (cos, "
        "sin) pair to head_dim slots; destination-passing style.");
}
