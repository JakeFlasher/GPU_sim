#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "kernel.h"

// Destination-passing-style entry per definition.json:
//   key_states         : [batch, num_kv_heads, new_seq_len, head_dim]      bf16
//   value_states       : [batch, num_kv_heads, new_seq_len, head_dim]      bf16
//   cos                : [batch, 1, new_seq_len, head_dim]                 bf16
//   sin                : [batch, 1, new_seq_len, head_dim]                 bf16
//   key_cache          : [batch, num_kv_heads, current_seq_len, head_dim]  bf16
//   value_cache        : [batch, num_kv_heads, current_seq_len, head_dim]  bf16
//   updated_key_cache  : [batch, num_kv_heads, updated_seq_len, head_dim]  bf16 (out)
//   updated_value_cache: [batch, num_kv_heads, updated_seq_len, head_dim]  bf16 (out)
// updated_seq_len = current_seq_len + new_seq_len.

void run(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor updated_key_cache,
    torch::Tensor updated_value_cache) {

    TORCH_CHECK(key_states.is_cuda(),         "key_states must be CUDA");
    TORCH_CHECK(value_states.is_cuda(),       "value_states must be CUDA");
    TORCH_CHECK(cos.is_cuda(),                "cos must be CUDA");
    TORCH_CHECK(sin.is_cuda(),                "sin must be CUDA");
    TORCH_CHECK(key_cache.is_cuda(),          "key_cache must be CUDA");
    TORCH_CHECK(value_cache.is_cuda(),        "value_cache must be CUDA");
    TORCH_CHECK(updated_key_cache.is_cuda(),  "updated_key_cache must be CUDA");
    TORCH_CHECK(updated_value_cache.is_cuda(),"updated_value_cache must be CUDA");

    TORCH_CHECK(key_states.dtype() == torch::kBFloat16,         "key_states must be bf16");
    TORCH_CHECK(value_states.dtype() == torch::kBFloat16,       "value_states must be bf16");
    TORCH_CHECK(cos.dtype() == torch::kBFloat16,                "cos must be bf16");
    TORCH_CHECK(sin.dtype() == torch::kBFloat16,                "sin must be bf16");
    TORCH_CHECK(key_cache.dtype() == torch::kBFloat16,          "key_cache must be bf16");
    TORCH_CHECK(value_cache.dtype() == torch::kBFloat16,        "value_cache must be bf16");
    TORCH_CHECK(updated_key_cache.dtype() == torch::kBFloat16,  "updated_key_cache must be bf16");
    TORCH_CHECK(updated_value_cache.dtype() == torch::kBFloat16,"updated_value_cache must be bf16");

    TORCH_CHECK(key_states.is_contiguous(),         "key_states must be contiguous");
    TORCH_CHECK(value_states.is_contiguous(),       "value_states must be contiguous");
    TORCH_CHECK(cos.is_contiguous(),                "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(),                "sin must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(),          "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(),        "value_cache must be contiguous");
    TORCH_CHECK(updated_key_cache.is_contiguous(),  "updated_key_cache must be contiguous");
    TORCH_CHECK(updated_value_cache.is_contiguous(),"updated_value_cache must be contiguous");

    TORCH_CHECK(key_states.dim() == 4, "key_states must be 4D");
    TORCH_CHECK(key_cache.dim() == 4,  "key_cache must be 4D");

    const auto B = key_states.size(0);
    const auto H = key_states.size(1);
    const auto new_seq_len = key_states.size(2);
    const auto D = key_states.size(3);
    const auto current_seq_len = key_cache.size(2);
    const auto updated_seq_len = current_seq_len + new_seq_len;

    TORCH_CHECK(D % 2 == 0, "head_dim must be even (RoPE rotate_half splits in half)");
    TORCH_CHECK(value_states.size(0) == B && value_states.size(1) == H &&
                value_states.size(2) == new_seq_len && value_states.size(3) == D,
                "value_states must match key_states shape");
    TORCH_CHECK(cos.size(0) == B && cos.size(1) == 1 &&
                cos.size(2) == new_seq_len && cos.size(3) == D,
                "cos must have shape [batch, 1, new_seq_len, head_dim]");
    TORCH_CHECK(sin.size(0) == B && sin.size(1) == 1 &&
                sin.size(2) == new_seq_len && sin.size(3) == D,
                "sin must have shape [batch, 1, new_seq_len, head_dim]");
    TORCH_CHECK(key_cache.size(0) == B && key_cache.size(1) == H && key_cache.size(3) == D,
                "key_cache must have shape [batch, num_kv_heads, *, head_dim]");
    TORCH_CHECK(value_cache.size(0) == B && value_cache.size(1) == H &&
                value_cache.size(2) == current_seq_len && value_cache.size(3) == D,
                "value_cache must match key_cache shape");
    TORCH_CHECK(updated_key_cache.size(0) == B && updated_key_cache.size(1) == H &&
                updated_key_cache.size(2) == updated_seq_len && updated_key_cache.size(3) == D,
                "updated_key_cache must have shape [batch, num_kv_heads, updated_seq_len, head_dim]");
    TORCH_CHECK(updated_value_cache.size(0) == B && updated_value_cache.size(1) == H &&
                updated_value_cache.size(2) == updated_seq_len && updated_value_cache.size(3) == D,
                "updated_value_cache must match updated_key_cache shape");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_071_kv_cache_update_with_rope_launcher(
        key_states, value_states, cos, sin,
        key_cache, value_cache,
        updated_key_cache, updated_value_cache, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Fused KV cache update with RoPE on new K states. One CTA per "
          "(batch, kv_head, output_seq_position); first current_seq_len "
          "positions copy from the existing cache, last new_seq_len "
          "positions apply RoPE to new K and copy V raw; head_dim threads "
          "per block, one element each.");
}
