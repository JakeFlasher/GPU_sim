#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "kernel.h"

// Destination-passing-style entry per definition.json:
//   query              : [batch, num_attention_heads, seq_len, head_dim]   bf16
//   key                : [batch, num_kv_heads, seq_len, head_dim]          bf16
//   value              : [batch, num_kv_heads, seq_len, head_dim]          bf16
//   position_ids       : [batch, seq_len]                                  int64
//   key_cache          : [batch, num_kv_heads, max_pos, head_dim]          bf16
//   value_cache        : [batch, num_kv_heads, max_pos, head_dim]          bf16
//   cache_position     : [seq_len]                                         int64
//   q_norm_weight      : [head_dim]                                        bf16
//   k_norm_weight      : [head_dim]                                        bf16
//   inv_freq           : [half_head_dim]                                   fp32
//   rms_norm_eps       : scalar                                            fp32
//   query_rotated      : [batch, num_attention_heads, seq_len, head_dim]   bf16 (out)
//   key_rotated        : [batch, num_kv_heads, seq_len, head_dim]          bf16 (out)
//   key_cache_out      : [batch, num_kv_heads, max_pos, head_dim]          bf16 (out)
//   value_cache_out    : [batch, num_kv_heads, max_pos, head_dim]          bf16 (out)
//
// head_dim is pinned at 128, num_attention_heads = 96, num_kv_heads = 8,
// max_position_embeddings = 262144 per the problem's const axes.

void run(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor position_ids,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor inv_freq,
    double rms_norm_eps,
    torch::Tensor query_rotated,
    torch::Tensor key_rotated,
    torch::Tensor key_cache_out,
    torch::Tensor value_cache_out) {

    TORCH_CHECK(query.is_cuda(),            "query must be CUDA");
    TORCH_CHECK(key.is_cuda(),              "key must be CUDA");
    TORCH_CHECK(value.is_cuda(),            "value must be CUDA");
    TORCH_CHECK(position_ids.is_cuda(),     "position_ids must be CUDA");
    TORCH_CHECK(key_cache.is_cuda(),        "key_cache must be CUDA");
    TORCH_CHECK(value_cache.is_cuda(),      "value_cache must be CUDA");
    TORCH_CHECK(cache_position.is_cuda(),   "cache_position must be CUDA");
    TORCH_CHECK(q_norm_weight.is_cuda(),    "q_norm_weight must be CUDA");
    TORCH_CHECK(k_norm_weight.is_cuda(),    "k_norm_weight must be CUDA");
    TORCH_CHECK(inv_freq.is_cuda(),         "inv_freq must be CUDA");
    TORCH_CHECK(query_rotated.is_cuda(),    "query_rotated must be CUDA");
    TORCH_CHECK(key_rotated.is_cuda(),      "key_rotated must be CUDA");
    TORCH_CHECK(key_cache_out.is_cuda(),    "key_cache_out must be CUDA");
    TORCH_CHECK(value_cache_out.is_cuda(),  "value_cache_out must be CUDA");

    TORCH_CHECK(query.dtype()           == torch::kBFloat16, "query must be bf16");
    TORCH_CHECK(key.dtype()             == torch::kBFloat16, "key must be bf16");
    TORCH_CHECK(value.dtype()           == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(position_ids.dtype()    == torch::kInt64,    "position_ids must be int64");
    TORCH_CHECK(key_cache.dtype()       == torch::kBFloat16, "key_cache must be bf16");
    TORCH_CHECK(value_cache.dtype()     == torch::kBFloat16, "value_cache must be bf16");
    TORCH_CHECK(cache_position.dtype()  == torch::kInt64,    "cache_position must be int64");
    TORCH_CHECK(q_norm_weight.dtype()   == torch::kBFloat16, "q_norm_weight must be bf16");
    TORCH_CHECK(k_norm_weight.dtype()   == torch::kBFloat16, "k_norm_weight must be bf16");
    TORCH_CHECK(inv_freq.dtype()        == torch::kFloat32,  "inv_freq must be fp32");
    TORCH_CHECK(query_rotated.dtype()   == torch::kBFloat16, "query_rotated must be bf16");
    TORCH_CHECK(key_rotated.dtype()     == torch::kBFloat16, "key_rotated must be bf16");
    TORCH_CHECK(key_cache_out.dtype()   == torch::kBFloat16, "key_cache_out must be bf16");
    TORCH_CHECK(value_cache_out.dtype() == torch::kBFloat16, "value_cache_out must be bf16");

    // position_ids is constructed by the reference get_inputs() via
    // `arange(...).unsqueeze(0).expand(batch_size, -1)`, which yields a
    // (B, S) broadcast view with stride (0, 1) for B > 1. The kernel
    // expects (S, 1) strides, so materialize a contiguous copy at the
    // boundary; this also covers any other non-contiguous input the
    // harness might pass.
    auto position_ids_c = position_ids.contiguous();
    TORCH_CHECK(query.is_contiguous(),           "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(),             "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(),           "value must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(),       "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(),     "value_cache must be contiguous");
    TORCH_CHECK(cache_position.is_contiguous(),  "cache_position must be contiguous");
    TORCH_CHECK(q_norm_weight.is_contiguous(),   "q_norm_weight must be contiguous");
    TORCH_CHECK(k_norm_weight.is_contiguous(),   "k_norm_weight must be contiguous");
    TORCH_CHECK(inv_freq.is_contiguous(),        "inv_freq must be contiguous");
    TORCH_CHECK(query_rotated.is_contiguous(),   "query_rotated must be contiguous");
    TORCH_CHECK(key_rotated.is_contiguous(),     "key_rotated must be contiguous");
    TORCH_CHECK(key_cache_out.is_contiguous(),   "key_cache_out must be contiguous");
    TORCH_CHECK(value_cache_out.is_contiguous(), "value_cache_out must be contiguous");

    TORCH_CHECK(query.dim() == 4, "query must be 4D");
    TORCH_CHECK(query.size(3) == 128,
                "head_dim must be 128 (pinned by the problem definition); got ",
                query.size(3));
    TORCH_CHECK(inv_freq.size(0) == 64,
                "inv_freq must have half_head_dim=64 entries; got ",
                inv_freq.size(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_018_fused_rope_with_qk_norm_and_kv_cache_update_launcher(
        query, key, value, position_ids_c,
        key_cache, value_cache, cache_position,
        q_norm_weight, k_norm_weight, inv_freq,
        static_cast<float>(rms_norm_eps),
        query_rotated, key_rotated, key_cache_out, value_cache_out,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Fused per-head RMS norm + RoPE on Q/K + scatter into the "
          "KV cache. One CTA per (batch, head, seq) for RMS+RoPE "
          "(128 threads per CTA — head_dim wide); cudaMemcpyAsync "
          "clones key_cache / value_cache into the output buffers, "
          "and two scatter kernels write the rotated K and raw V into "
          "the cache_position slots.");
}
