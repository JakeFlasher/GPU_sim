#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry for hybrid attention-mask
// preparation. Per definition.json:
//
//   batch_size_scalar              : int64 0-D scalar
//   seq_length_scalar              : int64 0-D scalar
//   past_key_values_length_scalar  : int64 0-D scalar
//   full_attention_mask            : [B, num_heads=64, T, total_len] bool (DPS)
//   sliding_window_attention_mask  : [B, num_heads=64, T, total_len] bool (DPS)
//
// Const axes pinned by definition.json AND enforced on every
// tensor that the kernel indexes with that fixed assumption:
//   num_attention_heads     = 64    pinned on full_attention_mask
//   swa_num_attention_heads = 64    pinned on sliding_window_attention_mask
//   sliding_window          = 128   passed to launcher as a constant
//                                   (the reference's swa_mask
//                                   masked_fill is a no-op so
//                                   sliding_window does not affect
//                                   the output; see kernel.cu).

namespace {

constexpr int64_t kNumAttentionHeads = 64;
constexpr int64_t kSwaNumAttentionHeads = 64;
constexpr int64_t kSlidingWindow = 128;

}  // namespace

void run(
    int64_t batch_size_scalar,
    int64_t seq_length_scalar,
    int64_t past_key_values_length_scalar,
    torch::Tensor full_attention_mask,
    torch::Tensor sliding_window_attention_mask
) {
    const int64_t batch_size = batch_size_scalar;
    const int64_t seq_length = seq_length_scalar;
    const int64_t past_kv_length = past_key_values_length_scalar;

    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(seq_length > 0, "seq_length must be positive");
    TORCH_CHECK(past_kv_length >= 0,
                "past_kv_length must be non-negative");

    const int64_t total_length = seq_length + past_kv_length;

    TORCH_CHECK(full_attention_mask.is_cuda(),
                "full_attention_mask must be CUDA");
    TORCH_CHECK(sliding_window_attention_mask.is_cuda(),
                "sliding_window_attention_mask must be CUDA");

    TORCH_CHECK(full_attention_mask.dtype() == torch::kBool,
                "full_attention_mask must be bool");
    TORCH_CHECK(sliding_window_attention_mask.dtype() == torch::kBool,
                "sliding_window_attention_mask must be bool");

    TORCH_CHECK(full_attention_mask.is_contiguous(),
                "full_attention_mask must be contiguous");
    TORCH_CHECK(sliding_window_attention_mask.is_contiguous(),
                "sliding_window_attention_mask must be contiguous");

    TORCH_CHECK(full_attention_mask.dim() == 4,
                "full_attention_mask must be 4D "
                "[B, num_heads, T, total_length]");
    TORCH_CHECK(sliding_window_attention_mask.dim() == 4,
                "sliding_window_attention_mask must be 4D "
                "[B, num_heads, T, total_length]");

    TORCH_CHECK(full_attention_mask.size(0) == batch_size,
                "full_attention_mask.size(0) must equal "
                "batch_size_scalar, got ",
                full_attention_mask.size(0));
    TORCH_CHECK(full_attention_mask.size(1) == kNumAttentionHeads,
                "full_attention_mask.size(1) must equal "
                "num_attention_heads=64, got ",
                full_attention_mask.size(1));
    TORCH_CHECK(full_attention_mask.size(2) == seq_length,
                "full_attention_mask.size(2) must equal "
                "seq_length_scalar, got ",
                full_attention_mask.size(2));
    TORCH_CHECK(full_attention_mask.size(3) == total_length,
                "full_attention_mask.size(3) must equal "
                "seq_length + past_key_values_length, got ",
                full_attention_mask.size(3));

    TORCH_CHECK(sliding_window_attention_mask.size(0) == batch_size,
                "sliding_window_attention_mask.size(0) must equal "
                "batch_size_scalar");
    TORCH_CHECK(sliding_window_attention_mask.size(1)
                == kSwaNumAttentionHeads,
                "sliding_window_attention_mask.size(1) must equal "
                "swa_num_attention_heads=64");
    TORCH_CHECK(sliding_window_attention_mask.size(2) == seq_length,
                "sliding_window_attention_mask.size(2) must equal "
                "seq_length_scalar");
    TORCH_CHECK(sliding_window_attention_mask.size(3) == total_length,
                "sliding_window_attention_mask.size(3) must equal "
                "seq_length + past_key_values_length");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_028_hybrid_attention_mask_preparation_launcher(
        batch_size, seq_length, past_kv_length, kSlidingWindow,
        full_attention_mask, sliding_window_attention_mask, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Hybrid attention-mask preparation: full causal mask + "
        "sliding-window mask. One pure-boolean CUDA kernel per "
        "(b, h, t) row writing both outputs in one pass. "
        "num_attention_heads=64, swa_num_attention_heads=64, "
        "sliding_window=128 pinned.");
}
