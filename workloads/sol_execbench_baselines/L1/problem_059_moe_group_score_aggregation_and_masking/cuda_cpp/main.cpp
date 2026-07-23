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
//   scores         : [num_tokens, num_experts=256] fp32
//   masked_scores  : [num_tokens, num_experts=256] fp32 (DPS, -inf masking)
//   group_mask     : [num_tokens, n_group=8] fp32 (DPS, {0.0, 1.0})
//
// num_experts=256, n_group=8, topk_group=4, experts_per_group=32 are
// all pinned by the problem definition. num_tokens is workload-variable.
void run(
    const torch::Tensor& scores,
    torch::Tensor masked_scores,
    torch::Tensor group_mask
) {
    TORCH_CHECK(scores.dim() == 2, "scores must be 2D [num_tokens, num_experts]");
    TORCH_CHECK(masked_scores.dim() == 2, "masked_scores must be 2D");
    TORCH_CHECK(group_mask.dim() == 2, "group_mask must be 2D");

    constexpr int64_t kNumExperts = 256;
    constexpr int64_t kNumGroups = 8;
    TORCH_CHECK(scores.size(1) == kNumExperts,
                "scores must have num_experts=", kNumExperts,
                " columns, got ", scores.size(1));
    TORCH_CHECK(masked_scores.sizes() == scores.sizes(),
                "masked_scores shape must match scores");
    TORCH_CHECK(group_mask.size(0) == scores.size(0) &&
                group_mask.size(1) == kNumGroups,
                "group_mask must be [num_tokens, n_group=", kNumGroups, "]");

    check_fp32_contiguous_cuda(scores, "scores");
    check_fp32_contiguous_cuda(masked_scores, "masked_scores");
    check_fp32_contiguous_cuda(group_mask, "group_mask");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_059_moe_group_score_aggregation_launcher(
        scores, masked_scores, group_mask, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 MoE group-score aggregation + masking: per-group top-2 "
        "sum aggregates 256 expert scores into 8 group scores; "
        "cross-group top-4 selects surviving groups; masked_scores "
        "= scores where group selected, -inf elsewhere. "
        "num_experts=256, n_group=8, topk_group=4 pinned.");
}
