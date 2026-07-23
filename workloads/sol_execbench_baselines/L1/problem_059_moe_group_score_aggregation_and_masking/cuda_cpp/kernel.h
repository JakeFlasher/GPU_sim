#pragma once

#ifndef PROBLEM_059_MOE_GROUP_SCORE_AGGREGATION_KERNEL_H
#define PROBLEM_059_MOE_GROUP_SCORE_AGGREGATION_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher for the fp32 MoE group-score aggregation + masking
// kernel. Pinned constants from the problem definition:
//   num_experts       = 256
//   n_group           = 8
//   topk_group        = 4
//   experts_per_group = 32
//
// For each token row t in [0, num_tokens):
//   group_scores[t, g] = sum_top2_over_lanes(scores[t, g*32 + lane])
//                        for g in [0, n_group)
//   selected_groups[t] = top-topk_group indices of group_scores[t, :]
//   group_mask[t, g] = 1.0 if g in selected_groups[t] else 0.0
//   masked_scores[t, e] = scores[t, e] if group_mask[t, e/32] == 1
//                         else -inf
//
// One CTA per token (Grid: (num_tokens,)); BLOCK_THREADS =
// num_experts = 256, so each thread handles exactly one expert.
// 8 warps (one per group) handle per-group top-2 via two
// warp-shuffle find-max passes; cross-warp top-4 runs on the
// 8 group_scores from a single warp (warp 0) via four
// find-argmax passes.
void problem_059_moe_group_score_aggregation_launcher(
    const torch::Tensor& scores,
    torch::Tensor& masked_scores,
    torch::Tensor& group_mask,
    cudaStream_t stream);

#endif  // PROBLEM_059_MOE_GROUP_SCORE_AGGREGATION_KERNEL_H
