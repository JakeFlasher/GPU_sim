#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Compute 3D position IDs for multi-modal rotary embeddings.
//
// Inputs:
//   input_ids          [B, S]                 int64
//   image_grid_thw     [total_num_images, 3]  int64
//   video_grid_thw     [total_num_videos, 3]  int64
//   second_per_grid_ts [total_num_videos]     float32
//   attention_mask     [B, S]                 int64
//   image_count        [B]                    int64  (per-batch vision_start->image_token events)
//   video_count        [B]                    int64  (per-batch vision_start->video_token events)
//   image_offset       [B]                    int64  (exclusive prefix of image_count)
//   video_offset       [B]                    int64  (exclusive prefix of video_count)
//   valid_lens         [B]                    int64  (sum(attention_mask, dim=1))
//
// Outputs (destination-passing-style, pre-allocated by the evaluator):
//   position_ids           [3, B, S]  int64
//   mrope_position_deltas  [B, 1]     int64
//
// The kernel processes each batch with one block; host-side PyTorch
// reductions provide the per-batch vision-event counts and offsets so
// the device side is a deterministic emit pass that exactly mirrors the
// reference's text-span + 3D-vision-span position formulas.
void problem_006_multimodal_rope_position_calculation_launcher(
    const torch::Tensor& input_ids,
    const torch::Tensor& image_grid_thw,
    const torch::Tensor& video_grid_thw,
    const torch::Tensor& second_per_grid_ts,
    const torch::Tensor& attention_mask,
    const torch::Tensor& image_count,
    const torch::Tensor& video_count,
    const torch::Tensor& image_offset,
    const torch::Tensor& video_offset,
    const torch::Tensor& valid_lens,
    torch::Tensor& position_ids,
    torch::Tensor& mrope_position_deltas,
    cudaStream_t stream);
