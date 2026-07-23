#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

namespace {

constexpr int64_t IMAGE_TOKEN_ID = 151655LL;
constexpr int64_t VIDEO_TOKEN_ID = 151656LL;
constexpr int64_t VISION_START_TOKEN_ID = 151652LL;

static void check_int64_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.dtype() == torch::kInt64,
        name, " must have int64 dtype, got ", t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Global no-vision branch: total_num_images == 0 AND total_num_videos == 0.
// Mirrors the reference's else-branch:
//   position_ids_1d = cumsum(mask, -1) - 1
//   position_ids_1d[mask == 0] = 1
//   position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1)
//   max_position_ids = pos1d.max(-1, keepdim=True)
//   mrope_position_deltas = max_position_ids + 1 - seq_len
static void run_no_vision_branch(
    const torch::Tensor& attention_mask,
    torch::Tensor position_ids,
    torch::Tensor mrope_position_deltas,
    int64_t seq_len) {
    auto mask_long = attention_mask.to(torch::kLong);
    auto pos1d = mask_long.cumsum(/*dim=*/-1) - 1;
    pos1d.masked_fill_(attention_mask == 0, 1);
    // position_ids[3, B, S] <- copy pos1d (broadcast over the leading 3).
    position_ids.copy_(pos1d.unsqueeze(0).expand({3, -1, -1}));
    auto max_pos_ids = std::get<0>(pos1d.max(/*dim=*/-1, /*keepdim=*/true));
    mrope_position_deltas.copy_(max_pos_ids + 1 - seq_len);
}

}  // namespace

// Destination-passing-style entry. The definition declares:
//   input_ids          [B, S]                 int64
//   image_grid_thw     [total_num_images, 3]  int64
//   video_grid_thw     [total_num_videos, 3]  int64
//   second_per_grid_ts [total_num_videos]     float32
//   attention_mask     [B, S]                 int64
//   position_ids       [3, B, S]              int64 (DPS, pre-alloc)
//   mrope_position_deltas [B, 1]              int64 (DPS, pre-alloc)
//
// Constants per definition.json axes section:
//   spatial_merge_size       = 2
//   image_token_id           = 151655
//   video_token_id           = 151656
//   vision_start_token_id    = 151652
//   tokens_per_second        = 2.0 (reference scalar)
void run(
    const torch::Tensor& input_ids,
    const torch::Tensor& image_grid_thw,
    const torch::Tensor& video_grid_thw,
    const torch::Tensor& second_per_grid_ts,
    const torch::Tensor& attention_mask,
    torch::Tensor position_ids,
    torch::Tensor mrope_position_deltas) {

    TORCH_CHECK(input_ids.dim() == 2, "input_ids must be 2D [B, S]; got ", input_ids.dim(), "D");
    TORCH_CHECK(attention_mask.dim() == 2, "attention_mask must be 2D [B, S]");
    TORCH_CHECK(
        attention_mask.sizes() == input_ids.sizes(),
        "attention_mask shape must match input_ids");
    TORCH_CHECK(image_grid_thw.dim() == 2 && image_grid_thw.size(1) == 3,
        "image_grid_thw must be [total_num_images, 3]");
    TORCH_CHECK(video_grid_thw.dim() == 2 && video_grid_thw.size(1) == 3,
        "video_grid_thw must be [total_num_videos, 3]");

    check_int64_cuda(input_ids, "input_ids");
    check_int64_cuda(attention_mask, "attention_mask");
    check_int64_cuda(image_grid_thw, "image_grid_thw");
    check_int64_cuda(video_grid_thw, "video_grid_thw");
    check_int64_cuda(position_ids, "position_ids");
    check_int64_cuda(mrope_position_deltas, "mrope_position_deltas");
    TORCH_CHECK(second_per_grid_ts.is_cuda(), "second_per_grid_ts must be CUDA");
    TORCH_CHECK(
        second_per_grid_ts.dtype() == torch::kFloat32,
        "second_per_grid_ts must be float32");

    const int64_t B = input_ids.size(0);
    const int64_t S = input_ids.size(1);
    const int64_t total_num_images = image_grid_thw.size(0);
    const int64_t total_num_videos = video_grid_thw.size(0);

    if (total_num_images == 0 && total_num_videos == 0) {
        run_no_vision_branch(attention_mask, position_ids, mrope_position_deltas, S);
        return;
    }

    // Per-batch vision-event counts:
    //   image_count[b] = #(t : t in [0, S-1) and input_ids[b,t] == VISION_START_TOKEN_ID
    //                                    and input_ids[b,t+1] == IMAGE_TOKEN_ID)
    //   video_count[b] = analogous with VIDEO_TOKEN_ID
    // The reference uses the same definition: vision_tokens =
    // input_id_seq[vision_start_indices + 1] then count == image/video token id.
    auto vs_mask = (input_ids == VISION_START_TOKEN_ID);   // [B, S]
    auto next_ids = input_ids.slice(/*dim=*/1, /*start=*/1, /*end=*/S);  // [B, S-1]
    auto vs_prev = vs_mask.slice(/*dim=*/1, /*start=*/0, /*end=*/S - 1);  // [B, S-1]
    auto img_match = vs_prev & (next_ids == IMAGE_TOKEN_ID);  // [B, S-1] bool
    auto vid_match = vs_prev & (next_ids == VIDEO_TOKEN_ID);
    auto image_count = img_match.to(torch::kLong).sum(/*dim=*/1);  // [B]
    auto video_count = vid_match.to(torch::kLong).sum(/*dim=*/1);  // [B]

    // Exclusive prefix offsets in batch order: image_offset[b] = sum(image_count[0..b-1])
    auto image_cum = image_count.cumsum(/*dim=*/0);
    auto image_offset = image_cum - image_count;
    auto video_cum = video_count.cumsum(/*dim=*/0);
    auto video_offset = video_cum - video_count;

    // valid_lens[b] = sum(attention_mask[b]) — host-computed once instead of
    // recomputed inside the kernel.
    auto valid_lens = attention_mask.sum(/*dim=*/1);  // [B] int64

    // position_ids is allocated by the evaluator (likely via torch.empty),
    // so its initial contents are undefined. The kernel writes every
    // position in [0, S) per batch: emitted positions in [0, out_idx) plus
    // a default-1 fill in [out_idx, S) mirroring the reference's initial
    // torch.ones() over positions the algorithm did not emit.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_006_multimodal_rope_position_calculation_launcher(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts,
        attention_mask,
        image_count,
        video_count,
        image_offset,
        video_offset,
        valid_lens,
        position_ids,
        mrope_position_deltas,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Multimodal-rope 3D position-IDs kernel. Computes per-batch text + "
        "vision-grid position spans using host-derived vision-event counts "
        "and prefix offsets. Output: [3, B, S] int64 position_ids + [B, 1] "
        "int64 mrope_position_deltas.");
}
