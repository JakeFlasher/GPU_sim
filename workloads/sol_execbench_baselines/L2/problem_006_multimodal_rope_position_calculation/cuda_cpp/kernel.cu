#include "kernel.h"
#include <cuda_runtime.h>
#include <cstdint>

// Single-batch, single-thread emit kernel for multimodal-rope 3D position IDs.
// One block per batch; thread 0 walks the batch's sequence in lockstep with
// the reference implementation. Other threads are idle — the per-batch work
// is sequential (find next image / video marker, then emit text + 3D-grid
// position spans), so the win over the PyTorch reference comes from
// eliminating the reference's Python loop / .tolist() / .index() / repeated
// torch.cat overhead rather than from intra-batch parallelism.
//
// Vision-event totals (image_count[b], video_count[b]) and global offsets
// (image_offset[b], video_offset[b]) are precomputed on the host with at::
// ops, so the kernel iterates exactly the right number of events per batch
// without scanning input_ids twice.

namespace {

constexpr int64_t IMAGE_TOKEN_ID = 151655LL;
constexpr int64_t VIDEO_TOKEN_ID = 151656LL;
constexpr int64_t SPATIAL_MERGE_SIZE = 2;
constexpr float TOKENS_PER_SECOND = 2.0f;

__device__ inline int64_t max3_i64(int64_t a, int64_t b, int64_t c) {
    int64_t m = a > b ? a : b;
    return m > c ? m : c;
}

__global__ void mrope_position_kernel(
    const int64_t* __restrict__ input_ids,
    const int64_t* __restrict__ attention_mask,
    const int64_t* __restrict__ image_grid_thw,
    const int64_t* __restrict__ video_grid_thw,
    const float* __restrict__ second_per_grid_ts,
    const int64_t* __restrict__ image_count,
    const int64_t* __restrict__ video_count,
    const int64_t* __restrict__ image_offset,
    const int64_t* __restrict__ video_offset,
    const int64_t* __restrict__ valid_lens,
    int batch_size,
    int seq_len,
    int64_t* __restrict__ position_ids,
    int64_t* __restrict__ mrope_position_deltas) {

    const int b = blockIdx.x;
    if (b >= batch_size) return;
    if (threadIdx.x != 0) return;

    (void)attention_mask;  // valid_lens already encodes the mask sum.

    const int64_t S = static_cast<int64_t>(seq_len);
    const int64_t B = static_cast<int64_t>(batch_size);
    const int64_t* in_ids_b = input_ids + static_cast<int64_t>(b) * S;
    int64_t* pos_t = position_ids + 0 * B * S + static_cast<int64_t>(b) * S;
    int64_t* pos_h = position_ids + 1 * B * S + static_cast<int64_t>(b) * S;
    int64_t* pos_w = position_ids + 2 * B * S + static_cast<int64_t>(b) * S;

    const int64_t valid_len = valid_lens[b];

    // Pure-text batch within the with-vision global branch.
    int64_t total_events = image_count[b] + video_count[b];
    if (total_events == 0) {
        for (int64_t k = 0; k < valid_len; k++) {
            pos_t[k] = k;
            pos_h[k] = k;
            pos_w[k] = k;
        }
        for (int64_t k = valid_len; k < S; k++) {
            pos_t[k] = 1;
            pos_h[k] = 1;
            pos_w[k] = 1;
        }
        mrope_position_deltas[b] = 0;
        return;
    }

    int64_t img_idx_global = image_offset[b];
    int64_t vid_idx_global = video_offset[b];

    int64_t st = 0;
    int64_t st_idx = 0;
    int64_t out_idx = 0;
    int64_t max_pos = -1;

    for (int64_t evt = 0; evt < total_events; evt++) {
        // Find next image / video marker at or after st (within valid_len).
        int64_t ed = -1;
        int is_video = 0;
        for (int64_t t = st; t < valid_len; t++) {
            const int64_t tok = in_ids_b[t];
            if (tok == IMAGE_TOKEN_ID) { ed = t; is_video = 0; break; }
            if (tok == VIDEO_TOKEN_ID) { ed = t; is_video = 1; break; }
        }
        if (ed < 0) break;  // Defensive: precomputed counts should cover this loop exactly.

        // Text run [st, ed): positions st_idx ... st_idx + text_len - 1.
        const int64_t text_len = ed - st;
        for (int64_t k = 0; k < text_len; k++) {
            const int64_t p = st_idx + k;
            if (out_idx < valid_len) {
                pos_t[out_idx] = p;
                pos_h[out_idx] = p;
                pos_w[out_idx] = p;
                out_idx++;
            }
            if (p > max_pos) max_pos = p;
        }

        // Grid dimensions for this vision span (post-spatial-merge).
        int64_t llm_grid_t, llm_grid_h, llm_grid_w;
        float second_per_grid_t;
        if (is_video == 0) {
            llm_grid_t = image_grid_thw[3 * img_idx_global + 0];
            llm_grid_h = image_grid_thw[3 * img_idx_global + 1] / SPATIAL_MERGE_SIZE;
            llm_grid_w = image_grid_thw[3 * img_idx_global + 2] / SPATIAL_MERGE_SIZE;
            second_per_grid_t = 0.0f;
            img_idx_global++;
        } else {
            llm_grid_t = video_grid_thw[3 * vid_idx_global + 0];
            llm_grid_h = video_grid_thw[3 * vid_idx_global + 1] / SPATIAL_MERGE_SIZE;
            llm_grid_w = video_grid_thw[3 * vid_idx_global + 2] / SPATIAL_MERGE_SIZE;
            second_per_grid_t = second_per_grid_ts[vid_idx_global];
            vid_idx_global++;
        }
        const int64_t hw = llm_grid_h * llm_grid_w;
        const int64_t num_vision = llm_grid_t * hw;
        const int64_t base = text_len + st_idx;

        // 3D vision-grid positions: t_idx encodes the per-frame time
        // shifted by second_per_grid_t * tokens_per_second; h_idx / w_idx
        // unroll the post-merge spatial grid.
        for (int64_t k = 0; k < num_vision; k++) {
            const int64_t t_idx_local = k / hw;
            const int64_t hw_pos = k - t_idx_local * hw;
            const int64_t h_idx_local = hw_pos / llm_grid_w;
            const int64_t w_idx_local = hw_pos - h_idx_local * llm_grid_w;
            const int64_t t_idx = static_cast<int64_t>(
                static_cast<float>(t_idx_local) * second_per_grid_t * TOKENS_PER_SECOND);
            const int64_t t_final = t_idx + base;
            const int64_t h_final = h_idx_local + base;
            const int64_t w_final = w_idx_local + base;
            if (out_idx < valid_len) {
                pos_t[out_idx] = t_final;
                pos_h[out_idx] = h_final;
                pos_w[out_idx] = w_final;
                out_idx++;
            }
            if (t_final > max_pos) max_pos = t_final;
            if (h_final > max_pos) max_pos = h_final;
            if (w_final > max_pos) max_pos = w_final;
        }

        // st_idx for next iteration: max of this vision span + 1.
        const int64_t span_max_t = static_cast<int64_t>(
            static_cast<float>(llm_grid_t - 1) * second_per_grid_t * TOKENS_PER_SECOND) + base;
        const int64_t span_max_h = (llm_grid_h - 1) + base;
        const int64_t span_max_w = (llm_grid_w - 1) + base;
        st_idx = max3_i64(span_max_t, span_max_h, span_max_w) + 1;

        st = ed + num_vision;
    }

    // Final text run after the last vision span (if any tokens remain).
    if (st < valid_len) {
        const int64_t text_len = valid_len - st;
        for (int64_t k = 0; k < text_len; k++) {
            const int64_t p = st_idx + k;
            if (out_idx < valid_len) {
                pos_t[out_idx] = p;
                pos_h[out_idx] = p;
                pos_w[out_idx] = p;
                out_idx++;
            }
            if (p > max_pos) max_pos = p;
        }
    }

    // Default-1 fill for [out_idx, seq_len) — matches the reference's
    // initial torch.ones() over positions the algorithm did not emit.
    for (int64_t k = out_idx; k < S; k++) {
        pos_t[k] = 1;
        pos_h[k] = 1;
        pos_w[k] = 1;
    }

    mrope_position_deltas[b] = max_pos + 1 - valid_len;
}

}  // namespace

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
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(input_ids.size(0));
    const int seq_len = static_cast<int>(input_ids.size(1));

    // Image / video grid tensors may be empty when total_num_{images,videos}=0.
    const int64_t* image_grid_ptr = image_grid_thw.numel() > 0
        ? image_grid_thw.data_ptr<int64_t>() : nullptr;
    const int64_t* video_grid_ptr = video_grid_thw.numel() > 0
        ? video_grid_thw.data_ptr<int64_t>() : nullptr;
    const float* sec_per_grid_ptr = second_per_grid_ts.numel() > 0
        ? second_per_grid_ts.data_ptr<float>() : nullptr;

    dim3 grid(batch_size);
    dim3 block(1);
    mrope_position_kernel<<<grid, block, 0, stream>>>(
        input_ids.data_ptr<int64_t>(),
        attention_mask.data_ptr<int64_t>(),
        image_grid_ptr,
        video_grid_ptr,
        sec_per_grid_ptr,
        image_count.data_ptr<int64_t>(),
        video_count.data_ptr<int64_t>(),
        image_offset.data_ptr<int64_t>(),
        video_offset.data_ptr<int64_t>(),
        valid_lens.data_ptr<int64_t>(),
        batch_size,
        seq_len,
        position_ids.data_ptr<int64_t>(),
        mrope_position_deltas.data_ptr<int64_t>());
}
