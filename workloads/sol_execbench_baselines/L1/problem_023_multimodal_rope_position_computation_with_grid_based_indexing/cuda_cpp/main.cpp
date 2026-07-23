#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Multimodal 3D RoPE position computation with grid-based indexing (Qwen3-VL).
// Produces bilinearly-interpolated patch position embeddings plus interleaved
// MRoPE cos / sin embeddings, all driven by the per-image grid_thw rows.
//
// Reference (problem_023_multimodal_rope_position_computation_with_grid_based_indexing):
//     spatial_merge_size = 2; num_grid_per_side = 35
//     hidden_size = pos_embed_weight.shape[1]; head_dim = inv_freq.shape[0] * 4
//     mrope_section = [24, 20, 20]
//
//     # Part 1 -- bilinear interpolation position embeddings
//     for (t, h, w) in grid_thw:
//         h_idxs = linspace(0, 34, h); w_idxs = linspace(0, 34, w)
//         h_floor = h_idxs.long(); w_floor = w_idxs.long()
//         h_ceil = (h_floor + 1).clamp(max=34); w_ceil = (w_floor + 1).clamp(max=34)
//         dh = h_idxs - h_floor.float(); dw = w_idxs - w_floor.float()
//         base_h = h_floor * 35; base_h_ceil = h_ceil * 35
//         indices[0..3] = (base*[:,None] + w*[None,:]).flatten()    # 4 corner gathers
//         weights[0..3] = ((1-/dh)[:,None] * (1-/dw)[None,:]).flatten()
//     idx_tensor    = tensor(idx_list)    # [4, sum(h*w)]
//     weight_tensor = tensor(weight_list) # [4, sum(h*w)]
//     pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]   # [4, N, hidden]
//     patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
//     patch_pos_embeds = patch_pos_embeds.split([h*w for each grid])
//     for (pos_embed, t, h, w):
//         pos_embed = pos_embed.repeat(t, 1)
//         pos_embed = pos_embed.view(t, h/2, 2, w/2, 2, -1).permute(0,1,3,2,4,5).flatten(0,4)
//     patch_pos_embeds = cat(...)                                             # output 1
//
//     # Part 2 -- rotary position embeddings with MRoPE interleave
//     max_hw = grid_thw[:, 1:].max(); seq = arange(max_hw); freqs = outer(seq, inv_freq)
//     total_tokens = prod(grid_thw, dim=1).sum()
//     pos_ids = empty((total_tokens, 2), long)
//     for (num_frames, height, width):
//         merged_h = height/2; merged_w = width/2
//         row_idx = block_rows[:,N,N,N]*2 + intra_row[N,N,:,N]   (expanded, reshaped -1)
//         col_idx = block_cols[N,:,N,N]*2 + intra_col[N,N,N,:]   (expanded, reshaped -1)
//         coords = stack((row_idx, col_idx), -1)
//         if num_frames > 1: coords = coords.repeat(num_frames, 1)
//         pos_ids[offset:offset+n] = coords
//     embeddings = freqs[pos_ids].flatten(1)                       # [total_tokens, 64]
//     freqs_3d = embeddings.unsqueeze(0).expand(3, -1, -1).clone() # [3, total_tokens, 64]
//     freqs_t = freqs_3d[0]
//     for dim, off in ((1,1),(2,2)):
//         idx = slice(off, mrope_section[dim]*3, 3)
//         freqs_t[..., idx] = freqs_3d[dim, ..., idx]
//     emb = cat((freqs_t, freqs_t), -1)                           # [total_tokens, 128]
//     cos_embeddings = emb.cos(); sin_embeddings = emb.sin()      # outputs 2 and 3
//
// Per definition.json (insertion order):
//   grid_thw         : [num_images, 3]                    int64   (per-row [t, h, w])
//   pos_embed_weight : [num_position_embeddings, hidden]  fp32
//   inv_freq         : [quarter_head_dim]                 fp32
//   patch_pos_embeds : [total_tokens, hidden_size]        fp32    (DPS, pre-alloc)
//   cos_embeddings   : [total_tokens, head_dim]           fp32    (DPS, pre-alloc)
//   sin_embeddings   : [total_tokens, head_dim]           fp32    (DPS, pre-alloc)
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The shapes are
// data dependent (grid_thw rows drive linspace lengths, the split, the per-grid
// permute, and the pos-id coordinate generation), so grid dims are read to the
// host with .item<int64_t>() and the loops replay the reference structure
// exactly. The 4-corner advanced index pos_embed_weight[idx] is reproduced as 4
// index_select gathers summed with their bilinear weights. The MRoPE interleave
// writes freqs_t in place via a strided slice copy, matching the reference. All
// three outputs are written in place (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grid_thw,
    const torch::Tensor& pos_embed_weight,
    const torch::Tensor& inv_freq,
    torch::Tensor patch_pos_embeds,
    torch::Tensor cos_embeddings,
    torch::Tensor sin_embeddings
) {
    check_fp32_cuda(pos_embed_weight, "pos_embed_weight");
    check_fp32_cuda(inv_freq, "inv_freq");
    check_fp32_cuda(patch_pos_embeds, "patch_pos_embeds");
    check_fp32_cuda(cos_embeddings, "cos_embeddings");
    check_fp32_cuda(sin_embeddings, "sin_embeddings");
    TORCH_CHECK(grid_thw.is_cuda(), "grid_thw must be a CUDA tensor");
    TORCH_CHECK(grid_thw.dtype() == torch::kInt64, "grid_thw must be int64, got ", grid_thw.dtype());
    TORCH_CHECK(grid_thw.dim() == 2 && grid_thw.size(1) == 3,
                "grid_thw must be 2D [num_images, 3], got ", grid_thw.sizes());
    TORCH_CHECK(pos_embed_weight.dim() == 2,
                "pos_embed_weight must be 2D [num_position_embeddings, hidden], got ",
                pos_embed_weight.dim(), "D");
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must be 1D [quarter_head_dim], got ", inv_freq.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grid_thw.device());

    constexpr int64_t kMergeSize = 2;
    constexpr int64_t kNumGridPerSide = 35;
    const std::vector<int64_t> kMropeSection = {24, 20, 20};

    const auto float_opts = pos_embed_weight.options();
    const auto long_opts = grid_thw.options();
    const int64_t num_images = grid_thw.size(0);

    // Read every grid row to the host once; shapes downstream depend on them.
    std::vector<int64_t> ts(num_images), hs(num_images), ws(num_images);
    for (int64_t i = 0; i < num_images; ++i) {
        ts[i] = grid_thw[i][0].item<int64_t>();
        hs[i] = grid_thw[i][1].item<int64_t>();
        ws[i] = grid_thw[i][2].item<int64_t>();
    }

    // --- Part 1: bilinear interpolation position embeddings ---------------
    // Accumulate the four corner index / weight channels across all grids,
    // mirroring the reference's per-channel list concatenation.
    std::vector<torch::Tensor> idx_ch[4];
    std::vector<torch::Tensor> wt_ch[4];

    for (int64_t i = 0; i < num_images; ++i) {
        const int64_t h = hs[i];
        const int64_t w = ws[i];

        const torch::Tensor h_idxs = at::linspace(0, kNumGridPerSide - 1, h, float_opts);
        const torch::Tensor w_idxs = at::linspace(0, kNumGridPerSide - 1, w, float_opts);

        const torch::Tensor h_floor = h_idxs.to(torch::kLong);
        const torch::Tensor w_floor = w_idxs.to(torch::kLong);
        const torch::Tensor h_ceil = at::clamp_max(h_floor + 1, kNumGridPerSide - 1);
        const torch::Tensor w_ceil = at::clamp_max(w_floor + 1, kNumGridPerSide - 1);

        const torch::Tensor dh = h_idxs - h_floor.to(torch::kFloat32);
        const torch::Tensor dw = w_idxs - w_floor.to(torch::kFloat32);

        const torch::Tensor base_h = h_floor * kNumGridPerSide;
        const torch::Tensor base_h_ceil = h_ceil * kNumGridPerSide;

        // 4 corner gather indices, flattened over (h, w).
        const torch::Tensor idx0 = (base_h.unsqueeze(1) + w_floor.unsqueeze(0)).flatten();
        const torch::Tensor idx1 = (base_h.unsqueeze(1) + w_ceil.unsqueeze(0)).flatten();
        const torch::Tensor idx2 = (base_h_ceil.unsqueeze(1) + w_floor.unsqueeze(0)).flatten();
        const torch::Tensor idx3 = (base_h_ceil.unsqueeze(1) + w_ceil.unsqueeze(0)).flatten();

        const torch::Tensor one_minus_dh = 1 - dh;
        const torch::Tensor one_minus_dw = 1 - dw;
        const torch::Tensor wt0 = (one_minus_dh.unsqueeze(1) * one_minus_dw.unsqueeze(0)).flatten();
        const torch::Tensor wt1 = (one_minus_dh.unsqueeze(1) * dw.unsqueeze(0)).flatten();
        const torch::Tensor wt2 = (dh.unsqueeze(1) * one_minus_dw.unsqueeze(0)).flatten();
        const torch::Tensor wt3 = (dh.unsqueeze(1) * dw.unsqueeze(0)).flatten();

        idx_ch[0].push_back(idx0); idx_ch[1].push_back(idx1);
        idx_ch[2].push_back(idx2); idx_ch[3].push_back(idx3);
        wt_ch[0].push_back(wt0); wt_ch[1].push_back(wt1);
        wt_ch[2].push_back(wt2); wt_ch[3].push_back(wt3);
    }

    // pos_embeds[c] = pos_embed_weight[idx_c] * weight_c[:, None]; sum the four.
    torch::Tensor patch_pos = at::zeros(
        {static_cast<int64_t>(0), pos_embed_weight.size(1)}, float_opts);
    bool have_patch_pos = false;
    for (int c = 0; c < 4; ++c) {
        const torch::Tensor idx_c = at::cat(idx_ch[c], /*dim=*/0);
        const torch::Tensor wt_c = at::cat(wt_ch[c], /*dim=*/0);
        const torch::Tensor gathered =
            pos_embed_weight.index_select(/*dim=*/0, idx_c) * wt_c.unsqueeze(1);
        if (!have_patch_pos) {
            patch_pos = gathered;
            have_patch_pos = true;
        } else {
            patch_pos = patch_pos + gathered;
        }
    }

    // Split into per-grid [h*w, hidden] segments, repeat over t, spatial-shuffle.
    std::vector<torch::Tensor> permuted;
    permuted.reserve(num_images);
    int64_t seg_offset = 0;
    for (int64_t i = 0; i < num_images; ++i) {
        const int64_t t = ts[i];
        const int64_t h = hs[i];
        const int64_t w = ws[i];
        const int64_t seg = h * w;

        torch::Tensor pe = patch_pos.slice(/*dim=*/0, seg_offset, seg_offset + seg);
        seg_offset += seg;

        pe = pe.repeat({t, 1});
        pe = pe.view({t, h / kMergeSize, kMergeSize, w / kMergeSize, kMergeSize, -1})
                 .permute({0, 1, 3, 2, 4, 5})
                 .flatten(/*start_dim=*/0, /*end_dim=*/4);
        permuted.push_back(pe);
    }
    const torch::Tensor patch_pos_result = at::cat(permuted, /*dim=*/0);

    // --- Part 2: rotary position embeddings with MRoPE interleave ---------
    const int64_t max_hw = grid_thw.slice(/*dim=*/1, /*start=*/1).max().item<int64_t>();
    const torch::Tensor seq = at::arange(max_hw, float_opts);
    const torch::Tensor freqs = at::outer(seq, inv_freq);  // [max_hw, quarter_head_dim]

    int64_t total_tokens = 0;
    for (int64_t i = 0; i < num_images; ++i) total_tokens += ts[i] * hs[i] * ws[i];

    torch::Tensor pos_ids = at::empty({total_tokens, 2}, long_opts);
    int64_t pos_offset = 0;
    for (int64_t i = 0; i < num_images; ++i) {
        const int64_t num_frames = ts[i];
        const int64_t merged_h = hs[i] / kMergeSize;
        const int64_t merged_w = ws[i] / kMergeSize;

        const torch::Tensor block_rows = at::arange(merged_h, long_opts);
        const torch::Tensor block_cols = at::arange(merged_w, long_opts);
        const torch::Tensor intra_row = at::arange(kMergeSize, long_opts);
        const torch::Tensor intra_col = at::arange(kMergeSize, long_opts);

        // row_idx[a,b,c,d] = block_rows[a]*2 + intra_row[c]; col mirror on b,d.
        torch::Tensor row_idx =
            block_rows.view({merged_h, 1, 1, 1}) * kMergeSize +
            intra_row.view({1, 1, kMergeSize, 1});
        torch::Tensor col_idx =
            block_cols.view({1, merged_w, 1, 1}) * kMergeSize +
            intra_col.view({1, 1, 1, kMergeSize});

        row_idx = row_idx.expand({merged_h, merged_w, kMergeSize, kMergeSize}).reshape({-1});
        col_idx = col_idx.expand({merged_h, merged_w, kMergeSize, kMergeSize}).reshape({-1});

        torch::Tensor coords = at::stack({row_idx, col_idx}, /*dim=*/-1);
        if (num_frames > 1) {
            coords = coords.repeat({num_frames, 1});
        }

        const int64_t num_tokens = coords.size(0);
        pos_ids.slice(/*dim=*/0, pos_offset, pos_offset + num_tokens).copy_(coords);
        pos_offset += num_tokens;
    }

    // freqs[pos_ids] -> [total_tokens, 2, quarter_head_dim] -> flatten -> [total_tokens, 64].
    torch::Tensor embeddings = freqs.index({pos_ids}).flatten(/*start_dim=*/1);

    // Broadcast to the 3 MRoPE dims, then strided-interleave dims 1 and 2 into dim 0.
    torch::Tensor freqs_3d = embeddings.unsqueeze(0).expand({3, -1, -1}).clone();
    torch::Tensor freqs_t = freqs_3d[0];
    for (int dim = 1; dim <= 2; ++dim) {
        const int64_t offset_dim = dim;
        const int64_t length = kMropeSection[dim] * 3;
        const torch::Tensor src =
            freqs_3d[dim].slice(/*dim=*/-1, /*start=*/offset_dim, /*end=*/length, /*step=*/3);
        freqs_t.slice(/*dim=*/-1, /*start=*/offset_dim, /*end=*/length, /*step=*/3).copy_(src);
    }

    const torch::Tensor emb = at::cat({freqs_t, freqs_t}, /*dim=*/-1);
    const torch::Tensor cos_result = emb.cos();
    const torch::Tensor sin_result = emb.sin();

    TORCH_CHECK(patch_pos_result.sizes() == patch_pos_embeds.sizes(),
                "computed patch_pos_embeds shape ", patch_pos_result.sizes(),
                " does not match pre-allocated output ", patch_pos_embeds.sizes());
    TORCH_CHECK(cos_result.sizes() == cos_embeddings.sizes(),
                "computed cos_embeddings shape ", cos_result.sizes(),
                " does not match pre-allocated output ", cos_embeddings.sizes());
    TORCH_CHECK(sin_result.sizes() == sin_embeddings.sizes(),
                "computed sin_embeddings shape ", sin_result.sizes(),
                " does not match pre-allocated output ", sin_embeddings.sizes());

    patch_pos_embeds.copy_(patch_pos_result);
    cos_embeddings.copy_(cos_result);
    sin_embeddings.copy_(sin_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Multimodal 3D RoPE: bilinear-interpolated patch position embeddings "
          "plus interleaved MRoPE cos / sin via ATen; fp32; destination-passing "
          "style.");
}
