#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Vision patch merger: spatial shuffle to merge 2x2 patches, layer norm, and a
// two-layer GELU MLP (Qwen3-Omni vision merger path).
//
// Reference (problem_020_vision_patch_merger_spatial_shuffle_mlp):
//     merge_size = 2; hidden_size = 1536; hidden_size_expanded = 6144
//     # layer norm over hidden_size, accumulated in fp32, cast back to bf16
//     hidden_fp32 = hidden.to(fp32)
//     mean = hidden_fp32.mean(-1, keepdim); var = hidden_fp32.var(-1, keepdim, unbiased=False)
//     hidden_norm = (hidden_fp32 - mean) / sqrt(var + eps)
//     hidden_norm = (hidden_norm * ln_weight.to(fp32) + ln_bias.to(fp32)).to(bf16)
//     # per-grid spatial shuffle over each grid_thw row
//     for each (t, h, w) in grid_thw:
//         patches = hidden_norm[offset : offset + t*h*w]
//         patches = patches.view(t, h/2, 2, w/2, 2, hidden_size)
//         patches = patches.permute(0, 1, 3, 2, 4, 5)
//         patches = patches.reshape(t * (h/2) * (w/2), hidden_size_expanded)
//     hidden_shuffled = cat(shuffled, dim=0)
//     # two-layer MLP with exact (erf-based) GELU
//     hidden_fc1  = linear(hidden_shuffled, fc1_weight, fc1_bias)
//     hidden_gelu = gelu(hidden_fc1)                 # approximate="none"
//     output      = linear(hidden_gelu, fc2_weight, fc2_bias)
//
// Per definition.json (insertion order):
//   hidden     : [num_patches, hidden_size]                    bf16
//   grid_thw   : [num_grids, 3]                                int64  (per-row [t, h, w])
//   ln_weight  : [hidden_size]                                 bf16
//   ln_bias    : [hidden_size]                                 bf16
//   fc1_weight : [hidden_size_expanded, hidden_size_expanded]  bf16
//   fc1_bias   : [hidden_size_expanded]                        bf16
//   fc2_weight : [out_hidden_size, hidden_size_expanded]       bf16
//   fc2_bias   : [out_hidden_size]                             bf16
//   eps        : scalar                                        fp32  (passed by value)
//   output     : [num_merged_patches, out_hidden_size]         bf16   (DPS, pre-alloc)
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The per-grid
// shuffle is data dependent (grid_thw drives the view/permute/reshape shapes),
// so grid dims are read to the host with .item<int64_t>() and the loop replays
// the reference's structure exactly. at::layer_norm upcasts the bf16 input to
// fp32 internally and uses biased variance, matching the reference's fp32
// mean / var(unbiased=False) accumulation and bf16 result. The permute makes
// each slice non-contiguous, so at::reshape (a copy when needed) stands in for
// the reference's .reshape. Output is written in place into the pre-allocated
// destination (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden,
    const torch::Tensor& grid_thw,
    const torch::Tensor& ln_weight,
    const torch::Tensor& ln_bias,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    double eps,
    torch::Tensor output
) {
    check_bf16_cuda(hidden, "hidden");
    check_bf16_cuda(ln_weight, "ln_weight");
    check_bf16_cuda(ln_bias, "ln_bias");
    check_bf16_cuda(fc1_weight, "fc1_weight");
    check_bf16_cuda(fc1_bias, "fc1_bias");
    check_bf16_cuda(fc2_weight, "fc2_weight");
    check_bf16_cuda(fc2_bias, "fc2_bias");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(grid_thw.is_cuda(), "grid_thw must be a CUDA tensor");
    TORCH_CHECK(grid_thw.dtype() == torch::kInt64, "grid_thw must be int64, got ", grid_thw.dtype());
    TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [num_patches, hidden_size], got ", hidden.dim(), "D");
    TORCH_CHECK(grid_thw.dim() == 2 && grid_thw.size(1) == 3,
                "grid_thw must be 2D [num_grids, 3], got ", grid_thw.sizes());

    const c10::cuda::CUDAGuard device_guard(hidden.device());

    constexpr int64_t kMergeSize = 2;
    const int64_t hidden_size = hidden.size(1);
    const int64_t hidden_size_expanded = hidden_size * kMergeSize * kMergeSize;

    // Layer norm over hidden_size (fp32 accumulation inside at::layer_norm,
    // result cast back to bf16 by the bf16 input/weight dtype).
    const torch::Tensor hidden_norm = at::layer_norm(
        hidden, /*normalized_shape=*/{hidden_size}, ln_weight, ln_bias, eps);

    // Per-grid 2x2 spatial shuffle, driven by the grid_thw rows on the host.
    const int64_t num_grids = grid_thw.size(0);
    std::vector<torch::Tensor> shuffled;
    shuffled.reserve(num_grids);
    int64_t offset = 0;
    for (int64_t i = 0; i < num_grids; ++i) {
        const int64_t t = grid_thw[i][0].item<int64_t>();
        const int64_t h = grid_thw[i][1].item<int64_t>();
        const int64_t w = grid_thw[i][2].item<int64_t>();

        const int64_t num_patches_this = t * h * w;
        const int64_t h_merged = h / kMergeSize;
        const int64_t w_merged = w / kMergeSize;

        torch::Tensor patches =
            hidden_norm.slice(/*dim=*/0, offset, offset + num_patches_this);
        patches = patches.view(
            {t, h_merged, kMergeSize, w_merged, kMergeSize, hidden_size});
        patches = patches.permute({0, 1, 3, 2, 4, 5});
        patches = patches.reshape({t * h_merged * w_merged, hidden_size_expanded});

        shuffled.push_back(patches);
        offset += num_patches_this;
    }

    const torch::Tensor hidden_shuffled = at::cat(shuffled, /*dim=*/0);

    // Two-layer MLP with exact (erf-based) GELU.
    const torch::Tensor hidden_fc1 = at::linear(hidden_shuffled, fc1_weight, fc1_bias);
    const torch::Tensor hidden_gelu = at::gelu(hidden_fc1, /*approximate=*/"none");
    const torch::Tensor result = at::linear(hidden_gelu, fc2_weight, fc2_bias);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "computed output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Vision patch merger: layer norm, per-grid 2x2 spatial shuffle, and a "
          "two-layer GELU MLP via ATen; bf16; destination-passing style.");
}
