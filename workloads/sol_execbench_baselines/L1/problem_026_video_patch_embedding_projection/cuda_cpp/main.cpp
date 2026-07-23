#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Video patch embedding projection (Wan2.2-T2V latent patchifier).
//
// Reference (problem_026_video_patch_embedding_projection): a Conv3D patch
// projection over a video tensor, followed by additive spatial and temporal
// positional embeddings and a final layer normalization over the hidden
// dimension. Reproduced op-for-op below.
//
// Per definition.json (insertion order):
//   inputs (all float16 except the scalar eps):
//     video                   : [batch_size, num_frames, in_channels, height, width]
//     patch_projection_weight : [hidden_size, in_channels, 1, patch_size, patch_size]
//     patch_projection_bias   : [hidden_size]
//     spatial_pos_embedding   : [1, num_spatial_patches, hidden_size]
//     temporal_pos_embedding  : [1, num_frames, 1, hidden_size]
//     norm_weight             : [hidden_size]
//     norm_bias               : [hidden_size]
//     eps                     : scalar (passed as double)
//   output (float16, DPS pre-alloc, written in place):
//     output                  : [batch_size, total_patches, hidden_size]
//
// The patch size is the spatial kernel extent of patch_projection_weight
// (its 4th dimension), and the conv stride along the spatial axes equals
// that patch size; the temporal kernel extent is 1 with temporal stride 1.
// These follow the weight shape rather than arriving as separate inputs, so
// they are derived here, matching the reference's patch_size read.
//
// Implementation: pure ATen / cuDNN dispatch -- NO custom CUDA kernels. The
// at::conv3d call routes through the same cuDNN path F.conv3d uses, and
// at::layer_norm carries the same fp16-in / internal-fp32-accumulate profile
// the reference's manual mean / variance normalization gets from PyTorch.
// Every tensor stays float16 throughout, exactly as the reference does; the
// layer norm upcasts its statistics internally. Output is written in place
// into the pre-allocated destination (destination-passing style).

static void check_fp16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat16, name, " must be float16, got ", t.dtype());
}

void run(
    const torch::Tensor& video,
    const torch::Tensor& patch_projection_weight,
    const torch::Tensor& patch_projection_bias,
    const torch::Tensor& spatial_pos_embedding,
    const torch::Tensor& temporal_pos_embedding,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias,
    double eps,
    torch::Tensor output
) {
    check_fp16_cuda(video, "video");
    check_fp16_cuda(patch_projection_weight, "patch_projection_weight");
    check_fp16_cuda(patch_projection_bias, "patch_projection_bias");
    check_fp16_cuda(spatial_pos_embedding, "spatial_pos_embedding");
    check_fp16_cuda(temporal_pos_embedding, "temporal_pos_embedding");
    check_fp16_cuda(norm_weight, "norm_weight");
    check_fp16_cuda(norm_bias, "norm_bias");
    check_fp16_cuda(output, "output");
    TORCH_CHECK(video.dim() == 5, "video must be 5D [B, F, C, H, W], got ", video.dim(), "D");
    TORCH_CHECK(patch_projection_weight.dim() == 5,
                "patch_projection_weight must be 5D [hidden, C, 1, p, p], got ",
                patch_projection_weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(video.device());

    const int64_t hidden_size = patch_projection_weight.size(0);
    const int64_t patch_size = patch_projection_weight.size(3);

    // Rearrange the video to (B, C, F, H, W) so the channel axis leads, the
    // layout Conv3D expects.
    auto video_ncfhw = video.permute({0, 2, 1, 3, 4});

    // Patch projection. Temporal kernel/stride is 1; spatial stride equals
    // the patch size; no padding -- the same call F.conv3d issues.
    auto patches = at::conv3d(
        video_ncfhw,
        patch_projection_weight,
        patch_projection_bias,
        /*stride=*/std::vector<int64_t>{1, patch_size, patch_size},
        /*padding=*/std::vector<int64_t>{0, 0, 0});

    // Move the frame axis ahead of the hidden axis: (B, F, hidden, H', W').
    patches = patches.permute({0, 2, 1, 3, 4});

    const int64_t batch_size = patches.size(0);
    const int64_t frames = patches.size(1);
    const int64_t num_h = patches.size(3);
    const int64_t num_w = patches.size(4);
    const int64_t num_spatial_patches = num_h * num_w;

    // Flatten the two spatial axes then bring the hidden axis last, giving
    // (B, F, num_spatial_patches, hidden).
    patches = patches.reshape({batch_size, frames, hidden_size, num_spatial_patches});
    patches = patches.permute({0, 1, 3, 2});

    // Spatial positional embeddings broadcast across frames; the unsqueeze
    // inserts the frame axis.
    patches = patches + spatial_pos_embedding.unsqueeze(1);

    // Temporal positional embeddings broadcast across spatial patches; the
    // input already carries the right shape for the broadcast.
    patches = patches + temporal_pos_embedding;

    // Flatten frame and spatial axes into the token axis: (B, total, hidden).
    patches = patches.reshape({batch_size, frames * num_spatial_patches, hidden_size});

    // Layer normalization over the hidden dimension. at::layer_norm reproduces
    // the reference's biased-variance normalization scaled by norm_weight and
    // shifted by norm_bias, with statistics accumulated in fp32 internally.
    auto result = at::layer_norm(
        patches,
        /*normalized_shape=*/std::vector<int64_t>{hidden_size},
        norm_weight,
        norm_bias,
        eps);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "layer_norm output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Video patch embedding projection via at::conv3d + positional "
          "embedding adds + at::layer_norm (pure ATen / cuDNN dispatch); "
          "fp16; destination-passing style.");
}
