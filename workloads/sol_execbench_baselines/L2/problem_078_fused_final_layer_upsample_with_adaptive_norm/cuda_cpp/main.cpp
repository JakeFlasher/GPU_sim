#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>
#include <string>

// Fused UNet upsampling path with timestep-conditioned adaptive GroupNorm,
// reproduced op-for-op from the reference run(). Transformer hidden states are
// reshaped to a spatial feature map and pushed through a ResBlock (GroupNorm +
// SiLU + conv, adaptive-GroupNorm affine driven by the timestep embedding, SiLU
// + conv, plus a 1x1 skip), then a final GroupNorm + SiLU + conv projects to the
// VAE latent space.
//
// Reference, op-for-op:
//   token_h, token_w factored from seq_len (square-ish; shrink token_h until it divides)
//   x_spatial = x.reshape(B, token_h, token_w, hidden).permute(0,3,1,2).contiguous()
//   emb = linear(silu(timestep_emb), time_emb_mlp_w, time_emb_mlp_b)   # computed, unused downstream
//   h = group_norm(x_spatial, 32, in_norm_w, in_norm_b, eps)
//   h = silu(h)
//   h = conv2d(h, in_conv_w, in_conv_b, padding=1)
//   emb_out = linear(silu(timestep_emb), emb_linear_w, emb_linear_b)
//   emb_out = emb_out[:, :, None, None]                                # [B, 2*hc, 1, 1]
//   h = group_norm(h, 32, out_norm_w, out_norm_b, eps)
//   scale, shift = chunk(emb_out, 2, dim=1)                            # each [B, hc, 1, 1]
//   h = h * (1.0 + scale) + shift
//   h = silu(h)
//   h = conv2d(h, out_conv_w, out_conv_b, padding=1)
//   skip = conv2d(x_spatial, skip_conv_w, skip_conv_b)                 # 1x1, no padding
//   h = skip + h
//   h = group_norm(h, 32, final_norm_w, final_norm_b, eps)
//   h = silu(h)
//   output = conv2d(h, final_conv_w, final_conv_b, padding=1)
//
// The time_emb_mlp branch (emb) is computed by the reference but never consumed;
// it is reproduced here for faithful transcription. The adaptive affine reads
// scale/shift from the resblock_emb_linear projection of the timestep embedding.
//
// Per definition.json (input insertion order, then output):
//   x                          : [batch, seq_len, hidden_size=4096]       bf16
//   timestep_emb               : [batch, emb_channels=4096]               bf16
//   time_emb_mlp_linear_weight : [double_hidden_channels=6144, 4096]      bf16
//   time_emb_mlp_linear_bias   : [6144]                                   bf16
//   resblock_in_norm_weight    : [hidden_size=4096]                       bf16
//   resblock_in_norm_bias      : [4096]                                   bf16
//   resblock_in_conv_weight    : [hidden_channels=3072, 4096, 3, 3]       bf16
//   resblock_in_conv_bias      : [3072]                                   bf16
//   resblock_emb_linear_weight : [6144, 4096]                             bf16
//   resblock_emb_linear_bias   : [6144]                                   bf16
//   resblock_out_norm_weight   : [3072]                                   bf16
//   resblock_out_norm_bias     : [3072]                                   bf16
//   resblock_out_conv_weight   : [3072, 3072, 3, 3]                       bf16
//   resblock_out_conv_bias     : [3072]                                   bf16
//   resblock_skip_conv_weight  : [3072, 4096, 1, 1]                       bf16
//   resblock_skip_conv_bias    : [3072]                                   bf16
//   final_norm_weight          : [3072]                                   bf16
//   final_norm_bias            : [3072]                                   bf16
//   final_conv_weight          : [out_channels=4, 3072, 3, 3]            bf16
//   final_conv_bias            : [4]                                      bf16
//   eps                        : scalar                                   fp32 -> double
//   output                     : [batch, out_channels=4, token_h, token_w] bf16 DPS
//
// token_h / token_w are recovered from seq_len by host integer factorization
// (shape-only arithmetic): start at floor(sqrt(seq_len)) and decrement token_h
// until it divides seq_len evenly, matching the reference loop exactly.
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- NO custom CUDA kernels.
// All three convolutions feeding the residual / final paths use padding 1; the
// 1x1 skip conv uses the default zero padding. at::group_norm uses 32 groups
// with the supplied eps; at::silu / at::linear / at::chunk cover the rest.
// Everything stays bfloat16 to match the reference -- no upcast of outputs. The
// scalar eps arrives as fp32 and is passed through as a double to at::group_norm.
// Wider accumulation inside the conv / matmul dispatch is left to ATen. Output is
// written in place (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& timestep_emb,
    const torch::Tensor& time_emb_mlp_linear_weight,
    const torch::Tensor& time_emb_mlp_linear_bias,
    const torch::Tensor& resblock_in_norm_weight,
    const torch::Tensor& resblock_in_norm_bias,
    const torch::Tensor& resblock_in_conv_weight,
    const torch::Tensor& resblock_in_conv_bias,
    const torch::Tensor& resblock_emb_linear_weight,
    const torch::Tensor& resblock_emb_linear_bias,
    const torch::Tensor& resblock_out_norm_weight,
    const torch::Tensor& resblock_out_norm_bias,
    const torch::Tensor& resblock_out_conv_weight,
    const torch::Tensor& resblock_out_conv_bias,
    const torch::Tensor& resblock_skip_conv_weight,
    const torch::Tensor& resblock_skip_conv_bias,
    const torch::Tensor& final_norm_weight,
    const torch::Tensor& final_norm_bias,
    const torch::Tensor& final_conv_weight,
    const torch::Tensor& final_conv_bias,
    double eps,
    torch::Tensor output
) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(timestep_emb, "timestep_emb");
    check_bf16_cuda(time_emb_mlp_linear_weight, "time_emb_mlp_linear_weight");
    check_bf16_cuda(time_emb_mlp_linear_bias, "time_emb_mlp_linear_bias");
    check_bf16_cuda(resblock_in_norm_weight, "resblock_in_norm_weight");
    check_bf16_cuda(resblock_in_norm_bias, "resblock_in_norm_bias");
    check_bf16_cuda(resblock_in_conv_weight, "resblock_in_conv_weight");
    check_bf16_cuda(resblock_in_conv_bias, "resblock_in_conv_bias");
    check_bf16_cuda(resblock_emb_linear_weight, "resblock_emb_linear_weight");
    check_bf16_cuda(resblock_emb_linear_bias, "resblock_emb_linear_bias");
    check_bf16_cuda(resblock_out_norm_weight, "resblock_out_norm_weight");
    check_bf16_cuda(resblock_out_norm_bias, "resblock_out_norm_bias");
    check_bf16_cuda(resblock_out_conv_weight, "resblock_out_conv_weight");
    check_bf16_cuda(resblock_out_conv_bias, "resblock_out_conv_bias");
    check_bf16_cuda(resblock_skip_conv_weight, "resblock_skip_conv_weight");
    check_bf16_cuda(resblock_skip_conv_bias, "resblock_skip_conv_bias");
    check_bf16_cuda(final_norm_weight, "final_norm_weight");
    check_bf16_cuda(final_norm_bias, "final_norm_bias");
    check_bf16_cuda(final_conv_weight, "final_conv_weight");
    check_bf16_cuda(final_conv_bias, "final_conv_bias");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, seq_len, hidden], got ", x.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    constexpr int64_t kNumGroups = 32;

    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t hidden_size = x.size(2);

    // Recover the token grid from seq_len: start near the square root and shrink
    // token_h until it divides seq_len exactly (shape-only host arithmetic,
    // matching the reference factorization loop).
    int64_t token_h = static_cast<int64_t>(std::sqrt(static_cast<double>(seq_len)));
    if (token_h < 1) {
        token_h = 1;
    }
    int64_t token_w = seq_len / token_h;
    while (token_h * token_w != seq_len) {
        token_h -= 1;
        TORCH_CHECK(token_h >= 1, "failed to factor seq_len ", seq_len, " into a token grid");
        token_w = seq_len / token_h;
    }

    // Sequence to spatial: [B, seq_len, hidden] -> [B, hidden, token_h, token_w].
    auto x_spatial = x.reshape({batch_size, token_h, token_w, hidden_size});
    x_spatial = x_spatial.permute({0, 3, 1, 2}).contiguous();  // [B, C, H, W]

    const std::vector<int64_t> stride{1, 1};
    const std::vector<int64_t> pad1{1, 1};

    // Time-embedding MLP. The reference computes this branch but never consumes
    // it downstream; reproduced for a faithful transcription.
    auto emb = at::silu(timestep_emb);
    emb = at::linear(emb, time_emb_mlp_linear_weight, time_emb_mlp_linear_bias);  // [B, 2*hc]
    (void)emb;

    // ResBlock input path: GroupNorm + SiLU + conv (padding 1).
    auto h = at::group_norm(x_spatial, kNumGroups, resblock_in_norm_weight, resblock_in_norm_bias, eps);
    h = at::silu(h);
    h = at::conv2d(h, resblock_in_conv_weight, resblock_in_conv_bias, /*stride=*/stride, /*padding=*/pad1);

    // Scale / shift for the adaptive normalization come from a SiLU + linear of
    // the timestep embedding, broadcast over the spatial dims.
    auto emb_out = at::silu(timestep_emb);
    emb_out = at::linear(emb_out, resblock_emb_linear_weight, resblock_emb_linear_bias);  // [B, 2*hc]
    emb_out = emb_out.unsqueeze(-1).unsqueeze(-1);  // [B, 2*hc, 1, 1]

    // Adaptive GroupNorm: normalize, then apply h * (1 + scale) + shift.
    h = at::group_norm(h, kNumGroups, resblock_out_norm_weight, resblock_out_norm_bias, eps);
    auto chunks = at::chunk(emb_out, /*chunks=*/2, /*dim=*/1);  // each [B, hc, 1, 1]
    auto scale = chunks[0];
    auto shift = chunks[1];
    h = h * (1.0 + scale) + shift;

    // ResBlock output path: SiLU + conv (padding 1).
    h = at::silu(h);
    h = at::conv2d(h, resblock_out_conv_weight, resblock_out_conv_bias, /*stride=*/stride, /*padding=*/pad1);

    // 1x1 skip projection (default zero padding) added to the residual branch.
    auto skip = at::conv2d(x_spatial, resblock_skip_conv_weight, resblock_skip_conv_bias);
    h = skip + h;

    // Final GroupNorm + SiLU + conv (padding 1) projecting to the VAE latent.
    h = at::group_norm(h, kNumGroups, final_norm_weight, final_norm_bias, eps);
    h = at::silu(h);
    auto result = at::conv2d(h, final_conv_weight, final_conv_bias, /*stride=*/stride, /*padding=*/pad1);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Fused UNet upsample ResBlock with timestep-conditioned adaptive "
          "GroupNorm via pure ATen dispatch: hidden states reshaped to a spatial "
          "feature map, ResBlock GroupNorm(32) + SiLU + conv3x3, adaptive affine "
          "h*(1+scale)+shift with scale/shift chunked from a SiLU+linear of the "
          "timestep embedding, SiLU + conv3x3, a 1x1 skip-conv residual, then a "
          "final GroupNorm(32) + SiLU + conv3x3 projecting to the VAE latent. "
          "bfloat16; destination-passing style.");
}
