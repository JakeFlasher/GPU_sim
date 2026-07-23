#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <cmath>
#include <vector>

// YaRN constants from definition.json. These are all axis constants
// so they're known at compile time.
namespace {

constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int HALF_DIM = QK_ROPE_HEAD_DIM / 2;       // 32
constexpr int ORIGINAL_MAX_POS = 4096;
constexpr float ROPE_THETA = 10000.0f;
constexpr float SCALING_FACTOR = 40.0f;
constexpr int BETA_FAST = 32;
constexpr int BETA_SLOW = 1;

// Mirror of the reference's yarn_find_correction_dim helper:
//   dim * log(max_pos / (n_rot * 2pi)) / (2 * log(base))
double yarn_find_correction_dim(
    double num_rotations, double dim, double base, double max_position) {
    constexpr double TWO_PI = 6.283185307179586;
    return (dim * std::log(max_position / (num_rotations * TWO_PI)))
        / (2.0 * std::log(base));
}

// Mirror of yarn_get_mscale: 0.1 * mscale * log(scaling) + 1.0 for
// scaling > 1, otherwise 1.0.
double yarn_get_mscale(double scaling, double mscale) {
    if (scaling <= 1.0) return 1.0;
    return 0.1 * mscale * std::log(scaling) + 1.0;
}

// Precompute the 32 inv_freq values via YaRN interpolation between the
// freq_extra (base) and freq_inter (scaling-aware) spectra. Once-per-
// invocation cost on the host; the result is staged to device for the
// kernel.
std::vector<float> compute_inv_freq() {
    std::vector<float> inv_freq(HALF_DIM);
    const double dim = static_cast<double>(QK_ROPE_HEAD_DIM);
    double low_raw = yarn_find_correction_dim(
        BETA_FAST, dim, ROPE_THETA, ORIGINAL_MAX_POS);
    double high_raw = yarn_find_correction_dim(
        BETA_SLOW, dim, ROPE_THETA, ORIGINAL_MAX_POS);
    double low = std::floor(low_raw);
    double high = std::ceil(high_raw);
    if (low == high) high += 0.001;
    for (int k = 0; k < HALF_DIM; ++k) {
        const double exponent = (2.0 * k) / dim;
        const double freq_extra = 1.0 / std::pow(ROPE_THETA, exponent);
        const double freq_inter =
            1.0 / (SCALING_FACTOR * std::pow(ROPE_THETA, exponent));
        const double linear = (static_cast<double>(k) - low) / (high - low);
        double ramp = linear;
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        const double mask = 1.0 - ramp;
        const double f = freq_inter * (1.0 - mask) + freq_extra * mask;
        inv_freq[k] = static_cast<float>(f);
    }
    return inv_freq;
}

}  // namespace

// FP8 YaRN RoPE position embedding.
//
// definition.json declares:
//   position_ids       [seq_len]       int64
//   mscale             scalar          fp32
//   mscale_all_dim     scalar          fp32
// Outputs (destination-passing style):
//   cos_emb            [seq_len, 64]   bf16
//   sin_emb            [seq_len, 64]   bf16
//
// The reference applies a faithful BlockWise1x128 FP8 e4m3 quantize/
// dequantize to the per-position frequency vector before computing
// cos / sin. We mirror that exactly inside the custom kernel — no
// precision substitution — so correctness lands within the upstream
// tolerance band.
void run(
    const torch::Tensor& position_ids,
    double mscale,
    double mscale_all_dim,
    torch::Tensor cos_emb,
    torch::Tensor sin_emb
) {
    TORCH_CHECK(position_ids.dim() == 1, "position_ids must be 1D");
    TORCH_CHECK(position_ids.dtype() == torch::kInt64,
                "position_ids must be int64");
    TORCH_CHECK(position_ids.is_cuda(), "position_ids must be on CUDA");
    TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");

    const int64_t seq_len = position_ids.size(0);
    TORCH_CHECK(
        cos_emb.dim() == 2 && cos_emb.size(0) == seq_len && cos_emb.size(1) == QK_ROPE_HEAD_DIM,
        "cos_emb must be [seq_len, 64]");
    TORCH_CHECK(
        sin_emb.dim() == 2 && sin_emb.size(0) == seq_len && sin_emb.size(1) == QK_ROPE_HEAD_DIM,
        "sin_emb must be [seq_len, 64]");
    TORCH_CHECK(cos_emb.dtype() == torch::kBFloat16, "cos_emb must be bf16");
    TORCH_CHECK(sin_emb.dtype() == torch::kBFloat16, "sin_emb must be bf16");
    TORCH_CHECK(cos_emb.is_contiguous() && sin_emb.is_contiguous(),
                "cos/sin must be contiguous");

    // Precompute the 32 inv_freq values on the host (constants only).
    const std::vector<float> inv_freq = compute_inv_freq();

    // mscale_combined = yarn_get_mscale(scaling, mscale)
    //                   / yarn_get_mscale(scaling, mscale_all_dim)
    const double mscale_num =
        yarn_get_mscale(SCALING_FACTOR, mscale);
    const double mscale_den =
        yarn_get_mscale(SCALING_FACTOR, mscale_all_dim);
    const float mscale_combined = static_cast<float>(mscale_num / mscale_den);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_014_fp8_yarn_rope_embedding_launcher(
        position_ids,
        inv_freq.data(),
        mscale_combined,
        cos_emb,
        sin_emb,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "FP8 YaRN RoPE position embedding (qk_rope_head_dim=64, "
        "BlockWise1x128 e4m3 quantize-dequantize on per-row freqs).");
}
