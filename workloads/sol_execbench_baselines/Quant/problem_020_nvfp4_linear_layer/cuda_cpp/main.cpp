#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "kernel.h"

namespace {

constexpr int SF_VEC_SIZE = 16;
constexpr float E2M1_MAX = 6.0f;
constexpr float FLOAT8_E4M3_MAX = 448.0f;

// Quantize x ([1, M, K] FP32) into the same
// (decode_scale_fp8, global_decode_scale, scaled_x_fp32) triple that
// the reference's ``BlockWiseScalerNVFP4.quantize`` produces.
struct QuantPrepass {
    at::Tensor scaled_x;          // [M, K]   FP32
    at::Tensor decode_scale_fp8;  // [M, K/16] FP8 e4m3
    float global_decode_scale;
};

QuantPrepass quantize_x(const at::Tensor& x_batched_fp32) {
    TORCH_CHECK(x_batched_fp32.dim() == 3
                    && x_batched_fp32.size(0) == 1,
                "x must be [1, M, K]");
    const int64_t M = x_batched_fp32.size(1);
    const int64_t K = x_batched_fp32.size(2);
    TORCH_CHECK(K % SF_VEC_SIZE == 0,
                "K must be a multiple of SF_VEC_SIZE (16)");

    const float global_amax =
        x_batched_fp32.abs().max().item<float>();

    const float fp32_max = std::numeric_limits<float>::max();
    float global_encode_scale;
    if (global_amax > 0.0f) {
        global_encode_scale = FLOAT8_E4M3_MAX * E2M1_MAX / global_amax;
        if (global_encode_scale > fp32_max) {
            global_encode_scale = fp32_max;
        }
    } else {
        global_encode_scale = 1.0f;
    }
    const float global_decode_scale = 1.0f / global_encode_scale;

    auto blocked = x_batched_fp32.view({1, M, K / SF_VEC_SIZE, SF_VEC_SIZE});
    auto block_max = blocked.abs().amax(-1);
    auto decode_scale = (block_max / E2M1_MAX) * global_encode_scale;
    decode_scale = decode_scale.clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX);
    auto decode_scale_fp8 = decode_scale.to(at::kFloat8_e4m3fn);

    auto encode_scale = (1.0f
        / (decode_scale_fp8.to(at::kFloat) * global_decode_scale))
        .clamp_max(fp32_max);

    auto scaled_x = blocked * encode_scale.unsqueeze(-1);
    auto clipped = scaled_x.clamp(-E2M1_MAX, E2M1_MAX)
        .reshape({M, K})
        .contiguous();

    QuantPrepass out;
    out.scaled_x = clipped;
    out.decode_scale_fp8 = decode_scale_fp8.squeeze(0).contiguous();
    out.global_decode_scale = global_decode_scale;
    return out;
}

}  // namespace

void run(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp4,
    const torch::Tensor& weight_scales,
    double weight_global_decode_scale,
    torch::Tensor output) {
    TORCH_CHECK(x.is_cuda() && weight_fp4.is_cuda()
                    && weight_scales.is_cuda() && output.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 3, "x must be [B, S, K]");
    TORCH_CHECK(output.dim() == 3, "output must be [B, S, N]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(output.dtype() == torch::kBFloat16,
                "output must be bf16");

    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t K = x.size(2);
    const int64_t N = weight_fp4.size(0);
    TORCH_CHECK(weight_fp4.size(1) * 2 == K,
                "weight_fp4 second dim must equal K/2");
    TORCH_CHECK(weight_scales.size(0) == N
                    && weight_scales.size(1) == K / SF_VEC_SIZE,
                "weight_scales shape must be [N, K/16]");
    TORCH_CHECK(output.size(0) == B && output.size(1) == S
                    && output.size(2) == N,
                "output shape must be [B, S, N]");

    const int64_t M = B * S;
    const int64_t K_blocks = K / SF_VEC_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. Quant prepass on x (mirrors reference's BlockWiseScalerNVFP4).
    auto x_flat = x.reshape({M, K}).to(at::kFloat).contiguous();
    auto x_batched = x_flat.unsqueeze(0);
    QuantPrepass q = quantize_x(x_batched);

    // 2. Pack scaled FP32 x into FP4 (E2M1 nibbles).
    auto x_fp4 = at::empty(
        {M, K / 2}, x.options().dtype(at::kByte));
    problem_020_fp4_pack_launcher(q.scaled_x, x_fp4, stream);

    // 3. Make sure weight_fp4 is contiguous uint8.
    at::Tensor w_bytes;
    if (weight_fp4.scalar_type() == at::kByte) {
        w_bytes = weight_fp4;
    } else {
        w_bytes = weight_fp4.view(at::kByte);
    }
    TORCH_CHECK(w_bytes.is_contiguous(),
                "weight_fp4 must be contiguous");

    // 4. Allocate SFA / SFB buffers in CUTLASS interleaved layout
    //    (sized at D * K_blocks, same element count as the plain
    //    layout — only the index pattern differs).
    auto x_sfa = at::empty(
        {M * K_blocks},
        x.options().dtype(at::kFloat8_e4m3fn));
    auto w_sfb = at::empty(
        {N * K_blocks},
        x.options().dtype(at::kFloat8_e4m3fn));

    // 5. Remap plain [D, K_blocks] FP8 scales into CUTLASS layout.
    problem_020_remap_scales_launcher(
        q.decode_scale_fp8, x_sfa, M, K_blocks, stream);
    problem_020_remap_scales_launcher(
        weight_scales, w_sfb, N, K_blocks, stream);

    // 6. Dispatch CUTLASS sm_120 dense block-scaled NVFP4 GEMM.
    const float global_scale =
        q.global_decode_scale * static_cast<float>(weight_global_decode_scale);
    auto y_2d = output.view({M, N});
    problem_020_fp4_gemm_launcher(
        x_fp4, x_sfa, w_bytes, w_sfb,
        y_2d, global_scale, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "NVFP4 linear layer (FP4-preserving, FP32 accumulator, BF16 out) "
        "via CUTLASS sm_120 dense block-scaled tensor cores.");
}
