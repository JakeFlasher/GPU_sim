#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <vector>
#include <string>
#include <cmath>

// Video-latent denoising U-Net block reproduced op-for-op from the reference
// run(). Four pre-norm + residual sub-blocks run in sequence over the video
// latents:
//   1. Temporal self-attention -- attention runs across frames for each fixed
//      spatial token (batch * spatial tokens are folded into the attention
//      batch, frames are the sequence axis).
//   2. Spatial self-attention -- attention runs across spatial tokens for each
//      fixed frame (batch * frames folded into the attention batch, spatial
//      tokens are the sequence axis).
//   3. Cross-attention -- queries from the video latents attend to keys / values
//      projected from the text embeddings.
//   4. GELU feed-forward -- a two-layer MLP with a GELU nonlinearity.
// Each sub-block layer-norms its residual input, runs its dense multi-head
// attention (or MLP), and adds the residual back.
//
// Fixed constants from the spec:
//   hidden_size         = 1024
//   num_attention_heads = 16
//   head_dim            = hidden_size / num_attention_heads = 64
//   intermediate_size   = 4096
//   scale               = 1 / sqrt(head_dim)
//   eps                 = 1e-5
//
// Per definition.json (insertion order) -- the 28 tensor inputs, then the two
// int32 scalars, then the output:
//   video_latents               : [B, video_seq_len, H]   fp32
//   text_embeddings             : [B, text_seq_len, H]     fp32
//   temporal_norm_weight        : [H]                      fp32
//   temporal_norm_bias          : [H]                      fp32
//   temporal_qkv_weight         : [3H, H]                  fp32
//   temporal_qkv_bias           : [3H]                     fp32
//   temporal_out_proj_weight    : [H, H]                   fp32
//   temporal_out_proj_bias      : [H]                      fp32
//   spatial_norm_weight         : [H]                      fp32
//   spatial_norm_bias           : [H]                      fp32
//   spatial_qkv_weight          : [3H, H]                  fp32
//   spatial_qkv_bias            : [3H]                     fp32
//   spatial_out_proj_weight     : [H, H]                   fp32
//   spatial_out_proj_bias       : [H]                      fp32
//   cross_attn_norm_weight      : [H]                      fp32
//   cross_attn_norm_bias        : [H]                      fp32
//   cross_attn_q_weight         : [H, H]                   fp32
//   cross_attn_q_bias           : [H]                      fp32
//   cross_attn_kv_weight        : [2H, H]                  fp32
//   cross_attn_kv_bias          : [2H]                     fp32
//   cross_attn_out_proj_weight  : [H, H]                   fp32
//   cross_attn_out_proj_bias    : [H]                      fp32
//   ffn_norm_weight             : [H]                      fp32
//   ffn_norm_bias               : [H]                      fp32
//   ffn_fc1_weight              : [I, H]                   fp32
//   ffn_fc1_bias                : [I]                      fp32
//   ffn_fc2_weight              : [H, I]                   fp32
//   ffn_fc2_bias                : [H]                      fp32
//   num_frames_scalar           : scalar                   int32 -> int64_t
//   num_spatial_tokens_scalar   : scalar                   int32 -> int64_t
//   output                      : [B, video_seq_len, H]    fp32 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels.
// at::layer_norm carries the spec's normalized_shape / eps; at::linear /
// at::matmul / at::softmax / at::gelu match the reference ops at the native
// fp32 precision profile. The output is written in place into the
// pre-allocated destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& video_latents,
    const torch::Tensor& text_embeddings,
    const torch::Tensor& temporal_norm_weight,
    const torch::Tensor& temporal_norm_bias,
    const torch::Tensor& temporal_qkv_weight,
    const torch::Tensor& temporal_qkv_bias,
    const torch::Tensor& temporal_out_proj_weight,
    const torch::Tensor& temporal_out_proj_bias,
    const torch::Tensor& spatial_norm_weight,
    const torch::Tensor& spatial_norm_bias,
    const torch::Tensor& spatial_qkv_weight,
    const torch::Tensor& spatial_qkv_bias,
    const torch::Tensor& spatial_out_proj_weight,
    const torch::Tensor& spatial_out_proj_bias,
    const torch::Tensor& cross_attn_norm_weight,
    const torch::Tensor& cross_attn_norm_bias,
    const torch::Tensor& cross_attn_q_weight,
    const torch::Tensor& cross_attn_q_bias,
    const torch::Tensor& cross_attn_kv_weight,
    const torch::Tensor& cross_attn_kv_bias,
    const torch::Tensor& cross_attn_out_proj_weight,
    const torch::Tensor& cross_attn_out_proj_bias,
    const torch::Tensor& ffn_norm_weight,
    const torch::Tensor& ffn_norm_bias,
    const torch::Tensor& ffn_fc1_weight,
    const torch::Tensor& ffn_fc1_bias,
    const torch::Tensor& ffn_fc2_weight,
    const torch::Tensor& ffn_fc2_bias,
    int64_t num_frames_scalar,
    int64_t num_spatial_tokens_scalar,
    torch::Tensor output
) {
    check_fp32_cuda(video_latents, "video_latents");
    check_fp32_cuda(text_embeddings, "text_embeddings");
    check_fp32_cuda(temporal_norm_weight, "temporal_norm_weight");
    check_fp32_cuda(temporal_norm_bias, "temporal_norm_bias");
    check_fp32_cuda(temporal_qkv_weight, "temporal_qkv_weight");
    check_fp32_cuda(temporal_qkv_bias, "temporal_qkv_bias");
    check_fp32_cuda(temporal_out_proj_weight, "temporal_out_proj_weight");
    check_fp32_cuda(temporal_out_proj_bias, "temporal_out_proj_bias");
    check_fp32_cuda(spatial_norm_weight, "spatial_norm_weight");
    check_fp32_cuda(spatial_norm_bias, "spatial_norm_bias");
    check_fp32_cuda(spatial_qkv_weight, "spatial_qkv_weight");
    check_fp32_cuda(spatial_qkv_bias, "spatial_qkv_bias");
    check_fp32_cuda(spatial_out_proj_weight, "spatial_out_proj_weight");
    check_fp32_cuda(spatial_out_proj_bias, "spatial_out_proj_bias");
    check_fp32_cuda(cross_attn_norm_weight, "cross_attn_norm_weight");
    check_fp32_cuda(cross_attn_norm_bias, "cross_attn_norm_bias");
    check_fp32_cuda(cross_attn_q_weight, "cross_attn_q_weight");
    check_fp32_cuda(cross_attn_q_bias, "cross_attn_q_bias");
    check_fp32_cuda(cross_attn_kv_weight, "cross_attn_kv_weight");
    check_fp32_cuda(cross_attn_kv_bias, "cross_attn_kv_bias");
    check_fp32_cuda(cross_attn_out_proj_weight, "cross_attn_out_proj_weight");
    check_fp32_cuda(cross_attn_out_proj_bias, "cross_attn_out_proj_bias");
    check_fp32_cuda(ffn_norm_weight, "ffn_norm_weight");
    check_fp32_cuda(ffn_norm_bias, "ffn_norm_bias");
    check_fp32_cuda(ffn_fc1_weight, "ffn_fc1_weight");
    check_fp32_cuda(ffn_fc1_bias, "ffn_fc1_bias");
    check_fp32_cuda(ffn_fc2_weight, "ffn_fc2_weight");
    check_fp32_cuda(ffn_fc2_bias, "ffn_fc2_bias");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(video_latents.dim() == 3,
                "video_latents must be 3D [B, video_seq_len, hidden_size], got ",
                video_latents.dim(), "D");
    TORCH_CHECK(text_embeddings.dim() == 3,
                "text_embeddings must be 3D [B, text_seq_len, hidden_size], got ",
                text_embeddings.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(video_latents.device());

    constexpr int64_t kHiddenSize = 1024;
    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = kHiddenSize / kNumHeads;  // 64
    constexpr double kEps = 1e-5;
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    const int64_t batch_size = video_latents.size(0);
    const int64_t video_seq_len = video_latents.size(1);
    const int64_t text_seq_len = text_embeddings.size(1);
    const int64_t num_frames = num_frames_scalar;
    const int64_t num_spatial_tokens = num_spatial_tokens_scalar;

    const std::vector<int64_t> normalized_shape{kHiddenSize};

    auto x = video_latents;

    // ===================== 1. Temporal self-attention =====================
    auto residual = x;
    auto x_norm = at::layer_norm(x, normalized_shape, temporal_norm_weight,
                                 temporal_norm_bias, kEps);

    // Fused QKV projection -> [B, F, S, 3, H], rearranged to [B, S, F, 3, H].
    auto qkv = at::linear(x_norm, temporal_qkv_weight, temporal_qkv_bias);
    qkv = qkv.view(std::vector<int64_t>{batch_size, num_frames, num_spatial_tokens, 3, kHiddenSize});
    qkv = qkv.permute({0, 2, 1, 3, 4});
    auto qkv_chunks = qkv.chunk(3, /*dim=*/3);
    auto q = qkv_chunks[0].squeeze(3);  // [B, S, F, H]
    auto k = qkv_chunks[1].squeeze(3);
    auto v = qkv_chunks[2].squeeze(3);

    // Fold (B, S) into the attention batch; frames become the sequence axis.
    q = q.reshape(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kHiddenSize});
    k = k.reshape(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kHiddenSize});
    v = v.reshape(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kHiddenSize});

    q = q.view(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kNumHeads, kHeadDim}).transpose(1, 2);
    k = k.view(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kNumHeads, kHeadDim}).transpose(1, 2);
    v = v.view(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kNumHeads, kHeadDim}).transpose(1, 2);

    auto attn_scores = at::matmul(q, k.transpose(-2, -1)) * scale;
    auto attn_probs = at::softmax(attn_scores, /*dim=*/-1);
    auto attn_output = at::matmul(attn_probs, v);

    // Restore frame-major [B, F*S, H] order before the output projection.
    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output = attn_output.view(std::vector<int64_t>{batch_size * num_spatial_tokens, num_frames, kHiddenSize});
    attn_output = attn_output.view(std::vector<int64_t>{batch_size, num_spatial_tokens, num_frames, kHiddenSize});
    attn_output = attn_output.permute({0, 2, 1, 3}).contiguous();
    attn_output = attn_output.view(std::vector<int64_t>{batch_size, num_frames * num_spatial_tokens, kHiddenSize});

    auto proj = at::linear(attn_output, temporal_out_proj_weight, temporal_out_proj_bias);
    x = proj + residual;

    // ===================== 2. Spatial self-attention =====================
    residual = x;
    x_norm = at::layer_norm(x, normalized_shape, spatial_norm_weight,
                            spatial_norm_bias, kEps);

    // Split frames apart, fold (B, F) into the attention batch; spatial tokens
    // become the sequence axis.
    auto x_reshaped = x_norm.view(std::vector<int64_t>{batch_size, num_frames, num_spatial_tokens, kHiddenSize});
    qkv = at::linear(
        x_reshaped.view(std::vector<int64_t>{batch_size * num_frames, num_spatial_tokens, kHiddenSize}),
        spatial_qkv_weight, spatial_qkv_bias);
    qkv = qkv.view(std::vector<int64_t>{batch_size * num_frames, num_spatial_tokens, 3, kHiddenSize});
    qkv_chunks = qkv.chunk(3, /*dim=*/2);
    q = qkv_chunks[0].squeeze(2);
    k = qkv_chunks[1].squeeze(2);
    v = qkv_chunks[2].squeeze(2);

    q = q.view(std::vector<int64_t>{batch_size * num_frames, num_spatial_tokens, kNumHeads, kHeadDim}).transpose(1, 2);
    k = k.view(std::vector<int64_t>{batch_size * num_frames, num_spatial_tokens, kNumHeads, kHeadDim}).transpose(1, 2);
    v = v.view(std::vector<int64_t>{batch_size * num_frames, num_spatial_tokens, kNumHeads, kHeadDim}).transpose(1, 2);

    attn_scores = at::matmul(q, k.transpose(-2, -1)) * scale;
    attn_probs = at::softmax(attn_scores, /*dim=*/-1);
    attn_output = at::matmul(attn_probs, v);

    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output = attn_output.view(std::vector<int64_t>{batch_size * num_frames, num_spatial_tokens, kHiddenSize});
    attn_output = attn_output.view(std::vector<int64_t>{batch_size, num_frames * num_spatial_tokens, kHiddenSize});

    proj = at::linear(attn_output, spatial_out_proj_weight, spatial_out_proj_bias);
    x = proj + residual;

    // ===================== 3. Cross-attention with text =====================
    residual = x;
    x_norm = at::layer_norm(x, normalized_shape, cross_attn_norm_weight,
                            cross_attn_norm_bias, kEps);

    // Query from the video latents; key / value from the text embeddings.
    q = at::linear(x_norm, cross_attn_q_weight, cross_attn_q_bias);
    auto kv = at::linear(text_embeddings, cross_attn_kv_weight, cross_attn_kv_bias);
    kv = kv.view(std::vector<int64_t>{batch_size, text_seq_len, 2, kHiddenSize});
    auto kv_chunks = kv.chunk(2, /*dim=*/2);
    k = kv_chunks[0].squeeze(2);
    v = kv_chunks[1].squeeze(2);

    q = q.view(std::vector<int64_t>{batch_size, video_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    k = k.view(std::vector<int64_t>{batch_size, text_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);
    v = v.view(std::vector<int64_t>{batch_size, text_seq_len, kNumHeads, kHeadDim}).transpose(1, 2);

    attn_scores = at::matmul(q, k.transpose(-2, -1)) * scale;
    attn_probs = at::softmax(attn_scores, /*dim=*/-1);
    attn_output = at::matmul(attn_probs, v);

    attn_output = attn_output.transpose(1, 2).contiguous();
    attn_output = attn_output.view(std::vector<int64_t>{batch_size, video_seq_len, kHiddenSize});

    proj = at::linear(attn_output, cross_attn_out_proj_weight, cross_attn_out_proj_bias);
    x = proj + residual;

    // ===================== 4. GELU feed-forward =====================
    residual = x;
    x_norm = at::layer_norm(x, normalized_shape, ffn_norm_weight,
                            ffn_norm_bias, kEps);

    auto x_ffn = at::linear(x_norm, ffn_fc1_weight, ffn_fc1_bias);
    x_ffn = at::gelu(x_ffn);
    x_ffn = at::linear(x_ffn, ffn_fc2_weight, ffn_fc2_bias);
    auto result = x_ffn + residual;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Video-latent denoising U-Net block via pure ATen dispatch: four "
          "pre-norm + residual sub-blocks -- temporal self-attention (frames "
          "as the sequence axis), spatial self-attention (spatial tokens as the "
          "sequence axis), text cross-attention (query from video latents, "
          "key / value from text embeddings), and a GELU feed-forward MLP; "
          "at::layer_norm / at::linear / at::softmax / at::gelu match the "
          "reference ops; no custom CUDA kernels; fp32; destination-passing "
          "style (output).");
}
