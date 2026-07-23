#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <string>

// Video spatial attention with 3D rotary position embedding.
//
// Reference (problem_027_video_spatial_attention_with_rope_3d), reproduced
// op-for-op. Multi-head self-attention over video tokens, where queries and
// keys carry a 3D RoPE that encodes temporal (frame), height, and width
// positions before the attention scores are formed.
//
//   num_attention_heads = 16 ; head_dim = 64
//   seq_len = num_frames * num_patches_per_frame
//
//   hidden_states_flat = hidden_states.reshape(B, seq_len, hidden_size)
//   qkv = linear(hidden_states_flat, qkv_weight, qkv_bias)          # [B,seq,3*hidden]
//   qkv = qkv.reshape(B, seq, 3, 16, 64).permute(2,0,3,1,4)         # [3,B,16,seq,64]
//   q, k, v = qkv[0], qkv[1], qkv[2]
//   q = apply_rope_3d(q, ...) ; k = apply_rope_3d(k, ...)
//   attn_scores = (q @ k.transpose(-2,-1)) * scale                  # [B,16,seq,seq]
//   attn_probs  = softmax(attn_scores, dim=-1, dtype=fp32)
//   attn_output = attn_probs @ v                                    # [B,16,seq,64]
//   attn_output = attn_output.transpose(1,2).reshape(B,seq,hidden)
//   output = linear(attn_output, out_weight, out_bias)
//   output = output.reshape(B, num_frames, num_patches, hidden)
//
// Position indices (as the reference derives them):
//   frame_positions  = arange(num_frames)[:,None].expand(num_frames, num_patches).reshape(-1)
//   patches_per_side = floor(sqrt(num_patches)); if not a perfect square, ceil(sqrt(num_patches))
//   height_positions = (arange(num_patches) // patches_per_side) broadcast over frames
//   width_positions  = (arange(num_patches) %  patches_per_side) broadcast over frames
//
// 3D RoPE segment construction (head_dim = 64):
//   temporal_dim = head_dim // 3 = 21
//   spatial_dim  = (head_dim - temporal_dim) // 2 = 21
//   remaining_dim= head_dim - temporal_dim - 2*spatial_dim = 1
//   x_temporal = x[..., :21]
//   x_height   = x[..., 21:42]
//   x_width    = x[..., 42:63]
//   x_remaining= x[..., 63:64]   (carried through untouched)
//
//   Per segment, apply_rope_rotation(x_part, positions, freqs):
//     dim = x_part.size(-1); half_dim = dim // 2
//     angles = positions[...,None] * freqs[:half_dim]
//     cos = cos(angles); sin = sin(angles)
//     x1 = x_part[..., 0::2]; x2 = x_part[..., 1::2]
//     min_dim = min(x1.size(-1), cos.size(-1))   (truncate the leading paired lanes)
//     x1,x2,cos,sin = (each)[..., :min_dim]
//     x_rot1 = x1*cos - x2*sin ; x_rot2 = x1*sin + x2*cos
//     x_out  = stack([x_rot1, x_rot2], dim=-1).flatten(-2)
//     if dim is odd: x_out = cat([x_out, x_part[..., -1:]], dim=-1)
//   The temporal segment uses temporal_freqs and frame positions; height and
//   width use spatial_freqs with height and width positions respectively.
//
// Per definition.json (insertion order):
//   hidden_states  : [batch_size, num_frames, num_patches_per_frame, hidden_size] fp32
//   qkv_weight     : [qkv_hidden=3*hidden, hidden_size]                           fp32
//   qkv_bias       : [qkv_hidden]                                                 fp32
//   out_weight     : [hidden_size, hidden_size]                                   fp32
//   out_bias       : [hidden_size]                                                fp32
//   temporal_freqs : [half_head_dim=32]                                           fp32
//   spatial_freqs  : [half_head_dim=32]                                           fp32
//   scale          : scalar                                                       fp32
//   output         : [batch_size, num_frames, num_patches_per_frame, hidden_size] fp32 DPS
//
// Implementation: pure ATen / cuBLAS / cuDNN dispatch -- NO custom CUDA
// kernels. Linears go through at::linear (cuBLAS GEMM); softmax stays in the
// native fp32 dtype. The RoPE rotation is built with at::arange, at::cos /
// at::sin, even / odd lane selection via strided at::slice, at::stack /
// flatten, and at::cat, matching the reference's truncate-then-rotate
// construction including the odd-width carry of the trailing lane. Output is
// written in place into the pre-allocated destination (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// One RoPE rotation applied to a [B, heads, seq, dim] segment with a per-token
// position vector [seq] and a frequency vector. Mirrors apply_rope_rotation in
// the reference, including the min_dim truncation and the odd-width carry.
static torch::Tensor apply_rope_rotation(
    const torch::Tensor& x_part,
    const torch::Tensor& positions,
    const torch::Tensor& freqs
) {
    const int64_t dim = x_part.size(-1);
    const int64_t half_dim = dim / 2;
    if (half_dim == 0) {
        return x_part;
    }

    // angles = positions[..., None] * freqs[:half_dim]  -> [seq, half_dim]
    auto freqs_head = freqs.slice(/*dim=*/0, /*start=*/0, /*end=*/half_dim);
    auto angles = positions.unsqueeze(-1) * freqs_head;
    auto cos_vals = at::cos(angles);
    auto sin_vals = at::sin(angles);

    // Even / odd lanes along the last axis.
    auto x1 = x_part.slice(/*dim=*/-1, /*start=*/0, /*end=*/dim, /*step=*/2);
    auto x2 = x_part.slice(/*dim=*/-1, /*start=*/1, /*end=*/dim, /*step=*/2);

    const int64_t min_dim = std::min(x1.size(-1), cos_vals.size(-1));
    x1 = x1.slice(/*dim=*/-1, /*start=*/0, /*end=*/min_dim);
    x2 = x2.slice(/*dim=*/-1, /*start=*/0, /*end=*/min_dim);
    cos_vals = cos_vals.slice(/*dim=*/-1, /*start=*/0, /*end=*/min_dim);
    sin_vals = sin_vals.slice(/*dim=*/-1, /*start=*/0, /*end=*/min_dim);

    // Broadcast cos / sin over batch and head axes (reference unsqueezes twice).
    auto cos_b = cos_vals.unsqueeze(0).unsqueeze(0);
    auto sin_b = sin_vals.unsqueeze(0).unsqueeze(0);

    auto x_rotated_1 = x1 * cos_b - x2 * sin_b;
    auto x_rotated_2 = x1 * sin_b + x2 * cos_b;

    auto x_out = at::stack({x_rotated_1, x_rotated_2}, /*dim=*/-1).flatten(/*start_dim=*/-2);

    if (dim % 2 == 1) {
        auto tail = x_part.slice(/*dim=*/-1, /*start=*/dim - 1, /*end=*/dim);
        x_out = at::cat({x_out, tail}, /*dim=*/-1);
    }
    return x_out;
}

// Apply the full 3D RoPE to a [B, heads, seq, head_dim] tensor: split into the
// temporal / height / width / remaining segments, rotate the first three with
// their respective positions and frequency tables, carry the remainder, then
// concatenate back.
static torch::Tensor apply_rope_3d(
    const torch::Tensor& x,
    const torch::Tensor& frame_pos,
    const torch::Tensor& height_pos,
    const torch::Tensor& width_pos,
    const torch::Tensor& temporal_freqs,
    const torch::Tensor& spatial_freqs
) {
    const int64_t head_dim = x.size(-1);
    const int64_t temporal_dim = head_dim / 3;
    const int64_t spatial_dim = (head_dim - temporal_dim) / 2;

    auto x_temporal = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/temporal_dim);
    auto x_height = x.slice(/*dim=*/-1, /*start=*/temporal_dim,
                            /*end=*/temporal_dim + spatial_dim);
    auto x_width = x.slice(/*dim=*/-1, /*start=*/temporal_dim + spatial_dim,
                           /*end=*/temporal_dim + 2 * spatial_dim);
    auto x_remaining = x.slice(/*dim=*/-1, /*start=*/temporal_dim + 2 * spatial_dim,
                               /*end=*/head_dim);

    x_temporal = apply_rope_rotation(x_temporal, frame_pos, temporal_freqs);
    x_height = apply_rope_rotation(x_height, height_pos, spatial_freqs);
    x_width = apply_rope_rotation(x_width, width_pos, spatial_freqs);

    return at::cat({x_temporal, x_height, x_width, x_remaining}, /*dim=*/-1);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& out_weight,
    const torch::Tensor& out_bias,
    const torch::Tensor& temporal_freqs,
    const torch::Tensor& spatial_freqs,
    double scale,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(qkv_weight, "qkv_weight");
    check_fp32_cuda(qkv_bias, "qkv_bias");
    check_fp32_cuda(out_weight, "out_weight");
    check_fp32_cuda(out_bias, "out_bias");
    check_fp32_cuda(temporal_freqs, "temporal_freqs");
    check_fp32_cuda(spatial_freqs, "spatial_freqs");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 4,
                "hidden_states must be 4D [B, num_frames, num_patches, hidden], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = 64;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t num_frames = hidden_states.size(1);
    const int64_t num_patches = hidden_states.size(2);
    const int64_t hidden_size = hidden_states.size(3);
    const int64_t seq_len = num_frames * num_patches;

    // --- Position indices (float), derived exactly as the reference does. ---
    const auto device = hidden_states.device();
    const auto long_opts = at::TensorOptions().dtype(at::kLong).device(device);

    // frame_positions: arange(num_frames)[:,None].expand(num_frames, num_patches).reshape(-1)
    auto frame_positions = at::arange(num_frames, long_opts)
                               .unsqueeze(1)
                               .expand({num_frames, num_patches})
                               .reshape({-1})
                               .to(at::kFloat);

    // patches_per_side = floor(sqrt(num_patches)); if not a perfect square, ceil(sqrt).
    int64_t patches_per_side = static_cast<int64_t>(
        std::sqrt(static_cast<double>(num_patches)));
    if (patches_per_side * patches_per_side != num_patches) {
        patches_per_side = static_cast<int64_t>(
            std::ceil(std::sqrt(static_cast<double>(num_patches))));
    }

    auto patch_arange = at::arange(num_patches, long_opts);
    auto height_idx = patch_arange.div(patches_per_side, /*rounding_mode=*/"floor");
    auto width_idx = patch_arange.remainder(patches_per_side);
    auto height_positions = height_idx.unsqueeze(0)
                                .expand({num_frames, num_patches})
                                .reshape({-1})
                                .to(at::kFloat);
    auto width_positions = width_idx.unsqueeze(0)
                               .expand({num_frames, num_patches})
                               .reshape({-1})
                               .to(at::kFloat);

    // --- QKV projection + bias. ---
    auto hidden_states_flat = hidden_states.reshape({batch_size, seq_len, hidden_size});
    auto qkv = at::linear(hidden_states_flat, qkv_weight, qkv_bias);  // [B,seq,3*hidden]

    // --- Reshape and split into q, k, v ([B,16,seq,64] each). ---
    qkv = qkv.reshape({batch_size, seq_len, 3, kNumHeads, kHeadDim});
    qkv = qkv.permute({2, 0, 3, 1, 4});  // [3,B,16,seq,64]
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    // --- 3D RoPE on q and k. ---
    q = apply_rope_3d(q, frame_positions, height_positions, width_positions,
                      temporal_freqs, spatial_freqs);
    k = apply_rope_3d(k, frame_positions, height_positions, width_positions,
                      temporal_freqs, spatial_freqs);

    // --- Attention scores, scale, native-dtype softmax, attn @ V. ---
    auto attn_scores = at::matmul(q, k.transpose(-2, -1)) * scale;  // [B,16,seq,seq]
    auto attn_probs = at::softmax(attn_scores, /*dim=*/-1);          // fp32 (native)
    auto attn_output = at::matmul(attn_probs, v);                    // [B,16,seq,64]

    // --- Merge heads, output projection + bias, reshape to source layout. ---
    attn_output = attn_output.transpose(1, 2).reshape({batch_size, seq_len, hidden_size});
    auto out = at::linear(attn_output, out_weight, out_bias);  // [B,seq,hidden]
    out = out.reshape({batch_size, num_frames, num_patches, hidden_size});

    // --- Destination-passing-style write. ---
    TORCH_CHECK(out.sizes() == output.sizes(),
                "output shape ", out.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Video spatial attention with 3D RoPE. Pure ATen / cuBLAS / cuDNN "
          "dispatch: QKV projection + bias, reshape/permute to heads, 3D RoPE "
          "(temporal / height / width segments rotated with interleaved "
          "even-odd lanes and per-token frame / height / width positions) on Q "
          "and K, scaled QK^T, fp32 softmax, attn @ V, head merge, output "
          "projection + bias. fp32; destination-passing style.");
}
