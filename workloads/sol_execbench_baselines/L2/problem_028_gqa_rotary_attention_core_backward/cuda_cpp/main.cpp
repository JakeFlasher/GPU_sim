#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the GQA rotary attention
// core backward (GLM-style interleaved RoPE pattern). Per
// definition.json:
//   grad_output         : [B, S, hidden_size=8192]                    bf16
//   hidden_states       : [B, S, hidden_size]                         bf16
//   q_weight            : [hidden_size, hidden_size]                  bf16
//   k_weight            : [kv_hidden_size=1024, hidden_size]          bf16
//   v_weight            : [kv_hidden_size, hidden_size]               bf16
//   o_weight            : [hidden_size, hidden_size]                  bf16
//   inv_freq            : [half_head_dim=64]                          fp32
//   scaling             : scalar fp32 (1/sqrt(head_dim))
//   grad_hidden_states  : [B, S, hidden_size]                         bf16 DPS
//   grad_q_weight       : [hidden_size, hidden_size]                  bf16 DPS
//   grad_k_weight       : [kv_hidden_size, hidden_size]               bf16 DPS
//   grad_v_weight       : [kv_hidden_size, hidden_size]               bf16 DPS
//   grad_o_weight       : [hidden_size, hidden_size]                  bf16 DPS
//
// Note: the reference recomputes the full forward (no saved
// intermediates as inputs). The slug binds bf16 4D batched matmul
// dispatches on `cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N`
// (the verified family witness) plus 2D bf16 cublasGemmEx
// projection dispatches (Q/K/V via F.linear OP_N OP_T which
// binds the pending family; O grad_input via F.linear(g, W.t())
// which is OP_N OP_T pending family). The 4D backward dispatches
// are the dominant verified family witness.
void run(
    torch::Tensor grad_output,
    torch::Tensor hidden_states,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor o_weight,
    torch::Tensor inv_freq,
    double scaling,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight,
    torch::Tensor grad_o_weight
) {
    check_bf16_contiguous_cuda(grad_output, "grad_output");
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(q_weight, "q_weight");
    check_bf16_contiguous_cuda(k_weight, "k_weight");
    check_bf16_contiguous_cuda(v_weight, "v_weight");
    check_bf16_contiguous_cuda(o_weight, "o_weight");
    check_fp32_contiguous_cuda(inv_freq, "inv_freq");
    check_bf16_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_contiguous_cuda(grad_q_weight, "grad_q_weight");
    check_bf16_contiguous_cuda(grad_k_weight, "grad_k_weight");
    check_bf16_contiguous_cuda(grad_v_weight, "grad_v_weight");
    check_bf16_contiguous_cuda(grad_o_weight, "grad_o_weight");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 8192;
    constexpr int64_t kNumAttentionHeads = 64;
    constexpr int64_t kNumKeyValueHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kHalfHeadDim = kHeadDim / 2;  // 64
    constexpr int64_t kNumKeyValueGroups =
        kNumAttentionHeads / kNumKeyValueHeads;  // 8
    constexpr int64_t kKvHiddenSize = kNumKeyValueHeads * kHeadDim;  // 1024

    TORCH_CHECK(hidden_states.dim() == 3 &&
                hidden_states.size(2) == kHiddenSize,
                "hidden_states must be [B, S, hidden_size=8192]");
    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);

    TORCH_CHECK(grad_output.dim() == 3 &&
                grad_output.size(0) == B &&
                grad_output.size(1) == S &&
                grad_output.size(2) == kHiddenSize,
                "grad_output must be [B, S, hidden_size]");
    TORCH_CHECK(q_weight.dim() == 2 &&
                q_weight.size(0) == kHiddenSize &&
                q_weight.size(1) == kHiddenSize,
                "q_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(k_weight.dim() == 2 &&
                k_weight.size(0) == kKvHiddenSize &&
                k_weight.size(1) == kHiddenSize,
                "k_weight must be [kv_hidden_size=1024, hidden_size]");
    TORCH_CHECK(v_weight.dim() == 2 &&
                v_weight.size(0) == kKvHiddenSize &&
                v_weight.size(1) == kHiddenSize,
                "v_weight must be [kv_hidden_size, hidden_size]");
    TORCH_CHECK(o_weight.dim() == 2 &&
                o_weight.size(0) == kHiddenSize &&
                o_weight.size(1) == kHiddenSize,
                "o_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(inv_freq.dim() == 1 &&
                inv_freq.size(0) == kHalfHeadDim,
                "inv_freq must be [half_head_dim=64]");

    TORCH_CHECK(grad_hidden_states.dim() == 3 &&
                grad_hidden_states.size(0) == B &&
                grad_hidden_states.size(1) == S &&
                grad_hidden_states.size(2) == kHiddenSize,
                "grad_hidden_states must be [B, S, hidden_size]");
    TORCH_CHECK(grad_q_weight.dim() == 2 &&
                grad_q_weight.size(0) == kHiddenSize &&
                grad_q_weight.size(1) == kHiddenSize,
                "grad_q_weight must be [hidden_size, hidden_size]");
    TORCH_CHECK(grad_k_weight.dim() == 2 &&
                grad_k_weight.size(0) == kKvHiddenSize &&
                grad_k_weight.size(1) == kHiddenSize,
                "grad_k_weight must be [kv_hidden_size, hidden_size]");
    TORCH_CHECK(grad_v_weight.dim() == 2 &&
                grad_v_weight.size(0) == kKvHiddenSize &&
                grad_v_weight.size(1) == kHiddenSize,
                "grad_v_weight must be [kv_hidden_size, hidden_size]");
    TORCH_CHECK(grad_o_weight.dim() == 2 &&
                grad_o_weight.size(0) == kHiddenSize &&
                grad_o_weight.size(1) == kHiddenSize,
                "grad_o_weight must be [hidden_size, hidden_size]");

    const float scaling_f = static_cast<float>(scaling);

    // ================================================================
    // FORWARD RECOMPUTATION (no saved intermediates as inputs).
    // ================================================================
    auto query_states = at::linear(hidden_states, q_weight)
                            .view({B, S, kNumAttentionHeads, kHeadDim})
                            .transpose(1, 2);  // [B, H, S, d]
    auto key_states = at::linear(hidden_states, k_weight)
                          .view({B, S, kNumKeyValueHeads, kHeadDim})
                          .transpose(1, 2);  // [B, kv_H, S, d]
    auto value_states = at::linear(hidden_states, v_weight)
                            .view({B, S, kNumKeyValueHeads, kHeadDim})
                            .transpose(1, 2);  // [B, kv_H, S, d]

    // Compute RoPE cos / sin:
    //   position_ids = arange(S).unsqueeze(0)  [1, S]
    //   freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
    //         = [1, S, half_head_dim]
    //   emb = cat([freqs, freqs], dim=-1)  [1, S, head_dim]
    //   cos = emb.cos(); sin = emb.sin()
    auto pos_ids = at::arange(S, hidden_states.options()
                              .dtype(torch::kFloat32)).unsqueeze(0);
    auto inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(-1);
    auto pos_ids_expanded = pos_ids.unsqueeze(1);
    auto freqs = at::matmul(inv_freq_expanded, pos_ids_expanded)
                    .transpose(1, 2);
    auto emb = at::cat({freqs, freqs}, /*dim=*/-1);
    auto cos = emb.cos();
    auto sin = emb.sin();

    // Build cos_interleaved / sin_interleaved (GLM-style):
    //   cos_interleaved = cos[..., : head_dim/2].repeat_interleave(2, -1)
    auto cos_unsqueezed = cos.unsqueeze(1);  // [1, 1, S, head_dim]
    auto sin_unsqueezed = sin.unsqueeze(1);
    auto cos_interleaved =
        cos_unsqueezed.slice(-1, 0, kHalfHeadDim)
            .repeat_interleave(2, /*dim=*/-1);
    auto sin_interleaved =
        sin_unsqueezed.slice(-1, 0, kHalfHeadDim)
            .repeat_interleave(2, /*dim=*/-1);

    // Apply RoPE (GLM-style interleaved pattern, fp32 math).
    auto q_float = query_states.to(torch::kFloat32);
    auto k_float = key_states.to(torch::kFloat32);

    // q1 = q[..., 0::2]; q2 = q[..., 1::2];
    // q_rotated = stack((-q2, q1), -1).flatten(-2)
    auto q_unflat = q_float.unflatten(-1, {kHalfHeadDim, 2});
    auto q1 = q_unflat.select(-1, 0);  // [B, H, S, half_head_dim]
    auto q2 = q_unflat.select(-1, 1);
    auto q_rotated = at::stack({-q2, q1}, /*dim=*/-1).flatten(-2);
    auto k_unflat = k_float.unflatten(-1, {kHalfHeadDim, 2});
    auto k1 = k_unflat.select(-1, 0);
    auto k2 = k_unflat.select(-1, 1);
    auto k_rotated = at::stack({-k2, k1}, /*dim=*/-1).flatten(-2);

    auto query_states_rope_f32 = (q_float * cos_interleaved)
                                 + (q_rotated * sin_interleaved);
    auto key_states_rope_f32 = (k_float * cos_interleaved)
                               + (k_rotated * sin_interleaved);
    auto query_states_rope =
        query_states_rope_f32.to(torch::kBFloat16);
    auto key_states_rope =
        key_states_rope_f32.to(torch::kBFloat16);

    // Expand K / V heads (GQA): [B, kv_H, S, d] -> [B, num_H, S, d].
    auto key_states_expanded = key_states_rope.unsqueeze(2)
        .expand({B, kNumKeyValueHeads, kNumKeyValueGroups, S, kHeadDim})
        .reshape({B, kNumAttentionHeads, S, kHeadDim});
    auto value_states_expanded = value_states.unsqueeze(2)
        .expand({B, kNumKeyValueHeads, kNumKeyValueGroups, S, kHeadDim})
        .reshape({B, kNumAttentionHeads, S, kHeadDim});

    // Q @ K^T * scaling (4D bf16 batched).
    auto attn_weights_raw = at::matmul(
        query_states_rope, key_states_expanded.transpose(2, 3))
        * scaling_f;

    // Causal mask.
    auto causal_mask = at::triu(at::ones({S, S},
        hidden_states.options().dtype(torch::kBool)), /*diagonal=*/1);
    auto attn_weights_masked = attn_weights_raw.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0),
        -std::numeric_limits<float>::infinity());

    // Softmax in fp32.
    auto attn_weights_float = at::softmax(
        attn_weights_masked.to(torch::kFloat32), /*dim=*/-1);
    auto attn_weights_dtype = attn_weights_float.to(torch::kBFloat16);

    // attn_output = attn_weights @ value (4D bf16 batched).
    auto attn_output_4d = at::matmul(
        attn_weights_dtype, value_states_expanded);
    auto attn_output_3d = attn_output_4d.transpose(1, 2).contiguous()
                              .reshape({B, S, kHiddenSize});

    // ================================================================
    // BACKWARD PASS.
    // ================================================================

    // Output projection backward: F.linear(grad_output, o_weight.t())
    // = grad_output @ o_weight.t().t() = grad_output @ o_weight.
    // The cuBLAS dispatch through at::linear with .t() arg binds
    // OP_N OP_T (pending family secondary surface).
    auto grad_attn_output_2d_flat = at::linear(
        grad_output, o_weight.transpose(0, 1));
    auto grad_output_2d = grad_output.reshape({-1, kHiddenSize});
    grad_o_weight.copy_(at::matmul(grad_output_2d.transpose(0, 1),
                                   attn_output_3d.reshape(
                                       {-1, kHiddenSize})));

    // Reshape grad_attn_output to [B, num_H, S, d].
    auto grad_attn_output_heads = grad_attn_output_2d_flat
                                      .view({B, S,
                                             kNumAttentionHeads,
                                             kHeadDim})
                                      .transpose(1, 2);

    // Attention-application backward (4D bf16 batched -- verified
    // family witness).
    auto grad_attn_weights = at::matmul(
        grad_attn_output_heads, value_states_expanded.transpose(2, 3));
    auto grad_value_states_expanded = at::matmul(
        attn_weights_dtype.transpose(2, 3), grad_attn_output_heads);

    // Softmax backward (fp32).
    auto grad_attn_weights_float = grad_attn_weights.to(torch::kFloat32);
    auto sum_grad = (grad_attn_weights_float * attn_weights_float)
                        .sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attn_scores_f32 = attn_weights_float
                                * (grad_attn_weights_float - sum_grad);
    auto grad_attn_scores_scaled = grad_attn_scores_f32 * scaling_f;

    // Scaled-dot-product backward (4D bf16 batched -- canonical OP_N
    // OP_N + OP_T OP_N verified family bindings; cast to bf16 first
    // matches the reference's `.to(original_dtype)`).
    auto grad_attn_scores_bf16 =
        grad_attn_scores_scaled.to(torch::kBFloat16);
    auto grad_query_states_rope = at::matmul(
        grad_attn_scores_bf16, key_states_expanded);
    auto grad_key_states_expanded =
        at::matmul(query_states_rope.transpose(2, 3),
                   grad_attn_scores_bf16)
            .transpose(2, 3);

    // GQA repeat reverse-sum.
    auto grad_key_states_rope = grad_key_states_expanded
        .view({B, kNumKeyValueHeads, kNumKeyValueGroups, S, kHeadDim})
        .sum(/*dim=*/2);
    auto grad_value_states = grad_value_states_expanded
        .view({B, kNumKeyValueHeads, kNumKeyValueGroups, S, kHeadDim})
        .sum(/*dim=*/2);

    // RoPE backward (GLM-style interleaved, fp32 math).
    //   grad_q_float = grad_query_rope * cos_interleaved
    //   grad_q_rotated = grad_query_rope * sin_interleaved
    //   grad_q_rotated.view(..., head_dim/2, 2):
    //     grad_q1_from_rotated = grad_q_rotated[..., 1]
    //     grad_q2_from_rotated = -grad_q_rotated[..., 0]
    //   grad_q_combined[..., 0::2] = grad_q_float[..., 0::2]
    //                                + grad_q1_from_rotated
    //   grad_q_combined[..., 1::2] = grad_q_float[..., 1::2]
    //                                + grad_q2_from_rotated
    auto grad_query_states_rope_f = grad_query_states_rope
                                        .to(torch::kFloat32);
    auto grad_q_float_direct = grad_query_states_rope_f * cos_interleaved;
    auto grad_q_rotated_f = grad_query_states_rope_f * sin_interleaved;
    auto grad_q_rotated_unflat = grad_q_rotated_f.unflatten(
        -1, {kHalfHeadDim, 2});
    auto grad_q1_from_rotated = grad_q_rotated_unflat.select(-1, 1);
    auto grad_q2_from_rotated = -grad_q_rotated_unflat.select(-1, 0);
    auto grad_q_direct_unflat = grad_q_float_direct.unflatten(
        -1, {kHalfHeadDim, 2});
    auto grad_q_direct_even = grad_q_direct_unflat.select(-1, 0);
    auto grad_q_direct_odd = grad_q_direct_unflat.select(-1, 1);
    auto grad_query_states_f32 = at::stack(
        {grad_q_direct_even + grad_q1_from_rotated,
         grad_q_direct_odd + grad_q2_from_rotated},
        /*dim=*/-1).flatten(-2);
    auto grad_query_states = grad_query_states_f32.to(torch::kBFloat16);

    auto grad_key_states_rope_f = grad_key_states_rope
                                      .to(torch::kFloat32);
    auto grad_k_float_direct = grad_key_states_rope_f * cos_interleaved;
    auto grad_k_rotated_f = grad_key_states_rope_f * sin_interleaved;
    auto grad_k_rotated_unflat = grad_k_rotated_f.unflatten(
        -1, {kHalfHeadDim, 2});
    auto grad_k1_from_rotated = grad_k_rotated_unflat.select(-1, 1);
    auto grad_k2_from_rotated = -grad_k_rotated_unflat.select(-1, 0);
    auto grad_k_direct_unflat = grad_k_float_direct.unflatten(
        -1, {kHalfHeadDim, 2});
    auto grad_k_direct_even = grad_k_direct_unflat.select(-1, 0);
    auto grad_k_direct_odd = grad_k_direct_unflat.select(-1, 1);
    auto grad_key_states_f32 = at::stack(
        {grad_k_direct_even + grad_k1_from_rotated,
         grad_k_direct_odd + grad_k2_from_rotated},
        /*dim=*/-1).flatten(-2);
    auto grad_key_states = grad_key_states_f32.to(torch::kBFloat16);

    // Transpose + reshape: [B, *_H, S, d] -> [B, S, *_proj_size].
    auto grad_query_proj = grad_query_states.transpose(1, 2)
                               .contiguous()
                               .view({B, S, kHiddenSize});
    auto grad_key_proj = grad_key_states.transpose(1, 2)
                             .contiguous()
                             .view({B, S, kKvHiddenSize});
    auto grad_value_proj = grad_value_states.transpose(1, 2)
                               .contiguous()
                               .view({B, S, kKvHiddenSize});

    // QKV projection backwards via F.linear(g, W.t()) (OP_N OP_T
    // pending family secondary surface) + g.t() @ x (OP_T OP_N).
    auto grad_hidden_from_q = at::linear(grad_query_proj,
                                         q_weight.transpose(0, 1));
    auto grad_hidden_from_k = at::linear(grad_key_proj,
                                         k_weight.transpose(0, 1));
    auto grad_hidden_from_v = at::linear(grad_value_proj,
                                         v_weight.transpose(0, 1));
    grad_hidden_states.copy_(
        grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v);

    auto hidden_states_2d = hidden_states.reshape({-1, kHiddenSize});
    grad_q_weight.copy_(
        at::matmul(grad_query_proj.reshape({-1, kHiddenSize})
                       .transpose(0, 1),
                   hidden_states_2d));
    grad_k_weight.copy_(
        at::matmul(grad_key_proj.reshape({-1, kKvHiddenSize})
                       .transpose(0, 1),
                   hidden_states_2d));
    grad_v_weight.copy_(
        at::matmul(grad_value_proj.reshape({-1, kKvHiddenSize})
                       .transpose(0, 1),
                   hidden_states_2d));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Grouped query attention rotary core backward (GLM-style "
        "interleaved RoPE). Pure ATen / cuBLAS dispatch chain: four "
        "4D bf16 cublasGemmStridedBatchedEx dispatches "
        "(attention-application + scaled-dot-product backwards; "
        "the verified cublasGemmStridedBatchedEx_4D_bf16_OP_N_OP_N "
        "family) + four 2D bf16 cublasGemmEx projection forward + "
        "backward dispatch pairs (Q/K/V/O via at::linear OP_N OP_T "
        "pending family secondary surface; grad_weight via "
        "g.t() @ x OP_T OP_N verified family secondary surface) + "
        "forward recomputation (no saved intermediates; cos/sin "
        "derived from inv_freq + position_ids) + softmax backward "
        "(fp32 math) + GLM-style interleaved RoPE backward + GQA "
        "repeat reverse-sum + causal mask. Const axes: "
        "hidden_size=8192, num_attention_heads=64, "
        "num_key_value_heads=8, head_dim=128, num_key_value_groups=8, "
        "kv_hidden_size=1024. Destination-passing style; 5 gradient "
        "outputs.");
}
