#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>
#include <vector>

// Backward pass for Gemma3 attention with Q/K RMSNorm, RoPE, and soft-capping.
//
// Reference (problem_037_qk_norm_rope_attention_core_backward): the analytic
// backward written out by hand as explicit tensor algebra (no torch.autograd).
// The reference recomputes the whole forward (Q/K/V projections, Q/K RMSNorm,
// RoPE, GQA repeat, scaled scores, tanh soft-capping, causal mask, softmax,
// value matmul, output projection) and then walks the gradients back through
// each stage. This port reproduces that op-for-op.
//
// Precision profile (matched exactly): every tensor is float32, the reference
// stays in float32 throughout, and softmax is taken with dtype=float32 (already
// float32 here). There are no cast points to track.
//
// Forward chain the backward inverts:
//   query/key/value = linear(hidden, {q,k,v}_weight)         (reshaped to heads)
//   q_normed = query * rsqrt(mean(query^2,-1) + eps)
//   query_states_normalized = q_normed * (1 + q_norm_weight) (likewise for K)
//   RoPE: rotate_half(x) = cat(-x2, x1); x_rope = x*cos + rotate_half(x)*sin
//   GQA: key/value repeated from kv heads to attention heads
//   attn_weights = (q_rope @ k_repeated^T) * scaling
//   capped = tanh(attn_weights / softcap) * softcap, then triu causal -inf mask
//   softmax(capped, dtype=float32) @ value_repeated -> attn_output -> o_proj
//
// Per definition.json (insertion order):
//   inputs (all float32):
//     grad_output     : [batch_size, seq_len, hidden_size]
//     hidden_states   : [batch_size, seq_len, hidden_size]
//     cos             : [batch_size, seq_len, head_dim]
//     sin             : [batch_size, seq_len, head_dim]
//     q_weight        : [q_proj_out, hidden_size]
//     k_weight        : [kv_proj_out, hidden_size]
//     v_weight        : [kv_proj_out, hidden_size]
//     o_weight        : [hidden_size, q_proj_out]
//     q_norm_weight   : [head_dim]
//     k_norm_weight   : [head_dim]
//     scaling                : scalar float32 -> double
//     attn_logit_softcapping : scalar float32 -> double
//     rms_norm_eps           : scalar float32 -> double
//   outputs (all float32, DPS pre-alloc, written in place):
//     grad_hidden_states : [batch_size, seq_len, hidden_size]
//     grad_q_weight      : [q_proj_out, hidden_size]
//     grad_k_weight      : [kv_proj_out, hidden_size]
//     grad_v_weight      : [kv_proj_out, hidden_size]
//     grad_o_weight      : [hidden_size, q_proj_out]
//     grad_q_norm_weight : [head_dim]
//     grad_k_norm_weight : [head_dim]
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// linear / matmul calls route through the same cuBLAS GEMM the reference's
// F.linear / torch.matmul use; the softmax Jacobian, the tanh soft-cap
// derivative, the RoPE backward (rotate_half with a sign flip), the RMSNorm
// backward algebra, and every reshape / transpose / GQA-reduction orientation
// match the reference exactly. The model dimensions (24 attention heads, 8 KV
// heads, head_dim 128) are the per-problem constants from definition.json.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& o_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    double scaling,
    double attn_logit_softcapping,
    double rms_norm_eps,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight,
    torch::Tensor grad_o_weight,
    torch::Tensor grad_q_norm_weight,
    torch::Tensor grad_k_norm_weight
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(cos, "cos");
    check_fp32_cuda(sin, "sin");
    check_fp32_cuda(q_weight, "q_weight");
    check_fp32_cuda(k_weight, "k_weight");
    check_fp32_cuda(v_weight, "v_weight");
    check_fp32_cuda(o_weight, "o_weight");
    check_fp32_cuda(q_norm_weight, "q_norm_weight");
    check_fp32_cuda(k_norm_weight, "k_norm_weight");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_size], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    // Per-problem constants from definition.json (Gemma3 attention config).
    const int64_t num_attention_heads = 24;
    const int64_t num_key_value_heads = 8;
    const int64_t head_dim = 128;
    const int64_t num_key_value_groups = num_attention_heads / num_key_value_heads;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    const int64_t q_half_dim = head_dim / 2;

    // ===== Forward recomputation =====
    // Q, K, V projections.
    auto query_states = at::linear(hidden_states, q_weight);
    auto key_states = at::linear(hidden_states, k_weight);
    auto value_states = at::linear(hidden_states, v_weight);

    // Reshape to multi-head format.
    query_states = query_states
        .view({batch_size, seq_len, num_attention_heads, head_dim})
        .transpose(1, 2);
    key_states = key_states
        .view({batch_size, seq_len, num_key_value_heads, head_dim})
        .transpose(1, 2);
    value_states = value_states
        .view({batch_size, seq_len, num_key_value_heads, head_dim})
        .transpose(1, 2);

    // RMSNorm for Q.
    auto q_variance = query_states.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto q_normed = query_states * at::rsqrt(q_variance + rms_norm_eps);
    auto query_states_normalized = q_normed * (1.0 + q_norm_weight);

    // RMSNorm for K.
    auto k_variance = key_states.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto k_normed = key_states * at::rsqrt(k_variance + rms_norm_eps);
    auto key_states_normalized = k_normed * (1.0 + k_norm_weight);

    // Apply RoPE.
    auto cos_expanded = cos.unsqueeze(1);
    auto sin_expanded = sin.unsqueeze(1);

    auto q1 = query_states_normalized.narrow(/*dim=*/-1, /*start=*/0, /*length=*/q_half_dim);
    auto q2 = query_states_normalized.narrow(/*dim=*/-1, /*start=*/q_half_dim, /*length=*/q_half_dim);
    auto q_rotated_half = at::cat({-q2, q1}, /*dim=*/-1);
    auto query_states_rope = (query_states_normalized * cos_expanded) + (q_rotated_half * sin_expanded);

    auto k1 = key_states_normalized.narrow(/*dim=*/-1, /*start=*/0, /*length=*/q_half_dim);
    auto k2 = key_states_normalized.narrow(/*dim=*/-1, /*start=*/q_half_dim, /*length=*/q_half_dim);
    auto k_rotated_half = at::cat({-k2, k1}, /*dim=*/-1);
    auto key_states_rope = (key_states_normalized * cos_expanded) + (k_rotated_half * sin_expanded);

    // Repeat KV for GQA.
    auto key_states_repeated = key_states_rope.unsqueeze(2)
        .expand({batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim})
        .reshape({batch_size, num_attention_heads, seq_len, head_dim});
    auto value_states_repeated = value_states.unsqueeze(2)
        .expand({batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim})
        .reshape({batch_size, num_attention_heads, seq_len, head_dim});

    // Attention scores.
    auto attn_weights = at::matmul(query_states_rope, key_states_repeated.transpose(2, 3)) * scaling;

    // Soft-capping.
    auto attn_weights_for_tanh = attn_weights / attn_logit_softcapping;
    auto attn_weights_tanh = at::tanh(attn_weights_for_tanh);
    auto attn_weights_capped = attn_weights_tanh * attn_logit_softcapping;

    // Apply causal mask.
    auto causal_mask = at::triu(
        at::ones({seq_len, seq_len},
                 hidden_states.options().dtype(torch::kBool)),
        /*diagonal=*/1);
    attn_weights_capped = attn_weights_capped.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0),
        -std::numeric_limits<double>::infinity());

    // Softmax (float32; tensors are already float32).
    auto attn_weights_softmax = at::softmax(attn_weights_capped, /*dim=*/-1);

    // Attention output.
    auto attn_output = at::matmul(attn_weights_softmax, value_states_repeated);

    // Reshape.
    attn_output = attn_output.transpose(1, 2).contiguous();
    auto attn_output_flat = attn_output.view({batch_size, seq_len, num_attention_heads * head_dim});

    // ===== Backward pass =====
    // Gradient of output projection.
    auto grad_attn_output_flat = at::linear(grad_output, o_weight.t());
    auto grad_o_weight_result = at::matmul(
        grad_output.view({-1, hidden_size}).t(),
        attn_output_flat.view({-1, num_attention_heads * head_dim}));

    // Reshape gradient back to multi-head format.
    auto grad_attn_output = grad_attn_output_flat
        .view({batch_size, seq_len, num_attention_heads, head_dim})
        .transpose(1, 2);

    // Gradient through attention matmul.
    auto grad_attn_weights_dropped = at::matmul(grad_attn_output, value_states_repeated.transpose(2, 3));
    auto grad_value_states_repeated = at::matmul(attn_weights_softmax.transpose(2, 3), grad_attn_output);

    // Gradient through softmax.
    auto sum_grad = (grad_attn_weights_dropped * attn_weights_softmax).sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attn_weights = attn_weights_softmax * (grad_attn_weights_dropped - sum_grad);

    // Gradient through soft-capping.
    auto grad_attn_weights_uncapped = grad_attn_weights * (1.0 - attn_weights_tanh.pow(2));

    // Gradient through attention score computation.
    auto grad_attn_weights_scaled = grad_attn_weights_uncapped * scaling;
    auto grad_query_states_rope = at::matmul(grad_attn_weights_scaled, key_states_repeated);
    auto grad_key_states_repeated = at::matmul(grad_attn_weights_scaled.transpose(2, 3), query_states_rope);

    // Gradient through KV repetition (GQA): sum over the repeated-group dim.
    auto grad_key_states_rope = grad_key_states_repeated
        .view({batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim})
        .sum(/*dim=*/2);
    auto grad_value_states = grad_value_states_repeated
        .view({batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim})
        .sum(/*dim=*/2);

    // Gradient through RoPE for Q.
    auto grad_q_from_cos = grad_query_states_rope * cos_expanded;
    auto grad_q_rotated = grad_query_states_rope * sin_expanded;
    auto grad_q_rot_1 = grad_q_rotated.narrow(/*dim=*/-1, /*start=*/0, /*length=*/q_half_dim);
    auto grad_q_rot_2 = grad_q_rotated.narrow(/*dim=*/-1, /*start=*/q_half_dim, /*length=*/q_half_dim);
    auto grad_q_from_sin = at::cat({grad_q_rot_2, -grad_q_rot_1}, /*dim=*/-1);
    auto grad_query_states_normalized = grad_q_from_cos + grad_q_from_sin;

    // Gradient through RoPE for K.
    auto grad_k_from_cos = grad_key_states_rope * cos_expanded;
    auto grad_k_rotated = grad_key_states_rope * sin_expanded;
    auto grad_k_rot_1 = grad_k_rotated.narrow(/*dim=*/-1, /*start=*/0, /*length=*/q_half_dim);
    auto grad_k_rot_2 = grad_k_rotated.narrow(/*dim=*/-1, /*start=*/q_half_dim, /*length=*/q_half_dim);
    auto grad_k_from_sin = at::cat({grad_k_rot_2, -grad_k_rot_1}, /*dim=*/-1);
    auto grad_key_states_normalized = grad_k_from_cos + grad_k_from_sin;

    // Gradient through Q RMSNorm.
    auto grad_q_normed = grad_query_states_normalized * (1.0 + q_norm_weight);
    auto grad_q_norm_weight_result =
        (grad_query_states_normalized * q_normed).sum(/*dim=*/std::vector<int64_t>{0, 1, 2});

    auto rsqrt_q_var = at::rsqrt(q_variance + rms_norm_eps);
    auto grad_q_var = -0.5 * (grad_q_normed * query_states).sum(/*dim=*/-1, /*keepdim=*/true) * rsqrt_q_var.pow(3);
    auto grad_query_states = grad_q_normed * rsqrt_q_var
        + 2.0 * query_states * grad_q_var / static_cast<double>(head_dim);

    // Gradient through K RMSNorm.
    auto grad_k_normed = grad_key_states_normalized * (1.0 + k_norm_weight);
    auto grad_k_norm_weight_result =
        (grad_key_states_normalized * k_normed).sum(/*dim=*/std::vector<int64_t>{0, 1, 2});

    auto rsqrt_k_var = at::rsqrt(k_variance + rms_norm_eps);
    auto grad_k_var = -0.5 * (grad_k_normed * key_states).sum(/*dim=*/-1, /*keepdim=*/true) * rsqrt_k_var.pow(3);
    auto grad_key_states = grad_k_normed * rsqrt_k_var
        + 2.0 * key_states * grad_k_var / static_cast<double>(head_dim);

    // Reshape gradients back to [B, S, *].
    grad_query_states = grad_query_states.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    grad_key_states = grad_key_states.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
    grad_value_states = grad_value_states.transpose(1, 2).contiguous().view({batch_size, seq_len, -1});

    // Gradient through Q/K/V projections.
    auto grad_hidden_states_q = at::linear(grad_query_states, q_weight.t());
    auto grad_q_weight_result = at::matmul(
        grad_query_states.view({-1, num_attention_heads * head_dim}).t(),
        hidden_states.view({-1, hidden_size}));

    auto grad_hidden_states_k = at::linear(grad_key_states, k_weight.t());
    auto grad_k_weight_result = at::matmul(
        grad_key_states.view({-1, num_key_value_heads * head_dim}).t(),
        hidden_states.view({-1, hidden_size}));

    auto grad_hidden_states_v = at::linear(grad_value_states, v_weight.t());
    auto grad_v_weight_result = at::matmul(
        grad_value_states.view({-1, num_key_value_heads * head_dim}).t(),
        hidden_states.view({-1, hidden_size}));

    // Sum gradients for hidden_states.
    auto grad_hidden_states_result = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v;

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_hidden_states_result.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states shape ", grad_hidden_states_result.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_q_weight_result.sizes() == grad_q_weight.sizes(),
                "grad_q_weight shape ", grad_q_weight_result.sizes(),
                " does not match pre-allocated ", grad_q_weight.sizes());
    TORCH_CHECK(grad_k_weight_result.sizes() == grad_k_weight.sizes(),
                "grad_k_weight shape ", grad_k_weight_result.sizes(),
                " does not match pre-allocated ", grad_k_weight.sizes());
    TORCH_CHECK(grad_v_weight_result.sizes() == grad_v_weight.sizes(),
                "grad_v_weight shape ", grad_v_weight_result.sizes(),
                " does not match pre-allocated ", grad_v_weight.sizes());
    TORCH_CHECK(grad_o_weight_result.sizes() == grad_o_weight.sizes(),
                "grad_o_weight shape ", grad_o_weight_result.sizes(),
                " does not match pre-allocated ", grad_o_weight.sizes());
    TORCH_CHECK(grad_q_norm_weight_result.sizes() == grad_q_norm_weight.sizes(),
                "grad_q_norm_weight shape ", grad_q_norm_weight_result.sizes(),
                " does not match pre-allocated ", grad_q_norm_weight.sizes());
    TORCH_CHECK(grad_k_norm_weight_result.sizes() == grad_k_norm_weight.sizes(),
                "grad_k_norm_weight shape ", grad_k_norm_weight_result.sizes(),
                " does not match pre-allocated ", grad_k_norm_weight.sizes());

    grad_hidden_states.copy_(grad_hidden_states_result);
    grad_q_weight.copy_(grad_q_weight_result);
    grad_k_weight.copy_(grad_k_weight_result);
    grad_v_weight.copy_(grad_v_weight_result);
    grad_o_weight.copy_(grad_o_weight_result);
    grad_q_norm_weight.copy_(grad_q_norm_weight_result);
    grad_k_norm_weight.copy_(grad_k_norm_weight_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for Gemma3 Q/K-norm + RoPE + soft-capped attention. Pure "
        "ATen / cuBLAS dispatch: forward recompute (Q/K/V projection, Q/K "
        "RMSNorm, RoPE, GQA repeat, scaled scores, tanh soft-cap, causal mask, "
        "softmax, value matmul, output projection) in float32, then the analytic "
        "backward (output-projection grad, softmax Jacobian, soft-cap "
        "derivative, GQA-repeat reductions, RoPE backward, RMSNorm backward, "
        "Q/K/V weight grads) reproduced op-for-op from the reference, native "
        "float32 throughout. Destination-passing style (7 outputs).");
}
