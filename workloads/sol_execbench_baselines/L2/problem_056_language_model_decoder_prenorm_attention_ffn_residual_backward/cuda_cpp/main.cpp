#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <string>
#include <vector>

// Backward pass for a complete pre-norm decoder transformer layer: RMSNorm,
// GQA attention with RoPE and a SwiGLU FFN, both wrapped in residual
// connections. The analytic backward written out by hand as explicit tensor
// algebra (no torch.autograd). Reproduced op-for-op from the reference.
//
// The forward saved every intermediate, so this port never recomputes the
// forward -- all of the residual, query/key/value states (pre- and post-RoPE),
// the GQA-repeated states, cos/sin, attn_weights, attn_output, the second
// residual, the SwiGLU pieces (gate / up / silu_up / swiglu_output), both
// RMSNorm variances, both normalized hidden states, and every projection
// weight arrive as inputs. The chain walks back FFN-down -> SwiGLU -> gate/up
// -> post-attention RMSNorm -> attention residual -> output projection ->
// attention value matmul -> softmax -> scaled scores -> GQA repeat -> RoPE ->
// input RMSNorm -> QKV projection.
//
// Precision profile (matched exactly): the layer's tensors are bfloat16, but
// the reference promotes to float32 for the SiLU derivative, the softmax
// Jacobian, and both RMSNorm backward reductions, then casts back to bfloat16
// at the points the reference does. The two layernorm-weight gradients
// (grad_input_ln_weight, grad_post_attn_ln_weight) and both variances and both
// normalized-hidden tensors stay float32, exactly as the reference declares
// them. Every other gradient is bfloat16. No casts beyond those the reference
// writes.
//
// Per definition.json (insertion order):
//   inputs:
//     grad_output, residual, attn_input (bf16, [B, S, hidden_size])
//     query_states         (bf16, [B, num_heads, S, head_dim])
//     key_states           (bf16, [B, num_kv_heads, S, head_dim])
//     value_states         (bf16, [B, num_kv_heads, S, head_dim])
//     query_states_rotated (bf16, [B, num_heads, S, head_dim])
//     key_states_rotated   (bf16, [B, num_kv_heads, S, head_dim])
//     key_states_repeated  (bf16, [B, num_heads, S, head_dim])
//     value_states_repeated(bf16, [B, num_heads, S, head_dim])
//     cos, sin             (bf16, [B, S, head_dim])
//     attn_weights         (bf16, [B, num_heads, S, S])
//     attn_output          (bf16, [B, S, num_heads*head_dim])
//     residual2, ffn_input (bf16, [B, S, hidden_size])
//     gate, up, silu_up, swiglu_output (bf16, [B, S, intermediate_size])
//     input_ln_weight      (bf16, [hidden_size])
//     q_weight             (bf16, [num_heads*head_dim, hidden_size])
//     k_weight, v_weight   (bf16, [num_kv_heads*head_dim, hidden_size])
//     o_weight             (bf16, [hidden_size, num_heads*head_dim])
//     post_attn_ln_weight  (bf16, [hidden_size])
//     gate_weight, up_weight (bf16, [intermediate_size, hidden_size])
//     down_weight          (bf16, [hidden_size, intermediate_size])
//     variance1, variance2 (fp32, [B, S, 1])
//     hidden_states_normalized1, hidden_states_normalized2 (fp32, [B, S, hidden_size])
//     eps                  : scalar float32 -> double
//   outputs (DPS pre-alloc, written in place):
//     grad_input            (bf16, [B, S, hidden_size])
//     grad_input_ln_weight  (fp32, [hidden_size])
//     grad_q_weight         (bf16, [num_heads*head_dim, hidden_size])
//     grad_k_weight         (bf16, [num_kv_heads*head_dim, hidden_size])
//     grad_v_weight         (bf16, [num_kv_heads*head_dim, hidden_size])
//     grad_o_weight         (bf16, [hidden_size, num_heads*head_dim])
//     grad_post_attn_ln_weight (fp32, [hidden_size])
//     grad_gate_weight      (bf16, [intermediate_size, hidden_size])
//     grad_up_weight        (bf16, [intermediate_size, hidden_size])
//     grad_down_weight      (bf16, [hidden_size, intermediate_size])
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// linear / matmul calls route through the same cuBLAS GEMM the reference's
// F.linear / torch.matmul use; the SwiGLU vjp, the softmax Jacobian, the RoPE
// backward (rotate_half with a -sin sign flip), the two RMSNorm backward
// reductions, the GQA-repeat reductions, and every reshape / transpose match
// the reference exactly. The model dimensions (32 heads, 8 KV heads, head_dim
// 160, intermediate_size 14336) are the per-problem constants from
// definition.json.

static void check_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

static torch::Tensor rotate_half(const torch::Tensor& x) {
    const int64_t last = x.size(-1);
    const int64_t half = last / 2;
    auto x1 = x.narrow(/*dim=*/-1, /*start=*/0, /*length=*/half);
    auto x2 = x.narrow(/*dim=*/-1, /*start=*/half, /*length=*/half);
    return at::cat({-x2, x1}, /*dim=*/-1);
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& residual,
    const torch::Tensor& attn_input,
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& query_states_rotated,
    const torch::Tensor& key_states_rotated,
    const torch::Tensor& key_states_repeated,
    const torch::Tensor& value_states_repeated,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attn_weights,
    const torch::Tensor& attn_output,
    const torch::Tensor& residual2,
    const torch::Tensor& ffn_input,
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const torch::Tensor& silu_up,
    const torch::Tensor& swiglu_output,
    const torch::Tensor& input_ln_weight,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& o_weight,
    const torch::Tensor& post_attn_ln_weight,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& variance1,
    const torch::Tensor& variance2,
    const torch::Tensor& hidden_states_normalized1,
    const torch::Tensor& hidden_states_normalized2,
    double eps,
    torch::Tensor grad_input,
    torch::Tensor grad_input_ln_weight,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight,
    torch::Tensor grad_o_weight,
    torch::Tensor grad_post_attn_ln_weight,
    torch::Tensor grad_gate_weight,
    torch::Tensor grad_up_weight,
    torch::Tensor grad_down_weight
) {
    check_cuda(grad_output, "grad_output");
    check_cuda(residual, "residual");
    check_cuda(attn_input, "attn_input");
    check_cuda(o_weight, "o_weight");
    check_cuda(down_weight, "down_weight");
    check_cuda(variance1, "variance1");
    check_cuda(variance2, "variance2");
    check_cuda(hidden_states_normalized1, "hidden_states_normalized1");
    check_cuda(hidden_states_normalized2, "hidden_states_normalized2");

    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must be 3D [B, S, hidden_size], got ",
                grad_output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    // Per-problem constants from definition.json.
    const int64_t num_heads = 32;
    const int64_t num_kv_heads = 8;
    const int64_t head_dim = 160;
    const int64_t intermediate_size = 14336;
    const int64_t num_key_value_groups = num_heads / num_kv_heads;

    const int64_t batch_size = grad_output.size(0);
    const int64_t seq_len = grad_output.size(1);
    const int64_t hidden_size = grad_output.size(2);
    const int64_t N = hidden_size;
    const double inv_sqrt_head_dim = 1.0 / std::sqrt(static_cast<double>(head_dim));

    // ============ Backward through FFN Block ============
    // Both the FFN-output residual and the FFN branch receive grad_output.
    auto grad_residual2 = grad_output;
    auto grad_ffn_output = grad_output;

    // Gradient through down projection.
    auto grad_swiglu_output = at::linear(grad_ffn_output, down_weight.t());
    auto grad_down_weight_result = at::matmul(
        grad_ffn_output.reshape({-1, hidden_size}).t(),
        swiglu_output.reshape({-1, intermediate_size}));

    // Gradient through SwiGLU: swiglu_output = gate * silu(up).
    auto grad_gate = grad_swiglu_output * silu_up;
    auto sigmoid_up = at::sigmoid(up.to(torch::kFloat32));
    auto grad_silu = sigmoid_up * (1.0 + up.to(torch::kFloat32) * (1.0 - sigmoid_up));
    auto grad_up = grad_swiglu_output * gate * grad_silu.to(up.dtype());

    // Gradient through gate and up projections.
    auto grad_ffn_input_gate = at::linear(grad_gate, gate_weight.t());
    auto grad_ffn_input_up = at::linear(grad_up, up_weight.t());
    auto grad_ffn_input = grad_ffn_input_gate + grad_ffn_input_up;

    auto grad_gate_weight_result = at::matmul(
        grad_gate.reshape({-1, intermediate_size}).t(),
        ffn_input.reshape({-1, hidden_size}));
    auto grad_up_weight_result = at::matmul(
        grad_up.reshape({-1, intermediate_size}).t(),
        ffn_input.reshape({-1, hidden_size}));

    // Gradient through the post-attention RMSNorm.
    auto grad_ffn_input_fp32 = grad_ffn_input.to(torch::kFloat32);
    auto grad_post_attn_ln_weight_result =
        (grad_ffn_input_fp32 * hidden_states_normalized2).sum(/*dim=*/std::vector<int64_t>{0, 1});

    auto rsqrt_var2 = at::rsqrt(variance2 + eps);
    auto grad_normalized2 = grad_ffn_input_fp32 * post_attn_ln_weight.to(torch::kFloat32);
    auto grad_hidden_states2 = grad_normalized2 * rsqrt_var2;
    auto grad_var2 = -0.5 * (grad_normalized2 * residual2.to(torch::kFloat32))
                                .sum(/*dim=*/-1, /*keepdim=*/true) * rsqrt_var2.pow(3);
    grad_hidden_states2 = grad_hidden_states2
        + (2.0 / static_cast<double>(N)) * residual2.to(torch::kFloat32) * grad_var2;
    auto grad_hidden_states2_bf16 = grad_hidden_states2.to(residual2.dtype());

    // Accumulate gradient from the FFN residual.
    auto grad_hidden_states_attn = grad_residual2 + grad_hidden_states2_bf16;

    // ============ Backward through Attention Block ============
    // Both the attention residual and the output projection receive this grad.
    auto grad_residual1 = grad_hidden_states_attn;
    auto grad_attn_output_proj = grad_hidden_states_attn;

    // Gradient through output projection.
    auto grad_attn_output = at::linear(grad_attn_output_proj, o_weight.t());
    auto grad_o_weight_result = at::matmul(
        grad_attn_output_proj.reshape({-1, hidden_size}).t(),
        attn_output.reshape({-1, num_heads * head_dim}));

    // Reshape gradient back to multi-head format.
    grad_attn_output = grad_attn_output
        .reshape({batch_size, seq_len, num_heads, head_dim})
        .transpose(1, 2);

    // Gradient through attn_output = matmul(attn_weights, value_states_repeated).
    auto grad_attn_weights = at::matmul(grad_attn_output, value_states_repeated.transpose(2, 3));
    auto grad_value_states_repeated = at::matmul(attn_weights.transpose(2, 3), grad_attn_output);

    // Gradient through softmax (float32 Jacobian, as in the reference).
    auto grad_attn_weights_fp32 = grad_attn_weights.to(torch::kFloat32);
    auto attn_weights_fp32 = attn_weights.to(torch::kFloat32);
    auto grad_attn_logits = attn_weights_fp32
        * (grad_attn_weights_fp32
           - (grad_attn_weights_fp32 * attn_weights_fp32).sum(/*dim=*/-1, /*keepdim=*/true));

    // Gradient through the 1/sqrt(head_dim) scaling, then cast back.
    grad_attn_logits = grad_attn_logits * inv_sqrt_head_dim;
    auto grad_attn_logits_bf16 = grad_attn_logits.to(query_states_rotated.dtype());

    // Gradient through attn_logits = query_rope @ key_repeated^T.
    auto grad_query_states_rotated = at::matmul(grad_attn_logits_bf16, key_states_repeated);
    auto grad_key_states_repeated = at::matmul(grad_attn_logits_bf16.transpose(2, 3), query_states_rotated);

    // Gradient through KV repetition (GQA): sum over the repeated-group dim.
    grad_key_states_repeated = grad_key_states_repeated
        .reshape({batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim});
    auto grad_key_states_rotated = grad_key_states_repeated.sum(/*dim=*/2);

    grad_value_states_repeated = grad_value_states_repeated
        .reshape({batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim});
    auto grad_value_states = grad_value_states_repeated.sum(/*dim=*/2);

    // Gradient through RoPE (rotate_half carries the -sin sign flip backward).
    auto cos_expanded = cos.unsqueeze(1);
    auto sin_expanded = sin.unsqueeze(1);
    auto grad_query_states = (grad_query_states_rotated * cos_expanded)
        + (rotate_half(grad_query_states_rotated) * (-sin_expanded));
    auto grad_key_states = (grad_key_states_rotated * cos_expanded)
        + (rotate_half(grad_key_states_rotated) * (-sin_expanded));

    // Reshape gradients back to [B, S, *].
    grad_query_states = grad_query_states.transpose(1, 2)
        .reshape({batch_size, seq_len, num_heads * head_dim});
    grad_key_states = grad_key_states.transpose(1, 2)
        .reshape({batch_size, seq_len, num_kv_heads * head_dim});
    grad_value_states = grad_value_states.transpose(1, 2)
        .reshape({batch_size, seq_len, num_kv_heads * head_dim});

    // Gradient through QKV projections.
    auto grad_attn_input_q = at::linear(grad_query_states, q_weight.t());
    auto grad_attn_input_k = at::linear(grad_key_states, k_weight.t());
    auto grad_attn_input_v = at::linear(grad_value_states, v_weight.t());
    auto grad_attn_input = grad_attn_input_q + grad_attn_input_k + grad_attn_input_v;

    auto grad_q_weight_result = at::matmul(
        grad_query_states.reshape({-1, num_heads * head_dim}).t(),
        attn_input.reshape({-1, hidden_size}));
    auto grad_k_weight_result = at::matmul(
        grad_key_states.reshape({-1, num_kv_heads * head_dim}).t(),
        attn_input.reshape({-1, hidden_size}));
    auto grad_v_weight_result = at::matmul(
        grad_value_states.reshape({-1, num_kv_heads * head_dim}).t(),
        attn_input.reshape({-1, hidden_size}));

    // Gradient through the input RMSNorm.
    auto grad_attn_input_fp32 = grad_attn_input.to(torch::kFloat32);
    auto grad_input_ln_weight_result =
        (grad_attn_input_fp32 * hidden_states_normalized1).sum(/*dim=*/std::vector<int64_t>{0, 1});

    auto rsqrt_var1 = at::rsqrt(variance1 + eps);
    auto grad_normalized1 = grad_attn_input_fp32 * input_ln_weight.to(torch::kFloat32);
    auto grad_hidden_states1 = grad_normalized1 * rsqrt_var1;
    auto grad_var1 = -0.5 * (grad_normalized1 * residual.to(torch::kFloat32))
                                .sum(/*dim=*/-1, /*keepdim=*/true) * rsqrt_var1.pow(3);
    grad_hidden_states1 = grad_hidden_states1
        + (2.0 / static_cast<double>(N)) * residual.to(torch::kFloat32) * grad_var1;
    auto grad_hidden_states1_bf16 = grad_hidden_states1.to(residual.dtype());

    // Accumulate gradient from the attention residual.
    auto grad_input_result = grad_residual1 + grad_hidden_states1_bf16;

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_input_result.sizes() == grad_input.sizes(),
                "grad_input shape ", grad_input_result.sizes(),
                " does not match pre-allocated ", grad_input.sizes());
    TORCH_CHECK(grad_input_ln_weight_result.sizes() == grad_input_ln_weight.sizes(),
                "grad_input_ln_weight shape ", grad_input_ln_weight_result.sizes(),
                " does not match pre-allocated ", grad_input_ln_weight.sizes());
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
    TORCH_CHECK(grad_post_attn_ln_weight_result.sizes() == grad_post_attn_ln_weight.sizes(),
                "grad_post_attn_ln_weight shape ", grad_post_attn_ln_weight_result.sizes(),
                " does not match pre-allocated ", grad_post_attn_ln_weight.sizes());
    TORCH_CHECK(grad_gate_weight_result.sizes() == grad_gate_weight.sizes(),
                "grad_gate_weight shape ", grad_gate_weight_result.sizes(),
                " does not match pre-allocated ", grad_gate_weight.sizes());
    TORCH_CHECK(grad_up_weight_result.sizes() == grad_up_weight.sizes(),
                "grad_up_weight shape ", grad_up_weight_result.sizes(),
                " does not match pre-allocated ", grad_up_weight.sizes());
    TORCH_CHECK(grad_down_weight_result.sizes() == grad_down_weight.sizes(),
                "grad_down_weight shape ", grad_down_weight_result.sizes(),
                " does not match pre-allocated ", grad_down_weight.sizes());

    grad_input.copy_(grad_input_result);
    grad_input_ln_weight.copy_(grad_input_ln_weight_result);
    grad_q_weight.copy_(grad_q_weight_result);
    grad_k_weight.copy_(grad_k_weight_result);
    grad_v_weight.copy_(grad_v_weight_result);
    grad_o_weight.copy_(grad_o_weight_result);
    grad_post_attn_ln_weight.copy_(grad_post_attn_ln_weight_result);
    grad_gate_weight.copy_(grad_gate_weight_result);
    grad_up_weight.copy_(grad_up_weight_result);
    grad_down_weight.copy_(grad_down_weight_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for a pre-norm decoder layer (RMSNorm + GQA attention "
        "with RoPE + SwiGLU FFN, both residual-wrapped). Pure ATen / cuBLAS "
        "dispatch with no forward recompute (every intermediate is a saved "
        "input): the analytic backward chain (down-projection grad, SwiGLU "
        "vjp, gate/up grads, post-attention RMSNorm backward, output-projection "
        "grad, attention value matmul, float32 softmax Jacobian, scaled-score "
        "grad, GQA-repeat reductions, RoPE backward, QKV projection grads, "
        "input RMSNorm backward) reproduced op-for-op from the reference. "
        "bfloat16 tensors with the reference's float32 promotions for the SiLU "
        "derivative, softmax Jacobian, and both RMSNorm reductions; the two "
        "layernorm-weight grads stay float32. Destination-passing style "
        "(10 outputs).");
}
