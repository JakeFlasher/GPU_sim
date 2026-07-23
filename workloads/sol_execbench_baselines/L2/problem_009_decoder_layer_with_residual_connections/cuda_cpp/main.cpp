#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>
#include <cmath>

// Complete decoder layer: pre-normalization, GQA self-attention with Q/K norm
// and rotary embeddings, a residual add, then a second pre-norm feeding a
// sparse 128-expert MoE MLP with another residual add. Reproduces the
// reference run() op-for-op via pure ATen / cuBLAS dispatch -- no custom device
// kernels.
//
// Pinned axes from the problem definition:
//   hidden_size           = 2048
//   num_attention_heads   = 32
//   num_key_value_heads   = 4
//   head_dim              = 128
//   moe_intermediate_size = 768
//   num_experts           = 128
//   num_experts_per_tok   = 8   (top_k)
//   q_out_features        = num_attention_heads * head_dim   = 4096
//   kv_out_features       = num_key_value_heads * head_dim   = 512
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states                    : [batch, seq, hidden]            float32
//     cos                              : [batch, seq, head_dim]          float32
//     sin                              : [batch, seq, head_dim]          float32
//     attention_mask                   : [batch, 1, seq, seq]            float32
//     input_layernorm_weight           : [hidden]                        float32
//     q_proj_weight                    : [q_out, hidden]                 float32
//     q_proj_bias                      : [q_out]                         float32
//     k_proj_weight                    : [kv_out, hidden]                float32
//     k_proj_bias                      : [kv_out]                        float32
//     v_proj_weight                    : [kv_out, hidden]                float32
//     v_proj_bias                      : [kv_out]                        float32
//     q_norm_weight                    : [head_dim]                      float32
//     k_norm_weight                    : [head_dim]                      float32
//     o_proj_weight                    : [hidden, q_out]                 float32
//     o_proj_bias                      : [hidden]                        float32
//     post_attention_layernorm_weight  : [hidden]                        float32
//     router_weight                    : [num_experts, hidden]           float32
//     expert_gate_weights              : [num_experts, moe_inter, hidden] float32
//     expert_up_weights                : [num_experts, moe_inter, hidden] float32
//     expert_down_weights              : [num_experts, hidden, moe_inter] float32
//     rms_norm_eps                     : scalar                          float32
//   output (float32, DPS pre-alloc, written in place):
//     output : [batch, seq, hidden]
//
// Everything stays in float32 -- the inputs are float32 and the reference's
// rms_norm upcasts to float32 then casts back to the (already float32) input
// dtype, so there is no narrowing anywhere. The variance uses the reference's
// pow(2).mean(-1) / rsqrt(var + eps) form; the softmax over the attention
// logits is computed in float32 (already the working dtype). The MoE tail is
// the reference's exact construction: router linear -> float32 softmax ->
// top-8 selection -> sum-normalization, then a host loop over the fixed expert
// count that gathers each expert's tokens (one_hot / permute / where /
// index_select), runs gate+SiLU / up / SwiGLU / down projections, weights each
// contribution by the per-(token, slot) routing weight, and accumulates via
// index_add_. All work is issued on the current PyTorch stream.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32,
                name, " must be float32, got ", t.dtype());
}

// rms_norm over the last dimension: float32 variance, rsqrt(var + eps),
// scaled by weight. Mirrors the reference helper (a no-op cast since the
// working dtype is already float32).
static torch::Tensor rms_norm(const torch::Tensor& x,
                              const torch::Tensor& weight,
                              double eps) {
    auto variance = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normed = x * at::rsqrt(variance + eps);
    return weight * normed;
}

// rotate_half: split the last dim in two halves and form (-x2, x1).
static torch::Tensor rotate_half(const torch::Tensor& x) {
    const int64_t half = x.size(-1) / 2;
    auto x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
    auto x2 = x.slice(/*dim=*/-1, /*start=*/half, /*end=*/x.size(-1));
    return at::cat({-x2, x1}, /*dim=*/-1);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attention_mask,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& q_proj_bias,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& k_proj_bias,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& v_proj_bias,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& router_weight,
    const torch::Tensor& expert_gate_weights,
    const torch::Tensor& expert_up_weights,
    const torch::Tensor& expert_down_weights,
    double rms_norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(cos, "cos");
    check_fp32_cuda(sin, "sin");
    check_fp32_cuda(attention_mask, "attention_mask");
    check_fp32_cuda(input_layernorm_weight, "input_layernorm_weight");
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(q_proj_bias, "q_proj_bias");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(k_proj_bias, "k_proj_bias");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(v_proj_bias, "v_proj_bias");
    check_fp32_cuda(q_norm_weight, "q_norm_weight");
    check_fp32_cuda(k_norm_weight, "k_norm_weight");
    check_fp32_cuda(o_proj_weight, "o_proj_weight");
    check_fp32_cuda(o_proj_bias, "o_proj_bias");
    check_fp32_cuda(post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_fp32_cuda(router_weight, "router_weight");
    check_fp32_cuda(expert_gate_weights, "expert_gate_weights");
    check_fp32_cuda(expert_up_weights, "expert_up_weights");
    check_fp32_cuda(expert_down_weights, "expert_down_weights");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(attention_mask.dim() == 4, "attention_mask must be 4D, got ", attention_mask.dim(), "D");
    TORCH_CHECK(router_weight.dim() == 2, "router_weight must be 2D, got ", router_weight.dim(), "D");
    TORCH_CHECK(expert_gate_weights.dim() == 3, "expert_gate_weights must be 3D, got ", expert_gate_weights.dim(), "D");
    TORCH_CHECK(expert_up_weights.dim() == 3, "expert_up_weights must be 3D, got ", expert_up_weights.dim(), "D");
    TORCH_CHECK(expert_down_weights.dim() == 3, "expert_down_weights must be 3D, got ", expert_down_weights.dim(), "D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    constexpr int64_t num_attention_heads = 32;
    constexpr int64_t num_key_value_heads = 4;
    constexpr int64_t head_dim = 128;
    constexpr int64_t num_key_value_groups = num_attention_heads / num_key_value_heads;
    const double scaling = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const int64_t num_experts = router_weight.size(0);
    constexpr int64_t top_k = 8;

    // First residual: around attention.
    auto residual = hidden_states;
    auto normed = rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps);

    // Q, K, V projections (linear with bias).
    auto query_states = at::linear(normed, q_proj_weight, q_proj_bias);
    auto key_states = at::linear(normed, k_proj_weight, k_proj_bias);
    auto value_states = at::linear(normed, v_proj_weight, v_proj_bias);

    query_states = query_states.view({batch_size, seq_len, num_attention_heads, head_dim});
    key_states = key_states.view({batch_size, seq_len, num_key_value_heads, head_dim});
    value_states = value_states.view({batch_size, seq_len, num_key_value_heads, head_dim});

    // Q/K normalization over head_dim.
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps);
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps);

    // Move heads ahead of sequence: [batch, heads, seq, head_dim].
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // Rotary position embeddings: cos/sin broadcast over the head axis.
    auto cos_b = cos.unsqueeze(1);
    auto sin_b = sin.unsqueeze(1);
    query_states = (query_states * cos_b) + (rotate_half(query_states) * sin_b);
    key_states = (key_states * cos_b) + (rotate_half(key_states) * sin_b);

    // Repeat KV heads for GQA: expand the group axis then fold it into heads.
    if (num_key_value_groups != 1) {
        key_states = key_states.unsqueeze(2)
                         .expand({batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim})
                         .reshape({batch_size, num_key_value_heads * num_key_value_groups, seq_len, head_dim});
        value_states = value_states.unsqueeze(2)
                           .expand({batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim})
                           .reshape({batch_size, num_key_value_heads * num_key_value_groups, seq_len, head_dim});
    }

    // Scaled dot-product attention with the additive mask; float32 softmax.
    auto attn_weights = at::matmul(query_states, key_states.transpose(2, 3)) * scaling;
    attn_weights = attn_weights + attention_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1);
    auto attn_output = at::matmul(attn_weights, value_states);

    attn_output = attn_output.transpose(1, 2).contiguous()
                      .view({batch_size, seq_len, num_attention_heads * head_dim});
    attn_output = at::linear(attn_output, o_proj_weight, o_proj_bias);

    auto post_attn = residual + attn_output;

    // Second residual: around the MoE MLP.
    auto residual2 = post_attn;
    auto mlp_in = rms_norm(post_attn, post_attention_layernorm_weight, rms_norm_eps);

    // Router: linear logits -> float32 softmax -> top-8 -> sum-normalization.
    auto hidden_states_flat = mlp_in.view({-1, hidden_size});
    auto router_logits = at::linear(hidden_states_flat, router_weight);
    auto routing_weights = at::softmax(router_logits, /*dim=*/1);
    auto topk = at::topk(routing_weights, /*k=*/top_k, /*dim=*/-1,
                         /*largest=*/true, /*sorted=*/true);
    auto routing_weights_topk = std::get<0>(topk);   // [tokens, top_k]
    auto selected_experts = std::get<1>(topk);       // [tokens, top_k] int64
    routing_weights_topk = routing_weights_topk /
        routing_weights_topk.sum(/*dim=*/-1, /*keepdim=*/true);

    // Expert accumulator, zero-initialized.
    auto final_hidden_states = at::zeros_like(hidden_states_flat);

    // expert_mask: [num_experts, top_k, tokens].
    auto expert_mask =
        at::one_hot(selected_experts, num_experts).permute({2, 1, 0});

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        // where on the [top_k, tokens] slice returns {slot indices, token
        // indices} in row-major order -- matching the reference's (idx, top_x)
        // unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto idx = nz[0];     // slot indices
        auto top_x = nz[1];   // token indices

        if (top_x.numel() == 0) {
            continue;
        }

        // Gather this expert's tokens.
        auto current_state = hidden_states_flat.index_select(/*dim=*/0, top_x);

        // Gate / up projections, SiLU SwiGLU gate, down projection.
        auto gate_out = at::linear(current_state, expert_gate_weights[expert_idx]);
        auto up_out = at::linear(current_state, expert_up_weights[expert_idx]);
        auto expert_out = at::silu(gate_out) * up_out;
        expert_out = at::linear(expert_out, expert_down_weights[expert_idx]);

        // Weight by the per-(token, slot) routing weight, then accumulate.
        auto weights = routing_weights_topk.index({top_x, idx}).unsqueeze(-1);
        auto current_hidden_states = expert_out * weights;
        final_hidden_states.index_add_(/*dim=*/0, top_x, current_hidden_states);
    }

    auto moe_out = final_hidden_states.view({batch_size, seq_len, hidden_size});
    auto result = residual2 + moe_out;

    TORCH_CHECK(output.sizes() == result.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", result.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Decoder layer with residual connections. Pure ATen dispatch: float32 "
        "pre-norm RMSNorm -> GQA self-attention (Q/K/V linear, Q/K norm, rotary "
        "embeddings, KV repeat, scaled dot-product with additive mask and "
        "float32 softmax, o_proj) -> first residual -> second pre-norm -> "
        "sparse 128-expert MoE MLP (router linear, float32 softmax, top-8 "
        "selection and normalization, per-expert host loop over the fixed "
        "expert count gathering tokens via one_hot / permute / where / "
        "index_select, gate+SiLU / up / SwiGLU / down projections, routing-"
        "weighted index_add_) -> second residual, reproduced op-for-op from the "
        "reference. Destination-passing style (single output written in place).");
}
