#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for attention softmax, dropout, and value matmul.
//
// Reference (problem_001_attention_softmax_dropout_value_matmul_backward): the
// analytic backward written out by hand as explicit tensor algebra (NO
// torch.autograd). Reproduced op-for-op below. The grad chain inverts the
// forward attention output path:
//   attn_output = attn_weights_dropped @ value_states_expanded
//   attn_weights_dropped = attn_weights * dropout_mask / (1 - p)
//   attn_weights = softmax(attn_scores)
// where value_states is GQA-expanded (kv heads broadcast into attention heads)
// before the value matmul.
//
// Per definition.json (insertion order):
//   inputs:
//     grad_attn_output     : [batch, seq_q, num_attn_heads, head_dim]   bf16
//     attn_weights         : [batch, num_attn_heads, seq_q, seq_kv]     bf16
//     attn_weights_dropped : [batch, num_attn_heads, seq_q, seq_kv]     bf16
//     value_states         : [batch, num_kv_heads, seq_kv, head_dim]    bf16
//     dropout_mask         : [batch, num_attn_heads, seq_q, seq_kv]     bool
//     attention_dropout    : scalar                                     fp32
//   outputs (DPS pre-alloc, written in place):
//     grad_attn_scores     : [batch, num_attn_heads, seq_q, seq_kv]     bf16
//     grad_value_states    : [batch, num_kv_heads, seq_kv, head_dim]    bf16
//
// PRECISION PROFILE (CLAUDE.md Rule 8): this reference is NOT all-bf16 and is
// NOT all-fp32 -- it upcasts a specific set of intermediates to fp32 and leaves
// the rest in their native dtype, then rounds the two destination buffers back
// to bf16. The exact cast sites are mirrored here, no more and no fewer:
//   * grad_attn_output_transposed = grad_attn_output.transpose(1,2).to(fp32)
//   * value_states_expanded.to(fp32) feeds the grad_attn_weights_dropped matmul
//   * attn_weights_dropped.to(fp32) feeds the grad_value_states_expanded matmul
//   * attn_weights_f32 = attn_weights.to(fp32) for the softmax-backward
// The dropout-mask multiply promotes the bool mask to the fp32 operand's dtype
// (bool * fp32 -> fp32), so grad_attn_weights stays fp32. Upcasting the WHOLE
// chain (or down-casting where the reference upcasts) would change the rounding
// profile and fail correctness, so the cast set is reproduced exactly.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// GQA expand uses unsqueeze + expand + reshape exactly as the reference does;
// its backward (grad_value_states) reshapes the expanded grad back to the
// grouped layout and sums over the group dim. The softmax backward uses the
// stable Jacobian-vector form grad_in = p * (grad_out - sum(grad_out * p)).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_attn_output,
    const torch::Tensor& attn_weights,
    const torch::Tensor& attn_weights_dropped,
    const torch::Tensor& value_states,
    const torch::Tensor& dropout_mask,
    double attention_dropout,
    torch::Tensor grad_attn_scores,
    torch::Tensor grad_value_states
) {
    check_bf16_cuda(grad_attn_output, "grad_attn_output");
    check_bf16_cuda(attn_weights, "attn_weights");
    check_bf16_cuda(attn_weights_dropped, "attn_weights_dropped");
    check_bf16_cuda(value_states, "value_states");
    check_bf16_cuda(grad_attn_scores, "grad_attn_scores");
    check_bf16_cuda(grad_value_states, "grad_value_states");

    TORCH_CHECK(dropout_mask.is_cuda(), "dropout_mask must be a CUDA tensor");
    TORCH_CHECK(dropout_mask.dtype() == torch::kBool,
                "dropout_mask must be bool, got ", dropout_mask.dtype());

    TORCH_CHECK(grad_attn_output.dim() == 4, "grad_attn_output must be 4D, got ", grad_attn_output.dim(), "D");
    TORCH_CHECK(attn_weights.dim() == 4, "attn_weights must be 4D, got ", attn_weights.dim(), "D");
    TORCH_CHECK(attn_weights_dropped.dim() == 4, "attn_weights_dropped must be 4D, got ", attn_weights_dropped.dim(), "D");
    TORCH_CHECK(value_states.dim() == 4, "value_states must be 4D, got ", value_states.dim(), "D");
    TORCH_CHECK(dropout_mask.dim() == 4, "dropout_mask must be 4D, got ", dropout_mask.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_attn_output.device());

    constexpr int64_t kNumAttnHeads = 80;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kNumKVGroups = kNumAttnHeads / kNumKVHeads;  // 10

    const int64_t batch_size = grad_attn_output.size(0);
    const int64_t seq_len_q = grad_attn_output.size(1);
    const int64_t seq_len_kv = value_states.size(2);
    const int64_t head_dim = value_states.size(3);

    // GQA expand of the value states: broadcast each kv head across its group.
    auto value_states_expanded = value_states;
    if (kNumKVGroups > 1) {
        value_states_expanded =
            value_states.unsqueeze(2)
                .expand({batch_size, kNumKVHeads, kNumKVGroups, seq_len_kv, head_dim})
                .reshape({batch_size, kNumAttnHeads, seq_len_kv, head_dim});
    }

    // Transpose grad to head-major layout, upcast to fp32 (reference cast site).
    auto grad_attn_output_transposed =
        grad_attn_output.transpose(1, 2).to(torch::kFloat32);  // [B, heads, seq_q, head_dim]

    // grad w.r.t. attn_weights_dropped = grad_attn_output @ value_states_expanded^T.
    // value_states_expanded upcast to fp32 (reference cast site).
    auto grad_attn_weights_dropped = at::matmul(
        grad_attn_output_transposed,
        value_states_expanded.to(torch::kFloat32).transpose(-2, -1));  // [B, heads, seq_q, seq_kv] fp32

    // grad through dropout: grad_dropped * mask / (1 - p). The bool mask promotes
    // to the fp32 operand's dtype, so grad_attn_weights stays fp32.
    torch::Tensor grad_attn_weights;
    if (attention_dropout > 0.0) {
        grad_attn_weights = grad_attn_weights_dropped * dropout_mask / (1.0 - attention_dropout);
    } else {
        grad_attn_weights = grad_attn_weights_dropped;
    }

    // grad through softmax: p * (grad_out - sum(grad_out * p)). attn_weights
    // upcast to fp32 (reference cast site); result rounded to bf16.
    auto attn_weights_f32 = attn_weights.to(torch::kFloat32);
    auto sum_term = (grad_attn_weights * attn_weights_f32).sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attn_scores_bf16 =
        (attn_weights_f32 * (grad_attn_weights - sum_term)).to(torch::kBFloat16);

    // grad w.r.t. value_states_expanded = attn_weights_dropped^T @ grad_attn_output.
    // attn_weights_dropped upcast to fp32 (reference cast site).
    auto grad_value_states_expanded = at::matmul(
        attn_weights_dropped.to(torch::kFloat32).transpose(-2, -1),
        grad_attn_output_transposed);  // [B, heads, seq_kv, head_dim] fp32

    // GQA backward: reshape the expanded grad to the grouped layout and reduce
    // over the group dim (the head broadcast accumulates here).
    torch::Tensor grad_value_states_bf16;
    if (kNumKVGroups > 1) {
        grad_value_states_expanded = grad_value_states_expanded.reshape(
            {batch_size, kNumKVHeads, kNumKVGroups, seq_len_kv, head_dim});
        grad_value_states_bf16 = grad_value_states_expanded.sum(/*dim=*/2).to(torch::kBFloat16);
    } else {
        grad_value_states_bf16 = grad_value_states_expanded.to(torch::kBFloat16);
    }

    TORCH_CHECK(grad_attn_scores_bf16.sizes() == grad_attn_scores.sizes(),
                "grad_attn_scores result shape ", grad_attn_scores_bf16.sizes(),
                " does not match pre-allocated ", grad_attn_scores.sizes());
    TORCH_CHECK(grad_value_states_bf16.sizes() == grad_value_states.sizes(),
                "grad_value_states result shape ", grad_value_states_bf16.sizes(),
                " does not match pre-allocated ", grad_value_states.sizes());

    grad_attn_scores.copy_(grad_attn_scores_bf16);
    grad_value_states.copy_(grad_value_states_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for attention softmax, dropout, and value matmul. Pure "
        "ATen / cuBLAS dispatch: analytic grad algebra (value matmul backward "
        "with GQA-expanded value states, dropout grad via the bool mask scaled "
        "by 1/(1-p), stable softmax backward p*(grad_out - sum(grad_out*p)), and "
        "GQA backward via reshape-to-groups + sum over the group dim) reproduced "
        "op-for-op from the reference. The reference's exact fp32 cast sites are "
        "mirrored (grad transpose, both matmul operands, softmax-backward "
        "attn_weights); the two outputs cast to bf16. Destination-passing "
        "style (grad_attn_scores, grad_value_states).");
}
