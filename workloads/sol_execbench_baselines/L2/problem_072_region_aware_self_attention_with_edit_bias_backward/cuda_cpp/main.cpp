#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Backward pass for region-aware self-attention with an edit bias. Computes
// gradients through the fused QKV projection, scaled dot-product attention with
// position-based and edit-region biases, softmax, dropout, and the output
// projection. The analytic backward written out by hand as explicit tensor
// algebra (no torch.autograd). Reproduced op-for-op from the reference.
//
// The forward saved every intermediate, so this port never recomputes the
// forward -- query/key/value, the masked attention scores, the softmax probs
// (pre- and post-dropout), the attention output, the dropout mask, the edit /
// attention masks, the bias parameters, and both projection weight/bias pairs
// all arrive as inputs. The chain walks back output projection -> attention
// value matmul -> dropout rescale -> softmax Jacobian -> attention-mask
// zeroing -> the three edit-bias gradients -> scaled Q @ K^T -> QKV projection.
//
// Precision profile (matched exactly): every tensor is float32 and the
// reference stays in float32 throughout, so there are no cast points to track.
//
// Edit-bias gradient detail. The within/cross edit masks reduce to
// [batch, 1, seq, seq] and broadcast against grad_attention_scores
// [batch, heads, seq, seq]; the reference sums over dims [0, 2, 3] and reshapes
// to [heads, 1, 1] for both within_edit_bias and cross_edit_bias. The
// position bias gradient is a narrowed write: only the leading
// [:, :seq_len, :seq_len] block of the [heads, max_pos, max_pos]
// grad_edit_region_bias receives grad_attention_scores.sum(dim=0) (the rest
// stays zero), reproduced here via a zeros_like buffer plus a narrow + copy_.
//
// Per definition.json (insertion order):
//   inputs (all float32):
//     grad_output, hidden_states (   [B, S, hidden_size])
//     edit_region_mask           (   [B, S])
//     qkv_weight                 (   [qkv_size, hidden_size])
//     qkv_bias                   (   [qkv_size])
//     out_weight                 (   [hidden_size, hidden_size])
//     out_bias                   (   [hidden_size])
//     edit_region_bias           (   [heads, max_pos, max_pos])
//     within_edit_bias           (   [heads, 1, 1])
//     cross_edit_bias            (   [heads, 1, 1])
//     attention_mask             (   [B, S])
//     query, key, value          (   [B, heads, S, head_dim])
//     attention_scores           (   [B, heads, S, S])
//     attention_probs            (   [B, heads, S, S])
//     attention_probs_dropped    (   [B, heads, S, S])
//     attention_output           (   [B, S, hidden_size])
//     dropout_mask               (   [B, heads, S, S])
//     scale    : scalar float32 -> double
//     dropout_p: scalar float32 -> double
//   outputs (all float32, DPS pre-alloc, written in place):
//     grad_hidden_states  ( [B, S, hidden_size])
//     grad_qkv_weight     ( [qkv_size, hidden_size])
//     grad_qkv_bias       ( [qkv_size])
//     grad_out_weight     ( [hidden_size, hidden_size])
//     grad_out_bias       ( [hidden_size])
//     grad_edit_region_bias ( [heads, max_pos, max_pos])
//     grad_within_edit_bias ( [heads, 1, 1])
//     grad_cross_edit_bias  ( [heads, 1, 1])
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// matmul calls route through the same cuBLAS GEMM the reference's torch.matmul
// uses; the dropout rescale, the softmax Jacobian, the attention-mask zeroing,
// the three edit-bias reductions, the QKV stack/permute regrouping, and the
// narrowed position-bias write all match the reference exactly. Native float32
// throughout. The model dimensions (16 heads, head_dim 64, hidden_size 1024,
// max_position_embeddings 1024) are the per-problem constants from
// definition.json.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& edit_region_mask,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& out_weight,
    const torch::Tensor& out_bias,
    const torch::Tensor& edit_region_bias,
    const torch::Tensor& within_edit_bias,
    const torch::Tensor& cross_edit_bias,
    const torch::Tensor& attention_mask,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& attention_scores,
    const torch::Tensor& attention_probs,
    const torch::Tensor& attention_probs_dropped,
    const torch::Tensor& attention_output,
    const torch::Tensor& dropout_mask,
    double scale,
    double dropout_p,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_qkv_weight,
    torch::Tensor grad_qkv_bias,
    torch::Tensor grad_out_weight,
    torch::Tensor grad_out_bias,
    torch::Tensor grad_edit_region_bias,
    torch::Tensor grad_within_edit_bias,
    torch::Tensor grad_cross_edit_bias
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(edit_region_mask, "edit_region_mask");
    check_fp32_cuda(qkv_weight, "qkv_weight");
    check_fp32_cuda(out_weight, "out_weight");
    check_fp32_cuda(edit_region_bias, "edit_region_bias");
    check_fp32_cuda(attention_mask, "attention_mask");
    check_fp32_cuda(query, "query");
    check_fp32_cuda(key, "key");
    check_fp32_cuda(value, "value");
    check_fp32_cuda(attention_probs, "attention_probs");
    check_fp32_cuda(attention_probs_dropped, "attention_probs_dropped");
    check_fp32_cuda(attention_output, "attention_output");
    check_fp32_cuda(dropout_mask, "dropout_mask");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_size], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    // Per-problem constants from definition.json.
    const int64_t num_attention_heads = 16;
    const int64_t head_dim = 64;
    const int64_t max_position_embeddings = 1024;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    const int64_t qkv_size = 3 * hidden_size;

    // Gradient through output projection.
    auto grad_attention_output = at::matmul(grad_output, out_weight);
    auto grad_out_weight_result = at::matmul(
        grad_output.reshape({-1, hidden_size}).t(),
        attention_output.reshape({-1, hidden_size}));
    auto grad_out_bias_result = grad_output.sum(/*dim=*/std::vector<int64_t>{0, 1});

    // Reshape gradient back to multi-head format.
    grad_attention_output = grad_attention_output
        .reshape({batch_size, seq_len, num_attention_heads, head_dim})
        .transpose(1, 2);

    // Gradient through attention-value multiplication.
    auto grad_attention_probs_dropped = at::matmul(
        grad_attention_output, value.transpose(-2, -1));
    auto grad_value = at::matmul(
        attention_probs_dropped.transpose(-2, -1), grad_attention_output);

    // Gradient through dropout (rescale by mask / keep-probability).
    torch::Tensor grad_attention_probs;
    if (dropout_p > 0.0) {
        grad_attention_probs = grad_attention_probs_dropped * dropout_mask / (1.0 - dropout_p);
    } else {
        grad_attention_probs = grad_attention_probs_dropped;
    }

    // Gradient through softmax.
    auto sum_grad = (grad_attention_probs * attention_probs).sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attention_scores = attention_probs * (grad_attention_probs - sum_grad);

    // Zero out the gradient at attention-mask-masked positions.
    auto attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2);
    grad_attention_scores = grad_attention_scores.masked_fill(
        attention_mask_expanded == 0, 0.0);

    // Gradient for cross_edit_bias.
    auto edit_mask_q = edit_region_mask.unsqueeze(2);
    auto edit_mask_k = edit_region_mask.unsqueeze(1);
    auto cross_edit = (edit_mask_q * (1 - edit_mask_k) + (1 - edit_mask_q) * edit_mask_k).unsqueeze(1);
    auto grad_cross_edit_bias_result =
        (grad_attention_scores * cross_edit).sum(/*dim=*/std::vector<int64_t>{0, 2, 3}, /*keepdim=*/true);
    grad_cross_edit_bias_result = grad_cross_edit_bias_result.squeeze(0).unsqueeze(-1).unsqueeze(-1);
    grad_cross_edit_bias_result = grad_cross_edit_bias_result.reshape({num_attention_heads, 1, 1});

    // Gradient for within_edit_bias.
    auto within_edit = (edit_mask_q * edit_mask_k).unsqueeze(1);
    auto grad_within_edit_bias_result =
        (grad_attention_scores * within_edit).sum(/*dim=*/std::vector<int64_t>{0, 2, 3}, /*keepdim=*/true);
    grad_within_edit_bias_result = grad_within_edit_bias_result.squeeze(0).unsqueeze(-1).unsqueeze(-1);
    grad_within_edit_bias_result = grad_within_edit_bias_result.reshape({num_attention_heads, 1, 1});

    // Gradient for edit_region_bias: only the leading [:seq_len, :seq_len] block
    // is touched; the rest of the [heads, max_pos, max_pos] tensor stays zero.
    auto grad_edit_region_bias_result = at::zeros_like(edit_region_bias);
    if (seq_len <= max_position_embeddings) {
        auto block = grad_attention_scores.sum(/*dim=*/0);
        grad_edit_region_bias_result
            .narrow(/*dim=*/1, /*start=*/0, /*length=*/seq_len)
            .narrow(/*dim=*/2, /*start=*/0, /*length=*/seq_len)
            .copy_(block);
    }

    // Gradient through the scaled Q @ K^T.
    auto grad_attention_scores_scaled = grad_attention_scores * scale;
    auto grad_query = at::matmul(grad_attention_scores_scaled, key);
    auto grad_key = at::matmul(grad_attention_scores_scaled.transpose(-2, -1), query);

    // Regroup the Q/K/V gradients back into the fused [B, S, 3*hidden] layout.
    auto grad_qkv = at::stack({grad_query, grad_key, grad_value}, /*dim=*/0);
    grad_qkv = grad_qkv.permute({1, 3, 0, 2, 4});
    grad_qkv = grad_qkv.reshape({batch_size, seq_len, qkv_size});

    // Gradient through the fused QKV projection.
    auto grad_hidden_states_result = at::matmul(grad_qkv, qkv_weight);
    auto grad_qkv_weight_result = at::matmul(
        grad_qkv.reshape({-1, qkv_size}).t(),
        hidden_states.reshape({-1, hidden_size}));
    auto grad_qkv_bias_result = grad_qkv.sum(/*dim=*/std::vector<int64_t>{0, 1});

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_hidden_states_result.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states shape ", grad_hidden_states_result.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_qkv_weight_result.sizes() == grad_qkv_weight.sizes(),
                "grad_qkv_weight shape ", grad_qkv_weight_result.sizes(),
                " does not match pre-allocated ", grad_qkv_weight.sizes());
    TORCH_CHECK(grad_qkv_bias_result.sizes() == grad_qkv_bias.sizes(),
                "grad_qkv_bias shape ", grad_qkv_bias_result.sizes(),
                " does not match pre-allocated ", grad_qkv_bias.sizes());
    TORCH_CHECK(grad_out_weight_result.sizes() == grad_out_weight.sizes(),
                "grad_out_weight shape ", grad_out_weight_result.sizes(),
                " does not match pre-allocated ", grad_out_weight.sizes());
    TORCH_CHECK(grad_out_bias_result.sizes() == grad_out_bias.sizes(),
                "grad_out_bias shape ", grad_out_bias_result.sizes(),
                " does not match pre-allocated ", grad_out_bias.sizes());
    TORCH_CHECK(grad_edit_region_bias_result.sizes() == grad_edit_region_bias.sizes(),
                "grad_edit_region_bias shape ", grad_edit_region_bias_result.sizes(),
                " does not match pre-allocated ", grad_edit_region_bias.sizes());
    TORCH_CHECK(grad_within_edit_bias_result.sizes() == grad_within_edit_bias.sizes(),
                "grad_within_edit_bias shape ", grad_within_edit_bias_result.sizes(),
                " does not match pre-allocated ", grad_within_edit_bias.sizes());
    TORCH_CHECK(grad_cross_edit_bias_result.sizes() == grad_cross_edit_bias.sizes(),
                "grad_cross_edit_bias shape ", grad_cross_edit_bias_result.sizes(),
                " does not match pre-allocated ", grad_cross_edit_bias.sizes());

    grad_hidden_states.copy_(grad_hidden_states_result);
    grad_qkv_weight.copy_(grad_qkv_weight_result);
    grad_qkv_bias.copy_(grad_qkv_bias_result);
    grad_out_weight.copy_(grad_out_weight_result);
    grad_out_bias.copy_(grad_out_bias_result);
    grad_edit_region_bias.copy_(grad_edit_region_bias_result);
    grad_within_edit_bias.copy_(grad_within_edit_bias_result);
    grad_cross_edit_bias.copy_(grad_cross_edit_bias_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for region-aware self-attention with an edit bias. Pure "
        "ATen / cuBLAS dispatch with no forward recompute (every intermediate "
        "is a saved input): the analytic backward chain (output-projection "
        "grad, attention value matmul, dropout rescale, softmax Jacobian, "
        "attention-mask zeroing, the within/cross edit-bias reductions summed "
        "over dims [0,2,3], the narrowed position-bias write into the leading "
        "[:seq_len, :seq_len] block, the scaled Q @ K^T grad, the QKV "
        "stack/permute regrouping, and the fused QKV projection grad) "
        "reproduced op-for-op from the reference, native float32 throughout. "
        "Destination-passing style (8 outputs).");
}
