#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Backward pass for SAM-HQ vision attention with 2D decomposed relative
// positional embeddings.
//
// Reference (problem_076_sam_hq_vision_attention_with_relative_position_backward):
// the analytic backward written out by hand as explicit tensor algebra (no
// torch.autograd). The reference recomputes the forward (QKV projection,
// head reshape, scaled query-key scores, decomposed relative-position bias
// built from interpolated embeddings via einsum, softmax, value matmul,
// output projection) and then walks the gradients back through each stage.
// This port reproduces that op-for-op.
//
// Precision profile (matched exactly): every tensor is float32, the reference
// stays in float32 throughout, and softmax is taken in float32 (already
// float32 here). There are no cast points to track.
//
// Relative-position interpolation: the reference resizes each embedding table
// to length 2*max(q,k)-1 with linear interpolation, then gathers a per-query /
// per-key pair table with integer relative coordinates. For these workloads
// q_size == k_size, so the resize target length equals the table length and
// the linear interpolation is the identity (its scale_factor folds to 1.0) --
// but the interpolation, the coordinate gather, and the index_add scatter on
// the backward are all reproduced faithfully exactly as the reference writes
// them, not short-circuited.
//
// Per definition.json (insertion order):
//   inputs (all float32):
//     grad_output   : [batch_size, height, width, hidden_size]
//     hidden_states : [batch_size, height, width, hidden_size]
//     qkv_weight    : [qkv_out_size, hidden_size]
//     qkv_bias      : [qkv_out_size]
//     proj_weight   : [hidden_size, hidden_size]
//     proj_bias     : [hidden_size]
//     rel_pos_h     : [rel_pos_h_size, head_dim]   (rel_pos_h_size = 2*height-1)
//     rel_pos_w     : [rel_pos_w_size, head_dim]   (rel_pos_w_size = 2*width-1)
//     scale         : scalar float32 -> double
//   outputs (all float32, DPS pre-alloc, written in place):
//     grad_hidden_states : [batch_size, height, width, hidden_size]
//     grad_qkv_weight    : [qkv_out_size, hidden_size]
//     grad_qkv_bias      : [qkv_out_size]
//     grad_proj_weight   : [hidden_size, hidden_size]
//     grad_proj_bias     : [hidden_size]
//     grad_rel_pos_h     : [rel_pos_h_size, head_dim]
//     grad_rel_pos_w     : [rel_pos_w_size, head_dim]
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// linear / matmul calls route through the same cuBLAS GEMM the reference's
// F.linear / torch.matmul use; the decomposed-bias einsums, the
// upsample_linear1d interpolation (forward and backward), the index gather /
// index_add scatter, the softmax Jacobian, and every reshape / permute match
// the reference exactly. num_attention_heads is the per-problem constant 12.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// Mirror of get_rel_pos_interpolated: resize the embedding table to length
// 2*max(q,k)-1 by linear interpolation, then gather the per-query / per-key
// relative-coordinate table. Returns [q_size, k_size, head_dim].
static torch::Tensor rel_pos_interpolated(
    int64_t q_size, int64_t k_size, const torch::Tensor& rel_pos) {
    const int64_t max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    const int64_t table_len = rel_pos.size(0);
    const int64_t feat = rel_pos.size(1);

    // reshape(1, n, -1).permute(0, 2, 1) -> [1, feat, table_len]
    auto resized = rel_pos.reshape({1, table_len, -1}).permute({0, 2, 1});
    resized = at::upsample_linear1d(
        resized, /*output_size=*/std::vector<int64_t>{max_rel_dist},
        /*align_corners=*/false, /*scales=*/std::optional<double>(c10::nullopt));
    // reshape(-1, max_rel_dist).permute(1, 0) -> [max_rel_dist, feat]
    resized = resized.reshape({-1, max_rel_dist}).permute({1, 0});

    const double q_ratio = std::max(static_cast<double>(k_size) / q_size, 1.0);
    const double k_ratio = std::max(static_cast<double>(q_size) / k_size, 1.0);
    auto q_coords = at::arange(q_size, rel_pos.options()).unsqueeze(1) * q_ratio;
    auto k_coords = at::arange(k_size, rel_pos.options()).unsqueeze(0) * k_ratio;
    auto relative_coords = (q_coords - k_coords) + (k_size - 1) * k_ratio;
    auto indices = relative_coords.to(torch::kLong);

    // Advanced index with a [q_size, k_size] long table -> [q_size, k_size, feat].
    auto gathered = resized.index_select(0, indices.reshape(-1));
    return gathered.reshape({q_size, k_size, feat});
}

// Mirror of get_rel_pos_interpolated_backward: scatter the per-pair gradient
// back into the resized table by index_add, undo the linear interpolation by
// re-interpolating to the original table length, and rescale by
// table_len / max_rel_dist. Returns [table_len, head_dim].
static torch::Tensor rel_pos_interpolated_backward(
    int64_t q_size, int64_t k_size, const torch::Tensor& rel_pos,
    const torch::Tensor& grad) {
    const int64_t max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    const int64_t table_len = rel_pos.size(0);
    const int64_t feat = grad.size(-1);

    const double q_ratio = std::max(static_cast<double>(k_size) / q_size, 1.0);
    const double k_ratio = std::max(static_cast<double>(q_size) / k_size, 1.0);
    auto q_coords = at::arange(q_size, rel_pos.options()).unsqueeze(1) * q_ratio;
    auto k_coords = at::arange(k_size, rel_pos.options()).unsqueeze(0) * k_ratio;
    auto relative_coords = (q_coords - k_coords) + (k_size - 1) * k_ratio;
    auto indices = relative_coords.to(torch::kLong);

    auto grad_resized = at::zeros({max_rel_dist, feat}, grad.options());
    auto grad_flat = grad.reshape({-1, grad.size(-1)});
    grad_resized = grad_resized.index_add(0, indices.reshape(-1), grad_flat);

    // permute(1, 0).reshape(1, -1, max_rel_dist) -> [1, feat, max_rel_dist]
    grad_resized = grad_resized.permute({1, 0}).reshape({1, -1, max_rel_dist});
    auto grad_input = at::upsample_linear1d(
        grad_resized, /*output_size=*/std::vector<int64_t>{table_len},
        /*align_corners=*/false, /*scales=*/std::optional<double>(c10::nullopt));
    // reshape(-1, table_len).permute(1, 0) -> [table_len, feat]
    auto grad_rel_pos = grad_input.reshape({-1, table_len}).permute({1, 0});

    const double scale_factor =
        static_cast<double>(table_len) / static_cast<double>(max_rel_dist);
    return grad_rel_pos * scale_factor;
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias,
    const torch::Tensor& rel_pos_h,
    const torch::Tensor& rel_pos_w,
    double scale,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_qkv_weight,
    torch::Tensor grad_qkv_bias,
    torch::Tensor grad_proj_weight,
    torch::Tensor grad_proj_bias,
    torch::Tensor grad_rel_pos_h,
    torch::Tensor grad_rel_pos_w
) {
    check_fp32_cuda(grad_output, "grad_output");
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(qkv_weight, "qkv_weight");
    check_fp32_cuda(qkv_bias, "qkv_bias");
    check_fp32_cuda(proj_weight, "proj_weight");
    check_fp32_cuda(proj_bias, "proj_bias");
    check_fp32_cuda(rel_pos_h, "rel_pos_h");
    check_fp32_cuda(rel_pos_w, "rel_pos_w");

    TORCH_CHECK(hidden_states.dim() == 4,
                "hidden_states must be 4D [B, H, W, hidden_size], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    // Per-problem constant from definition.json (SAM-HQ ViT-base config).
    const int64_t num_attention_heads = 12;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t height = hidden_states.size(1);
    const int64_t width = hidden_states.size(2);
    const int64_t hidden_size = hidden_states.size(3);
    const int64_t head_dim = hidden_size / num_attention_heads;
    const int64_t spatial = height * width;
    const int64_t bh = batch_size * num_attention_heads;

    // ===== Forward recomputation =====
    auto hidden_states_flat =
        hidden_states.reshape({batch_size * spatial, hidden_size});
    auto qkv_out = at::linear(hidden_states_flat, qkv_weight, qkv_bias);
    qkv_out = qkv_out
        .reshape({batch_size, spatial, 3, num_attention_heads, head_dim})
        .permute({2, 0, 3, 1, 4});

    auto qkv_grouped = qkv_out.reshape({3, bh, spatial, head_dim});
    auto query = qkv_grouped.select(0, 0);
    auto key = qkv_grouped.select(0, 1);
    auto value = qkv_grouped.select(0, 2);

    auto attn_weights = at::matmul(query * scale, key.transpose(-2, -1));

    auto rel_pos_h_interp = rel_pos_interpolated(height, height, rel_pos_h);
    auto rel_pos_w_interp = rel_pos_interpolated(width, width, rel_pos_w);

    auto query_2d = query.reshape({bh, height, width, head_dim});

    auto rel_h = at::einsum("bhwc,hkc->bhwk", {query_2d, rel_pos_h_interp});
    auto rel_w = at::einsum("bhwc,wkc->bhwk", {query_2d, rel_pos_w_interp});

    auto rel_pos_bias = rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2);
    rel_pos_bias =
        rel_pos_bias.reshape({batch_size, num_attention_heads, spatial, spatial});

    auto attn_weights_reshaped =
        attn_weights.reshape({batch_size, num_attention_heads, spatial, spatial});
    attn_weights_reshaped = attn_weights_reshaped + rel_pos_bias;
    attn_weights = attn_weights_reshaped.reshape({bh, spatial, spatial});

    auto attn_weights_softmax = at::softmax(attn_weights, /*dim=*/-1);
    auto attn_probs = attn_weights_softmax;

    auto attn_output = at::matmul(attn_probs, value);
    attn_output = attn_output
        .reshape({batch_size, num_attention_heads, height, width, head_dim})
        .permute({0, 2, 3, 1, 4})
        .reshape({batch_size, height, width, hidden_size});
    auto attn_output_flat =
        attn_output.reshape({batch_size * spatial, hidden_size});

    // ===== Backward pass =====
    auto grad_output_flat =
        grad_output.reshape({batch_size * spatial, hidden_size});

    auto grad_proj_weight_result = at::matmul(grad_output_flat.t(), attn_output_flat);
    auto grad_proj_bias_result = grad_output_flat.sum(/*dim=*/0);

    auto grad_attn_output_flat = at::matmul(grad_output_flat, proj_weight);
    auto grad_attn_output =
        grad_attn_output_flat.reshape({batch_size, height, width, hidden_size});

    grad_attn_output = grad_attn_output
        .reshape({batch_size, height, width, num_attention_heads, head_dim})
        .permute({0, 3, 1, 2, 4})
        .reshape({bh, spatial, head_dim});

    auto grad_attn_probs = at::matmul(grad_attn_output, value.transpose(-2, -1));
    auto grad_value = at::matmul(attn_probs.transpose(-2, -1), grad_attn_output);

    auto grad_attn_weights_softmax = grad_attn_probs;

    auto sum_grad =
        (grad_attn_weights_softmax * attn_weights_softmax).sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_attn_weights =
        attn_weights_softmax * (grad_attn_weights_softmax - sum_grad);

    auto grad_rel_pos_bias =
        grad_attn_weights.reshape({batch_size, num_attention_heads, spatial, spatial});

    auto grad_query = at::matmul(grad_attn_weights, key) * scale;
    auto grad_key = at::matmul(grad_attn_weights.transpose(-2, -1), query * scale);

    auto grad_rel_pos_bias_5d =
        grad_rel_pos_bias.reshape({bh, height, width, height, width});

    auto grad_rel_h = grad_rel_pos_bias_5d.sum(/*dim=*/-1);
    auto grad_rel_w = grad_rel_pos_bias_5d.sum(/*dim=*/-2);

    auto grad_query_from_rel_h =
        at::einsum("bhwk,hkc->bhwc", {grad_rel_h, rel_pos_h_interp});
    auto grad_rel_pos_h_interp =
        at::einsum("bhwc,bhwk->hkc", {query_2d, grad_rel_h});

    auto grad_query_from_rel_w =
        at::einsum("bhwk,wkc->bhwc", {grad_rel_w, rel_pos_w_interp});
    auto grad_rel_pos_w_interp =
        at::einsum("bhwc,bhwk->wkc", {query_2d, grad_rel_w});

    auto grad_query_2d = grad_query_from_rel_h + grad_query_from_rel_w;
    auto grad_query_from_relpos =
        grad_query_2d.reshape({bh, spatial, head_dim});
    grad_query = grad_query + grad_query_from_relpos;

    auto grad_rel_pos_h_result =
        rel_pos_interpolated_backward(height, height, rel_pos_h, grad_rel_pos_h_interp);
    auto grad_rel_pos_w_result =
        rel_pos_interpolated_backward(width, width, rel_pos_w, grad_rel_pos_w_interp);

    auto grad_qkv = at::stack({grad_query, grad_key, grad_value}, /*dim=*/0);
    grad_qkv = grad_qkv
        .reshape({3, batch_size, num_attention_heads, spatial, head_dim})
        .permute({1, 3, 0, 2, 4});
    auto grad_qkv_flat =
        grad_qkv.reshape({batch_size * spatial, 3 * hidden_size});

    auto grad_hidden_states_flat = at::matmul(grad_qkv_flat, qkv_weight);
    auto grad_hidden_states_result =
        grad_hidden_states_flat.reshape({batch_size, height, width, hidden_size});

    auto grad_qkv_weight_result = at::matmul(grad_qkv_flat.t(), hidden_states_flat);
    auto grad_qkv_bias_result = grad_qkv_flat.sum(/*dim=*/0);

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
    TORCH_CHECK(grad_proj_weight_result.sizes() == grad_proj_weight.sizes(),
                "grad_proj_weight shape ", grad_proj_weight_result.sizes(),
                " does not match pre-allocated ", grad_proj_weight.sizes());
    TORCH_CHECK(grad_proj_bias_result.sizes() == grad_proj_bias.sizes(),
                "grad_proj_bias shape ", grad_proj_bias_result.sizes(),
                " does not match pre-allocated ", grad_proj_bias.sizes());
    TORCH_CHECK(grad_rel_pos_h_result.sizes() == grad_rel_pos_h.sizes(),
                "grad_rel_pos_h shape ", grad_rel_pos_h_result.sizes(),
                " does not match pre-allocated ", grad_rel_pos_h.sizes());
    TORCH_CHECK(grad_rel_pos_w_result.sizes() == grad_rel_pos_w.sizes(),
                "grad_rel_pos_w shape ", grad_rel_pos_w_result.sizes(),
                " does not match pre-allocated ", grad_rel_pos_w.sizes());

    grad_hidden_states.copy_(grad_hidden_states_result);
    grad_qkv_weight.copy_(grad_qkv_weight_result);
    grad_qkv_bias.copy_(grad_qkv_bias_result);
    grad_proj_weight.copy_(grad_proj_weight_result);
    grad_proj_bias.copy_(grad_proj_bias_result);
    grad_rel_pos_h.copy_(grad_rel_pos_h_result);
    grad_rel_pos_w.copy_(grad_rel_pos_w_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for SAM-HQ vision attention with 2D decomposed relative "
        "position. Pure ATen / cuBLAS dispatch: forward recompute (QKV "
        "projection, head reshape, scaled query-key scores, decomposed "
        "relative-position bias from interpolated embeddings via einsum, "
        "softmax, value matmul, output projection) in float32, then the "
        "analytic backward (output-projection grad, softmax Jacobian, "
        "value/key/query matmul grads, decomposed-bias einsum grads with "
        "index_add scatter and upsample_linear1d interpolation backward, QKV "
        "weight grads) reproduced op-for-op from the reference, native float32 "
        "throughout. Destination-passing style (7 outputs).");
}
