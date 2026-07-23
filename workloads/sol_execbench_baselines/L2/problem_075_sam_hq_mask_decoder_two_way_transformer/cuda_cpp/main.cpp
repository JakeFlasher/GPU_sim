#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>

// Two-way transformer block for the SAM-HQ mask decoder. One block performs
// bidirectional cross-attention between sparse prompt tokens (queries) and
// dense image embeddings (keys) in four stages: self-attention on the queries,
// cross-attention tokens -> image, an MLP on the queries, and cross-attention
// image -> tokens. Self-attention runs at the full hidden width (head_dim 32);
// the two cross-attention stages downsample to the internal dim (head_dim 16).
//
// Reference (problem_075_sam_hq_mask_decoder_two_way_transformer), op-for-op:
//   hidden_size = 256   num_heads = 8   internal_dim = 128   mlp_dim = 2048
//   head_dim_self  = hidden_size  // num_heads = 32
//   head_dim_cross = internal_dim // num_heads = 16
//   eps = 1e-6
//
//   The reference's attention() flattens (batch, point_batch) into a single
//   leading axis, splits channels into (num_heads, head_dim), transposes to
//   put heads ahead of tokens, runs scaled-dot-product attention with an
//   fp32 softmax, then recombines heads back to (batch, point_batch, n, c).
//   layer_norm() is the manual mean / unbiased-population-variance form
//   (x - mean) / sqrt(var + eps) * weight + bias, NOT torch's F.layer_norm.
//
//   Stage 1 -- self-attention on queries:
//     q_input = queries (+ query_point_embedding when not skip_first_layer_pe)
//     q = linear(q_input, self_attn_q); k = linear(q_input, self_attn_k)
//     v = linear(queries, self_attn_v)
//     attn = attention(q, k, v, head_dim_self)
//     queries = layer_norm(queries + linear(attn, self_attn_out), ln1)
//   Stage 2 -- cross-attention tokens -> image:
//     q = linear(queries + query_point_embedding, cross_t2i_q)
//     k = linear(keys + key_point_embedding,      cross_t2i_k)
//     v = linear(keys, cross_t2i_v)
//     attn = attention(q, k, v, head_dim_cross)
//     queries = layer_norm(queries + linear(attn, cross_t2i_out), ln2)
//   Stage 3 -- MLP on queries:
//     mlp = linear(relu(linear(queries, mlp_lin1)), mlp_lin2)
//     queries = layer_norm(queries + mlp, ln3)
//   Stage 4 -- cross-attention image -> tokens (q from image, k from tokens):
//     q = linear(keys + key_point_embedding,      cross_i2t_q)
//     k = linear(queries + query_point_embedding, cross_i2t_k)
//     v = linear(queries, cross_i2t_v)
//     attn = attention(q, k, v, head_dim_cross)
//     keys = layer_norm(keys + linear(attn, cross_i2t_out), ln4)
//   return (queries, keys)
//
// Per definition.json the ~40 weight / activation tensors arrive in insertion
// order, then the bool scalar skip_first_layer_pe, then the two outputs
// (output_queries, output_keys) in destination-passing style.
//
// Notes on fidelity:
//   * skip_first_layer_pe is a host config flag: it only changes whether the
//     query point embedding is added before the stage-1 q/k projections. Both
//     arms are handled by a host-side if; the rest of the block is identical.
//   * The two cross-attention stages run at head_dim_cross = 16 because the
//     q/k/v projections emit internal_dim = 128 channels; the output
//     projection lifts internal_dim back to hidden_size = 256. Self-attention
//     runs at head_dim_self = 32 on the full hidden_size. These per-stage
//     head dims are baked as constants exactly as the reference derives them.
//   * The head split / recombine collapses (batch, point_batch) into one
//     leading axis, so attention is computed per (batch * point_batch, head).
//   * Precision: fp32 throughout, matching the reference. The softmax is
//     requested in fp32 (already the working dtype, so no cast is introduced).
//     layer_norm uses unbiased=false variance (divide by C) to mirror the
//     reference's ((x - mean) ** 2).mean(-1).
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// outputs are written in place into the pre-allocated destinations
// (destination-passing style, two outputs in definition.json order).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

namespace {

constexpr int64_t kNumHeads = 8;
constexpr int64_t kHiddenSize = 256;
constexpr int64_t kInternalDim = 128;
constexpr int64_t kHeadDimSelf = kHiddenSize / kNumHeads;    // 32
constexpr int64_t kHeadDimCross = kInternalDim / kNumHeads;  // 16
constexpr double kEps = 1e-6;

// Manual LayerNorm matching the reference: subtract the per-row mean, divide by
// sqrt(population variance + eps), then affine-scale. Unbiased=false variance
// (mean of squared deviations) mirrors ((x - mean) ** 2).mean(-1).
torch::Tensor manual_layer_norm(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias) {
    auto mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
    auto centered = x - mean;
    auto var = centered.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    return centered / at::sqrt(var + kEps) * weight + bias;
}

// Split the last channel axis into (num_heads, head_dim) and move heads ahead
// of the token axis after collapsing (batch, point_batch) into one leading dim.
// x: [B, P, N, C] -> [B * P, num_heads, N, head_dim].
torch::Tensor separate_heads(const torch::Tensor& x, int64_t head_dim) {
    const int64_t b = x.size(0);
    const int64_t pb = x.size(1);
    const int64_t n = x.size(2);
    return x.reshape({b * pb, n, kNumHeads, head_dim}).transpose(1, 2);
}

// Inverse of separate_heads: [B * P, num_heads, N, head_dim] ->
// [B, P, N, num_heads * head_dim].
torch::Tensor recombine_heads(const torch::Tensor& x, int64_t point_batch_size) {
    const int64_t bpb = x.size(0);
    const int64_t nh = x.size(1);
    const int64_t n = x.size(2);
    const int64_t hd = x.size(3);
    const int64_t b = bpb / point_batch_size;
    return x.transpose(1, 2).reshape({b, point_batch_size, n, nh * hd});
}

// Scaled-dot-product attention with an fp32 softmax, head-split inputs, and a
// head-recombined output, exactly as the reference's attention() helper.
torch::Tensor attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t head_dim,
    int64_t point_batch_size) {
    const double scaling = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto qh = separate_heads(q, head_dim);
    auto kh = separate_heads(k, head_dim);
    auto vh = separate_heads(v, head_dim);
    auto attn_weights = at::matmul(qh, kh.transpose(-2, -1)) * scaling;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    auto attn_output = at::matmul(attn_weights, vh);
    return recombine_heads(attn_output, point_batch_size);
}

}  // namespace

void run(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& query_point_embedding,
    const torch::Tensor& key_point_embedding,
    const torch::Tensor& self_attn_q_weight,
    const torch::Tensor& self_attn_q_bias,
    const torch::Tensor& self_attn_k_weight,
    const torch::Tensor& self_attn_k_bias,
    const torch::Tensor& self_attn_v_weight,
    const torch::Tensor& self_attn_v_bias,
    const torch::Tensor& self_attn_out_weight,
    const torch::Tensor& self_attn_out_bias,
    const torch::Tensor& layer_norm1_weight,
    const torch::Tensor& layer_norm1_bias,
    const torch::Tensor& cross_t2i_q_weight,
    const torch::Tensor& cross_t2i_q_bias,
    const torch::Tensor& cross_t2i_k_weight,
    const torch::Tensor& cross_t2i_k_bias,
    const torch::Tensor& cross_t2i_v_weight,
    const torch::Tensor& cross_t2i_v_bias,
    const torch::Tensor& cross_t2i_out_weight,
    const torch::Tensor& cross_t2i_out_bias,
    const torch::Tensor& layer_norm2_weight,
    const torch::Tensor& layer_norm2_bias,
    const torch::Tensor& mlp_lin1_weight,
    const torch::Tensor& mlp_lin1_bias,
    const torch::Tensor& mlp_lin2_weight,
    const torch::Tensor& mlp_lin2_bias,
    const torch::Tensor& layer_norm3_weight,
    const torch::Tensor& layer_norm3_bias,
    const torch::Tensor& cross_i2t_q_weight,
    const torch::Tensor& cross_i2t_q_bias,
    const torch::Tensor& cross_i2t_k_weight,
    const torch::Tensor& cross_i2t_k_bias,
    const torch::Tensor& cross_i2t_v_weight,
    const torch::Tensor& cross_i2t_v_bias,
    const torch::Tensor& cross_i2t_out_weight,
    const torch::Tensor& cross_i2t_out_bias,
    const torch::Tensor& layer_norm4_weight,
    const torch::Tensor& layer_norm4_bias,
    bool skip_first_layer_pe,
    torch::Tensor output_queries,
    torch::Tensor output_keys
) {
    check_fp32_cuda(queries, "queries");
    check_fp32_cuda(keys, "keys");
    check_fp32_cuda(query_point_embedding, "query_point_embedding");
    check_fp32_cuda(key_point_embedding, "key_point_embedding");
    check_fp32_cuda(self_attn_q_weight, "self_attn_q_weight");
    check_fp32_cuda(cross_t2i_q_weight, "cross_t2i_q_weight");
    check_fp32_cuda(cross_i2t_q_weight, "cross_i2t_q_weight");
    check_fp32_cuda(mlp_lin1_weight, "mlp_lin1_weight");
    check_fp32_cuda(output_queries, "output_queries");
    check_fp32_cuda(output_keys, "output_keys");

    TORCH_CHECK(queries.dim() == 4,
                "queries must be 4D [B, P, n_query_tokens, hidden_size], got ",
                queries.dim(), "D");
    TORCH_CHECK(keys.dim() == 4,
                "keys must be 4D [B, P, n_key_tokens, hidden_size], got ",
                keys.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(queries.device());

    const int64_t point_batch_size = queries.size(1);

    // ==================== Stage 1: self-attention on queries ====================
    // skip_first_layer_pe gates whether the query point embedding is folded into
    // the q / k inputs; v always reads the raw queries.
    auto q_input = skip_first_layer_pe ? queries : (queries + query_point_embedding);

    auto q_proj = at::linear(q_input, self_attn_q_weight, self_attn_q_bias);
    auto k_proj = at::linear(q_input, self_attn_k_weight, self_attn_k_bias);
    auto v_proj = at::linear(queries, self_attn_v_weight, self_attn_v_bias);

    auto attn_out = attention(q_proj, k_proj, v_proj, kHeadDimSelf, point_batch_size);
    attn_out = at::linear(attn_out, self_attn_out_weight, self_attn_out_bias);
    auto queries_state = queries + attn_out;
    queries_state = manual_layer_norm(queries_state, layer_norm1_weight, layer_norm1_bias);

    // ==================== Stage 2: cross-attention tokens -> image ====================
    q_input = queries_state + query_point_embedding;
    auto k_input = keys + key_point_embedding;

    q_proj = at::linear(q_input, cross_t2i_q_weight, cross_t2i_q_bias);
    k_proj = at::linear(k_input, cross_t2i_k_weight, cross_t2i_k_bias);
    v_proj = at::linear(keys, cross_t2i_v_weight, cross_t2i_v_bias);

    attn_out = attention(q_proj, k_proj, v_proj, kHeadDimCross, point_batch_size);
    attn_out = at::linear(attn_out, cross_t2i_out_weight, cross_t2i_out_bias);
    queries_state = queries_state + attn_out;
    queries_state = manual_layer_norm(queries_state, layer_norm2_weight, layer_norm2_bias);

    // ==================== Stage 3: MLP on queries ====================
    auto mlp_out = at::linear(queries_state, mlp_lin1_weight, mlp_lin1_bias);
    mlp_out = at::relu(mlp_out);
    mlp_out = at::linear(mlp_out, mlp_lin2_weight, mlp_lin2_bias);
    queries_state = queries_state + mlp_out;
    queries_state = manual_layer_norm(queries_state, layer_norm3_weight, layer_norm3_bias);

    // ==================== Stage 4: cross-attention image -> tokens ====================
    // Query is built from the image side, key from the token side.
    q_input = queries_state + query_point_embedding;
    k_input = keys + key_point_embedding;

    q_proj = at::linear(k_input, cross_i2t_q_weight, cross_i2t_q_bias);
    k_proj = at::linear(q_input, cross_i2t_k_weight, cross_i2t_k_bias);
    v_proj = at::linear(queries_state, cross_i2t_v_weight, cross_i2t_v_bias);

    attn_out = attention(q_proj, k_proj, v_proj, kHeadDimCross, point_batch_size);
    attn_out = at::linear(attn_out, cross_i2t_out_weight, cross_i2t_out_bias);
    auto keys_state = keys + attn_out;
    keys_state = manual_layer_norm(keys_state, layer_norm4_weight, layer_norm4_bias);

    // --- Destination-passing-style writes (definition.json output order). ---
    TORCH_CHECK(queries_state.sizes() == output_queries.sizes(),
                "output_queries shape ", queries_state.sizes(),
                " does not match pre-allocated ", output_queries.sizes());
    TORCH_CHECK(keys_state.sizes() == output_keys.sizes(),
                "output_keys shape ", keys_state.sizes(),
                " does not match pre-allocated ", output_keys.sizes());

    output_queries.copy_(queries_state);
    output_keys.copy_(keys_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "SAM-HQ mask decoder two-way transformer block via ATen (no custom "
          "CUDA kernels): four-stage bidirectional cross-attention "
          "(self-attention on tokens at head_dim 32, cross-attention "
          "tokens<->image at head_dim 16, an MLP, and an image->token "
          "cross-attention) with point-embedding adds, manual LayerNorm "
          "(eps 1e-6, population variance), and fp32 softmax. "
          "skip_first_layer_pe is a host-side config branch. fp32 throughout; "
          "destination-passing style (output_queries, output_keys).");
}
