#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>
#include <cmath>

// SAM-HQ vision-encoder windowed attention block. The image-shaped hidden
// state is layer-normed, padded up to a multiple of the 14x14 window, split
// into non-overlapping windows, run through multi-head attention with 2D
// relative-position bias, unpartitioned back to image shape, un-padded,
// residual-added, and finished with a second-residual GELU MLP.
//
// Reference (problem_077_sam_hq_vision_encoder_window_partition_attention),
// op-for-op:
//   window_size = 14   num_heads = 12   head_dim = 64   mlp_dim = 3072
//   scale = head_dim ** -0.5
//   hidden_states : [B, H, W, C]   channels C == hidden_size == 768
//
//   residual = hidden_states
//   LN1 (manual): mean over C, var(unbiased=false) over C,
//        (x - mean) / sqrt(var + eps) * ln1_weight + ln1_bias
//   pad_h = (window_size - H % window_size) % window_size  (likewise pad_w)
//   hidden_states = pad(hidden_states, (0,0,0,pad_w,0,pad_h))   # [B, Hp, Wp, C]
//   reshape to [B, Hp/ws, ws, Wp/ws, ws, C]; permute (0,1,3,2,4,5); contiguous;
//        reshape to [-1, ws, ws, C]               # windows, batch_windows leading
//   qkv = linear(windows, qkv_weight, qkv_bias)
//   qkv = qkv.reshape(bw, ws*ws, 3, num_heads, head_dim).permute(2,0,3,1,4)
//   q, k, v = qkv[0], qkv[1], qkv[2]              # [bw, nh, ws*ws, hd]
//   attn = (q @ k^T) * scale
//
//   # 2D relative-position bias
//   coords = arange(ws)
//   rel_coords = coords[:,None] - coords[None,:] + ws - 1        # [ws, ws]
//   rel_pos_h_emb = rel_pos_h[rel_coords_h.flatten()].reshape(ws, ws, hd)
//   rel_pos_w_emb = rel_pos_w[rel_coords_w.flatten()].reshape(ws, ws, hd)
//   q5 = q.reshape(bw, nh, ws, ws, hd)            # [b, n, qh, qw, c]
//   rel_h = einsum('bnijc,ikc->bnijk', q5, rel_pos_h_emb)        # [b,n,qh,qw,kh]
//   rel_w = einsum('bnijc,jkc->bnijk', q5, rel_pos_w_emb)        # [b,n,qh,qw,kw]
//   bias  = rel_h[...,:,None] + rel_w[...,None,:]                # [b,n,qh,qw,kh,kw]
//   bias  = bias.reshape(bw, nh, ws*ws, ws*ws)
//   attn  = attn + bias
//   attn  = softmax(attn, dim=-1, dtype=float32)
//   out   = (attn @ v).transpose(1,2).reshape(bw, ws, ws, C)
//   out   = linear(out, proj_weight, proj_bias)
//
//   # window unpartition + un-pad
//   out = out.reshape(B, Hp/ws, Wp/ws, ws, ws, -1).permute(0,1,3,2,4,5)
//   out = out.contiguous().reshape(B, Hp, Wp, -1)[:, :H, :W, :].contiguous()
//   hidden_states = residual + out
//
//   residual = hidden_states
//   LN2 (manual, same form as LN1)
//   mlp = linear(gelu(linear(hidden_states, mlp_lin1)), mlp_lin2)
//   output = residual + mlp
//
// Per definition.json the 15 weight / activation tensors arrive in insertion
// order, then the fp32 scalar layer_norm_eps, then the single output (DPS).
//
// Notes on fidelity:
//   * The reference only pads when pad_h or pad_w is nonzero; this port emits
//     an unconditional constant_pad_nd whose pad amounts are computed from the
//     static window arithmetic, so a zero pad is a faithful no-op and the
//     reshape arithmetic stays valid for both the padded and exact-fit cases.
//   * The window partition is the (0,1,3,2,4,5) permute that interleaves the
//     two window-grid axes ahead of the two within-window axes; the unpartition
//     applies the same permutation to invert it. Both call contiguous() before
//     the collapsing reshape, exactly as the reference does.
//   * The relative-position embedding is gathered with index_select on the
//     flattened (i - j + ws - 1) distance table, reproducing the reference's
//     advanced-index rel_pos_h[rel_coords_h.flatten()].
//   * The two einsums contract the head dim c: rel_h pairs the query row qh
//     with key row kh via rel_pos_h_emb[qh, kh, c]; rel_w pairs the query col
//     qw with key col kw via rel_pos_w_emb[qw, kw, c]. The broadcast add then
//     forms the full [kh, kw] bias before flattening to the key axis.
//   * GELU uses the default (erf) approximation, matching F.gelu's default.
//   * Precision: fp32 throughout, matching the reference. LN runs in the
//     working fp32 dtype; the softmax is requested in fp32 (no cast added).
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

namespace {

constexpr int64_t kWindowSize = 14;
constexpr int64_t kNumHeads = 12;
constexpr int64_t kHeadDim = 64;

// Manual LayerNorm over the last (channel) axis: subtract the mean, divide by
// sqrt(population variance + eps), then affine-scale. unbiased=false mirrors
// the reference's var(..., unbiased=False).
torch::Tensor manual_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
    auto mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
    auto var = x.var(/*dim=*/{-1}, /*unbiased=*/false, /*keepdim=*/true);
    return (x - mean) / at::sqrt(var + eps) * weight + bias;
}

}  // namespace

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias,
    const torch::Tensor& rel_pos_h,
    const torch::Tensor& rel_pos_w,
    const torch::Tensor& layer_norm1_weight,
    const torch::Tensor& layer_norm1_bias,
    const torch::Tensor& layer_norm2_weight,
    const torch::Tensor& layer_norm2_bias,
    const torch::Tensor& mlp_lin1_weight,
    const torch::Tensor& mlp_lin1_bias,
    const torch::Tensor& mlp_lin2_weight,
    const torch::Tensor& mlp_lin2_bias,
    double layer_norm_eps,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(qkv_weight, "qkv_weight");
    check_fp32_cuda(qkv_bias, "qkv_bias");
    check_fp32_cuda(proj_weight, "proj_weight");
    check_fp32_cuda(rel_pos_h, "rel_pos_h");
    check_fp32_cuda(rel_pos_w, "rel_pos_w");
    check_fp32_cuda(mlp_lin1_weight, "mlp_lin1_weight");
    check_fp32_cuda(mlp_lin2_weight, "mlp_lin2_weight");
    check_fp32_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 4,
                "hidden_states must be 4D [B, H, W, C], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t height = hidden_states.size(1);
    const int64_t width = hidden_states.size(2);
    const int64_t channels = hidden_states.size(3);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    // Keep the pre-attention activation for the first residual.
    auto residual = hidden_states;

    // ==================== LayerNorm 1 ====================
    auto normed = manual_layer_norm(
        hidden_states, layer_norm1_weight, layer_norm1_bias, layer_norm_eps);

    // ==================== Window partition (with padding) ====================
    // F.pad over [B, H, W, C] uses (C_l, C_r, W_l, W_r, H_l, H_r); pad the H/W
    // tails up to the next window multiple. A zero pad is a faithful no-op.
    const int64_t pad_h = (kWindowSize - height % kWindowSize) % kWindowSize;
    const int64_t pad_w = (kWindowSize - width % kWindowSize) % kWindowSize;
    auto padded = at::constant_pad_nd(
        normed, /*pad=*/std::vector<int64_t>{0, 0, 0, pad_w, 0, pad_h}, /*value=*/0);

    const int64_t pad_height = height + pad_h;
    const int64_t pad_width = width + pad_w;
    const int64_t num_windows_h = pad_height / kWindowSize;
    const int64_t num_windows_w = pad_width / kWindowSize;

    // Reshape into [B, nWh, ws, nWw, ws, C], interleave the window-grid axes
    // ahead of the within-window axes, then collapse to flat windows.
    auto windows = padded
        .reshape({batch_size, num_windows_h, kWindowSize, num_windows_w, kWindowSize, channels})
        .permute({0, 1, 3, 2, 4, 5})
        .contiguous()
        .reshape({-1, kWindowSize, kWindowSize, channels});

    const int64_t batch_windows = windows.size(0);
    const int64_t seq_len = kWindowSize * kWindowSize;

    // ==================== QKV projection ====================
    auto qkv = at::linear(windows, qkv_weight, qkv_bias);
    qkv = qkv.reshape({batch_windows, seq_len, 3, kNumHeads, kHeadDim})
              .permute({2, 0, 3, 1, 4});  // [3, bw, nh, seq, hd]
    auto qkv_unbound = qkv.unbind(/*dim=*/0);
    auto query = qkv_unbound[0];  // [bw, nh, seq, hd]
    auto key = qkv_unbound[1];
    auto value = qkv_unbound[2];

    // ==================== Attention scores ====================
    auto attn_weights = at::matmul(query, key.transpose(-2, -1)) * scale;  // [bw, nh, seq, seq]

    // ==================== 2D relative-position bias ====================
    // Distance table (i - j + ws - 1) indexes the rel-pos embedding rows.
    auto idx_opts = hidden_states.options().dtype(torch::kLong);
    auto coords = at::arange(kWindowSize, idx_opts);                 // [ws]
    auto rel_coords = coords.view({-1, 1}) - coords.view({1, -1}) + (kWindowSize - 1);  // [ws, ws]
    auto rel_coords_flat = rel_coords.reshape({-1});

    auto rel_pos_h_emb = at::index_select(rel_pos_h, /*dim=*/0, rel_coords_flat)
                             .reshape({kWindowSize, kWindowSize, kHeadDim});  // [qh, kh, c]
    auto rel_pos_w_emb = at::index_select(rel_pos_w, /*dim=*/0, rel_coords_flat)
                             .reshape({kWindowSize, kWindowSize, kHeadDim});  // [qw, kw, c]

    // Query reshaped to expose the within-window row / col axes for the einsums.
    auto query_for_bias =
        query.reshape({batch_windows, kNumHeads, kWindowSize, kWindowSize, kHeadDim});

    // rel_h[b,n,qh,qw,kh] = sum_c q[b,n,qh,qw,c] * rel_pos_h_emb[qh,kh,c]
    auto rel_h = at::einsum("bnijc,ikc->bnijk", {query_for_bias, rel_pos_h_emb});
    // rel_w[b,n,qh,qw,kw] = sum_c q[b,n,qh,qw,c] * rel_pos_w_emb[qw,kw,c]
    auto rel_w = at::einsum("bnijc,jkc->bnijk", {query_for_bias, rel_pos_w_emb});

    // Broadcast-add to the full [kh, kw] grid, then flatten the key axis.
    auto rel_pos_bias = rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2);  // [b,n,qh,qw,kh,kw]
    rel_pos_bias = rel_pos_bias.reshape({batch_windows, kNumHeads, seq_len, seq_len});

    attn_weights = attn_weights + rel_pos_bias;

    // ==================== Softmax + value aggregation ====================
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    auto attn_output = at::matmul(attn_weights, value);  // [bw, nh, seq, hd]
    attn_output = attn_output.transpose(1, 2)
                      .reshape({batch_windows, kWindowSize, kWindowSize, channels});

    // ==================== Output projection ====================
    attn_output = at::linear(attn_output, proj_weight, proj_bias);

    // ==================== Window unpartition + un-pad ====================
    attn_output = attn_output
        .reshape({batch_size, num_windows_h, num_windows_w, kWindowSize, kWindowSize, channels})
        .permute({0, 1, 3, 2, 4, 5})
        .contiguous()
        .reshape({batch_size, pad_height, pad_width, channels});
    attn_output = attn_output.narrow(/*dim=*/1, /*start=*/0, /*length=*/height)
                      .narrow(/*dim=*/2, /*start=*/0, /*length=*/width)
                      .contiguous();

    // First residual connection.
    auto hidden = residual + attn_output;

    // ==================== MLP block with second residual ====================
    auto residual2 = hidden;
    auto normed2 = manual_layer_norm(
        hidden, layer_norm2_weight, layer_norm2_bias, layer_norm_eps);
    auto mlp = at::linear(normed2, mlp_lin1_weight, mlp_lin1_bias);
    mlp = at::gelu(mlp);  // default (erf) approximation
    mlp = at::linear(mlp, mlp_lin2_weight, mlp_lin2_bias);
    auto result = residual2 + mlp;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "SAM-HQ vision-encoder windowed attention via ATen (no custom CUDA "
          "kernels): manual LayerNorm, window partition with tail padding, "
          "12-head attention with 2D relative-position bias (index_select "
          "gather + two head-dim einsums broadcast to the [kh, kw] grid), "
          "fp32 softmax, window unpartition and un-pad, two residual "
          "connections, and an erf-GELU MLP. fp32 throughout; "
          "destination-passing style (output).");
}
