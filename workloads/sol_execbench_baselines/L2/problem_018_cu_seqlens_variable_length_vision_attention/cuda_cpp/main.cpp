#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Variable-length vision attention via cu_seqlens.
//
// Reference (problem_018_cu_seqlens_variable_length_vision_attention),
// reproduced op-for-op:
//   embed_dim = 1152 ; num_heads = 16 ; head_dim = 72 ; qkv_dim = 3*embed_dim
//   scaling   = head_dim ** -0.5
//
//   qkv = F.linear(hidden_states, qkv_weight, qkv_bias)         # [T, 3*embed]
//   qkv = qkv.reshape(T, 3, num_heads, head_dim)
//   query_states, key_states, value_states = qkv.permute(1,0,2,3).unbind(0)
//   # each now [T, num_heads, head_dim]
//
//   def rotate_half(x):  x1,x2 = x[...,:hd/2], x[...,hd/2:]; cat((-x2, x1), -1)
//   q_embed = q*cos + rotate_half(q)*sin                        # bf16 throughout
//   k_embed = k*cos + rotate_half(k)*sin
//   # cos / sin are [T, num_heads, head_dim] -- elementwise, no broadcast.
//
//   cu_seqlens_with_zero = cat([zeros(1), cu_seqlens])
//   lengths = (cu_seqlens_with_zero[1:] - cu_seqlens_with_zero[:-1]).tolist()
//   lengths = [l for l in lengths if l > 0]
//   if no lengths: return zeros(T, embed_dim)
//   q/k/v splits = split(q/k/v, lengths, dim=0)
//   for q,k,v in splits:
//       q = q.transpose(0,1).unsqueeze(0)                       # [1, nh, L, hd]
//       k = k.transpose(0,1).unsqueeze(0)
//       v = v.transpose(0,1).unsqueeze(0)
//       attn = matmul(q, k.transpose(2,3)) * scaling            # [1, nh, L, L]
//       attn = softmax(attn, dim=-1, dtype=float32).to(bf16)
//       out  = matmul(attn, v).squeeze(0).transpose(0,1)        # [L, nh, hd]
//   attn_output = cat(outs, dim=0).reshape(T, embed_dim)
//   output = F.linear(attn_output, proj_weight, proj_bias)      # [T, embed_dim]
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states : [total_seq_len, embed_dim=1152]            bf16
//     cu_seqlens    : [num_sequences]                            int64
//     cos           : [total_seq_len, num_heads=16, head_dim=72] bf16
//     sin           : [total_seq_len, num_heads, head_dim]       bf16
//     qkv_weight    : [qkv_dim=3456, embed_dim]                  bf16
//     qkv_bias      : [qkv_dim]                                  bf16
//     proj_weight   : [embed_dim, embed_dim]                     bf16
//     proj_bias     : [embed_dim]                                bf16
//   output (DPS, pre-alloc):
//     output        : [total_seq_len, embed_dim]                bf16
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// cu_seqlens here is the cumulative form WITHOUT a leading zero, so a zero is
// prepended before differencing to recover per-sequence lengths; zero-length
// sequences are dropped exactly as the reference does. The per-sequence lengths
// are read on the host (each entry via .item<int64_t>(), matching the
// reference's cu_seqlens.cpu().tolist()) and drive at::split over the token
// axis. Per sequence: matmul + fp32 softmax (cast back to bf16) + matmul. RoPE
// runs in bf16 throughout (the reference applies no .float() cast here, and
// cos / sin already match the [T, num_heads, head_dim] head layout, so the
// rotation is elementwise with no broadcasting). The single output is written in
// place (destination-passing).

namespace idx = torch::indexing;

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// rotate_half: cat((-x[..., hd/2:], x[..., :hd/2]), dim=-1).
static torch::Tensor rotate_half(const torch::Tensor& x) {
    const int64_t half = x.size(-1) / 2;
    auto x1 = x.index({"...", idx::Slice(idx::None, half)});
    auto x2 = x.index({"...", idx::Slice(half, idx::None)});
    return at::cat({-x2, x1}, -1);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(qkv_weight, "qkv_weight");
    check_bf16_cuda(qkv_bias, "qkv_bias");
    check_bf16_cuda(proj_weight, "proj_weight");
    check_bf16_cuda(proj_bias, "proj_bias");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64,
                "cu_seqlens must be int64, got ", cu_seqlens.dtype());
    TORCH_CHECK(hidden_states.dim() == 2,
                "hidden_states must be 2D [total_seq_len, embed_dim], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kEmbedDim = 1152;
    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = 72;
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    const int64_t total_seq_len = hidden_states.size(0);
    const auto dtype = hidden_states.scalar_type();

    // --- QKV projection (with bias), reshape to heads, split q / k / v. ---
    auto qkv = at::linear(hidden_states, qkv_weight, qkv_bias);   // [T, 3*embed]
    qkv = qkv.reshape({total_seq_len, 3, kNumHeads, kHeadDim});   // [T, 3, nh, hd]
    auto parts = qkv.permute({1, 0, 2, 3}).unbind(0);             // 3 x [T, nh, hd]
    auto query_states = parts[0];
    auto key_states = parts[1];
    auto value_states = parts[2];

    // --- Rotary embeddings (bf16, elementwise over the matching head layout). ---
    query_states = (query_states * cos) + (rotate_half(query_states) * sin);
    key_states = (key_states * cos) + (rotate_half(key_states) * sin);

    // --- Reconstruct per-sequence lengths from cu_seqlens (no leading zero). ---
    auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU);
    const int64_t num_seqs = cu_seqlens_cpu.size(0);
    std::vector<int64_t> lengths;
    lengths.reserve(num_seqs);
    int64_t prev = 0;
    for (int64_t i = 0; i < num_seqs; ++i) {
        const int64_t cur = cu_seqlens_cpu[i].item<int64_t>();
        const int64_t len = cur - prev;
        prev = cur;
        if (len > 0) {
            lengths.push_back(len);
        }
    }

    if (lengths.empty()) {
        // No valid sequences: the reference returns zeros.
        output.zero_();
        return;
    }

    // --- Split Q, K, V by the per-sequence lengths over the token axis. ---
    auto query_splits = at::split(query_states, lengths, /*dim=*/0);
    auto key_splits = at::split(key_states, lengths, /*dim=*/0);
    auto value_splits = at::split(value_states, lengths, /*dim=*/0);

    // --- Per-sequence attention: matmul + fp32 softmax + matmul. ---
    std::vector<torch::Tensor> attn_outputs;
    attn_outputs.reserve(lengths.size());
    for (size_t s = 0; s < lengths.size(); ++s) {
        auto q = query_splits[s].transpose(0, 1).unsqueeze(0);  // [1, nh, L, hd]
        auto k = key_splits[s].transpose(0, 1).unsqueeze(0);
        auto v = value_splits[s].transpose(0, 1).unsqueeze(0);

        auto attn_weights = at::matmul(q, k.transpose(2, 3)) * scaling;
        attn_weights =
            at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(dtype);
        auto attn_output = at::matmul(attn_weights, v);  // [1, nh, L, hd]
        attn_output = attn_output.squeeze(0).transpose(0, 1);  // [L, nh, hd]
        attn_outputs.push_back(attn_output);
    }

    // --- Concatenate, reshape, and project. ---
    auto attn_output = at::cat(attn_outputs, /*dim=*/0);  // [T, nh, hd]
    attn_output = attn_output.reshape({total_seq_len, kEmbedDim});
    auto result = at::linear(attn_output, proj_weight, proj_bias);  // [T, embed]

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Vision variable-length attention via cu_seqlens. Pure ATen / cuBLAS "
          "dispatch: QKV linear (+bias), head reshape, bf16 RoPE (rotate_half) "
          "from cos/sin, then a host-side loop over the cu_seqlens-derived "
          "per-sequence lengths doing per-slice matmul + fp32 softmax + matmul, "
          "concat, and an output projection. bf16; destination-passing style "
          "(output).");
}
