#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Vision encoder attention with variable-length sequences via cu_seqlens.
//
// Reference (problem_021_vision_cu_seqlens_variable_length_attention),
// reproduced op-for-op:
//   hidden_size = 1280 ; num_heads = 16 ; head_dim = 80
//   scaling     = head_dim ** -0.5
//
//   qkv = F.linear(hidden_states, qkv_weight, qkv_bias)         # [T, 3*H]
//   qkv = qkv.reshape(T, 3, num_heads, head_dim)
//   qkv = qkv.permute(1, 0, 2, 3)                               # [3, T, nh, hd]
//   query_states, key_states, value_states = qkv.unbind(0)      # each [T, nh, hd]
//
//   cos_expanded = cos.unsqueeze(1).float()                     # [T, 1, hd]
//   sin_expanded = sin.unsqueeze(1).float()                     # [T, 1, hd]
//   q1, q2 = q[..., :hd/2], q[..., hd/2:]
//   q_rotated = cat((-q2, q1), dim=-1)
//   q_embed   = q*cos_expanded + q_rotated*sin_expanded         # then .to(dtype)
//   (the same rotate_half construction is applied to key_states)
//
//   query_states = query_states.transpose(0,1).unsqueeze(0)     # [1, nh, T, hd]
//   key_states   = key_states.transpose(0,1).unsqueeze(0)
//   value_states = value_states.transpose(0,1).unsqueeze(0)
//
//   for i in range(num_seqs - 1):                               # cu_seqlens slices
//       start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
//       if end - start == 0: continue
//       q_seq = query_states[:, :, start:end, :]
//       k_seq = key_states[:, :, start:end, :]
//       v_seq = value_states[:, :, start:end, :]
//       attn = matmul(q_seq, k_seq.transpose(2,3)) * scaling     # [1, nh, L, L]
//       attn = softmax(attn, dim=-1, dtype=float32).to(dtype)
//       out  = matmul(attn, v_seq).transpose(1, 2)               # [1, L, nh, hd]
//       attn_outputs.append(out)
//   attn_output = cat(attn_outputs, dim=1)                       # [1, T, nh, hd]
//   attn_output = attn_output.reshape(T, hidden_size).contiguous()
//   output = F.linear(attn_output, proj_weight, proj_bias)       # [T, hidden_size]
//
// Per definition.json (insertion order):
//   hidden_states : [total_seq_len, hidden_size=1280]   fp32
//   cu_seqlens    : [num_seqs]                           int64
//   cos           : [total_seq_len, head_dim=80]         fp32
//   sin           : [total_seq_len, head_dim=80]         fp32
//   qkv_weight    : [3*hidden_size=3840, hidden_size]    fp32
//   qkv_bias      : [3*hidden_size=3840]                 fp32
//   proj_weight   : [hidden_size, hidden_size]           fp32
//   proj_bias     : [hidden_size]                        fp32
//   output        : [total_seq_len, hidden_size]         fp32  (DPS, pre-alloc)
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// variable-length attention is a host-side loop over the cu_seqlens boundaries:
// each [start:end] slice is a contiguous view, and per-slice attention is
// matmul + fp32 softmax + matmul (the reference materializes the per-slice
// score matrix rather than calling a fused SDPA, so this port does the same).
// Boundary integers are read on the host with .item<int64_t>() exactly as the
// reference's cu_seqlens[i].item() does. All compute stays in fp32; the
// reference's .float() / .to(dtype) casts are no-ops at fp32 but reproduced for
// faithfulness. The single output is written in place (destination-passing).

namespace idx = torch::indexing;

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
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
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(cos, "cos");
    check_fp32_cuda(sin, "sin");
    check_fp32_cuda(qkv_weight, "qkv_weight");
    check_fp32_cuda(qkv_bias, "qkv_bias");
    check_fp32_cuda(proj_weight, "proj_weight");
    check_fp32_cuda(proj_bias, "proj_bias");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64,
                "cu_seqlens must be int64, got ", cu_seqlens.dtype());
    TORCH_CHECK(hidden_states.dim() == 2,
                "hidden_states must be 2D [total_seq_len, hidden], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 1280;
    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = 80;
    constexpr int64_t kHalfDim = kHeadDim / 2;  // 40
    const double scaling = std::pow(static_cast<double>(kHeadDim), -0.5);

    const int64_t total_seq_len = hidden_states.size(0);
    const auto dtype = hidden_states.scalar_type();

    // --- QKV projection (with bias), reshape to heads, split q / k / v. ---
    auto qkv = at::linear(hidden_states, qkv_weight, qkv_bias);  // [T, 3*H]
    qkv = qkv.reshape({total_seq_len, 3, kNumHeads, kHeadDim});  // [T, 3, nh, hd]
    qkv = qkv.permute({1, 0, 2, 3});                             // [3, T, nh, hd]
    auto parts = qkv.unbind(0);
    auto query_states = parts[0];  // [T, nh, hd]
    auto key_states = parts[1];    // [T, nh, hd]
    auto value_states = parts[2];  // [T, nh, hd]

    // --- Rotary embeddings broadcast over the head axis. ---
    auto q_float = query_states.to(torch::kFloat32);
    auto k_float = key_states.to(torch::kFloat32);
    auto cos_expanded = cos.unsqueeze(1).to(torch::kFloat32);  // [T, 1, hd]
    auto sin_expanded = sin.unsqueeze(1).to(torch::kFloat32);  // [T, 1, hd]

    // rotate_half for the query states.
    auto q1 = q_float.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto q2 = q_float.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto q_rotated = at::cat({-q2, q1}, -1);
    auto q_embed = (q_float * cos_expanded) + (q_rotated * sin_expanded);
    query_states = q_embed.to(dtype);

    // rotate_half for the key states.
    auto k1 = k_float.index({"...", idx::Slice(idx::None, kHalfDim)});
    auto k2 = k_float.index({"...", idx::Slice(kHalfDim, idx::None)});
    auto k_rotated = at::cat({-k2, k1}, -1);
    auto k_embed = (k_float * cos_expanded) + (k_rotated * sin_expanded);
    key_states = k_embed.to(dtype);

    // --- Reshape for attention: [T, nh, hd] -> [1, nh, T, hd]. ---
    query_states = query_states.transpose(0, 1).unsqueeze(0);
    key_states = key_states.transpose(0, 1).unsqueeze(0);
    value_states = value_states.transpose(0, 1).unsqueeze(0);

    // --- Per-sequence attention over the cu_seqlens boundaries. ---
    const int64_t num_seqs = cu_seqlens.size(0);
    auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU);
    std::vector<torch::Tensor> attn_outputs;

    for (int64_t i = 0; i + 1 < num_seqs; ++i) {
        const int64_t start = cu_seqlens_cpu[i].item<int64_t>();
        const int64_t end = cu_seqlens_cpu[i + 1].item<int64_t>();
        const int64_t seq_len = end - start;

        if (seq_len == 0) {
            continue;
        }

        auto q_seq = query_states.index(
            {idx::Slice(), idx::Slice(), idx::Slice(start, end), idx::Slice()});
        auto k_seq = key_states.index(
            {idx::Slice(), idx::Slice(), idx::Slice(start, end), idx::Slice()});
        auto v_seq = value_states.index(
            {idx::Slice(), idx::Slice(), idx::Slice(start, end), idx::Slice()});

        // Materialized per-slice scores -> fp32 softmax -> weighted values.
        auto attn_weights = at::matmul(q_seq, k_seq.transpose(2, 3)) * scaling;
        attn_weights =
            at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32).to(dtype);
        auto attn_out = at::matmul(attn_weights, v_seq);  // [1, nh, L, hd]
        attn_out = attn_out.transpose(1, 2);              // [1, L, nh, hd]
        attn_outputs.push_back(attn_out);
    }

    // --- Concatenate the per-sequence outputs and unwind the head layout. ---
    torch::Tensor attn_output;
    if (!attn_outputs.empty()) {
        attn_output = at::cat(attn_outputs, /*dim=*/1);  // [1, T, nh, hd]
    } else {
        attn_output = at::zeros({1, 0, kNumHeads, kHeadDim},
                                hidden_states.options().dtype(dtype));
    }
    attn_output = attn_output.reshape({total_seq_len, kHiddenSize}).contiguous();

    // --- Output projection (with bias). ---
    auto result = at::linear(attn_output, proj_weight, proj_bias);  // [T, hidden]

    // --- Destination-passing-style write. ---
    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Vision variable-length attention via cu_seqlens. Pure ATen / cuBLAS "
          "dispatch: QKV linear (+bias), head reshape, RoPE (rotate_half) from "
          "cos/sin, then a host-side loop over the cu_seqlens boundaries doing "
          "per-slice matmul + fp32 softmax + matmul, concat, and an output "
          "projection. fp32; destination-passing style (output).");
}
