#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Backward pass for variable-length audio encoder attention with chunking.
//
// Reference (problem_014_audio_encoder_varlen_attention_with_chunking_backward),
// reproduced op-for-op:
//   d_model = 1280 ; num_heads = 20 ; head_dim = 64 ; qkv_dim = 1280
//   scaling = head_dim ** -0.5
//
// Everything is upcast to fp32 first, all matmuls / softmax / reductions run in
// fp32, and the nine gradients are cast back to bf16 at the very end.
//
//   chunk_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
//
//   grad_attn_output = grad_output @ out_weight               # [T, qkv_dim]
//   query/key/value chunks = q/k/v.split(chunk_lengths, dim=2) # over total_seq
//   for q,k,v in chunks:                                       # recompute fwd
//       aw = softmax((q @ k^T) * scaling, dim=-1, fp32)
//       out = aw @ v
//   attn_output_recompute = cat(outs, dim=2)
//       .squeeze(0).transpose(0,1).reshape(T, -1).contiguous()
//   grad_out_weight = grad_output^T @ attn_output_recompute    # [d_model, qkv]
//   grad_out_bias   = grad_output.sum(dim=0)                   # [d_model]
//
//   grad_attn_output = grad_attn_output.reshape(T, nh, hd)
//       .transpose(0,1).unsqueeze(0)                           # [1, nh, T, hd]
//   grad_attn chunks = grad_attn_output.split(chunk_lengths, dim=2)
//   for grad_attn,q,k,v in chunks:                             # attention vjp
//       aw = softmax((q @ k^T) * scaling, dim=-1, fp32)
//       grad_v = aw^T @ grad_attn
//       grad_aw = grad_attn @ v^T
//       grad_aw = aw * (grad_aw - (aw * grad_aw).sum(dim=-1, keepdim=True))
//       grad_q = (grad_aw @ k) * scaling
//       grad_k = (grad_aw^T @ q) * scaling
//   grad_query/key/value = cat(..., dim=2)
//       .squeeze(0).transpose(0,1).reshape(T, -1).contiguous() # [T, qkv_dim]
//
//   grad_q_weight = grad_query^T @ hidden_states               # [qkv, d_model]
//   grad_q_bias   = grad_query.sum(dim=0)                      # [qkv]
//   (same for k, v)
//   grad_hidden_states = grad_query @ q_weight
//                      + grad_key   @ k_weight
//                      + grad_value @ v_weight                 # [T, d_model]
//
// Per definition.json (insertion order):
//   inputs:
//     grad_output   : [total_seq_len, d_model=1280]            bf16
//     hidden_states : [total_seq_len, d_model=1280]            bf16
//     query_states  : [1, num_heads=20, total_seq_len, hd=64]  bf16
//     key_states    : [1, num_heads, total_seq_len, head_dim]  bf16
//     value_states  : [1, num_heads, total_seq_len, head_dim]  bf16
//     cu_seqlens    : [num_chunks + 1]                         int32
//     q_weight      : [qkv_dim=1280, d_model]                  bf16
//     k_weight      : [qkv_dim, d_model]                       bf16
//     v_weight      : [qkv_dim, d_model]                       bf16
//     out_weight    : [d_model, qkv_dim]                       bf16
//   outputs (DPS, pre-alloc, written in this order):
//     grad_hidden_states : [total_seq_len, d_model]            bf16
//     grad_q_weight      : [qkv_dim, d_model]                  bf16
//     grad_q_bias        : [qkv_dim]                           bf16
//     grad_k_weight      : [qkv_dim, d_model]                  bf16
//     grad_k_bias        : [qkv_dim]                           bf16
//     grad_v_weight      : [qkv_dim, d_model]                  bf16
//     grad_v_bias        : [qkv_dim]                           bf16
//     grad_out_weight    : [d_model, qkv_dim]                  bf16
//     grad_out_bias      : [d_model]                           bf16
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// chunk structure is reconstructed on the host from cu_seqlens (each entry read
// with .item<int64_t>(), exactly as the reference's cu_seqlens.cpu().tolist()
// does), and the per-chunk attention plus its analytic backward are expressed as
// matmul / softmax / reductions. The reference recomputes the forward attention
// twice -- once to rebuild attn_output for grad_out_weight, once inside the q/k/v
// gradient loop -- so this port keeps the two loops separate to reproduce the
// numerics exactly. All compute runs in fp32; each of the nine grad outputs is
// cast to bf16 just before its destination-passing write.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

static void write_dps(const torch::Tensor& src, torch::Tensor dst, const std::string& name) {
    TORCH_CHECK(src.sizes() == dst.sizes(),
                name, " shape ", src.sizes(),
                " does not match pre-allocated ", dst.sizes());
    dst.copy_(src.to(torch::kBFloat16));
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& out_weight,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_q_bias,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_k_bias,
    torch::Tensor grad_v_weight,
    torch::Tensor grad_v_bias,
    torch::Tensor grad_out_weight,
    torch::Tensor grad_out_bias
) {
    check_bf16_cuda(grad_output, "grad_output");
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(query_states, "query_states");
    check_bf16_cuda(key_states, "key_states");
    check_bf16_cuda(value_states, "value_states");
    check_bf16_cuda(q_weight, "q_weight");
    check_bf16_cuda(k_weight, "k_weight");
    check_bf16_cuda(v_weight, "v_weight");
    check_bf16_cuda(out_weight, "out_weight");
    TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be a CUDA tensor");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32,
                "cu_seqlens must be int32, got ", cu_seqlens.dtype());
    TORCH_CHECK(hidden_states.dim() == 2,
                "hidden_states must be 2D [total_seq_len, d_model], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(query_states.dim() == 4,
                "query_states must be 4D [1, num_heads, total_seq_len, head_dim], got ",
                query_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t total_seq_len = hidden_states.size(0);
    const int64_t d_model = hidden_states.size(1);
    const int64_t num_heads = query_states.size(1);
    const int64_t head_dim = query_states.size(3);
    const double scaling = std::pow(static_cast<double>(head_dim), -0.5);

    // --- Upcast every input to fp32 (reference does this for stability). ---
    auto grad_output_f32 = grad_output.to(torch::kFloat32);
    auto hidden_states_f32 = hidden_states.to(torch::kFloat32);
    auto query_states_f32 = query_states.to(torch::kFloat32);
    auto key_states_f32 = key_states.to(torch::kFloat32);
    auto value_states_f32 = value_states.to(torch::kFloat32);
    auto q_weight_f32 = q_weight.to(torch::kFloat32);
    auto k_weight_f32 = k_weight.to(torch::kFloat32);
    auto v_weight_f32 = v_weight.to(torch::kFloat32);
    auto out_weight_f32 = out_weight.to(torch::kFloat32);

    // --- Reconstruct the chunk lengths from cu_seqlens on the host. ---
    auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU).to(torch::kInt64);
    const int64_t num_boundaries = cu_seqlens_cpu.size(0);
    std::vector<int64_t> chunk_lengths;
    chunk_lengths.reserve(num_boundaries > 0 ? num_boundaries - 1 : 0);
    for (int64_t i = 0; i + 1 < num_boundaries; ++i) {
        const int64_t start = cu_seqlens_cpu[i].item<int64_t>();
        const int64_t end = cu_seqlens_cpu[i + 1].item<int64_t>();
        chunk_lengths.push_back(end - start);
    }

    // --- Gradient w.r.t. the output projection input. ---
    auto grad_attn_output = at::matmul(grad_output_f32, out_weight_f32);  // [T, qkv]

    // --- Recompute the per-chunk forward attention to rebuild attn_output. ---
    auto query_chunks = at::split(query_states_f32, chunk_lengths, /*dim=*/2);
    auto key_chunks = at::split(key_states_f32, chunk_lengths, /*dim=*/2);
    auto value_chunks = at::split(value_states_f32, chunk_lengths, /*dim=*/2);

    std::vector<torch::Tensor> attn_outputs_recompute;
    attn_outputs_recompute.reserve(query_chunks.size());
    for (size_t c = 0; c < query_chunks.size(); ++c) {
        auto q = query_chunks[c];
        auto k = key_chunks[c];
        auto v = value_chunks[c];
        auto attn_weights = at::matmul(q, k.transpose(-2, -1)) * scaling;
        attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32);
        auto out = at::matmul(attn_weights, v);
        attn_outputs_recompute.push_back(out);
    }

    auto attn_output_recompute = at::cat(attn_outputs_recompute, /*dim=*/2);
    attn_output_recompute = attn_output_recompute.squeeze(0)
                                .transpose(0, 1)
                                .reshape({total_seq_len, -1})
                                .contiguous();

    // --- Output-projection weight / bias gradients. ---
    auto grad_out_weight_f32 = at::matmul(grad_output_f32.t(), attn_output_recompute);
    auto grad_out_bias_f32 = grad_output_f32.sum(/*dim=*/0);

    // --- Reshape grad_attn_output for the attention backward. ---
    grad_attn_output = grad_attn_output.reshape({total_seq_len, num_heads, head_dim});
    grad_attn_output = grad_attn_output.transpose(0, 1).unsqueeze(0);  // [1, nh, T, hd]
    auto grad_attn_chunks = at::split(grad_attn_output, chunk_lengths, /*dim=*/2);

    // --- Per-chunk gradients for Q, K, V (softmax vjp). ---
    std::vector<torch::Tensor> grad_query_chunks;
    std::vector<torch::Tensor> grad_key_chunks;
    std::vector<torch::Tensor> grad_value_chunks;
    grad_query_chunks.reserve(query_chunks.size());
    grad_key_chunks.reserve(query_chunks.size());
    grad_value_chunks.reserve(query_chunks.size());

    for (size_t c = 0; c < query_chunks.size(); ++c) {
        auto grad_attn = grad_attn_chunks[c];
        auto q = query_chunks[c];
        auto k = key_chunks[c];
        auto v = value_chunks[c];

        auto attn_weights = at::matmul(q, k.transpose(-2, -1)) * scaling;
        attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/torch::kFloat32);

        auto grad_v = at::matmul(attn_weights.transpose(-2, -1), grad_attn);
        auto grad_attn_weights = at::matmul(grad_attn, v.transpose(-2, -1));

        // Softmax backward: aw * (g - sum(aw * g)) along the last axis.
        auto weighted = (attn_weights * grad_attn_weights).sum(/*dim=*/-1, /*keepdim=*/true);
        grad_attn_weights = attn_weights * (grad_attn_weights - weighted);

        auto grad_q = at::matmul(grad_attn_weights, k) * scaling;
        auto grad_k = at::matmul(grad_attn_weights.transpose(-2, -1), q) * scaling;

        grad_query_chunks.push_back(grad_q);
        grad_key_chunks.push_back(grad_k);
        grad_value_chunks.push_back(grad_v);
    }

    auto grad_query = at::cat(grad_query_chunks, /*dim=*/2);
    auto grad_key = at::cat(grad_key_chunks, /*dim=*/2);
    auto grad_value = at::cat(grad_value_chunks, /*dim=*/2);

    grad_query = grad_query.squeeze(0).transpose(0, 1).reshape({total_seq_len, -1}).contiguous();
    grad_key = grad_key.squeeze(0).transpose(0, 1).reshape({total_seq_len, -1}).contiguous();
    grad_value = grad_value.squeeze(0).transpose(0, 1).reshape({total_seq_len, -1}).contiguous();

    // --- QKV-projection weight / bias gradients. ---
    auto grad_q_weight_f32 = at::matmul(grad_query.t(), hidden_states_f32);
    auto grad_q_bias_f32 = grad_query.sum(/*dim=*/0);
    auto grad_k_weight_f32 = at::matmul(grad_key.t(), hidden_states_f32);
    auto grad_k_bias_f32 = grad_key.sum(/*dim=*/0);
    auto grad_v_weight_f32 = at::matmul(grad_value.t(), hidden_states_f32);
    auto grad_v_bias_f32 = grad_value.sum(/*dim=*/0);

    // --- Gradient w.r.t. hidden_states (sum across Q, K, V projection paths). ---
    auto grad_hidden_states_f32 = at::matmul(grad_query, q_weight_f32)
                                + at::matmul(grad_key, k_weight_f32)
                                + at::matmul(grad_value, v_weight_f32);

    // --- Destination-passing-style writes (cast back to bf16). ---
    write_dps(grad_hidden_states_f32, grad_hidden_states, "grad_hidden_states");
    write_dps(grad_q_weight_f32, grad_q_weight, "grad_q_weight");
    write_dps(grad_q_bias_f32, grad_q_bias, "grad_q_bias");
    write_dps(grad_k_weight_f32, grad_k_weight, "grad_k_weight");
    write_dps(grad_k_bias_f32, grad_k_bias, "grad_k_bias");
    write_dps(grad_v_weight_f32, grad_v_weight, "grad_v_weight");
    write_dps(grad_v_bias_f32, grad_v_bias, "grad_v_bias");
    write_dps(grad_out_weight_f32, grad_out_weight, "grad_out_weight");
    write_dps(grad_out_bias_f32, grad_out_bias, "grad_out_bias");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Variable-length audio encoder attention backward via cu_seqlens. Pure "
          "ATen / cuBLAS dispatch: chunk lengths reconstructed on the host from "
          "cu_seqlens, fp32 recompute of per-chunk forward attention, then the "
          "analytic attention backward (softmax vjp, q/k/v grads) and the "
          "projection weight / bias gradients. All compute in fp32; nine grad "
          "outputs cast to bf16 on destination-passing write.");
}
