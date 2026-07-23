#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the Gemma3 fused QKV
// projection + Q/K RMSNorm single-kernel backward. Per
// definition.json:
//   grad_query           : [B, num_attention_heads=4,
//                           seq_len, head_dim=256]    bf16
//   grad_key             : [B, num_key_value_heads=1,
//                           seq_len, head_dim]        bf16
//   grad_value           : [B, num_key_value_heads,
//                           seq_len, head_dim]        bf16
//   hidden_states        : [B, seq_len,
//                           hidden_size=640]          bf16
//   q_weight             : [q_proj_size=1024,
//                           hidden_size=640]          bf16
//   k_weight             : [kv_proj_size=256,
//                           hidden_size=640]          bf16
//   v_weight             : [kv_proj_size,
//                           hidden_size=640]          bf16
//   q_norm_weight        : [head_dim=256]             bf16
//   k_norm_weight        : [head_dim=256]             bf16
//   query_transposed     : [B, num_attention_heads,
//                           seq_len, head_dim]        bf16  (unused-by-backward)
//   key_transposed       : [B, num_key_value_heads,
//                           seq_len, head_dim]        bf16  (unused-by-backward)
//   q_rstd               : [B, num_attention_heads,
//                           seq_len, 1]               fp32
//   k_rstd               : [B, num_key_value_heads,
//                           seq_len, 1]               fp32
//   q_normed             : [B, num_attention_heads,
//                           seq_len, head_dim]        bf16
//   k_normed             : [B, num_key_value_heads,
//                           seq_len, head_dim]        bf16
//   rms_norm_eps         : scalar fp32 (saved-forward)
//   grad_hidden_states   : [B, seq_len, hidden_size]  bf16 DPS
//   grad_q_weight        : [q_proj_size, hidden_size] bf16 DPS
//   grad_k_weight        : [kv_proj_size, hidden_size] bf16 DPS
//   grad_v_weight        : [kv_proj_size, hidden_size] bf16 DPS
//   grad_q_norm_weight   : [head_dim]                 bf16 DPS
//   grad_k_norm_weight   : [head_dim]                 bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   Q RMSNorm backward (Gemma3 uses (1 + q_norm_weight) as scale):
//     q_scale = (1 + q_norm_weight).float()
//     grad_q_norm_weight = (grad_query.float() * q_normed.float())
//                          .sum(dim=(0, 1, 2))
//     grad_q_normed_bf16 = (grad_query.float() * q_scale).to(bf16)
//     q_mean_term = (grad_q_normed_bf16.float()
//                    * q_normed.float()).mean(-1, keepdim=true)
//     grad_q_transposed = (q_rstd
//                          * (grad_q_normed_bf16.float()
//                             - q_mean_term * q_normed.float())).to(bf16)
//   K RMSNorm backward (analogous).
//   V passes through unchanged.
//   Transpose + reshape grad_{q,k,v}_transposed back to
//   [B, S, *_proj_size]. Apply 3 2D bf16 sgemm dispatch pairs:
//     grad_hidden_states = grad_q_proj @ q_weight
//                          + grad_k_proj @ k_weight
//                          + grad_v_proj @ v_weight
//                          (cublasGemmEx 2D bf16 OP_N OP_N -- the
//                           verified family canonical binding).
//     grad_q_weight = grad_q_proj_2d.t() @ hidden_states_2d
//                     (cublasGemmEx 2D bf16 OP_T OP_N).
//     grad_k_weight, grad_v_weight analogously.
void run(
    torch::Tensor grad_query,
    torch::Tensor grad_key,
    torch::Tensor grad_value,
    torch::Tensor hidden_states,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    torch::Tensor query_transposed,
    torch::Tensor key_transposed,
    torch::Tensor q_rstd,
    torch::Tensor k_rstd,
    torch::Tensor q_normed,
    torch::Tensor k_normed,
    double rms_norm_eps,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight,
    torch::Tensor grad_q_norm_weight,
    torch::Tensor grad_k_norm_weight
) {
    check_bf16_contiguous_cuda(grad_query, "grad_query");
    check_bf16_contiguous_cuda(grad_key, "grad_key");
    check_bf16_contiguous_cuda(grad_value, "grad_value");
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(q_weight, "q_weight");
    check_bf16_contiguous_cuda(k_weight, "k_weight");
    check_bf16_contiguous_cuda(v_weight, "v_weight");
    check_bf16_contiguous_cuda(q_norm_weight, "q_norm_weight");
    check_bf16_contiguous_cuda(k_norm_weight, "k_norm_weight");
    check_bf16_contiguous_cuda(query_transposed, "query_transposed");
    check_bf16_contiguous_cuda(key_transposed, "key_transposed");
    check_fp32_contiguous_cuda(q_rstd, "q_rstd");
    check_fp32_contiguous_cuda(k_rstd, "k_rstd");
    check_bf16_contiguous_cuda(q_normed, "q_normed");
    check_bf16_contiguous_cuda(k_normed, "k_normed");
    check_bf16_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_contiguous_cuda(grad_q_weight, "grad_q_weight");
    check_bf16_contiguous_cuda(grad_k_weight, "grad_k_weight");
    check_bf16_contiguous_cuda(grad_v_weight, "grad_v_weight");
    check_fp32_contiguous_cuda(grad_q_norm_weight, "grad_q_norm_weight");
    check_fp32_contiguous_cuda(grad_k_norm_weight, "grad_k_norm_weight");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 640;
    constexpr int64_t kNumAttentionHeads = 4;
    constexpr int64_t kNumKeyValueHeads = 1;
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kQProjSize = kNumAttentionHeads * kHeadDim;  // 1024
    constexpr int64_t kKvProjSize = kNumKeyValueHeads * kHeadDim;  // 256

    TORCH_CHECK(hidden_states.dim() == 3 &&
                hidden_states.size(2) == kHiddenSize,
                "hidden_states must be [B, S, hidden_size=640]");
    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);

    TORCH_CHECK(grad_query.dim() == 4 &&
                grad_query.size(0) == B &&
                grad_query.size(1) == kNumAttentionHeads &&
                grad_query.size(2) == S &&
                grad_query.size(3) == kHeadDim,
                "grad_query must be [B, num_attention_heads, S, head_dim]");
    TORCH_CHECK(grad_key.dim() == 4 &&
                grad_key.size(0) == B &&
                grad_key.size(1) == kNumKeyValueHeads &&
                grad_key.size(2) == S &&
                grad_key.size(3) == kHeadDim,
                "grad_key must be [B, num_key_value_heads, S, head_dim]");
    TORCH_CHECK(grad_value.dim() == 4 &&
                grad_value.size(0) == B &&
                grad_value.size(1) == kNumKeyValueHeads &&
                grad_value.size(2) == S &&
                grad_value.size(3) == kHeadDim,
                "grad_value must be [B, num_key_value_heads, S, head_dim]");
    TORCH_CHECK(q_weight.dim() == 2 &&
                q_weight.size(0) == kQProjSize &&
                q_weight.size(1) == kHiddenSize,
                "q_weight must be [q_proj_size, hidden_size]");
    TORCH_CHECK(k_weight.dim() == 2 &&
                k_weight.size(0) == kKvProjSize &&
                k_weight.size(1) == kHiddenSize,
                "k_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(v_weight.dim() == 2 &&
                v_weight.size(0) == kKvProjSize &&
                v_weight.size(1) == kHiddenSize,
                "v_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(q_norm_weight.dim() == 1 &&
                q_norm_weight.size(0) == kHeadDim,
                "q_norm_weight must be [head_dim]");
    TORCH_CHECK(k_norm_weight.dim() == 1 &&
                k_norm_weight.size(0) == kHeadDim,
                "k_norm_weight must be [head_dim]");
    TORCH_CHECK(q_rstd.dim() == 4 &&
                q_rstd.size(0) == B &&
                q_rstd.size(1) == kNumAttentionHeads &&
                q_rstd.size(2) == S &&
                q_rstd.size(3) == 1,
                "q_rstd must be [B, num_attention_heads, S, 1]");
    TORCH_CHECK(k_rstd.dim() == 4 &&
                k_rstd.size(0) == B &&
                k_rstd.size(1) == kNumKeyValueHeads &&
                k_rstd.size(2) == S &&
                k_rstd.size(3) == 1,
                "k_rstd must be [B, num_key_value_heads, S, 1]");
    TORCH_CHECK(q_normed.dim() == 4 &&
                q_normed.size(0) == B &&
                q_normed.size(1) == kNumAttentionHeads &&
                q_normed.size(2) == S &&
                q_normed.size(3) == kHeadDim,
                "q_normed must be [B, num_attention_heads, S, head_dim]");
    TORCH_CHECK(k_normed.dim() == 4 &&
                k_normed.size(0) == B &&
                k_normed.size(1) == kNumKeyValueHeads &&
                k_normed.size(2) == S &&
                k_normed.size(3) == kHeadDim,
                "k_normed must be [B, num_key_value_heads, S, head_dim]");

    TORCH_CHECK(grad_hidden_states.dim() == 3 &&
                grad_hidden_states.size(0) == B &&
                grad_hidden_states.size(1) == S &&
                grad_hidden_states.size(2) == kHiddenSize,
                "grad_hidden_states must be [B, S, hidden_size]");
    TORCH_CHECK(grad_q_weight.dim() == 2 &&
                grad_q_weight.size(0) == kQProjSize &&
                grad_q_weight.size(1) == kHiddenSize,
                "grad_q_weight must be [q_proj_size, hidden_size]");
    TORCH_CHECK(grad_k_weight.dim() == 2 &&
                grad_k_weight.size(0) == kKvProjSize &&
                grad_k_weight.size(1) == kHiddenSize,
                "grad_k_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(grad_v_weight.dim() == 2 &&
                grad_v_weight.size(0) == kKvProjSize &&
                grad_v_weight.size(1) == kHiddenSize,
                "grad_v_weight must be [kv_proj_size, hidden_size]");
    TORCH_CHECK(grad_q_norm_weight.dim() == 1 &&
                grad_q_norm_weight.size(0) == kHeadDim,
                "grad_q_norm_weight must be [head_dim]");
    TORCH_CHECK(grad_k_norm_weight.dim() == 1 &&
                grad_k_norm_weight.size(0) == kHeadDim,
                "grad_k_norm_weight must be [head_dim]");

    const std::vector<int64_t> kBatchHeadSeqDims = {0, 1, 2};

    // ----------------------------------------------------------------
    // Q RMSNorm backward (Gemma3 uses (1 + q_norm_weight) as scale).
    // Mirror the reference's float32 math then cast back to bf16.
    // ----------------------------------------------------------------
    auto grad_query_float = grad_query.to(torch::kFloat32);
    auto q_scale = (1.0 + q_norm_weight.to(torch::kFloat32));
    auto q_normed_float = q_normed.to(torch::kFloat32);

    grad_q_norm_weight.copy_(
        (grad_query_float * q_normed_float)
            .sum(kBatchHeadSeqDims));

    auto grad_q_normed = grad_query_float * q_scale;
    auto grad_q_normed_bf16 = grad_q_normed.to(torch::kBFloat16);
    auto grad_q_normed_bf16_float = grad_q_normed_bf16.to(torch::kFloat32);

    auto q_mean_term = (grad_q_normed_bf16_float * q_normed_float)
                           .mean(/*dim=*/-1, /*keepdim=*/true);
    auto grad_q_transposed_float = q_rstd
                                   * (grad_q_normed_bf16_float
                                      - q_mean_term * q_normed_float);
    auto grad_q_transposed = grad_q_transposed_float.to(torch::kBFloat16);

    // ----------------------------------------------------------------
    // K RMSNorm backward (analogous).
    // ----------------------------------------------------------------
    auto grad_key_float = grad_key.to(torch::kFloat32);
    auto k_scale = (1.0 + k_norm_weight.to(torch::kFloat32));
    auto k_normed_float = k_normed.to(torch::kFloat32);

    grad_k_norm_weight.copy_(
        (grad_key_float * k_normed_float)
            .sum(kBatchHeadSeqDims));

    auto grad_k_normed = grad_key_float * k_scale;
    auto grad_k_normed_bf16 = grad_k_normed.to(torch::kBFloat16);
    auto grad_k_normed_bf16_float = grad_k_normed_bf16.to(torch::kFloat32);

    auto k_mean_term = (grad_k_normed_bf16_float * k_normed_float)
                           .mean(/*dim=*/-1, /*keepdim=*/true);
    auto grad_k_transposed_float = k_rstd
                                   * (grad_k_normed_bf16_float
                                      - k_mean_term * k_normed_float);
    auto grad_k_transposed = grad_k_transposed_float.to(torch::kBFloat16);

    // V has no normalization; passes through.
    auto grad_v_transposed = grad_value;

    // ----------------------------------------------------------------
    // Transpose + reshape back to [B, S, *_proj_size].
    // ----------------------------------------------------------------
    auto grad_q_reshaped = grad_q_transposed.transpose(1, 2).contiguous()
                               .view({B, S, kQProjSize});
    auto grad_k_reshaped = grad_k_transposed.transpose(1, 2).contiguous()
                               .view({B, S, kKvProjSize});
    auto grad_v_reshaped = grad_v_transposed.transpose(1, 2).contiguous()
                               .view({B, S, kKvProjSize});

    // ----------------------------------------------------------------
    // Linear projection backwards: grad_hidden_states accumulates
    // contributions from Q + K + V branches (3 2D bf16 sgemm
    // dispatches OP_N OP_N -- verified family). grad_q/k/v_weight
    // each bind one 2D bf16 sgemm dispatch (OP_T OP_N -- same
    // verified family).
    // ----------------------------------------------------------------
    auto grad_hidden_from_q = at::matmul(grad_q_reshaped, q_weight);
    auto grad_hidden_from_k = at::matmul(grad_k_reshaped, k_weight);
    auto grad_hidden_from_v = at::matmul(grad_v_reshaped, v_weight);
    grad_hidden_states.copy_(
        grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v);

    auto grad_q_proj_2d = grad_q_reshaped.reshape({-1, kQProjSize});
    auto grad_k_proj_2d = grad_k_reshaped.reshape({-1, kKvProjSize});
    auto grad_v_proj_2d = grad_v_reshaped.reshape({-1, kKvProjSize});
    auto hidden_states_2d = hidden_states.reshape({-1, kHiddenSize});

    grad_q_weight.copy_(
        at::matmul(grad_q_proj_2d.transpose(0, 1), hidden_states_2d));
    grad_k_weight.copy_(
        at::matmul(grad_k_proj_2d.transpose(0, 1), hidden_states_2d));
    grad_v_weight.copy_(
        at::matmul(grad_v_proj_2d.transpose(0, 1), hidden_states_2d));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Gemma3 fused QKV projection + Q/K RMSNorm single-kernel "
        "backward. Pure ATen / cuBLAS dispatch chain: six 2D bf16 "
        "cublasGemmEx dispatches (three grad_input via at::matmul(g, W) "
        "OP_N OP_N -- the verified cublasGemmEx_2D_bf16_OP_N_OP_N "
        "family canonical binding; three grad_weight via g.t() @ x "
        "OP_T OP_N -- same family) + RMSNorm backward (with the "
        "Gemma3 1+w scaling convention) via ATen elementwise / "
        "reduction ops. Const axes: hidden_size=640, "
        "num_attention_heads=4, num_key_value_heads=1, head_dim=256, "
        "q_proj_size=1024, kv_proj_size=256. Destination-passing style; "
        "6 gradient outputs.");
}
