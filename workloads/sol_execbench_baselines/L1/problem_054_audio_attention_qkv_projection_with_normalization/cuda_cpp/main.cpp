#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the fp32 fused QKV projection
// with per-head RMS normalization on Q and K (V unchanged) for audio
// attention. Per definition.json:
//   hidden_states  : [B, S, hidden_size=1024]               fp32
//   q_proj_weight  : [qkv_out_size=1024, hidden_size=1024]  fp32
//   k_proj_weight  : [qkv_out_size=1024, hidden_size=1024]  fp32
//   v_proj_weight  : [qkv_out_size=1024, hidden_size=1024]  fp32
//   q_norm_weight  : [head_dim=128]                         fp32
//   k_norm_weight  : [head_dim=128]                         fp32
//   eps            : [scalar]                               fp32
//   query_states   : [B, S, num_heads=8, head_dim=128]      fp32 DPS
//   key_states     : [B, S, num_heads=8, head_dim=128]      fp32 DPS
//   value_states   : [B, S, num_heads=8, head_dim=128]      fp32 DPS
//
// Algorithm (mirrors the reference exactly, fp32 in/out throughout;
// no bias on the linear projections; per-head RMSNorm on Q and K only):
//   query = matmul(hidden_states, q_proj_weight.t())
//   key   = matmul(hidden_states, k_proj_weight.t())
//   value = matmul(hidden_states, v_proj_weight.t())
//   query = query.view(B, S, num_heads, head_dim)
//   key   = key.view(B, S, num_heads, head_dim)
//   value = value.view(B, S, num_heads, head_dim)
//   q_var = query.pow(2).mean(dim=-1, keepdim=True)
//   q_normed = query / sqrt(q_var + eps)
//   query_states = q_normed * q_norm_weight
//   (same for key with k_norm_weight)
//   value_states = value  # NO normalization on V
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size  = 1024
//   num_heads    = 8
//   head_dim     = 128
//   qkv_out_size = num_heads * head_dim = 1024
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. Three at::matmul(x, w.t()) calls dispatch cublasSgemm
// for fp32 inputs via PyTorch's cuBLAS handle on Sm80 — the same
// kernel the reference's torch.matmul path selects. RMSNorm chains
// for Q and K run as ATen pow / mean / sqrt / div / mul ops at
// head_dim=128 granularity (the per-head reduction is small;
// bandwidth-friendly). Probe runtime-contract fidelity is
// preserved at the GEMM layer.
void run(
    torch::Tensor hidden_states,
    torch::Tensor q_proj_weight,
    torch::Tensor k_proj_weight,
    torch::Tensor v_proj_weight,
    torch::Tensor q_norm_weight,
    torch::Tensor k_norm_weight,
    double eps,
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states
) {
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_contiguous_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_contiguous_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_contiguous_cuda(q_norm_weight, "q_norm_weight");
    check_fp32_contiguous_cuda(k_norm_weight, "k_norm_weight");
    check_fp32_contiguous_cuda(query_states, "query_states");
    check_fp32_contiguous_cuda(key_states, "key_states");
    check_fp32_contiguous_cuda(value_states, "value_states");

    constexpr int64_t kHiddenSize = 1024;
    constexpr int64_t kNumHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kQkvOutSize = kNumHeads * kHeadDim;  // 1024

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_size]");
    TORCH_CHECK(q_proj_weight.dim() == 2,
                "q_proj_weight must be 2D [qkv_out_size, hidden_size]");
    TORCH_CHECK(k_proj_weight.dim() == 2,
                "k_proj_weight must be 2D [qkv_out_size, hidden_size]");
    TORCH_CHECK(v_proj_weight.dim() == 2,
                "v_proj_weight must be 2D [qkv_out_size, hidden_size]");
    TORCH_CHECK(q_norm_weight.dim() == 1,
                "q_norm_weight must be 1D [head_dim]");
    TORCH_CHECK(k_norm_weight.dim() == 1,
                "k_norm_weight must be 1D [head_dim]");
    TORCH_CHECK(query_states.dim() == 4,
                "query_states must be 4D [B, S, num_heads, head_dim]");
    TORCH_CHECK(key_states.dim() == 4,
                "key_states must be 4D [B, S, num_heads, head_dim]");
    TORCH_CHECK(value_states.dim() == 4,
                "value_states must be 4D [B, S, num_heads, head_dim]");

    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);

    TORCH_CHECK(hidden_states.size(2) == kHiddenSize,
                "hidden_states.size(2) must equal hidden_size=1024, got ",
                hidden_states.size(2));

    TORCH_CHECK(q_proj_weight.size(0) == kQkvOutSize,
                "q_proj_weight.size(0) must equal qkv_out_size=1024");
    TORCH_CHECK(q_proj_weight.size(1) == kHiddenSize,
                "q_proj_weight.size(1) must equal hidden_size=1024");
    TORCH_CHECK(k_proj_weight.size(0) == kQkvOutSize,
                "k_proj_weight.size(0) must equal qkv_out_size=1024");
    TORCH_CHECK(k_proj_weight.size(1) == kHiddenSize,
                "k_proj_weight.size(1) must equal hidden_size=1024");
    TORCH_CHECK(v_proj_weight.size(0) == kQkvOutSize,
                "v_proj_weight.size(0) must equal qkv_out_size=1024");
    TORCH_CHECK(v_proj_weight.size(1) == kHiddenSize,
                "v_proj_weight.size(1) must equal hidden_size=1024");

    TORCH_CHECK(q_norm_weight.size(0) == kHeadDim,
                "q_norm_weight.size(0) must equal head_dim=128");
    TORCH_CHECK(k_norm_weight.size(0) == kHeadDim,
                "k_norm_weight.size(0) must equal head_dim=128");

    for (const auto& [name, t] : std::initializer_list<
             std::pair<const char*, const torch::Tensor&>>{
             {"query_states", query_states},
             {"key_states", key_states},
             {"value_states", value_states}}) {
        TORCH_CHECK(t.size(0) == B,
                    name, ".size(0) must equal hidden_states.size(0)");
        TORCH_CHECK(t.size(1) == S,
                    name, ".size(1) must equal hidden_states.size(1)");
        TORCH_CHECK(t.size(2) == kNumHeads,
                    name, ".size(2) must equal num_heads=8");
        TORCH_CHECK(t.size(3) == kHeadDim,
                    name, ".size(3) must equal head_dim=128");
    }

    // Three fp32 linear projections (no bias) via at::matmul, which
    // dispatches cublasSgemm via PyTorch's cuBLAS handle on Sm80.
    // The pattern matmul(x, w.t()) mirrors the reference exactly.
    auto query = at::matmul(hidden_states, q_proj_weight.t());
    auto key = at::matmul(hidden_states, k_proj_weight.t());
    auto value = at::matmul(hidden_states, v_proj_weight.t());

    // Reshape (B, S, qkv_out_size) -> (B, S, num_heads, head_dim).
    auto query_view = query.view({B, S, kNumHeads, kHeadDim});
    auto key_view = key.view({B, S, kNumHeads, kHeadDim});
    auto value_view = value.view({B, S, kNumHeads, kHeadDim});

    // eps comes in as a Python float (`scalar` in the canonical
    // definition.json); the binding signature receives it as a
    // double. ATen broadcasts the .add(eps) below cleanly.

    // Per-head RMS normalization for Q over the head_dim axis (last
    // dim). Mirrors the reference's pow/mean/sqrt/div/mul chain.
    auto q_var = query_view.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto q_normed = query_view / q_var.add(eps).sqrt();
    query_states.copy_(q_normed * q_norm_weight);

    // Per-head RMS normalization for K.
    auto k_var = key_view.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto k_normed = key_view / k_var.add(eps).sqrt();
    key_states.copy_(k_normed * k_norm_weight);

    // V is unchanged — just copy the reshaped tensor into the DPS
    // output.
    value_states.copy_(value_view);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32 fused QKV projection with per-head RMS normalization "
        "(Q and K only; V unchanged) for audio attention. Pure "
        "ATen/cuBLAS dispatch chain: three at::matmul fp32 cuBLAS "
        "sgemm dispatches (no bias) + three view reshapes + ATen "
        "RMSNorm chains for Q and K (pow / mean / sqrt / div / mul) "
        "+ fp32 .copy_() into pre-allocated DPS outputs. "
        "hidden_size=1024, num_heads=8, head_dim=128, "
        "qkv_out_size=1024 pinned. Destination-passing style.");
}
