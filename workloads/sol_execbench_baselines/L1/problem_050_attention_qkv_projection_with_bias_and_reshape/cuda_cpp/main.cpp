#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the bf16 Gemma3 fused QKV
// projection with bias and reshape. Per definition.json:
//   hidden_states : [B, S, hidden_size=640]                    bf16
//   q_weight      : [q_out_dim=1024, hidden_size=640]          bf16
//   q_bias        : [q_out_dim=1024]                           bf16
//   k_weight      : [kv_out_dim=256, hidden_size=640]          bf16
//   k_bias        : [kv_out_dim=256]                           bf16
//   v_weight      : [kv_out_dim=256, hidden_size=640]          bf16
//   v_bias        : [kv_out_dim=256]                           bf16
//   query_states  : [B, S, num_attention_heads=4, head_dim=256] bf16 DPS
//   key_states    : [B, S, num_key_value_heads=1, head_dim=256] bf16 DPS
//   value_states  : [B, S, num_key_value_heads=1, head_dim=256] bf16 DPS
//
// Algorithm (mirrors the reference exactly, bf16 in/out throughout
// — the reference does NOT upcast):
//   query_states = (hidden_states @ q_weight.t()) + q_bias
//   key_states   = (hidden_states @ k_weight.t()) + k_bias
//   value_states = (hidden_states @ v_weight.t()) + v_bias
//   query_states = query_states.view(B, S, num_attention_heads, head_dim)
//   key_states   = key_states.view(B, S, num_key_value_heads, head_dim)
//   value_states = value_states.view(B, S, num_key_value_heads, head_dim)
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size           = 640
//   num_attention_heads   = 4
//   num_key_value_heads   = 1
//   head_dim              = 256
//   q_out_dim             = num_attention_heads * head_dim = 1024
//   kv_out_dim            = num_key_value_heads * head_dim = 256
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. Three at::linear(x, w, bias) calls dispatch
// cublasGemmEx for bf16 inputs (CUDA_R_16BF) with fp32
// accumulator (CUBLAS_COMPUTE_32F) via PyTorch's cuBLAS handle on
// Sm80 — the same kernel the reference's torch.matmul + bias path
// selects. Probe runtime-contract fidelity is preserved (the
// cohort-B probe calls cublasGemmEx directly at the Q-projection
// shape).
void run(
    torch::Tensor hidden_states,
    torch::Tensor q_weight,
    torch::Tensor q_bias,
    torch::Tensor k_weight,
    torch::Tensor k_bias,
    torch::Tensor v_weight,
    torch::Tensor v_bias,
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states
) {
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(q_weight, "q_weight");
    check_bf16_contiguous_cuda(q_bias, "q_bias");
    check_bf16_contiguous_cuda(k_weight, "k_weight");
    check_bf16_contiguous_cuda(k_bias, "k_bias");
    check_bf16_contiguous_cuda(v_weight, "v_weight");
    check_bf16_contiguous_cuda(v_bias, "v_bias");
    check_bf16_contiguous_cuda(query_states, "query_states");
    check_bf16_contiguous_cuda(key_states, "key_states");
    check_bf16_contiguous_cuda(value_states, "value_states");

    constexpr int64_t kHiddenSize = 640;
    constexpr int64_t kNumAttentionHeads = 4;
    constexpr int64_t kNumKeyValueHeads = 1;
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kQOutDim = kNumAttentionHeads * kHeadDim;   // 1024
    constexpr int64_t kKvOutDim = kNumKeyValueHeads * kHeadDim;   // 256

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_size]");
    TORCH_CHECK(q_weight.dim() == 2,
                "q_weight must be 2D [q_out_dim, hidden_size]");
    TORCH_CHECK(q_bias.dim() == 1, "q_bias must be 1D [q_out_dim]");
    TORCH_CHECK(k_weight.dim() == 2,
                "k_weight must be 2D [kv_out_dim, hidden_size]");
    TORCH_CHECK(k_bias.dim() == 1, "k_bias must be 1D [kv_out_dim]");
    TORCH_CHECK(v_weight.dim() == 2,
                "v_weight must be 2D [kv_out_dim, hidden_size]");
    TORCH_CHECK(v_bias.dim() == 1, "v_bias must be 1D [kv_out_dim]");
    TORCH_CHECK(query_states.dim() == 4,
                "query_states must be 4D [B, S, num_attention_heads, head_dim]");
    TORCH_CHECK(key_states.dim() == 4,
                "key_states must be 4D [B, S, num_key_value_heads, head_dim]");
    TORCH_CHECK(value_states.dim() == 4,
                "value_states must be 4D [B, S, num_key_value_heads, head_dim]");

    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);

    TORCH_CHECK(hidden_states.size(2) == kHiddenSize,
                "hidden_states.size(2) must equal hidden_size=640, got ",
                hidden_states.size(2));

    TORCH_CHECK(q_weight.size(0) == kQOutDim,
                "q_weight.size(0) must equal q_out_dim=1024");
    TORCH_CHECK(q_weight.size(1) == kHiddenSize,
                "q_weight.size(1) must equal hidden_size=640");
    TORCH_CHECK(q_bias.size(0) == kQOutDim,
                "q_bias.size(0) must equal q_out_dim=1024");

    TORCH_CHECK(k_weight.size(0) == kKvOutDim,
                "k_weight.size(0) must equal kv_out_dim=256");
    TORCH_CHECK(k_weight.size(1) == kHiddenSize,
                "k_weight.size(1) must equal hidden_size=640");
    TORCH_CHECK(k_bias.size(0) == kKvOutDim,
                "k_bias.size(0) must equal kv_out_dim=256");

    TORCH_CHECK(v_weight.size(0) == kKvOutDim,
                "v_weight.size(0) must equal kv_out_dim=256");
    TORCH_CHECK(v_weight.size(1) == kHiddenSize,
                "v_weight.size(1) must equal hidden_size=640");
    TORCH_CHECK(v_bias.size(0) == kKvOutDim,
                "v_bias.size(0) must equal kv_out_dim=256");

    TORCH_CHECK(query_states.size(0) == B,
                "query_states.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(query_states.size(1) == S,
                "query_states.size(1) must equal hidden_states.size(1)");
    TORCH_CHECK(query_states.size(2) == kNumAttentionHeads,
                "query_states.size(2) must equal num_attention_heads=4");
    TORCH_CHECK(query_states.size(3) == kHeadDim,
                "query_states.size(3) must equal head_dim=256");

    TORCH_CHECK(key_states.size(0) == B,
                "key_states.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(key_states.size(1) == S,
                "key_states.size(1) must equal hidden_states.size(1)");
    TORCH_CHECK(key_states.size(2) == kNumKeyValueHeads,
                "key_states.size(2) must equal num_key_value_heads=1");
    TORCH_CHECK(key_states.size(3) == kHeadDim,
                "key_states.size(3) must equal head_dim=256");

    TORCH_CHECK(value_states.size(0) == B,
                "value_states.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(value_states.size(1) == S,
                "value_states.size(1) must equal hidden_states.size(1)");
    TORCH_CHECK(value_states.size(2) == kNumKeyValueHeads,
                "value_states.size(2) must equal num_key_value_heads=1");
    TORCH_CHECK(value_states.size(3) == kHeadDim,
                "value_states.size(3) must equal head_dim=256");

    // Three bf16 linear projections via at::linear, which
    // dispatches cublasGemmEx for bf16 inputs on Sm80.
    // at::linear(x, w, b) computes x @ w.t() + b, matching the
    // reference's torch.matmul(hidden_states, w.t()) + b pattern.
    auto q_out = at::linear(hidden_states, q_weight, q_bias);
    auto k_out = at::linear(hidden_states, k_weight, k_bias);
    auto v_out = at::linear(hidden_states, v_weight, v_bias);

    // Reshape (B, S, out_dim) -> (B, S, heads, head_dim) and
    // write into pre-allocated DPS outputs.
    query_states.copy_(
        q_out.reshape({B, S, kNumAttentionHeads, kHeadDim}));
    key_states.copy_(
        k_out.reshape({B, S, kNumKeyValueHeads, kHeadDim}));
    value_states.copy_(
        v_out.reshape({B, S, kNumKeyValueHeads, kHeadDim}));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "bf16 Gemma3 fused QKV projection with bias and reshape. "
        "Pure ATen/cuBLAS dispatch chain: three at::linear bf16 "
        "cuBLAS GEMM dispatches + reshape + bf16 .copy_() into "
        "pre-allocated DPS outputs. hidden_size=640, "
        "num_attention_heads=4, num_key_value_heads=1, "
        "head_dim=256, q_out_dim=1024, kv_out_dim=256 pinned. "
        "Destination-passing style.");
}
