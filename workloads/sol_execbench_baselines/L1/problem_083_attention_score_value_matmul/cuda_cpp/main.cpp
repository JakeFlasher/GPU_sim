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

// Destination-passing-style entry for the bf16 attention score-value
// matmul + transpose + reshape (second major matmul in attention).
// Per definition.json:
//   attention_weights : [B, num_heads=20, Q, K]            bf16
//   value             : [B, num_heads=20, K, head_dim=64]  bf16
//   output            : [B, Q, hidden_size=1280]           bf16 DPS
//
// Algorithm (mirrors the reference exactly, bf16 in/out):
//   attn = torch.matmul(attention_weights, value)
//     # [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
//   attn = attn.transpose(1, 2).contiguous()  # -> [B, Q, H, D]
//   output = attn.reshape(B, Q, hidden_size)  # -> [B, Q, H*D]
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   num_heads   = 20
//   head_dim    = 64
//   hidden_size = num_heads * head_dim = 1280
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. at::matmul for 4D bf16 inputs dispatches
// cublasGemmStridedBatchedEx (CUDA_R_16BF inputs/output,
// CUBLAS_COMPUTE_32F accumulator) — the batched analog of the
// regular cublasGemmEx used for 2D matmul. Probe runtime-contract
// fidelity is preserved (the cohort-B probe calls
// cublasGemmStridedBatchedEx directly at the chosen workload's
// shape).
void run(
    torch::Tensor attention_weights,
    torch::Tensor value,
    torch::Tensor output
) {
    check_bf16_contiguous_cuda(attention_weights, "attention_weights");
    check_bf16_contiguous_cuda(value, "value");
    check_bf16_contiguous_cuda(output, "output");

    constexpr int64_t kNumHeads = 20;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;  // 1280

    TORCH_CHECK(attention_weights.dim() == 4,
                "attention_weights must be 4D [B, num_heads, Q, K]");
    TORCH_CHECK(value.dim() == 4,
                "value must be 4D [B, num_heads, K, head_dim]");
    TORCH_CHECK(output.dim() == 3,
                "output must be 3D [B, Q, hidden_size]");

    const int64_t B = attention_weights.size(0);
    const int64_t Q = attention_weights.size(2);
    const int64_t K = attention_weights.size(3);

    TORCH_CHECK(attention_weights.size(1) == kNumHeads,
                "attention_weights.size(1) must equal num_heads=20");

    TORCH_CHECK(value.size(0) == B,
                "value.size(0) must equal attention_weights.size(0)");
    TORCH_CHECK(value.size(1) == kNumHeads,
                "value.size(1) must equal num_heads=20");
    TORCH_CHECK(value.size(2) == K,
                "value.size(2) must equal attention_weights.size(3) (K)");
    TORCH_CHECK(value.size(3) == kHeadDim,
                "value.size(3) must equal head_dim=64");

    TORCH_CHECK(output.size(0) == B,
                "output.size(0) must equal attention_weights.size(0)");
    TORCH_CHECK(output.size(1) == Q,
                "output.size(1) must equal attention_weights.size(2) (Q)");
    TORCH_CHECK(output.size(2) == kHiddenSize,
                "output.size(2) must equal hidden_size=1280");

    // One bf16 batched matmul via at::matmul; dispatches
    // cublasGemmStridedBatchedEx for 4D bf16 inputs on Sm80.
    //   [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
    auto attn = at::matmul(attention_weights, value);

    // Transpose (1, 2): [B, H, Q, D] -> [B, Q, H, D] with
    // contiguous() to materialize the row-major bf16 buffer.
    auto attn_qhd = attn.transpose(1, 2).contiguous();

    // Reshape [B, Q, H, D] -> [B, Q, hidden_size] and write into
    // the pre-allocated DPS output.
    output.copy_(attn_qhd.reshape({B, Q, kHiddenSize}));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "bf16 attention score-value matmul + transpose + reshape. "
        "Pure ATen/cuBLAS dispatch chain: one at::matmul 4D bf16 "
        "cublasGemmStridedBatchedEx dispatch + transpose(1, 2) + "
        "contiguous + reshape + bf16 .copy_() into pre-allocated "
        "DPS output. num_heads=20, head_dim=64, hidden_size=1280 "
        "pinned. Destination-passing style.");
}
