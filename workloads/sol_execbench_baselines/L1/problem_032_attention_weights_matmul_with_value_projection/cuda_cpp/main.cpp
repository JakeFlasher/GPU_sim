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

// Destination-passing-style entry for the bf16 attention weights @
// value matmul + transpose + reshape. Per definition.json:
//   attn_weights  : [B, num_attention_heads=40, S, S]      bf16
//   value_states  : [B, num_attention_heads=40, S, head_dim=128] bf16
//   output        : [B, S, hidden_size=5120]               bf16 DPS
//
// Algorithm (mirrors the reference exactly, bf16 in/out):
//   attn = torch.matmul(attn_weights, value_states)
//     # [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
//   attn = attn.transpose(1, 2)  # -> [B, S, H, D]
//   output = attn.reshape(B, S, hidden_size)
//     # -> [B, S, H*D=5120]
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   num_attention_heads = 40
//   head_dim            = 128
//   hidden_size         = num_attention_heads * head_dim = 5120
//
// Self-attention variant where Q=K=seq_len (single S axis); compare
// problem_083 which has separate seq_len_q and seq_len_kv axes.
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. at::matmul on 4D bf16 inputs dispatches
// cublasGemmStridedBatchedEx (CUDA_R_16BF inputs/output,
// CUBLAS_COMPUTE_32F accumulator) — the batched-strided analog of
// the regular cublasGemmEx used for 2D matmul. Probe runtime-
// contract fidelity is preserved (the cohort-B probe calls
// cublasGemmStridedBatchedEx directly at the chosen workload's
// shape).
void run(
    torch::Tensor attn_weights,
    torch::Tensor value_states,
    torch::Tensor output
) {
    check_bf16_contiguous_cuda(attn_weights, "attn_weights");
    check_bf16_contiguous_cuda(value_states, "value_states");
    check_bf16_contiguous_cuda(output, "output");

    constexpr int64_t kNumHeads = 40;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;  // 5120

    TORCH_CHECK(attn_weights.dim() == 4,
                "attn_weights must be 4D [B, num_heads, S, S]");
    TORCH_CHECK(value_states.dim() == 4,
                "value_states must be 4D [B, num_heads, S, head_dim]");
    TORCH_CHECK(output.dim() == 3,
                "output must be 3D [B, S, hidden_size]");

    const int64_t B = attn_weights.size(0);
    const int64_t S = attn_weights.size(2);

    TORCH_CHECK(attn_weights.size(1) == kNumHeads,
                "attn_weights.size(1) must equal num_attention_heads=40");
    TORCH_CHECK(attn_weights.size(3) == S,
                "attn_weights.size(3) (Skv) must equal "
                "attn_weights.size(2) (Sq) — self-attention contract");

    TORCH_CHECK(value_states.size(0) == B,
                "value_states.size(0) must equal attn_weights.size(0)");
    TORCH_CHECK(value_states.size(1) == kNumHeads,
                "value_states.size(1) must equal num_attention_heads=40");
    TORCH_CHECK(value_states.size(2) == S,
                "value_states.size(2) must equal attn_weights.size(2) (S)");
    TORCH_CHECK(value_states.size(3) == kHeadDim,
                "value_states.size(3) must equal head_dim=128");

    TORCH_CHECK(output.size(0) == B,
                "output.size(0) must equal attn_weights.size(0)");
    TORCH_CHECK(output.size(1) == S,
                "output.size(1) must equal attn_weights.size(2) (S)");
    TORCH_CHECK(output.size(2) == kHiddenSize,
                "output.size(2) must equal hidden_size=5120");

    // One bf16 batched matmul via at::matmul; dispatches
    // cublasGemmStridedBatchedEx for 4D bf16 inputs on Sm80.
    //   [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    auto attn = at::matmul(attn_weights, value_states);

    // Transpose (1, 2): [B, H, S, D] -> [B, S, H, D] with
    // contiguous() to materialize the row-major bf16 buffer.
    auto attn_shd = attn.transpose(1, 2).contiguous();

    // Reshape [B, S, H, D] -> [B, S, hidden_size] and write into
    // the pre-allocated DPS output.
    output.copy_(attn_shd.reshape({B, S, kHiddenSize}));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "bf16 attention weights @ value matmul + transpose + "
        "reshape. Pure ATen/cuBLAS dispatch chain: one at::matmul "
        "4D bf16 cublasGemmStridedBatchedEx dispatch + "
        "transpose(1, 2) + contiguous + reshape to (B, S, H*D) + "
        "bf16 .copy_() into pre-allocated DPS output. "
        "num_attention_heads=40, head_dim=128, hidden_size=5120 "
        "pinned. Destination-passing style.");
}
