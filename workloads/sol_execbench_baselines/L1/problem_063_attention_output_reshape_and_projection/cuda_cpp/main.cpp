#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the fused attention output
// reshape + projection. Per definition.json:
//   attn_output     : [B, num_heads=128, S, v_head_dim=128]    bf16
//   o_proj_weight   : [hidden_size=7168,
//                      intermediate_size=16384]                bf16
//   output          : [B, S, hidden_size=7168]                 bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   attn_output_transposed = attn_output.transpose(1, 2)
//     # [B, S, num_heads, v_head_dim] -- non-contiguous view.
//   attn_output_reshaped =
//       attn_output_transposed.reshape(B, S, intermediate_size)
//     # [B, S, intermediate_size=16384]. The .reshape() on the
//     # non-contiguous transposed view triggers a full
//     # contiguous-copy.
//   output = torch.matmul(attn_output_reshaped, o_proj_weight.t())
//     # [B, S, hidden_size=7168]. The explicit `.t()` on the
//     # second operand makes the cuBLAS dispatch transA = N,
//     # transB = T (the cublasGemmEx_2D_bf16_OP_N_OP_T family
//     # pinned by batch-043 + batch-044).
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   num_heads          = 128
//   v_head_dim         = 128
//   intermediate_size  = num_heads * v_head_dim = 16384
//   hidden_size        = 7168
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA
// kernels. The single 2D bf16 matmul dispatch routes through
// cublasGemmEx (transA = N, transB = T orientation; the
// cublasGemmEx_2D_bf16_OP_N_OP_T family). The transpose +
// contiguous-copy + reshape chain unwinds the multi-head
// attention layout to the flat per-token layout the projection
// consumes.
void run(
    torch::Tensor attn_output,
    torch::Tensor o_proj_weight,
    torch::Tensor output
) {
    check_bf16_contiguous_cuda(attn_output, "attn_output");
    check_bf16_contiguous_cuda(o_proj_weight, "o_proj_weight");
    check_bf16_contiguous_cuda(output, "output");

    const c10::cuda::CUDAGuard device_guard(attn_output.device());

    constexpr int64_t kNumHeads = 128;
    constexpr int64_t kVHeadDim = 128;
    constexpr int64_t kIntermediateSize = kNumHeads * kVHeadDim;  // 16384
    constexpr int64_t kHiddenSize = 7168;

    TORCH_CHECK(attn_output.dim() == 4 &&
                attn_output.size(1) == kNumHeads &&
                attn_output.size(3) == kVHeadDim,
                "attn_output must be [B, num_heads=128, S, v_head_dim=128]");
    const int64_t B = attn_output.size(0);
    const int64_t S = attn_output.size(2);
    TORCH_CHECK(o_proj_weight.dim() == 2 &&
                o_proj_weight.size(0) == kHiddenSize &&
                o_proj_weight.size(1) == kIntermediateSize,
                "o_proj_weight must be [hidden_size=7168, intermediate_size=16384]");
    TORCH_CHECK(output.dim() == 3 && output.size(0) == B &&
                output.size(1) == S && output.size(2) == kHiddenSize,
                "output must be [B, S, hidden_size=7168]");

    // Unwind the multi-head attention layout: transpose + reshape.
    // The .reshape() on the transposed (non-contiguous) view
    // triggers a full physical contiguous-copy. .contiguous()
    // before .reshape() is equivalent here and makes the copy
    // explicit at the source level.
    auto attn_output_reshaped = attn_output.transpose(1, 2)
                                    .contiguous()
                                    .view({B, S, kIntermediateSize});

    // Single matmul with explicit weight transpose. The cuBLAS
    // dispatch is transA = N, transB = T (the
    // cublasGemmEx_2D_bf16_OP_N_OP_T family).
    output.copy_(at::matmul(attn_output_reshaped, o_proj_weight.t()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused attention output reshape and projection. Pure ATen / "
        "cuBLAS dispatch chain: transpose + contiguous + view to unwind "
        "the multi-head layout to the flat per-token shape, then one "
        "bf16 2D at::matmul with explicit `.t()` on the weight operand "
        "(cublasGemmEx_2D_bf16_OP_N_OP_T family pinned by batch-043 + "
        "batch-044). Const axes: num_heads=128, v_head_dim=128, "
        "intermediate_size=16384, hidden_size=7168. Destination-passing "
        "style.");
}
