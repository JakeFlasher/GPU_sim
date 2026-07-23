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

// Destination-passing-style entry for the GQA-style attention
// value projection with reshape + transpose. Per definition.json:
//   hidden_states  : [B, S, hidden_size=5120]                  bf16
//   v_proj_weight  : [kv_out_dim=1024, hidden_size=5120]       bf16
//   value_states   : [B, num_kv_heads=8, S, head_dim=128]      bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   value_proj = F.linear(hidden_states, v_proj_weight)
//     # F.linear computes `hidden_states @ v_proj_weight.T`. On
//     # 2D bf16 contiguous operands this dispatches cublasGemmEx
//     # with bf16 input/output, FP32 compute type, transA = N,
//     # transB = T (the novel cublasGemmEx_2D_bf16_OP_N_OP_T
//     # family pinned by batch-043).
//     # shape: [B, S, kv_out_dim=1024]
//   value_states = value_proj.view(B, S, num_kv_heads=8, head_dim=128)
//   value_states = value_states.transpose(1, 2).contiguous()
//     # logical reshape into [B, num_kv_heads, S, head_dim]
//     # with a contiguous physical copy.
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size  = 5120
//   num_kv_heads = 8
//   head_dim     = 128
//   kv_out_dim   = num_kv_heads * head_dim = 1024
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA
// kernels. The single 2D bf16 matmul dispatch routes through
// cublasGemmEx (transA = N, transB = T orientation; the
// cublasGemmEx_2D_bf16_OP_N_OP_T family). The
// view + transpose + contiguous chain produces the GQA-attention
// layout that downstream attention kernels consume.
void run(
    torch::Tensor hidden_states,
    torch::Tensor v_proj_weight,
    torch::Tensor value_states
) {
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(v_proj_weight, "v_proj_weight");
    check_bf16_contiguous_cuda(value_states, "value_states");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 5120;
    constexpr int64_t kNumKvHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kKvOutDim = kNumKvHeads * kHeadDim;  // 1024

    TORCH_CHECK(hidden_states.dim() == 3 &&
                hidden_states.size(2) == kHiddenSize,
                "hidden_states must be [B, S, hidden_size=5120]");
    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);
    TORCH_CHECK(v_proj_weight.dim() == 2 &&
                v_proj_weight.size(0) == kKvOutDim &&
                v_proj_weight.size(1) == kHiddenSize,
                "v_proj_weight must be [kv_out_dim=1024, hidden_size=5120]");
    TORCH_CHECK(value_states.dim() == 4 && value_states.size(0) == B &&
                value_states.size(1) == kNumKvHeads &&
                value_states.size(2) == S &&
                value_states.size(3) == kHeadDim,
                "value_states must be [B, num_kv_heads=8, S, head_dim=128]");

    // Linear projection (the only cuBLAS dispatch on this path).
    auto value_proj = at::linear(hidden_states, v_proj_weight);

    // View into [B, S, num_kv_heads, head_dim] then transpose +
    // contiguous to match the GQA-attention layout. The
    // .contiguous() produces a fresh physical buffer; .copy_()
    // into the DPS output is the canonical materialisation.
    auto value_reshaped = value_proj.view({B, S, kNumKvHeads, kHeadDim});
    value_states.copy_(value_reshaped.transpose(1, 2).contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA-style attention value projection with reshape and "
        "transpose. Pure ATen / cuBLAS dispatch chain: one "
        "F.linear-equivalent 2D bf16 cublasGemmEx call at "
        "transA=N transB=T orientation (the novel "
        "cublasGemmEx_2D_bf16_OP_N_OP_T family pinned by "
        "batch-043), then view + transpose + contiguous to "
        "materialise the [B, num_kv_heads, S, head_dim] layout "
        "downstream attention consumes. Const axes: "
        "hidden_size=5120, num_kv_heads=8, head_dim=128, "
        "kv_out_dim=1024. Destination-passing style.");
}
