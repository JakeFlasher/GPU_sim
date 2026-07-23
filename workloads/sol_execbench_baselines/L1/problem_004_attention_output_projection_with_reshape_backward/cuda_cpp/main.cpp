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

// Destination-passing-style entry for the attention output
// projection's backward pass with reshape. Per definition.json:
//   grad_output       : [B, S, hidden_size=2048]     bf16
//   reshaped          : [B, S, hidden_size=2048]     bf16
//   weight            : [hidden_size=2048, hidden_size=2048] bf16
//   grad_attn_output  : [B, num_heads=32, S, head_dim=64] bf16 DPS
//   grad_weight       : [hidden_size=2048, hidden_size=2048] bf16 DPS
//
// Algorithm (mirroring the PyTorch reference, fp32 inner compute):
//   grad_output_f32  = grad_output.to(fp32)
//   reshaped_f32     = reshaped.to(fp32)
//   weight_f32       = weight.to(fp32)
//   grad_output_2d   = grad_output_f32.reshape(-1, hidden_size)
//   reshaped_2d      = reshaped_f32.reshape(-1, hidden_size)
//   grad_weight_f32  = grad_output_2d.t() @ reshaped_2d
//   grad_reshaped_2d = grad_output_2d @ weight_f32
//   grad_reshaped    = grad_reshaped_2d.reshape(B, S, hidden_size)
//   grad_transposed  = grad_reshaped.reshape(B, S, num_heads, head_dim)
//   grad_attn_output = grad_transposed.transpose(1, 2).contiguous()
//   output bf16 via .to(bf16)
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   num_heads   = 32
//   head_dim    = 64
//   hidden_size = 2048 (= num_heads * head_dim)
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. The two fp32 sgemm dispatches go through cuBLAS at the
// same shape PyTorch's .mm() uses, so probe runtime-contract
// fidelity is preserved (probe calls cublasSgemm directly at
// the same shape).
void run(
    torch::Tensor grad_output,
    torch::Tensor reshaped,
    torch::Tensor weight,
    torch::Tensor grad_attn_output,
    torch::Tensor grad_weight
) {
    check_bf16_contiguous_cuda(grad_output, "grad_output");
    check_bf16_contiguous_cuda(reshaped, "reshaped");
    check_bf16_contiguous_cuda(weight, "weight");
    check_bf16_contiguous_cuda(grad_attn_output, "grad_attn_output");
    check_bf16_contiguous_cuda(grad_weight, "grad_weight");

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;  // 2048

    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must be 3D [B, S, hidden_size]");
    TORCH_CHECK(reshaped.dim() == 3,
                "reshaped must be 3D [B, S, hidden_size]");
    TORCH_CHECK(weight.dim() == 2,
                "weight must be 2D [hidden_size, hidden_size]");
    TORCH_CHECK(grad_attn_output.dim() == 4,
                "grad_attn_output must be 4D [B, num_heads, S, head_dim]");
    TORCH_CHECK(grad_weight.dim() == 2,
                "grad_weight must be 2D [hidden_size, hidden_size]");

    const int64_t B = grad_output.size(0);
    const int64_t S = grad_output.size(1);

    TORCH_CHECK(grad_output.size(2) == kHiddenSize,
                "grad_output.size(2) must equal hidden_size=2048, got ",
                grad_output.size(2));
    TORCH_CHECK(reshaped.size(0) == B,
                "reshaped.size(0) must equal grad_output.size(0)");
    TORCH_CHECK(reshaped.size(1) == S,
                "reshaped.size(1) must equal grad_output.size(1)");
    TORCH_CHECK(reshaped.size(2) == kHiddenSize,
                "reshaped.size(2) must equal hidden_size=2048");
    TORCH_CHECK(weight.size(0) == kHiddenSize,
                "weight.size(0) must equal hidden_size=2048");
    TORCH_CHECK(weight.size(1) == kHiddenSize,
                "weight.size(1) must equal hidden_size=2048");
    TORCH_CHECK(grad_attn_output.size(0) == B,
                "grad_attn_output.size(0) must equal grad_output.size(0)");
    TORCH_CHECK(grad_attn_output.size(1) == kNumHeads,
                "grad_attn_output.size(1) must equal num_heads=32");
    TORCH_CHECK(grad_attn_output.size(2) == S,
                "grad_attn_output.size(2) must equal grad_output.size(1)");
    TORCH_CHECK(grad_attn_output.size(3) == kHeadDim,
                "grad_attn_output.size(3) must equal head_dim=64");
    TORCH_CHECK(grad_weight.size(0) == kHiddenSize,
                "grad_weight.size(0) must equal hidden_size=2048");
    TORCH_CHECK(grad_weight.size(1) == kHiddenSize,
                "grad_weight.size(1) must equal hidden_size=2048");

    // Upcast bf16 -> fp32 (matches reference exactly).
    auto grad_output_f32 = grad_output.to(torch::kFloat32);
    auto reshaped_f32 = reshaped.to(torch::kFloat32);
    auto weight_f32 = weight.to(torch::kFloat32);

    // Reshape to 2D (B*S, hidden_size).
    auto grad_output_2d = grad_output_f32.reshape({B * S, kHiddenSize});
    auto reshaped_2d = reshaped_f32.reshape({B * S, kHiddenSize});

    // GEMM A: grad_weight_f32 = grad_output_2d.t() @ reshaped_2d.
    // Dispatches cuBLAS sgemm at (M=hidden_size, K=B*S,
    // N=hidden_size).
    auto grad_weight_f32 = grad_output_2d.t().mm(reshaped_2d);
    grad_weight.copy_(grad_weight_f32.to(torch::kBFloat16));

    // GEMM B: grad_reshaped_2d = grad_output_2d @ weight_f32.
    // Dispatches cuBLAS sgemm at (M=B*S, K=hidden_size,
    // N=hidden_size).
    auto grad_reshaped_2d = grad_output_2d.mm(weight_f32);

    // Reshape (B*S, hidden_size) -> (B, S, num_heads, head_dim) ->
    // transpose to (B, num_heads, S, head_dim).
    auto grad_reshaped = grad_reshaped_2d.reshape(
        {B, S, kNumHeads, kHeadDim});
    auto grad_attn_output_f32 = grad_reshaped.transpose(1, 2).contiguous();
    grad_attn_output.copy_(grad_attn_output_f32.to(torch::kBFloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "bf16 attention output projection backward with reshape. "
        "Pure ATen/cuBLAS dispatch: bf16 -> fp32 upcast + 2 cuBLAS "
        "sgemm dispatches + reshape/transpose + bf16 downcast. "
        "num_heads=32, head_dim=64, hidden_size=2048 pinned. "
        "Destination-passing style.");
}
