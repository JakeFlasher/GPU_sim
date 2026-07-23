#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the GELU-approximate
// feedforward backward pass. Per definition.json:
//   grad_output         : [B, S, dim=1536]                fp32
//   hidden_states       : [B, S, dim=1536]                fp32
//   weight1             : [hidden_dim=6144, dim=1536]     fp32
//   weight2             : [dim=1536, hidden_dim=6144]     fp32
//   linear1_out         : [B, S, hidden_dim=6144]         fp32
//   tanh_inner          : [B, S, hidden_dim=6144]         fp32
//   gelu_out            : [B, S, hidden_dim=6144]         fp32
//   grad_hidden_states  : [B, S, dim=1536]                fp32 DPS
//   grad_weight1        : [hidden_dim=6144, dim=1536]     fp32 DPS
//   grad_weight2        : [dim=1536, hidden_dim=6144]     fp32 DPS
//
// Algorithm (mirrors the reference exactly):
//   Forward was:
//     linear1_out = hidden_states @ weight1.T
//     inner = sqrt(2/pi) * (linear1_out + 0.044715 * linear1_out^3)
//     tanh_inner = tanh(inner)
//     gelu_out = 0.5 * linear1_out * (1 + tanh_inner)
//     output = gelu_out @ weight2.T
//
//   Backward through Linear2:
//     grad_gelu_out = grad_output @ weight2                    [B,S,I]
//     grad_output_2d = grad_output.reshape(-1, D)
//     gelu_out_2d = gelu_out.reshape(-1, I)
//     grad_weight2 = grad_output_2d.t() @ gelu_out_2d          [D,I]
//
//   Backward through GELU (using precomputed tanh_inner):
//     gelu_grad_part1 = 0.5 * (1 + tanh_inner)
//     sech_squared = 1 - tanh_inner^2
//     d_inner_dx = sqrt(2/pi) * (1 + 0.134145 * linear1_out^2)
//                  # 0.134145 = 3 * 0.044715
//     gelu_grad_part2 = 0.5 * linear1_out * sech_squared * d_inner_dx
//     gelu_grad = gelu_grad_part1 + gelu_grad_part2
//     grad_linear1_out = grad_gelu_out * gelu_grad
//
//   Backward through Linear1:
//     grad_hidden_states = grad_linear1_out @ weight1          [B,S,D]
//     grad_linear1_out_2d = grad_linear1_out.reshape(-1, I)
//     hidden_states_2d = hidden_states.reshape(-1, D)
//     grad_weight1 = grad_linear1_out_2d.t() @ hidden_states_2d  [I,D]
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   dim         = 1536
//   hidden_dim  = 6144   (= 4 * dim, the expansion factor)
//
// Implementation: pure ATen / cuBLAS dispatch — NO custom CUDA
// kernels. The four 2D fp32 matmul dispatches all route through
// cublasSgemm; the grad_input path (OP_N OP_N) lands on the
// verified cublasSgemm_basic_fp32 family. The precomputed
// tanh_inner / linear1_out inputs short-circuit the forward
// recomputation that problem_073 has to perform inline.
void run(
    torch::Tensor grad_output,
    torch::Tensor hidden_states,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor linear1_out,
    torch::Tensor tanh_inner,
    torch::Tensor gelu_out,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_weight1,
    torch::Tensor grad_weight2
) {
    check_fp32_contiguous_cuda(grad_output, "grad_output");
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(weight1, "weight1");
    check_fp32_contiguous_cuda(weight2, "weight2");
    check_fp32_contiguous_cuda(linear1_out, "linear1_out");
    check_fp32_contiguous_cuda(tanh_inner, "tanh_inner");
    check_fp32_contiguous_cuda(gelu_out, "gelu_out");
    check_fp32_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_fp32_contiguous_cuda(grad_weight1, "grad_weight1");
    check_fp32_contiguous_cuda(grad_weight2, "grad_weight2");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    constexpr int64_t kDim = 1536;
    constexpr int64_t kHiddenDim = 6144;

    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(2) == kDim,
                "grad_output must be [B, S, dim=1536]");
    const int64_t B = grad_output.size(0);
    const int64_t S = grad_output.size(1);
    TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(0) == B &&
                hidden_states.size(1) == S && hidden_states.size(2) == kDim,
                "hidden_states must be [B, S, dim=1536]");
    TORCH_CHECK(weight1.dim() == 2 && weight1.size(0) == kHiddenDim &&
                weight1.size(1) == kDim,
                "weight1 must be [hidden_dim=6144, dim=1536]");
    TORCH_CHECK(weight2.dim() == 2 && weight2.size(0) == kDim &&
                weight2.size(1) == kHiddenDim,
                "weight2 must be [dim=1536, hidden_dim=6144]");
    TORCH_CHECK(linear1_out.dim() == 3 && linear1_out.size(0) == B &&
                linear1_out.size(1) == S &&
                linear1_out.size(2) == kHiddenDim,
                "linear1_out must be [B, S, hidden_dim=6144]");
    TORCH_CHECK(tanh_inner.dim() == 3 && tanh_inner.size(0) == B &&
                tanh_inner.size(1) == S && tanh_inner.size(2) == kHiddenDim,
                "tanh_inner must be [B, S, hidden_dim=6144]");
    TORCH_CHECK(gelu_out.dim() == 3 && gelu_out.size(0) == B &&
                gelu_out.size(1) == S && gelu_out.size(2) == kHiddenDim,
                "gelu_out must be [B, S, hidden_dim=6144]");
    TORCH_CHECK(grad_hidden_states.sizes() == hidden_states.sizes(),
                "grad_hidden_states shape must match hidden_states");
    TORCH_CHECK(grad_weight1.sizes() == weight1.sizes(),
                "grad_weight1 shape must match weight1");
    TORCH_CHECK(grad_weight2.sizes() == weight2.sizes(),
                "grad_weight2 shape must match weight2");

    constexpr double kSqrt2OverPi = 0.7978845608028654;
    constexpr double kGeluInnerCoeff = 0.044715;
    constexpr double kGeluDerivativeCoeff = 0.134145;  // = 3 * 0.044715

    // Backward through Linear2.
    auto grad_gelu_out = at::matmul(grad_output, weight2);
    auto grad_output_2d = grad_output.reshape({-1, kDim});
    auto gelu_out_2d = gelu_out.reshape({-1, kHiddenDim});
    (void)kSqrt2OverPi;  // referenced below in d_inner_dx
    grad_weight2.copy_(
        at::matmul(grad_output_2d.transpose(0, 1), gelu_out_2d));

    // Backward through GELU (tanh-approximation derivative) using
    // the precomputed `tanh_inner` and `linear1_out` inputs.
    auto gelu_grad_part1 = 0.5 * (1.0 + tanh_inner);
    auto sech_squared = 1.0 - tanh_inner * tanh_inner;
    auto d_inner_dx =
        kSqrt2OverPi * (1.0 + kGeluDerivativeCoeff * linear1_out * linear1_out);
    (void)kGeluInnerCoeff;  // documented in header; not reused at runtime
                            // because d_inner_dx already incorporates
                            // `3 * coeff` via kGeluDerivativeCoeff.
    auto gelu_grad_part2 =
        0.5 * linear1_out * sech_squared * d_inner_dx;
    auto gelu_grad = gelu_grad_part1 + gelu_grad_part2;
    auto grad_linear1_out = grad_gelu_out * gelu_grad;

    // Backward through Linear1.
    grad_hidden_states.copy_(at::matmul(grad_linear1_out, weight1));
    auto grad_linear1_out_2d = grad_linear1_out.reshape({-1, kHiddenDim});
    auto hidden_states_2d = hidden_states.reshape({-1, kDim});
    grad_weight1.copy_(
        at::matmul(grad_linear1_out_2d.transpose(0, 1), hidden_states_2d));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GELU-approximate feedforward backward (Linear -> GELU tanh-approx "
        "-> Linear). Pure ATen / cuBLAS dispatch chain: four fp32 2D "
        "at::matmul calls + elementwise GELU derivative (using the "
        "precomputed tanh_inner / linear1_out inputs that short-circuit "
        "forward recomputation). All matmuls dispatch cublasSgemm "
        "(legacy fp32 API); the grad_input path (OP_N OP_N) lands on "
        "the verified cublasSgemm_basic_fp32 family. Destination-passing "
        "style.");
}
