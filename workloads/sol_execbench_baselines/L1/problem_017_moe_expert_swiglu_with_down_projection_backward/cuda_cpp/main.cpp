#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for MoE expert SwiGLU with down projection.
//
// Reference (problem_017_moe_expert_swiglu_with_down_projection_backward):
// the analytic backward written out by hand as explicit tensor algebra
// (NO torch.autograd). Every grad is computed in float32 and cast back to
// bfloat16 for the four destination buffers. Reproduced op-for-op below.
//
// Per definition.json (insertion order):
//   inputs  (all bfloat16):
//     grad_output  : [num_tokens, hidden_size]
//     x            : [num_tokens, hidden_size]
//     gate_weight  : [intermediate_size, hidden_size]
//     up_weight    : [intermediate_size, hidden_size]
//     down_weight  : [hidden_size, intermediate_size]
//     gate         : [num_tokens, intermediate_size]
//     gate_sigmoid : [num_tokens, intermediate_size]   (PRECOMPUTED)
//     gate_silu    : [num_tokens, intermediate_size]   (PRECOMPUTED)
//     up           : [num_tokens, intermediate_size]
//     intermediate : [num_tokens, intermediate_size]
//   outputs (all bfloat16, DPS pre-alloc, written in place):
//     grad_x           : [num_tokens, hidden_size]
//     grad_gate_weight : [intermediate_size, hidden_size]
//     grad_up_weight   : [intermediate_size, hidden_size]
//     grad_down_weight : [hidden_size, intermediate_size]
//
// Forward chain the backward inverts:
//   gate         = x @ gate_weight.T          (F.linear)
//   gate_sigmoid = sigmoid(gate)
//   gate_silu    = gate * gate_sigmoid        (SiLU)
//   up           = x @ up_weight.T            (F.linear)
//   intermediate = gate_silu * up
//   output       = intermediate @ down_weight.T
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels.
// The .t().mm() / .mm() calls route through the SAME cuBLAS GEMM the
// reference's torch ops use (bf16 inputs upcast to fp32 here, fp32 accum),
// so they bit-match the reference's float32 algebra. The SiLU derivative,
// every .to(float32)/.to(bfloat16) cast point, and the grad_x accumulation
// order all match the reference exactly. gate_sigmoid / gate_silu are
// consumed as precomputed inputs (never recomputed).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& gate_weight,
    const torch::Tensor& up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& gate,
    const torch::Tensor& gate_sigmoid,
    const torch::Tensor& gate_silu,
    const torch::Tensor& up,
    const torch::Tensor& intermediate,
    torch::Tensor grad_x,
    torch::Tensor grad_gate_weight,
    torch::Tensor grad_up_weight,
    torch::Tensor grad_down_weight
) {
    check_bf16_cuda(grad_output, "grad_output");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(gate_weight, "gate_weight");
    check_bf16_cuda(up_weight, "up_weight");
    check_bf16_cuda(down_weight, "down_weight");
    check_bf16_cuda(gate, "gate");
    check_bf16_cuda(gate_sigmoid, "gate_sigmoid");
    check_bf16_cuda(gate_silu, "gate_silu");
    check_bf16_cuda(up, "up");
    check_bf16_cuda(intermediate, "intermediate");
    check_bf16_cuda(grad_x, "grad_x");
    check_bf16_cuda(grad_gate_weight, "grad_gate_weight");
    check_bf16_cuda(grad_up_weight, "grad_up_weight");
    check_bf16_cuda(grad_down_weight, "grad_down_weight");

    TORCH_CHECK(grad_output.dim() == 2, "grad_output must be 2D, got ", grad_output.dim(), "D");
    TORCH_CHECK(x.dim() == 2, "x must be 2D, got ", x.dim(), "D");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D, got ", gate_weight.dim(), "D");
    TORCH_CHECK(up_weight.dim() == 2, "up_weight must be 2D, got ", up_weight.dim(), "D");
    TORCH_CHECK(down_weight.dim() == 2, "down_weight must be 2D, got ", down_weight.dim(), "D");
    TORCH_CHECK(gate.dim() == 2, "gate must be 2D, got ", gate.dim(), "D");
    TORCH_CHECK(gate_sigmoid.dim() == 2, "gate_sigmoid must be 2D, got ", gate_sigmoid.dim(), "D");
    TORCH_CHECK(gate_silu.dim() == 2, "gate_silu must be 2D, got ", gate_silu.dim(), "D");
    TORCH_CHECK(up.dim() == 2, "up must be 2D, got ", up.dim(), "D");
    TORCH_CHECK(intermediate.dim() == 2, "intermediate must be 2D, got ", intermediate.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_output.device());

    const auto kF32 = torch::kFloat32;
    const auto kBF16 = torch::kBFloat16;

    // Convert to float32 for numerical stability (mirrors the reference).
    auto grad_output_f32 = grad_output.to(kF32);
    auto x_f32 = x.to(kF32);
    auto gate_weight_f32 = gate_weight.to(kF32);
    auto up_weight_f32 = up_weight.to(kF32);
    auto down_weight_f32 = down_weight.to(kF32);
    auto gate_f32 = gate.to(kF32);
    auto gate_sigmoid_f32 = gate_sigmoid.to(kF32);
    auto gate_silu_f32 = gate_silu.to(kF32);
    auto up_f32 = up.to(kF32);
    auto intermediate_f32 = intermediate.to(kF32);

    // Gradient w.r.t. down_weight.
    //   down_proj: output = intermediate @ down_weight.T
    //   grad_down_weight = grad_output.T @ intermediate   -> [hidden, intermediate]
    auto grad_down_weight_f32 = grad_output_f32.t().mm(intermediate_f32);

    // Gradient w.r.t. intermediate.
    //   grad_intermediate = grad_output @ down_weight      -> [num_tokens, intermediate]
    auto grad_intermediate = grad_output_f32.mm(down_weight_f32);

    // Gradient w.r.t. gate_silu and up (element-wise; intermediate = gate_silu * up).
    auto grad_gate_silu = grad_intermediate * up_f32;
    auto grad_up = grad_intermediate * gate_silu_f32;

    // Gradient w.r.t. gate (through SiLU activation).
    //   gate_silu = gate * sigmoid(gate)
    //   d(gate_silu)/d(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    auto grad_gate = grad_gate_silu * gate_sigmoid_f32 *
                     (1.0 + gate_f32 * (1.0 - gate_sigmoid_f32));

    // Gradient w.r.t. up_weight.
    //   up_proj: up = x @ up_weight.T
    //   grad_up_weight = grad_up.T @ x                     -> [intermediate, hidden]
    auto grad_up_weight_f32 = grad_up.t().mm(x_f32);

    // Gradient w.r.t. gate_weight.
    //   gate_proj: gate = x @ gate_weight.T
    //   grad_gate_weight = grad_gate.T @ x                 -> [intermediate, hidden]
    auto grad_gate_weight_f32 = grad_gate.t().mm(x_f32);

    // Gradient w.r.t. x (input). Accumulate gate and up projection paths,
    // in the reference's exact order: grad_gate @ gate_weight first.
    auto grad_x_f32 = grad_gate.mm(gate_weight_f32) + grad_up.mm(up_weight_f32);

    // Cast each result back to bfloat16 and write into the DPS buffers.
    auto grad_x_bf16 = grad_x_f32.to(kBF16);
    auto grad_gate_weight_bf16 = grad_gate_weight_f32.to(kBF16);
    auto grad_up_weight_bf16 = grad_up_weight_f32.to(kBF16);
    auto grad_down_weight_bf16 = grad_down_weight_f32.to(kBF16);

    TORCH_CHECK(grad_x_bf16.sizes() == grad_x.sizes(),
                "grad_x result shape ", grad_x_bf16.sizes(),
                " does not match pre-allocated ", grad_x.sizes());
    TORCH_CHECK(grad_gate_weight_bf16.sizes() == grad_gate_weight.sizes(),
                "grad_gate_weight result shape ", grad_gate_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_gate_weight.sizes());
    TORCH_CHECK(grad_up_weight_bf16.sizes() == grad_up_weight.sizes(),
                "grad_up_weight result shape ", grad_up_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_up_weight.sizes());
    TORCH_CHECK(grad_down_weight_bf16.sizes() == grad_down_weight.sizes(),
                "grad_down_weight result shape ", grad_down_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_down_weight.sizes());

    grad_x.copy_(grad_x_bf16);
    grad_gate_weight.copy_(grad_gate_weight_bf16);
    grad_up_weight.copy_(grad_up_weight_bf16);
    grad_down_weight.copy_(grad_down_weight_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for MoE expert SwiGLU with down projection. Pure "
        "ATen / cuBLAS dispatch: analytic grad algebra (down / intermediate "
        "/ gate_silu / up / gate / up_weight / gate_weight / x) reproduced "
        "op-for-op from the reference, fp32 compute with bf16 in/out, "
        "precomputed gate_sigmoid / gate_silu consumed directly. "
        "Destination-passing style (4 outputs).");
}
