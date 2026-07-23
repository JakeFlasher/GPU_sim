#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json input order is
// hidden_states, cos, sin, attention_mask, input_layernorm_weight,
// q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
// post_attention_layernorm_weight, gate_proj_weight, up_proj_weight,
// down_proj_weight, rms_norm_eps, output.

static void check_f32_cuda_contiguous(
    const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.dtype() == torch::kFloat32, name, " must be float32, got ",
        t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void run(
    torch::Tensor hidden_states,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor attention_mask,
    torch::Tensor input_layernorm_weight,
    torch::Tensor q_proj_weight,
    torch::Tensor k_proj_weight,
    torch::Tensor v_proj_weight,
    torch::Tensor o_proj_weight,
    torch::Tensor post_attention_layernorm_weight,
    torch::Tensor gate_proj_weight,
    torch::Tensor up_proj_weight,
    torch::Tensor down_proj_weight,
    double rms_norm_eps,
    torch::Tensor output) {

    check_f32_cuda_contiguous(hidden_states, "hidden_states");
    check_f32_cuda_contiguous(cos, "cos");
    check_f32_cuda_contiguous(sin, "sin");
    check_f32_cuda_contiguous(attention_mask, "attention_mask");
    check_f32_cuda_contiguous(input_layernorm_weight, "input_layernorm_weight");
    check_f32_cuda_contiguous(q_proj_weight, "q_proj_weight");
    check_f32_cuda_contiguous(k_proj_weight, "k_proj_weight");
    check_f32_cuda_contiguous(v_proj_weight, "v_proj_weight");
    check_f32_cuda_contiguous(o_proj_weight, "o_proj_weight");
    check_f32_cuda_contiguous(
        post_attention_layernorm_weight, "post_attention_layernorm_weight");
    check_f32_cuda_contiguous(gate_proj_weight, "gate_proj_weight");
    check_f32_cuda_contiguous(up_proj_weight, "up_proj_weight");
    check_f32_cuda_contiguous(down_proj_weight, "down_proj_weight");
    check_f32_cuda_contiguous(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D");
    TORCH_CHECK(
        hidden_states.size(2) == 4096, "hidden_size must be 4096");
    TORCH_CHECK(
        gate_proj_weight.size(0) == 14336,
        "intermediate_size must be 14336");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_002_decoder_layer_full_block_launcher(
        hidden_states, cos, sin, attention_mask,
        input_layernorm_weight,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        post_attention_layernorm_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight,
        static_cast<float>(rms_norm_eps),
        output,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "LLaMA decoder layer full block. Two custom CUDA kernels "
        "(RMSNorm and SiLU*up) anchor the elementwise fusions; the "
        "seven projections + attention matmuls dispatch through "
        "at::matmul (cuBLAS on the current stream); softmax / cat / "
        "transpose / reshape use at:: ops. Destination-passing style.");
}
