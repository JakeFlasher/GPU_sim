#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_bf16_contiguous(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kBFloat16,
        name, " must have bfloat16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_fp32_contiguous(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. Per upstream definition.json:
//   grad_output        : [batch, seq, 4096]   bfloat16
//   input_ids          : [batch, seq]         int64
//   hidden_states_fp32 : [batch, seq, 4096]   float32
//   rstd               : [batch, seq, 1]      float32
//   norm_weight        : [4096]               bfloat16
//   grad_embed_weight  : [65536, 4096]        bfloat16 (DPS)
//   grad_norm_weight   : [4096]               bfloat16 (DPS)
//
// vocab_size=65536 and hidden_size=4096 are pinned by the
// definition.json `const` axes.
void run(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_ids,
    const torch::Tensor& hidden_states_fp32,
    const torch::Tensor& rstd,
    const torch::Tensor& norm_weight,
    torch::Tensor grad_embed_weight,
    torch::Tensor grad_norm_weight) {

    TORCH_CHECK(
        grad_output.dim() == 3,
        "grad_output must be 3D [batch, seq, hidden]; got ",
        grad_output.dim(), "D");
    const int64_t hidden_size = grad_output.size(2);
    TORCH_CHECK(
        hidden_size == 4096,
        "hidden_size must be 4096 for problem_087, got ", hidden_size);
    TORCH_CHECK(
        input_ids.dim() == 2
            && input_ids.size(0) == grad_output.size(0)
            && input_ids.size(1) == grad_output.size(1),
        "input_ids must be [batch, seq]");
    TORCH_CHECK(
        input_ids.dtype() == torch::kInt64,
        "input_ids must be int64, got ", input_ids.dtype());
    TORCH_CHECK(
        hidden_states_fp32.dim() == 3
            && hidden_states_fp32.size(0) == grad_output.size(0)
            && hidden_states_fp32.size(1) == grad_output.size(1)
            && hidden_states_fp32.size(2) == hidden_size,
        "hidden_states_fp32 must be [batch, seq, 4096]");
    TORCH_CHECK(
        rstd.dim() == 3
            && rstd.size(0) == grad_output.size(0)
            && rstd.size(1) == grad_output.size(1)
            && rstd.size(2) == 1,
        "rstd must be [batch, seq, 1]");
    TORCH_CHECK(
        norm_weight.dim() == 1 && norm_weight.size(0) == hidden_size,
        "norm_weight must be 1D length 4096");
    TORCH_CHECK(
        grad_embed_weight.dim() == 2
            && grad_embed_weight.size(0) == 65536
            && grad_embed_weight.size(1) == hidden_size,
        "grad_embed_weight must be [65536, 4096]");
    TORCH_CHECK(
        grad_norm_weight.dim() == 1
            && grad_norm_weight.size(0) == hidden_size,
        "grad_norm_weight must be 1D length 4096");

    check_bf16_contiguous(grad_output, "grad_output");
    TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
    TORCH_CHECK(
        input_ids.is_contiguous(), "input_ids must be contiguous");
    check_fp32_contiguous(hidden_states_fp32, "hidden_states_fp32");
    check_fp32_contiguous(rstd, "rstd");
    check_bf16_contiguous(norm_weight, "norm_weight");
    check_bf16_contiguous(grad_embed_weight, "grad_embed_weight");
    check_bf16_contiguous(grad_norm_weight, "grad_norm_weight");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_087_embedding_layernorm_backward_launcher(
        grad_output,
        input_ids,
        hidden_states_fp32,
        rstd,
        norm_weight,
        grad_embed_weight,
        grad_norm_weight,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Embedding + RMSNorm backward; BF16 inputs/outputs, FP32 "
        "accumulators; one fused per-row scatter-add kernel plus "
        "two cast-to-bf16 finalize kernels; destination-passing "
        "style.");
}
