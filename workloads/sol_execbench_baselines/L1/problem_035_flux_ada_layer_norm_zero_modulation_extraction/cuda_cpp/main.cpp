#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the AdaLayerNormZero modulation
// extraction Linear-with-bias + 6-chunk multi-output kernel. Per
// definition.json:
//   emb        : [batch_size, inner_dim=3072] fp32
//   weight     : [output_dim=18432, inner_dim=3072] fp32
//   bias       : [output_dim=18432] fp32
//   shift_msa  : [batch_size, inner_dim=3072] fp32 (DPS)
//   scale_msa  : [batch_size, inner_dim=3072] fp32 (DPS)
//   gate_msa   : [batch_size, inner_dim=3072] fp32 (DPS)
//   shift_mlp  : [batch_size, inner_dim=3072] fp32 (DPS)
//   scale_mlp  : [batch_size, inner_dim=3072] fp32 (DPS)
//   gate_mlp   : [batch_size, inner_dim=3072] fp32 (DPS)
//
// inner_dim = 3072 and output_dim = inner_dim * 6 = 18432 are pinned
// by the problem definition. Const-axis pins are enforced on EVERY
// tensor that the kernel indexes with that fixed assumption,
// following the runtime-check audit discipline.

void run(
    torch::Tensor emb,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor shift_msa,
    torch::Tensor scale_msa,
    torch::Tensor gate_msa,
    torch::Tensor shift_mlp,
    torch::Tensor scale_mlp,
    torch::Tensor gate_mlp
) {
    constexpr int64_t kInnerDim = 3072;
    constexpr int64_t kOutputDim = 6 * kInnerDim;  // 18432

    check_fp32_contiguous_cuda(emb, "emb");
    check_fp32_contiguous_cuda(weight, "weight");
    check_fp32_contiguous_cuda(bias, "bias");
    check_fp32_contiguous_cuda(shift_msa, "shift_msa");
    check_fp32_contiguous_cuda(scale_msa, "scale_msa");
    check_fp32_contiguous_cuda(gate_msa, "gate_msa");
    check_fp32_contiguous_cuda(shift_mlp, "shift_mlp");
    check_fp32_contiguous_cuda(scale_mlp, "scale_mlp");
    check_fp32_contiguous_cuda(gate_mlp, "gate_mlp");

    TORCH_CHECK(emb.dim() == 2, "emb must be 2D [B, inner_dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [output_dim, inner_dim]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [output_dim]");
    TORCH_CHECK(shift_msa.dim() == 2, "shift_msa must be 2D");
    TORCH_CHECK(scale_msa.dim() == 2, "scale_msa must be 2D");
    TORCH_CHECK(gate_msa.dim() == 2, "gate_msa must be 2D");
    TORCH_CHECK(shift_mlp.dim() == 2, "shift_mlp must be 2D");
    TORCH_CHECK(scale_mlp.dim() == 2, "scale_mlp must be 2D");
    TORCH_CHECK(gate_mlp.dim() == 2, "gate_mlp must be 2D");

    const int64_t batch_size = emb.size(0);

    TORCH_CHECK(emb.size(1) == kInnerDim,
                "emb.size(1) must equal inner_dim=3072, got ", emb.size(1));
    TORCH_CHECK(weight.size(0) == kOutputDim,
                "weight.size(0) must equal output_dim=18432, got ",
                weight.size(0));
    TORCH_CHECK(weight.size(1) == kInnerDim,
                "weight.size(1) must equal inner_dim=3072, got ",
                weight.size(1));
    TORCH_CHECK(bias.size(0) == kOutputDim,
                "bias.size(0) must equal output_dim=18432, got ",
                bias.size(0));

    auto check_output = [&](const torch::Tensor& t, const std::string& name) {
        TORCH_CHECK(t.size(0) == batch_size,
                    name, ".size(0) must equal emb.size(0) (= batch_size)");
        TORCH_CHECK(t.size(1) == kInnerDim,
                    name, ".size(1) must equal inner_dim=3072, got ",
                    t.size(1));
    };
    check_output(shift_msa, "shift_msa");
    check_output(scale_msa, "scale_msa");
    check_output(gate_msa, "gate_msa");
    check_output(shift_mlp, "shift_mlp");
    check_output(scale_mlp, "scale_mlp");
    check_output(gate_mlp, "gate_mlp");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_035_flux_ada_layer_norm_zero_modulation_extraction_launcher(
        emb, weight, bias,
        shift_msa, scale_msa, gate_msa,
        shift_mlp, scale_mlp, gate_mlp,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "AdaLayerNormZero modulation extraction: fp32 Linear-with-bias "
        "(K=inner_dim=3072, N=output_dim=18432) + 6-chunk multi-output "
        "DPS in one custom CUDA kernel. inner_dim=3072 pinned on every "
        "input/output tensor.");
}
