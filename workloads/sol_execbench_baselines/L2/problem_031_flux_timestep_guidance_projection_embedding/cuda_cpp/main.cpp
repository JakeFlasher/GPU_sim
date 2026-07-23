#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_f32_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat32,
        name, " must have float32 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. The definition declares:
//   timestep                 [B]                          float32
//   pooled_projections       [B, 768]                     float32
//   freqs                    [384]                        float32
//   timestep_linear1_weight  [3072, 768]                  float32
//   timestep_linear1_bias    [3072]                       float32
//   timestep_linear2_weight  [3072, 3072]                 float32
//   timestep_linear2_bias    [3072]                       float32
//   text_embedder_weight     [3072, 768]                  float32
//   text_embedder_bias       [3072]                       float32
//   conditioning             [B, 3072]                    float32 (DPS, pre-alloc)
//
// Strategy: fuse the reference's pre/post-matmul elementwise work
// (timestep*1000 -> broadcast mul -> sin/cos -> concat, plus SiLU)
// into two small CUDA kernels and let cuBLAS handle the three
// matmuls via at::linear / at::linear_out. The reference path issues
// ~11 small kernel launches per call; this implementation collapses
// the elementwise chain to 2 launches plus 3 cuBLAS launches plus
// one final at::add_ — 6 launches total, all but the matmuls
// dominated by the kernel-launch overhead at the workload's small
// batch sizes (1..20).
void run(
    const torch::Tensor& timestep,
    const torch::Tensor& pooled_projections,
    const torch::Tensor& freqs,
    const torch::Tensor& timestep_linear1_weight,
    const torch::Tensor& timestep_linear1_bias,
    const torch::Tensor& timestep_linear2_weight,
    const torch::Tensor& timestep_linear2_bias,
    const torch::Tensor& text_embedder_weight,
    const torch::Tensor& text_embedder_bias,
    torch::Tensor conditioning) {
    TORCH_CHECK(timestep.dim() == 1, "timestep must be 1D [B]");
    TORCH_CHECK(pooled_projections.dim() == 2, "pooled_projections must be 2D [B, 768]");
    TORCH_CHECK(freqs.dim() == 1, "freqs must be 1D [384]");

    const int64_t B = timestep.size(0);
    TORCH_CHECK(
        pooled_projections.size(0) == B,
        "pooled_projections batch dim must match timestep batch dim");
    TORCH_CHECK(
        pooled_projections.size(1) == 768,
        "pooled_projection_dim must be 768, got ", pooled_projections.size(1));
    TORCH_CHECK(
        freqs.size(0) == 384,
        "freqs must have 384 elements, got ", freqs.size(0));
    TORCH_CHECK(
        conditioning.dim() == 2 && conditioning.size(0) == B
            && conditioning.size(1) == 3072,
        "conditioning must be [B, 3072]");

    check_f32_cuda(timestep, "timestep");
    check_f32_cuda(pooled_projections, "pooled_projections");
    check_f32_cuda(freqs, "freqs");
    check_f32_cuda(timestep_linear1_weight, "timestep_linear1_weight");
    check_f32_cuda(timestep_linear1_bias, "timestep_linear1_bias");
    check_f32_cuda(timestep_linear2_weight, "timestep_linear2_weight");
    check_f32_cuda(timestep_linear2_bias, "timestep_linear2_bias");
    check_f32_cuda(text_embedder_weight, "text_embedder_weight");
    check_f32_cuda(text_embedder_bias, "text_embedder_bias");
    check_f32_cuda(conditioning, "conditioning");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // (1) Generate timestep_embed [B, 768] = cat([cos(args), sin(args)], -1).
    auto timestep_embed = at::empty(
        {B, 768}, timestep.options().dtype(torch::kFloat32));
    problem_031_flux_timestep_embed_launcher(
        timestep, freqs, timestep_embed, stream);

    // (2) x = linear(timestep_embed, w1, b1)  ->  [B, 3072]   (cuBLAS).
    auto x = at::linear(
        timestep_embed, timestep_linear1_weight, timestep_linear1_bias);

    // (3) SiLU in-place: x <- x * sigmoid(x).
    problem_031_flux_silu_inplace_launcher(x, stream);

    // (4) text_embed = linear(pooled_projections, w_text, b_text) (cuBLAS),
    //     written directly into the DPS output buffer.
    at::linear_out(
        conditioning,
        pooled_projections,
        text_embedder_weight,
        text_embedder_bias);

    // (5) timestep_mlp = linear(x, w2, b2)  ->  [B, 3072] (cuBLAS).
    auto timestep_mlp = at::linear(
        x, timestep_linear2_weight, timestep_linear2_bias);

    // (6) conditioning += timestep_mlp.
    conditioning.add_(timestep_mlp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "FLUX timestep + guidance projection embedding: fused "
        "sin/cos/cat + SiLU CUDA kernels around three cuBLAS matmuls. "
        "Inputs / outputs are float32. Destination-passing style.");
}
