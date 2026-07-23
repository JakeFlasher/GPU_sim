#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the encoder RMSNorm step. Produces an FP16
// `normalized` output tensor that subsequently feeds into K/V
// projection matmuls done via `at::matmul` on the host side (so the
// reference's reduced-precision cuBLAS HGEMM path is used and our
// output matches the PyTorch reference exactly).
//
// Inputs / outputs are FP16; intermediate accumulation runs in FP32
// (the sum-of-squares and per-element scale path). The launcher
// expects `normalized` to be pre-allocated by the host caller and is
// destination-passing style.
void problem_073_rmsnorm_launcher(
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& norm_weight,
    torch::Tensor& normalized,
    float eps,
    cudaStream_t stream);
