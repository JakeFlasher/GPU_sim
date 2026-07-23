#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the fp16 GEMM `C = A @ B.T` baseline.
// A: [M, K] fp16 row-major. B: [N, K] fp16 row-major. C: [M, N] fp16
// row-major (destination-passing-style; pre-allocated by the eval
// driver). N and K are problem constants (N=128, K=2048); M varies
// per workload.
void problem_004_gemm_n128_k2048_launcher(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    cudaStream_t stream);
