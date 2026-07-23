#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry. Per definition.json:
//   A: [M, K] fp16
//   B: [N, K] fp16
//   C (output): [M, N] fp16, pre-allocated by the eval driver.
// N=28672, K=4096 are pinned by the problem definition; M varies.

void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2,
                "A/B/C must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "inner dim K must match: A is [M, K], B is [N, K]");
    TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(0),
                "C must be [M, N] with M=A.size(0), N=B.size(0)");
    TORCH_CHECK(B.size(0) == 28672,
                "N is pinned to 28672 by the problem definition");
    TORCH_CHECK(A.size(1) == 4096,
                "K is pinned to 4096 by the problem definition");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_011_gemm_n28672_k4096_launcher(A, B, C, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp16 GEMM C = A @ B.T via CUTLASS TensorOp on Sm80; "
        "destination-passing style (C pre-allocated).");
}
