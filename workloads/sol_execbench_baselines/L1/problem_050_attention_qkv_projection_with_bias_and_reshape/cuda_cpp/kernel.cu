// The deployed problem_050 baseline implements its entire run()
// via pure ATen/cuBLAS dispatch (three at::linear bf16
// cublasGemmEx dispatches + reshape + bf16 .copy_()) — no custom
// CUDA kernels are needed. This .cu file exists solely so that
// torch.utils.cpp_extension.load detects the build as a CUDA
// compilation and threads the CUDA toolkit include paths
// (e.g. cuda_runtime_api.h) through nvcc, since main.cpp's
// ATen includes transitively reference them. Same workaround
// used by problem_065_fused_statistics_projection_and_split and
// problem_004_attention_output_projection_with_reshape_backward.

#include <cuda_runtime.h>
