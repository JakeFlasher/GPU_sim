// The deployed problem_004 baseline implements its entire run()
// via pure ATen/cuBLAS dispatch (bf16 -> fp32 upcast + two
// cuBLAS sgemm calls + reshape/transpose + bf16 downcast) — no
// custom CUDA kernels are needed. This .cu file exists solely
// so that torch.utils.cpp_extension.load detects the build as a
// CUDA compilation and threads the CUDA toolkit include paths
// (e.g. cuda_runtime_api.h) through nvcc, since main.cpp's
// ATen includes transitively reference them. Same workaround
// used by problem_065_fused_statistics_projection_and_split.

#include <cuda_runtime.h>
