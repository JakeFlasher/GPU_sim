// This baseline implements its entire run() via ATen/cuBLAS
// dispatch (at::conv1d + narrow().copy_()) — no custom CUDA
// kernels are needed. This .cu file exists solely so that
// torch.utils.cpp_extension.load detects the build as a CUDA
// compilation and threads the CUDA toolkit include paths
// (e.g. cuda_runtime_api.h) through nvcc, since main.cpp's
// ATen includes transitively reference them.

#include <cuda_runtime.h>
