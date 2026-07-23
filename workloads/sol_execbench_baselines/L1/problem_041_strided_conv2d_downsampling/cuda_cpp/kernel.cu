// Stub translation unit.
//
// This port delegates all compute to ATen / cuDNN from main.cpp (no custom
// __global__ kernels). A bare .cu source is still required so torch's
// cpp_extension builds the baseline as a CUDA extension (adding the CUDA
// include paths); a .cpp-only extension omits them and main.cpp's
// <ATen/cuda/CUDAContext.h> include fails with
// "cuda_runtime_api.h: No such file or directory".
#include <cuda_runtime.h>
