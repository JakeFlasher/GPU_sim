// The deployed problem_028 baseline implements its entire run()
// via pure ATen / cuBLAS dispatch (GQA + RoPE attention-core
// backward: q/k/v/o projections through at::linear, attention
// and scaled-dot-product backwards through at::matmul, plus
// softmax backward + RoPE rotate-half + GQA head expansion
// expressed as ATen elementwise / reduction ops). No custom
// CUDA kernels are needed. This .cu file exists solely so that
// torch.utils.cpp_extension.load detects the build as a CUDA
// compilation and threads the CUDA toolkit include paths (e.g.
// cuda_runtime_api.h) through nvcc, since main.cpp's ATen/cuda
// includes (ATen/cuda/CUDAContext.h, c10/cuda/CUDAGuard.h)
// transitively reference them. Same workaround as problem_001,
// problem_003, problem_017, problem_064, problem_073, etc.

#include <cuda_runtime.h>
