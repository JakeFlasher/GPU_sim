// Minimal SM86 branch-target probe. The branch bodies are intentionally
// non-equivalent so nvcc retains divergent control flow instead of lowering
// the condition to a select.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

__device__ __noinline__ unsigned branch_a(unsigned x) {
  for (int i = 0; i < 8; ++i) x = x * 1664525u + 1013904223u;
  return x;
}

__device__ __noinline__ unsigned branch_b(unsigned x) {
  for (int i = 0; i < 8; ++i) x = (x ^ (x >> 3)) + (x << 7) + 0x9e3779b9u;
  return x;
}

template<int Mode>
__global__ void branch_domains(unsigned *out, int iterations) {
  unsigned t = threadIdx.x;
  unsigned x = t + 1;
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    if constexpr (Mode == 0) {
      if (t & 1u) {
        x = branch_a(x);
      } else {
        x = branch_b(x);
      }
    } else if constexpr (Mode == 1) {
      if (t < 16u) {
        x = branch_a(x);
      } else {
        x = branch_b(x);
      }
    } else {
      if (t < 32u) {
        x += 7u;
      }
    }
  }
  out[t] = x;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: branch_domains MODE ITERATIONS\n");
    return 2;
  }
  char *end = nullptr;
  errno = 0; long mode = std::strtol(argv[1], &end, 10);
  if (errno || !*argv[1] || *end) return 2;
  errno = 0; long iterations = std::strtol(argv[2], &end, 10);
  if (errno || !*argv[2] || *end) return 2;
  if (mode < 0 || mode > 2 || iterations < 1 || iterations > 4096) return 2;
  CUDA_OK(cudaSetDevice(0));
  unsigned *out = nullptr;
  CUDA_OK(cudaMalloc(&out, 32 * sizeof(unsigned)));
  CUDA_OK(cudaMemset(out, 0xa5, 32 * sizeof(unsigned)));
  if (mode == 0) branch_domains<0><<<1, 32>>>(out, iterations);
  if (mode == 1) branch_domains<1><<<1, 32>>>(out, iterations);
  if (mode == 2) branch_domains<2><<<1, 32>>>(out, iterations);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  unsigned values[32];
  CUDA_OK(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost));
  bool correct = true;
  for (unsigned t = 0; t < 32; ++t) {
    unsigned x = t + 1;
    for (int i = 0; i < iterations; ++i) {
      if (mode == 2) x += 7u;
      else for (int j = 0; j < 8; ++j) {
        if ((mode == 0 && (t & 1u)) || (mode == 1 && t < 16u))
          x = x * 1664525u + 1013904223u;
        else x = (x ^ (x >> 3)) + (x << 7) + 0x9e3779b9u;
      }
    }
    if (values[t] != x) correct = false;
  }
  CUDA_OK(cudaFree(out));
  if (!correct) { std::fprintf(stderr, "branch result mismatch\n"); return 1; }
  std::printf("correctness=pass mode=%ld iterations=%ld\n", mode, iterations);
}
