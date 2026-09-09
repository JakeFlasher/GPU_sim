#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ std::uint32_t step(std::uint32_t x) {
  std::uint32_t y;
  asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;" : "=r"(y) : "r"(x));
  return y;
}

__device__ __forceinline__ std::uint64_t clk() {
  std::uint64_t x;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(x));
  return x;
}

__global__ __launch_bounds__(32) void probe(std::uint64_t *times, int iters) {
  if (threadIdx.x != 0) return;
  std::uint32_t x = 0x9e3779b9u;
  times[0] = clk();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x = step(x);
  }
  times[iters + 1] = clk();
  if (x == 0) times[iters + 2] = 1;
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 256;
  std::vector<std::uint64_t> h(static_cast<std::size_t>(iters + 3));
  std::uint64_t *d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(*d));
  probe<<<1, 32>>>(d, iters);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) return 1;
  cudaMemcpy(h.data(), d, h.size() * sizeof(*d), cudaMemcpyDeviceToHost);
  std::printf("start=%llu end=%llu total=%llu\n",
      (unsigned long long)h[0], (unsigned long long)h[iters + 1],
      (unsigned long long)(h[iters + 1] - h[0]));
  for (int i = 0; i < iters && i < 16; ++i)
    std::printf("%d %llu\n", i, (unsigned long long)(h[i + 1] - h[i]));
  cudaFree(d);
  return 0;
}
