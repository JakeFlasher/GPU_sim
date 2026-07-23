#include <cuda_runtime.h>
constexpr int kBlocks = 36864;
constexpr int kActiveThreads = 128;
__global__ void gs_reuse(const float* __restrict__ x, float* __restrict__ y, int n) {
  if (threadIdx.x >= kActiveThreads) return;
  int i = blockIdx.x * kActiveThreads + threadIdx.x;
  // Adjacent blocks share one input tile. Cluster selection preserves more nearby
  // block context; independent block thinning can split that reuse context.
  int block_base = (blockIdx.x >> 1) * kActiveThreads;
  int lane = threadIdx.x;
  if (i < n) {
    y[i] = x[block_base + lane] + x[block_base + (lane ^ 1)]
         + x[block_base + ((lane + 32) & (kActiveThreads - 1))];
  }
}
int main() {
  const int N = kBlocks * kActiveThreads;
  float *x, *y; cudaMalloc(&x, N*sizeof(float)); cudaMalloc(&y, N*sizeof(float));
  cudaMemset(x, 0, N*sizeof(float)); cudaMemset(y, 0, N*sizeof(float));
  gs_reuse<<<kBlocks, 1024>>>(x, y, N); cudaDeviceSynchronize();
  if (cudaGetLastError()!=cudaSuccess) return 1;
  cudaFree(x); cudaFree(y); return 0;
}
