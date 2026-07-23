#include <cuda_runtime.h>
constexpr int kBlocks = 36864;       // buffer keeps >=128 segments after scheduler skew
constexpr int kActiveThreads = 128;  // bound trace volume; block residency stays 1
__global__ void gs_copy(const float* __restrict__ x, float* __restrict__ y, int n) {
  if (threadIdx.x >= kActiveThreads) return;
  int i = blockIdx.x * kActiveThreads + threadIdx.x;
  if (i < n) y[i] = x[i];
}
int main() {
  const int N = kBlocks * kActiveThreads;
  float *x, *y; cudaMalloc(&x, N*sizeof(float)); cudaMalloc(&y, N*sizeof(float));
  cudaMemset(x, 0, N*sizeof(float)); cudaMemset(y, 0, N*sizeof(float));
  gs_copy<<<kBlocks, 1024>>>(x, y, N); cudaDeviceSynchronize();
  if (cudaGetLastError()!=cudaSuccess) return 1;
  cudaFree(x); cudaFree(y); return 0;
}
