#include <cuda_runtime.h>
constexpr int kBlocks = 36864;
constexpr int kActiveThreads = 128;
constexpr int kTileItems = 512;
// Per-block VARYING stride -> varying distinct sectors per warp with a FIXED
// record count. Every warp executes the same four loads, while the per-class
// active-lane count changes their sector totals, so the calibrated count estimate
// is non-vacuous. Each block owns its input tile so the one-ghost replay protocol
// is not confounded by deeper cross-segment reuse.
__global__ void gs_var(const float* __restrict__ x, float* __restrict__ y) {
  if (threadIdx.x >= kActiveThreads) return;
  int out_i = blockIdx.x * kActiveThreads + threadIdx.x;
  int block_base = blockIdx.x * kTileItems;
  int block_class = int(blockIdx.x & 3);
  int stride = 1 << block_class;
  int active_lanes = 29 + (block_class % 3);
  int lane = threadIdx.x & 31;
  int warp_tile = (threadIdx.x >> 5) * 128;
  float acc = 0.0f;
  if (lane < active_lanes) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      int tile_i = warp_tile + ((lane * stride + k) & 127);
      acc += x[block_base + tile_i];
    }
  }
  y[out_i] = acc;
}
int main() {
  const int x_items = kBlocks * kTileItems;
  const int y_items = kBlocks * kActiveThreads;
  float *x, *y;
  cudaMalloc(&x, x_items*sizeof(float));
  cudaMalloc(&y, y_items*sizeof(float));
  cudaMemset(x, 0, x_items*sizeof(float));
  cudaMemset(y, 0, y_items*sizeof(float));
  gs_var<<<kBlocks, 1024>>>(x, y); cudaDeviceSynchronize();
  if (cudaGetLastError()!=cudaSuccess) return 1;
  cudaFree(x); cudaFree(y); return 0;
}
