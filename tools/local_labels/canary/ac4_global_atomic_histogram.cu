// Scorability witness (L2 surface): a global-atomic histogram. Each thread reads an
// input element and atomicAdd's to a global bin, so the kernel's memory traffic is
// dominated by global atomics (ATOM/RED.E -> L2). Pre-flip the ATOMIC records were
// stripped to `unsupported_memory_path`, so the sim's L2 sector surface for the
// atomic traffic was `na` (the histogram L2 sector-hit-rate was starved — the sim
// only replayed GLOBAL+LOCAL, so atomics resolving in L2 were invisible: sim 0.0%
// vs NCU 99.9%, per plans/phase2_sim_core_progress.md). Post-flip the atomic
// records are admitted with a resolved allocation_id, so the L2 surface becomes
// scorable.
#include <cuda_runtime.h>

#include <cstdio>

__global__ void atomic_histogram(const int* __restrict__ in,
                                 int* __restrict__ bins, int n, int nbins) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) {
    int b = in[t] % nbins;
    atomicAdd(&bins[b], 1);  // global atomic bin update -> L2 atomic traffic
  }
}

int main() {
  const int N = 4096, NBINS = 256;
  int *in, *bins;
  if (cudaMalloc(&in, N * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&bins, NBINS * sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "atomic_histogram: cudaMalloc failed\n");
    return 2;
  }
  cudaMemset(in, 0, N * sizeof(int));
  cudaMemset(bins, 0, NBINS * sizeof(int));
  atomic_histogram<<<(N + 127) / 128, 128>>>(in, bins, N, NBINS);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "atomic_histogram: launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaFree(in);
  cudaFree(bins);
  printf("atomic_histogram: ok\n");
  return 0;
}
