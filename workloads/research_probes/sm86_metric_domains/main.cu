// Deterministic one-warp memory-domain probe. No NCU-derived parameters.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

// Each block owns a 4 KiB region. Pattern 0: contiguous words (four sectors
// in one line); 1: same word (32 aliases); 2: separate 128B lines; 3: two
// lanes per word (two unique sectors). Return values are used, forbidding
// conversion of atom to red and preventing warp-aggregated atomics.
template<int Op>
__global__ void metric_domains(unsigned *data, unsigned *out, int pattern, int iterations) {
  unsigned t = threadIdx.x;
  unsigned index = pattern == 0 ? t : pattern == 1 ? 0 : pattern == 2 ? t * 32 : t / 2;
  unsigned *address = data + blockIdx.x * 1024 + index;
  unsigned sum = 0;
#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    unsigned value;
    if constexpr (Op == 0) {
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
      sum += value;
    } else if constexpr (Op == 1) {
      asm volatile("atom.global.add.u32 %0, [%1], 1;" : "=r"(value) : "l"(address) : "memory");
      sum += value;
    } else if constexpr (Op == 2) {
      asm volatile("red.global.add.u32 [%0], 1;" :: "l"(address) : "memory");
    } else {
      // Control-only launch: same allocation/initialization and output path,
      // but no data read/RMW. Useful for interpreting small-kernel DRAM noise.
      sum += t + static_cast<unsigned>(i);
      asm volatile("" : "+r"(sum));
    }
  }
  out[blockIdx.x * 32 + t] = sum;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: metric_domains read|atom|red|control PATTERN BLOCKS ITERATIONS\n");
    return 2;
  }
  int op = !std::strcmp(argv[1], "read") ? 0 : !std::strcmp(argv[1], "atom") ? 1
         : !std::strcmp(argv[1], "red") ? 2 : !std::strcmp(argv[1], "control") ? 3 : -1;
  int pattern = std::atoi(argv[2]), blocks = std::atoi(argv[3]), iterations = std::atoi(argv[4]);
  if (op < 0 || pattern < 0 || pattern > 3 || blocks < 1 || blocks > 4096
      || iterations < 1 || iterations > 4096) return 2;
  CUDA_OK(cudaSetDevice(0));
  std::vector<unsigned> input(blocks * 1024), expected, output(blocks * 32);
  for (size_t i = 0; i < input.size(); ++i) input[i] = unsigned(i * 1664525u + 1013904223u);
  expected = input;
  unsigned *data = nullptr, *out = nullptr;
  CUDA_OK(cudaMalloc(&data, input.size() * sizeof(unsigned)));
  CUDA_OK(cudaMalloc(&out, output.size() * sizeof(unsigned)));
  CUDA_OK(cudaMemcpy(data, input.data(), input.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  if (op == 0) metric_domains<0><<<blocks, 32>>>(data, out, pattern, iterations);
  if (op == 1) metric_domains<1><<<blocks, 32>>>(data, out, pattern, iterations);
  if (op == 2) metric_domains<2><<<blocks, 32>>>(data, out, pattern, iterations);
  if (op == 3) metric_domains<3><<<blocks, 32>>>(data, out, pattern, iterations);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(output.data(), out, output.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  if (op == 1 || op == 2) {
    for (int b = 0; b < blocks; ++b) for (unsigned t = 0; t < 32; ++t) {
      unsigned index = pattern == 0 ? t : pattern == 1 ? 0 : pattern == 2 ? t * 32 : t / 2;
      expected[b * 1024 + index] += iterations;
    }
    CUDA_OK(cudaMemcpy(input.data(), data, input.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (input != expected) { std::fprintf(stderr, "atomic/reduction result mismatch\n"); return 1; }
  } else {
    for (int b = 0; b < blocks; ++b) for (unsigned t = 0; t < 32; ++t) {
      unsigned index = pattern == 0 ? t : pattern == 1 ? 0 : pattern == 2 ? t * 32 : t / 2;
      unsigned want = 0;
      for (int i = 0; i < iterations; ++i) want += op == 0 ? input[b * 1024 + index] : t + unsigned(i);
      if (output[b * 32 + t] != want) { std::fprintf(stderr, "read/control result mismatch\n"); return 1; }
    }
  }
  CUDA_OK(cudaFree(out)); CUDA_OK(cudaFree(data));
  std::printf("correctness=pass op=%s pattern=%d blocks=%d iterations=%d\n", argv[1], pattern, blocks, iterations);
}
