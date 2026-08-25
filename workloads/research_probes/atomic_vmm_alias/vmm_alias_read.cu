#include <cuda_runtime.h>

#include <cstdint>

constexpr int kGenerations = 64;

extern "C" __global__ void vmm_alias_read_probe(
    unsigned long long* elapsed,
    unsigned long long* returned_sum,
    const unsigned long long* addresses) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const auto address = reinterpret_cast<const std::uint32_t*>(
        addresses[global_thread]);
    __syncthreads();

    unsigned long long checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        checksum += __ldcg(address);
    }
    const unsigned long long stop = clock64();
    returned_sum[global_thread] = checksum;
    if (threadIdx.x == 0) {
        elapsed[blockIdx.x] = stop - start;
    }
}
