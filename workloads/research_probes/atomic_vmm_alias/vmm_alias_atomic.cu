#include <cuda_runtime.h>

#include <cstdint>

extern "C" __global__ void atomic_vmm_mapping_clear(
    const unsigned long long* addresses) {
    const int entry = threadIdx.x;
    if (entry < 49) {
        *reinterpret_cast<std::uint32_t*>(addresses[entry * 32]) = 0;
    }
}

extern "C" __global__ void atomic_vmm_mapping_probe(
    unsigned long long* elapsed,
    unsigned long long* returned_sum,
    const unsigned long long* addresses,
    const std::uint32_t* generation_limits) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    auto address =
        reinterpret_cast<std::uint32_t*>(addresses[global_thread]);
    const std::uint32_t limit = generation_limits[blockIdx.x];
    __syncthreads();

    const unsigned long long start = clock64();
#pragma unroll 1
    for (std::uint32_t generation = 0; generation < limit; ++generation) {
        asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;"
                     :
                     : "l"(address), "r"(1u)
                     : "memory");
    }
    const unsigned long long stop = clock64();
    returned_sum[global_thread] = 0;
    if (threadIdx.x == 0) {
        elapsed[blockIdx.x] = stop - start;
    }
}
