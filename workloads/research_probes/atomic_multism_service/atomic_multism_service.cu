#include <cuda_runtime.h>

#include <cstdint>

#ifndef ATOMIC_RETURNING
#define ATOMIC_RETURNING 1
#endif

#ifndef ATOMIC_GENERATIONS
#define ATOMIC_GENERATIONS 1024
#endif

__global__ void atomic_multism_probe(unsigned long long* elapsed,
                                     unsigned long long* returned_sum,
                                     std::uint32_t* data,
                                     const std::uint32_t* indices) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t index =
        reinterpret_cast<const volatile std::uint32_t*>(indices)[global_thread];
    __syncthreads();

    unsigned long long checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < ATOMIC_GENERATIONS; ++generation) {
        if constexpr (ATOMIC_RETURNING != 0) {
            const std::uint32_t prior = atomicAdd(data + index, 1u);
            checksum += prior;
        } else {
            atomicAdd(data + index, 1u);
        }
    }
    const unsigned long long stop = clock64();
    returned_sum[global_thread] = checksum;
    if (threadIdx.x == 0) {
        elapsed[blockIdx.x] = stop - start;
    }
}
