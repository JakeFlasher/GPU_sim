#include <cuda_runtime.h>

#include <cstdint>

extern "C" __global__ void atomic_lts_mapping_clear(
    std::uint32_t* data, const std::uint32_t* indices) {
    const int entry = threadIdx.x;
    if (entry < 49) {
        data[indices[entry * 32]] = 0;
    }
}

extern "C" __global__ void atomic_lts_mapping_probe(
    unsigned long long* elapsed,
    unsigned long long* returned_sum,
    std::uint32_t* data,
    const std::uint32_t* indices,
    const std::uint32_t* generation_limits) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t index =
        reinterpret_cast<const volatile std::uint32_t*>(indices)[global_thread];
    const std::uint32_t limit = generation_limits[blockIdx.x];
    __syncthreads();

    const unsigned long long start = clock64();
#pragma unroll 1
    for (std::uint32_t generation = 0; generation < limit; ++generation) {
        atomicAdd(data + index, 1u);
    }
    const unsigned long long stop = clock64();
    returned_sum[global_thread] = 0;
    if (threadIdx.x == 0) {
        elapsed[blockIdx.x] = stop - start;
    }
}
