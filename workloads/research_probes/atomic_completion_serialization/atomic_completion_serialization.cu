#include <cuda_runtime.h>

#include <cstdint>

#ifndef ATOMIC_VARIANT
#define ATOMIC_VARIANT 0
#endif

#ifndef ATOMIC_GENERATIONS
#define ATOMIC_GENERATIONS 1024
#endif

constexpr int kAtomicVariant = ATOMIC_VARIANT;
constexpr int kAtomicGenerations = ATOMIC_GENERATIONS;

static_assert(kAtomicVariant >= 0 && kAtomicVariant < 8);

__global__ void atomic_probe(unsigned long long* elapsed,
                             unsigned long long* returned_sum,
                             std::uint32_t* data) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    constexpr bool returning = kAtomicVariant < 4;
    constexpr int layout = kAtomicVariant & 3;
    int index;
    if constexpr (layout == 0) {
        index = (warp * 32 + lane) * 32;
    } else if constexpr (layout == 1) {
        index = warp * 32 + lane;
    } else if constexpr (layout == 2) {
        index = warp * 32;
    } else {
        index = 0;
    }

    unsigned long long checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kAtomicGenerations; ++generation) {
        if constexpr (returning) {
            const std::uint32_t prior = atomicAdd(data + index, 1u);
            checksum += prior;
        } else {
            atomicAdd(data + index, 1u);
        }
    }
    const unsigned long long stop = clock64();
    returned_sum[threadIdx.x] = checksum;
    if (lane == 0) {
        elapsed[warp] = stop - start;
    }
}
