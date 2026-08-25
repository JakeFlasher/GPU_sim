#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#ifndef ATOMIC_RETURNING
#define ATOMIC_RETURNING 1
#endif

#ifndef ATOMIC_WIDTH
#define ATOMIC_WIDTH 32
#endif

#ifndef ATOMIC_GENERATIONS
#define ATOMIC_GENERATIONS 1024
#endif

static_assert(ATOMIC_WIDTH == 32 || ATOMIC_WIDTH == 64);

using AtomicWord =
    std::conditional_t<ATOMIC_WIDTH == 32, std::uint32_t, unsigned long long>;

__global__ void atomic_alias_probe(unsigned long long* elapsed,
                                   unsigned long long* returned_sum,
                                   AtomicWord* data,
                                   const std::uint32_t* indices) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const std::uint32_t index =
        reinterpret_cast<const volatile std::uint32_t*>(indices)[threadIdx.x];
    __syncthreads();

    unsigned long long checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < ATOMIC_GENERATIONS; ++generation) {
        if constexpr (ATOMIC_RETURNING != 0) {
            const AtomicWord prior = atomicAdd(data + index, AtomicWord{1});
            checksum += static_cast<unsigned long long>(prior);
        } else {
            atomicAdd(data + index, AtomicWord{1});
        }
    }
    const unsigned long long stop = clock64();
    returned_sum[threadIdx.x] = checksum;
    if (lane == 0) {
        elapsed[warp] = stop - start;
    }
}
