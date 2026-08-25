#include <cuda_runtime.h>

#include <cstdint>

#ifndef FRONT_END_GENERATIONS
#define FRONT_END_GENERATIONS 1024
#endif

#ifndef FRONT_END_UNROLL
#define FRONT_END_UNROLL 1
#endif

constexpr int kFrontEndGenerations = FRONT_END_GENERATIONS;
constexpr int kFrontEndUnroll = FRONT_END_UNROLL;

static_assert(kFrontEndUnroll == 1 || kFrontEndUnroll == 2 ||
              kFrontEndUnroll == 4 || kFrontEndUnroll == 8);
static_assert(kFrontEndGenerations % kFrontEndUnroll == 0);

// Each compiled U variant executes exactly kFrontEndGenerations copies of the
// measured body.  Only the number of copies between backward branches changes.
// The volatile PTX blocks and memory clobbers retain the same dataflow used by
// fixed_latency_result_bypass.
__global__ void footprint_fixed_search(unsigned long long* elapsed,
                                       std::uint32_t* sink,
                                       const std::uint32_t* zeros) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    std::uint32_t rolling_zero = zeros[threadIdx.x];
    std::uint32_t value = rolling_zero;
    std::uint32_t consumed = 0;
    std::uint32_t checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int group = 0; group < kFrontEndGenerations / kFrontEndUnroll;
         ++group) {
#pragma unroll
        for (int copy = 0; copy < kFrontEndUnroll; ++copy) {
            // Feeding the prior checksum back into the next producer prevents
            // ptxas from algebraically folding adjacent unrolled bodies.
            asm volatile("add.u32 %0, %1, 4;"
                         : "=r"(value)
                         : "r"(checksum)
                         : "memory");
            asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
            asm volatile("xor.b32 %0, %1, %2;"
                         : "=r"(consumed)
                         : "r"(value), "r"(rolling_zero)
                         : "memory");
            asm volatile("mad.lo.u32 %0, %0, 1664524, %1;"
                         : "+r"(checksum)
                         : "r"(consumed)
                         : "memory");
            asm volatile("and.b32 %0, %0, %1;"
                         : "+r"(rolling_zero)
                         : "r"(consumed)
                         : "memory");
        }
    }
    const unsigned long long stop = clock64();
    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = checksum;
    }
}

__global__ void footprint_global_search(unsigned long long* elapsed,
                                        std::uint32_t* sink,
                                        const std::uint32_t* inputs,
                                        const std::uint32_t* zeros) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    std::uint32_t lo;
    std::uint32_t hi;
    asm volatile("mov.b64 {%0, %1}, %2;"
                 : "=r"(lo), "=r"(hi)
                 : "l"(inputs + threadIdx.x)
                 : "memory");
    std::uint32_t rolling_zero = zeros[threadIdx.x];
    std::uint32_t final_hi = 0;
    std::uint32_t loaded = 0;
    std::uint32_t checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int group = 0; group < kFrontEndGenerations / kFrontEndUnroll;
         ++group) {
#pragma unroll
        for (int copy = 0; copy < kFrontEndUnroll; ++copy) {
            unsigned long long address;
            asm volatile("add.u32 %0, %1, %2;"
                         : "=r"(final_hi)
                         : "r"(hi), "r"(rolling_zero)
                         : "memory");
            asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
            asm volatile("mov.b64 %0, {%1, %2};"
                         : "=l"(address)
                         : "r"(lo), "r"(final_hi)
                         : "memory");
            asm volatile("ld.global.u32 %0, [%1];"
                         : "=r"(loaded)
                         : "l"(address)
                         : "memory");
            asm volatile("add.u32 %0, %0, %1;"
                         : "+r"(checksum)
                         : "r"(loaded)
                         : "memory");
            // The value is zero in the measured workload, but ptxas cannot
            // prove that.  This loop-carried dependency prevents load hoisting
            // while retaining the same prewarmed address on hardware.
            asm volatile("and.b32 %0, %0, %1;"
                         : "+r"(rolling_zero)
                         : "r"(loaded)
                         : "memory");
        }
    }
    const unsigned long long stop = clock64();
    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = checksum;
    }
}
