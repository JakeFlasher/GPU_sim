#include <cuda_runtime.h>

#include <cstdint>

#ifndef RESULT_BYPASS_GENERATIONS
constexpr int kBypassGenerations = 1025;
#else
constexpr int kBypassGenerations = RESULT_BYPASS_GENERATIONS;
#endif

__global__ void bypass_fixed_search(unsigned long long* elapsed,
                                    std::uint32_t* sink,
                                    const std::uint32_t* zeros) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    std::uint32_t zero = zeros[threadIdx.x];
    std::uint32_t value = zero;
    std::uint32_t consumed = 0;
    std::uint32_t checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kBypassGenerations; ++generation) {
        asm volatile("xor.b32 %0, %0, 0x4;"
                     : "+r"(value)
                     :
                     : "memory");
        asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
        asm volatile("xor.b32 %0, %1, %2;"
                     : "=r"(consumed)
                     : "r"(value), "r"(zero)
                     : "memory");
        asm volatile("add.u32 %0, %0, %1;"
                     : "+r"(checksum)
                     : "r"(consumed)
                     : "memory");
    }
    const unsigned long long stop = clock64();
    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = checksum;
    }
}

__global__ void bypass_global_search(unsigned long long* elapsed,
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
    const std::uint32_t zero = zeros[threadIdx.x];
    std::uint32_t final_hi = 0;
    std::uint32_t loaded = 0;
    std::uint32_t checksum = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kBypassGenerations; ++generation) {
        unsigned long long address;
        asm volatile("add.u32 %0, %1, %2;"
                     : "=r"(final_hi)
                     : "r"(hi), "r"(zero)
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
    }
    const unsigned long long stop = clock64();
    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = checksum;
    }
}

// Single-shot form of the paper's Listing 3. CUDA combines the two zero-add
// copies into one fixed IADD3 whose result is the high half of the following
// global-load address. Patched stall 4 must fault; stall 5 must load correctly.
__global__ void bypass_global_canary(std::uint32_t* sink,
                                    const std::uint32_t* inputs,
                                    const std::uint32_t* zeros) {
    std::uint32_t lo;
    std::uint32_t hi;
    std::uint32_t temporary_hi;
    std::uint32_t final_hi;
    unsigned long long address;
    std::uint32_t loaded;
    const std::uint32_t zero = zeros[threadIdx.x];
    asm volatile("mov.b64 {%0, %1}, %2;"
                 : "=r"(lo), "=r"(hi)
                 : "l"(inputs)
                 : "memory");
    asm volatile("add.u32 %0, %1, %2;"
                 : "=r"(temporary_hi)
                 : "r"(hi), "r"(zero)
                 : "memory");
    asm volatile("add.u32 %0, %1, %2;"
                 : "=r"(final_hi)
                 : "r"(temporary_hi), "r"(zero)
                 : "memory");
    asm volatile("mov.b64 %0, {%1, %2};"
                 : "=l"(address)
                 : "r"(lo), "r"(final_hi)
                 : "memory");
    asm volatile("ld.global.u32 %0, [%1];"
                 : "=r"(loaded)
                 : "l"(address)
                 : "memory");
    sink[threadIdx.x] = loaded;
}
