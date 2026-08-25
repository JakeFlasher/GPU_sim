#include <cuda_runtime.h>

#include <cstdint>

#ifndef FIXED_RESULT_ORDER_GENERATIONS
constexpr int kResultOrderGenerations = 8192;
#else
constexpr int kResultOrderGenerations = FIXED_RESULT_ORDER_GENERATIONS;
#endif

template <bool HaddFirst>
__global__ void result_order(unsigned long long* elapsed,
                             std::uint32_t* sink,
                             const std::uint32_t* zero_inputs,
                             const std::uint32_t* one_inputs) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int thread = threadIdx.x;
    std::uint32_t zero = zero_inputs[thread];
    std::uint32_t one = one_inputs[thread];
    std::uint32_t a =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    std::uint32_t b =
        0xbb67ae85u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    asm volatile("" : "+r"(zero), "+r"(one), "+r"(a), "+r"(b));

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kResultOrderGenerations; ++generation) {
        std::uint32_t half_result;
        std::uint32_t float_result;
        asm volatile(
            "add.rn.f16x2 %0, %2, %3;\n\t"
            "fma.rn.f32 %1, %2, %4, %3;"
            : "=r"(half_result), "=r"(float_result)
            : "r"(a), "r"(zero), "r"(one)
            : "memory");
        if constexpr (HaddFirst) {
            // This consumer order makes ptxas retain HADD2 then FFMA while
            // both contributions cancel exactly from the two accumulators.
            asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                         : "+r"(a)
                         : "r"(float_result)
                         : "memory");
            asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                         : "+r"(b)
                         : "r"(half_result)
                         : "memory");
        } else {
            // Reversing the cancelled consumers makes ptxas retain FFMA then
            // HADD2 without changing the final values or instruction count.
            asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                         : "+r"(a)
                         : "r"(half_result)
                         : "memory");
            asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                         : "+r"(b)
                         : "r"(float_result)
                         : "memory");
        }
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = a ^ b;
    }
}

template __global__ void result_order<true>(unsigned long long*, std::uint32_t*,
                                             const std::uint32_t*,
                                             const std::uint32_t*);
template __global__ void result_order<false>(unsigned long long*, std::uint32_t*,
                                              const std::uint32_t*,
                                              const std::uint32_t*);
