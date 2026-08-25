#include <cuda_runtime.h>

#include <cstdint>

#ifndef FIXED_SAME_BANK_GENERATIONS
constexpr int kSameBankGenerations = 8192;
#else
constexpr int kSameBankGenerations = FIXED_SAME_BANK_GENERATIONS;
#endif

template <bool HaddFirst>
__global__ void result_same_bank(unsigned long long* elapsed,
                                 std::uint32_t* sink,
                                 const std::uint32_t* zero_inputs,
                                 const std::uint32_t* one_inputs) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    std::uint32_t zero = zero_inputs[threadIdx.x];
    std::uint32_t one = one_inputs[threadIdx.x];
    std::uint32_t a = 0x3f800000u;
    std::uint32_t b = 0x3f800000u;
    std::uint32_t pad0 = 0x243f6a88u ^ static_cast<std::uint32_t>(threadIdx.x);
    std::uint32_t pad1 = 0x85a308d3u ^ static_cast<std::uint32_t>(threadIdx.x);
    asm volatile("" : "+r"(zero), "+r"(one), "+r"(a), "+r"(b));
    asm volatile("" : "+r"(pad0), "+r"(pad1));

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kSameBankGenerations; ++generation) {
        if constexpr (HaddFirst) {
            std::uint32_t half_result;
            asm volatile(
                "add.rn.f16x2 %0, %2, %3;\n\t"
                "fma.rn.f32 %1, %2, %4, %3;"
                : "=r"(half_result), "+r"(a)
                : "r"(b), "r"(zero), "r"(one)
                : "memory");
            asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                         : "+r"(b)
                         : "r"(half_result)
                         : "memory");
            asm volatile("lop3.b32 %0, %0, %0, %0, 0x96;"
                         : "+r"(a)
                         :
                         : "memory");
        } else {
            std::uint32_t float_result;
            asm volatile(
                "fma.rn.f32 %1, %2, %4, %3;\n\t"
                "add.rn.f16x2 %0, %2, %3;"
                : "+r"(b), "=r"(float_result)
                : "r"(a), "r"(zero), "r"(one)
                : "memory");
            asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                         : "+r"(a)
                         : "r"(float_result)
                         : "memory");
            asm volatile("lop3.b32 %0, %0, %0, %0, 0x96;"
                         : "+r"(b)
                         :
                         : "memory");
        }
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = a ^ b ^ pad0 ^ pad1;
    }
}

template __global__ void result_same_bank<false>(unsigned long long*,
                                                  std::uint32_t*,
                                                  const std::uint32_t*,
                                                  const std::uint32_t*);
template __global__ void result_same_bank<true>(unsigned long long*,
                                                 std::uint32_t*,
                                                 const std::uint32_t*,
                                                 const std::uint32_t*);
