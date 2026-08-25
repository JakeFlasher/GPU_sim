#include <cuda_runtime.h>

#include <cstdint>

#ifndef FIXED_CAPACITY_GENERATIONS
constexpr int kCapacityGenerations = 8192;
#else
constexpr int kCapacityGenerations = FIXED_CAPACITY_GENERATIONS;
#endif

__global__ void result_capacity_search(unsigned long long* elapsed,
                                       std::uint32_t* sink,
                                       const std::uint32_t* inputs) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int base = threadIdx.x * 9;
    std::uint32_t h0 = inputs[base];
    std::uint32_t h1 = inputs[base + 1];
    std::uint32_t h2 = inputs[base + 2];
    std::uint32_t h3 = inputs[base + 3];
    std::uint32_t h4 = inputs[base + 4];
    std::uint32_t f0 = inputs[base + 5];
    std::uint32_t f1 = inputs[base + 6];
    std::uint32_t f2 = inputs[base + 7];
    std::uint32_t f3 = inputs[base + 8];
    std::uint32_t acc = 0x6a09e667u ^ static_cast<std::uint32_t>(threadIdx.x);
    std::uint32_t seed = 0x3f800000u;
    float td0 = 0.0f, td1 = 0.0f, td2 = 0.0f, td3 = 0.0f;
    asm volatile("" : "+r"(h0), "+r"(h1), "+r"(h2), "+r"(h3), "+r"(h4));
    asm volatile("" : "+r"(f0), "+r"(f1), "+r"(f2), "+r"(f3), "+r"(acc), "+r"(seed));

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kCapacityGenerations; ++generation) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(td0), "+f"(td1), "+f"(td2), "+f"(td3)
            : "r"(h0), "r"(h1), "r"(h2), "r"(h3), "r"(f0), "r"(f1));
        seed += 1;
        std::uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8;
        asm volatile(
            "add.rn.f16x2 %0, %9, %10;\n\t"
            "fma.rn.f32 %1, %9, %15, %10;\n\t"
            "add.rn.f16x2 %2, %9, %11;\n\t"
            "fma.rn.f32 %3, %9, %16, %11;\n\t"
            "add.rn.f16x2 %4, %9, %12;\n\t"
            "fma.rn.f32 %5, %9, %17, %12;\n\t"
            "add.rn.f16x2 %6, %9, %13;\n\t"
            "fma.rn.f32 %7, %9, %18, %13;\n\t"
            "add.rn.f16x2 %8, %9, %14;"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4),
              "=r"(r5), "=r"(r6), "=r"(r7), "=r"(r8)
            : "r"(seed), "r"(h0), "r"(h1), "r"(h2), "r"(h3), "r"(h4),
              "r"(f0), "r"(f1), "r"(f2), "r"(f3)
            : "memory");
        asm volatile("bar.sync 0;" : : : "memory");
        std::uint32_t token;
        asm volatile("lop3.b32 %0, %1, %2, %2, 0x96;"
                     : "=r"(token) : "r"(r8), "r"(r0) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r1) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r2) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r3) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r4) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r5) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r6) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(token) : "r"(r7) : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(acc) : "r"(token) : "memory");
    }
    const unsigned long long stop = clock64();
    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = acc ^ __float_as_uint(td0) ^ __float_as_uint(td1)
                          ^ __float_as_uint(td2) ^ __float_as_uint(td3);
    }
}
