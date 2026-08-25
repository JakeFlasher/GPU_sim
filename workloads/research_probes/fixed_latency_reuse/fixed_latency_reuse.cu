#include <cstdint>

constexpr int kWarpSize = 32;
constexpr int kFactorCount = 8;

#ifndef FIXED_REUSE_GENERATIONS
constexpr int kGenerations = 8192;
#else
constexpr int kGenerations = FIXED_REUSE_GENERATIONS;
#endif

struct Factors {
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
    std::uint32_t d;
    std::uint32_t e;
    std::uint32_t f;
    std::uint32_t g;
    std::uint32_t h;
};

struct Outputs {
    std::uint32_t o0;
    std::uint32_t o1;
    std::uint32_t o2;
    std::uint32_t o3;
    std::uint32_t o4;
    std::uint32_t o5;
    std::uint32_t o6;
    std::uint32_t o7;
    std::uint32_t o8;
    std::uint32_t o9;
    std::uint32_t o10;
    std::uint32_t o11;
    std::uint32_t o12;
    std::uint32_t o13;
    std::uint32_t o14;
    std::uint32_t o15;
};

template <int Protocol>
__device__ __forceinline__ void hot_body(Outputs* out, const Factors& f) {
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o0) : "r"(f.a), "r"(f.b));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o1) : "r"(f.c), "r"(f.d));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o2) : "r"(f.e), "r"(f.f));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o3) : "r"(f.g), "r"(f.h));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o4) : "r"(f.b), "r"(f.c));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o5) : "r"(f.d), "r"(f.e));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o6) : "r"(f.f), "r"(f.g));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o7) : "r"(f.h), "r"(f.a));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o8) : "r"(f.a), "r"(f.c));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o9) : "r"(f.d), "r"(f.f));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o10) : "r"(f.e), "r"(f.g));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o11) : "r"(f.h), "r"(f.b));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o12) : "r"(f.c), "r"(f.e));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o13) : "r"(f.f), "r"(f.h));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o14) : "r"(f.g), "r"(f.a));
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(out->o15) : "r"(f.b), "r"(f.d));
    asm volatile("" : : "n"(Protocol));
}

template <int Protocol>
__device__ __forceinline__ void reuse_kernel(
    unsigned long long* elapsed, std::uint32_t* sink,
    const std::uint32_t* factor_input) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int thread = warp * kWarpSize + lane;
    const std::uint32_t* input = factor_input + thread * kFactorCount;
    Factors factors{input[0], input[1], input[2], input[3],
                    input[4], input[5], input[6], input[7]};
    const std::uint32_t seed =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    Outputs outputs{
        seed ^ 0x01010101u, seed ^ 0x12121212u,
        seed ^ 0x23232323u, seed ^ 0x34343434u,
        seed ^ 0x45454545u, seed ^ 0x56565656u,
        seed ^ 0x67676767u, seed ^ 0x78787878u,
        seed ^ 0x89898989u, seed ^ 0x9a9a9a9au,
        seed ^ 0xababababu, seed ^ 0xbcbcbcbcu,
        seed ^ 0xcdcdcdcdu, seed ^ 0xdedededeu,
        seed ^ 0xefefefefu, seed ^ 0xf0f0f0f0u,
    };

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        hot_body<Protocol>(&outputs, factors);
    }
    const unsigned long long stop = clock64();
    asm volatile("" : "+r"(factors.a), "+r"(factors.b), "+r"(factors.c),
                 "+r"(factors.d), "+r"(factors.e), "+r"(factors.f),
                 "+r"(factors.g), "+r"(factors.h));

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = outputs.o0 ^ (outputs.o1 << 1) ^ (outputs.o2 << 2) ^
                     (outputs.o3 << 3) ^ (outputs.o4 << 4) ^
                     (outputs.o5 << 5) ^ (outputs.o6 << 6) ^
                     (outputs.o7 << 7) ^ (outputs.o8 << 8) ^
                     (outputs.o9 << 9) ^ (outputs.o10 << 10) ^
                     (outputs.o11 << 11) ^ (outputs.o12 << 12) ^
                     (outputs.o13 << 13) ^ (outputs.o14 << 14) ^
                     (outputs.o15 << 15) ^ factors.a ^ (factors.b << 1) ^
                     (factors.c << 2) ^ (factors.d << 3) ^
                     (factors.e << 4) ^ (factors.f << 5) ^
                     (factors.g << 6) ^ (factors.h << 7);
    }
}

#define DEFINE_REUSE_KERNEL(name, protocol)                                     \
    extern "C" __global__ void name(unsigned long long* elapsed,               \
                                      std::uint32_t* sink,                       \
                                      const std::uint32_t* factors) {            \
        reuse_kernel<protocol>(elapsed, sink, factors);                          \
    }

DEFINE_REUSE_KERNEL(reuse_u10, 10)
DEFINE_REUSE_KERNEL(reuse_u11, 11)
DEFINE_REUSE_KERNEL(reuse_u12, 12)
DEFINE_REUSE_KERNEL(reuse_u20, 20)
DEFINE_REUSE_KERNEL(reuse_u22, 22)
