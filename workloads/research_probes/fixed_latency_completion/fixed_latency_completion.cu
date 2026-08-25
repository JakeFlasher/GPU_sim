#include <cstdint>

constexpr int kWarpSize = 32;
constexpr int kFactorCount = 8;

#ifndef FIXED_COMPLETION_GENERATIONS
constexpr int kGenerations = 8192;
#else
constexpr int kGenerations = FIXED_COMPLETION_GENERATIONS;
#endif

struct Accumulators {
    std::uint32_t o0;
    std::uint32_t o1;
    std::uint32_t o2;
    std::uint32_t o3;
    std::uint32_t o4;
    std::uint32_t o5;
    std::uint32_t o6;
    std::uint32_t o7;
};

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

template <int Distance>
__device__ __forceinline__ void hot_body(Accumulators* outputs,
                                         const Factors& factors) {
    if constexpr (Distance == 1) {
        asm volatile(
            "mad.lo.u32 %0, %8, %9, %0;\n\t"
            "mad.lo.u32 %0, %10, %11, %0;\n\t"
            "mad.lo.u32 %0, %12, %13, %0;\n\t"
            "mad.lo.u32 %0, %14, %15, %0;\n\t"
            "mad.lo.u32 %0, %9, %10, %0;\n\t"
            "mad.lo.u32 %0, %11, %12, %0;\n\t"
            "mad.lo.u32 %0, %13, %14, %0;\n\t"
            "mad.lo.u32 %0, %15, %8, %0;\n\t"
            "mad.lo.u32 %0, %8, %10, %0;\n\t"
            "mad.lo.u32 %0, %11, %13, %0;\n\t"
            "mad.lo.u32 %0, %12, %14, %0;\n\t"
            "mad.lo.u32 %0, %15, %9, %0;\n\t"
            "mad.lo.u32 %0, %10, %12, %0;\n\t"
            "mad.lo.u32 %0, %13, %15, %0;\n\t"
            "mad.lo.u32 %0, %14, %8, %0;\n\t"
            "mad.lo.u32 %0, %9, %11, %0;"
            : "+r"(outputs->o0), "+r"(outputs->o1), "+r"(outputs->o2),
              "+r"(outputs->o3), "+r"(outputs->o4), "+r"(outputs->o5),
              "+r"(outputs->o6), "+r"(outputs->o7)
            : "r"(factors.a), "r"(factors.b), "r"(factors.c),
              "r"(factors.d), "r"(factors.e), "r"(factors.f),
              "r"(factors.g), "r"(factors.h));
    } else if constexpr (Distance == 2) {
        asm volatile(
            "mad.lo.u32 %0, %8, %9, %0;\n\t"
            "mad.lo.u32 %1, %10, %11, %1;\n\t"
            "mad.lo.u32 %0, %12, %13, %0;\n\t"
            "mad.lo.u32 %1, %14, %15, %1;\n\t"
            "mad.lo.u32 %0, %9, %10, %0;\n\t"
            "mad.lo.u32 %1, %11, %12, %1;\n\t"
            "mad.lo.u32 %0, %13, %14, %0;\n\t"
            "mad.lo.u32 %1, %15, %8, %1;\n\t"
            "mad.lo.u32 %0, %8, %10, %0;\n\t"
            "mad.lo.u32 %1, %11, %13, %1;\n\t"
            "mad.lo.u32 %0, %12, %14, %0;\n\t"
            "mad.lo.u32 %1, %15, %9, %1;\n\t"
            "mad.lo.u32 %0, %10, %12, %0;\n\t"
            "mad.lo.u32 %1, %13, %15, %1;\n\t"
            "mad.lo.u32 %0, %14, %8, %0;\n\t"
            "mad.lo.u32 %1, %9, %11, %1;"
            : "+r"(outputs->o0), "+r"(outputs->o1), "+r"(outputs->o2),
              "+r"(outputs->o3), "+r"(outputs->o4), "+r"(outputs->o5),
              "+r"(outputs->o6), "+r"(outputs->o7)
            : "r"(factors.a), "r"(factors.b), "r"(factors.c),
              "r"(factors.d), "r"(factors.e), "r"(factors.f),
              "r"(factors.g), "r"(factors.h));
    } else if constexpr (Distance == 4) {
        asm volatile(
            "mad.lo.u32 %0, %8, %9, %0;\n\t"
            "mad.lo.u32 %1, %10, %11, %1;\n\t"
            "mad.lo.u32 %2, %12, %13, %2;\n\t"
            "mad.lo.u32 %3, %14, %15, %3;\n\t"
            "mad.lo.u32 %0, %9, %10, %0;\n\t"
            "mad.lo.u32 %1, %11, %12, %1;\n\t"
            "mad.lo.u32 %2, %13, %14, %2;\n\t"
            "mad.lo.u32 %3, %15, %8, %3;\n\t"
            "mad.lo.u32 %0, %8, %10, %0;\n\t"
            "mad.lo.u32 %1, %11, %13, %1;\n\t"
            "mad.lo.u32 %2, %12, %14, %2;\n\t"
            "mad.lo.u32 %3, %15, %9, %3;\n\t"
            "mad.lo.u32 %0, %10, %12, %0;\n\t"
            "mad.lo.u32 %1, %13, %15, %1;\n\t"
            "mad.lo.u32 %2, %14, %8, %2;\n\t"
            "mad.lo.u32 %3, %9, %11, %3;"
            : "+r"(outputs->o0), "+r"(outputs->o1), "+r"(outputs->o2),
              "+r"(outputs->o3), "+r"(outputs->o4), "+r"(outputs->o5),
              "+r"(outputs->o6), "+r"(outputs->o7)
            : "r"(factors.a), "r"(factors.b), "r"(factors.c),
              "r"(factors.d), "r"(factors.e), "r"(factors.f),
              "r"(factors.g), "r"(factors.h));
    } else if constexpr (Distance == 8) {
        asm volatile(
            "mad.lo.u32 %0, %8, %9, %0;\n\t"
            "mad.lo.u32 %1, %10, %11, %1;\n\t"
            "mad.lo.u32 %2, %12, %13, %2;\n\t"
            "mad.lo.u32 %3, %14, %15, %3;\n\t"
            "mad.lo.u32 %4, %9, %10, %4;\n\t"
            "mad.lo.u32 %5, %11, %12, %5;\n\t"
            "mad.lo.u32 %6, %13, %14, %6;\n\t"
            "mad.lo.u32 %7, %15, %8, %7;\n\t"
            "mad.lo.u32 %0, %8, %10, %0;\n\t"
            "mad.lo.u32 %1, %11, %13, %1;\n\t"
            "mad.lo.u32 %2, %12, %14, %2;\n\t"
            "mad.lo.u32 %3, %15, %9, %3;\n\t"
            "mad.lo.u32 %4, %10, %12, %4;\n\t"
            "mad.lo.u32 %5, %13, %15, %5;\n\t"
            "mad.lo.u32 %6, %14, %8, %6;\n\t"
            "mad.lo.u32 %7, %9, %11, %7;"
            : "+r"(outputs->o0), "+r"(outputs->o1), "+r"(outputs->o2),
              "+r"(outputs->o3), "+r"(outputs->o4), "+r"(outputs->o5),
              "+r"(outputs->o6), "+r"(outputs->o7)
            : "r"(factors.a), "r"(factors.b), "r"(factors.c),
              "r"(factors.d), "r"(factors.e), "r"(factors.f),
              "r"(factors.g), "r"(factors.h));
    }
}

template <int Distance>
__device__ __forceinline__ void completion_kernel(
    unsigned long long* elapsed, std::uint32_t* sink,
    const std::uint32_t* factor_input) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int thread = warp * kWarpSize + lane;
    const std::uint32_t* input = factor_input + thread * kFactorCount;
    const Factors factors{input[0], input[1], input[2], input[3],
                          input[4], input[5], input[6], input[7]};
    const std::uint32_t seed =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    Accumulators outputs{
        seed ^ 0x01010101u, seed ^ 0x12121212u,
        seed ^ 0x23232323u, seed ^ 0x34343434u,
        seed ^ 0x45454545u, seed ^ 0x56565656u,
        seed ^ 0x67676767u, seed ^ 0x78787878u,
    };

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        if constexpr (Distance != 0) {
            hot_body<Distance>(&outputs, factors);
        } else {
            asm volatile("" : "+r"(outputs.o0), "+r"(outputs.o1),
                         "+r"(outputs.o2), "+r"(outputs.o3),
                         "+r"(outputs.o4), "+r"(outputs.o5),
                         "+r"(outputs.o6), "+r"(outputs.o7));
        }
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = outputs.o0 ^ (outputs.o1 << 1) ^ (outputs.o2 << 2) ^
                     (outputs.o3 << 3) ^ (outputs.o4 << 4) ^
                     (outputs.o5 << 5) ^ (outputs.o6 << 6) ^
                     (outputs.o7 << 7) ^ factors.a ^ (factors.b << 1) ^
                     (factors.c << 2) ^ (factors.d << 3) ^
                     (factors.e << 4) ^ (factors.f << 5) ^
                     (factors.g << 6) ^ (factors.h << 7);
    }
}

extern "C" __global__ void completion_d0(unsigned long long* elapsed,
                                           std::uint32_t* sink,
                                           const std::uint32_t* factors) {
    completion_kernel<0>(elapsed, sink, factors);
}

extern "C" __global__ void completion_d1(unsigned long long* elapsed,
                                           std::uint32_t* sink,
                                           const std::uint32_t* factors) {
    completion_kernel<1>(elapsed, sink, factors);
}

extern "C" __global__ void completion_d2(unsigned long long* elapsed,
                                           std::uint32_t* sink,
                                           const std::uint32_t* factors) {
    completion_kernel<2>(elapsed, sink, factors);
}

extern "C" __global__ void completion_d4(unsigned long long* elapsed,
                                           std::uint32_t* sink,
                                           const std::uint32_t* factors) {
    completion_kernel<4>(elapsed, sink, factors);
}

extern "C" __global__ void completion_d8(unsigned long long* elapsed,
                                           std::uint32_t* sink,
                                           const std::uint32_t* factors) {
    completion_kernel<8>(elapsed, sink, factors);
}
