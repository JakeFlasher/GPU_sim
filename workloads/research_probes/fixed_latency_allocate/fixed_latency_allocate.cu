#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 16;
constexpr int kHotInstructions = 16;

#ifndef FIXED_ALLOCATE_GENERATIONS
constexpr int kGenerations = 8192;
#else
constexpr int kGenerations = FIXED_ALLOCATE_GENERATIONS;
#endif

enum Protocol : int {
    kControl = 0,
    kParityTriplets = 1,
    kAdjacentTriplets = 2,
    kStableTriplet = 3,
    kRotatedTriplet = 4,
    kSameBankTriplets = 5,
};

struct Registers {
    std::uint32_t o0;
    std::uint32_t o1;
    std::uint32_t o2;
    std::uint32_t o3;
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
    std::uint32_t d;
    std::uint32_t e;
    std::uint32_t f;
};

template <int Layout>
__device__ __forceinline__ void imad_group(Registers* registers) {
    if constexpr (Layout == kParityTriplets) {
        asm volatile(
            "mad.lo.u32 %0, %4, %6, %0;\n\t"
            "mad.lo.u32 %1, %5, %7, %1;\n\t"
            "mad.lo.u32 %2, %8, %4, %2;\n\t"
            "mad.lo.u32 %3, %9, %5, %3;"
            : "+r"(registers->o0), "+r"(registers->o1),
              "+r"(registers->o2), "+r"(registers->o3)
            : "r"(registers->a), "r"(registers->b),
              "r"(registers->c), "r"(registers->d),
              "r"(registers->e), "r"(registers->f));
    } else if constexpr (Layout == kAdjacentTriplets) {
        asm volatile(
            "mad.lo.u32 %0, %4, %5, %0;\n\t"
            "mad.lo.u32 %1, %6, %7, %1;\n\t"
            "mad.lo.u32 %2, %8, %9, %2;\n\t"
            "mad.lo.u32 %3, %5, %6, %3;"
            : "+r"(registers->o0), "+r"(registers->o1),
              "+r"(registers->o2), "+r"(registers->o3)
            : "r"(registers->a), "r"(registers->b),
              "r"(registers->c), "r"(registers->d),
              "r"(registers->e), "r"(registers->f));
    } else if constexpr (Layout == kStableTriplet) {
        asm volatile(
            "mad.lo.u32 %0, %4, %6, %0;\n\t"
            "mad.lo.u32 %1, %4, %6, %1;\n\t"
            "mad.lo.u32 %2, %4, %6, %2;\n\t"
            "mad.lo.u32 %3, %4, %6, %3;"
            : "+r"(registers->o0), "+r"(registers->o1),
              "+r"(registers->o2), "+r"(registers->o3)
            : "r"(registers->a), "r"(registers->b),
              "r"(registers->c), "r"(registers->d),
              "r"(registers->e), "r"(registers->f));
    } else if constexpr (Layout == kRotatedTriplet) {
        asm volatile(
            "mad.lo.u32 %0, %4, %6, %0;\n\t"
            "mad.lo.u32 %1, %6, %1, %4;\n\t"
            "mad.lo.u32 %2, %2, %4, %6;\n\t"
            "mad.lo.u32 %3, %4, %3, %6;"
            : "+r"(registers->o0), "+r"(registers->o1),
              "+r"(registers->o2), "+r"(registers->o3)
            : "r"(registers->a), "r"(registers->b),
              "r"(registers->c), "r"(registers->d),
              "r"(registers->e), "r"(registers->f));
    } else if constexpr (Layout == kSameBankTriplets) {
        asm volatile(
            "mad.lo.u32 %0, %4, %6, %0;\n\t"
            "mad.lo.u32 %1, %5, %7, %1;\n\t"
            "mad.lo.u32 %2, %4, %8, %2;\n\t"
            "mad.lo.u32 %3, %6, %8, %3;"
            : "+r"(registers->o0), "+r"(registers->o1),
              "+r"(registers->o2), "+r"(registers->o3)
            : "r"(registers->a), "r"(registers->b),
              "r"(registers->c), "r"(registers->d),
              "r"(registers->e), "r"(registers->f));
    }
}

template <int Layout>
__device__ __forceinline__ void hot_body(Registers* registers) {
#pragma unroll
    for (int group = 0; group < kHotInstructions / 4; ++group) {
        imad_group<Layout>(registers);
    }
}

template <int ProtocolValue>
__global__ void fixed_latency_allocate(unsigned long long* elapsed,
                                       std::uint32_t* sink) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const std::uint32_t seed =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    Registers registers{
        seed ^ 0x01010101u,
        seed ^ 0x12121212u,
        seed ^ 0x23232323u,
        seed ^ 0x34343434u,
        seed ^ 0x45454545u,
        seed ^ 0x56565656u,
        seed ^ 0x67676767u,
        seed ^ 0x78787878u,
        seed ^ 0x89898989u,
        seed ^ 0x9a9a9a9au,
    };

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        if constexpr (ProtocolValue == kControl) {
            asm volatile("" : "+r"(registers.o0), "+r"(registers.o1),
                         "+r"(registers.o2), "+r"(registers.o3));
        } else {
            hot_body<ProtocolValue>(&registers);
        }
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = registers.o0 ^ (registers.o1 << 1) ^
                     (registers.o2 << 2) ^ (registers.o3 << 3) ^
                     (registers.a << 4) ^ (registers.b << 5) ^
                     (registers.c << 6) ^ (registers.d << 7) ^
                     (registers.e << 8) ^ (registers.f << 9);
    }
}

template <int ProtocolValue>
void launch_kernel(int warps, unsigned long long* elapsed, std::uint32_t* sink) {
    fixed_latency_allocate<ProtocolValue><<<1, warps * kWarpSize>>>(elapsed, sink);
}

void launch_protocol(int protocol, int warps, unsigned long long* elapsed,
                     std::uint32_t* sink) {
    switch (protocol) {
        case kControl: launch_kernel<kControl>(warps, elapsed, sink); break;
        case kParityTriplets:
            launch_kernel<kParityTriplets>(warps, elapsed, sink);
            break;
        case kAdjacentTriplets:
            launch_kernel<kAdjacentTriplets>(warps, elapsed, sink);
            break;
        case kStableTriplet:
            launch_kernel<kStableTriplet>(warps, elapsed, sink);
            break;
        case kRotatedTriplet:
            launch_kernel<kRotatedTriplet>(warps, elapsed, sink);
            break;
        case kSameBankTriplets:
            launch_kernel<kSameBankTriplets>(warps, elapsed, sink);
            break;
        default: std::abort();
    }
}

const char* protocol_name(int protocol) {
    static constexpr const char* names[] = {
        "r00_control",
        "r10_parity_triplets",
        "r11_adjacent_triplets",
        "r20_stable_triplet",
        "r21_rotated_triplet",
        "r12_same_bank_triplets",
    };
    return protocol >= 0 && protocol < 6 ? names[protocol] : nullptr;
}

bool parse_protocol(const char* text, int* protocol) {
    static constexpr const char* short_names[] = {
        "r00", "r10", "r11", "r20", "r21", "r12",
    };
    for (int index = 0; index < 6; ++index) {
        if (std::strcmp(text, short_names[index]) == 0 ||
            std::strcmp(text, protocol_name(index)) == 0) {
            *protocol = index;
            return true;
        }
    }
    return false;
}

std::uint32_t mad(std::uint32_t lhs, std::uint32_t rhs,
                  std::uint32_t addend) {
    return lhs * rhs + addend;
}

void host_group(int protocol, Registers* registers) {
    std::uint32_t n0;
    std::uint32_t n1;
    std::uint32_t n2;
    std::uint32_t n3;
    if (protocol == kParityTriplets) {
        n0 = mad(registers->a, registers->c, registers->o0);
        n1 = mad(registers->b, registers->d, registers->o1);
        n2 = mad(registers->e, registers->a, registers->o2);
        n3 = mad(registers->f, registers->b, registers->o3);
    } else if (protocol == kAdjacentTriplets) {
        n0 = mad(registers->a, registers->b, registers->o0);
        n1 = mad(registers->c, registers->d, registers->o1);
        n2 = mad(registers->e, registers->f, registers->o2);
        n3 = mad(registers->b, registers->c, registers->o3);
    } else if (protocol == kStableTriplet) {
        n0 = mad(registers->a, registers->c, registers->o0);
        n1 = mad(registers->a, registers->c, registers->o1);
        n2 = mad(registers->a, registers->c, registers->o2);
        n3 = mad(registers->a, registers->c, registers->o3);
    } else if (protocol == kRotatedTriplet) {
        n0 = mad(registers->a, registers->c, registers->o0);
        n1 = mad(registers->c, registers->o1, registers->a);
        n2 = mad(registers->o2, registers->a, registers->c);
        n3 = mad(registers->a, registers->o3, registers->c);
    } else {
        n0 = mad(registers->a, registers->c, registers->o0);
        n1 = mad(registers->b, registers->d, registers->o1);
        n2 = mad(registers->a, registers->e, registers->o2);
        n3 = mad(registers->c, registers->e, registers->o3);
    }
    registers->o0 = n0;
    registers->o1 = n1;
    registers->o2 = n2;
    registers->o3 = n3;
}

std::uint32_t expected(int protocol, int warp, int lane) {
    const std::uint32_t seed =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    Registers registers{
        seed ^ 0x01010101u,
        seed ^ 0x12121212u,
        seed ^ 0x23232323u,
        seed ^ 0x34343434u,
        seed ^ 0x45454545u,
        seed ^ 0x56565656u,
        seed ^ 0x67676767u,
        seed ^ 0x78787878u,
        seed ^ 0x89898989u,
        seed ^ 0x9a9a9a9au,
    };
    if (protocol != kControl) {
        for (int generation = 0; generation < kGenerations; ++generation) {
            for (int group = 0; group < kHotInstructions / 4; ++group) {
                host_group(protocol, &registers);
            }
        }
    }
    return registers.o0 ^ (registers.o1 << 1) ^
           (registers.o2 << 2) ^ (registers.o3 << 3) ^
           (registers.a << 4) ^ (registers.b << 5) ^
           (registers.c << 6) ^ (registers.d << 7) ^
           (registers.e << 8) ^ (registers.f << 9);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s PROTOCOL WARPS_4_8_12_16 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    int protocol = -1;
    const int warps = std::atoi(argv[2]);
    const int trials = argc == 4 ? std::atoi(argv[3]) : 7;
    if (!parse_protocol(argv[1], &protocol) ||
        (warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        trials < 1 || trials > 32) {
        return 2;
    }

    unsigned long long* elapsed = nullptr;
    std::uint32_t* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed, kMaxWarps * sizeof(*elapsed)));
    CUDA_CHECK(cudaMalloc(&sink, kMaxWarps * sizeof(*sink)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaProfilerStart());
        launch_protocol(protocol, warps, elapsed, sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(samples[trial].data(), elapsed,
                              warps * sizeof(*elapsed),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<std::uint32_t> values(warps);
    CUDA_CHECK(cudaMemcpy(values.data(), sink, warps * sizeof(*sink),
                          cudaMemcpyDeviceToHost));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        correct = correct && values[warp] == expected(protocol, warp, 0);
        checksum = checksum * 131u + values[warp];
    }

    std::printf(
        "{\"protocol\":\"%s\",\"warps\":%d,"
        "\"warps_per_modeled_smsp\":%d,\"generations\":%d,"
        "\"hot_instructions_per_generation\":%d,\"samples_cycles\":[",
        protocol_name(protocol), warps, warps / 4, kGenerations,
        protocol == kControl ? 0 : kHotInstructions);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf(
        "],\"checksum\":%llu,\"correct\":%s}\n",
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(elapsed));
    return correct ? 0 : 1;
}
