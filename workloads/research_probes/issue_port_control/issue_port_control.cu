#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cmath>
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
constexpr int kMaxWarps = 8;
constexpr std::uint32_t kAdd = 0x3c6ef35fu;
constexpr std::uint32_t kMul = 1664525u;
constexpr std::uint32_t kMadAdd = 1013904223u;
constexpr std::uint32_t kLogicA = 0xa5a5a5a5u;
constexpr std::uint32_t kLogicB = 0x3c3c3c3cu;
constexpr float kFmaMul = 1.00000011920928955078125f;
constexpr float kFmaAdd = 0.00000095367431640625f;

#ifndef ISSUE_PORT_GENERATIONS
constexpr int kGenerations = 8192;
#else
constexpr int kGenerations = ISSUE_PORT_GENERATIONS;
#endif

enum Protocol : int {
    kControl = 0,
    kDependentIadd = 1,
    kDependentImad = 2,
    kDependentLop3 = 3,
    kBranchTaken = 4,
    kIndependentIadd = 5,
    kIndependentImad = 6,
    kIndependentLop3 = 7,
    kMixedIaddFfma = 8,
};

template <int Length>
__device__ __forceinline__ std::uint32_t dependent_iadd(std::uint32_t value) {
    std::uint32_t high = value ^ 0x5a5a5a5au;
#pragma unroll
    for (int index = 0; index < Length; ++index) {
        asm volatile(
            "add.cc.u32 %0, %0, %2;\n\t"
            "addc.u32 %1, %1, 0;"
            : "+r"(value), "+r"(high)
            : "r"(kAdd));
    }
    return value ^ (high << 1);
}

template <int Length>
__device__ __forceinline__ std::uint32_t dependent_imad(std::uint32_t value) {
#pragma unroll
    for (int index = 0; index < Length; ++index) {
        asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                     : "+r"(value)
                     : "r"(kMul), "r"(kMadAdd));
    }
    return value;
}

template <int Length>
__device__ __forceinline__ std::uint32_t dependent_lop3(std::uint32_t value) {
#pragma unroll
    for (int index = 0; index < Length; ++index) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(value)
                     : "r"(kLogicA), "r"(kLogicB));
    }
    return value;
}

template <int Length>
__device__ __forceinline__ std::uint32_t branch_taken(std::uint32_t value) {
#pragma unroll
    for (int index = 0; index < Length; ++index) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.u32 p, %0, 0;\n\t"
            "@p bra.uni branch_done%=;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "mad.lo.u32 %0, %0, 1664525, 1013904223;\n\t"
            "branch_done%=:\n\t"
            "}"
            : "+r"(value));
    }
    return value;
}

template <int Length>
__device__ __forceinline__ std::uint32_t independent_iadd(std::uint32_t value) {
    std::uint32_t al = value;
    std::uint32_t bl = value ^ 0x11111111u;
    std::uint32_t cl = value ^ 0x22222222u;
    std::uint32_t dl = value ^ 0x33333333u;
    std::uint32_t ah = value ^ 0x5a5a5a5au;
    std::uint32_t bh = value ^ 0x4b4b4b4bu;
    std::uint32_t ch = value ^ 0x78787878u;
    std::uint32_t dh = value ^ 0x69696969u;
#pragma unroll
    for (int group = 0; group < Length / 4; ++group) {
        asm volatile(
            "add.cc.u32 %0, %0, %8;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            "add.cc.u32 %2, %2, %9;\n\t"
            "addc.u32 %3, %3, 0;\n\t"
            "add.cc.u32 %4, %4, %10;\n\t"
            "addc.u32 %5, %5, 0;\n\t"
            "add.cc.u32 %6, %6, %11;\n\t"
            "addc.u32 %7, %7, 0;"
            : "+r"(al), "+r"(ah), "+r"(bl), "+r"(bh), "+r"(cl),
              "+r"(ch), "+r"(dl), "+r"(dh)
            : "r"(kAdd), "r"(kAdd + 2u), "r"(kAdd + 4u),
              "r"(kAdd + 6u));
    }
    return al ^ (ah << 1) ^ (bl << 2) ^ (bh << 3) ^ (cl << 4) ^
           (ch << 5) ^ (dl << 6) ^ (dh << 7);
}

template <int Length>
__device__ __forceinline__ std::uint32_t independent_imad(std::uint32_t value) {
    std::uint32_t a = value;
    std::uint32_t b = value ^ 0x11111111u;
    std::uint32_t c = value ^ 0x22222222u;
    std::uint32_t d = value ^ 0x33333333u;
#pragma unroll
    for (int group = 0; group < Length / 4; ++group) {
        asm volatile(
            "mad.lo.u32 %0, %0, %4, %5;\n\t"
            "mad.lo.u32 %1, %1, %4, %6;\n\t"
            "mad.lo.u32 %2, %2, %4, %7;\n\t"
            "mad.lo.u32 %3, %3, %4, %8;"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
            : "r"(kMul), "r"(kMadAdd), "r"(kMadAdd + 2u),
              "r"(kMadAdd + 4u), "r"(kMadAdd + 6u));
    }
    return a ^ (b << 1) ^ (c << 2) ^ (d << 3);
}

template <int Length>
__device__ __forceinline__ std::uint32_t independent_lop3(std::uint32_t value) {
    std::uint32_t a = value;
    std::uint32_t b = value ^ 0x11111111u;
    std::uint32_t c = value ^ 0x22222222u;
    std::uint32_t d = value ^ 0x33333333u;
#pragma unroll
    for (int group = 0; group < Length / 4; ++group) {
        asm volatile(
            "lop3.b32 %0, %0, %4, %5, 0x96;\n\t"
            "lop3.b32 %1, %1, %4, %5, 0x96;\n\t"
            "lop3.b32 %2, %2, %4, %5, 0x96;\n\t"
            "lop3.b32 %3, %3, %4, %5, 0x96;"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
            : "r"(kLogicA), "r"(kLogicB));
    }
    return a ^ (b << 1) ^ (c << 2) ^ (d << 3);
}

template <int Length>
__device__ __forceinline__ std::uint32_t mixed_iadd_ffma(std::uint32_t value,
                                                         float* floating) {
    std::uint32_t high = value ^ 0x5a5a5a5au;
    float fp = *floating;
#pragma unroll
    for (int index = 0; index < Length; ++index) {
        asm volatile(
            "add.cc.u32 %0, %0, %3;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            "fma.rn.f32 %2, %2, %4, %5;"
            : "+r"(value), "+r"(high), "+f"(fp)
            : "r"(kAdd), "f"(kFmaMul), "f"(kFmaAdd));
    }
    *floating = fp;
    return value ^ (high << 1);
}

template <int ProtocolValue, int Length>
__global__ void issue_port_control(unsigned long long* elapsed,
                                   std::uint32_t* integer_sink,
                                   std::uint32_t* float_bits_sink) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    std::uint32_t value =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    float floating = 1.0f + static_cast<float>(warp * 32 + lane) * 0.0001f;

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        if constexpr (ProtocolValue == kDependentIadd) {
            value = dependent_iadd<Length>(value);
        } else if constexpr (ProtocolValue == kDependentImad) {
            value = dependent_imad<Length>(value);
        } else if constexpr (ProtocolValue == kDependentLop3) {
            value = dependent_lop3<Length>(value);
        } else if constexpr (ProtocolValue == kBranchTaken) {
            value = branch_taken<Length>(value);
        } else if constexpr (ProtocolValue == kIndependentIadd) {
            value = independent_iadd<Length>(value);
        } else if constexpr (ProtocolValue == kIndependentImad) {
            value = independent_imad<Length>(value);
        } else if constexpr (ProtocolValue == kIndependentLop3) {
            value = independent_lop3<Length>(value);
        } else if constexpr (ProtocolValue == kMixedIaddFfma) {
            value = mixed_iadd_ffma<Length>(value, &floating);
        }
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        integer_sink[warp] = value;
        float_bits_sink[warp] = __float_as_uint(floating);
    }
}

template <int ProtocolValue, int Length>
void launch_kernel(int warps, unsigned long long* elapsed,
                   std::uint32_t* integer_sink,
                   std::uint32_t* float_bits_sink) {
    issue_port_control<ProtocolValue, Length><<<1, warps * kWarpSize>>>(
        elapsed, integer_sink, float_bits_sink);
}

template <int ProtocolValue>
void launch_length(int length, int warps, unsigned long long* elapsed,
                   std::uint32_t* integer_sink,
                   std::uint32_t* float_bits_sink) {
    switch (length) {
        case 4:
            launch_kernel<ProtocolValue, 4>(warps, elapsed, integer_sink,
                                            float_bits_sink);
            break;
        case 8:
            launch_kernel<ProtocolValue, 8>(warps, elapsed, integer_sink,
                                            float_bits_sink);
            break;
        case 16:
            launch_kernel<ProtocolValue, 16>(warps, elapsed, integer_sink,
                                             float_bits_sink);
            break;
        case 32:
            launch_kernel<ProtocolValue, 32>(warps, elapsed, integer_sink,
                                             float_bits_sink);
            break;
        case 64:
            launch_kernel<ProtocolValue, 64>(warps, elapsed, integer_sink,
                                             float_bits_sink);
            break;
        default: std::abort();
    }
}

void launch_protocol(int protocol, int length, int warps,
                     unsigned long long* elapsed,
                     std::uint32_t* integer_sink,
                     std::uint32_t* float_bits_sink) {
    switch (protocol) {
        case kControl:
            launch_length<kControl>(length, warps, elapsed, integer_sink,
                                    float_bits_sink);
            break;
        case kDependentIadd:
            launch_length<kDependentIadd>(length, warps, elapsed, integer_sink,
                                          float_bits_sink);
            break;
        case kDependentImad:
            launch_length<kDependentImad>(length, warps, elapsed, integer_sink,
                                          float_bits_sink);
            break;
        case kDependentLop3:
            launch_length<kDependentLop3>(length, warps, elapsed, integer_sink,
                                          float_bits_sink);
            break;
        case kBranchTaken:
            launch_length<kBranchTaken>(length, warps, elapsed, integer_sink,
                                        float_bits_sink);
            break;
        case kIndependentIadd:
            launch_length<kIndependentIadd>(length, warps, elapsed, integer_sink,
                                            float_bits_sink);
            break;
        case kIndependentImad:
            launch_length<kIndependentImad>(length, warps, elapsed, integer_sink,
                                            float_bits_sink);
            break;
        case kIndependentLop3:
            launch_length<kIndependentLop3>(length, warps, elapsed, integer_sink,
                                            float_bits_sink);
            break;
        case kMixedIaddFfma:
            launch_length<kMixedIaddFfma>(length, warps, elapsed, integer_sink,
                                          float_bits_sink);
            break;
        default: std::abort();
    }
}

const char* protocol_name(int protocol) {
    static constexpr const char* names[] = {
        "p00_control",          "p10_dependent_iadd",
        "p11_dependent_imad",   "p12_dependent_lop3",
        "p13_branch_taken",     "p20_independent_iadd",
        "p21_independent_imad", "p22_independent_lop3",
        "p31_mixed_iadd_ffma",
    };
    return protocol >= 0 && protocol < 9 ? names[protocol] : nullptr;
}

bool parse_protocol(const char* text, int* protocol) {
    static constexpr const char* short_names[] = {
        "p00", "p10", "p11", "p12", "p13", "p20", "p21", "p22", "p31",
    };
    for (int index = 0; index < 9; ++index) {
        if (std::strcmp(text, short_names[index]) == 0 ||
            std::strcmp(text, protocol_name(index)) == 0) {
            *protocol = index;
            return true;
        }
    }
    return false;
}

std::uint32_t independent_seed(std::uint32_t value, int lane) {
    return value ^ static_cast<std::uint32_t>(lane) * 0x11111111u;
}

std::uint64_t iadd_seed(std::uint32_t low, std::uint32_t high) {
    return (static_cast<std::uint64_t>(high) << 32) | low;
}

std::uint32_t fold_iadd(std::uint64_t value) {
    return static_cast<std::uint32_t>(value) ^
           (static_cast<std::uint32_t>(value >> 32) << 1);
}

std::uint32_t host_chain(int protocol, int length, std::uint32_t value,
                         float* floating) {
    if (protocol == kDependentIadd) {
        std::uint64_t wide = iadd_seed(value, value ^ 0x5a5a5a5au);
        for (int index = 0; index < length; ++index) wide += kAdd;
        value = fold_iadd(wide);
    } else if (protocol == kDependentImad) {
        for (int index = 0; index < length; ++index)
            value = value * kMul + kMadAdd;
    } else if (protocol == kDependentLop3) {
        for (int index = 0; index < length; ++index)
            value ^= kLogicA ^ kLogicB;
    } else if (protocol == kIndependentIadd ||
               protocol == kIndependentImad ||
               protocol == kIndependentLop3) {
        std::uint32_t lanes[4] = {independent_seed(value, 0),
                                  independent_seed(value, 1),
                                  independent_seed(value, 2),
                                  independent_seed(value, 3)};
        std::uint64_t wide_lanes[4] = {
            iadd_seed(lanes[0], value ^ 0x5a5a5a5au),
            iadd_seed(lanes[1], value ^ 0x4b4b4b4bu),
            iadd_seed(lanes[2], value ^ 0x78787878u),
            iadd_seed(lanes[3], value ^ 0x69696969u),
        };
        for (int group = 0; group < length / 4; ++group) {
            for (int lane = 0; lane < 4; ++lane) {
                if (protocol == kIndependentIadd) {
                    wide_lanes[lane] +=
                        kAdd + static_cast<std::uint32_t>(lane * 2);
                } else if (protocol == kIndependentImad) {
                    lanes[lane] = lanes[lane] * kMul + kMadAdd +
                                  static_cast<std::uint32_t>(lane * 2);
                } else {
                    lanes[lane] ^= kLogicA ^ kLogicB;
                }
            }
        }
        if (protocol == kIndependentIadd) {
            value = static_cast<std::uint32_t>(wide_lanes[0]) ^
                    (static_cast<std::uint32_t>(wide_lanes[0] >> 32) << 1) ^
                    (static_cast<std::uint32_t>(wide_lanes[1]) << 2) ^
                    (static_cast<std::uint32_t>(wide_lanes[1] >> 32) << 3) ^
                    (static_cast<std::uint32_t>(wide_lanes[2]) << 4) ^
                    (static_cast<std::uint32_t>(wide_lanes[2] >> 32) << 5) ^
                    (static_cast<std::uint32_t>(wide_lanes[3]) << 6) ^
                    (static_cast<std::uint32_t>(wide_lanes[3] >> 32) << 7);
        } else {
            value = lanes[0] ^ (lanes[1] << 1) ^ (lanes[2] << 2) ^
                    (lanes[3] << 3);
        }
    } else if (protocol == kMixedIaddFfma) {
        std::uint64_t wide = iadd_seed(value, value ^ 0x5a5a5a5au);
        for (int index = 0; index < length; ++index) {
            wide += kAdd;
            *floating = std::fma(*floating, kFmaMul, kFmaAdd);
        }
        value = fold_iadd(wide);
    } else if (protocol == kBranchTaken) {
        for (int branch = 0; branch < length; ++branch) {
            if (value != 0) continue;
            for (int instruction = 0; instruction < 8; ++instruction)
                value = value * kMul + kMadAdd;
        }
    }
    return value;
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void expected(int protocol, int length, int warp, int lane,
              std::uint32_t* integer, std::uint32_t* floating_bits) {
    std::uint32_t value =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    float floating = 1.0f + static_cast<float>(warp * 32 + lane) * 0.0001f;
    for (int generation = 0; generation < kGenerations; ++generation) {
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        value = host_chain(protocol, length, value, &floating);
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    *integer = value;
    *floating_bits = float_bits(floating);
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s PROTOCOL LENGTH WARPS [TRIALS]\n", argv[0]);
        return 2;
    }
    int protocol = -1;
    const int length = std::atoi(argv[2]);
    const int warps = std::atoi(argv[3]);
    const int trials = argc == 5 ? std::atoi(argv[4]) : 7;
    if (!parse_protocol(argv[1], &protocol) ||
        (length != 4 && length != 8 && length != 16 && length != 32 &&
         length != 64) ||
        warps < 1 || warps > kMaxWarps || trials < 1 || trials > 32) {
        return 2;
    }

    unsigned long long* elapsed = nullptr;
    std::uint32_t* integer_sink = nullptr;
    std::uint32_t* float_bits_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed, kMaxWarps * sizeof(*elapsed)));
    CUDA_CHECK(cudaMalloc(&integer_sink, kMaxWarps * sizeof(*integer_sink)));
    CUDA_CHECK(cudaMalloc(&float_bits_sink,
                          kMaxWarps * sizeof(*float_bits_sink)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaProfilerStart());
        launch_protocol(protocol, length, warps, elapsed, integer_sink,
                        float_bits_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(samples[trial].data(), elapsed,
                              warps * sizeof(*elapsed),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<std::uint32_t> integers(warps);
    std::vector<std::uint32_t> floating_bits(warps);
    CUDA_CHECK(cudaMemcpy(integers.data(), integer_sink,
                          warps * sizeof(*integer_sink),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(floating_bits.data(), float_bits_sink,
                          warps * sizeof(*float_bits_sink),
                          cudaMemcpyDeviceToHost));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        std::uint32_t expected_integer;
        std::uint32_t expected_float;
        expected(protocol, length, warp, 0, &expected_integer, &expected_float);
        correct = correct && integers[warp] == expected_integer;
        if (protocol == kMixedIaddFfma) {
            correct = correct && floating_bits[warp] == expected_float;
        }
        checksum = checksum * 131u + integers[warp] + floating_bits[warp];
    }

    const int static_ops_per_generation =
        protocol == kControl ? 0
        : protocol == kDependentIadd || protocol == kIndependentIadd ? 2 * length
        : protocol == kMixedIaddFfma                                  ? 3 * length
        : protocol == kBranchTaken                                    ? 2 * length
                                                                      : length;
    std::printf(
        "{\"protocol\":\"%s\",\"length\":%d,\"warps\":%d,"
        "\"generations\":%d,\"static_ops_per_generation\":%d,"
        "\"samples_cycles\":[",
        protocol_name(protocol), length, warps, kGenerations,
        static_ops_per_generation);
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

    CUDA_CHECK(cudaFree(float_bits_sink));
    CUDA_CHECK(cudaFree(integer_sink));
    CUDA_CHECK(cudaFree(elapsed));
    return correct ? 0 : 1;
}
