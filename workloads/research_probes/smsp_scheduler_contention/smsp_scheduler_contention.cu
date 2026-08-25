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
constexpr int kHotLength = 16;
constexpr std::uint32_t kAdd = 0x3c6ef35fu;
constexpr std::uint32_t kMul = 1664525u;
constexpr std::uint32_t kMadAdd = 1013904223u;
constexpr std::uint32_t kLogicA = 0xa5a5a5a5u;
constexpr std::uint32_t kLogicB = 0x3c3c3c3cu;

#ifndef SMSP_SCHEDULER_GENERATIONS
constexpr int kGenerations = 8192;
#else
constexpr int kGenerations = SMSP_SCHEDULER_GENERATIONS;
#endif

enum Protocol : int {
    kControl = 0,
    kDependentCarry = 1,
    kIndependentCarry = 2,
    kDependentImad = 3,
    kIndependentImad = 4,
    kDependentLop3 = 5,
    kIndependentLop3 = 6,
    kCarryThenImad = 7,
    kImadThenCarry = 8,
    kCarryThenLop3 = 9,
    kLop3ThenCarry = 10,
    kImadThenLop3 = 11,
    kLop3ThenImad = 12,
};

#define PHASE_VOTE(IMMEDIATE)                                                \
    "setp.ne.u32 phase_all, %0, " #IMMEDIATE ";\n\t"                         \
    "vote.sync.ballot.b32 %0, phase_all, 0xffffffff;\n\t"
#define PHASE_VOTES_4 PHASE_VOTE(0) PHASE_VOTE(1) PHASE_VOTE(2) PHASE_VOTE(3)
#define PHASE_VOTES_8                                                        \
    PHASE_VOTES_4 PHASE_VOTE(4) PHASE_VOTE(5) PHASE_VOTE(6) PHASE_VOTE(7)
#define PHASE_VOTES_16                                                       \
    PHASE_VOTES_8 PHASE_VOTE(8) PHASE_VOTE(9) PHASE_VOTE(10)               \
        PHASE_VOTE(11) PHASE_VOTE(12) PHASE_VOTE(13) PHASE_VOTE(14)         \
            PHASE_VOTE(15)

template <int Padding, int Threshold>
__device__ __forceinline__ std::uint32_t phase_block(std::uint32_t token,
                                                      int round) {
    if constexpr (Padding == 4) {
        asm volatile(
            "{\n\t.reg .pred phase_skip, phase_all;\n\t"
            "setp.lt.s32 phase_skip, %1, %2;\n\t"
            "@phase_skip bra.uni phase_done%=;\n\t"
            PHASE_VOTES_4 "phase_done%=:\n\t}"
            : "+r"(token)
            : "r"(round), "n"(Threshold));
    } else if constexpr (Padding == 8) {
        asm volatile(
            "{\n\t.reg .pred phase_skip, phase_all;\n\t"
            "setp.lt.s32 phase_skip, %1, %2;\n\t"
            "@phase_skip bra.uni phase_done%=;\n\t"
            PHASE_VOTES_8 "phase_done%=:\n\t}"
            : "+r"(token)
            : "r"(round), "n"(Threshold));
    } else if constexpr (Padding == 16) {
        asm volatile(
            "{\n\t.reg .pred phase_skip, phase_all;\n\t"
            "setp.lt.s32 phase_skip, %1, %2;\n\t"
            "@phase_skip bra.uni phase_done%=;\n\t"
            PHASE_VOTES_16 "phase_done%=:\n\t}"
            : "+r"(token)
            : "r"(round), "n"(Threshold));
    }
    return token;
}

// `round == warp / 4`: slots 0..3 are the first warp on each modeled SMSP,
// 4..7 the second, and so on.  Every extra round receives one more copy of
// the phase body.  The q00 control executes exactly the same phase path.
template <int NominalPhase>
__device__ __forceinline__ std::uint32_t apply_phase(std::uint32_t token,
                                                     int round) {
    static_assert(NominalPhase == 0 || NominalPhase == 4 ||
                      NominalPhase == 8 || NominalPhase == 16,
                  "unsupported phase");
    if constexpr (NominalPhase > 0) {
        token = phase_block<NominalPhase, 1>(token, round);
        token = phase_block<NominalPhase, 2>(token, round);
        token = phase_block<NominalPhase, 3>(token, round);
    }
    return token;
}

#undef PHASE_VOTES_16
#undef PHASE_VOTES_8
#undef PHASE_VOTES_4
#undef PHASE_VOTE

template <int Length>
__device__ __forceinline__ std::uint32_t dependent_carry(std::uint32_t value) {
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
__device__ __forceinline__ std::uint32_t independent_carry(std::uint32_t value) {
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

template <int ProtocolValue>
__device__ __forceinline__ std::uint32_t hot_body(std::uint32_t value,
                                                  int round) {
    if constexpr (ProtocolValue == kDependentCarry) {
        return dependent_carry<kHotLength>(value);
    } else if constexpr (ProtocolValue == kIndependentCarry) {
        return independent_carry<kHotLength>(value);
    } else if constexpr (ProtocolValue == kDependentImad) {
        return dependent_imad<kHotLength>(value);
    } else if constexpr (ProtocolValue == kIndependentImad) {
        return independent_imad<kHotLength>(value);
    } else if constexpr (ProtocolValue == kDependentLop3) {
        return dependent_lop3<kHotLength>(value);
    } else if constexpr (ProtocolValue == kIndependentLop3) {
        return independent_lop3<kHotLength>(value);
    } else if constexpr (ProtocolValue == kCarryThenImad) {
        return (round & 1) == 0 ? dependent_carry<kHotLength>(value)
                                : dependent_imad<kHotLength>(value);
    } else if constexpr (ProtocolValue == kImadThenCarry) {
        return (round & 1) == 0 ? dependent_imad<kHotLength>(value)
                                : dependent_carry<kHotLength>(value);
    } else if constexpr (ProtocolValue == kCarryThenLop3) {
        return (round & 1) == 0 ? dependent_carry<kHotLength>(value)
                                : dependent_lop3<kHotLength>(value);
    } else if constexpr (ProtocolValue == kLop3ThenCarry) {
        return (round & 1) == 0 ? dependent_lop3<kHotLength>(value)
                                : dependent_carry<kHotLength>(value);
    } else if constexpr (ProtocolValue == kImadThenLop3) {
        return (round & 1) == 0 ? dependent_imad<kHotLength>(value)
                                : dependent_lop3<kHotLength>(value);
    } else if constexpr (ProtocolValue == kLop3ThenImad) {
        return (round & 1) == 0 ? dependent_lop3<kHotLength>(value)
                                : dependent_imad<kHotLength>(value);
    } else {
        return value;
    }
}

template <int ProtocolValue, int NominalPhase>
__global__ void smsp_scheduler_contention(unsigned long long* elapsed,
                                          std::uint32_t* sink) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int round = warp / 4;
    std::uint32_t value =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    std::uint32_t phase_token =
        0x510e527fu ^ static_cast<std::uint32_t>(warp * 67 + lane);

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        phase_token = apply_phase<NominalPhase>(phase_token, round);
        value = hot_body<ProtocolValue>(value, round);
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = value ^ phase_token;
    }
}

template <int ProtocolValue, int NominalPhase>
void launch_kernel(int warps, unsigned long long* elapsed, std::uint32_t* sink) {
    smsp_scheduler_contention<ProtocolValue, NominalPhase>
        <<<1, warps * kWarpSize>>>(elapsed, sink);
}

template <int ProtocolValue>
void launch_phase(int phase, int warps, unsigned long long* elapsed,
                  std::uint32_t* sink) {
    switch (phase) {
        case 0: launch_kernel<ProtocolValue, 0>(warps, elapsed, sink); break;
        case 4: launch_kernel<ProtocolValue, 4>(warps, elapsed, sink); break;
        case 8: launch_kernel<ProtocolValue, 8>(warps, elapsed, sink); break;
        case 16: launch_kernel<ProtocolValue, 16>(warps, elapsed, sink); break;
        default: std::abort();
    }
}

void launch_protocol(int protocol, int phase, int warps,
                     unsigned long long* elapsed, std::uint32_t* sink) {
    switch (protocol) {
        case kControl:
            launch_phase<kControl>(phase, warps, elapsed, sink);
            break;
        case kDependentCarry:
            launch_phase<kDependentCarry>(phase, warps, elapsed, sink);
            break;
        case kIndependentCarry:
            launch_phase<kIndependentCarry>(phase, warps, elapsed, sink);
            break;
        case kDependentImad:
            launch_phase<kDependentImad>(phase, warps, elapsed, sink);
            break;
        case kIndependentImad:
            launch_phase<kIndependentImad>(phase, warps, elapsed, sink);
            break;
        case kDependentLop3:
            launch_phase<kDependentLop3>(phase, warps, elapsed, sink);
            break;
        case kIndependentLop3:
            launch_phase<kIndependentLop3>(phase, warps, elapsed, sink);
            break;
        case kCarryThenImad:
            launch_phase<kCarryThenImad>(phase, warps, elapsed, sink);
            break;
        case kImadThenCarry:
            launch_phase<kImadThenCarry>(phase, warps, elapsed, sink);
            break;
        case kCarryThenLop3:
            launch_phase<kCarryThenLop3>(phase, warps, elapsed, sink);
            break;
        case kLop3ThenCarry:
            launch_phase<kLop3ThenCarry>(phase, warps, elapsed, sink);
            break;
        case kImadThenLop3:
            launch_phase<kImadThenLop3>(phase, warps, elapsed, sink);
            break;
        case kLop3ThenImad:
            launch_phase<kLop3ThenImad>(phase, warps, elapsed, sink);
            break;
        default: std::abort();
    }
}

const char* protocol_name(int protocol) {
    static constexpr const char* names[] = {
        "q00_control",
        "q10_dependent_carry",
        "q11_independent_carry",
        "q12_dependent_imad",
        "q13_independent_imad",
        "q14_dependent_lop3",
        "q15_independent_lop3",
        "q20_carry_then_imad",
        "q21_imad_then_carry",
        "q22_carry_then_lop3",
        "q23_lop3_then_carry",
        "q24_imad_then_lop3",
        "q25_lop3_then_imad",
    };
    return protocol >= 0 && protocol < 13 ? names[protocol] : nullptr;
}

bool parse_protocol(const char* text, int* protocol) {
    static constexpr const char* short_names[] = {
        "q00", "q10", "q11", "q12", "q13", "q14", "q15",
        "q20", "q21", "q22", "q23", "q24", "q25",
    };
    for (int index = 0; index < 13; ++index) {
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

std::uint64_t carry_seed(std::uint32_t low, std::uint32_t high) {
    return (static_cast<std::uint64_t>(high) << 32) | low;
}

std::uint32_t fold_carry(std::uint64_t value) {
    return static_cast<std::uint32_t>(value) ^
           (static_cast<std::uint32_t>(value >> 32) << 1);
}

std::uint32_t host_dependent_carry(std::uint32_t value) {
    std::uint64_t wide = carry_seed(value, value ^ 0x5a5a5a5au);
    for (int index = 0; index < kHotLength; ++index) wide += kAdd;
    return fold_carry(wide);
}

std::uint32_t host_independent_carry(std::uint32_t value) {
    std::uint32_t lanes[4] = {
        independent_seed(value, 0), independent_seed(value, 1),
        independent_seed(value, 2), independent_seed(value, 3),
    };
    std::uint64_t wide[4] = {
        carry_seed(lanes[0], value ^ 0x5a5a5a5au),
        carry_seed(lanes[1], value ^ 0x4b4b4b4bu),
        carry_seed(lanes[2], value ^ 0x78787878u),
        carry_seed(lanes[3], value ^ 0x69696969u),
    };
    for (int group = 0; group < kHotLength / 4; ++group) {
        for (int lane = 0; lane < 4; ++lane) {
            wide[lane] += kAdd + static_cast<std::uint32_t>(lane * 2);
        }
    }
    return static_cast<std::uint32_t>(wide[0]) ^
           (static_cast<std::uint32_t>(wide[0] >> 32) << 1) ^
           (static_cast<std::uint32_t>(wide[1]) << 2) ^
           (static_cast<std::uint32_t>(wide[1] >> 32) << 3) ^
           (static_cast<std::uint32_t>(wide[2]) << 4) ^
           (static_cast<std::uint32_t>(wide[2] >> 32) << 5) ^
           (static_cast<std::uint32_t>(wide[3]) << 6) ^
           (static_cast<std::uint32_t>(wide[3] >> 32) << 7);
}

std::uint32_t host_dependent_imad(std::uint32_t value) {
    for (int index = 0; index < kHotLength; ++index)
        value = value * kMul + kMadAdd;
    return value;
}

std::uint32_t host_independent_imad(std::uint32_t value) {
    std::uint32_t lanes[4] = {
        independent_seed(value, 0), independent_seed(value, 1),
        independent_seed(value, 2), independent_seed(value, 3),
    };
    for (int group = 0; group < kHotLength / 4; ++group) {
        for (int lane = 0; lane < 4; ++lane) {
            lanes[lane] = lanes[lane] * kMul + kMadAdd +
                          static_cast<std::uint32_t>(lane * 2);
        }
    }
    return lanes[0] ^ (lanes[1] << 1) ^ (lanes[2] << 2) ^
           (lanes[3] << 3);
}

std::uint32_t host_dependent_lop3(std::uint32_t value) {
    for (int index = 0; index < kHotLength; ++index)
        value ^= kLogicA ^ kLogicB;
    return value;
}

std::uint32_t host_independent_lop3(std::uint32_t value) {
    std::uint32_t lanes[4] = {
        independent_seed(value, 0), independent_seed(value, 1),
        independent_seed(value, 2), independent_seed(value, 3),
    };
    for (int group = 0; group < kHotLength / 4; ++group) {
        for (std::uint32_t& lane : lanes) lane ^= kLogicA ^ kLogicB;
    }
    return lanes[0] ^ (lanes[1] << 1) ^ (lanes[2] << 2) ^
           (lanes[3] << 3);
}

std::uint32_t host_hot_body(int protocol, std::uint32_t value, int round) {
    switch (protocol) {
        case kControl: return value;
        case kDependentCarry: return host_dependent_carry(value);
        case kIndependentCarry: return host_independent_carry(value);
        case kDependentImad: return host_dependent_imad(value);
        case kIndependentImad: return host_independent_imad(value);
        case kDependentLop3: return host_dependent_lop3(value);
        case kIndependentLop3: return host_independent_lop3(value);
        case kCarryThenImad:
            return (round & 1) == 0 ? host_dependent_carry(value)
                                    : host_dependent_imad(value);
        case kImadThenCarry:
            return (round & 1) == 0 ? host_dependent_imad(value)
                                    : host_dependent_carry(value);
        case kCarryThenLop3:
            return (round & 1) == 0 ? host_dependent_carry(value)
                                    : host_dependent_lop3(value);
        case kLop3ThenCarry:
            return (round & 1) == 0 ? host_dependent_lop3(value)
                                    : host_dependent_carry(value);
        case kImadThenLop3:
            return (round & 1) == 0 ? host_dependent_imad(value)
                                    : host_dependent_lop3(value);
        case kLop3ThenImad:
            return (round & 1) == 0 ? host_dependent_lop3(value)
                                    : host_dependent_imad(value);
        default: std::abort();
    }
}

std::uint32_t expected(int protocol, int phase, int warp, int lane) {
    std::uint32_t value =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    const int round = warp / 4;
    std::uint32_t phase_token =
        0x510e527fu ^ static_cast<std::uint32_t>(warp * 67 + lane);
    for (int generation = 0; generation < kGenerations; ++generation) {
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        if (phase != 0 && round >= 1) phase_token = 0xffffffffu;
        value = host_hot_body(protocol, value, round);
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    return value ^ phase_token;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s PROTOCOL PHASE_0_4_8_16 WARPS_4_8_12_16 "
                     "[TRIALS]\n",
                     argv[0]);
        return 2;
    }
    int protocol = -1;
    const int phase = std::atoi(argv[2]);
    const int warps = std::atoi(argv[3]);
    const int trials = argc == 5 ? std::atoi(argv[4]) : 7;
    if (!parse_protocol(argv[1], &protocol) ||
        (phase != 0 && phase != 4 && phase != 8 && phase != 16) ||
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
        launch_protocol(protocol, phase, warps, elapsed, sink);
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
        correct = correct && values[warp] == expected(protocol, phase, warp, 0);
        checksum = checksum * 131u + values[warp];
    }

    const int hot_ops =
        protocol == kControl ? 0
        : protocol == kDependentCarry || protocol == kIndependentCarry
            ? 2 * kHotLength
            : kHotLength;
    std::printf(
        "{\"protocol\":\"%s\",\"phase\":%d,\"warps\":%d,"
        "\"warps_per_modeled_smsp\":%d,\"generations\":%d,"
        "\"hot_logical_ops_per_generation\":%d,"
        "\"samples_cycles\":[",
        protocol_name(protocol), phase, warps, warps / 4, kGenerations,
        hot_ops);
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
