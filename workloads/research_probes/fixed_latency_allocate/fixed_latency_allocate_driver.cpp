#include <cuda.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define DRIVER_CHECK(call)                                                      \
    do {                                                                        \
        const CUresult status = (call);                                         \
        if (status != CUDA_SUCCESS) {                                           \
            const char* name = nullptr;                                         \
            const char* description = nullptr;                                  \
            cuGetErrorName(status, &name);                                      \
            cuGetErrorString(status, &description);                             \
            std::fprintf(stderr, "%s failed: %s (%s)\n", #call,              \
                         name == nullptr ? "unknown" : name,                    \
                         description == nullptr ? "no description" : description); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 16;
constexpr int kHotInstructions = 16;

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

const char* function_name(int protocol) {
    static constexpr const char* names[] = {
        "_Z22fixed_latency_allocateILi0EEvPyPj",
        "_Z22fixed_latency_allocateILi1EEvPyPj",
        "_Z22fixed_latency_allocateILi2EEvPyPj",
        "_Z22fixed_latency_allocateILi3EEvPyPj",
        "_Z22fixed_latency_allocateILi4EEvPyPj",
        "_Z22fixed_latency_allocateILi5EEvPyPj",
    };
    return protocol >= 0 && protocol < 6 ? names[protocol] : nullptr;
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
    static constexpr const char* names[] = {
        "r00", "r10", "r11", "r20", "r21", "r12",
    };
    for (int index = 0; index < 6; ++index) {
        if (std::strcmp(text, names[index]) == 0 ||
            std::strcmp(text, protocol_name(index)) == 0) {
            *protocol = index;
            return true;
        }
    }
    return false;
}

bool parse_int(const char* text, int* value) {
    if (text == nullptr || *text == '\0' || value == nullptr) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

std::uint32_t mad(std::uint32_t lhs, std::uint32_t rhs,
                  std::uint32_t addend) {
    return lhs * rhs + addend;
}

void host_group(int protocol, bool patched_r11, Registers* registers) {
    std::uint32_t n0;
    std::uint32_t n1;
    std::uint32_t n2;
    std::uint32_t n3;
    if (protocol == kParityTriplets) {
        n0 = mad(registers->a, registers->c, registers->o0);
        n1 = mad(registers->b, registers->d, registers->o1);
        n2 = mad(registers->e, registers->a, registers->o2);
        n3 = mad(registers->f, registers->b, registers->o3);
    } else if (protocol == kAdjacentTriplets && patched_r11) {
        n0 = mad(registers->b, registers->d, registers->o0);
        n1 = mad(registers->b, registers->d, registers->o1);
        n2 = mad(registers->a, registers->c, registers->o2);
        n3 = mad(registers->d, registers->e, registers->o3);
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

std::uint32_t expected(int protocol, bool patched_r11, int generations,
                       int warp, int lane) {
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
        for (int generation = 0; generation < generations; ++generation) {
            for (int group = 0; group < kHotInstructions / 4; ++group) {
                host_group(protocol, patched_r11, &registers);
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
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s CUBIN PROTOCOL original|r11-same-bank "
                     "WARPS_4_8_12_16 GENERATIONS_64_8192 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    int protocol = -1;
    const bool patched_r11 = std::strcmp(argv[3], "r11-same-bank") == 0;
    const bool original = std::strcmp(argv[3], "original") == 0;
    int warps = 0;
    int generations = 0;
    int trials = 7;
    if (!parse_int(argv[4], &warps) || !parse_int(argv[5], &generations) ||
        (argc == 7 && !parse_int(argv[6], &trials))) {
        return 2;
    }
    if (!parse_protocol(argv[2], &protocol) || (!original && !patched_r11) ||
        (patched_r11 && protocol != kAdjacentTriplets) ||
        (warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        (generations != 64 && generations != 8192) ||
        trials < 1 || trials > 32) {
        return 2;
    }

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, function_name(protocol)));

    CUdeviceptr elapsed;
    CUdeviceptr sink;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink, kMaxWarps * sizeof(std::uint32_t)));
    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    bool elapsed_samples_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed, 0xff,
                                kMaxWarps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(sink, 0xa5,
                                kMaxWarps * sizeof(std::uint32_t)));
        void* arguments[] = {&elapsed, &sink};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, warps * kWarpSize, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
        for (int warp = 0; warp < warps; ++warp) {
            elapsed_samples_valid =
                elapsed_samples_valid &&
                samples[trial][warp] !=
                    std::numeric_limits<unsigned long long>::max() &&
                samples[trial][warp] > 0;
        }
    }
    std::vector<std::uint32_t> values(warps);
    DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink,
                              warps * sizeof(std::uint32_t)));
    bool correct = true;
    bool sink_valid = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        sink_valid = sink_valid && values[warp] != 0xa5a5a5a5u;
        correct = correct &&
                  values[warp] == expected(protocol, patched_r11, generations,
                                           warp, 0);
        checksum = checksum * 131u + values[warp];
    }

    std::printf(
        "{\"protocol\":\"%s\",\"variant\":\"%s\",\"warps\":%d,"
        "\"generations\":%d,\"samples_cycles\":[",
        protocol_name(protocol), patched_r11 ? "r11_same_bank" : "original",
        warps, generations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf(
        "],\"checksum\":%llu,\"correct\":%s,"
        "\"elapsed_samples_valid\":%s,\"sink_valid\":%s}\n",
        static_cast<unsigned long long>(checksum),
        (correct && elapsed_samples_valid && sink_valid) ? "true" : "false",
        elapsed_samples_valid ? "true" : "false", sink_valid ? "true" : "false");
    correct = correct && elapsed_samples_valid && sink_valid;

    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
