#include <cuda.h>

#include <cerrno>
#include <climits>
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
constexpr int kFactorCount = 8;
constexpr std::size_t kElapsedBytes = kMaxWarps * sizeof(unsigned long long);
constexpr std::size_t kSinkBytes = kMaxWarps * sizeof(std::uint32_t);
constexpr std::size_t kFactorBytes = kMaxWarps * kWarpSize * kFactorCount * sizeof(std::uint32_t);
constexpr std::size_t kMaxAllocationBytes = 1ull << 20;

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

const char* function_name(int distance) {
    switch (distance) {
        case 0: return "completion_d0";
        case 1: return "completion_d1";
        case 2: return "completion_d2";
        case 4: return "completion_d4";
        case 8: return "completion_d8";
        default: return nullptr;
    }
}

std::uint32_t factor_value(int warp, int lane) {
    return (0x9e3779b9u ^ static_cast<std::uint32_t>(warp * 131 + lane)) | 1u;
}

std::uint32_t expected(int distance, int generations, int warp, int lane) {
    const std::uint32_t seed =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    std::uint32_t outputs[] = {
        seed ^ 0x01010101u, seed ^ 0x12121212u,
        seed ^ 0x23232323u, seed ^ 0x34343434u,
        seed ^ 0x45454545u, seed ^ 0x56565656u,
        seed ^ 0x67676767u, seed ^ 0x78787878u,
    };
    const std::uint32_t factor = factor_value(warp, lane);
    if (distance != 0) {
        const std::uint32_t increment = factor * factor;
        const std::uint32_t additions =
            static_cast<std::uint32_t>((16 / distance) * generations);
        for (int index = 0; index < distance; ++index) {
            outputs[index] += increment * additions;
        }
    }
    std::uint32_t result = 0;
    for (int index = 0; index < 8; ++index) {
        result ^= outputs[index] << index;
        result ^= factor << index;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s CUBIN DISTANCE_0_1_2_4_8 split|bank3 "
                     "WARPS_4_8_12_16 GENERATIONS_64_8192 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    int distance = 0;
    int warps = 0;
    int generations = 0;
    int trials = 7;
    if (!parse_int(argv[2], &distance) || !parse_int(argv[4], &warps) ||
        !parse_int(argv[5], &generations) ||
        (argc == 7 && !parse_int(argv[6], &trials))) {
        std::fprintf(stderr, "numeric arguments must be signed decimal integers\n");
        return 2;
    }
    const bool split = std::strcmp(argv[3], "split") == 0;
    const bool bank3 = std::strcmp(argv[3], "bank3") == 0;
    if (function_name(distance) == nullptr || (!split && !bank3) ||
        (distance == 0 && bank3) ||
        (warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        (generations != 64 && generations != 8192) ||
        trials < 1 || trials > 32) {
        return 2;
    }

    std::vector<std::uint32_t> host_factors(
        kMaxWarps * kWarpSize * kFactorCount);
    for (int warp = 0; warp < kMaxWarps; ++warp) {
        for (int lane = 0; lane < kWarpSize; ++lane) {
            const std::uint32_t factor = factor_value(warp, lane);
            const int base = (warp * kWarpSize + lane) * kFactorCount;
            for (int index = 0; index < kFactorCount; ++index) {
                host_factors[base + index] = factor;
            }
        }
    }

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, function_name(distance)));

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    DRIVER_CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t requested_bytes = kElapsedBytes + kSinkBytes + kFactorBytes;
    if (requested_bytes > kMaxAllocationBytes || requested_bytes > free_bytes / 2) {
        std::fprintf(stderr,
                     "requested %zu bytes exceeds the bounded allocation budget "
                     "(%zu bytes, %zu free)\n",
                     requested_bytes, kMaxAllocationBytes, free_bytes);
        return 3;
    }

    CUdeviceptr elapsed;
    CUdeviceptr sink;
    CUdeviceptr factors;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kElapsedBytes));
    DRIVER_CHECK(cuMemAlloc(&sink, kSinkBytes));
    DRIVER_CHECK(cuMemAlloc(&factors, kFactorBytes));
    DRIVER_CHECK(cuMemcpyHtoD(factors, host_factors.data(), kFactorBytes));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed, 0xff, kElapsedBytes));
        DRIVER_CHECK(cuMemsetD8(sink, 0xa5, kSinkBytes));
        void* arguments[] = {&elapsed, &sink, &factors};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, warps * kWarpSize, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
    }
    std::vector<std::uint32_t> values(warps);
    DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink,
                              warps * sizeof(std::uint32_t)));
    bool correct = true;
    bool elapsed_valid = true;
    bool checksums_valid = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        correct = correct &&
                  values[warp] == expected(distance, generations, warp, 0);
        checksums_valid = checksums_valid && values[warp] != 0xa5a5a5a5u;
        for (int trial = 0; trial < trials; ++trial) {
            elapsed_valid = elapsed_valid && samples[trial][warp] != UINT64_MAX &&
                            samples[trial][warp] > 0;
        }
        checksum = checksum * 131u + values[warp];
    }
    correct = correct && elapsed_valid && checksums_valid;

    std::printf(
        "{\"distance\":%d,\"variant\":\"%s\",\"warps\":%d,"
        "\"warps_per_modeled_smsp\":%d,\"generations\":%d,"
        "\"hot_instructions_per_generation\":%d,\"samples_cycles\":[",
        distance, bank3 ? "bank3" : "split", warps, warps / 4, generations,
        distance == 0 ? 0 : 16);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf("],\"checksum\":%llu,\"correct\":%s,\"elapsed_samples_valid\":%s,"
                "\"checksum_expected\":%s}\n",
                static_cast<unsigned long long>(checksum),
                correct ? "true" : "false", elapsed_valid ? "true" : "false",
                checksums_valid ? "true" : "false");

    DRIVER_CHECK(cuMemFree(factors));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
