#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

const char* function_name(const char* protocol) {
    if (std::strcmp(protocol, "u10") == 0) return "reuse_u10";
    if (std::strcmp(protocol, "u11") == 0) return "reuse_u11";
    if (std::strcmp(protocol, "u12") == 0) return "reuse_u12";
    if (std::strcmp(protocol, "u20") == 0) return "reuse_u20";
    if (std::strcmp(protocol, "u22") == 0) return "reuse_u22";
    return nullptr;
}

std::uint32_t factor_value(int warp, int lane) {
    return (0x9e3779b9u ^ static_cast<std::uint32_t>(warp * 131 + lane)) | 1u;
}

std::uint32_t expected(int generations, int warp, int lane) {
    const std::uint32_t seed =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    std::uint32_t outputs[] = {
        seed ^ 0x01010101u, seed ^ 0x12121212u,
        seed ^ 0x23232323u, seed ^ 0x34343434u,
        seed ^ 0x45454545u, seed ^ 0x56565656u,
        seed ^ 0x67676767u, seed ^ 0x78787878u,
        seed ^ 0x89898989u, seed ^ 0x9a9a9a9au,
        seed ^ 0xababababu, seed ^ 0xbcbcbcbcu,
        seed ^ 0xcdcdcdcdu, seed ^ 0xdedededeu,
        seed ^ 0xefefefefu, seed ^ 0xf0f0f0f0u,
    };
    const std::uint32_t factor = factor_value(warp, lane);
    const std::uint32_t increment = factor * factor;
    std::uint32_t result = 0;
    for (int index = 0; index < 16; ++index) {
        outputs[index] += increment * static_cast<std::uint32_t>(generations);
        result ^= outputs[index] << index;
    }
    for (int index = 0; index < 8; ++index) {
        result ^= factor << index;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s CUBIN PROTOCOL a|b WARPS_4_8_12_16 "
                     "GENERATIONS_64_8192 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    const char* function = function_name(argv[2]);
    const bool variant_a = std::strcmp(argv[3], "a") == 0;
    const bool variant_b = std::strcmp(argv[3], "b") == 0;
    const int warps = std::atoi(argv[4]);
    const int generations = std::atoi(argv[5]);
    const int trials = argc == 7 ? std::atoi(argv[6]) : 7;
    if (function == nullptr || (!variant_a && !variant_b) ||
        (warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        (generations != 64 && generations != 8192) || trials < 1 || trials > 32) {
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
    CUfunction kernel;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&kernel, module, function));

    CUdeviceptr elapsed;
    CUdeviceptr sink;
    CUdeviceptr factors;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink, kMaxWarps * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&factors, host_factors.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(factors, host_factors.data(),
                              host_factors.size() * sizeof(std::uint32_t)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        void* arguments[] = {&elapsed, &sink, &factors};
        DRIVER_CHECK(cuLaunchKernel(kernel, 1, 1, 1, warps * kWarpSize, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
    }
    std::vector<std::uint32_t> values(warps);
    DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink,
                              warps * sizeof(std::uint32_t)));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        correct = correct && values[warp] == expected(generations, warp, 0);
        checksum = checksum * 131u + values[warp];
    }

    std::printf(
        "{\"protocol\":\"%s\",\"variant\":\"%s\",\"warps\":%d,"
        "\"warps_per_modeled_smsp\":%d,\"generations\":%d,"
        "\"hot_instructions_per_generation\":16,\"samples_cycles\":[",
        argv[2], variant_a ? "a" : "b", warps, warps / 4, generations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf("],\"checksum\":%llu,\"correct\":%s}\n",
                static_cast<unsigned long long>(checksum),
                correct ? "true" : "false");

    DRIVER_CHECK(cuMemFree(factors));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
