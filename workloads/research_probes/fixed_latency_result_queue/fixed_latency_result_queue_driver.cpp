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
constexpr const char* kFunction = "_Z18result_queue_pilotPyPjPKjPKf";

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr,
                     "usage: %s CUBIN VARIANT WARPS_4_8_12_16 "
                     "GENERATIONS_64_8192 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    const int warps = std::atoi(argv[3]);
    const int generations = std::atoi(argv[4]);
    const int trials = argc == 6 ? std::atoi(argv[5]) : 7;
    const bool valid_variant = std::strcmp(argv[2], "collision") == 0 ||
                               std::strcmp(argv[2], "separated") == 0;
    if (!valid_variant ||
        (warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        (generations != 64 && generations != 8192) || trials < 1 || trials > 32) {
        return 2;
    }

    std::vector<std::uint32_t> half(kMaxWarps * kWarpSize * 8, 0);
    std::vector<float> floating(kMaxWarps * kWarpSize * 8, 1.0f);
    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    CUdeviceptr elapsed;
    CUdeviceptr sink;
    CUdeviceptr half_device;
    CUdeviceptr float_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink, kMaxWarps * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&half_device, half.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&float_device, floating.size() * sizeof(float)));
    DRIVER_CHECK(cuMemcpyHtoD(half_device, half.data(), half.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(float_device, floating.data(), floating.size() * sizeof(float)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        void* arguments[] = {&elapsed, &sink, &half_device, &float_device};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, warps * kWarpSize, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
    }
    std::vector<std::uint32_t> values(warps);
    DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink, warps * sizeof(std::uint32_t)));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        const std::uint32_t sentinel =
            (0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131)) +
            static_cast<std::uint32_t>(generations);
        const std::uint32_t expected = sentinel;
        if (values[warp] != expected && warp == 0) {
            std::fprintf(stderr, "actual=0x%08x expected=0x%08x\n",
                         values[warp], expected);
        }
        correct = correct && values[warp] == expected;
        checksum = checksum * 131u + values[warp];
    }
    std::printf(
        "{\"variant\":\"%s\",\"warps\":%d,\"generations\":%d,"
        "\"samples_cycles\":[",
        argv[2], warps, generations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf("],\"checksum\":%llu,\"correct\":%s}\n",
                static_cast<unsigned long long>(checksum), correct ? "true" : "false");
    DRIVER_CHECK(cuMemFree(float_device));
    DRIVER_CHECK(cuMemFree(half_device));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
