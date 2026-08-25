#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
constexpr int kMaxWarps = 8;
constexpr const char* kFunction = "_Z22result_capacity_searchPyPjPKj";

int main(int argc, char** argv) {
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s CUBIN DEPTH_7_8_9 WARPS_1_4_8 "
                     "GENERATIONS_64_8192 TRIALS [reserved]\n",
                     argv[0]);
        return 2;
    }
    const int depth = std::atoi(argv[2]);
    const int warps = std::atoi(argv[3]);
    const int generations = std::atoi(argv[4]);
    const int trials = std::atoi(argv[5]);
    if ((depth < 7 || depth > 9) || (warps != 1 && warps != 4 && warps != 8) ||
        (generations != 64 && generations != 8192) || trials < 1 || trials > 32) {
        return 2;
    }

    const int threads = warps * kWarpSize;
    std::vector<std::uint32_t> inputs(threads * 9, 0);
    for (int thread = 0; thread < threads; ++thread) {
        for (int index = 5; index < 9; ++index) {
            inputs[thread * 9 + index] = 0x3f800000u;
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
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    CUdeviceptr elapsed;
    CUdeviceptr sink;
    CUdeviceptr input_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink, kMaxWarps * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&input_device, inputs.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(input_device, inputs.data(), inputs.size() * sizeof(std::uint32_t)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        void* arguments[] = {&elapsed, &sink, &input_device};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0, nullptr,
                                    arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
    }
    std::vector<std::uint32_t> values(warps);
    DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink, warps * sizeof(std::uint32_t)));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        const std::uint32_t expected = 0x6a09e667u ^ static_cast<std::uint32_t>(warp * 32);
        correct = correct && values[warp] == expected;
        checksum = checksum * 131u + values[warp];
    }

    std::printf(
        "{\"depth\":%d,\"warps\":%d,\"generations\":%d,"
        "\"samples_cycles\":[",
        depth, warps, generations);
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

    DRIVER_CHECK(cuMemFree(input_device));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
