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
constexpr const char* kFhFunction = "_Z12result_orderILb0EEvPyPjPKjS3_";
constexpr const char* kHfFunction = "_Z12result_orderILb1EEvPyPjPKjS3_";

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr,
                     "usage: %s CUBIN ORDER_hf_fh WARPS_4_TO_16 "
                     "GENERATIONS_64_8192 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    const bool is_hf = std::strcmp(argv[2], "hf") == 0;
    const bool is_fh = std::strcmp(argv[2], "fh") == 0;
    const int warps = std::atoi(argv[3]);
    const int generations = std::atoi(argv[4]);
    const int trials = argc == 6 ? std::atoi(argv[5]) : 7;
    if ((!is_hf && !is_fh) ||
        warps < 4 || warps > 16 ||
        (generations != 64 && generations != 8192) || trials < 1 || trials > 32) {
        return 2;
    }

    std::vector<std::uint32_t> zeros(kMaxWarps * kWarpSize, 0);
    std::vector<std::uint32_t> ones(kMaxWarps * kWarpSize, 0x3f800000u);
    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, is_hf ? kHfFunction : kFhFunction));

    CUdeviceptr elapsed;
    CUdeviceptr sink;
    CUdeviceptr zero_device;
    CUdeviceptr one_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink, kMaxWarps * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&zero_device, zeros.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&one_device, ones.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(zero_device, zeros.data(), zeros.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(one_device, ones.data(), ones.size() * sizeof(std::uint32_t)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        void* arguments[] = {&elapsed, &sink, &zero_device, &one_device};
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
        const std::uint32_t lane_zero = static_cast<std::uint32_t>(warp * 131);
        const std::uint32_t expected =
            (0x6a09e667u ^ lane_zero) ^ (0xbb67ae85u ^ lane_zero);
        correct = correct && values[warp] == expected;
        checksum = checksum * 131u + values[warp];
    }

    std::printf(
        "{\"order\":\"%s\",\"warps\":%d,\"generations\":%d,"
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

    DRIVER_CHECK(cuMemFree(one_device));
    DRIVER_CHECK(cuMemFree(zero_device));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
