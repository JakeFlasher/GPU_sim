#include <cuda.h>

#include <cerrno>
#include <cstdint>
#include <limits>
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

bool parse_int(const char* text, int* value) {
    if (text == nullptr || *text < '0' || *text > '9') return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed > std::numeric_limits<int>::max()) return false;
    *value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s CUBIN DEPTH_7_8_9 WARPS_1_4_8 "
                     "GENERATIONS_8192 TRIALS\n",
                     argv[0]);
        return 2;
    }
    int depth = 0, warps = 0, generations = 0, trials = 0;
    if (!parse_int(argv[2], &depth) || !parse_int(argv[3], &warps) ||
        !parse_int(argv[4], &generations) || !parse_int(argv[5], &trials) ||
        depth < 7 || depth > 9 || (warps != 1 && warps != 4 && warps != 8) ||
        generations != 8192 || trials < 1 || trials > 32) {
        std::fprintf(stderr, "expected depth 7..9, warps 1/4/8, generations 8192, trials 1..32\n");
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
    std::vector<std::uint32_t> values(warps);
    bool elapsed_samples_valid = true, sink_valid = true;
    std::uint64_t checksum = 0;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed, 0xff, kMaxWarps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(sink, 0xa5, kMaxWarps * sizeof(std::uint32_t)));
        void* arguments[] = {&elapsed, &sink, &input_device};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0, nullptr,
                                    arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink, warps * sizeof(std::uint32_t)));
        checksum = 0;
        for (int warp = 0; warp < warps; ++warp) {
            const auto sample = samples[trial][warp];
            elapsed_samples_valid = elapsed_samples_valid && sample > 0 && sample != UINT64_MAX;
            const std::uint32_t expected = 0x6a09e667u ^ static_cast<std::uint32_t>(warp * 32);
            sink_valid = sink_valid && values[warp] == expected;
            checksum = checksum * 131u + values[warp];
        }
    }
    const bool correct = elapsed_samples_valid && sink_valid;

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
    std::printf("],\"checksum\":%llu,\"correct\":%s,\"trials\":%d,"
                "\"elapsed_samples_valid\":%s,\"sink_valid\":%s,"
                "\"all_trials_checked\":true}\n",
                static_cast<unsigned long long>(checksum), correct ? "true" : "false", trials,
                elapsed_samples_valid ? "true" : "false", sink_valid ? "true" : "false");

    DRIVER_CHECK(cuMemFree(input_device));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
