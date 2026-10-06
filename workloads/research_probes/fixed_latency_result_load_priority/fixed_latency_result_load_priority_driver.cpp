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
                         name == nullptr ? "unknown" : name,                   \
                         description == nullptr ? "no description"             \
                                                : description);                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 16;
constexpr const char* kFunction =
    "_Z27result_load_priority_searchPyPjPKj";

bool parse_int(const char* text, int* value) {
    if (text == nullptr || *text == '\0' || value == nullptr) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) return false;
    *value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s CUBIN VARIANT WARPS GENERATIONS TRIALS\n",
                     argv[0]);
        return 2;
    }
    const char* variant = argv[2];
    int warps = 0;
    int generations = 0;
    int trials = 0;
    if (!parse_int(argv[3], &warps) || !parse_int(argv[4], &generations) ||
        !parse_int(argv[5], &trials)) return 2;
    const bool known_variant =
        std::strcmp(variant, "same_noncollision") == 0 ||
        std::strcmp(variant, "same_collision") == 0 ||
        std::strcmp(variant, "opposite_noncollision") == 0 ||
        std::strcmp(variant, "opposite_shifted") == 0;
    if (!known_variant ||
        (warps != 1 && warps != 4 && warps != 8 && warps != 12 &&
         warps != 16) ||
        generations != 1024 || trials < 1 ||
        trials > 32) {
        return 2;
    }

    const int threads = warps * kWarpSize;
    std::vector<std::uint32_t> inputs(2 * threads, 0);
    for (int thread = 0; thread < threads; ++thread) {
        inputs[thread] = 0x12345678u;
        inputs[thread + threads] = 0x00003c00u;
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

    CUdeviceptr elapsed_device;
    CUdeviceptr sink_device;
    CUdeviceptr inputs_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink_device,
                            kMaxWarps * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&inputs_device,
                            inputs.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(inputs_device, inputs.data(),
                              inputs.size() * sizeof(std::uint32_t)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    const std::uint32_t expected = 0x3f806400u;
    std::vector<std::uint32_t> sink(warps);
    bool correct = true, sink_valid = true;
    std::uint64_t checksum = 0;
    bool elapsed_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed_device, 0xff,
                                kMaxWarps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(sink_device, 0xa5,
                                kMaxWarps * sizeof(std::uint32_t)));
        void* arguments[] = {&elapsed_device, &sink_device, &inputs_device};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed_device,
                                  warps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemcpyDtoH(sink.data(), sink_device,
                                  warps * sizeof(std::uint32_t)));
        for (int warp = 0; warp < warps; ++warp) {
            elapsed_valid = elapsed_valid && samples[trial][warp] != UINT64_MAX &&
                            samples[trial][warp] > 0;
            correct = correct && sink[warp] == expected;
            sink_valid = sink_valid && sink[warp] != 0xa5a5a5a5u;
            checksum = checksum * 131u + sink[warp];
        }
    }

    std::printf(
        "{\"variant\":\"%s\",\"warps\":%d,\"generations\":%d,"
        "\"trials\":%d,\"all_trials_checked\":true,\"samples_cycles\":[",
        variant, warps, generations, trials);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf(
        "],\"expected\":\"%08x\",\"checksum\":%llu,\"correct\":%s,"
        "\"elapsed_samples_valid\":%s,\"sink_valid\":%s}\n",
        expected, static_cast<unsigned long long>(checksum),
        (correct && elapsed_valid && sink_valid) ? "true" : "false",
        elapsed_valid ? "true" : "false", sink_valid ? "true" : "false");

    DRIVER_CHECK(cuMemFree(inputs_device));
    DRIVER_CHECK(cuMemFree(sink_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return (correct && elapsed_valid && sink_valid) ? 0 : 1;
}
