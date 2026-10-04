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
constexpr const char* kFixedFunction =
    "_Z19bypass_fixed_searchPyPjPKj";
constexpr const char* kGlobalFunction =
    "_Z20bypass_global_searchPyPjPKjS2_";

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
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s CUBIN fixed|global WARPS GENERATIONS TRIALS "
                     "VARIANT\n",
                     argv[0]);
        return 2;
    }
    const bool fixed = std::strcmp(argv[2], "fixed") == 0;
    const bool global = std::strcmp(argv[2], "global") == 0;
    int warps = 0, generations = 0, trials = 0;
    if (!parse_int(argv[3], &warps) || !parse_int(argv[4], &generations) ||
        !parse_int(argv[5], &trials)) return 2;
    if ((!fixed && !global) ||
        warps < 1 || warps > kMaxWarps ||
        generations != 1025 || trials < 1 ||
        trials > 32) {
        return 2;
    }
    const int threads = warps * kWarpSize;
    std::vector<std::uint32_t> inputs(threads, 0x12345678u);
    std::vector<std::uint32_t> zeros(threads, 0);
    std::vector<std::uint32_t> sink(warps);
    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(
        &function, module, fixed ? kFixedFunction : kGlobalFunction));

    CUdeviceptr elapsed_device;
    CUdeviceptr sink_device;
    CUdeviceptr inputs_device;
    CUdeviceptr zeros_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink_device,
                            kMaxWarps * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&inputs_device,
                            inputs.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&zeros_device,
                            zeros.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(inputs_device, inputs.data(),
                              inputs.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(zeros_device, zeros.data(),
                              zeros.size() * sizeof(std::uint32_t)));

    bool elapsed_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed_device, 0xff,
                                kMaxWarps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(sink_device, 0xa5,
                                kMaxWarps * sizeof(std::uint32_t)));
        void* fixed_args[] = {&elapsed_device, &sink_device, &zeros_device};
        void* global_args[] = {
            &elapsed_device, &sink_device, &inputs_device, &zeros_device};
        DRIVER_CHECK(cuLaunchKernel(
            function, 1, 1, 1, threads, 1, 1, 0, nullptr,
            fixed ? fixed_args : global_args, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed_device,
                                  warps * sizeof(unsigned long long)));
        for (int warp = 0; warp < warps; ++warp)
            elapsed_valid = elapsed_valid && samples[trial][warp] != UINT64_MAX &&
                            samples[trial][warp] > 0;
    }
    DRIVER_CHECK(cuMemcpyDtoH(sink.data(), sink_device,
                              warps * sizeof(std::uint32_t)));
    const std::uint32_t expected = fixed
        ? static_cast<std::uint32_t>(((generations + 1ull) / 2ull) * 4ull)
        : static_cast<std::uint32_t>(generations * 0x12345678ull);
    bool correct = true;
    bool sink_valid = true;
    std::uint64_t checksum = 0;
    for (std::uint32_t value : sink) {
        correct = correct && value == expected;
        sink_valid = sink_valid && value != 0xa5a5a5a5u;
        checksum = checksum * 131u + value;
    }

    std::printf(
        "{\"variant\":\"%s\",\"kind\":\"%s\",\"warps\":%d,"
        "\"generations\":%d,\"samples_cycles\":[",
        argv[6], argv[2], warps, generations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    correct = correct && elapsed_valid && sink_valid;
    std::printf(
        "],\"expected\":\"%08x\",\"checksum\":%llu,\"correct\":%s,"
        "\"elapsed_samples_valid\":%s,\"sink_valid\":%s}\n",
        expected, static_cast<unsigned long long>(checksum),
        correct ? "true" : "false", elapsed_valid ? "true" : "false",
        sink_valid ? "true" : "false");
    DRIVER_CHECK(cuMemFree(zeros_device));
    DRIVER_CHECK(cuMemFree(inputs_device));
    DRIVER_CHECK(cuMemFree(sink_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
