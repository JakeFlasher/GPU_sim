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
constexpr std::size_t kElapsedBytes = kMaxWarps * sizeof(unsigned long long);
constexpr std::size_t kSinkBytes = kMaxWarps * sizeof(std::uint32_t);
constexpr std::size_t kInputBytes = kMaxWarps * kWarpSize * 8 * sizeof(std::uint32_t);
constexpr std::size_t kFloatInputBytes = kMaxWarps * kWarpSize * 8 * sizeof(float);
constexpr std::size_t kMaxAllocationBytes = 1ull << 20;
constexpr const char* kFunction = "_Z18result_queue_pilotPyPjPKjPKf";

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

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr,
                     "usage: %s CUBIN VARIANT WARPS_4_8_12_16 "
                     "GENERATIONS_8192 [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    int warps = 0;
    int generations = 0;
    int trials = 7;
    if (!parse_int(argv[3], &warps) || !parse_int(argv[4], &generations) ||
        (argc == 6 && !parse_int(argv[5], &trials))) {
        std::fprintf(stderr, "numeric arguments must be signed decimal integers\n");
        return 2;
    }
    const bool valid_variant = std::strcmp(argv[2], "collision") == 0 ||
                               std::strcmp(argv[2], "separated") == 0;
    if (!valid_variant ||
        (warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        generations != 8192 || trials < 1 || trials > 32) {
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
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    DRIVER_CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t requested_bytes = kElapsedBytes + kSinkBytes +
                                        kInputBytes + kFloatInputBytes;
    if (requested_bytes > kMaxAllocationBytes || requested_bytes > free_bytes / 2) {
        std::fprintf(stderr,
                     "requested %zu bytes exceeds the bounded allocation budget "
                     "(%zu bytes, %zu free)\n",
                     requested_bytes, kMaxAllocationBytes, free_bytes);
        return 3;
    }
    CUdeviceptr elapsed;
    CUdeviceptr sink;
    CUdeviceptr half_device;
    CUdeviceptr float_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed, kElapsedBytes));
    DRIVER_CHECK(cuMemAlloc(&sink, kSinkBytes));
    DRIVER_CHECK(cuMemAlloc(&half_device, kInputBytes));
    DRIVER_CHECK(cuMemAlloc(&float_device, kFloatInputBytes));
    DRIVER_CHECK(cuMemcpyHtoD(half_device, half.data(), kInputBytes));
    DRIVER_CHECK(cuMemcpyHtoD(float_device, floating.data(), kFloatInputBytes));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    std::vector<std::uint32_t> values(warps);
    bool elapsed_valid = true;
    bool checksums_valid = true;
    std::uint64_t checksum = 0;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed, 0xff, kElapsedBytes));
        DRIVER_CHECK(cuMemsetD8(sink, 0xa5, kSinkBytes));
        void* arguments[] = {&elapsed, &sink, &half_device, &float_device};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, warps * kWarpSize, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed,
                                  warps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemcpyDtoH(values.data(), sink,
                                  warps * sizeof(std::uint32_t)));
        checksum = 0;
        for (int warp = 0; warp < warps; ++warp) {
            const std::uint32_t wanted = ((0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131)) +
                 static_cast<std::uint32_t>(generations));
            if (values[warp] != wanted && checksums_valid) {
                std::fprintf(stderr, "trial=%d warp=%d actual=0x%08x expected=0x%08x\n",
                             trial, warp, values[warp], wanted);
            }
            // Compare every trial, not just the last launch. The observable
            // result is lane zero of each warp, not every lane's register state.
            checksums_valid = checksums_valid && values[warp] == wanted;
            elapsed_valid = elapsed_valid && samples[trial][warp] != UINT64_MAX &&
                            samples[trial][warp] > 0;
            checksum = checksum * 131u + values[warp];
        }
    }
    const bool correct = elapsed_valid && checksums_valid;

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
    std::printf("],\"checksum\":%llu,\"correct\":%s,\"elapsed_samples_valid\":%s,"
                "\"checksum_expected\":%s,\"sink_checks\":%d}\n",
                static_cast<unsigned long long>(checksum), correct ? "true" : "false",
                elapsed_valid ? "true" : "false", checksums_valid ? "true" : "false", trials * warps);
    DRIVER_CHECK(cuMemFree(float_device));
    DRIVER_CHECK(cuMemFree(half_device));
    DRIVER_CHECK(cuMemFree(sink));
    DRIVER_CHECK(cuMemFree(elapsed));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
