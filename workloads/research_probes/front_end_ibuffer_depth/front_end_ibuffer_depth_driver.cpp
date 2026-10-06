#include <cuda.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
constexpr int kTimestamps = 9;
constexpr const char* kFunction = "_Z22ibuffer_timeline_probePyPj";

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
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s CUBIN WARPS TRIALS VARIANT\n", argv[0]);
        return 2;
    }
    int warps = 0, trials = 0;
    if (!parse_int(argv[2], &warps) || !parse_int(argv[3], &trials) ||
        warps < 1 || warps > kMaxWarps || trials < 1 || trials > 32) {
        return 2;
    }
    std::vector<unsigned long long> timeline(warps * kTimestamps);
    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps * kTimestamps));
    std::vector<std::uint32_t> sink(warps);

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));

    CUdeviceptr timeline_device;
    CUdeviceptr sink_device;
    DRIVER_CHECK(cuMemAlloc(&timeline_device,
                            kMaxWarps * kTimestamps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&sink_device,
                            kMaxWarps * sizeof(std::uint32_t)));
    void* arguments[] = {&timeline_device, &sink_device};
    bool timeline_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(timeline_device, 0xff,
                                kMaxWarps * kTimestamps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(sink_device, 0xa5,
                                kMaxWarps * sizeof(std::uint32_t)));
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, warps * kWarpSize, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(timeline.data(), timeline_device,
                                  timeline.size() * sizeof(unsigned long long)));
        samples[trial] = timeline;
        for (int warp = 0; warp < warps; ++warp) {
            for (int index = 1; index < kTimestamps; ++index) {
                timeline_valid = timeline_valid &&
                    timeline[warp * kTimestamps + index] !=
                        std::numeric_limits<unsigned long long>::max() &&
                    timeline[warp * kTimestamps + index] >
                        timeline[warp * kTimestamps + index - 1];
            }
        }
    }
    DRIVER_CHECK(cuMemcpyDtoH(sink.data(), sink_device,
                              warps * sizeof(std::uint32_t)));

    bool correct = true;
    bool sink_valid = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        correct = correct && sink[warp] == 0x1b0f0000u + warp;
        sink_valid = sink_valid && sink[warp] != 0xa5a5a5a5u;
        checksum = checksum * 131u + sink[warp];
        for (int index = 1; index < kTimestamps; ++index) {
            correct = correct &&
                      samples.back()[warp * kTimestamps + index] >
                          samples.back()[warp * kTimestamps + index - 1];
        }
    }

    std::printf(
        "{\"variant\":\"%s\",\"warps\":%d,\"instructions\":32,"
        "\"timestamps_per_warp\":9,\"timeline_cycles\":[",
        argv[4], warps);
    for (int trial = 0; trial < trials; ++trial) {
        const unsigned long long origin =
            *std::min_element(samples[trial].begin(), samples[trial].end());
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s[", warp == 0 ? "" : ",");
            for (int index = 0; index < kTimestamps; ++index) {
                const unsigned long long value =
                    samples[trial][warp * kTimestamps + index] - origin;
                std::printf("%s%llu", index == 0 ? "" : ",", value);
            }
            std::printf("]");
        }
        std::printf("]");
    }
    std::printf(
        "],\"checksum\":%llu,\"correct\":%s,\"timeline_valid\":%s,"
        "\"sink_valid\":%s}\n",
        static_cast<unsigned long long>(checksum),
        (correct && timeline_valid && sink_valid) ? "true" : "false",
        timeline_valid ? "true" : "false", sink_valid ? "true" : "false");
    correct = correct && timeline_valid && sink_valid;
    return correct ? 0 : 1;
}
