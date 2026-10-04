#include <cuda.h>

#include <algorithm>
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
                         name == nullptr ? "unknown" : name,                   \
                         description == nullptr ? "no description"             \
                                                : description);                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 16;
constexpr int kMaxThreads = kMaxWarps * kWarpSize;
constexpr int kDataWords = kMaxThreads * 32;
constexpr std::size_t kElapsedBytes = kMaxWarps * sizeof(unsigned long long);
constexpr std::size_t kReturnedBytes = kMaxThreads * sizeof(unsigned long long);
constexpr std::size_t kDataBytes = kDataWords * sizeof(std::uint32_t);
constexpr std::size_t kMaxAllocationBytes = 4ull * 1024ull * 1024ull;
constexpr const char* kFunction = "_Z12atomic_probePyS_Pj";

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

int variant_index(const char* value) {
    static constexpr const char* names[] = {
        "ret_lane_line", "ret_lane_word", "ret_warp_word", "ret_global_word",
        "red_lane_line", "red_lane_word", "red_warp_word", "red_global_word",
    };
    for (int index = 0; index < 8; ++index) {
        if (std::strcmp(value, names[index]) == 0) return index;
    }
    return -1;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: %s CUBIN WARPS GENERATIONS TRIALS VARIANT LAYOUT "
                     "RETURNING\n",
                     argv[0]);
        return 2;
    }
    int warps = 0;
    int generations = 0;
    int trials = 0;
    int layout = 0;
    int returning_int = 0;
    if (!parse_int(argv[2], &warps) || !parse_int(argv[3], &generations) ||
        !parse_int(argv[4], &trials) || !parse_int(argv[6], &layout) ||
        !parse_int(argv[7], &returning_int)) {
        std::fprintf(stderr, "numeric arguments must be signed decimal integers\n");
        return 2;
    }
    const bool returning = returning_int != 0;
    const int variant = variant_index(argv[5]);
    if (warps < 1 || warps > kMaxWarps ||
        generations != 1024 || trials < 1 || trials > 32 || layout < 0 || layout > 3 ||
        (returning_int != 0 && returning_int != 1) ||
        variant < 0 || variant != layout + (returning ? 0 : 4)) {
        std::fprintf(stderr, "arguments exceed the bounded atomic probe contract\n");
        return 2;
    }
    const int threads = warps * kWarpSize;
    std::vector<unsigned long long> elapsed(warps);
    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    std::vector<unsigned long long> returned(threads);
    std::vector<std::uint32_t> zeros(kDataWords, 0);
    std::vector<std::uint32_t> data(kDataWords);

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
    const std::size_t requested_bytes = kElapsedBytes + kReturnedBytes + kDataBytes;
    if (requested_bytes > kMaxAllocationBytes || requested_bytes > free_bytes / 2) {
        std::fprintf(stderr,
                     "requested %zu bytes exceeds the bounded allocation budget "
                     "(%zu bytes, %zu free)\n",
                     requested_bytes, kMaxAllocationBytes, free_bytes);
        return 3;
    }

    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr data_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device, kElapsedBytes));
    DRIVER_CHECK(cuMemAlloc(&returned_device, kReturnedBytes));
    DRIVER_CHECK(cuMemAlloc(&data_device, kDataBytes));
    void* arguments[] = {&elapsed_device, &returned_device, &data_device};
    bool elapsed_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed_device, 0xff, kElapsedBytes));
        DRIVER_CHECK(cuMemsetD8(returned_device, 0xa5, kReturnedBytes));
        DRIVER_CHECK(cuMemcpyHtoD(data_device, zeros.data(),
                                  zeros.size() * sizeof(std::uint32_t)));
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed_device,
                                  warps * sizeof(unsigned long long)));
        for (int warp = 0; warp < warps; ++warp) {
            elapsed_valid = elapsed_valid && samples[trial][warp] != UINT64_MAX &&
                            samples[trial][warp] > 0;
        }
    }
    DRIVER_CHECK(cuMemcpyDtoH(returned.data(), returned_device,
                              threads * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemcpyDtoH(data.data(), data_device,
                              data.size() * sizeof(std::uint32_t)));

    const std::uint64_t operations =
        static_cast<std::uint64_t>(threads) * generations;
    std::uint64_t data_sum = 0;
    for (std::uint32_t value : data) {
        data_sum += value;
    }
    std::uint64_t returned_total = 0;
    for (unsigned long long value : returned) {
        returned_total += value;
    }
    std::uint64_t groups;
    std::uint64_t operations_per_group;
    if (layout <= 1) {
        groups = threads;
        operations_per_group = generations;
    } else if (layout == 2) {
        groups = warps;
        operations_per_group = static_cast<std::uint64_t>(kWarpSize) * generations;
    } else {
        groups = 1;
        operations_per_group = operations;
    }
    const std::uint64_t expected_returned = returning
        ? groups * operations_per_group * (operations_per_group - 1) / 2
        : 0;
    const bool returned_present = !returning ||
        std::all_of(returned.begin(), returned.end(), [](unsigned long long value) {
            return value != 0xa5a5a5a5a5a5a5a5ull;
        });
    const bool correct = data_sum == operations && returned_total == expected_returned &&
                         elapsed_valid && returned_present;

    std::printf(
        "{\"variant\":\"%s\",\"warps\":%d,\"generations\":%d,"
        "\"layout\":%d,\"returning\":%s,\"samples_cycles\":[",
        argv[5], warps, generations, layout, returning ? "true" : "false");
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf(
        "],\"operations\":%llu,\"data_sum\":%llu,"
        "\"returned_sum\":%llu,\"expected_returned_sum\":%llu,"
        "\"correct\":%s,\"elapsed_samples_valid\":%s,"
        "\"returned_output_present\":%s,\"allocation_bytes\":%zu}\n",
        static_cast<unsigned long long>(operations),
        static_cast<unsigned long long>(data_sum),
        static_cast<unsigned long long>(returned_total),
        static_cast<unsigned long long>(expected_returned),
        correct ? "true" : "false", elapsed_valid ? "true" : "false",
        returned_present ? "true" : "false", requested_bytes);
    DRIVER_CHECK(cuMemFree(data_device));
    DRIVER_CHECK(cuMemFree(returned_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
