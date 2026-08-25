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
constexpr const char* kFunction = "_Z12atomic_probePyS_Pj";

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: %s CUBIN WARPS GENERATIONS TRIALS VARIANT LAYOUT "
                     "RETURNING\n",
                     argv[0]);
        return 2;
    }
    const int warps = std::atoi(argv[2]);
    const int generations = std::atoi(argv[3]);
    const int trials = std::atoi(argv[4]);
    const int layout = std::atoi(argv[6]);
    const bool returning = std::atoi(argv[7]) != 0;
    if (warps < 1 || warps > kMaxWarps ||
        (generations != 64 && generations != 1024) ||
        trials < 1 || trials > 32 || layout < 0 || layout > 3) {
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

    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr data_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&returned_device,
                            kMaxThreads * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&data_device,
                            kDataWords * sizeof(std::uint32_t)));
    void* arguments[] = {&elapsed_device, &returned_device, &data_device};
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemcpyHtoD(data_device, zeros.data(),
                                  zeros.size() * sizeof(std::uint32_t)));
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed_device,
                                  warps * sizeof(unsigned long long)));
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
    const bool correct = data_sum == operations && returned_total == expected_returned;

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
        "\"correct\":%s}\n",
        static_cast<unsigned long long>(operations),
        static_cast<unsigned long long>(data_sum),
        static_cast<unsigned long long>(returned_total),
        static_cast<unsigned long long>(expected_returned),
        correct ? "true" : "false");
    return correct ? 0 : 1;
}
