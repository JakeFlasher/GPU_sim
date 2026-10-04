#include <cuda.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
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
constexpr int kMaxThreads = kWarpSize * kMaxWarps;
constexpr std::uint32_t kWarpPitchBytes = 16640;
constexpr const char* kFunction32 = "_Z18atomic_alias_probePyS_PjPKj";
constexpr const char* kFunction64 = "_Z18atomic_alias_probePyS_S_PKj";

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

enum class Pattern {
    Dense = 0,
    GroupStride = 1,
    CrossWarp = 2,
};

int main(int argc, char** argv) {
    if (argc != 12) {
        std::fprintf(
            stderr,
            "usage: %s CUBIN WARPS GENERATIONS TRIALS VARIANT WIDTH "
            "RETURNING PATTERN ALIAS STRIDE_BYTES SCATTERED\n",
            argv[0]);
        return 2;
    }
    int warps = 0, generations = 0, trials = 0, width = 0, returning_int = 0;
    int pattern_raw = 0, alias = 0, scattered_int = 0;
    if (!parse_int(argv[2], &warps) || !parse_int(argv[3], &generations) ||
        !parse_int(argv[4], &trials) || !parse_int(argv[6], &width) ||
        !parse_int(argv[7], &returning_int) || !parse_int(argv[8], &pattern_raw) ||
        !parse_int(argv[9], &alias) || !parse_int(argv[11], &scattered_int)) return 2;
    const bool returning = returning_int != 0;
    const std::uint32_t stride_bytes =
        static_cast<std::uint32_t>(std::strtoul(argv[10], nullptr, 10));
    const bool scattered = scattered_int != 0;
    if (warps < 1 || warps > kMaxWarps ||
        generations != 1024 || trials < 1 || trials > 32 ||
        (returning_int != 0 && returning_int != 1) ||
        (width != 32 && width != 64) || pattern_raw < 0 || pattern_raw > 2 ||
        alias < 1 || alias > kWarpSize || kWarpSize % alias != 0) {
        return 2;
    }
    const Pattern pattern = static_cast<Pattern>(pattern_raw);
    const std::uint32_t word_bytes = static_cast<std::uint32_t>(width / 8);
    if ((stride_bytes % word_bytes) != 0 ||
        (pattern == Pattern::CrossWarp && alias != kWarpSize)) {
        return 2;
    }

    const int threads = warps * kWarpSize;
    std::vector<std::uint32_t> indices(threads);
    std::uint32_t maximum_byte_offset = 0;
    for (int thread = 0; thread < threads; ++thread) {
        const int warp = thread / kWarpSize;
        const int lane = thread % kWarpSize;
        std::uint32_t byte_offset;
        if (pattern == Pattern::CrossWarp) {
            byte_offset = static_cast<std::uint32_t>(warp) * stride_bytes;
        } else {
            const int groups = kWarpSize / alias;
            const int group = scattered ? lane % groups : lane / alias;
            const std::uint32_t group_stride =
                pattern == Pattern::Dense ? word_bytes : stride_bytes;
            byte_offset = static_cast<std::uint32_t>(warp) * kWarpPitchBytes +
                          static_cast<std::uint32_t>(group) * group_stride;
        }
        if ((byte_offset % word_bytes) != 0) {
            return 2;
        }
        indices[thread] = byte_offset / word_bytes;
        maximum_byte_offset = std::max(maximum_byte_offset, byte_offset);
    }
    const std::size_t words = maximum_byte_offset / word_bytes + 1;
    const std::size_t data_bytes = words * word_bytes;
    if (data_bytes > (std::size_t{512} << 20)) return 3;
    std::vector<unsigned long long> elapsed(warps);
    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    std::vector<unsigned long long> returned(threads);
    const std::size_t backing_words = (data_bytes + sizeof(std::uint64_t) - 1) /
                                      sizeof(std::uint64_t);
    std::vector<std::uint64_t> zeros(backing_words, 0);
    std::vector<std::uint64_t> data(backing_words);

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(
        &function, module, width == 32 ? kFunction32 : kFunction64));

    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr data_device;
    CUdeviceptr indices_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device, kMaxWarps * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&returned_device,
                            kMaxThreads * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&data_device, data_bytes));
    DRIVER_CHECK(cuMemAlloc(&indices_device, indices.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(indices_device, indices.data(),
                              indices.size() * sizeof(std::uint32_t)));
    void* arguments[] = {
        &elapsed_device, &returned_device, &data_device, &indices_device};
    bool elapsed_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed_device, 0xff,
                                kMaxWarps * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(returned_device, 0xa5,
                                kMaxThreads * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemcpyHtoD(data_device, zeros.data(), data_bytes));
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
    DRIVER_CHECK(cuMemcpyDtoH(data.data(), data_device, data_bytes));

    const std::uint64_t operations =
        static_cast<std::uint64_t>(threads) * generations;
    std::uint64_t data_sum = 0;
    if (width == 32) {
        const auto* values = reinterpret_cast<const std::uint32_t*>(data.data());
        for (std::size_t index = 0; index < words; ++index) {
            data_sum += values[index];
        }
    } else {
        const auto* values =
            reinterpret_cast<const unsigned long long*>(data.data());
        for (std::size_t index = 0; index < words; ++index) {
            data_sum += values[index];
        }
    }
    std::uint64_t returned_total = 0;
    for (unsigned long long value : returned) {
        returned_total += value;
    }
    std::unordered_map<std::uint32_t, std::uint64_t> threads_per_word;
    for (std::uint32_t index : indices) {
        ++threads_per_word[index];
    }
    std::uint64_t expected_returned = 0;
    if (returning) {
        for (const auto& [index, count] : threads_per_word) {
            (void)index;
            const std::uint64_t group_operations = count * generations;
            expected_returned +=
                group_operations * (group_operations - 1) / 2;
        }
    }
    const bool returned_present = !returning ||
        std::all_of(returned.begin(), returned.end(), [](unsigned long long value) {
            return value != 0xa5a5a5a5a5a5a5a5ull;
        });
    const bool correct = data_sum == operations && returned_total == expected_returned &&
                         elapsed_valid && returned_present;

    std::printf(
        "{\"variant\":\"%s\",\"warps\":%d,\"generations\":%d,"
        "\"width\":%d,\"returning\":%s,\"pattern\":%d,"
        "\"alias\":%d,\"stride_bytes\":%u,\"scattered\":%s,"
        "\"samples_cycles\":[",
        argv[5], warps, generations, width, returning ? "true" : "false",
        pattern_raw, alias, stride_bytes, scattered ? "true" : "false");
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
        "\"unique_words\":%zu,\"correct\":%s,\"elapsed_samples_valid\":%s,"
        "\"returned_output_present\":%s,\"allocation_bytes\":%zu}\n",
        static_cast<unsigned long long>(operations),
        static_cast<unsigned long long>(data_sum),
        static_cast<unsigned long long>(returned_total),
        static_cast<unsigned long long>(expected_returned),
        threads_per_word.size(), correct ? "true" : "false",
        elapsed_valid ? "true" : "false", returned_present ? "true" : "false",
        data_bytes);
    DRIVER_CHECK(cuMemFree(indices_device));
    DRIVER_CHECK(cuMemFree(data_device));
    DRIVER_CHECK(cuMemFree(returned_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
