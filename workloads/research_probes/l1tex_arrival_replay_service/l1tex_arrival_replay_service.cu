#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <array>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 8;
constexpr int kWorkIterations = 256;
constexpr int kMaxDepth = 128;
constexpr std::size_t kMaxAllocationBytes = 512ull * 1024ull * 1024ull;

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

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (result == nullptr ||
        (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

enum TailKind : int {
    kControl = 0,
    kLoad = 1,
    kStore = 2,
    kReduction = 3,
};

enum LayoutKind : int {
    kLine = 0,
    kWord = 1,
};

__device__ __forceinline__ std::uint32_t dependent_step(std::uint32_t value) {
    std::uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;\n"
                 : "=r"(next)
                 : "r"(value));
    return next;
}

__host__ __device__ __forceinline__ std::size_t tail_word(
    int layout, int warps, int depth, int active_lanes, int block, int warp,
    int operation, int lane) {
    const std::size_t global_warp =
        static_cast<std::size_t>(block) * warps + warp;
    if (layout == kLine) {
        return ((global_warp * depth + operation) * active_lanes + lane) *
               kWarpSize;
    }
    return (global_warp * depth + operation) * kWarpSize + lane;
}

template <int Tail>
__global__ void arrival_replay_tail(std::uint32_t* words, int warps, int depth,
                                    int active_lanes, int layout,
                                    int gap_steps_per_warp) {
    __shared__ std::uint32_t work_sink[kMaxWarps * kWarpSize];
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(
                          (blockIdx.x * kMaxWarps + warp) * 131 + lane);

#pragma unroll 1
    for (int iteration = 0; iteration < kWorkIterations; ++iteration) {
        value = dependent_step(value);
    }
    __syncthreads();

    const int gap_steps = warp * gap_steps_per_warp;
#pragma unroll 1
    for (int step = 0; step < gap_steps; ++step) {
        value = dependent_step(value);
    }

#pragma unroll 1
    for (int operation = 0; operation < depth; ++operation) {
        value = dependent_step(value);
        if (lane < active_lanes) {
            const std::size_t word = tail_word(
                layout, warps, depth, active_lanes,
                static_cast<int>(blockIdx.x), warp, operation, lane);
            if constexpr (Tail == kLoad) {
                std::uint32_t loaded;
                asm volatile("ld.volatile.global.u32 %0, [%1];\n"
                             : "=r"(loaded)
                             : "l"(words + word)
                             : "memory");
                asm volatile("" : : "r"(loaded));
            } else if constexpr (Tail == kStore) {
                asm volatile("st.global.wb.u32 [%0], %1;\n"
                             :
                             : "l"(words + word), "r"(value)
                             : "memory");
            } else if constexpr (Tail == kReduction) {
                asm volatile("red.global.add.u32 [%0], 1;\n"
                             :
                             : "l"(words + word)
                             : "memory");
            }
        }
    }

    const unsigned shared_address = static_cast<unsigned>(
        __cvta_generic_to_shared(work_sink + threadIdx.x));
    asm volatile("st.volatile.shared.u32 [%0], %1;\n"
                 :
                 : "r"(shared_address), "r"(value)
                 : "memory");
}

const char* tail_name(int tail) {
    switch (tail) {
        case kControl: return "control";
        case kLoad: return "load";
        case kStore: return "store";
        case kReduction: return "red";
        default: return nullptr;
    }
}

const char* layout_name(int layout) {
    switch (layout) {
        case kLine: return "line";
        case kWord: return "word";
        default: return nullptr;
    }
}

template <typename Name>
bool parse_named(const char* value, int begin, int end, Name name, int* result) {
    for (int candidate = begin; candidate <= end; ++candidate) {
        if (std::strcmp(value, name(candidate)) == 0) {
            *result = candidate;
            return true;
        }
    }
    return false;
}

void launch_tail(int tail, int blocks, int warps, std::uint32_t* words,
                 int depth, int active_lanes, int layout,
                 int gap_steps_per_warp) {
    const int threads = warps * kWarpSize;
    switch (tail) {
        case kControl:
            arrival_replay_tail<kControl><<<blocks, threads>>>(
                words, warps, depth, active_lanes, layout,
                gap_steps_per_warp);
            break;
        case kLoad:
            arrival_replay_tail<kLoad><<<blocks, threads>>>(
                words, warps, depth, active_lanes, layout,
                gap_steps_per_warp);
            break;
        case kStore:
            arrival_replay_tail<kStore><<<blocks, threads>>>(
                words, warps, depth, active_lanes, layout,
                gap_steps_per_warp);
            break;
        case kReduction:
            arrival_replay_tail<kReduction><<<blocks, threads>>>(
                words, warps, depth, active_lanes, layout,
                gap_steps_per_warp);
            break;
        default: std::abort();
    }
}

bool allocation_words(int layout, int blocks, int warps, int depth,
                      int active_lanes, std::size_t* result) {
    std::size_t warp_operations = 0;
    std::size_t line_words = layout == kLine
        ? static_cast<std::size_t>(active_lanes) * kWarpSize
        : static_cast<std::size_t>(kWarpSize);
    return checked_mul(static_cast<std::size_t>(blocks),
                       static_cast<std::size_t>(warps), &warp_operations) &&
           checked_mul(warp_operations, static_cast<std::size_t>(depth),
                       &warp_operations) &&
           checked_mul(warp_operations, line_words, result);
}

std::uint32_t initial_value(std::size_t word) {
    return 0x6a09e667u ^ static_cast<std::uint32_t>(word * 0x27d4eb2du);
}

std::uint32_t host_dependent_step(std::uint32_t value) {
    return value * 1664525u + 1013904223u;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(
            stderr,
            "usage: %s TAIL LAYOUT BLOCKS WARPS DEPTH ACTIVE_LANES GAP_STEPS TRIALS\n",
            argv[0]);
        return 2;
    }
    int tail = -1;
    int layout = -1;
    int blocks = 0;
    int warps = 0;
    int depth = 0;
    int active_lanes = 0;
    int gap_steps = 0;
    int trials = 0;
    if (!parse_int(argv[3], &blocks) || !parse_int(argv[4], &warps) ||
        !parse_int(argv[5], &depth) || !parse_int(argv[6], &active_lanes) ||
        !parse_int(argv[7], &gap_steps) || !parse_int(argv[8], &trials)) {
        std::fprintf(stderr, "all numeric arguments must be signed decimal integers\n");
        return 2;
    }
    if (!parse_named(argv[1], kControl, kReduction, tail_name, &tail) ||
        !parse_named(argv[2], kLine, kWord, layout_name, &layout) ||
        blocks < 1 || blocks > 336 || warps < 1 || warps > kMaxWarps ||
        depth < 1 || depth > kMaxDepth || active_lanes < 1 ||
        active_lanes > kWarpSize || gap_steps < 0 || gap_steps > 256 ||
        trials < 1 || trials > 32) {
        std::fprintf(stderr, "arguments exceed the bounded probe contract\n");
        return 2;
    }

    std::size_t word_count = 0;
    if (!allocation_words(layout, blocks, warps, depth, active_lanes,
                         &word_count)) {
        std::fprintf(stderr, "word-count arithmetic overflows size_t\n");
        return 2;
    }
    std::size_t allocation_bytes = 0;
    if (!checked_mul(word_count, sizeof(std::uint32_t), &allocation_bytes)) {
        std::fprintf(stderr, "allocation size overflows size_t\n");
        return 2;
    }
    std::size_t free_bytes = 0;
    std::size_t device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &device_bytes));
    // Leave room for the CUDA context, profiler, and allocator metadata.
    if (allocation_bytes > kMaxAllocationBytes || allocation_bytes > free_bytes / 2) {
        std::fprintf(stderr,
                     "requested %zu bytes exceeds the bounded allocation budget "
                     "(%zu bytes, %zu free)\n",
                     allocation_bytes, kMaxAllocationBytes, free_bytes);
        return 3;
    }

    std::vector<std::uint32_t> initial;
    try {
        initial.resize(word_count);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "host allocation of %zu bytes failed\n", allocation_bytes);
        return 3;
    }
    for (std::size_t word = 0; word < word_count; ++word) {
        initial[word] = initial_value(word);
    }
    std::uint32_t* words = nullptr;
    CUDA_CHECK(cudaMalloc(&words, allocation_bytes));

    bool correct = true;
    std::uint64_t checksum = 0;
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaMemcpy(words, initial.data(), allocation_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStart());
        launch_tail(tail, blocks, warps, words, depth, active_lanes, layout,
                    gap_steps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());

        std::vector<std::uint32_t> observed(word_count);
        CUDA_CHECK(cudaMemcpy(observed.data(), words, allocation_bytes,
                              cudaMemcpyDeviceToHost));
        for (int block = 0; block < blocks; ++block) {
            for (int warp = 0; warp < warps; ++warp) {
                std::array<std::uint32_t, kWarpSize> expected_values{};
                for (int lane = 0; lane < active_lanes; ++lane) {
                    std::uint32_t value =
                        0x9e3779b9u ^ static_cast<std::uint32_t>(
                                          (block * kMaxWarps + warp) * 131 +
                                          lane);
                    for (int step = 0; step < kWorkIterations; ++step) {
                        value = host_dependent_step(value);
                    }
                    for (int step = 0; step < warp * gap_steps; ++step) {
                        value = host_dependent_step(value);
                    }
                    expected_values[lane] = value;
                }
                for (int operation = 0; operation < depth; ++operation) {
                    for (int lane = 0; lane < active_lanes; ++lane) {
                        expected_values[lane] =
                            host_dependent_step(expected_values[lane]);
                        const std::size_t word = tail_word(
                            layout, warps, depth, active_lanes, block, warp,
                            operation, lane);
                        const std::uint32_t initial_word = initial_value(word);
                        if (tail == kStore) {
                            correct = correct &&
                                      observed[word] == expected_values[lane];
                        } else if (tail == kReduction) {
                            correct = correct && observed[word] == initial_word + 1;
                        } else {
                            correct = correct && observed[word] == initial_word;
                        }
                        checksum = checksum * 131u + observed[word];
                    }
                }
            }
        }
    }

    const int ideal_wavefronts_per_instruction =
        layout == kLine ? (active_lanes + 3) / 4 : 1;
    const std::uint64_t ideal_wavefronts =
        static_cast<std::uint64_t>(blocks) * warps * depth *
        ideal_wavefronts_per_instruction;
    std::printf(
        "{\"tail\":\"%s\",\"layout\":\"%s\",\"blocks\":%d,"
        "\"warps\":%d,\"depth\":%d,\"active_lanes\":%d,"
        "\"gap_steps_per_warp\":%d,\"work_iterations\":%d,"
        "\"ideal_wavefronts\":%llu,\"trials\":%d,"
        "\"allocation_bytes\":%zu,\"checksum\":%llu,\"correct\":%s,"
        "\"dependent_work_observed\":%s}\n",
        tail_name(tail), layout_name(layout), blocks, warps, depth,
        active_lanes, gap_steps, kWorkIterations,
        static_cast<unsigned long long>(ideal_wavefronts), trials,
        allocation_bytes,
        static_cast<unsigned long long>(checksum), correct ? "true" : "false",
        tail == kControl ? "false" : "true");

    CUDA_CHECK(cudaFree(words));
    return correct ? 0 : 1;
}
