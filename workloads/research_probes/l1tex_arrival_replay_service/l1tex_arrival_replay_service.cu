#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::size_t allocation_words(int layout, int blocks, int warps, int depth,
                             int active_lanes) {
    const std::size_t warp_operations =
        static_cast<std::size_t>(blocks) * warps * depth;
    return layout == kLine ? warp_operations * active_lanes * kWarpSize
                           : warp_operations * kWarpSize;
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
    const int blocks = std::atoi(argv[3]);
    const int warps = std::atoi(argv[4]);
    const int depth = std::atoi(argv[5]);
    const int active_lanes = std::atoi(argv[6]);
    const int gap_steps = std::atoi(argv[7]);
    const int trials = std::atoi(argv[8]);
    if (!parse_named(argv[1], kControl, kReduction, tail_name, &tail) ||
        !parse_named(argv[2], kLine, kWord, layout_name, &layout) ||
        blocks < 1 || blocks > 336 || warps < 1 || warps > kMaxWarps ||
        depth < 1 || depth > kMaxDepth || active_lanes < 1 ||
        active_lanes > kWarpSize || gap_steps < 0 || gap_steps > 256 ||
        trials < 1 || trials > 32) {
        return 2;
    }

    const std::size_t word_count =
        allocation_words(layout, blocks, warps, depth, active_lanes);
    std::vector<std::uint32_t> initial(word_count);
    for (std::size_t word = 0; word < word_count; ++word) {
        initial[word] = initial_value(word);
    }
    std::uint32_t* words = nullptr;
    CUDA_CHECK(cudaMalloc(&words, word_count * sizeof(*words)));

    bool correct = true;
    std::uint64_t checksum = 0;
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaMemcpy(words, initial.data(), word_count * sizeof(*words),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStart());
        launch_tail(tail, blocks, warps, words, depth, active_lanes, layout,
                    gap_steps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());

        std::vector<std::uint32_t> observed(word_count);
        CUDA_CHECK(cudaMemcpy(observed.data(), words,
                              word_count * sizeof(*words),
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
        "\"allocation_bytes\":%zu,\"checksum\":%llu,\"correct\":%s}\n",
        tail_name(tail), layout_name(layout), blocks, warps, depth,
        active_lanes, gap_steps, kWorkIterations,
        static_cast<unsigned long long>(ideal_wavefronts), trials,
        word_count * sizeof(std::uint32_t),
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(words));
    return correct ? 0 : 1;
}
