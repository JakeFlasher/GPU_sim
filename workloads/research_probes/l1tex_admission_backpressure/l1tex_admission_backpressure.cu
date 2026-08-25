#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
constexpr int kPrefixSteps = 128;
constexpr std::size_t kL2EvictionBytes = 32u * 1024u * 1024u;

enum TailKind : int {
    kControl = 0,
    kStore = 1,
    kReduction = 2,
    kAtomic = 3,
};

enum LayoutKind : int {
    kLineUnique = 0,
    kWordUnique = 1,
    kLineReuse = 2,
    kWordReuse = 3,
    kWarpWordRegions = 4,
    kWarpWordSharedRegion = 5,
};

enum PatternKind : int {
    kAllWarps = 0,
    kOnePerSmsp = 1,
    kSameSmsp = 2,
    kTwoSmsps = 3,
};

__device__ __forceinline__ std::uint32_t dependent_step(std::uint32_t value) {
    std::uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;\n"
                 : "=r"(next)
                 : "r"(value));
    return next;
}

std::uint32_t host_dependent_step(std::uint32_t value) {
    return value * 1664525u + 1013904223u;
}

__host__ __device__ __forceinline__ bool selected_warp(int pattern, int warp) {
    switch (pattern) {
        case kAllWarps: return true;
        case kOnePerSmsp: return warp < 4;
        case kSameSmsp: return warp == 0 || warp == 4;
        case kTwoSmsps: return warp == 0 || warp == 1 || warp == 4 || warp == 5;
        default: return false;
    }
}

__host__ __device__ __forceinline__ std::size_t target_word(
    int layout, int blocks, int warps, int depth, int active_lanes, int block,
    int warp, int operation, int lane) {
    (void)blocks;
    const std::size_t global_warp =
        static_cast<std::size_t>(block) * warps + warp;
    switch (layout) {
        case kLineUnique:
            return ((global_warp * depth + operation) * active_lanes + lane) *
                   kWarpSize;
        case kWordUnique:
            return (global_warp * depth + operation) * kWarpSize + lane;
        case kLineReuse:
            return (global_warp * active_lanes + lane) * kWarpSize;
        case kWordReuse: return global_warp * kWarpSize + lane;
        case kWarpWordRegions:
            return (global_warp * depth + operation) * 64;
        case kWarpWordSharedRegion:
            return (static_cast<std::size_t>(block) * depth + operation) * 64 +
                   warp;
        default: return 0;
    }
}

template <int Tail>
__global__ void admission_backpressure_tail(
    std::uint32_t* words, unsigned long long* timestamps,
    unsigned long long* checksums, int warps, int depth, int active_lanes,
    int layout, int gap_steps, int pattern) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::size_t global_warp =
        static_cast<std::size_t>(blockIdx.x) * warps + warp;
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool memory_active =
        selected_warp(pattern, warp) && lane < active_lanes;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(global_thread * 131 + lane);
    unsigned long long checksum = global_thread + 1;

#pragma unroll 1
    for (int step = 0; step < kPrefixSteps; ++step) {
        value = dependent_step(value);
    }
    __syncthreads();

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int operation = 0; operation < depth; ++operation) {
        value = dependent_step(value);
        std::size_t word = 0;
        if (lane < active_lanes) {
            word = target_word(layout, gridDim.x, warps, depth, active_lanes,
                               static_cast<int>(blockIdx.x), warp, operation,
                               lane);
            checksum ^= static_cast<unsigned long long>(word) + value;
        }
        if (memory_active) {
            if constexpr (Tail == kStore) {
                const std::uint32_t stored = value | 1u;
                asm volatile("st.global.wb.u32 [%0], %1;\n"
                             :
                             : "l"(words + word), "r"(stored)
                             : "memory");
            } else if constexpr (Tail == kReduction) {
                asm volatile("red.global.add.u32 [%0], 1;\n"
                             :
                             : "l"(words + word)
                             : "memory");
            } else if constexpr (Tail == kAtomic) {
                std::uint32_t prior;
                asm volatile("atom.global.add.u32 %0, [%1], 1;\n"
                             : "=r"(prior)
                             : "l"(words + word)
                             : "memory");
                checksum += prior;
            }
        }
#pragma unroll 1
        for (int gap = 0; gap < gap_steps; ++gap) {
            value = dependent_step(value);
        }
    }
    const unsigned long long after_issue = clock64();
    asm volatile("membar.gl;\n" : : : "memory");
    const unsigned long long after_fence = clock64();

    checksums[global_thread] = checksum ^ value;
    if (lane == 0) {
        timestamps[global_warp * 3 + 0] = start;
        timestamps[global_warp * 3 + 1] = after_issue;
        timestamps[global_warp * 3 + 2] = after_fence;
    }
}

const char* tail_name(int tail) {
    switch (tail) {
        case kControl: return "control";
        case kStore: return "store";
        case kReduction: return "red";
        case kAtomic: return "atom";
        default: return nullptr;
    }
}

const char* layout_name(int layout) {
    switch (layout) {
        case kLineUnique: return "line_unique";
        case kWordUnique: return "word_unique";
        case kLineReuse: return "line_reuse";
        case kWordReuse: return "word_reuse";
        case kWarpWordRegions: return "warp_word_regions";
        case kWarpWordSharedRegion: return "warp_word_shared_region";
        default: return nullptr;
    }
}

const char* pattern_name(int pattern) {
    switch (pattern) {
        case kAllWarps: return "all";
        case kOnePerSmsp: return "one_per_smsp";
        case kSameSmsp: return "same_smsp";
        case kTwoSmsps: return "two_smsps";
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

std::size_t allocation_words(int layout, int blocks, int warps, int depth,
                             int active_lanes) {
    const std::size_t global_warps =
        static_cast<std::size_t>(blocks) * warps;
    switch (layout) {
        case kLineUnique:
            return global_warps * depth * active_lanes * kWarpSize;
        case kWordUnique: return global_warps * depth * kWarpSize;
        case kLineReuse: return global_warps * active_lanes * kWarpSize;
        case kWordReuse: return global_warps * kWarpSize;
        case kWarpWordRegions: return global_warps * depth * 64;
        case kWarpWordSharedRegion:
            return static_cast<std::size_t>(blocks) * depth * 64;
        default: return 0;
    }
}

void launch_tail(int tail, int blocks, int warps, std::uint32_t* words,
                 unsigned long long* timestamps,
                 unsigned long long* checksums, int depth, int active_lanes,
                 int layout, int gap_steps, int pattern) {
    const int threads = warps * kWarpSize;
    switch (tail) {
        case kControl:
            admission_backpressure_tail<kControl><<<blocks, threads>>>(
                words, timestamps, checksums, warps, depth, active_lanes,
                layout, gap_steps, pattern);
            break;
        case kStore:
            admission_backpressure_tail<kStore><<<blocks, threads>>>(
                words, timestamps, checksums, warps, depth, active_lanes,
                layout, gap_steps, pattern);
            break;
        case kReduction:
            admission_backpressure_tail<kReduction><<<blocks, threads>>>(
                words, timestamps, checksums, warps, depth, active_lanes,
                layout, gap_steps, pattern);
            break;
        case kAtomic:
            admission_backpressure_tail<kAtomic><<<blocks, threads>>>(
                words, timestamps, checksums, warps, depth, active_lanes,
                layout, gap_steps, pattern);
            break;
        default: std::abort();
    }
}

unsigned long long median(std::vector<unsigned long long> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] / 2) + (values[middle] / 2) +
           ((values[middle - 1] & 1u) && (values[middle] & 1u));
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::fprintf(
            stderr,
            "usage: %s TAIL LAYOUT BLOCKS WARPS DEPTH ACTIVE_LANES GAP_STEPS PATTERN TRIALS\n",
            argv[0]);
        return 2;
    }
    int tail = -1;
    int layout = -1;
    int pattern = -1;
    const int blocks = std::atoi(argv[3]);
    const int warps = std::atoi(argv[4]);
    const int depth = std::atoi(argv[5]);
    const int active_lanes = std::atoi(argv[6]);
    const int gap_steps = std::atoi(argv[7]);
    const int trials = std::atoi(argv[9]);
    if (!parse_named(argv[1], kControl, kAtomic, tail_name, &tail) ||
        !parse_named(argv[2], kLineUnique, kWarpWordSharedRegion, layout_name,
                     &layout) ||
        !parse_named(argv[8], kAllWarps, kTwoSmsps, pattern_name, &pattern) ||
        blocks < 1 || blocks > 336 || warps < 1 || warps > kMaxWarps ||
        depth < 1 || depth > 128 || active_lanes < 1 ||
        active_lanes > kWarpSize || gap_steps < 0 || gap_steps > 256 ||
        trials < 1 || trials > 32 ||
        (pattern != kAllWarps && warps != kMaxWarps)) {
        return 2;
    }

    const std::size_t word_count =
        allocation_words(layout, blocks, warps, depth, active_lanes);
    const std::size_t warp_count = static_cast<std::size_t>(blocks) * warps;
    const std::size_t thread_count = warp_count * kWarpSize;
    std::uint32_t* words = nullptr;
    unsigned long long* timestamps = nullptr;
    unsigned long long* checksums = nullptr;
    std::uint32_t* l2_eviction = nullptr;
    CUDA_CHECK(cudaMalloc(&words, word_count * sizeof(*words)));
    CUDA_CHECK(cudaMalloc(&timestamps, warp_count * 3 * sizeof(*timestamps)));
    CUDA_CHECK(cudaMalloc(&checksums, thread_count * sizeof(*checksums)));
    CUDA_CHECK(cudaMalloc(&l2_eviction, kL2EvictionBytes));

    bool correct = true;
    unsigned long long checksum = 0;
    std::vector<unsigned long long> issue_samples;
    std::vector<unsigned long long> fence_samples;
    int selected_warps_per_block = 0;
    for (int warp = 0; warp < warps; ++warp) {
        selected_warps_per_block += selected_warp(pattern, warp) ? 1 : 0;
    }
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaMemset(words, 0, word_count * sizeof(*words)));
        CUDA_CHECK(cudaMemset(timestamps, 0,
                              warp_count * 3 * sizeof(*timestamps)));
        CUDA_CHECK(cudaMemset(checksums, 0,
                              thread_count * sizeof(*checksums)));
        CUDA_CHECK(cudaMemset(l2_eviction, trial + 1, kL2EvictionBytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStart());
        launch_tail(tail, blocks, warps, words, timestamps, checksums, depth,
                    active_lanes, layout, gap_steps, pattern);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());

        std::vector<unsigned long long> observed_timestamps(warp_count * 3);
        std::vector<unsigned long long> observed_checksums(thread_count);
        CUDA_CHECK(cudaMemcpy(observed_timestamps.data(), timestamps,
                              observed_timestamps.size() * sizeof(timestamps[0]),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(observed_checksums.data(), checksums,
                              observed_checksums.size() * sizeof(checksums[0]),
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint32_t> observed_words;
        if (trials > 1 && tail != kControl) {
            observed_words.resize(word_count);
            CUDA_CHECK(cudaMemcpy(observed_words.data(), words,
                                  observed_words.size() * sizeof(words[0]),
                                  cudaMemcpyDeviceToHost));
            for (int block = 0; block < blocks; ++block) {
                for (int warp = 0; warp < warps; ++warp) {
                    if (!selected_warp(pattern, warp)) {
                        continue;
                    }
                    for (int lane = 0; lane < active_lanes; ++lane) {
                        const std::size_t global_thread =
                            (static_cast<std::size_t>(block) * warps + warp) *
                                kWarpSize +
                            lane;
                        std::uint32_t value =
                            0x9e3779b9u ^ static_cast<std::uint32_t>(
                                              global_thread * 131 + lane);
                        for (int step = 0; step < kPrefixSteps; ++step) {
                            value = host_dependent_step(value);
                        }
                        for (int operation = 0; operation < depth; ++operation) {
                            value = host_dependent_step(value);
                            const std::size_t word = target_word(
                                layout, blocks, warps, depth, active_lanes,
                                block, warp, operation, lane);
                            if (tail == kStore) {
                                if (layout == kWarpWordRegions ||
                                    layout == kWarpWordSharedRegion) {
                                    correct = correct && observed_words[word] != 0;
                                } else if (layout == kLineUnique ||
                                           layout == kWordUnique ||
                                           operation == depth - 1) {
                                    correct = correct &&
                                              observed_words[word] == (value | 1u);
                                }
                            } else {
                                std::uint32_t expected = 1;
                                if (layout == kLineReuse ||
                                    layout == kWordReuse) {
                                    expected = static_cast<std::uint32_t>(depth);
                                } else if (layout == kWarpWordRegions ||
                                           layout == kWarpWordSharedRegion) {
                                    expected =
                                        static_cast<std::uint32_t>(active_lanes);
                                }
                                correct = correct && observed_words[word] == expected;
                            }
                            for (int gap = 0; gap < gap_steps; ++gap) {
                                value = host_dependent_step(value);
                            }
                        }
                    }
                }
            }
        }
        for (std::size_t index = 0; index < observed_checksums.size(); ++index) {
            checksum = checksum * 131u + observed_checksums[index];
        }
        for (int block = 0; block < blocks; ++block) {
            for (int warp = 0; warp < warps; ++warp) {
                const std::size_t global_warp =
                    static_cast<std::size_t>(block) * warps + warp;
                const unsigned long long start =
                    observed_timestamps[global_warp * 3 + 0];
                const unsigned long long after_issue =
                    observed_timestamps[global_warp * 3 + 1];
                const unsigned long long after_fence =
                    observed_timestamps[global_warp * 3 + 2];
                correct = correct && after_issue >= start &&
                          after_fence >= after_issue;
                checksum = checksum * 131u + start;
                checksum = checksum * 131u + after_issue;
                checksum = checksum * 131u + after_fence;
                if (!selected_warp(pattern, warp)) {
                    continue;
                }
                issue_samples.push_back(after_issue - start);
                fence_samples.push_back(after_fence - after_issue);
                if (tail == kControl) {
                    continue;
                }
                const std::size_t word = target_word(
                    layout, blocks, warps, depth, active_lanes, block, warp,
                    depth - 1, 0);
                std::uint32_t observed = 0;
                CUDA_CHECK(cudaMemcpy(&observed, words + word, sizeof(observed),
                                      cudaMemcpyDeviceToHost));
                if (tail == kStore) {
                    correct = correct && observed != 0;
                } else {
                    std::uint32_t expected = 1;
                    if (layout == kLineReuse || layout == kWordReuse) {
                        expected = static_cast<std::uint32_t>(depth);
                    } else if (layout == kWarpWordRegions ||
                               layout == kWarpWordSharedRegion) {
                        expected = static_cast<std::uint32_t>(active_lanes);
                    }
                    correct = correct && observed == expected;
                }
                checksum = checksum * 131u + observed;
            }
        }
    }

    if (issue_samples.empty() || fence_samples.empty()) {
        correct = false;
        issue_samples.push_back(0);
        fence_samples.push_back(0);
    }
    const auto [issue_min, issue_max] =
        std::minmax_element(issue_samples.begin(), issue_samples.end());
    const auto [fence_min, fence_max] =
        std::minmax_element(fence_samples.begin(), fence_samples.end());
    const unsigned long long memory_instructions =
        tail == kControl
            ? 0
            : static_cast<unsigned long long>(blocks) *
                  selected_warps_per_block * depth;
    std::printf(
        "{\"tail\":\"%s\",\"layout\":\"%s\",\"blocks\":%d,"
        "\"warps\":%d,\"depth\":%d,\"active_lanes\":%d,"
        "\"gap_steps\":%d,\"pattern\":\"%s\","
        "\"selected_warps_per_block\":%d,\"prefix_steps\":%d,"
        "\"memory_instructions\":%llu,\"allocation_bytes\":%zu,"
        "\"l2_eviction_bytes\":%zu,"
        "\"issue_cycles_min\":%llu,\"issue_cycles_median\":%llu,"
        "\"issue_cycles_max\":%llu,\"fence_cycles_min\":%llu,"
        "\"fence_cycles_median\":%llu,\"fence_cycles_max\":%llu,"
        "\"trials\":%d,\"checksum\":%llu,\"correct\":%s}\n",
        tail_name(tail), layout_name(layout), blocks, warps, depth,
        active_lanes, gap_steps, pattern_name(pattern),
        selected_warps_per_block, kPrefixSteps, memory_instructions,
        word_count * sizeof(*words), kL2EvictionBytes, *issue_min,
        median(issue_samples),
        *issue_max, *fence_min, median(fence_samples), *fence_max, trials,
        checksum, correct ? "true" : "false");

    CUDA_CHECK(cudaFree(checksums));
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(words));
    CUDA_CHECK(cudaFree(l2_eviction));
    return correct ? 0 : 1;
}
