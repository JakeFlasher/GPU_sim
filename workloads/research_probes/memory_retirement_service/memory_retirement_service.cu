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
constexpr int kWorkIterations = 256;
constexpr int kMaxDepth = 64;

enum TailKind : int {
    kControl = 0,
    kLoadDiscard = 1,
    kStore = 2,
    kReduction = 3,
};

enum FenceKind : int {
    kNoFence = 0,
    kDeviceFence = 1,
    kSystemFence = 2,
};

enum LayoutKind : int {
    kLineStream = 0,
    kLineReuse = 1,
    kWordStream = 2,
    kSameStream = 3,
};

__device__ __forceinline__ std::uint32_t dependent_step(std::uint32_t value) {
    std::uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;\n"
                 : "=r"(next)
                 : "r"(value));
    return next;
}

__host__ __device__ __forceinline__ std::size_t tail_word(
    int layout, int blocks, int depth_index, int block, int lane) {
    const std::size_t block_lane =
        static_cast<std::size_t>(block) * kWarpSize + lane;
    switch (layout) {
        case kLineStream:
            return (static_cast<std::size_t>(depth_index) * blocks * kWarpSize +
                    block_lane) *
                   kWarpSize;
        case kLineReuse: return block_lane * kWarpSize;
        case kWordStream:
            return (static_cast<std::size_t>(depth_index) * blocks + block) *
                       kWarpSize +
                   lane;
        case kSameStream:
            return (static_cast<std::size_t>(depth_index) * blocks + block) *
                   kWarpSize;
        default: return 0;
    }
}

template <int Fence>
__device__ __forceinline__ void apply_fence() {
    if constexpr (Fence == kDeviceFence) {
        __threadfence();
    } else if constexpr (Fence == kSystemFence) {
        __threadfence_system();
    }
}

template <int Tail, int Fence>
__global__ __launch_bounds__(kWarpSize)
void retirement_tail(std::uint32_t* words, int depth, int active_lanes,
                     int layout) {
    __shared__ std::uint32_t work_sink[kWarpSize];
    const int lane = static_cast<int>(threadIdx.x);
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(blockIdx.x * 131 + lane);

#pragma unroll 1
    for (int iteration = 0; iteration < kWorkIterations; ++iteration) {
        value = dependent_step(value);
    }

#pragma unroll 1
    for (int operation = 0; operation < depth; ++operation) {
        value = dependent_step(value);
        if (lane < active_lanes) {
            const std::size_t word = tail_word(
                layout, static_cast<int>(gridDim.x), operation,
                static_cast<int>(blockIdx.x), lane);
            if constexpr (Tail == kLoadDiscard) {
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

    apply_fence<Fence>();
    const unsigned shared_address = static_cast<unsigned>(
        __cvta_generic_to_shared(work_sink + lane));
    asm volatile("st.volatile.shared.u32 [%0], %1;\n"
                 :
                 : "r"(shared_address), "r"(value)
                 : "memory");
}

const char* tail_name(int tail) {
    switch (tail) {
        case kControl: return "control";
        case kLoadDiscard: return "load";
        case kStore: return "store";
        case kReduction: return "red";
        default: return nullptr;
    }
}

const char* fence_name(int fence) {
    switch (fence) {
        case kNoFence: return "none";
        case kDeviceFence: return "device";
        case kSystemFence: return "system";
        default: return nullptr;
    }
}

const char* layout_name(int layout) {
    switch (layout) {
        case kLineStream: return "line_stream";
        case kLineReuse: return "line_reuse";
        case kWordStream: return "word_stream";
        case kSameStream: return "same_stream";
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

template <int Tail>
void launch_fence(int fence, int blocks, std::uint32_t* words, int depth,
                  int active_lanes, int layout) {
    switch (fence) {
        case kNoFence:
            retirement_tail<Tail, kNoFence><<<blocks, kWarpSize>>>(
                words, depth, active_lanes, layout);
            break;
        case kDeviceFence:
            retirement_tail<Tail, kDeviceFence><<<blocks, kWarpSize>>>(
                words, depth, active_lanes, layout);
            break;
        case kSystemFence:
            retirement_tail<Tail, kSystemFence><<<blocks, kWarpSize>>>(
                words, depth, active_lanes, layout);
            break;
        default: std::abort();
    }
}

void launch_tail(int tail, int fence, int blocks, std::uint32_t* words,
                 int depth, int active_lanes, int layout) {
    switch (tail) {
        case kControl:
            launch_fence<kControl>(fence, blocks, words, depth, active_lanes,
                                   layout);
            break;
        case kLoadDiscard:
            launch_fence<kLoadDiscard>(fence, blocks, words, depth,
                                       active_lanes, layout);
            break;
        case kStore:
            launch_fence<kStore>(fence, blocks, words, depth, active_lanes,
                                 layout);
            break;
        case kReduction:
            launch_fence<kReduction>(fence, blocks, words, depth,
                                     active_lanes, layout);
            break;
        default: std::abort();
    }
}

std::size_t allocation_words(int blocks, int depth) {
    return static_cast<std::size_t>(blocks) * kWarpSize * kWarpSize * depth;
}

std::uint32_t initial_value(std::size_t word) {
    return 0x6a09e667u ^ static_cast<std::uint32_t>(word * 0x27d4eb2du);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(
            stderr,
            "usage: %s TAIL FENCE LAYOUT BLOCKS DEPTH ACTIVE_LANES TRIALS\n",
            argv[0]);
        return 2;
    }
    int tail = -1;
    int fence = -1;
    int layout = -1;
    const int blocks = std::atoi(argv[4]);
    const int depth = std::atoi(argv[5]);
    const int active_lanes = std::atoi(argv[6]);
    const int trials = std::atoi(argv[7]);
    if (!parse_named(argv[1], kControl, kReduction, tail_name, &tail) ||
        !parse_named(argv[2], kNoFence, kSystemFence, fence_name, &fence) ||
        !parse_named(argv[3], kLineStream, kSameStream, layout_name, &layout) ||
        blocks < 1 || blocks > 168 || depth < 1 || depth > kMaxDepth ||
        active_lanes < 1 || active_lanes > kWarpSize || trials < 1 ||
        trials > 32 || (tail == kStore && layout == kSameStream)) {
        return 2;
    }

    const std::size_t word_count = allocation_words(blocks, depth);
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
        launch_tail(tail, fence, blocks, words, depth, active_lanes, layout);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());

        std::vector<std::uint32_t> observed(word_count);
        CUDA_CHECK(cudaMemcpy(observed.data(), words,
                              word_count * sizeof(*words),
                              cudaMemcpyDeviceToHost));
        for (int operation = 0; operation < depth; ++operation) {
            for (int block = 0; block < blocks; ++block) {
                for (int lane = 0; lane < active_lanes; ++lane) {
                    const std::size_t word =
                        tail_word(layout, blocks, operation, block, lane);
                    std::uint32_t expected = initial_value(word);
                    if (tail == kReduction) {
                        if (layout == kSameStream) {
                            expected += static_cast<std::uint32_t>(active_lanes);
                        } else if (layout == kLineReuse) {
                            expected += static_cast<std::uint32_t>(depth);
                        } else {
                            expected += 1;
                        }
                    } else if (tail == kStore) {
                        // Store values are deliberately not used as a timing
                        // oracle; verify that every targeted location changed.
                        correct = correct && observed[word] != initial_value(word);
                    }
                    if (tail != kStore) {
                        correct = correct && observed[word] == expected;
                    }
                    checksum = checksum * 131u + observed[word];
                }
            }
        }
    }

    std::printf(
        "{\"tail\":\"%s\",\"fence\":\"%s\",\"layout\":\"%s\","
        "\"blocks\":%d,\"depth\":%d,\"active_lanes\":%d,"
        "\"work_iterations\":%d,\"trials\":%d,\"allocation_bytes\":%zu,"
        "\"checksum\":%llu,\"correct\":%s}\n",
        tail_name(tail), fence_name(fence), layout_name(layout), blocks,
        depth, active_lanes, kWorkIterations, trials,
        word_count * sizeof(std::uint32_t),
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(words));
    return correct ? 0 : 1;
}
