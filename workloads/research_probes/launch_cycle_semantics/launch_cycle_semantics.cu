#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <limits>

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
constexpr std::size_t kWordsPerLane = 32;

enum TailKind : int {
    kControl = 0,
    kLoadDiscard = 1,
    kLoadUse = 2,
    kStore = 3,
    kReduction = 4,
    kAtomicUse = 5,
};

__device__ __forceinline__ std::uint32_t dependent_step(std::uint32_t value) {
    std::uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;\n"
                 : "=r"(next)
                 : "r"(value));
    return next;
}

template <int Tail>
__global__ __launch_bounds__(kWarpSize)
void launch_cycle_tail(std::uint32_t* tail_words,
                       std::uint32_t* return_words,
                       int iterations) {
    __shared__ std::uint32_t work_sink[kWarpSize];
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t logical =
        static_cast<std::size_t>(blockIdx.x) * kWarpSize + lane;
    const std::size_t word = logical * kWordsPerLane;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(logical * 0x85ebca6bu);

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = dependent_step(value);
    }

    const unsigned work_address = static_cast<unsigned>(
        __cvta_generic_to_shared(work_sink + lane));
    asm volatile("st.volatile.shared.u32 [%0], %1;\n"
                 :
                 : "r"(work_address), "r"(value)
                 : "memory");

    // Make the final operation the only difference after every warp has
    // completed the same dependent work.  The inline assembly is volatile so
    // the discard case remains a real outstanding memory request.
    __syncthreads();
    if constexpr (Tail == kControl) {
        asm volatile("" : : "r"(value) : "memory");
    } else if constexpr (Tail == kLoadDiscard) {
        std::uint32_t loaded;
        asm volatile("ld.volatile.global.u32 %0, [%1];\n"
                     : "=r"(loaded)
                     : "l"(tail_words + word)
                     : "memory");
        asm volatile("" : : "r"(loaded));
    } else if constexpr (Tail == kLoadUse) {
        std::uint32_t loaded;
        asm volatile("ld.global.cg.u32 %0, [%1];\n"
                     : "=r"(loaded)
                     : "l"(tail_words + word)
                     : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;\n"
                     :
                     : "l"(return_words + word), "r"(loaded)
                     : "memory");
    } else if constexpr (Tail == kStore) {
        asm volatile("st.global.wb.u32 [%0], %1;\n"
                     :
                     : "l"(tail_words + word), "r"(value)
                     : "memory");
    } else if constexpr (Tail == kReduction) {
        asm volatile("red.global.add.u32 [%0], 1;\n"
                     :
                     : "l"(tail_words + word)
                     : "memory");
    } else if constexpr (Tail == kAtomicUse) {
        std::uint32_t old;
        asm volatile("atom.global.add.u32 %0, [%1], 1;\n"
                     : "=r"(old)
                     : "l"(tail_words + word)
                     : "memory");
        asm volatile("st.global.wb.u32 [%0], %1;\n"
                     :
                     : "l"(return_words + word), "r"(old)
                     : "memory");
    }
}

const char* tail_name(int tail) {
    switch (tail) {
        case kControl: return "control";
        case kLoadDiscard: return "load_discard";
        case kLoadUse: return "load_use";
        case kStore: return "store";
        case kReduction: return "red";
        case kAtomicUse: return "atom_use";
        default: return nullptr;
    }
}

bool parse_tail(const char* value, int* tail) {
    for (int candidate = kControl; candidate <= kAtomicUse; ++candidate) {
        if (std::strcmp(value, tail_name(candidate)) == 0) {
            *tail = candidate;
            return true;
        }
    }
    return false;
}

void launch_tail(int tail, int blocks, std::uint32_t* tail_words,
                 std::uint32_t* return_words, int iterations) {
    switch (tail) {
        case kControl:
            launch_cycle_tail<kControl><<<blocks, kWarpSize>>>(
                tail_words, return_words, iterations);
            break;
        case kLoadDiscard:
            launch_cycle_tail<kLoadDiscard><<<blocks, kWarpSize>>>(
                tail_words, return_words, iterations);
            break;
        case kLoadUse:
            launch_cycle_tail<kLoadUse><<<blocks, kWarpSize>>>(
                tail_words, return_words, iterations);
            break;
        case kStore:
            launch_cycle_tail<kStore><<<blocks, kWarpSize>>>(
                tail_words, return_words, iterations);
            break;
        case kReduction:
            launch_cycle_tail<kReduction><<<blocks, kWarpSize>>>(
                tail_words, return_words, iterations);
            break;
        case kAtomicUse:
            launch_cycle_tail<kAtomicUse><<<blocks, kWarpSize>>>(
                tail_words, return_words, iterations);
            break;
        default: std::abort();
    }
}

std::uint32_t initial_value(std::size_t logical) {
    return 0x6a09e667u ^ static_cast<std::uint32_t>(logical * 0x27d4eb2du);
}

std::uint32_t work_value(std::size_t logical, int iterations) {
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(logical * 0x85ebca6bu);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = value * 1664525u + 1013904223u;
    }
    return value;
}

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
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr,
                     "usage: %s TAIL BLOCKS ITERATIONS CHAIN_LENGTH TRIALS "
                     "[SYNC_INSIDE_RANGE|SYNC_OUTSIDE_RANGE]\n",
                     argv[0]);
        return 2;
    }

    int tail = -1;
    int blocks = 0;
    int iterations = 0;
    int chain_length = 0;
    int trials = 0;
    if (!parse_int(argv[2], &blocks) || !parse_int(argv[3], &iterations) ||
        !parse_int(argv[4], &chain_length) || !parse_int(argv[5], &trials)) {
        return 2;
    }
    const bool sync_inside_range =
        argc == 6 || std::strcmp(argv[6], "SYNC_INSIDE_RANGE") == 0;
    if (!parse_tail(argv[1], &tail) || blocks < 1 || blocks > 168 ||
        iterations < 0 || iterations > (1 << 20) || chain_length < 1 ||
        chain_length > 16 || trials < 1 || trials > 32 ||
        (argc == 7 && !sync_inside_range &&
         std::strcmp(argv[6], "SYNC_OUTSIDE_RANGE") != 0)) {
        return 2;
    }

    const std::size_t logical_words =
        static_cast<std::size_t>(blocks) * kWarpSize;
    const std::size_t allocated_words = logical_words * kWordsPerLane;
    std::vector<std::uint32_t> initial(allocated_words, 0);
    for (std::size_t logical = 0; logical < logical_words; ++logical) {
        initial[logical * kWordsPerLane] = initial_value(logical);
    }

    std::uint32_t* tail_words = nullptr;
    std::uint32_t* return_words = nullptr;
    CUDA_CHECK(cudaMalloc(&tail_words, allocated_words * sizeof(*tail_words)));
    CUDA_CHECK(cudaMalloc(&return_words,
                          allocated_words * sizeof(*return_words)));

    bool correct = true;
    std::uint64_t checksum = 0;
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaMemcpy(tail_words, initial.data(),
                              initial.size() * sizeof(initial[0]),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(return_words, 0,
                              allocated_words * sizeof(*return_words)));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaProfilerStart());
        for (int launch = 0; launch < chain_length; ++launch) {
            launch_tail(tail, blocks, tail_words, return_words, iterations);
            CUDA_CHECK(cudaGetLastError());
        }
        if (sync_inside_range) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaProfilerStop());
        if (!sync_inside_range) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<std::uint32_t> observed_tail(allocated_words);
        std::vector<std::uint32_t> observed_return(allocated_words);
        CUDA_CHECK(cudaMemcpy(observed_tail.data(), tail_words,
                              observed_tail.size() * sizeof(observed_tail[0]),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(observed_return.data(), return_words,
                              observed_return.size() * sizeof(observed_return[0]),
                              cudaMemcpyDeviceToHost));

        for (std::size_t logical = 0; logical < logical_words; ++logical) {
            const std::size_t word = logical * kWordsPerLane;
            std::uint32_t expected_tail = initial_value(logical);
            std::uint32_t expected_return = 0;
            if (tail == kStore) {
                expected_tail = work_value(logical, iterations);
            } else if (tail == kReduction || tail == kAtomicUse) {
                expected_tail += static_cast<std::uint32_t>(chain_length);
                if (tail == kAtomicUse) {
                    expected_return = initial_value(logical) +
                                      static_cast<std::uint32_t>(chain_length - 1);
                }
            } else if (tail == kLoadUse) {
                expected_return = initial_value(logical);
            }
            correct = correct && observed_tail[word] == expected_tail &&
                      observed_return[word] == expected_return;
            checksum = checksum * 131u + observed_tail[word];
            checksum = checksum * 131u + observed_return[word];
        }
    }

    std::printf(
        "{\"tail\":\"%s\",\"blocks\":%d,\"threads_per_block\":%d,"
        "\"iterations\":%d,\"chain_length\":%d,\"trials\":%d,"
        "\"sync_placement\":\"%s\",\"tail_stride_bytes\":%zu,"
        "\"checksum\":%llu,\"correct\":%s}\n",
        tail_name(tail), blocks, kWarpSize, iterations, chain_length, trials,
        sync_inside_range ? "inside_range" : "outside_range",
        kWordsPerLane * sizeof(std::uint32_t),
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(return_words));
    CUDA_CHECK(cudaFree(tail_words));
    return correct ? 0 : 1;
}
