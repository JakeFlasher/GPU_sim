#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

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
constexpr int kThreads = kWarpSize * kMaxWarps;
constexpr int kCopiesPerGroup = 8;
constexpr int kBackgroundGroupsPerEpoch = 4;
constexpr int kBackgroundCopiesPerGroup =
    kCopiesPerGroup / kBackgroundGroupsPerEpoch;
constexpr int kPayloadBytes = 512;
constexpr int kSourcePositionsPerWarp = kCopiesPerGroup;
constexpr int kSourceBytesPerWarp = kSourcePositionsPerWarp * kPayloadBytes;
constexpr int kSharedBytesPerWarp = kCopiesPerGroup * kPayloadBytes;
constexpr int kDestinationBytes = kMaxWarps * kSharedBytesPerWarp;
constexpr int kSourceBytes = kMaxWarps * kSourceBytesPerWarp;
constexpr int kDefaultGroups = 8192;
constexpr int kMaxGroups = 65536;

struct ArmResult {
    unsigned long long issue_cycles;
    unsigned long long wait_cycles;
    unsigned long long total_cycles;
    unsigned long long commit_before_target_sum;
    unsigned long long overlap_background_sum;
    unsigned long long overlap_threshold_epochs;
    unsigned long long zero_overlap_epochs;
    std::uint32_t target_token;
    std::uint32_t warp_tokens[kMaxWarps];
};

struct PairedResult {
    ArmResult queued;
    ArmResult drained;
};

struct SharedObservation {
    unsigned long long commit_time[kMaxWarps];
    unsigned long long wait_end[kMaxWarps];
    unsigned long long target_issue_begin;
    unsigned long long target_wait_end;
    unsigned ready_backgrounds;
    unsigned target_done;
    ArmResult arms[2];
};

__device__ __forceinline__ unsigned long long read_clock64() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) : : "memory");
    return value;
}

__device__ __forceinline__ void copy_ca(unsigned destination,
                                        const char* source) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(destination), "l"(source)
                 : "memory");
}

__device__ __forceinline__ void copy_cg(unsigned destination,
                                        const char* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(destination), "l"(source)
                 : "memory");
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void wait_group_zero() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

template <bool Bypass>
__device__ __forceinline__ void issue_group(unsigned destination,
                                            const char* source) {
#pragma unroll
    for (int copy = 0; copy < kCopiesPerGroup; ++copy) {
        const char* copy_source = source + copy * kPayloadBytes;
        if constexpr (Bypass) {
            copy_cg(destination + copy * kPayloadBytes, copy_source);
        } else {
            copy_ca(destination + copy * kPayloadBytes, copy_source);
        }
    }
    commit_group();
}

template <bool Bypass>
__device__ __forceinline__ void issue_background_group(
    unsigned destination, const char* source, int group_index) {
#pragma unroll
    for (int copy = 0; copy < kBackgroundCopiesPerGroup; ++copy) {
        const int position =
            group_index * kBackgroundCopiesPerGroup + copy;
        const char* copy_source = source + position * kPayloadBytes;
        if constexpr (Bypass) {
            copy_cg(destination + position * kPayloadBytes, copy_source);
        } else {
            copy_ca(destination + position * kPayloadBytes, copy_source);
        }
    }
    commit_group();
}

__device__ __forceinline__ std::uint32_t consume_last_copy(
    const char* destination, int warp, int lane) {
    const auto* value = reinterpret_cast<const volatile uint4*>(
        destination + warp * kSharedBytesPerWarp +
        (kCopiesPerGroup - 1) * kPayloadBytes + lane * 16);
    return value->x ^ (value->y << 3) ^ (value->z << 7) ^ (value->w << 11);
}

template <bool Bypass>
__device__ void run_arm(const char* source, char* destination,
                        SharedObservation* observation, int groups,
                        int active_warps, bool queued, int result_index) {
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int target = active_warps - 1;
    const bool active = warp < active_warps;
    const bool background = active && warp != target;
    const bool measured = warp == target;
    const char* warp_source =
        source + warp * kSourceBytesPerWarp + lane * 16;
    const unsigned warp_destination = static_cast<unsigned>(
        __cvta_generic_to_shared(destination + warp * kSharedBytesPerWarp +
                                 lane * 16));

    unsigned long long issue_cycles = 0;
    unsigned long long wait_cycles = 0;
    unsigned long long total_cycles = 0;
    unsigned long long commit_before_target_sum = 0;
    unsigned long long overlap_background_sum = 0;
    unsigned long long overlap_threshold_epochs = 0;
    unsigned long long zero_overlap_epochs = 0;
    std::uint32_t token = 0x73d9a42bu ^ static_cast<std::uint32_t>(warp);

    __syncthreads();
#pragma unroll 1
    for (int group = 0; group < groups; ++group) {
        if (threadIdx.x == 0) {
            observation->ready_backgrounds = 0;
            observation->target_done = 0;
        }
        __syncthreads();

        if (background) {
            #pragma unroll 1
            for (int background_group = 0;
                 background_group < kBackgroundGroupsPerEpoch;
                 ++background_group) {
                issue_background_group<Bypass>(warp_destination, warp_source,
                                               background_group);
            }
            if (lane == 0) observation->commit_time[warp] = read_clock64();
            if (!queued) {
                wait_group_zero();
                if (lane == 0) observation->wait_end[warp] = read_clock64();
            }
            if (lane == 0) atomicAdd(&observation->ready_backgrounds, 1u);
            __syncwarp();
        }

        if (measured) {
            while (*reinterpret_cast<volatile unsigned*>(
                       &observation->ready_backgrounds) !=
                   static_cast<unsigned>(target)) {
            }
            const unsigned long long before_issue = read_clock64();
            if (lane == 0) observation->target_issue_begin = before_issue;
            issue_group<Bypass>(warp_destination, warp_source);
            const unsigned long long after_issue = read_clock64();
            wait_group_zero();
            const unsigned long long after_wait = read_clock64();
            if (lane == 0) observation->target_wait_end = after_wait;
            issue_cycles += after_issue - before_issue;
            wait_cycles += after_wait - after_issue;
            total_cycles += after_wait - before_issue;
            if (lane == 0) atomicExch(&observation->target_done, 1u);
            __syncwarp();
        }
        if (background && queued) {
            while (*reinterpret_cast<volatile unsigned*>(
                       &observation->target_done) == 0u) {
            }
            wait_group_zero();
            if (lane == 0) observation->wait_end[warp] = read_clock64();
        }

        // All destination writes are complete before correctness loads and
        // overlap classification. These operations are outside target clocks.
        __syncthreads();
        if (active) {
            token ^= consume_last_copy(destination, warp, lane) +
                     static_cast<std::uint32_t>(group);
        }
        if (measured && lane == 0) {
            unsigned commit_before = 0;
            unsigned overlap = 0;
            for (int background_warp = 0; background_warp < target;
                 ++background_warp) {
                const unsigned long long commit =
                    observation->commit_time[background_warp];
                const unsigned long long wait_end =
                    observation->wait_end[background_warp];
                commit_before += commit < observation->target_issue_begin;
                overlap += commit < observation->target_wait_end &&
                           wait_end > observation->target_issue_begin;
            }
            commit_before_target_sum += commit_before;
            overlap_background_sum += overlap;
            overlap_threshold_epochs +=
                overlap >= static_cast<unsigned>(queued ? target - 1 : 1);
            zero_overlap_epochs += overlap == 0;
        }
        __syncthreads();
    }

    if (lane == 0 && active) {
        observation->arms[result_index].warp_tokens[warp] = token;
    }
    if (measured && lane == 0) {
        ArmResult& result = observation->arms[result_index];
        result.issue_cycles = issue_cycles;
        result.wait_cycles = wait_cycles;
        result.total_cycles = total_cycles;
        result.commit_before_target_sum = commit_before_target_sum;
        result.overlap_background_sum = overlap_background_sum;
        result.overlap_threshold_epochs = overlap_threshold_epochs;
        result.zero_overlap_epochs = zero_overlap_epochs;
        result.target_token = token;
    }
    __syncthreads();
}

template <bool Bypass>
__global__ void queue_overlap_paired_kernel(const char* source, int groups,
                                            int active_warps, bool queued_first,
                                            PairedResult* result) {
    __shared__ __align__(16) char destination[kDestinationBytes];
    __shared__ SharedObservation observation;
    if (threadIdx.x == 0) {
        observation = SharedObservation{};
    }
    __syncthreads();
    if (queued_first) {
        run_arm<Bypass>(source, destination, &observation, groups, active_warps,
                        true, 0);
        run_arm<Bypass>(source, destination, &observation, groups, active_warps,
                        false, 1);
    } else {
        run_arm<Bypass>(source, destination, &observation, groups, active_warps,
                        false, 1);
        run_arm<Bypass>(source, destination, &observation, groups, active_warps,
                        true, 0);
    }
    if (threadIdx.x == 0) {
        result->queued = observation.arms[0];
        result->drained = observation.arms[1];
    }
}

template <bool Bypass>
__global__ void queue_overlap_endpoint_kernel(const char* source, int groups,
                                              int active_warps, bool queued,
                                              ArmResult* result) {
    __shared__ __align__(16) char destination[kDestinationBytes];
    __shared__ SharedObservation observation;
    if (threadIdx.x == 0) observation = SharedObservation{};
    __syncthreads();
    run_arm<Bypass>(source, destination, &observation, groups, active_warps,
                    queued, 0);
    if (threadIdx.x == 0) *result = observation.arms[0];
}

__global__ void prewarm_kernel(const char* source, std::uint32_t* sink) {
    std::uint32_t folded = 0;
    for (std::size_t offset = static_cast<std::size_t>(threadIdx.x) * 16;
         offset < kSourceBytes;
         offset += static_cast<std::size_t>(blockDim.x) * 16) {
        const auto* value = reinterpret_cast<const uint4*>(source + offset);
        const uint4 loaded = *value;
        folded ^= loaded.x ^ loaded.y ^ loaded.z ^ loaded.w;
    }
    atomicXor(sink, folded);
}

void launch_paired(bool bypass, const char* source, int groups,
                   int active_warps, bool queued_first, PairedResult* result) {
    if (bypass) {
        queue_overlap_paired_kernel<true><<<1, kThreads>>>(
            source, groups, active_warps, queued_first, result);
    } else {
        queue_overlap_paired_kernel<false><<<1, kThreads>>>(
            source, groups, active_warps, queued_first, result);
    }
}

void launch_endpoint(bool bypass, const char* source, int groups,
                     int active_warps, bool queued, ArmResult* result) {
    if (bypass) {
        queue_overlap_endpoint_kernel<true><<<1, kThreads>>>(
            source, groups, active_warps, queued, result);
    } else {
        queue_overlap_endpoint_kernel<false><<<1, kThreads>>>(
            source, groups, active_warps, queued, result);
    }
}

int parse_count(const char* text, int maximum) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 1 || value > maximum) return -1;
    return static_cast<int>(value);
}

std::uint32_t fold_source(const std::vector<std::uint32_t>& source, int warp,
                          int position) {
    const std::size_t first =
        (static_cast<std::size_t>(warp) * kSourceBytesPerWarp +
         static_cast<std::size_t>(position) * kPayloadBytes) /
        sizeof(std::uint32_t);
    return source[first] ^ (source[first + 1] << 3) ^
           (source[first + 2] << 7) ^ (source[first + 3] << 11);
}

std::uint32_t expected_token(const std::vector<std::uint32_t>& source,
                             int warp, int groups) {
    std::uint32_t token = 0x73d9a42bu ^ static_cast<std::uint32_t>(warp);
    const std::uint32_t folded =
        fold_source(source, warp, kCopiesPerGroup - 1);
    for (int group = 0; group < groups; ++group) {
        token ^= folded + static_cast<std::uint32_t>(group);
    }
    return token;
}

bool validate_arm(const ArmResult& result,
                  const std::vector<std::uint32_t>& source, int groups,
                  int active_warps, bool queued) {
    if (result.issue_cycles + result.wait_cycles != result.total_cycles) {
        return false;
    }
    for (int warp = 0; warp < kMaxWarps; ++warp) {
        const std::uint32_t expected =
            warp < active_warps
            ? expected_token(source, warp, groups)
            : 0;
        if (result.warp_tokens[warp] != expected) return false;
    }
    const unsigned long long backgrounds =
        static_cast<unsigned long long>(active_warps - 1);
    if (result.commit_before_target_sum != backgrounds * groups) return false;
    if (queued) {
        if (result.overlap_threshold_epochs * 100 <
            static_cast<unsigned long long>(groups) * 95) {
            return false;
        }
    } else if (result.overlap_background_sum != 0 ||
               result.zero_overlap_epochs != static_cast<unsigned long long>(groups)) {
        return false;
    }
    return result.target_token ==
           expected_token(source, active_warps - 1, groups);
}

void print_arm(const char* name, const ArmResult& result, int groups,
               int active_warps) {
    std::printf(
        "\"%s\":{\"issue_cycles\":%llu,\"wait_cycles\":%llu,"
        "\"total_cycles\":%llu,\"commit_before_target_sum\":%llu,"
        "\"overlap_background_sum\":%llu,\"overlap_threshold_epochs\":%llu,"
        "\"zero_overlap_epochs\":%llu,\"target_token\":%u,"
        "\"backgrounds\":%d,\"groups\":%d}",
        name, result.issue_cycles, result.wait_cycles, result.total_cycles,
        result.commit_before_target_sum, result.overlap_background_sum,
        result.overlap_threshold_epochs, result.zero_overlap_epochs,
        result.target_token, active_warps - 1, groups);
}

int main(int argc, char** argv) {
    int groups = kDefaultGroups;
    int active_warps = 4;
    bool bypass = false;
    bool paired = true;
    bool queued = true;
    bool queued_first = true;
    bool profiler_marked = false;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (std::strcmp(argv[i], "--groups") == 0) {
            groups = parse_count(argv[++i], kMaxGroups);
        } else if (std::strcmp(argv[i], "--active-warps") == 0) {
            active_warps = parse_count(argv[++i], kMaxWarps);
        } else if (std::strcmp(argv[i], "--route") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "l1") == 0) bypass = false;
            else if (std::strcmp(value, "l2") == 0) bypass = true;
            else return 2;
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "paired") == 0) paired = true;
            else if (std::strcmp(value, "endpoint") == 0) paired = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--arm") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "queued") == 0) queued = true;
            else if (std::strcmp(value, "drained") == 0) queued = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--order") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "qd") == 0) queued_first = true;
            else if (std::strcmp(value, "dq") == 0) queued_first = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--markers") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "on") == 0) profiler_marked = true;
            else if (std::strcmp(value, "off") == 0) profiler_marked = false;
            else return 2;
        } else {
            return 2;
        }
    }
    if (groups < 1 || (active_warps != 4 && active_warps != 8)) return 2;

    std::vector<std::uint32_t> host_source(kSourceBytes / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < host_source.size(); ++i) {
        host_source[i] = (static_cast<std::uint32_t>(i) * 1664525u + 1013904223u) ^
                         0x9e3779b9u;
    }
    char* source = nullptr;
    PairedResult* paired_result = nullptr;
    ArmResult* endpoint_result = nullptr;
    std::uint32_t* prewarm_sink = nullptr;
    PairedResult host_pair{};
    ArmResult host_endpoint{};
    CUDA_CHECK(cudaMalloc(&source, kSourceBytes));
    CUDA_CHECK(cudaMalloc(&paired_result, sizeof(PairedResult)));
    CUDA_CHECK(cudaMalloc(&endpoint_result, sizeof(ArmResult)));
    CUDA_CHECK(cudaMalloc(&prewarm_sink, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(source, host_source.data(), kSourceBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(paired_result, 0, sizeof(PairedResult)));
    CUDA_CHECK(cudaMemset(endpoint_result, 0, sizeof(ArmResult)));
    CUDA_CHECK(cudaMemset(prewarm_sink, 0, sizeof(std::uint32_t)));
    prewarm_kernel<<<1, kThreads>>>(source, prewarm_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
    if (paired) {
        launch_paired(bypass, source, groups, active_warps, queued_first,
                      paired_result);
    } else {
        launch_endpoint(bypass, source, groups, active_warps, queued,
                        endpoint_result);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStop());

    bool correct = false;
    if (paired) {
        CUDA_CHECK(cudaMemcpy(&host_pair, paired_result, sizeof(PairedResult),
                              cudaMemcpyDeviceToHost));
        correct = validate_arm(host_pair.queued, host_source, groups,
                               active_warps, true) &&
                  validate_arm(host_pair.drained, host_source, groups,
                               active_warps, false);
        std::printf(
            "{\"workload\":\"async_cross_warp_queue_overlap\","
            "\"mode\":\"paired\",\"route\":\"%s\",\"active_warps\":%d,"
            "\"groups\":%d,\"copies_per_group\":8,"
            "\"background_groups_per_epoch\":4,\"order\":\"%s\",",
            bypass ? "l2" : "l1", active_warps, groups,
            queued_first ? "qd" : "dq");
        print_arm("queued", host_pair.queued, groups, active_warps);
        std::printf(",");
        print_arm("drained", host_pair.drained, groups, active_warps);
        std::printf(",\"result_store_after_intervals\":true,\"correct\":%s}\n",
                    correct ? "true" : "false");
    } else {
        CUDA_CHECK(cudaMemcpy(&host_endpoint, endpoint_result, sizeof(ArmResult),
                              cudaMemcpyDeviceToHost));
        correct = validate_arm(host_endpoint, host_source, groups, active_warps,
                               queued);
        std::printf(
            "{\"workload\":\"async_cross_warp_queue_overlap\","
            "\"mode\":\"endpoint\",\"route\":\"%s\",\"arm\":\"%s\","
            "\"active_warps\":%d,\"groups\":%d,\"copies_per_group\":8,"
            "\"background_groups_per_epoch\":4,",
            bypass ? "l2" : "l1", queued ? "queued" : "drained",
            active_warps, groups);
        print_arm("result", host_endpoint, groups, active_warps);
        std::printf(",\"result_store_after_intervals\":true,\"correct\":%s}\n",
                    correct ? "true" : "false");
    }

    CUDA_CHECK(cudaFree(prewarm_sink));
    CUDA_CHECK(cudaFree(endpoint_result));
    CUDA_CHECK(cudaFree(paired_result));
    CUDA_CHECK(cudaFree(source));
    return correct ? 0 : 1;
}
