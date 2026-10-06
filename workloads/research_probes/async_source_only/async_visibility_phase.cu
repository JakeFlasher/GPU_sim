#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kPositions = 8;
constexpr std::size_t kSourcePositionStride = 1024;
constexpr std::size_t kSourceLaneStride = 32;
constexpr std::size_t kDestinationPositionStride = 4096;
constexpr std::size_t kCompactLaneStride = 32;
constexpr std::size_t kConflictLaneStride = 64;
constexpr std::size_t kSevereLaneStride = 128;
constexpr std::size_t kSkewLaneStride = 96;
constexpr std::size_t kDestinationBytes =
    kPositions * kDestinationPositionStride;
constexpr int kDefaultGroups = 8192;
constexpr int kMaxGroups = 65536;

struct PhaseResult {
    unsigned long long issue_cycles;
    unsigned long long wait_cycles;
    unsigned long long total_cycles;
    std::uint32_t token;
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

template <int Layout>
__device__ __forceinline__ unsigned destination_offset(int lane) {
    static_assert(Layout >= 0 && Layout <= 3);
    constexpr std::size_t lane_stride = Layout == 0
        ? kCompactLaneStride
        : (Layout == 1
               ? kConflictLaneStride
               : (Layout == 2 ? kSevereLaneStride : kSkewLaneStride));
    return static_cast<unsigned>(lane * lane_stride);
}

template <int Depth, bool Bypass, bool Streaming, int Layout>
__global__ void visibility_phase_kernel(const char* source, int groups,
                                        PhaseResult* result) {
    __shared__ __align__(16) char destination[kDestinationBytes];
    if (threadIdx.x >= kWarpSize) return;
    const int lane = static_cast<int>(threadIdx.x);
    const char* base = source;
    const unsigned destination_base = static_cast<unsigned>(
        __cvta_generic_to_shared(destination + destination_offset<Layout>(lane)));
    constexpr unsigned destination_stride =
        static_cast<unsigned>(kDestinationPositionStride);
    unsigned long long issue_cycles = 0;
    unsigned long long wait_cycles = 0;
    unsigned long long total_cycles = 0;
    std::uint32_t token = 0x510e527fu ^ static_cast<std::uint32_t>(lane);

#pragma unroll 1
    for (int group = 0; group < groups; ++group) {
        const std::size_t source_group = Streaming
            ? static_cast<std::size_t>(group) * Depth * kSourcePositionStride
            : 0u;
        const char* s0 = base + source_group + 0u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s1 = base + source_group + 1u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s2 = base + source_group + 2u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s3 = base + source_group + 3u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s4 = base + source_group + 4u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s5 = base + source_group + 5u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s6 = base + source_group + 6u * kSourcePositionStride +
                         lane * kSourceLaneStride;
        const char* s7 = base + source_group + 7u * kSourcePositionStride +
                         lane * kSourceLaneStride;

        const unsigned long long before_issue = read_clock64();
        if constexpr (Bypass) {
            if constexpr (Depth > 0) copy_cg(destination_base + 0u * destination_stride, s0);
            if constexpr (Depth > 1) copy_cg(destination_base + 1u * destination_stride, s1);
            if constexpr (Depth > 2) copy_cg(destination_base + 2u * destination_stride, s2);
            if constexpr (Depth > 3) copy_cg(destination_base + 3u * destination_stride, s3);
            if constexpr (Depth > 4) copy_cg(destination_base + 4u * destination_stride, s4);
            if constexpr (Depth > 5) copy_cg(destination_base + 5u * destination_stride, s5);
            if constexpr (Depth > 6) copy_cg(destination_base + 6u * destination_stride, s6);
            if constexpr (Depth > 7) copy_cg(destination_base + 7u * destination_stride, s7);
        } else {
            if constexpr (Depth > 0) copy_ca(destination_base + 0u * destination_stride, s0);
            if constexpr (Depth > 1) copy_ca(destination_base + 1u * destination_stride, s1);
            if constexpr (Depth > 2) copy_ca(destination_base + 2u * destination_stride, s2);
            if constexpr (Depth > 3) copy_ca(destination_base + 3u * destination_stride, s3);
            if constexpr (Depth > 4) copy_ca(destination_base + 4u * destination_stride, s4);
            if constexpr (Depth > 5) copy_ca(destination_base + 5u * destination_stride, s5);
            if constexpr (Depth > 6) copy_ca(destination_base + 6u * destination_stride, s6);
            if constexpr (Depth > 7) copy_ca(destination_base + 7u * destination_stride, s7);
        }
        const unsigned long long after_issue = read_clock64();
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        const unsigned long long after_wait = read_clock64();

        issue_cycles += after_issue - before_issue;
        wait_cycles += after_wait - after_issue;
        total_cycles += after_wait - before_issue;

        const volatile uint4* value =
            reinterpret_cast<const volatile uint4*>(destination);
        token ^= value->x + (value->y << 3) + (value->z << 7) +
                 (value->w << 11) + static_cast<std::uint32_t>(group);
    }

    // This is the assay's only global store and occurs after every sampled
    // interval. It makes the three clock frontiers host-observable without
    // contaminating issue_cycles or wait_cycles.
    if (lane == 0) {
        result->issue_cycles = issue_cycles;
        result->wait_cycles = wait_cycles;
        result->total_cycles = total_cycles;
        result->token = token;
    }
}

template <int Depth, bool Bypass, bool Streaming, int Layout>
void launch(const char* source, int groups, PhaseResult* result) {
    visibility_phase_kernel<Depth, Bypass, Streaming, Layout>
        <<<1, kWarpSize>>>(source, groups, result);
}

template <bool Bypass, bool Streaming, int Layout>
void launch_depth(int depth, const char* source, int groups, PhaseResult* result) {
    switch (depth) {
        case 1: launch<1, Bypass, Streaming, Layout>(source, groups, result); return;
        case 2: launch<2, Bypass, Streaming, Layout>(source, groups, result); return;
        case 4: launch<4, Bypass, Streaming, Layout>(source, groups, result); return;
        case 8: launch<8, Bypass, Streaming, Layout>(source, groups, result); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported depth: %d\n", depth);
    std::exit(2);
}

template <bool Bypass, bool Streaming>
void launch_layout(int depth, int layout, const char* source, int groups,
                   PhaseResult* result) {
    switch (layout) {
        case 0: launch_depth<Bypass, Streaming, 0>(depth, source, groups, result); return;
        case 1: launch_depth<Bypass, Streaming, 1>(depth, source, groups, result); return;
        case 2: launch_depth<Bypass, Streaming, 2>(depth, source, groups, result); return;
        case 3: launch_depth<Bypass, Streaming, 3>(depth, source, groups, result); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported layout index: %d\n", layout);
    std::exit(2);
}

void launch_for(int depth, bool bypass, bool streaming, int layout,
                const char* source, int groups, PhaseResult* result) {
    if (bypass && streaming) {
        launch_layout<true, true>(depth, layout, source, groups, result);
    } else if (bypass) {
        launch_layout<true, false>(depth, layout, source, groups, result);
    } else {
        launch_layout<false, false>(depth, layout, source, groups, result);
    }
}

int parse_int(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 1 || value > kMaxGroups) return -1;
    return static_cast<int>(value);
}

std::uint32_t source_word(std::size_t index) {
    return static_cast<std::uint32_t>(index) * 1664525u +
           1013904223u ^ 0x9e3779b9u;
}

std::uint32_t expected_lane_zero_token(int groups, int depth, bool streaming) {
    std::uint32_t token = 0x510e527fu;
    for (int group = 0; group < groups; ++group) {
        const std::size_t source_group = streaming
            ? static_cast<std::size_t>(group) * depth * kSourcePositionStride
            : 0u;
        const std::size_t word = source_group / sizeof(std::uint32_t);
        const std::uint32_t x = source_word(word + 0u);
        const std::uint32_t y = source_word(word + 1u);
        const std::uint32_t z = source_word(word + 2u);
        const std::uint32_t w = source_word(word + 3u);
        token ^= x + (y << 3) + (z << 7) + (w << 11) +
                 static_cast<std::uint32_t>(group);
    }
    return token;
}

int main(int argc, char** argv) {
    int groups = kDefaultGroups;
    int depth = 4;
    bool bypass = false;
    bool streaming = false;
    int layout = 0;
    bool profiler_marked = false;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (std::strcmp(argv[i], "--groups") == 0) groups = parse_int(argv[++i]);
        else if (std::strcmp(argv[i], "--depth") == 0) depth = parse_int(argv[++i]);
        else if (std::strcmp(argv[i], "--route") == 0) {
            const char* route = argv[++i];
            if (std::strcmp(route, "l1") == 0) {
                bypass = false;
                streaming = false;
            } else if (std::strcmp(route, "l2") == 0) {
                bypass = true;
                streaming = false;
            } else if (std::strcmp(route, "dram") == 0) {
                bypass = true;
                streaming = true;
            } else return 2;
        } else if (std::strcmp(argv[i], "--layout") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "compact") == 0) layout = 0;
            else if (std::strcmp(value, "conflict") == 0) layout = 1;
            else if (std::strcmp(value, "severe") == 0) layout = 2;
            else if (std::strcmp(value, "skew") == 0) layout = 3;
            else return 2;
        } else if (std::strcmp(argv[i], "--markers") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "on") == 0) profiler_marked = true;
            else if (std::strcmp(value, "off") == 0) profiler_marked = false;
            else return 2;
        } else return 2;
    }
    if (groups < 1 ||
        (depth != 1 && depth != 2 && depth != 4 && depth != 8)) return 2;

    const std::size_t chunks = streaming
        ? static_cast<std::size_t>(groups) * depth
        : static_cast<std::size_t>(kPositions);
    std::vector<std::uint32_t> host_source(
        chunks * kSourcePositionStride / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < host_source.size(); ++i) {
        host_source[i] = source_word(i);
    }
    char* source = nullptr;
    PhaseResult* device_result = nullptr;
    PhaseResult host_result{};
    CUDA_CHECK(cudaMalloc(&source, host_source.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(PhaseResult)));
    CUDA_CHECK(cudaMemcpy(source, host_source.data(),
                          host_source.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_result, 0xff, sizeof(PhaseResult)));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
    launch_for(depth, bypass, streaming, layout, source, groups, device_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaMemcpy(&host_result, device_result, sizeof(PhaseResult),
                          cudaMemcpyDeviceToHost));

    const bool sum_consistent =
        host_result.total_cycles != UINT64_MAX && host_result.issue_cycles > 0 &&
        host_result.wait_cycles > 0 && host_result.issue_cycles <= host_result.total_cycles &&
        host_result.wait_cycles == host_result.total_cycles - host_result.issue_cycles;
    const std::uint32_t expected_token =
        expected_lane_zero_token(groups, depth, streaming);
    const bool token_correct = host_result.token == expected_token;
    const bool correct = sum_consistent && token_correct;

    std::printf(
        "{\"workload\":\"async_visibility_phase\",\"depth\":%d,"
        "\"route\":\"%s\",\"layout\":\"%s\",\"groups\":%d,"
        "\"issue_cycles\":%llu,\"wait_cycles\":%llu,\"total_cycles\":%llu,"
        "\"sum_consistent\":%s,\"result_store_after_intervals\":true,"
        "\"token\":%u,\"expected_token\":%u,\"token_correct\":%s,"
        "\"correct\":%s}\n",
        depth, streaming ? "dram" : (bypass ? "l2" : "l1"),
        layout == 0
            ? "compact"
            : (layout == 1 ? "conflict" : (layout == 2 ? "severe" : "skew")),
        groups, host_result.issue_cycles, host_result.wait_cycles,
        host_result.total_cycles,
        sum_consistent ? "true" : "false",
        host_result.token, expected_token, token_correct ? "true" : "false",
        correct ? "true" : "false");

    CUDA_CHECK(cudaFree(device_result));
    CUDA_CHECK(cudaFree(source));
    return correct ? 0 : 1;
}
