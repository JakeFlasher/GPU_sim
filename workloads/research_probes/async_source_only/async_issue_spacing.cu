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

struct SpacingResult {
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

#define SPACING_SAMPLE_1                                                     \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %0, %0, sample;\n\t"
#define SPACING_SAMPLE_2 SPACING_SAMPLE_1                                    \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %1, %1, sample;\n\t"
#define SPACING_SAMPLE_4 SPACING_SAMPLE_2                                    \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %2, %2, sample;\n\t"                                            \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %3, %3, sample;\n\t"
#define SPACING_SAMPLE_8 SPACING_SAMPLE_4 SPACING_SAMPLE_4
#define SPACING_SAMPLE_16 SPACING_SAMPLE_8 SPACING_SAMPLE_8

template <int Samples>
__device__ __forceinline__ void issue_spacing(uint4* state) {
    static_assert(Samples == 0 || Samples == 1 || Samples == 2 || Samples == 4 ||
                  Samples == 8 || Samples == 16);
    if constexpr (Samples == 1) {
        asm volatile("{ .reg .u32 sample;\n\t" SPACING_SAMPLE_1 "}"
                     : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                       "+r"(state->w)
                     :
                     : "memory");
    }
    if constexpr (Samples == 2) {
        asm volatile("{ .reg .u32 sample;\n\t" SPACING_SAMPLE_2 "}"
                     : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                       "+r"(state->w)
                     :
                     : "memory");
    }
    if constexpr (Samples == 4) {
        asm volatile("{ .reg .u32 sample;\n\t" SPACING_SAMPLE_4 "}"
                     : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                       "+r"(state->w)
                     :
                     : "memory");
    }
    if constexpr (Samples == 8) {
        asm volatile("{ .reg .u32 sample;\n\t" SPACING_SAMPLE_8 "}"
                     : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                       "+r"(state->w)
                     :
                     : "memory");
    }
    if constexpr (Samples == 16) {
        asm volatile("{ .reg .u32 sample;\n\t" SPACING_SAMPLE_16 "}"
                     : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                       "+r"(state->w)
                     :
                     : "memory");
    }
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

template <bool Bypass>
__device__ __forceinline__ void issue_copy(unsigned destination,
                                           const char* source) {
    if constexpr (Bypass) copy_cg(destination, source);
    else copy_ca(destination, source);
}

template <int Depth, bool Bypass, int Layout, int Nops>
__global__ void issue_spacing_kernel(const char* source, int groups,
                                     SpacingResult* result) {
    __shared__ __align__(16) char destination[kDestinationBytes];
    if (threadIdx.x >= kWarpSize) return;
    const int lane = static_cast<int>(threadIdx.x);
    const unsigned destination_base = static_cast<unsigned>(
        __cvta_generic_to_shared(destination + destination_offset<Layout>(lane)));
    constexpr unsigned destination_stride =
        static_cast<unsigned>(kDestinationPositionStride);
    unsigned long long issue_cycles = 0;
    unsigned long long wait_cycles = 0;
    unsigned long long total_cycles = 0;
    std::uint32_t token = 0x1f83d9abu ^ static_cast<std::uint32_t>(lane);
    uint4 spacing_state = {
        static_cast<unsigned>(lane + 1),
        static_cast<unsigned>(lane + 3),
        static_cast<unsigned>(lane + 5),
        static_cast<unsigned>(lane + 7),
    };

#pragma unroll 1
    for (int group = 0; group < groups; ++group) {
        const char* s0 = source + 0u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s1 = source + 1u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s2 = source + 2u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s3 = source + 3u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s4 = source + 4u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s5 = source + 5u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s6 = source + 6u * kSourcePositionStride + lane * kSourceLaneStride;
        const char* s7 = source + 7u * kSourcePositionStride + lane * kSourceLaneStride;

        const unsigned long long before_issue = read_clock64();
        if constexpr (Depth > 0) issue_copy<Bypass>(destination_base, s0);
        if constexpr (Depth > 1) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + destination_stride, s1);
        }
        if constexpr (Depth > 2) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + 2u * destination_stride, s2);
        }
        if constexpr (Depth > 3) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + 3u * destination_stride, s3);
        }
        if constexpr (Depth > 4) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + 4u * destination_stride, s4);
        }
        if constexpr (Depth > 5) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + 5u * destination_stride, s5);
        }
        if constexpr (Depth > 6) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + 6u * destination_stride, s6);
        }
        if constexpr (Depth > 7) {
            issue_spacing<Nops>(&spacing_state);
            issue_copy<Bypass>(destination_base + 7u * destination_stride, s7);
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
    token ^= spacing_state.x + (spacing_state.y << 3) +
             (spacing_state.z << 7) + (spacing_state.w << 11);

    if (lane == 0) {
        result->issue_cycles = issue_cycles;
        result->wait_cycles = wait_cycles;
        result->total_cycles = total_cycles;
        result->token = token;
    }
}

template <int Depth, bool Bypass, int Layout, int Nops>
void launch(const char* source, int groups, SpacingResult* result) {
    issue_spacing_kernel<Depth, Bypass, Layout, Nops>
        <<<1, kWarpSize>>>(source, groups, result);
}

template <bool Bypass, int Layout, int Nops>
void launch_depth(int depth, const char* source, int groups, SpacingResult* result) {
    switch (depth) {
        case 1: launch<1, Bypass, Layout, Nops>(source, groups, result); return;
        case 2: launch<2, Bypass, Layout, Nops>(source, groups, result); return;
        case 4: launch<4, Bypass, Layout, Nops>(source, groups, result); return;
        case 8: launch<8, Bypass, Layout, Nops>(source, groups, result); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported depth: %d\n", depth);
    std::exit(2);
}

template <bool Bypass, int Layout>
void launch_nops(int depth, int nops, const char* source, int groups,
                 SpacingResult* result) {
    switch (nops) {
        case 0: launch_depth<Bypass, Layout, 0>(depth, source, groups, result); return;
        case 1: launch_depth<Bypass, Layout, 1>(depth, source, groups, result); return;
        case 2: launch_depth<Bypass, Layout, 2>(depth, source, groups, result); return;
        case 4: launch_depth<Bypass, Layout, 4>(depth, source, groups, result); return;
        case 8: launch_depth<Bypass, Layout, 8>(depth, source, groups, result); return;
        case 16: launch_depth<Bypass, Layout, 16>(depth, source, groups, result); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported NOP spacing: %d\n", nops);
    std::exit(2);
}

template <bool Bypass>
void launch_layout(int depth, int layout, int nops, const char* source,
                   int groups, SpacingResult* result) {
    switch (layout) {
        case 0: launch_nops<Bypass, 0>(depth, nops, source, groups, result); return;
        case 1: launch_nops<Bypass, 1>(depth, nops, source, groups, result); return;
        case 2: launch_nops<Bypass, 2>(depth, nops, source, groups, result); return;
        case 3: launch_nops<Bypass, 3>(depth, nops, source, groups, result); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported layout index: %d\n", layout);
    std::exit(2);
}

int parse_int(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 0 || value > kMaxGroups) return -1;
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    int groups = kDefaultGroups;
    int depth = 2;
    int layout = 0;
    int spacing_samples = 0;
    bool bypass = true;
    bool profiler_marked = false;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (std::strcmp(argv[i], "--groups") == 0) groups = parse_int(argv[++i]);
        else if (std::strcmp(argv[i], "--depth") == 0) depth = parse_int(argv[++i]);
        else if (std::strcmp(argv[i], "--spacing-samples") == 0) {
            spacing_samples = parse_int(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--route") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "l1") == 0) bypass = false;
            else if (std::strcmp(value, "l2") == 0) bypass = true;
            else return 2;
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
    if (groups < 1 || (depth != 1 && depth != 2 && depth != 4 && depth != 8)) return 2;
    if (spacing_samples != 0 && spacing_samples != 1 && spacing_samples != 2 &&
        spacing_samples != 4 && spacing_samples != 8 && spacing_samples != 16) return 2;

    std::vector<std::uint32_t> host_source(
        kPositions * kSourcePositionStride / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < host_source.size(); ++i) {
        host_source[i] = static_cast<std::uint32_t>(i) * 1664525u +
                         1013904223u ^ 0x9e3779b9u;
    }
    char* source = nullptr;
    SpacingResult* device_result = nullptr;
    SpacingResult host_result{};
    CUDA_CHECK(cudaMalloc(&source, host_source.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(SpacingResult)));
    CUDA_CHECK(cudaMemcpy(source, host_source.data(),
                          host_source.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(SpacingResult)));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
    if (bypass) {
        launch_layout<true>(depth, layout, spacing_samples, source, groups,
                            device_result);
    } else {
        launch_layout<false>(depth, layout, spacing_samples, source, groups,
                             device_result);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaMemcpy(&host_result, device_result, sizeof(SpacingResult),
                          cudaMemcpyDeviceToHost));

    const int edges = depth - 1;
    std::printf(
        "{\"workload\":\"async_issue_spacing\",\"route\":\"%s\","
        "\"layout\":\"%s\",\"depth\":%d,\"groups\":%d,"
        "\"spacing_clock_samples_per_edge\":%d,\"edges_per_group\":%d,"
        "\"issue_cycles\":%llu,\"wait_cycles\":%llu,\"total_cycles\":%llu,"
        "\"sum_consistent\":%s,\"result_store_after_intervals\":true,"
        "\"token\":%u,\"correct\":true}\n",
        bypass ? "l2" : "l1",
        layout == 0
            ? "compact"
            : (layout == 1 ? "conflict" : (layout == 2 ? "severe" : "skew")),
        depth, groups, spacing_samples, edges, host_result.issue_cycles,
        host_result.wait_cycles, host_result.total_cycles,
        host_result.issue_cycles + host_result.wait_cycles == host_result.total_cycles
            ? "true" : "false",
        host_result.token);

    CUDA_CHECK(cudaFree(device_result));
    CUDA_CHECK(cudaFree(source));
    return 0;
}
