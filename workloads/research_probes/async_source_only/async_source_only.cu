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
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
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
constexpr int kDefaultGroups = 64;
constexpr int kMaxGroups = 16384;

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
__global__ void source_only_kernel(const char* source, int groups) {
    // Every layout reserves exactly 32 KiB.  Only the per-position and
    // per-lane destination mapping changes, so occupancy/resource allocation
    // cannot explain a layout timing difference.
    __shared__ __align__(16) char destination[kDestinationBytes];
    if (threadIdx.x >= kWarpSize) return;
    const int lane = static_cast<int>(threadIdx.x);
    const char* base = source;
    std::uint32_t token = 0x6a09e667u ^ static_cast<std::uint32_t>(lane);
    const unsigned destination_base = static_cast<unsigned>(
        __cvta_generic_to_shared(destination + destination_offset<Layout>(lane)));
    constexpr unsigned destination_stride =
        static_cast<unsigned>(kDestinationPositionStride);
    const unsigned long long start = clock64();
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
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        // Volatile shared reads keep destination visibility and source work
        // live without adding a global store or a CTA barrier to this kernel.
        // Read one address written by both variants.  Warp-uniform shared
        // reads are broadcasts, avoiding a layout-dependent LDS confound.
        const volatile uint4* value =
            reinterpret_cast<const volatile uint4*>(destination);
        token ^= value->x + (value->y << 3) + (value->z << 7) +
                 (value->w << 11) + static_cast<std::uint32_t>(group);
    }
    const unsigned long long stop = clock64();
    // Keep the clocks/token observable to the compiler while retaining a
    // store-free target kernel.  The empty asm has no memory side effect.
    asm volatile("" : : "l"(start), "l"(stop), "r"(token) : "memory");
}

template <int Depth, bool Bypass, bool Streaming, int Layout>
void launch(const char* source, int groups) {
    source_only_kernel<Depth, Bypass, Streaming, Layout>
        <<<1, kWarpSize>>>(source, groups);
}

template <bool Bypass, bool Streaming, int Layout>
void launch_depth(int depth, const char* source, int groups) {
    switch (depth) {
        case 1: launch<1, Bypass, Streaming, Layout>(source, groups); return;
        case 4: launch<4, Bypass, Streaming, Layout>(source, groups); return;
        case 5: launch<5, Bypass, Streaming, Layout>(source, groups); return;
        case 6: launch<6, Bypass, Streaming, Layout>(source, groups); return;
        case 8: launch<8, Bypass, Streaming, Layout>(source, groups); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported depth: %d\n", depth);
    std::exit(2);
}

template <bool Bypass, bool Streaming>
void launch_layout(int depth, int layout, const char* source, int groups) {
    switch (layout) {
        case 0: launch_depth<Bypass, Streaming, 0>(depth, source, groups); return;
        case 1: launch_depth<Bypass, Streaming, 1>(depth, source, groups); return;
        case 2: launch_depth<Bypass, Streaming, 2>(depth, source, groups); return;
        case 3: launch_depth<Bypass, Streaming, 3>(depth, source, groups); return;
        default: break;
    }
    std::fprintf(stderr, "unsupported layout index: %d\n", layout);
    std::exit(2);
}

void launch_for(int depth, bool bypass, bool streaming, int layout,
                const char* source, int groups) {
    if (bypass && streaming) {
        launch_layout<true, true>(depth, layout, source, groups);
    } else if (bypass) {
        launch_layout<true, false>(depth, layout, source, groups);
    } else {
        launch_layout<false, false>(depth, layout, source, groups);
    }
}

int parse_int(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 1 || value > kMaxGroups) return -1;
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    int groups = kDefaultGroups;
    int depth = 4;
    bool bypass = false;
    bool streaming = false;
    int layout = 0;
    bool profiler_marked = true;
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
            }
            else return 2;
        } else if (std::strcmp(argv[i], "--markers") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "on") == 0) profiler_marked = true;
            else if (std::strcmp(value, "off") == 0) profiler_marked = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--layout") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "compact") == 0) layout = 0;
            else if (std::strcmp(value, "conflict") == 0) layout = 1;
            else if (std::strcmp(value, "severe") == 0) layout = 2;
            else if (std::strcmp(value, "skew") == 0) layout = 3;
            else return 2;
        } else return 2;
    }
    if (groups < 1 || depth < 1 ||
        (depth != 1 && depth != 4 && depth != 5 && depth != 6 && depth != 8)) return 2;

    const std::size_t chunks = streaming
        ? static_cast<std::size_t>(groups) * depth
        : static_cast<std::size_t>(kPositions);
    std::vector<std::uint32_t> host_source(
        chunks * kSourcePositionStride / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < host_source.size(); ++i) {
        host_source[i] = static_cast<std::uint32_t>(i) * 1664525u +
                         1013904223u ^ 0x9e3779b9u;
    }
    char* source = nullptr;
    CUDA_CHECK(cudaMalloc(&source, host_source.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(source, host_source.data(),
                          host_source.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
    launch_for(depth, bypass, streaming, layout, source, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStop());
    std::printf(
        "{\"workload\":\"async_source_only\",\"depth\":%d,"
        "\"route\":\"%s\",\"layout\":\"%s\",\"groups\":%d,\"block\":32,"
        "\"source_position_stride\":%zu,\"source_lane_stride\":%zu,"
        "\"shared_bytes\":%zu,\"destination_position_stride\":%zu,"
        "\"destination_lane_stride\":%zu,"
        "\"global_store_target\":false,\"cta_barrier_target\":false,"
        "\"correct\":null,\"functional_oracle\":\"no_host_visible_result\"}\n",
        depth, streaming ? "dram" : (bypass ? "l2" : "l1"),
        layout == 0
            ? "compact"
            : (layout == 1 ? "conflict" : (layout == 2 ? "severe" : "skew")),
        groups, kSourcePositionStride, kSourceLaneStride, kDestinationBytes,
        kDestinationPositionStride,
        layout == 0
            ? kCompactLaneStride
            : (layout == 1
                   ? kConflictLaneStride
                   : (layout == 2 ? kSevereLaneStride : kSkewLaneStride)));
    CUDA_CHECK(cudaFree(source));
    return 0;
}
