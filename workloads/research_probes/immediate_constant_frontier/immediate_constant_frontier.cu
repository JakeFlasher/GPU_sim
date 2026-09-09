// Independent SM86 Immediate Constant Cache (IMC) probe.
//
// The timed body uses inline SASS-visible c[bank][offset] operands.  Sites is
// compile-time specialized so the instruction stream contains exactly the
// requested number of CBANK references; no runtime branch or workload target
// participates in the measurement.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(status));                          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;

template <int Sites>
__device__ __forceinline__ std::uint32_t cbank_step(std::uint32_t value,
                                                    std::uint32_t a,
                                                    std::uint32_t b,
                                                    std::uint32_t c,
                                                    std::uint32_t d) {
    if constexpr (Sites >= 1) {
        value += a;
    }
    if constexpr (Sites >= 2) {
        value += b;
    }
    if constexpr (Sites >= 3) {
        value += c;
    }
    if constexpr (Sites >= 4) {
        value += d;
    }
    return value;
}

template <int Warps, int Sites>
__global__ __launch_bounds__(Warps * kWarpSize)
void immediate_constant_kernel(std::uint64_t* timestamps,
                               std::uint32_t* checksums, int iterations,
                               volatile std::uint32_t a, volatile std::uint32_t b,
                               volatile std::uint32_t c, volatile std::uint32_t d) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::size_t global_warp =
        static_cast<std::size_t>(blockIdx.x) * Warps + static_cast<std::size_t>(warp);
    std::uint32_t value = 0x9e3779b9u ^ static_cast<std::uint32_t>(lane + warp * 17);
    __syncthreads();
    const std::uint64_t start = clock64();
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        value = cbank_step<Sites>(value, a, b, c, d);
        asm volatile("" ::: "memory");
    }
    const std::uint64_t stop = clock64();
    if (lane == 0) {
        timestamps[global_warp * 2] = start;
        timestamps[global_warp * 2 + 1] = stop;
        checksums[global_warp] = value;
    }
}

template <int Sites>
void launch(int warps, int blocks, std::uint64_t* timestamps,
            std::uint32_t* checksums, int iterations,
            std::uint32_t a, std::uint32_t b, std::uint32_t c,
            std::uint32_t d) {
    switch (warps) {
        case 1: immediate_constant_kernel<1, Sites><<<blocks, 32>>>(timestamps, checksums, iterations, a, b, c, d); break;
        case 2: immediate_constant_kernel<2, Sites><<<blocks, 64>>>(timestamps, checksums, iterations, a, b, c, d); break;
        case 4: immediate_constant_kernel<4, Sites><<<blocks, 128>>>(timestamps, checksums, iterations, a, b, c, d); break;
        case 8: immediate_constant_kernel<8, Sites><<<blocks, 256>>>(timestamps, checksums, iterations, a, b, c, d); break;
        default: std::fprintf(stderr, "warps must be one of 1,2,4,8\n"); std::exit(2);
    }
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s BLOCKS WARPS ITERATIONS SITES(1|2|3|4)\n", argv[0]);
        return 2;
    }
    const int blocks = std::atoi(argv[1]);
    const int warps = std::atoi(argv[2]);
    const int iterations = std::atoi(argv[3]);
    const int sites = std::atoi(argv[4]);
    if (blocks < 1 || blocks > 4096 || (warps != 1 && warps != 2 && warps != 4 && warps != 8) ||
        iterations < 1 || iterations > (1 << 20) || sites < 1 || sites > 4) return 2;
    const std::size_t global_warps = static_cast<std::size_t>(blocks) * static_cast<std::size_t>(warps);
    std::uint64_t* timestamps = nullptr;
    std::uint32_t* checksums = nullptr;
    CUDA_CHECK(cudaMalloc(&timestamps, global_warps * 2 * sizeof(*timestamps)));
    CUDA_CHECK(cudaMalloc(&checksums, global_warps * sizeof(*checksums)));
    constexpr std::uint32_t a = 0x13579bdfu;
    constexpr std::uint32_t b = 0x2468ace0u;
    constexpr std::uint32_t c = 0x0f1e2d3cu;
    constexpr std::uint32_t d = 0x4b5a6978u;
    for (int warmup = 0; warmup < 3; ++warmup) {
        if (sites == 1) launch<1>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
        else if (sites == 2) launch<2>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
        else if (sites == 3) launch<3>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
        else launch<4>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (sites == 1) launch<1>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
    else if (sites == 2) launch<2>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
    else if (sites == 3) launch<3>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
    else launch<4>(warps, blocks, timestamps, checksums, iterations, a, b, c, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint64_t> host_timestamps(global_warps * 2);
    std::vector<std::uint32_t> host_checksums(global_warps);
    CUDA_CHECK(cudaMemcpy(host_timestamps.data(), timestamps,
                          host_timestamps.size() * sizeof(*timestamps), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_checksums.data(), checksums,
                          host_checksums.size() * sizeof(*checksums), cudaMemcpyDeviceToHost));
    std::uint64_t min_cycles = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_cycles = 0;
    std::uint64_t checksum = 0;
    for (std::size_t warp = 0; warp < global_warps; ++warp) {
        const auto cycles = host_timestamps[warp * 2 + 1] - host_timestamps[warp * 2];
        min_cycles = std::min(min_cycles, cycles);
        max_cycles = std::max(max_cycles, cycles);
        checksum = checksum * 131u + host_checksums[warp];
    }
    std::printf("{\"blocks\":%d,\"warps\":%d,\"iterations\":%d,\"sites\":%d,"
                "\"global_warps\":%zu,\"min_cycles\":%llu,\"max_cycles\":%llu,"
                "\"cycles_per_iteration_min\":%.6f,\"cycles_per_iteration_max\":%.6f,"
                "\"checksum\":%llu}\n",
                blocks, warps, iterations, sites, global_warps,
                static_cast<unsigned long long>(min_cycles), static_cast<unsigned long long>(max_cycles),
                static_cast<double>(min_cycles) / iterations, static_cast<double>(max_cycles) / iterations,
                static_cast<unsigned long long>(checksum));
    CUDA_CHECK(cudaFree(checksums));
    CUDA_CHECK(cudaFree(timestamps));
    return 0;
}
