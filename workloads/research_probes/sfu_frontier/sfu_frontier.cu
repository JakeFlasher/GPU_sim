// Unscored SM86 special-function (MUFU) latency/throughput probe.
//
// The kernel records a per-warp clock interval around either a compile-time
// one-chain or four-chain MUFU.EX2 body.  The chain count is a template
// parameter, so predication and a runtime mode branch cannot accidentally
// leave both bodies in the traced instruction stream.  It is deliberately separate
// from the scored simulator workloads: the result is used only to validate an
// architecture capability, never as a workload-specific timing correction.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

__device__ __forceinline__ float mufu_ex2(float value) {
    asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(value));
    return value;
}

template <int Warps, int Chains>
__global__ __launch_bounds__(Warps * kWarpSize)
void sfu_kernel(std::uint64_t* timestamps, std::uint32_t* checksums,
                int iterations) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::size_t global_warp =
        static_cast<std::size_t>(blockIdx.x) * Warps + static_cast<std::size_t>(warp);
    float x = 0.001f + static_cast<float>((lane & 7) + 1) * 0.0001f;
    float y = x + 0.25f;
    float z = x + 0.5f;
    float w = x + 0.75f;
    __syncthreads();
    const std::uint64_t start = clock64();
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        if constexpr (Chains == 1) {
            x = mufu_ex2(x);
        } else {
            x = mufu_ex2(x);
            y = mufu_ex2(y);
            z = mufu_ex2(z);
            w = mufu_ex2(w);
        }
    }
    const std::uint64_t stop = clock64();
    if (lane == 0) {
        timestamps[global_warp * 2] = start;
        timestamps[global_warp * 2 + 1] = stop;
        checksums[global_warp] = __float_as_uint(x) ^ __float_as_uint(y) ^
                                 __float_as_uint(z) ^ __float_as_uint(w);
    }
}

template <int Chains>
void launch_chains(int warps, int blocks, std::uint64_t* timestamps,
                   std::uint32_t* checksums, int iterations) {
    switch (warps) {
        case 1: sfu_kernel<1, Chains><<<blocks, 32>>>(timestamps, checksums, iterations); break;
        case 2: sfu_kernel<2, Chains><<<blocks, 64>>>(timestamps, checksums, iterations); break;
        case 4: sfu_kernel<4, Chains><<<blocks, 128>>>(timestamps, checksums, iterations); break;
        case 8: sfu_kernel<8, Chains><<<blocks, 256>>>(timestamps, checksums, iterations); break;
        default: std::fprintf(stderr, "warps must be one of 1,2,4,8\n"); std::exit(2);
    }
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s BLOCKS WARPS ITERATIONS MODE(dependent|quad)\n", argv[0]);
        return 2;
    }
    const int blocks = std::atoi(argv[1]);
    const int warps = std::atoi(argv[2]);
    const int iterations = std::atoi(argv[3]);
    const bool quad = std::strcmp(argv[4], "quad") == 0;
    if (blocks < 1 || blocks > 4096 || (warps != 1 && warps != 2 && warps != 4 && warps != 8) ||
        iterations < 1 || iterations > (1 << 20) || (!quad && std::strcmp(argv[4], "dependent") != 0)) {
        return 2;
    }
    const std::size_t global_warps = static_cast<std::size_t>(blocks) * static_cast<std::size_t>(warps);
    std::uint64_t* timestamps = nullptr;
    std::uint32_t* checksums = nullptr;
    CUDA_CHECK(cudaMalloc(&timestamps, global_warps * 2 * sizeof(*timestamps)));
    CUDA_CHECK(cudaMalloc(&checksums, global_warps * sizeof(*checksums)));
    for (int warmup = 0; warmup < 3; ++warmup) {
        if (quad) launch_chains<4>(warps, blocks, timestamps, checksums, iterations);
        else launch_chains<1>(warps, blocks, timestamps, checksums, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (quad) launch_chains<4>(warps, blocks, timestamps, checksums, iterations);
    else launch_chains<1>(warps, blocks, timestamps, checksums, iterations);
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
    std::printf("{\"blocks\":%d,\"warps\":%d,\"iterations\":%d,\"mode\":\"%s\","
                "\"global_warps\":%zu,\"min_cycles\":%llu,\"max_cycles\":%llu,"
                "\"cycles_per_iteration_min\":%.6f,\"cycles_per_iteration_max\":%.6f,"
                "\"checksum\":%llu}\n",
                blocks, warps, iterations, quad ? "quad" : "dependent", global_warps,
                static_cast<unsigned long long>(min_cycles), static_cast<unsigned long long>(max_cycles),
                static_cast<double>(min_cycles) / iterations, static_cast<double>(max_cycles) / iterations,
                static_cast<unsigned long long>(checksum));
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(checksums));
    return 0;
}
