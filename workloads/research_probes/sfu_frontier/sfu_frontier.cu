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
#include <cerrno>
#include <climits>
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
constexpr int kMaxIterations = 1 << 18;
constexpr std::size_t kMaxAllocationBytes = 64ull * 1024ull * 1024ull;

bool parse_int(const char* text, int* value) {
    if (text == nullptr || *text == '\0' || value == nullptr) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (result == nullptr ||
        (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

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
    int blocks = 0;
    int warps = 0;
    int iterations = 0;
    if (!parse_int(argv[1], &blocks) || !parse_int(argv[2], &warps) ||
        !parse_int(argv[3], &iterations)) {
        std::fprintf(stderr, "numeric arguments must be signed decimal integers\n");
        return 2;
    }
    const bool quad = std::strcmp(argv[4], "quad") == 0;
    if (blocks < 1 || blocks > 4096 || (warps != 1 && warps != 2 && warps != 4 && warps != 8) ||
        iterations < 1 || iterations > kMaxIterations ||
        (!quad && std::strcmp(argv[4], "dependent") != 0)) {
        std::fprintf(stderr, "arguments exceed the bounded probe contract\n");
        return 2;
    }
    std::size_t global_warps = 0;
    std::size_t timestamp_count = 0;
    std::size_t timestamp_bytes = 0;
    std::size_t checksum_bytes = 0;
    std::size_t allocation_bytes = 0;
    if (!checked_mul(static_cast<std::size_t>(blocks), static_cast<std::size_t>(warps),
                     &global_warps) ||
        !checked_mul(global_warps, 2, &timestamp_count) ||
        !checked_mul(timestamp_count, sizeof(std::uint64_t), &timestamp_bytes) ||
        !checked_mul(global_warps, sizeof(std::uint32_t), &checksum_bytes) ||
        timestamp_bytes > std::numeric_limits<std::size_t>::max() - checksum_bytes) {
        std::fprintf(stderr, "allocation size overflows size_t\n");
        return 2;
    }
    allocation_bytes = timestamp_bytes + checksum_bytes;
    std::size_t free_bytes = 0;
    std::size_t device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &device_bytes));
    if (allocation_bytes > kMaxAllocationBytes || allocation_bytes > free_bytes / 2) {
        std::fprintf(stderr,
                     "requested %zu bytes exceeds the bounded allocation budget "
                     "(%zu bytes, %zu free)\n",
                     allocation_bytes, kMaxAllocationBytes, free_bytes);
        return 3;
    }
    std::uint64_t* timestamps = nullptr;
    std::uint32_t* checksums = nullptr;
    CUDA_CHECK(cudaMalloc(&timestamps, timestamp_bytes));
    CUDA_CHECK(cudaMalloc(&checksums, checksum_bytes));
    CUDA_CHECK(cudaMemset(timestamps, 0xff, timestamp_bytes));
    CUDA_CHECK(cudaMemset(checksums, 0xa5, checksum_bytes));
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
    bool timestamps_valid = true;
    bool checksums_present = true;
    for (std::size_t warp = 0; warp < global_warps; ++warp) {
        const std::uint64_t start = host_timestamps[warp * 2];
        const std::uint64_t stop = host_timestamps[warp * 2 + 1];
        timestamps_valid = timestamps_valid && start != UINT64_MAX && stop != UINT64_MAX && stop >= start;
        const auto cycles = stop >= start ? stop - start : 0;
        min_cycles = std::min(min_cycles, cycles);
        max_cycles = std::max(max_cycles, cycles);
        checksums_present = checksums_present && host_checksums[warp] != 0xa5a5a5a5u;
        checksum = checksum * 131u + host_checksums[warp];
    }
    const bool correct = timestamps_valid && checksums_present;
    std::printf("{\"blocks\":%d,\"warps\":%d,\"iterations\":%d,\"mode\":\"%s\","
                "\"global_warps\":%zu,\"min_cycles\":%llu,\"max_cycles\":%llu,"
                "\"cycles_per_iteration_min\":%.6f,\"cycles_per_iteration_max\":%.6f,"
                "\"checksum\":%llu,\"correct\":%s,\"timestamp_bounds\":%s,"
                "\"checksum_present\":%s,\"allocation_bytes\":%zu}\n",
                blocks, warps, iterations, quad ? "quad" : "dependent", global_warps,
                static_cast<unsigned long long>(min_cycles), static_cast<unsigned long long>(max_cycles),
                static_cast<double>(min_cycles) / iterations, static_cast<double>(max_cycles) / iterations,
                static_cast<unsigned long long>(checksum), correct ? "true" : "false",
                timestamps_valid ? "true" : "false", checksums_present ? "true" : "false",
                allocation_bytes);
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(checksums));
    return correct ? 0 : 1;
}
