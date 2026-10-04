// Unscored SM86 L2 request/response-pressure probe.
//
// The measured body contains only independent .cg global loads.  It varies
// resident warp count, load burst depth, address stride, and reuse while
// keeping the SASS family fixed.  NCU is used only to observe the resulting
// L1TEX/LTS interfaces; no scored-workload target is consumed by the probe or
// by the simulator.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) {
    if (result == nullptr || rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

std::uint32_t expected_warp_checksum(std::size_t global_warp, int loads) {
    std::uint32_t checksum = static_cast<std::uint32_t>(global_warp + 1);
    constexpr std::uint32_t kDataWord = 0x5a5a5a5au;
    for (int index = 0; index < loads; ++index) {
        // Only lane zero publishes its private checksum.  Each lane executes
        // the same loop, but the kernel does not combine the 32 lane values.
        checksum ^= kDataWord + static_cast<std::uint32_t>(index * 17);
    }
    return checksum;
}

__device__ __forceinline__ std::uint32_t load_cg(const std::uint32_t* ptr) {
    std::uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];\n"
                 : "=r"(value)
                 : "l"(ptr));
    return value;
}

template <int Warps>
__global__ __launch_bounds__(Warps * kWarpSize)
void l2_queue_kernel(const std::uint32_t* data,
                     std::uint64_t* timestamps,
                     std::uint32_t* checksums,
                     int loads,
                     int stride_lines,
                     int reuse_lines,
                     int mode) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::size_t global_warp =
        static_cast<std::size_t>(blockIdx.x) * Warps + warp;
    const std::size_t slot =
        static_cast<std::size_t>(global_warp) * static_cast<std::size_t>(loads);
    __syncthreads();
    const std::uint64_t start = clock64();
    std::uint32_t checksum = static_cast<std::uint32_t>(global_warp + 1);
#pragma unroll 1
    for (int index = 0; index < loads; ++index) {
        std::size_t line = slot + static_cast<std::size_t>(index);
        if (mode != 0) {
            line = static_cast<std::size_t>(global_warp) *
                       static_cast<std::size_t>(reuse_lines) +
                   static_cast<std::size_t>(index % reuse_lines);
        }
        line *= static_cast<std::size_t>(stride_lines);
        const std::uint32_t* address = data + line * kWarpSize + lane;
        checksum ^= load_cg(address) + static_cast<std::uint32_t>(index * 17 + lane);
    }
    const std::uint64_t stop = clock64();
    if (lane == 0) {
        timestamps[global_warp * 2 + 0] = start;
        timestamps[global_warp * 2 + 1] = stop;
        checksums[global_warp] = checksum;
    }
}

const char* mode_name(int mode) {
    return mode == 0 ? "unique" : "reuse";
}

void launch(int warps, int blocks, const std::uint32_t* data,
            std::uint64_t* timestamps, std::uint32_t* checksums,
            int loads, int stride_lines, int reuse_lines, int mode) {
    switch (warps) {
        case 1:
            l2_queue_kernel<1><<<blocks, 32>>>(data, timestamps, checksums, loads, stride_lines, reuse_lines, mode);
            break;
        case 2:
            l2_queue_kernel<2><<<blocks, 64>>>(data, timestamps, checksums, loads, stride_lines, reuse_lines, mode);
            break;
        case 4:
            l2_queue_kernel<4><<<blocks, 128>>>(data, timestamps, checksums, loads, stride_lines, reuse_lines, mode);
            break;
        case 8:
            l2_queue_kernel<8><<<blocks, 256>>>(data, timestamps, checksums, loads, stride_lines, reuse_lines, mode);
            break;
        default:
            std::fprintf(stderr, "warps must be one of 1,2,4,8\n");
            std::exit(2);
    }
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: %s MODE BLOCKS WARPS LOADS STRIDE_LINES REUSE_LINES WARMUPS\n",
                     argv[0]);
        return 2;
    }
    const bool unique = std::strcmp(argv[1], "unique") == 0;
    const bool reuse = std::strcmp(argv[1], "reuse") == 0;
    if (!unique && !reuse) return 2;
    int blocks = 0;
    int warps = 0;
    int loads = 0;
    int stride_lines = 0;
    int reuse_lines = 0;
    int warmups = 0;
    if (!parse_int(argv[2], &blocks) || !parse_int(argv[3], &warps) ||
        !parse_int(argv[4], &loads) || !parse_int(argv[5], &stride_lines) ||
        !parse_int(argv[6], &reuse_lines) || !parse_int(argv[7], &warmups)) {
        std::fprintf(stderr, "all numeric arguments must be signed decimal integers\n");
        return 2;
    }
    if (blocks < 1 || blocks > 4096 ||
        (warps != 1 && warps != 2 && warps != 4 && warps != 8) ||
        loads < 1 || loads > (1 << 16) || stride_lines < 1 || stride_lines > (1 << 16) ||
        reuse_lines < 1 || reuse_lines > (1 << 16) || warmups < 1 || warmups > 16) {
        std::fprintf(stderr, "arguments exceed the bounded probe contract\n");
        return 2;
    }
    const std::size_t global_warps = static_cast<std::size_t>(blocks) * warps;
    std::size_t line_count = 0;
    const std::size_t line_span = unique ? static_cast<std::size_t>(loads)
                                         : static_cast<std::size_t>(reuse_lines);
    if (!checked_mul(global_warps, line_span, &line_count) ||
        !checked_mul(line_count, static_cast<std::size_t>(stride_lines), &line_count)) {
        std::fprintf(stderr, "data extent overflows size_t\n");
        return 2;
    }
    std::size_t words = 0;
    if (!checked_mul(line_count, static_cast<std::size_t>(kWarpSize), &words)) {
        std::fprintf(stderr, "data word count overflows size_t\n");
        return 2;
    }
    std::size_t data_bytes = 0;
    std::size_t timestamp_count = 0;
    std::size_t timestamp_bytes = 0;
    std::size_t checksum_bytes = 0;
    std::size_t total_bytes = 0;
    if (!checked_mul(words, sizeof(std::uint32_t), &data_bytes) ||
        !checked_mul(global_warps, 2, &timestamp_count) ||
        !checked_mul(timestamp_count, sizeof(std::uint64_t), &timestamp_bytes) ||
        !checked_mul(global_warps, sizeof(std::uint32_t), &checksum_bytes) ||
        !checked_add(data_bytes, timestamp_bytes, &total_bytes) ||
        !checked_add(total_bytes, checksum_bytes, &total_bytes)) {
        std::fprintf(stderr, "allocation size overflows size_t\n");
        return 2;
    }
    std::size_t free_bytes = 0;
    std::size_t device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &device_bytes));
    // Leave headroom for the CUDA context, profiler, and allocator metadata.
    if (total_bytes > free_bytes / 2) {
        std::fprintf(stderr,
                     "requested %zu bytes exceeds the bounded half-free-memory budget "
                     "(%zu free)\n",
                     total_bytes, free_bytes);
        return 3;
    }
    std::uint32_t* data = nullptr;
    std::uint64_t* timestamps = nullptr;
    std::uint32_t* checksums = nullptr;
    CUDA_CHECK(cudaMalloc(&data, data_bytes));
    CUDA_CHECK(cudaMalloc(&timestamps, timestamp_bytes));
    CUDA_CHECK(cudaMalloc(&checksums, checksum_bytes));
    CUDA_CHECK(cudaMemset(data, 0x5a, data_bytes));
    CUDA_CHECK(cudaMemset(timestamps, 0xff, timestamp_bytes));
    CUDA_CHECK(cudaMemset(checksums, 0xff, checksum_bytes));
    for (int warmup = 0; warmup < warmups; ++warmup) {
        launch(warps, blocks, data, timestamps, checksums, loads, stride_lines, reuse_lines, reuse ? 1 : 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    launch(warps, blocks, data, timestamps, checksums, loads, stride_lines, reuse_lines, reuse ? 1 : 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint64_t> host_timestamps(global_warps * 2);
    std::vector<std::uint32_t> host_checksums(global_warps);
    CUDA_CHECK(cudaMemcpy(host_timestamps.data(), timestamps,
                          host_timestamps.size() * sizeof(*timestamps),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_checksums.data(), checksums,
                          host_checksums.size() * sizeof(*checksums),
                          cudaMemcpyDeviceToHost));
    std::uint64_t min_cycles = UINT64_MAX;
    std::uint64_t max_cycles = 0;
    std::uint64_t checksum = 0;
    bool timestamps_valid = true;
    bool checksums_valid = true;
    for (std::size_t warp = 0; warp < global_warps; ++warp) {
        const std::uint64_t start = host_timestamps[warp * 2 + 0];
        const std::uint64_t stop = host_timestamps[warp * 2 + 1];
        timestamps_valid = timestamps_valid && start != UINT64_MAX && stop != UINT64_MAX && stop >= start;
        const std::uint64_t cycles = stop >= start ? stop - start : 0;
        min_cycles = std::min(min_cycles, cycles);
        max_cycles = std::max(max_cycles, cycles);
        checksums_valid = checksums_valid &&
            host_checksums[warp] == expected_warp_checksum(warp, loads);
        checksum = checksum * 131u + host_checksums[warp];
    }
    const bool correct = timestamps_valid && checksums_valid;
    std::printf(
        "{\"mode\":\"%s\",\"blocks\":%d,\"warps\":%d,\"loads\":%d,"
        "\"stride_lines\":%d,\"reuse_lines\":%d,\"warmups\":%d,"
        "\"global_warps\":%zu,\"min_cycles\":%llu,\"max_cycles\":%llu,"
        "\"checksum\":%llu,\"correct\":%s,\"timestamp_bounds\":%s,"
        "\"checksum_expected\":%s,\"allocation_bytes\":%zu}\n",
        mode_name(reuse ? 1 : 0), blocks, warps, loads, stride_lines, reuse_lines,
        warmups, global_warps,
        static_cast<unsigned long long>(min_cycles),
        static_cast<unsigned long long>(max_cycles),
        static_cast<unsigned long long>(checksum), correct ? "true" : "false",
        timestamps_valid ? "true" : "false", checksums_valid ? "true" : "false",
        total_bytes);
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(checksums));
    return correct ? 0 : 4;
}
