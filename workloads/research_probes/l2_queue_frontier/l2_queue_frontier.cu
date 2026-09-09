// Unscored SM86 L2 request/response-pressure probe.
//
// The measured body contains only independent .cg global loads.  It varies
// resident warp count, load burst depth, address stride, and reuse while
// keeping the SASS family fixed.  NCU is used only to observe the resulting
// L1TEX/LTS interfaces; no scored-workload target is consumed by the probe or
// by the simulator.

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
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;

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
    const int blocks = std::atoi(argv[2]);
    const int warps = std::atoi(argv[3]);
    const int loads = std::atoi(argv[4]);
    const int stride_lines = std::atoi(argv[5]);
    const int reuse_lines = std::atoi(argv[6]);
    const int warmups = std::atoi(argv[7]);
    if (blocks < 1 || blocks > 4096 ||
        (warps != 1 && warps != 2 && warps != 4 && warps != 8) ||
        loads < 1 || loads > (1 << 16) || stride_lines < 1 || stride_lines > (1 << 16) ||
        reuse_lines < 1 || reuse_lines > (1 << 16) || warmups < 1 || warmups > 16) {
        return 2;
    }
    const std::size_t global_warps = static_cast<std::size_t>(blocks) * warps;
    const std::size_t lines = unique
        ? global_warps * static_cast<std::size_t>(loads) * static_cast<std::size_t>(stride_lines)
        : global_warps * static_cast<std::size_t>(reuse_lines) * static_cast<std::size_t>(stride_lines);
    const std::size_t words = lines * kWarpSize;
    std::uint32_t* data = nullptr;
    std::uint64_t* timestamps = nullptr;
    std::uint32_t* checksums = nullptr;
    CUDA_CHECK(cudaMalloc(&data, words * sizeof(*data)));
    CUDA_CHECK(cudaMalloc(&timestamps, global_warps * 2 * sizeof(*timestamps)));
    CUDA_CHECK(cudaMalloc(&checksums, global_warps * sizeof(*checksums)));
    CUDA_CHECK(cudaMemset(data, 0x5a, words * sizeof(*data)));
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
    for (std::size_t warp = 0; warp < global_warps; ++warp) {
        const std::uint64_t cycles = host_timestamps[warp * 2 + 1] - host_timestamps[warp * 2];
        min_cycles = std::min(min_cycles, cycles);
        max_cycles = std::max(max_cycles, cycles);
        checksum = checksum * 131u + host_checksums[warp];
    }
    std::printf(
        "{\"mode\":\"%s\",\"blocks\":%d,\"warps\":%d,\"loads\":%d,"
        "\"stride_lines\":%d,\"reuse_lines\":%d,\"warmups\":%d,"
        "\"global_warps\":%zu,\"min_cycles\":%llu,\"max_cycles\":%llu,"
        "\"checksum\":%llu}\n",
        mode_name(reuse ? 1 : 0), blocks, warps, loads, stride_lines, reuse_lines,
        warmups, global_warps,
        static_cast<unsigned long long>(min_cycles),
        static_cast<unsigned long long>(max_cycles),
        static_cast<unsigned long long>(checksum));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(checksums));
    return 0;
}
