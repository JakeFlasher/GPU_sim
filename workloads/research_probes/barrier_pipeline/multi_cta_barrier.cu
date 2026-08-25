#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

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
constexpr int kTotalWarpsPerSm = 8;
constexpr int kSharedBytesPerSm = 96 * 1024;

#ifndef MULTI_CTA_BARRIER_GENERATIONS
constexpr int kGenerations = 16384;
#else
constexpr int kGenerations = MULTI_CTA_BARRIER_GENERATIONS;
#endif

enum Protocol : int { kControl = 0, kOneBarrier = 1 };

__device__ __forceinline__ unsigned current_smid() {
    unsigned value;
    asm volatile("mov.u32 %0, %smid;" : "=r"(value));
    return value;
}

template <int ProtocolValue>
__global__ void multi_cta_barrier(unsigned long long* elapsed,
                                  std::uint32_t* accumulators,
                                  unsigned* block_smids) {
    extern __shared__ unsigned char residency[];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    std::uint32_t value = 0x6a09e667u ^
                          static_cast<std::uint32_t>(warp * 131 + lane);
    if (threadIdx.x == 0) {
        residency[0] = static_cast<unsigned char>(blockIdx.x);
    }
    __syncthreads();

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        value = value * 1664525u + 1013904223u;
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        if constexpr (ProtocolValue == kOneBarrier) {
            __syncthreads();
        }
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    const unsigned long long stop = clock64();

    if (threadIdx.x == 0) {
        elapsed[blockIdx.x] = stop - start;
        accumulators[blockIdx.x] = value ^ residency[0];
        block_smids[blockIdx.x] = current_smid();
    }
}

template <int ProtocolValue>
void configure_and_launch(int blocks, int threads, int shared_bytes,
                          unsigned long long* elapsed,
                          std::uint32_t* accumulators, unsigned* block_smids) {
    CUDA_CHECK(cudaFuncSetAttribute(
        multi_cta_barrier<ProtocolValue>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    multi_cta_barrier<ProtocolValue><<<blocks, threads, shared_bytes>>>(
        elapsed, accumulators, block_smids);
}

void launch_protocol(int protocol, int blocks, int threads, int shared_bytes,
                     unsigned long long* elapsed,
                     std::uint32_t* accumulators, unsigned* block_smids) {
    if (protocol == kControl) {
        configure_and_launch<kControl>(blocks, threads, shared_bytes, elapsed,
                                      accumulators, block_smids);
    } else {
        configure_and_launch<kOneBarrier>(blocks, threads, shared_bytes,
                                         elapsed, accumulators, block_smids);
    }
}

const char* protocol_name(int protocol) {
    return protocol == kControl ? "m00_control" : "m10_one_barrier";
}

bool parse_protocol(const char* text, int* protocol) {
    if (std::strcmp(text, "m00") == 0 ||
        std::strcmp(text, "m00_control") == 0) {
        *protocol = kControl;
        return true;
    }
    if (std::strcmp(text, "m10") == 0 ||
        std::strcmp(text, "m10_one_barrier") == 0) {
        *protocol = kOneBarrier;
        return true;
    }
    return false;
}

std::uint32_t expected_accumulator(int block, int warp, int lane) {
    std::uint32_t value = 0x6a09e667u ^
                          static_cast<std::uint32_t>(warp * 131 + lane);
    for (int generation = 0; generation < kGenerations; ++generation) {
        value = value * 1664525u + 1013904223u;
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    return value ^ static_cast<unsigned char>(block);
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s PROTOCOL CTAS_PER_SM WARPS_PER_CTA [TRIALS]\n",
                     argv[0]);
        return 2;
    }
    int protocol = -1;
    const int ctas_per_sm = std::atoi(argv[2]);
    const int warps_per_cta = std::atoi(argv[3]);
    const int trials = argc == 5 ? std::atoi(argv[4]) : 7;
    if (!parse_protocol(argv[1], &protocol) ||
        (ctas_per_sm != 1 && ctas_per_sm != 2 && ctas_per_sm != 4) ||
        ctas_per_sm * warps_per_cta != kTotalWarpsPerSm ||
        trials < 1 || trials > 32) {
        return 2;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.multiProcessorCount <= 0 ||
        properties.sharedMemPerMultiprocessor < kSharedBytesPerSm) {
        std::fprintf(stderr, "device cannot support the declared residency construction\n");
        return 2;
    }
    const int blocks = properties.multiProcessorCount * ctas_per_sm;
    const int threads = warps_per_cta * kWarpSize;
    const int shared_bytes = kSharedBytesPerSm / ctas_per_sm;

    unsigned long long* elapsed = nullptr;
    std::uint32_t* accumulators = nullptr;
    unsigned* block_smids = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed, blocks * sizeof(*elapsed)));
    CUDA_CHECK(cudaMalloc(&accumulators, blocks * sizeof(*accumulators)));
    CUDA_CHECK(cudaMalloc(&block_smids, blocks * sizeof(*block_smids)));

    std::vector<unsigned long long> host_elapsed(blocks);
    std::vector<unsigned> host_smids(blocks);
    std::vector<std::uint32_t> host_accumulators(blocks);
    std::vector<unsigned long long> sample_min(trials);
    std::vector<unsigned long long> sample_max(trials);
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaProfilerStart());
        launch_protocol(protocol, blocks, threads, shared_bytes, elapsed,
                        accumulators, block_smids);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(host_elapsed.data(), elapsed,
                              blocks * sizeof(*elapsed), cudaMemcpyDeviceToHost));
        sample_min[trial] = host_elapsed[0];
        sample_max[trial] = host_elapsed[0];
        for (unsigned long long value : host_elapsed) {
            if (value < sample_min[trial]) sample_min[trial] = value;
            if (value > sample_max[trial]) sample_max[trial] = value;
        }
    }
    CUDA_CHECK(cudaMemcpy(host_smids.data(), block_smids,
                          blocks * sizeof(*block_smids), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_accumulators.data(), accumulators,
                          blocks * sizeof(*accumulators), cudaMemcpyDeviceToHost));

    bool correct = true;
    std::vector<int> sm_block_counts(properties.multiProcessorCount, 0);
    std::uint64_t checksum = 0;
    for (int block = 0; block < blocks; ++block) {
        correct = correct &&
                  host_accumulators[block] == expected_accumulator(block, 0, 0);
        if (host_smids[block] >= static_cast<unsigned>(properties.multiProcessorCount)) {
            correct = false;
        } else {
            ++sm_block_counts[host_smids[block]];
        }
        checksum = checksum * 131u + host_accumulators[block];
    }
    for (int count : sm_block_counts) {
        correct = correct && count == ctas_per_sm;
    }

    std::printf(
        "{\"protocol\":\"%s\",\"ctas_per_sm\":%d,"
        "\"warps_per_cta\":%d,\"total_warps_per_sm\":%d,"
        "\"sm_count\":%d,\"blocks\":%d,\"shared_bytes_per_cta\":%d,"
        "\"generations\":%d,\"sample_min_cycles\":[",
        protocol_name(protocol), ctas_per_sm, warps_per_cta,
        kTotalWarpsPerSm, properties.multiProcessorCount, blocks, shared_bytes,
        kGenerations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s%llu", trial == 0 ? "" : ",", sample_min[trial]);
    }
    std::printf("],\"sample_max_cycles\":[");
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s%llu", trial == 0 ? "" : ",", sample_max[trial]);
    }
    std::printf("],\"sm_block_counts\":[");
    for (int sm = 0; sm < properties.multiProcessorCount; ++sm) {
        std::printf("%s%d", sm == 0 ? "" : ",", sm_block_counts[sm]);
    }
    std::printf(
        "],\"accumulator_checksum\":%llu,\"correct\":%s}\n",
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(block_smids));
    CUDA_CHECK(cudaFree(accumulators));
    CUDA_CHECK(cudaFree(elapsed));
    return correct ? 0 : 1;
}
