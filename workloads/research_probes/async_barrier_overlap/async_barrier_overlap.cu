#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
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
constexpr int kBytesPerLane = 16;
constexpr int kPayloadBytes = kWarpSize * kBytesPerLane;
constexpr int kCopiesPerGroup = 8;
constexpr int kMaxWarps = 8;
constexpr int kSharedBytesPerWarp = kCopiesPerGroup * kPayloadBytes;

#ifndef ASYNC_BARRIER_GROUPS
constexpr int kGroups = 16384;
#else
constexpr int kGroups = ASYNC_BARRIER_GROUPS;
#endif

enum Protocol : int {
    kBaseline = 0,
    kBarrierOnly = 1,
    kWaitOnly = 2,
    kWaitBarrier = 3,
    kProducerConsumer = 4,
};

struct Locality {
    const char* name;
    std::size_t region_bytes;
    std::size_t source_stride;
    bool prewarm;
};

__device__ __forceinline__ void copy_cache_all(unsigned destination,
                                                const char* source) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                     "r"(destination), "l"(source)
                 : "memory");
}

__device__ __forceinline__ std::uint32_t independent_gap_64(
    std::uint32_t value) {
#pragma unroll
    for (int step = 0; step < 7; ++step) {
        asm volatile(
            "mul.lo.u32 %0, %0, 1664525;\n"
            "xor.b32 %0, %0, 1013904223;\n"
            : "+r"(value));
    }
    return value;
}

__global__ void prewarm_source(const std::uint32_t* source,
                               std::size_t words,
                               std::uint32_t* prewarm_sink) {
    std::uint32_t value = 0;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < words; index += gridDim.x * blockDim.x) {
        value ^= source[index];
    }
    if (value != 0) {
        atomicXor(prewarm_sink, value);
    }
}

template <int ProtocolValue>
__global__ void async_barrier_overlap(
    const char* source, char* sink, unsigned long long* elapsed,
    std::uint32_t* accumulator_sink, int gap_cycles,
    std::size_t source_stride, std::size_t source_mask) {
    extern __shared__ __align__(16) char shared[];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    char* const warp_shared = shared + warp * kSharedBytesPerWarp;
    bool async_warp = false;
    if constexpr (ProtocolValue == kWaitOnly ||
                  ProtocolValue == kWaitBarrier) {
        async_warp = true;
    } else if constexpr (ProtocolValue == kProducerConsumer) {
        async_warp = warp == 0;
    }
    std::uint32_t accumulator =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);

    __syncthreads();
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int group = 0; group < kGroups; ++group) {
        const std::size_t source_offset =
            (static_cast<std::size_t>(group) * source_stride) & source_mask;
        const char* const group_source = source + source_offset;
        accumulator ^= static_cast<std::uint32_t>(source_offset) +
                       static_cast<std::uint32_t>(group) +
                       static_cast<std::uint32_t>(
                           reinterpret_cast<std::uintptr_t>(group_source) >> 4);

        if (async_warp) {
#pragma unroll
            for (int copy = 0; copy < kCopiesPerGroup; ++copy) {
                const unsigned destination = static_cast<unsigned>(
                    __cvta_generic_to_shared(
                        warp_shared + copy * kPayloadBytes +
                        lane * kBytesPerLane));
                copy_cache_all(destination,
                               group_source + lane * kBytesPerLane);
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        if (gap_cycles != 0) {
            accumulator = independent_gap_64(accumulator);
        }

        if (async_warp) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        if constexpr (ProtocolValue == kBarrierOnly ||
                      ProtocolValue == kWaitBarrier ||
                      ProtocolValue == kProducerConsumer) {
            __syncthreads();
        }
        accumulator ^= 0xbb67ae85u + static_cast<std::uint32_t>(group);
    }
    const unsigned long long stop = clock64();

    if (async_warp) {
        const uint4 value = reinterpret_cast<const uint4*>(
            warp_shared + (kCopiesPerGroup - 1) * kPayloadBytes)[lane];
        reinterpret_cast<uint4*>(sink + warp * kPayloadBytes)[lane] = value;
    } else {
        reinterpret_cast<std::uint32_t*>(
            sink + warp * kPayloadBytes)[lane] = accumulator;
    }
    if (lane == 0) {
        elapsed[warp] = stop - start;
        accumulator_sink[warp] = accumulator;
    }
}

template <int ProtocolValue>
void launch_protocol_kernel(
    int warps, const char* source, char* sink, unsigned long long* elapsed,
    std::uint32_t* accumulator_sink, int gap_cycles,
    std::size_t source_stride, std::size_t source_mask) {
    async_barrier_overlap<ProtocolValue>
        <<<1, warps * kWarpSize, warps * kSharedBytesPerWarp>>>(
            source, sink, elapsed, accumulator_sink, gap_cycles,
            source_stride, source_mask);
}

void launch_protocol(
    int protocol, int warps, const char* source, char* sink,
    unsigned long long* elapsed, std::uint32_t* accumulator_sink,
    int gap_cycles, std::size_t source_stride, std::size_t source_mask) {
    switch (protocol) {
        case kBaseline:
            launch_protocol_kernel<kBaseline>(
                warps, source, sink, elapsed, accumulator_sink, gap_cycles,
                source_stride, source_mask);
            break;
        case kBarrierOnly:
            launch_protocol_kernel<kBarrierOnly>(
                warps, source, sink, elapsed, accumulator_sink, gap_cycles,
                source_stride, source_mask);
            break;
        case kWaitOnly:
            launch_protocol_kernel<kWaitOnly>(
                warps, source, sink, elapsed, accumulator_sink, gap_cycles,
                source_stride, source_mask);
            break;
        case kWaitBarrier:
            launch_protocol_kernel<kWaitBarrier>(
                warps, source, sink, elapsed, accumulator_sink, gap_cycles,
                source_stride, source_mask);
            break;
        case kProducerConsumer:
            launch_protocol_kernel<kProducerConsumer>(
                warps, source, sink, elapsed, accumulator_sink, gap_cycles,
                source_stride, source_mask);
            break;
        default: std::abort();
    }
}

const char* protocol_name(int protocol) {
    switch (protocol) {
        case kBaseline: return "c00_baseline";
        case kBarrierOnly: return "c01_barrier_only";
        case kWaitOnly: return "c10_wait_only";
        case kWaitBarrier: return "c11_wait_barrier";
        case kProducerConsumer: return "c12_producer_consumer";
        default: return nullptr;
    }
}

bool parse_protocol(const char* value, int* protocol) {
    static constexpr const char* names[] = {
        "c00", "c01", "c10", "c11", "c12",
    };
    for (int index = 0; index < 5; ++index) {
        if (std::strcmp(value, names[index]) == 0 ||
            std::strcmp(value, protocol_name(index)) == 0) {
            *protocol = index;
            return true;
        }
    }
    return false;
}

bool parse_locality(const char* value, Locality* locality) {
    if (std::strcmp(value, "l1") == 0 ||
        std::strcmp(value, "l1_reuse") == 0 ||
        std::strcmp(value, "na") == 0) {
        *locality = {"l1_reuse", 512, 0, true};
        return true;
    }
    if (std::strcmp(value, "l2") == 0 ||
        std::strcmp(value, "l2_rotation") == 0) {
        *locality = {"l2_rotation", 2u * 1024u * 1024u, 512, true};
        return true;
    }
    if (std::strcmp(value, "dram") == 0 ||
        std::strcmp(value, "dram_stream") == 0) {
        *locality = {"dram_stream", 64u * 1024u * 1024u, 4096, false};
        return true;
    }
    return false;
}

bool protocol_uses_async(int protocol, int warp) {
    return protocol == kWaitOnly || protocol == kWaitBarrier ||
           (protocol == kProducerConsumer && warp == 0);
}

unsigned char source_value(std::size_t index) {
    return static_cast<unsigned char>(
        (index * 17u + (index >> 8u) * 29u + 0x5au) & 0xffu);
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr,
                     "usage: %s PROTOCOL LOCALITY GAP_0_OR_64 WARPS [TRIALS]\n",
                     argv[0]);
        return 2;
    }

    int protocol = -1;
    Locality locality{};
    const int gap_cycles = std::atoi(argv[3]);
    const int warps = std::atoi(argv[4]);
    const int trials = argc == 6 ? std::atoi(argv[5]) : 7;
    if (!parse_protocol(argv[1], &protocol) ||
        !parse_locality(argv[2], &locality) ||
        (gap_cycles != 0 && gap_cycles != 64) || warps < 1 ||
        warps > kMaxWarps || trials < 1 || trials > 32) {
        return 2;
    }

    std::vector<unsigned char> host_source(locality.region_bytes);
    for (std::size_t index = 0; index < host_source.size(); ++index) {
        host_source[index] = source_value(index);
    }

    char* source = nullptr;
    char* sink = nullptr;
    unsigned long long* elapsed = nullptr;
    std::uint32_t* accumulator_sink = nullptr;
    std::uint32_t* prewarm_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&source, locality.region_bytes));
    CUDA_CHECK(cudaMalloc(&sink, kMaxWarps * kPayloadBytes));
    CUDA_CHECK(cudaMalloc(&elapsed, kMaxWarps * sizeof(*elapsed)));
    CUDA_CHECK(cudaMalloc(&accumulator_sink,
                          kMaxWarps * sizeof(*accumulator_sink)));
    CUDA_CHECK(cudaMalloc(&prewarm_sink, sizeof(*prewarm_sink)));
    CUDA_CHECK(cudaMemcpy(source, host_source.data(), locality.region_bytes,
                          cudaMemcpyHostToDevice));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaMemset(sink, 0, kMaxWarps * kPayloadBytes));
        CUDA_CHECK(cudaMemset(prewarm_sink, 0, sizeof(*prewarm_sink)));
        if (locality.prewarm) {
            prewarm_source<<<std::min<std::size_t>(256,
                                                  locality.region_bytes / 256),
                             256>>>(
                reinterpret_cast<const std::uint32_t*>(source),
                locality.region_bytes / sizeof(std::uint32_t), prewarm_sink);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaProfilerStart());
        launch_protocol(protocol, warps, source, sink, elapsed,
                        accumulator_sink, gap_cycles,
                        locality.source_stride, locality.region_bytes - 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(samples[trial].data(), elapsed,
                              warps * sizeof(*elapsed),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<unsigned char> host_sink(kMaxWarps * kPayloadBytes);
    std::vector<std::uint32_t> host_accumulator(kMaxWarps);
    CUDA_CHECK(cudaMemcpy(host_sink.data(), sink, host_sink.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_accumulator.data(), accumulator_sink,
                          host_accumulator.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));

    const std::size_t final_source_offset =
        (static_cast<std::size_t>(kGroups - 1) * locality.source_stride) &
        (locality.region_bytes - 1);
    bool correct = true;
    for (int warp = 0; warp < warps; ++warp) {
        if (!protocol_uses_async(protocol, warp)) continue;
        for (int byte = 0; byte < kPayloadBytes; ++byte) {
            if (host_sink[warp * kPayloadBytes + byte] !=
                host_source[final_source_offset + byte]) {
                correct = false;
            }
        }
    }

    std::uint64_t checksum = 0;
    std::uint64_t accumulator_checksum = 0;
    for (int index = 0; index < warps * kPayloadBytes; ++index) {
        checksum = checksum * 131u + host_sink[index];
    }
    for (int warp = 0; warp < warps; ++warp) {
        accumulator_checksum = accumulator_checksum * 131u +
                               host_accumulator[warp];
    }

    std::printf(
        "{\"protocol\":\"%s\",\"locality\":\"%s\",\"gap_cycles\":%d,"
        "\"warps\":%d,\"groups\":%d,\"copies_per_group\":%d,"
        "\"payload_bytes_per_copy\":%d,\"region_bytes\":%zu,"
        "\"source_stride\":%zu,\"samples_cycles\":[",
        protocol_name(protocol), locality.name, gap_cycles, warps, kGroups,
        kCopiesPerGroup, kPayloadBytes, locality.region_bytes,
        locality.source_stride);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf(
        "],\"checksum\":%llu,\"accumulator_checksum\":%llu,"
        "\"correct\":%s}\n",
        static_cast<unsigned long long>(checksum),
        static_cast<unsigned long long>(accumulator_checksum),
        correct ? "true" : "false");

    CUDA_CHECK(cudaFree(prewarm_sink));
    CUDA_CHECK(cudaFree(accumulator_sink));
    CUDA_CHECK(cudaFree(elapsed));
    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(source));
    return correct ? 0 : 1;
}
