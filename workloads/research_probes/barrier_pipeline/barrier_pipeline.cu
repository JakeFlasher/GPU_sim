#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
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
constexpr int kMaxWarps = 16;

#ifndef BARRIER_PIPELINE_GENERATIONS
constexpr int kGenerations = 16384;
#else
constexpr int kGenerations = BARRIER_PIPELINE_GENERATIONS;
#endif

enum Protocol : int {
    kControl = 0,
    kOneBarrier = 1,
    kTwoBarriers = 2,
    kSkewBarrier = 3,
    kUniformSkewCalibration = 4,
};

__device__ __forceinline__ std::uint32_t chain_32(std::uint32_t value) {
    asm volatile(
        "mul.lo.u32 %0, %0, 1664525;\n"
        "xor.b32 %0, %0, 1013904223;\n"
        "mul.lo.u32 %0, %0, 1664525;\n"
        "xor.b32 %0, %0, 1013904223;\n"
        "mul.lo.u32 %0, %0, 1664525;\n"
        : "+r"(value));
    return value;
}

__device__ __forceinline__ std::uint32_t chain_64(std::uint32_t value) {
    asm volatile(
        "mul.lo.u32 %0, %0, 1664525;\n"
        "xor.b32 %0, %0, 1013904223;\n"
        "mul.lo.u32 %0, %0, 1664525;\n"
        "xor.b32 %0, %0, 1013904223;\n"
        "mul.lo.u32 %0, %0, 1664525;\n"
        "xor.b32 %0, %0, 1013904223;\n"
        "mul.lo.u32 %0, %0, 1664525;\n"
        : "+r"(value));
    return value;
}

__device__ __forceinline__ std::uint32_t apply_skew(std::uint32_t value,
                                                     int skew) {
    if (skew == 32) {
        return chain_32(value);
    }
    if (skew == 64) {
        return chain_64(value);
    }
    return value;
}

template <int ProtocolValue>
__global__ void barrier_pipeline(unsigned long long* elapsed,
                                 std::uint32_t* accumulator_sink, int skew) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    std::uint32_t accumulator =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);

    __syncthreads();
    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        accumulator = accumulator * 1664525u + 1013904223u;
        accumulator ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;

        if constexpr (ProtocolValue == kSkewBarrier) {
            if (warp == 0) {
                accumulator = apply_skew(accumulator, skew);
            }
        } else if constexpr (ProtocolValue == kUniformSkewCalibration) {
            accumulator = apply_skew(accumulator, skew);
        }

        if constexpr (ProtocolValue == kOneBarrier ||
                      ProtocolValue == kSkewBarrier) {
            __syncthreads();
        } else if constexpr (ProtocolValue == kTwoBarriers) {
            asm volatile("bar.sync 0;\nbar.sync 0;\n" ::: "memory");
        }

        accumulator ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        accumulator_sink[warp] = accumulator;
    }
}

template <int ProtocolValue>
void launch_kernel(int warps, unsigned long long* elapsed,
                   std::uint32_t* accumulator_sink, int skew) {
    barrier_pipeline<ProtocolValue><<<1, warps * kWarpSize>>>(
        elapsed, accumulator_sink, skew);
}

void launch_protocol(int protocol, int warps, unsigned long long* elapsed,
                     std::uint32_t* accumulator_sink, int skew) {
    switch (protocol) {
        case kControl:
            launch_kernel<kControl>(warps, elapsed, accumulator_sink, skew);
            break;
        case kOneBarrier:
            launch_kernel<kOneBarrier>(warps, elapsed, accumulator_sink, skew);
            break;
        case kTwoBarriers:
            launch_kernel<kTwoBarriers>(warps, elapsed, accumulator_sink, skew);
            break;
        case kSkewBarrier:
            launch_kernel<kSkewBarrier>(warps, elapsed, accumulator_sink, skew);
            break;
        case kUniformSkewCalibration:
            launch_kernel<kUniformSkewCalibration>(warps, elapsed,
                                                   accumulator_sink, skew);
            break;
        default:
            std::abort();
    }
}

const char* protocol_name(int protocol) {
    switch (protocol) {
        case kControl: return "b00_control";
        case kOneBarrier: return "b10_one_barrier";
        case kTwoBarriers: return "b20_two_barriers";
        case kSkewBarrier: return "b11_skew_barrier";
        case kUniformSkewCalibration: return "cal_uniform_skew";
        default: return nullptr;
    }
}

bool parse_protocol(const char* value, int* protocol) {
    static constexpr const char* names[] = {
        "b00", "b10", "b20", "b11", "cal",
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

std::uint32_t host_chain(std::uint32_t value, int instructions) {
    for (int instruction = 0; instruction < instructions; ++instruction) {
        if ((instruction & 1) == 0) {
            value *= 1664525u;
        } else {
            value ^= 1013904223u;
        }
    }
    return value;
}

std::uint32_t expected_accumulator(int protocol, int warp, int lane, int skew) {
    std::uint32_t value =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    for (int generation = 0; generation < kGenerations; ++generation) {
        value = value * 1664525u + 1013904223u;
        value ^= static_cast<std::uint32_t>(generation) + 0x9e3779b9u;
        const bool apply = protocol == kUniformSkewCalibration ||
                           (protocol == kSkewBarrier && warp == 0);
        if (apply) {
            value = host_chain(value, skew == 32 ? 5 : skew == 64 ? 7 : 0);
        }
        value ^= 0xbb67ae85u + static_cast<std::uint32_t>(generation);
    }
    return value;
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s PROTOCOL WARPS SKEW_0_32_64 [TRIALS]\n",
                     argv[0]);
        return 2;
    }

    int protocol = -1;
    const int warps = std::atoi(argv[2]);
    const int skew = std::atoi(argv[3]);
    const int trials = argc == 5 ? std::atoi(argv[4]) : 7;
    if (!parse_protocol(argv[1], &protocol) || warps < 1 ||
        warps > kMaxWarps || (skew != 0 && skew != 32 && skew != 64) ||
        trials < 1 || trials > 32) {
        return 2;
    }
    if ((protocol == kControl || protocol == kOneBarrier ||
         protocol == kTwoBarriers) &&
        skew != 0) {
        return 2;
    }
    if ((protocol == kSkewBarrier || protocol == kUniformSkewCalibration) &&
        skew == 0) {
        return 2;
    }

    unsigned long long* elapsed = nullptr;
    std::uint32_t* accumulator_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed, kMaxWarps * sizeof(*elapsed)));
    CUDA_CHECK(cudaMalloc(&accumulator_sink,
                          kMaxWarps * sizeof(*accumulator_sink)));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaProfilerStart());
        launch_protocol(protocol, warps, elapsed, accumulator_sink, skew);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(samples[trial].data(), elapsed,
                              warps * sizeof(*elapsed),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<std::uint32_t> host_accumulator(warps);
    CUDA_CHECK(cudaMemcpy(host_accumulator.data(), accumulator_sink,
                          host_accumulator.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        const std::uint32_t expected = expected_accumulator(protocol, warp, 0, skew);
        correct = correct && host_accumulator[warp] == expected;
        checksum = checksum * 131u + host_accumulator[warp];
    }

    std::printf(
        "{\"protocol\":\"%s\",\"warps\":%d,\"skew\":%d,"
        "\"generations\":%d,\"samples_cycles\":[",
        protocol_name(protocol), warps, skew, kGenerations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf(
        "],\"accumulator_checksum\":%llu,\"correct\":%s}\n",
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(accumulator_sink));
    CUDA_CHECK(cudaFree(elapsed));
    return correct ? 0 : 1;
}
