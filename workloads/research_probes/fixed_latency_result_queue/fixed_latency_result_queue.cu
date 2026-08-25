#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

#ifndef FIXED_RESULT_GENERATIONS
constexpr int kGenerations = 8192;
#else
constexpr int kGenerations = FIXED_RESULT_GENERATIONS;
#endif

__global__ void result_queue_pilot(unsigned long long* elapsed,
                                   std::uint32_t* sink,
                                   const std::uint32_t* half_inputs,
                                   const float* float_inputs) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int thread = warp * kWarpSize + lane;
    std::uint32_t half_a0 = half_inputs[thread * 8];
    std::uint32_t half_a1 = half_inputs[thread * 8 + 2];
    std::uint32_t half_a2 = half_inputs[thread * 8 + 4];
    std::uint32_t half_a3 = half_inputs[thread * 8 + 6];
    std::uint32_t one_a0 = __float_as_uint(float_inputs[thread * 8]);
    std::uint32_t one_b0 = __float_as_uint(float_inputs[thread * 8 + 1]);
    std::uint32_t one_a1 = __float_as_uint(float_inputs[thread * 8 + 2]);
    std::uint32_t one_b1 = __float_as_uint(float_inputs[thread * 8 + 3]);
    std::uint32_t one_a2 = __float_as_uint(float_inputs[thread * 8 + 4]);
    std::uint32_t one_b2 = __float_as_uint(float_inputs[thread * 8 + 5]);
    std::uint32_t one_a3 = __float_as_uint(float_inputs[thread * 8 + 6]);
    std::uint32_t one_b3 = __float_as_uint(float_inputs[thread * 8 + 7]);
    std::uint32_t h0 = 0;
    std::uint32_t h1 = 0;
    std::uint32_t h2 = 0;
    std::uint32_t h3 = 0;
    std::uint32_t f0 = one_a0;
    std::uint32_t f1 = one_a1;
    std::uint32_t f2 = one_a2;
    std::uint32_t f3 = one_a3;
    std::uint32_t sentinel =
        0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131 + lane);
    asm volatile("" : "+r"(half_a0), "+r"(half_a1), "+r"(half_a2),
                 "+r"(half_a3), "+r"(one_a0), "+r"(one_b0),
                 "+r"(one_a1), "+r"(one_b1), "+r"(one_a2),
                 "+r"(one_b2), "+r"(one_a3), "+r"(one_b3));

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kGenerations; ++generation) {
        asm volatile("add.rn.f16x2 %0, %1, %2;"
                     : "=r"(h0)
                     : "r"(sentinel), "r"(half_a0)
                     : "memory");
        asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                     : "=r"(f0)
                     : "r"(sentinel), "r"(one_a0), "r"(half_a0)
                     : "memory");
        asm volatile("add.rn.f16x2 %0, %1, %2;"
                     : "=r"(h1)
                     : "r"(sentinel), "r"(half_a1)
                     : "memory");
        asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                     : "=r"(f1)
                     : "r"(sentinel), "r"(one_a1), "r"(half_a1)
                     : "memory");
        asm volatile("add.rn.f16x2 %0, %1, %2;"
                     : "=r"(h2)
                     : "r"(sentinel), "r"(half_a2)
                     : "memory");
        asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                     : "=r"(f2)
                     : "r"(sentinel), "r"(one_a2), "r"(half_a2)
                     : "memory");
        asm volatile("add.rn.f16x2 %0, %1, %2;"
                     : "=r"(h3)
                     : "r"(sentinel), "r"(half_a3)
                     : "memory");
        asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                     : "=r"(f3)
                     : "r"(sentinel), "r"(one_a3), "r"(half_a3)
                     : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(sentinel)
                     : "r"(h0), "r"(f0)
                     : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(sentinel)
                     : "r"(h1), "r"(f1)
                     : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(sentinel)
                     : "r"(h2), "r"(f2)
                     : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(sentinel)
                     : "r"(h3), "r"(f3)
                     : "memory");
        asm volatile("add.u32 %0, %0, 1;" : "+r"(sentinel) : : "memory");
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = sentinel;
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s WARPS_4_8_12_16 [TRIALS]\n", argv[0]);
        return 2;
    }
    const int warps = std::atoi(argv[1]);
    const int trials = argc == 3 ? std::atoi(argv[2]) : 7;
    if ((warps != 4 && warps != 8 && warps != 12 && warps != 16) ||
        trials < 1 || trials > 32) {
        return 2;
    }

    unsigned long long* elapsed = nullptr;
    std::uint32_t* sink = nullptr;
    std::uint32_t* half_inputs = nullptr;
    float* float_inputs = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed, kMaxWarps * sizeof(*elapsed)));
    CUDA_CHECK(cudaMalloc(&sink, kMaxWarps * sizeof(*sink)));
    CUDA_CHECK(cudaMalloc(&half_inputs, kMaxWarps * kWarpSize * 8 * sizeof(*half_inputs)));
    CUDA_CHECK(cudaMalloc(&float_inputs, kMaxWarps * kWarpSize * 8 * sizeof(*float_inputs)));
    CUDA_CHECK(cudaMemset(half_inputs, 0, kMaxWarps * kWarpSize * 8 * sizeof(*half_inputs)));
    std::vector<float> host_float(kMaxWarps * kWarpSize * 8, 1.0f);
    CUDA_CHECK(cudaMemcpy(float_inputs, host_float.data(),
                          host_float.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(warps));
    for (int trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cudaProfilerStart());
        result_queue_pilot<<<1, warps * kWarpSize>>>(elapsed, sink, half_inputs,
                                                     float_inputs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(samples[trial].data(), elapsed,
                              warps * sizeof(*elapsed), cudaMemcpyDeviceToHost));
    }
    std::vector<std::uint32_t> values(warps);
    CUDA_CHECK(cudaMemcpy(values.data(), sink, warps * sizeof(*sink),
                          cudaMemcpyDeviceToHost));
    bool correct = true;
    std::uint64_t checksum = 0;
    for (int warp = 0; warp < warps; ++warp) {
        const std::uint32_t sentinel =
            (0x6a09e667u ^ static_cast<std::uint32_t>(warp * 131)) +
            static_cast<std::uint32_t>(kGenerations);
        const std::uint32_t expected = sentinel;
        if (values[warp] != expected && warp == 0) {
            std::fprintf(stderr, "actual=0x%08x expected=0x%08x\n",
                         values[warp], expected);
        }
        correct = correct && values[warp] == expected;
        checksum = checksum * 131u + values[warp];
    }
    std::printf(
        "{\"warps\":%d,\"generations\":%d,\"samples_cycles\":[",
        warps, kGenerations);
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s[", trial == 0 ? "" : ",");
        for (int warp = 0; warp < warps; ++warp) {
            std::printf("%s%llu", warp == 0 ? "" : ",",
                        samples[trial][warp]);
        }
        std::printf("]");
    }
    std::printf("],\"checksum\":%llu,\"correct\":%s}\n",
                static_cast<unsigned long long>(checksum),
                correct ? "true" : "false");
    CUDA_CHECK(cudaFree(float_inputs));
    CUDA_CHECK(cudaFree(half_inputs));
    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(elapsed));
    return correct ? 0 : 1;
}
