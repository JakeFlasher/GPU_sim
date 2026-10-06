#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(status));                           \
            return 1;                                                           \
        }                                                                       \
    } while (0)

struct Result {
    unsigned long long elapsed;
    std::uint32_t kind;
    std::uint32_t iterations;
    float accumulators[32];
};

__device__ __forceinline__ void mma_tf32(float &d0, float &d1, float &d2,
                                         float &d3, std::uint32_t a0,
                                         std::uint32_t a1, std::uint32_t a2,
                                         std::uint32_t a3, std::uint32_t b0,
                                         std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool Independent>
__global__ void tf32_issue_kernel(Result *result, std::uint32_t iterations) {
    // The operands are uniform constants.  No global load, barrier, or
    // reduction is inside the timed loop; only the tensor-pipeline relation
    // differs between the dependent and independent arms.
    const std::uint32_t a0 = 0x3f800000u;
    const std::uint32_t a1 = 0x3f800000u;
    const std::uint32_t a2 = 0x3f800000u;
    const std::uint32_t a3 = 0x3f800000u;
    const std::uint32_t b0 = 0x3f800000u;
    const std::uint32_t b1 = 0x3f800000u;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
    float i0 = 0.0f, i1 = 0.0f, i2 = 0.0f, i3 = 0.0f;
    float j0 = 0.0f, j1 = 0.0f, j2 = 0.0f, j3 = 0.0f;
    float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (std::uint32_t iter = 0; iter < iterations; ++iter) {
        mma_tf32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        if constexpr (Independent) {
            mma_tf32(e0, e1, e2, e3, a0, a1, a2, a3, b0, b1);
            mma_tf32(f0, f1, f2, f3, a0, a1, a2, a3, b0, b1);
            mma_tf32(g0, g1, g2, g3, a0, a1, a2, a3, b0, b1);
            mma_tf32(h0, h1, h2, h3, a0, a1, a2, a3, b0, b1);
            mma_tf32(i0, i1, i2, i3, a0, a1, a2, a3, b0, b1);
            mma_tf32(j0, j1, j2, j3, a0, a1, a2, a3, b0, b1);
            mma_tf32(k0, k1, k2, k3, a0, a1, a2, a3, b0, b1);
        }
    }
    const unsigned long long stop = clock64();
    const float values[32] = {d0, d1, d2, d3, e0, e1, e2, e3,
                             f0, f1, f2, f3, g0, g1, g2, g3,
                             h0, h1, h2, h3, i0, i1, i2, i3,
                             j0, j1, j2, j3, k0, k1, k2, k3};
    // Observe every lane/accumulator after the timed loop. The host checks
    // each one independently; a cancelling checksum cannot hide an error.
    Result &lane_result = result[threadIdx.x];
    lane_result.elapsed = stop - start;
    lane_result.kind = Independent ? 1u : 0u;
    lane_result.iterations = iterations;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        lane_result.accumulators[i] = values[i];
    }
}

int parse_u32(const char *text) {
    if (text[0] < '0' || text[0] > '9') return -1;
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || end[0] || value == 0 || value > 1000000ull) return -1;
    return static_cast<int>(value);
}

int main(int argc, char **argv) {
    if (argc != 3 || (std::strcmp(argv[1], "dependent") != 0 &&
                      std::strcmp(argv[1], "independent") != 0)) {
        std::fprintf(stderr, "usage: %s dependent|independent iterations\n", argv[0]);
        return 2;
    }
    const int iterations = parse_u32(argv[2]);
    if (iterations < 1) return 2;
    Result *device_result = nullptr;
    Result host_result[32]{};
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(host_result)));
    CUDA_CHECK(cudaMemset(device_result, 0xff, sizeof(host_result)));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStart());
    const bool independent = std::strcmp(argv[1], "independent") == 0;
    if (independent) {
        tf32_issue_kernel<true><<<1, 32>>>(device_result, iterations);
    } else {
        tf32_issue_kernel<false><<<1, 32>>>(device_result, iterations);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaMemcpy(host_result, device_result, sizeof(host_result),
                          cudaMemcpyDeviceToHost));
    // All A/B fragments contain 1.0. Every m16n8k8 operation adds exactly
    // eight to each of four accumulators per lane. These values are exactly
    // representable throughout the bounded iteration domain.
    const float expected = 8.0f * iterations;
    unsigned mismatches = 0;
    bool elapsed_valid = true;
    bool metadata_valid = true;
    for (const auto &lane : host_result) {
        elapsed_valid &= lane.elapsed > 0 && lane.elapsed != ~0ull;
        metadata_valid &= lane.kind == (independent ? 1u : 0u) &&
                          lane.iterations == static_cast<std::uint32_t>(iterations);
        for (int i = 0; i < 32; ++i) {
            mismatches += lane.accumulators[i] != (independent || i < 4 ? expected : 0.0f);
        }
    }
    const bool correct = elapsed_valid && metadata_valid && mismatches == 0;
    std::printf("{\"correct\":%s,\"kind\":\"%s\",\"iterations\":%d,"
                "\"elapsed\":%llu,\"elapsed_valid\":%s,\"metadata_valid\":%s,"
                "\"accumulators_checked\":1024,\"mismatches\":%u}\n",
                correct ? "true" : "false", argv[1], iterations,
                host_result[0].elapsed, elapsed_valid ? "true" : "false",
                metadata_valid ? "true" : "false", mismatches);
    CUDA_CHECK(cudaFree(device_result));
    return correct ? 0 : 1;
}
