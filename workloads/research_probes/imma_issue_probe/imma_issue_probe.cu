#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) do { const cudaError_t s = (call); if (s != cudaSuccess) { \
  std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(s)); return 1; \
} } while (0)

struct Result {
    unsigned long long elapsed;
    std::uint32_t kind;
    std::uint32_t iterations;
    std::int32_t accumulators[32];
};

__device__ __forceinline__ void mma_int8(std::int32_t &d0, std::int32_t &d1,
                                         std::int32_t &d2, std::int32_t &d3,
                                         std::uint32_t a0, std::uint32_t a1,
                                         std::uint32_t a2, std::uint32_t a3,
                                         std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool Independent>
__global__ void issue_kernel(Result *result, std::uint32_t iterations) {
    // Four packed signed-INT8 1s per A register and two packed B registers.
    const std::uint32_t one = 0x01010101u;
    std::int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    std::int32_t e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    std::int32_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    std::int32_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    std::int32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
    std::int32_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    std::int32_t j0 = 0, j1 = 0, j2 = 0, j3 = 0;
    std::int32_t k0 = 0, k1 = 0, k2 = 0, k3 = 0;
    const unsigned long long start = clock64();
#pragma unroll 1
    for (std::uint32_t iter = 0; iter < iterations; ++iter) {
        mma_int8(d0, d1, d2, d3, one, one, one, one, one, one);
        if constexpr (Independent) {
            mma_int8(e0, e1, e2, e3, one, one, one, one, one, one);
            mma_int8(f0, f1, f2, f3, one, one, one, one, one, one);
            mma_int8(g0, g1, g2, g3, one, one, one, one, one, one);
            mma_int8(h0, h1, h2, h3, one, one, one, one, one, one);
            mma_int8(i0, i1, i2, i3, one, one, one, one, one, one);
            mma_int8(j0, j1, j2, j3, one, one, one, one, one, one);
            mma_int8(k0, k1, k2, k3, one, one, one, one, one, one);
        }
    }
    const unsigned long long stop = clock64();
    const std::int32_t values[32] = {d0,d1,d2,d3,e0,e1,e2,e3,f0,f1,f2,f3,
        g0,g1,g2,g3,h0,h1,h2,h3,i0,i1,i2,i3,j0,j1,j2,j3,k0,k1,k2,k3};
    Result &out = result[threadIdx.x];
    out.elapsed = stop - start;
    out.kind = Independent ? 1u : 0u;
    out.iterations = iterations;
#pragma unroll
    for (int i = 0; i < 32; ++i) out.accumulators[i] = values[i];
}

int parse_u32(const char *text) {
    if (text[0] < '0' || text[0] > '9') return -1;
    char *end = nullptr; errno = 0;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || end[0] || value == 0 || value > 1000000ull) return -1;
    return static_cast<int>(value);
}

int main(int argc, char **argv) {
    if (argc != 3 || (std::strcmp(argv[1], "dependent") != 0 &&
                      std::strcmp(argv[1], "independent") != 0)) return 2;
    int iterations = parse_u32(argv[2]); if (iterations < 1) return 2;
    Result *device = nullptr; Result host[32]{};
    CUDA_CHECK(cudaMalloc(&device, sizeof(host)));
    CUDA_CHECK(cudaMemset(device, 0xff, sizeof(host)));
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaProfilerStart());
    if (std::strcmp(argv[1], "independent") == 0)
        issue_kernel<true><<<1,32>>>(device, iterations);
    else issue_kernel<false><<<1,32>>>(device, iterations);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost));
    const bool independent = std::strcmp(argv[1], "independent") == 0;
    unsigned mismatches = 0; bool metadata = true, elapsed = true;
    const std::int32_t expected = 32 * iterations;
    for (const auto &lane : host) {
        elapsed &= lane.elapsed > 0 && lane.elapsed != ~0ull;
        metadata &= lane.kind == (independent ? 1u : 0u) &&
                    lane.iterations == static_cast<std::uint32_t>(iterations);
        for (int i = 0; i < 32; ++i)
            mismatches += lane.accumulators[i] != (independent || i < 4 ? expected : 0);
    }
    bool correct = elapsed && metadata && mismatches == 0;
    std::printf("{\"correct\":%s,\"kind\":\"%s\",\"iterations\":%d,"
                "\"elapsed\":%llu,\"elapsed_valid\":%s,\"metadata_valid\":%s,"
                "\"accumulators_checked\":1024,\"mismatches\":%u}\n",
                correct ? "true" : "false", argv[1], iterations, host[0].elapsed,
                elapsed ? "true" : "false", metadata ? "true" : "false", mismatches);
    CUDA_CHECK(cudaFree(device)); return correct ? 0 : 1;
}
