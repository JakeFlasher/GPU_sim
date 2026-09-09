#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// The read-only pointer and __ldg intrinsic are intentional: this probe is
// used only to inspect the emitted SM86 constant/read-only load path.  NCU
// targets from the scored workload campaign are never consumed here.
__global__ void constant_load_path(const float* __restrict__ input,
                                   std::uint32_t* output, int iterations,
                                   int mode) {
    const int lane = static_cast<int>(threadIdx.x);
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + lane;
    float value = __ldg(input + index);
    unsigned long long start = clock64();
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        if (mode == 0) {
            // A dependent chain: every load address and value depends on the
            // preceding result, exposing RAW/return latency.
            const int next = (static_cast<int>(__float_as_int(value)) & 1023);
            value = __ldg(input + next) + value;
        } else {
            // Four independent read-only loads per iteration, exposing
            // constant-path issue/queue throughput while keeping work fixed.
            value += __ldg(input + ((index + i) & 1023));
            value += __ldg(input + ((index + i + 256) & 1023));
            value += __ldg(input + ((index + i + 512) & 1023));
            value += __ldg(input + ((index + i + 768) & 1023));
        }
    }
    const unsigned long long elapsed = clock64() - start;
    output[index] = static_cast<std::uint32_t>(elapsed) ^
                    static_cast<std::uint32_t>(__float_as_int(value));
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s MODE BLOCKS THREADS ITERATIONS WARMUPS\n",
                     argv[0]);
        return 2;
    }
    const int mode = std::atoi(argv[1]);
    const int blocks = std::atoi(argv[2]);
    const int threads = std::atoi(argv[3]);
    const int iterations = std::atoi(argv[4]);
    const int warmups = std::atoi(argv[5]);
    if (mode < 0 || mode > 1 || blocks < 1 || threads < 32 ||
        threads > 256 || threads % 32 != 0 || iterations < 1 ||
        iterations > (1 << 20) || warmups < 1 || warmups > 16) {
        return 2;
    }
    const std::size_t elements = static_cast<std::size_t>(blocks) * threads;
    std::vector<float> host(2048, 1.0f);
    float* input = nullptr;
    std::uint32_t* output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, host.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(input, host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    for (int i = 0; i < warmups; ++i) {
        constant_load_path<<<blocks, threads>>>(input, output, iterations, mode);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    constant_load_path<<<blocks, threads>>>(input, output, iterations, mode);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> observed(elements);
    CUDA_CHECK(cudaMemcpy(observed.data(), output,
                          observed.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    std::uint32_t checksum = 0;
    for (auto value : observed) checksum = checksum * 131u + value;
    std::printf("{\"mode\":%d,\"blocks\":%d,\"threads\":%d,"
                "\"iterations\":%d,\"warmups\":%d,\"checksum\":%u}\n",
                mode, blocks, threads, iterations, warmups, checksum);
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(input));
    return 0;
}
