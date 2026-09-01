#include <cuda_runtime.h>

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
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__device__ __forceinline__ std::uint32_t dependent_step(std::uint32_t value) {
    std::uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;\n"
                 : "=r"(next)
                 : "r"(value));
    return next;
}

__global__ __launch_bounds__(256)
void launch_envelope_control(int iterations) {
    __shared__ std::uint32_t work_sink[256];
    const std::size_t logical =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(logical * 0x85ebca6bu);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = dependent_step(value);
    }
    const unsigned work_address = static_cast<unsigned>(
        __cvta_generic_to_shared(work_sink + threadIdx.x));
    asm volatile("st.volatile.shared.u32 [%0], %1;\n"
                 :
                 : "r"(work_address), "r"(value)
                 : "memory");
}

// This launch is intentionally after the selected NCU target.  It verifies
// the same arithmetic without adding a global-memory tail to the envelope
// control itself.
__global__ __launch_bounds__(256)
void launch_envelope_verify(std::uint32_t* output, int iterations) {
    const std::size_t logical =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(logical * 0x85ebca6bu);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = dependent_step(value);
    }
    output[logical] = value;
}

std::uint32_t expected_value(std::size_t logical, int iterations) {
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(logical * 0x85ebca6bu);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = value * 1664525u + 1013904223u;
    }
    return value;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s MODE BLOCKS THREADS ITERATIONS WARMUPS\n",
                     argv[0]);
        return 2;
    }
    const bool control = std::strcmp(argv[1], "control") == 0;
    const bool store = std::strcmp(argv[1], "store") == 0;
    const int blocks = std::atoi(argv[2]);
    const int threads = std::atoi(argv[3]);
    const int iterations = std::atoi(argv[4]);
    const int warmups = std::atoi(argv[5]);
    if ((!control && !store) || blocks < 1 || blocks > (1 << 20) ||
        threads < 32 || threads > 256 ||
        threads % 32 != 0 || iterations < 0 || iterations > (1 << 20) ||
        warmups < 1 || warmups > 16) {
        return 2;
    }

    const std::size_t words =
        static_cast<std::size_t>(blocks) * static_cast<std::size_t>(threads);
    std::uint32_t* output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, words * sizeof(*output)));

    for (int warmup = 0; warmup < warmups; ++warmup) {
        if (control) {
            launch_envelope_control<<<blocks, threads>>>(iterations);
        } else {
            launch_envelope_verify<<<blocks, threads>>>(output, iterations);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (control) {
        launch_envelope_control<<<blocks, threads>>>(iterations);
    } else {
        launch_envelope_verify<<<blocks, threads>>>(output, iterations);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_envelope_verify<<<blocks, threads>>>(output, iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint32_t> observed(words);
    CUDA_CHECK(cudaMemcpy(observed.data(), output, words * sizeof(*output),
                          cudaMemcpyDeviceToHost));
    std::uint64_t checksum = 0;
    bool correct = true;
    for (std::size_t logical = 0; logical < words; ++logical) {
        correct = correct && observed[logical] == expected_value(logical, iterations);
        checksum = checksum * 131u + observed[logical];
    }

    std::printf(
        "{\"mode\":\"%s\",\"blocks\":%d,\"threads\":%d,\"warps_per_cta\":%d,"
        "\"iterations\":%d,\"warmups\":%d,\"scored_launches\":1,"
        "\"checksum\":%llu,\"correct\":%s}\n",
        control ? "control" : "store", blocks, threads, threads / 32,
        iterations, warmups,
        static_cast<unsigned long long>(checksum), correct ? "true" : "false");

    CUDA_CHECK(cudaFree(output));
    return correct ? 0 : 1;
}
