#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
                                   std::uint32_t* output,
                                   unsigned long long* elapsed_output,
                                   int iterations, int mode) {
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
    // Keep the functional result separate from the clock sample.  The host
    // can therefore validate the computation without treating a timer value
    // as a correctness oracle.
    output[index] = static_cast<std::uint32_t>(__float_as_int(value));
    elapsed_output[index] = elapsed;
}

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

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s MODE BLOCKS THREADS ITERATIONS WARMUPS\n",
                     argv[0]);
        return 2;
    }
    int mode = 0, blocks = 0, threads = 0, iterations = 0, warmups = 0;
    if (!parse_int(argv[1], &mode) || !parse_int(argv[2], &blocks) ||
        !parse_int(argv[3], &threads) || !parse_int(argv[4], &iterations) ||
        !parse_int(argv[5], &warmups) || mode < 0 || mode > 1 || blocks < 1 ||
        threads < 32 || threads > 256 || threads % 32 != 0 || iterations < 1 ||
        iterations > (1 << 20) || warmups < 0 || warmups > 16) {
        return 2;
    }
    const std::size_t elements = static_cast<std::size_t>(blocks) * threads;
    const std::size_t input_elements = std::max<std::size_t>(2048, elements);
    std::vector<float> host(input_elements, 1.0f);
    float* input = nullptr;
    std::uint32_t* output = nullptr;
    unsigned long long* elapsed_output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, host.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&elapsed_output,
                          elements * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(input, host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    for (int i = 0; i < warmups; ++i) {
        constant_load_path<<<blocks, threads>>>(input, output, elapsed_output,
                                                iterations, mode);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(output, 0xcd, elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(elapsed_output, 0xff,
                          elements * sizeof(unsigned long long)));
    constant_load_path<<<blocks, threads>>>(input, output, elapsed_output,
                                            iterations, mode);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> observed(elements);
    std::vector<unsigned long long> elapsed(elements);
    CUDA_CHECK(cudaMemcpy(observed.data(), output,
                          observed.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(elapsed.data(), elapsed_output,
                          elapsed.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    const float expected_value =
        1.0f + static_cast<float>(mode == 0 ? iterations : 4LL * iterations);
    std::uint32_t expected_bits = 0;
    static_assert(sizeof(expected_bits) == sizeof(expected_value));
    std::memcpy(&expected_bits, &expected_value, sizeof(expected_bits));
    bool correct = true;
    bool elapsed_valid = true;
    std::uint64_t checksum = 0;
    unsigned long long min_cycles = std::numeric_limits<unsigned long long>::max();
    unsigned long long max_cycles = 0;
    for (std::size_t index = 0; index < elements; ++index) {
        correct = correct && observed[index] == expected_bits;
        elapsed_valid = elapsed_valid &&
                        elapsed[index] != std::numeric_limits<unsigned long long>::max() &&
                        elapsed[index] > 0;
        checksum = checksum * 131u + observed[index];
        min_cycles = std::min(min_cycles, elapsed[index]);
        max_cycles = std::max(max_cycles, elapsed[index]);
    }
    correct = correct && elapsed_valid;
    std::printf("{\"mode\":%d,\"blocks\":%d,\"threads\":%d,"
                "\"iterations\":%d,\"warmups\":%d,\"checksum\":%llu,"
                "\"expected_bits\":%u,\"correct\":%s,"
                "\"elapsed_samples_valid\":%s,\"min_cycles\":%llu,"
                "\"max_cycles\":%llu}\n",
                mode, blocks, threads, iterations, warmups,
                static_cast<unsigned long long>(checksum), expected_bits,
                correct ? "true" : "false", elapsed_valid ? "true" : "false",
                min_cycles, max_cycles);
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(elapsed_output));
    CUDA_CHECK(cudaFree(input));
    return correct ? 0 : 1;
}
