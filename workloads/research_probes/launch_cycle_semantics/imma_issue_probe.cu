#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__device__ __forceinline__ void imma_s8(
    int& d0, int& d1, int& d2, int& d3,
    std::uint32_t a0, std::uint32_t a1, std::uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

constexpr int kIterations = 256;

template <int Streams>
__global__ __launch_bounds__(32)
void imma_issue_kernel(std::uint64_t* result) {
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t a0 = 0x01010101u ^ lane;
    const std::uint32_t a1 = 0x02020202u ^ lane;
    const std::uint32_t b0 = 0x01010101u + lane;
    int d00 = 1, d01 = 2, d02 = 3, d03 = 4;
    int d10 = 5, d11 = 6, d12 = 7, d13 = 8;
    int d20 = 9, d21 = 10, d22 = 11, d23 = 12;
    int d30 = 13, d31 = 14, d32 = 15, d33 = 16;

    __syncthreads();
    const std::uint64_t start = clock64();
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        imma_s8(d00, d01, d02, d03, a0, a1, b0);
        if constexpr (Streams >= 2) {
            imma_s8(d10, d11, d12, d13, a0, a1, b0);
        }
        if constexpr (Streams >= 3) {
            imma_s8(d20, d21, d22, d23, a0, a1, b0);
        }
        if constexpr (Streams >= 4) {
            imma_s8(d30, d31, d32, d33, a0, a1, b0);
        }
    }
    const std::uint64_t stop = clock64();
    if (lane == 0) {
        result[0] = stop - start;
        result[1] =
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d00)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d01)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d02)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d03)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d10)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d11)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d12)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d13)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d20)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d21)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d22)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d23)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d30)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d31)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d32)) +
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(d33));
    }
}

template <int Streams>
void run(std::uint64_t* device_result, int trials) {
    for (int trial = 0; trial < trials; ++trial) {
        imma_issue_kernel<Streams><<<1, 32>>>(device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::uint64_t result[2]{};
        CUDA_CHECK(cudaMemcpy(result, device_result, sizeof(result),
                              cudaMemcpyDeviceToHost));
        const double cycles_per_imma =
            static_cast<double>(result[0]) /
            static_cast<double>(kIterations * Streams);
        std::printf(
            "{\"streams\":%d,\"trial\":%d,\"iterations\":%d,"
            "\"imma_instructions\":%d,\"elapsed_cycles\":%llu,"
            "\"cycles_per_imma\":%.9f,\"checksum\":%llu}\n",
            Streams, trial + 1, kIterations, kIterations * Streams,
            static_cast<unsigned long long>(result[0]), cycles_per_imma,
            static_cast<unsigned long long>(result[1]));
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s TRIALS\n", argv[0]);
        return 2;
    }
    const int trials = std::atoi(argv[1]);
    if (trials < 1 || trials > 64) {
        return 2;
    }

    std::uint64_t* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&device_result, 2 * sizeof(*device_result)));
    run<1>(device_result, trials);
    run<2>(device_result, trials);
    run<4>(device_result, trials);
    CUDA_CHECK(cudaFree(device_result));
    return 0;
}
