#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// One selected global load and one contiguous global store per active lane.
// The SM86 coalescer analysis consumes the same stride/offset/active shape.
__global__ void sector_probe(const unsigned* input, unsigned* output,
                             unsigned stride, unsigned offset,
                             unsigned active, unsigned n) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31u;
    if (lane < active && tid < n) {
        output[tid] = input[tid * stride + offset];
    }
}

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    unsigned stride = std::strtoul(argv[1], nullptr, 0);
    unsigned offset = std::strtoul(argv[2], nullptr, 0);
    unsigned active = std::strtoul(argv[3], nullptr, 0);
    unsigned blocks = std::strtoul(argv[4], nullptr, 0);
    if (stride == 0 || active == 0 || active > 32 || blocks == 0) return 2;
    constexpr unsigned threads = 256;
    constexpr unsigned n = 1u << 20;
    size_t input_bytes = (static_cast<size_t>(n - 1) * stride + offset + 1) * sizeof(unsigned);
    size_t output_bytes = static_cast<size_t>(n) * sizeof(unsigned);
    unsigned *input = nullptr, *output = nullptr;
    if (cudaMalloc(&input, input_bytes) != cudaSuccess ||
        cudaMalloc(&output, output_bytes) != cudaSuccess) return 3;
    cudaMemset(input, 0, input_bytes);
    sector_probe<<<blocks, threads>>>(input, output, stride, offset, active, n);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    cudaFree(input);
    cudaFree(output);
    return 0;
}
