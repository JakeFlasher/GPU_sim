#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const CUresult status = (call);                                        \
        if (status != CUDA_SUCCESS) {                                          \
            const char *message = nullptr;                                     \
            cuGetErrorString(status, &message);                                \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         message ? message : "unknown CUDA error");            \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main(int argc, char **argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 84;
    unsigned iterations = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 256;
    if (blocks < 1 || blocks > 4096 || iterations == 0) {
        std::fprintf(stderr, "usage: %s BLOCKS ITERATIONS\n", argv[0]);
        return 2;
    }

    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    CUmodule module;
    CUDA_CHECK(cuModuleLoad(&module, "imc_kernel.cubin"));
    CUfunction function;
    CUDA_CHECK(cuModuleGetFunction(&function, module, "imc_kernel"));

    const std::size_t warp_count = static_cast<std::size_t>(blocks) * 32;
    CUdeviceptr output;
    CUDA_CHECK(cuMemAlloc(&output, warp_count * 24));
    unsigned a = 0x13579bdfu;
    unsigned b = 0x2468ace0u;
    unsigned c = 0x0f1e2d3cu;
    unsigned d = 0x4b5a6978u;
    void *arguments[] = {&output, &iterations, &a, &b, &c, &d};
    CUDA_CHECK(cuLaunchKernel(function, blocks, 1, 1, 32, 1, 1, 0, nullptr,
                              arguments, nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    std::vector<unsigned char> host(warp_count * 24);
    CUDA_CHECK(cuMemcpyDtoH(host.data(), output, host.size()));
    unsigned long long minimum = ~0ull;
    unsigned long long maximum = 0;
    for (std::size_t warp = 0; warp < warp_count; ++warp) {
        const auto *record = reinterpret_cast<const unsigned long long *>(
            host.data() + warp * 24);
        const unsigned long long elapsed = record[1] - record[0];
        minimum = std::min(minimum, elapsed);
        maximum = std::max(maximum, elapsed);
    }
    std::printf("{\"blocks\":%d,\"iterations\":%u,\"warps\":%zu,"
                "\"min_cycles\":%llu,\"max_cycles\":%llu}\n",
                blocks, iterations, warp_count,
                static_cast<unsigned long long>(minimum),
                static_cast<unsigned long long>(maximum));
    CUDA_CHECK(cuMemFree(output));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));
    return 0;
}
