#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define DRIVER_CHECK(call)                                                      \
    do {                                                                        \
        const CUresult status = (call);                                         \
        if (status != CUDA_SUCCESS) {                                           \
            const char* name = nullptr;                                         \
            const char* description = nullptr;                                  \
            cuGetErrorName(status, &name);                                      \
            cuGetErrorString(status, &description);                             \
            std::fprintf(stderr, "%s failed: %s (%s)\n", #call,              \
                         name == nullptr ? "unknown" : name,                   \
                         description == nullptr ? "no description"             \
                                                : description);                 \
            return 1;                                                           \
        }                                                                       \
    } while (0)

constexpr const char* kFunction = "_Z20bypass_global_canaryPjPKjS1_";

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }
    const int trials = std::atoi(argv[2]);
    constexpr int threads = 32;
    std::vector<std::uint32_t> inputs(threads, 0x12345678u);
    std::vector<std::uint32_t> zeros(threads, 0);
    std::vector<std::uint32_t> sink(threads);
    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    CUdeviceptr sink_device;
    CUdeviceptr inputs_device;
    CUdeviceptr zeros_device;
    DRIVER_CHECK(cuMemAlloc(&sink_device, threads * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&inputs_device, threads * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&zeros_device, threads * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(inputs_device, inputs.data(),
                              threads * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(zeros_device, zeros.data(),
                              threads * sizeof(std::uint32_t)));
    for (int trial = 0; trial < trials; ++trial) {
        void* arguments[] = {&sink_device, &inputs_device, &zeros_device};
        DRIVER_CHECK(cuLaunchKernel(function, 1, 1, 1, threads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
    }
    DRIVER_CHECK(cuMemcpyDtoH(sink.data(), sink_device,
                              threads * sizeof(std::uint32_t)));
    const bool correct = sink == inputs;
    std::printf("{\"trials\":%d,\"first\":\"%08x\",\"correct\":%s}\n",
                trials, sink[0], correct ? "true" : "false");
    return correct ? 0 : 1;
}
