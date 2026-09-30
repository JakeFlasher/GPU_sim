#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void check(CUresult result, const char *where) {
  if (result == CUDA_SUCCESS) return;
  const char *name = nullptr, *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(where) + ":" + (name ? name : "unknown") + ":" + (message ? message : "unknown"));
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s BRIDGED_CUBIN [REPETITIONS]\n", argv[0]);
    return 2;
  }
  const int repetitions = argc == 3 ? std::atoi(argv[2]) : 256;
  if (repetitions <= 0) return 2;
  try {
    check(cuInit(0), "cuInit");
    CUdevice device;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context;
    check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");
    CUmodule module;
    check(cuModuleLoad(&module, argv[1]), "cuModuleLoad");
    CUfunction function;
    check(cuModuleGetFunction(&function, module, "value_kernel"), "cuModuleGetFunction");
    CUdeviceptr output, input;
    check(cuMemAlloc(&output, 128), "cuMemAlloc output");
    check(cuMemAlloc(&input, 128), "cuMemAlloc input");
    uint32_t host[32];
    for (int i = 0; i < 32; ++i) host[i] = static_cast<uint32_t>(i);
    check(cuMemcpyHtoD(input, host, sizeof(host)), "cuMemcpyHtoD");
    void *parameters[] = {&output, &input};
    for (int i = 0; i < repetitions; ++i) {
      check(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, parameters, nullptr), "cuLaunchKernel");
    }
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    check(cuMemcpyDtoH(host, output, sizeof(host)), "cuMemcpyDtoH");
    std::printf("values=%u,%u,%u,%u repetitions=%d\n", host[0], host[1], host[2], host[3], repetitions);
    check(cuMemFree(input), "cuMemFree input");
    check(cuMemFree(output), "cuMemFree output");
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
