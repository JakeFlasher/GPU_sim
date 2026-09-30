// Load and launch an SM86 cubin produced by nvJitLink through the CUDA driver
// API. This is an execution adapter, not a correctness or verification test.

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name ? name : "CUDA_ERROR_UNKNOWN") + " (" +
                           (text ? text : "") + ")");
}

std::vector<char> read_binary(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error(std::string("cannot open ") + path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: nvjitlink_launch_sm86 INPUT.cubin [launches]\n");
    return 2;
  }
  const int launches = argc == 3 ? std::atoi(argv[2]) : 1000;
  if (launches <= 0) {
    std::fprintf(stderr, "launches must be positive\n");
    return 2;
  }
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUdeviceptr output = 0;
  CUevent start = nullptr;
  CUevent stop = nullptr;
  try {
    check(cuInit(0), "cuInit");
    CUdevice device = 0;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");
    const std::vector<char> image = read_binary(argv[1]);
    check(cuModuleLoadData(&module, image.data()), "cuModuleLoadData");
    CUfunction function = nullptr;
    check(cuModuleGetFunction(&function, module, "value_kernel"),
          "cuModuleGetFunction(value_kernel)");
    check(cuMemAlloc(&output, 32 * sizeof(std::uint32_t)), "cuMemAlloc");
    check(cuMemsetD32(output, 0, 32), "cuMemsetD32");
    check(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
    check(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
    void* args[] = {&output};
    check(cuEventRecord(start, nullptr), "cuEventRecord(start)");
    for (int i = 0; i < launches; ++i) {
      check(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args,
                           nullptr),
            "cuLaunchKernel(value_kernel)");
    }
    check(cuEventRecord(stop, nullptr), "cuEventRecord(stop)");
    check(cuEventSynchronize(stop), "cuEventSynchronize(stop)");
    float milliseconds = 0.0f;
    check(cuEventElapsedTime(&milliseconds, start, stop),
          "cuEventElapsedTime");
    std::uint32_t host[32] = {};
    check(cuMemcpyDtoH(host, output, sizeof(host)), "cuMemcpyDtoH");
    std::printf("device=0 launches=%d event_ms=%.6f first_values=%u,%u,%u,%u\n",
                launches, milliseconds, host[0], host[1], host[2], host[3]);
    check(cuEventDestroy(stop), "cuEventDestroy(stop)");
    check(cuEventDestroy(start), "cuEventDestroy(start)");
    check(cuMemFree(output), "cuMemFree");
    check(cuModuleUnload(module), "cuModuleUnload");
    check(cuCtxDestroy(context), "cuCtxDestroy");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "nvjitlink_launch_sm86: %s\n", error.what());
    if (stop) cuEventDestroy(stop);
    if (start) cuEventDestroy(start);
    if (output) cuMemFree(output);
    if (module) cuModuleUnload(module);
    if (context) cuCtxDestroy(context);
    return 1;
  }
}
