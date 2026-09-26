// Real driver-API regression for exact module/library ownership. Run via
// module_origin.sh; no application-specific simulator or NCU data is used.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static void check(CUresult rc, const char* call) {
  if (rc == CUDA_SUCCESS) return;
  const char* detail = nullptr;
  cuGetErrorName(rc, &detail);
  std::fprintf(stderr, "%s: %d (%s)\n", call, int(rc),
               detail ? detail : "unknown");
  std::exit(1);
}
#define CU(call) check((call), #call)

static void launch(CUfunction function, unsigned expected) {
  CUdeviceptr output = 0;
  CU(cuMemAlloc(&output, 32 * sizeof(unsigned)));
  void* args[] = {&output};
  CU(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
  CU(cuCtxSynchronize());
  unsigned host[32] = {};
  CU(cuMemcpyDtoH(host, output, sizeof(host)));
  CU(cuMemFree(output));
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (host[lane] != expected + lane) {
      std::fprintf(stderr, "wrong kernel: lane=%u value=%u expected=%u\n",
                   lane, host[lane], expected + lane);
      std::exit(1);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const bool library_mode = std::strncmp(argv[1], "library", 7) == 0;
  const bool function_mode = std::strcmp(argv[1], "library-function") == 0;
  const bool precontext = std::strcmp(argv[1], "library-precontext") == 0;
  CU(cuInit(0));
  CUdevice device;
  CU(cuDeviceGet(&device, 0));

  // A load before context creation can be invisible to NVBit 1.8. That case
  // must remain explicitly unavailable rather than reuse a same-name origin.
  CUcontext ctx = nullptr;
  if (!precontext) CU(cuCtxCreate(&ctx, nullptr, 0, device));
  CUlibrary libraries[2] = {};
  if (library_mode) {
    for (int i = 0; i < 2; ++i)
      CU(cuLibraryLoadFromFile(&libraries[i], argv[i + 2], nullptr, nullptr,
                               0, nullptr, nullptr, 0));
  }
  if (precontext) CU(cuCtxCreate(&ctx, nullptr, 0, device));
  CUfunction functions[2] = {};
  CUmodule modules[2] = {};
  for (int i = 0; i < 2; ++i) {
    if (library_mode) {
      CUkernel kernel = nullptr;
      CU(cuLibraryGetKernel(&kernel, libraries[i], "origin_collision"));
      if (function_mode) CU(cuKernelGetFunction(&functions[i], kernel));
      else functions[i] = reinterpret_cast<CUfunction>(kernel);
    } else {
      CU(cuModuleLoad(&modules[i], argv[i + 2]));
      CU(cuModuleGetFunction(&functions[i], modules[i], "origin_collision"));
    }
  }
  // A and B have the same name and are simultaneously live. The final A
  // catches a "last name wins" resolver even if both outputs happen to work.
  launch(functions[0], 11);
  launch(functions[1], 29);
  launch(functions[0], 11);
  if (library_mode) {
    CU(cuCtxDestroy(ctx));
    // Same libraries, a new context: library origins must survive teardown.
    CU(cuCtxCreate(&ctx, nullptr, 0, device));
    CUkernel kernel = nullptr;
    CU(cuLibraryGetKernel(&kernel, libraries[0], "origin_collision"));
    CUfunction function = reinterpret_cast<CUfunction>(kernel);
    if (function_mode) CU(cuKernelGetFunction(&function, kernel));
    launch(function, 11);
    for (auto library : libraries) CU(cuLibraryUnload(library));
  } else {
    CU(cuModuleUnload(modules[0]));
    // Reuse the symbol after unloading A. Old function handles must not
    // override the new module's actual origin.
    CU(cuModuleLoad(&modules[0], argv[3]));
    CU(cuModuleGetFunction(&functions[0], modules[0], "origin_collision"));
    launch(functions[0], 29);
    // Anonymous in-memory cubin with the same symbol must stay unresolved.
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.empty()) return 1;
    CUmodule anonymous = nullptr;
    CU(cuModuleLoadData(&anonymous, bytes.data()));
    CUfunction function = nullptr;
    CU(cuModuleGetFunction(&function, anonymous, "origin_collision"));
    launch(function, 11);
    CU(cuModuleUnload(anonymous));
    for (auto module : modules) CU(cuModuleUnload(module));
  }
  CU(cuCtxDestroy(ctx));
  std::printf("module_origin_probe %s: PASS\n", argv[1]);
}
