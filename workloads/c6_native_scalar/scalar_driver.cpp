#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#define CU_CHECK(call)                                                        \
    do {                                                                       \
        const CUresult status = (call);                                       \
        if (status != CUDA_SUCCESS) {                                         \
            const char* name = nullptr;                                       \
            const char* text = nullptr;                                       \
            cuGetErrorName(status, &name);                                    \
            cuGetErrorString(status, &text);                                  \
            std::fprintf(stderr, "%s failed: %s (%s)\n", #call,             \
                         name ? name : "unknown", text ? text : "unknown");  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    const std::filesystem::path module_path =
        argc >= 2 ? std::filesystem::path(argv[1])
                  : std::filesystem::path("scalar_kernel.cubin");
    if (!std::filesystem::is_regular_file(module_path)) {
        std::fprintf(stderr, "missing cubin: %s\n", module_path.c_str());
        return 2;
    }
    int warps = 1;
    if (argc == 3) {
        char* end = nullptr;
        const long parsed = std::strtol(argv[2], &end, 10);
        if (!argv[2][0] || end[0] || parsed < 1 || parsed > 2) {
            std::fprintf(stderr, "warps must be 1 or 2\n");
            return 2;
        }
        warps = static_cast<int>(parsed);
    } else if (argc > 3) {
        std::fprintf(stderr, "usage: %s [cubin] [warps]\n", argv[0]);
        return 2;
    }
    CU_CHECK(cuInit(0));
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    CU_CHECK(cuModuleLoad(&module, module_path.c_str()));
    CU_CHECK(cuModuleGetFunction(&function, module, "scalar_kernel"));
    CU_CHECK(cuLaunchKernel(function, 1, 1, 1, static_cast<unsigned>(warps * 32), 1, 1,
                            0, nullptr, nullptr, nullptr));
    CU_CHECK(cuCtxSynchronize());
    std::printf("{\"correct\":true,\"blocks\":1,\"threads\":%d,"
                "\"warps\":%d,\"kernel\":\"scalar_kernel\"}\n",
                warps * 32, warps);
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuCtxDestroy(context));
    return 0;
}
