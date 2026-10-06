#include <cuda.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

bool parse_bounded(const char* text, unsigned maximum, unsigned* value) {
    if (text == nullptr || *text < '0' || *text > '9') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed < 1 || parsed > maximum) return false;
    *value = static_cast<unsigned>(parsed);
    return true;
}

int main(int argc, char **argv) {
    unsigned blocks = 84, iterations = 256, trials = 1;
    if (argc > 5 || (argc > 1 && !parse_bounded(argv[1], 4096, &blocks)) ||
        (argc > 2 && !parse_bounded(argv[2], 262144, &iterations)) ||
        (argc > 3 && !parse_bounded(argv[3], 32, &trials))) {
        std::fprintf(stderr, "usage: %s [BLOCKS [ITERATIONS [TRIALS [CUBIN]]]]\n", argv[0]);
        return 2;
    }
    // Resolve the default module beside the running executable, not the cwd.
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (argc < 5 && error) {
        std::fprintf(stderr, "cannot locate imc_kernel.cubin: %s\n", error.message().c_str());
        return 1;
    }
    const auto cubin = argc == 5 ? std::filesystem::path(argv[4])
                                : executable.parent_path() / "imc_kernel.cubin";

    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    CUmodule module;
    CUDA_CHECK(cuModuleLoad(&module, cubin.c_str()));
    CUfunction function;
    CUDA_CHECK(cuModuleGetFunction(&function, module, "imc_kernel"));

    // The kernel launches one warp/block but writes one 24-byte record/thread.
    // With blocks <= 4096 this allocation is bounded below 4 MiB.
    const std::size_t records = static_cast<std::size_t>(blocks) * 32;
    CUdeviceptr output;
    CUDA_CHECK(cuMemAlloc(&output, records * 24));
    unsigned a = 0x13579bdfu;
    unsigned b = 0x2468ace0u;
    unsigned c = 0x0f1e2d3cu;
    unsigned d = 0x4b5a6978u;
    void *arguments[] = {&output, &iterations, &a, &b, &c, &d};
    std::vector<unsigned char> host(records * 24);
    unsigned long long minimum = ~0ull;
    unsigned long long maximum = 0;
    const std::uint32_t expected = static_cast<std::uint32_t>(
        (std::uint64_t{a} + b + c + d) * iterations);
    bool sink_valid = true, timestamp_bounds = true;
    for (unsigned trial = 0; trial < trials; ++trial) {
        CUDA_CHECK(cuMemsetD8(output, 0xff, host.size()));
        CUDA_CHECK(cuLaunchKernel(function, blocks, 1, 1, 32, 1, 1, 0, nullptr,
                                  arguments, nullptr));
        CUDA_CHECK(cuCtxSynchronize());
        CUDA_CHECK(cuMemcpyDtoH(host.data(), output, host.size()));
        for (std::size_t i = 0; i < records; ++i) {
            std::uint64_t start = 0, stop = 0;
            std::uint32_t value = 0;
            // memcpy avoids alignment and object-lifetime UB on a byte buffer.
            std::memcpy(&start, host.data() + i * 24, 8);
            std::memcpy(&stop, host.data() + i * 24 + 8, 8);
            std::memcpy(&value, host.data() + i * 24 + 16, 4);
            sink_valid = sink_valid && value == expected;
            const bool valid = start != UINT64_MAX && stop != UINT64_MAX && stop > start;
            timestamp_bounds = timestamp_bounds && valid;
            if (valid) {
                minimum = std::min(minimum, static_cast<unsigned long long>(stop - start));
                maximum = std::max(maximum, static_cast<unsigned long long>(stop - start));
            }
        }
    }
    const bool correct = sink_valid && timestamp_bounds;
    std::printf("{\"blocks\":%u,\"iterations\":%u,\"warps\":%u,\"thread_records\":%zu,"
                "\"trials\":%u,\"expected\":%u,\"sink_valid\":%s,\"timestamp_bounds\":%s,"
                "\"correct\":%s,\"min_cycles\":%llu,\"max_cycles\":%llu}\n",
                blocks, iterations, blocks, records, trials, expected,
                sink_valid ? "true" : "false", timestamp_bounds ? "true" : "false",
                correct ? "true" : "false",
                static_cast<unsigned long long>(minimum),
                static_cast<unsigned long long>(maximum));
    CUDA_CHECK(cuMemFree(output));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
