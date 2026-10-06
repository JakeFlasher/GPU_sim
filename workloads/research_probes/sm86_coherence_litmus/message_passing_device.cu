#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>

__global__ void message_passing_device(unsigned* data, unsigned* flag,
                                       unsigned* out, unsigned expected) {
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(*flag);
    if (blockIdx.x == 0) {
        *data = expected;
        ready.store(1u, cuda::memory_order_release);
    } else {
        unsigned seen = 0;
        for (unsigned spins = 0; spins < 1000000u && seen == 0; ++spins)
            seen = ready.load(cuda::memory_order_acquire);
        *out = seen == 0u ? 2u : (*data == expected ? 1u : 0u);
    }
}

static void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc > 2) return 2;
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = argc == 2 ? std::strtoul(argv[1], &end, 10) : 64;
    if (errno || (argc == 2 && (!*argv[1] || *end || *argv[1] == '-')) ||
        parsed < 1 || parsed > 1024) return 2;
    const unsigned iterations = static_cast<unsigned>(parsed);
    unsigned *data = nullptr, *flag = nullptr, *out = nullptr;
    check(cudaMalloc(&data, sizeof(unsigned)), "data allocation");
    check(cudaMalloc(&flag, sizeof(unsigned)), "flag allocation");
    check(cudaMalloc(&out, sizeof(unsigned)), "out allocation");
    unsigned failures = 0, timeouts = 0;
    for (unsigned iteration = 1; iteration <= iterations; ++iteration) {
        unsigned zero = 0;
        check(cudaMemcpy(data, &zero, sizeof(zero), cudaMemcpyHostToDevice), "data reset");
        check(cudaMemcpy(flag, &zero, sizeof(zero), cudaMemcpyHostToDevice), "flag reset");
        check(cudaMemcpy(out, &zero, sizeof(zero), cudaMemcpyHostToDevice), "out reset");
        message_passing_device<<<2, 1>>>(data, flag, out, iteration);
        check(cudaGetLastError(), "message launch");
        check(cudaDeviceSynchronize(), "message synchronize");
        unsigned host_flag = 0, host_out = 0;
        check(cudaMemcpy(&host_flag, flag, sizeof(host_flag), cudaMemcpyDeviceToHost), "flag copy");
        check(cudaMemcpy(&host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost), "out copy");
        if (host_out == 2u) ++timeouts;
        if (host_out != 1u || host_flag != 1u) ++failures;
    }
    std::printf("LITMUS {\"schema\":\"sm86_message_passing_device_litmus_v1\",\"iterations\":%u,\"failures\":%u,\"timeouts\":%u,\"status\":\"%s\"}\n",
                iterations, failures, timeouts,
                failures == 0 && timeouts == 0 ? "ok" : "mismatch");
    cudaFree(out); cudaFree(flag); cudaFree(data);
    return (failures == 0 && timeouts == 0) ? 0 : 1;
}
