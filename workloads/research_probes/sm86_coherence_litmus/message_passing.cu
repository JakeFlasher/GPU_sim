#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void message_passing(unsigned *data, unsigned *flag, unsigned *out,
                                unsigned expected) {
    if (blockIdx.x == 0) {
        *data = expected;
        __threadfence();
        atomicExch(flag, 1u);
    } else {
        unsigned seen = 0;
        for (unsigned spins = 0; spins < 100000000u && seen == 0; ++spins)
            seen = atomicAdd(flag, 0u);
        *out = (seen == 1u && *data == expected) ? 1u : 0u;
    }
}

static void check(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

int main(int argc, char **argv) {
    unsigned iterations = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 1000;
    unsigned *data = nullptr, *flag = nullptr, *out = nullptr;
    check(cudaMallocManaged(&data, sizeof(unsigned)), "data allocation");
    check(cudaMallocManaged(&flag, sizeof(unsigned)), "flag allocation");
    check(cudaMallocManaged(&out, sizeof(unsigned)), "out allocation");
    unsigned failures = 0, timeouts = 0;
    for (unsigned iteration = 1; iteration <= iterations; ++iteration) {
        *data = 0; *flag = 0; *out = 0;
        check(cudaDeviceSynchronize(), "initial synchronize");
        message_passing<<<2, 1>>>(data, flag, out, iteration);
        check(cudaGetLastError(), "message launch");
        check(cudaDeviceSynchronize(), "message synchronize");
        if (*flag != 1u) ++timeouts;
        if (*out != 1u) ++failures;
    }
    std::printf("LITMUS {\"schema\":\"sm86_message_passing_litmus_v1\",\"iterations\":%u,\"failures\":%u,\"timeouts\":%u,\"status\":\"%s\"}\n",
                iterations, failures, timeouts, failures == 0 && timeouts == 0 ? "ok" : "mismatch");
    cudaFree(data); cudaFree(flag); cudaFree(out);
    return (failures == 0 && timeouts == 0) ? 0 : 1;
}
