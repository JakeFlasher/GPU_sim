#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void message_passing_device(unsigned* data, unsigned* flag,
                                       unsigned* out, unsigned expected) {
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

static void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    unsigned iterations = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 64;
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
        if (host_flag != 1u) ++timeouts;
        if (host_out != 1u) ++failures;
    }
    std::printf("LITMUS {\"schema\":\"sm86_message_passing_device_litmus_v1\",\"iterations\":%u,\"failures\":%u,\"timeouts\":%u,\"status\":\"%s\"}\n",
                iterations, failures, timeouts,
                failures == 0 && timeouts == 0 ? "ok" : "mismatch");
    cudaFree(out); cudaFree(flag); cudaFree(data);
    return (failures == 0 && timeouts == 0) ? 0 : 1;
}
