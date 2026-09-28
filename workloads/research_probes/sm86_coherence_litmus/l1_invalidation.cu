#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void l1_invalidation(unsigned *data, unsigned *flag,
                                unsigned *preload, unsigned *out,
                                unsigned expected) {
    if (blockIdx.x == 0) {
        *data = expected;
        __threadfence();
        atomicExch(flag, 1u);
    } else {
        unsigned before = *data;
        *preload = before;
        unsigned seen = 0;
        for (unsigned spins = 0; spins < 100000000u && seen == 0; ++spins)
            seen = atomicAdd(flag, 0u);
        unsigned after = *data;
        *out = (seen == 1u && after == expected) ? 1u : 0u;
    }
}

__global__ void l1_invalidation_volatile(unsigned *data, unsigned *flag,
                                         unsigned *preload, unsigned *out,
                                         unsigned expected) {
    volatile unsigned *volatile_data = data;
    if (blockIdx.x == 0) {
        *volatile_data = expected;
        __threadfence();
        atomicExch(flag, 1u);
    } else {
        unsigned before = *volatile_data;
        *preload = before;
        unsigned seen = 0;
        for (unsigned spins = 0; spins < 100000000u && seen == 0; ++spins)
            seen = atomicAdd(flag, 0u);
        unsigned after = *volatile_data;
        *out = (seen == 1u && after == expected) ? 1u : 0u;
    }
}

__global__ void l1_invalidation_consumer_fence(unsigned *data, unsigned *flag,
                                               unsigned *preload, unsigned *out,
                                               unsigned expected) {
    if (blockIdx.x == 0) {
        *data = expected;
        __threadfence();
        atomicExch(flag, 1u);
    } else {
        unsigned before = *data;
        *preload = before;
        unsigned seen = 0;
        for (unsigned spins = 0; spins < 100000000u && seen == 0; ++spins)
            seen = atomicAdd(flag, 0u);
        __threadfence();
        unsigned after = *data;
        *out = (seen == 1u && after == expected) ? 1u : 0u;
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
    unsigned *data = nullptr, *flag = nullptr, *preload = nullptr, *out = nullptr;
    check(cudaMallocManaged(&data, sizeof(unsigned)), "data allocation");
    check(cudaMallocManaged(&flag, sizeof(unsigned)), "flag allocation");
    check(cudaMallocManaged(&preload, sizeof(unsigned)), "preload allocation");
    check(cudaMallocManaged(&out, sizeof(unsigned)), "out allocation");
    unsigned failures = 0, timeouts = 0, producer_won_preload = 0;
    unsigned volatile_failures = 0, consumer_fence_failures = 0;
    for (unsigned iteration = 1; iteration <= iterations; ++iteration) {
        *data = 0; *flag = 0; *preload = 0; *out = 0;
        check(cudaDeviceSynchronize(), "initial synchronize");
        l1_invalidation<<<2, 1>>>(data, flag, preload, out, iteration);
        check(cudaGetLastError(), "l1 invalidation launch");
        check(cudaDeviceSynchronize(), "l1 invalidation synchronize");
        if (*flag != 1u) ++timeouts;
        if (*preload == iteration) ++producer_won_preload;
        if (*out != 1u) ++failures;
        *data = 0; *flag = 0; *preload = 0; *out = 0;
        check(cudaDeviceSynchronize(), "volatile initial synchronize");
        l1_invalidation_volatile<<<2, 1>>>(data, flag, preload, out, iteration);
        check(cudaGetLastError(), "volatile l1 invalidation launch");
        check(cudaDeviceSynchronize(), "volatile l1 invalidation synchronize");
        if (*out != 1u) ++volatile_failures;
        *data = 0; *flag = 0; *preload = 0; *out = 0;
        check(cudaDeviceSynchronize(), "consumer fence initial synchronize");
        l1_invalidation_consumer_fence<<<2, 1>>>(data, flag, preload, out, iteration);
        check(cudaGetLastError(), "consumer fence l1 invalidation launch");
        check(cudaDeviceSynchronize(), "consumer fence l1 invalidation synchronize");
        if (*out != 1u) ++consumer_fence_failures;
    }
    std::printf("LITMUS {\"schema\":\"sm86_l1_invalidation_litmus_v1\",\"iterations\":%u,\"failures\":%u,\"timeouts\":%u,\"producer_won_preload\":%u,\"volatile_failures\":%u,\"consumer_fence_failures\":%u,\"status\":\"%s\"}\n",
                iterations, failures, timeouts, producer_won_preload,
                volatile_failures, consumer_fence_failures,
                failures == 0 && timeouts == 0 && volatile_failures == 0 && consumer_fence_failures == 0 ? "ok" : "mismatch");
    cudaFree(data); cudaFree(flag); cudaFree(preload); cudaFree(out);
    return (failures == 0 && timeouts == 0 && volatile_failures == 0 && consumer_fence_failures == 0) ? 0 : 1;
}
