// P-chase microbenchmark kernels (counter-free by construction).
//
// Dependent-pointer-chase latency probes timed with clock64() INSIDE
// the kernel — latencies are reported in CYCLES, which are
// boost-clock-invariant to first order, so no clock locking is needed
// (and none is performed; the driver state is never touched). No
// profiler of any kind is involved: the only measurement surface is
// clock64() and the host-side read-only telemetry the Python
// orchestrator records around each run.
//
// Probes:
//   chase <footprint_bytes> <iters> <reps> [carveout_pct]
//       single-thread dependent chase over a host-permuted ring of
//       cache lines in GLOBAL memory; one warmup pass then `reps`
//       timed passes; one JSON line per rep on stdout.
//   shared_chase <footprint_bytes> <iters> <reps>
//       same chase over a SHARED-memory index ring.
//   mshr <k> <iters> <reps>
//       k independent chases interleaved in one thread over a
//       DRAM-resident footprint; cycles per k-load batch probes
//       outstanding-miss capacity.
//   conflict <n_lines> <stride_bytes> <iters> <reps>
//       chase over n_lines spaced exactly stride_bytes apart
//       (set-conflict probe; capacity excluded by tiny n_lines).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <random>
#include <cuda_runtime.h>

#define CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
                     cudaGetErrorName(err__), __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

#define LINE_BYTES 128ull

__global__ void init_ring(void **buf, const unsigned *order, long n) {
    // Single thread builds the pointer ring: buf[order[i]] points at
    // buf[order[i+1]] (a permuted cycle over the selected slots).
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (long i = 0; i < n; ++i) {
            buf[order[i] * (LINE_BYTES / sizeof(void *))] =
                (void *)&buf[order[(i + 1) % n]
                             * (LINE_BYTES / sizeof(void *))];
        }
    }
}

__global__ void chase_kernel(void **start, long iters,
                             unsigned long long *out_cycles,
                             void **out_sink) {
    void **p = start;
    // Warmup traversal inside the kernel keeps the timed pass at a
    // settled cache state for cache-resident footprints.
    for (long i = 0; i < iters; ++i) p = (void **)*p;
    unsigned long long t0 = clock64();
    for (long i = 0; i < iters; ++i) p = (void **)*p;
    unsigned long long t1 = clock64();
    *out_cycles = t1 - t0;
    *out_sink = p;  // defeat dead-code elimination
}

extern __shared__ unsigned smem_ring[];

__global__ void shared_chase_kernel(const unsigned *order, long n,
                                    long iters,
                                    unsigned long long *out_cycles,
                                    unsigned *out_sink) {
    if (threadIdx.x != 0) return;
    // Build the index ring in shared memory (stride = one 128B line
    // worth of unsigned slots to mirror the global layout).
    const unsigned slots = LINE_BYTES / sizeof(unsigned);
    for (long i = 0; i < n; ++i) {
        smem_ring[order[i] * slots] = order[(i + 1) % n] * slots;
    }
    unsigned s = order[0] * slots;
    for (long i = 0; i < iters; ++i) s = smem_ring[s];
    unsigned long long t0 = clock64();
    for (long i = 0; i < iters; ++i) s = smem_ring[s];
    unsigned long long t1 = clock64();
    *out_cycles = t1 - t0;
    *out_sink = s;
}

#define MSHR_MAX_WAYS 32

// K must be a compile-time constant so the chain pointers live in
// REGISTERS with a fully unrolled batch: a runtime-indexed local array
// would serialize each batch through local memory and hide the
// outstanding-miss overlap this probe exists to measure.
template <int K>
__global__ void mshr_kernel_t(void **starts, long iters,
                              unsigned long long *out_cycles,
                              void **out_sink) {
    void *p[K];
#pragma unroll
    for (int j = 0; j < K; ++j) p[j] = starts[j];
    unsigned long long t0 = clock64();
    for (long i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < K; ++j) p[j] = *(void **)p[j];
    }
    unsigned long long t1 = clock64();
    *out_cycles = t1 - t0;
    void *acc = nullptr;
#pragma unroll
    for (int j = 0; j < K; ++j) acc = (void *)((char *)acc
                                               + (size_t)p[j]);
    *out_sink = acc;
}

static void launch_mshr(int k, void **d_starts, long iters,
                        unsigned long long *d_cycles, void **d_sink) {
    switch (k) {
    case 1: mshr_kernel_t<1><<<1, 1>>>(d_starts, iters, d_cycles,
                                       d_sink); break;
    case 2: mshr_kernel_t<2><<<1, 1>>>(d_starts, iters, d_cycles,
                                       d_sink); break;
    case 3: mshr_kernel_t<3><<<1, 1>>>(d_starts, iters, d_cycles,
                                       d_sink); break;
    case 4: mshr_kernel_t<4><<<1, 1>>>(d_starts, iters, d_cycles,
                                       d_sink); break;
    case 6: mshr_kernel_t<6><<<1, 1>>>(d_starts, iters, d_cycles,
                                       d_sink); break;
    case 8: mshr_kernel_t<8><<<1, 1>>>(d_starts, iters, d_cycles,
                                       d_sink); break;
    case 12: mshr_kernel_t<12><<<1, 1>>>(d_starts, iters, d_cycles,
                                         d_sink); break;
    case 16: mshr_kernel_t<16><<<1, 1>>>(d_starts, iters, d_cycles,
                                         d_sink); break;
    case 24: mshr_kernel_t<24><<<1, 1>>>(d_starts, iters, d_cycles,
                                         d_sink); break;
    case 32: mshr_kernel_t<32><<<1, 1>>>(d_starts, iters, d_cycles,
                                         d_sink); break;
    default:
        std::fprintf(stderr, "unsupported k=%d\n", k);
        std::exit(2);
    }
}

static std::vector<unsigned> permutation(long n, unsigned seed) {
    std::vector<unsigned> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    return order;
}

static void emit(const char *probe, long footprint, long iters, int rep,
                 unsigned long long cycles, double per_load,
                 const char *extra_key, long extra_val) {
    std::printf("{\"probe\":\"%s\",\"footprint_bytes\":%ld,"
                "\"iters\":%ld,\"rep\":%d,\"cycles\":%llu,"
                "\"cycles_per_load\":%.4f",
                probe, footprint, iters, rep, cycles, per_load);
    if (extra_key) std::printf(",\"%s\":%ld", extra_key, extra_val);
    std::printf("}\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s chase|shared_chase|mshr|conflict ...\n",
                     argv[0]);
        return 2;
    }
    const char *probe = argv[1];

    if (std::strcmp(probe, "chase") == 0) {
        long footprint = std::atol(argv[2]);
        long iters = std::atol(argv[3]);
        int reps = std::atoi(argv[4]);
        int carveout = (argc > 5) ? std::atoi(argv[5]) : -1;
        long n = footprint / LINE_BYTES;
        if (n < 2) { std::fprintf(stderr, "footprint too small\n");
                     return 2; }
        void **buf = nullptr;
        unsigned *d_order = nullptr;
        unsigned long long *d_cycles = nullptr;
        void **d_sink = nullptr;
        CHECK(cudaMalloc(&buf, (size_t)footprint));
        CHECK(cudaMalloc(&d_order, n * sizeof(unsigned)));
        CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
        CHECK(cudaMalloc(&d_sink, sizeof(void *)));
        auto order = permutation(n, 0xC0FFEEu);
        CHECK(cudaMemcpy(d_order, order.data(), n * sizeof(unsigned),
                         cudaMemcpyHostToDevice));
        init_ring<<<1, 1>>>(buf, d_order, n);
        CHECK(cudaDeviceSynchronize());
        if (carveout >= 0) {
            CHECK(cudaFuncSetAttribute(
                chase_kernel,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                carveout));
        }
        void **start = &buf[order[0] * (LINE_BYTES / sizeof(void *))];
        for (int r = 0; r < reps; ++r) {
            chase_kernel<<<1, 1>>>(start, iters, d_cycles, d_sink);
            CHECK(cudaDeviceSynchronize());
            unsigned long long cycles = 0;
            CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles),
                             cudaMemcpyDeviceToHost));
            emit("chase", footprint, iters, r, cycles,
                 (double)cycles / (double)iters,
                 carveout >= 0 ? "carveout_pct" : nullptr, carveout);
        }
        return 0;
    }

    if (std::strcmp(probe, "shared_chase") == 0) {
        long footprint = std::atol(argv[2]);
        long iters = std::atol(argv[3]);
        int reps = std::atoi(argv[4]);
        long n = footprint / LINE_BYTES;
        if (n < 2) { std::fprintf(stderr, "footprint too small\n");
                     return 2; }
        unsigned *d_order = nullptr;
        unsigned long long *d_cycles = nullptr;
        unsigned *d_sink = nullptr;
        CHECK(cudaMalloc(&d_order, n * sizeof(unsigned)));
        CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
        CHECK(cudaMalloc(&d_sink, sizeof(unsigned)));
        auto order = permutation(n, 0xC0FFEEu);
        CHECK(cudaMemcpy(d_order, order.data(), n * sizeof(unsigned),
                         cudaMemcpyHostToDevice));
        CHECK(cudaFuncSetAttribute(
            shared_chase_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)footprint));
        for (int r = 0; r < reps; ++r) {
            shared_chase_kernel<<<1, 32, footprint>>>(
                d_order, n, iters, d_cycles, d_sink);
            CHECK(cudaDeviceSynchronize());
            unsigned long long cycles = 0;
            CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles),
                             cudaMemcpyDeviceToHost));
            emit("shared_chase", footprint, iters, r, cycles,
                 (double)cycles / (double)iters, nullptr, 0);
        }
        return 0;
    }

    if (std::strcmp(probe, "mshr") == 0) {
        int k = std::atoi(argv[2]);
        long iters = std::atol(argv[3]);
        int reps = std::atoi(argv[4]);
        if (k < 1 || k > MSHR_MAX_WAYS) { return 2; }
        // DRAM-resident footprint per chain (64 MiB total) so every
        // access in every chain is a miss.
        long footprint = 64ll * 1024 * 1024;
        long n = footprint / LINE_BYTES;
        void **buf = nullptr;
        unsigned *d_order = nullptr;
        unsigned long long *d_cycles = nullptr;
        void **d_sink = nullptr;
        void **d_starts = nullptr;
        CHECK(cudaMalloc(&buf, (size_t)footprint));
        CHECK(cudaMalloc(&d_order, n * sizeof(unsigned)));
        CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
        CHECK(cudaMalloc(&d_sink, sizeof(void *)));
        CHECK(cudaMalloc(&d_starts, k * sizeof(void *)));
        auto order = permutation(n, 0xC0FFEEu);
        CHECK(cudaMemcpy(d_order, order.data(), n * sizeof(unsigned),
                         cudaMemcpyHostToDevice));
        init_ring<<<1, 1>>>(buf, d_order, n);
        CHECK(cudaDeviceSynchronize());
        // k starts spread evenly around the ring.
        std::vector<void *> starts(k);
        for (int j = 0; j < k; ++j) {
            starts[j] = (void *)&buf[order[(size_t)(n / k) * j]
                                     * (LINE_BYTES / sizeof(void *))];
        }
        CHECK(cudaMemcpy(d_starts, starts.data(), k * sizeof(void *),
                         cudaMemcpyHostToDevice));
        for (int r = 0; r < reps; ++r) {
            launch_mshr(k, d_starts, iters, d_cycles, d_sink);
            CHECK(cudaDeviceSynchronize());
            unsigned long long cycles = 0;
            CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles),
                             cudaMemcpyDeviceToHost));
            // cycles per BATCH of k loads.
            emit("mshr", footprint, iters, r, cycles,
                 (double)cycles / (double)iters, "k", k);
        }
        return 0;
    }

    if (std::strcmp(probe, "conflict") == 0) {
        long n_lines = std::atol(argv[2]);
        long stride = std::atol(argv[3]);
        long iters = std::atol(argv[4]);
        int reps = std::atoi(argv[5]);
        // Allocate n_lines slots spaced `stride` apart; chase visits
        // them in a fixed shuffled cycle.
        size_t span = (size_t)stride * (size_t)n_lines + LINE_BYTES;
        char *raw = nullptr;
        CHECK(cudaMalloc(&raw, span));
        unsigned long long *d_cycles = nullptr;
        void **d_sink = nullptr;
        CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
        CHECK(cudaMalloc(&d_sink, sizeof(void *)));
        auto order = permutation(n_lines, 0xC0FFEEu);
        // Build the ring host-side as device addresses, copy in one go.
        std::vector<void *> host_ring(n_lines);
        for (long i = 0; i < n_lines; ++i) {
            char *slot = raw + (size_t)order[i] * stride;
            char *next = raw + (size_t)order[(i + 1) % n_lines] * stride;
            host_ring[i] = next;
            CHECK(cudaMemcpy(slot, &host_ring[i], sizeof(void *),
                             cudaMemcpyHostToDevice));
        }
        void **start = (void **)(raw + (size_t)order[0] * stride);
        for (int r = 0; r < reps; ++r) {
            chase_kernel<<<1, 1>>>(start, iters, d_cycles, d_sink);
            CHECK(cudaDeviceSynchronize());
            unsigned long long cycles = 0;
            CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(cycles),
                             cudaMemcpyDeviceToHost));
            emit("conflict", n_lines * stride, iters, r, cycles,
                 (double)cycles / (double)iters, "stride_bytes", stride);
        }
        return 0;
    }

    std::fprintf(stderr, "unknown probe: %s\n", probe);
    return 2;
}
