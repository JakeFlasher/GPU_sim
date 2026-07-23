// sm_120 L2 associativity probe.
//
// Goal: empirically determine `associativity_l2` for the GB206 (RTX
// 5060 Laptop) L2 cache by measuring how the L2 hit rate degrades
// when N conflicting cache lines compete for the same set.
//
// Method (classic Saavedra-style):
//   1. Pre-build a pointer-chase ring of N nodes spaced at a wide
//      power-of-2 stride S. The stride is wider than (L2 capacity /
//      num_slices / associativity) so that all N nodes map to a
//      small number of L2 sets (in the limit, the same set on the
//      same slice, modulo NVIDIA's slice-mapping hash).
//   2. Warm pass: walk the ring once (load every line into L2).
//   3. Measured pass: walk the ring TWICE in a row. The L2 hit
//      rate of the measured pass reveals whether L2 retained the
//      ring or evicted it under conflict pressure.
//   4. Vary N from 1 to 32 (or wider). N ≤ associativity → hit rate
//      near 100%; N > associativity → hit rate drops to 0% (every
//      access evicts the line we'll need next).
//
// Per Critical Rule 2, this kernel does NOT mutate clock policy.
// Per Critical Rule 1, this kernel itself runs counter-free; the
// L2 hit rate is harvested OUT-OF-BAND by running NCU around the
// kernel (the calibration-only carve-out).
//
// Build:
//   nvcc -arch=sm_120 -O2 -o probe probe.cu
//
// Run (probe with N=8):
//   ./probe 8
//
// Harvest with NCU (in offline calibration carve-out):
//   ncu --clock-control none \
//       --metrics lts__t_sectors_op_read.sum,\
//                 lts__t_sectors_op_read_lookup_hit.sum \
//       --kernel-name "pointer_chase_kernel" \
//       --kernel-name-base mangled \
//       --launch-skip-before-match 0 --launch-count 1 \
//       ./probe N

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// Each "node" in the chase ring fits in one 128 B cache line and
// stores the byte offset of the next node. Padding ensures one node
// per line so adjacent nodes never piggyback.
struct alignas(128) Node {
    std::uint64_t next_offset;
    std::uint64_t pad[15];
};

// First pass: walk the ring N steps (warm-up).
__global__ void warmup_kernel(const Node* __restrict__ ring,
                              int steps) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t off = 0;
        for (int i = 0; i < steps; ++i) {
            off = ring[off / sizeof(Node)].next_offset;
        }
        // Force the compiler to keep the walk by using `off`.
        if (off == 0xdeadbeef) {
            // Unreachable in normal data, but the dependency stops
            // the compiler from dead-code-eliminating the loop.
            printf("warmup terminator hit\n");
        }
    }
}

// Measured pass: walk the ring 2*N steps. NCU captures
// lts__t_sectors_op_read_lookup_hit.sum / lts__t_sectors_op_read.sum
// across this kernel only.
__global__ void pointer_chase_kernel(const Node* __restrict__ ring,
                                      int steps) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t off = 0;
        for (int i = 0; i < steps; ++i) {
            off = ring[off / sizeof(Node)].next_offset;
        }
        if (off == 0xdeadbeef) {
            printf("measure terminator hit\n");
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "Usage: %s <N>\n  N = ring length\n",
                     argv[0]);
        return 2;
    }
    int N = std::atoi(argv[1]);
    if (N < 1 || N > 4096) {
        std::fprintf(stderr,
                     "N out of range (1..4096): %d\n", N);
        return 2;
    }

    // GB206 L2 = 32 MiB; we want a stride such that
    // (N * stride) significantly exceeds (L2_capacity / num_slices),
    // forcing the N nodes to compete inside one slice's working set.
    //
    // L2 capacity per slice ≈ 32 MiB / 24 ≈ 1365 KB.
    // Stride 2 MiB = 2_097_152 bytes is comfortably above per-slice
    // capacity; N * 2 MiB nodes saturate any one slice's set bank.
    //
    // Allocation size: N * stride bytes. N=64 * 2 MiB = 128 MiB,
    // fits in the 8 GiB device.
    const std::size_t stride = 2 * 1024 * 1024;
    const std::size_t buf_bytes = static_cast<std::size_t>(N) * stride;

    Node* d_ring = nullptr;
    cudaError_t err = cudaMalloc(&d_ring, buf_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cudaMalloc(%zu) failed: %s\n",
                     buf_bytes, cudaGetErrorString(err));
        return 1;
    }

    // Build the chase ring on the host: node i points to node
    // (i+1) % N, by byte offset.
    Node* h_ring = static_cast<Node*>(std::calloc(N, sizeof(Node)));
    if (h_ring == nullptr) {
        std::fprintf(stderr, "host calloc failed\n");
        cudaFree(d_ring);
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        std::size_t next = static_cast<std::size_t>((i + 1) % N) * stride;
        h_ring[i].next_offset = static_cast<std::uint64_t>(next);
    }

    // Copy nodes into the device buffer at their corresponding
    // stride positions. Each node lives at d_ring + i*stride (in
    // byte terms). We can't write a strided pattern from the host
    // in one go, so do per-node memcpys.
    for (int i = 0; i < N; ++i) {
        char* dst = reinterpret_cast<char*>(d_ring) + static_cast<std::size_t>(i) * stride;
        err = cudaMemcpy(dst, &h_ring[i], sizeof(Node),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "cudaMemcpy node %d failed: %s\n",
                         i, cudaGetErrorString(err));
            std::free(h_ring);
            cudaFree(d_ring);
            return 1;
        }
    }
    std::free(h_ring);

    cudaDeviceSynchronize();

    // Warm pass: walk N steps.
    warmup_kernel<<<1, 1>>>(d_ring, N);
    cudaDeviceSynchronize();

    // Measured pass: walk 4*N steps. The first N hit-load every
    // line into L2; the remaining 3N are the population we measure.
    // NCU captures the AGGREGATE hit/total over the measured kernel,
    // so we report the steady-state hit rate the population reaches.
    pointer_chase_kernel<<<1, 1>>>(d_ring, 4 * N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_ring);
        return 1;
    }

    std::printf("probe ok: N=%d stride=%zu buf=%zu\n",
                N, stride, buf_bytes);
    cudaFree(d_ring);
    return 0;
}
