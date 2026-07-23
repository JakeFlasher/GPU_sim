// sm_120 L2 associativity probe v2 — data-only streaming variant.
//
// The v1 probe was dominated by instruction-fetch + compulsory misses;
// a single pointer-chase walk of N nodes has too few real data reads
// to give a clean L2 hit-rate signal.
//
// v2 design:
//   * Build a ring of N nodes at a fixed wide stride.
//   * Warm the ring once.
//   * The measured kernel walks the ring REPEATEDLY (millions of
//     hops) to drown out compulsory misses. Steady-state hit rate
//     converges to a clean value: ~100% when the ring fits L2,
//     ~0% when the ring overflows whatever set/slice the hash maps
//     it to.
//
// The hit-rate signal across N reveals two things:
//   * The CLIFF in N is a multiple of associativity (depending on
//     how the hash spreads the ring across sets/slices).
//   * The cliff location, divided by num_l2_slices=24, gives the
//     per-slice working set at which any one slice's associativity
//     is exceeded.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

struct alignas(128) Node {
    std::uint64_t next_offset;
    std::uint64_t pad[15];
};

__global__ void warmup_kernel(const Node* __restrict__ ring,
                              int steps) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t off = 0;
        for (int i = 0; i < steps; ++i) {
            off = ring[off / sizeof(Node)].next_offset;
        }
        if (off == 0xdeadbeefULL) {
            printf("warmup terminator\n");
        }
    }
}

__global__ void steady_state_kernel(const Node* __restrict__ ring,
                                     int hops) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        std::uint64_t off = 0;
        for (int i = 0; i < hops; ++i) {
            off = ring[off / sizeof(Node)].next_offset;
        }
        if (off == 0xdeadbeefULL) {
            printf("steady terminator\n");
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "Usage: %s <N>\n",
                     argv[0]);
        return 2;
    }
    int N = std::atoi(argv[1]);
    if (N < 1 || N > 8192) {
        std::fprintf(stderr, "N out of range: %d\n", N);
        return 2;
    }

    // Stride 2 MiB; ring spans N * 2 MiB.
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

    for (int i = 0; i < N; ++i) {
        Node h_node{};
        h_node.next_offset = static_cast<std::uint64_t>(
            static_cast<std::size_t>((i + 1) % N) * stride);
        char* dst = reinterpret_cast<char*>(d_ring)
                  + static_cast<std::size_t>(i) * stride;
        err = cudaMemcpy(dst, &h_node, sizeof(Node),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "memcpy %d: %s\n",
                         i, cudaGetErrorString(err));
            cudaFree(d_ring);
            return 1;
        }
    }

    cudaDeviceSynchronize();
    warmup_kernel<<<1, 1>>>(d_ring, 2 * N);
    cudaDeviceSynchronize();

    // Steady-state: walk for 4_000_000 hops. With one hop per
    // iteration the loop is ~4M data reads, which drowns out
    // instruction-fetch traffic.
    const int hops = 4000000;
    steady_state_kernel<<<1, 1>>>(d_ring, hops);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "steady: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_ring);
        return 1;
    }

    std::printf("probe_v2 ok: N=%d stride=%zu hops=%d\n",
                N, stride, hops);
    cudaFree(d_ring);
    return 0;
}
