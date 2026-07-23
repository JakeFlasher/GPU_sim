// sm_120 L2 associativity probe v3 — massively-parallel pointer chase.
//
// v1 single-thread chase: instruction-fetch baseline dominated hit
// rate; ~70% flat across N.
// v2 single-thread steady-state: same problem; ~90% flat.
// v3 design: many independent warps chase private rings concurrently.
// Data reads now dominate L2 traffic. Hit rate is a clean signal of
// whether each warp's working set fits in L2.
//
// Each thread chases its own ring of N nodes. With T threads, total
// distinct cache lines touched = T * N (assuming no sharing). When
// T * N * line_size (~128 B) is well under L2 capacity (32 MiB),
// hit rate should saturate near 100% after warmup. When it exceeds
// L2 capacity, hit rate drops.
//
// To probe ASSOCIATIVITY rather than just capacity, run the probe
// with a small total working set (T*N ~ a few thousand lines) but
// place all nodes at strides that the slice-mapping hash might map
// to the same set. We don't know the hash, so sweep many strides.
//
// Build:
//   nvcc -arch=sm_120 -O2 -o probe_v3 probe_v3.cu
//
// Run:
//   ./probe_v3 <ring_size_N> <stride_bytes>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Each node fits in one 128 B cache line.
struct alignas(128) Node {
    std::uint64_t next_idx;
    std::uint64_t pad[15];
};

// Each warp chases its own private ring. The chase is dependency-
// chained (each load depends on the previous result) so the
// hardware prefetcher cannot mask misses, and the compiler cannot
// reorder.
//
// `rings` is a flat array of (T_warps * N) Nodes laid out so warp
// w's ring occupies elements [w*N, (w+1)*N).
__global__ void parallel_chase_kernel(
    const Node* __restrict__ rings,
    int N,
    int hops
) {
    int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    // Each warp has 32 lanes chasing in lockstep; only lane 0 does
    // useful work, the rest sit idle. This keeps the chase
    // dependency-chained and per-warp.
    if (lane != 0) return;
    const Node* my_ring = rings + warp_id * N;
    std::uint64_t off = 0;
    for (int i = 0; i < hops; ++i) {
        off = my_ring[off].next_idx;
    }
    // Force the compiler to retain the chase.
    if (off == 0xdeadbeefULL) {
        printf("terminator hit (warp %d)\n", warp_id);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "Usage: %s <N> <stride_bytes>\n"
            "  N = ring length per warp\n"
            "  stride_bytes = bytes between consecutive nodes\n"
            "                 in a warp's ring (>= 128, ptr2 of 2)\n",
            argv[0]);
        return 2;
    }
    int N = std::atoi(argv[1]);
    long stride = std::atol(argv[2]);
    if (N < 1 || N > 4096) {
        std::fprintf(stderr, "N out of range: %d\n", N);
        return 2;
    }
    if (stride < 128) {
        std::fprintf(stderr, "stride < 128: %ld\n", stride);
        return 2;
    }

    // T warps total. Use 32 SMs × 4 warps/SM = 128 warps as the
    // default load. Per the sm_120 / CC 12.0 card, the device has
    // ~24 SMs. 128 warps comfortably saturates L2 traffic without
    // blowing up the buffer size.
    const int T_warps = 128;
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (T_warps + warps_per_block - 1) / warps_per_block;

    // Allocate one Node-buffer per warp, with `stride` bytes
    // between adjacent nodes inside a warp's ring. Total per-warp
    // span = N * stride.
    const std::size_t per_warp_span = static_cast<std::size_t>(N) * stride;
    const std::size_t total_bytes = static_cast<std::size_t>(T_warps) * per_warp_span;

    if (total_bytes > 6ull * 1024ull * 1024ull * 1024ull) {
        std::fprintf(stderr,
            "buffer size %zu too large (>6 GiB)\n",
            total_bytes);
        return 2;
    }

    Node* d_rings = nullptr;
    cudaError_t err = cudaMalloc(&d_rings, total_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "cudaMalloc(%zu) failed: %s\n",
            total_bytes, cudaGetErrorString(err));
        return 1;
    }

    // For NCU's view we want a single flat array indexed by
    // `next_idx`; convert byte stride into element stride. Each
    // Node is 128 B, so element_stride = stride / 128.
    const std::uint64_t element_stride =
        static_cast<std::uint64_t>(stride) / sizeof(Node);

    // For each warp, build a ring of N nodes where node i points
    // to node ((i+1) % N). next_idx is an element index into the
    // warp's private ring (not bytes).
    for (int w = 0; w < T_warps; ++w) {
        for (int i = 0; i < N; ++i) {
            Node h{};
            h.next_idx = (static_cast<std::uint64_t>((i + 1) % N))
                         * element_stride;
            char* dst = reinterpret_cast<char*>(d_rings)
                      + static_cast<std::size_t>(w) * per_warp_span
                      + static_cast<std::size_t>(i) * stride;
            err = cudaMemcpy(dst, &h, sizeof(Node),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                    "memcpy warp %d node %d: %s\n",
                    w, i, cudaGetErrorString(err));
                cudaFree(d_rings);
                return 1;
            }
        }
    }

    // Each warp's `rings` base pointer is at d_rings +
    // w * per_warp_span. We pass `d_rings` as the single base
    // pointer and the kernel computes the offset.
    //
    // Wait — the kernel does `rings + warp_id * N`, but our
    // strided layout means warp w's nodes are at offsets
    // (w * per_warp_span / 128) elements from `rings`. We need to
    // pass that, not `N`.
    //
    // For simplicity, pass `per_warp_span / 128` as the per-warp
    // ELEMENT offset; refactor the kernel call below.
    const int per_warp_element_offset =
        static_cast<int>(per_warp_span / sizeof(Node));

    // Warmup: shorter loop to populate caches.
    cudaDeviceSynchronize();
    parallel_chase_kernel<<<blocks, threads_per_block>>>(
        d_rings, per_warp_element_offset, 2 * N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "warmup: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_rings);
        return 1;
    }

    // Steady-state: 500k hops per warp (* 128 warps = 64M loads).
    const int hops = 500000;
    parallel_chase_kernel<<<blocks, threads_per_block>>>(
        d_rings, per_warp_element_offset, hops);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "steady: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_rings);
        return 1;
    }

    std::printf(
        "probe_v3 ok: N=%d stride=%ld warps=%d hops=%d total=%zu\n",
        N, stride, T_warps, hops, total_bytes);
    cudaFree(d_rings);
    return 0;
}
