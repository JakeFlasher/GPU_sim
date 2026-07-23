// slug: cluster_remote_reduction
// tier: P
// arch_gate: sm_90a sm_120
//
// Cluster-launched remote reduction retained from the historical cluster
// microbench lineage. Each block in
// the cluster owns its own seeded tile (seed + cluster_rank) and computes
// a per-element local partial sum. Every block then uses
// `cluster.map_shared_rank(accum, 0)` to obtain a pointer into the
// cluster-leader (block-rank 0)'s shared accumulator and `atomicAdd`s its
// partial into that peer DSMEM slot. After `cluster.sync()` the leader
// publishes the accumulator to global memory.
//
// This matches the reference at
// `references/cluster_dsmem.py::cluster_remote_reduction`:
//     for c in cluster_size:  tile_c = seeded((tiles, elements), seed + c)
//     out[e] = sum_c sum_t tile_c[t][e]
//
// Requires host-side cooperative launch via `cudaLaunchKernelEx` with
// `cudaLaunchAttributeClusterDimension = cluster_size`.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

// Peak expected cluster footprint on the correctness shape: 64 elements.
// The shared accumulator is sized to this upper bound so block-rank 0's
// shared memory always has room for the full per-element sum regardless
// of which `elements` value the host driver passes (as long as it stays
// under kMaxElements). A later refactor can make this dynamic-smem if a
// shape > kMaxElements needs to be covered.
constexpr int kMaxElements = 1024;

extern "C" __global__ void cluster_remote_reduction_kernel(
    const float *__restrict__ tiles,   // [cluster_size, tiles, elements]
    float *__restrict__ out,            // [elements]
    int32_t n_tiles,
    int32_t elements)
{
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();

    __shared__ float accum[kMaxElements];

    // Only block-rank 0 initializes the shared accumulator to zero. The
    // cluster.sync() below makes that zeroing visible to every peer before
    // any peer starts atomicAdd-ing into it. (Equivalent to the legacy
    // pre-reduction barrier at dsmem_remote_reduction.cu:48.)
    if (rank == 0) {
        for (int e = threadIdx.x; e < elements; e += blockDim.x) {
            accum[e] = 0.0f;
        }
    }
    cluster.sync();

    // Each block in the cluster computes its per-element local partial
    // over its (tiles × elements) seeded tile, then pushes the partial
    // into block-rank-0's shared accumulator via DSMEM peer atomics.
    const float *my_tile = tiles + ((size_t)rank * n_tiles * elements);
    float *peer_accum = cluster.map_shared_rank(accum, 0);
    for (int e = threadIdx.x; e < elements; e += blockDim.x) {
        float partial = 0.0f;
        for (int t = 0; t < n_tiles; ++t) {
            partial += my_tile[t * elements + e];
        }
        atomicAdd(&peer_accum[e], partial);
    }

    // Trailing barrier: no block may exit until every peer atomicAdd has
    // landed. Without this the leader could publish a partial sum.
    cluster.sync();

    if (rank == 0) {
        for (int e = threadIdx.x; e < elements; e += blockDim.x) {
            out[e] = accum[e];
        }
    }
}

extern "C" cudaError_t cluster_remote_reduction_launch(
    const float *d_tiles,
    float *d_out,
    int cluster_size, int n_tiles, int elements, cudaStream_t stream)
{
    // Opt in to nonportable cluster sizes, matching the legacy pattern.
    // For the correctness shape (cluster_size=2) this is harmless, but
    // timing shapes with cluster_size > 8 require it.
    cudaError_t np_rc = cudaFuncSetAttribute(
        reinterpret_cast<const void *>(cluster_remote_reduction_kernel),
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (np_rc != cudaSuccess) {
        // Non-fatal on portable sizes; return the error so the host
        // driver can report it if needed, but keep going so the launch
        // below still fires for the correctness case.
    }

    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = static_cast<unsigned int>(cluster_size);
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(static_cast<unsigned int>(cluster_size), 1, 1);
    cfg.blockDim = dim3(static_cast<unsigned int>(elements), 1, 1);
    cfg.stream   = stream;
    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    return cudaLaunchKernelEx(&cfg, cluster_remote_reduction_kernel,
                              d_tiles, d_out, n_tiles, elements);
}
