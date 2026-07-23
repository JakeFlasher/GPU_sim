// slug: dsmem_pairwise_latency
// tier: P
// arch_gate: sm_90a sm_120
//
// Pairwise DSMEM latency: cluster of 2 CTAs. CTA 0 stages a seeded
// (tiles, elements) FP32 tile in shared memory. CTA 1 obtains a peer
// pointer into CTA 0's shared accumulator via `cluster.map_shared_rank`
// and reads the bytes into global out. Cubin contains MAPA / cluster
// barrier (UCGABAR_*) markers per BL-20260424-kernel-contract-vs-shape-match.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

extern "C" __global__ void dsmem_pairwise_latency_kernel(
    const float *__restrict__ in,
    float *__restrict__ out,
    int32_t tiles, int32_t elements)
{
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;

    extern __shared__ float tile_smem[];
    const int total = tiles * elements;

    if (rank == 0) {
        for (int i = tid; i < total; i += lanes) tile_smem[i] = in[i];
    }
    cluster.sync();

    if (rank == 1) {
        // Map peer (rank 0)'s shared array into our address space.
        float *peer_tile = cluster.map_shared_rank(tile_smem, 0);
        for (int i = tid; i < total; i += lanes) {
            out[i] = peer_tile[i];
        }
    }
    cluster.sync();
}

extern "C" cudaError_t dsmem_pairwise_latency_launch(
    const float *d_in, float *d_out,
    int tiles, int elements, cudaStream_t stream)
{
    int threads = 32;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = (size_t)tiles * elements * sizeof(float);
    cfg.stream = stream;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    return cudaLaunchKernelEx(&cfg, dsmem_pairwise_latency_kernel, d_in, d_out, tiles, elements);
}
