// slug: dsmem_ring_bandwidth
// tier: P
// arch_gate: sm_90a sm_120
//
// DSMEM ring bandwidth: cluster of `cluster_size` CTAs, each writes its
// own seeded tile to its right neighbor's accumulator via
// `cluster.map_shared_rank`. After cluster.sync, each CTA emits its
// post-write accumulator to global, producing the cyclically-shifted
// sequence. Reference: references/cluster_dsmem.py::dsmem_ring_bandwidth.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

extern "C" __global__ void dsmem_ring_bandwidth_kernel(
    const float *__restrict__ in,        // [cluster_size, elements]
    float *__restrict__ out,              // [cluster_size, elements]
    int32_t elements)
{
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    const int cluster_size = cluster.dim_blocks().x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;

    extern __shared__ float accum[];

    // Initialize accumulator to zero on every CTA.
    for (int i = tid; i < elements; i += lanes) accum[i] = 0.0f;
    __syncthreads();
    cluster.sync();

    // Each rank writes ITS seeded tile to RIGHT NEIGHBOR's accumulator.
    int right = (rank + 1) % cluster_size;
    float *peer_accum = cluster.map_shared_rank(accum, right);
    const float *my_tile = in + (size_t)rank * elements;
    for (int i = tid; i < elements; i += lanes) {
        peer_accum[i] = my_tile[i];
    }
    cluster.sync();

    // Each CTA publishes its accumulator to global at its own rank index.
    float *my_out = out + (size_t)rank * elements;
    for (int i = tid; i < elements; i += lanes) my_out[i] = accum[i];
}

extern "C" cudaError_t dsmem_ring_bandwidth_launch(
    const float *d_in, float *d_out,
    int cluster_size, int elements, cudaStream_t stream)
{
    int threads = 32;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cluster_size, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = (size_t)elements * sizeof(float);
    cfg.stream = stream;

    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_size;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    return cudaLaunchKernelEx(&cfg, dsmem_ring_bandwidth_kernel, d_in, d_out, elements);
}
