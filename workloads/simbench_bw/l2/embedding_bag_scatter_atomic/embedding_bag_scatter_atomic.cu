// slug: embedding_bag_scatter_atomic
// tier: P
// arch_gate: sm_90a sm_120
//
// Bag-sum embedding: out[b][d] = sum_{i in bag_size} E[indices[b][i]][d].
// Uses atomicAdd per output dim so the kernel's reduction order is
// non-deterministic (row=ND); tolerance absorbs the ordering diff.
// Matches references/embedding.py::_bag_scatter_reference.
//
// Grid: one block per (batch, bag-slot). Threads cover hidden dim and
// atomic-add into out[b][d]. For a deterministic reference, the host
// would do a per-batch scan; the kernel uses atomic add to maximize
// concurrency across bag slots.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void embedding_bag_scatter_atomic_kernel(
    const __nv_bfloat16 *__restrict__ table,   // [vocab_size, hidden]
    const int32_t       *__restrict__ indices, // [batch, bag_size]
    float               *__restrict__ out_fp32, // [batch, hidden] FP32 for atomicAdd
    int32_t batch,
    int32_t bag_size,
    int32_t hidden)
{
    const int b = blockIdx.x;
    const int i = blockIdx.y;
    const int d = threadIdx.x;
    if (b >= batch || i >= bag_size) return;
    const int row = indices[b * bag_size + i];
    for (int dd = d; dd < hidden; dd += blockDim.x) {
        float v = __bfloat162float(table[row * hidden + dd]);
        atomicAdd(out_fp32 + b * hidden + dd, v);
    }
}

extern "C" __global__ void embedding_bag_scatter_atomic_finalize_kernel(
    const float         *__restrict__ acc,   // [batch, hidden] FP32
    __nv_bfloat16       *__restrict__ out,   // [batch, hidden] BF16
    int32_t batch,
    int32_t hidden)
{
    const int b = blockIdx.x;
    const int d = threadIdx.x;
    if (b >= batch) return;
    for (int dd = d; dd < hidden; dd += blockDim.x) {
        out[b * hidden + dd] = __float2bfloat16(acc[b * hidden + dd]);
    }
}

extern "C" void embedding_bag_scatter_atomic_launch(
    const __nv_bfloat16 *d_table,
    const int32_t       *d_indices,
    float               *d_acc_fp32,
    __nv_bfloat16       *d_out,
    int batch, int bag_size, int hidden, cudaStream_t stream)
{
    cudaMemsetAsync(d_acc_fp32, 0, (size_t)batch * hidden * sizeof(float), stream);
    dim3 grid_s(batch, bag_size);
    dim3 block(128);
    embedding_bag_scatter_atomic_kernel<<<grid_s, block, 0, stream>>>(
        d_table, d_indices, d_acc_fp32, batch, bag_size, hidden);
    dim3 grid_f(batch);
    embedding_bag_scatter_atomic_finalize_kernel<<<grid_f, block, 0, stream>>>(
        d_acc_fp32, d_out, batch, hidden);
}
