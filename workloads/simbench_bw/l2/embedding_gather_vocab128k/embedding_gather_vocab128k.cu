// slug: embedding_gather_vocab128k
// tier: P
// arch_gate: sm_90a sm_120
//
// Vocab-scale embedding gather: out[b][w][d] = E[indices[b][w]][d].
// Matches references/embedding.py::_gather_reference.
//
// Grid: one block per (batch, width). Each thread copies one dim.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void embedding_gather_vocab128k_kernel(
    const __nv_bfloat16 *__restrict__ table,  // [vocab_size, hidden]
    const int32_t       *__restrict__ indices, // [batch, width]
    __nv_bfloat16 *__restrict__ out,           // [batch, width, hidden]
    int32_t batch,
    int32_t width,
    int32_t hidden)
{
    const int b = blockIdx.x;
    const int w = blockIdx.y;
    const int d = threadIdx.x;
    if (b >= batch || w >= width) return;
    const int row = indices[b * width + w];
    for (int dd = d; dd < hidden; dd += blockDim.x) {
        out[(b * width + w) * hidden + dd] = table[row * hidden + dd];
    }
}

extern "C" void embedding_gather_vocab128k_launch(
    const __nv_bfloat16 *d_table,
    const int32_t       *d_indices,
    __nv_bfloat16 *d_out,
    int batch, int width, int hidden, cudaStream_t stream)
{
    dim3 grid(batch, width);
    dim3 block(256);
    embedding_gather_vocab128k_kernel<<<grid, block, 0, stream>>>(
        d_table, d_indices, d_out, batch, width, hidden);
}
