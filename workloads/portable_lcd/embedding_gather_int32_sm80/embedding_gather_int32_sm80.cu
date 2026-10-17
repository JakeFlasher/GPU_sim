// slug: embedding_gather_int32_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 embedding lookup with INT32 indices, staged through
// shared memory via cp.async:
//
//   out[b, s, d] = table[indices[b, s], d]
//
// Each CTA owns a chunk of `CHUNK` consecutive `(b, s)` index
// positions and the full D dimension. The CTA's threads first
// prefetch the chunk's INT32 indices into shared memory via
// `cp.async.ca.shared.global` (which lowers to `LDGSTS` SASS on
// sm_80+), then `__syncthreads()`, then perform the
// gather/store loop reading staged indices from shared memory.
// The structure is still atomic-free — each output element has a
// unique destination — but the SASS now exercises the cp.async
// staged-load primitive in addition to plain `LDG`/`STG`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int CHUNK = 16;

__device__ inline uint32_t cp_async_smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
} // namespace

extern "C" __global__ void embedding_gather_int32_sm80_kernel(
    const int32_t       *__restrict__ indices, // [B*S]
    const __nv_bfloat16 *__restrict__ table,   // [V, D]
    __nv_bfloat16       *__restrict__ out,     // [B*S, D]
    int32_t n_pairs,
    int32_t D)
{
    const int chunk_id    = blockIdx.x;
    const int chunk_start = chunk_id * CHUNK;
    const int tid         = threadIdx.x;
    const int lanes       = blockDim.x;

    __shared__ int32_t s_indices[CHUNK];

    // Prefetch the chunk's indices into shared memory via cp.async.
    // Lanes 0..CHUNK-1 each issue one 4-byte cp.async; remaining
    // lanes idle for this stage.
    if (tid < CHUNK) {
        int gpos = chunk_start + tid;
        if (gpos < n_pairs) {
            uint32_t s_addr = cp_async_smem_addr(&s_indices[tid]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(s_addr), "l"(indices + gpos));
        } else {
            s_indices[tid] = 0;
        }
    }
    asm volatile("cp.async.commit_group;\n" :: );
    asm volatile("cp.async.wait_all;\n"     :: );
    __syncthreads();

    const int chunk_outputs = CHUNK * D;
    for (int local = tid; local < chunk_outputs; local += lanes) {
        int local_pair = local / D;
        int d          = local - local_pair * D;
        int gpos       = chunk_start + local_pair;
        if (gpos >= n_pairs) continue;
        int32_t idx = s_indices[local_pair];
        out[(int64_t)gpos * D + d] = table[(int64_t)idx * D + d];
    }
}

extern "C" void embedding_gather_int32_sm80_launch(
    const int32_t       *d_indices,
    const __nv_bfloat16 *d_table,
    __nv_bfloat16       *d_out,
    int B, int S, int D,
    cudaStream_t stream)
{
    int n_pairs = B * S;
    int grid    = (n_pairs + CHUNK - 1) / CHUNK;
    int block   = 256;
    embedding_gather_int32_sm80_kernel<<<grid, block, 0, stream>>>(
        d_indices, d_table, d_out, n_pairs, D);
}
