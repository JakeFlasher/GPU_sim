// slug: softmax_safe_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 numerically-stable (safe) softmax using a TWO-PASS ONLINE
// NORMALIZATION shape:
//
//   Pass 1: per-thread strided online (m, z) accumulation, then a
//           tree-reduce in shared memory using the online combining
//           rule:
//
//             m_new = max(m_a, m_b)
//             z_new = z_a * exp(m_a - m_new) + z_b * exp(m_b - m_new)
//
//           (the per-element step uses the same combiner with one
//           operand a single (x, 1) pair.)
//
//   Pass 2: each thread writes BF16 out[c] = exp(x[c] - row_m) / row_z
//           for its strided column set.
//
// Compared to a 3-pass (max-reduce, exp+sum-reduce, normalize) shape,
// the 2-pass online form makes a single streaming pass over the row
// instead of two, which is the canonical FlashAttention-style
// numerically-stable accumulation pattern. Both reductions still use
// a shared-memory tree with __syncthreads() between halving strides,
// so the SASS regression material the suite produces is unchanged
// (BAR.SYNC + the divergent if(tid<stride) BSSY/BSYNC pair plus
// CALL/RET from expf).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

namespace {
// Online combiner for (m, z) softmax-running-statistics pairs.
// Logically:
//   m_new = max(m_a, m_b)
//   z_new = z_a * exp(m_a - m_new) + z_b * exp(m_b - m_new)
// Defined as __device__ to keep it inlined into the reduction loop.
__device__ inline void combine_mz(float &m, float &z, float m_other, float z_other) {
    float m_new = (m > m_other) ? m : m_other;
    z = z * expf(m - m_new) + z_other * expf(m_other - m_new);
    m = m_new;
}
} // namespace

extern "C" __global__ void softmax_safe_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ x,    // [rows, cols]
    __nv_bfloat16       *__restrict__ out,  // [rows, cols]
    int32_t cols)
{
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lanes = blockDim.x;

    // shared[2*tid + 0] = m,  shared[2*tid + 1] = z.
    extern __shared__ float shared[];

    // ---- Pass 1: streaming online accumulation over the row ----
    float m = -FLT_MAX;
    float z = 0.0f;
    for (int c = tid; c < cols; c += lanes) {
        float x_c = __bfloat162float(x[row * cols + c]);
        // Single-element online update: combine current (m, z) with
        // the (x_c, 1) pair representing one fresh sample.
        combine_mz(m, z, x_c, 1.0f);
    }
    shared[2 * tid + 0] = m;
    shared[2 * tid + 1] = z;
    __syncthreads();

    // Tree reduce per-thread (m, z) pairs using the same combiner.
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m_other = shared[2 * (tid + stride) + 0];
            float z_other = shared[2 * (tid + stride) + 1];
            float m_self  = shared[2 * tid + 0];
            float z_self  = shared[2 * tid + 1];
            combine_mz(m_self, z_self, m_other, z_other);
            shared[2 * tid + 0] = m_self;
            shared[2 * tid + 1] = z_self;
        }
        __syncthreads();
    }
    const float row_m = shared[0];
    const float row_z = shared[1];
    __syncthreads();

    const float inv_z = 1.0f / row_z;

    // ---- Pass 2: write exp(x - row_m) / row_z into BF16 output ----
    for (int c = tid; c < cols; c += lanes) {
        float xv = __bfloat162float(x[row * cols + c]);
        float yv = expf(xv - row_m) * inv_z;
        out[row * cols + c] = __float2bfloat16(yv);
    }
}

extern "C" void softmax_safe_bf16_sm80_launch(
    const __nv_bfloat16 *d_x,
    __nv_bfloat16 *d_out,
    int rows, int cols, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem = 2 * block.x * sizeof(float);
    softmax_safe_bf16_sm80_kernel<<<grid, block, shmem, stream>>>(d_x, d_out, cols);
}
