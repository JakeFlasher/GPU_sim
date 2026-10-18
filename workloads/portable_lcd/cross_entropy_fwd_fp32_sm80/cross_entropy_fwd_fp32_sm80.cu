// slug: cross_entropy_fwd_fp32_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// FP32 per-sample cross-entropy from logits + INT32 labels:
//
//   m   = max(logits[b])
//   z   = sum(exp(logits[b][v] - m))   v in [0, V)
//   lse = m + log(z)
//   loss[b] = lse - logits[b][labels[b]]
//
// One CTA per row, shared-memory tree reductions for both the
// max-reduce and sum-of-exp-reduce passes. The divergent
// `if (tid < stride)` reduce loops contribute BSSY/BSYNC SASS
// (already covered by rmsnorm/softmax) and the per-row label
// lookup adds an integer-indexed scalar load distinct from those
// kernels' patterns.

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {
// `__noinline__` forces nvcc to keep this as an out-of-line device
// function so the kernel emits a CALL/RET pair to invoke it. The
// helper itself just computes log-sum-exp from the per-row max and
// running sum — small enough that nvcc would otherwise inline it
// and only an `EXIT` would appear at the end of the kernel.
__device__ __noinline__ float compute_log_sum_exp(float row_max, float row_z) {
    return row_max + logf(row_z);
}
} // namespace

extern "C" __global__ void cross_entropy_fwd_fp32_sm80_kernel(
    const float   *__restrict__ logits, // [B, V]
    const int32_t *__restrict__ labels, // [B]
    float         *__restrict__ loss,   // [B]
    int32_t V)
{
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lanes = blockDim.x;

    extern __shared__ float shared[];

    // Pass 1: max-reduce over logits[row]
    float local_max = -FLT_MAX;
    for (int c = tid; c < V; c += lanes) {
        float xv = logits[row * V + c];
        if (xv > local_max) local_max = xv;
    }
    shared[tid] = local_max;
    __syncthreads();
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = shared[tid];
            float b = shared[tid + stride];
            shared[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    // Pass 2: sum-reduce of exp(x - max)
    float local_sum = 0.0f;
    for (int c = tid; c < V; c += lanes) {
        local_sum += expf(logits[row * V + c] - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float row_z = shared[0];
    __syncthreads();

    // Lane 0 writes the loss for this row, calling the noinline
    // log-sum-exp helper so nvcc emits a CALL/RET pair.
    if (tid == 0) {
        int32_t lbl = labels[row];
        float lse  = compute_log_sum_exp(row_max, row_z);
        loss[row] = lse - logits[row * V + lbl];
    }
}

extern "C" void cross_entropy_fwd_fp32_sm80_launch(
    const float   *d_logits,
    const int32_t *d_labels,
    float         *d_loss,
    int B, int V, cudaStream_t stream)
{
    dim3 grid(B);
    dim3 block(256);
    size_t shmem = block.x * sizeof(float);
    cross_entropy_fwd_fp32_sm80_kernel<<<grid, block, shmem, stream>>>(
        d_logits, d_labels, d_loss, V);
}
