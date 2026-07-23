// slug: rmsnorm_fwd_bf16
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// BF16 RMSNorm forward: out[r][c] = (x[r][c] / sqrt(mean(x[r]^2) + eps)) * gamma[c].
// Reduction is done in FP32 for numerical stability (plan §8d FP8/BF16
// tolerance band). One block per row; block reduces via shared memory.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void rmsnorm_fwd_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,     // [rows, hidden]
    const __nv_bfloat16 *__restrict__ gamma, // [hidden]
    __nv_bfloat16 *__restrict__ out,         // [rows, hidden]
    int32_t hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;

    extern __shared__ float shared_ms[];
    float local = 0.0f;
    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        local += xv * xv;
    }
    shared_ms[tid] = local;
    __syncthreads();
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_ms[tid] += shared_ms[tid + stride];
        }
        __syncthreads();
    }
    float ms = shared_ms[0] / (float)hidden;
    float inv_rms = rsqrtf(ms + eps);
    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        float gv = __bfloat162float(gamma[c]);
        out[row * hidden + c] = __float2bfloat16(xv * inv_rms * gv);
    }
}

// Thin host driver for the harness smoke test. The orchestrator will
// allocate input buffers from the roster's correctness_shape, seed them
// with the deterministic_seeded_tensor values mirrored from references/norm.py,
// and compare the kernel's output against the CPU reference under the
// (BF16, tree_sum, norm) tolerance triplet.
extern "C" void rmsnorm_fwd_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_gamma,
    __nv_bfloat16 *d_out,
    int rows, int hidden, float eps, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem = block.x * sizeof(float);
    rmsnorm_fwd_bf16_kernel<<<grid, block, shmem, stream>>>(d_x, d_gamma, d_out, hidden, eps);
}
