// slug: rmsnorm_bwd_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 RMSNorm backward (input gradient dx) for the forward
//
//   y[r][c] = (x[r][c] / sqrt(mean(x[r]^2) + eps)) * gamma[c]
//
// Given upstream grad_y, the input gradient is
//
//   dx[r][c] = grad_y[r][c] * gamma[c] * inv_rms[r]
//            - x[r][c] * S[r] * inv_rms[r]^3 / H
//
// where
//
//   ms[r]      = mean(x[r]^2)
//   inv_rms[r] = 1 / sqrt(ms[r] + eps)
//   S[r]       = sum_c (grad_y[r][c] * gamma[c] * x[r][c])
//   H          = hidden
//
// One CTA per row. Two shared-memory tree reductions:
//   (a) sum of x^2 to recover inv_rms (variance reuse — same
//       shape as the forward kernel's reduction so the verifier
//       sees the same BAR.SYNC discipline);
//   (b) sum of grad_y * gamma * x for the S-term.
// `__syncthreads()` between strides lowers to BAR.SYNC on every
// LCD-target arch.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void rmsnorm_bwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ x,      // [rows, hidden]
    const __nv_bfloat16 *__restrict__ gamma,  // [hidden]
    const __nv_bfloat16 *__restrict__ grad_y, // [rows, hidden]
    __nv_bfloat16       *__restrict__ dx,     // [rows, hidden]
    int32_t hidden,
    float eps)
{
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lanes = blockDim.x;

    extern __shared__ float shared[];
    float *sum_x2 = shared;
    float *sum_S  = shared + lanes;

    float local_x2 = 0.0f;
    float local_S  = 0.0f;
    for (int c = tid; c < hidden; c += lanes) {
        float xv  = __bfloat162float(x[row * hidden + c]);
        float gv  = __bfloat162float(gamma[c]);
        float gyv = __bfloat162float(grad_y[row * hidden + c]);
        local_x2 += xv * xv;
        local_S  += gyv * gv * xv;
    }
    sum_x2[tid] = local_x2;
    sum_S[tid]  = local_S;
    __syncthreads();

    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_x2[tid] += sum_x2[tid + stride];
            sum_S[tid]  += sum_S[tid  + stride];
        }
        __syncthreads();
    }

    float ms      = sum_x2[0] / (float)hidden;
    float inv_rms = rsqrtf(ms + eps);
    float S       = sum_S[0];
    float coeff   = inv_rms * inv_rms * inv_rms / (float)hidden;

    for (int c = tid; c < hidden; c += lanes) {
        float xv  = __bfloat162float(x[row * hidden + c]);
        float gv  = __bfloat162float(gamma[c]);
        float gyv = __bfloat162float(grad_y[row * hidden + c]);
        float dxv = gyv * gv * inv_rms - xv * S * coeff;
        dx[row * hidden + c] = __float2bfloat16(dxv);
    }
}

extern "C" void rmsnorm_bwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_gamma,
    const __nv_bfloat16 *d_grad_y,
    __nv_bfloat16 *d_dx,
    int rows, int hidden, float eps, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem = 2 * block.x * sizeof(float);
    rmsnorm_bwd_bf16_sm80_kernel<<<grid, block, shmem, stream>>>(
        d_x, d_gamma, d_grad_y, d_dx, hidden, eps);
}
