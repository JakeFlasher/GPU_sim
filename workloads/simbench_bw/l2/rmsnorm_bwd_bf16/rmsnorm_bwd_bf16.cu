// slug: rmsnorm_bwd_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// RMSNorm BACKWARD: given x, gamma, and the upstream output gradient
// dO, compute dX[r][c] = inv_rms * gamma[c] * dO[c] - x[r][c] * s /
// (hidden * rms^3), where s = sum_c dO[r][c] * gamma[c] * x[r][c] and
// inv_rms = 1 / sqrt(mean(x[r]^2) + eps). Matches
// references/norm.py::_rmsnorm_bwd (R39 replaces the R38
// forward-alias with a true backward).
//
// Grid: one block per row. Two reductions: sum of x^2 (for inv_rms) and
// sum of dO*gamma*x (for the scalar s). Then per-element output.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void rmsnorm_bwd_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gamma,
    const __nv_bfloat16 *__restrict__ dO,
    __nv_bfloat16 *__restrict__ dX,
    int32_t hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float *s_ms = smem;
    float *s_s  = smem + lanes;

    float local_ms = 0.0f;
    float local_s  = 0.0f;
    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        float gv = __bfloat162float(gamma[c]);
        float dv = __bfloat162float(dO[row * hidden + c]);
        local_ms += xv * xv;
        local_s  += dv * gv * xv;
    }
    s_ms[tid] = local_ms;
    s_s [tid] = local_s;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) {
            s_ms[tid] += s_ms[tid + st];
            s_s [tid] += s_s [tid + st];
        }
        __syncthreads();
    }
    const float ms = s_ms[0] / (float)hidden;
    const float inv_rms = rsqrtf(ms + eps);
    const float s = s_s[0];
    const float factor = s * inv_rms * inv_rms * inv_rms / (float)hidden;

    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        float gv = __bfloat162float(gamma[c]);
        float dv = __bfloat162float(dO[row * hidden + c]);
        float dx = inv_rms * gv * dv - xv * factor;
        dX[row * hidden + c] = __float2bfloat16(dx);
    }
}

extern "C" void rmsnorm_bwd_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_gamma, const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dX,
    int rows, int hidden, float eps, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem = 2 * block.x * sizeof(float);
    rmsnorm_bwd_bf16_kernel<<<grid, block, shmem, stream>>>(d_x, d_gamma, d_dO, d_dX, hidden, eps);
}
