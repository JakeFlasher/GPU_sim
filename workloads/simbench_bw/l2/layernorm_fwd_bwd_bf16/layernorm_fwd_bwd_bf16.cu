// slug: layernorm_fwd_bwd_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// LayerNorm BACKWARD (emits dX; the "fwd+bwd" row name reflects that
// a production deployment would compute both halves; the benchmark
// scoring surface is the backward gradient because it dominates the
// compute). Given x, gamma, and upstream dO:
//   mean_x = sum(x)/n; var_x = sum((x-mean_x)^2)/n
//   inv_std = 1/sqrt(var_x + eps); x_hat = (x - mean_x) * inv_std
//   sum_dy_g      = sum(dO * gamma) / n
//   sum_dy_g_xhat = sum(dO * gamma * x_hat) / n
//   dX[c] = inv_std * (dO[c]*gamma[c] - sum_dy_g - x_hat[c]*sum_dy_g_xhat)
// Matches references/norm.py::_layernorm_bwd (R39 repair).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void layernorm_fwd_bwd_bf16_kernel(
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
    float *s_sum = smem;              // [lanes] -- sum for mean
    float *s_ss  = smem + lanes;      // [lanes] -- sum of squares deviation (after mean)
    float *s_dg  = smem + 2 * lanes;  // [lanes] -- sum dO * gamma
    float *s_dgh = smem + 3 * lanes;  // [lanes] -- sum dO * gamma * x_hat

    // Pass 1: mean.
    float m = 0.0f;
    for (int c = tid; c < hidden; c += lanes) m += __bfloat162float(x[row * hidden + c]);
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)hidden;

    // Pass 2: variance.
    float v = 0.0f;
    for (int c = tid; c < hidden; c += lanes) {
        float d = __bfloat162float(x[row * hidden + c]) - mean;
        v += d * d;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float var = s_ss[0] / (float)hidden;
    const float inv_std = rsqrtf(var + eps);

    // Pass 3: sum(dO * gamma), sum(dO * gamma * x_hat).
    float dg = 0.0f, dgh = 0.0f;
    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        float gv = __bfloat162float(gamma[c]);
        float dv = __bfloat162float(dO[row * hidden + c]);
        float xh = (xv - mean) * inv_std;
        float prod = dv * gv;
        dg  += prod;
        dgh += prod * xh;
    }
    s_dg[tid]  = dg;
    s_dgh[tid] = dgh;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) {
            s_dg[tid]  += s_dg[tid + st];
            s_dgh[tid] += s_dgh[tid + st];
        }
        __syncthreads();
    }
    const float sum_dy_g = s_dg[0] / (float)hidden;
    const float sum_dy_g_xhat = s_dgh[0] / (float)hidden;

    for (int c = tid; c < hidden; c += lanes) {
        float xv = __bfloat162float(x[row * hidden + c]);
        float gv = __bfloat162float(gamma[c]);
        float dv = __bfloat162float(dO[row * hidden + c]);
        float xh = (xv - mean) * inv_std;
        float dx = inv_std * (dv * gv - sum_dy_g - xh * sum_dy_g_xhat);
        dX[row * hidden + c] = __float2bfloat16(dx);
    }
}

extern "C" void layernorm_fwd_bwd_bf16_launch(
    const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_gamma, const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dX,
    int rows, int hidden, float eps, cudaStream_t stream)
{
    dim3 grid(rows);
    dim3 block(256);
    size_t shmem = 4 * block.x * sizeof(float);
    layernorm_fwd_bwd_bf16_kernel<<<grid, block, shmem, stream>>>(d_x, d_gamma, d_dO, d_dX, hidden, eps);
}
