// slug: flux_adaln_zero_mod_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Flux AdaLN-Zero modulation block.
// Matches references/diffusion.py::_adaln_zero.
//
// 2-kernel pipeline:
//   K1: compute scale[b, h], shift[b, h], gate[b, h] via three linear
//       projections of cond_vec[b, :] through W_{scale,shift,gate}.
//   K2: per (b, s): out[b, s] = x[b, s] + gate[b] *
//                  (LayerNorm(x[b, s]) * (1 + scale[b]) + shift[b])

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define ADALN_EPS 1e-5f

// K1: condition projection. grid (B, hidden), threads do 3 dot products.
extern "C" __global__ void flux_adaln_cond_proj_kernel(
    const __nv_bfloat16 *__restrict__ cond,
    const __nv_bfloat16 *__restrict__ W_scale,
    const __nv_bfloat16 *__restrict__ W_shift,
    const __nv_bfloat16 *__restrict__ W_gate,
    float *__restrict__ scale,
    float *__restrict__ shift,
    float *__restrict__ gate,
    int32_t hidden, int32_t Cd)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    if (tid != 0) return;
    float acc_s = 0.0f, acc_h = 0.0f, acc_g = 0.0f;
    for (int c = 0; c < Cd; ++c) {
        float cv = __bfloat162float(cond[b * Cd + c]);
        acc_s += cv * __bfloat162float(W_scale[c * hidden + h]);
        acc_h += cv * __bfloat162float(W_shift[c * hidden + h]);
        acc_g += cv * __bfloat162float(W_gate [c * hidden + h]);
    }
    scale[b * hidden + h] = acc_s;
    shift[b * hidden + h] = acc_h;
    gate [b * hidden + h] = acc_g;
}

// K2: per-token LayerNorm + FiLM modulation + gated residual.
extern "C" __global__ void flux_adaln_apply_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ scale,
    const float *__restrict__ shift,
    const float *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int32_t S, int32_t hidden)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float *s_sum = smem;
    float *s_ss  = smem + lanes;
    const size_t row_base = ((size_t)b * S + s) * hidden;

    float m = 0.0f;
    for (int d = tid; d < hidden; d += lanes) m += __bfloat162float(x[row_base + d]);
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)hidden;
    float v = 0.0f;
    for (int d = tid; d < hidden; d += lanes) {
        float t = __bfloat162float(x[row_base + d]) - mean;
        v += t * t;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_ss[0] / (float)hidden + ADALN_EPS);
    const size_t cond_base = (size_t)b * hidden;
    for (int d = tid; d < hidden; d += lanes) {
        float xv = __bfloat162float(x[row_base + d]);
        float normed = (xv - mean) * inv_std;
        float modulated = normed * (1.0f + scale[cond_base + d]) + shift[cond_base + d];
        float outv = xv + gate[cond_base + d] * modulated;
        out[row_base + d] = __float2bfloat16(outv);
    }
}

extern "C" void flux_adaln_zero_mod_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_cond,
    const __nv_bfloat16 *d_W_scale,
    const __nv_bfloat16 *d_W_shift,
    const __nv_bfloat16 *d_W_gate,
    float *d_scale, float *d_shift, float *d_gate,
    __nv_bfloat16 *d_out,
    int B, int S, int hidden, int Cd,
    cudaStream_t stream)
{
    {
        dim3 grid_bh(B, hidden);
        dim3 block_1(1);
        flux_adaln_cond_proj_kernel<<<grid_bh, block_1, 0, stream>>>(
            d_cond, d_W_scale, d_W_shift, d_W_gate, d_scale, d_shift, d_gate, hidden, Cd);
    }
    int threads = 256;
    dim3 grid_bs(B, S);
    dim3 block_t(threads);
    size_t shmem = 2 * (size_t)threads * sizeof(float);
    flux_adaln_apply_kernel<<<grid_bs, block_t, shmem, stream>>>(
        d_x, d_scale, d_shift, d_gate, d_out, S, hidden);
}
